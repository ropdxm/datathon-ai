"""FastAPI entry point: uvicorn webapp.api:app --workers 1.

The jury key is only accepted by /api/v1. Browser routes use a private random,
HttpOnly session and require an exact same-origin Origin for mutations.
"""

import asyncio
from contextlib import asynccontextmanager, suppress
import hmac
import json
from pathlib import Path
import tempfile
import threading
import unicodedata
import uuid
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from starlette.concurrency import run_in_threadpool

from .config import MIB, Settings
from .jobs import JobCapacityError, JobManager
from .schemas import AnalysisResult
from .previews import PreviewBusyError, PreviewStore, redact_result


COOKIE_NAME = "forte_ui_session"
STATIC_DIR = Path(__file__).resolve().parent / "static"
JURY_KEY = APIKeyHeader(name="X-API-Key", auto_error=False)


class UploadTooLarge(Exception):
    pass


class UploadBodyLimit:
    """Bound the raw multipart stream before it can fill Starlette's temp spool."""

    def __init__(self, app, max_upload_bytes):
        self.app = app
        self.limit = max_upload_bytes + 64 * 1024  # finite multipart header envelope

    async def __call__(self, scope, receive, send):
        upload = (scope["type"] == "http" and scope["method"] == "POST"
                  and scope["path"] in {"/api/v1/analyze", "/api/ui/analyze", "/api/ui/previews"})
        if not upload:
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Некорректный размер запроса."}, 400)(scope, receive, send)
        if content_length > self.limit:
            return await JSONResponse({"detail": "Видео превышает допустимый размер."}, 413)(scope, receive, send)
        consumed = 0

        async def bounded_receive():
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.limit:
                    raise UploadTooLarge()
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except UploadTooLarge:
            await JSONResponse({"detail": "Видео превышает допустимый размер."}, 413)(scope, receive, send)


def _safe_filename(value):
    name = unicodedata.normalize("NFC", (value or "").replace("\\", "/").rsplit("/", 1)[-1])
    name = "".join(character for character in name if not unicodedata.category(character).startswith("C"))
    # User-facing name only; storage always uses a random server-generated ID.
    return name[:200].strip()


def _origin(request):
    parsed = urlsplit(str(request.base_url))
    return f"{parsed.scheme}://{parsed.netloc}"


def _same_origin(request, *, required=False):
    origin = request.headers.get("origin")
    if ((required and origin is None) or (origin is not None and origin != _origin(request))
            or request.headers.get("sec-fetch-site") == "cross-site"):
        raise HTTPException(403, "Допускаются только запросы с этой страницы сервиса.")


def create_app(settings: Settings | None = None, *, analyzer=None, capabilities_fn=None):
    settings = settings or Settings.from_env()
    # Do not recover jobs during import: a spawned analyzer may import this module.
    jobs = None
    previews = None
    preview_slots = threading.BoundedSemaphore(2)

    @asynccontextmanager
    async def lifespan(application):
        nonlocal jobs, previews
        if jobs is None:
            jobs = JobManager(settings, analyzer=analyzer)
            application.state.jobs = jobs
        if previews is None:
            previews = PreviewStore(settings)
            application.state.previews = previews
        jobs.start()

        async def expire_previews():
            while True:
                await asyncio.sleep(min(60, max(1, settings.retention_seconds / 2)))
                await run_in_threadpool(previews.cleanup)

        cleanup_task = asyncio.create_task(expire_previews())
        try:
            yield
        finally:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task
            jobs.stop()

    application = FastAPI(title="Forte video analysis", version="1.0.0", lifespan=lifespan)
    application.state.settings = settings
    application.add_middleware(UploadBodyLimit, max_upload_bytes=settings.max_upload_bytes)

    @application.middleware("http")
    async def private_response_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def capabilities():
        configured = capabilities_fn
        if configured is None:
            from .analysis import capabilities as configured
        try:
            result = configured(bundle_path=settings.model_bundle, analyzer_callable=settings.analyzer_callable)
            return {"ready": result["ready"] is True,
                    "mode": "model" if result["ready"] is True else "unconfigured",
                    "model_name": result.get("model_name"),
                    "limitations": result.get("limitations", []),
                    "max_upload_mb": settings.max_upload_bytes / MIB}
        except Exception:
            return {"ready": False, "mode": "unconfigured", "model_name": None,
                    "limitations": ["Модель недоступна. Проверьте настройки сервиса."],
                    "max_upload_mb": settings.max_upload_bytes / MIB}

    def require_key(key: str | None = Security(JURY_KEY)):
        if not key or not settings.api_key or not hmac.compare_digest(key.encode(), settings.api_key.encode()):
            raise HTTPException(401, "Требуется действующий X-API-Key.")

    def session_owner(request: Request):
        _same_origin(request)
        owner = jobs.session_owner(request.cookies.get(COOKIE_NAME))
        if owner is None:
            raise HTTPException(401, "Откройте страницу сервиса для создания сеанса.")
        return owner

    async def upload(request: Request, owner: str):
        if not capabilities()["ready"]:
            raise HTTPException(503, "Обученная модель не настроена или недоступна. Анализ временно невозможен.")
        try:
            jobs.reserve_slot()
        except JobCapacityError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "5"}) from exc
        reservation_owned = True
        part_path = final_path = None
        try:
            async with request.form(max_files=1, max_fields=0, max_part_size=64 * 1024) as form:
                if len(form) != 1 or not isinstance(form.get("file"), UploadFile):
                    raise HTTPException(422, "Передайте один видеофайл в поле file.")
                file = form["file"]
                filename = _safe_filename(file.filename)
                suffix = Path(filename).suffix.lower()
                if suffix not in {".mov", ".mp4"}:
                    raise HTTPException(415, "Поддерживаются видео MOV и MP4.")
                job_id = uuid.uuid4().hex
                part_path = jobs.upload_dir / f"{job_id}.part"
                final_path = jobs.upload_dir / f"{job_id}{suffix}"
                total = 0
                with part_path.open("xb") as output:
                    part_path.chmod(0o600)
                    while chunk := await file.read(1024 * 1024):
                        total += len(chunk)
                        if total > settings.max_upload_bytes:
                            raise HTTPException(413, "Видео превышает допустимый размер.")
                        output.write(chunk)
                if total == 0:
                    raise HTTPException(422, "Видеофайл пуст.")
                part_path.replace(final_path)
                # submit consumes the reservation on both success and failure.
                reservation_owned = False
                jobs.submit(final_path, filename, owner, reserved=True, job_id=job_id)
                final_path = None
                return JSONResponse({"job_id": job_id, "status": "processing"}, status_code=202)
        finally:
            if reservation_owned:
                jobs.release_slot()
            if part_path is not None:
                part_path.unlink(missing_ok=True)
            if final_path is not None:
                final_path.unlink(missing_ok=True)

    @application.get("/healthz")
    def health():
        return {"status": "ok"}

    @application.post("/api/v1/analyze", dependencies=[Depends(require_key)], status_code=202)
    async def jury_analyze(request: Request):
        return await upload(request, "jury")

    @application.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    def jury_job(job_id: str):
        jobs.cleanup()
        result = jobs.get(job_id)
        if result is None:
            raise HTTPException(404, "Задание не найдено или срок хранения истёк.")
        try:
            return redact_result(result)
        except Exception:
            raise HTTPException(503, "Не удалось подготовить кадры с размытием лиц.") from None

    @application.post("/api/ui/previews")
    async def ui_create_preview(request: Request, owner: str = Depends(session_owner)):
        _same_origin(request, required=True)
        if not preview_slots.acquire(blocking=False):
            raise HTTPException(429, "Подготовка предпросмотра занята. Повторите позже.")
        try:
            staging = settings.data_dir / "preview-uploads"
            staging.mkdir(exist_ok=True, mode=0o700)
            staging.chmod(0o700)
            with tempfile.TemporaryDirectory(dir=staging) as temporary:
                async with request.form(max_files=1, max_fields=0, max_part_size=64 * 1024) as form:
                    if len(form) != 1 or not isinstance(form.get("file"), UploadFile):
                        raise HTTPException(422, "Передайте один видеофайл в поле file.")
                    file = form["file"]
                    suffix = Path(_safe_filename(file.filename)).suffix.lower()
                    if suffix not in {".mov", ".mp4"}:
                        raise HTTPException(415, "Поддерживаются видео MOV и MP4.")
                    source = Path(temporary) / ("source" + suffix)
                    total = 0
                    with source.open("xb") as output:
                        source.chmod(0o600)
                        while chunk := await file.read(1024 * 1024):
                            total += len(chunk)
                            if total > settings.max_upload_bytes:
                                raise HTTPException(413, "Видео превышает допустимый размер.")
                            output.write(chunk)
                    if not total:
                        raise HTTPException(422, "Видеофайл пуст.")
                    try:
                        return await run_in_threadpool(previews.create, source, owner)
                    except PreviewBusyError:
                        raise HTTPException(429, "Подготовка предпросмотра занята. Повторите позже.") from None
                    except Exception:
                        raise HTTPException(503, "Не удалось размыть лица. Исходное видео не будет показано.") from None
        finally:
            preview_slots.release()

    @application.get("/api/ui/previews/{preview_id}/video")
    def ui_preview_video(preview_id: str, owner: str = Depends(session_owner)):
        path = previews.video_path(preview_id, owner)
        if path is None:
            raise HTTPException(404, "Предпросмотр не найден или срок хранения истёк.")
        return FileResponse(path, media_type="video/mp4",
                            headers={"Cache-Control": "no-store", "Content-Disposition": "inline"})

    @application.get("/api/ui/config")
    def ui_config(request: Request, response: Response):
        _same_origin(request)
        token = request.cookies.get(COOKIE_NAME)
        if jobs.session_owner(token) is None:
            token = jobs.issue_session()
        response.set_cookie(COOKIE_NAME, token, max_age=int(settings.session_ttl_seconds),
                            httponly=True, secure=request.url.scheme == "https",
                            samesite="strict", path="/api/ui")
        response.headers["Cache-Control"] = "no-store"
        demo_dir = settings.data_dir / "demo"
        return {**capabilities(), "demo_available":
                (demo_dir / "result.json").is_file() and (demo_dir / "video.mp4").is_file()}

    def demo_files():
        directory = settings.data_dir / "demo"
        video, result = directory / "video.mp4", directory / "result.json"
        if not video.is_file() or not result.is_file():
            raise HTTPException(404, "Демонстрационный ролик ещё не подготовлен на этом сервере.")
        return video, result

    @application.get("/api/ui/demo")
    def ui_demo(owner: str = Depends(session_owner)):
        video_path, result_path = demo_files()
        try:
            if result_path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Oversized demo result")
            data = json.loads(result_path.read_text(encoding="utf8"))
            if data.get("demo") is not True or data.get("demo_source") != "provided_video":
                raise ValueError("Demo must identify its annotated source")
            analysis = AnalysisResult.model_validate(data).model_dump(exclude_none=True)
            filename = _safe_filename(data.get("filename")) or "Пример из датасета.mov"
        except (OSError, ValueError, TypeError, AttributeError):
            raise HTTPException(503, "Демо подготовлено некорректно. Пересоздайте его из исходного видео.") from None
        try:
            previews.get_or_create_for_job(video_path, "demo", "dataset-demo")
            analysis = redact_result(analysis)
        except PreviewBusyError:
            raise HTTPException(429, "Подготовка предпросмотра занята. Повторите позже.") from None
        except Exception:
            raise HTTPException(503, "Не удалось подготовить демо с размытием лиц.") from None
        return {**analysis, "job_id": "dataset-demo", "status": "done", "demo": True,
                "demo_source": "provided_video", "filename": filename,
                "model_name": "Исходная разметка", "elapsed_sec": 0,
                "video_url": "/api/ui/demo/video"}

    @application.get("/api/ui/demo/video")
    def ui_demo_video(owner: str = Depends(session_owner)):
        video, _ = demo_files()
        try:
            video = previews.get_or_create_for_job(video, "demo", "dataset-demo")
        except PreviewBusyError:
            raise HTTPException(429, "Подготовка предпросмотра занята. Повторите позже.") from None
        except Exception:
            raise HTTPException(503, "Не удалось подготовить демо с размытием лиц.") from None
        return FileResponse(video, media_type="video/mp4",
                            headers={"Cache-Control": "no-store", "Content-Disposition": "inline"})

    @application.post("/api/ui/analyze", status_code=202)
    async def ui_analyze(request: Request, owner: str = Depends(session_owner)):
        _same_origin(request, required=True)
        return await upload(request, owner)

    @application.get("/api/ui/jobs/{job_id}")
    def ui_job(job_id: str, owner: str = Depends(session_owner)):
        jobs.cleanup()
        result = jobs.get(job_id, owner)
        if result is None:
            raise HTTPException(404, "Задание не найдено или срок хранения истёк.")
        try:
            return redact_result(result)
        except Exception:
            raise HTTPException(503, "Не удалось подготовить кадры с размытием лиц.") from None

    @application.get("/api/ui/jobs/{job_id}/video")
    def ui_video(job_id: str, owner: str = Depends(session_owner)):
        jobs.cleanup()
        path = jobs.video_path(job_id, owner)
        if path is None:
            raise HTTPException(404, "Видео не найдено или срок хранения истёк.")
        try:
            path = previews.get_or_create_for_job(path, owner, job_id)
        except PreviewBusyError:
            raise HTTPException(429, "Подготовка предпросмотра занята. Повторите позже.") from None
        except Exception:
            raise HTTPException(503, "Не удалось размыть лица. Исходное видео не будет показано.") from None
        return FileResponse(path, media_type="video/mp4",
                            headers={"Cache-Control": "no-store", "Content-Disposition": "inline"})

    @application.get("/", include_in_schema=False)
    def index():
        path = STATIC_DIR / "index.html"
        if not path.is_file():
            raise HTTPException(503, "Интерфейс сервиса ещё не установлен.")
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    application.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")
    return application


app = create_app()

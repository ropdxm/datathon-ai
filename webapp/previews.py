"""Private, expiring video copies with local face redaction.

Original uploads never become preview responses. Model inference keeps its
original input; only display copies and exported evidence are redacted.
"""
from __future__ import annotations

import base64
import copy
import io
import json
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

from PIL import Image

from .privacy import blur_faces, redact_video_with_timeout

REDACTION_VERSION = "yunet-geometry-v2"


class PreviewBusyError(RuntimeError):
    """A new preview is rejected promptly while the native worker is occupied."""


class PreviewStore:
    def __init__(self, settings):
        self.root = settings.data_dir / "previews"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.retention = settings.retention_seconds
        self.max_retained = settings.max_retained_jobs
        self._lock = threading.RLock()
        self._render_gate = threading.Lock()
        self._active = set()
        # This service runs with one process. Startup can safely remove upload
        # temporaries left by an interrupted previous process.
        staging = settings.data_dir / "preview-uploads"
        if staging.is_dir() and not staging.is_symlink():
            for temporary in staging.iterdir():
                if temporary.is_symlink() or temporary.is_file():
                    temporary.unlink(missing_ok=True)
                elif temporary.is_dir():
                    shutil.rmtree(temporary, ignore_errors=True)
        self.cleanup()

    def _read(self, preview_id):
        if not isinstance(preview_id, str) or not re.fullmatch(r"[0-9a-f]{32}", preview_id):
            return None
        directory = self.root / preview_id
        if directory.is_symlink():
            return None
        try:
            data = json.loads((directory / "metadata.json").read_text(encoding="utf8"))
            if data.get("redaction_version") != REDACTION_VERSION:
                shutil.rmtree(directory, ignore_errors=True)
                return None
            if time.time() - data["created_at"] > self.retention:
                shutil.rmtree(directory, ignore_errors=True)
                return None
            return data if (directory / "video.mp4").is_file() else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _cleanup(self, reserve=0):
        complete = []
        for directory in self.root.iterdir():
            if directory.is_symlink() or not directory.is_dir() or directory.name in self._active:
                continue
            data = self._read(directory.name)
            if data:
                complete.append((data["created_at"], directory))
            elif directory.exists() and time.time() - directory.stat().st_mtime > self.retention:
                shutil.rmtree(directory, ignore_errors=True)
        for _, directory in sorted(complete)[:max(0, len(complete) - self.max_retained + reserve)]:
            shutil.rmtree(directory, ignore_errors=True)

    def cleanup(self):
        with self._lock:
            self._cleanup()

    def create(self, source: Path, owner: str, preview_id: str | None = None):
        preview_id = preview_id or uuid.uuid4().hex
        if not re.fullmatch(r"[0-9a-f]{32}", preview_id):
            raise ValueError("Invalid preview identifier")
        with self._lock:
            cached = self._read(preview_id)
            if cached and cached.get("owner") == owner:
                return self._public(preview_id, cached)
        # Cached reads never wait for the renderer. All entrypoints, including
        # demo/job GETs, share this nonblocking admission gate.
        if not self._render_gate.acquire(blocking=False):
            raise PreviewBusyError("Подготовка предпросмотра занята. Повторите позже.")
        directory = self.root / preview_id
        owns_directory = False
        try:
            with self._lock:
                cached = self._read(preview_id)
                if cached:
                    if cached.get("owner") != owner:
                        raise ValueError("Preview belongs to another session")
                    return self._public(preview_id, cached)
                self._cleanup(reserve=1)
                if directory.is_symlink():
                    raise ValueError("Invalid preview storage")
                if directory.exists():
                    shutil.rmtree(directory)
                directory.mkdir(mode=0o700)
                owns_directory = True
                self._active.add(preview_id)
            destination = directory / "video.mp4"
            details = redact_video_with_timeout(Path(source), destination, timeout_seconds=120)
            if not destination.is_file() or destination.stat().st_size == 0:
                raise ValueError("Redacted video is missing")
            destination.chmod(0o600)
            data = {"owner": owner, "created_at": time.time(), "redaction_version": REDACTION_VERSION,
                    "duration_sec": details["duration_sec"],
                    "privacy": {"method": "local_yunet", "frames_processed": details.get("frames_processed", 0),
                                "face_detections": details.get("face_detections", 0),
                                "limitation": "Автоматический детектор может пропускать лица."}}
            with self._lock:
                metadata = directory / "metadata.json"
                metadata.write_text(json.dumps(data, ensure_ascii=False), encoding="utf8")
                metadata.chmod(0o600)
            return self._public(preview_id, data)
        except BaseException:
            if owns_directory:
                shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            with self._lock:
                self._active.discard(preview_id)
            self._render_gate.release()

    @staticmethod
    def _public(preview_id, data):
        return {"preview_id": preview_id, "video_url": f"/api/ui/previews/{preview_id}/video",
                "duration_sec": data["duration_sec"], "privacy": data["privacy"]}

    def video_path(self, preview_id: str, owner: str):
        with self._lock:
            data = self._read(preview_id)
            if data is None or data.get("owner") != owner:
                return None
            return self.root / preview_id / "video.mp4"

    def get_or_create_for_job(self, source: Path, owner: str, job_id: str):
        source = Path(source)
        stat = source.stat()
        preview_id = uuid.uuid5(uuid.NAMESPACE_URL,
                              f"forte-preview:{REDACTION_VERSION}:{owner}:{job_id}:{stat.st_size}:{stat.st_mtime_ns}").hex
        result = self.create(source, owner, preview_id)
        return self.root / result["preview_id"] / "video.mp4"


def redact_result(result: dict) -> dict:
    """Also protect old stored results and demo JPEGs before they leave the API."""
    result = copy.deepcopy(result)
    for event in result.get("violations", []):
        for frame in event.get("evidence_frames", []):
            raw = base64.b64decode(frame["image_base64"], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                redacted = blur_faces(image.convert("RGB"))
            output = io.BytesIO()
            redacted.save(output, format="JPEG", quality=90)
            frame["image_base64"] = base64.b64encode(output.getvalue()).decode("ascii")
    return result

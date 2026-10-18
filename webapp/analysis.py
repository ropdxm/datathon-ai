"""Local inference boundary and original-video evidence for the web service.

Configuration is supplied by the backend. This module never reads environment
variables, calls a remote model, trains weights, or substitutes demo predictions.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping
import importlib
import inspect
import io
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

import av
from PIL import Image


MAX_EVENTS = 20
MAX_DESCRIPTION_LENGTH = 4000
MAX_SOURCE_PIXELS = 16_777_216
MAX_DECODED_FRAMES = 1_000_000
MAX_EVIDENCE_SIDE = 1280
# Base64 plus event metadata must remain within the backend's 16 MiB result cap.
MAX_EVIDENCE_BYTES = 10 * 1024 * 1024

_UNCONFIGURED = "Модель не настроена: укажите локальный FORTE_MODEL_BUNDLE или FORTE_ANALYZER."
_MISSING_RULE = "Нормативное основание для одного или нескольких фрагментов моделью не предоставлено."
_BASELINE_LIMITATIONS = [
    "Бинарная модель выделяет подозрительные окна; тип события она не классифицирует.",
    "Время соответствует приблизительному центру подозрительного окна, а не распознанному моменту нарушения.",
    "Нормативное основание бинарная модель не определяет; фрагменты требуют проверки человеком.",
    "Оценка модели не является калиброванной вероятностью нарушения.",
]


def _callable_parts(path: str) -> tuple[str, str]:
    """Validate configuration syntax without importing or executing user code."""
    if not isinstance(path, str) or path.count(":") != 1:
        raise ValueError("Ожидается путь к анализатору в формате package.module:function.")
    module_name, function_name = path.split(":")
    if not module_name or not function_name or not all(part.isidentifier() for part in module_name.split(".")) or not function_name.isidentifier():
        raise ValueError("Некорректный путь к анализатору package.module:function.")
    return module_name, function_name


def _load_callable(path: str):
    # Only analyze_video, running in the deadline-controlled worker, calls this.
    module_name, function_name = _callable_parts(path)
    function = getattr(importlib.import_module(module_name), function_name)
    call_method = getattr(function, "__call__", None)
    if (not callable(function) or inspect.isclass(function) or inspect.iscoroutinefunction(function)
            or inspect.isasyncgenfunction(function) or inspect.iscoroutinefunction(call_method)
            or inspect.isasyncgenfunction(call_method)):
        raise ValueError("Анализатор должен быть синхронной функцией, принимающей путь к видео.")
    return function


def _bundle_config(bundle_path: Path) -> dict:
    required = ("config.json", "head.pt", "encoder/encoder_config.json", "encoder/encoder_state.pt")
    if not all((bundle_path / name).is_file() for name in required):
        raise ValueError("Неполный пакет модели: нужны config.json, head.pt и локальные файлы encoder.")
    config = json.loads((bundle_path / "config.json").read_text(encoding="utf8"))
    encoder = json.loads((bundle_path / "encoder/encoder_config.json").read_text(encoding="utf8"))
    if not isinstance(config, dict) or not isinstance(encoder, dict):
        raise ValueError("Некорректная конфигурация пакета модели.")
    if config.get("model_id") not in {"dinov2_gru", "r3d_18", "r2plus1d_18", "swin3d_t"} or encoder.get("model_id") != config["model_id"]:
        raise ValueError("Неизвестная или несогласованная архитектура модели.")
    if encoder.get("bundle_format_version") != 1:
        raise ValueError("Неподдерживаемая версия пакета модели.")
    if config["model_id"] == "dinov2_gru" and not (bundle_path / "encoder/dinov2_source/hubconf.py").is_file():
        raise ValueError("В пакете DINOv2 отсутствуют локальные исходники encoder/dinov2_source.")
    return config


def capabilities(bundle_path: Path | None = None, analyzer_callable: str | None = None) -> dict:
    """Report configuration readiness without importing custom code or weights.

    An explicitly configured callable takes precedence over a bundle. Readiness
    means its entry-point syntax is valid or the bundle structure exists. It does
    not guarantee a successful import, model initialization, or inference. Those
    checks happen only in the spawned analysis worker under the job deadline.
    """
    try:
        if analyzer_callable:
            _callable_parts(analyzer_callable)
            return {"ready": True, "mode": "model", "model_name": analyzer_callable, "limitations": []}
        if bundle_path is not None:
            config = _bundle_config(Path(bundle_path))
            return {"ready": True, "mode": "model", "model_name": f"Colab: {config['model_id']}",
                    "limitations": list(_BASELINE_LIMITATIONS)}
    except Exception as error:
        return {"ready": False, "mode": "unconfigured", "model_name": None,
                "limitations": [f"Не удалось подключить локальную модель или анализатор: {type(error).__name__}: {error}"]}
    return {"ready": False, "mode": "unconfigured", "model_name": None, "limitations": [_UNCONFIGURED]}


def _number(value, field: str) -> float:
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{field}: требуется конечное число.")
    return float(value)


def _text(value, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{field}: требуется непустая строка до {limit} символов.")
    return value.strip()


def _frame_time(frame, stream) -> float:
    time_base = frame.time_base or stream.time_base
    if frame.pts is None or time_base is None:
        raise ValueError("У кадра видео отсутствует исходная метка времени PTS.")
    timestamp = float(frame.pts * time_base)
    if not math.isfinite(timestamp):
        raise ValueError("Видео содержит некорректные метки времени PTS.")
    return timestamp


def _check_dimensions(frame) -> None:
    if frame.width < 1 or frame.height < 1 or frame.width * frame.height > MAX_SOURCE_PIXELS:
        raise ValueError("Размер кадра видео превышает допустимый предел декодирования.")


def _video_duration(video_path: Path) -> float:
    """Validate the decodable stream and measure its PTS timeline using bounded memory."""
    try:
        with av.open(str(video_path)) as container:
            if not container.streams.video:
                raise ValueError("В файле отсутствует видеопоток.")
            stream = container.streams.video[0]
            origin = previous = last = None
            last_duration = None
            for count, frame in enumerate(container.decode(stream), start=1):
                if count > MAX_DECODED_FRAMES:
                    raise ValueError("Видео превышает допустимое число декодированных кадров.")
                _check_dimensions(frame)
                timestamp = _frame_time(frame, stream)
                if last is not None and timestamp < last:
                    raise ValueError("Нарушен порядок меток времени кадров видео.")
                origin = timestamp if origin is None else origin
                previous, last = last, timestamp
                duration_ticks = getattr(frame, "duration", 0)
                last_duration = float(duration_ticks * (frame.time_base or stream.time_base)) if duration_ticks else None
            if last is None:
                raise ValueError("Видео не содержит декодируемых кадров.")
            if last_duration is None or not math.isfinite(last_duration) or last_duration <= 0:
                if stream.duration is not None and stream.time_base is not None:
                    declared_duration = float(stream.duration * stream.time_base)
                    last_duration = declared_duration - (last - origin)
                if last_duration is None or not math.isfinite(last_duration) or last_duration <= 0:
                    last_duration = last - previous if previous is not None and last > previous else 1 / float(stream.average_rate or 30)
            duration = last - origin + last_duration
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Невозможно определить положительную длительность видео.")
            return duration
    except (av.error.FFmpegError, OSError, OverflowError) as error:
        raise ValueError("Не удалось декодировать видео. Загрузите исправный MOV-файл.") from error


def _rule(value) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("rule: ожидается объект с полями title, clause, url.")
    rule = {}
    for field, limit in (("title", 300), ("clause", 1000), ("url", 2048)):
        if value.get(field) is not None:
            rule[field] = _text(value[field], f"rule.{field}", limit)
    if "url" in rule:
        url = rule["url"]
        try:
            parsed = urlsplit(url)
            valid = (parsed.scheme.lower() in {"http", "https"} and parsed.hostname and not parsed.username
                     and not parsed.password and not any(character.isspace() or ord(character) < 32 for character in url))
            # Accessing port validates malformed/out-of-range port numbers as well.
            parsed.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("rule.url: разрешены только абсолютные HTTP/HTTPS-ссылки без учётных данных.")
    return rule or None


def _normalize(payload, duration: float) -> dict:
    if not isinstance(payload, Mapping):
        raise ValueError("Анализатор должен вернуть объект результата.")
    verdict = payload.get("verdict")
    if verdict not in ("violation", "no_violation"):
        raise ValueError("verdict: требуется violation или no_violation.")
    source_events = payload.get("violations")
    if not isinstance(source_events, list) or len(source_events) > MAX_EVENTS:
        raise ValueError(f"violations: требуется список не более {MAX_EVENTS} событий.")
    if (verdict == "no_violation" and source_events) or (verdict == "violation" and not source_events):
        raise ValueError("Вердикт не согласован со списком нарушений.")
    limitations = payload.get("limitations", [])
    if not isinstance(limitations, list) or len(limitations) > 20:
        raise ValueError("limitations: требуется список не более 20 строк.")
    limitations = [_text(item, "limitations", 1000) for item in limitations]
    events = []
    for raw in source_events:
        if not isinstance(raw, Mapping):
            raise ValueError("Событие должно быть объектом.")
        timestamp = _number(raw.get("timecode_sec"), "timecode_sec")
        if not 0 <= timestamp <= duration:
            raise ValueError("timecode_sec: время события выходит за длительность исходного видео.")
        event = {"type": _text(raw.get("type"), "type", 160), "timecode_sec": timestamp,
                 "description": _text(raw.get("description"), "description", MAX_DESCRIPTION_LENGTH)}
        rule = _rule(raw.get("rule"))
        event["rule"] = rule
        if rule:
            source = [rule["title"]] if "title" in rule else []
            if "clause" in rule:
                source.append(f"пункт {rule['clause']}")
            if "url" in rule:
                source.append(rule["url"])
            event["description"] += "\nНормативное основание (предоставлено моделью): " + "; ".join(source)
        elif _MISSING_RULE not in limitations:
            limitations.append(_MISSING_RULE)
        # Incoming evidence, probability fields, and unknown metadata are ignored.
        # Only frames decoded from this job's original source can enter results.
        events.append(event)
    return {"verdict": verdict, "violations": events, "limitations": list(dict.fromkeys(limitations)), "duration_sec": duration}


def _attach_evidence(video_path: Path, result: dict) -> None:
    duration = result["duration_sec"]
    targets_by_event = [sorted({min(duration, max(0.0, event["timecode_sec"] + offset)) for offset in (-0.5, 0.0, 0.5)})
                        for event in result["violations"]]
    targets = sorted({target for times in targets_by_event for target in times})
    if not targets:
        return
    selected = {}
    encoded_frames = {}
    total_bytes = 0

    def remember(target, frame, timestamp):
        nonlocal total_bytes
        if timestamp not in encoded_frames:
            _check_dimensions(frame)
            picture = frame.to_image().convert("RGB")
            picture.thumbnail((MAX_EVIDENCE_SIDE, MAX_EVIDENCE_SIDE), Image.Resampling.LANCZOS)
            from .privacy import blur_faces
            picture = blur_faces(picture)
            output = io.BytesIO()
            picture.save(output, format="JPEG", quality=85)
            jpeg = output.getvalue()
            total_bytes += len(jpeg)
            if total_bytes > MAX_EVIDENCE_BYTES:
                raise ValueError("Объём кадров-доказательств превышает допустимый предел.")
            encoded_frames[timestamp] = {"timestamp_sec": timestamp, "image_base64": base64.b64encode(jpeg).decode("ascii")}
        selected[target] = encoded_frames[timestamp]

    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            origin = previous_time = previous_frame = None
            target_index = 0
            for frame in container.decode(stream):
                absolute = _frame_time(frame, stream)
                origin = absolute if origin is None else origin
                timestamp = absolute - origin
                while target_index < len(targets) and targets[target_index] <= timestamp:
                    target = targets[target_index]
                    if previous_frame is not None and target - previous_time <= timestamp - target:
                        remember(target, previous_frame, previous_time)
                    else:
                        remember(target, frame, timestamp)
                    target_index += 1
                previous_frame, previous_time = frame, timestamp
                if target_index == len(targets):
                    break
            if previous_frame is None:
                raise ValueError("В видео нет кадров для формирования доказательств.")
            for target in targets[target_index:]:
                remember(target, previous_frame, previous_time)
    except (av.error.FFmpegError, OSError, OverflowError) as error:
        raise ValueError("Не удалось извлечь кадры-доказательства из исходного видео.") from error
    for event, times in zip(result["violations"], targets_by_event):
        unique = {selected[target]["timestamp_sec"]: selected[target] for target in times}
        event["evidence_frames"] = [unique[timestamp] for timestamp in sorted(unique)]


def analyze_video(video_path: Path, *, bundle_path: Path | None = None,
                  analyzer_callable: str | None = None, device: str = "auto") -> dict:
    """Analyze an uploaded video with a configured local callable or Colab bundle.

    A callable is invoked synchronously as ``function(video_path: Path)`` and must
    return the documented verdict/violations mapping. Exceptions propagate to the
    backend's failed job state. There is deliberately no fallback prediction.
    """
    info = capabilities(bundle_path=bundle_path, analyzer_callable=analyzer_callable)
    if not info["ready"]:
        raise RuntimeError(" ".join(info["limitations"]))
    video_path = Path(video_path)
    duration = _video_duration(video_path)
    if analyzer_callable:
        payload = _load_callable(analyzer_callable)(video_path)
    else:
        # Importing the predictor and torch is deferred until local inference is
        # actually requested. Its exported encoder loader never downloads weights.
        predictor = importlib.import_module("colab_forte.forte_benchmark.predict")
        baseline = predictor.predict_video(Path(bundle_path), video_path, device=device)
        if not isinstance(baseline, Mapping) or baseline.get("verdict") not in ("violation", "no_violation"):
            raise ValueError("Бинарная модель вернула некорректный вердикт.")
        payload = {"verdict": baseline["verdict"], "violations": [], "limitations": list(_BASELINE_LIMITATIONS)}
        if baseline["verdict"] == "violation":
            payload["violations"] = [{"type": "Подозрительный фрагмент", "timecode_sec": baseline.get("timecode_sec"),
                "description": "Бинарная модель выделила подозрительное окно. Тип события не классифицирован, "
                "нормативное основание не определено. Время приблизительное; необходима проверка человеком."}]
    result = _normalize(payload, duration)
    _attach_evidence(video_path, result)
    result["model_name"] = info["model_name"]
    return result

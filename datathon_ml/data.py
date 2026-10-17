"""Validated, duplicate-aware training manifests for the cashier video corpus.

Paths, IDs, hashes, annotation times and recording metadata are bookkeeping.
They must never be included in classifier inputs. Annotation times provide
training supervision only; inference must scan the complete unseen video.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata
from typing import Any


VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm"}


def normalize_video_id(value: Any) -> str:
    """Normalize Latin X/Cyrillic Х, spaces, dashes and optional extensions."""
    if value is None or not str(value).strip():
        raise ValueError("Video ID cannot be empty")
    # Handle workbook paths from either operating system.
    stem = str(value).strip().replace("\\", "/").split("/")[-1]
    if Path(stem).suffix.lower() in VIDEO_EXTENSIONS:
        stem = Path(stem).stem
    stem = unicodedata.normalize("NFKC", stem).translate(
        str.maketrans({"Х": "X", "х": "x", "–": "-", "—": "-", "−": "-"})
    )
    stem = re.sub(r"\s+", "", stem).upper()
    match = re.fullmatch(r"X-?(\d+)", stem)
    if match:
        return f"X-{int(match.group(1))}"
    if not stem:
        raise ValueError("Video ID cannot be empty")
    return stem


def parse_timestamp(value: Any) -> float | None:
    """Read seconds, MM:SS, HH:MM:SS, Excel times or Russian/English units."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"Invalid timestamp: {value!r}")
    if isinstance(value, dt.timedelta):
        seconds = value.total_seconds()
    elif isinstance(value, dt.time):
        seconds = value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    elif isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        text = unicodedata.normalize("NFKC", value).strip().lower().replace(",", ".")
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            seconds = float(text)
        elif ":" in text:
            fields = text.split(":")
            if len(fields) not in (2, 3) or not all(
                re.fullmatch(r"\d+(?:\.\d+)?", part.strip()) for part in fields
            ):
                raise ValueError(f"Invalid timestamp: {value!r}")
            numbers = [float(part.strip()) for part in fields]
            if any(number >= 60 for number in numbers[1:]) or any(
                not number.is_integer() for number in numbers[:-1]
            ):
                raise ValueError(f"Invalid clock timestamp: {value!r}")
            seconds = sum(number * 60 ** index for index, number in enumerate(reversed(numbers)))
        else:
            pattern = re.compile(
                r"(\d+(?:\.\d+)?)\s*"
                r"(час(?:а|ов)?|hours?|hrs?|h|минут(?:а|ы)?|мин\.?|minutes?|mins?|m|"
                r"секунд(?:а|ы)?|сек\.?|seconds?|secs?|s)",
                re.IGNORECASE,
            )
            matches = list(pattern.finditer(text))
            remainder = pattern.sub("", text).strip(" .")
            if not matches or remainder:
                raise ValueError(f"Invalid timestamp: {value!r}")
            seconds = 0.0
            for match in matches:
                unit = match.group(2)
                factor = 3600 if unit.startswith(("ч", "h")) else 60 if unit.startswith(("м", "m")) else 1
                seconds += float(match.group(1)) * factor
    else:
        raise ValueError(f"Unsupported timestamp: {value!r}")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Timestamp must be finite and non-negative: {value!r}")
    return float(seconds)


def _parse_label(value: Any) -> int:
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    negatives = {"0", "без нарушения", "без нарушений", "норма", "normal", "compliant", "no violation"}
    positives = {"1", "нарушение", "нарушения", "violation", "non-compliant", "noncompliant"}
    if text in negatives:
        return 0
    if text in positives:
        return 1
    raise ValueError(f"Unknown binary label: {value!r}")


def _read_label_rows(path: Path) -> list[tuple[Any, Any, Any, int]]:
    """Return (video ID, label, timestamp, Excel row number)."""
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError("Install openpyxl to read Excel labels") from exc
    aliases = {
        "id": {"наименование файла", "имя файла", "filename", "file", "video", "video_id"},
        "time": {"таймстемп", "таймстамп", "timestamp", "time", "event_time", "event_time_sec"},
        "label": {"нарушение / без нарушения", "нарушение/без нарушения", "label", "target", "violation"},
    }
    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    candidates = []
    try:
        for sheet in workbook.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            for index, row in enumerate(rows[:20]):
                normalized = [re.sub(r"\s+", " ", str(cell).strip().lower()) for cell in row]
                columns = {
                    key: next((i for i, cell in enumerate(normalized) if cell in options), None)
                    for key, options in aliases.items()
                }
                if all(column is not None for column in columns.values()):
                    records = []
                    for row_number, content in enumerate(rows[index + 1 :], start=index + 2):
                        if not any(cell is not None and str(cell).strip() for cell in content):
                            continue
                        def get(key: str) -> Any:
                            column = columns[key]
                            return content[column] if column < len(content) else None
                        records.append((get("id"), get("label"), get("time"), row_number))
                    candidates.append(records)
                    break
    finally:
        workbook.close()
    if len(candidates) != 1:
        raise ValueError(f"Expected one sheet with ID, timestamp and label headers in {path}; found {len(candidates)}")
    if not candidates[0]:
        raise ValueError(f"Labels workbook is empty: {path}")
    return candidates[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate(value: Any) -> float:
    try:
        if isinstance(value, str) and "/" in value:
            numerator, denominator = value.split("/", 1)
            return float(numerator) / float(denominator)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _validate_metadata(metadata: dict[str, Any], path: Path) -> dict[str, Any]:
    for key in ("duration_sec", "fps", "width", "height"):
        value = metadata.get(key, 0)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {key} for video {path}: {value!r}")
    return metadata


def _probe_video(path: Path) -> dict[str, Any]:
    """Use ffprobe when available, then fall back to OpenCV."""
    errors = []
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height,avg_frame_rate,r_frame_rate,duration:format=duration", "-of", "json", str(path)],
                capture_output=True, text=True, encoding="utf-8", check=True, timeout=60,
            )
            payload = json.loads(result.stdout)
            stream = payload["streams"][0]
            metadata = {
                "duration_sec": _rate(stream.get("duration")) or _rate(payload.get("format", {}).get("duration")),
                "fps": _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate")),
                "width": int(stream["width"]), "height": int(stream["height"]),
            }
            return _validate_metadata(metadata, path)
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
            errors.append(f"ffprobe: {exc}")
    try:
        import cv2
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError("VideoCapture could not open video")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            metadata = {
                "duration_sec": float(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / fps if fps > 0 else 0,
                "fps": fps, "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            return _validate_metadata(metadata, path)
        finally:
            capture.release()
    except (ImportError, OSError, ValueError) as exc:
        errors.append(f"OpenCV: {exc}")
    raise ValueError(f"Cannot read video metadata for {path}: {'; '.join(errors)}")


def _sort_id(video_id: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"(.*?)(\d+)", video_id)
    return (match.group(1), int(match.group(2)), video_id) if match else (video_id, -1, video_id)


def _write_manifest(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(records, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_dir / "manifest.json")
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_manifest(
    data_root: str | Path,
    output_dir: str | Path | None = None,
    labels_path: str | Path | None = None,
    verified_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Validate labeled videos and return one record per exact SHA-256 duplicate.

    ``data_root`` can contain ``open/`` or be the data directory itself. Only
    videos referenced by the workbook are included; unrelated videos are not
    silently treated as normal. Positive labels require valid point timestamps.
    Duplicate annotations are merged, including distinct points on identical
    videos. Conflicting labels and ambiguous or missing files raise ValueError.
    ``verified_aliases`` can explicitly resolve a missing source ID to an existing
    duplicate verified in an earlier audit. This is opt-in; no alias is inferred
    from filename patterns. Targets must exist directly (alias chains fail).
    If requested, atomically save a JSON list at ``output_dir/manifest.json``.
    """
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    if labels_path is None:
        workbooks = sorted(p for p in root.rglob("*.xlsx") if not p.name.startswith("~$"))
        preferred = [p for p in workbooks if p.name.lower() == "open_labels.xlsx"]
        candidates = preferred or workbooks
        if len(candidates) != 1:
            raise ValueError(f"Expected one labels workbook below {root}; found {len(candidates)}. Set labels_path explicitly.")
        workbook = candidates[0]
    else:
        workbook = Path(labels_path).expanduser()
        workbook = workbook if workbook.is_absolute() else root / workbook
        if not workbook.is_file():
            raise FileNotFoundError(f"Labels workbook does not exist: {workbook}")
    videos: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.setdefault(normalize_video_id(path.name), []).append(path.resolve())
    aliases = {}
    for source, target in (verified_aliases or {}).items():
        source_id, target_id = normalize_video_id(source), normalize_video_id(target)
        if source_id in aliases and aliases[source_id] != target_id:
            raise ValueError(f"Conflicting verified aliases for {source_id}")
        aliases[source_id] = target_id
    by_hash: dict[str, dict[str, Any]] = {}
    path_hashes: dict[Path, str] = {}
    for raw_id, raw_label, raw_time, row_number in _read_label_rows(workbook):
        try:
            video_id = normalize_video_id(raw_id)
            label = _parse_label(raw_label)
            timestamp = parse_timestamp(raw_time)
            if label == 1 and timestamp is None:
                raise ValueError("Positive label requires an event timestamp")
            if label == 0 and timestamp is not None:
                raise ValueError("Compliant label cannot carry a violation timestamp")
            matches = videos.get(video_id, [])
            alias_target = aliases.get(video_id)
            if not matches and alias_target is not None:
                matches = videos.get(alias_target, [])
            if len(matches) != 1:
                raise ValueError(f"Expected one video for {video_id}; found {len(matches)}")
            path = matches[0]
            if path not in path_hashes:
                path_hashes[path] = _sha256(path)
            digest = path_hashes[path]
            if digest not in by_hash:
                metadata = _probe_video(path)
                _validate_metadata(metadata, path)
                by_hash[digest] = {
                    "video_id": normalize_video_id(path.name), "path": str(path), "sha256": digest,
                    "label": label, "event_times": [], "aliases": [],
                    **metadata, "group_id": digest,
                }
            record = by_hash[digest]
            if record["label"] != label:
                raise ValueError(f"Conflicting binary labels on exact duplicate videos {record['aliases']} and {video_id}")
            if timestamp is not None and timestamp >= record["duration_sec"]:
                raise ValueError(f"Timestamp {timestamp}s is outside video duration {record['duration_sec']:.3f}s")
            if video_id not in record["aliases"]:
                record["aliases"].append(video_id)
            if timestamp is not None and timestamp not in record["event_times"]:
                record["event_times"].append(timestamp)
            path_id = normalize_video_id(path.name)
            if _sort_id(path_id) < _sort_id(record["video_id"]):
                record["video_id"], record["path"] = path_id, str(path)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid annotation at Excel row {row_number} ({raw_id!r}): {exc}") from exc
    records = sorted(by_hash.values(), key=lambda item: _sort_id(item["video_id"]))
    for record in records:
        record["event_times"].sort()
        record["aliases"].sort(key=_sort_id)
    if output_dir is not None:
        _write_manifest(records, Path(output_dir).expanduser().resolve())
    return records

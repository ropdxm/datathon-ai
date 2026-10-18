"""Immutable manifests, content-addressed video discovery, and weak-label windows.

Filenames and aliases are display metadata. They never determine labels, splits,
or identity. A manifest contains one record per unique source video; exact file
copies are resolved to one deterministic path by ``load_dataset``.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re


SPLITS = frozenset({"train", "val", "test"})


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _validate_record(row):
    if not isinstance(row, dict):
        raise ValueError("Each manifest record must be a dictionary")
    for field in ("video_id", "name", "group_id"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    if not isinstance(row.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
        raise ValueError(f"{row['video_id']}: sha256 must be 64 lowercase hexadecimal characters")
    if type(row.get("label")) is not int or row["label"] not in (0, 1):
        raise ValueError(f"{row['video_id']}: label must be integer 0 or 1")
    if row.get("split") not in SPLITS:
        raise ValueError(f"{row['video_id']}: split must be train, val, or test")
    aliases = row.get("aliases")
    if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
        raise ValueError(f"{row['video_id']}: aliases must be a list of nonempty strings")
    duration = _number(row.get("duration_sec"), "duration_sec")
    if duration <= 0:
        raise ValueError(f"{row['video_id']}: duration_sec must be positive")
    events = row.get("event_times")
    if not isinstance(events, list):
        raise ValueError(f"{row['video_id']}: event_times must be a list")
    for value in events:
        timestamp = _number(value, "event_times")
        if not 0 <= timestamp <= duration:
            raise ValueError(f"{row['video_id']}: event time {timestamp} is outside the video")
    if row["label"] == 1 and not events:
        raise ValueError(f"{row['video_id']}: positive videos require at least one event time")
    if row["label"] == 0 and events:
        raise ValueError(f"{row['video_id']}: negative videos cannot contain event annotations")


def validate_manifest(manifest: list[dict]) -> None:
    """Reject malformed annotations and leakage in a complete three-way split.

    Each split must contain both labels. A video hash may appear only once, and
    every group must belong to a single split. Raises ``ValueError`` on failure.
    Validation does not modify its input.
    """
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("The manifest must be a nonempty list")
    ids, hashes, groups = set(), set(), {}
    labels = {split: set() for split in SPLITS}
    for row in manifest:
        _validate_record(row)
        video_id, digest, group = row["video_id"], row["sha256"], row["group_id"]
        if video_id in ids:
            raise ValueError(f"Duplicate video_id: {video_id}")
        if digest in hashes:
            raise ValueError(f"Duplicate content hash: {digest}; use aliases in one canonical record")
        if group in groups and groups[group] != row["split"]:
            raise ValueError(f"Group {group} crosses splits")
        ids.add(video_id)
        hashes.add(digest)
        groups[group] = row["split"]
        labels[row["split"]].add(row["label"])
    for split, present in sorted(labels.items()):
        if present != {0, 1}:
            raise ValueError(f"Split {split} must contain both labels 0 and 1")


def dataset_fingerprint(manifest: list[dict]) -> str:
    """SHA-256 of canonical annotations and split membership, excluding paths.

    Record order, alias order, and event-time order do not change the fingerprint.
    Additional metadata is included, so an annotation correction changes it.
    """
    canonical = []
    for original in manifest:
        row = {key: deepcopy(value) for key, value in original.items() if key != "path"}
        for field in ("aliases", "event_times"):
            if field in row:
                row[field] = sorted(row[field])
        canonical.append(row)
    canonical.sort(key=lambda row: row["video_id"])
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(root: Path, manifest: list[dict]) -> list[dict]:
    """Find required .MOV content recursively and return copied rows with paths.

    Extension matching is case-insensitive. Unrelated files are ignored. If
    multiple files have identical bytes, the lexicographically first relative
    path is used. No label is inferred from any filename. Every required hash
    must be present or ``FileNotFoundError`` is raised before training begins.
    """
    validate_manifest(manifest)
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    required = {row["sha256"] for row in manifest}
    found = {}
    paths = sorted((path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".mov"),
                   key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        digest = _sha256(path)
        if digest in required:
            found.setdefault(digest, str(path))
    missing = [row["video_id"] for row in manifest if row["sha256"] not in found]
    if missing:
        raise FileNotFoundError("Missing required video content (check upload and SHA-256): " + ", ".join(missing))
    result = deepcopy(manifest)
    for row in result:
        row["path"] = found[row["sha256"]]
    return result


def _uniform_starts(duration: float, window: float, stride: float) -> list[float]:
    tail = max(0.0, duration - window)
    # Multiplication avoids cumulative floating-point drift from repeated addition.
    starts = [i * stride for i in range(int(math.floor(tail / stride)) + 1)]
    if not math.isclose(starts[-1], tail, rel_tol=0, abs_tol=1e-9):
        starts.append(tail)
    else:
        starts[-1] = tail
    return starts


def make_windows(records: list[dict], window_sec: float = 8,
                 stride_sec: float = 2, training: bool = False) -> list[dict]:
    """Build deterministic windows without treating unannotated time as negative.

    ``training=False`` samples every video uniformly, including an exact tail
    window; annotations never influence inference sampling. ``label`` remains
    the video-level ground truth for later aggregation; ``weak_label`` is None.

    ``training=True`` accepts train records only. Negative videos use all uniform
    windows with weak_label=0. Positive videos use one window per annotated event
    with weak_label=1; no other positive-video windows are labelled negative.
    Event-centered windows shift inward at video boundaries to retain the full
    requested duration. Videos shorter than a window use their complete content.
    Duplicate windows from repeated or boundary-clamped annotations are removed.
    ``annotation_source`` explains the sampling rule; only weak_label is suitable
    as the training target.
    """
    window, stride = _number(window_sec, "window_sec"), _number(stride_sec, "stride_sec")
    if window <= 0 or stride <= 0 or stride > window:
        raise ValueError("Require 0 < stride_sec <= window_sec to cover videos without gaps")
    windows = []
    seen_ids = set()
    for row in records:
        _validate_record(row)
        if row["video_id"] in seen_ids:
            raise ValueError(f"Duplicate video_id: {row['video_id']}")
        seen_ids.add(row["video_id"])
        if "path" not in row:
            raise ValueError(f"{row['video_id']}: load_dataset must resolve a path before sampling")
        if training and row["split"] != "train":
            raise ValueError("Training windows may only use records in the train split")
        duration = float(row["duration_sec"])
        if training and row["label"]:
            starts = sorted({max(0.0, min(float(event) - window / 2, max(0.0, duration - window)))
                             for event in row["event_times"]})
            source = "event_centered"
        else:
            starts = _uniform_starts(duration, window, stride)
            source = "uniform"
        for index, start in enumerate(starts):
            end = min(duration, start + window)
            windows.append({
                "sample_id": f"{row['video_id']}:{'train' if training else 'infer'}:{index:05d}",
                "video_id": row["video_id"], "split": row["split"], "label": row["label"],
                "weak_label": row["label"] if training else None,
                "start_sec": start, "end_sec": end, "center_sec": (start + end) / 2,
                "path": str(row["path"]), "duration_sec": duration,
                "event_times": list(row["event_times"]), "is_training": training,
                "annotation_source": source,
            })
    return windows

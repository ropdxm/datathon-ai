"""Prepare small silent RGB frame packs locally, without annotation sampling.

Pixels are decoded only in this native Python process and saved privately for
native training. No image, caption, event description or audio is sent to an AI
API. Each per-source metadata file is deliberately free of target labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datathon_ml.features import iter_video_frames, make_views
from datathon_ml.data import VIDEO_EXTENSIONS
from tools.build_custom_manifest import write_json
from tools.verify_custom_labels import sha256


PREPROCESSING = "raw_rgb_uint8_224_letterbox"
VIEWS = ["full", "middle"]
SAFE_RECORD_FIELDS = {
    "sha256", "source_sha256", "prepared_sha256", "parent_sha256", "original_sha256",
    "lineage_sha256s", "label", "event_times", "domain", "dataset", "group_id",
    "source_family", "label_status", "event_times_verified", "human_verified",
    "evaluation_ground_truth_eligible", "group_verified", "grouping_basis",
    "full_session_group_id", "full_session_grouping_enforced",
    "individual_scene_pairing_verified", "reviewed_workbook_sha256",
    "duration_sec", "source_duration_sec", "fps", "audio_used",
}


def _key(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Each frame pack needs a lowercase content SHA-256")
    return value


def resolve_media_inputs(inputs, media_root):
    """Recover moved files only by exact content SHA inside one explicit root."""
    inputs = [{"sha256": _key(row["sha256"]), "path": str(row["path"])} for row in inputs]
    missing = {row["sha256"] for row in inputs if not Path(row["path"]).is_file()}
    if not missing:
        return inputs, 0
    root = Path(media_root).resolve(strict=True)
    found = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        key = sha256(path)
        if key in missing:
            found.setdefault(key, str(path.resolve()))
    if set(found) != missing:
        raise ValueError("Some moved source hashes have no exact local media match")
    result = []
    for row in inputs:
        result.append({"sha256": row["sha256"],
                       "path": row["path"] if Path(row["path"]).is_file() else found[row["sha256"]]})
    return result, sum(row["path"] != original["path"] for row, original in zip(result, inputs))


def _metadata_valid(metadata, key, sample_fps, jpeg_quality):
    if (metadata.get("source_sha256") != key or metadata.get("views") != VIEWS
            or metadata.get("preprocessing") != PREPROCESSING
            or metadata.get("sample_fps") != sample_fps
            or metadata.get("jpeg_quality") != jpeg_quality):
        raise ValueError("Existing frame pack has a different source or fixed preprocessing protocol")
    times, images = metadata.get("timestamps"), metadata.get("images")
    if not isinstance(times, list) or not times or not isinstance(images, list) or len(times) != len(images):
        raise ValueError("Incomplete frame pack time/image mapping")
    if (any(isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value < 0 for value in times)
            or any(right <= left for left, right in zip(times, times[1:]))):
        raise ValueError("Frame timestamps must be finite, nonnegative and strictly increasing")
    for index, row in enumerate(images):
        if row != [f"{index:06d}_{view}.jpg" for view in VIEWS]:
            raise ValueError("Frame image names must be contained canonical relative JPEG paths")


def prepare_source(source, frame_root, *, sample_fps=1., jpeg_quality=85):
    """Consume only path and actual decoded-file SHA, never target annotations."""
    key = _key(source["sha256"])
    path = Path(source["path"])
    if not path.is_file() or sha256(path) != key:
        raise ValueError("Source media is missing or its bytes differ from the declared SHA")
    if not math.isfinite(sample_fps) or sample_fps <= 0 or not 1 <= jpeg_quality <= 100:
        raise ValueError("Sampling frequency and JPEG quality must be valid")
    directory = Path(frame_root) / key
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path, receipt_path = directory / "metadata.json", directory / "_integrity.json"
    if metadata_path.is_file() and receipt_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        _metadata_valid(metadata, key, sample_fps, jpeg_quality)
        receipt = json.loads(receipt_path.read_text())
        expected_names = {name for row in metadata["images"] for name in row} | {"metadata.json"}
        hashes = receipt.get("members_sha256", {})
        if receipt.get("completed") is not True or set(hashes) != expected_names:
            raise ValueError("Existing frame pack integrity receipt is incomplete")
        if any(not (directory / name).is_file() or sha256(directory / name) != digest
               for name, digest in hashes.items()):
            raise ValueError("Existing frame pack bytes changed")
        return metadata, True
    if metadata_path.exists() or receipt_path.exists() or any(directory.glob("*.jpg")):
        raise ValueError("An incomplete private frame pack already exists; use a new output directory")
    timestamps, images, hashes = [], [], {}
    previous = -math.inf
    for index, (timestamp, rgb) in enumerate(iter_video_frames(path, sample_fps)):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
            raise ValueError("Decoded sampling timestamps must increase strictly")
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise ValueError("Decoder must provide raw RGB uint8 pixels")
        previous = timestamp
        # Exactly the existing CLIP geometry: full aspect-preserving letterbox
        # and a square middle crop along the long image axis. No normalization.
        views = make_views(rgb, 224, VIEWS)
        names = []
        for view, image in zip(VIEWS, views):
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError("Frame geometry must remain RGB uint8 224 square")
            name = f"{index:06d}_{view}.jpg"
            success, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not success:
                raise ValueError("Native JPEG encoding failed")
            payload = encoded.tobytes()
            (directory / name).write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
            names.append(name)
        timestamps.append(timestamp)
        images.append(names)
    if not timestamps:
        raise ValueError("No frames were sampled from the source recording")
    metadata = {"source_sha256": key, "timestamps": timestamps, "images": images,
        "views": VIEWS, "preprocessing": PREPROCESSING, "sample_fps": float(sample_fps),
        "jpeg_quality": jpeg_quality,
        "sampling": "sequential_uniform_elapsed_seconds_plus_final_valid_frame",
        "audio_used": False, "label_guided_sampling": False}
    _metadata_valid(metadata, key, sample_fps, jpeg_quality)
    write_json(metadata_path, metadata)
    hashes["metadata.json"] = sha256(metadata_path)
    write_json(receipt_path, {"completed": True, "members_sha256": hashes})
    return metadata, False


def portable_manifest(records):
    """Separate numeric supervision from image metadata and strip private text."""
    result = []
    for source in records:
        row = {key: value for key, value in source.items() if key in SAFE_RECORD_FIELDS}
        _key(row["sha256"])
        row["frame_dir"] = "frames/" + row["sha256"]
        result.append(row)
    return result


def write_archive(records, split, frame_root, output, *, sample_fps=1., jpeg_quality=85):
    """Write only explicit image/metadata members and separate supervision JSON."""
    members = {
        "dataset/manifest.json": json.dumps(portable_manifest(records), indent=2,
                                            allow_nan=False).encode(),
        "dataset/split.json": json.dumps(split, indent=2, allow_nan=False).encode(),
    }
    hashes, image_count, time_count = {}, 0, 0
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=3) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        for row in sorted(records, key=lambda item: item["sha256"]):
            key = _key(row["sha256"])
            directory = Path(frame_root) / key
            metadata = json.loads((directory / "metadata.json").read_text())
            _metadata_valid(metadata, key, sample_fps, jpeg_quality)
            receipt = json.loads((directory / "_integrity.json").read_text())
            paths = [directory / name for names in metadata["images"] for name in names]
            image_count += len(paths)
            time_count += len(metadata["timestamps"])
            paths += [directory / "metadata.json", directory / "_integrity.json"]
            for path in paths:
                name = f"frames/{key}/{path.name}"
                payload = path.read_bytes()
                actual = hashlib.sha256(payload).hexdigest()
                if path.name != "_integrity.json" and receipt["members_sha256"].get(path.name) != actual:
                    raise ValueError("Frame pack changed while the private archive was being built")
                archive.writestr(name, payload)
                hashes[name] = actual
        archive.writestr("FRAMEPACK_MANIFEST.json", json.dumps({
            "schema_version": 1, "recordings": len(records), "time_samples": time_count,
            "jpeg_images": image_count, "views": VIEWS, "preprocessing": PREPROCESSING,
            "contains_private_video_frames": True, "contains_audio": False,
            "contains_free_text_event_descriptions": False,
            "labels_separate_from_per_source_image_metadata": True,
            "members_sha256": hashes}, indent=2))
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Private frame archive CRC verification failed")
    temporary.replace(output)
    return {"recordings": len(records), "time_samples": time_count,
            "jpeg_images": image_count, "members_with_checksums": len(hashes),
            "archive_bytes": output.stat().st_size, "archive_sha256": sha256(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/reviewed_dataset/manifest.json"))
    parser.add_argument("--split", type=Path, default=Path("artifacts/reviewed_dataset/split.json"))
    parser.add_argument("--frame-root", type=Path, default=Path("artifacts/reviewed_visual_framepack/frames"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reviewed_visual_framepack.zip"))
    parser.add_argument("--audit-output", type=Path, default=Path("artifacts/visual_framepack_audit.json"))
    parser.add_argument("--sample-fps", type=float, default=1.)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--media-root", type=Path, default=ROOT / "open",
                        help="Exact-SHA fallback for moved files, without modifying the frozen manifest")
    args = parser.parse_args()
    from tools.benchmark_semantic_vlm import blocked_network, configure_offline_environment
    configure_offline_environment()
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    records = json.loads(args.manifest.read_text())
    split = json.loads(args.split.read_text())
    if len(records) != 58 or len({row["sha256"] for row in records}) != 58:
        raise ValueError("This reviewed framepack requires the frozen 58 unique recordings")
    if set().union(*(set(split[name]) for name in ("train", "validation", "test"))) != {row["sha256"] for row in records}:
        raise ValueError("Framepack source membership differs from the frozen split")
    # Strip every annotation and split role before native decoding/sampling.
    inputs = [{"sha256": row["sha256"], "path": row["path"]} for row in records]
    started, resumed, metadata = time.perf_counter(), 0, []
    relocated = 0
    with blocked_network() as network:
        try:
            inputs, relocated = resolve_media_inputs(inputs, args.media_root)
            for index, row in enumerate(inputs, 1):
                current, was_resumed = prepare_source(row, args.frame_root,
                    sample_fps=args.sample_fps, jpeg_quality=args.jpeg_quality)
                metadata.append(current)
                resumed += was_resumed
                print(json.dumps({"completed_sources": index, "total_sources": len(inputs),
                                  "sampled_times": sum(len(entry["timestamps"]) for entry in metadata)}), flush=True)
            summary = write_archive(records, split, args.frame_root, args.output,
                                     sample_fps=args.sample_fps, jpeg_quality=args.jpeg_quality)
        except Exception as error:
            # Source filenames and frame data do not appear in console errors.
            raise RuntimeError("Native framepack failed after " + str(len(metadata)) +
                               " sources; error type " + type(error).__name__) from None
    audit = {"schema_version": 1, **summary, "completed": True,
        "elapsed_seconds": time.perf_counter() - started, "resumed_sources": resumed,
        "relocated_source_paths_resolved_by_exact_sha256": relocated,
        "sample_fps": args.sample_fps, "jpeg_quality": args.jpeg_quality,
        "preprocessing": PREPROCESSING, "views": VIEWS,
        "source_hashes_verified_before_decode": True, "source_count": len(inputs),
        "manifest_sha256": sha256(args.manifest), "split_sha256": sha256(args.split),
        "script_sha256": sha256(Path(__file__)), "opencv_version": cv2.__version__,
        "audio_used": False, "network_attempts": network["attempts"],
        "private_pixels_sent_to_hosted_ai": False,
        "label_or_timestamp_guided_sampling": False,
        "long_recordings_truncated": False,
        "maximum_final_timestamp_seconds": max(row["timestamps"][-1] for row in metadata),
        "domain_counts": dict(Counter(row["domain"] for row in records))}
    write_json(args.audit_output, audit)
    print("FRAMEPACK_COMPLETE", json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()

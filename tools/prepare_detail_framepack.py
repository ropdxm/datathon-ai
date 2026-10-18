"""Privately regenerate RGB448 frame packs on the exact frozen RGB224 timeline.

Only the native video decoder sees source pixels. Labels and supplied anchors
never affect sampling. The RGB224 files and metadata are read-only references;
all new images, receipts and archives live in a separate output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datathon_ml.features import iter_video_frames, make_views
from tools.build_custom_manifest import write_json
from tools.prepare_visual_framepack import (PREPROCESSING as REFERENCE_PREPROCESSING,
    _key, portable_manifest, resolve_media_inputs)
from tools.verify_custom_labels import sha256

PREPROCESSING = "raw_rgb_uint8_448_letterbox"
VIEWS = ["full", "middle"]
IMAGE_SIZE = 448
SAMPLE_FPS = 1.
JPEG_QUALITY = 90


def reference_metadata(reference_root, key):
    directory = Path(reference_root) / _key(key)
    path, receipt_path = directory / "metadata.json", directory / "_integrity.json"
    metadata = json.loads(path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (metadata.get("source_sha256") != key or metadata.get("preprocessing") != REFERENCE_PREPROCESSING
            or metadata.get("views") != VIEWS or metadata.get("sample_fps") != SAMPLE_FPS):
        raise ValueError("The frozen reference source, views or one-fps timeline differs")
    if (receipt.get("completed") is not True
            or receipt.get("members_sha256", {}).get("metadata.json") != sha256(path)):
        raise ValueError("Frozen reference metadata integrity mismatch")
    times = metadata.get("timestamps")
    if (not isinstance(times, list) or not times
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in times)
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError("Frozen reference timeline must be finite and increasing")
    return metadata, sha256(path)


def validate_metadata(metadata, key, reference, reference_sha):
    if (metadata.get("source_sha256") != key or metadata.get("preprocessing") != PREPROCESSING
            or metadata.get("views") != VIEWS or metadata.get("image_size") != IMAGE_SIZE
            or metadata.get("sample_fps") != SAMPLE_FPS or metadata.get("jpeg_quality") != JPEG_QUALITY
            or metadata.get("reference_224_metadata_sha256") != reference_sha
            or metadata.get("timestamps") != reference["timestamps"]):
        raise ValueError("Detail pack differs from the frozen source, geometry or exact reference timeline")
    images = metadata.get("images")
    if not isinstance(images, list) or len(images) != len(reference["timestamps"]):
        raise ValueError("Detail image rows do not cover the complete reference timeline")
    for index, row in enumerate(images):
        if row != [f"{index:06d}_{view}.jpg" for view in VIEWS]:
            raise ValueError("Detail image paths must be canonical contained relative JPEG names")


def verify_receipt(directory, metadata):
    directory = Path(directory)
    receipt = json.loads((directory / "_integrity.json").read_text())
    names = {name for row in metadata["images"] for name in row} | {"metadata.json"}
    hashes = receipt.get("members_sha256", {})
    if receipt.get("completed") is not True or set(hashes) != names:
        raise ValueError("Detail integrity receipt is incomplete")
    for name in sorted(names):
        path = directory / name
        if not path.is_file() or sha256(path) != hashes[name]:
            raise ValueError("Detail frame pack bytes changed")
        if name.endswith(".jpg"):
            encoded = np.frombuffer(path.read_bytes(), np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None or image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                raise ValueError("Detail JPEG geometry is not RGB448")
    return receipt


def prepare_source(source, frame_root, reference_root):
    """Consume only source path/SHA; reference timestamps are annotation-free."""
    key, source_path = _key(source["sha256"]), Path(source["path"])
    if not source_path.is_file() or sha256(source_path) != key:
        raise ValueError("Native source differs from its declared SHA")
    output_root, reference_root = Path(frame_root).resolve(), Path(reference_root).resolve()
    if (output_root == reference_root or output_root.is_relative_to(reference_root)
            or reference_root.is_relative_to(output_root)):
        raise ValueError("Detail output must be separate from the frozen224 reference")
    reference, reference_sha = reference_metadata(reference_root, key)
    directory = output_root / key
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path, receipt_path = directory / "metadata.json", directory / "_integrity.json"
    if metadata_path.exists() and receipt_path.exists():
        metadata = json.loads(metadata_path.read_text())
        validate_metadata(metadata, key, reference, reference_sha)
        verify_receipt(directory, metadata)
        return metadata, True
    if metadata_path.exists() or receipt_path.exists() or any(directory.glob("*.jpg")):
        raise ValueError("Incomplete detail source exists; use a new output directory")
    times, images, hashes = [], [], {}
    for index, (timestamp, rgb) in enumerate(iter_video_frames(source_path, SAMPLE_FPS)):
        timestamp = float(timestamp)
        if index >= len(reference["timestamps"]) or timestamp != reference["timestamps"][index]:
            raise ValueError("Decoded detail timestamp differs from the frozen224 reference")
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise ValueError("Native source must decode to RGB uint8")
        views = make_views(rgb, IMAGE_SIZE, VIEWS)
        names = []
        for view, pixels in zip(VIEWS, views):
            if pixels.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or pixels.dtype != np.uint8:
                raise ValueError("Native detail view geometry differs from RGB448")
            name = f"{index:06d}_{view}.jpg"
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise ValueError("Native detail JPEG encoding failed")
            payload = encoded.tobytes()
            (directory / name).write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
            names.append(name)
        times.append(timestamp); images.append(names)
    if times != reference["timestamps"]:
        raise ValueError("Detail decode did not cover the complete frozen224 timeline")
    metadata = {"source_sha256": key, "timestamps": times, "images": images, "views": VIEWS,
        "preprocessing": PREPROCESSING, "image_size": IMAGE_SIZE, "sample_fps": SAMPLE_FPS,
        "jpeg_quality": JPEG_QUALITY, "reference_224_metadata_sha256": reference_sha,
        "sampling": "exact_same_sequential_uniform_elapsed_seconds_plus_final_valid_frame_as224",
        "source_pixels_redecoded_at_native_resolution": True, "audio_used": False,
        "label_guided_sampling": False, "timestamps_equal_frozen_224": True}
    validate_metadata(metadata, key, reference, reference_sha)
    write_json(metadata_path, metadata)
    hashes["metadata.json"] = sha256(metadata_path)
    write_json(receipt_path, {"completed": True, "members_sha256": hashes})
    verify_receipt(directory, metadata)
    return metadata, False


def write_archive(records, split, frame_root, reference_root, output):
    members = {"dataset/manifest.json": json.dumps(portable_manifest(records), indent=2, allow_nan=False).encode(),
               "dataset/split.json": json.dumps(split, indent=2, allow_nan=False).encode()}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    hashes, image_count, time_count, sizes = {}, 0, 0, []
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=3) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload); hashes[name] = hashlib.sha256(payload).hexdigest()
        for row in sorted(records, key=lambda r: r["sha256"]):
            key = _key(row["sha256"])
            directory = Path(frame_root) / key
            metadata = json.loads((directory / "metadata.json").read_text())
            reference, reference_sha = reference_metadata(reference_root, key)
            validate_metadata(metadata, key, reference, reference_sha)
            receipt = verify_receipt(directory, metadata)
            names = [name for names in metadata["images"] for name in names] + ["metadata.json", "_integrity.json"]
            image_count += len(metadata["images"]) * len(VIEWS)
            time_count += len(metadata["timestamps"])
            for name in names:
                path = directory / name
                payload = path.read_bytes()
                actual = hashlib.sha256(payload).hexdigest()
                if name != "_integrity.json" and actual != receipt["members_sha256"][name]:
                    raise ValueError("Detail source bytes changed during private archive write")
                member = f"frames/{key}/{name}"
                archive.writestr(member, payload); hashes[member] = actual
                if name.endswith(".jpg"):
                    sizes.append(len(payload))
        inventory = {"schema_version": 1, "recordings": len(records), "time_samples": time_count,
            "jpeg_images": image_count, "views": VIEWS, "preprocessing": PREPROCESSING,
            "image_size": IMAGE_SIZE, "sample_fps": SAMPLE_FPS, "jpeg_quality": JPEG_QUALITY,
            "timestamps_equal_frozen_224": True, "contains_private_video_frames": True,
            "contains_audio": False, "contains_free_text_event_descriptions": False,
            "labels_separate_from_per_source_image_metadata": True, "members_sha256": hashes}
        archive.writestr("DETAIL_FRAMEPACK_MANIFEST.json", json.dumps(inventory, indent=2))
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Detail archive CRC verification failed")
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("Detail archive member checksum failed")
    temporary.replace(output)
    return {"recordings": len(records), "time_samples": time_count, "jpeg_images": image_count,
            "members_with_checksums": len(hashes), "archive_bytes": output.stat().st_size,
            "archive_sha256": sha256(output), "jpeg_bytes_median": float(np.median(sizes)),
            "jpeg_bytes_total": sum(sizes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/reviewed_dataset/manifest.json")
    parser.add_argument("--split", type=Path, default=ROOT / "artifacts/reviewed_dataset/split.json")
    parser.add_argument("--reference-root", type=Path, default=ROOT / "artifacts/reviewed_visual_framepack/frames")
    parser.add_argument("--frame-root", type=Path, default=ROOT / "artifacts/reviewed_detail_framepack/frames")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reviewed_detail_framepack.zip")
    parser.add_argument("--audit-output", type=Path, default=ROOT / "artifacts/detail_framepack_audit.json")
    parser.add_argument("--media-root", type=Path, default=ROOT / "open")
    args = parser.parse_args()
    from tools.benchmark_semantic_vlm import blocked_network, configure_offline_environment
    configure_offline_environment()
    cv2.setNumThreads(2)
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    records, split = json.loads(args.manifest.read_text()), json.loads(args.split.read_text())
    if len(records) != 58 or len({r["sha256"] for r in records}) != 58:
        raise ValueError("Detail protocol requires the exact58unique frozen sources")
    from tools.prepare_reviewed_protocol import validate_split
    validate_split(records, split)
    inputs = [{"sha256": row["sha256"], "path": row["path"]} for row in records]
    started, resumed, metadata = time.perf_counter(), 0, []
    with blocked_network() as network:
        try:
            inputs, relocated = resolve_media_inputs(inputs, args.media_root)
            for index, source in enumerate(inputs, 1):
                current, was_resumed = prepare_source(source, args.frame_root, args.reference_root)
                resumed += was_resumed; metadata.append(current)
                print(json.dumps({"completed_sources": index, "total_sources": len(inputs),
                                  "time_samples": sum(len(r["timestamps"]) for r in metadata)}), flush=True)
            if sum(len(r["timestamps"]) for r in metadata) != 1879:
                raise ValueError("The exact frozen detail protocol requires1879time samples")
            result = write_archive(records, split, args.frame_root, args.reference_root, args.output)
        except Exception as error:
            raise RuntimeError("Native detail preparation failed after " + str(len(metadata)) +
                               " sources; error type " + type(error).__name__) from None
    audit = {"schema_version": 1, **result, "completed": True, "elapsed_seconds": time.perf_counter() - started,
        "resumed_sources": resumed, "relocated_source_paths_resolved_by_exact_sha256": relocated,
        "image_size": IMAGE_SIZE, "sample_fps": SAMPLE_FPS, "jpeg_quality": JPEG_QUALITY,
        "preprocessing": PREPROCESSING, "views": VIEWS, "timestamps_equal_frozen_224": True,
        "reference_224_images_or_metadata_modified": False, "source_hashes_verified_before_decode": True,
        "manifest_sha256": sha256(args.manifest), "split_sha256": sha256(args.split),
        "script_sha256": sha256(__file__), "opencv_version": cv2.__version__, "audio_used": False,
        "network_attempts": network["attempts"], "private_pixels_sent_to_hosted_ai": False,
        "label_or_timestamp_guided_sampling": False, "long_recordings_truncated": False,
        "maximum_final_timestamp_seconds": max(r["timestamps"][-1] for r in metadata),
        "domain_counts": dict(Counter(r["domain"] for r in records))}
    write_json(args.audit_output, audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()

"""Six-window, two-resolution local annotation diagnostic; never estimates accuracy.

No labels, teacher heads, audio, frame outputs, generated answers or per-video
observations are consumed/exported. Only aggregate code counts are retained.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CAMERAS = ("iphone_mov", "samsung_mp4")
RESOLUTIONS = (448, 896)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def uniform_indices(length, count):
    if count < 2 or length < count:
        raise ValueError("Uniform selection needs at least count distinct observations")
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def select_records(records):
    """Three SHA-order quantiles per camera, independent of any supplied labels."""
    selected = []
    for camera in CAMERAS:
        eligible = sorted((r for r in records if r.get("camera_id") == camera
            and math.isfinite(float(r.get("duration_sec", 0))) and float(r["duration_sec"]) >= 8.),
            key=lambda row: row["sha256"])
        if len({r["sha256"] for r in eligible}) != len(eligible):
            raise ValueError("Repeated prepared SHA in diagnostic candidates")
        for index in uniform_indices(len(eligible), 3):
            r = eligible[index]
            duration = float(r["duration_sec"])
            start = (duration - 8.) / 2.
            selected.append({"sha256": r["sha256"], "source_sha256": r.get("source_sha256"),
                "camera_id": camera, "path": str(Path(r["path"]).resolve()),
                "duration_sec": duration, "start_sec": start, "end_sec": start + 8.})
    if len({r["sha256"] for r in selected}) != 6:
        raise ValueError("The six selected recordings must be distinct")
    return selected


def middle_images(record, size, frame_reader, resize):
    """Eight uniformly spaced samples strictly inside the same middle window."""
    frames = []
    for timestamp, pixels in frame_reader(record["path"], 2.):
        if timestamp >= record["end_sec"]:
            break
        if timestamp >= record["start_sec"]:
            frames.append(resize(pixels, "full", size))
    return [frames[index] for index in uniform_indices(len(frames), 8)]


def worker(args):
    from tools.benchmark_semantic_vlm import blocked_network, configure_offline_environment
    configure_offline_environment()
    started = time.perf_counter()
    report = {"resolution": args.resolution, "status": "failed", "completed_windows": 0,
        "valid_windows": 0, "invalid_windows": 0,
        "valid_code_counts": {str(code): 0 for code in (0, 1, 2, 3, 4, 5, 9)},
        "inference_seconds": 0.,
        "accuracy_evaluation": False, "verified_labels_created": 0, "audio_used": False,
        "hosted_ai_api_used": False}
    network = {"attempts": None}
    latencies = []
    torch = None
    try:
        with blocked_network() as network:
            import torch
            from huggingface_hub import snapshot_download
            from datathon_ml.custom_annotation import (AnnotationConfig, LocalAnnotator,
                MODEL_ID, MODEL_REVISION, PROMPT, SYSTEM, assert_silent)
            from datathon_ml.features import iter_video_frames, sha256_file
            from datathon_ml.local_vlm import _resize_view, checkpoint_fingerprint
            if args.device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA diagnostic requested without a GPU")
                torch.cuda.reset_peak_memory_stats()
            checkpoint = (args.checkpoint.resolve(strict=True) if args.checkpoint else
                Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)))
            config = AnnotationConfig(image_size=args.resolution)
            protocol = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "checkpoint_sha256": checkpoint_fingerprint(checkpoint), "config": asdict(config),
                "system": SYSTEM, "prompt": PROMPT,
                "annotation_module_sha256": sha256_file(ROOT / "datathon_ml/custom_annotation.py"),
                "sampling": "middle8s-eight-uniform-decoded-2fps-strictly-inside-v1"}
            protocol_json = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
            report["protocol_sha256"] = hashlib.sha256(protocol_json.encode()).hexdigest()
            report["checkpoint_sha256"] = protocol["checkpoint_sha256"]
            report["annotation_config"] = asdict(config)
            report["model_id"], report["model_revision"] = MODEL_ID, MODEL_REVISION
            load_started = time.perf_counter()
            observer = LocalAnnotator(checkpoint, config, args.device)
            if args.device == "cuda":
                torch.cuda.synchronize()
            report["model_load_seconds"] = time.perf_counter() - load_started
            selected = json.loads((args.output_dir / "selection.json").read_text())["records"]
            for record in selected:
                assert_silent(record["path"])
                if sha256_file(record["path"]) != record["sha256"]:
                    raise ValueError("Prepared recording SHA differs from selection audit")
                images = middle_images(record, args.resolution, iter_video_frames, _resize_view)
                began = time.perf_counter()
                code, _, valid = observer.observe(images)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - began
                report["completed_windows"] += 1
                report["inference_seconds"] += elapsed
                latencies.append(elapsed)
                report["valid_windows"] += int(valid)
                report["invalid_windows"] += int(not valid)
                if valid:
                    report["valid_code_counts"][str(code)] += 1
                del images, code
            report["status"] = "completed"
    except Exception as error:
        is_oom = torch is not None and isinstance(error, torch.OutOfMemoryError)
        report.update(status="cuda_out_of_memory" if is_oom else "failed", error_type=type(error).__name__)
    finally:
        report.update(process_seconds=time.perf_counter() - started, network_guard=network)
        report["latency_seconds"] = {"count": len(latencies),
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None}
        if torch is not None and args.device == "cuda" and torch.cuda.is_available():
            report["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
            report["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024 ** 2
            torch.cuda.empty_cache()
        if network.get("attempts") not in (None, 0):
            report["status"] = "blocked_network_attempt"
        write_json(args.output_dir / f"resolution_{args.resolution}.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--budget-seconds", type=float, default=300.)
    parser.add_argument("--resolution", type=int, choices=RESOLUTIONS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resolution:
        return worker(args)
    if args.manifest is None or not 0 < args.budget_seconds <= 300:
        raise ValueError("Provide a manifest and a positive diagnostic budget no larger than 300 seconds")
    began = time.monotonic()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = select_records(payload.get("records") if isinstance(payload, dict) else payload)
    audit = {"selection_policy": "three SHA-order quantiles per camera; duration at least8s; no labels read",
        "records": selected, "resolutions": list(RESOLUTIONS), "frames_per_window": 8,
        "audio_used": False, "original_trained_teacher_used": False,
        "underlying_scenes_may_be_paired": True, "accuracy_evaluation": False,
        "diagnostic_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.output_dir / "selection.json", audit)
    summary = {"budget_seconds": args.budget_seconds, "selected_recordings": 6,
        "recordings_per_camera": 3, "resolutions": {}, "accuracy_evaluation": False,
        "verified_labels_created": 0, "audio_used": False, "hosted_ai_api_used": False,
        "interpretation": "Resolution sensitivity of unverified numeric review-pattern observations; no correctness or absence of violations established."}
    for resolution in RESOLUTIONS:
        remaining = args.budget_seconds - (time.monotonic() - began)
        path = args.output_dir / f"resolution_{resolution}.json"
        if remaining <= 1:
            row = {"resolution": resolution, "status": "not_run_total_deadline", "network_guard": {"attempts": None}}
        else:
            path.unlink(missing_ok=True)  # Never mistake a previous run for this fresh worker.
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--output-dir", str(args.output_dir),
                       "--device", args.device, "--resolution", str(resolution)]
            if args.checkpoint:
                command += ["--checkpoint", str(args.checkpoint.resolve())]
            with (args.output_dir / f"resolution_{resolution}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    cwd=ROOT, start_new_session=(os.name != "nt"))
                try:
                    returncode = process.wait(timeout=min(145., remaining))
                    row = json.loads(path.read_text()) if path.is_file() else {
                        "resolution": resolution, "status": "worker_failed", "returncode": returncode,
                        "network_guard": {"attempts": None}}
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                    row = {"resolution": resolution, "status": "hard_timeout", "network_guard": {"attempts": None}}
                    write_json(path, row)
        summary["resolutions"][str(resolution)] = row
        summary["elapsed_seconds"] = time.monotonic() - began
        write_json(args.output_dir / "summary.json", summary)
        print("CUSTOM_DETAIL_DIAGNOSTIC", json.dumps({"resolution": resolution, **row}), flush=True)
    summary["all_completed_workers_zero_network_attempts"] = all(
        row.get("network_guard", {}).get("attempts") == 0 for row in summary["resolutions"].values())
    write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()

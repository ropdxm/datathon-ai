"""Offline custom-video augmentation experiment; stdout is aggregate metadata.

The archive contains silent derivatives. Local Qwen proposals remain unverified
and only enter explicitly marked train-only sensitivity experiments. Original
40-video labels and historical test folds are the evaluation targets.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import stat
import subprocess
import sys
import time
import zipfile


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_records(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rows = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Manifest requires a list of records")
    return rows


def safe_extract(archive, destination, expected_sha256):
    """Only contained regular MP4/JSON files; reject links and duplicate names."""
    if sha256_file(archive) != expected_sha256.lower():
        raise ValueError("Uploaded custom archive SHA-256 differs from the requested archive")
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as packed:
        entries, seen = [], set()
        if sum(info.file_size for info in packed.infolist()) > 2 * 1024 ** 3:
            raise ValueError("Custom archive exceeds the two-GiB extraction budget")
        for info in packed.infolist():
            name = info.filename
            relative = PurePosixPath(name)
            target = (destination / Path(*relative.parts)).resolve()
            if ("\\" in name or ":" in name or relative.is_absolute()
                    or ".." in relative.parts or not target.is_relative_to(destination)
                    or stat.S_ISLNK(info.external_attr >> 16) or target in seen):
                raise ValueError("Archive contains an unsafe, linked or duplicate path")
            seen.add(target)
            if info.is_dir():
                continue
            if target.suffix.lower() not in {".mp4", ".json"}:
                raise ValueError("Custom archive may contain only prepared MP4s and manifest JSON")
            entries.append((info, target))
        for info, target in entries:
            target.parent.mkdir(parents=True, exist_ok=True)
            with packed.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    return destination


def prepare_portable_manifest(data_dir, output_dir, expected_count):
    """Verify every prepared byte stream, silence and local timeline before use."""
    source = data_dir / "portable_manifest.json"
    if not source.is_file():
        source = data_dir / "manifest.json"
    if not source.is_file():
        raise FileNotFoundError("The custom archive needs portable_manifest.json or manifest.json")
    records = read_records(source)
    if len(records) != expected_count:
        raise ValueError("Custom manifest count differs from the expected number of prepared videos")
    seen_sha, seen_source = set(), set()
    for record in records:
        raw_path = str(record["path"])
        relative = PurePosixPath(raw_path)
        path = (data_dir / Path(*relative.parts)).resolve()
        if (relative.is_absolute() or ".." in relative.parts or "\\" in raw_path
                or ":" in raw_path or not path.is_relative_to(data_dir.resolve())
                or path.suffix.lower() != ".mp4" or not path.is_file()):
            raise ValueError("Portable prepared path is not an existing contained MP4")
        digest = record["sha256"]
        if digest != record.get("prepared_sha256", digest) or sha256_file(path) != digest:
            raise ValueError("Prepared video bytes differ from their recorded SHA-256")
        if digest in seen_sha or record["source_sha256"] in seen_source:
            raise ValueError("Custom manifest repeats a prepared or original source")
        seen_sha.add(digest)
        seen_source.add(record["source_sha256"])
        probe = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file",
            "-show_entries", "stream=codec_type:format=duration", "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=30)
        metadata = json.loads(probe.stdout)
        streams = metadata.get("streams", [])
        if len(streams) != 1 or streams[0].get("codec_type") != "video":
            raise ValueError("Prepared files must contain exactly one video stream and no audio")
        duration = float(metadata["format"]["duration"])
        if abs(duration - float(record["duration_sec"])) > .15:
            raise ValueError("Prepared duration differs from its timestamp mapping manifest")
        if record.get("label") is not None or record.get("event_times", []):
            raise ValueError("Portable custom input must remain unlabeled until explicit proposal merge")
        record.update(path=str(path), label=None, event_times=[], label_status="unlabeled",
                      audio_used=False, audio_streams=0, evaluation_ground_truth_eligible=False)
    write_json(output_dir / "custom_unlabeled_manifest.json", records)
    write_json(output_dir / "portable_manifest.json", json.loads(source.read_text(encoding="utf-8")))
    write_json(output_dir / "archive_audit.json", {"records": len(records),
        "all_prepared_sha256_verified": True, "all_prepared_video_only": True,
        "verified_custom_labels": 0, "custom_sessions": 1,
        "same_scenes_simultaneous_two_camera_views": True, "pairing_resolved": False,
        "audio_used": False, "hosted_ai_api_used": False})
    return records


def _worker(args, workspace, output):
    """Each GPU/model phase lives in a fresh process with a Python socket guard."""
    sys.path.insert(0, str(workspace))
    from tools.benchmark_semantic_vlm import blocked_network, configure_offline_environment
    configure_offline_environment()
    started = time.perf_counter()
    runtime = {"phase": args.worker, "completed": False, "audio_used": False,
               "hosted_ai_api_used": False, "guard_scope": "Python sockets, DNS and datagrams; not an OS firewall"}
    counts = {"attempts": None}
    try:
        with blocked_network() as counts:
            if args.worker == "clip":
                from datathon_ml.clip_features import ClipFeatureConfig, extract_clip_features
                records = read_records(output / "custom_unlabeled_manifest.json")
                paths = extract_clip_features(records, output / "clip_features",
                    ClipFeatureConfig(device="cuda", batch_size=32, allow_download=False))
                runtime["completed_records"] = len(paths)
            elif args.worker == "semantic":
                from datathon_ml.semantic_clip import convert_clip_cache
                paths = convert_clip_cache(output / "clip_features", output / "semantic_features",
                                           device="cpu", offline=True)
                runtime["completed_records"] = len(paths)
            elif args.worker == "annotation":
                from datathon_ml.custom_annotation import main
                main(["--manifest", str(output / "custom_unlabeled_manifest.json"),
                      "--output-dir", str(output / "annotation"), "--device", "cuda"])
            else:
                from custom_experiment import main
                policy = args.worker.removeprefix("compare_")
                original_features = (workspace / "runs/semantic_temporal_v1/semantic_clip/features"
                                     if policy == "semantic" else workspace / "runs/clip_b32/features")
                auxiliary_features = output / ("semantic_features" if policy == "semantic" else "clip_features")
                main(["compare", "--original-manifest", str(workspace / "runs/clip_b32/manifest.json"),
                    "--aux-manifest", str(output / "custom_aux_manifest.json"),
                    "--original-features", str(original_features), "--aux-features", str(auxiliary_features),
                    "--folds-from", str(workspace / "reference/clip_baseline_cv.json"),
                    "--output-dir", str(output / f"comparison_{policy}"), "--policy", policy,
                    "--allow-machine-aux", "--aux-weight", ".25", "--fit-final"])
        runtime["completed"] = True
    finally:
        runtime.update(elapsed_seconds=time.perf_counter() - started, network_guard=counts)
        write_json(output / "logs" / f"{args.worker}_runtime.json", runtime)
    if counts["attempts"] != 0:
        raise RuntimeError("A worker attempted a blocked network operation")


def run_worker(phase, workspace, output, timeout):
    log = output / "logs" / f"{phase}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print("CUSTOM_PHASE_STARTED", json.dumps({"phase": phase, "timeout_seconds": timeout}), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
            "--workspace", str(workspace), "--output-dir", str(output), "--worker", phase],
            cwd=workspace, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"))
        try:
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(phase, timeout)
                try:
                    code = process.wait(timeout=min(30, remaining))
                    break
                except subprocess.TimeoutExpired:
                    progress = {"phase": phase, "elapsed_seconds": round(time.monotonic() - started, 1)}
                    proposal_path = output / "annotation/proposals.json"
                    if phase == "annotation" and proposal_path.is_file():
                        progress["annotation"] = json.loads(proposal_path.read_text())["summary"]
                    print("CUSTOM_PHASE_PROGRESS", json.dumps(progress), flush=True)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            write_json(output / "logs" / f"{phase}_timeout.json", {"phase": phase,
                "hard_timeout_seconds": timeout, "completed": False,
                "network_attempts": None, "reason": "killed before guard finalization"})
            raise TimeoutError(f"{phase} exceeded its hard time budget; its log remains local")
    if code:
        raise RuntimeError(f"{phase} failed with exit {code}; inspect its local log")
    runtime = json.loads((output / "logs" / f"{phase}_runtime.json").read_text())
    if not runtime["completed"] or runtime["network_guard"]["attempts"] != 0:
        raise RuntimeError("Worker did not complete with zero Python network attempts")
    print("CUSTOM_PHASE_COMPLETE", json.dumps({"phase": phase,
        "elapsed_seconds": round(runtime["elapsed_seconds"], 2), "network_attempts": 0}), flush=True)
    return runtime


def export_results(workspace, output, destination):
    """Include compact proposals/reports/trusted heads/source; omit video/caches."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    top_files = ("summary.json", "runtime.json", "archive_audit.json", "portable_manifest.json",
                 "custom_unlabeled_manifest.json", "custom_aux_manifest.json")
    paths = [output / name for name in top_files]
    paths += [output / "annotation" / name for name in ("proposals.json", "protocol.json")]
    paths += list((output / "logs").glob("*.log")) + list((output / "logs").glob("*.json"))
    for policy in ("semantic", "point"):
        paths += [p for p in (output / f"comparison_{policy}").rglob("*")
                  if p.is_file() and p.suffix in {".json", ".joblib"}]
    sources = [*workspace.glob("datathon_ml/*.py"), *workspace.glob("tools/*.py"),
               *workspace.glob("tests/test_*.py"), *workspace.glob("*experiment.py"),
               *workspace.glob("requirements*.txt"), workspace / "README_ML.md",
               workspace / "case_requirement.txt", workspace / "duplicate_aliases.json",
               workspace / "reference/clip_baseline_cv.json"]
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            if path.is_file():
                archive.write(path, "runs/custom_v1/" + path.relative_to(output).as_posix())
        source_manifest = {}
        for path in sorted(set(sources)):
            if path.is_file():
                name = path.relative_to(workspace).as_posix()
                archive.write(path, "source/" + name)
                source_manifest[name] = sha256_file(path)
        archive.writestr("source_sha256_manifest.json", json.dumps(source_manifest, indent=2))
    print("CUSTOM_RESULTS_EXPORT_READY", json.dumps({"path": str(destination),
        "bytes": destination.stat().st_size, "sha256": sha256_file(destination),
        "videos_included": False, "feature_caches_included": False}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/datathon_ml_workspace"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-sha256", default="9d6ddccba50084bc9b23856e860319d7dcb85e231130f30d7ff4663a0660e343")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/custom_v1"))
    parser.add_argument("--results-archive", type=Path)
    parser.add_argument("--expected-videos", type=int, default=39)
    parser.add_argument("--annotation-timeout", type=float, default=2700.)
    parser.add_argument("--worker", choices=("clip", "semantic", "annotation", "compare_semantic", "compare_point"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve(strict=True)
    output = (workspace / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    if not output.is_relative_to(workspace) or output == workspace:
        raise ValueError("Output must be a dedicated directory inside the declared workspace")
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(workspace)
    if args.worker:
        return _worker(args, workspace, output)
    if args.archive is None or args.expected_videos < 1 or not 0 < args.annotation_timeout < float("inf"):
        raise ValueError("Provide a custom archive, positive video count and finite annotation time budget")
    sys.path.insert(0, str(workspace))
    destination = args.results_archive or (Path("/content/datathon_custom_results.zip")
        if Path("/content").is_dir() else workspace / "artifacts/datathon_custom_results.zip")
    original_manifest = workspace / "runs/clip_b32/manifest.json"
    if len(read_records(original_manifest)) != 40:
        raise ValueError("The comparison requires the original 40 unique recordings")
    for path in (workspace / "runs/clip_b32/features", workspace / "reference/clip_baseline_cv.json",
                 workspace / "runs/semantic_temporal_v1/semantic_clip/features/_text"):
        if not path.exists():
            raise FileNotFoundError("Required original cache/reference is missing: " + str(path))
    summary = {"status": "running", "models": {}, "target_accuracy": .87,
        "original_test_recordings": 40, "custom_human_verified_labels": 0,
        "independent_87_percent_accuracy_established": False,
        "audio_used": False, "hosted_ai_api_used": False,
        "limitations": ["Custom labels are unverified machine proposals, never evaluation ground truth.",
            "Paired simultaneous views represent fewer underlying scenes; pairing is unresolved; custom footage is train-only.",
            "All reported accuracy uses historical original test folds, repeatedly inspected development data."]}
    started = time.perf_counter()
    try:
        data = safe_extract(args.archive.resolve(strict=True), output / "data", args.archive_sha256)
        records = prepare_portable_manifest(data, output, args.expected_videos)
        print("CUSTOM_ARCHIVE_VERIFIED", json.dumps({"videos": len(records), "audio_streams": 0}), flush=True)
        run_worker("clip", workspace, output, 1800)
        # Copy the tiny exact original text prototypes, never recompute them on
        # a different precision/device and accidentally change semantic features.
        prototype_dir = output / "semantic_features/_text"
        prototype_dir.mkdir(parents=True, exist_ok=True)
        prototypes = list((workspace / "runs/semantic_temporal_v1/semantic_clip/features/_text").glob("text_*.npz"))
        if not prototypes:
            raise FileNotFoundError("Exact original CLIP text prototypes are required")
        for path in prototypes:
            shutil.copy2(path, prototype_dir / path.name)
        run_worker("semantic", workspace, output, 600)
        try:
            run_worker("annotation", workspace, output, args.annotation_timeout)
            summary["annotation_execution_status"] = "completed"
        except TimeoutError:
            summary["annotation_execution_status"] = "hard_timeout_partial_proposals"
            if not (output / "annotation/proposals.json").is_file():
                raise
        proposals = json.loads((output / "annotation/proposals.json").read_text())
        from tools.build_custom_manifest import _apply_proposals
        auxiliary = copy.deepcopy(records)
        _apply_proposals(auxiliary, proposals, allow_machine_labels=True)
        write_json(output / "custom_aux_manifest.json", auxiliary)
        summary["annotation"] = proposals["summary"]
        summary["custom_eligible_machine_labels"] = sum(r.get("label") is not None for r in auxiliary)
        summary["custom_unresolved"] = sum(r.get("label") is None for r in auxiliary)
        write_json(output / "summary.json", summary)
        for policy in ("semantic", "point"):
            run_worker("compare_" + policy, workspace, output, 1800)
            report = json.loads((output / f"comparison_{policy}/custom_cv_report.json").read_text())
            summary["models"][policy] = {"status": report["status"],
                "policies": {name: value["metrics"] for name, value in report["policies"].items()},
                "paired_change": report.get("paired_change"),
                "auxiliary_supervision_sha256": report["auxiliary"]["supervision_manifest"]["sha256"]}
            summary["target_achieved_on_original_development"] = any(
                values["accuracy"] >= .87 for model in summary["models"].values() for values in model["policies"].values())
            write_json(output / "summary.json", summary)
            print("CUSTOM_MEASURED_RESULT", json.dumps({"policy": policy, **summary["models"][policy]}), flush=True)
            export_results(workspace, output, destination)
        summary["status"] = "completed_weak_label_sensitivity"
    except Exception as error:
        summary.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "summary.json", summary)
        write_json(output / "runtime.json", {"elapsed_seconds": summary["elapsed_seconds"],
            "status": summary["status"], "annotation_timeout_seconds": args.annotation_timeout,
            "annotation_latency_is_deployment_inference_latency": False})
        export_results(workspace, output, destination)


if __name__ == "__main__":
    main()

"""Fresh-process, offline local perception benchmark; never prints observations.

Run in a new Python process after public checkpoint downloads have completed.
Use an external subprocess timeout for a hard deadline; --budget-seconds only
evaluates measured completion time. Explicit --model inputs must be trusted
joblib artifacts from this project. This script does not estimate accuracy.
"""
from __future__ import annotations

import time

# Start before importing any model, numerical library or argument parser.
PROCESS_STARTED = time.perf_counter()

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback


def configure_offline_environment():
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_PROGRESS_BARS"):
        os.environ[key] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


@contextmanager
def blocked_network():
    """Reject and count Python socket connection/DNS/datagram attempts.

    This is a process-level Python guard, not an OS firewall. It covers the
    project's HTTP/Hugging Face clients and also prevents DNS resolution; no
    network probe is sent to demonstrate blocking. Addresses are never logged.
    """
    counts = {"attempts": 0, "by_method": {}}
    originals = []

    def block(name):
        def denied(*args, **kwargs):
            counts["attempts"] += 1
            counts["by_method"][name] = counts["by_method"].get(name, 0) + 1
            raise OSError("Network disabled by local inference benchmark")
        return denied

    targets = [(socket.socket, name) for name in ("connect", "connect_ex", "sendto")]
    targets += [(socket, name) for name in ("create_connection", "getaddrinfo",
        "gethostbyname", "gethostbyname_ex", "gethostbyaddr")]
    try:
        for owner, name in targets:
            originals.append((owner, name, getattr(owner, name)))
            setattr(owner, name, block(name))
        yield counts
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def _source_fingerprint(module):
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def _trusted_model(path):
    """Fail before perception if a trusted head belongs to different code."""
    import joblib
    import datathon_ml.temporal as temporal
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("kind") not in {"semantic_temporal", "semantic_head"}:
        raise ValueError("Expected a trusted semantic_temporal or semantic_head artifact")
    provenance = artifact.get("provenance", {})
    if artifact["kind"] == "semantic_temporal":
        saved = [provenance[key] for key in ("code_sha256", "temporal_code_sha256") if key in provenance]
        if not saved or any(value != _source_fingerprint(temporal) for value in saved):
            raise ValueError("Temporal source fingerprint differs; use matching fitted source")
    else:
        import datathon_ml.semantic_heads as heads
        if (provenance.get("code_sha256") != _source_fingerprint(heads)
                or provenance.get("temporal_code_sha256") != _source_fingerprint(temporal)):
            raise ValueError("Semantic head/temporal source fingerprint differs; use matching fitted source")
    return artifact


def _config(args, artifact):
    from datathon_ml.semantic_vlm import SemanticVLMConfig, perception_protocol
    from datathon_ml.temporal import CONCEPT_NAMES
    metadata = artifact["feature_config"] if artifact is not None else {}
    if artifact is not None:
        if (metadata.get("backbone") != "qwen3_vl_4b_observable_states"
                or metadata.get("views") != ["full"]
                or metadata.get("concept_names") != list(CONCEPT_NAMES)):
            raise ValueError("The supplied head was not trained on this Qwen numeric-state schema")
    context = metadata.get("context_config", {})
    config = SemanticVLMConfig(sample_fps=metadata.get("sample_fps", .5),
        image_size=metadata.get("image_size", 448), clip_seconds=context.get("clip_seconds", 4.),
        frames_per_window=context.get("frames_per_window", 4),
        max_new_tokens=context.get("max_new_tokens", 100), batch_size=args.batch_size,
        device=args.device, allow_download=False,
        local_checkpoint=str(args.local_checkpoint.resolve()) if args.local_checkpoint else None)
    if artifact is not None and perception_protocol(config) != context:
        raise ValueError("Saved observation protocol differs from this fixed local implementation")
    return config


def _gpu_measurements(torch, runtime):
    if runtime.type != "cuda":
        return {"gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None}
    torch.cuda.synchronize(runtime)
    return {"gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(runtime)),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(runtime))}


def _compare_caches(fresh, reference):
    """Aggregate numeric drift only; never return state arrays or event text."""
    import numpy as np
    from datathon_ml.model import semantic_feature_config
    if fresh["sha256"] != reference["sha256"]:
        raise ValueError("Reference cache belongs to different source video bytes")
    if (fresh["features"].shape != reference["features"].shape
            or not np.array_equal(fresh["timestamps"], reference["timestamps"])):
        raise ValueError("Reference cache has different window times or state geometry")
    if (semantic_feature_config(fresh["config"]) != semantic_feature_config(reference["config"])
            or fresh["config"].get("protocol_sha256") != reference["config"].get("protocol_sha256")
            or fresh["config"].get("concept_names") != reference["config"].get("concept_names")):
        raise ValueError("Reference cache has a different observation protocol or checkpoint")
    same = fresh["features"] == reference["features"]
    difference = np.abs(fresh["features"] - reference["features"])
    return {"fraction_identical_states": float(np.mean(same)),
        "fraction_identical_windows": float(np.mean(same.reshape(len(same), -1).all(axis=1))),
        "max_absolute_state_difference": float(np.max(difference)),
        "mean_absolute_state_difference": float(np.mean(difference)),
        "reference_successful_batch_sizes": reference["provenance"].get("successful_batch_sizes", [1]),
        "comparison_is_accuracy_evidence": False}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--video", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--model", type=Path, help="Optional trusted semantic temporal/head joblib artifact")
    result.add_argument("--batch-size", type=int, default=2)
    result.add_argument("--device", default="auto")
    result.add_argument("--local-checkpoint", type=Path)
    result.add_argument("--reference-cache", type=Path,
        help="Optional existing sequential cache of the same clip; compare only aggregate numeric drift")
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--budget-seconds", type=float, default=300.,
        help="Measured latency target; enforce a hard timeout in the parent process")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.threads < 1 or not 0 < args.budget_seconds < float("inf"):
        raise ValueError("Need positive threads and a finite positive latency budget")
    output = args.output_dir.expanduser().resolve()
    feature_dir = output / "features"
    if feature_dir.exists() and next(feature_dir.rglob("*.npz"), None) is not None:
        raise FileExistsError("Fresh-cache benchmark refuses an output feature directory containing NPZ files")
    source = args.video.expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("--video must be a local file")
    if args.reference_cache is not None:
        args.reference_cache = args.reference_cache.expanduser().resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    configure_offline_environment()
    # Running `python tools/script.py` must work without an installed package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    report = {"status": "failed", "fresh_feature_cache": True,
        "accuracy_evaluation": False, "hosted_inference_api_used": False,
        "accuracy_of_batch2_not_established": True,
        "allow_download": False, "batch_size_requested": args.batch_size,
        "budget_seconds": args.budget_seconds,
        "numerical_warning": "Batch 1 and batch 2 can produce different floating-point observations and predictions. Compare fresh-cache scores separately; this runtime test is not accuracy evidence.",
        "network_guard": "Python socket connections, DNS and datagrams blocked; not an OS firewall"}
    network = {"attempts": 0, "by_method": {}}
    code = 1
    with (output / "process.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log), blocked_network() as network:
            try:
                import torch
                from datathon_ml.features import load_feature_cache, sha256_file
                from datathon_ml.semantic_vlm import _runtime_device, extract_semantic_states
                torch.set_num_threads(args.threads)
                artifact = _trusted_model(args.model.resolve()) if args.model else None
                config = _config(args, artifact)
                runtime = _runtime_device(config)
                if runtime.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(runtime)
                report.update(device=str(runtime), torch_version=str(torch.__version__),
                    model_kind=artifact["kind"] if artifact is not None else None)
                digest = sha256_file(source)
                started = time.perf_counter()
                # No label, video ID, annotation anchor or duration is passed.
                paths = extract_semantic_states([{"path": str(source), "sha256": digest}],
                    feature_dir, config)
                report.update(_gpu_measurements(torch, runtime))
                report["feature_extraction_seconds"] = time.perf_counter() - started
                cache = load_feature_cache(paths[digest])
                provenance = cache["provenance"]
                report.update(video_sha256=digest, feature_file=str(paths[digest]),
                    feature_cache_reused=False,
                    schema_valid_windows=provenance["schema_valid_windows"],
                    total_windows=provenance["total_windows"],
                    batch_size_effective=provenance["batch_size_effective"],
                    successful_batch_sizes=provenance["successful_batch_sizes"],
                    mixed_generation_batch_sizes=provenance["mixed_generation_batch_sizes"],
                    protocol_sha256=cache["config"]["protocol_sha256"],
                    checkpoint_sha256=cache["config"]["weights_sha256"])
                if args.reference_cache is not None:
                    reference_cache = load_feature_cache(args.reference_cache)
                    report["batch_comparison"] = _compare_caches(cache, reference_cache)
                if artifact is not None:
                    started = time.perf_counter()
                    if artifact["kind"] == "semantic_temporal":
                        from datathon_ml.temporal import predict_temporal_features as predict
                    else:
                        from datathon_ml.semantic_heads import predict_semantic_head_features as predict
                    prediction = predict(paths[digest], args.model.resolve())
                    (output / "prediction.json").write_text(json.dumps(prediction, indent=2, allow_nan=False), encoding="utf-8")
                    report["prediction_seconds"] = time.perf_counter() - started
                    report["prediction_file"] = str(output / "prediction.json")
                    if args.reference_cache is not None:
                        reference_prediction = predict(args.reference_cache, args.model.resolve())
                        (output / "reference_prediction.json").write_text(
                            json.dumps(reference_prediction, indent=2, allow_nan=False), encoding="utf-8")
                        report["batch_comparison"].update(
                            prediction_agreement=prediction["prediction"] == reference_prediction["prediction"],
                            fresh_prediction_score=float(prediction["score"]),
                            reference_prediction_score=float(reference_prediction["score"]),
                            absolute_score_difference=abs(float(prediction["score"])-float(reference_prediction["score"])))
                report["status"] = "passed" if network["attempts"] == 0 else "network_attempt_rejected"
                code = 0 if network["attempts"] == 0 else 1
            except Exception as error:
                report.update(error_type=type(error).__name__, error_message=str(error))
                traceback.print_exc(file=log)
    report.update(network_attempts=network["attempts"], network_attempts_by_method=network["by_method"],
        total_fresh_process_seconds=time.perf_counter()-PROCESS_STARTED)
    report["budget_met"] = report["status"] == "passed" and report["total_fresh_process_seconds"] <= args.budget_seconds
    (output / "runtime.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    public_keys = ("status", "total_fresh_process_seconds", "feature_extraction_seconds", "prediction_seconds",
        "network_attempts", "schema_valid_windows", "total_windows", "batch_size_requested", "batch_size_effective",
        "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "budget_met", "batch_comparison",
        "accuracy_of_batch2_not_established")
    print(json.dumps({key: report[key] for key in public_keys if key in report}, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())

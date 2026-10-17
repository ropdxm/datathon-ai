"""Predict an unlabeled video using a trusted CLIP semantic model, offline by default."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import time

import joblib

from datathon_ml import temporal
from datathon_ml.clip_features import (BACKBONE as IMAGE_BACKBONE, EXPECTED_WEIGHT_SHA256,
                                      ClipFeatureConfig, extract_clip_features)
from datathon_ml.features import _canonical_json, _configuration_hash, sha256_file
from datathon_ml.model import semantic_feature_config
from datathon_ml.semantic_clip import BACKBONE, TEMPERATURE, convert_clip_cache, prompt_protocol


def _validate_artifact(artifact):
    kind = artifact.get("kind")
    if kind not in {"semantic_temporal", "semantic_head"}:
        raise ValueError("Use a trusted semantic_temporal or semantic_head artifact")
    metadata = artifact["feature_config"]
    if metadata.get("backbone") != BACKBONE:
        raise ValueError("This fresh-video entry supports CLIP semantic models only; other backbones are unsupported")
    provenance = artifact.get("provenance", {})
    current_temporal = hashlib.sha256(Path(temporal.__file__).read_bytes()).hexdigest()
    if kind == "semantic_head":
        from datathon_ml import semantic_heads
        if provenance.get("code_sha256") != hashlib.sha256(Path(semantic_heads.__file__).read_bytes()).hexdigest():
            raise ValueError("Semantic head code fingerprint differs; use matching sources or refit")
        if provenance.get("temporal_code_sha256") != current_temporal:
            raise ValueError("Temporal feature code fingerprint differs; use matching sources or refit")
        predict = semantic_heads.predict_semantic_head_features
    else:
        saved = [provenance[key] for key in ("code_sha256", "temporal_code_sha256") if key in provenance]
        if not saved or any(value != current_temporal for value in saved):
            raise ValueError("Temporal feature code fingerprint differs; use matching sources or refit")
        predict = temporal.predict_temporal_features
    source = metadata.get("source_feature_config", {})
    if source.get("backbone") != IMAGE_BACKBONE or source.get("weights_sha256") != EXPECTED_WEIGHT_SHA256:
        raise ValueError("Artifact does not specify the verified original CLIP source representation")
    protocol_hash = _configuration_hash(_canonical_json(prompt_protocol()))
    preprocessing = f"semantic-clip18-fixed-v1;protocol={protocol_hash};source={_configuration_hash(_canonical_json(source))}"
    if (metadata.get("prompts_sha256") != protocol_hash or metadata.get("preprocessing") != preprocessing
            or metadata.get("temperature") != TEMPERATURE or metadata.get("weights_sha256") != EXPECTED_WEIGHT_SHA256):
        raise ValueError("Saved semantic CLIP prompt/projection protocol differs from current sources")
    if semantic_feature_config(metadata) != artifact.get("feature_semantic_config"):
        raise ValueError("Artifact semantic protocol fields are inconsistent")
    if (metadata.get("continuous_window_coverage") is True or artifact.get("continuous_window_coverage") is True
            or metadata.get("coverage_certification") != artifact.get("coverage_certification")):
        raise ValueError("Sampled CLIP frames cannot supply certified continuous action coverage")
    return source, predict


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--video", type=Path, required=True)
    result.add_argument("--model", type=Path, required=True, help="Trusted exported CLIP semantic temporal/head artifact.")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="auto")
    result.add_argument("--batch-size", type=int)
    result.add_argument("--allow-download", action="store_true", help="Opt in to downloading public weights; inference remains in-process.")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.perf_counter()
    model_path, video = args.model.resolve(strict=True), args.video.resolve(strict=True)
    artifact = joblib.load(model_path)  # Trusted files only.
    source, predict = _validate_artifact(artifact)  # Reject before expensive vision execution.
    config = ClipFeatureConfig(backbone=source["backbone"], sample_fps=source["sample_fps"],
        image_size=source["image_size"], views=tuple(source["views"]),
        batch_size=args.batch_size if args.batch_size is not None else 32,
        device=args.device, allow_download=args.allow_download)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(video)
    # There are no label/workbook/identifier/annotation inputs in this path.
    with redirect_stdout(io.StringIO()):
        extract_clip_features([{"path": str(video), "sha256": digest}], output / "image_features", config)
        paths = convert_clip_cache(output / "image_features", output / "semantic_features",
                                   device=args.device, offline=not args.allow_download)
        prediction = predict(paths[digest], model_path)
    prediction["elapsed_seconds"] = time.perf_counter() - started
    prediction["offline_requested"] = not args.allow_download
    (output / "prediction.json").write_text(json.dumps(prediction, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"verdict": prediction["verdict"], "score": prediction["score"],
        "event_count": len(prediction.get("events", [])), "elapsed_seconds": prediction["elapsed_seconds"]}), flush=True)
    return prediction


if __name__ == "__main__":
    main()

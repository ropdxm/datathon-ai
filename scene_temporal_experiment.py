"""Audit verified targets or train/predict local point-guided scene temporal heads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datathon_ml.scene_temporal import (SceneTemporalConfig, apply_verified_groups,
    fit_scene_temporal_final, original_targets, predict_scene_temporal_features,
    run_scene_temporal_cv, supervision_audit)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("audit", "cv", "fit", "predict"))
    result.add_argument("--manifest", type=Path)
    result.add_argument("--feature-dir", type=Path)
    result.add_argument("--output-dir", type=Path, default=Path("artifacts/scene_temporal"))
    result.add_argument("--folds-from", type=Path, help="Exact original source-SHA outer assignment; also verifies original targets")
    result.add_argument("--originals-from", type=Path, help="Historical report binding original source hashes to original labels")
    result.add_argument("--groups-json", type=Path, help="Explicit verified source-SHA-to-scene metadata only")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--outer-folds", type=int, default=5)
    result.add_argument("--inner-folds", type=int, default=3)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--epochs", type=int, nargs=2, default=(80, 120), metavar=("SMALL", "LARGE"))
    result.add_argument("--feature", type=Path)
    result.add_argument("--model", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "predict":
        if args.feature is None or args.model is None:
            raise ValueError("predict requires --feature and --model")
        result = predict_scene_temporal_features(args.feature, args.model)
        (args.output_dir / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"prediction": result["prediction"], "score": result["score"],
                          "event_count": len(result["events"])}), flush=True)
        return result
    if args.manifest is None:
        raise ValueError("audit/cv/fit require --manifest")
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = apply_verified_groups(records, args.groups_json)
    reference = args.originals_from if args.originals_from is not None else args.folds_from
    if args.command == "audit":
        audit = supervision_audit(records, original_targets(reference))
        # Keep individual numeric decisions in the artifact, print counts only.
        (args.output_dir / "scene_supervision_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in audit.items() if key not in {"accepted", "excluded"}}), flush=True)
        return audit
    if args.feature_dir is None:
        raise ValueError("cv/fit require --feature-dir")
    config = SceneTemporalConfig(seed=args.seed, outer_folds=args.outer_folds,
        inner_folds=args.inner_folds, device=args.device, threads=args.threads, epochs=tuple(args.epochs))
    if args.command == "cv":
        return run_scene_temporal_cv(records, args.feature_dir, args.output_dir / "evaluation", config,
            folds_from=args.folds_from, originals_from=reference)
    path = fit_scene_temporal_final(records, args.feature_dir, args.output_dir / "model", config,
        originals_from=reference)
    print("Final development model: " + str(path), flush=True)
    return path


if __name__ == "__main__":
    main()

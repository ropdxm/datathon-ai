"""Fixed semantic-head family comparison. No hosted inference or frame decoding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datathon_ml.semantic_heads import (SemanticHeadConfig, fit_semantic_head_final,
    predict_semantic_head_features, run_semantic_head_cv)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("pipeline", "cv", "fit", "predict"))
    result.add_argument("--manifest", type=Path)
    result.add_argument("--feature-dir", type=Path)
    result.add_argument("--output-dir", type=Path, default=Path("artifacts/semantic_heads"))
    result.add_argument("--folds-from", type=Path, help="Required for CV/pipeline: historical CV report with the exact outer source-SHA map.")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--outer-folds", type=int, default=5)
    result.add_argument("--inner-folds", type=int, default=3)
    result.add_argument("--no-calibration", action="store_true")
    result.add_argument("--feature", type=Path)
    result.add_argument("--model", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "predict":
        if args.feature is None or args.model is None:
            raise ValueError("predict requires --feature and --model")
        prediction = predict_semantic_head_features(args.feature, args.model)
        (args.output_dir / "prediction.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")
        print(json.dumps({"prediction": prediction["prediction"], "score": prediction["score"],
                          "rule_event_count": len(prediction["rule_events"])}), flush=True)
        return prediction
    if args.manifest is None or args.feature_dir is None:
        raise ValueError("Training commands require --manifest and --feature-dir")
    if args.command in {"cv", "pipeline"} and args.folds_from is None:
        raise ValueError("CV/pipeline require --folds-from for the fixed outer comparison")
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = SemanticHeadConfig(seed=args.seed, outer_folds=args.outer_folds, inner_folds=args.inner_folds,
                                calibrate=not args.no_calibration)
    report = None
    if args.command in {"cv", "pipeline"}:
        report = run_semantic_head_cv(records, args.feature_dir, args.output_dir / "evaluation", config,
                                      folds_from=args.folds_from)
        if args.command == "cv":
            return report
    path = fit_semantic_head_final(records, args.feature_dir, args.output_dir / "model", config)
    print("Final semantic head development model: " + str(path), flush=True)
    return {"report": report, "model": str(path)} if args.command == "pipeline" else path


if __name__ == "__main__":
    main()

"""Fit validation-selected local models, then open an explicit locked test once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datathon_ml.reviewed_models import (ReviewedConfig, fit_reviewed_selection,
    predict_reviewed_features, score_reviewed_test)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("fit", "score", "run", "predict"))
    result.add_argument("--manifest", type=Path)
    result.add_argument("--feature-dir", type=Path)
    result.add_argument("--feature-set", action="append", default=[], metavar="NAME=PATH",
                        help="Predeclare multiple representations before selection; one global winner opens test")
    result.add_argument("--split", type=Path)
    result.add_argument("--originals-from", type=Path)
    result.add_argument("--output-dir", type=Path, default=Path("artifacts/reviewed_models"))
    result.add_argument("--selection-metric", choices=("accuracy", "balanced_accuracy", "f1", "f2"), default="accuracy")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--no-catboost", action="store_true", help="Declare this exclusion before selection starts")
    result.add_argument("--feature", type=Path)
    result.add_argument("--model", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "predict":
        if args.feature is None or args.model is None:
            raise ValueError("predict requires --feature and --model")
        result = predict_reviewed_features(args.feature, args.model)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"prediction": result["prediction"], "score": result["score"], "event_count": len(result["events"])}), flush=True)
        return result
    if args.manifest is None or (args.feature_dir is None and not args.feature_set) or args.split is None:
        raise ValueError("fit/score/run require --manifest, feature directory/set(s), and --split")
    if args.feature_dir is not None and args.feature_set:
        raise ValueError("Choose either --feature-dir or explicit named --feature-set entries")
    feature_dirs = args.feature_dir
    if args.feature_set:
        feature_dirs = {}
        for entry in args.feature_set:
            if "=" not in entry:
                raise ValueError("--feature-set expects NAME=PATH")
            name, directory = entry.split("=", 1)
            if not name.strip() or not directory.strip() or name in feature_dirs:
                raise ValueError("Feature-set names and paths must be nonempty and unique")
            feature_dirs[name] = Path(directory)
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    if args.command in {"fit", "run"}:
        config = ReviewedConfig(seed=args.seed, threads=args.threads, primary_metric=args.selection_metric,
                                include_catboost=not args.no_catboost)
        model = fit_reviewed_selection(records, feature_dirs, split, args.output_dir, config, originals_from=args.originals_from)
        if args.command == "fit":
            print("Frozen validation-selected model: " + str(model), flush=True)
            return model
    else:
        model = args.model if args.model is not None else args.output_dir / "model_reviewed.joblib"
    return score_reviewed_test(records, feature_dirs, split, model, args.output_dir, originals_from=args.originals_from)


if __name__ == "__main__":
    main()

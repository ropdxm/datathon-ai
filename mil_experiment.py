"""Optional point-guided MIL objective comparison using existing frozen caches."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cv", "fit", "pipeline", "predict"))
    parser.add_argument("--manifest", type=Path, help="Validated manifest JSON; videos are not read when supplied")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--verified-aliases", type=Path)
    parser.add_argument("--groups-csv", type=Path)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mil"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--epochs", type=int, nargs=2, default=(100, 200), metavar=("LINEAR", "MLP32"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--feature-cache", type=Path, help="Unlabeled NPZ for predict")
    parser.add_argument("--model", type=Path, help="Trusted model_mil.joblib for predict")
    parser.add_argument("--prediction-output", type=Path)
    args = parser.parse_args(argv)
    from datathon_ml.mil import (MilConfig, _write_json, fit_mil_final,
                                predict_mil_features, run_mil_cross_validation)
    if args.command == "predict":
        if args.feature_cache is None or args.model is None:
            parser.error("predict requires --feature-cache and --model")
        result = predict_mil_features(args.feature_cache, args.model)
        if args.prediction_output:
            _write_json(args.prediction_output, result)
        print(json.dumps(result, indent=2), flush=True)
        return result
    if args.feature_dir is None:
        parser.error("Training/evaluation requires --feature-dir with existing frozen caches")
    if args.manifest:
        records = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("Manifest must be a JSON list of validated records")
    else:
        from datathon_ml.data import build_manifest
        aliases = json.loads(args.verified_aliases.read_text(encoding="utf-8")) if args.verified_aliases else None
        records = build_manifest(args.data_root, verified_aliases=aliases)
    if args.groups_csv:
        from experiment import apply_groups
        records = apply_groups(records, args.groups_csv)
    config = MilConfig(seed=args.seed, outer_folds=args.outer_folds, inner_folds=args.inner_folds,
                       pca_components=args.pca_components, epochs=tuple(args.epochs), threads=args.threads)
    output = args.output_dir.resolve()
    _write_json(output / "manifest.json", records)
    result = None
    if args.command in ("cv", "pipeline"):
        result = run_mil_cross_validation(records, args.feature_dir, output / "evaluation", config)
    if args.command in ("fit", "pipeline"):
        result = fit_mil_final(records, args.feature_dir, output / "model", config)
        print("Final MIL development model: " + str(result), flush=True)
    return result


if __name__ == "__main__":
    main()

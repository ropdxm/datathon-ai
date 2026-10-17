"""Compare predeclared recording-level CatBoost feature arms without APIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datathon_ml.boosted import (BoostedConfig, fit_boosted_final,
    predict_boosted_features, run_boosted_comparison)


def _read_records(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("Manifest requires a records list")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("compare", "fit"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--original-manifest", required=True, type=Path)
        sub.add_argument("--clip-features", required=True, type=Path)
        sub.add_argument("--semantic-features", required=True, type=Path)
        sub.add_argument("--output-dir", required=True, type=Path)
        sub.add_argument("--threads", type=int, default=2)
        if command == "compare":
            sub.add_argument("--folds-from", required=True, type=Path)
            sub.add_argument("--fit-final", action="store_true")
    sub = subparsers.add_parser("predict")
    sub.add_argument("--model", required=True, type=Path)
    sub.add_argument("--clip-feature", type=Path)
    sub.add_argument("--semantic-feature", type=Path)
    args = parser.parse_args(argv)
    if args.command == "predict":
        print(json.dumps(predict_boosted_features(args.model,
            clip_feature_path=args.clip_feature, semantic_feature_path=args.semantic_feature), indent=2))
        return
    records = _read_records(args.original_manifest)
    config = BoostedConfig(threads=args.threads)
    if args.command == "compare":
        report = run_boosted_comparison(records, args.clip_features, args.semantic_features,
            args.output_dir, folds_from=args.folds_from, config=config)
        print("BOOSTED_MEASURED_RESULT", json.dumps({"primary_metrics": report["metrics"],
            "arm_metrics": {arm: row["metrics"] for arm, row in report["arm_comparisons"].items()},
            "protocol_sha256": report["protocol_sha256"]}), flush=True)
    if args.command == "fit" or args.fit_final:
        path = fit_boosted_final(records, args.clip_features, args.semantic_features,
            args.output_dir / "model", config=config)
        print("BOOSTED_FITTED_MODEL", json.dumps({"path": str(path)}), flush=True)


if __name__ == "__main__":
    main()

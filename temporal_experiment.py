"""Audit semantic caches, evaluate/final-fit temporal models, or predict locally."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datathon_ml.temporal import (CONCEPT_NAMES, TemporalConfig, fit_temporal_final,
    load_states, predict_temporal_features, run_temporal_cross_validation, transform_states)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("audit", "cv", "fit", "predict"))
    result.add_argument("--data-root", type=Path, default=Path("."))
    result.add_argument("--manifest", type=Path, help="Existing canonical manifest JSON; otherwise audit source labels/videos.")
    result.add_argument("--feature-dir", type=Path)
    result.add_argument("--output-dir", type=Path, default=Path("artifacts/semantic_temporal"))
    result.add_argument("--verified-aliases", type=Path)
    result.add_argument("--groups-csv", type=Path)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--outer-folds", type=int, default=5)
    result.add_argument("--inner-folds", type=int, default=3)
    result.add_argument("--folds-from", type=Path, help="Reuse and validate the exact outer SHA-to-fold map from an existing CV report.")
    result.add_argument("--no-calibration", action="store_true")
    result.add_argument("--feature", type=Path, help="Unlabeled semantic cache for predict.")
    result.add_argument("--model", type=Path, help="Trusted model_temporal.joblib for predict.")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "predict":
        if args.feature is None or args.model is None:
            raise ValueError("predict requires --feature and --model")
        result = predict_temporal_features(args.feature, args.model)
        (args.output_dir / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        # Keep observable descriptions/private trajectories in the local artifact.
        print(json.dumps({"prediction": result["prediction"], "score": result["score"],
                          "rule_event_count": len(result["rule_events"])}), flush=True)
        return result
    if args.feature_dir is None:
        raise ValueError("audit/cv/fit require --feature-dir")
    if args.manifest:
        records = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        from datathon_ml.data import build_manifest
        aliases = json.loads(args.verified_aliases.read_text(encoding="utf-8")) if args.verified_aliases else None
        records = build_manifest(args.data_root, output_dir=args.output_dir, verified_aliases=aliases)
    if args.groups_csv:
        from experiment import apply_groups
        records = apply_groups(records, args.groups_csv)
    from datathon_ml.model import _canonical_records
    records = _canonical_records(records)
    (args.output_dir / "manifest.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    config = TemporalConfig(seed=args.seed, outer_folds=args.outer_folds, inner_folds=args.inner_folds,
                            calibrate=not args.no_calibration)
    if args.command == "audit":
        rows = []
        for record in records:
            x, t, meta = load_states(args.feature_dir / f"{record['sha256']}.npz")
            derived = transform_states(x, t, config, continuous_window_coverage=meta.get("continuous_window_coverage") is True)
            rows.append({"sha256": record["sha256"], "windows": len(t),
                         "rule_events": len(derived["events"]), "rule_score": derived["rule_score"]})
        result = {"unique_videos": len(rows), "concept_count": len(CONCEPT_NAMES),
                  "windows": sum(row["windows"] for row in rows), "records": rows}
        (args.output_dir / "semantic_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "records"}), flush=True)
        return result
    if args.command == "cv":
        return run_temporal_cross_validation(records, args.feature_dir, args.output_dir / "evaluation", config,
                                             folds_from=args.folds_from)
    path = fit_temporal_final(records, args.feature_dir, args.output_dir / "model", config)
    print("Final temporal development model: " + str(path), flush=True)
    return path


if __name__ == "__main__":
    main()

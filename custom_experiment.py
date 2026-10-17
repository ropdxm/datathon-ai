"""Frozen original-fold augmentation comparison; no audio or hosted inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datathon_ml.custom_training import CustomTrainingConfig, fit_custom_final, run_custom_comparison


def read_records(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("videos"))
    if not isinstance(payload, list):
        raise ValueError("Manifest must be a list or a dictionary containing records/videos")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compare", "fit"))
    parser.add_argument("--original-manifest", required=True, type=Path)
    parser.add_argument("--aux-manifest", required=True, type=Path)
    parser.add_argument("--original-features", required=True, type=Path)
    parser.add_argument("--aux-features", required=True, type=Path)
    parser.add_argument("--folds-from", type=Path, default=Path("reference/clip_baseline_cv.json"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--policy", choices=("semantic", "point"), default="semantic")
    parser.add_argument("--allow-unverified-aux", action="store_true", help="Explicit weak sensitivity using friend-supplied unverified labels")
    parser.add_argument("--allow-machine-aux", action="store_true", help="Explicit weak sensitivity using machine proposals; never evaluation targets")
    parser.add_argument("--aux-weight", type=float, default=.25)
    parser.add_argument("--fit-final", action="store_true", help="Also fit/export the declared policy after comparison")
    args = parser.parse_args(argv)
    config = CustomTrainingConfig(policy=args.policy, auxiliary_weight=args.aux_weight)
    original, auxiliary = read_records(args.original_manifest), read_records(args.aux_manifest)
    kwargs = dict(config=config, allow_unverified_aux=args.allow_unverified_aux,
                  allow_machine_aux=args.allow_machine_aux)
    if args.command == "compare":
        report = run_custom_comparison(original, auxiliary, args.original_features, args.aux_features,
            args.output_dir, folds_from=args.folds_from, **kwargs)
        print(json.dumps({"status": report["status"], "metrics": {k: v["metrics"] for k, v in report["policies"].items()},
                          "paired_change": report.get("paired_change")}, indent=2))
    if args.command == "fit" or args.fit_final:
        path = fit_custom_final(original, auxiliary, args.original_features, args.aux_features,
            args.output_dir / "model", **kwargs)
        print(json.dumps({"fitted_model": str(path)}))


if __name__ == "__main__":
    main()

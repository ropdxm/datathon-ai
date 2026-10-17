"""Train on original labels and adapt projection using unlabeled custom video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datathon_ml.unlabeled_adaptation import UnlabeledConfig, fit_unlabeled_final, run_unlabeled_comparison


def read_records(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rows = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Manifest needs a records list")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compare", "fit"))
    parser.add_argument("--original-manifest", required=True, type=Path)
    parser.add_argument("--unlabeled-manifest", required=True, type=Path)
    parser.add_argument("--original-features", required=True, type=Path)
    parser.add_argument("--unlabeled-features", required=True, type=Path)
    parser.add_argument("--folds-from", type=Path, default=Path("reference/clip_baseline_cv.json"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fit-final", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    originals, custom = read_records(args.original_manifest), read_records(args.unlabeled_manifest)
    config = UnlabeledConfig(threads=args.threads)
    if args.command == "compare":
        report = run_unlabeled_comparison(originals, custom, args.original_features,
            args.unlabeled_features, args.output_dir, folds_from=args.folds_from, config=config)
        print("UNLABELED_MEASURED_RESULT", json.dumps({"metrics": {name: row["metrics"] for name, row in report["policies"].items()},
            "paired_change": report["paired_change"], "custom_labels_used": 0,
            "unlabeled_custom_recordings": report["unlabeled_custom_recordings"],
            "unlabeled_custom_groups": report["unlabeled_custom_groups"]}), flush=True)
    if args.command == "fit" or args.fit_final:
        path = fit_unlabeled_final(originals, custom, args.original_features,
            args.unlabeled_features, args.output_dir / "model", config=config)
        print("UNLABELED_FITTED_MODEL", json.dumps({"path": str(path), "custom_labels_used": 0}), flush=True)


if __name__ == "__main__":
    main()

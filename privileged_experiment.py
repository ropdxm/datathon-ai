"""Train-owned privileged note experiment; video-only feature inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datathon_ml.privileged_notes import (fit_privileged_final, predict_privileged_features,
                                        run_privileged_cv)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compare", "fit", "predict"))
    parser.add_argument("--original-manifest", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--note-cache", type=Path)
    parser.add_argument("--notes-dir", type=Path, help="Private per-source JSON directory; lazy fit-owned preparation")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds-from", type=Path, default=Path("reference/clip_baseline_cv.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fit-final", action="store_true")
    parser.add_argument("--feature", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args(argv)
    if args.command == "predict":
        if not args.feature or not args.model:
            parser.error("predict requires --feature and --model")
        result = predict_privileged_features(args.feature, args.model)
        print(json.dumps(result))
        return result
    if not all((args.original_manifest, args.features, args.note_cache, args.output_dir)):
        parser.error("training requires --original-manifest, --features, --note-cache and --output-dir")
    records = json.loads(args.original_manifest.read_text(encoding="utf-8-sig"))
    if isinstance(records, dict):
        records = records["records"]
    loader = None
    if args.notes_dir:
        from tools.prepare_privileged_notes import LazyNotePreparer
        loader = LazyNotePreparer(args.notes_dir, args.note_cache, records,
                                  checkpoint=args.checkpoint, device=args.device).load
    if args.command == "compare":
        report = run_privileged_cv(records, args.features, args.note_cache, args.output_dir,
            folds_from=args.folds_from, loader=loader)
        print("PRIVILEGED_MEASURED_RESULT", json.dumps({"metrics": report["metrics"], "notes_verified": False}))
    if args.command == "fit" or args.fit_final:
        path = fit_privileged_final(records, args.features, args.note_cache, args.output_dir / "model", loader=loader)
        print("PRIVILEGED_FITTED_MODEL", json.dumps({"path": str(path), "inference_requires_notes": False}))


if __name__ == "__main__":
    main()

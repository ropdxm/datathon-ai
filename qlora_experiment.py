"""Native supervised Qwen QLoRA: synthetic probe, fit, score or label-free predict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datathon_ml.qlora_vlm import QLoRAConfig, fit_qlora, predict_qlora, probe_qlora, score_qlora


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "fit", "score", "predict"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--detail-frame-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--originals-from", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/qlora_vlm_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--source-sha256")
    parser.add_argument("--prediction-output", type=Path)
    parser.add_argument("--allow-resource-fallback", action="store_true")
    args = parser.parse_args(argv)
    config = QLoRAConfig(device=args.device, threads=args.threads)
    if args.command == "probe":
        return probe_qlora(args.checkpoint, args.output_dir, config,
                           allow_resource_fallback=args.allow_resource_fallback)
    if args.allow_resource_fallback:
        raise ValueError("Resource fallback is synthetic-probe only, before private fitting")
    if args.detail_frame_root is None:
        raise ValueError("A private native detail frame root is required")
    if args.command == "predict":
        if not args.source_sha256 or args.prediction_output is None:
            raise ValueError("Label-free prediction needs source SHA and a separate numeric output path")
        result = predict_qlora(args.detail_frame_root, args.source_sha256, args.checkpoint, args.output_dir)
        args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
        args.prediction_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"score": result["score"], "prediction": result["prediction"], "event_count": 0}))
        return result
    if args.manifest is None or args.split is None:
        raise ValueError("fit/score require manifest and frozen split")
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    if args.command == "fit":
        return fit_qlora(records, args.detail_frame_root, split, args.checkpoint, args.output_dir,
                         config, originals_from=args.originals_from)
    return score_qlora(records, args.detail_frame_root, split, args.checkpoint, args.output_dir,
                       originals_from=args.originals_from)


if __name__ == "__main__":
    main()

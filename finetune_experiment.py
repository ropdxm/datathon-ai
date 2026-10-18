"""Fine tune native CLIP vision blocks; score the already exposed development set."""
import argparse
import json
from pathlib import Path

from datathon_ml.finetune_visual import (FinetuneConfig, fit_visual_selection,
    load_finetuned, read_frame_pack, evaluate_pack, score_visual_development, _device)
from datathon_ml import scene_temporal as scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "score", "predict"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--originals-from", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/visual_finetune_v2"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "predict":
        if args.model is None or args.source_sha256 is None:
            raise ValueError("predict needs --model and --source-sha256")
        model, artifact = load_finetuned(args.model, args.checkpoint, _device(FinetuneConfig(device=args.device)))
        pack = read_frame_pack(args.frame_root, args.source_sha256)
        if pack["views"] != artifact["views"]:
            raise ValueError("Inference views differ from fitted protocol")
        result = evaluate_pack(model, pack, FinetuneConfig(**artifact["config"]), artifact["threshold"])
        scene._write(args.output_dir / "prediction.json", result)
        print(json.dumps({"score": result["score"], "prediction": result["prediction"], "event_count": len(result["events"])}))
        return result
    if args.manifest is None or args.split is None:
        raise ValueError("fit/score need --manifest and --split")
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    if args.command == "fit":
        path = fit_visual_selection(records, args.frame_root, split, args.checkpoint, args.output_dir,
            FinetuneConfig(device=args.device, threads=args.threads), originals_from=args.originals_from)
        print("Validation-selected supervised visual checkpoint: " + str(path), flush=True)
        return path
    if args.model is None:
        raise ValueError("score needs the selected --model from visual_selection.json")
    return score_visual_development(records, args.frame_root, split, args.checkpoint, args.model,
        args.output_dir, originals_from=args.originals_from)


if __name__ == "__main__":
    main()

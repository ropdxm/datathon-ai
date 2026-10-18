"""Native pretrained temporal video fine tuning and exposed development scoring."""
import argparse
import json
from pathlib import Path

from datathon_ml.temporal_video_finetune import (TemporalVideoConfig, fit_temporal_video_selection,
    score_temporal_video_development, load_finetuned_temporal, evaluate_video_pack)
from datathon_ml import finetune_visual as visual
from datathon_ml import scene_temporal as scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "score", "predict"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--originals-from", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/temporal_video_v3"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--source-sha256")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "predict":
        if args.model is None or args.source_sha256 is None:
            raise ValueError("predict needs --model and --source-sha256")
        model, artifact = load_finetuned_temporal(args.model, args.base_dir, visual._device(TemporalVideoConfig(device=args.device)))
        pack = visual.read_frame_pack(args.frame_root, args.source_sha256)
        if pack["views"] != artifact["views"]:
            raise ValueError("Inference views differ from fitted temporal protocol")
        result = evaluate_video_pack(model, pack, TemporalVideoConfig(**artifact["config"]), artifact["threshold"])
        scene._write(args.output_dir / "prediction.json", result)
        print(json.dumps({"score": result["score"], "prediction": result["prediction"], "event_count": len(result["events"])}))
        return result
    if args.manifest is None or args.split is None:
        raise ValueError("fit/score need --manifest and --split")
    records = json.loads(args.manifest.read_text())
    split = json.loads(args.split.read_text())
    if args.command == "fit":
        path = fit_temporal_video_selection(records, args.frame_root, split, args.base_dir, args.output_dir,
            TemporalVideoConfig(device=args.device, threads=args.threads), originals_from=args.originals_from)
        print("Validation-selected temporal checkpoint: " + str(path), flush=True)
        return path
    if args.model is None:
        raise ValueError("score needs the selected --model")
    return score_temporal_video_development(records, args.frame_root, split, args.base_dir,
        args.model, args.output_dir, originals_from=args.originals_from)


if __name__ == "__main__": main()

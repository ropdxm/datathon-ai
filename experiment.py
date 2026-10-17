"""Reproducible ML experiments. No web server or external inference API."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import sys
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def environment():
    import importlib.metadata
    import torch
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "opencv-python-headless", "openpyxl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "code_sha256": {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(Path(__file__).parent.glob("datathon_ml/*.py"))},
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("audit", "extract", "cv", "fit", "pipeline", "predict"))
    p.add_argument("--data-root", type=Path, default=Path("."))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/run"))
    p.add_argument("--feature-dir", type=Path)
    p.add_argument("--backbone", choices=("dinov2_vits14", "dinov2_vitb14", "dinov2_vits14_reg", "dinov2_vitb14_reg", "clip_vit_b32_openai", "x3d_s", "x3d_xs", "qwen3_vl_embedding_2b"))
    p.add_argument("--sample-fps", type=float)
    p.add_argument("--image-size", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--views", nargs="+")
    p.add_argument("--clip-seconds", type=float, help="Qwen only: temporal context length in seconds; default 8 or the trained model protocol.")
    p.add_argument("--frames-per-window", type=int, help="Qwen only: ordered frames per context; default 4 or the trained model protocol.")
    p.add_argument("--device", default="auto")
    p.add_argument("--local-repo", type=Path)
    p.add_argument("--local-checkpoint", type=Path)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pca-components", type=int, default=32)
    p.add_argument("--groups-csv", type=Path, help="Optional video_id,group_id CSV for known related filming takes.")
    p.add_argument("--verified-aliases", type=Path, help="Explicit label ID to existing video ID mapping for previously verified duplicate files.")
    p.add_argument("--video", type=Path, help="Unlabeled .mov for the predict command.")
    p.add_argument("--model", type=Path, help="Trusted model.joblib produced by fit.")
    p.add_argument("--prediction-output", type=Path)
    return p


def apply_groups(records, path):
    import csv
    if path is None:
        return records
    from datathon_ml.data import normalize_video_id
    with path.open(encoding="utf-8-sig", newline="") as handle:
        mapping = {}
        for row in csv.DictReader(handle):
            ident = normalize_video_id(row["video_id"])
            group = row["group_id"].strip()
            if ident in mapping and mapping[ident] != group:
                raise ValueError("Conflicting filming groups for " + ident)
            mapping[ident] = group
    known = set()
    for record in records:
        ids = [record["video_id"], *record.get("aliases", [])]
        selected = {mapping[normalize_video_id(name)] for name in ids if normalize_video_id(name) in mapping}
        if len(selected) > 1:
            raise ValueError("Duplicate aliases assigned different groups: " + str(ids))
        if selected:
            group = selected.pop()
            if not group:
                raise ValueError("Group IDs cannot be empty")
            record["group_id"] = group
        known.update(normalize_video_id(name) for name in ids)
    if set(mapping) - known:
        raise ValueError("Unknown video IDs in groups CSV: " + str(sorted(set(mapping) - known)))
    return records


def main(argv=None):
    args = parser().parse_args(argv)
    defaults = {"backbone": "dinov2_vits14", "sample_fps": 2.0, "image_size": 224,
                "views": ["full", "top", "middle", "bottom"], "batch_size": 32}
    trained = None
    if args.command == "predict" and args.model:
        import joblib
        trained = joblib.load(args.model.resolve(strict=True))
    selected_backbone = args.backbone or (trained["feature_config"]["backbone"] if trained else defaults["backbone"])
    if selected_backbone.startswith("x3d_"):
        defaults.update(sample_fps=1.0, image_size=182, views=["full", "middle"], batch_size=8)
    elif selected_backbone == "qwen3_vl_embedding_2b":
        defaults.update(sample_fps=.25, image_size=448, views=["full"], batch_size=1)
    if trained:
        defaults.update({k: v for k, v in trained["feature_config"].items() if k in defaults})
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if (args.clip_seconds is not None or args.frames_per_window is not None) and args.backbone != "qwen3_vl_embedding_2b":
        raise ValueError("--clip-seconds and --frames-per-window are supported only for the Qwen backbone")
    if args.clip_seconds is not None and (not math.isfinite(args.clip_seconds) or args.clip_seconds <= 0):
        raise ValueError("--clip-seconds must be finite and positive")
    if args.frames_per_window is not None and args.frames_per_window < 2:
        raise ValueError("--frames-per-window must be at least 2")
    if args.sample_fps <= 0 or args.batch_size < 1 or args.image_size < 14:
        raise ValueError("Sampling FPS, batch size and image size must be positive")
    if args.backbone.startswith("x3d_"):
        from datathon_ml.motion import MotionFeatureConfig as EncoderConfig, extract_motion_features as extract
    elif args.backbone == "clip_vit_b32_openai":
        from datathon_ml.clip_features import ClipFeatureConfig as EncoderConfig, extract_clip_features as extract
    elif args.backbone == "qwen3_vl_embedding_2b":
        from datathon_ml.local_vlm import LocalVLMConfig as EncoderConfig, extract_features as extract
    else:
        from datathon_ml.features import FeatureConfig as EncoderConfig, extract_features as extract
    output = args.output_dir.resolve()
    feature_dir = (args.feature_dir or output / "features").resolve()
    encoder_kwargs = dict(
        backbone=args.backbone, sample_fps=args.sample_fps, image_size=args.image_size,
        batch_size=args.batch_size, views=tuple(args.views), device=args.device,
        local_repo=str(args.local_repo.resolve()) if args.local_repo else None,
        local_checkpoint=str(args.local_checkpoint.resolve()) if args.local_checkpoint else None,
        allow_download=not args.offline,
    )
    if args.backbone == "qwen3_vl_embedding_2b":
        if args.local_repo:
            raise ValueError("Qwen uses native Transformers; pass the complete model directory with --local-checkpoint")
        encoder_kwargs.pop("local_repo")
        if trained:
            context = trained["feature_config"].get("context_config", {})
            encoder_kwargs.update({key: context[key] for key in
                                   ("clip_seconds", "frames_per_window", "instruction", "quantize_4bit")
                                   if key in context})
        # A supplied CLI setting overrides the restored training protocol. The
        # predictor still rejects incompatible semantic feature configurations.
        for key in ("clip_seconds", "frames_per_window"):
            if getattr(args, key) is not None:
                encoder_kwargs[key] = getattr(args, key)
    feature_config = EncoderConfig(**encoder_kwargs)
    if args.command == "predict":
        if args.video is None or args.model is None:
            raise ValueError("predict requires --video and --model")
        video = args.video.resolve(strict=True)
        digest = hashlib.sha256(video.read_bytes()).hexdigest()
        paths = extract([{"path": str(video), "sha256": digest}], feature_dir, feature_config)
        if trained.get("kind") == "point_guided_mil":
            from datathon_ml.mil import predict_mil_features as predict
        else:
            from datathon_ml.model import predict_features as predict
        result = predict(paths[digest], args.model.resolve(strict=True))
        if args.prediction_output:
            write_json(args.prediction_output, result)
        print(json.dumps(result, indent=2, default=str), flush=True)
        return result

    from datathon_ml.data import build_manifest
    aliases = json.loads(args.verified_aliases.read_text(encoding="utf-8")) if args.verified_aliases else None
    records = build_manifest(args.data_root, output_dir=output, verified_aliases=aliases)
    records = apply_groups(records, args.groups_csv)
    write_json(output / "manifest.json", records)
    write_json(output / "environment.json", environment())
    if args.command in ("extract", "pipeline") or not (output / "feature_config.json").exists():
        write_json(output / "feature_config.json", dataclasses.asdict(feature_config))
    print(json.dumps({"unique_videos": len(records), "violations": sum(r["label"] for r in records),
                      "normal": sum(1-r["label"] for r in records),
                      "groups": len({r["group_id"] for r in records}),
                      "output": str(output)}, indent=2), flush=True)
    if args.command == "audit":
        return records
    if args.command in ("extract", "pipeline"):
        extract(records, feature_dir, feature_config)
        if args.command == "extract":
            return records

    from datathon_ml.model import TrainingConfig, fit_final, run_cross_validation
    training_config = TrainingConfig(seed=args.seed, outer_folds=args.outer_folds,
                                     inner_folds=args.inner_folds, pca_components=args.pca_components)
    if args.command in ("cv", "pipeline"):
        result = run_cross_validation(records, feature_dir, output / "evaluation", training_config)
        if args.command == "cv":
            return result
    if args.command in ("fit", "pipeline"):
        model_path = fit_final(records, feature_dir, output / "model", training_config)
        print("Final development model: " + str(model_path), flush=True)
        return model_path


if __name__ == "__main__":
    main()

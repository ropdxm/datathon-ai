"""Supervised native 16-frame X-CLIP fine tuning with true temporal attention.

Microsoft's MIT-licensed Kinetics-pretrained video branch includes cross-frame
communication and a learned temporal-position integration transformer. Neither
its text encoder nor prompting modules are retained or executed. The already
examined mixed partition remains development data, never independent acceptance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import expit
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F

from . import finetune_visual as visual
from . import model as base
from . import reviewed_models as reviewed
from . import scene_temporal as scene


MODEL_ID = "microsoft/xclip-base-patch32-16-frames"
REVISION = "663a1e6d4b3600f9604b7b1c140bc1cf4a97d2a7"
TRANSFORMERS_VERSION = "4.57.6"
BASE_FILES = {
    "config.json": "909bbe1d1993731a8a88e1707c00af1a9d0d21bf470ea51f1ef96d4cde7edd72",
    "preprocessor_config.json": "3376c13ec67e5505734b3d5b61d51b7f753cb1896d9661171c25ad35afcb69b6",
    "model.safetensors": "f67aa8bc3f994af8685621a7fcd571a7b2495accf43455a2ee1da5e7c7038939",
}
IMAGE_MEAN = (.485, .456, .406)
IMAGE_STD = (.229, .224, .225)
MODEL_CARD = "https://huggingface.co/microsoft/xclip-base-patch32-16-frames"
LICENSE_SOURCE = "https://github.com/microsoft/VideoX/blob/master/LICENSE"


@dataclass(frozen=True)
class TemporalVideoConfig:
    seed: int = 42
    threads: int = 2
    device: str = "auto"
    clip_frames: int = 16
    frame_step_seconds: float = 1.
    evaluation_stride_seconds: float = 4.
    clips_per_bag: int = 3
    bags_per_batch: int = 2
    clip_batch_size: int = 2
    top_k: int = 1
    head_learning_rate: float = .001
    weight_decay: float = .05
    primary_metric: str = "accuracy"

    def __post_init__(self):
        if self.clip_frames != 16 or self.frame_step_seconds != 1.:
            raise ValueError("This pinned protocol uses exactly sixteen frames at one-second spacing")
        if min(self.threads, self.clips_per_bag, self.bags_per_batch, self.clip_batch_size, self.top_k) < 1:
            raise ValueError("Thread, sampling, batch and pooling limits must be positive")
        if self.device not in {"auto", "cpu", "cuda"} or self.primary_metric not in {"accuracy", "balanced_accuracy", "f1", "f2"}:
            raise ValueError("Invalid device or metric")
        if self.evaluation_stride_seconds <= 0 or self.head_learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid stride/optimizer settings")


def candidate_grid():
    return [{"name": "xclip16_last1_lr1e5_epoch8", "last_blocks": 1, "learning_rate": .00001, "epochs": 8},
            {"name": "xclip16_last2_lr3e5_epoch12", "last_blocks": 2, "learning_rate": .00003, "epochs": 12}]


def _source_hashes():
    return {"temporal_video_finetune": reviewed._hash_file(__file__),
        "finetune_visual": reviewed._hash_file(visual.__file__), "reviewed_models": reviewed._hash_file(reviewed.__file__),
        "scene_temporal": reviewed._hash_file(scene.__file__), "model": reviewed._hash_file(base.__file__)}


def temporal_indices(times, center, config):
    """Nearest observed frames in temporal order; boundaries repeat edge frames."""
    targets = center + (np.arange(config.clip_frames) - (config.clip_frames - 1) / 2) * config.frame_step_seconds
    right = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right).astype(int)


def evaluation_centers(times, config):
    centers = np.arange(float(times[0]), float(times[-1]) + 1e-7, config.evaluation_stride_seconds)
    if len(centers) == 0 or times[-1] - centers[-1] > 1e-6:
        centers = np.r_[centers, float(times[-1])]
    return centers


def training_centers(times, row, config, rng):
    anchors = [float(point) for point in row["event_times"] if times[0] - 1 <= point <= times[-1] + 1] if row["label"] == 1 else []
    anchors = list(dict.fromkeys(anchors))
    limit = max(1, config.clips_per_bag // 2)
    if len(anchors) > limit:
        anchors = rng.choice(anchors, limit, replace=False).astype(float).tolist()
    remaining = config.clips_per_bag - len(anchors)
    random = rng.choice(times, size=remaining, replace=len(times) < remaining).astype(float).tolist()
    centers = np.asarray(anchors + random)
    masks = np.r_[np.ones(len(anchors), dtype=bool), np.zeros(len(random), dtype=bool)]
    order = np.argsort(centers, kind="stable")
    return centers[order], masks[order]


def clip_tensor(pack, center, view, config, *, training=False, rng=None):
    """One consistent augmentation per clip; independent frame jitter is forbidden."""
    from torchvision.transforms import functional as TF
    indices = temporal_indices(pack["timestamps"], center, config)
    if training:
        if rng is None:
            raise ValueError("Training augmentation requires its seeded random generator")
        brightness, contrast, saturation = rng.uniform(.85, 1.15), rng.uniform(.85, 1.15), rng.uniform(.9, 1.1)
        shift = [int(rng.integers(-4, 5)), int(rng.integers(-4, 5))]
    output = []
    for index in indices:
        with Image.open(pack["paths"][int(index)][view]) as source:
            image = source.convert("RGB")
            if image.size != (224, 224):
                raise ValueError("Temporal frame pack images must be RGB224")
            if training:
                image = TF.affine(image, angle=0., translate=shift, scale=1., shear=[0., 0.], fill=[124, 116, 104])
                image = TF.adjust_brightness(image, brightness)
                image = TF.adjust_contrast(image, contrast)
                image = TF.adjust_saturation(image, saturation)
            output.append(TF.normalize(TF.to_tensor(image), IMAGE_MEAN, IMAGE_STD))
    return torch.stack(output)


class VideoOnly(nn.Module):
    def __init__(self, vision_model, visual_projection, mit):
        super().__init__()
        self.vision_model, self.visual_projection, self.mit = vision_model, visual_projection, mit

    def forward(self, pixels):
        batch, frames, channels, height, width = pixels.shape
        if frames != 16:
            raise ValueError("The pretrained temporal-position table requires sixteen frames")
        pooled = self.vision_model(pixel_values=pixels.reshape(batch * frames, channels, height, width),
                                   return_dict=True).pooler_output
        embeddings = self.visual_projection(pooled).reshape(batch, frames, -1)
        return self.mit(embeddings, return_dict=True).pooler_output


def load_video_only(base_dir):
    base_dir = Path(base_dir)
    if version("transformers") != TRANSFORMERS_VERSION:
        raise ValueError("Install pinned transformers==" + TRANSFORMERS_VERSION)
    for name, digest in BASE_FILES.items():
        if not (base_dir / name).is_file() or reviewed._hash_file(base_dir / name) != digest:
            raise ValueError("Local X-CLIP base file differs from the pinned official revision")
    from transformers import XCLIPModel
    complete = XCLIPModel.from_pretrained(str(base_dir), local_files_only=True, use_safetensors=True,
                                         attn_implementation="eager")
    if complete.config.vision_config.num_frames != 16:
        raise ValueError("Unexpected temporal pretrained architecture")
    result = VideoOnly(complete.vision_model, complete.visual_projection, complete.mit)
    del complete  # No text encoder, token embeddings or prompting modules survive.
    return result


def configure_trainable_video(backbone, last_blocks):
    backbone.requires_grad_(False)
    layers = backbone.vision_model.encoder.layers
    if not 1 <= last_blocks < len(layers):
        raise ValueError("At least one vision prefix block must stay frozen")
    for layer in layers[-last_blocks:]:
        layer.requires_grad_(True)
    backbone.vision_model.post_layernorm.requires_grad_(True)
    backbone.visual_projection.requires_grad_(True)
    backbone.mit.requires_grad_(True)
    return sorted(name for name, parameter in backbone.named_parameters() if parameter.requires_grad)


class TemporalVideoMil(nn.Module):
    def __init__(self, backbone, width=512):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(.25), nn.Linear(width, 1))

    def forward(self, pixels):
        embeddings = F.normalize(self.backbone(pixels).float(), dim=-1)
        return self.head(embeddings).reshape(-1)


def _train(train, packs, base_dir, candidate, config):
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = visual._device(config)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    backbone = load_video_only(base_dir)
    trainable = configure_trainable_video(backbone, candidate["last_blocks"])
    model = TemporalVideoMil(backbone).to(device)
    optimizer = torch.optim.AdamW([{"params": [p for p in backbone.parameters() if p.requires_grad], "lr": candidate["learning_rate"]},
        {"params": model.head.parameters(), "lr": config.head_learning_rate}], weight_decay=config.weight_decay)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    for epoch in range(candidate["epochs"]):
        # Interleave independently shuffled classes so two-source steps include
        # normal hard negatives and positive bags without changing source counts.
        classes = [rng.permutation([i for i, row in enumerate(train) if row["label"] == label]).tolist() for label in (0, 1)]
        order = [classes[label][i] for i in range(max(map(len, classes))) for label in (0, 1) if i < len(classes[label])]
        losses = []
        model.train()
        for offset in range(0, len(order), config.bags_per_batch):
            batch = [train[i] for i in order[offset:offset + config.bags_per_batch]]
            descriptors, lengths, masks = [], [], []
            for row in batch:
                pack = packs[row["sha256"]]
                centers, mask = training_centers(pack["timestamps"], row, config, rng)
                for center, anchor in zip(centers, mask):
                    view = 0 if anchor else int(rng.integers(len(pack["views"])))
                    descriptors.append((pack, float(center), view))
                lengths.append(len(centers))
                masks.append(torch.from_numpy(mask).to(device))
            optimizer.zero_grad(set_to_none=True)
            chunks = []
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                for start in range(0, len(descriptors), config.clip_batch_size):
                    tensor = torch.stack([clip_tensor(pack, center, view, config, training=True, rng=rng)
                        for pack, center, view in descriptors[start:start + config.clip_batch_size]]).to(device)
                    chunks.append(model(tensor))
                logits = list(torch.cat(chunks).split(lengths))
                loss = visual.point_bag_loss(logits, [row["label"] for row in batch], masks, config.top_k)["total"]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite temporal video fine-tuning loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "training_loss": float(np.mean(losses))})
        print(f"Temporal video {candidate['name']} epoch {epoch + 1}/{candidate['epochs']} complete", flush=True)
    return model.eval(), trainable, history


def _prediction_from_logits(logits, times, threshold, config):
    score = float(expit(np.sort(logits)[-min(config.top_k, len(logits)):].mean()))
    predicted = int(score >= threshold)
    windows = expit(logits)
    events = base._events(windows, times, threshold, base.TrainingConfig(smoothing_seconds=0.)) if predicted else []
    return {"score": score, "prediction": predicted, "threshold": float(threshold),
        "verdict": "violation" if predicted else "no_violation", "localization_available": True,
        "score_calibrated": False, "events": [{"type": "violation", "timestamp_sec": event["timestamp_sec"],
            "timecode": event["timecode"], "score": event["score"]} for event in events],
        "window_scores": windows.tolist(), "window_timestamps": times.tolist(),
        "clip_span_seconds": 15., "timestamp_interpretation": "center_of_sixteen_observed_one_second_frames_with_edge_repetition"}


def evaluate_video_pack(model, pack, config, threshold=.5):
    """Annotation-free sliding centers and view averaging; no labels accepted."""
    centers = evaluation_centers(pack["timestamps"], config)
    descriptors = [(float(center), view) for center in centers for view in range(len(pack["views"]))]
    device = next(model.parameters()).device
    output = []
    with torch.inference_mode():
        for start in range(0, len(descriptors), config.clip_batch_size):
            tensor = torch.stack([clip_tensor(pack, center, view, config) for center, view in descriptors[start:start + config.clip_batch_size]]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(tensor)
            output.extend(logits.float().cpu().tolist())
    logits = np.asarray(output).reshape(len(centers), len(pack["views"])).mean(axis=1)
    return _prediction_from_logits(logits, centers, threshold, config)


def _evaluate(model, rows, packs, config, threshold, originals):
    return [{"sha256": row["sha256"], "group_id": row["group_id"], "label": row["label"],
        "event_times": row["event_times"], "domain": "original" if row["sha256"] in originals else "custom",
        **evaluate_video_pack(model, packs[row["sha256"]], config, threshold)} for row in rows]


LIMITATIONS = [*visual.LIMITATIONS,
    "The temporal checkpoint was trained on Kinetics400, not banking procedure labels; no external raw videos were added.",
    "One-fps sixteen-frame clips cover fifteen elapsed seconds and cannot recover motion lost between samples.",
    "Clip-center predictions provide coarse point diagnostics, not independently verified action boundaries."]


def fit_temporal_video_selection(records, frame_root, split, base_dir, output_dir, config=None, *, originals_from=None):
    config = config or TemporalVideoConfig()
    originals = scene.original_targets(originals_from)
    records = reviewed._verified_records(records, originals, split)
    partitions, frozen = reviewed.validate_split(records, split)
    train, validation = partitions["train"], partitions["validation"]
    packs = {row["sha256"]: visual.read_frame_pack(frame_root, row["sha256"]) for row in train + validation}
    if len({tuple(pack["views"]) for pack in packs.values()}) != 1:
        raise ValueError("Frame pack view orders differ")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = []
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with threadpool_limits(limits=config.threads):
            for candidate in candidate_grid():
                model, trainable, history = _train(train, packs, base_dir, candidate, config)
                rows = _evaluate(model, validation, packs, config, .5, originals)
                threshold, _ = base._threshold(np.asarray([row["label"] for row in rows]), np.asarray([row["score"] for row in rows]),
                    base.TrainingConfig(primary_metric=config.primary_metric))
                for row in rows:
                    # Preserve the exact score used to select the threshold;
                    # probability/logit round trips can change boundary ties.
                    row["prediction"] = int(row["score"] >= threshold)
                    row["threshold"] = float(threshold)
                    row["verdict"] = "violation" if row["prediction"] else "no_violation"
                    events = base._events(np.asarray(row["window_scores"]), np.asarray(row["window_timestamps"]), threshold,
                        base.TrainingConfig(smoothing_seconds=0.)) if row["prediction"] else []
                    row["events"] = [{"type": "violation", "timestamp_sec": event["timestamp_sec"],
                        "timecode": event["timecode"], "score": event["score"]} for event in events]
                metrics = reviewed._metrics(rows)
                artifact = {"format_version": 1, "kind": "supervised_temporal_xclip_mil", "candidate": candidate,
                    "config": asdict(config), "threshold": float(threshold), "source_sha256": _source_hashes(),
                    "model_id": MODEL_ID, "revision": REVISION, "base_files_sha256": BASE_FILES,
                    "license": "MIT", "model_card": MODEL_CARD, "license_source": LICENSE_SOURCE,
                    "preprocessing": {"image_mean": list(IMAGE_MEAN), "image_std": list(IMAGE_STD),
                        "geometry": "existing_RGB224_letterbox_full_and_middle_adaptation_not_publisher_center_crop",
                        "frame_spacing_seconds": 1., "clip_frames": 16, "evaluation_center_stride_seconds": 4.},
                    "transformers_version": TRANSFORMERS_VERSION, "split": frozen,
                    "training_sha256": sorted(row["sha256"] for row in train), "trainable_backbone_parameters": trainable,
                    "supervision_sha256": reviewed._supervision_hash(records),
                    "frame_pack_metadata_sha256": {key: pack["metadata_sha256"] for key, pack in packs.items()},
                    "frame_pack_integrity_sha256": {key: pack["integrity_sha256"] for key, pack in packs.items()},
                    "views": next(iter(packs.values()))["views"],
                    "backbone_delta_state": {name: parameter.detach().cpu() for name, parameter in model.backbone.named_parameters() if name in trainable},
                    "head_state": {name: value.detach().cpu() for name, value in model.head.state_dict().items()},
                    "validation_metrics": metrics, "training_history": history, "limitations": LIMITATIONS,
                    "external_inference_api_used": False, "audio_used": False, "text_encoder_used": False}
                path = output_dir / (candidate["name"] + ".pt")
                torch.save(artifact, path)
                leaderboard.append({"candidate": candidate, "threshold": float(threshold), "validation_metrics": metrics,
                    "artifact": path.name, "artifact_sha256": reviewed._hash_file(path), "training_history": history,
                    "validation_predictions": rows, "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad)})
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    finally:
        torch.set_num_threads(old_threads)
    leaderboard.sort(key=lambda row: (row["validation_metrics"][config.primary_metric], row["validation_metrics"]["balanced_accuracy"],
        row["validation_metrics"]["recall"]), reverse=True)
    selection = {"evaluation": "native_temporal_xclip_train_only_fit_validation_selection", "candidates": candidate_grid(),
        "config": asdict(config), "winner": leaderboard[0], "leaderboard": leaderboard, "split": frozen,
        "source_sha256": _source_hashes(), "development_partition_opened_for_selection": False,
        "external_inference_api_used": False, "audio_used": False, "text_encoder_used": False, "limitations": LIMITATIONS}
    scene._write(output_dir / "temporal_video_selection.json", selection)
    scene._write(output_dir / "temporal_video_selection_lock.json", {"winner_artifact": leaderboard[0]["artifact"],
        "winner_sha256": leaderboard[0]["artifact_sha256"], "selection_sha256": reviewed._hash_file(output_dir / "temporal_video_selection.json"),
        "split_sha256": frozen["sha256"], "source_sha256": _source_hashes()})
    return output_dir / leaderboard[0]["artifact"]


def load_finetuned_temporal(model_path, base_dir, device):
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    if artifact.get("kind") != "supervised_temporal_xclip_mil" or artifact.get("source_sha256") != _source_hashes():
        raise ValueError("Use a trusted temporal checkpoint and exact fitted sources")
    backbone = load_video_only(base_dir)
    expected = configure_trainable_video(backbone, artifact["candidate"]["last_blocks"])
    if sorted(artifact["backbone_delta_state"]) != expected:
        raise ValueError("Temporal delta parameter schema differs from its declared architecture")
    backbone.load_state_dict(artifact["backbone_delta_state"], strict=False)
    model = TemporalVideoMil(backbone)
    model.head.load_state_dict(artifact["head_state"])
    return model.to(device).eval(), artifact


def score_temporal_video_development(records, frame_root, split, base_dir, model_path, output_dir, *, originals_from=None):
    lock = json.loads((Path(output_dir) / "temporal_video_selection_lock.json").read_text())
    if (Path(model_path).name != lock["winner_artifact"] or reviewed._hash_file(model_path) != lock["winner_sha256"]
            or reviewed._hash_file(Path(output_dir) / "temporal_video_selection.json") != lock["selection_sha256"]):
        raise ValueError("Only the frozen validation winner can score development")
    artifact_preview = torch.load(model_path, map_location="cpu", weights_only=True)
    config = TemporalVideoConfig(**artifact_preview["config"])
    originals = scene.original_targets(originals_from)
    records = reviewed._verified_records(records, originals, split)
    partitions, frozen = reviewed.validate_split(records, split)
    if frozen["sha256"] != artifact_preview["split"]["sha256"] or reviewed._supervision_hash(records) != artifact_preview["supervision_sha256"]:
        raise ValueError("Supervised dataset/split changed after training")
    packs = {row["sha256"]: visual.read_frame_pack(frame_root, row["sha256"]) for row in partitions["test"]}
    model, artifact = load_finetuned_temporal(model_path, base_dir, visual._device(config))
    if any(pack["views"] != artifact["views"] for pack in packs.values()):
        raise ValueError("Development views differ from the fitted temporal protocol")
    with threadpool_limits(limits=config.threads):
        rows = _evaluate(model, partitions["test"], packs, config, artifact["threshold"], originals)
    report = {"evaluation": "previously_exposed_mixed_development_after_true_temporal_xclip_finetuning", "independent_test": False,
        "metrics": reviewed._metrics(rows), "domain_metrics": {domain: reviewed._metrics([row for row in rows if row["domain"] == domain])
            for domain in sorted({row["domain"] for row in rows})}, "predictions": rows, "candidate": artifact["candidate"],
        "threshold": artifact["threshold"], "model_sha256": reviewed._hash_file(model_path), "split": frozen,
        "model_card": MODEL_CARD, "license": "MIT", "limitations": LIMITATIONS,
        "external_inference_api_used": False, "audio_used": False, "text_encoder_used": False}
    scene._write(Path(output_dir) / "temporal_video_development_report.json", report)
    print(json.dumps({"evaluation": report["evaluation"], "metrics": report["metrics"], "domain_metrics": report["domain_metrics"]}), flush=True)
    return report

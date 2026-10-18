"""Native supervised CLIP visual fine-tuning on verified local frame packs.

Two declared fits update the final vision blocks, rather than another frozen
feature classifier. Their selection uses train/validation only. The previously
opened mixed test partition is explicitly development data in this experiment.
No video descriptions, hosted inference APIs, text encoder or audio are used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import expit
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F

from . import clip_features as clip
from . import model as base
from . import reviewed_models as reviewed
from . import scene_temporal as scene


@dataclass(frozen=True)
class FinetuneConfig:
    seed: int = 42
    threads: int = 2
    device: str = "auto"
    bags_per_batch: int = 4
    train_frames_per_bag: int = 8
    evaluation_frames_per_bag: int = 0  # Zero retains every packed 1-fps time.
    image_batch_size: int = 32
    top_k: int = 3
    positive_radius: float = 1.
    head_learning_rate: float = .001
    weight_decay: float = .05
    primary_metric: str = "accuracy"

    def __post_init__(self):
        if min(self.threads, self.bags_per_batch, self.train_frames_per_bag,
               self.image_batch_size, self.top_k) < 1 or self.evaluation_frames_per_bag < 0:
            raise ValueError("Sampling, batch, thread and pooling limits must be positive")
        if self.device not in {"auto", "cpu", "cuda"} or self.primary_metric not in {"accuracy", "balanced_accuracy", "f1", "f2"}:
            raise ValueError("Invalid device or selection metric")
        if min(self.positive_radius, self.weight_decay) < 0 or self.head_learning_rate <= 0:
            raise ValueError("Invalid loss/optimizer value")


def candidate_grid():
    return [{"name": "clip_last1_lr1e5_epoch8", "last_blocks": 1, "learning_rate": .00001, "epochs": 8},
            {"name": "clip_last2_lr3e5_epoch12", "last_blocks": 2, "learning_rate": .00003, "epochs": 12}]


def _source_hashes():
    return {"finetune_visual": reviewed._hash_file(__file__), "clip_features": reviewed._hash_file(clip.__file__),
        "reviewed_models": reviewed._hash_file(reviewed.__file__), "scene_temporal": reviewed._hash_file(scene.__file__),
        "model": reviewed._hash_file(base.__file__)}


def read_frame_pack(frame_root, source_sha256):
    directory = (Path(frame_root) / source_sha256).resolve()
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    times = np.asarray(metadata.get("timestamps", []), dtype=float)
    images, views = metadata.get("images"), metadata.get("views")
    if metadata.get("source_sha256") != source_sha256 or metadata.get("preprocessing") != "raw_rgb_uint8_224_letterbox":
        raise ValueError("Frame pack source/preprocessing differs from the declared protocol")
    if (not len(times) or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0)
            or not isinstance(views, list) or not views or views[0] != "full"
            or len(set(views)) != len(views) or not isinstance(images, list) or len(images) != len(times)):
        raise ValueError("Frame pack needs finite increasing elapsed times and complete full-first view rows")
    paths = []
    for row in images:
        if not isinstance(row, list) or len(row) != len(views):
            raise ValueError("Frame image rows must align with the fixed views")
        resolved = []
        for name in row:
            if not isinstance(name, str):
                raise ValueError("Frame paths must be local relative strings")
            path = (directory / name).resolve()
            if not path.is_relative_to(directory) or not path.is_file():
                raise ValueError("Frame path escapes its source directory or is missing")
            resolved.append(path)
        paths.append(resolved)
    receipt_path = directory / "_integrity.json"
    if not receipt_path.is_file():
        raise ValueError("Frame pack requires its native per-source integrity receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    hashes = receipt.get("members_sha256", {})
    members = [metadata_path, *[path for row in paths for path in row]]
    if (receipt.get("completed") is not True or not isinstance(hashes, dict)
            or any(hashes.get(path.relative_to(directory).as_posix()) != reviewed._hash_file(path) for path in members)):
        raise ValueError("Frame metadata/image bytes differ from their native integrity receipt")
    return {"timestamps": times, "paths": paths, "views": views,
            "metadata_sha256": reviewed._hash_file(metadata_path), "integrity_sha256": reviewed._hash_file(receipt_path),
            "source_sha256": source_sha256}


def uniform_indices(length, limit):
    return np.unique(np.linspace(0, length - 1, min(length, limit)).astype(int))


def training_indices(times, row, config, rng):
    """Training alone can consult positive point anchors; never infer negatives."""
    anchors = []
    if row["label"] == 1:
        for point in row["event_times"]:
            distance = np.abs(times - point)
            index = int(distance.argmin())
            if distance[index] <= max(config.positive_radius, 1.):
                anchors.append(index)
    anchors = list(dict.fromkeys(anchors))[:max(1, config.train_frames_per_bag // 2)]
    available = np.setdiff1d(np.arange(len(times)), np.asarray(anchors, dtype=int))
    remaining = min(config.train_frames_per_bag - len(anchors), len(available))
    random = rng.choice(available, remaining, replace=False).tolist() if remaining else []
    indices = np.asarray(sorted(anchors + random), dtype=int)
    mask = np.zeros(len(indices), dtype=bool)
    if row["label"] == 1:
        mask = np.asarray([any(abs(times[index] - point) <= max(config.positive_radius, 1.)
                               for point in row["event_times"]) for index in indices])
    return indices, mask


def _transforms(training):
    from torchvision import transforms as T
    steps = []
    if training:
        steps += [T.RandomAffine(degrees=0, translate=(.02, .02), fill=(123, 117, 104)),
                  T.ColorJitter(brightness=.15, contrast=.15, saturation=.10)]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(clip.CLIP_MEAN, clip.CLIP_STD)])


def _image_tensor(path, transform):
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.size != (224, 224):
            raise ValueError("Frame packs must contain native RGB224 images")
        return transform(image)


def configure_trainable_visual(visual, last_blocks):
    visual.requires_grad_(False)
    blocks = visual.transformer.resblocks
    if not 1 <= last_blocks < len(blocks):
        raise ValueError("Fine tuning must retain at least one completely frozen vision block")
    for block in blocks[-last_blocks:]:
        block.requires_grad_(True)
    visual.ln_post.requires_grad_(True)
    if visual.proj is not None:
        visual.proj.requires_grad_(True)
    return sorted(name for name, parameter in visual.named_parameters() if parameter.requires_grad)


class VisualMil(nn.Module):
    def __init__(self, visual, input_dim=512):
        super().__init__()
        self.visual = visual
        self.head = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Dropout(.25), nn.Linear(64, 1))

    def forward(self, pixels):
        features = F.normalize(self.visual(pixels).float(), dim=-1)
        return self.head(features).reshape(-1)


def point_bag_loss(logits, labels, point_masks, top_k=3):
    """Top-k bag BCE plus positive anchors and selected confirmed-normal frames."""
    bags, positive, negative = [], [], []
    zero = logits[0].sum() * 0
    for values, label, mask in zip(logits, labels, point_masks):
        pooled = values.topk(min(top_k, len(values))).values.mean()
        bags.append(F.binary_cross_entropy_with_logits(pooled, pooled.new_tensor(float(label))))
        if label == 1 and bool(mask.any()):
            positive.append(F.softplus(-values[mask]).mean())
        if label == 0:
            negative.append(F.softplus(values).mean())
    balanced = torch.stack([torch.stack([loss for loss, label in zip(bags, labels) if label == cls]).mean()
                            for cls in (0, 1) if cls in labels]).mean()
    points = torch.stack(positive).mean() if positive else zero
    normals = torch.stack(negative).mean() if negative else zero
    return {"total": balanced + points + .5 * normals, "bag": balanced, "positive": points, "normal": normals}


def _device(config):
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device("cuda" if config.device == "cuda" or config.device == "auto" and torch.cuda.is_available() else "cpu")


def load_visual(checkpoint):
    if reviewed._hash_file(checkpoint) != clip.EXPECTED_WEIGHT_SHA256:
        raise ValueError("Base CLIP checkpoint differs from the official pinned SHA-256")
    if clip._open_clip_version() != clip.OPEN_CLIP_VERSION:
        raise ValueError("Install pinned open_clip_torch==3.2.0")
    encoder = clip._load_clip_encoder(Path(checkpoint), torch.device("cpu"))
    visual = encoder.visual
    del encoder
    return visual


def _train_candidate(train, packs, checkpoint, candidate, config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = _device(config)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    visual = load_visual(checkpoint)
    trainable = configure_trainable_visual(visual, candidate["last_blocks"])
    model = VisualMil(visual).to(device)
    optimizer = torch.optim.AdamW([{"params": [p for p in model.visual.parameters() if p.requires_grad],
        "lr": candidate["learning_rate"]}, {"params": model.head.parameters(), "lr": config.head_learning_rate}],
        weight_decay=config.weight_decay)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    transform = _transforms(True)
    history = []
    for epoch in range(candidate["epochs"]):
        order = rng.permutation(len(train))
        model.train()
        losses = []
        for offset in range(0, len(train), config.bags_per_batch):
            batch = [train[int(i)] for i in order[offset:offset + config.bags_per_batch]]
            pixels, lengths, masks = [], [], []
            for row in batch:
                pack = packs[row["sha256"]]
                indices, point_mask = training_indices(pack["timestamps"], row, config, rng)
                for index, is_anchor in zip(indices, point_mask):
                    # A positive anchor is always shown in the complete view;
                    # a cropped-out event is not invented as a positive target.
                    view = 0 if is_anchor else int(rng.integers(len(pack["views"])))
                    pixels.append(_image_tensor(pack["paths"][int(index)][view], transform))
                lengths.append(len(indices))
                masks.append(torch.from_numpy(point_mask).to(device))
            tensor = torch.stack(pixels).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                all_logits = model(tensor)
                logits = list(all_logits.split(lengths))
                loss = point_bag_loss(logits, [row["label"] for row in batch], masks, config.top_k)["total"]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite visual fine-tuning loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "training_loss": float(np.mean(losses))})
        print(f"Visual fine tune {candidate['name']} epoch {epoch + 1}/{candidate['epochs']} complete", flush=True)
    return model.eval(), trainable, history


def evaluate_pack(model, pack, config, threshold=.5):
    """Uniform annotation-free view/time sampling; never accepts a source label."""
    device = next(model.parameters()).device
    indices = uniform_indices(len(pack["timestamps"]), config.evaluation_frames_per_bag or len(pack["timestamps"]))
    transform = _transforms(False)
    references = [(int(index), view) for index in indices for view in range(len(pack["views"]))]
    output = []
    with torch.inference_mode():
        for start in range(0, len(references), config.image_batch_size):
            pixels = torch.stack([_image_tensor(pack["paths"][index][view], transform)
                                  for index, view in references[start:start + config.image_batch_size]]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(pixels)
            output.extend(logits.float().cpu().tolist())
    logits = np.asarray(output, dtype=float).reshape(len(indices), len(pack["views"])).mean(axis=1)
    score = float(expit(np.sort(logits)[-min(config.top_k, len(logits)):].mean()))
    predicted = int(score >= threshold)
    times, windows = pack["timestamps"][indices], expit(logits)
    events = base._events(windows, times, threshold, base.TrainingConfig(smoothing_seconds=0.)) if predicted else []
    return {"score": score, "prediction": predicted, "threshold": float(threshold),
        "verdict": "violation" if predicted else "no_violation", "localization_available": True,
        "score_calibrated": False, "events": [{"type": "violation", "timestamp_sec": e["timestamp_sec"],
            "timecode": e["timecode"], "score": e["score"]} for e in events],
        "window_scores": windows.tolist(), "window_timestamps": times.tolist()}


def _evaluate_records(model, rows, packs, config, threshold, originals):
    return [{"sha256": row["sha256"], "group_id": row["group_id"], "label": row["label"],
        "event_times": row["event_times"], "domain": "original" if row["sha256"] in originals else "custom",
        **evaluate_pack(model, packs[row["sha256"]], config, threshold)} for row in rows]


LIMITATIONS = ["The former v1 mixed test has already been inspected; v2 scores on it are development scores, not independent acceptance.",
    "Original recordings were repeatedly inspected in prior experiments; the camera-exclusive custom subset is within one filming batch.",
    "Training points are incomplete and do not supply full event intervals; event precision/IoU remain unidentified.",
    "Fine tuning can overfit the small training and validation sets; synthetic tests establish mechanics, not model accuracy."]


def fit_visual_selection(records, frame_root, split, checkpoint, output_dir, config=None, *, originals_from=None):
    config = config or FinetuneConfig()
    originals = scene.original_targets(originals_from)
    records = reviewed._verified_records(records, originals, split)
    partitions, frozen = reviewed.validate_split(records, split)
    train, validation = partitions["train"], partitions["validation"]
    # Development/test packs remain unopened throughout candidate training.
    packs = {row["sha256"]: read_frame_pack(frame_root, row["sha256"]) for row in train + validation}
    if len({tuple(pack["views"]) for pack in packs.values()}) != 1:
        raise ValueError("All frame packs must share the fixed view order")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    try:
        with threadpool_limits(limits=config.threads):
            for candidate in candidate_grid():
                model, trainable, history = _train_candidate(train, packs, checkpoint, candidate, config)
                predictions = _evaluate_records(model, validation, packs, config, .5, originals)
                threshold, _ = base._threshold(np.asarray([row["label"] for row in predictions]),
                    np.asarray([row["score"] for row in predictions]), base.TrainingConfig(primary_metric=config.primary_metric))
                # Reconstruct thresholded events from the same deterministic
                # inference, avoiding a second GPU pass through private frames.
                for row in predictions:
                    row["prediction"] = int(row["score"] >= threshold)
                    row["threshold"] = float(threshold)
                    row["verdict"] = "violation" if row["prediction"] else "no_violation"
                    events = base._events(np.asarray(row["window_scores"]), np.asarray(row["window_timestamps"]),
                        threshold, base.TrainingConfig(smoothing_seconds=0.)) if row["prediction"] else []
                    row["events"] = [{"type": "violation", "timestamp_sec": e["timestamp_sec"], "timecode": e["timecode"], "score": e["score"]} for e in events]
                metrics = reviewed._metrics(predictions)
                artifact = {"format_version": 1, "kind": "supervised_clip_visual_mil", "candidate": candidate,
                    "config": asdict(config), "threshold": float(threshold), "source_sha256": _source_hashes(),
                    "base_weights_sha256": clip.EXPECTED_WEIGHT_SHA256, "split": frozen,
                    "training_sha256": sorted(row["sha256"] for row in train), "trainable_visual_parameters": trainable,
                    "supervision_sha256": reviewed._supervision_hash(records),
                    "frame_pack_metadata_sha256": {key: value["metadata_sha256"] for key, value in packs.items()},
                    "frame_pack_integrity_sha256": {key: value["integrity_sha256"] for key, value in packs.items()},
                    "views": next(iter(packs.values()))["views"],
                    "visual_delta_state": {name: parameter.detach().cpu() for name, parameter in model.visual.named_parameters() if name in trainable},
                    "head_state": {name: value.detach().cpu() for name, value in model.head.state_dict().items()},
                    "validation_metrics": metrics, "training_history": history, "limitations": LIMITATIONS,
                    "external_inference_api_used": False, "audio_used": False}
                path = output_dir / (candidate["name"] + ".pt")
                torch.save(artifact, path)
                leaderboard.append({"candidate": candidate, "threshold": float(threshold), "validation_metrics": metrics,
                    "artifact": path.name, "artifact_sha256": reviewed._hash_file(path), "training_history": history,
                    "validation_predictions": predictions, "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad)})
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        torch.set_num_threads(previous_threads)
    leaderboard.sort(key=lambda row: (row["validation_metrics"][config.primary_metric],
        row["validation_metrics"]["balanced_accuracy"], row["validation_metrics"]["recall"]), reverse=True)
    selection = {"evaluation": "supervised_visual_train_only_fit_validation_selection", "candidates": candidate_grid(),
        "config": asdict(config), "winner": leaderboard[0], "leaderboard": leaderboard, "split": frozen,
        "source_sha256": _source_hashes(), "development_partition_opened_for_selection": False,
        "external_inference_api_used": False, "audio_used": False, "limitations": LIMITATIONS}
    scene._write(output_dir / "visual_selection.json", selection)
    scene._write(output_dir / "visual_selection_lock.json", {"winner_artifact": leaderboard[0]["artifact"],
        "winner_sha256": leaderboard[0]["artifact_sha256"], "selection_sha256": reviewed._hash_file(output_dir / "visual_selection.json"),
        "split_sha256": frozen["sha256"], "source_sha256": _source_hashes()})
    return output_dir / leaderboard[0]["artifact"]


def load_finetuned(model_path, checkpoint, device):
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    if artifact.get("kind") != "supervised_clip_visual_mil" or artifact.get("source_sha256") != _source_hashes():
        raise ValueError("Use a trusted visual fine-tuned checkpoint with exact fitted source bytes")
    visual = load_visual(checkpoint)
    expected = configure_trainable_visual(visual, artifact["candidate"]["last_blocks"])
    if sorted(artifact["visual_delta_state"]) != expected:
        raise ValueError("Fine-tuned vision parameter schema differs from the declared final blocks")
    visual.load_state_dict(artifact["visual_delta_state"], strict=False)
    model = VisualMil(visual)
    model.head.load_state_dict(artifact["head_state"])
    return model.to(device).eval(), artifact


def score_visual_development(records, frame_root, split, checkpoint, model_path, output_dir, *, originals_from=None):
    lock_path = Path(output_dir) / "visual_selection_lock.json"
    if not lock_path.is_file():
        raise ValueError("Freeze visual validation selection before development scoring")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (Path(model_path).name != lock["winner_artifact"] or reviewed._hash_file(model_path) != lock["winner_sha256"]
            or reviewed._hash_file(Path(output_dir) / "visual_selection.json") != lock["selection_sha256"]):
        raise ValueError("Only the frozen validation winner can score the exposed development partition")
    artifact_preview = torch.load(model_path, map_location="cpu", weights_only=True)
    config = FinetuneConfig(**artifact_preview["config"])
    originals = scene.original_targets(originals_from)
    records = reviewed._verified_records(records, originals, split)
    partitions, frozen = reviewed.validate_split(records, split)
    if frozen["sha256"] != artifact_preview["split"]["sha256"] or reviewed._supervision_hash(records) != artifact_preview["supervision_sha256"]:
        raise ValueError("Dataset/split changed after visual training")
    packs = {row["sha256"]: read_frame_pack(frame_root, row["sha256"]) for row in partitions["test"]}
    model, artifact = load_finetuned(model_path, checkpoint, _device(config))
    if any(pack["views"] != artifact["views"] for pack in packs.values()):
        raise ValueError("Development/inference frame view order differs from training")
    with threadpool_limits(limits=config.threads):
        rows = _evaluate_records(model, partitions["test"], packs, config, artifact["threshold"], originals)
    report = {"evaluation": "previously_exposed_mixed_development_partition_after_visual_finetuning",
        "independent_test": False, "metrics": reviewed._metrics(rows),
        "domain_metrics": {domain: reviewed._metrics([row for row in rows if row["domain"] == domain])
            for domain in sorted({row["domain"] for row in rows})},
        "predictions": rows, "candidate": artifact["candidate"], "threshold": artifact["threshold"],
        "model_sha256": reviewed._hash_file(model_path), "split": frozen, "limitations": LIMITATIONS,
        "external_inference_api_used": False, "audio_used": False}
    scene._write(Path(output_dir) / "visual_development_report.json", report)
    print(json.dumps({"evaluation": report["evaluation"], "metrics": report["metrics"], "domain_metrics": report["domain_metrics"]}), flush=True)
    return report

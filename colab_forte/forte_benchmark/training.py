"""Deterministic training of small heads over frozen, cached visual features."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Heads have no nondeterministic operators; this catches accidental additions.
    torch.use_deterministic_algorithms(True)


def video_balanced_weights(labels: torch.Tensor, video_ids: Sequence[str]) -> torch.Tensor:
    """Give each class equal total weight, each video equal weight within class.

    Every window of a video receives 1/(windows_in_video * videos_in_class),
    then weights are normalized to mean one for weighted BCE. Thus a long normal
    video cannot dominate merely by producing many more training windows.
    """
    labels = torch.as_tensor(labels).detach().cpu().reshape(-1)
    if len(labels) != len(video_ids) or not len(labels):
        raise ValueError("Labels and video_ids must have the same nonzero length")
    if not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Training labels must be finite binary values 0 or 1")
    assignments: dict[str, int] = {}
    for video_id, label in zip(video_ids, labels.tolist()):
        if video_id in assignments and assignments[video_id] != int(label):
            raise ValueError(f"Video {video_id!r} has inconsistent training labels")
        assignments[video_id] = int(label)
    class_videos = Counter(assignments.values())
    if set(class_videos) != {0, 1}:
        raise ValueError("Training requires both binary classes")
    window_counts = Counter(video_ids)
    weights = torch.tensor([1.0 / (window_counts[video_id] * class_videos[int(label)]) for video_id, label in zip(video_ids, labels.tolist())], dtype=torch.float32)
    return weights * (len(weights) / weights.sum())


def _validated_features(features: torch.Tensor) -> torch.Tensor:
    features = torch.as_tensor(features).detach().cpu().to(torch.float32)
    if features.ndim != 3 or features.shape[1] < 1 or features.shape[2] < 1:
        raise ValueError("Features must have shape (N, T, D) with nonempty T and D")
    if not torch.isfinite(features).all():
        raise ValueError("Features contain NaN or infinity")
    return features


def predict_scores(head: nn.Module, features: torch.Tensor, device: str | torch.device = "cpu", batch_size: int = 32) -> list[float]:
    features = _validated_features(features)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    original_device = next(head.parameters()).device
    was_training = head.training
    scores: list[float] = []
    try:
        head.to(device).eval()
        with torch.no_grad():
            for batch in features.split(batch_size):
                scores.extend(torch.sigmoid(head(batch.to(device))).reshape(-1).cpu().tolist())
    finally:
        head.to(original_device).train(was_training)
    return scores


def train_head(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    video_ids: Sequence[str],
    val_callback: Callable[[nn.Module], dict],
    output_dir: Path,
    device: str | torch.device = "cpu",
    seed: int = 42,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 16,
    patience: int = 4,
) -> tuple[nn.Module, list[dict], int]:
    """Train head parameters only, selecting epochs exclusively with validation.

    Best epoch ordering: higher validation F1, higher recall, lower validation
    loss when available. This function never receives or accesses test labels.
    Saves best_head.pt as a CPU state_dict, reloads it and returns the head on CPU.
    """
    features = _validated_features(features)
    labels = torch.as_tensor(labels).detach().cpu().to(torch.float32).reshape(-1)
    if len(features) != len(labels):
        raise ValueError("Feature and label counts differ")
    weights = video_balanced_weights(labels, video_ids)
    if epochs < 1 or batch_size < 1 or patience < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("epochs, batch_size, patience and lr must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    head.to(device)
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Head has no trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=lr)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    history: list[dict] = []
    best_key = (-math.inf, -math.inf, -math.inf)
    best_epoch = 0
    stale_epochs = 0
    best_state = None
    checkpoint_path = output_dir / "best_head.pt"
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        indices = torch.randperm(len(features), generator=generator)
        for batch_indices in indices.split(batch_size):
            batch = features[batch_indices].to(device)
            target = labels[batch_indices].to(device)
            batch_weights = weights[batch_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch).reshape(-1)
            loss = (criterion(logits, target) * batch_weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_indices)
        head.eval()
        with torch.no_grad():
            validation = dict(val_callback(head))
        if "f1" not in validation or "recall" not in validation:
            raise ValueError("val_callback must return f1 and recall")
        if not all(math.isfinite(float(validation[name])) for name in ("f1", "recall")):
            raise ValueError("Validation F1 and recall must be finite")
        val_loss = float(validation.get("loss", math.inf))
        if math.isnan(val_loss):
            raise ValueError("Validation loss must not be NaN")
        key = (float(validation["f1"]), float(validation["recall"]), -val_loss)
        record = {"epoch": epoch, "train_loss": total_loss / len(features), "val_f1": key[0], "val_recall": key[1]}
        if math.isfinite(val_loss):
            record["val_loss"] = val_loss
        history.append(record)
        if key > best_key:
            best_key, best_epoch, stale_epochs = key, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
            torch.save(best_state, checkpoint_path)
        else:
            stale_epochs += 1
        print(f"epoch {epoch:02d} train_loss={record['train_loss']:.4f} val_f1={key[0]:.4f} val_recall={key[1]:.4f}")
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("No valid validation checkpoint was produced")
    head.to("cpu")
    head.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    head.eval()
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return head, history, best_epoch

"""Point-guided multiple-instance learning on frozen video feature caches.

Each recording is a bag of sampled frame/clip embeddings. Train-only scaling
and PCA precede a small linear or MLP head. The objective combines video bag
labels, positive point labels, and dense negative supervision only on confirmed
normal videos. Unmarked positive-video windows are never negative targets.

This is a targeted development comparison. It cannot establish event taxonomy,
legal interpretation, independent-camera generalization or an accuracy promise.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from .model import (TrainingConfig, _canonical_records, _events,
                    _localization_metrics, _threshold, classification_metrics,
                    grouped_splits, load_features, semantic_feature_config)


@dataclass(frozen=True)
class MilConfig:
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    pca_components: int = 32
    epochs: tuple[int, int] = (100, 200)  # linear, then MLP32; exactly two candidates.
    top_k: int = 3
    positive_radius: float = 1.0
    learning_rate: float = 0.01
    weight_decay: float = 0.001
    dropout: float = 0.25
    anchor_weight: float = 1.0
    normal_instance_weight: float = 0.5
    smoothness_weight: float = 0.01
    primary_metric: str = "balanced_accuracy"
    threads: int = 1
    smoothing_seconds: float = 2.0
    event_merge_seconds: float = 4.0
    max_events: int = 5

    def __post_init__(self):
        if min(self.outer_folds, self.inner_folds) < 2:
            raise ValueError("Grouped CV requires at least two folds")
        if len(self.epochs) != 2 or min(self.epochs) < 1:
            raise ValueError("epochs must specify positive fixed budgets for linear and MLP32")
        if min(self.pca_components, self.top_k, self.threads, self.max_events) < 1:
            raise ValueError("Component, window, thread and event limits must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        values = (self.learning_rate, self.weight_decay, self.positive_radius,
                  self.anchor_weight, self.normal_instance_weight, self.smoothness_weight,
                  self.smoothing_seconds, self.event_merge_seconds)
        if any(not math.isfinite(value) or value < 0 for value in values) or self.learning_rate == 0:
            raise ValueError("Loss/optimizer/time settings must be finite and valid")
        if self.primary_metric not in {"balanced_accuracy", "accuracy", "f1", "f2"}:
            raise ValueError("Unsupported primary_metric")


@contextmanager
def _cpu_threads(count: int):
    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _metric_config(config: MilConfig) -> TrainingConfig:
    return TrainingConfig(seed=config.seed, outer_folds=config.outer_folds,
        inner_folds=config.inner_folds, top_k=config.top_k,
        primary_metric=config.primary_metric, smoothing_seconds=config.smoothing_seconds,
        event_merge_seconds=config.event_merge_seconds, max_events=config.max_events)


def _candidates(config: MilConfig) -> list[dict]:
    return [{"architecture": name, "epochs": int(epochs)}
            for name, epochs in zip(("linear", "mlp32"), config.epochs)]


def frozen_rows(features: np.ndarray) -> np.ndarray:
    """Flatten views of each frozen frame/clip row; no labels or timestamps."""
    normalized = features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8)
    return normalized.reshape(len(features), -1).astype(np.float32, copy=False)


def _prepare(records: list[dict], feature_dir: str | Path) -> dict[str, dict]:
    entries, shape, metadata = {}, None, None
    for record in records:
        features, times, meta = load_features(Path(feature_dir) / f"{record['sha256']}.npz")
        if shape is None:
            shape, metadata = features.shape[1:], meta
        if features.shape[1:] != shape or meta != metadata:
            raise ValueError("MIL needs one consistent encoder configuration and feature shape")
        entries[record["sha256"]] = {"x": frozen_rows(features), "timestamps": times,
            "metadata": meta, "feature_shape": features.shape[1:]}
    return entries


def _fit_preprocessing(records: list[dict], entries: dict, config: MilConfig) -> dict:
    # All windows are retained, including unmarked positive windows. Their use
    # in unsupervised preprocessing introduces no negative label assumption.
    x = np.concatenate([entries[r["sha256"]]["x"] for r in records])
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    pca = PCA(n_components=min(config.pca_components, z.shape[0] - 1, z.shape[1]),
              svd_solver="randomized", iterated_power=2, random_state=config.seed)
    pca.fit(z)
    return {"scaler": scaler, "pca": pca,
            "fit_sha256": [r["sha256"] for r in records], "fit_window_count": len(x)}


def _project(entries: dict, records: list[dict], preprocessing: dict) -> dict[str, np.ndarray]:
    return {r["sha256"]: preprocessing["pca"].transform(
        preprocessing["scaler"].transform(entries[r["sha256"]]["x"])).astype(np.float32)
        for r in records}


def _point_mask(times: np.ndarray, anchors: list[float], radius: float) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    if anchors:
        distance = np.abs(times[:, None] - np.asarray(anchors)[None])
        mask |= distance.min(axis=1) <= radius
        mask[np.argmin(distance, axis=0)] = True
    return mask


def _build_head(input_dim: int, candidate: dict, config: MilConfig) -> nn.Module:
    if candidate["architecture"] == "linear":
        return nn.Linear(input_dim, 1)
    if candidate["architecture"] == "mlp32":
        return nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(),
                             nn.Dropout(config.dropout), nn.Linear(32, 1))
    raise ValueError("Unknown MIL architecture")


def mil_loss(logits: list[torch.Tensor], labels: list[int], point_masks: list[torch.Tensor],
             config: MilConfig) -> dict[str, torch.Tensor]:
    """Bag top-k + positive-anchor + confirmed-normal-instance + smoothness loss.

    Every normal window is considered for hard top-k mining AND receives a
    negative instance target. Positive-video unmarked windows receive neither
    negative instance targets nor pseudo-background targets.
    """
    zero = logits[0].sum() * 0
    bag_losses, anchor_losses, normal_losses, smooth_losses = [], [], [], []
    for values, label, point_mask in zip(logits, labels, point_masks):
        values = values.reshape(-1)
        pooled = torch.topk(values, min(config.top_k, len(values))).values.mean()
        bag_losses.append(F.binary_cross_entropy_with_logits(pooled, pooled.new_tensor(float(label))))
        if label == 1 and bool(point_mask.any()):
            anchor_losses.append(F.softplus(-values[point_mask]).mean())
        if label == 0:
            normal_losses.append(F.softplus(values).mean())
        if len(values) > 1:
            probability = values.sigmoid()
            smooth_losses.append((probability[1:] - probability[:-1]).abs().mean())
    # Each class contributes equally; within class every source video has equal
    # bag weight, regardless of duration or the number of annotated points.
    class_losses = [torch.stack([loss for loss, label in zip(bag_losses, labels) if label == cls]).mean()
                    for cls in (0, 1) if cls in labels]
    bag = torch.stack(class_losses).mean()
    anchor = torch.stack(anchor_losses).mean() if anchor_losses else zero
    normal = torch.stack(normal_losses).mean() if normal_losses else zero
    smooth = torch.stack(smooth_losses).mean() if smooth_losses else zero
    total = bag + config.anchor_weight * anchor + config.normal_instance_weight * normal + config.smoothness_weight * smooth
    return {"total": total, "bag": bag, "anchor": anchor,
            "normal_instance": normal, "smoothness": smooth}


def _train_head(records: list[dict], projected: dict, entries: dict,
                candidate: dict, config: MilConfig) -> nn.Module:
    torch.manual_seed(config.seed)
    xs = [torch.from_numpy(projected[r["sha256"]]) for r in records]
    x = torch.cat(xs)
    lengths = [len(values) for values in xs]
    labels = [r["label"] for r in records]
    masks = [torch.from_numpy(_point_mask(entries[r["sha256"]]["timestamps"],
              r["event_times"] if r["label"] == 1 else [], config.positive_radius)) for r in records]
    head = _build_head(x.shape[1], candidate, config)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # Fixed, predeclared budgets. No validation or outer-test values determine
    # stopping; the inner folds choose between the two completed candidates.
    head.train()
    for _ in range(candidate["epochs"]):
        optimizer.zero_grad(set_to_none=True)
        logits = list(head(x).reshape(-1).split(lengths))
        loss = mil_loss(logits, labels, masks, config)["total"]
        if not torch.isfinite(loss):
            raise ValueError("Non-finite MIL training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
        optimizer.step()
    return head.eval()


def _score_head(head: nn.Module, x: np.ndarray, config: MilConfig) -> tuple[float, np.ndarray]:
    head.eval()
    with torch.inference_mode():
        logits = head(torch.from_numpy(x)).reshape(-1).numpy().astype(float)
    top = np.sort(logits)[-min(config.top_k, len(logits)):]
    return float(expit(top.mean())), expit(logits)


def _select(records: list[dict], entries: dict, config: MilConfig) -> dict:
    candidates = _candidates(config)
    splits = grouped_splits(records, config.inner_folds, config.seed + 1)
    if not splits:
        return {"candidate": candidates[0], "threshold": .5, "selection_metrics": None,
                "selection_folds": [], "leaderboard": [],
                "note": "Insufficient class groups for inner CV; fixed first candidate and threshold 0.5"}
    scores = [dict() for _ in candidates]
    details = []
    for fold, (train_idx, validation_idx) in enumerate(splits):
        train = [records[i] for i in train_idx]
        validation = [records[i] for i in validation_idx]
        preprocessing = _fit_preprocessing(train, entries, config)
        projected = _project(entries, train + validation, preprocessing)
        details.append({"fold": fold, "train_groups": sorted({r["group_id"] for r in train}),
            "validation_groups": sorted({r["group_id"] for r in validation}),
            "preprocessing_fit_sha256": preprocessing["fit_sha256"]})
        for index, candidate in enumerate(candidates):
            head = _train_head(train, projected, entries, candidate, config)
            for record in validation:
                scores[index][record["sha256"]] = _score_head(head, projected[record["sha256"]], config)[0]
    labels = np.asarray([r["label"] for r in records])
    leaderboard = []
    for candidate, by_hash in zip(candidates, scores):
        raw = np.asarray([by_hash[r["sha256"]] for r in records])
        threshold, metrics = _threshold(labels, raw, _metric_config(config))
        leaderboard.append({"candidate": candidate, "threshold": threshold, "selection_metrics": metrics})
    leaderboard.sort(key=lambda row: (row["selection_metrics"][config.primary_metric],
        row["selection_metrics"]["accuracy"], row["selection_metrics"]["recall"]), reverse=True)
    return {**leaderboard[0], "selection_folds": details, "leaderboard": leaderboard,
            "note": "Inner grouped OOF model/threshold selection; selection score is development only"}


def _predict(head: nn.Module, x: np.ndarray, times: np.ndarray,
             selection: dict, config: MilConfig) -> dict:
    # This boundary receives no source record, label, ID or point annotation.
    score, probabilities = _score_head(head, x, config)
    predicted = int(score >= selection["threshold"])
    events = _events(probabilities, times, selection["threshold"], _metric_config(config)) if predicted else []
    return {"score": score, "prediction": predicted, "threshold": selection["threshold"],
        "verdict": "violation" if predicted else "no_violation", "events": events,
        "score_calibrated": False, "localization_available": True,
        "window_scores": probabilities.tolist(), "window_timestamps": times.astype(float).tolist(),
        "aggregation": "sigmoid_of_mean_top_k_logits"}


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_mil_cross_validation(records: list[dict], feature_dir: str | Path,
                             output_dir: str | Path, config: MilConfig | None = None) -> dict:
    config = config or MilConfig()
    records = _canonical_records(records)
    splits = grouped_splits(records, config.outer_folds, config.seed)
    if not splits:
        raise ValueError("MIL outer CV needs at least two independent groups of each class")
    entries = _prepare(records, feature_dir)
    output_dir = Path(output_dir)
    predictions, fold_reports = [], []
    with _cpu_threads(config.threads):
        for fold, (train_idx, test_idx) in enumerate(splits):
            train, test = [records[i] for i in train_idx], [records[i] for i in test_idx]
            print(f"MIL outer fold {fold + 1}/{len(splits)}: train={len(train)}, test={len(test)}", flush=True)
            selection = _select(train, entries, config)
            preprocessing = _fit_preprocessing(train, entries, config)
            projected = _project(entries, train + test, preprocessing)
            head = _train_head(train, projected, entries, selection["candidate"], config)
            current = []
            for record in test:
                key = record["sha256"]
                result = _predict(head, projected[key], entries[key]["timestamps"], selection, config)
                row = {"sha256": key, "video_id": record.get("video_id", key),
                    "group_id": record["group_id"], "label": record["label"],
                    "event_times": record["event_times"], "fold": fold,
                    "selected_model": selection["candidate"], **result}
                predictions.append(row)
                current.append(row)
            metrics = classification_metrics([r["label"] for r in current],
                [r["score"] for r in current], [r["prediction"] for r in current])
            fold_reports.append({"fold": fold, "selection": selection, "metrics": metrics,
                "train_groups": sorted({r["group_id"] for r in train}),
                "test_groups": sorted({r["group_id"] for r in test}),
                "preprocessing_fit_sha256": preprocessing["fit_sha256"]})
            _write_json(output_dir / "mil_cv_progress.json", {"completed_folds": len(fold_reports),
                        "folds": fold_reports, "predictions": predictions})
    metrics = classification_metrics([r["label"] for r in predictions],
        [r["score"] for r in predictions], [r["prediction"] for r in predictions])
    metrics["localization"] = _localization_metrics(predictions)
    report = {"evaluation": "nested_grouped_mil_out_of_fold_development", "config": asdict(config),
        "metrics": metrics, "folds": fold_reports, "predictions": predictions,
        "candidates": _candidates(config), "created_at": datetime.now(timezone.utc).isoformat(),
        "limitations": ["This optional objective comparison was designed after inspecting earlier encoder results; this corpus is development data.",
            "No independent recording/session/camera test or event taxonomy is available.",
            "Scores are uncalibrated and pooled AP/AUC combine different fitted models.",
            "Point labels are incomplete; event precision and interval IoU are not identifiable.",
            "No accuracy guarantee or semantic/legal event explanation is established."]}
    _write_json(output_dir / "mil_cv_report.json", report)
    print(json.dumps({"evaluation": report["evaluation"], "metrics": metrics}, indent=2), flush=True)
    return report


def fit_mil_final(records: list[dict], feature_dir: str | Path,
                   output_dir: str | Path, config: MilConfig | None = None) -> Path:
    config = config or MilConfig()
    records = _canonical_records(records)
    entries = _prepare(records, feature_dir)
    with _cpu_threads(config.threads):
        selection = _select(records, entries, config)
        preprocessing = _fit_preprocessing(records, entries, config)
        projected = _project(entries, records, preprocessing)
        head = _train_head(records, projected, entries, selection["candidate"], config)
    first = entries[records[0]["sha256"]]
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(),
        "training_sha256": [r["sha256"] for r in records],
        "training_groups": sorted({r["group_id"] for r in records}),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__), "selection_score_is_test_score": False,
        "external_inference_api_used": False, "annotation_mode": "binary_bags_and_positive_points"}
    artifact = {"format_version": 1, "kind": "point_guided_mil", "config": asdict(config),
        "candidate": selection["candidate"], "selection": selection, "preprocessing": preprocessing,
        "input_dim": int(projected[records[0]["sha256"]].shape[1]),
        "state_dict": {name: value.detach().cpu().numpy() for name, value in head.state_dict().items()},
        "feature_config": first["metadata"], "feature_shape": first["feature_shape"],
        "feature_semantic_config": semantic_feature_config(first["metadata"]), "provenance": provenance,
        "limitations": "Development model; no semantic event types, legal grounding, independent test or accuracy guarantee."}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model_mil.joblib"
    joblib.dump(artifact, path, compress=3)
    _write_json(output_dir / "mil_final_selection.json", {"selection": selection, "provenance": provenance,
                "limitations": artifact["limitations"]})
    return path


def predict_mil_features(feature_path: str | Path, model_path: str | Path) -> dict:
    """Predict from an unlabeled cache; load only trusted joblib artifacts."""
    artifact = joblib.load(model_path)
    if artifact.get("kind") != "point_guided_mil":
        raise ValueError("Expected a trusted point_guided_mil artifact")
    config = MilConfig(**artifact["config"])
    features, times, metadata = load_features(feature_path)
    if tuple(features.shape[1:]) != tuple(artifact["feature_shape"]) or semantic_feature_config(metadata) != artifact["feature_semantic_config"]:
        raise ValueError("Inference feature configuration differs from MIL training")
    preprocessing = artifact["preprocessing"]
    x = preprocessing["pca"].transform(preprocessing["scaler"].transform(frozen_rows(features))).astype(np.float32)
    head = _build_head(artifact["input_dim"], artifact["candidate"], config)
    head.load_state_dict({name: torch.from_numpy(value) for name, value in artifact["state_dict"].items()})
    with _cpu_threads(config.threads):
        result = _predict(head, x, times, artifact["selection"], config)
    result["feature_provenance_differences"] = {
        key: {"training": artifact["feature_config"].get(key), "inference": metadata.get(key)}
        for key in sorted(set(metadata) | set(artifact["feature_config"]))
        if metadata.get(key) != artifact["feature_config"].get(key)}
    return result

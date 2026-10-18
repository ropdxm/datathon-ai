"""Small point-guided temporal heads on local frozen visual feature caches.

The two declared TCNs learn ordered visual changes rather than whole-recording
summaries. Every supervised fit owns its scaler, PCA, labels and point anchors.
Unmarked positive windows are never negative targets. Custom targets must be
human verified; model proposals and camera-pair guesses are not ground truth.
No network calls, video decoding, text generation or audio processing occur.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F

from . import model as base
from . import temporal


@dataclass(frozen=True)
class SceneTemporalConfig:
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    pca_components: int = 32
    preprocessing_rows_per_video: int = 64
    epochs: tuple[int, int] = (80, 120)
    learning_rate: float = .003
    weight_decay: float = .01
    dropout: float = .25
    top_k: int = 3
    anchor_weight: float = 1.
    normal_instance_weight: float = .5
    smoothness_weight: float = .01
    positive_radius: float = .75
    sample_period_seconds: float = .5
    max_gap_seconds: float = 1.5
    event_merge_seconds: float = 4.
    max_events: int = 5
    primary_metric: str = "balanced_accuracy"
    threads: int = 2
    device: str = "auto"

    def __post_init__(self):
        if min(self.outer_folds, self.inner_folds) < 2:
            raise ValueError("At least two grouped folds are required")
        if len(self.epochs) != 2 or any(not isinstance(x, int) or x < 1 for x in self.epochs):
            raise ValueError("Exactly two positive fixed epoch budgets are required")
        if min(self.pca_components, self.preprocessing_rows_per_video, self.top_k, self.max_events, self.threads) < 1:
            raise ValueError("Component, sampling, event and thread limits must be positive")
        if not 0 <= self.dropout < 1 or self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Invalid dropout or device")
        values = (self.learning_rate, self.weight_decay, self.anchor_weight,
            self.normal_instance_weight, self.smoothness_weight, self.positive_radius,
            self.sample_period_seconds, self.max_gap_seconds, self.event_merge_seconds)
        if any(not math.isfinite(v) or v < 0 for v in values) or min(self.learning_rate, self.sample_period_seconds) == 0:
            raise ValueError("Optimizer, loss and time values must be finite and valid")
        if self.max_gap_seconds < self.sample_period_seconds:
            raise ValueError("max_gap_seconds cannot be shorter than the model sample period")
        if self.primary_metric not in {"accuracy", "balanced_accuracy", "f1", "f2"}:
            raise ValueError("Unsupported primary metric")


def candidate_grid(config: SceneTemporalConfig) -> list[dict]:
    """Exactly two predeclared candidates; stable ties prefer the smaller one."""
    return [{"name": "tcn16_3sec", "hidden": 16, "dilations": [1, 2], "epochs": config.epochs[0]},
            {"name": "tcn32_7sec", "hidden": 32, "dilations": [1, 2, 4], "epochs": config.epochs[1]}]


def _metric_config(config):
    return base.TrainingConfig(seed=config.seed, outer_folds=config.outer_folds,
        inner_folds=config.inner_folds, primary_metric=config.primary_metric,
        top_k=config.top_k, smoothing_seconds=0., event_merge_seconds=config.event_merge_seconds,
        max_events=config.max_events)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _read_report(source):
    return json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else source


def original_targets(source) -> dict[str, int]:
    """Trusted historical report binds original content hashes to original labels."""
    if source is None:
        return {}
    report = _read_report(source)
    rows = report.get("predictions", [])
    if not rows:
        raise ValueError("Original-target allowlist needs historical predictions with sha256 and label")
    mapping = {}
    for row in rows:
        key, label = row.get("sha256"), row.get("label")
        if not isinstance(key, str) or len(key) != 64 or label not in (0, 1) or isinstance(label, bool):
            raise ValueError("Historical target allowlist needs source SHA-256 and integer binary labels")
        if key in mapping:
            raise ValueError("Repeated source SHA in original-target allowlist")
        mapping[key] = int(label)
    return mapping


def supervision_audit(records: list[dict], originals: dict[str, int]) -> dict:
    """Check verification *before* reading any feature file; never promote a proposal."""
    accepted, excluded = [], []
    for source in records:
        row = dict(source)
        key, label = row.get("sha256"), row.get("label")
        status = row.get("label_status")
        reason = None
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            reason = "invalid_source_sha256"
        elif label not in (0, 1) or isinstance(label, bool):
            reason = "missing_or_nonbinary_target"
        elif status in {"machine_proposal", "supplied_unverified", "unlabeled", "pseudo", "pseudo_label"}:
            reason = "unverified_target"
        elif key in originals:
            if int(label) != originals[key]:
                reason = "original_label_differs_from_reference"
            elif status not in (None, "original_provided", "human_verified"):
                reason = "unsupported_original_label_status"
        elif status != "human_verified" or row.get("evaluation_ground_truth_eligible") is not True:
            reason = "custom_target_not_human_verified"
        elif row.get("group_verified") is not True or not str(row.get("group_id", "")).strip():
            reason = "custom_scene_not_verified"
        elif label == 1 and (not row.get("event_times") or row.get("event_times_verified") is not True):
            reason = "custom_positive_timestamps_not_verified"
        if reason:
            excluded.append({"sha256": key, "reason": reason, "label_status": status})
        else:
            accepted.append(row)
    return {"accepted": accepted, "excluded": excluded, "accepted_count": len(accepted),
        "excluded_count": len(excluded), "original_reference_count": len(originals),
        "policy": "original_reference_targets_or_human_verified_custom_labels_points_and_scenes"}


def verified_records(records, originals):
    audit = supervision_audit(records, originals)
    if audit["excluded"]:
        counts = {}
        for row in audit["excluded"]:
            counts[row["reason"]] = counts.get(row["reason"], 0) + 1
        raise ValueError("Unverified/inconsistent supervised targets rejected before cache reads: " + json.dumps(counts, sort_keys=True))
    result = base._canonical_records(audit["accepted"])
    if not result:
        raise ValueError("No verified supervised recordings")
    # A transcode or a second view must not enter another outer/inner group.
    owners = {}
    for row in result:
        lineage = {row["sha256"], *row.get("lineage_sha256s", [])}
        for key in ("source_sha256", "parent_sha256", "original_sha256"):
            if row.get(key):
                lineage.add(row[key])
        for key in lineage:
            if key in owners and owners[key] != row["group_id"]:
                raise ValueError("Source/parent lineage crosses scene groups")
            owners[key] = row["group_id"]
    return sorted(result, key=lambda row: row["sha256"])


def apply_verified_groups(records, source):
    """Apply only explicit verified SHA-to-scene metadata, never visual guesses."""
    if source is None:
        return [dict(row) for row in records]
    payload = _read_report(source)
    rows = payload.get("records", []) if isinstance(payload, dict) else payload
    known = {row["sha256"] for row in records}
    mapping = {}
    for row in rows:
        key, group = row.get("sha256"), str(row.get("group_id", "")).strip()
        if key not in known or key in mapping:
            raise ValueError("Group metadata contains an unknown or repeated source SHA")
        if row.get("group_verified") is not True or not group or not row.get("verification_source"):
            raise ValueError("Every scene mapping requires group_verified, group_id and verification_source")
        mapping[key] = row
    result = []
    for source_row in records:
        row = dict(source_row)
        if row["sha256"] in mapping:
            metadata = mapping[row["sha256"]]
            row.update(group_id=metadata["group_id"], group_verified=True,
                       group_verification_source=metadata["verification_source"])
        result.append(row)
    return result


def _resample_rows(x: np.ndarray, times: np.ndarray, config: SceneTemporalConfig) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Fixed-second grid; interpolate only within observed contiguous segments.

    Large unsampled gaps split sequences. Their two sides never become temporal
    neighbors. Absolute recording clocks never enter the model's feature array.
    """
    boundaries = np.r_[0, np.flatnonzero(np.diff(times) > config.max_gap_seconds) + 1, len(times)]
    grids, pieces, lengths = [], [], []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        t = times[start:stop].astype(np.float64)
        if len(t) == 1:
            grid, values = t.copy(), x[start:stop].copy()
        else:
            count = int(np.floor((t[-1] - t[0]) / config.sample_period_seconds + 1e-6)) + 1
            grid = t[0] + np.arange(count, dtype=np.float64) * config.sample_period_seconds
            grid = grid[grid <= t[-1] + 1e-6]
            values = np.column_stack([np.interp(grid, t, x[start:stop, col]) for col in range(x.shape[1])]).astype(np.float32)
        grids.append(grid)
        pieces.append(values)
        lengths.append(len(grid))
    return np.concatenate(pieces), np.concatenate(grids), lengths


def _prepare(records, feature_dir, config):
    entries, shape, semantic = {}, None, None
    for row in records:
        x, times, meta = base.load_features(Path(feature_dir) / f"{row['sha256']}.npz")
        current = _feature_protocol(meta)
        if shape is None:
            shape, semantic = x.shape[1:], current
        if x.shape[1:] != shape or current != semantic:
            raise ValueError("Use one consistent visual encoder and sampling geometry")
        x = _visual_rows(x, meta)
        x, grid, segments = _resample_rows(x, times, config)
        entries[row["sha256"]] = {"x": x, "timestamps": grid, "segments": segments,
            "metadata": meta, "feature_shape": tuple(shape)}
    return entries


def _visual_rows(features, metadata):
    if metadata.get("concept_names"):
        # Observable-state probabilities carry meaningful magnitude. Unit
        # normalization would turn an unknown/low-visibility row into evidence.
        names = metadata["concept_names"]
        if not isinstance(names, list) or len(names) != features.shape[-1] or len(set(names)) != len(names):
            raise ValueError("Semantic cache needs one distinct concept name per feature column")
        if np.any(features < 0) or np.any(features > 1):
            raise ValueError("Declared semantic-state scores must be in [0,1]")
        return features.reshape(len(features), -1).astype(np.float32)
    normalized = features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8)
    return normalized.reshape(len(features), -1).astype(np.float32)


def _feature_protocol(metadata):
    result = base.semantic_feature_config(metadata)
    if metadata.get("concept_names"):
        # Generic encoder compatibility excludes these semantic projection
        # fields; retain their exact order and scoring protocol for this head.
        result.update({key: metadata.get(key) for key in ("concept_names", "temperature",
            "prompts_sha256", "text_prototypes_sha256", "source_feature_config",
            "continuous_window_coverage", "calibrated_probabilities") if key in metadata})
    return result


def _owner_manifest(records):
    rows = [{"sha256": row["sha256"], "group_id": row["group_id"], "label": row["label"],
             "event_times": row["event_times"], "label_status": row.get("label_status", "original_provided"),
             "group_verified": bool(row.get("group_verified", False)),
             "event_times_verified": bool(row.get("event_times_verified", False))}
            for row in sorted(records, key=lambda item: item["sha256"])]
    return {"records": rows, "sha256": _digest(rows)}


def _fit_preprocessing(records, entries, config):
    arrays = []
    for row in records:
        x = entries[row["sha256"]]["x"]
        indices = np.unique(np.linspace(0, len(x) - 1, min(len(x), config.preprocessing_rows_per_video)).astype(int))
        arrays.append(x[indices])
    x = np.concatenate(arrays)
    if len(x) < 2:
        raise ValueError("Preprocessing needs at least two training observations")
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    pca = PCA(n_components=min(config.pca_components, z.shape[0] - 1, z.shape[1]),
              svd_solver="randomized", iterated_power=2, random_state=config.seed).fit(z)
    # Equalize retained component scales using *only* training-owner samples.
    component_scaler = StandardScaler().fit(pca.transform(z))
    return {"scaler": scaler, "pca": pca, "component_scaler": component_scaler,
        "fit_sha256": sorted(row["sha256"] for row in records), "fit_window_count": len(x),
        "owner_manifest": _owner_manifest(records)}


def _project(records, entries, preprocessing):
    return {row["sha256"]: preprocessing["component_scaler"].transform(preprocessing["pca"].transform(
        preprocessing["scaler"].transform(entries[row["sha256"]]["x"]))).astype(np.float32)
        for row in records}


def _point_mask(times, anchors, config):
    result = np.zeros(len(times), dtype=bool)
    for anchor in anchors:
        distance = np.abs(times - anchor)
        result |= distance <= config.positive_radius
        closest = int(distance.argmin())
        # Do not invent an observation in a large sampling gap.
        if distance[closest] <= max(config.positive_radius, config.sample_period_seconds):
            result[closest] = True
    return result


class TemporalHead(nn.Module):
    """Masked residual dilated Conv1d; recordings never share a sequence axis."""
    def __init__(self, input_dim, candidate, dropout):
        super().__init__()
        hidden = candidate["hidden"]
        self.projection = nn.Conv1d(input_dim, hidden, 1)
        self.convolutions = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, dilation=d, padding=d)
                                          for d in candidate["dilations"]])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in candidate["dilations"]])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv1d(hidden, 1, 1)

    def forward(self, values, valid):
        mask = valid[:, None].to(values.dtype)
        hidden = F.gelu(self.projection(values)) * mask
        for convolution, norm in zip(self.convolutions, self.norms):
            updated = convolution(hidden)
            updated = norm(updated.transpose(1, 2)).transpose(1, 2)
            hidden = (hidden + self.dropout(F.gelu(updated))) * mask
        return self.output(hidden)[:, 0] * valid.to(values.dtype)


def temporal_loss(logits, labels, masks, segments, config):
    """Class-balanced bags + positive points + normal windows + within-gap TV."""
    zero = logits[0].sum() * 0
    bags, points, normals, smooth = [], [], [], []
    for values, label, point_mask, lengths in zip(logits, labels, masks, segments):
        pooled = torch.topk(values, min(config.top_k, len(values))).values.mean()
        bags.append(F.binary_cross_entropy_with_logits(pooled, pooled.new_tensor(float(label))))
        if label == 1 and bool(point_mask.any()):
            points.append(F.softplus(-values[point_mask]).mean())
        if label == 0:
            normals.append(F.softplus(values).mean())
        pieces = values.sigmoid().split(lengths)
        terms = [(piece[1:] - piece[:-1]).abs().mean() for piece in pieces if len(piece) > 1]
        if terms:
            smooth.append(torch.stack(terms).mean())
    bag = torch.stack([torch.stack([loss for loss, label in zip(bags, labels) if label == cls]).mean()
                       for cls in (0, 1) if cls in labels]).mean()
    point = torch.stack(points).mean() if points else zero
    normal = torch.stack(normals).mean() if normals else zero
    variation = torch.stack(smooth).mean() if smooth else zero
    return {"total": bag + config.anchor_weight * point + config.normal_instance_weight * normal + config.smoothness_weight * variation,
            "bag": bag, "point": point, "normal": normal, "smoothness": variation}


def _pack(records, projected, entries, device):
    pieces, ownership = [], []
    for row in records:
        key = row["sha256"]
        start = 0
        owned = []
        for length in entries[key]["segments"]:
            owned.append(len(pieces))
            pieces.append(projected[key][start:start + length])
            start += length
        ownership.append(owned)
    max_len, input_dim = max(len(x) for x in pieces), pieces[0].shape[1]
    if max_len > 8192:
        raise ValueError("A temporal segment exceeds the declared 8192-step memory bound; split its recording before this experiment")
    values = np.zeros((len(pieces), input_dim, max_len), dtype=np.float32)
    valid = np.zeros((len(pieces), max_len), dtype=bool)
    for index, piece in enumerate(pieces):
        values[index, :, :len(piece)] = piece.T
        valid[index, :len(piece)] = True
    return torch.from_numpy(values).to(device), torch.from_numpy(valid).to(device), ownership


def _device(config):
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA explicitly requested but unavailable")
    return "cuda" if config.device == "cuda" or config.device == "auto" and torch.cuda.is_available() else "cpu"


@contextmanager
def _compute(config):
    previous = torch.get_num_threads()
    old_benchmark, old_deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    torch.set_num_threads(config.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        with threadpool_limits(limits=config.threads):
            yield
    finally:
        torch.set_num_threads(previous)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old_benchmark, old_deterministic


def _train_head(records, projected, entries, candidate, config):
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = _device(config)
    values, valid, ownership = _pack(records, projected, entries, device)
    head = TemporalHead(values.shape[1], candidate, config.dropout).to(device)
    masks = [torch.from_numpy(_point_mask(entries[row["sha256"]]["timestamps"], row["event_times"]
        if row["label"] == 1 else [], config)).to(device) for row in records]
    labels = [row["label"] for row in records]
    segments = [entries[row["sha256"]]["segments"] for row in records]
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    head.train()
    for _ in range(candidate["epochs"]):
        optimizer.zero_grad(set_to_none=True)
        output = head(values, valid)
        logits = [torch.cat([output[index][valid[index]] for index in owned]) for owned in ownership]
        loss = temporal_loss(logits, labels, masks, segments, config)["total"]
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite temporal loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
        optimizer.step()
    return head.cpu().eval()


def _score_head(head, x, segments, config):
    # Inference accepts visual arrays and segment lengths, never labels/points.
    fake = [{"sha256": "inference"}]
    values, valid, _ = _pack(fake, {"inference": x}, {"inference": {"segments": segments}}, "cpu")
    with torch.inference_mode():
        output = head.eval()(values, valid)
        logits = torch.cat([output[index][valid[index]] for index in range(len(values))]).numpy().astype(float)
    score = float(expit(np.sort(logits)[-min(config.top_k, len(logits)):].mean()))
    return score, expit(logits)


def _select(records, entries, config):
    candidates = candidate_grid(config)
    splits = base.grouped_splits(records, config.inner_folds, config.seed + 1)
    if not splits:
        raise ValueError("Inner selection needs >=2 independent groups of each class; do not fall back to test-guided tuning")
    scores, folds = [dict() for _ in candidates], []
    for fold, (train_index, valid_index) in enumerate(splits):
        train, validation = [records[i] for i in train_index], [records[i] for i in valid_index]
        preprocessing = _fit_preprocessing(train, entries, config)
        projected = _project(train + validation, entries, preprocessing)
        folds.append({"fold": fold, "train_groups": sorted({r["group_id"] for r in train}),
            "validation_groups": sorted({r["group_id"] for r in validation}),
            "validation_sha256": sorted(r["sha256"] for r in validation),
            "preprocessing_fit_sha256": preprocessing["fit_sha256"],
            "supervision_owners_sha256": preprocessing["owner_manifest"]["sha256"]})
        for index, candidate in enumerate(candidates):
            head = _train_head(train, projected, entries, candidate, config)
            for row in validation:
                key = row["sha256"]
                scores[index][key] = _score_head(head, projected[key], entries[key]["segments"], config)[0]
    labels = np.asarray([row["label"] for row in records])
    leaderboard = []
    for candidate, by_hash in zip(candidates, scores):
        raw = np.asarray([by_hash[row["sha256"]] for row in records])
        threshold, metrics = base._threshold(labels, raw, _metric_config(config))
        leaderboard.append({"candidate": candidate, "threshold": float(threshold), "selection_metrics": metrics})
    leaderboard.sort(key=lambda row: (row["selection_metrics"][config.primary_metric],
        row["selection_metrics"]["accuracy"], row["selection_metrics"]["recall"]), reverse=True)
    return {**leaderboard[0], "selection_folds": folds, "leaderboard": leaderboard,
        "selection_source_sha256": sorted(row["sha256"] for row in records),
        "note": "Architecture and threshold selected only from training-owner grouped inner OOF predictions"}


def _predict(head, x, entry, selection, config):
    score, windows = _score_head(head, x, entry["segments"], config)
    predicted = int(score >= selection["threshold"])
    times = entry["timestamps"]
    events, observed_seconds, start = [], 0., 0
    for length in entry["segments"]:
        stop = start + length
        segment_times, segment_scores = times[start:stop], windows[start:stop]
        observed_seconds += float(segment_times[-1] - segment_times[0])
        if predicted and segment_scores.max() >= selection["threshold"]:
            events.extend(base._events(segment_scores, segment_times, selection["threshold"], _metric_config(config)))
        start = stop
    events = sorted(sorted(events, key=lambda event: event["score"], reverse=True)[:config.max_events],
                    key=lambda event: event["timestamp_sec"])
    # Keep the requested simple label; no invented event taxonomy.
    events = [{"type": "violation", "timestamp_sec": event["timestamp_sec"],
               "timecode": event["timecode"], "score": event["score"]} for event in events]
    return {"score": score, "prediction": predicted, "threshold": selection["threshold"],
        "verdict": "violation" if predicted else "no_violation", "events": events,
        "localization_available": True, "score_calibrated": False,
        "observed_seconds": observed_seconds,
        "window_scores": windows.tolist(), "window_timestamps": times.tolist(),
        "aggregation": "sigmoid_mean_top3_logits" if config.top_k == 3 else "sigmoid_mean_topk_logits"}


def _metrics(rows):
    metrics = base.classification_metrics([r["label"] for r in rows], [r["score"] for r in rows], [r["prediction"] for r in rows])
    metrics["localization"] = base._localization_metrics(rows)
    negatives = [row for row in rows if row["label"] == 0]
    observation_seconds = sum(row["observed_seconds"] for row in negatives)
    false_events = sum(len(row["events"]) for row in negatives)
    metrics["normal_video_false_alarm_events"] = false_events
    metrics["false_alarms_per_observed_hour"] = false_events / (observation_seconds / 3600) if observation_seconds else None
    metrics["normal_observed_seconds"] = observation_seconds
    metrics["false_alarm_exposure_note"] = "Sum of within-segment sampled spans excludes unsampled gaps; source duration is not a model predictor."
    return metrics


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_hashes():
    return {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in (("scene_temporal", __import__(__name__, fromlist=["__name__"])),
                                 ("model", base), ("temporal", temporal))}


LIMITATIONS = [
    "This model family was designed after earlier comparisons; original recordings remain repeatedly inspected development data.",
    "Group folds prevent declared related views crossing folds; original source-hash groups are recording groups, not verified independent scenes.",
    "Point anchors are incomplete; point recall is measurable, event precision and interval IoU are not identifiable.",
    "Scores are uncalibrated; pooled AP/AUC combine independently fitted outer models.",
    "Frozen frame embeddings may lose hand/object detail and motion between samples; no new visual evidence is created by interpolation.",
    "Max/top-k summaries and sequence boundaries can still encode observation length, despite no duration/filename predictor.",
    "No accuracy guarantee, legal event interpretation, or verified custom labels follows from training or synthetic tests.",
]


def run_scene_temporal_cv(records, feature_dir, output_dir, config=None, *, folds_from=None, originals_from=None):
    config = config or SceneTemporalConfig()
    originals = original_targets(originals_from if originals_from is not None else folds_from)
    records = verified_records(records, originals)
    if folds_from is not None:
        splits, source = temporal.frozen_outer_splits(records, folds_from, temporal.TemporalConfig(
            seed=config.seed, outer_folds=config.outer_folds, inner_folds=config.inner_folds))
    else:
        splits = base.grouped_splits(records, config.outer_folds, config.seed)
        assignment = {records[index]["sha256"]: fold for fold, (_, test) in enumerate(splits) for index in test}
        source = {"kind": "generated_declared_groups", "fold_assignment_sha256": _digest(assignment),
                  "independent_test": False}
    if not splits:
        raise ValueError("Outer evaluation needs >=2 independent scene groups of each class")
    entries = _prepare(records, feature_dir, config)
    predictions, folds = [], []
    with _compute(config):
        for fold, (train_index, test_index) in enumerate(splits):
            train, test = [records[i] for i in train_index], [records[i] for i in test_index]
            print(f"Scene TCN outer fold {fold + 1}/{len(splits)}: train={len(train)}, test={len(test)}", flush=True)
            selection = _select(train, entries, config)
            preprocessing = _fit_preprocessing(train, entries, config)
            projected = _project(train + test, entries, preprocessing)
            head = _train_head(train, projected, entries, selection["candidate"], config)
            current = []
            for row in test:
                key = row["sha256"]
                result = _predict(head, projected[key], entries[key], selection, config)
                current.append({"sha256": key, "group_id": row["group_id"], "label": row["label"],
                    "event_times": row["event_times"], "fold": fold, **result})
            predictions.extend(current)
            folds.append({"fold": fold, "train_groups": sorted({r["group_id"] for r in train}),
                "test_groups": sorted({r["group_id"] for r in test}),
                "test_sha256": sorted(row["sha256"] for row in test),
                "preprocessing_fit_sha256": preprocessing["fit_sha256"],
                "supervision_owners_sha256": preprocessing["owner_manifest"]["sha256"],
                "selection": selection, "metrics": _metrics(current)})
            _write(Path(output_dir) / "scene_temporal_cv_progress.json", {"completed_folds": len(folds),
                "folds": folds, "predictions": predictions})
    grouping = ("verified_scene_groups" if all(row.get("group_verified") is True for row in records)
                else "recording_groups_not_independently_scene_verified")
    report = {"evaluation": "nested_grouped_point_guided_tcn_development", "grouping": grouping, "config": asdict(config),
        "candidates": candidate_grid(config), "fold_source": source, "metrics": _metrics(predictions),
        "folds": folds, "predictions": predictions, "source_sha256": _source_hashes(),
        "supervision": _owner_manifest(records), "created_at": datetime.now(timezone.utc).isoformat(),
        "external_inference_api_used": False, "audio_used": False, "limitations": LIMITATIONS}
    _write(Path(output_dir) / "scene_temporal_cv_report.json", report)
    print(json.dumps({"evaluation": report["evaluation"], "metrics": report["metrics"]}, indent=2), flush=True)
    return report


def fit_scene_temporal_final(records, feature_dir, output_dir, config=None, *, originals_from=None):
    config = config or SceneTemporalConfig()
    records = verified_records(records, original_targets(originals_from))
    entries = _prepare(records, feature_dir, config)
    with _compute(config):
        selection = _select(records, entries, config)
        preprocessing = _fit_preprocessing(records, entries, config)
        projected = _project(records, entries, preprocessing)
        head = _train_head(records, projected, entries, selection["candidate"], config)
    first = entries[records[0]["sha256"]]
    provenance = {"training_sha256": preprocessing["fit_sha256"], "training_groups": sorted({r["group_id"] for r in records}),
        "source_sha256": _source_hashes(), "supervision_owners_sha256": preprocessing["owner_manifest"]["sha256"],
        "created_at": datetime.now(timezone.utc).isoformat(), "torch_version": str(torch.__version__),
        "annotation_mode": "verified_binary_bags_positive_points_and_normal_windows",
        "external_inference_api_used": False, "audio_used": False, "selection_score_is_test_score": False}
    artifact = {"format_version": 1, "kind": "scene_point_guided_tcn", "config": asdict(config),
        "selection": selection, "candidate": selection["candidate"], "preprocessing": preprocessing,
        "input_dim": next(iter(projected.values())).shape[1], "feature_shape": first["feature_shape"],
        "feature_semantic_config": _feature_protocol(first["metadata"]),
        "state_dict": {key: value.detach().cpu().numpy() for key, value in head.state_dict().items()},
        "provenance": provenance, "limitations": LIMITATIONS}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model_scene_temporal.joblib"
    joblib.dump(artifact, path, compress=3)
    _write(output_dir / "scene_temporal_final_selection.json", {"selection": selection, "provenance": provenance,
        "limitations": LIMITATIONS})
    return path


def predict_scene_temporal_features(feature_path, model_path):
    """Unlabeled numeric cache inference, with exact fitted-source provenance."""
    artifact = joblib.load(model_path)  # Trusted local artifacts only.
    if artifact.get("kind") != "scene_point_guided_tcn":
        raise ValueError("Expected a trusted scene_point_guided_tcn artifact")
    if artifact.get("provenance", {}).get("source_sha256") != _source_hashes():
        raise ValueError("Temporal model source fingerprints differ; use the fitted source or refit")
    config = SceneTemporalConfig(**artifact["config"])
    x, times, metadata = base.load_features(feature_path)
    if tuple(x.shape[1:]) != tuple(artifact["feature_shape"]) or _feature_protocol(metadata) != artifact["feature_semantic_config"]:
        raise ValueError("Inference visual feature protocol differs from training")
    x = _visual_rows(x, metadata)
    x, times, segments = _resample_rows(x, times, config)
    preprocessing = artifact["preprocessing"]
    with _compute(config):
        projected = preprocessing["component_scaler"].transform(preprocessing["pca"].transform(
            preprocessing["scaler"].transform(x))).astype(np.float32)
        head = TemporalHead(artifact["input_dim"], artifact["candidate"], config.dropout)
        head.load_state_dict({key: torch.from_numpy(value) for key, value in artifact["state_dict"].items()})
        return _predict(head, projected, {"timestamps": times, "segments": segments}, artifact["selection"], config)

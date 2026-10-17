"""Small-data video models with point supervision and nested grouped validation.

The frozen visual encoder runs separately. This module consumes feature caches,
never pixels, filenames, recording duration, or annotation times at inference.
Unmarked windows of positive videos are *unlabeled*, not negative examples.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import sklearn
from scipy.special import expit
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, fbeta_score,
    precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    window_radii: tuple[float, ...] = (1.0, 2.0, 4.0)
    positive_radius: float = 1.0
    max_train_windows_per_video: int = 24
    pca_components: int = 32
    preprocessing_modes: tuple[str, ...] = ("pca", "raw")
    logistic_cs: tuple[float, ...] = (0.01, 0.1, 1.0)
    rbf_cs: tuple[float, ...] = (1.0, 10.0)
    heads: tuple[str, ...] = ("point", "pooled")
    aggregations: tuple[str, ...] = ("max", "smooth_max", "topk")
    smoothing_seconds: float = 2.0
    top_k: int = 3
    event_merge_seconds: float = 4.0
    max_events: int = 5
    primary_metric: str = "balanced_accuracy"

    def __post_init__(self):
        if self.primary_metric not in ("balanced_accuracy", "accuracy", "f1", "f2"):
            raise ValueError("Unsupported primary_metric")
        if not self.heads or set(self.heads) - {"point", "pooled"}:
            raise ValueError("heads must contain point and/or pooled")
        if not self.aggregations or set(self.aggregations) - {"max", "smooth_max", "topk"}:
            raise ValueError("Unsupported aggregation")
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("Grouped fold counts must be >= 2")
        if not self.window_radii or min(self.window_radii) <= 0 or self.positive_radius < 0:
            raise ValueError("Temporal radii must be valid")
        if min(self.max_train_windows_per_video, self.top_k, self.max_events) < 1 or self.pca_components < 0:
            raise ValueError("Sample/component/event limits must be positive")
        if not self.preprocessing_modes or set(self.preprocessing_modes) - {"pca", "raw"}:
            raise ValueError("preprocessing_modes must contain pca and/or raw")
        if not self.logistic_cs and not self.rbf_cs:
            raise ValueError("At least one classifier is required")
        if any(c <= 0 for c in (*self.logistic_cs, *self.rbf_cs)):
            raise ValueError("Regularization C values must be positive")


def load_features(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read and validate the extractor's stable [time, view, feature] contract."""
    with np.load(path, allow_pickle=False) as cache:
        features = np.asarray(cache["features"], dtype=np.float32)
        timestamps = np.asarray(cache["timestamps"], dtype=np.float32)
        config_json = str(cache["config_json"].item()) if "config_json" in cache else "{}"
        metadata = json.loads(config_json)
        if "config_hash" in cache:
            computed = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
            if str(cache["config_hash"].item()) != computed:
                raise ValueError(f"Feature configuration hash mismatch: {path}")
        if "sha256" in cache and str(cache["sha256"].item()) != Path(path).stem:
            raise ValueError(f"Feature cache filename does not match embedded source SHA-256: {path}")
    if features.ndim == 2:
        features = features[:, None, :]
    if features.ndim != 3 or not all(features.shape) or timestamps.shape != (len(features),):
        raise ValueError(f"Invalid feature/timestamp shape in {path}")
    if not np.isfinite(features).all() or not np.isfinite(timestamps).all():
        raise ValueError(f"Nonfinite feature cache: {path}")
    if np.any(np.diff(timestamps) <= 0) or timestamps[0] < 0:
        raise ValueError(f"Timestamps must increase strictly: {path}")
    return features, timestamps, metadata


def _normalise(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norm, 1e-8)


def window_features(features: np.ndarray, timestamps: np.ndarray,
                    radii: Iterable[float] = (1., 2., 4.)) -> np.ndarray:
    """Deterministic multiscale appearance, change and ordering descriptors.

    No fitted preprocessing occurs here. Every timestamp, including video
    boundaries, gets one descriptor. Anchors are deliberately not an argument.
    """
    radii = tuple(radii)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 2:
        features = features[:, None, :]
    timestamps = np.asarray(timestamps, dtype=np.float32)
    if features.ndim != 3 or len(features) == 0 or len(features) != len(timestamps):
        raise ValueError("Expected nonempty aligned temporal features")
    f = _normalise(features)
    n = len(f)
    zero = np.zeros_like(f[:1])
    prefix = np.concatenate([zero, np.cumsum(f, axis=0, dtype=np.float32)])
    squares = np.concatenate([zero, np.cumsum(f * f, axis=0, dtype=np.float32)])
    delta = np.concatenate([zero, np.abs(np.diff(f, axis=0))])
    delta_prefix = np.concatenate([zero, np.cumsum(delta, axis=0, dtype=np.float32)])
    parts = [f.reshape(n, -1)]
    for radius in radii:
        left = np.searchsorted(timestamps, timestamps - radius, side="left")
        right = np.searchsorted(timestamps, timestamps + radius, side="right")
        count = (right - left)[:, None, None]
        mean = (prefix[right] - prefix[left]) / count
        variance = np.maximum((squares[right] - squares[left]) / count - mean * mean, 0)
        change = (delta_prefix[right] - delta_prefix[np.minimum(left + 1, right)])
        change /= np.maximum(count - 1, 1)
        parts.extend([mean.reshape(n, -1), np.sqrt(variance).reshape(n, -1), change.reshape(n, -1)])
    # Signed before/after context retains movement order, unlike whole-video pooling.
    radius = max(radii)
    center = np.arange(n)
    left = np.searchsorted(timestamps, timestamps - radius, side="left")
    right = np.searchsorted(timestamps, timestamps + radius, side="right")
    before = (prefix[center + 1] - prefix[left]) / (center + 1 - left)[:, None, None]
    after = (prefix[right] - prefix[center]) / (right - center)[:, None, None]
    parts.append((after - before).reshape(n, -1))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def pooled_features(features: np.ndarray) -> np.ndarray:
    """Frozen whole-video baseline; contains no counts, duration or timestamps."""
    f = _normalise(np.asarray(features, dtype=np.float32))
    change = np.abs(np.diff(f, axis=0)).mean(axis=0) if len(f) > 1 else np.zeros_like(f[0])
    return np.concatenate([f.mean(axis=0).ravel(), f.std(axis=0).ravel(), change.ravel()])[None]


def _canonical_records(records: list[dict]) -> list[dict]:
    by_hash: dict[str, dict] = {}
    for source in records:
        r = dict(source)
        if int(r["label"]) not in (0, 1):
            raise ValueError("Labels must be binary 0/1")
        r["label"] = int(r["label"])
        r["group_id"] = str(r.get("group_id", r["sha256"]))
        anchors = sorted(set(float(x) for x in r.get("event_times", [])))
        if any(not np.isfinite(x) or x < 0 for x in anchors):
            raise ValueError("Invalid event timestamp")
        if r["label"] == 0 and anchors:
            raise ValueError("A confirmed-normal video cannot have violation anchors")
        r["event_times"] = anchors
        key = str(r["sha256"])
        if key in by_hash:
            if r["label"] != by_hash[key]["label"] or r["group_id"] != by_hash[key]["group_id"]:
                raise ValueError("Conflicting labels or groups for duplicate video content")
            by_hash[key]["event_times"] = sorted(set(by_hash[key]["event_times"] + anchors))
        else:
            by_hash[key] = r
    return list(by_hash.values())


def grouped_splits(records: list[dict], folds: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Choose feasible stratified folds and explicitly check group separation."""
    y = np.asarray([r["label"] for r in records])
    groups = np.asarray([r.get("group_id", r["sha256"]) for r in records])
    possible = min(folds, *(len(set(groups[y == k])) for k in (0, 1)))
    for n_splits in range(possible, 1, -1):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y, groups))
        if all(len(set(y[train])) == 2 and len(set(y[test])) == 2 for train, test in splits):
            for train, test in splits:
                assert not set(groups[train]) & set(groups[test]), "Group leakage"
            return splits
    return []


def _prepare(records: list[dict], feature_dir: str | Path, config: TrainingConfig) -> dict:
    data = {}
    expected_shape, expected_config = None, None
    for r in records:
        key = r["sha256"]
        f, t, meta = load_features(Path(feature_dir) / f"{key}.npz")
        if expected_shape is None:
            expected_shape, expected_config = f.shape[1:], meta
        if f.shape[1:] != expected_shape or meta != expected_config:
            raise ValueError("Mixed feature shapes or extractor configurations; use a single feature run")
        data[key] = {"point": window_features(f, t, config.window_radii),
                     "pooled": pooled_features(f), "timestamps": t, "metadata": meta,
                     "feature_shape": f.shape[1:]}
    return data


def _training_data(records: list[dict], data: dict, head: str, config: TrainingConfig):
    xs, ys, weights = [], [], []
    for r in records:
        x = data[r["sha256"]][head]
        if head == "pooled":
            idx = np.array([0])
        elif r["label"] == 0:
            idx = np.unique(np.linspace(0, len(x) - 1,
                min(len(x), config.max_train_windows_per_video)).astype(int))
        elif r["event_times"]:
            t = data[r["sha256"]]["timestamps"]
            distance = np.min(np.abs(t[:, None] - np.asarray(r["event_times"])[None]), axis=1)
            idx = np.flatnonzero(distance <= config.positive_radius)
            # A point between sparse sample times must still receive supervision.
            nearest = np.argmin(np.abs(t[:, None] - np.asarray(r["event_times"])[None]), axis=0)
            idx = np.unique(np.concatenate([idx, nearest]))
            if len(idx) > config.max_train_windows_per_video:
                nearest = np.unique(nearest)
                other = np.setdiff1d(idx, nearest)
                budget = max(config.max_train_windows_per_video - len(nearest), 0)
                selected = other[np.linspace(0, len(other) - 1, min(len(other), budget)).astype(int)] if budget else []
                idx = np.unique(np.concatenate([nearest, selected])).astype(int)
        else:
            # Missing point labels do not authorize labeling the entire positive video.
            continue
        xs.append(x[idx])
        ys.extend([r["label"]] * len(idx))
        weights.extend([1. / len(idx)] * len(idx))
    if not xs or len(set(ys)) != 2:
        raise ValueError(f"Head {head} needs labeled examples of both classes")
    y, w = np.asarray(ys), np.asarray(weights)
    # Equal total weight per source video, then equal total weight per class.
    for label in (0, 1):
        w[y == label] *= len(y) / (2 * w[y == label].sum())
    return np.concatenate(xs), y, w


def _preprocess_fit(x: np.ndarray, config: TrainingConfig, mode: str = "pca"):
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    if mode == "raw" or config.pca_components == 0:
        identity = FunctionTransformer()
        return scaler, identity, identity.fit_transform(z)
    components = min(config.pca_components, z.shape[1], z.shape[0] - 1)
    pca = PCA(n_components=components, svd_solver="randomized", random_state=config.seed)
    return scaler, pca, pca.fit_transform(z)


def _classifier(kind: str, c: float, seed: int, preprocessing: str = "pca"):
    if kind == "logistic":
        # In the unreduced n_features >> n_samples case, the dual linear solver
        # avoids expensive dense high-dimensional LBFGS iterations.
        if preprocessing == "raw":
            return LogisticRegression(C=c, max_iter=2000, solver="liblinear", dual=True, random_state=seed)
        return LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=seed)
    return SVC(C=c, kernel="rbf", gamma="scale", probability=False, random_state=seed)


def _candidates(config: TrainingConfig) -> list[dict]:
    modes = config.preprocessing_modes if config.pca_components else ("raw",)
    candidates = [{"head": head, "classifier": kind, "C": float(c), "preprocessing": mode}
        for head in config.heads
        for mode in modes
        for kind, values in (("logistic", config.logistic_cs), ("rbf", config.rbf_cs))
        for c in values if mode != "raw" or kind == "logistic"]
    if not candidates:
        raise ValueError("Raw feature mode requires at least one logistic C value")
    return candidates


def _fit(records: list[dict], data: dict, candidate: dict, config: TrainingConfig) -> dict:
    x, y, w = _training_data(records, data, candidate["head"], config)
    scaler, pca, z = _preprocess_fit(x, config, candidate.get("preprocessing", "pca"))
    clf = _classifier(candidate["classifier"], candidate["C"], config.seed, candidate.get("preprocessing", "pca"))
    clf.fit(z, y, sample_weight=w)
    return {"scaler": scaler, "pca": pca, "classifier": clf, "candidate": candidate}


def _score(model: dict, x: np.ndarray) -> np.ndarray:
    z = model["pca"].transform(model["scaler"].transform(x))
    clf = model["classifier"]
    return (clf.predict_proba(z)[:, 1] if hasattr(clf, "predict_proba")
            else expit(clf.decision_function(z))).astype(float)


def _smooth(scores: np.ndarray, times: np.ndarray, seconds: float) -> np.ndarray:
    radius = max(seconds, 0.) / 2
    left = np.searchsorted(times, times - radius, side="left")
    right = np.searchsorted(times, times + radius, side="right")
    prefix = np.concatenate([[0.], np.cumsum(scores)])
    return (prefix[right] - prefix[left]) / np.maximum(right - left, 1)


def _aggregate(scores: np.ndarray, times: np.ndarray, aggregation: str, config: TrainingConfig) -> float:
    if len(scores) == 1:
        return float(scores[0])
    if aggregation == "smooth_max":
        return float(_smooth(scores, times, config.smoothing_seconds).max())
    if aggregation == "topk":
        return float(np.sort(scores)[-min(config.top_k, len(scores)):].mean())
    return float(scores.max())


def classification_metrics(labels: Iterable[int], scores: Iterable[float], predictions: Iterable[int]) -> dict:
    y, s, p = np.asarray(list(labels)), np.asarray(list(scores)), np.asarray(list(predictions))
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    both = len(set(y)) == 2
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y, p, zero_division=0)),
        "f2": float(fbeta_score(y, p, beta=2, zero_division=0)),
        "average_precision": float(average_precision_score(y, s)) if both else None,
        "roc_auc": float(roc_auc_score(y, s)) if both else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "confusion_order": [["TN", "FP"], ["FN", "TP"]]}


def _threshold(labels: np.ndarray, scores: np.ndarray, config: TrainingConfig):
    unique = np.unique(scores)
    thresholds = np.unique(np.r_[0., .5, (unique[:-1] + unique[1:]) / 2, np.nextafter(unique.max(), np.inf)])
    best = None
    for threshold in thresholds:
        metrics = classification_metrics(labels, scores, scores >= threshold)
        key = (metrics[config.primary_metric], metrics["accuracy"], metrics["recall"], -abs(threshold - .5))
        if best is None or key > best[0]:
            best = key, float(threshold), metrics
    return best[1], best[2]


def _select(records: list[dict], data: dict, config: TrainingConfig) -> dict:
    """Select on inner OOF scores. No held-out outer record enters this method."""
    candidates = _candidates(config)
    splits = grouped_splits(records, config.inner_folds, config.seed + 1)
    if not splits:
        # Preserve the predeclared candidate order, skipping heads that cannot
        # train (e.g. binary positive videos without point annotations).
        feasible_heads = set()
        for head in config.heads:
            try:
                _training_data(records, data, head, config)
                feasible_heads.add(head)
            except ValueError:
                pass
        feasible = [c for c in candidates if c["head"] in feasible_heads]
        if not feasible:
            raise ValueError("No head has labeled training examples of both classes")
        candidate = feasible[0]
        aggregation = config.aggregations[0] if candidate["head"] == "point" else "max"
        return {"candidate": candidate, "aggregation": aggregation, "threshold": .5,
                "selection_metrics": None, "selection_folds": [],
                "note": "Too few class groups for inner CV; fixed first feasible candidate and threshold 0.5",
                "leaderboard": []}
    raw = {i: {} for i in range(len(candidates))}
    failed = set()
    split_details = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        train = [records[i] for i in train_idx]
        val = [records[i] for i in val_idx]
        split_details.append({"fold": fold, "train_groups": sorted({r["group_id"] for r in train}),
                              "validation_groups": sorted({r["group_id"] for r in val})})
        # The scaler/PCA are fitted once per head/split and reused across Cs.
        for head in config.heads:
            try:
                x, y, weights = _training_data(train, data, head, config)
            except ValueError:
                failed.update(i for i, c in enumerate(candidates) if c["head"] == head)
                continue
            for mode in (config.preprocessing_modes if config.pca_components else ("raw",)):
                scaler, pca, z = _preprocess_fit(x, config, mode)
                val_z = {r["sha256"]: pca.transform(scaler.transform(data[r["sha256"]][head])) for r in val}
                for i, candidate in enumerate(candidates):
                    if candidate["head"] != head or candidate["preprocessing"] != mode:
                        continue
                    clf = _classifier(candidate["classifier"], candidate["C"], config.seed, candidate["preprocessing"])
                    clf.fit(z, y, sample_weight=weights)
                    for r in val:
                        f = val_z[r["sha256"]]
                        s = clf.predict_proba(f)[:, 1] if hasattr(clf, "predict_proba") else expit(clf.decision_function(f))
                        raw[i][r["sha256"]] = s
    labels = np.asarray([r["label"] for r in records])
    leaderboard = []
    for i, candidate in enumerate(candidates):
        if i in failed or len(raw[i]) != len(records):
            continue
        aggregations = config.aggregations if candidate["head"] == "point" else ("max",)
        for aggregation in aggregations:
            scores = np.asarray([_aggregate(raw[i][r["sha256"]], data[r["sha256"]]["timestamps"], aggregation, config)
                                 for r in records])
            threshold, metrics = _threshold(labels, scores, config)
            leaderboard.append({"candidate": candidate, "aggregation": aggregation,
                                "threshold": threshold, "selection_metrics": metrics})
    if not leaderboard:
        raise ValueError("No valid inner-CV model; check label/point availability")
    # Stable ties prefer earlier/simpler predefined candidates, never outer scores.
    leaderboard.sort(key=lambda x: (x["selection_metrics"][config.primary_metric],
                                    x["selection_metrics"]["accuracy"],
                                    x["selection_metrics"]["recall"]), reverse=True)
    return {**leaderboard[0], "selection_folds": split_details,
            "note": "Inner grouped OOF selection; this score is development, not an unbiased test result",
            "leaderboard": leaderboard}


def _events(scores: np.ndarray, times: np.ndarray, threshold: float, config: TrainingConfig) -> list[dict]:
    smooth = _smooth(scores, times, config.smoothing_seconds)
    eligible = np.flatnonzero(smooth >= threshold)
    # A max/top-k verdict can be positive when smoothing is below its threshold.
    if len(eligible) == 0 and len(scores):
        eligible = np.array([int(scores.argmax())])
    # Merge a continuous excursion (and short gaps) into one event, rather than
    # emitting another event every few seconds throughout a long action.
    clusters = np.split(eligible, np.flatnonzero(np.diff(times[eligible]) > config.event_merge_seconds) + 1)
    peaks = [int(cluster[np.argmax(smooth[cluster])]) for cluster in clusters if len(cluster)]
    ranked = sorted(peaks, key=lambda i: float(smooth[i]), reverse=True)[:config.max_events]
    chosen = []
    for idx in ranked:
        t = float(times[idx])
        chosen.append({"timestamp_sec": t, "timecode": _timecode(t),
                       "score": float(scores[idx]), "type": "potential_violation",
                       "explanation": "Visual candidate learned from binary labels; semantic event type and legal basis require separate annotation/review."})
    return sorted(chosen, key=lambda e: e["timestamp_sec"])


def _timecode(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def _prediction(model: dict, entry: dict, selection: dict, config: TrainingConfig) -> dict:
    head = model["candidate"]["head"]
    raw = _score(model, entry[head])
    score = _aggregate(raw, entry["timestamps"], selection["aggregation"], config)
    predicted = int(score >= selection["threshold"])
    # Pooled models provide only a video verdict and cannot localize an event.
    events = _events(raw, entry["timestamps"], selection["threshold"], config) if predicted and head == "point" else []
    return {"score": score, "prediction": predicted, "threshold": selection["threshold"],
            "verdict": "violation" if predicted else "no_violation", "events": events,
            "score_calibrated": False, "localization_available": head == "point",
            "window_scores": raw.tolist() if head == "point" else [],
            "window_timestamps": entry["timestamps"].astype(float).tolist() if head == "point" else []}


def _localization_metrics(predictions: list[dict]) -> dict:
    errors, matched = [], {1: 0, 2: 0, 5: 0}
    total, localized = 0, 0
    for r in predictions:
        if r["label"] != 1:
            continue
        anchors = r["event_times"]
        total += len(anchors)
        event_times = [e["timestamp_sec"] for e in r["events"]]
        if anchors and event_times:
            localized += len(anchors)
            # Per-anchor nearest errors are diagnostic only, not event-matching AP.
            errors.extend(min(abs(a - e) for e in event_times) for a in anchors)
        for tolerance in matched:
            matched[tolerance] += _match_points(anchors, event_times, tolerance)
    return {"annotated_points": total, "points_with_any_prediction": localized,
        "median_error_sec_conditional": float(np.median(errors)) if errors else None,
        "p90_error_sec_conditional": float(np.percentile(errors, 90)) if errors else None,
        "point_recall_at_tolerance": {f"{t}s": matched[t] / total if total else None for t in matched},
        "note": "Unofficial point diagnostics; error conditional on a detected event. Missing detections count against recall. Unmarked true events are unknown, so event precision and interval IoU are not identifiable."}


def _match_points(anchors: list[float], event_times: list[float], tolerance: float) -> int:
    """Maximum-cardinality one-to-one assignment, with distance tie-breaking."""
    if not anchors or not event_times:
        return 0
    distance = np.abs(np.asarray(anchors)[:, None] - np.asarray(event_times)[None])
    # One forbidden assignment must cost more than all possible valid distances.
    # Hungarian assignment therefore first maximizes valid cardinality and then
    # minimizes the total timing error, unlike nearest-pair greedy matching.
    penalty = (min(distance.shape) + 1) * (max(tolerance, 0.) + 1.)
    cost = np.where(distance <= tolerance, distance, penalty)
    row, col = linear_sum_assignment(cost)
    return int(np.sum(distance[row, col] <= tolerance))


def _write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def semantic_feature_config(metadata: dict) -> dict:
    """Separate model compatibility from device/library provenance.

    Cache reuse remains strict in the extractor. A trained feature classifier
    can nevertheless move from CUDA fp16 extraction to local CPU fp32 inference
    when the weights, input geometry and sampling semantics remain identical.
    """
    keys = ("schema_version", "backbone", "image_size", "views", "preprocessing",
            "decoder", "weights_sha256", "hub_repo", "sample_fps")
    result = {key: metadata[key] for key in keys if key in metadata}
    return result if result else dict(metadata)


def run_cross_validation(records: list[dict], feature_dir: str | Path, output_dir: str | Path,
                         config: TrainingConfig | None = None) -> dict:
    """Run honest outer OOF evaluation of the complete inner model-selection policy.

    The outer folds estimate the predefined selection procedure. Choosing a new
    candidate family after inspecting these scores requires a new external test.
    """
    config = config or TrainingConfig()
    records = _canonical_records(records)
    splits = grouped_splits(records, config.outer_folds, config.seed)
    if not splits:
        raise ValueError("Need at least two independent groups of each class for outer CV")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _prepare(records, feature_dir, config)
    predictions, folds = [], []
    comparisons: dict[str, list] = {head: [] for head in config.heads}
    for fold, (train_idx, test_idx) in enumerate(splits):
        train = [records[i] for i in train_idx]
        test = [records[i] for i in test_idx]
        print(f"Outer fold {fold + 1}/{len(splits)}: train={len(train)}, test={len(test)}", flush=True)
        selection = _select(train, data, config)
        selected_model = _fit(train, data, selection["candidate"], config)
        fold_predictions = []
        for r in test:
            p = _prediction(selected_model, data[r["sha256"]], selection, config)
            row = {"video_id": r.get("video_id", r["sha256"]), "sha256": r["sha256"],
                   "group_id": r["group_id"], "label": r["label"], "event_times": r["event_times"],
                   "fold": fold, "selected_model": selection["candidate"], **p}
            predictions.append(row)
            fold_predictions.append(row)
        # Separate outer-CV baselines: each head's settings still selected only inside train.
        for head in config.heads:
            head_options = [x for x in selection["leaderboard"] if x["candidate"]["head"] == head]
            if not head_options:
                if selection["candidate"]["head"] == head:
                    head_selection = selection
                else:
                    continue
            else:
                head_selection = head_options[0]
            model = selected_model if head_selection["candidate"] == selection["candidate"] else _fit(train, data, head_selection["candidate"], config)
            for r in test:
                comparisons[head].append({"label": r["label"], "event_times": r["event_times"],
                    "sha256": r["sha256"], "video_id": r.get("video_id", r["sha256"]), "fold": fold,
                    **_prediction(model, data[r["sha256"]], head_selection, config)})
        fold_metrics = classification_metrics([r["label"] for r in fold_predictions],
                         [r["score"] for r in fold_predictions], [r["prediction"] for r in fold_predictions])
        folds.append({"fold": fold, "train_groups": sorted({r["group_id"] for r in train}),
                      "test_groups": sorted({r["group_id"] for r in test}),
                      "selection": selection, "metrics": fold_metrics})
        _write_json(output_dir / "cv_progress.json", {"completed_folds": len(folds), "folds": folds, "predictions": predictions})
    metrics = classification_metrics([r["label"] for r in predictions], [r["score"] for r in predictions], [r["prediction"] for r in predictions])
    metrics["localization"] = _localization_metrics(predictions)
    # Fold-dependent SVM/logistic score scales limit interpretation of pooled AP/AUC.
    report = {"evaluation": "nested_grouped_out_of_fold", "config": asdict(config),
              "metrics": metrics, "predictions": predictions, "folds": folds,
              "model_comparisons": {head: {**classification_metrics([r["label"] for r in rows],
                  [r["score"] for r in rows], [r["prediction"] for r in rows]),
                  "localization": _localization_metrics(rows)} for head, rows in comparisons.items() if rows},
              "head_predictions": comparisons,
              "limitations": ["Only the supplied staged corpus is evaluated; no independent new-angle test.",
                  "Group exact/perceptual duplicate takes and filming sessions before claiming generalization.",
                  "Binary point labels cannot supervise legal event types or exhaustive event boundaries.",
                  "Aggregate AP/AUC combine uncalibrated scores from different fitted outer models.",
                  "Outer OOF accuracy evaluates this fixed model-selection policy; later tuning to it makes it development data."],
              "created_at": datetime.now(timezone.utc).isoformat()}
    _write_json(output_dir / "cv_report.json", report)
    print(json.dumps({"evaluation": report["evaluation"], "metrics": metrics}, indent=2), flush=True)
    return report


def fit_final(records: list[dict], feature_dir: str | Path, output_dir: str | Path,
              config: TrainingConfig | None = None) -> Path:
    """Choose settings with grouped development CV, then train on all supplied data."""
    config = config or TrainingConfig()
    records = _canonical_records(records)
    data = _prepare(records, feature_dir, config)
    selection = _select(records, data, config)
    model = _fit(records, data, selection["candidate"], config)
    first = data[records[0]["sha256"]]
    artifact = {"format_version": 1, "model": model, "config": asdict(config),
        "selection": selection, "feature_config": first["metadata"], "feature_shape": first["feature_shape"],
        "feature_semantic_config": semantic_feature_config(first["metadata"]),
        "provenance": {"created_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "training_sha256": [r["sha256"] for r in records],
            "training_groups": sorted({r["group_id"] for r in records}),
            "annotation_mode": "binary_video_labels_and_positive_points",
            "selection_score_is_test_score": False, "llm_api_used": False},
        "limitations": "Binary visual model; scores uncalibrated; no semantic event types, legal interpretation or accuracy guarantee."}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model.joblib"
    point_selection = (selection if selection["candidate"]["head"] == "point" else
        next((row for row in selection["leaderboard"] if row["candidate"]["head"] == "point"), None))
    point_path = path if selection["candidate"]["head"] == "point" else None
    if point_selection is not None and point_path is None:
        point_model = _fit(records, data, point_selection["candidate"], config)
        point_artifact = {**artifact, "model": point_model, "selection": point_selection,
                          "artifact_role": "point_localization_model_selected_by_inner_cv"}
        point_path = output_dir / "model_point.joblib"
        joblib.dump(point_artifact, point_path, compress=3)
    artifact["artifact_role"] = "primary_binary_model_selected_by_inner_cv"
    artifact["point_model_filename"] = point_path.name if point_path else None
    joblib.dump(artifact, path, compress=3)
    _write_json(output_dir / "final_selection.json", {"selection": selection,
                "provenance": artifact["provenance"], "limitations": artifact["limitations"],
                "model_files": {"primary": path.name, "point": point_path.name if point_path else None},
                "point_selection": point_selection})
    return path


def predict_features(feature_path: str | Path, model_path: str | Path) -> dict:
    """Inference receives an unlabeled feature cache, never annotation anchors.

    Only load joblib artifacts produced by a trusted training run.
    """
    artifact = joblib.load(model_path)
    config = TrainingConfig(**artifact["config"])
    f, t, meta = load_features(feature_path)
    training_meta = artifact["feature_config"]
    trained_semantics = artifact.get("feature_semantic_config", semantic_feature_config(training_meta))
    if tuple(f.shape[1:]) != tuple(artifact["feature_shape"]) or semantic_feature_config(meta) != trained_semantics:
        raise ValueError("Inference feature shape or semantic extractor configuration differs from training")
    head = artifact["model"]["candidate"]["head"]
    entry = {"timestamps": t, head: window_features(f, t, config.window_radii) if head == "point" else pooled_features(f)}
    result = _prediction(artifact["model"], entry, artifact["selection"], config)
    different = {key: {"training": training_meta.get(key), "inference": meta.get(key)}
                 for key in sorted(set(training_meta) | set(meta))
                 if training_meta.get(key) != meta.get(key)}
    result["feature_provenance_differences"] = different
    result["inference_warnings"] = (["Feature extraction device/library provenance differs from training; semantic configuration matches. Validate precision portability on representative videos."] if different else [])
    return result

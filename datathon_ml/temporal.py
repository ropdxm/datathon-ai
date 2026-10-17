"""Observable semantic states, conservative temporal rules and grouped learning.

Inputs are fixed numeric state scores, not images, text descriptions, identifiers
or annotation timestamps. Rules propose observable events for review; they do
not establish legal violations. A small video classifier learns binary labels
using nested grouped validation. The frozen rules are reported separately.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import sklearn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from .model import (TrainingConfig, _canonical_records, _localization_metrics,
                    _threshold, classification_metrics, grouped_splits,
                    load_features, semantic_feature_config)


CONCEPT_NAMES = (
    "cashier_present", "customer_present", "loose_cash_visible", "cash_handover",
    "cash_counting", "cash_secured", "access_open", "phone_interaction",
    "object_to_pocket", "document_check", "transaction_active", "transaction_start",
    "transaction_end", "workstation_departure", "scene_clear", "hands_visible",
    "cash_area_visible", "bag_or_seal_handling",
)
RULE_NAMES = ("unattended_cash", "departure_during_service", "unsecured_leave",
              "handover_without_visible_count")
EXPOSURE_LENGTH_LIMITATION = (
    "Even without a video-duration column, capped state/visibility dwell, run counts "
    "and max/top-mean summaries can encode observation or exposure length; validate "
    "matched-length recordings and independent sessions/cameras."
)


@dataclass(frozen=True)
class TemporalConfig:
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    logistic_cs: tuple[float, ...] = (.01, .1, 1.)
    calibrate: bool = True
    primary_metric: str = "balanced_accuracy"
    on_threshold: float = .7
    off_threshold: float = .55
    visibility_threshold: float = .7
    min_dwell_seconds: float = 2.
    max_gap_seconds: float = 3.
    prior_presence_seconds: float = 30.
    sequence_seconds: float = 15.
    min_cycle_seconds: float = 2.
    max_cycle_seconds: float = 120.
    duration_cap_seconds: float = 12.
    event_merge_seconds: float = 1.
    frozen_rule_threshold: float = .5

    def __post_init__(self):
        if min(self.outer_folds, self.inner_folds) < 2:
            raise ValueError("Grouped folds must be >=2")
        if not self.logistic_cs or any(not np.isfinite(c) or c <= 0 for c in self.logistic_cs):
            raise ValueError("Regularization C values must be positive")
        if not .5 < self.off_threshold < self.on_threshold <= 1:
            raise ValueError("Hysteresis requires .5 < off < on <= 1; unknown .5 must stop a run")
        if not .5 < self.visibility_threshold <= 1 or not 0 <= self.frozen_rule_threshold <= 1:
            raise ValueError("Invalid visibility/rule threshold")
        times = (self.min_dwell_seconds, self.max_gap_seconds, self.prior_presence_seconds,
                 self.sequence_seconds, self.min_cycle_seconds, self.max_cycle_seconds,
                 self.duration_cap_seconds, self.event_merge_seconds)
        if any(not np.isfinite(v) or v < 0 for v in times) or min(self.max_gap_seconds, self.duration_cap_seconds) == 0:
            raise ValueError("Time parameters must be finite and valid")
        if self.max_cycle_seconds < self.min_cycle_seconds:
            raise ValueError("Cycle maximum cannot precede minimum")
        if self.primary_metric not in {"accuracy", "balanced_accuracy", "f1", "f2"}:
            raise ValueError("Unsupported primary metric")


def _validate_states(features: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    t = np.asarray(timestamps, dtype=np.float64)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    if x.ndim != 2 or x.shape != (len(t), len(CONCEPT_NAMES)) or not len(t):
        raise ValueError("Expected semantic scores [T,1,18] and aligned timestamps")
    if not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or t[0] < 0:
        raise ValueError("Scores/timestamps must be finite and times strictly increasing")
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("Semantic scores must be [0,1], with 0.5 representing unknown")
    return x, t


def load_states(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    x, t, metadata = load_features(path)
    names = metadata.get("concept_names")
    if not isinstance(names, list) or len(names) != len(CONCEPT_NAMES) or set(names) != set(CONCEPT_NAMES):
        raise ValueError("Cache must declare the complete fixed concept_names schema")
    if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] != len(names):
        raise ValueError("Semantic cache must be [T,1,18]")
    x = x[:, :, [names.index(name) for name in CONCEPT_NAMES]]
    x, t = _validate_states(x, t)
    return x, t, metadata


def hysteresis(scores: np.ndarray, times: np.ndarray, config: TemporalConfig) -> np.ndarray:
    active = False
    result = np.zeros(len(scores), dtype=bool)
    for i, score in enumerate(scores):
        if i and times[i] - times[i - 1] > config.max_gap_seconds:
            active = False
        if score >= config.on_threshold:
            active = True
        elif score < config.off_threshold:
            active = False
        result[i] = active
    return result


def _runs(mask: np.ndarray, times: np.ndarray, max_gap: float) -> list[tuple[int, int]]:
    spans, start = [], None
    for i, active in enumerate(mask):
        if start is not None and (not active or times[i] - times[i - 1] > max_gap):
            spans.append((start, i - 1))
            start = None
        if active and start is None:
            start = i
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def _duration(mask: np.ndarray, times: np.ndarray, config: TemporalConfig) -> float:
    return float(sum(times[b] - times[a] for a, b in _runs(mask, times, config.max_gap_seconds)))


def _event(rule: str, a: int, b: int, score: np.ndarray, times: np.ndarray, explanation: str) -> dict:
    peak = a + int(np.argmax(score[a:b + 1]))
    return {"type": "observable_review_candidate", "rule_ids": [rule],
        "start_sec": float(times[a]), "end_sec": float(times[b]),
        "timestamp_sec": float(times[peak]), "score": float(np.min(score[a:b + 1])),
        "explanation": explanation, "legal_violation_established": False}


def _merge_events(events: list[dict], seconds: float) -> list[dict]:
    result = []
    for event in sorted(events, key=lambda e: (e["start_sec"], e["end_sec"])):
        components = event.get("component_events", [{key: value for key, value in event.items()
                                                    if key not in {"component_events", "score_scope"}}])
        if result and event["start_sec"] <= result[-1]["end_sec"] + seconds:
            last = result[-1]
            last["end_sec"] = max(last["end_sec"], event["end_sec"])
            last["rule_ids"] = sorted(set(last["rule_ids"] + event["rule_ids"]))
            last["component_events"].extend(dict(component) for component in components)
            if event["score"] > last["score"]:
                last["timestamp_sec"], last["score"] = event["timestamp_sec"], event["score"]
            if event["explanation"] not in last["explanation"]:
                last["explanation"] += " " + event["explanation"]
        else:
            result.append({**event, "component_events": [dict(component) for component in components],
                           "score_scope": "maximum_component_score_not_union_confidence"})
    return result


def _precedence(a: np.ndarray, b: np.ndarray, times: np.ndarray, config: TemporalConfig) -> float:
    result = 0.
    segment = np.r_[0, np.cumsum(np.diff(times) > config.max_gap_seconds)]
    for i in np.flatnonzero(b >= config.on_threshold):
        before = ((times < times[i]) & (times >= times[i] - config.sequence_seconds)
                  & (segment == segment[i]) & (a >= config.on_threshold))
        if before.any():
            result = max(result, min(float(a[before].max()), float(b[i])))
    return result


def transform_states(features: np.ndarray, timestamps: np.ndarray,
                     config: TemporalConfig | None = None, *, continuous_window_coverage: bool = False) -> dict:
    """Fixed descriptors and conservative rules; accepts no labels or anchors.

    A producer must independently certify continuous action coverage to enable
    a missing-count proposal. Ordinary sampled-frame caches do not certify it.
    Unknown state scores (.5), occlusion and sampling gaps block that rule.
    """
    config = config or TemporalConfig()
    x, times = _validate_states(features, timestamps)
    state = {name: x[:, i] for i, name in enumerate(CONCEPT_NAMES)}
    active = {name: hysteresis(score, times, config) for name, score in state.items()}
    absent = hysteresis(1 - state["cashier_present"], times, config)
    clear = state["scene_clear"] >= config.visibility_threshold
    cash_clear = clear & (state["cash_area_visible"] >= config.visibility_threshold)
    full_clear = cash_clear & (state["hands_visible"] >= config.visibility_threshold)
    events, trace = [], []
    rule_signals = {name: np.zeros(len(times), dtype=float) for name in RULE_NAMES}

    def emit_runs(name, mask, signal, explanation):
        for a, b in _runs(mask, times, config.max_gap_seconds):
            dwell = float(times[b] - times[a])
            accepted = dwell >= config.min_dwell_seconds
            trace.append({"rule": name, "start_sec": float(times[a]), "end_sec": float(times[b]),
                          "dwell_seconds": dwell, "emitted": accepted,
                          "reason": "observable_conditions_and_dwell" if accepted else "insufficient_dwell"})
            if accepted:
                rule_signals[name][a:b + 1] = signal[a:b + 1]
                events.append(_event(name, a, b, signal, times, explanation))

    cash_unattended = active["loose_cash_visible"] & absent & cash_clear
    unattended_score = np.minimum.reduce([state["loose_cash_visible"], 1 - state["cashier_present"],
                                         state["scene_clear"], state["cash_area_visible"]])
    emit_runs("unattended_cash", cash_unattended, unattended_score,
              "Loose cash remains visible while the cashier is absent from a clearly visible workstation.")

    previous_presence = np.zeros(len(times), dtype=bool)
    last_present = -np.inf
    for i, t in enumerate(times):
        if i and t - times[i - 1] > config.max_gap_seconds:
            last_present = -np.inf
        previous_presence[i] = t - last_present <= config.prior_presence_seconds
        if active["cashier_present"][i] and clear[i]:
            last_present = t
    service = active["transaction_active"] | active["loose_cash_visible"]
    departure_mask = absent & clear & service & previous_presence
    departure_score = np.minimum.reduce([1 - state["cashier_present"], state["scene_clear"],
        np.maximum(state["transaction_active"], state["loose_cash_visible"])])
    emit_runs("departure_during_service", departure_mask, departure_score,
              "The cashier was visible, then left while visible transaction activity or loose cash remained.")

    unsecured = departure_mask & active["access_open"] & active["loose_cash_visible"] & cash_clear
    # A clear positive unsecured-access state is required; lack of a visible
    # cash-securing action alone never establishes unsecured cash.
    unsecured_score = np.minimum.reduce([departure_score, state["access_open"], state["loose_cash_visible"], state["cash_area_visible"]])
    emit_runs("unsecured_leave", unsecured, unsecured_score,
              "The cashier departs while loose cash and an open cash-access area remain clearly visible.")

    cycle_start = None
    prior_start, prior_end = False, False
    for i in range(len(times)):
        start_now, end_now = bool(active["transaction_start"][i]), bool(active["transaction_end"][i])
        if start_now and not prior_start:
            cycle_start = i
        if end_now and not prior_end and cycle_start is not None and i > cycle_start:
            a, b = cycle_start, i
            cycle_seconds = float(times[b] - times[a])
            region = slice(a, b + 1)
            handovers = np.flatnonzero(active["cash_handover"][region]) + a
            reason = None
            if not continuous_window_coverage:
                reason = "continuous_action_coverage_not_certified"
            elif not config.min_cycle_seconds <= cycle_seconds <= config.max_cycle_seconds:
                reason = "incomplete_or_unsupported_cycle_duration"
            elif not full_clear[region].all() or np.any(np.diff(times[region]) > config.max_gap_seconds):
                reason = "cycle_visibility_or_sampling_gap"
            elif not len(handovers):
                reason = "no_observed_handover"
            elif np.any(state["cash_counting"][region] > 1 - config.on_threshold):
                reason = "counting_observed_or_uncertain"
            trace.append({"rule": "handover_without_visible_count", "start_sec": float(times[a]),
                "end_sec": float(times[b]), "emitted": reason is None,
                "reason": reason or "full_observed_cycle_and_explicit_absence"})
            if reason is None:
                h = int(handovers[np.argmax(state["cash_handover"][handovers])])
                confidence = float(min(state["transaction_start"][a], state["transaction_end"][b],
                    state["cash_handover"][h], state["scene_clear"][region].min(),
                    state["hands_visible"][region].min(), state["cash_area_visible"][region].min(),
                    (1 - state["cash_counting"][region]).min()))
                rule_signals["handover_without_visible_count"][h] = confidence
                event = _event("handover_without_visible_count", h, h,
                    rule_signals["handover_without_visible_count"], times,
                    "A complete, independently certified visible transaction cycle includes a handover without visible counting; review procedural exceptions.")
                events.append(event)
            cycle_start = None
        prior_start, prior_end = start_now, end_now

    values, names = [], []

    def add(name, value):
        names.append(name)
        values.append(float(value))

    # Only durations of observable states are used, capped to avoid introducing
    # an uncapped recording-length feature. No filename, ID or absolute time.
    supported = np.diff(times) <= config.max_gap_seconds
    support_seconds = float(np.diff(times)[supported].sum()) if len(times) > 1 else 0.
    for name in CONCEPT_NAMES:
        score, mask = state[name], active[name]
        spans = _runs(mask, times, config.max_gap_seconds)
        dwell = max((times[b] - times[a] for a, b in spans), default=0.)
        add(name + ".max", score.max())
        add(name + ".top3mean", np.sort(score)[-min(3, len(score)):].mean())
        add(name + ".mean", score.mean())
        add(name + ".active_fraction", _duration(mask, times, config) / support_seconds if support_seconds else float(mask.mean()))
        add(name + ".dwell_capped", min(dwell, config.duration_cap_seconds) / config.duration_cap_seconds)
        add(name + ".change", np.abs(np.diff(score))[supported].mean() if supported.any() else 0.)
        add(name + ".runs_capped", min(len(spans), 3) / 3.)
        add(name + ".unknown_fraction", np.mean(np.abs(score - .5) < .1))
    for name in RULE_NAMES:
        signal = rule_signals[name]
        mask = signal > 0
        add("rule." + name + ".max", signal.max())
        add("rule." + name + ".top3mean", np.sort(signal)[-min(3, len(signal)):].mean())
        add("rule." + name + ".dwell_capped", min(_duration(mask, times, config), config.duration_cap_seconds) / config.duration_cap_seconds)
    for a, b in (("cash_counting", "cash_handover"), ("cash_handover", "cash_secured"),
                 ("document_check", "cash_handover"), ("cashier_present", "workstation_departure")):
        add(a + ".before." + b, _precedence(state[a], state[b], times, config))
    add("phone_and_cash", np.minimum(state["phone_interaction"], state["loose_cash_visible"]).max())
    add("pocket_and_cash", np.minimum(state["object_to_pocket"], state["loose_cash_visible"]).max())
    rule_scores = {name: float(rule_signals[name].max()) for name in RULE_NAMES}
    return {"features": np.asarray(values, dtype=np.float32), "feature_names": names,
        "events": _merge_events(events, config.event_merge_seconds), "rule_scores": rule_scores,
        "rule_score": max(rule_scores.values()), "trace": trace,
        "continuous_window_coverage": bool(continuous_window_coverage),
        "interpretation": "Observable review candidates; legal violation and regulatory exceptions require separate review."}


def _prepare(records, feature_dir, config):
    result, expected = {}, None
    for record in records:
        x, t, metadata = load_states(Path(feature_dir) / f"{record['sha256']}.npz")
        if expected is None:
            expected = metadata
        if metadata != expected:
            raise ValueError("Training requires one consistent semantic extractor configuration")
        result[record["sha256"]] = {"states": x, "times": t, "metadata": metadata,
            "derived": transform_states(x, t, config, continuous_window_coverage=metadata.get("continuous_window_coverage") is True)}
    return result


def _fit(records, data, c):
    x = np.stack([data[r["sha256"]]["derived"]["features"] for r in records])
    y = np.asarray([r["label"] for r in records])
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs", max_iter=2000)
    classifier.fit(scaler.transform(x), y)
    return {"scaler": scaler, "classifier": classifier, "C": c,
            "fit_sha256": [r["sha256"] for r in records]}


def _logits(model, records, data):
    x = np.stack([data[r["sha256"]]["derived"]["features"] for r in records])
    return model["classifier"].decision_function(model["scaler"].transform(x))


def _probabilities(logits, calibrator):
    logits = np.asarray(logits)
    return expit(logits) if calibrator is None else calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]


def _metric_config(config):
    return TrainingConfig(seed=config.seed, outer_folds=config.outer_folds, inner_folds=config.inner_folds,
                          primary_metric=config.primary_metric)


def _select(records, data, config):
    splits = grouped_splits(records, config.inner_folds, config.seed + 1)
    if not splits:
        return {"C": config.logistic_cs[0], "threshold": .5, "calibrator": None,
                "selection_metrics": None, "selection_folds": [], "leaderboard": [],
                "calibration_method": "none_insufficient_groups", "calibration_fit_sha256": []}
    labels = np.asarray([r["label"] for r in records])
    candidate_logits = {c: np.zeros(len(records), dtype=float) for c in config.logistic_cs}
    folds = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        train, val = [records[i] for i in train_idx], [records[i] for i in val_idx]
        folds.append({"fold": fold, "train_groups": sorted({r["group_id"] for r in train}),
                      "validation_groups": sorted({r["group_id"] for r in val}),
                      "preprocessing_fit_sha256": [r["sha256"] for r in train]})
        for c in config.logistic_cs:
            model = _fit(train, data, c)
            candidate_logits[c][val_idx] = _logits(model, val, data)
    rows = []
    for c in config.logistic_cs:
        raw = candidate_logits[c]
        calibrator = LogisticRegression(C=1., solver="lbfgs", max_iter=1000).fit(raw[:, None], labels) if config.calibrate else None
        scores = _probabilities(raw, calibrator)
        threshold, metrics = _threshold(labels, scores, _metric_config(config))
        rows.append({"C": c, "threshold": threshold, "calibrator": calibrator, "selection_metrics": metrics})
    rows.sort(key=lambda row: (row["selection_metrics"][config.primary_metric], row["selection_metrics"]["accuracy"],
                              row["selection_metrics"]["recall"]), reverse=True)
    return {**rows[0], "selection_folds": folds,
            "leaderboard": [{k: v for k, v in row.items() if k != "calibrator"} for row in rows],
            "calibration_method": "sigmoid_fit_on_training_group_oof_logits" if config.calibrate else "none",
            "calibration_fit_sha256": [r["sha256"] for r in records] if config.calibrate else [],
            "note": "Calibration and threshold use outer-training labels only; inner selection scores are development scores."}


def _selection_json(selection):
    return {key: value for key, value in selection.items() if key != "calibrator"}


def _prediction(model, derived, selection):
    x = derived["features"][None]
    raw = model["classifier"].decision_function(model["scaler"].transform(x))
    score = float(_probabilities(raw, selection["calibrator"])[0])
    predicted = int(score >= selection["threshold"])
    return {"score": score, "prediction": predicted, "verdict": "violation" if predicted else "no_violation",
        "threshold": selection["threshold"], "calibration_method": selection["calibration_method"],
        "calibration_validated_independently": False, "events": derived["events"] if predicted else [],
        "localization_available": bool(derived["events"]), "rule_events": derived["events"],
        "rule_score": derived["rule_score"], "rule_scores": derived["rule_scores"], "rule_trace": derived["trace"],
        "legal_violation_established": False}


def _metrics(rows):
    result = classification_metrics([r["label"] for r in rows], [r["score"] for r in rows], [r["prediction"] for r in rows])
    result["brier_score"] = float(brier_score_loss([r["label"] for r in rows], [r["score"] for r in rows]))
    result["localization"] = _localization_metrics(rows)
    return result


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _assignment_digest(assignment: dict[str, int]) -> str:
    encoded = json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frozen_outer_splits(records: list[dict], source: str | Path | dict,
                        config: TemporalConfig) -> tuple[list, dict]:
    """Load a fixed historical OOF assignment without regenerating its folds.

    A seed alone does not guarantee identical StratifiedGroupKFold partitions
    across scikit-learn versions. This contract matches each exact source SHA
    and validates current grouping before any preprocessing/model fitting.
    """
    if isinstance(source, (str, Path)):
        path = Path(source).resolve(strict=True)
        raw = path.read_bytes()
        report = json.loads(raw)
        source_path = str(path)
    elif isinstance(source, dict):
        report = source
        raw = json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
        source_path = None
    else:
        raise TypeError("folds_from must be an existing CV report path or report dictionary")
    predictions = report.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Frozen fold report needs a nonempty predictions list")
    expected = {record["sha256"]: record for record in records}
    if len(expected) != len(records):
        raise ValueError("Canonicalize duplicate recordings before freezing folds")
    assignment = {}
    for row in predictions:
        key, fold = row.get("sha256"), row.get("fold")
        if not isinstance(key, str) or not isinstance(fold, int) or isinstance(fold, bool) or fold < 0:
            raise ValueError("Frozen predictions need sha256 and nonnegative integer fold")
        if key in assignment:
            raise ValueError("Frozen fold report repeats a source SHA")
        assignment[key] = fold
        if key in expected and "label" in row and row["label"] != expected[key]["label"]:
            raise ValueError("Frozen fold reference labels differ from the current manifest")
    if set(assignment) != set(expected):
        raise ValueError("Frozen fold SHA coverage differs from the current canonical manifest")
    fold_ids = sorted(set(assignment.values()))
    if fold_ids != list(range(len(fold_ids))) or len(fold_ids) != config.outer_folds:
        raise ValueError("Frozen folds must be contiguous from zero and match configured outer_folds")
    reference_seed = report.get("config", {}).get("seed")
    if reference_seed is not None and reference_seed != config.seed:
        raise ValueError("Frozen fold reference seed differs from configured seed")
    grouped = {}
    for record in records:
        group = record.get("group_id", record["sha256"])
        grouped.setdefault(group, set()).add(assignment[record["sha256"]])
    if any(len(folds) != 1 for folds in grouped.values()):
        raise ValueError("Frozen fold assignment splits a current recording group")
    splits = []
    for fold in fold_ids:
        # Stable source order keeps fitting reproducible when a manifest is reordered.
        train = sorted((i for i, r in enumerate(records) if assignment[r["sha256"]] != fold), key=lambda i: records[i]["sha256"])
        test = sorted((i for i, r in enumerate(records) if assignment[r["sha256"]] == fold), key=lambda i: records[i]["sha256"])
        if any(len({records[i]["label"] for i in part}) != 2 for part in (train, test)):
            raise ValueError("Every frozen train/test fold must contain both classes")
        splits.append((np.asarray(train), np.asarray(test)))
    provenance = {"kind": "frozen_report", "path": source_path,
        "file_sha256": hashlib.sha256(raw).hexdigest(), "reference_seed": reference_seed,
        "reference_evaluation": report.get("evaluation"),
        "fold_assignment_sha256": _assignment_digest(assignment)}
    return splits, provenance


def run_temporal_cross_validation(records: list[dict], feature_dir: str | Path,
                                  output_dir: str | Path, config: TemporalConfig | None = None,
                                  *, folds_from: str | Path | dict | None = None) -> dict:
    config = config or TemporalConfig()
    records = _canonical_records(records)
    if folds_from is None:
        splits = grouped_splits(records, config.outer_folds, config.seed)
        assignment = {records[i]["sha256"]: fold for fold, (_, test) in enumerate(splits) for i in test}
        fold_source = {"kind": "generated", "seed": config.seed, "sklearn_version": sklearn.__version__,
                       "fold_assignment_sha256": _assignment_digest(assignment)}
    else:
        splits, fold_source = frozen_outer_splits(records, folds_from, config)
    if not splits:
        raise ValueError("Need >=2 independent groups of each class for temporal outer CV")
    data = _prepare(records, feature_dir, config)
    predictions, folds = [], []
    rule_rows = []
    for record in records:
        derived = data[record["sha256"]]["derived"]
        score = derived["rule_score"]
        predicted = int(score >= config.frozen_rule_threshold)
        rule_rows.append({"label": record["label"], "sha256": record["sha256"], "event_times": record["event_times"],
            "score": score, "prediction": predicted, "events": derived["events"] if predicted else []})
    for fold, (train_idx, test_idx) in enumerate(splits):
        train, test = [records[i] for i in train_idx], [records[i] for i in test_idx]
        print(f"Temporal outer fold {fold + 1}/{len(splits)}: train={len(train)}, test={len(test)}", flush=True)
        selection = _select(train, data, config)
        model = _fit(train, data, selection["C"])
        rows = []
        for record in test:
            result = _prediction(model, data[record["sha256"]]["derived"], selection)
            row = {"sha256": record["sha256"], "video_id": record.get("video_id", record["sha256"]),
                   "group_id": record["group_id"], "label": record["label"], "event_times": record["event_times"],
                   "fold": fold, **result}
            rows.append(row)
            predictions.append(row)
        folds.append({"fold": fold, "train_groups": sorted({r["group_id"] for r in train}),
                      "test_groups": sorted({r["group_id"] for r in test}), "preprocessing_fit_sha256": model["fit_sha256"],
                      "test_sha256": sorted(r["sha256"] for r in test),
                      "test_sha256_digest": hashlib.sha256("\n".join(sorted(r["sha256"] for r in test)).encode("ascii")).hexdigest(),
                      "selection": _selection_json(selection), "metrics": _metrics(rows)})
        _write(Path(output_dir) / "temporal_cv_progress.json", {"completed_folds": len(folds), "folds": folds, "predictions": predictions})
    report = {"evaluation": "nested_grouped_semantic_temporal_development", "config": asdict(config),
        "fold_source": fold_source,
        "metrics": _metrics(predictions), "frozen_rules": {"threshold": config.frozen_rule_threshold,
            "metrics": _metrics(rule_rows), "predictions": rule_rows}, "predictions": predictions, "folds": folds,
        "created_at": datetime.now(timezone.utc).isoformat(), "concept_names": list(CONCEPT_NAMES),
        "limitations": ["State extraction and this rule/model family were designed after prior development experiments; an independent test remains required.",
            "Binary labels supervise video classification; point annotations are used for localization diagnostics only.",
            "Observable rule candidates do not establish legal violations or resolve regulatory exceptions.",
            "Missing-count rule is disabled without independently certified continuous action coverage.",
            "Calibration/threshold selection scores are development only; outer scores mix different fitted calibrators.",
            EXPOSURE_LENGTH_LIMITATION,
            "No accuracy target is established by implementation or synthetic tests."]}
    _write(Path(output_dir) / "temporal_cv_report.json", report)
    print(json.dumps({"evaluation": report["evaluation"], "metrics": report["metrics"],
                      "frozen_rule_metrics": report["frozen_rules"]["metrics"]}, indent=2), flush=True)
    return report


def fit_temporal_final(records: list[dict], feature_dir: str | Path, output_dir: str | Path,
                       config: TemporalConfig | None = None) -> Path:
    config = config or TemporalConfig()
    records = _canonical_records(records)
    data = _prepare(records, feature_dir, config)
    selection = _select(records, data, config)
    model = _fit(records, data, selection["C"])
    first = data[records[0]["sha256"]]
    artifact = {"format_version": 1, "kind": "semantic_temporal", "model": model,
        "selection": selection, "config": asdict(config), "concept_names": list(CONCEPT_NAMES),
        "feature_names": first["derived"]["feature_names"], "feature_config": first["metadata"],
        "feature_semantic_config": semantic_feature_config(first["metadata"]),
        "continuous_window_coverage": first["metadata"].get("continuous_window_coverage") is True,
        "coverage_certification": first["metadata"].get("coverage_certification"),
        "limitations": ["Development classifier; no independent accuracy guarantee or established legal event classification.",
                        EXPOSURE_LENGTH_LIMITATION],
        "provenance": {"created_at": datetime.now(timezone.utc).isoformat(), "training_sha256": model["fit_sha256"],
            "training_groups": sorted({r["group_id"] for r in records}), "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "temporal_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "annotation_mode": "binary_video_labels_points_only_for_evaluation", "selection_score_is_test_score": False,
            "external_inference_api_used": False}}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model_temporal.joblib"
    joblib.dump(artifact, path, compress=3)
    _write(output_dir / "temporal_final_selection.json", {"selection": _selection_json(selection),
           "provenance": artifact["provenance"], "limitations": artifact["limitations"]})
    return path


def predict_temporal_features(feature_path: str | Path, model_path: str | Path) -> dict:
    artifact = joblib.load(model_path)  # Load only trusted model artifacts.
    if artifact.get("kind") != "semantic_temporal":
        raise ValueError("Expected a semantic_temporal artifact")
    provenance = artifact.get("provenance", {})
    saved_codes = [provenance[key] for key in ("code_sha256", "temporal_code_sha256") if key in provenance]
    current_code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not saved_codes or any(saved_code != current_code for saved_code in saved_codes):
        raise ValueError("Temporal rule/feature code fingerprint differs from the artifact; use matching code or refit")
    x, times, metadata = load_states(feature_path)
    if semantic_feature_config(metadata) != artifact["feature_semantic_config"]:
        raise ValueError("Inference semantic extraction protocol differs from training")
    trained_coverage = artifact.get("continuous_window_coverage", artifact["feature_config"].get("continuous_window_coverage") is True)
    trained_certification = artifact.get("coverage_certification", artifact["feature_config"].get("coverage_certification"))
    if ((metadata.get("continuous_window_coverage") is True) != trained_coverage
            or metadata.get("coverage_certification") != trained_certification):
        raise ValueError("Inference continuous coverage/certification differs from training")
    config = TemporalConfig(**artifact["config"])
    derived = transform_states(x, times, config, continuous_window_coverage=metadata.get("continuous_window_coverage") is True)
    if derived["feature_names"] != artifact["feature_names"]:
        raise ValueError("Temporal feature schema differs from the trained artifact")
    result = _prediction(artifact["model"], derived, artifact["selection"])
    result["feature_provenance_differences"] = {key: {"training": artifact["feature_config"].get(key), "inference": metadata.get(key)}
        for key in sorted(set(metadata) | set(artifact["feature_config"])) if metadata.get(key) != artifact["feature_config"].get(key)}
    return result

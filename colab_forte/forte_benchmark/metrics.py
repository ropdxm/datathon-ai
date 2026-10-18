"""Video-level detection and timestamp metrics with validation-only calibration.

A prediction is exactly ``{video_id, score, timecode_sec}``. Each scored video
must have one prediction, even when its score is below the decision threshold.
Undefined rates are ``None`` (JSON null); false negatives remain in the joint
timestamp recall denominator. No third-party package is required.
"""

from __future__ import annotations

import math


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _validated_pairs(records, predictions):
    if not records:
        raise ValueError("Evaluation requires at least one video")
    by_id = {}
    for row in records:
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in by_id:
            raise ValueError("Evaluation records require unique nonempty video IDs")
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError(f"{video_id}: label must be integer 0 or 1")
        duration = _finite(row.get("duration_sec"), "duration_sec")
        if duration <= 0:
            raise ValueError(f"{video_id}: duration must be positive")
        events = row.get("event_times")
        if not isinstance(events, list) or (row["label"] == 1 and not events) or (row["label"] == 0 and events):
            raise ValueError(f"{video_id}: event annotations disagree with the video label")
        for event in events:
            if not 0 <= _finite(event, "event time") <= duration:
                raise ValueError(f"{video_id}: event time is outside the video")
        by_id[video_id] = row
    submitted = {}
    for prediction in predictions:
        if not isinstance(prediction, dict) or set(prediction) != {"video_id", "score", "timecode_sec"}:
            raise ValueError("Each prediction must contain exactly video_id, score, timecode_sec")
        video_id = prediction["video_id"]
        if not isinstance(video_id, str) or video_id not in by_id or video_id in submitted:
            raise ValueError(f"Unknown or repeated prediction video_id: {video_id}")
        score = _finite(prediction["score"], "score")
        timecode = _finite(prediction["timecode_sec"], "timecode_sec")
        if not 0 <= score <= 1:
            raise ValueError(f"{video_id}: score must lie in [0, 1]")
        if not 0 <= timecode <= by_id[video_id]["duration_sec"]:
            raise ValueError(f"{video_id}: predicted timecode is outside the video")
        submitted[video_id] = prediction
    if set(submitted) != set(by_id):
        raise ValueError("Predictions must cover every video exactly once; missing: " +
                         ", ".join(sorted(set(by_id) - set(submitted))))
    return [(row, submitted[row["video_id"]]) for row in records]


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _rank_metrics(pairs):
    positives = [prediction["score"] for row, prediction in pairs if row["label"]]
    negatives = [prediction["score"] for row, prediction in pairs if not row["label"]]
    auroc = None
    if positives and negatives:
        auroc = sum((pos > neg) + .5 * (pos == neg) for pos in positives for neg in negatives)
        auroc /= len(positives) * len(negatives)
    average_precision = None
    if positives:
        # Tied scores enter together, matching threshold-based average precision.
        cumulative_positive = cumulative_total = 0
        average_precision = 0.0
        grouped = {}
        for row, prediction in pairs:
            grouped.setdefault(prediction["score"], []).append(row["label"])
        for score in sorted(grouped, reverse=True):
            labels = grouped[score]
            added_positive = sum(labels)
            cumulative_positive += added_positive
            cumulative_total += len(labels)
            average_precision += (added_positive / len(positives)) * (cumulative_positive / cumulative_total)
    return auroc, average_precision


def evaluate_predictions(records: list[dict], predictions: list[dict], threshold: float,
                         tolerances=(1, 2, 5)) -> dict:
    """Evaluate complete video predictions; detection is ``score >= threshold``.

    Timestamp error is distance to the closest annotation in the canonical
    video's event union. ``hit@Ns_detected_positive`` uses true positive videos;
    ``joint_recall@Ns`` uses *all* positive videos, including missed detections.
    ``mean_abs_error_on_detected_positive`` is conditional on true positives and
    should always be read alongside joint recall. AUROC is tie-aware; AP is the
    non-interpolated average precision. Undefined metrics are None.
    """
    threshold = _finite(threshold, "threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    tolerances = tuple(_finite(tolerance, "tolerance") for tolerance in tolerances)
    if any(tolerance < 0 for tolerance in tolerances):
        raise ValueError("Timestamp tolerances cannot be negative")
    pairs = _validated_pairs(records, predictions)
    tp = fp = fn = tn = 0
    errors = []
    for row, prediction in pairs:
        detected = prediction["score"] >= threshold
        if row["label"] and detected:
            tp += 1
            errors.append(min(abs(prediction["timecode_sec"] - event) for event in set(row["event_times"])))
        elif row["label"]:
            fn += 1
        elif detected:
            fp += 1
        else:
            tn += 1
    recall, specificity = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    auroc, ap = _rank_metrics(pairs)
    result = {
        "threshold": threshold, "n_videos": len(pairs), "n_positive": tp + fn, "n_negative": tn + fp,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": _ratio(tp, tp + fp), "recall": recall, "specificity": specificity,
        "F1": _ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "AUROC": auroc, "AP": ap,
        "mean_abs_error_on_detected_positive": _ratio(sum(errors), len(errors)),
    }
    for tolerance in tolerances:
        suffix = f"{tolerance:g}s"
        hits = sum(error <= tolerance for error in errors)
        result[f"hit@{suffix}_detected_positive"] = _ratio(hits, len(errors))
        result[f"joint_recall@{suffix}"] = _ratio(hits, tp + fn)
    return result


def select_threshold(records: list[dict], predictions: list[dict]) -> float:
    """Maximize validation F1, then recall, specificity, and threshold in order.

    Only validation rows are accepted; test labels or timestamps cannot enter
    calibration. Candidate thresholds are all observed scores plus 0 and 1.
    Single-class inputs are handled deterministically, using zero only for an
    undefined comparison metric. Timestamp accuracy does not affect calibration.
    """
    if not records or any(row.get("split") != "val" for row in records):
        raise ValueError("Threshold selection accepts nonempty validation records only")
    _validated_pairs(records, predictions)
    candidates = sorted({0.0, 1.0, *(float(prediction["score"]) for prediction in predictions)})
    best_key, best_threshold = None, None
    for threshold in candidates:
        report = evaluate_predictions(records, predictions, threshold, tolerances=())
        key = (report["F1"] or 0, report["recall"] or 0, report["specificity"] or 0, threshold)
        if best_key is None or key > best_key:
            best_key, best_threshold = key, threshold
    return best_threshold

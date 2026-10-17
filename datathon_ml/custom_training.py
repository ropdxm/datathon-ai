"""Train-only auxiliary recordings on frozen original test folds.

Custom pseudo labels never serve as evaluation targets. Every auxiliary view
defaults to one filming session until a person verifies a different grouping.
The historical temporal and embedding modules remain byte-for-byte unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from . import model as base
from . import temporal


@dataclass(frozen=True)
class CustomTrainingConfig:
    policy: str = "semantic"
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    primary_metric: str = "balanced_accuracy"
    auxiliary_weight: float = .25
    semantic_cs: tuple[float, ...] = (.01, .1, 1.)
    point_logistic_cs: tuple[float, ...] = (.1, 1.)
    point_rbf_cs: tuple[float, ...] = (1.,)

    def __post_init__(self):
        if self.policy not in {"semantic", "point"}:
            raise ValueError("policy must be semantic or point")
        if not 0 < self.auxiliary_weight <= 1:
            raise ValueError("auxiliary_weight must be in (0,1]")
        if min(self.outer_folds, self.inner_folds) < 2:
            raise ValueError("At least two folds are required")
        if not self.semantic_cs or any(c <= 0 for c in (*self.semantic_cs, *self.point_logistic_cs, *self.point_rbf_cs)):
            raise ValueError("Regularization strengths must be positive")
        self.training_config()  # Validate the metric and candidate contract.

    def temporal_config(self):
        return temporal.TemporalConfig(seed=self.seed, outer_folds=self.outer_folds,
            inner_folds=self.inner_folds, primary_metric=self.primary_metric,
            logistic_cs=self.semantic_cs)

    def training_config(self):
        return base.TrainingConfig(seed=self.seed, outer_folds=self.outer_folds,
            inner_folds=self.inner_folds, primary_metric=self.primary_metric,
            heads=("point",), preprocessing_modes=("pca",), pca_components=32,
            logistic_cs=self.point_logistic_cs, rbf_cs=self.point_rbf_cs,
            aggregations=("smooth_max",))


def _lineage(record):
    result = {str(record["sha256"])}
    for name in ("source_sha256", "parent_sha256", "original_sha256"):
        if record.get(name):
            result.add(str(record[name]))
    for name in ("lineage_sha256s", "parent_sha256s"):
        values = record.get(name, [])
        if isinstance(values, str):
            values = [values]
        result.update(str(value) for value in values if value)
    return result


def select_auxiliary_records(records, originals=(), *, allow_unverified=False,
                             allow_machine=False):
    """Return eligible records and exclusions, never silently promote proposals.

    ``label_status`` is human_verified, supplied_unverified or machine_proposal.
    Other/missing statuses remain excluded. Metadata about related transcodes
    must survive in source_sha256/parent_sha256/lineage_sha256s.
    """
    allowed = {"human_verified"}
    if allow_unverified:
        allowed.add("supplied_unverified")
    if allow_machine:
        allowed.add("machine_proposal")
    original_lineage = set().union(*(_lineage(r) for r in originals)) if originals else set()
    accepted, excluded, seen = [], [], set()
    lineage_labels = {}
    for source in records:
        r = dict(source)
        status = r.get("label_status", "unlabeled")
        reason = None
        if status not in allowed or r.get("label") is None:
            reason = "label_status_not_authorized"
        else:
            r = base._canonical_records([r])[0]
            for digest in _lineage(r):
                if digest in lineage_labels and lineage_labels[digest] != r["label"]:
                    raise ValueError("Conflicting auxiliary labels for the same source/parent lineage")
            if _lineage(r) & original_lineage:
                reason = "original_content_or_parent_overlap"
            elif _lineage(r) & seen:
                reason = "duplicate_auxiliary_source_or_derivative"
        if reason:
            excluded.append({"sha256": r.get("sha256"), "label_status": status, "reason": reason})
            continue
        # Same-scene iPhone/Samsung views have unresolved correspondences. They
        # are one conservative session, never 39 independent recording groups.
        if not r.get("group_verified", False):
            r["group_id"] = "custom_unresolved_paired_session"
        elif not r.get("group_id"):
            raise ValueError("Verified auxiliary grouping needs group_id")
        canonical = base._canonical_records([r])[0]
        accepted.append(canonical)
        seen.update(_lineage(canonical))
        lineage_labels.update({digest: canonical["label"] for digest in _lineage(canonical)})
    return accepted, excluded


def auxiliary_for_training(auxiliary, blocked):
    """Exclude every view related to an outer/inner held-out source or group."""
    blocked_groups = {r.get("group_id", r["sha256"]) for r in blocked}
    blocked_lineage = set().union(*(_lineage(r) for r in blocked)) if blocked else set()
    return [r for r in auxiliary if r["group_id"] not in blocked_groups
            and not (_lineage(r) & blocked_lineage)]


def auxiliary_supervision_manifest(records):
    """Fingerprint the exact accepted labels, source lineage and teacher protocol."""
    rows = [{"sha256": r["sha256"],
             "prepared_sha256": r.get("prepared_sha256", r["sha256"]),
             "source_sha256": r.get("source_sha256"),
             "lineage_sha256s": sorted(_lineage(r)),
             "label": int(r["label"]), "event_times": list(r.get("event_times", [])),
             "label_status": r.get("label_status"),
             "proposal_protocol_sha256": r.get("proposal_protocol_sha256", r.get("protocol_sha256")),
             "training_supervision": r.get("training_supervision"),
             "evaluation_ground_truth_eligible": r.get("evaluation_ground_truth_eligible"),
             "group_id": r["group_id"], "group_verified": bool(r.get("group_verified", False))}
            for r in sorted(records, key=lambda item: item["sha256"])]
    payload = {"schema_version": 1, "records": rows}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest(),
            "hash_method": "sha256_of_utf8_canonical_json_schema_version_and_records"}


def _assert_original_lineage_groups(records):
    groups = {}
    for r in records:
        for digest in _lineage(r):
            group = r.get("group_id", r["sha256"])
            if digest in groups and groups[digest] != group:
                raise ValueError("Related original source/transcode appears in different groups")
            groups[digest] = group


def prepare_data(originals, auxiliary, original_feature_dir, auxiliary_feature_dir, config):
    """Read separate cache roots into one mapping without copying cache files."""
    prepare = temporal._prepare if config.policy == "semantic" else base._prepare
    protocol = config.temporal_config() if config.policy == "semantic" else config.training_config()
    original_data = prepare(originals, original_feature_dir, protocol)
    auxiliary_data = prepare(auxiliary, auxiliary_feature_dir, protocol) if auxiliary else {}
    if set(original_data) & set(auxiliary_data):
        raise ValueError("Auxiliary and original feature SHA keys overlap")
    combined = {**original_data, **auxiliary_data}
    metadata = [entry["metadata"] for entry in combined.values()]
    if metadata and any(value != metadata[0] for value in metadata[1:]):
        raise ValueError("Original and auxiliary caches need identical extractor metadata")
    return combined


def _candidates(config):
    if config.policy == "semantic":
        return [{"C": float(c)} for c in config.semantic_cs]
    return base._candidates(config.training_config())


def _balanced_weights(labels, weights):
    weights = np.asarray(weights, dtype=float).copy()
    for label in (0, 1):
        mask = labels == label
        if not mask.any():
            raise ValueError("Both training classes are required")
        weights[mask] *= len(labels) / (2. * weights[mask].sum())
    return weights


def _fit_candidate(original_train, auxiliary, data, candidate, config):
    records = original_train + auxiliary
    aux_sha = {r["sha256"] for r in auxiliary}
    if config.policy == "semantic":
        if not auxiliary:
            return temporal._fit(original_train, data, candidate["C"])
        from sklearn.preprocessing import StandardScaler
        x = np.stack([data[r["sha256"]]["derived"]["features"] for r in records])
        y = np.asarray([r["label"] for r in records])
        weights = np.asarray([config.auxiliary_weight if r["sha256"] in aux_sha else 1. for r in records])
        scaler = StandardScaler().fit(x, sample_weight=weights)
        classifier = LogisticRegression(C=candidate["C"], solver="lbfgs", max_iter=2000)
        classifier.fit(scaler.transform(x), y, sample_weight=_balanced_weights(y, weights))
        return {"scaler": scaler, "classifier": classifier, "C": candidate["C"],
                "fit_sha256": [r["sha256"] for r in records]}
    protocol = config.training_config()
    if not auxiliary:
        return base._fit(original_train, data, candidate, protocol)
    # The original helper balances each source equally within each class. Its
    # row ordering follows the records, so recover source-wise row counts using
    # its same point/normal sampling protocol before applying weak-label weight.
    x, y, weights = base._training_data(records, data, "point", protocol)
    factors = []
    for r in records:
        values, times = data[r["sha256"]]["point"], data[r["sha256"]]["timestamps"]
        if r["label"] == 0:
            count = len(np.unique(np.linspace(0, len(values) - 1,
                min(len(values), protocol.max_train_windows_per_video)).astype(int)))
        elif r["event_times"]:
            distance = np.abs(times[:, None] - np.asarray(r["event_times"])[None])
            nearest = np.unique(np.argmin(distance, axis=0))
            index = np.unique(np.r_[np.flatnonzero(distance.min(axis=1) <= protocol.positive_radius), nearest])
            count = min(len(index), max(protocol.max_train_windows_per_video, len(nearest)))
        else:
            count = 0
        factors.extend([config.auxiliary_weight if r["sha256"] in aux_sha else 1.] * count)
    if len(factors) != len(y):
        raise ValueError("Point sampling/auxiliary weighting contract changed")
    weights = _balanced_weights(y, weights * np.asarray(factors))
    scaler, pca, z = base._preprocess_fit(x, protocol, candidate["preprocessing"])
    classifier = base._classifier(candidate["classifier"], candidate["C"], config.seed, candidate["preprocessing"])
    classifier.fit(z, y, sample_weight=weights)
    return {"scaler": scaler, "pca": pca, "classifier": classifier, "candidate": candidate}


def _scores(model, records, data, config):
    if config.policy == "semantic":
        return temporal._logits(model, records, data)
    protocol = config.training_config()
    return np.asarray([base._aggregate(base._score(model, data[r["sha256"]]["point"]),
        data[r["sha256"]]["timestamps"], "smooth_max", protocol) for r in records])


def select_on_originals(original_train, auxiliary, data, config):
    """Inner validation/calibration/threshold targets are exclusively originals."""
    splits = base.grouped_splits(original_train, config.inner_folds, config.seed + 1)
    candidates = _candidates(config)
    if not splits:
        raise ValueError("Not enough independent original groups for inner selection")
    raw = [np.zeros(len(original_train)) for _ in candidates]
    selection_folds = []
    for fold, (train_idx, validation_idx) in enumerate(splits):
        train = [original_train[i] for i in train_idx]
        validation = [original_train[i] for i in validation_idx]
        aux_train = auxiliary_for_training(auxiliary, validation)
        selection_folds.append({"fold": fold,
            "original_training_sha256": [r["sha256"] for r in train],
            "auxiliary_training_sha256": [r["sha256"] for r in aux_train],
            "validation_sha256": [r["sha256"] for r in validation],
            "preprocessing_fit_sha256": [r["sha256"] for r in train + aux_train]})
        for i, candidate in enumerate(candidates):
            fitted = _fit_candidate(train, aux_train, data, candidate, config)
            raw[i][validation_idx] = _scores(fitted, validation, data, config)
    labels = np.asarray([r["label"] for r in original_train])
    rows = []
    for candidate, values in zip(candidates, raw):
        calibrator = None
        if config.policy == "semantic":
            calibrator = LogisticRegression(C=1., solver="lbfgs", max_iter=1000).fit(values[:, None], labels)
            scores = temporal._probabilities(values, calibrator)
        else:
            scores = values
        threshold, metrics = base._threshold(labels, scores, config.training_config())
        row = {"candidate": candidate, "threshold": threshold, "calibrator": calibrator,
               "selection_metrics": metrics, "aggregation": "smooth_max"}
        if config.policy == "semantic":
            row["C"] = candidate["C"]
        rows.append(row)
    rows.sort(key=lambda r: (r["selection_metrics"][config.primary_metric],
        r["selection_metrics"]["accuracy"], r["selection_metrics"]["recall"]), reverse=True)
    return {**rows[0], "selection_folds": selection_folds,
        "leaderboard": [_public(row) for row in rows],
        "calibration_method": "sigmoid_original_training_oof_only" if config.policy == "semantic" else "none",
        "calibration_fit_sha256": [r["sha256"] for r in original_train] if config.policy == "semantic" else [],
        "threshold_fit_sha256": [r["sha256"] for r in original_train],
        "auxiliary_validation_targets": 0}


def _public(selection):
    return {key: value for key, value in selection.items() if key != "calibrator"}


def _predict(fitted, record, data, selection, config):
    entry = data[record["sha256"]]
    if config.policy == "semantic":
        return temporal._prediction(fitted, entry["derived"], selection)
    return base._prediction(fitted, entry, selection, config.training_config())


def run_custom_comparison(original_records, auxiliary_records, original_feature_dir,
                          auxiliary_feature_dir, output_dir, *, folds_from,
                          config=None, allow_unverified_aux=False, allow_machine_aux=False):
    """Paired original-only/augmentation evaluation on exact historical test SHAs."""
    config = config or CustomTrainingConfig()
    originals = base._canonical_records(original_records)
    _assert_original_lineage_groups(originals)
    auxiliary, excluded = select_auxiliary_records(auxiliary_records, originals,
        allow_unverified=allow_unverified_aux, allow_machine=allow_machine_aux)
    splits, fold_source = temporal.frozen_outer_splits(originals, folds_from, config.temporal_config())
    data = prepare_data(originals, auxiliary, original_feature_dir, auxiliary_feature_dir, config)
    results = {}
    policies = [("original_only", [])]
    if auxiliary:
        policies.append(("original_plus_custom", auxiliary))
    for name, aux in policies:
        predictions, folds = [], []
        for fold, (train_idx, test_idx) in enumerate(splits):
            train, test = [originals[i] for i in train_idx], [originals[i] for i in test_idx]
            aux_train = auxiliary_for_training(aux, test)
            print(f"{config.policy} {name} fold {fold + 1}/{len(splits)}: original_train={len(train)}, auxiliary_train={len(aux_train)}, test={len(test)}", flush=True)
            selection = select_on_originals(train, aux_train, data, config)
            fitted = _fit_candidate(train, aux_train, data, selection["candidate"], config)
            rows = [{"sha256": r["sha256"], "video_id": r.get("video_id", r["sha256"]),
                "group_id": r["group_id"], "label": r["label"], "event_times": r["event_times"],
                "fold": fold, **_predict(fitted, r, data, selection, config)} for r in test]
            predictions.extend(rows)
            folds.append({"fold": fold, "original_training_sha256": [r["sha256"] for r in train],
                "auxiliary_training_sha256": [r["sha256"] for r in aux_train],
                "test_sha256": [r["sha256"] for r in test],
                "preprocessing_fit_sha256": [r["sha256"] for r in train + aux_train],
                "selection": _public(selection), "metrics": temporal._metrics(rows)})
            temporal._write(Path(output_dir) / f"{name}_progress.json", {"folds": folds, "predictions": predictions})
        results[name] = {"metrics": temporal._metrics(predictions), "predictions": predictions, "folds": folds}
    weak = any(r.get("label_status") != "human_verified" for r in auxiliary)
    report = {"evaluation": "frozen_original_outer_test_train_only_auxiliary_development",
        "status": "weak_label_sensitivity_not_verified_expansion" if weak else "reviewed_auxiliary_development" if auxiliary else "no_eligible_custom_labels",
        "config": asdict(config), "fold_source": fold_source, "policies": results,
        "auxiliary": {"eligible_count": len(auxiliary), "eligible_sha256": [r["sha256"] for r in auxiliary],
            "supervision_manifest": auxiliary_supervision_manifest(auxiliary),
            "label_status_counts": {s: sum(r.get("label_status") == s for r in auxiliary) for s in sorted({r.get("label_status") for r in auxiliary})},
            "groups": sorted({r["group_id"] for r in auxiliary}), "excluded": excluded,
            "weight": config.auxiliary_weight, "validation_targets": 0,
            "custom_views_paired_underlying_scenes": True, "pairing_resolved": False},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "custom_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": ["Original test labels were repeatedly inspected previously; this is a development comparison, not independent acceptance.",
            "Machine proposals and supplied unverified labels are not verified ground truth; agreement with them is never reported as accuracy.",
            "Custom views are paired observations of fewer underlying scenes; pairing is unresolved and no independent custom test is available.",
            "All auxiliary data is train-only; validation, calibration and thresholds use original inner held-out recordings only.",
            "Binary point labels cannot establish complete event precision, interval mAP, legal event types or an accuracy guarantee."]}
    if "original_plus_custom" in results:
        before = {r["sha256"]: r for r in results["original_only"]["predictions"]}
        after = results["original_plus_custom"]["predictions"]
        report["paired_change"] = {"accuracy_difference": results["original_plus_custom"]["metrics"]["accuracy"] - results["original_only"]["metrics"]["accuracy"],
            "baseline_correct_augmented_wrong": sum(before[r["sha256"]]["prediction"] == r["label"] and r["prediction"] != r["label"] for r in after),
            "baseline_wrong_augmented_correct": sum(before[r["sha256"]]["prediction"] != r["label"] and r["prediction"] == r["label"] for r in after)}
    temporal._write(Path(output_dir) / "custom_cv_report.json", report)
    return report


def fit_custom_final(original_records, auxiliary_records, original_feature_dir,
                     auxiliary_feature_dir, output_dir, *, config=None,
                     allow_unverified_aux=False, allow_machine_aux=False):
    """Fit a declared policy, retaining the existing trusted predictor format."""
    config = config or CustomTrainingConfig()
    originals = base._canonical_records(original_records)
    _assert_original_lineage_groups(originals)
    auxiliary, excluded = select_auxiliary_records(auxiliary_records, originals,
        allow_unverified=allow_unverified_aux, allow_machine=allow_machine_aux)
    data = prepare_data(originals, auxiliary, original_feature_dir, auxiliary_feature_dir, config)
    selection = select_on_originals(originals, auxiliary, data, config)
    fitted = _fit_candidate(originals, auxiliary, data, selection["candidate"], config)
    first = data[originals[0]["sha256"]]
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(),
        "training_sha256": [r["sha256"] for r in originals + auxiliary],
        "training_groups": sorted({r["group_id"] for r in originals + auxiliary}),
        "auxiliary_sha256": [r["sha256"] for r in auxiliary],
        "auxiliary_supervision_manifest": auxiliary_supervision_manifest(auxiliary),
        "auxiliary_weight": config.auxiliary_weight, "auxiliary_validation_targets": 0,
        "weak_label_sensitivity": any(r.get("label_status") != "human_verified" for r in auxiliary),
        "custom_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_score_is_test_score": False, "external_inference_api_used": False}
    artifact = {"format_version": 1, "model": fitted, "selection": selection,
        "feature_config": first["metadata"], "feature_semantic_config": base.semantic_feature_config(first["metadata"]),
        "provenance": provenance, "limitations": ["Development artifact; custom labels may be unverified. No independent accuracy guarantee."]}
    if config.policy == "semantic":
        temporal_sha = hashlib.sha256(Path(temporal.__file__).read_bytes()).hexdigest()
        provenance.update(code_sha256=temporal_sha, temporal_code_sha256=temporal_sha)
        artifact.update(kind="semantic_temporal", config=asdict(config.temporal_config()),
            concept_names=list(temporal.CONCEPT_NAMES), feature_names=first["derived"]["feature_names"],
            continuous_window_coverage=first["metadata"].get("continuous_window_coverage") is True,
            coverage_certification=first["metadata"].get("coverage_certification"))
        filename = "model_temporal.joblib"
    else:
        artifact.update(config=asdict(config.training_config()), feature_shape=first["feature_shape"],
            artifact_role="train_only_auxiliary_point_model")
        filename = "model.joblib"
    destination = Path(output_dir) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination, compress=3)
    temporal._write(destination.parent / "final_selection.json", {"selection": _public(selection),
        "provenance": provenance, "excluded_auxiliary": excluded, "model_file": filename})
    return destination

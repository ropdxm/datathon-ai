"""Training-only unsupervised projection adaptation using unlabeled recordings.

Custom labels, proposals and event points are discarded. The scaler/PCA sees
uniform training windows; the classifier sees original supervised samples only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import sklearn
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import model as base
from . import temporal
from .custom_training import _assert_original_lineage_groups, _lineage, auxiliary_for_training


@dataclass(frozen=True)
class UnlabeledConfig:
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    pca_components: int = 32
    max_projection_windows: int = 24
    logistic_cs: tuple[float, ...] = (.01, .1, 1.)
    rbf_cs: tuple[float, ...] = (1.,)
    primary_metric: str = "balanced_accuracy"
    threads: int = 2

    def __post_init__(self):
        if min(self.pca_components, self.max_projection_windows, self.threads) < 1:
            raise ValueError("Projection, sampling and thread limits must be positive")
        self.training_config()

    def training_config(self):
        return base.TrainingConfig(seed=self.seed, outer_folds=self.outer_folds,
            inner_folds=self.inner_folds, pca_components=self.pca_components,
            heads=("point",), preprocessing_modes=("pca",), aggregations=("smooth_max",),
            logistic_cs=self.logistic_cs, rbf_cs=self.rbf_cs, primary_metric=self.primary_metric)


def select_unlabeled_records(records, originals):
    """Strip all labels and retain one source/derivative without original overlap."""
    original_lineage = set().union(*(_lineage(r) for r in originals)) if originals else set()
    selected, excluded, seen = [], [], set()
    for source in sorted(records, key=lambda r: r["sha256"]):
        lineage = _lineage(source)
        reason = ("original_source_or_parent_overlap" if lineage & original_lineage else
                  "duplicate_custom_source_or_derivative" if lineage & seen else None)
        if reason:
            excluded.append({"sha256": source["sha256"], "reason": reason})
            continue
        r = {key: source[key] for key in ("sha256", "path", "source_sha256", "parent_sha256",
             "original_sha256", "lineage_sha256s", "parent_sha256s") if key in source}
        r["group_id"] = (str(source["group_id"]) if source.get("group_verified")
                         else "custom_unresolved_paired_session")
        selected.append(r)
        seen.update(lineage)
    return selected, excluded


def prepare_data(originals, custom, original_features, custom_features, config):
    protocol = config.training_config()
    data = base._prepare(originals, original_features, protocol)
    added = base._prepare(custom, custom_features, protocol) if custom else {}
    if set(data) & set(added):
        raise ValueError("Original and custom cache SHA keys overlap")
    data.update(added)
    metadata = [entry["metadata"] for entry in data.values()]
    if any(value != metadata[0] for value in metadata[1:]):
        raise ValueError("Adaptation requires identical original/custom extractor metadata")
    return data


def projection_indices(length, limit):
    if length < 1 or limit < 1:
        raise ValueError("Projection sampling requires a nonempty timeline and positive limit")
    return np.unique(np.linspace(0, length - 1, min(length, limit)).astype(int))


def fit_projection(original_train, custom_train, data, config):
    """Fit once per split, uniformly without consulting labels or event points."""
    records = sorted(original_train + custom_train, key=lambda r: r["sha256"])
    rows, counts = [], {}
    for r in records:
        values = data[r["sha256"]]["point"]
        indices = projection_indices(len(values), config.max_projection_windows)
        rows.append(values[indices])
        counts[r["sha256"]] = len(indices)
    x = np.concatenate(rows)
    if len(x) < 2:
        raise ValueError("At least two training projection windows are required")
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    pca = PCA(n_components=min(config.pca_components, len(z) - 1, z.shape[1]),
        svd_solver="randomized", iterated_power=2, random_state=config.seed).fit(z)
    return {"scaler": scaler, "pca": pca, "audit": {
        "projection_fit_sha256": [r["sha256"] for r in records],
        "projection_original_sha256": sorted(r["sha256"] for r in original_train),
        "projection_custom_sha256": sorted(r["sha256"] for r in custom_train),
        "projection_window_count_by_sha256": counts, "projection_windows": len(x),
        "projection_components": int(pca.n_components_),
        "projection_labels_used": 0,
        "sampling": "uniform_max_distinct_temporal_rows_no_padding_no_anchor_access"}}


def _supervised_rows(original_train, data, projection, config):
    x, labels, weights = base._training_data(original_train, data, "point", config.training_config())
    return projection["pca"].transform(projection["scaler"].transform(x)), labels, weights


def _classifier(candidate, config):
    return base._classifier(candidate["classifier"], candidate["C"], config.seed, "pca")


def fit_head(original_train, data, projection, candidate, config):
    z, labels, weights = _supervised_rows(original_train, data, projection, config)
    classifier = _classifier(candidate, config)
    classifier.fit(z, labels, sample_weight=weights)
    return {"scaler": projection["scaler"], "pca": projection["pca"],
        "classifier": classifier, "candidate": candidate,
        "supervised_training_sha256": [r["sha256"] for r in original_train]}


def _probabilities(classifier, values):
    return (classifier.predict_proba(values)[:, 1] if hasattr(classifier, "predict_proba")
            else expit(classifier.decision_function(values))).astype(float)


def select_model(original_train, custom, data, config):
    """Only original inner validation labels select candidates and threshold."""
    protocol = config.training_config()
    candidates = base._candidates(protocol)
    splits = base.grouped_splits(original_train, config.inner_folds, config.seed + 1)
    if not splits:
        raise ValueError("Need independent original class groups for inner selection")
    scores = [np.zeros(len(original_train)) for _ in candidates]
    folds = []
    for fold, (train_idx, validation_idx) in enumerate(splits):
        train = [original_train[i] for i in train_idx]
        validation = [original_train[i] for i in validation_idx]
        custom_train = auxiliary_for_training(custom, validation)
        projection = fit_projection(train, custom_train, data, config)
        z, labels, weights = _supervised_rows(train, data, projection, config)
        projected = {r["sha256"]: projection["pca"].transform(projection["scaler"].transform(
            data[r["sha256"]]["point"])) for r in validation}
        folds.append({"fold": fold, **projection["audit"],
            "supervised_training_sha256": [r["sha256"] for r in train],
            "validation_sha256": [r["sha256"] for r in validation],
            "validation_custom_targets": 0})
        # Scaler/PCA and projected validation rows are reused across all Cs.
        for i, candidate in enumerate(candidates):
            classifier = _classifier(candidate, config)
            classifier.fit(z, labels, sample_weight=weights)
            for index in validation_idx:
                r = original_train[index]
                values = _probabilities(classifier, projected[r["sha256"]])
                scores[i][index] = base._aggregate(values, data[r["sha256"]]["timestamps"], "smooth_max", protocol)
    labels = np.asarray([r["label"] for r in original_train])
    leaderboard = []
    for candidate, values in zip(candidates, scores):
        threshold, metrics = base._threshold(labels, values, protocol)
        leaderboard.append({"candidate": candidate, "aggregation": "smooth_max", "threshold": threshold,
            "selection_metrics": metrics})
    leaderboard.sort(key=lambda row: (row["selection_metrics"][config.primary_metric],
        row["selection_metrics"]["accuracy"], row["selection_metrics"]["recall"]), reverse=True)
    return {**leaderboard[0], "leaderboard": leaderboard, "selection_folds": folds,
        "threshold_fit_sha256": [r["sha256"] for r in original_train],
        "calibration_method": "none_uncalibrated_point_scores",
        "custom_labels_used": 0, "auxiliary_evaluation_targets": 0}


def _manifest(custom):
    rows = [{"sha256": r["sha256"], "source_sha256": r.get("source_sha256"),
             "lineage_sha256s": sorted(_lineage(r)), "group_id": r["group_id"]}
            for r in sorted(custom, key=lambda r: r["sha256"])]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"records": rows, "sha256": hashlib.sha256(encoded).hexdigest(), "labels_included": False}


def _records(original_records, custom_records):
    originals = sorted(base._canonical_records(original_records), key=lambda r: r["sha256"])
    _assert_original_lineage_groups(originals)
    custom, excluded = select_unlabeled_records(custom_records, originals)
    return originals, custom, excluded


def run_unlabeled_comparison(original_records, custom_records, original_features,
                            custom_features, output_dir, *, folds_from, config=None):
    config = config or UnlabeledConfig()
    originals, custom, excluded = _records(original_records, custom_records)
    splits, fold_source = temporal.frozen_outer_splits(originals, folds_from,
        temporal.TemporalConfig(seed=config.seed, outer_folds=config.outer_folds, inner_folds=config.inner_folds))
    data = prepare_data(originals, custom, original_features, custom_features, config)
    arms = (("original_unsupervised_projection", []),
            ("original_plus_unlabeled_custom_projection", custom))
    results = {}
    with threadpool_limits(limits=config.threads):
        for name, auxiliary in arms:
            predictions, folds = [], []
            for fold, (train_idx, test_idx) in enumerate(splits):
                train, test = [originals[i] for i in train_idx], [originals[i] for i in test_idx]
                custom_train = auxiliary_for_training(auxiliary, test)
                print(f"UNLABELED_FOLD {name} {fold + 1}/{len(splits)} original_train={len(train)} custom_projection_only={len(custom_train)} test={len(test)}", flush=True)
                selection = select_model(train, custom_train, data, config)
                projection = fit_projection(train, custom_train, data, config)
                fitted = fit_head(train, data, projection, selection["candidate"], config)
                rows = [{"sha256": r["sha256"], "video_id": r.get("video_id", r["sha256"]),
                    "group_id": r["group_id"], "label": r["label"], "event_times": r["event_times"],
                    "fold": fold, **base._prediction(fitted, data[r["sha256"]], selection,
                        config.training_config())} for r in test]
                predictions.extend(rows)
                folds.append({"fold": fold, **projection["audit"],
                    "supervised_training_sha256": [r["sha256"] for r in train],
                    "test_sha256": [r["sha256"] for r in test], "selection": selection,
                    "metrics": temporal._metrics(rows), "custom_labels_used": 0})
                temporal._write(Path(output_dir) / f"{name}_progress.json", {"folds": folds, "predictions": predictions})
            results[name] = {"metrics": temporal._metrics(predictions), "folds": folds, "predictions": predictions}
    before = {r["sha256"]: r for r in results[arms[0][0]]["predictions"]}
    after = results[arms[1][0]]["predictions"]
    report = {"evaluation": "frozen_original_folds_unlabeled_train_only_projection_adaptation",
        "config": asdict(config), "fold_source": fold_source, "policies": results,
        "unlabeled_custom_manifest": _manifest(custom), "unlabeled_custom_excluded": excluded,
        "unlabeled_custom_recordings": len(custom), "unlabeled_custom_groups": len({r["group_id"] for r in custom}),
        "custom_labels_used": 0, "auxiliary_evaluation_targets": 0,
        "paired_change": {"accuracy_difference": results[arms[1][0]]["metrics"]["accuracy"] - results[arms[0][0]]["metrics"]["accuracy"],
            "baseline_correct_adapted_wrong": sum(before[r["sha256"]]["prediction"] == r["label"] and r["prediction"] != r["label"] for r in after),
            "baseline_wrong_adapted_correct": sum(before[r["sha256"]]["prediction"] != r["label"] and r["prediction"] == r["label"] for r in after)},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"numpy": np.__version__, "sklearn": sklearn.__version__, "python": platform.python_version()},
        "limitations": ["Unlabeled footage trains only an unsupervised representation; it supplies no verified custom labels or independent test.",
            "Both new arms fit projection on uniformly selected training windows; this differs from historical PCA fitted to supervised point/normal samples.",
            "The historical62.5% semantic model is a separate prior development comparator, not this new projection baseline.",
            "Repeatedly inspected original test folds remain development data; no 87% generalization guarantee is established.",
            "Two-camera simultaneous custom views share one unresolved session; they are not39 independent sessions.",
            "Scores are uncalibrated; pooled ranking metrics mix fitted outer-model scales. Point labels do not establish complete event precision."]}
    temporal._write(Path(output_dir) / "unlabeled_cv_report.json", report)
    return report


def fit_unlabeled_final(original_records, custom_records, original_features,
                        custom_features, output_dir, *, config=None):
    config = config or UnlabeledConfig()
    originals, custom, excluded = _records(original_records, custom_records)
    data = prepare_data(originals, custom, original_features, custom_features, config)
    with threadpool_limits(limits=config.threads):
        selection = select_model(originals, custom, data, config)
        projection = fit_projection(originals, custom, data, config)
        fitted = fit_head(originals, data, projection, selection["candidate"], config)
    first = data[originals[0]["sha256"]]
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(), **projection["audit"],
        "training_sha256": sorted(r["sha256"] for r in originals + custom),
        "supervised_training_sha256": [r["sha256"] for r in originals],
        "supervised_training_groups": sorted({r["group_id"] for r in originals}),
        "unsupervised_custom_groups": sorted({r["group_id"] for r in custom}),
        "unlabeled_custom_manifest": _manifest(custom), "custom_labels_used": 0,
        "auxiliary_evaluation_targets": 0, "selection_score_is_test_score": False,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_model_code_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        "external_inference_api_used": False}
    artifact = {"format_version": 1, "model": fitted, "selection": selection,
        "config": asdict(config.training_config()), "feature_config": first["metadata"],
        "feature_semantic_config": base.semantic_feature_config(first["metadata"]),
        "feature_shape": first["feature_shape"], "provenance": provenance,
        "artifact_role": "original_supervised_point_head_with_unlabeled_custom_projection",
        "limitations": "Development artifact; custom labels unused; no independent accuracy guarantee."}
    destination = Path(output_dir) / "model.joblib"
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination, compress=3)
    temporal._write(destination.parent / "final_selection.json", {"selection": selection,
        "provenance": provenance, "unlabeled_custom_excluded": excluded, "model_file": destination.name})
    return destination

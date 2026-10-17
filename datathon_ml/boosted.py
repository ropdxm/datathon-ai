"""Predeclared CatBoost comparison on frozen, recording-level visual features.

Four representations and three tree settings are selected using grouped inner
OOF predictions only. The outer source-SHA assignment is loaded unchanged.
No unmarked window is assigned a binary target; these are recording heads and
provide no event localization. The feature transforms never inspect labels.
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import model as base
from . import temporal


CATBOOST_VERSION = "1.2.10"
ARMS = ("semantic_raw", "clip_all_raw", "clip_all_pca16", "clip_full_raw")


@dataclass(frozen=True)
class BoostedConfig:
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    iterations: int = 160
    learning_rate: float = .03
    pca_components: int = 16
    threads: int = 2
    primary_metric: str = "balanced_accuracy"

    def __post_init__(self):
        if min(self.outer_folds, self.inner_folds) < 2:
            raise ValueError("Grouped fold counts must be >=2")
        if min(self.iterations, self.pca_components, self.threads) < 1:
            raise ValueError("Iterations, PCA components and threads must be positive")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("Invalid learning rate")
        if self.primary_metric not in {"accuracy", "balanced_accuracy", "f1", "f2"}:
            raise ValueError("Unsupported primary metric")

    def temporal_config(self):
        return temporal.TemporalConfig(seed=self.seed, outer_folds=self.outer_folds,
            inner_folds=self.inner_folds, calibrate=False, primary_metric=self.primary_metric)


def candidate_grid():
    """Fixed before the first outer score; stable ties favor the earlier row."""
    settings = ((2, 10.), (3, 10.), (3, 30.))
    return [{"arm": arm, "depth": depth, "l2_leaf_reg": l2}
            for arm in ARMS for depth, l2 in settings]


def candidate_id(candidate):
    return f"{candidate['arm']}_depth{candidate['depth']}_l2{candidate['l2_leaf_reg']:g}"


def _classifier(candidate, config):
    import catboost
    if catboost.__version__ != CATBOOST_VERSION:
        raise ValueError(f"Use pinned catboost=={CATBOOST_VERSION}")
    return catboost.CatBoostClassifier(iterations=config.iterations,
        learning_rate=config.learning_rate, depth=candidate["depth"],
        l2_leaf_reg=candidate["l2_leaf_reg"], loss_function="Logloss",
        boosting_type="Plain", bootstrap_type="No", random_strength=0.,
        border_count=32, random_seed=config.seed, thread_count=config.threads,
        task_type="CPU", allow_writing_files=False, verbose=False)


def _canonical(records):
    return sorted(base._canonical_records(records), key=lambda r: r["sha256"])


def _clip_vectors(features, metadata):
    views = metadata.get("views")
    if not isinstance(views, list) or views != ["full", "top", "middle", "bottom"]:
        raise ValueError("Expected the existing ordered four-view CLIP cache")
    if features.shape[1] != len(views):
        raise ValueError("CLIP cache view shape differs from its metadata")
    all_views = base.pooled_features(features)[0]
    full = base.pooled_features(features[:, :1])[0]
    return {"clip_all_raw": all_views, "clip_all_pca16": all_views,
            "clip_full_raw": full}


def prepare_data(records, clip_features, semantic_features, config):
    """Deterministic summaries only; no fitting or annotation-derived features."""
    data, expected = {}, None
    tconfig = config.temporal_config()
    for record in records:
        key = record["sha256"]
        x, _, clip_meta = base.load_features(Path(clip_features) / f"{key}.npz")
        sx, st, semantic_meta = temporal.load_states(Path(semantic_features) / f"{key}.npz")
        signature = {"clip": clip_meta, "semantic": semantic_meta,
                     "clip_shape": list(x.shape[1:])}
        if expected is None:
            expected = signature
        if signature != expected:
            raise ValueError("Mixed extraction protocols or shapes across recordings")
        derived = temporal.transform_states(sx, st, tconfig,
            continuous_window_coverage=semantic_meta.get("continuous_window_coverage") is True)
        data[key] = {**_clip_vectors(x, clip_meta), "semantic_raw": derived["features"]}
    if not data:
        raise ValueError("No recordings provided")
    return data, {**expected, "semantic_feature_names": derived["feature_names"]}


def _matrix(records, data, arm):
    if arm not in ARMS:
        raise ValueError("Unknown feature arm")
    x = np.stack([data[r["sha256"]][arm] for r in records]).astype(np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite recording feature vectors")
    return x


def fit_preprocessing(records, data, arm, config):
    x = _matrix(records, data, arm)
    scaler = pca = None
    if arm == "clip_all_pca16":
        if len(x) < 2:
            raise ValueError("PCA requires at least two training recordings")
        scaler = StandardScaler().fit(x)
        pca = PCA(n_components=min(config.pca_components, len(x) - 1, x.shape[1]),
            svd_solver="randomized", iterated_power=2, random_state=config.seed)
        pca.fit(scaler.transform(x))
    return {"arm": arm, "scaler": scaler, "pca": pca,
        "fit_sha256": [r["sha256"] for r in records], "input_dimensions": int(x.shape[1]),
        "output_dimensions": int(pca.n_components_) if pca is not None else int(x.shape[1]),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()) if pca is not None else None}


def _transform(preprocessing, matrix):
    if preprocessing["pca"] is None:
        return matrix
    return preprocessing["pca"].transform(preprocessing["scaler"].transform(matrix))


def _fit(records, data, candidate, config, preprocessing):
    labels = np.asarray([r["label"] for r in records], dtype=int)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both classes are required for training")
    # Exactly one row per recording, followed by train-only class balancing.
    weights = len(labels) / (2 * counts[labels])
    classifier = _classifier(candidate, config)
    classifier.fit(_transform(preprocessing, _matrix(records, data, candidate["arm"])),
                   labels, sample_weight=weights)
    return {"classifier": classifier, "candidate": dict(candidate), "preprocessing": preprocessing,
            "fit_sha256": [r["sha256"] for r in records], "train_class_counts": counts.tolist()}


def _scores(model, records, data):
    matrix = _matrix(records, data, model["candidate"]["arm"])
    return np.asarray(model["classifier"].predict_proba(
        _transform(model["preprocessing"], matrix))[:, 1], dtype=float)


def _audit(preprocessing):
    return {k: v for k, v in preprocessing.items() if k not in {"scaler", "pca"}}


def select_model(records, data, config):
    """Select feature arm, tree setting and threshold using training groups only."""
    splits = base.grouped_splits(records, config.inner_folds, config.seed + 1)
    if not splits:
        raise ValueError("Need independent class groups for inner model selection")
    candidates = candidate_grid()
    raw = {candidate_id(c): np.full(len(records), np.nan) for c in candidates}
    details = []
    for fold, (train_idx, validation_idx) in enumerate(splits):
        train = [records[i] for i in train_idx]
        validation = [records[i] for i in validation_idx]
        preprocessing = {arm: fit_preprocessing(train, data, arm, config) for arm in ARMS}
        details.append({"fold": fold, "train_groups": sorted({r["group_id"] for r in train}),
            "validation_groups": sorted({r["group_id"] for r in validation}),
            "train_sha256": [r["sha256"] for r in train],
            "validation_sha256": [r["sha256"] for r in validation],
            "preprocessing": {arm: _audit(value) for arm, value in preprocessing.items()}})
        for candidate in candidates:
            model = _fit(train, data, candidate, config, preprocessing[candidate["arm"]])
            raw[candidate_id(candidate)][validation_idx] = _scores(model, validation, data)
    labels = np.asarray([r["label"] for r in records])
    rows = []
    metric_config = temporal._metric_config(config.temporal_config())
    for candidate in candidates:
        scores = raw[candidate_id(candidate)]
        if not np.isfinite(scores).all():
            raise ValueError("Incomplete inner OOF predictions")
        threshold, metrics = base._threshold(labels, scores, metric_config)
        rows.append({"candidate": candidate, "threshold": threshold, "selection_metrics": metrics})
    # Python stable sort preserves the preregistered order for exact ties.
    rows.sort(key=lambda row: (row["selection_metrics"][config.primary_metric],
        row["selection_metrics"]["accuracy"], row["selection_metrics"]["recall"]), reverse=True)
    return {"primary": rows[0], "arms": {arm: next(row for row in rows if row["candidate"]["arm"] == arm)
        for arm in ARMS}, "leaderboard": rows, "selection_folds": details,
        "threshold_fit_sha256": [r["sha256"] for r in records],
        "calibration_method": "none_uncalibrated_catboost_probabilities"}


def _prediction(score, selection):
    predicted = int(score >= selection["threshold"])
    return {"score": float(score), "prediction": predicted, "threshold": selection["threshold"],
        "selected_model": selection["candidate"], "verdict": "violation" if predicted else "no_violation",
        "events": [], "localization_available": False, "score_calibrated": False}


def _metrics(rows):
    result = base.classification_metrics([r["label"] for r in rows],
        [r["score"] for r in rows], [r["prediction"] for r in rows])
    result["localization"] = {"available": False,
        "reason": "Recording-level feature pooling cannot locate a violation timestamp"}
    return result


def _source_hashes():
    return {"boosted.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model.py": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        "temporal.py": hashlib.sha256(Path(temporal.__file__).read_bytes()).hexdigest()}


def _protocol(config):
    payload = {"config": asdict(config), "candidate_grid": candidate_grid(),
        "source_hashes": _source_hashes(), "catboost_version": CATBOOST_VERSION,
        "annotation_mode": "one_binary_target_per_recording_no_event_points_used",
        "class_balance": "train_only_equal_total_class_weight", "early_stopping": False}
    payload["protocol_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return payload


def run_boosted_comparison(records, clip_features, semantic_features, output_dir,
                           *, folds_from, config=None):
    config = config or BoostedConfig()
    records = _canonical(records)
    splits, fold_source = temporal.frozen_outer_splits(records, folds_from, config.temporal_config())
    with threadpool_limits(limits=config.threads):
        data, feature_protocol = prepare_data(records, clip_features, semantic_features, config)
        predictions, folds = [], []
        arm_predictions = {arm: [] for arm in ARMS}
        for fold, (train_idx, test_idx) in enumerate(splits):
            train, test = [records[i] for i in train_idx], [records[i] for i in test_idx]
            print(f"BOOSTED_OUTER_FOLD {fold + 1}/{len(splits)} train={len(train)} test={len(test)}", flush=True)
            selection = select_model(train, data, config)
            preprocessing = {arm: fit_preprocessing(train, data, arm, config) for arm in ARMS}
            models = {arm: _fit(train, data, chosen["candidate"], config, preprocessing[arm])
                      for arm, chosen in selection["arms"].items()}
            primary_id = candidate_id(selection["primary"]["candidate"])
            current = []
            for arm, chosen in selection["arms"].items():
                scores = _scores(models[arm], test, data)
                for record, score in zip(test, scores):
                    row = {"sha256": record["sha256"], "group_id": record["group_id"],
                        "label": record["label"], "fold": fold, **_prediction(score, chosen)}
                    arm_predictions[arm].append(row)
                    if candidate_id(chosen["candidate"]) == primary_id:
                        predictions.append(row)
                        current.append(row)
            folds.append({"fold": fold, "train_sha256": [r["sha256"] for r in train],
                "test_sha256": [r["sha256"] for r in test],
                "train_groups": sorted({r["group_id"] for r in train}),
                "test_groups": sorted({r["group_id"] for r in test}),
                "preprocessing": {arm: _audit(value) for arm, value in preprocessing.items()},
                "selection": selection, "metrics": _metrics(current)})
            temporal._write(Path(output_dir) / "boosted_cv_progress.json", {
                "completed_folds": len(folds), "fold_source": fold_source,
                "folds": folds, "predictions": predictions})
    report = {"evaluation": "predeclared_catboost_nested_grouped_recording_development",
        **_protocol(config), "fold_source": fold_source, "feature_protocol": feature_protocol,
        "metrics": _metrics(predictions), "predictions": predictions, "folds": folds,
        "primary_policy": "Feature arm, tree setting and threshold selected on inner grouped OOF only",
        "arm_comparisons": {arm: {"metrics": _metrics(rows), "predictions": rows}
                            for arm, rows in arm_predictions.items()},
        "created_at": datetime.now(timezone.utc).isoformat(), "limitations": _limitations()}
    temporal._write(Path(output_dir) / "boosted_cv_report.json", report)
    return report


def _limitations():
    return ["Development corpus has informed multiple earlier experiments; no independent test accuracy guarantee.",
        "The primary inner-selected policy differs from posthoc selection of the best outer-scoring arm.",
        "Pooled recording classifiers provide no timestamps; event recall and event mAP are unavailable.",
        "No filename, ID, label, annotation point, raw frame count or recording duration is a predictor.",
        temporal.EXPOSURE_LENGTH_LIMITATION,
        "Aggregate AP/AUC mix uncalibrated score scales from different fitted outer models."]


def fit_boosted_final(records, clip_features, semantic_features, output_dir, config=None):
    config = config or BoostedConfig()
    records = _canonical(records)
    with threadpool_limits(limits=config.threads):
        data, feature_protocol = prepare_data(records, clip_features, semantic_features, config)
        selection = select_model(records, data, config)
        common = {"format_version": 1, "kind": "boosted_recording", "protocol": _protocol(config),
            "feature_protocol": feature_protocol, "limitations": _limitations(),
            "provenance": {"training_sha256": [r["sha256"] for r in records],
                "training_groups": sorted({r["group_id"] for r in records}),
                "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "selection_score_is_test_score": False, "external_inference_api_used": False}}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for arm, chosen in selection["arms"].items():
            preprocessing = fit_preprocessing(records, data, arm, config)
            artifact = {**common, "selection": chosen,
                "model": _fit(records, data, chosen["candidate"], config, preprocessing),
                "artifact_role": "arm_specific_inner_selected_development_model"}
            path = output_dir / f"model_{arm}.joblib"
            joblib.dump(artifact, path, compress=3)
            files[arm] = path.name
            if candidate_id(chosen["candidate"]) == candidate_id(selection["primary"]["candidate"]):
                primary = artifact
        primary = {**primary, "artifact_role": "primary_inner_selected_feature_arm_policy",
                   "arm_model_files": files}
        path = output_dir / "model_boosted.joblib"
        joblib.dump(primary, path, compress=3)
        temporal._write(output_dir / "boosted_final_selection.json", {
            "primary": selection["primary"], "arms": selection["arms"],
            "protocol": common["protocol"], "provenance": common["provenance"], "arm_model_files": files})
    return path


def predict_boosted_features(model_path, *, clip_feature_path=None, semantic_feature_path=None):
    """Load a trusted local artifact; inference does not accept annotations."""
    artifact = joblib.load(model_path)
    if artifact.get("kind") != "boosted_recording":
        raise ValueError("Expected a boosted_recording artifact")
    if artifact["protocol"]["source_hashes"] != _source_hashes():
        raise ValueError("Boosted inference source fingerprint differs; use matching source or refit")
    import catboost
    if catboost.__version__ != artifact["protocol"]["catboost_version"]:
        raise ValueError("CatBoost version differs from the fitted artifact")
    arm = artifact["selection"]["candidate"]["arm"]
    expected = artifact["feature_protocol"]
    if arm == "semantic_raw":
        if semantic_feature_path is None:
            raise ValueError("Selected arm requires semantic features")
        x, t, metadata = temporal.load_states(semantic_feature_path)
        training = expected["semantic"]
        if (base.semantic_feature_config(metadata) != base.semantic_feature_config(training)
            or metadata.get("continuous_window_coverage") != training.get("continuous_window_coverage")
            or metadata.get("coverage_certification") != training.get("coverage_certification")):
            raise ValueError("Semantic inference extraction protocol differs from training")
        config = BoostedConfig(**artifact["protocol"]["config"])
        derived = temporal.transform_states(x, t, config.temporal_config(),
            continuous_window_coverage=metadata.get("continuous_window_coverage") is True)
        if derived["feature_names"] != expected["semantic_feature_names"]:
            raise ValueError("Semantic feature schema differs from training")
        vector = derived["features"]
    else:
        if clip_feature_path is None:
            raise ValueError("Selected arm requires CLIP features")
        x, _, metadata = base.load_features(clip_feature_path)
        if (base.semantic_feature_config(metadata) != base.semantic_feature_config(expected["clip"])
            or list(x.shape[1:]) != expected["clip_shape"]):
            raise ValueError("CLIP inference extraction protocol differs from training")
        vector = _clip_vectors(x, metadata)[arm]
    model = artifact["model"]
    score = float(model["classifier"].predict_proba(
        _transform(model["preprocessing"], vector[None]))[0, 1])
    return _prediction(score, artifact["selection"])

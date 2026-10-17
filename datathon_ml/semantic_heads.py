"""Predeclared nonlinear comparisons on the unchanged semantic temporal vector.

Exactly seven candidates are compared: three regularized linear heads, three
RBF SVM heads and one shallow ExtraTrees head. Model family, hyperparameters,
sigmoid calibration and threshold are selected exclusively inside the outer
training recordings. A frozen historical outer SHA map is mandatory for CV.
Each family receives its own nested OOF report, separately from the primary
policy that chooses a family inside each training fold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, logit
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .model import _canonical_records, _threshold, grouped_splits, semantic_feature_config
from .temporal import (CONCEPT_NAMES, EXPOSURE_LENGTH_LIMITATION, TemporalConfig, _metric_config, _metrics,
    _prepare, _write, frozen_outer_splits, load_states, transform_states)
from . import temporal as temporal_module


FAMILIES = ("linear", "rbf", "extra_trees")


@dataclass(frozen=True)
class SemanticHeadConfig:
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    calibrate: bool = True
    primary_metric: str = "balanced_accuracy"

    def __post_init__(self):
        if min(self.outer_folds, self.inner_folds) < 2:
            raise ValueError("Grouped fold counts must be >=2")
        if self.primary_metric not in {"accuracy", "balanced_accuracy", "f1", "f2"}:
            raise ValueError("Unsupported primary metric")


def _temporal_config(config: SemanticHeadConfig) -> TemporalConfig:
    # The semantic features and rules remain identical to the fixed temporal model.
    return TemporalConfig(seed=config.seed, outer_folds=config.outer_folds,
        inner_folds=config.inner_folds, calibrate=config.calibrate, primary_metric=config.primary_metric)


def candidate_grid() -> list[dict]:
    return ([{"family": "linear", "C": c} for c in (.01, .1, 1.)]
            + [{"family": "rbf", "C": c, "gamma": "scale"} for c in (.1, 1., 10.)]
            + [{"family": "extra_trees", "n_estimators": 128, "max_depth": 3,
                "min_samples_leaf": 2}])


def _id(candidate):
    return candidate["family"] + ("_C" + str(candidate["C"]) if "C" in candidate else "_128_depth3_leaf2")


def _classifier(candidate: dict, seed: int):
    if candidate["family"] == "linear":
        return LogisticRegression(C=candidate["C"], class_weight="balanced", solver="lbfgs", max_iter=2000,
                                  random_state=seed)
    if candidate["family"] == "rbf":
        return SVC(C=candidate["C"], kernel="rbf", gamma=candidate["gamma"],
                   class_weight="balanced", random_state=seed)
    if candidate["family"] == "extra_trees":
        return ExtraTreesClassifier(n_estimators=candidate["n_estimators"], max_depth=candidate["max_depth"],
                                    min_samples_leaf=candidate["min_samples_leaf"],
                                    class_weight="balanced", random_state=seed, n_jobs=1)
    raise ValueError("Unknown semantic head family")


def _matrix(records, data):
    return np.stack([data[record["sha256"]]["derived"]["features"] for record in records])


def fit_preprocessing(records, data):
    """Fit on the supplied training sources only, never a concatenated holdout."""
    return {"scaler": StandardScaler().fit(_matrix(records, data)),
            "fit_sha256": [record["sha256"] for record in records]}


def _fit(records, data, candidate, config, preprocessing=None):
    preprocessing = preprocessing or fit_preprocessing(records, data)
    classifier = _classifier(candidate, config.seed)
    classifier.fit(preprocessing["scaler"].transform(_matrix(records, data)),
                   np.asarray([record["label"] for record in records]))
    return {**preprocessing, "classifier": classifier, "candidate": candidate}


def _raw_scores(model, matrix):
    z = model["scaler"].transform(matrix)
    classifier = model["classifier"]
    if hasattr(classifier, "decision_function"):
        return np.asarray(classifier.decision_function(z), dtype=float)
    probability = classifier.predict_proba(z)[:, list(classifier.classes_).index(1)]
    return logit(np.clip(probability, 1e-6, 1 - 1e-6))


def _probabilities(raw, calibrator):
    raw = np.asarray(raw)
    return expit(raw) if calibrator is None else calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _public(selection):
    return {key: value for key, value in selection.items() if key != "calibrator"}


def _select(records, data, config):
    candidates = candidate_grid()
    splits = grouped_splits(records, config.inner_folds, config.seed + 1)
    details = []
    if not splits:
        rows = [{"candidate": candidate, "threshold": .5, "calibrator": None,
                 "selection_metrics": None, "calibration_method": "none_insufficient_groups"}
                for candidate in candidates]
    else:
        raw = {_id(candidate): np.zeros(len(records), dtype=float) for candidate in candidates}
        for fold, (train_idx, val_idx) in enumerate(splits):
            train, val = [records[i] for i in train_idx], [records[i] for i in val_idx]
            preprocessing = fit_preprocessing(train, data)
            details.append({"fold": fold, "train_groups": sorted({r["group_id"] for r in train}),
                "validation_groups": sorted({r["group_id"] for r in val}),
                "preprocessing_fit_sha256": preprocessing["fit_sha256"]})
            for candidate in candidates:
                model = _fit(train, data, candidate, config, preprocessing)
                raw[_id(candidate)][val_idx] = _raw_scores(model, _matrix(val, data))
        labels = np.asarray([record["label"] for record in records])
        rows = []
        for candidate in candidates:
            logits = raw[_id(candidate)]
            calibrator = (LogisticRegression(C=1., solver="lbfgs", max_iter=1000,
                random_state=config.seed).fit(logits[:, None], labels) if config.calibrate else None)
            scores = _probabilities(logits, calibrator)
            threshold, metrics = _threshold(labels, scores, _metric_config(_temporal_config(config)))
            rows.append({"candidate": candidate, "threshold": threshold, "calibrator": calibrator,
                "selection_metrics": metrics,
                "calibration_method": "sigmoid_fit_on_training_group_oof_logits" if config.calibrate else "none"})
        rows.sort(key=lambda row: (row["selection_metrics"][config.primary_metric],
            row["selection_metrics"]["accuracy"], row["selection_metrics"]["recall"]), reverse=True)
    return {"primary": rows[0], "families": {family: next(row for row in rows if row["candidate"]["family"] == family)
                                              for family in FAMILIES},
        "selection_folds": details, "leaderboard": [_public(row) for row in rows],
        "calibration_fit_sha256": [record["sha256"] for record in records] if config.calibrate and splits else [],
        "note": "Inner scores select candidates/calibration/threshold; only outer OOF scores assess the selection policy."}


def _predict(model, derived, selection):
    score = float(_probabilities(_raw_scores(model, derived["features"][None]), selection["calibrator"])[0])
    predicted = int(score >= selection["threshold"])
    return {"score": score, "prediction": predicted, "verdict": "violation" if predicted else "no_violation",
        "threshold": selection["threshold"], "selected_model": selection["candidate"],
        "calibration_method": selection["calibration_method"], "calibration_validated_independently": False,
        "events": derived["events"] if predicted else [], "rule_events": derived["events"],
        "rule_score": derived["rule_score"], "rule_scores": derived["rule_scores"], "rule_trace": derived["trace"],
        "localization_available": bool(derived["events"]), "legal_violation_established": False}


def run_semantic_head_cv(records: list[dict], feature_dir: str | Path, output_dir: str | Path,
                         config: SemanticHeadConfig | None = None, *, folds_from=None) -> dict:
    config = config or SemanticHeadConfig()
    if folds_from is None:
        raise ValueError("Provide folds_from with the historical outer CV report for a fixed comparison")
    records = _canonical_records(records)
    temporal_config = _temporal_config(config)
    splits, fold_source = frozen_outer_splits(records, folds_from, temporal_config)
    data = _prepare(records, feature_dir, temporal_config)
    predictions, reports = [], []
    family_predictions = {family: [] for family in FAMILIES}
    for fold, (train_idx, test_idx) in enumerate(splits):
        train, test = [records[i] for i in train_idx], [records[i] for i in test_idx]
        print(f"Semantic head outer fold {fold + 1}/{len(splits)}: train={len(train)}, test={len(test)}", flush=True)
        selection = _select(train, data, config)
        preprocessing = fit_preprocessing(train, data)
        models = {}
        for row in selection["families"].values():
            key = _id(row["candidate"])
            models[key] = _fit(train, data, row["candidate"], config, preprocessing)
        current = []
        for record in test:
            base = {"sha256": record["sha256"], "video_id": record.get("video_id", record["sha256"]),
                "group_id": record["group_id"], "label": record["label"], "event_times": record["event_times"], "fold": fold}
            derived = data[record["sha256"]]["derived"]
            primary = selection["primary"]
            row = {**base, **_predict(models[_id(primary["candidate"])], derived, primary)}
            predictions.append(row)
            current.append(row)
            for family, chosen in selection["families"].items():
                family_predictions[family].append({**base, **_predict(models[_id(chosen["candidate"])], derived, chosen)})
        reports.append({"fold": fold, "train_groups": sorted({r["group_id"] for r in train}),
            "test_groups": sorted({r["group_id"] for r in test}), "preprocessing_fit_sha256": preprocessing["fit_sha256"],
            "test_sha256": sorted(r["sha256"] for r in test),
            "test_sha256_digest": hashlib.sha256("\n".join(sorted(r["sha256"] for r in test)).encode("ascii")).hexdigest(),
            "selection": {"primary": _public(selection["primary"]),
                "families": {family: _public(row) for family, row in selection["families"].items()},
                "selection_folds": selection["selection_folds"],
                "calibration_fit_sha256": selection["calibration_fit_sha256"], "leaderboard": selection["leaderboard"]},
            "metrics": _metrics(current)})
        _write(Path(output_dir) / "semantic_head_cv_progress.json", {"completed_folds": len(reports),
               "fold_source": fold_source, "folds": reports, "predictions": predictions})
    rule_rows = []
    for record in records:
        derived = data[record["sha256"]]["derived"]
        score = derived["rule_score"]
        rule_rows.append({"label": record["label"], "event_times": record["event_times"], "score": score,
            "prediction": int(score >= temporal_config.frozen_rule_threshold), "events": derived["events"]})
    report = {"evaluation": "nested_grouped_semantic_head_family_development", "config": asdict(config),
        "temporal_config": asdict(temporal_config), "candidate_grid": candidate_grid(), "fold_source": fold_source,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "temporal_code_sha256": hashlib.sha256(Path(temporal_module.__file__).read_bytes()).hexdigest(),
        "metrics": _metrics(predictions), "predictions": predictions,
        "primary_policy": "Family, hyperparameters, calibration and threshold selected using only each outer training set",
        "family_comparisons": {family: {"metrics": _metrics(rows), "predictions": rows}
                               for family, rows in family_predictions.items()},
        "frozen_rules": {"threshold": temporal_config.frozen_rule_threshold, "metrics": _metrics(rule_rows)},
        "folds": reports, "created_at": datetime.now(timezone.utc).isoformat(),
        "limitations": ["Optional predeclared development comparison; the corpus has already informed earlier architectures.",
            "The primary inner-selected policy is distinct from choosing the highest outer-scoring family after evaluation.",
            "Binary labels train video heads; no unmarked positive windows are labeled negative and anchors are used only for evaluation.",
            "No independent recording/session/camera test, semantic/legal event labels or accuracy guarantee is available.",
            EXPOSURE_LENGTH_LIMITATION,
            "Calibration has little training data; aggregate AP/AUC mix fitted outer-model score scales."]}
    _write(Path(output_dir) / "semantic_head_cv_report.json", report)
    print(json.dumps({"evaluation": report["evaluation"], "primary_metrics": report["metrics"],
        "family_metrics": {family: value["metrics"] for family, value in report["family_comparisons"].items()}}, indent=2), flush=True)
    return report


def fit_semantic_head_final(records: list[dict], feature_dir: str | Path, output_dir: str | Path,
                            config: SemanticHeadConfig | None = None) -> Path:
    config = config or SemanticHeadConfig()
    records = _canonical_records(records)
    temporal_config = _temporal_config(config)
    data = _prepare(records, feature_dir, temporal_config)
    selection = _select(records, data, config)
    preprocessing = fit_preprocessing(records, data)
    first = data[records[0]["sha256"]]
    metadata = first["metadata"]
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(), "training_sha256": preprocessing["fit_sha256"],
        "training_groups": sorted({r["group_id"] for r in records}), "sklearn_version": sklearn.__version__,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "selection_score_is_test_score": False,
        "temporal_code_sha256": hashlib.sha256(Path(temporal_module.__file__).read_bytes()).hexdigest(),
        "annotation_mode": "binary_video_labels_points_only_for_evaluation", "external_inference_api_used": False}
    common = {"format_version": 1, "kind": "semantic_head", "config": asdict(config),
        "temporal_config": asdict(temporal_config), "concept_names": list(CONCEPT_NAMES),
        "feature_names": first["derived"]["feature_names"], "feature_config": metadata,
        "feature_semantic_config": semantic_feature_config(metadata),
        "continuous_window_coverage": metadata.get("continuous_window_coverage") is True,
        "coverage_certification": metadata.get("coverage_certification"), "provenance": provenance,
        "limitations": ["Development classifier; primary family selection differs from posthoc selection of the best outer-scoring family.",
                        "No independent accuracy guarantee or established legal event classification.", EXPOSURE_LENGTH_LIMITATION]}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts, files = {}, {}
    for family, chosen in selection["families"].items():
        artifact = {**common, "model": _fit(records, data, chosen["candidate"], config, preprocessing), "selection": chosen,
                    "artifact_role": "family_specific_inner_selected_development_model"}
        path = output_dir / f"model_semantic_{family}.joblib"
        joblib.dump(artifact, path, compress=3)
        artifacts[family], files[family] = artifact, path.name
    primary = selection["primary"]
    artifact = {**artifacts[primary["candidate"]["family"]], "selection": primary,
                "artifact_role": "primary_inner_selected_family_policy", "family_model_files": files}
    path = output_dir / "model_semantic_head.joblib"
    joblib.dump(artifact, path, compress=3)
    _write(output_dir / "semantic_head_final_selection.json", {"primary": _public(primary),
        "families": {family: _public(row) for family, row in selection["families"].items()},
        "calibration_fit_sha256": selection["calibration_fit_sha256"], "provenance": provenance,
        "family_model_files": files, "limitations": common["limitations"]})
    return path


def predict_semantic_head_features(feature_path: str | Path, model_path: str | Path) -> dict:
    artifact = joblib.load(model_path)  # Trusted artifacts only.
    if artifact.get("kind") != "semantic_head":
        raise ValueError("Expected a semantic_head artifact")
    provenance = artifact.get("provenance", {})
    if provenance.get("code_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Semantic-head inference code fingerprint differs from the artifact; use matching code or refit")
    if provenance.get("temporal_code_sha256") != hashlib.sha256(Path(temporal_module.__file__).read_bytes()).hexdigest():
        raise ValueError("Temporal rule/feature code fingerprint differs from the artifact; use matching code or refit")
    x, times, metadata = load_states(feature_path)
    if semantic_feature_config(metadata) != artifact["feature_semantic_config"]:
        raise ValueError("Inference semantic extraction protocol differs from training")
    if ((metadata.get("continuous_window_coverage") is True) != artifact["continuous_window_coverage"]
            or metadata.get("coverage_certification") != artifact["coverage_certification"]):
        raise ValueError("Inference continuous coverage/certification differs from training")
    derived = transform_states(x, times, TemporalConfig(**artifact["temporal_config"]),
        continuous_window_coverage=metadata.get("continuous_window_coverage") is True)
    if derived["feature_names"] != artifact["feature_names"]:
        raise ValueError("Frozen temporal feature schema differs from training")
    result = _predict(artifact["model"], derived, artifact["selection"])
    result["feature_provenance_differences"] = {key: {"training": artifact["feature_config"].get(key), "inference": metadata.get(key)}
        for key in sorted(set(metadata) | set(artifact["feature_config"])) if metadata.get(key) != artifact["feature_config"].get(key)}
    return result

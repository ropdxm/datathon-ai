"""Bounded transfer test retaining X3D-S's pretrained 400-action classifier.

Images remain in native Python. The frozen extractor uses a distinct logits
cache. Recording summaries are mean/std/max over normalized logits, with no
timestamps, frame counts, identifiers or event annotations as predictors.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from datathon_ml import boosted
from datathon_ml import model as base
from datathon_ml import motion
from datathon_ml import temporal


ARM = "x3d_logits_mean_std_max"


def candidates():
    return [{"arm": ARM, "depth": d, "l2_leaf_reg": l2}
            for d, l2 in ((2, 10.), (3, 10.), (3, 30.))]


def pooled_logits(features):
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (2, 400) or not len(values):
        raise ValueError("Expected nonempty X3D logits [T,2,400]")
    if not np.isfinite(values).all():
        raise ValueError("Logits must be finite")
    values = base._normalise(values)
    return np.concatenate([values.mean(axis=0).ravel(), values.std(axis=0).ravel(),
                           values.max(axis=0).ravel()]).astype(np.float32)


def prepare(records, feature_dir):
    data, expected = {}, None
    for record in records:
        key = record["sha256"]
        values, _, metadata = base.load_features(Path(feature_dir) / f"{key}.npz")
        preprocessing = json.loads(metadata.get("preprocessing", "{}"))
        if (metadata.get("backbone") != "x3d_s" or metadata.get("views") != ["full", "middle"]
                or preprocessing.get("representation") != "logits"
                or preprocessing.get("normalization") != "l2-output"):
            raise ValueError("Expected distinct frozen X3D-S K400 normalized-logit cache")
        if expected is None:
            expected = metadata
        if metadata != expected:
            raise ValueError("Mixed motion extraction protocols")
        data[key] = pooled_logits(values)
    if not data:
        raise ValueError("No recording features")
    return data, expected


def _matrix(records, data):
    return np.stack([data[r["sha256"]] for r in records])


def _fit(records, data, candidate, config):
    labels = np.asarray([r["label"] for r in records], dtype=int)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both training classes required")
    classifier = boosted._classifier(candidate, config)
    classifier.fit(_matrix(records, data), labels,
                   sample_weight=len(labels) / (2 * counts[labels]))
    return {"classifier": classifier, "candidate": candidate,
            "fit_sha256": [r["sha256"] for r in records],
            "class_counts": counts.tolist()}


def _scores(model, records, data):
    return np.asarray(model["classifier"].predict_proba(_matrix(records, data))[:, 1], dtype=float)


def select(records, data, config):
    splits = base.grouped_splits(records, config.inner_folds, config.seed + 1)
    if not splits:
        raise ValueError("Need class groups for inner OOF selection")
    grid = candidates()
    raw = [np.full(len(records), np.nan) for _ in grid]
    folds = []
    for fold, (train_idx, validation_idx) in enumerate(splits):
        train, validation = [records[i] for i in train_idx], [records[i] for i in validation_idx]
        folds.append({"fold": fold, "fit_sha256": [r["sha256"] for r in train],
            "validation_sha256": [r["sha256"] for r in validation],
            "train_groups": sorted({r["group_id"] for r in train}),
            "validation_groups": sorted({r["group_id"] for r in validation})})
        for i, candidate in enumerate(grid):
            model = _fit(train, data, candidate, config)
            raw[i][validation_idx] = _scores(model, validation, data)
    labels = np.asarray([r["label"] for r in records])
    rows = []
    metric_config = base.TrainingConfig(primary_metric=config.primary_metric)
    for candidate, scores in zip(grid, raw):
        if not np.isfinite(scores).all():
            raise ValueError("Incomplete inner OOF predictions")
        threshold, metrics = base._threshold(labels, scores, metric_config)
        rows.append({"candidate": candidate, "threshold": threshold, "selection_metrics": metrics})
    rows.sort(key=lambda r: (r["selection_metrics"][config.primary_metric],
        r["selection_metrics"]["accuracy"], r["selection_metrics"]["recall"]), reverse=True)
    return {**rows[0], "leaderboard": rows, "selection_folds": folds,
            "threshold_fit_sha256": [r["sha256"] for r in records]}


def _prediction(score, selection):
    predicted = int(score >= selection["threshold"])
    return {"score": float(score), "prediction": predicted, "threshold": selection["threshold"],
        "selected_model": selection["candidate"], "events": [], "localization_available": False,
        "verdict": "violation" if predicted else "no_violation", "score_calibrated": False}


def _metrics(rows):
    result = base.classification_metrics([r["label"] for r in rows],
        [r["score"] for r in rows], [r["prediction"] for r in rows])
    result["localization"] = {"available": False, "reason": "Recording pooling provides no event timestamp"}
    return result


def _hashes():
    return {"motion_action_experiment.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **{Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
           for module in (motion, boosted, base, temporal)}}


def _protocol(config):
    return {"config": asdict(config), "candidate_grid": candidates(), "source_hashes": _hashes(),
        "catboost_version": boosted.CATBOOST_VERSION, "pooling": "mean_std_max_over_time_per_view",
        "descriptor_dimensions": 2400, "fitted_preprocessing": False,
        "annotation_mode": "one_binary_label_per_recording_no_event_points_used",
        "external_inference_api_used": False, "audio_used": False,
        "predeclared_hypothesis": "Retain all 400 pretrained action scores instead of discarding action classifier head"}


def _limitations():
    return ["Repeatedly inspected original development corpus; independent test performance is not established.",
        "Generic pretrained action scores are not verified violation states or legal evidence.",
        "Pooling removes ordering and provides no event timestamps.",
        "Max pooling can depend on observation length despite no duration/count predictor.",
        "Uncalibrated probabilities from separate folds limit pooled AP/AUC interpretation.",
        "The 87% target is not established by this implementation or synthetic tests."]


def compare(records, feature_dir, output_dir, folds_from, config):
    records = sorted(base._canonical_records(records), key=lambda r: r["sha256"])
    splits, fold_source = temporal.frozen_outer_splits(records, folds_from, config.temporal_config())
    data, feature_protocol = prepare(records, feature_dir)
    predictions, folds = [], []
    with threadpool_limits(limits=config.threads):
        for fold, (train_idx, test_idx) in enumerate(splits):
            train, test = [records[i] for i in train_idx], [records[i] for i in test_idx]
            selection = select(train, data, config)
            model = _fit(train, data, selection["candidate"], config)
            rows = [{"sha256": record["sha256"], "group_id": record["group_id"],
                "label": record["label"], "fold": fold, **_prediction(score, selection)}
                for record, score in zip(test, _scores(model, test, data))]
            predictions.extend(rows)
            folds.append({"fold": fold, "fit_sha256": model["fit_sha256"],
                "test_sha256": [r["sha256"] for r in test],
                "train_groups": sorted({r["group_id"] for r in train}),
                "test_groups": sorted({r["group_id"] for r in test}),
                "selection": selection, "metrics": _metrics(rows)})
            temporal._write(Path(output_dir) / "motion_action_cv_progress.json", {
                "completed_folds": len(folds), "predictions": predictions, "folds": folds})
            print(json.dumps({"motion_action_completed_fold": fold + 1, "total_folds": len(splits)}), flush=True)
    report = {"evaluation": "predeclared_x3d_kinetics400_logits_catboost_nested_grouped_development",
        "protocol": _protocol(config), "feature_protocol": feature_protocol, "fold_source": fold_source,
        "metrics": _metrics(predictions), "predictions": predictions, "folds": folds,
        "created_at": datetime.now(timezone.utc).isoformat(), "limitations": _limitations()}
    temporal._write(Path(output_dir) / "motion_action_cv_report.json", report)
    return report


def fit_final(records, feature_dir, output_dir, config):
    records = sorted(base._canonical_records(records), key=lambda r: r["sha256"])
    data, feature_protocol = prepare(records, feature_dir)
    with threadpool_limits(limits=config.threads):
        selection = select(records, data, config)
        model = _fit(records, data, selection["candidate"], config)
    artifact = {"format_version": 1, "kind": "x3d_action_logits_recording",
        "model": model, "selection": selection, "protocol": _protocol(config),
        "feature_protocol": feature_protocol, "limitations": _limitations(),
        "provenance": {"training_sha256": model["fit_sha256"],
            "training_groups": sorted({r["group_id"] for r in records}),
            "selection_score_is_test_score": False, "external_inference_api_used": False}}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model_motion_action.joblib"
    joblib.dump(artifact, path, compress=3)
    temporal._write(output_dir / "motion_action_final_selection.json", {
        "selection": selection, "protocol": artifact["protocol"], "provenance": artifact["provenance"]})
    return path


def predict(model_path, feature_path):
    artifact = joblib.load(model_path)  # Only trusted locally produced artifacts.
    if artifact.get("kind") != "x3d_action_logits_recording" or artifact["protocol"]["source_hashes"] != _hashes():
        raise ValueError("Wrong artifact or source fingerprints differ; use matching sources or refit")
    import catboost
    if catboost.__version__ != artifact["protocol"]["catboost_version"]:
        raise ValueError("Fitted CatBoost version differs")
    values, _, metadata = base.load_features(feature_path)
    if base.semantic_feature_config(metadata) != base.semantic_feature_config(artifact["feature_protocol"]):
        raise ValueError("Motion inference protocol differs from training")
    vector = pooled_logits(values)
    score = float(artifact["model"]["classifier"].predict_proba(vector[None])[0, 1])
    return _prediction(score, artifact["selection"])


def _records(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("videos"))
    if not isinstance(payload, list):
        raise ValueError("Manifest needs list or records/videos list")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "compare", "run", "predict"))
    parser.add_argument("--original-manifest", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--folds-from", type=Path, default=Path("reference/clip_baseline_cv.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--local-repo", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--fit-final", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--feature-path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "predict":
        if args.model is None or args.feature_path is None:
            parser.error("predict requires --model and --feature-path")
        print(json.dumps(predict(args.model, args.feature_path)), flush=True)
        return
    if args.original_manifest is None or args.features is None:
        parser.error("--original-manifest and --features are required")
    records = _records(args.original_manifest)
    if args.command in {"extract", "run"}:
        config = motion.MotionFeatureConfig(backbone="x3d_s", sample_fps=1., image_size=182,
            views=("full", "middle"), representation="logits", allow_download=False,
            local_checkpoint=str(args.checkpoint) if args.checkpoint else None,
            local_repo=str(args.local_repo) if args.local_repo else None, device=args.device)
        paths = motion.extract_motion_features(records, args.features, config)
        print(json.dumps({"motion_action_extracted_sources": len(paths)}), flush=True)
    if args.command in {"compare", "run"}:
        if args.output_dir is None:
            parser.error("--output-dir is required for comparison")
        config = boosted.BoostedConfig()
        report = compare(records, args.features, args.output_dir, args.folds_from, config)
        print(json.dumps({"evaluation": report["evaluation"], "metrics": report["metrics"]}), flush=True)
        if args.fit_final:
            path = fit_final(records, args.features, args.output_dir / "model", config)
            print(json.dumps({"fitted_motion_action_model": str(path)}), flush=True)


if __name__ == "__main__":
    main()

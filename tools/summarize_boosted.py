"""Verify numeric CatBoost reports against the frozen original 40 recordings."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datathon_ml.model import classification_metrics
from datathon_ml import boosted
from tools.summarize_temporal import (_atomic_text, _numeric_rows, _read, _sha, _verify,
    exact_mcnemar, wilson95)


ARMS = ("semantic_raw", "clip_all_raw", "clip_all_pca16", "clip_full_raw")
LABELS = {"semantic_raw": "CatBoost: semantic temporal vector", "clip_all_raw": "CatBoost: all-view CLIP",
    "clip_all_pca16": "CatBoost: all-view CLIP + PCA16", "clip_full_raw": "CatBoost: full-frame CLIP"}


def _source_files(hashes):
    verified = {}
    for name, digest in hashes.items():
        path = ROOT / name if name.endswith("experiment.py") else ROOT / "datathon_ml" / name
        if not path.is_file() or _sha(path) != digest:
            raise ValueError("Report ML source fingerprint differs: " + name)
        verified[name] = digest
    if not verified:
        raise ValueError("Missing ML source fingerprints")
    return verified


def _entry(identifier, label, role, path, policy, baseline, *, pooled=True, artifact=None):
    predictions = policy["predictions"]
    rows = _numeric_rows(predictions)
    _verify(rows, baseline, identifier)
    metrics = classification_metrics([r["label"] for r in predictions],
        [r["score"] for r in predictions], [r["prediction"] for r in predictions])
    for key, value in metrics.items():
        reported = policy["metrics"].get(key)
        if isinstance(value, (int, float)) and reported is not None:
            if not math.isclose(value, reported, abs_tol=1e-10):
                raise ValueError(identifier + ": reported metric differs from predictions: " + key)
        elif key == "confusion_matrix" and reported != value:
            raise ValueError(identifier + ": reported confusion matrix differs")
    if pooled:
        if any(r.get("events") or r.get("localization_available") is not False for r in predictions):
            raise ValueError(identifier + ": pooled classifier must not claim event localization")
        metrics["localization"] = {"available": False, "point_recall_5s": None,
            "reason": "Recording-level pooling provides no event timestamp"}
    else:
        metrics["localization"] = policy["metrics"].get("localization", {})
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    path = Path(path)
    artifact = Path(artifact) if artifact is not None else None
    return {"model": identifier, "label": label, "role": role, "metrics": metrics,
        "source_report_path": str(path.resolve()), "source_report_sha256": _sha(path),
        "original_SHA_labels_folds_verified": True, "classification_metrics_recomputed": True,
        "accuracy_wilson95": wilson95(tn + tp, tn + fp + fn + tp),
        "paired_vs_original_clip": exact_mcnemar(baseline, rows),
        "artifact_path": str(artifact.resolve()) if artifact and artifact.is_file() else None,
        "artifact_sha256": _sha(artifact) if artifact and artifact.is_file() else None,
        "independent_test": False}


def _assert_ownership(report, baseline, *, motion=False):
    if len(report.get("folds", [])) != 5:
        raise ValueError("Expected all five completed outer folds")
    for fold in report["folds"]:
        expected_test = {key for key, row in baseline.items() if row["fold"] == fold["fold"]}
        test = set(fold["test_sha256"])
        train = set(fold["fit_sha256"] if motion else fold["train_sha256"])
        if test != expected_test or train != set(baseline) - test:
            raise ValueError("Outer fit/test source ownership differs from frozen reference")
        if set(fold["train_groups"]) & set(fold["test_groups"]):
            raise ValueError("Outer scene-group overlap")
        selection = fold["selection"]
        if set(selection["threshold_fit_sha256"]) != train:
            raise ValueError("Threshold ownership differs from outer training set")
        validation_owners = [key for inner in selection["selection_folds"] for key in inner["validation_sha256"]]
        if len(validation_owners) != len(train) or set(validation_owners) != train:
            raise ValueError("Inner validation does not cover every training source exactly once")
        for inner in selection["selection_folds"]:
            fit = set(inner["fit_sha256"] if motion else inner["train_sha256"])
            validation = set(inner["validation_sha256"])
            if fit & validation or fit | validation != train or (fit | validation) & test:
                raise ValueError("Inner validation/training source leakage")
            if set(inner["train_groups"]) & set(inner["validation_groups"]):
                raise ValueError("Inner scene-group overlap")
            if not motion:
                for p in inner["preprocessing"].values():
                    if set(p["fit_sha256"]) != fit:
                        raise ValueError("Inner preprocessing ownership differs from inner training set")
        if not motion:
            for p in fold["preprocessing"].values():
                if set(p["fit_sha256"]) != train:
                    raise ValueError("Outer preprocessing ownership differs from outer training set")
            leaderboard = selection["leaderboard"]
            candidates = report["candidate_grid"]
            if len(leaderboard) != 12 or sorted(json.dumps(r["candidate"], sort_keys=True) for r in leaderboard) != sorted(json.dumps(c, sort_keys=True) for c in candidates):
                raise ValueError("Inner candidate coverage differs from the preregistered 12 candidates")


def _runtime(path):
    if not path.is_file():
        return {"status": "runtime_report_unavailable", "network_attempts": None}
    raw = _read(path)
    keys = ("status", "returncode", "exitcode", "elapsed_seconds", "budget_seconds", "network_attempts", "completed", "test_count")
    return {"source_runtime_sha256": _sha(path), **{k: raw[k] for k in keys if k in raw}}


def summarize(results_root, baseline_path, previous_summary, *, preprocessing_audit=None, public_data_audit=None):
    results_root = Path(results_root).resolve()
    run = results_root / "runs/boosted_v1" if (results_root / "runs/boosted_v1").is_dir() else results_root
    baseline_path = Path(baseline_path).resolve()
    original = _read(baseline_path)
    baseline = _numeric_rows(original["predictions"])
    if len(baseline) != 40 or {r["fold"] for r in baseline.values()} != set(range(5)):
        raise ValueError("Expected frozen original40 reference with five folds")
    rows = [_entry("historical_clip", "Previous CLIP baseline", "historical_baseline", baseline_path,
                   original, baseline, pooled=False)]
    previous_summary = Path(previous_summary)
    old = _read(previous_summary)
    if old.get("baseline_report_sha256") != _sha(baseline_path) or not old.get("all_report_predictions_original_SHA_labels_folds_and_points_verified"):
        raise ValueError("Previous summary does not certify the same frozen original predictions")
    prior = next(r for r in old["model_comparisons"] if r["model"] == "semantic/original_only")
    rows.append({"model": "previous_semantic_temporal", "label": "Previous best: semantic temporal",
        "role": "previous_development_best", "metrics": prior["metrics"],
        "source_report_path": str(previous_summary.resolve()), "source_report_sha256": _sha(previous_summary),
        "verification": "Imported unchanged from previously verified numeric original40 aggregate",
        "classification_metrics_recomputed": False, "original_SHA_labels_folds_verified": True,
        "independent_test": False})
    main_path = run / "boosted_cv_report.json"
    statuses = {"boosted": _runtime(run / "runtime_boosted.json"),
                "motion_action": _runtime(run / "runtime_motion_action.json"),
                "implementation_tests": _runtime(run / "runtime_tests.json")}
    source_hashes, protocol = {}, None
    if main_path.is_file():
        report = _read(main_path)
        config = boosted.BoostedConfig(**report["config"])
        if report["candidate_grid"] != boosted.candidate_grid() or report["protocol_sha256"] != boosted._protocol(config)["protocol_sha256"]:
            raise ValueError("Reported CatBoost protocol differs from matching preregistered source")
        if (config.iterations, config.learning_rate, config.pca_components) != (160, .03, 16):
            raise ValueError("Production comparison settings differ from the preregistered protocol")
        _assert_ownership(report, baseline)
        source_hashes["boosted"] = _source_files(report["source_hashes"])
        protocol = {k: report[k] for k in ("protocol_sha256", "config", "candidate_grid", "fold_source")}
        if set(report["arm_comparisons"]) != set(ARMS) or len(report["candidate_grid"]) != 12:
            raise ValueError("Unexpected CatBoost representation arms or grid size")
        rows.append(_entry("boosted_primary", "CatBoost: inner-selected policy", "primary_inner_selected_policy",
            main_path, report, baseline, artifact=run / "model/model_boosted.joblib"))
        for arm in ARMS:
            rows.append(_entry(arm, LABELS[arm], "predeclared_arm_development_comparison", main_path,
                report["arm_comparisons"][arm], baseline, artifact=run / f"model/model_{arm}.joblib"))
        statuses["boosted"]["completed_cv_report"] = True
    else:
        statuses["boosted"]["completed_cv_report"] = False
    motion_path = run / "motion_action/motion_action_cv_report.json"
    if motion_path.is_file():
        report = _read(motion_path)
        _assert_ownership(report, baseline, motion=True)
        source_hashes["motion_action"] = _source_files(report["protocol"]["source_hashes"])
        rows.append(_entry("motion_action", "X3D pretrained action scores + CatBoost", "separate_predeclared_transfer_test",
            motion_path, report, baseline, artifact=run / "motion_action/model/model_motion_action.joblib"))
        statuses["motion_action"]["completed_cv_report"] = True
    else:
        statuses["motion_action"]["completed_cv_report"] = False
    audit = None
    path = Path(preprocessing_audit) if preprocessing_audit else run / "preprocessing_audit.json"
    if path.is_file():
        audit = _read(path)
        for fold in audit.get("pca_folds", []):
            expected_test = {key for key, row in baseline.items() if row["fold"] == fold["fold"]}
            if set(fold["test_sha256"]) != expected_test or set(fold["pca_fit_owner_sha256"]) != set(baseline) - expected_test:
                raise ValueError("Preprocessing diagnostic PCA ownership differs from frozen folds")
            if fold["labels_used_to_fit_projection"] != 0 or fold["anchors_used_to_fit_projection"] != 0:
                raise ValueError("PCA diagnostic fit used supervision")
        audit = {"source_report_sha256": _sha(path), **audit}
    public = Path(public_data_audit) if public_data_audit else ROOT / "reference/public_video_data_research.json"
    public_info = {"path": str(public.resolve()), "sha256": _sha(public), "research": _read(public)} if public.is_file() else None
    best = max(rows, key=lambda r: (r["metrics"]["balanced_accuracy"], r["metrics"]["accuracy"]))
    return {"status": "completed_available_boosted_experiments", "created_at": datetime.now(timezone.utc).isoformat(),
        "tool_sha256": _sha(__file__), "n": 40, "baseline_report_sha256": _sha(baseline_path),
        "all_completed_new_predictions_SHA_labels_folds_and_metrics_verified": True,
        "nested_fit_preprocessing_and_threshold_ownership_verified": True,
        "ML_source_fingerprints_verified": source_hashes, "protocol": protocol,
        "model_comparisons": rows, "posthoc_best_development_model": best,
        "primary_inner_selected_policy": next((r for r in rows if r["model"] == "boosted_primary"), None),
        "execution": statuses, "preprocessing_audit": audit, "public_data_audit": public_info,
        "external_training_videos_added": 0, "target_accuracy": .87,
        "target_achieved_on_development": any(r["metrics"]["accuracy"] >= .87 for r in rows),
        "independent_87_percent_accuracy_established": False, "case_ready": False,
        "limitations": ["Repeated development comparisons on the same 40 original recordings; no independent test.",
            "The primary inner-selected policy differs from selecting the highest outer-scoring arm after evaluation.",
            "Recording-level heads cannot supply timestamp recall; unavailable is not a measured zero.",
            "PCA retained variance is not violation-class separability or proof of preprocessing adequacy.",
            "Public datasets require domain, label, license and source-overlap review before training."]}


def write_outputs(summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(output_dir / "boosted_experiment_summary.json", json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    table = []
    for row in summary["model_comparisons"]:
        m = row["metrics"]
        tn, fp = m["confusion_matrix"][0]
        fn, tp = m["confusion_matrix"][1]
        table.append({"model": row["model"], "role": row["role"], "n": m["n"],
            **{k: m.get(k) for k in ("accuracy", "balanced_accuracy", "recall", "specificity", "precision", "f1", "f2", "average_precision", "roc_auc")},
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "localization_available": False if m.get("localization", {}).get("available") is False else "previous_report",
            "source_report_sha256": row["source_report_sha256"], "independent_test": False})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
    _atomic_text(output_dir / "boosted_model_comparison.csv", buffer.getvalue())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = summary["model_comparisons"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(17, max(6, len(rows) * .65 + 2)), sharey=True)
    for i, row in enumerate(rows):
        accuracy = row["metrics"]["accuracy"] * 100
        left.barh(i, accuracy, color="#197f7a" if row["role"] == "primary_inner_selected_policy" else "#839bb9", height=.6)
        left.text(min(accuracy + 1, 90), i, f"{accuracy:.1f}%", va="center", fontsize=9)
    left.set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=9)
    left.invert_yaxis()
    left.axvline(87, color="#b63a44", linestyle="--", label="87% accuracy target")
    left.legend(loc="lower right", fontsize=8)
    left.set_title("Accuracy on original frozen outer folds")
    for key, label, marker, color in (("balanced_accuracy", "Balanced accuracy", "o", "#376ab2"),
        ("recall", "Violation recall", "s", "#218777"), ("specificity", "Specificity", "^", "#cd8738")):
        right.scatter([100 * r["metrics"][key] for r in rows], range(len(rows)), marker=marker, color=color, label=label)
    right.set_title("Detection tradeoffs")
    right.legend(loc="lower right", fontsize=8)
    for axis in (left, right):
        axis.set_xlim(0, 100)
        axis.set_xticks(range(0, 101, 10))
        axis.set_xlabel("Percent, full 0–100 scale")
        axis.grid(axis="x", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle("Pretrained representations + CatBoost · original 40 development recordings", fontsize=13)
    fig.text(.01, .015, "Primary policy selects its arm inside training folds. Branch maxima are posthoc development comparisons. Pooled models provide no event timestamps.", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .95))
    fig.savefig(output_dir / "boosted_model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "reference/clip_baseline_cv.json")
    parser.add_argument("--previous-summary", type=Path, default=ROOT / "artifacts/custom_experiment_summary.json")
    parser.add_argument("--preprocessing-audit", type=Path)
    parser.add_argument("--public-data-audit", type=Path, default=ROOT / "reference/public_video_data_research.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args(argv)
    summary = summarize(args.results_root, args.baseline, args.previous_summary,
        preprocessing_audit=args.preprocessing_audit, public_data_audit=args.public_data_audit)
    write_outputs(summary, args.output_dir)
    print(json.dumps({"models": {r["model"]: r["metrics"] for r in summary["model_comparisons"]},
        "target_achieved_on_development": summary["target_achieved_on_development"]}, indent=2))


if __name__ == "__main__":
    main()

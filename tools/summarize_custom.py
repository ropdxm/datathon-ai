"""Aggregate completed numeric custom-data experiments on frozen original folds.

Only report metrics/outcomes, aggregate annotation summaries and audit metadata
are consumed. Raw notes, observations, generated answers and media are excluded.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.summarize_temporal import (_entry, _metrics, _numeric_rows, _read, _sha,
                                     _verify, _atomic_text)


def _privileged_runtime(path, metrics_included):
    result = {"status": "completed_cv_report" if metrics_included else "not_included_without_completed_report",
              "metrics_included": bool(metrics_included), "notes_verified": False}
    if path is None:
        return result
    source = Path(path).resolve()
    raw = _read(source)
    if not isinstance(raw, dict):
        raise ValueError("Privileged runtime must be an aggregate JSON object")
    statuses = {"completed", "failed", "not_run", "timed_out", "hard_timeout", "blocked", "cancelled",
                "no_usable_training_prototypes", "failed_no_usable_visual_phrases", "failed_no_usable_prototypes"}
    result.update(status=raw.get("status") if raw.get("status") in statuses else "unrecognized_runtime_status",
                  source_runtime_sha256=_sha(source))
    error_type = raw.get("error_type")
    if isinstance(error_type, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,79}", error_type):
        result["error_type"] = error_type
    for key in ("exitcode", "returncode", "elapsed_seconds", "total_seconds", "budget_seconds"):
        value = raw.get(key)
        if type(value) in (int, float) and math.isfinite(value):
            result[key] = value
    if type(raw.get("completed")) is bool:
        result["completed"] = raw["completed"]
    count = raw.get("network_attempts")
    if count is None and isinstance(raw.get("network_guard"), dict):
        count = raw["network_guard"].get("attempts")
    if count is None or (type(count) is int and count >= 0):
        result["network_attempts"] = count
    # Error messages, excerpts, translations, note text and arbitrary fields
    # are deliberately absent from this fixed metadata allowlist.
    return result


def _completed_entry(identifier, label, role, path, policy, baseline, reference, artifact=None):
    rows = _numeric_rows(policy["predictions"])
    _verify(rows, baseline, identifier)
    from datathon_ml.model import _localization_metrics
    numeric_points = []
    for row in policy["predictions"]:
        points = sorted(float(t) for t in row.get("event_times", []))
        expected = sorted(float(t) for t in reference[row["sha256"]].get("event_times", []))
        if points != expected:
            raise ValueError(identifier + ": annotation anchors differ from the original reference")
        numeric_points.append({"label": row["label"], "event_times": points,
            "events": [{"timestamp_sec": float(event["timestamp_sec"])} for event in row.get("events", [])]})
    measured_points = _localization_metrics(numeric_points)
    reported = dict(policy["metrics"])
    old_points = reported.get("localization", {}).get("point_recall_at_tolerance", {})
    for key, value in measured_points["point_recall_at_tolerance"].items():
        if key in old_points and old_points[key] != value:
            raise ValueError(identifier + ": localization metric differs from numeric event timestamps")
    reported["localization"] = measured_points
    entry = _entry(identifier, label, role, "clip", path, rows, reported, baseline, artifact)
    if artifact is None:
        entry["artifact_kind"] = "report_only_no_exported_head_for_this_arm"
    return entry


def summarize(results_root, baseline_path, previous_summary=None, privileged_report=None,
              dataset_audit=None, quality_report=None, privileged_runtime=None):
    results_root, baseline_path = Path(results_root).resolve(), Path(baseline_path).resolve()
    run = results_root / "runs/custom_v1" if (results_root / "runs/custom_v1").is_dir() else results_root
    original = _read(baseline_path)
    baseline = _numeric_rows(original["predictions"])
    if len(baseline) != 40 or len({r["fold"] for r in baseline.values()}) != 5:
        raise ValueError("Expected original40-source reference on five frozen outer folds")
    reference = {r["sha256"]: r for r in original["predictions"]}
    rows = [_completed_entry("historical_clip", "Previous CLIP embeddings", "historical_baseline",
        baseline_path, original, baseline, reference)]
    reports, context = {}, {}
    for policy, label in (("semantic", "Semantic temporal (previous best)"),
                          ("point", "Point policy (original only)")):
        path = run / f"comparison_{policy}/custom_cv_report.json"
        report = _read(path)
        if not report.get("policies"):
            raise ValueError("Custom comparison has no completed policies")
        reports[policy] = report
        for arm, comparison in report["policies"].items():
            identifier = policy + "/" + arm
            rows.append(_completed_entry(identifier, label if arm == "original_only" else policy + " + custom weak labels",
                "current_original_policy" if arm == "original_only" else "weak_label_sensitivity",
                path, comparison, baseline, reference,
                path.parent / "model" / ("model_temporal.joblib" if policy == "semantic" else "model.joblib")))
        context[policy] = {"status": report.get("status"),
            "eligible_custom_labels": report.get("auxiliary", {}).get("eligible_count"),
            "auxiliary_validation_targets": report.get("auxiliary", {}).get("validation_targets"),
            "source_report_sha256": _sha(path)}
    ssl_path = run / "unlabeled_adaptation/unlabeled_cv_report.json"
    ssl = _read(ssl_path)
    if ssl.get("custom_labels_used") != 0 or ssl.get("auxiliary_evaluation_targets") != 0:
        raise ValueError("Unlabeled adaptation report must certify zero custom labels/targets")
    for arm, comparison in ssl["policies"].items():
        adapted = arm == "original_plus_unlabeled_custom_projection"
        rows.append(_completed_entry("unlabeled/" + arm,
            "Uniform PCA +39 unlabeled custom" if adapted else "Uniform PCA; original only",
            "unlabeled_adaptation" if adapted else "matched_projection_baseline", ssl_path,
            comparison, baseline, reference, ssl_path.parent / "model/model.joblib" if adapted else None))
    if privileged_report is not None:
        path = Path(privileged_report).resolve()
        report = _read(path)
        row = _completed_entry("privileged_train_owned_notes", "Train-owned note prototypes (unverified)",
            "unverified_privileged_supervision", path, report, baseline, reference,
            path.parent / "model/model_privileged.joblib")
        row.update(notes_verified=False, inference_requires_notes=False)
        rows.append(row)
    annotation = _read(run / "annotation/proposals.json").get("summary")
    if not isinstance(annotation, dict):
        raise ValueError("Annotation output needs an aggregate summary")
    annotation = {key: value for key, value in annotation.items()
                  if isinstance(value, (int, float, bool)) or value is None}
    detail = _read(run / "detail_diagnostic/summary.json") if (run / "detail_diagnostic/summary.json").is_file() else None
    previous = None
    if previous_summary is not None and Path(previous_summary).is_file():
        old = _read(previous_summary)
        best = old["posthoc_best_development_model"]
        previous = {"model": best["model"], "metrics": best["metrics"],
            "source_summary_sha256": _sha(previous_summary), "evaluation": "previous_repeated_development_aggregate"}
    best = max(rows, key=lambda r: (r["metrics"]["balanced_accuracy"], r["metrics"]["accuracy"]))
    models = {r["model"]: {"status": "completed_development", "policies": {"measured": r["metrics"]}}
              for r in rows}
    audit = _read(dataset_audit) if dataset_audit is not None and Path(dataset_audit).is_file() else None
    if audit is not None:
        keys = ("video_count", "source_extensions", "source_total_bytes", "prepared_total_bytes",
            "all_prepared_audio_streams_zero", "audio_used_for_pairing_annotation_or_training",
            "human_verified_label_count", "independent_session_count",
            "simultaneous_two_camera_shared_scenes_user_confirmed", "cross_camera_correspondence_verified")
        audit = {key: audit.get(key) for key in keys}
    quality = _read(quality_report).get("summary") if quality_report is not None and Path(quality_report).is_file() else None
    return {"status": "completed_custom_data_experiments", "created_at": datetime.now(timezone.utc).isoformat(),
        "tool_sha256": _sha(__file__), "results_root": str(run), "n": 40,
        "baseline_report_sha256": _sha(baseline_path),
        "all_report_predictions_original_SHA_labels_folds_and_points_verified": True,
        "model_comparisons": rows, "models": models, "posthoc_best_development_model": best,
        "previous_best_development_aggregate": previous, "custom_policy_context": context,
        "annotation": annotation, "custom_unresolved": annotation.get("unresolved"),
        "custom_human_verified_labels": annotation.get("human_verified_labels", 0),
        "optional_experiments": {"privileged_notes": _privileged_runtime(privileged_runtime, privileged_report is not None)},
        "detail_diagnostic": detail, "dataset_audit": audit, "compression_quality": quality,
        "unlabeled_adaptation": {"custom_labels_used": 0, "auxiliary_evaluation_targets": 0,
            "custom_recordings": ssl.get("unlabeled_custom_recordings"),
            "custom_groups": ssl.get("unlabeled_custom_groups"), "paired_change": ssl.get("paired_change")},
        "target_accuracy": .87, "target_achieved_on_development": any(r["metrics"]["accuracy"] >= .87 for r in rows),
        "independent_87_percent_accuracy_established": False, "case_ready": False,
        "limitations": ["All original folds were repeatedly inspected; these are development comparisons, not an independent test.",
            "Custom two-camera views are paired observations of fewer scenes and share one unresolved session.",
            "Machine annotation outputs and optional train-owned notes are unverified, never custom evaluation ground truth.",
            "The new uniform-projection baseline differs from earlier supervised-sample PCA; the paired adaptation delta uses its matched baseline.",
            "Point recall is not exhaustive event precision, interval mAP or legal correctness."]}


def write_outputs(summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(output_dir / "custom_experiment_summary.json", json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    table = []
    for row in summary["model_comparisons"]:
        m = row["metrics"]
        tn, fp = m["confusion_matrix"][0]
        fn, tp = m["confusion_matrix"][1]
        table.append({"model": row["model"], "role": row["role"], "n": m["n"],
            **{key: m.get(key) for key in ("accuracy", "balanced_accuracy", "recall", "specificity", "precision", "f1", "f2", "average_precision", "roc_auc")},
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "point_recall_5s": m["localization"]["point_recall_at_tolerance"].get("5s"),
            "source_report_sha256": row["source_report_sha256"], "independent_test": False})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
    _atomic_text(output_dir / "custom_model_comparison.csv", buffer.getvalue())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = summary["model_comparisons"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, max(5, .7 * len(rows) + 2)), sharey=True)
    for i, row in enumerate(rows):
        m = row["metrics"]
        accuracy = 100 * m["accuracy"]
        color = "#a44f52" if row["role"] == "unlabeled_adaptation" else "#237e7d" if row["model"].startswith("semantic/") else "#718cae"
        left.barh(i, accuracy, color=color, height=.6)
        left.text(min(accuracy + 1, 89), i, f"{accuracy:.1f}%", va="center", fontsize=9)
        left.text(1, i + .28, f"CM {m['confusion_matrix']}", fontsize=7, color="#333333")
    left.set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=9)
    left.invert_yaxis()
    left.axvline(87, color="#b63838", linestyle="--", label="87% accuracy target")
    left.set_title("Original held-out recording accuracy")
    left.legend(loc="lower right", fontsize=8)
    series = [("balanced_accuracy", "Balanced accuracy", "o", "#386bb3"),
              ("recall", "Violation recall", "s", "#238b7b"),
              ("specificity", "Specificity", "^", "#ce8a38"),
              ("point5s", "Point recall ±5s", "D", "#8b5ca5")]
    for key, label, marker, color in series:
        xs, ys = [], []
        for i, row in enumerate(rows):
            value = (row["metrics"]["localization"]["point_recall_at_tolerance"].get("5s")
                     if key == "point5s" else row["metrics"].get(key))
            if value is not None:
                xs.append(100 * value)
                ys.append(i)
        right.scatter(xs, ys, marker=marker, color=color, s=42, label=label)
    right.set_title("Detection and point-localization diagnostics")
    right.legend(loc="lower right", fontsize=8)
    for ax in (left, right):
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 10))
        ax.set_xlabel("Percent (full 0–100 scale)")
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Custom-data experiments · same original 40 frozen test folds", fontsize=13)
    fig.text(.01, .015, "Repeated development comparisons; no independent custom-camera test. CM=[[TN,FP],[FN,TP]]. Custom labels remain unverified.", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .95))
    fig.savefig(output_dir / "custom_model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--baseline", "--baseline-cv", type=Path, default=ROOT / "reference/clip_baseline_cv.json")
    parser.add_argument("--previous-summary", type=Path, default=ROOT / "artifacts/temporal_experiment_summary.json")
    parser.add_argument("--privileged-report", type=Path)
    parser.add_argument("--privileged-runtime", type=Path,
        help="Optional aggregate execution status; failure adds metadata without inventing metrics")
    parser.add_argument("--dataset-audit", type=Path, default=ROOT / "artifacts/custom_dataset_audit.json")
    parser.add_argument("--quality-report", type=Path, default=ROOT / "artifacts/custom_compression_quality.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args(argv)
    summary = summarize(args.results_root, args.baseline, args.previous_summary, args.privileged_report,
                        args.dataset_audit, args.quality_report, args.privileged_runtime)
    write_outputs(summary, args.output_dir)
    best = summary["posthoc_best_development_model"]
    print(json.dumps({"validated_original_recordings": 40, "comparisons": len(summary["model_comparisons"]),
        "best_development_model": best["model"], "metrics": best["metrics"],
        "custom_unresolved": summary["custom_unresolved"], "target_achieved_on_development": summary["target_achieved_on_development"],
        "independent_test": False}, indent=2))


if __name__ == "__main__":
    main()

"""Audit supplied point labels and completed numeric model reports only.

Reads XLSX/JSON files, never video bytes, frames, feature arrays or inference
services. Model mistakes prioritize human review; they do not change labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datathon_ml.data import _parse_label, _read_label_rows, normalize_video_id, parse_timestamp


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_source_value(value):
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def _id_key(value):
    normalized = normalize_video_id(value)
    suffix = normalized.rsplit("-", 1)[-1]
    return (int(suffix), normalized) if suffix.isdigit() else (10**9, normalized)


def _workbook_structure(path):
    import openpyxl
    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    sheets = []
    try:
        for sheet in workbook.worksheets:
            nonempty = [row for row in sheet.iter_rows(values_only=True)
                        if any(value is not None and str(value).strip() for value in row)]
            headers = [str(value) for value in nonempty[0] if value is not None] if nonempty else []
            sheets.append({"name": sheet.title, "headers": headers,
                           "nonempty_data_rows": max(len(nonempty) - 1, 0)})
    finally:
        workbook.close()
    return sheets


def point_audit(duration, points, radii=(1., 2., 4.)):
    result = []
    for point in points:
        valid = isinstance(point, (int, float)) and math.isfinite(point) and 0 <= point < duration
        clipped = [float(radius) for radius in radii if valid and (point - radius < 0 or point + radius > duration)]
        result.append({"provided_timestamp_sec": point, "within_video": valid,
            "seconds_from_start": float(point) if valid else None,
            "seconds_to_end": float(duration - point) if valid else None,
            "context_clipped_radii_sec": clipped,
            "suggested_review_window_sec": [max(0., point - max(radii)), min(duration, point + max(radii))] if valid else None,
            "review_window_is_event_bounds": False})
    return result


def _outcome(label, prediction):
    if prediction not in (0, 1):
        return "invalid_prediction"
    return "FN" if label == 1 and prediction == 0 else "FP" if label == 0 and prediction == 1 else "correct"


def _fold_signature(predictions):
    pairs = sorted((str(row["sha256"]), int(row["fold"])) for row in predictions)
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()


def _report_integrity(report, manifest, reference):
    source = {row["sha256"]: row for row in manifest}
    rows = report["predictions"]
    by_hash = {row["sha256"]: row for row in rows}
    label_mismatches = [key for key in set(source) & set(reference) if source[key]["label"] != reference[key]["label"]]
    prediction_label_mismatches = [key for key in set(by_hash) & set(reference) if by_hash[key]["label"] != reference[key]["label"]]
    point_mismatches = [key for key in set(by_hash) & set(reference)
                        if sorted(by_hash[key].get("event_times", [])) != sorted(reference[key].get("event_times", []))]
    outer_overlaps, inner_overlaps = [], []
    for fold in report["folds"]:
        test = set(fold["test_groups"])
        if set(fold["train_groups"]) & test:
            outer_overlaps.append(fold["fold"])
        for inner in fold.get("selection", {}).get("selection_folds", []):
            train, validation = set(inner["train_groups"]), set(inner["validation_groups"])
            if train & validation or (train | validation) & test:
                inner_overlaps.append({"outer_fold": fold["fold"], "inner_fold": inner["fold"]})
    return {"manifest_records": len(manifest), "prediction_records": len(rows),
        "duplicate_manifest_sha256": len(manifest) - len(source), "duplicate_prediction_sha256": len(rows) - len(by_hash),
        "missing_reference_sha256": sorted(set(reference) - set(by_hash)),
        "unexpected_prediction_sha256": sorted(set(by_hash) - set(reference)),
        "manifest_label_mismatches": label_mismatches, "prediction_label_mismatches": prediction_label_mismatches,
        "prediction_point_mismatches": point_mismatches,
        "outer_group_overlap_folds": outer_overlaps, "inner_group_overlap_folds": inner_overlaps,
        "fold_assignment_sha256": _fold_signature(rows)}


def _temporal_trace(prediction, report, key):
    if prediction.get("window_timestamps"):
        return prediction["window_timestamps"], "primary"
    point = next((row for row in report.get("head_predictions", {}).get("point", []) if row["sha256"] == key), None)
    return (point["window_timestamps"], "separate_point_head") if point and point.get("window_timestamps") else ([], "unavailable")


def audit_annotations(labels_path, reports_root, context_radii=(1., 2., 4.)):
    labels_path, reports_root = Path(labels_path).resolve(), Path(reports_root).resolve()
    report_paths = sorted([*reports_root.rglob("cv_report.json"), *reports_root.rglob("mil_cv_report.json")])
    if not report_paths:
        raise ValueError(f"No completed CV reports below {reports_root}")
    reports, manifests, sources = {}, {}, {}
    for path in report_paths:
        name = path.parent.parent.name
        if name in reports:
            raise ValueError(f"Ambiguous duplicate report name: {name}")
        reports[name] = _read_json(path)
        manifest_path = path.parent.parent / "manifest.json"
        manifests[name] = _read_json(manifest_path)
        sources[name] = {"report_path": str(path), "manifest_path": str(manifest_path),
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    reference_name = sorted(manifests)[0]
    manifest = manifests[reference_name]
    reference = {row["sha256"]: row for row in manifest}
    if len(reference) != len(manifest):
        raise ValueError("Reference manifest contains duplicate canonical records")
    alias_map = {}
    for row in manifest:
        for alias in [row["video_id"], *row.get("aliases", [])]:
            key = normalize_video_id(alias)
            if key in alias_map and alias_map[key] != row["sha256"]:
                raise ValueError(f"Conflicting canonical alias: {key}")
            alias_map[key] = row["sha256"]
    workbook_rows, unmatched_rows, source_issues = [], [], []
    rows_by_hash = {key: [] for key in reference}
    for raw_id, raw_label, raw_time, row_number in _read_label_rows(labels_path):
        source = {"excel_row": row_number, "provided_file_id": _safe_source_value(raw_id),
            "provided_label_text": _safe_source_value(raw_label), "provided_timestamp_text": _safe_source_value(raw_time)}
        try:
            source.update(normalized_video_id=normalize_video_id(raw_id),
                          parsed_label=_parse_label(raw_label), parsed_timestamp_sec=parse_timestamp(raw_time))
            key = alias_map.get(source["normalized_video_id"])
            source["canonical_sha256"] = key
            if key is None:
                unmatched_rows.append(source)
            else:
                rows_by_hash[key].append(source)
                if source["parsed_label"] != reference[key]["label"]:
                    source_issues.append({"excel_row": row_number, "issue": "workbook_manifest_label_mismatch", "sha256": key})
        except ValueError as exc:
            source["parse_error"] = str(exc)
            source_issues.append({"excel_row": row_number, "issue": "invalid_workbook_value", "detail": str(exc)})
        workbook_rows.append(source)
    integrity = {name: _report_integrity(report, manifests[name], reference) for name, report in reports.items()}
    predictions = {name: {row["sha256"]: row for row in report["predictions"]} for name, report in reports.items()}
    queue, temporal_summary = [], {}
    for key, row in reference.items():
        source = rows_by_hash[key]
        label, points = row["label"], sorted(row.get("event_times", []))
        margins = point_audit(row["duration_sec"], points, context_radii)
        provided_points = sorted(set(item["parsed_timestamp_sec"] for item in source if item.get("parsed_timestamp_sec") is not None))
        reasons = ["event_taxonomy_not_supplied"]
        critical = False
        if not source or provided_points != points or any(item["parsed_label"] != label for item in source):
            reasons.append("source_manifest_annotation_mismatch")
            critical = True
        if label == 1:
            reasons += ["point_does_not_define_event_start_or_end", "unmarked_positive_spans_are_unlabeled"]
            if not points or any(not item["within_video"] for item in margins):
                reasons.append("missing_or_out_of_bounds_positive_point")
                critical = True
        elif points:
            reasons.append("negative_video_has_positive_point")
            critical = True
        else:
            reasons.append("normal_video_exhaustiveness_not_independently_verified")
        edge = any(item["context_clipped_radii_sec"] for item in margins)
        if edge:
            reasons.append("temporal_context_clipped_by_video_boundary")
        aliases = row.get("aliases", [row["video_id"]])
        distinct_alias_points = len(aliases) > 1 and len(provided_points) > 1
        if distinct_alias_points:
            reasons.append("duplicate_aliases_have_distinct_point_annotations_verify_event_count")
        details, folds = {}, set()
        for name, by_hash in predictions.items():
            prediction = by_hash.get(key)
            if prediction is None:
                details[name] = {"error": "missing", "prediction": None}
                critical = True
                continue
            folds.add(prediction["fold"])
            times, origin = _temporal_trace(prediction, reports[name], key)
            nearest = [min(abs(float(t) - point) for t in times) for point in points] if times else []
            trace = temporal_summary.setdefault(name, {"videos_with_trace": 0, "points_with_trace": 0,
                "max_anchor_to_nearest_sample_sec": None, "outside_video_trace_ids": []})
            if times:
                trace["videos_with_trace"] += 1
                trace["points_with_trace"] += len(nearest)
                if nearest:
                    trace["max_anchor_to_nearest_sample_sec"] = max(nearest + ([trace["max_anchor_to_nearest_sample_sec"]] if trace["max_anchor_to_nearest_sample_sec"] is not None else []))
                if min(times) < 0 or max(times) > row["duration_sec"] + .1 or any(a >= b for a, b in zip(times, times[1:])):
                    trace["outside_video_trace_ids"].append(row["video_id"])
            details[name] = {"prediction": prediction["prediction"], "error": _outcome(label, prediction["prediction"]),
                "score_uncalibrated": prediction["score"], "threshold": prediction["threshold"],
                "candidate_event_times_sec": [event["timestamp_sec"] for event in prediction.get("events", [])],
                "trace_origin": origin, "nearest_sample_distance_per_provided_point_sec": nearest}
        if len(folds) > 1:
            reasons.append("existing_model_fold_assignments_disagree")
            critical = True
        errors = sum(detail["error"] in ("FN", "FP") for detail in details.values())
        seen = sum(detail["prediction"] is not None for detail in details.values())
        positives = sum(detail["prediction"] == 1 for detail in details.values())
        if errors:
            reasons.append(f"numeric_model_errors_{errors}_of_{seen}_correlated_development_runs")
        if 0 < positives < seen:
            reasons.append("model_predictions_disagree")
        high = edge or distinct_alias_points or (seen > 0 and errors >= math.ceil(.75 * seen))
        priority = "critical" if critical else "high" if high else "medium" if errors or label == 0 else "routine"
        queue.append({"video_id": row["video_id"], "sha256": key, "aliases": aliases,
            "duration_sec": row["duration_sec"], "provided_label": label,
            "provided_event_times_sec": points, "provided_source_rows": source,
            "event_type_supplied": False, "event_boundaries_supplied": False,
            "point_audit": margins, "existing_outer_fold": next(iter(folds)) if len(folds) == 1 else None,
            "models_available": seen, "model_errors": errors, "models_predicting_violation": positives,
            "model_error_details": details, "review_priority": priority, "review_reasons": reasons,
            "review_status": "pending", "reviewed_label": None, "reviewed_event_type": None,
            "reviewed_event_start_sec": None, "reviewed_event_end_sec": None,
            "reviewed_visibility": None, "reviewed_rule_id": None, "reviewer": None, "review_notes": None})
    order = {"critical": 0, "high": 1, "medium": 2, "routine": 3}
    queue.sort(key=lambda item: (order[item["review_priority"]], -item["model_errors"], _id_key(item["video_id"])))
    for rank, row in enumerate(queue, 1):
        row["review_rank"] = rank
    sheets = _workbook_structure(labels_path)
    all_valid = [point["within_video"] for row in queue for point in row["point_audit"]]
    durations = {}
    for label in (0, 1):
        values = [row["duration_sec"] for row in queue if row["provided_label"] == label]
        durations[str(label)] = {"n": len(values), "median_sec": statistics.median(values),
                                "mean_sec": statistics.mean(values), "at_most_30_sec": sum(value <= 30 for value in values)}
    return {"schema_version": 2, "audit_mode": "xlsx_and_numeric_json_only_no_media_or_external_inference",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_workbook": {"path": str(labels_path), "sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            "sheets": sheets, "provided_rows": len(workbook_rows),
            "supplied_fields": ["video_identifier", "positive_point_timestamp", "binary_violation_label"],
            "semantic_event_types_supplied": False, "event_boundaries_supplied": False,
            "normal_annotation_exhaustiveness_independently_verified": False},
        "reference_manifest_model": reference_name, "completed_models": sorted(reports), "model_sources": sources,
        "summary": {"canonical_videos": len(queue), "label_counts": dict(Counter(str(row["provided_label"]) for row in queue)),
            "provided_positive_points": len(all_valid), "valid_points": sum(all_valid),
            "videos_with_context_clipping": [row["video_id"] for row in queue if any(point["context_clipped_radii_sec"] for point in row["point_audit"])],
            "duplicate_groups": [{"video_id": row["video_id"], "aliases": row["aliases"],
                "provided_points_sec": row["provided_event_times_sec"]} for row in queue if len(row["aliases"]) > 1],
            "review_priority_counts": dict(Counter(row["review_priority"] for row in queue)),
            "label_duration_distribution_diagnostic_only": durations,
            "provided_labels_modified": 0, "semantic_labels_invented": 0,
            "models_are_correlated_not_independent_annotation_votes": True},
        "unmatched_workbook_rows": unmatched_rows, "source_issues": source_issues,
        "report_integrity": integrity,
        "existing_outer_folds_identical_across_models": len({item["fold_assignment_sha256"] for item in integrity.values()}) == 1,
        "temporal_trace_coverage": temporal_summary,
        "evaluation_policy": {
            "folds": "Keep the existing SHA/group outer folds unchanged. Re-run all compared models after reviewed annotations are versioned; obtain new filming sessions/camera footage for a locked independent test.",
            "point_metrics_now": "One-to-one maximum-cardinality recall at +/-1, +/-2, +/-5 seconds. Missed points count against recall; timing median/p90 are conditional diagnostics.",
            "not_identifiable_now": ["exhaustive event precision", "interval temporal IoU", "semantic event-type macro-F1", "regulatory-grounding accuracy"],
            "positive_windows": "Unmarked positive-video spans remain unlabeled. Suggested review windows are navigation aids, never inferred event intervals.",
            "negative_windows": "Preserve provided negative labels. Ask a reviewer to confirm exhaustive normality before treating every negative-video window as a trusted background/action negative in revised training.",
            "boundary_events": "Review events near a video edge for visibility and censoring; mark unknown/unobservable portions instead of inventing normal labels.",
            "review_queue": "Error counts are human-review priorities only; do not auto-relabel, invert model scores, add filename/duration features or move difficult videos between folds.",
            "model_scores": "Scores from different models/folds are uncalibrated; do not average them as annotation confidence."},
        "review_queue": queue}


def write_outputs(audit, json_path, csv_path):
    json_path, csv_path = Path(json_path), Path(csv_path)
    for path in (json_path, csv_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    models = audit["completed_models"]
    fields = ["review_rank", "review_priority", "video_id", "sha256", "aliases", "duration_sec",
        "provided_label", "provided_label_texts", "provided_event_times_sec", "provided_excel_rows",
        "existing_outer_fold", "models_available", "model_errors", "models_predicting_violation",
        "context_clipped_radii_sec", "review_reasons", "review_status", "reviewed_label", "reviewed_event_type",
        "reviewed_event_start_sec", "reviewed_event_end_sec", "reviewed_visibility", "reviewed_rule_id", "reviewer", "review_notes"]
    for model in models:
        fields += [f"{model}_prediction", f"{model}_error", f"{model}_score_uncalibrated", f"{model}_candidate_times_sec"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in audit["review_queue"]:
            row = {key: source.get(key) for key in fields if key in source}
            row.update(aliases=";".join(source["aliases"]),
                provided_label_texts=";".join(sorted({item["provided_label_text"] for item in source["provided_source_rows"]})),
                provided_event_times_sec=json.dumps(source["provided_event_times_sec"]),
                provided_excel_rows=";".join(str(item["excel_row"]) for item in source["provided_source_rows"]),
                context_clipped_radii_sec=json.dumps(sorted({radius for point in source["point_audit"] for radius in point["context_clipped_radii_sec"]})),
                review_reasons=";".join(source["review_reasons"]))
            for model in models:
                detail = source["model_error_details"][model]
                row.update({f"{model}_prediction": detail["prediction"], f"{model}_error": detail["error"],
                    f"{model}_score_uncalibrated": detail.get("score_uncalibrated"),
                    f"{model}_candidate_times_sec": json.dumps(detail.get("candidate_event_times_sec", []))})
            writer.writerow(row)


def self_test():
    class AuditTests(unittest.TestCase):
        def test_point_is_not_expanded_to_an_event_interval(self):
            result = point_audit(5., [4., 5., -1.])
            self.assertTrue(result[0]["within_video"])
            self.assertEqual(result[0]["context_clipped_radii_sec"], [2., 4.])
            self.assertFalse(result[0]["review_window_is_event_bounds"])
            self.assertFalse(result[1]["within_video"])
            self.assertFalse(result[2]["within_video"])
        def test_numeric_mismatches_detected_without_relabel(self):
            original = {"sha256": "a", "label": 1, "event_times": [3.]}
            report = {"predictions": [{"sha256": "a", "label": 0, "event_times": [], "fold": 0}],
                      "folds": [{"fold": 0, "train_groups": ["train"], "test_groups": ["test"], "selection": {}}]}
            result = _report_integrity(report, [original], {"a": original})
            self.assertEqual(result["prediction_label_mismatches"], ["a"])
            self.assertEqual(result["prediction_point_mismatches"], ["a"])
            self.assertEqual(original["label"], 1)
            self.assertEqual(_outcome(1, 0), "FN")
            self.assertEqual(_outcome(0, 1), "FP")
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "open" / "open_labels.xlsx")
    parser.add_argument("--reports-root", type=Path, default=ROOT / "artifacts" / "colab_results" / "runs")
    parser.add_argument("--json-output", type=Path, default=ROOT / "artifacts" / "annotation_audit_v2.json")
    parser.add_argument("--csv-output", type=Path, default=ROOT / "artifacts" / "annotation_review_v2.csv")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    audit = audit_annotations(args.labels, args.reports_root)
    write_outputs(audit, args.json_output, args.csv_output)
    print(json.dumps({"summary": audit["summary"], "completed_models": audit["completed_models"],
        "existing_folds_unchanged": audit["existing_outer_folds_identical_across_models"],
        "source_issues": audit["source_issues"], "json": str(args.json_output), "csv": str(args.csv_output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

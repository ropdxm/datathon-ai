"""Write local numeric annotation proposals; never modify supplied ground truth.

Only manifest JSON and numeric semantic NPZ caches are read. No videos, frames,
descriptions, model services or hosted inference are used. Standard output is
limited to aggregate counts; per-record observations stay in the local CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datathon_ml.model import _canonical_records
from datathon_ml.temporal import CONCEPT_NAMES, TemporalConfig, load_states, transform_states


STATUS = "machine_proposal_requires_human_review"
REVIEW_FIELDS = ("reviewed_label", "reviewed_event_type", "reviewed_start_sec", "reviewed_end_sec", "reviewer", "review_notes")
FIELDS = ("proposal_id", "video_id", "sha256", "provided_label", "provided_point_index",
    "provided_timestamp_sec", "provided_aliases", "status", "annotation_scope",
    "unmarked_positive_spans", "nearest_state_timestamp_sec", "nearest_state_distance_sec",
    "semantic_sample_index", "state_scores_calibrated", "proposed_rule_ids",
    "proposed_start_sec", "proposed_end_sec", "proposed_rule_peak_sec",
    "proposed_rule_support_uncalibrated", "rule_interval_origin", "rule_support_aggregation",
    "nearby_rule_window_radius_sec", "legal_violation_established", *REVIEW_FIELDS,
    *("proposed_state_score__" + name for name in CONCEPT_NAMES))


def _component_events(events):
    """Prefer actual component intervals over merged unions when available."""
    for event in events:
        components = event.get("component_events") or event.get("components")
        if components:
            for component in components:
                yield component, "actual_rule_component"
        else:
            yield event, "rule_candidate_interval"


def _nearby_events(events, point, radius):
    result, seen = [], set()
    for event, origin in _component_events(events):
        start, end = float(event["start_sec"]), float(event["end_sec"])
        if not math.isfinite(start) or not math.isfinite(end) or start > end:
            raise ValueError("Rule proposal has invalid numeric event bounds")
        if start > point + radius or end < point - radius:
            continue
        rules = event.get("rule_ids", [event["rule_id"]] if "rule_id" in event else [])
        key = (start, end, tuple(sorted(rules)))
        if key not in seen:
            seen.add(key)
            result.append((event, origin, rules))
    return sorted(result, key=lambda item: (item[0]["start_sec"], item[0]["end_sec"], item[2]))


def proposal_rows(record, states, times, events, radius=2.):
    """One row per supplied point and overlapping actual rule interval.

    If no interval overlaps +/-radius, one point-only row is emitted with blank
    proposed bounds. Multiple overlapping rules produce separate rows so their
    intervals and support scores are not invented by taking a union.
    """
    points = record.get("event_times", [])
    if record["label"] == 0:
        if points:
            raise ValueError("A provided negative video unexpectedly contains positive points")
        return
    for point_index, point in enumerate(points):
        if not math.isfinite(point) or point < 0:
            raise ValueError("Invalid supplied point timestamp")
        if "duration_sec" in record and point >= record["duration_sec"]:
            raise ValueError("Supplied point lies outside its provided video duration")
        nearest = int(np.argmin(np.abs(times - point)))
        nearby = _nearby_events(events, point, radius)
        for candidate_index, candidate in enumerate(nearby or [None]):
            row = {"proposal_id": f"{record['sha256']}:{point_index}:{candidate_index}",
                "video_id": record.get("video_id", record["sha256"]), "sha256": record["sha256"],
                "provided_label": record["label"], "provided_point_index": point_index,
                "provided_timestamp_sec": point, "provided_aliases": json.dumps(record.get("aliases", []), ensure_ascii=False),
                "status": STATUS, "annotation_scope": "provided_positive_point",
                "unmarked_positive_spans": "unlabeled", "nearest_state_timestamp_sec": float(times[nearest]),
                "nearest_state_distance_sec": abs(float(times[nearest]) - point),
                "semantic_sample_index": nearest, "state_scores_calibrated": False,
                "proposed_rule_ids": "[]", "proposed_start_sec": None, "proposed_end_sec": None,
                "proposed_rule_peak_sec": None, "proposed_rule_support_uncalibrated": None,
                "rule_interval_origin": "no_overlapping_rule_candidate", "rule_support_aggregation": None,
                "nearby_rule_window_radius_sec": radius, "legal_violation_established": False,
                **{name: None for name in REVIEW_FIELDS}}
            row.update({"proposed_state_score__" + name: float(states[nearest, index])
                        for index, name in enumerate(CONCEPT_NAMES)})
            if candidate is not None:
                event, origin, rules = candidate
                row.update(proposed_rule_ids=json.dumps(rules), proposed_start_sec=float(event["start_sec"]),
                    proposed_end_sec=float(event["end_sec"]), proposed_rule_peak_sec=event.get("timestamp_sec"),
                    proposed_rule_support_uncalibrated=event.get("score", event.get("support_score")),
                    rule_interval_origin=origin,
                    rule_support_aggregation=event.get("score_aggregation", "single_component" if origin == "actual_rule_component" else "unspecified_rule_candidate"))
            yield row


def write_proposals(manifest_path, feature_dir, output_path, radius=2.):
    manifest_path, feature_dir, output_path = Path(manifest_path).resolve(), Path(feature_dir).resolve(), Path(output_path).resolve()
    if output_path == manifest_path or output_path.suffix.lower() != ".csv":
        raise ValueError("Output must be a separate machine-proposal CSV")
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("Review radius must be finite and non-negative")
    if output_path.exists():
        with output_path.open(encoding="utf-8-sig", newline="") as handle:
            if any(any(str(row.get(name, "")).strip() for name in REVIEW_FIELDS) for row in csv.DictReader(handle)):
                raise ValueError("Output contains human review fields; preserve it and choose a new proposal output")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or any(row.get("label") not in (0, 1) for row in raw):
        raise ValueError("Expected a validated manifest with preserved binary labels")
    records = _canonical_records(raw)
    config = TemporalConfig()
    summary = {"canonical_records": len(records), "positive_records": 0, "normal_records_skipped": 0,
        "provided_points": 0, "proposal_rows": 0, "points_with_overlapping_rule_candidate": 0,
        "rows_with_rule_candidate": 0, "semantic_state_columns": len(CONCEPT_NAMES),
        "ground_truth_labels_modified": 0, "verified_event_types_created": 0,
        "human_review_required": True}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=output_path.parent, suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for record in records:
                if record["label"] == 0:
                    summary["normal_records_skipped"] += 1
                    continue
                summary["positive_records"] += 1
                states, times, metadata = load_states(feature_dir / f"{record['sha256']}.npz")
                # Point-sampled feature caches never certify absence of an action
                # between frames. This tool deliberately keeps omission rules off.
                derived = transform_states(states, times, config, continuous_window_coverage=False)
                rows = list(proposal_rows(record, states, times, derived["events"], radius))
                summary["provided_points"] += len(record.get("event_times", []))
                covered = {row["provided_point_index"] for row in rows if row["proposed_start_sec"] is not None}
                summary["points_with_overlapping_rule_candidate"] += len(covered)
                for row in rows:
                    writer.writerow(row)
                    summary["proposal_rows"] += 1
                    summary["rows_with_rule_candidate"] += row["proposed_start_sec"] is not None
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return summary


def self_test():
    class ProposalTests(unittest.TestCase):
        def test_point_only_proposal_preserves_source_and_blanks_review(self):
            record = {"sha256": "synthetic", "label": 1, "event_times": [3.], "duration_sec": 10.}
            states, times = np.full((3, 18), .5), np.array([0., 4., 8.])
            rows = list(proposal_rows(record, states, times, []))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["provided_timestamp_sec"], 3.)
            self.assertEqual(rows[0]["nearest_state_timestamp_sec"], 4.)
            self.assertEqual(rows[0]["status"], STATUS)
            self.assertIsNone(rows[0]["proposed_start_sec"])
            self.assertTrue(all(rows[0][name] is None for name in REVIEW_FIELDS))
            self.assertEqual(sum(name.startswith("proposed_state_score__") for name in rows[0]), 18)
            self.assertEqual(record["event_times"], [3.])
        def test_actual_components_are_separate_and_far_events_do_not_supply_bounds(self):
            record = {"sha256": "synthetic", "label": 1, "event_times": [4.]}
            components = [{"start_sec": 2., "end_sec": 3., "rule_ids": ["rule_a"], "score": .6},
                          {"start_sec": 5., "end_sec": 6., "rule_ids": ["rule_b"], "score": .9}]
            events = [{"start_sec": 2., "end_sec": 6., "rule_ids": ["rule_a", "rule_b"], "component_events": components},
                      {"start_sec": 9., "end_sec": 10., "rule_ids": ["far"]}]
            rows = list(proposal_rows(record, np.full((3, 18), .5), np.array([0., 4., 8.]), events))
            self.assertEqual(len(rows), 2)
            self.assertEqual([(row["proposed_start_sec"], row["proposed_end_sec"]) for row in rows], [(2., 3.), (5., 6.)])
            self.assertEqual([row["proposed_rule_support_uncalibrated"] for row in rows], [.6, .9])
            self.assertEqual(list(proposal_rows({"sha256": "normal", "label": 0, "event_times": []}, np.full((3, 18), .5), np.arange(3), [])), [])
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProposalTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--radius-sec", type=float, default=2.)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.manifest is None or args.feature_dir is None or args.output is None:
        parser.error("--manifest, --feature-dir and --output are required")
    summary = write_proposals(args.manifest, args.feature_dir, args.output, args.radius_sec)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Observable sequence rules and nested training/inference boundaries."""
import io
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import joblib
import numpy as np

from datathon_ml.temporal import (CONCEPT_NAMES, TemporalConfig, fit_temporal_final,
    _merge_events, frozen_outer_splits, hysteresis, load_states, predict_temporal_features, run_temporal_cross_validation,
    transform_states)


def states(length=12):
    x = np.zeros((length, 1, len(CONCEPT_NAMES)), dtype=np.float32)
    for name in ("cashier_present", "scene_clear", "hands_visible", "cash_area_visible"):
        x[:, 0, CONCEPT_NAMES.index(name)] = .95
    return x


def set_state(x, name, value, region=slice(None)):
    x[region, 0, CONCEPT_NAMES.index(name)] = value


def synthetic_cache(root, n=18):
    rng = np.random.default_rng(143)
    records = []
    metadata = {"backbone": "synthetic_semantic18", "concept_names": list(CONCEPT_NAMES),
                "continuous_window_coverage": False, "preprocessing": "fixed-synthetic-state-schema-v1"}
    for i in range(n):
        x = states()
        x += rng.uniform(0, .02, size=x.shape).astype(np.float32)
        label = i % 2
        if label:
            set_state(x, "cashier_present", .05, slice(5, 10))
            set_state(x, "loose_cash_visible", .95, slice(4, 11))
            set_state(x, "transaction_active", .9, slice(3, 10))
        key = f"{i:064x}"
        np.savez(root / f"{key}.npz", features=x, timestamps=np.arange(len(x), dtype=np.float32),
                 config_json=json.dumps(metadata))
        records.append({"sha256": key, "group_id": key, "video_id": "untrusted_" + str(1000 - i),
                        "label": label, "event_times": [6.] if label else []})
    return records


class TemporalTests(unittest.TestCase):
    def test_sustained_unattended_cash_and_departure_are_observable(self):
        x = states()
        set_state(x, "cashier_present", .05, slice(5, 11))
        set_state(x, "loose_cash_visible", .95, slice(4, 11))
        set_state(x, "access_open", .95, slice(5, 11))
        derived = transform_states(x, np.arange(len(x)))
        self.assertGreater(derived["rule_scores"]["unattended_cash"], .8)
        self.assertGreater(derived["rule_scores"]["departure_during_service"], .8)
        self.assertGreater(derived["rule_scores"]["unsecured_leave"], .8)
        self.assertEqual(len(derived["events"]), 1)  # dependent rules merged
        self.assertFalse(derived["events"][0]["legal_violation_established"])

    def test_short_absence_occlusion_unknown_and_gaps_do_not_bridge_dwell(self):
        for issue in ("short", "occluded", "unknown", "gap"):
            x = states(8)
            set_state(x, "loose_cash_visible", .95)
            region = slice(3, 5) if issue == "short" else slice(3, 6)
            set_state(x, "cashier_present", .05, region)
            times = np.arange(8, dtype=float)
            if issue == "occluded":
                set_state(x, "scene_clear", .5, region)
            if issue == "unknown":
                set_state(x, "cashier_present", .5, region)
            if issue == "gap":
                times[4:] += 10
            derived = transform_states(x, times)
            self.assertEqual(derived["rule_score"], 0., issue)

    def test_unknown_hysteresis_stops_even_after_high_confidence(self):
        actual = hysteresis(np.array([.9, .6, .5, .65, .9]), np.arange(5), TemporalConfig())
        np.testing.assert_array_equal(actual, [True, True, False, False, True])

    def test_handover_omission_requires_complete_certified_visible_cycle(self):
        x = states(10)
        set_state(x, "transaction_start", .95, slice(1, 2))
        set_state(x, "cash_handover", .95, slice(5, 6))
        set_state(x, "transaction_end", .95, slice(8, 9))
        t = np.arange(10)
        score = lambda y, certified: transform_states(y, t, continuous_window_coverage=certified)["rule_scores"]["handover_without_visible_count"]
        self.assertEqual(score(x, False), 0.)
        self.assertGreater(score(x, True), .8)
        for name, value, region in (("cash_counting", .5, slice(3, 4)), ("cash_counting", .95, slice(3, 4)),
                                    ("hands_visible", .5, slice(3, 4)), ("transaction_start", 0., slice(None)),
                                    ("transaction_end", 0., slice(None))):
            changed = x.copy()
            set_state(changed, name, value, region)
            self.assertEqual(score(changed, True), 0., name)

    def test_order_features_distinguish_count_before_handover(self):
        x = states(10)
        set_state(x, "cash_counting", .95, slice(2, 3))
        set_state(x, "cash_handover", .95, slice(7, 8))
        a = transform_states(x, np.arange(10))
        b = transform_states(x[::-1].copy(), np.arange(10))
        index = a["feature_names"].index("cash_counting.before.cash_handover")
        self.assertGreater(a["features"][index], b["features"][index])
        self.assertNotIn("video_duration", a["feature_names"])

    def test_order_features_do_not_bridge_gaps_or_unknown_prior_state(self):
        x = states(10)
        set_state(x, "cash_counting", .95, slice(2, 3))
        set_state(x, "cash_handover", .95, slice(7, 8))
        times = np.arange(10, dtype=float)
        times[4:] += 10
        separated = transform_states(x, times)
        index = separated["feature_names"].index("cash_counting.before.cash_handover")
        self.assertEqual(separated["features"][index], 0.)
        set_state(x, "cash_counting", .5, slice(2, 3))
        uncertain = transform_states(x, np.arange(10))
        self.assertEqual(uncertain["features"][index], 0.)

    def test_merged_events_preserve_rule_specific_support_intervals(self):
        components = [{"rule_ids": ["unattended_cash"], "start_sec": 0., "end_sec": 10.,
                       "timestamp_sec": 1., "score": .6, "explanation": "First observable condition."},
                      {"rule_ids": ["unsecured_leave"], "start_sec": 4., "end_sec": 6.,
                       "timestamp_sec": 5., "score": .95, "explanation": "Second observable condition."}]
        merged = _merge_events(components, 1.)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["component_events"], components)
        self.assertEqual(merged[0]["score"], .95)
        self.assertEqual(merged[0]["score_scope"], "maximum_component_score_not_union_confidence")
        self.assertEqual(merged[0]["component_events"][1]["start_sec"], 4.)
        self.assertEqual(merged[0]["component_events"][1]["end_sec"], 6.)

    def test_nested_groups_calibration_and_export_inference_ignore_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = synthetic_cache(root)
            config = TemporalConfig(outer_folds=3, inner_folds=2, logistic_cs=(.01, .1))
            with redirect_stdout(io.StringIO()):
                report = run_temporal_cross_validation(records, root, root / "evaluation", config)
            self.assertEqual(report["metrics"]["n"], len(records))
            self.assertGreaterEqual(report["metrics"]["balanced_accuracy"], .8)
            self.assertEqual(report["frozen_rules"]["threshold"], .5)
            for fold in report["folds"]:
                heldout = set(fold["test_groups"])
                self.assertFalse(set(fold["preprocessing_fit_sha256"]) & heldout)
                self.assertFalse(set(fold["selection"]["calibration_fit_sha256"]) & heldout)
                for inner in fold["selection"]["selection_folds"]:
                    self.assertFalse(set(inner["train_groups"]) & set(inner["validation_groups"]))
                    self.assertFalse(set(inner["preprocessing_fit_sha256"]) & heldout)
            artifact_path = fit_temporal_final(records, root, root / "final", config)
            cache = root / f"{records[1]['sha256']}.npz"
            before = predict_temporal_features(cache, artifact_path)
            with np.load(cache) as stored:
                payload = {key: stored[key] for key in stored.files}
            payload["label"] = np.array(0)
            payload["event_times"] = np.array([9999.])
            renamed = root / "other_name.npz"
            np.savez(renamed, **payload)
            self.assertEqual(before, predict_temporal_features(renamed, artifact_path))
            metadata = json.loads(str(payload["config_json"].item()))
            for changed in ({**metadata, "continuous_window_coverage": True},
                            {**metadata, "coverage_certification": {"source": "new-claim"}}):
                changed_payload = {**payload, "config_json": json.dumps(changed)}
                np.savez(renamed, **changed_payload)
                with self.assertRaisesRegex(ValueError, "coverage/certification"):
                    predict_temporal_features(renamed, artifact_path)
            artifact = joblib.load(artifact_path)
            self.assertFalse(artifact["provenance"]["selection_score_is_test_score"])
            self.assertNotIn("event_times", artifact)
            self.assertTrue(any("exposure length" in text for text in artifact["limitations"]))
            for field in ("code_sha256", "temporal_code_sha256"):
                tampered = {**artifact, "provenance": {**artifact["provenance"], field: "0" * 64}}
                tampered_path = root / f"tampered_{field}.joblib"
                joblib.dump(tampered, tampered_path)
                with self.assertRaisesRegex(ValueError, "code fingerprint"):
                    predict_temporal_features(cache, tampered_path)
            from temporal_experiment import main
            with redirect_stdout(io.StringIO()):
                cli_result = main(["predict", "--feature", str(cache), "--model", str(artifact_path),
                                   "--output-dir", str(root / "cli_prediction")])
            self.assertEqual(before, cli_result)
            self.assertTrue((root / "cli_prediction" / "prediction.json").is_file())
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(list(reversed(records))), encoding="utf-8")
            reference = root / "evaluation" / "temporal_cv_report.json"
            with redirect_stdout(io.StringIO()):
                frozen_report = main(["cv", "--manifest", str(manifest), "--feature-dir", str(root),
                    "--output-dir", str(root / "cli_frozen"), "--outer-folds", "3", "--inner-folds", "2",
                    "--folds-from", str(reference)])
            original_map = {row["sha256"]: row["fold"] for row in report["predictions"]}
            frozen_map = {row["sha256"]: row["fold"] for row in frozen_report["predictions"]}
            self.assertEqual(original_map, frozen_map)
            self.assertEqual(frozen_report["fold_source"]["file_sha256"], hashlib.sha256(reference.read_bytes()).hexdigest())
            self.assertTrue(all(fold["test_sha256_digest"] for fold in frozen_report["folds"]))

    def test_frozen_fold_map_validates_exact_coverage_and_current_groups(self):
        records = [{"sha256": f"{i:064x}", "label": i % 2, "group_id": str(i // 2)} for i in range(12)]
        reference = {"config": {"seed": 42}, "predictions": [
            {"sha256": r["sha256"], "label": r["label"], "fold": (i // 2) % 3} for i, r in enumerate(records)]}
        config = TemporalConfig(outer_folds=3, inner_folds=2)
        first, provenance = frozen_outer_splits(records, reference, config)
        reversed_records = records[::-1]
        second, _ = frozen_outer_splits(reversed_records, reference, config)
        self.assertEqual([{records[i]["sha256"] for i in test} for _, test in first],
                         [{reversed_records[i]["sha256"] for i in test} for _, test in second])
        self.assertTrue(provenance["fold_assignment_sha256"])
        missing = {**reference, "predictions": reference["predictions"][:-1]}
        with self.assertRaisesRegex(ValueError, "coverage"):
            frozen_outer_splits(records, missing, config)
        split_group = {**reference, "predictions": [dict(row) for row in reference["predictions"]]}
        split_group["predictions"][1]["fold"] = 1
        with self.assertRaisesRegex(ValueError, "group"):
            frozen_outer_splits(records, split_group, config)
        with self.assertRaisesRegex(ValueError, "seed"):
            frozen_outer_splits(records, reference, TemporalConfig(outer_folds=3, seed=43))

    def test_named_schema_reorders_correctly_and_rejects_missing_concepts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "named.npz"
            x = states(3)
            names = list(CONCEPT_NAMES)[::-1]
            np.savez(path, features=x[:, :, ::-1], timestamps=np.arange(3), config_json=json.dumps({"concept_names": names}))
            actual, _, _ = load_states(path)
            np.testing.assert_array_equal(actual, x[:, 0])
            np.savez(path, features=x[:, :, :-1], timestamps=np.arange(3), config_json=json.dumps({"concept_names": names[:-1]}))
            with self.assertRaises(ValueError):
                load_states(path)


if __name__ == "__main__":
    unittest.main()

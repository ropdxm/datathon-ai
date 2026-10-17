"""Behavioral tests for the leakage and weak-supervision boundaries."""
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import joblib
import numpy as np

from datathon_ml.model import (TrainingConfig, _candidates, _canonical_records, _events, _match_points, _prepare, _select,
    _training_data, classification_metrics, fit_final, grouped_splits,
    load_features, predict_features, run_cross_validation, window_features)
from datathon_ml.model import semantic_feature_config


def synthetic_dataset(directory, n=18):
    rng = np.random.default_rng(123)
    records = []
    for i in range(n):
        label = i % 2
        features = rng.normal(0, .04, size=(13, 2, 8)).astype(np.float32)
        features[:, :, 1] += 1
        if label:
            features[5:8, :, 0] += 2
        key = f"{i:064x}"
        np.savez_compressed(Path(directory) / f"{key}.npz", features=features,
            timestamps=np.arange(13, dtype=np.float32), config_json=json.dumps({"encoder": "synthetic"}))
        records.append({"video_id": f"untrusted_name_{10000-i}", "path": "unused.mov",
            "sha256": key, "group_id": key, "label": label,
            "event_times": [6.] if label else []})
    return records


def quick_config(**kwargs):
    return TrainingConfig(outer_folds=3, inner_folds=2, pca_components=4,
        logistic_cs=(.1, 1.), rbf_cs=(), aggregations=("smooth_max", "topk"),
        **kwargs)


class WeakSupervisionTests(unittest.TestCase):
    def test_multiscale_features_are_finite_at_boundaries(self):
        f = np.random.default_rng(1).normal(size=(5, 2, 6)).astype(np.float32)
        actual = window_features(f, np.arange(5), (1., 2., 4.))
        self.assertEqual(actual.shape, (5, 2 * 6 * 11))
        self.assertTrue(np.isfinite(actual).all())
        singleton = window_features(f[:1], np.array([0.]), (1.,))
        self.assertTrue(np.isfinite(singleton).all())

    def test_unmarked_positive_windows_never_become_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = synthetic_dataset(tmp, n=2)
            config = quick_config()
            data = _prepare(records, tmp, config)
            x, y, weights = _training_data(records, data, "point", config)
            self.assertEqual(int((y == 1).sum()), 3)  # 5, 6, 7 around point 6
            self.assertEqual(int((y == 0).sum()), 13)  # only the normal video
            self.assertAlmostEqual(weights[y == 0].sum(), weights[y == 1].sum())
            self.assertEqual(len(x), len(y))

    def test_dedup_merges_point_annotations_and_rejects_conflicts(self):
        row = {"video_id": "a", "sha256": "abc", "label": 1, "event_times": [3.]}
        rows = _canonical_records([row, {**row, "video_id": "b", "event_times": [9.]}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_times"], [3., 9.])
        with self.assertRaises(ValueError):
            _canonical_records([row, {**row, "label": 0, "event_times": []}])

    def test_grouped_folds_keep_related_recordings_together(self):
        records = [{"sha256": str(i), "group_id": str(i // 2), "label": (i // 2) % 2} for i in range(24)]
        folds = grouped_splits(records, 3)
        self.assertEqual(len(folds), 3)
        heldout = []
        for train, test in folds:
            self.assertFalse({records[i]["group_id"] for i in train} & {records[i]["group_id"] for i in test})
            heldout.extend(test)
        self.assertEqual(sorted(heldout), list(range(24)))

    def test_nested_validation_and_final_inference_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = synthetic_dataset(root)
            config = quick_config()
            report = run_cross_validation(records, root, root / "results", config)
            self.assertEqual(report["metrics"]["n"], 18)
            self.assertGreaterEqual(report["metrics"]["balanced_accuracy"], .8)
            self.assertEqual(len({p["sha256"] for p in report["predictions"]}), 18)
            for fold in report["folds"]:
                self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
                for inner in fold["selection"]["selection_folds"]:
                    self.assertFalse(set(inner["train_groups"]) & set(inner["validation_groups"]))
                    self.assertFalse(set(inner["validation_groups"]) & set(fold["test_groups"]))
            model_path = fit_final(records, root, root / "final", config)
            cache_path = root / f"{records[1]['sha256']}.npz"
            before = predict_features(cache_path, model_path)
            with np.load(cache_path) as src:
                copied = {k: src[k] for k in src.files}
            # Extra annotations in a cache cannot affect prediction: never read.
            copied["event_times"] = np.array([12345.])
            renamed = root / "completely_different_filename.npz"
            np.savez(renamed, **copied)
            after = predict_features(renamed, model_path)
            self.assertEqual(before, after)
            artifact = joblib.load(model_path)
            self.assertFalse(artifact["provenance"]["selection_score_is_test_score"])
            self.assertNotIn("event_times", artifact)
            gpu_metadata = {"backbone": "dinov2_vits14", "image_size": 224,
                "views": ["full", "middle"], "sample_fps": 2., "weights_sha256": "abc",
                "hub_repo": "pinned-dinov2", "precision": "cuda-float16", "torch_version": "2.11"}
            cpu_metadata = {**gpu_metadata, "precision": "cpu-float32", "torch_version": "2.9"}
            artifact["feature_config"] = gpu_metadata
            artifact["feature_semantic_config"] = semantic_feature_config(gpu_metadata)
            portable_model = root / "portable.joblib"
            joblib.dump(artifact, portable_model)
            copied["config_json"] = json.dumps(cpu_metadata)
            np.savez(renamed, **copied)
            portable_prediction = predict_features(renamed, portable_model)
            self.assertEqual(portable_prediction["prediction"], before["prediction"])
            self.assertEqual(portable_prediction["score"], before["score"])
            self.assertTrue(portable_prediction["inference_warnings"])
            for semantic_change in ({"weights_sha256": "different"}, {"backbone": "different"}):
                copied["config_json"] = json.dumps({**cpu_metadata, **semantic_change})
                np.savez(renamed, **copied)
                with self.assertRaisesRegex(ValueError, "semantic"):
                    predict_features(renamed, portable_model)

    def test_metric_confusion_and_continuous_event_merging(self):
        m = classification_metrics([0, 0, 1, 1], [.1, .8, .2, .9], [0, 1, 0, 1])
        self.assertEqual(m["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(m["specificity"], .5)
        events = _events(np.ones(30) * .9, np.arange(30), .5, quick_config())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "potential_violation")

    def test_point_matching_maximizes_cardinality_before_distance(self):
        self.assertEqual(_match_points([1., 3.], [0., 1.9], 2.), 2)
        self.assertEqual(_match_points([1., 3.], [1.9], 2.), 1)
        self.assertEqual(_match_points([1., 3.], [20.], 2.), 0)
        self.assertEqual(_match_points([], [1.], 2.), 0)

    def test_tiny_training_fallback_has_fixed_threshold_and_feasible_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = synthetic_dataset(tmp, n=2)
            records[1]["event_times"] = []  # pooled is trainable; point head is not
            config = quick_config()
            data = _prepare(records, tmp, config)
            selection = _select(records, data, config)
            self.assertEqual(selection["candidate"]["head"], "pooled")
            self.assertEqual(selection["threshold"], .5)
            self.assertEqual(selection["selection_folds"], [])
            self.assertIsNone(selection["selection_metrics"])

    def test_cache_integrity_when_embedded_hashes_are_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = synthetic_dataset(tmp, n=2)
            key = records[0]["sha256"]
            cache_path = Path(tmp) / f"{key}.npz"
            with np.load(cache_path) as cache:
                payload = {k: cache[k] for k in cache.files}
            config_json = str(payload["config_json"].item())
            payload["config_hash"] = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
            payload["sha256"] = key
            np.savez(cache_path, **payload)
            load_features(cache_path)
            wrong_name = Path(tmp) / ("f" * 64 + ".npz")
            np.savez(wrong_name, **payload)
            with self.assertRaisesRegex(ValueError, "embedded source"):
                load_features(wrong_name)
            payload["config_json"] = json.dumps({"encoder": "tampered"})
            np.savez(cache_path, **payload)
            with self.assertRaisesRegex(ValueError, "configuration hash"):
                load_features(cache_path)

    def test_pooled_winner_preserves_separate_point_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = synthetic_dataset(tmp, n=6)
            pooled = {"candidate": {"head": "pooled", "classifier": "logistic", "C": .1},
                      "aggregation": "max", "threshold": .5, "selection_metrics": None}
            point = {"candidate": {"head": "point", "classifier": "logistic", "C": .1},
                     "aggregation": "smooth_max", "threshold": .5, "selection_metrics": None}
            selection = {**pooled, "leaderboard": [pooled, point], "selection_folds": [], "note": "fixture"}
            with patch("datathon_ml.model._select", return_value=selection):
                primary = fit_final(records, tmp, Path(tmp) / "final", quick_config())
            artifact = joblib.load(primary)
            self.assertEqual(artifact["model"]["candidate"]["head"], "pooled")
            self.assertEqual(artifact["point_model_filename"], "model_point.joblib")
            point_path = primary.parent / artifact["point_model_filename"]
            point_artifact = joblib.load(point_path)
            self.assertEqual(point_artifact["model"]["candidate"]["head"], "point")
            prediction = predict_features(Path(tmp) / f"{records[1]['sha256']}.npz", point_path)
            self.assertTrue(prediction["localization_available"])

    def test_zero_components_trains_unreduced_linear_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = synthetic_dataset(tmp, n=6)
            config = TrainingConfig(outer_folds=2, inner_folds=2, pca_components=0,
                heads=("point",), logistic_cs=(.1,), rbf_cs=(1.,), aggregations=("max",))
            self.assertEqual(_candidates(config), [{"head": "point", "classifier": "logistic",
                                                   "C": .1, "preprocessing": "raw"}])
            model_path = fit_final(records, tmp, Path(tmp) / "final", config)
            artifact = joblib.load(model_path)
            self.assertEqual(artifact["model"]["classifier"].solver, "liblinear")
            self.assertEqual(artifact["selection"]["candidate"]["preprocessing"], "raw")
            prediction = predict_features(Path(tmp) / f"{records[1]['sha256']}.npz", model_path)
            self.assertIn(prediction["verdict"], ("violation", "no_violation"))


if __name__ == "__main__":
    unittest.main()

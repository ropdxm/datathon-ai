"""MIL weak-supervision and nested validation behavior, using synthetic caches."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import torch

from datathon_ml.mil import (MilConfig, _fit_preprocessing, _point_mask, _prepare,
    fit_mil_final, mil_loss, predict_mil_features, run_mil_cross_validation)


def synthetic(directory, count=12):
    rng = np.random.default_rng(521)
    records = []
    for index in range(count):
        label = index % 2
        features = rng.normal(0, .025, (9, 1, 6)).astype(np.float32)
        features[:, :, 1] += 1
        if label:
            features[3:6, :, 0] += 2
        key = f"{index:064x}"
        np.savez_compressed(Path(directory) / f"{key}.npz", features=features,
            timestamps=np.arange(9, dtype=np.float32), config_json=json.dumps({"backbone": "synthetic"}))
        records.append({"sha256": key, "video_id": "untrusted_" + str(1000-index),
            "group_id": key, "label": label, "event_times": [4.] if label else []})
    return records


def quick_config():
    return MilConfig(outer_folds=3, inner_folds=2, pca_components=4, epochs=(30, 40),
                     learning_rate=.03, dropout=.1, threads=1)


class MilTests(unittest.TestCase):
    def test_unmarked_positive_windows_have_no_negative_target(self):
        logits = torch.tensor([3., 2., 1., 0.], requires_grad=True)
        config = MilConfig(top_k=1, smoothness_weight=0)
        losses = mil_loss([logits], [1], [torch.tensor([False, False, True, False])], config)
        losses["total"].backward()
        self.assertEqual(float(losses["normal_instance"].detach()), 0)
        self.assertLess(float(logits.grad[0]), 0)  # positive bag's hardest selected window
        self.assertLess(float(logits.grad[2]), 0)  # positive point anchor
        self.assertEqual(float(logits.grad[1]), 0)
        self.assertEqual(float(logits.grad[3]), 0)

    def test_all_normal_windows_receive_negative_instance_supervision(self):
        logits = torch.tensor([3., 2., 1., -2.], requires_grad=True)
        losses = mil_loss([logits], [0], [torch.zeros(4, dtype=torch.bool)],
                           MilConfig(top_k=1, smoothness_weight=0))
        losses["total"].backward()
        self.assertTrue(torch.all(logits.grad > 0))
        self.assertEqual(float(losses["anchor"].detach()), 0)

    def test_sparse_point_uses_nearest_sample_without_changing_timeline(self):
        times = np.array([0., 4., 8.])
        np.testing.assert_array_equal(_point_mask(times, [2.1], .1), [False, True, False])
        np.testing.assert_array_equal(_point_mask(times, [], 1.), [False, False, False])

    def test_preprocessing_fits_only_training_source_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            records = synthetic(directory, 4)
            entries = _prepare(records, directory)
            train = records[:2]
            config = quick_config()
            before = _fit_preprocessing(train, entries, config)
            for record in records[2:]:
                entries[record["sha256"]]["x"][:] = 1e9
            after = _fit_preprocessing(train, entries, config)
            np.testing.assert_array_equal(before["scaler"].mean_, after["scaler"].mean_)
            np.testing.assert_array_equal(before["pca"].components_, after["pca"].components_)
            self.assertEqual(before["fit_window_count"], 18)
            self.assertEqual(before["fit_sha256"], [r["sha256"] for r in train])

    def test_nested_cv_export_and_annotation_free_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = synthetic(root)
            config = quick_config()
            report = run_mil_cross_validation(records, root, root / "evaluation", config)
            self.assertEqual(report["metrics"]["n"], len(records))
            self.assertGreaterEqual(report["metrics"]["balanced_accuracy"], .8)
            self.assertEqual(len({r["sha256"] for r in report["predictions"]}), len(records))
            for fold in report["folds"]:
                self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
                self.assertFalse(set(fold["preprocessing_fit_sha256"]) & set(fold["test_groups"]))
                for inner in fold["selection"]["selection_folds"]:
                    self.assertFalse(set(inner["train_groups"]) & set(inner["validation_groups"]))
                    self.assertFalse(set(inner["preprocessing_fit_sha256"]) & set(inner["validation_groups"]))
                    self.assertFalse(set(inner["preprocessing_fit_sha256"]) & set(fold["test_groups"]))
            model_path = fit_mil_final(records, root, root / "model", config)
            cache = root / f"{records[1]['sha256']}.npz"
            before = predict_mil_features(cache, model_path)
            with np.load(cache, allow_pickle=False) as source:
                copied = {key: source[key] for key in source.files}
            copied["event_times"] = np.array([123456.])
            copied["label"] = np.array(0)
            copied["video_id"] = np.array("false_metadata")
            renamed = root / "renamed_without_label.npz"
            np.savez_compressed(renamed, **copied)
            after = predict_mil_features(renamed, model_path)
            self.assertEqual(before, after)
            artifact = joblib.load(model_path)
            self.assertEqual(artifact["kind"], "point_guided_mil")
            self.assertFalse(artifact["provenance"]["selection_score_is_test_score"])
            self.assertNotIn("event_times", artifact)
            self.assertTrue((root / "evaluation" / "mil_cv_report.json").is_file())


if __name__ == "__main__":
    unittest.main()

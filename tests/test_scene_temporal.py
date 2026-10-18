"""Ownership, sparse supervision and temporal inference checks on numeric fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import torch

from datathon_ml.scene_temporal import (SceneTemporalConfig, TemporalHead,
    _fit_preprocessing, _point_mask, _prepare, _resample_rows, _visual_rows,
    apply_verified_groups, candidate_grid, fit_scene_temporal_final, original_targets,
    predict_scene_temporal_features, run_scene_temporal_cv, supervision_audit,
    temporal_loss, verified_records)


def synthetic(directory, count=48):
    rng = np.random.default_rng(987)
    records = []
    # Each scene has two views. There are 12 independent groups of each class,
    # large enough for grouped heuristics across supported sklearn versions.
    for index in range(count):
        label = (index // 2) % 2
        key = f"{index + 1:064x}"
        features = np.clip(rng.normal(.12, .01, (11, 1, 6)), 0, 1).astype(np.float32)
        features[:, :, 1] = .8
        if label:
            features[3:6, :, 0] = .9
            features[6:8, :, 2] = .7
        metadata = {"backbone": "synthetic_semantic", "concept_names": [f"c{k}" for k in range(6)],
                    "sample_fps": 2., "views": ["full"]}
        np.savez_compressed(Path(directory) / f"{key}.npz", features=features,
            timestamps=np.arange(11, dtype=np.float32) / 2., config_json=json.dumps(metadata))
        records.append({"sha256": key, "group_id": f"scene_{index // 2}", "group_verified": True,
            "label": label, "event_times": [2.] if label else [], "label_status": "human_verified",
            "event_times_verified": True, "evaluation_ground_truth_eligible": True})
    return records


def quick_config():
    return SceneTemporalConfig(outer_folds=3, inner_folds=2, pca_components=4,
        epochs=(4, 5), threads=1, device="cpu", dropout=.1)


class SceneTemporalTests(unittest.TestCase):
    def test_unverified_custom_targets_rejected_before_cache_access(self):
        key = "a" * 64
        normal = {"sha256": key, "label": 0, "event_times": [], "group_id": key}
        with self.assertRaisesRegex(ValueError, "custom_target_not_human_verified"):
            verified_records([normal], {})
        self.assertEqual(len(verified_records([normal], {key: 0})), 1)
        malicious = {**normal, "label_status": "machine_proposal"}
        with self.assertRaisesRegex(ValueError, "unverified_target"):
            verified_records([malicious], {key: 0})
        verified = {**normal, "label_status": "human_verified", "evaluation_ground_truth_eligible": True,
                    "group_verified": True, "label": 1, "event_times": [1.]}
        self.assertEqual(supervision_audit([verified], {})["excluded"][0]["reason"], "custom_positive_timestamps_not_verified")
        verified["event_times_verified"] = True
        self.assertEqual(len(verified_records([verified], {})), 1)
        self.assertEqual(original_targets({"predictions": [{"sha256": key, "label": 0}]}), {key: 0})

    def test_verified_group_metadata_and_lineage_prevent_cross_camera_leakage(self):
        rows = [{"sha256": str(k) * 64, "group_id": str(k), "label": 0, "event_times": []} for k in (1, 2)]
        payload = {"records": [{"sha256": rows[0]["sha256"], "group_id": "shared_scene",
            "group_verified": False, "verification_source": "guess"}]}
        with self.assertRaisesRegex(ValueError, "Every scene"):
            apply_verified_groups(rows, payload)
        payload["records"][0]["group_verified"] = True
        grouped = apply_verified_groups(rows, payload)
        self.assertEqual(grouped[0]["group_id"], "shared_scene")
        self.assertEqual(grouped[1]["group_id"], "2")
        grouped[1]["parent_sha256"] = grouped[0]["sha256"]
        with self.assertRaisesRegex(ValueError, "lineage crosses"):
            verified_records(grouped, {row["sha256"]: 0 for row in rows})

    def test_semantic_magnitudes_are_preserved_and_gaps_are_not_fabricated(self):
        states = np.full((3, 1, 2), .5, dtype=np.float32)
        np.testing.assert_array_equal(_visual_rows(states, {"concept_names": ["a", "b"]}), states[:, 0])
        normalized = _visual_rows(states, {"backbone": "embedding"})
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1., atol=1e-6)
        config = quick_config()
        x, t, segments = _resample_rows(np.array([[0.], [1.], [3.]], dtype=np.float32),
                                        np.array([0., 1., 10.]), config)
        np.testing.assert_array_equal(t, [0., .5, 1., 10.])
        np.testing.assert_array_equal(x[:, 0], [0., .5, 1., 3.])
        self.assertEqual(segments, [3, 1])
        self.assertFalse(_point_mask(t, [5.], config).any())
        shifted = _resample_rows(np.array([[0.], [1.], [3.]], dtype=np.float32), np.array([100., 101., 110.]), config)
        np.testing.assert_array_equal(shifted[0], x)

    def test_pca_and_scalers_use_training_owners_only(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = synthetic(directory, 8)
            entries = _prepare(rows, directory, quick_config())
            before = _fit_preprocessing(rows[:4], entries, quick_config())
            for row in rows[4:]:
                entries[row["sha256"]]["x"][:] = 1e6
                row["label"] = 1 - row["label"]
                row["event_times"] = [999.]
            after = _fit_preprocessing(rows[:4], entries, quick_config())
            np.testing.assert_array_equal(before["scaler"].mean_, after["scaler"].mean_)
            np.testing.assert_array_equal(before["pca"].components_, after["pca"].components_)
            np.testing.assert_array_equal(before["component_scaler"].scale_, after["component_scaler"].scale_)
            self.assertEqual(before["owner_manifest"], after["owner_manifest"])
            self.assertEqual(before["fit_window_count"], 44)

    def test_sparse_positive_and_dense_normal_loss_have_correct_gradients(self):
        config = SceneTemporalConfig(top_k=1, smoothness_weight=0)
        positive = torch.tensor([3., 2., 1., 0.], requires_grad=True)
        loss = temporal_loss([positive], [1], [torch.tensor([False, False, True, False])], [[4]], config)
        loss["total"].backward()
        self.assertEqual(float(loss["normal"].detach()), 0.)
        self.assertLess(float(positive.grad[0]), 0.)
        self.assertLess(float(positive.grad[2]), 0.)
        self.assertEqual(float(positive.grad[1]), 0.)
        self.assertEqual(float(positive.grad[3]), 0.)
        normal = torch.tensor([3., 2., 1., -2.], requires_grad=True)
        temporal_loss([normal], [0], [torch.zeros(4, dtype=torch.bool)], [[4]], config)["total"].backward()
        self.assertTrue(torch.all(normal.grad > 0))

    def test_convolution_respects_recording_boundaries_padding_and_order(self):
        torch.manual_seed(91)
        head = TemporalHead(4, candidate_grid(quick_config())[0], 0).eval()
        first = torch.randn(1, 4, 7)
        with torch.inference_mode():
            alone = head(first, torch.ones(1, 7, dtype=torch.bool))
            together = torch.zeros(2, 4, 12)
            together[0, :, :7] = first[0]
            together[1] = torch.randn(4, 12) * 100
            valid = torch.zeros(2, 12, dtype=torch.bool)
            valid[0, :7], valid[1] = True, True
            batch = head(together, valid)
            reversed_back = head(first.flip(-1), torch.ones(1, 7, dtype=torch.bool)).flip(-1)
        torch.testing.assert_close(alone[0], batch[0, :7], atol=1e-6, rtol=1e-5)
        self.assertFalse(torch.allclose(alone, reversed_back))

    def test_nested_scene_cv_export_and_annotation_free_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = synthetic(root)
            config = quick_config()
            report = run_scene_temporal_cv(records, root, root / "evaluation", config)
            self.assertEqual(report["grouping"], "verified_scene_groups")
            self.assertEqual(report["metrics"]["n"], len(records))
            self.assertEqual(report["metrics"]["localization"]["annotated_points"], 24)
            self.assertEqual(len({row["sha256"] for row in report["predictions"]}), len(records))
            for fold in report["folds"]:
                self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
                self.assertFalse(set(fold["preprocessing_fit_sha256"]) & set(fold["test_sha256"]))
                for inner in fold["selection"]["selection_folds"]:
                    self.assertFalse(set(inner["train_groups"]) & set(inner["validation_groups"]))
                    self.assertFalse(set(inner["preprocessing_fit_sha256"]) & set(inner["validation_sha256"]))
                    self.assertFalse(set(inner["preprocessing_fit_sha256"]) & set(fold["test_sha256"]))
            model_path = fit_scene_temporal_final(records, root, root / "model", config)
            cache = root / f"{records[1]['sha256']}.npz"
            before = predict_scene_temporal_features(cache, model_path)
            with np.load(cache, allow_pickle=False) as source:
                copied = {key: source[key] for key in source.files}
            copied.update(event_times=np.array([999.]), label=np.array(1), video_id=np.array("misleading"))
            renamed = root / "without_labels.npz"
            np.savez_compressed(renamed, **copied)
            after = predict_scene_temporal_features(renamed, model_path)
            self.assertEqual(before, after)
            artifact = joblib.load(model_path)
            self.assertEqual(artifact["kind"], "scene_point_guided_tcn")
            artifact["provenance"]["source_sha256"]["scene_temporal"] = "0" * 64
            poisoned = root / "wrong_source.joblib"
            joblib.dump(artifact, poisoned)
            with self.assertRaisesRegex(ValueError, "fingerprints differ"):
                predict_scene_temporal_features(cache, poisoned)


if __name__ == "__main__":
    unittest.main()

"""Train-owned preprocessing, frozen groups and feature-only CatBoost export."""
import copy
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from datathon_ml import boosted
from datathon_ml.temporal import CONCEPT_NAMES


def examples(root, n=24):
    clip, semantic = root / "clip", root / "semantic"
    clip.mkdir()
    semantic.mkdir()
    rng = np.random.default_rng(43)
    rows = []
    for i in range(n):
        key = f"{i:064x}"
        group = i // 2  # Two distinct recordings of the same underlying scene.
        label = group % 2
        values = rng.normal(size=(10, 4, 8)).astype(np.float32)
        values[:, :, 0] += 2 * label
        states = rng.uniform(.1, .3, size=(10, 1, len(CONCEPT_NAMES))).astype(np.float32)
        states[:, 0, CONCEPT_NAMES.index("loose_cash_visible")] = .2 + .7 * label
        times = np.arange(10, dtype=np.float32) / 2
        metadata = {"backbone": "synthetic_clip", "views": ["full", "top", "middle", "bottom"],
                    "sample_fps": 2., "image_size": 224, "weights_sha256": "synthetic"}
        smeta = {"backbone": "synthetic_semantic", "concept_names": list(CONCEPT_NAMES),
                 "continuous_window_coverage": False, "preprocessing": "test_semantic18"}
        for directory, array, config in ((clip, values, metadata), (semantic, states, smeta)):
            encoded = json.dumps(config)
            np.savez(directory / f"{key}.npz", features=array, timestamps=times,
                sha256=key, config_json=encoded, config_hash=hashlib.sha256(encoded.encode()).hexdigest())
        rows.append({"sha256": key, "group_id": f"scene{group}", "label": label,
                     "video_id": f"arbitrary{i}", "event_times": [2.] if label else []})
    reference = {"config": {"seed": 42}, "predictions": [{"sha256": r["sha256"],
        "label": r["label"], "fold": (i // 4) % 2} for i, r in enumerate(rows)]}
    return rows, clip, semantic, reference


class BoostedTests(unittest.TestCase):
    def config(self):
        return boosted.BoostedConfig(outer_folds=2, inner_folds=2, iterations=5,
                                     pca_components=4, threads=1)

    def test_fixed_grid_and_single_recording_vectors_ignore_annotations(self):
        candidates = boosted.candidate_grid()
        self.assertEqual(len(candidates), 12)
        self.assertEqual({c["arm"] for c in candidates}, set(boosted.ARMS))
        with tempfile.TemporaryDirectory() as tmp:
            rows, clip, semantic, _ = examples(Path(tmp))
            data, _ = boosted.prepare_data(rows, clip, semantic, self.config())
            altered = copy.deepcopy(rows)
            for r in altered:
                r.update(label=1 - r["label"], event_times=[900.], video_id="leaky_name",
                         duration=999999, path="never-open-this-file", spoken_words="unused")
            changed, _ = boosted.prepare_data(altered, clip, semantic, self.config())
            for key in data:
                for arm in boosted.ARMS:
                    np.testing.assert_array_equal(data[key][arm], changed[key][arm])
            self.assertEqual(data[rows[0]["sha256"]]["semantic_raw"].shape, (162,))
            self.assertEqual(data[rows[0]["sha256"]]["clip_all_raw"].shape, (96,))
            self.assertEqual(data[rows[0]["sha256"]]["clip_full_raw"].shape, (24,))

    def test_projection_does_not_fit_holdout_outlier_or_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, clip, semantic, _ = examples(Path(tmp))
            data, _ = boosted.prepare_data(rows, clip, semantic, self.config())
            train, holdout = rows[:16], rows[16:]
            before = boosted.fit_preprocessing(train, data, "clip_all_pca16", self.config())
            for row in holdout:
                data[row["sha256"]]["clip_all_pca16"][:] = 1e9
            changed = copy.deepcopy(train)
            for row in changed:
                row["label"], row["event_times"] = 1 - row["label"], [999.]
            after = boosted.fit_preprocessing(changed, data, "clip_all_pca16", self.config())
            np.testing.assert_array_equal(before["scaler"].mean_, after["scaler"].mean_)
            np.testing.assert_array_equal(before["pca"].components_, after["pca"].components_)
            self.assertEqual(before["fit_sha256"], [r["sha256"] for r in train])

    def test_frozen_scene_groups_and_all_inner_preprocessing_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, clip, semantic, reference = examples(root)
            with patch.object(boosted, "fit_preprocessing", wraps=boosted.fit_preprocessing) as fit:
                with redirect_stdout(io.StringIO()):
                    report = boosted.run_boosted_comparison(rows, clip, semantic, root / "out",
                        folds_from=reference, config=self.config())
            self.assertEqual(fit.call_count, 2 * (2 + 1) * 4)
            expected = {r["sha256"]: r["fold"] for r in reference["predictions"]}
            self.assertEqual({r["sha256"]: r["fold"] for r in report["predictions"]}, expected)
            self.assertEqual(len(report["predictions"]), len(rows))
            for arm in boosted.ARMS:
                self.assertEqual({r["sha256"]: r["fold"] for r in report["arm_comparisons"][arm]["predictions"]}, expected)
            for fold in report["folds"]:
                outer_test = set(fold["test_sha256"])
                self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
                self.assertFalse(outer_test & set(fold["selection"]["threshold_fit_sha256"]))
                for preprocessing in fold["preprocessing"].values():
                    self.assertFalse(outer_test & set(preprocessing["fit_sha256"]))
                for inner in fold["selection"]["selection_folds"]:
                    forbidden = outer_test | set(inner["validation_sha256"])
                    self.assertFalse(set(inner["train_groups"]) & set(inner["validation_groups"]))
                    for preprocessing in inner["preprocessing"].values():
                        self.assertFalse(forbidden & set(preprocessing["fit_sha256"]))
            self.assertTrue(all(not row["events"] and row["localization_available"] is False
                                for row in report["predictions"]))

    def test_heldout_values_cannot_change_inner_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            # More independent scenes keep the feasible-fold check stable across
            # sklearn releases; a tiny eight-group layout degenerates in 1.6.
            rows, clip, semantic, _ = examples(Path(tmp), n=48)
            data, _ = boosted.prepare_data(rows, clip, semantic, self.config())
            train, holdout = rows[:40], rows[40:]
            before = boosted.select_model(train, data, self.config())
            for row in holdout:
                row["label"] = 1 - row["label"]
                for arm in boosted.ARMS:
                    data[row["sha256"]][arm][:] = 1e6
            after = boosted.select_model(train, data, self.config())
            self.assertEqual(before, after)

    def test_export_predict_all_arms_and_reject_mismatched_source_or_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, clip, semantic, _ = examples(root)
            path = boosted.fit_boosted_final(rows, clip, semantic, root / "model", self.config())
            self.assertTrue(path.is_file())
            for arm in boosted.ARMS:
                model_path = path.parent / f"model_{arm}.joblib"
                result = boosted.predict_boosted_features(model_path,
                    clip_feature_path=clip / f"{rows[0]['sha256']}.npz",
                    semantic_feature_path=semantic / f"{rows[0]['sha256']}.npz")
                self.assertEqual(result["events"], [])
                self.assertFalse(result["localization_available"])
                self.assertGreaterEqual(result["score"], 0.)
                self.assertLessEqual(result["score"], 1.)
            artifact = joblib.load(path)
            artifact["protocol"]["source_hashes"]["boosted.py"] = "wrong"
            joblib.dump(artifact, path)
            with self.assertRaisesRegex(ValueError, "source fingerprint"):
                boosted.predict_boosted_features(path)
            # Same shape is insufficient: a changed sampling protocol must fail.
            cache = clip / f"{rows[0]['sha256']}.npz"
            with np.load(cache) as value:
                x, times, meta = value["features"], value["timestamps"], json.loads(str(value["config_json"].item()))
            meta["sample_fps"] = 4.
            np.savez(cache, features=x, timestamps=times, config_json=json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "extraction protocol"):
                boosted.predict_boosted_features(path.parent / "model_clip_all_raw.joblib", clip_feature_path=cache)

    def test_frozen_reference_rejects_split_views_of_same_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, clip, semantic, reference = examples(Path(tmp))
            reference["predictions"][0]["fold"] = 1
            with self.assertRaisesRegex(ValueError, "splits a current recording group"):
                boosted.run_boosted_comparison(rows, clip, semantic, Path(tmp) / "out",
                    folds_from=reference, config=self.config())


if __name__ == "__main__":
    unittest.main()

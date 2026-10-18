"""Test ownership and one-time locked-test semantics with synthetic features."""
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from datathon_ml.reviewed_models import (ReviewedConfig, _context_features, _entry,
    _training_matrix, _verified_records, candidate_grid, fit_reviewed_selection,
    predict_reviewed_features, score_reviewed_test, validate_split)


def fixture(directory, count=32):
    rng = np.random.default_rng(1307)
    rows = []
    for index in range(count):
        key, label = f"{index + 301:064x}", (index // 2) % 2
        values = np.clip(rng.normal(.1, .01, (9, 1, 4)), 0, 1).astype(np.float32)
        values[:, :, 1] = .8
        if label:
            values[3:6, :, 0] = .9
        metadata = {"backbone": "synthetic_semantic", "concept_names": ["a", "b", "c", "d"],
                    "sample_fps": 2., "views": ["semantic"]}
        np.savez_compressed(Path(directory) / f"{key}.npz", features=values,
            timestamps=np.arange(9, dtype=np.float32) / 2, config_json=json.dumps(metadata))
        rows.append({"sha256": key, "group_id": f"scene_{index // 2}", "group_verified": True,
            "label": label, "event_times": [2.] if label else [], "label_status": "human_verified",
            "event_times_verified": True, "evaluation_ground_truth_eligible": True})
    split = {"train": [row["sha256"] for row in rows[:20]],
             "validation": [row["sha256"] for row in rows[20:26]],
             "test": [row["sha256"] for row in rows[26:]], "protocol": {"type": "synthetic_verified_scenes", "seed": 42}}
    return rows, split


def config():
    return ReviewedConfig(threads=1, include_catboost=False, pooled_pca_components=4, point_pca_components=4)


class ReviewedModelTests(unittest.TestCase):
    def test_split_rejects_shared_scene_sources_and_parent_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, split = fixture(directory)
            validate_split(rows, split)
            bad = {**split, "test": split["test"] + split["train"][:1]}
            with self.assertRaisesRegex(ValueError, "overlap or repeat"):
                validate_split(rows, bad)
            changed = [dict(row) for row in rows]
            changed[-1]["group_id"] = changed[0]["group_id"]
            with self.assertRaisesRegex(ValueError, "group crosses"):
                validate_split(changed, split)
            changed[-1]["group_id"] = "new_group"
            changed[-1]["parent_sha256"] = changed[0]["sha256"]
            with self.assertRaisesRegex(ValueError, "lineage crosses"):
                validate_split(changed, split)

    def test_camera_exclusion_is_distinct_from_human_scene_verification(self):
        protocol = {"type": "camera_exclusive_source_overlap_development", "selected_camera": ".mov",
            "opposite_camera_excluded_globally": True,
            "grouping_basis": "native_duplicate_overlap_prevention_not_human_scene_identity",
            "family_selection_basis": "largest_usable_label_coverage_before_model_scores", "grouping_map_sha256": "f" * 64}
        row = {"sha256": "a" * 64, "group_id": "native_overlap_1", "group_verified": False,
            "label": 1, "event_times": [2.], "event_times_verified": True, "source_family": "mov",
            "label_status": "human_verified", "evaluation_ground_truth_eligible": True}
        accepted = _verified_records([row], {}, {"protocol": protocol})
        self.assertFalse(accepted[0]["group_verified"])
        with self.assertRaisesRegex(ValueError, "opposite/unknown"):
            _verified_records([{**row, "source_family": "mp4"}], {}, {"protocol": protocol})
        with self.assertRaisesRegex(ValueError, "Unverified/pseudo"):
            _verified_records([{**row, "label_status": "machine_proposal"}], {}, {"protocol": protocol})
        with self.assertRaisesRegex(ValueError, "custom_scene_not_verified"):
            _verified_records([row], {}, {"protocol": {"type": "other"}})

    def test_context_uses_order_without_crossing_unsampled_gaps_or_absolute_clocks(self):
        values = np.array([[0.], [1.], [99.]], dtype=np.float32)
        times = np.array([0., .5, 10.])
        actual = _context_features(values, times, config())
        first_segment = _context_features(values[:2], times[:2], config())
        np.testing.assert_array_equal(actual[:2], first_segment)
        np.testing.assert_array_equal(actual, _context_features(values, times + 100., config()))
        self.assertEqual(actual.shape, (3, 8))
        reversed_order = _context_features(values[:2][::-1], times[:2], config())
        self.assertGreater(float(actual[0, -1]), 0.)
        self.assertLess(float(reversed_order[0, -1]), 0.)

    def test_point_supervision_never_assigns_unmarked_positive_windows_negative_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _ = fixture(directory)
            selected = rows[:4]
            entries = {row["sha256"]: _entry(Path(directory) / f"{row['sha256']}.npz", config()) for row in selected}
            training = _training_matrix(selected, entries, "point", config())
            # Two normal owners contribute all 9 samples; the two positive
            # owners contribute only five observations around their anchors.
            self.assertEqual(int((training["y"] == 0).sum()), 18)
            self.assertEqual(int((training["y"] == 1).sum()), 10)
            self.assertAlmostEqual(float(training["weights"][training["y"] == 0].sum()),
                                   float(training["weights"][training["y"] == 1].sum()))
            self.assertEqual(len(training["fit_sha256"]), 4)

    def test_model_selection_does_not_open_test_caches_or_use_their_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, split = fixture(root)
            held_out = []
            for key in split["test"]:
                path = root / f"{key}.npz"
                held_out.append((path, path.read_bytes()))
                path.unlink()
            first = fit_reviewed_selection(rows, root, split, root / "first", config())
            changed = [dict(row) for row in rows]
            for row in changed[26:]:
                row["label"] = 1 - row["label"]
                row["event_times"] = [2.] if row["label"] else []
            second = fit_reviewed_selection(changed, root, split, root / "second", config())
            before, after = joblib.load(first), joblib.load(second)
            self.assertEqual(before["candidate"], after["candidate"])
            self.assertEqual(before["threshold"], after["threshold"])
            np.testing.assert_array_equal(before["preprocessing"]["scaler"].mean_, after["preprocessing"]["scaler"].mean_)
            for row in rows[:3]:
                cache = root / f"{row['sha256']}.npz"
                self.assertEqual(predict_reviewed_features(cache, first), predict_reviewed_features(cache, second))
            selection = json.loads((root / "first" / "selection.json").read_text())
            self.assertEqual(selection["test_feature_files_opened"], 0)
            self.assertEqual(len(selection["leaderboard"]), 24)
            for candidate in selection["leaderboard"]:
                self.assertFalse(set(candidate["preprocessing_fit_sha256"]) & set(split["test"] + split["validation"]))
            for path, raw in held_out:
                path.write_bytes(raw)
            report = score_reviewed_test(rows, root, split, first, root / "first")
            self.assertEqual(report["metrics"]["n"], 6)
            self.assertTrue(report["selected_without_test"])
            with self.assertRaises(FileExistsError):
                score_reviewed_test(rows, root, split, first, root / "first")
            with self.assertRaises(FileExistsError):
                fit_reviewed_selection(rows, root, split, root / "first", config())

    def test_multiple_representations_open_only_global_winners_test_and_predict_without_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_dir, second_dir = root / "first_features", root / "second_features"
            first_dir.mkdir()
            second_dir.mkdir()
            rows, split = fixture(first_dir)
            fixture(second_dir)
            for key in split["test"]:
                (second_dir / f"{key}.npz").unlink()
            feature_sets = {"first": first_dir, "second": second_dir}
            model = fit_reviewed_selection(rows, feature_sets, split, root / "run", config())
            artifact = joblib.load(model)
            self.assertEqual(artifact["candidate"]["representation"], "first")
            self.assertEqual(artifact["feature_sets"], ["first", "second"])
            report = score_reviewed_test(rows, feature_sets, split, model, root / "run")
            self.assertEqual(report["test_open_count"], 1)
            cache = first_dir / f"{rows[0]['sha256']}.npz"
            with np.load(cache, allow_pickle=False) as loaded:
                copied = {key: loaded[key] for key in loaded.files}
            copied.update(label=np.array(1), event_times=np.array([10000.]), video_id=np.array("ignore_me"))
            renamed = first_dir / "unlabeled.npz"
            np.savez_compressed(renamed, **copied)
            self.assertEqual(predict_reviewed_features(cache, model), predict_reviewed_features(renamed, model))
            artifact["source_sha256"]["reviewed_models"] = "0" * 64
            poisoned = root / "changed_source.joblib"
            joblib.dump(artifact, poisoned)
            with self.assertRaisesRegex(ValueError, "matching fitted source"):
                predict_reviewed_features(cache, poisoned)

    def test_declared_candidate_count_is_bounded_and_reproducible(self):
        self.assertEqual(len(candidate_grid(ReviewedConfig())), 26)
        self.assertEqual(len(candidate_grid(config())), 24)
        self.assertEqual(candidate_grid(config()), candidate_grid(config()))


if __name__ == "__main__":
    unittest.main()

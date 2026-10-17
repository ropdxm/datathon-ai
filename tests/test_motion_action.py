"""Recording ownership and feature-only inference for retained action logits."""
import copy
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

import numpy as np

import motion_action_experiment as action
from datathon_ml.boosted import BoostedConfig


def examples(root):
    rng = np.random.default_rng(43)
    rows = []
    metadata = {"backbone": "x3d_s", "views": ["full", "middle"], "sample_fps": 1.,
        "image_size": 182, "weights_sha256": "test_only",
        "preprocessing": json.dumps({"representation": "logits", "normalization": "l2-output"})}
    # Enough paired scenes for older sklearn's stratified group heuristic to
    # retain both classes in every outer-training and nested training split.
    # Production folds are independently frozen by source SHA in the report.
    for i in range(48):
        key = f"{i:064x}"
        group = i // 2
        label = group % 2
        values = rng.normal(size=(4, 2, 400)).astype(np.float32)
        values[:, :, 0] += label * 2
        encoded = json.dumps(metadata)
        np.savez(root / f"{key}.npz", features=values, timestamps=np.arange(4, dtype=np.float32),
            sha256=key, config_json=encoded, config_hash=hashlib.sha256(encoded.encode()).hexdigest())
        rows.append({"sha256": key, "label": label, "group_id": f"scene{group}",
                     "event_times": [2.] if label else [], "video_id": f"name{i}"})
    reference = {"config": {"seed": 42}, "predictions": [{"sha256": r["sha256"],
        "label": r["label"], "fold": (i // 4) % 2} for i, r in enumerate(rows)]}
    return rows, reference


class MotionActionTests(unittest.TestCase):
    def config(self):
        return BoostedConfig(outer_folds=2, inner_folds=2, iterations=3, threads=1)

    def test_descriptors_ignore_annotations_and_have_fixed_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = examples(Path(tmp))
            first, _ = action.prepare(rows, tmp)
            changed = copy.deepcopy(rows)
            for row in changed:
                row.update(label=1-row["label"], event_times=[9999], duration=9999,
                           video_id="irrelevant", path="do_not_open", spoken_words="never_used")
            second, _ = action.prepare(changed, tmp)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
                self.assertEqual(first[key].shape, (2400,))
            self.assertEqual(len(action.candidates()), 3)

    def test_original_folds_and_all_inner_owner_partitions_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, reference = examples(root)
            with redirect_stdout(io.StringIO()):
                report = action.compare(rows, root, root / "out", reference, self.config())
            expected = {r["sha256"]: r["fold"] for r in reference["predictions"]}
            self.assertEqual({r["sha256"]: r["fold"] for r in report["predictions"]}, expected)
            self.assertEqual(len(report["predictions"]), len(rows))
            for fold in report["folds"]:
                forbidden = set(fold["test_sha256"])
                self.assertFalse(forbidden & set(fold["fit_sha256"]))
                self.assertFalse(forbidden & set(fold["selection"]["threshold_fit_sha256"]))
                self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
                for inner in fold["selection"]["selection_folds"]:
                    self.assertFalse((forbidden | set(inner["validation_sha256"])) & set(inner["fit_sha256"]))
                    self.assertFalse(set(inner["train_groups"]) & set(inner["validation_groups"]))
            self.assertFalse(report["metrics"]["localization"]["available"])

    def test_final_export_predicts_features_without_labels_and_checks_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, _ = examples(root)
            path = action.fit_final(rows, root, root / "out", self.config())
            result = action.predict(path, root / f"{rows[0]['sha256']}.npz")
            self.assertIn(result["prediction"], (0, 1))
            self.assertEqual(result["events"], [])
            self.assertFalse(result["localization_available"])
            self.assertTrue(np.isfinite(result["score"]))
            feature = root / f"{rows[0]['sha256']}.npz"
            with np.load(feature, allow_pickle=False) as cached:
                values = {key: cached[key] for key in cached.files}
            config = json.loads(str(values["config_json"]))
            config["weights_sha256"] = "other_weights"
            encoded = json.dumps(config)
            values.update(config_json=np.array(encoded), config_hash=np.array(hashlib.sha256(encoded.encode()).hexdigest()))
            np.savez(feature, **values)
            with self.assertRaisesRegex(ValueError, "protocol differs"):
                action.predict(path, feature)


if __name__ == "__main__":
    unittest.main()

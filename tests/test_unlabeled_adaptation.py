import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from datathon_ml import unlabeled_adaptation as adaptation
from datathon_ml.model import predict_features


def examples(root, n=20, offset=0):
    rng = np.random.default_rng(42 + offset)
    rows = []
    for i in range(n):
        key = f"{i + offset:064x}"
        label = i % 2
        values = rng.normal(size=(13, 2, 4)).astype(np.float32)
        values[5:8, :, 0] += 3 * label
        np.savez(root / f"{key}.npz", features=values, timestamps=np.arange(13, dtype=np.float32),
                 config_json=json.dumps({"backbone": "synthetic_image"}))
        rows.append({"sha256": key, "group_id": key, "path": "unused.mp4",
                     "label": label, "event_times": [6.] if label else []})
    return rows


class UnlabeledAdaptationTests(unittest.TestCase):
    def config(self):
        return adaptation.UnlabeledConfig(outer_folds=2, inner_folds=2, pca_components=4,
            max_projection_windows=6, logistic_cs=(.1, 1.), rbf_cs=(), threads=1)

    def test_unknown_or_adversarial_custom_labels_are_discarded(self):
        rows = [{"sha256": f"{i:064x}", "source_sha256": f"{100 + i:064x}",
                 "group_id": str(i), "label": {"untrusted": "anything"}, "event_times": [-999]}
                for i in range(4)]
        selected, excluded = adaptation.select_unlabeled_records(rows, [])
        self.assertEqual(len(selected), 4)
        self.assertEqual(excluded, [])
        self.assertEqual({r["group_id"] for r in selected}, {"custom_unresolved_paired_session"})
        self.assertTrue(all("label" not in r and "event_times" not in r for r in selected))

    def test_projection_is_uniform_and_label_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, custom = examples(root), examples(root, 4, 100)
            config = self.config()
            clean = adaptation.select_unlabeled_records(custom, original)[0]
            data = adaptation.prepare_data(original, clean, root, root, config)
            before = adaptation.fit_projection(original, clean, data, config)
            changed = copy.deepcopy(original)
            for r in changed:
                r["label"] = 1 - r["label"]
                r["event_times"] = [-900, 900]
            after = adaptation.fit_projection(changed, clean, data, config)
            np.testing.assert_array_equal(before["scaler"].mean_, after["scaler"].mean_)
            np.testing.assert_array_equal(before["pca"].components_, after["pca"].components_)
            self.assertEqual(adaptation.projection_indices(3, 24).tolist(), [0, 1, 2])
            self.assertTrue(all(n <= 6 for n in before["audit"]["projection_window_count_by_sha256"].values()))

    def test_original_heldout_and_custom_labels_cannot_change_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, custom = examples(root), examples(root, 4, 100)
            config = self.config()
            clean = adaptation.select_unlabeled_records(custom, original)[0]
            data = adaptation.prepare_data(original, clean, root, root, config)
            train, test = original[:16], original[16:]
            before = adaptation.select_model(train, clean, data, config)
            for r in test:
                r["label"] = 1 - r["label"]
                data[r["sha256"]]["point"][:] = 9999
            for r in custom:
                r["label"], r["event_times"] = "ignored", [-100]
            clean2 = adaptation.select_unlabeled_records(custom, original)[0]
            after = adaptation.select_model(train, clean2, data, config)
            self.assertEqual(before, after)

    def test_frozen_test_shas_and_projection_reuse_across_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, custom = examples(root), examples(root, 4, 100)
            reference = {"config": {"seed": 42}, "predictions": [
                {"sha256": r["sha256"], "label": r["label"], "fold": (i // 2) % 2}
                for i, r in enumerate(original)]}
            with patch.object(adaptation, "fit_projection", wraps=adaptation.fit_projection) as fitted:
                with redirect_stdout(io.StringIO()):
                    report = adaptation.run_unlabeled_comparison(original, custom, root, root, root / "out",
                        folds_from=reference, config=self.config())
            self.assertEqual(fitted.call_count, 2 * 2 * (2 + 1))  # arms * outer * (inner + outer fit)
            expected = {r["sha256"]: r["fold"] for r in reference["predictions"]}
            custom_sha = {r["sha256"] for r in custom}
            for policy in report["policies"].values():
                self.assertEqual({r["sha256"]: r["fold"] for r in policy["predictions"]}, expected)
                for fold in policy["folds"]:
                    heldout = set(fold["test_sha256"])
                    self.assertFalse(heldout & set(fold["projection_fit_sha256"]))
                    self.assertFalse(custom_sha & set(fold["supervised_training_sha256"]))
                    for inner in fold["selection"]["selection_folds"]:
                        self.assertFalse((heldout | set(inner["validation_sha256"])) & set(inner["projection_fit_sha256"]))
                        self.assertFalse(custom_sha & set(inner["supervised_training_sha256"]))
            self.assertEqual(report["custom_labels_used"], 0)
            self.assertEqual(report["unlabeled_custom_groups"], 1)

    def test_parent_overlap_excluded_and_export_uses_existing_predictor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, custom = examples(root), examples(root, 4, 100)
            custom[0]["parent_sha256"] = original[0]["sha256"]
            selected, excluded = adaptation.select_unlabeled_records(custom, original)
            self.assertEqual(len(selected), 3)
            self.assertEqual(excluded[0]["reason"], "original_source_or_parent_overlap")
            path = adaptation.fit_unlabeled_final(original, custom, root, root, root / "model", config=self.config())
            result = predict_features(root / f"{original[0]['sha256']}.npz", path)
            self.assertEqual(len(result["window_scores"]), 13)
            artifact = joblib.load(path)
            self.assertEqual(artifact["provenance"]["custom_labels_used"], 0)
            self.assertFalse({r["sha256"] for r in custom} & set(artifact["provenance"]["supervised_training_sha256"]))


if __name__ == "__main__":
    unittest.main()

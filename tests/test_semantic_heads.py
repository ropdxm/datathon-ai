"""Fixed-family nested CV, training boundaries and portable semantic artifacts."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

import joblib
import numpy as np

from datathon_ml.semantic_heads import (FAMILIES, SemanticHeadConfig, candidate_grid,
    fit_preprocessing, predict_semantic_head_features, run_semantic_head_cv)
from datathon_ml.temporal import CONCEPT_NAMES, TemporalConfig, _prepare


def fixture(root, n=12):
    rng = np.random.default_rng(943)
    records = []
    metadata = {"backbone": "synthetic_semantic18", "concept_names": list(CONCEPT_NAMES),
                "continuous_window_coverage": False, "preprocessing": "fixed-state-schema-v1"}
    for i in range(n):
        x = rng.uniform(0, .01, size=(12, 1, len(CONCEPT_NAMES))).astype(np.float32)
        for name in ("cashier_present", "scene_clear", "hands_visible", "cash_area_visible"):
            x[:, 0, CONCEPT_NAMES.index(name)] = .95
        label = i % 2
        if label:
            x[5:10, 0, CONCEPT_NAMES.index("cashier_present")] = .05
            x[4:11, 0, CONCEPT_NAMES.index("loose_cash_visible")] = .95
            x[4:11, 0, CONCEPT_NAMES.index("transaction_active")] = .95
        key = f"{i:064x}"
        np.savez(root / f"{key}.npz", features=x, timestamps=np.arange(12, dtype=np.float32),
                 config_json=json.dumps(metadata))
        # Paired distinct recordings belong to one related filming group.
        records.append({"sha256": key, "video_id": f"untrusted_{1000-i}", "group_id": str(i // 2),
                        "label": label, "event_times": [6.] if label else []})
    report = {"config": {"seed": 42}, "evaluation": "historical_frozen_fixture", "predictions": [
        {"sha256": row["sha256"], "label": row["label"], "fold": (i // 2) % 3} for i, row in enumerate(records)]}
    manifest, frozen = root / "manifest.json", root / "old_cv.json"
    manifest.write_text(json.dumps(list(reversed(records))), encoding="utf-8")
    frozen.write_text(json.dumps(report), encoding="utf-8")
    return records, manifest, frozen


class SemanticHeadTests(unittest.TestCase):
    def test_fixed_candidate_family_and_frozen_outer_map_required(self):
        grid = candidate_grid()
        self.assertEqual(len(grid), 7)
        self.assertEqual([row["C"] for row in grid if row["family"] == "linear"], [.01, .1, 1.])
        self.assertEqual([row["C"] for row in grid if row["family"] == "rbf"], [.1, 1., 10.])
        self.assertEqual(grid[-1], {"family": "extra_trees", "n_estimators": 128, "max_depth": 3, "min_samples_leaf": 2})
        with self.assertRaisesRegex(ValueError, "historical outer"):
            run_semantic_head_cv([], ".", ".")

    def test_scaler_ignores_heldout_feature_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, _, _ = fixture(root)
            data = _prepare(records, root, TemporalConfig(outer_folds=3, inner_folds=2))
            train = records[:4]
            before = fit_preprocessing(train, data)
            for row in records[4:]:
                data[row["sha256"]]["derived"]["features"][:] = 1e9
            after = fit_preprocessing(train, data)
            np.testing.assert_array_equal(before["scaler"].mean_, after["scaler"].mean_)
            self.assertEqual(before["fit_sha256"], [row["sha256"] for row in train])

    def test_cli_pipeline_frozen_cv_export_no_anchors_and_cache_protocol(self):
        from semantic_head_experiment import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, manifest, frozen = fixture(root)
            with redirect_stdout(io.StringIO()):
                result = main(["pipeline", "--manifest", str(manifest), "--feature-dir", str(root),
                    "--output-dir", str(root / "run"), "--folds-from", str(frozen),
                    "--outer-folds", "3", "--inner-folds", "2"])
            report, path = result["report"], Path(result["model"])
            expected = {row["sha256"]: row["fold"] for row in json.loads(frozen.read_text())["predictions"]}
            self.assertEqual({row["sha256"]: row["fold"] for row in report["predictions"]}, expected)
            self.assertEqual(report["fold_source"]["file_sha256"], hashlib.sha256(frozen.read_bytes()).hexdigest())
            self.assertEqual(set(report["family_comparisons"]), set(FAMILIES))
            self.assertGreaterEqual(report["metrics"]["balanced_accuracy"], .8)
            for fold in report["folds"]:
                heldout = set(fold["test_sha256"])
                self.assertFalse(set(fold["preprocessing_fit_sha256"]) & heldout)
                self.assertFalse(set(fold["selection"]["calibration_fit_sha256"]) & heldout)
                for inner in fold["selection"]["selection_folds"]:
                    self.assertFalse(set(inner["train_groups"]) & set(inner["validation_groups"]))
                    self.assertFalse(set(inner["preprocessing_fit_sha256"]) & heldout)
                for family, chosen in fold["selection"]["families"].items():
                    self.assertEqual(chosen["candidate"]["family"], family)
            for family, comparison in report["family_comparisons"].items():
                self.assertEqual(comparison["metrics"]["n"], len(records))
                self.assertEqual({row["sha256"]: row["fold"] for row in comparison["predictions"]}, expected)
                self.assertTrue(all(row["selected_model"]["family"] == family for row in comparison["predictions"]))
            artifact = joblib.load(path)
            self.assertEqual(artifact["kind"], "semantic_head")
            self.assertFalse(artifact["provenance"]["selection_score_is_test_score"])
            self.assertEqual(set(artifact["family_model_files"]), set(FAMILIES))
            cache = root / f"{records[1]['sha256']}.npz"
            before = predict_semantic_head_features(cache, path)
            self.assertTrue(any("exposure length" in text for text in artifact["limitations"]))
            self.assertTrue(any("exposure length" in text for text in report["limitations"]))
            for field in ("code_sha256", "temporal_code_sha256"):
                tampered = {**artifact, "provenance": {**artifact["provenance"], field: "0" * 64}}
                tampered_path = root / f"tampered_{field}.joblib"
                joblib.dump(tampered, tampered_path)
                with self.assertRaisesRegex(ValueError, "code fingerprint"):
                    predict_semantic_head_features(cache, tampered_path)
            for file in artifact["family_model_files"].values():
                prediction = predict_semantic_head_features(cache, path.parent / file)
                self.assertTrue(0 <= prediction["score"] <= 1)
            with np.load(cache) as stored:
                payload = {key: stored[key] for key in stored.files}
            payload.update(label=np.array(0), event_times=np.array([99999.]), video_id=np.array("wrong-name"))
            renamed = root / "renamed_without_labels.npz"
            np.savez(renamed, **payload)
            self.assertEqual(before, predict_semantic_head_features(renamed, path))
            metadata = json.loads(str(payload["config_json"].item()))
            for change in ({"continuous_window_coverage": True}, {"coverage_certification": "new-claim"},
                           {"preprocessing": "different-protocol"}):
                altered = {**payload, "config_json": json.dumps({**metadata, **change})}
                np.savez(renamed, **altered)
                with self.assertRaises(ValueError):
                    predict_semantic_head_features(renamed, path)
            with redirect_stdout(io.StringIO()):
                cli_prediction = main(["predict", "--feature", str(cache), "--model", str(path),
                                       "--output-dir", str(root / "prediction")])
            self.assertEqual(before, cli_prediction)


if __name__ == "__main__":
    unittest.main()

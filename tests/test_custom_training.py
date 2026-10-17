"""Auxiliary views must not become validation targets or cross source groups."""
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

import numpy as np

from datathon_ml import custom_training as custom
from datathon_ml.temporal import CONCEPT_NAMES, predict_temporal_features
from datathon_ml.model import predict_features


def examples(root, n=20, offset=0, status=None):
    records = []
    for i in range(n):
        label = i % 2
        key = f"{i + offset:064x}"
        values = np.full((12, 1, len(CONCEPT_NAMES)), .1, np.float32)
        for name in ("cashier_present", "hands_visible", "cash_area_visible", "scene_clear"):
            values[:, 0, CONCEPT_NAMES.index(name)] = .95
        if label:
            values[4:10, 0, CONCEPT_NAMES.index("cashier_present")] = .05
            values[3:11, 0, CONCEPT_NAMES.index("loose_cash_visible")] = .95
        metadata = {"backbone": "synthetic_semantic", "concept_names": list(CONCEPT_NAMES),
                    "continuous_window_coverage": False}
        np.savez(root / f"{key}.npz", features=values, timestamps=np.arange(12, dtype=np.float32),
                 config_json=json.dumps(metadata))
        row = {"sha256": key, "group_id": key, "label": label, "event_times": [6.] if label else []}
        if status:
            row["label_status"] = status
        records.append(row)
    return records


class CustomTrainingTests(unittest.TestCase):
    def config(self):
        return custom.CustomTrainingConfig(outer_folds=2, inner_folds=2, semantic_cs=(.1,))

    def test_machine_and_unverified_labels_require_separate_explicit_opt_in(self):
        records = [{"sha256": f"{i:064x}", "group_id": str(i), "label": i % 2,
                    "event_times": [1.] if i % 2 else [], "label_status": status}
                   for i, status in enumerate(("machine_proposal", "supplied_unverified", "human_verified", "unlabeled"))]
        accepted, excluded = custom.select_auxiliary_records(records)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(excluded), 3)
        self.assertEqual(len(custom.select_auxiliary_records(records, allow_machine=True)[0]), 2)
        self.assertEqual(len(custom.select_auxiliary_records(records, allow_unverified=True)[0]), 2)

    def test_unresolved_views_one_session_and_parent_overlap_excluded(self):
        original = [{"sha256": "a" * 64, "group_id": "old"}]
        records = [{"sha256": str(i) * 64, "group_id": str(i), "label": 0, "event_times": [],
                    "label_status": "human_verified"} for i in (1, 2, 3)]
        records[2]["source_sha256"] = "a" * 64
        accepted, excluded = custom.select_auxiliary_records(records, original)
        self.assertEqual({r["group_id"] for r in accepted}, {"custom_unresolved_paired_session"})
        self.assertEqual(excluded[0]["reason"], "original_content_or_parent_overlap")
        self.assertEqual(custom.auxiliary_for_training(accepted, [{"sha256": "b" * 64,
            "group_id": "custom_unresolved_paired_session"}]), [])

    def test_lineage_blocking_includes_reencoded_parent(self):
        aux = [{"sha256": "a" * 64, "group_id": "new", "parent_sha256": "c" * 64}]
        self.assertEqual(custom.auxiliary_for_training(aux, [{"sha256": "c" * 64, "group_id": "old"}]), [])

    def test_duplicate_parent_conflicting_labels_fail(self):
        records = [{"sha256": str(i) * 64, "source_sha256": "a" * 64,
                    "label": i % 2, "event_times": [1.] if i % 2 else [],
                    "label_status": "human_verified"} for i in (1, 2)]
        with self.assertRaisesRegex(ValueError, "Conflicting auxiliary"):
            custom.select_auxiliary_records(records)

    def test_supervision_fingerprint_preserves_labels_lineage_and_teacher_protocol(self):
        records = [{"sha256": str(i) * 64, "source_sha256": chr(96 + i) * 64,
                    "prepared_sha256": str(i) * 64, "label": i % 2,
                    "event_times": [1.] if i % 2 else [], "label_status": "machine_proposal",
                    "proposal_protocol_sha256": "f" * 64,
                    "training_supervision": "machine_pseudo_label",
                    "evaluation_ground_truth_eligible": False} for i in (1, 2)]
        accepted, _ = custom.select_auxiliary_records(records, allow_machine=True)
        before = custom.auxiliary_supervision_manifest(accepted)
        self.assertEqual(before["sha256"], custom.auxiliary_supervision_manifest(accepted[::-1])["sha256"])
        self.assertEqual(before["records"][0]["source_sha256"], "a" * 64)
        self.assertEqual(before["records"][0]["proposal_protocol_sha256"], "f" * 64)
        self.assertFalse(before["records"][0]["evaluation_ground_truth_eligible"])
        accepted[0]["proposal_protocol_sha256"] = "e" * 64
        self.assertNotEqual(before["sha256"], custom.auxiliary_supervision_manifest(accepted)["sha256"])

    def test_auxiliary_train_only_and_frozen_original_test_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = examples(root)
            aux = examples(root, 4, 100, "machine_proposal")
            reference = {"config": {"seed": 42}, "predictions": [
                {"sha256": r["sha256"], "label": r["label"], "fold": (i // 2) % 2}
                for i, r in enumerate(original)]}
            with redirect_stdout(io.StringIO()):
                report = custom.run_custom_comparison(original, aux, root, root, root / "out",
                    folds_from=reference, config=self.config(), allow_machine_aux=True)
            aux_sha = {r["sha256"] for r in aux}
            expected = {r["sha256"]: r["fold"] for r in reference["predictions"]}
            self.assertEqual(report["status"], "weak_label_sensitivity_not_verified_expansion")
            self.assertEqual(len(report["auxiliary"]["supervision_manifest"]["records"]), 4)
            for policy in report["policies"].values():
                self.assertEqual({r["sha256"]: r["fold"] for r in policy["predictions"]}, expected)
                for fold in policy["folds"]:
                    test = set(fold["test_sha256"])
                    self.assertFalse(test & set(fold["preprocessing_fit_sha256"]))
                    selection = fold["selection"]
                    self.assertFalse(aux_sha & set(selection["calibration_fit_sha256"]))
                    self.assertFalse(aux_sha & set(selection["threshold_fit_sha256"]))
                    for inner in selection["selection_folds"]:
                        self.assertFalse(aux_sha & set(inner["validation_sha256"]))
                        self.assertFalse(test & set(inner["preprocessing_fit_sha256"]))
                        self.assertFalse(set(inner["validation_sha256"]) & set(inner["preprocessing_fit_sha256"]))

    def test_heldout_labels_and_features_do_not_affect_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = examples(root)
            aux = examples(root, 2, 100, "human_verified")
            config = self.config()
            aux = custom.select_auxiliary_records(aux)[0]
            data = custom.prepare_data(original, aux, root, root, config)
            train, heldout = original[:16], original[16:]
            before = custom.select_on_originals(train, aux, data, config)
            for r in heldout:
                r["label"] = 1 - r["label"]
                data[r["sha256"]]["derived"]["features"][:] = 9999
            after = custom.select_on_originals(train, aux, data, config)
            self.assertEqual(custom._public(before), custom._public(after))

    def test_final_semantic_artifact_works_with_unchanged_historical_predictor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = examples(root)
            aux = examples(root, 2, 100, "human_verified")
            source_sha = hashlib.sha256(Path(custom.temporal.__file__).read_bytes()).hexdigest()
            path = custom.fit_custom_final(original, aux, root, root, root / "model", config=self.config())
            result = predict_temporal_features(root / f"{original[0]['sha256']}.npz", path)
            self.assertIn(result["verdict"], ("violation", "no_violation"))
            self.assertEqual(source_sha, hashlib.sha256(Path(custom.temporal.__file__).read_bytes()).hexdigest())

    def test_point_weighted_training_and_existing_predictor_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = examples(root)
            aux = examples(root, 4, 100, "machine_proposal")
            rng = np.random.default_rng(42)
            for r in original + aux:
                values = rng.normal(size=(12, 4, 8)).astype(np.float32)
                values[4:8, :, 0] += 3 * r["label"]
                np.savez(root / f"{r['sha256']}.npz", features=values,
                    timestamps=np.arange(12, dtype=np.float32),
                    config_json=json.dumps({"backbone": "synthetic_image"}))
            config = custom.CustomTrainingConfig(policy="point", outer_folds=2,
                inner_folds=2, point_logistic_cs=(.1,), point_rbf_cs=())
            path = custom.fit_custom_final(original, aux, root, root, root / "model",
                config=config, allow_machine_aux=True)
            result = predict_features(root / f"{original[0]['sha256']}.npz", path)
            self.assertTrue(result["localization_available"])
            self.assertEqual(len(result["window_scores"]), 12)

    def test_frozen_reference_rejects_missing_original_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = examples(root)
            with self.assertRaisesRegex(ValueError, "coverage"):
                custom.run_custom_comparison(records, [], root, root, root / "out",
                    folds_from={"predictions": [{"sha256": records[0]["sha256"], "fold": 0}]}, config=self.config())


if __name__ == "__main__":
    unittest.main()

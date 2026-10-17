"""Synthetic only: ownership, held-out poisoning and note-free inference."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from datathon_ml import privileged_notes as notes
from datathon_ml.clip_features import BACKBONE, EXPECTED_WEIGHT_SHA256
from tools.prepare_privileged_notes import (LazyNotePreparer, parse_phrases,
                                           read_owner_notes, remove_nonvisual_context)


def examples(root, n=20):
    features, cache = root / "features", root / "vectors"
    features.mkdir(); cache.mkdir()
    records = []
    rng = np.random.default_rng(42)
    for i in range(n):
        owner, label = f"{i:064x}", i % 2
        vector = np.eye(512, dtype=np.float32)[label]
        values = rng.normal(0, .001, (12, 4, 512)).astype(np.float32) + vector
        np.savez(features / f"{owner}.npz", features=values, timestamps=np.arange(12, dtype=np.float32),
            config_json=json.dumps({"backbone": BACKBONE, "weights_sha256": EXPECTED_WEIGHT_SHA256,
                                   "views": ["full", "top", "middle", "bottom"]}))
        metadata = {"owner_sha256": owner, "input_notes_sha256": "a" * 64,
                    "protocol_sha256": "b" * 64, "usable": True,
                    "supervision": "unverified_friend_privileged_notes"}
        np.savez(cache / f"{owner}.npz", vector=vector, metadata_json=json.dumps(metadata))
        records.append({"sha256": owner, "group_id": owner, "label": label,
                        "event_times": [6.] if label else [], "aliases": [f"X-{i}"]})
    return records, features, cache


class PrivilegedNotesTests(unittest.TestCase):
    def config(self):
        return notes.PrivilegedConfig(outer_folds=2, inner_folds=2)

    def test_owner_rejected_before_any_file_is_opened(self):
        owner = "a" * 64
        with patch("numpy.load") as opened:
            with self.assertRaises(PermissionError):
                notes.load_note_vector("missing", owner, allowed_owners=[])
            opened.assert_not_called()
        with patch.object(Path, "read_text") as read:
            with self.assertRaises(PermissionError):
                read_owner_notes("missing", owner, allowed_owners=[])
            read.assert_not_called()

    def test_heldout_vector_poisoning_does_not_change_training_bank_or_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, features, cache = examples(root)
            data = notes.prepare_features(records, features)
            train, heldout = records[:16], records[16:]
            before = notes.select_threshold(train, data, cache, self.config())
            bank = notes.fit_bank(train, cache)
            for row in heldout:
                (cache / f"{row['sha256']}.npz").write_bytes(b"poisoned heldout notes")
            after = notes.select_threshold(train, data, cache, self.config())
            bank_after = notes.fit_bank(train, cache)
            self.assertEqual(before, after)
            np.testing.assert_array_equal(bank["positive"], bank_after["positive"])
            self.assertEqual(bank["fit_note_manifest_sha256"], bank_after["fit_note_manifest_sha256"])

    def test_inner_validation_notes_never_reach_loader_or_prototypes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, features, cache = examples(root)
            data = notes.prepare_features(records, features)
            train_idx, val_idx = notes.base.grouped_splits(records, 2, 43)[0]
            train, validation = [records[i] for i in train_idx], [records[i] for i in val_idx]
            expected = frozenset(r["sha256"] for r in train)
            calls = []
            def loader(owner, *, allowed_owners):
                self.assertEqual(allowed_owners, expected)
                self.assertIn(owner, expected)
                calls.append(owner)
                return notes.load_note_vector(cache, owner, allowed_owners=allowed_owners)
            bank = notes.fit_bank(train, cache, loader=loader)
            before = [notes._video_score(bank, data[r["sha256"]], self.config()) for r in validation]
            for row in validation:
                (cache / f"{row['sha256']}.npz").write_bytes(b"poisoned validation notes")
            after_bank = notes.fit_bank(train, cache, loader=loader)
            after = [notes._video_score(after_bank, data[r["sha256"]], self.config()) for r in validation]
            self.assertEqual(before, after)
            self.assertEqual(set(calls), set(expected))

    def test_alias_duplicate_has_one_source_vote_and_missing_class_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            records, _, cache = examples(Path(tmp))
            duplicate = {**records[0], "aliases": ["X-47"]}
            bank = notes.fit_bank(records + [duplicate], cache)
            self.assertEqual(len(bank["fit_owner_allowlist"]), len(records))
            self.assertEqual(len(bank["negative"]), len(records) // 2)
            with self.assertRaisesRegex(ValueError, "Both training classes"):
                notes.fit_bank([r for r in records if r["label"] == 0], cache)

    def test_final_model_predicts_after_notes_and_vector_files_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, features, cache = examples(root)
            model = notes.fit_privileged_final(records, features, cache, root / "model", config=self.config())
            feature = features / f"{records[0]['sha256']}.npz"
            before = notes.predict_privileged_features(feature, model)
            for file in cache.iterdir():
                file.unlink()
            cache.rmdir()
            after = notes.predict_privileged_features(feature, model)
            self.assertEqual(before, after)
            self.assertFalse(after["inference_notes_used"])

    def test_frozen_folds_and_every_inner_bank_use_training_owners_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, features, cache = examples(root)
            reference = {"config": {"seed": 42}, "predictions": [{"sha256": r["sha256"],
                "label": r["label"], "fold": (i // 2) % 2} for i, r in enumerate(records)]}
            with redirect_stdout(io.StringIO()):
                report = notes.run_privileged_cv(records, features, cache, root / "out",
                    folds_from=reference, config=self.config())
            self.assertEqual({r["sha256"]: r["fold"] for r in report["predictions"]},
                             {r["sha256"]: r["fold"] for r in reference["predictions"]})
            for fold in report["folds"]:
                test = set(fold["test_sha256"])
                self.assertFalse(test & set(fold["prototype_fit_sha256"]))
                for inner in fold["selection"]["selection_folds"]:
                    self.assertFalse((test | set(inner["validation_sha256"])) & set(inner["prototype_fit_sha256"]))

    def test_lazy_preparation_strips_labels_times_and_ignores_payload_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); private = root / "by_source"; private.mkdir()
            owner, other = "1" * 64, "2" * 64
            payload = {"source_sha256": owner, "provided_label": 1, "event_times": [999],
                "note_fields": [{"text": "00:05 X-47.MOV worker counts cash. worker must leave."}]}
            (private / f"{owner}.json").write_text(json.dumps(payload), encoding="utf-8")
            with patch("tools.prepare_privileged_notes.NativeSanitizer") as model:
                with self.assertRaises(PermissionError):
                    LazyNotePreparer(root, root / "cache", [{"sha256": owner}]).load(other, allowed_owners={owner})
                model.assert_not_called()
            calls = []
            def sanitizer(texts):
                calls.append(texts)
                return ["a worker counts cash"]
            vector = np.eye(512, dtype=np.float32)[0]
            preparer = LazyNotePreparer(root, root / "cache", [{"sha256": owner}],
                sanitizer=sanitizer, encoder=lambda _: vector, protocol={"synthetic": True})
            with redirect_stdout(io.StringIO()) as stdout:
                prepared = preparer.load(owner, allowed_owners={owner})
                preparer.load(owner, allowed_owners={owner})
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], ["worker counts cash."])
            self.assertTrue(prepared["provenance"]["usable"])

    def test_parser_rejects_speech_identity_class_words_and_hidden_intent(self):
        for invalid in ('["a violation happens"]', '["john steals cash"]', '["worker says hello"]',
                        '["worker intends to leave"]', '["a worker counts 20 bills"]', 'not json'):
            self.assertEqual(parse_phrases(invalid), [])
        self.assertEqual(parse_phrases('["a worker counts cash"]'), ["a worker counts cash"])
        self.assertEqual(remove_nonvisual_context("worker must count cash.\nworker says hello."), "")

    def test_complete_longer_array_keeps_first_two_individually_safe_phrases(self):
        answer = json.dumps(["a violation happens", "a worker counts cash", "a worker intends to leave",
                             "a worker opens a drawer", "a worker handles documents"])
        self.assertEqual(parse_phrases(answer), ["a worker counts cash", "a worker opens a drawer"])
        self.assertEqual(parse_phrases(json.dumps(["a worker counts cash"] * 17)), [])
        self.assertEqual(parse_phrases('["a worker counts cash", "a worker opens a drawer"'), [])
        self.assertEqual(parse_phrases('{"phrases": ["a worker counts cash"]}'), [])

    def test_provenance_never_copies_private_text_from_cache_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            records, _, cache = examples(Path(tmp))
            owner = records[0]["sha256"]
            path = cache / f"{owner}.npz"
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata_json"].item()))
                vector = saved["vector"].copy()
            metadata["private_text"] = "synthetic descriptor must remain private"
            np.savez(path, vector=vector, metadata_json=json.dumps(metadata))
            bank = notes.fit_bank(records, cache)
            self.assertNotIn("private_text", json.dumps(bank["fit_note_provenance"]))


if __name__ == "__main__":
    unittest.main()

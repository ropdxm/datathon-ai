"""Direct local A/B ownership, uniform query sampling and no-text inference."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from datathon_ml.direct_vlm import (DirectVLMConfig, LocalDirectVLM, demo_owners,
    fit_direct, make_bundle, messages_and_videos, sample_indices, score_direct, video_metadata)


def fixture(root):
    rows = []
    for index in range(12):
        key = f"{index + 1:064x}"
        rows.append({"sha256": key, "group_id": key, "domain": "original", "label": index % 2,
                     "event_times": [2.] if index % 2 else []})
    split = {"train": [r["sha256"] for r in rows[:4]],
             "validation": [r["sha256"] for r in rows[4:8]],
             "test": [r["sha256"] for r in rows[8:]], "protocol": {}}
    reference = {"predictions": [{"sha256": r["sha256"], "label": r["label"]} for r in rows]}
    return rows, split, reference


def pack(root, key):
    directory = root / key
    directory.mkdir()
    images, hashes = [], {}
    for index in range(5):
        names = []
        for view in ("full", "middle"):
            name = f"{index:06d}_{view}.jpg"
            Image.new("RGB", (224, 224), color=(20, 30, 40)).save(directory / name)
            hashes[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            names.append(name)
        images.append(names)
    metadata = {"source_sha256": key, "timestamps": [0., 1., 2., 3., 4.],
        "images": images, "views": ["full", "middle"], "preprocessing": "raw_rgb_uint8_224_letterbox"}
    (directory / "metadata.json").write_text(json.dumps(metadata))
    hashes["metadata.json"] = hashlib.sha256((directory / "metadata.json").read_bytes()).hexdigest()
    (directory / "_integrity.json").write_text(json.dumps({"completed": True, "members_sha256": hashes}))


class FakeEngine:
    provenance = {"synthetic_engine_for_implementation_test": True}

    def __init__(self):
        self.calls = []

    def score(self, query, demonstrations):
        assert "label" not in query and "event_times" not in query
        self.calls.append((query["source_sha256"], [row["sha256"] for row in demonstrations]))
        return .3 if int(query["source_sha256"], 16) % 2 else .7


class DirectVLMTests(unittest.TestCase):
    def test_query_sampling_is_uniform_and_training_point_is_explicit(self):
        times = np.arange(100, dtype=float)
        uniform = sample_indices(times, 16)
        self.assertEqual(uniform[0], 0)
        self.assertEqual(uniform[-1], 99)
        anchored = sample_indices(times, 4, positive_training_point=30.)
        self.assertTrue(all(abs(times[i] - 30.) <= 2 for i in anchored))
        self.assertEqual(len(sample_indices([0., .5, .9], 16)) % 2, 0)

    def test_train_demo_selection_cannot_depend_on_heldout_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, split, _ = fixture(Path(directory))
            before = [row["sha256"] for row in demo_owners(rows[:4])]
            for row in rows[4:]:
                row.update(label=1 - row["label"], event_times=[999.])
            after = [row["sha256"] for row in demo_owners(rows[:4][::-1])]
            self.assertEqual(before, after)
            self.assertTrue(set(before).issubset(split["train"]))

    def test_video_timestamp_clock_preserves_measured_seconds(self):
        bundle = {"timestamps": [0., 1.25, 5.003, 8.]}
        metadata = video_metadata(bundle)
        np.testing.assert_allclose(np.asarray(metadata["frames_indices"]) / metadata["fps"], bundle["timestamps"])
        self.assertEqual(metadata["width"], 224)

    def test_native_prompt_contains_no_source_identity_or_private_description(self):
        bundle = {"source_sha256": "a" * 64, "frames": np.zeros((4, 224, 224, 3), np.uint8),
                  "timestamps": [0., 1., 2., 3.], "description": "DO_NOT_INCLUDE"}
        messages, videos, metadata = messages_and_videos(bundle,
            [{"sha256": "b" * 64, "label": 1, "bundle": bundle}])
        text = json.dumps(messages)
        self.assertNotIn("a" * 64, text)
        self.assertNotIn("b" * 64, text)
        self.assertNotIn("DO_NOT_INCLUDE", text)
        self.assertEqual(len(videos), 2)
        self.assertEqual(len(metadata), 2)
        self.assertEqual(messages[2]["content"], "A")

    def test_last_token_forward_scores_a_b_without_generation(self):
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return "fixed template"

            def __call__(self, **kwargs):
                self.arguments = kwargs
                return {"input_ids": torch.ones((1, 4), dtype=torch.long),
                        "pixel_values_videos": torch.zeros((1, 2)), "token_type_ids": torch.zeros((1, 4))}

        class Model:
            def __call__(self, **kwargs):
                self.arguments = kwargs
                logits = torch.zeros((1, 1, 8))
                logits[0, 0, 3] = np.log(9)
                return SimpleNamespace(logits=logits)

            def generate(self, **kwargs):
                raise AssertionError("Free text must never be generated")

        engine = LocalDirectVLM.__new__(LocalDirectVLM)
        engine.device, engine.answer_ids = torch.device("cpu"), [3, 5]
        engine.processor, engine.model = Processor(), Model()
        bundle = {"frames": np.zeros((4, 224, 224, 3), np.uint8), "timestamps": [0., 1., 2., 3.]}
        self.assertAlmostEqual(engine.score(bundle, []), .9, places=6)
        self.assertFalse(engine.processor.arguments["do_sample_frames"])
        self.assertFalse(engine.processor.arguments["do_resize"])
        self.assertEqual(engine.model.arguments["logits_to_keep"], 1)
        self.assertNotIn("token_type_ids", engine.model.arguments)

    def test_selection_does_not_need_test_frames_and_scoring_is_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, split, reference = fixture(root)
            for row in rows[:8]:
                pack(root, row["sha256"])
            engine = FakeEngine()
            run = root / "run"
            fit_direct(rows, root, split, root / "unused", run, DirectVLMConfig(device="cpu"),
                       originals_from=reference, engine=engine)
            self.assertFalse(any((root / key).exists() for key in split["test"]))
            self.assertTrue(all(key in split["validation"] for key, _ in engine.calls))
            self.assertTrue(all(set(demos).issubset(split["train"]) for _, demos in engine.calls))
            recipe = json.loads((run / "model_direct_vlm.json").read_text())
            self.assertFalse(recipe["independent_test"])
            self.assertEqual(recipe["test_frame_files_opened"], 0)
            for row in rows[8:]:
                pack(root, row["sha256"])
            report = score_direct(rows, root, split, root / "unused", run,
                                  originals_from=reference, engine=engine)
            self.assertEqual({row["sha256"] for row in report["predictions"]}, set(split["test"]))
            self.assertFalse(report["localization_available"])
            self.assertFalse(report["generation_performed"])
            self.assertTrue(all(not row["events"] for row in report["predictions"]))
            with self.assertRaisesRegex(FileExistsError, "already opened"):
                score_direct(rows, root, split, root / "unused", run, originals_from=reference, engine=engine)
            with self.assertRaisesRegex(FileExistsError, "froze selection"):
                fit_direct(rows, root, split, root / "unused", run, originals_from=reference, engine=engine)

    def test_changed_source_protocol_rejected_before_test_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, split, reference = fixture(root)
            for row in rows[:8]:
                pack(root, row["sha256"])
            run, engine = root / "run", FakeEngine()
            fit_direct(rows, root, split, root / "unused", run, DirectVLMConfig(device="cpu"),
                       originals_from=reference, engine=engine)
            with patch("datathon_ml.direct_vlm._sources", return_value={"wrong": "0" * 64}):
                with self.assertRaisesRegex(ValueError, "fingerprints differ"):
                    score_direct(rows, root, split, root / "unused", run, originals_from=reference, engine=engine)
            self.assertFalse((run / "TEST_OPENED.json").exists())


if __name__ == "__main__":
    unittest.main()

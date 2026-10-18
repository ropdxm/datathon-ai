"""Synthetic detail resolution, TRAIN ownership and locked development scoring."""
import copy
from dataclasses import replace
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

from datathon_ml.direct_context_vlm import (ContextVLMConfig, LocalContextVLM,
    STRATEGIES, demo_owners, fit_context, make_bundle, messages_and_videos,
    probe_context, read_detail_pack, score_context, video_metadata)


def fixture():
    rows = []
    for index in range(16):
        domain = "original" if (index < 4 or 8 <= index < 10 or 12 <= index < 14) else "custom"
        key, label = f"{index + 1:064x}", index % 2
        rows.append({"sha256": key, "group_id": key, "group_verified": True,
            "domain": domain, "label": label, "event_times": [2.] if label else [],
            "label_status": "human_verified", "event_times_verified": True,
            "evaluation_ground_truth_eligible": True})
    split = {"train": [row["sha256"] for row in rows[:8]],
        "validation": [row["sha256"] for row in rows[8:12]],
        "test": [row["sha256"] for row in rows[12:]], "protocol": {"type": "synthetic_verified_scenes"}}
    reference = {"predictions": [{"sha256": row["sha256"], "label": row["label"]}
                                  for row in rows if row["domain"] == "original"]}
    return rows, split, reference


def pack(root, key, *, size=448, length=5):
    directory = Path(root) / key
    directory.mkdir(parents=True, exist_ok=True)
    images, hashes = [], {}
    for index in range(length):
        names = []
        for view in ("full", "middle"):
            name = f"{index:06d}_{view}.jpg"
            Image.new("RGB", (size, size), (index * 10, 30, 40)).save(directory / name)
            names.append(name)
            hashes[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        images.append(names)
    metadata = {"source_sha256": key, "timestamps": list(map(float, range(length))),
        "images": images, "views": ["full", "middle"], "preprocessing": f"raw_rgb_uint8_{size}_letterbox"}
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    hashes["metadata.json"] = hashlib.sha256((directory / "metadata.json").read_bytes()).hexdigest()
    (directory / "_integrity.json").write_text(json.dumps({"completed": True, "members_sha256": hashes}))
    return metadata


class FakeEngine:
    provenance = {"synthetic_engine_for_implementation_test": True}

    def __init__(self):
        self.calls = []

    def score(self, query, demonstrations):
        assert "label" not in query and "event_times" not in query
        self.calls.append((query["source_sha256"], [row["sha256"] for row in demonstrations]))
        return .3 if int(query["source_sha256"], 16) % 2 else .7


class DirectContextTests(unittest.TestCase):
    def test_demo_owners_are_train_only_and_balanced_by_domain_class(self):
        rows, split, _ = fixture()
        owners = demo_owners(rows[:8][::-1])
        self.assertEqual(len(owners), 8)
        for domain in ("original", "custom"):
            for label in (0, 1):
                self.assertEqual(sum(row["domain"] == domain and row["label"] == label for row in owners), 2)
        before = [row["sha256"] for row in owners]
        for row in rows[8:]:
            row.update(label=1 - row["label"], event_times=[9999.])
        self.assertEqual(before, [row["sha256"] for row in demo_owners(rows[:8])])
        self.assertTrue(set(before).issubset(split["train"]))
        with self.assertRaisesRegex(ValueError, "two training owners"):
            demo_owners(rows[:7])

    def test_detail_queries_uniform448_training_anchors224(self):
        with tempfile.TemporaryDirectory() as directory:
            root, key = Path(directory), "a" * 64
            pack(root, key, length=41)
            config = ContextVLMConfig(device="cpu")
            query = make_bundle(root, key, config)
            demo = make_bundle(root, key, config, training_demo={"sha256": key, "label": 1, "event_times": [20.]})
            self.assertEqual(query["frames"].shape, (32, 448, 448, 3))
            self.assertEqual(demo["frames"].shape, (4, 224, 224, 3))
            self.assertEqual(query["timestamps"][0], 0.)
            self.assertEqual(query["timestamps"][-1], 40.)
            self.assertTrue(all(abs(value - 20.) <= 2 for value in demo["timestamps"]))
            self.assertEqual(demo["sampling"], "first_training_point_nearest")
            self.assertNotIn("label", query)
            self.assertNotIn("event_times", query)
            with self.assertRaisesRegex(ValueError, "does not own"):
                make_bundle(root, key, config, training_demo={"sha256": "b" * 64, "label": 0})

    def test_detail_integrity_and_geometry_cannot_silently_reuse224(self):
        with tempfile.TemporaryDirectory() as directory:
            key = "a" * 64
            pack(directory, key, size=224)
            with self.assertRaisesRegex(ValueError, "RGB448"):
                read_detail_pack(directory, key)
            metadata = pack(directory, key)
            path = Path(directory) / key / metadata["images"][0][0]
            Image.new("RGB", (448, 448), (99, 40, 20)).save(path)
            with self.assertRaisesRegex(ValueError, "integrity receipt"):
                read_detail_pack(directory, key)
            metadata = pack(directory, key)
            metadata["images"][0][0] = "../outside.jpg"
            (Path(directory) / key / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "escapes"):
                read_detail_pack(directory, key)
            with self.assertRaisesRegex(ValueError, "opaque source"):
                read_detail_pack(directory, "../outside")

    def test_metadata_clock_supports_mixed_query_and_demo_sizes(self):
        query = {"frames": np.zeros((4, 448, 448, 3), np.uint8), "timestamps": [0., 1.25, 2., 5.003]}
        demo = {"frames": np.zeros((4, 224, 224, 3), np.uint8), "timestamps": [1., 2., 3., 4.]}
        np.testing.assert_allclose(np.asarray(video_metadata(query)["frames_indices"]) / 1000, query["timestamps"])
        self.assertEqual(video_metadata(query)["width"], 448)
        self.assertEqual(video_metadata(demo)["width"], 224)

    def test_messages_have_only_generic_prompt_video_tensors_and_binary_targets(self):
        query = {"source_sha256": "a" * 64, "frames": np.zeros((4, 448, 448, 3), np.uint8),
                 "timestamps": [0., 1., 2., 3.], "description": "PRIVATE_DESCRIPTION_MUST_NOT_APPEAR"}
        demo = {**query, "frames": np.zeros((4, 224, 224, 3), np.uint8)}
        examples = [{"sha256": f"{index + 1:064x}", "label": index % 2, "bundle": demo} for index in range(8)]
        messages, videos, metadata = messages_and_videos(query, examples)
        text = json.dumps(messages)
        self.assertEqual(len(videos), 9)
        self.assertEqual(len(metadata), 9)
        self.assertEqual(metadata[-1]["width"], 448)
        self.assertEqual(metadata[0]["width"], 224)
        self.assertNotIn(query["source_sha256"], text)
        self.assertNotIn(query["description"], text)
        self.assertEqual([row["content"] for row in messages if row["role"] == "assistant"], ["B", "A"] * 4)

    def test_last_token_logits_use_no_generation_or_processor_resampling(self):
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                return "synthetic template"

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
                raise AssertionError("Free text is prohibited")

        engine = LocalContextVLM.__new__(LocalContextVLM)
        engine.device, engine.answer_ids = torch.device("cpu"), [3, 5]
        engine.processor, engine.model = Processor(), Model()
        query = {"frames": np.zeros((4, 448, 448, 3), np.uint8), "timestamps": [0., 1., 2., 3.]}
        self.assertAlmostEqual(engine.score(query, []), .9, places=6)
        self.assertFalse(engine.processor.arguments["do_sample_frames"])
        self.assertFalse(engine.processor.arguments["do_resize"])
        self.assertEqual(engine.model.arguments["logits_to_keep"], 1)
        self.assertFalse(engine.model.arguments["use_cache"])
        self.assertNotIn("token_type_ids", engine.model.arguments)

    def test_synthetic_probe_exercises_zero_and_eight_shot_before_private_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = FakeEngine()
            report = probe_context("unused", directory, ContextVLMConfig(device="cpu"), engine=engine)
            self.assertTrue(report["completed"])
            self.assertFalse(report["private_frames_used"])
            self.assertEqual([len(demos) for _, demos in engine.calls], [0, 8])
            self.assertEqual(set(report["conditional_A_probabilities"]), set(STRATEGIES))
            self.assertEqual(report["attempts"], [{"query_frames": 32, "outcome": "completed"}])

    def test_24frame_fallback_requires_recorded_preprivate_synthetic_cuda_oom(self):
        with self.assertRaisesRegex(ValueError, "synthetic-OOM"):
            ContextVLMConfig(query_frames=24)

        class OomEngine(FakeEngine):
            def score(self, query, demonstrations):
                if len(query["frames"]) == 32 and demonstrations:
                    raise torch.cuda.OutOfMemoryError("Synthetic only")
                return super().score(query, demonstrations)

        # The fake engine never initializes CUDA. Exercise CUDA bookkeeping
        # branches deterministically without requiring any physical GPU index.
        with tempfile.TemporaryDirectory() as directory, \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.reset_peak_memory_stats") as reset_peak, \
                patch("torch.cuda.max_memory_allocated", return_value=0) as peak_memory, \
                patch("torch.cuda.empty_cache") as empty_cache:
            report = probe_context("unused", directory, ContextVLMConfig(device="cuda:1"),
                allow_resource_fallback=True, engine=OomEngine())
            self.assertEqual(report["config"]["query_frames"], 24)
            self.assertEqual(report["attempts"], [{"query_frames": 32, "outcome": "cuda_oom"},
                                                 {"query_frames": 24, "outcome": "completed"}])
            self.assertEqual(reset_peak.call_count, 2)
            peak_memory.assert_called_once_with(torch.device("cuda:1"))
            empty_cache.assert_called_once()
            (Path(directory) / "FIT_STARTED.json").write_text("{}")
            with self.assertRaisesRegex(FileExistsError, "cannot change"):
                probe_context("unused", directory, ContextVLMConfig(device="cpu"), engine=FakeEngine())

    def test_private_fit_requires_matching_probe_before_any_detail_open(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, split, reference = fixture()
            with patch("datathon_ml.direct_context_vlm.make_bundle", side_effect=AssertionError("No private open")):
                with self.assertRaises(FileNotFoundError):
                    fit_context(rows, directory, split, "unused", directory,
                                ContextVLMConfig(device="cpu"), originals_from=reference, engine=FakeEngine())

    def test_validation_selection_needs_no_test_pixels_then_test_opens_once(self):
        class ContextEngine(FakeEngine):
            def score(self, query, demonstrations):
                value = super().score(query, demonstrations)
                return value if demonstrations else .5

        with tempfile.TemporaryDirectory() as directory:
            root, config = Path(directory), ContextVLMConfig(device="cpu")
            rows, split, reference = fixture()
            for row in rows[:12]:
                pack(root, row["sha256"])
            run, engine = root / "run", ContextEngine()
            probe_context("unused", run, config, engine=engine)
            engine.calls.clear()
            fit_context(rows, root, split, "unused", run, config, originals_from=reference, engine=engine)
            artifact = json.loads((run / "model_context_vlm.json").read_text())
            self.assertEqual(len(artifact["demonstrations"]), 8)
            self.assertEqual(artifact["winner"]["strategy"], "dense_balanced_context")
            self.assertFalse(artifact["independent_test"])
            self.assertEqual(artifact["test_frame_files_opened"], 0)
            self.assertTrue(all(key in split["validation"] for key, _ in engine.calls))
            self.assertEqual({len(demos) for _, demos in engine.calls}, {0, 8})
            self.assertTrue(all(set(demos).issubset(split["train"]) for _, demos in engine.calls))
            self.assertFalse(any((root / key).exists() for key in split["test"]))
            for row in rows[12:]:
                pack(root, row["sha256"])
            report = score_context(rows, root, split, "unused", run, originals_from=reference, engine=engine)
            self.assertEqual(set(split["test"]), {row["sha256"] for row in report["predictions"]})
            self.assertFalse(report["independent_test"])
            self.assertFalse(report["generation_performed"])
            self.assertFalse(report["localization_available"])
            self.assertTrue(all(not row["events"] for row in report["predictions"]))
            self.assertTrue(all(len(demos) == 8 for _, demos in engine.calls[-4:]))
            with self.assertRaisesRegex(FileExistsError, "already opened"):
                score_context(rows, root, split, "unused", run, originals_from=reference, engine=engine)
            with self.assertRaisesRegex(FileExistsError, "already started"):
                fit_context(rows, root, split, "unused", run, config, originals_from=reference, engine=engine)

    def test_changed_heldout_targets_do_not_change_demo_selection_or_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config = Path(directory), ContextVLMConfig(device="cpu")
            rows, split, reference = fixture()
            for row in rows[:12]:
                pack(root, row["sha256"])
            first, second = root / "first", root / "second"
            for run in (first, second):
                probe_context("unused", run, config, engine=FakeEngine())
            fit_context(rows, root, split, "unused", first, config, originals_from=reference, engine=FakeEngine())
            altered = copy.deepcopy(rows)
            for row in altered[12:]:
                row["label"] = 1 - row["label"]
                row["event_times"] = [2.] if row["label"] else []
            altered_reference = {"predictions": [{"sha256": row["sha256"], "label": row["label"]}
                for row in altered if row["domain"] == "original"]}
            fit_context(altered, root, split, "unused", second, config,
                        originals_from=altered_reference, engine=FakeEngine())
            before = json.loads((first / "model_context_vlm.json").read_text())
            after = json.loads((second / "model_context_vlm.json").read_text())
            self.assertEqual(before["winner"], after["winner"])
            self.assertEqual(before["demonstrations"], after["demonstrations"])

    def test_changed_source_is_rejected_before_test_pixel_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config = Path(directory), ContextVLMConfig(device="cpu")
            rows, split, reference = fixture()
            for row in rows[:12]:
                pack(root, row["sha256"])
            run, engine = root / "run", FakeEngine()
            probe_context("unused", run, config, engine=engine)
            fit_context(rows, root, split, "unused", run, config, originals_from=reference, engine=engine)
            with patch("datathon_ml.direct_context_vlm._sources", return_value={"different": "0" * 64}):
                with self.assertRaisesRegex(ValueError, "fingerprints differ"):
                    score_context(rows, root, split, "unused", run, originals_from=reference, engine=engine)
            self.assertFalse((run / "TEST_OPENED.json").exists())


if __name__ == "__main__":
    unittest.main()

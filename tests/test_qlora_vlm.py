"""Synthetic supervised adapter ownership, gradient masking and checkpoint locks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from datathon_ml.qlora_vlm import (CANDIDATES, QLoRAConfig, NativeQLoRA,
    conditional_ab_loss, fit_qlora, language_targets, make_bundle, predict_qlora,
    probe_qlora, score_qlora, training_indices)
from tests.test_direct_context_vlm import fixture, pack


class TinyAdapterEngine:
    """Actual autograd/optimizer/checkpoint mechanics, no public weight download."""
    provenance = {"synthetic_native_adapter_implementation_test": True}
    loss_owners = []
    score_owners = []

    def __init__(self, checkpoint, config, adapter_path=None):
        self.device = torch.device("cpu")
        self.lora_weight = nn.Parameter(torch.tensor([.2, -.1]))
        self.frozen = nn.Parameter(torch.tensor([.5]), requires_grad=False)
        if adapter_path is not None:
            values = json.loads((Path(adapter_path) / "adapter_model.safetensors").read_text())
            with torch.no_grad():
                self.lora_weight.copy_(torch.tensor(values))

    def parameters(self):
        return [self.lora_weight]

    def named_parameters(self):
        return [("language_model.layers.31.self_attn.q_proj.lora_A", self.lora_weight),
                ("visual.frozen", self.frozen)]

    def loss(self, bundle, label):
        self.loss_owners.append(bundle["source_sha256"])
        return conditional_ab_loss(self.lora_weight.reshape(1, 2), label)

    def score(self, bundle):
        assert "label" not in bundle and "event_times" not in bundle
        self.score_owners.append(bundle["source_sha256"])
        return .3 if int(bundle["source_sha256"], 16) % 2 else .7

    def save_adapter(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_config.json").write_text(json.dumps({"synthetic": True}))
        (directory / "adapter_model.safetensors").write_text(json.dumps(self.lora_weight.detach().tolist()))

    def close(self):
        pass


class QLoRATests(unittest.TestCase):
    def setUp(self):
        TinyAdapterEngine.loss_owners.clear()
        TinyAdapterEngine.score_owners.clear()

    def test_conditional_objective_correctly_maps_violation_to_a_and_masks_other_tokens(self):
        logits = torch.tensor([[2., -2.]], requires_grad=True)
        positive = conditional_ab_loss(logits, 1)
        normal = conditional_ab_loss(logits, 0)
        self.assertLess(float(positive.detach()), float(normal.detach()))
        positive.backward()
        self.assertLess(float(logits.grad[0, 0]), 0.)
        self.assertGreater(float(logits.grad[0, 1]), 0.)
        with self.assertRaisesRegex(ValueError, "A/B logits"):
            conditional_ab_loss(torch.zeros((1, 100)), 1)

    def test_adapter_targets_last_eight_language_q_v_only(self):
        names = []
        for layer in range(12):
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                names.append((f"model.language_model.layers.{layer}.self_attn.{projection}", None))
                names.append((f"model.visual.blocks.{layer}.self_attn.{projection}", None))
        model = SimpleNamespace(named_modules=lambda: iter(names))
        targets = language_targets(model)
        self.assertEqual(len(targets), 16)
        self.assertTrue(all(".q_proj" in name or ".v_proj" in name for name in targets))
        self.assertTrue(all("language_model" in name and int(name.split(".")[3]) >= 4 for name in targets))

    def test_training_points_inserted_only_into_training_sequences(self):
        times = np.arange(101, dtype=float)
        row = {"label": 1, "event_times": [27., 73.]}
        indices = training_indices(times, row)
        self.assertEqual(len(indices), 16)
        self.assertIn(27, indices)
        self.assertIn(73, indices)
        self.assertEqual(len(set(indices)), 16)
        normal = training_indices(times, {"label": 0, "event_times": [27., 73.]})
        self.assertNotEqual(normal, indices)
        with self.assertRaisesRegex(ValueError, "point anchor"):
            training_indices(times, {"label": 1, "event_times": []})

    def test_label_free_evaluation_uniform448_and_train_budget16(self):
        with tempfile.TemporaryDirectory() as directory:
            key, config = "a" * 64, QLoRAConfig(device="cpu")
            pack(directory, key, length=41)
            query = make_bundle(directory, key, config)
            train = make_bundle(directory, key, config,
                training_row={"sha256": key, "label": 1, "event_times": [19.]})
            self.assertEqual(query["frames"].shape, (32, 448, 448, 3))
            self.assertEqual(train["frames"].shape, (16, 448, 448, 3))
            self.assertIn(19., train["timestamps"])
            self.assertEqual(query["timestamps"][0], 0.)
            self.assertEqual(query["timestamps"][-1], 40.)
            self.assertNotIn("label", query)
            self.assertNotIn("event_times", query)

    def test_probe_executes_real_backward_only_adapter_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            report = probe_qlora("unused", directory, QLoRAConfig(device="cpu"), engine_factory=TinyAdapterEngine)
            self.assertTrue(report["completed"])
            self.assertTrue(report["adapter_reload_verified"])
            self.assertLessEqual(report["adapter_reload_probability_absolute_difference"], 1e-4)
            self.assertFalse(report["private_frames_used"])
            self.assertEqual(report["gradient_audit"]["frozen_parameter_gradient_tensors"], 0)
            self.assertGreater(report["gradient_audit"]["nonzero_adapter_gradient_tensors"], 0)
            self.assertEqual(set(TinyAdapterEngine.loss_owners), {"0" * 64})
            self.assertEqual(set(TinyAdapterEngine.score_owners), {"0" * 64})

    def test_frozen_parameter_gradient_is_rejected(self):
        class BadEngine(TinyAdapterEngine):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.frozen.requires_grad_(True)

            def loss(self, bundle, label):
                return super().loss(bundle, label) + self.frozen.sum()

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "frozen vision/base/head"):
                probe_qlora("unused", directory, QLoRAConfig(device="cpu"), engine_factory=BadEngine)

    def test_probe_is_required_before_private_bundles_and_no_postfit_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, split, reference = fixture()
            with patch("datathon_ml.qlora_vlm.make_bundle", side_effect=AssertionError("No private frames")):
                with self.assertRaises(FileNotFoundError):
                    fit_qlora(rows, directory, split, "unused", directory, originals_from=reference,
                              engine_factory=TinyAdapterEngine)
            (Path(directory) / "FIT_STARTED.json").write_text("{}")
            with self.assertRaisesRegex(FileExistsError, "cannot change"):
                probe_qlora("unused", directory, QLoRAConfig(device="cpu"), engine_factory=TinyAdapterEngine)
            with self.assertRaisesRegex(ValueError, "synthetic-OOM"):
                QLoRAConfig(image_size=224)

    def test_training_uses_only_train_owners_and_selects_without_test_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config = Path(directory), QLoRAConfig(device="cpu")
            rows, split, reference = fixture()
            for row in rows[:12]:
                pack(root, row["sha256"])
            run = root / "run"
            probe_qlora("unused", run, config, engine_factory=TinyAdapterEngine)
            TinyAdapterEngine.loss_owners.clear()
            TinyAdapterEngine.score_owners.clear()
            fit_qlora(rows, root, split, "unused", run, config, originals_from=reference,
                      engine_factory=TinyAdapterEngine)
            self.assertTrue(set(TinyAdapterEngine.loss_owners).issubset(split["train"]))
            self.assertEqual(len(TinyAdapterEngine.loss_owners), len(split["train"]) * sum(row[2] for row in CANDIDATES))
            self.assertTrue(set(TinyAdapterEngine.score_owners).issubset(split["validation"]))
            self.assertFalse(any((root / key).exists() for key in split["test"]))
            artifact = json.loads((run / "model_qlora_vlm.json").read_text())
            self.assertFalse(artifact["independent_test"])
            self.assertEqual(artifact["test_frame_files_opened"], 0)
            for row in rows[12:]:
                pack(root, row["sha256"])
            report = score_qlora(rows, root, split, "unused", run, originals_from=reference,
                                 engine_factory=TinyAdapterEngine)
            self.assertEqual(len(report["predictions"]), 4)
            self.assertFalse(report["independent_test"])
            self.assertFalse(report["localization_available"])
            with self.assertRaisesRegex(FileExistsError, "already opened"):
                score_qlora(rows, root, split, "unused", run, originals_from=reference,
                            engine_factory=TinyAdapterEngine)
            result = predict_qlora(root, rows[-1]["sha256"], "unused", run, engine_factory=TinyAdapterEngine)
            self.assertNotIn("label", result)
            self.assertEqual(result["events"], [])

    def test_changed_adapter_checkpoint_is_rejected_before_test_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config = Path(directory), QLoRAConfig(device="cpu")
            rows, split, reference = fixture()
            for row in rows[:12]:
                pack(root, row["sha256"])
            run = root / "run"
            probe_qlora("unused", run, config, engine_factory=TinyAdapterEngine)
            fit_qlora(rows, root, split, "unused", run, config, originals_from=reference,
                      engine_factory=TinyAdapterEngine)
            (run / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "adapter bytes"):
                score_qlora(rows, root, split, "unused", run, originals_from=reference,
                            engine_factory=TinyAdapterEngine)
            self.assertFalse((run / "TEST_OPENED.json").exists())

    def test_native_forward_prompt_has_no_answer_or_private_description(self):
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return "fixed public template"

            def __call__(self, **kwargs):
                self.arguments = kwargs
                return {"input_ids": torch.ones((1, 2), dtype=torch.long), "pixel_values_videos": torch.zeros((1, 2))}

        class Model:
            def __call__(self, **kwargs):
                self.arguments = kwargs
                return SimpleNamespace(logits=torch.zeros((1, 1, 7), requires_grad=True))

        engine = NativeQLoRA.__new__(NativeQLoRA)
        engine.device, engine.answer_ids = torch.device("cpu"), [2, 4]
        engine.processor, engine.model = Processor(), Model()
        bundle = {"source_sha256": "a" * 64, "description": "NEVER_INCLUDE",
            "frames": np.zeros((2, 448, 448, 3), np.uint8), "timestamps": [0., 1.]}
        logits = engine._logits(bundle)
        self.assertEqual(logits.shape, (1, 2))
        text = json.dumps(engine.processor.messages)
        self.assertNotIn("NEVER_INCLUDE", text)
        self.assertNotIn("a" * 64, text)
        self.assertFalse(any(row["role"] == "assistant" for row in engine.processor.messages))
        self.assertFalse(engine.processor.arguments["do_sample_frames"])
        self.assertFalse(engine.processor.arguments["do_resize"])
        self.assertEqual(engine.model.arguments["logits_to_keep"], 1)


if __name__ == "__main__":
    unittest.main()

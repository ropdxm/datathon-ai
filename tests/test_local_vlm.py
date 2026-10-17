"""No downloads/GPU required: temporal policy, cache and forward-only contract."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from datathon_ml.features import load_feature_cache, sha256_file
from datathon_ml.local_vlm import (LocalVLMConfig, TransformersWindowEncoder,
    _resolve_checkpoint, checkpoint_fingerprint, extract_features, last_token_pool, uniform_windows)


class GuardedRecord(dict):
    def get(self, key, default=None):
        if key not in {"path", "video_path", "sha256"}:
            raise AssertionError("Extractor accessed labels, anchors, or an identifier")
        return super().get(key, default)


class FakeEncoder:
    provenance = {"device": "cpu", "precision": "cpu-float32"}

    def __init__(self):
        self.calls = 0

    def encode(self, clips):
        self.calls += 1
        # Synthetic pixels only. Embeddings never leave this private runtime.
        return np.asarray([np.arange(1, 9) + sum(np.asarray(im).mean() for im in clip) / 255.
                           for clip in clips], dtype=np.float32)


def fixture(root):
    weights = root / "weights"
    weights.mkdir()
    (weights / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    (weights / "model.safetensors").write_bytes(b"fake-test-weights")
    source = root / "synthetic.mov"
    source.write_bytes(b"synthetic-video-source")
    records = [GuardedRecord(path=str(source), sha256=sha256_file(source), label=1, event_times=[9999.])]
    frames = [(float(t), np.full((48, 32, 3), t * 10, dtype=np.uint8)) for t in range(0, 11, 2)]
    return weights, records, frames


class LocalVLMTests(unittest.TestCase):
    def test_config_is_small_and_rejects_invalid_sampling(self):
        self.assertEqual(LocalVLMConfig().backbone, "qwen3_vl_embedding_2b")
        self.assertEqual(LocalVLMConfig().sample_fps, .25)
        for fields in ({"sample_fps": 0}, {"image_size": 449}, {"backbone": "hosted-provider"}, {"frames_per_window": 1}):
            with self.assertRaises(ValueError):
                LocalVLMConfig(**fields)

    def test_uniform_contexts_have_no_annotation_input_and_cover_end(self):
        centers, indices = uniform_windows(np.arange(0., 11., 2.), LocalVLMConfig())
        np.testing.assert_array_equal(centers, [0., 4., 8., 10.])
        self.assertEqual(indices.shape, (4, 4))
        self.assertTrue(np.all(np.diff(indices, axis=1) >= 0))
        self.assertEqual(indices[0, 0], 0)
        self.assertEqual(indices[-1, -1], 5)
        centers, indices = uniform_windows(np.array([0.]), LocalVLMConfig())
        np.testing.assert_array_equal(centers, [0.])
        np.testing.assert_array_equal(indices, np.zeros((1, 4), dtype=int))

    def test_pooling_handles_left_and_right_padding(self):
        import torch
        hidden = torch.arange(30).reshape(2, 5, 3).float()
        mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]])
        pooled = last_token_pool(hidden, mask)
        torch.testing.assert_close(pooled, torch.stack([hidden[0, 2], hidden[1, 4]]))
        with self.assertRaises(ValueError):
            last_token_pool(hidden, torch.zeros_like(mask))

    def test_extraction_ignores_annotations_and_reuses_valid_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, records, frames = fixture(root)
            config = LocalVLMConfig(local_checkpoint=str(checkpoint), allow_download=False, image_size=64, batch_size=2)
            engine = FakeEncoder()
            with patch("datathon_ml.local_vlm.iter_video_frames", return_value=iter(frames)):
                paths = extract_features(records, root / "cache", config, engine=engine)
            self.assertEqual(engine.calls, 2)
            cache = load_feature_cache(next(iter(paths.values())))
            self.assertEqual(cache["features"].shape, (4, 1, 8))
            self.assertEqual(cache["features"].dtype, np.float32)
            np.testing.assert_allclose(np.linalg.norm(cache["features"], axis=-1), 1., atol=1e-6)
            self.assertFalse(cache["provenance"]["generation_used"])
            self.assertFalse(cache["provenance"]["hosted_inference_used"])
            self.assertNotIn("event_times", cache["config"])
            with patch("datathon_ml.local_vlm.iter_video_frames", side_effect=AssertionError("Should reuse")):
                reused = extract_features(records + records, root / "cache", config, engine=engine)
            self.assertEqual(paths, reused)
            self.assertEqual(engine.calls, 2)

    def test_source_and_checkpoint_bytes_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, records, frames = fixture(root)
            fingerprint = checkpoint_fingerprint(checkpoint)
            (checkpoint / "model.safetensors").write_bytes(b"changed-test-weights")
            self.assertNotEqual(fingerprint, checkpoint_fingerprint(checkpoint))
            records[0]["sha256"] = "0" * 64
            config = LocalVLMConfig(local_checkpoint=str(checkpoint), allow_download=False, image_size=64)
            with self.assertRaisesRegex(ValueError, "Source bytes"):
                extract_features(records, root / "cache", config, engine=FakeEncoder())

    def test_chat_template_is_included_in_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, _, _ = fixture(Path(tmp))
            template = checkpoint / "chat_template.jinja"
            template.write_text("original-template", encoding="utf-8")
            before = checkpoint_fingerprint(checkpoint)
            template.write_text("changed-template", encoding="utf-8")
            self.assertNotEqual(before, checkpoint_fingerprint(checkpoint))

    def test_snapshot_download_retains_native_chat_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, _, _ = fixture(Path(tmp))
            captured = []

            def fake_snapshot_download(**kwargs):
                captured.append(kwargs)
                return str(checkpoint)

            fake_hub = SimpleNamespace(snapshot_download=fake_snapshot_download)
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                _resolve_checkpoint(LocalVLMConfig())
                _resolve_checkpoint(LocalVLMConfig(allow_download=False))
            self.assertIn("*.jinja", captured[0]["allow_patterns"])
            self.assertEqual(captured[0]["revision"], "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda")
            self.assertFalse(captured[0]["local_files_only"])
            self.assertTrue(captured[1]["local_files_only"])

    def test_native_forward_never_calls_generate(self):
        import torch

        class FakeProcessor:
            def apply_chat_template(self, conversations, **kwargs):
                self.conversations = conversations
                return ["fixed-local-template"] * len(conversations)

            def __call__(self, **kwargs):
                return {"input_ids": torch.ones((len(kwargs["text"]), 5), dtype=torch.long),
                        "attention_mask": torch.tensor([[1, 1, 1, 0, 0]] * len(kwargs["text"]))}

        class FakeTop:
            def model(self, **kwargs):
                if kwargs["use_cache"] or not kwargs["return_dict"]:
                    raise AssertionError("Unbounded/unsupported forward")
                count = len(kwargs["input_ids"])
                return SimpleNamespace(last_hidden_state=torch.arange(count * 5 * 6).reshape(count, 5, 6).float())

            def generate(self, *args, **kwargs):
                raise AssertionError("Text generation must never run")

        encoder = TransformersWindowEncoder.__new__(TransformersWindowEncoder)
        encoder.config = LocalVLMConfig(image_size=64)
        encoder.device = torch.device("cpu")
        encoder.processor, encoder.model = FakeProcessor(), FakeTop()
        synthetic = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        embeddings = encoder.encode([[synthetic] * 4, [synthetic] * 4])
        self.assertEqual(embeddings.shape, (2, 6))
        self.assertTrue(np.isfinite(embeddings).all())
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1., atol=1e-6)
        self.assertEqual(len(encoder.processor.conversations), 2)


if __name__ == "__main__":
    unittest.main()

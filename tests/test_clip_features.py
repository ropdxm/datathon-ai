"""CLIP extraction contracts without downloading pretrained weights."""

from contextlib import ExitStack
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from datathon_ml.clip_features import (BACKBONE, ClipFeatureConfig,
    _encode_clip_views, _resolve_clip_weights, extract_clip_features)
from datathon_ml.features import load_feature_cache


class ClipFeatureTests(unittest.TestCase):
    def test_config_rejects_incompatible_encoder_geometry_and_text(self):
        for kwargs in ({"image_size": 256}, {"sample_fps": 0}, {"sample_fps": float("nan")},
                       {"backbone": "other"}, {"views": ("full", "full")},
                       {"use_text_features": True}, {"local_repo": "other"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ClipFeatureConfig(**kwargs)

    def test_local_checkpoint_hash_and_offline_gate(self):
        class TorchHub:
            class hub:
                @staticmethod
                def get_dir():
                    return str(root / "hub")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"known checkpoint")
            config = ClipFeatureConfig(local_checkpoint=str(checkpoint), allow_download=False)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                _resolve_clip_weights(config, TorchHub)
            with patch("datathon_ml.clip_features.EXPECTED_WEIGHT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
                self.assertEqual(_resolve_clip_weights(config, TorchHub), checkpoint)
            with self.assertRaises(FileNotFoundError):
                _resolve_clip_weights(ClipFeatureConfig(allow_download=False), TorchHub)

    def test_mocked_extraction_full_timeline_dedup_cache_and_annotation_independence(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.mov"
            video.write_bytes(b"mock source bytes")
            digest = hashlib.sha256(video.read_bytes()).hexdigest()
            checkpoint = root / "weights.pt"
            checkpoint.write_bytes(b"mock weights")
            frames = [(t, np.full((16, 8, 3), i * 40, dtype=np.uint8)) for i, t in enumerate((0., .5, 1.3))]
            def encode(images, *_):
                values = np.ones((len(images), 512), dtype=np.float32)
                return values / np.sqrt(512.)
            config = ClipFeatureConfig(device="cpu", batch_size=8, allow_download=False)
            with ExitStack() as stack:
                stack.enter_context(patch("datathon_ml.clip_features._open_clip_version", return_value="3.2.0"))
                stack.enter_context(patch("datathon_ml.clip_features._resolve_clip_weights", return_value=checkpoint))
                loader = stack.enter_context(patch("datathon_ml.clip_features._load_clip_encoder", return_value=object()))
                decoder = stack.enter_context(patch("datathon_ml.clip_features.iter_video_frames", return_value=iter(frames)))
                stack.enter_context(patch("datathon_ml.clip_features._encode_clip_views", side_effect=encode))
                rows = [{"path": str(video), "sha256": digest, "label": 1, "event_times": [9999.]},
                        {"path": str(video), "sha256": digest, "label": 0, "event_times": []}]
                paths = extract_clip_features(rows, root / "cache", config)
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(decoder.call_count, 1)
                cached = load_feature_cache(paths[digest])
                self.assertEqual(cached["features"].shape, (3, 4, 512))
                np.testing.assert_allclose(cached["timestamps"], [0., .5, 1.3])
                self.assertEqual(cached["config"]["backbone"], BACKBONE)
                self.assertFalse(cached["config"]["use_text_features"])
                self.assertNotIn("event_times", cached["config"])
                loader.reset_mock()
                decoder.reset_mock()
                changed = [{"path": str(video), "sha256": digest, "label": 0, "event_times": [0.]}]
                self.assertEqual(extract_clip_features(changed, root / "cache", config), paths)
                loader.assert_not_called()
                decoder.assert_not_called()
                self.assertFalse(list((root / "cache").glob("*.tmp")))

    def test_actual_tensor_encoder_normalization_and_batching(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        class FakeEncoder:
            def __init__(self):
                self.batches = []
            def encode_image(self, tensor):
                self.batches.append(tensor.detach().cpu().clone())
                self.assert_no_grad = not torch.is_grad_enabled()
                return torch.ones((len(tensor), 512))
        encoder = FakeEncoder()
        images = np.full((5, 224, 224, 3), 255, dtype=np.uint8)
        values = _encode_clip_views(images, encoder, ClipFeatureConfig(device="cpu", batch_size=2), torch.device("cpu"), torch)
        self.assertEqual([len(batch) for batch in encoder.batches], [2, 2, 1])
        self.assertTrue(encoder.assert_no_grad)
        self.assertAlmostEqual(float(encoder.batches[0][0, 0, 0, 0]), (1-.48145466)/.26862954, places=5)
        np.testing.assert_allclose(np.linalg.norm(values, axis=-1), np.ones(5), atol=1e-6)


if __name__ == "__main__":
    unittest.main()

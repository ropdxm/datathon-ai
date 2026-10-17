"""Feature geometry, sequential decoding and cache correctness; no model download."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from datathon_ml.features import (
    FeatureConfig,
    _canonical_json,
    _configuration_hash,
    cache_is_valid,
    iter_video_frames,
    load_feature_cache,
    make_views,
)


class FeatureTests(unittest.TestCase):
    def test_portrait_letterbox_keeps_full_height_and_distinct_crops(self):
        image = np.zeros((12, 6, 3), dtype=np.uint8)
        image[:4] = (255, 0, 0)
        image[4:8] = (0, 255, 0)
        image[8:] = (0, 0, 255)
        full, top, middle, bottom = make_views(image, 12, ("full", "top", "middle", "bottom"))
        np.testing.assert_array_equal(full[:, 3:9], image)
        np.testing.assert_array_equal(full[0, 0], (124, 116, 104))
        self.assertGreater(float(top[..., 0].mean()), float(bottom[..., 0].mean()))
        self.assertGreater(float(bottom[..., 2].mean()), float(top[..., 2].mean()))
        self.assertGreater(float(middle[..., 1].mean()), float(top[..., 1].mean()))

    def test_invalid_sample_rates_and_patch_sizes_rejected(self):
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                FeatureConfig(sample_fps=invalid)
        with self.assertRaises(ValueError):
            FeatureConfig(image_size=225)

    def test_cache_rejects_different_source_configuration_and_corruption(self):
        config_json = _canonical_json({"views": ["full"], "sample_fps": 2.0})
        source_sha = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            np.savez_compressed(
                path, features=np.ones((2, 1, 3), dtype=np.float32),
                timestamps=np.array([0.0, 0.5], dtype=np.float32),
                sha256=np.array(source_sha), config_json=np.array(config_json),
                config_hash=np.array(_configuration_hash(config_json)),
                provenance_json=np.array(json.dumps({"weights_sha256": "b" * 64})),
            )
            self.assertTrue(cache_is_valid(path, source_sha, config_json))
            self.assertFalse(cache_is_valid(path, "c" * 64, config_json))
            self.assertFalse(cache_is_valid(path, source_sha, config_json + " "))
            self.assertEqual(load_feature_cache(path)["features"].shape, (2, 1, 3))
            path.write_bytes(b"interrupted write")
            self.assertFalse(cache_is_valid(path, source_sha, config_json))

    def test_sampling_uses_complete_timeline_and_final_valid_frame(self):
        class FakeCapture:
            def __init__(self):
                self.index = -1
                self.released = False
            def isOpened(self):
                return True
            def get(self, prop):
                if prop == cv2.CAP_PROP_FPS:
                    return 10.0
                if prop == cv2.CAP_PROP_POS_MSEC:
                    return self.index * 100.0
                return 9999  # Deliberately incorrect frame-count metadata.
            def grab(self):
                self.index += 1
                return self.index < 14
            def retrieve(self):
                return True, np.full((3, 2, 3), self.index, dtype=np.uint8)
            def release(self):
                self.released = True
        capture = FakeCapture()
        with patch("datathon_ml.features.cv2.VideoCapture", return_value=capture):
            samples = list(iter_video_frames("unused.mov", 2.0))
        np.testing.assert_allclose([item[0] for item in samples], [0.0, 0.5, 1.0, 1.3])
        self.assertEqual(samples[-1][1][0, 0, 0], 13)
        self.assertTrue(capture.released)

    def test_broken_pts_falls_back_to_frame_index_without_duplicate_times(self):
        class FakeCapture:
            def __init__(self):
                self.index = -1
            def isOpened(self):
                return True
            def get(self, prop):
                return 10.0 if prop == cv2.CAP_PROP_FPS else 0.0
            def grab(self):
                self.index += 1
                return self.index < 7
            def retrieve(self):
                return True, np.zeros((2, 2, 3), dtype=np.uint8)
            def release(self):
                pass
        with patch("datathon_ml.features.cv2.VideoCapture", return_value=FakeCapture()):
            samples = list(iter_video_frames("unused.mov", 2.0))
        np.testing.assert_allclose([item[0] for item in samples], [0.0, 0.5, 0.6])


if __name__ == "__main__":
    unittest.main()

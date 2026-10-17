"""Temporal clip geometry and head correctness without network/model downloads."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from datathon_ml.motion import MotionConfig, configure_motion_head, iter_motion_clips, select_clip


class MotionTests(unittest.TestCase):
    def test_official_temporal_configuration(self):
        small = MotionConfig()
        self.assertEqual(small.num_frames, 13)
        self.assertAlmostEqual(small.clip_duration_sec, 2.6)
        self.assertEqual(MotionConfig(backbone="x3d_xs").num_frames, 4)
        self.assertAlmostEqual(MotionConfig(backbone="x3d_xs").clip_duration_sec, 1.6)
        for invalid in (0, float("nan"), -1):
            with self.assertRaises(ValueError):
                MotionConfig(sample_fps=invalid)

    def test_nearest_time_clip_has_temporal_order_and_boundary_padding(self):
        times = [0.0, 1.0, 2.0]
        frames = [np.full((2, 3, 3, 3), index, np.uint8) for index in range(3)]
        clip = select_clip(times, frames, center=0.5, duration=4, num_frames=5)
        self.assertEqual(clip.shape, (2, 3, 5, 3, 3))
        np.testing.assert_array_equal(clip[0, 0, :, 0, 0], [0, 0, 0, 1, 2])

    def test_stream_covers_start_middle_and_final_without_anchors(self):
        config = MotionConfig(image_size=32, views=("middle",))
        frames = [(index / 10, np.full((8, 4, 3), index, np.uint8)) for index in range(37)]
        with patch("datathon_ml.motion.iter_video_frames", return_value=iter(frames)):
            windows = list(iter_motion_clips("unused.mov", config))
        np.testing.assert_allclose([window[0] for window in windows], [0, 1, 2, 3, 3.6])
        self.assertEqual(windows[0][1].shape, (1, 3, 13, 32, 32))
        self.assertEqual(windows[0][1][0, 0, 0, 0, 0], 0)
        self.assertEqual(windows[-1][1][0, 0, -1, 0, 0], 36)

    def test_very_short_recording_still_produces_valid_clips(self):
        frames = [(0.0, np.zeros((8, 4, 3), np.uint8)), (0.1, np.ones((8, 4, 3), np.uint8))]
        with patch("datathon_ml.motion.iter_video_frames", return_value=iter(frames)):
            windows = list(iter_motion_clips("unused.mov", MotionConfig(image_size=32)))
        np.testing.assert_allclose([window[0] for window in windows], [0.0, 0.1])
        self.assertEqual(windows[0][1].shape, (2, 3, 13, 32, 32))

    def test_feature_head_removes_classifier_and_freezes_parameters(self):
        import torch
        class DummyX3D(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = torch.nn.Conv3d(3, 2, kernel_size=1)
                self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
                self.blocks[0].pool = torch.nn.Module()
                self.blocks[0].pool.pool = torch.nn.AvgPool3d((13, 5, 5))
                self.blocks[0].proj = torch.nn.Linear(2048, 400)
                self.blocks[0].activation = torch.nn.Softmax(dim=1)
        model = configure_motion_head(DummyX3D(), "penultimate", torch)
        self.assertIsInstance(model.blocks[-1].proj, torch.nn.Identity)
        self.assertIsInstance(model.blocks[-1].pool.pool, torch.nn.AdaptiveAvgPool3d)
        self.assertIsNone(model.blocks[-1].activation)
        self.assertFalse(model.training)
        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 0)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()

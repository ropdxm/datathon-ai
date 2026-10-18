"""No private media is used in these grouping tests."""
import unittest

import numpy as np

from tools.reviewed_scene_groups import FPS, HEIGHT, WIDTH, components, frame_features, overlap_evidence


class ReviewedSceneGroupsTests(unittest.TestCase):
    def test_ambiguous_edges_union_transitively(self):
        groups = components(["a", "b", "c", "d"], [("a", "b"), ("b", "c")])
        self.assertEqual(groups, [["a", "b", "c"], ["d"]])
        with self.assertRaises(ValueError):
            components(["a"], [("a", "missing")])

    def test_shifted_dynamic_overlap_is_grouped(self):
        rng = np.random.default_rng(42)
        frames = rng.integers(0, 255, (30, HEIGHT, WIDTH), dtype=np.uint8)
        result = overlap_evidence(frame_features(frames[4:]), frame_features(frames[:-4]))
        self.assertTrue(result["plausible_overlap"])
        self.assertEqual(result["offset_seconds"], 4 / FPS)
        self.assertEqual(result["pixel_mae"], 0.)

    def test_static_same_background_does_not_establish_overlap(self):
        frames = np.full((20, HEIGHT, WIDTH), 120, dtype=np.uint8)
        result = overlap_evidence(frame_features(frames), frame_features(frames))
        self.assertFalse(result["plausible_overlap"])
        self.assertEqual(result["dynamic_steps"], 0)

    def test_unrelated_video_and_short_video_do_not_establish_overlap(self):
        rng = np.random.default_rng(20)
        a = rng.integers(0, 255, (20, HEIGHT, WIDTH), dtype=np.uint8)
        b = rng.integers(0, 255, a.shape, dtype=np.uint8)
        self.assertFalse(overlap_evidence(frame_features(a), frame_features(b))["plausible_overlap"])
        self.assertFalse(overlap_evidence(frame_features(a[:3]), frame_features(a))["reviewable"])


if __name__ == "__main__":
    unittest.main()

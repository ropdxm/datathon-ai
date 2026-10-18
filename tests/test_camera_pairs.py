"""Synthetic tests for pairing timing, abstention, and privacy boundaries."""
import unittest

import numpy as np

from tools.match_custom_cameras import (PROTOCOL, acceptance_reasons, align_motion,
                                       creation_time, metadata_reliability, partial_assignment, video_command)


class CameraPairsTests(unittest.TestCase):
    def test_motion_offset_sign_and_partial_overlap(self):
        rng = np.random.default_rng(7)
        base = rng.gamma(1.4, 2., (180, 4))
        # Right recording begins two seconds earlier than left recording.
        result = align_motion(base[10:160], base[:150])
        self.assertEqual(result["offset_seconds"], 2.)
        self.assertGreater(result["motion_score"], .98)
        self.assertGreater(min(result["split_correlations"]), .98)
        self.assertEqual(result["drift_seconds"], 0.)

    def test_constant_or_short_video_cannot_be_accepted(self):
        for trace in (np.zeros((100, 4)), np.ones((20, 4))):
            result = align_motion(trace, trace)
            reasons = acceptance_reasons(result, .5, 0., True)
            self.assertTrue(reasons)
            self.assertIn("insufficient_motion_variation", reasons)

    def test_metadata_proximity_alone_cannot_accept(self):
        rng = np.random.default_rng(100)
        result = align_motion(rng.random((160, 4)), rng.random((160, 4)))
        self.assertIn("motion_correlation_below_threshold", acceptance_reasons(result, .5, 0., True))

    def test_empty_trace_abstains(self):
        result = align_motion(np.empty((0, 4)), np.ones((40, 4)))
        self.assertFalse(result["informative"])
        self.assertIsNone(result["offset_seconds"])

    def test_conflicting_creation_tags_are_not_capture_time_proof(self):
        records = [{"source_suffix": ".mov", "creation_time_unix": 100. + i * 5,
                    "duration_seconds": 20.} for i in range(3)]
        result = metadata_reliability(records)
        self.assertEqual(result["mov"]["adjacent_gaps_shorter_than_both_durations"], 2)
        self.assertFalse(result["mov"]["creation_tag_capture_time_assumption_supported"])

    def test_clock_bias_needs_other_pairs(self):
        rng = np.random.default_rng(10)
        base = rng.random((150, 4))
        result = align_motion(base, base)
        self.assertIn("device_clock_bias_not_independently_supported",
                      acceptance_reasons(result, .2, None, False))

    def test_partial_assignment_abstains_and_has_unique_owners(self):
        result = partial_assignment([[.9, .2], [.8, .7], [-.1, -.5]])
        self.assertEqual(set(result), {(0, 0), (1, 1)})
        self.assertEqual(len({a for a, _ in result}), len(result))
        self.assertEqual(len({b for _, b in result}), len(result))

    def test_video_only_command_and_creation_timezone(self):
        command = video_command("silent.mp4")
        self.assertIn("-an", command)
        self.assertEqual(command[command.index("-map") + 1], "0:v:0")
        self.assertNotIn("0:a", command)
        metadata = {"source_ffprobe": {"format": {"tags": {"creation_time": "2026-10-17T12:00:00Z"}}}}
        self.assertIsInstance(creation_time(metadata), float)
        metadata["source_ffprobe"]["format"]["tags"]["creation_time"] = "2026-10-17T12:00:00"
        self.assertIsNone(creation_time(metadata))
        self.assertIn("same conservative", PROTOCOL["validation_group_policy"])


if __name__ == "__main__":
    unittest.main()

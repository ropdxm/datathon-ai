import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from datathon_ml.custom_annotation import (AnnotationConfig, aggregate_windows, annotate,
    parse_answer, window_indices, timeline_support, valid_cached_proposal)
from datathon_ml.features import sha256_file


class CustomAnnotationTests(unittest.TestCase):
    def test_strict_numeric_answers_and_invalid_output_remains_unknown(self):
        self.assertEqual(parse_answer('[2, 3]', 8), (2, 3, True))
        self.assertEqual(parse_answer('[0, -1]', 8), (0, -1, True))
        for value in ('[true,1]', '[1,8]', '[0,0]', '[7,1]', '[1,2.0]', 'violation at 3s'):
            self.assertEqual(parse_answer(value, 8), (9, -1, False))

    def test_fixed_uniform_windows_cover_recording_without_labels(self):
        config = AnnotationConfig()
        times = np.arange(0, 21, .5)
        windows = window_indices(times, 21, config)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], 21)
        self.assertTrue(all(len(indices) == 8 for _, _, indices in windows))
        self.assertTrue(all(a[1] >= b[0] for a, b in zip(windows, windows[1:])))
        self.assertEqual(windows[-1][2][-1], len(times) - 1)
        for start, end, indices in window_indices(np.arange(0, 47.65, .5), 47.65, config):
            self.assertTrue(all(start <= t < end for t in np.arange(0, 47.65, .5)[indices]))

    def test_unknown_and_short_recordings_do_not_become_normal(self):
        config = AnnotationConfig()
        self.assertIsNone(aggregate_windows([], 1.2, config)['label'])
        self.assertIsNone(aggregate_windows([{'valid': True, 'code': 9}], 10, config)['label'])
        self.assertIsNone(aggregate_windows([{'valid': True, 'code': 5,
            'event_time_sec': 4.}], 10, config)['label'])
        self.assertEqual(aggregate_windows([{'valid': True, 'code': 0}], 10, config)['label'], 'no violation')
        rows = [{'valid': True, 'code': 2, 'event_time_sec': t} for t in (4., 5., 12.)]
        self.assertEqual(aggregate_windows(rows, 20, config)['event_times'], [4., 12.])
        with self.assertRaises(ValueError):
            aggregate_windows([{'valid': True, 'code': 1, 'event_time_sec': 20}], 20, config)

    def test_missing_timeline_support_cannot_become_a_negative(self):
        config = AnnotationConfig()
        self.assertTrue(timeline_support(np.arange(0, 5, .5), 5, config)['sampling_support_complete'])
        for times in ([0, .5, 1.0], [0, .5, 3., 4.5], [2., 2.5, 3., 4.5]):
            support = timeline_support(times, 5, config)
            self.assertFalse(support['sampling_support_complete'])
            self.assertIsNone(aggregate_windows([{'valid': True, 'code': 0}], 5, config,
                support['sampling_support_complete'])['label'])

    def test_silent_visual_proposals_ignore_given_labels_and_resume(self):
        class Observer:
            def __init__(self): self.calls = 0
            def observe(self, images):
                self.calls += 1
                return 2, 2, True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'video.mp4'
            video.write_bytes(b'test source')
            record = {'sha256': sha256_file(video), 'source_sha256': 'a' * 64,
                'path': str(video), 'duration_sec': 5., 'label': 0, 'event_times': [4.5]}
            pixels = [(t, np.zeros((64, 64, 3), dtype=np.uint8)) for t in np.arange(0, 5, .5)]
            observer = Observer()
            with patch('datathon_ml.custom_annotation.assert_silent'), patch(
                    'datathon_ml.custom_annotation.iter_video_frames', return_value=iter(pixels)):
                result = annotate([record], root / 'out', observer, AnnotationConfig(), 'protocol')
            proposed = result['records'][0]
            self.assertEqual(proposed['label'], 'violation')
            self.assertFalse(proposed['audio_used'])
            self.assertFalse(proposed['evaluation_ground_truth_eligible'])
            self.assertTrue(valid_cached_proposal(proposed, record, AnnotationConfig(), 'protocol'))
            broken = {**proposed, 'label': 'no violation'}
            self.assertFalse(valid_cached_proposal(broken, record, AnnotationConfig(), 'protocol'))
            self.assertFalse(valid_cached_proposal(proposed, {**record, 'duration_sec': 7.}, AnnotationConfig(), 'protocol'))
            self.assertNotIn('windows', json.loads((root / 'out/proposals.json').read_text())['records'][0])
            record['label'], record['event_times'] = 1, [1.5]
            with patch('datathon_ml.custom_annotation.assert_silent'), patch(
                    'datathon_ml.custom_annotation.iter_video_frames', side_effect=AssertionError('cache should resume')):
                resumed = annotate([record], root / 'out', observer, AnnotationConfig(), 'protocol')
            self.assertEqual(observer.calls, 1)
            self.assertEqual(result['records'], resumed['records'])


if __name__ == '__main__':
    unittest.main()

import copy
import unittest

from tools.benchmark_custom_annotation_detail import middle_images, select_records, uniform_indices


class CustomAnnotationDetailTests(unittest.TestCase):
    def records(self):
        return [{"sha256": f"{camera * 100 + index:064x}", "source_sha256": f"{camera * 1000 + index:064x}",
                 "camera_id": name, "path": f"video_{camera}_{index}.mp4", "duration_sec": 12.,
                 "label": index % 2, "event_times": [3.]}
                for camera, name in enumerate(("iphone_mov", "samsung_mp4"), 1) for index in range(5)]

    def test_balanced_fixed_quantile_selection_excludes_label_information(self):
        records = self.records()
        before = select_records(records)
        changed = copy.deepcopy(records[::-1])
        for row in changed:
            row["label"] = "malicious irrelevant label"
            row["event_times"] = [-999]
        self.assertEqual(before, select_records(changed))
        self.assertEqual([sum(r["camera_id"] == camera for r in before)
                          for camera in ("iphone_mov", "samsung_mp4")], [3, 3])
        self.assertTrue(all(r["end_sec"] - r["start_sec"] == 8 for r in before))
        self.assertTrue(all("label" not in r and "event_times" not in r for r in before))

    def test_short_recordings_cannot_replace_missing_context(self):
        records = self.records()
        for row in records[:3]:
            row["duration_sec"] = 7.9
        with self.assertRaises(ValueError):
            select_records(records)

    def test_exact_same_middle_window_eight_frames_at_both_resolutions(self):
        record = {"path": "silent.mp4", "start_sec": 2., "end_sec": 10.}
        reader = lambda path, fps: iter((i / 2, i / 2) for i in range(28))
        resize = lambda pixels, view, size: (pixels, size)
        low = middle_images(record, 448, reader, resize)
        high = middle_images(record, 896, reader, resize)
        self.assertEqual(len(low), 8)
        self.assertEqual([r[0] for r in low], [r[0] for r in high])
        self.assertTrue(all(2 <= timestamp < 10 for timestamp, _ in low))
        self.assertEqual({size for _, size in high}, {896})

    def test_missing_frames_do_not_duplicate_the_only_observation(self):
        with self.assertRaises(ValueError):
            uniform_indices(7, 8)
        self.assertEqual(uniform_indices(5, 3), [0, 2, 4])


if __name__ == "__main__":
    unittest.main()

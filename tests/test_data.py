"""Regression tests for duplicate leakage and annotation integrity."""

import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datathon_ml.data import build_manifest, normalize_video_id, parse_timestamp


class TimestampTests(unittest.TestCase):
    def test_russian_units_and_clock(self):
        for value in ("1минута 8 секунда", "1 мин 8 сек.", "01:08", "0:01:08", 68, dt.time(0, 1, 8)):
            with self.subTest(value=value):
                self.assertEqual(parse_timestamp(value), 68.0)
        self.assertEqual(parse_timestamp("1m 8.5s"), 68.5)
        self.assertEqual(parse_timestamp("2 часа 3 минуты 4 секунды"), 7384.0)
        self.assertEqual(parse_timestamp(dt.timedelta(seconds=3.5)), 3.5)
        self.assertIsNone(parse_timestamp(None))
        self.assertIsNone(parse_timestamp(" "))

    def test_invalid_times_fail_instead_of_becoming_zero(self):
        for value in (-1, float("nan"), float("inf"), True, "1:60", "not a timestamp", "1m junk", "-2s"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_timestamp(value)

    def test_identifier_normalization(self):
        self.assertEqual(normalize_video_id(" Х – 004.MOV "), "X-4")
        self.assertEqual(normalize_video_id(r"C:\data\х-47.mov"), "X-47")
        self.assertEqual(normalize_video_id("X-2"), "X-2")
        with self.assertRaises(ValueError):
            normalize_video_id(None)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = self.root / "open"
        self.videos = self.data / "videos"
        self.videos.mkdir(parents=True)
        (self.data / "open_labels.xlsx").write_bytes(b"mock workbook")
        self.metadata = {"duration_sec": 100.0, "fps": 30.0, "width": 464, "height": 848}

    def video(self, name, content):
        path = self.videos / name
        path.write_bytes(content)
        return path

    def build(self, rows, **kwargs):
        with patch("datathon_ml.data._read_label_rows", return_value=rows), patch(
            "datathon_ml.data._probe_video", return_value=self.metadata
        ) as probe:
            records = build_manifest(self.root, **kwargs)
            return records, probe.call_count

    def test_deduplicate_merge_points_and_atomic_export(self):
        self.video("Х-47.mov", b"same positive video")
        canonical = self.video("Х-4.mov", b"same positive video")
        self.video("Х-5.mov", b"same compliant video")
        self.video("Х-48.mov", b"same compliant video")
        rows = [
            ("Х-47", "нарушение", "1минута 8 секунда", 2),
            ("X-4", "нарушение", "57 секунда", 3),
            ("Х-5", "без нарушения", None, 4),
            ("Х-48", "без нарушения", None, 5),
        ]
        output = self.root / "artifacts"
        records, probes = self.build(rows, output_dir=output)
        self.assertEqual(probes, 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["event_times"], [57.0, 68.0])
        self.assertEqual(records[0]["aliases"], ["X-4", "X-47"])
        self.assertEqual(records[0]["path"], str(canonical.resolve()))
        self.assertEqual(records[0]["group_id"], records[0]["sha256"])
        self.assertEqual(records[1]["event_times"], [])
        self.assertEqual(json.loads((output / "manifest.json").read_text(encoding="utf-8")), records)
        self.assertEqual(list(output.glob("*.tmp")), [])

    def test_conflicting_duplicate_labels_rejected(self):
        self.video("X-1.mov", b"same")
        self.video("Х-2.mov", b"same")
        rows = [("X-1", 1, 1, 2), ("X-2", 0, None, 3)]
        with self.assertRaisesRegex(ValueError, "Conflicting binary labels"):
            self.build(rows)

    def test_explicit_verified_missing_duplicate_alias(self):
        canonical = self.video("Х-4.mov", b"positive video")
        rows = [("Х-47", 1, 68, 2), ("X-4", 1, 57, 3)]
        records, probes = self.build(rows, verified_aliases={"X-47": "Х-4.mov"})
        self.assertEqual(probes, 1)
        self.assertEqual(records[0]["video_id"], "X-4")
        self.assertEqual(records[0]["path"], str(canonical.resolve()))
        self.assertEqual(records[0]["aliases"], ["X-4", "X-47"])
        self.assertEqual(records[0]["event_times"], [57.0, 68.0])

    def test_explicit_alias_conflicting_label_and_missing_target_rejected(self):
        self.video("X-4.mov", b"positive video")
        with self.assertRaisesRegex(ValueError, "Conflicting binary labels"):
            self.build([("X-4", 1, 57, 2), ("X-47", 0, None, 3)], verified_aliases={"X-47": "X-4"})
        with self.assertRaisesRegex(ValueError, "found 0"):
            self.build([("X-47", 1, 68, 2)], verified_aliases={"X-47": "X-99"})

    def test_alias_does_not_override_existing_file(self):
        self.video("X-4.mov", b"first video")
        self.video("X-47.mov", b"different video")
        records, probes = self.build(
            [("X-4", 1, 57, 2), ("X-47", 1, 68, 3)], verified_aliases={"X-47": "X-4"}
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(probes, 2)

    def test_missing_and_ambiguous_videos_rejected(self):
        with self.assertRaisesRegex(ValueError, "found 0"):
            self.build([("X-9", 0, None, 2)])
        self.video("X-1.mov", b"one")
        self.video("Х-1.mp4", b"two")
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.build([("X-1", 0, None, 2)])

    def test_bad_annotations_rejected_before_training(self):
        self.video("X-1.mov", b"video")
        for label, timestamp, message in (
            (1, None, "requires an event timestamp"),
            (0, 3, "cannot carry"),
            (1, 100, "outside video duration"),
            (1, -3, "non-negative"),
            ("unknown", None, "Unknown binary label"),
        ):
            with self.subTest(label=label, timestamp=timestamp), self.assertRaisesRegex(ValueError, message):
                self.build([("X-1", label, timestamp, 7)])

    def test_root_can_be_open_directory_and_relative_workbook(self):
        self.video("X-1.mov", b"video")
        with patch("datathon_ml.data._read_label_rows", return_value=[("X-1", 0, None, 2)]), patch(
            "datathon_ml.data._probe_video", return_value=self.metadata
        ):
            records = build_manifest(self.data, labels_path="open_labels.xlsx")
        self.assertEqual(len(records), 1)

    def test_invalid_metadata_rejected(self):
        self.video("X-1.mov", b"video")
        self.metadata["fps"] = 0
        with self.assertRaisesRegex(ValueError, "Invalid fps"):
            self.build([("X-1", 0, None, 2)])

    def test_unlabeled_video_does_not_become_negative(self):
        self.video("X-1.mov", b"labeled")
        self.video("X-9.mov", b"unlabeled")
        records, _ = self.build([("X-1", 0, None, 2)])
        self.assertEqual([record["video_id"] for record in records], ["X-1"])


if __name__ == "__main__":
    unittest.main()

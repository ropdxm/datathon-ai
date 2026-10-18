"""Native JPEG integrity, annotation-free sampling and complete time coverage."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np

from tools.prepare_visual_framepack import (PREPROCESSING, prepare_source,
                                           portable_manifest, resolve_media_inputs, write_archive)
from tools.verify_custom_labels import sha256


class VisualFramepackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"synthetic native video fixture")
        self.record = {"sha256": sha256(self.source), "path": str(self.source),
                       "label": 1, "event_times": [9.], "domain": "custom"}
        self.rgb = np.zeros((100, 50, 3), dtype=np.uint8)
        self.rgb[:, :, 0] = 255

    def samples(self):
        return iter([(0., self.rgb.copy()), (1., self.rgb.copy()), (240.5, self.rgb.copy())])

    def test_rgb_uint8_geometry_and_full_duration_without_cap(self):
        with patch("tools.prepare_visual_framepack.iter_video_frames", side_effect=lambda *a: self.samples()) as decode:
            metadata, resumed = prepare_source(self.record, self.root / "frames")
        self.assertFalse(resumed)
        self.assertEqual(metadata["timestamps"], [0., 1., 240.5])
        self.assertEqual(metadata["preprocessing"], PREPROCESSING)
        self.assertEqual(decode.call_args.args[1], 1.)
        self.assertNotIn("label", metadata)
        self.assertNotIn("event_times", metadata)
        for name in metadata["images"][0]:
            bgr = cv2.imread(str(self.root / "frames" / self.record["sha256"] / name))
            self.assertEqual(bgr.shape, (224, 224, 3))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.assertGreater(int(rgb[112, 112, 0]), 245)
            self.assertLess(int(rgb[112, 112, 2]), 10)

    def test_source_hash_checked_before_any_pixel_decode(self):
        with patch("tools.prepare_visual_framepack.iter_video_frames") as decode:
            with self.assertRaisesRegex(ValueError, "declared SHA"):
                prepare_source(dict(self.record, sha256="a" * 64), self.root / "frames")
            decode.assert_not_called()

    def test_moved_source_resolves_by_content_only_and_strips_supervision(self):
        row = dict(self.record, path=str(self.root / "missing_original_name.MOV"))
        resolved, count = resolve_media_inputs([row], self.root)
        self.assertEqual(count, 1)
        self.assertEqual(Path(resolved[0]["path"]), self.source)
        self.assertEqual(set(resolved[0]), {"sha256", "path"})
        self.assertEqual(self.record["path"], str(self.source))
        with self.assertRaisesRegex(ValueError, "no exact local"):
            resolve_media_inputs([dict(row, sha256="b" * 64)], self.root)

    def test_existing_wrong_bytes_are_not_silently_replaced_by_another_path(self):
        wrong = self.root / "wrong.mp4"
        wrong.write_bytes(b"wrong source")
        row = dict(self.record, path=str(wrong))
        resolved, count = resolve_media_inputs([row], self.root)
        self.assertEqual(count, 0)
        with self.assertRaisesRegex(ValueError, "declared SHA"):
            prepare_source(resolved[0], self.root / "frames")

    def test_labels_names_and_anchors_do_not_change_sampling_or_frame_bytes(self):
        with patch("tools.prepare_visual_framepack.iter_video_frames", side_effect=lambda *a: self.samples()):
            first, _ = prepare_source(self.record, self.root / "one")
            changed = dict(self.record, label=0, event_times=[], video_id="misleading", description="private text")
            second, _ = prepare_source(changed, self.root / "two")
        self.assertEqual(first, second)
        for names in first["images"]:
            for name in names:
                a = self.root / "one" / self.record["sha256"] / name
                b = self.root / "two" / self.record["sha256"] / name
                self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_resume_checks_integrity_and_rejects_changed_jpeg(self):
        with patch("tools.prepare_visual_framepack.iter_video_frames", side_effect=lambda *a: self.samples()) as decode:
            metadata, _ = prepare_source(self.record, self.root / "frames")
            again, resumed = prepare_source(self.record, self.root / "frames")
            self.assertTrue(resumed)
            self.assertEqual(again, metadata)
            self.assertEqual(decode.call_count, 1)
        image = self.root / "frames" / self.record["sha256"] / metadata["images"][0][0]
        image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            prepare_source(self.record, self.root / "frames")

    def test_bad_timestamps_cannot_complete_source_metadata(self):
        with patch("tools.prepare_visual_framepack.iter_video_frames",
                   return_value=iter([(1., self.rgb), (1., self.rgb)])):
            with self.assertRaisesRegex(ValueError, "increase strictly"):
                prepare_source(self.record, self.root / "frames")
        self.assertFalse((self.root / "frames" / self.record["sha256"] / "metadata.json").exists())

    def test_private_archive_separates_numeric_labels_and_strips_private_text(self):
        record = dict(self.record, source_filename="private_source_name", description="DO_NOT_INCLUDE_PRIVATE_TEXT")
        with patch("tools.prepare_visual_framepack.iter_video_frames", side_effect=lambda *a: self.samples()):
            prepare_source(record, self.root / "frames")
        split = {"train": [record["sha256"]], "validation": [], "test": []}
        target = self.root / "private.zip"
        result = write_archive([record], split, self.root / "frames", target)
        self.assertEqual(result["jpeg_images"], 6)
        self.assertEqual(result["time_samples"], 3)
        with ZipFile(target) as archive:
            self.assertIsNone(archive.testzip())
            dataset = json.loads(archive.read("dataset/manifest.json"))
            self.assertEqual(dataset[0]["label"], 1)
            self.assertNotIn("path", dataset[0])
            self.assertNotIn("source_filename", dataset[0])
            manifest = json.loads(archive.read("FRAMEPACK_MANIFEST.json"))
            self.assertTrue(manifest["contains_private_video_frames"])
            self.assertFalse(manifest["contains_audio"])
            for name in archive.namelist():
                if name.endswith(".json"):
                    self.assertNotIn(b"DO_NOT_INCLUDE_PRIVATE_TEXT", archive.read(name))
                    self.assertNotIn(b"private_source_name", archive.read(name))

    def test_real_native_video_sampling_preserves_last_valid_frame(self):
        source = self.root / "native.avi"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10., (32, 24))
        self.assertTrue(writer.isOpened())
        for index in range(20):
            writer.write(np.full((24, 32, 3), index * 10, dtype=np.uint8))
        writer.release()
        record = {"sha256": sha256(source), "path": str(source)}
        metadata, _ = prepare_source(record, self.root / "native_frames")
        self.assertEqual(len(metadata["timestamps"]), 3)
        self.assertGreaterEqual(metadata["timestamps"][-1], 1.8)
        self.assertEqual(len(metadata["images"][-1]), 2)


if __name__ == "__main__":
    unittest.main()

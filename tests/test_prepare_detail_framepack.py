"""Exact timeline, native detail geometry, privacy and archive integrity checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np

from tools.prepare_detail_framepack import PREPROCESSING, prepare_source, write_archive
from tools.prepare_visual_framepack import prepare_source as prepare_reference
from tools.verify_custom_labels import sha256


class DetailFramepackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"synthetic exact source bytes")
        self.row = {"sha256": sha256(self.source), "path": str(self.source), "label": 1, "event_times": [7.]}
        self.rgb = np.zeros((600, 300, 3), np.uint8)
        self.rgb[:, :, 0] = 255
        self.samples = [(0., self.rgb), (1., self.rgb), (240.5, self.rgb)]
        with patch("tools.prepare_visual_framepack.iter_video_frames", side_effect=lambda *a: iter(self.samples)):
            prepare_reference(self.row, self.root / "reference")

    def build(self, folder="detail", samples=None, row=None):
        with patch("tools.prepare_detail_framepack.iter_video_frames", side_effect=lambda *a: iter(self.samples if samples is None else samples)):
            return prepare_source(self.row if row is None else row, self.root / folder, self.root / "reference")

    def test_native448_geometry_rgb_exact_reference_times_and_original224_preservation(self):
        ref = self.root / "reference" / self.row["sha256"]
        before = {p.name: sha256(p) for p in ref.iterdir()}
        metadata, resumed = self.build()
        self.assertFalse(resumed)
        self.assertEqual(metadata["timestamps"], [0., 1., 240.5])
        self.assertEqual(metadata["preprocessing"], PREPROCESSING)
        self.assertEqual(metadata["jpeg_quality"], 90)
        self.assertNotIn("label", metadata)
        self.assertNotIn("event_times", metadata)
        image = cv2.imread(str(self.root / "detail" / self.row["sha256"] / metadata["images"][0][0]))
        self.assertEqual(image.shape, (448, 448, 3))
        self.assertGreater(int(image[224, 224, 2]), 245)
        self.assertLess(int(image[224, 224, 0]), 10)
        self.assertEqual(before, {p.name: sha256(p) for p in ref.iterdir()})

    def test_reference_timestamp_drift_or_truncation_cannot_complete_metadata(self):
        for folder, samples in (("drift", [(0., self.rgb), (1.001, self.rgb)]),
                                ("short", self.samples[:2])):
            with self.assertRaisesRegex(ValueError, "timestamp|complete"):
                self.build(folder, samples)
            self.assertFalse((self.root / folder / self.row["sha256"] / "metadata.json").exists())

    def test_wrong_source_or_tampered_reference_rejects_before_decode(self):
        with patch("tools.prepare_detail_framepack.iter_video_frames") as decode:
            with self.assertRaisesRegex(ValueError, "declared SHA"):
                prepare_source(dict(self.row, sha256="f" * 64), self.root / "detail", self.root / "reference")
            decode.assert_not_called()
        ref = self.root / "reference" / self.row["sha256"] / "metadata.json"
        payload = json.loads(ref.read_text()); payload["timestamps"][-1] = 241.
        ref.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.build()

    def test_supervision_changes_do_not_change_detail_pixels_or_sampling(self):
        first, _ = self.build("one")
        second, _ = self.build("two", row=dict(self.row, label=0, event_times=[], description="never_read"))
        self.assertEqual(first, second)
        for names in first["images"]:
            for name in names:
                self.assertEqual((self.root / "one" / self.row["sha256"] / name).read_bytes(),
                                 (self.root / "two" / self.row["sha256"] / name).read_bytes())

    def test_resume_verifies_every_image_and_rejects_mutation(self):
        metadata, _ = self.build()
        with patch("tools.prepare_detail_framepack.iter_video_frames") as decode:
            _, resumed = prepare_source(self.row, self.root / "detail", self.root / "reference")
            self.assertTrue(resumed); decode.assert_not_called()
        image = self.root / "detail" / self.row["sha256"] / metadata["images"][0][0]
        image.write_bytes(b"mutated")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            self.build()

    def test_private_archive_has_complete_checksums_and_no_source_names_or_descriptions(self):
        self.build()
        row = dict(self.row, description="DO_NOT_EXPORT", source_filename="PRIVATE_NAME")
        result = write_archive([row], {"train": [row["sha256"]]}, self.root / "detail",
                               self.root / "reference", self.root / "private.zip")
        self.assertEqual(result["jpeg_images"], 6)
        with ZipFile(self.root / "private.zip") as archive:
            self.assertIsNone(archive.testzip())
            inventory = json.loads(archive.read("DETAIL_FRAMEPACK_MANIFEST.json"))
            self.assertTrue(inventory["timestamps_equal_frozen_224"])
            self.assertEqual(inventory["image_size"], 448)
            self.assertFalse(inventory["contains_audio"])
            self.assertEqual(set(inventory["members_sha256"]), set(archive.namelist()) - {"DETAIL_FRAMEPACK_MANIFEST.json"})
            for name in archive.namelist():
                if name.endswith(".json"):
                    self.assertNotIn(b"DO_NOT_EXPORT", archive.read(name))
                    self.assertNotIn(b"PRIVATE_NAME", archive.read(name))

    def test_real_native_source_has_same_last_valid_timestamp_at_both_resolutions(self):
        native = self.root / "native.avi"
        writer = cv2.VideoWriter(str(native), cv2.VideoWriter_fourcc(*"MJPG"), 10., (32, 24))
        self.assertTrue(writer.isOpened())
        for index in range(20): writer.write(np.full((24, 32, 3), index * 10, np.uint8))
        writer.release()
        row = {"sha256": sha256(native), "path": str(native)}
        reference, _ = prepare_reference(row, self.root / "native_reference")
        detail, _ = prepare_source(row, self.root / "native_detail", self.root / "native_reference")
        self.assertEqual(detail["timestamps"], reference["timestamps"])
        self.assertGreaterEqual(detail["timestamps"][-1], 1.8)


if __name__ == "__main__": unittest.main()

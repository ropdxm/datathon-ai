"""Prevent audio, stale XLSX and unverified-label leakage in custom data."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.build_custom_manifest import (CUSTOM_GROUP, TIME_MAPPING,
                                        audit_custom_dataset, build_custom_manifest)


class CustomManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        self.prepared = self.root / "prepared"
        self.source.mkdir()
        self.prepared.mkdir()
        self.manifest_path = self.root / "preparation_manifest.json"
        self.preparation = {"complete": True, "source_dir": str(self.source),
                            "output_dir": str(self.prepared), "videos": []}
        for i, suffix in enumerate(("MOV", "mp4")):
            name = f"custom{i}.{suffix}"
            source_bytes, prepared_bytes = f"source{i}".encode(), f"prepared{i}".encode()
            (self.source / name).write_bytes(source_bytes)
            (self.prepared / f"custom{i}.mp4").write_bytes(prepared_bytes)
            self.preparation["videos"].append({
                "source_relative_path": name, "output_relative_path": f"custom{i}.mp4",
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "output_sha256": hashlib.sha256(prepared_bytes).hexdigest(),
                "source": {"duration_seconds": 10.},
                "output": {"duration_seconds": 10., "width": 1280, "height": 720,
                           "fps": "15/1", "audio_streams": 0, "stream_types": ["video"]},
                "verification": {"verified": True, "annotation_time_mapping": TIME_MAPPING,
                                 "duration_error_seconds": 0., "duration_tolerance_seconds": .15}})
        self.save()

    def save(self):
        self.manifest_path.write_text(json.dumps(self.preparation), encoding="utf-8")

    def proposals(self, entries):
        path = self.root / "proposals.json"
        path.write_text(json.dumps({"records": entries}), encoding="utf-8")
        return path

    def proposal(self, index=0, **overrides):
        row = self.preparation["videos"][index]
        return {"source_sha256": row["source_sha256"],
                "source_filename": row["source_relative_path"],
                "label": "violation", "event_times": [3.],
                "label_status": "machine_proposal", **overrides}

    def test_no_automatic_labels_and_all_views_grouped(self):
        records = build_custom_manifest(self.manifest_path)
        self.assertEqual(len(records), 2)
        self.assertEqual({record["group_id"] for record in records}, {CUSTOM_GROUP})
        self.assertEqual({record["camera_id"] for record in records}, {"iphone_mov", "samsung_mp4"})
        for record in records:
            self.assertIsNone(record["label"])
            self.assertEqual(record["event_times"], [])
            self.assertEqual(record["label_status"], "unlabeled")
            self.assertFalse(record["evaluation_ground_truth_eligible"])
            self.assertFalse(record["eligible_for_supervised_training"])
            self.assertFalse(record["audio_used"])
            self.assertNotEqual(record["sha256"], record["source_sha256"])
            self.assertEqual(record["sha256"], record["prepared_sha256"])

    def test_proposals_do_not_become_ground_truth_by_default(self):
        path = self.proposals([self.proposal()])
        record = build_custom_manifest(self.manifest_path, path)[0]
        self.assertIsNone(record["label"])
        self.assertEqual(record["proposed_label"], "violation")
        self.assertEqual(record["proposed_event_times"], [3.])
        self.assertEqual(record["annotation_status"], "machine_proposal_requires_human_review")
        self.assertEqual(record["label_status"], "unlabeled")

    def test_explicit_machine_training_keeps_evaluation_ineligible(self):
        path = self.proposals([self.proposal(), self.proposal(1, label="no violation", event_times=[])])
        records = build_custom_manifest(self.manifest_path, path, allow_machine_labels=True)
        self.assertEqual([record["label"] for record in records], [1, 0])
        for record in records:
            self.assertEqual(record["label_status"], "machine_proposal")
            self.assertEqual(record["training_supervision"], "machine_pseudo_label")
            self.assertTrue(record["eligible_for_supervised_training"])
            self.assertFalse(record["evaluation_ground_truth_eligible"])
            self.assertFalse(record["human_verified"])

    def test_unknown_proposals_excluded_even_with_opt_in(self):
        path = self.proposals([self.proposal(label=None, event_times=[])])
        records = build_custom_manifest(self.manifest_path, path, True)
        self.assertTrue(all(record["label"] is None for record in records))
        self.assertTrue(all(not record["eligible_for_supervised_training"] for record in records))

    def test_original_ids_and_conflicting_identifiers_rejected(self):
        invalid = [{"source_filename": "X-1.mov", "label": "violation", "event_times": [3.]},
                   self.proposal(source_filename="different.mp4")]
        for entry in invalid:
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                build_custom_manifest(self.manifest_path, self.proposals([entry]))

    def test_invalid_points_and_labels_rejected(self):
        invalid = [self.proposal(event_times=[]), self.proposal(event_times=[10.]),
                   self.proposal(event_times=[True]), self.proposal(event_times=[-1]),
                   self.proposal(event_times=[float("nan")]),
                   self.proposal(label="no violation"), self.proposal(label="normal"),
                   self.proposal(label=None), self.proposal(label_status="human_verified")]
        for entry in invalid:
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                build_custom_manifest(self.manifest_path, self.proposals([entry]))

    def test_duplicate_proposals_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate proposal"):
            build_custom_manifest(self.manifest_path, self.proposals([self.proposal(), self.proposal()]))

    def test_changed_video_bytes_rejected(self):
        for directory, name in ((self.source, "custom0.MOV"), (self.prepared, "custom0.mp4")):
            path = directory / name
            original = path.read_bytes()
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed source/prepared"):
                build_custom_manifest(self.manifest_path)
            path.write_bytes(original)

    def test_audio_and_incomplete_or_retimed_preparation_rejected(self):
        original = copy.deepcopy(self.preparation)
        mutations = [lambda d: d.update(complete=False),
                     lambda d: d["videos"][0]["output"].update(audio_streams=1),
                     lambda d: d["videos"][0]["output"].update(stream_types=["video", "audio"]),
                     lambda d: d["videos"][0]["verification"].update(verified=False),
                     lambda d: d["videos"][0]["verification"].update(annotation_time_mapping="retimed"),
                     lambda d: d["videos"][0]["verification"].update(duration_error_seconds=.2),
                     lambda d: d["videos"][0]["verification"].update(duration_error_seconds=float("nan")),
                     lambda d: d["videos"][0]["output"].update(duration_seconds=11.)]
        for mutate in mutations:
            self.preparation = copy.deepcopy(original)
            mutate(self.preparation)
            self.save()
            with self.assertRaises(ValueError):
                build_custom_manifest(self.manifest_path)

    def test_path_escape_and_duplicate_sources_rejected(self):
        self.preparation["videos"][0]["source_relative_path"] = "../escape.mov"
        self.save()
        with self.assertRaisesRegex(ValueError, "remain inside"):
            build_custom_manifest(self.manifest_path)
        self.preparation["videos"][0]["source_relative_path"] = "custom0.MOV"
        self.preparation["videos"].append(copy.deepcopy(self.preparation["videos"][0]))
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            build_custom_manifest(self.manifest_path)

    def test_copied_workbook_detected_without_label_attachment(self):
        records = build_custom_manifest(self.manifest_path)
        rows = [("X-1", "violation", 3, 2)]
        with patch("tools.build_custom_manifest._read_label_rows", return_value=rows):
            audit = audit_custom_dataset(records, "custom.xlsx", "original.xlsx")
        self.assertTrue(audit["annotation_workbook"]["first_three_columns_equal_original_workbook"])
        self.assertEqual(audit["annotation_workbook"]["matching_custom_ids"], 0)
        self.assertEqual(audit["annotation_workbook"]["custom_files_without_matching_label"], 2)
        self.assertTrue(all(record["label"] is None for record in records))

    def test_explicit_pseudo_labels_require_proposal_input(self):
        with self.assertRaisesRegex(ValueError, "requires --proposals"):
            build_custom_manifest(self.manifest_path, allow_machine_labels=True)

    def test_missing_original_corpus_cannot_report_zero_overlap(self):
        records = build_custom_manifest(self.manifest_path)
        for root in (self.root / "missing", self.root / "empty"):
            if root.name == "empty":
                root.mkdir()
            with self.subTest(root=root), self.assertRaises(ValueError):
                audit_custom_dataset(records, original_video_root=root)


if __name__ == "__main__":
    unittest.main()

"""Human annotation integrity: no pseudo-label or wrong-source promotion."""

import copy
import csv
import tempfile
from pathlib import Path
import unittest

from tools.verify_custom_labels import (CUSTOM_GROUP, GROUPING_BASIS,
                                        ingest_review_rows, parse_event_times,
                                        read_review, write_review_template)


class VerifyCustomLabelsTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "source_sha256": "a" * 64, "source_filename": "camera.MOV",
            "path": "/private/silent.mp4", "duration_sec": 12.,
            "source_duration_sec": 11.99, "group_id": CUSTOM_GROUP,
            "pair_id": None, "label": None, "event_times": [],
        }
        self.row = {
            "source_sha256": "a" * 64, "source_filename": "camera.MOV",
            "label": "violation", "event_times": "[1.5, 10]",
            "review_status": "human_verified", "full_video_reviewed": "yes",
            "reviewer": "annotator_1", "reviewed_at_utc": "2026-01-01T00:00:00Z",
        }

    def test_verified_targets_keep_scene_group_and_do_not_mutate_input(self):
        before = copy.deepcopy(self.record)
        records, audit = ingest_review_rows([self.record], [self.row])
        self.assertEqual(self.record, before)
        self.assertEqual(records[0]["event_times"], [1.5, 10.])
        self.assertEqual(records[0]["group_id"], CUSTOM_GROUP)
        self.assertTrue(records[0]["group_verified"])
        self.assertEqual(records[0]["grouping_basis"], GROUPING_BASIS)
        self.assertFalse(records[0]["individual_scene_pairing_verified"])
        self.assertTrue(records[0]["event_times_verified"])
        self.assertTrue(records[0]["human_verified"])
        self.assertEqual(audit["verified_timestamps"], 2)

    def test_filled_label_without_attestation_is_not_groundtruth(self):
        for status in ("unreviewed", "uncertain", ""):
            row = dict(self.row, review_status=status)
            verified, audit = ingest_review_rows([self.record], [row])
            self.assertEqual(verified, [])
            self.assertEqual(audit["filled_labels_without_human_verification"], 1)

    def test_exact_sha_filename_join_and_duplicate_rejection(self):
        for change in ({"source_sha256": "b" * 64}, {"source_filename": "other.MOV"}):
            with self.assertRaises(ValueError):
                ingest_review_rows([self.record], [dict(self.row, **change)])
        with self.assertRaisesRegex(ValueError, "Duplicate source row"):
            ingest_review_rows([self.record], [self.row, self.row])

    def test_positive_and_negative_timestamp_consistency(self):
        for change in ({"label": "violation", "event_times": "[]"},
                       {"label": "no_violation", "event_times": "[1]"},
                       {"label": "normal", "event_times": "[]"}):
            with self.assertRaises(ValueError):
                ingest_review_rows([self.record], [dict(self.row, **change)])
        result, _ = ingest_review_rows([self.record],
                                      [dict(self.row, label="no_violation", event_times="[]")])
        self.assertEqual(result[0]["label"], 0)
        self.assertEqual(result[0]["event_times"], [])
        self.assertFalse(result[0]["event_times_verified"])

    def test_group_comes_from_shared_batch_not_labels_or_proposals(self):
        other_record = dict(self.record, source_filename="other.mp4", source_sha256="b" * 64)
        other_row = dict(self.row, source_filename="other.mp4", source_sha256="b" * 64,
                         label="no_violation", event_times="[]", group_id="label_derived_group")
        result, _ = ingest_review_rows([self.record, other_record], [self.row, other_row])
        self.assertEqual({row["group_id"] for row in result}, {CUSTOM_GROUP})
        self.assertTrue(all(row["grouping_basis"] == GROUPING_BASIS for row in result))
        changed = dict(self.record, group_id="individual_guessed_pair")
        with self.assertRaisesRegex(ValueError, "conservative shared"):
            ingest_review_rows([changed], [self.row])
        with self.assertRaisesRegex(ValueError, "machine proposals"):
            ingest_review_rows([self.record], [dict(self.row, review_status="machine_proposal",
                                                   group_id=CUSTOM_GROUP)])

    def test_timestamp_bounds_use_shorter_source_not_only_prepared(self):
        self.assertEqual(parse_event_times("00:01.5; 10", 11.99), [1.5, 10.])
        for points in ("[11.99]", "[-1]", "[true]", "[NaN]", "[1, 1]", "[null]"):
            with self.assertRaises(ValueError):
                ingest_review_rows([self.record], [dict(self.row, event_times=points)])

    def test_machine_or_incomplete_attestation_rejected(self):
        changes = ({"review_status": "machine_proposal"}, {"full_video_reviewed": "no"},
                   {"reviewer": "model"}, {"reviewer": ""},
                   {"reviewed_at_utc": "2026-01-01T00:00:00"},
                   {"reviewed_at_utc": "2026-01-01T00:00:00+05:00"},
                   {"reviewed_at_utc": "2999-01-01T00:00:00Z"})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                ingest_review_rows([self.record], [dict(self.row, **change)])

    def test_template_stays_blank_and_never_overwrites_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            write_review_template([self.record], path)
            rows = read_review(path)
            self.assertEqual(rows[0]["label"], "")
            self.assertEqual(rows[0]["review_status"], "unreviewed")
            self.assertEqual(ingest_review_rows([self.record], rows)[0], [])
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_review_template([self.record], path)
            self.assertEqual(path.read_bytes(), before)

    def test_review_csv_requires_provenance_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            path.write_text("source_filename,label,event_times\ncamera.MOV,violation,[1]\n")
            with self.assertRaisesRegex(ValueError, "missing required"):
                read_review(path)


if __name__ == "__main__":
    unittest.main()

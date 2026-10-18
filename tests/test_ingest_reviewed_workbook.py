"""Attested workbook identity, joins and timestamp integrity on numeric fixtures."""
from pathlib import Path
import tempfile
import unittest

from tools.build_custom_manifest import CUSTOM_GROUP
from tools.ingest_reviewed_workbook import reviewed_rows
from tools.verify_custom_labels import sha256


class ReviewedWorkbookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source, self.prepared = root / "camera.MOV", root / "camera__mov__hash.mp4"
        self.source.write_bytes(b"original")
        self.prepared.write_bytes(b"silent derivative")
        self.record = {
            "source_filename": self.source.name, "source_path": str(self.source),
            "source_sha256": sha256(self.source), "path": str(self.prepared),
            "prepared_sha256": sha256(self.prepared), "sha256": sha256(self.prepared),
            "duration_sec": 12., "source_duration_sec": 11.99,
            "group_id": CUSTOM_GROUP, "audio_streams": 0, "label": None,
        }
        self.arguments = dict(workbook_sha256="a" * 64, attested_sha256="a" * 64,
            attestation_date="2026-10-18", attestation_source="human_user_chat")

    def run_rows(self, rows, records=None):
        return reviewed_rows(records or [self.record], rows, **self.arguments)

    def test_direct_user_attestation_is_sufficient_without_csv_fields(self):
        result, audit, excluded = self.run_rows([("camera.MOV", "violation", 9, 6)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["event_times"], [9.])
        self.assertTrue(result[0]["event_times_verified"])
        self.assertTrue(result[0]["group_verified"])
        self.assertEqual(result[0]["group_id"], CUSTOM_GROUP)
        self.assertFalse(result[0]["individual_scene_pairing_verified"])
        self.assertEqual(result[0]["reviewed_workbook_sha256"], "a" * 64)
        self.assertTrue(audit["source_and_silent_derivative_hashes_verified"])
        self.assertEqual(excluded, [])

    def test_wrong_attested_bytes_fail_closed(self):
        self.arguments["attested_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "exact bytes"):
            self.run_rows([("camera.MOV", "violation", 9, 6)])

    def test_prepared_filename_alias_resolves_to_original_identity(self):
        result, _, _ = self.run_rows([(self.prepared.name, "no_violation", None, 6)])
        self.assertEqual(result[0]["source_sha256"], self.record["source_sha256"])
        self.assertEqual(result[0]["label"], 0)
        self.assertEqual(result[0]["event_times"], [])

    def test_blank_ambiguous_unknown_and_duplicate_rows_are_excluded(self):
        other = dict(self.record, source_sha256="b" * 64)
        result, audit, _ = self.run_rows([("camera.MOV", "violation", 3, 6)], [self.record, other])
        self.assertEqual(result, [])
        self.assertEqual(audit["exclusion_reason_counts"], {"ambiguous_file_id": 1})
        result, audit, _ = self.run_rows([("unknown.MOV", "violation", 3, 6),
                                         ("camera.MOV", None, None, 7)])
        self.assertEqual(result, [])
        self.assertEqual(audit["exclusion_reason_counts"], {"unknown_file_id": 1, "missing_label": 1})
        result, audit, _ = self.run_rows([("camera.MOV", "violation", 3, 6),
                                         ("camera.MOV", "no violation", None, 7)])
        self.assertEqual(result, [])
        self.assertEqual(audit["exclusion_reason_counts"], {"duplicate_source_rows": 2})

    def test_bounds_and_missing_points_never_invented(self):
        for point in (11.99, 12, -1, float("nan"), True):
            result, audit, _ = self.run_rows([("camera.MOV", "violation", point, 6)])
            self.assertEqual(result, [])
            self.assertEqual(audit["exclusion_reason_counts"], {"invalid_or_out_of_bounds_timestamp": 1})
        result, audit, _ = self.run_rows([("camera.MOV", "violation", None, 6)])
        self.assertEqual(result, [])
        self.assertEqual(audit["exclusion_reason_counts"], {"positive_without_timestamp": 1})

    def test_missing_or_changed_prepared_data_is_not_used(self):
        self.prepared.write_bytes(b"changed")
        result, audit, _ = self.run_rows([("camera.MOV", "violation", 9, 6)])
        self.assertEqual(result, [])
        self.assertEqual(audit["exclusion_reason_counts"], {"prepared_hash_changed": 1})
        self.prepared.unlink()
        result, audit, _ = self.run_rows([("camera.MOV", "violation", 9, 6)])
        self.assertEqual(result, [])
        self.assertEqual(audit["exclusion_reason_counts"], {"missing_silent_prepared_file": 1})

    def test_omitted_missing_recordings_do_not_block_reviewed_subset(self):
        other = dict(self.record, source_filename="omitted.MOV", source_sha256="b" * 64,
                     source_path="/missing/original.MOV", path="/missing/silent.mp4")
        result, audit, _ = self.run_rows([("camera.MOV", "no violation", None, 6)], [self.record, other])
        self.assertEqual(len(result), 1)
        self.assertEqual(audit["preparation_sources_without_accepted_label"], 1)


if __name__ == "__main__":
    unittest.main()

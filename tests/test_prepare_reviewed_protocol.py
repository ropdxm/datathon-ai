"""Fixed split integrity with same-source overlaps and mixed camera domains."""
import copy
import unittest

from tools.prepare_reviewed_protocol import fixed_split, validate_split


def fixture():
    records = []
    for index in range(40):
        key = f"{index + 1:064x}"
        records.append({"sha256": key, "source_sha256": key,
                        "label": int(index < 21), "domain": "original", "group_id": key})
    for index in range(18):
        key = f"{index + 101:064x}"
        # Two positives and one normal belong to one overlapping source group.
        overlap = index in (0, 1, 8)
        records.append({"sha256": key, "source_sha256": f"{index + 201:064x}",
                        "label": int(index < 8), "domain": "custom", "source_family": "mov",
                        "group_id": "overlap3" if overlap else f"native_{index}"})
    return records


class ReviewedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.records = fixture()
        self.split = fixed_split(self.records, "a" * 64)

    def test_exact58_mixed_domain_class_coverage_and_group_separation(self):
        self.assertEqual([len(self.split[name]) for name in ("train", "validation", "test")], [34, 12, 12])
        by_sha = {row["sha256"]: row for row in self.records}
        groups = []
        for name in ("train", "validation", "test"):
            rows = [by_sha[key] for key in self.split[name]]
            groups.append({row["group_id"] for row in rows})
            for domain in ("original", "custom"):
                self.assertEqual({row["label"] for row in rows if row["domain"] == domain}, {0, 1})
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(sum("overlap3" in group for group in groups), 1)
        self.assertFalse(self.split["protocol"]["human_verified_scene_groups"])
        self.assertFalse(self.split["protocol"]["independent_new_scene_test"])

    def test_order_invariant_and_ignores_any_performance_or_feature_fields(self):
        records = copy.deepcopy(self.records[::-1])
        for row in records:
            row.update(score=1e6, model_prediction=1 - row["label"], features=[999., -999.])
        self.assertEqual(fixed_split(records, "a" * 64), self.split)

    def test_no_alternate_seed_or_unbound_group_provenance(self):
        with self.assertRaisesRegex(ValueError, "single fixed seed"):
            fixed_split(self.records, "a" * 64, seed=43)
        with self.assertRaisesRegex(ValueError, "exact SHA"):
            fixed_split(self.records, "not-a-hash")

    def test_opposite_camera_cannot_enter_any_partition(self):
        self.records[-1]["source_family"] = "mp4"
        with self.assertRaisesRegex(ValueError, "excluded globally"):
            fixed_split(self.records, "a" * 64)

    def test_duplicate_sources_and_missing_domain_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Repeated recording"):
            fixed_split(self.records + [self.records[0]], "a" * 64)
        with self.assertRaisesRegex(ValueError, "Both original and custom"):
            fixed_split(self.records[:40], "a" * 64)

    def test_tampered_lineage_or_overlap_cannot_cross_partitions(self):
        records = copy.deepcopy(self.records)
        by_sha = {row["sha256"]: row for row in records}
        a, b = self.split["train"][0], self.split["test"][0]
        by_sha[b]["parent_sha256"] = a
        with self.assertRaisesRegex(ValueError, "lineage|overlap group"):
            validate_split(records, self.split)
        by_sha[b].pop("parent_sha256")
        by_sha[b]["group_id"] = by_sha[a]["group_id"]
        with self.assertRaisesRegex(ValueError, "overlap group"):
            validate_split(records, self.split)

    def test_split_coverage_and_repeated_assignment_are_checked(self):
        split = copy.deepcopy(self.split)
        split["train"].append(split["test"][0])
        with self.assertRaises(ValueError):
            validate_split(self.records, split)


if __name__ == "__main__":
    unittest.main()

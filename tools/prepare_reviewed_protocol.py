"""Freeze a camera-exclusive, mixed-domain development protocol before fitting.

Only identities, human labels, supplied timestamps and native overlap grouping
are used. No features, model predictions or performance metrics select splits.
This is within-session development, not independent new-scene validation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datathon_ml.data import build_manifest
from tools.build_custom_manifest import CUSTOM_GROUP, write_json
from tools.verify_custom_labels import sha256


PROTOCOL_TYPE = "camera_exclusive_source_overlap_development"
GROUPING_BASIS = "native_duplicate_overlap_prevention_not_human_scene_identity"
SEED = 42


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _domain_allocation(records, seed):
    """Exact nearest 20%-per-class validation/test allocation with whole groups.

    Dynamic programming resolves mixed-class overlap groups. SHA-seeded group
    order is the only tie-break; there is one predeclared seed and no resampling
    based on model results. Training receives the remaining records.
    """
    groups = defaultdict(list)
    for row in records:
        groups[row["group_id"]].append(row)
    counts = Counter(row["label"] for row in records)
    targets = {label: max(1, int(counts[label] * .2 + .5)) for label in (0, 1)}
    if any(counts[label] <= 2 * targets[label] for label in (0, 1)):
        raise ValueError("Every domain/class needs records in train, validation and test")
    ordered = sorted(groups, key=lambda group: digest([seed, records[0]["domain"], group]))
    # State: validation normal/positive, test normal/positive -> chosen group bins.
    states = {(0, 0, 0, 0): ()}
    goal = (targets[0], targets[1], targets[0], targets[1])
    for group in ordered:
        normal = sum(row["label"] == 0 for row in groups[group])
        positive = len(groups[group]) - normal
        next_states = {}
        for state, decisions in states.items():
            for split in ("train", "validation", "test"):
                new = list(state)
                if split == "validation":
                    new[0] += normal
                    new[1] += positive
                elif split == "test":
                    new[2] += normal
                    new[3] += positive
                key = tuple(new)
                if all(value <= limit for value, limit in zip(key, goal)):
                    next_states.setdefault(key, decisions + (split,))
        states = next_states
    if goal not in states:
        raise ValueError("Whole overlap groups cannot satisfy the declared domain/class targets; do not choose another seed after viewing performance")
    result = {split: [] for split in ("train", "validation", "test")}
    for group, split in zip(ordered, states[goal]):
        result[split].extend(row["sha256"] for row in groups[group])
    return result


def fixed_split(records, grouping_map_sha256, seed=SEED):
    if seed != SEED:
        raise ValueError("This protocol declares a single fixed seed of 42")
    if (not isinstance(grouping_map_sha256, str) or len(grouping_map_sha256) != 64
            or any(char not in "0123456789abcdef" for char in grouping_map_sha256)):
        raise ValueError("Native grouping provenance needs an exact SHA-256")
    if len({row["sha256"] for row in records}) != len(records):
        raise ValueError("Repeated recording content in mixed manifest")
    if {row["domain"] for row in records} != {"original", "custom"}:
        raise ValueError("Both original and custom domains are required")
    groups = defaultdict(set)
    for row in records:
        if row["domain"] not in {"original", "custom"}:
            raise ValueError("Unknown evaluation domain")
        if row["domain"] == "custom" and row.get("source_family") != "mov":
            raise ValueError("Opposite custom camera family must be excluded globally")
        groups[row["group_id"]].add(row["domain"])
    if any(len(domains) != 1 for domains in groups.values()):
        raise ValueError("A declared overlap group crosses domains; resolve lineage before splitting")
    result = {split: [] for split in ("train", "validation", "test")}
    for domain in ("original", "custom"):
        current = _domain_allocation([row for row in records if row["domain"] == domain], seed)
        for split in result:
            result[split].extend(current[split])
    for values in result.values():
        values.sort()
    protocol = {
        "type": PROTOCOL_TYPE, "seed": SEED, "selected_camera": ".mov",
        "opposite_camera_excluded_globally": True,
        "grouping_basis": GROUPING_BASIS,
        "grouping_map_sha256": grouping_map_sha256,
        "family_selection_basis": "largest_usable_label_coverage_before_model_scores",
        "allocation": "exact_nearest_20_percent_per_domain_class_with_whole_native_overlap_groups",
        "full_session_grouping_enforced": False,
        "human_verified_scene_groups": False,
        "independent_new_scene_test": False,
        "test_locked_before_candidate_fitting": True,
        "test_metrics_viewed": False,
        "selection_policy": "Fit candidates on train, select and freeze winner using validation only, evaluate test once after winner freeze.",
        "limitations": [
            "All custom MOV views come from one filming batch. Same-session evaluation does not establish new-session or unseen-scene generalization.",
            "Native source-overlap groups prevent known duplicate/overlap leakage; they do not certify distinct physical scenes.",
            "Original recordings and their earlier model results were already repeatedly examined as development data.",
            "Opposite-camera custom MP4 recordings are excluded from training, validation, test and model selection.",
        ],
    }
    result["protocol"] = protocol
    result["assignment_sha256"] = digest({key: result[key] for key in ("train", "validation", "test")})
    validate_split(records, result)
    return result


def validate_split(records, split):
    by_sha = {row["sha256"]: row for row in records}
    owner, lineage = {}, {}
    for name in ("train", "validation", "test"):
        rows = [by_sha[key] for key in split[name]]
        if len(set(split[name])) != len(split[name]):
            raise ValueError("Duplicate assignment inside a split")
        for domain in ("original", "custom"):
            if {row["label"] for row in rows if row["domain"] == domain} != {0, 1}:
                raise ValueError("Each split must contain both classes from both domains")
        for row in rows:
            key = row["sha256"]
            if key in owner:
                raise ValueError("Recording assigned to more than one split")
            owner[key] = name
            for identifier in {row["group_id"], row["sha256"], row.get("source_sha256", row["sha256"]),
                               row.get("parent_sha256", row["sha256"]), *row.get("lineage_sha256s", [])}:
                if identifier in lineage and lineage[identifier] != name:
                    raise ValueError("Source content or overlap group crosses splits")
                lineage[identifier] = name
    if set(owner) != set(by_sha):
        raise ValueError("Split does not cover exactly the selected mixed dataset")


def prepare_records(data_root, labels_path, aliases_path, reviewed_path, groups_path, reference_path):
    aliases = json.loads(Path(aliases_path).read_text())
    originals = build_manifest(data_root, labels_path=labels_path, verified_aliases=aliases)
    # Read only hash/target fields from the historical identity allowlist.
    reference = json.loads(Path(reference_path).read_text())
    expected = {row["sha256"]: row["label"] for row in reference["predictions"]}
    if len(originals) != 40 or {row["sha256"]: row["label"] for row in originals} != expected:
        raise ValueError("Fresh original sources/targets differ from the frozen 40-record identity allowlist")
    for row in originals:
        row.update(domain="original", dataset="original", label_status="original_provided",
                   source_sha256=row["sha256"], source_path=row["path"],
                   group_verified=False, grouping_basis="original_exact_content_hash_recording_group",
                   previously_examined_development=True, audio_used=False)
    reviewed = json.loads(Path(reviewed_path).read_text())
    camera_counts = Counter(Path(row["source_filename"]).suffix.lower() for row in reviewed)
    if camera_counts.get(".mov", 0) < camera_counts.get(".mp4", 0):
        raise ValueError("Predeclared MOV selection must match largest reviewed camera coverage")
    groups = json.loads(Path(groups_path).read_text())
    mapping = groups["camera_exclusive_group_by_source"]
    custom = []
    for source in reviewed:
        if Path(source["source_filename"]).suffix.lower() != ".mov":
            continue
        row = dict(source)
        if row.get("label_status") != "human_verified" or row.get("evaluation_ground_truth_eligible") is not True:
            raise ValueError("Only human-attested custom targets are allowed")
        if row["label"] == 1 and row.get("event_times_verified") is not True:
            raise ValueError("Reviewed custom positives need verified supplied timestamps")
        if row["source_sha256"] not in mapping:
            raise ValueError("Selected custom source lacks native overlap grouping")
        for path, expected_hash in ((row["path"], row["sha256"]),
                                    (row["source_path"], row["source_sha256"])):
            if not Path(path).is_file() or sha256(path) != expected_hash:
                raise ValueError("Selected custom source or silent derivative changed")
        row.update(domain="custom", source_family="mov", group_id=mapping[row["source_sha256"]],
                   group_verified=False, grouping_basis=GROUPING_BASIS,
                   full_session_group_id=CUSTOM_GROUP, full_session_grouping_enforced=False,
                   individual_scene_pairing_verified=False, opposite_camera_excluded_globally=True)
        custom.append(row)
    if len(custom) != 18 or sum(row["label"] for row in custom) != 8:
        raise ValueError("Declared reviewed camera subset must contain 18 MOV recordings with 8 positives")
    return sorted(originals + custom, key=lambda row: row["sha256"]), camera_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("open"))
    parser.add_argument("--labels-path", type=Path, default=Path("open_labels.xlsx"))
    parser.add_argument("--aliases", type=Path, default=Path("duplicate_aliases.json"))
    parser.add_argument("--reviewed-manifest", type=Path,
                        default=Path("open/customVideos_prepared/custom_labels_reviewed_manifest.json"))
    parser.add_argument("--groups", type=Path,
                        default=Path("artifacts/custom_camera_pairing_private/reviewed_scene_groups_private.json"))
    parser.add_argument("--original-reference", type=Path, default=Path("reference/clip_baseline_cv.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reviewed_dataset"))
    parser.add_argument("--audit-output", type=Path, default=Path("artifacts/reviewed_protocol_audit.json"))
    args = parser.parse_args()
    records, cameras = prepare_records(args.data_root, args.labels_path, args.aliases,
        args.reviewed_manifest, args.groups, args.original_reference)
    split = fixed_split(records, sha256(args.groups))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.output_dir / "split.json"
    if split_path.exists() and json.loads(split_path.read_text()) != split:
        raise ValueError("Existing locked split differs; refusing to overwrite it")
    write_json(args.output_dir / "manifest.json", records)
    write_json(split_path, split)
    by_sha = {row["sha256"]: row for row in records}
    summary = {}
    for name in ("train", "validation", "test"):
        rows = [by_sha[key] for key in split[name]]
        summary[name] = {"records": len(rows), "groups": len({row["group_id"] for row in rows}),
            "domain_class_counts": {domain: dict(Counter("violation" if row["label"] else "no_violation"
                for row in rows if row["domain"] == domain)) for domain in ("original", "custom")}}
    audit = {"schema_version": 1, "protocol": split["protocol"], "seed": SEED,
        "total_records": len(records), "domain_counts": dict(Counter(row["domain"] for row in records)),
        "source_family_reviewed_counts_before_selection": dict(cameras),
        "custom_opposite_camera_records_excluded_globally": cameras.get(".mp4", 0),
        "split_summary": summary, "assignment_sha256": split["assignment_sha256"],
        "manifest_sha256": sha256(args.output_dir / "manifest.json"),
        "split_file_sha256": sha256(split_path), "reviewed_manifest_sha256": sha256(args.reviewed_manifest),
        "source_overlap_groups_disjoint": True, "source_lineage_disjoint": True,
        "features_or_model_scores_used_to_assign_splits": False, "audio_used": False,
        "original_source_hashes_and_labels_match_frozen_40": True,
        "actual_source_paths_freshly_rebuilt": True}
    write_json(args.audit_output, audit)
    print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), "split": str(split_path),
                      "total_records": len(records), "split_summary": summary,
                      "assignment_sha256": split["assignment_sha256"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()

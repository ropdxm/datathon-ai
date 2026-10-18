"""Join the explicitly user-reviewed workbook to exact silent video lineage.

The human's chat attestation is sufficient: no second CSV attestation is needed.
Only file IDs, simple verdicts and numeric point timestamps are ingested. Other
workbook columns, including free-text event descriptions, never enter outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datathon_ml.data import _parse_label, _read_label_rows, normalize_video_id
from tools.build_custom_manifest import CUSTOM_GROUP, build_custom_manifest, write_json
from tools.verify_custom_labels import GROUPING_BASIS, parse_event_times, sha256


def _blank(value):
    return value is None or not str(value).strip()


def _aliases(records):
    result = defaultdict(set)
    for record in records:
        for value in [record["source_filename"], Path(record["path"]).name,
                      *record.get("aliases", [])]:
            result[normalize_video_id(value)].add(record["source_sha256"])
    return result


def reviewed_rows(records, rows, *, workbook_sha256, attested_sha256,
                  attestation_date, attestation_source, verify_files=True):
    """Produce only exact joined, bounded labels certified by the human user.

    Exclusions contain only row numbers, file IDs and reason codes. They are a
    private review artifact; compact public audit reports contain counts only.
    An ambiguous or duplicated source is excluded rather than inferred by order.
    """
    if (not re.fullmatch(r"[0-9a-f]{64}", str(attested_sha256))
            or workbook_sha256 != attested_sha256):
        raise ValueError("Workbook differs from the exact bytes attested by the human user")
    try:
        date = dt.date.fromisoformat(attestation_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("Attestation date must be an ISO calendar date") from exc
    if not str(attestation_source).strip():
        raise ValueError("Record the human user attestation source")
    by_sha = {record["source_sha256"]: record for record in records}
    if len(by_sha) != len(records):
        raise ValueError("Preparation manifest contains duplicate source hashes")
    aliases = _aliases(records)
    exclusions, candidates = [], []
    source_rows = Counter()
    for identifier, raw_label, raw_time, row_number in rows:
        reason, key = None, None
        if _blank(identifier):
            reason = "missing_file_id"
        else:
            matches = aliases.get(normalize_video_id(identifier), set())
            if len(matches) != 1:
                reason = "unknown_file_id" if not matches else "ambiguous_file_id"
            else:
                key = next(iter(matches))
                source_rows[key] += 1
        if reason is None and _blank(raw_label):
            reason = "missing_label"
        if reason is None:
            try:
                if isinstance(raw_label, bool):
                    raise ValueError("Boolean is not an explicit human verdict")
                label = _parse_label(raw_label.replace("_", " ") if isinstance(raw_label, str) else raw_label)
            except (ValueError, TypeError):
                reason = "unrecognized_label"
        if reason is None:
            record = by_sha[key]
            try:
                value = raw_time
                if not isinstance(value, (list, str)):
                    value = [] if value is None else [value]
                points = parse_event_times(value,
                    min(record["duration_sec"], record["source_duration_sec"]))
            except (ValueError, TypeError):
                reason = "invalid_or_out_of_bounds_timestamp"
            else:
                if bool(label) != bool(points):
                    reason = "positive_without_timestamp" if label else "normal_with_violation_timestamp"
        if reason is None and record.get("group_id") != CUSTOM_GROUP:
            reason = "unexpected_recording_group"
        if reason is None and verify_files:
            source, prepared = Path(record["source_path"]), Path(record["path"])
            if not source.is_file():
                reason = "missing_source_file"
            elif not prepared.is_file():
                reason = "missing_silent_prepared_file"
            elif sha256(source) != record["source_sha256"]:
                reason = "source_hash_changed"
            elif sha256(prepared) != record["prepared_sha256"]:
                reason = "prepared_hash_changed"
        if reason:
            exclusions.append({"row": row_number, "file_id": str(identifier) if identifier else None,
                               "reason": reason})
            continue
        candidates.append((row_number, identifier, key, label, points))
    accepted = []
    for row_number, identifier, key, label, points in candidates:
        if source_rows[key] != 1:
            exclusions.append({"row": row_number, "file_id": str(identifier),
                               "reason": "duplicate_source_rows"})
            continue
        record = dict(by_sha[key])
        record.update({
            "label": label, "event_times": points, "label_status": "human_verified",
            "annotation_status": "human_verified", "human_verified": True,
            "training_supervision": "human_verified",
            "eligible_for_supervised_training": True,
            "evaluation_ground_truth_eligible": True,
            "event_times_verified": bool(label),
            "group_verified": True, "grouping_basis": GROUPING_BASIS,
            "individual_scene_pairing_verified": False,
            "reviewer": "dataset_owner_via_chat",
            "review_attestation_date": date.isoformat(),
            "review_attestation_source": str(attestation_source),
            "reviewed_workbook_sha256": workbook_sha256,
            "reviewed_workbook_row": row_number,
            "verification_basis": "explicit_human_user_workbook_attestation_and_numeric_integrity_checks",
            "audio_used": False,
        })
        accepted.append(record)
    audit = {
        "schema_version": 1,
        "workbook_sha256": workbook_sha256,
        "human_user_attestation_accepted": True,
        "attestation_date": date.isoformat(), "attestation_source": str(attestation_source),
        "preparation_source_count": len(records), "annotation_rows": len(rows),
        "accepted_records": len(accepted),
        "class_counts": dict(Counter("violation" if row["label"] else "no_violation" for row in accepted)),
        "verified_point_count": sum(len(row["event_times"]) for row in accepted),
        "excluded_rows": len(exclusions),
        "exclusion_reason_counts": dict(Counter(row["reason"] for row in exclusions)),
        "preparation_sources_without_accepted_label": len(records) - len(accepted),
        "source_and_silent_derivative_hashes_verified": bool(verify_files),
        "all_accepted_prepared_audio_streams_zero": all(row.get("audio_streams") == 0 for row in accepted),
        "grouping_basis": GROUPING_BASIS,
        "accepted_group_count": len({row["group_id"] for row in accepted}),
        "individual_scene_pairing_verified": False,
        "timestamp_policy": "Supplied points only; 0 <= point < min(source duration, silent derivative duration).",
        "semantic_visual_verification_performed_by_assistant": False,
        "private_description_columns_used": False,
        "predictions_used_as_ground_truth": False,
        "audio_used": False,
    }
    return accepted, audit, exclusions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path,
                        default=Path("open/customVideos_prepared/custom_labels_proposed.xlsx"))
    parser.add_argument("--preparation-manifest", type=Path,
                        default=Path("open/customVideos_prepared/preparation_manifest.json"))
    parser.add_argument("--attested-workbook-sha256", required=True)
    parser.add_argument("--attestation-date", required=True)
    parser.add_argument("--attestation-source", default="explicit_human_user_chat_confirmation")
    parser.add_argument("--output", type=Path,
                        default=Path("open/customVideos_prepared/custom_labels_reviewed_manifest.json"))
    parser.add_argument("--exclusions-output", type=Path,
                        default=Path("open/customVideos_prepared/custom_labels_reviewed_exclusions.json"))
    parser.add_argument("--audit-output", type=Path,
                        default=Path("artifacts/reviewed_workbook_audit.json"))
    args = parser.parse_args()
    digest = sha256(args.workbook)
    # A missing recording that has no workbook label is irrelevant to this
    # supervised subset. Verify both original and derivative for every used row.
    records = build_custom_manifest(args.preparation_manifest, verify_files=False)
    rows = _read_label_rows(args.workbook)
    accepted, audit, exclusions = reviewed_rows(records, rows,
        workbook_sha256=digest, attested_sha256=args.attested_workbook_sha256,
        attestation_date=args.attestation_date, attestation_source=args.attestation_source)
    if sha256(args.workbook) != digest:
        raise ValueError("Workbook changed while annotation ingestion was running")
    audit.update(preparation_manifest_sha256=sha256(args.preparation_manifest),
                 created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write_json(args.output, accepted)
    write_json(args.exclusions_output, exclusions)
    write_json(args.audit_output, audit)
    print(json.dumps({"accepted_records": len(accepted), "class_counts": audit["class_counts"],
        "verified_point_count": audit["verified_point_count"],
        "excluded_rows": len(exclusions), "exclusion_reason_counts": audit["exclusion_reason_counts"],
        "group_count": audit["accepted_group_count"], "audit": str(args.audit_output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()

"""Audit custom annotation joins and ingest explicit offline human reviews.

This tool never looks at video pixels, audio or free-text event descriptions.
It cannot verify what a video depicts. A completed human review is required
before a custom recording can supply supervised targets. Predictions, copied
organizer labels and a populated label cell alone are never ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datathon_ml.data import _read_label_rows, normalize_video_id, parse_timestamp
from tools.build_custom_manifest import CUSTOM_GROUP, build_custom_manifest, write_json


REVIEW_COLUMNS = (
    "source_filename", "source_sha256", "prepared_path", "duration_sec",
    "label", "event_times", "review_status", "full_video_reviewed",
    "reviewer", "reviewed_at_utc",
)
GROUPING_BASIS = "user_confirmed_shared_filming_batch_conservative_union"
REVIEW_INSTRUCTIONS = (
    "Open the local silent prepared_path video and review its complete duration. "
    "Use label=violation with event_times in elapsed video seconds, for example "
    "[12.5, 27], or label=no_violation with event_times=[]. Leave uncertain cases "
    "unreviewed. After visual verification, set review_status=human_verified, "
    "full_video_reviewed=yes, reviewer to your identifier and reviewed_at_utc "
    "to an ISO UTC timestamp such as 2026-10-18T10:00:00Z. Do not copy labels "
    "between cameras: visibility and recording start times can differ. Scene "
    "pairing is a separate review; labels do not establish pair correspondence."
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _blank(value):
    return value is None or not str(value).strip()


def parse_event_times(value, duration):
    """Read a numeric JSON list or semicolon-separated seconds/clock times."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            values = []
        elif text.startswith("["):
            try:
                values = json.loads(text)
            except (ValueError, TypeError) as exc:
                raise ValueError("Invalid event_times JSON") from exc
        else:
            values = text.split(";")
    else:
        values = value
    if not isinstance(values, list):
        raise ValueError("event_times must be a list or semicolon-separated times")
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError("Video duration must be finite and positive")
    points = []
    for value in values:
        try:
            point = parse_timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid review timestamp") from exc
        if point is None or not math.isfinite(point) or not 0 <= point < duration:
            raise ValueError("Review timestamp must be inside the source and prepared video")
        points.append(float(point))
    if len(set(points)) != len(points):
        raise ValueError("Duplicate review timestamps")
    return sorted(points)


def _verified_utc(value):
    try:
        parsed = dt.datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Human review needs reviewed_at_utc as an ISO UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError("Human review timestamp must explicitly use UTC")
    if parsed > dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5):
        raise ValueError("Human review timestamp cannot be in the future")
    return parsed.isoformat()


def ingest_review_rows(records, rows):
    """Validate exact source identity and human attestation, with fail-closed joins.

    ``human_verified`` records a human's explicit attestation, not an automated
    semantic judgment. This function never changes recording groups or pairing.
    The already declared shared filming-batch union is safe for grouped splits,
    even while individual scene/camera correspondence remains unresolved.
    """
    by_sha = {record["source_sha256"]: record for record in records}
    if len(by_sha) != len(records):
        raise ValueError("Custom manifest contains duplicate source hashes")
    seen = set()
    verified = []
    status_counts = Counter()
    filled_unverified = 0
    for row in rows:
        digest = str(row.get("source_sha256", "")).strip()
        record = by_sha.get(digest)
        if record is None:
            raise ValueError("Review source SHA-256 does not match a custom original")
        if row.get("source_filename") != record["source_filename"]:
            raise ValueError("Review source filename and SHA-256 disagree")
        if digest in seen:
            raise ValueError("Duplicate source row in human review")
        seen.add(digest)
        status = str(row.get("review_status", "unreviewed")).strip().lower() or "unreviewed"
        if status not in {"unreviewed", "uncertain", "human_verified"}:
            raise ValueError("Unknown review status; machine proposals cannot become verified labels")
        status_counts[status] += 1
        if status != "human_verified":
            filled_unverified += int(not _blank(row.get("label")))
            continue
        if str(row.get("full_video_reviewed", "")).strip().lower() != "yes":
            raise ValueError("Human verification requires full_video_reviewed=yes")
        reviewer = str(row.get("reviewer", "")).strip()
        if not reviewer or reviewer.lower() in {"model", "machine", "automatic", "ai", "llm"}:
            raise ValueError("Human verification requires a human reviewer identifier")
        reviewed_at = _verified_utc(row.get("reviewed_at_utc"))
        label = str(row.get("label", "")).strip().lower()
        if label not in {"violation", "no_violation", "no violation"}:
            raise ValueError("Human label must be violation or no_violation")
        duration = min(record["duration_sec"], record["source_duration_sec"])
        points = parse_event_times(row.get("event_times", ""), duration)
        positive = label == "violation"
        if positive != bool(points):
            raise ValueError("A violation requires timestamps; no_violation must have none")
        if record.get("group_id") != CUSTOM_GROUP:
            raise ValueError("Verified custom labels require the conservative shared filming-batch group")
        item = dict(record)
        item.update({
            "label": int(positive), "event_times": points,
            "label_status": "human_verified", "human_verified": True,
            "annotation_status": "human_verified", "training_supervision": "human_verified",
            "eligible_for_supervised_training": True,
            "evaluation_ground_truth_eligible": True,
            "reviewer": reviewer, "reviewed_at_utc": reviewed_at,
            "event_times_verified": bool(positive),
            "group_verified": True, "grouping_basis": GROUPING_BASIS,
            "individual_scene_pairing_verified": False,
            "verification_basis": "explicit_human_full_video_review_attestation",
        })
        verified.append(item)
    return verified, {
        "review_rows": len(seen), "review_status_counts": dict(status_counts),
        "filled_labels_without_human_verification": filled_unverified,
        "verified_labels": len(verified),
        "verified_violation_labels": sum(record["label"] for record in verified),
        "verified_no_violation_labels": sum(not record["label"] for record in verified),
        "verified_timestamps": sum(len(record["event_times"]) for record in verified),
        "custom_sources_without_verified_labels": len(records) - len(verified),
        "scene_groups_changed_by_label_ingestion": False,
    }


def write_review_template(records, path):
    """Create a blank private worksheet; never replace completed human work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "source_filename": record["source_filename"],
                "source_sha256": record["source_sha256"],
                "prepared_path": record["path"],
                "duration_sec": min(record["duration_sec"], record["source_duration_sec"]),
                "label": "", "event_times": "[]", "review_status": "unreviewed",
                "full_video_reviewed": "", "reviewer": "", "reviewed_at_utc": "",
            })


def read_review(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(REVIEW_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError("Review CSV is missing required identity and attestation columns")
        return list(reader)


def audit_workbook(path, records, original_rows=None):
    """Read only the ID, simple label and timestamp columns; never emit text."""
    rows = _read_label_rows(Path(path))
    custom_ids = {normalize_video_id(record["source_filename"]) for record in records}
    ids = {normalize_video_id(row[0]) for row in rows if not _blank(row[0])}
    matching = [row for row in rows if not _blank(row[0]) and normalize_video_id(row[0]) in custom_ids]
    return {
        "filename": Path(path).name, "sha256": sha256(path),
        "size_bytes": Path(path).stat().st_size,
        "provided_rows": len(rows), "matching_custom_ids": len(ids & custom_ids),
        "matching_custom_rows_with_filled_labels": sum(not _blank(row[1]) for row in matching),
        "matching_custom_rows_with_filled_timestamps": sum(not _blank(row[2]) for row in matching),
        "nonblank_label_rows": sum(not _blank(row[1]) for row in rows),
        "nonblank_timestamp_rows": sum(not _blank(row[2]) for row in rows),
        "first_three_columns_equal_original": rows == original_rows if original_rows is not None else None,
        "semantic_verification_from_workbook_content": False,
        "automatic_label_attachment": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-manifest", type=Path,
                        default=Path("open/customVideos_prepared/preparation_manifest.json"))
    parser.add_argument("--original-labels", type=Path, default=Path("open/open_labels.xlsx"))
    parser.add_argument("--workbook", type=Path, action="append", default=[])
    parser.add_argument("--review-csv", type=Path,
                        default=Path("open/customVideos_prepared/custom_labels_review.csv"))
    parser.add_argument("--verified-output", type=Path,
                        default=Path("open/customVideos_prepared/custom_labels_verified.json"))
    parser.add_argument("--audit-output", type=Path,
                        default=Path("artifacts/custom_label_verification.json"))
    args = parser.parse_args()
    records = build_custom_manifest(args.preparation_manifest, verify_files=True)
    original_rows = _read_label_rows(args.original_labels)
    paths = args.workbook or [Path("open/customVideos/open_labels.xlsx"),
                              Path("open/customVideos_prepared/custom_labels_proposed.xlsx")]
    workbook_audits = [audit_workbook(path, records, original_rows) for path in paths if path.is_file()]
    if not args.review_csv.exists():
        write_review_template(records, args.review_csv)
    verified, review_audit = ingest_review_rows(records, read_review(args.review_csv))
    # A full reviewed manifest is private. The tracked report contains counts,
    # checksums and procedure only, never private frames or event descriptions.
    write_json(args.verified_output, verified)
    audit = {
        "schema_version": 1, "audit_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_count": len(records), "source_and_prepared_file_hashes_verified": True,
        "silent_prepared_recordings_verified": all(record["audio_streams"] == 0 for record in records),
        "preparation_manifest_sha256": sha256(args.preparation_manifest),
        "review_csv_sha256": sha256(args.review_csv),
        "original_labels_sha256": sha256(args.original_labels),
        "workbooks": workbook_audits, "review": review_audit,
        "custom_semantics_verified_by_automation": False,
        "timestamp_bounds_policy": "0 <= each point < min(source_duration, prepared_duration)",
        "audio_used": False, "private_pixels_or_descriptions_sent_to_hosted_ai": False,
        "machine_predictions_accepted_as_ground_truth": False,
        "scene_grouping_policy": "All custom views retain one user-confirmed shared filming-batch union. This grouping is verified only as a conservative leakage-prevention union; individual scene/camera pairs remain unverified.",
        "grouping_basis": GROUPING_BASIS,
        "individual_scene_pairing_verified_by_label_ingestion": False,
        "needs_human_review": bool(len(verified) < len(records)),
        "review_instructions": REVIEW_INSTRUCTIONS,
    }
    write_json(args.audit_output, audit)
    print(json.dumps({"source_count": len(records), **review_audit,
                      "silent_video_only": True, "audit": str(args.audit_output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()

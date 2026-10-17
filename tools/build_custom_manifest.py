"""Build video-only custom manifests without turning predictions into ground truth.

The prepared-video manifest supplies source lineage and audio/timing checks.
Labels are never discovered from an XLSX automatically. Optional numeric local
model proposals remain unverified; using them for training requires an explicit
flag and never makes them eligible as evaluation ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datathon_ml.data import _read_label_rows, normalize_video_id


CUSTOM_GROUP = "all_custom_simultaneous_session_20261017"
TIME_MAPPING = "output_elapsed_seconds = source_video_elapsed_seconds"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Write JSON atomically; incomplete writes must not look like a manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sha(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive(value, field):
    if isinstance(value, bool):
        raise ValueError(f"Invalid {field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Invalid {field}")
    return number


def _fps(value):
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        return _positive(numerator, "fps numerator") / _positive(denominator, "fps denominator")
    return _positive(value, "fps")


def _contained(root, relative):
    relative = Path(relative)
    root = Path(root).expanduser().resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Preparation paths must remain inside their declared directories")
    return path


def _apply_proposals(records, payload, allow_machine_labels):
    entries = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("Proposals must contain a records list")
    by_sha = {record["source_sha256"]: record for record in records}
    by_name = {record["source_filename"]: record for record in records}
    seen = set()
    for proposal in entries:
        if not isinstance(proposal, dict):
            raise ValueError("Proposal entries must be objects")
        source_sha = proposal.get("source_sha256")
        filename = proposal.get("source_filename")
        record = by_sha.get(source_sha) if source_sha is not None else by_name.get(filename)
        if record is None:
            raise ValueError("Proposal refers to an unknown custom source; copied original labels cannot attach")
        if source_sha is not None and source_sha != record["source_sha256"]:
            raise ValueError("Proposal source hash does not match")
        if filename is not None and filename != record["source_filename"]:
            raise ValueError("Proposal filename and source hash disagree")
        if record["source_sha256"] in seen:
            raise ValueError("Duplicate proposal for a custom source")
        seen.add(record["source_sha256"])
        if proposal.get("label_status", "machine_proposal") != "machine_proposal":
            raise ValueError("This input only accepts unverified machine proposals")
        label = proposal.get("label")
        if label not in (None, "violation", "no violation"):
            raise ValueError("Proposal label must be violation, no violation or null")
        points = proposal.get("event_times", [])
        if not isinstance(points, list):
            raise ValueError("Proposal event_times must be a numeric list")
        if any(isinstance(point, bool) or not isinstance(point, (int, float))
               or not math.isfinite(point) or point < 0
               or point >= min(record["duration_sec"], record["source_duration_sec"])
               for point in points):
            raise ValueError("Proposal timestamp is outside the source or prepared video")
        points = sorted(set(float(point) for point in points))
        if label == "violation" and not points:
            raise ValueError("Violation proposals require a timestamp")
        if label != "violation" and points:
            raise ValueError("Unknown or no violation proposals cannot carry violation timestamps")
        protocol = proposal.get("protocol_sha256")
        if protocol is not None:
            _sha(protocol, "proposal protocol_sha256")
        record.update({
            "proposed_label": label,
            "proposed_event_times": points,
            "proposal_protocol_sha256": protocol,
            "annotation_status": "machine_proposal_requires_human_review",
        })
        if allow_machine_labels and label is not None:
            record.update({"label": int(label == "violation"), "event_times": points,
                           "label_status": "machine_proposal",
                           "training_supervision": "machine_pseudo_label",
                           "eligible_for_supervised_training": True})


def build_custom_manifest(preparation_manifest, proposals_path=None,
                          allow_machine_labels=False, verify_files=True):
    """Return existing-model-compatible records with explicit source provenance.

    ``sha256`` identifies the actual prepared pixels used by feature extraction;
    ``source_sha256`` identifies the preserved original recording. All custom
    views share a conservative recording group because the user confirmed
    simultaneous filming of the same scenes with two cameras but not pairing.
    """
    preparation = _read_json(preparation_manifest)
    if preparation.get("complete") is not True:
        raise ValueError("Video preparation must be complete before building a dataset")
    videos = preparation.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("Preparation manifest has no videos")
    records = []
    seen_source, seen_prepared, seen_names = set(), set(), set()
    for row in videos:
        source = _contained(preparation["source_dir"], row["source_relative_path"])
        prepared = _contained(preparation["output_dir"], row["output_relative_path"])
        source_sha = _sha(row.get("source_sha256"), "source_sha256")
        prepared_sha = _sha(row.get("output_sha256"), "output_sha256")
        if source_sha in seen_source or prepared_sha in seen_prepared or source.name in seen_names:
            raise ValueError("Duplicate source, prepared recording or source filename in preparation manifest")
        seen_source.add(source_sha)
        seen_prepared.add(prepared_sha)
        seen_names.add(source.name)
        metadata = row["output"]
        verification = row.get("verification", {})
        if metadata.get("audio_streams") != 0 or metadata.get("stream_types") != ["video"]:
            raise ValueError("Prepared video must contain one video stream and no audio")
        if verification.get("verified") is not True or verification.get("annotation_time_mapping") != TIME_MAPPING:
            raise ValueError("Prepared video lacks verified unchanged annotation timing")
        tolerance = _positive(verification.get("duration_tolerance_seconds"), "duration tolerance")
        error = verification.get("duration_error_seconds")
        if (isinstance(error, bool) or not isinstance(error, (int, float))
                or not math.isfinite(error) or abs(error) > tolerance):
            raise ValueError("Prepared duration exceeds the verified timing tolerance")
        duration = _positive(metadata.get("duration_seconds"), "duration_seconds")
        source_duration = _positive(row["source"].get("duration_seconds"), "source duration_seconds")
        if abs(duration - source_duration) > tolerance:
            raise ValueError("Prepared metadata durations exceed the verified timing tolerance")
        width = _positive(metadata.get("width"), "width")
        height = _positive(metadata.get("height"), "height")
        if not width.is_integer() or not height.is_integer():
            raise ValueError("Video dimensions must be integer values")
        if verify_files:
            for path, expected in ((source, source_sha), (prepared, prepared_sha)):
                if not path.is_file() or _digest(path) != expected:
                    raise ValueError(f"Missing or changed source/prepared video: {path.name}")
        records.append({
            "video_id": "CUSTOM-" + source_sha[:16], "path": str(prepared),
            "sha256": prepared_sha, "source_path": str(source),
            "source_filename": source.name, "source_sha256": source_sha,
            "prepared_sha256": prepared_sha, "dataset": "custom",
            "duration_sec": duration, "source_duration_sec": source_duration,
            "fps": _fps(metadata.get("fps")), "width": int(width), "height": int(height),
            "label": None, "event_times": [], "label_status": "unlabeled",
            "annotation_status": "unlabeled", "training_supervision": None,
            "eligible_for_supervised_training": False,
            "evaluation_ground_truth_eligible": False, "human_verified": False,
            "aliases": [source.name], "group_id": CUSTOM_GROUP,
            "pair_id": None, "pairing_status": "unresolved_simultaneous_views",
            "camera_id": "iphone_mov" if source.suffix.lower() == ".mov" else "samsung_mp4",
            "audio_used": False, "audio_streams": 0,
            "annotation_time_mapping": TIME_MAPPING,
        })
    if proposals_path is not None:
        _apply_proposals(records, _read_json(proposals_path), allow_machine_labels)
    elif allow_machine_labels:
        raise ValueError("--allow-machine-labels requires --proposals")
    return records


def audit_custom_dataset(records, custom_labels=None, original_labels=None,
                         original_video_root=None):
    """Audit only identifiers, numeric annotations, metadata and exact hashes."""
    filenames = {normalize_video_id(record["source_filename"]) for record in records}
    annotation_audit = {"automatic_label_attachment": False}
    if custom_labels is not None:
        rows = _read_label_rows(Path(custom_labels))
        row_ids = {normalize_video_id(row[0]) for row in rows}
        annotation_audit.update({"provided_rows": len(rows),
                                 "matching_custom_ids": len(filenames & row_ids),
                                 "custom_files_without_matching_label": len(filenames - row_ids),
                                 "referenced_ids_missing_from_custom": len(row_ids - filenames)})
        if original_labels is not None:
            annotation_audit["first_three_columns_equal_original_workbook"] = (
                rows == _read_label_rows(Path(original_labels)))
    original_hashes = defaultdict(list)
    if original_video_root is not None:
        original_video_root = Path(original_video_root)
        if not original_video_root.is_dir():
            raise ValueError("Original video directory does not exist")
        for path in original_video_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".mov", ".mp4"}:
                original_hashes[_digest(path)].append(path.name)
        if not original_hashes:
            raise ValueError("Original video directory has no recordings to check for overlap")
    exact_original_overlap = [record["source_filename"] for record in records
                              if record["source_sha256"] in original_hashes]
    return {
        "schema_version": 1, "video_count": len(records),
        "source_extensions": dict(Counter(Path(record["source_filename"]).suffix.lower() for record in records)),
        "source_total_bytes": sum(Path(record["source_path"]).stat().st_size for record in records),
        "prepared_total_bytes": sum(Path(record["path"]).stat().st_size for record in records),
        "all_prepared_audio_streams_zero": all(record["audio_streams"] == 0 for record in records),
        "audio_used_for_pairing_annotation_or_training": False,
        "label_status_counts": dict(Counter(record["label_status"] for record in records)),
        "human_verified_label_count": 0,
        "source_recordings_below_two_seconds": [record["source_filename"] for record in records
                                               if record["source_duration_sec"] < 2],
        "annotation_workbook": annotation_audit,
        "original_unique_source_count_for_exact_overlap_check": len(original_hashes) if original_video_root is not None else None,
        "exact_source_overlap_with_original": exact_original_overlap,
        "known_recording_groups": [CUSTOM_GROUP], "independent_session_count": 1,
        "simultaneous_two_camera_shared_scenes_user_confirmed": True,
        "cross_camera_correspondence_verified": False,
        "metadata_only_matching_is_not_proof_of_pairing": True,
        "evaluation_policy": "Keep all custom views in one recording group; machine proposals are training-only and never evaluation ground truth. Compare custom-assisted and original-only models on identical original held-out SHA folds.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compression-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/custom_dataset"))
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--allow-machine-labels", action="store_true",
                        help="Opt into unverified pseudo labels for training, never evaluation")
    parser.add_argument("--custom-labels", type=Path)
    parser.add_argument("--original-labels", type=Path)
    parser.add_argument("--original-video-root", type=Path)
    parser.add_argument("--audit-output", type=Path, default=Path("artifacts/custom_dataset_audit.json"))
    args = parser.parse_args()
    records = build_custom_manifest(args.compression_manifest, args.proposals, args.allow_machine_labels)
    audit = audit_custom_dataset(records, args.custom_labels, args.original_labels, args.original_video_root)
    write_json(args.output_dir / "manifest.json", records)
    write_json(args.audit_output, audit)
    print(json.dumps({"records": len(records), "label_status_counts": audit["label_status_counts"],
                      "manifest": str(args.output_dir / "manifest.json"),
                      "audit": str(args.audit_output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()

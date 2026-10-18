"""Audit conservative groups without exposing private pixels or using labels.

The full custom batch always retains one filming-session group. A separately
named camera-exclusive development protocol can exclude every opposite-camera
view, then unite native same-camera overlap candidates. It is not a new-session
test and the inferred groups are not human-verified scene identities.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

FPS = 2
WIDTH, HEIGHT = 64, 36
PROTOCOL = {
    "version": 1,
    "protocol_name": "camera_exclusive_within_batch_source_development",
    "family_selection": "choose by usable-label coverage before inspecting any model score",
    "opposite_camera_policy": "all opposite-family views excluded from train validation and test",
    "minimum_reviewable_duration_seconds": 3.,
    "minimum_overlap_seconds": 3.,
    "minimum_shorter_fraction": .5,
    "plausible_overlap_mae": 6.,
    "plausible_overlap_phash_median": 10.,
    "plausible_overlap_motion_correlation": .60,
    "minimum_dynamic_steps": 3,
    "dynamic_step_mean_absolute_change": .50,
    "group_policy": "union every plausible same-camera overlap candidate; abstain on short sources",
    "accuracy_interpretation": "within-filming-batch development only; no independent-session or cross-camera claim",
    "group_verification": "native duplicate/overlap prevention; not human-verified physical scene identity",
    "audio": "never decoded; input prepared-file hash and silent lineage verified",
    "labels": "not read or used by this audit",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def components(nodes, edges):
    parent = {n: n for n in nodes}
    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for a, b in edges:
        if a not in parent or b not in parent:
            raise ValueError("Unknown source in grouping edge")
        aa, bb = find(a), find(b)
        parent[max(aa, bb)] = min(aa, bb)
    groups = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return sorted([sorted(g) for g in groups.values()], key=lambda g: g[0])


def frame_features(frames):
    frames = np.asarray(frames, np.uint8)
    hashes = []
    for frame in frames:
        dct = cv2.dct(cv2.resize(frame, (32, 32)).astype(np.float32))[:8, :8].ravel()
        hashes.append(dct > np.median(dct[1:]))
    return {"pixels": frames.astype(np.float32).reshape(len(frames), -1),
            "phash": np.asarray(hashes, bool),
            "motion": np.mean(np.abs(np.diff(frames.astype(float), axis=0)), axis=(1, 2))}


def decode_features(path, ffmpeg="ffmpeg"):
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-vf",
               f"setpts=PTS-STARTPTS,fps={FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
               f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,format=gray",
               "-threads", "1", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    result = subprocess.run(command, capture_output=True, check=True)
    if len(result.stdout) % (WIDTH * HEIGHT):
        raise ValueError("Truncated native frame stream")
    frames = np.frombuffer(result.stdout, np.uint8).reshape(-1, HEIGHT, WIDTH)
    return frame_features(frames)


def overlap_evidence(left, right):
    x, y = left["pixels"], right["pixels"]
    minimum = max(int(PROTOCOL["minimum_overlap_seconds"] * FPS),
                  int(np.ceil(PROTOCOL["minimum_shorter_fraction"] * min(len(x), len(y)))))
    best = {"plausible_overlap": False, "reviewable": min(len(x), len(y)) >= minimum,
            "offset_seconds": None, "overlap_seconds": 0., "pixel_mae": None,
            "phash_median_distance": None, "motion_correlation": None,
            "dynamic_steps": 0, "matching_frame_fraction": 0.}
    if not best["reviewable"]:
        return best
    best_rank = (-1., -1., -1.)
    for offset in range(-(len(x) - minimum), len(y) - minimum + 1):
        start_x, start_y = max(0, -offset), max(0, offset)
        count = min(len(x) - start_x, len(y) - start_y)
        if count < minimum:
            continue
        a, b = x[start_x:start_x + count], y[start_y:start_y + count]
        errors = np.mean(np.abs(a - b), axis=1)
        mean_error = float(np.mean(errors))
        # Global photometric similarity is necessary for same-camera overlap;
        # shared static backgrounds alone cannot establish physical overlap.
        if mean_error > PROTOCOL["plausible_overlap_mae"] * 2:
            continue
        hamming = np.sum(left["phash"][start_x:start_x + count] !=
                         right["phash"][start_y:start_y + count], axis=1)
        hamming_median = float(np.median(hamming))
        ma = left["motion"][start_x:start_x + count - 1]
        mb = right["motion"][start_y:start_y + count - 1]
        dynamic = int(np.sum((ma >= PROTOCOL["dynamic_step_mean_absolute_change"]) &
                             (mb >= PROTOCOL["dynamic_step_mean_absolute_change"])))
        corr = (float(np.corrcoef(ma, mb)[0, 1]) if np.std(ma) > 1e-5 and np.std(mb) > 1e-5 else 0.)
        fraction = float(np.mean(errors <= PROTOCOL["plausible_overlap_mae"]))
        plausible = (mean_error <= PROTOCOL["plausible_overlap_mae"] and
                     hamming_median <= PROTOCOL["plausible_overlap_phash_median"] and
                     corr >= PROTOCOL["plausible_overlap_motion_correlation"] and
                     dynamic >= PROTOCOL["minimum_dynamic_steps"] and fraction >= .75)
        rank = (float(plausible), fraction - mean_error / 100., corr)
        if rank > best_rank:
            best_rank = rank
            best.update({"plausible_overlap": bool(plausible), "offset_seconds": offset / FPS,
                         "overlap_seconds": count / FPS, "pixel_mae": mean_error,
                         "phash_median_distance": hamming_median, "motion_correlation": corr,
                         "dynamic_steps": dynamic, "matching_frame_fraction": fraction})
    return best


def audit(preparation_path, pairing_path, private_dir, summary_path, ffmpeg="ffmpeg"):
    started = time.monotonic()
    prep = json.loads(Path(preparation_path).read_text(encoding="utf-8"))
    pairing = json.loads(Path(pairing_path).read_text(encoding="utf-8"))
    if not prep.get("complete") or pairing["preparation_manifest_sha256"] != sha256(preparation_path):
        raise ValueError("Complete matching preparation lineage required")
    root = Path(prep["output_dir"]).resolve()
    records, features = [], {}
    for row in prep["videos"]:
        if row["output"]["stream_types"] != ["video"] or row["output"]["audio_streams"]:
            raise ValueError("Only verified silent derivatives are accepted")
        path = (root / row["output_relative_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Prepared source containment mismatch")
        source = row["source_sha256"]
        family = Path(row["source_relative_path"]).suffix.lower().removeprefix(".")
        if family not in ("mov", "mp4"):
            raise ValueError("Unknown camera family")
        available = path.is_file()
        if available and sha256(path) != row["output_sha256"]:
            raise ValueError("Available prepared source hash mismatch")
        records.append({"source_sha256": source, "prepared_sha256": row["output_sha256"],
                        "source_family": family, "duration_seconds": row["source"]["duration_seconds"],
                        "prepared_file_available": available,
                        "camera_exclusive_reviewable": available and row["source"]["duration_seconds"] >=
                        PROTOCOL["minimum_reviewable_duration_seconds"]})
        if available:
            features[source] = decode_features(path, ffmpeg)
    nodes = [r["source_sha256"] for r in records]
    available_nodes = [r["source_sha256"] for r in records if r["prepared_file_available"]]
    available_set = set(available_nodes)
    graph_sensitivity = []
    for threshold in (.50, .60, .70, .80):
        edges = [(c["left_source_sha256"], c["right_source_sha256"])
                 for c in pairing["candidates"] if c["motion_score"] >= threshold]
        groups = components(nodes, edges)
        available_edges = [(a, b) for a, b in edges if a in available_set and b in available_set]
        available_groups = components(available_nodes, available_edges)
        graph_sensitivity.append({"motion_threshold": threshold, "candidate_edges": len(edges),
                                  "component_count": len(groups), "component_sizes": sorted([len(g) for g in groups], reverse=True),
                                  "available_source_count": len(available_nodes),
                                  "available_component_count": len(available_groups),
                                  "available_component_sizes": sorted([len(g) for g in available_groups], reverse=True)})
    results, family_reports, group_map = [], {}, {}
    for family in ("mov", "mp4"):
        members = sorted([r for r in records if r["source_family"] == family], key=lambda r: r["source_sha256"])
        edges = []
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if not a["prepared_file_available"] or not b["prepared_file_available"]:
                    continue
                evidence = overlap_evidence(features[a["source_sha256"]], features[b["source_sha256"]])
                evidence.update({"left_source_sha256": a["source_sha256"], "right_source_sha256": b["source_sha256"],
                                 "source_family": family})
                results.append(evidence)
                if evidence["plausible_overlap"]:
                    edges.append((a["source_sha256"], b["source_sha256"]))
        groups = components([r["source_sha256"] for r in members], edges)
        for group in groups:
            group_id = "camera_exclusive_" + family + "_" + hashlib.sha256("|".join(group).encode()).hexdigest()[:16]
            for source in group:
                group_map[source] = group_id
        available_count = sum(r["prepared_file_available"] for r in members)
        available_groups = [[n for n in group if n in available_set] for group in groups]
        available_groups = [g for g in available_groups if g]
        family_reports[family] = {"source_count": len(members), "available_source_count": available_count,
                                  "unavailable_source_count": len(members) - available_count,
                                  "pair_comparisons": available_count * (available_count - 1) // 2,
                                  "plausible_overlap_edges": len(edges), "source_overlap_groups": len(groups),
                                  "group_sizes": sorted([len(g) for g in groups], reverse=True),
                                  "available_source_overlap_groups": len(available_groups),
                                  "available_group_sizes": sorted([len(g) for g in available_groups], reverse=True),
                                  "reviewable_source_count": sum(r["camera_exclusive_reviewable"] for r in members),
                                  "short_sources_require_abstention": sum(r["prepared_file_available"] and not r["camera_exclusive_reviewable"] for r in members)}
    private = {"protocol": PROTOCOL, "preparation_sha256": sha256(preparation_path),
               "records": records, "same_camera_pair_evidence": results,
               "camera_exclusive_group_by_source": group_map,
               "full_custom_batch_group": "all_custom_simultaneous_session_20261017",
               "human_verified_scene_groups": False}
    private_dir = Path(private_dir)
    private_dir.mkdir(parents=True, exist_ok=True)
    (private_dir / "reviewed_scene_groups_private.json").write_text(json.dumps(private, indent=2, allow_nan=False), encoding="utf-8")
    summary = {"schema_version": 1, "protocol": PROTOCOL, "implementation_sha256": sha256(__file__),
               "source_count": len(records), "source_label_values_read": 0,
               "full_custom_batch_conservative_group_count": 1,
               "ambiguous_cross_camera_graph_sensitivity": graph_sensitivity,
               "camera_exclusive_family_audits": family_reports,
               "all_opposite_family_views_must_be_excluded": True,
               "independent_scene_verification": False, "independent_session_validation": False,
               "private_group_mapping_relative_path": str(private_dir / "reviewed_scene_groups_private.json"),
               "limitations": ["No overlap candidate is not proof of independent physical scenes.",
                               "Perspective transforms speed changes or very brief overlaps can evade this diagnostic.",
                               "Static matching frames alone do not establish duplicate dynamic footage.",
                               "The camera-exclusive protocol is source-based development within one filming batch."],
               "elapsed_seconds": time.monotonic() - started}
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", default="open/customVideos_prepared/preparation_manifest.json")
    parser.add_argument("--pairing", default="artifacts/custom_camera_pairing_private/camera_pairs_private.json")
    parser.add_argument("--private-dir", default="artifacts/custom_camera_pairing_private")
    parser.add_argument("--summary", default="artifacts/reviewed_scene_group_audit.json")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    result = audit(args.preparation, args.pairing, args.private_dir, args.summary, args.ffmpeg)
    print(json.dumps({"camera_exclusive_family_audits": result["camera_exclusive_family_audits"],
                      "ambiguous_cross_camera_graph_sensitivity": result["ambiguous_cross_camera_graph_sensitivity"],
                      "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()

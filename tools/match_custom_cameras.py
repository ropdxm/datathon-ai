"""Silent native-CV camera pairing audit with explicit abstention.

This does not inspect semantic actions, derive compliance labels, or establish
human-verified camera pairs. Only already silent prepared video streams are
decoded. Source metadata and private per-recording numeric results remain local.
All custom recordings retain their conservative common filming-session group.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d

GROUP = "all_custom_simultaneous_session_20261017"
PROTOCOL = {
    "schema_version": 1,
    "sample_fps": 5,
    "width": 192,
    "height": 108,
    "minimum_overlap_seconds": 6.0,
    "minimum_shorter_overlap_fraction": .70,
    "automatic_minimum_duration_seconds": 8.0,
    "automatic_motion_score": .70,
    "automatic_split_correlation": .50,
    "automatic_channel_correlation": .55,
    "automatic_assignment_margin": .08,
    "automatic_remote_peak_margin": .05,
    "automatic_maximum_drift_seconds": .60,
    "automatic_clock_residual_seconds": 2.0,
    "offset_convention": "right_elapsed_seconds = left_elapsed_seconds + offset_seconds",
    "clock_offset_convention": "right_creation_minus_left_creation_plus_motion_offset",
    "assignment": "Hungarian partial matching with zero-score dummy abstentions",
    "audio_policy": "decode prepared video stream only using FFmpeg -an -sn -dn",
    "label_policy": "no semantic labels or timestamps are inferred",
    "validation_group_policy": "all custom views retain the same conservative filming-session group",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def creation_time(row):
    metadata = row["source_ffprobe"]
    containers = [metadata.get("format", {})] + [s for s in metadata.get("streams", [])
                                                   if s.get("codec_type") == "video"]
    values = [c.get("tags", {}).get("creation_time") for c in containers]
    for value in values:
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                return parsed.timestamp()
        except ValueError:
            pass
    return None


def video_command(path, ffmpeg="ffmpeg"):
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-vf",
            f"setpts=PTS-STARTPTS,fps={PROTOCOL['sample_fps']},"
            "scale=192:108:force_original_aspect_ratio=decrease,"
            "pad=192:108:(ow-iw)/2:(oh-ih)/2,format=gray",
            "-threads", "2", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]


def native_features(path, ffmpeg="ffmpeg"):
    """Frames remain transient process memory; only numeric traces are returned."""
    result = subprocess.run(video_command(path, ffmpeg), capture_output=True, check=True)
    width, height = PROTOCOL["width"], PROTOCOL["height"]
    if len(result.stdout) % (width * height):
        raise ValueError("Incomplete native grayscale frame stream")
    frames = np.frombuffer(result.stdout, np.uint8).reshape(-1, height, width)
    if len(frames) < 2:
        return np.empty((0, 4)), np.empty(0), None, None
    # Estimate shared scene geometry from a temporal median, never export pixels.
    reference = np.median(frames[::max(1, len(frames) // 20)], axis=0).astype(np.uint8)
    points, descriptors = cv2.ORB_create(nfeatures=1000).detectAndCompute(reference, None)
    coordinates = np.array([p.pt for p in points], np.float32) if points else None
    rows = []
    for previous, current in zip(frames[:-1], frames[1:]):
        change = cv2.absdiff(previous, current).astype(np.float32)
        # Camera motion is approximately spatially coherent. Subtract the
        # median flow vector before computing action-sensitive residual energy.
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, .5, 2, 11, 2, 5, 1.1, 0)
        centered = flow - np.median(flow.reshape(-1, 2), axis=0)
        magnitude = np.linalg.norm(centered, axis=-1)
        rows.append([float(np.mean(change)), float(np.percentile(change, 90)),
                     float(np.percentile(magnitude, 90)), float(np.mean(change > 12))])
    # Midpoint times of adjacent sampled frames.
    times = (np.arange(len(rows), dtype=np.float64) + .5) / PROTOCOL["sample_fps"]
    return np.asarray(rows), times, coordinates, descriptors


def prepare_traces(values):
    values = np.log1p(np.asarray(values, dtype=float))
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("Motion traces must be finite with four channels")
    if not len(values):
        return values
    # Per-recording robust normalization is unsupervised and used for pairing
    # only. These traces are never features in an accuracy experiment.
    values = gaussian_filter1d(values, sigma=.75, axis=0)
    center = np.median(values, axis=0)
    scale = np.maximum(np.percentile(values, 85, axis=0) -
                       np.percentile(values, 15, axis=0), 1e-6)
    return np.clip((values - center) / scale, -3, 6)


def correlations(left, right):
    x = left - np.mean(left, axis=0)
    y = right - np.mean(right, axis=0)
    den = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
    return np.divide(np.sum(x * y, axis=0), den, out=np.zeros(4), where=den > 1e-10)


def _overlap(left, right, offset):
    # Both traces have the same FPS and start midpoint; positive offset means
    # right is already further into its recording at the same physical moment.
    left_start, right_start = max(0, -offset), max(0, offset)
    length = min(len(left) - left_start, len(right) - right_start)
    if length < 1:
        return left[:0], right[:0]
    return left[left_start:left_start + length], right[right_start:right_start + length]


def align_motion(left, right, fps=5):
    """Search motion alignment without consulting recording timestamps."""
    x, y = prepare_traces(left), prepare_traces(right)
    minimum = max(int(np.ceil(PROTOCOL["minimum_overlap_seconds"] * fps)),
                  int(np.ceil(PROTOCOL["minimum_shorter_overlap_fraction"] * min(len(x), len(y)))))
    blank = {"motion_score": 0., "offset_seconds": None, "overlap_seconds": 0.,
             "overlap_fraction": 0., "channel_correlations": [0.] * 4,
             "split_correlations": [0., 0.], "remote_peak_margin": 0.,
             "drift_seconds": None, "informative": False}
    if min(len(x), len(y)) < minimum or minimum < 2:
        return blank
    candidates = []
    for offset in range(-(len(x) - minimum), len(y) - minimum + 1):
        a, b = _overlap(x, y, offset)
        if len(a) < minimum:
            continue
        corr = correlations(a, b)
        # Median avoids one numerically strong feature deciding the pairing.
        score = float(np.median(corr))
        candidates.append((score, offset, corr, len(a)))
    candidates.sort(key=lambda row: (-row[0], abs(row[1]), row[1]))
    score, offset, corr, length = candidates[0]
    a, b = _overlap(x, y, offset)
    halfway = length // 2
    splits = [float(np.median(correlations(a[:halfway], b[:halfway]))),
              float(np.median(correlations(a[halfway:], b[halfway:])))]
    remote = [c[0] for c in candidates if abs(c[1] - offset) >= fps]
    # Estimate independent local realignments in the two halves. Their offset
    # difference is a diagnostic, not a calibrated camera-clock drift estimate.
    shifts = []
    for start, stop in ((0, halfway), (halfway, length)):
        local = []
        for delta in range(-fps, fps + 1):
            left_start = max(start, -delta)
            right_start = left_start + delta
            count = min(stop - left_start, stop - right_start)
            if count >= max(5, (stop - start) // 2):
                local.append((float(np.median(correlations(
                    a[left_start:left_start + count], b[right_start:right_start + count]))), delta))
        shifts.append(max(local, key=lambda row: (row[0], -abs(row[1])))[1])
    informative = bool(np.median(np.std(a, axis=0)) > .15 and
                       np.median(np.std(b, axis=0)) > .15)
    return {"motion_score": score, "offset_seconds": offset / fps,
            "overlap_seconds": length / fps, "overlap_fraction": length / min(len(x), len(y)),
            "channel_correlations": [float(v) for v in corr], "split_correlations": splits,
            "remote_peak_margin": score - max(remote) if remote else 0.,
            "drift_seconds": abs(shifts[1] - shifts[0]) / fps, "informative": informative}


def geometry_evidence(left, right):
    """RANSAC correspondence counts only; same background is not same event."""
    pa, da = left
    pb, db = right
    result = {"ratio_matches": 0, "ransac_inliers": 0, "inlier_fraction": 0.,
              "coverage_fraction": 0., "supporting_same_scene_geometry": False}
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return result
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [a for pair in matches if len(pair) == 2 for a, b in [pair] if a.distance < .70 * b.distance]
    result["ratio_matches"] = len(good)
    if len(good) < 8:
        return result
    xy_a = np.array([pa[m.queryIdx] for m in good])
    xy_b = np.array([pb[m.trainIdx] for m in good])
    cv2.setRNGSeed(42)
    _, mask = cv2.findHomography(xy_a, xy_b, cv2.RANSAC, 3.)
    if mask is None:
        return result
    selected = mask.ravel().astype(bool)
    count = int(selected.sum())
    coverage = min(float(cv2.contourArea(cv2.convexHull(points[selected]))) /
                   (PROTOCOL["width"] * PROTOCOL["height"])
                   for points in (xy_a, xy_b)) if count >= 3 else 0.
    result.update(ransac_inliers=count, inlier_fraction=count / len(good),
                  coverage_fraction=coverage,
                  supporting_same_scene_geometry=bool(count >= 12 and count / len(good) >= .4 and coverage >= .05))
    return result


def partial_assignment(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Assignment scores must be a finite matrix")
    rows, cols = scores.shape
    # Separate zero-valued dummy columns permit every left view to abstain.
    augmented = np.concatenate([scores, np.zeros((rows, rows))], axis=1)
    r, c = linear_sum_assignment(-augmented)
    return [(int(i), int(j)) for i, j in zip(r, c) if j < cols and scores[i, j] > 0]


def acceptance_reasons(candidate, margin, clock_residual, enough_clock_support):
    reasons = []
    if candidate["motion_score"] < PROTOCOL["automatic_motion_score"]:
        reasons.append("motion_correlation_below_threshold")
    if not candidate["informative"]:
        reasons.append("insufficient_motion_variation")
    if candidate["overlap_seconds"] < PROTOCOL["automatic_minimum_duration_seconds"]:
        reasons.append("overlap_too_short_for_automatic_pairing")
    if sum(v >= PROTOCOL["automatic_channel_correlation"] for v in candidate["channel_correlations"]) < 3:
        reasons.append("fewer_than_three_correlated_motion_channels")
    if min(candidate["split_correlations"]) < PROTOCOL["automatic_split_correlation"]:
        reasons.append("split_window_alignment_not_corroborated")
    if candidate["remote_peak_margin"] < PROTOCOL["automatic_remote_peak_margin"]:
        reasons.append("ambiguous_motion_offset")
    if candidate["drift_seconds"] is None or candidate["drift_seconds"] > PROTOCOL["automatic_maximum_drift_seconds"]:
        reasons.append("local_offset_inconsistency")
    if margin < PROTOCOL["automatic_assignment_margin"]:
        reasons.append("another_camera_candidate_has_similar_score")
    if not enough_clock_support or clock_residual is None:
        reasons.append("device_clock_bias_not_independently_supported")
    elif abs(clock_residual) > PROTOCOL["automatic_clock_residual_seconds"]:
        reasons.append("source_creation_time_disagrees_with_motion_alignment")
    return reasons


def metadata_reliability(records):
    """Audit whether creation tags can safely be assumed capture start times.

    A temporal gap smaller than both neighboring recording durations conflicts
    with sequential non-overlapping capture on the same claimed camera, whether
    creation_time denotes capture start or end. This is a warning about metadata
    semantics, not proof of a wrong clock or an incorrect user statement.
    """
    result = {}
    for suffix in (".mov", ".mp4"):
        ordered = sorted([r for r in records if r["source_suffix"] == suffix
                          and r["creation_time_unix"] is not None], key=lambda r: r["creation_time_unix"])
        conflicts = sum((b["creation_time_unix"] - a["creation_time_unix"]) <
                        min(a["duration_seconds"], b["duration_seconds"]) for a, b in zip(ordered, ordered[1:]))
        result[suffix.removeprefix(".")] = {
            "available_count": len(ordered), "adjacent_pairs": max(0, len(ordered) - 1),
            "adjacent_gaps_shorter_than_both_durations": conflicts,
            "creation_tag_capture_time_assumption_supported": bool(len(ordered) > 1 and not conflicts),
        }
    return result


def audit(preparation_path, private_dir, summary_path, ffmpeg="ffmpeg"):
    started = time.monotonic()
    preparation_path, private_dir = Path(preparation_path), Path(private_dir)
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation.get("complete") is not True:
        raise ValueError("Complete prepared-video manifest required")
    private_dir.mkdir(parents=True, exist_ok=True)
    records = []
    traces, geometries = {}, {}
    extractor_fingerprint = hashlib.sha256((inspect.getsource(native_features) +
        inspect.getsource(video_command) + cv2.__version__ + str(PROTOCOL["sample_fps"]) +
        str(PROTOCOL["width"]) + str(PROTOCOL["height"])).encode()).hexdigest()
    root = Path(preparation["output_dir"]).resolve()
    for row in preparation["videos"]:
        if row["output"]["stream_types"] != ["video"] or row["output"]["audio_streams"]:
            raise ValueError("Only manifest-verified silent prepared files may be decoded")
        path = (root / row["output_relative_path"]).resolve()
        if not path.is_relative_to(root) or digest(path) != row["output_sha256"]:
            raise ValueError("Prepared-file hash or containment mismatch")
        sha = row["source_sha256"]
        feature_path = private_dir / f"motion_{sha}.npz"
        cache_valid = False
        if feature_path.exists():
            with np.load(feature_path, allow_pickle=False) as cached:
                if str(cached["prepared_sha256"]) != row["output_sha256"]:
                    raise ValueError("Cached trace lineage mismatch")
                cache_valid = ("extractor_sha256" in cached and str(cached["extractor_sha256"]) == extractor_fingerprint)
                if cache_valid:
                    values, times = cached["values"], cached["times"]
                    points, descriptors = cached["points"], cached["descriptors"]
                    if not len(points):
                        points = descriptors = None
        if not cache_valid:
            values, times, points, descriptors = native_features(path, ffmpeg)
            np.savez_compressed(feature_path, values=values, times=times,
                                points=points if points is not None else np.empty((0, 2), np.float32),
                                descriptors=descriptors if descriptors is not None else np.empty((0, 32), np.uint8),
                                prepared_sha256=row["output_sha256"], extractor_sha256=extractor_fingerprint)
        traces[sha], geometries[sha] = values, (points, descriptors)
        records.append({"source_sha256": sha, "source_relative_path": row["source_relative_path"],
                        "source_suffix": Path(row["source_relative_path"]).suffix.lower(),
                        "prepared_sha256": row["output_sha256"], "prepared_relative_path": row["output_relative_path"],
                        "creation_time_unix": creation_time(row), "duration_seconds": row["source"]["duration_seconds"],
                        "motion_samples": len(times), "validation_group": GROUP})
    left = sorted([r for r in records if r["source_suffix"] == ".mov"], key=lambda r: r["source_sha256"])
    right = sorted([r for r in records if r["source_suffix"] == ".mp4"], key=lambda r: r["source_sha256"])
    if not left or not right:
        raise ValueError("Both MOV and MP4 source families required")
    candidates, scores = [], np.zeros((len(left), len(right)))
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            item = align_motion(traces[a["source_sha256"]], traces[b["source_sha256"]])
            item.update({"left_index": i, "right_index": j, "left_source_sha256": a["source_sha256"],
                         "right_source_sha256": b["source_sha256"],
                         "duration_difference_seconds": b["duration_seconds"] - a["duration_seconds"],
                         "creation_time_difference_seconds": b["creation_time_unix"] - a["creation_time_unix"]
                         if a["creation_time_unix"] is not None and b["creation_time_unix"] is not None else None,
                         "geometry": geometry_evidence(geometries[a["source_sha256"]], geometries[b["source_sha256"]])})
            scores[i, j] = max(0., item["motion_score"])
            candidates.append(item)
    lookup = {(c["left_index"], c["right_index"]): c for c in candidates}
    assignments = partial_assignment(scores)
    # Estimate inter-device clock bias only from strongly correlated, mutually
    # best motion pairs, excluding each tested pair itself when accepting it.
    clock_seeds = []
    for i, j in assignments:
        c = lookup[i, j]
        if (c["motion_score"] >= .70 and c["creation_time_difference_seconds"] is not None
                and c["offset_seconds"] is not None and np.argmax(scores[i]) == j and np.argmax(scores[:, j]) == i
                and min(c["split_correlations"]) >= .50):
            clock_seeds.append((i, j, c["creation_time_difference_seconds"] + c["offset_seconds"]))
    assigned_keys = set(assignments)
    for c in candidates:
        i, j = c["left_index"], c["right_index"]
        alternatives = [*np.delete(scores[i], j), *np.delete(scores[:, j], i)]
        margin = float(scores[i, j] - max(alternatives, default=0.))
        seeds = [value for si, sj, value in clock_seeds if (si, sj) != (i, j)]
        bias = float(np.median(seeds)) if len(seeds) >= 3 else None
        residual = (c["creation_time_difference_seconds"] + c["offset_seconds"] - bias
                    if bias is not None and c["creation_time_difference_seconds"] is not None
                    and c["offset_seconds"] is not None else None)
        reasons = acceptance_reasons(c, margin, residual, len(seeds) >= 3)
        if (i, j) not in assigned_keys:
            reasons.append("not_selected_by_one_to_one_assignment")
        c.update({"assigned": (i, j) in assigned_keys, "assignment_margin": margin,
                  "leave_pair_out_clock_bias_seconds": bias, "clock_residual_seconds": residual,
                  "status": "automatic_pair_requires_human_confirmation" if not reasons else "uncertain",
                  "human_verified": False, "abstention_reasons": reasons,
                  "confidence_interpretation": "fixed-threshold evidence score, not calibrated probability"})
    details = {"protocol": PROTOCOL, "preparation_manifest_sha256": digest(preparation_path),
               "records": records, "candidates": candidates,
               "assigned_candidates": [lookup[key] for key in assignments],
               "source_validation_group": GROUP, "labels_inferred": 0, "human_verified_pairs": 0}
    (private_dir / "camera_pairs_private.json").write_text(json.dumps(details, indent=2, allow_nan=False), encoding="utf-8")
    selected = [lookup[key] for key in assignments]
    accepted = [c for c in selected if c["status"] != "uncertain"]
    reliability = metadata_reliability(records)
    summary = {"schema_version": 1, "protocol": PROTOCOL,
               "implementation_sha256": digest(__file__), "extractor_sha256": extractor_fingerprint,
               "opencv_version": cv2.__version__,
               "source_manifest_sha256": digest(preparation_path), "source_file_count": len(records),
               "camera_family_counts": {"mov": len(left), "mp4": len(right)},
               "source_creation_time_available_count": sum(r["creation_time_unix"] is not None for r in records),
               "creation_metadata_reliability": reliability,
               "creation_time_warning": "Creation tags are not assumed to be capture starts; proximity alone never confirms a pair.",
               "all_cross_camera_candidates": len(candidates), "hungarian_candidate_count": len(selected),
               "automatic_pair_count": len(accepted), "human_verified_pair_count": 0,
               "assigned_uncertain_pair_count": len(selected) - len(accepted),
               "unmatched_view_count_after_automatic_abstention": len(records) - 2 * len(accepted),
               "all_sources_keep_single_validation_group": True, "validation_group_count": 1,
               "source_pixels_and_audio_shared_with_hosted_services": False,
               "audio_decoded": False, "labels_or_event_timestamps_inferred": False,
               "motion_score_summary_assigned": {"min": min(c["motion_score"] for c in selected),
                   "median": float(np.median([c["motion_score"] for c in selected])),
                   "max": max(c["motion_score"] for c in selected)},
               "assigned_abstention_reason_counts": dict(Counter(reason for c in selected for reason in c["abstention_reasons"])),
               "private_mapping_relative_path": str(private_dir / "camera_pairs_private.json"),
               "elapsed_seconds": time.monotonic() - started}
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", default="open/customVideos_prepared/preparation_manifest.json")
    parser.add_argument("--private-dir", default="artifacts/custom_camera_pairing_private")
    parser.add_argument("--summary", default="artifacts/custom_pairing_audit.json")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    result = audit(args.preparation, args.private_dir, args.summary, args.ffmpeg)
    print(json.dumps({key: result[key] for key in ("source_file_count", "automatic_pair_count",
          "human_verified_pair_count", "assigned_uncertain_pair_count", "elapsed_seconds")}))


if __name__ == "__main__":
    main()

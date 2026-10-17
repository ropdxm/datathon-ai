"""Numeric-only visual-cache diagnostics with training-owned PCA fits.

This tool does not open videos, captions or private notes. It does not fit a
classifier or use labels, filenames, recording length or annotation times as
predictors. Point times are read only after representation construction, for
sampling-coverage diagnostics. PCA never sees the current held-out fold.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datathon_ml import model as base
from datathon_ml import temporal


def summary(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not len(values):
        return {"n": 0}
    if not np.isfinite(values).all():
        raise ValueError("Diagnostic values must be finite")
    return {"n": int(len(values)), "min": float(values.min()),
            "p05": float(np.quantile(values, .05)),
            "median": float(np.median(values)), "mean": float(values.mean()),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def uniform_indices(length, limit):
    if length < 1 or limit < 1:
        raise ValueError("Positive sampling limits and nonempty timelines required")
    return np.unique(np.linspace(0, length - 1, min(length, limit)).astype(int))


def nearest_anchor_gaps(timestamps, anchors):
    times = np.asarray(timestamps, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)
    if not len(anchors):
        return np.empty(0, dtype=float)
    return np.abs(times[:, None] - anchors[None, :]).min(axis=0)


def _read_records(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("videos"))
    if not isinstance(payload, list):
        raise ValueError("Manifest needs records/videos list")
    return base._canonical_records(payload)


def _retained(ratios, n):
    return float(ratios[:n].sum()) if len(ratios) >= n else None


def _threshold_rank(ratios, threshold):
    cumulative = np.cumsum(ratios)
    hits = np.flatnonzero(cumulative >= threshold)
    return int(hits[0] + 1) if len(hits) else None


def audit(records, feature_dir, folds_from, *, components=64, windows=24, threads=2):
    if min(components, windows, threads) < 1:
        raise ValueError("Component/window/thread limits must be positive")
    records = base._canonical_records(records)
    frozen = json.loads(Path(folds_from).read_text(encoding="utf-8-sig"))
    fold_count = len({row["fold"] for row in frozen["predictions"]})
    splits, provenance = temporal.frozen_outer_splits(records, folds_from,
        temporal.TemporalConfig(outer_folds=fold_count))
    data = {}
    norms, gaps, temporal_cosines, anchor_gaps, per_source_samples = [], [], [], [], []
    pair_cosines = {}
    expected_meta = None
    final_frame_anchor_outside = 0
    annotation_count = 0
    for record in records:
        key = record["sha256"]
        features, times, metadata = base.load_features(Path(feature_dir) / f"{key}.npz")
        if features.shape[1:] != (4, 512):
            raise ValueError("This audit expects the original four-view 512-d CLIP cache")
        if expected_meta is None:
            expected_meta = metadata
        if metadata != expected_meta:
            raise ValueError("Mixed extractor configurations")
        f = base._normalise(features)
        norms.extend(np.linalg.norm(features, axis=-1).ravel().tolist())
        gaps.extend(np.diff(times).tolist())
        per_source_samples.append(len(times))
        if len(times) > 1:
            temporal_cosines.extend(np.sum(f[1:] * f[:-1], axis=-1).ravel().tolist())
        names = metadata.get("views", ["full", "top", "middle", "bottom"])
        for i in range(features.shape[1]):
            for j in range(i + 1, features.shape[1]):
                name = f"{names[i]}__{names[j]}"
                pair_cosines.setdefault(name, []).extend(np.sum(f[:, i] * f[:, j], axis=-1).tolist())
        # Deterministic descriptors are constructed before inspecting event points.
        desc = base.window_features(features, times)
        indices = uniform_indices(len(times), windows)
        data[key] = desc[indices]
        anchors = record.get("event_times", [])
        anchor_gaps.extend(nearest_anchor_gaps(times, anchors).tolist())
        annotation_count += len(anchors)
        final_frame_anchor_outside += sum(float(a) > float(times[-1]) for a in anchors)
    folds = []
    for fold, (train_indices, test_indices) in enumerate(splits):
        train_keys = [records[i]["sha256"] for i in train_indices]
        test_keys = [records[i]["sha256"] for i in test_indices]
        if set(train_keys) & set(test_keys):
            raise AssertionError("PCA owner leakage")
        x = np.concatenate([data[key] for key in train_keys])
        test = np.concatenate([data[key] for key in test_keys])
        with threadpool_limits(limits=threads):
            scaler = StandardScaler().fit(x)
            z = scaler.transform(x)
            pca = PCA(n_components=min(components, len(x) - 1, x.shape[1]),
                svd_solver="randomized", iterated_power=2, random_state=42).fit(z)
            val = scaler.transform(test)
            centered_val = val - pca.mean_
            reconstructed = pca.inverse_transform(pca.transform(val))
        ratios = pca.explained_variance_ratio_
        baseline_mse = float(np.mean(centered_val ** 2))
        mse = float(np.mean((val - reconstructed) ** 2))
        folds.append({"fold": fold, "pca_fit_owner_sha256": train_keys,
            "test_sha256": test_keys, "train_test_owner_overlap": 0,
            "training_rows": int(len(x)), "held_out_rows": int(len(test)),
            "input_dimensions": int(x.shape[1]), "fitted_components": int(len(ratios)),
            "training_variance_retained": {"8": _retained(ratios, 8),
                "16": _retained(ratios, 16), "32": _retained(ratios, 32),
                "64": _retained(ratios, 64)},
            "rank_for_80pct_within_fitted_components": _threshold_rank(ratios, .8),
            "rank_for_90pct_within_fitted_components": _threshold_rank(ratios, .9),
            "variance_ratio": ratios.astype(float).tolist(),
            "scaler_training_raw_variance": summary(scaler.var_),
            "dimensions_with_raw_variance_below_1e_8": int(np.sum(scaler.var_ < 1e-8)),
            "held_out_reconstruction_mse_standardized": mse,
            "held_out_centered_baseline_mse": baseline_mse,
            "held_out_relative_reconstruction_error": mse / baseline_mse if baseline_mse else None,
            "labels_used_to_fit_projection": 0, "anchors_used_to_fit_projection": 0})
        print(json.dumps({"completed_pca_fold": fold + 1,
            "variance_retained_32": _retained(ratios, 32),
            "variance_retained_all_fitted": float(ratios.sum())}), flush=True)
    result = {"status": "numeric_preprocessing_audit_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": len(records), "extractor_config": expected_meta,
        "feature_norm": summary(norms), "sample_gap_sec": summary(gaps),
        "samples_per_source_diagnostic_only": summary(per_source_samples),
        "adjacent_frame_cosine": summary(temporal_cosines),
        "view_pair_cosine": {key: summary(value) for key, value in pair_cosines.items()},
        "provided_point_nearest_sample_gap_sec": summary(anchor_gaps),
        "provided_points": annotation_count,
        "provided_points_after_last_decoded_sample": int(final_frame_anchor_outside),
        "provided_points_with_nearest_gap_over_0_5sec": int(np.sum(np.asarray(anchor_gaps) > .5)),
        "pca_protocol": {"sampling": "uniform max distinct temporal rows per training source",
            "max_windows_per_source": windows, "max_components": components,
            "scaler": "training_only_standard_scaler", "seed": 42,
            "fit_inputs_exclude_labels_points_filenames_durations": True,
            "representation": "unchanged_model_window_features_radii_1_2_4_seconds"},
        "fold_source": provenance, "pca_folds": folds,
        "limitations": [
            "Retained variance and reconstruction error do not measure violation separability or prove visual detail is adequate.",
            "PCA is fit on uniformly sampled original training windows, not the historical supervised-point sample distribution.",
            "A null variance-rank means the requested threshold was not reached within fitted components; it is not the full data rank.",
            "High adjacent or cross-view cosine does not prove the views or actions are redundant.",
            "Point-to-sample gaps audit timestamp coverage only; brief actions between samples can still be missed.",
            "The audited folds are repeatedly inspected development folds, not an independent test."]}
    return result


def self_test():
    assert uniform_indices(3, 24).tolist() == [0, 1, 2]
    assert uniform_indices(30, 24).size == 24
    assert np.allclose(nearest_anchor_gaps([0., .5, 1.], [.25, .99, 1.4]), [.25, .01, .4])
    assert nearest_anchor_gaps([0.], []).size == 0
    assert _retained(np.array([.5, .3]), 3) is None
    assert _threshold_rank(np.array([.5, .3]), .8) == 2
    assert _threshold_rank(np.array([.5, .3]), .9) is None
    assert summary([]) == {"n": 0}
    print(json.dumps({"self_test": "passed", "checks": 8}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-manifest", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--folds-from", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--max-windows-per-source", type=int, default=24)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if any(getattr(args, key) is None for key in ("original_manifest", "features", "folds_from", "output")):
        parser.error("--original-manifest, --features, --folds-from and --output are required")
    report = audit(_read_records(args.original_manifest), args.features, args.folds_from,
        components=args.pca_components, windows=args.max_windows_per_source, threads=args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "sources": report["sources"],
        "sample_gap_sec": report["sample_gap_sec"],
        "provided_point_nearest_sample_gap_sec": report["provided_point_nearest_sample_gap_sec"],
        "pca_32_variance_retained_by_fold": [row["training_variance_retained"]["32"] for row in report["pca_folds"]]},
        allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

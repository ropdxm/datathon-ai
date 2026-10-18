"""Numeric train/validation diagnostics; never opens test feature caches.

These fixed diagnostic probes are not candidates and must not select or retune
the active experiment. No media, descriptions, or per-source predictions leave
the native process. Source-level grouped cross-validation owns its transforms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datathon_ml.model import load_features


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = np.asarray(values, float)
    return {"count": len(values), "min": float(values.min()), "median": float(np.median(values)),
            "mean": float(values.mean()), "max": float(values.max())} if len(values) else None


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def probe(x, y, groups, sample_weight=None):
    output = np.zeros(len(y))
    fit_ownership = []
    for fit, held in StratifiedGroupKFold(3, shuffle=True, random_state=42).split(x, y, groups):
        if set(groups[fit]) & set(groups[held]):
            raise ValueError("Probe ownership leakage")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight="balanced", max_iter=1000))
        args = {"logisticregression__sample_weight": sample_weight[fit]} if sample_weight is not None else {}
        model.fit(x[fit], y[fit], **args)
        output[held] = model.predict_proba(x[held])[:, 1]
        fit_ownership.append({"fit_groups": len(set(groups[fit])), "held_groups": len(set(groups[held]))})
    return {"roc_auc": float(roc_auc_score(y, output)),
            "balanced_accuracy_at_fixed_half": float(balanced_accuracy_score(y, output >= .5)),
            "accuracy_at_fixed_half": float(accuracy_score(y, output >= .5)), "fold_group_counts": fit_ownership}, output


def analyze(records, split, feature_dir, representation):
    train_ids, val_ids, test_ids = set(split["train"]), set(split["validation"]), set(split["test"])
    allowed = train_ids | val_ids
    by_id = {r["sha256"]: r for r in records if r["sha256"] in allowed}
    if len(by_id) != 46 or len(train_ids) != 34 or len(val_ids) != 12 or allowed & test_ids:
        raise ValueError("Requires unchanged34train12validation ownership and disjoint test")
    entries, sizes, ranges, norms, gaps = {}, [], [], [], []
    for key in sorted(allowed):
        if key in test_ids or key not in allowed:
            raise ValueError("Refuse non-owned feature read")
        features, times, _ = load_features(Path(feature_dir) / (key + ".npz"))
        x = features[:, 0, :].astype(float)
        if representation == "clip512":
            norms.extend(np.linalg.norm(x, axis=1).tolist())
            x = unit(x)
        ranges.extend([float(x.min()), float(x.max())])
        entries[key] = (x, times)
        sizes.append(len(times))
        gaps.extend(np.diff(times).tolist())
    train = [by_id[k] for k in sorted(train_ids)]
    val = [by_id[k] for k in sorted(val_ids)]
    means = np.stack([entries[r["sha256"]][0].mean(axis=0) for r in train])
    val_means = np.stack([entries[r["sha256"]][0].mean(axis=0) for r in val])
    domain = np.asarray([int(r["domain"] == "custom") for r in train])
    val_domain = np.asarray([int(r["domain"] == "custom") for r in val])
    labels = np.asarray([r["label"] for r in train])
    groups = np.asarray([r["group_id"] for r in train])
    domain_cv, _ = probe(means, domain, groups)
    domain_model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight="balanced", max_iter=1000))
    domain_model.fit(means, domain)
    domain_val = domain_model.predict_proba(val_means)[:, 1]
    centroid = means[domain == 1].mean(axis=0) - means[domain == 0].mean(axis=0)
    scale = np.maximum(means.std(axis=0), 1e-6)
    cosine = unit(val_means) @ unit(means).T
    nearest = cosine.argmax(axis=1)
    coverage = {}
    for d, name in ((0, "original"), (1, "custom")):
        selected = val_domain == d
        coverage[name] = {"sources": int(selected.sum()),
            "nearest_training_cosine": summary(cosine.max(axis=1)[selected]),
            "nearest_training_same_domain_fraction": float(np.mean(domain[nearest[selected]] == d)),
            "nearest_training_same_label_fraction": float(np.mean(labels[nearest[selected]] ==
                np.asarray([r["label"] for r in val])[selected]))}
    point_x, point_y, point_groups, point_owners, point_gaps = [], [], [], [], []
    positive_owner_means, normal_owner_means = [], []
    missed_anchors = 0
    for r in train:
        x, times = entries[r["sha256"]]
        if r["label"]:
            mask = np.zeros(len(times), bool)
            for point in r["event_times"]:
                distance = np.abs(times - point)
                point_gaps.append(float(distance.min()))
                mask |= distance <= 1.
                if distance.min() <= .5:
                    mask[int(distance.argmin())] = True
                if not np.any(distance <= 1.):
                    missed_anchors += 1
            indices = np.flatnonzero(mask)
        else:
            indices = np.unique(np.linspace(0, len(times) - 1, min(16, len(times))).astype(int))
        if not len(indices):
            continue
        (positive_owner_means if r["label"] else normal_owner_means).append(x[indices].mean(axis=0))
        point_x.append(x[indices]); point_y.extend([r["label"]] * len(indices))
        point_groups.extend([r["group_id"]] * len(indices)); point_owners.extend([r["sha256"]] * len(indices))
    point_y, point_groups, point_owners = np.asarray(point_y), np.asarray(point_groups), np.asarray(point_owners)
    weights = np.asarray([1. / np.sum(point_owners == owner) for owner in point_owners])
    weights *= len(weights) / weights.sum()
    point_cv, oof = probe(np.concatenate(point_x), point_y, point_groups, weights)
    point_cv["window_class_counts"] = {"normal": int((point_y == 0).sum()),
                                      "positive_anchor": int((point_y == 1).sum())}
    point_cv["positive_anchor_oof_score"] = summary(oof[point_y == 1])
    point_cv["normal_window_oof_score"] = summary(oof[point_y == 0])
    owner_y, owner_scores = [], []
    for owner in sorted(set(point_owners)):
        mask = point_owners == owner
        owner_y.append(int(point_y[mask][0])); owner_scores.append(float(oof[mask].mean()))
    point_cv["source_mean_roc_auc"] = float(roc_auc_score(owner_y, owner_scores))
    owner_y, owner_scores = np.asarray(owner_y), np.asarray(owner_scores)
    point_cv["source_balanced_positive_minus_normal_oof_score_margin"] = float(
        owner_scores[owner_y == 1].mean() - owner_scores[owner_y == 0].mean())
    pos_mean, normal_mean = np.mean(positive_owner_means, axis=0), np.mean(normal_owner_means, axis=0)
    result = {"feature_dimension": means.shape[1], "view": "first_full_view_only" if representation == "clip512" else "single_semantic_view",
        "source_caches_opened": len(entries), "test_feature_caches_opened": 0,
        "sample_count_distribution": summary(sizes), "sample_gap_seconds": summary(gaps),
        "feature_value_min": min(ranges), "feature_value_max": max(ranges),
        "input_unit_norm_distribution": summary(norms),
        "training_grouped_domain_probe": domain_cv,
        "validation_domain_probe": {"roc_auc": float(roc_auc_score(val_domain, domain_val)),
            "balanced_accuracy_at_fixed_half": float(balanced_accuracy_score(val_domain, domain_val >= .5))},
        "domain_centroid_standardized_rms_difference": float(np.sqrt(np.mean((centroid / scale) ** 2))),
        "validation_nearest_training_coverage": coverage,
        "training_point_positive_vs_normal_probe": point_cv,
        "training_point_anchor_gap_seconds": summary(point_gaps), "training_anchors_without_sample_within_one_second": missed_anchors,
        "source_balanced_positive_normal_centroid_cosine": float(unit(pos_mean[None])[0] @ unit(normal_mean[None])[0]),
        "positive_training_owner_count": len(positive_owner_means), "normal_training_owner_count": len(normal_owner_means),
        "probe_protocol": "fixedC0.1linear_balanced_logistic_train_owned_standardization_3groupfolds_seed42_no_modelselection"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/reviewed_dataset")
    parser.add_argument("--features", type=Path, default=ROOT / "artifacts/reviewed_features")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reviewed_representation_audit.json")
    args = parser.parse_args()
    started = time.perf_counter()
    manifest, split_path = args.dataset / "manifest.json", args.dataset / "split.json"
    records, split = json.loads(manifest.read_text()), json.loads(split_path.read_text())
    with threadpool_limits(limits=2):
        results = {name: analyze(records, split, args.features / folder, name)
                   for name, folder in (("clip512", "clip"), ("semantic18", "semantic"))}
    report = {"schema_version": 1, "implementation_sha256": digest(__file__),
        "manifest_sha256": digest(manifest), "split_sha256": digest(split_path),
        "train_source_count": 34, "validation_source_count": 12, "test_feature_files_opened": 0,
        "model_scores_or_model_artifacts_opened": 0, "media_or_descriptions_opened": False,
        "diagnostic_only_not_model_selection": True, "representations": results,
        "limitations": ["Diagnostics do not establish causality or independent-session performance.",
                        "Domain separation can reflect camera background or other visual factors; no semantic cause is inferred.",
                        "Training-window labels use sparse positive anchors and confirmed normal recordings only.",
                        "High mean-feature cosine proximity does not establish duplicate footage or matching event semantics.",
                        "No test features or test scores were inspected and no training source was changed."],
        "elapsed_seconds": time.perf_counter() - started}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

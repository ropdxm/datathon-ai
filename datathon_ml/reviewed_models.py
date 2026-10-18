"""Regularized frozen-feature models with a separately locked test partition.

All candidates and thresholds use train/validation only. Test feature files are
first opened after the selected model and its split have been saved and hashed.
Only verified custom labels and explicit scene groups are accepted; no pseudo
targets, filenames, absolute clocks or audio become model inputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from . import model as base
from . import scene_temporal as scene


@dataclass(frozen=True)
class ReviewedConfig:
    seed: int = 42
    threads: int = 2
    primary_metric: str = "accuracy"
    include_catboost: bool = True
    pooled_pca_components: int = 16
    point_pca_components: int = 32
    max_normal_windows: int = 32
    positive_radius: float = 1.
    context_radii: tuple[float, float] = (1., 3.)
    max_gap_seconds: float = 1.5
    top_k: int = 3
    event_merge_seconds: float = 4.
    max_events: int = 5

    def __post_init__(self):
        if self.primary_metric not in {"accuracy", "balanced_accuracy", "f1", "f2"}:
            raise ValueError("Unsupported selection metric")
        if min(self.threads, self.pooled_pca_components, self.point_pca_components,
               self.max_normal_windows, self.top_k, self.max_events) < 1:
            raise ValueError("Component, sampling, event and thread limits must be positive")
        if not self.context_radii or any(not np.isfinite(r) or r <= 0 for r in self.context_radii):
            raise ValueError("Context radii must be finite and positive")
        if any(not np.isfinite(r) or r < 0 for r in (self.positive_radius, self.max_gap_seconds, self.event_merge_seconds)):
            raise ValueError("Time values must be finite and nonnegative")


def candidate_grid(config):
    """Twenty-six declared candidates; no test score changes this grid."""
    result = []
    for preprocessing, cs in (("raw", (.01, .1, 1., 10.)), ("pca", (.1, 1.))):
        for c in cs:
            result.append({"head": "pooled", "classifier": "logistic", "preprocessing": preprocessing, "C": c, "aggregation": "pooled"})
    for preprocessing in ("raw", "pca"):
        for c in (.1, 1., 10.):
            result.append({"head": "pooled", "classifier": "rbf_svm", "preprocessing": preprocessing, "C": c, "aggregation": "pooled"})
    if config.include_catboost:
        for depth, l2 in ((2, 10.), (3, 30.)):
            result.append({"head": "pooled", "classifier": "catboost", "preprocessing": "raw", "depth": depth,
                "l2_leaf_reg": l2, "iterations": 200, "aggregation": "pooled"})
    for classifier, cs in (("logistic", (.01, .1, 1.)), ("rbf_svm", (.1, 1., 10.))):
        for c in cs:
            for aggregation in ("max", "topk"):
                result.append({"head": "point", "classifier": classifier, "preprocessing": "pca", "C": c, "aggregation": aggregation})
    return result


def _candidate_id(candidate):
    return scene._digest(candidate)[:12]


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    return {"reviewed_models": _hash_file(__file__), "scene_temporal": _hash_file(scene.__file__),
            "model": _hash_file(base.__file__), "temporal": _hash_file(scene.temporal.__file__)}


def _write(path, value):
    scene._write(Path(path), value)


def _supervision_hash(records):
    return scene._owner_manifest(records)["sha256"]


def _split_payload(split):
    return json.loads(Path(split).read_text(encoding="utf-8")) if isinstance(split, (str, Path)) else split


def _verified_records(records, originals, split):
    """Keep true scene verification separate from a camera-exclusion protocol."""
    protocol = _split_payload(split).get("protocol", {})
    camera_exclusive = protocol.get("type") == "camera_exclusive_source_overlap_development"
    if not camera_exclusive:
        return scene.verified_records(records, originals)
    selected = str(protocol.get("selected_camera", "")).lower().lstrip(".")
    grouping_digest = protocol.get("grouping_map_sha256", "")
    if (selected not in {"mov", "mp4"} or protocol.get("opposite_camera_excluded_globally") is not True
            or protocol.get("grouping_basis") != "native_duplicate_overlap_prevention_not_human_scene_identity"
            or protocol.get("family_selection_basis") != "largest_usable_label_coverage_before_model_scores"
            or not isinstance(grouping_digest, str) or len(grouping_digest) != 64
            or any(char not in "0123456789abcdef" for char in grouping_digest)):
        raise ValueError("Camera-exclusive development needs its explicit frozen exclusion, selection and overlap-map provenance")
    accepted = []
    for source in records:
        row = dict(source)
        key = row.get("sha256")
        if not isinstance(key, str) or len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("Invalid supervised source SHA-256")
        if row.get("label") not in (0, 1) or isinstance(row.get("label"), bool):
            raise ValueError("Missing binary target")
        if row.get("label_status") in {"machine_proposal", "supplied_unverified", "unlabeled", "pseudo", "pseudo_label"}:
            raise ValueError("Unverified/pseudo targets cannot enter this experiment")
        if key in originals:
            if row["label"] != originals[key]:
                raise ValueError("Original target differs from historical reference")
        else:
            if row.get("label_status") != "human_verified" or row.get("evaluation_ground_truth_eligible") is not True:
                raise ValueError("Custom targets must be explicitly human verified")
            if str(row.get("source_family", "")).lower().lstrip(".") != selected:
                raise ValueError("An opposite/unknown custom camera family entered the exclusive protocol")
            if row["label"] == 1 and (row.get("event_times_verified") is not True or not row.get("event_times")):
                raise ValueError("Custom positive timestamps must be explicitly verified")
        if not str(row.get("group_id", "")).strip():
            raise ValueError("Explicit source-overlap group is required")
        accepted.append(row)
    result = base._canonical_records(accepted)
    if not result:
        raise ValueError("No verified supervised sources")
    return sorted(result, key=lambda row: row["sha256"])


def _feature_sets(feature_dir):
    if isinstance(feature_dir, dict):
        if not feature_dir or any(not isinstance(name, str) or not name.strip() for name in feature_dir):
            raise ValueError("Feature-set names must be nonempty")
        return {name: Path(path) for name, path in feature_dir.items()}
    return {"primary": Path(feature_dir)}


def validate_split(records, split):
    """Validate full source coverage, scene separation and shared-parent lineage."""
    split = _split_payload(split)
    known = {row["sha256"]: row for row in records}
    partitions, used = {}, set()
    for name in ("train", "validation", "test"):
        values = split.get(name)
        if not isinstance(values, list) or not values or any(not isinstance(key, str) for key in values):
            raise ValueError("Every train/validation/test partition needs a nonempty source-SHA list")
        if len(set(values)) != len(values) or used & set(values):
            raise ValueError("Train/validation/test source sets overlap or repeat")
        if set(values) - set(known):
            raise ValueError("Split contains unknown source SHA")
        partitions[name] = [known[key] for key in sorted(values)]
        used.update(values)
    if used != set(known):
        raise ValueError("Explicit split must cover every selected supervised recording exactly once")
    owners, lineage_owners = {}, {}
    for name, rows in partitions.items():
        for row in rows:
            group = row["group_id"]
            if group in owners and owners[group] != name:
                raise ValueError("Declared scene/camera group crosses train/validation/test")
            owners[group] = name
            lineage = {row["sha256"], *row.get("lineage_sha256s", [])}
            for field in ("source_sha256", "parent_sha256", "original_sha256"):
                if row.get(field):
                    lineage.add(row[field])
            for key in lineage:
                if key in lineage_owners and lineage_owners[key] != name:
                    raise ValueError("Related source/transcode lineage crosses partitions")
                lineage_owners[key] = name
    for name in ("train", "validation"):
        if {row["label"] for row in partitions[name]} != {0, 1}:
            raise ValueError("Training and validation each need both binary classes")
    frozen = {name: sorted(row["sha256"] for row in partitions[name]) for name in partitions}
    frozen["protocol"] = split.get("protocol", {})
    return partitions, {**frozen, "sha256": scene._digest(frozen)}


def _context_features(x, times, config):
    """Fixed seconds of appearance, variation and ordering, separately per gap."""
    starts = np.r_[0, np.flatnonzero(np.diff(times) > config.max_gap_seconds) + 1, len(times)]
    output = []
    for start, stop in zip(starts[:-1], starts[1:]):
        values, t = x[start:stop], times[start:stop]
        prefix = np.r_[np.zeros_like(values[:1]), np.cumsum(values, axis=0)]
        squares = np.r_[np.zeros_like(values[:1]), np.cumsum(values * values, axis=0)]
        changes = np.r_[np.zeros_like(values[:1]), np.abs(np.diff(values, axis=0))]
        change_prefix = np.r_[np.zeros_like(values[:1]), np.cumsum(changes, axis=0)]
        parts = [values]
        for radius in config.context_radii:
            left, right = np.searchsorted(t, t - radius, side="left"), np.searchsorted(t, t + radius, side="right")
            count = (right - left)[:, None]
            mean = (prefix[right] - prefix[left]) / count
            variance = np.maximum((squares[right] - squares[left]) / count - mean * mean, 0)
            change = (change_prefix[right] - change_prefix[np.minimum(left + 1, right)]) / np.maximum(count - 1, 1)
            parts.extend((mean, np.sqrt(variance), change))
        radius = max(config.context_radii)
        centers = np.arange(len(t))
        left, right = np.searchsorted(t, t - radius, side="left"), np.searchsorted(t, t + radius, side="right")
        before = (prefix[centers + 1] - prefix[left]) / (centers + 1 - left)[:, None]
        after = (prefix[right] - prefix[centers]) / (right - centers)[:, None]
        parts.append(after - before)
        output.append(np.concatenate(parts, axis=1).astype(np.float32))
    return np.concatenate(output)


def _entry(feature_path, config):
    features, times, metadata = base.load_features(feature_path)
    x = scene._visual_rows(features, metadata)
    continuous = np.diff(times) <= config.max_gap_seconds
    differences = np.abs(np.diff(x, axis=0))[continuous]
    change = differences.mean(axis=0) if len(differences) else np.zeros(x.shape[1], dtype=np.float32)
    pooled = np.r_[x.mean(axis=0), x.std(axis=0), x.max(axis=0), change][None].astype(np.float32)
    point = _context_features(x, times, config)
    return {"pooled": pooled, "point": point, "timestamps": times,
        "feature_shape": tuple(features.shape[1:]), "feature_protocol": scene._feature_protocol(metadata)}


def _prepare(records, feature_dir, config, expected=None):
    entries = {}
    for row in records:
        entry = _entry(Path(feature_dir) / f"{row['sha256']}.npz", config)
        current = {"feature_shape": entry["feature_shape"], "feature_protocol": entry["feature_protocol"]}
        if expected is None:
            expected = current
        if current != expected:
            raise ValueError("Mixed visual encoder, state column order or sampling protocols")
        entries[row["sha256"]] = entry
    return entries, expected


def _training_matrix(records, entries, head, config):
    arrays, labels, owners = [], [], []
    for row in records:
        key = row["sha256"]
        values = entries[key][head]
        if head == "pooled":
            indices = np.array([0])
        elif row["label"] == 0:
            indices = np.unique(np.linspace(0, len(values) - 1, min(len(values), config.max_normal_windows)).astype(int))
        else:
            times = entries[key]["timestamps"]
            mask = np.zeros(len(times), dtype=bool)
            for point in row["event_times"]:
                distance = np.abs(times - point)
                mask |= distance <= config.positive_radius
                closest = int(distance.argmin())
                if distance[closest] <= max(config.positive_radius, .5):
                    mask[closest] = True
            indices = np.flatnonzero(mask)
        if not len(indices):
            continue  # Positive bags without a visible sampled anchor remain unlabeled.
        arrays.append(values[indices])
        labels.extend([row["label"]] * len(indices))
        owners.extend([key] * len(indices))
    if not arrays or set(labels) != {0, 1}:
        raise ValueError("This supervised head needs usable training owners in both classes")
    y, owners = np.asarray(labels), np.asarray(owners)
    weights = np.zeros(len(y), dtype=np.float64)
    for label in (0, 1):
        class_owners = set(owners[y == label])
        for owner in class_owners:
            idx = owners == owner
            weights[idx] = .5 / (len(class_owners) * int(idx.sum()))
    weights *= len(weights)  # Mean weight one keeps fixed C values comparable.
    return {"x": np.concatenate(arrays), "y": y, "weights": weights,
        "fit_sha256": sorted(set(owners)), "row_count": len(y),
        "supervision_owner_sha256": _supervision_hash([row for row in records if row["sha256"] in set(owners)])}


def _fit_preprocessing(training, preprocessing, head, config):
    scaler = StandardScaler().fit(training["x"])
    z = scaler.transform(training["x"])
    pca = None
    if preprocessing == "pca":
        count = config.pooled_pca_components if head == "pooled" else config.point_pca_components
        pca = PCA(n_components=min(count, z.shape[0] - 1, z.shape[1]), svd_solver="randomized",
                  iterated_power=2, random_state=config.seed).fit(z)
    return {"scaler": scaler, "pca": pca, "fit_sha256": training["fit_sha256"],
        "fit_row_count": training["row_count"], "supervision_owner_sha256": training["supervision_owner_sha256"]}


def _transform(values, preprocessing):
    z = preprocessing["scaler"].transform(values)
    return preprocessing["pca"].transform(z) if preprocessing["pca"] is not None else z


def _classifier(candidate, config):
    if candidate["classifier"] == "logistic":
        return LogisticRegression(C=candidate["C"], solver="lbfgs", max_iter=2000, random_state=config.seed)
    if candidate["classifier"] == "rbf_svm":
        return SVC(C=candidate["C"], kernel="rbf", gamma="scale", random_state=config.seed)
    import catboost
    if catboost.__version__ != "1.2.10":
        raise ValueError("Install the pinned optional catboost==1.2.10 before this declared comparison")
    return catboost.CatBoostClassifier(iterations=candidate["iterations"], depth=candidate["depth"],
        l2_leaf_reg=candidate["l2_leaf_reg"], learning_rate=.03, loss_function="Logloss",
        boosting_type="Plain", bootstrap_type="No", random_strength=0., border_count=32,
        random_seed=config.seed, thread_count=config.threads, task_type="CPU",
        allow_writing_files=False, verbose=False)


def _scores(classifier, values, candidate):
    if candidate["classifier"] == "catboost":
        return np.asarray(classifier.predict_proba(values)[:, 1], dtype=np.float64)
    return expit(np.asarray(classifier.decision_function(values), dtype=np.float64))


def _metric_config(config):
    return base.TrainingConfig(primary_metric=config.primary_metric, top_k=config.top_k,
        smoothing_seconds=0., event_merge_seconds=config.event_merge_seconds, max_events=config.max_events)


def _predict(classifier, preprocessing, candidate, entry, threshold, config):
    windows = _scores(classifier, _transform(entry[candidate["head"]], preprocessing), candidate)
    if candidate["aggregation"] == "pooled":
        score = float(windows[0])
    elif candidate["aggregation"] == "max":
        score = float(windows.max())
    else:
        score = float(np.sort(windows)[-min(config.top_k, len(windows)):].mean())
    predicted = int(score >= threshold)
    events = []
    if candidate["head"] == "point" and predicted:
        times = entry["timestamps"]
        bounds = np.r_[0, np.flatnonzero(np.diff(times) > config.max_gap_seconds) + 1, len(times)]
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if windows[start:stop].max() >= threshold:
                events.extend(base._events(windows[start:stop], times[start:stop], threshold, _metric_config(config)))
        events = sorted(sorted(events, key=lambda e: e["score"], reverse=True)[:config.max_events], key=lambda e: e["timestamp_sec"])
        events = [{"type": "violation", "timestamp_sec": event["timestamp_sec"], "timecode": event["timecode"],
                   "score": event["score"]} for event in events]
    return {"score": score, "prediction": predicted, "threshold": float(threshold),
        "verdict": "violation" if predicted else "no_violation", "events": events,
        "localization_available": candidate["head"] == "point", "score_calibrated": False,
        "window_scores": windows.tolist() if candidate["head"] == "point" else [],
        "window_timestamps": entry["timestamps"].astype(float).tolist() if candidate["head"] == "point" else []}


def _rows(records, entries, classifier, preprocessing, candidate, threshold, config, originals):
    return [{"sha256": row["sha256"], "group_id": row["group_id"], "label": row["label"],
             "event_times": row["event_times"], "domain": "original" if row["sha256"] in originals else "custom",
             **_predict(classifier, preprocessing, candidate, entries[row["sha256"]], threshold, config)} for row in records]


def _metrics(rows):
    result = base.classification_metrics([row["label"] for row in rows], [row["score"] for row in rows], [row["prediction"] for row in rows])
    result["localization"] = base._localization_metrics(rows) if all(row["localization_available"] for row in rows) else None
    grouped = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(int(row["label"] == row["prediction"]))
    result["mean_group_accuracy"] = float(np.mean([np.mean(values) for values in grouped.values()]))
    result["groups"] = len(grouped)
    n, successes = len(rows), sum(row["label"] == row["prediction"] for row in rows)
    z, proportion = 1.959963984540054, successes / n
    denominator = 1 + z * z / n
    middle = (proportion + z * z / (2 * n)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    result["accuracy_wilson_95pct"] = [max(0., float(middle - radius)), min(1., float(middle + radius))]
    result["confidence_interval_note"] = "Descriptive recording-level interval; related recording groups reduce effective sample size."
    return result


LIMITATIONS = [
    "A locked split protects this experiment's test from model selection; original data were repeatedly inspected in previous development.",
    "Grouping is only as strong as the explicit protocol; excluding the opposite camera prevents paired-view leakage but does not establish new-session generalization.",
    "Small validation sets and the declared candidate search can overfit validation; do not optimize further using locked test results.",
    "Pooled heads provide recording verdicts without timestamps. Sparse point labels do not identify complete event precision or interval IoU.",
    "Scores are uncalibrated. Visual features can omit small hand/object details and cannot establish legal intent.",
    "No filename, duration, absolute recording clock or domain indicator is a predictor; temporal pooling can still encode exposure length.",
]


def fit_reviewed_selection(records, feature_dir, split, output_dir, config=None, *, originals_from=None):
    config = config or ReviewedConfig()
    output_dir = Path(output_dir)
    if (output_dir / "locked_test_report.json").exists() or (output_dir / "LOCKED_TEST_USED.json").exists():
        raise FileExistsError("This run already opened its locked test; model selection cannot be repeated here")
    originals = scene.original_targets(originals_from)
    records = _verified_records(records, originals, split)
    partitions, split_provenance = validate_split(records, split)
    train, validation = partitions["train"], partitions["validation"]
    # Deliberately never prepare test features before the artifact is frozen.
    feature_sets = _feature_sets(feature_dir)
    feature_entries, feature_protocols = {}, {}
    for name, directory in feature_sets.items():
        feature_entries[name], feature_protocols[name] = _prepare(train + validation, directory, config)
    candidates = [{"representation": name, **candidate} for name in feature_sets for candidate in candidate_grid(config)]
    leaderboard, fitted, prepared = [], {}, {}
    with threadpool_limits(limits=config.threads):
        for index, candidate in enumerate(candidates):
            head, mode = candidate["head"], candidate["preprocessing"]
            representation = candidate["representation"]
            entries = feature_entries[representation]
            preparation_key = representation + "/" + head + "/" + mode
            if preparation_key not in prepared:
                training = _training_matrix(train, entries, head, config)
                preprocessing = _fit_preprocessing(training, mode, head, config)
                prepared[preparation_key] = training, preprocessing, _transform(training["x"], preprocessing)
            training, preprocessing, x = prepared[preparation_key]
            # Aggregations share one fitted window classifier; they do not
            # silently create extra training or read additional targets.
            classifier_key = scene._digest({key: value for key, value in candidate.items() if key != "aggregation"})
            if classifier_key not in fitted:
                classifier = _classifier(candidate, config)
                classifier.fit(x, training["y"], sample_weight=training["weights"])
                fitted[classifier_key] = classifier
            classifier = fitted[classifier_key]
            rows = _rows(validation, entries, classifier, preprocessing, candidate, .5, config, originals)
            threshold, _ = base._threshold(np.asarray([row["label"] for row in rows]),
                                          np.asarray([row["score"] for row in rows]), _metric_config(config))
            rows = _rows(validation, entries, classifier, preprocessing, candidate, threshold, config, originals)
            leaderboard.append({"candidate_id": _candidate_id(candidate), "candidate": candidate, "threshold": float(threshold),
                "validation_metrics": _metrics(rows), "preprocessing_fit_sha256": preprocessing["fit_sha256"],
                "training_supervision_owner_sha256": preprocessing["supervision_owner_sha256"],
                "validation_predictions": [{key: row[key] for key in ("sha256", "label", "score", "prediction", "domain")} for row in rows]})
            print(f"Reviewed candidate {index + 1}/{len(candidates)} {candidate['head']}/{candidate['classifier']}: validation complete", flush=True)
        leaderboard.sort(key=lambda row: (row["validation_metrics"][config.primary_metric],
            row["validation_metrics"]["balanced_accuracy"], row["validation_metrics"]["recall"]), reverse=True)
        winner = leaderboard[0]
        candidate = winner["candidate"]
        training, preprocessing, _ = prepared[candidate["representation"] + "/" + candidate["head"] + "/" + candidate["preprocessing"]]
        classifier_key = scene._digest({key: value for key, value in candidate.items() if key != "aggregation"})
        classifier = fitted[classifier_key]
    artifact = {"format_version": 1, "kind": "reviewed_locked_feature_model", "config": asdict(config),
        "candidate": candidate, "threshold": winner["threshold"], "classifier": classifier, "preprocessing": preprocessing,
        "feature_contract": feature_protocols[candidate["representation"]], "feature_sets": list(feature_sets), "source_sha256": _source_hashes(),
        "split": split_provenance, "all_supervision_sha256": _supervision_hash(records),
        "training_sha256": sorted(row["sha256"] for row in train),
        "validation_sha256": sorted(row["sha256"] for row in validation),
        "test_sha256": sorted(row["sha256"] for row in partitions["test"]),
        "selection_metric": config.primary_metric, "selected_without_test": True,
        "training_refit_on_validation": False, "limitations": LIMITATIONS}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model_reviewed.joblib"
    joblib.dump(artifact, path, compress=3)
    selection = {"evaluation": "train_only_fit_validation_selected_locked_test_pending", "config": asdict(config),
        "candidates": candidates, "feature_sets": list(feature_sets), "leaderboard": leaderboard, "winner": winner, "source_sha256": _source_hashes(),
        "split": split_provenance, "model_sha256": _hash_file(path), "all_supervision_sha256": _supervision_hash(records),
        "created_at": datetime.now(timezone.utc).isoformat(), "test_feature_files_opened": 0,
        "external_inference_api_used": False, "audio_used": False, "limitations": LIMITATIONS}
    _write(output_dir / "selection.json", selection)
    _write(output_dir / "selection_lock.json", {"model_sha256": _hash_file(path),
        "selection_sha256": _hash_file(output_dir / "selection.json"), "split_sha256": split_provenance["sha256"],
        "all_supervision_sha256": _supervision_hash(records), "source_sha256": _source_hashes(),
        "locked_at": datetime.now(timezone.utc).isoformat()})
    return path


def score_reviewed_test(records, feature_dir, split, model_path, output_dir, *, originals_from=None):
    """Open the declared test once after verifying the frozen model and lock."""
    output_dir, model_path = Path(output_dir), Path(model_path)
    if (output_dir / "LOCKED_TEST_USED.json").exists() or (output_dir / "locked_test_report.json").exists():
        raise FileExistsError("Locked test already opened for this run; it is not another selection signal")
    lock_path = output_dir / "selection_lock.json"
    if not lock_path.is_file():
        raise ValueError("Fit and freeze selection before opening test features")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["model_sha256"] != _hash_file(model_path) or lock["selection_sha256"] != _hash_file(output_dir / "selection.json"):
        raise ValueError("Frozen selected model or selection report changed before test")
    artifact = joblib.load(model_path)
    if artifact.get("source_sha256") != _source_hashes() or artifact.get("kind") != "reviewed_locked_feature_model":
        raise ValueError("Use the exact fitted source bytes and trusted reviewed model")
    originals = scene.original_targets(originals_from)
    records = _verified_records(records, originals, split)
    partitions, frozen = validate_split(records, split)
    if frozen["sha256"] != lock["split_sha256"] or _supervision_hash(records) != lock["all_supervision_sha256"]:
        raise ValueError("Locked split or target supervision changed before test")
    config = ReviewedConfig(**artifact["config"])
    feature_sets = _feature_sets(feature_dir)
    representation = artifact["candidate"]["representation"]
    if representation not in feature_sets:
        raise ValueError("Provide the frozen winner's declared feature representation")
    # Persist intent before any test-cache read, so failed test work is still
    # recorded instead of silently rerun after altering its model.
    _write(output_dir / "LOCKED_TEST_USED.json", {"opened_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": lock["model_sha256"], "split_sha256": frozen["sha256"], "completed": False})
    with threadpool_limits(limits=config.threads):
        entries, _ = _prepare(partitions["test"], feature_sets[representation], config, expected=artifact["feature_contract"])
        rows = _rows(partitions["test"], entries, artifact["classifier"], artifact["preprocessing"],
            artifact["candidate"], artifact["threshold"], config, originals)
    metrics = _metrics(rows)
    domains = {domain: _metrics([row for row in rows if row["domain"] == domain])
               for domain in sorted({row["domain"] for row in rows})}
    train_labels = [row["label"] for row in partitions["train"]]
    majority = int(sum(train_labels) > len(train_labels) / 2)
    report = {"evaluation": "single_locked_test_after_train_validation_model_selection", "metrics": metrics,
        "domain_metrics": domains, "predictions": rows, "selected_candidate": artifact["candidate"],
        "threshold": artifact["threshold"], "model_sha256": lock["model_sha256"], "split": frozen,
        "train_majority_constant": majority,
        "train_majority_baseline_test_accuracy": float(np.mean([row["label"] == majority for row in rows])),
        "partition_counts": {key: {"recordings": len(values), "groups": len({row["group_id"] for row in values}),
            "positive": sum(row["label"] for row in values), "negative": sum(1 - row["label"] for row in values)}
            for key, values in partitions.items()},
        "selected_without_test": True, "test_open_count": 1, "external_inference_api_used": False,
        "audio_used": False, "created_at": datetime.now(timezone.utc).isoformat(), "limitations": LIMITATIONS}
    _write(output_dir / "locked_test_report.json", report)
    _write(output_dir / "LOCKED_TEST_USED.json", {"opened_at": report["created_at"], "completed": True,
        "model_sha256": lock["model_sha256"], "split_sha256": frozen["sha256"], "test_open_count": 1})
    print(json.dumps({"evaluation": report["evaluation"], "metrics": metrics, "domain_metrics": domains}, indent=2), flush=True)
    return report


def predict_reviewed_features(feature_path, model_path):
    artifact = joblib.load(model_path)
    if artifact.get("kind") != "reviewed_locked_feature_model" or artifact.get("source_sha256") != _source_hashes():
        raise ValueError("Use a trusted reviewed model with matching fitted source bytes")
    config = ReviewedConfig(**artifact["config"])
    with threadpool_limits(limits=config.threads):
        entry = _entry(feature_path, config)
        current = {"feature_shape": entry["feature_shape"], "feature_protocol": entry["feature_protocol"]}
        if current != artifact["feature_contract"]:
            raise ValueError("Inference feature protocol differs from the fitted model")
        return _predict(artifact["classifier"], artifact["preprocessing"], artifact["candidate"],
                        entry, artifact["threshold"], config)

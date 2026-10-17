"""Fold-owned note prototypes; inference receives only visual CLIP features.

Friend-supplied notes are unverified privileged supervision, never evaluation
inputs or corrected ground truth. No raw/sanitized note text enters reports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
from scipy.special import expit

from . import model as base
from . import temporal
from .clip_features import BACKBONE, EXPECTED_WEIGHT_SHA256
from .custom_training import _assert_original_lineage_groups


@dataclass(frozen=True)
class PrivilegedConfig:
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    top_k: int = 3
    temperature: float = .07
    smoothing_seconds: float = 2.
    primary_metric: str = "balanced_accuracy"

    def __post_init__(self):
        if self.top_k < 1 or not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid prototype scoring policy")
        self.training_config()

    def training_config(self):
        return base.TrainingConfig(seed=self.seed, outer_folds=self.outer_folds,
            inner_folds=self.inner_folds, primary_metric=self.primary_metric,
            smoothing_seconds=self.smoothing_seconds)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _owner(owner):
    if not isinstance(owner, str) or not re.fullmatch(r"[0-9a-f]{64}", owner):
        raise ValueError("Note owner must be an exact source SHA256")
    return owner


def load_note_vector(cache_dir, owner, *, allowed_owners):
    """Check fit ownership BEFORE touching the cache file."""
    _owner(owner)
    if owner not in set(allowed_owners):
        raise PermissionError("Held-out note owner is not authorized for this fit")
    path = Path(cache_dir) / f"{owner}.npz"
    if path.is_symlink():
        raise ValueError("Linked note-vector files are not accepted")
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"].item()))
        vector = np.asarray(saved["vector"], dtype=np.float32)
    if metadata.get("owner_sha256") != owner:
        raise ValueError("Note vector cache owner differs from requested source")
    for key in ("input_notes_sha256", "protocol_sha256"):
        _owner(metadata.get(key))
    if type(metadata.get("usable")) is not bool or vector.shape != (512,) or not np.isfinite(vector).all():
        raise ValueError("Invalid note vector cache")
    if metadata["usable"] and not np.isclose(np.linalg.norm(vector), 1., atol=1e-4):
        raise ValueError("Usable text vector must have unit length")
    public = {key: metadata[key] for key in ("owner_sha256", "input_notes_sha256", "protocol_sha256", "usable")}
    public["supervision"] = "unverified_friend_privileged_notes"
    public["vector_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"vector": vector if metadata["usable"] else None, "provenance": public}


def fit_bank(records, note_cache, *, loader=None):
    records = sorted(base._canonical_records(records), key=lambda r: r["sha256"])
    allowed = frozenset(r["sha256"] for r in records)
    vectors, owners, provenance, protocols = {0: [], 1: []}, {0: [], 1: []}, [], set()
    for record in records:
        owner = record["sha256"]
        row = (loader(owner, allowed_owners=allowed) if loader else
               load_note_vector(note_cache, owner, allowed_owners=allowed))
        meta = row["provenance"]
        if meta.get("owner_sha256") != owner:
            raise ValueError("Prepared vector owner differs from training source")
        protocols.add(meta["protocol_sha256"])
        provenance.append(dict(meta))
        if row["vector"] is not None:
            vectors[record["label"]].append(row["vector"])
            owners[record["label"]].append(owner)
    if len(protocols) != 1:
        raise ValueError("Training note vectors must use one fixed preparation protocol")
    if not all(vectors.values()):
        counts = {str(label): len(rows) for label, rows in vectors.items()}
        raise ValueError("Both training classes require usable source-owned note prototypes; "
                         "no notes are invented. Usable counts=" + json.dumps(counts) +
                         "; unusable_sources=" + str(sum(not p["usable"] for p in provenance)))
    return {"negative": np.stack(vectors[0]), "positive": np.stack(vectors[1]),
            "owners_by_class": owners, "note_protocol_sha256": next(iter(protocols)),
            "fit_owner_allowlist": sorted(allowed), "fit_note_provenance": provenance,
            "fit_note_manifest_sha256": _digest(provenance),
            "unusable_sources": sum(not p["usable"] for p in provenance)}


def frame_scores(features, bank, config):
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (4, 512) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected four-view CLIP image features [T,4,512]")
    values = values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)
    scores = {}
    for label, key in ((0, "negative"), (1, "positive")):
        similarity = np.einsum("tvd,kd->tvk", values, bank[key]).mean(axis=1)
        k = min(config.top_k, similarity.shape[1])
        scores[label] = np.sort(similarity, axis=1)[:, -k:].mean(axis=1)
    return expit((scores[1] - scores[0]) / config.temperature)


def prepare_features(records, feature_dir):
    data, expected = {}, None
    for record in records:
        values, times, metadata = base.load_features(Path(feature_dir) / f"{record['sha256']}.npz")
        if (values.shape[1:] != (4, 512) or metadata.get("backbone") != BACKBONE
                or metadata.get("weights_sha256") != EXPECTED_WEIGHT_SHA256
                or metadata.get("views") != ["full", "top", "middle", "bottom"]):
            raise ValueError("Note prototypes require matching original four-view CLIP image caches")
        if expected is None:
            expected = metadata
        if metadata != expected:
            raise ValueError("Mixed CLIP image extractor configurations")
        data[record["sha256"]] = {"features": values, "timestamps": times, "metadata": metadata}
    return data


def _video_score(bank, entry, config):
    scores = frame_scores(entry["features"], bank, config)
    return base._aggregate(scores, entry["timestamps"], "smooth_max", config.training_config())


def select_threshold(records, data, note_cache, config, *, loader=None):
    splits = base.grouped_splits(records, config.inner_folds, config.seed + 1)
    if not splits:
        raise ValueError("Insufficient independent original groups for note-policy inner validation")
    scores = np.zeros(len(records))
    audits = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        train, validation = [records[i] for i in train_idx], [records[i] for i in val_idx]
        bank = fit_bank(train, note_cache, loader=loader)
        for index in val_idx:
            scores[index] = _video_score(bank, data[records[index]["sha256"]], config)
        audits.append({"fold": fold, "prototype_fit_sha256": bank["fit_owner_allowlist"],
            "validation_sha256": [r["sha256"] for r in validation],
            "fit_note_manifest_sha256": bank["fit_note_manifest_sha256"],
            "usable_prototypes_by_class": {str(k): len(v) for k, v in bank["owners_by_class"].items()},
            "unusable_sources": bank["unusable_sources"]})
    threshold, metrics = base._threshold(np.asarray([r["label"] for r in records]), scores, config.training_config())
    return {"threshold": threshold, "selection_metrics": metrics, "selection_folds": audits,
            "threshold_fit_sha256": [r["sha256"] for r in records],
            "note_content_used_as_validation_input": False, "score_calibrated": False}


def _prediction(bank, entry, selection, config):
    scores = frame_scores(entry["features"], bank, config)
    score = base._aggregate(scores, entry["timestamps"], "smooth_max", config.training_config())
    prediction = int(score >= selection["threshold"])
    return {"score": score, "prediction": prediction, "threshold": selection["threshold"],
        "verdict": "violation" if prediction else "no_violation", "score_calibrated": False,
        "events": base._events(scores, entry["timestamps"], selection["threshold"], config.training_config()) if prediction else [],
        "localization_available": True, "window_scores": scores.tolist(),
        "window_timestamps": entry["timestamps"].astype(float).tolist(), "inference_notes_used": False}


def run_privileged_cv(records, feature_dir, note_cache, output_dir, *, folds_from, config=None, loader=None):
    config = config or PrivilegedConfig()
    records = sorted(base._canonical_records(records), key=lambda r: r["sha256"])
    _assert_original_lineage_groups(records)
    splits, fold_source = temporal.frozen_outer_splits(records, folds_from,
        temporal.TemporalConfig(seed=config.seed, outer_folds=config.outer_folds, inner_folds=config.inner_folds))
    data = prepare_features(records, feature_dir)
    predictions, audits = [], []
    for fold, (train_idx, test_idx) in enumerate(splits):
        train, test = [records[i] for i in train_idx], [records[i] for i in test_idx]
        selection = select_threshold(train, data, note_cache, config, loader=loader)
        bank = fit_bank(train, note_cache, loader=loader)
        predictions.extend({"sha256": r["sha256"], "video_id": r.get("video_id", r["sha256"]),
            "label": r["label"], "event_times": r["event_times"], "fold": fold,
            **_prediction(bank, data[r["sha256"]], selection, config)} for r in test)
        audits.append({"fold": fold, "test_sha256": [r["sha256"] for r in test],
            "prototype_fit_sha256": bank["fit_owner_allowlist"], "fit_note_provenance": bank["fit_note_provenance"],
            "fit_note_manifest_sha256": bank["fit_note_manifest_sha256"], "selection": selection,
            "unusable_sources": bank["unusable_sources"],
            "usable_prototypes_by_class": {str(k): len(v) for k, v in bank["owners_by_class"].items()}})
        print("PRIVILEGED_NOTE_FOLD", json.dumps({"completed": fold + 1, "total": len(splits),
            "training_sources": len(train), "test_sources": len(test), "unusable_sources": bank["unusable_sources"]}), flush=True)
        temporal._write(Path(output_dir) / "privileged_progress.json", {"folds": audits, "predictions": predictions})
    report = {"evaluation": "frozen_original_folds_train_owned_unverified_privileged_note_prototypes",
        "notes_verified": False, "heldout_notes_used": False, "class_labels_enter_translator": False,
        "config": asdict(config), "fold_source": fold_source, "folds": audits, "predictions": predictions,
        "metrics": temporal._metrics(predictions), "created_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": ["Friend-supplied notes are unverified privileged supervision, not corrected labels.",
            "Each inner and outer prototype bank uses its training owners only; held-out notes never enter scoring or threshold inputs.",
            "Original folds are repeatedly inspected development data, not an independent acceptance test.",
            "Image/text cosine scores and sampled point events do not establish legal violations or calibrated probabilities.",
            "Video max aggregation depends on observation/exposure length; matched-length independent recordings remain necessary.",
            "No accuracy guarantee or complete event precision follows from this comparison."]}
    temporal._write(Path(output_dir) / "privileged_cv_report.json", report)
    return report


def fit_privileged_final(records, feature_dir, note_cache, output_dir, *, config=None, loader=None):
    config = config or PrivilegedConfig()
    records = sorted(base._canonical_records(records), key=lambda r: r["sha256"])
    _assert_original_lineage_groups(records)
    data = prepare_features(records, feature_dir)
    selection = select_threshold(records, data, note_cache, config, loader=loader)
    bank = fit_bank(records, note_cache, loader=loader)
    metadata = data[records[0]["sha256"]]["metadata"]
    artifact = {"kind": "privileged_text_prototypes", "format_version": 1, "bank": bank,
        "selection": selection, "config": asdict(config), "feature_config": metadata,
        "feature_semantic_config": base.semantic_feature_config(metadata),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "notes_verified": False, "inference_requires_notes": False}
    path = Path(output_dir) / "model_privileged.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)
    temporal._write(path.parent / "final_selection.json", {"selection": selection,
        "fit_owner_allowlist": bank["fit_owner_allowlist"], "fit_note_provenance": bank["fit_note_provenance"],
        "fit_note_manifest_sha256": bank["fit_note_manifest_sha256"], "model_file": path.name})
    return path


def predict_privileged_features(feature_path, model_path):
    artifact = joblib.load(model_path)  # Trusted artifacts only.
    if (artifact.get("kind") != "privileged_text_prototypes" or
            artifact.get("code_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError("Privileged prototype artifact/code fingerprint differs")
    values, times, metadata = base.load_features(feature_path)
    if base.semantic_feature_config(metadata) != artifact["feature_semantic_config"]:
        raise ValueError("Inference CLIP feature protocol differs from training")
    return _prediction(artifact["bank"], {"features": values, "timestamps": times},
        artifact["selection"], PrivilegedConfig(**artifact["config"]))

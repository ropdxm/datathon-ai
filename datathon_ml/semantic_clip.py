"""Local CLIP prompt contrasts as 18 observable-state *scores*.

These are uncalibrated image/text similarities, not verified actions, probabilities
of misconduct, or legal conclusions. Transaction starts/ends and departures are
single-frame visual proxies; a downstream temporal model must establish order.
Phone use alone is never classified as a violation by this extractor.

The registry, temperature and view policies are fixed independently of labels.
Original OpenAI CLIP image caches are reused; no pixels need decoding and no
recordings, embeddings, or prompts are sent to an inference service. Text runs
in the current Python process using the same original MIT-licensed checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence
from zipfile import BadZipFile

import numpy as np

from .clip_features import (
    BACKBONE as IMAGE_BACKBONE,
    ClipFeatureConfig,
    EXPECTED_WEIGHT_SHA256,
    HUB_REPO,
    OPEN_CLIP_VERSION,
    _load_clip_encoder,
    _open_clip_version,
    _resolve_clip_weights,
)
from .features import (
    _canonical_json,
    _configuration_hash,
    cache_is_valid,
    load_feature_cache,
    sha256_file,
)
from .temporal import CONCEPT_NAMES


BACKBONE = "clip_vit_b32_semantic18_openai"
TEMPERATURE = 0.07
TEMPLATES = (
    "A CCTV image of a bank counter: {}.",
    "A surveillance view of a bank teller: {}.",
)
# Concrete opposite scenes are used instead of relying on CLIP to understand
# syntactic negation. These prompts describe generic observations, not this corpus.
PROMPTS = {
    "cashier_present": (
        ("a bank employee is at the teller workstation", "a cashier is working behind the counter"),
        ("the teller workstation is empty", "an unattended desk behind a bank counter")),
    "customer_present": (
        ("a customer is standing at the teller window", "a visitor is being served at the bank counter"),
        ("the customer side of the bank counter is empty", "an empty waiting area in front of the teller window")),
    "loose_cash_visible": (
        ("loose banknotes are lying on the counter", "paper money is openly visible on the desk"),
        ("a clear counter with only office equipment", "an empty desk surface with stationery")),
    "cash_handover": (
        ("a cashier and a customer are passing banknotes between their hands", "paper money is being handed across the counter"),
        ("people have their hands on their own sides of the counter", "banknotes remain resting on the desk")),
    "cash_counting": (
        ("hands are counting individual banknotes", "a cashier is sorting and counting paper money"),
        ("hands rest beside a stationary stack of banknotes", "a person is typing on a computer keyboard")),
    "cash_secured": (
        ("banknotes are stored inside a cash drawer or safe", "a person is putting cash into a cash drawer or safe"),
        ("loose banknotes are exposed on the counter", "paper money is lying on an open desk surface")),
    "access_open": (
        ("the cash drawer or safe door is clearly open", "an open safe or open cash drawer"),
        ("the cash drawer and safe door are closed", "a closed safe and closed cash drawer")),
    "phone_interaction": (
        ("a person is holding and using a mobile phone", "a cashier is looking at a smartphone in their hand"),
        ("hands are using a keyboard and office documents", "a person is handling banknotes with both hands")),
    "object_to_pocket": (
        ("a person is putting a small object into a clothing pocket", "a hand is moving an item inside a jacket pocket"),
        ("a person is placing an item on the counter", "both hands are visible above the desk surface")),
    "document_check": (
        ("a cashier is examining a document or identity card", "a person is reading and checking paperwork"),
        ("a person is handling only banknotes", "a person is waiting with empty hands")),
    "transaction_active": (
        ("a cashier is serving a customer at the counter", "a bank employee and customer are exchanging documents or cash"),
        ("an empty customer counter and idle workstation", "a cashier is alone at the desk between customers")),
    "transaction_start": (
        ("a customer is approaching the teller counter for service", "a visitor is presenting documents to begin bank service"),
        ("a customer is walking away from the teller counter", "a customer is already engaged in a cash transaction")),
    "transaction_end": (
        ("a customer is collecting their belongings and leaving the teller counter", "a cashier is returning documents after completing service"),
        ("a customer is presenting documents to start service", "a cashier is actively counting a customer's banknotes")),
    "workstation_departure": (
        ("a cashier is walking away from the teller desk", "a bank employee is leaving the workstation through a doorway"),
        ("a cashier remains seated at the workstation", "a bank employee is standing at the teller desk")),
    "scene_clear": (
        ("the entire teller workstation is clearly visible", "an unobstructed camera view of the whole cashier work area"),
        ("the workstation is obscured by people or furniture", "the camera view cuts off most of the cashier work area")),
    "hands_visible": (
        ("people's hands are clearly visible above the counter", "both hands can be seen handling items on the desk"),
        ("hands are hidden below the counter", "people's hands are outside the camera view")),
    "cash_area_visible": (
        ("the cash handling surface is clearly visible", "an unobstructed view of the counter and cash tray"),
        ("a person's body blocks the cash handling surface", "the cash tray and counter are obscured from view")),
    "bag_or_seal_handling": (
        ("a person is handling a cash collection bag or its security seal", "hands are inspecting or closing a sealed cash bag"),
        ("hands are handling loose papers on a desk", "a person is using the computer keyboard")),
}
SCENE_CONCEPTS = frozenset({
    "cashier_present", "customer_present", "access_open", "transaction_active",
    "transaction_start", "transaction_end", "workstation_departure", "scene_clear",
})
SCENE_VIEW_WEIGHTS = {"full": .7, "middle": .3}
DETAIL_VIEW_WEIGHTS = {"full": .25, "top": .25, "middle": .25, "bottom": .25}


def prompt_protocol() -> dict:
    """Everything that fixes the semantic meaning of the output columns."""
    if tuple(PROMPTS) != CONCEPT_NAMES:
        raise ValueError("Semantic concept registry and prompt order differ")
    return {
        "version": 1, "concept_names": list(CONCEPT_NAMES), "prompts": PROMPTS,
        "templates": TEMPLATES, "temperature": TEMPERATURE,
        "scene_concepts": sorted(SCENE_CONCEPTS),
        "scene_view_weights": SCENE_VIEW_WEIGHTS,
        "detail_view_weights": DETAIL_VIEW_WEIGHTS,
        "ensemble": "mean-unit-text-embeddings-then-unit-normalize",
        "projection": "sigmoid(weighted-cosine-positive-minus-negative/temperature)",
        "weights_sha256": EXPECTED_WEIGHT_SHA256, "open_clip_version": OPEN_CLIP_VERSION,
        "calibration": "none; fixed prompts; no label-based prompt/temperature tuning",
    }


def _array_hash(positive: np.ndarray, negative: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(positive, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(negative, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _get_text_prototypes(cache_dir: Path, device: str, offline: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    protocol_json = _canonical_json(prompt_protocol())
    protocol_hash = _configuration_hash(protocol_json)
    destination = cache_dir / f"text_{protocol_hash}.npz"
    try:
        with np.load(destination, allow_pickle=False) as saved:
            positive, negative = saved["positive"], saved["negative"]
            checksum = str(saved["prototypes_sha256"].item())
            if str(saved["protocol_json"].item()) != protocol_json:
                raise ValueError("Text-prototype protocol differs")
            if str(saved["protocol_sha256"].item()) != protocol_hash or str(saved["weights_sha256"].item()) != EXPECTED_WEIGHT_SHA256:
                raise ValueError("Text-prototype model/protocol checksum differs")
            if positive.shape != (18, 512) or negative.shape != (18, 512):
                raise ValueError("Invalid text-prototype shape")
            if positive.dtype != np.float32 or negative.dtype != np.float32:
                raise ValueError("Text prototypes must be float32")
            if not np.isfinite(positive).all() or not np.isfinite(negative).all():
                raise ValueError("Nonfinite text prototypes")
            if checksum != _array_hash(positive, negative):
                raise ValueError("Text-prototype checksum differs")
            for value in (positive, negative):
                if not np.allclose(np.linalg.norm(value, axis=-1), 1., atol=1e-5):
                    raise ValueError("Text prototypes are not normalized")
        return positive, negative, {"protocol_sha256": protocol_hash, "prototypes_sha256": checksum,
            "weights_sha256": EXPECTED_WEIGHT_SHA256, "text_cache_reused": True}
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        pass
    import torch
    import open_clip

    if _open_clip_version() != OPEN_CLIP_VERSION:
        raise RuntimeError(f"Semantic CLIP requires open_clip_torch=={OPEN_CLIP_VERSION}")
    runtime = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if runtime.type not in {"cpu", "cuda"} or (runtime.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Unsupported or unavailable text-encoder device: {device}")
    checkpoint = _resolve_clip_weights(ClipFeatureConfig(allow_download=not offline), torch)
    encoder = _load_clip_encoder(checkpoint, runtime)
    texts, groups = [], []
    for concept in CONCEPT_NAMES:
        for phrases in PROMPTS[concept]:
            start = len(texts)
            texts.extend(template.format(phrase) for phrase in phrases for template in TEMPLATES)
            groups.append((start, len(texts)))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    encoded = []
    with torch.inference_mode():
        for offset in range(0, len(texts), 32):
            tokens = tokenizer(texts[offset : offset + 32]).to(runtime)
            vectors = encoder.encode_text(tokens)
            vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
            encoded.append(vectors.cpu().numpy())
    embedded = np.concatenate(encoded)
    prototypes = []
    for start, end in groups:
        vector = embedded[start:end].mean(axis=0)
        prototypes.append(vector / max(float(np.linalg.norm(vector)), 1e-8))
    prototypes = np.asarray(prototypes, dtype=np.float32).reshape(18, 2, 512)
    positive, negative = prototypes[:, 0].copy(), prototypes[:, 1].copy()
    checksum = _array_hash(positive, negative)
    _atomic_npz(destination, positive=positive, negative=negative,
        protocol_json=np.array(protocol_json), protocol_sha256=np.array(protocol_hash),
        prototypes_sha256=np.array(checksum), weights_sha256=np.array(EXPECTED_WEIGHT_SHA256))
    return positive, negative, {"protocol_sha256": protocol_hash, "prototypes_sha256": checksum,
        "weights_sha256": EXPECTED_WEIGHT_SHA256, "text_cache_reused": False}


def semantic_scores(features: np.ndarray, views: Sequence[str], positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Map aligned CLIP embeddings to float32 [time,1,18] contrast scores."""
    features = np.asarray(features, dtype=np.float32)
    positive, negative = np.asarray(positive, dtype=np.float32), np.asarray(negative, dtype=np.float32)
    if features.ndim != 3 or features.shape[1:] != (len(views), 512) or not len(features):
        raise ValueError("Expected nonempty CLIP features [time,view,512]")
    if not views or len(set(views)) != len(views) or set(views) - set(DETAIL_VIEW_WEIGHTS):
        raise ValueError("Input views must be distinct full/top/middle/bottom names")
    if positive.shape != (18, 512) or negative.shape != (18, 512):
        raise ValueError("Expected 18 positive and negative 512-dimensional text prototypes")
    if not all(np.isfinite(value).all() for value in (features, positive, negative)):
        raise ValueError("Semantic projection requires finite inputs")
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Zero image embedding cannot define cosine similarity")
    images = features / norm
    positive_norm = np.linalg.norm(positive, axis=-1, keepdims=True)
    negative_norm = np.linalg.norm(negative, axis=-1, keepdims=True)
    if np.any(positive_norm < 1e-8) or np.any(negative_norm < 1e-8):
        raise ValueError("Zero text prototype cannot define cosine similarity")
    positive = positive / positive_norm
    negative = negative / negative_norm
    contrasts = np.einsum("tvd,kd->tvk", images, positive - negative)
    combined = np.empty((len(features), 18), dtype=np.float32)
    for column, concept in enumerate(CONCEPT_NAMES):
        policy = SCENE_VIEW_WEIGHTS if concept in SCENE_CONCEPTS else DETAIL_VIEW_WEIGHTS
        weights = np.asarray([policy.get(view, 0.) for view in views], dtype=np.float32)
        # A source cache containing only detail crops still has a usable scene proxy.
        if weights.sum() <= 0:
            weights[:] = 1
        weights /= weights.sum()
        combined[:, column] = contrasts[:, :, column] @ weights
    logits = np.clip(combined / TEMPERATURE, -80, 80)
    return (1. / (1. + np.exp(-logits))).astype(np.float32)[:, None, :]


def convert_clip_cache(inputdir: str | Path, outputdir: str | Path,
                       device: str = "auto", offline: bool = False) -> dict[str, Path]:
    """Convert a flat directory of original CLIP NPZs to 18 semantic columns.

    Text prototypes live in outputdir/_text. A cached conversion is reused only
    when the protocol, source video SHA and exact source feature-cache bytes match.
    Per-record checksums stay outside config_json so all recordings share a schema.
    """
    inputdir, outputdir = Path(inputdir), Path(outputdir)
    if inputdir.resolve() == outputdir.resolve():
        raise ValueError("Semantic outputdir must differ from the original image-cache directory")
    sources = sorted(inputdir.glob("*.npz"))
    if not sources:
        raise FileNotFoundError(f"No CLIP feature caches found in {inputdir}")
    positive, negative, text_metadata = _get_text_prototypes(outputdir / "_text", device, offline)
    protocol_hash = text_metadata["protocol_sha256"]
    outputs = {}
    source_semantics = None
    for source in sources:
        cached = load_feature_cache(source)
        metadata = cached["config"]
        if metadata.get("backbone") != IMAGE_BACKBONE or metadata.get("weights_sha256") != EXPECTED_WEIGHT_SHA256:
            raise ValueError("Semantic projection requires the verified original OpenAI CLIP image representation")
        semantic_keys = ("schema_version", "backbone", "image_size", "views", "preprocessing",
                         "decoder", "weights_sha256", "hub_repo", "sample_fps")
        semantic = {key: metadata[key] for key in semantic_keys if key in metadata}
        if source_semantics is None:
            source_semantics = semantic
        elif semantic != source_semantics:
            raise ValueError("Mixed original CLIP extractor configurations; convert each run separately")
        if source.stem != cached["sha256"]:
            raise ValueError("Source cache filename differs from its embedded video SHA-256")
        source_config_hash = _configuration_hash(_canonical_json(semantic))
        signature = {
            "schema_version": 1, "backbone": BACKBONE,
            "sample_fps": metadata["sample_fps"], "image_size": metadata["image_size"],
            "views": ["semantic"], "concept_names": list(CONCEPT_NAMES),
            "preprocessing": f"semantic-clip18-fixed-v1;protocol={protocol_hash};source={source_config_hash}",
            "decoder": metadata["decoder"], "weights_sha256": EXPECTED_WEIGHT_SHA256,
            "hub_repo": HUB_REPO, "temperature": TEMPERATURE,
            "prompts_sha256": protocol_hash, "text_prototypes_sha256": text_metadata["prototypes_sha256"],
            "source_feature_config": semantic, "calibrated_probabilities": False,
            "continuous_window_coverage": False,
        }
        config_json = _canonical_json(signature)
        source_file_hash = sha256_file(source)
        destination = outputdir / source.name
        if cache_is_valid(destination, cached["sha256"], config_json):
            previous = load_feature_cache(destination)
            if previous["provenance"].get("source_feature_file_sha256") == source_file_hash:
                outputs[cached["sha256"]] = destination
                continue
        scores = semantic_scores(cached["features"], metadata["views"], positive, negative)
        provenance = {
            "source_feature_file_sha256": source_file_hash,
            "source_video_sha256": cached["sha256"], "source_feature_config": semantic,
            **text_metadata, "inference_location": "local_python_process",
            "external_inference_api_used": False, "license": "MIT (original OpenAI CLIP and OpenCLIP)",
            "limitations": "Uncalibrated visual proxies; no identity recognition, verified event timing, or legal determination.",
        }
        _atomic_npz(destination, features=scores, timestamps=cached["timestamps"],
            sha256=np.array(cached["sha256"]), config_json=np.array(config_json),
            config_hash=np.array(_configuration_hash(config_json)),
            provenance_json=np.array(_canonical_json(provenance)))
        load_feature_cache(destination)
        outputs[cached["sha256"]] = destination
        print(f"Semantic CLIP {len(outputs)}: {cached['sha256'][:12]}, {len(scores)} times x 18 states", flush=True)
    return outputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--offline", action="store_true")
    options = parser.parse_args()
    convert_clip_cache(options.input_dir, options.output_dir, options.device, options.offline)

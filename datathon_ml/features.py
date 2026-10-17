"""Frozen, local DINOv2 features for complete videos.

``extract_features`` accepts mappings or objects with ``path`` and optionally
``sha256``. Labels and annotated times are deliberately never inspected. It
returns ``{source_sha256: Path(cache_file)}``. Each cache contains normalized
float32 ``features[T,V,D]``, decoded ``timestamps[T]``, source SHA-256, and
JSON/hash metadata. Original video bytes are hashed before cache reuse.

Inference uses ordinary PyTorch on the current machine/notebook; downloading
public code/weights does not send recordings to an inference service. Set
``allow_download=False`` for a strictly offline run. Model weights are frozen.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence
import warnings
from zipfile import BadZipFile

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
HUB_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
HUB_REPO = f"facebookresearch/dinov2:{HUB_REVISION}"
SCHEMA_VERSION = 1
SUPPORTED_BACKBONES = {
    "dinov2_vits14": ("dinov2_vits14", "dinov2_vits14_pretrain.pth"),
    "dinov2_vitb14": ("dinov2_vitb14", "dinov2_vitb14_pretrain.pth"),
    "dinov2_vitl14": ("dinov2_vitl14", "dinov2_vitl14_pretrain.pth"),
    "dinov2_vits14_reg": ("dinov2_vits14", "dinov2_vits14_reg4_pretrain.pth"),
    "dinov2_vitb14_reg": ("dinov2_vitb14", "dinov2_vitb14_reg4_pretrain.pth"),
    "dinov2_vitl14_reg": ("dinov2_vitl14", "dinov2_vitl14_reg4_pretrain.pth"),
}


@dataclass(frozen=True)
class FeatureConfig:
    backbone: str = "dinov2_vits14"
    sample_fps: float = 2.0
    image_size: int = 224
    batch_size: int = 32  # Images, not videos; automatically reduced after OOM.
    device: str = "auto"
    views: tuple[str, ...] = ("full", "top", "middle", "bottom")
    local_repo: str | None = None
    local_checkpoint: str | None = None
    allow_download: bool = True

    def __post_init__(self) -> None:
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported original DINOv2 backbone: {self.backbone}")
        if not math.isfinite(self.sample_fps) or self.sample_fps <= 0:
            raise ValueError("sample_fps must be finite and positive")
        if self.image_size < 14 or self.image_size % 14:
            raise ValueError("image_size must be a positive multiple of patch size 14")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.views or len(set(self.views)) != len(self.views):
            raise ValueError("views must contain at least one distinct view")
        if set(self.views) - {"full", "top", "middle", "bottom"}:
            raise ValueError(f"Unknown crop views: {self.views}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_views(rgb: np.ndarray, image_size: int, views: Sequence[str]) -> np.ndarray:
    """Return uint8 RGB ``[V,S,S,3]`` without stretching the full portrait.

    The three square crops overlap along the long axis. On portrait footage
    these are top, middle and bottom; on landscape footage they are left,
    middle and right. Their positions depend only on image geometry.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("Expected a nonempty HxWx3 RGB image")
    height, width = rgb.shape[:2]
    side = min(height, width)
    result = []
    for view in views:
        if view == "full":
            scale = image_size / max(height, width)
            new_width = max(1, min(image_size, int(round(width * scale))))
            new_height = max(1, min(image_size, int(round(height * scale))))
            # ImageNet-mean padding becomes approximately zero after normalization.
            square = np.empty((image_size, image_size, 3), dtype=np.uint8)
            square[...] = (124, 116, 104)
            x, y = (image_size - new_width) // 2, (image_size - new_height) // 2
            resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
            square[y : y + new_height, x : x + new_width] = resized
        else:
            if view not in {"top", "middle", "bottom"}:
                raise ValueError(f"Unknown view: {view}")
            fraction = {"top": 0.0, "middle": 0.5, "bottom": 1.0}[view]
            x = int(round((width - side) * fraction))
            y = int(round((height - side) * fraction))
            crop = rgb[y : y + side, x : x + side]
            square = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
        result.append(square)
    return np.stack(result)


def iter_video_frames(path: str | Path, sample_fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """Decode sequentially, sampling by decoded time and preserving the final frame.

    PTS from OpenCV is used where monotonic; frame index / FPS is the fallback
    for broken/missing PTS. There are no random seeks or metadata-based endpoint
    reads. Only the current decoded frame is retained. Retrieving each frame
    lets the final *valid* frame survive inaccurate container frame counts.
    """
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be finite and positive")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {path}")
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fallback_fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else 30.0
    step = 1.0 / sample_fps
    next_sample = 0.0
    frame_index = 0
    previous_time = -math.inf
    previous_pts = -math.inf
    last_yielded = -math.inf
    last_rgb = None
    last_time = 0.0
    try:
        while capture.grab():
            success, bgr = capture.retrieve()
            if not success or bgr is None or bgr.size == 0:
                frame_index += 1
                continue
            pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if math.isfinite(pts) and pts >= 0 and pts > previous_pts and pts > previous_time:
                timestamp = pts
            else:
                timestamp = frame_index / fallback_fps
                if timestamp <= previous_time:
                    timestamp = previous_time + 1.0 / fallback_fps
            previous_time = timestamp
            if math.isfinite(pts):
                previous_pts = pts
            frame_index += 1
            last_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            last_time = timestamp
            if timestamp + 1e-7 >= next_sample:
                yield timestamp, last_rgb
                last_yielded = timestamp
                next_sample = (math.floor((timestamp + 1e-7) / step) + 1) * step
        if last_rgb is None:
            raise ValueError(f"Video contains no decodable frames: {path}")
        if last_time > last_yielded + 1e-6:
            yield last_time, last_rgb
    finally:
        capture.release()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _configuration_hash(config_json: str) -> str:
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    """Load a cache without pickle, rejecting malformed/incomplete feature arrays."""
    with np.load(path, allow_pickle=False) as cached:
        features = cached["features"]
        timestamps = cached["timestamps"]
        config_json = str(cached["config_json"].item())
        config_hash = str(cached["config_hash"].item())
        source_sha = str(cached["sha256"].item())
        provenance = json.loads(str(cached["provenance_json"].item()))
    config = json.loads(config_json)
    if config_hash != _configuration_hash(config_json):
        raise ValueError("Feature-cache configuration hash mismatch")
    if features.dtype != np.float32 or features.ndim != 3 or min(features.shape) < 1:
        raise ValueError("Invalid feature shape/dtype; expected float32 [T,V,D]")
    if timestamps.dtype != np.float32 or timestamps.shape != (features.shape[0],):
        raise ValueError("Invalid timestamp shape/dtype")
    if features.shape[1] != len(config["views"]):
        raise ValueError("Feature view count does not match cache metadata")
    if not np.isfinite(features).all() or not np.isfinite(timestamps).all():
        raise ValueError("Feature cache contains non-finite values")
    if np.any(np.diff(timestamps) <= 0) or timestamps[0] < 0:
        raise ValueError("Feature timestamps must be nonnegative and strictly increasing")
    if len(source_sha) != 64 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("Invalid source SHA-256 in cache")
    return {
        "features": features,
        "timestamps": timestamps,
        "sha256": source_sha,
        "config": config,
        "config_json": config_json,
        "config_hash": config_hash,
        "provenance": provenance,
    }


def cache_is_valid(path: str | Path, source_sha256: str, config_json: str) -> bool:
    try:
        cached = load_feature_cache(path)
        return cached["sha256"] == source_sha256 and cached["config_json"] == config_json
    except (OSError, ValueError, KeyError, EOFError, BadZipFile, json.JSONDecodeError):
        return False


def _code_hash(repo_path: Path) -> str:
    """Fingerprint local Python source, including uncommitted local modifications."""
    digest = hashlib.sha256()
    files = sorted(repo_path.rglob("*.py"))
    if not files or not (repo_path / "hubconf.py").is_file():
        raise ValueError(f"local_repo must be the DINOv2 repository root: {repo_path}")
    for path in files:
        digest.update(path.relative_to(repo_path).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _resolve_weights(config: FeatureConfig, torch: Any) -> Path:
    if config.local_checkpoint is not None:
        checkpoint = Path(config.local_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing local checkpoint: {checkpoint}")
        return checkpoint
    model_directory, filename = SUPPORTED_BACKBONES[config.backbone]
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not checkpoint.is_file():
        if not config.allow_download:
            raise FileNotFoundError(
                f"Offline weights not found: {checkpoint}. Supply local_checkpoint or cache these weights first."
            )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://dl.fbaipublicfiles.com/dinov2/{model_directory}/{filename}"
        temporary = checkpoint.with_suffix(".downloading")
        try:
            torch.hub.download_url_to_file(url, str(temporary), progress=True)
            os.replace(temporary, checkpoint)
        finally:
            temporary.unlink(missing_ok=True)
    return checkpoint


def _load_encoder(config: FeatureConfig, checkpoint: Path, device: Any, torch: Any) -> Any:
    # Reference attention avoids an optional xformers installation changing behavior.
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    cached_repo = Path(torch.hub.get_dir()) / f"facebookresearch_dinov2_{HUB_REVISION}"
    repository = Path(config.local_repo).expanduser().resolve() if config.local_repo else cached_repo
    if config.local_repo or repository.is_dir():
        if not repository.is_dir():
            raise FileNotFoundError(f"Missing local DINOv2 repository: {repository}")
        model = torch.hub.load(str(repository), config.backbone, source="local", pretrained=False)
    else:
        if not config.allow_download:
            raise FileNotFoundError(
                "Offline DINOv2 source not found. Supply local_repo pointing to a complete DINOv2 checkout."
            )
        model = torch.hub.load(
            HUB_REPO, config.backbone, source="github", pretrained=False,
            trust_repo=True, skip_validation=True,
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model.to(device=device, dtype=torch.float32)


def _encode_views(images: np.ndarray, encoder: Any, config: FeatureConfig, device: Any, torch: Any) -> np.ndarray:
    result = []
    offset = 0
    batch_size = config.batch_size
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).reshape(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).reshape(1, 3, 1, 1)
    while offset < len(images):
        tensor = None
        embedding = None
        try:
            chunk = images[offset : offset + batch_size]
            tensor = torch.from_numpy(np.ascontiguousarray(chunk.transpose(0, 3, 1, 2)))
            tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
            tensor = (tensor - mean) / std
            autocast = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with torch.inference_mode(), autocast:
                embedding = encoder(tensor)
                if isinstance(embedding, dict):
                    embedding = embedding["x_norm_clstoken"]
                embedding = torch.nn.functional.normalize(embedding.float(), p=2, dim=-1)
            result.append(embedding.cpu().numpy().astype(np.float32, copy=False))
            offset += len(chunk)
        except torch.OutOfMemoryError as error:
            if device.type != "cuda" or batch_size == 1:
                raise RuntimeError(
                    "Encoder ran out of memory even at its minimum batch. Use a smaller backbone/image_size."
                ) from error
            del tensor, embedding
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            warnings.warn(f"CUDA memory pressure: retrying with image batch_size={batch_size}", RuntimeWarning)
    return np.concatenate(result, axis=0)


def _record_value(record: Any, key: str, default: Any = None) -> Any:
    return record.get(key, default) if isinstance(record, Mapping) else getattr(record, key, default)


def extract_features(
    records: Iterable[Any], cache_dir: str | Path, config: FeatureConfig | None = None,
) -> dict[str, Path]:
    """Extract/cached frozen embeddings, returning paths keyed by exact-video SHA.

    The only consumed record fields are ``path`` (or ``video_path``) and optional
    ``sha256``. Duplicate recordings are computed once. Incompatible caches are
    rebuilt atomically; model, checkpoint bytes, preprocessing, device precision,
    dependency versions, sampling, and source bytes participate in cache identity.
    """
    import torch

    config = config or FeatureConfig()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select CPU or install a CUDA PyTorch build")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported feature-extraction devices are CPU and CUDA")
    checkpoint = _resolve_weights(config, torch)
    local_code_sha = _code_hash(Path(config.local_repo).expanduser().resolve()) if config.local_repo else None
    signature = {
        "schema_version": SCHEMA_VERSION,
        "backbone": config.backbone,
        "sample_fps": float(config.sample_fps),
        "image_size": config.image_size,
        "views": list(config.views),
        "preprocessing": "rgb-imagenet-letterbox-square-overlap-v1",
        "decoder": "opencv-sequential-monotonic-pts-final-valid-v1",
        "hub_repo": HUB_REPO,
        "local_code_sha256": local_code_sha,
        "weights_sha256": sha256_file(checkpoint),
        "precision": "cuda-float16" if device.type == "cuda" else "cpu-float32",
        "torch_version": str(torch.__version__),
        "opencv_version": str(cv2.__version__),
    }
    config_json = _canonical_json(signature)
    provenance = {
        "checkpoint": str(checkpoint.resolve()),
        "local_repo": str(Path(config.local_repo).resolve()) if config.local_repo else None,
        "hub_repo": HUB_REPO,
        "weights_sha256": signature["weights_sha256"],
        "device": str(device),
        "batch_size_requested": config.batch_size,
        "license": "Apache-2.0 (original DINOv2)",
    }
    output: dict[str, Path] = {}
    encoder = None
    for record in records:
        video_path = _record_value(record, "path", _record_value(record, "video_path"))
        if video_path is None:
            raise ValueError("Each record must contain a path or video_path")
        video_path = Path(video_path)
        source_sha = sha256_file(video_path)
        expected_sha = _record_value(record, "sha256")
        if expected_sha and str(expected_sha).lower() != source_sha:
            raise ValueError(f"Source video changed after manifest generation: {video_path}")
        if source_sha in output:
            continue
        cache_path = cache_dir / f"{source_sha}.npz"
        if cache_is_valid(cache_path, source_sha, config_json):
            output[source_sha] = cache_path
            LOGGER.info("Reused feature cache %s", cache_path.name)
            continue
        if encoder is None:
            torch.manual_seed(0)
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cuda.matmul.allow_tf32 = False
            encoder = _load_encoder(config, checkpoint, device, torch)
        blocks = []
        timestamps = []
        pending = []
        frames_per_batch = max(1, config.batch_size // len(config.views))
        for timestamp, rgb in iter_video_frames(video_path, config.sample_fps):
            timestamps.append(timestamp)
            pending.append(make_views(rgb, config.image_size, config.views))
            if len(pending) >= frames_per_batch:
                images = np.concatenate(pending, axis=0)
                embeddings = _encode_views(images, encoder, config, device, torch)
                blocks.append(embeddings.reshape(len(pending), len(config.views), -1))
                pending.clear()
        if pending:
            embeddings = _encode_views(np.concatenate(pending, axis=0), encoder, config, device, torch)
            blocks.append(embeddings.reshape(len(pending), len(config.views), -1))
        features = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
        frame_times = np.asarray(timestamps, dtype=np.float32)
        handle = tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False)
        temporary = Path(handle.name)
        handle.close()
        try:
            np.savez_compressed(
                temporary, features=features, timestamps=frame_times,
                sha256=np.array(source_sha), config_json=np.array(config_json),
                config_hash=np.array(_configuration_hash(config_json)),
                provenance_json=np.array(_canonical_json(provenance)),
            )
            # Verify the complete file before it replaces an earlier working cache.
            load_feature_cache(temporary)
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        output[source_sha] = cache_path
        # The SHA is portable even on Windows consoles without Cyrillic encoding.
        print(f"Features {len(output)}: {source_sha[:12]}, {features.shape[0]} frames x {features.shape[1]} views x {features.shape[2]}")
    return output

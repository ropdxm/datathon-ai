"""Frozen X3D action features, evaluated locally or inside the user's GPU notebook.

Official model/transform references:
https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/hub/x3d.py
https://github.com/facebookresearch/pytorchvideo/blob/main/LICENSE

PyTorchVideo's repository is Apache-2.0. Its official hub links the Kinetics-400
checkpoints; a separate checkpoint-specific license was not found. Provenance
records this basis explicitly instead of asserting an independent weight license.

Install ``fvcore`` and ``iopath`` alongside torch/torchvision before first use.
No PyTorchVideo data/transform APIs, remote inference, or LLM APIs are used.
The NPZ contract matches ``datathon_ml.features`` so the same grouped evaluation
and classifiers can compare spatial and motion representations.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Sequence
import warnings

import cv2
import numpy as np

from .features import (
    _canonical_json,
    _configuration_hash,
    _record_value,
    cache_is_valid,
    iter_video_frames,
    load_feature_cache,
    make_views,
    sha256_file,
)


X3D_REVISION = "2ac6ed361309cb25144636ad82dd8bd6a4970e0c"
X3D_HUB_REPO = f"facebookresearch/pytorchvideo:{X3D_REVISION}"
X3D_CHECKPOINT_ROOT = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics"
# Official tutorial: num_frames, raw-frame sampling stride, reference FPS=30.
X3D_SPECS = {"x3d_xs": (4, 12, "X3D_XS.pyth"), "x3d_s": (13, 6, "X3D_S.pyth")}
LICENSE_URL = "https://github.com/facebookresearch/pytorchvideo/blob/main/LICENSE"


@dataclass(frozen=True)
class MotionFeatureConfig:
    backbone: str = "x3d_s"
    sample_fps: float = 1.0  # Window-center rate; internal clip frames are separate.
    image_size: int = 182
    batch_size: int = 8  # View-clips; not full videos.
    device: str = "auto"
    views: tuple[str, ...] = ("full", "middle")
    local_repo: str | None = None
    local_checkpoint: str | None = None
    allow_download: bool = True
    representation: str = "penultimate"  # 2048 features, or 400 raw Kinetics logits.

    def __post_init__(self) -> None:
        if self.backbone not in X3D_SPECS:
            raise ValueError("Motion backbone must be x3d_xs or x3d_s")
        if not math.isfinite(self.sample_fps) or self.sample_fps <= 0:
            raise ValueError("sample_fps must be finite and positive")
        if self.image_size < 32 or self.batch_size < 1:
            raise ValueError("image_size must be >=32 and batch_size must be positive")
        if self.representation not in {"penultimate", "logits"}:
            raise ValueError("representation must be penultimate or logits")
        if not self.views or len(set(self.views)) != len(self.views):
            raise ValueError("At least one distinct motion view is required")
        if set(self.views) - {"full", "top", "middle", "bottom"}:
            raise ValueError(f"Unknown motion crop views: {self.views}")

    @property
    def num_frames(self) -> int:
        return X3D_SPECS[self.backbone][0]

    @property
    def clip_duration_sec(self) -> float:
        count, sampling_stride, _ = X3D_SPECS[self.backbone]
        return count * sampling_stride / 30.0

    @property
    def center_stride_sec(self) -> float:
        return 1.0 / self.sample_fps


# A short alias is convenient in exploratory notebooks.
MotionConfig = MotionFeatureConfig


def _motion_views(rgb: np.ndarray, config: MotionConfig) -> np.ndarray:
    result = make_views(rgb, config.image_size, config.views)
    if "full" in config.views:
        # X3D uses mean=.45, std=.225 rather than ImageNet normalization.
        full = result[config.views.index("full")]
        height, width = rgb.shape[:2]
        scale = config.image_size / max(height, width)
        new_width = max(1, min(config.image_size, int(round(width * scale))))
        new_height = max(1, min(config.image_size, int(round(height * scale))))
        left = (config.image_size - new_width) // 2
        top = (config.image_size - new_height) // 2
        full[:, :left] = 115
        full[:, left + new_width :] = 115
        full[:top] = 115
        full[top + new_height :] = 115
    return result


def select_clip(
    frame_times: Sequence[float], frames: Sequence[np.ndarray], center: float,
    duration: float, num_frames: int,
) -> np.ndarray:
    """Nearest decoded frames for a centered clip, clamping unavailable context.

    Input frames are ``[V,H,W,3]``; result is uint8 ``[V,3,T,H,W]``. Uniform
    targets span the official tutorial's clip duration. Start/end padding repeats
    the first/last visible frame; it does not synthesize motion or use annotations.
    """
    times = np.asarray(frame_times, dtype=np.float64)
    if len(times) == 0 or len(times) != len(frames):
        raise ValueError("A clip needs matching, nonempty frame and timestamp buffers")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Frame timestamps must be strictly increasing")
    targets = np.linspace(center - duration / 2, center + duration / 2, num_frames)
    after = np.searchsorted(times, targets).clip(0, len(times) - 1)
    before = (after - 1).clip(0, len(times) - 1)
    indices = np.where(np.abs(times[before] - targets) <= np.abs(times[after] - targets), before, after)
    selected = np.stack([frames[int(index)] for index in indices], axis=0)
    return np.ascontiguousarray(selected.transpose(1, 4, 0, 2, 3))


def iter_motion_clips(path: str | Path, config: MotionConfig) -> Iterator[tuple[float, np.ndarray]]:
    """Bounded-memory, sequential clip extraction over the entire recording.

    Every decoded frame can supply a clip target. The ring buffer holds only the
    context for the next centered window, already resized, instead of retaining
    full-resolution videos. All window centers are independent of label anchors.
    """
    time_buffer: deque[float] = deque()
    frame_buffer: deque[np.ndarray] = deque()
    next_center = 0.0
    last_time = None
    last_emitted = -math.inf
    half = config.clip_duration_sec / 2

    def emit(center: float) -> tuple[float, np.ndarray]:
        return center, select_clip(
            list(time_buffer), list(frame_buffer), center, config.clip_duration_sec, config.num_frames,
        )

    # Oversampling the decoder emits every source frame (ordinary CCTV <1000 FPS).
    for timestamp, rgb in iter_video_frames(path, sample_fps=1000.0):
        time_buffer.append(timestamp)
        frame_buffer.append(_motion_views(rgb, config))
        last_time = timestamp
        while timestamp + 1e-7 >= next_center + half:
            yield emit(next_center)
            last_emitted = next_center
            next_center += config.center_stride_sec
        # Retain one earlier frame for nearest-time selection and enough recent
        # context for a final EOF window, including when the center stride is large.
        earliest = min(next_center - half, timestamp - config.clip_duration_sec)
        while len(time_buffer) > 1 and time_buffer[1] < earliest:
            time_buffer.popleft()
            frame_buffer.popleft()
    if last_time is None:
        raise ValueError(f"No valid frames in video: {path}")
    # At EOF, remaining centers use real available context with edge repetition.
    while next_center <= last_time + 1e-7:
        yield emit(next_center)
        last_emitted = next_center
        next_center += config.center_stride_sec
    if last_time > last_emitted + 1e-5:
        yield emit(last_time)


def _checkpoint(config: MotionConfig, torch: Any) -> Path:
    if config.local_checkpoint:
        path = Path(config.local_checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing local X3D checkpoint: {path}")
        return path
    filename = X3D_SPECS[config.backbone][2]
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not path.is_file():
        if not config.allow_download:
            raise FileNotFoundError(f"Offline X3D checkpoint not cached: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".downloading")
        try:
            torch.hub.download_url_to_file(f"{X3D_CHECKPOINT_ROOT}/{filename}", str(temporary))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def configure_motion_head(model: Any, representation: str, torch: Any) -> Any:
    """Use X3D's pretrained 2048-channel projected pool before its final Linear.

    Adaptive spatial/temporal averaging removes fixed training crop-size pooling
    constraints. Convolution, BN, SE and projection weights stay unchanged. The
    optional logits representation retains the pretrained 400-way Linear layer.
    """
    if representation not in {"penultimate", "logits"}:
        raise ValueError(f"Unknown motion representation: {representation}")
    head = model.blocks[-1]
    if not hasattr(head.pool, "pool"):
        raise ValueError("Unexpected X3D head: projected pooling layer not found")
    head.pool.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
    head.activation = None
    if representation == "penultimate":
        head.proj = torch.nn.Identity()
    model.eval().requires_grad_(False)
    return model


def _load_model(config: MotionConfig, checkpoint: Path, device: Any, torch: Any) -> Any:
    cached_source = Path(torch.hub.get_dir()) / f"facebookresearch_pytorchvideo_{X3D_REVISION}"
    repository = Path(config.local_repo).expanduser().resolve() if config.local_repo else cached_source
    try:
        if config.local_repo or repository.is_dir():
            if not (repository / "hubconf.py").is_file():
                raise FileNotFoundError(f"local_repo is not a complete PyTorchVideo checkout: {repository}")
            model = torch.hub.load(str(repository), config.backbone, source="local", pretrained=False)
        else:
            if not config.allow_download:
                raise FileNotFoundError("Offline PyTorchVideo source missing; supply local_repo")
            model = torch.hub.load(
                X3D_HUB_REPO, config.backbone, source="github", pretrained=False,
                trust_repo=True, skip_validation=True,
            )
    except ModuleNotFoundError as error:
        raise RuntimeError("X3D needs fvcore and iopath. Install: pip install fvcore iopath") from error
    # Only a tensor/state-dict checkpoint is accepted; arbitrary pickle execution is disabled.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    configure_motion_head(model, config.representation, torch)
    return model.to(device=device, dtype=torch.float32)


def _encode(clips: np.ndarray, model: Any, config: MotionConfig, device: Any, torch: Any) -> np.ndarray:
    output = []
    offset = 0
    batch_size = config.batch_size
    while offset < len(clips):
        tensor = None
        embeddings = None
        try:
            batch = clips[offset : offset + batch_size]
            tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(device=device, dtype=torch.float32)
            tensor = (tensor / 255.0 - 0.45) / 0.225
            autocast = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with torch.inference_mode(), autocast:
                embeddings = model(tensor)
                if embeddings.ndim != 2:
                    raise ValueError(f"Expected X3D [batch,dimension] output; got {tuple(embeddings.shape)}")
                embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=-1)
            output.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
            offset += len(batch)
        except torch.OutOfMemoryError as error:
            if device.type != "cuda" or batch_size == 1:
                raise RuntimeError("X3D exhausted memory: reduce image_size or use x3d_xs") from error
            del tensor, embeddings
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            warnings.warn(f"CUDA memory pressure: retrying X3D batch_size={batch_size}", RuntimeWarning)
    return np.concatenate(output)


def _local_source_hash(path: str | None) -> str | None:
    if path is None:
        return None
    root = Path(path).expanduser().resolve()
    if not (root / "hubconf.py").is_file():
        raise FileNotFoundError(f"Missing PyTorchVideo hubconf.py in {root}")
    digest = hashlib.sha256()
    for source in sorted(root.rglob("*.py")):
        digest.update(source.relative_to(root).as_posix().encode("utf-8"))
        digest.update(source.read_bytes())
    return digest.hexdigest()


def extract_motion_features(
    records: Iterable[Any], cache_dir: str | Path, config: MotionConfig | None = None,
) -> dict[str, Path]:
    """Return ``{source_sha256: npz_path}`` with ``features[T,V,D]`` and centers.

    Only record.path/video_path and record.sha256 are inspected. Exact duplicates
    are processed once. Label verdicts, annotation times and recording IDs never
    enter clip selection or feature construction.
    """
    import torch

    config = config or MotionConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Motion features support CPU and CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = _checkpoint(config, torch)
    local_source_hash = _local_source_hash(config.local_repo)
    # This entire string is checked by the existing export/inference model code.
    preprocessing = _canonical_json({
        "name": "x3d-centered-nearest-rgb-mean045-std0225-adaptivepool-v1",
        "num_frames": config.num_frames,
        "clip_duration_sec": config.clip_duration_sec,
        "center_stride_sec": config.center_stride_sec,
        "representation": config.representation,
        "normalization": "l2-output",
    })
    signature = {
        "schema_version": 1,
        "backbone": config.backbone,
        "sample_fps": 1.0 / config.center_stride_sec,
        "image_size": config.image_size,
        "views": list(config.views),
        "preprocessing": preprocessing,
        "decoder": "opencv-sequential-allframes-centered-ringbuffer-final-valid-v1",
        "weights_sha256": sha256_file(checkpoint),
        "hub_repo": X3D_HUB_REPO,
        "local_code_sha256": local_source_hash,
        "precision": "cuda-float16" if device.type == "cuda" else "cpu-float32",
        "torch_version": str(torch.__version__),
        "opencv_version": str(cv2.__version__),
    }
    config_json = _canonical_json(signature)
    provenance = {
        "checkpoint": str(checkpoint.resolve()),
        "weights_sha256": signature["weights_sha256"],
        "hub_repo": X3D_HUB_REPO,
        "local_repo": str(Path(config.local_repo).expanduser().resolve()) if config.local_repo else None,
        "local_code_sha256": local_source_hash,
        "pretrained_dataset": "Kinetics-400",
        "representation": config.representation,
        "license": "Repository Apache-2.0; official linked checkpoint; no separate weight license found",
        "license_url": LICENSE_URL,
        "license_basis": "https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/hub/x3d.py",
        "device": str(device),
    }
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    model = None
    windows_per_batch = max(1, config.batch_size // len(config.views))
    for record in records:
        video_path = _record_value(record, "path", _record_value(record, "video_path"))
        if video_path is None:
            raise ValueError("Each motion record requires path or video_path")
        source_sha = sha256_file(video_path)
        expected_sha = _record_value(record, "sha256")
        if expected_sha and str(expected_sha).lower() != source_sha:
            raise ValueError("Video bytes changed since manifest generation")
        if source_sha in paths:
            continue
        cache_path = cache_dir / f"{source_sha}.npz"
        if cache_is_valid(cache_path, source_sha, config_json):
            paths[source_sha] = cache_path
            continue
        if model is None:
            torch.manual_seed(0)
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cuda.matmul.allow_tf32 = False
            model = _load_model(config, checkpoint, device, torch)
        pending = []
        blocks = []
        timestamps = []
        for timestamp, clips in iter_motion_clips(video_path, config):
            pending.append(clips)
            timestamps.append(timestamp)
            if len(pending) >= windows_per_batch:
                embeddings = _encode(np.concatenate(pending), model, config, device, torch)
                blocks.append(embeddings.reshape(len(pending), len(config.views), -1))
                pending.clear()
        if pending:
            embeddings = _encode(np.concatenate(pending), model, config, device, torch)
            blocks.append(embeddings.reshape(len(pending), len(config.views), -1))
        features = np.concatenate(blocks).astype(np.float32, copy=False)
        handle = tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False)
        temporary = Path(handle.name)
        handle.close()
        try:
            np.savez_compressed(
                temporary, features=features, timestamps=np.asarray(timestamps, dtype=np.float32),
                sha256=np.array(source_sha), config_json=np.array(config_json),
                config_hash=np.array(_configuration_hash(config_json)),
                provenance_json=np.array(_canonical_json(provenance)),
            )
            load_feature_cache(temporary)
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        paths[source_sha] = cache_path
        print(f"Motion {len(paths)}: {source_sha[:12]}, {features.shape[0]} clips x {features.shape[1]} views x {features.shape[2]}", flush=True)
    return paths

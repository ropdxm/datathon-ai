"""Frozen pretrained video encoders and small binary classification heads.

Spatial preprocessing deliberately uses full-view letterboxing at native model
crop size, rather than the pretrained recipes' center crop. This retains camera
edges and portrait scenes, with some loss of effective object resolution.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .video import VideoFrameCache

MODEL_IDS = ("dinov2_gru", "r3d_18", "r2plus1d_18", "swin3d_t")
PREPROCESSING_VERSION = "pts15-fullview-letterbox-v1"


class GRUHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1)
        self.config = {"model_id": "dinov2_gru", "feature_dim": feature_dim, "hidden_size": hidden_size}

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(features)
        return self.classifier(hidden[-1]).squeeze(-1)


class LinearHead(nn.Module):
    def __init__(self, model_id: str, feature_dim: int):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, 1)
        self.config = {"model_id": model_id, "feature_dim": feature_dim}

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features.mean(dim=1)).squeeze(-1)


def make_head(model_id: str, feature_dim: int) -> nn.Module:
    if model_id not in MODEL_IDS or feature_dim < 1:
        raise ValueError(f"Unknown model_id or invalid feature_dim: {model_id!r}, {feature_dim}")
    return GRUHead(feature_dim) if model_id == "dinov2_gru" else LinearHead(model_id, feature_dim)


def _video_backbone(model_id: str, pretrained: bool):
    from torchvision.models import video

    builders = {
        "r3d_18": (video.r3d_18, video.R3D_18_Weights.KINETICS400_V1),
        "r2plus1d_18": (video.r2plus1d_18, video.R2Plus1D_18_Weights.KINETICS400_V1),
        "swin3d_t": (video.swin3d_t, video.Swin3D_T_Weights.KINETICS400_V1),
    }
    builder, weights = builders[model_id]
    model = builder(weights=weights if pretrained else None)
    # Public model attributes; forward retains native global pooling.
    if model_id == "swin3d_t":
        feature_dim = model.head.in_features
        model.head = nn.Identity()
    else:
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    transform = weights.transforms()
    return model, feature_dim, list(transform.mean), list(transform.std), list(transform.crop_size), str(weights)


def _dino_source_root(model: nn.Module) -> Path:
    for parent in Path(inspect.getfile(type(model))).resolve().parents:
        if (parent / "hubconf.py").is_file() and (parent / "dinov2").is_dir():
            return parent
    raise RuntimeError("Cannot locate the DINOv2 source required for an offline encoder bundle")


def _source_hash(source_root: Path) -> str:
    digest = hashlib.sha256()
    files = [source_root / "hubconf.py", *sorted((source_root / "dinov2").rglob("*.py"))]
    for path in files:
        digest.update(str(path.relative_to(source_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def subclip_timestamps(start_sec: float, end_sec: float, frame_count: int, fps: float = 15.0, subclips: int = 3) -> np.ndarray:
    """Native-rate dense clips spaced across the complete requested window.

    Short windows repeat their final frame. Three dense subclips do leave gaps
    in long windows; their mean feature is a lightweight frozen-backbone baseline.
    """
    if not np.isfinite([start_sec, end_sec, fps]).all() or start_sec < 0 or end_sec < start_sec or fps <= 0 or frame_count < 1 or subclips < 1:
        raise ValueError("Invalid temporal window or subclip configuration")
    span = (frame_count - 1) / fps
    starts = np.linspace(start_sec, max(start_sec, end_sec - span), subclips)
    return np.minimum(starts[:, None] + np.arange(frame_count)[None, :] / fps, end_sec)


class FrozenEncoder:
    def __init__(self, model: nn.Module, config: dict, device: str | torch.device):
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.device = torch.device(device)
        self.config = dict(config)
        self.feature_dim = int(config["feature_dim"])
        self.sequence_length = int(config["sequence_length"])
        self.cache = VideoFrameCache(side=int(config["cache_side"]), fps=float(config["cache_fps"]), max_bytes=int(config["cache_max_bytes"]))

    def clear_cache(self) -> None:
        """Release the last video so end-to-end latency includes decoding."""
        self.cache.clear()

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        # Cache frames are already square letterboxes. Resizing the square keeps
        # their original aspect ratio and includes the complete camera scene.
        batch = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
        size = self.config["spatial_size"]
        if list(batch.shape[-2:]) != list(size):
            batch = F.interpolate(batch, size=size, mode="bilinear", align_corners=False, antialias=True)
        mean = torch.tensor(self.config["mean"], dtype=batch.dtype)[None, :, None, None]
        std = torch.tensor(self.config["std"], dtype=batch.dtype)[None, :, None, None]
        return (batch - mean) / std

    def encode(self, path: str | Path, start_sec: float, end_sec: float) -> torch.Tensor:
        if not np.isfinite([start_sec, end_sec]).all() or start_sec < 0 or end_sec < start_sec:
            raise ValueError("Encoding requires a finite nonnegative time window")
        self.model.eval()
        with torch.no_grad():
            if self.config["model_id"] == "dinov2_gru":
                times = np.linspace(start_sec, end_sec, self.sequence_length)
                frames = self._preprocess(self.cache.sample(path, times))
                # Small batches keep the frozen ViT comfortable on Colab GPUs.
                features = [self.model(batch.to(self.device)).float().cpu() for batch in frames.split(8)]
                result = torch.cat(features)
            else:
                times = subclip_timestamps(start_sec, end_sec, int(self.config["clip_frames"]), self.config["native_fps"], self.config["subclips"])
                features = []
                for clip_times in times:
                    frames = self._preprocess(self.cache.sample(path, clip_times))
                    clip = frames.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
                    features.append(self.model(clip).float().cpu())
                result = torch.stack(features).mean(dim=0)
        if result.shape != (self.sequence_length, self.feature_dim) or not torch.isfinite(result).all():
            raise RuntimeError(f"Encoder produced invalid feature shape/values: {tuple(result.shape)}")
        return result.contiguous()

    def save_bundle(self, directory: str | Path) -> Path:
        """Save weights plus local DINO source, so deployment needs no network."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.config["model_id"] == "dinov2_gru":
            source = _dino_source_root(self.model)
            destination = directory / "dinov2_source"
            destination.mkdir(parents=True, exist_ok=True)
            # Python sources are sufficient for the public ViT-S/14 constructor;
            # deliberately omit datasets, checkpoints, notebooks, docs and assets.
            for path in [source / "hubconf.py", *sorted((source / "dinov2").rglob("*.py"))]:
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            for name in ("LICENSE", "LICENSE.txt", "NOTICE"):
                if (source / name).is_file():
                    shutil.copy2(source / name, destination / name)
        torch.save({name: value.detach().cpu() for name, value in self.model.state_dict().items()}, directory / "encoder_state.pt")
        (directory / "encoder_config.json").write_text(json.dumps(self.config, indent=2, sort_keys=True), encoding="utf-8")
        return directory


def make_encoder(model_id: str, device: str | torch.device, pretrained: bool = True) -> FrozenEncoder:
    """Production calls use pretrained=True; False is for offline smoke tests."""
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown model_id: {model_id!r}")
    import torchvision

    config = {
        "bundle_format_version": 1,
        "model_id": model_id,
        "pretrained": bool(pretrained),
        "preprocessing_version": PREPROCESSING_VERSION,
        "spatial_policy": "full_view_letterbox_native_crop_size_no_center_crop",
        "timestamp_policy": "source_pts_relative_to_first_frame_nearest_cached_pts",
        "cache_side": 224,
        "cache_fps": 15.0,
        "cache_max_bytes": 384 * 1024**2,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
    }
    if model_id == "dinov2_gru":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained, trust_repo=True)
        config.update({
            "backbone": "dinov2_vits14", "weights": "facebookresearch/dinov2:dinov2_vits14" if pretrained else "random_test_only",
            "feature_dim": 384, "sequence_length": 16, "spatial_size": [224, 224],
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "temporal_policy": "16_uniform_frames_across_window", "dino_source_sha256": _source_hash(_dino_source_root(model)),
        })
    else:
        model, dimension, mean, std, crop, weights_name = _video_backbone(model_id, pretrained)
        config.update({
            "backbone": model_id, "weights": weights_name if pretrained else "random_test_only",
            "feature_dim": dimension, "sequence_length": 1, "spatial_size": crop,
            "mean": mean, "std": std, "clip_frames": 32 if model_id == "swin3d_t" else 16,
            "native_fps": 15.0, "subclips": 3, "temporal_policy": "3_dense_native_rate_subclips_spaced_across_window_mean_features",
        })
    return FrozenEncoder(model, config, device)


def load_encoder_bundle(directory: str | Path, device: str | torch.device = "cpu") -> FrozenEncoder:
    """Construct only from local source/weights, without any pretrained download."""
    directory = Path(directory)
    config = json.loads((directory / "encoder_config.json").read_text(encoding="utf-8"))
    if config.get("bundle_format_version") != 1 or config.get("model_id") not in MODEL_IDS:
        raise ValueError("Unsupported encoder bundle configuration")
    if config["model_id"] == "dinov2_gru":
        source = directory / "dinov2_source"
        if _source_hash(source) != config.get("dino_source_sha256"):
            raise ValueError("DINO source bundle does not match the exported source hash")
        model = torch.hub.load(str(source.resolve()), "dinov2_vits14", source="local", pretrained=False)
    else:
        model, _, _, _, _, _ = _video_backbone(config["model_id"], pretrained=False)
    model.load_state_dict(torch.load(directory / "encoder_state.pt", map_location="cpu", weights_only=True), strict=True)
    return FrozenEncoder(model, config, device)

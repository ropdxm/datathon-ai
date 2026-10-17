"""Optional frozen original OpenAI CLIP ViT-B/32 frame representation.

All image inference happens in the current Python process. Only the public
checkpoint is downloaded; recordings and embeddings are never transmitted.
This is a visual representation comparison, not evidence that particular
actions or violations occur in this dataset. No generative model is used.

Install ``open_clip_torch==3.2.0``. Original OpenAI CLIP and OpenCLIP code use
MIT licenses: https://github.com/openai/CLIP/blob/main/LICENSE and
https://github.com/mlfoundations/open_clip/blob/v3.2.0/LICENSE . The checkpoint
URL/hash is pinned to the original OpenAI release, rather than LAION weights.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
import warnings

import cv2
import numpy as np

from .features import (_canonical_json, _configuration_hash, _record_value,
                       cache_is_valid, iter_video_frames, load_feature_cache,
                       make_views, sha256_file)


BACKBONE = "clip_vit_b32_openai"
HUB_REPO = "openai/CLIP:a1d071733d7111c9c014f024669f959182114e33"
OPEN_CLIP_VERSION = "3.2.0"
EXPECTED_WEIGHT_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
WEIGHTS_URL = f"https://openaipublic.azureedge.net/clip/models/{EXPECTED_WEIGHT_SHA256}/ViT-B-32.pt"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class ClipFeatureConfig:
    backbone: str = BACKBONE
    sample_fps: float = 2.0
    image_size: int = 224
    batch_size: int = 32
    device: str = "auto"
    views: tuple[str, ...] = ("full", "top", "middle", "bottom")
    local_checkpoint: str | None = None
    allow_download: bool = True
    local_repo: str | None = None
    use_text_features: bool = False

    def __post_init__(self):
        if self.backbone != BACKBONE:
            raise ValueError(f"This extractor supports only {BACKBONE}")
        if not math.isfinite(self.sample_fps) or self.sample_fps <= 0:
            raise ValueError("sample_fps must be finite and positive")
        if self.image_size != 224:
            raise ValueError("Original CLIP ViT-B/32 requires image_size=224")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self.views or len(set(self.views)) != len(self.views) or set(self.views) - {"full", "top", "middle", "bottom"}:
            raise ValueError("views must contain distinct supported crop names")
        if self.local_repo is not None:
            raise ValueError("CLIP uses the installed pinned open_clip_torch package; omit local_repo")
        if self.use_text_features:
            raise ValueError("This backbone exports image embeddings only; use_text_features must be False")


def _resolve_clip_weights(config: ClipFeatureConfig, torch: Any) -> Path:
    explicit = config.local_checkpoint is not None
    checkpoint = (Path(config.local_checkpoint).expanduser().resolve() if explicit else
                  Path(torch.hub.get_dir()) / "checkpoints" / "clip_openai_ViT-B-32.pt")
    if checkpoint.is_file() and sha256_file(checkpoint) == EXPECTED_WEIGHT_SHA256:
        return checkpoint
    if explicit:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing original CLIP checkpoint: {checkpoint}")
        raise ValueError(f"CLIP checkpoint SHA-256 differs from the pinned original OpenAI release: {checkpoint}")
    if not config.allow_download:
        raise FileNotFoundError(f"Verified offline CLIP weights not found: {checkpoint}; supply local_checkpoint")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=checkpoint.parent, suffix=".downloading", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.hub.download_url_to_file(WEIGHTS_URL, str(temporary), progress=True)
        if sha256_file(temporary) != EXPECTED_WEIGHT_SHA256:
            raise ValueError("Downloaded CLIP weights failed SHA-256 validation")
        os.replace(temporary, checkpoint)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint


def _open_clip_version() -> str:
    try:
        return version("open_clip_torch")
    except PackageNotFoundError as exc:
        raise ImportError("Install open_clip_torch==3.2.0 for the optional CLIP comparison") from exc


def _load_clip_encoder(checkpoint: Path, device: Any) -> Any:
    # The loader reconstructs the original QuickGELU architecture from its JIT
    # state dict. Passing the verified path prevents any hidden model download.
    from open_clip.openai import load_openai_model
    model = load_openai_model(str(checkpoint), device="cpu", precision="fp32")
    return model.eval().requires_grad_(False).to(device=device)


def _encode_clip_views(images: np.ndarray, encoder: Any, config: ClipFeatureConfig, device: Any, torch: Any) -> np.ndarray:
    if images.ndim != 4 or images.shape[1:] != (224, 224, 3) or not len(images):
        raise ValueError("Expected nonempty uint8 RGB images[N,224,224,3]")
    if images.dtype != np.uint8:
        raise ValueError("CLIP input images must be uint8 RGB")
    mean = torch.tensor(CLIP_MEAN, device=device).reshape(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).reshape(1, 3, 1, 1)
    output, offset, batch_size = [], 0, config.batch_size
    while offset < len(images):
        tensor = embedding = None
        try:
            chunk = images[offset : offset + batch_size]
            tensor = torch.from_numpy(np.ascontiguousarray(chunk.transpose(0, 3, 1, 2))).to(
                device=device, dtype=torch.float32).div_(255.)
            tensor = (tensor - mean) / std
            autocast = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with torch.inference_mode(), autocast:
                embedding = encoder.encode_image(tensor)
                embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
            values = embedding.cpu().numpy().astype(np.float32, copy=False)
            if values.shape != (len(chunk), 512) or not np.isfinite(values).all():
                raise ValueError("Expected finite 512-dimensional original CLIP image features")
            output.append(values)
            offset += len(chunk)
        except torch.OutOfMemoryError as exc:
            if device.type != "cuda" or batch_size == 1:
                raise RuntimeError("CLIP encoder exhausted memory at its minimum batch size") from exc
            del tensor, embedding
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            warnings.warn(f"CLIP CUDA memory pressure: retrying batch_size={batch_size}", RuntimeWarning)
    return np.concatenate(output, axis=0)


def extract_clip_features(records: Iterable[Any], cache_dir: str | Path,
                          config: ClipFeatureConfig | None = None) -> dict[str, Path]:
    """Return SHA-keyed NPZ paths with frozen ``features[T,V,512]``.

    Only source path and optional SHA are consumed from each record. Labels,
    names and annotated points never influence decoding, crops or embeddings.
    Use a separate cache directory for each encoder/configuration comparison.
    """
    import torch
    config = config or ClipFeatureConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("CLIP extraction supports CPU and CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    package_version = _open_clip_version()
    if package_version != OPEN_CLIP_VERSION:
        raise RuntimeError(f"Install open_clip_torch=={OPEN_CLIP_VERSION}; found {package_version}")
    checkpoint = _resolve_clip_weights(config, torch)
    signature = {
        "schema_version": 1, "backbone": BACKBONE,
        "sample_fps": float(config.sample_fps), "image_size": config.image_size,
        "views": list(config.views), "hub_repo": HUB_REPO,
        "weights_sha256": EXPECTED_WEIGHT_SHA256,
        "preprocessing": "rgb-clipnorm-letterbox-imagenetpad-square-overlap-v1;prompts=none",
        "decoder": "opencv-sequential-monotonic-pts-final-valid-v1",
        "encoder": "open_clip.openai.load_openai_model-quickgelu",
        "use_text_features": False, "prompts_sha256": _configuration_hash("[]"),
        "open_clip_version": package_version, "torch_version": str(torch.__version__),
        "opencv_version": str(cv2.__version__),
        "precision": "cuda-float16" if device.type == "cuda" else "cpu-float32",
    }
    config_json = _canonical_json(signature)
    provenance = {
        "checkpoint": str(checkpoint.resolve()), "hub_repo": HUB_REPO,
        "weights_sha256": EXPECTED_WEIGHT_SHA256, "weights_url": WEIGHTS_URL,
        "implementation": f"mlfoundations/open_clip:v{OPEN_CLIP_VERSION}",
        "license": "MIT (original OpenAI CLIP and OpenCLIP)",
        "device": str(device), "batch_size_requested": config.batch_size,
        "inference_location": "local_python_process", "external_inference_api_used": False,
    }
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output, encoder = {}, None
    for record in records:
        path = _record_value(record, "path", _record_value(record, "video_path"))
        if path is None:
            raise ValueError("Each record needs a path or video_path")
        path = Path(path)
        digest = sha256_file(path)
        expected = _record_value(record, "sha256")
        if expected and str(expected).lower() != digest:
            raise ValueError(f"Source changed after manifest generation: {path}")
        if digest in output:
            continue
        destination = cache_dir / f"{digest}.npz"
        if cache_is_valid(destination, digest, config_json):
            output[digest] = destination
            continue
        if encoder is None:
            torch.manual_seed(0)
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cuda.matmul.allow_tf32 = False
            encoder = _load_clip_encoder(checkpoint, device)
        blocks, times, pending = [], [], []
        frames_per_batch = max(1, config.batch_size // len(config.views))
        for timestamp, rgb in iter_video_frames(path, config.sample_fps):
            times.append(timestamp)
            pending.append(make_views(rgb, config.image_size, config.views))
            if len(pending) >= frames_per_batch:
                values = _encode_clip_views(np.concatenate(pending), encoder, config, device, torch)
                blocks.append(values.reshape(len(pending), len(config.views), 512))
                pending.clear()
        if pending:
            values = _encode_clip_views(np.concatenate(pending), encoder, config, device, torch)
            blocks.append(values.reshape(len(pending), len(config.views), 512))
        if not blocks:
            raise ValueError(f"No decodable CLIP samples: {path}")
        features = np.concatenate(blocks).astype(np.float32, copy=False)
        timestamps = np.asarray(times, dtype=np.float32)
        handle = tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False)
        temporary = Path(handle.name)
        handle.close()
        try:
            np.savez_compressed(temporary, features=features, timestamps=timestamps,
                sha256=np.array(digest), config_json=np.array(config_json),
                config_hash=np.array(_configuration_hash(config_json)),
                provenance_json=np.array(_canonical_json(provenance)))
            load_feature_cache(temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        output[digest] = destination
        print(f"CLIP features {len(output)}: {digest[:12]}, {len(times)} frames x {len(config.views)} views x 512", flush=True)
    return output

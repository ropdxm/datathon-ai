"""Frozen, directly executed Qwen3-VL embeddings for uniform video windows.

No hosted inference, HTTP model server, generation, captions, or LLM API is
used. Public weights are downloaded once; pixels stay inside the current
Python runtime. The native Transformers multimodal base model produces numeric
hidden states, pooled using the official embedding model's last-token method.

Official references (verified 2026-10-17):
https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B (Apache-2.0)
https://huggingface.co/docs/transformers/model_doc/qwen3_vl
https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/
  9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda/scripts/qwen3_vl_embedding.py

We use ordered image inputs for bounded four-frame contexts, rather than
running the repository's remote helper or its video downloader. The system
instruction, chat template and last-nonpadding-token pooling follow the model's
embedding protocol. No textual outputs are produced or retained.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from .features import cache_is_valid, iter_video_frames, load_feature_cache, sha256_file


MODEL_SPECS = {
    "qwen3_vl_embedding_2b": ("Qwen/Qwen3-VL-Embedding-2B", "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"),
    "qwen3_vl_2b": ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"),
}
DEFAULT_INSTRUCTION = (
    "Represent the visible actions and object interactions in the user's ordered "
    "cashier-workstation frames. Focus on changes in positions, hand movements, "
    "cash or document handling and workstation occupancy. Describe only visible "
    "behavior; do not identify people or decide legality."
)
USER_CONTEXT = "These consecutive frames show one short video window, ordered from earliest to latest."


@dataclass(frozen=True)
class LocalVLMConfig:
    backbone: str = "qwen3_vl_embedding_2b"
    sample_fps: float = .25  # Window centers, independent of labels/anchors.
    image_size: int = 448   # Longest side and maximum square pixel budget.
    batch_size: int = 1    # Windows; four images are placed in each context.
    device: str = "auto"
    views: tuple[str, ...] = ("full",)
    local_checkpoint: str | None = None  # A complete Hugging Face model directory.
    allow_download: bool = True
    clip_seconds: float = 8.
    frames_per_window: int = 4
    quantize_4bit: bool = False
    instruction: str = DEFAULT_INSTRUCTION

    def __post_init__(self):
        if self.backbone not in MODEL_SPECS:
            raise ValueError("Only the pinned original Apache-2.0 Qwen model variants are supported")
        if not math.isfinite(self.sample_fps) or self.sample_fps <= 0:
            raise ValueError("sample_fps must be finite and positive")
        if not math.isfinite(self.clip_seconds) or self.clip_seconds <= 0:
            raise ValueError("clip_seconds must be finite and positive")
        if self.image_size < 64 or self.image_size % 32:
            raise ValueError("image_size must be a multiple of 32 and >= 64")
        if self.batch_size < 1 or self.frames_per_window < 2:
            raise ValueError("Need a positive batch_size and at least two ordered frames")
        if not self.views or len(set(self.views)) != len(self.views) or set(self.views) - {"full", "top", "middle", "bottom"}:
            raise ValueError("Invalid crop views")
        if not self.instruction.strip():
            raise ValueError("Embedding instruction cannot be empty")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _value(record: Any, key: str):
    return record.get(key) if isinstance(record, Mapping) else getattr(record, key, None)


def _resolve_checkpoint(config: LocalVLMConfig) -> Path:
    if config.local_checkpoint:
        path = Path(config.local_checkpoint).expanduser().resolve()
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("Install transformers>=4.57.1 and huggingface_hub for local Qwen embeddings") from error
        repo_id, revision = MODEL_SPECS[config.backbone]
        # This downloads public model files only. It never uploads input data.
        path = Path(snapshot_download(repo_id=repo_id, revision=revision,
            local_files_only=not config.allow_download,
            allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "LICENSE*"]))
    if not path.is_dir() or not (path / "config.json").is_file() or not list(path.glob("*.safetensors")):
        raise FileNotFoundError("local_checkpoint must contain config.json and original .safetensors weights")
    return path


def checkpoint_fingerprint(path: Path) -> str:
    """Hash actual weights and processor/tokenizer files, not a model name alone."""
    digest = hashlib.sha256()
    selected = sorted(p for p in path.rglob("*") if p.is_file()
                      and p.suffix in {".safetensors", ".json", ".jinja", ".txt", ".model"}
                      and ".cache" not in p.relative_to(path).parts)
    if not selected:
        raise ValueError("Empty model checkpoint")
    for file in selected:
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        digest.update(sha256_file(file).encode("ascii"))
    return digest.hexdigest()


def last_token_pool(hidden_state: Any, attention_mask: Any):
    """The official embedding pooler, valid for either padding direction."""
    import torch

    if hidden_state.ndim != 3 or tuple(attention_mask.shape) != tuple(hidden_state.shape[:2]):
        raise ValueError("Hidden states and attention mask have incompatible shapes")
    if not bool((attention_mask.sum(dim=1) > 0).all()):
        raise ValueError("Cannot pool an entirely padded input")
    last_index = attention_mask.shape[1] - attention_mask.flip(dims=[1]).argmax(dim=1) - 1
    return hidden_state[torch.arange(hidden_state.shape[0], device=hidden_state.device), last_index]


class TransformersWindowEncoder:
    """Direct PyTorch forward execution; there is intentionally no generate call."""

    def __init__(self, checkpoint: Path, config: LocalVLMConfig):
        import torch
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError("Qwen3-VL requires transformers>=4.57.1; use native Transformers support") from error

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if self.device.type not in {"cuda", "cpu"}:
            raise ValueError("Supported devices are CPU and CUDA")
        kwargs = {"local_files_only": True, "trust_remote_code": False,
                  "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                  "attn_implementation": "sdpa"}
        if config.quantize_4bit:
            if self.device.type != "cuda":
                raise ValueError("The optional NF4 path requires CUDA and bitsandbytes")
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
            except ImportError as error:
                raise RuntimeError("Optional 4-bit inference requires accelerate and bitsandbytes") from error
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = {"": str(self.device)}
        self.processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True,
            trust_remote_code=False, min_pixels=4096, max_pixels=config.image_size ** 2,
            padding_side="right")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(checkpoint), **kwargs)
        if not config.quantize_4bit:
            self.model.to(self.device)
        self.model.eval().requires_grad_(False)
        self.processor.tokenizer.padding_side = "right"
        self.provenance = {"torch_version": str(torch.__version__), "device": str(self.device),
            "precision": "nf4-float16" if config.quantize_4bit else ("cuda-float16" if self.device.type == "cuda" else "cpu-float32")}

    def encode(self, clips: Sequence[Sequence[Image.Image]]) -> np.ndarray:
        import torch

        conversations = []
        for clip in clips:
            content = [{"type": "image", "image": image} for image in clip]
            content.append({"type": "text", "text": USER_CONTEXT})
            conversations.append([
                {"role": "system", "content": [{"type": "text", "text": self.config.instruction}]},
                {"role": "user", "content": content},
            ])
        text = self.processor.apply_chat_template(conversations, tokenize=False,
            add_generation_prompt=True, add_vision_id=True)
        # PIL images are already local. No URL, upload, video fetcher, helper
        # server or remote model code participates in preprocessing.
        inputs = self.processor(text=text, images=[image for clip in clips for image in clip],
            padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            # Calling the native multimodal base avoids a vocabulary-sized LM
            # logits tensor. last_hidden_state is the documented base output.
            outputs = self.model.model(**inputs, use_cache=False, return_dict=True)
            hidden = outputs.last_hidden_state
            embedding = last_token_pool(hidden, inputs["attention_mask"])
            embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
        result = embedding.cpu().numpy().astype(np.float32, copy=False)
        del inputs, outputs, hidden, embedding
        return result


def uniform_windows(timestamps: np.ndarray, config: LocalVLMConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return regular center times and nearest frame indices, including endpoints."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if not len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Expected nonempty strictly increasing decoded times")
    last = float(timestamps[-1])
    centers = np.arange(0., last + 1e-7, 1. / config.sample_fps)
    if last - centers[-1] > 1e-5:
        centers = np.r_[centers, last]
    offsets = np.linspace(-config.clip_seconds / 2, config.clip_seconds / 2, config.frames_per_window)
    wanted = np.clip(centers[:, None] + offsets[None], timestamps[0], last)
    right = np.minimum(np.searchsorted(timestamps, wanted), len(timestamps) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(timestamps[right] - wanted) < abs(timestamps[left] - wanted), right, left)
    return centers.astype(np.float32), indices


def _resize_view(rgb: np.ndarray, view: str, size: int) -> Image.Image:
    height, width = rgb.shape[:2]
    if view != "full":
        side = min(height, width)
        fraction = {"top": 0., "middle": .5, "bottom": 1.}[view]
        x, y = int(round((width - side) * fraction)), int(round((height - side) * fraction))
        rgb = rgb[y:y + side, x:x + side]
        height, width = rgb.shape[:2]
    scale = size / max(height, width)
    rgb = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    return Image.fromarray(rgb)


def extract_features(records: Iterable[Any], cache_dir: str | Path,
                     config: LocalVLMConfig | None = None, *, engine: Any = None) -> dict[str, Path]:
    """Extract label-independent numeric caches compatible with model.py.

    Only path/video_path and optional sha256 are read from each record. Uniform
    windows are formed before the encoder sees them. ``engine`` is an optional
    local dependency-injection hook for tests; its encode method must return a
    numeric [batch, embedding] array. No observations are logged.
    """
    config = config or LocalVLMConfig()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _resolve_checkpoint(config)
    repo_id, revision = MODEL_SPECS[config.backbone]
    weights_hash = checkpoint_fingerprint(checkpoint)
    semantic_protocol = {"clip_seconds": config.clip_seconds, "frames_per_window": config.frames_per_window,
                         "instruction": config.instruction, "user_context": USER_CONTEXT,
                         "quantize_4bit": config.quantize_4bit, "pooling": "last-nonpadding-l2",
                         "chat": "system-instruction-ordered-images-generation-prefix-vision-ids"}
    import importlib.metadata
    versions = {}
    for library in ("torch", "transformers", "opencv-python", "Pillow"):
        try:
            versions[library] = importlib.metadata.version(library)
        except importlib.metadata.PackageNotFoundError:
            versions[library] = "not-installed"
    if engine is not None:
        runtime_device = getattr(engine, "provenance", {}).get("device", config.device)
    elif config.device == "auto":
        import torch
        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        runtime_device = config.device
    precision = "nf4-float16" if config.quantize_4bit else ("cuda-float16" if str(runtime_device).startswith("cuda") else "cpu-float32")
    signature = {"schema_version": 1, "backbone": config.backbone, "sample_fps": config.sample_fps,
        "image_size": config.image_size, "views": list(config.views), "weights_sha256": weights_hash,
        "hub_repo": f"{repo_id}@{revision}", "decoder": "opencv-sequential-monotonic-pts-final-valid-v1",
        "preprocessing": "qwen3vl-context-last-token-v1-" + hashlib.sha256(_json(semantic_protocol).encode()).hexdigest(),
        "context_config": semantic_protocol, "requested_device": config.device, "device": str(runtime_device),
        "precision": precision, "library_versions": versions, "extractor_code_sha256": sha256_file(Path(__file__))}
    config_json = _json(signature)
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    provenance = {"checkpoint": str(checkpoint), "model_id": repo_id, "revision": revision,
                  "weights_sha256": weights_hash, "license": "Apache-2.0 (original Qwen model)",
                  "local_forward_only": True, "generation_used": False, "hosted_inference_used": False,
                  "batch_size_requested": config.batch_size}
    result = {}
    for record in records:
        source = _value(record, "path")
        if source is None:
            source = _value(record, "video_path")
        if source is None:
            raise ValueError("Each record needs a local path")
        source = Path(source)
        source_hash = sha256_file(source)
        expected = _value(record, "sha256")
        if expected and str(expected).lower() != source_hash:
            raise ValueError("Source bytes differ from manifest hash")
        if source_hash in result:
            continue
        destination = cache_dir / f"{source_hash}.npz"
        if cache_is_valid(destination, source_hash, config_json):
            result[source_hash] = destination
            continue
        if engine is None:
            engine = TransformersWindowEncoder(checkpoint, config)
        started = time.perf_counter()
        # Low-rate thumbnails keep the complete video's context sampling small.
        decode_fps = max(config.sample_fps * 2, config.frames_per_window / config.clip_seconds)
        frame_times, frames = [], []
        for timestamp, rgb in iter_video_frames(source, decode_fps):
            frame_times.append(timestamp)
            # Retain compact thumbnails; create per-view contexts below.
            scale = min(1., config.image_size / max(rgb.shape[:2]))
            frames.append(cv2.resize(rgb, (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))), interpolation=cv2.INTER_AREA))
        centers, indices = uniform_windows(np.asarray(frame_times), config)
        tasks = [(i, view) for i in range(len(centers)) for view in config.views]
        vectors, offset, batch_size = [], 0, config.batch_size
        while offset < len(tasks):
            batch = tasks[offset:offset + batch_size]
            clips = [[_resize_view(frames[int(j)], view, config.image_size) for j in indices[i]] for i, view in batch]
            try:
                encoded = np.asarray(engine.encode(clips), dtype=np.float32)
            except Exception as error:
                # Retry smaller window batches only for explicit CUDA OOM.
                if error.__class__.__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"} and batch_size > 1:
                    import torch
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    continue
                raise
            if encoded.ndim != 2 or len(encoded) != len(batch) or not encoded.shape[1] or not np.isfinite(encoded).all():
                raise ValueError("Local encoder returned invalid numeric embeddings")
            encoded /= np.maximum(np.linalg.norm(encoded, axis=1, keepdims=True), 1e-8)
            vectors.append(encoded)
            offset += len(batch)
        features = np.concatenate(vectors).reshape(len(centers), len(config.views), -1)
        run_provenance = {**provenance, **getattr(engine, "provenance", {}),
                          "elapsed_seconds": time.perf_counter() - started, "windows": len(centers)}
        handle = tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False)
        temporary = Path(handle.name)
        handle.close()
        try:
            np.savez_compressed(temporary, features=features.astype(np.float32), timestamps=centers,
                sha256=np.array(source_hash), config_json=np.array(config_json), config_hash=np.array(config_hash),
                provenance_json=np.array(_json(run_provenance)))
            load_feature_cache(temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        result[source_hash] = destination
        print(f"Local embeddings {len(result)}: {len(centers)} windows x {len(config.views)} views; {time.perf_counter() - started:.1f}s", flush=True)
    return result

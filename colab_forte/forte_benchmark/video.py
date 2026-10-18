"""PTS-aware, bounded, one-video decoding cache for short surveillance clips.

Frames retain their original presentation times, normalized by the first frame.
The cache keeps the nearest source frame on a 15 Hz grid plus the final frame.
Arbitrary requests then select the nearest cached PTS (at most one grid step of
additional sampling error). No nominal frame rate is used to derive timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class DecodedVideo:
    timestamps: np.ndarray
    frames: list[np.ndarray]
    duration_sec: float


def _letterbox_frame(frame, side: int) -> np.ndarray:
    """Resize in FFmpeg before RGB conversion, retaining the entire field of view."""
    scale = side / max(frame.width, frame.height)
    width = max(1, min(side, round(frame.width * scale)))
    height = max(1, min(side, round(frame.height * scale)))
    pixels = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
    result = np.zeros((side, side, 3), dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    result[top:top + height, left:left + width] = pixels
    return result


class VideoFrameCache:
    """Decode a video once; replacing the video releases its previous RGB frames.

    At the default 224 pixels and 15 Hz, a 145-second video occupies about
    328 MB of RGB arrays, below the 384 MiB cap. Decode holds only two original
    frames at once, and never stacks a second copy of the entire cached video.
    """

    def __init__(self, side: int = 224, fps: float = 15.0, max_bytes: int = 384 * 1024**2):
        if side < 1 or not np.isfinite(fps) or fps <= 0 or max_bytes < side * side * 3:
            raise ValueError("Cache requires a positive size/fps and room for one RGB frame")
        self.side, self.fps, self.max_bytes = int(side), float(fps), int(max_bytes)
        self._key = None
        self._video: DecodedVideo | None = None
        self.nbytes = 0

    def clear(self) -> None:
        """Release decoded RGB data, including before an uncached timing run."""
        self._key, self._video, self.nbytes = None, None, 0

    def get(self, path: str | Path) -> DecodedVideo:
        import av

        path = Path(path).expanduser().resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key == self._key and self._video is not None:
            return self._video
        # Release the previous clip before allocating its replacement.
        self._key, self._video, self.nbytes = None, None, 0
        frames: list[np.ndarray] = []
        times: list[float] = []
        frame_bytes = self.side * self.side * 3

        def retain(frame, timestamp: float) -> None:
            if times and timestamp <= times[-1] + 1e-9:
                return
            if self.nbytes + frame_bytes > self.max_bytes:
                raise MemoryError("Video frame cache exceeded max_bytes; lower side/fps or increase the explicit cache limit")
            frames.append(_letterbox_frame(frame, self.side))
            times.append(float(timestamp))
            self.nbytes += frame_bytes

        try:
            with av.open(str(path)) as container:
                if not container.streams.video:
                    raise ValueError(f"No video stream in {path}")
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                origin = None
                previous = None
                previous_time = 0.0
                next_grid = 0.0
                for frame in container.decode(stream):
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError(f"Video has a frame without presentation timestamps: {path}")
                    absolute_time = float(frame.pts * frame.time_base)
                    if origin is None:
                        origin = absolute_time
                    timestamp = absolute_time - origin
                    if previous is not None and timestamp < previous_time - 1e-7:
                        raise ValueError(f"Video PTS are not in presentation order: {path}")
                    if previous is None:
                        retain(frame, 0.0)
                        next_grid = 1.0 / self.fps
                    while previous is not None and next_grid <= timestamp + 1e-9:
                        if next_grid - previous_time <= timestamp - next_grid:
                            retain(previous, previous_time)
                        else:
                            retain(frame, timestamp)
                        next_grid += 1.0 / self.fps
                    previous, previous_time = frame, timestamp
                if previous is None:
                    raise ValueError(f"Video contains no decoded frames: {path}")
                retain(previous, previous_time)
            self._video = DecodedVideo(np.asarray(times, dtype=np.float64), frames, previous_time)
            self._key = key
            return self._video
        except Exception:
            self._key, self._video, self.nbytes = None, None, 0
            raise

    def sample(self, path: str | Path, timestamps: Sequence[float]) -> np.ndarray:
        requested = np.asarray(timestamps, dtype=np.float64)
        if requested.ndim != 1 or not np.isfinite(requested).all():
            raise ValueError("Requested timestamps must be a finite one-dimensional sequence")
        video = self.get(path)
        if not len(requested):
            return np.empty((0, self.side, self.side, 3), dtype=np.uint8)
        right = np.searchsorted(video.timestamps, requested, side="left")
        right = np.clip(right, 0, len(video.timestamps) - 1)
        left = np.maximum(right - 1, 0)
        choose_left = np.abs(requested - video.timestamps[left]) <= np.abs(video.timestamps[right] - requested)
        indices = np.where(choose_left, left, right)
        return np.stack([video.frames[int(index)] for index in indices])


# Both names are retained for readability at call sites.
FrameCache = VideoFrameCache
_DEFAULT_CACHE = VideoFrameCache()


def sample_frames(path: str | Path, timestamps: Sequence[float], cache: VideoFrameCache | None = None) -> np.ndarray:
    return (cache if cache is not None else _DEFAULT_CACHE).sample(path, timestamps)

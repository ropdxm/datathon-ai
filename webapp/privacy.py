"""Local face detection and pixel redaction; never identifies people or downloads weights.

YuNet boxes are detections, not a guarantee that every face is observable. Any
detector failure raises an error; it never silently publishes the original.
"""
from __future__ import annotations

from fractions import Fraction
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import threading
import uuid

import av
from PIL import Image, ImageFilter


MODEL_PATH = Path(__file__).resolve().parent / "assets/face_detection_yunet_2023mar.onnx"
MAX_PREVIEW_SIDE = 960
MAX_SOURCE_PIXELS = 16_777_216
_DETECTOR = None
_DETECTOR_LOCK = threading.Lock()


def _needs_mirror_check(row, size: tuple[int, int]) -> bool:
    x, y, width, height = map(float, row[:4])
    if len(row) < 15 or not all(math.isfinite(float(value)) for value in row):
        raise RuntimeError("Детектор лиц вернул некорректные ориентиры лица.")
    if width <= 0 or height <= 0:
        raise RuntimeError("Детектор лиц вернул некорректные координаты.")
    large = width * height > size[0] * size[1] * .01 or height > size[1] * .08
    if not large:
        return False
    points = [tuple(map(float, row[index:index + 2])) for index in range(4, 14, 2)]
    center_offset = abs(sum(point[0] for point in points) / 5 - (x + width / 2)) / width
    eye_span = math.hypot(points[0][0] - points[1][0], points[0][1] - points[1][1]) / width
    # A large rectangle with tightly clustered, off-center facial landmarks can
    # be a torso false positive. Confirm only that ambiguous geometry in a
    # mirrored pass; ordinary small faces and profiles keep their original recall.
    return center_offset > .22 and eye_span < .30


def _box_iou(first, second) -> float:
    x, y, width, height = first
    other_x, other_y, other_width, other_height = second
    area = max(0., min(x + width, other_x + other_width) - max(x, other_x)) * max(
        0., min(y + height, other_y + other_height) - max(y, other_y))
    return area / max(1e-9, width * height + other_width * other_height - area)


def _detect_faces(image: Image.Image) -> list[tuple[float, float, float, float]]:
    """Return local YuNet face rectangles; discard landmarks without storing them."""
    global _DETECTOR
    if not MODEL_PATH.is_file():
        raise RuntimeError("Локальная модель детектора лиц YuNet отсутствует.")
    try:
        import cv2
        import numpy as np
        with _DETECTOR_LOCK:
            cv2.setNumThreads(1)
            cv2.ocl.setUseOpenCL(False)
            if _DETECTOR is None:
                _DETECTOR = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", image.size, .6, .3, 5000)
            _DETECTOR.setInputSize(image.size)
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            _, detections = _DETECTOR.detect(bgr)
            mirror_boxes = []
            if detections is not None and any(_needs_mirror_check(row, image.size) for row in detections):
                _, mirrored = _DETECTOR.detect(cv2.flip(bgr, 1))
                for row in mirrored if mirrored is not None else []:
                    if len(row) < 15 or not all(math.isfinite(float(value)) for value in row):
                        raise RuntimeError("Зеркальная проверка вернула некорректное лицо.")
                    if float(row[14]) >= .6:
                        x, y, width, height = map(float, row[:4])
                        mirror_boxes.append((image.width - x - width, y, width, height))
        boxes = []
        if detections is not None:
            for detection in detections:
                x, y, width, height = map(float, detection[:4])
                if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
                    raise RuntimeError("Детектор лиц вернул некорректные координаты.")
                if (_needs_mirror_check(detection, image.size)
                        and not any(_box_iou((x, y, width, height), other) >= .4 for other in mirror_boxes)):
                    continue
                boxes.append((x, y, width, height))
        return boxes
    except Exception as error:
        raise RuntimeError("Не удалось выполнить локальное скрытие лиц: " + str(error)) from error


def _redact_image(image: Image.Image) -> tuple[Image.Image, int]:
    if not isinstance(image, Image.Image) or image.width * image.height > MAX_SOURCE_PIXELS:
        raise ValueError("Недопустимое изображение для скрытия лиц.")
    result = image.convert("RGB").copy()
    boxes = _detect_faces(result)
    for x, y, width, height in boxes:
        margin_x, margin_y = max(4., width * .18), max(4., height * .18)
        rectangle = (max(0, math.floor(x - margin_x)), max(0, math.floor(y - margin_y)),
                     min(result.width, math.ceil(x + width + margin_x)), min(result.height, math.ceil(y + height + margin_y)))
        left, top, right, bottom = rectangle
        if right <= left or bottom <= top:
            raise RuntimeError("Детектор лиц вернул рамку за пределами изображения.")
        region = result.crop(rectangle)
        # Coarse pixels plus Gaussian smoothing suppress facial detail. Only the
        # enlarged face rectangle changes; cash, hands and the rest remain visible.
        cells = (max(1, min(8, region.width // 16)), max(1, min(8, region.height // 16)))
        region = region.resize(cells, Image.Resampling.BOX).resize(region.size, Image.Resampling.NEAREST)
        region = region.filter(ImageFilter.GaussianBlur(radius=max(4., min(region.size) / 8)))
        result.paste(region, (left, top))
    return result, len(boxes)


def blur_faces(image: Image.Image) -> Image.Image:
    """Return an independent RGB image with detected face pixels masked locally."""
    return _redact_image(image)[0]


def redact_video(source: Path, destination: Path) -> dict:
    """Write a private H264/yuv420p preview, every frame masked, without audio.

    Source PTS are preserved relative to the first frame. face_detections counts
    rectangles across frames, never identities or a count of distinct people.
    The destination is replaced only after a complete successful encode.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError("Исходное видео не найдено.")
    if source == destination:
        raise ValueError("Копия со скрытыми лицами не должна заменять исходное видео.")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.partial.mp4")
    frames_processed = face_detections = 0
    origin = previous = None
    durations = {}
    try:
        with av.open(str(source)) as input_video:
            if not input_video.streams.video:
                raise ValueError("В файле отсутствует видеопоток.")
            track = input_video.streams.video[0]
            track.codec_context.thread_count = 1
            time_base = Fraction(track.time_base or Fraction(1, 1_000_000))
            rate = track.average_rate or Fraction(30)
            with av.open(str(temporary), "w", options={"movflags": "+faststart"}) as output_video:
                output = output_video.add_stream("libx264", rate=rate)
                output.pix_fmt = "yuv420p"
                output.time_base = output.codec_context.time_base = time_base
                output.codec_context.thread_count = 2
                output.options = {"crf": "23", "preset": "veryfast", "bf": "0"}

                def mux(packet):
                    if packet.pts in durations:
                        packet.duration = durations[packet.pts]
                    output_video.mux(packet)

                for original in input_video.decode(track):
                    if original.pts is None or original.time_base is None:
                        raise ValueError("У исходного кадра отсутствует метка времени PTS.")
                    if original.width * original.height > MAX_SOURCE_PIXELS or frames_processed >= 1_000_000:
                        raise ValueError("Видео превышает предел безопасной обработки.")
                    absolute = original.pts * original.time_base
                    origin = absolute if origin is None else origin
                    timestamp = absolute - origin
                    if previous is not None and timestamp <= previous:
                        raise ValueError("Нарушен порядок временных меток видео.")
                    picture = original.to_image().convert("RGB")
                    picture.thumbnail((MAX_PREVIEW_SIDE, MAX_PREVIEW_SIDE), Image.Resampling.LANCZOS)
                    # H264 yuv420p requires even dimensions; the full frame is
                    # resized by at most one pixel, never cropped.
                    size = (max(2, picture.width - picture.width % 2), max(2, picture.height - picture.height % 2))
                    if picture.size != size:
                        picture = picture.resize(size, Image.Resampling.LANCZOS)
                    picture, detected = _redact_image(picture)
                    if frames_processed == 0:
                        output.width, output.height = picture.size
                    elif picture.size != (output.width, output.height):
                        raise ValueError("Размер кадров видео меняется во время записи.")
                    frame = av.VideoFrame.from_image(picture)
                    frame.pts, frame.time_base = round(timestamp / time_base), time_base
                    duration = original.duration * original.time_base if original.duration else Fraction(1, 1) / rate
                    frame.duration = max(1, round(duration / time_base))
                    durations[frame.pts] = frame.duration
                    for packet in output.encode(frame):
                        mux(packet)
                    frames_processed += 1
                    face_detections += detected
                    previous = timestamp
                if not frames_processed:
                    raise ValueError("Видео не содержит декодируемых кадров.")
                for packet in output.encode():
                    mux(packet)
        with av.open(str(temporary)) as completed:
            track = completed.streams.video[0]
            duration_sec = float(track.duration * track.time_base)
        temporary.chmod(0o600)
        os.replace(temporary, destination)
        return {"duration_sec": duration_sec, "frames_processed": frames_processed, "face_detections": face_detections}
    finally:
        temporary.unlink(missing_ok=True)


def _redact_child(source: str, destination: str, connection) -> None:
    try:
        connection.send((True, redact_video(Path(source), Path(destination))))
    except Exception as error:
        connection.send((False, f"Не удалось скрыть лица в видео: {type(error).__name__}: {error}"))
    finally:
        connection.close()


def redact_video_with_timeout(source: Path, destination: Path, timeout_seconds: float = 120) -> dict:
    """Run redaction in a spawned daemon child; timeout never publishes originals."""
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("Ограничение времени должно быть положительным конечным числом.")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Копия не должна заменять исходное видео.")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    workspace = Path(tempfile.mkdtemp(prefix=".privacy-", dir=destination.parent))
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_redact_child, args=(str(source), str(workspace / "video.mp4"), sender), daemon=True)
    try:
        process.start()
        sender.close()
        if not receiver.poll(timeout_seconds):
            raise TimeoutError("Превышено время локального скрытия лиц в видео.")
        try:
            success, result = receiver.recv()
        except EOFError as error:
            raise RuntimeError("Процесс скрытия лиц завершился без результата.") from error
        if not success:
            raise RuntimeError(result)
        os.replace(workspace / "video.mp4", destination)
        return result
    finally:
        if process.pid is not None:
            process.join(timeout=.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        receiver.close()
        sender.close()
        shutil.rmtree(workspace, ignore_errors=True)

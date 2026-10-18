"""Pixel-level face masking and bounded source-timeline video redaction."""
from fractions import Fraction
import base64
import importlib
import io

import av
from PIL import Image
import pytest


def privacy():
    return importlib.import_module("webapp.privacy")


def patterned_image(size=(96, 64)):
    image = Image.new("RGB", size)
    image.putdata([(255, 255, 255) if (x + y) % 2 else (0, 0, 0)
                   for y in range(size[1]) for x in range(size[0])])
    return image


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "source.MOV"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 10)
        for pts in [50, 52, 55, 59, 64]:
            frame = av.VideoFrame.from_image(patterned_image())
            frame.pts, frame.time_base = pts, Fraction(1, 10)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def test_pixels_are_masked_only_in_expanded_face_box_and_input_unchanged(monkeypatch):
    original = patterned_image()
    before = original.tobytes()
    monkeypatch.setattr(privacy(), "_detect_faces", lambda image: [(30, 20, 20, 20)])
    masked = privacy().blur_faces(original)
    assert original.tobytes() == before and masked is not original
    assert masked.getpixel((40, 30)) != original.getpixel((40, 30))
    assert masked.crop((0, 0, 10, 64)).tobytes() == original.crop((0, 0, 10, 64)).tobytes()
    assert masked.crop((75, 0, 96, 64)).tobytes() == original.crop((75, 0, 96, 64)).tobytes()


def test_no_faces_returns_independent_image_and_detection_error_is_not_silent(monkeypatch):
    original = patterned_image()
    monkeypatch.setattr(privacy(), "_detect_faces", lambda image: [])
    masked = privacy().blur_faces(original)
    assert masked is not original and masked.tobytes() == original.tobytes()
    def failed(image):
        raise RuntimeError("Detector unavailable")
    monkeypatch.setattr(privacy(), "_detect_faces", failed)
    with pytest.raises(RuntimeError, match="Detector unavailable"):
        privacy().blur_faces(original)


def test_face_box_crossing_image_edge_is_clamped(monkeypatch):
    monkeypatch.setattr(privacy(), "_detect_faces", lambda image: [(-5, -5, 25, 25)])
    masked = privacy().blur_faces(patterned_image())
    assert masked.size == (96, 64) and masked.getpixel((0, 0)) != (0, 0, 0)


def test_redacted_video_processes_every_frame_and_preserves_relative_pts(video, tmp_path, monkeypatch):
    monkeypatch.setattr(privacy(), "_detect_faces", lambda image: [(30, 20, 20, 20)])
    destination = tmp_path / "private-preview.mp4"
    stats = privacy().redact_video(video, destination)
    assert stats["frames_processed"] == 5 and stats["face_detections"] == 5
    assert stats["duration_sec"] == pytest.approx(1.5, abs=.001)
    with av.open(str(destination)) as output:
        stream = output.streams.video[0]
        assert stream.codec_context.name == "h264" and not output.streams.audio
        frames = list(output.decode(stream))
        assert [frame.time for frame in frames] == pytest.approx([0, .2, .5, .9, 1.4])
        assert all(frame.width <= 960 and frame.height <= 960 for frame in frames)
        assert all(60 < frame.to_image().getpixel((40, 30))[0] < 195 for frame in frames)


def test_detector_failure_never_publishes_unmasked_video(video, tmp_path, monkeypatch):
    destination = tmp_path / "failed.mp4"
    def failed(image):
        raise RuntimeError("Mask failed")
    monkeypatch.setattr(privacy(), "_detect_faces", failed)
    with pytest.raises(RuntimeError, match="Mask failed"):
        privacy().redact_video(video, destination)
    assert not destination.exists() and not list(tmp_path.glob("*.partial.mp4"))


def test_evidence_is_masked_before_jpeg_encoding_and_pts_is_retained(video, monkeypatch):
    from webapp import analysis
    def mask(image):
        return Image.new("RGB", image.size, "#d01020")
    monkeypatch.setattr(privacy(), "blur_faces", mask)
    result = {"duration_sec": 1.5, "violations": [{"timecode_sec": .5}]}
    analysis._attach_evidence(video, result)
    evidence = result["violations"][0]["evidence_frames"]
    assert [round(frame["timestamp_sec"], 4) for frame in evidence] == [0, .5, .9]
    with Image.open(io.BytesIO(base64.b64decode(evidence[1]["image_base64"]))) as image:
        red, green, blue = image.getpixel((40, 30))
        assert red > 190 and green < 35 and blue < 50


def test_missing_weights_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(privacy(), "MODEL_PATH", tmp_path / "missing.onnx")
    with pytest.raises(RuntimeError, match="YuNet|детектор|модел"):
        privacy().blur_faces(patterned_image())


def test_timeout_terminates_child_and_removes_staged_video(video, tmp_path):
    destination = tmp_path / "timeout.mp4"
    with pytest.raises(TimeoutError):
        privacy().redact_video_with_timeout(video, destination, timeout_seconds=.000001)
    assert not destination.exists()
    assert not list(tmp_path.glob(".privacy-*"))


def test_spawned_redaction_publishes_complete_video_and_metadata(video, tmp_path):
    destination = tmp_path / "completed.mp4"
    stats = privacy().redact_video_with_timeout(video, destination, timeout_seconds=20)
    assert stats["frames_processed"] == 5 and isinstance(stats["face_detections"], int)
    assert stats["duration_sec"] == pytest.approx(1.5, abs=.001)
    with av.open(str(destination)) as completed:
        assert len(list(completed.decode(completed.streams.video[0]))) == 5
    assert not list(tmp_path.glob(".privacy-*"))


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timeout_is_rejected(video, tmp_path, timeout):
    with pytest.raises(ValueError):
        privacy().redact_video_with_timeout(video, tmp_path / "no.mp4", timeout_seconds=timeout)


# Captured detector output contains numerical geometry only, no source pixels.
TRUE_FACE = [155.4237, 316.4737, 24.0677, 30.6257,
             166.7443, 328.3295, 174.2178, 328.2975, 172.7495, 334.8299,
             167.6275, 339.2181, 172.6455, 340.2185, .854]
FALSE_TORSO = [169.6986, 412.5927, 66.3889, 94.3764,
               180.4859, 443.0857, 196.3155, 443.2448, 175.5046, 457.0576,
               180.5836, 473.7976, 193.6026, 475.5116, .8155]
SMALL_PROFILE = [178.1204, 300.0895, 18.1016, 24.8200,
                 190.3459, 310.5410, 194.0925, 310.4848, 196.5549, 315.6696,
                 191.8762, 319.1025, 193.2132, 320.2420, .6806]


def detector_rows(monkeypatch, original, mirrored):
    import numpy as np
    class Detector:
        def setInputSize(self, size):
            pass
        def detect(self, pixels):
            rows = original if pixels[0, 0, 0] == 0 else mirrored
            if isinstance(rows, Exception):
                raise rows
            return 1, np.asarray(rows, dtype=np.float32) if rows else None
    monkeypatch.setattr(privacy(), "_DETECTOR", Detector())


def test_large_false_torso_requires_matching_mirrored_detection(monkeypatch):
    image = patterned_image((464, 848))
    mirror = FALSE_TORSO.copy()
    mirror[0] = image.width - mirror[0] - mirror[2]
    mirror[14] = .5516
    detector_rows(monkeypatch, [TRUE_FACE, FALSE_TORSO, SMALL_PROFILE], [mirror])
    boxes = privacy()._detect_faces(image)
    assert len(boxes) == 2
    assert boxes[0] == pytest.approx(TRUE_FACE[:4], abs=.001)
    assert boxes[1] == pytest.approx(SMALL_PROFILE[:4], abs=.001)
    masked = privacy().blur_faces(image)
    assert masked.crop((165, 440, 235, 510)).tobytes() == image.crop((165, 440, 235, 510)).tobytes()


def test_small_profile_and_large_frontal_face_do_not_require_mirror(monkeypatch):
    large = TRUE_FACE.copy()
    for start in (0, 4, 6, 8, 10, 12):
        large[start] = 100 + (TRUE_FACE[start] - TRUE_FACE[0]) * 4
        large[start + 1] = 100 + (TRUE_FACE[start + 1] - TRUE_FACE[1]) * 4
    large[2], large[3] = TRUE_FACE[2] * 4, TRUE_FACE[3] * 4
    detector_rows(monkeypatch, [large, SMALL_PROFILE], RuntimeError("Unexpected mirror check"))
    boxes = privacy()._detect_faces(patterned_image((464, 848)))
    assert len(boxes) == 2 and boxes[0] == pytest.approx(large[:4], abs=.001)


def test_suspicious_box_mirror_failure_remains_explicit(monkeypatch):
    detector_rows(monkeypatch, [FALSE_TORSO], RuntimeError("Mirror detector failed"))
    with pytest.raises(RuntimeError, match="Mirror detector failed"):
        privacy().blur_faces(patterned_image((464, 848)))


def test_mirrored_face_must_match_the_same_region(monkeypatch):
    unrelated = TRUE_FACE.copy()
    unrelated[0] = 464 - unrelated[0] - unrelated[2]
    unrelated[14] = .99
    detector_rows(monkeypatch, [FALSE_TORSO], [unrelated])
    assert privacy()._detect_faces(patterned_image((464, 848))) == []
    matching = FALSE_TORSO.copy()
    matching[0] = 464 - matching[0] - matching[2]
    matching[14] = .75
    detector_rows(monkeypatch, [FALSE_TORSO], [matching])
    boxes = privacy()._detect_faces(patterned_image((464, 848)))
    assert len(boxes) == 1 and boxes[0] == pytest.approx(FALSE_TORSO[:4], abs=.001)


def test_face_padding_preserves_nearby_scene_pixels(monkeypatch):
    original = patterned_image((320, 320))
    monkeypatch.setattr(privacy(), "_detect_faces", lambda image: [(100, 100, 100, 100)])
    masked = privacy().blur_faces(original)
    assert masked.crop((70, 100, 80, 200)).tobytes() == original.crop((70, 100, 80, 200)).tobytes()
    assert masked.getpixel((150, 150)) != original.getpixel((150, 150))

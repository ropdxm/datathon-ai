"""Private, source-video demo preparation; no model or bank fixtures are needed."""
import base64
from fractions import Fraction
import importlib.util
import io
import json
from pathlib import Path

import av
from PIL import Image
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/create-demo.py"


@pytest.fixture
def prepare():
    spec = importlib.util.spec_from_file_location("forte_demo_prepare", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "Исходная камера.MOV"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 10)
        for pts, color in zip([50, 52, 55, 59, 64], ["red", "lime", "blue", "yellow", "magenta"]):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 32), color))
            frame.pts, frame.time_base = pts, Fraction(1, 10)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def test_provided_source_metadata_real_jpegs_and_full_browser_video(prepare, source, tmp_path):
    destination = tmp_path / "private-demo"
    prepare.prepare_demo(source, timecode=.5, label="violation", output_dir=destination)
    result = json.loads((destination / "result.json").read_text())
    assert result["filename"] == "Исходная камера.MOV"
    assert result["demo_source"] == "provided_video" and result["demo"] is True
    assert result["model_name"] == "Исходная разметка"
    assert result["video_url"] == "/api/ui/demo/video"
    assert result["verdict"] == "violation" and result["duration_sec"] == pytest.approx(1.5, abs=1 / 30)
    event = result["violations"][0]
    assert event["rule"] is None and event["timecode_sec"] == .5
    assert "разметк" in event["description"] and "ML" in event["description"]
    assert [round(frame["timestamp_sec"], 4) for frame in event["evidence_frames"]] == [0.0, .5, .9]
    with Image.open(io.BytesIO(base64.b64decode(event["evidence_frames"][1]["image_base64"], validate=True))) as image:
        assert image.format == "JPEG" and image.size == (64, 32)
        red, green, blue = image.convert("RGB").getpixel((32, 16))
        assert blue > 220 and red < 25 and green < 25
    with av.open(str(destination / "video.mp4")) as video:
        assert video.streams.video[0].codec_context.name == "h264" and not video.streams.audio
        frames = list(video.decode(video.streams.video[0]))
        assert len(frames) == 45 and frames[0].time == 0
        assert frames[-1].time == pytest.approx(44 / 30)


def test_negative_source_annotation_has_no_events(prepare, source, tmp_path):
    destination = tmp_path / "negative-demo"
    prepare.prepare_demo(source, timecode=.5, label="no_violation", output_dir=destination)
    result = json.loads((destination / "result.json").read_text())
    assert result["verdict"] == "no_violation" and result["violations"] == []
    assert result["limitations"] and (destination / "video.mp4").is_file()


@pytest.mark.parametrize("timecode", [-1, 1.51, float("nan"), float("inf")])
def test_invalid_time_does_not_write_output(prepare, source, tmp_path, timecode):
    destination = tmp_path / "invalid-demo"
    with pytest.raises(ValueError):
        prepare.prepare_demo(source, timecode=timecode, output_dir=destination)
    assert not destination.exists()


def test_output_cannot_be_public_static_directory(prepare, source):
    with pytest.raises(ValueError, match="static|публич"):
        prepare.prepare_demo(source, timecode=.5, output_dir=SCRIPT.parents[1] / "webapp/static/real-demo")


def test_missing_source_and_nonempty_output_are_preserved(prepare, source, tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare.prepare_demo(tmp_path / "missing.mov", timecode=.5, output_dir=tmp_path / "absent")
    destination = tmp_path / "keep-files"
    destination.mkdir()
    (destination / "user-document.txt").write_text("preserve")
    with pytest.raises(ValueError):
        prepare.prepare_demo(source, timecode=.5, output_dir=destination)
    assert (destination / "user-document.txt").read_text() == "preserve"


def test_default_output_follows_private_data_directory(prepare, source, tmp_path, monkeypatch):
    monkeypatch.setenv("FORTE_DATA_DIR", str(tmp_path / "custom-runtime"))
    prepare.prepare_demo(source, timecode=.5, label="no_violation")
    assert (tmp_path / "custom-runtime/demo/result.json").is_file()


def test_import_does_not_process_video(monkeypatch):
    def forbidden_processing(*args, **kwargs):
        raise AssertionError("Import must not open or generate video")
    monkeypatch.setattr(av, "open", forbidden_processing)
    spec = importlib.util.spec_from_file_location("forte_demo_import_only", SCRIPT)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_failed_transcode_preserves_previous_complete_demo(prepare, source, tmp_path, monkeypatch):
    destination = tmp_path / "replace-demo"
    prepare.prepare_demo(source, timecode=.5, output_dir=destination)
    previous_video = (destination / "video.mp4").read_bytes()
    previous_manifest = (destination / "result.json").read_bytes()
    monkeypatch.setattr(prepare, "_ffmpeg", lambda: "/usr/bin/false")
    with pytest.raises(RuntimeError, match="ffmpeg"):
        prepare.prepare_demo(source, timecode=.5, label="no_violation", output_dir=destination)
    assert (destination / "video.mp4").read_bytes() == previous_video
    assert (destination / "result.json").read_bytes() == previous_manifest
    assert sorted(path.name for path in destination.parent.glob(".replace-demo-prepare-*")) == []


def test_custom_context_frames_use_requested_source_pts_and_supplied_text(prepare, source, tmp_path):
    destination = tmp_path / "context-demo"
    prepare.prepare_demo(source, timecode=.5, output_dir=destination,
                         title="Наблюдаемое событие", description="Предоставленное описание фрагмента.",
                         evidence_times=[.19, .89, 1.39])
    event = json.loads((destination / "result.json").read_text())["violations"][0]
    assert event["type"] == "Наблюдаемое событие" and event["timecode_sec"] == .5
    assert "Предоставленное описание фрагмента." in event["description"]
    assert "ML" in event["description"] and event["rule"] is None
    assert [round(frame["timestamp_sec"], 4) for frame in event["evidence_frames"]] == [.2, .9, 1.4]
    with Image.open(io.BytesIO(base64.b64decode(event["evidence_frames"][0]["image_base64"]))) as image:
        red, green, blue = image.convert("RGB").getpixel((32, 16))
        assert green > 220 and red < 25 and blue < 25


@pytest.mark.parametrize("arguments", [
    {"title": " "}, {"title": 7}, {"title": "x" * 201},
    {"description": ""}, {"description": "x" * 4001},
    {"evidence_times": []}, {"evidence_times": [0, .2, .5, .9]},
    {"evidence_times": [float("nan")]}, {"evidence_times": [float("inf")]},
    {"evidence_times": [True]}, {"evidence_times": [".5"]},
    {"evidence_times": [-.1]}, {"evidence_times": [1.51]},
])
def test_invalid_custom_context_is_rejected_before_output(prepare, source, tmp_path, arguments):
    destination = tmp_path / "invalid-context"
    with pytest.raises(ValueError):
        prepare.prepare_demo(source, timecode=.5, output_dir=destination, **arguments)
    assert not destination.exists()


def test_cli_accepts_context_options(prepare, source, tmp_path):
    destination = tmp_path / "cli-context"
    assert prepare.main(["--video", str(source), "--timecode", ".5", "--output", str(destination),
                         "--title", "Размеченный фрагмент", "--description", "Данные разметки.",
                         "--evidence-times", ".19", ".89", "1.39"]) == 0
    event = json.loads((destination / "result.json").read_text())["violations"][0]
    assert event["type"] == "Размеченный фрагмент"
    assert [round(frame["timestamp_sec"], 4) for frame in event["evidence_frames"]] == [.2, .9, 1.4]

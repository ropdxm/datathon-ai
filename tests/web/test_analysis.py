"""Adapter contract tests use only local, synthetic video and no pretrained weights."""
from __future__ import annotations

import base64
from fractions import Fraction
import importlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType

import av
from PIL import Image
import pytest


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "исходный фрагмент.MOV"
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


def adapter():
    return importlib.import_module("webapp.analysis")


def callable_path(monkeypatch, payload):
    module = ModuleType("forte_test_analyzer")
    module.analyze = lambda video_path: payload
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return "forte_test_analyzer:analyze"


def positive(timecode=0.5, **fields):
    event = {"type": "Проверка процедуры", "timecode_sec": timecode, "description": "Требуется проверка оператором."}
    event.update(fields)
    return {"verdict": "violation", "violations": [event]}


def bundle(tmp_path):
    path = tmp_path / "bundle"
    (path / "encoder").mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_id": "r3d_18", "feature_dim": 512, "threshold": 0.5,
        "protocol": {"window_sec": 8.0, "stride_sec": 4.0}}))
    (path / "head.pt").write_bytes(b"test-only model boundary double")
    (path / "encoder" / "encoder_config.json").write_text(json.dumps({"model_id": "r3d_18", "bundle_format_version": 1}))
    (path / "encoder" / "encoder_state.pt").write_bytes(b"test-only encoder boundary double")
    return path


def test_unconfigured_is_unavailable_even_when_environment_is_set(video, monkeypatch):
    monkeypatch.setenv("FORTE_ANALYZER", callable_path(monkeypatch, {"verdict": "no_violation", "violations": []}))
    info = adapter().capabilities()
    assert info["ready"] is False and info["mode"] == "unconfigured"
    assert info["model_name"] is None and info["limitations"]
    with pytest.raises(RuntimeError, match="модел|анализатор|настро"):
        adapter().analyze_video(video)


def test_callable_is_imported_and_receives_original_path(video, tmp_path, monkeypatch):
    (tmp_path / "local_forte_adapter.py").write_text(
        "from pathlib import Path\ndef run(video_path):\n"
        "    assert isinstance(video_path, Path) and video_path.name == 'исходный фрагмент.MOV'\n"
        "    return {'verdict': 'no_violation', 'violations': [], 'limitations': ['Только тестовая процедура.']}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    info = adapter().capabilities(analyzer_callable="local_forte_adapter:run")
    assert info["ready"] is True and info["mode"] == "model"
    assert info["model_name"] == "local_forte_adapter:run"
    result = adapter().analyze_video(video, analyzer_callable="local_forte_adapter:run")
    assert result["verdict"] == "no_violation" and result["violations"] == []
    assert result["duration_sec"] == pytest.approx(1.5)
    assert "Только тестовая процедура." in result["limitations"]


@pytest.mark.parametrize("path,error", [("missing.module:run", ModuleNotFoundError),
    ("forte_test_analyzer:missing", AttributeError), ("forte_test_analyzer:SomeClass", ValueError),
    ("forte_test_analyzer:async_run", ValueError)])
def test_configured_callable_is_loaded_and_validated_only_during_analysis(video, monkeypatch, path, error):
    callable_path(monkeypatch, {})
    module = sys.modules["forte_test_analyzer"]
    module.SomeClass = type("SomeClass", (), {})
    async def async_run(video_path):
        return {}
    module.async_run = async_run
    info = adapter().capabilities(analyzer_callable=path)
    assert info["ready"] is True and info["mode"] == "model"
    with pytest.raises(error):
        adapter().analyze_video(video, analyzer_callable=path)


@pytest.mark.parametrize("path", ["no_colon", "a:b:c", "a..b:run", "module:", ":run", "module:some.function"])
def test_malformed_callable_configuration_is_unavailable(video, path):
    assert adapter().capabilities(analyzer_callable=path)["ready"] is False
    with pytest.raises(RuntimeError):
        adapter().analyze_video(video, analyzer_callable=path)


def test_capabilities_does_not_execute_configured_module(video, tmp_path, monkeypatch):
    marker = tmp_path / "module-executed"
    source = tmp_path / "uninitialized_forte_model.py"
    source.write_text("from pathlib import Path\n" + f"Path({str(marker)!r}).write_text('executed')\n"
                      + "raise RuntimeError('MODEL_IMPORT_INSIDE_WORKER_ONLY')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    info = adapter().capabilities(analyzer_callable="uninitialized_forte_model:analyze")
    assert not marker.exists() and "uninitialized_forte_model" not in sys.modules
    assert info["ready"] is True and info["model_name"] == "uninitialized_forte_model:analyze"
    with pytest.raises(RuntimeError, match="MODEL_IMPORT_INSIDE_WORKER_ONLY"):
        adapter().analyze_video(video, analyzer_callable="uninitialized_forte_model:analyze")
    assert marker.read_text() == "executed"


def test_multiple_events_have_real_pts_jpegs_not_callable_images(video, monkeypatch):
    payload = positive(0.51, evidence_frames=[{"timestamp_sec": 1234, "image_base64": "FORGED"}])
    payload["violations"].append(positive(1.31)["violations"][0])
    result = adapter().analyze_video(video, analyzer_callable=callable_path(monkeypatch, payload))
    assert len(result["violations"]) == 2
    actual_pts = {0.0, 0.2, 0.5, 0.9, 1.4}
    for event in result["violations"]:
        assert 1 <= len(event["evidence_frames"]) <= 3
        assert len({frame["timestamp_sec"] for frame in event["evidence_frames"]}) == len(event["evidence_frames"])
        for frame in event["evidence_frames"]:
            assert round(frame["timestamp_sec"], 4) in actual_pts
            encoded = base64.b64decode(frame["image_base64"], validate=True)
            assert encoded.startswith(b"\xff\xd8")
            with Image.open(io.BytesIO(encoded)) as image:
                assert image.format == "JPEG" and image.size == (64, 32)
                if round(frame["timestamp_sec"], 4) == 0.5:
                    red, green, blue = image.convert("RGB").getpixel((32, 16))
                    assert blue > 220 and red < 25 and green < 25
    assert any(abs(frame["timestamp_sec"] - 0.5) < 0.001 for frame in result["violations"][0]["evidence_frames"])


def test_missing_rule_remains_missing_with_explicit_limitation(video, monkeypatch):
    result = adapter().analyze_video(video, analyzer_callable=callable_path(monkeypatch, positive()))
    assert result["violations"][0].get("rule") is None
    assert any("норматив" in item.lower() for item in result["limitations"])


def test_rule_metadata_is_visible_in_jury_description(video, monkeypatch):
    rule = {"title": "Предоставленная инструкция", "clause": "4.2", "url": "https://example.org/rules#4"}
    result = adapter().analyze_video(video, analyzer_callable=callable_path(monkeypatch, positive(rule=rule)))
    event = result["violations"][0]
    assert event["rule"] == rule
    assert all(value in event["description"] for value in rule.values())


def test_rule_may_have_only_url_without_invented_title(video, monkeypatch):
    rule = {"url": "https://example.org/source"}
    result = adapter().analyze_video(video, analyzer_callable=callable_path(monkeypatch, positive(rule=rule)))
    event = result["violations"][0]
    assert event["rule"] == rule and rule["url"] in event["description"]


def test_evidence_response_budget_is_enforced(video, monkeypatch):
    monkeypatch.setattr(adapter(), "MAX_EVIDENCE_BYTES", 1)
    with pytest.raises(ValueError, match="Объём кадров"):
        adapter().analyze_video(video, analyzer_callable=callable_path(monkeypatch, positive()))


def test_decode_budget_cannot_be_bypassed_with_negative_prediction(video, monkeypatch):
    monkeypatch.setattr(adapter(), "MAX_DECODED_FRAMES", 2)
    path = callable_path(monkeypatch, {"verdict": "no_violation", "violations": []})
    with pytest.raises(ValueError, match="число декодированных"):
        adapter().analyze_video(video, analyzer_callable=path)


@pytest.mark.parametrize("payload", [None, [], {}, {"verdict": "maybe", "violations": []},
    {"verdict": "no_violation", "violations": positive()["violations"]},
    {"verdict": "violation", "violations": []},
    {"verdict": "no_violation", "violations": {}},
    {"verdict": "no_violation", "violations": [], "limitations": "hidden warning"},
    positive(True), positive("0.5"), positive(float("nan")), positive(float("inf")), positive(-0.01), positive(1.51),
    positive(type=""), positive(type=12), positive(description=None), positive(description="x" * 4001),
    {"verdict": "violation", "violations": positive()["violations"] * 21},
    positive(rule="made-up rule"), positive(rule={"url": "javascript:alert(1)"}),
    positive(rule={"url": "https://user:secret@example.org/rules"}), positive(rule={"url": "https://example.org/\nheader"})])
def test_invalid_or_inconsistent_model_output_fails(video, monkeypatch, payload):
    with pytest.raises(ValueError):
        adapter().analyze_video(video, analyzer_callable=callable_path(monkeypatch, payload))


def test_malformed_video_cannot_become_negative_result(tmp_path, monkeypatch):
    broken = tmp_path / "broken.mov"
    broken.write_bytes(b"not a video")
    path = callable_path(monkeypatch, {"verdict": "no_violation", "violations": []})
    with pytest.raises(ValueError, match="видео|видеопоток|декод"):
        adapter().analyze_video(broken, analyzer_callable=path)


def test_custom_callable_failure_propagates_without_negative_prediction(video, monkeypatch):
    path = callable_path(monkeypatch, {})
    def failed(_):
        raise NotImplementedError("Подключите модель команды")
    sys.modules["forte_test_analyzer"].analyze = failed
    with pytest.raises(NotImplementedError, match="модель команды"):
        adapter().analyze_video(video, analyzer_callable=path)


def test_incomplete_bundle_is_not_ready(tmp_path):
    info = adapter().capabilities(bundle_path=tmp_path)
    assert info["ready"] is False and info["mode"] == "unconfigured"


@pytest.mark.parametrize("verdict", ["violation", "no_violation"])
def test_binary_bundle_adapter_does_not_invent_event_or_rule(video, tmp_path, monkeypatch, verdict):
    module = ModuleType("colab_forte.forte_benchmark.predict")
    def predict_video(bundle_path, video_path, device="auto"):
        assert Path(video_path) == video and device == "cpu"
        return {"model_id": "r3d_18", "duration_sec": 1.5, "score": .8, "threshold": .5,
            "verdict": verdict, "timecode_sec": .51, "candidate_window": {"start_sec": 0., "end_sec": 1.5},
            "event_type": None, "normative_source": None,
            "timecode_method": "highest-scoring window center; approximate candidate, not event-frame recognition"}
    module.predict_video = predict_video
    monkeypatch.setitem(sys.modules, module.__name__, module)
    local_bundle = bundle(tmp_path)
    info = adapter().capabilities(bundle_path=local_bundle)
    assert info["ready"] is True and "r3d_18" in info["model_name"]
    assert any("бинар" in item.lower() for item in info["limitations"])
    result = adapter().analyze_video(video, bundle_path=local_bundle, device="cpu")
    assert result["verdict"] == verdict and result["limitations"]
    if verdict == "no_violation":
        assert result["violations"] == []
    else:
        event = result["violations"][0]
        assert event["type"] == "Подозрительный фрагмент" and event.get("rule") is None
        assert event.get("event_type") is None and event.get("normative_source") is None
        assert "тип" in event["description"].lower() and "норматив" in event["description"].lower()
        assert "%" not in event["description"]
        assert event["evidence_frames"]


def test_large_frames_are_resized_without_changing_original_aspect(tmp_path, monkeypatch):
    path = tmp_path / "large.mov"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 1600, 800, "yuv420p"
        frame = av.VideoFrame.from_image(Image.new("RGB", (1600, 800), "blue"))
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    result = adapter().analyze_video(path, analyzer_callable=callable_path(monkeypatch, positive(0)))
    evidence = result["violations"][0]["evidence_frames"]
    assert len(evidence) == 1 and evidence[0]["timestamp_sec"] == 0
    with Image.open(io.BytesIO(base64.b64decode(evidence[0]["image_base64"]))) as image:
        assert image.size == (1280, 640)


def test_extremely_large_integer_timestamp_is_rejected(video, monkeypatch):
    with pytest.raises(ValueError):
        adapter().analyze_video(video, analyzer_callable=callable_path(monkeypatch, positive(10 ** 1000)))


def test_async_callable_object_fails_during_analysis(video, monkeypatch):
    path = callable_path(monkeypatch, {})
    class AsyncAnalyzer:
        async def __call__(self, video_path):
            return {"verdict": "no_violation", "violations": []}
    sys.modules["forte_test_analyzer"].analyze = AsyncAnalyzer()
    assert adapter().capabilities(analyzer_callable=path)["ready"] is True
    with pytest.raises(ValueError, match="синхронной"):
        adapter().analyze_video(video, analyzer_callable=path)


def test_callable_takes_precedence_over_bundle(video, tmp_path, monkeypatch):
    path = callable_path(monkeypatch, {"verdict": "no_violation", "violations": []})
    result = adapter().analyze_video(video, bundle_path=tmp_path / "missing", analyzer_callable=path)
    assert result["verdict"] == "no_violation"

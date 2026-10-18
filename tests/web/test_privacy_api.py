"""Privacy preview transport must never expose the original upload as fallback.

The pixel-level redaction tests live with the privacy engine. These HTTP tests
replace only the expensive engine with a deterministic synthetic MP4 writer;
session authorization, multipart admission, storage and media responses are real.
"""

import base64
from fractions import Fraction
import io
from pathlib import Path
import time

import av
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from webapp.api import create_app
from webapp.config import Settings


ORIGIN = {"Origin": "http://testserver"}
ORIGINAL = b"PRIVATE ORIGINAL MUST NEVER BE RETURNED"


def _write_video(path: Path, color="blue"):
    with av.open(str(path), "w", format="mp4") as output:
        stream = output.add_stream("mpeg4", rate=5)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 5)
        for index in range(5):
            frame = av.VideoFrame.from_image(Image.new("RGB", (32, 32), color))
            frame.pts, frame.time_base = index, Fraction(1, 5)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


@pytest.fixture
def rendered_video(tmp_path):
    path = tmp_path / "expected-redacted.mp4"
    _write_video(path)
    return path.read_bytes()


def _app(tmp_path, **overrides):
    return create_app(Settings(data_dir=tmp_path / "private-runtime", **overrides))


def _submit(client, filename="Камера.MOV", content=ORIGINAL, headers=ORIGIN):
    return client.post("/api/ui/previews", headers=headers,
                       files={"file": (filename, content, "video/quicktime")})


def test_preview_requires_issued_session_and_same_origin(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        assert _submit(client).status_code == 401
        client.get("/api/ui/config")
        for headers in ({}, {"Origin": "https://other.example"}, {"Origin": "null"}):
            assert _submit(client, headers=headers).status_code == 403


@pytest.fixture
def redaction_stub(monkeypatch, rendered_video):
    def redact(source, destination, **kwargs):
        assert Path(source).is_file()
        Path(destination).write_bytes(rendered_video)
        return {"duration_sec": 1.0, "frames_processed": 5, "face_detections": 1}

    monkeypatch.setattr("webapp.previews.redact_video_with_timeout", redact)


def test_preview_works_without_model_and_serves_only_redacted_owner_media(
        tmp_path, redaction_stub, rendered_video):
    app = _app(tmp_path)
    with TestClient(app) as owner:
        assert owner.get("/api/ui/config").json()["ready"] is False
        response = _submit(owner)
        assert response.status_code == 200, response.text
        metadata = response.json()
        assert metadata["preview_id"]
        assert metadata["video_url"].startswith("/api/ui/previews/")
        assert metadata["duration_sec"] == 1.0
        assert metadata["privacy"]
        assert response.headers["cache-control"] == "no-store"
        video_url = metadata["video_url"]
        media = owner.get(video_url)
        assert media.status_code == 200
        assert media.headers["content-type"] == "video/mp4"
        assert media.headers["cache-control"] == "no-store"
        assert media.content == rendered_video
        assert ORIGINAL not in media.content
        segment = owner.get(video_url, headers={"Range": "bytes=4-15"})
        assert segment.status_code == 206
        assert segment.content == rendered_video[4:16]
        assert owner.get(video_url, headers={"Range": "bytes=999999-"}).status_code == 416
        assert owner.get(video_url, headers={"Origin": "https://other.example"}).status_code == 403
        with TestClient(app) as stranger:
            assert stranger.get(video_url).status_code == 401
            stranger.get("/api/ui/config")
            assert stranger.get(video_url).status_code == 404
        with app.state.jobs._db() as database:
            assert database.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        for path in app.state.settings.data_dir.rglob("*"):
            if path.is_file():
                assert ORIGINAL not in path.read_bytes(), path


@pytest.mark.parametrize("filename,content,status", [
    ("empty.mov", b"", 422),
    ("wrong.txt", b"private", 415),
    ("large.mov", b"x" * 65, 413),
])
def test_invalid_preview_upload_is_rejected_without_retaining_source(
        tmp_path, filename, content, status):
    app = _app(tmp_path, max_upload_bytes=64)
    with TestClient(app) as client:
        client.get("/api/ui/config")
        response = _submit(client, filename, content)
        assert response.status_code == status, response.text
        assert not list(app.state.settings.data_dir.rglob("*.mov"))
        assert not list(app.state.settings.data_dir.rglob("*.part"))


def test_redaction_failure_returns_error_and_deletes_partial_output(tmp_path, monkeypatch):
    def failed(source, destination, **kwargs):
        Path(destination).write_bytes(b"PARTIAL PRIVATE OUTPUT")
        raise RuntimeError("detector failed with PRIVATE INTERNAL DETAIL")

    monkeypatch.setattr("webapp.previews.redact_video_with_timeout", failed)
    app = _app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.get("/api/ui/config")
        response = _submit(client)
        assert response.status_code == 503, response.text
        assert ORIGINAL not in response.content
        assert "PRIVATE INTERNAL DETAIL" not in response.text
        assert not list(app.state.settings.data_dir.rglob("*.mov"))
        assert not list(app.state.settings.data_dir.rglob("*.mp4"))
        assert not list(app.state.settings.data_dir.rglob("*.part"))
        for path in app.state.settings.data_dir.rglob("*"):
            if path.is_file():
                assert ORIGINAL not in path.read_bytes(), path
                assert b"PARTIAL PRIVATE OUTPUT" not in path.read_bytes(), path


def test_preview_stream_limit_applies_before_multipart_spooling(tmp_path):
    app = _app(tmp_path, max_upload_bytes=64)

    def body():
        yield b'--preview-boundary\r\nContent-Disposition: form-data; name="file"; filename="x.mov"\r\n\r\n'
        for _ in range(100):
            yield b"x" * 1024
        yield b"\r\n--preview-boundary--\r\n"

    with TestClient(app) as client:
        client.get("/api/ui/config")
        response = client.post("/api/ui/previews", headers={
            **ORIGIN, "Content-Type": "multipart/form-data; boundary=preview-boundary"},
            content=body())
        assert response.status_code == 413
        assert not list(app.state.settings.data_dir.rglob("*.mov"))


def test_expired_preview_is_no_longer_served(tmp_path, redaction_stub):
    app = _app(tmp_path, retention_seconds=0.1)
    with TestClient(app) as client:
        client.get("/api/ui/config")
        response = _submit(client)
        assert response.status_code == 200, response.text
        route = response.json()["video_url"]
        assert client.get(route).status_code == 200
        time.sleep(0.15)
        assert client.get(route).status_code == 404
        assert not list(app.state.previews.root.rglob("*.mp4"))


def negative_result(path, **kwargs):
    return {"verdict": "no_violation", "violations": [], "duration_sec": 1.0}


def ready(**kwargs):
    return {"ready": True, "mode": "model", "model_name": "test-only", "limitations": []}


def positive_result(path, **kwargs):
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(output, format="JPEG")
    return {"verdict": "violation", "duration_sec": 1.0, "violations": [{
        "type": "test-only-event", "timecode_sec": 0.2, "description": "Synthetic test event",
        "evidence_frames": [{"timestamp_sec": 0.2,
                             "image_base64": base64.b64encode(output.getvalue()).decode()}],
    }]}


def test_analysis_job_video_uses_redacted_mp4_instead_of_original(
        tmp_path, redaction_stub, rendered_video):
    app = create_app(Settings(data_dir=tmp_path / "private-runtime"),
                     analyzer=negative_result, capabilities_fn=ready)
    with TestClient(app) as client:
        client.get("/api/ui/config")
        response = client.post("/api/ui/analyze", headers=ORIGIN,
                               files={"file": ("source.mov", ORIGINAL, "video/quicktime")})
        assert response.status_code == 202, response.text
        route = "/api/ui/jobs/" + response.json()["job_id"] + "/video"
        media = client.get(route)
        assert media.status_code == 200, media.text
        assert media.headers["content-type"] == "video/mp4"
        assert media.content == rendered_video
        assert ORIGINAL not in media.content
        segment = client.get(route, headers={"Range": "bytes=4-15"})
        assert segment.status_code == 206
        assert segment.content == rendered_video[4:16]


def test_evidence_redaction_failure_never_returns_stored_jpegs_to_ui_or_jury(tmp_path, monkeypatch):
    def failed(image):
        raise RuntimeError("PRIVATE DETECTOR DETAIL")

    monkeypatch.setattr("webapp.previews.blur_faces", failed)
    app = create_app(Settings(data_dir=tmp_path / "private-runtime", api_key="test-only"),
                     analyzer=positive_result, capabilities_fn=ready)
    with TestClient(app) as client:
        client.get("/api/ui/config")
        response = client.post("/api/ui/analyze", headers=ORIGIN,
                               files={"file": ("source.mov", ORIGINAL, "video/quicktime")})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            stored = app.state.jobs.get(job_id)
            if stored["status"] != "processing":
                break
            time.sleep(0.02)
        assert stored["status"] == "done", stored
        original_jpeg = stored["violations"][0]["evidence_frames"][0]["image_base64"]
        for route, headers in [("/api/ui/jobs/" + job_id, {}),
                               ("/api/v1/jobs/" + job_id, {"X-API-Key": "test-only"})]:
            response = client.get(route, headers=headers)
            assert response.status_code == 503, response.text
            assert original_jpeg not in response.text
            assert "PRIVATE DETECTOR DETAIL" not in response.text

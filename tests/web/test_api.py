"""HTTP contract, upload boundaries and private browser-session behavior."""

import base64
import io
import time

import pytest
from fastapi.testclient import TestClient

from webapp.api import create_app
from webapp.config import Settings


KEY = {"X-API-Key": "jury-secret"}
ORIGIN = {"Origin": "http://testserver"}


def negative_result(path, **kwargs):
    return {"verdict": "no_violation", "violations": [], "duration_sec": 1,
            "model_name": "test-model", "limitations": []}


def positive_result(path, **kwargs):
    from PIL import Image
    output = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(output, format="JPEG")
    return {"verdict": "violation", "duration_sec": 2.0, "violations": [{
        "type": "possible_event", "timecode_sec": 1.0, "description": "Тестовое событие",
        "evidence_frames": [{"timestamp_sec": 1.0,
                             "image_base64": base64.b64encode(output.getvalue()).decode()}],
        "rule": {"url": "https://example.org/rule"}}]}


def forged_jpeg_result(path, **kwargs):
    result = positive_result(path)
    result["violations"][0]["evidence_frames"][0]["image_base64"] = base64.b64encode(
        b"\xff\xd8\xffnot a JPEG\xff\xd9").decode()
    return result


def unavailable(**kwargs):
    return {"ready": False, "mode": "unconfigured", "model_name": None,
            "limitations": ["Обученная модель не настроена."]}


def ready(**kwargs):
    return {"ready": True, "mode": "model", "model_name": "test-model", "limitations": []}


@pytest.fixture
def app(tmp_path):
    return create_app(Settings(data_dir=tmp_path, api_key="jury-secret"),
                      analyzer=negative_result, capabilities_fn=ready)


def terminal(client, path, headers=None):
    end = time.monotonic() + 8
    while time.monotonic() < end:
        response = client.get(path, headers=headers)
        assert response.status_code == 200, response.text
        if response.json()["status"] != "processing":
            return response.json()
        time.sleep(0.03)
    pytest.fail("Job did not finish")


def test_jury_key_required_for_upload_and_polling(app):
    with TestClient(app) as client:
        for headers in ({}, {"X-API-Key": "wrong"}):
            assert client.post("/api/v1/analyze", headers=headers,
                               files={"file": ("a.mov", b"video")}).status_code == 401
            assert client.get("/api/v1/jobs/unknown", headers=headers).status_code == 401
        assert client.get("/healthz").json() == {"status": "ok"}


def test_non_ascii_invalid_api_key_is_denied_without_server_error(app):
    with TestClient(app) as client:
        response = client.get("/api/v1/jobs/unknown", headers=[(b"X-API-Key", b"\xff")])
        assert response.status_code == 401


def test_jury_upload_is_async_and_preserves_unicode_filename_safely(app):
    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", headers=KEY,
                               files={"file": ("../../Камера.MOV", b"video", "video/quicktime")})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        job = terminal(client, f"/api/v1/jobs/{job_id}", KEY)
        assert job["status"] == "done"
        assert job["filename"] == "Камера.MOV"
        assert job["verdict"] == "no_violation"
        assert job["violations"] == []
        assert "jury-secret" not in response.text + str(job)


def test_positive_result_has_actual_jpeg_evidence_and_optional_rule_title(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, api_key="jury-secret"),
                     analyzer=positive_result, capabilities_fn=ready)
    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", headers=KEY,
                               files={"file": ("a.mov", b"video")})
        job = terminal(client, "/api/v1/jobs/" + response.json()["job_id"], KEY)
        assert job["status"] == "done", job
        event = job["violations"][0]
        assert job["verdict"] == "violation"
        assert event["rule"] == {"url": "https://example.org/rule"}
        from PIL import Image
        with Image.open(io.BytesIO(base64.b64decode(event["evidence_frames"][0]["image_base64"]))) as image:
            assert image.format == "JPEG"


def test_malformed_jpeg_produces_failed_job_without_verdict(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, api_key="jury-secret"),
                     analyzer=forged_jpeg_result, capabilities_fn=ready)
    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", headers=KEY,
                               files={"file": ("a.mov", b"video")})
        job = terminal(client, "/api/v1/jobs/" + response.json()["job_id"], KEY)
        assert job["status"] == "failed"
        assert "verdict" not in job


def test_importing_application_does_not_create_runtime_before_startup(tmp_path):
    runtime = tmp_path / "private-runtime"
    app = create_app(Settings(data_dir=runtime, api_key="jury-secret"),
                     analyzer=negative_result, capabilities_fn=ready)
    assert not runtime.exists()
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert runtime.is_dir()


@pytest.mark.parametrize("filename,content,status", [
    ("empty.mov", b"", 422), ("script.html", b"video", 415), ("x.mov", b"x" * 65, 413),
])
def test_invalid_uploads_are_rejected_without_leaking_files(tmp_path, filename, content, status):
    app = create_app(Settings(data_dir=tmp_path, api_key="jury-secret", max_upload_bytes=64),
                     analyzer=negative_result, capabilities_fn=ready)
    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", headers=KEY,
                               files={"file": (filename, content)})
        assert response.status_code == status, response.text
        assert list(app.state.jobs.upload_dir.iterdir()) == []


def test_chunked_request_body_is_bounded_before_multipart_spooling(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, api_key="jury-secret", max_upload_bytes=64),
                     analyzer=negative_result, capabilities_fn=ready)
    def body():
        yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="a.mov"\r\n\r\n'
        for _ in range(100):
            yield b"x" * 1024
        yield b"\r\n--boundary--\r\n"
    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", headers={**KEY, "Content-Type": "multipart/form-data; boundary=boundary"},
                               content=body())
        assert response.status_code == 413
        assert list(app.state.jobs.upload_dir.iterdir()) == []


def test_unconfigured_model_has_explicit_503_and_no_job(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, api_key="jury-secret"),
                     analyzer=negative_result, capabilities_fn=unavailable)
    with TestClient(app) as client:
        config = client.get("/api/ui/config")
        assert config.json()["ready"] is False
        assert config.json()["max_upload_mb"] == 100
        assert config.headers["cache-control"] == "no-store"
        response = client.post("/api/v1/analyze", headers=KEY,
                               files={"file": ("a.mov", b"video")})
        assert response.status_code == 503
        assert "модель" in response.json()["detail"].lower()


def test_ui_jobs_and_source_video_are_owned_by_http_only_session(app):
    with TestClient(app) as owner:
        config = owner.get("/api/ui/config")
        assert "httponly" in config.headers["set-cookie"].lower()
        assert "samesite=strict" in config.headers["set-cookie"].lower()
        assert "jury-secret" not in config.text
        response = owner.post("/api/ui/analyze", headers=ORIGIN,
                              files={"file": ("a.mov", b"0123456789", "video/quicktime")})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        route = f"/api/ui/jobs/{job_id}"
        assert terminal(owner, route)["status"] == "done"
        source = owner.get(route + "/video", headers={"Range": "bytes=2-5"})
        # This deliberately malformed source must never be served raw as a
        # fallback when the privacy renderer cannot decode/redact it.
        assert source.status_code == 503
        assert b"0123456789" not in source.content
        assert source.headers["cache-control"] == "no-store"
        assert owner.get("/static/../uploads/a.mov").status_code == 404
        with TestClient(app) as stranger:
            stranger.get("/api/ui/config")
            assert stranger.get(route).status_code == 404
            assert stranger.get(route + "/video").status_code == 404
        assert owner.get(f"/api/v1/jobs/{job_id}", headers=KEY).status_code == 200


def test_browser_writes_require_matching_origin_and_issued_session(app):
    with TestClient(app) as client:
        assert client.post("/api/ui/analyze", headers=ORIGIN,
                           files={"file": ("x.mov", b"video")}).status_code == 401
        client.get("/api/ui/config")
        for headers in ({}, {"Origin": "https://attacker.example"}, {"Origin": "null"}):
            assert client.post("/api/ui/analyze", headers=headers,
                               files={"file": ("x.mov", b"video")}).status_code == 403
        assert client.get("/api/ui/config", headers={"Origin": "https://attacker.example"}).status_code == 403


def test_forged_session_cookie_does_not_authenticate(app):
    with TestClient(app) as client:
        client.cookies.set("forte_ui_session", "a" * 64, path="/api/ui")
        assert client.get("/api/ui/jobs/unknown").status_code == 401

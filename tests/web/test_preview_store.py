"""Preview concurrency, crash recovery and raw staging retention regressions."""

import json
from pathlib import Path
import threading

from webapp.config import Settings
import webapp.previews as previews_module


def write_redacted_fixture(source, destination, **kwargs):
    """The store contract needs a completed file, without running video inference."""
    Path(destination).write_bytes(b"completed redacted preview")
    return {"duration_sec": 1.0, "frames_processed": 1, "face_detections": 1}


def start_call(function):
    done = threading.Event()
    outcome = []

    def run():
        try:
            outcome.append(("ok", function()))
        except BaseException as error:
            outcome.append(("error", error))
        finally:
            done.set()

    thread = threading.Thread(target=run)
    thread.start()
    return thread, done, outcome


def blocking_redactor(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def redact(source, destination, **kwargs):
        started.set()
        if not release.wait(5):
            raise TimeoutError("Test did not release the blocked redactor")
        return write_redacted_fixture(source, destination, **kwargs)

    monkeypatch.setattr(previews_module, "redact_video_with_timeout", redact)
    return started, release


def test_cached_video_is_readable_while_another_preview_is_rendering(tmp_path, monkeypatch):
    store = previews_module.PreviewStore(Settings(data_dir=tmp_path))
    monkeypatch.setattr(previews_module, "redact_video_with_timeout", write_redacted_fixture)
    cached_id = "a" * 32
    store.create(tmp_path / "source.mov", "owner", cached_id)
    expected_path = store.root / cached_id / "video.mp4"
    started, release = blocking_redactor(monkeypatch)
    worker, worker_done, worker_result = start_call(
        lambda: store.create(tmp_path / "another.mov", "owner", "b" * 32))
    reader = None
    try:
        assert started.wait(2), "The unrelated preview never entered redaction"
        reader, read_done, read_result = start_call(lambda: store.video_path(cached_id, "owner"))
        assert read_done.wait(.5), "A cached video read waited for unrelated video redaction"
        assert read_result == [("ok", expected_path)]
        assert expected_path.read_bytes() == b"completed redacted preview"
        assert not worker_done.is_set(), "The cached read was only served after rendering finished"
    finally:
        release.set()
        worker.join(2)
        if reader is not None:
            reader.join(2)
    assert worker_result[0][0] == "ok"
    assert not worker.is_alive()


def test_concurrent_render_is_rejected_without_waiting(tmp_path, monkeypatch):
    # Imported here so the other regression cases can first run against old code.
    from webapp.previews import PreviewBusyError

    store = previews_module.PreviewStore(Settings(data_dir=tmp_path))
    started, release = blocking_redactor(monkeypatch)
    first, _, first_result = start_call(
        lambda: store.create(tmp_path / "first.mov", "owner", "a" * 32))
    second = None
    try:
        assert started.wait(2), "The first preview never entered redaction"
        second, second_done, second_result = start_call(
            lambda: store.create(tmp_path / "second.mov", "owner", "b" * 32))
        assert second_done.wait(.5), "The second preview queued behind an active render"
        assert second_result[0][0] == "error"
        assert isinstance(second_result[0][1], PreviewBusyError)
        assert not (store.root / ("b" * 32)).exists(), "A rejected render left incomplete output"
    finally:
        release.set()
        first.join(2)
        if second is not None:
            second.join(2)
    assert first_result[0][0] == "ok"
    assert not first.is_alive()


def test_fixed_preview_id_recovers_an_incomplete_directory(tmp_path, monkeypatch):
    store = previews_module.PreviewStore(Settings(data_dir=tmp_path))
    preview_id = "c" * 32
    incomplete = store.root / preview_id
    incomplete.mkdir()
    (incomplete / "video.mp4").write_bytes(b"unfinished previous attempt")
    source = tmp_path / "original.mov"
    source.write_bytes(b"unchanged original")
    monkeypatch.setattr(previews_module, "redact_video_with_timeout", write_redacted_fixture)

    result = store.create(source, "owner", preview_id)

    assert result["preview_id"] == preview_id
    assert store.video_path(preview_id, "owner") == incomplete / "video.mp4"
    assert (incomplete / "video.mp4").read_bytes() == b"completed redacted preview"
    assert json.loads((incomplete / "metadata.json").read_text())["owner"] == "owner"
    assert source.read_bytes() == b"unchanged original"


def test_startup_removes_orphan_raw_preview_uploads_only(tmp_path):
    orphan = tmp_path / "preview-uploads" / "aborted-temporary-upload"
    orphan.mkdir(parents=True)
    (orphan / "source.mov").write_bytes(b"raw bank video left after a crash")
    existing_job_upload = tmp_path / "uploads" / "existing-job.mov"
    existing_job_upload.parent.mkdir()
    existing_job_upload.write_bytes(b"original retained by an existing analysis job")

    previews_module.PreviewStore(Settings(data_dir=tmp_path))

    assert not orphan.exists(), "Startup retained an orphan containing an unredacted video"
    assert existing_job_upload.read_bytes() == b"original retained by an existing analysis job"


def test_old_masking_policy_does_not_reuse_stale_preview(tmp_path, monkeypatch):
    source = tmp_path / "source.mov"
    source.write_bytes(b"original")
    monkeypatch.setattr(previews_module, "redact_video_with_timeout", write_redacted_fixture)
    store = previews_module.PreviewStore(Settings(data_dir=tmp_path))
    old_path = store.get_or_create_for_job(source, "owner", "job-one")
    old_path.write_bytes(b"obsolete oversized masks")
    monkeypatch.setattr(previews_module, "REDACTION_VERSION", "next-mask-policy", raising=False)

    new_path = store.get_or_create_for_job(source, "owner", "job-one")

    assert new_path != old_path
    assert new_path.read_bytes() == b"completed redacted preview"

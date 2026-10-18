"""Behavioral coverage for the durable, deadline-bound inference queue."""

import time
from pathlib import Path

import pytest

from webapp.config import Settings
from webapp.jobs import JobCapacityError, JobManager


def negative_result(path, **kwargs):
    return {"verdict": "no_violation", "violations": [], "duration_sec": 2.0,
            "limitations": [], "model_name": "test-model"}


def blocked_result(path, **kwargs):
    time.sleep(10)
    return negative_result(path)


def invalid_result(path, **kwargs):
    return {"verdict": "no_violation", "violations": [{"type": "invented"}]}


def wait_terminal(manager, job_id, timeout=8):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        job = manager.get(job_id)
        if job["status"] != "processing":
            return job
        time.sleep(0.02)
    pytest.fail("Job did not become terminal")


def settings(tmp_path, **overrides):
    return Settings(data_dir=tmp_path, api_key="test-key", **overrides)


def source(manager):
    path = manager.upload_dir / "test.mov"
    path.write_bytes(b"test input")
    return path


def test_job_returns_async_id_and_persists_valid_negative_result(tmp_path):
    manager = JobManager(settings(tmp_path), analyzer=negative_result)
    manager.start()
    try:
        job_id = manager.submit(source(manager), "Видео.MOV", "owner")
        job = wait_terminal(manager, job_id)
        assert job["verdict"] == "no_violation"
        assert job["violations"] == []
        assert job["model_name"] == "test-model"
        assert job["filename"] == "Видео.MOV"
    finally:
        manager.stop()
    reopened = JobManager(settings(tmp_path), analyzer=negative_result)
    assert reopened.get(job_id)["status"] == "done"


def test_deadline_kills_work_and_late_result_cannot_overwrite_failure(tmp_path):
    manager = JobManager(settings(tmp_path, job_timeout_seconds=0.25), analyzer=blocked_result)
    manager.start()
    try:
        job_id = manager.submit(source(manager), "slow.mov", "owner")
        job = wait_terminal(manager, job_id)
        assert job["status"] == "failed"
        assert "время" in job["error"].lower()
        assert "verdict" not in job
        time.sleep(0.2)
        assert manager.get(job_id)["status"] == "failed"
    finally:
        manager.stop()


def test_capacity_includes_queued_work_and_queue_wait_counts_toward_deadline(tmp_path):
    manager = JobManager(settings(tmp_path, max_pending_jobs=2, job_timeout_seconds=0.3),
                         analyzer=blocked_result)
    first = manager.submit(source(manager), "first.mov", "owner")
    second = manager.submit(source(manager), "second.mov", "owner")
    with pytest.raises(JobCapacityError):
        manager.submit(source(manager), "third.mov", "owner")
    time.sleep(0.35)
    manager.start()
    try:
        assert wait_terminal(manager, first)["status"] == "failed"
        assert wait_terminal(manager, second)["status"] == "failed"
    finally:
        manager.stop()


def test_restart_recovers_abandoned_processing_job_as_failure(tmp_path):
    original = JobManager(settings(tmp_path), analyzer=negative_result)
    job_id = original.submit(source(original), "abandoned.mov", "owner")
    restarted = JobManager(settings(tmp_path), analyzer=negative_result)
    recovered = restarted.get(job_id)
    assert recovered["status"] == "failed"
    assert "verdict" not in recovered


def test_malformed_model_result_is_failure_never_a_negative_verdict(tmp_path):
    manager = JobManager(settings(tmp_path), analyzer=invalid_result)
    manager.start()
    try:
        job_id = manager.submit(source(manager), "malformed.mov", "owner")
        job = wait_terminal(manager, job_id)
        assert job["status"] == "failed"
        assert "verdict" not in job
        assert job["error"]
    finally:
        manager.stop()


def test_cleanup_removes_expired_private_upload_and_job(tmp_path):
    manager = JobManager(settings(tmp_path, retention_seconds=0.1), analyzer=negative_result)
    manager.start()
    try:
        path = source(manager)
        job_id = manager.submit(path, "old.mov", "owner")
        assert wait_terminal(manager, job_id)["status"] == "done"
        time.sleep(0.15)
        manager.cleanup()
        assert manager.get(job_id) is None
        assert not path.exists()
    finally:
        manager.stop()

"""Durable jobs for ONE Uvicorn process with ONE sequential inference worker.

Run with --workers 1 and without --reload. A per-job spawned subprocess lets the
supervisor enforce the full queue + inference deadline and stop stuck native code.
SQLite updates use a processing-only guard so a late result cannot revive a job.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import queue
import secrets
import sqlite3
import threading
import time
import uuid

from .config import Settings
from .schemas import AnalysisResult


TIMEOUT_ERROR = "Превышено допустимое время обработки видео (до 300 секунд с учётом очереди)."
RESTART_ERROR = "Обработка прервана перезапуском сервиса. Загрузите видео повторно."
MODEL_ERROR = "Не удалось обработать видео. Проверьте видеофайл и настройку модели."


class JobCapacityError(Exception):
    pass


def _analyze_child(connection, video_path, analyzer, model_options):
    """Only validated public output crosses the process boundary; never traceback."""
    try:
        if analyzer is None:
            from .analysis import analyze_video
            analyzer = analyze_video
        result = AnalysisResult.model_validate(analyzer(Path(video_path), **model_options))
        encoded = result.model_dump_json(exclude_none=True)
        if len(encoded.encode("utf-8")) > 16 * 1024 * 1024:
            raise ValueError("Result exceeds the response limit")
        connection.send((True, encoded))
    except Exception:
        connection.send((False, MODEL_ERROR))
    finally:
        connection.close()


class JobManager:
    def __init__(self, settings: Settings, analyzer=None):
        self.settings = settings
        self.analyzer = analyzer
        settings.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        settings.data_dir.chmod(0o700)
        self.upload_dir = settings.data_dir / "uploads"
        self.upload_dir.mkdir(exist_ok=True, mode=0o700)
        self.upload_dir.chmod(0o700)
        self.db_path = settings.data_dir / "jobs.sqlite3"
        self._lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(settings.max_pending_jobs)
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self._active_process = None
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, filename TEXT NOT NULL,
                    video_path TEXT NOT NULL, created_at REAL NOT NULL,
                    deadline REAL NOT NULL, finished_at REAL, status TEXT NOT NULL,
                    result_json TEXT, error TEXT
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
            """)
            db.execute("UPDATE jobs SET status='failed', error=?, finished_at=? WHERE status='processing'",
                       (RESTART_ERROR, time.time()))
        self.db_path.chmod(0o600)
        self.cleanup(remove_orphans=True)

    @contextmanager
    def _db(self):
        with self._lock:
            connection = sqlite3.connect(self.db_path, timeout=10)
            connection.row_factory = sqlite3.Row
            try:
                with connection:
                    yield connection
            finally:
                connection.close()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._work, name="forte-inference", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        # The worker terminates its active child within one polling interval.
        with self._db() as db:
            db.execute("UPDATE jobs SET status='failed', error=?, finished_at=? WHERE status='processing'",
                       (RESTART_ERROR, time.time()))

    def reserve_slot(self):
        self.cleanup()
        if not self._slots.acquire(blocking=False):
            raise JobCapacityError("Очередь заполнена. Повторите загрузку позже.")

    def release_slot(self):
        self._slots.release()

    def submit(self, video_path: Path, filename: str, owner: str, *, reserved=False, job_id=None):
        if not reserved:
            self.reserve_slot()
        job_id = job_id or uuid.uuid4().hex
        created = time.time()
        try:
            with self._db() as db:
                db.execute("INSERT INTO jobs (id,owner,filename,video_path,created_at,deadline,status) "
                           "VALUES (?,?,?,?,?,?,'processing')",
                           (job_id, owner, filename, str(video_path), created,
                            created + self.settings.job_timeout_seconds))
            self._queue.put(job_id)
        except Exception:
            self.release_slot()
            raise
        return job_id

    def _record(self, job_id):
        with self._db() as db:
            return db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()

    def get(self, job_id, owner=None):
        record = self._record(job_id)
        if record is None or (owner is not None and record["owner"] != owner):
            return None
        now = time.time()
        if record["status"] == "processing" and now >= record["deadline"]:
            self._finish(job_id, error=TIMEOUT_ERROR)
            record = self._record(job_id)
        output = {
            "job_id": record["id"], "status": record["status"], "filename": record["filename"],
            "created_at": datetime.fromtimestamp(record["created_at"], timezone.utc).isoformat(),
            "elapsed_sec": round(max(0, (record["finished_at"] or now) - record["created_at"]), 3),
        }
        if record["status"] == "done":
            output.update(json.loads(record["result_json"]))
        elif record["status"] == "failed":
            output["error"] = record["error"]
        return output

    def video_path(self, job_id, owner):
        record = self._record(job_id)
        if record is None or record["owner"] != owner:
            return None
        path = Path(record["video_path"])
        return path if path.is_file() else None

    def issue_session(self):
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self._db() as db:
            db.execute("INSERT INTO sessions VALUES (?,?)",
                       (token_hash, time.time() + self.settings.session_ttl_seconds))
        return token

    def session_owner(self, token):
        if not token or len(token) > 200:
            return None
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self._db() as db:
            session = db.execute("SELECT expires_at FROM sessions WHERE token_hash=?",
                                 (token_hash,)).fetchone()
        return token_hash if session is not None and session["expires_at"] > time.time() else None

    def _finish(self, job_id, *, result_json=None, error=None):
        with self._db() as db:
            # The deadline is checked in the transaction as well as by the supervisor.
            record = db.execute("SELECT deadline,status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if record is None or record["status"] != "processing":
                return
            now = time.time()
            if now >= record["deadline"]:
                result_json, error = None, TIMEOUT_ERROR
            db.execute("UPDATE jobs SET status=?,result_json=?,error=?,finished_at=? "
                       "WHERE id=? AND status='processing'",
                       ("failed" if error else "done", result_json, error, now, job_id))

    def _run(self, record):
        remaining = record["deadline"] - time.time()
        if remaining <= 0:
            self._finish(record["id"], error=TIMEOUT_ERROR)
            return
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_analyze_child, args=(
            sender, record["video_path"], self.analyzer,
            {"bundle_path": self.settings.model_bundle,
             "analyzer_callable": self.settings.analyzer_callable, "device": self.settings.device},
        ), daemon=True)
        self._active_process = process
        try:
            process.start()
            sender.close()
            while not self._stop.is_set():
                remaining = record["deadline"] - time.time()
                if remaining <= 0:
                    self._finish(record["id"], error=TIMEOUT_ERROR)
                    return
                if receiver.poll(min(0.1, remaining)):
                    try:
                        ok, payload = receiver.recv()
                    except EOFError:
                        self._finish(record["id"], error=MODEL_ERROR)
                    else:
                        self._finish(record["id"], result_json=payload if ok else None,
                                     error=None if ok else payload)
                    return
                if not process.is_alive():
                    self._finish(record["id"], error=MODEL_ERROR)
                    return
            self._finish(record["id"], error=RESTART_ERROR)
        except Exception:
            self._finish(record["id"], error=MODEL_ERROR)
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
                process.close()
            receiver.close()
            sender.close()
            self._active_process = None

    def _work(self):
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.1)
            except queue.Empty:
                self.cleanup()
                continue
            try:
                record = self._record(job_id)
                if record is not None and record["status"] == "processing":
                    self._run(record)
            finally:
                self.release_slot()
                self._queue.task_done()
                self.cleanup()

    def cleanup(self, *, remove_orphans=False):
        """Bound terminal job count and age; remove source files together with metadata."""
        now = time.time()
        with self._db() as db:
            expired = db.execute(
                "SELECT id,video_path FROM jobs WHERE status != 'processing' AND "
                "(finished_at < ? OR id IN (SELECT id FROM jobs WHERE status != 'processing' "
                "ORDER BY finished_at DESC LIMIT -1 OFFSET ?))",
                (now - self.settings.retention_seconds, self.settings.max_retained_jobs),
            ).fetchall()
            for record in expired:
                Path(record["video_path"]).unlink(missing_ok=True)
                db.execute("DELETE FROM jobs WHERE id=?", (record["id"],))
            db.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            if remove_orphans:
                known = {row[0] for row in db.execute("SELECT video_path FROM jobs")}
                for path in self.upload_dir.iterdir():
                    if path.is_file() and str(path) not in known:
                        path.unlink(missing_ok=True)

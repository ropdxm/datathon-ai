"""Runtime settings. Secrets are read from the environment, never client assets."""

from dataclasses import dataclass, field
import os
from pathlib import Path


MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(".forte-runtime").resolve())
    api_key: str = ""
    model_bundle: Path | None = None
    analyzer_callable: str | None = None
    device: str = "auto"
    max_upload_bytes: int = 100 * MIB
    max_pending_jobs: int = 4
    job_timeout_seconds: float = 300
    retention_seconds: float = 3600
    max_retained_jobs: int = 20
    session_ttl_seconds: float = 86400

    def __post_init__(self):
        object.__setattr__(self, "data_dir", Path(self.data_dir).resolve())
        if self.model_bundle is not None:
            object.__setattr__(self, "model_bundle", Path(self.model_bundle).resolve())
        if not 0 < self.max_upload_bytes <= 100 * MIB:
            raise ValueError("max_upload_bytes must be between 1 and 100 MiB")
        if not 0 < self.job_timeout_seconds <= 300:
            raise ValueError("job_timeout_seconds must be between 0 and 300")
        if min(self.max_pending_jobs, self.max_retained_jobs) < 1:
            raise ValueError("Job capacity must be positive")
        if min(self.retention_seconds, self.session_ttl_seconds) <= 0:
            raise ValueError("Retention must be positive")

    @classmethod
    def from_env(cls):
        bundle = os.environ.get("FORTE_MODEL_BUNDLE", "").strip()
        return cls(
            data_dir=Path(os.environ.get("FORTE_DATA_DIR", ".forte-runtime")),
            api_key=os.environ.get("FORTE_API_KEY", "").strip(),
            model_bundle=Path(bundle) if bundle else None,
            analyzer_callable=os.environ.get("FORTE_ANALYZER", "").strip() or None,
            device=os.environ.get("FORTE_DEVICE", "auto"),
            max_pending_jobs=int(os.environ.get("FORTE_MAX_PENDING_JOBS", "4")),
            job_timeout_seconds=float(os.environ.get("FORTE_JOB_TIMEOUT_SECONDS", "300")),
            retention_seconds=float(os.environ.get("FORTE_RETENTION_SECONDS", "3600")),
            max_retained_jobs=int(os.environ.get("FORTE_MAX_RETAINED_JOBS", "20")),
        )

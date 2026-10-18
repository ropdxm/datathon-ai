"""Validate inference output again before persisting a public API result."""

import base64
import io
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from PIL import Image


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class EvidenceFrame(PublicModel):
    timestamp_sec: float = Field(ge=0, allow_inf_nan=False)
    image_base64: str = Field(min_length=4, max_length=3 * 1024 * 1024)

    @field_validator("image_base64")
    @classmethod
    def jpeg_only(cls, value):
        try:
            decoded = base64.b64decode(value, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Evidence must contain base64 JPEG") from exc
        if not decoded.startswith(b"\xff\xd8\xff") or not decoded.endswith(b"\xff\xd9"):
            raise ValueError("Evidence must contain JPEG")
        try:
            with Image.open(io.BytesIO(decoded)) as image:
                if image.format != "JPEG" or image.width * image.height > 16_777_216:
                    raise ValueError("Invalid evidence dimensions")
                image.load()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError("Evidence must contain a decodable JPEG") from exc
        return value


class Rule(PublicModel):
    title: str | None = Field(default=None, min_length=1, max_length=1000)
    url: str | None = Field(default=None, max_length=2048)
    clause: str | None = Field(default=None, max_length=2000)

    @field_validator("url")
    @classmethod
    def safe_url(cls, value):
        if value is not None:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
                raise ValueError("Rule URL must be HTTP(S)")
        return value


class Violation(PublicModel):
    type: str = Field(min_length=1, max_length=200)
    timecode_sec: float = Field(ge=0, allow_inf_nan=False)
    description: str = Field(min_length=1, max_length=8000)
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)
    rule: Rule | None = None


class AnalysisResult(PublicModel):
    verdict: Literal["violation", "no_violation"]
    violations: list[Violation] = Field(max_length=30)
    limitations: list[str] = Field(default_factory=list, max_length=30)
    duration_sec: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    model_name: str | None = Field(default=None, max_length=300)

    @field_validator("limitations")
    @classmethod
    def bounded_limitations(cls, value):
        if any(len(item) > 4000 for item in value):
            raise ValueError("Limitation text too long")
        return value

    @model_validator(mode="after")
    def verdict_matches_events(self):
        if (self.verdict == "violation") != bool(self.violations):
            raise ValueError("Verdict and violations must agree")
        if self.duration_sec is not None:
            for event in self.violations:
                if event.timecode_sec > self.duration_sec + 0.1:
                    raise ValueError("Event outside source video")
                if any(frame.timestamp_sec > self.duration_sec + 0.1 for frame in event.evidence_frames):
                    raise ValueError("Evidence outside source video")
        return self

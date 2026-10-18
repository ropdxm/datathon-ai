"""Skeleton for a team's real local model; this file never fabricates predictions.

Set FORTE_ANALYZER=examples.custom_analyzer:analyze in the backend environment.
The adapter calls analyze(video_path: pathlib.Path) synchronously, without kwargs.
Load team-owned weights locally; do not send bank videos to external services.
The capabilities endpoint checks only the entry-point syntax and never imports
this module. A configured/ready status does not guarantee successful loading.
Import, callable validation, model initialization, and inference happen inside
the isolated analysis worker and are covered by the job's processing deadline.

Exact return contract (Python dict):
    {
        "verdict": "violation" | "no_violation",
        "violations": [
            {
                "type": str,                 # nonempty, at most 160 characters
                "timecode_sec": int | float, # finite, 0 <= time <= video duration
                "description": str,          # nonempty, at most 4000 characters
                "rule": {                    # optional; omit if unknown
                    "title": str,            # optional, at most 300 characters
                    "clause": str,           # optional, at most 1000 characters
                    "url": str,              # optional HTTP(S), at most 2048 chars
                },
            },
        ],
        "limitations": [str],                # optional, max 20 x 1000 characters
    }

Return a maximum of 20 events. A no_violation result MUST have violations=[].
A violation result MUST have at least one event. Booleans and numeric strings
are not timestamps. Do not invent a type, normative basis, or certainty score:
if the model only detects suspicious windows, say so explicitly in type,
description, and limitations. Optional rule values must be nonempty strings;
URLs must be absolute HTTP(S) URLs without embedded credentials.

The web adapter validates results and extracts 1–3 real JPEG evidence frames
per event from this exact video. Each evidence timestamp is an original frame's
PTS relative to the first source frame. Evidence supplied by a callable is
ignored. Rule metadata is also appended to descriptions for jury compatibility.
Raise a descriptive exception if inference is unavailable or fails; never turn
an initialization failure into a no_violation result.
"""
from pathlib import Path


def analyze(video_path: Path) -> dict:
    """Replace this body with real local inference, then return the contract above."""
    raise NotImplementedError("Подключите обученную модель команды в examples/custom_analyzer.py.")

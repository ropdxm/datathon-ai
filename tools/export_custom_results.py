"""Reproduce the final measured results ZIP using exact privacy allowlists.

Input videos, feature/note caches, translated phrases and optional-note logs are
excluded. Failed optional experiments contribute controlled metadata only.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_custom_colab import export_results as export_base, sha256_file

PREFIX = "runs/custom_v1/"
PHASES = ("clip", "semantic", "annotation", "compare_semantic", "compare_point")
TOP = {"summary.json", "runtime.json", "archive_audit.json", "portable_manifest.json",
       "custom_unlabeled_manifest.json", "custom_aux_manifest.json"}
COMPARISON = {"custom_cv_report.json", "original_only_progress.json", "original_plus_custom_progress.json",
              "model/model.joblib", "model/model_temporal.joblib", "model/final_selection.json"}
UNLABELED = ("unlabeled_cv_report.json", "original_unsupervised_projection_progress.json",
    "original_plus_unlabeled_custom_projection_progress.json", "model/model.joblib", "model/final_selection.json")
DETAIL = ("summary.json", "resolution_448.json", "resolution_896.json", "selection.json",
          "448/summary.json", "896/summary.json")
PRIVILEGED = ("privileged_cv_report.json", "privileged_progress.json",
              "model/model_privileged.joblib", "model/final_selection.json")
AGGREGATES = ("custom_experiment_summary.json", "custom_model_comparison.csv", "custom_model_comparison.png")
METADATA = ("privileged_runtime.json", "privileged_diagnosis.json", "final_test_report.json")
PRIVATE_PARTS = {"data", "features", "clip_features", "semantic_features", "observations", "translations", "by_source"}
PRIVATE_SUFFIXES = {".npz", ".npy", ".mov", ".mp4", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".safetensors", ".pt", ".pth", ".zip"}


def safe_member(name):
    path = PurePosixPath(name)
    parts = {part.lower() for part in path.parts}
    return (not path.is_absolute() and ".." not in path.parts and "\\" not in name
        and not parts & PRIVATE_PARTS and path.suffix.lower() not in PRIVATE_SUFFIXES
        and not any(part.startswith(("privileged_note_cache", "privileged_notes_cache",
            "private_original_notes", "original_privileged_notes")) for part in parts))


def base_member_allowed(name):
    if not safe_member(name):
        return False
    if name == "source_sha256_manifest.json":
        return True
    if name.startswith("source/"):
        return PurePosixPath(name).suffix in {".py", ".json", ".md", ".txt"}
    if not name.startswith(PREFIX):
        return False
    relative = name.removeprefix(PREFIX)
    if relative in TOP or relative in {"annotation/proposals.json", "annotation/protocol.json"}:
        return True
    if relative in {f"logs/{phase}{suffix}" for phase in PHASES
                    for suffix in (".log", "_runtime.json", "_timeout.json")}:
        return True
    return any(relative.startswith(f"comparison_{policy}/")
        and relative.removeprefix(f"comparison_{policy}/") in COMPARISON for policy in ("semantic", "point"))


_SKIP = object()
def numeric_metadata(value, key=""):
    """Retain numeric structure and fixed metadata identifiers, never raw text."""
    if key.lower() in {"text", "notes", "phrases", "translations", "answer", "response", "prompt",
            "raw_notes", "error_message", "traceback", "stdout", "stderr", "log", "vector", "vectors",
            "embeddings", "features", "observations", "windows"}:
        return _SKIP
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float):
        return value if math.isfinite(value) else _SKIP
    if isinstance(value, dict):
        result = {}
        for child_key, child in value.items():
            clean = numeric_metadata(child, str(child_key))
            if clean is not _SKIP:
                result[str(child_key)] = clean
        return result
    if isinstance(value, list):
        return [clean for child in value if (clean := numeric_metadata(child, key)) is not _SKIP]
    if isinstance(value, str):
        if key in {"status", "error_type", "reason_code"} and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,95}", value):
            return value
        if key.endswith("sha256") and re.fullmatch(r"[0-9a-fA-F]{64}", value):
            return value
    return _SKIP


def _contained_file(root, relative):
    candidate = root / relative
    path = candidate.resolve()
    if not path.is_relative_to(root.resolve()) or candidate.is_symlink():
        raise ValueError("Export candidate must remain within its declared result directory")
    return path if path.is_file() else None


def export_final(workspace, output, destination, privileged_dir=None):
    workspace, output, destination = (Path(p).resolve() for p in (workspace, output, destination))
    if not output.is_relative_to(workspace) or output == workspace or destination.suffix.lower() != ".zip":
        raise ValueError("Use a dedicated result directory inside the workspace and a ZIP destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    privileged = None
    if privileged_dir is not None:
        raw = Path(privileged_dir)
        privileged = raw.resolve() if raw.is_absolute() else (workspace / raw).resolve()
        if not privileged.exists() and not raw.is_absolute():
            privileged = (output / raw).resolve()
        if not privileged.is_relative_to(output) or privileged.name not in {"privileged_notes", "privileged_notes_v2"}:
            raise ValueError("Optional notes result directory must be the declared private-safe result child")
    members = {}
    runtime_path = _contained_file(output, "privileged_runtime.json")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path else {}
    include_privileged = privileged is not None and runtime.get("status") == "completed"
    with tempfile.TemporaryDirectory(prefix="datathon_export_", dir=destination.parent) as directory:
        temporary_root = Path(directory).resolve()
        if not temporary_root.is_relative_to(destination.parent.resolve()):
            raise ValueError("Temporary export path escaped the selected destination directory")
        base = temporary_root / "base.zip"
        with redirect_stdout(io.StringIO()):
            export_base(workspace, output, base)
        with ZipFile(base) as archive:
            if archive.testzip() is not None:
                raise ValueError("Base results ZIP failed CRC validation")
            for name in archive.namelist():
                if base_member_allowed(name):
                    if name in members:
                        raise ValueError("Base export contains duplicate members")
                    members[name] = archive.read(name)
        for folder, names in ((output / "unlabeled_adaptation", UNLABELED),
                              (output / "detail_diagnostic", DETAIL)):
            for relative in names:
                path = _contained_file(folder, relative)
                if path:
                    members[PREFIX + path.relative_to(output).as_posix()] = path.read_bytes()
        if include_privileged:
            for relative in PRIVILEGED:
                path = _contained_file(privileged, relative)
                if path:
                    members[PREFIX + path.relative_to(output).as_posix()] = path.read_bytes()
        for relative in METADATA:
            path = _contained_file(output, relative)
            if path:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Controlled runtime/test metadata must be JSON objects")
                clean = numeric_metadata(payload)
                clean["source_file_sha256"] = sha256_file(path)
                members[PREFIX + relative] = json.dumps(clean, indent=2, sort_keys=True, allow_nan=False).encode()
        for relative in AGGREGATES:
            path = _contained_file(workspace / "artifacts", relative)
            if path:
                members["artifacts/" + relative] = path.read_bytes()
        if not all(safe_member(name) for name in members):
            raise ValueError("Private inputs or caches cannot enter the final archive")
        source_manifest = json.loads(members["source_sha256_manifest.json"])
        for name, digest in source_manifest.items():
            if hashlib.sha256(members["source/" + name]).hexdigest() != digest:
                raise ValueError("Source checksum differs after final export")
        receipt = {"format_version": 1, "input_media_included": False, "private_notes_or_caches_included": False,
            "privileged_completed_outputs_included": include_privileged,
            "members_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in sorted(members.items())}}
        members["EXPORT_MANIFEST.json"] = json.dumps(receipt, indent=2, sort_keys=True).encode()
        final = temporary_root / "final.zip"
        with ZipFile(final, "w", ZIP_DEFLATED, compresslevel=9) as archive:
            for name, value in sorted(members.items()):
                info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, value)
        with ZipFile(final) as archive:
            if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
                raise ValueError("Final ZIP CRC/duplicate-member validation failed")
        os.replace(final, destination)
    result = {"path": str(destination), "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination), "members": len(members),
        "privileged_completed_outputs_included": include_privileged,
        "input_media_included": False, "private_notes_or_caches_included": False,
        "source_checksums_verified": True, "crc_and_unique_members_verified": True}
    print("CUSTOM_FINAL_EXPORT_READY", json.dumps(result), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/custom_v1"))
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--privileged-dir", type=Path)
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else workspace / args.output_dir
    export_final(workspace, output, args.destination, args.privileged_dir)


if __name__ == "__main__":
    main()

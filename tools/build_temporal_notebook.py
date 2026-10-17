"""Build the separate, self-contained semantic/temporal Colab notebook.

Run again after source changes to refresh the embedded source and companion ZIP.
The source archive contains code, tests and an original fold-reference report;
it deliberately includes no input recordings, frames, model weights or caches.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
ORGANIZER_FILE_ID = "1H35d4mF1cEUzZYsnViJ0da_PxSmiaiDq"


def make_cell(kind: str, source: str) -> dict:
    cell = {"cell_type": kind, "id": hashlib.sha256((kind + source).encode()).hexdigest()[:12],
            "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def default_baseline(root: Path) -> Path:
    candidates = [root / "dtFinal/runs/clip_b32/evaluation/cv_report.json",
                  root / "reference/clip_baseline_cv.json",
                  root / "artifacts/colab_results/runs/clip_b32/evaluation/cv_report.json"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Provide --baseline-cv pointing to the original CLIP outer-fold cv_report.json")


def source_bundle(root: Path, baseline: Path, measured_summary: Path | None = None) -> bytes:
    required = [root / name for name in (
        "experiment.py", "mil_experiment.py", "temporal_experiment.py", "semantic_head_experiment.py",
        "semantic_predict_experiment.py", "duplicate_aliases.json",
        "tools/run_temporal_colab.py", "tools/propose_event_review.py",
        "tools/build_temporal_notebook.py", "tools/benchmark_semantic_vlm.py", "tools/summarize_temporal.py",
    )]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Source is not ready for packaging: " + ", ".join(missing))
    paths = set(required)
    paths.update(root.glob("*experiment.py"))
    paths.update(root.glob("datathon_ml/*.py"))
    paths.update(root.glob("tests/test_*.py"))
    paths.update(root.glob("tools/*.py"))
    paths.update(root.glob("requirements*.txt"))
    paths.update(path for path in root.glob("README*.md") if path.is_file())
    members = {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(paths)}
    reference = json.loads(baseline.read_text(encoding="utf-8"))
    if not reference.get("predictions") or not reference.get("folds"):
        raise ValueError("The baseline reference must include original outer predictions and folds")
    members["reference/clip_baseline_cv.json"] = baseline.read_bytes()
    if measured_summary is not None:
        summary = json.loads(measured_summary.read_text(encoding="utf-8"))
        legacy = isinstance(summary.get("models"), dict) and bool(summary["models"])
        structured = (isinstance(summary.get("model_comparisons"), list) and bool(summary["model_comparisons"])
            and isinstance(summary.get("posthoc_best_development_model"), dict)
            and isinstance(summary["posthoc_best_development_model"].get("metrics"), dict)
            and isinstance(summary["posthoc_best_development_model"]["metrics"].get("accuracy"), (int, float)))
        if not (legacy or structured):
            raise ValueError("A measured summary must contain completed model results")
        members["reference/measured_temporal_summary.json"] = measured_summary.read_bytes()
    manifest = {"format_version": 1, "contains_recordings": False,
                "members_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in sorted(members.items())}}
    members["SOURCE_MANIFEST.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(members.items()):
            entry = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, content)
    return buffer.getvalue()


def notebook_cells(bundle: bytes, measured: bool = False) -> list[dict]:
    with ZipFile(io.BytesIO(bundle)) as archive:
        test_count = sum(1 for name in archive.namelist() if name.startswith("tests/test_") and name.endswith(".py")
            for node in ast.walk(ast.parse(archive.read(name).decode("utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))
        recorded = (json.loads(archive.read("reference/measured_temporal_summary.json"))
            if measured and "reference/measured_temporal_summary.json" in archive.namelist() else None)
    payload = base64.b64encode(bundle).decode("ascii")
    bundle_hash = hashlib.sha256(bundle).hexdigest()
    bootstrap = f'''# @title Restore the embedded, checksummed project source
import base64, hashlib, io, json, os, sys, zipfile
from pathlib import Path
BASE = Path('/content') if Path('/content').is_dir() else Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
WORKSPACE = BASE / 'datathon_temporal_workspace'
WORKSPACE.mkdir(parents=True, exist_ok=True)
SOURCE_ZIP = base64.b64decode({payload!r})
assert hashlib.sha256(SOURCE_ZIP).hexdigest() == {bundle_hash!r}, 'Embedded source checksum mismatch'
with zipfile.ZipFile(io.BytesIO(SOURCE_ZIP)) as archive:
    for member in archive.infolist():
        target = (WORKSPACE / member.filename).resolve()
        if not target.is_relative_to(WORKSPACE.resolve()):
            raise ValueError('Unsafe source archive path')
    manifest = json.loads(archive.read('SOURCE_MANIFEST.json'))
    for name, checksum in manifest['members_sha256'].items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == checksum, 'Source member checksum mismatch'
    archive.extractall(WORKSPACE)
os.chdir(WORKSPACE)
sys.path.insert(0, str(WORKSPACE))
print('Temporal project ready:', WORKSPACE)
print('Source SHA-256:', hashlib.sha256(SOURCE_ZIP).hexdigest())
'''
    setup = '''import subprocess, sys

def run_command(arguments):
    process = subprocess.Popen(list(map(str, arguments)), cwd=WORKSPACE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, process.args)

run_command([sys.executable, '-m', 'pip', 'install', '-q',
    '-r', 'requirements-ml.txt', '-r', 'requirements-experiments.txt',
    '-r', 'requirements-vlm.txt', 'gdown>=5,<6'])
'''
    gpu = '''import torch
print('Torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
assert torch.cuda.is_available(), 'Select a GPU runtime for the video feature extraction.'
'''
    data = f'''# Preserve an existing DATA_ROOT, or set a path here before running this cell.
DATA_ROOT = globals().get('DATA_ROOT', None)
if DATA_ROOT is not None:
    DATA_ROOT = Path(DATA_ROOT).expanduser().resolve()
else:
    preferred = [WORKSPACE / 'data/open', BASE / 'open', BASE / 'datathon_ml_workspace/open']
    DATA_ROOT = next((path for path in preferred if (path / 'open_labels.xlsx').is_file()), None)
    if DATA_ROOT is None:
        matches = sorted(BASE.rglob('open_labels.xlsx'))
        if matches:
            label_hashes = {{hashlib.sha256(path.read_bytes()).hexdigest() for path in matches}}
            if len(label_hashes) != 1:
                raise ValueError('Several different label workbooks exist; set DATA_ROOT to the intended organizer dataset.')
            DATA_ROOT = matches[0].parent
if DATA_ROOT is None:
    import gdown
    ORGANIZER_ZIP = BASE / 'datathon_original.zip'
    if not ORGANIZER_ZIP.is_file():
        downloaded = gdown.download(id={ORGANIZER_FILE_ID!r}, output=str(ORGANIZER_ZIP), quiet=False)
        if not downloaded:
            raise RuntimeError('Organizer ZIP download failed; set DATA_ROOT to an existing dataset folder.')
    DATA_DIRECTORY = WORKSPACE / 'data'
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ORGANIZER_ZIP) as archive:
        for member in archive.infolist():
            target = (DATA_DIRECTORY / member.filename).resolve()
            if not target.is_relative_to(DATA_DIRECTORY.resolve()):
                raise ValueError('Unsafe dataset archive path')
        archive.extractall(DATA_DIRECTORY)
    matches = list(DATA_DIRECTORY.rglob('open_labels.xlsx'))
    if len(matches) != 1:
        raise ValueError('Expected exactly one organizer label workbook after extraction.')
    DATA_ROOT = matches[0].parent
assert DATA_ROOT.is_dir(), 'DATA_ROOT must be an existing dataset directory.'
print('Dataset directory:', DATA_ROOT)
'''
    tests = f'''# Run all {test_count} regression tests in this source snapshot.
run_command([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
'''
    reference = '''REFERENCE = WORKSPACE / 'reference/clip_baseline_cv.json'
baseline = json.loads(REFERENCE.read_text(encoding='utf-8'))
print('Original CLIP reference metrics:')
print(json.dumps(baseline['metrics'], indent=2))
print('Outer fold count:', len(baseline['folds']))
print('This is the original development reference, not a result from the temporal pipeline.')
'''
    run = '''# Default: the measured best CLIP linear path, frozen-fold CV, fit and export.
SKIP_CLIP = False
SKIP_QWEN = True  # Set False to reproduce the slower, unsuccessful Qwen comparison.
SKIP_HEADS = True  # Set False for fixed 7-candidate linear/RBF/shallow-tree comparison.
arguments = [sys.executable, '-u', 'tools/run_temporal_colab.py',
    '--workspace', WORKSPACE, '--data-root', DATA_ROOT]
if SKIP_CLIP:
    arguments.append('--skip-clip')
if SKIP_QWEN:
    arguments.append('--skip-qwen')
if SKIP_HEADS:
    arguments.append('--skip-heads')
run_command(arguments)
'''
    results = '''RECORDED_SUMMARY = WORKSPACE / 'reference/measured_temporal_summary.json'
if RECORDED_SUMMARY.is_file():
    recorded_summary = json.loads(RECORDED_SUMMARY.read_text(encoding='utf-8'))
    print('Bundled completed development measurements:')
    print(json.dumps(recorded_summary, indent=2))
SUMMARY = WORKSPACE / 'runs/semantic_temporal_v1/summary.json'
if SUMMARY.is_file():
    measured_summary = json.loads(SUMMARY.read_text(encoding='utf-8'))
    print('Measurements from the current workspace run:')
    print(json.dumps(measured_summary, indent=2))
else:
    print('The current workspace has not completed a run yet.')
'''
    export = '''EXPORT = BASE / 'datathon_temporal_results.zip'
if not EXPORT.is_file():
    EXPORT = WORKSPACE / 'artifacts/datathon_temporal_results.zip'
assert EXPORT.is_file(), 'The driver exports completed results; finish the pipeline before downloading.'
print('Results archive:', EXPORT)
print('Bytes:', EXPORT.stat().st_size)
# Optional download in Colab:
# from google.colab import files
# files.download(str(EXPORT))
'''
    status = "**No measured summary is included.** The results cell reads only bundled or newly completed measurements."
    if recorded is not None:
        best = recorded.get("posthoc_best_development_model")
        if best is None:
            best = max(recorded["models"].values(), key=lambda row: row["metrics"]["accuracy"])
        accuracy = best["metrics"]["accuracy"]
        target = recorded.get("target_accuracy", .87)
        target_text = "unmet" if accuracy < target else "reached on development only"
        status = (f"**Measured best development accuracy: {accuracy:.1%}; {target:.0%} target {target_text}.** "
            "This is a posthoc result from the same 40 development recordings, not an independent test. "
            "The original measured summary is included unchanged in `reference/measured_temporal_summary.json`.")
    return [
        make_cell("markdown", "# Observable states and temporal procedure models\n\n"
            + status + f"\n\nThe default reproduces the CLIP semantic-state linear temporal model using the **original frozen outer folds**. The embedded snapshot includes {test_count} regression tests. Slower Qwen and nonlinear-head comparisons are optional. All vision and language inference executes in this runtime. No hosted AI endpoint receives videos, frames or descriptions.\n\n"
            "**Event labels and machine observations remain unverified.** Phone interaction alone is not a legal violation. A fresh longest-video Qwen4B test with batch size 2 failed to finish within the 300-second budget and was stopped; it is excluded from the default path. No independent new-camera accuracy or case-readiness claim is established.\n"),
        make_cell("markdown", "## 1. Restore source\nThe embedded ZIP contains reviewed Python source, tests, explicit duplicate aliases and the original CLIP fold report. Recordings and model weights are downloaded separately. This notebook leaves `datathon.ipynb` unchanged.\n"),
        make_cell("code", bootstrap),
        make_cell("markdown", "## 2. Install dependencies and check the GPU\nUse a Colab GPU runtime. Public checkpoints download into the runtime cache; inference happens locally in Python.\n"),
        make_cell("code", setup),
        make_cell("code", gpu),
        make_cell("markdown", "## 3. Locate or download organizer data\nSet `DATA_ROOT` to an existing dataset directory to reuse a private session upload. Otherwise this cell downloads the organizer ZIP by file ID and extracts it into this workspace. No dataset is sent to an inference service.\n"),
        make_cell("code", data),
        make_cell("markdown", f"## 4. Check the implementation and reference folds\nThe {test_count} regression tests cover sampling, unknown states, source integrity, grouping, temporal rules and train-only head selection. The original report fixes outer test assignments; all fitted preprocessing, calibration and model selection stay inside training folds.\n"),
        make_cell("code", tests),
        make_cell("code", reference),
        make_cell("markdown", "## 5. Reproduce the best measured path\nThe default runs CLIP semantic states and the linear temporal model. Set `SKIP_QWEN=False` to repeat Qwen4B observations, or `SKIP_HEADS=False` to repeat the fixed **7-candidate** comparison: three linear heads, three RBF SVMs and one shallow ExtraTrees head. Those optional branches did not improve the measured best accuracy. Head selection stays inside each training set; separate family metrics are also reported. Picking the highest outer-evaluated family score remains a posthoc development choice.\n\nAll comparisons reuse the frozen outer folds. The driver fits development models, proposes annotations for human review, and exports results. Cached perception stages can resume after interruption. Qwen generation defaults to one window per call to reproduce its development experiment. Its optional CLI `--batch-size 2` can change floating-point observations and did not satisfy the 300-second longest-video deadline. The fixed numeric prompts request no captions, identities, legal judgment or video verdict.\n"),
        make_cell("code", run),
        make_cell("markdown", "## 6. Read measured results and export\nOnly completed measurements appear below. Training-video predictions and machine proposals are not new accuracy evidence. Keep independent new takes and a real second-camera test for acceptance.\n"),
        make_cell("code", results),
        make_cell("code", export),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cv", type=Path)
    parser.add_argument("--measured-summary", type=Path)
    parser.add_argument("--notebook", type=Path, default=ROOT / "datathon_temporal.ipynb")
    parser.add_argument("--bundle-out", type=Path, default=ROOT / "artifacts/datathon_temporal_source_v5.zip")
    args = parser.parse_args(argv)
    baseline = args.baseline_cv or default_baseline(ROOT)
    bundle = source_bundle(ROOT, baseline, args.measured_summary)
    notebook = {"cells": notebook_cells(bundle, args.measured_summary is not None),
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "accelerator": "GPU",
            "colab": {"name": "datathon_temporal.ipynb", "provenance": []}},
        "nbformat": 4, "nbformat_minor": 5}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<temporal-notebook-cell>", "exec")
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    args.notebook.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    args.bundle_out.parent.mkdir(parents=True, exist_ok=True)
    args.bundle_out.write_bytes(bundle)
    print(json.dumps({"notebook": str(args.notebook), "cells": len(notebook["cells"]),
        "bundle": str(args.bundle_out), "bundle_bytes": len(bundle),
        "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()

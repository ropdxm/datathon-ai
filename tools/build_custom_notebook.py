"""Build a clean self-contained custom-data Colab notebook, without media.

The original-fold report and source bytes are embedded and checksummed. An
optional --measured-summary adds completed numeric measurements unchanged.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_temporal_notebook import ORGANIZER_FILE_ID, default_baseline, make_cell, source_bundle


def custom_bundle(root, baseline, measured_summary=None):
    original = source_bundle(root, baseline)
    with ZipFile(io.BytesIO(original)) as archive:
        members = {name: archive.read(name) for name in archive.namelist() if name != "SOURCE_MANIFEST.json"}
    measured = None
    if measured_summary is not None:
        payload = Path(measured_summary).read_bytes()
        measured = json.loads(payload)
        models = measured.get("models")
        if not isinstance(models, dict) or not models:
            raise ValueError("Measured custom summary needs completed model comparisons")
        for model in models.values():
            policies = model.get("policies")
            if not isinstance(policies, dict) or not policies:
                raise ValueError("Measured custom model needs completed policy metrics")
            for metrics in policies.values():
                accuracy = metrics.get("accuracy")
                if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)) or not 0 <= accuracy <= 1:
                    raise ValueError("Measured custom accuracy must be numeric and in [0,1]")
        members["reference/measured_custom_summary.json"] = payload
    manifest = {"format_version": 1, "contains_recordings": False,
                "members_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in sorted(members.items())}}
    members["SOURCE_MANIFEST.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode()
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(members.items()):
            info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, value)
    return buffer.getvalue(), measured


def notebook_cells(bundle, measured):
    with ZipFile(io.BytesIO(bundle)) as archive:
        test_count = sum(isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            for name in archive.namelist() if name.startswith("tests/test_") and name.endswith(".py")
            for node in ast.walk(ast.parse(archive.read(name).decode("utf-8"))))
    payload = base64.b64encode(bundle).decode("ascii")
    checksum = hashlib.sha256(bundle).hexdigest()
    status = ("No custom comparison measurements are bundled yet. The previous best original-corpus "
              "development accuracy was **62.5%**, below the 87% target.")
    if measured is not None:
        scores = [metrics["accuracy"] for model in measured["models"].values() for metrics in model["policies"].values()]
        status = (f"Bundled completed comparisons contain a highest original-fold development accuracy of **{max(scores):.1%}**. "
                  "This is a posthoc development maximum across the listed policies. The results cell shows each measured policy; "
                  "it does not establish independent custom-camera accuracy.")
    bootstrap = f'''# Restore exact source bytes; reuse an existing original feature cache in this workspace.
import base64, hashlib, io, json, os, sys, zipfile
from pathlib import Path
BASE = Path('/content') if Path('/content').is_dir() else Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
WORKSPACE = BASE / 'datathon_ml_workspace'
WORKSPACE.mkdir(parents=True, exist_ok=True)
SOURCE_ZIP = base64.b64decode({payload!r})
assert hashlib.sha256(SOURCE_ZIP).hexdigest() == {checksum!r}, 'Embedded source checksum differs'
with zipfile.ZipFile(io.BytesIO(SOURCE_ZIP)) as archive:
    seen = set()
    for member in archive.infolist():
        target = (WORKSPACE / member.filename).resolve()
        if not target.is_relative_to(WORKSPACE.resolve()) or target in seen:
            raise ValueError('Unsafe embedded source path')
        seen.add(target)
    manifest = json.loads(archive.read('SOURCE_MANIFEST.json'))
    for name, digest in manifest['members_sha256'].items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == digest, 'Source member checksum differs'
    archive.extractall(WORKSPACE)
os.chdir(WORKSPACE)
sys.path.insert(0, str(WORKSPACE))
print('Workspace:', WORKSPACE)
print('Source SHA-256:', hashlib.sha256(SOURCE_ZIP).hexdigest())
'''
    setup = '''import shutil, subprocess, sys

def run_command(arguments):
    process = subprocess.Popen(list(map(str, arguments)), cwd=WORKSPACE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, process.args)

run_command([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-ml.txt',
    '-r', 'requirements-experiments.txt', '-r', 'requirements-vlm.txt', 'gdown>=5,<6'])
if not shutil.which('ffprobe'):
    assert Path('/content').is_dir(), 'Install ffmpeg/ffprobe in this local runtime first.'
    run_command(['apt-get', '-qq', 'update'])
    run_command(['apt-get', '-qq', 'install', '-y', 'ffmpeg'])
import torch
assert torch.cuda.is_available(), 'Select a GPU runtime before continuing.'
print('Torch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
'''
    weights = '''# Public downloads are permitted during setup only; no private input is sent.
from datathon_ml.clip_features import ClipFeatureConfig, _resolve_clip_weights
from datathon_ml.custom_annotation import MODEL_ID, MODEL_REVISION
from huggingface_hub import snapshot_download
CLIP_CHECKPOINT = _resolve_clip_weights(ClipFeatureConfig(allow_download=True), torch)
QWEN_CHECKPOINT = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
    allow_patterns=['*.safetensors', '*.json', '*.jinja', '*.txt', '*.model']))
print('Verified original CLIP checkpoint cached:', CLIP_CHECKPOINT.is_file())
print('Pinned local annotation checkpoint cached:', QWEN_CHECKPOINT.is_dir())
print('Setup downloads complete. The driver uses offline-only loading and a Python socket guard.')
'''
    original = f'''# An existing complete original image cache is reused. Otherwise bootstrap it.
from datathon_ml.data import build_manifest
from datathon_ml.clip_features import extract_clip_features, ClipFeatureConfig
from datathon_ml.semantic_clip import convert_clip_cache
from tools.benchmark_semantic_vlm import blocked_network, configure_offline_environment
ORIGINAL = WORKSPACE / 'runs/clip_b32'
ORIGINAL_MANIFEST = ORIGINAL / 'manifest.json'
REFERENCE = json.loads((WORKSPACE / 'reference/clip_baseline_cv.json').read_text())
expected = {{r['sha256']: r['label'] for r in REFERENCE['predictions']}}
assert len(expected) == 40, 'Expected the original 40 unique sources'
original_records = json.loads(ORIGINAL_MANIFEST.read_text()) if ORIGINAL_MANIFEST.is_file() else None
complete = isinstance(original_records, list) and all((ORIGINAL / 'features' / (r['sha256'] + '.npz')).is_file() for r in original_records)
if not complete:
    DATA_ROOT = globals().get('DATA_ROOT', None)
    if DATA_ROOT is not None:
        DATA_ROOT = Path(DATA_ROOT).expanduser().resolve()
    else:
        candidates = [WORKSPACE / 'open', WORKSPACE / 'data/open', BASE / 'open']
        DATA_ROOT = next((p for p in candidates if (p / 'open_labels.xlsx').is_file()), None)
    if DATA_ROOT is None:
        import gdown
        organizer_zip = BASE / 'datathon_original.zip'
        if not organizer_zip.is_file():
            assert gdown.download(id={ORGANIZER_FILE_ID!r}, output=str(organizer_zip), quiet=False), 'Organizer download failed'
        destination = WORKSPACE / 'organizer_data'
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(organizer_zip) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError('Unsafe organizer archive member')
            archive.extractall(destination)
        matches = list(destination.rglob('open_labels.xlsx'))
        assert len(matches) == 1, 'Select the organizer DATA_ROOT explicitly when several workbooks exist'
        DATA_ROOT = matches[0].parent
    labels_path = DATA_ROOT / 'open_labels.xlsx'
    assert labels_path.is_file(), 'DATA_ROOT must contain the organizer open_labels.xlsx'
    aliases = json.loads((WORKSPACE / 'duplicate_aliases.json').read_text())
    original_records = build_manifest(DATA_ROOT, output_dir=ORIGINAL,
        labels_path=labels_path, verified_aliases=aliases)
    assert {{r['sha256']: r['label'] for r in original_records}} == expected, 'Organizer sources/labels differ from the frozen reference'
    configure_offline_environment()
    with blocked_network() as bootstrap_network:
        extract_clip_features(original_records, ORIGINAL / 'features',
            ClipFeatureConfig(device='cuda', batch_size=32, allow_download=False))
    assert bootstrap_network['attempts'] == 0
    print('Original image extraction network attempts:', bootstrap_network['attempts'])
assert {{r['sha256']: r['label'] for r in original_records}} == expected, 'Cached originals do not match the frozen reference'
SEMANTIC_ORIGINAL = WORKSPACE / 'runs/semantic_temporal_v1/semantic_clip/features'
# Reuses exact existing prototypes; a fresh workspace constructs them on CPU,
# matching the historical original semantic prototype protocol.
configure_offline_environment()
with blocked_network() as semantic_network:
    convert_clip_cache(ORIGINAL / 'features', SEMANTIC_ORIGINAL, device='cpu', offline=True)
assert semantic_network['attempts'] == 0
print('Original unique sources:', len(original_records))
print('Original semantic extraction network attempts:', semantic_network['attempts'])
'''
    upload = '''# Upload the PREPARED silent archive privately using the Colab Files pane.
# Keep your original recordings separately. Do not publish this private archive.
CUSTOM_ARCHIVE = Path(globals().get('CUSTOM_ARCHIVE', BASE / 'custom_videos_silent.zip')).expanduser().resolve()
CUSTOM_ARCHIVE_SHA256 = '9d6ddccba50084bc9b23856e860319d7dcb85e231130f30d7ff4663a0660e343'
assert CUSTOM_ARCHIVE.is_file(), 'Upload custom_videos_silent.zip privately to session storage, then rerun this cell.'
assert hashlib.sha256(CUSTOM_ARCHIVE.read_bytes()).hexdigest() == CUSTOM_ARCHIVE_SHA256, 'Prepared archive differs from this experiment'
print('Private prepared archive verified; bytes:', CUSTOM_ARCHIVE.stat().st_size)
print('Expected prepared files: 39, all video-only; the driver independently verifies every source.')
'''
    run = '''# This explicitly runs a WEAK-LABEL sensitivity experiment. Proposed labels
# are not human-verified and are never used as evaluation targets.
run_command([sys.executable, '-u', 'tools/run_custom_colab.py',
    '--workspace', WORKSPACE, '--archive', CUSTOM_ARCHIVE,
    '--archive-sha256', CUSTOM_ARCHIVE_SHA256, '--output-dir', 'runs/custom_v1',
    '--annotation-timeout', '2700'])
'''
    adaptation = '''# Train with all custom recordings WITHOUT needing their class labels.
# A fresh guarded subprocess fits scaler/PCA on training windows only; original
# labels alone supervise heads and threshold selection on the frozen folds.
GUARDED_ADAPTATION = """
import runpy, sys
from pathlib import Path
workspace = Path(sys.argv[1])
sys.path.insert(0, str(workspace))
from tools.benchmark_semantic_vlm import configure_offline_environment, blocked_network
configure_offline_environment()
sys.argv = ['unlabeled_experiment.py', 'compare',
    '--original-manifest', str(workspace / 'runs/clip_b32/manifest.json'),
    '--unlabeled-manifest', str(workspace / 'runs/custom_v1/custom_unlabeled_manifest.json'),
    '--original-features', str(workspace / 'runs/clip_b32/features'),
    '--unlabeled-features', str(workspace / 'runs/custom_v1/clip_features'),
    '--folds-from', str(workspace / 'reference/clip_baseline_cv.json'),
    '--output-dir', str(workspace / 'runs/custom_v1/unlabeled_adaptation'), '--fit-final']
with blocked_network() as network:
    runpy.run_module('unlabeled_experiment', run_name='__main__')
assert network['attempts'] == 0
print('UNLABELED_NETWORK_ATTEMPTS', network['attempts'], flush=True)
"""
run_command([sys.executable, '-u', '-c', GUARDED_ADAPTATION, WORKSPACE])
'''
    optional_notes = '''# Optional branch: private, UNVERIFIED original-source notes.
# First create the private notes ZIP locally with the repository tools, then
# upload/extract it into this private runtime. Set NOTES_DIR to the directory
# containing index.json/by_source/<SHA>.json (or by_source itself).
# Do not send notes to a hosted AI service.
RUN_PRIVILEGED_NOTES = False
NOTES_DIR = BASE / 'private_original_notes'
PRIVILEGED_REPORT = WORKSPACE / 'runs/custom_v1/privileged_notes/privileged_cv_report.json'
PRIVILEGED_RUNTIME = PRIVILEGED_REPORT.parent / 'optional_runtime.json'
if RUN_PRIVILEGED_NOTES:
    assert NOTES_DIR.is_dir(), 'Set NOTES_DIR to privately extracted owner-SHA JSON files.'
    import time
    started = time.perf_counter()
    PRIVILEGED_REPORT.unlink(missing_ok=True)  # Do not reuse a stale completed report after failure.
    optional_status = {'status': 'failed', 'completed': False, 'notes_verified': False}
    try:
        run_command([sys.executable, '-u', 'privileged_experiment.py', 'compare',
            '--original-manifest', WORKSPACE / 'runs/clip_b32/manifest.json',
            '--features', WORKSPACE / 'runs/clip_b32/features', '--notes-dir', NOTES_DIR,
            '--note-cache', WORKSPACE / 'runs/custom_v1/privileged_note_cache',
            '--folds-from', WORKSPACE / 'reference/clip_baseline_cv.json',
            '--output-dir', PRIVILEGED_REPORT.parent, '--device', 'cuda', '--fit-final'])
        optional_status.update(status='completed', completed=True, returncode=0)
    except subprocess.CalledProcessError as error:
        optional_status.update(error_type=type(error).__name__, returncode=error.returncode)
    finally:
        optional_status['elapsed_seconds'] = time.perf_counter() - started
        PRIVILEGED_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
        PRIVILEGED_RUNTIME.write_text(json.dumps(optional_status, indent=2))
    print('PRIVILEGED_OPTIONAL_STATUS', json.dumps(optional_status))
'''
    aggregate = '''# Aggregate only completed numeric reports, never raw notes or observations.
arguments = [sys.executable, '-u', 'tools/summarize_custom.py',
    '--results-root', WORKSPACE / 'runs/custom_v1',
    '--baseline', WORKSPACE / 'reference/clip_baseline_cv.json',
    '--output-dir', WORKSPACE / 'artifacts']
if globals().get('RUN_PRIVILEGED_NOTES', False) and PRIVILEGED_REPORT.is_file():
    arguments += ['--privileged-report', PRIVILEGED_REPORT]
if globals().get('RUN_PRIVILEGED_NOTES', False) and PRIVILEGED_RUNTIME.is_file():
    arguments += ['--privileged-runtime', PRIVILEGED_RUNTIME]
run_command(arguments)
'''
    results = '''BUNDLED = WORKSPACE / 'reference/measured_custom_summary.json'
CURRENT = WORKSPACE / 'artifacts/custom_experiment_summary.json'
for label, path in [('Bundled completed measurements', BUNDLED), ('Current workspace measurements', CURRENT)]:
    if path.is_file():
        summary = json.loads(path.read_text())
        print(label)
        print(json.dumps({'status': summary.get('status'), 'models': summary.get('models', {}),
            'annotation': summary.get('annotation'), 'custom_unresolved': summary.get('custom_unresolved'),
            'custom_human_verified_labels': summary.get('custom_human_verified_labels', 0),
            'independent_87_percent_accuracy_established': summary.get('independent_87_percent_accuracy_established', False)}, indent=2))
print('Metrics evaluate original held-out videos. Agreement with machine labels is not an accuracy score.')
'''
    export = '''RESULTS_ZIP = BASE / 'datathon_custom_results.zip'
if not RESULTS_ZIP.is_file():
    RESULTS_ZIP = WORKSPACE / 'artifacts/datathon_custom_results.zip'
assert RESULTS_ZIP.is_file(), 'Finish the driver before exporting results.'
print('Results archive:', RESULTS_ZIP, 'bytes:', RESULTS_ZIP.stat().st_size)
print('Contains trusted small heads, source, numeric reports and compact unverified label proposals; excludes videos and feature caches.')
print('Simple proposed labels and timestamps:', WORKSPACE / 'runs/custom_v1/annotation/proposals.json')
# Only exact known numeric reports and fitted heads are candidates for export.
UNLABELED_RESULTS = WORKSPACE / 'runs/custom_v1/unlabeled_adaptation'
candidates = [UNLABELED_RESULTS / name for name in [
    'unlabeled_cv_report.json', 'original_unsupervised_projection_progress.json',
    'original_plus_unlabeled_custom_projection_progress.json',
    'model/model.joblib', 'model/final_selection.json']]
candidates += [WORKSPACE / 'artifacts' / name for name in [
    'custom_experiment_summary.json', 'custom_model_comparison.csv', 'custom_model_comparison.png']]
if globals().get('RUN_PRIVILEGED_NOTES', False):
    candidates += [PRIVILEGED_REPORT, PRIVILEGED_RUNTIME,
                   PRIVILEGED_REPORT.parent / 'privileged_progress.json']
    optional_status = json.loads(PRIVILEGED_RUNTIME.read_text()) if PRIVILEGED_RUNTIME.is_file() else {}
    if optional_status.get('status') == 'completed':
        candidates += [PRIVILEGED_REPORT.parent / name for name in [
            'model/model_privileged.joblib', 'model/final_selection.json']]
from pathlib import PurePosixPath
PRIVATE_PARTS = {'by_source', 'translations', 'privileged_note_cache',
                 'private_original_notes', 'original_privileged_notes',
                 'data', 'features', 'clip_features', 'semantic_features', 'observations'}
PRIVATE_SUFFIXES = {'.npz', '.npy', '.mov', '.mp4', '.mp3', '.wav', '.m4a', '.flac',
                    '.aac', '.safetensors', '.pt', '.pth', '.zip'}
def safe_result_member(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and '..' not in path.parts and '\\\\' not in name
            and not (set(path.parts) & PRIVATE_PARTS) and path.suffix.lower() not in PRIVATE_SUFFIXES)
with zipfile.ZipFile(RESULTS_ZIP, 'a', zipfile.ZIP_DEFLATED) as archive:
    existing = set(archive.namelist())
    assert all(safe_result_member(name) for name in existing), 'Existing ZIP contains a private input/cache path; regenerate the driver export.'
    for path in candidates:
        if not path.is_file():
            continue
        name = path.relative_to(WORKSPACE).as_posix()
        assert safe_result_member(name), 'Private input/cache paths cannot be exported.'
        if name not in existing:
            archive.write(path, name)
            existing.add(name)
print('Final archive includes completed adaptation reports and heads; private notes are excluded.')
# Optional private download to your computer:
# from google.colab import files
# files.download(str(RESULTS_ZIP))
'''
    return [
        make_cell("markdown", "# Silent custom-video augmentation\n\n" + status + "\n\n"
            "**All 39 custom recordings are currently unlabeled by a verified human.** The supplied custom workbook repeats the original labels and does not match these new filenames. This notebook creates **unverified local model proposals**, then runs an explicit train-only weak-label sensitivity comparison. It cannot certify their correctness.\n\n"
            "A separate label-free experiment uses all custom recordings to train an unsupervised scaler/PCA representation while keeping classifier supervision entirely on original labels. Its matched projection baseline is reported separately.\n\n"
            "The iPhone and Samsung videos are simultaneous views of the same scenes, with unresolved pairing. They remain one conservative training session. Only the original **40 unique recordings**, on their exact historical held-out folds, supply evaluation targets. No independent custom-camera test or 87% accuracy guarantee is established.\n\n"
            f"The embedded source includes {test_count} regression tests. Audio is ignored throughout; the custom archive contains physically silent compressed derivatives. No hosted LLM API, speech recognition, captions or web interface is used.\n"),
        make_cell("markdown", "## 1. Restore the exact source snapshot\nThis embedded ZIP contains code, tests, documentation and the original numeric fold reference. It contains no recordings, extracted frames or model weights. Existing original caches in `datathon_ml_workspace` can be reused.\n"),
        make_cell("code", bootstrap),
        make_cell("markdown", "## 2. Install dependencies and cache public checkpoints\nSelect a GPU runtime. Download official CLIP and the pinned Qwen3-VL-4B-Instruct weights during setup only. Public downloads contain no private video. Subsequent perception executes in this runtime with offline loading and a Python socket guard.\n"),
        make_cell("code", setup),
        make_cell("code", weights),
        make_cell("markdown", f"## 3. Check the implementation\nRun the {test_count} regression tests in this exact source snapshot. Test counts describe implementation checks, not model accuracy.\n"),
        make_cell("code", "run_command([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])\n"),
        make_cell("markdown", "## 4. Reuse or bootstrap the organizer feature cache\nWhen the original cache is absent, locate `DATA_ROOT/open_labels.xlsx` or download the public organizer ZIP. The workbook path is explicit, so the copied custom workbook cannot attach to original videos. Extract CLIP at 2 fps, 224 pixels and four views; build the original semantic prototypes on CPU. These are feature-cache setup steps, without retraining the earlier model families.\n"),
        make_cell("code", original),
        make_cell("markdown", "## 5. Upload the silent custom archive privately\nUse the Colab Files pane to upload `custom_videos_silent.zip` to session storage. Its required SHA identifies the 39 prepared derivatives. Preserve original recordings separately. The driver rejects audio streams, changed bytes, unsafe paths and mismatched timestamps.\n"),
        make_cell("code", upload),
        make_cell("markdown", "## 6. Generate proposals and train the two declared policies\nThe local Qwen annotator requests fixed numeric codes and timestamps from silent frames. It sees neither supplied labels nor predictions from the original trained heads. Unknown proposals remain excluded. Proposed positive labels contain a timestamp and are marked for human review.\n\n"
            "The driver compares original-only training against original plus custom proposals at weight **0.25**, separately for the fixed semantic linear policy and the small point-supervised linear/RBF policy. Custom footage stays in training during every outer and inner split. Calibration and thresholds use only held-out original inner-fold labels. Accuracy, balanced accuracy, recall, specificity, F1/F2, ranking metrics and point-localization diagnostics are reported.\n\n"
            "This can take many minutes. Local annotation has a hard **2,700-second** budget; partial completion is marked explicitly. Aggregate progress prints about every 30 seconds. Annotation time is not a claim about deployment latency.\n"),
        make_cell("code", run),
        make_cell("markdown", "## 7. Train using custom footage without inventing labels\nBoth arms fit a scaler/PCA32 on up to 24 uniformly selected windows per training source. One uses original training sources only; the other also uses the 39 unlabeled custom views. Candidate classifiers and thresholds use original inner-fold labels exclusively. No held-out original features enter projection fitting, and no custom labels enter any loss. The four candidate heads share each fitted projection. This adds actual unlabeled-data training even when every annotation proposal is unresolved. It can reduce accuracy; read the measured comparison before choosing a model.\n"),
        make_cell("code", adaptation),
        make_cell("markdown", "## 8. Optional private train-owned notes\nThis disabled branch requires a notes ZIP derived locally from the original supplied notes, uploaded privately and extracted to `NOTES_DIR`. Notes remain unverified privileged training supervision. Owner checks prevent reading held-out-source notes during a fit; inference uses video features alone. Raw notes are excluded from result exports and never sent to a hosted AI service.\n"),
        make_cell("code", optional_notes),
        make_cell("markdown", "## 9. Read actual measurements and export\nBundled measurements appear only when the builder received a completed measured summary. Aggregate reports revalidate every original SHA, label, fold and point annotation. All comparisons remain development measurements on the previously inspected original corpus. Review proposed labels and timestamps before treating them as verified supervision.\n"),
        make_cell("code", aggregate),
        make_cell("code", results),
        make_cell("code", export),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cv", type=Path)
    parser.add_argument("--measured-summary", type=Path)
    parser.add_argument("--notebook", type=Path, default=ROOT / "datathon_custom.ipynb")
    parser.add_argument("--bundle-out", type=Path, default=ROOT / "artifacts/datathon_custom_notebook_source.zip")
    args = parser.parse_args(argv)
    baseline = args.baseline_cv or default_baseline(ROOT)
    bundle, measured = custom_bundle(ROOT, baseline, args.measured_summary)
    cells = notebook_cells(bundle, measured)
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<custom-notebook-cell>", "exec")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": "datathon_custom.ipynb", "provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    args.notebook.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    args.bundle_out.parent.mkdir(parents=True, exist_ok=True)
    args.bundle_out.write_bytes(bundle)
    print(json.dumps({"notebook": str(args.notebook), "cells": len(cells), "bundle": str(args.bundle_out),
        "bundle_bytes": len(bundle), "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "measured_summary_included": measured is not None}))


if __name__ == "__main__":
    main()

"""Build a self-contained Colab/Kaggle notebook from the reviewed Python files."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[1]


def make_cell(kind, source):
    cell = {"cell_type": kind, "id": hashlib.sha256((kind + source).encode()).hexdigest()[:12],
            "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def main():
    members = [ROOT / "experiment.py", ROOT / "mil_experiment.py", ROOT / "requirements-ml.txt", ROOT / "requirements-experiments.txt", ROOT / "requirements-vlm.txt", ROOT / "duplicate_aliases.json"]
    members += sorted((ROOT / "datathon_ml").glob("*.py"))
    members += sorted((ROOT / "tests").glob("test_*.py"))
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for path in members:
            archive.writestr(path.relative_to(ROOT).as_posix(), path.read_bytes())
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    bootstrap = f'''# Embedded project source: generated from the adjacent Python files.
import base64, io, os, sys, zipfile
from pathlib import Path
WORKSPACE = Path('/kaggle/working' if Path('/kaggle/working').exists() else '/content' if Path('/content').exists() else '.') / 'datathon_ml_workspace'
WORKSPACE.mkdir(parents=True, exist_ok=True)
SOURCE_BUNDLE = {payload!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_BUNDLE))) as archive:
    for item in archive.infolist():
        destination = (WORKSPACE / item.filename).resolve()
        if not destination.is_relative_to(WORKSPACE.resolve()):
            raise ValueError('Unsafe source archive path')
    archive.extractall(WORKSPACE)
os.chdir(WORKSPACE)
sys.path.insert(0, str(WORKSPACE))
print('Project source ready:', WORKSPACE)
'''
    setup = '''import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'openpyxl>=3.1,<4', 'scikit-learn>=1.4,<2', 'opencv-python-headless>=4.8,<5', 'joblib>=1.3,<2', 'tqdm>=4.66,<5'])
import torch
print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
assert torch.cuda.is_available(), 'Enable a free GPU runtime before full feature extraction.'
def run_command(arguments):
    process = subprocess.Popen(list(map(str, arguments)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, process.args)
def run_cli(*arguments):
    return run_command([sys.executable, '-u', 'experiment.py', *arguments])
'''
    data = '''import zipfile
from pathlib import Path
DATA_ROOT = None  # Set this to a folder containing open/open_labels.xlsx, or open_labels.xlsx itself's folder.
if DATA_ROOT is None:
    for candidate in [Path('/content/open'), Path('/kaggle/working/open'), WORKSPACE / 'open']:
        if (candidate / 'open_labels.xlsx').is_file():
            DATA_ROOT = candidate
            break
if DATA_ROOT is None:
    archives = [Path('/content/datathon_data.zip'), Path('/kaggle/working/datathon_data.zip')]
    archives += sorted(Path('/kaggle/input').rglob('datathon_data.zip')) if Path('/kaggle/input').exists() else []
    for path in archives:
        if path.is_file():
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    target = (WORKSPACE / item.filename).resolve()
                    if not target.is_relative_to(WORKSPACE.resolve()):
                        raise ValueError('Unsafe dataset archive path')
                archive.extractall(WORKSPACE)
            DATA_ROOT = WORKSPACE / 'open'
            break
if DATA_ROOT is None and Path('/kaggle/input').exists():
    matches = list(Path('/kaggle/input').rglob('open_labels.xlsx'))
    if len(matches) == 1:
        DATA_ROOT = matches[0].parent
assert DATA_ROOT is not None, 'Upload artifacts/datathon_data.zip using the Colab Files pane, or attach a PRIVATE Kaggle input.'
BACKBONE = 'clip_vit_b32_openai'  # Best measured development baseline; 57.5%, not case-ready.
RUN_MIL = False  # Set True to reproduce the optional MIL comparison; it did not improve this corpus.
SAMPLE_FPS, BATCH_SIZE = ((.25, 1) if BACKBONE == 'qwen3_vl_embedding_2b' else (1., 8) if BACKBONE.startswith('x3d_') else (2., 32))
QWEN_CLIP_SECONDS, QWEN_FRAMES = 8., 4  # Dense comparison: 4., 8, with SAMPLE_FPS=.5.
RUN_TAG = BACKBONE + '_fps' + str(SAMPLE_FPS)
if BACKBONE == 'qwen3_vl_embedding_2b':
    RUN_TAG += '_' + str(QWEN_CLIP_SECONDS) + 's_' + str(QWEN_FRAMES) + 'frames'
RUN_DIR = WORKSPACE / 'runs' / RUN_TAG
FEATURE_DIR = RUN_DIR / 'features'
COMMON = ['--data-root', str(DATA_ROOT), '--output-dir', str(RUN_DIR), '--verified-aliases', 'duplicate_aliases.json']
FEATURE_ARGS = ['--backbone', BACKBONE, '--sample-fps', str(SAMPLE_FPS), '--batch-size', str(BATCH_SIZE)]
if BACKBONE == 'qwen3_vl_embedding_2b':
    FEATURE_ARGS += ['--clip-seconds', str(QWEN_CLIP_SECONDS), '--frames-per-window', str(QWEN_FRAMES)]
print('Data:', DATA_ROOT, 'Run:', RUN_DIR)
'''
    cells = [
        make_cell("markdown", "# Datathon: local vision model experiments\n\nFrozen visual/action embeddings, timestamp-supervised temporal classifiers, nested grouped validation, and an exported inference model. No hosted inference API. Optional Qwen embeddings execute directly in this runtime, without text generation.\n\nThe best measured development baseline is CLIP at **57.5% on 40 unique videos**. The 87% target has not been achieved; semantic event annotations and independent validation remain necessary. This notebook defaults to that baseline.\n\nChoose BACKBONE below: dinov2_vits14, dinov2_vitb14_reg, clip_vit_b32_openai, x3d_s, or qwen3_vl_embedding_2b. Upload `artifacts/datathon_data.zip` privately to session storage. Exact duplicate aliases are explicitly declared in `duplicate_aliases.json`; positive timestamps supervise training only. Unknown event types and legal interpretations require additional annotation.\n"),
        make_cell("code", bootstrap),
        make_cell("code", setup),
        make_cell("code", data),
        make_cell("markdown", "## Audit and regression tests\nThe manifest merges exact duplicates and their event anchors. Tests exercise parsing, grouping, sampling and leakage safeguards.\n"),
        make_cell("code", "run_command([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])\nrun_cli('audit', *COMMON)\n"),
        make_cell("markdown", "## Extract visual features\nEach video is scanned uniformly. Full-frame letterboxing plus three overlapping crops preserve small-object detail. Features are cached by video content and encoder configuration.\n"),
        make_cell("code", "if BACKBONE == 'clip_vit_b32_openai':\n    run_command([sys.executable, '-m', 'pip', 'install', '-q', 'open_clip_torch==3.2.0'])\nelif BACKBONE.startswith('x3d_'):\n    run_command([sys.executable, '-m', 'pip', 'install', '-q', 'fvcore', 'iopath'])\nelif BACKBONE == 'qwen3_vl_embedding_2b':\n    run_command([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-vlm.txt'])\nrun_cli('extract', *COMMON, *FEATURE_ARGS)\n"),
        make_cell("markdown", "## Nested grouped cross-validation\nOuter folds estimate development performance. Inner folds choose model regularization, aggregation and threshold. No test event anchors enter prediction. This is not an independent new-angle test.\n"),
        make_cell("code", "run_cli('cv', *COMMON, *FEATURE_ARGS)\n"),
        make_cell("code", "import json\nfor path in sorted((RUN_DIR / 'evaluation').glob('*.json')):\n    print(path.name)\n    report = json.loads(path.read_text())\n    if isinstance(report, dict) and 'metrics' in report:\n        print(json.dumps(report['metrics'], indent=2))\n"),
        make_cell("markdown", "## Fit the development model\nThis model uses all training videos after grouped model selection. Its training-set predictions are not a validation score. Keep an independent new-take/new-angle test locked.\n"),
        make_cell("code", "run_cli('fit', *COMMON, *FEATURE_ARGS)\n"),
        make_cell("markdown", "## Point-guided multiple-instance comparison\nA second training objective uses complete normal timelines, video bag labels, and supplied positive anchors. It never marks unannotated positive spans as normal. Its two small heads and thresholds are selected inside grouped inner folds. This comparison remains development work.\n"),
        make_cell("code", "if RUN_MIL:\n    run_command([sys.executable, '-u', 'mil_experiment.py', 'pipeline', '--manifest', RUN_DIR / 'manifest.json', '--feature-dir', FEATURE_DIR, '--output-dir', RUN_DIR / 'mil'])\n"),
        make_cell("markdown", "## Export results\nThe archive below contains source, reports and the trained head; it excludes the input videos. For offline end-to-end inference also retain the encoder repository and checkpoint identified in feature provenance.\n"),
        make_cell("code", "import zipfile\nEXPORT = WORKSPACE.parent / 'datathon_model_results.zip'\nwith zipfile.ZipFile(EXPORT, 'w', compression=zipfile.ZIP_DEFLATED) as archive:\n    for path in RUN_DIR.rglob('*'):\n        if path.is_file() and 'features' not in path.relative_to(RUN_DIR).parts:\n            archive.write(path, 'run/' + path.relative_to(RUN_DIR).as_posix())\n    for path in [*Path('datathon_ml').glob('*.py'), Path('experiment.py'), Path('mil_experiment.py'), *Path('.').glob('requirements*.txt'), Path('duplicate_aliases.json')]:\n        archive.write(path, 'source/' + path.as_posix())\nprint('Export:', EXPORT, 'bytes:', EXPORT.stat().st_size)\n# In Colab, uncomment to download:\n# from google.colab import files; files.download(str(EXPORT))\n"),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "datathon.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    bundle = ROOT / "artifacts" / "datathon_source.zip"
    bundle.parent.mkdir(exist_ok=True)
    bundle.write_bytes(buffer.getvalue())
    print(target, "cells=", len(cells), "source bundle bytes=", len(buffer.getvalue()))


if __name__ == "__main__":
    main()

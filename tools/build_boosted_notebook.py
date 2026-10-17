"""Build a clean self-contained CatBoost/PCA and pretrained-action notebook."""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_custom_notebook import notebook_cells as custom_cells
from tools.build_temporal_notebook import default_baseline, make_cell, source_bundle


def boosted_bundle(root, baseline, measured_summary=None):
    with ZipFile(io.BytesIO(source_bundle(root, baseline))) as archive:
        members = {name: archive.read(name) for name in archive.namelist() if name != "SOURCE_MANIFEST.json"}
    for path, name in ((root / "artifacts/custom_experiment_summary.json", "reference/previous_custom_summary.json"),
                       (root / "reference/public_video_data_research.json", "reference/public_video_data_research.json")):
        if path.is_file():
            members[name] = path.read_bytes()
    measured = None
    if measured_summary is not None:
        payload = Path(measured_summary).read_bytes()
        measured = json.loads(payload)
        if not measured.get("all_completed_new_predictions_SHA_labels_folds_and_metrics_verified"):
            raise ValueError("Measured summary must certify original-source prediction verification")
        if not any(r["model"] == "boosted_primary" for r in measured.get("model_comparisons", [])):
            raise ValueError("Measured summary needs a completed CatBoost primary policy")
        members["reference/measured_boosted_summary.json"] = payload
    members["SOURCE_MANIFEST.json"] = json.dumps({"format_version": 1, "contains_recordings": False,
        "members_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in sorted(members.items())}},
        indent=2, sort_keys=True).encode()
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(members.items()):
            item = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            item.compress_type = ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            archive.writestr(item, value)
    return buffer.getvalue(), measured


def notebook_cells(bundle, measured=None):
    # Reuse the audited source restore and organizer/cache bootstrap exactly.
    prior = custom_cells(bundle, None)
    bootstrap = next("".join(c["source"]) for c in prior if c["cell_type"] == "code"
                     and "SOURCE_ZIP = base64.b64decode(" in "".join(c["source"]))
    original = next("".join(c["source"]) for c in prior if c["cell_type"] == "code"
                    and "ORIGINAL_MANIFEST = ORIGINAL /" in "".join(c["source"]))
    with ZipFile(io.BytesIO(bundle)) as archive:
        tests = sum(isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            for name in archive.namelist() if name.startswith("tests/test_") and name.endswith(".py")
            for node in ast.walk(ast.parse(archive.read(name).decode())))
    setup = '''import shutil, subprocess, sys

def run_command(arguments):
    subprocess.run(list(map(str, arguments)), cwd=WORKSPACE, check=True)

run_command([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-ml.txt',
    '-r', 'requirements-experiments.txt', 'gdown>=5,<6'])
if not shutil.which('ffprobe'):
    assert Path('/content').is_dir(), 'Install ffmpeg/ffprobe in this runtime first.'
    run_command(['apt-get', '-qq', 'update'])
    run_command(['apt-get', '-qq', 'install', '-y', 'ffmpeg'])
import torch, catboost
assert catboost.__version__ == '1.2.10'
print('Torch:', torch.__version__, 'CatBoost:', catboost.__version__, 'CUDA:', torch.cuda.is_available())
RUN_MOTION = True
'''
    weights = '''# Public checkpoints/source are cached before any private perception.
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ.pop(key, None)
from datathon_ml.clip_features import ClipFeatureConfig, _resolve_clip_weights
CLIP_CHECKPOINT = _resolve_clip_weights(ClipFeatureConfig(allow_download=True), torch)
print('CLIP checkpoint cached:', CLIP_CHECKPOINT.is_file())
if RUN_MOTION:
    from datathon_ml.motion import MotionFeatureConfig, _checkpoint, _load_model
    motion_config = MotionFeatureConfig(backbone='x3d_s', representation='logits', device='cpu', allow_download=True)
    X3D_CHECKPOINT = _checkpoint(motion_config, torch)
    cached_motion_model = _load_model(motion_config, X3D_CHECKPOINT, torch.device('cpu'), torch)
    del cached_motion_model
    print('Pinned X3D-S source/checkpoint cached:', X3D_CHECKPOINT.is_file())
'''
    guarded = '''# Every experimental command runs in a fresh process under a socket guard.
import time
RUN = WORKSPACE / 'runs/boosted_v1'
RUN.mkdir(parents=True, exist_ok=True)
GUARD_WRAPPER = r"""
import json, runpy, sys, time
from pathlib import Path
module, output, arguments = sys.argv[1:]
from tools.benchmark_semantic_vlm import configure_offline_environment, blocked_network
configure_offline_environment()
started = time.perf_counter()
result = {'status': 'started', 'completed': False}
sys.argv = [module + '.py'] + json.loads(arguments)
with blocked_network() as network:
    try:
        runpy.run_module(module, run_name='__main__')
        result.update(status='completed', completed=True)
    except SystemExit as error:
        if error.code in (None, 0):
            result.update(status='completed', completed=True)
        else:
            result.update(status='failed', error_type='SystemExit')
        raise
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        result.update(network_attempts=network['attempts'], elapsed_seconds=time.perf_counter()-started)
        Path(output).write_text(json.dumps(result, indent=2))
        print('GUARDED_RUNTIME', json.dumps(result), flush=True)
"""

def run_guarded(module, arguments, runtime_name, budget):
    runtime = RUN / runtime_name
    log = runtime.with_suffix('.log')
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONHASHSEED='42', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    started = time.perf_counter()
    with log.open('w') as output:
        process = subprocess.Popen([sys.executable, '-u', '-c', GUARD_WRAPPER, module,
            str(runtime), json.dumps(list(map(str, arguments)))], cwd=WORKSPACE,
            stdout=output, stderr=subprocess.STDOUT, env=env)
        try:
            code = process.wait(timeout=budget)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            process.kill()
            process.wait()
            report = {'status': 'timed_out' if isinstance(error, subprocess.TimeoutExpired) else 'cancelled',
                      'completed': False, 'network_attempts': None, 'budget_seconds': budget,
                      'elapsed_seconds': time.perf_counter()-started}
            runtime.write_text(json.dumps(report, indent=2))
            print(report)
            raise
    report = json.loads(runtime.read_text()) if runtime.is_file() else {'status': 'failed_without_runtime_report'}
    report.update(returncode=code, budget_seconds=budget)
    runtime.write_text(json.dumps(report, indent=2))
    print('Runtime:', report, 'Log:', log)
    if code:
        raise RuntimeError('Guarded command failed; inspect its local log.')
    assert report.get('network_attempts') == 0, 'Unexpected network attempt'
    return report
'''
    train = '''run_guarded('boosted_experiment', ['compare',
    '--original-manifest', ORIGINAL_MANIFEST, '--clip-features', ORIGINAL / 'features',
    '--semantic-features', SEMANTIC_ORIGINAL, '--folds-from', 'reference/clip_baseline_cv.json',
    '--output-dir', RUN, '--threads', '2', '--fit-final'], 'runtime_boosted.json', 1800)
'''
    motion = '''if RUN_MOTION:
    assert torch.cuda.is_available(), 'Select a GPU runtime for fresh X3D extraction.'
    run_guarded('motion_action_experiment', ['run', '--original-manifest', ORIGINAL_MANIFEST,
        '--features', RUN / 'x3d_logits_features', '--output-dir', RUN / 'motion_action',
        '--folds-from', 'reference/clip_baseline_cv.json', '--device', 'cuda', '--fit-final'],
        'runtime_motion_action.json', 1800)
else:
    (RUN / 'runtime_motion_action.json').write_text(json.dumps({'status': 'not_run', 'completed': False,
                                                            'network_attempts': None}))
'''
    inference = '''# Features are extracted from video alone; no labels or points enter prediction.
key = sorted(r['sha256'] for r in original_records)[0]
run_guarded('boosted_experiment', ['predict', '--model', RUN / 'model/model_boosted.joblib',
    '--clip-feature', ORIGINAL / 'features' / (key + '.npz'),
    '--semantic-feature', SEMANTIC_ORIGINAL / (key + '.npz')], 'runtime_inference.json', 120)
if RUN_MOTION and (RUN / 'motion_action/model/model_motion_action.joblib').is_file():
    run_guarded('motion_action_experiment', ['predict', '--model', RUN / 'motion_action/model/model_motion_action.joblib',
        '--feature-path', RUN / 'x3d_logits_features' / (key + '.npz')], 'runtime_motion_inference.json', 120)
'''
    aggregate = '''# PCA diagnostics use training sources only; variance is not model accuracy.
run_guarded('tools.audit_preprocessing', ['--original-manifest', ORIGINAL_MANIFEST,
    '--features', ORIGINAL / 'features', '--folds-from', 'reference/clip_baseline_cv.json',
    '--output', RUN / 'preprocessing_audit.json'], 'runtime_preprocessing.json', 600)
run_command([sys.executable, 'tools/summarize_boosted.py', '--results-root', RUN,
    '--baseline', 'reference/clip_baseline_cv.json',
    '--previous-summary', 'reference/previous_custom_summary.json', '--output-dir', 'artifacts'])
summary = json.loads((WORKSPACE / 'artifacts/boosted_experiment_summary.json').read_text())
print('MODEL / ACCURACY / BALANCED ACCURACY / RECALL / SPECIFICITY')
for row in summary['model_comparisons']:
    m = row['metrics']
    print(row['model'], *(f"{m[k]:.1%}" for k in ('accuracy', 'balanced_accuracy', 'recall', 'specificity')))
print('87% target reached on development:', summary['target_achieved_on_development'])
print('Independent 87% test accuracy established:', summary['independent_87_percent_accuracy_established'])
from IPython.display import display, Image
display(Image(filename=str(WORKSPACE / 'artifacts/boosted_model_comparison.png')))
measured_reference = WORKSPACE / 'reference/measured_boosted_summary.json'
if measured_reference.is_file():
    print('Bundled immutable measured reference rows:', len(json.loads(measured_reference.read_text())['model_comparisons']))
'''
    export = '''# Export exact numeric reports, trained artifacts and checksummed source only.
RESULTS_ZIP = BASE / 'datathon_boosted_results.zip'
candidates = list(RUN.glob('*.json')) + list((RUN / 'motion_action').glob('*.json'))
candidates += list((RUN / 'model').glob('*.joblib')) + list((RUN / 'model').glob('*.json'))
candidates += list((RUN / 'motion_action/model').glob('*.joblib')) + list((RUN / 'motion_action/model').glob('*.json'))
candidates += [WORKSPACE / 'artifacts' / name for name in ('boosted_experiment_summary.json',
               'boosted_model_comparison.csv', 'boosted_model_comparison.png')]
candidates += [WORKSPACE / 'datathon_boosted.ipynb']
members = {}
with zipfile.ZipFile(io.BytesIO(SOURCE_ZIP)) as source:
    for name in source.namelist():
        members['source/' + name] = source.read(name)
for path in candidates:
    if path.is_file():
        name = path.relative_to(WORKSPACE).as_posix()
        assert not (set(Path(name).parts) & {'features', 'x3d_logits_features', 'private_training_notes', 'data', 'open'})
        assert path.suffix.lower() in {'.json', '.csv', '.png', '.joblib', '.ipynb'}
        if path.suffix.lower() == '.ipynb':
            clean_notebook = json.loads(path.read_text())
            assert all(not cell.get('outputs') and cell.get('execution_count') is None
                       for cell in clean_notebook['cells'] if cell['cell_type'] == 'code')
        members[name] = path.read_bytes()
members['EXPORT_MANIFEST.json'] = json.dumps({'format_version': 1, 'contains_private_media_or_feature_caches': False,
    'members_sha256': {name: hashlib.sha256(value).hexdigest() for name, value in sorted(members.items())}}, indent=2).encode()
with zipfile.ZipFile(RESULTS_ZIP, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name, value in sorted(members.items()):
        archive.writestr(name, value)
print('Export:', RESULTS_ZIP, 'Bytes:', RESULTS_ZIP.stat().st_size)
print('SHA256:', hashlib.sha256(RESULTS_ZIP.read_bytes()).hexdigest())
# Download privately when ready:
# from google.colab import files
# files.download(str(RESULTS_ZIP))
'''
    status = "No measured CatBoost results are bundled yet; 87% is a target, not a promise."
    if measured is not None:
        primary = measured["primary_inner_selected_policy"]["metrics"]
        status = f"Bundled completed primary-policy development accuracy: **{primary['accuracy']:.1%}**, balanced accuracy **{primary['balanced_accuracy']:.1%}**. Independent test accuracy is unestablished."
    return [
        make_cell("markdown", "# Pretrained visual features, PCA and CatBoost\n\n" + status + "\n\n"
            "This notebook compares 12 fixed candidates on the original 40 recordings with the exact historical outer folds. It includes semantic summaries, all-view CLIP, train-only PCA16, and a full-frame view ablation. A separate X3D-S transfer test retains its pretrained 400-action classifier.\n\n"
            "All private perception runs locally in this runtime. Audio, hosted LLM APIs, captions, private notes and unverified custom labels are unused. Related views must share scene groups. Existing custom footage remains unlabeled and is excluded from supervised evaluation.\n\n"
            f"The embedded snapshot includes {tests} implementation regression tests; test counts are not accuracy measurements."),
        make_cell("markdown", "## Restore source and install dependencies\nThe embedded source archive contains no recordings, frames, weights or feature caches. Public model downloads occur only during setup."),
        make_cell("code", bootstrap), make_cell("code", setup), make_cell("code", weights),
        make_cell("markdown", "## Reuse or build original feature caches\nThe organizer labels path is explicit. Existing CLIP and semantic caches are reused. Fresh extraction requires a GPU and verifies the original 40 source hashes and labels."),
        make_cell("code", original), make_cell("code", guarded),
        make_cell("markdown", "## Validate implementation and run the fixed comparison\nCatBoost uses 160 iterations, learning rate 0.03, depth 2/3 and L2 10/30. Inner grouped OOF selects representation, tree settings and threshold; every scaler/PCA fit uses training sources only. No outer early stopping or calibration occurs. The full comparison has an 1,800-second hard budget."),
        make_cell("code", "run_guarded('unittest', ['discover', '-s', 'tests', '-v'], 'runtime_tests.json', 900)\n"),
        make_cell("code", train),
        make_cell("markdown", "## Separate pretrained action transfer test\nX3D-S processes ordered clips through frozen pretrained weights, retaining all 400 action logits. A recording head pools mean/std/max and selects among the same three tree settings. This is a separate preregistered development comparison with its own 1,800-second budget."),
        make_cell("code", motion),
        make_cell("markdown", "## Offline prediction, diagnostics and measured comparison\nRecording heads provide binary recording decisions. Event timestamp recall is unavailable, not zero. The chart uses a full 0–100% scale. All completed OOF outputs are checked against the frozen SHA, label and fold reference, with nested training ownership and model source fingerprints verified."),
        make_cell("code", inference), make_cell("code", aggregate),
        make_cell("markdown", "## Export reproducible artifacts\nThe export includes trained models, numeric reports and exact source bytes. Private footage, audio, extracted images, feature caches and private notes are excluded. Public data candidates and suitability/license findings are documented in `reference/public_video_data_research.json`; no external training footage has been silently added."),
        make_cell("code", export)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cv", type=Path)
    parser.add_argument("--measured-summary", type=Path)
    parser.add_argument("--notebook", type=Path, default=ROOT / "datathon_boosted.ipynb")
    parser.add_argument("--bundle-out", type=Path, default=ROOT / "artifacts/datathon_boosted_notebook_source.zip")
    args = parser.parse_args(argv)
    bundle, measured = boosted_bundle(ROOT, args.baseline_cv or default_baseline(ROOT), args.measured_summary)
    cells = notebook_cells(bundle, measured)
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<boosted-notebook-cell>", "exec")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": args.notebook.name, "provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    args.notebook.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    args.bundle_out.parent.mkdir(parents=True, exist_ok=True)
    args.bundle_out.write_bytes(bundle)
    print(json.dumps({"notebook": str(args.notebook), "cells": len(cells),
        "bundle_bytes": len(bundle), "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "measured_summary_included": measured is not None}))


if __name__ == "__main__":
    main()

"""Create a clean temporal notebook with exact source and no private caches."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_custom_notebook import notebook_cells as custom_cells
from tools.build_temporal_notebook import default_baseline, make_cell, source_bundle


def scene_bundle():
    with ZipFile(io.BytesIO(source_bundle(ROOT, default_baseline(ROOT)))) as archive:
        members = {name: archive.read(name) for name in archive.namelist()
                   if name != 'SOURCE_MANIFEST.json'}
    for filename in ('boosted_experiment_summary.json', 'custom_label_verification.json',
                     'custom_pairing_audit.json'):
        path = ROOT/'artifacts'/filename
        if not path.is_file():
            path = ROOT/'reference'/filename
        if path.is_file():
            members['reference/'+filename] = path.read_bytes()
    members['SOURCE_MANIFEST.json'] = json.dumps({'contains_recordings': False,
        'members_sha256': {name: hashlib.sha256(value).hexdigest()
                           for name,value in sorted(members.items())}}, indent=2).encode()
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w', ZIP_DEFLATED, compresslevel=9) as archive:
        for name,value in sorted(members.items()):
            archive.writestr(name,value)
    return buffer.getvalue()


def notebook_cells(bundle, measured=None):
    bootstrap = next(''.join(cell['source']) for cell in custom_cells(bundle, None)
        if cell['cell_type'] == 'code' and 'SOURCE_ZIP = base64.b64decode(' in ''.join(cell['source']))
    setup = '''# CPU is sufficient for these small heads over existing visual features.
import subprocess, torch
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    '-r', 'requirements-ml.txt'], cwd=WORKSPACE, check=True)
print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
'''
    restore = '''# Upload artifacts/scene_temporal_cache.zip using Colab's Files pane.
# It contains the original 40 numeric semantic caches and supplied targets.
# No video, frame, audio, private description or encoder weight is embedded here.
CACHE_ZIP = BASE / 'scene_temporal_cache.zip'
assert CACHE_ZIP.is_file(), 'Upload the private numeric cache archive first.'
with zipfile.ZipFile(CACHE_ZIP) as archive:
    inventory = json.loads(archive.read('CACHE_MANIFEST.json'))
    assert inventory['contains_video_or_frames'] is False
    for name, digest in inventory['members_sha256'].items():
        assert (WORKSPACE/name).resolve().is_relative_to(WORKSPACE.resolve())
        assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    archive.extractall(WORKSPACE)
manifest = WORKSPACE / 'cache/original_manifest.json'
reference = WORKSPACE / 'reference/clip_baseline_cv.json'
records = json.loads(manifest.read_text())
expected = {row['sha256']: row['label'] for row in json.loads(reference.read_text())['predictions']}
assert len(records) == len(expected) == 40
assert {row['sha256']: row['label'] for row in records} == expected
print('Original sources:', len(records), 'positive:', sum(row['label'] for row in records))
'''
    tests = '''# Focused regression checks validate implementation, not cashier accuracy.
from tools.run_scene_temporal_colab import run_guarded
RUN = WORKSPACE / 'runs/scene_temporal_v1'
run_guarded(WORKSPACE, RUN, 'focused_tests', 'unittest', ['discover',
    '-s', 'tests', '-p', 'test_scene_temporal.py', '-v'], 300)
'''
    train = '''# Two fixed TCN candidates; architectures/thresholds selected inside training folds.
# No outer-test early stopping, pseudo targets, camera-ID or duration predictors.
subprocess.run([sys.executable, '-u', 'tools/run_scene_temporal_colab.py',
    '--workspace', WORKSPACE, '--budget', '1800'], cwd=WORKSPACE, check=True)
'''
    inspect = '''report = json.loads((RUN/'evaluation/scene_temporal_cv_report.json').read_text())
print('Measured original-recording development results:')
print(json.dumps(report['metrics'], indent=2))
print('Grouping:', report['grouping'])
print('87% development target reached:', report['metrics']['accuracy'] >= .87)
print('This is not an independent new-scene test.')
'''
    export = '''# Export the fitted small head, numeric reports, clean notebook and exact source.
import io
members = {}
for path in RUN.rglob('*'):
    if path.is_file() and path.suffix in {'.json', '.joblib', '.log'}:
        members[path.relative_to(WORKSPACE).as_posix()] = path.read_bytes()
with zipfile.ZipFile(io.BytesIO(SOURCE_ZIP)) as source:
    for name in source.namelist():
        members['source/'+name] = source.read(name)
notebook_path = WORKSPACE/'datathon_scene_temporal.ipynb'
if notebook_path.is_file():
    clean = json.loads(notebook_path.read_text())
    assert all(not c.get('outputs') and c.get('execution_count') is None
               for c in clean['cells'] if c['cell_type']=='code')
    members[notebook_path.name] = notebook_path.read_bytes()
members['EXPORT_MANIFEST.json'] = json.dumps({'contains_media_or_feature_caches': False,
    'members_sha256': {name: hashlib.sha256(value).hexdigest() for name,value in members.items()}},
    indent=2).encode()
RESULTS_ZIP = BASE/'datathon_scene_temporal_results.zip'
with zipfile.ZipFile(RESULTS_ZIP, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name,value in sorted(members.items()):
        archive.writestr(name,value)
print('Export:', RESULTS_ZIP, 'Bytes:', RESULTS_ZIP.stat().st_size)
print('SHA256:', hashlib.sha256(RESULTS_ZIP.read_bytes()).hexdigest())
if Path('/content').is_dir():
    from google.colab import files
    files.download(str(RESULTS_ZIP))
'''
    intro = '''# Point-guided temporal learning over local pretrained features

The head uses supplied binary labels, positive timestamps and normal recordings.
It compares two small temporal convolution networks on the exact original five
outer folds. Preprocessing, architecture and decision threshold use training
owners only. Audio and hosted inference APIs are unused.

The custom filming batch has no verified labels yet. Its views remain one
conservative session group. Camera matching proposals do not establish labels.
The original source groups are not verified independent scenes; report the run
as recording-group development evaluation. The 87% target is aspirational.
'''
    future = '''## Reviewed custom scenes

Complete `open/customVideos_prepared/custom_labels_review.csv` locally and run
`python tools/verify_custom_labels.py`. Each reviewed positive requires simple
`violation` and timestamp(s); normal videos use `no_violation`. Full-video review
and explicit human verification keep guesses out of ground truth. All views of
the custom filming batch stay in its conservative union group.

After compatible feature extraction, provide the reviewed combined manifest to
`scene_temporal_experiment.py cv --originals-from reference/clip_baseline_cv.json`
without `--folds-from` to generate grouped folds. Use verified scene mappings via
`--groups-json` if available. The CLI rejects unverified custom labels/points or
groups and rejects related source lineage crossing folds. Keep an entirely new
filming session aside for acceptance; these original recordings are development
data. The notebook never substitutes predictions for that missing review.
'''
    cells = [make_cell('markdown', intro), make_cell('code', bootstrap),
        make_cell('code', setup), make_cell('code', restore), make_cell('code', tests),
        make_cell('code', train), make_cell('code', inspect), make_cell('markdown', future)]
    if measured is not None:
        compact = {key: measured[key] for key in ('evaluation', 'grouping',
            'independent_new_scene_evaluation_completed', 'model_comparisons',
            'point_localization_metrics', 'custom_supervised_training_sources',
            'custom_ground_truth_blocker', 'runtime', 'full_regression_tests') if key in measured}
        cells.append(make_cell('markdown', '## Saved completed measurements\n\n```json\n'+
            json.dumps(compact, indent=2)+'\n```\n'))
    cells.append(make_cell('code', export))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'datathon_scene_temporal.ipynb')
    parser.add_argument('--measured-summary', type=Path)
    args = parser.parse_args()
    bundle = scene_bundle()
    measured = json.loads(args.measured_summary.read_text()) if args.measured_summary else None
    cells = notebook_cells(bundle, measured)
    for cell in cells:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<notebook>', 'exec')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
        'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 5}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1)+'\n', encoding='utf-8')
    source_path = ROOT/'artifacts/scene_temporal_source.zip'
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(bundle)
    print(json.dumps({'notebook': str(args.output), 'cells': len(cells),
        'source_bytes': len(bundle), 'source_sha256': hashlib.sha256(bundle).hexdigest()}))


if __name__ == '__main__':
    main()

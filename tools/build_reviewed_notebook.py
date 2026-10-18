"""Build the clean, checksummed reviewed-data notebook without private inputs."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_custom_notebook import notebook_cells as custom_cells
from tools.build_temporal_notebook import default_baseline, make_cell, source_bundle


AUDITS = ("reviewed_workbook_audit.json", "reviewed_scene_group_audit.json",
          "reviewed_protocol_audit.json")


def reviewed_bundle(root, baseline, measured_summary=None):
    """Code plus safe numeric audits; native rebuilds resolve reference copies."""
    root = Path(root)
    with ZipFile(io.BytesIO(source_bundle(root, baseline))) as archive:
        members = {name: archive.read(name) for name in archive.namelist()
                   if name != "SOURCE_MANIFEST.json"}
    for filename in AUDITS:
        path = next((candidate for candidate in (root / "artifacts" / filename,
                    root / "reference" / filename) if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError("A safe reviewed audit is missing: " + filename)
        members["reference/" + filename] = path.read_bytes()
    measured = None
    if measured_summary is not None:
        payload = Path(measured_summary).read_bytes()
        measured = json.loads(payload)
        if (not isinstance(measured, dict)
                or measured.get("split_labels_sources_groups_model_threshold_and_metrics_verified") is not True
                or not isinstance(measured.get("test_metrics", {}).get("accuracy"), (int, float))):
            raise ValueError("Measured summary must certify completed split/model/metric verification")
        members["reference/measured_reviewed_summary.json"] = payload
    members["SOURCE_MANIFEST.json"] = json.dumps({"format_version": 1,
        "contains_recordings": False, "contains_feature_caches": False,
        "contains_workbooks_or_private_descriptions": False,
        "members_sha256": {name: hashlib.sha256(value).hexdigest()
                           for name, value in sorted(members.items())}},
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
    bootstrap = next("".join(cell["source"]) for cell in custom_cells(bundle, None)
        if cell["cell_type"] == "code" and "SOURCE_ZIP = base64.b64decode(" in "".join(cell["source"]))
    bootstrap = bootstrap.replace("BASE / 'datathon_ml_workspace'", "BASE / 'datathon_reviewed_workspace'")
    setup = '''# Public packages only; this notebook performs no video/LLM API calls.
import subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    '-r', 'requirements-ml.txt', 'catboost==1.2.10'], cwd=WORKSPACE, check=True)
import torch, catboost
assert catboost.__version__ == '1.2.10'
print('Torch:', torch.__version__, 'CatBoost:', catboost.__version__, 'CPU heads')
'''
    restore = '''# Upload the private reviewed_numeric_dataset.zip in Colab's Files pane.
# Only train/validation NPZs are materialized before the winner is frozen.
DATASET_ZIP = next((candidate for candidate in (
    BASE/'reviewed_numeric_dataset.zip', WORKSPACE/'reviewed_numeric_dataset.zip')
    if candidate.is_file()), None)
assert DATASET_ZIP is not None, 'Upload the separate private numeric dataset ZIP first.'
with zipfile.ZipFile(DATASET_ZIP) as archive:
    names = archive.namelist()
    assert len(names) == len(set(names)), 'Duplicate dataset archive members'
    inventory = json.loads(archive.read('DATASET_MANIFEST.json'))
    checksums = inventory['members_sha256']
    assert set(names) == set(checksums) | {'DATASET_MANIFEST.json'}
    for name in checksums:
        assert (WORKSPACE/name).resolve().is_relative_to(WORKSPACE.resolve())
        assert name in {'dataset/manifest.json', 'dataset/split.json'} or (
            len(Path(name).parts) == 3 and Path(name).parts[:2] in
            {('features', 'raw_clip'), ('features', 'semantic18')} and Path(name).suffix == '.npz')
    def read_verified_member(name):
        payload = archive.read(name)
        assert hashlib.sha256(payload).hexdigest() == checksums[name], 'Dataset member checksum mismatch'
        return payload
    records = json.loads(read_verified_member('dataset/manifest.json'))
    split = json.loads(read_verified_member('dataset/split.json'))
    all_keys = {row['sha256'] for row in records}
    assert len(records) == len(all_keys) == 58
    assert set(split['train']) | set(split['validation']) | set(split['test']) == all_keys
    assert [len(split[name]) for name in ('train', 'validation', 'test')] == [34, 12, 12]
    assert split['protocol']['type'] == 'camera_exclusive_source_overlap_development'
    assert split['protocol']['opposite_camera_excluded_globally'] is True
    expected_members = {'dataset/manifest.json', 'dataset/split.json'} | {
        f'features/{representation}/{key}.npz' for representation in ('raw_clip', 'semantic18') for key in all_keys}
    assert set(checksums) == expected_members
    for name in ('dataset/manifest.json', 'dataset/split.json'):
        target = WORKSPACE/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(read_verified_member(name))
    for representation in ('raw_clip', 'semantic18'):
        for key in sorted(set(split['train']) | set(split['validation'])):
            name = f'features/{representation}/{key}.npz'
            target = WORKSPACE/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(read_verified_member(name))
    assert all(not (WORKSPACE/f'features/{representation}/{key}.npz').exists()
        for representation in ('raw_clip', 'semantic18') for key in split['test']), (
        'Test caches already exist: use a fresh workspace for this experiment.')
RUN = WORKSPACE/'runs/reviewed_v1'
RUN.mkdir(parents=True, exist_ok=True)
(RUN/'dataset_archive_receipt.json').write_text(json.dumps({
    'dataset_zip_sha256': hashlib.sha256(DATASET_ZIP.read_bytes()).hexdigest(),
    'members_sha256_count': len(checksums), 'records': len(records),
    'train': len(split['train']), 'validation': len(split['validation']), 'test': len(split['test']),
    'test_feature_files_materialized_before_selection': sum(
        (WORKSPACE/f'features/{representation}/{key}.npz').exists()
        for representation in ('raw_clip', 'semantic18') for key in split['test'])}, indent=2))
print('Dataset restored: 58 recordings, train34 / validation12 / test12.')
print('Test feature files physically withheld until the selection lock exists.')
'''
    guarded = '''# Fresh processes enforce an offline socket guard and bounded run times.
from tools.run_scene_temporal_colab import run_guarded
COMMON = ['--manifest', 'dataset/manifest.json', '--split', 'dataset/split.json',
    '--originals-from', 'reference/clip_baseline_cv.json',
    '--feature-set', 'raw_clip=features/raw_clip',
    '--feature-set', 'semantic18=features/semantic18', '--output-dir', RUN]
run_guarded(WORKSPACE, RUN, 'focused_tests', 'unittest',
    ['discover', '-s', 'tests', '-p', 'test_reviewed_models.py', '-v'], 600)
RUN_FULL_TESTS = False
if RUN_FULL_TESTS:
    run_guarded(WORKSPACE, RUN, 'full_tests', 'unittest',
        ['discover', '-s', 'tests', '-v'], 900)
'''
    fit = '''# Classifiers, representation and threshold are selected on validation only.
assert not (RUN/'LOCKED_TEST_USED.json').exists(), 'This run already opened its test.'
run_guarded(WORKSPACE, RUN, 'fit', 'reviewed_experiment', ['fit', *COMMON,
    '--threads', '2', '--selection-metric', 'accuracy'], 1800)
selection = json.loads((RUN/'selection.json').read_text())
lock = json.loads((RUN/'selection_lock.json').read_text())
assert selection['test_feature_files_opened'] == 0
assert hashlib.sha256((RUN/'model_reviewed.joblib').read_bytes()).hexdigest() == lock['model_sha256']
assert hashlib.sha256((RUN/'selection.json').read_bytes()).hexdigest() == lock['selection_sha256']
print('Frozen validation-selected winner:', selection['winner']['candidate'])
print('Validation metrics:', selection['winner']['validation_metrics'])
print('Test feature files opened during fitting:', selection['test_feature_files_opened'])
'''
    score = '''# Open only the frozen winner's test representation, once, after selection.
from datetime import datetime, timezone
lock = json.loads((RUN/'selection_lock.json').read_text())
assert hashlib.sha256((RUN/'model_reviewed.joblib').read_bytes()).hexdigest() == lock['model_sha256']
assert hashlib.sha256((RUN/'selection.json').read_bytes()).hexdigest() == lock['selection_sha256']
assert not (RUN/'LOCKED_TEST_USED.json').exists(), 'The locked test cannot be reopened.'
archive_sha256 = hashlib.sha256(DATASET_ZIP.read_bytes()).hexdigest()
archive_receipt = json.loads((RUN/'dataset_archive_receipt.json').read_text())
assert archive_sha256 == archive_receipt['dataset_zip_sha256'], 'Dataset archive changed after restoration'
assert archive_receipt['test_feature_files_materialized_before_selection'] == 0
winner_representation = selection['winner']['candidate']['representation']
with zipfile.ZipFile(DATASET_ZIP) as archive:
    checksums = json.loads(archive.read('DATASET_MANIFEST.json'))['members_sha256']
    for key in split['test']:
        name = f'features/{winner_representation}/{key}.npz'
        payload = archive.read(name)
        assert hashlib.sha256(payload).hexdigest() == checksums[name]
        (WORKSPACE/name).write_bytes(payload)
(RUN/'test_feature_release.json').write_text(json.dumps({
    'model_sha256': lock['model_sha256'],
    'selection_lock_sha256': hashlib.sha256((RUN/'selection_lock.json').read_bytes()).hexdigest(),
    'archive_sha256': archive_sha256, 'released_test_sha256': sorted(split['test']),
    'representation': winner_representation,
    'released_at': datetime.now(timezone.utc).isoformat(), 'completed': True}, indent=2))
run_guarded(WORKSPACE, RUN, 'score', 'reviewed_experiment', ['score', *COMMON], 1800)
report = json.loads((RUN/'locked_test_report.json').read_text())
print('Locked mixed-domain development test metrics:')
print(json.dumps(report['metrics'], indent=2))
print('Per-domain metrics:', json.dumps(report['domain_metrics'], indent=2))
print('Development accuracy >80%:', report['metrics']['accuracy'] > .80)
print('Development accuracy >=87%:', report['metrics']['accuracy'] >= .87)
print('This within-session development test does not establish new-scene/session accuracy.')
'''
    inference = '''# Cached-feature inference receives no verdict or event timestamp inputs.
representative = sorted(split['train'])[0]
run_guarded(WORKSPACE, RUN, 'inference', 'reviewed_experiment', ['predict',
    '--feature', WORKSPACE/f'features/{winner_representation}/{representative}.npz',
    '--model', RUN/'model_reviewed.joblib', '--output-dir', RUN/'inference'], 120)
print('Prediction verification covers an existing feature cache; it is not uncached video latency.')
'''
    summarize = '''# Recompute metrics and verify split ownership, model lock and release chronology.
run_guarded(WORKSPACE, RUN, 'summary', 'tools.summarize_reviewed', [
    '--results-root', RUN, '--manifest', 'dataset/manifest.json',
    '--split', 'dataset/split.json', '--output', RUN/'reviewed_experiment_summary.json'], 300)
summary = json.loads((RUN/'reviewed_experiment_summary.json').read_text())
assert summary['split_labels_sources_groups_model_threshold_and_metrics_verified'] is True
print('Verified test accuracy:', summary['test_metrics']['accuracy'])
print('Verified candidate count:', summary['candidate_count'])
'''
    export = '''# Generate the clean notebook against exact installed source bytes.
build_args = [sys.executable, 'tools/build_reviewed_notebook.py',
    '--baseline-cv', 'reference/clip_baseline_cv.json',
    '--notebook', str(WORKSPACE/'datathon_reviewed.ipynb'),
    '--bundle-out', str(RUN/'export_source.zip')]
measured_path = RUN/'reviewed_experiment_summary.json'
if measured_path.is_file():
    build_args += ['--measured-summary', str(measured_path)]
else:
    measured_reference = WORKSPACE/'reference/measured_reviewed_summary.json'
    if measured_reference.is_file():
        build_args += ['--measured-summary', str(measured_reference)]
subprocess.run(build_args, cwd=WORKSPACE, check=True)
members = {}
with zipfile.ZipFile(RUN/'export_source.zip') as source:
    for name in source.namelist():
        if name.endswith('.py'):
            assert source.read(name) == (WORKSPACE/name).read_bytes()
        members['source/'+name] = source.read(name)
for path in RUN.rglob('*'):
    if path.is_file() and path.suffix in {'.json', '.joblib', '.log'}:
        assert not (set(path.relative_to(WORKSPACE).parts) & {'features', 'dataset', 'open'})
        members[path.relative_to(WORKSPACE).as_posix()] = path.read_bytes()
for name in ('reviewed_experiment_summary.json', 'reviewed_model_comparison.csv',
             'reviewed_model_comparison.png'):
    path = WORKSPACE/'artifacts'/name
    if path.is_file():
        members['artifacts/'+name] = path.read_bytes()
notebook_path = WORKSPACE/'datathon_reviewed.ipynb'
clean = json.loads(notebook_path.read_text())
assert all(not cell.get('outputs') and cell.get('execution_count') is None
           for cell in clean['cells'] if cell['cell_type'] == 'code')
members[notebook_path.name] = notebook_path.read_bytes()
members['EXPORT_MANIFEST.json'] = json.dumps({
    'contains_media_workbooks_private_descriptions_or_feature_caches': False,
    'members_sha256': {name: hashlib.sha256(value).hexdigest()
                       for name,value in sorted(members.items())}}, indent=2).encode()
RESULTS_ZIP = BASE/'datathon_reviewed_results.zip'
with zipfile.ZipFile(RESULTS_ZIP, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name,value in sorted(members.items()):
        archive.writestr(name,value)
print('Export:', RESULTS_ZIP, 'Bytes:', RESULTS_ZIP.stat().st_size)
print('SHA256:', hashlib.sha256(RESULTS_ZIP.read_bytes()).hexdigest())
if Path('/content').is_dir():
    from google.colab import files
    files.download(str(RESULTS_ZIP))
'''
    intro = '''# Reviewed custom labels and a locked mixed-domain development test

The human-reviewed workbook supplies simple violation/no_violation targets and
16 custom point timestamps. The fixed study uses 40 original recordings and 18
reviewed MOV recordings: train34 / validation12 / test12. The MOV family had
the largest usable label coverage before model scores. Every custom MP4 is
excluded from training, validation, test and model selection.

Known native source overlaps stay together. These groups are not human-verified
physical scene identities. Custom footage comes from one filming batch, and the
original recordings were examined during earlier development. This camera-exclusive
within-session test therefore does not establish unseen-scene/session accuracy.

Frozen visual features are computed locally before upload. Training runs in native
Python with no audio, hosted LLM APIs, private descriptions or video uploads to AI
services. More than 80% and at least 87% are goals, not assumed outcomes.
'''
    cells = [make_cell("markdown", intro), make_cell("code", bootstrap), make_cell("code", setup),
        make_cell("markdown", "## Restore private numeric inputs\nOnly checksummed train/validation caches are extracted initially. Test caches stay physically unavailable to the fitting command."),
        make_cell("code", restore), make_cell("code", guarded),
        make_cell("markdown", "## Fit and freeze the validation-selected winner\nThe fixed candidate list uses training data for preprocessing and fitted heads. Validation selects the representation, candidate and threshold. Test features remain unopened."),
        make_cell("code", fit),
        make_cell("markdown", "## Score the locked test once\nThe model and selection hashes must match the frozen lock before the winner's test caches are extracted. The scoring command refuses to reopen a used test."),
        make_cell("code", score), make_cell("code", inference), make_cell("code", summarize)]
    if measured is not None:
        compact = {key: measured[key] for key in ("evaluation", "partition_counts", "candidate_count",
            "selected_candidate", "validation_metrics", "test_metrics", "test_domain_metrics",
            "target_achieved_on_this_locked_test", "original_87pct_target_achieved_on_this_locked_test",
            "accuracy_above_80pct_on_new_sessions_established", "comparison_note") if key in measured}
        cells.append(make_cell("markdown", "## Saved completed measurements\n\n```json\n" +
                               json.dumps(compact, indent=2) + "\n```\n"))
    cells += [make_cell("markdown", "## Export the reproducible result\nExport trained heads, numeric reports, runtimes, exact source and the clean notebook. Private data, feature caches, audio, frames and workbooks remain excluded."),
              make_cell("code", export)]
    return cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cv", type=Path)
    parser.add_argument("--measured-summary", type=Path)
    parser.add_argument("--notebook", type=Path, default=ROOT / "datathon_reviewed.ipynb")
    parser.add_argument("--bundle-out", type=Path, default=ROOT / "artifacts/reviewed_source.zip")
    args = parser.parse_args(argv)
    bundle, measured = reviewed_bundle(ROOT, args.baseline_cv or default_baseline(ROOT), args.measured_summary)
    cells = notebook_cells(bundle, measured)
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<reviewed-notebook-cell>", "exec")
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
        "language": "python", "name": "python3"}, "language_info": {"name": "python"},
        "colab": {"name": args.notebook.name, "provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    args.notebook.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    args.bundle_out.parent.mkdir(parents=True, exist_ok=True)
    args.bundle_out.write_bytes(bundle)
    print(json.dumps({"notebook": str(args.notebook), "cells": len(cells),
        "source_bytes": len(bundle), "source_sha256": hashlib.sha256(bundle).hexdigest(),
        "measured_summary_included": measured is not None}))


if __name__ == "__main__":
    main()

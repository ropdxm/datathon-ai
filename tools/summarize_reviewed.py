"""Recompute locked mixed-data results and verify split/model ownership."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datathon_ml import model as base
from datathon_ml.reviewed_models import _metrics, _source_hashes, validate_split


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(run, records, split):
    partitions, provenance = validate_split(records, split)
    train = {row['sha256'] for row in partitions['train']}
    validation = {row['sha256'] for row in partitions['validation']}
    test = {row['sha256']: row for row in partitions['test']}
    selection = read(run/'selection.json')
    lock = read(run/'selection_lock.json')
    report = read(run/'locked_test_report.json')
    receipt = read(run/'LOCKED_TEST_USED.json')
    release = read(run/'test_feature_release.json')
    if selection['test_feature_files_opened'] != 0 or not report['selected_without_test']:
        raise ValueError('Test data were available to selection')
    if not receipt.get('completed') or receipt.get('test_open_count') != 1 or report['test_open_count'] != 1:
        raise ValueError('Exactly one completed locked test is required')
    if lock['model_sha256'] != sha(run/'model_reviewed.joblib') or lock['selection_sha256'] != sha(run/'selection.json'):
        raise ValueError('Frozen model or validation selection bytes changed')
    if (not release.get('completed') or release['model_sha256'] != lock['model_sha256']
            or release['selection_lock_sha256'] != sha(run/'selection_lock.json')
            or set(release['released_test_sha256']) != set(test)
            or len(release['released_test_sha256']) != len(test)):
        raise ValueError('Test feature release does not match the frozen model and owners')
    if not (datetime.fromisoformat(lock['locked_at']) <= datetime.fromisoformat(release['released_at'])
            <= datetime.fromisoformat(receipt['opened_at'])):
        raise ValueError('Test features were not released between model freeze and test opening')
    if not (provenance['sha256'] == lock['split_sha256'] == report['split']['sha256']):
        raise ValueError('Split differs from the predeclared model lock')
    if lock['source_sha256'] != _source_hashes() or selection['source_sha256'] != _source_hashes():
        raise ValueError('Current ML source differs from the fitted source')
    for candidate in selection['leaderboard']:
        if set(candidate['preprocessing_fit_sha256']) != train:
            raise ValueError('Preprocessing owner differs from training partition')
        rows = candidate['validation_predictions']
        if len(rows) != len(validation) or {row['sha256'] for row in rows} != validation:
            raise ValueError('Candidate selection predictions differ from validation owners')
        measured = base.classification_metrics([row['label'] for row in rows],
            [row['score'] for row in rows], [row['prediction'] for row in rows])
        for key in ('accuracy','balanced_accuracy','precision','recall','specificity','f1','f2'):
            if abs(measured[key]-candidate['validation_metrics'][key]) > 1e-12:
                raise ValueError('Validation metric differs from numeric predictions')
    rows = report['predictions']
    if len(rows) != len(test) or {row['sha256'] for row in rows} != set(test):
        raise ValueError('Test predictions do not cover locked test exactly once')
    for row in rows:
        original = test[row['sha256']]
        if any(row[key] != original[key] for key in ('label','event_times','group_id','domain')):
            raise ValueError('Test labels, anchors, groups or domains differ from frozen dataset')
    metrics = _metrics(rows)
    for key in ('accuracy','balanced_accuracy','precision','recall','specificity','f1','f2','mean_group_accuracy'):
        if abs(metrics[key]-report['metrics'][key]) > 1e-12:
            raise ValueError('Test metric differs from numeric predictions')
    if metrics['confusion_matrix'] != report['metrics']['confusion_matrix']:
        raise ValueError('Test confusion matrix differs from predictions')
    if selection['winner']['candidate'] != report['selected_candidate']:
        raise ValueError('Locked test model is not the validation-selected global winner')
    if selection['winner']['threshold'] != report['threshold'] or report['model_sha256'] != lock['model_sha256']:
        raise ValueError('Model/threshold changed between selection and test')
    return selection, report, receipt, release, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/reviewed_experiment_summary.json')
    args = parser.parse_args()
    records, split = read(args.manifest), read(args.split)
    selection, report, receipt, release, metrics = verify(args.results_root, records, split)
    runtime = {}
    for path in args.results_root.glob('*_runtime.json'):
        runtime[path.stem] = read(path)
    for phase in ('fit', 'score', 'inference'):
        value = runtime.get(phase+'_runtime')
        if value is None or not value.get('completed') or value.get('returncode') != 0 or value.get('network_attempts') != 0:
            raise ValueError('Missing successful offline phase: '+phase)
    audits = {}
    for name in ('reviewed_workbook_audit.json','reviewed_scene_group_audit.json','reviewed_protocol_audit.json'):
        path = ROOT/'artifacts'/name
        if not path.is_file():
            path = ROOT/'reference'/name
        if path.is_file():
            audits[name] = read(path)
    result = {'schema_version':1, 'evaluation':report['evaluation'],
        'target_accuracy_strictly_greater_than':.80,
        'target_achieved_on_this_locked_test':metrics['accuracy'] > .80,
        'original_87pct_target_achieved_on_this_locked_test':metrics['accuracy'] > .87,
        'accuracy_above_80pct_on_new_sessions_established':False,
        'split_labels_sources_groups_model_threshold_and_metrics_verified':True,
        'grouping_protocol':split['protocol'], 'partition_counts':report['partition_counts'],
        'candidate_count':len(selection['candidates']), 'candidate_grid':selection['candidates'],
        'selected_candidate':report['selected_candidate'], 'threshold':report['threshold'],
        'validation_metrics':selection['winner']['validation_metrics'],
        'test_metrics':metrics, 'test_domain_metrics':report['domain_metrics'],
        'test_predictions':report['predictions'], 'test_open_receipt':receipt,
        'test_feature_release_receipt':release,
        'selected_without_test':True, 'training_refit_on_validation':False,
        'test_features_physically_extracted_after_selection_lock':True,
        'source_sha256':selection['source_sha256'], 'model_sha256':report['model_sha256'],
        'split_file_sha256':sha(args.split), 'manifest_file_sha256':sha(args.manifest),
        'runtime':runtime, 'audits':audits,
        'previous_original_fold_reference_accuracy':.625,
        'comparison_note':'Previous62.5%used40originalOOFdevelopment; this mixed12-recordtest is a different protocol and not a paired accuracy improvement.',
        'audio_used':False, 'hosted_LLM_or_vision_API_used':False,
        'limitations':report['limitations']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'test_metrics':metrics,
        'domain_metrics':report['domain_metrics'],
        'target_achieved':result['target_achieved_on_this_locked_test']}))


if __name__ == '__main__':
    main()

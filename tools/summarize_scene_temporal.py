"""Verify native temporal results against frozen source targets and fold owners."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datathon_ml.scene_temporal import _metrics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(run, reference, source_root):
    report = read(run/'evaluation/scene_temporal_cv_report.json')
    expected = {row['sha256']: row for row in reference['predictions']}
    predictions = report['predictions']
    actual = {row['sha256']: row for row in predictions}
    if len(predictions) != len(actual) or set(actual) != set(expected):
        raise ValueError('Predictions must cover every frozen original source exactly once')
    for key,row in actual.items():
        if row['label'] != expected[key]['label'] or row['fold'] != expected[key]['fold']:
            raise ValueError('Prediction label or outer fold differs from the frozen reference')
    for fold in report['folds']:
        test = set(fold['test_sha256'])
        train = set(expected)-test
        if set(fold['preprocessing_fit_sha256']) != train:
            raise ValueError('Outer preprocessing owners differ from training sources')
        if set(fold['selection']['selection_source_sha256']) != train:
            raise ValueError('Architecture/threshold selection owners differ from training sources')
        if set(fold['train_groups']) & set(fold['test_groups']):
            raise ValueError('Related groups cross an outer fold')
        for inner in fold['selection']['selection_folds']:
            owners = set(inner['preprocessing_fit_sha256'])
            valid = set(inner['validation_sha256'])
            if owners & valid or owners | valid != train:
                raise ValueError('Inner ownership does not partition outer training sources')
            if set(inner['train_groups']) & set(inner['validation_groups']):
                raise ValueError('Related groups cross an inner fold')
    measured = _metrics(predictions)
    for key in ('accuracy','balanced_accuracy','precision','recall','specificity','f1','f2'):
        if abs(measured[key]-report['metrics'][key]) > 1e-12:
            raise ValueError('Stored metrics differ from numeric OOF predictions')
    if measured['confusion_matrix'] != report['metrics']['confusion_matrix']:
        raise ValueError('Stored confusion matrix differs from predictions')
    for name,relative in [('scene_temporal','datathon_ml/scene_temporal.py'),
                          ('model','datathon_ml/model.py'),('temporal','datathon_ml/temporal.py')]:
        if hashlib.sha256((source_root/relative).read_bytes()).hexdigest() != report['source_sha256'][name]:
            raise ValueError('Measured ML source differs from the current source')
    runtime = {}
    for name in ('cv','fit','inference','focused_tests'):
        value = read(run/(name+'_runtime.json'))
        if not value.get('completed') or value.get('returncode') != 0 or value.get('network_attempts') != 0:
            raise ValueError('Missing completed network-guarded phase: '+name)
        runtime[name] = value
    return report, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=ROOT/'reference/clip_baseline_cv.json')
    previous_path = ROOT/'artifacts/boosted_experiment_summary.json'
    if not previous_path.is_file():
        previous_path = ROOT/'reference/boosted_experiment_summary.json'
    parser.add_argument('--previous-summary', type=Path, default=previous_path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/scene_temporal_experiment_summary.json')
    args = parser.parse_args()
    report,runtime = verify(args.results_root, read(args.reference), args.source_root)
    previous = read(args.previous_summary)
    best = max(previous['model_comparisons'], key=lambda row: row['metrics']['accuracy'])
    comparison = [{'model':best['model'], 'metrics':best['metrics'], 'origin':'previous_development'},
                  {'model':'point_guided_tcn', 'metrics':report['metrics'], 'origin':'new_native_colab'}]
    def audit(name):
        path = ROOT/'artifacts'/name
        if not path.is_file():
            path = ROOT/'reference'/name
        return read(path) if path.is_file() else None
    custom = audit('custom_label_verification.json')
    pairs = audit('custom_pairing_audit.json')
    result = {'schema_version':1, 'target_accuracy':.87,
        'new_model_target_achieved_on_development':report['metrics']['accuracy'] >= .87,
        'evaluation':report['evaluation'], 'grouping':report['grouping'],
        'independent_new_scene_evaluation_completed':False,
        'all_original_SHA_labels_folds_and_nested_ownership_verified':True,
        'measured_source_bytes_match_current_ML_source':True,
        'model_comparisons':comparison, 'runtime':runtime,
        'custom_label_audit':custom, 'custom_pairing_audit':pairs,
        'custom_supervised_training_sources':0,
        'custom_ground_truth_blocker':'No human-verified custom targets; camera-pair proposals do not verify labels.',
        'point_localization_metrics':report['metrics']['localization'],
        'candidate_grid':report['candidates'], 'config':report['config'],
        'source_sha256':report['source_sha256'],
        'predictions':report['predictions'], 'folds':report['folds'],
        'audio_used':False, 'hosted_LLM_or_vision_API_used':False,
        'limitations':report['limitations']}
    full_tests = args.results_root/'full_tests_runtime.json'
    if full_tests.is_file():
        result['full_regression_tests'] = read(full_tests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'output':str(args.output), 'metrics':report['metrics'],
        'new_model_target_achieved':result['new_model_target_achieved_on_development']}))


if __name__ == '__main__':
    main()

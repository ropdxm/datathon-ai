"""Run semantic perception + temporal rules in the team's Colab GPU runtime.

No data, frames, descriptions, model-server calls, or hosted inference API.
Stdout contains aggregate measurements; state/event proposals stay in local files.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def run(arguments, *, timeout=None):
    process = subprocess.Popen([str(a) for a in arguments], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait(timeout=timeout):
        raise RuntimeError("Command failed: " + str(arguments[:4]))


def export(workspace):
    destination = Path('/content/datathon_temporal_results.zip') if Path('/content').is_dir() else workspace/'artifacts'/'datathon_temporal_results.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_root = workspace/'runs'/'semantic_temporal_v1'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in run_root.rglob('*'):
            if path.is_file() and path.suffix in {'.json','.joblib','.csv','.npz'}:
                archive.write(path, path.relative_to(workspace))
        sources = [*workspace.glob('datathon_ml/*.py'), *workspace.glob('tests/test_*.py'),
            *workspace.glob('tools/*temporal*.py'), workspace/'tools'/'propose_event_review.py',
            *workspace.glob('*experiment.py'),
            *workspace.glob('requirements*.txt'), workspace/'duplicate_aliases.json',
            workspace/'reference'/'clip_baseline_cv.json']
        for path in sources:
            if path.is_file():
                archive.write(path, 'source/'+path.relative_to(workspace).as_posix())
    print('TEMPORAL_RESULTS_EXPORT_READY', str(destination), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path('/content/datathon_ml_workspace'))
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--skip-qwen',action='store_true')
    parser.add_argument('--skip-clip',action='store_true')
    parser.add_argument('--skip-heads',action='store_true',
        help='Omit the fixed seven-candidate linear/RBF/shallow-tree comparison.')
    args=parser.parse_args()
    workspace=args.workspace.resolve()
    os.chdir(workspace)
    run_root=workspace/'runs'/'semantic_temporal_v1'
    run_root.mkdir(parents=True,exist_ok=True)
    reference=workspace/'reference'/'clip_baseline_cv.json'
    if not reference.is_file():
        raise FileNotFoundError('Original held-out fold report required: '+str(reference))
    summary={'evaluation':'development_comparison_same_original_outer_folds',
        'target_accuracy':.87,'models':{},'hosted_inference_api_used':False,
        'event_annotations_status':'machine_proposals_require_human_verification',
        'limitations':['40 unique videos; repeated development comparisons',
            'No independent new-camera test','No verified event types or complete intervals',
            'Observable dwell/max summaries can correlate with exposure length; no duration field is a predictor',
            'Frozen rules propose visible patterns; legal violations are not established']}
    summary_path=run_root/'summary.json'
    if summary_path.exists():
        summary['models'].update(json.loads(summary_path.read_text())['models'])
    base=workspace/'runs'/'clip_b32'
    try:
        if not args.skip_clip or not (base/'manifest.json').exists():
            run([sys.executable,'-u','experiment.py','extract','--data-root',args.data_root.resolve(),
                '--output-dir',base,'--verified-aliases','duplicate_aliases.json',
                '--backbone','clip_vit_b32_openai'])
        manifest=base/'manifest.json'
        for family in (['semantic_clip'] if args.skip_qwen else ['semantic_clip','semantic_qwen4b']):
            if args.skip_clip and family=='semantic_clip':
                continue
            destination=run_root/family
            started=time.perf_counter()
            if family=='semantic_clip':
                run([sys.executable,'-u','-m','datathon_ml.semantic_clip',
                    '--input-dir',base/'features','--output-dir',destination/'features','--device','cpu','--offline'])
            else:
                run([sys.executable,'-u','-m','datathon_ml.semantic_vlm',
                    '--manifest',manifest,'--output-dir',destination/'features'])
            run([sys.executable,'-u','temporal_experiment.py','cv','--manifest',manifest,
                '--feature-dir',destination/'features','--output-dir',destination,'--folds-from',reference])
            run([sys.executable,'-u','temporal_experiment.py','fit','--manifest',manifest,
                '--feature-dir',destination/'features','--output-dir',destination])
            report=json.loads((destination/'evaluation'/'temporal_cv_report.json').read_text())
            summary['models'][family]={'metrics':report['metrics'],
                'frozen_rule_metrics':report['frozen_rules']['metrics'],
                'elapsed_seconds':time.perf_counter()-started,
                'model_path':str((destination/'model'/'model_temporal.joblib').relative_to(workspace))}
            print('REAL_SEMANTIC_TEMPORAL_RESULT',family,json.dumps(summary['models'][family]),flush=True)
            if not args.skip_heads:
                head_started=time.perf_counter()
                head_destination=destination/'nonlinear_heads'
                run([sys.executable,'-u','semantic_head_experiment.py','pipeline','--manifest',manifest,
                    '--feature-dir',destination/'features','--output-dir',head_destination,
                    '--folds-from',reference])
                head_report=json.loads((head_destination/'evaluation'/'semantic_head_cv_report.json').read_text())
                key=family+'_selected_heads'
                summary['models'][key]={'metrics':head_report['metrics'],
                    'family_comparisons':{name:row['metrics'] for name,row in head_report['family_comparisons'].items()},
                    'frozen_rule_metrics':head_report['frozen_rules']['metrics'],
                    'candidate_count':len(head_report['candidate_grid']),
                    'primary_policy':head_report['primary_policy'],
                    'elapsed_seconds':time.perf_counter()-head_started,
                    'model_path':str((head_destination/'model'/'model_semantic_head.joblib').relative_to(workspace))}
                print('REAL_SEMANTIC_SELECTED_HEAD_RESULT',key,json.dumps(summary['models'][key]),flush=True)
            run([sys.executable,'-u','tools/propose_event_review.py','--manifest',manifest,
                '--feature-dir',destination/'features','--output',destination/'annotation_machine_proposals.csv'])
            summary['target_achieved_on_development']=any(row['metrics']['accuracy']>=.87 for row in summary['models'].values())
            summary_path.write_text(json.dumps(summary,indent=2))
            export(workspace)
    finally:
        summary_path.write_text(json.dumps(summary,indent=2))
        export(workspace)


if __name__=='__main__':
    main()

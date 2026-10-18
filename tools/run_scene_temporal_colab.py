"""Run the bounded temporal comparison on numeric caches in native Python."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


WORKER = r'''
import json, runpy, sys, time
from pathlib import Path
from tools.benchmark_semantic_vlm import blocked_network, configure_offline_environment
module, report_path, arguments = sys.argv[1:]
configure_offline_environment()
started = time.perf_counter()
report = {'status': 'started', 'completed': False}
sys.argv = [module + '.py'] + json.loads(arguments)
with blocked_network() as network:
    try:
        runpy.run_module(module, run_name='__main__')
        report.update(status='completed', completed=True)
    except SystemExit as error:
        report.update(status='completed' if error.code in (None, 0) else 'failed',
                      completed=error.code in (None, 0))
        raise
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        report.update(network_attempts=network['attempts'],
                      elapsed_seconds=time.perf_counter()-started)
        Path(report_path).write_text(json.dumps(report, indent=2))
        print('OFFLINE_RUNTIME', json.dumps(report), flush=True)
'''


def run_guarded(workspace, destination, name, module, arguments, budget=1800):
    destination.mkdir(parents=True, exist_ok=True)
    runtime = destination / (name + '_runtime.json')
    log = destination / (name + '.log')
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONHASHSEED='42', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    started = time.perf_counter()
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen([sys.executable, '-u', '-c', WORKER, module,
            str(runtime.resolve()), json.dumps(list(map(str, arguments)))],
            cwd=workspace, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=budget)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            process.kill()
            process.wait()
            runtime.write_text(json.dumps({'status': 'timed_out' if isinstance(error,
                subprocess.TimeoutExpired) else 'cancelled', 'completed': False,
                'network_attempts': None, 'elapsed_seconds': time.perf_counter()-started,
                'budget_seconds': budget}, indent=2))
            raise
    report = json.loads(runtime.read_text()) if runtime.exists() else {'status': 'failed_without_report'}
    report.update(returncode=code, budget_seconds=budget)
    runtime.write_text(json.dumps(report, indent=2))
    print(name.upper(), json.dumps(report), flush=True)
    if code or not report.get('completed') or report.get('network_attempts') != 0:
        raise RuntimeError('Native guarded command failed; inspect ' + str(log))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('/content/datathon_ml_workspace'))
    parser.add_argument('--manifest', type=Path, default=Path('cache/original_manifest.json'))
    parser.add_argument('--features', type=Path, default=Path('cache/semantic_clip'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/scene_temporal_v1'))
    parser.add_argument('--budget', type=int, default=1800)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    run = (workspace / args.output_dir).resolve()
    manifest = (workspace / args.manifest).resolve()
    features = (workspace / args.features).resolve()
    reference = workspace / 'reference/clip_baseline_cv.json'
    original = json.loads(manifest.read_text())
    expected = {r['sha256']: int(r['label']) for r in json.loads(reference.read_text())['predictions']}
    if {r['sha256']: int(r['label']) for r in original} != expected or len(original) != len(expected):
        raise ValueError('Cached sources/targets must equal the frozen original reference')
    common = ['--manifest', manifest, '--feature-dir', features, '--output-dir', run,
              '--originals-from', reference, '--threads', '2', '--device', 'cpu']
    run_guarded(workspace, run, 'cv', 'scene_temporal_experiment',
        ['cv', *common, '--folds-from', reference], args.budget)
    run_guarded(workspace, run, 'fit', 'scene_temporal_experiment', ['fit', *common], args.budget)
    model = run / 'model/model_scene_temporal.joblib'
    if not model.is_file():
        raise FileNotFoundError('Fitted temporal artifact missing')
    run_guarded(workspace, run, 'inference', 'scene_temporal_experiment',
        ['predict', '--model', model, '--feature', features / (sorted(expected)[0]+'.npz'),
         '--output-dir', run / 'inference'], 120)
    source_names = ['datathon_ml/scene_temporal.py', 'scene_temporal_experiment.py',
                    'tools/run_scene_temporal_colab.py']
    (run / 'source_hashes.json').write_text(json.dumps({name: hashlib.sha256(
        (workspace/name).read_bytes()).hexdigest() for name in source_names}, indent=2))
    print('TEMPORAL_EXPERIMENT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

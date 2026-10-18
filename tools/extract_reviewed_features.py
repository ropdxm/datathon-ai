"""Extract fixed local visual features for a locked reviewed-data experiment.

The encoder consumes media bytes and paths only. Labels, event timestamps,
split roles and filenames never select frames, crops or model inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/reviewed_features')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--sample-fps', type=float, default=2.)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError('threads must be positive')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(args.threads)
    import torch
    from datathon_ml.clip_features import ClipFeatureConfig, extract_clip_features
    from datathon_ml.semantic_clip import convert_clip_cache
    from tools.benchmark_semantic_vlm import blocked_network, configure_offline_environment
    torch.set_num_threads(args.threads)
    records = json.loads(args.manifest.read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError('Expected a nonempty reviewed manifest')
    if len({row['sha256'] for row in records}) != len(records):
        raise ValueError('Repeated source/derivative SHA in reviewed manifest')
    # Strip supervision before passing data to the immutable visual extractor.
    inputs = [{'sha256': row['sha256'], 'path': row['path']} for row in records]
    configure_offline_environment()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runtime = {'status': 'started', 'completed': False, 'audio_used': False,
        'hosted_inference_API_used': False,
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with blocked_network() as network:
        try:
            paths = extract_clip_features(inputs, args.output_dir/'clip',
                ClipFeatureConfig(device=args.device, batch_size=32,
                    sample_fps=args.sample_fps, allow_download=False))
            semantic = convert_clip_cache(args.output_dir/'clip',
                args.output_dir/'semantic', device='cpu', offline=True)
            if set(paths) != {row['sha256'] for row in records}:
                raise ValueError('Image feature ownership differs from the manifest')
            if not set(paths).issubset(semantic):
                raise ValueError('Semantic conversion is incomplete')
            runtime.update(status='completed', completed=True,
                completed_sources=len(paths), torch_version=str(torch.__version__),
                device=args.device, threads=args.threads, sample_fps=args.sample_fps,
                semantic_sources=len(paths))
        except BaseException as error:
            runtime.update(status='failed', error_type=type(error).__name__)
            raise
        finally:
            runtime.update(elapsed_seconds=time.perf_counter()-started,
                network_attempts=network['attempts'])
            (args.output_dir/'extraction_runtime.json').write_text(json.dumps(runtime, indent=2))
            print('REVIEWED_EXTRACTION_RUNTIME', json.dumps(runtime), flush=True)
    if runtime['network_attempts']:
        raise RuntimeError('Unexpected network attempt during native feature extraction')


if __name__ == '__main__':
    main()

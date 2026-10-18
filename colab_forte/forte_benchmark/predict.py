"""Offline binary baseline inference; this is not the complete jury API or a legal-rule engine.

From the extracted inference_bundle directory:
python -m forte_benchmark.predict --bundle . --video /path/input.mov --device auto
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import av
import torch

from .data import make_windows
from .models import load_encoder_bundle, make_head
from .training import predict_scores
from .workflow import aggregate_predictions, choose_device, environment, sync


def video_duration(path):
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('Input has no video stream.')
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base
        else:
            origin, last, previous = None, None, None
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError('Video frame has no PTS.')
                t = float(frame.pts * frame.time_base)
                origin = t if origin is None else origin
                previous, last = last, t
            if last is None:
                raise ValueError('Video has no decodable frames.')
            duration = last - origin + (last - previous if previous is not None else 1 / 30)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Cannot determine a positive finite duration.')
        return duration


def export_evidence(path, target_time, destination):
    """Return actual source PTS relative to the first frame, never an invented frame timestamp."""
    closest, closest_time, distance, origin = None, None, float('inf'), None
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            if frame.pts is None:
                raise ValueError('Evidence frame has no PTS.')
            t = float(frame.pts * frame.time_base)
            origin = t if origin is None else origin
            t -= origin
            if abs(t - target_time) < distance:
                closest, closest_time, distance = frame.to_image(), t, abs(t - target_time)
            if t > target_time:
                break
    if closest is None:
        raise ValueError('No evidence frame could be decoded.')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    closest.save(destination, format='JPEG', quality=92)
    return closest_time


def predict_video(bundle, video, device='auto', evidence_path=None):
    bundle, video = Path(bundle).resolve(), Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    device = choose_device() if device == 'auto' else device
    started = time.perf_counter()
    config = json.loads((bundle / 'config.json').read_text(encoding='utf8'))
    encoder = load_encoder_bundle(bundle / 'encoder', device=device)
    head = make_head(config['model_id'], int(config['feature_dim']))
    head.load_state_dict(torch.load(bundle / 'head.pt', map_location='cpu', weights_only=True), strict=True)
    load_seconds = time.perf_counter() - started
    duration = video_duration(video)
    record = dict(video_id='input', name=video.name, aliases=[video.name], group_id='input',
                  sha256='0' * 64, label=0, split='test', event_times=[], duration_sec=duration, path=str(video))
    # Dummy label only satisfies the generic window record; sampling is label-independent.
    protocol = config['protocol']
    windows = make_windows([record], protocol['window_sec'], protocol['stride_sec'], training=False)
    encoder.clear_cache()
    sync(device)
    start_analysis = time.perf_counter()
    features = torch.stack([encoder.encode(str(video), w['start_sec'], w['end_sec']) for w in windows])
    scores = predict_scores(head, features, device=device)
    prediction = aggregate_predictions(windows, scores)[0]
    sync(device)
    analysis_seconds = time.perf_counter() - start_analysis
    winner = max(zip(windows, scores), key=lambda pair: (pair[1], -pair[0]['center_sec']))[0]
    result = dict(model_id=config['model_id'], video=str(video), duration_sec=duration,
                  score=prediction['score'], threshold=config['threshold'],
                  verdict='violation' if prediction['score'] >= config['threshold'] else 'no_violation',
                  timecode_sec=prediction['timecode_sec'],
                  candidate_window=dict(start_sec=winner['start_sec'], end_sec=winner['end_sec']),
                  event_type=None, normative_source=None,
                  timecode_method='highest-scoring window center; approximate candidate, not event-frame recognition',
                  model_load_seconds=load_seconds, analysis_seconds=analysis_seconds,
                  environment=environment(device))
    if evidence_path:
        result['timecode_sec'] = export_evidence(video, prediction['timecode_sec'], evidence_path)
        result['evidence_frame'] = str(Path(evidence_path).resolve())
        result['timecode_method'] += '; snapped to the nearest original frame PTS'
    encoder.clear_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = predict_video(args.bundle, args.video, args.device, args.evidence)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf8')
    print(payload)


if __name__ == '__main__':
    main()

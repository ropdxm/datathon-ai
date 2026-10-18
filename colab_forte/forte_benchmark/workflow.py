"""Colab orchestration. Validation chooses settings; test remains an explicit final action."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
import stat
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .data import dataset_fingerprint, load_dataset, make_windows, validate_manifest
from .metrics import evaluate_predictions, select_threshold

PROTOCOL = {
    'version': 'forte-window-binary-v1', 'window_sec': 8.0, 'stride_sec': 2.0,
    'train_negative_windows_per_video': 12,
    'positive_training': 'annotation-centered weak positive; all other positive windows unknown',
    'inference': 'uniform full-duration windows, independent of labels and event timestamps',
    'aggregation': 'maximum window score; earliest maximum; timestamp is window center',
    'localization_reference': 'nearest supplied annotation; ambiguous duplicates disclosed separately',
    'threshold': 'validation F1, then recall, then specificity, then higher threshold',
    'target': 'supplied binary video label; no invented event-type labels',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    pending.replace(path)


def new_run_directory(output_root, model_id):
    """Keep old test results separate from new weights, while sharing feature caches."""
    directory = Path(output_root) / model_id / f'run_{time.time_ns()}'
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def safe_extract_zip(archive, destination):
    """Validate the whole archive before writing; never follow archive symlinks."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            target = (destination / entry.filename).resolve()
            mode = entry.external_attr >> 16
            if (not target.is_relative_to(destination) or stat.S_ISLNK(mode)
                    or '\\' in entry.filename):
                raise ValueError(f'Unsafe archive member: {entry.filename}')
        if sum(e.file_size for e in z.infolist()) > 5 * 1024**3:
            raise ValueError('Archive expands beyond the 5 GiB dataset limit.')
        destination.mkdir(parents=True, exist_ok=True)
        z.extractall(destination)
    return destination


def aggregate_predictions(windows, scores):
    if len(windows) != len(scores) or not windows:
        raise ValueError('Every inference window needs one score.')
    best = {}
    for window, score in zip(windows, scores):
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Scores must be finite probabilities.')
        candidate = dict(video_id=window['video_id'], score=score,
                         timecode_sec=float(window['center_sec']))
        previous = best.get(candidate['video_id'])
        if previous is None or score > previous['score'] or (
                score == previous['score'] and candidate['timecode_sec'] < previous['timecode_sec']):
            best[candidate['video_id']] = candidate
    return [best[k] for k in sorted(best)]


def compare_reports(reports):
    """Reject mixed protocols/cohorts instead of presenting a misleading leaderboard."""
    reports = [json.loads(Path(r).read_text()) if isinstance(r, (str, Path)) else r for r in reports]
    if not reports:
        return []
    keys = ['dataset_fingerprint', 'protocol_fingerprint', 'metrics_fingerprint', 'split', 'evaluation_ids']
    reference = reports[0]
    for r in reports:
        for key in keys:
            if key not in r or r[key] != reference.get(key):
                raise ValueError(f'Incomparable reports: {key} differs or is missing.')
    if len({r['model_id'] for r in reports}) != len(reports):
        raise ValueError('Select one report per model; duplicate model IDs found.')
    return sorted(reports, key=lambda r: (-r['metrics']['F1'], -r['metrics'].get('recall', 0), r['model_id']))


def choose_device():
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def sync(device):
    import torch
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    elif str(device) == 'mps':
        torch.mps.synchronize()


def environment(device):
    import torch
    import torchvision
    import av
    return dict(python=platform.python_version(), platform=platform.platform(),
                torch=torch.__version__, torchvision=torchvision.__version__, av=av.__version__,
                device=str(device), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)


def runtime_requirements():
    """Export the actual training versions, without a CUDA-only local wheel suffix."""
    from importlib.metadata import version
    return ''.join(f'{name}=={version(name).split("+")[0]}\n'
                   for name in ['torch', 'torchvision', 'av', 'numpy', 'pillow', 'tqdm'])


def extract_features(encoder, windows, cache_dir, identity, force=False):
    import torch
    from tqdm.auto import tqdm
    window_identity = [{k: w[k] for k in ['sample_id', 'video_id', 'start_sec', 'end_sec']} for w in windows]
    key = digest(dict(identity=identity, encoder=encoder.config, windows=window_identity))
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'{key}.pt'
    if cache_file.exists() and not force:
        saved = torch.load(cache_file, map_location='cpu', weights_only=True)
        if saved.get('key') == key and saved['features'].shape[0] == len(windows):
            return saved['features'], dict(cache_hit=True, extraction_seconds=None)
    started = time.perf_counter()
    features = []
    for window in tqdm(windows, desc='Извлечение признаков'):
        feature = encoder.encode(window['path'], window['start_sec'], window['end_sec'])
        if not torch.isfinite(feature).all():
            raise ValueError(f'Nonfinite features: {window["sample_id"]}')
        features.append(feature.cpu().float())
    values = torch.stack(features)
    pending = cache_file.with_suffix('.tmp')
    torch.save(dict(key=key, features=values), pending)
    pending.replace(cache_file)
    return values, dict(cache_hit=False, extraction_seconds=time.perf_counter() - started)


def cap_training_negatives(windows, limit):
    grouped = defaultdict(list)
    for w in windows:
        grouped[w['video_id']].append(w)
    result = []
    for video_id in sorted(grouped):
        rows = grouped[video_id]
        if rows[0]['weak_label'] == 0 and len(rows) > limit:
            indices = sorted({round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)})
            rows = [rows[i] for i in indices]
        result.extend(rows)
    return result


def save_predictions(directory, split, records, windows, scores, predictions, threshold):
    directory = Path(directory)
    by_id = {r['video_id']: r for r in records}
    with (directory / f'predictions_{split}.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['video_id', 'name', 'label', 'score', 'predicted_label', 'timecode_sec', 'reference_times']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pred in predictions:
            record = by_id[pred['video_id']]
            writer.writerow(dict(pred, name=record['name'], label=record['label'],
                                 predicted_label=int(pred['score'] >= threshold),
                                 reference_times=json.dumps(record['event_times'])))
    with (directory / f'windows_{split}.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'start_sec', 'end_sec', 'center_sec', 'score'])
        writer.writeheader()
        for w, score in zip(windows, scores):
            writer.writerow({**{k: w[k] for k in ['video_id', 'start_sec', 'end_sec', 'center_sec']}, 'score': score})


def metric_identity():
    return hashlib.sha256(Path(__file__).with_name('metrics.py').read_bytes()).hexdigest()


def build_report(model_id, split, records, predictions, threshold, identity, extra=None):
    report = dict(model_id=model_id, split=split, dataset_fingerprint=identity,
                  protocol_fingerprint=digest(PROTOCOL), metrics_fingerprint=metric_identity(),
                  protocol=PROTOCOL, evaluation_ids=sorted(r['video_id'] for r in records),
                  threshold=float(threshold), metrics=evaluate_predictions(records, predictions, threshold),
                  predictions=predictions,
                  ambiguous_reference_video_ids=[r['video_id'] for r in records if r.get('ambiguous_timestamp')],
                  limitations=['Only supplied binary labels are trained; no event-type or legal-ground truth.',
                               'Timecode is the maximum-scoring window center, not a directly detected event frame.',
                               'Small inferred-scenario holdout; same camera and actor; no new-angle proof.'])
    unambiguous = [r for r in records if not r.get('ambiguous_timestamp')]
    if len(unambiguous) != len(records):
        ids = {r['video_id'] for r in unambiguous}
        report['metrics_excluding_ambiguous_time_references'] = evaluate_predictions(
            unambiguous, [p for p in predictions if p['video_id'] in ids], threshold)
    if extra:
        report.update(extra)
    return report


def run_experiment(model_id, manifest, data_root, output_root, config=None):
    import torch
    from .models import make_encoder, make_head
    from .training import train_head, predict_scores
    config = dict(config or {})
    validate_manifest(manifest)
    records = load_dataset(Path(data_root), manifest)
    identity = dataset_fingerprint(manifest)
    device = config.get('device') or choose_device()
    seed = int(config.get('seed', 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    output = new_run_directory(output_root, model_id)
    feature_cache = output.parent / 'cache'
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'protocol.json', PROTOCOL)
    write_json(output / 'environment.json', environment(device))
    print('Устройство:', device, '| split:', dict(Counter(r['split'] for r in records)))
    started = time.perf_counter()
    encoder = make_encoder(model_id, device=device, pretrained=True)
    load_seconds = time.perf_counter() - started
    training_records = [r for r in records if r['split'] == 'train']
    validation_records = [r for r in records if r['split'] == 'val']
    train_windows = cap_training_negatives(make_windows(training_records, training=True),
                                          PROTOCOL['train_negative_windows_per_video'])
    val_windows = make_windows(validation_records, training=False)
    x_train, extraction_train = extract_features(encoder, train_windows, feature_cache, identity)
    x_val, extraction_val = extract_features(encoder, val_windows, feature_cache, identity)
    # Deterministic head initialisation happens after encoder extraction as well.
    torch.manual_seed(seed)
    head = make_head(model_id, encoder.feature_dim)
    def validation_callback(candidate):
        scores = predict_scores(candidate, x_val, device=device)
        predictions = aggregate_predictions(val_windows, scores)
        threshold = select_threshold(validation_records, predictions)
        metrics = evaluate_predictions(validation_records, predictions, threshold)
        return dict(f1=metrics['F1'], recall=metrics['recall'], threshold=threshold)
    start_training = time.perf_counter()
    head, history, best_epoch = train_head(
        head, x_train, torch.tensor([w['weak_label'] for w in train_windows], dtype=torch.float32),
        [w['video_id'] for w in train_windows], validation_callback, output,
        device=device, seed=seed, epochs=int(config.get('epochs', 20)),
        lr=float(config.get('lr', 1e-3)), batch_size=int(config.get('batch_size', 16)),
        patience=int(config.get('patience', 4)))
    train_seconds = time.perf_counter() - start_training
    scores = predict_scores(head, x_val, device=device)
    predictions = aggregate_predictions(val_windows, scores)
    threshold = select_threshold(validation_records, predictions)
    extra = dict(environment=environment(device), best_epoch=best_epoch, seed=seed,
                 encoder=encoder.config, head_trainable_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
                 training_video_count=len(training_records), training_window_count=len(train_windows),
                 training_label_counts=dict(Counter(str(w['weak_label']) for w in train_windows)),
                 timings=dict(model_load_seconds=load_seconds, train_feature_extraction=extraction_train,
                              validation_feature_extraction=extraction_val, head_training_seconds=train_seconds),
                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None)
    report = build_report(model_id, 'val', validation_records, predictions, threshold, identity, extra)
    write_json(output / 'report_val.json', report)
    write_json(output / 'history.json', history)
    save_predictions(output, 'val', validation_records, val_windows, scores, predictions, threshold)
    bundle = output / 'inference_bundle'
    bundle.mkdir(exist_ok=True)
    encoder.save_bundle(bundle / 'encoder')
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, bundle / 'head.pt')
    write_json(bundle / 'config.json', dict(model_id=model_id, feature_dim=encoder.feature_dim,
                                          threshold=threshold, protocol=PROTOCOL, dataset_fingerprint=identity,
                                          encoder_config=encoder.config, version='1.0.0'))
    shutil.copytree(Path(__file__).parent, bundle / 'forte_benchmark', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (bundle / 'requirements.txt').write_text(runtime_requirements())
    encoder.clear_cache()
    print(json.dumps(report['metrics'], indent=2, ensure_ascii=False))
    return dict(model_id=model_id, manifest=manifest, records=records, encoder=encoder, head=head,
                threshold=threshold, identity=identity, output=output, device=device, report=report)


def evaluate_final_test(state):
    """Call once after choosing the final configuration using validation only."""
    from .training import predict_scores
    records = [r for r in state['records'] if r['split'] == 'test']
    windows = make_windows(records, training=False)
    features, extraction = extract_features(state['encoder'], windows, state['output'] / 'cache', state['identity'])
    scores = predict_scores(state['head'], features, device=state['device'])
    predictions = aggregate_predictions(windows, scores)
    report = build_report(state['model_id'], 'test', records, predictions, state['threshold'], state['identity'],
                          dict(environment=environment(state['device']), test_feature_extraction=extraction,
                               threshold_source='validation; frozen before test'))
    write_json(state['output'] / 'report_test.json', report)
    save_predictions(state['output'], 'test', records, windows, scores, predictions, state['threshold'])
    return report


def benchmark_validation(state):
    """Cold decoded-video cache, preloaded weights; includes decode/encode/head, excludes model load."""
    from .training import predict_scores
    import torch
    rows = []
    for record in [r for r in state['records'] if r['split'] == 'val']:
        state['encoder'].clear_cache()
        sync(state['device'])
        start = time.perf_counter()
        windows = make_windows([record], training=False)
        features = torch.stack([state['encoder'].encode(w['path'], w['start_sec'], w['end_sec']) for w in windows])
        scores = predict_scores(state['head'], features, device=state['device'])
        sync(state['device'])
        seconds = time.perf_counter() - start
        rows.append(dict(video_id=record['video_id'], duration_sec=record['duration_sec'],
                         analysis_seconds=seconds, window_count=len(windows),
                         prediction=aggregate_predictions(windows, scores)[0]))
    report = dict(environment=environment(state['device']), measurement='preloaded weights, cold decoded-video cache; complete uniform window scan',
                  videos=rows, max_analysis_seconds=max(r['analysis_seconds'] for r in rows),
                  mean_analysis_seconds=sum(r['analysis_seconds'] for r in rows) / len(rows))
    write_json(state['output'] / 'latency_validation.json', report)
    return report


def package_results(state):
    """Exclude private original videos and feature caches from the shareable result bundle."""
    output = Path(state['output'])
    archive = output.parent / f'{state["model_id"]}_{output.name}_results.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob('*')):
            relative = path.relative_to(output)
            if path.is_file() and 'cache' not in relative.parts and '__pycache__' not in relative.parts:
                z.write(path, arcname=str(Path(state['model_id']) / relative))
    return archive

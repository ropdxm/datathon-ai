"""Build a clean supervised QLoRA notebook and exact fitted-adapter export."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import build_context_notebook as context

DETAIL_ZIP_SHA = "3708bc48d39eb68ec3e4a1fca105434d79bc541f697153434050ceeca20a5138"
DETAIL_MANIFEST_SHA = "5659208315115d0b97789fa477fdacd047a3f70c545ff8f4a3482de531a4781c"
DETAIL_BIN_SHA = "8b56e60e0b1955b4ae6bbb4eabe3c9fe7484b217e3e25f71f05381f44e0a35fc"
PINNED = {
    "datathon_ml/qlora_vlm.py": "34841cbf90c8fec49f25503ea83e32f1e169997257869e12a59605a752a5be78",
    "qlora_experiment.py": "a3ee13eea3027ac55cad832c1f88c75125f67831fe2b373ca98081cb16d959ad",
    "datathon_ml/direct_context_vlm.py": "de9bb811fca065166b63e0a0e4b3cc271da22067675ceda4f0c8ada53e5531bd",
    "context_vlm_experiment.py": "9cd1a537113f12d49cb6063ce2cdecb63abeb2246ba5ff4418e559acdb7e68b8",
    "datathon_ml/direct_vlm.py": "b69ce15fad95889ff001ee540cf0e4c9c933c660e17dff4444f8951612e7b4a6",
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def collect_source(root):
    members = context.collect_source(root)
    for name, expected in PINNED.items():
        if name not in members or digest(members[name]) != expected:
            raise ValueError("A frozen supervised/inherited model source changed: " + name)
    required = ("tests/__init__.py", "tests/test_qlora_vlm.py", "tests/test_direct_context_vlm.py")
    if any(name not in members for name in required):
        raise FileNotFoundError("The supervised source/test snapshot is incomplete")
    members["tools/build_qlora_notebook.py"] = Path(__file__).read_bytes()
    return members


def bootstrap(bundle):
    return context.bootstrap(bundle).replace("V5_WORKSPACE", "V6_WORKSPACE").replace(
        "V5_RUN", "V6_RUN").replace("datathon_context_vlm_workspace", "datathon_qlora_workspace").replace(
        "runs/context_vlm_v5", "runs/qlora_v6")


SETUP = '''# Public setup only. Keep compatible public package versions fixed.
subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6',
    'peft==0.18.0','bitsandbytes==0.48.2','open_clip_torch==3.2.0',
    'safetensors','accelerate','-r','requirements-ml.txt'],check=True)
import torch
from importlib.metadata import version
assert torch.cuda.is_available(),'Enable a native GPU runtime'
assert version('transformers')=='4.57.6' and version('peft')=='0.18.0' and version('bitsandbytes')=='0.48.2'
from huggingface_hub import snapshot_download
from datathon_ml.direct_vlm import MODEL_ID,REVISION,WEIGHT_SHA256
def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
QWEN_CHECKPOINT=Path(globals().get('QWEN_CHECKPOINT',BASE/'public_qwen3vl_4b'))
if not all((QWEN_CHECKPOINT/name).is_file() and digest_file(QWEN_CHECKPOINT/name)==expected for name,expected in WEIGHT_SHA256.items()):
    snapshot_download(MODEL_ID,revision=REVISION,local_dir=str(QWEN_CHECKPOINT),
        allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','README.md'])
assert all(digest_file(QWEN_CHECKPOINT/name)==expected for name,expected in WEIGHT_SHA256.items())
QWEN_DEVICE='cuda:0'
print({'public_checkpoint_verified':True,'torch':str(torch.__version__),'cuda':torch.version.cuda,
    'transformers':version('transformers'),'peft':version('peft'),'bitsandbytes':version('bitsandbytes'),
    'license':'Apache-2.0','hosted_inference_API_used':False})
'''


RESTORE = '''# Read the private single-file transport in native Python; never display frames.
import hashlib,io,json,re,zipfile
from pathlib import Path
WORKSPACE=Path(V6_WORKSPACE); RUN=Path(V6_RUN)
RUN.mkdir(parents=True,exist_ok=True)
assert not any((RUN/name).exists() for name in ('FIT_STARTED.json','selection_lock.json','TEST_OPENED.json'))
BASE=Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('/content') if Path('/content').is_dir() else Path.cwd()
INPUT=Path('/kaggle/input') if Path('/kaggle/input').is_dir() else Path(globals().get('INPUT',BASE))
def digest_bytes(value): return hashlib.sha256(value).hexdigest()
DETAIL_ZIP_SHA=''' + repr(DETAIL_ZIP_SHA) + '''
DETAIL_MANIFEST_SHA=''' + repr(DETAIL_MANIFEST_SHA) + '''
DETAIL_BIN_SHA=''' + repr(DETAIL_BIN_SHA) + '''
def exact_input_candidates(filename):
    candidates=[BASE/filename,WORKSPACE/filename,INPUT/filename]
    if INPUT.is_dir(): candidates.extend(sorted(INPUT.rglob(filename)))
    return list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))
detail_archive=None; detail_root=None; transport_sha=None
for path in exact_input_candidates('reviewed_detail_framepack.bin'):
    raw=path.read_bytes()
    if digest_bytes(raw)!=DETAIL_BIN_SHA: continue
    header=b'DATATHON_PRIVATE_FRAMEPACK_V1\\n'
    assert raw.startswith(header) and len(header)==30
    payload=raw[len(header):]; assert digest_bytes(payload)==DETAIL_ZIP_SHA
    detail_archive=zipfile.ZipFile(io.BytesIO(payload)); transport_sha=DETAIL_BIN_SHA
    del raw,payload
    break
if detail_archive is None:
    for path in exact_input_candidates('reviewed_detail_framepack.zip'):
        if digest_bytes(path.read_bytes())==DETAIL_ZIP_SHA:
            detail_archive=zipfile.ZipFile(path); break
if detail_archive is not None:
    inventory_payload=detail_archive.read('DETAIL_FRAMEPACK_MANIFEST.json')
else:
    roots=[path.parent for path in exact_input_candidates('DETAIL_FRAMEPACK_MANIFEST.json')
        if digest_bytes(path.read_bytes())==DETAIL_MANIFEST_SHA]
    assert len(roots)==1,'Attach reviewed_detail_framepack.bin as one private data file, or the verified original ZIP'
    detail_root=roots[0]; inventory_payload=(detail_root/'DETAIL_FRAMEPACK_MANIFEST.json').read_bytes()
assert digest_bytes(inventory_payload)==DETAIL_MANIFEST_SHA
detail_inventory=json.loads(inventory_payload); DETAIL_CHECKSUMS=detail_inventory['members_sha256']
def read_detail_member(name):
    assert name in DETAIL_CHECKSUMS
    relative=Path(name); assert not relative.is_absolute() and '..' not in relative.parts
    if detail_archive is not None: value=detail_archive.read(name)
    else:
        path=(detail_root/name).resolve(); assert path.is_relative_to(detail_root.resolve())
        value=path.read_bytes()
    assert digest_bytes(value)==DETAIL_CHECKSUMS[name]
    return value
if detail_archive is not None:
    names=detail_archive.namelist(); assert len(names)==len(set(names))
    assert set(names)==set(DETAIL_CHECKSUMS)|{'DETAIL_FRAMEPACK_MANIFEST.json'}
assert len(DETAIL_CHECKSUMS)==3876
assert detail_inventory['recordings']==58 and detail_inventory['time_samples']==1879 and detail_inventory['jpeg_images']==3758
assert detail_inventory['image_size']==448 and detail_inventory['contains_audio'] is False
assert detail_inventory['timestamps_equal_frozen_224'] is True
records=json.loads(read_detail_member('dataset/manifest.json')); split=json.loads(read_detail_member('dataset/split.json'))
from datathon_ml.direct_vlm import _validate
records,partitions,split_provenance=_validate(records,split,WORKSPACE/'reference/clip_baseline_cv.json')
assert [len(partitions[name]) for name in ('train','validation','test')]==[34,12,12]
assert split['protocol']['type']=='camera_exclusive_source_overlap_development'
assert split['protocol']['opposite_camera_excluded_globally'] is True
all_keys={row['sha256'] for row in records}; assert len(all_keys)==58
for name in DETAIL_CHECKSUMS:
    parts=Path(name).parts
    assert name in ('dataset/manifest.json','dataset/split.json') or (
        len(parts)==3 and parts[0]=='frames' and parts[1] in all_keys and (
        parts[2] in ('metadata.json','_integrity.json') or re.fullmatch(r'[0-9]{6}_(full|middle)\\.jpg',parts[2])))
    read_detail_member(name)
for name in ('dataset/manifest.json','dataset/split.json'):
    target=WORKSPACE/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(read_detail_member(name))
assert all(not (WORKSPACE/'frames'/key).exists() for key in split['test']),'Use a fresh fitting workspace'
fit_keys=set(split['train'])|set(split['validation'])
for name in DETAIL_CHECKSUMS:
    parts=Path(name).parts
    if parts[0]=='frames' and parts[1] in fit_keys:
        target=(WORKSPACE/name).resolve(); assert target.is_relative_to(WORKSPACE.resolve())
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(read_detail_member(name))
(RUN/'detail_restore_receipt.json').write_text(json.dumps({'detail_manifest_sha256':DETAIL_MANIFEST_SHA,
    'detail_zip_sha256':DETAIL_ZIP_SHA,'binary_transport_sha256':transport_sha,
    'verified_members':len(DETAIL_CHECKSUMS),'fit_sources':46,'former_test_sources_materialized':0,
    'split_sha256':split_provenance['sha256'],'independent_test':False},indent=2))
print({'private_members_verified':len(DETAIL_CHECKSUMS),'fit_sources_materialized':46,
    'former_test_sources_materialized':0,'audio_used':False})
'''


PROBE = '''from tools.run_scene_temporal_colab import run_guarded
COMMON=['--manifest','dataset/manifest.json','--split','dataset/split.json','--detail-frame-root','frames',
    '--checkpoint',str(QWEN_CHECKPOINT),'--originals-from','reference/clip_baseline_cv.json',
    '--output-dir',str(V6_RUN),'--device',QWEN_DEVICE,'--threads','2']
run_guarded(WORKSPACE,V6_RUN,'focused_tests','unittest',
    ['discover','-s','tests','-p','test_qlora_vlm.py','-v'],600)
# This probe uses synthetic pixels only: backward, optimizer step, save/reload and uniform evaluation.
run_guarded(WORKSPACE,V6_RUN,'probe','qlora_experiment',['probe','--checkpoint',str(QWEN_CHECKPOINT),
    '--output-dir',str(V6_RUN),'--device',QWEN_DEVICE,'--threads','2','--allow-resource-fallback'],1800)
probe=json.loads((V6_RUN/'resource_probe.json').read_text())
assert probe['completed'] and probe['synthetic_probe'] and probe['private_frames_used'] is False
assert probe['adapter_reload_verified'] and probe['gradient_audit']['frozen_parameter_gradient_tensors']==0
print({'native_image_size':probe['config']['image_size'],'backward_and_reload_verified':True})
'''


FIT = '''# Two fixed schedules use34 TRAIN owners; only12 validation owners select the winner/threshold.
assert all(not (WORKSPACE/'frames'/key).exists() for key in split['test'])
run_guarded(WORKSPACE,V6_RUN,'fit','qlora_experiment',['fit',*COMMON],5400)
selection=json.loads((V6_RUN/'selection.json').read_text())
assert selection['test_frame_files_opened']==0 and selection['independent_test'] is False
print('Validation winner:',selection['winner']['candidate'])
print('Validation metrics:',selection['winner']['validation_metrics'])
'''


SCORE = '''# Release the former test only after adapter/threshold selection. It remains exposed development.
from datetime import datetime,timezone
lock=json.loads((V6_RUN/'selection_lock.json').read_text())
assert digest_bytes((V6_RUN/'model_qlora_vlm.json').read_bytes())==lock['model_sha256']
assert digest_bytes((V6_RUN/'selection.json').read_bytes())==lock['selection_sha256']
assert digest_bytes((V6_RUN/'resource_probe.json').read_bytes())==lock['resource_probe_sha256']
assert all(digest_bytes((V6_RUN/'adapter'/name).read_bytes())==expected for name,expected in lock['adapter_sha256'].items())
assert not (V6_RUN/'TEST_OPENED.json').exists(),'This branch already opened the former test'
for name in DETAIL_CHECKSUMS:
    parts=Path(name).parts
    if parts[0]=='frames' and parts[1] in set(split['test']):
        target=(WORKSPACE/name).resolve(); assert target.is_relative_to(WORKSPACE.resolve())
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(read_detail_member(name))
(V6_RUN/'development_release_receipt.json').write_text(json.dumps({'selection_lock_sha256':digest_bytes((V6_RUN/'selection_lock.json').read_bytes()),
    'model_sha256':lock['model_sha256'],'adapter_sha256':lock['adapter_sha256'],
    'detail_manifest_sha256':DETAIL_MANIFEST_SHA,'released_sources':sorted(split['test']),
    'released_at':datetime.now(timezone.utc).isoformat(),'completed':True,'independent_test':False},indent=2))
run_guarded(WORKSPACE,V6_RUN,'score','qlora_experiment',['score',*COMMON],1800)
report=json.loads((V6_RUN/'qlora_test_report.json').read_text())
assert report['independent_test'] is False and report['localization_available'] is False
assert all(not row['events'] for row in report['predictions'])
print('Exposed development metrics:',report['metrics'])
print('Domain metrics:',report['domain_metrics'])
print('This exposed12-recording comparison does not establish independent80%/87% accuracy.')
'''


INFERENCE = '''# A new query needs only a verified native detail pack and opaque source SHA, never its label.
query_key=sorted(split['train'])[0]
run_guarded(WORKSPACE,V6_RUN,'inference','qlora_experiment',['predict',
    '--detail-frame-root','frames','--source-sha256',query_key,'--checkpoint',str(QWEN_CHECKPOINT),
    '--output-dir',str(V6_RUN),'--prediction-output',str(V6_RUN/'inference/prediction.json')],1800)
prediction=json.loads((V6_RUN/'inference/prediction.json').read_text())
assert 'label' not in prediction and prediction['events']==[] and prediction['localization_available'] is False
print('Recording classification only. Cached-pack inference excludes video decoding and includes model loading.')
'''


def live_export_cell(builder_bytes):
    return '''# Export selected/candidate adapters and exact fitted code. Private pixels and base weights stay excluded.
import base64,hashlib,json,math,subprocess,sys,tempfile,zipfile
from pathlib import Path
def export_digest(value): return hashlib.sha256(value).hexdigest()
assert all(name in globals() for name in ('V6_WORKSPACE','V6_RUN'))
workspace=Path(V6_WORKSPACE); run=Path(V6_RUN)
artifact=json.loads((run/'model_qlora_vlm.json').read_text()); lock=json.loads((run/'selection_lock.json').read_text())
selection=json.loads((run/'selection.json').read_text()); probe=json.loads((run/'resource_probe.json').read_text())
report=json.loads((run/'qlora_test_report.json').read_text())
assert export_digest((run/'model_qlora_vlm.json').read_bytes())==lock['model_sha256']
assert export_digest((run/'selection.json').read_bytes())==lock['selection_sha256']
assert export_digest((run/'resource_probe.json').read_bytes())==lock['resource_probe_sha256']
assert artifact['source_sha256']==lock['source_sha256']==probe['source_sha256']==report['source_sha256']
assert probe['completed'] and probe['synthetic_probe'] and probe['private_frames_used'] is False and probe['adapter_reload_verified']
assert probe['gradient_audit']['frozen_parameter_gradient_tensors']==0
assert artifact['resource_probe_sha256']==lock['resource_probe_sha256']
assert artifact['config']==probe['config'] and artifact['model_provenance']==probe['model_provenance']==report['model_provenance']
assert selection['winner']==artifact['winner']==report['winner']
assert report['model_sha256']==lock['model_sha256'] and report['adapter_sha256']==lock['adapter_sha256']
assert json.loads((run/'FIT_STARTED.json').read_text())['completed'] is True
opened=json.loads((run/'TEST_OPENED.json').read_text()); assert opened['completed'] is True and opened['test_open_count']==1
for phase in ('focused_tests','probe','fit','score','inference'):
    runtime=json.loads((run/(phase+'_runtime.json')).read_text())
    assert runtime['completed'] and runtime['network_attempts']==0 and runtime.get('returncode',0)==0
source_paths={}
for module,expected in artifact['source_sha256'].items():
    path=workspace/f'{module}.py' if module.endswith('_experiment') else workspace/'datathon_ml'/f'{module}.py'
    assert path.is_file() and export_digest(path.read_bytes())==expected,'Fitted source changed'
    source_paths[path.relative_to(workspace).as_posix()]=expected
from datathon_ml.direct_vlm import _validate
from datathon_ml.reviewed_models import _supervision_hash
from datathon_ml.model import classification_metrics
data=json.loads((workspace/'dataset/manifest.json').read_text()); split=json.loads((workspace/'dataset/split.json').read_text())
data,partitions,provenance=_validate(data,split,workspace/'reference/clip_baseline_cv.json')
assert provenance['sha256']==lock['split_sha256']==artifact['split']['sha256']==report['split']['sha256']
assert _supervision_hash(data)==lock['all_supervision_sha256']==artifact['all_supervision_sha256']
for fold in ('training','validation','test'):
    owner='train' if fold=='training' else fold
    assert artifact[fold+'_sha256']==sorted(split[owner])
truth={row['sha256']:row for row in partitions['test']}; predictions=report['predictions']
assert len(predictions)==len(truth)==12 and {row['sha256'] for row in predictions}==set(truth)
for row in predictions:
    owner=truth[row['sha256']]
    assert row['label']==owner['label'] and row['domain']==owner.get('domain','original')
    assert math.isfinite(row['score']) and 0<=row['score']<=1
    assert row['prediction']==int(row['score']>=artifact['winner']['threshold']) and row['events']==[]
recomputed=classification_metrics([r['label'] for r in predictions],[r['score'] for r in predictions],[r['prediction'] for r in predictions])
assert recomputed==report['metrics']
for domain,metrics in report['domain_metrics'].items():
    group=[r for r in predictions if r['domain']==domain]
    assert classification_metrics([r['label'] for r in group],[r['score'] for r in group],[r['prediction'] for r in group])==metrics
assert report['independent_test'] is False and artifact['independent_test'] is False
assert report['localization_available'] is False and report['generation_performed'] is False
assert report['audio_used'] is False and report['hosted_inference_API_used'] is False
allowed_adapters={'adapter':lock['adapter_sha256']}
for candidate in selection['leaderboard']: allowed_adapters[candidate['candidate']]=candidate['adapter_sha256']
for directory,hashes in allowed_adapters.items():
    assert directory in {'adapter','lr1e-4_epochs3','lr2e-4_epochs5'}
    assert set(hashes)=={'adapter_config.json','adapter_model.safetensors'}
    for name,expected in hashes.items(): assert export_digest((run/directory/name).read_bytes())==expected
export_base=Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('/content') if Path('/content').is_dir() else Path.cwd()
merge=Path(tempfile.mkdtemp(prefix='datathon_qlora_export_source_',dir=export_base))
paths=set(workspace.glob('*.py'))
for directory in ('datathon_ml','tools','tests'): paths.update((workspace/directory).glob('*.py'))
paths.update(workspace.glob('requirements*.txt')); paths.update(workspace.glob('README*.md'))
paths.add(workspace/'reference/clip_baseline_cv.json')
source_members={p.relative_to(workspace).as_posix():p.read_bytes() for p in paths if p.is_file()}
assert 'tests/__init__.py' in source_members
source_members['tools/build_qlora_notebook.py']=base64.b64decode(''' + repr(base64.b64encode(builder_bytes).decode()) + ''')
for name,value in source_members.items():
    path=(merge/name).resolve(); assert path.is_relative_to(merge.resolve())
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(value)
notebook_path=merge/'datathon_qlora.ipynb'; bundle_path=merge/'qlora_source.zip'
subprocess.run([sys.executable,str(merge/'tools/build_qlora_notebook.py'),'--root',str(merge),
    '--notebook',str(notebook_path),'--bundle-out',str(bundle_path)],cwd=merge,check=True)
clean=json.loads(notebook_path.read_text())
assert all(c.get('execution_count') is None and c.get('outputs',[])==[] for c in clean['cells'] if c['cell_type']=='code')
members={'datathon_qlora.ipynb':notebook_path.read_bytes()}
with zipfile.ZipFile(bundle_path) as archive:
    for name in archive.namelist():
        if name.endswith('.py'): assert archive.read(name)==(merge/name).read_bytes()
        members['source/'+name]=archive.read(name)
for path in run.rglob('*'):
    if not path.is_file(): continue
    relative=path.relative_to(run)
    assert not set(relative.parts)&{'frames','dataset','features','open','base_weights'}
    if path.suffix=='.safetensors':
        assert len(relative.parts)==2 and relative.parts[0] in allowed_adapters and relative.parts[1]=='adapter_model.safetensors'
        members['runs/qlora_v6/'+relative.as_posix()]=path.read_bytes()
    elif path.suffix in {'.json','.log'}:
        members['runs/qlora_v6/'+relative.as_posix()]=path.read_bytes()
members['FITTED_SOURCE_RECEIPT.json']=json.dumps({'actual_fitted_source_sha256':source_paths,
    'actual_fitted_ML_bytes_verified':True,'split_labels_threshold_metrics_verified':True,
    'selected_adapter_sha256':lock['adapter_sha256'],'resource_probe_sha256':lock['resource_probe_sha256'],
    'tests_package_initializer_included':True,'contains_private_frames':False,'contains_base_weights':False},indent=2).encode()
assert not any(name.lower().endswith(('.jpg','.jpeg','.png','.mp4','.mov','.xlsx','.pt','.npz')) for name in members)
members['EXPORT_MANIFEST.json']=json.dumps({'contains_private_frames':False,'contains_raw_media':False,
    'contains_base_weights':False,'contains_selected_and_candidate_adapters':True,'independent_test':False,
    'members_sha256':{name:export_digest(value) for name,value in sorted(members.items())}},indent=2).encode()
QLORA_RESULTS_ZIP=export_base/'datathon_qlora_results.zip'
with zipfile.ZipFile(QLORA_RESULTS_ZIP,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
    for name,value in sorted(members.items()): archive.writestr(name,value)
with zipfile.ZipFile(QLORA_RESULTS_ZIP) as archive: assert archive.testzip() is None
print({'qlora_export_bytes':QLORA_RESULTS_ZIP.stat().st_size,'export_members':len(members),
    'qlora_export_sha256':export_digest(QLORA_RESULTS_ZIP.read_bytes()),'private_frames_exported':False})
from IPython.display import FileLink,display
display(FileLink(str(QLORA_RESULTS_ZIP)))
if not Path('/kaggle/working').is_dir() and Path('/content').is_dir():
    from google.colab import files
    files.download(str(QLORA_RESULTS_ZIP))
'''


def notebook_cells(bundle):
    intro = '''# Supervised native Qwen QLoRA video classification

This notebook uses the pinned Apache2.0 Qwen3-VL4B public weights, NF4 double
quantization, FP16 compute and rank8 q/v adapters in the final eight language
attention layers. Vision and base weights remain frozen. No hosted LLM API,
audio, generated caption or generated reasoning is used.

The synthetic GPU probe must verify backward gradients, an optimizer step,
adapter saving and fresh-base reload before private training. The initial
budget is16 training frames and32 evaluation frames at448px. Only a recorded
synthetic CUDA OOM before private fitting permits224px instead. Training alone
inserts supplied positive point anchors into otherwise uniform context frames.
Evaluation uniformly samples the whole recording without query labels/points.

Two schedules are fixed: LR0.0001 for3epochs and LR0.0002 for5epochs, accumulation4.
Training34 sources fits adapters; validation12 selects the adapter and threshold.
The former12 test sources remain physically withheld until selection. Earlier
experiments already exposed this test, so these are development comparisons,
not independent evidence for80%/87% accuracy. Native overlap groups do not imply
new-session generalization. This recording classifier does not localize events.
'''
    return [context.cell('markdown', intro), context.cell('code', bootstrap(bundle)), context.cell('code', SETUP),
        context.cell('markdown', 'Attach the separate private reviewed_detail_framepack.bin file. It preserves the exact original checksummed ZIP; pixels never enter the notebook source or exports.'),
        context.cell('code', RESTORE), context.cell('markdown', 'Ten implementation tests and the native synthetic backward/reload probe precede private losses.'),
        context.cell('code', PROBE), context.cell('markdown', 'Train the two fixed schedules and freeze validation selection.'), context.cell('code', FIT),
        context.cell('markdown', 'Release the former test after the selection lock and score this branch once as development.'), context.cell('code', SCORE),
        context.cell('markdown', 'The predict CLI also accepts a separate verified detail pack and opaque source SHA without query labels.'),
        context.cell('code', INFERENCE), context.cell('markdown', 'Export exact fitted source, a clean notebook, selected/candidate adapters and verified numeric reports/logs. Private frames and public base weights stay excluded.'),
        context.cell('code', live_export_cell(Path(__file__).read_bytes()))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--notebook', type=Path, default=ROOT/'datathon_qlora.ipynb')
    parser.add_argument('--bundle-out', type=Path, default=ROOT/'artifacts/qlora_notebook_source.zip')
    parser.add_argument('--cells-out', type=Path)
    args = parser.parse_args(argv)
    bundle, checksums = context.source_zip(collect_source(args.root))
    cells = notebook_cells(bundle)
    for value in cells:
        if value['cell_type'] == 'code':
            compile(''.join(value['source']), '<qlora-notebook-cell>', 'exec')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'colab': {'name': args.notebook.name, 'provenance': []},
        'source_snapshot_sha256': digest(bundle), 'gpu_required': True}, 'nbformat': 4, 'nbformat_minor': 5}
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    args.notebook.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    args.bundle_out.parent.mkdir(parents=True, exist_ok=True)
    args.bundle_out.write_bytes(bundle)
    if args.cells_out:
        execution = {'public_setup': SETUP, 'private_restore': RESTORE, 'probe_and_tests': PROBE,
            'fit': FIT, 'release_score': SCORE, 'inference': INFERENCE,
            'live_export': live_export_cell(Path(__file__).read_bytes())}
        for source in execution.values(): compile(source, '<qlora-execution-cell>', 'exec')
        args.cells_out.parent.mkdir(parents=True, exist_ok=True)
        args.cells_out.write_text(json.dumps(execution, indent=2), encoding='utf-8')
    print(json.dumps({'notebook': str(args.notebook), 'cells': len(cells), 'source_members': len(checksums),
        'source_bytes': len(bundle), 'source_sha256': digest(bundle), 'builder_sha256': digest(Path(__file__).read_bytes())}))


if __name__ == '__main__':
    main()

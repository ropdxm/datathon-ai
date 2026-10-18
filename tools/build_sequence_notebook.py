"""Self-contained source notebooks and provenance-preserving V3/V4 exports."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
FRAME_ZIP_SHA = "43a04bb8ae5e8cc34b29c86dc828e614019f2a513093aa5f2376c1e1cd2b4fcf"
FRAME_MANIFEST_SHA = "744a2b03b6cf4d8c4dab14bc2a019c190af3d606ce3874bd0aca1a0b3e1c4443"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def cell(kind, source):
    value = {"cell_type":kind,"id":digest((kind+source).encode())[:12],"metadata":{},"source":source.splitlines(keepends=True)}
    if kind == "code": value.update(execution_count=None,outputs=[])
    return value


def collect_source(root):
    root = Path(root)
    paths = set(root.glob('*.py'))
    for directory in ('datathon_ml','tools','tests'): paths.update((root/directory).glob('*.py'))
    paths.update(root.glob('requirements*.txt'))
    paths.update(root.glob('README*.md'))
    members = {path.relative_to(root).as_posix():path.read_bytes() for path in sorted(paths) if path.is_file()}
    required = ('temporal_video_experiment.py','direct_vlm_experiment.py','datathon_ml/temporal_video_finetune.py',
        'datathon_ml/direct_vlm.py','tests/__init__.py','tests/test_temporal_video_finetune.py','tests/test_direct_vlm.py')
    if any(name not in members for name in required): raise FileNotFoundError('Combined sequence source is incomplete')
    baseline = next((path for path in (root/'reference/clip_baseline_cv.json',
        root/'dtFinal/runs/clip_b32/evaluation/cv_report.json',root/'artifacts/colab_results/runs/clip_b32/evaluation/cv_report.json') if path.is_file()),None)
    if baseline is None: raise FileNotFoundError('Original numeric fold reference is required')
    members['reference/clip_baseline_cv.json'] = baseline.read_bytes()
    members['tools/build_sequence_notebook.py'] = Path(__file__).read_bytes()
    return members


def source_zip(members):
    checksums = {name:digest(value) for name,value in sorted(members.items())}
    files = {**members,'SOURCE_MANIFEST.json':json.dumps({'contains_private_frames':False,'contains_private_dataset':False,
        'contains_base_weights':False,'members_sha256':checksums},indent=2).encode()}
    output = io.BytesIO()
    with ZipFile(output,'w',ZIP_DEFLATED,compresslevel=9) as archive:
        for name,value in sorted(files.items()):
            item=ZipInfo(name,(2026,1,1,0,0,0)); item.compress_type=ZIP_DEFLATED; item.external_attr=0o644<<16
            archive.writestr(item,value)
    return output.getvalue(),checksums


def bootstrap(bundle):
    return '''# Always restore this exact embedded snapshot; old attached source ZIPs are ignored.
import base64,hashlib,io,json,os,subprocess,sys,zipfile
from pathlib import Path
BASE=Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('/content') if Path('/content').is_dir() else Path.cwd()
INPUT=Path('/kaggle/input') if Path('/kaggle/input').is_dir() else BASE
WORKSPACE=BASE/'datathon_sequence_workspace'
WORKSPACE.mkdir(parents=True,exist_ok=True)
def digest_bytes(value): return hashlib.sha256(value).hexdigest()
SOURCE_ZIP=base64.b64decode(''' + repr(base64.b64encode(bundle).decode()) + ''')
assert digest_bytes(SOURCE_ZIP)==''' + repr(digest(bundle)) + '''
with zipfile.ZipFile(io.BytesIO(SOURCE_ZIP)) as archive:
    inventory=json.loads(archive.read('SOURCE_MANIFEST.json'))
    SOURCE_MEMBERS=inventory['members_sha256']
    assert len(archive.namelist())==len(set(archive.namelist()))
    assert set(archive.namelist())==set(SOURCE_MEMBERS)|{'SOURCE_MANIFEST.json'}
    for name,expected in SOURCE_MEMBERS.items():
        target=(WORKSPACE/name).resolve(); assert target.is_relative_to(WORKSPACE.resolve())
        value=archive.read(name); assert digest_bytes(value)==expected
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(value)
assert (WORKSPACE/'tests/__init__.py').is_file()
os.chdir(WORKSPACE)
if str(WORKSPACE) not in sys.path: sys.path.insert(0,str(WORKSPACE))
V3_WORKSPACE=V4_WORKSPACE=WORKSPACE
V3_RUN=WORKSPACE/'runs/temporal_video_v3'; V4_RUN=WORKSPACE/'runs/direct_vlm_v4'
RUN=WORKSPACE/'runs/sequence_receipts'
for path in (V3_RUN,V4_RUN,RUN): path.mkdir(parents=True,exist_ok=True)
(RUN/'source_restore_receipt.json').write_text(json.dumps({'source_zip_sha256':digest_bytes(SOURCE_ZIP),
    'members_sha256':SOURCE_MEMBERS,'tests_package_initializer_included':True},indent=2))
print({'verified_source_members':len(SOURCE_MEMBERS),'self_contained_snapshot':True})
'''


SETUP = '''# Public checkpoint/package setup is the only network-enabled phase.
subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6',
    'open_clip_torch==3.2.0','safetensors','accelerate','-r','requirements-ml.txt'],check=True)
import torch
assert torch.cuda.is_available(),'Enable a GPU runtime'
from huggingface_hub import snapshot_download
from datathon_ml.temporal_video_finetune import MODEL_ID as X_ID,REVISION as X_REV,BASE_FILES
from datathon_ml.direct_vlm import MODEL_ID as Q_ID,REVISION as Q_REV,WEIGHT_SHA256
XCLIP_BASE=Path(globals().get('XCLIP_BASE',BASE/'xclip16_base'))
QWEN_BASE=Path(globals().get('QWEN_BASE',BASE/'qwen3vl4b_base'))
if not all((XCLIP_BASE/name).is_file() and digest_bytes((XCLIP_BASE/name).read_bytes())==expected for name,expected in BASE_FILES.items()):
    snapshot_download(X_ID,revision=X_REV,local_dir=str(XCLIP_BASE),
        allow_patterns=['config.json','preprocessor_config.json','model.safetensors','README.md'])
if not all((QWEN_BASE/name).is_file() and digest_bytes((QWEN_BASE/name).read_bytes())==expected for name,expected in WEIGHT_SHA256.items()):
    snapshot_download(Q_ID,revision=Q_REV,local_dir=str(QWEN_BASE),
        allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','README.md'])
assert all(digest_bytes((XCLIP_BASE/name).read_bytes())==expected for name,expected in BASE_FILES.items())
assert all(digest_bytes((QWEN_BASE/name).read_bytes())==expected for name,expected in WEIGHT_SHA256.items())
QWEN_DEVICE='cuda:1' if torch.cuda.device_count()>1 else 'cuda:0'
print({'public_checkpoints_verified':True,'gpu_count':torch.cuda.device_count(),
    'xclip_license':'MIT','qwen_license':'Apache-2.0','hosted_AI_API_used':False})
'''


RESTORE = '''# Private images remain native and are never rendered or sent to hosted AI.
FRAME_ZIP_SHA=''' + repr(FRAME_ZIP_SHA) + '''
FRAME_MANIFEST_SHA=''' + repr(FRAME_MANIFEST_SHA) + '''
archive_path=next((p for p in INPUT.rglob('reviewed_visual_framepack.zip') if digest_bytes(p.read_bytes())==FRAME_ZIP_SHA),None)
archive_handle=zipfile.ZipFile(archive_path) if archive_path else None
if archive_handle:
    payload=archive_handle.read('FRAMEPACK_MANIFEST.json')
else:
    candidates=[p.parent for p in INPUT.rglob('FRAMEPACK_MANIFEST.json') if digest_bytes(p.read_bytes())==FRAME_MANIFEST_SHA]
    assert len(candidates)==1,'Attach the separate private framepack input'
    private_root=candidates[0]; payload=(private_root/'FRAMEPACK_MANIFEST.json').read_bytes()
assert digest_bytes(payload)==FRAME_MANIFEST_SHA
frame_inventory=json.loads(payload); FRAME_CHECKSUMS=frame_inventory['members_sha256']
def read_private_member(name):
    assert name in FRAME_CHECKSUMS
    path=Path(name); assert not path.is_absolute() and '..' not in path.parts
    value=archive_handle.read(name) if archive_handle else (private_root/name).read_bytes()
    assert digest_bytes(value)==FRAME_CHECKSUMS[name]
    return value
if archive_handle:
    names=archive_handle.namelist(); assert len(names)==len(set(names)) and set(names)==set(FRAME_CHECKSUMS)|{'FRAMEPACK_MANIFEST.json'}
for name in FRAME_CHECKSUMS: read_private_member(name)
assert frame_inventory['recordings']==58 and frame_inventory['contains_audio'] is False
records=json.loads(read_private_member('dataset/manifest.json')); split=json.loads(read_private_member('dataset/split.json'))
from datathon_ml.reviewed_models import validate_split
validate_split(records,split)
assert [len(split[name]) for name in ('train','validation','test')]==[34,12,12]
for name in ('dataset/manifest.json','dataset/split.json'):
    target=WORKSPACE/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(read_private_member(name))
assert all(not (WORKSPACE/'frames'/key).exists() for key in split['test']),'Use a fresh sequence fit workspace'
fit_keys=set(split['train'])|set(split['validation'])
for name in FRAME_CHECKSUMS:
    parts=Path(name).parts
    if parts[0]=='frames' and parts[1] in fit_keys:
        target=WORKSPACE/name; assert target.resolve().is_relative_to(WORKSPACE.resolve())
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(read_private_member(name))
(RUN/'framepack_restore_receipt.json').write_text(json.dumps({'framepack_manifest_sha256':FRAME_MANIFEST_SHA,
    'framepack_zip_sha256':FRAME_ZIP_SHA,'verified_members':len(FRAME_CHECKSUMS),'fit_sources':46,
    'former_test_sources_in_fit_workspace':0,'independent_test':False},indent=2))
print({'private_members_verified':len(FRAME_CHECKSUMS),'fit_sources_materialized':46,'former_test_sources_materialized':0})
'''


XCLIP_PROBE_WORKER = '''import argparse,json
from pathlib import Path
import torch
from datathon_ml.temporal_video_finetune import load_video_only,configure_trainable_video,TemporalVideoMil
p=argparse.ArgumentParser(); p.add_argument('--base-dir',required=True); p.add_argument('--output-dir',required=True); a=p.parse_args()
b=load_video_only(a.base_dir); configure_trainable_video(b,1); m=TemporalVideoMil(b).to('cuda:0').train()
x=torch.zeros(1,16,3,224,224,device='cuda:0')
with torch.autocast('cuda',dtype=torch.float16): y=m(x); loss=y.sum()
loss.backward()
r={'synthetic_probe':True,'finite':bool(torch.isfinite(y).all()),'output_shape':list(y.shape),
   'frozen_prefix':b.vision_model.embeddings.patch_embedding.weight.grad is None,
   'last_vision_gradient':next(b.vision_model.encoder.layers[-1].parameters()).grad is not None,
   'temporal_position_gradient':b.mit.position_embedding.grad is not None,
   'trainable_parameters':sum(v.numel() for v in m.parameters() if v.requires_grad)}
assert r['finite'] and r['frozen_prefix'] and r['last_vision_gradient'] and r['temporal_position_gradient']
d=Path(a.output_dir); d.mkdir(parents=True,exist_ok=True); (d/'probe.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r))
'''


QWEN_INFERENCE_WORKER = '''import argparse,json
from pathlib import Path
from datathon_ml.direct_vlm import DirectVLMConfig,LocalDirectVLM,make_bundle,_sources
p=argparse.ArgumentParser(); p.add_argument('--workspace',required=True); p.add_argument('--run',required=True)
p.add_argument('--checkpoint',required=True); p.add_argument('--query-sha',required=True); p.add_argument('--output-dir',required=True); a=p.parse_args()
w=Path(a.workspace); run=Path(a.run); artifact=json.loads((run/'model_direct_vlm.json').read_text())
assert artifact['source_sha256']==_sources()
config=DirectVLMConfig(**artifact['config']); train=set(artifact['training_sha256'])
targets={row['sha256']:row for row in json.loads((w/'dataset/manifest.json').read_text()) if row['sha256'] in train}
examples=[]
if artifact['winner']['strategy']=='two_shot_train_owned':
    for receipt in artifact['demonstrations']:
        assert receipt['sha256'] in train
        owner=targets[receipt['sha256']]; bundle=make_bundle(w/'frames',owner['sha256'],config,training_demo=owner)
        assert all(bundle[key]==receipt[key] for key in ('metadata_sha256','integrity_sha256'))
        examples.append({'sha256':owner['sha256'],'label':owner['label'],'bundle':bundle})
engine=LocalDirectVLM(a.checkpoint,config); assert engine.provenance==artifact['model_provenance']
# Query receives only an opaque key and its native video tensor, never its label/anchors.
probability=engine.score(make_bundle(w/'frames',a.query_sha,config),examples)
result={'score':probability,'prediction':int(probability>=artifact['winner']['threshold']),
    'threshold':artifact['winner']['threshold'],'events':[],'localization_available':False,
    'generation_performed':False,'generated_text_retained':False}
d=Path(a.output_dir); d.mkdir(parents=True,exist_ok=True); (d/'prediction.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''


def probes_and_tests():
    return '''from tools.run_scene_temporal_colab import run_guarded
V3_COMMON=['--manifest','dataset/manifest.json','--split','dataset/split.json','--frame-root','frames',
    '--base-dir',str(XCLIP_BASE),'--originals-from','reference/clip_baseline_cv.json','--output-dir',str(V3_RUN),'--device','cuda','--threads','2']
V4_COMMON=['--manifest','dataset/manifest.json','--split','dataset/split.json','--frame-root','frames',
    '--checkpoint',str(QWEN_BASE),'--originals-from','reference/clip_baseline_cv.json','--output-dir',str(V4_RUN),'--device',QWEN_DEVICE,'--threads','2']
for pattern,name,destination in [('test_temporal_video_finetune.py','focused_tests',V3_RUN),('test_direct_vlm.py','focused_tests',V4_RUN)]:
    run_guarded(WORKSPACE,destination,name,'unittest',['discover','-s','tests','-p',pattern,'-v'],600)
(WORKSPACE/'tools/sequence_xclip_probe.py').write_text(''' + repr(XCLIP_PROBE_WORKER) + ''')
run_guarded(WORKSPACE,V3_RUN,'probe','tools.sequence_xclip_probe',['--base-dir',str(XCLIP_BASE),'--output-dir',str(V3_RUN/'probe')],600)
run_guarded(WORKSPACE,V4_RUN,'probe','direct_vlm_experiment',['probe','--checkpoint',str(QWEN_BASE),
    '--device',QWEN_DEVICE,'--output-dir',str(V4_RUN/'probe')],900)
'''


FIT_V3 = '''assert all(not (WORKSPACE/'frames'/key).exists() for key in split['test'])
run_guarded(WORKSPACE,V3_RUN,'fit','temporal_video_experiment',['fit',*V3_COMMON],3600)
v3selection=json.loads((V3_RUN/'temporal_video_selection.json').read_text())
V3_MODEL=V3_RUN/v3selection['winner']['artifact']
assert v3selection['development_partition_opened_for_selection'] is False
print('V3 validation winner:',v3selection['winner']['candidate'])
print('V3 validation metrics:',v3selection['winner']['validation_metrics'])
'''

FIT_V4 = '''assert all(not (WORKSPACE/'frames'/key).exists() for key in split['test'])
run_guarded(WORKSPACE,V4_RUN,'fit','direct_vlm_experiment',['fit',*V4_COMMON],3600)
v4selection=json.loads((V4_RUN/'selection.json').read_text())
assert v4selection['test_frame_files_opened']==0
print('V4 validation winner:',v4selection['winner']['strategy'])
print('V4 validation metrics:',v4selection['winner']['validation_metrics'])
'''

RELEASE_SCORE = '''# Both model families must finish validation selection before any release.
v3lock=json.loads((V3_RUN/'temporal_video_selection_lock.json').read_text())
v4lock=json.loads((V4_RUN/'selection_lock.json').read_text())
assert digest_bytes(V3_MODEL.read_bytes())==v3lock['winner_sha256']
assert digest_bytes((V3_RUN/'temporal_video_selection.json').read_bytes())==v3lock['selection_sha256']
assert digest_bytes((V4_RUN/'model_direct_vlm.json').read_bytes())==v4lock['model_sha256']
assert digest_bytes((V4_RUN/'selection.json').read_bytes())==v4lock['selection_sha256']
for name in FRAME_CHECKSUMS:
    parts=Path(name).parts
    if parts[0]=='frames' and parts[1] in set(split['test']):
        target=WORKSPACE/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(read_private_member(name))
(RUN/'development_release_receipt.json').write_text(json.dumps({'both_validation_locks_verified':True,
    'released_sources':sorted(split['test']),'independent_test':False},indent=2))
run_guarded(WORKSPACE,V3_RUN,'score','temporal_video_experiment',['score',*V3_COMMON,'--model',str(V3_MODEL)],1800)
run_guarded(WORKSPACE,V4_RUN,'score','direct_vlm_experiment',['score',*V4_COMMON],3600)
v3report=json.loads((V3_RUN/'temporal_video_development_report.json').read_text())
v4report=json.loads((V4_RUN/'direct_vlm_test_report.json').read_text())
assert v3report['independent_test'] is False and v4report['independent_test'] is False
assert v4report['localization_available'] is False and all(not row['events'] for row in v4report['predictions'])
print('V3 exposed development metrics:',v3report['metrics'])
print('V4 exposed development metrics:',v4report['metrics'])
print('These results cannot establish unseen-session80%/87% accuracy.')
'''


def inference_cell():
    return '''query_key=sorted(split['train'])[0]
run_guarded(WORKSPACE,V3_RUN,'inference','temporal_video_experiment',['predict','--frame-root','frames',
    '--source-sha256',query_key,'--base-dir',str(XCLIP_BASE),'--model',str(V3_MODEL),
    '--output-dir',str(V3_RUN/'inference'),'--device','cuda'],900)
(WORKSPACE/'tools/sequence_qwen_inference.py').write_text(''' + repr(QWEN_INFERENCE_WORKER) + ''')
run_guarded(WORKSPACE,V4_RUN,'inference','tools.sequence_qwen_inference',['--workspace',str(WORKSPACE),
    '--run',str(V4_RUN),'--checkpoint',str(QWEN_BASE),'--query-sha',query_key,'--output-dir',str(V4_RUN/'inference')],900)
qwen_inference=json.loads((V4_RUN/'inference/prediction.json').read_text())
assert qwen_inference['localization_available'] is False and qwen_inference['events']==[]
print('Native framepack inference verified; original video decoding is excluded from these timings.')
'''


def live_export_cell(builder_bytes):
    return '''# Works with either one notebook workspace or the live disjoint V3/V4 workspaces.
import base64,hashlib,json,sys,subprocess,zipfile
from pathlib import Path
def export_digest(value): return hashlib.sha256(value).hexdigest()
assert all(name in globals() for name in ('V3_WORKSPACE','V3_RUN','V4_WORKSPACE','V4_RUN'))
branches={'v3':(Path(V3_WORKSPACE),Path(V3_RUN)),'v4':(Path(V4_WORKSPACE),Path(V4_RUN))}
export_base=Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('/content') if Path('/content').is_dir() else Path.cwd()
merge=export_base/'datathon_sequence_export_source'; merge.mkdir(parents=True,exist_ok=True)
source_members={}; origins={}; fitted_receipts={}
for branch,(workspace,run) in branches.items():
    selection_file='temporal_video_selection.json' if branch=='v3' else 'model_direct_vlm.json'
    artifact=json.loads((run/selection_file).read_text())
    fitted=artifact['source_sha256']; fitted_receipts[branch]={'source_sha256':fitted,'verified':True}
    for module,expected in fitted.items():
        path=workspace/'datathon_ml'/f'{module}.py'
        assert path.is_file() and export_digest(path.read_bytes())==expected, 'Actual fitted ML source changed'
    paths=set(workspace.glob('*.py'))
    for directory in ('datathon_ml','tools','tests'): paths.update((workspace/directory).glob('*.py'))
    paths.update(workspace.glob('requirements*.txt')); paths.update(workspace.glob('README*.md'))
    paths.add(workspace/'reference/clip_baseline_cv.json')
    for path in paths:
        if not path.is_file(): continue
        name=path.relative_to(workspace).as_posix(); value=path.read_bytes()
        if name in source_members and source_members[name]!=value:
            assert path.suffix in {'.md','.txt'}, 'Disjoint workspaces disagree on source code bytes'
            origins.setdefault(name,[]).append({'branch':branch,'sha256':export_digest(value),'documentation_difference':True})
            continue
        source_members[name]=value; origins.setdefault(name,[]).append({'branch':branch,'sha256':export_digest(value)})
assert 'tests/__init__.py' in source_members
source_members['tools/build_sequence_notebook.py']=base64.b64decode(''' + repr(base64.b64encode(builder_bytes).decode()) + ''')
for name,value in source_members.items():
    path=(merge/name).resolve(); assert path.is_relative_to(merge.resolve())
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(value)
notebook_path=merge/'datathon_sequence.ipynb'; source_zip_path=merge/'sequence_source.zip'
subprocess.run([sys.executable,str(merge/'tools/build_sequence_notebook.py'),'--root',str(merge),
    '--notebook',str(notebook_path),'--bundle-out',str(source_zip_path)],cwd=merge,check=True)
clean=json.loads(notebook_path.read_text())
assert all(c.get('execution_count') is None and c.get('outputs',[])==[] for c in clean['cells'] if c['cell_type']=='code')
members={'datathon_sequence.ipynb':notebook_path.read_bytes()}
with zipfile.ZipFile(source_zip_path) as archive:
    for name in archive.namelist():
        if name.endswith('.py'): assert archive.read(name)==(merge/name).read_bytes()
        members['source/'+name]=archive.read(name)
for branch,(workspace,run) in branches.items():
    for path in run.rglob('*'):
        if not path.is_file() or path.suffix not in {'.pt','.json','.log'}: continue
        relative=path.relative_to(run)
        assert not set(relative.parts)&{'frames','dataset','features','open','base_weights'}
        members[f'runs/{branch}/'+relative.as_posix()]=path.read_bytes()
members['DISJOINT_SOURCE_RECEIPT.json']=json.dumps({'fitted_receipts':fitted_receipts,'source_origins':origins,
    'tests_package_initializer_included':True,'actual_fitted_ML_bytes_verified':True,
    'contains_private_frames':False,'contains_base_weights':False},indent=2).encode()
assert not any(name.lower().endswith(('.jpg','.jpeg','.png','.mp4','.mov','.xlsx','.safetensors')) for name in members)
members['EXPORT_MANIFEST.json']=json.dumps({'contains_private_frames':False,'contains_raw_media':False,
    'contains_base_weights':False,'independent_test':False,
    'members_sha256':{name:export_digest(value) for name,value in sorted(members.items())}},indent=2).encode()
SEQUENCE_RESULTS_ZIP=export_base/'datathon_sequence_results.zip'
with zipfile.ZipFile(SEQUENCE_RESULTS_ZIP,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
    for name,value in sorted(members.items()): archive.writestr(name,value)
with zipfile.ZipFile(SEQUENCE_RESULTS_ZIP) as archive: assert archive.testzip() is None
print({'sequence_export_bytes':SEQUENCE_RESULTS_ZIP.stat().st_size,'export_members':len(members),
    'sequence_export_sha256':export_digest(SEQUENCE_RESULTS_ZIP.read_bytes()),'private_frames_exported':False})
from IPython.display import FileLink,display
display(FileLink(str(SEQUENCE_RESULTS_ZIP)))
if not Path('/kaggle/working').is_dir() and Path('/content').is_dir():
    from google.colab import files
    files.download(str(SEQUENCE_RESULTS_ZIP))
'''


def notebook_cells(bundle):
    intro='''# Native temporal and direct-video development experiments

The source snapshot includes both X-CLIP temporal fine tuning and native Qwen
A/B video scoring. Public base checkpoints are downloaded during setup only.
All private pixels remain in the native GPU workspace. No hosted LLM APIs,
audio, private descriptions or text generation are used.

The fixed58-source study uses34 training recordings,12 validation recordings,
and12 recordings already examined by earlier models. Those12 are exposed
development data, not an independent test. Both validation winners must be
frozen before their frames enter the fitting workspace. Qwen's recording-level
scores provide no event timestamps. Neither80% nor87% is assumed or promised.
'''
    builder=Path(__file__).read_bytes()
    return [cell('markdown',intro),cell('code',bootstrap(bundle)),cell('code',SETUP),
        cell('markdown','Verify the separate private input and restore only train/validation frames.'),cell('code',RESTORE),
        cell('markdown','Run local-package tests and synthetic native GPU probes before any private fit.'),cell('code',probes_and_tests()),
        cell('markdown','V3: two fixed supervised temporal fits; model/threshold selection uses validation only.'),cell('code',FIT_V3),
        cell('markdown','V4: compare the fixed zero-shot and train-owned two-shot strategies on validation only.'),cell('code',FIT_V4),
        cell('markdown','Release the already exposed development recordings after both validation locks exist.'),cell('code',RELEASE_SCORE),
        cell('code',inference_cell()),cell('markdown','Export exact fitted sources, model deltas/decision artifacts, numeric reports and a clean notebook. Private frames and public base weights stay excluded.'),
        cell('code',live_export_cell(builder))]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--notebook',type=Path,default=ROOT/'datathon_sequence.ipynb')
    parser.add_argument('--bundle-out',type=Path,default=ROOT/'artifacts/sequence_notebook_source.zip')
    parser.add_argument('--cells-out',type=Path)
    args=parser.parse_args(argv)
    members=collect_source(args.root); bundle,checksums=source_zip(members); cells=notebook_cells(bundle)
    for value in cells:
        if value['cell_type']=='code': compile(''.join(value['source']),'<sequence-notebook-cell>','exec')
    value={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'colab':{'name':args.notebook.name,'provenance':[]},
        'source_snapshot_sha256':digest(bundle),'gpu_required':True},'nbformat':4,'nbformat_minor':5}
    args.notebook.parent.mkdir(parents=True,exist_ok=True); args.notebook.write_text(json.dumps(value,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    args.bundle_out.parent.mkdir(parents=True,exist_ok=True); args.bundle_out.write_bytes(bundle)
    if args.cells_out:
        execution={'public_setup':SETUP,'private_restore':RESTORE,'probes_and_tests':probes_and_tests(),
            'fit_v3':FIT_V3,'fit_v4':FIT_V4,'release_score':RELEASE_SCORE,'inference':inference_cell(),
            'live_export':live_export_cell(Path(__file__).read_bytes())}
        for source in execution.values(): compile(source,'<sequence-execution-cell>','exec')
        args.cells_out.parent.mkdir(parents=True,exist_ok=True); args.cells_out.write_text(json.dumps(execution,indent=2))
    print(json.dumps({'notebook':str(args.notebook),'cells':len(cells),'source_members':len(checksums),
        'source_bytes':len(bundle),'source_sha256':digest(bundle),'builder_sha256':digest(Path(__file__).read_bytes())}))


if __name__=='__main__': main()

"""Build a clean native Kaggle/Colab visual fine-tuning notebook.

Private frame packs are supplied separately and are never embedded or exported.
The source snapshot keeps the exact fitted ML bytes while adding this builder.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "ecd6ed18cad9d6a201d0ce9951281249597b8521b4f362c9a936555bc49d321f"
FRAME_SHA = "43a04bb8ae5e8cc34b29c86dc828e614019f2a513093aa5f2376c1e1cd2b4fcf"
MODEL_SHA = "a1bff5b688873ce11b8ee668fb9648d92f5276b082d29a5361f1d181777dd7ce"


def make_cell(kind, source):
    value = {"cell_type": kind, "id": hashlib.sha256((kind + source).encode()).hexdigest()[:12],
             "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def bootstrap_cell(source_members, embedded_bundle=None, builder_bytes=None):
    embedded = base64.b64encode(embedded_bundle).decode() if embedded_bundle is not None else None
    builder64 = base64.b64encode(builder_bytes).decode() if builder_bytes is not None else None
    return '''# Restore exact source only; no private frame decoding or AI API calls.
import hashlib, io, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
BASE = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('/content') if Path('/content').is_dir() else Path.cwd()
INPUT = Path('/kaggle/input') if Path('/kaggle/input').is_dir() else BASE
WORKSPACE = BASE/'datathon_visual_workspace'
WORKSPACE.mkdir(parents=True, exist_ok=True)
SOURCE_MEMBERS = ''' + repr(source_members) + '''
def digest_bytes(value): return hashlib.sha256(value).hexdigest()
def source_root_valid(directory):
    return all((directory/name).is_file() and digest_bytes((directory/name).read_bytes()) == digest
        for name,digest in SOURCE_MEMBERS.items())
roots = [path.parent for path in INPUT.rglob('finetune_experiment.py')
    if source_root_valid(path.parent)]
source_archive = next((path for path in INPUT.rglob('visual_finetune_source.zip')
    if digest_bytes(path.read_bytes()) == ''' + repr(SOURCE_SHA) + '''), None)
if roots:
    assert len(roots) == 1, 'Multiple source roots: select one exact private input'
    for name,digest in SOURCE_MEMBERS.items():
        target = WORKSPACE/name
        assert target.resolve().is_relative_to(WORKSPACE.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((roots[0]/name).read_bytes())
else:
    embedded = ''' + repr(embedded) + '''
    assert source_archive is not None or embedded, 'Attach the private source input, or use the self-contained notebook'
    payload = source_archive.read_bytes() if source_archive is not None else __import__('base64').b64decode(embedded)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name,digest in SOURCE_MEMBERS.items():
            value = archive.read(name)
            assert digest_bytes(value) == digest
            target = WORKSPACE/name
            assert target.resolve().is_relative_to(WORKSPACE.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
builder64 = ''' + repr(builder64) + '''
if builder64:
    target = WORKSPACE/'tools/build_visual_notebook.py'
    target.write_bytes(__import__('base64').b64decode(builder64))
assert source_root_valid(WORKSPACE)
assert digest_bytes((WORKSPACE/'datathon_ml/finetune_visual.py').read_bytes()) == ''' + repr(MODEL_SHA) + '''
os.chdir(WORKSPACE)
if str(WORKSPACE) not in sys.path: sys.path.insert(0, str(WORKSPACE))
RUN = WORKSPACE/'runs/visual_finetune_v2'
RUN.mkdir(parents=True, exist_ok=True)
(RUN/'source_restore_receipt.json').write_text(json.dumps({'verified_source_members':len(SOURCE_MEMBERS),
    'original_uploaded_source_zip_sha256':''' + repr(SOURCE_SHA) + ''', 'finetune_module_sha256':''' + repr(MODEL_SHA) + '''},indent=2))
print({'verified_source_files':len(SOURCE_MEMBERS),'workspace_ready':True})
'''


def frame_restore_cell(frame_manifest_sha):
    return '''# Verify every private member; materialize train/validation frames only.
FRAME_MANIFEST_SHA = ''' + repr(frame_manifest_sha) + '''
FRAME_ZIP_SHA = ''' + repr(FRAME_SHA) + '''
FRAME_ARCHIVE = next((path for path in INPUT.rglob('reviewed_visual_framepack.zip')
    if digest_bytes(path.read_bytes()) == FRAME_ZIP_SHA), None)
extracted = [path.parent for path in INPUT.rglob('FRAMEPACK_MANIFEST.json')
    if digest_bytes(path.read_bytes()) == FRAME_MANIFEST_SHA]
if FRAME_ARCHIVE is not None:
    FRAME_ZIP_HANDLE=zipfile.ZipFile(FRAME_ARCHIVE)
    manifest_payload=FRAME_ZIP_HANDLE.read('FRAMEPACK_MANIFEST.json')
elif extracted:
    assert len(extracted)==1, 'Multiple matching private framepack roots'
    FRAME_INPUT_ROOT = extracted[0]
    manifest_payload = (FRAME_INPUT_ROOT/'FRAMEPACK_MANIFEST.json').read_bytes()
else:
    raise FileNotFoundError('Attach the separate private reviewed_visual_framepack.zip input')
assert digest_bytes(manifest_payload) == FRAME_MANIFEST_SHA
FRAME_INVENTORY = json.loads(manifest_payload)
FRAME_CHECKSUMS = FRAME_INVENTORY['members_sha256']
assert FRAME_INVENTORY['recordings']==58 and FRAME_INVENTORY['time_samples']==1879
assert FRAME_INVENTORY['jpeg_images']==3758 and FRAME_INVENTORY['contains_audio'] is False
if FRAME_ARCHIVE is not None:
    names=FRAME_ZIP_HANDLE.namelist()
    assert len(names)==len(set(names)) and set(names)==set(FRAME_CHECKSUMS)|{'FRAMEPACK_MANIFEST.json'}
def read_private_member(name):
    assert name in FRAME_CHECKSUMS
    parts=Path(name).parts
    assert not Path(name).is_absolute() and '..' not in parts
    assert name in {'dataset/manifest.json','dataset/split.json'} or (
        len(parts)==3 and parts[0]=='frames' and len(parts[1])==64 and
        (parts[2].endswith('.jpg') or parts[2] in {'metadata.json','_integrity.json'}))
    if FRAME_ARCHIVE is not None:
        value=FRAME_ZIP_HANDLE.read(name)
    else:
        path=(FRAME_INPUT_ROOT/name).resolve()
        assert path.is_relative_to(FRAME_INPUT_ROOT.resolve())
        value=path.read_bytes()
    assert digest_bytes(value)==FRAME_CHECKSUMS[name], 'Private framepack member checksum mismatch'
    return value
# Read bytes for checksum verification only; no images, captions or audio enter outputs.
for name in FRAME_CHECKSUMS: read_private_member(name)
records=json.loads(read_private_member('dataset/manifest.json'))
split=json.loads(read_private_member('dataset/split.json'))
assert len(records)==len({row['sha256'] for row in records})==58
assert [len(split[name]) for name in ('train','validation','test')]==[34,12,12]
from datathon_ml.reviewed_models import validate_split
validate_split(records,split)
assert split['protocol']['type']=='camera_exclusive_source_overlap_development'
for name in ('dataset/manifest.json','dataset/split.json'):
    target=WORKSPACE/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(read_private_member(name))
assert all(not (WORKSPACE/'frames'/key).exists() for key in split['test']), 'Use a fresh fit workspace; former-test frames are already materialized'
fit_keys=set(split['train'])|set(split['validation'])
for name in FRAME_CHECKSUMS:
    parts=Path(name).parts
    if parts[0]=='frames' and parts[1] in fit_keys:
        target=WORKSPACE/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(read_private_member(name))
(RUN/'framepack_restore_receipt.json').write_text(json.dumps({'framepack_zip_sha256':FRAME_ZIP_SHA,
    'framepack_manifest_sha256':FRAME_MANIFEST_SHA,'verified_member_count':len(FRAME_CHECKSUMS),
    'records':58,'train':34,'validation':12,'exposed_development':12,
    'former_test_frame_directories_in_fit_workspace':sum((WORKSPACE/'frames'/key).exists() for key in split['test']),
    'former_test_is_independent':False,'audio_used':False},indent=2))
print({'verified_private_members':len(FRAME_CHECKSUMS),'fit_framepacks':len(fit_keys),'former_test_materialized':0})
'''


SETUP = '''# Public setup only. A previously downloaded official checkpoint is reused.
from importlib.metadata import version, PackageNotFoundError
try: ready=version('open_clip_torch')=='3.2.0'
except PackageNotFoundError: ready=False
if not ready:
    subprocess.run([sys.executable,'-m','pip','install','-q','open_clip_torch==3.2.0',
        '-r',str(WORKSPACE/'requirements-ml.txt')],check=True)
import torch
assert torch.cuda.is_available(), 'Enable a GPU runtime before this visual fine-tuning experiment'
from datathon_ml.clip_features import ClipFeatureConfig,_resolve_clip_weights,EXPECTED_WEIGHT_SHA256
available=[globals().get('BASE_WEIGHTS'),globals().get('CHECKPOINT'),BASE/'clip_openai_ViT-B-32.pt']
available=[Path(path) for path in available if path is not None and Path(path).is_file()
    and digest_bytes(Path(path).read_bytes())==EXPECTED_WEIGHT_SHA256]
CHECKPOINT=available[0] if available else _resolve_clip_weights(ClipFeatureConfig(allow_download=True),torch)
assert digest_bytes(CHECKPOINT.read_bytes())==EXPECTED_WEIGHT_SHA256
print({'gpu_count':torch.cuda.device_count(),'gpu_name':torch.cuda.get_device_name(0),
    'base_checkpoint_verified':True,'checkpoint_bytes':CHECKPOINT.stat().st_size})
'''


GUARDED = '''# All subsequent subprocesses block network access, including inference.
from tools.run_scene_temporal_colab import run_guarded
COMMON=['--manifest','dataset/manifest.json','--split','dataset/split.json',
    '--frame-root','frames','--checkpoint',str(CHECKPOINT),
    '--originals-from','reference/clip_baseline_cv.json','--output-dir',str(RUN),
    '--device','cuda','--threads','2']
run_guarded(WORKSPACE,RUN,'focused_tests','unittest',
    ['discover','-s','tests','-p','test_finetune_visual.py','-v'],600)
'''


FIT = '''# Two predeclared visual fits; no former-test frames in this fit workspace.
assert all(not (WORKSPACE/'frames'/key).exists() for key in split['test'])
run_guarded(WORKSPACE,RUN,'fit','finetune_experiment',['fit',*COMMON],3600)
selection=json.loads((RUN/'visual_selection.json').read_text())
selection_lock=json.loads((RUN/'visual_selection_lock.json').read_text())
BEST_MODEL=RUN/selection['winner']['artifact']
assert digest_bytes(BEST_MODEL.read_bytes())==selection_lock['winner_sha256']
assert digest_bytes((RUN/'visual_selection.json').read_bytes())==selection_lock['selection_sha256']
assert selection['development_partition_opened_for_selection'] is False
print('Frozen validation-selected visual candidate:',selection['winner']['candidate'])
print('Validation metrics:',json.dumps(selection['winner']['validation_metrics'],indent=2))
'''


SCORE = '''# Release the formerly exposed development partition only after selection.
selection_lock=json.loads((RUN/'visual_selection_lock.json').read_text())
assert digest_bytes(BEST_MODEL.read_bytes())==selection_lock['winner_sha256']
assert digest_bytes((RUN/'visual_selection.json').read_bytes())==selection_lock['selection_sha256']
for name in FRAME_CHECKSUMS:
    parts=Path(name).parts
    if parts[0]=='frames' and parts[1] in set(split['test']):
        target=WORKSPACE/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(read_private_member(name))
(RUN/'development_frame_release.json').write_text(json.dumps({'released_sources':sorted(split['test']),
    'selection_lock_sha256':digest_bytes((RUN/'visual_selection_lock.json').read_bytes()),
    'winner_sha256':selection_lock['winner_sha256'],'independent_test':False},indent=2))
run_guarded(WORKSPACE,RUN,'score','finetune_experiment',['score',*COMMON,'--model',str(BEST_MODEL)],1800)
report=json.loads((RUN/'visual_development_report.json').read_text())
assert report['independent_test'] is False
print('Previously exposed mixed development metrics:',json.dumps(report['metrics'],indent=2))
print('Per-domain metrics:',json.dumps(report['domain_metrics'],indent=2))
print('Development accuracy >80%:',report['metrics']['accuracy']>.80)
print('Development accuracy >=87%:',report['metrics']['accuracy']>=.87)
print('These scores do not establish accuracy on unseen filming sessions.')
'''


PREDICT = '''# Native frame-pack prediction receives no target labels or point annotations.
source_sha=sorted(split['train'])[0]
run_guarded(WORKSPACE,RUN,'inference','finetune_experiment',['predict',
    '--frame-root','frames','--source-sha256',source_sha,'--checkpoint',str(CHECKPOINT),
    '--model',str(BEST_MODEL),'--output-dir',str(RUN/'inference'),'--device','cuda'],600)
print('Fresh-process frame-pack inference includes checkpoint loading, excludes original video decoding.')
'''


SUMMARY = '''# Independently recompute classification/localization and verify saved ownership.
from datathon_ml.model import classification_metrics,_localization_metrics
rows=report['predictions']
assert {row['sha256'] for row in rows}==set(split['test']) and len(rows)==len(split['test'])
targets={row['sha256']:row for row in records}
for row in rows:
    expected=targets[row['sha256']]
    assert row['label']==expected['label'] and row['event_times']==expected['event_times']
    assert row['group_id']==expected['group_id']
recomputed=classification_metrics([r['label'] for r in rows],[r['score'] for r in rows],[r['prediction'] for r in rows])
for key,value in recomputed.items():
    if isinstance(value,float): assert abs(value-report['metrics'][key])<1e-10
    else: assert value==report['metrics'][key]
assert _localization_metrics(rows)==report['metrics']['localization']
assert report['model_sha256']==digest_bytes(BEST_MODEL.read_bytes())
for row in selection['leaderboard']:
    artifact=torch.load(RUN/row['artifact'],map_location='cpu',weights_only=True)
    assert set(artifact['training_sha256'])==set(split['train'])
    assert not set(artifact['frame_pack_metadata_sha256'])&set(split['test'])
    assert artifact['source_sha256']['finetune_visual']==''' + repr(MODEL_SHA) + '''
runtimes={name:json.loads((RUN/f'{name}_runtime.json').read_text()) for name in ('focused_tests','fit','score','inference')}
assert all(value['completed'] and value['network_attempts']==0 for value in runtimes.values())
summary={'evaluation':report['evaluation'],'independent_test':False,'metrics':report['metrics'],
    'domain_metrics':report['domain_metrics'],'validation_metrics':selection['winner']['validation_metrics'],
    'selected_candidate':selection['winner']['candidate'],'threshold':selection['winner']['threshold'],
    'candidate_count':len(selection['leaderboard']),'training_recordings':34,'validation_recordings':12,
    'previously_exposed_development_recordings':12,'source_sha256':selection['source_sha256'],
    'runtimes':runtimes,'labels_groups_predictions_metrics_and_artifact_verified':True,
    'new_session_accuracy_established':False,'hosted_llm_api_used':False,'audio_used':False}
(RUN/'visual_experiment_summary.json').write_text(json.dumps(summary,indent=2))
print('Verified visual development accuracy:',summary['metrics']['accuracy'])
'''


def export_cell(builder_bytes):
    return '''# Export model deltas, numeric reports, source and a clean notebook; no frames.
builder=WORKSPACE/'tools/build_visual_notebook.py'
if not builder.is_file(): builder.write_bytes(__import__('base64').b64decode(''' + repr(base64.b64encode(builder_bytes).decode()) + '''))
source_input=RUN/'source_snapshot_input.zip'
with zipfile.ZipFile(source_input,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for name,digest in SOURCE_MEMBERS.items():
        value=(WORKSPACE/name).read_bytes(); assert digest_bytes(value)==digest
        archive.writestr(name,value)
subprocess.run([sys.executable,str(builder),'--source-zip',str(source_input),
    '--frame-manifest-sha',FRAME_MANIFEST_SHA,'--notebook',str(WORKSPACE/'datathon_visual.ipynb'),
    '--bundle-out',str(RUN/'export_source.zip')],cwd=WORKSPACE,check=True)
notebook=json.loads((WORKSPACE/'datathon_visual.ipynb').read_text())
assert all(cell.get('outputs',[])==[] and cell.get('execution_count') is None for cell in notebook['cells'] if cell['cell_type']=='code')
members={'datathon_visual.ipynb':(WORKSPACE/'datathon_visual.ipynb').read_bytes()}
with zipfile.ZipFile(RUN/'export_source.zip') as archive:
    for name in archive.namelist():
        if name.endswith('.py'): assert archive.read(name)==(WORKSPACE/name).read_bytes()
        members['source/'+name]=archive.read(name)
for path in RUN.rglob('*'):
    if path.is_file() and path.suffix in {'.pt','.json','.log'}:
        relative=path.relative_to(WORKSPACE)
        assert not set(relative.parts)&{'dataset','frames','features','open'}
        members[relative.as_posix()]=path.read_bytes()
assert not any(name.endswith(('.jpg','.jpeg','.png','.mp4','.mov','.xlsx')) for name in members)
members['EXPORT_MANIFEST.json']=json.dumps({'contains_private_frames':False,'contains_raw_media':False,
    'contains_base_clip_checkpoint':False,'contains_supervised_visual_parameter_deltas':True,
    'members_sha256':{name:digest_bytes(value) for name,value in sorted(members.items())}},indent=2).encode()
RESULTS_ZIP=BASE/'datathon_visual_results.zip'
with zipfile.ZipFile(RESULTS_ZIP,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
    for name,value in sorted(members.items()): archive.writestr(name,value)
print({'export_bytes':RESULTS_ZIP.stat().st_size,'export_sha256':digest_bytes(RESULTS_ZIP.read_bytes()),
    'export_members':len(members),'contains_private_frames':False})
from IPython.display import FileLink,display
display(FileLink(str(RESULTS_ZIP)))
if not Path('/kaggle/working').is_dir() and Path('/content').is_dir():
    from google.colab import files
    files.download(str(RESULTS_ZIP))
'''


def build_source(source_zip):
    with ZipFile(source_zip) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):
            raise ValueError("Duplicate source archive members")
        members={name:archive.read(name) for name in names if name!="SOURCE_MANIFEST.json"}
    if hashlib.sha256(members["datathon_ml/finetune_visual.py"]).hexdigest()!=MODEL_SHA:
        raise ValueError("The ML module differs from the frozen visual experiment")
    for name in members:
        path=Path(name)
        if path.is_absolute() or ".." in path.parts or path.suffix not in {".py",".txt",".json",".md"}:
            raise ValueError("Source snapshot contains an unexpected member")
    members["tools/build_visual_notebook.py"]=Path(__file__).read_bytes()
    checksums={name:hashlib.sha256(value).hexdigest() for name,value in sorted(members.items())}
    members["SOURCE_MANIFEST.json"]=json.dumps({"contains_frames":False,"contains_private_dataset":False,
        "original_uploaded_source_zip_sha256":SOURCE_SHA,"members_sha256":checksums},indent=2).encode()
    buffer=io.BytesIO()
    with ZipFile(buffer,"w",ZIP_DEFLATED,compresslevel=9) as archive:
        for name,value in sorted(members.items()):
            item=ZipInfo(name,(2026,1,1,0,0,0)); item.compress_type=ZIP_DEFLATED; item.external_attr=0o644<<16
            archive.writestr(item,value)
    return buffer.getvalue(),checksums


def notebook_cells(bundle,checksums,frame_manifest_sha):
    intro='''# Supervised visual fine tuning in a native GPU workspace

This experiment updates the final one/two official CLIP vision blocks and a
small point-guided MIL head. Fixed fits train on34 recordings;12 validation
recordings select the model and threshold. The previous12-recording mixed test
has already been examined and is explicitly development data here.

Use a private Kaggle input containing the separate source and framepack ZIPs, or
upload the private framepack in Colab. Enable a GPU. All1879 sampled times and
3758 RGB224 images stay in native Python; no descriptions, LLM APIs or audio are
used. The opposite custom camera family remains excluded everywhere. These
within-filming-batch results cannot establish unseen-session80%/87% accuracy.
'''
    cells=[make_cell("markdown",intro),
        make_cell("code",bootstrap_cell(checksums,bundle)),make_cell("code",SETUP),
        make_cell("markdown","Verify private input checksums and restore only train/validation images. Do not publish or preview the private framepack."),
        make_cell("code",frame_restore_cell(frame_manifest_sha)),make_cell("code",GUARDED),
        make_cell("markdown","Run two fixed supervised visual fits. Only training pixels/anchors update weights; validation alone selects the final candidate and threshold."),
        make_cell("code",FIT),make_cell("markdown","Score the already exposed development partition after the validation winner is saved and hashed."),
        make_cell("code",SCORE),make_cell("code",PREDICT),make_cell("code",SUMMARY),
        make_cell("markdown","Export exact source, fine-tuned parameter deltas, numeric reports and a clean notebook. Raw frames and the public base checkpoint are excluded."),
        make_cell("code",export_cell(Path(__file__).read_bytes()))]
    for cell in cells:
        if cell["cell_type"]=="code": compile("".join(cell["source"]),"<visual-notebook-cell>","exec")
    return cells


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip",type=Path,default=ROOT/'artifacts/visual_finetune_source.zip')
    parser.add_argument("--frame-zip",type=Path,default=ROOT/'artifacts/reviewed_visual_framepack.zip')
    parser.add_argument("--frame-manifest-sha")
    parser.add_argument("--notebook",type=Path,default=ROOT/'datathon_visual.ipynb')
    parser.add_argument("--bundle-out",type=Path,default=ROOT/'artifacts/visual_notebook_source.zip')
    parser.add_argument("--cells-out",type=Path)
    args=parser.parse_args(argv)
    if args.frame_manifest_sha:
        frame_manifest_sha=args.frame_manifest_sha
    else:
        with ZipFile(args.frame_zip) as archive:
            frame_manifest_sha=hashlib.sha256(archive.read('FRAMEPACK_MANIFEST.json')).hexdigest()
    bundle,checksums=build_source(args.source_zip)
    cells=notebook_cells(bundle,checksums,frame_manifest_sha)
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python3","language":"python","name":"python3"},
        "language_info":{"name":"python"},"colab":{"name":args.notebook.name,"provenance":[]},
        "source_snapshot_sha256":hashlib.sha256(bundle).hexdigest(),"gpu_required":True},"nbformat":4,"nbformat_minor":5}
    args.notebook.parent.mkdir(parents=True,exist_ok=True)
    args.notebook.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    args.bundle_out.parent.mkdir(parents=True,exist_ok=True); args.bundle_out.write_bytes(bundle)
    if args.cells_out:
        # Short external-source cells are convenient when operating an already
        # opened Kaggle notebook; the delivered notebook remains self-contained.
        with ZipFile(args.source_zip) as archive:
            original_checksums={name:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if name!='SOURCE_MANIFEST.json'}
        external={"bootstrap":bootstrap_cell(original_checksums),"setup":SETUP,
            "frame_restore":frame_restore_cell(frame_manifest_sha),"guarded_tests":GUARDED,"fit":FIT,
            "score":SCORE,"predict":PREDICT,"summary":SUMMARY,"export":export_cell(Path(__file__).read_bytes())}
        for value in external.values(): compile(value,'<visual-execution-cell>','exec')
        args.cells_out.parent.mkdir(parents=True,exist_ok=True); args.cells_out.write_text(json.dumps(external,indent=2))
    print(json.dumps({"notebook":str(args.notebook),"cells":len(cells),"source_members":len(checksums),
        "source_bundle_bytes":len(bundle),"source_bundle_sha256":hashlib.sha256(bundle).hexdigest(),
        "frame_manifest_sha256":frame_manifest_sha,"builder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))


if __name__=='__main__': main()

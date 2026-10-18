"""Build the native dense/context experiment and exact fitted-source export."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
DETAIL_ZIP_SHA = "3708bc48d39eb68ec3e4a1fca105434d79bc541f697153434050ceeca20a5138"
DETAIL_MANIFEST_SHA = "5659208315115d0b97789fa477fdacd047a3f70c545ff8f4a3482de531a4781c"
CONTEXT_SHA = "de9bb811fca065166b63e0a0e4b3cc271da22067675ceda4f0c8ada53e5531bd"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def cell(kind, source):
    value = {"cell_type": kind, "id": digest((kind + source).encode())[:12],
             "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def collect_source(root):
    root = Path(root)
    paths = set(root.glob("*.py"))
    for directory in ("datathon_ml", "tools", "tests"):
        paths.update((root / directory).glob("*.py"))
    paths.update(root.glob("requirements*.txt"))
    paths.update(root.glob("README*.md"))
    members = {path.relative_to(root).as_posix(): path.read_bytes()
               for path in sorted(paths) if path.is_file()}
    required = ("context_vlm_experiment.py", "datathon_ml/direct_context_vlm.py",
                "datathon_ml/direct_vlm.py", "tests/__init__.py", "tests/test_direct_context_vlm.py")
    if any(name not in members for name in required):
        raise FileNotFoundError("The context source snapshot is incomplete")
    if digest(members["datathon_ml/direct_context_vlm.py"]) != CONTEXT_SHA:
        raise ValueError("The frozen context model source changed")
    baseline = next((path for path in (root / "reference/clip_baseline_cv.json",
        root / "dtFinal/runs/clip_b32/evaluation/cv_report.json",
        root / "artifacts/colab_results/runs/clip_b32/evaluation/cv_report.json") if path.is_file()), None)
    if baseline is None:
        raise FileNotFoundError("The numeric original-target reference is required")
    members["reference/clip_baseline_cv.json"] = baseline.read_bytes()
    members["tools/build_context_notebook.py"] = Path(__file__).read_bytes()
    return members


def source_zip(members):
    checksums = {name: digest(value) for name, value in sorted(members.items())}
    files = {**members, "SOURCE_MANIFEST.json": json.dumps({"contains_private_frames": False,
        "contains_private_dataset": False, "contains_base_weights": False,
        "members_sha256": checksums}, indent=2).encode()}
    output = io.BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(files.items()):
            item = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            item.compress_type = ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            archive.writestr(item, value)
    return output.getvalue(), checksums


def bootstrap(bundle):
    return '''# Restore the embedded source snapshot, never an older attached source bundle.
import base64,hashlib,io,json,os,subprocess,sys,zipfile
from pathlib import Path
BASE=Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('/content') if Path('/content').is_dir() else Path.cwd()
INPUT=Path('/kaggle/input') if Path('/kaggle/input').is_dir() else BASE
V5_WORKSPACE=WORKSPACE=BASE/'datathon_context_vlm_workspace'
V5_RUN=RUN=WORKSPACE/'runs/context_vlm_v5'
assert not any((RUN/name).exists() for name in ('FIT_STARTED.json','selection_lock.json','TEST_OPENED.json')),'Restore source into a fresh fit workspace'
RUN.mkdir(parents=True,exist_ok=True)
def digest_bytes(value): return hashlib.sha256(value).hexdigest()
SOURCE_ZIP=base64.b64decode(''' + repr(base64.b64encode(bundle).decode()) + ''')
assert digest_bytes(SOURCE_ZIP)==''' + repr(digest(bundle)) + '''
with zipfile.ZipFile(io.BytesIO(SOURCE_ZIP)) as archive:
    inventory=json.loads(archive.read('SOURCE_MANIFEST.json')); SOURCE_MEMBERS=inventory['members_sha256']
    names=archive.namelist(); assert len(names)==len(set(names)) and set(names)==set(SOURCE_MEMBERS)|{'SOURCE_MANIFEST.json'}
    for name,expected in SOURCE_MEMBERS.items():
        target=(WORKSPACE/name).resolve(); assert target.is_relative_to(WORKSPACE.resolve())
        value=archive.read(name); assert digest_bytes(value)==expected
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(value)
assert (WORKSPACE/'tests/__init__.py').is_file()
os.chdir(WORKSPACE)
if str(WORKSPACE) not in sys.path: sys.path.insert(0,str(WORKSPACE))
(RUN/'source_restore_receipt.json').write_text(json.dumps({'source_zip_sha256':digest_bytes(SOURCE_ZIP),
    'members_sha256':SOURCE_MEMBERS,'tests_package_initializer_included':True},indent=2))
print({'verified_source_members':len(SOURCE_MEMBERS),'self_contained_snapshot':True})
'''


SETUP = '''# Public packages and pinned public checkpoint only; no private inference here.
subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6',
    'open_clip_torch==3.2.0','safetensors','accelerate','-r','requirements-ml.txt'],check=True)
import torch
assert torch.cuda.is_available(),'Enable a GPU runtime'
from huggingface_hub import snapshot_download
from datathon_ml.direct_vlm import MODEL_ID,REVISION,WEIGHT_SHA256
QWEN_CHECKPOINT=Path(globals().get('QWEN_CHECKPOINT',BASE/'qwen3vl4b_base'))
if not all((QWEN_CHECKPOINT/name).is_file() and digest_bytes((QWEN_CHECKPOINT/name).read_bytes())==expected for name,expected in WEIGHT_SHA256.items()):
    snapshot_download(MODEL_ID,revision=REVISION,local_dir=str(QWEN_CHECKPOINT),
        allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','README.md'])
assert all(digest_bytes((QWEN_CHECKPOINT/name).read_bytes())==expected for name,expected in WEIGHT_SHA256.items())
QWEN_DEVICE='cuda:1' if torch.cuda.device_count()>1 else 'cuda:0'
print({'public_checkpoint_verified':True,'gpu_count':torch.cuda.device_count(),
    'license':'Apache-2.0','hosted_inference_API_used':False})
'''


RESTORE = '''# Verify every private input member natively; never render its images.
import hashlib,json,zipfile
from pathlib import Path
WORKSPACE=Path(V5_WORKSPACE); RUN=Path(V5_RUN)
INPUT=Path('/kaggle/input') if Path('/kaggle/input').is_dir() else Path(globals().get('INPUT',Path.cwd()))
def digest_bytes(value): return hashlib.sha256(value).hexdigest()
DETAIL_ZIP_SHA=''' + repr(DETAIL_ZIP_SHA) + '''
DETAIL_MANIFEST_SHA=''' + repr(DETAIL_MANIFEST_SHA) + '''
archive_path=next((p for p in INPUT.rglob('reviewed_detail_framepack.zip') if digest_bytes(p.read_bytes())==DETAIL_ZIP_SHA),None)
detail_archive=zipfile.ZipFile(archive_path) if archive_path else None
if detail_archive:
    payload=detail_archive.read('DETAIL_FRAMEPACK_MANIFEST.json')
else:
    candidates=[p.parent for p in INPUT.rglob('DETAIL_FRAMEPACK_MANIFEST.json') if digest_bytes(p.read_bytes())==DETAIL_MANIFEST_SHA]
    assert len(candidates)==1,'Attach the separate private RGB448 detail input'
    detail_root=candidates[0]; payload=(detail_root/'DETAIL_FRAMEPACK_MANIFEST.json').read_bytes()
assert digest_bytes(payload)==DETAIL_MANIFEST_SHA
detail_inventory=json.loads(payload); DETAIL_CHECKSUMS=detail_inventory['members_sha256']
def read_detail_member(name):
    assert name in DETAIL_CHECKSUMS
    relative=Path(name); assert not relative.is_absolute() and '..' not in relative.parts
    value=detail_archive.read(name) if detail_archive else (detail_root/name).read_bytes()
    assert digest_bytes(value)==DETAIL_CHECKSUMS[name]
    return value
if detail_archive:
    names=detail_archive.namelist(); assert len(names)==len(set(names)) and set(names)==set(DETAIL_CHECKSUMS)|{'DETAIL_FRAMEPACK_MANIFEST.json'}
for name in DETAIL_CHECKSUMS: read_detail_member(name)
assert len(DETAIL_CHECKSUMS)==3876
assert detail_inventory['recordings']==58 and detail_inventory['time_samples']==1879 and detail_inventory['jpeg_images']==3758
assert detail_inventory['image_size']==448 and detail_inventory['contains_audio'] is False
assert detail_inventory['timestamps_equal_frozen_224'] is True
records=json.loads(read_detail_member('dataset/manifest.json')); split=json.loads(read_detail_member('dataset/split.json'))
from datathon_ml.reviewed_models import validate_split
validate_split(records,split)
assert [len(split[name]) for name in ('train','validation','test')]==[34,12,12]
for name in ('dataset/manifest.json','dataset/split.json'):
    target=WORKSPACE/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(read_detail_member(name))
assert all(not (WORKSPACE/'frames'/key).exists() for key in split['test']),'Use a fresh fitting workspace'
fit_keys=set(split['train'])|set(split['validation'])
for name in DETAIL_CHECKSUMS:
    parts=Path(name).parts
    if parts[0]=='frames' and parts[1] in fit_keys:
        target=WORKSPACE/name; assert target.resolve().is_relative_to(WORKSPACE.resolve())
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(read_detail_member(name))
(RUN/'detail_restore_receipt.json').write_text(json.dumps({'detail_manifest_sha256':DETAIL_MANIFEST_SHA,
    'detail_zip_sha256':DETAIL_ZIP_SHA,'verified_members':len(DETAIL_CHECKSUMS),'fit_sources':46,
    'former_test_sources_materialized':0,'independent_test':False},indent=2))
print({'private_members_verified':len(DETAIL_CHECKSUMS),'fit_sources_materialized':46,'former_test_sources_materialized':0})
'''


PROBE = '''from tools.run_scene_temporal_colab import run_guarded
COMMON=['--manifest','dataset/manifest.json','--split','dataset/split.json','--detail-frame-root','frames',
    '--checkpoint',str(QWEN_CHECKPOINT),'--originals-from','reference/clip_baseline_cv.json',
    '--output-dir',str(V5_RUN),'--device',QWEN_DEVICE,'--threads','2']
run_guarded(WORKSPACE,V5_RUN,'focused_tests','unittest',
    ['discover','-s','tests','-p','test_direct_context_vlm.py','-v'],600)
# Only this synthetic32-frame+eight-demo probe may trigger the preregistered24-frame OOM fallback.
run_guarded(WORKSPACE,V5_RUN,'probe','context_vlm_experiment',['probe','--checkpoint',str(QWEN_CHECKPOINT),
    '--output-dir',str(V5_RUN),'--device',QWEN_DEVICE,'--threads','2','--allow-resource-fallback'],1200)
probe=json.loads((V5_RUN/'resource_probe.json').read_text())
assert probe['completed'] and probe['synthetic_probe'] and probe['private_frames_used'] is False
print({'native_resource_query_frames':probe['config']['query_frames'],'probe_completed':True})
'''


FIT = '''assert all(not (WORKSPACE/'frames'/key).exists() for key in split['test'])
run_guarded(WORKSPACE,V5_RUN,'fit','context_vlm_experiment',['fit',*COMMON],5400)
selection=json.loads((V5_RUN/'selection.json').read_text())
assert selection['test_frame_files_opened']==0 and selection['independent_test'] is False
print('Validation winner:',selection['winner']['strategy'])
print('Validation metrics:',selection['winner']['validation_metrics'])
'''


SCORE = '''# Former test is exposed development, released only after this validation lock.
lock=json.loads((V5_RUN/'selection_lock.json').read_text())
assert digest_bytes((V5_RUN/'model_context_vlm.json').read_bytes())==lock['model_sha256']
assert digest_bytes((V5_RUN/'selection.json').read_bytes())==lock['selection_sha256']
assert digest_bytes((V5_RUN/'resource_probe.json').read_bytes())==lock['resource_probe_sha256']
for name in DETAIL_CHECKSUMS:
    parts=Path(name).parts
    if parts[0]=='frames' and parts[1] in set(split['test']):
        target=WORKSPACE/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(read_detail_member(name))
(RUN/'development_release_receipt.json').write_text(json.dumps({'validation_lock_verified':True,
    'released_sources':sorted(split['test']),'independent_test':False},indent=2))
run_guarded(WORKSPACE,V5_RUN,'score','context_vlm_experiment',['score',*COMMON],5400)
report=json.loads((V5_RUN/'context_vlm_test_report.json').read_text())
assert report['independent_test'] is False and report['localization_available'] is False
assert all(not row['events'] for row in report['predictions'])
print('Exposed development metrics:',report['metrics'])
print('Domain metrics:',report['domain_metrics'])
print('These results cannot establish independent unseen-scene80%/87% accuracy.')
'''


INFERENCE_WORKER = '''import argparse,json
from pathlib import Path
from datathon_ml.direct_context_vlm import ContextVLMConfig,LocalContextVLM,make_bundle,_receipt,_sources,STRATEGIES
p=argparse.ArgumentParser(); p.add_argument('--workspace',required=True); p.add_argument('--run',required=True)
p.add_argument('--checkpoint',required=True); p.add_argument('--query-sha',required=True); p.add_argument('--output-dir',required=True)
p.add_argument('--query-frame-root'); a=p.parse_args()
w=Path(a.workspace); run=Path(a.run); artifact=json.loads((run/'model_context_vlm.json').read_text())
assert artifact['source_sha256']==_sources()
config=ContextVLMConfig(**artifact['config']); train=set(artifact['training_sha256'])
targets={row['sha256']:row for row in json.loads((w/'dataset/manifest.json').read_text()) if row['sha256'] in train}
examples=[]
if artifact['winner']['strategy']==STRATEGIES[1]:
    for receipt in artifact['demonstrations']:
        assert receipt['sha256'] in train
        owner=targets[receipt['sha256']]; bundle=make_bundle(w/'frames',owner['sha256'],config,training_demo=owner)
        example={'sha256':owner['sha256'],'domain':owner.get('domain','original'),'label':owner['label'],'bundle':bundle}
        assert _receipt(example,bundle)==receipt
        examples.append(example)
engine=LocalContextVLM(a.checkpoint,config); assert engine.provenance==artifact['model_provenance']
# A new query needs only a verified RGB448 native framepack and its opaque SHA, never its label or points.
query_root=Path(a.query_frame_root) if a.query_frame_root else w/'frames'
probability=engine.score(make_bundle(query_root,a.query_sha,config),examples)
result={'score':probability,'prediction':int(probability>=artifact['winner']['threshold']),
    'threshold':artifact['winner']['threshold'],'events':[],'localization_available':False,
    'generation_performed':False,'hosted_inference_API_used':False}
d=Path(a.output_dir); d.mkdir(parents=True,exist_ok=True); (d/'prediction.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''


def inference_cell():
    return '''# Demonstrate label-free cached-pack inference; no new test selection is performed.
import json
from pathlib import Path
from tools.run_scene_temporal_colab import run_guarded
WORKSPACE=Path(V5_WORKSPACE); V5_RUN=Path(V5_RUN)
split=json.loads((WORKSPACE/'dataset/split.json').read_text())
(WORKSPACE/'tools/context_pack_inference.py').write_text(''' + repr(INFERENCE_WORKER) + ''')
query_key=sorted(split['train'])[0]
run_guarded(WORKSPACE,V5_RUN,'inference','tools.context_pack_inference',['--workspace',str(WORKSPACE),
    '--run',str(V5_RUN),'--checkpoint',str(QWEN_CHECKPOINT),'--query-sha',query_key,
    '--output-dir',str(V5_RUN/'inference')],1200)
prediction=json.loads((V5_RUN/'inference/prediction.json').read_text())
assert prediction['events']==[] and prediction['localization_available'] is False
print('Recording classification only; video decoding is excluded from cached-pack inference timings.')
'''


def live_export_cell(builder_bytes):
    return '''# Export current fitted bytes, never newer local model code or private frames.
import base64,hashlib,json,sys,subprocess,tempfile,zipfile
from pathlib import Path
def export_digest(value): return hashlib.sha256(value).hexdigest()
assert all(name in globals() for name in ('V5_WORKSPACE','V5_RUN'))
workspace=Path(V5_WORKSPACE); run=Path(V5_RUN)
artifact=json.loads((run/'model_context_vlm.json').read_text())
lock=json.loads((run/'selection_lock.json').read_text())
probe=json.loads((run/'resource_probe.json').read_text())
assert export_digest((run/'model_context_vlm.json').read_bytes())==lock['model_sha256']
assert export_digest((run/'selection.json').read_bytes())==lock['selection_sha256']
assert export_digest((run/'resource_probe.json').read_bytes())==lock['resource_probe_sha256']
assert artifact['source_sha256']==lock['source_sha256']==probe['source_sha256']
assert probe['completed'] and probe['synthetic_probe'] and probe['private_frames_used'] is False
assert artifact['resource_probe_sha256']==lock['resource_probe_sha256']
assert artifact['config']==probe['config'] and artifact['model_provenance']==probe['model_provenance']
source_paths={}
for module,expected in artifact['source_sha256'].items():
    path=workspace/f'{module}.py' if module=='context_vlm_experiment' else workspace/'datathon_ml'/f'{module}.py'
    assert path.is_file() and export_digest(path.read_bytes())==expected,'Fitted source changed'
    source_paths[path.relative_to(workspace).as_posix()]=expected
export_base=Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('/content') if Path('/content').is_dir() else Path.cwd()
merge=Path(tempfile.mkdtemp(prefix='datathon_context_export_source_',dir=export_base))
paths=set(workspace.glob('*.py'))
for directory in ('datathon_ml','tools','tests'): paths.update((workspace/directory).glob('*.py'))
paths.update(workspace.glob('requirements*.txt')); paths.update(workspace.glob('README*.md'))
paths.add(workspace/'reference/clip_baseline_cv.json')
source_members={p.relative_to(workspace).as_posix():p.read_bytes() for p in paths if p.is_file()}
assert 'tests/__init__.py' in source_members
source_members['tools/build_context_notebook.py']=base64.b64decode(''' + repr(base64.b64encode(builder_bytes).decode()) + ''')
for name,value in source_members.items():
    path=(merge/name).resolve(); assert path.is_relative_to(merge.resolve())
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(value)
notebook_path=merge/'datathon_context.ipynb'; bundle_path=merge/'context_source.zip'
subprocess.run([sys.executable,str(merge/'tools/build_context_notebook.py'),'--root',str(merge),
    '--notebook',str(notebook_path),'--bundle-out',str(bundle_path)],cwd=merge,check=True)
clean=json.loads(notebook_path.read_text())
assert all(c.get('execution_count') is None and c.get('outputs',[])==[] for c in clean['cells'] if c['cell_type']=='code')
members={'datathon_context.ipynb':notebook_path.read_bytes()}
with zipfile.ZipFile(bundle_path) as archive:
    for name in archive.namelist():
        if name.endswith('.py'): assert archive.read(name)==(merge/name).read_bytes()
        members['source/'+name]=archive.read(name)
for path in run.rglob('*'):
    if not path.is_file() or path.suffix not in {'.json','.log'}: continue
    relative=path.relative_to(run)
    assert not set(relative.parts)&{'frames','dataset','features','open','base_weights'}
    members['runs/context_vlm_v5/'+relative.as_posix()]=path.read_bytes()
members['FITTED_SOURCE_RECEIPT.json']=json.dumps({'actual_fitted_source_sha256':source_paths,
    'actual_fitted_ML_bytes_verified':True,'resource_probe_sha256':lock['resource_probe_sha256'],
    'tests_package_initializer_included':True,'contains_private_frames':False,'contains_base_weights':False},indent=2).encode()
assert not any(name.lower().endswith(('.jpg','.jpeg','.png','.mp4','.mov','.xlsx','.safetensors','.pt')) for name in members)
members['EXPORT_MANIFEST.json']=json.dumps({'contains_private_frames':False,'contains_raw_media':False,
    'contains_base_weights':False,'independent_test':False,
    'members_sha256':{name:export_digest(value) for name,value in sorted(members.items())}},indent=2).encode()
CONTEXT_RESULTS_ZIP=export_base/'datathon_context_results.zip'
with zipfile.ZipFile(CONTEXT_RESULTS_ZIP,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
    for name,value in sorted(members.items()): archive.writestr(name,value)
with zipfile.ZipFile(CONTEXT_RESULTS_ZIP) as archive: assert archive.testzip() is None
print({'context_export_bytes':CONTEXT_RESULTS_ZIP.stat().st_size,'export_members':len(members),
    'context_export_sha256':export_digest(CONTEXT_RESULTS_ZIP.read_bytes()),'private_frames_exported':False})
from IPython.display import FileLink,display
display(FileLink(str(CONTEXT_RESULTS_ZIP)))
if not Path('/kaggle/working').is_dir() and Path('/content').is_dir():
    from google.colab import files
    files.download(str(CONTEXT_RESULTS_ZIP))
'''


def notebook_cells(bundle):
    intro = '''# Native dense/context video classification

This notebook restores exact source bytes, verifies the private RGB448 framepack,
and uses the pinned Apache2.0 Qwen3-VL4B checkpoint in native Python. Setup alone
uses network access. Guarded private runs use no LLM API, audio or text generation.

The two fixed arms are dense zero-shot and eight TRAIN-owned examples balanced
by class and domain. Their thresholds and winner use validation only. The resource
probe first attempts32 uniformly sampled448px query frames plus eight four-frame
224px demonstrations; only a recorded synthetic CUDA OOM permits24 query frames.

The58-source split remains34train/12validation/12former-test. Former-test results
are exposed development comparisons after earlier experiments, not independent
acceptance evidence. Native overlap groups do not establish new-scene or
new-session generalization. This recording classifier produces no event times.
'''
    return [cell('markdown', intro), cell('code', bootstrap(bundle)), cell('code', SETUP),
        cell('markdown', 'Verify all3876 private members and materialize only46 train/validation framepacks.'),
        cell('code', RESTORE), cell('markdown', 'Synthetic implementation tests and native resource probe precede private fitting.'),
        cell('code', PROBE), cell('markdown', 'Freeze the arm and threshold using validation only.'), cell('code', FIT),
        cell('markdown', 'Release former-test framepacks after selection; report them as exposed development.'), cell('code', SCORE),
        cell('markdown', 'The inference helper also accepts a separate verified RGB448 query pack using --query-frame-root. Training demonstration packs remain required for the context arm.'),
        cell('code', inference_cell()), cell('markdown', 'Export exact fitted source, decision recipes, numeric reports/logs and a clean notebook. Frames and public weights stay excluded.'),
        cell('code', live_export_cell(Path(__file__).read_bytes()))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--notebook', type=Path, default=ROOT / 'datathon_context.ipynb')
    parser.add_argument('--bundle-out', type=Path, default=ROOT / 'artifacts/context_notebook_source.zip')
    parser.add_argument('--cells-out', type=Path)
    args = parser.parse_args(argv)
    members = collect_source(args.root)
    bundle, checksums = source_zip(members)
    cells = notebook_cells(bundle)
    for value in cells:
        if value['cell_type'] == 'code':
            compile(''.join(value['source']), '<context-notebook-cell>', 'exec')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'colab': {'name': args.notebook.name, 'provenance': []},
        'source_snapshot_sha256': digest(bundle), 'gpu_required': True}, 'nbformat': 4, 'nbformat_minor': 5}
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    args.notebook.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    args.bundle_out.parent.mkdir(parents=True, exist_ok=True)
    args.bundle_out.write_bytes(bundle)
    if args.cells_out:
        execution = {'public_setup': SETUP, 'private_restore': RESTORE, 'probe_and_tests': PROBE,
            'fit': FIT, 'release_score': SCORE, 'inference': inference_cell(),
            'live_export': live_export_cell(Path(__file__).read_bytes())}
        for source in execution.values():
            compile(source, '<context-execution-cell>', 'exec')
        args.cells_out.parent.mkdir(parents=True, exist_ok=True)
        args.cells_out.write_text(json.dumps(execution, indent=2))
    print(json.dumps({'notebook': str(args.notebook), 'cells': len(cells), 'source_members': len(checksums),
        'source_bytes': len(bundle), 'source_sha256': digest(bundle), 'builder_sha256': digest(Path(__file__).read_bytes())}))


if __name__ == '__main__':
    main()

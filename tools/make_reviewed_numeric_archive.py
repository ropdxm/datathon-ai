"""Package exact reviewed identities and numeric caches for private training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datathon_ml.features import load_feature_cache
from tools.prepare_reviewed_protocol import validate_split


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'artifacts/reviewed_dataset')
    parser.add_argument('--features',type=Path,default=ROOT/'artifacts/reviewed_features')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/reviewed_numeric_dataset.zip')
    args=parser.parse_args()
    records=json.loads((args.dataset/'manifest.json').read_text())
    split=json.loads((args.dataset/'split.json').read_text())
    validate_split(records,split)
    if len(records)!=58 or [len(split[key]) for key in ('train','validation','test')]!=[34,12,12]:
        raise ValueError('Unexpected frozen reviewed protocol')
    runtime=json.loads((args.features/'extraction_runtime.json').read_text())
    if not runtime.get('completed') or runtime.get('completed_sources')!=58 or runtime.get('network_attempts')!=0:
        raise ValueError('Complete guarded visual extraction is required before packaging')
    members={f'dataset/{name}.json':(args.dataset/f'{name}.json').read_bytes()
             for name in ('manifest','split')}
    for representation,folder in (('raw_clip','clip'),('semantic18','semantic')):
        for row in records:
            digest=row['sha256']
            path=args.features/folder/(digest+'.npz')
            cached=load_feature_cache(path)
            if cached['sha256']!=digest:
                raise ValueError('Numeric cache source differs from frozen dataset')
            members[f'features/{representation}/{digest}.npz']=path.read_bytes()
    members['DATASET_MANIFEST.json']=json.dumps({'format_version':1,
        'contains_video_frames_audio_or_private_descriptions':False,
        'source_recordings':len(records),'extraction_runtime':runtime,
        'members_sha256':{name:hashlib.sha256(value).hexdigest()
                          for name,value in sorted(members.items())}},indent=2).encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(args.output,'w',ZIP_DEFLATED,compresslevel=6) as archive:
        for name,value in sorted(members.items()):
            item=ZipInfo(name,(2026,1,1,0,0,0));item.compress_type=ZIP_DEFLATED
            item.external_attr=0o644<<16
            archive.writestr(item,value)
    print(json.dumps({'archive':str(args.output),'bytes':args.output.stat().st_size,
        'members':len(members),'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'recordings':len(records),'contains_media_or_workbook':False}))


if __name__=='__main__':
    main()

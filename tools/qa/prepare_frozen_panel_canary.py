#!/usr/bin/env python3
"""CPU-only canary inputs replaying the twenty actual admitted pilot geometries."""
import argparse
import gzip
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from avengine.qa.frozen_panel_versions import read,parent_hash,digest


def archived_json(root,manifest,path):
    record=next(x for x in manifest['native_files'] if x['source']==str(path))
    if record['kind']!='file':raise ValueError('expected lossless JSON witness')
    data=gzip.decompress((root/'retention'/record['payload']).read_bytes())
    import hashlib
    if hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('canary witness changed')
    return json.loads(data)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--pilot-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.pilot_root.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('canary input output must be new')
    # Verify retained evidence before copying poses. Never read scoring answers.
    for line in (root/'retention/COMMITMENTS.sha256').read_text().splitlines():
        sha,rel=line.split('  ',1)
        if digest(root/'retention'/rel)!=sha:raise ValueError('retention commitment changed')
    c=read(root/'inputs/config.json');houses={h['house']:[] for h in c['houses']}
    for i,certpath in enumerate(sorted((root/'scoring_private/certificates').glob('*.json'))):
        cert=read(certpath);m=read(root/'retention/groups'/(cert['group']+'.json'))
        if digest(certpath)!=m['certificate_sha256']:raise ValueError('pilot certificate changed')
        recipe=archived_json(root,m,cert['provenance']['recipe'])
        plan=archived_json(root,m,cert['construction']['plan_paths'][0]);geometry=recipe['geometry']
        j=i%c['groups_per_house'];hi=i//c['groups_per_house']
        houses[cert['house']].append({'index':i,'ordinal':j,'house_index':hi,'house':cert['house'],
            'room':cert['room'],'scene_seed':plan['seed'],'quota':cert['quota'],'geometry':geometry,
            'physical_parent_sha256':parent_hash(cert['house'],geometry),'status':'cpu_preflight_pass',
            'answers_generated':False,'replayed_pilot_group':cert['group'],
            'baseline_certificate_sha256':digest(certpath),'new_materials_native_rgb':'not_run'})
    out.mkdir(mode=0o700,parents=True)
    for house,groups in houses.items():
        d=out/house;d.mkdir();(d/'prepared.json').write_text(json.dumps({'status':'pass','house':house,'groups':groups},ensure_ascii=False,indent=2)+'\n')
    c.update(output=str(out.parent/'canary_new_version'),panel_id='canary_new_version',prepared_geometry_root=str(out),
             retain_compact=True,excluded_physical_parent_sha256=[],excluded_video_pair_sha256=[])
    (out/'canary_base_config.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'pass','groups':sum(map(len,houses.values())),'answers_read':False,'rendering':'not_run','output':str(out)}))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Run a five-house pilot, freeze its rules, then evaluate a read-only snapshot."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO)]
import numpy as np
from avengine.acoustics.hole_patch import RULES, loop_candidates, write_json


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()


def prepare(a):
    root=a.output;inputs=root/'inputs';inputs.mkdir(exist_ok=False)
    shutil.copyfile(a.results,inputs/'all_house_results.json')
    rows=read(inputs/'all_house_results.json')
    blocked=[r for r in rows if any('GEOMETRIC_ESCAPE' in x for x in r['reasons'])]
    def size(r):return read(r['acoustic_package_manifest'])['geometry']['triangle_count']
    pilot=[];control=[]
    for family,n in (('hm3d',3),('mp3d',2)):
        group=sorted([r for r in blocked if r['family']==family],key=lambda r:(size(r),r['house']))
        pilot.extend(r['house'] for r in group[:n])
        group=sorted([r for r in rows if r['family']==family and r['status']=='pass'],key=lambda r:r['house'])
        control.extend(r['house'] for r in group[:5])
    registries={};sources=[a.results,a.hm3d_manifest,a.mp3d_manifest,a.thresholds]
    for r in blocked+[r for r in rows if r['house'] in control]:
        p=r['input']['registry_source']
        if p not in registries:
            dest=inputs/f'registry_{len(registries)}.jsonl';shutil.copyfile(p,dest)
            registries[p]=str(dest);sources.append(Path(p))
    manifests={}
    for family,path in (('hm3d',a.hm3d_manifest),('mp3d',a.mp3d_manifest)):
        dest=inputs/(family+'_source_manifest.json');shutil.copyfile(path,dest);manifests[family]=read(dest)
    shutil.copyfile(a.thresholds,inputs/'thresholds.frozen.yaml')
    write_json(inputs/'rule_draft.json',RULES)
    plan={'created_at':now(),'results_source':str(a.results),'results_snapshot_sha256':sha(inputs/'all_house_results.json'),
          'base_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
          'blocked':[r['house'] for r in blocked],'pilot':pilot,'controls':control,'registry_snapshots':registries,
          'family_counts':dict(Counter(r['family'] for r in blocked)),
          'input_files':[{'path':str(p),'sha256':sha(p)} for p in sources],
          'pilot_selection':'smallest acoustic triangle counts per family, then house identity; three HM3D and two MP3D',
          'controls_selection':'first five passing houses per family by house identity; chosen before repair evaluation',
          'room_scope':'same preselected acoustic rooms as original result; unmeasured candidate rooms cannot be certified'}
    write_json(inputs/'selection.json',plan)
    print(json.dumps({k:plan[k] for k in ('family_counts','pilot','controls')},ensure_ascii=False),flush=True)


def plan_house(a):
    rows=read(a.output/'inputs/all_house_results.json');r=next(r for r in rows if r['house']==a.house)
    p=Path(r['acoustic_package_manifest']);m=read(p)
    v=np.load(p.parent/m['arrays']['vertices']['path'],mmap_mode='r',allow_pickle=False)
    t=np.load(p.parent/m['arrays']['triangles']['path'],mmap_mode='r',allow_pickle=False)
    labels=np.empty(len(t),dtype=object)
    for o in m['objects']:
        start=o['triangle_offset'];labels[start:start+o['triangle_count']]=o['source_material_name']
    path=a.output/'pilot_plan'/r['house'];path.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    candidates,rays,clusters,stats=loop_candidates(v,t,labels,r['leakage'],RULES)
    record={'house':r['house'],'family':r['family'],'base_package_manifest':str(p),
            'rules':RULES,'candidates':candidates,'escaped_rays':rays,'exit_clusters':clusters,'boundary_stats':stats,
            'wall_seconds':time.monotonic()-start,'finished_at':now(),'qualification':'geometry_candidates_only'}
    write_json(path/'candidates.json',record)
    print(json.dumps({'house':r['house'],'seconds':record['wall_seconds'],'stats':stats,'candidates':len(candidates),
                      'eligible_geometry':sum(c['status']=='candidate' for c in candidates),
                      'semantic_counts':dict(Counter(c['semantic_label'] for c in candidates))}),flush=True)


def allocation(a):
    def sample():
        out={}
        for s in Path('/proc/stat').read_text().splitlines():
            f=s.split()
            if f[0].startswith('cpu') and f[0][3:].isdigit():out[int(f[0][3:])]=list(map(int,f[1:9]))
        return out
    x=sample();time.sleep(1);y=sample();busy={}
    for c,v in y.items():
        d=np.asarray(v)-x[c];busy[c]=float(1-(d[3]+d[4])/sum(d)) if sum(d) else 0.
    idle=sorted((c for c,b in busy.items() if b<.2),key=lambda c:(busy[c],c))
    workers=min(a.workers,len(idle)//16)
    if not workers:raise ValueError('No spare eight-core CPU budget with reserve')
    data={'time':now(),'logical_cores':len(busy),'idle_below_20_percent':len(idle),'workers':workers,
          'threads_per_native_worker':8,'affinity_groups':[idle[i*8:i*8+8] for i in range(workers)],
          'busy_fraction_by_cpu':busy,'measure':'one-second /proc/stat; leave at least half idle cores unallocated','cpu_only':True}
    write_json(a.output/'evidence'/('cpu_allocation_'+a.stage+'.json'),data)
    print(','.join(map(str,data['affinity_groups'][0])))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    prep=s.add_parser('prepare')
    for key in ('output','results','hm3d_manifest','mp3d_manifest','thresholds'):prep.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    q=s.add_parser('plan-house');q.add_argument('--output',type=Path,required=True);q.add_argument('--house',required=True)
    al=s.add_parser('allocate');al.add_argument('--output',type=Path,required=True);al.add_argument('--stage',required=True);al.add_argument('--workers',type=int,default=1)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError("CUDA_VISIBLE_DEVICES must be empty")
    {'prepare':prepare,'plan-house':plan_house,'allocate':allocation}[a.command](a)
if __name__=='__main__':main()

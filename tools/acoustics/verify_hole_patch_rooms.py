#!/usr/bin/env python3
"""Verify each candidate room of actually patched houses with original gates."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO)]
from avengine.acoustics.hole_patch import write_json
from tools.acoustics.accept_houses import compact_room
from tools.acoustics.run_hole_patch import assert_frozen


def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()


def verify_house(root,record_path,timeout):
    record=read(record_path);house=record['house'];selection=read(root/'inputs/selection.json')
    original=next(r for r in read(root/'inputs/all_house_results.json') if r['house']==house)
    h=original['input'];registry=selection['registry_snapshots'][h['registry_source']]
    rooms=[json.loads(s) for s in Path(registry).read_text().splitlines() if s.strip()]
    rooms=[r for r in rooms if r['house']==house and r['room_label'] in h['candidate_room_labels']]
    assert len(rooms)==h['candidate_room_count'], 'Candidate room count differs from original registry'
    target=root/'room_verification'/house;target.mkdir(parents=True,exist_ok=True)
    outcomes=[]
    for r in rooms:
        label=r['room_label'];dest=target/label
        complete=dest/'complete.json'
        if complete.exists():outcomes.append(read(complete));continue
        dest.mkdir(exist_ok=False)
        one=dict(h);one['acoustic_rooms']=[compact_room(r)]
        manifest=read(root/'inputs'/(h['family']+'_source_manifest.json'));manifest['houses']=[one]
        mp=dest/'acceptance_manifest.json';write_json(mp,manifest)
        row={'house':house,'room_label':label,'runs':{},'criteria':manifest['criteria']}
        for method,package in (('baseline',record['base_package_manifest']),('local',record['local_package_manifest'])):
            out=dest/method;log=dest/(method+'.log')
            command=[sys.executable,str(REPO/'tools/acoustics/accept_houses.py'),'measure','--manifest',str(mp),
                     '--output',str(out),'--house',house,'--reuse-package',package]
            with log.open('x') as f:
                p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
                write_json(dest/(method+'_process.json'),{'pid':p.pid,'command':command,'log':str(log),'started_at':now()})
                try:rc=p.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                    raise RuntimeError('Owned room acceptance timed out: '+str(p.pid))
            rp=out/'houses'/house/'result.json'
            if rc!=0 or not rp.exists():raise RuntimeError('Missing native room result: '+str(rp))
            result=read(rp);row['runs'][method]={'result_path':str(rp),'status':result['status'],'reasons':result['reasons']}
        row['rescued']=row['runs']['baseline']['status']!='pass' and row['runs']['local']['status']=='pass'
        row['finished_at']=now();write_json(complete,row);outcomes.append(row)
        print(json.dumps({'house':house,'room':label,'baseline':row['runs']['baseline']['status'],'local':row['runs']['local']['status']}),flush=True)
    data={'house':house,'family':h['family'],'room_count':len(outcomes),'rooms':outcomes,
          'all_local_rooms_pass':all(r['runs']['local']['status']=='pass' for r in outcomes),'finished_at':now()}
    write_json(target/'complete.json',data);return data


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=float,default=5400)
    a=p.parse_args();root=a.output
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    assert_frozen(root)
    selection=read(root/'inputs/selection.json');records=[]
    for h in selection['blocked']+selection['controls']:
        stage='pilot' if h in selection['pilot'] else 'batch'
        rp=root/stage/'houses'/h/'patch_record.json'
        if rp.exists() and read(rp)['patch_count'] and not (root/'room_verification'/h/'complete.json').exists():records.append(rp)
    allocation=read(root/'evidence/cpu_allocation_batch.json');slots=queue.Queue()
    for g in allocation['affinity_groups']:slots.put(g)
    def one(path):
        g=slots.get()
        try:
            # Each thread launches a process, so CPU affinity is per worker and
            # does not mutate the parent's or another collaborator's affinity.
            h=read(path)['house'];log=root/'logs'/('verify_rooms_'+h+'.log')
            command=['taskset','-c',','.join(map(str,g)),sys.executable,str(Path(__file__).resolve()),'--output',str(root),'--one-house',str(path),'--timeout',str(a.timeout)]
            with log.open('x') as f:
                proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
                write_json(root/'evidence'/('room_worker_'+h+'.json'),{'pid':proc.pid,'command':command,'log':str(log),'started_at':now()})
                rc=proc.wait()
            return {'house':h,'returncode':rc,'log':str(log)}
        finally:slots.put(g)
    with ThreadPoolExecutor(max_workers=allocation['workers']) as pool:
        for f in as_completed([pool.submit(one,r) for r in records]):print(json.dumps(f.result()),flush=True)
    missing=[read(r)['house'] for r in records if not (root/'room_verification'/read(r)['house']/'complete.json').exists()]
    write_json(root/'evidence/room_verification_execution.json',{'finished_at':now(),'scheduled_houses':[read(r)['house'] for r in records],'missing':missing})
    if missing:raise ValueError('Incomplete candidate-room verification: '+', '.join(missing))


if __name__=='__main__':
    if '--one-house' in sys.argv:
        # Worker mode is kept explicit and remains subject to the same freeze.
        p=argparse.ArgumentParser();p.add_argument('--one-house',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=float,default=5400)
        a=p.parse_args();assert_frozen(a.output);verify_house(a.output,a.one_house,a.timeout)
    else:main()

#!/usr/bin/env python3
"""Register and measure every available MP3D house with frozen room rules."""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from tools.rooms.room_screening.mp3d_geometry import read_house
from tools.rooms.room_selection.run import process_house, write_json, load_parameters
from tools.rooms.room_selection import PROTOCOL_VERSION

def register(a):
    inv=json.loads(a.inventory.read_text());_,p=load_parameters(a.thresholds)
    jobs=[];errors=[]
    for h in sorted((h for h in inv['houses'] if h['family']=='mp3d'),key=lambda h:h['house_id']):
        scene=Path(h['scene_directory']);sid=h['house_id'];house='mp3d_'+sid
        assets={k:scene/(sid+s) for k,s in [('annotation_source','.house'),('semantic_source','_semantic.ply'),('raw_visual','.glb'),('navmesh_source','.navmesh')]}
        try:
            for f in assets.values():
                if not f.is_file():raise FileNotFoundError(f)
            ann,regions,levels=read_house(assets['annotation_source'])
            rows=[]
            for rid,native in sorted(regions.items()):
                members=[v for v in ann.values() if v['region_id']==rid]
                rows.append(dict(schema=PROTOCOL_VERSION,family='mp3d',house=house,room_label='R'+str(rid),region_id=rid,semantic_region_present=True,semantic_instance_count=len(members),semantic_categories=sorted({v['category'] for v in members}),native_region=native,legacy_room=None,legacy_rooms_source=None,human=None,stage0=dict(status='registered',reason_codes=[]),stage1=dict(status='not_run',reason_codes=[]),stage2=dict(status='not_run',reason_codes=[]),stage4=dict(status='not_run',reason_codes=['NO_HISTORICAL_MP3D_HOUSE_GATE']),scene_directory=str(scene),annotation_source=str(assets['annotation_source']),semantic_source=str(assets['semantic_source']),navmesh_source=str(assets['navmesh_source']),metrics=None,human_review_status='待 smy 审房间'))
            job=dict(house=house,family='mp3d',scene_directory=str(scene),navmesh=str(assets['navmesh_source']),rows=rows,native_levels=sorted(levels))
            write_json(a.output/'jobs'/(house+'.json'),job);jobs.append(job)
        except Exception as e:errors.append(dict(house=house,status='review',reason=str(e)))
    write_json(a.output/'inputs.json',dict(houses=[j['house'] for j in jobs],thresholds_source=str(a.thresholds),inventory_source=str(a.inventory),parameters=p,registration_errors=errors,cpu_only=True,room_count=sum(len(j['rows']) for j in jobs),human_review_status='待 smy 审房间'))
    print('MP3D_REGISTER',len(jobs),sum(len(j['rows']) for j in jobs),errors,flush=True)

def worker(a):
    job=json.loads((a.output/'jobs'/(a.house+'.json')).read_text())
    args=SimpleNamespace(thresholds=a.thresholds,output=a.output,skip_splitting=False,media_root=a.output/'unavailable_human_media',overhead_base=None)
    try:process_house(args,job)
    except Exception as e:
        import traceback
        for r in job['rows']:
            r['stage1']=dict(status='review',reason_codes=['MP3D_MEASUREMENT_FAILED'])
            r['stage2']=dict(status='not_run',reason_codes=['MP3D_MEASUREMENT_FAILED'])
        write_json(a.output/'houses'/(a.house+'.json'),dict(house=a.house,status='review',rows=job['rows'],error=repr(e),traceback=traceback.format_exc()))
        raise

def measure(a):
    inputs=json.loads((a.output/'inputs.json').read_text());houses=inputs['houses']
    if a.house:houses=[a.house]
    if a.limit:houses=houses[:a.limit]
    def one(h):
        dest=a.output/'houses'/(h+'.json')
        if dest.exists():return h,json.loads(dest.read_text()).get('status')
        log=a.output/'logs'/(h+'.log');log.parent.mkdir(exist_ok=True)
        cmd=[sys.executable,'-u','-m','tools.rooms.room_selection.mp3d_train_pool','worker','--output',str(a.output),'--thresholds',str(a.thresholds),'--house',h]
        with log.open('x') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            write_json(a.output/'logs'/(h+'.process.json'),dict(pid=proc.pid,command=cmd,log=str(log)))
            try:rc=proc.wait(timeout=a.house_timeout)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                rc=-15
        if not dest.exists():
            job=json.loads((a.output/'jobs'/(h+'.json')).read_text())
            for r in job['rows']:r['stage1']=dict(status='review',reason_codes=['MP3D_WORKER_FAILED'])
            write_json(dest,dict(house=h,status='review',rows=job['rows'],error='worker returncode '+str(rc)))
        return h,json.loads(dest.read_text()).get('status')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for h,status in pool.map(one,houses):print('MP3D_DONE',h,status,flush=True)
    all_rows=[]
    for h in inputs['houses']:
        f=a.output/'houses'/(h+'.json')
        if f.exists():all_rows.extend(json.loads(f.read_text())['rows'])
    with (a.output/'rooms_registry.jsonl').open('w') as f:
        for row in all_rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    print('MP3D_STAGE1',dict(Counter(r['stage1']['status'] for r in all_rows)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['register','measure','worker']);p.add_argument('--output',type=Path,required=True);p.add_argument('--thresholds',type=Path,required=True);p.add_argument('--inventory',type=Path);p.add_argument('--house');p.add_argument('--limit',type=int);p.add_argument('--workers',type=int,default=2);p.add_argument('--house-timeout',type=int,default=1800);a=p.parse_args();globals()[a.stage](a)

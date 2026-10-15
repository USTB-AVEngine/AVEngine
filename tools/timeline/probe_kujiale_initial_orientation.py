#!/usr/bin/env python3
"""Fresh ordinary UE parent proving seeded natural initial orientations."""
from __future__ import annotations
import argparse,copy,datetime,json,os,subprocess,time
from pathlib import Path
from shapely.geometry import shape
from avengine.timeline.v7_kujiale_manifest import append_task
from avengine.timeline.v7_kujiale_scheduler import resources,can_start,isolated_command
from tools.timeline.run_v7_kujiale_rooms import load


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--physical-gpu',type=int,required=True);p.add_argument('--max-attempts',type=int,default=8);a=p.parse_args();m=load(a.manifest);root=Path(m['output_root']);registration=load(m['formal_kujiale_registration']);eligible={(r['house'],r['room_id'])for r in registration['rooms']if r['smoke_status']=='pass'};areas={}
    for house in load(m['houses_ue'])['houses']:
        for room in load(house['input']['ue_config']['acceptance_plan_path'])['rooms']:areas[(house['house'],room['room_id'])]=shape(room['floor_polygon']).area
    task=min((t for t in m['parents']if t.get('automatic_registered_room')and not t['row'].get('required_sound_classes')and(t['row']['house'],t['row']['room_id'])in eligible),key=lambda t:(-areas[(t['row']['house'],t['row']['room_id'])],t['slot']));request={k:task['row'][k]for k in ('house','room_id')};request['scene_seed']=int(m['scene_seed_base'])+2000000+len(m['parents'])*1009;request['question_quota_base']=len(m['parents'])*20;base=m['campaign_id']+'_initial_orientation_'+request['house']+'_'+request['room_id']+'_q'+str(len(m['parents']));deadline=datetime.datetime.fromisoformat(load(root/'inputs/task_scope.json')['deadline_utc'])
    for attempt in range(a.max_attempts):
        while not can_start(resources(root),active_parents=0,max_parents=1):
            if datetime.datetime.now(datetime.timezone.utc)>=deadline:return
            time.sleep(5)
        if datetime.datetime.now(datetime.timezone.utc)>=deadline:return
        m=load(a.manifest);current=copy.deepcopy(task);current.update(slot=len(m['parents']),ue_physical_gpu=a.physical_gpu,initial_orientation_probe=True);current['row']={**request,'construction_attempt':attempt,'scene_id':base+f'_a{attempt:02d}'};current=append_task(a.manifest,current);slot=current['slot'];log=root/'logs'/f'parent_{slot:03d}_initial_orientation.log';print('FRESH_INITIAL_ORIENTATION',slot,current['row'],flush=True)
        with log.open('w')as stream:rc=subprocess.call(isolated_command(a.manifest,slot,a.physical_gpu,os.environ.get('GAP_PY',os.sys.executable),m['worktree'],root),cwd=m['worktree'],stdout=stream,stderr=subprocess.STDOUT)
        job=load(root/'evidence/jobs'/f'{slot:03d}.json')
        with(root/'PROGRESS.md').open('a')as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+' G 随机场景初始朝向实测 '+current['row']['scene_id']+' '+job['status']+' reason='+str(job.get('reason',''))+' family='+json.dumps(job.get('export',{}).get('question_family_counts',{}),ensure_ascii=False)+'\n')
        if rc==0:return
    raise SystemExit(2)
if __name__=='__main__':main()

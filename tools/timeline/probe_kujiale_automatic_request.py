#!/usr/bin/env python3
"""Unattended required-class smoke in the first previously unstarted legal room."""
from __future__ import annotations
import argparse,copy,datetime,json,os,subprocess,time
from pathlib import Path
from avengine.timeline.v7_kujiale_scheduler import resources,isolated_command,assign_gpu
from tools.timeline.run_v7_kujiale_rooms import load,write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--required-sound-class',required=True);p.add_argument('--physical-gpu',type=int,required=True);p.add_argument('--python',default=os.environ.get('GAP_PY',os.sys.executable));p.add_argument('--max-attempts',type=int,default=8);a=p.parse_args();m=load(a.manifest);root=Path(m['output_root']);task=next(t for t in m['parents']if t.get('automatic_registered_room')and not(root/'evidence/jobs'/f'{t["slot"]:03d}.json').exists());request=copy.deepcopy(task['row']);base='v7_gap_auto_class_'+request['house']+'_'+request['room_id'];request['required_sound_classes']=[a.required_sound_class]
    for attempt in range(a.max_attempts):
        while True:
            s=resources(root)
            if s['MemAvailable_bytes']>=30e9 and s['PSS_bytes']+5e9<=40e9 and s['data_free_bytes']>=s['minimum_disk_free_bytes']:break
            print('PAUSE_NEW_REQUIRED_CLASS',s,flush=True);time.sleep(10)
        m=load(a.manifest);current=copy.deepcopy(task);current['slot']=len(m['parents']);current['ue_physical_gpu']=a.physical_gpu;current['focus']='taxonomy';current['row']=copy.deepcopy(request);current['row'].update(scene_id=base+f'_a{attempt:02d}',construction_attempt=attempt);m['parents'].append(current);write(a.manifest,m);slot=current['slot'];log=root/'logs'/f'parent_{slot:03d}_automatic_required_class.log';print('AUTOMATIC_REQUIRED_CLASS',slot,current['row'],flush=True)
        with log.open('w')as stream:rc=subprocess.call(isolated_command(a.manifest,slot,a.physical_gpu,a.python,m['worktree'],root),stdout=stream,stderr=subprocess.STDOUT,cwd=m['worktree'])
        job=load(root/'evidence/jobs'/f'{slot:03d}.json')
        with(root/'PROGRESS.md').open('a')as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+' G 任意未跑房间自动'+a.required_sound_class+'构造，'+current['row']['scene_id']+' '+job['status']+' reason='+str(job.get('reason',''))+' family='+json.dumps(job.get('export',{}).get('question_family_counts',{}),ensure_ascii=False)+'\n')
        if rc==0:return
    raise SystemExit(2)
if __name__=='__main__':main()

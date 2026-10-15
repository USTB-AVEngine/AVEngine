#!/usr/bin/env python3
"""Unattended required-class smoke using ordinary legal room/category inputs."""
from __future__ import annotations
import argparse,copy,datetime,json,os,subprocess,time
from pathlib import Path
from avengine.timeline.v7_kujiale_manifest import append_task
from tools.timeline.run_v7_kujiale_rooms import AttachedChild
from avengine.timeline.v7_kujiale_scheduler import resources,isolated_command,assign_gpu
from tools.timeline.run_v7_kujiale_rooms import load,write

def select_ordinary_task(tasks,contract,required_categories,*,started_slots=(),passed_houses=(),room_areas=None,attempted_rooms=()):
    """Prefer current legal witnesses, a new house and an unstarted room.

    Missing categories remain eligible: normal current-native construction will
    supply their witnesses, without dropping a requested person or animal.
    """
    cells={r['key']:r for r in contract['rooms']}
    def rank(task):
        row=task['row'];cell=cells[row['house']+'::'+row['room_id']]
        return (len(set(required_categories)-set(cell['categories'])),(row['house'],row['room_id'])in attempted_rooms,row['house']in passed_houses,-float((room_areas or{}).get(cell['key'],0)),task['slot']in started_slots,task['slot'])
    return min((t for t in tasks if t.get('automatic_registered_room')and not t['row'].get('required_sound_classes')),key=rank)


def construction_attempts(first,deadline,*,clock=None):
    """Keep constructing until the explicit task deadline, not an arbitrary try count."""
    clock=clock or (lambda:datetime.datetime.now(datetime.timezone.utc))
    attempt=first
    while clock()<deadline:
        yield attempt
        attempt+=1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--required-sound-class',required=True);p.add_argument('--physical-gpu',type=int,required=True);p.add_argument('--python',default=os.environ.get('GAP_PY',os.sys.executable));p.add_argument('--max-attempts',type=int,default=8);p.add_argument('--resume-slot',type=int);p.add_argument('--resume-pid',type=int);a=p.parse_args();m=load(a.manifest);root=Path(m['output_root']);runtime=load(m['runtime_ue']);contract=load(runtime['production_input_contract']);inventory=load(runtime['kujiale_ue_asset_admission']);required={'human.standing'}
    from avengine.timeline.v7_production_inputs import asset_category
    for asset in inventory['assets']:
        if a.required_sound_class in(asset.get('allowed_v7_classes_candidate')or[]):
            native=copy.deepcopy(asset);native.setdefault('species_id',native.get('category'));native.setdefault('kind','animal'if native.get('category')in {'dog','cat','bird'}else'object');required.add(asset_category(native,{},'kujiale'))
    from shapely.geometry import shape
    areas={}
    for house in load(m['houses_ue'])['houses']:
        for room in load(house['input']['ue_config']['acceptance_plan_path'])['rooms']:
            areas[house['house']+'::'+room['room_id']]=shape(room['floor_polygon']).area
    jobs=[load(p)for p in(root/'evidence/jobs').glob('*.json')];task=m['parents'][a.resume_slot]if a.resume_slot is not None else select_ordinary_task(m['parents'],contract,required,started_slots={j['slot']for j in jobs},passed_houses={j['house']for j in jobs if j['status']=='passed'},room_areas=areas);request=copy.deepcopy(task['row']);base=m.get('campaign_id','v7_fresh')+'_required_'+a.required_sound_class+'_'+request['house']+'_'+request['room_id']+'_q'+str(len(m['parents']));request['required_sound_classes']=[a.required_sound_class];request['scene_seed']=int(m.get('scene_seed_base',261015915000))+1000000+len(m['parents'])*1009
    first=0
    if a.resume_slot is not None:
        job_path=root/'evidence/jobs'/f'{a.resume_slot:03d}.json';current=m['parents'][a.resume_slot];request=copy.deepcopy(current['row']);base=request['scene_id'].rsplit('_a',1)[0];first=int(request['construction_attempt'])+1
        if not job_path.exists()and a.resume_pid is not None:
            child=AttachedChild(a.resume_pid,a.resume_slot,a.manifest)
            while child.poll()is None:time.sleep(10)
        if job_path.exists():
            job=load(job_path)
            if job['status']=='passed':return
            if job['status']=='running':
                try:
                    child=AttachedChild(job['pid'],a.resume_slot,a.manifest)
                    while child.poll()is None:time.sleep(10)
                except (OSError,ValueError):pass
                if load(job_path)['status']=='passed':return
    deadline=datetime.datetime.fromisoformat(load(root/'inputs/task_scope.json')['deadline_utc'])
    for attempt in construction_attempts(first,deadline):
        while True:
            if datetime.datetime.now(datetime.timezone.utc)>=deadline:return
            s=resources(root)
            if s['MemAvailable_bytes']>=30e9 and s['PSS_bytes']+5e9<=40e9 and s['data_free_bytes']>=s['minimum_disk_free_bytes']:break
            print('PAUSE_NEW_REQUIRED_CLASS',s,flush=True);time.sleep(10)
        m=load(a.manifest)
        if attempt>0 and attempt%3==0:
            jobs=[load(p)for p in(root/'evidence/jobs').glob('*.json')];attempted={(j['house'],j['room_id'])for j in jobs if a.required_sound_class in m['parents'][j['slot']]['row'].get('required_sound_classes',[])}
            task=select_ordinary_task(m['parents'],contract,required,started_slots={j['slot']for j in jobs},passed_houses={j['house']for j in jobs if j['status']=='passed'},room_areas=areas,attempted_rooms=attempted);request=copy.deepcopy(task['row']);request['required_sound_classes']=[a.required_sound_class];request['scene_seed']=int(m.get('scene_seed_base',261015915000))+1000000+len(m['parents'])*1009;base=m.get('campaign_id','v7_fresh')+'_required_'+a.required_sound_class+'_'+request['house']+'_'+request['room_id']+'_q'+str(len(m['parents']))
        current=copy.deepcopy(task);current['slot']=len(m['parents']);current['ue_physical_gpu']=a.physical_gpu;current['focus']='taxonomy';current['row']=copy.deepcopy(request);current['row'].update(scene_id=base+f'_a{attempt:02d}',construction_attempt=attempt);current=append_task(a.manifest,current);slot=current['slot'];log=root/'logs'/f'parent_{slot:03d}_automatic_required_class.log';print('AUTOMATIC_REQUIRED_CLASS',slot,current['row'],flush=True)
        with log.open('w')as stream:rc=subprocess.call(isolated_command(a.manifest,slot,a.physical_gpu,a.python,m['worktree'],root),stdout=stream,stderr=subprocess.STDOUT,cwd=m['worktree'])
        job=load(root/'evidence/jobs'/f'{slot:03d}.json')
        with(root/'PROGRESS.md').open('a')as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+' G 任意未跑房间自动'+a.required_sound_class+'构造，'+current['row']['scene_id']+' '+job['status']+' reason='+str(job.get('reason',''))+' family='+json.dumps(job.get('export',{}).get('question_family_counts',{}),ensure_ascii=False)+'\n')
        if rc==0:return
    return

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Unattended native Kujiale room construction, bounded retries and full smoke.

The input is an ordinary runtime/asset/room manifest, with no per-scene layout.
Each attempt is immutable. Existing passed rooms are counted from real readers.
"""
from __future__ import annotations
import argparse,copy,datetime,json,os,shlex,subprocess,time
from pathlib import Path
from avengine.timeline.v7_kujiale_scheduler import resources,can_start,assign_gpu,isolated_command

def load(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def enqueue(manifest_path,gpus):
    m=load(manifest_path);root=Path(m['output_root']);houses=load(m['houses_ue'])['houses'];covered=set();counts={};slots=[]
    for task in m['parents']:
        job=root/'evidence/jobs'/f'{task["slot"]:03d}.json'
        if job.exists()and load(job)['status']in {'passed','running'}:covered.add((task['row']['house'],task['row']['room_id']))
        if task.get('automatic_registered_room'):covered.add((task['row']['house'],task['row']['room_id']));slots.append(task['slot']);counts[task['ue_physical_gpu']]=counts.get(task['ue_physical_gpu'],0)+1
    runtime=load(m['parents'][0].get('runtime_inputs')or m['runtime_ue']);runtime.update(registered_question_workers=1,registered_question_cpu_ids=[12],registered_ue_capture_scale=1,suspend_unused_registered_capture_sensors=True,registered_offscreen_viewport_resolution_hw=[64,64],registered_actor_yaw_offset_deg=-90.,construct_registered_dialogue_camera=True)
    runtime_path=root/'inputs/runtime_automatic_rooms.json';write(runtime_path,runtime)
    for gpu in gpus:
        wrapper=root/'scripts'/f'ue_gpu{gpu}'
        if not wrapper.exists():
            arguments=[os.environ.get('GAP_PY',os.sys.executable),str(Path(m['worktree'])/'tools/ue/launch_kujiale_reserved_gpu.py'),'--output-root',str(root),'--physical-gpu',str(gpu)]
            wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec '+shlex.join(arguments)+' \"$@\"\n');wrapper.chmod(0o755)
    for house in houses:
        for room in house['checked_room_ids']:
            if (house['house'],room)in covered:continue
            slot=len(m['parents']);gpu=assign_gpu(gpus,counts);counts[gpu]=counts.get(gpu,0)+1;seed=2610157000+slot;scene=f'v7_gap_auto_{house["house"]}_{room}_a00';row={'house':house['house'],'room_id':room,'scene_id':scene,'scene_seed':seed,'construction_attempt':0,'question_quota_base':slot*20};m['parents'].append({'slot':slot,'family':'kujiale','device':'ue','profile':'registered','focus':'content_binding','automatic_registered_room':True,'ue_physical_gpu':gpu,'row':row,'runtime_inputs':str(runtime_path)});slots.append(slot)
    m['code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=m['worktree'],text=True).strip();write(manifest_path,m);return slots

def run(args):
    slots=enqueue(args.manifest,args.physical_gpus);m=load(args.manifest);root=Path(m['output_root']);state=root/'evidence/all_rooms_scheduler.json';children={};finished=set();retry_slots=set();deadline=datetime.datetime.fromisoformat(args.deadline_utc);last_progress=0
    while True:
        m=load(args.manifest)
        for slot,(child,stream)in list(children.items()):
            if child.poll()is None:continue
            stream.close();del children[slot];finished.add(slot)
            job_path=root/'evidence/jobs'/f'{slot:03d}.json'
            if job_path.exists():
                job=load(job_path);task=m['parents'][slot]
                if job['status']=='failed'and task['row']['construction_attempt']+1<args.max_attempts:
                    retry=copy.deepcopy(task);retry['slot']=len(m['parents']);retry['row']['construction_attempt']+=1;retry['row']['scene_id']=retry['row']['scene_id'].rsplit('_a',1)[0]+f'_a{retry["row"]["construction_attempt"]:02d}';m['parents'].append(retry);slots.append(retry['slot']);write(args.manifest,m)
        now=datetime.datetime.now(datetime.timezone.utc);snapshot=resources(root)
        pending=[s for s in slots if s not in finished and s not in children and not(root/'evidence/jobs'/f'{s:03d}.json').exists()]
        if pending and now<deadline and can_start(snapshot,active_parents=len(children),max_parents=args.max_parents):
            slot=pending[0];task=m['parents'][slot];log=root/'logs'/f'parent_{slot:03d}_automatic.log';stream=log.open('w');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(task['ue_physical_gpu']),AVENGINE_REGISTERED_QUESTION_WORKERS='1',AVENGINE_REGISTERED_QUESTION_CPUS='12,13,14,15');child=subprocess.Popen(isolated_command(args.manifest,slot,task['ue_physical_gpu'],args.python,m['worktree'],root),stdout=stream,stderr=subprocess.STDOUT,cwd=m['worktree'],env=env);children[slot]=(child,stream)
        status={'updated_utc':now.isoformat(),'pid':os.getpid(),'active':{str(s):p.pid for s,(p,_)in children.items()},'pending':pending,'finished_slots':sorted(finished),'resources':snapshot,'deadline_utc':args.deadline_utc,'stop_mode':'pause new launches; never kill running jobs'};write(state,status)
        if time.time()-last_progress>=900:
            with(root/'PROGRESS.md').open('a')as f:f.write(now.isoformat()+' G 无人值守逐房自动构造/重试；'+json.dumps(status,ensure_ascii=False)+'\n')
            last_progress=time.time()
        if (not pending or now>=deadline)and not children:break
        time.sleep(10)
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--physical-gpus',type=int,nargs='+',default=[2]);p.add_argument('--python',default=os.environ.get('GAP_PY','python'));p.add_argument('--deadline-utc',required=True);p.add_argument('--max-parents',type=int,default=8);p.add_argument('--max-attempts',type=int,default=8);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    if a.plan_only:print(json.dumps(enqueue(a.manifest,a.physical_gpus)))
    else:raise SystemExit(run(a))
if __name__=='__main__':main()

"""Wait outside P19 for personal headroom, then enqueue one checkpointed supplement."""
from __future__ import annotations
from pathlib import Path
import argparse,fcntl,hashlib,json,os,socket,subprocess,sys,time,uuid

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.'+uuid.uuid4().hex)
    with tmp.open('x')as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def enough_space(state,minimum=120_000_000_000):
    if minimum!=120_000_000_000:raise ValueError('owner 120GB gate cannot change')
    if state.get('status')!='pass':return False
    for key,limit in [('global_age_seconds',900),('own_age_seconds',60)]:
        age=state.get(key)
        if type(age)not in (int,float)or not 0<=age<=limit:return False
    value=state.get('estimated_owner_allocated_bytes')
    return type(value)is int and value>=0 and state.get('quota_bytes')==2_000_000_000_000 and state.get('reserve_bytes')==80_000_000_000 and 2_000_000_000_000-80_000_000_000-value>=minimum


def transform_queue(before,old_plan,new_plan=None):
    lines=before.splitlines(keepends=True);out=[];count=0
    for line in lines:
        tok=line.split(b'#',1)[0].split()
        if tok and tok[0]==str(old_plan).encode():
            out.append(b'# R19 hold own space-stopped r13 until >=120GB personal headroom; '+line);count+=1
        else:out.append(line)
    if count>1:raise ValueError('duplicate old queue entry; do not guess')
    if new_plan:
        if any(line.split(b'#',1)[0].split()[:1]==[str(new_plan).encode()]for line in lines):raise ValueError('new plan already queued')
        out.insert(0,(str(new_plan)+' 2 # R19 >=120GB measured personal headroom; resume 277 unstarted +2 interrupted; retain61; CPU40/.35; no STOP\n').encode())
    return b''.join(out),count

def compare_and_replace(queue,before,after):
    queue=Path(queue)
    with queue.open('rb')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if queue.read_bytes()!=before:raise ValueError('queue changed concurrently; no overwrite')
        info=queue.stat();tmp=queue.with_name('.v7_r19_queue_'+uuid.uuid4().hex)
        with tmp.open('xb')as f:f.write(after);f.flush();os.fsync(f.fileno())
        os.chmod(tmp,info.st_mode&0o777)
        if os.geteuid()==0:os.chown(tmp,info.st_uid,info.st_gid)
        os.replace(tmp,queue)

def validate_scope(c):
    owner=Path(c['owner_root']).resolve();h=owner/'hardbank';task=Path(c['task_root']).resolve();watch=Path(c['watch_root']).resolve()
    if task.parent.parent!=h/'generation'or task.name!='supplement_r16_1'or not task.parent.name.startswith('v7_s_'):raise ValueError('unexpected supplement scope')
    if watch.parent!=task or watch.name!='space_resume_r19':raise ValueError('watch escaped its task')
    if Path(c['queue']).resolve()!=h/'p19_queue.txt'or Path(c['runner']).resolve()!=h/'p19_run.sh':raise ValueError('shared queue/runner scope differs')
    for key in ('profile','new_plan','old_plan'):
        if not Path(c[key]).resolve().is_relative_to(task):raise ValueError('plan/profile escaped task')
    if c['minimum_headroom_bytes']!=120_000_000_000 or c['quota_bytes']!=2_000_000_000_000 or c['reserve_bytes']!=80_000_000_000:raise ValueError('owner capacity rules changed')
    if sha(c['runner'])!=c['runner_SHA256']:raise ValueError('P19 runner changed; recheck without modifying it')
    return task,watch

def quiet_task(task):
    if (task/'coordinator.lock').exists()or(task/'STOP').exists():raise ValueError('active/unknown coordinator or STOP: do not hand off')
    progress=json.loads((task/'PROGRESS.json').read_text())
    if progress.get('status')!='stopped'or progress.get('active_parents'):raise ValueError('previous production is not at a verified quiet breakpoint')

def dry(c,path,label,require_plan=False):
    watch=Path(c['watch_root']);env=dict(os.environ,P19_DRY='1',P19_QUEUE=str(path),LADDER_JOB_ID=c['owner_job_id'])
    p=subprocess.run(['bash',c['runner']],env=env,text=True,capture_output=True,timeout=120)
    log=watch/(label+'_'+str(time.time_ns())+'.log');log.write_text(p.stdout+p.stderr)
    if p.returncode:raise ValueError('P19 dry validation failed: '+str(log))
    if require_plan:
        previews=[]
        for line in p.stdout.splitlines():
            try:d=json.loads(line)
            except ValueError:continue
            if d.get('event')=='dispatch_preview':previews.append(d)
        active=Path(c['owner_root'])/'hardbank/tmp/p19_active'/c['plan_name']
        if not any(d.get('plan')==c['plan_name']for d in previews)and not active.is_file():raise ValueError('new plan missing from dry preview')
    if sha(c['runner'])!=c['runner_SHA256']:raise ValueError('runner changed during dry check')
    return str(log)

def retire_old(c):
    task,watch=validate_scope(c);watch.mkdir(exist_ok=True);quiet_task(task)
    q=Path(c['queue']);before=q.read_bytes();after,count=transform_queue(before,c['old_plan'])
    if count==0:return {'status':'old_plan_already_held','queue_changed':False}
    stamp=str(time.time_ns());backup=watch/('queue_before_hold_'+stamp+'.txt');candidate=watch/('queue_candidate_hold_'+stamp+'.txt');backup.write_bytes(before);candidate.write_bytes(after)
    one=dry(c,candidate,'dry_hold_candidate');quiet_task(task);compare_and_replace(q,before,after);two=dry(c,q,'dry_hold_actual')
    r={'status':'old_plan_held_not_newly_enqueued','only_own_old_line_changed':True,'other_lines_unchanged':True,'backup':str(backup),'dry_before':one,'dry_after':two,'STOP_used':False,'runner_SHA256':c['runner_SHA256']};atomic(watch/'HOLD_RECEIPT.json',r);return r

def sample(c,max_wait=1200):
    task,watch=validate_scope(c);watch.mkdir(exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from avengine.timeline.v7_owner_quota import CachedOwnerQuota
    begin=time.time();q=CachedOwnerQuota(c['owner_root'],task,watch/'quota_cache/personal_allocation.json',c['quota_bytes'],c['reserve_bytes']);q._schedule(force_global=True)
    state=None
    try:
        while True:
            state=q.poll();fresh=q.global_checked_this_run and state.get('global_age_seconds',float('inf'))<=900 and state.get('own_age_seconds',float('inf'))<=60
            value=state.get('estimated_owner_allocated_bytes');headroom=None if value is None else c['quota_bytes']-c['reserve_bytes']-value
            result={'status':'space_ready'if fresh and enough_space(state)else 'waiting_for_120GB','UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'host':socket.gethostname(),'pid':os.getpid(),'headroom_after_reserve_bytes':headroom,'minimum_headroom_bytes':c['minimum_headroom_bytes'],'new_personal_measurement_this_call':q.global_checked_this_run,'measurement_seconds':time.time()-begin,'quota_state':state,'queue_mutated':False,'GPU_started':False,'house_access':0}
            atomic(watch/'SPACE_STATUS.json',result)
            if fresh or time.time()-begin>=max_wait:return result
            time.sleep(5)
    finally:q.close()

def enqueue_ready(c,measurement):
    task,watch=validate_scope(c);receipt=watch/'QUEUE_RECEIPT.json';q=Path(c['queue']);plan=Path(c['new_plan']);profile=Path(c['profile'])
    if (watch/'PAUSE_SPACE_RESUME_R19').exists():return {'status':'paused_owner_marker','queue_mutated':False}
    if receipt.is_file():return json.loads(receipt.read_text())
    if measurement.get('status')!='space_ready'or not measurement.get('new_personal_measurement_this_call')or not enough_space(measurement['quota_state']):return {'status':'waiting_for_120GB','queue_mutated':False}
    observed=time.strptime(measurement['UTC'],'%Y-%m-%dT%H:%M:%SZ')
    if time.time()-__import__('calendar').timegm(observed)>900:raise ValueError('space sample expired before queue operation')
    quiet_task(task);p=json.loads(profile.read_text());spec=json.loads(plan.read_text())
    if spec['name']!=c['plan_name']or p['plan_name']!=c['plan_name']or p['production_work_directory']!='production_r12'or p['workers']!=40 or p['training_slowdown_threshold']!=.35:raise ValueError('resume plan/runtime rules changed')
    if sha(p['seeds'])!=c['seed_SHA256']:raise ValueError('original supplement seeds changed')
    if p.get('discard_material_cache_after_seal')is not False or not p.get('resume_checkpoint'):raise ValueError('R19 must retain intermediates and use checkpoint partition')
    before=q.read_bytes();tokens=[x.split(b'#',1)[0].split()[:1]for x in before.splitlines()]
    if [str(plan).encode()]in tokens:
        r={'status':'already_queued_recovered_after_commit','plan':str(plan),'queue_mutated':False,'original_measurement':measurement,'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};atomic(receipt,r);return r
    after,_count=transform_queue(before,c['old_plan'],plan);stamp=str(time.time_ns());backup=watch/('queue_before_enqueue_'+stamp+'.txt');candidate=watch/('queue_candidate_enqueue_'+stamp+'.txt');backup.write_bytes(before);candidate.write_bytes(after)
    one=dry(c,candidate,'dry_enqueue_candidate',True);quiet_task(task)
    if time.time()-__import__('calendar').timegm(observed)>900:raise ValueError('space sample aged while validating queue; remeasure')
    if (watch/'PAUSE_SPACE_RESUME_R19').exists():return {'status':'paused_owner_marker','queue_mutated':False}
    compare_and_replace(q,before,after);two=dry(c,q,'dry_enqueued_actual',True)
    r={'status':'queued','plan':str(plan),'profile':str(profile),'queue_position_zero_based':0,'backup':str(backup),'dry_before':one,'dry_after':two,'original_measurement':measurement,'producer_commit':p['generator_commit'],'seed_SHA256':c['seed_SHA256'],'resume_work_directory':'production_r12','CPU_workers_limit':40,'training_slowdown_threshold':.35,'STOP_used':False,'other_lines_unchanged':True,'runner_SHA256':c['runner_SHA256'],'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};atomic(receipt,r);return r

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--config',type=Path,required=True);a.add_argument('--operation',choices=['sample','hold_old','enqueue_if_ready'],required=True);args=a.parse_args();c=json.loads(args.config.read_text())
    if args.operation=='hold_old':r=retire_old(c)
    else:
        task,watch=validate_scope(c)
        if args.operation=='enqueue_if_ready'and(watch/'PAUSE_SPACE_RESUME_R19').exists():
            print(json.dumps({'status':'paused_owner_marker','queue_mutated':False}),flush=True);return
        if args.operation=='enqueue_if_ready'and(watch/'QUEUE_RECEIPT.json').is_file():r=json.loads((watch/'QUEUE_RECEIPT.json').read_text())
        else:
            measurement=sample(c);r=enqueue_ready(c,measurement)if args.operation=='enqueue_if_ready'else measurement
    print(json.dumps(r,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

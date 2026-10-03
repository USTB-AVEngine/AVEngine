#!/usr/bin/env python3
"""Rank-sharded, bounded CPU stages plus one render lane per selected GPU.

The plan contains argv lists, dependencies and output paths, not shell code.
Each interrupted stage reruns in a fresh attempt; only verified durable output
is skipped. Root ownership and a launch-level flock fence concurrent jobs.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from resources import GIB, allocate, probe, write_json

PHASES=('plan','render','audio','deliver')
STOP=False
CHILDREN={}
CURRENT_JOB=''
VERIFY_CACHE={}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def token(value):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',value):raise ValueError('invalid name: '+value)
    return value

def load(path):
    return json.loads(Path(path).read_text())

def contained(path, root):
    p=Path(path).resolve(); r=Path(root).resolve()
    if not p.is_relative_to(r):raise ValueError(f'path escapes output root: {p}')
    return p

def tree_bytes(directory):
    total=0
    if not Path(directory).exists():return total
    for parent,_,files in os.walk(directory,followlinks=False):
        for name in files:
            path=Path(parent)/name
            try:
                if not path.is_symlink():total+=path.stat().st_size
            except FileNotFoundError:pass
    return total

def storage_usage(root, scratch_budget, media_budget):
    # Counts interrupted attempts as well. Never deletes them to make room.
    usage={'scratch_bytes':tree_bytes(root/'scratch'),'delivery_bytes':tree_bytes(root/'worlds')}
    if usage['scratch_bytes']>scratch_budget or usage['delivery_bytes']>media_budget:
        raise RuntimeError('owned generation storage budget exceeded: '+json.dumps(usage))
    return usage

def files_ok(records, hashes=True):
    if not records:return False
    for r in records:
        p=Path(r['path'])
        if not p.is_file() or p.stat().st_size!=r['bytes']:return False
        key=(str(p),p.stat().st_mtime_ns,r['sha256'])
        if hashes and key not in VERIFY_CACHE:
            if sha(p)!=r['sha256']:return False
            VERIFY_CACHE[key]=True
    return True

def receipt(root, world, phase, plan_sha):
    p=root/'receipts'/world/(phase+'.json')
    if not p.is_file():return None
    r=load(p)
    return r if r.get('plan_sha256')==plan_sha and files_ok(r.get('files',[]), hashes=r.get('job')!=CURRENT_JOB) else None

def stop(signum,frame):
    global STOP
    STOP=True
    for pid,process in list(CHILDREN.items()):
        if process.pid==pid and process.poll() is None:
            try:os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError:pass

def execute(root, world, phase, spec, all_receipts, python, job, plan_sha, gpu, affinity, nice):
    attempt=contained(root/'scratch'/world/(phase+'-'+uuid.uuid4().hex),root)
    attempt.mkdir(parents=True)
    replacements={'python':python,'attempt':str(attempt),'world_root':str(root/'worlds'/world),'world':world,'gpu':gpu or ''}
    for key,r in all_receipts.items():
        replacements[key+'_root']=r['attempt']
    argv=[x.format_map(replacements) for x in spec['argv']]
    env=dict(os.environ)
    for k in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','GROUP_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','TORCHELASTIC_RUN_ID'):
        env.pop(k,None)
    env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','AVENGINE_RLR_THREADS')})
    env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=gpu or '',TMPDIR=str(attempt),CUDA_CACHE_PATH=str(attempt/'cuda_cache'),__GL_SHADER_DISK_CACHE_PATH=str(attempt/'gl_shader_cache'))
    prefix=['nice','-n',str(nice)]
    if affinity:prefix+=['taskset','-c',','.join(map(str,affinity))]
    log=root/'logs'/job/(world+'--'+phase+'--'+attempt.name+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
    started=time.time()
    with log.open('x') as out:
        process=subprocess.Popen(prefix+argv,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        CHILDREN[process.pid]=process
        if STOP:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        write_json(attempt/'process.json',{'pid':process.pid,'argv':prefix+argv,'host':socket.gethostname(),'job':job,'stage':phase,'started_unix':started})
        code=process.wait(); CHILDREN.pop(process.pid,None)
    if code:raise RuntimeError(f'{world}/{phase} exited {code}; {log}')
    outputs=[contained(Path(str(x).format_map(replacements)),root) for x in spec['outputs']]
    if phase=='deliver' and any(not p.is_relative_to(root/'worlds'/world) for p in outputs):raise ValueError('durable outputs must be outside scratch, in this world directory')
    if not outputs or any(not p.is_file() for p in outputs):raise RuntimeError('missing required outputs: '+str(outputs))
    records=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in outputs]
    for output in outputs:
        if output.name in ('stage.json','ARTIFACTS.json'):
            for item in load(output).get('files',[]):
                target=contained(item['path'],root/'worlds'/world if phase=='deliver' else root)
                if target.stat().st_size!=item['bytes'] or sha(target)!=item['sha256']:raise RuntimeError('stage artifact hash readback failed')
                records.append(item)
    r={'job':job,'world':world,'phase':phase,'attempt':str(attempt),'plan_sha256':plan_sha,'files':records,'wall_seconds':time.time()-started,'log':str(log),'gpu':gpu,'affinity':affinity,'rlr_threads':1}
    write_json(root/'receipts'/world/(phase+'.json'),r)
    return r

def main():
    global CURRENT_JOB
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--run-root',required=True);p.add_argument('--python',default=sys.executable)
    p.add_argument('--mode',choices=['dedicated','mixed','cpu'],default='dedicated');p.add_argument('--render-gpus',default=None)
    p.add_argument('--training-ranks',type=int,default=0);p.add_argument('--loader-workers',type=int,default=0);p.add_argument('--training-gib',type=float,default=33)
    p.add_argument('--worker-gib',type=float,default=4);p.add_argument('--nice',type=int,default=19)
    p.add_argument('--scratch-budget-gib',type=float,default=32);p.add_argument('--media-budget-gib',type=float,default=380000000000/GIB)
    p.add_argument('--timeout-seconds',type=int,default=14400);p.add_argument('--prune-owned-intermediates',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(); root=Path(a.run_root).resolve(); plan_path=Path(a.manifest).resolve(); plan=load(plan_path); plan_sha=sha(plan_path)
    release=Path(os.environ.get('CCI_DATAGEN_BASE',''))/'RELEASE.json'
    if plan.get('source_commit') and (not release.is_file() or load(release).get('source_commit')!=plan['source_commit']):raise ValueError('source Git version changed; prepare a new accepted manifest/run root')
    if plan.get('schema')!='cci_datagen_stages_v1':raise ValueError('unsupported generation manifest')
    if plan.get('formal_frozen_test'):raise ValueError('formal test panel production is not enabled before protocol freeze and owner handoff')
    worlds=plan['worlds']; ids=[token(w['world_id']) for w in worlds]
    if len(set(ids))!=len(ids):raise ValueError('duplicate world IDs')
    if any(w.get('split')!='train' for w in worlds) and plan.get('train_only',True):raise ValueError('pilot contains a held-out house')
    for w in worlds:
        if w.get('request') and (not w.get('request_sha256') or sha(w['request'])!=w['request_sha256']):raise ValueError('accepted request content changed; prepare a new manifest/run root')
        if set(w['stages'])!=set(PHASES):raise ValueError('world needs plan/render/audio/deliver stages')
        if w.get('scratch_bytes_peak',0)<=0 or w.get('delivery_bytes_upper',0)<=0:raise ValueError('declare per-world storage bounds')
        for phase,s in w['stages'].items():
            if not isinstance(s.get('argv'),list) or not s['argv'] or not s.get('outputs'):raise ValueError('stage requires argv and output readback')
    if sum(w['delivery_bytes_upper'] for w in worlds)>a.media_budget_gib*GIB:raise ValueError('manifest exceeds reserved delivery budget')
    if a.mode=='mixed' and (a.training_ranks<=0 or a.render_gpus is None or a.nice<10):raise ValueError('mixed mode requires explicit GPU reservation and training resource declarations')
    job=token(os.environ.get('DATAGEN_LAUNCH_ID',''));CURRENT_JOB=job
    rank=int(os.environ.get('RANK','0')); world_size=int(os.environ.get('WORLD_SIZE','1'))
    if not 0<=rank<world_size:raise ValueError('invalid torchrun rank')
    if os.environ.get('CCI_DATAGEN_BASE'):contained(root,os.environ['CCI_DATAGEN_BASE'])
    root.mkdir(parents=True,exist_ok=True)
    owner=root/'OWNER.json'
    if rank!=0:
        wait_until=time.monotonic()+30
        while not owner.exists():
            if time.monotonic()>wait_until:raise RuntimeError('rank 0 did not initialize run ownership')
            time.sleep(.1)
    if owner.exists():
        if load(owner).get('manifest_sha256')!=plan_sha:raise ValueError('run root belongs to another manifest')
    else:
        try:
            write_json(owner,{'manifest_sha256':plan_sha,'created_by':'cci_datagen'})
        except FileExistsError:
            if load(owner).get('manifest_sha256')!=plan_sha:raise
    report=probe(); reports_dir=root/'resources'/job; reports_dir.mkdir(parents=True,exist_ok=True)
    write_json(reports_dir/f'rank{rank:05d}.json',report)
    deadline=time.monotonic()+a.timeout_seconds
    while len(list(reports_dir.glob('rank*.json')))<world_size:
        if time.monotonic()>deadline:raise RuntimeError('timed out waiting for rank resource reports')
        time.sleep(.2)
    reports=[load(x) for x in sorted(reports_dir.glob('rank*.json'))]
    allocations=allocate(reports,mode=a.mode,training_ranks=a.training_ranks,loader_workers=a.loader_workers,training_gib=a.training_gib,worker_gib=a.worker_gib)
    host=report['host']; local=allocations[host]; cpus=local['worker_affinity'][str(rank)]
    host_rows=sorted([r for r in reports if r['host']==host],key=lambda r:int(r['ranks'].get('RANK') or 0))
    gpu_rows=report['gpus']['visible']; tokens=[r['index'] for r in gpu_rows]
    if a.render_gpus is not None:
        requested=a.render_gpus.split(',') if a.render_gpus else []
        if not set(requested)<=set(tokens):raise ValueError('requested GPU not visible in allocation')
        tokens=requested
    if a.mode=='cpu':tokens=[]
    render_ranks=[int(r['ranks'].get('RANK') or 0) for r in host_rows if local['cpu_workers'][str(int(r['ranks'].get('RANK') or 0))]>0][:len(tokens)]
    gpu=tokens[render_ranks.index(rank)] if rank in render_ranks else None
    count=len(cpus)
    cpu_ranks=sorted(int(r['ranks'].get('RANK') or 0) for r in reports if allocations[r['host']]['cpu_workers'][str(int(r['ranks'].get('RANK') or 0))]>0)
    if not cpu_ranks:raise ValueError('no CPU capacity remains after reservations')
    assignment={'rank':rank,'world_size':world_size,'cpu_processes':count,'render_gpu':gpu,'allocation':local,'cpu_shards':cpu_ranks,'shard_count':len(cpu_ranks)}
    write_json(reports_dir/f'allocation{rank:05d}.json',assignment)
    print(json.dumps(assignment),flush=True)
    if a.dry_run:return
    if a.mode!='cpu' and not any(r['gpus']['visible'] for r in reports):raise RuntimeError('GPU mode requested on a CPU-only host')
    if a.mode=='cpu' and any(not w.get('precompleted',{}).get('render') for w in worlds):raise ValueError('CPU-only mode requires pre-rendered capture receipts for every world')
    max_scratch=max(w['scratch_bytes_peak'] for w in worlds); slots=int(a.scratch_budget_gib*GIB//max_scratch)
    if slots<1:raise ValueError('scratch budget cannot fit one world')
    # Claims and slots live in the unique launch namespace; restart ignores
    # interrupted claims while completed receipts remain bound to the manifest.
    claims=root/'claims'/job;claims.mkdir(parents=True,exist_ok=True)
    slot_root=root/'slots'/job;slot_root.mkdir(parents=True,exist_ok=True)
    failures=[];running={};owned_slots={};finished=False;storage_check=0
    def take_slot(wid):
        if wid in owned_slots:return True
        for i in range(slots):
            d=slot_root/f'{i:05d}'
            try:d.mkdir()
            except FileExistsError:
                try:
                    if (d/'world').read_text()==wid:return True
                except FileNotFoundError:pass
                continue
            (d/'world').write_text(wid);owned_slots[wid]=d;return True
        return False
    def release_slot(wid):
        for d in slot_root.iterdir():
            if (d/'world').exists() and (d/'world').read_text()==wid:
                (d/'world').unlink();d.rmdir()
        owned_slots.pop(wid,None)
    for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,stop)
    pool=ThreadPoolExecutor(max_workers=max(1,count)); render_pool=ThreadPoolExecutor(max_workers=1)
    try:
        while not STOP:
            if time.monotonic()>=storage_check:
                usage=storage_usage(root,a.scratch_budget_gib*GIB,a.media_budget_gib*GIB)
                write_json(reports_dir/f'storage{rank:05d}.json',usage)
                storage_check=time.monotonic()+10
            for key,(future,claim) in list(running.items()):
                if future.done():
                    try:future.result()
                    except Exception as e:failures.append(str(e));write_json(root/'failures'/job/(key+'.json'),{'error':str(e)})
                    del running[key];claim.unlink(missing_ok=True)
            complete=0
            for index,w in enumerate(worlds):
                wid=w['world_id'];done=root/'worlds'/wid/'COMPLETE.json'
                if done.exists() and load(done).get('plan_sha256')==plan_sha and files_ok(load(done).get('files',[]), hashes=load(done).get('job')!=CURRENT_JOB):
                    complete+=1;continue
                rs={}
                for phase in PHASES:
                    r=receipt(root,wid,phase,plan_sha)
                    if not r and phase in w.get('precompleted',{}):
                        r=w['precompleted'][phase]
                        if not files_ok(r.get('files',[]), hashes=r.get('job')!=CURRENT_JOB):raise ValueError('precompleted capture failed hash readback')
                    if r:rs[phase]=r
                if 'deliver' in rs:
                    write_json(done,{'world':wid,'job':job,'plan_sha256':plan_sha,'files':rs['deliver']['files']})
                    if a.prune_owned_intermediates:
                        scratch=contained(root/'scratch'/wid,root)
                        if scratch.is_dir():shutil.rmtree(scratch)
                    release_slot(wid);complete+=1;continue
                for phase in PHASES:
                    if phase in rs:continue
                    spec=w['stages'][phase]
                    deps=spec.get('depends_on',list(PHASES[:PHASES.index(phase)]))
                    if any(dep not in rs for dep in deps):continue
                    is_render=phase=='render'
                    if is_render:
                        if gpu is None or len(running)>=count or any('--render' in k for k in running):continue
                    elif cpu_ranks[index%len(cpu_ranks)]!=rank or count<1 or len(running)>=count:continue
                    key=wid+'--'+phase
                    if key in running or failures:continue
                    if not take_slot(wid):continue
                    claim=claims/key
                    try:
                        with claim.open('x') as f:json.dump({'rank':rank,'pid':os.getpid(),'host':host},f)
                    except FileExistsError:continue
                    executor=render_pool if is_render else pool
                    aff=cpus[-1:] if is_render else cpus[:count]
                    running[key]=(executor.submit(execute,root,wid,phase,spec,rs,a.python,job,plan_sha,gpu if is_render else None,aff,a.nice),claim)
                    break
            if complete==len(worlds):
                finished=True
                write_json(root/'complete'/f'{job}-rank{rank}.json',{'rank':rank,'worlds':complete})
                if rank==0:write_json(root/'ALL_DONE.json',{'worlds':complete,'manifest_sha256':plan_sha,'job':job})
                break
            if failures and not running:raise RuntimeError('; '.join(failures))
            if time.monotonic()>deadline:raise RuntimeError('generation timeout; unfinished stages are resumable')
            time.sleep(.5)
    finally:
        stop(signal.SIGTERM,None);pool.shutdown(wait=True,cancel_futures=True);render_pool.shutdown(wait=True,cancel_futures=True)
    if STOP and not finished:raise SystemExit(143)
if __name__=='__main__':main()

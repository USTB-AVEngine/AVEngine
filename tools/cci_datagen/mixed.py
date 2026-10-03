#!/usr/bin/env python3
"""One reserved renderer GPU plus explicit owner-selected training commands.

Invoke with the existing P19 launcher Python, without sourcing the generation
activation into the training environment. No queued/running experiment is read.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from resources import gpu_inventory,write_json
from generate import contained

p=argparse.ArgumentParser()
p.add_argument('--training-commands',required=True)
p.add_argument('--generation-manifest',required=True)
p.add_argument('--run-root',required=True)
p.add_argument('--loader-workers',type=int,required=True)
p.add_argument('--training-gib',type=float,default=33)
a=p.parse_args()
base=Path(os.environ.get('CCI_DATAGEN_BASE','/mnt/afs/xinyuanqian/jzy/hardbank/generation/cci_datagen_20261003_codex'))
if int(os.environ.get('SENSECORE_PYTORCH_NNODES','1'))!=1:
    raise ValueError('mixed training needs node-specific experiment handoff for multi-node jobs')
if not (base/'evidence/NATIVE_READY.json').is_file():
    raise ValueError('generation native environment has not passed its CPU preflight')
manifest=Path(a.generation_manifest).resolve()
if not manifest.is_file():raise ValueError('accepted generation manifest is missing')
commands=json.loads(Path(a.training_commands).read_text())
gpus=gpu_inventory()['visible']
if len(gpus)<2:raise ValueError('mixed mode needs a reserved renderer card and at least one training card')
if not commands or a.loader_workers<0:raise ValueError('explicit training commands and loader budget are required')
reserved=gpus[0]['index']
training_gpus=[r['index'] for r in gpus[1:]]
root=contained(a.run_root,base)
for row in commands:
    if not isinstance(row.get('argv'),list) or not row['argv'] or not 0<=row['gpu_slot']<len(training_gpus):
        raise ValueError('invalid training argv or GPU slot')
    if not Path(row['cwd']).resolve().is_relative_to('/mnt/afs/xinyuanqian/jzy'):
        raise ValueError('training cwd must stay in owner AFS root')
root.mkdir(parents=True,exist_ok=True)
import fcntl
lock=(root/'.supervisor.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
source=Path(__file__).resolve().parent
children=[]
interrupted=False
rank_keys=('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','GROUP_RANK','GROUP_WORLD_SIZE','TORCHELASTIC_RUN_ID')
parent_env={k:v for k,v in os.environ.items() if k not in rank_keys}
job='mixed_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+str(os.getpid())
gen_env=dict(parent_env,CCI_DATAGEN_BASE=str(base),DATAGEN_LAUNCH_ID=job,DATAGEN_MANIFEST=str(manifest),DATAGEN_RUN_ROOT=str(root/'generation'))
# Validate the whole manifest/resource declaration before starting training.
validation=[str(base/'env/bin/python'),'-B',str(source/'generate.py'),'--manifest',str(manifest),'--run-root',str(root/'generation'),'--dry-run','--mode','mixed','--render-gpus',reserved,'--training-ranks',str(len(commands)),'--loader-workers',str(a.loader_workers),'--training-gib',str(a.training_gib)]
subprocess.run(validation,env=gen_env,check=True)
scene=base/'mirror/data/datasets/habitat_data/scene_datasets/hm3d/train/00059-kJxT5qssH4H/kJxT5qssH4H.glb'
if not scene.is_file():raise ValueError('renderer preflight scene is missing')
preflight_env=dict(gen_env)
preflight_env['LD_LIBRARY_PATH']=str(base/'private_glvnd')+':'+str(base/'env/lib')
preflight_env['PYTHONPATH']=str(base/'code/src')+':'+str(base/'code')+':'+str(base/'overlay')
subprocess.run([str(base/'env/bin/python'),'-B',str(source/'native_preflight.py'),'--render','--gpu-device-id',reserved,'--scene',str(scene),'--output',str(root/'egl.json')],env=preflight_env,check=True)

def terminate_owned(signum=None,frame=None):
    global interrupted
    if signum is not None:interrupted=True
    for child in children:
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass

for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,terminate_owned)
try:
    for i,row in enumerate(commands):
        env=dict(parent_env)
        env.update(row.get('env',{}))
        env['CUDA_VISIBLE_DEVICES']=training_gpus[row['gpu_slot']]
        log=root/f'train-{i}.log'
        with log.open('a') as f:
            child=subprocess.Popen(row['argv'],cwd=row['cwd'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child)
        write_json(root/f'train-{i}.process.json',{'pid':child.pid,'argv':row['argv'],'gpu':env['CUDA_VISIBLE_DEVICES'],'log':str(log)})
    argv=['bash',str(source/'run_p19.sh'),'pilot','--mode','mixed','--render-gpus',reserved,'--training-ranks',str(len(commands)),'--loader-workers',str(a.loader_workers),'--training-gib',str(a.training_gib),'--nice','19','--prune-owned-intermediates']
    log=root/'generation.log'
    with log.open('a') as f:
        child=subprocess.Popen(argv,env=gen_env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(child)
    write_json(root/'generation.process.json',{'pid':child.pid,'argv':argv,'render_gpu':reserved,'log':str(log)})
    while not interrupted:
        codes=[child.poll() for child in children]
        if any(code is not None and code!=0 for code in codes):
            write_json(root/'FAILED.json',{'exit_codes':codes});raise SystemExit(1)
        if all(code==0 for code in codes):
            write_json(root/'COMPLETE.json',{'exit_codes':codes});break
        time.sleep(1)
    if interrupted:raise SystemExit(143)
finally:
    terminate_owned()
    deadline=time.monotonic()+20
    while any(child.poll() is None for child in children) and time.monotonic()<deadline:time.sleep(.2)
    for child in children:
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        child.wait()

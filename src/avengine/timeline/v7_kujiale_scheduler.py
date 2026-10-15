"""Production resource admission for complete native room parents.

A failed/resource-limited request is retained and queued. No running process is
stopped by this module. Card identities come from the explicit task policy.
"""
from __future__ import annotations
import datetime,json,os,shutil,subprocess
from pathlib import Path
from avengine.timeline.v7_storage_budget import task_disk_policy
from avengine.capture.kujiale_gpu_queue import own_pss

def resources(root):
    root=Path(root);available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:'));disk=task_disk_policy(root,100000000000)
    return {'MemAvailable_bytes':available,'data_free_bytes':shutil.disk_usage(root).free,'minimum_disk_free_bytes':disk['minimum_free_bytes'],'PSS_bytes':own_pss(root),'new_parent_memory_reserve_bytes':4000000000}

def can_start(snapshot,*,active_parents,max_parents=8):
    return active_parents<max_parents and snapshot['MemAvailable_bytes']>=30000000000 and snapshot['PSS_bytes']+snapshot.get('new_parent_memory_reserve_bytes',0)<=40000000000 and snapshot['data_free_bytes']>=snapshot['minimum_disk_free_bytes']

def assign_gpu(gpus,counts):
    if not gpus or any(g not in [1,2,3]for g in gpus):raise ValueError('explicit physical GPU1/2/3 scope required; GPU0 excluded')
    return min(gpus,key=lambda gpu:(counts.get(gpu,0),gpu))

def isolated_command(manifest,slot,gpu,python,worktree,root):
    root=Path(root);worktree=Path(worktree);command=['nice','-n',str(max(0,10-os.getpriority(os.PRIO_PROCESS,0))),'/usr/bin/bwrap','--unshare-user','--ro-bind','/','/','--dev','/dev','--proc','/proc','--bind',str(root),str(root),'--bind',str(root/'ue_tmp'),'/tmp','--bind',str(root/'ue_shm'),'/dev/shm','--bind',str(root/'ue_config'),str(Path.home()/'.config')]
    for name in ['nvidiactl','nvidia'+str(gpu),'nvidia-uvm','nvidia-uvm-tools']:command+=['--dev-bind','/dev/'+name,'/dev/'+name]
    return command+['--chdir',str(worktree),python,str(worktree/'tools/timeline/run_v7_acceptance.py'),'--manifest',str(manifest),'--worker',str(slot)]

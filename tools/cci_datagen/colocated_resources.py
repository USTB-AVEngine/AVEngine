#!/usr/bin/env python3
"""Observe the scheduler lease and colocated ranks without executing in them."""
import copy,math,os,re,socket
from pathlib import Path
from resources import GIB,probe,cgroup_limits,read,cgroup_spare
RANK_KEYS=('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','GROUP_WORLD_SIZE','SENSECORE_PYTORCH_NNODES','SENSECORE_PYTORCH_NODE_RANK','TORCHELASTIC_RUN_ID','LADDER_JOB_ID','LADDER_WORKERS_PER_GPU','CUDA_VISIBLE_DEVICES')

def process_env(pid):
 try:
  values=Path(f'/proc/{int(pid)}/environ').read_bytes().split(b'\0')
  return {k.decode():v.decode(errors='replace') for item in values if b'=' in item for k,v in [item.split(b'=',1)] if k.decode(errors='replace') in RANK_KEYS}
 except (OSError,ValueError):return {}

def worker_processes(parent_env):
 job=parent_env.get('LADDER_JOB_ID') or parent_env.get('TORCHELASTIC_RUN_ID')
 result=[]
 if not job:return result
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   argv=(p/'cmdline').read_bytes().split(b'\0')
   if not any(re.fullmatch(rb'.*/p19_worker(?:\.[0-9a-f]+)?\.py',arg) for arg in argv):continue
   env=process_env(int(p.name))
   if (env.get('LADDER_JOB_ID') or env.get('TORCHELASTIC_RUN_ID'))!=job:continue
   if 'LOCAL_RANK' not in env:continue
   result.append((int(p.name),env))
  except OSError:continue
 return sorted(result,key=lambda item:int(item[1].get('RANK',item[1]['LOCAL_RANK'])))

def child_loader_counts(workers,exclude_parent):
 # pt_data_worker is PyTorch's process name. This is a lower bound; defaults
 # also reserve loaders that may start after this observation.
 parents={};loader_pids=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   status=(p/'status').read_text();ppid=int(next(l.split()[1] for l in status.splitlines() if l.startswith('PPid:')))
   parents[int(p.name)]=ppid
   if (p/'comm').read_text().strip()=='pt_data_worker':loader_pids.append(int(p.name))
  except (OSError,StopIteration,ValueError):continue
 counts={pid:0 for pid,_ in workers if pid!=exclude_parent}
 for child in loader_pids:
  seen=set();pid=child
  while pid in parents and pid not in seen:
   seen.add(pid);pid=parents[pid]
   if pid in counts:counts[pid]+=1;break
 return counts

def observe(afs_root,io=False):
 parent=os.getppid();penv=process_env(parent)
 argv=(read(f'/proc/{parent}/cmdline') or '').replace('\x00',' ')
 leased=bool(re.search(r'/p19_worker(?:\.[0-9a-f]+)?\.py',argv) and penv.get('LOCAL_WORLD_SIZE') and penv.get('LOCAL_RANK'))
 local_world=int(penv.get('LOCAL_WORLD_SIZE',os.environ.get('LOCAL_WORLD_SIZE','1')))
 local_rank=int(penv.get('LOCAL_RANK',os.environ.get('LOCAL_RANK','0')))
 current=probe(afs_root,io);current['ranks'].update({k:v for k,v in penv.items() if k in current['ranks']})
 workers=worker_processes(penv) if leased else []
 rows=[]
 for pid,env in workers:
  # A sibling may finish between the process listing and this snapshot.
  try:
   raw,limits=cgroup_limits(pid);cpus=sorted(os.sched_getaffinity(pid))
  except (ProcessLookupError,PermissionError):continue
  row=copy.deepcopy(current);row.update(pid=pid,observer_pid=os.getpid(),observation_source='read-only /proc snapshot, not code executed by this rank',affinity_cpus=cpus,cgroup_raw=raw,cgroup_limits=limits,afs_io=None,afs_io_shared_record='observer.json')
  row['effective_cpu_cores']=min([len(cpus)]+[r['cpu_quota_cores'] for r in limits if 'cpu_quota_cores' in r])
  row['memory_limit_bytes']=min([current['memory_info']['MemTotal']]+[r['memory_limit_bytes'] for r in limits if 'memory_limit_bytes' in r])
  row['memory_available_bytes']=min([current['memory_info']['MemAvailable'],row['memory_limit_bytes']]+[cgroup_spare(r) for r in limits if 'memory_limit_bytes' in r and r.get('memory.current')])
  row['ranks'].update({k:v for k,v in env.items() if k in row['ranks']});row['nproc_observed_affinity_count']=len(cpus);row['nproc']['provenance']='command executed by observer; observed rank affinity count is separate'
  per_gpu=int(env.get('LADDER_WORKERS_PER_GPU','1'));cards=local_world//per_gpu
  row['scheduler_gpu_index_in_visible_list']=int(env['LOCAL_RANK'])%cards if cards else None
  visible=env.get('CUDA_VISIBLE_DEVICES')
  allowed=[g for g in current['gpus']['all'] if Path('/dev/nvidia'+g['index']).exists() and g['memory_mib'].replace('.','',1).isdigit()]
  if visible is not None:allowed=[g for g in allowed if g['index'] in visible.split(',') or any(g['uuid'].startswith(t) for t in visible.split(',') if t)]
  row['gpus'].update(visible=allowed,visible_count=len(allowed),CUDA_VISIBLE_DEVICES=visible,provenance='read-only observer NVML, peer process env and device access; lease index inferred from unchanged worker mapping')
  rows.append(row)
 counts=child_loader_counts(workers,parent)
 return {'observer':current,'rank_observations':rows,'scheduler_lease':{'verified_parent_worker':leased,'parent_pid':parent,'global_rank':int(penv.get('RANK',str(local_rank))),'local_rank':local_rank,'world_size':int(penv.get('WORLD_SIZE',str(local_world))),'local_world_size':local_world,'nodes':int(penv.get('GROUP_WORLD_SIZE',penv.get('SENSECORE_PYTORCH_NNODES','1'))),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES',''),'rank_env_source':f'/proc/{parent}/environ (allowlisted keys only)','observed_local_ranks':len(rows),'unobserved_local_ranks':max(0,local_world-len(rows))},'observed_loader_counts':counts}

def capacity(observation,policy):
 r=observation['observer'];lease=observation['scheduler_lease'];peers=[p for p in observation['rank_observations'] if p['pid']!=lease['parent_pid']]
 training=max(0,lease['local_world_size']-1);unknown=max(0,training-len(peers))
 loaders=max(int(policy.get('loader_workers_per_training_rank',4)),max(observation.get('observed_loader_counts',{}).values(),default=0))
 cpu_fraction=float(policy.get('cpu_reserve_fraction',.10));mem_fraction=float(policy.get('memory_reserve_fraction',.10));train_mem=float(policy.get('training_gib_per_rank',33))*GIB;worker_mem=float(policy.get('worker_gib',4))*GIB
 def shared(scope,field):
  return unknown+sum(any(x.get('canonical_path',x['path'])==scope and field in x for x in p['cgroup_limits']) for p in peers)
 bounds=[]
 # Affinity is a permission boundary. Cgroup bounds below decide shared vs
 # private quotas; an isolated generation cgroup does not inherit peer costs.
 affinity_train=unknown+sum(bool(set(p['affinity_cpus'])&set(r['affinity_cpus'])) for p in peers)
 cores=len(r['affinity_cpus']);bounds.append({'scope':'affinity','cores':cores,'training_ranks':affinity_train,'reserve':max(1,math.ceil(cores*cpu_fraction)),'usable':max(0,math.floor(cores)-max(1,math.ceil(cores*cpu_fraction))-affinity_train*loaders)})
 memory_caps=[r['memory_available_bytes'],int(r['memory_info']['MemTotal']*(1-mem_fraction))-training*train_mem]
 for x in r['cgroup_limits']:
  scope=x.get('canonical_path',x['path'])
  if 'cpu_quota_cores' in x:
   c=x['cpu_quota_cores'];n=shared(scope,'cpu_quota_cores');res=max(1,math.ceil(c*cpu_fraction));bounds.append({'scope':scope,'cores':c,'training_ranks':n,'reserve':res,'usable':max(0,math.floor(c)-res-n*loaders)})
  if 'memory_limit_bytes' in x:
   n=shared(scope,'memory_limit_bytes');memory_caps.append(int(x['memory_limit_bytes']*(1-mem_fraction))-n*train_mem)
 spare=max(0,int(min(memory_caps)));count=min(min(x['usable'] for x in bounds),int(spare//worker_mem));cpus=r['affinity_cpus'][:count]
 return {'cpu_processes':count,'affinity_cpus':cpus,'effective_cpu_cores':r['effective_cpu_cores'],'cpu_bounds':bounds,'other_training_ranks':training,'loader_workers_per_training_rank':loaders,'loader_estimate_source':'max(policy conservative default, locally observed pt_data_worker count); main training overhead is in CPU reserve','training_memory_bytes_per_rank':int(train_mem),'generation_memory_available_bytes':spare,'worker_memory_budget_bytes':int(worker_mem),'nice':19,'gpu_lease':lease['gpu'],'generation_scheduler_ranks':1,'torchrun_children_launched':0,'training_processes_modified':0}

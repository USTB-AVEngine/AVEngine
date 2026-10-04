#!/usr/bin/env python3
"""Measure full native UE+RLR+question pilots under bounded GPU concurrency."""
from __future__ import annotations
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
from copy import deepcopy
import psutil

def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')

def choose_topology(snapshot, instances, measured_peak_mib):
 if instances not in (1,2,4,8):raise ValueError('supported concurrency is 1, 2, 4 or 8')
 if measured_peak_mib<=0:raise ValueError('measured peak must be positive')
 ordered=sorted((g for g in snapshot if g['free_mib'] is not None),key=lambda g:g['free_mib'],reverse=True)
 single=[g for g in ordered if g['free_mib']>=instances*measured_peak_mib+4096]
 if single:return [single[0]['index']]*instances
 if instances==8:
  four=[g for g in ordered if g['free_mib']>=4*measured_peak_mib+4096]
  if len(four)>=2:return [four[0]['index']]*4+[four[1]['index']]*4
 raise RuntimeError(f'n{instances} has no authorized topology with measured peaks plus 4GiB per card')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--rounds',type=int,nargs='+',default=[1,2,4,8]);p.add_argument('--output-name',default='concurrency_original_v1');p.add_argument('--source-profile',type=Path);a=p.parse_args()
 if not a.output_name.replace('_','').isalnum():raise ValueError('output name must be alphanumeric/underscore')
 o=a.root.resolve();u=o/'ue_v1';r=o/'ue_round7';repo=Path(__file__).resolve().parents[2];py=os.sys.executable
 import sys;sys.path.insert(0,str(repo))
 from tools.rooms.v7_kujiale_ue_guard import gpu_snapshot,logical_bytes
 root=r/a.output_name;root.mkdir(exist_ok=True)
 restore=r/'evidence/original_ue_material_restore.json'
 native_profile=read(a.source_profile)['input']['ue_config'] if a.source_profile else None
 if read(restore)['status']!='pass' or read(restore)['restored_mic_count']!=75:raise ValueError('original native material restoration is incomplete')
 all_rounds=[];measured_peak_mib=3876.
 for n in a.rounds:
  d=root/f'n{n}';d.mkdir(exist_ok=False);initial=gpu_snapshot()
  # Shared allocations change between rounds: select currently feasible cards,
  # using the measured maximum and a 4GiB reserve for every participating card.
  topology=choose_topology(initial,n,measured_peak_mib)
  for gpu in set(topology):
   count=topology.count(gpu);free=initial[gpu]['free_mib']
   if free is None or free<count*measured_peak_mib+4096:raise RuntimeError(f'n{n} GPU{gpu} has insufficient aggregate reserve')
  # The complete 30 s original-material pilot grew this task root by 189 MB
  # at its sampled peak. This 500 MB per-worker startup estimate keeps more
  # than twice that measured footprint. Individual guards retain their
  # original 750 MB task-cap buffer, the 80 GB cap and 150 GB disk floor.
  storage_reserve_bytes=500000000
  if logical_bytes(o)+n*storage_reserve_bytes>80000000000:raise RuntimeError('task storage reserve insufficient')
  if __import__('shutil').disk_usage(o).free-n*storage_reserve_bytes<150000000000:raise RuntimeError('external audio storage reserve insufficient')
  began=time.monotonic();jobs=[];cpu_io={};global_samples=[]
  receipt={'parallel_instances':n,'initial_gpu':initial,'topology_gpu_indices':topology,'storage_startup_estimate_bytes_per_worker':storage_reserve_bytes,'minimum_shared_disk_free_bytes':150000000000,'scope':'full unchanged v3 30s two-human speech research pilot, cold per-job RIR cache and warm task-local DDC; not formal v4 quota admission','original_material_restore_receipt':str(restore),'source_profile':str(a.source_profile) if a.source_profile else None,'native_animation_sampling_unpaused':bool(native_profile and native_profile.get('unpause_animation_sampling')),'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':[]}
  try:
   for i,gpu in enumerate(topology):
    h='0022' if i%2==0 else '0030';job=d/f'job{i}';job.mkdir();base=r/f'generation_inputs/kujiale_{h}_original_v1';oldbase=u/f'generation_inputs/kujiale_{h}_production_v2'
    house=read(base/'house.json');house['input']['ue_config'].update(graphics_adapter=gpu,rpc_port=0)
    if native_profile:
     house['input']['ue_config'].update({k:v for k,v in native_profile.items() if k in ('native_source_material_bindings','original_material_restore_receipt','unpause_animation_sampling','pose_settle_native_frames','multi_view_foot_contact')})
    row=read(base/'seed_row.json');row['scene_id']=f'v7_kujiale_native_benchmark_{a.output_name}_n{n}_job{i}_{h}'
    write(job/'house.json',house);write(job/'seed_row.json',row)
    q=read(u/f'scripts/full_pilot_{h}_guard_request_v3.json')
    q['command']=[s.replace(str(oldbase/'house.json'),str(job/'house.json')).replace(str(oldbase/'seed_row.json'),str(job/'seed_row.json')).replace(str(u/f'generation/kujiale_{h}_full_pilot_v3'),str(job/'generation')) for s in q['command']]
    q.update(gpu_index=gpu,previous_peak_mib=measured_peak_mib,run_directory=str(job/'guard'),cpu_affinity=list(range(32+8*i,40+8*i)),storage_limit_bytes=80000000000,minimum_disk_free_bytes=150000000000,allowed_auxiliary_gpu_indices=[k for k in range(4) if k!=gpu],maximum_auxiliary_gpu_mib=64,maximum_seconds=7200)
    q['environment']['OMP_NUM_THREADS']='8';q['environment']['PYTHONPATH']=f'{repo}/src:{repo}:{u}/python_pkgs';qpath=job/'guard_request.json';write(qpath,q)
    log=(job/'coordinator.log').open('x');proc=subprocess.Popen([py,str(repo/'tools/rooms/v7_kujiale_ue_guard.py'),'--request',str(qpath)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    jobs.append({'process':proc,'log':log,'directory':job,'gpu':gpu,'start':time.monotonic(),'pid_create_time':psutil.Process(proc.pid).create_time()})
    write(d/'launch.json',{'jobs':[{'pid':j['process'].pid,'create_time':j['pid_create_time'],'request':str(j['directory']/'guard_request.json')} for j in jobs]})
   with (d/'system_samples.jsonl').open('x') as stream:
    while any(j['process'].poll() is None for j in jobs):
     sample={'elapsed_seconds':time.monotonic()-began,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu,utilization.memory,power.draw','--format=csv,noheader,nounits'],text=True),'system_cpu_percent':psutil.cpu_percent(interval=None),'memory':psutil.virtual_memory()._asdict(),'swap':psutil.swap_memory()._asdict(),'load_average':os.getloadavg(),'disk_io_system':psutil.disk_io_counters()._asdict(),'task_owned_process_io':[]}
     for j in jobs:
      try:
       parent=psutil.Process(j['process'].pid)
       if parent.create_time()!=j['pid_create_time']:continue
       for proc in parent.children(recursive=True):
        try:
         key=f'{proc.pid}:{proc.create_time()}';io=proc.io_counters();cpu=proc.cpu_times()
         cpu_io[key]={'pid':proc.pid,'create_time':proc.create_time(),'read_bytes':io.read_bytes,'write_bytes':io.write_bytes,'cpu_seconds':cpu.user+cpu.system,'rss_bytes':proc.memory_info().rss}
         sample['task_owned_process_io'].append(cpu_io[key])
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
      except psutil.NoSuchProcess:pass
     stream.write(json.dumps(sample)+'\n');stream.flush();time.sleep(1)
   for i,j in enumerate(jobs):
    j['log'].close();guard=read(j['directory']/'guard/result.json');pilot=read(j['directory']/'generation/result.json') if (j['directory']/'generation/result.json').exists() else {}
    receipt['jobs'].append({'job_index':i,'gpu_index':j['gpu'],'guard_status':guard['status'],'pilot_status':pilot.get('status'),'guard_machine_seconds':guard['elapsed_seconds'],'pilot_machine_seconds':pilot.get('elapsed_seconds'),'owned_cpu_seconds_lower_bound':guard.get('owned_tree_cpu_seconds_lower_bound'),'peak_mib':guard['peak_mib'],'pilot_output_bytes':pilot.get('logical_output_bytes'),'windows':len(pilot.get('windows',[])),'guard_result':str(j['directory']/'guard/result.json'),'pilot_result':str(j['directory']/'generation/result.json'),'error':pilot.get('error')})
   receipt.update(status='completed' if all(x['guard_status']=='pass' and x['pilot_status']=='complete_research_candidate' for x in receipt['jobs']) else 'failed',group_machine_seconds=time.monotonic()-began,task_owned_io_lower_bound={'read_bytes':sum(x['read_bytes'] for x in cpu_io.values()),'write_bytes':sum(x['write_bytes'] for x in cpu_io.values())})
   receipt['completed_scenes']=sum(x['pilot_status']=='complete_research_candidate' for x in receipt['jobs']);receipt['scenes_per_hour']=3600*receipt['completed_scenes']/receipt['group_machine_seconds']
   receipt['mean_cpu_occupied_cores_lower_bound']=sum(x['owned_cpu_seconds_lower_bound'] or 0 for x in receipt['jobs'])/receipt['group_machine_seconds']
  finally:
   # Only owned guards receive signals. Each guard verifies its own complete
   # PID/create-time/command tree before stopping child editors.
   for j in jobs:
    if j['process'].poll() is None:
     proc=psutil.Process(j['process'].pid)
     if proc.create_time()==j['pid_create_time'] and str(j['directory']/'guard_request.json') in proc.cmdline():proc.terminate()
   for j in jobs:
    if j['process'].poll() is None:j['process'].wait(timeout=45)
   receipt.setdefault('group_machine_seconds',time.monotonic()-began);write(d/'result.json',receipt)
  measured_peak_mib=max(measured_peak_mib,max((j['peak_mib'] for j in receipt.get('jobs',[])),default=0))
  all_rounds.append(receipt);write(root/'live_result.json',{'rounds':all_rounds})
  with (o/'PROGRESS_zh.md').open('a') as f:f.write('\n\n## '+datetime.datetime.now(datetime.timezone.utc).isoformat()+f' {n}实例并行全场景实测完成\n\n- {receipt.get("status")}，拓扑{topology}，整批{receipt["group_machine_seconds"]:.3f}秒，实际完成{receipt.get("completed_scenes",0)}场景，吞吐{receipt.get("scenes_per_hour",0):.3f}场景/小时。来源ue_round7/{a.output_name}/n{n}/result.json；仅旧v3研究协议，未代表完整v4配额准入。\n')
  print(json.dumps({k:v for k,v in receipt.items() if k not in ('initial_gpu','jobs')},ensure_ascii=False),flush=True)
  if receipt['status']!='completed':
   write(root/'result.json',{'rounds':all_rounds,'status':'failed'})
   raise RuntimeError('concurrency round failed; dependent rounds were not launched')
 write(root/'result.json',{'rounds':all_rounds,'status':'completed'})

if __name__=='__main__':main()

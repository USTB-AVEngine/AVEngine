#!/usr/bin/env python3
"""One ordinary scheduler experiment: observe its lease, then run CPU generation.

The existing scheduler's eval receipt ABI is used for a generic job. The
checkpoint argument is an explicit generator-profile JSON, never a model.
"""
import argparse,fcntl,json,os,signal,socket,subprocess,sys,time,uuid
from pathlib import Path
from resources import write_json,GIB
from generate import sha,tree_bytes,contained
from colocated_resources import observe,capacity
from space_budget import pool_allowance

CHILD=None

def stop(sig,frame):
 if CHILD is not None and CHILD.poll() is None:
  try:os.killpg(CHILD.pid,signal.SIGTERM)
  except ProcessLookupError:pass

def owned_run(argv,log,gpu=None,restart_root=None):
 global CHILD
 from gpu_guard import GPUWatch,require_headroom
 if gpu:require_headroom(gpu)
 watch=None
 with Path(log).open('x') as out:
  CHILD=subprocess.Popen(argv,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  write_json(Path(log).with_suffix('.process.json'),{'pid':CHILD.pid,'host':socket.gethostname(),'argv':argv})
  if gpu:watch=GPUWatch(gpu,CHILD,Path(log).with_suffix('.gpu_memory.json'),strict_ownership=False).start()
  intentional_restart=None
  if restart_root is not None:
   from trial_limits import monitor_generation
   intentional_restart=monitor_generation(CHILD,Path(restart_root),Path(log),argv)
  rc=CHILD.wait()
 if watch:watch.close()
 CHILD=None
 if intentional_restart:
  if rc not in (143,-signal.SIGTERM):raise RuntimeError(f'intended restart returned unexpected code {rc}: {log}')
  return intentional_restart
 if rc:raise RuntimeError(f'owned command exited {rc}: {log}')

def finish(output,mode,result,profile_sha):
 ident='cci_datagen_'+mode
 record={'id':ident,'task_kind':'resource_probe' if mode=='probe' else 'dataset_generation_pilot','generator_profile_sha256':profile_sha,'is_model_evaluation':False}
 (output/'input_manifest.jsonl').write_text(json.dumps(record)+'\n')
 (output/'predictions.jsonl').write_text(json.dumps({'id':ident,'status':'complete','result':result})+'\n')
 write_json(output/'run.json',{'status':'complete','count':1,'task_kind':record['task_kind'],'is_model_evaluation':False,'completed_worlds':result.get('completed_worlds',0),'receipt_format':'existing ordinary scheduler generic job ABI; no model predictions or checkpoint are created'})

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['probe','pilot'],required=True);p.add_argument('--generator-profile',required=True);p.add_argument('--output',required=True);p.add_argument('--standalone-probe',action='store_true')
 a=p.parse_args();profile_path=Path(a.generator_profile).resolve();profile=json.loads(profile_path.read_text());out=Path(a.output).resolve()
 base=Path(profile['base']).resolve()
 if not out.is_relative_to(base):raise ValueError('scheduler output must stay in personal generation root')
 out.mkdir(parents=True,exist_ok=False)
 from trial_limits import identity
 write_json(out/'coordinator.process.json',{'host':socket.gethostname(),**identity(os.getpid()),'signal_target':'this normal_entry PID only; never the shared worker/torchrun'})
 try:
  os.nice(19)
  for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
  observation=observe(out/'afs_io',io=True);lease=observation['scheduler_lease']
  if not lease['verified_parent_worker'] and not (a.standalone_probe and a.mode=='probe'):raise ValueError('ordinary worker parent lease was not verified; refuse generation')
  assignment=capacity(observation,profile.get('resources',{}));write_json(out/'observer.json',observation['observer']);write_json(out/'scheduler_lease.json',lease);write_json(out/'resource_assignment.json',assignment)
  for rank in observation['rank_observations']:
   write_json(out/'resources'/f"rank{int(rank['ranks'].get('RANK') or rank['ranks'].get('LOCAL_RANK') or 0):05d}.json",rank)
  write_json(out/'probe_summary.json',{'lease':lease,'assignment':assignment,'observed_rank_count':len(observation['rank_observations']),'egl_status':'unverified CPU-only metadata probe; the leased GPU pilot checks actual EGL','generator_profile_sha256':sha(profile_path),'peer_probe_limit':'same node process namespace only; no commands run inside training ranks; unobservable ranks are recorded explicitly'})
  if a.mode=='probe':finish(out,a.mode,{'completed_worlds':0,'probe_summary':str(out/'probe_summary.json')},sha(profile_path));return
  if profile.get('production_allowed',False):raise ValueError('this 64-world pilot entry is not enabled for formal production')
  if profile.get('infrastructure_trial') and profile.get('training_eligible') is not False:raise ValueError('infrastructure artifacts must be excluded from training')
  workspace=contained(profile['workspace_root'],base);workspace.mkdir(parents=True,exist_ok=True)
  lock=(workspace/'generation.lock').open('a')
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('another generation attempt owns this personal workspace; do not take its CPU/GPU')
  requests=Path(profile['request_paths'])
  if not requests.is_file():raise ValueError('train-only 64 pilot request list is missing: '+str(requests))
  if assignment['cpu_processes']<1:raise RuntimeError('no CPU capacity remains after colocated training reservations')
  code=Path(__file__).resolve().parents[2];generator=Path(profile['generator_code_root']).resolve();release=Path(profile['generator_release'])
  if not generator.is_relative_to(base) or not release.is_relative_to(base):raise ValueError('replaceable generator must be a deployed private release under this task root')
  source=json.loads(release.read_text())
  if profile.get('expected_generator_commit') and source['source_commit']!=profile['expected_generator_commit']:raise ValueError('generator commit does not match selected profile')
  if profile.get('expected_generator_base_commit') and source.get('generator_base_commit')!=profile['expected_generator_base_commit']:raise ValueError('v7 handoff commit does not match selected profile')
  os.environ.update(AVENGINE_GENERATOR_CODE_ROOT=str(generator),CCI_DATAGEN_BASE=str(base),CCI_REQUIRE_GPU_GUARD='1',GIT_CONFIG_GLOBAL=str(generator/'.git/CCI_SAFE_CONFIG'))
  gpu_policy=profile.get('gpu_boundary',{})
  os.environ.update(CCI_RENDER_CAP_GIB=str(gpu_policy.get('max_renderer_gib',12)),CCI_RENDER_TRAINING_RESERVE_GIB=str(gpu_policy.get('minimum_free_training_reserve_gib',4)))
  inherited=[x for x in os.environ.get('PYTHONPATH','').split(':') if x and x not in (str(code),str(code/'src'))]
  os.environ['PYTHONPATH']=':'.join([str(generator/'src'),str(generator),*inherited])
  run_root=contained(profile['generation_run_root'],workspace);run_root.mkdir(parents=True,exist_ok=True)
  manifest=run_root/'MANIFEST.json'
  if not manifest.exists():
   cmd=[sys.executable,'-B',str(code/'tools/cci_datagen'/('make_v7_infra.py' if profile.get('infrastructure_trial') else 'make_pilot.py')),'--requests',str(requests),'--recommended-houses',profile['recommended_houses'],'--code-root',str(code),'--generator-profile',str(profile_path),'--output',str(manifest),'--world-count','64','--scratch-gib-per-world',str(profile.get('scratch_gib_per_world',4))]
   if profile.get('pre_rendered_captures'):cmd+=['--pre-rendered-captures',profile['pre_rendered_captures']]
   owned_run(cmd,out/'make_manifest.log')
  du=out/'owner_usage_initial.txt'
  owned_run(['nice','-n','19','timeout','300','du','-sB1','/mnt/afs/xinyuanqian/jzy'],du)
  used=int(du.read_text().split()[0]);budget=min(pool_allowance(used,tree_bytes(workspace),**profile.get('owner_reserve_policy',{})),int(profile.get('workspace_operating_limit_bytes',10**30)));write_json(out/'storage_allowance.json',{'owner_used_bytes':used,'generation_pool_allowance_bytes':budget,'media_and_scratch_share_one_allowance':True,'owner_reserve_policy':profile.get('owner_reserve_policy',{}),'owner_reserve_total_bytes':sum(profile.get('owner_reserve_policy',{}).values()) if profile.get('owner_reserve_policy') else 124000000000,'limit_bytes':2_000_000_000_000})
  visible=observation['observer']['gpus']['visible'];lease_gpu=lease['gpu']
  if ',' in lease_gpu:raise ValueError('ordinary generation may use only its one leased GPU')
  gpu=next((x['index'] for x in visible if x['index']==lease_gpu or x['uuid'].startswith(lease_gpu)),None)
  cpu_only=bool(profile.get('pre_rendered_captures')) and gpu is None
  if gpu is None and not cpu_only:raise RuntimeError('no usable leased GPU and no matching pre-rendered captures; no EGL job started')
  if gpu:
   # Only a training house is used for the true EGL check.
   world=json.loads(Path(json.loads(requests.read_text())[0]).read_text());scene=world['house']['input']['visual_source'] if profile.get('infrastructure_trial') else world['runtime']['cci_preflight_scene']
   cmd=[sys.executable,'-B',str(code/'tools/cci_datagen/native_preflight.py'),'--output',str(out/'egl_preflight.json'),'--render','--scene',scene,'--gpu-device-id',gpu]
   owned_run(cmd,out/'egl_preflight.log',gpu=gpu)
  if profile.get('infrastructure_trial'):
   from trial_metrics import afs_io,training_snapshot
   write_json(out/'afs_io_64MiB.json',afs_io(out/'afs_io_64MiB.tmp'))
   write_json(out/'training_observation_before.json',training_snapshot(observation))
  os.environ.update(RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',LOCAL_WORLD_SIZE='1',GROUP_RANK='0',GROUP_WORLD_SIZE='1',DATAGEN_LAUNCH_ID='normal-'+uuid.uuid4().hex)
  mode='cpu' if cpu_only else ('mixed' if assignment['other_training_ranks'] else 'dedicated')
  cmd=[sys.executable,'-B',str(code/'tools/cci_datagen/generate.py'),'--manifest',str(manifest),'--run-root',str(run_root),'--resource-assignment',str(out/'resource_assignment.json'),'--mode',mode,'--render-gpus',gpu or '', '--training-ranks',str(assignment['other_training_ranks']),'--loader-workers',str(assignment['loader_workers_per_training_rank']),'--nice','19','--generation-space-budget-bytes',str(budget),'--storage-workspace',str(workspace),'--storage-owner-root','/mnt/afs/xinyuanqian/jzy','--prune-owned-intermediates']
  if profile.get('infrastructure_trial'):
   from trial_limits import slots
   active=min(profile['max_active_worlds'],slots(assignment['cpu_processes'],limit=budget))
   if active<1:raise RuntimeError('space budget cannot reserve 64 deliveries and one active world')
   reserve_file=out/'owner_reserve_policy.json';write_json(reserve_file,profile['owner_reserve_policy'])
   write_json(out/'trial_concurrency.json',{'cpu_upper':assignment['cpu_processes'],'active_worlds_max':active,'gpu_render_lanes':1,'maximum_simultaneous_CPU_audio_worlds':max(0,active-1),'workspace_budget_bytes':budget,'trial_total_stop_bytes':profile['trial_stop_limit_bytes'],'native_scratch_bound_measured':False})
   cmd+=['--storage-reserve-policy',str(reserve_file),'--max-active-worlds',str(active),'--storage-check-seconds','1','--media-budget-gib',str(64*profile['delivery_mib_per_world']/1024)]
   os.environ.update(CCI_INFRASTRUCTURE_TRIAL='1',CCI_TRIAL_PROFILE=str(profile_path))
   restart=owned_run(cmd,out/'generation.log',restart_root=run_root)
   if restart:
    os.environ['DATAGEN_LAUNCH_ID']='normal-resume-'+uuid.uuid4().hex
    owned_run(cmd,out/'generation_resume.log',restart_root=run_root)
  else:owned_run(cmd,out/'generation.log')
  if profile.get('infrastructure_trial'):
   from trial_metrics import summarize
   summarize(run_root)
   write_json(out/'training_observation_after.json',training_snapshot(observation))
  done=json.loads((run_root/'ALL_DONE.json').read_text())
  if done.get('worlds')!=64:raise RuntimeError('pilot did not deliver exactly 64 worlds')
  finish(out,a.mode,{'completed_worlds':64,'scope':'infrastructure_trial' if profile.get('infrastructure_trial') else 'research_pilot','training_eligible':False,'formal_dataset_count':0,'completion':str(run_root/'ALL_DONE.json')},sha(profile_path))
 except Exception as e:
  stop(signal.SIGTERM,None);write_json(out/'job_failure.json',{'status':'failed','error':str(e),'completed_worlds':'see durable per-world receipts; not claimed as complete'});raise
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Two ordered finite diagnostics on our one leased P19 rank; no dataset output."""
import argparse,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
from resources import write_json
from egl_diagnostic_entry import run_child
from private_nvidia_egl import live_version,prepare,prove_cuda_lease,device_files
from egl_device_diag import metadata
from trial_limits import identity,verified_stop
from colocated_resources import observe,capacity
from trial_metrics import training_snapshot
from pixel_compare import compare

def frame_command(code,profile,out,*,gpu=None,full=False,samples=1):
 cmd=[sys.executable,'-B',str(code/'tools/cci_datagen/render_backend_sample.py'),'--request',profile['request'],'--output',str(out),'--samples',str(samples)]
 if gpu is not None:cmd+=['--gpu',gpu]
 if full:cmd+=['--full','--capture-references']
 return cmd

def history(observation):
 record=training_snapshot(observation);record['logs']=[]
 # Only log names supplied by peer workers, never walk or change another run.
 for row in observation['rank_observations']:
  if row['pid']==observation['scheduler_lease']['parent_pid']:continue
  try:
   links=[];pids=[row['pid']]+[x['pid'] for x in row.get('loading_details',{}).get('python_processes',[])]
   for pid in pids:
    worker=Path('/proc')/str(pid)
    for n in ('1','2'):
     p=worker/'fd'/n
     if p.is_symlink():links.append(os.readlink(p))
   for name in sorted(set(links)):
    p=Path(name)
    if p.is_file() and str(p).startswith('/mnt/afs/xinyuanqian/jzy/') and p.suffix=='.log':
     with p.open('rb') as f:f.seek(max(0,p.stat().st_size-8192));tail=f.read().decode(errors='replace')
     record['logs'].append({'worker_rank':row['ranks'].get('RANK'),'path':name,'bytes':p.stat().st_size,'mtime':p.stat().st_mtime,'last_8192_bytes':tail})
  except (OSError,ValueError):continue
 return record

def main():
 p=argparse.ArgumentParser();p.add_argument('--generator-profile',required=True);p.add_argument('--output',required=True);p.add_argument('--route',choices=['nvidia','mesa'],required=True);p.add_argument('--devbox-cpu',action='store_true');a=p.parse_args()
 profile=json.loads(Path(a.generator_profile).read_text());base=Path(profile['base']).resolve();out=Path(a.output).resolve();code=Path(__file__).resolve().parents[2]
 if not out.is_relative_to(base) or profile.get('training_eligible') is not False:raise ValueError('private diagnostic output contract failed')
 out.mkdir(parents=True,exist_ok=False);os.nice(19);start=time.monotonic();observation=observe(out/'metadata_only',io=False);lease=observation['scheduler_lease'];write_json(out/'resource_observation.json',observation)
 if not lease['verified_parent_worker'] and not (a.devbox_cpu and a.route=='mesa'):raise ValueError('P19 own-worker lease required for hardware diagnostic')
 write_json(out/'coordinator.process.json',{'host':socket.gethostname(),**identity(os.getpid()),'route':a.route,'scope':'stop only this own diagnostic PID; never worker/torchrun/training'})
 policy={'loader_workers_per_training_rank':4,'cpu_reserve_fraction':.1,'memory_reserve_fraction':.1,'training_gib_per_rank':33,'worker_gib':3};allocation=capacity(observation,policy);write_json(out/'capacity_policy.json',allocation)
 # Finite diagnostic uses one CPU, never spends the full production allowance.
 if not a.devbox_cpu and allocation['cpu_processes']<1:raise RuntimeError('no conservative CPU allocation remains after training loaders and reserve')
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,[allowed[-1]]);write_json(out/'metadata.json',metadata());write_json(out/'training_before.json',history(observation));runs=[];result={'scope':'EGL_paths_diagnostics_only','route':a.route,'training_eligible':False,'formal_dataset_count':0,'hostname':socket.gethostname(),'lease':lease,'CPU_threads':1,'CPU_nice':19,'peer_training_processes_changed':0,'cases':runs}
 try:
  if a.route=='nvidia':
   gpu=lease['gpu']
   if not str(gpu).isdigit() or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:raise ValueError('exact single physical worker GPU lease required')
   driver=live_version(gpu,out);write_json(out/'LIVE_DRIVER.json',driver);before=device_files(gpu);write_json(out/'device_files_before.json',before);runtime=prepare(base,driver,out);mapping=prove_cuda_lease(driver);write_json(out/'CUDA_LEASE_PROOF.json',mapping)
   env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'LD_PRELOAD':'','LIBGL_ALWAYS_SOFTWARE':'0','__EGL_VENDOR_LIBRARY_FILENAMES':str(runtime/'10_nvidia_private.json'),'LD_LIBRARY_PATH':str(runtime/'lib')+':'+profile['base_native_library_path'],'CCI_INFRA_HABITAT_DEVICE_ID':str(mapping['cuda_ordinal']),'CCI_INFRA_HABITAT_PHYSICAL_UUID':mapping['physical_uuid'],'CCI_RENDER_CAP_GIB':'2','CCI_RENDER_TRAINING_RESERVE_GIB':'4','CCI_RENDER_CPU_COUNT':'1','MESA_SHADER_CACHE_DISABLE':'true','CUDA_CACHE_PATH':str(base/'tmp/cuda_cache'),'__GL_SHADER_DISK_CACHE_PATH':str(base/'tmp/gl_shader_cache')}
   for k in ('MESA_LOADER_DRIVER_OVERRIDE','LIBGL_DRIVERS_PATH','EGL_PLATFORM'):env.pop(k,None)
   # NVIDIA EGL device enumeration can map all handles; initialize only UUID-proved CUDA ordinal.
   for name,platform in [('nvidia_private',None),('nvidia_device','device')]:
    variant={**env}
    if platform:variant['EGL_PLATFORM']=platform
    enum=[sys.executable,'-B',str(code/'tools/cci_datagen/egl_device_diag.py'),'--gpu',str(mapping['cuda_ordinal']),'--library',profile['egl_library'],'--initialize-leased-device','--expected-uuid',mapping['physical_uuid']]
    runs.append({'name':name+'_devices',**run_child(enum,variant,out/(name+'_devices.log'),20,gpu=gpu)})
    try:query=json.loads((out/(name+'_devices.log')).read_text().splitlines()[-1])
    except (ValueError,IndexError,OSError):query={}
    if query.get('status')!='metadata_complete' or not query.get('device_display_initialized'):
     runs.append({'name':name+'_64x64','status':'not_run','reason':'EGL device platform did not initialize the UUID-proved CUDA lease; no default-device Habitat fallback attempted'});continue
    tiny=[sys.executable,'-B',str(code/'tools/cci_datagen/native_preflight.py'),'--output',str(out/(name+'_habitat.json')),'--render','--scene',profile['scene'],'--gpu-device-id',str(mapping['cuda_ordinal'])]
    case={'name':name+'_64x64',**run_child(tiny,variant,out/(name+'_habitat.log'),45,gpu=gpu)};runs.append(case)
    if case['status']=='passed':
     full={'name':name+'_30s_720p',**run_child(frame_command(code,profile,out/'hardware_30s',gpu=gpu,full=True),variant,out/'hardware_30s.log',profile['hardware_timeout_seconds'],gpu=gpu)};runs.append(full)
     if full['status']=='passed':
      result['render_result']=json.loads((out/'hardware_30s/RESULT.json').read_text());write_json(base/'evidence/HARDWARE_REFERENCE.json',{'directory':str(out/'hardware_30s'),'P19_job':lease,'reference_type':'P19 hardware same request, not 48g hardware','source_commit':profile['source_commit']})
      ref=Path(profile['reference_48_cpu'])
      if all((ref/f'frame_{i:04d}.npz').exists() for i in (0,149,300,449)):result['48g_comparison']=compare(out/'hardware_30s',ref,out/'P19_GPU_VS_48G_CPU_PIXEL_COMPARISON.json','P19 NVIDIA hardware vs same-scene 48g Habitat llvmpipe; 48g hardware unavailable, not claimed')
     break
   after=device_files(gpu);write_json(out/'device_files_after.json',after)
   if after!=before:raise RuntimeError('device metadata changed; no further graphics attempts')
  else:
   software=Path(profile['mesa_root']);selector=Path(profile['cpu_selector']);env={**os.environ,'CUDA_VISIBLE_DEVICES':'','LIBGL_ALWAYS_SOFTWARE':'1','MESA_LOADER_DRIVER_OVERRIDE':'llvmpipe','EGL_PLATFORM':'surfaceless','__EGL_VENDOR_LIBRARY_FILENAMES':profile['mesa_vendor_json'],'LD_PRELOAD':str(selector),'LIBGL_DRIVERS_PATH':str(software/'lib/dri'),'LD_LIBRARY_PATH':str(software/'lib')+':'+profile['base_native_library_path'],'LP_NUM_THREADS':'1','CCI_RENDER_CPU_COUNT':'1','MESA_SHADER_CACHE_DISABLE':'true'}
   for k in ('CCI_INFRA_RENDER_GPU','CCI_INFRA_HABITAT_DEVICE_ID','CCI_INFRA_HABITAT_PHYSICAL_UUID'):env.pop(k,None)
   full={'name':'condaforge_llvmpipe_30s_720p_1CPU',**run_child(frame_command(code,profile,out/'software_30s',full=True),env,out/'software_30s.log',profile['software_timeout_seconds'])};runs.append(full)
   if full['status']=='passed':
    result['render_result']=json.loads((out/'software_30s/RESULT.json').read_text());marker=base/'evidence/HARDWARE_REFERENCE.json'
    if marker.exists():
     ref=json.loads(marker.read_text())
     if ref['source_commit']!=profile['source_commit']:raise ValueError('hardware reference is a different generator version')
     result['hardware_pixel_comparison']=compare(ref['directory'],out/'software_30s',out/'P19_CPU_GPU_PIXEL_COMPARISON.json','same-scene P19 NVIDIA EGL vs conda-forge Mesa llvmpipe')
    else:write_json(out/'GPU_PIXEL_COMPARISON_NOT_RUN.json',{'reason':'route1 has no successful full-world hardware reference; no fabricated GPU pixel difference'})
    ref=Path(profile['reference_48_cpu'])
    if all((ref/f'frame_{i:04d}.npz').exists() for i in (0,149,300,449)):result['48g_software_comparison']=compare(out/'software_30s',ref,out/'CCI_CF_MESA_VS_48G_MESA_PIXEL_COMPARISON.json','CCI conda-forge Mesa vs previous 48g Habitat llvmpipe; differing Mesa versions recorded')
  result['render_passed']=any(r.get('status')=='passed' and ('30s' in r['name']) for r in runs)
 except Exception as e:
  result.update(render_passed=False,exception=str(e),traceback=traceback.format_exc());(out/'exception.log').write_text(result['traceback'])
 finally:
  write_json(out/'training_after.json',history(observation))
 result.update(wall_seconds=time.monotonic()-start,observed_CPU_affinity=list(os.sched_getaffinity(0)),training_speed_impact='read-only before/during/after logs and NVML, observational only; no controlled causal attribution')
 write_json(out/'DIAGNOSTIC.json',result);name='cci_egl_paths_r6_'+a.route;(out/'input_manifest.jsonl').write_text(json.dumps({'id':name,'task_kind':'EGL_paths_diagnostics_only'})+'\n');(out/'predictions.jsonl').write_text(json.dumps({'id':name,'status':'complete','result':result})+'\n');write_json(out/'run.json',{'status':'complete','count':1,'task_kind':'EGL_paths_diagnostics_only','is_model_evaluation':False,'completed_worlds':0,'rendering_success':result['render_passed'],'note':'diagnostic stage complete, failure recorded honestly; later Mesa stage may proceed'});print(json.dumps(result))
if __name__=='__main__':main()

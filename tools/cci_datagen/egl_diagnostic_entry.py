#!/usr/bin/env python3
"""Ordinary one-rank EGL diagnostics; no RLR, training, dataset generation or retry."""
import argparse,json,os,sys,time,subprocess,signal,socket
from pathlib import Path
from resources import write_json
from generate import sha
from egl_device_diag import metadata
OWN_CHILD=None
OWN_IDENTITY=None
def stop_own(sig,frame):
 if OWN_CHILD is not None and OWN_CHILD.poll() is None:
  from trial_limits import verified_stop
  verified_stop(OWN_CHILD,OWN_IDENTITY)
 raise SystemExit(128+sig)
def run_child(argv,env,log,seconds,gpu=None):
 from gpu_guard import GPUWatch,require_headroom
 global OWN_CHILD,OWN_IDENTITY
 result={'argv':argv,'timeout_seconds':seconds}
 if gpu:
  try:result['initial_GPU']=require_headroom(gpu,cap_gib=2,reserve_gib=4)
  except Exception as e:return {**result,'status':'skipped_GPU_boundary','error':str(e)}
 start=time.monotonic()
 with log.open('x') as out:
  child=subprocess.Popen(argv,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
  from trial_limits import identity,verified_stop
  expected=identity(child.pid);OWN_CHILD=child;OWN_IDENTITY=expected;write_json(log.with_suffix('.process.json'),{'host':socket.gethostname(),**expected})
  def forward_own(signum,frame):
   if child.poll() is None:verified_stop(child,expected)
   raise SystemExit(128+signum)
  handlers={sig:signal.signal(sig,forward_own) for sig in (signal.SIGTERM,signal.SIGINT)}
  watch=GPUWatch(gpu,child,log.with_suffix('.gpu.json'),cap_gib=2,reserve_gib=4,strict_ownership=any(str(arg).endswith('/render_backend_sample.py') for arg in argv)).start() if gpu else None
  try:rc=child.wait(timeout=seconds)
  except subprocess.TimeoutExpired:
   verified_stop(child,expected)
   try:rc=child.wait(timeout=15)
   except subprocess.TimeoutExpired:
    if identity(child.pid)!=expected:raise RuntimeError('own process changed; refuse forced signal')
    os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
   result['timed_out']=True
  for sig,handler in handlers.items():signal.signal(sig,handler)
  if watch:
   try:watch.close()
   except Exception as e:result['GPU_guard_error']=str(e)
 OWN_CHILD=None;OWN_IDENTITY=None
 result.update(returncode=rc,wall_seconds=time.monotonic()-start,log=str(log),status='passed' if rc==0 and not result.get('GPU_guard_error') else 'blocked')
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--generator-profile',required=True);p.add_argument('--output',required=True);p.add_argument('--devbox-mig',action='store_true');a=p.parse_args()
 profile=json.loads(Path(a.generator_profile).read_text());base=Path(profile['base']).resolve();out=Path(a.output).resolve()
 if not out.is_relative_to(base):raise ValueError('diagnostic output outside owned personal root')
 out.mkdir(parents=True,exist_ok=False);signal.signal(signal.SIGTERM,stop_own);signal.signal(signal.SIGINT,stop_own);os.nice(19);os.sched_setaffinity(0,[max(os.sched_getaffinity(0))]);started=time.monotonic();code=Path(__file__).resolve().parents[2]
 from trial_limits import identity
 write_json(out/'coordinator.process.json',{'host':socket.gethostname(),**identity(os.getpid()),'scope':'diagnostic only; stop this PID, never shared worker/torchrun'})
 from colocated_resources import observe
 observation=observe(out/'metadata_only',io=False);lease=observation['scheduler_lease']
 if not lease['verified_parent_worker'] and not a.devbox_mig:raise ValueError('P19 worker lease not verified')
 raw=metadata();write_json(out/'metadata.json',raw);write_json(out/'scheduler_lease.json',lease)
 mig='MIG ' in raw['nvidia_list'].get('stdout','')
 if a.devbox_mig and not mig:raise ValueError('standalone GPU diagnostic is authorized only for the observed MIG devbox')
 gpu=lease.get('gpu') if not a.devbox_mig else 'MIG-devbox-no-physical-graphics-lease'
 if not a.devbox_mig and (not gpu or not gpu.isdigit()):raise ValueError('missing exact physical GPU lease')
 vendor=[r['path'] for r in raw['vendors'] if 'nvidia' in (r['path']+' '+r['text']).lower()]
 variants=[('inherited',{}),('EGL_PLATFORM_device',{'EGL_PLATFORM':'device'})]
 if vendor:variants.append(('device_actual_NVIDIA_vendor',{'EGL_PLATFORM':'device','__EGL_VENDOR_LIBRARY_FILENAMES':':'.join(vendor)}))
 else:write_json(out/'NVIDIA_VENDOR_OVERRIDE_NOT_RUN.json',{'reason':'no actual NVIDIA vendor JSON exists; no fabricated JSON or system library change'})
 runs=[];hardware_sample_variant=None
 for name,changes in variants:
  env={**os.environ,**changes};cmd=[sys.executable,'-B',str(code/'tools/cci_datagen/egl_device_diag.py'),'--gpu',gpu,'--library',profile['egl_library']]
  cmd+=['--mig'] if mig else ['--initialize-leased-device']
  runs.append({'variant':name,'kind':'EGL_device_metadata',**run_child(cmd,env,out/(name+'_devices.log'),15)})
  if not mig:
   native=[sys.executable,'-B',str(code/'tools/cci_datagen/native_preflight.py'),'--output',str(out/(name+'_habitat.json')),'--render','--scene',profile['scene'],'--gpu-device-id',gpu]
   hardware_result=run_child(native,env,out/(name+'_habitat.log'),35,gpu=gpu)
   runs.append({'variant':name,'kind':'native_64x64_frame',**hardware_result})
   if hardware_result['status']=='passed' and hardware_sample_variant is None:
    gpu_env={**env,'CUDA_VISIBLE_DEVICES':gpu,'CCI_INFRA_RENDER_GPU':gpu,'CCI_RENDER_CPU_COUNT':'1'}
    gpu_env.pop('LD_PRELOAD',None);gpu_env['LIBGL_ALWAYS_SOFTWARE']='0'
    cmd=[sys.executable,'-B',str(code/'tools/cci_datagen/render_backend_sample.py'),'--request',profile['request'],'--output',str(out/'hardware_sample'),'--samples','1','--gpu',gpu]
    sample=run_child(cmd,gpu_env,out/'hardware_sample.log',45,gpu=gpu)
    runs.append({'variant':name,'kind':'native_four_720p_reference_frames',**sample})
    if sample['status']=='passed':hardware_sample_variant=name
  else:write_json(out/(name+'_HABITAT_NOT_RUN.json'),{'reason':'H100 MIG has no graphics lease; physical parent GPU renderer not started'})
 software=Path(profile['software_root']);env={**os.environ,'CUDA_VISIBLE_DEVICES':'','LIBGL_ALWAYS_SOFTWARE':'1','MESA_LOADER_DRIVER_OVERRIDE':'llvmpipe','EGL_PLATFORM':'surfaceless','__EGL_VENDOR_LIBRARY_FILENAMES':str(software/'egl_vendor.d/50_mesa.json'),'LD_PRELOAD':str(software/'libegl_cpu_only.so'),'LIBGL_DRIVERS_PATH':str(software/'lib/dri'),'LD_LIBRARY_PATH':str(software/'lib')+':'+profile['base_native_library_path'],'LP_NUM_THREADS':'1','CCI_RENDER_CPU_COUNT':'1','MESA_SHADER_CACHE_DISABLE':'true'}
 env.pop('CCI_INFRA_RENDER_GPU',None)
 cpu=[sys.executable,'-B',str(code/'tools/cci_datagen/render_backend_sample.py'),'--request',profile['request'],'--output',str(out/'software_sample'),'--samples','24']
 if (software/'egl_vendor.d/50_mesa.json').is_file():runs.append({'variant':'private_CPU_llvmpipe','kind':'native_sparse_720p_frames',**run_child(cpu,env,out/'software_sample.log',120)})
 else:runs.append({'variant':'private_CPU_llvmpipe','status':'not_run','reason':'private software dependencies not deployed'})
 comparison={'status':'not_run','reason':'matching hardware sample unavailable; no pixel equivalence claim'}
 if hardware_sample_variant and (out/'software_sample/RESULT.json').is_file():
  import numpy as np
  pairs=[]
  for index in (0,149,300,449):
   with np.load(out/'hardware_sample'/f'frame_{index:04d}.npz') as g,np.load(out/'software_sample'/f'frame_{index:04d}.npz') as c:
    rgbdiff=np.abs(g['rgb'].astype(np.int16)-c['rgb'].astype(np.int16))
    depthdiff=np.abs(g['depth'].astype(np.float64)-c['depth'].astype(np.float64))
    pairs.append({'frame_index':index,'rgb_pixels_different_fraction':float(np.any(rgbdiff!=0,axis=2).mean()),'rgb_channel_MAE_0_255':float(rgbdiff.mean()),'rgb_channel_max_0_255':int(rgbdiff.max()),'depth_MAE_m':float(depthdiff.mean()),'depth_max_m':float(depthdiff.max()),'semantic_pixels_different_fraction':float((g['semantic']!=c['semantic']).mean())})
  comparison={'status':'measured','GPU_variant':hardware_sample_variant,'same_request':profile['request'],'pairs':pairs,'no_assumption_of_bitwise_backend_equivalence':True}
 write_json(out/'CPU_GPU_PIXEL_COMPARISON.json',comparison)
 result={'scope':'EGL_diagnostics_only','training_eligible':False,'formal_dataset_count':0,'completed_worlds':0,'MIG_devbox':mig,'runs':runs,'wall_seconds':time.monotonic()-started,'P19_same_checks_completed':not a.devbox_mig,'peer_training_processes_changed':0}
 write_json(out/'DIAGNOSTIC.json',result)
 (out/'input_manifest.jsonl').write_text(json.dumps({'id':'cci_egl_diagnostic_r5','task_kind':'EGL_diagnostics_only'})+'\n')
 (out/'predictions.jsonl').write_text(json.dumps({'id':'cci_egl_diagnostic_r5','status':'complete','result':result})+'\n')
 write_json(out/'run.json',{'status':'complete','count':1,'task_kind':'EGL_diagnostics_only','is_model_evaluation':False,'completed_worlds':0,'note':'diagnosis finished; blocked rendering is recorded per-case, never marked successful rendering'})
 print(json.dumps(result))
if __name__=='__main__':main()

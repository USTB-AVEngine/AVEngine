"""Readonly timing/IO evidence; scientific and training eligibility stay false."""
import hashlib,json,os,statistics,time
from pathlib import Path
from resources import write_json

def afs_io(path,size=64*1024**2):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);chunk=hashlib.sha256(b'cci-infrastructure-io').digest()*(1024**2//32)
 start=time.monotonic();h=hashlib.sha256()
 with p.open('xb') as f:
  for _ in range(size//len(chunk)):f.write(chunk);h.update(chunk)
  f.flush();os.fsync(f.fileno())
 write_s=time.monotonic()-start;start=time.monotonic();actual=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):actual.update(b)
 read_s=time.monotonic()-start
 if actual.digest()!=h.digest():raise RuntimeError('AFS IO hash readback differs')
 result={'bytes':size,'sha256':h.hexdigest(),'write_fsync_seconds':write_s,'warm_read_hash_seconds':read_s,'write_MB_s':size/write_s/1e6,'warm_read_hash_MB_s':size/read_s/1e6,'cache_not_flushed':True}
 p.unlink();return result

def training_snapshot(observation):
 from colocated_resources import peer_loading_details
 return {'unix_time':time.time(),'ranks':[{'rank':r['ranks'].get('RANK'),'worker_pid':r['pid'],'loading':peer_loading_details(r['pid'])} for r in observation['rank_observations'] if r['pid']!=observation['scheduler_lease']['parent_pid']],
         'interpretation':'configured prefetch and observed processes/threads; not core utilization or a causal training-speed measurement'}

def summarize(root):
 root=Path(root);done=json.loads((root/'ALL_DONE.json').read_text());phases={};gpu=[];errors=[];windows=0
 for f in (root/'receipts').glob('*/*.json'):
  r=json.loads(f.read_text());phases.setdefault(r['phase'],[]).append(r['wall_seconds'])
  if r.get('gpu_memory_summary'):gpu.append(r['gpu_memory_summary'])
 for f in (root/'worlds').glob('*/ARTIFACTS.json'):
  r=json.loads(f.read_text());windows+=r['native_media_windows'];errors+=r['question_errors']
 result={'scope':'infrastructure_trial','training_eligible':False,'formal_dataset_count':0,'media_worlds':done['worlds'],'media_windows':windows,'elapsed_with_own_restart_seconds':done['elapsed_with_restart_seconds'],'media_worlds_per_hour':done['worlds']*3600/done['elapsed_with_restart_seconds'],
 'stage_wall_seconds':{k:{'count':len(v),'sum':sum(v),'mean':statistics.mean(v),'min':min(v),'max':max(v)} for k,v in phases.items()},
 'renderer_peak_sampled_bytes':max((g['own_renderer_peak_bytes'] for g in gpu if g['own_renderer_peak_bytes'] is not None),default=None),
 'GPU_free_min_bytes':min((g['minimum_free_bytes'] for g in gpu if g['minimum_free_bytes'] is not None),default=None),'native_question_failure_windows':len(errors),'question_errors':errors,'scientific_admission':False,'real_host_4h_restart':'unverified unless a separate cross-host attempt is recorded','training_speed_impact':'not a controlled causal benchmark'}
 write_json(root/'BENCHMARK.json',result);return result

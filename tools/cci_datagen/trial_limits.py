"""Trial-wide space monitoring and one verified own-process restart drill."""
import json,os,signal,time
from pathlib import Path
from generate import sha,tree_bytes,files_ok
from resources import write_json

def slots(cpu_processes,world_count=64,*,limit=140_000_000_000,scratch=8*1024**3,delivery=512*1024**2,overhead=2_000_000_000):
 if min(cpu_processes,world_count,limit,scratch,delivery)<0 or scratch==0:raise ValueError('invalid trial budget')
 return max(0,min(cpu_processes,world_count,(limit-world_count*delivery-overhead)//scratch))

def identity(pid):
 p=Path('/proc')/str(pid);raw=(p/'stat').read_text();tail=raw[raw.rfind(')')+2:].split()
 return {'pid':pid,'start_ticks':int(tail[19]),'ppid':int(tail[1]),'pgid':int(tail[2]),'argv':[(x.decode(errors='replace')) for x in (p/'cmdline').read_bytes().split(b'\0') if x]}

def verified_stop(process,expected):
 if process.poll() is not None:return False
 actual=identity(process.pid)
 if actual!=expected or actual['ppid']!=os.getpid() or actual['pgid']!=process.pid:raise RuntimeError('own subprocess identity changed; refuse signal')
 os.killpg(process.pid,signal.SIGTERM);return True

def completed(root):
 result={}
 for p in sorted((root/'worlds').glob('*/COMPLETE.json')):
  d=json.loads(p.read_text())
  if not files_ok(d.get('files',[]),hashes=True):raise RuntimeError('trial delivery hash readback failed: '+str(p))
  result[str(p)]=sha(p)
 return result

def monitor_generation(process,root,log,argv):
 profile=json.loads(Path(os.environ['CCI_TRIAL_PROFILE']).read_text());base=Path(profile['base']);restart_file=root/'OWN_RESTART_DRILL.json'
 expected=None
 if process.poll() is None:
  try:expected=identity(process.pid)
  except OSError:
   if process.poll() is None:raise
  if expected and expected['argv']!=argv:raise RuntimeError('generation child argv does not match planned command')
 prior=json.loads(restart_file.read_text()) if restart_file.exists() else None
 if prior:
  for p,digest in prior['complete_receipt_sha256'].items():
   if sha(p)!=digest:raise RuntimeError('completed world changed across restart')
 start=time.monotonic();last=0;peak=0;planned=None
 while process.poll() is None:
  now=time.monotonic()
  if now-last>=1:
   used=tree_bytes(base);peak=max(peak,used);last=now
   if used>profile['trial_stop_limit_bytes']:
    verified_stop(process,expected)
    write_json(Path(log).with_suffix('.trial_space_failure.json'),{'trial_bytes':used,'limit_bytes':profile['trial_stop_limit_bytes'],'stopped_own_process':expected})
    raise RuntimeError('trial space boundary reached; only own generator stopped')
   write_json(Path(log).with_suffix('.trial_space.json'),{'trial_actual_bytes':used,'trial_peak_sampled_bytes':peak,'limit_bytes':profile['trial_stop_limit_bytes'],'operating_workspace_limit_bytes':profile['workspace_operating_limit_bytes'],'sampling_seconds':1,'hardware_quota':False})
  if not prior and profile.get('restart_drill_after_completed_worlds',0)>0:
   paths=list((root/'worlds').glob('*/COMPLETE.json'))
   if len(paths)>=profile['restart_drill_after_completed_worlds']:
    hashes=completed(root)
    if verified_stop(process,expected):
     planned={'type':'verified_own_SIGTERM_then_resume','process_identity':expected,'complete_receipt_sha256':hashes,'elapsed_seconds':time.monotonic()-start,'real_host_4h_restart':'unverified'}
     write_json(restart_file,planned);break
  time.sleep(.25)
 if prior and process.poll()==0:
  after=completed(root)
  for p,digest in prior['complete_receipt_sha256'].items():
   if after.get(p)!=digest:raise RuntimeError('completed world hash changed after resume')
  write_json(root/'OWN_RESTART_READBACK.json',{'status':'pass','prior_worlds':len(prior['complete_receipt_sha256']),'completed_worlds':len(after),'real_host_4h_restart':'unverified'})
 return planned

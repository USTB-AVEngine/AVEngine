"""A shared AFS allowance for retained deliveries and active world reservations."""
import os,subprocess,time
from pathlib import Path
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
from resources import write_json

def pool_allowance(owner_used_bytes,workspace_used_bytes,*,owner_limit=2_000_000_000_000,checkpoint_reserve=64_000_000_000,panel_reserve=32_000_000_000,validation_reserve=16_000_000_000,deployment_reserve=12_000_000_000):
 return max(0,int(owner_limit-owner_used_bytes+workspace_used_bytes-checkpoint_reserve-panel_reserve-validation_reserve-deployment_reserve))

class AFSSpace:
 def __init__(self,workspace,run_root,budget_bytes,worlds):
  self.workspace=Path(workspace).resolve();self.root=Path(run_root).resolve();self.limit=int(budget_bytes);self.worlds={w['world_id']:w for w in worlds}
  if not self.root.is_relative_to(self.workspace):raise ValueError('AFS run root escapes generation workspace')
  if self.limit<=0:raise ValueError('no generation storage allowance')
 def snapshot(self,active):
  total=tree_bytes(self.workspace);exclude=0;reservation=0
  for wid in set(active):
   if wid not in self.worlds:raise ValueError('unknown active world')
   exclude+=tree_bytes(self.root/'scratch'/wid)+tree_bytes(self.root/'worlds'/wid)
   w=self.worlds[wid];reservation+=int(w['scratch_bytes_peak'])+int(w['delivery_bytes_upper'])
  committed=max(0,total-exclude);projected=committed+reservation
  return {'workspace':str(self.workspace),'allowance_bytes':self.limit,'actual_bytes':total,'committed_nonactive_bytes':committed,'reserved_active_world_bytes':reservation,'projected_bytes':projected,'headroom_bytes':self.limit-projected,'active_worlds':len(set(active)),'temporary_location':'AFS personal generation workspace; no 32 GiB local scratch cap'}
 def can_admit(self,wid,active):
  s=self.snapshot(active);w=self.worlds[wid]
  return s['headroom_bytes']>=int(w['scratch_bytes_peak'])+int(w['delivery_bytes_upper'])+min(512*1024**2,max(0,self.limit//1000))
 def check(self,active):
  s=self.snapshot(active)
  if s['actual_bytes']>self.limit or s['projected_bytes']>self.limit:raise RuntimeError('generation AFS allowance exceeded: '+str(s))
  for wid in set(active):
   if tree_bytes(self.root/'scratch'/wid)>self.worlds[wid]['scratch_bytes_peak']:raise RuntimeError('world scratch peak declaration exceeded: '+wid)
  return s

class OwnerUsageWatch:
 """Periodically remeasure shared owner usage; only its own du child is stopped."""
 def __init__(self,owner_root,workspace,log_root,interval=600):
  self.owner=Path(owner_root);self.workspace=Path(workspace);self.log=Path(log_root);self.interval=interval;self.next_check=0;self.child=None;self.handle=None;self.latest=None
 def poll(self):
  if self.child is not None and self.child.poll() is not None:
   rc=self.child.returncode;self.handle.close();self.handle=None;self.child=None
   if rc:raise RuntimeError('owner du failed; storage boundary cannot be refreshed')
   used=int((self.log/'owner_usage_latest.txt').read_text().split()[0]);self.latest=pool_allowance(used,tree_bytes(self.workspace));write_json(self.log/'owner_usage_latest.json',{'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'owner_used_bytes':used,'generation_pool_allowance_bytes':self.latest,'command':['nice','-n','19','timeout','300','du','-sB1',str(self.owner)],'checkpoint_growth_reserve_bytes':64_000_000_000})
  if self.child is None and time.monotonic()>=self.next_check:
   self.log.mkdir(parents=True,exist_ok=True);self.handle=(self.log/'owner_usage_latest.txt').open('w');self.child=subprocess.Popen(['nice','-n','19','timeout','300','du','-sB1',str(self.owner)],stdout=self.handle,stderr=subprocess.DEVNULL,start_new_session=True);self.next_check=time.monotonic()+self.interval
  return self.latest
 def close(self):
  if self.child is not None and self.child.poll() is None:
   try:os.killpg(self.child.pid,15)
   except ProcessLookupError:pass
   self.child.wait()
  if self.handle:self.handle.close()

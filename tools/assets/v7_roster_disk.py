"""Enforce a 50 GiB disk reserve and at most 15 GiB net growth after cleanup."""
from pathlib import Path
import json,os,subprocess,time
MIN_FREE=50*1024**3
MAX_GROWTH=15*1024**3
def task_bytes(T):
 # A concurrent UE cache initialization made coreutils du abort with SIGABRT.
 # Walk only this task tree, without following symlinks, and count hardlinks once.
 pending=[Path(T)];seen=set();total=0
 while pending:
  p=pending.pop()
  try:st=p.lstat()
  except FileNotFoundError:continue
  key=(st.st_dev,st.st_ino)
  if key in seen:continue
  seen.add(key);total+=st.st_size
  if p.is_dir() and not p.is_symlink():
   try:
    with os.scandir(p)as entries:pending.extend(Path(e.path)for e in entries)
   except FileNotFoundError:continue
 return total
def initialize(task_root):
 T=Path(task_root);p=T/'round3/evidence/disk_budget_base.json'
 if p.exists():return json.loads(p.read_text())
 st=os.statvfs('/data');base={'time':time.time(),'task_bytes_after_cleanup':task_bytes(T),'free_bytes':st.f_bavail*st.f_frsize,'minimum_free_bytes':MIN_FREE,'maximum_net_growth_bytes':MAX_GROWTH}
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(base,indent=2)+'\n');return base
def check(task_root,planned_bytes=0):
 T=Path(task_root);st=os.statvfs('/data');free=st.f_bavail*st.f_frsize;p=T/'round3/evidence/disk_budget_base.json'
 if not p.exists():raise RuntimeError('Disk budget baseline missing; initialize explicitly before builds')
 base=json.loads(p.read_text());used=task_bytes(T);growth=used-base['task_bytes_after_cleanup']
 if free-planned_bytes<MIN_FREE:raise RuntimeError(f'Disk reserve: free={free}, planned={planned_bytes}, minimum={MIN_FREE}; no asset write')
 if growth+planned_bytes>MAX_GROWTH:raise RuntimeError(f'Task growth: current={growth}, planned={planned_bytes}, limit={MAX_GROWTH}; no asset write')
 return {'free_bytes':free,'planned_bytes':planned_bytes,'minimum_free_bytes':MIN_FREE,'task_bytes':used,'net_growth_bytes':growth,'maximum_new_growth_bytes':MAX_GROWTH}

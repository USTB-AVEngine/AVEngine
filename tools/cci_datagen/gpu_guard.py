"""Read-only NVML checks and bounded monitoring of the renderer's own process tree."""
import math,os,re,signal,subprocess,threading,time,xml.etree.ElementTree as ET
from pathlib import Path
from resources import GIB,write_json

def mib(s):
 m=re.fullmatch(r'([0-9.]+) MiB',s or '')
 if not m:raise RuntimeError('GPU memory query unavailable: '+str(s))
 return int(float(m.group(1))*1024**2)

def snapshot(gpu):
 if not gpu or ',' in gpu:raise ValueError('exactly the assigned GPU is required')
 p=subprocess.run(['nvidia-smi','--id='+gpu,'-q','-x'],capture_output=True,text=True,timeout=5)
 if p.returncode:raise RuntimeError('GPU metadata query failed: '+p.stderr[-1000:])
 root=ET.fromstring(p.stdout);rows=root.findall('gpu')
 if len(rows)!=1:raise RuntimeError('assigned GPU selector did not resolve exactly one GPU')
 row=rows[0];memory=row.find('fb_memory_usage')
 result={'gpu':gpu,'uuid':row.findtext('uuid'),'name':row.findtext('product_name'),'total_bytes':mib(memory.findtext('total')),'used_bytes':mib(memory.findtext('used')),'free_bytes':mib(memory.findtext('free')),'processes':[],'unix_time':time.time()}
 for proc in row.findall('./processes/process_info'):
  used=proc.findtext('used_memory')
  result['processes'].append({'pid':int(proc.findtext('pid')),'type':proc.findtext('type'),'used_bytes':mib(used) if used not in (None,'N/A') else None})
 return result

def owned_descendant(pid,root_pid):
 seen=set()
 while pid>1 and pid not in seen:
  if pid==root_pid:return True
  seen.add(pid)
  try:
   text=Path(f'/proc/{pid}/status').read_text();pid=int(next(l.split()[1] for l in text.splitlines() if l.startswith('PPid:')))
  except (OSError,StopIteration,ValueError):return False
 return False

def boundaries(cap_gib=None,reserve_gib=None):
 cap=float(os.environ.get('CCI_RENDER_CAP_GIB','12')) if cap_gib is None else float(cap_gib)
 reserve=float(os.environ.get('CCI_RENDER_TRAINING_RESERVE_GIB','4')) if reserve_gib is None else float(reserve_gib)
 if not math.isfinite(cap) or not math.isfinite(reserve) or cap<=0 or reserve<0:raise ValueError('invalid renderer GPU budget')
 return cap,reserve

def require_headroom(gpu,cap_gib=None,reserve_gib=None):
 cap_gib,reserve_gib=boundaries(cap_gib,reserve_gib)
 row=snapshot(gpu)
 if row['free_bytes']<(cap_gib+reserve_gib)*GIB:raise RuntimeError('leased GPU has insufficient renderer headroom: '+str(row))
 return row

class GPUWatch:
 def __init__(self,gpu,process,path,cap_gib=None,reserve_gib=None,strict_ownership=True):
  cap_gib,reserve_gib=boundaries(cap_gib,reserve_gib)
  self.strict_ownership=strict_ownership
  self.gpu=gpu;self.process=process;self.path=path;self.cap=cap_gib*GIB;self.reserve=reserve_gib*GIB;self.done=threading.Event();self.rows=[];self.error=None;self.thread=threading.Thread(target=self.run,daemon=True)
 def run(self):
  while not self.done.is_set() and self.process.poll() is None:
   try:
    row=snapshot(self.gpu);owned=[p for p in row['processes'] if owned_descendant(p['pid'],self.process.pid)]
    if any(p['used_bytes'] is None for p in owned):raise RuntimeError('own renderer memory is not observable in NVML')
    row['own_renderer_bytes']=sum(p['used_bytes'] for p in owned) if owned else None;self.rows.append(row)
    if (row['own_renderer_bytes'] is not None and row['own_renderer_bytes']>self.cap) or row['free_bytes']<self.reserve:raise RuntimeError('renderer memory boundary crossed')
   except Exception as e:
    self.error=str(e)
    if self.process.poll() is None:
     try:os.killpg(self.process.pid,signal.SIGTERM)
     except ProcessLookupError:pass
    break
   self.done.wait(1.)
 def start(self):self.thread.start();return self
 def close(self):
  self.done.set();self.thread.join(timeout=6)
  peaks=[r['own_renderer_bytes'] for r in self.rows if r['own_renderer_bytes'] is not None]
  if self.strict_ownership and not peaks and self.error is None:self.error='own renderer GPU memory was not observed; no zero peak is claimed'
  write_json(self.path,{'assigned_gpu':self.gpu,'renderer_pid':self.process.pid,'cap_bytes':int(self.cap),'training_free_reserve_bytes':int(self.reserve),'own_renderer_peak_bytes':max(peaks,default=None),'own_renderer_memory_observed':bool(peaks),'minimum_free_bytes':min((r['free_bytes'] for r in self.rows),default=None),'samples':self.rows,'monitor_error':self.error,'sampling_note':'one-second NVML sampling; subsecond peaks may be missed; software stop is not a hardware memory partition'})
  if self.error:raise RuntimeError(self.error)

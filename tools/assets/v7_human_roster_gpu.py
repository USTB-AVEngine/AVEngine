"""One owned GPU process tree at a time, with free-memory reserve and evidence."""
import argparse,json,os,signal,subprocess,time,ctypes,xml.etree.ElementTree as ET
from pathlib import Path

def gpu(index):
 root=ET.fromstring(subprocess.check_output(['nvidia-smi','-i',str(index),'-q','-x'],text=True));g=root.find('gpu');free=int(g.findtext('fb_memory_usage/free').split()[0]);procs=[]
 for p in g.findall('processes/process_info'):
  m=p.findtext('used_memory','0 MiB')
  try:mem=int(m.split()[0])
  except ValueError:mem=0
  procs.append({'pid':int(p.findtext('pid')),'type':p.findtext('type'),'used_mib':mem,'name':p.findtext('process_name')})
 return {'uuid':g.findtext('uuid'),'free_mib':free,'processes':procs}

def descendants(root):
 pp={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:s=(p/'stat').read_text();pp[int(p.name)]=int(s[s.rfind(')')+2:].split()[1])
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 result={root}
 while True:
  more={p for p,parent in pp.items() if parent in result}
  if more<=result:return result
  result|=more

from contextlib import contextmanager
import fcntl

@contextmanager
def exclusive_gpu_job(root):
 root=Path(root)
 with (root/'gpu1_job.lock').open('a') as handle:
  try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError as error:raise RuntimeError('Another GPU job in this task has not exited; do not overlap jobs') from error
  try:yield
  finally:fcntl.flock(handle,fcntl.LOCK_UN)

def run(a):
 assert a.gpu==1,'This task authorizes only GPU 1'
 if not a.disk_root:raise ValueError('Task disk root is required for the scoped GPU job lock and reserve checks')
 with exclusive_gpu_job(a.disk_root):return _run_one_gpu(a)

def _run_one_gpu(a):
 assert a.gpu==1,'This task authorizes only GPU 1'
 disk_before=None
 if a.disk_root:
  from v7_roster_disk import check
  disk_before=check(a.disk_root,a.planned_disk_mib*1024**2)
 state=gpu(a.gpu);needed=a.seed_peak_mib+4096
 if state['free_mib']<needed:raise RuntimeError(f"GPU1 free {state['free_mib']} < peak seed {a.seed_peak_mib}+4096")
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=state['uuid'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
 if a.env_json:env.update(json.loads(a.env_json.read_text()))
 a.output.parent.mkdir(parents=True,exist_ok=True);log=a.output.with_suffix('.log')
 rec={'machine':os.uname().nodename,'gpu_index':1,'gpu_uuid':state['uuid'],'command':a.command,'seed_peak_mib':a.seed_peak_mib,'seed_peak_source':a.seed_source,'required_reserve_mib':4096,'preflight':state,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'samples':[],'disk_preflight':disk_before}
 with log.open('w')as f:
  parent_pid=os.getpid()
  def child_parent_death():
   ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)
   if os.getppid()!=parent_pid:os.kill(os.getpid(),signal.SIGTERM)
  def stop_signal(signum,frame):
   raise KeyboardInterrupt('GPU guardian received signal '+str(signum))
  for signum in [signal.SIGHUP,signal.SIGTERM]:signal.signal(signum,stop_signal)
  proc=subprocess.Popen(a.command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=child_parent_death);rec['pid']=proc.pid;rec['log']=str(log);a.output.write_text(json.dumps(rec,indent=2))
  peak=0
  try:
   while proc.poll()is None:
    if a.disk_root:
     try:rec['disk_latest']=check(a.disk_root)
     except RuntimeError as error:
      rec['stop_reason']=str(error);os.killpg(proc.pid,signal.SIGTERM)
    cur=gpu(a.gpu);owned=descendants(proc.pid)
    allgpu=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True));wrong=[]
    for device in allgpu.findall('gpu'):
     if device.findtext('uuid')==state['uuid']:continue
     for pidnode in device.findall('processes/process_info'):
      if int(pidnode.findtext('pid')) in owned:wrong.append({'gpu_uuid':device.findtext('uuid'),'pid':int(pidnode.findtext('pid'))})
    if wrong:
     rec['stop_reason']='owned process appeared on unauthorized GPU';rec['unauthorized_gpu_processes']=wrong;os.killpg(proc.pid,signal.SIGTERM)
    my=[p for p in cur['processes'] if p['pid']in owned];mem=sum(p['used_mib']for p in my);peak=max(peak,mem);rec['samples'].append({'time':time.time(),'free_mib':cur['free_mib'],'owned_processes':my})
    # Stop only this verified process group if the safety reserve is lost.
    if cur['free_mib']<4096:
     rec['stop_reason']='free memory reserve fell below 4096 MiB';os.killpg(proc.pid,signal.SIGTERM)
    a.output.write_text(json.dumps({**rec,'measured_peak_mib':peak},indent=2));time.sleep(1)
  except BaseException as error:
   rec['stop_reason']=str(error)or type(error).__name__
   if proc.poll()is None:os.killpg(proc.pid,signal.SIGTERM)
  rec.update(returncode=proc.wait(),measured_peak_mib=peak,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));a.output.write_text(json.dumps(rec,indent=2))
 return rec['returncode']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=1);p.add_argument('--seed-peak-mib',type=int,required=True);p.add_argument('--seed-source',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--env-json',type=Path);p.add_argument('--disk-root',type=Path);p.add_argument('--planned-disk-mib',type=int,default=512);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
 if a.command and a.command[0]=='--':a.command=a.command[1:]
 raise SystemExit(run(a))

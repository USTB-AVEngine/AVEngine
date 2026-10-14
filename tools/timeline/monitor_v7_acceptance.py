#!/usr/bin/env python3
"""Bound and inspect only this acceptance batch's verified process trees."""
import argparse,datetime,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path


def load(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def proc(pid):
 try:
  s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  return {'pid':pid,'ppid':int(s[1]),'cpu_s':(int(s[11])+int(s[12]))/os.sysconf('SC_CLK_TCK'),'start_ticks':s[19],'cmd':cmd,'state':s[0]}
 except (OSError,ValueError,IndexError):return None

def processes(roots):
 allp={int(p.name):proc(int(p.name))for p in Path('/proc').iterdir()if p.name.isdigit()};allp={p:v for p,v in allp.items()if v}
 owned=set(roots)
 for _ in range(12):
  more={p for p,v in allp.items()if v['ppid']in owned}
  if more<=owned:break
  owned.update(more)
 result=[]
 for pid in owned:
  if pid not in allp:continue
  v=allp[pid]
  try:
   v['pss_bytes']=next(int(s.split()[1])*1024 for s in (Path('/proc')/str(pid)/'smaps_rollup').read_text().splitlines()if s.startswith('Pss:'))
   v['wchan']=(Path('/proc')/str(pid)/'wchan').read_text().strip()
  except (OSError,StopIteration):v['pss_bytes']=0;v['wchan']='unknown'
  result.append(v)
 return result

def stop(records):
 for v in sorted(records,key=lambda v:v['pid'],reverse=True):
  actual=proc(v['pid'])
  if actual and actual['start_ticks']==v['start_ticks']and actual['cmd']==v['cmd']:
   os.kill(v['pid'],signal.SIGTERM)

def allocated(root):
 return int(subprocess.check_output(['du','-sx','-B1',str(root)],text=True).split()[0])


class AdoptedProcess:
 def __init__(self,record):self.pid=record['pid'];self.record=record
 def poll(self):
  live=proc(self.pid)
  return None if live and live['start_ticks']==self.record['start_ticks']and live['cmd']==self.record['cmd']and live['state']!='Z' else 0

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--smoke-only',action='store_true');a=p.parse_args();m=load(a.manifest);o=Path(m['output_root']);w=Path(m['worktree'])
 pending=[s for s in range(len(m['parents']))if not (o/'evidence/jobs'/f'{s:03d}.json').exists()];active={};completed=[];last_watch=0;last_size=0;size=0;previous={};peaks={'pss_bytes':0,'concurrent_parents':0};pause='';failed_reasons={}
 (o/'logs').mkdir(exist_ok=True)
 for f in (o/'evidence/jobs').glob('[0-9][0-9][0-9].json'):
  d=load(f)
  if d.get('status')=='running':
   record=proc(d['pid'])
   if record and str(o)in record['cmd']and 'run_v7_acceptance.py'in record['cmd']:
    active[d['slot']]={'process':AdoptedProcess(record),'stream':(o/'logs'/f"parent_{d['slot']:03d}.log").open('a')}

 for n in ['ue_tmp','ue_shm','ue_config']:(o/n).mkdir(exist_ok=True)
 while pending or active:
  now=datetime.datetime.now(datetime.timezone.utc);sg=now+datetime.timedelta(hours=8);mem=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:'));free=shutil.disk_usage('/data').free
  records=processes([v['process'].pid for v in active.values()]);pss=sum(v['pss_bytes']for v in records);limit=120_000_000_000 if sg.hour<4 else 140_000_000_000
  if time.monotonic()-last_size>120:size=allocated(o)+allocated(w);last_size=time.monotonic()
  if pss>limit or free<100_000_000_000 or size>60_000_000_000 or sg.hour>=7 and sg.minute>=24:
   pause='resource_or_delivery_deadline_stop';stop(records);pending=[]
  for slot,v in list(active.items()):
   rc=v['process'].poll()
   if rc is None:continue
   v['stream'].close();receipt=o/'evidence/jobs'/f'{slot:03d}.json';d=load(receipt)if receipt.exists()else{'slot':slot,'status':'failed','reason':'worker exited before producer receipt','exit_code':rc}
   completed.append(d);del active[slot]
   if d['status']!='passed':
    reason=d.get('reason','unknown');failed_reasons[reason]=failed_reasons.get(reason,0)+1
    if failed_reasons[reason]>=2:pause='repeated_failure_needs_review';write(o/'REPEATED_FAILURE.json',{'reason':reason,'count':failed_reasons[reason]})
  if time.monotonic()-last_watch>=300 or not last_watch:
   rows=[]
   for slot,v in active.items():
    tree=processes([v['process'].pid]);cpu=sum(r['cpu_s']for r in tree);old=previous.get(slot,{});log=o/'logs'/f'parent_{slot:03d}.log';mtime=log.stat().st_mtime if log.exists()else 0
    entry={'slot':slot,'pid':v['process'].pid,'cpu_s':cpu,'cpu_growth_s':cpu-old.get('cpu_s',0),'log_mtime':mtime,'log_updated':mtime>old.get('log_mtime',0),'pipe_write_pids':[r['pid']for r in tree if 'pipe_write'in r['wchan']]};rows.append(entry);previous[slot]=entry
   snap={'time_utc':now.isoformat(),'active':rows,'PSS_bytes':pss,'MemAvailable_bytes':mem,'data_free_bytes':free,'allocated_task_bytes':size,'completed':len(completed),'pause':pause}
   with (o/'WATCH_LOG.md').open('a')as f:f.write(now.isoformat()+' '+json.dumps(snap,ensure_ascii=False)+'\n')
   write(o/'evidence/WATCH_CURRENT.json',snap);print('WATCH',json.dumps(snap),flush=True);last_watch=time.monotonic()
  peaks['pss_bytes']=max(peaks['pss_bytes'],pss);peaks['concurrent_parents']=max(peaks['concurrent_parents'],len(active))
  smoke=(o/'SMOKE_PASSED.json').exists()
  if not pause and mem>=30_000_000_000 and free>=100_000_000_000 and size<=60_000_000_000:
   from avengine.timeline.v7_demand import read_demand_state,demand_snapshot
   state=read_demand_state(o/'inputs/demand_state.json');gaps=demand_snapshot(state,next_questions=70)['family_deficits']
   families={'binding':['QA-01','QA-02','QA-03','QA-08','QA-09','QA-20'],'human_motion':['QA-06','QA-13','QA-15','QA-16','QA-17','QA-18','QA-19'],'animal_motion':['QA-06','QA-13','QA-15','QA-16','QA-17','QA-18'],'count':['QA-22','QA-23','STARSS-SPEAKERS'],'occlusion':['QA-10','QA-03','QA-12'],'taxonomy':['QA-21','V7-CLASS34','STARSS-CLASS13'],'content_binding':['QA-26','QA-27','QA-28','V7-CONTENT-BIND']}
   def demand_order(slot):
    target=families[m['parents'][slot]['focus']];return (-sum(max(0.,gaps.get(f,0.))for f in target)/len(target),slot)
   for slot in sorted(pending,key=demand_order):
    task=m['parents'][slot];moving=task.get('focus')in {'human_motion','animal_motion'};ue=task['device']=='ue'
    if not smoke and (slot!=0 or active or completed):continue
    if a.smoke_only and slot!=0:continue
    if len(active)>=16 or ue and any(m['parents'][s]['device']=='ue'for s in active):continue
    reserve=sum(30_000_000_000 if m['parents'][s].get('focus')in{'human_motion','animal_motion'}else 7_000_000_000 for s in active)+ (30_000_000_000 if moving else 7_000_000_000)
    if max(reserve,pss+(30_000_000_000 if moving else 7_000_000_000))>limit-5_000_000_000:continue
    env=dict(os.environ);cmd=[sys.executable,str(w/'tools/timeline/run_v7_acceptance.py'),'--manifest',str(a.manifest),'--worker',str(slot)]
    if ue:
     for k in ['LD_PRELOAD','LIBGL_ALWAYS_SOFTWARE','MESA_LOADER_DRIVER_OVERRIDE','EGL_PLATFORM','__EGL_VENDOR_LIBRARY_FILENAMES']:env.pop(k,None)
     env.update(CUDA_VISIBLE_DEVICES='2',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
     prefix=['/usr/bin/bwrap','--unshare-user','--ro-bind','/','/','--dev','/dev','--proc','/proc','--bind',str(o),str(o),'--bind',str(o/'ue_tmp'),'/tmp','--bind',str(o/'ue_shm'),'/dev/shm','--bind',str(o/'ue_config'),'/data/jzy/.config']
     for n in ['nvidiactl','nvidia2','nvidia-uvm','nvidia-uvm-tools']:
      if Path('/dev/'+n).exists():prefix+=['--dev-bind','/dev/'+n,'/dev/'+n]
     cmd=prefix+['--chdir',str(w)]+cmd
    stream=(o/'logs'/f'parent_{slot:03d}.log').open('w');process=subprocess.Popen(['nice','-n','10',*cmd],env=env,cwd=w,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    active[slot]={'process':process,'stream':stream};pending.remove(slot);write(o/'evidence/launches'/f'{slot:03d}.json',{'pid':process.pid,'argv':cmd,'log':str(o/'logs'/f'parent_{slot:03d}.log'),'started_utc':now.isoformat()});print('START',slot,process.pid,flush=True)
    if not smoke:break
  if not smoke and completed:
   if completed[0]['status']=='passed':write(o/'SMOKE_PASSED.json',completed[0]);smoke=True
   else:pause='smoke_failed';pending=[]
  if a.smoke_only and completed and not active:pending=[]
  if (o/'RESUME_AFTER_REVIEW.json').exists()and pause=='repeated_failure_needs_review':pause='';failed_reasons={};(o/'RESUME_AFTER_REVIEW.json').rename(o/f'REVIEWED_{time.time_ns()}.json')
  time.sleep(10)
 write(o/'evidence/MONITOR_FINAL.json',{'completed':len(completed),'pause':pause,'peaks':peaks});print('FINISHED',len(completed),pause,flush=True)

if __name__=='__main__':main()

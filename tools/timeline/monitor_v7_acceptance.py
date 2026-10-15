#!/usr/bin/env python3
"""Admit and inspect acceptance jobs; resource guards never terminate active work."""
import argparse,datetime,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path


def load(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def proc(pid):
 try:
  s=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  return {'pid':pid,'ppid':int(s[1]),'cpu_s':sum(int(s[i])for i in (11,12,13,14))/os.sysconf('SC_CLK_TCK'),'start_ticks':s[19],'cmd':cmd,'state':s[0]}
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

def combined_peak_budget(active_reserve,local_pss,requested_reserve,peer_pss):
 return max(active_reserve+requested_reserve,local_pss+requested_reserve)+peer_pss


def parent_launch_headroom(active_reserve,local_pss,requested,limit,available_memory):
 return combined_peak_budget(active_reserve,local_pss,requested,0)<=limit and available_memory-max(0,active_reserve-local_pss)-requested>=30_000_000_000


def admission_limits(m,now):
 sg=now+datetime.timedelta(hours=8)
 return {"maximum_pss_bytes":int(m.get("maximum_pss_bytes",120_000_000_000 if sg.hour<4 else 140_000_000_000)),
         "maximum_parents":int(m.get("maximum_parents",16)),"maximum_cpu_processes":80}


def output_processes(outputs):
 outputs=[str(Path(output).resolve())for output in outputs]
 roots=[]
 for path in Path("/proc").iterdir():
  if not path.name.isdigit():continue
  live=proc(int(path.name))
  if live and live["state"]!="Z"and any(output in live["cmd"]for output in outputs):roots.append(live["pid"])
 return processes(roots)


def peer_processes(m):
 # Helpers may own independent SDK prefetch processes as well as their parent
 # worker. Match only the three explicitly declared helper output roots, and
 # include their descendants without ever taking ownership or signalling them.
 outputs=[str(Path(output).resolve()) for output in m.get('peer_output_roots',[])]
 if not outputs:return []
 roots=[]
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  live=proc(int(path.name))
  if live and live['state']!='Z'and any(output in live['cmd'] for output in outputs):roots.append(live['pid'])
 return processes(roots)


def launch_pause_reason(*, now, deadline, pss, limit, free, allocated_size, available_memory,combined_pss=0,combined_limit=None,disk_policy=None):
 if now >= deadline:return 'delivery_deadline_no_new_jobs'
 if pss >= limit:return 'PSS_limit_no_new_jobs'
 if combined_limit is not None and combined_pss>=combined_limit:return 'combined_PSS_limit_no_new_jobs'
 if available_memory<20_000_000_000:return 'critical_memory_ranked_alert_no_new_jobs'
 from avengine.timeline.v7_storage_budget import launch_minimum_free_bytes
 if free < launch_minimum_free_bytes(disk_policy,now):return 'disk_headroom_no_new_jobs'
 if allocated_size >= 60_000_000_000:return 'task_disk_limit_no_new_jobs'
 if available_memory < 30_000_000_000:return 'memory_headroom_no_new_jobs'
 return ''


def append_progress(o,m,now,active):
 report_path=o/'PILOT_SUMMARY.json';report=load(report_path)if report_path.exists()else{}
 scenes=[m['parents'][slot]['row']['scene_id']for slot in active]
 with(o/'PROGRESS.md').open('a')as f:
  f.write(now.isoformat()+' 构造/续跑场景='+','.join(scenes)+'；当前正式去重后模态平衡独立题='+str(report.get('independent_questions',0))+'；房子='+str(report.get('houses',0))+'；>=3题族='+str(report.get('families_at_least_three',0))+'；统计时间='+str(report.get('generated_utc','awaiting first final reader/report'))+'；题数只取PILOT_SUMMARY，原逐父候选/原始表不冒充最终总数。\n')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--smoke-only',action='store_true');a=p.parse_args();m=load(a.manifest);o=Path(m['output_root']);w=Path(m['worktree'])
 pending=[s for s in range(len(m['parents']))if not (o/'evidence/jobs'/f'{s:03d}.json').exists()];active={};completed=[];last_watch=0;last_size=0;last_progress=0;size=0;previous={};peaks={'pss_bytes':0,'concurrent_parents':0};pause='';failed_reasons={};finalizer=None;last_finalize=0.
 (o/'logs').mkdir(exist_ok=True)
 if m.get('live_resource_guard_enabled',True):
  end=datetime.datetime.fromisoformat(m.get('stop_new_jobs_utc','2026-10-15T07:00:00+00:00'))+datetime.timedelta(minutes=30)
  stream=(o/'logs/live_resource_guard.log').open('a')
  guard=subprocess.Popen(['nice','-n','10',sys.executable,str(Path(__file__).with_name('guard_v7_live_resources.py')),'--manifest',str(a.manifest),'--until-utc',end.isoformat()],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  stream.close();write(o/'evidence/live_resource_guard_launch.json',{'pid':guard.pid,'source':str(Path(__file__).with_name('guard_v7_live_resources.py')),'until_utc':end.isoformat(),'signals_workers':False})
 for f in (o/'evidence/jobs').glob('[0-9][0-9][0-9].json'):
  d=load(f)
  if d.get('status')=='running':
   record=proc(d['pid'])
   if record and str(o)in record['cmd']and 'run_v7_acceptance.py'in record['cmd']:
    active[d['slot']]={'process':AdoptedProcess(record),'stream':(o/'logs'/f"parent_{d['slot']:03d}.log").open('a')}

 for launch in (o/'evidence/launches').glob('*.json'):
  slot=int(launch.stem)
  if slot in active:continue
  d=load(launch);live=proc(d['pid'])
  if live and str(o)in live['cmd'] and 'run_v7_acceptance.py'in live['cmd'] and ('--worker '+str(slot)+' ')in (live['cmd']+' '):
   active[slot]={'process':AdoptedProcess(live),'stream':(o/'logs'/f'parent_{slot:03d}.log').open('a')}
 pending=[slot for slot in pending if slot not in active]

 for n in ['ue_tmp','ue_shm','ue_config']:(o/n).mkdir(exist_ok=True)
 while pending or active:
  m=load(a.manifest)
  if m.get('rebalance_owned_cpu_contention') and time.monotonic()-locals().get('last_cpu_rebalance',0)>=300:
   from avengine.timeline.v7_cpu_placement import rebalance_owned_parents
   rebalance_owned_parents(m,a.manifest);last_cpu_rebalance=time.monotonic()
  pending=sorted(set(pending)|{slot for slot in range(len(m['parents']))if slot not in active and not (o/'evidence/jobs'/f'{slot:03d}.json').exists()})
  now=datetime.datetime.now(datetime.timezone.utc);sg=now+datetime.timedelta(hours=8);mem=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:'));free=shutil.disk_usage('/data').free
  records=output_processes([o]);pss=sum(v['pss_bytes']for v in records);peers=peer_processes(m);peer_pss=sum(v['pss_bytes']for v in peers);limits=admission_limits(m,now);limit=limits["maximum_pss_bytes"];forecast_cpu=__import__('avengine.timeline.v7_cpu_budget',fromlist=['compute_process_count']).compute_process_count([*records,*peers])+2
  if time.monotonic()-last_size>120:size=allocated(o)+allocated(w);last_size=time.monotonic()
  deadline=datetime.datetime.fromisoformat(m.get('stop_new_jobs_utc','2026-10-15T04:00:00+00:00'))
  resource_pause=launch_pause_reason(now=now,deadline=deadline,pss=pss,limit=limit,free=free,allocated_size=size,available_memory=mem,combined_pss=pss+peer_pss,combined_limit=m.get('maximum_combined_pss_bytes',190_000_000_000),disk_policy=m.get('launch_disk_policy'))
  if resource_pause=='delivery_deadline_no_new_jobs':pending=[]
  from tools.timeline.finalize_v7_acceptance import delivery_fingerprint
  delivered=o/'evidence/FINAL_MODEL_READER.json'
  unchanged=delivered.is_file() and load(delivered).get('input_fingerprint')==delivery_fingerprint(m)
  if m.get('post_dedup_modality_balance')and not unchanged and (finalizer is None or finalizer.poll()is not None)and time.monotonic()-last_finalize>=300:
   current=Path(m['worktree']);entry=current/'tools/timeline/finalize_v7_acceptance.py'
   if entry.is_file():
    env=dict(os.environ);env['PYTHONPATH']=str(current/'src')+os.pathsep+str(current)+os.pathsep+env.get('PYTHONPATH','')
    log=(o/'logs/finalize_model_export.log').open('a');finalizer=subprocess.Popen(['nice','-n','10',sys.executable,str(entry),'--manifest',str(a.manifest)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();last_finalize=time.monotonic()
  if time.monotonic()-last_progress>=900 or not last_progress:
   append_progress(o,m,now,active);last_progress=time.monotonic()
  for slot,v in list(active.items()):
   rc=v['process'].poll()
   if rc is None:continue
   v['stream'].close();receipt=o/'evidence/jobs'/f'{slot:03d}.json';d=load(receipt)if receipt.exists()else{'slot':slot,'status':'failed','reason':'worker exited before producer receipt','exit_code':rc}
   completed.append(d);del active[slot]
   if d['status']=='passed':
    with(o/'PROGRESS.md').open('a')as f:f.write(now.isoformat()+' FRESH_PARENT_EXPORTED '+d['scene_id']+' 独立题='+str(d['independent_questions'])+' 房子='+d['house']+' 表='+str(o/'export/parent_rows'/f'{slot:03d}.jsonl')+'；四项原生导出门禁和reader通过。\n')
   if d['status']!='passed':
    from avengine.timeline.v7_construction_retry import queue_construction_retry
    retry=queue_construction_retry(a.manifest,d)
    if retry is not None:
     with(o/'PROGRESS.md').open('a')as f:f.write(now.isoformat()+' AUTO_CONSTRUCTION_RETRY failed_slot='+str(slot)+' next_slot='+str(retry)+'；保留人数/题族/基本种子，自动改初始朝向或同房其他合法区域，不动真值门禁。\n')
     continue
    from tools.timeline.generate_v7_demand_scenes import failure_signature
    reason=failure_signature(d);failed_reasons[reason]=failed_reasons.get(reason,0)+1
    if failed_reasons[reason]>=2:pause='repeated_failure_needs_review';write(o/'REPEATED_FAILURE.json',{'reason':reason,'count':failed_reasons[reason]})
  if time.monotonic()-last_watch>=300 or not last_watch:
   rows=[]
   for slot,v in active.items():
    tree=processes([v['process'].pid]);cpu=sum(r['cpu_s']for r in tree);old=previous.get(slot,{});log=o/'logs'/f'parent_{slot:03d}.log';mtime=log.stat().st_mtime if log.exists()else 0
    entry={'slot':slot,'pid':v['process'].pid,'cpu_s':cpu,'cpu_growth_s':cpu-old.get('cpu_s',0),'log_mtime':mtime,'log_updated':mtime>old.get('log_mtime',0),'pipe_write_pids':[r['pid']for r in tree if 'pipe_write'in r['wchan']]};rows.append(entry);previous[slot]=entry
   snap={'time_utc':now.isoformat(),'active':rows,'PSS_bytes':pss,'peer_PSS_bytes':peer_pss,'combined_PSS_bytes':pss+peer_pss,'MemAvailable_bytes':mem,'data_free_bytes':free,'allocated_task_bytes':size,'admission_limits':limits,'resource_limit_scope':'own acceptance workload; peers observed; host memory floor shared','completed':len(completed),'pause':pause or resource_pause}
   with (o/'WATCH_LOG.md').open('a')as f:f.write(now.isoformat()+' '+json.dumps(snap,ensure_ascii=False)+'\n')
   write(o/'evidence/WATCH_CURRENT.json',snap);print('WATCH',json.dumps(snap),flush=True);last_watch=time.monotonic()
  peaks['pss_bytes']=max(peaks['pss_bytes'],pss);peaks['concurrent_parents']=max(peaks['concurrent_parents'],len(active))
  smoke=(o/'SMOKE_PASSED.json').exists() or not m.get('require_single_parent_smoke',True)
  if not pause and not resource_pause:
   from avengine.timeline.v7_demand import read_demand_state,demand_snapshot
   state=read_demand_state(o/'inputs/demand_state.json');gaps=demand_snapshot(state,next_questions=70)['family_deficits']
   families={'binding':['QA-01','QA-02','QA-03','QA-08','QA-09','QA-20'],'human_motion':['QA-06','QA-13','QA-15','QA-16','QA-17','QA-18','QA-19'],'animal_motion':['QA-06','QA-13','QA-15','QA-16','QA-17','QA-18'],'count':['QA-22','QA-23','STARSS-SPEAKERS'],'occlusion':['QA-10','QA-03','QA-12'],'taxonomy':['QA-21','V7-CLASS34','STARSS-CLASS13'],'content_binding':['QA-26','QA-27','QA-28','V7-CONTENT-BIND']}
   def demand_order(slot):
    task=m['parents'][slot];explicit=task.get('target_family')
    target=[explicit]if explicit else families.get(task['focus'],['QA-06','QA-15','QA-16','QA-17'])
    observed=state.get('family_counts',{})
    floor=max((max(0,3-observed.get(f,0))for f in target),default=0)
    return (-floor,-sum(max(0.,gaps.get(f,0.))for f in target)/len(target),slot)
   for slot in sorted(pending,key=demand_order):
    task=m['parents'][slot]
    if task.get('launch_disabled'):continue
    w=Path(task.get('snapshot_worktree',m['worktree']))
    moving=task.get('focus')in {'human_motion','animal_motion','motion_quartet_template'};ue=task['device']=='ue'
    if not smoke and m.get('require_single_parent_smoke',True) and (slot!=0 or active or completed):continue
    if a.smoke_only and slot!=0:continue
    if len(active)>=limits["maximum_parents"] or ue and any(m['parents'][s]['device']=='ue'for s in active):continue
    reserve=sum(30_000_000_000 if m['parents'][s].get('focus')in{'human_motion','animal_motion','motion_quartet_template'}else 7_000_000_000 for s in active)
    requested=30_000_000_000 if moving else 7_000_000_000
    if not parent_launch_headroom(reserve,pss,requested,limit,mem):continue
    if pss+peer_pss+requested>int(m.get('maximum_combined_pss_bytes',190_000_000_000)):continue
    if forecast_cpu+4>limits["maximum_cpu_processes"]:continue
    env=dict(os.environ);env['PYTHONPATH']=str(w/'src')+os.pathsep+str(w)+os.pathsep+env.get('PYTHONPATH','');cmd=[sys.executable,str(w/'tools/timeline/run_v7_acceptance.py'),'--manifest',str(a.manifest),'--worker',str(slot)]
    if ue:
     for k in ['LD_PRELOAD','LIBGL_ALWAYS_SOFTWARE','MESA_LOADER_DRIVER_OVERRIDE','EGL_PLATFORM','__EGL_VENDOR_LIBRARY_FILENAMES']:env.pop(k,None)
     env.update(CUDA_VISIBLE_DEVICES='2',VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
     prefix=['/usr/bin/bwrap','--unshare-user','--ro-bind','/','/','--dev','/dev','--proc','/proc','--bind',str(o),str(o),'--bind',str(o/'ue_tmp'),'/tmp','--bind',str(o/'ue_shm'),'/dev/shm','--bind',str(o/'ue_config'),'/data/jzy/.config']
     for n in ['nvidiactl','nvidia2','nvidia-uvm','nvidia-uvm-tools']:
      if Path('/dev/'+n).exists():prefix+=['--dev-bind','/dev/'+n,'/dev/'+n]
     cmd=prefix+['--chdir',str(w)]+cmd
    stream=(o/'logs'/f'parent_{slot:03d}.log').open('x');process=subprocess.Popen(['nice','-n','10',*cmd],env=env,cwd=w,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    active[slot]={'process':process,'stream':stream};forecast_cpu+=4;pending.remove(slot);write(o/'evidence/launches'/f'{slot:03d}.json',{'pid':process.pid,'argv':cmd,'log':str(o/'logs'/f'parent_{slot:03d}.log'),'started_utc':now.isoformat()});print('START',slot,process.pid,flush=True)
    if not smoke and m.get('require_single_parent_smoke',True):break
  if not smoke and completed and m.get('require_single_parent_smoke',True):
   if completed[0]['status']=='passed':write(o/'SMOKE_PASSED.json',completed[0]);smoke=True
   else:pause='smoke_failed';pending=[]
  if a.smoke_only and completed and not active:pending=[]
  if (o/'RESUME_AFTER_REVIEW.json').exists()and pause=='repeated_failure_needs_review':pause='';failed_reasons={};(o/'RESUME_AFTER_REVIEW.json').rename(o/f'REVIEWED_{time.time_ns()}.json')
  time.sleep(10)
 write(o/'evidence/MONITOR_FINAL.json',{'completed':len(completed),'pause':pause,'peaks':peaks});print('FINISHED',len(completed),pause,flush=True)

if __name__=='__main__':main()

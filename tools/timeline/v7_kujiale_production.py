"""Bounded Kujiale v7-S production; original formal pipeline and gates.

A passed one-house canary is compulsory. Sealed batches are moved only after
CCI SHA readback, and this coordinator deletes only its own generated files.
"""
from pathlib import Path
from collections import Counter
import argparse,json,os,subprocess,time,signal,hashlib,shutil,traceback
import psutil
ROOT=Path('/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003')
RUN=ROOT/'ue_round14';REPO=Path('/data/jzy/code/AVEngine-codex-v7-kujiale-round14-20261005')
GENERATOR='33b5b12df1a45ca265bf0c4fbe9b26b664577a7c'
CCI='/mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_s_kujiale_20261005'

def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def progress(message):
 with (ROOT/'PROGRESS_zh.md').open('a') as f:f.write('\n第14回合 '+time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())+'：'+message+'\n')

def safe():
 v=json.loads((RUN/'resource_sampler/resource_snapshot.json').read_text())
 if time.time()-v['sample_epoch']>8 or v['disk_stop_reason']:raise RuntimeError('resource snapshot unsafe: '+str(v))
 if (RUN/'STOP_PRODUCTION').exists():raise RuntimeError('own STOP_PRODUCTION requested')
 return v

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--max-workers',type=int,default=4)
 args=parser.parse_args()
 if not 1<=args.max_workers<=4 or os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.nice(0)<10 or len(os.sched_getaffinity(0))>16:raise ValueError('GPU2 / nice10 / CPU16 contract missing')
 canary=json.loads((RUN/'evidence/canary_result.json').read_text())
 if canary['status']!='pass':raise RuntimeError('formal one-house canary has not passed')
 canary_release=Path(canary['release']);cm=json.loads((canary_release/'DELIVERY_MANIFEST.json').read_text())
 if cm['generator_commit']!=GENERATOR or cm['render_path']!='ue_spear':raise ValueError('different formal generator/native backend')
 rows=[json.loads(s) for s in (RUN/'inputs/kujiale_v7_S_train.jsonl').read_text().splitlines() if s.strip()]
 counts=Counter((r['house'],r['room_id']) for r in rows)
 if len(rows)!=132 or len(counts)!=33 or set(counts.values())!={4} or any(r['split']!='train' for r in rows):raise ValueError('frozen 33 x4 train-only budget differs')
 from avengine.timeline.v7_storage_budget import configure_kujiale_production
 from avengine.timeline.v7_alpha_delivery import export_alpha
 from avengine.timeline.v7_s_batch_checks import check_critical_delivery,open_delivered_media
 # One writer, with actual process identity retained for safe interruption.
 lock=RUN/'production_coordinator.lock'
 try:lock.mkdir()
 except FileExistsError:raise RuntimeError('coordinator lock exists: verify previous PID before resuming')
 proc=psutil.Process();atomic(lock/'owner.json',{'pid':proc.pid,'create_time':proc.create_time(),'command':proc.cmdline()})
 started=time.time();state={'status':'running','generator_commit':GENERATOR,'planned_parents':132,'batches':[],'formal_parents':0,'formal_windows':0,'formal_questions':0,'transferred_parents':0,'family_counts':{},'speakers_per_window':{},'pid':os.getpid(),'started_epoch':started};active=None
 def stop(number,frame):
  (RUN/'STOP_PRODUCTION').touch()
  if active and active.poll() is None:
   # The guard handles SIGTERM and verifies every descendant before stopping it.
   active.terminate()
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 queue={'schema':'kujiale_v7_s_transfer_queue_v1','complete':False,'batches':[]}
 peak=max(3814.,float(canary['peak_mib']))
 canary_ids={r['scene_id'] for r in map(json.loads,(RUN/'inputs/canary.jsonl').read_text().splitlines())}
 pending=[r for r in rows if r['scene_id'] not in canary_ids]
 # Complete this same house first, including its quartet parents; no recipe
 # or seed is changed, only dispatch order.
 canary_house=next(r['house'] for r in rows if r['scene_id'] in canary_ids)
 pending.sort(key=lambda r:r['house']!=canary_house)
 configure_kujiale_production(RUN/'production',RUN,workers=args.max_workers)
 def deliver(name,release,phase,seedrows):
  nonlocal state,queue
  safe();summary=json.loads((release/'summary.json').read_text())
  coverage=check_critical_delivery(release,require_full=False);media=open_delivered_media(release)
  if coverage['status']!='pass' or media['status']!='pass':raise ValueError('original formal gates failed before transfer')
  atomic(release/'DELIVERY_MANIFEST.json',{**summary,'status':'sealed','generator_commit':GENERATOR,'native_backend_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'render_path':'ue_spear','batch_id':name,'coverage':coverage,'media_readback':media,'additional_training_batch':True,'primary_500_parent_release':False,'expected_total_additional_parents':132})
  files=[]
  for p in sorted(release.rglob('*')):
   if p.is_symlink():raise ValueError('delivery symlink forbidden')
   if p.is_file():files.append({'relative':str(p.relative_to(release)),'bytes':p.stat().st_size,'sha256':sha(p)})
  atomic(release/'TRANSFER_SHA256.json',{'schema':'v7_delivery_sha256_v1','batch_id':name,'files':files})
  listing=RUN/'transfer_lists'/(name+'.txt');listing.parent.mkdir(exist_ok=True)
  listing.write_text(''.join(str(p).lstrip('/')+'\n' for p in sorted(release.rglob('*')) if p.is_file()))
  item={'batch_id':name,'list_path':str(listing),'release_path':str(release),'parents':len(seedrows),'formal_windows':summary['formal_windows'],'formal_questions':summary['formal_logical_questions']}
  queue['batches'].append(item);atomic(RUN/'transfer_queue.json',queue)
  state['formal_parents']+=len(seedrows);state['formal_windows']+=summary['formal_windows'];state['formal_questions']+=summary['formal_logical_questions']
  state['family_counts']=dict(Counter(state['family_counts'])+Counter(summary['family_counts']))
  state['speakers_per_window']=dict(Counter(state['speakers_per_window'])+Counter(summary['actual_speakers_per_window']))
  entry={**item,'delivered_bytes':sum(r['bytes'] for r in files),'status':'sealed_awaiting_CCI_SHA'};state['batches'].append(entry);atomic(RUN/'PRODUCTION_STATUS.json',state)
  evidence=RUN/'batch_evidence'/name;evidence.mkdir(parents=True)
  for file in ['summary.json','DELIVERY_MANIFEST.json','parents.csv','windows.csv','rejections.json','TRANSFER_SHA256.json']:shutil.copyfile(release/file,evidence/file)
  progress(f"{name} 已封存 {len(seedrows)} 父场景、{summary['formal_windows']} 窗口、{summary['formal_logical_questions']} 题；等待 CCI SHA 复核，不先删副本。")
  receipt=RUN/'transfer_receipts'/(name+'.json');deadline=time.time()+7200
  while not receipt.is_file():
   safe()
   if time.time()>deadline:raise RuntimeError('CCI transfer timed out; source retained')
   time.sleep(5)
  proof=json.loads(receipt.read_text())
  if proof['status']!='verified' or proof['batch_id']!=name or proof['file_count']!=len(files) or proof['bytes']!=sum(r['bytes'] for r in files):raise ValueError('CCI verification receipt differs')
  # Preserve a complete index locally; CCI media paths resolve to this exact
  # SHA-verified release. No incomplete batch is exposed as a complete 132.
  index=RUN/'additional_train_index.jsonl'
  with index.open('a') as out:
   for line in (release/'train.jsonl').read_text().splitlines():
    r=json.loads(line)
    for k in ('video','audio_foa','audio_mono','private_evidence'):r[k]=str(Path(proof['release_directory'])/r[k])
    r['delivery_batch_id']=name;out.write(json.dumps(r,ensure_ascii=False)+'\n')
  for target in [release,phase/'production']:
   if not target.resolve().is_relative_to(RUN.resolve()) or target.is_symlink():raise ValueError('cleanup escaped own Round14')
   if target.is_dir():shutil.rmtree(target)
  entry.update(status='CCI_SHA_verified_local_copies_removed',CCI_release=proof['release_directory'],transfer_wall_seconds=proof['wall_seconds'])
  state['transferred_parents']+=len(seedrows);state['wall_seconds']=time.time()-started;atomic(RUN/'PRODUCTION_STATUS.json',state)
  atomic(evidence/'CCI_verified.json',proof);progress(f"{name} CCI {proof['file_count']} 文件 SHA 全一致；已删除本任务该批中间件和 48g 交付副本。")
 try:
  deliver('batch_000',canary_release,canary_release.parent,[r for r in rows if r['scene_id'] in canary_ids])
  for i in range(0,len(pending),4):
   safe();batch=pending[i:i+4];name=f'batch_{i//4+1:03d}';phase=RUN/'production_batches'/name;phase.mkdir(parents=True,exist_ok=False)
   snap=safe();free=snap['gpus'][2]['free_mib'];workers=min(args.max_workers,len(batch),int(max(0,free-4096)//peak))
   if workers<1:raise RuntimeError('GPU2 lacks measured peak +4GB; no foreign process touched')
   seeds=phase/'seeds.jsonl';seeds.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in batch))
   cmd=[os.sys.executable,str(REPO/'tools/timeline/generate_v7_registered.py'),'--houses',str(RUN/'inputs/houses.json'),'--seeds',str(seeds),'--runtime-inputs',str(RUN/'inputs/runtime.json'),'--assets',str(RUN/'inputs/assets.json'),'--source-admission',str(RUN/'inputs/source_admission_v2.json'),'--human-admission',str(RUN/'inputs/human_admission.json'),'--output',str(phase/'production'),'--storage-budget-root',str(RUN),'--workers',str(workers),'--limit',str(len(batch)),'--render-backend','ue_spear']
   request={'budget_root':str(ROOT),'run_directory':str(phase/'guard'),'command':cmd,'cwd':str(REPO),'gpu_index':2,'cpu_affinity':list(range(80,96)),'previous_peak_mib':workers*peak,'minimum_free_mib':workers*peak+4096,'minimum_disk_free_bytes':100_000_000_000,'storage_limit_bytes':snap['logical_bytes']+max(0,30_000_000_000-snap['new_allocated_bytes']),'reserve_bytes':1_000_000_000,'stop_margin_bytes':500_000_000,'maximum_seconds':21600,'allowed_auxiliary_gpu_indices':[0],'maximum_auxiliary_gpu_mib':64.,'resource_snapshot_path':str(RUN/'resource_sampler/resource_snapshot.json'),'environment':{'CUDA_VISIBLE_DEVICES':'2','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}}
   atomic(phase/'guard_request.json',request);state.update(active_batch=name,active_workers=workers);atomic(RUN/'PRODUCTION_STATUS.json',state)
   with (phase/'driver.log').open('x') as log:active=subprocess.Popen([os.sys.executable,str(REPO/'tools/rooms/v7_kujiale_ue_guard.py'),'--request',str(phase/'guard_request.json')],stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
   atomic(phase/'guard_process.json',{'pid':active.pid,'create_time':psutil.Process(active.pid).create_time(),'command':psutil.Process(active.pid).cmdline()})
   result=active.wait();active=None;guard=json.loads((phase/'guard/result.json').read_text())
   if result or guard['status']!='pass':raise RuntimeError('formal batch failed; original failure receipts retained: '+name)
   peak=max(peak,guard['peak_mib']/workers)
   summary=export_alpha(seeds,[phase/'production'],phase/'release',expected_parents=len(batch));deliver(name,phase/'release',phase,batch)
  state.update(status='complete',wall_seconds=time.time()-started);queue['complete']=True;atomic(RUN/'transfer_queue.json',queue)
  progress(f"酷家乐 v7-S 已完成并转存：{state['formal_parents']} 父场景、{state['formal_windows']} 窗口、{state['formal_questions']} 题；墙钟 {state['wall_seconds']/3600:.3f} 小时。")
 except BaseException as e:
  if active and active.poll() is None:active.terminate();active.wait(timeout=60)
  state.update(status='stopped_not_complete',error=str(e),traceback=traceback.format_exc(),wall_seconds=time.time()-started)
  progress('量产停止；未放宽门槛、未换种子：'+str(e))
  raise
 finally:
  atomic(RUN/'PRODUCTION_STATUS.json',state)
if __name__=='__main__':main()

"""Resume fixed v7 UE requests without padding rejected scenes or changing seeds."""
from pathlib import Path
from collections import Counter
import argparse,json,os,subprocess,time,signal,hashlib,shutil,traceback
import psutil
ROOT=Path('/data/jzy/tmp/claude_hardbank_20260923/v7_kujiale_20261003');RUN=ROOT/'ue_round14';REPO=Path('/data/jzy/code/AVEngine-codex-v7-kujiale-round14-20261005')
GENERATOR='33b5b12df1a45ca265bf0c4fbe9b26b664577a7c'

def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def progress(message):
 with (ROOT/'PROGRESS_zh.md').open('a') as f:f.write('\n第14回合 '+time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())+'：'+message+'\n')

def safe():
 v=json.loads((RUN/'resource_sampler/resource_snapshot.json').read_text())
 if time.time()-v['sample_epoch']>8 or v['disk_stop_reason']:raise RuntimeError('resource snapshot unsafe')
 if (RUN/'STOP_PRODUCTION').exists():raise RuntimeError('own STOP_PRODUCTION requested')
 return v

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def safe_own_path(p):
 p=Path(p)
 if p.is_symlink() or not p.resolve().is_relative_to(RUN.resolve()):raise ValueError('own cleanup escaped Round14')
 return p

def retain_witness_then_remove(parent,label):
 parent=safe_own_path(parent)
 if not parent.is_dir():return
 target=RUN/'scene_witnesses'/label/parent.name;target.mkdir(parents=True,exist_ok=True)
 # All original JSON readbacks, rejected QA pools, and recipes stay private.
 for f in parent.rglob('*.json'):
  dest=target/f.relative_to(parent);dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():shutil.copyfile(f,dest)
 # A small original image witness per world is sufficient for visual review;
 # retain exact PNG SHA as provenance while storing JPEG as required.
 from PIL import Image
 images=[]
 for world in parent.glob('visual_*_attempt_*'):
  pngs=sorted(world.glob('native_frame_*.png'))
  for f in [pngs[i] for i in sorted({0,len(pngs)//2,len(pngs)-1})] if pngs else []:
   dest=target/world.name/(f.stem+'.jpg');dest.parent.mkdir(exist_ok=True);Image.open(f).convert('RGB').save(dest,'JPEG',quality=90);images.append({'original_png':str(f),'original_sha256':sha(f),'witness_JPEG':str(dest)})
 atomic(target/'WITNESS.json',{'status':'witness_retained_before_own_intermediate_cleanup','source_parent':str(parent),'images':images,'original_data_deleted':False,'scene_not_resampled':True})
 shutil.rmtree(parent)

def scientific_rejection(result):
 if result.get('error_type')!='ValueError':return False
 message=result.get('reason','')
 return any(word in message for word in ['legal candidates','counterfactual','qualified placement','native appearance reviews','static contact','body envelopes overlap','environmental emitter direction not admitted','native material or mask restoration qualification','original native highlight/contact','complete utterance fitting','admitted non-speech mouth-source deficit'])

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--max-workers',type=int,default=4);args=parser.parse_args()
 if not 1<=args.max_workers<=4 or os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.nice(0)<10 or len(os.sched_getaffinity(0))>16:raise ValueError('GPU2 / nice10 / CPU16 contract missing')
 from avengine.timeline.v7_registered_generation import PRODUCER_QA_REVISION
 from avengine.timeline.v7_storage_budget import configure_kujiale_production
 from avengine.timeline.v7_alpha_delivery import export_alpha
 from avengine.timeline.v7_s_batch_checks import check_critical_delivery,open_delivered_media
 canary=json.loads((RUN/'evidence/canary_result.json').read_text())
 if canary.get('status')!='pass' or canary.get('producer_qa_revision')!=PRODUCER_QA_REVISION:raise RuntimeError('corrected one-house canary required')
 rows=[json.loads(s) for s in (RUN/'inputs/kujiale_v7_S_train.jsonl').read_text().splitlines() if s.strip()];byid={r['scene_id']:r for r in rows};counts=Counter((r['house'],r['room_id']) for r in rows)
 if len(rows)!=132 or len(counts)!=33 or set(counts.values())!={4} or any(r['split']!='train' for r in rows):raise ValueError('fixed 33 x4 train-only requests differ')
 # A new private lock keeps the old stopped coordinator's identity intact.
 lock=RUN/'production_coordinator_v2.lock';lock.mkdir(exist_ok=True)
 if (lock/'owner.json').is_file():
  owner=json.loads((lock/'owner.json').read_text())
  try:
   p=psutil.Process(owner['pid'])
   if p.create_time()==owner['create_time'] and p.status()!=psutil.STATUS_ZOMBIE:raise RuntimeError('own v2 coordinator still live')
  except psutil.NoSuchProcess:pass
  with (lock/'previous_owners.jsonl').open('a') as f:f.write(json.dumps(owner)+'\n')
 proc=psutil.Process();atomic(lock/'owner.json',{'pid':proc.pid,'create_time':proc.create_time(),'command':proc.cmdline()})
 state=json.loads((RUN/'PRODUCTION_STATUS.json').read_text());queue=json.loads((RUN/'transfer_queue.json').read_text());started=time.time();active=None
 native_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
 if state.get('coordinator_schema')!='fixed-v7-UE-resume-v2':
  state.update(coordinator_schema='fixed-v7-UE-resume-v2',producer_qa_revision=PRODUCER_QA_REVISION,dispatches={},rejections={},attempted_parent_ids=[],generation_sessions=[])
  old=json.loads((RUN/'evidence/endpoint_fix_requalification.json').read_text())
  for failed in old['results']:
   if failed['status']!='failed_original_gate':raise ValueError('unexpected previous requalification state')
   state['rejections'][failed['scene_id']]=failed;state['attempted_parent_ids'].append(failed['scene_id'])
  requalified=json.loads((RUN/'evidence/sealed_media_requalification.json').read_text())
  for failed in requalified['parents']:
   if failed['status']=='formal_alpha_admitted':continue
   if failed['status']!='failed_original_gate' or failed.get('error_type')!='ValueError':raise ValueError('sealed media requalification interface error remains')
   state['rejections'][failed['scene_id']]=failed;state['attempted_parent_ids'].append(failed['scene_id'])
 state.update(status='running',pid=os.getpid(),started_epoch=started,native_backend_commit=native_commit,error=None,active_workers=0)
 state['generation_sessions'].append({'started_epoch':started,'pid':os.getpid(),'native_backend_commit':native_commit})
 atomic(RUN/'PRODUCTION_STATUS.json',state)
 def stop(number,frame):
  (RUN/'STOP_PRODUCTION').touch()
  if active and active.poll() is None:active.terminate()
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 peak=max(3814.,float(canary['peak_mib']))
 configure_kujiale_production(RUN/'production',RUN,workers=args.max_workers)
 def deliver(name,release,phase,seedrows,remove_whole_phase=False):
  nonlocal state,queue
  safe();release=safe_own_path(release);phase=safe_own_path(phase)
  evidence=RUN/'batch_evidence'/name;entry=next((e for e in state['batches'] if e['batch_id']==name),None)
  if entry and entry['status']=='CCI_SHA_verified_local_copies_removed':return
  if not entry:
   summary=json.loads((release/'summary.json').read_text());coverage=check_critical_delivery(release,require_full=False);media=open_delivered_media(release)
   if coverage['status']!='pass' or media['status']!='pass':raise ValueError('original per-window delivery gates failed')
   atomic(release/'DELIVERY_MANIFEST.json',{**summary,'status':'sealed','generator_commit':GENERATOR,'producer_qa_revision':PRODUCER_QA_REVISION,'native_backend_commit':native_commit,'render_path':'ue_spear','batch_id':name,'coverage':coverage,'media_readback':media,'additional_training_batch':True,'primary_500_parent_release':False,'expected_total_additional_requests':132})
   files=[{'relative':str(p.relative_to(release)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(release.rglob('*')) if p.is_file() and p.name!='TRANSFER_SHA256.json']
   if any(p.is_symlink() for p in release.rglob('*')):raise ValueError('delivery symlink forbidden')
   atomic(release/'TRANSFER_SHA256.json',{'schema':'v7_delivery_sha256_v1','batch_id':name,'files':files})
   listing=RUN/'transfer_lists'/(name+'.txt');listing.write_text(''.join(str(p).lstrip('/')+'\n' for p in sorted(release.rglob('*')) if p.is_file()))
   item={'batch_id':name,'list_path':str(listing),'release_path':str(release),'parents':len(seedrows),'formal_windows':summary['formal_windows'],'formal_questions':summary['formal_logical_questions']}
   queue['batches'].append(item);atomic(RUN/'transfer_queue.json',queue)
   state['formal_parents']+=len(seedrows);state['formal_windows']+=summary['formal_windows'];state['formal_questions']+=summary['formal_logical_questions'];state['family_counts']=dict(Counter(state['family_counts'])+Counter(summary['family_counts']));state['speakers_per_window']=dict(Counter(state['speakers_per_window'])+Counter(summary['actual_speakers_per_window']))
   state['attempted_parent_ids']=sorted(set(state['attempted_parent_ids'])|{r['scene_id'] for r in seedrows});entry={**item,'delivered_bytes':sum(r['bytes'] for r in files),'status':'sealed_awaiting_CCI_SHA'};state['batches'].append(entry);atomic(RUN/'PRODUCTION_STATUS.json',state)
   evidence.mkdir(parents=True,exist_ok=True)
   for file in ['summary.json','DELIVERY_MANIFEST.json','parents.csv','windows.csv','rejections.json','TRANSFER_SHA256.json']:shutil.copyfile(release/file,evidence/file)
   progress(f"{name}原门禁封存{len(seedrows)}父场景、{summary['formal_windows']}窗口、{summary['formal_logical_questions']}题；等待CCI SHA。")
  receipt=RUN/'transfer_receipts'/(name+'.json');deadline=time.time()+7200
  while not receipt.is_file():
   safe()
   if time.time()>deadline:raise RuntimeError('CCI transfer timeout; own copies retained')
   time.sleep(5)
  proof=json.loads(receipt.read_text());files=json.loads((evidence/'TRANSFER_SHA256.json').read_text())['files']
  if proof['status']!='verified' or proof['batch_id']!=name or proof['file_count']!=len(files) or proof['bytes']!=sum(x['bytes'] for x in files):raise ValueError('CCI verification differs')
  atomic(evidence/'CCI_verified.json',proof)
  if (release/'train.jsonl').is_file():
   old_ids=set()
   index=RUN/'additional_train_index_v2.jsonl'
   if index.is_file():old_ids={json.loads(l)['question_id'] for l in index.read_text().splitlines()}
   with index.open('a') as out:
    for line in (release/'train.jsonl').read_text().splitlines():
     r=json.loads(line)
     if r['question_id'] in old_ids:continue
     for k in ('video','audio_foa','audio_mono','private_evidence'):r[k]=str(Path(proof['release_directory'])/r[k])
     r['delivery_batch_id']=name;out.write(json.dumps(r,ensure_ascii=False)+'\n')
  for row in seedrows:retain_witness_then_remove(phase/'production/parents'/row['scene_id'],name)
  if release.is_dir():shutil.rmtree(release)
  if remove_whole_phase:
   # Recovered own CCI copies were SHA-checked above and are still retained
   # immutably at CCI; keep the original archive SHA and derivation receipts.
   target=phase/'immutable_CCI_sources'
   if target.is_dir():shutil.rmtree(safe_own_path(target))
  entry.update(status='CCI_SHA_verified_local_copies_removed',CCI_release=proof['release_directory'],transfer_wall_seconds=proof['wall_seconds'])
  state['transferred_parents']=sum(b['parents'] for b in state['batches'] if b['status']=='CCI_SHA_verified_local_copies_removed');state['wall_seconds']=time.time()-started;atomic(RUN/'PRODUCTION_STATUS.json',state)
  progress(name+'全文件CCI SHA一致，清理仅本任务通过场景中间件。')
 def finish_phase(name):
  nonlocal peak,state
  phase=RUN/'production_batches'/name;batch=[byid[x] for x in state['dispatches'][name]['parent_ids']];g=json.loads((phase/'guard/result.json').read_text())
  if g.get('stop_reason') or g.get('returncode') not in (0,2):raise RuntimeError('native guard or integration failure '+name)
  # A crash after verified cleanup can leave the dispatch marker behind.
  # Its immutable summary and ledger, rather than deleted raw files, own the
  # already-completed outcome.
  delivered=next((e for e in state['batches'] if e['batch_id']==name+'_passed' and e['status']=='CCI_SHA_verified_local_copies_removed'),None)
  if delivered:
   rejected=sum(x.get('batch')==name for x in state['rejections'].values())
   assert delivered['parents']+rejected==len(batch)
   state['dispatches'][name].update(status='processed',passed_parents=delivered['parents'],rejected_parents=rejected,wall_seconds=g['elapsed_seconds'],peak_mib=g['peak_mib']);atomic(RUN/'PRODUCTION_STATUS.json',state);return
  passed=[]
  for row in batch:
   parent=phase/'production/parents'/row['scene_id'];marker=parent/'alpha_seal_completed.json'
   if marker.is_file():
    proof=json.loads(marker.read_text());assert proof['result']['status']=='formal_alpha_admitted' and proof['recipe']['input_recipe']['producer_qa_revision']==PRODUCER_QA_REVISION;passed.append(row)
   else:
    failures=sorted(parent.glob('failed_*.json'),key=lambda p:p.stat().st_mtime)
    if not failures:raise RuntimeError('no terminal original-gate receipt '+row['scene_id'])
    f=json.loads(failures[-1].read_text())
    if not scientific_rejection(f):raise RuntimeError('unclassified generation error; stop for review '+row['scene_id']+': '+f.get('reason',''))
    state['rejections'][row['scene_id']]={**f,'batch':name,'witness':str(RUN/'scene_witnesses'/name/row['scene_id'])};state['attempted_parent_ids']=sorted(set(state['attempted_parent_ids'])|{row['scene_id']});atomic(RUN/'PRODUCTION_STATUS.json',state);retain_witness_then_remove(parent,name)
  workers=state['dispatches'][name]['workers'];peak=max(peak,float(g['peak_mib'])/workers)
  if passed:
   seeds=phase/'passed_seeds.jsonl';seeds.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in passed));release=phase/'release'
   if not (release/'summary.json').is_file():export_alpha(seeds,[phase/'production'],release,expected_parents=len(passed))
   deliver(name+'_passed',release,phase,passed)
  state['dispatches'][name].update(status='processed',passed_parents=len(passed),rejected_parents=len(batch)-len(passed),wall_seconds=g['elapsed_seconds'],peak_mib=g['peak_mib']);state['active_workers']=0;atomic(RUN/'PRODUCTION_STATUS.json',state)
  progress(f'{name}完成{len(batch)}个固定请求，通过{len(passed)}、原门禁拒绝{len(batch)-len(passed)}；不补题、不换种子。')
 try:
  phase=Path(canary['release']).parent;fixrows=[byid[x] for x in canary['parent_ids']]
  deliver('batch_002_endpoint_fixed',Path(canary['release']),phase,fixrows,remove_whole_phase=True)
  for name,d in state['dispatches'].items():
   if d['status']=='processed':continue
   receipt=RUN/'production_batches'/name/'guard_process.json'
   if receipt.is_file():
    v=json.loads(receipt.read_text())
    try:
     p=psutil.Process(v['pid'])
     if p.create_time()==v['create_time'] and p.status()!=psutil.STATUS_ZOMBIE:raise RuntimeError('previous own guard still live; no duplicate dispatch')
    except psutil.NoSuchProcess:pass
   finish_phase(name)
  pending=[r for r in rows if r['scene_id'] not in set(state['attempted_parent_ids'])];pending.sort(key=lambda r:r['house']!='kujiale_0022')
  for i in range(0,len(pending),4):
   snap=safe();batch=pending[i:i+4];free=snap['gpus'][2]['free_mib'];workers=min(args.max_workers,len(batch),int(max(0,free-4096)//peak))
   if workers<1:raise RuntimeError('GPU2 lacks measured peak+4GiB; foreign jobs untouched')
   seq=3
   while (RUN/'production_batches'/f'batch_{seq:03d}').exists():seq+=1
   name=f'batch_{seq:03d}';phase=RUN/'production_batches'/name;phase.mkdir();seeds=phase/'seeds.jsonl';seeds.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in batch))
   cmd=[os.sys.executable,str(REPO/'tools/timeline/generate_v7_registered.py'),'--houses',str(RUN/'inputs/houses.json'),'--seeds',str(seeds),'--runtime-inputs',str(RUN/'inputs/runtime.json'),'--assets',str(RUN/'inputs/assets.json'),'--source-admission',str(RUN/'inputs/source_admission_v2.json'),'--human-admission',str(RUN/'inputs/human_admission.json'),'--output',str(phase/'production'),'--storage-budget-root',str(RUN),'--workers',str(workers),'--limit',str(len(batch)),'--render-backend','ue_spear']
   request={'budget_root':str(ROOT),'run_directory':str(phase/'guard'),'command':cmd,'cwd':str(REPO),'gpu_index':2,'cpu_affinity':list(range(80,96)),'previous_peak_mib':workers*peak,'minimum_free_mib':workers*peak+4096,'minimum_disk_free_bytes':100_000_000_000,'storage_limit_bytes':snap['logical_bytes']+max(0,30_000_000_000-snap['new_allocated_bytes']),'reserve_bytes':1_000_000_000,'stop_margin_bytes':500_000_000,'maximum_seconds':21600,'allowed_auxiliary_gpu_indices':[0],'maximum_auxiliary_gpu_mib':64.,'resource_snapshot_path':str(RUN/'resource_sampler/resource_snapshot.json'),'environment':{'CUDA_VISIBLE_DEVICES':'2','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}}
   atomic(phase/'guard_request.json',request);state['dispatches'][name]={'status':'started','parent_ids':[r['scene_id'] for r in batch],'workers':workers};state.update(active_batch=name,active_workers=workers);atomic(RUN/'PRODUCTION_STATUS.json',state)
   with (phase/'driver.log').open('x') as log:active=subprocess.Popen([os.sys.executable,str(REPO/'tools/rooms/v7_kujiale_ue_guard.py'),'--request',str(phase/'guard_request.json')],stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
   p=psutil.Process(active.pid);atomic(phase/'guard_process.json',{'pid':p.pid,'create_time':p.create_time(),'command':p.cmdline()});active.wait();active=None;finish_phase(name)
  queue['dispatch_finished']=True;queue['complete']=state['formal_parents']==132;atomic(RUN/'transfer_queue.json',queue);state.update(status='complete' if queue['complete'] else 'finished_with_original_rejections',wall_seconds=time.time()-started,active_workers=0)
  progress(f"固定132请求已处理：实际封存{state['formal_parents']}父/{state['formal_windows']}窗口/{state['formal_questions']}题，拒绝{len(state['rejections'])}；未完成完整132准入时不得标为完整训练交付。")
 except BaseException as e:
  if active and active.poll() is None:active.terminate();active.wait(timeout=60)
  state.update(status='stopped_not_complete',error=str(e),traceback=traceback.format_exc(),wall_seconds=time.time()-started,active_workers=0);progress('v2停止：'+str(e)+'；门槛不变、失败证据保留。');raise
 finally:
  state['generation_sessions'][-1].update(ended_epoch=time.time(),wall_seconds=time.time()-started);atomic(RUN/'PRODUCTION_STATUS.json',state)

if __name__=='__main__':main()

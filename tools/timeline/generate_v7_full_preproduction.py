"""Explicit TRAIN-only 1500-world preproduction, separate from the eight-world review.

The frozen profile, training whitelist, seed authority, resources and receipts
are checked in workers too. Original native/14-question gates are untouched.
Only a matching verified CCI receipt permits removal of this task's local shard.
"""
from __future__ import annotations
import argparse,collections,copy,fcntl,hashlib,json,os,resource,shutil,signal,socket,subprocess,sys,tarfile,time,traceback,uuid
from pathlib import Path

ROOT=Path('/data/jzy/tmp/claude_hardbank_20260923/v7_batch1_20261003/round17_full_preproduction1500_20261005')
AFS_ROOT='/mnt/afs/xinyuanqian/jzy/hardbank/generation/v7_full_preproduction1500_20261005'
SCHEMA='v7-full1500-train-preproduction-r17-v1'
EXCLUDED={'hm3d_val_00808_y9hTuugGdiq','hm3d_val_00877_4ok3usBNeis','hm3d_val_00880_Nfvxx8J5NCo','hm3d_val_00862_LTyLHJXK6rR'}
CPU_ENV={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','LP_NUM_THREADS':'1','LIBGL_ALWAYS_SOFTWARE':'1','MESA_LOADER_DRIVER_OVERRIDE':'llvmpipe'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
 return h.hexdigest()

def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
 with tmp.open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def jsonl(path):return [json.loads(l)for l in Path(path).read_text().splitlines()if l.strip()]

def load_contract(profile_path):
 p=json.loads(Path(profile_path).read_text());root=Path(p['output']).resolve()
 if p.get('schema')!=SCHEMA or root!=ROOT or p['AFS_destination']!=AFS_ROOT:raise ValueError('explicit owner full-preproduction scope mismatch')
 if not 1<=p['workers']<=24 or p['nice']!=10 or p['shard_size']!=50:raise ValueError('authorized CPU/nice/shard bounds mismatch')
 if len(p['cpu_affinity'])!=24 or len(set(p['cpu_affinity']))!=24 or any(not isinstance(c,int)or c<0 or c>=os.cpu_count()for c in p['cpu_affinity']):raise ValueError('exact authorized 24-core affinity required')
 if p['minimum_data_free_bytes']!=100_000_000_000 or p['maximum_full_root_bytes']>138_000_000_000 or p['maximum_review_root_bytes']!=10_000_000_000:raise ValueError('disk discipline differs from owner contract')
 if any(os.environ.get(k)!=v for k,v in CPU_ENV.items())or not os.environ.get('LD_PRELOAD'):raise ValueError('explicit single-thread CPU software runtime required')
 required_roles={'walk','sit','content27','content28','dog'}
 if set(p.get('qualified_native_episodes',{}))!=required_roles:raise ValueError('batch2 native qualification receipts incomplete')
 for role,path in p['qualified_native_episodes'].items():
  if path not in p['input_SHA256']:raise ValueError('qualification receipt not frozen')
  result=json.loads(Path(path).read_text())['result']
  if result.get('status')!='formal_alpha_admitted' or result.get('alpha_formal_windows',0)<1:raise ValueError('batch2 episode not actually sealed: '+role)
 for path,digest in p['input_SHA256'].items():
  if sha(path)!=digest:raise ValueError('frozen input changed: '+path)
 if len(p['source_commit'])!=40 or any(c not in '0123456789abcdef'for c in p['source_commit']):raise ValueError('missing frozen commit')
 rows=jsonl(p['seeds']);authority=jsonl(p['seed_authority']);s_rows=jsonl(p['S_reference_seeds'])
 if len(rows)!=1500 or rows!=authority:raise ValueError('first1500 authority changed; feature plans must be separate')
 if len({r['scene_id']for r in rows})!=1500 or len({r['scene_seed']for r in rows})!=1500:raise ValueError('duplicate world ID or seed')
 if {r['scene_id']for r in rows}&{r['scene_id']for r in s_rows}or {r['scene_seed']for r in rows}&{r['scene_seed']for r in s_rows}:raise ValueError('full worlds overlap S worlds')
 houses=json.loads(Path(p['houses']).read_text())['houses'];by_house={h['house']:h for h in houses}
 if len(by_house)!=48 or collections.Counter(h['family']for h in houses)!={'hm3d':39,'mp3d':9}:raise ValueError('strict training whitelist changed')
 excluded=EXCLUDED|{r['house']for r in s_rows if r['split']!='train'}
 if any(h['split']!='train'or h['house']in excluded or h['acoustic_status']!='pass'for h in houses):raise ValueError('unqualified or held-out physical house')
 for r in rows:
  if r['split']!='train'or r['house']not in by_house or r['house']in excluded or r['room_id']not in by_house[r['house']]['checked_room_ids']:raise ValueError('row escaped original strict train room gate')
 features=json.loads(Path(p['features']).read_text());valid={r['scene_id']for r in rows}
 if not set(features)<=valid:raise ValueError('feature plans reference other worlds')
 allowed={'motion_requests','additional_source_assets','semantic_scenario_id','semantic_sound_subset_ids','semantic_sound_ids','semantic_role_start_offsets_s','binding_camera_yaw_offset_deg'}
 for sid,f in features.items():
  if not set(f)<=allowed:raise ValueError('feature plan may not replace houses/counts/events/answers')
  for q in f.get('motion_requests',{}).values():
   if q.get('action')not in {'walk','sit'}:raise ValueError('unqualified object motion is not admitted into full preproduction')
 return p,rows,by_house,features

def input_row(row,features):
 row=copy.deepcopy(row);row['production_scenario'].update(copy.deepcopy(features.get(row['scene_id'],{})))
 return row

def storage_policy(p,parent=None):
 if parent is not None and (Path(parent).is_symlink() or not Path(parent).resolve().is_relative_to(ROOT/'shards')):raise ValueError('worker output escaped its own authorized shard root')
 from avengine.timeline.v7_storage_budget import ENV_KEY,check_write,allocated_bytes
 if allocated_bytes(p['review_root'])>p['maximum_review_root_bytes']:raise RuntimeError('own review root exceeds 10 GB; no new work')
 if parent is None:
  policy={'root':str(ROOT),'filesystem':'/data','minimum_free_bytes':100_000_000_000,'maximum_task_bytes':p['maximum_full_root_bytes'],'concurrent_write_margin_bytes':32_400_000_000,'policy':SCHEMA}
 else:
  # A bounded parent ensures <=67.5 GB per fifty worlds. Reserve the other
  # 23 bounded writers in shared df; the original review policy is unchanged.
  policy={'root':str(parent),'filesystem':'/data','minimum_free_bytes':132_400_000_000,'maximum_task_bytes':1_350_000_000,'concurrent_write_margin_bytes':64_000_000,'policy':SCHEMA+'-bounded-parent'}
 os.environ[ENV_KEY]=json.dumps(policy,sort_keys=True);return check_write(parent or ROOT,expected_bytes=0)

def boot_id():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def process_start(pid):
 try:return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
 except (FileNotFoundError,IndexError):return None

def own_worker_live(record,profile_path,sid):
 if record.get('hostname')!=socket.gethostname()or record.get('boot_id')!=boot_id()or process_start(record['pid'])!=record.get('process_start'):return False
 try:cmd=Path('/proc/'+str(record['pid'])+'/cmdline').read_bytes().split(b'\0')
 except FileNotFoundError:return False
 return str(Path(__file__).resolve()).encode() in cmd and str(Path(profile_path).resolve()).encode()in cmd and sid.encode()in cmd and b'--worker-scene-id'in cmd

def parent_worker(profile_path,sid,shard_index):
 p,rows,by_house,features=load_contract(profile_path)
 if shard_index not in range(30) or sid not in {r['scene_id']for r in rows[shard_index*50:(shard_index+1)*50]}:raise ValueError('worker world is outside this exact shard')
 row=next(r for r in rows if r['scene_id']==sid);row=input_row(row,features)
 shard=ROOT/'shards'/f'shard_{shard_index:03d}';work=shard/'work';parent=work/'parents'/sid
 os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)));parent.mkdir(parents=True,exist_ok=True);storage_policy(p,parent)
 start=time.perf_counter();usage=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN);cpu0=usage.ru_utime+usage.ru_stime+children.ru_utime+children.ru_stime
 try:
  from avengine.timeline.v7_registered_generation import generate_registered_parent
  inputs=json.loads(Path(p['runtime_inputs']).read_text());inputs['render_backend']='cpu_llvmpipe'
  if row['production_scenario'].get('motion_requests'):
   import fcl
   for name in ('v7_asset_roster','moving_energy_budget'):
    if not inputs.get(name) or not Path(inputs[name]).is_file():raise ValueError('motion runtime requires existing frozen '+name)
  assets=json.loads(Path(p['assets']).read_text());assets=assets['assets']if isinstance(assets,dict)else assets
  result=generate_registered_parent(row,by_house[row['house']],inputs,assets,json.loads(Path(p['source_admission']).read_text()),json.loads(Path(p['human_admission']).read_text()),work)
  if result.get('resumed_sealed_delivery')and not list(work.glob('shard_'+sid+'*.json')):raise ValueError('seal survived restart but original successful timing record is missing; retain without inventing CPU time')
 except Exception as error:
  u=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
  result={'status':'failed_not_formal','scene_id':sid,'house':row['house'],'reason':str(error),'error_type':type(error).__name__,'traceback':traceback.format_exc(),'alpha_formal_windows':0,'alpha_formal_logical_questions':0,'wall_seconds':time.perf_counter()-start,'cpu_seconds_including_children':u.ru_utime+u.ru_stime+c.ru_utime+c.ru_stime-cpu0}
  atomic(parent/f'failed_{time.time_ns()}.json',result)
 if not result.get('resumed_sealed_delivery'):atomic(work/('shard_'+sid+'_'+str(time.time_ns())+'.json'),{'parents':[result]})
 atomic(shard/'results'/(sid+'.json'),result);print(json.dumps({k:result.get(k)for k in ['scene_id','status','reason','alpha_formal_windows','alpha_formal_logical_questions','wall_seconds','cpu_seconds_including_children']}),flush=True)
 return 0 if result['status']=='formal_alpha_admitted'else 2

def receipt_matches(receipt,ready):
 return (receipt.get('status')=='verified'and receipt.get('archive_SHA256')==ready['archive_SHA256']and receipt.get('archive_bytes')==ready['archive_bytes']and receipt.get('source_commit')==ready['source_commit']and receipt.get('shard_id')==ready['shard_id']and receipt.get('AFS_destination')==ready['AFS_destination']and receipt.get('actual_media_readback',{}).get('status')=='pass'and receipt.get('every_file_SHA256_verified')is True and all(receipt.get(key)==ready.get(key) for key in ('requested_parents','qualified_parents','windows','questions')))

def package_shard(p,rows,features,shard,index):
 release=shard/'release';results={r['scene_id']:json.loads((shard/'results'/(r['scene_id']+'.json')).read_text())for r in rows}
 selected=[input_row(r,features)for r in rows if results[r['scene_id']]['status']=='formal_alpha_admitted']
 if not (shard/'TRANSFER_READY.json').is_file():
  storage_policy(p)
  if selected and not release.exists():
   from avengine.timeline.v7_alpha_delivery import export_alpha
   from avengine.timeline.v7_s_batch_checks import check_critical_delivery,open_delivered_media
   manifest=shard/'qualified_seeds.jsonl';manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in selected));stage=shard/('export_attempt_'+str(time.time_ns()))
   # Interrupted exports remain untouched. A fresh attempt reads the same
   # complete native seals and must pass the unchanged export gates.
   summary=export_alpha(manifest,[shard/'work'],stage,expected_parents=len(selected));coverage=check_critical_delivery(stage,require_full=False);media=open_delivered_media(stage)
   if coverage['status']!='pass'or media['status']!='pass':raise ValueError('unchanged media/question gates failed')
   atomic(stage/'DELIVERY_MANIFEST.json',{'status':'sealed','batch_kind':'full1500-preproduction-qualified-shard','generator_commit':p['source_commit'],'requested_parent_scenes':len(rows),'independent_parent_scenes':len(selected),'formal_windows':summary['formal_windows'],'formal_logical_questions':summary['formal_logical_questions'],'coverage':coverage,'media_readback':media,'test_house_access':0,'GPU_started':False,'no_gate_changed':True,'full_paper_dataset_ready':False})
   os.rename(stage,release)
  witnesses=shard/'native_witnesses';witnesses.mkdir(exist_ok=True)
  # Preserve full measured native JSON/foot/visibility/candidate receipts,
  # while generated pose meshes and temporary material caches are rebuildable.
  work=shard/'work'
  for file in work.glob('parents/**/*.json'):
   dest=witnesses/file.relative_to(work);dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():os.link(file,dest)
  atomic(shard/'SOURCE_PROVENANCE.json',{'schema':SCHEMA,'source_commit':p['source_commit'],'profile_SHA256':sha(p['profile_path']),'requested_rows':rows,'applied_feature_requests':{r['scene_id']:features.get(r['scene_id'],{})for r in rows},'source_inputs_SHA256':p['input_SHA256'],'house_access_scope':'TRAIN original strict48 only','S_world_overlap':0,'test_house_access':0,'no_gate_changed':True,'parents':list(results.values())})
  payload=[shard/'SOURCE_PROVENANCE.json',witnesses]
  if release.exists():payload.append(release)
  inventory={}
  for item in payload:
   files=[item]if item.is_file()else sorted(x for x in item.rglob('*')if x.is_file())
   for file in files:
    if file.is_symlink():raise ValueError('symlink in immutable transfer payload')
    inventory[str(file.relative_to(shard))]=sha(file)
  atomic(shard/'FILE_SHA256.json',inventory);payload.append(shard/'FILE_SHA256.json')
  archive=shard/f'shard_{index:03d}.tar.gz'
  if archive.exists():
   retained=archive.with_name(archive.name+'.interrupted_'+str(time.time_ns()));os.rename(archive,retained)
   atomic(shard/'archive_restart_audit.json',{'retained_partial_archive':str(retained),'partial_SHA256':sha(retained),'source_commit':p['source_commit']})
  from avengine.timeline.v7_storage_budget import check_write
  upper_bound=sum(item.stat().st_size if item.is_file()else sum(f.stat().st_size for f in item.rglob('*')if f.is_file())for item in payload)+50_000_000
  check_write(archive,expected_bytes=upper_bound)
  with tarfile.open(archive,'x:gz',compresslevel=1,dereference=True)as tf:
   for item in payload:tf.add(item,arcname=str(item.relative_to(shard)),recursive=True)
  ready={'schema':SCHEMA,'status':'ready_for_CCI_pull','shard_id':index,'source_commit':p['source_commit'],'archive':str(archive),'archive_SHA256':sha(archive),'archive_bytes':archive.stat().st_size,'AFS_destination':AFS_ROOT+f'/shards/shard_{index:03d}','requested_parents':len(rows),'qualified_parents':len(selected),'failed_parents':len(rows)-len(selected),'windows':sum(results[r['scene_id']]['alpha_formal_windows']for r in rows),'questions':sum(results[r['scene_id']]['alpha_formal_logical_questions']for r in rows),'test_house_access':0,'GPU_started':False,'created_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
  atomic(shard/'TRANSFER_READY.json',ready);atomic(ROOT/'TRANSFER_PENDING.json',ready)
 return json.loads((shard/'TRANSFER_READY.json').read_text())

def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--profile',type=Path,required=True);a.add_argument('--worker-scene-id');a.add_argument('--shard-index',type=int);a.add_argument('--validate-only',action='store_true');args=a.parse_args()
 p,rows,houses,features=load_contract(args.profile);p['profile_path']=str(args.profile.resolve())
 if args.worker_scene_id:return parent_worker(args.profile.resolve(),args.worker_scene_id,args.shard_index)
 if args.validate_only:
  print(json.dumps({'status':'contract_pass','parents':len(rows),'train_houses':len(houses),'feature_parents':len(features),'CPU_max':p['workers'],'S_overlap':0,'test_house_access':0,'native_calls':0}));return 0
 ROOT.mkdir(parents=True,exist_ok=True);os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)));os.sched_setaffinity(0,set(p['cpu_affinity']))
 lock=(ROOT/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 atomic(ROOT/'coordinator_owner.json',{'pid':os.getpid(),'hostname':socket.gethostname(),'boot_id':boot_id(),'profile':str(args.profile.resolve()),'source_commit':p['source_commit'],'started_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
 stopping=False
 def stop(sig,frame):
  nonlocal stopping;stopping=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 start=time.monotonic();children={};attempted=qualified=windows=questions=0;last_cpu=0.;limit=p['workers'];sample_previous=None
 try:
  for index in range(30):
   shard=ROOT/'shards'/f'shard_{index:03d}'
   if shard.is_symlink() or not shard.resolve().is_relative_to(ROOT/'shards'):raise ValueError('shard parent escaped owner scope')
   shard.mkdir(parents=True,exist_ok=True);chunk=rows[index*50:(index+1)*50]
   transferred=ROOT/'transferred'/f'shard_{index:03d}.json'
   if transferred.exists():
    done=json.loads(transferred.read_text());attempted+=done['requested_parents'];qualified+=done['qualified_parents'];windows+=done['windows'];questions+=done['questions'];continue
   pending=[]
   for row in chunk:
    sid=row['scene_id'];result=shard/'results'/(sid+'.json');active=shard/'active'/(sid+'.json')
    if result.exists():continue
    if active.exists()and own_worker_live(json.loads(active.read_text()),args.profile,sid):raise RuntimeError('previous own native worker still alive; retain it and resume after it naturally exits: '+sid)
    pending.append(row)
   while pending or children:
    if (ROOT/'PAUSE_NEW_WORK').exists():stopping=True
    load=float(Path('/proc/loadavg').read_text().split()[0])
    if load>90:limit=max(1,min(limit-1,p['workers']//2))
    elif load<80:limit=min(p['workers'],limit+1)
    storage=storage_policy(p)
    while pending and not stopping and len(children)<limit:
     row=pending.pop(0);sid=row['scene_id'];log_path=shard/'logs'/(sid+'_'+str(time.time_ns())+'.log');log_path.parent.mkdir(exist_ok=True);log=log_path.open('x')
     busy_cores={item[2] for item in children.values()}
     affinity=next(core for core in p['cpu_affinity'] if core not in busy_cores)
     cmd=['taskset','-c',str(affinity),sys.executable,str(Path(__file__).resolve()),'--profile',str(args.profile.resolve()),'--worker-scene-id',sid,'--shard-index',str(index)]
     proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),start_new_session=True,cwd=p['code']);children[sid]=(proc,log,affinity)
     atomic(shard/'active'/(sid+'.json'),{'pid':proc.pid,'hostname':socket.gethostname(),'boot_id':boot_id(),'process_start':process_start(proc.pid),'command':cmd,'log':str(log_path),'CPU_core':affinity})
    for sid,(proc,log,affinity)in list(children.items()):
     if proc.poll()is None:continue
     log.close();del children[sid]
     result=shard/'results'/(sid+'.json')
     if not result.exists():atomic(result,{'status':'failed_native_process_exit','scene_id':sid,'reason':'native process ended before durable result','returncode':proc.returncode,'alpha_formal_windows':0,'alpha_formal_logical_questions':0})
    recent=[json.loads(f.read_text())for f in (shard/'results').glob('*.json')]if (shard/'results').exists()else []
    now=time.monotonic();actual=qualified+sum(r['status']=='formal_alpha_admitted'for r in recent)
    atomic(ROOT/'PROGRESS.json',{'status':'draining'if stopping else 'running','source_commit':p['source_commit'],'parents_requested':1500,'attempted_parents':attempted+len(recent),'formal_parents':actual,'windows':windows+sum(r['alpha_formal_windows']for r in recent),'questions':questions+sum(r['alpha_formal_logical_questions']for r in recent),'workers_limit':limit,'active_parents':list(children),'current_shard':index,'host_load_1m':load,'storage':storage,'elapsed_seconds':now-start,'formal_parents_per_hour':actual/max(now-start,1)*3600,'checked_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'GPU_started':False,'test_house_access':0})
    if now-last_cpu>=60:
     with(ROOT/'CPU.jsonl').open('a')as f:f.write(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'proc_stat':Path('/proc/stat').read_text(),'loadavg':Path('/proc/loadavg').read_text().strip(),'workers_limit':limit,'active_parents':len(children),'affinity':p['cpu_affinity']})+'\n')
     last_cpu=now
    if stopping and not children:break
    time.sleep(10)
   if stopping:atomic(ROOT/'PAUSED.json',{'status':'paused_after_own_workers_drained','current_shard':index,'time_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});return 0
   ready=package_shard(p,chunk,features,shard,index);receipt=ROOT/'CCI_RECEIPTS'/f'shard_{index:03d}.json'
   atomic(ROOT/'PROGRESS.json',{'status':'waiting_for_CCI_SHA_and_media','current_shard':index,'transfer_ready':ready,'formal_parents':qualified+ready['qualified_parents'],'windows':windows+ready['windows'],'questions':questions+ready['questions'],'active_parents':[],'workers_limit':limit,'test_house_access':0,'GPU_started':False})
   while not receipt.exists():
    if stopping or(ROOT/'PAUSE_NEW_WORK').exists():return 0
    storage_policy(p);time.sleep(30)
   verification=json.loads(receipt.read_text())
   if not receipt_matches(verification,ready):raise ValueError('CCI receipt failed exact archive/source/shard/media match; no local deletion')
   listing=[{'path':str(f.relative_to(shard)),'bytes':f.stat().st_size}for f in shard.rglob('*')if f.is_file()]
   atomic(ROOT/'deletion_lists'/f'shard_{index:03d}.json',{'own_root':str(shard),'paths':listing,'CCI_receipt':verification,'authorized_by':'owner R13 SHA then delete own48 copies only'})
   if shard.is_symlink()or not shard.resolve().is_relative_to(ROOT/'shards'):raise ValueError('unsafe local deletion path')
   shutil.rmtree(shard);atomic(transferred,{**ready,'CCI_receipt':verification,'local_copy_removed_after_SHA':True})
   attempted+=ready['requested_parents'];qualified+=ready['qualified_parents'];windows+=ready['windows'];questions+=ready['questions']
  atomic(ROOT/'PROGRESS.json',{'status':'all1500_attempts_transferred','attempted_parents':attempted,'formal_parents':qualified,'windows':windows,'questions':questions,'source_commit':p['source_commit'],'elapsed_seconds':time.monotonic()-start,'test_house_access':0,'GPU_started':False,'full_paper_dataset_ready':False});return 0
 finally:
  for sid,(proc,log,affinity)in children.items():
   record={'pid':proc.pid,'hostname':socket.gethostname(),'boot_id':boot_id(),'process_start':process_start(proc.pid)}
   if proc.poll()is None and own_worker_live(record,args.profile,sid):
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
     if proc.poll()is None and own_worker_live(record,args.profile,sid):os.killpg(proc.pid,signal.SIGKILL)
     proc.wait(timeout=5)
   log.close()
  lock.close()

if __name__=='__main__':raise SystemExit(main())

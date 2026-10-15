#!/usr/bin/env python3
"""Unattended CPU scene construction from global registries and legal rooms.

There is no seed-recipe, source-slot, pinned-position or handwritten plan input.
The existing native producer, export and media reader remain the authorities.
"""
from __future__ import annotations
import argparse,datetime,fcntl,hashlib,json,os,re,signal,subprocess,sys,time,traceback
from pathlib import Path
W=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(W/'src'),str(W)]
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','LP_NUM_THREADS'):os.environ[name]='1'
os.environ.update(CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',EGL_PLATFORM='surfaceless')


def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def commit():
    from avengine.timeline.v7_frozen_code import source_commit
    return source_commit(W)

def clean_source():
    from avengine.timeline.v7_frozen_code import verify_source
    return verify_source(W)


def emit_progress(root,cfg,message):
    line=utc()+' '+root.name+' '+message+'\n'
    for path in {root/'PROGRESS.md',Path(cfg['authorized_output_root'])/'PROGRESS.md'}:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a')as stream:stream.write(line)


def load_global(path):
    cfg=read(path)
    allowed={'schema','authorized_output_root','contract','houses','assets','runtime','human_phase2',
        'human_registered','feedback_roots','resources','extra_python_paths','cpu_egl_preload',
        'cleanup_failed_after_unix'}
    if cfg.get('schema')!='v7-global-family-runtime-v1'or set(cfg)-allowed:
        raise ValueError('only global registries/resource policy are accepted; no scene configuration input')
    return cfg


def global_fingerprints(cfg):
    paths={cfg[key]for key in ('contract','houses','assets','runtime','human_phase2','human_registered','cpu_egl_preload')if cfg.get(key)}
    if cfg.get('contract'):
        paths.add(read(cfg['contract'])['inputs']['source_admission']['path'])
    if cfg.get('runtime'):
        runtime=read(cfg['runtime'])
        paths.update(runtime[key]for key in ('semantic_manifest','v7_asset_roster','moving_energy_budget')if runtime.get(key))
    return {str(Path(path).resolve()):hashlib.sha256(Path(path).read_bytes()).hexdigest()for path in sorted(paths)}


def data(cfg):
    contract=read(cfg['contract']);assets=read(cfg['assets']);assets=assets.get('assets',assets)if isinstance(assets,dict)else assets
    runtime=read(cfg['runtime']);source=read(contract['inputs']['source_admission']['path'])
    semantic=read(runtime['semantic_manifest']);houses={h['house']:h for h in read(cfg['houses'])['houses']}
    return contract,assets,runtime,source,semantic,houses


def child_env(cfg,root):
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',LP_NUM_THREADS='1',LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',
        EGL_PLATFORM='surfaceless',MESA_SHADER_CACHE_DISABLE='true',PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH']=':'.join([str(W/'src'),str(W),*cfg.get('extra_python_paths',[])])
    if cfg.get('cpu_egl_preload'):env['LD_PRELOAD']=cfg['cpu_egl_preload']
    env['__EGL_VENDOR_LIBRARY_FILENAMES']='/usr/share/glvnd/egl_vendor.d/50_mesa.json'
    for name,relative in [('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('MPLCONFIGDIR','mpl'),('PYTHONPYCACHEPREFIX','pycache')]:
        path=root/'cache'/relative;path.mkdir(parents=True,exist_ok=True);env[name]=str(path)
    return env


def worker(a,cfg,root):
    from avengine.timeline.v7_family_constructors import construct_family_scene
    from avengine.timeline.v7_unattended_resources import ResourcePolicy,configure_worker,runtime_resource_view,launch_disk_policy
    from avengine.timeline.v7_native_resource_slots import acquire_native_resource
    policy=ResourcePolicy(**cfg.get('resources',{}));cpus=json.loads(a.cpus)
    configure_worker(Path(a.run_root),policy,cpus)
    lease=acquire_native_resource('parent')
    contract,assets,runtime,source,semantic,houses=data(cfg)
    room=contract['rooms'][a.room_index];house=houses[room['house']]
    snapshot=root/'inputs/construction_feedback.json'
    if not snapshot.exists():write(snapshot,read(a.feedback))
    state=read(snapshot);sid=a.scene_id+'__candidate_'+str(a.candidate).zfill(3)
    started=time.time();result=None
    try:
        frozen_inputs=read(Path(a.run_root)/'AUTOMATION_RUN.json')['input_registry_sha256']
        if global_fingerprints(cfg)!=frozen_inputs:raise ValueError('global input registry changed; preserve this attempt and use a new frozen run')
        if commit()!=a.source_commit or not clean_source():
            raise ValueError('native rehearsal needs the exact clean frozen worktree; no mid-run code changes')
        plan=construct_family_scene(room,house,state,assets,source['records'],semantic,contract,runtime,
            scene_id=sid,seed=a.seed,family=a.family,attempt=a.candidate)
        row=plan['row'];row['independent_parent_scene_id']=a.scene_id
        runtime=runtime_resource_view(runtime,policy,cpus,cache_root=Path(a.run_root));runtime['launch_disk_policy']=launch_disk_policy(policy)
        write(root/'inputs/runtime_cpu.json',runtime);write(root/'inputs/demand_state.json',state)
        # Machine compilation cache only: no pre-existing handwritten row is consumed.
        cache=root/'inputs/planned/000.json'
        if cache.exists()and read(cache)!=row:raise ValueError('machine construction cache changed; preserve the old attempt')
        write(cache,row)
        manifest={'output_root':str(root),'worktree':str(W),'code_commit':a.source_commit,
            'dataset_role':'acceptance_only','training_dataset_merge':False,'CCI_transfer':False,
            'storage_budget_root':str(Path(a.run_root).resolve()),'storage_admission_prechecked':True,
            'cpu_affinity_ids':[cpus[0]],'fresh_rir_workers':runtime['fresh_rir_workers'],'launch_disk_policy':launch_disk_policy(policy),'concurrent_write_margin_bytes':0,'temporary_minimum_free_bytes':policy.temporary_minimum_free_bytes,
            'temporary_minimum_free_until_unix':policy.temporary_minimum_free_until_unix,
            'contract':cfg['contract'],'runtime_cpu':str(root/'inputs/runtime_cpu.json'),
            'houses_cpu':cfg['houses'],'assets_cpu':cfg['assets'],'human_cpu':cfg['human_phase2'],
            'human_registered_cpu':cfg['human_registered'],
            'parents':[{'slot':0,'device':'cpu','profile':plan['profile'],'family':room['family'],'row':row}]}
        write(root/'inputs/generated_manifest.json',manifest)
        from tools.timeline.run_v7_acceptance import run_worker
        code=run_worker(manifest,0);native=read(root/'evidence/jobs/000.json')
        result={**native,'native_receipt':str(root/'evidence/jobs/000.json'),
            'requested_scene_id':a.scene_id,'room_id':room['room_id'],'target_family':a.family,
            'export_rows_path':str(root/'export/parent_rows/000.jsonl'),'automatic_attempt_root':str(root)}
        counts=native.get('export',{}).get('question_family_counts',{})
        required=4 if a.family=='V7-CONTENT-BIND'else 1
        realized=counts.get(a.family,0)>=required
        if code==0 and a.family=='QA-17':
            rows=[json.loads(line)for line in Path(result['export_rows_path']).read_text().splitlines()]
            realized=realized and any(r['qa_id']=='QA-17'and str(r['canonical_answer']).lower()=='yes'for r in rows)
        if code==0 and not realized:
            result.update(status='construction_unfulfilled',reason='constructed family not realized: '+a.family)
        result['machine_compiled_cache']=str(cache)
        if result.get('status')=='passed':
            parent=Path(native['producer_result']['delivery_directory']).parent
            actual=read(parent/'input_recipe.json').get('program',{}).get('records',[])
            result['actual_voice_ids']=sorted({r['voice_reference']['speaker_id']for r in actual if r.get('source_type')=='author_tts'and r.get('voice_reference')})
        if global_fingerprints(cfg)!=frozen_inputs or commit()!=a.source_commit or not clean_source():
            result.update(status='source_changed_not_a_frozen_rehearsal',reason='worktree or global registry changed during the unattended native attempt')
    except Exception as error:
        result={'status':'failed','scene_id':sid,'requested_scene_id':a.scene_id,'house':room['house'],
            'room_id':room['room_id'],'target_family':a.family,'reason':str(error),
            'error_type':type(error).__name__,'traceback':traceback.format_exc(),'automatic_attempt_root':str(root)}
    finally:
        if lease:lease.close()
    result.update(automatic_started_unix=started,automatic_finished_utc=utc(),source_commit=a.source_commit,
        pid=os.getpid(),handwritten_scene_configuration_inputs_read=[],global_configuration=str(Path(a.config).resolve()))
    log_attempt_failure(result)
    write(root/'automatic_result.json',result)
    return 0 if result['status']=='passed'else 2


def cleanup_new_failed(root,result,policy_after,log):
    """Only newly created explicitly failed native attempts, after durable receipts."""
    if result.get('status')not in {'failed','construction_unfulfilled'}or not policy_after:return
    started=result['automatic_started_unix']
    if started<policy_after:return
    native=root/'native/parents'
    # Qualified native seals remain useful checkpoints even if a family was absent.
    if list(native.glob('*/integration_seal_completed.json'))or list(native.glob('*/alpha_seal_completed.json')):return
    with log.open('rb')as stream:os.fsync(stream.fileno())
    receipt=root/'automatic_result.json'
    with receipt.open('rb')as stream:os.fsync(stream.fileno())
    files=[]
    for path in native.glob('**/*'):
        if not path.is_file()or path.is_symlink()or path.suffix not in {'.npy','.wav','.mp4','.glb','.bin'}:continue
        stat=path.stat()
        if stat.st_size<1_000_000 or stat.st_mtime<max(started,policy_after):continue
        h=hashlib.sha256()
        with path.open('rb')as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        files.append({'path':str(path),'bytes':stat.st_size,'sha256':h.hexdigest()})
    audit=root/'failed_large_copy_cleanup.json'
    write(audit,{'status':'failed_receipt_and_log_saved_before_removal','receipt':str(receipt),'log':str(log),
        'authorized_new_attempts_after_unix':policy_after,'files':files})
    with audit.open('rb')as stream:os.fsync(stream.fileno())
    for item in files:Path(item['path']).unlink()
    write(audit,{**read(audit),'status':'removed_only_new_rebuildable_large_failed_copies','removed_bytes':sum(r['bytes']for r in files)})



def failure_signature(result):
    """Compare causes across candidates and parents, ignoring attempt paths."""
    reason=str(result.get('reason')or result.get('status','unknown failure')).splitlines()[0]
    if reason.startswith('write escaped the declared own task output root'):
        reason='write escaped the declared own task output root'
    else:
        reason=re.sub(r'/[^\s;,)]+','<path>',reason)
        reason=re.sub(r'auto_[A-Za-z0-9_]+','<scene>',reason)
        reason=re.sub(r'\bv7_[A-Za-z0-9_]+','<registered_asset>',reason)
    return str(result.get('error_type',''))+':'+reason


def failure_stop(result,history):
    """Stop dispatch for a code error or a second occurrence of one cause."""
    if result.get('status')=='passed':return None
    key=failure_signature(result);history[key]=history.get(key,0)+1
    fatal=result.get('error_type')in {'KeyError','ImportError','ModuleNotFoundError',
        'FileExistsError','SyntaxError','StorageBudgetExceeded'}
    fatal=fatal or result.get('status')=='source_changed_not_a_frozen_rehearsal'
    if fatal or history[key]>=2:
        return {'status':'stopped_before_next_candidate','time_utc':utc(),
            'cause_signature':key,'same_cause_occurrences':history[key],
            'reason':'systemic code/input failure'if fatal else'same failure cause occurred twice',
            'failed_attempt':result.get('automatic_attempt_root'),
            'action':'retain logs and receipts; fix the cause before a new batch; never kill running work'}
    return None



def log_attempt_failure(result):
    """Keep the exact path/root, error and source traceback in worker.log."""
    if result.get('status')=='passed':return
    print(json.dumps({key:result.get(key)for key in ('status','error_type','reason','native_receipt')},
        ensure_ascii=False),file=sys.stderr,flush=True)
    if result.get('traceback'):print(result['traceback'],file=sys.stderr,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parents',type=int,default=1);p.add_argument('--families',nargs='+');p.add_argument('--room-id');p.add_argument('--unseen-houses',action='store_true')
    p.add_argument('--plan-only',action='store_true');p.add_argument('--exercise-all-families',action='store_true');p.add_argument('--maximum-candidates',type=int,default=16)
    p.add_argument('--worker',action='store_true');p.add_argument('--room-index',type=int);p.add_argument('--candidate',type=int)
    p.add_argument('--family');p.add_argument('--scene-id');p.add_argument('--seed',type=int);p.add_argument('--cpus');p.add_argument('--source-commit');p.add_argument('--feedback');p.add_argument('--run-root')
    a=p.parse_args();cfg=load_global(a.config);sys.path.extend(cfg.get('extra_python_paths',[]));root=a.output.resolve();allowed=Path(cfg['authorized_output_root']).resolve()
    if root==allowed or not root.is_relative_to(allowed):raise ValueError('output must be a fresh descendant of the authorized task directory')
    os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
    if a.worker:raise SystemExit(worker(a,cfg,root))
    from avengine.timeline.v7_family_constructors import FAMILIES,REVISION,next_family,stable_seed,construct_family_scene
    from avengine.timeline.v7_automatic_feedback import from_passed_exports
    from avengine.timeline.v7_unattended_resources import ResourcePolicy,admission,idle_cpu_order
    if not 1<=a.parents or not 1<=a.maximum_candidates<=64:raise ValueError('positive parent count and bounded construction attempts required')
    scope=tuple(a.families or FAMILIES)
    if set(scope)-set(FAMILIES):raise ValueError('requested family belongs to another helper constructor')
    contract,assets,runtime,source,semantic,houses=data(cfg);policy=ResourcePolicy(**cfg.get('resources',{}))
    state=from_passed_exports([*cfg.get('feedback_roots',[]),root])
    baseline_path=root/'FEEDBACK_BASELINE.json'
    exposure_state=read(baseline_path)if baseline_path.exists()else state
    rooms=[(i,r)for i,r in enumerate(contract['rooms'])if r['family']in {'hm3d','mp3d'}and r['house']in houses and houses[r['house']]['split']=='train'
        and 'standing_human'in r['categories']and(not a.room_id or r['room_id']==a.room_id)
        and(not a.unseen_houses or r['house']not in exposure_state.get('observed_houses',[]))]
    rooms.sort(key=lambda pair:(pair[1]['house'],pair[1]['room_id']))
    if not rooms:raise ValueError('no original legal train room matches the requested untouched-room scope')
    root.mkdir(parents=True,exist_ok=True)
    coordinator_lock=(root/'coordinator.lock').open('a')
    fcntl.flock(coordinator_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    metadata=root/'AUTOMATION_RUN.json'
    identity={'schema':'v7-automatic-run-v1','plan_only':a.plan_only,'source_commit':commit(),'constructor_revision':REVISION,'global_config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),'input_registry_sha256':global_fingerprints(cfg),
        'family_scope':list(scope),'requested_parents':a.parents,'room_filter':a.room_id,'unseen_houses':a.unseen_houses,
        'exercise_all_families':a.exercise_all_families,'maximum_candidates':a.maximum_candidates,
        'initial_feedback_sha256':hashlib.sha256(json.dumps(exposure_state,sort_keys=True).encode()).hexdigest(),
        'handwritten_scene_configuration_inputs_read':[],'dataset_role':'acceptance_only','GPU':False,'CCI':False,'training_merge':False}
    if metadata.exists()and read(metadata)!=identity:raise ValueError('frozen global configuration or code changed; use a fresh output directory')
    write(metadata,identity)
    if not baseline_path.exists():write(baseline_path,exposure_state)
    write(root/'feedback.json',state)
    os.environ.update(child_env(cfg,root))
    outcomes=[];stop_requested=False;failure_history={};stop_reason=None
    def request_stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
        write(root/'STOP_REQUESTED.json',{'time_utc':utc(),'action':'finish the running parent; do not dispatch another'})
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    for index in range(a.parents):
        if stop_requested:break
        ri,room=rooms[index%len(rooms)];family=scope[index%len(scope)]if (a.plan_only or a.exercise_all_families)else next_family(state,scope)
        scene='auto_'+re.sub('[^A-Za-z0-9_]','_',root.name)+'_'+str(index).zfill(4);seed=stable_seed(REVISION,scene,room['key'])
        if a.plan_only:
            plan=construct_family_scene(room,houses[room['house']],state,assets,source['records'],semantic,contract,runtime,scene_id=scene,seed=seed,family=family)
            if plan['profile']=='phase2':
                from avengine.timeline.v7_family_clock import build_constructed_scene
                _,preview,program,_=build_constructed_scene(plan['row'],houses[room['house']],source['records'],assets,semantic)
            else:
                from avengine.timeline.v7_registered_sampling import choose_assets,build_registered_scene
                from avengine.timeline.v7_visible_sources import production_audio_record_views
                selected=choose_assets(assets,plan['row'])
                _,preview,program=build_registered_scene(plan['row'],houses[room['house']],production_audio_record_views(source['records']),selected,semantic)
            plan['program_preview']={'native_admitted':False,'events':preview['events'],
                'source_record_ids':[r['sound_asset_id']for r in program['records']],
                'source_types':[r.get('source_type')for r in program['records']],
                'moving_source_kinds':{key:value.get('mechanism')for key,value in preview['sources'].items()}}
            write(root/'proposals'/f'{index:04d}.json',plan);outcomes.append({'scene_id':scene,'family':family,'room_id':room['room_id'],'status':'proposal_only_native_not_run'});continue
        record=root/'evidence/jobs'/f'{index:04d}.json'
        if record.exists()and read(record).get('status')=='passed':outcomes.append(read(record));continue
        for candidate in range(a.maximum_candidates):
            attempt=root/'attempts'/(scene+'__'+str(candidate).zfill(3))
            if (attempt/'automatic_result.json').exists():
                result=read(attempt/'automatic_result.json');write(record,result)
                if result['status']=='passed':break
                stop_reason=failure_stop(result,failure_history)
                if stop_reason:
                    stop_requested=True;write(root/'AUTOMATION_STOP.json',stop_reason)
                    emit_progress(root,cfg,'entire batch dispatch stopped '+json.dumps(stop_reason));break
                continue
            while True:
                if stop_requested:break
                snapshot=admission(policy);write(root/'ADMISSION_LATEST.json',{**snapshot,'time_utc':utc(),'scene_id':scene,'candidate':candidate})
                if snapshot['status']=='admitted':break
                emit_progress(root,cfg,'resource pause before new parent '+json.dumps(snapshot))
                time.sleep(30)
            if stop_requested:break
            cpus=idle_cpu_order()[:policy.maximum_cpu_processes]
            attempt.mkdir(parents=True);log=attempt/'worker.log'
            command=[sys.executable,str(Path(__file__).resolve()),'--worker','--config',str(a.config.resolve()),'--output',str(attempt),
                '--run-root',str(root),'--room-index',str(ri),'--family',family,'--scene-id',scene,'--seed',str(seed),
                '--candidate',str(candidate),'--cpus',json.dumps(cpus),'--source-commit',identity['source_commit'],'--feedback',str(root/'feedback.json')]
            with log.open('xb')as stream:
                process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=child_env(cfg,root),start_new_session=True)
                live={'status':'running','pid':process.pid,'pgid':process.pid,'log':str(log),'scene_id':scene,'house':room['house'],'room_id':room['room_id'],
                    'target_family':family,'automatic_attempt_root':str(attempt),'source_commit':identity['source_commit']}
                write(record,live);last=0
                while process.poll()is None:
                    if time.monotonic()-last>=600:
                        emit_progress(root,cfg,'unattended '+json.dumps(live))
                        last=time.monotonic()
                    time.sleep(5)
                stream.flush();os.fsync(stream.fileno())
            if not (attempt/'automatic_result.json').exists():
                write(record,{**live,'status':'worker_ended_without_receipt','exit_code':process.returncode})
                raise RuntimeError('retain all files: worker ended without its failed or successful receipt')
            result=read(attempt/'automatic_result.json');write(record,result)
            emit_progress(root,cfg,'construction result '+json.dumps({**{k:result.get(k)for k in ('scene_id','status','reason','independent_questions','target_family')},'family_counts':result.get('export',{}).get('question_family_counts',{})}))
            if result['status']=='passed':break
            cleanup_new_failed(attempt,result,cfg.get('cleanup_failed_after_unix'),log)
            stop_reason=failure_stop(result,failure_history)
            if stop_reason:
                stop_requested=True;write(root/'AUTOMATION_STOP.json',stop_reason)
                emit_progress(root,cfg,'entire batch dispatch stopped '+json.dumps(stop_reason));break
        if record.exists():outcomes.append(read(record))
        state=from_passed_exports([*cfg.get('feedback_roots',[]),root]);write(root/'feedback.json',state)
    write(root/'AUTOMATION_RESULT.json',{'source_commit':identity['source_commit'],'plan_only':a.plan_only,'outcomes':outcomes,
        'handwritten_scene_configuration_inputs_read':[],'stop_requested':stop_requested,'stop_reason':stop_reason,'native_passed_parents':sum(r['status']=='passed'for r in outcomes),'family_counts':state['family_counts']})
    print(json.dumps({'output':str(root),'plan_only':a.plan_only,'native_passed_parents':sum(r['status']=='passed'for r in outcomes),'outcomes':[{'scene':r.get('scene_id'),'status':r['status']}for r in outcomes]},ensure_ascii=False))

if __name__=='__main__':main()

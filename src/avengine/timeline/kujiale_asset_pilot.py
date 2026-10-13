"""Train-only asset pilot behind the existing Kujiale production coordinator."""
from __future__ import annotations
import argparse,datetime,fcntl,importlib.util,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from copy import deepcopy
from avengine.timeline.continuous_pilot import write_json

def live_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.pending')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def progress(root,message):
    with (root/'PROGRESS.md').open('a') as f:f.write('- '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'：'+message+'\n')

def parent_result_directory(root):
    directory=Path(root)/'evidence/parent_results'
    directory.mkdir(parents=True,exist_ok=True)
    return directory


def latest_parent_result_path(root,scene_id):
    import re
    if not isinstance(scene_id,str) or Path(scene_id).name!=scene_id or scene_id in ('.','..'):
        raise ValueError('parent scene ID must be one filename component')
    directory=parent_result_directory(root)
    pattern=re.compile(re.escape(scene_id)+r'(?:\.(?:attempt)_\d+|_\d+)?\.json$')
    candidates=[path for path in directory.iterdir() if pattern.fullmatch(path.name)]
    for path in candidates:
        if not path.resolve().is_relative_to(Path(root).resolve()):raise ValueError('parent result escaped own output root')
    return max(candidates,key=lambda path:(path.stat().st_mtime_ns,path.name),default=None)


def read_parent_result(root,scene_id):
    path=latest_parent_result_path(root,scene_id)
    return json.loads(path.read_text()) if path else None


def publish_parent_result(root,scene_id,result):
    """Keep the original canonical record and every attempt immutable."""
    latest_parent_result_path(root,scene_id)  # Validate the ID and existing paths first.
    directory=parent_result_directory(root)
    with (directory/(scene_id+'.result.lock')).open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        canonical=directory/(scene_id+'.json')
        path=canonical if not canonical.exists() else directory/f'{scene_id}.attempt_{time.time_ns()}.json'
        write_json(path,result)
        live_json(directory/(scene_id+'.latest.json'),{'scene_id':scene_id,'result_path':str(path),
            'status':result.get('status'),'recorded_utc':utc_now()})
    return path


def freeze_inputs(root):
    """Choose the newest authorized audio admission when input sealing starts."""
    old=root.parent/'ue_round14/inputs';prior=root.parent/'ue_round20_assets'
    src=Path(json.loads((root/'evidence/planning_summary.json').read_text())['source_admission_inspected'])
    candidate=src.with_name('source_admission_v11_classes27_english_sao.json')
    actual=candidate if candidate.is_file() else src
    source=json.loads(actual.read_text())
    from avengine.timeline.v7_audio_sources import filter_admission
    _,rejected=filter_admission(source['records'])
    if rejected:raise ValueError('authorized sound admission contains blocked records: '+str(rejected[:3]))
    receipt={'status':'sealed_inputs','source_admission':str(actual),'input_sealing_started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'v11_present_at_input_sealing':candidate.is_file(),'records':len(source['records']),'owner_rule':'use v11 if available when sealing starts'}
    write_json(root/'inputs/source_admission.json',source)
    inputs=json.loads((old/'runtime.json').read_text())
    inputs.update(render_backend='ue_spear',semantic_manifest=None,kujiale_ue_asset_admission=json.loads((root/'inputs/task_scope.json').read_text())['asset_admission_path'])
    write_json(root/'inputs/runtime.json',inputs)
    import shutil
    for name in ['assets.json','human_admission.json']:shutil.copy2(old/name,root/'inputs'/name)
    write_json(root/'evidence/input_seal.json',receipt)
    progress(root,'封存本轮输入：'+actual.name+'，'+str(len(source['records']))+' 个声音记录；不修改源名册。')


def read_scope(root):
    return json.loads((Path(root)/'inputs/task_scope.json').read_text())


def own_input_directory(root,directory=None):
    root=Path(root).resolve();directory=Path(directory or root/'inputs')
    if not directory.is_absolute():directory=root/directory
    directory=directory.resolve()
    if not directory.is_relative_to(root) or not directory.is_dir():
        raise ValueError('pilot input directory must exist below its own output root')
    return directory


def own_seed_path(root,seed_path=None):
    root=Path(root).resolve();path=Path(seed_path or read_scope(root)['seed_path'])
    if not path.is_absolute():path=root/path
    path=path.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('pilot seed file must exist below its own output root')
    return path


def authorized_gpus(scope):
    values=scope.get('physical_gpus',[2,3])
    if (not isinstance(values,list) or not values or len(set(values))!=len(values)
            or any(type(gpu) is not int or gpu not in (1,2,3) for gpu in values)):
        raise ValueError('only explicitly assigned physical GPUs 1/2/3 are allowed; GPU0 is excluded')
    return sorted(values)


def dispatch_gpus(scope):
    """Schedule only selected cards; retain authorized device and readback checks."""
    allowed=authorized_gpus(scope)
    values=scope.get('dispatch_gpus',allowed)
    if (not isinstance(values,list) or not values or len(set(values))!=len(values)
            or any(type(gpu) is not int or gpu not in allowed for gpu in values)):
        raise ValueError('dispatch GPUs must be a nonempty unique subset of authorized physical GPUs')
    return sorted(values)


def load_pilot_inputs(root,input_directory=None):
    directory=own_input_directory(root,input_directory);bundle={}
    for name in ('houses','runtime','assets','source_admission','human_admission'):
        path=(directory/(name+'.json')).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError('pilot JSON input escaped own output root: '+name)
        bundle[name]=json.loads(path.read_text())
    if bundle['runtime'].get('render_backend')!='ue_spear':
        raise ValueError('Kujiale pilot requires the native UE/SPEAR visual backend')
    return directory,bundle


def validate_training_room(root,row,houses):
    matches=[h for h in houses if h['house']==row['house']]
    if len(matches)!=1:raise ValueError('one admitted training house required for this seed')
    house=deepcopy(matches[0]);room=row['room_id']
    if (row.get('split')!='train' or house.get('split')!='train' or house.get('family')!='kujiale'
            or house.get('acoustic_status')!='pass' or room not in house.get('checked_room_ids',[])):
        raise ValueError('pilot accepts admitted Kujiale training houses and checked rooms only')
    checked=[r for r in house.get('checked_rooms',[]) if r.get('room_label')==room]
    if (len(checked)!=1 or checked[0].get('native_qa_status')!='pass'
            or checked[0].get('floor_reference',{}).get('status')!='pass'):
        raise ValueError('selected room lacks its passed native UE floor/QA admission')
    # The retained original runtime is the authority for the 19-house/33-room
    # admission list. A derived input directory cannot invent another room.
    original=json.loads((Path(root)/'inputs/runtime.json').read_text())['houses']
    admitted={(h['house'],r['room_label']) for h in original
              if h.get('split')=='train' and h.get('family')=='kujiale' and h.get('acoustic_status')=='pass'
              for r in h.get('checked_rooms',[]) if r.get('room_label') in h.get('checked_room_ids',[])
              and r.get('native_qa_status')=='pass' and r.get('floor_reference',{}).get('status')=='pass'}
    if (row['house'],room) not in admitted:raise ValueError('derived seed room is outside the original native UE admission list')
    return house


def prepare_worker_runtime(cpu_id=None,root=None):
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','OPENCV_FOR_THREADS_NUM'):
        os.environ[key]='1'
    if os.nice(0)<10:raise ValueError('pilot worker requires nice >=10')
    cpus=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{cpus[0] if cpu_id is None else int(cpu_id)})
    import cv2
    cv2.setNumThreads(1)
    if root is not None:
        helper=load_local_helper(root,'pilot_resource_snapshot_v3.py')
        proof=Path(helper.write_thread_configuration_evidence(root,cv2_module=cv2))
        if not helper.configuration_verifies_single_pool(json.loads(proof.read_text())):
            raise ValueError('actual worker OpenCV/BLAS pools are not single threaded')


def worker(root,index,gpu,seed_path=None,input_directory=None,shared_worker_pool=False,cpu_id=None):
    prepare_worker_runtime(cpu_id,root)
    from avengine.timeline.v7_registered_generation import generate_registered_parent
    from avengine.timeline.v7_storage_budget import configure_kujiale_production
    scope=read_scope(root)
    if gpu not in authorized_gpus(scope):raise ValueError('physical GPU was not assigned to this task')
    rows=[json.loads(l) for l in own_seed_path(root,seed_path).read_text().splitlines() if l.strip()]
    row=rows[index];directory,bundle=load_pilot_inputs(root,input_directory)
    if formal_native_parent_passed(root,row['scene_id']):raise ValueError('completed native parent must not be regenerated')
    house=validate_training_room(root,row,bundle['houses']['houses'])
    ue=house['input']['ue_config']
    wrapper=Path(root)/'scripts'/('ue_shared' if shared_worker_pool else '')/f'ue_gpu{gpu}'
    if not wrapper.is_file():raise ValueError('assigned GPU UE wrapper is missing: '+str(wrapper))
    ue.update(graphics_adapter=gpu-1 if shared_worker_pool else 0,
              physical_gpu_index=gpu,unreal_editor=str(wrapper))
    (root/'evidence/parent_results').mkdir(parents=True,exist_ok=True)
    configure_kujiale_production(root/'production',root,workers=min(4,int(scope.get('maximum_workers_per_gpu',4))))
    start=time.time()
    try:
        result=generate_registered_parent(row,house,bundle['runtime'],bundle['assets']['assets'],
                                          bundle['source_admission'],bundle['human_admission'],root/'production')
        from avengine.qa.kujiale_pilot_truth import audit_sealed_parent
        truth=audit_sealed_parent(root/'production/parents'/row['scene_id'],result['delivery_directory'])
        if truth['status']!='pass' or truth['mismatches']:raise ValueError('independent sealed-parent truth audit failed')
        result['independent_truth']={'status':truth['status'],'checked_forms':truth['checked_forms'],'mismatches':len(truth['mismatches'])}
        result['pilot_scope']='local native integration trial; no training table export or CCI'
        result['pilot_input_directory']=str(directory)
    except BaseException as error:
        result={'scene_id':row['scene_id'],'house':row['house'],'status':'failed','error_type':type(error).__name__,'reason':str(error),'traceback':traceback.format_exc(),'wall_seconds':time.time()-start}
        publish_parent_result(root,row['scene_id'],result);progress(root,row['scene_id']+' 失败：'+str(error));raise
    publish_parent_result(root,row['scene_id'],result)
    progress(root,row['scene_id']+(' 正式通过：原生封存和独立真值均通过。' if formal_native_parent_passed(root,row['scene_id']) else ' 完成生成；缺少正式封存或独立真值，不作通过结论。'))
    print(json.dumps(result,ensure_ascii=False),flush=True)


class NativePilotBatchGate:
    """Require one fully sealed native parent, then stop on repeated failures."""
    def __init__(self,canary_passed=False,reviewed_parallel_batch_limit=2,canary_parallel_limit=1):
        self.canary_passed=canary_passed
        if type(reviewed_parallel_batch_limit) is not int or not 1<=reviewed_parallel_batch_limit<=12:
            raise ValueError('reviewed native batch limit must be from one to twelve')
        self.reviewed_parallel_batch_limit=reviewed_parallel_batch_limit
        self.last_failure=None
        self.consecutive_failures=0
        self.last_batch_failed=False
        if type(canary_parallel_limit) is not int or canary_parallel_limit not in (1,3):
            raise ValueError('explicit native canary cohort must contain one or three parents')
        self.canary_parallel_limit=canary_parallel_limit
        self.canary_failure_counts={}

    @property
    def batch_limit(self):
        return self.reviewed_parallel_batch_limit if self.canary_passed else self.canary_parallel_limit

    def observe(self,result):
        if result.get('status')=='formal_alpha_admitted' and result.get('independent_truth',{}).get('status')=='pass':
            self.canary_passed=True
            self.last_failure=None
            self.consecutive_failures=max(self.canary_failure_counts.values(),default=0) if self.canary_parallel_limit==3 else 0
            self.last_batch_failed=False
            return
        self.last_batch_failed=True
        import re
        reason=result.get('reason','worker exited before recording a gate result').split(':',1)[0]
        reason=re.sub(r'only \d+ measured binding questions','insufficient measured binding questions',reason)
        signature=(result.get('error_type','native_gate'),reason)
        if self.canary_parallel_limit==3:
            key=json.dumps(signature)
            self.canary_failure_counts[key]=self.canary_failure_counts.get(key,0)+1
            self.consecutive_failures=max(self.canary_failure_counts.values())
        else:
            self.consecutive_failures=self.consecutive_failures+1 if signature==self.last_failure else 1
        self.last_failure=signature

    def require_next_batch(self):
        if self.last_batch_failed and not self.canary_passed and self.canary_parallel_limit==1:
            raise RuntimeError('native canary failed; inspect and repair before starting another parent')
        if self.consecutive_failures>=2:
            raise RuntimeError('two consecutive native parents failed the same gate; rollout stopped for diagnosis')

    def state(self):
        return {'canary_passed':self.canary_passed,'batch_limit':self.batch_limit,
                'last_failure':self.last_failure,'consecutive_failures':self.consecutive_failures,
                'last_batch_failed':self.last_batch_failed,
                'reviewed_parallel_batch_limit':self.reviewed_parallel_batch_limit,
                'canary_parallel_limit':self.canary_parallel_limit,
                'canary_failure_counts':dict(self.canary_failure_counts)}


def formal_native_parent_passed(root,scene_id):
    terminal=latest_parent_result_path(root,scene_id)
    parent=root/'production/parents'/scene_id
    proof=parent/'independent_truth_audit.json'
    if terminal is None or not proof.exists() or not (parent/'alpha_seal_completed.json').exists():return False
    result=json.loads(terminal.read_text());truth=json.loads(proof.read_text())
    return result.get('status')=='formal_alpha_admitted' and truth.get('status')=='pass' and not truth.get('mismatches')


def load_local_helper(root,name):
    if name=='pilot_resource_snapshot_v3.py':name=read_scope(root).get('resource_snapshot_helper',name)
    if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):
        raise ValueError('task-local helper must be one filename component')
    scripts=(Path(root)/'scripts').resolve();path=(scripts/name).resolve()
    if not scripts.is_relative_to(Path(root).resolve()) or not path.is_relative_to(scripts) or not path.is_file():
        raise ValueError('task-local helper is missing or escaped own scripts: '+name)
    spec=importlib.util.spec_from_file_location('_kujiale_pilot_'+name.replace('.','_'),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def update_job(root,name,updates,register=False):
    """Merge one own job under the same flock used by the old trackers."""
    root=Path(root)
    with (root/'JOBS.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        jobs=json.loads((root/'JOBS.json').read_text()) if (root/'JOBS.json').exists() else []
        matches=[job for job in jobs if job.get('name')==name]
        if len(matches)>1:raise RuntimeError('own JOBS contains duplicate unique job names')
        if not matches:
            job={'name':name};jobs.append(job)
        else:job=matches[0]
        if register and job.get('status') in ('completed','failed','stopped'):
            updates={key:value for key,value in updates.items() if key!='status'}
        job.update(updates)
        live_json(root/'JOBS.json',jobs)
        return dict(job)


def parallel_canary_ids(scope):
    ids=scope.get('canary_scene_ids')
    if ids is None:return []
    if (not isinstance(ids,list) or len(ids)!=3 or len(set(ids))!=3 or
            any(not isinstance(sid,str) or Path(sid).name!=sid or sid in ('.','..') for sid in ids)):
        raise ValueError('parallel canary scope must declare exactly three unique scene IDs')
    if authorized_gpus(scope)!=[1,2,3]:
        raise ValueError('parallel canaries require explicitly assigned physical GPUs 1/2/3')
    return list(ids)


def parallel_canary_directory(root,scope):
    directory=own_input_directory(root,scope.get('canary_input_directory'))
    if not scope.get('canary_input_directory') or not directory.is_relative_to((Path(root)/'inputs').resolve()):
        raise ValueError('parallel canary input directory must exist inside its own inputs')
    return directory


def require_initial_canary_batch(root,scope,rows,directory,limit,start_index):
    ids=parallel_canary_ids(scope)
    if ids:
        if (directory!=parallel_canary_directory(root,scope) or limit!=3 or start_index!=0 or
                len(rows)!=3 or {row['scene_id'] for row in rows}!=set(ids)):
            raise RuntimeError('only the three explicitly declared native canaries are authorized before formal pass')
    elif directory!=own_input_directory(root):
        raise RuntimeError('required formal native canary has not passed; full rollout is not authorized yet')


def automatic_initial_batch_limit(scope,passed):
    value=scope.get('automatic_initial_reviewed_batch_limit',1)
    if type(value) is not int or not 1<=value<=min(4,reviewed_batch_limit(scope,True)):
        raise ValueError('automatic initial batch must be within the reviewed one to four parent limit')
    if value>1 and not passed:
        raise ValueError('larger automatic initial batch requires an actual formally passed canary')
    return value


def formal_canary_passed(root,scope,rows):
    ids=parallel_canary_ids(scope)
    if ids:return any(formal_native_parent_passed(root,sid) for sid in ids)
    required=scope.get('required_formal_canary_scene_id')
    if required:return formal_native_parent_passed(root,required)
    return any(formal_native_parent_passed(root,row['scene_id']) for row in rows)


def reviewed_batch_limit(scope,shared_worker_pool=False):
    value=scope.get('reviewed_parallel_batch_limit',1 if shared_worker_pool else 2)
    if type(value) is not int or not 1<=value<=12:
        raise ValueError('scope reviewed_parallel_batch_limit must be one to twelve')
    return value


def live_parent_workers(root):
    path=Path(root)/'JOBS.json'
    if not path.exists():return []
    helper=load_local_helper(root,'pilot_resource_snapshot_v3.py')
    result=[]
    for job in json.loads(path.read_text()):
        if job.get('status') not in ('starting','running') or not job.get('name','').startswith('parent_'):continue
        record=helper.proc_identity(job.get('pid',0))
        if record is None:continue
        if not helper.valid_child(record,job):
            raise RuntimeError('recorded own worker PID has a different command/start time; do not reuse its slot')
        result.append(job)
    return result


def require_launch_authorization(root,index,gpu,seed_path,input_directory,shared_worker_pool):
    scope=read_scope(root);gpus=authorized_gpus(scope)
    if gpu not in gpus:raise ValueError('requested physical GPU is outside the task scope')
    rows=[json.loads(line) for line in own_seed_path(root,seed_path).read_text().splitlines() if line.strip()]
    directory,bundle=load_pilot_inputs(root,input_directory)
    validate_training_room(root,rows[index],bundle['houses']['houses'])
    if formal_native_parent_passed(root,rows[index]['scene_id']):raise ValueError('completed native parent must not be regenerated')
    active=live_parent_workers(root)
    if any(job.get('scene_id')==rows[index]['scene_id'] or
           (job.get('parent_index')==index and job.get('seed_path')==str(own_seed_path(root,seed_path)))
           for job in active):raise RuntimeError('native parent is already running; duplicate launch rejected')
    passed=formal_canary_passed(root,scope,rows)
    ids=parallel_canary_ids(scope)
    limit=reviewed_batch_limit(scope,shared_worker_pool) if passed else (len(ids) or 1)
    if len(active)>=limit:raise RuntimeError('native pilot batch capacity reached; wait for current results')
    if scope.get('require_formal_canary_before_rollout') and not passed:
        required=scope.get('required_formal_canary_scene_id')
        if ids:
            require_initial_canary_batch(root,scope,rows,directory,len(rows),0)
            if rows[index]['scene_id'] not in ids:
                raise RuntimeError('native parent is outside the explicitly declared canary cohort')
            if any(job.get('gpu_physical')==gpu for job in active):
                raise RuntimeError('parallel native canaries use one instance per physical GPU')
        elif directory!=own_input_directory(root) or (required and rows[index]['scene_id']!=required) or (
                not required and index!=scope.get('canary_parent_index',0)):
            raise RuntimeError('required formal native canary has not passed; full rollout is not authorized yet')
    if shared_worker_pool:require_shared_gpu_isolation(scope)
    return scope,rows[index],directory


def require_shared_gpu_isolation(scope,device_directory=Path('/dev')):
    """Shared mode must run inside the caller's GPU0-hiding bwrap."""
    allowed=authorized_gpus(scope);device_directory=Path(device_directory)
    if (device_directory/'nvidia0').exists():
        raise ValueError('shared pilot wrapper must hide physical GPU0')
    for gpu in (1,2,3):
        if (device_directory/('nvidia'+str(gpu))).exists()!=(gpu in allowed):
            raise ValueError('shared pilot device visibility differs from explicitly assigned GPUs')


class ResourceLaunchHold(RuntimeError):
    def __init__(self,measurement):
        self.measurement=measurement
        super().__init__('pilot resource preflight blocks launch: '+', '.join(measurement['launch_hold_reasons']))


def launch_reservations(scope,measurement,active,shared_worker_pool,tracker_pending=False):
    """Reserve only the unobserved part of each measured peak, never an extra full peak."""
    peak=int(scope.get('shared_parent_peak_non_ue_processes' if shared_worker_pool else
                       'per_parent_peak_non_ue_processes',0))
    pss_peak=int(scope.get('per_parent_peak_pss_bytes',0))
    if peak<(2 if shared_worker_pool else 4) or pss_peak<=0:
        raise ValueError('measured conservative parent process/PSS reservations are required before dispatch')
    reserve_processes=peak+(1 if shared_worker_pool and tracker_pending else 0)
    reserve_pss=pss_peak
    for handle in active:
        records=[record for record in measurement.get('processes',[]) if handle.name in record.get('jobs',[])]
        actual_non_ue=sum(record.get('role')!='ue' for record in records)
        actual_pss=sum(record.get('pss_bytes') or 0 for record in records)
        reserve_processes+=max(0,int(handle.peak_non_ue)-actual_non_ue)
        reserve_pss+=max(0,int(handle.peak_pss)-actual_pss)
    return reserve_processes,reserve_pss


def resource_snapshot(root,scope,include_gpu=False,previous=None):
    helper=load_local_helper(root,'pilot_resource_snapshot_v3.py')
    return helper.snapshot(root,scope=scope,include_gpu=include_gpu,previous_snapshot=previous)


def preflight_launch(root,gpu,active,shared_worker_pool,previous=None):
    scope=read_scope(root)
    measurement=resource_snapshot(root,scope,include_gpu=True,previous=previous)
    tracker_pending=False
    if shared_worker_pool:
        from multiprocessing import resource_tracker
        tracker_pending=resource_tracker._resource_tracker._pid is None
    try:
        processes,pss=launch_reservations(scope,measurement,active,shared_worker_pool,tracker_pending)
    except ValueError as error:
        measurement['launch_allowed']=False
        measurement['launch_hold_reasons'].append(str(error))
    else:
        helper=load_local_helper(root,'pilot_resource_snapshot_v3.py')
        measurement.update(helper.evaluate_limits(measurement,scope,processes,pss))
    policy=json.loads((Path(root)/'kujiale_native_gpu_policy.json').read_text())
    expected=policy.get('gpu_uuids',{}).get(str(gpu))
    cards=[card for card in measurement.get('gpus',[]) if expected and card.get('uuid')==expected]
    if gpu not in authorized_gpus(scope) or gpu not in policy.get('allowed_gpu_indices',[]) or len(cards)!=1:
        measurement['launch_hold_reasons'].append('assigned_physical_GPU_UUID_not_visible')
    elif cards[0].get('free_bytes') is None or cards[0]['free_bytes']<(
            float(policy['measured_peak_mib'])+float(policy['headroom_mib']))*1024**2:
        measurement['launch_hold_reasons'].append('GPU_free_below_measured_peak_and_headroom')
    measurement['launch_allowed']=not measurement['launch_hold_reasons']
    measurement['requested_physical_gpu']=gpu;measurement['requested_physical_gpu_uuid']=expected
    live_json(Path(root)/'evidence/dispatch_resource_latest.json',measurement)
    if measurement.get('hard_stop_reasons'):raise RuntimeError('hard task resource limit: '+', '.join(measurement['hard_stop_reasons']))
    if not measurement['launch_allowed']:raise ResourceLaunchHold(measurement)
    return scope,measurement


class WorkerHandle:
    def __init__(self,process,kind,name,index,gpu,identity,scope):
        self.process=process;self.kind=kind;self.name=name;self.index=index;self.gpu=gpu
        self.pid=process.pid;self.identity=identity;self.started=time.time();self.observed_descendants={}
        self.peak_non_ue=int(scope['shared_parent_peak_non_ue_processes' if kind=='shared' else 'per_parent_peak_non_ue_processes'])
        self.peak_pss=int(scope['per_parent_peak_pss_bytes'])
    def poll(self):
        return self.process.exitcode if self.kind=='shared' else self.process.poll()
    def wait(self,timeout=None):
        if self.kind!='shared':return self.process.wait(timeout=timeout)
        self.process.join(timeout)
        if self.process.is_alive():raise subprocess.TimeoutExpired(self.name,timeout)
        return self.process.exitcode


def stable_process_identity(root,process):
    helper=load_local_helper(root,'pilot_resource_snapshot_v3.py')
    previous=None
    for _ in range(100):
        current=helper.proc_identity(process.pid)
        if current and current['command'][:3]==['nice','-n','10']:
            previous=None;time.sleep(.02);continue
        if current and previous and current['command']==previous['command'] and current['start_ticks']==previous['start_ticks']:
            return current
        previous=current;time.sleep(.02)
    raise RuntimeError('fresh own worker has no stable command/start-time identity')


def record_handle(root,handle,row,input_directory,seed_path,log_path):
    identity=handle.identity
    update_job(root,handle.name,{'status':'running','pid':handle.pid,'command':identity['command'],
        'start_ticks':identity['start_ticks'],'started_utc':utc_now(),'create_time_start':time.time(),
        'parent_index':handle.index,'scene_id':row['scene_id'],'gpu':handle.gpu,
        'coordinator_pid':os.getpid(),'uid':os.getuid(),'job_type':'native_parent_worker',
        'task_root':str(Path(root).resolve()),'gpu_physical':handle.gpu,'seed_path':str(seed_path),'input_directory':str(input_directory),
        'worker_mode':handle.kind,'log':str(log_path),'peak_non_ue_processes':handle.peak_non_ue,
        'peak_pss_bytes':handle.peak_pss,'stop_command':'verify PID/start_ticks/command, then task-local owned_process_cleanup_v1.stop_owned_tree'},register=True)


def shared_worker_entry(root,index,gpu,seed_path,input_directory,cpu_id,label,log_path):
    """Spawn inherits the coordinator's one shared resource_tracker fd."""
    root=Path(root)
    with Path(log_path).open('x',buffering=1) as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        sys.stdout=os.fdopen(os.dup(1),'w',buffering=1);sys.stderr=os.fdopen(os.dup(2),'w',buffering=1)
        helper=load_local_helper(root,'pilot_resource_snapshot_v3.py');identity=helper.proc_identity(os.getpid())
        update_job(root,label,{'status':'running','pid':os.getpid(),'command':identity['command'],
            'start_ticks':identity['start_ticks'],'create_time_start':time.time(),'started_utc':utc_now(),
            'parent_index':index,'gpu':gpu,'gpu_physical':gpu,'uid':os.getuid(),
            'job_type':'native_parent_worker','task_root':str(root.resolve()),'worker_mode':'shared','log':str(log_path)},register=True)
        def interrupted(signum,frame):raise RuntimeError('own worker interrupted by signal '+str(signum))
        for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,interrupted)
        started=time.time();started_ns=time.time_ns();code=0
        try:worker(root,index,gpu,Path(seed_path),Path(input_directory),True,cpu_id)
        except BaseException as error:
            code=1;traceback.print_exc()
            rows=[json.loads(line) for line in own_seed_path(root,seed_path).read_text().splitlines() if line.strip()]
            row=rows[index];path=latest_parent_result_path(root,row['scene_id'])
            if not formal_native_parent_passed(root,row['scene_id']) and (path is None or path.stat().st_mtime_ns<started_ns):
                result={'scene_id':row['scene_id'],'house':row['house'],'status':'failed',
                    'error_type':type(error).__name__,'reason':str(error),'traceback':traceback.format_exc()}
                publish_parent_result(root,row['scene_id'],result);progress(root,row['scene_id']+' worker 初始化失败：'+str(error))
            raise
        finally:
            update_job(root,label,{'status':'completed' if code==0 else 'failed','returncode':code,
                'finished_utc':utc_now(),'wall_seconds':time.time()-started,
                'gpu_assigned_wall_seconds':time.time()-started,'gpu_allocated_seconds':time.time()-started})


def launch_worker(root,index,gpu,slot,seed_path=None,input_directory=None,
                  shared_worker_pool=False,cpu_id=None,active=(),previous_snapshot=None):
    root=Path(root);seed=own_seed_path(root,seed_path)
    scope,row,directory=require_launch_authorization(root,index,gpu,seed,input_directory,shared_worker_pool)
    scope,measurement=preflight_launch(root,gpu,active,shared_worker_pool,previous_snapshot)
    repo=Path(__file__).resolve().parents[3]
    env=os.environ.copy();env.update(json.loads((root/'ue_env.json').read_text()))
    env.update(PYTHONPATH=str(repo/'src')+':'+str(repo)+':'+str(root/'pydeps')+':'+env.get('PYTHONPATH',''),
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',OPENCV_FOR_THREADS_NUM='1',TMPDIR=str(root/'tmp'))
    label=f'parent_{index:03d}_gpu{gpu}_{time.time_ns()}'
    log_path=root/'logs'/f'{label}_native.log';log_path.parent.mkdir(parents=True,exist_ok=True)
    if shared_worker_pool:
        import multiprocessing
        env.pop('CUDA_VISIBLE_DEVICES',None)
        os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        # All children, including the dry-audio subprocess, inherit this shared
        # tracker. No per-parent supervisor or bwrap consumes a CPU process slot.
        os.environ.update(env)
        process=multiprocessing.get_context('spawn').Process(target=shared_worker_entry,
            args=(root,index,gpu,seed,directory,cpu_id,label,log_path),name=label)
        process.start();kind='shared'
    else:
        cpus=sorted(os.sched_getaffinity(0));affinity=cpus[(slot*16)%len(cpus):][:16]
        cmd=['taskset','-c',','.join(map(str,affinity)),'/data/jzy/envs/room-selection-merged-20261003/bin/python',
            str(repo/'tools/timeline/v7_kujiale_production_v2.py'),'--pilot-root',str(root),
            '--parent-index',str(index),'--physical-gpu',str(gpu),'--seed-path',str(seed),
            '--input-directory',str(directory)]
        if cpu_id is not None:cmd+=['--worker-cpu',str(cpu_id)]
        sandbox=['/usr/bin/bwrap','--die-with-parent','--unshare-user','--ro-bind','/','/','--dev','/dev','--proc','/proc',
            '--bind',str(root),str(root),'--bind',str(root/'tmp'),'/tmp','--bind',str(root/'shm'),'/dev/shm',
            '--bind',str(root/'ue_config'),'/data/jzy/.config','--dev-bind','/dev/nvidiactl','/dev/nvidiactl',
            '--dev-bind','/dev/nvidia'+str(gpu),'/dev/nvidia'+str(gpu),'--dev-bind','/dev/nvidia-uvm','/dev/nvidia-uvm',
            '--dev-bind','/dev/nvidia-uvm-tools','/dev/nvidia-uvm-tools','--chdir',str(repo)]
        env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with log_path.open('x') as log:
            process=subprocess.Popen(['nice','-n','10',*sandbox,*cmd],env=env,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        kind='legacy'
    try:
        identity=stable_process_identity(root,process)
        handle=WorkerHandle(process,kind,label,index,gpu,identity,scope)
        record_handle(root,handle,row,directory,seed,log_path)
    except BaseException:
        helper=load_local_helper(root,'owned_process_cleanup_v1.py');identity=helper.snapshot(process.pid)
        running=process.exitcode is None if kind=='shared' else process.poll() is None
        if running and identity and identity['ppid']==os.getpid():
            helper.stop_owned_tree(process,identity['command'])
        raise
    return handle


def remember_descendants(active,measurement):
    for handle in active:
        for record in measurement.get('processes',[]):
            if handle.name in record.get('jobs',[]) and record['pid']!=handle.pid:
                handle.observed_descendants[record['pid']]=record


def cleanup_worker(root,handle):
    """Never signal a reused PID or an unrelated command."""
    helper=load_local_helper(root,'owned_process_cleanup_v1.py')
    current=helper.snapshot(handle.pid)
    if current is not None:
        if not helper.still_same(handle.identity):
            raise RuntimeError('own cleanup command/start time changed; refusing PID '+str(handle.pid))
        stopped=helper.stop_owned_tree(handle,handle.identity['command'])
    else:stopped=[]
    # An unexpectedly exited parent can leave already observed own children.
    # These identities were recorded while anchored beneath this exact parent.
    for record in handle.observed_descendants.values():
        if helper.still_same(record):
            class ExactPID:
                pid=record['pid']
            stopped.extend(helper.stop_owned_tree(ExactPID(),record['command']))
    handle.wait(timeout=25)
    update_job(root,handle.name,{'status':'stopped','finished_utc':utc_now(),'returncode':handle.poll(),
        'verified_cleanup_pids':[record['pid'] for record in stopped],
        'gpu_assigned_wall_seconds':time.time()-handle.started,'gpu_allocated_seconds':time.time()-handle.started})


def result_for_worker(root,row,returncode):
    result=read_parent_result(root,row['scene_id']) or {
        'scene_id':row['scene_id'],'house':row['house'],'status':'failed','error_type':'WorkerExit',
        'reason':'worker exited before recording a gate result','returncode':returncode}
    if formal_native_parent_passed(root,row['scene_id']):
        return dict(result,independent_truth={'status':'pass'})
    if result.get('status')=='formal_alpha_admitted':
        return dict(result,status='failed',reason='formal parent lacks completed native seal or passed independent truth')
    return result


def review_completed_batch(root,scene_ids,returncodes):
    """Read actual seals/truth after every member exits; no image-viewing judgement."""
    checks=[]
    for scene_id in scene_ids:
        parent=Path(root)/'production/parents'/scene_id;failure=None
        seal_path=parent/'alpha_seal_completed.json';truth_path=parent/'independent_truth_audit.json'
        result_path=latest_parent_result_path(root,scene_id)
        try:
            result=read_parent_result(root,scene_id) or {}
            seal=json.loads(seal_path.read_text());truth=json.loads(truth_path.read_text())
            attempt=(parent/seal['attempt']).resolve()
            files=seal.get('required_files',[])
            if (not attempt.is_relative_to(parent.resolve()) or not files or
                    any(not (attempt/name).resolve().is_relative_to(attempt) or not (attempt/name).is_file() for name in files)):
                failure='completed native seal files are missing or escaped'
            elif (returncodes.get(scene_id)!=0 or result.get('status')!='formal_alpha_admitted'
                    or seal.get('result',{}).get('status')!='formal_alpha_admitted'
                    or result.get('independent_truth',{}).get('status')!='pass'
                    or result.get('independent_truth',{}).get('mismatches')!=0
                    or truth.get('status')!='pass' or truth.get('mismatches')
                    or int(truth.get('checked_forms',0))<=0):
                failure='worker, completed native seal, or independent truth has not passed'
        except (OSError,KeyError,TypeError,ValueError) as error:failure=type(error).__name__+': '+str(error)
        checks.append({'scene_id':scene_id,'status':'fail' if failure else 'pass','failure':failure,
            'latest_result':str(result_path) if result_path else None,'native_seal':str(seal_path),
            'independent_truth':str(truth_path),'worker_returncode':returncodes.get(scene_id)})
    review={'status':'pass' if checks and all(check['status']=='pass' for check in checks) else 'fail',
            'utc':utc_now(),'parents':len(checks),'checks':checks,'native_gates_relaxed':False}
    path=Path(root)/'evidence'/f'batch_review_{time.time_ns()}.json';write_json(path,review)
    progress(Path(root),'自动批次核读 '+str(len(checks))+' 个父场景：'+review['status']+'；证据 '+str(path))
    return review,path


def record_automatic_batch_limit(root,limit,review_path=None):
    # Only this coordinator changes the opt-in batch state. Merge fresh scope
    # fields under a lock; deadline and resource values are never replaced.
    with (Path(root)/'inputs/task_scope.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX);scope=read_scope(root)
        scope['reviewed_parallel_batch_limit']=limit
        scope['automatic_native_batch_review']={'mode':'continue-reviewed-batches','next_batch_limit':limit,
            'last_review':str(review_path) if review_path else None,'utc':utc_now()}
        live_json(Path(root)/'inputs/task_scope.json',scope)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-root',type=Path,required=True);p.add_argument('--parent-index',type=int)
    p.add_argument('--seed-path',type=Path);p.add_argument('--input-directory',type=Path)
    p.add_argument('--physical-gpu',type=int);p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--limit',type=int,default=12);p.add_argument('--workers-per-gpu',type=int,default=2)
    p.add_argument('--shared-worker-pool',action='store_true');p.add_argument('--worker-cpu',type=int)
    p.add_argument('--continue-reviewed-batches',action='store_true')
    a=p.parse_args();root=a.pilot_root.resolve()
    if root.name!='ue_round21_pilot':raise ValueError('pilot output must be the authorized round21 directory')
    if a.parent_index is not None:
        worker(root,a.parent_index,a.physical_gpu,a.seed_path,a.input_directory,a.shared_worker_pool,a.worker_cpu);return
    if os.nice(0)<10:raise ValueError('pilot coordinator requires nice >=10')
    if not 1<=a.workers_per_gpu<=4:raise ValueError('one to four instances per assigned card')
    if a.limit<1 or a.start_index<0:raise ValueError('pilot limit/start index must be positive/nonnegative')
    if not (root/'evidence/input_seal.json').exists():
        if own_input_directory(root,a.input_directory)!=own_input_directory(root):
            raise ValueError('derived full inputs require retained original input seal')
        freeze_inputs(root)
    scope=read_scope(root);gpus=dispatch_gpus(scope)
    if a.shared_worker_pool:require_shared_gpu_isolation(scope)
    directory,bundle=load_pilot_inputs(root,a.input_directory);seed=own_seed_path(root,a.seed_path)
    rows=[json.loads(line) for line in seed.read_text().splitlines() if line.strip()][:a.limit]
    if len({row['scene_id'] for row in rows})!=len(rows):raise ValueError('seed scene IDs must be unique')
    for row in rows:validate_training_room(root,row,bundle['houses']['houses'])
    if live_parent_workers(root):raise RuntimeError('existing own parent workers are still running; no duplicate coordinator dispatch')
    passed=formal_canary_passed(root,scope,rows)
    if not passed:require_initial_canary_batch(root,scope,rows,directory,a.limit,a.start_index)
    initial_limit=automatic_initial_batch_limit(scope,passed) if a.continue_reviewed_batches else reviewed_batch_limit(scope,a.shared_worker_pool)
    cohort_limit=3 if parallel_canary_ids(scope) and not passed else 1
    gate=NativePilotBatchGate(passed,initial_limit,cohort_limit)
    pending=[index for index,row in enumerate(rows) if index>=a.start_index and
             not formal_native_parent_passed(root,row['scene_id'])]
    active=[];processed=[];finished_results=[];last_snapshot=None;last_monitor=0.;hold={}
    batch_indices=[];batch_returncodes={};batch_reviews=[];automatic_failure=False
    checkpoint=root/'evidence/controlled_gate_checkpoint.json'
    import hashlib
    digest=hashlib.sha256()
    for source_file in sorted(Path(__file__).resolve().parents[1].rglob('*.py')):
        digest.update(str(source_file.relative_to(Path(__file__).resolve().parents[1])).encode());digest.update(source_file.read_bytes())
    digest.update(seed.read_bytes())
    for name in ('houses','runtime','assets','source_admission','human_admission'):
        path=directory/(name+'.json');digest.update(path.name.encode());digest.update(path.read_bytes())
    revision=digest.hexdigest()
    if checkpoint.exists():
        previous=json.loads(checkpoint.read_text())
        if previous.get('code_and_seed_revision')==revision:
            gate.last_failure=tuple(previous['gate']['last_failure']) if previous['gate'].get('last_failure') else None
            gate.consecutive_failures=previous['gate']['consecutive_failures']
            gate.last_batch_failed=previous['gate'].get('last_batch_failed',False)
            gate.canary_failure_counts=dict(previous['gate'].get('canary_failure_counts',{}))
    gate.require_next_batch()
    lock=(root/'coordinator.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpus=scope.get('non_ue_worker_cpu_ids',scope.get('worker_cpu_ids',sorted(os.sched_getaffinity(0))))
    if not cpus or any(type(cpu) is not int or cpu<0 for cpu in cpus):raise ValueError('worker_cpu_ids must name actual allowed compute CPUs')
    controller_cpu=scope.get('controller_cpu_id',cpus[0])
    if type(controller_cpu) is not int or controller_cpu<0:raise ValueError('invalid controller CPU')
    os.sched_setaffinity(0,{controller_cpu})
    def interrupted(signum,frame):raise RuntimeError('own coordinator interrupted by signal '+str(signum))
    for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,interrupted)
    if a.continue_reviewed_batches:record_automatic_batch_limit(root,initial_limit)
    try:
        # One invocation dispatches only its reviewed batch. An early result
        # never opens a new slot before the rest of that batch has completed.
        dispatch_done=False
        while pending or active:
            scope=read_scope(root)
            if time.time()>=float(scope['deadline_epoch'])-int(scope.get('cleanup_before_deadline_seconds',180)):
                raise RuntimeError('authorized pilot deadline cleanup reached')
            if (root/'STOP').exists():raise RuntimeError('own STOP requested')
            if active and time.monotonic()-last_monitor>=30:
                last_snapshot=resource_snapshot(root,scope,previous=last_snapshot)
                remember_descendants(active,last_snapshot);last_monitor=time.monotonic()
                live_json(root/'evidence/runtime_resource_latest.json',last_snapshot)
                if last_snapshot.get('hard_stop_reasons'):
                    raise RuntimeError('hard task resource limit: '+', '.join(last_snapshot['hard_stop_reasons']))
            if not dispatch_done:
                gate.require_next_batch();batch_limit=min(gate.batch_limit,len(gpus)*a.workers_per_gpu)
                per_gpu_limit=1 if cohort_limit==3 and not gate.canary_passed else a.workers_per_gpu
                for gpu in gpus:
                    count=sum(handle.gpu==gpu for handle in active)
                    while pending and count<per_gpu_limit and len(active)<batch_limit:
                        index=pending[0]
                        previous_recipe=root/'production/parents'/rows[index]['scene_id']/'input_recipe.json'
                        if previous_recipe.exists() and json.loads(previous_recipe.read_text())['house']['input']['ue_config']['physical_gpu_index']!=gpu:break
                        try:
                            handle=launch_worker(root,index,gpu,len(active),seed,directory,a.shared_worker_pool,
                                cpus[len(processed)%len(cpus)],active,last_snapshot)
                        except ResourceLaunchHold as error:
                            hold={'gpu':gpu,'reasons':error.measurement['launch_hold_reasons']};last_snapshot=error.measurement
                            break
                        pending.pop(0);processed.append(index);batch_indices.append(index);active.append(handle);count+=1
                        live_json(root/'evidence/coordinator_state.json',{'status':'running',
                            'active':[{'pid':h.pid,'start_ticks':h.identity['start_ticks'],'parent_index':h.index,'gpu':h.gpu,'worker_mode':h.kind} for h in active],
                            'pending':pending,'input_directory':str(directory),'gate':gate.state(),
                            'original_generator':'33b5b12df1a45ca265bf0c4fbe9b26b664577a7c'})
                if active:dispatch_done=True
                elif pending:
                    live_json(root/'evidence/coordinator_state.json',{'status':'resource_hold','pending':pending,
                        'hold':hold,'gate':gate.state(),'input_directory':str(directory)})
            for handle in list(active):
                returncode=handle.poll()
                if returncode is None:continue
                handle.wait(timeout=1);active.remove(handle)
                update_job(root,handle.name,{'status':'completed' if returncode==0 else 'failed',
                    'returncode':returncode,'finished_utc':utc_now(),'gpu_assigned_wall_seconds':time.time()-handle.started,
                    'gpu_allocated_seconds':time.time()-handle.started})
                result=result_for_worker(root,rows[handle.index],returncode);finished_results.append(result);gate.observe(result)
                batch_returncodes[rows[handle.index]['scene_id']]=returncode
                live_json(checkpoint,{'code_and_seed_revision':revision,'gate':gate.state()})
                live_json(root/'evidence/controlled_rollout_state.json',{'status':'batch_result','gate':gate.state(),
                    'scene_id':rows[handle.index]['scene_id'],'result':result,'pending':pending})
                # A failed Python parent must not leave any observed native descendants.
                cleanup=load_local_helper(root,'owned_process_cleanup_v1.py')
                if any(cleanup.still_same(row) for row in handle.observed_descendants.values()):cleanup_worker(root,handle)
            if batch_indices and not active:
                if not a.continue_reviewed_batches:
                    progress(root,'本批 '+str(len(processed))+' 个父场景全部结束；剩余 '+str(len(pending))+' 个不自动排队，先检查本批结果。')
                    break
                review,path=review_completed_batch(root,[rows[index]['scene_id'] for index in batch_indices],batch_returncodes)
                batch_reviews.append(str(path))
                if review['status']!='pass':automatic_failure=True;break
                gate.require_next_batch()
                # A resource-truncated batch does not certify a higher level.
                if len(batch_indices)==gate.batch_limit:
                    gate.reviewed_parallel_batch_limit=min(4,gate.batch_limit*2)
                record_automatic_batch_limit(root,gate.reviewed_parallel_batch_limit,path)
                batch_indices=[];batch_returncodes={};dispatch_done=False
                if pending:continue
                break
            time.sleep(5)
    finally:
        cleanup_errors=[]
        for handle in active:
            try:cleanup_worker(root,handle)
            except BaseException as error:cleanup_errors.append(str(error))
        lock.close()
        if cleanup_errors:raise RuntimeError('exact own-worker cleanup failed: '+str(cleanup_errors))
    live_json(root/'evidence/coordinator_state.json',{'status':'automatic_batch_failed' if automatic_failure else 'batch_review_required' if pending else 'finished',
        'pending_not_queued':pending,'parents':len(finished_results),
        'passed':sum(formal_native_parent_passed(root,rows[index]['scene_id']) for index in processed),
        'results':finished_results,'gate':gate.state(),'input_directory':str(directory),
        'continue_reviewed_batches':a.continue_reviewed_batches,'batch_reviews':batch_reviews,
        'training_exported':False,'CCI_transferred':False})
    progress(root,'试产协调器结束，所有本批 worker 已退出，结果留在本轮目录；下一批需审看结果。')
if __name__=='__main__':main()

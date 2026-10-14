#!/usr/bin/env python3
"""Restartable private phase-two registered CPU integration shards."""
from __future__ import annotations
import argparse,json,multiprocessing,os,resource,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path


def pin_integration_worker_cpu():
    """Pin before native imports; restrict any pre-existing worker threads too."""
    if not hasattr(os,'sched_getaffinity'):
        raise RuntimeError('integration pilot requires Linux CPU affinity')
    allowed=sorted(os.sched_getaffinity(0))
    selected={allowed[os.getpid()%len(allowed)]}
    os.sched_setaffinity(0,selected)
    for task in Path('/proc/self/task').iterdir():
        try:
            os.sched_setaffinity(int(task.name),selected)
        except ProcessLookupError:
            continue
    return next(iter(selected))


def worker(task):
    start=time.perf_counter();usage=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu=usage.ru_utime+usage.ru_stime+child.ru_utime+child.ru_stime
    try:
        if task[2].get('integration_pilot'):
            selected_cpu=pin_integration_worker_cpu()
            import datetime
            identity={'pid':os.getpid(),'parent_pid':os.getppid(),'scene_id':task[0]['scene_id'],
                'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'nice':os.getpriority(os.PRIO_PROCESS,0),'cpu_affinity':sorted(os.sched_getaffinity(0)),
                'single_cpu_selected_before_native_imports':selected_cpu,'one_parent_per_worker':True}
            receipt_root=Path(task[-1])/'receipts';receipt_root.mkdir(exist_ok=True)
            (receipt_root/f"worker_started_{os.getpid()}.json").write_text(json.dumps(identity,indent=2))
            from avengine.timeline.v7_phase2_fallback import generate_parent_with_element_fallback as generate_phase2_parent
            result=generate_phase2_parent(*task)
            identity.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                status=result['status'],maximum_resident_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                ending_cpu_affinity=sorted(os.sched_getaffinity(0)))
            (receipt_root/f"worker_finished_{os.getpid()}.json").write_text(json.dumps(identity,indent=2))
            return {**result,'worker_resource_receipt':identity}
        from avengine.timeline.v7_registered_generation import generate_registered_parent as generate_alpha_parent
        return generate_alpha_parent(*task)
    except Exception as error:
        row=task[0];root=Path(task[-1])/'parents'/row['scene_id'];root.mkdir(parents=True,exist_ok=True)
        result={'status':'failed_not_formal','scene_id':row['scene_id'],'house':row['house'],
            'error_type':type(error).__name__,'reason':str(error),'traceback':traceback.format_exc(),
            'alpha_formal_windows':0,'alpha_formal_logical_questions':0}
        usage=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
        result.update(wall_seconds=time.perf_counter()-start,cpu_seconds_including_children=usage.ru_utime+usage.ru_stime+child.ru_utime+child.ru_stime-cpu)
        from avengine.timeline.continuous_pilot import write_json
        try:write_json(root/f'failed_{time.time_ns()}.json',result)
        except Exception as report_error:
            print('failure receipt stopped by storage guard: '+str(report_error),flush=True)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['houses','seeds','runtime-inputs','assets','source-admission','human-admission','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--demand-state',type=Path,help='Task-owned actual selected-family feedback; plan each next batch from its deficits')
    p.add_argument('--demand-focus',choices=['binding','human_motion','animal_motion','count'],help='Explicit validation focus; otherwise use deficit-weighted scene requests')
    p.add_argument('--workers',type=int,default=1);p.add_argument('--offset',type=int,default=0);p.add_argument('--limit',type=int,default=1)
    p.add_argument('--storage-budget-root',type=Path)
    p.add_argument('--maximum-task-bytes',type=int,help='Tighter task-owned storage cap within the existing pilot allowance')
    p.add_argument('--production-contract',type=Path,help='Frozen v7 room/source/motion contract; automatically applied to every recipe')
    p.add_argument('--incremental-storage-baseline',type=Path)
    p.add_argument('--incremental-storage-root',type=Path)
    p.add_argument('--integration-pilot',action='store_true',help='Explicit registered motion/seat pilot; seal privately without a training export')
    p.add_argument('--inline-worker',action='store_true',help='One isolated single-CPU parent without idle process-pool helpers')
    p.add_argument('--render-backend',choices=['cpu_llvmpipe'],default='cpu_llvmpipe')
    a=p.parse_args()
    if not a.integration_pilot:raise ValueError('phase-two entry requires --integration-pilot; no formal export')
    if not 1<=a.workers<=32 or a.offset<0 or a.limit<1:raise ValueError('invalid CPU shard bounds')
    required={'CUDA_VISIBLE_DEVICES':'','LIBGL_ALWAYS_SOFTWARE':'1','MESA_LOADER_DRIVER_OVERRIDE':'llvmpipe','LP_NUM_THREADS':'1'}
    if any(os.environ.get(k)!=v for k,v in required.items()) or not os.environ.get('LD_PRELOAD'):
        raise ValueError('CPU llvmpipe env and CPU EGL device selector must be explicitly supplied')
    by_house={h['house']:h for h in json.loads(a.houses.read_text())['houses']}
    rows=[json.loads(l) for l in a.seeds.read_text().splitlines() if l.strip()][a.offset:a.offset+a.limit]
    if not rows or len({r['scene_id'] for r in rows})!=len(rows):raise ValueError('empty or duplicate seed shard')
    if a.inline_worker and (a.workers!=1 or len(rows)!=1):raise ValueError('inline worker requires exactly one parent and one worker')
    from avengine.timeline.v7_storage_budget import configure_local_review
    if a.integration_pilot:
        from avengine.timeline.v7_storage_budget import configure_integration_pilot
        from avengine.timeline.v7_phase2_sources import PHASE2_SCHEMA as SCHEMA
        if any(r.get('schema')!=SCHEMA or r.get('split')!='train' for r in rows):raise ValueError('integration pilot accepts only its explicit training-house schema')
        if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('integration pilot must run with nice -n 10')
        if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('integration pilot requires one thread per CPU worker')
        storage_policy=configure_integration_pilot(a.output,a.storage_budget_root,parents=len(rows),workers=a.workers)
        if bool(a.incremental_storage_baseline) != bool(a.incremental_storage_root):
            raise ValueError('incremental storage needs both immutable baseline and whole task root')
        if a.incremental_storage_baseline:
            from avengine.timeline.v7_storage_budget import ENV_KEY, allocated_bytes, check_write
            import hashlib
            baseline_path=a.incremental_storage_baseline.resolve()
            whole=a.incremental_storage_root.resolve()
            if (whole==Path('/data') or not whole.is_relative_to(Path('/data'))
                    or not a.output.resolve().is_relative_to(whole)
                    or not baseline_path.is_relative_to(a.storage_budget_root.resolve())):
                raise ValueError('incremental storage baseline/root escaped the authorized pilot')
            baseline=json.loads(baseline_path.read_text())
            initial=int(baseline['whole_pilot_disk_baseline_bytes'])
            allowance=int(baseline['additional_limit_bytes'])
            if initial<0 or not 0<allowance<=80_000_000_000:
                raise ValueError('invalid owner incremental disk allowance')
            worktree=Path(__file__).resolve().parents[2]
            code_added=max(0,allocated_bytes(worktree)-int(baseline['worktree_disk_baseline_bytes']))
            storage_policy.update(root=str(whole),maximum_task_bytes=initial+allowance-code_added-64_000_000,
                immutable_baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                baseline_path=str(baseline_path),additional_limit_bytes=allowance,
                worktree_added_bytes_at_launch=code_added,worktree_future_reserve_bytes=64_000_000,
                policy='owner phase2 incremental whole-pilot disk budget including original baseline and code growth')
            os.environ[ENV_KEY]=json.dumps(storage_policy,sort_keys=True)
            check_write(a.output,expected_bytes=0)
    else:storage_policy=configure_local_review(a.output,a.storage_budget_root,parents=len(rows),workers=a.workers)
    if a.maximum_task_bytes is not None:
        if not 0<a.maximum_task_bytes<=storage_policy['maximum_task_bytes']:raise ValueError('storage override may only tighten the original pilot allowance')
        from avengine.timeline.v7_storage_budget import ENV_KEY,check_write
        storage_policy['maximum_task_bytes']=a.maximum_task_bytes
        os.environ[ENV_KEY]=json.dumps(storage_policy,sort_keys=True);check_write(a.output,expected_bytes=0)
    inputs=json.loads(a.runtime_inputs.read_text());inputs['render_backend']=a.render_backend
    if a.production_contract:inputs['production_input_contract']=str(a.production_contract.resolve())
    if inputs.get('phase2_diagnostic_native_context_recycling'):
        raise ValueError('unqualified native context recycling is diagnostic only and cannot seal parents')
    inputs['phase2']=True
    if a.integration_pilot:inputs['integration_pilot']=True
    if any(r.get('motion_requests')for r in rows):
        import fcl  # private task dependency must be usable before native work
        for name in ('v7_asset_roster','moving_energy_budget'):
            if not inputs.get(name) or not Path(inputs[name]).is_file():raise ValueError('motion runtime requires an existing frozen '+name)
    assets=json.loads(a.assets.read_text());assets=assets['assets'] if isinstance(assets,dict) else assets
    source=json.loads(a.source_admission.read_text());human=json.loads(a.human_admission.read_text())
    if a.integration_pilot:
        from avengine.timeline.v7_visible_sources import validate_english_source_admission
        validate_english_source_admission(source)
    if a.demand_state:
        if not a.storage_budget_root or not a.demand_state.resolve().is_relative_to(a.storage_budget_root.resolve()):
            raise ValueError('demand feedback state must stay inside the task-owned storage root')
        from avengine.timeline.v7_demand import read_demand_state,plan_phase2_demand
        contract_path=inputs.get('production_input_contract')
        if not contract_path:raise ValueError('demand planning requires the original production input contract')
        contract=json.loads(Path(contract_path).read_text());state=read_demand_state(a.demand_state)
        rows=[row if row.get('scene_demand') else plan_phase2_demand(row,state,assets,source['records'],contract,focus=a.demand_focus) for row in rows]
        if any(row.get('motion_requests') for row in rows):
            import fcl
            for name in ('v7_asset_roster','moving_energy_budget'):
                if not inputs.get(name) or not Path(inputs[name]).is_file():raise ValueError('demand motion requires the existing '+name)
    tasks=[(row,by_house[row['house']],inputs,assets,source,human,str(a.output)) for row in rows]
    a.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter();results=[]
    def result_stream():
        if a.inline_worker:
            yield worker(tasks[0])
            return
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
            futures=[pool.submit(worker,task) for task in tasks]
            for future in as_completed(futures):
                yield future.result()
    for result in result_stream():
        results.append(result)
        if a.demand_state:
            from avengine.timeline.v7_demand import record_selected_parent
            record_selected_parent(a.demand_state,result)
        if a.integration_pilot:
            import datetime
            line=datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+result['scene_id']+' '+result['status']+' '+result.get('reason','父场景机器检查和私有封存完成')+'\n'
            with (a.output/'PROGRESS.md').open('a') as progress:progress.write(line)
        print(json.dumps({k:result.get(k) for k in ['scene_id','status','alpha_formal_windows','alpha_formal_logical_questions','wall_seconds','cpu_seconds_including_children','reason']},ensure_ascii=False),flush=True)
    from avengine.timeline.continuous_pilot import write_json
    from avengine.timeline.v7_registered_generation import ALPHA_RELEASE_PROFILE
    profile=ALPHA_RELEASE_PROFILE;pilot_fields={}
    if a.integration_pilot:
        from avengine.timeline.v7_phase2_generation import PILOT_PROFILE
        profile=PILOT_PROFILE;pilot_fields={'train_table_exported':False}
    success_status='integration_pilot_admitted' if a.integration_pilot else 'formal_alpha_admitted'
    manifest={'profile':profile,**pilot_fields,'selected_parents':len(rows),'workers':a.workers,'storage_policy':storage_policy,
        'controller_wall_seconds':time.perf_counter()-start,'parents':results,
        'alpha_formal_windows':sum(r['alpha_formal_windows'] for r in results),
        'alpha_formal_logical_questions':sum(r['alpha_formal_logical_questions'] for r in results),
        'full_v7_generation_ready':False,'status':'pass' if all(r['status']==success_status for r in results) else 'failed_not_all_parents_admitted'}
    path=a.output/f'shard_{a.offset:06d}_{len(rows):06d}_{time.time_ns()}.json';write_json(path,manifest)
    print(str(path),flush=True)
    if manifest['status']!='pass':raise SystemExit(2)

if __name__=='__main__':main()

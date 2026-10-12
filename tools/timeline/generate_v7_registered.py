#!/usr/bin/env python3
"""Restartable CPU-only formal static/recorded alpha shards."""
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
            pin_integration_worker_cpu()
            from avengine.timeline.v7_visible_generation import generate_visible_parent
            return generate_visible_parent(*task)
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
    p.add_argument('--workers',type=int,default=1);p.add_argument('--offset',type=int,default=0);p.add_argument('--limit',type=int,default=1)
    p.add_argument('--storage-budget-root',type=Path)
    p.add_argument('--integration-pilot',action='store_true',help='Explicit stationary visible-source pilot; seal privately without a training export')
    p.add_argument('--render-backend',choices=['cpu_llvmpipe'],default='cpu_llvmpipe')
    a=p.parse_args()
    if not 1<=a.workers<=64 or a.offset<0 or a.limit<1:raise ValueError('invalid CPU shard bounds')
    required={'CUDA_VISIBLE_DEVICES':'','LIBGL_ALWAYS_SOFTWARE':'1','MESA_LOADER_DRIVER_OVERRIDE':'llvmpipe','LP_NUM_THREADS':'1'}
    if any(os.environ.get(k)!=v for k,v in required.items()) or not os.environ.get('LD_PRELOAD'):
        raise ValueError('CPU llvmpipe env and CPU EGL device selector must be explicitly supplied')
    by_house={h['house']:h for h in json.loads(a.houses.read_text())['houses']}
    rows=[json.loads(l) for l in a.seeds.read_text().splitlines() if l.strip()][a.offset:a.offset+a.limit]
    if not rows or len({r['scene_id'] for r in rows})!=len(rows):raise ValueError('empty or duplicate seed shard')
    from avengine.timeline.v7_storage_budget import configure_local_review
    if a.integration_pilot:
        from avengine.timeline.v7_storage_budget import configure_integration_pilot
        from avengine.timeline.v7_visible_sources import SCHEMA
        if any(r.get('schema')!=SCHEMA or r.get('split')!='train' for r in rows):raise ValueError('integration pilot accepts only its explicit training-house schema')
        if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('integration pilot must run with nice -n 10')
        if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('integration pilot requires one thread per CPU worker')
        storage_policy=configure_integration_pilot(a.output,a.storage_budget_root,parents=len(rows),workers=a.workers)
    else:storage_policy=configure_local_review(a.output,a.storage_budget_root,parents=len(rows),workers=a.workers)
    inputs=json.loads(a.runtime_inputs.read_text());inputs['render_backend']=a.render_backend
    if a.integration_pilot:inputs['integration_pilot']=True
    if any(r.get('production_scenario',{}).get('motion_requests')for r in rows):
        import fcl  # private task dependency must be usable before native work
        for name in ('v7_asset_roster','moving_energy_budget'):
            if not inputs.get(name) or not Path(inputs[name]).is_file():raise ValueError('motion runtime requires an existing frozen '+name)
    assets=json.loads(a.assets.read_text());assets=assets['assets'] if isinstance(assets,dict) else assets
    source=json.loads(a.source_admission.read_text());human=json.loads(a.human_admission.read_text())
    if a.integration_pilot:
        from avengine.timeline.v7_visible_sources import validate_english_source_admission
        validate_english_source_admission(source)
    tasks=[(row,by_house[row['house']],inputs,assets,source,human,str(a.output)) for row in rows]
    a.output.mkdir(parents=True,exist_ok=True);start=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,task) for task in tasks]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            if a.integration_pilot:
                import datetime
                line=datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+result['scene_id']+' '+result['status']+' '+result.get('reason','父场景机器检查和私有封存完成')+'\n'
                with (a.storage_budget_root/'PROGRESS.md').open('a') as progress:progress.write(line)
            print(json.dumps({k:result.get(k) for k in ['scene_id','status','alpha_formal_windows','alpha_formal_logical_questions','wall_seconds','cpu_seconds_including_children','reason']},ensure_ascii=False),flush=True)
    from avengine.timeline.continuous_pilot import write_json
    from avengine.timeline.v7_registered_generation import ALPHA_RELEASE_PROFILE
    profile=ALPHA_RELEASE_PROFILE;pilot_fields={}
    if a.integration_pilot:
        from avengine.timeline.v7_visible_generation import PILOT_PROFILE
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

"""Optional exact-pose work queue while an existing fresh-RIR caller waits.

This lets a serial caller consume a task-owned parallel cache without stopping
its process. Every job still uses a separate native SDK process. The active
caller remains blocked until publication, so it cannot race checkpoint writes.
The current reference position is excluded because its caller already owns it.
"""
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,time,traceback,subprocess,sys,resource
import numpy as np
from avengine.timeline.continuous_pilot import PositionRIRCache,write_json


def run_prefetch_plan(request_path, request, setup_report):
    root=Path(request_path).resolve().parent.parent
    plan_path=root/'prefetch_plan.json'
    if not plan_path.is_file():return None
    plan=json.loads(plan_path.read_text())
    if plan.get('schema')!='v7-fresh-rir-prefetch-plan-v1':raise ValueError('unknown fresh RIR prefetch plan')
    plan_id=plan['plan_id']
    if not plan_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'for c in plan_id):
        raise ValueError('invalid prefetch plan ID')
    parent=root.parent.parent
    if Path(plan['parent_directory']).resolve()!=parent:raise ValueError('prefetch plan parent differs from its active request')
    cache_root=Path(plan['cache_directory']).resolve()
    if not cache_root.is_relative_to(parent):raise ValueError('prefetch cache leaves the active parent')
    from avengine.timeline.v7_phase2_fresh_rir import FreshNativeContextRenderer,REVISION
    identity={**request['expected_native_identity'],'phase2_context_policy':REVISION}
    if plan['cache_identity']!=identity:raise ValueError('prefetch configuration differs from the active native request')
    claim=root/('prefetch_claim_'+plan_id)
    try:claim.mkdir()
    except FileExistsError:return None
    started=time.monotonic();current=tuple(request['source_position_m']);inputs=deepcopy(request['inputs'])
    inputs.update(fresh_rir_workers=int(plan['workers']),fresh_rir_cpu_ids=list(plan['cpu_ids']))
    renderer=FreshNativeContextRenderer(inputs,request['house'],request['expected_native_identity'],setup_report,claim/'native')
    cache=PositionRIRCache(cache_root,renderer,identity)
    result={'status':'running','plan':str(plan_path),'plan_id':plan_id,'coordinator_pid':os.getpid(),
            'active_request':str(request_path),'current_role':request['evidence_role'],
            'current_position_excluded_from_prefetch':list(current),
            'primary_and_reference_native_processes_separate':True,'thresholds_changed':False}
    references=[]
    try:
        primary=[p for p in plan['primary_positions_m']if tuple(p)!=current]
        cache.prefetch(primary)
        jobs=[]
        for job in plan['references']:
            ep=job['endpoint_id']
            if not ep or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'for c in ep):
                raise ValueError('invalid prefetch endpoint ID')
            if tuple(job['position_m'])==current:continue
            jobs.append(job)
        def reference(job):
            p=job['position_m'];sample=int(job['sample']);ep=job['endpoint_id']
            checkpoint=parent/'independent_native_reference_checkpoint'/ep/(str(sample).zfill(8)+'.json')
            if checkpoint.is_file():return str(checkpoint)
            folder=claim/'references'/ep;folder.mkdir(parents=True,exist_ok=True);items=[]
            for trial in range(3):
                batch=renderer.render_independent_reference([p],trial)
                path=folder/(str(sample).zfill(8)+'_'+str(trial)+'.npy')
                with path.open('xb')as stream:np.save(stream,batch.samples[0],allow_pickle=False)
                items.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                              'native_process_evidence':batch.native_process_evidence})
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            write_json(checkpoint,{'actual_position_m':p,'native_identity':identity,'references':items,
                'source':'three unmodified independent native renders; resumable only at exact pose/configuration'})
            return str(checkpoint)
        with ThreadPoolExecutor(max_workers=int(plan['workers']))as pool:references=list(pool.map(reference,jobs))
        primary_records=[];native_cpu=0.
        for native_report in renderer.process_evidence:
            value=json.loads(Path(native_report).read_text());native_cpu+=value['process_cpu_seconds_total']
            if value['evidence_role']!='primary':continue
            key=json.dumps(value['source_position_m'],separators=(',',':'))
            index,completed=cache.db.execute('SELECT id,completed FROM rirs WHERE pose=?',(key,)).fetchone()
            path=cache_root/f'rir_{index:06d}.npy';sha=hashlib.sha256(path.read_bytes()).hexdigest()
            if not completed or sha!=value['output_sha256']:raise ValueError('prefetched primary differs from its fresh native report')
            primary_records.append({'position_m':value['source_position_m'],'cache_file':str(path),
                                    'sha256':sha,'native_report':native_report})
        result.update(status='pass',kind='measured_cache_prefetch_only_not_parent_admission',
                      primary_cache_statistics=cache.stats.copy(),primary_records=primary_records,
                      reference_checkpoints=references,native_process_evidence=list(renderer.process_evidence),
                      native_jobs=len(renderer.process_evidence),native_cpu_seconds_total=native_cpu)
    except Exception as error:result.update(status='fail',reason=str(error),traceback=traceback.format_exc())
    finally:cache.db.close()
    result['wall_seconds']=time.monotonic()-started
    write_json(claim/'prefetch_result.json',result)
    return str(claim/'prefetch_result.json')


def maybe_schedule_prefetch(request_path, request, setup_report):
    """Keep the measuring native process separate from the cache coordinator."""
    path=Path(request_path).resolve();root=path.parent.parent
    if not(root/'prefetch_plan.json').is_file():return None
    if int(request.get('inputs',{}).get('fresh_rir_workers',1))!=1:return None
    finished=root/'prefetch_finished.json'
    if finished.is_file():return json.loads(finished.read_text()).get('receipt')
    setup=path.parent/'prefetch_setup.json';write_json(setup,setup_report)
    with(path.parent/'prefetch_scheduler.log').open('xb')as log:
        result=subprocess.run([sys.executable,'-B','-m','avengine.timeline.v7_phase2_rir_prefetch',
            '--request',str(path),'--setup-report',str(setup)],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise ValueError('optional prefetch scheduler failed: '+str(path.parent/'prefetch_scheduler.log'))
    return json.loads(finished.read_text()).get('receipt')if finished.is_file()else None


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--setup-report',required=True)
    args=p.parse_args();path=Path(args.request).resolve();request=json.loads(path.read_text())
    receipt=run_prefetch_plan(path,request,json.loads(Path(args.setup_report).read_text()))
    if receipt is not None:
        finished=path.parent.parent/'prefetch_finished.json'
        write_json(finished,{'receipt':receipt,'scheduler_pid':os.getpid(),
            'scheduler_process_cpu_seconds':time.process_time(),
            'native_descendant_cpu_seconds':resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime})

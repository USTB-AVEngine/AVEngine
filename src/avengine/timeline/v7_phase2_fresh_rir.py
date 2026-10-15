"""Fresh native RIR processes, with bounded task-selected CPU concurrency.

Primary and independent references retain separate native uploads and PIDs.
Only scheduling changes; every pose, ray budget and engineering check remains.
"""
from __future__ import annotations
import json,os,pathlib,subprocess,sys,time,uuid,hashlib,queue,fcntl,shutil
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
from avengine.timeline.v7_storage_budget import check_write
REVISION='fresh-native-process-per-rir-independent-midpoints-v1'


def _launch_native_child(command,log,inputs):
    # Shared lock is opened read-only: helpers never write another task's files.
    configured=inputs.get('shared_rir_policy')
    if configured is None and inputs.get('production_input_contract'):
        candidate=pathlib.Path(inputs['production_input_contract']).parent/'RIR_SHARED_POLICY.json'
        if candidate.is_file():configured=str(candidate)
    if not configured:
        return subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    policy=json.loads(pathlib.Path(configured).read_text())
    from tools.timeline.monitor_v7_acceptance import proc,processes
    from avengine.timeline.v7_storage_budget import launch_minimum_free_bytes,resolve_launch_disk_policy
    last_log=0.
    with pathlib.Path(policy['lock']).open('rb')as lock:
        while True:
            policy=json.loads(pathlib.Path(configured).read_text())
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                roots=[];peer_roots=[]
                for path in pathlib.Path('/proc').iterdir():
                    if not path.name.isdigit():continue
                    value=proc(int(path.name))
                    if value and value['state']!='Z':
                        if any(root in value['cmd']for root in policy['output_roots']):roots.append(value['pid'])
                        elif any(root in value['cmd']for root in policy.get('peer_output_roots',[])):peer_roots.append(value['pid'])
                records=processes(roots);pss=sum(r['pss_bytes']for r in records)
                from avengine.timeline.v7_cpu_budget import compute_process_count
                peers=processes(peer_roots);owned_ids={r['pid']for r in records};peers=[r for r in peers if r['pid']not in owned_ids]
                peer_pss=sum(r['pss_bytes']for r in peers)
                root_pid=int(inputs.get('native_resource_parent_pid',os.getpid()))
                if proc(root_pid)is None:root_pid=os.getpid()
                tree=processes([root_pid]);tree_pss=sum(r['pss_bytes']for r in tree)
                sdk=[r for r in records if 'avengine.timeline.v7_phase2_fresh_rir'in r['cmd']and '--request'in r['cmd']]
                reserve=int(policy.get('native_process_reserve_bytes',1_500_000_000))
                forecast=pss+sum(max(0,reserve-r['pss_bytes'])for r in sdk)+reserve
                all_sdk=[r for r in [*records,*peers]if 'avengine.timeline.v7_phase2_fresh_rir'in r['cmd']and '--request'in r['cmd']]
                tree_sdk=[r for r in tree if 'avengine.timeline.v7_phase2_fresh_rir'in r['cmd']and '--request'in r['cmd']]
                combined_forecast=pss+peer_pss+sum(max(0,reserve-r['pss_bytes'])for r in all_sdk)+reserve
                tree_forecast=tree_pss+sum(max(0,reserve-r['pss_bytes'])for r in tree_sdk)+reserve
                available=next(int(s.split()[1])*1024 for s in pathlib.Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:'))
                if (len(sdk)<min(16,int(policy.get("maximum_native_processes",16))) and compute_process_count([*records,*peers])<80 and forecast<=int(policy.get("maximum_pss_bytes",150_000_000_000))
                    and combined_forecast<=int(policy.get("maximum_combined_pss_bytes",190_000_000_000))
                    and tree_forecast<=int(policy.get("maximum_parent_tree_pss_bytes",30_000_000_000))
                    and available-sum(max(0,reserve-r['pss_bytes'])for r in all_sdk)-reserve>=30_000_000_000 and shutil.disk_usage('/data').free>=launch_minimum_free_bytes(resolve_launch_disk_policy(policy.get('launch_disk_policy',inputs.get('launch_disk_policy'))))):
                    return subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                if time.monotonic()-last_log>=30:
                    log.write(('waiting_for_new_native_process_admission '+json.dumps({'SDK_processes':len(sdk),'PSS_bytes':pss,'forecast_bytes':forecast,'combined_forecast_bytes':combined_forecast,'parent_tree_forecast_bytes':tree_forecast,'parent_tree_root_pid':root_pid,'MemAvailable_bytes':available,'critical_below_20GB':available<20_000_000_000})+'\n').encode());log.flush();last_log=time.monotonic()
            finally:fcntl.flock(lock,fcntl.LOCK_UN)
            time.sleep(.5)


class FreshNativeContextRenderer:
    def __init__(self,inputs,house,identity,setup_report,root):
        self.inputs=inputs;self.house=house;self.identity=identity
        self.root=pathlib.Path(root);check_write(self.root,expected_bytes=4_000_000)
        self.root.mkdir(parents=True,exist_ok=False)
        self.parallel_workers=min(16,max(1,int(inputs.get('fresh_rir_workers',1))))
        cpu_ids=inputs.get('fresh_rir_cpu_ids',[])
        self.cpu_pool=queue.Queue() if cpu_ids else None
        for cpu in cpu_ids:self.cpu_pool.put(int(cpu))
        self.setup_report={**setup_report,'phase2_context_policy':REVISION,
            'primary_and_reference_native_processes_separate':True,
            'reference_has_no_primary_cache_or_interpolated_estimate':True,
            'maximum_parallel_native_rirs':self.parallel_workers,
            'task_native_rir_cpu_ids':list(cpu_ids),'native_process_nice_at_least':10}
        self.process_evidence=[]

    def _render_one(self,position,evidence_role,reference_trial):
        point=np.asarray(position,float)
        if point.shape!=(3,)or not np.isfinite(point).all():raise ValueError('fresh native source must be finite xyz')
        token=uuid.uuid4().hex;directory=self.root/token
        check_write(directory,expected_bytes=4_000_000);directory.mkdir()
        request=directory/'request.json';output=directory/'rir.npy';report=directory/'result.json'
        request.write_text(json.dumps({'inputs':self.inputs,'house':self.house,
            'source_position_m':point.tolist(),'evidence_role':evidence_role,
            'reference_trial':reference_trial,'request_id':token,
            'expected_native_identity':self.identity,'output':str(output),'report':str(report)},allow_nan=False))
        selected_cpu=self.cpu_pool.get() if self.cpu_pool is not None else None
        try:
            command=[sys.executable,'-B','-m','avengine.timeline.v7_phase2_fresh_rir','--request',str(request)]
            if selected_cpu is not None:command=['taskset','--cpu-list',str(selected_cpu),*command]
            with(directory/'native.log').open('xb')as log:
                child=_launch_native_child(['nice','-n','10',*command],log,self.inputs)
                exit_code=child.wait()
            if exit_code:raise ValueError('fresh native process failed: '+str(directory/'native.log'))
            result=json.loads(report.read_text());wave=np.load(output,allow_pickle=False)
            if result['pid']!=child.pid or result['ppid']!=os.getpid() or result['pid']==os.getpid():
                raise ValueError('fresh RIR process lineage missing')
            if result['request_id']!=token or result['source_position_m']!=point.tolist() or result['native_identity']!=self.identity:
                raise ValueError('fresh RIR native scene/pose/configuration identity mismatch')
            if result['evidence_role']!=evidence_role or result['reference_trial']!=reference_trial:
                raise ValueError('fresh RIR primary/reference role changed')
            if result['output_sha256']!=hashlib.sha256(output.read_bytes()).hexdigest():
                raise ValueError('fresh native RIR bytes changed')
            if wave.ndim!=2 or wave.shape[0]!=4 or not np.isfinite(wave).all() or not np.any(wave):
                raise ValueError('fresh native RIR invalid')
            return wave,str(report),result['process_cpu_seconds_total']
        finally:
            if selected_cpu is not None:self.cpu_pool.put(selected_cpu)

    def render(self,positions,*,evidence_role='primary',reference_trial=None):
        started=time.perf_counter();positions=list(positions)
        if self.parallel_workers>1 and len(positions)>1:
            with ThreadPoolExecutor(max_workers=self.parallel_workers)as pool:
                values=list(pool.map(lambda p:self._render_one(p,evidence_role,reference_trial),positions))
        else:values=[self._render_one(p,evidence_role,reference_trial)for p in positions]
        waves,reports,cpu=zip(*values);self.process_evidence.extend(reports)
        # Native SDK tails have measured, position-dependent lengths. Keep
        # each complete raw buffer; scalar callers retain their prior shape.
        samples=np.stack(waves) if len({wave.shape for wave in waves})==1 else list(waves)
        return SimpleNamespace(samples=samples,wall_seconds=time.perf_counter()-started,
            process_cpu_seconds=sum(cpu),native_process_evidence=list(reports))

    def render_independent_reference(self,positions,trial):
        return self.render(positions,evidence_role='independent_midpoint_reference',reference_trial=int(trial))

def _worker(request_path):
    # Pin before SDK imports; threads and priority are inherited from CPU worker.
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpus[0]})
    path=pathlib.Path(request_path).resolve();request=json.loads(path.read_text())
    output=pathlib.Path(request['output']).resolve();report=pathlib.Path(request['report']).resolve()
    if output.parent!=path.parent or report.parent!=path.parent:raise ValueError('fresh process output escaped its own request directory')
    if request['evidence_role']not in{'primary','independent_midpoint_reference'}:raise ValueError('invalid native RIR evidence role')
    from avengine.timeline.v7_native_resource_slots import acquire_native_resource
    resource_lease=acquire_native_resource('rir')
    started=time.perf_counter();cpu=time.process_time()
    from avengine.acoustics.runtime import load_habitat_runtime
    load_habitat_runtime(runtime_mode='current-installed',**request['inputs']['runtime'])
    from tools.timeline.pilot_continuous_scenes import native_renderer
    renderer,identity=native_renderer(request['inputs'],request['house'])
    if identity!=request['expected_native_identity']:raise ValueError('fresh child native identity differs from declared estimator')
    wave=renderer.render([request['source_position_m']]).samples[0]
    check_write(output,expected_bytes=4_000_000)
    with output.open('xb')as f:np.save(f,wave,allow_pickle=False)
    value={'status':'pass','revision':REVISION,'pid':os.getpid(),'ppid':os.getppid(),
        'request_id':request['request_id'],'source_position_m':request['source_position_m'],
        'evidence_role':request['evidence_role'],'reference_trial':request['reference_trial'],
        'native_identity':identity,'native_setup_report':renderer.setup_report,
        'native_renders_after_fresh_upload':1,'primary_cache_read':False,'estimate_read':False,
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'nice':os.getpriority(os.PRIO_PROCESS,0),
        'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'process_cpu_seconds_total':time.process_time()-cpu,'wall_seconds_total':time.perf_counter()-started}
    if resource_lease is not None:value['task_resource_lease']=resource_lease.receipt
    # Optional scheduling runs in a separate coordinator. In particular, an
    # independent measuring process never reads any primary kernel or estimate.
    try:
        from avengine.timeline.v7_phase2_rir_prefetch import maybe_schedule_prefetch
        prefetched=maybe_schedule_prefetch(path,request,renderer.setup_report)
        if prefetched is not None:value['prefetch_receipt']=prefetched
    except Exception:
        import traceback
        (path.parent/'optional_prefetch_error.log').write_text(traceback.format_exc())
    import resource
    children=resource.getrusage(resource.RUSAGE_CHILDREN)
    value.update(process_cpu_seconds_self=time.process_time()-cpu,
        descendant_cpu_seconds=children.ru_utime+children.ru_stime,
        process_cpu_seconds_total=time.process_time()-cpu+children.ru_utime+children.ru_stime,
        wall_seconds_total=time.perf_counter()-started)
    with report.open('x')as f:json.dump(value,f,allow_nan=False)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
    _worker(parser.parse_args().request)

"""Optional task-local queues for native compute and dense motion capture.

A lease only delays a new native operation. It never terminates or changes an
operation, native sampling, material, waveform, or engineering comparison.
"""
from __future__ import annotations
import atexit,fcntl,json,os,time
from pathlib import Path

class NativeResourceLease:
    def __init__(self,stream,kind,index,cpu=None):
        self.stream=stream;self.receipt={'kind':kind,'slot':index,'cpu_id':cpu,'pid':os.getpid()}
        atexit.register(self.close)
    def close(self):
        if self.stream is not None:
            self.stream.seek(0);self.stream.truncate();self.stream.flush()
            fcntl.flock(self.stream,fcntl.LOCK_UN);self.stream.close();self.stream=None

def acquire_native_resource(kind,expected_incremental_bytes=0):
    from avengine.timeline.v7_storage_budget import ENV_KEY
    raw=os.environ.get(ENV_KEY)
    if not raw:return None
    root=Path(json.loads(raw)['root']).resolve();policy_path=root/'inputs/native_resource_policy.json'
    if not policy_path.is_file():return None
    policy=json.loads(policy_path.read_text());section=policy.get(kind)
    if not section:return None
    import psutil
    directory=root/'resource_slots'/kind;directory.mkdir(parents=True,exist_ok=True)
    count=int(section['slots']);cpus=section.get('cpu_ids',[])
    required_increment=max(int(expected_incremental_bytes),int(section.get('expected_incremental_bytes',0)))
    if count<1 or (cpus and len(cpus)!=count):raise ValueError('invalid task native resource slots')
    while True:
        reserved=0
        for relative in section.get('legacy_active_jobs',[]):
            legacy=relative if isinstance(relative,dict)else {'path':relative}
            file=root/legacy['path']
            if not file.is_file():continue
            job=json.loads(file.read_text())
            if job.get('status')!='running':continue
            try:
                process=psutil.Process(job['pid'])
                if process.status()!=psutil.STATUS_ZOMBIE and 'run_v7_acceptance.py' in ' '.join(process.cmdline()) and (not legacy.get('pid') or (process.pid==legacy['pid'] and process.create_time()==legacy['create_time'])):reserved+=1
            except psutil.Error:pass
        available=psutil.virtual_memory().available
        tree_pss=0
        try:
            owners=[p for p in psutil.Process().parents() if any(name in ' '.join(p.cmdline())
                for name in ('run_v7_acceptance.py','generate_v7_demand_scenes.py'))]
            if owners:
                owner=owners[-1]
                tree_pss=sum(p.memory_full_info().pss for p in [owner,*owner.children(recursive=True)] if p.status()!=psutil.STATUS_ZOMBIE)
        except psutil.Error:pass
        pressure=available<20_000_000_000 or tree_pss>int(policy.get('maximum_task_pss_bytes',40_000_000_000))
        if pressure:
            alert={'time_unix':time.time(),'kind':kind,'pid':os.getpid(),'available_memory_bytes':available,'own_parent_tree_pss_bytes':tree_pss,'action':'pause new native operation; never terminate active work'}
            with(root/'evidence/native_resource_alerts.jsonl').open('a')as stream:stream.write(json.dumps(alert)+'\n')
        if not pressure and available>=int(section.get('minimum_available_bytes',30_000_000_000))+required_increment:
            for index in range(min(reserved,count),count):
                stream=(directory/f'{index:02d}.lock').open('a+')
                try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:stream.close();continue
                with(directory.parent/'memory_admission.lock').open('a+')as guard:
                    fcntl.flock(guard,fcntl.LOCK_EX)
                    pending=0
                    for existing in directory.parent.glob('*/*.lock'):
                        if existing==Path(stream.name):continue
                        try:
                            # Old processes may leave metadata after unlocking.
                            # Only a currently held lease reserves future RAM.
                            with existing.open('r')as probe:
                                try:fcntl.flock(probe,fcntl.LOCK_SH|fcntl.LOCK_NB)
                                except BlockingIOError:pass
                                else:
                                    fcntl.flock(probe,fcntl.LOCK_UN);continue
                            record=json.loads(existing.read_text());owner=psutil.Process(record['pid'])
                            if owner.create_time()==record['create_time']:
                                pending+=max(0,record['reserved_incremental_bytes']-max(0,owner.memory_full_info().pss-record['baseline_pss_bytes']))
                        except (ValueError,KeyError,psutil.Error):pass
                    task_pss=0
                    if section.get('maximum_task_pss_bytes'):
                        for process in psutil.process_iter():
                            try:
                                marker=json.loads(process.environ().get('AVENGINE_V7_STORAGE_BUDGET','{}'))
                                if marker.get('root')==str(root):task_pss+=process.memory_full_info().pss
                            except (ValueError,psutil.Error):pass
                    task_stop=bool(section.get('maximum_task_pss_bytes')and task_pss+pending+required_increment>int(section['maximum_task_pss_bytes']))
                    system_pss=0
                    if section.get('maximum_system_pss_bytes'):
                        for process in psutil.process_iter():
                            try:system_pss+=process.memory_full_info().pss
                            except psutil.Error:pass
                    pss_stop=(tree_pss+pending+required_increment>int(policy.get('maximum_task_pss_bytes',40_000_000_000)) or bool(section.get('maximum_system_pss_bytes') and system_pss+pending+required_increment>int(section['maximum_system_pss_bytes'])))
                    if task_stop or pss_stop or psutil.virtual_memory().available-pending<int(section.get('minimum_available_bytes',30_000_000_000))+required_increment:
                        fcntl.flock(stream,fcntl.LOCK_UN);stream.close();continue
                    stream.seek(0);stream.truncate();stream.write(json.dumps({'pid':os.getpid(),'create_time':psutil.Process().create_time(),'reserved_incremental_bytes':required_increment,'baseline_pss_bytes':psutil.Process().memory_full_info().pss}));stream.flush()
                cpu=int(cpus[index]) if cpus else None
                if cpu is not None:os.sched_setaffinity(0,{cpu})
                return NativeResourceLease(stream,kind,index,cpu)
        time.sleep(.5)

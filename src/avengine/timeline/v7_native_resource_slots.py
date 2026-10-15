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
            fcntl.flock(self.stream,fcntl.LOCK_UN);self.stream.close();self.stream=None

def acquire_native_resource(kind):
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
    if not 1<=count<=16 or (cpus and len(cpus)!=count):raise ValueError('invalid task native resource slots')
    while True:
        reserved=0
        for relative in section.get('legacy_active_jobs',[]):
            file=root/relative
            if not file.is_file():continue
            job=json.loads(file.read_text())
            if job.get('status')!='running':continue
            try:
                process=psutil.Process(job['pid'])
                if process.status()!=psutil.STATUS_ZOMBIE and 'run_v7_acceptance.py' in ' '.join(process.cmdline()):reserved+=1
            except psutil.Error:pass
        if psutil.virtual_memory().available>=int(section.get('minimum_available_bytes',30_000_000_000)):
            for index in range(min(reserved,count),count):
                stream=(directory/f'{index:02d}.lock').open('a+')
                try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:stream.close();continue
                cpu=int(cpus[index]) if cpus else None
                if cpu is not None:os.sched_setaffinity(0,{cpu})
                return NativeResourceLease(stream,kind,index,cpu)
        time.sleep(.5)

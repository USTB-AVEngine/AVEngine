"""Production admission policy. Pressure pauses new work; no process is killed."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,os,shutil,time

# Temporary exceptions are supplied by the authorized task configuration.
TEMPORARY_DISK_EXPIRY=0.

@dataclass(frozen=True)
class ResourcePolicy:
    maximum_cpu_processes:int=16
    maximum_task_pss_bytes:int=40_000_000_000
    minimum_available_bytes:int=30_000_000_000
    minimum_free_bytes:int=100_000_000_000
    temporary_minimum_free_bytes:int=40_000_000_000
    temporary_minimum_free_until_unix:float=TEMPORARY_DISK_EXPIRY
    fresh_rir_workers:int=8

    def __post_init__(self):
        if not 2<=self.maximum_cpu_processes<=16:raise ValueError('owner CPU process budget is 2..16 including the native parent')
        if self.maximum_task_pss_bytes<=0 or self.minimum_available_bytes<30_000_000_000:
            raise ValueError('invalid task PSS or available-memory contract')
        if self.minimum_free_bytes<100_000_000_000 or self.temporary_minimum_free_bytes<40_000_000_000:
            raise ValueError('disk floor may not undercut the owner contract')
        if not 1<=self.fresh_rir_workers<=16:raise ValueError('independent RIR workers must leave a parent CPU slot')

    def disk_floor(self,now=None):
        return self.temporary_minimum_free_bytes if (time.time()if now is None else now)<self.temporary_minimum_free_until_unix else self.minimum_free_bytes


def launch_disk_policy(policy):
    import datetime
    if not policy.temporary_minimum_free_until_unix:return {}
    return {'minimum_free_bytes':policy.temporary_minimum_free_bytes,
        'expires_utc':datetime.datetime.fromtimestamp(policy.temporary_minimum_free_until_unix,datetime.timezone.utc).isoformat(),
        'after_expiry_minimum_free_bytes':policy.minimum_free_bytes}


def admission(policy,*,filesystem='/data',available=None,free=None,own_pss=0,now=None):
    if available is None:available=int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')))*1024
    if free is None:free=shutil.disk_usage(filesystem).free
    allowed=available>=policy.minimum_available_bytes and free>=policy.disk_floor(now) and own_pss<=policy.maximum_task_pss_bytes
    return {'status':'admitted'if allowed else'paused_before_new_job','available_memory_bytes':available,
        'disk_free_bytes':free,'required_disk_free_bytes':policy.disk_floor(now),'own_pss_bytes':own_pss,
        'action':'launch only after admission; do not stop any running process'}


def idle_cpu_order(allowed=None,interval=.15):
    """Pick inside the caller's allowed set using current read-only CPU counters."""
    allowed=sorted(os.sched_getaffinity(0))if allowed is None else sorted(allowed)
    def read():
        out={}
        for line in Path('/proc/stat').read_text().splitlines():
            key,*parts=line.split()
            if key.startswith('cpu')and key[3:].isdigit():
                values=list(map(int,parts));out[int(key[3:])]=(sum(values[:8]),values[3]+values[4])
        return out
    first=read();time.sleep(interval);last=read()
    def busy(cpu):
        total=last[cpu][0]-first[cpu][0];idle=last[cpu][1]-first[cpu][1]
        return (1-idle/max(1,total),cpu)
    return sorted(allowed,key=busy)


def runtime_resource_view(runtime,policy,cpus):
    if len(cpus)<2:raise ValueError('need one parent core and an independent fresh RIR core')
    result=dict(runtime);count=min(policy.fresh_rir_workers,len(cpus)-1,policy.maximum_cpu_processes-1)
    result.update(fresh_rir_workers=count,fresh_rir_cpu_ids=list(cpus[1:1+count]))
    return result


def configure_worker(root,policy,cpus,*,maximum_task_bytes=60_000_000_000):
    from avengine.timeline.v7_storage_budget import ENV_KEY,check_write
    root=Path(root).resolve()
    budget={'root':str(root),'filesystem':'/data','maximum_task_bytes':maximum_task_bytes,
        'minimum_free_bytes':policy.minimum_free_bytes,'temporary_minimum_free_bytes':policy.temporary_minimum_free_bytes,
        'temporary_minimum_free_until_unix':policy.temporary_minimum_free_until_unix,
        'concurrent_write_margin_bytes':0,'launch_disk_policy':launch_disk_policy(policy),'policy':'unattended task-owned CPU-only construction'}
    os.environ[ENV_KEY]=json.dumps(budget);check_write(root,expected_bytes=0)
    budget['admission_only']=True;os.environ[ENV_KEY]=json.dumps(budget)
    directory=root/'inputs';directory.mkdir(parents=True,exist_ok=True)
    # Resource slots are shared by new native subprocesses in this task root.
    sections={'parent':{'slots':1,'cpu_ids':[cpus[0]],'expected_incremental_bytes':6_000_000_000},
        'rir':{'slots':min(len(cpus)-1,policy.maximum_cpu_processes-1),'cpu_ids':list(cpus[1:policy.maximum_cpu_processes]),'expected_incremental_bytes':4_000_000_000}}
    for section in sections.values():section['minimum_available_bytes']=policy.minimum_available_bytes
    path=directory/'native_resource_policy.json'
    value={**sections,'maximum_task_pss_bytes':policy.maximum_task_pss_bytes,
        'provenance':'resource leases reused from motion helper c0f39d60; scheduling only',
        'policy':asdict(policy)}
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
    (root/'evidence').mkdir(exist_ok=True)
    return budget

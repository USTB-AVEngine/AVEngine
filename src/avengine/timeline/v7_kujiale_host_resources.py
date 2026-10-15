"""Host-side task PSS, visible consistently to isolated UE device namespaces."""
import datetime,json,os,shutil,time
from pathlib import Path
from avengine.capture.kujiale_gpu_queue import owned_processes
from avengine.timeline.v7_kujiale_manifest import atomic_write

def sample_host(root):
    root=Path(root);rows=owned_processes(root);value={'sample_unix':time.time(),'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sampler_pid':os.getpid(),'PSS_bytes':sum(r['pss_bytes']for r in rows),'MemAvailable_bytes':next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')),'data_free_bytes':shutil.disk_usage(root).free,'processes':rows,'scope':'host procfs; exact own command roots and all descendants, not sibling-namespace smaps'};atomic_write(root/'evidence/host_resource_sample.json',value);return value

def read_host(root,max_age_s=30):
    path=Path(root)/'evidence/host_resource_sample.json'
    try:value=json.loads(path.read_text())
    except (OSError,ValueError):return None
    return value if 0<=time.time()-value['sample_unix']<=max_age_s else None

def subtree_pss(sample,pid):
    records={r['pid']:r for r in sample['processes']};selected={pid}
    while True:
        descendants={p for p,r in records.items()if r['ppid']in selected}
        if descendants<=selected:break
        selected|=descendants
    return sum(records[p]['pss_bytes']for p in selected if p in records)

class LegacyAdmissionBridge:
    """Pause only unclaimed CPU leases while old isolated workers drain."""
    def __init__(self,root):self.root=Path(root);self.held={}
    def update(self,sample):
        import fcntl
        from avengine.timeline.v7_kujiale_cpu_phases import legacy_workers
        directory=self.root/'evidence/cpu_slots';directory.mkdir(parents=True,exist_ok=True)
        with(directory/'queue.lock').open('a+')as guard:
            fcntl.flock(guard,fcntl.LOCK_EX);free={};busy=[]
            for index in range(16):
                if index in self.held:continue
                stream=(directory/f'slot_{index}.json').open('a+')
                try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.seek(0)
                    try:busy.append(json.load(stream))
                    except ValueError:busy.append({'pid':-1,'stage':'unknown','estimate_bytes':6000000000})
                    stream.close()
                else:free[index]=stream
            legacy=legacy_workers(self.root,{r['pid']for r in busy});reserved=0
            for row in busy:
                stage=row.get('stage','');expected=max(row.get('estimate_bytes',0),6000000000 if 'native'in stage else 4000000000 if stage=='construct_room'else 2000000000)
                reserved+=max(0,expected-subtree_pss(sample,row['pid']))
            for pid in legacy:reserved+=max(0,2000000000-subtree_pss(sample,pid))
            headroom=40000000000-sample['PSS_bytes']-reserved;allow=min(1,max(0,int(headroom//6000000000)))
            target=(16-len(busy))if allow==0 else max(0,16-len(busy)-len(legacy)-allow)
            target=min(target,len(self.held)+len(free))
            while len(self.held)>target:
                index,stream=self.held.popitem();fcntl.flock(stream,fcntl.LOCK_UN);stream.close()
            for index,stream in list(free.items()):
                if len(self.held)<target:
                    stream.seek(0);stream.truncate();json.dump({'pid':os.getpid(),'stage':'host_resource_reservation','estimate_bytes':0,'pause_new_phase_only':True},stream);stream.flush();self.held[index]=stream
                else:fcntl.flock(stream,fcntl.LOCK_UN);stream.close()
            value={'time_utc':sample['time_utc'],'PSS_bytes':sample['PSS_bytes'],'pending_memory_reservations':reserved,'legacy_unleased_workers':legacy,'new_phase_allowance':allow,'unclaimed_CPU_slots_paused':len(self.held),'running_workers_stopped':False};atomic_write(self.root/'evidence/legacy_admission_bridge.json',value)

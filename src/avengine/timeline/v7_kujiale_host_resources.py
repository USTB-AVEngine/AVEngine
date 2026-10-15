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

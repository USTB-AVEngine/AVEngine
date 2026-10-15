"""Per-UUID native launch leases; resource pressure only queues new instances."""
from __future__ import annotations
import csv,datetime,fcntl,json,os,shutil,subprocess,time
from pathlib import Path
from avengine.timeline.v7_storage_budget import task_disk_policy

def live(record):
    try:
        text=(Path('/proc')/str(record['pid'])/'stat').read_text();fields=text[text.rfind(')')+2:].split()
        return fields[0]!='Z' and fields[19]==record['start_ticks']
    except (OSError,KeyError):return False

def own_pss(root):
    records={};owned=set()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():continue
        try:
            text=(entry/'stat').read_text();fields=text[text.rfind(')')+2:].split();pid=int(entry.name);records[pid]=(int(fields[1]),entry)
            if str(root).encode()in(entry/'cmdline').read_bytes():owned.add(pid)
        except (OSError,ValueError):continue
    while True:
        children={pid for pid,(parent,_)in records.items()if parent in owned}
        if children<=owned:break
        owned|=children
    total=0
    for pid in owned:
        try:total+=sum(int(line.split()[1])*1024 for line in(records[pid][1]/'smaps_rollup').read_text().splitlines()if line.startswith('Pss:'))
        except OSError:continue
    return total

def launch_reserved_gpu(output,gpu,arguments,*,executable=None):
    from avengine.capture.kujiale_native_gpu_policy import check_gpu_policy
    output=Path(output).resolve();policy=check_gpu_policy(output,gpu)
    if policy is None:raise ValueError('native queue requires explicit task GPU authorization')
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('native queue requires nice>=10')
    executable=executable or policy['native_unreal_editor']
    uuid=policy['gpu_uuids'][str(gpu)];directory=output/'native_gpu_queue';directory.mkdir(exist_ok=True);last=0
    while True:
        disk=task_disk_policy(output,100000000000)
        if shutil.disk_usage(output).free>=disk['minimum_free_bytes']:break
        if time.time()-last>60:print('PAUSE_NEW_GPU_DISK',disk,flush=True);last=time.time()
        time.sleep(5)
    admitted=datetime.datetime.now(datetime.timezone.utc).isoformat()
    while True:
        with(directory/'queue.lock').open('a+')as guard:
            fcntl.flock(guard,fcntl.LOCK_EX);leases=[];reserved=[]
            for index in range(4):
                stream=(directory/f'gpu{gpu}_slot{index}.json').open('a+')
                try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.seek(0)
                    try:reserved.append(json.load(stream)['pid'])
                    except (ValueError,KeyError):reserved.append(-1)
                    stream.close();continue
                stream.seek(0)
                try:prior=json.load(stream)
                except ValueError:prior={}
                if live(prior):reserved.append(prior['pid']);stream.close()
                else:leases.append((index,stream))
            rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True).splitlines()))
            actual={int(r[1])for r in rows if len(r)>=3 and r[0].strip()==uuid and 'UnrealEditor'in r[2]}
            count=len(actual|{p for p in reserved if p>0})+reserved.count(-1)
            mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:'));pss=own_pss(output)
            if leases and count<4 and mem>=30000000000 and pss+int(policy.get('minimum_new_ue_pss_headroom_bytes',0))<=40000000000:
                index,lease=leases.pop(0)
                for _,other in leases:other.close()
                text=(Path('/proc')/str(os.getpid())/'stat').read_text();fields=text[text.rfind(')')+2:].split();cpu=(gpu-1)*4+index
                receipt={'pid':os.getpid(),'start_ticks':fields[19],'physical_gpu':gpu,'gpu_uuid':uuid,'gpu_slot':index,'cpu_affinity':[cpu],'disk_admitted_utc':admitted,'minimum_disk_free_bytes_at_admission':disk['minimum_free_bytes'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'MemAvailable_bytes':mem,'PSS_bytes':pss,'foreign_and_existing_UE_pids':sorted(actual),'arguments':arguments}
                lease.seek(0);lease.truncate();json.dump(receipt,lease);lease.flush();os.fsync(lease.fileno());os.set_inheritable(lease.fileno(),True);(directory/f'launch_{os.getpid()}.json').write_text(json.dumps(receipt,indent=2)+'\n');os.sched_setaffinity(0,{cpu});break
            for _,stream in leases:stream.close()
        if time.time()-last>60:print('PAUSE_NEW_GPU_CAPACITY',{'gpu':gpu,'total_UE':count,'MemAvailable':mem,'PSS':pss},flush=True);last=time.time()
        time.sleep(5)
    os.environ['CUDA_VISIBLE_DEVICES']=uuid
    os.execv(executable,[executable,*arguments])

"""Task-wide CPU phase leases for native preparation and question production.

Old live workers are counted until they finish. Resource pressure queues new
phases; it never stops an active native or CPU worker.
"""
from __future__ import annotations
from contextlib import contextmanager
import datetime,fcntl,json,os,time
from pathlib import Path
from avengine.capture.kujiale_gpu_queue import own_pss,live


def phase_state(root,state,stage):
    directory=Path(root)/'evidence/cpu_phases';directory.mkdir(parents=True,exist_ok=True);path=directory/(str(os.getpid())+'.json');value={'pid':os.getpid(),'state':state,'stage':stage,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value)+'\n');temporary.replace(path)


def legacy_workers(root,leased):
    root=Path(root);result=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal()or int(entry.name)in leased:continue
        try:
            command=(entry/'cmdline').read_bytes()
            if 'python'not in Path(os.fsdecode(command.split(b'\0')[0])).name:continue
            if b'/tools/timeline/run_v7_acceptance.py'not in command or str(root/'inputs/GAP_PLAN.json').encode()not in command:continue
            state=root/'evidence/cpu_phases'/(entry.name+'.json')
            if state.exists()and json.loads(state.read_text())['state']in {'waiting','idle'}:continue
            stat=(entry/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
            if fields[0]!='Z':result.append(int(entry.name))
        except (OSError,ValueError,KeyError):continue
    return result


def phase_fits(actual_pss,remaining_reserved,new_estimate,leased_count,legacy_count,mem_available,*,max_cpu=16,pss_limit=40000000000):
    return leased_count+legacy_count<max_cpu and actual_pss+remaining_reserved+new_estimate<=pss_limit and mem_available>=30000000000


def phase_memory_estimate(stage,declared):
    # Current-native category qualification includes the retained full room
    # geometry and a renderer: host samples observed 8.39GB for Python alone.
    return max(declared,12000000000 if stage=='new_native_category'else 6000000000 if 'native'in stage else 4000000000 if stage=='construct_room'else 0)


@contextmanager
def cpu_phase(root,stage,*,estimate_bytes=4000000000):
    estimate_bytes=phase_memory_estimate(stage,estimate_bytes)
    root=Path(root);directory=root/'evidence/cpu_slots';directory.mkdir(parents=True,exist_ok=True);phase_state(root,'waiting',stage);last=0;prior=set(os.sched_getaffinity(0))
    while True:
        with(directory/'queue.lock').open('a+')as guard:
            fcntl.flock(guard,fcntl.LOCK_EX);available=[];busy=[]
            for index in range(16):
                lease=(directory/f'slot_{index}.json').open('a+')
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.seek(0)
                    try:busy.append(json.load(lease))
                    except ValueError:busy.append({'pid':-1,'estimate_bytes':estimate_bytes})
                    lease.close();continue
                available.append((index,lease))
            from avengine.timeline.v7_kujiale_host_resources import read_host,subtree_pss
            host=read_host(root);leased={r['pid']for r in busy};legacy=legacy_workers(root,leased);reserved=0
            if host is not None:
                for row in busy:reserved+=max(0,row.get('estimate_bytes',estimate_bytes)-subtree_pss(host,row['pid']))
                for pid in legacy:reserved+=max(0,2000000000-subtree_pss(host,pid))
            actual=host['PSS_bytes']if host else 40000000001;mem=host['MemAvailable_bytes']if host else 0
            if available and phase_fits(actual,reserved,estimate_bytes,len(busy),len(legacy),mem):
                preferred=range(12,16)if stage=='audio_questions_reader'else range(16)
                rank={value:i for i,value in enumerate(preferred)};available.sort(key=lambda row:rank.get(row[0],100+row[0]))
                index,lease=available.pop(0)
                for _,other in available:other.close()
                receipt={'pid':os.getpid(),'stage':stage,'estimate_bytes':estimate_bytes,'CPU_index':index,'PSS_at_admission':actual,'pending_memory_reservations':reserved,'legacy_workers':legacy,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};lease.seek(0);lease.truncate();json.dump(receipt,lease);lease.flush();os.sched_setaffinity(0,{index});phase_state(root,'active',stage);break
            for _,lease in available:lease.close()
        if time.time()-last>=60:print('QUEUE_NEW_CPU_PHASE',stage,{'PSS':actual,'reserved':reserved,'leased':len(busy),'legacy':len(legacy)},flush=True);last=time.time()
        time.sleep(5)
    try:yield receipt
    finally:
        phase_state(root,'idle',stage);fcntl.flock(lease,fcntl.LOCK_UN);lease.close();os.sched_setaffinity(0,prior)

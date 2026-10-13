"""Exclusive single CPU assignments within an explicitly recorded task pool."""
from pathlib import Path
import fcntl,json,os

def process_identity(pid):
    try:
        p=Path('/proc',str(pid))
        return {'pid':pid,'start_ticks':int(p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]),
                'argv':p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')}
    except (OSError,IndexError,ValueError):return None

def pin_registered_cpu_task(root):
    root=Path(root);policy=root/'inputs/cpu_assignment_pool.json'
    if not policy.is_file():
        allowed=sorted(os.sched_getaffinity(0));cpu=allowed[os.getpid()%len(allowed)]
        os.sched_setaffinity(0,{cpu});return cpu
    pool=json.loads(policy.read_text())['cpu_ids']
    if not pool or len(pool)>28:raise ValueError('invalid explicit task CPU pool')
    me=process_identity(os.getpid())
    if str(root)not in me['argv']:raise ValueError('CPU assignment requires explicit task-root command')
    with(root/'cpu_assignment.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'cpu_assignments.json';saved=json.loads(path.read_text())if path.exists()else []
        active=[]
        for entry in saved:
            actual=process_identity(entry['pid'])
            if actual and actual['start_ticks']==entry['start_ticks']and str(root)in actual['argv']:active.append(entry)
        mine=next((e for e in active if e['pid']==me['pid']),None)
        free=[cpu for cpu in pool if cpu not in {e['cpu_id']for e in active}]
        if mine is None and not free:raise ValueError('no free single CPU task slot')
        cpu=mine['cpu_id']if mine else free[0]
        os.sched_setaffinity(0,{cpu})
        if mine is None:active.append({**me,'cpu_id':cpu})
        tmp=path.with_suffix('.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(active,indent=2)+'\n');os.replace(tmp,path)
        return cpu

"""CPU visual threads stay inside the worker's admitted CPU envelope."""
import json,os
ENV='AVENGINE_V7_CPU_VISUAL_ENVELOPE'


def contain_own_threads(cpu_ids):
    from pathlib import Path
    allowed=set(int(cpu)for cpu in cpu_ids);changed=[]
    if not allowed:raise ValueError('native thread CPU envelope is empty')
    for task in Path('/proc/self/task').iterdir():
        if not task.name.isdigit():continue
        tid=int(task.name)
        try:
            current=set(os.sched_getaffinity(tid))
            if current-allowed:
                target=current&allowed or allowed;os.sched_setaffinity(tid,target);changed.append(tid)
            if os.getpriority(os.PRIO_PROCESS,tid)<10:os.setpriority(os.PRIO_PROCESS,tid,10)
        except ProcessLookupError:continue
    return {'allowed_cpu_ids':sorted(allowed),'threads_rebound':len(changed),'thread_ids_rebound':changed}


def configure_cpu_visual(inputs,*,cpu_ids=None):
    requested=int(inputs.get('cpu_visual_threads',os.environ.get('LP_NUM_THREADS','1')))
    if cpu_ids is not None:
        cpus=list(cpu_ids);os.environ[ENV]=json.dumps({'owner_pid':os.getpid(),'cpu_ids':cpus})
    else:
        cpus=sorted(os.sched_getaffinity(0));raw=os.environ.get(ENV)
        if raw:
            import psutil
            value=json.loads(raw);ancestors={p.pid for p in psutil.Process().parents()}|{os.getpid()}
            if value.get('owner_pid')in ancestors:cpus=list(value['cpu_ids'])
    count=min(4,max(1,requested),len(cpus))
    os.environ['LP_NUM_THREADS']=str(count)
    if count>1:os.sched_setaffinity(0,set(cpus))
    containment=contain_own_threads(cpus if count>1 else os.sched_getaffinity(0))
    return {'backend':'cpu_llvmpipe','LP_NUM_THREADS':count,'cpu_ids':cpus,
            'thread_containment':containment,'fresh_RIR_workers_each_one_core':True,'resolution_or_FOV_changed':False}

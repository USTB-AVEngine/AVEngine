"""Move own CPU-only phases onto cores not currently occupied by own UE."""
from pathlib import Path
import json,os
from avengine.timeline.v7_kujiale_host_resources import read_host
from avengine.timeline.v7_kujiale_manifest import atomic_write


def assign_cpu_phases(pids,renderer_cpus,allowed=tuple(range(16))):
    available=[c for c in allowed if c not in renderer_cpus]or list(allowed)
    return {pid:available[i%len(available)]for i,pid in enumerate(sorted(pids))}


def rebalance(root):
    root=Path(root);sample=read_host(root)
    if sample is None:return {'status':'waiting_host_sample','changes':[]}
    records={r['pid']:r for r in sample['processes']};renderers={p for p,r in records.items()if '/UnrealEditor 'in r['command']and str(root)in r['command']};renderer_cores=set()
    for pid in renderers:
        try:renderer_cores.update(os.sched_getaffinity(pid))
        except OSError:pass
    descendants=set(renderers)
    while True:
        parents={records[p]['ppid']for p in descendants if p in records}
        if parents<=descendants:break
        descendants|=parents
    candidates=[]
    for pid,r in records.items():
        argv=r['command']
        if pid in descendants or '/tools/timeline/run_v7_acceptance.py'not in argv or str(root/'inputs/GAP_PLAN.json')not in argv or 'python'not in Path(argv.split()[0]).name:continue
        state=root/'evidence/cpu_phases'/(str(pid)+'.json')
        if state.exists():
            try:
                phase=json.loads(state.read_text())
                if phase['state']in {'waiting','idle'}or phase['stage'].startswith('native_'):continue
            except (OSError,ValueError):continue
        if r['state']!='Z':candidates.append(pid)
    allocation=assign_cpu_phases(candidates,renderer_cores);changes=[]
    for pid,cpu in allocation.items():
        try:
            text=(Path('/proc')/str(pid)/'stat').read_text();fields=text[text.rfind(')')+2:].split()
            if fields[19]!=records[pid]['start_ticks']:continue
            before=sorted(os.sched_getaffinity(pid))
            if before!=[cpu]:
                os.sched_setaffinity(pid,{cpu});changes.append({'pid':pid,'start_ticks':fields[19],'previous_CPUs':before,'assigned_CPU':cpu,'command':records[pid]['command']})
        except OSError:continue
    result={'status':'sampled_own_phases','time_utc':sample['time_utc'],'own_UE_renderer_CPUs':sorted(renderer_cores),'CPU_only_phase_allocation':allocation,'changes':changes,'no_processes_stopped':True};atomic_write(root/'evidence/cpu_balance_current.json',result);return result

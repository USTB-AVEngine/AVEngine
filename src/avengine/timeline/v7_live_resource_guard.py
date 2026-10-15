"""Committed task-scoped live resource observations and admission decisions.

Peers are read-only observations. A decision only delays this task's next
parent/SDK operation; this module has no process signalling or deletion code.
"""
from __future__ import annotations
from pathlib import Path
import datetime,json,os,shutil


def decide(snapshot,policy):
    own_limit=int(policy.get('maximum_pss_bytes',150_000_000_000))
    combined_limit=int(policy.get('maximum_combined_pss_bytes',190_000_000_000))
    memory_floor=int(policy.get('minimum_available_bytes',30_000_000_000))
    reserve=int(policy.get('native_process_reserve_bytes',1_500_000_000))
    reasons=[]
    if snapshot['MemAvailable_bytes']<memory_floor:reasons.append('available_memory_below_floor')
    if snapshot['own_pss_bytes']+reserve>own_limit:reasons.append('own_PSS_reservation')
    if snapshot['combined_pss_bytes']+reserve>combined_limit:reasons.append('combined_PSS_reservation')
    own_over=[r for r in snapshot['ranked_parent_trees']if r['owned']and r['PSS_bytes']>r['limit_bytes']]
    if own_over:reasons.append('owned_parent_tree_over_limit')
    if snapshot['processes']>=int(policy.get('maximum_cpu_processes',80)):reasons.append('CPU_process_budget')
    disk_pause=snapshot['data_free_bytes']<snapshot['minimum_disk_free_bytes']
    return {'pause_new_native_operations':bool(reasons),'pause_new_parents':bool(reasons)or disk_pause,
        'reasons':reasons+(['disk_floor_for_new_parents']if disk_pause else[]),
        'critical_below_20GB':snapshot['MemAvailable_bytes']<20_000_000_000,
        'blocked_owned_scene_ids':[r['scene_id']for r in own_over],
        'maximum_native_processes':0 if reasons else min(16,int(policy.get('fresh_rir_workers',16))),
        'active_processes_signalled':False}


def observe(manifest):
    from tools.timeline.monitor_v7_acceptance import output_processes,processes,proc
    from avengine.timeline.v7_storage_budget import acceptance_minimum_free_bytes
    output=Path(manifest['output_root']).resolve();peers=[Path(p).resolve()for p in manifest.get('peer_output_roots',[])]
    all_records=output_processes([output,*peers]);own_records=output_processes([output]);trees=[];groups={}
    for root in [output,*peers]:
        roots=[]
        for path in(root/'evidence/jobs').glob('[0-9][0-9][0-9].json'):
            try:job=json.loads(path.read_text());pid=job.get('pid');live=proc(pid)if pid else None
            except(OSError,ValueError):continue
            if job.get('status')!='running'or not live or str(root)not in live['cmd']:continue
            tree=processes([pid]);roots.append(pid)
            trees.append({'task':root.name,'owned':root==output,'scene_id':job['scene_id'],'root_pid':pid,
                'PSS_bytes':sum(p['pss_bytes']for p in tree),'limit_bytes':int(job.get('maximum_parent_tree_pss_bytes',manifest.get('maximum_parent_tree_pss_bytes',30_000_000_000))),
                'root_start_ticks':live['start_ticks'],'cpu_including_reaped_children_s':sum(p['cpu_s']for p in tree),
                'log_mtime':(root/'logs'/f"parent_{job['slot']:03d}.log").stat().st_mtime if(root/'logs'/f"parent_{job['slot']:03d}.log").exists()else None,
                'pipe_write_pids':[p['pid']for p in tree if'pipe_write'in p['wchan']],'cmd':live['cmd']})
        records=output_processes([root]);groups[root.name]={'roots':roots,'pss_bytes':sum(p['pss_bytes']for p in records),
            'processes':len(records),'pipe_writers':[p['pid']for p in records if'pipe_write'in p['wchan']]}
    now=datetime.datetime.now(datetime.timezone.utc)
    result={'time_utc':now.isoformat(),'groups':groups,'own_pss_bytes':sum(p['pss_bytes']for p in own_records),
        'combined_pss_bytes':sum(p['pss_bytes']for p in all_records),'processes':len(all_records),
        'MemAvailable_bytes':next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')),
        'data_free_bytes':shutil.disk_usage(manifest.get('resource_filesystem','/data')).free,
        'minimum_disk_free_bytes':acceptance_minimum_free_bytes(now),
        'ranked_parent_trees':sorted(trees,key=lambda p:p['PSS_bytes'],reverse=True),'active_processes_signalled':False}
    decision=decide(result,manifest);result.update(decision);result['paused_new_jobs']=decision['pause_new_parents'];result['pss_bytes']=result['own_pss_bytes']
    result['parent_tree_over_limit']=[r for r in trees if r['PSS_bytes']>r['limit_bytes']]
    return result


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def publish(manifest,snapshot):
    import fcntl
    output=Path(manifest['output_root']).resolve()
    save(output/'evidence/GLOBAL_RESOURCE_CURRENT.json',snapshot);save(output/'evidence/RESOURCE_ALERT_CURRENT.json',snapshot)
    policy_path=output/'inputs/RIR_SHARED_POLICY.json'
    if policy_path.exists():
        with(output/'inputs/rir_policy_update.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);policy=json.loads(policy_path.read_text())
            policy.update(maximum_native_processes=snapshot['maximum_native_processes'],
                maximum_pss_bytes=int(manifest.get('maximum_pss_bytes',150_000_000_000)),
                maximum_combined_pss_bytes=int(manifest.get('maximum_combined_pss_bytes',190_000_000_000)),
                maximum_parent_tree_pss_bytes=int(manifest.get('maximum_parent_tree_pss_bytes',30_000_000_000)),
                blocked_parent_scene_ids=snapshot['blocked_owned_scene_ids'],ranked_alert=str(output/'evidence/RESOURCE_ALERT_CURRENT.json'))
            save(policy_path,policy)

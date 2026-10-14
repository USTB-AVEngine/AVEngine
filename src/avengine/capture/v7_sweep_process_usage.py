"""Identify live sweep roots across independent and retired controllers."""
from pathlib import Path
import json


def tracked_sweep_roots(task_root,identity):
    task_root=Path(task_root)
    roots=set()
    for name in ['JOBS.json','production_JOBS.json','auxiliary_jobs.json',
                 'room_legality_native_JOBS.json','semantic_native_recheck_JOBS.json']:
        path=task_root/name
        if not path.is_file():continue
        data=json.loads(path.read_text())
        records=data if isinstance(data,list)else data.get('jobs',[])+data.get('runs',[])
        if isinstance(data,dict):
            records=records+[data.get('scheduler',{}),data.get('controller',{})]
        for job in records:
            if not isinstance(job,dict)or 'pid'not in job:continue
            expected=job.get('process_identity',{})
            actual=identity(job.get('pid',-1))
            if actual and actual['start_ticks']==expected.get('start_ticks')and str(task_root)in actual['argv']:
                roots.add(actual['pid'])
    return roots

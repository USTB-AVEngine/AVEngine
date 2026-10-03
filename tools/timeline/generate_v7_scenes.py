#!/usr/bin/env python3
"""Seed-driven restartable CPU native candidates. Formal mode fails closed.

Only assigned, acoustically passed room scopes are schedulable. Unqualified
interpolation and incomplete family coverage cannot publish formal examples.
"""
from __future__ import annotations
import argparse,json,multiprocessing,os,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from avengine.timeline.v7_candidate_generation import generate_parent,FORMAL_BLOCKERS
from avengine.timeline.continuous_pilot import write_json


def worker(task):return generate_parent(*task)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--houses',type=Path,required=True);p.add_argument('--seeds',type=Path,required=True)
    p.add_argument('--runtime-inputs',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--offset',type=int,default=0)
    p.add_argument('--limit',type=int,default=1);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--compact-completed',action='store_true');p.add_argument('--candidate-only',action='store_true');p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if not 1<=a.workers<=64 or a.offset<0 or a.limit<1:raise ValueError('invalid shard bounds or CPU worker count')
    houses=json.loads(a.houses.read_text())['houses'];by_id={x['house']:x for x in houses}
    rows=[json.loads(l) for l in a.seeds.read_text().splitlines() if l.strip()]
    rows=rows[a.offset:a.offset+a.limit]
    if not rows or len({r['scene_id'] for r in rows})!=len(rows):raise ValueError('empty shard or duplicate parents')
    if any(r['assignment_status']!='admitted' or r['house'] not in by_id for r in rows):
        raise ValueError('shard contains unassigned or unadmitted houses')
    if not a.candidate_only and not a.plan_only:
        raise RuntimeError('formal generation blocked: '+', '.join(FORMAL_BLOCKERS))
    if a.plan_only:
        from avengine.timeline.v7_candidate_generation import seeded_scene
        from copy import deepcopy
        from avengine.timeline.v7_scene_quotas import QUOTA_INPUT_VERSION,validate_quota_row,quota_summary
        quota_rows=[r for r in rows if r.get('schema')==QUOTA_INPUT_VERSION]
        for r in rows:
            check=deepcopy(r)
            if r.get('schema')==QUOTA_INPUT_VERSION:
                validate_quota_row(r);check['schema']=r['base_seed_schema']
            seeded_scene(check,by_id[r['house']])
        print(json.dumps({'status':'input_plan_valid','parents':len(rows),'members':sum(len(r['members']) for r in rows),
                          'formal_generation_ready':False,'blocking_reasons':list(FORMAL_BLOCKERS),
                          'scenario_quota_status':'requested_only_native_not_admitted' if quota_rows else 'legacy_two_person_canary',
                          'scenario_quotas':quota_summary(quota_rows) if quota_rows else None}));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('candidate command requires CUDA_VISIBLE_DEVICES empty')
    inputs=json.loads(a.runtime_inputs.read_text());assets=json.loads(a.assets.read_text())
    if isinstance(assets,dict):assets=assets['assets']
    tasks=[(r,by_id[r['house']],inputs,assets,str(a.output),a.compact_completed) for r in rows]
    a.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(worker,tasks))
    receipt=a.output/f'shard_{a.offset:06d}_{len(rows):06d}_{time.time_ns()}.json'
    write_json(receipt,{'status':'candidate_only','scenes':results,'formal_dataset_count':0})
    print(str(receipt),flush=True)

if __name__=='__main__':main()

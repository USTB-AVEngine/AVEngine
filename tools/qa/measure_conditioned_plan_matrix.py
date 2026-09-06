#!/usr/bin/env python3
"""Measure fixed-profile plan feasibility without rendering or replacing quotas."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from avengine.capture.qa_plan_adapters import load_planning_resources
from avengine.rooms.conditioned_sampler import (build_conditioned_plan, ConditionedPlanningFailure,
    load_conditioned_sound_pool, resolve_condition_profile)
from avengine.rooms.furniture_layout import clock_config
from avengine.runtime_profiles import load_source_asset_runtime_registry

FIXTURES={};REGISTRY=None;SOUNDS=None;TEMPLATE=None;SETTINGS=None;CLOCK=None


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def trial(item):
    room_id,count,bin_index,seed=item
    request=deepcopy(TEMPLATE);request.pop('source_asset_ids',None)
    request.update(episode_id=f'matrix_{room_id}_{count}_{bin_index}_{seed}',seed=seed,sampling_policy='conditioned_static_v2',
        entities={'total_count':count,'silent_count':0,'source_classes':'articulated_human'},
        camera={'motion':'static','fov_deg':85.,'height_above_floor_m':1.55},
        profile={'anchor_count':SETTINGS['anchor_count'],'separation_bin_deg':SETTINGS['bins'][bin_index],
                 'anchor_visibility':'in_fov','anchor_line_of_sight':'clear','speech_motion':SETTINGS['motion'],
                 'event_relation':'sequential','reserve_tail_s':3.,'min_gap_between_audible_windows_s':.5,
                 'retry_budget_within_profile':SETTINGS['attempts_per_profile']},
        sound_selection={'max_clip_s':5.,'min_audible_s':1.5,'unique_first_utterance_transcripts':True})
    profile=resolve_condition_profile(request,REGISTRY);room,space,mesh,_layout=FIXTURES[room_id]
    started=time.monotonic();cpu=time.process_time()
    row={'room_id':room_id,'total_count':count,'speaking_count':count,'silent_count':0,'bin_index':bin_index,
         'separation_bin_deg':SETTINGS['bins'][bin_index],'seed':seed,'anchor_count':SETTINGS['anchor_count'],
         'clip_span_fit_policy':'filter_to_remaining_budget_then_uniform','condition_profile':profile,
         'native_execution':'not_run'}
    try:
        plan=build_conditioned_plan(room=room,request=request,source_registry=REGISTRY,sounds=SOUNDS,
                                    space=space,mesh=mesh,clock=CLOCK,condition_profile=profile)
        row.update(success=True,attempts=plan['planning_result']['attempts'],failure_histogram=plan['planning_result']['failure_histogram'],
                   legal_camera_count=plan['planned_conditions']['legal_candidate_count'],camera=plan['visual_plan']['camera'],
                   selected_assets=[a['asset_id'] for a in plan['visual_plan']['actors']],
                   events=[{k:e[k] for k in ('actor_id','sound_asset_id','sample_count','start_sample','end_sample','planned_audible_interval_samples')} for e in plan['audio_events']])
    except ConditionedPlanningFailure as exc:
        row.update(success=False,**{k:exc.result[k] for k in ('attempts','failure_histogram','gap_category')})
    row.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
    return row


def main():
    global FIXTURES,REGISTRY,SOUNDS,TEMPLATE,SETTINGS,CLOCK
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-template',type=Path,required=True)
    parser.add_argument('--room-catalog',type=Path,action='append',required=True)
    parser.add_argument('--prepared-set',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trials',type=int,default=50)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--counts',default='2,3,4')
    parser.add_argument('--anchor-count',type=int,default=1)
    parser.add_argument('--attempts-per-profile',type=int,default=1,
        help='1 matches the earlier per-route feasibility measurement; production default is 200')
    parser.add_argument('--motion',choices=['speaker_moving','competitor_moving','all_still'],default='speaker_moving')
    args=parser.parse_args()
    if args.trials<1 or not 1<=args.workers<=8 or not 1<=args.attempts_per_profile<=200:parser.error('invalid bounded trial/worker/retry count')
    output=args.output.expanduser().resolve();output.mkdir(parents=True,exist_ok=False)
    TEMPLATE=json.loads(args.request_template.read_text());REGISTRY=load_source_asset_runtime_registry(TEMPLATE.get('source_registry','examples/runtime/source_asset_runtime_profiles.json'))
    SOUNDS=load_conditioned_sound_pool(json.loads(args.prepared_set.read_text()),source_path=args.prepared_set)
    CLOCK=clock_config(frame_count=int(TEMPLATE.get('frame_count',240)),frame_rate_hz=float(TEMPLATE.get('frame_rate_hz',15)),sample_rate_hz=int(TEMPLATE.get('sample_rate_hz',16000)))
    SETTINGS={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='room_catalog'},
              'room_catalogs':[str(p.resolve()) for p in args.room_catalog], 'bins':[[15,30],[30,60],[60,90],[90,180]],
              'production_default_retry_budget':200,'denominator':'fixed seeds per room x entity count x anchor separation bin',
              'comparison_boundary':'one route attempt per seed matches prior plan-only measurement; this is not production yield or native answerability',
              'clock':CLOCK,'producer':{'cwd':str(Path.cwd()),'python':sys.executable,'python_version':platform.python_version(),
                  'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  'working_tree_changes_at_launch':subprocess.check_output(['git','status','--porcelain'],text=True).splitlines()}}
    write(output/'measurement_config.json',SETTINGS)
    for catalog in args.room_catalog:
        raw=json.loads(catalog.read_text());rooms=raw.get('rooms',raw) if isinstance(raw,dict) else raw
        for room in rooms:
            if room['room_id'] in FIXTURES:raise ValueError('duplicate room id in matrix input')
            began=time.monotonic();space,mesh,layout=load_planning_resources(room,TEMPLATE);FIXTURES[room['room_id']]=(room,space,mesh,layout)
            write(output/(room['room_id']+'_fixture.json'),{'room':room,'navigation':space.metadata,'mesh':mesh.source,
                      'setup_seconds':time.monotonic()-began})
            print('loaded',room['room_id'],flush=True)
    counts=list(map(int,args.counts.split(',')));tasks=[(room,n,b,202609060000+ri*10000+n*1000+b*100+k)
        for ri,room in enumerate(FIXTURES) for n in counts for b in range(4) for k in range(args.trials)]
    rows=[];started=time.monotonic()
    with (output/'trials.jsonl').open('x') as stream:
        with mp.get_context('fork').Pool(args.workers) as pool:
            for row in pool.imap_unordered(trial,tasks,chunksize=1):
                rows.append(row);stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                if len(rows)%25==0 or len(rows)==len(tasks):print('progress',len(rows),'/',len(tasks),'wall_s',round(time.monotonic()-started,2),flush=True)
    groups=[]
    for key,group in itertools.groupby(sorted(rows,key=lambda r:(r['room_id'],r['total_count'],r['bin_index'])),lambda r:(r['room_id'],r['total_count'],r['bin_index'])):
        group=list(group);failures=Counter()
        for row in group:failures.update(row['failure_histogram'])
        groups.append({'room_id':key[0],'total_count':key[1],'bin_deg':SETTINGS['bins'][key[2]],'trials':len(group),
                'successes':sum(r['success'] for r in group),'failure_histogram':dict(failures),
                'wall_mean_s':float(np.mean([r['wall_seconds'] for r in group])),
                'wall_p95_s':float(np.quantile([r['wall_seconds'] for r in group],.95)),
                'anchor_count':args.anchor_count,'clip_span_filter':True})
    write(output/'summary.json',{'groups':groups,'trials':len(rows),'successes':sum(r['success'] for r in rows),
                                  'elapsed_wall_seconds':time.monotonic()-started,'native_execution':'not_run'})
    print('complete',len(rows),'trials',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

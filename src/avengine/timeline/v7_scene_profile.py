"""Aggregate-only scene requests, separate from native evidence and QA truth.

The default 90% profile / 10% coverage mixture preserves independent seeds and
small/full prefixes. Frozen alpha inputs have their own explicit schema and
are unaffected. This module does not open any real recording or return gold.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import numpy as np

PROFILE_INPUT_VERSION = 'v7-production-inputs-aggregate-profile-20261004-v6'
DEFAULT_CONFIG = Path(__file__).with_name('v7_scene_profile_defaults.json')


def load_profile_config(path=None):
    config=json.loads(Path(path or DEFAULT_CONFIG).read_text())
    if config.get('schema')!='v7-aggregate-scene-profile-v1':
        raise ValueError('unsupported aggregate sampler configuration')
    for dist in [config['people_count']['distribution'],config['posture']['resolved_distribution'],
                 config['posture']['unresolved_distribution'],config['concurrent_speakers']['distribution']]:
        if not dist or any(not math.isfinite(v) or v<0 for v in dist.values()) or abs(sum(dist.values())-1)>1e-8:
            raise ValueError('sampling probabilities must be nonnegative and sum to one')
    m=config['mixture'];period=m['coverage_period']
    if not isinstance(period,int) or period<2 or not 0<=m['coverage_slot']<period or m['coverage_speaker_counts']!=list(range(9)):
        raise ValueError('invalid zero-to-eight coverage schedule')
    if abs(m['coverage_parent_fraction']-1/period)>1e-8 or abs(m['profile_parent_fraction']+m['coverage_parent_fraction']-1)>1e-8:
        raise ValueError('mixture fraction differs from deterministic coverage schedule')
    p=config['posture']
    if p['resolved_branch_probability']<0 or p['unresolved_branch_probability']<0 or abs(p['resolved_branch_probability']+p['unresolved_branch_probability']-1)>1e-8:
        raise ValueError('invalid posture branch probabilities')
    if config['azimuth']['distribution']!='uniform' or config['azimuth']['range_deg']!=[-180.,180.] or config['azimuth']['hfov_deg']!=85.:
        raise ValueError('default direction policy is uniform azimuth with 85-degree HFOV')
    _distance_cdf(config)
    return config


def _rng(seed,purpose):
    return np.random.default_rng(np.random.SeedSequence([int(seed),int(purpose)]))


def _categorical(rng,distribution):
    keys=list(distribution);weights=np.asarray(list(distribution.values()),float)
    return keys[int(rng.choice(len(keys),p=weights/weights.sum()))]


def _distance_cdf(config):
    d=config['speech_distance_m'];edges=d['histogram_edges'];weights=d['histogram_proportions']
    if len(edges)!=len(weights)+1 or any(not math.isfinite(w) or w<0 for w in weights) or abs(sum(weights)-1)>1e-8:
        raise ValueError('invalid aggregate distance histogram')
    points={float(edges[0]):0.};cumulative=0.
    for edge,weight in zip(edges[1:],weights,strict=True):
        cumulative+=weight
        if edge is not None:points[float(edge)]=min(1.,cumulative)
        elif weight>0:raise ValueError('an infinite distance tail needs an explicit bounded design policy')
    # Zero-mass remote bins do not extend the inverse-CDF support.
    first_one=min(x for x,p in points.items() if p>=1-1e-10)
    points={x:p for x,p in points.items() if x<=first_one}
    points[first_one]=1.
    for probability,distance in d['quantiles'].items():points[float(distance)]=float(probability)
    ordered=sorted(points.items());x=np.asarray([p for _,p in ordered]);y=np.asarray([v for v,_ in ordered])
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x)<=0) or np.any(np.diff(y)<=0) or x[0]!=0 or x[-1]!=1:
        raise ValueError('distance histogram and quantiles must define a monotone CDF')
    return x,y


def _profile_events(count,duration_ticks,seed,config):
    rng=_rng(seed,71);weights={k:v for k,v in config['concurrent_speakers']['distribution'].items() if int(k)<=count}
    if sum(weights.values())<=0:raise ValueError('speaker population has no feasible concurrency mass')
    active=[];blocks=[];tick=0
    low,high=config['concurrent_speakers']['hold_ticks_inclusive']
    if not (isinstance(low,int) and isinstance(high,int) and 1<=low<=high):raise ValueError('invalid independent temporal block length')
    while tick<duration_ticks:
        n=int(_categorical(rng,weights));end=min(duration_ticks,tick+int(rng.integers(low,high+1)))
        identities=sorted(int(x) for x in rng.choice(np.arange(1,count+1),size=n,replace=False))
        blocks.append({'start_tick':tick,'end_tick':end,'actor_ids':identities});tick=end
    # Merge a continuing actor across adjacent blocks; actor identity never
    # turns into a new event just because another actor enters or leaves.
    for actor in range(1,count+1):
        for block in blocks:
            if actor not in block['actor_ids']:continue
            if active and active[-1]['actor_id']==actor and active[-1]['end_tick']==block['start_tick']:
                active[-1]['end_tick']=block['end_tick']
            else:active.append({'actor_id':actor,'start_tick':block['start_tick'],'end_tick':block['end_tick']})
    return sorted(active,key=lambda e:(e['start_tick'],e['actor_id']))


def sample_scene_profile(row,config=None):
    config=load_profile_config() if config is None else deepcopy(config)
    i=int(row['scene_index']);seed=int(row['scene_seed']);m=config['mixture']
    coverage=i%m['coverage_period']==m['coverage_slot']
    count_rng=_rng(seed,11);pose_rng=_rng(row['seeds']['pose'],31);place_rng=_rng(row['seeds']['trajectory'],51)
    if coverage:
        n=(i//m['coverage_period'])%9;people=(n+m['coverage_visible_count_offset'])%9
    else:
        people=int(_categorical(count_rng,config['people_count']['distribution']))
        n=max(config['concurrent_speakers']['default_identity_pool_minimum'],people)
    # A zero-speech coverage world has visible silent people. A zero visible
    # target may instead have occluded/out-of-band speakers; occupancy differs.
    if n==0 and people==0:raise ValueError('zero speech requires a visible silent bystander')
    body_count=max(n,people+1);postures=[]
    for actor in range(1,body_count+1):
        resolved=pose_rng.random()<config['posture']['resolved_branch_probability']
        distribution=config['posture']['resolved_distribution' if resolved else 'unresolved_distribution']
        postures.append({'actor_id':actor,'branch':'resolved_proxy' if resolved else 'unknown_independent',
            'posture':_categorical(pose_rng,distribution),'action_admission':'pending registered action; no standing fallback'})
    probabilities,distances=_distance_cdf(config);slots=[]
    for actor in range(1,n+1):
        distance=float(place_rng.uniform(*config['speech_distance_m']['coverage_range_m'])) if coverage else float(np.interp(place_rng.random(),probabilities,distances))
        slots.append({'actor_id':actor,'distance_from_microphone_m':distance,
            'camera_relative_azimuth_deg':float(place_rng.uniform(-180,180)),
            'native_placement_status':'requested_only; navigation, body contact and mouth position required'})
    ticks=round(float(row['duration_s'])*10)
    if ticks<300:raise ValueError('aggregate profile requires a continuous world of at least 30 seconds')
    if coverage:
        from avengine.timeline.v7_scene_quotas import speech_events
        events=[{'actor_id':e['source_id'],'start_tick':e['start_tick'],'end_tick':e['end_tick']} for e in speech_events(n,row['duration_s'])]
    else:events=_profile_events(n,ticks,row['seeds']['event_schedule'],config)
    return {'schema':config['schema'],'branch':'coverage' if coverage else 'aggregate_profile',
        'profile_provenance':deepcopy(config['provenance']),'visible_people_360_band_target':people,
        'visible_count_scope':config['people_count']['scope'],'speech_actor_pool_count':n,
        'identity_pool_is_engineering_prior':not coverage,'physical_actor_slots_minimum':body_count,
        'posture_requests':postures,'speech_slot_requests':slots,'speech_activity_requests':events,
        'speech_count_scope':config['concurrent_speakers']['scope'],
        'camera_yaw_deg':float(_rng(row['seeds']['camera'],91).uniform(-180,180)),
        'hfov_deg':85.,'azimuth_policy':'uniform; no frustum-frequency tuning',
        'weak_proxies':deepcopy(config['weak_proxies']),
        'independent_factors':['scene count','posture','placement','camera','appearance','voice','event schedule'],
        'native_resolution_status':'requested_only; unadmitted actions/positions reject, never fallback',
        'formal_media_qualified_questions':0}


def add_profile_requests(rows,config=None):
    from avengine.timeline.v7_scene_quotas import add_quotas,QUOTA_INPUT_VERSION,validate_quota_row
    config=load_profile_config() if config is None else config
    result=[]
    for original in rows:
        row=deepcopy(original)
        if row.get('schema')!=QUOTA_INPUT_VERSION:
            if row.get('schema')!='v7-production-inputs-20261003-v3':
                raise ValueError('profile projection needs base v3 or balanced v4 seeds; diagnostic and frozen alpha inputs cannot be silently replaced')
            row=add_quotas([row])[0]
        validate_quota_row(row)
        request=sample_scene_profile(row,config);q=row['scene_quotas'];n=request['speech_actor_pool_count']
        row['upstream_quota_input_schema']=row['schema'];row['schema']=PROFILE_INPUT_VERSION
        q['upstream_balanced_speaker_request']=q['speakers'];q['speakers']=n;q['same_actor_repeated']=n>0
        q['multiple_people_same_direction']=n>=2 and (row['scene_index']//9)%2==0
        if n==0 and q['starss13_focus_class'] in (0,1):q['starss13_focus_class']=2+(q['starss13_focus_class']%11)
        if request['branch']=='aggregate_profile' and request['speech_slot_requests']:
            angle=request['speech_slot_requests'][0]['camera_relative_azimuth_deg']
            q['focus_frustum']='in_frame' if abs(angle)<42.5 else 'out_of_frame'
        q['camera_yaw_deg']=request['camera_yaw_deg']
        row['scene_profile_request']=request
        row['native_quota_status']='aggregate_request_awaiting_native_layout_recorded_dry_actions_and_QA'
        result.append(row)
    return result


def validate_profile_row(row):
    from avengine.timeline.v7_scene_quotas import QUOTA_INPUT_VERSION,validate_quota_row
    if row.get('schema')!=PROFILE_INPUT_VERSION:raise ValueError('unsupported aggregate-profile seed schema')
    check=deepcopy(row);check['schema']=QUOTA_INPUT_VERSION;validate_quota_row(check)
    r=row['scene_profile_request'];n=r['speech_actor_pool_count'];ticks=round(row['duration_s']*10)
    if r['schema']!='v7-aggregate-scene-profile-v1' or r['branch'] not in ('coverage','aggregate_profile') or not 0<=r['visible_people_360_band_target']<=8 or n!=row['scene_quotas']['speakers']:
        raise ValueError('invalid people or speech request')
    if r['hfov_deg']!=85 or r['formal_media_qualified_questions']!=0:raise ValueError('profile requests are not native truth')
    if len(r['speech_slot_requests'])!=n or {s['actor_id'] for s in r['speech_slot_requests']}!=set(range(1,n+1)):
        raise ValueError('missing physical speaker request')
    for slot in r['speech_slot_requests']:
        if not math.isfinite(slot['distance_from_microphone_m']) or slot['distance_from_microphone_m']<=0 or not -180<=slot['camera_relative_azimuth_deg']<180:
            raise ValueError('invalid distance/direction request')
    last={}
    for e in sorted(r['speech_activity_requests'],key=lambda e:(e['start_tick'],e['actor_id'])):
        actor=e['actor_id'];lo=e['start_tick'];hi=e['end_tick']
        if not 1<=actor<=n or not 0<=lo<hi<=ticks or lo<last.get(actor,0):raise ValueError('invalid/overlapping actor activity')
        last[actor]=hi
    return r


def profile_request_summary(rows):
    groups={};posture_counts=Counter();distances=[];azimuths=[]
    for row in rows:
        r=validate_profile_row(row);branch=r['branch'];g=groups.setdefault(branch,{'parents':0,'visible_people_360':Counter(),'distinct_speaker_pool':Counter(),'concurrent_speech_ticks':Counter(),'total_ticks':0,'distance_m':[],'azimuth_octants':Counter(),'horizontal_speech_out_ticks':0,'speech_actor_ticks':0})
        g['parents']+=1;g['visible_people_360'][str(r['visible_people_360_band_target'])]+=1;g['distinct_speaker_pool'][str(r['speech_actor_pool_count'])]+=1
        ticks=round(row['duration_s']*10);mask=np.zeros((r['speech_actor_pool_count'],ticks),bool)
        for e in r['speech_activity_requests']:mask[e['actor_id']-1,e['start_tick']:e['end_tick']]=True
        g['concurrent_speech_ticks'].update(str(x) for x in mask.sum(axis=0));g['total_ticks']+=ticks
        for slot in r['speech_slot_requests']:
            value=slot['distance_from_microphone_m'];angle=slot['camera_relative_azimuth_deg'];g['distance_m'].append(value);distances.append(value);azimuths.append(angle)
            g['azimuth_octants'][str(min(7,int((angle+180)//45)))]+=1
            weight=int(mask[slot['actor_id']-1].sum());g['speech_actor_ticks']+=weight
            if abs(angle)>=42.5:g['horizontal_speech_out_ticks']+=weight
        for posture in r['posture_requests']:posture_counts[(posture['branch'],posture['posture'])]+=1
    for g in groups.values():
        g['concurrent_speech_fraction']={str(i):g['concurrent_speech_ticks'][str(i)]/g['total_ticks'] for i in range(9)}
        g['distance_quantiles_m']={str(p):float(np.quantile(g['distance_m'],p)) for p in (.05,.25,.5,.75,.95)} if g['distance_m'] else {}
        g['horizontal_speech_out_fraction']=g['horizontal_speech_out_ticks']/g['speech_actor_ticks'] if g['speech_actor_ticks'] else None
        del g['distance_m']
    return {'status':'aggregate_and_coverage_requests_only','independent_parents':len(rows),'branches':groups,
        'posture_by_branch':{branch:{p:posture_counts[(branch,p)] for p in ('sitting','standing','walking','crouching')} for branch in ('resolved_proxy','unknown_independent')},
        'actual_generated_questions':0,'formal_media_qualified_questions':0,'native_sampler_render_status':'not_run; frozen alpha release remains separate',
        'distance_weighting':'independent actor slots; concurrency independent of distances. Native realized speech-actor-seconds require later media audit.'}

"""Explicit recorded/static alpha sampling; full v5 quotas remain separate.

Parent seeds and physical splits survive the alpha projection. Requested v5
quotas are retained as requests and never counted as realized media coverage.
Only admitted inputs from the selected voice roster may enter a program.
"""
from __future__ import annotations
from copy import deepcopy
import math
import numpy as np
from avengine.timeline.v7_audio_sources import PROGRAM_SCHEMA,filter_admission

ALPHA_INPUT_SCHEMA='v7-S-alpha-static-recorded-v1'


def alpha_row(original,layout,index,*,force_quartet=True):
    row=deepcopy(original)
    if row['assignment_status']!='admitted' or row['house']!=layout['house'] or layout['status']!='pass':
        raise ValueError('alpha requires an admitted house and observed static layout')
    row.update(schema=ALPHA_INPUT_SCHEMA,upstream_schema=original['schema'],
        upstream_room_id=original['room_id'],room_id=layout['room_id'],
        upstream_members=original['members'],members=['v0_a0','v0_a1','v1_a0','v1_a1'] if force_quartet else deepcopy(original['members']),
        upstream_window_start_ticks=original['window_start_ticks'],window_start_ticks=original['window_start_ticks'],
        question_quota_base=index*8 if force_quartet else original['question_quota_base'],
        counterfactual_plan='appearance_x_emitter_association' if force_quartet else original['counterfactual_plan'],
        allowed_member_changes=['actor_appearance_assignment','dry_bus_to_emitter_assignment'] if force_quartet else deepcopy(original['allowed_member_changes']),
        alpha_scope={'human_actions':['idle_pose0'],'excluded_families':['QA-26','QA-27','QA-28'],
            'full_v5_quotas_realized':False,'full_v7_generation_ready':False},
        measured_static_layout=deepcopy(layout))
    return row


def parent_appearance_assets(assets,row):
    """Use only the independent appearance seed; keep all emitter slots fixed."""
    endpoints=[a['endpoint_id'] for a in assets]
    permutation=np.random.default_rng(row['seeds']['appearance']).permutation(len(assets))
    result=[deepcopy(assets[int(i)]) for i in permutation]
    for asset,endpoint in zip(result,endpoints,strict=True):asset['endpoint_id']=endpoint
    return result


def camera_knots(starts,yaw):
    knots={0.:float(yaw)}
    for start in starts:
        for offset,delta in [(0.,100.),(.5,85.),(1.8,0.),(8.4,0.),(10.,-100.)]:
            knots[round(start+offset,8)]=float(yaw+delta)
    knots[30.]=float(yaw)
    return [{'time_s':t,'yaw_deg':value} for t,value in sorted(knots.items())]


def build_alpha_scene(row,house,records):
    if row['schema']!=ALPHA_INPUT_SCHEMA or row['house']!=house['house']:
        raise ValueError('explicit alpha row required; full quota fallback is forbidden')
    if house['acoustic_status']!='pass' or row['room_id'] not in house['checked_room_ids']:
        raise ValueError('alpha physical room did not pass the house acceptance')
    if {'development':'dev'}.get(house['split'],house['split'])!=row['split']:
        raise ValueError('physical house crossed splits')
    if house.get('known_exposure') and row['split']!='train':raise ValueError('exposed house cannot be held out')
    admitted,rejected=filter_admission(records)
    if rejected:raise ValueError('alpha source pool contains unadmitted records')
    # The current four colored bodies are male. Voice identities vary
    # independently of color and camera, while their original sex labels remain.
    roster_split='train' if row['split']=='train' else 'eval'
    usable=[r for r in admitted if r.get('speaker_id') and r.get('gender')=='M'
        and r.get('legacy_library_split')==roster_split and r.get('transcript')]
    speakers=sorted({r['speaker_id'] for r in usable})
    if len(speakers)<2:raise ValueError('two admitted recorded speakers required')
    rng=np.random.default_rng(row['seeds']['audio']);voices=list(rng.choice(speakers,size=2,replace=False))
    choices={ep:[r for r in usable if r['speaker_id']==speaker] for ep,speaker in zip(['voice1','voice2'],voices,strict=True)}
    starts=[t/10 for t in row['window_start_ticks'][:2]]
    if any(abs(t*15-round(t*15))>1e-8 for t in starts):raise ValueError('window boundaries must align native frames')
    layout=row['measured_static_layout'];yaw=layout['center_yaw_deg'];angle=math.radians(yaw)
    scene={'scene_id':row['scene_id'],'independent_parent_scene_id':row['scene_id'],'seed':row['scene_seed'],
        'split':row['split'],'duration_s':30.,'video_fps':15,'sample_rate_hz':16000,
        'listener_m':layout['listener_m'],'camera_forward':[math.sin(angle),0.,-math.cos(angle)],
        'camera_yaw_knots':camera_knots(starts,yaw),'window_starts_s':starts,'sources':{},'events':[],
        'alpha_input_schema':ALPHA_INPUT_SCHEMA,'unrealized_upstream_scene_quotas':row.get('scene_quotas'),
        'source_motion_policy':'all human idle pose0; no human translation or walking',
        'sampling_limitations':['two live male recorded speakers','four static colored male bodies',
            'requested full v5 scenario quotas are not fulfilled by alpha projection']}
    for i,placement in enumerate(layout['valid'][:4]):
        ep='voice'+str(i+1);position=placement['position_m']
        scene['sources'][ep]={'actor_id':i+1,'class_id':1,'content_kind':'speech',
            'body_yaw_deg':placement.get('body_yaw_deg'),'mechanism':'human_live_recorded','dry_gain':float(rng.uniform(.8,1.2)) if i<2 else 1.,
            'trajectory':[{'time_s':0.,'position_m':position},{'time_s':30.,'position_m':position}]}
    program={'schema':PROGRAM_SCHEMA,'records':[],'events':[]};by_id={}
    # Fresh utterances per parent; repeated actor identity is retained. Carry-in
    # and pauses exist in the full 30s world, before any 10s crop is assembled.
    selected={ep:list(rng.permutation(rows)) for ep,rows in choices.items()}
    scheduled=[]
    for wi,start in enumerate(starts):
        schedule=[('voice1',start-.8),('voice1',start+1.8),('voice2',start+2.4),
            ('voice1',start+4.4),('voice2',start+5.0),('voice1',start+7.0),('voice2',start+7.6)]
        for ep,when in schedule:
            candidates=selected[ep]
            # Prefer a whole utterance fitting the intended 1--2.5s slot;
            # failures are data deficits, never speech stretching/truncation.
            record=next((r for r in candidates if .8<=r['duration_s']<=min(2.5,30.-max(0.,when))),None)
            if record is None:
                selected[ep]=list(rng.permutation(choices[ep]));candidates=selected[ep]
                record=next((r for r in candidates if .8<=r['duration_s']<=min(2.5,30.-max(0.,when))),None)
                if record is None:raise ValueError('speaker has no short complete admitted utterance')
            candidates.remove(record)
            if not candidates:selected[ep]=list(rng.permutation(choices[ep]))
            lo=round(max(0.,when)*10);ticks=math.ceil(record['header']['sample_count']/1600)
            hi=lo+ticks
            if hi>300 or any(old_ep==ep and lo<old_hi and hi>old_lo for old_ep,old_lo,old_hi in scheduled):
                raise ValueError('recorded utterance schedule overlaps itself or leaves the world')
            event_id=f'utterance_{len(scene["events"]):04d}';actor=scene['sources'][ep]['actor_id']
            scene['events'].append({'event_id':event_id,'endpoint_id':ep,'source_id':actor,'class_id':1,
                'start_tick':lo,'end_tick':hi,'mechanism':'human_live_recorded'})
            program['events'].append({'event_id':event_id,'sound_asset_id':record['sound_asset_id'],
                'crop_start_sample':0,'preserve_full_recorded_clip':True})
            by_id[record['sound_asset_id']]=record;scheduled.append((ep,lo,hi))
    program['records']=list(by_id.values())
    working=deepcopy(house);working['room']=next(r for r in house['checked_rooms'] if r['room_label']==row['room_id'])
    working['room']['placement']=deepcopy(working['room']['placement']);working['room']['placement']['camera_m']=scene['listener_m']
    return working,scene,program

"""Bridge existing v3 post-sound conditions into v7 physical scene proposals.

The compiler/solver own every frame interval. These are construction requests;
native actions, contacts, RIRs, wet tails and shipped QA judges still admit them.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json,math
import numpy as np
from avengine.qa import generation_conditions as gc
from avengine.rooms import conditioned_motion as cm
from avengine.rooms import conditioned_sampler as cs

POST_SOUND_FAMILIES=('QA-13','QA-16','QA-17')
REVISION='v3-post-sound-conditions-v7-five-crop-adapter-v2-authored-clock'


def registered_human_motion_registry(assets,inputs,house,*,frame_rate_hz=15.):
    """Read real admitted clip names/periods; never admit a new physical route."""
    from avengine.assets.roster_motion.io import read_glb
    from avengine.assets.roster_motion.pose import animation_times
    path=Path(inputs['v7_asset_roster']);roster=json.loads(path.read_text())
    family='mp3d' if house['house'].startswith('mp3d') else 'hm3d'
    result=[];lineage=[]
    for asset in assets:
        record=deepcopy(asset)
        if asset.get('kind')=='human' and asset.get('walk_allowed'):
            major=next(r for r in roster['humans']['majors'] if r['major_id']==asset['body_identity_id'])
            variant=next(r for r in major['minor_variants'] if r['color']==asset['color'])
            if not variant.get('walk_allowed') or variant['habitat']['walk']['status']!='pass_admitted_routes':
                raise ValueError('registered walk authority missing: '+asset['asset_id'])
            master=Path(roster['artifact_root'])/variant['habitat']['motion_master']
            document,binary=read_glb(master)
            action=next(a for a in document['animations'] if a['name']=='Walking_Contact_'+family)
            times=animation_times(document,binary,action);period=float(times[-1]-times[0])
            record['timeline']={'idle_action_id':asset['habitat_glb'],
                'walking_action_id':action['name'],'body_plan_id':asset['body_identity_id'],
                'walk_phase_period_frames':max(1,round(period*frame_rate_hz))}
            lineage.append({'asset_id':asset['asset_id'],'master':str(master),
                'master_sha256':hashlib.sha256(master.read_bytes()).hexdigest(),
                'clip':action['name'],'measured_period_s':period,
                'new_house_admission_required':True})
        result.append(record)
    return {'assets':result,'source_roster':str(path),
        'source_roster_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'lineage':lineage}


def prepared_sound(record):
    """Use measured activity in the actual complete 16kHz source coordinates."""
    header=record['header'];activity=(record.get('screens',{}).get('automatic_activity',{}).get('activity_intervals_samples')
        or record.get('source_normalization',{}).get('measured_activity',{}).get('source_activity_intervals_samples'))
    if not activity:raise ValueError('measured source activity missing: '+record['sound_asset_id'])
    return {'sound_asset_id':record['sound_asset_id'],'sample_rate_hz':header['sample_rate_hz'],
        'sample_count':header['sample_count'],'audible_start_sample':min(a for a,b in activity),
        'audible_end_sample_exclusive':max(b for a,b in activity),
        'source_activity_intervals_samples':activity,
        'source_origin':record.get('normalized_path',record.get('path'))}


def solve_post_sound_window(qa_id,*,target,companion,assets,registry,records,
        speed_m_s=1.,minimum_motion_s=2.,measured_wet_tail_s=None):
    if qa_id not in POST_SOUND_FAMILIES:raise ValueError('post-sound adapter only owns QA-13/16/17')
    instances=[{'entity_instance_id':a['asset_id'],'asset_id':a['asset_id'],
        'source_class':a['entity_class'],'role':'anchor' if a['asset_id']==target else 'candidate'}
        for a in assets]
    def compile_one(family):
        return gc.compile_generation_conditions(
            {'qa_id':family,'target_instance_ids':[target],'event':{'kind':'target_audible_window'}},
            branch='yes' if family=='QA-17' else None,
            instances=instances,registry=registry,capabilities=cs)
    primary=compile_one(qa_id)
    # QA-13 itself leaves motion unconstrained. Choose its legal moving subcase
    # using the existing QA-17 condition, while retaining the original QA-13 plan.
    movement=compile_one('QA-17') if qa_id=='QA-13' else primary
    clock=cm.EpisodeClock(150,15.,16000)
    budget=cm.MotionBudget(walk_speed_range_mps=(speed_m_s,speed_m_s),
        earliest_start_s=0.,minimum_motion_s=minimum_motion_s)
    kwargs={'clock':clock,'budget':budget,'sounds':{aid:prepared_sound(r) for aid,r in records.items()},
        'registry':registry,'camera':{'motion':'static'},
        'event_start_s':{companion:0.,target:3.},
        'measured_wet_tail_s':measured_wet_tail_s}
    # Let the existing event scheduler and motion solver find a joint legal
    # suffix. Scheduling uses v7 program ticks; waveform activity stays at 16kHz.
    prepared=kwargs['sounds']
    events={aid:{'actor_id':aid,'entity_instance_id':aid,
        'audible_start_sample':math.floor(r['audible_start_sample']/1600),
        'audible_end_sample_exclusive':math.ceil(r['audible_end_sample_exclusive']/1600)}
        for aid,r in prepared.items()}
    last_tick=math.floor((clock.duration_s-budget.reserve_tail_s-records[target]['header']['sample_count']/16000)*10)
    profile={'event_relation':'sequential','first_speaker_instance_id':companion,
        'min_gap_between_audible_windows_s':budget.min_gap_between_audible_windows_s}
    for tick in range(last_tick+1):
        starts={companion:[[0,0]],target:[[tick,tick]]}
        scheduled=cs.schedule_legal_events(events,starts,{'sample_rate_hz':10},profile,np.random.default_rng(0))
        if scheduled is None:continue
        kwargs['event_start_s']={aid:value/10 for aid,value in scheduled.items()}
        primary_solution=cm.solve_motion_windows(primary,**kwargs)
        motion_solution=cm.solve_motion_windows(movement,**kwargs) if qa_id=='QA-13' else primary_solution
        if primary_solution.solved and motion_solution.solved:
            return primary,movement,primary_solution,motion_solution
    # Preserve the existing solver's named rejection rather than inventing a pass.
    if 'primary_solution'not in locals():
        kwargs['event_start_s']={companion:0.,target:max(0,last_tick)/10}
        primary_solution=cm.solve_motion_windows(primary,**kwargs)
        motion_solution=cm.solve_motion_windows(movement,**kwargs) if qa_id=='QA-13' else primary_solution
    return primary,movement,primary_solution,motion_solution


def plan_post_sound_motion(row,*,qa_id,target,companion,assets,registry,
        records_by_asset,speed_m_s=1.,minimum_motion_s=2.,epochs_s=(0.,10.,20.)):
    row=deepcopy(row)
    knots=row.get('camera_yaw_knots',[])
    if (not knots or any(abs((float(k['yaw_deg'])-float(knots[0]['yaw_deg'])+180)%360-180)>1e-9 for k in knots)
            or row.get('semantic_camera_sector_targets_deg')):
        raise ValueError('v3 post-sound solver requires the actual v7 camera to remain static')
    if set(row.get('source_classes',{}).get(target,[]))!={'speech_playback'}:
        raise ValueError('post-sound human must keep one speaking source, without automatic footsteps')
    if len(epochs_s)!=len(records_by_asset[target]):raise ValueError('each native epoch needs its actual complete target recording')
    selected=[a for a in assets if a['asset_id'] in {target,companion}]
    if len(selected)!=2:raise ValueError('two actual independently referable source bodies required')
    proposals=[];starts=[];intervals=[];distance_windows=[]
    for index,epoch in enumerate(epochs_s):
        records={aid:values[min(index,len(values)-1)] for aid,values in records_by_asset.items() if aid in {target,companion}}
        primary,movement,question,solution=solve_post_sound_window(qa_id,target=target,
            companion=companion,assets=selected,registry=registry,records=records,
            speed_m_s=speed_m_s,minimum_motion_s=minimum_motion_s)
        if not question.solved or not solution.solved:
            raise cm.ConditionedMotionError(json.dumps({'qa_id':qa_id,'question':question.to_dict(),
                'motion':solution.to_dict()},ensure_ascii=False))
        requirement=solution.requirement_for(target);placement=solution.placement_for(target)
        if not requirement.must_move or requirement.moving_frames is None:
            raise cm.ConditionedMotionError('compiled post-sound branch did not request target movement')
        lo,hi=requirement.moving_frames;lo=epoch+lo/15.;hi=epoch+hi/15.
        intervals.append([lo,hi]);distance_windows.append([lo,hi])
        start=epoch+placement.start_sample/16000
        if abs(start*10-round(start*10))>1e-8:raise ValueError('solver event start is not representable on v7 100ms program clock')
        starts.append(start)
        proposals.append({'epoch_start_s':epoch,'compiled_conditions':primary.to_dict(),
            'construction_conditions':movement.to_dict(),'question_solution':question.to_dict(),
            'movement_solution':solution.to_dict(),
            'requires_native_wet_tail_requery':True,'native_admission':'not_yet_measured'})
    request={'action':'walk','speed_m_s':speed_m_s,'intervals_s':intervals,
        'minimum_listener_horizontal_distance_m':1.5,
        'conditioned_post_motion_revision':REVISION}
    if qa_id=='QA-16':
        request.update(distance_trend=row.get('scene_demand',{}).get('distance_trend','approaching'),
            distance_trend_windows_s=distance_windows,minimum_window_distance_change_m=.5)
    row['motion_requests']={target:request}
    row.setdefault('phase2_recorded_speech_schedule_s',{})[target]=starts
    row.setdefault('phase2_recorded_sound_ids',{})[target]=[r['sound_asset_id'] for r in records_by_asset[target]]
    if companion in row.get('semantic_human_asset_ids',[]):
        row['semantic_start_offsets_s']=[0. for _ in row['semantic_human_asset_ids']]
        row.get('phase2_recorded_speech_schedule_s',{}).pop(companion,None)
        row.get('phase2_recorded_sound_ids',{}).pop(companion,None)
    else:row.setdefault('phase2_recorded_speech_schedule_s',{})[companion]=list(epochs_s)
    if companion in row.get('semantic_human_asset_ids',[]):
        row['phase2_authored_speech_schedule_s']={companion:list(epochs_s)}
    row['acceptance_window_starts_s']=[0.,5.,10.,15.,20.];row['window_count']=5
    row['conditioned_post_motion']={'revision':REVISION,'qa_id':qa_id,'target_asset_id':target,
        'companion_asset_id':companion,'five_crop_world_starts_s':[0.,5.,10.,15.,20.],
        'native_pose_readback_hz':30,'solver_clock_hz':15,'epochs':proposals,
        'motion_registry':registry,'native_gates_changed':False,
        'query_authority':'actual native wet support and original QA _silent_after; planned reserve is not proof'}
    row.setdefault('scene_demand',{}).pop('required_speaking_motion_contrast',None)
    row['scene_demand']['conditioned_post_motion']=row['conditioned_post_motion']
    row['scene_demand']['required_post_sound_motion_contrast']={'moving_asset_id':target,'static_asset_id':companion}
    row['conditioned_post_motion']['event_scheduler']='avengine.rooms.conditioned_sampler.schedule_legal_events'
    return row


def apply_authored_post_schedule(row,scene,program,assets):
    requested=row.get('phase2_authored_speech_schedule_s',{})
    if not requested:return scene,program
    scene=deepcopy(scene);program=deepcopy(program)
    byid={r['sound_asset_id']:r for r in program['records']}
    for aid,starts in requested.items():
        actor=next(a for a in assets if a['asset_id']==aid);ep=actor['endpoint_id']
        events=[e for e in scene['events'] if e['endpoint_id']==ep]
        bindings={b['event_id']:b for b in program['events']}
        if not events:raise ValueError('authored post schedule has no actual source event')
        identities={bindings[e['event_id']]['sound_asset_id'] for e in events}
        if len(identities)!=1:raise ValueError('explicit authored post schedule needs one fixed complete utterance')
        record=byid[next(iter(identities))]
        if record.get('source_type')!='author_tts':raise ValueError('authored schedule cannot relabel a recorded source')
        template=deepcopy(events[0]);binding=deepcopy(bindings[template['event_id']])
        if binding.get('crop_start_sample',0)!=0:raise ValueError('authored speech must remain complete')
        old={e['event_id'] for e in events}
        scene['events']=[e for e in scene['events'] if e['event_id']not in old]
        program['events']=[b for b in program['events'] if b['event_id']not in old]
        h=record['header'];ticks=(h['sample_count']*10+h['sample_rate_hz']-1)//h['sample_rate_hz']
        previous=-1
        for i,start in enumerate(starts):
            lo=round(start*10)
            if abs(start*10-lo)>1e-8 or lo<previous or lo+ticks>300:
                raise ValueError('complete authored post programme conflicts with the native clock')
            eid='conditioned_author_'+ep+'_'+str(i)
            scene['events'].append({**template,'event_id':eid,'start_tick':lo,'end_tick':lo+ticks})
            program['events'].append({**binding,'event_id':eid})
            previous=lo+ticks
    scene['authored_post_schedule_source']='compiled post-sound epoch EventPlacement, not the legacy 0/15s schedule'
    return scene,program


def build_conditioned_post_scene(row,house,records,catalog,semantic_manifest=None):
    from avengine.timeline.v7_phase2_sources import build_phase2_scene
    working,scene,program,assets=build_phase2_scene(row,house,records,catalog,semantic_manifest)
    scene,program=apply_authored_post_schedule(row,scene,program,assets)
    return working,scene,program,assets


def generate_post_sound_parent(*args,**kwargs):
    """Compatibility entry; the producer calls the adapter directly in source."""
    from avengine.timeline.v7_phase2_generation import generate_phase2_parent
    return generate_phase2_parent(*args,**kwargs)

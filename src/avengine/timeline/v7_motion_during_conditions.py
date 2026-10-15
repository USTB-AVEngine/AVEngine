"""Execute QA v3's QA-06/15 motion plans on v7's native parent clock.

The v3 compiler/solver and the delivered QA gates stay unchanged. A ten second
static-camera solution is offset into the parent; five delivery crops still
come from the one native 30 second world. Post-sound planning is a separate
adapter owned by the post-sound task.
"""
from copy import deepcopy
import math
import numpy as np
from avengine.qa import generation_conditions as gc
from avengine.rooms import conditioned_motion as cm,conditioned_sampler as cs


def _compile_and_solve(config):
    compiled=gc.compile_generation_conditions(config['query'],branch=config['branch'],
        instances=config['instances'],registry=config['registry'],capabilities=cs)
    return cm.solve_motion_windows(compiled,clock=cm.EpisodeClock(150,15.,16000),
        budget=cm.MotionBudget(**config['budget']),sounds=config['sounds'],
        registry=config['registry'],camera={'motion':'static'},
        event_start_s=config['event_start_s'],distance_trend_criterion=config.get('distance_trend_criterion'),
        measured_wet_tail_s=config.get('measured_wet_tail_s'))


def prepare_motion_during_row(row,catalog,records):
    """Construct an explicitly requested during-sound plan before native work."""
    requests=row.get('motion_during_condition_requests')
    if not requests or row.get('motion_during_condition_plan'):return row
    from avengine.timeline.v7_visible_sources import production_audio_record_views
    from avengine.timeline.v7_environmental_sources import recording_header
    result=deepcopy(row);byasset={a['asset_id']:a for a in catalog};byrecord={r['sound_asset_id']:r for r in production_audio_record_views(records)}
    instances=[];registry={'assets':[]}
    for aid in result['asset_ids']:
        asset=byasset[aid];kind='articulated_human' if asset['kind']=='human' else 'articulated_animal'
        instances.append({'entity_instance_id':aid,'asset_id':aid,'source_class':kind})
        timeline={'idle_action_id':'idle','body_plan_id':'human' if asset['kind']=='human' else 'animal','walk_phase_period_frames':30}
        if asset.get('walk_allowed'):timeline['walking_action_id']='walk'
        admission=result.get('motion_requests',{}).get(aid,{}).get('animal_motion_admission',{})
        if admission.get('mode')=='slide_no_animation':
            timeline['walking_action_id']='registered_native_slide_no_animation'
            timeline['v7_locomotion_mode']='slide_no_animation'
        registry['assets'].append({'asset_id':aid,'entity_class':kind,'timeline':timeline,
            'v7_registered_source_registry_sha256':asset.get('source_registry_sha256')})
    plans=[];motion={}
    for request in requests:
        qa=request['qa_id'];branch=request['branch'];target=request['target_asset_id']
        if qa not in {'QA-06','QA-15','QA-16','QA-17'} or (qa=='QA-06' and branch!='moving') or (qa=='QA-17' and branch!='yes'):raise ValueError('unsupported original motion condition branch')
        if target not in result.get('motion_requests',{}):raise ValueError('condition target lacks an admitted v7 motion request')
        sounds={};starts={}
        for aid in result['asset_ids']:
            sid=result.get('proposed_sound_asset_ids',{}).get(aid)
            if sid not in byrecord or aid in result.get('silent_asset_ids',[]):continue
            record=byrecord[sid]
            if not record.get('header'):record=recording_header(record)
            header=record['header'];n=int(header['sample_count']);rate=int(header['sample_rate_hz'])
            if rate!=16000:raise ValueError('conditioned v7 sound must retain the 16kHz source clock')
            intervals=record.get('screens',{}).get('automatic_activity',{}).get('activity_intervals_samples') or record.get('source_normalization',{}).get('measured_activity',{}).get('source_activity_intervals_samples')
            if not intervals:raise ValueError('conditioned motion needs measured source activity')
            profile=result.get('phase2_animal_program_samples',{}).get(aid)
            if profile:
                begin=int(profile.get('crop_start_sample',0));n=int(profile['sample_count'])
                if profile.get('tail_muted'):
                    from .v7_tail_muting import read_tail_muted_program
                    _,measured=read_tail_muted_program(record,profile)
                    intervals=measured['activity_intervals_samples']
                    if not intervals:raise ValueError('tail-muted motion sound has no actual activity')
                else:
                    required=profile.get('required_activity_intervals_samples',intervals)
                    if profile.get('selected_activity_index')is not None and required!=[intervals[int(profile['selected_activity_index'])]]:raise ValueError('selected complete call differs from original measured activity')
                    if any(int(lo)<begin or int(hi)>begin+n for lo,hi in required):raise ValueError('planned crop would remove the complete selected measured activity')
                    intervals=[[max(int(lo),begin)-begin,min(int(hi),begin+n)-begin]for lo,hi in intervals if int(hi)>begin and int(lo)<begin+n]
            count=math.ceil(n/1600)*1600
            sounds[aid]={'sound_asset_id':sid,'sample_rate_hz':rate,'sample_count':count,
                'audible_start_sample':int(intervals[0][0]),'audible_end_sample_exclusive':int(intervals[-1][1]),
                'source_activity_intervals_samples':intervals,'source_origin':record.get('normalized_path') or record.get('path')}
            starts[aid]=float(result.get('motion_source_start_s',{}).get(aid,
                result.get('phase2_whole_animal_recordings',{}).get(aid,
                result.get('phase2_recorded_speech_schedule_s',{}).get(aid,[.3]))[0]))
        if target not in sounds:raise ValueError('conditioned target needs a complete measured sound program')
        config={'query':{'qa_id':qa,'target_instance_ids':[target],'event':{'kind':'target_audible_window'}},
            'branch':branch,'instances':[{**x,'role':'anchor' if x['asset_id']==target else 'competitor'}for x in instances],
            'registry':registry,'sounds':sounds,'event_start_s':starts,
            'budget':{'walk_speed_range_mps':request.get('speed_range_m_s',[float(request['speed_m_s'])]*2),'reserve_tail_s':float(request.get('reserve_tail_s',3.)),'minimum_motion_s':float(request.get('minimum_motion_s',0.)),'end_hold_s':float(request.get('end_hold_s',0.))}}
        if qa in {'QA-16','QA-17'}:
            tail=request.get('wet_tail_planning_bound_s',request.get('measured_wet_tail_s',4.1))
            config['measured_wet_tail_s']=None if tail is None else float(tail)
            config['wet_tail_planning_provenance']=('pending actual native wet measurement; original delivered _silent_after is authoritative' if tail is None else 'conservative configured native IR bound; not a per-event measurement; delivered QA uses actual measured wet support')
        if qa=='QA-15':config['distance_trend_criterion']={'net_change_at_least_m':float(request.get('minimum_distance_change_m',.5))}
        solution=_compile_and_solve(config)
        if not solution.solved:raise ValueError('conditioned_motion rejected: '+str(solution.rejections))
        requirement=solution.requirement_for(target)
        if requirement is None or requirement.moving_frames is None:raise ValueError('solver did not produce the requested moving interval')
        offsets=request.get('offsets_s',[0.]);intervals=[]
        for offset in offsets:
            lo,hi=requirement.moving_frames
            # Smooth acceleration is physical execution outside the mandatory
            # stable sounding interval, not a change to the compiled interval.
            intervals.append([max(0.,float(offset)+lo/15.-.3),min(30.,float(offset)+hi/15.+.3)])
        motion.setdefault(target,[]).extend(intervals)
        target_request=result['motion_requests'][target]
        if qa in {'QA-06','QA-15'}:
            placement=solution.placement_for(target)
            straight=[[float(offset)+placement.start_frame/15.,float(offset)+placement.end_frame/15.]for offset in offsets]
            target_request.setdefault('sounding_straight_windows_s',[]).extend(straight)
        else:
            latest=solution.query_window['latest_query_frame_exclusive']/15.
            earliest=solution.query_window['earliest_query_frame']/15.
            straight=[[float(offset)+max(earliest,latest-2.05),float(offset)+latest]for offset in offsets]
            if not result.get('motion_quartet_template_revision') and any(hi-lo<2.-1e-8 for lo,hi in straight):raise ValueError('construct a shorter whole recording to leave >=2 seconds beyond wet tail')
        target_request.setdefault('straight_motion_windows_s',[]).extend(straight)
        if qa=='QA-15':
            result['motion_requests'][target]['distance_trend']='approaching' if branch=='nearer' else 'receding'
            placement=solution.placement_for(target)
            result['motion_requests'][target]['distance_trend_intervals_s']=[[float(offset)+placement.start_frame/15.,float(offset)+placement.end_frame/15.]for offset in offsets]
            result['motion_requests'][target]['minimum_distance_change_m']=float(request.get('minimum_distance_change_m',.5))
        result['motion_requests'][target]['speed_m_s']=float(request['speed_m_s'])
        plans.append({'config':config,'solution':solution.to_dict(),'target_asset_id':target,'offsets_s':offsets,'execution_intervals_s':intervals})
    for aid,intervals in motion.items():
        merged=[]
        for lo,hi in sorted(intervals):
            if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
            else:merged.append([lo,hi])
        result['motion_requests'][aid]['intervals_s']=merged
    yaw=float(result['camera_yaw_deg']);result['camera_yaw_knots']=[{'time_s':0.,'yaw_deg':yaw},{'time_s':30.,'yaw_deg':yaw}]
    result['motion_during_condition_plan']={'status':'solved','owner':'motion helper v3 sound and post-sound adapter','solver':cm.solver_signature(),
        'clock_mapping':'v3 150 frames at 15Hz + explicit parent offsets; original five half-overlap v7 crops',
        'static_camera_constructed':True,'plans':plans,'qa_predicates_changed':False,'native_admission_still_required':True}
    return result


TREND_SMOOTHING_S=.5    # owner 10-15: "嘴部随头晃动导致中途有12厘米回摆，像这种，就没必要限制了吧"


def smooth_track(track,frame_rate_hz=15.,window_s=TREND_SMOOTHING_S):
    """Centered moving average of a (T,3) track: removes gait sway of the mouth, keeps the walking trend."""
    track=np.asarray(track,float);n=max(1,int(round(window_s*frame_rate_hz)))
    if n<=1 or len(track)<n:return track
    pad=n//2;padded=np.pad(track,((pad,n-1-pad),(0,0)),mode='edge')
    kernel=np.ones(n)/n
    return np.column_stack([np.convolve(padded[:,k],kernel,mode='valid') for k in range(track.shape[1])])


def verify_motion_during_native(row,scene,visual):
    plan=row.get('motion_during_condition_plan')
    if not plan:return None
    from avengine.timeline.continuous_scene import position_at
    byasset={s['asset_id']:s for s in scene['sources'].values() if not s.get('owner_endpoint_id')}
    reports=[]
    for item in plan['plans']:
        solution=_compile_and_solve(item['config'])
        for offset in item['offsets_s']:
            positions={aid:np.asarray([position_at(source['trajectory'],float(offset)+f/15.)for f in range(150)]) for aid,source in byasset.items()}
            roots={}
            admissions=visual.get('native_motion_admissions',{})
            for ep,source in scene['sources'].items():
                if source.get('owner_endpoint_id'):continue
                aid=source['asset_id'];record=admissions.get(ep)
                if record:
                    # This same double-precision track has already passed all
                    # 901 native root readbacks. Mouth bobbing is source motion,
                    # not locomotion speed; v3 exposes separate root_positions_m.
                    samples=np.asarray(record['world_roots_m'],float)
                    source_times=np.arange(len(samples))/30.
                    at=float(offset)+np.arange(150)/15.
                    roots[aid]=np.column_stack([np.interp(at,source_times,samples[:,axis])for axis in range(3)])
                else:roots[aid]=positions[aid]
            # Distance trends are judged on the 0.5 s smoothed mouth track (owner 10-15); net change and reversal
            # limits are unchanged, so walking back is still caught while head sway is not counted.
            # owner 10-15: head motion is not source locomotion.  A moving body's trend is read from its certified
            # root track at the mouth's mean height (a big dog's mouth sits ~0.45 m ahead of the root and swung
            # 17 cm when it turned its head); static bodies keep their mouth track.
            trend_tracks={}
            for aid,track in positions.items():
                if aid in roots and admissions:
                    body=np.asarray(roots[aid],float);mouth_height=float(np.mean(np.asarray(track,float)[:,1]))
                    trend_tracks[aid]=np.column_stack((body[:,0],np.full(len(body),mouth_height),body[:,2]))
                else:trend_tracks[aid]=track
            smoothed={aid:smooth_track(track) for aid,track in trend_tracks.items()}
            report=cm.verify_motion_candidate(solution,positions_m=smoothed,
                root_positions_m=roots if admissions else None,listener_position_m=scene['listener_m'])
            report['distance_trend_smoothing_s']=TREND_SMOOTHING_S
            report['distance_trend_smoothing_authority']='owner 10-15: mouth sway from head motion is not a reversal'
            report['distance_trend_track']='moving bodies: certified root at mouth height; static: mouth'
            reports.append({'offset_s':offset,'qa_id':solution.qa_id,'report':report})
    return {'status':'pass' if all(x['report']['status']=='pass'for x in reports)else'fail',
        'source':'actual native attached mouth trajectories; original v3 verifier','reports':reports,'delivered_qa_gates_unchanged':True}


def write_native_motion_condition_report(directory,report):
    """Keep previous failed evidence and make each immutable replay addressable."""
    import hashlib,json,uuid
    from pathlib import Path
    from avengine.timeline.continuous_pilot import write_json
    saved=json.loads(json.dumps({**report,'adapter_revision':'separate-native-certified-root-locomotion-v2'},sort_keys=True))
    digest=hashlib.sha256(json.dumps(saved,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
    attempt=Path(directory)/('conditioned_motion_attempt_'+uuid.uuid4().hex)
    attempt.mkdir()
    path=attempt/('conditioned_motion_native_verification_'+digest+'.json')
    write_json(path,saved)
    return path


_SOLUTION_CACHE={}


def _cached_solution(config):
    import hashlib,json
    key=hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest()
    if key not in _SOLUTION_CACHE:_SOLUTION_CACHE[key]=_compile_and_solve(config)
    return _SOLUTION_CACHE[key]


def verify_planned_trajectories(plan,positions_by_asset,listener_m):
    """The judge's own verifier on planned (or natively re-placed) tracks, before any rendering (Claude 10-15).

    positions_by_asset maps asset_id -> (T,3) world positions sampled at 30 Hz over the 30 s world; bodies not
    given stand still at their first sample.  Returns {qa_id: bool} -- a family passes only if every offset passes.
    owner 10-15: "为什么运动条件不一定成立" -- planning must use the same ruler as the judge.
    """
    result={}
    for item in plan['plans']:
        solution=_cached_solution(item['config'])
        ok=True
        for offset in item['offsets_s']:
            at=float(offset)+np.arange(150)/15.
            positions={}
            for aid,track in positions_by_asset.items():
                track=np.asarray(track,float);times=np.arange(len(track))/30.
                positions[aid]=np.column_stack([np.interp(at,times,track[:,axis])for axis in range(3)])
            report=cm.verify_motion_candidate(solution,positions_m=positions,root_positions_m=None,listener_position_m=listener_m)
            ok=ok and report.get('status')=='pass'
        result[solution.qa_id]=result.get(solution.qa_id,True) and ok
    return result

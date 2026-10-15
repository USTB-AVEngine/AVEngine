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
        event_start_s=config['event_start_s'],distance_trend_criterion=config.get('distance_trend_criterion'))


def prepare_motion_during_row(row,catalog,records):
    """Construct an explicitly requested during-sound plan before native work."""
    requests=row.get('motion_during_condition_requests')
    if not requests or row.get('motion_during_condition_plan'):return row
    result=deepcopy(row);byasset={a['asset_id']:a for a in catalog};byrecord={r['sound_asset_id']:r for r in records}
    instances=[];registry={'assets':[]}
    for aid in result['asset_ids']:
        asset=byasset[aid];kind='articulated_human' if asset['kind']=='human' else 'articulated_animal'
        instances.append({'entity_instance_id':aid,'asset_id':aid,'source_class':kind})
        timeline={'idle_action_id':'idle','body_plan_id':'human' if asset['kind']=='human' else 'animal','walk_phase_period_frames':30}
        if asset.get('walk_allowed'):timeline['walking_action_id']='walk'
        registry['assets'].append({'asset_id':aid,'entity_class':kind,'timeline':timeline,
            'v7_registered_source_registry_sha256':asset.get('source_registry_sha256')})
    plans=[];motion={}
    for request in requests:
        qa=request['qa_id'];branch=request['branch'];target=request['target_asset_id']
        if qa not in {'QA-06','QA-15'} or (qa=='QA-06' and branch!='moving'):raise ValueError('during adapter owns QA-06 moving and QA-15 only')
        if target not in result.get('motion_requests',{}):raise ValueError('condition target lacks an admitted v7 motion request')
        sounds={};starts={}
        for aid in result['asset_ids']:
            sid=result.get('proposed_sound_asset_ids',{}).get(aid)
            if sid not in byrecord or aid in result.get('silent_asset_ids',[]):continue
            record=byrecord[sid];header=record['header'];n=int(header['sample_count']);rate=int(header['sample_rate_hz'])
            if rate!=16000:raise ValueError('conditioned v7 sound must retain the 16kHz source clock')
            intervals=record['screens']['automatic_activity']['activity_intervals_samples']
            if not intervals:raise ValueError('conditioned motion needs measured source activity')
            count=math.ceil(n/1600)*1600
            sounds[aid]={'sound_asset_id':sid,'sample_rate_hz':rate,'sample_count':count,
                'audible_start_sample':int(intervals[0][0]),'audible_end_sample_exclusive':int(intervals[-1][1]),
                'source_activity_intervals_samples':intervals,'source_origin':record['path']}
            starts[aid]=float(result.get('phase2_whole_animal_recordings',{}).get(aid,
                result.get('phase2_recorded_speech_schedule_s',{}).get(aid,[.3]))[0])
        if target not in sounds:raise ValueError('conditioned target needs a complete measured sound program')
        config={'query':{'qa_id':qa,'target_instance_ids':[target],'event':{'kind':'target_audible_window'}},
            'branch':branch,'instances':[{**x,'role':'anchor' if x['asset_id']==target else 'competitor'}for x in instances],
            'registry':registry,'sounds':sounds,'event_start_s':starts,
            'budget':{'walk_speed_range_mps':request.get('speed_range_m_s',[float(request['speed_m_s'])]*2),'reserve_tail_s':3.,'minimum_motion_s':float(request.get('minimum_motion_s',0.))}}
        if qa=='QA-15':config['distance_trend_criterion']={'net_change_at_least_m':float(request.get('minimum_distance_change_m',1.5))}
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
        if qa=='QA-15':
            result['motion_requests'][target]['distance_trend']='approaching' if branch=='nearer' else 'receding'
            placement=solution.placement_for(target)
            result['motion_requests'][target]['distance_trend_intervals_s']=[[float(offset)+placement.start_frame/15.,float(offset)+placement.end_frame/15.]for offset in offsets]
            result['motion_requests'][target]['minimum_distance_change_m']=float(request.get('minimum_distance_change_m',1.5))
        result['motion_requests'][target]['speed_m_s']=float(request['speed_m_s'])
        plans.append({'config':config,'solution':solution.to_dict(),'target_asset_id':target,'offsets_s':offsets,'execution_intervals_s':intervals})
    for aid,intervals in motion.items():
        merged=[]
        for lo,hi in sorted(intervals):
            if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
            else:merged.append([lo,hi])
        result['motion_requests'][aid]['intervals_s']=merged
    yaw=float(result['camera_yaw_deg']);result['camera_yaw_knots']=[{'time_s':0.,'yaw_deg':yaw},{'time_s':30.,'yaw_deg':yaw}]
    result['motion_during_condition_plan']={'status':'solved','owner':'motion helper QA-06/QA-15','solver':cm.solver_signature(),
        'clock_mapping':'v3 150 frames at 15Hz + explicit parent offsets; original five half-overlap v7 crops',
        'static_camera_constructed':True,'plans':plans,'qa_predicates_changed':False,'native_admission_still_required':True}
    return result


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
            report=cm.verify_motion_candidate(solution,positions_m=positions,
                root_positions_m=roots if admissions else None,listener_position_m=scene['listener_m'])
            reports.append({'offset_s':offset,'qa_id':solution.qa_id,'report':report})
    return {'status':'pass' if all(x['report']['status']=='pass'for x in reports)else'fail',
        'source':'actual native attached mouth trajectories; original v3 verifier','reports':reports,'delivered_qa_gates_unchanged':True}

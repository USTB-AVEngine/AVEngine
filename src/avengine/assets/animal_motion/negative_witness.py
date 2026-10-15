"""Real moving comparator obligations for an animal's later stationary call.

Only the animal family chooses this clock and sound relation. Shared core route,
body, map and whole-track framing checks remain authoritative for proposals;
actual human gait, support, masks and original QA still run in native preparation.
"""
from copy import deepcopy
import math
import numpy as np


def construct_negative_companion(context, camera, standing, body, animal_actor, *,
                                  entity_id, animal_call_s, animal_duration_s):
    from avengine.construction.joint.routes import construct_route,ConstructionError
    from shapely.geometry import LineString
    start=float(animal_call_s)-.5
    end=float(animal_call_s)+float(animal_duration_s)+.8
    speed=.85
    length=speed*(end-start-.3)
    origin=np.asarray(standing,float).copy()
    animal_line=LineString(np.asarray(animal_actor.positions_m)[:,[0,2]])
    for heading in np.arange(16)*math.pi/8:
        finish=origin.copy();finish[[0,2]]+=length*np.array([math.cos(heading),math.sin(heading)])
        if animal_line.distance(LineString([origin[[0,2]],finish[[0,2]]]))<1.1:continue
        try:
            actor=construct_route(context,[origin,finish],species='human',speed_m_s=speed,
                start_s=start,camera=camera,entity_id=entity_id,body=body,required_framing='outside')
        except ConstructionError:continue
        delta=np.asarray(actor.positions_m)-camera.position_m
        bearings=np.degrees(np.arctan2(delta@camera.right,delta@camera.forward))
        if np.min(np.abs(bearings))<68. or np.min(np.linalg.norm(delta[:,[0,2]],axis=1))<1.6:continue
        ray=getattr(context,'floor_raycaster',None)
        if ray is not None:
            hit,_=ray.first_hit([origin+[0.,.2,0.]],[[0.,-1.,0.]])
            if not np.isfinite(hit[0])or not 0<=hit[0]<=.85:continue
            floor=float(origin[1]+.2-hit[0])
            if abs(floor-context.walkable_map.floor_reference)>.03:continue
            actor.positions_m[:,1]=floor
            actor.metadata['waypoints_m']=[[float(p[0]),floor,float(p[2])]for p in(origin,finish)]
            actor.metadata['original_scan_root_floor_m']=floor
        actor.metadata['animal_static_negative_comparator']={
            'animal_call_s':float(animal_call_s),'animal_duration_s':float(animal_duration_s),
            'minimum_abs_root_azimuth_deg':float(np.min(np.abs(bearings))),
            'required_native_901_pose_and_whole_body_outside':True}
        return actor
    raise ConstructionError('animal_negative_comparator_finite_route_domain_empty')


def companion_motion_request(actor, camera, walkable_map, speech_start_s, speech_duration_s):
    """Bind the exact shared-core clock to the owner's admitted human executor."""
    from avengine.assets.roster_motion.navmesh_execution import REVISION
    slot={'start_s':float(speech_start_s),'end_s':float(speech_start_s+speech_duration_s),'qa15':False}
    route={'reflect_walk':True,'joint_exact_route':True,
        'positions_xz_m':np.asarray(actor.positions_m)[:,[0,2]].tolist(),
        'travel_yaws_rad':np.asarray(actor.headings_rad).tolist(),
        'speed_m_s':float(actor.metadata['speed_m_s']),'speed_range_m_s':[.8,1.4],
        'species':'human','template':'animal_static_negative_comparator',
        'emission_slot':slot,'emission_slots':[slot],
        'walkable_map_path':str(walkable_map.path),'walkable_map_room_id':walkable_map.metadata['room_id'],
        'walkable_map_version':walkable_map.metadata['version'],'walkable_map_bypassed':False,
        'listener_m':list(camera.position_m),'camera_yaw_deg':camera.yaw_deg,
        'minimum_camera_distance_m':1.6,'candidate_count':1,'candidate_trials':[],
        'selected_attempt':'shared_core_exact_animal_negative_comparator','require_rest_poses_in_frame':False}
    return {'action':'walk','speed_m_s':float(actor.metadata['speed_m_s']),
        'scene_speed_range_m_s':[.8,1.4],'intervals_s':deepcopy(actor.metadata['walk_intervals_s']),
        'motion_execution_revision':REVISION,'motion_quartet_route':route,
        'require_clear_direct_path_during_sound':False,'maximum_native_route_attempts':1,
        'allow_registered_walk_tts':False}


def verify_negative_comparator(plan, animal_id, companion_id, call_s, duration_s, *, authority='planned_roots'):
    from avengine.construction.joint import CheckResult,track_framing_check
    from avengine.qa.unified_catalog import (_motion_series,motion_state_during_audible_window,
        moved_at_any_frame,distractors_equal_gold)
    actors={}
    for aid in(animal_id,companion_id):
        points=np.asarray(plan.actors[aid].positions_m)[:900:2].tolist()
        speeds,moving=_motion_series(points,frame_rate_hz=15.)
        actors[aid]={'root_positions_m':points,'moving':moving}
    facts={'time':{'frame_count':450,'frame_rate_hz':15.,'sample_rate_hz':16000,'duration_seconds':30.},'actors':actors}
    event={'event_id':'planned_later_animal_call','actor_id':animal_id,'start_s':call_s,
           'end_s':call_s+duration_s,'start_frame':math.floor(call_s*15),'end_frame':math.ceil((call_s+duration_s)*15)}
    states={aid:motion_state_during_audible_window(facts,{**event,'actor_id':aid})for aid in actors}
    begin=event['end_frame'];end=math.floor(min(26.6,call_s+duration_s+2.5)*15)
    after={aid:moved_at_any_frame(actor['moving'][begin:end])for aid,actor in actors.items()}
    during_ok=states[animal_id]['moving']is False and states[companion_id]['moving']is True
    after_ok=after[animal_id]is False and after[companion_id]is True
    for family,ok,values in [('QA-06',during_ok,{aid:s['moving']for aid,s in states.items()}),('QA-17',after_ok,after)]:
        separated=not distractors_equal_gold(values,animal_id)
        yield CheckResult('animal:static_negative:'+family,'pass'if ok and separated else'fail',
            'unchanged_catalog_motion_and_real_distractor_predicates',
            evidence={'planned_real_actor_values':values,'source_call_s':[call_s,call_s+duration_s],
                      'input_authority':authority,'native_truth_and_full_pose_still_required':authority=='planned_roots'})
    yield track_framing_check(plan.camera,plan.actors[companion_id],required='outside')


def native_negative_companion_checks(video, entries, plan, animal_id, companion_id, call_s, duration_s):
    from dataclasses import replace
    from avengine.construction.joint import CheckResult,native_full_mesh_sweep
    from avengine.construction.joint.geometry import framing_vertices
    observed={}
    byid={entry['declaration']['asset_id']:entry for entry in entries}
    for aid,actor in plan.actors.items():
        record=byid[aid].get('motion_episode',{})
        matrices=np.asarray([r['body_to_world_m']for r in record.get('native_30Hz_readbacks',[])],float)
        if matrices.shape!=(901,4,4):
            return (CheckResult('animal:negative_native_clock','fail','complete_native_moving_comparator_required'),)
        observed[aid]=replace(actor,positions_m=matrices[:,:3,3])
    actual=replace(plan,actors=observed)
    checks=list(verify_negative_comparator(actual,animal_id,companion_id,call_s,duration_s,
                                          authority='actual_native_901_roots'))
    # This is the union of every compiled posed full-skin world AABB, not a
    # species proxy. Expand for the separately checked2mm root/yaw replay error.
    bounds=np.asarray(byid[companion_id]['motion_episode'].get('full_skin_bounds_m'),float)
    valid=bounds.shape==(2,3)and np.isfinite(bounds).all()
    state='unknown';margin=None
    if valid:
        low,high=bounds[0]-.005,bounds[1]+.005
        corners=np.array([[x,y,z]for x in(low[0],high[0])for y in(low[1],high[1])for z in(low[2],high[2])])
        sensor=video.sim._sensors['rgb']._sensor_object
        view=np.linalg.inv(np.asarray(sensor.node.absolute_transformation(),float))
        projection=np.asarray(sensor.render_camera.projection_matrix,float)
        state,margin=framing_vertices(corners,view,projection,video.width,video.height)
    checks.append(CheckResult('animal:negative_native_whole_skin_outside',
        'pass'if state=='outside'else'fail','all_posed_full_skin_world_union_with_actual_native_camera',
        evidence={'framing':state,'edge_margin_px':margin if margin is not None and np.isfinite(margin)else None,
                  'compiled_pose_frames':901,'world_union_expansion_m':.005,'actual_901_root_and_yaw_checked_separately':True}))
    if all(check.status=='pass'for check in checks):checks.extend(native_full_mesh_sweep(video,entries))
    return tuple(checks)

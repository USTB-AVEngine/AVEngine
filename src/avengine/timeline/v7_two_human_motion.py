"""Two independently walking speakers, with no unsupported resting turns.

Each person walks one complete straight bout.  Opposite, parallel lanes keep
people apart at every frame and give preselected opposite radial answers.
Complete stationary replies cover the remaining parent timeline, outside the
post-sound certification intervals.  All answers remain native readbacks.
"""
from copy import deepcopy
import math
import numpy as np
from shapely.geometry import LineString, shape

FAMILIES=('QA-06','QA-15','QA-16','QA-17')
REVISION='two-walking-speakers-disjoint-bouts-v1'


def single_bout_at(route, *, start_s=2.6, offset_s=0., reverse=False, shift_xz=(0.,0.), duration_s=None):
    result=deepcopy(route);old=route['emission_slot'];delta=float(start_s)-float(old['start_s'])
    t=np.arange(901)/30.;source=np.asarray(route['positions_xz_m'],float)
    points=np.column_stack([np.interp(t-delta,t,source[:,i])for i in (0,1)])
    if reverse:points=source[0]+source[-1]-points
    points+=np.asarray(shift_xz)
    heading=float(route['short_window_walk']['heading_rad'])+(math.pi if reverse else 0.)
    slot=deepcopy(old)
    for key in ('start_s','end_s','post_until_s'):
        if slot.get(key)is not None:slot[key]=float(slot[key])+delta
    slot['start_s']=float(start_s)
    slot['offset_s']=float(offset_s)
    if duration_s is not None:slot['end_s']=float(start_s)+math.ceil(float(duration_s)*10-1e-9)/10
    if reverse:slot['branch']='farther'if old['branch']=='nearer'else'nearer'
    for key in ('leg','window_frames'):slot[key]=[round(v+delta*30)for v in old[key]]
    radial=np.linalg.norm(points-np.asarray(route['listener_m'])[[0,2]],axis=1)
    i,j=round(slot['start_s']*30),round(slot['end_s']*30);trim=round(.2*(j-i))
    slot['radial_change_m']=float(radial[j-trim]-radial[i+trim])
    window=deepcopy(route['short_window_walk'])
    for key in ('walk_start_s','walk_end_s','silent_query_until_s'):window[key]=float(window[key])+delta
    window['emission_s']=[slot['start_s'],slot['end_s']];window['heading_rad']=heading
    result.update(positions_xz_m=points.tolist(),travel_yaws_rad=[heading]*901,
        emission_slot=slot,emission_slots=[slot],short_window_walk=window,
        prefer_planned_placement=True,require_rest_poses_in_frame=True,
        human_rest_turns=[],human_motion_event_count=1,construction='one_straight_bout_no_turn')
    return result


def negative_control_bystander(room,pathfinder,listener,yaw_deg,routes):
    """A real silent body's root supplies the stock wrong-origin control.

    The existing audio code admits every registered drawable as a wrong
    origin, including a silent entity's measured root (no invented sound).
    """
    from avengine.assets.roster_motion.reflect_navmesh import native_free_predicate
    from avengine.assets.roster_motion.straight_route import rest_framing
    from avengine.construction.joint.routes import minimum_track_distance
    floor=float(room['measured_floor_y_m']);listener=np.asarray(listener,float);L=listener[[0,2]]
    polygon=shape(room['floor_polygon']);pred=native_free_predicate(pathfinder,polygon,floor,listener,species='human',camera_margin=1.5)
    tracks=[np.column_stack((np.asarray(r['positions_xz_m'])[:,0],np.full(901,listener[1]+.3),np.asarray(r['positions_xz_m'])[:,1]))for r in routes]
    roots=[np.column_stack((t[:,0],np.full(901,floor),t[:,2]))for t in tracks]
    choices=[]
    for relative in (30.,-30.,33.,-33.,35.,-35.,28.,-28.,65.,-65.,90.,-90.,130.,-130.):
        angle=math.radians(float(yaw_deg)+relative);direction=np.array([math.sin(angle),-math.cos(angle)])
        for radius in (3.8,4.2,3.4,4.8,2.8,2.,1.6):
            xz=L+radius*direction
            if not pred(xz):continue
            framing=rest_framing(xz,L,yaw_deg)
            if framing=='edge':continue
            depth=float(np.dot(xz-L,np.array([math.sin(math.radians(yaw_deg)),-math.cos(math.radians(yaw_deg))])))
            if framing=='inside'and depth<2.5:continue
            point=np.array([xz[0],floor,xz[1]])
            if any(minimum_track_distance(t,np.tile(point,(901,1)))<1.1 for t in roots):continue
            wrong=point-listener;minimum=180.
            for track in tracks:
                wanted=track-listener;cos=np.sum(wanted*wrong,axis=1)/(np.linalg.norm(wanted,axis=1)*np.linalg.norm(wrong))
                minimum=min(minimum,float(np.degrees(np.arccos(np.clip(cos,-1,1))).min()))
            if minimum<24.:continue
            choices.append(((framing=='inside',minimum,-radius),{'root_m':point.tolist(),'minimum_planned_wrong_angle_deg':minimum,
                'frame_state':framing,'relative_azimuth_deg':relative,'source_position_policy':'silent registered drawable root, matching original audio negative-control code'}))
    return max(choices,key=lambda item:item[0])[1]if choices else None


def paired_corridors(seed_route,room,pathfinder,*,first_branch='nearer',second_duration_s=1.6,require_negative_control=False,pair_mode='opposing'):
    """Finite analytic lane offsets; no second actor can cross the first."""
    from avengine.assets.roster_motion.reflect_navmesh import native_free_predicate
    from avengine.assets.roster_motion.straight_route import frame_rest_poses
    from avengine.construction.joint.routes import minimum_track_distance
    from avengine.construction.joint.context import camera_legal
    from avengine.construction.joint.types import CameraPose
    listener=np.asarray(seed_route['listener_m']);L=listener[[0,2]]
    local_start=round(float(seed_route['emission_slot']['start_s'])-float(seed_route['emission_slot']['offset_s']),1)
    a=single_bout_at(seed_route,start_s=local_start,offset_s=0.,reverse=seed_route['emission_slot']['branch']!=first_branch)
    p=np.asarray(a['positions_xz_m']);direction=p[-1]-p[0];direction/=np.linalg.norm(direction)
    lateral=np.array([-direction[1],direction[0]])
    floor=float(room['measured_floor_y_m']);polygon=shape(room['floor_polygon']);coverage=polygon.buffer(.002)
    pred=native_free_predicate(pathfinder,polygon,floor,listener,camera_margin=2.6,species='human')
    if not coverage.covers(LineString(p)) or not all(pred(v) for v in np.unique(p,axis=0)):return
    for separation in (1.25,1.4,1.6,1.8):
        for side in (1.,-1.):
            if pair_mode not in {'opposing','following'}:raise ValueError('unknown planned human pair mode')
            if pair_mode=='following' and side<0:continue
            shift=(-separation*direction)if pair_mode=='following'else(side*separation*lateral)
            b=single_bout_at(a,start_s=15.+local_start,offset_s=15.,reverse=pair_mode=='opposing',shift_xz=shift,duration_s=second_duration_s)
            q=np.asarray(b['positions_xz_m'])
            if not coverage.covers(LineString(q)):continue
            if not all(pred(v)for v in np.unique(q,axis=0)):continue
            continuous_distance=minimum_track_distance(np.column_stack((p[:,0],np.full(len(p),floor),p[:,1])),np.column_stack((q[:,0],np.full(len(q),floor),q[:,1])))
            if continuous_distance<1.2:continue
            rest=[p[0],p[-1],q[0],q[-1]];target=np.mean(rest,axis=0)-L
            yaw,framing=frame_rest_poses(rest,L,math.degrees(math.atan2(target[0],-target[1])))
            if framing['states']!=['inside']*4:continue
            if not camera_legal(room,CameraPose(tuple(listener),yaw,width=1280,height=720))[0]:continue
            radii=np.linalg.norm(q-L,axis=1);slot=b['emission_slot'];i,j=round(slot['start_s']*30),round(slot['end_s']*30)
            trim=round(.2*(j-i));v=radii[i+trim:j-trim+1];sign=-1 if slot['branch']=='nearer'else 1
            if sign*(v[-1]-v[0])<.5 or np.any(sign*np.diff(v)<-.005):continue
            control=negative_control_bystander(room,pathfinder,listener,yaw,(a,b))if require_negative_control else None
            if require_negative_control and control is None:continue
            for route in (a,b):route.update(camera_yaw_deg=yaw,rest_pose_framing=framing)
            yield a,b,{'lane_separation_m':separation,'side':side,'minimum_pair_distance_m':continuous_distance,'rest_framing':framing,'negative_control':control,'pair_mode':pair_mode}


def two_human_motion_template(base_row,room,catalog,semantic_manifest,source_records,pathfinder,*,attempt=0,contract=None):
    from avengine.timeline.v7_motion_quartet_template import motion_quartet_template,authored_pairs,MotionTemplateUnavailable
    from avengine.assets.roster_motion.human_event_route import find_human_seed_route
    from avengine.timeline.v7_motion_during_conditions import prepare_motion_during_row,verify_planned_trajectories
    from avengine.timeline.v7_registered_sampling import semantic_record
    from avengine.timeline.v7_export_scene_policy import measured_actor_metrics
    from avengine.construction.joint.families.motion_human import motion_certificate,certificate_record
    import soundfile as sf
    pairs=[p for p in authored_pairs(semantic_manifest)
           if 1.0<=float(p[1]['synthesis_readback']['duration_s'])<=float(p[0]['synthesis_readback']['duration_s'])<=1.8
           and float(p[0]['synthesis_readback']['duration_s'])>=1.2]
    if not pairs:raise MotionTemplateUnavailable('no complete short two-voice pair for independent walking bouts')
    pair=pairs[(int(base_row['scene_seed'])+attempt)%len(pairs)]
    duration_a=float(pair[0]['synthesis_readback']['duration_s'])
    local_start=round(4.4-math.ceil(duration_a*10-1e-9)/10,1)
    seed_row=deepcopy(base_row);seed_row.update(human_motion_event_count=1,human_motion_require_all_families=True,
        human_motion_require_two_walkers=True,human_event_start_local_s=local_start,human_post_motion_s=.6,human_minimum_motion_s=.8,human_query_until_local_s=7.4,
        motion_semantic_pair_ids=[s['sound_asset_id']for s in pair],motion_template_focus='M1')
    base=motion_quartet_template(seed_row,room,catalog,semantic_manifest,source_records,pathfinder,attempt=attempt,contract=contract,source_type='human_speech')
    first,second=base['semantic_human_asset_ids'];bystander=base['asset_ids'][2]
    base['asset_ids']=[first,second,bystander];base['silent_asset_ids']=[bystander]
    measured=[];byasset={a['asset_id']:a for a in catalog}
    for aid,sound in zip((first,second),pair):
        record=semantic_record(sound,'F'if byasset[aid]['visual_gender']=='female'else'M')
        samples,rate=sf.read(record['normalized_path'],dtype='float64');nz=np.flatnonzero(samples!=0.)
        if rate!=16000 or samples.ndim!=1 or not len(nz):raise MotionTemplateUnavailable('complete short speech must retain 16kHz mono activity')
        record.setdefault('screens',{})['automatic_activity']={'activity_intervals_samples':[[int(nz[0]),int(nz[-1]+1)]],'method':'measured exact complete source support'}
        measured.append(record)
    duration_a=float(pair[0]['synthesis_readback']['duration_s']);duration_b=float(pair[1]['synthesis_readback']['duration_s'])
    pair_mode=base_row.get('human_motion_pair_mode','opposing')
    desired='nearer'if int(base_row['scene_seed'])%2 else'farther'
    errors=[]
    import time
    deadline=time.process_time()+30.
    def partner_possible(candidate):
        return next(paired_corridors(candidate,room,pathfinder,first_branch=desired,second_duration_s=duration_b,require_negative_control=True,pair_mode=pair_mode),None)is not None
    for geometry_attempt in range(2):
        try:
            seed_route=(base['motion_requests'][first]['motion_quartet_route']if geometry_attempt==0 else
                find_human_seed_route(room,pathfinder,int(base_row['scene_seed']),attempt+geometry_attempt,
                    utterance_s=duration_a,local_start_s=local_start,post_motion_s=.6,candidate_validator=partner_possible,cpu_deadline=deadline))
        except ValueError as exc:errors.append(str(exc));continue
        for route_a,route_b,layout in paired_corridors(seed_route,room,pathfinder,first_branch=desired,second_duration_s=duration_b,require_negative_control=True,pair_mode=pair_mode):
            row=deepcopy(base);row['human_motion_event_count']=2;row['human_motion_strategy']='two_walkers'
            row['motion_quartet_template_revision']=REVISION
            row['phase2_source_event_schedule_s']={first:[local_start,11.,22.],second:[7.5,15.+local_start,27.5]}
            row['semantic_start_offsets_s']=[local_start,7.5];row['motion_source_start_s']={first:local_start,second:7.5}
            row['listener_m']=route_a['listener_m'];row['camera_yaw_deg']=route_a['camera_yaw_deg']
            request_base={k:v for k,v in base['motion_requests'][first].items()if k not in ('motion_during_condition_plan_for_fit','straight_motion_windows_s','sounding_straight_windows_s','distance_trend','distance_trend_intervals_s')}
            row['motion_requests']={aid:{**deepcopy(request_base),'motion_quartet_route':route,'speed_m_s':route['speed_m_s'],'intervals_s':[[0.,30.]],'maximum_native_route_attempts':1}
                for aid,route in ((first,route_a),(second,route_b))}
            plans=[];requests=[];verifications=[]
            tracks={aid:np.column_stack((np.asarray(route['positions_xz_m'])[:,0],np.full(901,row['listener_m'][1]+.3),np.asarray(route['positions_xz_m'])[:,1]))for aid,route in ((first,route_a),(second,route_b))}
            croot=np.asarray(layout['negative_control']['root_m']);tracks[bystander]=np.tile(croot,(901,1))
            valid=True
            for aid,route,starts in ((first,route_a,{first:local_start,second:7.5}),(second,route_b,{first:7.,second:local_start})):
                slot=route['emission_slot'];offset=slot['offset_s'];end_local=route['short_window_walk']['walk_end_s']-offset-.1
                reqs=[{'qa_id':qa,'branch':branch,'target_asset_id':aid,'speed_m_s':route['speed_m_s'],
                    'speed_range_m_s':route['speed_range_m_s'],'minimum_motion_s':.8,'end_hold_s':10.-(7.4 if aid==first else 6.9),
                    'offsets_s':[offset],'reserve_tail_s':.5,'wet_tail_planning_bound_s':None,'minimum_distance_change_m':.5}
                    for qa,branch in [('QA-06','moving'),('QA-15',slot['branch']),('QA-16',None),('QA-17','yes')]]
                sub=deepcopy(row);sub.pop('motion_during_condition_plan',None);sub['motion_during_condition_requests']=reqs;sub['motion_source_start_s']=starts
                try:sub=prepare_motion_during_row(sub,catalog,list(source_records)+measured)
                except ValueError as exc:errors.append(str(exc));valid=False;break
                certificate=motion_certificate(sub['motion_during_condition_plan'],tracks,row['listener_m'])
                verdict={family:all(c.status=='pass'for c in certificate.checks if c.evidence.get('family')==family)for family in FAMILIES}
                if {c.evidence.get('family')for c in certificate.checks}!=set(FAMILIES):valid=False;break
                if set(verdict)!=set(FAMILIES)or not all(verdict.values()):errors.append(str(verdict));valid=False;break
                plans.extend(sub['motion_during_condition_plan']['plans']);requests.extend(reqs)
                row['motion_requests'][aid]['motion_during_condition_plan_for_fit']=deepcopy(sub['motion_during_condition_plan'])
                verifications.append({'asset_id':aid,'offset_s':offset,'planned_branch':slot['branch'],'families':verdict})
            if not valid:continue
            speed_checks=[]
            for offset in (0.,5.,10.,15.,20.):
                actors={aid:{'asset_id':aid,'species_id':'human','root_positions_m':track[round(offset*30):round((offset+10)*30):2].tolist()}for aid,track in tracks.items()}
                speed_checks.append({'offset_s':offset,'metrics':measured_actor_metrics(actors,15.)})
            if any(metric['status']!='pass'for check in speed_checks for metric in check['metrics']):errors.append('half-overlap crop isolates a slow start/stop ramp');continue
            plan=deepcopy(sub['motion_during_condition_plan']);plan['plans']=plans
            row['motion_during_condition_plan']=plan;row['motion_during_condition_requests']=requests
            row['joint_motion_witness_certificate']=certificate_record(motion_certificate(plan,tracks,row['listener_m']))
            row['joint_common_geometry_status']='pending explicit native-NavMesh fallback API; cached map is a construction preference'
            row['camera_yaw_knots']=[{'time_s':0.,'yaw_deg':row['camera_yaw_deg']},{'time_s':30.,'yaw_deg':row['camera_yaw_deg']}]
            row['motion_quartet_planned_verification']={f:True for f in FAMILIES};row['human_motion_event_planned_verification']=verifications
            row['planned_crop_speed_checks']=speed_checks
            row['primary_motion_speech_anchors_s']={first:route_a['emission_slot']['start_s'],second:route_b['emission_slot']['start_s']}
            row['motion_wrong_emitter_control']={**layout['negative_control'],'asset_id':bystander}
            row['native_placement_preferences_m']={aid:[float(track[0,0]),float(room['measured_floor_y_m']),float(track[0,2])]for aid,track in tracks.items()}
            row['scene_demand'].update(constructor=REVISION,condition_families={'motion_quartet':list(FAMILIES)},required_actual_candidate_families=list(FAMILIES))
            row['motion_quartet_construction'].update(revision=REVISION,route=route_a,routes={first:route_a,second:route_b},
                independent_walking_speakers=2,independent_emissions=6,layout=layout,
                no_turns=True,planned_verification=verifications,source_events_spread_over_parent=True)
            return row
    raise MotionTemplateUnavailable('bounded two-person corridor construction exhausted: '+' | '.join(errors[:6]))

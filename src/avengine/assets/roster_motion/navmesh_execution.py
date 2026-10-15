"""Scene-speed playback and smooth native NavMesh waypoint loops.

Raw authored clips keep their intake limits. Each executed root is independently
snapped and the original full-skin/foot/native renderer gates still certify it.
"""
from copy import deepcopy
from types import SimpleNamespace
import math
import numpy as np
from avengine.assets.roster_motion.distance_constraints import matches_radial_window
from itertools import islice
from shapely.geometry import shape,LineString,Point,box
REVISION='species-scene-speeds-native-curved-waypoints-v1'
SPEED_RANGES={'human':(.8,1.4),'dog':(.5,1.2),'cat':(.3,.8),'slide':(.2,.5)}

def motion_yaw(record,index):
    values=record.get('yaw_rad_per_frame')
    return float(values[index] if values is not None else record['yaw_rad'])

def prepare_scene_motion(row,catalog):
    if row.get('motion_quartet_template_revision'):return row
    if row.get('motion_execution_revision')!=REVISION:return row
    row=deepcopy(row);assets={a['asset_id']:a for a in catalog}
    for number,(aid,request)in enumerate(sorted(row.get('motion_requests',{}).items())):
        mode=request.get('animal_motion_admission',{}).get('mode');species=assets[aid].get('species_id') or 'human'
        kind='slide'if mode=='slide_no_animation'else species
        if kind not in SPEED_RANGES:raise ValueError('scene locomotion speed species missing')
        speed=float(np.random.default_rng(int(row['scene_seed'])+97001+number).uniform(*SPEED_RANGES[kind]))
        request.update(speed_m_s=speed,motion_execution_revision=REVISION,scene_speed_range_m_s=list(SPEED_RANGES[kind]),require_clear_direct_path_during_sound=False)
        for condition in row.get('motion_during_condition_requests',[]):
            if condition['target_asset_id']==aid:
                condition['speed_m_s']=speed
                condition['speed_range_m_s']=list(SPEED_RANGES[kind])
        if row.get('source_classes',{}).get(aid)==['footsteps']:request.pop('distance_trend',None)
    return row

def direct_visible(video,listener,source):
    listener=np.asarray(listener,float);delta=np.asarray(source,float)-listener;distance=float(np.linalg.norm(delta))
    if distance<=.05:return False
    ignored={int(a['object'].object_id)for a in video.actors}
    ray=video.hs.geo.Ray(video.mn.Vector3(listener),video.mn.Vector3(delta/distance))
    return not any(int(hit.object_id)not in ignored for hit in video.sim.cast_ray(ray,distance-.05).hits)

def sounding_at(scene,endpoint,time_s):
    return any(e['endpoint_id']==endpoint and e['start_tick']/10<=time_s<e['end_tick']/10 for e in scene['events'])

def curved_route_proposals(video,entry,scene,distance,local_forward,request):
    """Closed smooth waypoint routes; search alternate centres around obstacles."""
    if request.get('straight_motion_windows_s'):
        from avengine.assets.roster_motion.straight_window_routes import straight_window_routes
        proposals=list(islice(straight_window_routes(video,entry,scene,distance,local_forward,request),max(24,int(request.get('route_ranking_candidate_budget',64)))))
        proposals.sort(key=lambda p:p.receipt.get('minimum_native_obstacle_clearance_m',0.),reverse=True)
        for rank,proposal in enumerate(proposals,1):
            proposal.receipt.update(native_clearance_rank=rank,ranked_valid_route_candidates=len(proposals),maximum_full_gate_routes=max(24,int(request.get('maximum_native_route_attempts',32))))
            yield proposal
        return
    poly=shape(scene['production_scenario']['motion_floor_polygon']);start=np.asarray(entry['applied_root_input'][:3,3],float)
    radius_body=float(request.get('route_body_margin_m',.32 if entry['declaration']['kind']=='human'else .25))
    free=poly.buffer(-radius_body);obstacles=[]
    for actor in video.actors:
        if actor is entry or actor.get('phase2_layout_qualified')is False:continue
        root=actor['applied_root_input'];v=np.asarray(actor['original_mesh'].vertices)@root[:3,:3].T+root[:3,3]
        obstacles.append(box(v[:,0].min()-radius_body,v[:,2].min()-radius_body,v[:,0].max()+radius_body,v[:,2].max()+radius_body))
    random=np.random.default_rng(int(scene['seed'])+741);local_angle=math.atan2(local_forward[2],local_forward[0]);proposal=0
    for radius in (1.6,1.35,1.1,.85,.65):
        centres=free.buffer(-radius)
        if centres.is_empty:continue
        x0,z0,x1,z1=centres.bounds;candidates=[centres.representative_point()]
        candidates += [Point(x,z)for x in np.linspace(x0,x1,9)for z in np.linspace(z0,z1,9)]
        candidates=[p for p in candidates if centres.covers(p)];random.shuffle(candidates)
        for centre in candidates:
            for angle in random.permutation(np.linspace(0,2*math.pi,12,endpoint=False)):
                for sense in (-1,1):
                    theta=angle+sense*np.asarray(distance)/radius
                    positions=np.column_stack((centre.x+radius*np.cos(theta),np.full(len(theta),start[1]),centre.y+radius*np.sin(theta)))
                    line=LineString(positions[:,[0,2]])
                    if not free.covers(line)or any(line.intersects(o)for o in obstacles):continue
                    if np.min(np.linalg.norm(positions[:,[0,2]]-np.asarray(scene['listener_m'])[[0,2]],axis=1))<.6:continue
                    snaps=np.asarray([video.pathfinder.snap_point(p)for p in positions]);errors=np.linalg.norm((snaps-positions)[:,[0,2]],axis=1)
                    if not np.isfinite(snaps).all()or float(errors.max())>.002:continue
                    positions[:,[0,2]]=snaps[:,[0,2]]
                    # Actual tangent of the 30Hz executed track; pauses hold yaw.
                    tangents=np.column_stack((-sense*np.sin(theta),sense*np.cos(theta)))
                    yaws=np.unwrap(local_angle-np.arctan2(tangents[:,1],tangents[:,0]))
                    if request.get('distance_trend'):
                        radii=np.linalg.norm(positions-np.asarray(scene['listener_m']),axis=1);intervals=request.get('distance_trend_intervals_s',[[0,10]])
                        sign=-1 if request['distance_trend']=='approaching'else 1;valid=True
                        for lo,hi in intervals:
                            values=radii[max(0,math.floor(lo*30)):min(len(radii),math.ceil(hi*30)+1)]
                            if not matches_radial_window(values,sign,float(request.get('minimum_distance_change_m',.5))):valid=False;break
                        if not valid:continue
                    # Certify native connectivity between the sampled waypoints.
                    links=[]
                    for i,j in zip(np.linspace(0,len(snaps)-1,9,dtype=int)[:-1],np.linspace(0,len(snaps)-1,9,dtype=int)[1:]):
                        path=video.hs.ShortestPath();path.requested_start=snaps[i];path.requested_end=snaps[j]
                        if not video.pathfinder.find_path(path):break
                        links.append({'start_frame':int(i),'end_frame':int(j),'native_geodesic_m':float(path.geodesic_distance)})
                    if len(links)!=8:continue
                    proposal+=1
                    yield SimpleNamespace(positions=positions,yaws=yaws,receipt={'revision':REVISION,'proposal':proposal,
                        'route_kind':'smooth closed native NavMesh waypoint loop','radius_m':radius,'centre_xz_m':[centre.x,centre.y],
                        'speed_m_s':float(request['speed_m_s']),'root_samples':len(positions),'maximum_horizontal_snap_error_m':float(errors.max()),
                        'native_waypoint_connections':links,'heading_follows_route_tangent':True,'native_full_pose_checks_still_required':True})


def human_distance_driven_clock(source_times,double_support,speed,intervals,nominal=1.2,speed_range=(.8,1.4),timed_route=None):
    """Fit complete observed steps, then drive animation by translated distance.

    Extending a requested movement envelope to the next observed two-foot pause
    changes scene construction only. Each original pose still goes through IK,
    exact skin collision, contacts and native readback.
    """
    times=np.arange(901)/30.;duration=float(source_times[-1]-source_times[0])
    phase0=float(min(double_support));velocity=np.zeros(len(times));moving=np.zeros(len(times),bool)
    if timed_route is not None and timed_route.get('single_pass'):
        from avengine.assets.roster_motion.timed_straight_clock import single_pass_clock
        distance,velocity=single_pass_clock(times,speed,timed_route.get('turn_speed_m_s',.4));desired=phase0+distance[-1]/nominal
        endings=sorted(float(p)+cycle*duration for cycle in range(math.ceil(desired/duration)+2)for p in double_support if float(p)+cycle*duration>=desired-1e-9)
        target=endings[0];target_distance=(target-phase0)*nominal
        lo,hi=7.9,9.7
        if single_pass_clock(times,speed,timed_route.get('turn_speed_m_s',.4),hi)[0][-1]<target_distance:raise ValueError('complete observed step cannot finish before the next sounding window; choose another admitted gait')
        for _ in range(45):
            mid=(lo+hi)/2
            if single_pass_clock(times,speed,timed_route.get('turn_speed_m_s',.4),mid)[0][-1]<target_distance:lo=mid
            else:hi=mid
        distance,velocity=single_pass_clock(times,speed,timed_route.get('turn_speed_m_s',.4),hi)
        ratio=target_distance/distance[-1];distance*=ratio;velocity*=ratio
        return times,velocity>1e-8,velocity,distance,phase0+distance/nominal,{'method':'single straight pass; exact scene speed during speech; complete-step quiet finish','complete_step_source_phase_s':target,'distance_clock_ratio':ratio,'executed_motion_end_s':hi,'emission_speed_m_s':speed,'nominal_source_speed_m_s':nominal,'maximum_stable_speed_m_s':float(max(velocity)),'maximum_clock_distance_error_s':0.}

    if timed_route is not None:
        from avengine.assets.roster_motion.timed_straight_clock import timed_straight_clock
        distance,velocity,profile=timed_straight_clock(times,speed,straight_start_s=timed_route['straight_start_s'],straight_end_s=timed_route['straight_end_s'],turn_speed_m_s=timed_route['turn_speed_m_s'])
        return times,np.ones(len(times),bool),velocity,distance,phase0+distance/nominal,{**profile,'maximum_clock_distance_error_s':0.,'nominal_source_speed_m_s':nominal,'maximum_stable_speed_m_s':float(np.max(velocity))}
    if intervals==[[0.,30.]] or intervals==[[0,30]]:
        edge=np.clip(times/.25,0,1);velocity=speed*edge**2*(3-2*edge)
        distance=np.r_[0.,np.cumsum((velocity[1:]+velocity[:-1])/60)]
        return times,np.ones(len(times),bool),velocity,distance,phase0+distance/nominal,{
            'method':'exact integrated root distance; continuous walking through clip end',
            'nominal_source_speed_m_s':nominal,'maximum_stable_speed_m_s':float(np.max(velocity)),
            'maximum_clock_distance_error_s':0.,'clip_end_requires_planted_pause':False}
    base=phase0;executed=[]
    for number,(lo,hi)in enumerate(intervals):
        lo=float(lo);hi=float(hi);limit=float(intervals[number+1][0])if number+1<len(intervals) else 30.
        if not 0<=lo<hi<=limit<=30:raise ValueError('overlapping or invalid real-speed human intervals')
        active=(times>=lo)&(times<hi);edge=np.clip(np.minimum(times-lo,hi-times)/.25,0,1)
        original=np.where(active,speed*edge**2*(3-2*edge),0.);desired=base+float(np.sum((original[1:]+original[:-1])*np.diff(times)/2))/nominal
        cycles=math.ceil((base+35)/duration)+2
        endings=sorted({float(p)+cycle*duration for cycle in range(cycles)for p in double_support if float(p)+cycle*duration>=desired-1e-9})
        accepted=None
        for target in endings:
            end=math.ceil((lo+(target-base)*nominal/speed+.25)*30-1e-9)/30
            if end>limit+1e-9:break
            active=(times>=lo)&(times<end);edge=np.clip(np.minimum(times-lo,end-times)/.25,0,1)
            v=np.where(active,speed*edge**2*(3-2*edge),0.);length=float(np.sum((v[1:]+v[:-1])*np.diff(times)/2))
            if length<=0:continue
            ratio=(target-base)*nominal/length;steady=speed*ratio
            if not float(speed_range[0])-1e-9<=steady<=float(speed_range[1])+1e-9:continue
            accepted=(end,v*ratio,active,target,steady);break
        if accepted is None:raise ValueError('no complete observed double-support gait fits this real-speed envelope; construct a longer envelope')
        end,v,active,target,steady=accepted;velocity+=v;moving|=active
        executed.append({'requested_s':[lo,hi],'executed_s':[lo,end],'stable_speed_m_s':steady,
                         'start_source_phase_s':base,'end_source_phase_s':target})
        base=target
    distance=np.r_[0.,np.cumsum((velocity[1:]+velocity[:-1])/60)]
    clock=phase0+distance/nominal
    return times,moving,velocity,distance,clock,{'method':'source phase equals exact integrated root distance / admitted nominal stride speed',
        'nominal_source_speed_m_s':nominal,'intervals':executed,'pause_source_phase_s':base,
        'maximum_stable_speed_m_s':float(max(velocity)),
        'maximum_clock_distance_error_s':float(np.max(np.abs(clock-phase0-distance/nominal)))}

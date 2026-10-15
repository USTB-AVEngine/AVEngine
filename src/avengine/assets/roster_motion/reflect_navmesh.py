"""Reflection proposals on the original native NavMesh; no answer predicates."""
import math
from collections import Counter
from types import SimpleNamespace
import numpy as np
from shapely.geometry import Point, shape, LineString, box
from .reflect_walk import reflect_walk, schedule_emissions, SPECIES_SPEED_M_S


BODY_CLEARANCE_M={'human':.35,'dog':.25,'cat':.20,'slide':.20}     # same radii as the walkable map layers
CAMERA_MARGIN_M={'human':1.5}                                         # owner rule is about people; animals 1.0 m


def native_free_predicate(pathfinder, polygon, floor_y, listener, *, obstacles=(), camera_margin=None,
                          walkable_map=None, species='human'):
    # Claude 10-15: clearance and camera margin follow the body that walks (a cat was held to the human 35 cm /
    # 1.5 m and found no 2.5 m line in a normal room); construction preferences only, no answer depends on them.
    if camera_margin is None:camera_margin=CAMERA_MARGIN_M.get(species,1.)
    clearance_m=BODY_CLEARANCE_M.get(species,.35)
    listener=np.asarray(listener,float)[[0,2]]
    def is_free(xz):
        xz=np.asarray(xz,float)
        if walkable_map is not None and not walkable_map.contains(xz,species):return False
        if not np.isfinite(xz).all():return False
        if not polygon.covers(Point(*xz)) or np.linalg.norm(xz-listener)<camera_margin:return False
        if any(o.covers(Point(*xz)) for o in obstacles):return False
        p=np.array([xz[0],floor_y,xz[1]])
        snap=np.asarray(pathfinder.snap_point(p),float)
        if not np.isfinite(snap).all() or np.linalg.norm((snap-p)[[0,2]])>.002:return False
        return (not hasattr(pathfinder,'distance_to_closest_obstacle') or
                float(pathfinder.distance_to_closest_obstacle(snap,1.))>=clearance_m)
    def segment_is_free(a,b):
        if walkable_map is not None and not walkable_map.segment_free(a,b,species):return False
        line=LineString([a,b])
        if not polygon.covers(line) or line.distance(Point(*listener))<camera_margin:return False
        if any(line.intersects(o)for o in obstacles):return False
        # The cached cells cover the entire segment. The native point tests
        # remain independent at every emitted frame and look-ahead endpoint.
        return is_free(b)
    if walkable_map is not None:is_free.segment_is_free=segment_is_free
    return is_free



SHORT_WALK_LEAD_S=.5
SHORT_WALK_RAMP_S=.3


def _short_window_walk(walk,slot):
    """Stand -> walk one straight line around the emission -> stand (Claude 2026-10-15).

    The source walks only while the question needs motion: from SHORT_WALK_LEAD_S before the emission to
    SHORT_WALK_LEAD_S after the silent query interval (QA-16/17) or after the emission (QA-06/15), at the constant
    leg speed with smooth SHORT_WALK_RAMP_S start/stop ramps.  It walks ONE straight line through the planned
    emission start with the heading of travel there, so a straight-leg slot keeps its emission geometry (QA-15
    radial conditions are rechecked natively).  No turning while walking: following the reflected walk's corners
    (measured 10-15: ~100 deg in 0.2 s plus stop-and-go holds) swings the body around a planted foot, and the
    unchanged IK/pelvis gate failed at every placement; a straight replay of the same window passed with a 2.7 cm
    maximum pelvis drop (limit 8 cm).  The caller checks that the whole line is free.
    """
    fps=walk.fps;n=len(walk.positions)
    start=float(slot['start_s']);end=float(slot['end_s'])
    tail=slot.get('post_until_s') if (slot.get('qa17') or slot.get('qa16')) and slot.get('post_until_s') else end+.6
    k=min(n-1,int(round(start*fps)))
    if slot.get('leg') is not None:
        a,b=slot['leg'];w0=max(a/fps,start-SHORT_WALK_LEAD_S);w1=min(b/fps,float(tail)+SHORT_WALK_LEAD_S)
        d=np.asarray(walk.positions[b],float)-np.asarray(walk.positions[a],float)
    else:
        w0=max(0.,start-SHORT_WALK_LEAD_S);w1=min((n-1)/fps,float(tail)+SHORT_WALK_LEAD_S)
        d=np.array([math.cos(float(walk.yaws[k])),math.sin(float(walk.yaws[k]))])
    if float(np.linalg.norm(d))<1e-9 or w1<=w0+2*SHORT_WALK_RAMP_S:raise ValueError('short window walk has no straight heading or time')
    heading=float(math.atan2(d[1],d[0]));direction=np.array([math.cos(heading),math.sin(heading)])
    speed=float(np.max(walk.speeds))
    t=np.arange(n)/fps;r=SHORT_WALK_RAMP_S
    edge=np.clip(np.minimum(t-w0,w1-t)/r,0,1)
    v=np.where((t>=w0)&(t<=w1),speed*edge*edge*(3-2*edge),0.)
    s=np.r_[0.,np.cumsum((v[1:]+v[:-1])/2/fps)]
    length=float(s[-1])
    if length<1e-6:raise ValueError('short window walk has no length')
    positions=np.asarray(walk.positions[k],float)+np.outer(s-s[k],direction)
    yaws=np.full(n,heading)
    return positions,yaws,{'walk_start_s':w0,'walk_end_s':w1,'ramp_s':r,'speed_m_s':speed,'length_m':length,
                           'emission_s':[start,end],'silent_query_until_s':float(tail),'heading_rad':heading,'straight_line':True,
                           'reason':'owner 10-15: walk only where the question needs motion; one straight line, no turning while walking'}


def find_reflected_route(room,pathfinder,seed,attempt=0,*,species='human',utterance_s=1.5,focus='M1',speed_range=None):
    # Claude 10-15: construct the executed straight line directly; the reflected search stays as a fallback.
    from .straight_route import find_straight_route
    try:return find_straight_route(room,pathfinder,seed,attempt,species=species,utterance_s=utterance_s,focus=focus,speed_range=speed_range)
    except ValueError as error:straight_error=str(error)
    rng=np.random.default_rng(int(seed)+int(attempt)*101)
    polygon=shape(room['floor_polygon']);floor=float(room['measured_floor_y_m'])
    listener=np.asarray(room['placement']['camera_m'],float)
    from avengine.rooms.walkable_map import load_room_map
    walkable_map=load_room_map(room.get('room_label',room.get('room_id')),path=room.get('walkable_map_path'),floor_reference=floor)
    is_free=native_free_predicate(pathfinder,polygon,floor,listener,walkable_map=walkable_map,species=species)
    lo,hi=SPECIES_SPEED_M_S[species]
    if speed_range is None:speed_range=(lo+.02,hi-.02)
    if focus=='M1':speed_range=(.82,.9) if species=='human' else (lo+.02,hi-.02)
    speed=float(rng.uniform(max(lo+.01,speed_range[0]),min(hi-.01,speed_range[1])))
    x0,z0,x1,z1=polygon.bounds
    points=[np.array([x,z]) for x in np.linspace(x0,x1,25) for z in np.linspace(z0,z1,25)]
    rng.shuffle(points);starts=[p for p in points if is_free(p)]
    if not starts:raise ValueError('reflection NavMesh prefilter found no clear start')
    best=None;trials=[]
    duration=math.ceil(float(utterance_s)*10-1e-9)/10
    for k in range(min(256,max(64,len(starts)))):
        start=starts[k%len(starts)];heading=float(rng.uniform(-math.pi,math.pi));straight_rejected=0
        walk=reflect_walk(is_free,start,heading,speed,min_speed=lo+.005,duration_s=30.,rng=rng)
        if walk is None:trials.append({'attempt':k+1,'reason':'boxed in or native free predicate failed'});continue
        slots=schedule_emissions(walk,listener[[0,2]],utterance_s=(duration,duration),post_s=1.5,wet_tail_s=.5,min_radial_m=.5,margin_s=SHORT_WALK_LEAD_S)
        if focus=='M1':
            # QA-06/QA-17 only need a stable moving state, so long sounds may span reflections.
            from .reflect_walk import schedule_moving_windows
            slots=slots+schedule_moving_windows(walk,utterance_s=duration,post_s=2.0,wet_tail_s=.5,lead_s=SHORT_WALK_LEAD_S)
        # The original solver runs in one of the five 10-second delivery crops.
        for slot in slots:
            slot['start_s']=math.ceil(slot['start_s']*10-1e-8)/10
            slot['end_s']=slot['start_s']+duration
            fits=[o for o in (0.,5.,10.,15.,20.) if 0.<=slot['start_s']-o<=7. and slot['end_s']+.6<=o+10]
            if not fits:continue
            leg_end=(slot['leg'][1] if slot.get('leg') is not None else slot['window_frames'][1])/30.
            offset=max(fits,key=lambda o:min(leg_end,o+10)-slot['end_s'])
            available=min(leg_end-.1,offset+9.8)-slot['end_s']
            slot['qa17']=bool(slot['qa17'] and available>=2.0)
            slot['qa16']=bool(slot['qa16'] and slot['qa17'])
            slot['offset_s']=offset
            score=(int(slot['qa15'])+int(slot['qa16'])) if focus=='M2' else (int(slot['qa06'])+int(slot['qa17']))
            if focus=='M2' and not slot['qa15']:continue
            # The executed walk is one straight line (see _short_window_walk); keep only slots whose whole line,
            # including the start/stop ramps, is free here.  The native stage re-places it and rechecks everything.
            try:line,_,window=_short_window_walk(walk,slot)
            except ValueError:straight_rejected+=1;continue
            moving_frames=np.flatnonzero((np.arange(len(line))/30.>=window['walk_start_s']-1e-9)&(np.arange(len(line))/30.<=window['walk_end_s']+1e-9))
            if not all(is_free(line[i]) for i in np.r_[moving_frames[::4],moving_frames[-1:]]):straight_rejected+=1;continue
            candidate=(score,walk,slot,k+1)
            if best is None or score>best[0]:best=candidate
        trials.append({'attempt':k+1,'reason':'usable leg found' if slots else 'no whole-source sounding leg','slots':len(slots),'straight_line_rejected':straight_rejected})
        if best is not None and best[0]>=2:break
    if best is None:raise ValueError('bounded native reflection search exhausted: '+str(trials)+'; '+straight_error)
    _,walk,slot,selected=best
    positions,yaws,window=_short_window_walk(walk,slot)
    target=np.mean(positions,axis=0)-listener[[0,2]]
    from .straight_route import frame_rest_poses
    camera_yaw,framing=frame_rest_poses([positions[0],positions[-1]],listener[[0,2]],math.degrees(math.atan2(target[0],-target[1])))
    return {'reflect_walk':True,'positions_xz_m':positions.tolist(),'travel_yaws_rad':yaws.tolist(),'rest_pose_framing':framing,
        'short_window_walk':window,
        'walkable_map_path':getattr(walkable_map,'path',None),
        'walkable_map_version':walkable_map.metadata['version']if walkable_map is not None else None,
        'walkable_map_room_id':walkable_map.metadata['room_id']if walkable_map is not None else None,
        'speed_m_s':speed,'speed_range_m_s':[lo,hi],'listener_m':listener.tolist(),
        'camera_yaw_deg':camera_yaw,
        'species':species,'template':focus,'emission_slot':slot,'candidate_count':len(trials),
        'candidate_trials':trials,'selected_attempt':selected,'candidate_native_root_samples':901,
        'geometry_status':'native NavMesh proposal only; original pose/contact/collision/readback gates pending'}


def reflected_route_proposals(video,entry,scene,local_forward,request):
    route=request['motion_quartet_route'];xz=np.asarray(route['positions_xz_m'],float)
    floor=float(entry['applied_root_input'][1,3]);polygon=shape(scene['production_scenario']['motion_floor_polygon'])
    from avengine.rooms.walkable_map import load_room_map,SPECIES_RADII_M
    species=route.get('species',entry.get('declaration',{}).get('species_id','human'))
    if species not in SPECIES_RADII_M:species='human'
    radius=SPECIES_RADII_M[species]
    walkable_map=None if route.get('walkable_map_bypassed') else load_room_map(route.get('walkable_map_room_id'),path=route.get('walkable_map_path'))
    obstacles=[]
    for actor in video.actors:
        if actor is entry or actor.get('phase2_layout_qualified') is False:continue
        root=actor['applied_root_input'];v=np.asarray(actor['original_mesh'].vertices)@root[:3,:3].T+root[:3,3]
        obstacles.append(box(v[:,0].min()-radius,v[:,2].min()-radius,v[:,0].max()+radius,v[:,2].max()+radius))
    # Registered seats/props are separate native rigid instances. Include their
    # measured world AABBs as well as the actors' original skin bounds.
    seen={int(a['object'].object_id)for a in video.actors if a.get('object')is not None}
    sim=getattr(video,'sim',None)
    if sim is not None:
        for obj in sim.get_rigid_object_manager().get_objects_by_handle_substring().values():
            if int(obj.object_id)in seen:continue
            node=obj.root_scene_node;node.compute_cumulative_bb();bounds=node.cumulative_bb
            low=np.asarray(bounds.min,float);high=np.asarray(bounds.max,float)
            corners=np.array([[x,y,z]for x in (low[0],high[0])for y in (low[1],high[1])for z in (low[2],high[2])])
            root=np.asarray(node.absolute_transformation(),float)
            v=corners@root[:3,:3].T+root[:3,3]
            if not np.isfinite(v).all():raise ValueError('dynamic native prop has invalid world bounds')
            obstacles.append(box(v[:,0].min()-radius,v[:,2].min()-radius,v[:,0].max()+radius,v[:,2].max()+radius))
    is_free=native_free_predicate(video.pathfinder,polygon,floor,scene['listener_m'],obstacles=obstacles,walkable_map=walkable_map,species=species)
    from avengine.timeline.v7_spatial_layout import spacing_rejection
    listener=np.asarray(scene['listener_m'],float)[[0,2]]
    rejected=Counter()
    slot=route['emission_slot'];i0=round(slot['start_s']*30);i1=round(slot['end_s']*30)
    plan_for_fit=request.get('motion_during_condition_plan_for_fit')
    if plan_for_fit is not None:
        from avengine.timeline.v7_motion_during_conditions import verify_planned_trajectories
        mover_id=entry['declaration'].get('asset_id');mouth_y=float(scene['listener_m'][1])+.3
        ids={inst['entity_instance_id'] for item in plan_for_fit['plans'] for inst in item['config']['instances']}
        far=np.asarray(scene['listener_m'],float)
        static_tracks={aid:np.tile(far+[25.,0.,25.+5.*k],(2,1)) for k,aid in enumerate(sorted(ids))}
    centre=np.mean(xz,axis=0);candidates=[];tested=0
    # Place the short walk anywhere on the native floor (0.3 m grid x 16 headings), not only near where it was
    # planned: the planned room outline differs from the native motion floor, so local shifts mostly fell outside.
    rotations=[0.]+[k*math.pi/8 for k in range(1,16)]
    gx0,gz0,gx1,gz1=polygon.bounds
    grid=[np.array([x,z]) for x in np.arange(gx0,gx1+1e-9,.3) for z in np.arange(gz0,gz1+1e-9,.3) if polygon.covers(Point(x,z))]
    rng=np.random.default_rng(int(abs(hash(str(route.get('selected_attempt')))) % (2**32)))
    rng.shuffle(grid)
    # The planned placement itself first (it already passed the planning predicates), then local moves, then the
    # whole native floor grid.
    local=[np.array([x,z])for x in (0.,.25,-.25,.5,-.5,1.,-1.)for z in (0.,.25,-.25,.5,-.5,1.,-1.)]
    shifts=local+[g-centre for g in grid]
    limit=max(24,int(request.get('native_route_candidate_pool',96)))
    for shift in shifts:
        for angle in rotations:
            tested+=1;rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
            points=(xz-centre)@rotation.T+centre+shift
            if walkable_map is not None and not walkable_map.contains_many(points,species).all():rejected['walkable_map_static']+=1;continue
            line=LineString(points)
            if not polygon.covers(line):rejected['native_scene_floor_polygon']+=1;continue
            if any(line.intersects(o)for o in obstacles):rejected['qualified_body_expanded_bbox']+=1;continue
            radii=np.linalg.norm(points-listener,axis=1)
            if float(radii.min())<CAMERA_MARGIN_M.get(species,1.):rejected['camera_distance_1_5m']+=1;continue
            # Rotating/translating preserves the complete distance-driven gait
            # clock, but every requested radial condition must be rechecked.
            if slot['qa15']:
                delta=points[i1]-points[i0];towards=listener-points[i0]
                cosine=abs(float(np.dot(delta,towards)))/(np.linalg.norm(delta)*np.linalg.norm(towards)+1e-12)
                sign=-1 if slot['branch']=='nearer'else 1
                values=radii[i0:i1+1];trim=int(round(.2*(i1-i0)))
                values=values[trim:len(values)-trim] if len(values)-2*trim>=2 else values      # audible-window margin
                # QA-15's judge reads a monotonic >= 0.5 m distance change; no angle (the planner's tangent and this
                # chord disagreed on curved routes and rejected the planned placement itself).
                if sign*(values[-1]-values[0])<.5 or np.any(sign*np.diff(values)<-.005):rejected['radial_slot']+=1;continue
            positions=np.column_stack((points[:,0],np.full(len(points),floor),points[:,1]))
            if spacing_rejection(entry['declaration'],positions,[a for a in video.actors if a is not entry]):rejected['body_centre_spacing']+=1;continue
            if plan_for_fit is not None:
                # Same ruler as the judge after the rigid re-placement (owner 10-15): every planned family must hold.
                track3=np.column_stack((points[:,0],np.full(len(points),mouth_y),points[:,1]))
                verdict=verify_planned_trajectories(plan_for_fit,{**static_tracks,mover_id:track3},scene['listener_m'])
                if not all(verdict.values()):rejected['judge_motion_conditions']+=1;continue
            unique=np.unique(points,axis=0)
            if walkable_map is not None and not all(walkable_map.segment_free(a,b,species)for a,b in zip(points[:-1],points[1:])if np.linalg.norm(b-a)>1e-10):rejected['walkable_map_crossed_cell']+=1;continue
            if not all(is_free(p) for p in unique):rejected['native_snap_clearance_or_camera']+=1;continue
            base=math.atan2(local_forward[2],local_forward[0])
            yaws=np.unwrap(base-np.asarray(route['travel_yaws_rad'],float)-angle)
            selected=np.linspace(0,len(positions)-1,17,dtype=int)
            clearance=min(float(video.pathfinder.distance_to_closest_obstacle(video.pathfinder.snap_point(positions[i]),1.))for i in selected)if hasattr(video.pathfinder,'distance_to_closest_obstacle')else .35
            candidates.append((clearance,positions,yaws,angle,shift))
            if len(candidates)>=limit:break
        if len(candidates)>=limit:break
    diagnostics={'proposals_tested':tested,'native_admissible_route_pool':len(candidates),'rejections':dict(rejected),
        'native_scene_floor_bounds_xz_m':list(polygon.bounds),'planned_route_bounds_xz_m':list(LineString(xz).bounds),
        'qualified_obstacle_count':len(obstacles),'walkable_map_path':getattr(walkable_map,'path',None),
        'walkable_map_version':walkable_map.metadata['version']if walkable_map is not None else None,'native_root_y_m':floor,'camera_m':scene['listener_m'],
        'original_gates_unchanged':True}
    entry['reflection_native_prefilter']=diagnostics
    if request.get('_reflection_diagnostics_path'):
        from .diagnostics import write_route_summary
        from pathlib import Path
        write_route_summary(Path(request['_reflection_diagnostics_path']),diagnostics)
    candidates.sort(key=lambda c:-c[0])
    for index,(clearance,positions,yaws,angle,shift)in enumerate(candidates):
        yield SimpleNamespace(positions=positions,yaws=yaws,receipt={
            'route_kind':'reflected native NavMesh walk','candidate_count':route['candidate_count'],
            'candidate_trials':route['candidate_trials'],'selected_attempt':route['selected_attempt'],
            'native_geometric_proposals_tested':tested,'native_admissible_route_pool':len(candidates),
            'native_route_rank':index+1,'native_obstacle_clearance_score_m':clearance,
            'rigid_route_rotation_rad':angle,'rigid_route_translation_xz_m':shift.tolist(),
            'species':route['species'],'root_samples':len(positions),'native_full_pose_checks_still_required':True,
            'minimum_obstacle_clearance_m':.35,'minimum_camera_distance_m':1.5})


def reflected_distance_clock(route,nominal,phase0=0.):
    xz=np.asarray(route['positions_xz_m'],float)
    steps=np.linalg.norm(np.diff(xz,axis=0),axis=1)
    distance=np.r_[0.,np.cumsum(steps)];velocity=np.r_[steps[0],steps]*30.
    return distance,velocity,phase0+distance/nominal

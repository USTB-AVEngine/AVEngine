"""Reflection proposals on the original native NavMesh; no answer predicates."""
import math
from types import SimpleNamespace
import numpy as np
from shapely.geometry import Point, shape, LineString, box
from .reflect_walk import reflect_walk, schedule_emissions, SPECIES_SPEED_M_S


def native_free_predicate(pathfinder, polygon, floor_y, listener, *, obstacles=(), camera_margin=1.5):
    listener=np.asarray(listener,float)[[0,2]]
    def is_free(xz):
        xz=np.asarray(xz,float)
        if not polygon.covers(Point(*xz)) or np.linalg.norm(xz-listener)<camera_margin:return False
        if any(o.covers(Point(*xz)) for o in obstacles):return False
        p=np.array([xz[0],floor_y,xz[1]])
        snap=np.asarray(pathfinder.snap_point(p),float)
        if not np.isfinite(snap).all() or np.linalg.norm((snap-p)[[0,2]])>.002:return False
        return (not hasattr(pathfinder,'distance_to_closest_obstacle') or
                float(pathfinder.distance_to_closest_obstacle(snap,1.))>=.35)
    return is_free


def find_reflected_route(room,pathfinder,seed,attempt=0,*,species='human',utterance_s=1.5,focus='M1',speed_range=None):
    rng=np.random.default_rng(int(seed)+int(attempt)*101)
    polygon=shape(room['floor_polygon']);floor=float(room['measured_floor_y_m'])
    listener=np.asarray(room['placement']['camera_m'],float)
    is_free=native_free_predicate(pathfinder,polygon,floor,listener)
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
        start=starts[k%len(starts)];heading=float(rng.uniform(-math.pi,math.pi))
        walk=reflect_walk(is_free,start,heading,speed,min_speed=lo+.005,duration_s=30.,rng=rng)
        if walk is None:trials.append({'attempt':k+1,'reason':'boxed in or native free predicate failed'});continue
        slots=schedule_emissions(walk,listener[[0,2]],utterance_s=(duration,duration),post_s=1.5,wet_tail_s=.5,min_radial_m=.5)
        # The original solver runs in one of the five 10-second delivery crops.
        for slot in slots:
            slot['start_s']=math.ceil(slot['start_s']*10-1e-8)/10
            slot['end_s']=slot['start_s']+duration
            fits=[o for o in (0.,5.,10.,15.,20.) if 0.<=slot['start_s']-o<=7. and slot['end_s']+.2<=o+10]
            if not fits:continue
            offset=max(fits,key=lambda o:min(slot['leg'][1]/30.,o+10)-slot['end_s'])
            available=min(slot['leg'][1]/30.-.1,offset+9.8)-slot['end_s']
            slot['qa17']=bool(slot['qa17'] and available>=2.0)
            slot['qa16']=bool(slot['qa16'] and slot['qa17'])
            slot['offset_s']=offset
            score=(int(slot['qa15'])+int(slot['qa16'])) if focus=='M2' else (int(slot['qa06'])+int(slot['qa17']))
            if focus=='M2' and not slot['qa15']:continue
            candidate=(score,walk,slot,k+1)
            if best is None or score>best[0]:best=candidate
        trials.append({'attempt':k+1,'reason':'usable leg found' if slots else 'no whole-source sounding leg','slots':len(slots)})
        if best is not None and best[0]>=2:break
    if best is None:raise ValueError('bounded native reflection search exhausted: '+str(trials))
    _,walk,slot,selected=best
    target=np.mean(walk.positions,axis=0)-listener[[0,2]]
    return {'reflect_walk':True,'positions_xz_m':walk.positions.tolist(),'travel_yaws_rad':walk.yaws.tolist(),
        'speed_m_s':speed,'speed_range_m_s':[lo,hi],'listener_m':listener.tolist(),
        'camera_yaw_deg':math.degrees(math.atan2(target[0],-target[1])),
        'species':species,'template':focus,'emission_slot':slot,'candidate_count':len(trials),
        'candidate_trials':trials,'selected_attempt':selected,'candidate_native_root_samples':901,
        'geometry_status':'native NavMesh proposal only; original pose/contact/collision/readback gates pending'}


def reflected_route_proposals(video,entry,scene,local_forward,request):
    route=request['motion_quartet_route'];xz=np.asarray(route['positions_xz_m'],float)
    floor=float(entry['applied_root_input'][1,3]);polygon=shape(scene['production_scenario']['motion_floor_polygon'])
    obstacles=[]
    for actor in video.actors:
        if actor is entry or actor.get('phase2_layout_qualified') is False:continue
        root=actor['applied_root_input'];v=np.asarray(actor['original_mesh'].vertices)@root[:3,:3].T+root[:3,3]
        obstacles.append(box(v[:,0].min()-.35,v[:,2].min()-.35,v[:,0].max()+.35,v[:,2].max()+.35))
    is_free=native_free_predicate(video.pathfinder,polygon,floor,scene['listener_m'],obstacles=obstacles)
    from avengine.timeline.v7_spatial_layout import spacing_rejection
    listener=np.asarray(scene['listener_m'],float)[[0,2]]
    slot=route['emission_slot'];i0=round(slot['start_s']*30);i1=round(slot['end_s']*30)
    centre=np.mean(xz,axis=0);candidates=[];tested=0
    rotations=[0.,math.pi/16,-math.pi/16,math.pi/8,-math.pi/8,math.pi/4,-math.pi/4,math.pi/2,math.pi,-math.pi/2]
    shifts=[np.array([x,z])for x in (0.,.25,-.25,.5,-.5,1.,-1.)for z in (0.,.25,-.25,.5,-.5,1.,-1.)]
    limit=max(24,int(request.get('native_route_candidate_pool',96)))
    for shift in shifts:
        for angle in rotations:
            tested+=1;rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
            points=(xz-centre)@rotation.T+centre+shift
            line=LineString(points)
            if not polygon.covers(line) or any(line.intersects(o)for o in obstacles):continue
            radii=np.linalg.norm(points-listener,axis=1)
            if float(radii.min())<1.5:continue
            # Rotating/translating preserves the complete distance-driven gait
            # clock, but every requested radial condition must be rechecked.
            if slot['qa15']:
                delta=points[i1]-points[i0];towards=listener-points[i0]
                cosine=abs(float(np.dot(delta,towards)))/(np.linalg.norm(delta)*np.linalg.norm(towards)+1e-12)
                sign=-1 if slot['branch']=='nearer'else 1
                values=radii[i0:i1+1]
                if cosine<math.cos(math.radians(30)) or sign*(values[-1]-values[0])<.5 or np.any(sign*np.diff(values)<-.005):continue
            positions=np.column_stack((points[:,0],np.full(len(points),floor),points[:,1]))
            if spacing_rejection(entry['declaration'],positions,[a for a in video.actors if a is not entry]):continue
            if not all(is_free(p) for p in points):continue
            base=math.atan2(local_forward[2],local_forward[0])
            yaws=np.unwrap(base-np.asarray(route['travel_yaws_rad'],float)-angle)
            selected=np.linspace(0,len(positions)-1,17,dtype=int)
            clearance=min(float(video.pathfinder.distance_to_closest_obstacle(video.pathfinder.snap_point(positions[i]),1.))for i in selected)if hasattr(video.pathfinder,'distance_to_closest_obstacle')else .35
            candidates.append((clearance,positions,yaws,angle,shift))
            if len(candidates)>=limit:break
        if len(candidates)>=limit:break
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

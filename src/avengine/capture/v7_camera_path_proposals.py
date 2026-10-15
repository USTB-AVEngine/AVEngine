"""Filter native route proposals for a fixed camera; never set QA visibility."""
import math
import numpy as np


def matches_camera_path(positions,times,listener,request):
    condition=request.get('camera_path_condition')
    if not condition:return True
    points=np.asarray(positions,float);times=np.asarray(times,float);listener=np.asarray(listener,float)
    forward=np.asarray(condition['camera_forward'],float);right=np.cross(forward,[0.,1.,0.])
    delta=points-listener
    angle=np.degrees(np.arctan2(delta@right,delta@forward))
    kind=condition['kind']
    if kind=='enter':
        sign=int(condition['required_entry_side'])
        return bool(sign*angle[0]>=condition['initial_outside_minimum_abs_deg']
            and abs(angle[-1])<=condition['final_inside_maximum_abs_deg'])
    if kind=='cross_view':
        radii=np.linalg.norm(delta[:,[0,2]],axis=1)
        lo,hi=request['intervals_s'][0][0],request['intervals_s'][-1][1]
        start=float(np.interp(lo,times,angle));end=float(np.interp(hi,times,angle))
        return bool(np.min(radii)>=condition['minimum_crossing_radius_m']
            and np.max(np.abs(angle))<=condition['maximum_crossing_abs_deg']
            and start*end<0 and abs(end-start)>=condition['minimum_angular_span_deg'])
    raise ValueError('unsupported fixed-camera route construction: '+str(kind))


def fixed_camera_route_proposals(video,entry,scene,distance,local_forward,request):
    """Construct native straight entries/crossings instead of eroding a full circle.

    The supplied gait distance clock, listener and camera never change. This is
    a proposal search; the caller still certifies every actual original pose.
    """
    from types import SimpleNamespace
    from shapely.geometry import shape,Point,LineString,box
    from avengine.timeline.v7_spatial_layout import spacing_rejection
    poly=shape(scene['production_scenario']['motion_floor_polygon'])
    margin=float(request.get('route_body_margin_m',.32));free=poly.buffer(-margin)
    if free.is_empty:return
    distance=np.asarray(distance,float);times=np.arange(len(distance))/30.
    original=np.asarray(entry['applied_root_input'][:3,3],float);listener=np.asarray(scene['listener_m'],float)
    x0,z0,x1,z1=free.bounds
    starts=[original.copy()]+[np.array([x,original[1],z])for x in np.arange(x0,x1+.0001,.3)for z in np.arange(z0,z1+.0001,.3)if free.covers(Point(x,z))]
    starts.sort(key=lambda p:float(np.linalg.norm((p-original)[[0,2]])))
    others=[a for a in video.actors if a is not entry and a.get('phase2_layout_qualified')is not False]
    obstacles=[]
    for actor in others:
        root=actor['applied_root_input'];v=np.asarray(actor['original_mesh'].vertices)@root[:3,:3].T+root[:3,3]
        obstacles.append(box(v[:,0].min()-margin,v[:,2].min()-margin,v[:,0].max()+margin,v[:,2].max()+margin))
    headings=np.arange(32)*math.pi/16;local_angle=math.atan2(local_forward[2],local_forward[0])
    budget=max(24,min(256,int(request.get('route_ranking_candidate_budget',96))))
    results=[];seen=set();checked=0
    for start in starts:
        for heading in headings:
            vector=np.array([math.cos(heading),0.,math.sin(heading)])
            positions=start+distance[:,None]*vector;checked+=1
            if not matches_camera_path(positions,times,listener,request):continue
            line=LineString([positions[0,[0,2]],positions[-1,[0,2]]])
            if not free.covers(line)or any(line.intersects(o)for o in obstacles):continue
            if spacing_rejection(entry['declaration'],positions,others,camera_position=listener,
                    camera_minimum_m=float(request.get('minimum_listener_horizontal_distance_m',1.5))):continue
            snaps=np.asarray([video.pathfinder.snap_point(p)for p in positions]);error=np.linalg.norm((snaps-positions)[:,[0,2]],axis=1)
            if not np.isfinite(snaps).all()or float(error.max())>.002:continue
            path=video.hs.ShortestPath();path.requested_start=snaps[0];path.requested_end=snaps[-1]
            if not video.pathfinder.find_path(path):continue
            indices=np.linspace(0,len(positions)-1,17,dtype=int)
            clearance=min((float(video.pathfinder.distance_to_closest_obstacle(snaps[i],2.))for i in indices),default=0.)if hasattr(video.pathfinder,'distance_to_closest_obstacle') else 0.
            key=tuple(np.round(np.r_[start[[0,2]],heading],5))
            if key in seen:continue
            seen.add(key)
            results.append(SimpleNamespace(positions=positions,yaws=np.full(len(positions),local_angle-heading),
                receipt={'route_kind':'fixed-camera native straight '+request['camera_path_condition']['kind'],
                    'root_samples':len(positions),'minimum_native_obstacle_clearance_m':clearance,
                    'maximum_horizontal_snap_error_m':float(error.max()),'native_geodesic_m':float(path.geodesic_distance),
                    'proposal_clock_changed':False,'camera_changed':False,'native_full_pose_checks_still_required':True}))
            if len(results)>=budget:break
        if len(results)>=budget:break
    results.sort(key=lambda r:r.receipt['minimum_native_obstacle_clearance_m'],reverse=True)
    for rank,result in enumerate(results,1):
        result.receipt.update(native_clearance_rank=rank,ranked_valid_route_candidates=len(results),geometric_proposals_examined=checked)
        yield result

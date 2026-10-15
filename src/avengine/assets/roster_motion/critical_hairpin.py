"""Two straight motion windows connected by a real smooth NavMesh turn."""
import math
import numpy as np
from shapely.geometry import shape,Point,LineString


def hairpin_samples(distance,length,radius):
    s=np.asarray(distance,float);points=np.empty((len(s),2));tangent=np.empty_like(points);segments=np.full(len(s),-1)
    first=s<=length;points[first]=np.column_stack((-length/2+s[first],np.full(first.sum(),radius)));tangent[first]=[1.,0.];segments[first]=0
    arc=(s>length)&(s<length+math.pi*radius);a=(s[arc]-length)/radius
    points[arc]=np.column_stack((length/2+radius*np.sin(a),radius*np.cos(a)));tangent[arc]=np.column_stack((np.cos(a),-np.sin(a)))
    last=s>=length+math.pi*radius;q=s[last]-length-math.pi*radius
    points[last]=np.column_stack((length/2-q,np.full(last.sum(),-radius)));tangent[last]=[-1.,0.];segments[last]=1
    return points,tangent,segments


def find_critical_hairpin(room,pathfinder,seed,speed_range,quiet_speed,*,minimum_camera_distance_m=1.5):
    """Propose geometry only; the original complete native gates remain mandatory."""
    from avengine.assets.roster_motion.timed_straight_clock import single_pass_clock
    polygon=shape(room['floor_polygon']);free=polygon.convex_hull.buffer(-.35)
    if free.is_empty:return None
    rng=np.random.default_rng(seed+20731);x0,z0,x1,z1=free.bounds
    centres=[free.representative_point()]+[Point(x,z)for x in np.linspace(x0,x1,21)for z in np.linspace(z0,z1,21)]
    centres=[p for p in centres if free.covers(p)];rng.shuffle(centres)
    floor=float(room['measured_floor_y_m']);times=np.arange(901)/30.;tested=0
    for speed in sorted(float(v)for v in rng.uniform(*speed_range,4)):
        distance,_=single_pass_clock(times,speed,quiet_speed,9.7);length=float(distance[90])
        for radius in (.85,.65,1.,.5):
            turn_end=float(np.interp(length+math.pi*radius,distance[:292],times[:292]))
            minimum_stop=max(7.9,turn_end+2.5)
            if minimum_stop>9.2:continue
            p,tangent,segments=hairpin_samples(distance,length,radius)
            for heading in rng.permutation(np.arange(16)*math.pi/8):
                rotation=np.array([[math.cos(heading),-math.sin(heading)],[math.sin(heading),math.cos(heading)]])
                for centre in centres:
                    tested+=1;xz=p@rotation.T+[centre.x,centre.y]
                    if not free.covers(LineString(xz)):continue
                    camera=np.array([length/2+radius+max(.8,minimum_camera_distance_m+.1),radius])@rotation.T+[centre.x,centre.y]
                    if not free.covers(Point(*camera)):continue
                    listener=np.array([camera[0],floor+1.25,camera[1]])
                    camera_snap=np.asarray(pathfinder.snap_point(listener-[0,1.25,0]),float)
                    if not np.isfinite(camera_snap).all()or np.linalg.norm((camera_snap-listener)[[0,2]])>.002:continue
                    positions=np.column_stack((xz[:,0],np.full(len(xz),floor),xz[:,1]))
                    preview=np.linspace(0,291,25,dtype=int);bad=False
                    for i in preview:
                        snapped=np.asarray(pathfinder.snap_point(positions[i]),float)
                        if not np.isfinite(snapped).all()or np.linalg.norm((snapped-positions[i])[[0,2]])>.002 or (hasattr(pathfinder,'distance_to_closest_obstacle')and pathfinder.distance_to_closest_obstacle(snapped,1.)<.35):bad=True;break
                    if bad:continue
                    snaps=np.asarray([pathfinder.snap_point(point)for point in positions])
                    if not np.isfinite(snaps).all()or np.max(np.linalg.norm((snaps-positions)[:,[0,2]],axis=1))>.002:continue
                    if hasattr(pathfinder,'distance_to_closest_obstacle')and any(pathfinder.distance_to_closest_obstacle(point,1.)<.35 for point in snaps):continue
                    radial=np.linalg.norm(positions+[0.,1.5,0.]-listener,axis=1)
                    from avengine.assets.roster_motion.construction_distance import accepts_distance_trend
                    if not accepts_distance_trend(radial[9:82],-1,.5):continue
                    delta=positions[60]+[0.,1.5,0.]-listener
                    return {'single_pass':True,'critical_hairpin':True,'speed_m_s':speed,
                        'speed_range_m_s':[quiet_speed,speed_range[1]],'emission_speed_range_m_s':list(speed_range),
                        'centre_xz_m':[centre.x,centre.y],'heading_rad':float(heading),'straight_length_m':length,
                        'turn_radius_m':radius,'phase_distance_offset_m':0.,'straight_start_s':0.,'straight_end_s':3.,
                        'turn_speed_m_s':quiet_speed,'minimum_stop_s':minimum_stop,'post_turn_straight_start_s':turn_end,
                        'listener_m':listener.tolist(),'camera_yaw_deg':math.degrees(math.atan2(delta[0],-delta[2])),
                        'distance_branches':[{'offset_s':0.,'branch':'nearer'}],
                        'minimum_emission_radial_cosine':1.,'maximum_radial_angle_deg':30.,'candidate_count':tested,
                        'candidate_native_root_samples':901,'motion_template_windows_s':[0.],
                        'minimum_silent_straight_until_s':minimum_stop-.3,'all_four_families_retained':True,
                        'geometry_status':'native NavMesh proposal; complete exact native skin/contact/pixel gates still required'}
    return None

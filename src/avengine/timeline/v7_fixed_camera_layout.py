"""Owner 2026-10-15: fixed cameras and naturally separated standing people."""
import math
import numpy as np
POLICY='owner-fixed-camera-natural-layout-20261015-v2'


def natural_person_yaws(positions,listener,*,seed=0,room_center=None,return_modes=False):
    """Seeded natural orientations without assigning a pose by person index."""
    points=np.asarray(positions,float);listener=np.asarray(listener,float);rng=np.random.default_rng(int(seed));center=np.asarray(room_center,float)if room_center is not None else points.mean(axis=0)
    for _ in range(128):
        result=[];modes=[]
        for i,p in enumerate(points):
            mode=str(rng.choice(['conversation','room_center','free_direction'],p=[.4,.3,.3]));modes.append(mode)
            if mode=='free_direction':yaw=float(rng.uniform(-180.,180.))
            else:
                if mode=='conversation'and len(points)>1:
                    peer=int(rng.choice([j for j in range(len(points))if j!=i]));direction=points[peer]-p
                else:direction=center-p
                if np.linalg.norm(direction[[0,2]])<1e-9:yaw=float(rng.uniform(-180.,180.))
                else:yaw=math.degrees(math.atan2(direction[0],-direction[2]))+float(rng.normal(0.,12.))
            result.append((yaw+180.)%360.-180.)
        y=np.radians(result);spread=float(np.degrees(np.ptp(np.angle(np.exp(1j*(y-y[0]))))))
        toward=listener-points;bearings=np.arctan2(toward[:,0],-toward[:,2]);frontal=np.abs(np.angle(np.exp(1j*(y-bearings))))<=math.radians(25.)
        if (len(points)<2 or spread>=30.)and not frontal.all():return(result,modes)if return_modes else result
    raise ValueError('could not construct diverse natural initial orientations')


def require_fixed_camera_plan(scene):
    values=np.asarray([r['yaw_deg']for r in scene['camera_yaw_knots']],float)
    if not len(values)or not np.isfinite(values).all()or np.ptp(np.unwrap(np.radians(values)))>1e-9:raise ValueError('owner fixed camera requires one yaw throughout the complete30s')
    point=np.asarray(scene['listener_m'],float)
    if point.shape!=(3,)or not np.isfinite(point).all():raise ValueError('fixed native camera position invalid')
    return True


def assert_fixed_native_camera(visual):
    rows=visual['camera_frame_readbacks']
    if len(rows)!=450:raise ValueError('fixed camera needs all450 actual native transforms')
    yaw=float(np.degrees(np.ptp(np.unwrap(np.radians([r['yaw_deg']for r in rows])))));positions=np.asarray([r['position_m']for r in rows],float);spread=float(np.linalg.norm(np.ptp(positions,axis=0)))
    if yaw>.5+1e-9 or spread>.01+1e-9:raise ValueError(f'owner fixed camera failed: yaw range{yaw:.6f}deg, position range{spread:.6f}m')
    return {'status':'pass','measured_frames':450,'yaw_range_deg':yaw,'position_range_m':spread,'yaw_limit_deg':.5,'position_limit_m':.01,'policy':POLICY}


def assert_native_human_camera_clearance(visual):
    """Fail immediately after native capture, before spending time on questions."""
    rows=visual['camera_frame_readbacks'];frames=visual['frames']
    humans={r['asset_id']for r in visual['standing_admission']}
    if not humans or len(rows)!=450 or len(frames)!=450:raise ValueError('human camera clearance needs450 complete native frames and standing identities')
    minimum=float('inf')
    for camera,frame in zip(rows,frames):
        actors={a['asset_id']:a for a in frame['actors']}
        if not humans<=set(actors):raise ValueError('human missing in native camera clearance frame')
        eye=np.asarray(camera['position_m'],float)
        for identity in humans:
            root=np.asarray(actors[identity]['position_m'],float)
            distance=float(np.linalg.norm((root-eye)[[0,2]]))
            if not np.isfinite(distance):raise ValueError('nonfinite native human camera distance')
            minimum=min(minimum,distance)
    if minimum<1.2-1e-6:raise ValueError(f'owner human camera distance {minimum:.9f}m below1.2m')
    return {'status':'pass','measured_frames':450,'minimum_human_camera_distance_m':minimum,'minimum_required_m':1.2,'basis':'horizontal XZ native root-to-camera, every person in every frame'}

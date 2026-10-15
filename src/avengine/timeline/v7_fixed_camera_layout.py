"""Owner 2026-10-15: fixed cameras and naturally separated standing people."""
import math
import numpy as np
POLICY='owner-fixed-camera-natural-layout-20261015-v2'


def natural_person_yaws(positions,listener):
    points=np.asarray(positions,float);listener=np.asarray(listener,float);result=[]
    def unit(v):
        v=np.asarray(v,float).copy();v[1]=0.;return v/max(np.linalg.norm(v),1e-12)
    for i,p in enumerate(points):
        if i<2:direction=.6*unit(points[1-i]-p)+.4*unit(listener-p)
        else:
            direction=unit(points[:2].mean(axis=0)-p)
            if np.linalg.norm(direction)<1e-9:direction=unit(points[0]-p)
        result.append(math.degrees(math.atan2(direction[0],-direction[2])))
    return result


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

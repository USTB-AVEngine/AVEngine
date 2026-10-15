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

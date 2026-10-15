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

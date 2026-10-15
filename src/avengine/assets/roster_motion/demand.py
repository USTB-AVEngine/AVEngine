"""Physical route proposals for requested distance changes, never QA truth."""
from __future__ import annotations
import numpy as np


def matches_distance_request(positions,times,listener,request):
    trend=request.get('distance_trend')
    if trend is None:return True
    if trend not in {'approaching','receding'}:raise ValueError('unsupported demanded distance trend')
    points=np.asarray(positions);times=np.asarray(times);listener=np.asarray(listener)
    radii=np.linalg.norm(points[:,[0,2]]-listener[[0,2]],axis=1)
    sign=-1 if trend=='approaching' else 1
    from avengine.assets.roster_motion.distance_constraints import matches_radial_window
    minimum=float(request.get('minimum_window_distance_change_m',request.get('minimum_distance_change_m',.5)))
    if not np.isfinite(minimum) or minimum<.2:raise ValueError('distance proposal margin must cover the original 0.2m QA margin')
    windows=request.get('distance_trend_windows_s')or request.get('distance_trend_intervals_s')or [[times[0],times[-1]]]
    for lo,hi in windows:
        if not times[0]<=lo<hi<=times[-1]:raise ValueError('distance demand window outside native route')
        selected=np.r_[np.interp(lo,times,radii),radii[(times>lo)&(times<hi)],np.interp(hi,times,radii)]
        if not matches_radial_window(selected,sign,minimum):return False
    return True

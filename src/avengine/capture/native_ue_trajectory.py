"""CPU navigation preflight for unchanged v7 native actor trajectories."""
from __future__ import annotations
import numpy as np
from avengine.timeline.continuous_scene import FPS, position_at


def check_native_trajectories(scene, pathfinder, *, source_height_m=1.2,
                              source_clearance_m=.15, snap_tolerance_m=.15):
    """Check every rendered frame, preserving actual seeded motion and rules.

    This rejects recipes, never projects a track onto a different route.
    The snap tolerance is the native backend's existing support assertion;
    clearance comes from the frozen room source clearance, passed by callers.
    """
    rows=[]
    for endpoint, source in scene['sources'].items():
        max_snap=0.;minimum_clearance=float('inf');failures=[]
        for index in range(round(scene['duration_s']*FPS)):
            point=position_at(source['trajectory'],index/FPS)
            intended=point-[0.,source_height_m,0.]
            snapped=np.asarray(pathfinder.snap_point(intended.astype(np.float32)),float)
            finite=bool(np.isfinite(snapped).all())
            distance=float(np.linalg.norm(snapped-intended)) if finite else float('inf')
            clearance=float(pathfinder.distance_to_closest_obstacle(snapped.astype(np.float32),source_clearance_m+.01)) if finite else 0.
            max_snap=max(max_snap,distance);minimum_clearance=min(minimum_clearance,clearance)
            if not finite or distance>snap_tolerance_m or clearance<source_clearance_m:
                if len(failures)<8:failures.append({'frame_index':index,'time_s':index/FPS,
                    'intended_nav_point_m':intended.tolist(),'snap_distance_m':distance if finite else None,
                    'obstacle_clearance_m':clearance})
        rows.append({'endpoint_id':endpoint,'frame_count':round(scene['duration_s']*FPS),
                     'max_snap_distance_m':max_snap if np.isfinite(max_snap) else None,
                     'minimum_obstacle_clearance_m':minimum_clearance,
                     'status':'fail' if failures else 'pass','failures_sample':failures})
    return {'status':'pass' if rows and all(r['status']=='pass' for r in rows) else 'fail',
            'source_height_m':source_height_m,'source_clearance_m':source_clearance_m,
            'snap_tolerance_m':snap_tolerance_m,'trajectory_rewritten':False,'checks':rows}

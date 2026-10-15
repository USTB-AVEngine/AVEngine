"""Exact joint placements are executed once, retaining every native check."""
from copy import deepcopy
import math


def is_solved_support(support):
    return bool(support and support.get('joint_solved_pose')is True)


def solved_support_proposals(support):
    if not is_solved_support(support):raise ValueError('explicit joint_solved_pose support required')
    if not math.isfinite(float(support['yaw_rad'])):raise ValueError('solved support yaw must be finite')
    result=deepcopy(support);result['point']=result['original_surface_point_m']
    result.setdefault('_outward_clearance_m',0.)
    return [result]

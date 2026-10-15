"""Record solved-pose appearance diagnostics after actual native geometry checks."""
import math


def solved_pose_quality(frame,maximum_reach_error_m,quality,*,record_only):
    feet=list(frame['feet'].values());contacts=[f for f in feet if f['in_contact']]
    reach=float(maximum_reach_error_m);pelvis=float(frame['pelvis_lowering_m'])
    numbers=[reach,pelvis,*[float(f['floor_delta_m'])for f in feet],*[float(f['support_lock_error_m'])for f in contacts]]
    if not all(math.isfinite(v)for v in numbers):raise ValueError('nonfinite actual pose quality measurements')
    observations=[]
    foot_bad=min(f['floor_delta_m']for f in feet)<-.008 or any(
        abs(f['floor_delta_m'])>quality['foot_contact_limit_m']+1e-12
        or f['support_lock_error_m']>quality['support_lock_limit_m']+1e-12 for f in contacts)
    if foot_bad:
        if not record_only:raise ValueError('motion source foot contact or lock violates declared human quality policy')
        observations.append({'kind':'foot_contact_or_slide_quality','actual_feet':feet,'quality_targets':dict(quality)})
    if reach>=.001 or pelvis>=.08:
        if not record_only:raise ValueError('motion IK reach or pelvis bound failed')
        observations.append({'kind':'solved_pose_kinematic_quality','cumulative_reach_error_m':reach,'pelvis_lowering_m':pelvis})
    if observations:frame['visual_quality_observations']=observations
    return observations

"""Explicit owner-selected dog motion policy; strict historical defaults remain intact."""
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import math

OWNER093X = "owner_20261014_093x"

@dataclass(frozen=True)
class MotionPolicy:
    name: str = OWNER093X
    forward_max_deg: float = 20.0
    contact_max_m: float = .012
    stance_step_path_max_m: float = .015
    maximum_edge_growth: float = 2.0
    cross_region_area_max_m2: float = 400e-6
    cross_region_depth_max_m: float = .010
    same_region_depth_is_criterion: bool = False
    per_asset_audio_required: bool = False

def get_motion_policy(name):
    if name is None:
        return None
    if isinstance(name, MotionPolicy):
        if name != MotionPolicy():
            raise ValueError("only the explicitly authorized owner motion preset is supported")
        return name
    if name != OWNER093X:
        raise ValueError("unknown owner motion policy")
    return MotionPolicy()

@contextmanager
def selected_motion_thresholds(policy):
    from avengine.assets.animal_motion import checks
    policy = get_motion_policy(policy)
    if policy is None:
        yield
        return
    values = {"DIRECTION_DEG": policy.forward_max_deg,
              "CONTACT_M": policy.contact_max_m,
              "PLANT_DRIFT_M": policy.stance_step_path_max_m,
              "PLANT_SPEED_M_S": math.inf}
    old = {name: getattr(checks, name) for name in values}
    try:
        for name, value in values.items():
            setattr(checks, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(checks, name, value)

def stance_step_paths(frames):
    previous, totals = {}, {}
    for frame in frames:
        for side, foot in frame["feet"].items():
            if not foot["in_contact"]:
                previous.pop(side, None)
                continue
            key = (side, foot["support_segment"])
            point = foot["support_world_m"]
            if len(point) != 3 or not all(math.isfinite(x) for x in point):
                raise ValueError("nonfinite support point")
            totals.setdefault(key, 0.)
            if side in previous and previous[side][0] == key:
                totals[key] += math.dist(point, previous[side][1])
            previous[side] = (key, point)
    return [{"paw": key[0], "support_segment": key[1], "path_m": value}
            for key, value in totals.items()]

def policy_frame_checks(frames, *, mode="walk_animation", native=False, policy=None):
    from avengine.assets.animal_motion.checks import frame_checks
    policy = get_motion_policy(policy)
    with selected_motion_thresholds(policy):
        result = frame_checks(frames, mode=mode, native=native, policy='owner_relaxed_20261014' if policy else 'strict')
    if policy is None:
        return result
    result["motion_policy"] = asdict(policy)
    result["instantaneous_stance_speed_is_admission_criterion"] = False
    if mode == "walk_animation":
        paths = stance_step_paths(frames)
        result["stance_step_paths"] = paths
        result["max_stance_step_path_m"] = max((x["path_m"] for x in paths), default=0.)
        for path in paths:
            if path["path_m"] > policy.stance_step_path_max_m + 1e-9:
                result["failures"].append("stance_step_path_over_15mm:" + str(path))
        result["status"] = "fail" if result["failures"] else "pass"
    return result

def policy_direction_check(*args, policy=None):
    from avengine.assets.animal_motion.checks import direction_check
    with selected_motion_thresholds(policy):
        return direction_check(*args, policy='owner_relaxed_20261014' if policy else 'strict')

def policy_direction_negative_controls(frames, *, policy=None):
    from avengine.assets.animal_motion.checks import direction_negative_controls
    with selected_motion_thresholds(policy):
        return direction_negative_controls(frames, policy='owner_relaxed_20261014' if policy else 'strict')

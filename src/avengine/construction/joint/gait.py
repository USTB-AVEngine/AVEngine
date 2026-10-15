"""Read an admitted human source once; fit its original complete-step clock."""
from __future__ import annotations
import numpy as np


def admitted_human_clock_source(asset, runtime, house):
    """Asset intake only; no camera, support, route or native placement search."""
    from avengine.assets.roster_motion.episode import admitted_master
    from avengine.assets.roster_motion.pose import pose, animation_times
    from avengine.assets.roster_motion.contact import foot_masks, foot_min
    entry = {"declaration": asset, "prepared_glb": asset["habitat_glb"]}
    d, b, action, lineage = admitted_master(entry, runtime, house)
    source_times = animation_times(d, b, action)
    masks = foot_masks(d, b)
    double_support = []
    for time in source_times:
        parts, _ = pose(d, b, action, float(time), ground=False)
        if max(abs(foot_min(parts, masks, side)[0]) for side in ("L", "R")) < .025:
            double_support.append(float(time))
    if not double_support:
        raise ValueError("admitted human source has no observed double-support phase")
    return {"source_times_s": source_times, "double_support_s": tuple(double_support),
            "nominal_speed_m_s": 1.2, "speed_range_m_s": (.8, 1.4),
            "source_lineage": lineage, "asset_id": asset["asset_id"]}


def complete_human_gait_clock(source, speed_m_s, intervals_s):
    """Exactly the public native fitter; no truncation or rescaling by the caller."""
    from avengine.assets.roster_motion.navmesh_execution import human_distance_driven_clock
    values = human_distance_driven_clock(
        source["source_times_s"], source["double_support_s"], speed_m_s, intervals_s,
        source["nominal_speed_m_s"], source["speed_range_m_s"])
    times, moving, velocity, distance, phase, certificate = values
    return {"times_s": times, "moving": moving, "velocity_m_s": velocity,
            "distance_m": distance, "phase_s": phase,
            "certificate": certificate, "source_lineage": source["source_lineage"]}

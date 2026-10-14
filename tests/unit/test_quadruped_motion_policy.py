"""The explicit owner preset changes admission without corrupting historical gates."""
import math
import pytest
from avengine.assets.animal_motion import checks
from avengine.assets.quadruped_motion_policy import (
    OWNER093X, get_motion_policy, selected_motion_thresholds,
    policy_frame_checks, policy_direction_negative_controls)
from avengine.assets.quadruped_penetration import visible_penetration_gate

def frames(yaw=0., foot_excursion=None):
    out=[]
    for i in range(12):
        t=i*.1;root=[0.,0.,-.03*t];angle=math.radians(yaw)
        feet={}
        for offset,side in enumerate(("FL","FR","HL","HR")):
            phase=(i+offset)%4
            x=(foot_excursion[i%4] if side=="FL" and foot_excursion else offset*.1)
            feet[side]={"floor_delta_m":.004,"in_contact":phase!=3,
                        "support_segment":(i+offset)//4,"support_world_m":[x,0.,0.]}
        out.append({"time_s":t,"root_world_m":root,
                    "head_world_m":[math.sin(angle),.4,root[2]-math.cos(angle)],
                    "tail_world_m":root[:],"feet":feet,
                    "body_floor_min_m":.0004,"body_head_tail_pitch_deg":0.})
    return out

def test_seventeen_degrees_is_only_admitted_by_explicit_owner_preset():
    f=frames(17)
    assert checks.frame_checks(f)["status"]=="fail"
    new=policy_frame_checks(f,policy=OWNER093X)
    assert new["status"]=="pass"
    assert new["direction"]["threshold_deg"]==20.
    assert checks.frame_checks(f)["status"]=="fail"

def test_backward_and_sideways_still_rejected_under_twenty_degree_policy():
    assert policy_direction_negative_controls(frames(17),policy=OWNER093X)["status"]=="pass"

def test_per_step_travel_twelve_mm_passes_even_when_old_anchor_gate_fails():
    f=frames(foot_excursion=[-.003,.003,-.003,0])
    assert checks.frame_checks(f)["status"]=="fail"
    new=policy_frame_checks(f,policy=OWNER093X)
    assert new["status"]=="pass"
    assert new["max_stance_step_path_m"]==pytest.approx(.012)

def test_accumulated_sliding_is_rejected_even_with_anchor_below_fifteen_mm():
    new=policy_frame_checks(frames(foot_excursion=[-.004,.004,-.004,0]),policy=OWNER093X)
    assert new["max_stance_drift_m"]<.015
    assert new["max_stance_step_path_m"]==pytest.approx(.016)
    assert new["status"]=="fail"

def test_policy_globals_restore_after_exception():
    old=(checks.DIRECTION_DEG,checks.PLANT_DRIFT_M,checks.PLANT_SPEED_M_S)
    with pytest.raises(RuntimeError):
        with selected_motion_thresholds(OWNER093X):
            assert checks.DIRECTION_DEG==20.
            raise RuntimeError("intentional interruption")
    assert (checks.DIRECTION_DEG,checks.PLANT_DRIFT_M,checks.PLANT_SPEED_M_S)==old

def relaxed(groups):
    p=get_motion_policy(OWNER093X)
    return visible_penetration_gate(groups,cross_region_area_limit_m2=p.cross_region_area_max_m2,
        depth_limit_m=p.cross_region_depth_max_m,same_region_depth_is_criterion=False)

def test_eight_mm_two_hundred_mm2_crossing_passes_only_new_preset():
    group={"region_pair":["Body","FL"],"new_area_m2":200e-6,"new_maximum_depth_m":.008}
    assert visible_penetration_gate([group])
    assert not relaxed([group])

def test_same_region_ignored_but_cross_region_total_still_limited():
    same={"region_pair":["FL","FL"],"new_area_m2":1.,"new_maximum_depth_m":1.}
    assert not relaxed([same])
    groups=[{"region_pair":["Body",side],"new_area_m2":250e-6,"new_maximum_depth_m":.001} for side in ("FL","FR")]
    assert relaxed(groups)[0]["area_rejected"]

def test_unknown_or_modified_policy_cannot_silently_relax_other_gates():
    with pytest.raises(ValueError):get_motion_policy("unreviewed_relaxation")

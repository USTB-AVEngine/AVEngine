import numpy as np
import pytest
from avengine.timeline.v7_motion_speed import certify_moving_root_speed,SPEED_RANGES
from avengine.timeline.v7_motion_source_roles import QUIET_SPEEDS
pytestmark=pytest.mark.fast_unit

@pytest.mark.parametrize('kind',list(SPEED_RANGES))
def test_measured_speed_accepts_only_owner_species_band(kind):
    lo,hi=SPEED_RANGES[kind];times=np.arange(901)/30.
    for value in (lo,(lo+hi)/2,hi):
        roots=np.column_stack((times*value,np.zeros(len(times)),np.zeros(len(times))))
        assert certify_moving_root_speed(roots,times,np.ones(len(times),bool),kind)['median_m_s']==pytest.approx(value)
    roots=np.column_stack((times*lo/2,np.zeros(len(times)),np.zeros(len(times))))
    with pytest.raises(ValueError,match='outside'):certify_moving_root_speed(roots,times,np.ones(len(times),bool),kind)

def test_stationary_clip_cannot_admit_requested_movement():
    with pytest.raises(ValueError,match='no measured translated'):certify_moving_root_speed(np.zeros((3,3)),[0.,.1,.2],[True]*3,'human')

def test_human_silent_walking_speed_does_not_revert_to_old_slow_value():
    assert QUIET_SPEEDS['human_speech']>=.8
    assert QUIET_SPEEDS['human_other']>=.8

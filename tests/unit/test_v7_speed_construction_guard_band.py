import numpy as np
import pytest
from avengine.timeline.v7_motion_source_roles import QUIET_SPEEDS,EMISSION_SPEEDS
from avengine.timeline.v7_motion_speed import SPEED_RANGES
from avengine.assets.roster_motion.timed_straight_clock import single_pass_clock
pytestmark=pytest.mark.fast_unit

@pytest.mark.parametrize('source,kind',[('human_speech','human'),('human_other','human'),('dog_walk','dog'),('cat_walk','cat'),('animal_slide','slide')])
def test_float32_native_positions_stay_within_unchanged_speed_bounds(source,kind):
    times=np.arange(901)/30.;speed=sum(EMISSION_SPEEDS[source])/2
    distance,_=single_pass_clock(times,speed,QUIET_SPEEDS[source])
    direction=np.array([np.cos(.71),0.,np.sin(.71)])
    native=(np.array([100.,0.,-80.])+distance[:,None]*direction).astype(np.float32).astype(float)
    measured=np.linalg.norm(np.diff(native[100:210:2][:,[0,2]],axis=0),axis=1)*15.
    lo,hi=SPEED_RANGES[kind]
    assert measured.min()>=lo and measured.max()<=hi

def test_old_boundary_speed_reproduces_float32_underflow():
    times=np.arange(150)/15.;positions=(10.+times*.2).astype(np.float32).astype(float)
    assert (np.diff(positions)*15.).min()<.2

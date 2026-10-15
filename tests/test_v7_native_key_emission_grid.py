import numpy as np
import pytest
from avengine.acoustics.moving_rir_calibration import CorrelatedRIRTimeline,effective_emission_filter,correlated_convolve
pytestmark=pytest.mark.fast_unit

def test_measured_native_kernel_is_used_at_its_actual_emission_sample():
    filters=[]
    for tail_sample,gain in [(9,.2),(17,.9),(25,.4)]:
        w=np.zeros(64);w[3]=1.;w[tail_sample]=gain
        filters.append(np.array([w,-w,np.zeros(64),np.zeros(64)]))
    keys=[0,37,127];timeline=CorrelatedRIRTimeline(filters,keys)
    actual=effective_emission_filter(timeline,37,128,block_samples=16)
    np.testing.assert_allclose(actual,filters[1],atol=1e-14)
    dry=np.zeros(128);dry[37]=1.
    wet=correlated_convolve(dry,filters,keys,block_samples=16)
    np.testing.assert_allclose(wet[:,37:37+64],filters[1],atol=1e-12)

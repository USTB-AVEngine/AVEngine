import numpy as np
import pytest
from avengine.acoustics.moving_rir_calibration import CorrelatedRIRTimeline,effective_emission_filter,correlated_convolve
from avengine.timeline.v7_phase2_motion_audio import exact_native_endpoint_checks
pytestmark=pytest.mark.fast_unit

def wave(gain=1.):
    rng=np.random.default_rng(902);t=np.arange(16000)/16000
    w=rng.normal(size=len(t))*np.exp(-9*t)*.01;w[:140]=0;w[140]=.08
    return gain*np.stack((w,np.zeros(len(t)),np.zeros(len(t)),np.sqrt(3)*w))

def setup():
    a,b=wave(),wave(.99)
    return [a,b],CorrelatedRIRTimeline([a,b],[0,1]),[{'time_s':0.,'position_m':[3.,0.,0.]},{'time_s':1.,'position_m':[3.,0.,0.]}]

def checks(timeline,trajectory,refs,lo=0,hi=1,n=2):
    return exact_native_endpoint_checks(timeline,lo,hi,n,trajectory,refs,listener=np.zeros(3),forward=np.array([1.,0.,0.]))

def test_adjacent_actual_convolution_uses_both_original_native_kernels():
    kernels,timeline,trajectory=setup();seen=[]
    def refs(sample):seen.append(sample);return[kernels[sample].copy()for _ in range(3)]
    rows=checks(timeline,trajectory,refs)
    assert seen==[0,1]and all(r['qualification']['status']=='pass'for r in rows)
    assert all(len(r['raw_checks'])==3 and r['midpoint_sample']is None for r in rows)
    for sample in (0,1):
        dry=np.zeros(2);dry[sample]=1
        actual=correlated_convolve(dry,kernels,[0,1]);effective=effective_emission_filter(timeline,sample,2)
        np.testing.assert_allclose(actual[:,sample:sample+effective.shape[1]],effective,atol=1e-12)
        np.testing.assert_array_equal(effective[:,:kernels[sample].shape[1]],kernels[sample])

@pytest.mark.parametrize('error',['gain','delay'])
def test_one_bad_independent_repeat_still_fails_at_an_exact_endpoint(error):
    kernels,timeline,trajectory=setup()
    def refs(sample):
        copies=[kernels[sample].copy()for _ in range(3)]
        if sample==1:copies[2]=copies[2]*2 if error=='gain'else np.pad(copies[2],((0,0),(8,0)))
        return copies
    rows=checks(timeline,trajectory,refs)
    assert rows[0]['qualification']['status']=='pass'
    assert rows[1]['qualification']['status']=='fail'

@pytest.mark.parametrize('lo,hi,n',[(0,2,3),(-1,0,2),(0,1,1)])
def test_endpoint_only_proof_cannot_certify_an_interior_or_invalid_sample(lo,hi,n):
    kernels,timeline,trajectory=setup()
    with pytest.raises(ValueError,match='adjacent integer'):checks(timeline,trajectory,lambda s:[kernels[0]]*3,lo,hi,n)

def test_missing_independent_repeat_is_rejected():
    kernels,timeline,trajectory=setup()
    with pytest.raises(ValueError,match='three independent'):checks(timeline,trajectory,lambda s:[kernels[s]]*2)

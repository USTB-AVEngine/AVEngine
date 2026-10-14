import numpy as np
from avengine.assets.animal_motion.rir_crossfade import FrequencyCoherentInterval,FrequencyCoherentTimeline,frequency_correlated_convolve
from avengine.acoustics.moving_rir_calibration import effective_emission_filter

def kernels():
    rng=np.random.default_rng(741)
    a=np.zeros(1024);b=np.zeros(1024);a[20]=1.;b[22]=.9
    a[60:]=rng.normal(size=964)*np.exp(-np.arange(964)/240)*.012
    b[60:]=rng.normal(size=964)*np.exp(-np.arange(964)/240)*.012
    direction=np.array([1.,.3,-.15,1.4])
    return direction[:,None]*a,direction[:,None]*b

def test_stationary_endpoint_exactness_and_common_foa_direction():
    a,b=kernels()
    assert np.array_equal(FrequencyCoherentInterval(a,a).sample(.37),a)
    middle=FrequencyCoherentInterval(a,b).sample(.5)
    assert np.isfinite(middle).all()
    for channel,ratio in enumerate([1.,.3,-.15,1.4]):
        assert np.max(np.abs(middle[channel]-ratio*middle[0]))<1e-12

def test_actual_emission_convolution_matches_independent_impulse_kernel():
    a,b=kernels();n=641;keys=[0,320,640]
    for emitted in [79,157,321,579]:
        dry=np.zeros(n);dry[emitted]=1.
        wet=frequency_correlated_convolve(dry,[a,b,a],keys)
        expected=effective_emission_filter(FrequencyCoherentTimeline([a,b,a],keys),emitted,n)
        assert np.max(np.abs(wet[:,:emitted]))<1e-12
        assert np.max(np.abs(wet[:,emitted:emitted+expected.shape[1]]-expected))<1e-11
        assert np.max(np.abs(wet[:,emitted+expected.shape[1]:]),initial=0)<1e-11

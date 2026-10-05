from copy import deepcopy
import numpy as np
import pytest
from avengine.acoustics.moving_rir_calibration import (
    calibrate,qualify_calibrated,integer_arrival_crossfade,CorrelatedRIRTimeline,
    effective_emission_filter,correlated_convolve,
)
pytestmark=pytest.mark.fast_unit


def metric(energy=0.):
    return {'status':'measured','foa_error_deg':0.,'geometry_error_deg':0.,
        'direct_energy_difference_db':energy,'total_energy_difference_db':energy,'drr_difference_db':0.,
        'direct_delay_difference_ms':0.,'rt60_difference_s':0.,'rt60_reference_s':.6,'tail_truncated':False,
        'octave_energy':[{'center_hz':n,'energy_difference_db':energy} for n in (125,250,500,1000,2000,4000)]}


def test_calibration_keeps_bad_gain_direction_and_tail_rejected():
    samples=[{**metric(.2),'split':'calibration','house':'h','pose_index':i} for i in range(32) for _ in range(3)]
    samples += [{**metric(10.),'split':'heldout_good','house':'h','pose_index':40}]
    config={'calibration_poses':32,'bootstrap_seed':7,'arrival_extra_margin_ms':.125,
        'energy_and_DRR_extra_margin_db':.5,'revision':'test'}
    budget=calibrate(samples,config)
    assert budget['metrics']['total_energy_difference_db']['maximum_allowed_abs_error']==pytest.approx(.7)
    good=qualify_calibrated([metric(.6)],budget)
    assert good['status']=='pass'
    for bad in [metric(3.),{**metric(),'geometry_error_deg':6.},{**metric(),'tail_truncated':True},
        {**metric(),'direct_delay_difference_ms':.5},{'status':'fail'}]:
        assert qualify_calibrated([bad],budget)['status']=='fail'
    with pytest.raises(ValueError,match='distinct'):calibrate(samples[:3],config)


def native_like_rir(delay):
    rng=np.random.default_rng(8)
    w=np.zeros(1500);w[delay]=1.;w[delay+100:]=rng.normal(0,.02,len(w)-delay-100)*np.exp(-np.arange(len(w)-delay-100)/200)
    return np.stack([w,w*.2,w*.1,w*.4])


def test_coherent_reflection_does_not_receive_equal_power_three_db_boost():
    a=native_like_rir(32);b=a*1.0001
    mixed=integer_arrival_crossfade(a,b,.5)
    np.testing.assert_allclose(mixed,(a+b)*.5,rtol=1e-8,atol=1e-11)
    np.testing.assert_array_equal(integer_arrival_crossfade(a,a,.35),a)


def test_virtual_convolution_preserves_single_arrival_and_complete_tail():
    a=native_like_rir(32);b=native_like_rir(33)
    filters=[a,b];keys=[0,997];timeline=CorrelatedRIRTimeline(filters,keys)
    length=1200
    # The previous two-center overlap double-pulsed this one-sample arrival.
    for emission in (1,479,499,731,1199):
        dry=np.zeros(length);dry[emission]=1.
        wet=correlated_convolve(dry,filters,keys,block_samples=160)
        effective=effective_emission_filter(timeline,emission,length,block_samples=160)
        np.testing.assert_allclose(wet[:,emission:emission+effective.shape[1]],effective,atol=1e-12)
        assert wet.shape[1]>=emission+effective.shape[1]
        assert np.square(wet[:,emission+20:emission+50]).sum()>np.square(a[:,:50]).sum()*.98
    static=correlated_convolve(np.ones(31),[a],[0])
    from scipy.signal import fftconvolve
    np.testing.assert_allclose(static,np.stack([fftconvolve(np.ones(31),h) for h in a]))


def test_crossfade_remains_finite_at_vanishing_tail_powers():
    a=native_like_rir(32);b=native_like_rir(33)
    a=np.pad(a,((0,0),(0,2000)));b=np.pad(b,((0,0),(0,2000)))
    a[:,-2000:]=np.random.default_rng(4).normal(0,1e-15,(4,2000))
    b[:,-2000:]=np.random.default_rng(5).normal(0,1e-16,(4,2000))
    with np.errstate(invalid='raise',divide='raise'):
        assert np.isfinite(integer_arrival_crossfade(a,b,.5)).all()

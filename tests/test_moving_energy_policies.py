import numpy as np
import pytest
from avengine.acoustics.moving_energy_policies import adaptive_keyframes,summarize_checks
pytestmark=pytest.mark.fast_unit

def measured(**changes):
    return {'status':'measured','foa_error_deg':0.,'geometry_error_deg':0.,'direct_energy_difference_db':0.,
        'total_energy_difference_db':0.,'drr_difference_db':0.,'direct_delay_difference_ms':0.,'rt60_difference_s':0.,
        'rt60_reference_s':.5,'tail_truncated':False,'octave_energy':[{'energy_difference_db':0.}],**changes}

def test_failed_samples_not_number_of_reasons():
    result=summarize_checks([measured(direct_energy_difference_db=2.,drr_difference_db=2.),measured()])
    assert result['pass_count']==1 and result['failed_samples']==1 and len(result['failures'])==2

def test_observed_native_corners_cannot_be_coarsened_to_make_refinement_eligible():
    track=[{'time_s':0.,'position_m':[0,0,0]},{'time_s':1.,'position_m':[1,0,0]}]
    keys=[{'sample':i*1066,'time_s':i*1066/16000,'position_m':[i*.066625,0,0]} for i in range(16)]
    def no_native(_):raise AssertionError('no eligible interval should request a native RIR')
    selected,stats=adaptive_keyframes(track,keys,no_native,np.array([2,0,0]),np.array([1,0,0]))
    assert selected==keys and stats['eligible_initial_intervals']==0 and stats['added_keyframes']==0

@pytest.mark.parametrize('value',[0.,-1.])
def test_invalid_refinement_bound_rejected(value):
    with pytest.raises(ValueError):adaptive_keyframes([],[],None,None,None,minimum_step_s=value)


def test_missing_reference_is_fail_not_an_aggregation_exception():
    result=summarize_checks([])
    assert result['status']=='fail' and result['sample_count']==0

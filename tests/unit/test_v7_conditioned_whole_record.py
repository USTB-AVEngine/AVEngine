import pytest
from avengine.timeline.v7_visible_sources import conditioned_recorded_duration_fits
pytestmark=pytest.mark.fast_unit

def test_only_solver_fitted_exact_whole_recording_bypasses_short_constructor_preference():
    r={'sound_asset_id':'whole','duration_s':6.}
    assert not conditioned_recorded_duration_fits(r,'speaker',{})
    row={'motion_during_condition_plan':{'plans':[{'solution':{'status':'solved'},'config':{'sounds':{'speaker':{'sound_asset_id':'whole'}}}}]}}
    assert conditioned_recorded_duration_fits(r,'speaker',row)
    assert not conditioned_recorded_duration_fits(r,'other_speaker',row)
    assert not conditioned_recorded_duration_fits({**r,'sound_asset_id':'different'},'speaker',row)

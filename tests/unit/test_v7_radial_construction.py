import pytest
from avengine.assets.roster_motion.distance_constraints import matches_radial_window
pytestmark=pytest.mark.fast_unit

def test_half_metre_construction_allows_the_original_five_cm_retreat():
    assert matches_radial_window([0.,.3,.6,.56,.8],1)
    assert matches_radial_window([2.,1.7,1.4,1.44,1.2],-1)
    assert not matches_radial_window([0.,.3,.6,.57,.54,.9],1)
    assert not matches_radial_window([0.,.2,.3],1)


def test_distance_condition_reads_only_its_declared_sound_window():
    from avengine.assets.roster_motion.demand import matches_distance_request
    assert matches_distance_request([[3.,0.,0.],[2.7,0.,0.],[2.4,0.,0.],[2.9,0.,0.]],[0.,1.,2.,3.],[0.,0.,0.],{'distance_trend':'approaching','distance_trend_intervals_s':[[0.,2.]],'minimum_distance_change_m':.5})

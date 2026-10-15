import pytest
from avengine.timeline.v7_visibility_construction import occlusion_construction_distances
pytestmark=pytest.mark.fast_unit


def test_occlusion_proposal_radius_is_derived_from_kept_spacing_not_old_room_size_heuristic():
    p=occlusion_construction_distances();radius=p['minimum_target_radius_m'];fraction=p['foreground_depth_fraction']
    assert radius==2.8
    assert radius*fraction>=1.5
    assert radius*(1-fraction)>=1.0
    assert p['preferred_target_radius_m']>=radius


def test_larger_required_separation_or_camera_standoff_expands_the_proposal():
    p=occlusion_construction_distances(2.,2.)
    assert p['minimum_target_radius_m']*p['foreground_depth_fraction']>=2.
    assert p['minimum_target_radius_m']*(1-p['foreground_depth_fraction'])>=2.

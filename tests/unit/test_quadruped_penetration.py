import numpy as np
from avengine.assets.quadruped_penetration import inward_footprint, visible_penetration_gate


def test_crossing_patch_clipped_area_and_depth():
    target = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    source = np.array([[.1, .1, -.001], [.3, .1, .001], [.1, .3, .001]])
    area, depth = inward_footprint(source[None], target[None])
    # The negative half is a right triangle with 0.1 m legs, slightly tilted.
    assert np.isclose(area[0], .005 * np.sqrt(1.0002), rtol=1e-10)
    assert np.isclose(depth[0], .001)


def test_projection_outside_triangle_has_no_footprint():
    target = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    source = np.array([[2., 2., -.001], [3., 2., .001], [2., 3., .001]])
    area, depth = inward_footprint(source[None], target[None])
    assert area[0] == depth[0] == 0


def test_area_is_conserved_under_translation_and_scaling():
    target = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    source = np.array([[.1, .1, -.001], [.3, .1, .001], [.1, .3, .001]])
    a, d = inward_footprint(source[None], target[None])
    offset = np.array([7., -2., 4.])
    a2, d2 = inward_footprint((source * 2 + offset)[None], (target * 2 + offset)[None])
    assert np.allclose(a2, a * 4, atol=1e-12)
    assert np.allclose(d2, d * 2, atol=1e-12)


def test_visible_limits_and_internal_area_exemption():
    cross = dict(region_pair=["Body", "FL"], new_area_m2=25e-6, new_maximum_depth_m=.002)
    assert visible_penetration_gate([cross]) == []
    assert visible_penetration_gate([dict(cross, new_area_m2=25.01e-6)])[0]["area_rejected"]
    assert visible_penetration_gate([dict(cross, new_maximum_depth_m=.00201)])[0]["depth_rejected"]
    internal = dict(cross, region_pair=["FL", "FL"], new_area_m2=.1)
    assert visible_penetration_gate([internal]) == []
    assert visible_penetration_gate([dict(internal, new_maximum_depth_m=.00201)])[0]["depth_rejected"]

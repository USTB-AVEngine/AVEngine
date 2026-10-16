import numpy as np
import pytest
from types import SimpleNamespace
from avengine.rooms.walkable_map import WalkableMap, VERSION, SPECIES_RADII_M
from avengine.construction.joint.connectivity import walkable_components, reachable_point_mask
from avengine.construction.joint.routes import route_between, ConstructionError

pytestmark = pytest.mark.fast_unit


def context(mask):
    mask = np.array(mask, bool)
    m = WalkableMap([0., 0.], .05, 0., {k: mask.copy() for k in SPECIES_RADII_M},
        {"version": VERSION, "species_radii_m": SPECIES_RADII_M,
         "footprint_policy": "independent-radius-coverage-v2"})
    points = np.array([[.025, 0., .025], [.075, 0., .025], [.075, 0., .075]])
    return SimpleNamespace(walkable_map=m, points_m=points)


def test_diagonal_touch_does_not_connect_but_cardinal_bridge_does():
    c = context([[True, False], [False, True]])
    first = walkable_components(c)
    assert first.component_count == 2
    assert not first.connected(c.points_m[0], c.points_m[-1])
    np.testing.assert_array_equal(reachable_point_mask(c, c.points_m[0]), [True, False, False])
    with pytest.raises(ConstructionError, match="no_grid_route") as exc:
        route_between(c, c.points_m[0], c.points_m[-1])
    assert exc.value.details["expanded"] == 0
    # This fixture owns its in-memory mask: a changed/derived domain must not
    # accidentally borrow stale components from a previous map identity.
    c.walkable_map.layers["human"][0, 1] = True
    second = walkable_components(c)
    assert second is not first and second.component_count == 1
    assert second.connected(c.points_m[0], c.points_m[-1])
    assert len(route_between(c, c.points_m[0], c.points_m[-1])) >= 2


def test_species_cache_out_of_domain_and_read_only_labels():
    c = context([[True, True], [True, True]])
    c.walkable_map.layers["cat"][0, 1] = False
    human = walkable_components(c)
    assert walkable_components(c) is human
    assert walkable_components(c, "cat").mask_sha256 != human.mask_sha256
    assert human.labels_for_points([[-1., 0., 0.], [np.nan, 0., 0.]]).tolist() == [0, 0]
    assert not reachable_point_mask(c, [-1., 0., 0.]).any()
    with pytest.raises(ValueError):
        human.labels_zx[0, 0] = 0
    np.testing.assert_array_equal(c.walkable_map.layers["human"], np.ones((2, 2), bool))

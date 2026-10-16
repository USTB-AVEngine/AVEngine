import numpy as np
import pytest
from types import SimpleNamespace
from avengine.rooms.walkable_map import WalkableMap, VERSION, SPECIES_RADII_M
from avengine.construction.joint import BodySpec, CameraPose, MeshRaycaster, DepthBuffer, MeshInstance
from avengine.construction.joint.shadows import mesh_shadow_foreground_candidates, _clip_polygon

pytestmark = pytest.mark.fast_unit


def fixture():
    import trimesh
    mask = np.ones((120, 120), bool)
    m = WalkableMap([-3., -6.], .05, 0., {k: mask.copy() for k in SPECIES_RADII_M},
        {"version": VERSION, "species_radii_m": SPECIES_RADII_M,
         "footprint_policy": "independent-radius-coverage-v2"})
    camera = CameraPose((0., 1.1, 0.), 0., width=96, height=54)
    depth = DepthBuffer(camera, np.full((54, 96), np.inf), source="visual_mesh")
    c = SimpleNamespace(walkable_map=m, pathfinder=None, allow_native_map_fallback=False,
                        depth=lambda requested: depth)
    target = trimesh.creation.box((.4, 1.65, .2)); target.apply_translation([0., .825, 0.])
    occluder = trimesh.creation.box((1., 1.95, .3)); occluder.apply_translation([0., .975, 0.])
    transform = np.eye(4); transform[:3, 3] = [.071, 0., -5.137]
    return c, camera, MeshInstance("target", MeshRaycaster(target), transform), MeshInstance("fore", MeshRaycaster(occluder))


def test_continuous_foreground_domain_produces_registered_exact_mesh_full_shadow():
    c, camera, target, fore = fixture(); diag = {}
    result = list(mesh_shadow_foreground_candidates(c, camera, target, fore,
        target_body=BodySpec("human", .25, 1.65, 0.),
        foreground_body=BodySpec("human", .55, 1.95, 0.),
        max_candidates=2, budget_s=10., diagnostics=diag))
    assert len(result) == 2 and diag["exact_mesh_tests"] >= 2
    assert diag["free_cells"] == 14400 and diag["bounding_feasible_cells"] > 0
    for proposal in result:
        assert proposal.prediction.state == "fully_occluded"
        assert proposal.prediction.visible_pixels_estimate == 0
        assert proposal.prediction.occluder_ids == ("fore",)
        assert proposal.prediction.evidence["unknown_occluder_pixels"] == 0
        assert np.allclose(proposal.foreground.transform_m[:3, :3], fore.transform_m[:3, :3])
        assert np.linalg.norm((proposal.foreground.transform_m[:3, 3]-target.transform_m[:3, 3])[[0, 2]]) >= 1.
        assert proposal.evidence["native_pixel_truth"] is False
    np.testing.assert_array_equal(fore.transform_m, np.eye(4))
    assert diag["continuous_infeasibility_proved"] is False


def test_shadow_budget_and_missing_depth_are_not_silently_certified():
    c, camera, target, fore = fixture(); diag = {}
    assert not list(mesh_shadow_foreground_candidates(c, camera, target, fore, budget_s=0., diagnostics=diag))
    assert diag["status"] == "budget_exhausted"
    c.depth = lambda camera: None
    assert not list(mesh_shadow_foreground_candidates(c, camera, target, fore, diagnostics=diag))
    assert diag["status"] == "missing_original_visual_depth"


def test_shadow_clipping_retains_subcell_continuous_region():
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    polygon = _clip_polygon(square, np.array([1., 0.]), .137)
    polygon = _clip_polygon(polygon, np.array([-1., 0.]), -.123)
    assert np.min(polygon[:, 0]) == pytest.approx(.123)
    assert np.max(polygon[:, 0]) == pytest.approx(.137)
    assert polygon[:, 0].mean() == pytest.approx(.13)

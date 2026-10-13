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

def test_semantic_regions_do_not_depend_on_bone_weight_ownership():
    from avengine.assets.quadruped_penetration import semantic_vertex_regions
    feet=np.array([[-.12,0,-.25],[.12,0,-.25],[-.12,0,.25],[.12,0,.25]])
    points=np.vstack([[[0,.65,0],[0,.8,-.5],[0,.55,.6]],feet+[0,.25,0]])
    labels=semantic_vertex_regions(points,feet,[-.1,.6,-.6],[.1,.9,-.4],[0,.5,.6],0,1)
    np.testing.assert_array_equal(labels,np.arange(7))

def test_existing_internal_depth_is_not_new_when_pair_first_crosses():
    from avengine.assets.quadruped_penetration import relative_pair_footprints
    area,depth=relative_pair_footprints(np.array([[.0001,.0002]]),np.array([[.0125,.021]]),
                                      np.array([[.0001,.00019]]),np.array([[.012,.020]]))
    np.testing.assert_allclose(area,[[0,.00001]])
    np.testing.assert_allclose(depth,[[.0005,.001]])
    assert visible_penetration_gate([dict(region_pair=["Body","Body"],new_area_m2=float(area.sum()),new_maximum_depth_m=float(depth.max()))])==[]

def test_material_depth_preserves_rigid_motion_of_preexisting_internal_overlap():
    target=np.array([[[0.,0,0],[1.,0,0],[0,1.,0]]])
    source=np.array([[[.1,.1,-.02],[.3,.1,-.02],[.1,.3,-.02]]])
    angle=.3
    rot=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    shift=np.array([2.,3.,4.])
    _,depth=inward_footprint(source@rot.T+shift,target@rot.T+shift,source_binding=source,target_binding=target,region_rotation=rot[None])
    assert depth[0]<1e-12


def test_material_depth_measures_new_half_millimetre_inside_overlap():
    target=np.array([[[0.,0,0],[1.,0,0],[0,1.,0]]])
    source=np.array([[[.1,.1,-.02],[.3,.1,-.02],[.1,.3,-.02]]])
    _,depth=inward_footprint(source+[0,0,-.0005],target,source_binding=source,target_binding=target)
    assert np.isclose(depth[0],.0005,atol=1e-12)

def test_five_millimetre_tangential_motion_does_not_increase_depth():
    target=np.array([[[0.,0,0],[1.,0,0],[0,1.,0]]])
    source=np.array([[[.1,.1,-.02],[.3,.1,-.02],[.1,.3,-.02]]])
    _,depth=inward_footprint(source+[.005,0,0],target,source_binding=source,target_binding=target)
    assert depth[0]<1e-12

def test_cross_region_area_limit_sums_all_region_pairs():
    from avengine.assets.quadruped_penetration import visible_penetration_gate
    groups = [
        {"region_pair": ["Body", "FL"], "new_area_m2": 15e-6, "new_maximum_depth_m": .0001},
        {"region_pair": ["Body", "FR"], "new_area_m2": 15e-6, "new_maximum_depth_m": .0001},
    ]
    failures = visible_penetration_gate(groups)
    assert len(failures) == 1
    assert failures[0]["area_rejected"]
    assert failures[0]["area_scope"] == "all_different_region_pairs_in_this_frame"


def test_internal_region_area_is_excluded_from_cross_region_sum():
    from avengine.assets.quadruped_penetration import visible_penetration_gate
    groups = [
        {"region_pair": ["Body", "FL"], "new_area_m2": 10e-6, "new_maximum_depth_m": .0001},
        {"region_pair": ["FL", "FL"], "new_area_m2": .05, "new_maximum_depth_m": .001},
    ]
    assert visible_penetration_gate(groups) == []

"""Counterexamples for the shared planner; no QA truth or scorer modifications."""
import math
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from avengine.rooms.walkable_map import WalkableMap, VERSION, SPECIES_RADII_M
from avengine.construction.joint import *
from avengine.construction.joint.routes import minimum_track_distance

pytestmark = pytest.mark.fast_unit


class FlatNav:
    def snap_point(self, p):
        return np.asarray(p)


def context(tmp_path=None, holes=()):
    mask = np.ones((180, 180), bool)
    for z, x in holes:
        mask[z, x] = False
    meta = {"version": VERSION, "species_radii_m": SPECIES_RADII_M,
            "footprint_policy": "independent-radius-coverage-v2", "room_id": "fixture"}
    m = WalkableMap([-4.5, -7.5], .05, 0., {k: mask.copy() for k in SPECIES_RADII_M}, meta)
    room = {"room_label": "fixture", "floor_polygon": {"type": "Polygon", "coordinates": [
        [[-4.5, -7.5], [4.5, -7.5], [4.5, 1.5], [-4.5, 1.5], [-4.5, -7.5]]]},
        "measured_floor_y_m": 0., "placement": {"camera_m": [0., 1.25, .5]}}
    c = RoomContext(room, walkable_map=m, pathfinder=FlatNav(), cache_dir=tmp_path,
                    camera_limit=6, grid_step_m=.5)
    c.cameras = (CameraPose((0., 1.25, .5), 0.),)
    return c


def depth(camera, z=np.inf, source="visual_mesh"):
    return DepthBuffer(camera, np.full((camera.height, camera.width), z), source=source)


@pytest.mark.parametrize("yaw", [0., 90., -135., 179.])
def test_projection_sign_rotation_and_focal(yaw):
    c = CameraPose((1., 1.25, 2.), yaw)
    p = np.array(c.position_m)+4*c.forward+c.right
    uv, z = project_points(c, [p])
    assert uv[0, 0] > c.width/2 and z[0] == pytest.approx(4)
    assert uv[0, 1] == pytest.approx(c.height/2)


def test_horizontal_center_does_not_hide_cropped_feet():
    c = CameraPose((0., 1.5, 0.), 0.)
    assert framing(c, [0, 0, -1.6])[0] == "edge"
    assert framing(c, [0, 0, -4.])[0] == "inside"
    assert framing(c, [8, 0, -4.])[0] == "outside"


def test_missing_depth_is_unknown_except_wholly_outside():
    c = CameraPose((0., 1.25, 0.), 0.)
    assert predict_visibility(c, [0, 0, -4]).state == "unknown"
    assert predict_visibility(c, [8, 0, -4]).state == "out_of_view"


def test_zero_pixels_occlusion_is_not_out_of_view_and_proxy_not_real_lower_bound():
    c = CameraPose((0., 1.25, 0.), 0.)
    hidden = predict_visibility(c, [0, 0, -4], depth=depth(c, 2.))
    assert hidden.state == "fully_occluded"
    assert hidden.confidence == "outer_box_depth_margin"
    seen = predict_visibility(c, [0, 0, -4], depth=depth(c))
    assert seen.state == "visible_clear" and seen.visible_pixels_estimate > 100
    assert seen.visible_pixels_lower == 0
    assert seen.evidence["inner_proxy_visible_pixels"] > 100


def test_depth_holes_and_acoustic_proxies_cannot_prove_occlusion():
    c = CameraPose((0., 1.25, 0.), 0.)
    d = depth(c, 2.)
    d.depth_m[170:190, 310:330] = np.nan
    assert predict_visibility(c, [0, 0, -4], depth=d).state == "unknown"
    assert predict_visibility(c, [0, 0, -4], depth=depth(c, 2., "acoustic_proxy_mesh")).state == "unknown"


def test_registered_actor_proxy_reports_identity_but_not_full_occlusion_proof():
    c = CameraPose((0., 1.25, 0.), 0.)
    p = predict_visibility(c, [0, 0, -4.5], depth=depth(c),
         occluders=(("speaker", [0, 0, -3], BodySpec("slide", .25, 1.)),))
    assert p.state == "visible_occluded"
    assert p.occluder_ids == ("speaker",)
    assert .1 < p.occluded_fraction < .9
    assert p.evidence["proxy_occluders"]


def test_real_triangle_hole_stays_visible_and_forward_depth_not_range():
    import trimesh
    c = CameraPose((0., 1.25, 0.), 0., width=160, height=90)
    mesh = trimesh.creation.box((8., 4., .1)); mesh.apply_translation([0, 1., -2.])
    ray = MeshRaycaster(mesh, face_ids=["wall"]*len(mesh.faces))
    d = build_depth_buffer(c, ray)
    assert np.allclose(d.depth_m[20:60, 20:140], 1.95, atol=1e-5)
    p = predict_visibility(c, [0, 0, -4], depth=d)
    assert p.state == "fully_occluded" and p.occluder_ids == ("wall",)


def test_continuous_pair_distance_catches_between_frame_collision():
    a = np.array([[-1., 0, 0], [1., 0, 0]])
    b = -a
    assert minimum_track_distance(a, b) == pytest.approx(0)


def test_map_supercover_catches_thin_blocked_cell():
    c = context(holes=[(90, 90)])
    a, b = [-.2, 0, -3.], [.2, 0, -3.]
    with pytest.raises(ConstructionError, match="segment"):
        validate_track(c, [a, b])


def test_astar_goes_around_obstacle_and_budget_is_explicit():
    c = context(holes=[(z, 90) for z in range(40, 130)])
    a, b = [-1, 0, -3], [1, 0, -3]
    path = route_between(c, a, b)
    assert len(path) > 2
    assert all(c.walkable_map.segment_free(x[[0,2]], y[[0,2]]) for x, y in zip(path[:-1], path[1:]))
    with pytest.raises(ConstructionError, match="search_budget"):
        route_between(c, a, b, max_expansions=1)


@pytest.mark.parametrize("species,speed", [("human", 1.), ("dog", .7), ("cat", .5), ("slide", .3)])
def test_rounded_route_arclength_speed_and_full_track(species, speed):
    c = context()
    p = construct_route(c, [[-2, 0, -4], [0, 0, -4], [0, 0, -6]],
                        species=species, speed_m_s=speed, camera=c.cameras[0])
    assert p.positions_m.shape == (901, 3)
    assert 3 < p.metadata["route_length_m"] < 4
    v = np.linalg.norm(np.diff(p.positions_m[:, [0,2]], axis=0), axis=1)*30
    assert v.max() == pytest.approx(speed, abs=.003)
    np.testing.assert_allclose(p.positions_m[-1], [0,0,-6])
    assert p.metadata["validation"]["map_full_frames"] == 901


def test_dwell_forces_stop_and_second_leg_after_dwell():
    c = context()
    p = construct_route(c, [[-2,0,-4], [0,0,-4], [0,0,-6]], dwell_s=[0,1.,0])
    intervals = p.metadata["walk_intervals_s"]
    assert intervals[1][0]-intervals[0][1] == pytest.approx(1.)
    at = round((intervals[0][1]+.5)*30)
    np.testing.assert_allclose(p.positions_m[at], [0,0,-4])


def test_species_speed_floor_snap_and_parent_duration_rejected():
    c = context()
    with pytest.raises(ConstructionError, match="speed_band"):
        construct_route(c, [[0,0,-3], [0,0,-4]], species="cat", speed_m_s=1.)
    with pytest.raises(ConstructionError, match="duration"):
        construct_route(c, [[0,0,-3], [0,0,-6]], duration_s=1.)
    with pytest.raises(ConstructionError, match="changes_floor"):
        validate_track(c, [[0,3,-3]])


def test_certification_rejects_missing_nav_and_multiple_human_sound_classes():
    c = context()
    a = static_actor("a", [0,0,-4])
    p = ScenePlan(c.cameras[0], {"a": a})
    assert certify_plan(c,p).passed
    p.sounds = (SoundEvent("a","s1","speech",1,2),SoundEvent("a","s2","cough",4,5))
    assert not certify_plan(c,p).passed
    c.pathfinder = None
    assert any(x.status == "unknown" for x in certify_plan(c,p).checks)


def test_grid_cache_reused_and_changed_camera_does_not_reuse(tmp_path):
    c = context(tmp_path)
    g = c.grid(c.cameras[0])
    c2 = context(tmp_path)
    g2 = c2.grid(c2.cameras[0])
    np.testing.assert_array_equal(g.frustum_state,g2.frustum_state)
    assert c2.cache_hits == 1 and c2.cache_misses == 0
    c2.grid(replace(c2.cameras[0],yaw_deg=15.))
    assert c2.cache_misses == 1


def test_solver_no_checks_not_certified_and_plugin_bug_not_infeasibility():
    c = context()
    p = ScenePlan(c.cameras[0], {"a": static_actor("a", [0,0,-4])})
    plugin = SimpleNamespace(candidates=lambda c,r: iter([p]), verify=lambda c,r,p: ())
    result = solve(c, FamilyRequest("QA-21","class",("a",)), plugin)
    assert result.status == "unavailable" and result.plan is None
    plugin.verify = lambda c,r,p: [CheckResult("branch", "pass")]
    assert solve(c, FamilyRequest("QA-21","class",("a",)), plugin).status == "certified"
    plugin.verify = lambda c,r,p: (_ for _ in ()).throw(RuntimeError("plugin bug"))
    with pytest.raises(RuntimeError,match="plugin bug"):
        solve(c, FamilyRequest("QA-21","class",("a",)), plugin)


def test_static_role_layout_constructs_inside_outside_without_human_collision():
    c = context()
    bodies = {"target":BodySpec(), "other":BodySpec()}
    plans = list(enumerate_static_layouts(c, bodies, {"target":"inside","other":"outside"}, max_candidates=2))
    assert len(plans) == 2
    for p in plans:
        assert certify_plan(c,p).passed
        assert framing(p.camera,p.actors["target"].positions_m[0])[0] == "inside"
        assert framing(p.camera,p.actors["other"].positions_m[0])[0] == "outside"


def test_exact_mesh_refinement_keeps_occluder_hole_and_identity():
    import trimesh
    c = CameraPose((0.,1.25,0.),0.,width=160,height=90)
    target_mesh=trimesh.creation.box((.6,1.8,.2))
    t=np.eye(4);t[:3,3]=[0,.9,-4]
    target=MeshInstance("target",MeshRaycaster(target_mesh),t)
    pieces=[]
    for x in (-.5,.5):
        m=trimesh.creation.box((.6,3.,.1));m.apply_translation([x,1.,-2]);pieces.append(m)
    barrier=MeshInstance("with_gap",MeshRaycaster(trimesh.util.concatenate(pieces)))
    p=predict_mesh_visibility(c,target,scene_depth=depth(c),occluders=(barrier,))
    assert p.visible_pixels_estimate > 0
    assert p.state != "fully_occluded"
    full=trimesh.creation.box((3.,3.,.1));full.apply_translation([0,1.,-2])
    b=MeshInstance("solid",MeshRaycaster(full))
    p=predict_mesh_visibility(c,target,scene_depth=depth(c),occluders=(b,))
    assert p.state == "fully_occluded" and p.occluder_ids == ("solid",)
    assert p.evidence["target_only_pixels"] > 0
    assert not p.evidence["native_pixel_truth"]


def test_exact_mesh_scaled_transform_depth_and_target_centroid():
    import trimesh
    c=CameraPose((0.,1.25,0.),0.,width=160,height=90)
    mesh=trimesh.creation.box((.6,1.8,.2))
    t=np.eye(4);t[:3,:3]*=.5;t[:3,3]=[-1.,.45,-4.]
    target=MeshInstance("t",MeshRaycaster(mesh),t)
    d=raster_mesh_depth(c,target)
    assert np.min(d) == pytest.approx(3.95,abs=1e-5)
    p=predict_mesh_visibility(c,target,scene_depth=depth(c))
    assert p.state == "visible_clear"
    assert p.evidence["target_centroid_xy_px"][0] < c.width/2
    with pytest.raises(ValueError,match="own occluders"):
        predict_mesh_visibility(c,target,scene_depth=depth(c),occluders=(target,))


def test_depth_cache_survives_new_context_and_new_registered_occluder_isolated(tmp_path):
    import trimesh
    c=context(tmp_path)
    wall=trimesh.creation.box((9.,4.,.1));wall.apply_translation([0,1.,-6.])
    c.raycaster=MeshRaycaster(wall,evidence={"fixture":"wall"})
    c.identity["geometry"]={"fixture":"wall"};c.identity["geometry_source"]="visual_mesh"
    d=c.depth(c.cameras[0]);assert c.cache_misses==1
    c._depth={};c.raycaster.first_hit=lambda *a: (_ for _ in ()).throw(AssertionError("cache missed"))
    np.testing.assert_allclose(c.depth(c.cameras[0]).depth_m,d.depth_m)
    assert c.cache_hits==1
    obj=trimesh.creation.box((1.,2.,.1));obj.apply_translation([0,1.,-2.])
    oc=MeshInstance("registered",MeshRaycaster(obj,evidence={"fixture":"foreground"}))
    c.raycaster=MeshRaycaster(wall,evidence={"fixture":"wall"})
    altered=c.with_occluders(oc)
    assert altered._owned_video is None
    d2=altered.depth(c.cameras[0])
    assert d2.depth_m[c.cameras[0].height//2,c.cameras[0].width//2] < d.depth_m[c.cameras[0].height//2,c.cameras[0].width//2]
    assert altered._key(c.cameras[0]) != c._key(c.cameras[0])


def test_scheduler_measured_activity_respects_silent_query_and_moving_window():
    req=SoundRequirement("a","clip","speech",((.2,1.2),),(0.,5.),((2.,4.),))
    r=schedule_sound_events([req],silent_intervals_s=((4.,5.),))
    assert r.status=="solved"
    e=r.events[0]
    assert 2. <= e.start_s < e.end_s <= 3.5
    assert e.wet_end_s <= 4.
    assert e.end_s-e.start_s == pytest.approx(1.)


def test_scheduler_backtracks_starts_and_does_not_overlap_same_voice():
    a=SoundRequirement("a","one","speech",((0.,1.),),(0.,4.))
    b=SoundRequirement("a","two","speech",((0.,1.),),(0.,0.))
    r=schedule_sound_events([a,b])
    assert r.status=="solved" and r.starts_s[0] >= 1.5 and r.starts_s[1] == 0.
    assert len(r.events)==2


def test_scheduler_zero_domain_and_budget_are_different():
    a=SoundRequirement("a","one","speech",((0.,1.),),(0.,0.))
    r=schedule_sound_events([a],silent_intervals_s=((0.,3.),))
    assert r.status=="unavailable" and r.reasons[0].startswith("empty_start_domain")
    b=SoundRequirement("a","two","speech",((0.,1.),),(0.,4.))
    r=schedule_sound_events([a,b],max_nodes=1)
    assert r.status=="budget_exhausted"


def test_navmesh_drift_uses_original_native_two_millimetre_limit():
    c=context()
    c.pathfinder=SimpleNamespace(snap_point=lambda p:np.asarray(p)+[.01,0,0])
    with pytest.raises(ConstructionError,match="snap_tolerance"):
        validate_track(c,[[0,0,-3]])


def test_camera_search_includes_interior_when_inset_perimeter_blocked():
    c=context()
    for layer in c.walkable_map.layers.values():
        layer[:]=False
        layer[65:115,65:115]=True
    cameras=candidate_cameras(c.room,c.walkable_map,c.pathfinder,limit=6)
    assert cameras
    for camera in cameras:
        assert c.walkable_map.contains(np.asarray(camera.position_m)[[0,2]],"human")


def test_elevated_navmesh_does_not_float_actor_or_remove_legal_camera():
    c=context()
    c.pathfinder=SimpleNamespace(snap_point=lambda p:np.asarray(p)+[0,.166,0])
    q,report=validate_track(c,[[0,0,-3],[0,0,-4]])
    np.testing.assert_array_equal(q[:,1],[0,0])
    assert report["navmesh_y_offset_range_m"] == pytest.approx([.166,.166])
    assert candidate_cameras(c.room,c.walkable_map,c.pathfinder,limit=6)

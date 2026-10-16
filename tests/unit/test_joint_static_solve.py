"""Regressions for joint domains, emitter-distance constraints and new recipe binding."""
from copy import deepcopy
import math
from types import SimpleNamespace
import numpy as np
import pytest
from avengine.rooms.walkable_map import WalkableMap, VERSION, SPECIES_RADII_M
from avengine.construction.joint import BodySpec, CameraPose, RoomContext, ScenePlan, framing, static_actor
from avengine.construction.joint.routes import ConstructionError
from avengine.construction.joint.static_solve import enumerate_joint_static_layouts
from avengine.construction.joint.registered import registered_row_for_plan

pytestmark = pytest.mark.fast_unit


def room_context():
    mask = np.ones((240, 240), bool)
    meta = {"version": VERSION, "species_radii_m": SPECIES_RADII_M,
            "footprint_policy": "independent-radius-coverage-v2", "room_id": "static_joint"}
    walk = WalkableMap([-6., -6.], .05, 0., {key: mask.copy() for key in SPECIES_RADII_M}, meta)
    room = {"room_label": "static_joint", "floor_polygon": {"type": "Polygon", "coordinates": [
        [[-6., -6.], [6., -6.], [6., 6.], [-6., 6.], [-6., -6.]]]},
        "measured_floor_y_m": 0., "placement": {"camera_m": [0., 1.25, 0.]}}
    context = RoomContext(room, walkable_map=walk, camera_limit=1, grid_step_m=1.)
    context.cameras = ()
    return context


def single_camera(monkeypatch, context):
    import avengine.construction.joint.camera_solve as cameras
    monkeypatch.setattr(cameras, "_positions", lambda c: (
        ((np.array([0., 0., 0.]), 6., [(0., 2*math.pi)]),), {}))


def test_four_roles_are_solved_without_legacy_camera_or_point_beam():
    c = room_context()
    bodies = {key: BodySpec("human", .25, 1.7, .02) for key in ("a", "b", "c", "d")}
    roles = {"a": "inside", "b": "inside", "c": "outside", "d": "inside"}
    diag = {}
    plans = list(enumerate_joint_static_layouts(c, bodies, roles,
        minimum_distance_gaps=[("a", "b", .6)], max_candidates=2, budget_s=15., diagnostics=diag))
    assert len(plans) == 2 and not c.cameras
    assert diag["camera_sites"] > 24
    for plan in plans:
        for key, actor in plan.actors.items():
            assert framing(plan.camera, actor.positions_m[0], actor.body)[0] == roles[key]
        distance = plan.metadata["reference_distances_m"]
        assert abs(distance["a"]-distance["b"]) >= .6
        for a in bodies:
            for b in bodies:
                if a != b:
                    assert np.linalg.norm((plan.actors[a].positions_m[0]-plan.actors[b].positions_m[0])[[0, 2]]) >= 1.
    assert diag["status"] == "candidate_limit"
    assert diag["continuous_room_infeasibility_proved"] is False


def test_feasible_actor_beyond_old_first_24_points_is_retained(monkeypatch):
    c = room_context()
    c.points_m = np.array([[0., 0., -4.]] + [[i*.008, 0., -4.] for i in range(1, 36)] + [[2., 0., -4.]])
    single_camera(monkeypatch, c)
    a = np.zeros(len(c.points_m), bool); a[0] = True
    b = ~a
    body = BodySpec("human", .2, 1.7, .02)
    plans = list(enumerate_joint_static_layouts(c, {"a": body, "b": body},
        {"a": "inside", "b": "inside"}, point_masks={"a": a, "b": b}, max_candidates=1))
    assert len(plans) == 1
    assert plans[0].metadata["selected_point_indices"] == {"a": 0, "b": 36}


def test_distance_gap_uses_actual_reference_offset_not_only_root(monkeypatch):
    c = room_context()
    c.points_m = np.array([[0., 0., -4.], [0., 0., -5.2]])
    single_camera(monkeypatch, c)
    body = BodySpec("human", .2, 1.7, .02)
    masks = {"a": np.array([True, False]), "b": np.array([False, True])}
    kwargs = dict(point_masks=masks, minimum_distance_gaps=[("a", "b", .6)], max_candidates=1)
    assert list(enumerate_joint_static_layouts(c, {"a": body, "b": body},
        {"a": "inside", "b": "inside"}, **kwargs))
    diag = {}
    assert not list(enumerate_joint_static_layouts(c, {"a": body, "b": body},
        {"a": "inside", "b": "inside"}, distance_offsets_m={"b": [0., 0., 1.2]},
        diagnostics=diag, **kwargs))
    assert diag["distance_gap_rejections"] > 0


def test_exhausted_compute_budget_is_not_a_room_infeasibility_claim():
    c = room_context(); diag = {}
    assert not list(enumerate_joint_static_layouts(c, {"a": BodySpec()}, {"a": "inside"},
        budget_s=0., diagnostics=diag))
    assert diag["status"] == "budget_exhausted"
    assert diag["continuous_room_infeasibility_proved"] is False


def test_explicit_bands_and_masks_cannot_be_escaped(monkeypatch):
    c = room_context(); c.points_m = np.array([[0., 0., -4.], [0., 0., -5.2]])
    single_camera(monkeypatch, c)
    body = BodySpec("human", .2, 1.7, .02)
    assert not list(enumerate_joint_static_layouts(c, {"a": body}, {"a": "inside"},
        point_masks={"a": np.array([True, False])}, distance_bands_m={"a": (5., 6.)}))
    with pytest.raises(ValueError, match="boolean"):
        list(enumerate_joint_static_layouts(c, {"a": body}, {"a": "inside"},
            point_masks={"a": [0, 1]}))


def recipe_fixture():
    c = room_context(); camera = CameraPose((0., 1.25, 0.), 0.)
    points = {"a": [-1., 0., -4.], "b": [1., 0., -5.5], "c": [4., 0., 3.]}
    actors = {key: static_actor(key, p, BodySpec("human", .25, 1.7, .02)) for key, p in points.items()}
    for actor in actors.values():
        actor.metadata["native_yaw_rad"] = 0.
    plan = ScenePlan(camera, actors, metadata={"roles": {"a": "inside", "b": "inside", "c": "outside"},
        "construction": "joint_static_points_and_continuous_camera_yaw"})
    assets = [{"asset_id": key, "kind": "human", "mouth_local_glb_m": [.1, 1.3, .2]} for key in points]
    row = {"scene_id": "fresh", "members": ["v0_a0", "v0_a1", "v1_a0", "v1_a1"],
           "native_endpoint_distance_bands_m": {"voice1": [4.9, 6.]},
           "native_minimum_endpoint_distance_gaps": [{"endpoints": ["voice1", "voice2"], "minimum_difference_m": .6}],
           "native_view_roles": {"c": {"view": "outside", "minimum_abs_azimuth_deg": 60.}},
           "production_scenario": {"asset_ids": ["a", "b", "c"], "semantic_epoch_sound_ids": [["s1", "s2", "s3"]]},
           "measured_static_layout": {"valid": [], "listener_m": [9., 9., 9.]}}
    return c, row, plan, assets


def test_registered_recipe_records_heuristics_and_preserves_answer_gap_program_and_members():
    c, original, plan, assets = recipe_fixture(); before = deepcopy(original)
    row = registered_row_for_plan(c, original, plan, assets,
        record_only_distance_bands=True, record_only_offscreen_angle=True)
    assert original == before
    assert "native_endpoint_distance_bands_m" not in row
    assert row["construction_constraint_observation"]["native_endpoint_distance_bands_m"]["requested"] == before["native_endpoint_distance_bands_m"]
    assert row["native_minimum_endpoint_distance_gaps"] == before["native_minimum_endpoint_distance_gaps"]
    assert row["members"] == before["members"]
    assert row["production_scenario"]["semantic_epoch_sound_ids"] == [["s1", "s2", "s3"]]
    assert row["native_view_roles"]["c"]["minimum_abs_azimuth_deg"] == 0.
    assert row["native_view_roles"]["c"]["view"] == "outside"
    np.testing.assert_allclose(row["measured_static_layout"]["valid"][0]["position_m"], [-.9, 1.3, -3.8])
    assert row["listener_m"] == list(plan.camera.position_m)
    assert row["joint_geometry_proposal"]["native_verified"] is False


def test_registered_recipe_never_silently_removes_caller_constraints():
    c, original, plan, assets = recipe_fixture()
    row = registered_row_for_plan(c, original, plan, assets)
    assert row["native_endpoint_distance_bands_m"] == original["native_endpoint_distance_bands_m"]
    assert row["native_view_roles"]["c"]["minimum_abs_azimuth_deg"] == 60.
    plan.actors["a"].positions_m[-1, 0] += 1.
    with pytest.raises(ConstructionError, match="static"):
        registered_row_for_plan(c, original, plan, assets)


def test_visible_roles_prefer_separate_projections_over_hidden_shirt(monkeypatch):
    c = room_context()
    c.points_m = np.array([[0., 0., -6.], [0., 0., -4.], [2., 0., -4.]])
    single_camera(monkeypatch, c)
    body = BodySpec("human", .3, 1.7, .02)
    plans = list(enumerate_joint_static_layouts(c, {"a": body, "b": body},
        {"a": "inside", "b": "inside"},
        point_masks={"a": np.array([True, False, False]), "b": np.array([False, True, True])},
        max_candidates=1))
    assert plans[0].metadata["selected_point_indices"]["b"] == 2

import numpy as np
import pytest

from avengine.rooms.room_polygon import (RoomPolygon, RoomPolygonError,
    PolygonWalkableSpace, constrain_space_to_room, constrain_pathfinder_to_room)
from avengine.rooms.walkable_space import RasterWalkableSpace, HabitatWalkableSpace, camera_grid
from avengine.routes.raster_pathfinder import RasterPathfinder
from avengine.rooms import conditioned_sampler as cs

pytestmark = pytest.mark.fast_unit

L_SHAPE = {"type": "Polygon", "coordinates": [
    [[0, 0], [6, 0], [6, 2], [2, 2], [2, 6], [0, 6], [0, 0]],
    [[.8, .8], [1.2, .8], [1.2, 1.2], [.8, 1.2], [.8, .8]]]}


def space():
    pf = RasterPathfinder(np.ones((24, 24), dtype=bool),
                          bounds_m=[[0, -1, 0], [6, 1, 6]], floor_height_m=0.)
    return RasterWalkableSpace(pf, {"floor_height_m": 0., "resolution_m": .25})


def test_fills_holes_but_preserves_concavity_and_strict_boundary():
    p = RoomPolygon(L_SHAPE)
    assert p.placement_contains([1., 0., 1.])
    assert not p.placement_contains([3., 0., 3.])
    assert not p.placement_contains([0., 0., 1.])
    assert not p.placement_contains([.249, 0., 1.])
    assert p.placement_contains([.25, 0., 1.])
    assert RoomPolygon(L_SHAPE, boundary_margin_m=.1).placement_contains([.2, 0., 1.])


def test_multipolygon_retains_disconnected_components():
    second = [[[10, 0], [12, 0], [12, 2], [10, 2], [10, 0]]]
    p = RoomPolygon({"type": "MultiPolygon", "coordinates": [L_SHAPE["coordinates"], second]})
    assert p.placement_contains([11., 0., 1.])
    assert not p.placement_contains([8., 0., 1.])
    assert p.route_reason([[1, 0, 1], [11, 0, 1]]) is not None


def test_navigable_sampling_and_camera_grid_filter():
    raw = space()
    constrained = constrain_space_to_room(raw, {"floor_polygon_xz_m": L_SHAPE})
    rng = np.random.default_rng(42)
    points = [constrained.sample_navigable(rng) for _ in range(500)]
    assert constrained.room_polygon.placement_mask(points).all()
    assert all(raw.is_navigable(point) for point in points)
    grid = camera_grid(constrained)
    assert grid and len(grid) < len(camera_grid(raw))
    assert constrained.room_polygon.placement_mask(grid).all()
    assert camera_grid(raw, floor_polygon_xz_m=L_SHAPE) == grid


def test_internal_furniture_holes_still_blocked_by_navigation():
    binary = np.ones((24, 24), dtype=bool)
    binary[3:5, 3:5] = False
    pf = RasterPathfinder(binary, bounds_m=[[0, -1, 0], [6, 1, 6]], floor_height_m=0.)
    raw = RasterWalkableSpace(pf, {"floor_height_m": 0., "resolution_m": .25})
    bounded = constrain_space_to_room(raw, {"floor_polygon_xz_m": L_SHAPE})
    assert bounded.room_polygon.placement_contains([1., 0., 1.])
    assert not bounded.is_navigable([1., 0., 1.])
    assert not any(np.allclose(p[[0, 2]], [1., 1.]) for p in bounded.points())


def test_routes_check_vertices_densification_and_whole_segments():
    p = RoomPolygon(L_SHAPE)
    assert p.route_reason([[5, 0, 1], [1, 0, 1], [1, 0, 5]]) is None
    assert p.route_reason([[5, 0, 1], [1, 0, 5]]) is not None
    samples = p.route_samples([[5, 0, 1], [1, 0, 1], [1, 0, 5]])
    assert np.max(np.linalg.norm(np.diff(samples, axis=0), axis=1)) <= .25
    assert [1., 0., 1.] in samples.tolist()
    assert p.route_reason([[.1, 0, 1], [1, 0, 1]]) == "room_polygon_route_endpoint_below_boundary_margin"


def test_paths_are_not_eroded_and_have_explicit_rounding_tolerance():
    p = RoomPolygon(L_SHAPE)
    assert p.route_reason([[1, 0, 1], [.05, 0, 2], [1, 0, 5]]) is None
    assert p.route_reason([[1, 0, 1], [-.09, 0, 2], [1, 0, 5]]) is None
    assert p.route_reason([[1, 0, 1], [-.11, 0, 2], [1, 0, 5]]) is not None


def test_shortest_path_rejects_exit_even_with_legal_endpoints():
    bounded = constrain_space_to_room(space(), {"floor_polygon_xz_m": L_SHAPE})
    assert bounded.shortest_path([5., 0., 1.], [1., 0., 5.]) is None
    assert bounded.last_polygon_failure.startswith("room_polygon_route_")


def test_route_bank_is_filtered_before_native_group_selection():
    raw = space()
    raw.route_bank = lambda: [
        {"route_id": "inside", "points_m": [[1, 0, 1], [1, 0, 5]]},
        {"route_id": "outside", "points_m": [[5, 0, 1], [1, 0, 5]]},
        {"route_id": "edge", "points_m": [[.1, 0, 1], [1, 0, 1]]}]
    bounded = constrain_space_to_room(raw, {"floor_polygon_xz_m": L_SHAPE})
    assert [r["route_id"] for r in bounded.route_bank()] == ["inside"]


def test_route_retry_has_limit_and_explicit_reason(monkeypatch):
    seen = []
    def candidate(*args, **kwargs):
        seen.append(1)
        return ([np.array([[5., 0, 1.], [1., 0, 5.]])], None, None, None, None, {})
    monkeypatch.setattr(cs, "_sample_routes", candidate)
    room = {"floor_polygon_xz_m": L_SHAPE, "room_polygon_placement": {"max_route_attempts": 3}}
    with pytest.raises(cs.CandidateFailure, match="room_polygon_route_attempts_exhausted:3"):
        cs.sample_routes(space(), [], {}, {}, np.random.default_rng(1), room=room)
    assert len(seen) == 3


def test_route_retry_accepts_later_legal_candidate(monkeypatch):
    candidates = iter([[[5, 0, 1], [1, 0, 5]], [[1, 0, 1], [1, 0, 5]]])
    monkeypatch.setattr(cs, "_sample_routes", lambda *a, **k:
                        ([np.array(next(candidates), float)], None, None, None, None, {}))
    result = cs.sample_routes(space(), [], {}, {}, np.random.default_rng(1),
                             room={"floor_polygon_xz_m": L_SHAPE})
    assert result[0][0][-1].tolist() == [1., 0., 5.]


def test_no_polygon_preserves_identity_rng_and_outputs():
    raw = space()
    assert constrain_space_to_room(raw, {"room_id": "legacy"}) is raw
    assert constrain_pathfinder_to_room(raw.pathfinder, {}) is raw.pathfinder
    a, b = np.random.default_rng(17), np.random.default_rng(17)
    assert np.array_equal(raw.sample_navigable(a),
                          constrain_space_to_room(raw, {}).sample_navigable(b))
    assert a.bit_generator.state == b.bit_generator.state


def test_empty_margin_fails_without_unbounded_sampling():
    tiny = {"type": "Polygon", "coordinates": [[[0, 0], [.3, 0], [.3, .3], [0, .3], [0, 0]]]}
    bounded = constrain_space_to_room(space(), {"floor_polygon_xz_m": tiny})
    with pytest.raises(RoomPolygonError, match="no_navigable_points_with_boundary_margin"):
        bounded.sample_navigable(np.random.default_rng(1))
    assert camera_grid(bounded) == []


def test_native_sampling_has_bound_and_does_not_create_a_simulator():
    class PF:
        calls = 0
        def seed(self, seed): pass
        def get_bounds(self): return [[0, -1, 0], [6, 1, 6]]
        def get_random_navigable_point(self):
            self.calls += 1
            return [3., 0., 3.]
        def is_navigable(self, point): return True
    pf = PF()
    bounded = constrain_space_to_room(HabitatWalkableSpace(pf, {"floor_height_m": 0.}),
        {"floor_polygon_xz_m": L_SHAPE, "room_polygon_placement": {"max_point_attempts": 7}})
    with pytest.raises(RoomPolygonError, match="point_attempts_exhausted:7"):
        bounded.sample_navigable(np.random.default_rng(1))
    assert pf.calls == 7


@pytest.mark.parametrize("geometry", [
    None,
    {"type": "Point", "coordinates": [1, 1]},
    {"type": "Polygon", "coordinates": []},
    {"type": "Polygon", "coordinates": [[[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]]]}])
def test_malformed_polygons_fail_closed(geometry):
    with pytest.raises(RoomPolygonError, match="invalid floor_polygon_xz_m"):
        RoomPolygon(geometry)

def test_v7_retained_standing_placements_are_filtered_without_changing_old_layout():
    from avengine.timeline.v7_registered_sampling import _placements
    layout = {"valid": [{"position_m": [1., 1.3, 1.]}, {"position_m": [3., 1.3, 3.]}],
              "listener_m": [1., 1.55, 1.], "room_floor_polygon": L_SHAPE}
    assert _placements(layout, 2) == layout["valid"]
    assert _placements(layout, 1, room={"floor_polygon_xz_m": L_SHAPE}) == layout["valid"][:1]
    assert len(layout["valid"]) == 2

def test_reusing_a_space_for_another_room_rebinds_the_polygon():
    raw = space()
    first = constrain_space_to_room(raw, {"floor_polygon_xz_m": L_SHAPE})
    smaller = {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}
    second = constrain_space_to_room(first, {"floor_polygon_xz_m": smaller})
    assert second.base is raw
    assert not second.room_polygon.placement_contains([1., 0., 5.])


def test_request_margin_is_kept_when_sample_routes_reuses_loaded_room():
    room = {"floor_polygon_xz_m": L_SHAPE}
    bounded = constrain_space_to_room(space(), room, {"room_polygon_placement": {"boundary_margin_m": .5}})
    assert constrain_space_to_room(bounded, room) is bounded
    assert bounded.room_polygon.boundary_margin_m == .5

def test_catalog_polygon_survives_explicit_package_expansion(monkeypatch):
    from avengine.rooms import room_package
    from avengine.rooms.room_providers import planning_room_mapping
    package = {"room_id": "r", "planning_inputs": {}, "legacy_catalog_entry": {
        "floor_polygon_xz_m": {"type": "Polygon", "coordinates": [
            [[0, 0], [6, 0], [6, 6], [0, 6], [0, 0]]]}, "floor_y_m": 5.}}
    monkeypatch.setattr(room_package, "_package_from_catalog_entry", lambda *a, **k: package.copy())
    entry = {"room_id": "r", "room_package": package, "floor_polygon_xz_m": L_SHAPE, "floor_y_m": 0.}
    expanded = room_package.package_from_catalog_entry(entry)
    room = planning_room_mapping(expanded)
    assert room["floor_y_m"] == 0.
    assert room["floor_polygon_xz_m"] == L_SHAPE
    bounded = constrain_space_to_room(space(), room)
    assert bounded.room_polygon.placement_contains([1., 0., 1.])
    assert not bounded.room_polygon.placement_contains([3., 0., 3.])


def test_legacy_package_path_does_not_trigger_geometry_or_change_sampling():
    raw = space()
    room = {"room_id": "legacy", "room_package": "registered/room.json"}
    assert constrain_space_to_room(raw, room) is raw
    assert constrain_pathfinder_to_room(raw.pathfinder, room) is raw.pathfinder


def test_v7_additional_candidates_use_split_exterior_and_margin(monkeypatch):
    from avengine.timeline.v7_registered_sampling import build_registered_scene, REGISTERED_INPUT_SCHEMA
    from avengine.timeline import v7_environmental_sources
    monkeypatch.setattr(v7_environmental_sources, "attach_environmental_program", lambda *args: None)
    room = {"room_label": "r", "placement": {}, "floor_polygon_xz_m": L_SHAPE}
    house = {"house": "h", "split": "train", "acoustic_status": "pass",
             "checked_room_ids": ["r"], "checked_rooms": [room]}
    layout = {"valid": [{"position_m": [1., 1.3, 1.], "body_yaw_deg": 0.},
                        {"position_m": [1., 1.3, 5.], "body_yaw_deg": 0.}],
              "listener_m": [1.5, 1.55, 1.5], "room_floor_polygon": {
                  "type": "Polygon", "coordinates": [[[0, 0], [6, 0], [6, 6], [0, 6], [0, 0]]]}}
    row = {"schema": REGISTERED_INPUT_SCHEMA, "scene_id": "s", "scene_seed": 1,
           "house": "h", "split": "train", "room_id": "r", "seeds": {"audio": 1},
           "production_scenario": {"speaker_count": 0, "camera_yaw_deg": 0.,
               "non_speech_class": None, "environment_sound_classes": ["ambient"]},
           "measured_static_layout": layout, "window_start_ticks": [0, 100]}
    assets = [{"endpoint_id": "a", "visual_gender": "male"},
              {"endpoint_id": "b", "visual_gender": "female"}]
    working, scene, _ = build_registered_scene(row, house, [], assets)
    polygon = RoomPolygon(L_SHAPE)
    assert scene["native_placement_candidates"]
    assert polygon.placement_mask(scene["native_placement_candidates"]).all()
    assert working["room"]["floor_polygon_xz_m"] == L_SHAPE
    assert all(polygon.placement_contains(source["trajectory"][0]["position_m"])
               for source in scene["sources"].values())
    row["measured_static_layout"]["listener_m"] = [3., 1.55, 3.]
    with pytest.raises(ValueError, match="room_polygon_retained_camera_below_boundary_margin"):
        build_registered_scene(row, house, [], assets)

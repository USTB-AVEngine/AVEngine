"""Room-mesh readings for static placements, on a synthetic room with a roof.

The room is a closed box four metres square and three metres tall, with a
separate roof slab a metre above its ceiling.  That is the shape that made the
two D5 episodes fail: a point between the ceiling and the roof is covered from
above, so a vertical probe alone calls it indoors, and only the line to a
walkable point in the room tells the two apart.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from avengine.qa.answerability import MeshHandle
from avengine.rooms.placement_geometry import (
    PlacementCheckConfig,
    acoustic_pose_refusal,
    body_span_along_normal,
    check_acoustic_scene_points,
    evaluate_placement,
    local_mesh,
    measure_contact_offset,
    nearest_surface_distance,
    segment_hits,
    vertical_enclosure,
)
from avengine.rooms.source_placement import (
    SourcePlacementError,
    plan_source_placement,
    plan_static_source_placements,
)

ROOM_X = 4.0
ROOM_Y = 3.0
ROOM_Z = 4.0
ROOF_Y = 4.0


def _quad(a, b, c, d, vertices, triangles):
    base = len(vertices)
    vertices.extend([a, b, c, d])
    triangles.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])


def room_mesh():
    """A closed box room plus a roof slab a metre above its ceiling."""
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    x, y, z = ROOM_X, ROOM_Y, ROOM_Z
    _quad([0, 0, 0], [x, 0, 0], [x, 0, z], [0, 0, z], vertices, triangles)      # floor
    _quad([0, y, 0], [x, y, 0], [x, y, z], [0, y, z], vertices, triangles)      # ceiling
    _quad([0, 0, 0], [0, y, 0], [0, y, z], [0, 0, z], vertices, triangles)      # wall x=0
    _quad([x, 0, 0], [x, y, 0], [x, y, z], [x, 0, z], vertices, triangles)      # wall x=X
    _quad([0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0], vertices, triangles)      # wall z=0
    _quad([0, 0, z], [x, 0, z], [x, y, z], [0, y, z], vertices, triangles)      # wall z=Z
    _quad([-1, ROOF_Y, -1], [x + 1, ROOF_Y, -1], [x + 1, ROOF_Y, z + 1],
          [-1, ROOF_Y, z + 1], vertices, triangles)                             # roof
    return MeshHandle(np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64))


def walkable_points():
    """Standing eye points on a coarse grid inside the room."""
    return np.asarray([[px, 1.5, pz]
                       for px in np.arange(0.5, ROOM_X, 0.5)
                       for pz in np.arange(0.5, ROOM_Z, 0.5)], dtype=float)


# A 10 cm deep box whose contact plane is its own -x face, so the body grows
# along its plane normal. This is the wall-mounted shape.
WALL_BOX = {
    "bounds_min_m": [0.0, -0.05, -0.05],
    "bounds_max_m": [0.10, 0.05, 0.05],
    "plane_normal_m": [1.0, 0.0, 0.0],
    "base_plane_offset_m": 0.0,
}
# A 5 cm deep disc whose contact plane is its own +y face, so the body hangs
# along the opposite direction. This is the ceiling-mounted shape.
CEILING_DISC = {
    "bounds_min_m": [-0.04, 0.0, -0.04],
    "bounds_max_m": [0.04, 0.05, 0.04],
    "plane_normal_m": [0.0, 1.0, 0.0],
    "base_plane_offset_m": 0.05,
}


def _corners(box, contact, mounting, frame_u, frame_v):
    """World corners of a box seated with its contact face on ``contact``."""
    low = np.asarray(box["bounds_min_m"], dtype=float)
    high = np.asarray(box["bounds_max_m"], dtype=float)
    normal = np.asarray(box["plane_normal_m"], dtype=float)
    origin = normal * box["base_plane_offset_m"]
    basis = np.stack([np.asarray(frame_u, float), np.asarray(frame_v, float),
                      np.asarray(mounting, float)])
    span = body_span_along_normal(low, high, normal, box["base_plane_offset_m"])
    local_axes = [axis for axis in range(3) if abs(normal[axis]) < 0.5]
    out = []
    for first in (low[local_axes[0]], high[local_axes[0]]):
        for second in (low[local_axes[1]], high[local_axes[1]]):
            for depth in (0.0, span["depth_m"]):
                offset = np.zeros(3)
                offset[local_axes[0]] = first - origin[local_axes[0]]
                offset[local_axes[1]] = second - origin[local_axes[1]]
                out.append(np.asarray(contact, float)
                           + basis[0] * offset[local_axes[0]]
                           + basis[1] * offset[local_axes[1]]
                           + np.asarray(mounting, float) * depth)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------
# The primitives


def test_segment_hits_finds_both_faces_of_the_room():
    mesh = room_mesh()
    # Off the quads' own diagonals, so each surface is met exactly once.
    upward = segment_hits(mesh, [1.0, 1.5, 2.0], [0.0, 1.0, 0.0], t_max=10.0)
    assert [round(float(value), 3) for value in upward] == [1.5, 2.5]


def test_vertical_enclosure_separates_the_room_the_plenum_and_the_roof():
    mesh = room_mesh()
    inside = vertical_enclosure(mesh, [2.0, 1.5, 2.0], margin_m=0.5)
    plenum = vertical_enclosure(mesh, [2.0, 3.5, 2.0], margin_m=0.5)
    on_roof = vertical_enclosure(mesh, [2.0, 4.2, 2.0], margin_m=0.5)
    assert inside["status"] == "pass"
    # The roof covers the plenum too, which is exactly why the vertical probe
    # alone cannot decide and the walkable line has to.
    assert plenum["status"] == "pass"
    assert on_roof["status"] == "fail"


def test_nearest_surface_distance_is_exact_inside_its_radius():
    mesh = room_mesh()
    assert nearest_surface_distance(mesh, [0.02, 1.5, 2.0], radius=0.05) == pytest.approx(0.02)
    assert nearest_surface_distance(mesh, [2.0, 1.5, 2.0], radius=0.05) is None


def test_local_mesh_keeps_the_reading_and_drops_the_rest_of_the_room():
    mesh = room_mesh()
    near = local_mesh(mesh, [0.0, 1.5, 2.0], radius=0.3)
    assert near.source["triangle_count"] < near.source["room_triangle_count"]
    assert [round(float(v), 3) for v in segment_hits(near, [0.1, 1.5, 2.0], [-1.0, 0.0, 0.0], t_max=0.2)] \
        == [round(float(v), 3) for v in segment_hits(mesh, [0.1, 1.5, 2.0], [-1.0, 0.0, 0.0], t_max=0.2)]


def test_body_span_reads_the_mounting_direction_from_the_measured_bounds():
    wall = body_span_along_normal(WALL_BOX["bounds_min_m"], WALL_BOX["bounds_max_m"],
                                  WALL_BOX["plane_normal_m"], WALL_BOX["base_plane_offset_m"])
    ceiling = body_span_along_normal(CEILING_DISC["bounds_min_m"], CEILING_DISC["bounds_max_m"],
                                     CEILING_DISC["plane_normal_m"], CEILING_DISC["base_plane_offset_m"])
    assert wall["sign"] == 1.0 and wall["depth_m"] == pytest.approx(0.10)
    assert ceiling["sign"] == -1.0 and ceiling["depth_m"] == pytest.approx(0.05)


# --------------------------------------------------------------------------
# The seat measurement


def test_contact_offset_moves_a_fitted_plane_onto_the_real_wall():
    mesh = room_mesh()
    config = PlacementCheckConfig(max_contact_search_m=0.25)
    measured = measure_contact_offset(
        mesh, plane_point=[0.08, 1.5, 2.0], out_direction=[1.0, 0.0, 0.0],
        body_depth_m=0.10, config=config,
    )
    assert measured["status"] == "measured"
    assert measured["surface_offset_m"] == pytest.approx(-0.08)
    assert measured["offset_m"] == pytest.approx(-0.08 + config.contact_gap_m)


def test_contact_offset_refuses_a_plane_further_than_the_declared_tolerance():
    mesh = room_mesh()
    measured = measure_contact_offset(
        mesh, plane_point=[0.50, 1.5, 2.0], out_direction=[1.0, 0.0, 0.0],
        body_depth_m=0.10, config=PlacementCheckConfig(max_contact_search_m=0.25),
    )
    assert measured["status"] == "fail"
    assert "plane tolerance" in measured["reason"]


def test_contact_offset_seats_on_the_most_protruding_part_of_the_footprint():
    """A plane that leans against the wall seats on the corner that sticks out most."""
    mesh = room_mesh()
    config = PlacementCheckConfig(max_contact_search_m=0.25)
    # Two footprint probes on the same fitted plane, one of them 2 cm deeper
    # into the wall than the other; the seat has to clear that one as well.
    measured = measure_contact_offset(
        mesh, plane_point=[0.08, 1.5, 2.0], out_direction=[1.0, 0.0, 0.0],
        body_depth_m=0.10, config=config,
        footprint_points=[[0.08, 1.4, 2.0], [0.08, 1.6, 2.0]],
    )
    assert measured["status"] == "measured"
    assert measured["probe_count"] == 3


# --------------------------------------------------------------------------
# The seated placement


def test_a_seated_wall_box_passes_every_reading():
    mesh = room_mesh()
    contact = np.asarray([0.003, 1.5, 2.0])
    mounting = np.asarray([1.0, 0.0, 0.0])
    corners = _corners(WALL_BOX, contact, mounting, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    report = evaluate_placement(
        mesh, contact_point=contact, out_direction=mounting,
        plane_u=[0.0, 1.0, 0.0], plane_v=[0.0, 0.0, 1.0], body_depth_m=0.10,
        corners_m=corners, emitter_point=contact + mounting * 0.05,
        interior_points=walkable_points(), config=PlacementCheckConfig(max_contact_search_m=0.25),
    )
    assert report["status"] == "pass", report["failed_checks"]


def test_a_box_mounted_into_the_wall_is_refused_as_penetration():
    mesh = room_mesh()
    contact = np.asarray([0.003, 1.5, 2.0])
    mounting = np.asarray([-1.0, 0.0, 0.0])
    corners = _corners(WALL_BOX, contact, mounting, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    report = evaluate_placement(
        mesh, contact_point=contact, out_direction=mounting,
        plane_u=[0.0, 1.0, 0.0], plane_v=[0.0, 0.0, 1.0], body_depth_m=0.10,
        corners_m=corners, emitter_point=contact + mounting * 0.05,
        interior_points=walkable_points(), config=PlacementCheckConfig(max_contact_search_m=0.25),
    )
    assert report["status"] == "fail"
    assert "body_volume" in report["failed_checks"]


def test_a_placement_in_the_plenum_is_refused_even_though_the_roof_covers_it():
    """A disc hung from the roof is seated, free and covered, and still outside."""
    mesh = room_mesh()
    contact = np.asarray([1.0, ROOF_Y - 0.003, 2.0])
    mounting = np.asarray([0.0, -1.0, 0.0])
    corners = _corners(CEILING_DISC, contact, mounting, [1.0, 0.0, 0.0], [0.0, 0.0, -1.0])
    report = evaluate_placement(
        mesh, contact_point=contact, out_direction=mounting,
        plane_u=[1.0, 0.0, 0.0], plane_v=[0.0, 0.0, -1.0], body_depth_m=0.05,
        corners_m=corners, emitter_point=contact + mounting * 0.03,
        interior_points=walkable_points(), config=PlacementCheckConfig(max_contact_search_m=0.25),
    )
    assert report["failed_checks"] == ["interior"]
    assert report["support_contact"]["status"] == "pass"
    assert report["body_volume"]["status"] == "pass"
    assert report["interior"]["vertical_cover"]["status"] == "pass"
    assert report["interior"]["walkable_reference"]["status"] == "fail"


def test_an_emitter_on_the_wall_is_refused_for_clearance():
    mesh = room_mesh()
    contact = np.asarray([0.003, 1.5, 2.0])
    mounting = np.asarray([1.0, 0.0, 0.0])
    corners = _corners(WALL_BOX, contact, mounting, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    report = evaluate_placement(
        mesh, contact_point=contact, out_direction=mounting,
        plane_u=[0.0, 1.0, 0.0], plane_v=[0.0, 0.0, 1.0], body_depth_m=0.10,
        corners_m=corners, emitter_point=[0.01, 1.5, 2.0],
        interior_points=walkable_points(), config=PlacementCheckConfig(max_contact_search_m=0.25),
    )
    assert report["status"] == "fail"
    assert "emitter_clearance" in report["failed_checks"]


# --------------------------------------------------------------------------
# The planner, end to end on the synthetic room


def _registry(bounds_min, bounds_max, normal, offset, emitter_offset):
    basis_u, basis_v = _orthogonal_basis(normal)
    return {
        "schema": "avengine_source_asset_runtime_registry_v1",
        "assets": [{
            "asset_id": "fixture_device", "revision": "fixture_v1",
            "entity_class": "rigid_object", "default_emitter_anchor_id": "speaker",
            "emitter_anchors": [{"anchor_id": "speaker", "anchor_type": "object_speaker",
                                 "offset_m": list(emitter_offset),
                                 "offset_space": "final_scaled_asset_root"}],
            "runtime_backends": {"habitat": {"resting_pose": {
                "attachment_surface": "wall", "base_plane_offset_m": offset,
                "plane_normal_m": list(normal), "plane_basis_u_m": list(basis_u),
                "plane_basis_v_m": list(basis_v), "footprint_extent_m": [0.1, 0.1],
                "height_m": 0.1, "measured_plane": "wall_back"}}},
        }],
    }


def _orthogonal_basis(normal):
    normal = np.asarray(normal, dtype=float)
    helper = np.asarray([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.asarray([1.0, 0.0, 0.0])
    basis_u = np.cross(helper, normal)
    basis_u = basis_u / np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    return basis_u.tolist(), basis_v.tolist()


def _layout(normal, origin):
    basis_u, basis_v = _orthogonal_basis(normal)
    return {"support_surfaces": [{
        "surface_id": "wall_01", "surface_kind": "wall", "room_id": "room_fixture",
        "origin_m": list(origin), "normal_m": list(normal),
        "basis_u_m": basis_u, "basis_v_m": basis_v,
        "bounds_u_m": [-0.5, 0.5], "bounds_v_m": [-0.5, 0.5],
        "geometry_ref": {"authority": "visual_geometry",
                         "geometry_id": "room_visual_fixture_v1",
                         "triangle_indices": [0, 1]},
    }]}


def _visual_geometry(origin, normal):
    basis_u, basis_v = (np.asarray(value) for value in _orthogonal_basis(normal))
    origin = np.asarray(origin, dtype=float)
    corners = [origin + basis_u * u + basis_v * v
               for u, v in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))]
    return {"authority": "visual_geometry", "geometry_id": "room_visual_fixture_v1",
            "source_ref": "fixture://room/visual",
            "vertices_m": [corner.tolist() for corner in corners],
            "triangles": [[0, 1, 2], [0, 2, 3]]}


def _config():
    return {"normal_tolerance_deg": 1.0, "plane_tolerance_m": 0.25,
            "min_inter_instance_gap_m": 0.0,
            "candidate_search": {"grid_step_m": 0.2, "max_candidates": 9,
                                 "edge_margin_m": 0.05}}


def test_planner_flips_a_support_normal_that_points_out_of_the_room():
    """The fitted wall normal points into the wall; the asset belongs on the room side."""
    mesh = room_mesh()
    normal = [-1.0, 0.0, 0.0]
    origin = [0.0, 1.5, 2.0]
    result = plan_source_placement(
        _registry(WALL_BOX["bounds_min_m"], WALL_BOX["bounds_max_m"],
                  WALL_BOX["plane_normal_m"], WALL_BOX["base_plane_offset_m"], [0.05, 0.0, 0.0]),
        {"room_id": "room_fixture"}, _layout(normal, origin), _visual_geometry(origin, normal),
        {"instance_id": "source1", "asset_id": "fixture_device",
         "support_surface_id": "wall_01", "candidate_index": 4, "yaw_deg": 0.0,
         "asset_geometry": {"source_ref": "fixture://asset",
                            "bounds_min_m": WALL_BOX["bounds_min_m"],
                            "bounds_max_m": WALL_BOX["bounds_max_m"],
                            "plane_normal_m": WALL_BOX["plane_normal_m"],
                            "base_plane_offset_m": WALL_BOX["base_plane_offset_m"]}},
        config=_config(), room_mesh=mesh, interior_points=walkable_points(),
    )
    checks = result["placement_checks"]
    assert checks["status"] == "pass", checks["failed_checks"]
    assert checks["normal_flipped"] is True
    # The asset now grows into the room rather than through the wall.
    assert result["root_transform"]["translation_m"][0] > 0.0
    assert result["emitter_transform"]["position_m"][0] > 0.0


def test_planner_seats_a_fitted_plane_that_floats_off_the_wall():
    mesh = room_mesh()
    normal = [1.0, 0.0, 0.0]
    origin = [0.12, 1.5, 2.0]
    result = plan_source_placement(
        _registry(WALL_BOX["bounds_min_m"], WALL_BOX["bounds_max_m"],
                  WALL_BOX["plane_normal_m"], WALL_BOX["base_plane_offset_m"], [0.05, 0.0, 0.0]),
        {"room_id": "room_fixture"}, _layout(normal, origin), _visual_geometry(origin, normal),
        {"instance_id": "source1", "asset_id": "fixture_device",
         "support_surface_id": "wall_01", "candidate_index": 4, "yaw_deg": 0.0,
         "asset_geometry": {"source_ref": "fixture://asset",
                            "bounds_min_m": WALL_BOX["bounds_min_m"],
                            "bounds_max_m": WALL_BOX["bounds_max_m"],
                            "plane_normal_m": WALL_BOX["plane_normal_m"],
                            "base_plane_offset_m": WALL_BOX["base_plane_offset_m"]}},
        config=_config(), room_mesh=mesh, interior_points=walkable_points(),
    )
    checks = result["placement_checks"]
    assert checks["status"] == "pass", checks["failed_checks"]
    # The fitted plane floated 12 cm off the wall; the asset is seated on the wall.
    assert checks["mesh_contact_offset"]["surface_offset_m"] == pytest.approx(-0.12, abs=1e-6)
    assert result["support_identity"]["contact_point_m"][0] == pytest.approx(0.003, abs=1e-6)


def test_planner_without_a_mesh_plans_exactly_what_the_catalog_states():
    normal = [1.0, 0.0, 0.0]
    origin = [0.12, 1.5, 2.0]
    request = {"instance_id": "source1", "asset_id": "fixture_device",
               "support_surface_id": "wall_01", "candidate_index": 4, "yaw_deg": 0.0,
               "asset_geometry": {"source_ref": "fixture://asset",
                                  "bounds_min_m": WALL_BOX["bounds_min_m"],
                                  "bounds_max_m": WALL_BOX["bounds_max_m"],
                                  "plane_normal_m": WALL_BOX["plane_normal_m"],
                                  "base_plane_offset_m": WALL_BOX["base_plane_offset_m"]}}
    registry = _registry(WALL_BOX["bounds_min_m"], WALL_BOX["bounds_max_m"],
                         WALL_BOX["plane_normal_m"], WALL_BOX["base_plane_offset_m"],
                         [0.05, 0.0, 0.0])
    result = plan_source_placement(
        registry, {"room_id": "room_fixture"}, _layout(normal, origin),
        _visual_geometry(origin, normal), request, config=_config(),
    )
    assert result["placement_checks"]["status"] == "not_run"
    assert result["support_identity"]["contact_point_m"][0] == pytest.approx(0.12)


def test_batch_rejects_a_surface_the_room_refuses_and_says_why():
    """A wall surface fitted a metre inside the room seats on nothing."""
    mesh = room_mesh()
    normal = [1.0, 0.0, 0.0]
    origin = [1.5, 1.5, 2.0]
    result = plan_static_source_placements(
        _registry(WALL_BOX["bounds_min_m"], WALL_BOX["bounds_max_m"],
                  WALL_BOX["plane_normal_m"], WALL_BOX["base_plane_offset_m"], [0.05, 0.0, 0.0]),
        {"room_id": "room_fixture"}, _layout(normal, origin), _visual_geometry(origin, normal),
        [{"instance_id": "source1", "asset_id": "fixture_device",
          "support_surface_id": "wall_01", "yaw_deg": 0.0,
          "asset_geometry": {"source_ref": "fixture://asset",
                             "bounds_min_m": WALL_BOX["bounds_min_m"],
                             "bounds_max_m": WALL_BOX["bounds_max_m"],
                             "plane_normal_m": WALL_BOX["plane_normal_m"],
                             "base_plane_offset_m": WALL_BOX["base_plane_offset_m"]}}],
        config=_config(), room_mesh=mesh, interior_points=walkable_points(),
    )
    assert result["status"] == "partial"
    row = result["instances"][0]
    assert row["status"] == "rejected"
    assert row["reason"]["code"] == "room_surface_checks_refused_every_candidate"
    assert row["reason"]["details"]["refused_count"] >= 1
    assert result["room_surface_checks"]["status"] == "measured"


def test_batch_moves_to_another_candidate_when_one_spot_is_refused():
    """One blocked spot does not condemn the surface; the search keeps looking."""
    mesh = room_mesh()
    normal = [1.0, 0.0, 0.0]
    origin = [0.0, 1.5, 2.0]
    result = plan_static_source_placements(
        _registry(WALL_BOX["bounds_min_m"], WALL_BOX["bounds_max_m"],
                  WALL_BOX["plane_normal_m"], WALL_BOX["base_plane_offset_m"], [0.05, 0.0, 0.0]),
        {"room_id": "room_fixture"}, _layout(normal, origin), _visual_geometry(origin, normal),
        [{"instance_id": "source1", "asset_id": "fixture_device",
          "support_surface_id": "wall_01", "yaw_deg": 0.0,
          "asset_geometry": {"source_ref": "fixture://asset",
                             "bounds_min_m": WALL_BOX["bounds_min_m"],
                             "bounds_max_m": WALL_BOX["bounds_max_m"],
                             "plane_normal_m": WALL_BOX["plane_normal_m"],
                             "base_plane_offset_m": WALL_BOX["base_plane_offset_m"]}}],
        config=_config(), room_mesh=mesh, interior_points=walkable_points(),
    )
    assert result["status"] == "planned"
    row = result["instances"][0]
    assert row["placement_checks"]["status"] == "pass"
    assert "rejected_for_room_surface_checks" in row["candidate"]


# --------------------------------------------------------------------------
# The pre-audio reading


def test_acoustic_gate_refuses_a_source_in_the_plenum_and_one_on_a_wall():
    mesh = room_mesh()
    report = check_acoustic_scene_points(mesh, [
        {"role": "source", "id": "plenum", "position_m": [2.0, 3.5, 2.0]},
        {"role": "source", "id": "on_wall", "position_m": [0.01, 1.5, 2.0]},
        {"role": "source", "id": "in_room", "position_m": [2.0, 1.2, 2.0]},
        {"role": "listener", "id": "listener", "position_m": [1.0, 1.55, 1.0]},
    ])
    assert report["status"] == "fail"
    assert report["interior_criterion"] == "applied"
    by_id = {row["id"]: row for row in report["rows"]}
    assert by_id["on_wall"]["failed_checks"] == ["clearance"]
    assert by_id["in_room"]["status"] == "pass"
    # The plenum is covered by the roof, so this room's vertical reading passes
    # it; the planner is where that placement is refused.
    assert by_id["plenum"]["interior"]["status"] == "pass"
    message = acoustic_pose_refusal(report)
    assert "on_wall" in message and "0.030 m" in message


def test_acoustic_gate_stands_down_where_the_room_has_no_ceiling():
    """A scan without a ceiling cannot answer the vertical question for anyone."""
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    _quad([0, 0, 0], [4, 0, 0], [4, 0, 4], [0, 0, 4], vertices, triangles)
    open_room = MeshHandle(np.asarray(vertices, float), np.asarray(triangles, np.int64))
    report = check_acoustic_scene_points(open_room, [
        {"role": "source", "id": "source1", "position_m": [2.0, 1.2, 2.0]},
        {"role": "listener", "id": "listener", "position_m": [1.0, 1.55, 1.0]},
    ])
    assert report["interior_criterion"] == "not_applicable"
    assert report["status"] == "pass"
    assert acoustic_pose_refusal(report) is None


def test_acoustic_gate_reports_not_run_without_a_mesh():
    report = check_acoustic_scene_points(None, [])
    assert report["status"] == "not_run"
    assert acoustic_pose_refusal(report) is None


# --------------------------------------------------------------------------
# The camera stage's reading of a static target

from avengine.rooms import conditioned_sampler as cs  # noqa: E402
from avengine.rooms.furniture_layout import clock_config  # noqa: E402
from avengine.capture.qa_plan_adapters import RasterWalkableSpace  # noqa: E402
from avengine.routes.raster_pathfinder import RasterPathfinder  # noqa: E402


def _camera_registry():
    return {"assets": [{
        "asset_id": f"device_{index}", "revision": "v1", "entity_class": "rigid_object",
        "identity": {"species_id": "device"}, "display_label": f"device {index}",
        "realized_attributes": {"body_color": "white"},
        "default_emitter_anchor_id": "speaker",
        "emitter_anchors": [{"anchor_id": "speaker", "offset_m": [0.0, 0.0, 0.0],
                             "offset_space": "final_scaled_asset_root"}],
    } for index in range(2)]}


def _camera_sounds():
    return [{"sound_asset_id": f"chime_{index}", "sound_class": "speech", "gender": "M",
             "transcript": f"utterance {index}", "sample_count": 32000, "sample_rate_hz": 16000,
             "audible_start_sample": 800, "audible_end_sample_exclusive": 31200,
             "active_duration_s": 1.9, "source_activity_intervals_samples": [[800, 31200]],
             "path": f"/prepared/{index}.wav"} for index in range(2)]


def _camera_space():
    finder = RasterPathfinder(np.ones((32, 32), dtype=bool),
                              bounds_m=[[0, -1, 0], [8, 1, 8]], floor_height_m=0.0)
    return RasterWalkableSpace(finder, {"floor_height_m": 0.0, "resolution_m": 0.25,
                                        "authority": "fixture_retained_grid"})


def _camera_request():
    return {"episode_id": "fixture", "seed": 111, "sampling_policy": cs.POLICY,
            "source_asset_ids": ["device_0", "device_1"],
            "profile": {"anchor_count": 1, "separation_bin_deg": [15, 180],
                        "speech_motion": "all_still", "event_relation": "sequential",
                        "reserve_tail_s": 1.0, "retry_budget_within_profile": 30,
                        # The request makes no visibility claim at all, which is
                        # exactly the shape that let a wall device be planned
                        # behind the camera and rendered fully occluded.
                        "anchor_visibility": "any", "competitor_visibility": "any",
                        "distance_range_m": [1.0, 8.0]}}


def _wall_placement(instance_id, centre, depth=0.1):
    low = [centre[0] - 0.05, centre[1] - 0.05, centre[2] - depth / 2]
    high = [centre[0] + 0.05, centre[1] + 0.05, centre[2] + depth / 2]
    return {
        "instance_id": instance_id, "status": "planned", "asset_id": "device_0",
        "support_identity": {"surface_id": "wall_01", "surface_kind": "wall"},
        "asset_bounds": {
            "world_aabb_min_m": low, "world_aabb_max_m": high,
            "world_corners_m": [[x, y, z] for x in (low[0], high[0])
                                for y in (low[1], high[1]) for z in (low[2], high[2])],
        },
    }


def _camera_scene():
    request = _camera_request()
    profile = cs.resolve_condition_profile(request, _camera_registry())
    clock = clock_config(frame_count=120, frame_rate_hz=15, sample_rate_hz=16000)
    frames = clock["frame_count"]
    actors = [cs.neutral_source_declaration(record, f"source{index + 1}")
              for index, record in enumerate(_camera_registry()["assets"])]
    points = np.repeat(np.array([[[0.9, 1.2, -3.0]], [[-0.9, 1.2, -3.0]]]), frames, axis=1)
    selected = {index: {**_camera_sounds()[index], "actor_id": actor["actor_id"]}
                for index, actor in enumerate(actors)}
    # Both sources are wall devices, which is what the D5 episodes are: no
    # actor stands on the floor, so the camera grid comes from the room's own
    # floor reference rather than from a walking actor's height.
    placements = {"instances": [
        _wall_placement(actor["entity_instance_id"], points[index, 0])
        for index, actor in enumerate(actors)]}
    return request, profile, clock, actors, points, selected, placements


def test_camera_refuses_a_static_target_it_cannot_see(monkeypatch):
    request, profile, clock, actors, points, selected, placements = _camera_scene()
    frames = clock["frame_count"]
    monkeypatch.setattr(cs, "camera_grid", lambda *args, **kwargs: [[0.0, 1.55, 0.0]])
    anchor_emitter = tuple(points[profile["anchor_indices"][0], 0])

    def trace(_mesh, _origin, target):
        return "blocked" if tuple(np.asarray(target, float)) == anchor_emitter else "clear"

    monkeypatch.setattr(cs, "line_of_sight", trace)
    with pytest.raises(cs.CandidateFailure) as refused:
        cs.select_camera_and_schedule(
            _camera_space(), object(), points, np.zeros((2, frames), dtype=bool),
            points, points, actors, selected, profile, clock, request,
            np.random.default_rng(0), static_placements=placements)
    assert "no_camera_frames_the_static_target_with_a_clear_line" in str(refused.value)


def test_camera_accepts_a_static_target_it_frames_with_a_clear_line(monkeypatch):
    request, profile, clock, actors, points, selected, placements = _camera_scene()
    frames = clock["frame_count"]
    monkeypatch.setattr(cs, "camera_grid", lambda *args, **kwargs: [[0.0, 1.55, 0.0]])
    monkeypatch.setattr(cs, "line_of_sight", lambda *args: "clear")
    camera, events, conditions = cs.select_camera_and_schedule(
        _camera_space(), object(), points, np.zeros((2, frames), dtype=bool),
        points, points, actors, selected, profile, clock, request,
        np.random.default_rng(0), static_placements=placements)
    assert camera["candidate_id"] in conditions["legal_candidate_ids"]
    assert len(events) == 2
    # The request still states no visibility claim; the static target is framed
    # because it is static, not because the request asked for it.
    assert conditions["anchor_visibility"] == "any"
    origin = np.asarray(camera["position_m"], dtype=float)
    forward = np.asarray(camera["basis"]["forward"], dtype=float)
    anchor_index = profile["anchor_indices"][0]
    assert float(np.dot(points[anchor_index, 0] - origin, forward)) > 0.1


def test_corners_framed_keeps_a_margin_and_needs_horizontal_distance():
    forwards = np.asarray([[0.0, 0.0, -1.0]])
    rights = np.asarray([[1.0, 0.0, 0.0]])
    tangent = math_tan_half(85.0)
    aspect = 1280 / 720
    centred = np.asarray([[0.0, 1.55, -3.0]] * 8, dtype=float)
    assert bool(cs._corners_framed(centred, [0.0, 1.55, 0.0], forwards, rights,
                                  tangent=tangent, aspect=aspect, margin=0.08)[0])
    # A ceiling device two metres above the lens needs the horizontal distance
    # the fixed camera cannot buy by pitching.
    high_near = np.asarray([[0.0, 3.55, -1.5]] * 8, dtype=float)
    high_far = np.asarray([[0.0, 3.55, -6.0]] * 8, dtype=float)
    assert not bool(cs._corners_framed(high_near, [0.0, 1.55, 0.0], forwards, rights,
                                       tangent=tangent, aspect=aspect, margin=0.08)[0])
    assert bool(cs._corners_framed(high_far, [0.0, 1.55, 0.0], forwards, rights,
                                  tangent=tangent, aspect=aspect, margin=0.08)[0])


def math_tan_half(fov_deg):
    import math
    return math.tan(math.radians(fov_deg) / 2)


def test_placement_world_corners_prefers_the_rotated_box():
    placement = _wall_placement("device", [1.0, 1.0, 1.0])
    corners = cs._placement_world_corners(placement)
    assert corners.shape == (8, 3)
    stripped = {"instance_id": "device", "status": "planned",
                "asset_bounds": {k: v for k, v in placement["asset_bounds"].items()
                                 if k != "world_corners_m"}}
    assert cs._placement_world_corners(stripped).shape == (8, 3)


def test_static_target_edge_margin_is_declared_and_validated():
    assert cs._static_target_edge_margin({}) == pytest.approx(
        cs.STATIC_TARGET_EDGE_MARGIN_FRACTION)
    assert cs._static_target_edge_margin(
        {"camera": {"static_target_edge_margin_fraction": 0.2}}) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        cs._static_target_edge_margin({"camera": {"static_target_edge_margin_fraction": 1.5}})


# --------------------------------------------------------------------------
# Attaching a registered wall or ceiling source without being told where


def _attached_registry(asset_id, surface, *, emitter_offset, bounds_min, bounds_max,
                       plane_normal, base_plane_offset):
    basis_u, basis_v = _orthogonal_basis(plane_normal)
    return {"assets": [{
        "asset_id": asset_id, "revision": "v1", "entity_class": "rigid_object",
        "identity": {"species_id": "device"}, "display_label": asset_id,
        "realized_attributes": {"body_color": "white"},
        "default_emitter_anchor_id": "speaker",
        "emitter_anchors": [{"anchor_id": "speaker", "offset_m": list(emitter_offset),
                             "offset_space": "final_scaled_asset_root"}],
        "runtime_backends": {"habitat": {"resting_pose": {
            "attachment_surface": surface, "base_plane_offset_m": base_plane_offset,
            "plane_normal_m": list(plane_normal), "plane_basis_u_m": list(basis_u),
            "plane_basis_v_m": list(basis_v), "footprint_extent_m": [0.1, 0.1],
            "height_m": 0.1, "measured_plane": surface + "_base"}}},
    }]}


def _room_catalog(tmp_path, surfaces, measurements):
    """A support surface catalog shaped as the room tool writes one."""
    vertices = tmp_path / "vertices.npy"
    triangles = tmp_path / "triangles.npy"
    mesh = room_mesh()
    np.save(vertices, mesh.vertices)
    np.save(triangles, mesh.triangles)
    catalog = {
        "schema": "avengine_support_surface_catalog_v1",
        "room": {"room_id": "room_fixture",
                 "coordinate_frame": {"handedness": "right", "linear_unit": "meter",
                                      "up_axis": "+Y"}},
        "layout": {"support_surfaces": surfaces},
        "visual_geometry": {
            "authority": "visual_geometry", "geometry_id": "room_visual_fixture_v1",
            "source_ref": str(vertices), "vertices_path": str(vertices),
            "triangles_path": str(triangles),
        },
        "asset_visual_geometry_measurements": measurements,
        "placement_config": {
            "normal_tolerance_deg": 1.0, "plane_tolerance_m": 0.25,
            "min_inter_instance_gap_m": 0.0,
            "candidate_search": {"grid_step_m": 0.5, "max_candidates": 32,
                                 "edge_margin_m": 0.05},
        },
    }
    path = tmp_path / "support_surface_catalog.json"
    path.write_text(json.dumps(catalog))
    return path


def _room_wall_surface():
    """The x = 0 wall of the fixture room, its normal pointing into the room."""
    mesh = room_mesh()
    indices = [index for index, triangle in enumerate(mesh.triangles)
               if np.allclose(mesh.vertices[triangle][:, 0], 0.0)]
    basis_u, basis_v = (np.asarray(value) for value in _orthogonal_basis([1.0, 0.0, 0.0]))
    origin = np.asarray([0.0, 1.5, 2.0])
    points = mesh.vertices[mesh.triangles[indices]].reshape(-1, 3) - origin
    along_u = points @ basis_u
    along_v = points @ basis_v
    return {
        "surface_id": "room_fixture_mesh_wall_01", "surface_kind": "wall",
        "room_id": "room_fixture", "origin_m": [0.0, 1.5, 2.0],
        "normal_m": [1.0, 0.0, 0.0],
        "basis_u_m": [float(value) for value in basis_u],
        "basis_v_m": [float(value) for value in basis_v],
        # The bounds are read off the named triangles, which is what the
        # planner's own cross-check requires of a real catalog too.
        "bounds_u_m": [float(along_u.min()), float(along_u.max())],
        "bounds_v_m": [float(along_v.min()), float(along_v.max())],
        "geometry_ref": {"authority": "visual_geometry",
                         "geometry_id": "room_visual_fixture_v1",
                         "triangle_indices": indices},
    }


def _wall_measurement():
    return {
        "support_kind": "wall", "source_ref": "fixture://asset",
        "bounds_min_m": WALL_BOX["bounds_min_m"], "bounds_max_m": WALL_BOX["bounds_max_m"],
        "plane_normal_m": WALL_BOX["plane_normal_m"],
        "base_plane_offset_m": WALL_BOX["base_plane_offset_m"],
    }


def test_registered_attachment_surface_reads_the_registry():
    registry = _attached_registry(
        "wall_device", "wall", emitter_offset=[0.05, 0.0, 0.0],
        bounds_min=WALL_BOX["bounds_min_m"], bounds_max=WALL_BOX["bounds_max_m"],
        plane_normal=WALL_BOX["plane_normal_m"],
        base_plane_offset=WALL_BOX["base_plane_offset_m"])
    assert cs.registered_attachment_surface(registry, "wall_device") == "wall"
    assert cs.registered_attachment_surface(registry, "not_registered") is None


def test_room_package_declares_where_its_support_surfaces_are(tmp_path):
    declared = str(tmp_path / "support_surface_catalog.json")
    room = {"room_package": {"planning_inputs": {"support_surface_catalog": declared}}}
    assert cs.room_support_surface_catalog_path(room) == declared
    assert cs.room_support_surface_catalog_path({"room_package": {}}) is None
    # A binding that was never expanded is not a path and must not be read as one.
    unexpanded = {"room_package": {"planning_inputs": {
        "support_surface_catalog": "${AVENGINE_SUPPORT_SURFACE_ROOT}/x.json"}}}
    assert cs.room_support_surface_catalog_path(unexpanded) is None


def test_a_request_that_names_no_surface_still_hangs_its_wall_device(tmp_path):
    """The request knows the asset; the room knows its walls; nobody knows both."""
    registry = _attached_registry(
        "wall_device", "wall", emitter_offset=[0.05, 0.0, 0.0],
        bounds_min=WALL_BOX["bounds_min_m"], bounds_max=WALL_BOX["bounds_max_m"],
        plane_normal=WALL_BOX["plane_normal_m"],
        base_plane_offset=WALL_BOX["base_plane_offset_m"])
    catalog = _room_catalog(tmp_path, [_room_wall_surface()],
                            {"wall_device": _wall_measurement()})
    room = {"room_id": "room_fixture",
            "room_package": {"planning_inputs": {"support_surface_catalog": str(catalog)}}}
    actors = [{"entity_instance_id": "device_1", "asset_id": "wall_device",
               "entity_class": "rigid_object"}]
    plan = cs.attached_static_placement_plan(
        actors, registry, room, space=None, mesh=room_mesh(),
        interior_points=walkable_points(), cache={})
    assert plan["status"] == "planned"
    row = plan["instances"][0]
    assert row["support_identity"]["surface_kind"] == "wall"
    # Seated on the wall the room actually has, growing into the room.
    assert row["root_transform"]["translation_m"][0] == pytest.approx(0.0, abs=0.02)
    assert row["emitter_transform"]["position_m"][0] > 0.0


def test_a_request_with_no_attached_source_is_left_alone(tmp_path):
    registry = _attached_registry(
        "floor_device", "floor", emitter_offset=[0.0, 0.1, 0.0],
        bounds_min=[-0.1, 0.0, -0.1], bounds_max=[0.1, 0.2, 0.1],
        plane_normal=[0.0, 1.0, 0.0], base_plane_offset=0.0)
    actors = [{"entity_instance_id": "device_1", "asset_id": "floor_device",
               "entity_class": "rigid_object"}]
    assert cs.attached_static_placement_plan(
        actors, registry, {"room_id": "room_fixture"}, space=None,
        mesh=room_mesh(), cache={}) is None


def test_a_room_without_support_surfaces_refuses_instead_of_using_the_floor(tmp_path):
    registry = _attached_registry(
        "wall_device", "wall", emitter_offset=[0.05, 0.0, 0.0],
        bounds_min=WALL_BOX["bounds_min_m"], bounds_max=WALL_BOX["bounds_max_m"],
        plane_normal=WALL_BOX["plane_normal_m"],
        base_plane_offset=WALL_BOX["base_plane_offset_m"])
    actors = [{"entity_instance_id": "device_1", "asset_id": "wall_device",
               "entity_class": "rigid_object"}]
    with pytest.raises(cs.CandidateFailure) as refused:
        cs.attached_static_placement_plan(
            actors, registry, {"room_id": "room_fixture"}, space=None,
            mesh=room_mesh(), cache={})
    assert "declares_no_support_surfaces" in str(refused.value)


def test_an_emitter_buried_in_its_own_mounting_face_is_refused_by_name(tmp_path):
    """A device whose emitter sits inside the face it mounts by cannot be placed.

    Standing it far enough off the wall to clear the acoustic minimum would
    mean it is not mounted on that wall at all, so the asset is refused and the
    reason names the asset rather than the room.
    """
    registry = _attached_registry(
        "buried_device", "wall", emitter_offset=[0.002, 0.0, 0.0],
        bounds_min=WALL_BOX["bounds_min_m"], bounds_max=WALL_BOX["bounds_max_m"],
        plane_normal=WALL_BOX["plane_normal_m"],
        base_plane_offset=WALL_BOX["base_plane_offset_m"])
    catalog = _room_catalog(tmp_path, [_room_wall_surface()],
                            {"buried_device": _wall_measurement()})
    room = {"room_id": "room_fixture",
            "room_package": {"planning_inputs": {"support_surface_catalog": str(catalog)}}}
    actors = [{"entity_instance_id": "device_1", "asset_id": "buried_device",
               "entity_class": "rigid_object"}]
    with pytest.raises(cs.CandidateFailure) as refused:
        cs.attached_static_placement_plan(
            actors, registry, room, space=None, mesh=room_mesh(),
            interior_points=walkable_points(), cache={})
    assert "no_support_surface_in_this_room_accepts" in str(refused.value)


def test_the_acoustic_minimum_decides_how_far_a_device_is_seated_off_its_surface():
    """A shallow emitter buys the millimetres it needs, and only those."""
    mesh = room_mesh()
    normal = [1.0, 0.0, 0.0]
    origin = [0.0, 1.5, 2.0]
    registry = _registry(WALL_BOX["bounds_min_m"], WALL_BOX["bounds_max_m"],
                         WALL_BOX["plane_normal_m"], WALL_BOX["base_plane_offset_m"],
                         [0.02, 0.0, 0.0])
    request = {"instance_id": "source1", "asset_id": "fixture_device",
               "support_surface_id": "wall_01", "candidate_index": 4, "yaw_deg": 0.0,
               "asset_geometry": {"source_ref": "fixture://asset",
                                  "bounds_min_m": WALL_BOX["bounds_min_m"],
                                  "bounds_max_m": WALL_BOX["bounds_max_m"],
                                  "plane_normal_m": WALL_BOX["plane_normal_m"],
                                  "base_plane_offset_m": WALL_BOX["base_plane_offset_m"]}}
    result = plan_source_placement(
        registry, {"room_id": "room_fixture"}, _layout(normal, origin),
        _visual_geometry(origin, normal), request, config=_config(),
        room_mesh=mesh, interior_points=walkable_points())
    checks = result["placement_checks"]
    assert checks["status"] == "pass", checks["failed_checks"]
    # The emitter is 2 cm in front of the contact plane, so the asset stands
    # 1 cm off the wall and the emitter keeps its 3 cm.
    assert checks["emitter_depth_in_front_of_contact_m"] == pytest.approx(0.02)
    assert checks["seated_standoff_m"] == pytest.approx(0.01)
    assert result["emitter_transform"]["position_m"][0] == pytest.approx(0.03, abs=1.0e-6)

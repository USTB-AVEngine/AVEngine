"""Unit tests for the qualification geometry measurement layer.

Every test here runs on synthetic geometry so the suite stays independent of
the retained captures and the asset library.  The real-data reproductions live
in the C05 attempt directory, not in this file.
"""
from __future__ import annotations

import json
import math

from pathlib import Path

import numpy as np
import pytest

from avengine.assets import qualification_geometry as qg
from avengine.dataset.source_asset_qualification import FORBIDDEN_EVIDENCE_KEYS


def _box_mesh(
    extent: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """An axis-aligned box as triangles, resting with its base at origin y."""
    x, y, z = extent
    ox, oy, oz = origin
    vertices = np.array(
        [
            [ox, oy, oz], [ox + x, oy, oz], [ox + x, oy, oz + z], [ox, oy, oz + z],
            [ox, oy + y, oz], [ox + x, oy + y, oz], [ox + x, oy + y, oz + z], [ox, oy + y, oz + z],
        ],
        dtype=float,
    )
    triangles = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # base, wound so the outward normal points down
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, triangles


# --------------------------------------------------------------------------
# boundary planes and contact classification
# --------------------------------------------------------------------------

def test_a_box_presents_six_boundary_faces_with_measured_area_shares():
    vertices, triangles = _box_mesh((2.0, 1.0, 3.0))
    candidates = qg.boundary_plane_candidates(vertices, triangles)
    assert len(candidates) == 6
    assert {(row["axis"], row["side"]) for row in candidates} == {
        (axis, side) for axis in range(3) for side in ("min", "max")
    }
    total = sum(row["area_share"] for row in candidates)
    assert total == pytest.approx(1.0, abs=1.0e-9)
    base = next(row for row in candidates if row["axis"] == 1 and row["side"] == "min")
    assert base["is_horizontal"] is True
    assert base["residual_q95_m"] == pytest.approx(0.0, abs=1.0e-12)


def test_the_contact_class_of_a_box_is_measured_not_assumed():
    vertices, triangles = _box_mesh()
    classes = qg.classify_contact_plane(qg.boundary_plane_candidates(vertices, triangles))
    assert qg.CONTACT_PLANE_HORIZONTAL_BASE in classes["measured_contact_plane_classes"]
    assert qg.CONTACT_PLANE_HORIZONTAL_TOP in classes["measured_contact_plane_classes"]
    assert "does not establish which room surface class" in classes["measurement_boundary"]


def test_a_mesh_tilted_off_its_own_axes_reports_no_horizontal_base():
    """The boundary-face method needs a mesh authored on its own axes.

    A box rolled 45 degrees has no face lying flat against any axis-aligned
    boundary, so no horizontal base candidate is produced at all.  The absence
    is the honest answer -- inventing a base plane for such a mesh would put a
    fabricated contact plane into the catalog.
    """
    vertices, triangles = _box_mesh()
    angle = math.radians(45.0)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ]
    )
    candidates = qg.boundary_plane_candidates(vertices @ rotation.T, triangles)
    assert not [row for row in candidates if row["axis"] == 1 and row["side"] == "min"]
    classes = qg.classify_contact_plane(candidates)
    assert qg.CONTACT_PLANE_HORIZONTAL_BASE not in classes["measured_contact_plane_classes"]
    assert classes["horizontal_base_area_share"] is None


# --------------------------------------------------------------------------
# support kind: an assumption is never promoted to an observation
# --------------------------------------------------------------------------

def _record(**pose) -> dict:
    return {
        "asset_id": "asset_under_test",
        "entity_class": "rigid_object",
        "runtime_backends": {"habitat": {"resting_pose": pose}},
    }


def test_an_assumed_attachment_surface_yields_no_support_kind():
    resolved = qg.resolve_support_kind(
        _record(attachment_surface="floor", attachment_surface_assumed=True),
        [qg.CONTACT_PLANE_HORIZONTAL_BASE],
    )
    assert resolved["support_kind"] is None
    assert resolved["support_kind_basis"] == "registry_assumed_not_declared"
    assert "assumed, not declared" in resolved["support_kind_absent_reason"]


def test_a_declared_attachment_surface_is_used_and_checked_against_the_mesh():
    resolved = qg.resolve_support_kind(
        _record(attachment_surface="wall", attachment_surface_assumed=False),
        [qg.CONTACT_PLANE_VERTICAL_MOUNT],
    )
    assert resolved["support_kind"] == "wall"
    assert resolved["support_kind_basis"] == "registry_declared"
    assert resolved["contact_class_consistent_with_registry"] is True


def test_a_declared_surface_inconsistent_with_the_mesh_is_flagged_not_hidden():
    resolved = qg.resolve_support_kind(
        _record(attachment_surface="wall", attachment_surface_assumed=False),
        [qg.CONTACT_PLANE_HORIZONTAL_BASE],
    )
    assert resolved["support_kind"] == "wall"
    assert resolved["contact_class_consistent_with_registry"] is False


def test_a_prior_catalog_kind_is_labelled_declared_intent_not_a_measurement():
    """A kind carried from an earlier catalog is that caller's intent.

    The mesh identifies a contact plane orientation; it does not identify a room
    surface class, so the row must not present the carried value as something the
    mesh established.
    """
    resolved = qg.resolve_support_kind(
        _record(attachment_surface="floor", attachment_surface_assumed=True),
        [qg.CONTACT_PLANE_HORIZONTAL_BASE],
        prior_support_kind="tabletop",
        prior_source_ref="t06_catalog.json",
    )
    assert resolved["support_kind"] == "tabletop"
    assert resolved["support_kind_basis"] == "prior_catalog_declared_intent"
    assert resolved["support_kind_basis_ref"] == "t06_catalog.json"
    assert "does not identify this kind on its own" in resolved["support_kind_basis_note"]


# --------------------------------------------------------------------------
# stage frame
# --------------------------------------------------------------------------

def test_a_z_up_stage_is_rotated_into_the_habitat_world_frame():
    rotation = qg.stage_world_from_asset(up=(0, 0, 1), front=(0, 1, 0))
    assert np.allclose(rotation @ np.array([0.0, 0.0, 1.0]), [0.0, 1.0, 0.0])
    assert np.allclose(rotation @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, -1.0])
    assert np.allclose(rotation @ np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0])
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0)


def test_an_already_y_up_stage_needs_no_rotation():
    rotation = qg.stage_world_from_asset(up=(0, 1, 0), front=(0, 0, -1))
    assert np.allclose(rotation, np.eye(3))


def test_non_orthogonal_declared_axes_are_refused():
    with pytest.raises(qg.QualificationGeometryError):
        qg.stage_world_from_asset(up=(0, 0, 1), front=(0, 1, 1))


def test_stage_axes_come_from_the_declared_dataset_config():
    axes = qg.stage_axes_from_dataset_config(
        {"stages": {"default_attributes": {"up": [0, 0, 1], "front": [0, 1, 0]}}}
    )
    assert axes["up"] == [0.0, 0.0, 1.0]
    with pytest.raises(qg.QualificationGeometryError):
        qg.stage_axes_from_dataset_config({"stages": {"default_attributes": {}}})


# --------------------------------------------------------------------------
# room collision
# --------------------------------------------------------------------------

def test_an_empty_room_clears_the_box_and_reports_the_distance():
    vertices, triangles = _box_mesh((1.0, 1.0, 1.0), origin=(10.0, 10.0, 10.0))
    observation = qg.measure_room_collision(
        world_aabb_min=[0.0, 0.0, 0.0],
        world_aabb_max=[1.0, 1.0, 1.0],
        room_vertices=vertices,
        room_triangles=triangles,
    )
    assert observation["status"] == "pass"
    assert observation["penetrating_triangle_count"] == 0
    assert observation["nearest_room_geometry"]["minimum_distance_lower_bound_m"] > 8.0


def test_resting_contact_on_a_support_plane_is_not_a_collision():
    floor = np.array(
        [[-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [5.0, 0.0, 5.0], [-5.0, 0.0, 5.0]], dtype=float
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    observation = qg.measure_room_collision(
        world_aabb_min=[0.0, 0.0, 0.0],
        world_aabb_max=[0.4, 0.4, 0.4],
        room_vertices=floor,
        room_triangles=triangles,
        support_plane_y_m=0.0,
    )
    assert observation["status"] == "pass"
    assert observation["penetrating_triangle_count"] == 0
    assert observation["supporting_contact_triangle_count"] == 2
    assert observation["non_supporting_contact_triangle_count"] == 0
    assert observation["nearest_room_geometry"]["minimum_distance_lower_bound_m"] == 0.0


def test_geometry_reaching_inside_the_box_is_unresolved_rather_than_failed():
    slab = np.array(
        [[-5.0, 0.2, -5.0], [5.0, 0.2, -5.0], [5.0, 0.2, 5.0], [-5.0, 0.2, 5.0]], dtype=float
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    observation = qg.measure_room_collision(
        world_aabb_min=[0.0, 0.0, 0.0],
        world_aabb_max=[1.0, 1.0, 1.0],
        room_vertices=slab,
        room_triangles=triangles,
        support_plane_y_m=0.0,
        contact_tolerance_m=0.05,
    )
    assert observation["penetrating_triangle_count"] == 2
    # An axis-aligned box encloses the asset mesh, so an overlap is an upper
    # bound and must not be reported as a failed clearance.
    assert observation["status"] == "not_run"
    assert "does not establish" in observation["reason"]
    assert observation["maximum_penetration_extent_m"] > 0.0


def test_a_small_object_keeps_an_interior_to_test():
    floor = np.array(
        [[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], dtype=float
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    observation = qg.measure_room_collision(
        world_aabb_min=[0.0, 0.0, 0.0],
        world_aabb_max=[0.05, 0.02, 0.05],
        room_vertices=floor,
        room_triangles=triangles,
        support_plane_y_m=0.0,
        contact_tolerance_m=0.05,
    )
    assert observation["measurement"] == "measured"
    assert observation["contact_tolerance_m"] < observation["requested_contact_tolerance_m"]
    assert observation["status"] == "pass"


def test_a_degenerate_box_is_refused():
    vertices, triangles = _box_mesh()
    with pytest.raises(qg.QualificationGeometryError):
        qg.measure_room_collision(
            world_aabb_min=[1.0, 1.0, 1.0],
            world_aabb_max=[0.0, 0.0, 0.0],
            room_vertices=vertices,
            room_triangles=triangles,
        )


def test_annotating_a_plan_keeps_every_planner_field():
    floor = np.array(
        [[-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [5.0, 0.0, 5.0], [-5.0, 0.0, 5.0]], dtype=float
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    plan = {
        "episode_id": "ep",
        "instances": [
            {
                "instance_id": "source2",
                "asset_id": "asset",
                "status": "planned",
                "support_identity": {"surface_id": "s1", "surface_kind": "floor"},
                "root_transform": {"translation_m": [0.0, 0.0, 0.0]},
                "asset_bounds": {
                    "world_aabb_min_m": [0.0, 0.0, 0.0],
                    "world_aabb_max_m": [0.4, 0.4, 0.4],
                },
                "clearance": {"inter_instance_aabb": {"status": "pass"}},
            },
            {"instance_id": "source3", "asset_id": "other", "status": "rejected"},
        ],
    }
    annotated = qg.annotate_placement_plan_room_collision(
        plan, room_vertices=floor, room_triangles=triangles
    )
    instance = annotated["instances"][0]
    assert instance["root_transform"] == {"translation_m": [0.0, 0.0, 0.0]}
    assert instance["support_identity"]["surface_id"] == "s1"
    assert instance["clearance"]["inter_instance_aabb"] == {"status": "pass"}
    assert instance["clearance"]["room_collision"]["status"] == "pass"
    assert annotated["instances"][1] == {
        "instance_id": "source3", "asset_id": "other", "status": "rejected"
    }
    assert annotated["c05_room_collision_annotation"]["annotated_instance_count"] == 1
    assert plan["instances"][0].get("clearance", {}).get("room_collision") is None


def test_a_planned_row_without_world_bounds_says_what_is_missing():
    floor = np.array(
        [[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]], dtype=float
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    annotated = qg.annotate_placement_plan_room_collision(
        {"instances": [{"instance_id": "a", "asset_id": "b", "status": "planned"}]},
        room_vertices=floor,
        room_triangles=triangles,
    )
    room = annotated["instances"][0]["clearance"]["room_collision"]
    assert room["measurement"] == "not_run"
    assert "asset_bounds.world_aabb_min_m" in room["missing_inputs"]


# --------------------------------------------------------------------------
# support surfaces from depth and semantics
# --------------------------------------------------------------------------

def _synthetic_camera() -> dict:
    return {
        "frame_index": 0,
        "resolution_hw": [64, 64],
        "fx_px": 40.0,
        "fy_px": 40.0,
        "cx_px": 31.5,
        "cy_px": 31.5,
        "projection": "pinhole",
        "camera_position_m": [0.0, 0.0, 0.0],
        "camera_basis_rows": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "camera_basis_source": "synthetic",
        "camera_pose_source": "synthetic",
    }


def _floor_depth(camera: dict, floor_y: float) -> np.ndarray:
    height, width = camera["resolution_hw"]
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    vertical = (camera["cy_px"] - rows) / camera["fy_px"]
    depth = np.zeros((height, width), dtype=float)
    below = vertical < -1.0e-3
    z = np.where(below, floor_y / np.where(below, vertical, 1.0), 0.0)
    depth[:] = np.broadcast_to(z, (height, width))
    depth[~np.broadcast_to(below, (height, width))] = 0.0
    return depth


def test_a_flat_floor_is_recovered_from_depth_and_semantics():
    camera = _synthetic_camera()
    depth = _floor_depth(camera, floor_y=-1.5)
    semantic = np.where(depth > 0.0, 7, 0).astype(np.uint32)
    surface = qg.fit_semantic_support_surface(
        depth,
        semantic,
        np.zeros(depth.shape, dtype=bool),
        camera,
        semantic_ids=[7],
        surface_id="synthetic_floor",
        surface_kind="floor",
        orientation="horizontal",
        minimum_pixels=100,
    )
    assert surface["surface_kind"] == "floor"
    assert surface["origin_m"][1] == pytest.approx(-1.5, abs=1.0e-6)
    assert abs(surface["normal_m"][1]) == pytest.approx(1.0, abs=1.0e-6)
    assert surface["evidence"]["normal_tilt_from_up_deg"] == pytest.approx(0.0, abs=1.0e-3)
    # the bounds are the observed extent, never an extrapolation across the room
    assert surface["bounds_u_m"][1] - surface["bounds_u_m"][0] < 60.0
    assert "lower bound on it" in surface["measurement_boundary"]


def test_an_absent_semantic_id_is_not_run_rather_than_an_empty_surface():
    camera = _synthetic_camera()
    depth = _floor_depth(camera, floor_y=-1.5)
    semantic = np.zeros(depth.shape, dtype=np.uint32)
    surface = qg.fit_semantic_support_surface(
        depth, semantic, np.zeros(depth.shape, dtype=bool), camera,
        semantic_ids=[7], surface_id="missing", surface_kind="floor",
        orientation="horizontal",
    )
    assert surface["measurement"] == "not_run"
    assert "semantic pixels" in surface["missing_inputs"][0]


def test_a_height_band_with_too_few_pixels_is_not_run():
    camera = _synthetic_camera()
    depth = _floor_depth(camera, floor_y=-1.5)
    semantic = np.where(depth > 0.0, 7, 0).astype(np.uint32)
    surface = qg.fit_semantic_support_surface(
        depth, semantic, np.zeros(depth.shape, dtype=bool), camera,
        semantic_ids=[7], surface_id="banded", surface_kind="floor",
        orientation="horizontal", height_band_m=(5.0, 6.0), minimum_pixels=100,
    )
    assert surface["measurement"] == "not_run"
    assert surface["banded_pixel_count"] == 0


def test_a_horizontal_plane_is_refused_where_a_vertical_one_was_requested():
    camera = _synthetic_camera()
    depth = _floor_depth(camera, floor_y=-1.5)
    semantic = np.where(depth > 0.0, 7, 0).astype(np.uint32)
    surface = qg.fit_semantic_support_surface(
        depth, semantic, np.zeros(depth.shape, dtype=bool), camera,
        semantic_ids=[7], surface_id="not_a_wall", surface_kind="wall",
        orientation="vertical", minimum_pixels=100,
    )
    assert surface["measurement"] == "not_run"
    assert "not vertical" in surface["reason"]


def test_excluded_pixels_are_left_out_of_the_fit():
    camera = _synthetic_camera()
    depth = _floor_depth(camera, floor_y=-1.5)
    semantic = np.where(depth > 0.0, 7, 0).astype(np.uint32)
    exclude = np.zeros(depth.shape, dtype=bool)
    exclude[40:, :] = True
    full = qg.fit_semantic_support_surface(
        depth, semantic, np.zeros(depth.shape, dtype=bool), camera,
        semantic_ids=[7], surface_id="full", surface_kind="floor",
        orientation="horizontal", minimum_pixels=10,
    )
    masked = qg.fit_semantic_support_surface(
        depth, semantic, exclude, camera,
        semantic_ids=[7], surface_id="masked", surface_kind="floor",
        orientation="horizontal", minimum_pixels=10,
    )
    assert masked["evidence"]["fit_pixel_count"] < full["evidence"]["fit_pixel_count"]


# --------------------------------------------------------------------------
# footprint fit against a measured patch
# --------------------------------------------------------------------------

def test_a_footprint_larger_than_the_observed_patch_is_inconclusive_not_refuted():
    observation = qg.surface_fit_observation(
        {"footprint_extent_m": [0.5, 0.4]},
        {"surface_id": "patch", "surface_kind": "tabletop",
         "bounds_u_m": [-0.1, 0.1], "bounds_v_m": [-0.1, 0.1]},
    )
    assert observation["footprint_fits_measured_patch"] is False
    assert "does not refute" in observation["interpretation"]


def test_a_footprint_inside_the_observed_patch_fits():
    observation = qg.surface_fit_observation(
        {"footprint_extent_m": [0.1, 0.1]},
        {"surface_id": "patch", "surface_kind": "tabletop",
         "bounds_u_m": [-1.0, 1.0], "bounds_v_m": [-1.0, 1.0]},
    )
    assert observation["footprint_fits_measured_patch"] is True


# --------------------------------------------------------------------------
# articulated support
# --------------------------------------------------------------------------

def _audit(frames: int = 3) -> dict:
    records = []
    for index in range(frames):
        offset = 0.01 * index
        records.append(
            {
                "frame_index": index,
                "mesh_min_y_m": 0.05 + offset,
                "contact_joint_positions_m": {
                    "paw_front_left": [0.20 + offset, 0.09, 0.08],
                    "paw_front_right": [-0.25, 0.10, -0.10],
                    "paw_hind_left": [-0.30, 0.09, 0.03],
                    "paw_hind_right": [0.18, 0.08, -0.09],
                },
            }
        )
    return {
        "schema": "avengine_p12_habitat_mesh_grounding_audit_v1",
        "inputs": {"visual_glb": {"path": "/somewhere/visual.glb"}},
        "actions": {"idle": {"frame_records": records, "mesh_max_y_m": 0.68}},
    }


def test_the_articulated_footprint_is_the_area_the_feet_sweep():
    support = qg.summarise_articulated_support(_audit(frames=3))
    assert support["measurement"] == "measured"
    assert support["contact_anchor_sample_count"] == 12
    # x spans -0.30 to 0.22 once the walk offset is included, z spans -0.10 to 0.08
    assert support["contact_footprint_extent_m"][0] == pytest.approx(0.52, abs=1.0e-6)
    assert support["contact_footprint_extent_m"][1] == pytest.approx(0.18, abs=1.0e-6)
    assert support["support_plane_y_actor_m"] == pytest.approx(0.05)
    assert "not soles" in support["measurement_boundary"]


def test_an_audit_without_contact_positions_is_not_run():
    audit = _audit()
    for record in audit["actions"]["idle"]["frame_records"]:
        record.pop("contact_joint_positions_m")
    support = qg.summarise_articulated_support(audit)
    assert support["measurement"] == "not_run"
    assert "contact_joint_positions_m" in support["missing_inputs"][0]


def test_an_audit_with_no_actions_is_not_run():
    support = qg.summarise_articulated_support({"actions": {}})
    assert support["measurement"] == "not_run"


# --------------------------------------------------------------------------
# catalog assembly and the verdict boundary
# --------------------------------------------------------------------------

def _registry() -> dict:
    return {
        "registry_id": "test_registry",
        "revision": "v1",
        "assets": [
            {
                "asset_id": "rigid_declared_wall",
                "entity_class": "rigid_object",
                "runtime_backends": {
                    "habitat": {
                        "glb_path": "/does/not/exist.glb",
                        "resting_pose": {
                            "attachment_surface": "wall",
                            "attachment_surface_assumed": False,
                        },
                    }
                },
            }
        ],
    }


def test_an_unreadable_asset_is_listed_as_unmeasured_not_silently_dropped():
    catalog = qg.build_geometry_measurement_catalog(_registry())
    assert catalog["measured_asset_count"] == 0
    assert catalog["requested_asset_count"] == 1
    assert catalog["unmeasured"][0]["asset_id"] == "rigid_declared_wall"
    assert "does not exist" in catalog["unmeasured"][0]["reason"]


def test_an_asset_id_outside_the_registry_is_reported():
    catalog = qg.build_geometry_measurement_catalog(
        _registry(), asset_ids=["not_registered"]
    )
    assert catalog["unmeasured"][0]["reason"] == "asset id is not in the registry"


def test_the_catalog_carries_no_verdict_key_anywhere(monkeypatch):
    def fake(record, **kwargs):
        return {
            "asset_id": record["asset_id"],
            "bounds_min_m": [0.0, 0.0, 0.0],
            "bounds_max_m": [1.0, 1.0, 1.0],
            "support_kind": "wall",
        }

    monkeypatch.setattr(qg, "measure_rigid_contact_geometry", fake)
    catalog = qg.build_geometry_measurement_catalog(_registry())
    assert catalog["measured_asset_count"] == 1

    def walk(value, path="catalog"):
        if isinstance(value, dict):
            for key, child in value.items():
                assert key not in FORBIDDEN_EVIDENCE_KEYS, f"{path} carries verdict key {key}"
                walk(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(json.loads(json.dumps(catalog)))
    assert catalog["qualification_claim"] is False


def test_a_missing_asset_file_raises_rather_than_returning_empty_geometry():
    with pytest.raises(qg.QualificationGeometryError):
        qg.load_asset_triangles("/does/not/exist.glb")


# --------------------------------------------------------------------------
# native pose decoding
# --------------------------------------------------------------------------

def _binding(order, *, scrambled=False):
    """A runtime binding whose offsets do not follow the authored order."""
    offsets = list(range(0, 4 * len(order), 4))
    if scrambled:
        offsets = offsets[::-1]
    return {
        "joint_position_count": 4 * len(order),
        "links": [
            {"link_name": name, "joint_position_offset": offset, "joint_position_count": 4}
            for name, offset in zip(order, offsets)
        ],
    }


def test_a_native_pose_is_decoded_by_declared_offset_not_by_position():
    order = ["a", "b", "c"]
    flat = np.zeros(12)
    # identity quaternion for every joint, with a marker on the w of each
    for index in range(3):
        flat[4 * index + 3] = 1.0
    flat[0] = 0.0
    binding = _binding(order, scrambled=True)
    pose, facts = qg.decode_native_joint_pose(flat, binding, order)
    assert pose.shape == (3, 4)
    assert facts["joint_count"] == 3
    # joint "a" must read the LAST block, because that is the offset declared
    assert np.allclose(pose[0], [0.0, 0.0, 0.0, 1.0])
    assert facts["decoded_by"].startswith("habitat_runtime_binding")


def test_a_joint_without_a_declared_offset_is_refused():
    binding = _binding(["a", "b"])
    with pytest.raises(qg.QualificationGeometryError):
        qg.decode_native_joint_pose(np.zeros(8), binding, ["a", "b", "c"])


def test_a_frame_of_the_wrong_length_is_refused():
    order = ["a", "b"]
    with pytest.raises(qg.QualificationGeometryError):
        qg.decode_native_joint_pose(np.zeros(12), _binding(order), order)


def test_readback_drift_is_renormalised_but_gross_error_is_refused():
    order = ["a"]
    binding = _binding(order)
    drifted = np.array([0.0, 0.0, 0.0, 1.0 + 3.0e-8])
    pose, facts = qg.decode_native_joint_pose(drifted, binding, order)
    assert float(np.linalg.norm(pose[0])) == pytest.approx(1.0, abs=1e-12)
    assert facts["max_quaternion_unit_drift"] < qg.MAX_NATIVE_QUATERNION_DRIFT
    with pytest.raises(qg.QualificationGeometryError):
        qg.decode_native_joint_pose(
            np.array([0.0, 0.0, 0.0, 1.2]), binding, order
        )


# --------------------------------------------------------------------------
# exact narrowphase
# --------------------------------------------------------------------------

def _tri(*points):
    return np.array([list(point) for point in points], dtype=float)[None, :, :]


def test_two_crossing_triangles_are_detected_with_a_point():
    a = _tri((0, 0, 0), (1, 0, 0), (0, 1, 0))
    b = _tri((0.2, 0.2, -1), (0.2, 0.2, 1), (0.6, 0.3, 1))
    hits, points = qg._triangle_pairs_intersect(a, b)
    assert bool(hits[0]) is True
    assert points[0][2] == pytest.approx(0.0, abs=1e-9)


def test_triangles_sharing_a_bounding_box_but_not_crossing_are_not_hits():
    # both live in the same box; one sits entirely above the other's plane
    a = _tri((0, 0, 0), (1, 0, 0), (0, 1, 0))
    b = _tri((0, 0, 0.5), (1, 0, 0.5), (0, 1, 0.5))
    hits, _ = qg._triangle_pairs_intersect(a, b)
    assert bool(hits[0]) is False


def test_an_empty_bounding_box_corner_is_not_reported_as_a_collision(tmp_path):
    """Broadphase may pair them; the exact test must still say no."""
    a = _tri((0, 0, 0), (0.1, 0, 0), (0, 0.1, 0))
    b = _tri((0.9, 0.9, 0.9), (1.0, 0.9, 0.9), (0.9, 1.0, 0.9))
    hits, _ = qg._triangle_pairs_intersect(a, b)
    assert bool(hits[0]) is False


def test_the_resting_contact_allowance_applies_along_the_normal_only(monkeypatch):
    """A coplanar support face is contact; a vertical face at the same distance is not."""
    asset_vertices = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.2, 0.0], [0.0, 0.2, 0.0],
         [0.0, 0.0, 0.2], [0.2, 0.0, 0.2], [0.2, 0.2, 0.2], [0.0, 0.2, 0.2]],
        dtype=float,
    )
    asset_triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                                [0, 1, 5], [0, 5, 4]], dtype=np.int64)

    class FakeScene:
        vertices = asset_vertices
        triangles = asset_triangles
        source_sha256 = "fake"
        source_byte_size = 0
        source_node_instance_count = 1
        source_primitive_count = 1

    monkeypatch.setattr(qg, "extract_triangle_scene", lambda path: FakeScene())
    monkeypatch.setattr(Path, "is_file", lambda self: True)

    # a horizontal room face slicing through the asset at y = 0.1
    room_vertices = np.array(
        [[-1.0, 0.1, -1.0], [1.0, 0.1, -1.0], [1.0, 0.1, 1.0], [-1.0, 0.1, 1.0]], dtype=float
    )
    room_triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    coplanar = qg.measure_asset_mesh_room_intersection(
        asset_glb="fake.glb",
        root_transform_matrix_row_major=identity,
        room_vertices=room_vertices,
        room_triangles=room_triangles,
        support_plane={
            "surface_id": "s", "origin_m": [0.0, 0.1, 0.0], "normal_m": [0.0, 1.0, 0.0],
            "plane_residual_q95_m": 0.01, "source_ref": "catalog",
        },
        asset_id="asset",
    )
    assert coplanar["intersecting_pair_count"] > 0
    assert coplanar["penetrating_pair_count"] == 0
    assert coplanar["status"] == "pass"
    assert coplanar["support_contact_exclusion"]["applied"] is True
    assert "along the surface normal only" in (
        coplanar["support_contact_exclusion"]["allowance_basis"]
    )

    # the same distance, but the declared support plane is vertical: the crossing
    # face is no longer aligned with it and must not be excused
    tilted = qg.measure_asset_mesh_room_intersection(
        asset_glb="fake.glb",
        root_transform_matrix_row_major=identity,
        room_vertices=room_vertices,
        room_triangles=room_triangles,
        support_plane={
            "surface_id": "s", "origin_m": [0.0, 0.1, 0.0], "normal_m": [1.0, 0.0, 0.0],
            "plane_residual_q95_m": 0.01, "source_ref": "catalog",
        },
        asset_id="asset",
    )
    assert tilted["penetrating_pair_count"] > 0
    assert tilted["status"] == "fail"


def test_a_support_plane_without_a_measured_residual_excuses_nothing(monkeypatch):
    asset_vertices = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.2, 0.0], [0.0, 0.2, 0.0],
         [0.0, 0.0, 0.2], [0.2, 0.0, 0.2], [0.2, 0.2, 0.2], [0.0, 0.2, 0.2]],
        dtype=float,
    )
    asset_triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    class FakeScene:
        vertices = asset_vertices
        triangles = asset_triangles
        source_sha256 = "fake"
        source_byte_size = 0
        source_node_instance_count = 1
        source_primitive_count = 1

    monkeypatch.setattr(qg, "extract_triangle_scene", lambda path: FakeScene())
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    room_vertices = np.array(
        [[-1.0, 0.1, -1.0], [1.0, 0.1, -1.0], [1.0, 0.1, 1.0], [-1.0, 0.1, 1.0]], dtype=float
    )
    room_triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    result = qg.measure_asset_mesh_room_intersection(
        asset_glb="fake.glb",
        root_transform_matrix_row_major=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        room_vertices=room_vertices,
        room_triangles=room_triangles,
        support_plane={"surface_id": "s", "origin_m": [0.0, 0.1, 0.0], "normal_m": [0.0, 1.0, 0.0]},
        asset_id="asset",
    )
    assert result["support_contact_exclusion"]["applied"] is False
    assert "plane_residual_q95_m" in result["support_contact_exclusion"]["missing"]


def test_a_root_transform_that_is_not_sixteen_numbers_is_refused():
    with pytest.raises(qg.QualificationGeometryError):
        qg.measure_asset_mesh_room_intersection(
            asset_glb="fake.glb",
            root_transform_matrix_row_major=[1, 0, 0],
            room_vertices=np.zeros((3, 3)),
            room_triangles=np.array([[0, 1, 2]]),
        )


# --------------------------------------------------------------------------
# package layouts
# --------------------------------------------------------------------------

def test_both_package_layouts_resolve_their_inputs(tmp_path):
    p12 = tmp_path / "p12"
    (p12 / "package").mkdir(parents=True)
    (p12 / "package" / "visual.glb").write_bytes(b"")
    (p12 / "package" / "emitter_anchors.json").write_text("{}")
    (p12 / "joint_mapping.json").write_text("{}")
    (p12 / "actions.npz").write_bytes(b"")
    (p12 / "contacts.json").write_text("{}")
    resolved = qg.discover_articulated_package(p12 / "package" / "visual.glb")
    assert resolved["package_layout"] == "p12_package"
    assert resolved["joint_mapping"] == str(p12 / "joint_mapping.json")
    assert resolved["actions_npz"] == str(p12 / "actions.npz")

    m2 = tmp_path / "m2"
    (m2 / "habitat").mkdir(parents=True)
    (m2 / "actions").mkdir()
    (m2 / "contacts").mkdir()
    (m2 / "visual.glb").write_bytes(b"")
    (m2 / "emitter_anchors.json").write_text("{}")
    (m2 / "habitat" / "joint_mapping.json").write_text("{}")
    (m2 / "actions" / "idle.npz").write_bytes(b"")
    (m2 / "contacts" / "contact_phases.json").write_text("{}")
    resolved = qg.discover_articulated_package(m2 / "visual.glb")
    assert resolved["package_layout"] == "m2_dataset"
    assert resolved["joint_mapping"] == str(m2 / "habitat" / "joint_mapping.json")
    assert resolved["actions_npz"] == str(m2 / "actions" / "idle.npz")
    assert resolved["runtime_binding"] is None


def test_a_missing_runtime_binding_names_itself(tmp_path):
    package = tmp_path / "pkg"
    (package / "package").mkdir(parents=True)
    (package / "package" / "visual.glb").write_bytes(b"")
    (package / "joint_mapping.json").write_text("{}")
    (package / "contacts.json").write_text("{}")
    result = qg.measure_native_foot_contact(
        package_root=package / "package" / "visual.glb",
        capture_root=tmp_path / "nowhere",
        actor_index=0,
        actor_id="source1",
        asset_id="asset",
    )
    assert result["measurement"] == "not_run"
    assert any("habitat_runtime_binding" in item for item in result["missing_inputs"])


# --------------------------------------------------------------------------
# world identity
# --------------------------------------------------------------------------

def _plan(tmp_path, **extra):
    path = tmp_path / "episode_plan.json"
    payload = {"episode_id": "ep_v0", "scene": {"room_id": "room_a", "scene_id": "room_a"}}
    payload.update(extra)
    path.write_text(json.dumps(payload))
    return path


def test_a_stage_directory_name_is_an_alias_not_a_world_id(tmp_path):
    stage = tmp_path / "p09_some_stage_name"
    stage.mkdir()
    identity = qg.resolve_world_identity(plan_path=_plan(tmp_path), stage_root=stage)
    assert identity["world_id"] is None
    assert identity["aliases"]["stage_directory_name"] == "p09_some_stage_name"
    assert identity["episode_id"] == "ep_v0"
    assert identity["room_id"] == "room_a"
    assert "not promoted" in identity["world_id_absent_reason"]
    assert identity["join_keys"] == ["episode_id", "room_id"]


def test_a_declared_world_id_is_kept_with_its_source(tmp_path):
    identity = qg.resolve_world_identity(
        plan_path=_plan(tmp_path), declared_world_id="world_x"
    )
    assert identity["world_id"] == "world_x"
    assert identity["world_id_source"] == "caller declared"
    assert identity["join_keys"][0] == "world_id"


def test_a_plan_declared_world_id_wins_over_nothing(tmp_path):
    identity = qg.resolve_world_identity(plan_path=_plan(tmp_path, world_id="world_p"))
    assert identity["world_id"] == "world_p"
    assert identity["world_id_source"] == "episode plan"


# --------------------------------------------------------------------------
# room visual geometry resolution
# --------------------------------------------------------------------------

def test_a_room_with_no_stage_glb_refuses_to_substitute_another(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "scene": {"room_id": "ue_room"},
        "resources": {"backend": "ue_spear", "room_id": "ue_room"},
    }))
    result = qg.resolve_room_visual_geometry(path)
    assert result["measurement"] == "not_run"
    assert result["backend"] == "ue_spear"
    assert any("scene_glb" in item for item in result["missing_inputs"])
    assert "refused" in result["substitution_refused"]


def test_a_plan_without_a_dataset_config_is_not_run(tmp_path):
    glb = tmp_path / "scene.glb"
    glb.write_bytes(b"")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "scene": {"room_id": "r"},
        "resources": {"scene_glb": str(glb), "backend": "habitat"},
    }))
    result = qg.resolve_room_visual_geometry(path)
    assert result["measurement"] == "not_run"
    assert "dataset_config" in result["missing_inputs"][0]


# --------------------------------------------------------------------------
# support query
# --------------------------------------------------------------------------

def _slab(y_top, y_bottom, half=1.0, offset=(0.0, 0.0)):
    """A box-less pair of horizontal faces: a surface with thickness."""
    ox, oz = offset
    vertices = []
    triangles = []
    for height in (y_top, y_bottom):
        base = len(vertices)
        vertices.extend([
            [ox - half, height, oz - half], [ox + half, height, oz - half],
            [ox + half, height, oz + half], [ox - half, height, oz + half],
        ])
        triangles.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    return np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64)


def test_the_navigation_floor_is_required_and_cannot_be_inferred():
    vertices, triangles = _slab(1.0, 0.9)
    with pytest.raises(qg.QualificationGeometryError):
        qg.query_support_under_footprint(
            room_vertices=vertices, room_triangles=triangles,
            centre_world_m=[0.0, 1.5, 0.0], footprint_extent_m=[0.4, 0.4],
            navigation_floor_m=None, navigation_floor_source="",
        )


def test_a_full_surface_covers_the_footprint_and_reports_its_top():
    vertices, triangles = _slab(1.0, 0.9)
    result = qg.query_support_under_footprint(
        room_vertices=vertices, room_triangles=triangles,
        centre_world_m=[0.0, 1.5, 0.0], footprint_extent_m=[0.4, 0.4],
        navigation_floor_m=0.0, navigation_floor_source="test plan navigation",
    )
    assert result["measurement"] == "measured"
    level = result["supporting_level"]
    assert level is not None
    assert level["footprint_coverage_fraction"] == pytest.approx(1.0)
    # faces 0.1 m apart do not chain, so the top level is the 1.0 m face alone
    assert level["support_top_q95_m"] == pytest.approx(1.0, abs=1e-6)
    assert level["height_spread_m"] == pytest.approx(0.0, abs=1e-6)
    assert "navigation component" in result["storey_basis"]


def test_a_stepped_surface_chains_into_one_level_whose_middle_is_not_its_top():
    """Single-linkage chaining is why the median height must not be used.

    Two faces 8 mm apart fall inside the cluster width and merge into one level.
    An asset rests on the higher of them, so the correction reads a high quantile
    of the per-sample tops, not the level's median height.
    """
    low_v, low_t = _slab(1.000, 0.90, half=0.5, offset=(-0.25, 0.0))
    high_v, high_t = _slab(1.008, 0.90, half=0.5, offset=(0.25, 0.0))
    vertices = np.vstack([low_v, high_v])
    triangles = np.vstack([low_t, high_t + len(low_v)])
    result = qg.query_support_under_footprint(
        room_vertices=vertices, room_triangles=triangles,
        centre_world_m=[0.0, 1.5, 0.0], footprint_extent_m=[0.9, 0.5],
        navigation_floor_m=0.0, navigation_floor_source="test plan navigation",
        grid=11,
    )
    level = result["supporting_level"]
    assert level is not None
    assert level["height_spread_m"] > 0.005
    assert level["support_top_q95_m"] > level["height_median_m"]
    assert level["support_top_max_m"] == pytest.approx(1.008, abs=1e-6)


def test_a_small_object_on_a_surface_is_occupancy_not_a_support():
    floor_v, floor_t = _slab(1.0, 0.9, half=1.0)
    box_v, box_t = _slab(1.2, 1.1, half=0.05)
    vertices = np.vstack([floor_v, box_v])
    triangles = np.vstack([floor_t, box_t + len(floor_v)])
    result = qg.query_support_under_footprint(
        room_vertices=vertices, room_triangles=triangles,
        centre_world_m=[0.0, 2.0, 0.0], footprint_extent_m=[0.8, 0.8],
        navigation_floor_m=0.0, navigation_floor_source="test plan navigation",
        grid=11,
    )
    heights = [row["support_top_q95_m"] for row in result["levels"]]
    assert any(abs(h - 1.0) < 1e-6 for h in heights)
    assert any(abs(h - 1.2) < 1e-6 for h in heights)
    high = [row for row in result["levels"] if row["support_top_q95_m"] > 1.15][0]
    assert high["covers_footprint"] is False
    assert high["footprint_coverage_fraction"] < 0.2
    # the surface, not the object on it, is the support
    assert result["supporting_level"]["support_top_q95_m"] == pytest.approx(1.0, abs=1e-6)


def test_uncovered_samples_separate_mesh_holes_from_occupancy():
    floor_v, floor_t = _slab(1.0, 0.9, half=0.15)          # smaller than the footprint
    box_v, box_t = _slab(1.2, 1.1, half=0.05, offset=(0.3, 0.0))
    vertices = np.vstack([floor_v, box_v])
    triangles = np.vstack([floor_t, box_t + len(floor_v)])
    result = qg.query_support_under_footprint(
        room_vertices=vertices, room_triangles=triangles,
        centre_world_m=[0.0, 2.0, 0.0], footprint_extent_m=[0.9, 0.9],
        navigation_floor_m=0.0, navigation_floor_source="test plan navigation",
        grid=11,
    )
    base = [row for row in result["levels"] if row["support_top_q95_m"] < 1.05][0]
    assert base["uncovered_samples_with_no_room_geometry"] > 0
    assert base["uncovered_samples_occupied_above"] > 0
    assert "holes in the scene mesh" in base["uncovered_breakdown_note"]


def test_nothing_under_the_footprint_is_reported_rather_than_guessed():
    vertices, triangles = _slab(1.0, 0.9, half=0.05, offset=(5.0, 5.0))
    result = qg.query_support_under_footprint(
        room_vertices=vertices, room_triangles=triangles,
        centre_world_m=[0.0, 1.5, 0.0], footprint_extent_m=[0.4, 0.4],
        navigation_floor_m=0.0, navigation_floor_source="test plan navigation",
    )
    assert result["levels"] == []
    assert result["supporting_level"] is None
    assert "no near-horizontal room face" in result["reason"]


# --------------------------------------------------------------------------
# root correction
# --------------------------------------------------------------------------

def _support(top, coverage=1.0, occupied_above=0, levels=None):
    level = {
        "height_median_m": top - 0.01,
        "support_top_median_m": top,
        "support_top_q95_m": top,
        "support_top_max_m": top,
        "support_top_spread_m": 0.0,
        "normal_m": [0.0, 1.0, 0.0],
        "footprint_coverage_fraction": coverage,
        "covers_footprint": coverage >= 0.9,
        "uncovered_samples_occupied_above": occupied_above,
    }
    return {
        "measurement": "measured",
        "supporting_level": level if level["covers_footprint"] else None,
        "levels": levels if levels is not None else [level],
        "level_count": 1,
        "navigation_floor_source": "test plan navigation",
    }


def test_a_correction_lands_the_contact_on_the_measured_top():
    correction = qg.plan_root_contact_correction(
        support_query=_support(1.0),
        contact_offset={
            "measurement": "measured", "root_above_contact_m": 0.05,
            "asset_height_m": 0.3, "evidence_kind": "placed_rigid_mesh",
        },
        current_root_world_m=[0.0, 1.02, 0.0],
    )
    assert correction["prediction"] == "cpu_prediction"
    assert correction["native_verification"] == "not_run"
    assert correction["corrected_root_world_m"][1] == pytest.approx(1.05)
    assert correction["root_delta_m"] == pytest.approx(0.03)
    assert correction["current_gap_to_support_m"] == pytest.approx(-0.03)
    assert correction["unobstructed"] is True


def test_a_level_that_covers_only_part_of_the_footprint_is_not_corrected_onto():
    correction = qg.plan_root_contact_correction(
        support_query=_support(1.0, coverage=0.5),
        contact_offset={"measurement": "measured", "root_above_contact_m": 0.05},
        current_root_world_m=[0.0, 1.02, 0.0],
    )
    assert correction["measurement"] == "not_run"
    assert "occupancy" in correction["reason"]


def test_something_standing_in_the_way_blocks_the_correction():
    blocker = {
        "height_median_m": 1.1, "support_top_median_m": 1.12, "support_top_q95_m": 1.12,
        "footprint_coverage_fraction": 0.6, "covers_footprint": False,
        "uncovered_samples_occupied_above": 0,
    }
    base = {
        "height_median_m": 0.99, "support_top_median_m": 1.0, "support_top_q95_m": 1.0,
        "support_top_max_m": 1.0, "support_top_spread_m": 0.0, "normal_m": [0.0, 1.0, 0.0],
        "footprint_coverage_fraction": 1.0, "covers_footprint": True,
        "uncovered_samples_occupied_above": 0,
    }
    query = {
        "measurement": "measured", "supporting_level": base, "levels": [base, blocker],
        "level_count": 2, "navigation_floor_source": "test plan navigation",
    }
    correction = qg.plan_root_contact_correction(
        support_query=query,
        contact_offset={
            "measurement": "measured", "root_above_contact_m": 0.05, "asset_height_m": 0.3,
        },
        current_root_world_m=[0.0, 1.5, 0.0],
    )
    assert correction["headroom_obstructed"] is True
    assert correction["unobstructed"] is False
    assert correction["headroom_obstruction_max_coverage_fraction"] == pytest.approx(0.6)
    assert correction["headroom_obstructions"][0]["height_above_support_m"] == pytest.approx(0.12)


def test_an_unmeasured_support_or_offset_yields_not_run():
    assert qg.plan_root_contact_correction(
        support_query={"measurement": "not_run", "reason": "no geometry"},
        contact_offset={"measurement": "measured", "root_above_contact_m": 0.05},
        current_root_world_m=[0.0, 1.0, 0.0],
    )["measurement"] == "not_run"
    assert qg.plan_root_contact_correction(
        support_query=_support(1.0),
        contact_offset={"measurement": "not_run", "reason": "no mesh"},
        current_root_world_m=[0.0, 1.0, 0.0],
    )["measurement"] == "not_run"


def test_the_correction_never_claims_an_executed_contact():
    correction = qg.plan_root_contact_correction(
        support_query=_support(1.0),
        contact_offset={"measurement": "measured", "root_above_contact_m": 0.05},
        current_root_world_m=[0.0, 1.02, 0.0],
    )
    assert correction["native_verification"] == "not_run"
    assert "has not run" in correction["claim_boundary"]
    assert "regenerate" in correction["consumer_contract"]

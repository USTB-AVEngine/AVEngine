from __future__ import annotations

import math

import numpy as np
import pytest

from avengine.m6x.geometry import (
    ELEVATED_OBJECT,
    GROUND_BLOCKER,
    M6XGeometryError,
    UNKNOWN_OBSTACLE_ROLE,
    WALKABLE_FLOOR_COVERING,
    build_runtime_obstacle_map,
    evaluate_source_center_gate,
    extract_loaded_rigid_obstacles,
    point_to_world_obb_clearance,
)


class _Bounds:
    def __init__(
        self, minimum: tuple[float, float, float], maximum: tuple[float, float, float]
    ):
        self.min = np.asarray(minimum, dtype=np.float64)
        self.max = np.asarray(maximum, dtype=np.float64)


class _Transform:
    def __init__(self, rotation: np.ndarray, translation: tuple[float, float, float]):
        self.rotation = np.asarray(rotation, dtype=np.float64)
        self.translation = np.asarray(translation, dtype=np.float64)

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        return self.rotation @ np.asarray(point, dtype=np.float64) + self.translation


class _Object:
    def __init__(
        self,
        object_id: int,
        handle: str,
        *,
        rotation: np.ndarray | None = None,
        translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        semantic_id: int | None = None,
        collision_bounds: tuple[
            tuple[float, float, float], tuple[float, float, float]
        ] = ((-0.5, -0.25, -0.2), (0.5, 0.25, 0.2)),
    ):
        self.object_id = object_id
        self.handle = handle
        if semantic_id is not None:
            self.semantic_id = semantic_id
        self.collision_shape_aabb = _Bounds(*collision_bounds)
        self.transformation = _Transform(
            np.eye(3) if rotation is None else rotation,
            translation,
        )


class _Manager:
    def __init__(self, *objects: _Object):
        self.objects = {value.handle: value for value in objects}

    def get_objects_by_handle_substring(self):
        return self.objects


class _Mn:
    @staticmethod
    def Vector3(value):
        return np.asarray(value, dtype=np.float64)


class _PathFinder:
    is_loaded = True

    def __init__(self) -> None:
        self.changed = False
        self.is_navigable_call_count = 0

    def get_topdown_view(self, meters_per_pixel: float, height_m: float):
        assert meters_per_pixel == pytest.approx(0.1)
        assert height_m == pytest.approx(0.4)
        result = np.ones((20, 20), dtype=np.uint8)
        result[:, 9:11] = 0
        if self.changed:
            result[0, 0] = 0
        return result

    def get_bounds(self):
        return np.asarray(((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)), dtype=np.float64)

    def is_navigable(self, point, maximum_y_delta):
        self.is_navigable_call_count += 1
        x, _, z = np.asarray(point, dtype=np.float64)
        return 0.0 <= x <= 2.0 and 0.0 <= z <= 2.0 and not 0.9 <= x <= 1.1

    def snap_point(self, point):
        x, _, z = np.asarray(point, dtype=np.float64)
        if 0.9 <= x <= 1.1:
            x = 0.8
        return np.asarray((min(max(x, 0.0), 2.0), 0.4, min(max(z, 0.0), 2.0)))

    def distance_to_closest_obstacle(self, point, maximum_search_radius):
        x, _, z = np.asarray(point, dtype=np.float64)
        return min(x, 2.0 - x, z, 2.0 - z, abs(x - 0.9), abs(x - 1.1))


def _rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.asarray(
        (
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        )
    )


def test_extracts_every_loaded_rigid_obb_and_honors_explicit_exclusions() -> None:
    furniture = _Object(
        7,
        "table_07",
        rotation=_rotation_y(30.0),
        translation=(1.5, 0.5, 1.25),
    )
    marker = _Object(8, "source_marker_0_:0000")
    records = extract_loaded_rigid_obstacles(
        _Manager(marker, furniture),
        _Mn,
        excluded_handle_prefixes=("source_marker_",),
    )
    assert [record["object_id"] for record in records] == [7]
    assert records[0]["source"] == "live_habitat_rigid_collision_shape"
    assert len(records[0]["world_corners_m"]) == 8
    assert len(records[0]["footprint_xz_m"]) == 4
    axes = np.asarray(records[0]["world_obb"]["axes_xyz"])
    assert axes @ axes.T == pytest.approx(np.eye(3), abs=1.0e-8)


def test_point_to_world_obb_clearance_distinguishes_inside_and_outside() -> None:
    record = extract_loaded_rigid_obstacles(
        _Manager(_Object(1, "chair", translation=(1.0, 0.5, 1.0))), _Mn
    )[0]
    distance, inside = point_to_world_obb_clearance((1.0, 0.5, 1.0), record)
    assert distance == pytest.approx(0.0)
    assert inside is True
    distance, inside = point_to_world_obb_clearance((2.0, 0.5, 1.0), record)
    assert distance == pytest.approx(0.5)
    assert inside is False


def test_source_center_gate_uses_same_navmesh_and_loaded_object_snapshot() -> None:
    manager = _Manager(_Object(4, "cabinet", translation=(1.6, 0.5, 1.6)))
    pathfinder = _PathFinder()
    obstacle_map = build_runtime_obstacle_map(
        pathfinder,
        manager,
        _Mn,
        floor_height_m=0.4,
        meters_per_pixel=0.1,
    )
    report = evaluate_source_center_gate(
        pathfinder,
        obstacle_map,
        {
            "source_clear": [[0.5, 1.5, 0.5], [0.6, 1.5, 0.6]],
            "source_nav_hole": [[1.0, 1.5, 0.5]],
            "source_in_cabinet": [[1.6, 0.5, 1.6]],
        },
    )
    assert report["status"] == "fail"
    assert report["full_body_collision_claim"] is False
    assert report["pathfinder_snapshot_match"] is True
    assert report["sources"]["source_clear"]["status"] == "pass"
    assert report["sources"]["source_nav_hole"]["failed_frame_indices"] == [0]
    rigid_frame = report["sources"]["source_in_cabinet"]["frames"][0]
    assert rigid_frame["inside_loaded_rigid_obstacle"] is True
    assert rigid_frame["nearest_rigid_obstacle"]["handle"] == "cabinet"
    assert report["authority"]["rigid_obstacle_count"] == 1


def test_runtime_map_retains_all_113_replicacad_style_rigid_obbs() -> None:
    objects = tuple(
        _Object(
            object_id,
            f"replicacad_furniture_{object_id:03d}",
            translation=(0.25 + object_id * 0.001, 0.5, 1.5),
        )
        for object_id in range(113)
    )
    obstacle_map = build_runtime_obstacle_map(
        _PathFinder(),
        _Manager(*objects),
        _Mn,
        floor_height_m=0.4,
        meters_per_pixel=0.1,
    )
    assert len(obstacle_map.rigid_obstacles) == 113
    assert [record["object_id"] for record in obstacle_map.rigid_obstacles] == list(
        range(113)
    )
    assert obstacle_map.summary()["rigid_obstacle_count"] == 113


def test_roles_require_semantics_geometry_and_live_navmesh_confirmation() -> None:
    rug = _Object(
        1,
        "opaque_asset_name_a",
        translation=(0.5, 0.4, 0.5),
        semantic_id=98,
        collision_bounds=((-0.3, -0.04, -0.3), (0.3, 0.04, 0.3)),
    )
    chair = _Object(
        2,
        "opaque_asset_name_b",
        translation=(1.5, 0.4, 1.5),
        semantic_id=20,
    )
    picture = _Object(
        3,
        "opaque_asset_name_c",
        translation=(0.5, 1.5, 1.5),
        semantic_id=59,
        collision_bounds=((-0.2, -0.1, -0.05), (0.2, 0.1, 0.05)),
    )
    unresolved = _Object(4, "opaque_asset_name_d", translation=(0.5, 0.4, 1.0))
    pathfinder = _PathFinder()
    obstacle_map = build_runtime_obstacle_map(
        pathfinder,
        _Manager(rug, chair, picture, unresolved),
        _Mn,
        floor_height_m=0.4,
        meters_per_pixel=0.1,
        semantic_categories_by_id={20: "chair", 59: "picture", 98: "rug"},
    )
    by_id = {item["object_id"]: item for item in obstacle_map.rigid_obstacles}
    assert by_id[1]["obstacle_role"] == WALKABLE_FLOOR_COVERING
    assert by_id[1]["blocks_source_center"] is False
    assert by_id[1]["semantic_category"] == "rug"
    assert by_id[1]["role_evidence"]["navmesh_floor_confirmation"] is True
    assert by_id[2]["obstacle_role"] == GROUND_BLOCKER
    assert by_id[3]["obstacle_role"] == ELEVATED_OBJECT
    assert by_id[4]["obstacle_role"] == UNKNOWN_OBSTACLE_ROLE
    assert pathfinder.is_navigable_call_count == 5
    assert by_id[2]["role_evidence"]["navmesh_floor_probe_count"] == 0
    assert by_id[2]["role_evidence"]["navmesh_floor_confirmation"] is None
    assert obstacle_map.summary()["rigid_obstacle_role_counts"] == {
        ELEVATED_OBJECT: 1,
        GROUND_BLOCKER: 1,
        UNKNOWN_OBSTACLE_ROLE: 1,
        WALKABLE_FLOOR_COVERING: 1,
    }

    # A source center may occupy the confirmed thin floor covering, while an
    # unresolved OBB remains conservative and blocks the same exact 3-D test.
    report = evaluate_source_center_gate(
        pathfinder,
        obstacle_map,
        {
            "on_rug": [[0.5, 0.4, 0.5]],
            "inside_unknown": [[0.5, 0.4, 1.0]],
            "near_elevated": [[0.5, 1.7, 1.5]],
            "inside_elevated": [[0.5, 1.5, 1.5]],
        },
    )
    assert report["sources"]["on_rug"]["status"] == "pass"
    rug_frame = report["sources"]["on_rug"]["frames"][0]
    assert rug_frame["inside_loaded_rigid_obstacle"] is True
    assert rug_frame["inside_blocking_loaded_rigid_obstacle"] is False
    assert rug_frame["nearest_rigid_obstacle"]["obstacle_role"] == (
        WALKABLE_FLOOR_COVERING
    )
    assert rug_frame["nearest_blocking_rigid_obstacle"] is not None
    assert rug_frame["rigid_obstacle_clearance_m"] == pytest.approx(0.0)
    assert rug_frame["blocking_rigid_obstacle_clearance_m"] > 0.0
    assert report["sources"]["inside_unknown"]["status"] == "fail"
    assert report["sources"]["near_elevated"]["status"] == "pass"
    assert report["sources"]["inside_elevated"]["status"] == "fail"
    assert (
        report["sources"]["inside_elevated"]["frames"][0][
            "nearest_rigid_obstacle"
        ]["obstacle_role"]
        == ELEVATED_OBJECT
    )


def test_floor_covering_semantics_alone_cannot_bypass_collision_gate() -> None:
    thick_or_unsupported_rug = _Object(
        1,
        "not_used_for_classification",
        translation=(1.0, 0.7, 0.5),
        semantic_id=98,
        collision_bounds=((-0.3, -0.3, -0.3), (0.3, 0.3, 0.3)),
    )
    obstacle_map = build_runtime_obstacle_map(
        _PathFinder(),
        _Manager(thick_or_unsupported_rug),
        _Mn,
        floor_height_m=0.4,
        meters_per_pixel=0.1,
        semantic_categories_by_id={98: "rug"},
    )
    record = obstacle_map.rigid_obstacles[0]
    assert record["obstacle_role"] == UNKNOWN_OBSTACLE_ROLE
    assert record["blocks_source_center"] is True


def test_gate_rejects_a_different_or_reloaded_pathfinder() -> None:
    pathfinder = _PathFinder()
    obstacle_map = build_runtime_obstacle_map(
        pathfinder,
        _Manager(),
        _Mn,
        floor_height_m=0.4,
        meters_per_pixel=0.1,
    )
    trajectory = {"source0": [[0.5, 1.5, 0.5]]}

    with pytest.raises(M6XGeometryError, match="differs"):
        evaluate_source_center_gate(_PathFinder(), obstacle_map, trajectory)

    pathfinder.changed = True
    with pytest.raises(M6XGeometryError, match="navmesh changed"):
        evaluate_source_center_gate(pathfinder, obstacle_map, trajectory)


def test_runtime_map_accepts_a_negative_operating_floor_height() -> None:
    class _NegativeFloorPathFinder(_PathFinder):
        def get_topdown_view(self, meters_per_pixel: float, height_m: float):
            assert meters_per_pixel == pytest.approx(0.1)
            assert height_m == pytest.approx(-0.4)
            return np.ones((20, 20), dtype=np.uint8)

    obstacle_map = build_runtime_obstacle_map(
        _NegativeFloorPathFinder(),
        _Manager(),
        _Mn,
        floor_height_m=-0.4,
        meters_per_pixel=0.1,
    )
    assert obstacle_map.floor_height_m == pytest.approx(-0.4)


def test_rejects_unloaded_or_empty_runtime_geometry() -> None:
    pathfinder = _PathFinder()
    pathfinder.is_loaded = False
    with pytest.raises(M6XGeometryError, match="no loaded navmesh"):
        build_runtime_obstacle_map(
            pathfinder,
            _Manager(),
            _Mn,
            floor_height_m=0.4,
            meters_per_pixel=0.1,
        )

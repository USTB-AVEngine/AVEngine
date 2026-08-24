from __future__ import annotations

from collections import OrderedDict
import json
import math

import numpy as np
import pytest

import avengine.m6x.apartment_viewpoints as viewpoint_module
from avengine.m6x.apartment_viewpoints import (
    ApartmentRasterPrefilterError,
    StraightCorridor,
    camera_ue_yaw_degrees,
    circular_yaw_distance_degrees,
    evaluate_raster_pose_prefilter,
    generate_straight_corridors,
    normalize_corridors,
    normalize_semantic_targets,
    prefilter_apartment_viewpoints,
)
from avengine.m6x.geometry import RuntimeObstacleMap, point_to_world_obb_clearance
from avengine.m6x.room_feasibility import RoomFeasibilityCompiler


def _obstacle_map(
    *,
    shape_hw: tuple[int, int] = (32, 32),
    extent_xz_m: tuple[float, float] = (8.0, 8.0),
    navmesh: np.ndarray | None = None,
    rigid_obstacles: tuple[dict[str, object], ...] = (),
) -> RuntimeObstacleMap:
    binary = (
        np.ones(shape_hw, dtype=np.uint8) if navmesh is None else np.asarray(navmesh)
    )
    return RuntimeObstacleMap(
        binary_navmesh=binary,
        bounds_m=(
            (0.0, 0.0, 0.0),
            (extent_xz_m[0], 3.0, extent_xz_m[1]),
        ),
        floor_height_m=0.0,
        meters_per_pixel=min(
            extent_xz_m[0] / shape_hw[1], extent_xz_m[1] / shape_hw[0]
        ),
        rigid_obstacles=rigid_obstacles,
    )


def _quaternion_for_yaw(yaw_deg: float) -> list[float]:
    half = math.radians(yaw_deg) * 0.5
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def _base_request(
    *, yaw_deg: float = 0.0, position_m: tuple[float, float, float] = (1.0, 1.5, 1.0)
) -> dict[str, object]:
    return {
        "primary_camera_rig": {
            "view_id": "view0",
            "world_from_rig": {
                "translation_m": list(position_m),
                "rotation_xyzw": _quaternion_for_yaw(yaw_deg),
            },
            "shared_calibration": {
                "projection": "pinhole",
                "resolution_hw": [720, 1280],
                "hfov_degrees": 105.0,
                "near_m": 0.05,
                "far_m": 50.0,
            },
        }
    }


def _targets() -> OrderedDict[str, object]:
    return OrderedDict(
        (
            (
                "living_region",
                {
                    "region_xz_m": [[2.5, 2.5], [4.0, 4.0]],
                    "role": "living_reverse",
                },
            ),
            (
                "kitchen_wall",
                {
                    "surface_position_m": [1.0, 1.0, 7.9],
                    "focus_position_m": [1.0, 0.0, 7.625],
                    "role": "kitchen_reverse",
                },
            ),
            (
                "window_boundary",
                {
                    "surface_position_m": [8.2, 1.4, 4.0],
                    "focus_position_m": [7.625, 0.0, 4.0],
                    "role": "window_facing",
                },
            ),
            (
                "living_duplicate",
                {
                    "region_xz_m": [[2.5, 2.5], [4.0, 4.0]],
                    "role": "living_reverse",
                },
            ),
        )
    )


def _retained_corridors() -> list[dict[str, object]]:
    return [
        {
            "corridor_id": "z_duplicate",
            "start": [1.0, 1.0],
            "end": [4.2, 1.0],
        },
        {
            "corridor_id": "a_retained",
            "start": [4.2, 1.0],
            "end": [1.0, 1.0],
        },
        {"start": [2.0, 5.0], "end": [5.2, 5.0]},
    ]


def _prefilter(**overrides):
    arguments = {
        "base_m1_camera": _base_request(),
        "semantic_targets": _targets(),
        "selection_count": 6,
        "candidate_spacing_m": 0.75,
        "raster_sample_spacing_m": 0.5,
        "maximum_positions_per_target": 6,
        "minimum_reverse_yaw_difference_deg": 10.0,
        "minimum_reverse_pose_difference_proxy": 0.1,
        "minimum_pairwise_position_m": 0.5,
        "minimum_pairwise_yaw_degrees": 5.0,
        "minimum_pairwise_raster_signature_jaccard": 0.005,
    }
    arguments.update(overrides)
    return prefilter_apartment_viewpoints(_obstacle_map(), **arguments)


def test_prefilter_is_deterministic_order_free_and_explicitly_non_native() -> None:
    targets = _targets()
    corridors = _retained_corridors()
    first = _prefilter(
        corridor_segments=corridors,
        corridor_coordinate_space="habitat_xz_m",
    )
    second = _prefilter(
        semantic_targets=OrderedDict(reversed(tuple(targets.items()))),
        corridor_segments=list(reversed(corridors)),
        corridor_coordinate_space="habitat_xz_m",
    )

    assert first.to_record() == second.to_record()
    assert first.raw_candidate_count > first.exact_deduplicated_candidate_count
    assert first.viewpoints[0].viewpoint_id == "apartment_view0"
    assert all(item.view_id == "view0" for item in first.viewpoints)
    serialized = json.dumps(first.to_record(), sort_keys=True).lower()
    assert "deterministic_2d_raster_prefilter_only" in serialized
    assert "certified" not in serialized
    assert "visible" not in serialized
    for viewpoint in first.viewpoints:
        assert viewpoint.native_pixel_validation_status == "pending"
        if viewpoint.height_role in ("low", "high"):
            assert "vertical_fov" in viewpoint.native_validation_pending
            assert "first_person_composition" in viewpoint.native_validation_pending


def test_semantic_surface_region_and_boundary_focus_derive_navigable_proxies() -> None:
    obstacle_map = _obstacle_map()
    targets = normalize_semantic_targets(
        obstacle_map,
        {
            "region": {"region_xz_m": [[2.0, 2.0], [3.5, 3.5]]},
            "surface_only": {"surface_position_m": [8.2, 1.0, 6.0]},
            "boundary_focus": {
                "surface_position_m": [8.2, 1.0, 4.0],
                "focus_position_m": [7.625, 0.0, 4.0],
            },
        },
    )

    by_id = {item.target_id: item for item in targets}
    assert by_id["region"].focus_derivation.startswith("nearest_region")
    assert by_id["surface_only"].focus_derivation == ("surface_nearest_navmesh_sample")
    for target in targets:
        x, _y, z = target.focus_position_m
        assert 0.0 <= x < 8.0
        assert 0.0 <= z < 8.0
    assert by_id["boundary_focus"].surface_position_m[0] > 8.0
    assert by_id["boundary_focus"].focus_position_m[0] < 8.0


def test_wall_adjacent_kitchen_reverse_uses_pose_proxy_not_half_plane() -> None:
    result = _prefilter(
        required_roles=("kitchen_reverse",),
        selection_count=4,
    )
    kitchen_views = [
        item for item in result.viewpoints if "kitchen_reverse" in item.roles
    ]

    assert kitchen_views
    assert all(item.pose_difference_proxy_from_legacy >= 0.1 for item in kitchen_views)
    assert any(item.camera_position_m[2] < 7.625 for item in kitchen_views)


def test_actual_shape_coverage_is_bounded_vectorized_and_los_queries_are_sparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (282, 239)
    obstacle_map = _obstacle_map(
        shape_hw=shape,
        extent_xz_m=(9.56, 11.28),
    )
    targets = normalize_semantic_targets(
        obstacle_map,
        {
            "front": {"focus_position_m": [2.0, 0.0, 1.0], "position_m": [2, 1]},
            "right": {"focus_position_m": [3.0, 0.0, 2.0], "position_m": [3, 2]},
        },
    )
    corridor = StraightCorridor(
        corridor_id="c0",
        start_xz_m=(2.0, 1.0),
        end_xz_m=(3.5, 1.0),
        length_m=1.5,
        origin="synthetic",
        direction_ids=("c0:forward", "c0:reverse"),
    )
    calls = 0
    original = viewpoint_module._supercover_line

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(viewpoint_module, "_supercover_line", counted)
    result = evaluate_raster_pose_prefilter(
        obstacle_map,
        camera_position_m=(2.0, 1.5, 3.0),
        camera_yaw_deg=0.0,
        hfov_degrees=105.0,
        near_m=0.05,
        far_m=50.0,
        semantic_targets=targets,
        corridors=(corridor,),
        raster_sample_spacing_m=0.5,
        minimum_lateral_c=0.0,
    )

    assert result.total_raster_sample_count < 1_000
    assert shape[0] * shape[1] > 100 * result.total_raster_sample_count
    assert result.grid_los_query_count == 5
    assert calls == 5


def test_corridor_endpoint_midpoint_and_lateral_proxy_fields() -> None:
    right = StraightCorridor(
        corridor_id="right",
        start_xz_m=(2.8, 2.0),
        end_xz_m=(3.4, 0.5),
        length_m=math.dist((2.8, 2.0), (3.4, 0.5)),
        origin="synthetic",
        direction_ids=("right:forward", "right:reverse"),
    )
    center = StraightCorridor(
        corridor_id="center",
        start_xz_m=(2.0, 2.0),
        end_xz_m=(2.0, 0.5),
        length_m=1.5,
        origin="synthetic",
        direction_ids=("center:forward", "center:reverse"),
    )
    obstacle_map = _obstacle_map(shape_hw=(20, 20), extent_xz_m=(4.0, 4.0))
    targets = normalize_semantic_targets(
        obstacle_map, {"front": {"focus_position_m": [2.0, 1.0]}}
    )
    result = evaluate_raster_pose_prefilter(
        obstacle_map,
        camera_position_m=(2.0, 1.5, 3.5),
        camera_yaw_deg=0.0,
        hfov_degrees=105.0,
        near_m=0.01,
        far_m=10.0,
        semantic_targets=targets,
        corridors=(center, right),
        minimum_lateral_c=0.2,
    )
    checks = {item.corridor_id: item for item in result.corridor_checks}

    assert checks["right"].point_prefilter_pass == (True, True, True)
    assert checks["right"].prefilter_pass is True
    assert min(checks["right"].lateral_c) > 0.2
    assert checks["center"].point_prefilter_pass == (True, True, True)
    assert checks["center"].minimum_abs_lateral_c == pytest.approx(0.0)
    assert checks["center"].prefilter_pass is False


def test_exact_corridor_supercover_rejects_a_thin_crossed_blocker() -> None:
    navmesh = np.ones((3, 4), dtype=np.uint8)
    navmesh[0, 1] = 0
    obstacle_map = _obstacle_map(
        shape_hw=(3, 4),
        extent_xz_m=(4.0, 3.0),
        navmesh=navmesh,
    )
    grid = viewpoint_module._validated_grid(obstacle_map)
    start = (0.5, 0.5)
    end = (3.5, 2.5)
    length = math.dist(start, end)

    traversed = viewpoint_module._exact_corridor_line_pixels(grid, start, end)
    assert (0, 1) in traversed
    corridors = generate_straight_corridors(
        obstacle_map,
        corridor_source_center_heights_m=(1.0,),
        minimum_length_m=length - 1.0e-9,
        maximum_length_m=length + 1.0e-9,
        minimum_clearance_m=0.0,
        endpoint_spacing_m=1.0,
        maximum_endpoint_count=12,
        maximum_corridor_count=64,
    )
    forbidden_geometry = tuple(sorted((start, end)))
    assert all(
        tuple(sorted((item.start_xz_m, item.end_xz_m))) != forbidden_geometry
        for item in corridors
    )


def test_corridor_generation_requires_and_intersects_all_source_heights() -> None:
    obstacle = {
        "object_id": 7,
        "handle": "dog_height_blocker",
        "obstacle_role": "ground_blocker",
        "blocks_source_center": True,
        "footprint_xz_m": [
            [3.0, 3.0],
            [5.0, 3.0],
            [5.0, 5.0],
            [3.0, 5.0],
        ],
        "world_obb": {
            "center_m": [4.0, 0.7, 4.0],
            "axes_xyz": np.eye(3).tolist(),
            "half_extents_m": [1.0, 0.3, 1.0],
        },
    }
    obstacle_map = _obstacle_map(rigid_obstacles=(obstacle,))

    with pytest.raises(TypeError):
        generate_straight_corridors(obstacle_map)  # type: ignore[call-arg]
    corridors = generate_straight_corridors(
        obstacle_map,
        corridor_source_center_heights_m=(0.7, 1.8),
        minimum_clearance_m=0.2,
        endpoint_spacing_m=0.75,
        maximum_corridor_count=32,
    )
    compiler = RoomFeasibilityCompiler(obstacle_map)
    dog = compiler.compile(
        source_center_height_m=0.7,
        minimum_navmesh_clearance_m=0.2,
        minimum_rigid_clearance_m=0.2,
        sample_spacing_m=0.75,
    ).feasible_mask
    human = compiler.compile(
        source_center_height_m=1.8,
        minimum_navmesh_clearance_m=0.2,
        minimum_rigid_clearance_m=0.2,
        sample_spacing_m=0.75,
    ).feasible_mask
    intersection = dog & human

    assert np.count_nonzero(intersection) < np.count_nonzero(human)
    assert corridors
    for corridor in corridors:
        assert corridor.source_center_heights_m == (0.7, 1.8)
        assert 3.0 <= corridor.length_m <= 3.8
        assert all(intersection[pixel] for pixel in corridor.sample_pixels_rc)
        assert corridor.direction_ids == (
            f"{corridor.corridor_id}:forward",
            f"{corridor.corridor_id}:reverse",
        )


def test_target_role_requires_its_own_focus_prefilter_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obstacle_map = _obstacle_map()
    semantic = {
        "split_focus": {
            "surface_position_m": [4.0, 0.0, 2.0],
            "focus_position_m": [2.0, 0.0, 4.0],
            "role": "split_focus",
        }
    }
    targets = normalize_semantic_targets(
        obstacle_map, semantic, maximum_focus_snap_distance_m=4.0
    )
    forced_position = (2.0, 2.0)
    forced_yaw = -90.0
    pose = evaluate_raster_pose_prefilter(
        obstacle_map,
        camera_position_m=(2.0, 1.5, 2.0),
        camera_yaw_deg=forced_yaw,
        hfov_degrees=105.0,
        near_m=0.05,
        far_m=50.0,
        semantic_targets=targets,
    )
    assert "split_focus" not in pose.passed_semantic_focus_ids

    def forced_placement(*_args, **_kwargs):
        return ((forced_position, forced_yaw, 0.5),)

    monkeypatch.setattr(viewpoint_module, "_target_camera_positions", forced_placement)
    with pytest.raises(
        ApartmentRasterPrefilterError,
        match="fewer than four exact raster candidates",
    ):
        prefilter_apartment_viewpoints(
            obstacle_map,
            base_m1_camera=_base_request(),
            semantic_targets=semantic,
            maximum_focus_snap_distance_m=4.0,
            selection_count=4,
            required_roles=("split_focus",),
            minimum_pairwise_position_m=0.0,
            minimum_pairwise_yaw_degrees=0.0,
            minimum_pairwise_raster_signature_jaccard=0.0,
            minimum_pairwise_corridor_jaccard=0.0,
        )


def test_pairwise_thresholds_reject_substantively_repeated_candidates() -> None:
    with pytest.raises(
        ApartmentRasterPrefilterError,
        match="satisfy all pairwise difference thresholds",
    ):
        _prefilter(
            selection_count=4,
            required_roles=(),
            minimum_pairwise_position_m=20.0,
            minimum_pairwise_yaw_degrees=0.0,
            minimum_pairwise_raster_signature_jaccard=0.0,
        )


def test_camera_candidates_use_height_specific_rigid_clearance() -> None:
    obstacle = {
        "object_id": 19,
        "handle": "tall_table",
        "obstacle_role": "ground_blocker",
        "blocks_source_center": True,
        "footprint_xz_m": [
            [3.25, 3.25],
            [4.75, 3.25],
            [4.75, 4.75],
            [3.25, 4.75],
        ],
        "world_obb": {
            "center_m": [4.0, 1.5, 4.0],
            "axes_xyz": np.eye(3).tolist(),
            "half_extents_m": [0.75, 0.9, 0.75],
        },
    }
    result = prefilter_apartment_viewpoints(
        _obstacle_map(rigid_obstacles=(obstacle,)),
        base_m1_camera=_base_request(),
        semantic_targets=_targets(),
        selection_count=6,
        candidate_spacing_m=0.75,
        raster_sample_spacing_m=0.5,
        minimum_camera_navmesh_clearance_m=0.2,
        minimum_camera_rigid_clearance_m=0.5,
        minimum_reverse_yaw_difference_deg=10.0,
        minimum_reverse_pose_difference_proxy=0.1,
        minimum_pairwise_position_m=0.5,
        minimum_pairwise_yaw_degrees=5.0,
        minimum_pairwise_raster_signature_jaccard=0.005,
    )

    for viewpoint in result.viewpoints:
        clearance, inside = point_to_world_obb_clearance(
            viewpoint.camera_position_m, obstacle
        )
        assert inside is False
        assert clearance >= 0.5


def test_raw_navmesh_is_checked_before_uint8_cast() -> None:
    obstacle_map = RuntimeObstacleMap(
        binary_navmesh=np.asarray([[1, 256]], dtype=np.uint16),
        bounds_m=((0.0, 0.0, 0.0), (2.0, 3.0, 1.0)),
        floor_height_m=0.0,
        meters_per_pixel=1.0,
        rigid_obstacles=(),
    )

    with pytest.raises(ApartmentRasterPrefilterError, match="invalid binary navmesh"):
        normalize_semantic_targets(obstacle_map, {"target": [0.5, 0.5]})


@pytest.mark.parametrize(
    ("habitat_yaw", "ue_yaw"),
    ((0.0, -90.0), (90.0, -180.0), (180.0, 90.0), (-90.0, 0.0)),
)
def test_four_canonical_m1_to_ue_yaw_conversions(
    habitat_yaw: float, ue_yaw: float
) -> None:
    assert camera_ue_yaw_degrees(habitat_yaw) == ue_yaw


def test_normalize_corridors_is_order_free_and_deduplicates_reversal() -> None:
    raw = _retained_corridors()
    first = normalize_corridors(raw, coordinate_space="habitat_xz_m")
    second = normalize_corridors(list(reversed(raw)), coordinate_space="habitat_xz_m")

    assert [item.to_record() for item in first] == [item.to_record() for item in second]
    assert len(first) == 2
    assert first[0].corridor_id == "a_retained"


def test_fails_when_fewer_than_four_exact_raster_candidates_exist() -> None:
    obstacle_map = _obstacle_map(shape_hw=(1, 1), extent_xz_m=(1.0, 1.0))
    with pytest.raises(
        ApartmentRasterPrefilterError,
        match="fewer than four exact raster candidates",
    ):
        prefilter_apartment_viewpoints(
            obstacle_map,
            base_m1_camera=_base_request(position_m=(0.5, 1.5, 0.5)),
            semantic_targets={"living": [0.5, 0.5]},
            selection_count=4,
            candidate_spacing_m=1.0,
            minimum_camera_navmesh_clearance_m=0.0,
            maximum_positions_per_target=1,
            minimum_reverse_yaw_difference_deg=0.0,
            minimum_reverse_pose_difference_proxy=0.0,
            required_roles=(),
        )


def test_circular_yaw_distance_wraps_at_180() -> None:
    assert circular_yaw_distance_degrees(179.0, -179.0) == pytest.approx(2.0)

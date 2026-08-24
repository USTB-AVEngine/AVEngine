from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import avengine.optional_backends.kujiale_viewpoints as VIEWPOINTS
from avengine.m6x.geometry import RuntimeObstacleMap
from avengine.m6x.raster_pathfinder import build_polygon_raster_obstacle_map
from avengine.m6x.room_feasibility import RoomFeasibilityCompiler
from avengine.optional_backends.kujiale_viewpoints import (
    CPU_PREFILTER_BOUNDARY,
    KujialeViewpointError,
    build_viewpoint_plan,
    generate_polyline_corridors,
    generate_straight_corridors,
    natural_distance_band,
)


REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/m6z/build_kujiale_viewpoint_bank.py"
SPEC = importlib.util.spec_from_file_location("build_kujiale_viewpoint_bank", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)

CANARY_PATH = REPOSITORY / "tools/m6z/run_spear_kujiale_canary.py"
CANARY_SPEC = importlib.util.spec_from_file_location(
    "run_spear_kujiale_canary", CANARY_PATH
)
assert CANARY_SPEC is not None and CANARY_SPEC.loader is not None
CANARY = importlib.util.module_from_spec(CANARY_SPEC)
CANARY_SPEC.loader.exec_module(CANARY)


def _motion_profiles() -> dict:
    return {
        "profiles": {
            "human": {"walk_speed_mps_min": 0.4, "walk_speed_mps_max": 0.8},
            "dog": {"walk_speed_mps_min": 0.4, "walk_speed_mps_max": 0.8},
        },
        "clip_seconds": 5.0,
    }


def _regions(*, blocker: bool = False):
    obstacles = ()
    if blocker:
        obstacles = (
            {
                "object_id": "divider",
                "blocks_source_center": True,
                "footprint_xz_m": [
                    [-0.25, -3.0],
                    [0.25, -3.0],
                    [0.25, 3.0],
                    [-0.25, 3.0],
                ],
                "world_obb": {
                    "center_m": [0.0, 0.5, 0.0],
                    "axes_xyz": np.eye(3).tolist(),
                    "half_extents_m": [0.25, 0.5, 3.0],
                },
            },
        )
    _pathfinder, obstacle_map = build_polygon_raster_obstacle_map(
        polygon_xz_m=[[-4.0, -3.0], [4.0, -3.0], [4.0, 3.0], [-4.0, 3.0]],
        rigid_obstacles=obstacles,
        floor_height_m=0.0,
        meters_per_pixel=0.2,
        padding_m=0.4,
        minimum_clearance_m=0.0,
        authority="synthetic_polygon_footprint_test",
    )
    compiler = RoomFeasibilityCompiler(obstacle_map)
    return {
        slot: compiler.compile(
            source_center_height_m=height,
            sample_spacing_m=0.5,
        )
        for slot, height in (("source1", 1.6), ("source2", 0.45))
    }


def _scene_binding(
    *,
    room_id: str,
    room_scope: str = "room_scope",
    scene_id: str = "scene",
    map_path: str = "/Game/AVEngine/Test/scene",
    stage_name: str = "scene.usda",
) -> dict:
    return {
        "dataset_id": "spatialverse/InteriorAgent",
        "scene_id": scene_id,
        "room_id": room_id,
        "room_scope": room_scope,
        "map_path": map_path,
        "source_stage_name": stage_name,
    }


def _camera_catalog(
    binding: dict,
    *,
    positions_xy: tuple[tuple[float, float], ...] = (
        (-3.0, -2.0),
        (3.0, -2.0),
        (-3.0, 2.0),
        (3.0, 2.0),
    ),
    room_type: str = "unknown room kind",
) -> dict:
    heights = {"low": 1.0, "base": 1.5, "high": 1.9}
    yaws = list(range(0, 360, 30))
    placements = []
    for position_index, (x, y) in enumerate(positions_xy):
        for height_id, height in heights.items():
            for yaw in yaws:
                placements.append(
                    {
                        "placement_id": f"camera_{len(placements) + 1:06d}",
                        "placement_status": "camera_center_feasible_only",
                        "position_m": [x, height, y],
                        "height_id": height_id,
                        "camera_height_above_floor_m": height,
                        "yaw_deg": yaw,
                        "raster_pixel_rc": [position_index, 0],
                        "room_instance_id": binding["room_id"],
                        "room_scope": binding["room_scope"],
                        "room_type_annotation": room_type,
                        "semantic_target": None,
                        "source_path_join_status": "pending",
                        "difficulty_status": "not_evaluated",
                    }
                )
    return {
        "candidate_filtering": "none",
        "studio_used": False,
        "room_id": binding["room_id"],
        "room_scope": binding["room_scope"],
        "room_type_annotation": room_type,
        "scene_binding": dict(binding),
        "authority": "synthetic_target_agnostic_camera_catalog",
        "obstacle_map": {"floor_height_m": 0.0},
        "meters_per_pixel": 0.2,
        "free_space_raster_pixel_count": 100,
        "camera_height_variants_above_floor_m": heights,
        "yaw_degrees": yaws,
        "sampled_floor_position_count": len(positions_xy),
        "placement_count": len(placements),
        "placements": placements,
    }


def test_natural_distance_band_is_shared_across_species() -> None:
    assert natural_distance_band(
        _motion_profiles(), ["human", "dog"]
    ) == pytest.approx((0.4 * 74 / 15, 0.8 * 74 / 15))
    with pytest.raises(KujialeViewpointError, match="missing species"):
        natural_distance_band(_motion_profiles(), ["cat"])


def test_straight_corridors_never_cross_a_blocking_raster() -> None:
    corridors = generate_straight_corridors(
        region_by_source=_regions(blocker=True),
        motion_profiles=_motion_profiles(),
        maximum_corridors=80,
    )
    assert corridors
    for corridor in corridors:
        start_x = corridor["start_xy_m"][0]
        end_x = corridor["end_xy_m"][0]
        assert not (start_x < -0.25 and end_x > 0.25)
        assert 0.4 * 74 / 15 - 1.0e-9 <= corridor["length_m"]
        assert corridor["length_m"] <= 0.8 * 74 / 15 + 1.0e-9
        assert corridor["claim_boundary"] == CPU_PREFILTER_BOUNDARY


def test_polyline_corridors_are_multisegment_simple_and_exactly_feasible() -> None:
    regions = _regions()
    corridors = generate_polyline_corridors(
        region_by_source=regions,
        motion_profiles=_motion_profiles(),
        maximum_corridors=48,
        waypoint_spacing_m=0.5,
        maximum_segments=4,
        maximum_segment_length_m=1.8,
    )
    assert corridors
    for corridor in corridors:
        assert corridor["route_geometry"] == "polyline"
        assert len(corridor["waypoints_xy_m"]) >= 3
        assert len(set(map(tuple, corridor["waypoints_xy_m"]))) == len(
            corridor["waypoints_xy_m"]
        )
        assert corridor["simple_non_repeated_edges"] is True
        assert 0.4 * 74 / 15 - 1.0e-9 <= corridor["length_m"]
        assert corridor["length_m"] <= 0.8 * 74 / 15 + 1.0e-9
        assert corridor["replay_speed_mps"] <= 0.8 + 1.0e-9
        for left, right in zip(
            corridor["waypoints_xy_m"], corridor["waypoints_xy_m"][1:]
        ):
            assert VIEWPOINTS._segment_clear(
                regions["source1"].obstacle_map,
                regions["source1"].feasible_mask,
                left,
                right,
            )
            assert VIEWPOINTS._segment_clear(
                regions["source2"].obstacle_map,
                regions["source2"].feasible_mask,
                left,
                right,
            )


def test_raster_los_conservatively_keeps_corner_touch_cells() -> None:
    obstacle_map = RuntimeObstacleMap(
        binary_navmesh=np.asarray([[1, 0], [1, 1]], dtype=np.uint8),
        bounds_m=((0.0, -0.5, 0.0), (2.0, 3.5, 2.0)),
        floor_height_m=0.0,
        meters_per_pixel=1.0,
        rigid_obstacles=(),
    )
    mask = np.asarray(obstacle_map.binary_navmesh, dtype=np.bool_)
    assert not VIEWPOINTS._segment_clear(obstacle_map, mask, (0.5, 0.5), (1.5, 1.5))


def test_nonadjacent_collinear_overlap_and_endpoint_touch_are_not_simple() -> None:
    assert VIEWPOINTS._segments_intersect_or_touch(
        (0.0, 0.0), (2.0, 0.0), (3.0, 0.0), (1.0, 0.0)
    )
    assert VIEWPOINTS._segments_intersect_or_touch(
        (0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (1.0, 0.0)
    )
    assert not VIEWPOINTS._segments_intersect_or_touch(
        (0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 1.0)
    )


def test_viewpoint_selection_is_deterministic_diverse_and_pending_pixels() -> None:
    regions = _regions()
    corridors = generate_straight_corridors(
        region_by_source=regions,
        motion_profiles=_motion_profiles(),
        maximum_corridors=96,
    )
    binding = _scene_binding(room_id="kujiale_test_living")
    kwargs = {
        "room_id": "kujiale_test_living",
        "room_type": "living room",
        "scene_binding": binding,
        "camera_catalog": _camera_catalog(binding),
        "region_by_source": regions,
        "corridors": corridors,
        "view_count": 4,
    }
    first = build_viewpoint_plan(**kwargs)
    second = build_viewpoint_plan(**kwargs)
    assert first == second
    bank = first["bank"]
    assert bank["status"] == "cpu_prefilter_complete"
    assert bank["viewpoint_count"] == 4
    assert bank["native_pixel_validation_status"] == "pending"
    assert bank["ue_spear_certification_status"] == "pending"
    assert {view["position_xyz_m"][2] for view in bank["viewpoints"]}.issuperset(
        {1.0, 1.9}
    )
    poses = {
        (*view["position_xyz_m"], view["yaw_ue_deg"]) for view in bank["viewpoints"]
    }
    assert len(poses) == 4
    assert all(view["eligible_corridor_count"] > 0 for view in bank["viewpoints"])
    assert all(
        view["directed_pair_combination_count"] > view["described_pair_count"] > 0
        for view in bank["viewpoints"]
    )
    assert bank["camera_pool_placement_count"] == 4 * 3 * 12
    assert all(
        view["camera_catalog_placement_id"].startswith("camera_")
        for view in bank["viewpoints"]
    )
    assert set(first["corridor_subsets"]) == {
        view["viewpoint_id"] for view in bank["viewpoints"]
    }


@pytest.mark.parametrize("view_count", [3, 9])
def test_view_count_outside_four_to_eight_is_rejected(view_count: int) -> None:
    regions = _regions()
    corridors = generate_straight_corridors(
        region_by_source=regions,
        motion_profiles=_motion_profiles(),
        maximum_corridors=32,
    )
    binding = _scene_binding(room_id="room")
    with pytest.raises(KujialeViewpointError, match="view_count"):
        build_viewpoint_plan(
            room_id="room",
            room_type="test",
            scene_binding=binding,
            camera_catalog=_camera_catalog(binding),
            region_by_source=regions,
            corridors=corridors,
            view_count=view_count,
        )


def test_distinct_route_precondition_fails_instead_of_modulo_reuse() -> None:
    regions = _regions()
    corridors = generate_straight_corridors(
        region_by_source=regions,
        motion_profiles=_motion_profiles(),
        maximum_corridors=4,
    )[:3]
    binding = _scene_binding(room_id="room")
    with pytest.raises(KujialeViewpointError, match="3 distinct corridors"):
        build_viewpoint_plan(
            room_id="room",
            room_type="unknown",
            scene_binding=binding,
            camera_catalog=_camera_catalog(binding),
            region_by_source=regions,
            corridors=corridors,
            view_count=4,
        )


def test_camera_pool_is_catalog_owned_not_source_mask_sampled() -> None:
    regions = _regions()
    corridors = generate_straight_corridors(
        region_by_source=regions,
        motion_profiles=_motion_profiles(),
        maximum_corridors=16,
    )
    binding = _scene_binding(room_id="room")
    outside_source_mask = (
        (20.0, 20.0),
        (22.0, 20.0),
        (20.0, 22.0),
        (22.0, 22.0),
    )
    result = build_viewpoint_plan(
        room_id="room",
        room_type="unknown",
        scene_binding=binding,
        camera_catalog=_camera_catalog(binding, positions_xy=outside_source_mask),
        region_by_source=regions,
        corridors=corridors,
        view_count=4,
    )
    assert {
        tuple(view["position_xyz_m"][:2]) for view in result["bank"]["viewpoints"]
    }.issubset(set(outside_source_mask))
    assert all(
        view["visible_route_probe_count"] == 0
        for view in result["bank"]["viewpoints"]
    )


def test_camera_catalog_cross_binding_and_complete_grid_are_checked() -> None:
    regions = _regions()
    corridors = generate_straight_corridors(
        region_by_source=regions,
        motion_profiles=_motion_profiles(),
        maximum_corridors=8,
    )
    binding = _scene_binding(room_id="room")
    mismatched = _camera_catalog(binding)
    mismatched["scene_binding"]["map_path"] = "/Game/Wrong"
    with pytest.raises(KujialeViewpointError, match="map_path differ"):
        build_viewpoint_plan(
            room_id="room",
            room_type="unknown",
            scene_binding=binding,
            camera_catalog=mismatched,
            region_by_source=regions,
            corridors=corridors,
        )

    incomplete = _camera_catalog(binding)
    incomplete["placements"].pop()
    incomplete["placement_count"] -= 1
    with pytest.raises(KujialeViewpointError, match="completely cross"):
        build_viewpoint_plan(
            room_id="room",
            room_type="unknown",
            scene_binding=binding,
            camera_catalog=incomplete,
            region_by_source=regions,
            corridors=corridors,
        )


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_cli_is_fresh_no_clobber_and_keeps_certification_pending(
    tmp_path: Path,
) -> None:
    metadata = {
        "schema": "avengine_optional_residential_scene_metadata_v1",
        "dataset_id": "spatialverse/InteriorAgent",
        "scene_id": "kujiale_test",
        "room_id": "kujiale_test_living",
        "room_scope": "livingroom_test",
        "room_type": "living room",
        "room_polygon_xy_m": [
            [-4.0, -3.0],
            [4.0, -3.0],
            [4.0, 3.0],
            [-4.0, 3.0],
        ],
        "floor_z_m": 0.0,
        "objects": [],
        "object_role_counts": {},
        "source_reference": "/external/kujiale_test_full_home.usda",
    }
    profile = {
        "schema": "avengine_optional_residential_source_profile_v1",
        "dataset_id": "spatialverse/InteriorAgent",
        "scene_id": "kujiale_test",
        "map_path": "/Game/AVEngine/Test/kujiale_test_full_home_v1",
        "source_center_margin_m": 0.03,
        "emitter_heights_m": {"human0": 1.6, "dog0": 0.45},
    }
    visual_profile = {
        "schema_version": "avengine_optional_interioragent_kujiale_profile_v1",
        "dataset_id": "spatialverse/InteriorAgent",
        "scene_id": "kujiale_test",
        "map_path": "/Game/AVEngine/Test/kujiale_test_full_home_v1",
        "source_stage_name": "kujiale_test_full_home.usda",
        "selected_scopes": ["livingroom_test", "wall", "ceiling", "floor"],
    }
    binding = _scene_binding(
        room_id="kujiale_test_living",
        room_scope="livingroom_test",
        scene_id="kujiale_test",
        map_path="/Game/AVEngine/Test/kujiale_test_full_home_v1",
        stage_name="kujiale_test_full_home.usda",
    )
    args = argparse.Namespace(
        scene_metadata=_write_json(tmp_path / "metadata.json", metadata),
        source_profile=_write_json(tmp_path / "profile.json", profile),
        visual_profile=_write_json(tmp_path / "visual.json", visual_profile),
        motion_profiles=_write_json(tmp_path / "motion.json", _motion_profiles()),
        camera_catalog=_write_json(
            tmp_path / "camera_catalog.json", _camera_catalog(binding)
        ),
        species=["human", "dog"],
        route_geometry="straight",
        view_count=4,
        hfov_degrees=105.0,
        meters_per_pixel=0.2,
        minimum_clearance_m=None,
        sample_spacing_m=0.5,
        side_margin=0.06,
        maximum_corridors=96,
        waypoint_spacing_m=0.25,
        maximum_route_segments=4,
        maximum_segment_length_m=1.8,
        screening_corridor_count=32,
        minimum_pair_separation_m=0.60,
        maximum_pair_records=64,
        output=tmp_path / "fresh",
    )
    output = TOOL.run(args)
    assert output == args.output.resolve()
    delivery = json.loads((output / "delivery.json").read_text())
    assert delivery["status"] == "cpu_prefilter_complete"
    assert delivery["native_pixel_validation_status"] == "pending"
    assert delivery["ue_spear_certification_status"] == "pending"
    assert delivery["scene_binding"]["map_path"] == (
        "/Game/AVEngine/Test/kujiale_test_full_home_v1"
    )
    assert delivery["scene_binding"]["source_stage_name"] == (
        "kujiale_test_full_home.usda"
    )
    assert (output / "viewpoint_topdown.png").is_file()
    assert len(list((output / "corridors").glob("*.json"))) == 4
    selected_route_ids = set()
    for subset_path in (output / "corridors").glob("*.json"):
        subset = json.loads(subset_path.read_text())
        assert subset["minimum_pair_separation_m"] == 0.60
        assert subset["directed_pair_combination_count"] > 0
        assert subset["described_pair_count"] == 64
        assert "compatible_both_moving_pair_count" not in subset
        assert "compatible_both_moving_pairs" not in subset
        assert any(
            pair["continuous_closest_planar_distance_m"] < 0.60
            for pair in subset["described_pair_records"]
        )
        assert {
            item["motion_case"] for item in subset["four_point_motion_cases"]
        } == {
            "static_static",
            "source1_moving_source2_static",
            "source1_static_source2_moving",
            "both_moving",
        }
        route_ids = {
            item["route_selection"]["corridor_id"]
            for item in subset["four_point_motion_cases"]
        }
        assert len(route_ids) == 1
        selected_route_ids |= route_ids
    assert len(selected_route_ids) == 4
    with pytest.raises(FileExistsError, match="refusing to replace"):
        TOOL.run(args)


def test_cli_polyline_unknown_room_has_no_straight_fallback(
    tmp_path: Path,
) -> None:
    metadata = {
        "schema": "avengine_optional_residential_scene_metadata_v1",
        "dataset_id": "spatialverse/InteriorAgent",
        "scene_id": "kujiale_unknown",
        "room_id": "kujiale_unknown_room_17",
        "room_scope": "room_17",
        "room_type": "never seen room label",
        "room_polygon_xy_m": [
            [-3.0, -2.5],
            [3.0, -2.5],
            [3.0, 2.5],
            [-3.0, 2.5],
        ],
        "floor_z_m": 0.0,
        "objects": [],
        "object_role_counts": {},
        "source_reference": "/external/kujiale_unknown_full_home.usda",
    }
    profile = {
        "schema": "avengine_optional_residential_source_profile_v1",
        "dataset_id": "spatialverse/InteriorAgent",
        "scene_id": "kujiale_unknown",
        "map_path": "/Game/AVEngine/Test/kujiale_unknown_full_home_v1",
        "source_center_margin_m": 0.03,
        "emitter_heights_m": {"human0": 1.6, "dog0": 0.45},
    }
    visual = {
        "schema_version": "avengine_optional_interioragent_kujiale_profile_v1",
        "dataset_id": "spatialverse/InteriorAgent",
        "scene_id": "kujiale_unknown",
        "map_path": "/Game/AVEngine/Test/kujiale_unknown_full_home_v1",
        "source_stage_name": "kujiale_unknown_full_home.usda",
        "selected_scopes": ["room_17", "wall", "ceiling", "floor"],
    }
    binding = _scene_binding(
        room_id="kujiale_unknown_room_17",
        room_scope="room_17",
        scene_id="kujiale_unknown",
        map_path="/Game/AVEngine/Test/kujiale_unknown_full_home_v1",
        stage_name="kujiale_unknown_full_home.usda",
    )
    args = TOOL.parse_args(
        [
            "--scene-metadata",
            str(_write_json(tmp_path / "metadata.json", metadata)),
            "--source-profile",
            str(_write_json(tmp_path / "profile.json", profile)),
            "--visual-profile",
            str(_write_json(tmp_path / "visual.json", visual)),
            "--motion-profiles",
            str(_write_json(tmp_path / "motion.json", _motion_profiles())),
            "--camera-catalog",
            str(
                _write_json(
                    tmp_path / "camera_catalog.json",
                    _camera_catalog(binding, room_type="never seen room label"),
                )
            ),
            "--route-geometry",
            "polyline",
            "--meters-per-pixel",
            "0.2",
            "--sample-spacing-m",
            "0.5",
            "--maximum-corridors",
            "32",
            "--screening-corridor-count",
            "16",
            "--maximum-pair-records",
            "16",
            "--output",
            str(tmp_path / "polyline"),
        ]
    )
    output = TOOL.run(args)
    delivery = json.loads((output / "delivery.json").read_text())
    routes = json.loads((output / "polyline_corridors.json").read_text())
    assert delivery["route_geometry"] == "polyline"
    assert delivery["route_fallback_used"] is False
    assert routes["room_type_annotation"] == "never seen room label"
    assert routes["room_type_role"] == "post_generation_annotation_only"
    assert not (output / "straight_corridors.json").exists()


def test_polyline_failure_is_fresh_and_does_not_fallback(
    tmp_path: Path,
) -> None:
    metadata = {
        "schema": "avengine_optional_residential_scene_metadata_v1",
        "dataset_id": "spatialverse/InteriorAgent",
        "scene_id": "tiny",
        "room_id": "tiny_room",
        "room_scope": "tiny_scope",
        "room_type": "unknown",
        "room_polygon_xy_m": [[0, 0], [0.2, 0], [0.2, 0.2], [0, 0.2]],
        "floor_z_m": 0.0,
        "objects": [],
        "object_role_counts": {},
        "source_reference": "/external/tiny.usda",
    }
    profile = {
        "schema": "avengine_optional_residential_source_profile_v1",
        "dataset_id": "spatialverse/InteriorAgent",
        "scene_id": "tiny",
        "map_path": "/Game/AVEngine/Test/tiny",
        "emitter_heights_m": {"human0": 1.6, "dog0": 0.45},
    }
    visual = {
        "schema_version": "avengine_optional_interioragent_kujiale_profile_v1",
        "dataset_id": "spatialverse/InteriorAgent",
        "scene_id": "tiny",
        "map_path": "/Game/AVEngine/Test/tiny",
        "source_stage_name": "tiny.usda",
        "selected_scopes": ["tiny_scope", "wall", "ceiling", "floor"],
    }
    binding = _scene_binding(
        room_id="tiny_room",
        room_scope="tiny_scope",
        scene_id="tiny",
        map_path="/Game/AVEngine/Test/tiny",
        stage_name="tiny.usda",
    )
    args = TOOL.parse_args(
        [
            "--scene-metadata",
            str(_write_json(tmp_path / "tiny.json", metadata)),
            "--source-profile",
            str(_write_json(tmp_path / "tiny_profile.json", profile)),
            "--visual-profile",
            str(_write_json(tmp_path / "tiny_visual.json", visual)),
            "--motion-profiles",
            str(_write_json(tmp_path / "tiny_motion.json", _motion_profiles())),
            "--camera-catalog",
            str(
                _write_json(
                    tmp_path / "tiny_camera_catalog.json",
                    _camera_catalog(binding),
                )
            ),
            "--route-geometry",
            "polyline",
            "--meters-per-pixel",
            "0.1",
            "--output",
            str(tmp_path / "failed"),
        ]
    )
    with pytest.raises(KujialeViewpointError, match="polyline"):
        TOOL.run(args)
    assert (args.output / "FAILED.json").is_file()
    assert not (args.output / "straight_corridors.json").exists()


def _binding_inputs() -> tuple[dict, dict, dict]:
    metadata = {
        "schema": "avengine_optional_residential_scene_metadata_v1",
        "dataset_id": "spatialverse/InteriorAgent",
        "scene_id": "kujiale_0020",
        "room_id": "kujiale_0020_livingroom_491",
        "room_scope": "livingroom_491",
        "source_reference": "/external/kujiale_0020_full_home_preview_surface.usda",
    }
    source_profile = {
        "schema": "avengine_optional_residential_source_profile_v1",
        "dataset_id": "spatialverse/InteriorAgent",
        "scene_id": "kujiale_0020",
        "map_path": "/Game/AVEngine/Optional/Kujiale/kujiale_0020_full_home_v1",
    }
    visual_profile = {
        "schema_version": "avengine_optional_interioragent_kujiale_profile_v1",
        "dataset_id": "spatialverse/InteriorAgent",
        "scene_id": "kujiale_0020",
        "map_path": "/Game/AVEngine/Optional/Kujiale/kujiale_0020_full_home_v1",
        "source_stage_name": "kujiale_0020_full_home_preview_surface.usda",
        "selected_scopes": ["livingroom_491", "wall", "ceiling", "floor"],
    }
    return metadata, source_profile, visual_profile


def test_current_living_profiles_bind_full_home_map_and_stage() -> None:
    metadata, _, _ = _binding_inputs()
    source_profile = json.loads(
        (
            REPOSITORY / "examples/m6z/interioragent_kujiale_0020_source_episode.json"
        ).read_text(encoding="utf-8")
    )
    visual_profile = json.loads(
        (
            REPOSITORY
            / "examples/m6z/interioragent_kujiale_0020_full_home_visual_profile.json"
        ).read_text(encoding="utf-8")
    )
    binding = TOOL._validate_scene_binding(
        metadata=metadata,
        source_profile=source_profile,
        visual_profile=visual_profile,
    )
    assert binding["map_path"].endswith("/kujiale_0020_full_home_v1")
    assert binding["source_stage_name"] == (
        "kujiale_0020_full_home_preview_surface.usda"
    )


def test_scene_binding_rejects_cross_wired_scene() -> None:
    metadata, source_profile, visual_profile = _binding_inputs()
    source_profile["scene_id"] = "kujiale_other"
    with pytest.raises(RuntimeError, match="scene_id differ"):
        TOOL._validate_scene_binding(
            metadata=metadata,
            source_profile=source_profile,
            visual_profile=visual_profile,
        )


def test_scene_binding_rejects_cross_wired_map() -> None:
    metadata, source_profile, visual_profile = _binding_inputs()
    source_profile["map_path"] = "/Game/AVEngine/Optional/Kujiale/kujiale_0020_live_v2"
    with pytest.raises(RuntimeError, match="map_path differ"):
        TOOL._validate_scene_binding(
            metadata=metadata,
            source_profile=source_profile,
            visual_profile=visual_profile,
        )


def test_scene_binding_rejects_cross_wired_stage() -> None:
    metadata, source_profile, visual_profile = _binding_inputs()
    metadata["source_reference"] = "/external/kujiale_0020_live.usda"
    with pytest.raises(RuntimeError, match="source stage differ"):
        TOOL._validate_scene_binding(
            metadata=metadata,
            source_profile=source_profile,
            visual_profile=visual_profile,
        )


def test_scene_binding_does_not_use_room_semantics_as_candidate_authority() -> None:
    metadata, source_profile, visual_profile = _binding_inputs()
    metadata["room_type"] = "previously unseen room label"
    binding = TOOL._validate_scene_binding(
        metadata=metadata,
        source_profile=source_profile,
        visual_profile=visual_profile,
    )
    assert binding["room_scope"] == "livingroom_491"


def test_kujiale_montage_supports_an_odd_view_count(tmp_path: Path) -> None:
    import cv2

    records = []
    for index in range(5):
        path = tmp_path / f"view_{index}.png"
        assert cv2.imwrite(str(path), np.full((12, 18, 3), index * 20, dtype=np.uint8))
        records.append(
            {
                "path": str(path),
                "view_id": f"view_{index}",
                "yaw_deg": float(index * 15),
            }
        )
    montage = CANARY._write_montage(tmp_path, records, backend_role="production_visual")
    image = cv2.imread(str(montage), cv2.IMREAD_COLOR)
    assert montage.name == "kujiale_viewpoint_montage.png"
    assert image.shape == (36, 36, 3)

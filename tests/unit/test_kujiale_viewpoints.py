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


def test_natural_distance_band_is_shared_across_species() -> None:
    assert natural_distance_band(_motion_profiles(), ["human", "dog"]) == (
        2.0,
        4.0,
    )
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
        assert 2.0 - 1.0e-9 <= corridor["length_m"] <= 4.0 + 1.0e-9
        assert corridor["claim_boundary"] == CPU_PREFILTER_BOUNDARY


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


def test_viewpoint_selection_is_deterministic_diverse_and_pending_pixels() -> None:
    regions = _regions()
    corridors = generate_straight_corridors(
        region_by_source=regions,
        motion_profiles=_motion_profiles(),
        maximum_corridors=96,
    )
    kwargs = {
        "room_id": "kujiale_test_living",
        "room_type": "living room",
        "region_by_source": regions,
        "corridors": corridors,
        "view_count": 4,
        "camera_heights_m": [1.0, 1.5, 1.9],
        "camera_spacing_m": 1.0,
        "coverage_spacing_m": 0.5,
        "yaw_step_degrees": 30.0,
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
        view["compatible_both_moving_pair_count"] > 0 for view in bank["viewpoints"]
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
    with pytest.raises(KujialeViewpointError, match="view_count"):
        build_viewpoint_plan(
            room_id="room",
            room_type="test",
            region_by_source=regions,
            corridors=corridors,
            view_count=view_count,
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
    args = argparse.Namespace(
        scene_metadata=_write_json(tmp_path / "metadata.json", metadata),
        source_profile=_write_json(tmp_path / "profile.json", profile),
        visual_profile=_write_json(tmp_path / "visual.json", visual_profile),
        motion_profiles=_write_json(tmp_path / "motion.json", _motion_profiles()),
        species=["human", "dog"],
        view_count=4,
        hfov_degrees=105.0,
        camera_heights_m=[1.0, 1.5, 1.9],
        meters_per_pixel=0.2,
        minimum_clearance_m=None,
        sample_spacing_m=0.5,
        camera_spacing_m=1.0,
        coverage_spacing_m=0.5,
        yaw_step_degrees=30.0,
        side_margin=0.06,
        maximum_corridors=96,
        maximum_camera_positions=32,
        maximum_coverage_samples=96,
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
    for subset_path in (output / "corridors").glob("*.json"):
        subset = json.loads(subset_path.read_text())
        assert subset["minimum_pair_separation_m"] == 0.60
        assert subset["compatible_both_moving_pair_count"] > 0
        assert all(
            pair["minimum_continuous_linear_separation_m"] >= 0.60
            and pair["opposite_sides_at_first_anchor"] is True
            for pair in subset["compatible_both_moving_pairs"]
        )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        TOOL.run(args)


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

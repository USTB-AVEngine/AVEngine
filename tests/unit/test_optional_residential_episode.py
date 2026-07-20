from __future__ import annotations

from copy import deepcopy

import pytest

from avengine.m5.timeline import TIMELINE_SCHEMA, json_schema_errors
from avengine.optional_backends.residential_episode import (
    DOG_SOURCE_ID,
    EPISODE_SCHEMA,
    HUMAN_SOURCE_ID,
    PROFILE_SCHEMA,
    ResidentialEpisodeError,
    SCENE_METADATA_SCHEMA,
    build_residential_source_episode,
    classify_object_bounds,
    dataset_z_up_to_habitat,
    object_footprint_rectangles_xy,
    point_in_polygon_xy,
)


def _metadata() -> dict:
    return {
        "schema": SCENE_METADATA_SCHEMA,
        "dataset_id": "test/residential",
        "scene_id": "room0",
        "room_id": "room0_living",
        "room_polygon_xy_m": [[-2, -3], [2, -3], [2, 3], [-2, 3]],
        "objects": [
            {
                "object_id": "sofa0",
                "bounds_xyz_m": [[-1.8, -0.4, 0.0], [-1.2, 0.4, 0.9]],
                "navigation_role": "ground_blocker",
            },
            {
                "object_id": "rug0",
                "bounds_xyz_m": [[-1.0, -2.0, 0.0], [1.0, 2.0, 0.03]],
                "navigation_role": "walkable_floor_covering",
            },
            {
                "object_id": "lamp0",
                "bounds_xyz_m": [[-0.2, -0.2, 2.1], [0.2, 0.2, 2.7]],
                "navigation_role": "elevated_object",
            },
        ],
        "claim_boundary": "synthetic test fixture",
    }


def _profile() -> dict:
    return {
        "schema": PROFILE_SCHEMA,
        "scene_id": "room0",
        "map_path": "/Game/AVEngine/Test/room0",
        "camera": {
            "position_xyz_m": [0.0, -2.5, 1.55],
            "yaw_ue_deg": 90.0,
            "horizontal_fov_deg": 105.0,
        },
        "routes": {
            "dog0": {
                "start_xyz_m": [-0.4, 2.0, 0.0],
                "end_xyz_m": [-0.4, -1.5, 0.0],
            },
            "human0": {
                "start_xyz_m": [0.4, -1.5, 0.064],
                "end_xyz_m": [0.4, 2.0, 0.064],
            },
        },
        "source_center_margin_m": 0.03,
        "emitter_heights_m": {"dog0": 0.45, "human0": 1.60},
        "review_lights": [],
        "acoustic_proxy": {"label": "test_proxy"},
    }


def test_builds_schema_valid_closed_two_source_episode() -> None:
    result = build_residential_source_episode(
        scene_metadata=_metadata(), profile=_profile()
    )

    assert result["schema"] == EPISODE_SCHEMA
    assert json_schema_errors(result["timeline"], TIMELINE_SCHEMA) == []
    assert len(result["timeline"]["frames"]) == 75
    assert [actor["actor_id"] for actor in result["timeline"]["actors"]] == [
        "dog0",
        "human0",
    ]
    assert set(result["source_trajectories_habitat_m"]) == {
        DOG_SOURCE_ID,
        HUMAN_SOURCE_ID,
    }
    assert result["qualification"]["source_center_gate"]["status"] == "pass"
    assert result["qualification"]["listener"]["orientation_wxyz"] == pytest.approx(
        [0.0, 0.0, -1.0, 0.0], abs=1.0e-12
    )
    assert result["visual_plan"]["camera"]["ue_position_cm"] == [0.0, -250.0, 155.0]
    assert result["visual_plan"]["camera"]["ue_yaw_deg"] == 90.0
    assert result["visual_plan"]["authority"]["backend_may_replan"] is False
    assert result["visual_lighting"] == {
        "profile_id": "explicit_review_lights",
        "native_usd_rendering_scope_retained": True,
        "generated_review_light_count": 0,
        "claim_boundary": (
            "generated visual review lights; not reconstructed physical lighting"
        ),
    }
    assert all(
        0.70 < result["route_metrics"][actor_id]["mean_speed_mps"] < 0.75
        for actor_id in ("dog0", "human0")
    )


def test_actor_yaw_tracks_opposite_motion_and_events_overlap() -> None:
    result = build_residential_source_episode(
        scene_metadata=_metadata(), profile=_profile()
    )
    states = {
        item["actor_id"]: item
        for item in result["visual_plan"]["frames"][0]["actor_states"]
    }
    assert states["dog0"]["actor_yaw_ue_deg"] == pytest.approx(-90.0)
    assert states["human0"]["actor_yaw_ue_deg"] == pytest.approx(0.0)
    assert any(
        dog and human
        for dog, human in zip(
            result["source_activity_by_frame"][DOG_SOURCE_ID],
            result["source_activity_by_frame"][HUMAN_SOURCE_ID],
            strict=True,
        )
    )


def test_polyline_route_accumulates_distance_and_updates_heading() -> None:
    profile = _profile()
    profile["routes"]["dog0"] = {
        "waypoints_xyz_m": [
            [-0.4, 2.0, 0.0],
            [-0.4, -1.5, 0.0],
            [0.0, -1.5, 0.0],
            [0.0, 2.0, 0.0],
        ]
    }
    result = build_residential_source_episode(
        scene_metadata=_metadata(), profile=profile
    )
    assert result["route_metrics"]["dog0"]["mean_speed_mps"] > 1.45
    yaws = {
        round(item["actor_yaw_ue_deg"], 3)
        for frame in result["visual_plan"]["frames"]
        for item in frame["actor_states"]
        if item["actor_id"] == "dog0"
    }
    assert len(yaws) >= 3


def test_center_gate_rejects_obstacle_but_ignores_rug_and_chandelier() -> None:
    profile = _profile()
    profile["routes"]["human0"] = {
        "start_xyz_m": [-1.5, -0.2, 0.064],
        "end_xyz_m": [-1.5, 0.2, 0.064],
    }
    with pytest.raises(ResidentialEpisodeError, match="route gate failed"):
        build_residential_source_episode(scene_metadata=_metadata(), profile=profile)

    assert classify_object_bounds([[-1, -1, 0], [1, 1, 0.04]]) == (
        "walkable_floor_covering"
    )
    assert classify_object_bounds([[-1, -1, 2], [1, 1, 2.6]]) == "elevated_object"
    assert classify_object_bounds([[-1, -1, 0.5], [1, 1, 1.0]]) == (
        "elevated_object"
    )


def test_mesh_part_footprints_do_not_fill_empty_parent_aabb() -> None:
    metadata = _metadata()
    metadata["objects"][0] = {
        "object_id": "l_shaped_cabinet",
        "bounds_xyz_m": [[-1.8, -2.5, 0.0], [1.8, 2.5, 1.0]],
        "navigation_role": "ground_blocker",
        "footprint_parts_xy_m": [
            [[-1.8, -2.5], [-1.2, 2.5]],
            [[1.2, -2.5], [1.8, 2.5]],
        ],
    }
    result = build_residential_source_episode(
        scene_metadata=metadata, profile=_profile()
    )
    assert result["qualification"]["source_center_gate"]["status"] == "pass"
    assert object_footprint_rectangles_xy(metadata["objects"][0]) == [
        [[-1.8, -2.5], [-1.2, 2.5]],
        [[1.2, -2.5], [1.8, 2.5]],
    ]


def test_empty_mesh_footprint_falls_back_to_declared_bounds() -> None:
    item = {
        "object_id": "missing_low_slice",
        "bounds_xyz_m": [[-0.5, -0.25, 0.0], [0.5, 0.25, 1.0]],
        "navigation_role": "ground_blocker",
        "footprint_parts_xy_m": [],
    }
    assert object_footprint_rectangles_xy(item) == [
        [[-0.5, -0.25], [0.5, 0.25]]
    ]


def test_coordinate_and_polygon_helpers_are_bounded() -> None:
    assert dataset_z_up_to_habitat([1.0, 2.0, 3.0]) == [1.0, 3.0, 2.0]
    polygon = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
    assert point_in_polygon_xy([1.0, 1.0], polygon)
    assert point_in_polygon_xy([0.0, 1.0], polygon)
    assert not point_in_polygon_xy([3.0, 1.0], polygon)


def test_profile_scene_mismatch_is_rejected() -> None:
    profile = deepcopy(_profile())
    profile["scene_id"] = "different"
    with pytest.raises(ResidentialEpisodeError, match="scene_id differ"):
        build_residential_source_episode(scene_metadata=_metadata(), profile=profile)

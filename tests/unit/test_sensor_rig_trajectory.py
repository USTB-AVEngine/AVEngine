from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from avengine.capture.orientation import habitat_yaw_degrees_from_xyzw
from avengine.m6x.trajectory import resample_polyline_by_arc_length
from avengine.sensor_rig_trajectory import (
    DURATION_TICKS,
    FRAME_COUNT,
    TICKS_PER_FRAME,
    TIME_BASE_HZ,
    SensorRigTrajectoryError,
    compute_sensor_rig_pose_hash,
    materialize_sensor_rig_trajectory,
    validate_sensor_rig_trajectory,
)


def _yaw_at(record: dict, frame_index: int) -> float:
    return habitat_yaw_degrees_from_xyzw(
        record["frames"][frame_index]["world_from_rig"]["rotation_xyzw"]
    )


def test_hold_is_one_deterministic_colocated_pose_on_timeline_v2_clock() -> None:
    program = {
        "kind": "HOLD",
        "position_m": [1, 1.6, -2],
        "yaw_deg": 55,
    }
    first = materialize_sensor_rig_trajectory(
        trajectory_id="rig_hold_v1",
        program=program,
    )
    second = materialize_sensor_rig_trajectory(
        trajectory_id="rig_hold_v1",
        program=program,
    )

    assert first == second
    assert validate_sensor_rig_trajectory(first) == []
    assert first["schema"] == "avengine_sensor_rig_trajectory_v1"
    assert first["time_base_hz"] == TIME_BASE_HZ == 48_000
    assert first["duration_ticks"] == DURATION_TICKS == 240_000
    assert first["frame_count"] == FRAME_COUNT == 75
    assert first["ticks_per_frame"] == TICKS_PER_FRAME == 3_200
    assert first["camera_listener_coupling"] == "rigid_colocated_cooriented"
    assert first["rig_from_camera"] == first["rig_from_listener"]
    assert first["rig_from_listener"] == {
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    assert [frame["frame_index"] for frame in first["frames"]] == list(
        range(FRAME_COUNT)
    )
    assert [frame["pts_ticks"] for frame in first["frames"]] == [
        frame_index * TICKS_PER_FRAME for frame_index in range(FRAME_COUNT)
    ]
    assert len({frame["pose_hash"] for frame in first["frames"]}) == 1
    assert first["frames"][0]["pose_hash"] == compute_sensor_rig_pose_hash(
        first["frames"][0]["world_from_rig"]
    )
    assert math.isclose(_yaw_at(first, 0), 55.0, abs_tol=1.0e-12)


def test_rotate_in_place_uses_the_shortest_yaw_arc_without_translation() -> None:
    result = materialize_sensor_rig_trajectory(
        trajectory_id="rig_rotate_wrap_v1",
        program={
            "kind": "ROTATE_IN_PLACE",
            "position_m": [0.5, 1.5, -0.25],
            "start_yaw_deg": 170,
            "end_yaw_deg": -170,
            "yaw_interpolation": "SHORTEST_ARC",
        },
    )

    assert validate_sensor_rig_trajectory(result) == []
    assert math.isclose(_yaw_at(result, 0), 170.0, abs_tol=1.0e-12)
    assert math.isclose(abs(_yaw_at(result, 37)), 180.0, abs_tol=1.0e-12)
    assert math.isclose(_yaw_at(result, 74), -170.0, abs_tol=1.0e-12)
    assert {
        tuple(frame["world_from_rig"]["translation_m"])
        for frame in result["frames"]
    } == {(0.5, 1.5, -0.25)}
    assert len({frame["pose_hash"] for frame in result["frames"]}) == FRAME_COUNT


@pytest.mark.parametrize("kind", ["POLYLINE_MOVE", "GEODESIC_MOVE"])
def test_move_resamples_by_arc_length_and_can_follow_camera_forward(
    kind: str,
) -> None:
    program = {
        "kind": kind,
        "path_points_m": [
            [0.0, 1.6, 0.0],
            [1.0, 1.6, 0.0],
            [1.0, 1.6, -3.0],
        ],
        "position_interpolation": "ARC_LENGTH",
        "heading_policy": "PATH_TANGENT_KEEP_LAST_ON_HOLD",
        "initial_yaw_deg": 25.0,
    }
    if kind == "GEODESIC_MOVE":
        program["pathfinder_evidence_ref"] = (
            "tmp/native/sensor_rig_pathfinder_probe/receipt.json"
        )

    result = materialize_sensor_rig_trajectory(
        trajectory_id=f"rig_{kind.lower()}_v1",
        program=program,
    )

    assert validate_sensor_rig_trajectory(result) == []
    assert np.allclose(
        result["frames"][0]["world_from_rig"]["translation_m"],
        [0.0, 1.6, 0.0],
    )
    # Total length is 4 m, so Timeline midpoint is exactly 2 m along the route.
    assert np.allclose(
        result["frames"][37]["world_from_rig"]["translation_m"],
        [1.0, 1.6, -1.0],
    )
    assert np.allclose(
        result["frames"][74]["world_from_rig"]["translation_m"],
        [1.0, 1.6, -3.0],
    )
    # Habitat local forward is -Z: +X motion is yaw -90, then -Z is yaw 0.
    assert math.isclose(_yaw_at(result, 0), -90.0, abs_tol=1.0e-12)
    assert math.isclose(_yaw_at(result, 37), 0.0, abs_tol=1.0e-12)
    assert math.isclose(_yaw_at(result, 74), 0.0, abs_tol=1.0e-12)


def test_polyline_move_can_hold_an_authored_yaw_independent_of_motion() -> None:
    result = materialize_sensor_rig_trajectory(
        trajectory_id="rig_side_step_v1",
        program={
            "kind": "POLYLINE_MOVE",
            "path_points_m": [[0, 1.5, 0], [2, 1.5, 0]],
            "position_interpolation": "ARC_LENGTH",
            "heading_policy": "FIXED_YAW",
            "initial_yaw_deg": 15,
        },
    )

    assert validate_sensor_rig_trajectory(result) == []
    assert all(
        math.isclose(_yaw_at(result, frame_index), 15.0, abs_tol=1.0e-12)
        for frame_index in range(FRAME_COUNT)
    )


def test_waypoint_route_composes_rotation_motion_and_a_final_hold() -> None:
    result = materialize_sensor_rig_trajectory(
        trajectory_id="rig_waypoint_life_scene_v1",
        program={
            "kind": "WAYPOINT_ROUTE",
            "waypoints": [
                {"frame_index": 0, "position_m": [0, 1.6, 0], "yaw_deg": 0},
                {"frame_index": 10, "position_m": [0, 1.6, 0], "yaw_deg": 90},
                {"frame_index": 60, "position_m": [-2, 1.6, 0], "yaw_deg": 90},
            ],
            "interpolation": "LINEAR_POSITION_SHORTEST_YAW",
        },
    )

    assert validate_sensor_rig_trajectory(result) == []
    assert np.allclose(
        result["frames"][5]["world_from_rig"]["translation_m"],
        [0.0, 1.6, 0.0],
    )
    assert math.isclose(_yaw_at(result, 5), 45.0, abs_tol=1.0e-12)
    assert np.allclose(
        result["frames"][35]["world_from_rig"]["translation_m"],
        [-1.0, 1.6, 0.0],
    )
    for frame_index in range(60, FRAME_COUNT):
        assert np.allclose(
            result["frames"][frame_index]["world_from_rig"]["translation_m"],
            [-2.0, 1.6, 0.0],
        )
        assert math.isclose(_yaw_at(result, frame_index), 90.0, abs_tol=1.0e-12)


def test_validator_recomputes_materialized_pose_and_pose_hash() -> None:
    result = materialize_sensor_rig_trajectory(
        trajectory_id="rig_tamper_gate_v1",
        program={
            "kind": "HOLD",
            "position_m": [0, 1.6, 0],
            "yaw_deg": 0,
        },
    )
    changed_pose = deepcopy(result)
    changed_pose["frames"][12]["world_from_rig"]["translation_m"][0] = 0.01
    errors = validate_sensor_rig_trajectory(changed_pose)
    assert "frames[12].world_from_rig differs from materialized program" in errors

    changed_hash = deepcopy(result)
    changed_hash["frames"][12]["pose_hash"] = "0" * 64
    errors = validate_sensor_rig_trajectory(changed_hash)
    assert "frames[12].pose_hash does not bind world_from_rig" in errors


def test_invalid_programs_fail_closed_before_materialization() -> None:
    with pytest.raises(SensorRigTrajectoryError, match="strictly increasing"):
        materialize_sensor_rig_trajectory(
            trajectory_id="bad_waypoints",
            program={
                "kind": "WAYPOINT_ROUTE",
                "waypoints": [
                    {"frame_index": 0, "position_m": [0, 1.6, 0], "yaw_deg": 0},
                    {"frame_index": 0, "position_m": [1, 1.6, 0], "yaw_deg": 0},
                ],
                "interpolation": "LINEAR_POSITION_SHORTEST_YAW",
            },
        )
    with pytest.raises(SensorRigTrajectoryError, match="pathfinder_evidence_ref"):
        materialize_sensor_rig_trajectory(
            trajectory_id="bad_geodesic",
            program={
                "kind": "GEODESIC_MOVE",
                "path_points_m": [[0, 1.6, 0], [1, 1.6, 0]],
                "position_interpolation": "ARC_LENGTH",
                "heading_policy": "FIXED_YAW",
                "initial_yaw_deg": 0,
            },
        )
    with pytest.raises(SensorRigTrajectoryError, match="finite"):
        materialize_sensor_rig_trajectory(
            trajectory_id="bad_nonfinite",
            program={
                "kind": "HOLD",
                "position_m": [0, float("nan"), 0],
                "yaw_deg": 0,
            },
        )
    with pytest.raises(SensorRigTrajectoryError, match="zero length"):
        materialize_sensor_rig_trajectory(
            trajectory_id="bad_zero_length",
            program={
                "kind": "POLYLINE_MOVE",
                "path_points_m": [[0, 1.6, 0], [0, 1.6, 0]],
                "position_interpolation": "ARC_LENGTH",
                "heading_policy": "FIXED_YAW",
                "initial_yaw_deg": 0,
            },
        )


def test_shared_arc_length_helper_retains_existing_uniform_sampling_semantics() -> None:
    result = resample_polyline_by_arc_length(
        [[0, 0, 0], [1, 0, 0], [1, 0, 3]],
        5,
    )
    assert result.flags.c_contiguous
    assert np.allclose(
        result,
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 0, 3],
        ],
    )

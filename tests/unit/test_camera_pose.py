from __future__ import annotations

import math
from pathlib import Path

import pytest

from avengine.camera_pose import (
    CameraPoseError,
    apply_camera_listener_pose,
    normalized_yaw_degrees,
    yaw_rotation_xyzw,
)
from avengine.contracts.json_io import load_json
from avengine.m1.contracts import validate_capture_request
from avengine.m6x.apartment import listener_yaw_degrees_from_request

ROOT = Path(__file__).resolve().parents[2]
BASE_REQUEST = ROOT / "examples/m6x/fixed_apartment/m1_capture_request_review_720p.json"


def test_yaw_quaternion_uses_one_canonical_hemisphere() -> None:
    expected = [0.0, 1.0, 0.0, 0.0]
    assert yaw_rotation_xyzw(180.0) == expected
    assert yaw_rotation_xyzw(-180.0) == expected
    assert yaw_rotation_xyzw(540.0) == expected
    assert yaw_rotation_xyzw(-540.0) == expected


def test_arbitrary_camera_pose_remains_a_valid_colocated_listener_request():
    base = load_json(BASE_REQUEST)
    result = apply_camera_listener_pose(
        base,
        request_id="camera_pose_test",
        position_m=(1.25, 1.55, -2.75),
        yaw_deg=237.0,
        horizontal_fov_deg=90.0,
    )

    assert validate_capture_request(result, room_id=base["room_id"]) == []
    assert result["primary_camera_rig"]["world_from_rig"]["translation_m"] == [
        1.25,
        1.55,
        -2.75,
    ]
    assert result["listener"] == base["listener"]
    assert result["primary_camera_rig"]["shared_calibration"]["hfov_degrees"] == 90.0
    assert math.isclose(listener_yaw_degrees_from_request(result), -123.0)
    assert base["primary_camera_rig"]["world_from_rig"]["translation_m"] == [
        -0.7,
        1.471,
        0.65,
    ]


def test_camera_pose_rejects_invalid_position_yaw_and_fov():
    base = load_json(BASE_REQUEST)
    with pytest.raises(CameraPoseError, match="position"):
        apply_camera_listener_pose(
            base,
            request_id="bad",
            position_m=(0.0, float("nan"), 0.0),
            yaw_deg=0.0,
        )
    with pytest.raises(CameraPoseError, match="yaw"):
        normalized_yaw_degrees(float("inf"))
    with pytest.raises(CameraPoseError, match="FOV"):
        apply_camera_listener_pose(
            base,
            request_id="bad",
            position_m=(0.0, 1.5, 0.0),
            yaw_deg=0.0,
            horizontal_fov_deg=180.0,
        )

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from avengine.camera_pose import (
    apply_camera_listener_look_at,
    CameraPoseError,
    apply_camera_listener_pose,
    normalized_yaw_degrees,
    yaw_rotation_xyzw,
)
from avengine.contracts.json_io import load_json
from avengine.rooms.contracts import validate_capture_request
from avengine.rooms.apartment import listener_yaw_degrees_from_request

ROOT = Path(__file__).resolve().parents[2]
BASE_REQUEST = ROOT / "examples/routes/fixed_apartment/m1_capture_request_review_720p.json"


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



def test_look_at_keeps_camera_and_listener_cooriented():
    request = load_json(BASE_REQUEST)
    result = apply_camera_listener_look_at(
        request, request_id="look-at", position_m=[0, 1, 0], target_m=[1, 0, -1])
    q = result["primary_camera_rig"]["world_from_rig"]["rotation_xyzw"]
    x, y, z, w = q
    rotation = np.asarray([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    forward = rotation @ np.asarray([0, 0, -1.0])
    expected = np.asarray([1, -1, -1.0]) / np.sqrt(3)
    assert forward == pytest.approx(expected)
    assert result["listener"]["rig_from_listener"]["rotation_xyzw"] == [0, 0, 0, 1]


def test_explicit_room_id_rebinds_scene_without_mutating_base_request():
    base = load_json(BASE_REQUEST)
    result = apply_camera_listener_pose(
        base,
        request_id="kujiale-pose",
        room_id="interioragent_kujiale_0020_livingroom_491",
        position_m=(1.0, 1.47, 2.0),
        yaw_deg=15.0,
    )
    assert result["room_id"] == "interioragent_kujiale_0020_livingroom_491"
    assert base["room_id"] == "legacy_ue_apartment_0000_v1"
    assert validate_capture_request(
        result, room_id="interioragent_kujiale_0020_livingroom_491"
    ) == []
    with pytest.raises(CameraPoseError, match="room_id"):
        apply_camera_listener_pose(
            base,
            request_id="bad-room",
            room_id=" ",
            position_m=(1.0, 1.47, 2.0),
            yaw_deg=15.0,
        )

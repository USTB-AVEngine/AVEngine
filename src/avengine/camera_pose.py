"""Small, explicit camera/listener pose editing helpers.

The formal AVEngine view is a single co-located and co-oriented camera/listener
rig.  A new pose therefore belongs in the M1 capture request, upstream of
Habitat visual capture, Topdown, RIR generation and optional UE rendering.
This module changes only that pose; it does not silently move sources or reuse
audio rendered for another listener position.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from avengine.rooms.contracts import ContractError, validate_capture_request


class CameraPoseError(ValueError):
    """A requested camera/listener pose is malformed."""


def _finite_position(value: Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise CameraPoseError("camera position must contain three finite numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CameraPoseError("camera position must contain three finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise CameraPoseError("camera position must contain three finite numbers")
        result.append(number)
    return (result[0], result[1], result[2])


def normalized_yaw_degrees(value: float) -> float:
    """Return one deterministic yaw in ``[-180, 180)``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CameraPoseError("camera yaw must be finite")
    yaw = float(value)
    if not math.isfinite(yaw):
        raise CameraPoseError("camera yaw must be finite")
    return (yaw + 180.0) % 360.0 - 180.0


def yaw_rotation_xyzw(yaw_degrees: float) -> list[float]:
    """Return one canonical Habitat +Y yaw quaternion.

    Unit quaternions have a two-to-one representation.  The scalar component
    selects the positive hemisphere; at an exact half turn, where that
    component is zero, the first non-zero vector component selects it.  This
    keeps equivalent inputs such as ``180``, ``-180`` and ``540`` byte-for-byte
    identical in downstream semantic records.
    """

    half = math.radians(normalized_yaw_degrees(yaw_degrees)) * 0.5
    y = float(math.sin(half))
    w = float(math.cos(half))
    if abs(y) < 1.0e-15:
        y = 0.0
    if abs(w) < 1.0e-15:
        w = 0.0
    if w < 0.0 or (w == 0.0 and y < 0.0):
        y = -y
        w = -w
    return [0.0, y, 0.0, w]


def _look_at_quaternion_xyzw(
    position_m: Sequence[float], target_m: Sequence[float], up: Sequence[float] = (0, 1, 0),
) -> list[float]:
    """Orient camera local -Z at a target while keeping local +Y upright."""
    position = _finite_position(position_m)
    target = _finite_position(target_m)
    up_vector = _finite_position(up)
    forward = [target[i] - position[i] for i in range(3)]
    length = math.sqrt(sum(value * value for value in forward))
    up_length = math.sqrt(sum(value * value for value in up_vector))
    if length <= 1.0e-12 or up_length <= 1.0e-12:
        raise CameraPoseError("look-at target and up vector must be nondegenerate")
    forward = [value / length for value in forward]
    up_vector = [value / up_length for value in up_vector]
    right = [forward[1] * up_vector[2] - forward[2] * up_vector[1],
             forward[2] * up_vector[0] - forward[0] * up_vector[2],
             forward[0] * up_vector[1] - forward[1] * up_vector[0]]
    right_length = math.sqrt(sum(value * value for value in right))
    if right_length <= 1.0e-12:
        raise CameraPoseError("look-at direction cannot be parallel to up")
    right = [value / right_length for value in right]
    camera_up = [right[1] * forward[2] - right[2] * forward[1],
                 right[2] * forward[0] - right[0] * forward[2],
                 right[0] * forward[1] - right[1] * forward[0]]
    rotation = [[right[0], camera_up[0], -forward[0]],
                [right[1], camera_up[1], -forward[1]],
                [right[2], camera_up[2], -forward[2]]]
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        q = [(rotation[2][1] - rotation[1][2]) / scale,
             (rotation[0][2] - rotation[2][0]) / scale,
             (rotation[1][0] - rotation[0][1]) / scale, 0.25 * scale]
    else:
        index = max(range(3), key=lambda i: rotation[i][i])
        if index == 0:
            scale = math.sqrt(1 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2
            q = [0.25 * scale, (rotation[0][1] + rotation[1][0]) / scale,
                 (rotation[0][2] + rotation[2][0]) / scale,
                 (rotation[2][1] - rotation[1][2]) / scale]
        elif index == 1:
            scale = math.sqrt(1 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2
            q = [(rotation[0][1] + rotation[1][0]) / scale, 0.25 * scale,
                 (rotation[1][2] + rotation[2][1]) / scale,
                 (rotation[0][2] - rotation[2][0]) / scale]
        else:
            scale = math.sqrt(1 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2
            q = [(rotation[0][2] + rotation[2][0]) / scale,
                 (rotation[1][2] + rotation[2][1]) / scale, 0.25 * scale,
                 (rotation[1][0] - rotation[0][1]) / scale]
    norm = math.sqrt(sum(value * value for value in q))
    q = [value / norm for value in q]
    if q[3] < 0 or (q[3] == 0 and next((x for x in q[:3] if x), 0) < 0):
        q = [-value for value in q]
    return [0.0 if abs(value) < 1.0e-15 else float(value) for value in q]


def apply_camera_listener_look_at(
    request: Mapping[str, Any], *, request_id: str, position_m: Sequence[float],
    target_m: Sequence[float], up: Sequence[float] = (0, 1, 0),
    horizontal_fov_deg: float | None = None,
) -> dict[str, Any]:
    """Return a co-located camera/listener request aimed at a declared target."""
    result = apply_camera_listener_pose(
        request, request_id=request_id, position_m=position_m, yaw_deg=0,
        horizontal_fov_deg=horizontal_fov_deg)
    result["primary_camera_rig"]["world_from_rig"]["rotation_xyzw"] = (
        _look_at_quaternion_xyzw(position_m, target_m, up))
    errors = validate_capture_request(result, room_id=result.get("room_id"))
    if errors:
        raise ContractError(errors)
    return result


def apply_camera_listener_pose(
    request: Mapping[str, Any],
    *,
    request_id: str,
    position_m: Sequence[float],
    yaw_deg: float,
    horizontal_fov_deg: float | None = None,
) -> dict[str, Any]:
    """Return a validated M1 request with one new formal camera/listener pose.

    The camera and listener remain co-located/co-oriented by construction.
    Downstream geometry and acoustics must consume this returned request.
    """

    if not isinstance(request, Mapping):
        raise CameraPoseError("base capture request must be an object")
    if not isinstance(request_id, str) or not request_id.strip():
        raise CameraPoseError("request_id must be non-empty")
    position = _finite_position(position_m)
    yaw = normalized_yaw_degrees(yaw_deg)
    result = deepcopy(dict(request))
    try:
        rig = result["primary_camera_rig"]
        listener = result["listener"]
        transform = rig["world_from_rig"]
        calibration = rig["shared_calibration"]
    except (KeyError, TypeError) as error:
        raise CameraPoseError(
            "base request lacks the formal camera/listener rig"
        ) from error
    if listener.get("attached_to") != rig.get("rig_id") or listener.get(
        "rig_from_listener"
    ) != {
        "translation_m": [0, 0, 0],
        "rotation_xyzw": [0, 0, 0, 1],
    }:
        raise CameraPoseError(
            "base request must keep listener0 rigidly co-located with camera_rig_0"
        )
    transform["translation_m"] = list(position)
    transform["rotation_xyzw"] = yaw_rotation_xyzw(yaw)
    if horizontal_fov_deg is not None:
        if (
            isinstance(horizontal_fov_deg, bool)
            or not isinstance(horizontal_fov_deg, (int, float))
            or not math.isfinite(float(horizontal_fov_deg))
            or not 0.0 < float(horizontal_fov_deg) < 180.0
        ):
            raise CameraPoseError("camera horizontal FOV must lie within (0,180)")
        calibration["hfov_degrees"] = float(horizontal_fov_deg)
    result["request_id"] = request_id.strip()
    errors = validate_capture_request(result, room_id=result.get("room_id"))
    if errors:
        raise ContractError(errors)
    return result


__all__ = [
    "CameraPoseError",
    "apply_camera_listener_look_at",
    "apply_camera_listener_pose",
    "normalized_yaw_degrees",
    "yaw_rotation_xyzw",
]


def ue_yaw_to_pose_yaw_degrees(ue_yaw_degrees: float) -> float:
    """Convert a UE actor yaw into this module's ``yaw_deg`` convention.

    The two conventions differ by a quarter turn and a sign: the UE timeline
    yaw measures the camera's facing in the UE ``(x, y)`` plane, while the pose
    helpers here rotate the habitat rig whose forward is ``-Z``. Writing the
    relation once, here, keeps callers from re-deriving it — an off-by-90
    degree guess produces a request whose listener faces elsewhere than the
    rendered picture, which is exactly the silent audio/video mismatch that
    ``assert_listener_matches_capture_yaw`` now refuses at render time.
    """

    return normalized_yaw_degrees(-(float(ue_yaw_degrees) + 90.0))


def apply_camera_listener_pose_ue(
    request,
    *,
    request_id: str,
    position_m,
    ue_yaw_degrees: float,
    horizontal_fov_deg: float | None = None,
):
    """``apply_camera_listener_pose`` addressed by UE yaw instead of pose yaw."""

    return apply_camera_listener_pose(
        request,
        request_id=request_id,
        position_m=position_m,
        yaw_deg=ue_yaw_to_pose_yaw_degrees(ue_yaw_degrees),
        horizontal_fov_deg=horizontal_fov_deg,
    )

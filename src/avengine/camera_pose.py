"""Small, explicit camera/listener pose editing helpers.

The formal AVEngine view is a single co-located and co-oriented camera/listener
rig.  A new pose therefore belongs in the M1 capture request, upstream of
Habitat visual capture, Topdown, RIR generation and optional UE rendering.
This module changes only that pose; it does not silently move sources or reuse
audio rendered for another listener position.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

from avengine.m1.contracts import ContractError, validate_capture_request


class CameraPoseError(ValueError):
    """A requested camera/listener pose is malformed."""


def _finite_position(value: Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise CameraPoseError("camera position must contain three finite numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CameraPoseError(
                "camera position must contain three finite numbers"
            )
        number = float(item)
        if not math.isfinite(number):
            raise CameraPoseError(
                "camera position must contain three finite numbers"
            )
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
    """Return the Habitat +Y yaw quaternion for one horizontal camera pose."""

    half = math.radians(normalized_yaw_degrees(yaw_degrees)) * 0.5
    return [0.0, float(math.sin(half)), 0.0, float(math.cos(half))]


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
        raise CameraPoseError("base request lacks the formal camera/listener rig") from error
    if (
        listener.get("attached_to") != rig.get("rig_id")
        or listener.get("rig_from_listener")
        != {
            "translation_m": [0, 0, 0],
            "rotation_xyzw": [0, 0, 0, 1],
        }
    ):
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
    "apply_camera_listener_pose",
    "normalized_yaw_degrees",
    "yaw_rotation_xyzw",
]

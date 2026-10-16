"""Deterministic, scene-agnostic camera framing for complete world AABBs.

The solver is deliberately pure CPU and data driven.  Callers provide world
AABBs, camera candidates, pinhole calibration, and room-gate evidence.  The
module neither queries a simulator nor knows about a room, actor class,
diagnostic frame, or dataset revision.

Habitat's canonical camera convention is used: +Y is up and local -Z is
forward.  Static candidates are compatible with the shared canonical
SensorRigTrajectory contract through its existing materializer and validator.
"""

from __future__ import annotations

from copy import deepcopy
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

from avengine.camera_pose import normalized_yaw_degrees
from avengine.sensor_rig_trajectory import (
    SensorRigTrajectoryError,
    materialize_sensor_rig_trajectory,
    validate_sensor_rig_trajectory,
)

FRAMING_EVIDENCE_SCHEMA = "avengine_camera_framing_evidence_v1"
PROJECTION_EVIDENCE_SCHEMA = "avengine_camera_aabb_projection_v1"
ACTOR_GATE_ORDER = ("near_plane", "margins", "image_containment")
FRAME_GATE_ORDER = ("near_plane", "margins", "full_bbox_order")
DEFAULT_NEAR_TOLERANCE_M = 1.0e-6


class CameraFramingError(ValueError):
    """Framing input or a canonical SensorRig binding is invalid."""


def _field(value: Any, name: str, *, owner: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise CameraFramingError(f"{owner} is missing {name}")
        return value[name]
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise CameraFramingError(f"{owner} is missing {name}") from exc


def _optional_field(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CameraFramingError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CameraFramingError(f"{owner} must be a finite number")
    return 0.0 if result == 0.0 else result


def _vec3(value: Any, *, owner: str) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise CameraFramingError(f"{owner} must contain three finite numbers")
    return tuple(
        _finite(component, owner=f"{owner}[{index}]")
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _aabb(value: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    owner = "world AABB"
    if isinstance(value, Mapping):
        low_value = value.get("minimum_m", value.get("min_m"))
        high_value = value.get("maximum_m", value.get("max_m"))
    else:
        low_value = getattr(value, "minimum_m", getattr(value, "min_m", None))
        high_value = getattr(value, "maximum_m", getattr(value, "max_m", None))
    low = _vec3(low_value, owner=f"{owner}.minimum_m")
    high = _vec3(high_value, owner=f"{owner}.maximum_m")
    if any(minimum >= maximum for minimum, maximum in zip(low, high)):
        raise CameraFramingError(
            "world AABB minimum_m must be strictly smaller than maximum_m"
        )
    return low, high


def _aabb_corners(
    low: Sequence[float], high: Sequence[float]
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (x, y, z)
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    )


def _calibration(value: Any) -> dict[str, Any]:
    resolution = _field(value, "resolution_hw", owner="calibration")
    if (
        isinstance(resolution, (str, bytes))
        or not isinstance(resolution, Sequence)
        or len(resolution) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, Integral) or item <= 0
            for item in resolution
        )
    ):
        raise CameraFramingError(
            "calibration.resolution_hw must contain positive integer height and width"
        )
    height, width = (int(resolution[0]), int(resolution[1]))
    hfov = _finite(
        _field(value, "hfov_degrees", owner="calibration"),
        owner="calibration.hfov_degrees",
    )
    if not 0.0 < hfov < 180.0:
        raise CameraFramingError("calibration.hfov_degrees must lie within (0, 180)")
    near = _finite(
        _field(value, "near_m", owner="calibration"),
        owner="calibration.near_m",
    )
    if near <= 0.0:
        raise CameraFramingError("calibration.near_m must be positive")
    near_tolerance = _finite(
        _optional_field(value, "near_tolerance_m", DEFAULT_NEAR_TOLERANCE_M),
        owner="calibration.near_tolerance_m",
    )
    if near_tolerance < 0.0:
        raise CameraFramingError("calibration.near_tolerance_m must be nonnegative")
    margins_value = _field(value, "margins_px", owner="calibration")
    margins: dict[str, float] = {}
    for side in ("left", "right", "top", "bottom"):
        margin = _finite(
            _field(margins_value, side, owner="calibration.margins_px"),
            owner=f"calibration.margins_px.{side}",
        )
        if margin < 0.0:
            raise CameraFramingError(
                f"calibration.margins_px.{side} must be nonnegative"
            )
        margins[side] = margin
    if margins["left"] + margins["right"] >= width:
        raise CameraFramingError("horizontal margins leave no image area")
    if margins["top"] + margins["bottom"] >= height:
        raise CameraFramingError("vertical margins leave no image area")
    return {
        "projection": "pinhole",
        "coordinate_convention": "habitat_world_y_up_camera_forward_negative_z",
        "resolution_hw": [height, width],
        "hfov_degrees": hfov,
        "near_m": near,
        "near_tolerance_m": near_tolerance,
        "effective_near_m": near + near_tolerance,
        "margins_px": margins,
    }


def _camera_pose(value: Any) -> dict[str, Any]:
    position = _vec3(
        _field(value, "position_m", owner="camera candidate"),
        owner="camera candidate.position_m",
    )
    yaw = _finite(
        _field(value, "yaw_deg", owner="camera candidate"),
        owner="camera candidate.yaw_deg",
    )
    try:
        normalized_yaw = normalized_yaw_degrees(yaw)
    except ValueError as exc:
        raise CameraFramingError(f"camera candidate yaw is invalid: {exc}") from exc
    return {"position_m": list(position), "yaw_deg": normalized_yaw}


def _camera_local(
    point: Sequence[float], camera_pose: Mapping[str, Any]
) -> tuple[float, float, float]:
    position = camera_pose["position_m"]
    dx = float(point[0]) - float(position[0])
    dy = float(point[1]) - float(position[1])
    dz = float(point[2]) - float(position[2])
    yaw = math.radians(float(camera_pose["yaw_deg"]))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    # Inverse +Y yaw.  At yaw=0 local -Z is forward; positive yaw turns
    # forward toward world -X, matching the canonical Habitat rig contract.
    return (
        cosine * dx - sine * dz,
        dy,
        sine * dx + cosine * dz,
    )


def _gate(status: str, **evidence: Any) -> dict[str, Any]:
    return {"status": status, **evidence}


def project_world_aabb(
    *,
    aabb_world_m: Any,
    camera_pose: Any,
    calibration: Any,
) -> dict[str, Any]:
    """Project all eight AABB corners and evaluate ordered hard gates.

    Near-plane failure stops pixel projection, so a behind-camera corner can
    never create a misleading finite box.  When projection is safe, margin
    containment is evaluated before raw full-image containment and both facts
    remain independently visible in the evidence.
    """

    low, high = _aabb(aabb_world_m)
    pose = _camera_pose(camera_pose)
    normalized_calibration = _calibration(calibration)
    corners = _aabb_corners(low, high)
    local = tuple(_camera_local(corner, pose) for corner in corners)
    depths = tuple(-point[2] for point in local)
    near = float(normalized_calibration["near_m"])
    near_tolerance = float(normalized_calibration["near_tolerance_m"])
    effective_near = float(normalized_calibration["effective_near_m"])
    near_pass = all(depth >= effective_near for depth in depths)
    near_gate = _gate(
        "pass" if near_pass else "fail",
        configured_near_m=near,
        tolerance_m=near_tolerance,
        effective_threshold_m=effective_near,
        minimum_corner_depth_m=min(depths),
        maximum_corner_depth_m=max(depths),
        passing_corner_count=sum(depth >= effective_near for depth in depths),
        required_corner_count=len(corners),
    )

    result: dict[str, Any] = {
        "schema": PROJECTION_EVIDENCE_SCHEMA,
        "gate_order": list(ACTOR_GATE_ORDER),
        "camera_pose": pose,
        "calibration": normalized_calibration,
        "aabb_world_m": {"minimum_m": list(low), "maximum_m": list(high)},
        "corner_count": len(corners),
        "world_corners_m": [list(corner) for corner in corners],
        "camera_corner_depths_m": list(depths),
        "projected_corners_px": None,
        "projected_bbox_px": None,
        "gates": {
            "near_plane": near_gate,
            "margins": _gate("not_evaluated", reason="near_plane_failed"),
            "image_containment": _gate("not_evaluated", reason="near_plane_failed"),
        },
        "hard_gates_pass": False,
    }
    if not near_pass:
        return result

    height, width = normalized_calibration["resolution_hw"]
    tan_half_h = math.tan(
        math.radians(float(normalized_calibration["hfov_degrees"])) * 0.5
    )
    tan_half_v = tan_half_h * (float(height) / float(width))
    projected: list[list[float]] = []
    for x, y, z in local:
        depth = -z
        x_ndc = x / (depth * tan_half_h)
        y_ndc = y / (depth * tan_half_v)
        projected.append(
            [
                (x_ndc + 1.0) * 0.5 * float(width),
                (1.0 - y_ndc) * 0.5 * float(height),
            ]
        )
    xs = [point[0] for point in projected]
    ys = [point[1] for point in projected]
    bbox = {
        "left": min(xs),
        "top": min(ys),
        "right": max(xs),
        "bottom": max(ys),
    }
    margins = normalized_calibration["margins_px"]
    margin_checks = {
        "left": bbox["left"] >= margins["left"],
        "right": bbox["right"] <= float(width) - margins["right"],
        "top": bbox["top"] >= margins["top"],
        "bottom": bbox["bottom"] <= float(height) - margins["bottom"],
    }
    image_checks = {
        "left": bbox["left"] >= 0.0,
        "right": bbox["right"] <= float(width),
        "top": bbox["top"] >= 0.0,
        "bottom": bbox["bottom"] <= float(height),
    }
    margins_pass = all(margin_checks.values())
    image_pass = all(image_checks.values())
    result["projected_corners_px"] = projected
    result["projected_bbox_px"] = bbox
    result["gates"]["margins"] = _gate(
        "pass" if margins_pass else "fail",
        checks=margin_checks,
        required_inset_px=deepcopy(margins),
    )
    result["gates"]["image_containment"] = _gate(
        "pass" if image_pass else "fail",
        checks=image_checks,
        image_bounds_px={
            "left": 0.0,
            "top": 0.0,
            "right": float(width),
            "bottom": float(height),
        },
    )
    result["hard_gates_pass"] = near_pass and margins_pass and image_pass
    return result


def _room_gate(value: Any, *, candidate_id: str) -> dict[str, Any]:
    owner = f"candidate {candidate_id!r} room_gate"
    evidence = _structured_authority(value, owner=owner)
    hard_gates = evidence.get("hard_gates")
    if not isinstance(hard_gates, Mapping) or not hard_gates:
        raise CameraFramingError(f"{owner}.hard_gates must be non-empty")
    for gate_name, gate in hard_gates.items():
        if not isinstance(gate_name, str) or not gate_name:
            raise CameraFramingError(f"{owner}.hard_gates names must be non-empty")
        if not isinstance(gate, Mapping):
            raise CameraFramingError(f"{owner}.hard_gates.{gate_name} must be evidence")
        if gate.get("passed") is not True and gate.get("status") not in {
            "pass",
            "qualified",
        }:
            raise CameraFramingError(
                f"{owner}.hard_gates.{gate_name} must explicitly pass"
            )
    return evidence


def _candidate(value: Any) -> dict[str, Any]:
    candidate_id = _field(value, "candidate_id", owner="camera candidate")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise CameraFramingError("camera candidate_id must be non-empty")
    candidate_id = candidate_id.strip()
    priority = _finite(
        _optional_field(value, "priority", 0.0),
        owner=f"candidate {candidate_id!r} priority",
    )
    room_evidence = _room_gate(
        _field(value, "room_gate", owner=f"candidate {candidate_id!r}"),
        candidate_id=candidate_id,
    )
    return {
        "candidate_id": candidate_id,
        "priority": priority,
        "camera_pose": _camera_pose(value),
        "room_gate_evidence": room_evidence,
        "room_gate_pass": True,
    }


def _structured_authority(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise CameraFramingError(f"{owner} must be a non-empty evidence object")
    evidence = deepcopy(dict(value))
    passed = evidence.get("passed")
    status = evidence.get("status")
    if passed is not True and status not in {"pass", "qualified"}:
        raise CameraFramingError(f"{owner} must carry passing authority evidence")
    authority_id = evidence.get("authority_id")
    if not isinstance(authority_id, str) or not authority_id.strip():
        raise CameraFramingError(f"{owner}.authority_id must be non-empty")
    return evidence


def _actor_bounds(value: Any, *, actor_id: str, frame_index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CameraFramingError(
            f"actor {actor_id!r} frame {frame_index} bounds must be an object"
        )
    owner = f"actor {actor_id!r} frame {frame_index}"
    low, high = _aabb(value)
    bounds_authority = _structured_authority(
        _field(value, "bounds_authority", owner=owner),
        owner=f"{owner} bounds_authority",
    )
    coordinate_chain = _structured_authority(
        _field(value, "coordinate_chain", owner=owner),
        owner=f"{owner} coordinate_chain",
    )
    action_coverage = _structured_authority(
        _field(value, "action_coverage", owner=owner),
        owner=f"{owner} action_coverage",
    )
    actual_action_id = _field(value, "action_id", owner=owner)
    if not isinstance(actual_action_id, str) or not actual_action_id.strip():
        raise CameraFramingError(f"{owner} action_id must be non-empty")
    coverage_action_id = action_coverage.get("action_id")
    if not isinstance(coverage_action_id, str) or not coverage_action_id.strip():
        raise CameraFramingError(f"{owner} action_coverage.action_id must be non-empty")
    if actual_action_id != coverage_action_id:
        raise CameraFramingError(
            f"{owner} action_id differs from action_coverage.action_id"
        )
    for field_name in ("asset_id", "revision_id", "action_scope"):
        field_value = bounds_authority.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise CameraFramingError(
                f"{owner} bounds_authority.{field_name} must be non-empty"
            )
    if bounds_authority["action_scope"] != actual_action_id:
        raise CameraFramingError(
            f"{owner} bounds_authority.action_scope differs from action_id"
        )
    from_frame = coordinate_chain.get("from_frame")
    to_frame = coordinate_chain.get("to_frame")
    operations = coordinate_chain.get("operations")
    if not isinstance(from_frame, str) or not from_frame.strip():
        raise CameraFramingError(
            f"{owner} coordinate_chain.from_frame must be non-empty"
        )
    if to_frame != "avengine_world_right_handed_y_up_m":
        raise CameraFramingError(
            f"{owner} coordinate_chain.to_frame must be canonical AVEngine world"
        )
    if (
        isinstance(operations, (str, bytes))
        or not isinstance(operations, Sequence)
        or not operations
        or any(
            not isinstance(operation, str) or not operation for operation in operations
        )
    ):
        raise CameraFramingError(
            f"{owner} coordinate_chain.operations must be non-empty strings"
        )
    covered = action_coverage.get(
        "covered_frame_indices", action_coverage.get("frame_indices")
    )
    if (
        isinstance(covered, (str, bytes))
        or not isinstance(covered, Sequence)
        or not covered
        or any(
            isinstance(item, bool) or not isinstance(item, Integral) for item in covered
        )
    ):
        raise CameraFramingError(
            f"{owner} action_coverage must list covered_frame_indices"
        )
    covered_indices = [int(item) for item in covered]
    if len(covered_indices) != len(set(covered_indices)):
        raise CameraFramingError(
            f"{owner} action_coverage frame indices must be unique"
        )
    if frame_index not in covered_indices:
        raise CameraFramingError(f"{owner} is not covered by action_coverage")
    return {
        "minimum_m": list(low),
        "maximum_m": list(high),
        "action_id": actual_action_id,
        "bounds_authority": bounds_authority,
        "coordinate_chain": coordinate_chain,
        "action_coverage": {
            **action_coverage,
            "covered_frame_indices": covered_indices,
        },
    }


def _frames(values: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not values
    ):
        raise CameraFramingError("frames must be a non-empty sequence")
    normalized: list[dict[str, Any]] = []
    for ordinal, value in enumerate(values):
        frame_index = _field(value, "frame_index", owner=f"frame[{ordinal}]")
        if isinstance(frame_index, bool) or not isinstance(frame_index, Integral):
            raise CameraFramingError(f"frame[{ordinal}].frame_index must be an integer")
        aabbs = _optional_field(value, "actor_aabbs", None)
        if aabbs is None:
            aabbs = _field(value, "aabbs", owner=f"frame[{ordinal}]")
        if not isinstance(aabbs, Mapping) or not aabbs:
            raise CameraFramingError(
                f"frame[{ordinal}] actor_aabbs must be a non-empty mapping"
            )
        actor_aabbs: dict[str, dict[str, Any]] = {}
        for actor_id, aabb_value in aabbs.items():
            if not isinstance(actor_id, str) or not actor_id:
                raise CameraFramingError("actor IDs must be non-empty strings")
            actor_aabbs[actor_id] = _actor_bounds(
                aabb_value,
                actor_id=actor_id,
                frame_index=int(frame_index),
            )
        normalized.append({"frame_index": int(frame_index), "actor_aabbs": actor_aabbs})
    indices = [frame["frame_index"] for frame in normalized]
    if len(indices) != len(set(indices)):
        raise CameraFramingError("frame_index values must be unique")
    normalized.sort(key=lambda frame: frame["frame_index"])
    actor_ids = sorted(normalized[0]["actor_aabbs"])
    if any(sorted(frame["actor_aabbs"]) != actor_ids for frame in normalized):
        raise CameraFramingError("every frame must contain the same actor IDs")
    return normalized, actor_ids


def _ordered_actor_ids(value: Any, actor_ids: Sequence[str]) -> list[str]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise CameraFramingError(
            "ordered_actor_ids must be a sequence of non-empty actor IDs"
        )
    result = list(value)
    if len(result) < 2 or len(result) != len(set(result)):
        raise CameraFramingError(
            "ordered_actor_ids must contain at least two unique actor IDs"
        )
    if set(result) != set(actor_ids):
        raise CameraFramingError(
            "ordered_actor_ids must exactly close the frame actor IDs"
        )
    return result


def evaluate_static_camera_candidate(
    *,
    frames: Any,
    candidate: Any,
    calibration: Any,
    ordered_actor_ids: Any,
    minimum_order_gap_px: Any,
) -> dict[str, Any]:
    """Evaluate any non-empty frame subset without claiming SensorRig closure."""

    normalized_frames, actor_ids = _frames(frames)
    normalized_calibration = _calibration(calibration)
    normalized_candidate = _candidate(candidate)
    ordered = _ordered_actor_ids(ordered_actor_ids, actor_ids)
    gap = _finite(minimum_order_gap_px, owner="minimum_order_gap_px")
    if gap < 0.0:
        raise CameraFramingError("minimum_order_gap_px must be nonnegative")
    evaluation: dict[str, Any] = {
        "candidate_id": normalized_candidate["candidate_id"],
        "priority": normalized_candidate["priority"],
        "camera_pose": deepcopy(normalized_candidate["camera_pose"]),
        "room_gate": {
            "status": ("pass" if normalized_candidate["room_gate_pass"] else "fail"),
            "evidence": deepcopy(normalized_candidate["room_gate_evidence"]),
        },
        "ordered_actor_ids": ordered,
        "minimum_order_gap_px": gap,
        "frame_evaluations": [],
        "all_frames_hard_gates_pass": False,
        "selectable": False,
    }
    if not normalized_candidate["room_gate_pass"]:
        return evaluation
    all_frames_pass = True
    for frame in normalized_frames:
        actors: list[dict[str, Any]] = []
        projection_by_actor: dict[str, dict[str, Any]] = {}
        actors_pass = True
        for actor_id in actor_ids:
            projection = project_world_aabb(
                aabb_world_m=frame["actor_aabbs"][actor_id],
                camera_pose=normalized_candidate["camera_pose"],
                calibration=normalized_calibration,
            )
            projection_by_actor[actor_id] = projection
            actor_pass = bool(projection["hard_gates_pass"])
            actors_pass = actors_pass and actor_pass
            actors.append(
                {
                    "actor_id": actor_id,
                    "action_id": frame["actor_aabbs"][actor_id]["action_id"],
                    "bounds_authority": deepcopy(
                        frame["actor_aabbs"][actor_id]["bounds_authority"]
                    ),
                    "coordinate_chain": deepcopy(
                        frame["actor_aabbs"][actor_id]["coordinate_chain"]
                    ),
                    "action_coverage": deepcopy(
                        frame["actor_aabbs"][actor_id]["action_coverage"]
                    ),
                    "hard_gates_pass": actor_pass,
                    "projection": projection,
                }
            )
        pair_checks: list[dict[str, Any]] = []
        order_pass = actors_pass
        for left_actor_id, right_actor_id in zip(ordered, ordered[1:]):
            left_bbox = projection_by_actor[left_actor_id]["projected_bbox_px"]
            right_bbox = projection_by_actor[right_actor_id]["projected_bbox_px"]
            passed = bool(
                left_bbox is not None
                and right_bbox is not None
                and float(left_bbox["right"]) + gap <= float(right_bbox["left"])
            )
            order_pass = order_pass and passed
            pair_checks.append(
                {
                    "left_actor_id": left_actor_id,
                    "right_actor_id": right_actor_id,
                    "left_bbox_right_px": (
                        left_bbox["right"] if left_bbox is not None else None
                    ),
                    "right_bbox_left_px": (
                        right_bbox["left"] if right_bbox is not None else None
                    ),
                    "minimum_gap_px": gap,
                    "status": "pass" if passed else "fail",
                }
            )
        frame_pass = actors_pass and order_pass
        all_frames_pass = all_frames_pass and frame_pass
        evaluation["frame_evaluations"].append(
            {
                "frame_index": frame["frame_index"],
                "actors": actors,
                "full_bbox_order": {
                    "status": "pass" if order_pass else "fail",
                    "ordered_actor_ids": ordered,
                    "pair_checks": pair_checks,
                },
                "hard_gates_pass": frame_pass,
            }
        )
    evaluation["all_frames_hard_gates_pass"] = all_frames_pass
    evaluation["selectable"] = all_frames_pass
    return evaluation


def _static_sensor_rig(
    *,
    trajectory_id: str,
    camera_pose: Mapping[str, Any],
    sensor_rig_trajectory: Any | None,
) -> tuple[dict[str, Any], str]:
    if not isinstance(trajectory_id, str) or not trajectory_id.strip():
        raise CameraFramingError("trajectory_id must be non-empty")
    try:
        expected = materialize_sensor_rig_trajectory(
            trajectory_id=trajectory_id.strip(),
            program={
                "kind": "HOLD",
                "position_m": list(camera_pose["position_m"]),
                "yaw_deg": float(camera_pose["yaw_deg"]),
            },
        )
    except SensorRigTrajectoryError as exc:
        raise CameraFramingError(
            f"cannot materialize canonical SensorRig: {exc}"
        ) from exc
    if sensor_rig_trajectory is None:
        return expected, "materialized_hold"
    errors = validate_sensor_rig_trajectory(sensor_rig_trajectory)
    if errors:
        raise CameraFramingError(
            "provided SensorRigTrajectory is invalid: " + "; ".join(errors)
        )
    provided = deepcopy(dict(sensor_rig_trajectory))
    if provided.get("trajectory_id") != trajectory_id.strip():
        raise CameraFramingError("provided SensorRigTrajectory trajectory_id differs")
    expected_transforms = [frame["world_from_rig"] for frame in expected["frames"]]
    provided_transforms = [frame["world_from_rig"] for frame in provided["frames"]]
    if provided_transforms != expected_transforms:
        raise CameraFramingError(
            "provided SensorRigTrajectory is not the selected static camera pose"
        )
    return provided, "provided_validated"


def solve_static_camera_candidates(
    *,
    frames: Any,
    candidates: Any,
    calibration: Any,
    trajectory_id: str,
    ordered_actor_ids: Any,
    minimum_order_gap_px: Any,
    sensor_rig_trajectory: Any | None = None,
) -> dict[str, Any]:
    """Choose the first deterministic static candidate passing every hard gate.

    Candidate input order has no authority.  Candidates are sorted by numeric
    priority and then by candidate ID.  A candidate with a failed room gate is
    retained in evidence but never projected or selected.  A projected
    candidate is selectable only when every actor AABB in every supplied frame
    passes all ordered framing hard gates.
    """

    normalized_frames, actor_ids = _frames(frames)
    normalized_calibration = _calibration(calibration)
    ordered = _ordered_actor_ids(ordered_actor_ids, actor_ids)
    gap = _finite(minimum_order_gap_px, owner="minimum_order_gap_px")
    if gap < 0.0:
        raise CameraFramingError("minimum_order_gap_px must be nonnegative")
    if (
        isinstance(candidates, (str, bytes))
        or not isinstance(candidates, Sequence)
        or not candidates
    ):
        raise CameraFramingError("candidates must be a non-empty sequence")
    normalized_candidates = [_candidate(value) for value in candidates]
    candidate_ids = [value["candidate_id"] for value in normalized_candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise CameraFramingError("candidate_id values must be unique")
    normalized_candidates.sort(
        key=lambda value: (value["priority"], value["candidate_id"])
    )
    # Ask the existing canonical materializer for its clock; this module does
    # not duplicate or hard-code the formal frame count.
    clock_probe, _source = _static_sensor_rig(
        trajectory_id=trajectory_id,
        camera_pose=normalized_candidates[0]["camera_pose"],
        sensor_rig_trajectory=None,
    )
    input_frame_indices = [frame["frame_index"] for frame in normalized_frames]
    canonical_frame_indices = [frame["frame_index"] for frame in clock_probe["frames"]]
    if input_frame_indices != canonical_frame_indices:
        raise CameraFramingError(
            "framing frame_indices must exactly match canonical SensorRig frames"
        )

    evaluations: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for candidate in normalized_candidates:
        evaluation = evaluate_static_camera_candidate(
            frames=normalized_frames,
            candidate={
                "candidate_id": candidate["candidate_id"],
                "priority": candidate["priority"],
                **candidate["camera_pose"],
                "room_gate": candidate["room_gate_evidence"],
            },
            calibration=normalized_calibration,
            ordered_actor_ids=ordered,
            minimum_order_gap_px=gap,
        )
        evaluations.append(evaluation)
        if selected is None and evaluation["selectable"]:
            selected = evaluation

    result: dict[str, Any] = {
        "schema": FRAMING_EVIDENCE_SCHEMA,
        "status": (
            "pass_cpu_declared_bounds_framing"
            if selected is not None
            else "no_qualifying_cpu_declared_bounds_candidate"
        ),
        "solver": "deterministic_static_candidate_v1",
        "claim_boundary": {
            "authority": "immutable_declared_world_aabbs_and_pinhole_projection",
            "permitted_claim": "static_camera_candidate_planning",
            "native_pixels_are_validated": False,
            "release_or_dataset_qualification_permitted": False,
        },
        "native_pixel_validation_status": "pending",
        "qualification_claim": False,
        "formal_episode_count": 0,
        "candidate_order": "ascending_priority_then_candidate_id",
        "hard_gate_order": list(FRAME_GATE_ORDER),
        "calibration": normalized_calibration,
        "frame_count": len(normalized_frames),
        "frame_indices": [frame["frame_index"] for frame in normalized_frames],
        "actor_ids": actor_ids,
        "ordered_actor_ids": ordered,
        "minimum_order_gap_px": gap,
        "candidate_count": len(evaluations),
        "candidate_evaluations": evaluations,
        "selected_candidate_id": (
            selected["candidate_id"] if selected is not None else None
        ),
        "selected_camera_pose": (
            deepcopy(selected["camera_pose"]) if selected is not None else None
        ),
        "sensor_rig_binding": None,
    }
    if selected is not None:
        rig, source = _static_sensor_rig(
            trajectory_id=trajectory_id,
            camera_pose=selected["camera_pose"],
            sensor_rig_trajectory=sensor_rig_trajectory,
        )
        rig_frame_indices = [frame["frame_index"] for frame in rig["frames"]]
        if input_frame_indices != rig_frame_indices:
            raise CameraFramingError(
                "framing frame_indices must exactly match canonical SensorRig frames"
            )
        result["sensor_rig_binding"] = {
            "source": source,
            "validation_errors": [],
            "trajectory": rig,
        }
    elif sensor_rig_trajectory is not None:
        raise CameraFramingError(
            "provided SensorRigTrajectory cannot bind without a selected candidate"
        )
    return result


__all__ = [
    "ACTOR_GATE_ORDER",
    "DEFAULT_NEAR_TOLERANCE_M",
    "FRAME_GATE_ORDER",
    "FRAMING_EVIDENCE_SCHEMA",
    "PROJECTION_EVIDENCE_SCHEMA",
    "CameraFramingError",
    "evaluate_static_camera_candidate",
    "project_world_aabb",
    "solve_static_camera_candidates",
]

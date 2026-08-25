"""Versioned yaw-only trajectories for the formal camera/listener rig.

The trajectory is a pure AVEngine data contract.  It does not query Habitat,
advance a simulator, render RIRs or move sources.  A ``GEODESIC_MOVE`` embeds
an already-qualified Pathfinder polyline and its evidence reference; this
module only performs deterministic arc-length sampling on Timeline v2's
visual-frame clock.
"""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator
import numpy as np

from avengine.camera_pose import normalized_yaw_degrees, yaw_rotation_xyzw
from avengine.contracts.json_io import canonical_json_sha256, load_json
from avengine.contracts.transforms import normalized_quaternion_xyzw
from avengine.capture.orientation import habitat_yaw_degrees_from_xyzw
from avengine.routes.trajectory import (
    M6XTrajectoryError,
    materialize_route,
    resample_polyline_by_arc_length,
)


SENSOR_RIG_TRAJECTORY_SCHEMA = "avengine_sensor_rig_trajectory_v1"
SENSOR_RIG_POSE_SCHEMA = "avengine_sensor_rig_pose_v1"
POSE_HASH_ALGORITHM = "avengine_sensor_rig_pose_hash_v1"
SCHEMA_FILENAME = "sensor_rig_trajectory_v1.schema.json"

RIG_ID = "camera_rig_0"
FORMAL_VIEW_ID = "view0"
LISTENER_ID = "listener0"
COORDINATE_FRAME = "avengine_world_right_handed_y_up_m"
CAMERA_LISTENER_COUPLING = "rigid_colocated_cooriented"
POSE_MODEL = "yaw_only_about_world_positive_y"

TIME_BASE_HZ = 48_000
DURATION_TICKS = 240_000
FRAME_RATE_HZ = 15
FRAME_COUNT = 75
TICKS_PER_FRAME = 3_200

IDENTITY_TRANSFORM = {
    "translation_m": [0.0, 0.0, 0.0],
    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
}

_MOVE_KINDS = {"POLYLINE_MOVE", "GEODESIC_MOVE"}
_HEADING_POLICIES = {"FIXED_YAW", "PATH_TANGENT_KEEP_LAST_ON_HOLD"}


class SensorRigTrajectoryError(ValueError):
    """A sensor-rig trajectory contract cannot be materialized or validated."""

    def __init__(self, errors: str | Iterable[str]):
        self.errors = (errors,) if isinstance(errors, str) else tuple(errors)
        super().__init__("; ".join(self.errors))


def sensor_rig_trajectory_schema_path() -> Path:
    """Return the source-tree or installed SensorRigTrajectory v1 schema."""

    source = Path(__file__).resolve().parents[2] / "schemas" / SCHEMA_FILENAME
    installed = Path(sys.prefix) / "share" / "avengine" / "schemas" / SCHEMA_FILENAME
    path = source if source.is_file() else installed
    if not path.is_file():
        raise FileNotFoundError(
            f"AVEngine sensor-rig trajectory schema is unavailable: {SCHEMA_FILENAME}"
        )
    return path


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise SensorRigTrajectoryError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SensorRigTrajectoryError(f"{owner} must be a finite number")
    return 0.0 if result == 0.0 else result


def _finite_vec3(value: Any, *, owner: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SensorRigTrajectoryError(
            f"{owner} must contain exactly three finite numbers"
        )
    if len(value) != 3:
        raise SensorRigTrajectoryError(
            f"{owner} must contain exactly three finite numbers"
        )
    return [
        _finite_number(component, owner=f"{owner}[{index}]")
        for index, component in enumerate(value)
    ]


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, owner: str
) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise SensorRigTrajectoryError(
            f"{owner} fields are invalid: {', '.join(details)}"
        )


def _normalize_program(program: Any) -> dict[str, Any]:
    if not isinstance(program, Mapping):
        raise SensorRigTrajectoryError("program must be an object")
    kind = program.get("kind")
    if kind == "HOLD":
        _require_exact_keys(
            program,
            {"kind", "position_m", "yaw_deg"},
            owner="HOLD program",
        )
        return {
            "kind": kind,
            "position_m": _finite_vec3(
                program["position_m"], owner="HOLD position_m"
            ),
            "yaw_deg": _finite_number(program["yaw_deg"], owner="HOLD yaw_deg"),
        }
    if kind == "ROTATE_IN_PLACE":
        _require_exact_keys(
            program,
            {
                "kind",
                "position_m",
                "start_yaw_deg",
                "end_yaw_deg",
                "yaw_interpolation",
            },
            owner="ROTATE_IN_PLACE program",
        )
        if program["yaw_interpolation"] != "SHORTEST_ARC":
            raise SensorRigTrajectoryError(
                "ROTATE_IN_PLACE yaw_interpolation must be SHORTEST_ARC"
            )
        return {
            "kind": kind,
            "position_m": _finite_vec3(
                program["position_m"], owner="ROTATE_IN_PLACE position_m"
            ),
            "start_yaw_deg": _finite_number(
                program["start_yaw_deg"], owner="ROTATE_IN_PLACE start_yaw_deg"
            ),
            "end_yaw_deg": _finite_number(
                program["end_yaw_deg"], owner="ROTATE_IN_PLACE end_yaw_deg"
            ),
            "yaw_interpolation": "SHORTEST_ARC",
        }
    if kind in _MOVE_KINDS:
        expected = {
            "kind",
            "path_points_m",
            "position_interpolation",
            "heading_policy",
            "initial_yaw_deg",
        }
        if kind == "GEODESIC_MOVE":
            expected.add("pathfinder_evidence_ref")
        _require_exact_keys(program, expected, owner=f"{kind} program")
        points_value = program["path_points_m"]
        if (
            isinstance(points_value, (str, bytes))
            or not isinstance(points_value, Sequence)
            or len(points_value) < 2
        ):
            raise SensorRigTrajectoryError(
                f"{kind} path_points_m must contain at least two finite points"
            )
        if program["position_interpolation"] != "ARC_LENGTH":
            raise SensorRigTrajectoryError(
                f"{kind} position_interpolation must be ARC_LENGTH"
            )
        heading_policy = program["heading_policy"]
        if heading_policy not in _HEADING_POLICIES:
            raise SensorRigTrajectoryError(
                f"{kind} heading_policy is unsupported: {heading_policy!r}"
            )
        result = {
            "kind": kind,
            "path_points_m": [
                _finite_vec3(point, owner=f"{kind} path_points_m[{index}]")
                for index, point in enumerate(points_value)
            ],
            "position_interpolation": "ARC_LENGTH",
            "heading_policy": heading_policy,
            "initial_yaw_deg": _finite_number(
                program["initial_yaw_deg"], owner=f"{kind} initial_yaw_deg"
            ),
        }
        if kind == "GEODESIC_MOVE":
            evidence_ref = program["pathfinder_evidence_ref"]
            if not isinstance(evidence_ref, str) or not evidence_ref.strip():
                raise SensorRigTrajectoryError(
                    "GEODESIC_MOVE pathfinder_evidence_ref must be non-empty"
                )
            result["pathfinder_evidence_ref"] = evidence_ref
        return result
    if kind == "WAYPOINT_ROUTE":
        _require_exact_keys(
            program,
            {"kind", "waypoints", "interpolation"},
            owner="WAYPOINT_ROUTE program",
        )
        if program["interpolation"] != "LINEAR_POSITION_SHORTEST_YAW":
            raise SensorRigTrajectoryError(
                "WAYPOINT_ROUTE interpolation must be "
                "LINEAR_POSITION_SHORTEST_YAW"
            )
        waypoints_value = program["waypoints"]
        if (
            isinstance(waypoints_value, (str, bytes))
            or not isinstance(waypoints_value, Sequence)
            or not waypoints_value
            or len(waypoints_value) > FRAME_COUNT
        ):
            raise SensorRigTrajectoryError(
                "WAYPOINT_ROUTE must contain between 1 and 75 waypoints"
            )
        waypoints: list[dict[str, Any]] = []
        for ordinal, waypoint in enumerate(waypoints_value):
            if not isinstance(waypoint, Mapping):
                raise SensorRigTrajectoryError(
                    f"WAYPOINT_ROUTE waypoint[{ordinal}] must be an object"
                )
            _require_exact_keys(
                waypoint,
                {"frame_index", "position_m", "yaw_deg"},
                owner=f"WAYPOINT_ROUTE waypoint[{ordinal}]",
            )
            frame_index = waypoint["frame_index"]
            if (
                isinstance(frame_index, (bool, np.bool_))
                or not isinstance(frame_index, int)
                or not 0 <= frame_index < FRAME_COUNT
            ):
                raise SensorRigTrajectoryError(
                    f"WAYPOINT_ROUTE waypoint[{ordinal}].frame_index "
                    "must lie in [0,74]"
                )
            waypoints.append(
                {
                    "frame_index": frame_index,
                    "position_m": _finite_vec3(
                        waypoint["position_m"],
                        owner=f"WAYPOINT_ROUTE waypoint[{ordinal}].position_m",
                    ),
                    "yaw_deg": _finite_number(
                        waypoint["yaw_deg"],
                        owner=f"WAYPOINT_ROUTE waypoint[{ordinal}].yaw_deg",
                    ),
                }
            )
        indices = [waypoint["frame_index"] for waypoint in waypoints]
        if indices[0] != 0 or indices != sorted(set(indices)):
            raise SensorRigTrajectoryError(
                "WAYPOINT_ROUTE frame indices must start at 0 and be strictly "
                "increasing"
            )
        return {
            "kind": kind,
            "waypoints": waypoints,
            "interpolation": "LINEAR_POSITION_SHORTEST_YAW",
        }
    raise SensorRigTrajectoryError(f"unsupported sensor-rig program kind: {kind!r}")


def _route_positions(
    points: Sequence[Sequence[float]],
    frame_indices: Sequence[int],
    *,
    interpolation: str,
) -> np.ndarray:
    anchor_ids = [f"rig_waypoint_{index:02d}" for index in range(len(points))]
    anchor_library = {
        "anchors": [
            {"anchor_id": anchor_id, "position_m": list(point)}
            for anchor_id, point in zip(anchor_ids, points)
        ]
    }
    route = {
        "anchor_ids": anchor_ids,
        "anchor_frame_indices": list(frame_indices),
        "interpolation": interpolation,
    }
    try:
        return materialize_route(
            route,
            anchor_library=anchor_library,
            frame_count=FRAME_COUNT,
        )
    except M6XTrajectoryError as exc:
        raise SensorRigTrajectoryError(str(exc)) from exc


def _shortest_arc_yaws(
    start_yaw_deg: float, end_yaw_deg: float, sample_count: int
) -> np.ndarray:
    start = normalized_yaw_degrees(start_yaw_deg)
    delta = normalized_yaw_degrees(end_yaw_deg - start_yaw_deg)
    unwrapped = np.linspace(start, start + delta, sample_count)
    return np.asarray(
        [normalized_yaw_degrees(float(value)) for value in unwrapped],
        dtype=np.float64,
    )


def _waypoint_yaws(waypoints: Sequence[Mapping[str, Any]]) -> np.ndarray:
    result = np.empty(FRAME_COUNT, dtype=np.float64)
    if len(waypoints) == 1:
        result[:] = normalized_yaw_degrees(float(waypoints[0]["yaw_deg"]))
        return result
    for segment, (start, end) in enumerate(zip(waypoints, waypoints[1:])):
        start_frame = int(start["frame_index"])
        end_frame = int(end["frame_index"])
        segment_yaws = _shortest_arc_yaws(
            float(start["yaw_deg"]),
            float(end["yaw_deg"]),
            end_frame - start_frame + 1,
        )
        result[start_frame : end_frame + 1] = segment_yaws
        if segment > 0:
            # Both adjacent segments write the same authored waypoint.  This
            # assignment documents that the later segment is authoritative.
            result[start_frame] = segment_yaws[0]
    result[int(waypoints[-1]["frame_index"]) :] = normalized_yaw_degrees(
        float(waypoints[-1]["yaw_deg"])
    )
    return result


def _path_tangent_yaws(
    positions: np.ndarray, *, initial_yaw_deg: float
) -> np.ndarray:
    result = np.empty(FRAME_COUNT, dtype=np.float64)
    retained_yaw = normalized_yaw_degrees(initial_yaw_deg)
    for frame_index in range(FRAME_COUNT):
        if frame_index + 1 < FRAME_COUNT:
            delta = positions[frame_index + 1] - positions[frame_index]
            dx = float(delta[0])
            dz = float(delta[2])
            if math.hypot(dx, dz) > 1.0e-12:
                # Habitat camera/listener local forward is -Z.  Solving
                # [-sin(yaw), -cos(yaw)] = normalized [dx, dz] gives this yaw.
                retained_yaw = normalized_yaw_degrees(
                    math.degrees(math.atan2(-dx, -dz))
                )
        result[frame_index] = retained_yaw
    return result


def _materialize_program(program: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    kind = program["kind"]
    if kind == "HOLD":
        positions = _route_positions(
            [program["position_m"]],
            [0],
            interpolation="hold",
        )
        yaws = np.full(
            FRAME_COUNT,
            normalized_yaw_degrees(float(program["yaw_deg"])),
            dtype=np.float64,
        )
        return positions, yaws
    if kind == "ROTATE_IN_PLACE":
        positions = _route_positions(
            [program["position_m"]],
            [0],
            interpolation="hold",
        )
        yaws = _shortest_arc_yaws(
            float(program["start_yaw_deg"]),
            float(program["end_yaw_deg"]),
            FRAME_COUNT,
        )
        return positions, yaws
    if kind in _MOVE_KINDS:
        try:
            positions = resample_polyline_by_arc_length(
                program["path_points_m"],
                FRAME_COUNT,
                owner=f"{kind} path",
            )
        except M6XTrajectoryError as exc:
            raise SensorRigTrajectoryError(str(exc)) from exc
        if program["heading_policy"] == "FIXED_YAW":
            yaws = np.full(
                FRAME_COUNT,
                normalized_yaw_degrees(float(program["initial_yaw_deg"])),
                dtype=np.float64,
            )
        else:
            yaws = _path_tangent_yaws(
                positions,
                initial_yaw_deg=float(program["initial_yaw_deg"]),
            )
        return positions, yaws
    if kind == "WAYPOINT_ROUTE":
        waypoints = program["waypoints"]
        positions = _route_positions(
            [waypoint["position_m"] for waypoint in waypoints],
            [int(waypoint["frame_index"]) for waypoint in waypoints],
            interpolation="hold" if len(waypoints) == 1 else "piecewise_linear",
        )
        return positions, _waypoint_yaws(waypoints)
    raise AssertionError(f"normalized program has unsupported kind: {kind!r}")


def _canonical_quaternion_from_yaw(yaw_deg: float) -> list[float]:
    quaternion = yaw_rotation_xyzw(yaw_deg)
    return [
        0.0 if abs(float(value)) < 1.0e-15 else float(value)
        for value in quaternion
    ]


def _canonical_world_from_rig(value: Any) -> dict[str, list[float]]:
    if not isinstance(value, Mapping) or set(value) != {
        "translation_m",
        "rotation_xyzw",
    }:
        raise SensorRigTrajectoryError(
            "world_from_rig must contain exactly translation_m and rotation_xyzw"
        )
    translation = _finite_vec3(
        value["translation_m"], owner="world_from_rig.translation_m"
    )
    try:
        original = np.asarray(value["rotation_xyzw"], dtype=np.float64)
        quaternion = normalized_quaternion_xyzw(value["rotation_xyzw"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise SensorRigTrajectoryError(
            "world_from_rig.rotation_xyzw must be a finite unit quaternion"
        ) from exc
    if original.shape != (4,) or not np.all(np.isfinite(original)):
        raise SensorRigTrajectoryError(
            "world_from_rig.rotation_xyzw must be a finite unit quaternion"
        )
    if not np.allclose(original, quaternion, rtol=0.0, atol=1.0e-9):
        raise SensorRigTrajectoryError(
            "world_from_rig.rotation_xyzw must already be unit normalized"
        )
    if abs(float(quaternion[0])) > 1.0e-9 or abs(float(quaternion[2])) > 1.0e-9:
        raise SensorRigTrajectoryError(
            "world_from_rig.rotation_xyzw must be yaw-only about +Y"
        )
    yaw = habitat_yaw_degrees_from_xyzw(quaternion)
    return {
        "translation_m": translation,
        "rotation_xyzw": _canonical_quaternion_from_yaw(yaw),
    }


def compute_sensor_rig_pose_hash(world_from_rig: Any) -> str:
    """Hash one canonical rig pose independently of its frame occurrence."""

    transform = _canonical_world_from_rig(world_from_rig)
    return canonical_json_sha256(
        {
            "schema": SENSOR_RIG_POSE_SCHEMA,
            "coordinate_frame": COORDINATE_FRAME,
            "rig_id": RIG_ID,
            "world_from_rig": transform,
        }
    )


def _world_from_position_yaw(
    position_m: Sequence[float], yaw_deg: float
) -> dict[str, list[float]]:
    return {
        "translation_m": [
            0.0 if float(component) == 0.0 else float(component)
            for component in position_m
        ],
        "rotation_xyzw": _canonical_quaternion_from_yaw(yaw_deg),
    }


def _json_schema_errors(value: Any) -> list[str]:
    schema = load_json(sensor_rig_trajectory_schema_path())
    errors: list[str] = []
    for error in sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    ):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"JSON Schema {location}: {error.message}")
    return errors


def _all_numbers_finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, Real):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_numbers_finite(item) for item in value)
    return False


def validate_sensor_rig_trajectory(value: Any) -> list[str]:
    """Return structural and deterministic-materialization contract errors."""

    errors = _json_schema_errors(value)
    if not isinstance(value, Mapping):
        return errors
    if not _all_numbers_finite(value):
        errors.append("trajectory must contain only finite JSON numbers")
    if errors:
        return errors
    try:
        program = _normalize_program(value["program"])
        expected_positions, expected_yaws = _materialize_program(program)
    except SensorRigTrajectoryError as exc:
        errors.extend(exc.errors)
        return errors

    for frame_index, frame in enumerate(value["frames"]):
        prefix = f"frames[{frame_index}]"
        if frame["frame_index"] != frame_index:
            errors.append(f"{prefix}.frame_index must equal its array position")
        expected_pts = frame_index * TICKS_PER_FRAME
        if frame["pts_ticks"] != expected_pts:
            errors.append(f"{prefix}.pts_ticks must equal {expected_pts}")
        expected_transform = _world_from_position_yaw(
            expected_positions[frame_index],
            float(expected_yaws[frame_index]),
        )
        if frame["world_from_rig"] != expected_transform:
            errors.append(f"{prefix}.world_from_rig differs from materialized program")
            continue
        expected_hash = compute_sensor_rig_pose_hash(expected_transform)
        if frame["pose_hash"] != expected_hash:
            errors.append(f"{prefix}.pose_hash does not bind world_from_rig")
    return errors


def materialize_sensor_rig_trajectory(
    *,
    trajectory_id: str,
    program: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize one SensorRigTrajectory v1 document on the fixed clock."""

    normalized_program = _normalize_program(program)
    positions, yaws = _materialize_program(normalized_program)
    frames: list[dict[str, Any]] = []
    for frame_index in range(FRAME_COUNT):
        world_from_rig = _world_from_position_yaw(
            positions[frame_index],
            float(yaws[frame_index]),
        )
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * TICKS_PER_FRAME,
                "world_from_rig": world_from_rig,
                "pose_hash": compute_sensor_rig_pose_hash(world_from_rig),
            }
        )
    result = {
        "schema": SENSOR_RIG_TRAJECTORY_SCHEMA,
        "trajectory_id": trajectory_id,
        "rig_id": RIG_ID,
        "formal_view_id": FORMAL_VIEW_ID,
        "listener_id": LISTENER_ID,
        "coordinate_frame": COORDINATE_FRAME,
        "camera_listener_coupling": CAMERA_LISTENER_COUPLING,
        "rig_from_camera": {
            key: list(values) for key, values in IDENTITY_TRANSFORM.items()
        },
        "rig_from_listener": {
            key: list(values) for key, values in IDENTITY_TRANSFORM.items()
        },
        "pose_model": POSE_MODEL,
        "pose_hash_algorithm": POSE_HASH_ALGORITHM,
        "time_base_hz": TIME_BASE_HZ,
        "duration_ticks": DURATION_TICKS,
        "frame_rate_hz": FRAME_RATE_HZ,
        "frame_count": FRAME_COUNT,
        "ticks_per_frame": TICKS_PER_FRAME,
        "program": normalized_program,
        "frames": frames,
    }
    errors = validate_sensor_rig_trajectory(result)
    if errors:
        raise SensorRigTrajectoryError(errors)
    return result


__all__ = [
    "CAMERA_LISTENER_COUPLING",
    "COORDINATE_FRAME",
    "DURATION_TICKS",
    "FORMAL_VIEW_ID",
    "FRAME_COUNT",
    "FRAME_RATE_HZ",
    "LISTENER_ID",
    "POSE_HASH_ALGORITHM",
    "POSE_MODEL",
    "RIG_ID",
    "SENSOR_RIG_POSE_SCHEMA",
    "SENSOR_RIG_TRAJECTORY_SCHEMA",
    "SensorRigTrajectoryError",
    "TICKS_PER_FRAME",
    "TIME_BASE_HZ",
    "compute_sensor_rig_pose_hash",
    "materialize_sensor_rig_trajectory",
    "sensor_rig_trajectory_schema_path",
    "validate_sensor_rig_trajectory",
]

"""M7 binding for the shared camera/listener SensorRigTrajectory contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256
from avengine.m5_1.orientation import habitat_yaw_degrees_from_xyzw
from avengine.sensor_rig_trajectory import (
    SensorRigTrajectoryError,
    materialize_sensor_rig_trajectory,
    validate_sensor_rig_trajectory,
)


class M7SensorRigError(ValueError):
    """An M7 episode cannot bind a complete formal sensor-rig trajectory."""


@dataclass(frozen=True)
class M7SensorRigPoseSeries:
    """Validated per-frame camera/listener poses in both quaternion layouts."""

    positions_m: np.ndarray
    rotations_xyzw: np.ndarray
    orientations_wxyz: np.ndarray
    yaws_deg: np.ndarray
    pose_hashes: tuple[str, ...]


def resolve_m7_sensor_rig_trajectory(
    *,
    sensor_rig_trajectory: Mapping[str, Any] | None,
    listener_position_m: Sequence[float],
    listener_yaw_deg: float,
) -> dict[str, Any]:
    """Return a complete M7 rig trajectory, materializing the fixed fallback.

    The returned document is always a validated SensorRigTrajectory v1.  This
    keeps the historical fixed-listener route explicit instead of representing
    it as empty per-frame view hashes.
    """

    if sensor_rig_trajectory is not None:
        errors = validate_sensor_rig_trajectory(sensor_rig_trajectory)
        if errors:
            raise M7SensorRigError(
                "sensor_rig_trajectory is invalid: " + "; ".join(errors)
            )
        return deepcopy(dict(sensor_rig_trajectory))

    try:
        listener_position = np.asarray(listener_position_m, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise M7SensorRigError(
            "listener_position_m must be a finite vec3"
        ) from error
    if listener_position.shape != (3,) or not np.all(
        np.isfinite(listener_position)
    ):
        raise M7SensorRigError("listener_position_m must be a finite vec3")
    if isinstance(listener_yaw_deg, (bool, np.bool_)):
        raise M7SensorRigError("listener_yaw_deg must be finite")
    try:
        normalized_yaw = float(listener_yaw_deg)
    except (TypeError, ValueError, OverflowError) as error:
        raise M7SensorRigError("listener_yaw_deg must be finite") from error
    if not np.isfinite(normalized_yaw):
        raise M7SensorRigError("listener_yaw_deg must be finite")

    identity = canonical_json_sha256(
        {
            "schema": "avengine_m7_fixed_sensor_rig_hold_seed_v1",
            "position_m": listener_position.tolist(),
            "yaw_deg": normalized_yaw,
        }
    )
    try:
        return materialize_sensor_rig_trajectory(
            trajectory_id=f"m7_fixed_hold_{identity[:24]}",
            program={
                "kind": "HOLD",
                "position_m": listener_position.tolist(),
                "yaw_deg": normalized_yaw,
            },
        )
    except SensorRigTrajectoryError as error:
        raise M7SensorRigError(
            f"fixed listener cannot form a SensorRigTrajectory: {error}"
        ) from error


def m7_sensor_rig_binding(
    sensor_rig_trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the compact identity shared by visual and acoustic artifacts."""

    errors = validate_sensor_rig_trajectory(sensor_rig_trajectory)
    if errors:
        raise M7SensorRigError(
            "sensor_rig_trajectory is invalid: " + "; ".join(errors)
        )
    frames = sensor_rig_trajectory["frames"]
    first_pose_hash = frames[0]["pose_hash"]
    return {
        "trajectory_id": sensor_rig_trajectory["trajectory_id"],
        "content_sha256": canonical_json_sha256(sensor_rig_trajectory),
        "pose_hash_algorithm": sensor_rig_trajectory[
            "pose_hash_algorithm"
        ],
        "first_pose_hash": first_pose_hash,
        "last_pose_hash": frames[-1]["pose_hash"],
        "dynamic": any(
            frame["pose_hash"] != first_pose_hash for frame in frames[1:]
        ),
    }


def m7_sensor_rig_pose_series(
    sensor_rig_trajectory: Mapping[str, Any],
) -> M7SensorRigPoseSeries:
    """Decode a validated SensorRigTrajectory without changing its values."""

    binding = m7_sensor_rig_binding(sensor_rig_trajectory)
    del binding
    frames = sensor_rig_trajectory["frames"]
    positions = np.asarray(
        [
            frame["world_from_rig"]["translation_m"]
            for frame in frames
        ],
        dtype=np.float64,
    )
    rotations_xyzw = np.asarray(
        [
            frame["world_from_rig"]["rotation_xyzw"]
            for frame in frames
        ],
        dtype=np.float64,
    )
    orientations_wxyz = rotations_xyzw[:, (3, 0, 1, 2)]
    yaws = np.asarray(
        [
            habitat_yaw_degrees_from_xyzw(rotation)
            for rotation in rotations_xyzw
        ],
        dtype=np.float64,
    )
    frame_count = int(sensor_rig_trajectory["frame_count"])
    if (
        positions.shape != (frame_count, 3)
        or rotations_xyzw.shape != (frame_count, 4)
        or orientations_wxyz.shape != (frame_count, 4)
        or yaws.shape != (frame_count,)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(rotations_xyzw))
        or not np.all(np.isfinite(yaws))
    ):
        raise M7SensorRigError(
            "sensor_rig_trajectory has invalid pose arrays"
        )
    return M7SensorRigPoseSeries(
        positions_m=np.ascontiguousarray(positions),
        rotations_xyzw=np.ascontiguousarray(rotations_xyzw),
        orientations_wxyz=np.ascontiguousarray(orientations_wxyz),
        yaws_deg=np.ascontiguousarray(yaws),
        pose_hashes=tuple(str(frame["pose_hash"]) for frame in frames),
    )


def validate_m7_rir_listener_alignment(
    *,
    rir_job_plan: Mapping[str, Any],
    sensor_rig_trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check every sampled RIR use against the same-frame rig pose."""

    binding = m7_sensor_rig_binding(sensor_rig_trajectory)
    poses = m7_sensor_rig_pose_series(sensor_rig_trajectory)
    mode = rir_job_plan.get("listener_pose_mode", "fixed")
    if mode == "fixed":
        planned_position = np.asarray(
            rir_job_plan.get("listener_position_m"), dtype=np.float64
        )
        planned_orientation = np.asarray(
            rir_job_plan.get("listener_orientation_wxyz"), dtype=np.float64
        )
        if (
            binding["dynamic"]
            or planned_position.shape != (3,)
            or planned_orientation.shape != (4,)
            or not np.allclose(
                planned_position,
                poses.positions_m[0],
                rtol=0.0,
                atol=1.0e-9,
            )
            or not math.isclose(
                abs(
                    float(
                        np.dot(
                            planned_orientation,
                            poses.orientations_wxyz[0],
                        )
                    )
                ),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise M7SensorRigError(
                "SensorRigTrajectory differs from the fixed RIR Listener pose"
            )
        return {
            "listener_pose_mode": "fixed",
            "checked_use_count": int(
                rir_job_plan.get("requested_pair_state_count", 0)
            ),
            "acoustic_state_binding": rir_job_plan.get(
                "acoustic_state_binding",
                "legacy_fixed_listener_plan",
            ),
        }
    if mode != "per_episode_frame":
        raise M7SensorRigError(
            f"unsupported RIR listener_pose_mode: {mode!r}"
        )
    jobs = rir_job_plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise M7SensorRigError("per-frame Listener RIR plan has no jobs")
    checked_uses = 0
    for job in jobs:
        if not isinstance(job, Mapping):
            raise M7SensorRigError("per-frame Listener RIR job is malformed")
        listener_position = np.asarray(
            job.get("listener_position_m"), dtype=np.float64
        )
        listener_orientation = np.asarray(
            job.get("listener_orientation_wxyz"), dtype=np.float64
        )
        uses = job.get("uses")
        if (
            listener_position.shape != (3,)
            or listener_orientation.shape != (4,)
            or not isinstance(uses, list)
            or not uses
        ):
            raise M7SensorRigError(
                "per-frame Listener RIR job omits its pose"
            )
        for use in uses:
            frame_index = (
                use.get("frame_index")
                if isinstance(use, Mapping)
                else None
            )
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or not 0 <= frame_index < len(poses.pose_hashes)
                or not np.allclose(
                    listener_position,
                    poses.positions_m[frame_index],
                    rtol=0.0,
                    atol=1.0e-9,
                )
                or not math.isclose(
                    abs(
                        float(
                            np.dot(
                                listener_orientation,
                                poses.orientations_wxyz[frame_index],
                            )
                        )
                    ),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ):
                raise M7SensorRigError(
                    "RIR job Listener pose differs from SensorRigTrajectory"
                )
            checked_uses += 1
    if checked_uses != rir_job_plan.get("requested_pair_state_count"):
        raise M7SensorRigError(
            "RIR Listener-pose check did not cover every requested use"
        )
    return {
        "listener_pose_mode": mode,
        "checked_use_count": checked_uses,
        "acoustic_state_binding": "source_listener_pose_per_job_v1",
    }


def validate_m7_visual_listener_alignment(
    *,
    timeline: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    sensor_rig_trajectory: Mapping[str, Any],
    listener_positions_m_by_frame: Sequence[Sequence[float]],
    listener_yaws_deg_by_frame: Sequence[float],
) -> dict[str, Any]:
    """Close Timeline, manifest, Topdown/DOA pose arrays over one rig.

    RGB and UE runtime readbacks are checked by their native capture gates.
    This gate covers the authored visual contract that those runtimes consume
    and the exact Listener arrays passed to Topdown and DOA composition.
    """

    binding = m7_sensor_rig_binding(sensor_rig_trajectory)
    poses = m7_sensor_rig_pose_series(sensor_rig_trajectory)
    frames = timeline.get("frames")
    video = timeline.get("video")
    if (
        not isinstance(frames, list)
        or len(frames) != len(poses.pose_hashes)
        or not isinstance(video, Mapping)
        or video.get("frame_count") != len(poses.pose_hashes)
    ):
        raise M7SensorRigError(
            "Timeline frame clock differs from SensorRigTrajectory"
        )
    for frame_index, (frame, pose_hash) in enumerate(
        zip(frames, poses.pose_hashes, strict=True)
    ):
        view_hashes = (
            frame.get("view_pose_hashes")
            if isinstance(frame, Mapping)
            else None
        )
        if (
            not isinstance(frame, Mapping)
            or frame.get("frame_index") != frame_index
            or not isinstance(view_hashes, Mapping)
            or view_hashes.get("view0") != pose_hash
        ):
            raise M7SensorRigError(
                "Timeline view0 pose hash differs from "
                f"SensorRigTrajectory frame {frame_index}"
            )

    listener = source_manifest.get("listener")
    declared = (
        listener.get("sensor_rig_trajectory")
        if isinstance(listener, Mapping)
        else None
    )
    if (
        not isinstance(declared, Mapping)
        or declared.get("trajectory_id") != binding["trajectory_id"]
        or declared.get("content_sha256") != binding["content_sha256"]
    ):
        raise M7SensorRigError(
            "source manifest SensorRigTrajectory binding differs"
        )

    try:
        positions = np.asarray(
            listener_positions_m_by_frame, dtype=np.float64
        )
        yaws = np.asarray(listener_yaws_deg_by_frame, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise M7SensorRigError(
            "Topdown/DOA Listener pose arrays are invalid"
        ) from error
    if (
        positions.shape != poses.positions_m.shape
        or yaws.shape != poses.yaws_deg.shape
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(yaws))
        or not np.allclose(
            positions, poses.positions_m, rtol=0.0, atol=1.0e-9
        )
        or np.any(
            np.abs((yaws - poses.yaws_deg + 180.0) % 360.0 - 180.0)
            > 1.0e-9
        )
    ):
        raise M7SensorRigError(
            "Topdown/DOA Listener poses differ from SensorRigTrajectory"
        )
    return {
        "schema": "avengine_m7_visual_listener_alignment_gate_v1",
        "status": "pass",
        "sensor_rig_trajectory": binding,
        "checked_frame_count": len(poses.pose_hashes),
        "timeline_view_id": "view0",
        "timeline_pose_hashes_match": True,
        "source_manifest_binding_matches": True,
        "topdown_listener_poses_match": True,
        "doa_listener_poses_match": True,
    }


__all__ = [
    "M7SensorRigPoseSeries",
    "M7SensorRigError",
    "m7_sensor_rig_binding",
    "m7_sensor_rig_pose_series",
    "resolve_m7_sensor_rig_trajectory",
    "validate_m7_rir_listener_alignment",
    "validate_m7_visual_listener_alignment",
]

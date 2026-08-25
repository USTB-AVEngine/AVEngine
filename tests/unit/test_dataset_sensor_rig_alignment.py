from copy import deepcopy

import numpy as np
import pytest

from avengine.dataset.sensor_rig import (
    M7SensorRigError,
    m7_sensor_rig_binding,
    m7_sensor_rig_pose_series,
    validate_m7_visual_listener_alignment,
)
from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory


def _inputs() -> tuple[dict, dict, dict, np.ndarray, np.ndarray]:
    trajectory = materialize_sensor_rig_trajectory(
        trajectory_id="dynamic_review",
        program={
            "kind": "WAYPOINT_ROUTE",
            "waypoints": [
                {
                    "frame_index": 0,
                    "position_m": [0.0, 1.5, 0.0],
                    "yaw_deg": 0.0,
                },
                {
                    "frame_index": 74,
                    "position_m": [1.0, 1.5, -1.0],
                    "yaw_deg": 90.0,
                },
            ],
            "interpolation": "LINEAR_POSITION_SHORTEST_YAW",
        },
    )
    poses = m7_sensor_rig_pose_series(trajectory)
    timeline = {
        "video": {"frame_count": 75},
        "frames": [
            {
                "frame_index": frame_index,
                "view_pose_hashes": {"view0": pose_hash},
            }
            for frame_index, pose_hash in enumerate(poses.pose_hashes)
        ],
    }
    binding = m7_sensor_rig_binding(trajectory)
    manifest = {
        "listener": {
            "sensor_rig_trajectory": {
                "trajectory_id": binding["trajectory_id"],
                "content_sha256": binding["content_sha256"],
            }
        }
    }
    return (
        trajectory,
        timeline,
        manifest,
        poses.positions_m,
        poses.yaws_deg,
    )


def test_visual_listener_alignment_closes_every_frame() -> None:
    trajectory, timeline, manifest, positions, yaws = _inputs()
    result = validate_m7_visual_listener_alignment(
        timeline=timeline,
        source_manifest=manifest,
        sensor_rig_trajectory=trajectory,
        listener_positions_m_by_frame=positions,
        listener_yaws_deg_by_frame=yaws,
    )
    assert result["status"] == "pass"
    assert result["checked_frame_count"] == 75
    assert result["timeline_pose_hashes_match"] is True
    assert result["topdown_listener_poses_match"] is True
    assert result["doa_listener_poses_match"] is True


@pytest.mark.parametrize("owner", ("timeline", "manifest", "topdown"))
def test_visual_listener_alignment_fails_closed(owner: str) -> None:
    trajectory, timeline, manifest, positions, yaws = _inputs()
    if owner == "timeline":
        timeline = deepcopy(timeline)
        timeline["frames"][37]["view_pose_hashes"]["view0"] = "0" * 64
    elif owner == "manifest":
        manifest = deepcopy(manifest)
        manifest["listener"]["sensor_rig_trajectory"][
            "content_sha256"
        ] = "0" * 64
    else:
        positions = positions.copy()
        positions[37, 0] += 0.1
    with pytest.raises(M7SensorRigError):
        validate_m7_visual_listener_alignment(
            timeline=timeline,
            source_manifest=manifest,
            sensor_rig_trajectory=trajectory,
            listener_positions_m_by_frame=positions,
            listener_yaws_deg_by_frame=yaws,
        )

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    write_json,
)
from avengine.m7.sensor_rig import (
    m7_sensor_rig_binding,
    m7_sensor_rig_pose_series,
)
from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory
from tools.m7.build_asset_bound_visual_reviews import (
    AssetBoundReviewError,
    LISTENER_POSITION_M,
    LISTENER_YAW_DEG,
    _assert_audio_sensor_rig_binding,
    _capture_sensor_rig,
)


def _dynamic_trajectory() -> dict[str, object]:
    return materialize_sensor_rig_trajectory(
        trajectory_id="review_dynamic_rig",
        program={
            "kind": "WAYPOINT_ROUTE",
            "waypoints": [
                {
                    "frame_index": 0,
                    "position_m": [-0.7, 1.471, 0.65],
                    "yaw_deg": 55.0,
                },
                {
                    "frame_index": 74,
                    "position_m": [0.2, 1.471, -0.1],
                    "yaw_deg": -25.0,
                },
            ],
            "interpolation": "LINEAR_POSITION_SHORTEST_YAW",
        },
    )


def _write_dynamic_capture(root: Path) -> dict[str, object]:
    trajectory = _dynamic_trajectory()
    binding = m7_sensor_rig_binding(trajectory)
    poses = m7_sensor_rig_pose_series(trajectory)
    arrays = root / "arrays"
    arrays.mkdir(parents=True)
    write_json(root / "sensor_rig_trajectory.json", trajectory)
    np.save(
        arrays / "listener_positions_m.npy",
        poses.positions_m,
        allow_pickle=False,
    )
    np.save(
        arrays / "listener_rotations_xyzw.npy",
        poses.rotations_xyzw,
        allow_pickle=False,
    )
    frame_readbacks = []
    for index in range(75):
        expected = deepcopy(trajectory["frames"][index]["world_from_rig"])
        camera = deepcopy(expected)
        listener = deepcopy(expected)
        frame_readbacks.append(
            {
                "frame_index": index,
                "pts_ticks": index * 3_200,
                "sensor_rig": {
                    "trajectory_id": binding["trajectory_id"],
                    "view_pose_hash": poses.pose_hashes[index],
                    "expected_world_from_rig": expected,
                    "agent_readback": deepcopy(expected),
                    "camera_readback": camera,
                    "listener_readback": listener,
                    "sensor_readbacks": {
                        "rgb_sensor": deepcopy(camera),
                        "depth_sensor": deepcopy(camera),
                        "listener_sensor": deepcopy(listener),
                    },
                    "transform_errors": {
                        "agent": 0.0,
                        "camera": 0.0,
                        "listener": 0.0,
                        "all_sensors": 0.0,
                    },
                },
            }
        )
    write_json(
        root / "frame_readback.json",
        frame_readbacks,
    )
    position_array = np.load(
        arrays / "listener_positions_m.npy",
        allow_pickle=False,
    )
    rotation_array = np.load(
        arrays / "listener_rotations_xyzw.npy",
        allow_pickle=False,
    )
    evidence = {
        "status": "pass",
        "research_only": True,
        "sensor_rig_trajectory": trajectory,
        "sensor_rig_binding": {
            "trajectory_id": binding["trajectory_id"],
            "content_sha256": binding["content_sha256"],
            "artifact": file_record(
                root / "sensor_rig_trajectory.json",
                relative_to=root,
            ),
        },
        "readback": {
            "maximum_sensor_rig_transform_error": {
                "agent": 0.0,
                "camera": 0.0,
                "listener": 0.0,
                "all_sensors": 0.0,
            },
            "frame_records": file_record(
                root / "frame_readback.json",
                relative_to=root,
            ),
        },
        "array_artifacts": {
            "listener_positions_m": {
                **file_record(
                    arrays / "listener_positions_m.npy",
                    relative_to=root,
                ),
                "dtype": position_array.dtype.str,
                "shape": list(position_array.shape),
                "readback_verified": True,
            },
            "listener_rotations_xyzw": {
                **file_record(
                    arrays / "listener_rotations_xyzw.npy",
                    relative_to=root,
                ),
                "dtype": rotation_array.dtype.str,
                "shape": list(rotation_array.shape),
                "readback_verified": True,
            },
        },
    }
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    write_json(root / "evidence.json", evidence)
    return trajectory


def _refresh_capture_evidence(root: Path) -> None:
    evidence_path = root / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["sensor_rig_binding"]["artifact"] = file_record(
        root / "sensor_rig_trajectory.json",
        relative_to=root,
    )
    for name in ("listener_positions_m", "listener_rotations_xyzw"):
        path = root / "arrays" / f"{name}.npy"
        array = np.load(path, allow_pickle=False)
        evidence["array_artifacts"][name] = {
            **file_record(path, relative_to=root),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "readback_verified": True,
        }
    evidence["readback"]["frame_records"] = file_record(
        root / "frame_readback.json",
        relative_to=root,
    )
    evidence.pop("evidence_content_sha256", None)
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    write_json(evidence_path, evidence)


def test_dynamic_capture_listener_arrays_and_pose_hashes_close_exactly(
    tmp_path: Path,
) -> None:
    trajectory = _write_dynamic_capture(tmp_path)
    expected_binding = m7_sensor_rig_binding(trajectory)

    loaded = _capture_sensor_rig(tmp_path)

    assert loaded.binding == expected_binding
    assert loaded.positions_m.shape == (75, 3)
    assert loaded.yaws_deg[0] == pytest.approx(55.0)
    assert loaded.yaws_deg[-1] == pytest.approx(-25.0)
    assert loaded.cross_modal_check["status"] == "pass"
    assert loaded.cross_modal_check["checked_pose_hash_count"] == 75
    assert (
        loaded.cross_modal_check[
            "actual_agent_camera_listener_readbacks_match"
        ]
        is True
    )
    assert (
        loaded.cross_modal_check[
            "reported_transform_errors_match_actual_readbacks"
        ]
        is True
    )
    assert set(
        loaded.cross_modal_check["validated_artifact_sha256"]
    ) == {
        "sensor_rig_trajectory",
        "listener_positions_m",
        "listener_rotations_xyzw",
        "frame_readback",
    }
    assert (
        _assert_audio_sensor_rig_binding(
            sample={"sensor_rig_trajectory": expected_binding},
            capture_sensor_rig=loaded,
        )
        == "exact_dynamic_sensor_rig_binding"
    )


def test_dynamic_capture_requires_exact_audio_sensor_rig_binding(
    tmp_path: Path,
) -> None:
    _write_dynamic_capture(tmp_path)
    loaded = _capture_sensor_rig(tmp_path)

    with pytest.raises(AssetBoundReviewError, match="exact audio binding"):
        _assert_audio_sensor_rig_binding(
            sample={},
            capture_sensor_rig=loaded,
        )


def test_dynamic_capture_rejects_listener_array_content_hash_drift(
    tmp_path: Path,
) -> None:
    _write_dynamic_capture(tmp_path)
    path = tmp_path / "arrays" / "listener_positions_m.npy"
    positions = np.load(path, allow_pickle=False)
    positions[37, 0] += 0.01
    np.save(path, positions, allow_pickle=False)

    with pytest.raises(
        AssetBoundReviewError,
        match="content hash differs",
    ):
        _capture_sensor_rig(tmp_path)


def test_dynamic_capture_rejects_rehashed_listener_array_numeric_drift(
    tmp_path: Path,
) -> None:
    _write_dynamic_capture(tmp_path)
    path = tmp_path / "arrays" / "listener_positions_m.npy"
    positions = np.load(path, allow_pickle=False)
    positions[37, 0] += 0.01
    np.save(path, positions, allow_pickle=False)
    _refresh_capture_evidence(tmp_path)

    with pytest.raises(
        AssetBoundReviewError,
        match="Listener arrays differ from SensorRigTrajectory",
    ):
        _capture_sensor_rig(tmp_path)


def test_dynamic_capture_rejects_frame_pose_hash_drift(tmp_path: Path) -> None:
    _write_dynamic_capture(tmp_path)
    readback_path = tmp_path / "frame_readback.json"

    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    changed = deepcopy(readback)
    changed[18]["sensor_rig"]["view_pose_hash"] = "0" * 64
    write_json(readback_path, changed)
    _refresh_capture_evidence(tmp_path)

    with pytest.raises(
        AssetBoundReviewError,
        match="pose hash/readback",
    ):
        _capture_sensor_rig(tmp_path)


def test_legacy_fixed_capture_without_sidecar_remains_compatible(
    tmp_path: Path,
) -> None:
    write_json(tmp_path / "evidence.json", {})

    loaded = _capture_sensor_rig(tmp_path)

    assert loaded.binding is None
    assert loaded.positions_m[0] == pytest.approx(LISTENER_POSITION_M)
    assert np.all(loaded.yaws_deg == LISTENER_YAW_DEG)
    assert (
        _assert_audio_sensor_rig_binding(
            sample={},
            capture_sensor_rig=loaded,
        )
        == "legacy_fixed_audio_without_sensor_rig_binding"
    )


def test_partial_capture_sensor_rig_closure_fails_closed(
    tmp_path: Path,
) -> None:
    trajectory = _dynamic_trajectory()
    binding = m7_sensor_rig_binding(trajectory)
    write_json(tmp_path / "sensor_rig_trajectory.json", trajectory)
    evidence = {
        "sensor_rig_binding": {
            "trajectory_id": binding["trajectory_id"],
            "content_sha256": binding["content_sha256"],
            "artifact": file_record(
                tmp_path / "sensor_rig_trajectory.json",
                relative_to=tmp_path,
            ),
        },
    }
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    write_json(tmp_path / "evidence.json", evidence)

    with pytest.raises(
        AssetBoundReviewError,
        match="listener_positions_m array must be a regular file",
    ):
        _capture_sensor_rig(tmp_path)


def test_dynamic_capture_rejects_frame_readback_content_replacement(
    tmp_path: Path,
) -> None:
    _write_dynamic_capture(tmp_path)
    readback_path = tmp_path / "frame_readback.json"
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    readback[7]["sensor_rig"]["agent_readback"]["translation_m"][0] += 0.01
    write_json(readback_path, readback)

    with pytest.raises(
        AssetBoundReviewError,
        match="frame readback artifact.*content hash differs",
    ):
        _capture_sensor_rig(tmp_path)


def test_dynamic_capture_rejects_hand_edited_artifact_record_without_evidence_rehash(
    tmp_path: Path,
) -> None:
    _write_dynamic_capture(tmp_path)
    evidence_path = tmp_path / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["readback"]["frame_records"]["sha256"] = "0" * 64
    write_json(evidence_path, evidence)

    with pytest.raises(
        AssetBoundReviewError,
        match="evidence content hash is missing or invalid",
    ):
        _capture_sensor_rig(tmp_path)


@pytest.mark.parametrize(
    "missing_field",
    (
        "agent_readback",
        "camera_readback",
        "listener_readback",
        "transform_errors",
    ),
)
def test_dynamic_capture_rejects_missing_actual_readback_fields(
    tmp_path: Path,
    missing_field: str,
) -> None:
    _write_dynamic_capture(tmp_path)
    readback_path = tmp_path / "frame_readback.json"
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    readback[12]["sensor_rig"].pop(missing_field)
    write_json(readback_path, readback)
    _refresh_capture_evidence(tmp_path)

    with pytest.raises(
        AssetBoundReviewError,
        match=(
            "actual agent/camera/listener readback is missing"
            if missing_field != "transform_errors"
            else "transform_errors are missing"
        ),
    ):
        _capture_sensor_rig(tmp_path)


@pytest.mark.parametrize(
    ("role", "sensor_id"),
    (
        ("camera", "rgb_sensor"),
        ("listener", "listener_sensor"),
    ),
)
def test_dynamic_capture_rejects_rehashed_actual_camera_or_listener_drift(
    tmp_path: Path,
    role: str,
    sensor_id: str,
) -> None:
    _write_dynamic_capture(tmp_path)
    readback_path = tmp_path / "frame_readback.json"
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    rig = readback[31]["sensor_rig"]
    rig[f"{role}_readback"]["translation_m"][0] += 0.01
    rig["sensor_readbacks"][sensor_id]["translation_m"][0] += 0.01
    rig["transform_errors"][role] = 0.01
    rig["transform_errors"]["all_sensors"] = 0.01
    write_json(readback_path, readback)
    _refresh_capture_evidence(tmp_path)

    with pytest.raises(
        AssetBoundReviewError,
        match="actual agent/camera/listener readback differs",
    ):
        _capture_sensor_rig(tmp_path)


def test_dynamic_capture_rejects_falsified_transform_errors(
    tmp_path: Path,
) -> None:
    _write_dynamic_capture(tmp_path)
    readback_path = tmp_path / "frame_readback.json"
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    readback[22]["sensor_rig"]["transform_errors"]["camera"] = 0.25
    write_json(readback_path, readback)
    _refresh_capture_evidence(tmp_path)

    with pytest.raises(
        AssetBoundReviewError,
        match="transform_errors differ from actual readback",
    ):
        _capture_sensor_rig(tmp_path)


def test_dynamic_capture_rejects_falsified_maximum_transform_errors(
    tmp_path: Path,
) -> None:
    _write_dynamic_capture(tmp_path)
    evidence_path = tmp_path / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["readback"]["maximum_sensor_rig_transform_error"][
        "listener"
    ] = 0.5
    evidence.pop("evidence_content_sha256")
    evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
    write_json(evidence_path, evidence)

    with pytest.raises(
        AssetBoundReviewError,
        match="maximum sensor-rig transform_errors differ",
    ):
        _capture_sensor_rig(tmp_path)

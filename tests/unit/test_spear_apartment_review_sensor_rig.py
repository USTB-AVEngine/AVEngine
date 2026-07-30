import json
from pathlib import Path
import sys

import numpy as np
import pytest

from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory
from tools.m7 import build_spear_apartment_review as review


class _ObstacleMap:
    def summary(self) -> dict:
        return {"authority": "test"}


def _trajectory() -> dict:
    return materialize_sensor_rig_trajectory(
        trajectory_id="spear_review_dynamic_rig_v1",
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
                    "position_m": [0.8, 1.471, -1.25],
                    "yaw_deg": -35.0,
                },
            ],
            "interpolation": "LINEAR_POSITION_SHORTEST_YAW",
        },
    )


def _spec() -> dict:
    frames = np.linspace(0.0, 0.74, 75)
    return {
        "scenario_id": "dynamic_spear_review_test",
        "room_backend": "spear_apartment_0000",
        "render_config": {
            "n_frames": 75,
            "fps": 15,
            "width": 4,
            "height": 3,
        },
        "camera_configs": [{"fov_deg": 90.0}],
        "mic": {"pos_m": [0.5, 0.15, 1.2], "yaw_deg": 145.0},
        "audio_config": {
            "output_channels": 2,
            "sample_rate_hz": 48_000,
            "n_samples": 240_000,
        },
        "sources": [
            {
                "tag": "source_a",
                "asset_id": "source_a",
                "asset_class": "human",
                "audio_lookup": "speech",
                "kind": "human",
                "trajectory_m": [
                    [float(value), 0.0, 0.0] for value in frames
                ],
                "audio_source_height_offset_m": 1.5,
            },
            {
                "tag": "source_b",
                "asset_id": "source_b",
                "asset_class": "dog",
                "audio_lookup": "bark",
                "kind": "dog",
                "trajectory_m": [
                    [float(value), 1.0, 0.0] for value in frames
                ],
                "audio_source_height_offset_m": 0.4,
            },
        ],
    }


def _camera_readbacks(trajectory: dict) -> list[dict]:
    poses = review.m7_sensor_rig_pose_series(trajectory)
    return [
        {
            "frame_index": frame_index,
            "location_cm": list(
                review.habitat_point_to_apartment_ue_cm(
                    poses.positions_m[frame_index].tolist()
                )
            ),
            "rotation_deg": [
                0.0,
                0.0,
                review.camera_ue_yaw_degrees(
                    float(poses.yaws_deg[frame_index])
                ),
            ],
            "expected_pose_hash": frame["pose_hash"],
        }
        for frame_index, frame in enumerate(trajectory["frames"])
    ]


def _run_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_sensor_rig: bool,
) -> tuple[dict, dict[str, dict]]:
    spec = _spec()
    trajectory = _trajectory()
    spec_path = tmp_path / "scenario.json"
    visual_path = tmp_path / "evidence.json"
    video_path = tmp_path / "ue.mp4"
    audio_path = tmp_path / "audio.wav"
    feasibility_root = tmp_path / "feasibility"
    feasibility_root.mkdir()
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    visual_path.write_text(
        json.dumps(
            {
                "capture_warmup": {"status": "passed"},
                "sources": [{"tag": "source_a"}, {"tag": "source_b"}],
                "rig_direction_evidence": {
                    "source_a": {"status": "passed"},
                    "source_b": {"status": "passed"},
                },
                "root_readback": {"camera": {"status": "pass"}},
                "runtime_readbacks": {
                    "camera_root": _camera_readbacks(trajectory)
                },
            }
        ),
        encoding="utf-8",
    )
    video_path.write_bytes(b"video")
    audio_path.write_bytes(b"audio")
    (feasibility_root / "feasible_region.json").write_bytes(b"{}")
    (feasibility_root / "feasible_region_source1.npz").write_bytes(b"npz")
    trajectory_path = tmp_path / "sensor_rig_trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")

    captured: dict[str, dict] = {}
    monkeypatch.setattr(
        review,
        "_audio_contract",
        lambda *_args, **_kwargs: {"status": "pass"},
    )
    monkeypatch.setattr(
        review,
        "_obstacle_map",
        lambda _root: (_ObstacleMap(), object()),
    )
    monkeypatch.setattr(
        review,
        "evaluate_source_center_gate",
        lambda _pathfinder, _obstacle_map, center_paths, **_kwargs: {
            "status": "pass",
            "sources": {
                source_id: {
                    "status": "pass",
                    "minimum_navmesh_clearance_m": 1.0,
                    "failed_frame_indices": [],
                    "frames": [
                        {"navmesh_clearance_m": 1.0}
                        for _ in range(75)
                    ],
                }
                for source_id in center_paths
            },
        },
    )

    def fake_topdown(*_args, **kwargs):
        captured["topdown"] = kwargs
        return np.zeros((75, 3, 4, 3), dtype=np.uint8)

    def fake_compose(**kwargs):
        captured["compose"] = kwargs
        return np.zeros((75, 3, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(review, "render_runtime_topdown_frames", fake_topdown)
    monkeypatch.setattr(
        review,
        "_decode_video_rgb",
        lambda *_args, **_kwargs: np.zeros((75, 3, 4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(review, "compose_annotated_frames", fake_compose)
    monkeypatch.setattr(
        review,
        "encode_annotated_review",
        lambda *_args, **_kwargs: {"status": "pass"},
    )

    output_path = tmp_path / "review.mp4"
    review.build_review(
        spec_path=spec_path,
        ue_video_path=video_path,
        visual_metadata_path=visual_path,
        audio_path=audio_path,
        feasibility_root=feasibility_root,
        output_path=output_path,
        sensor_rig_trajectory_path=(
            trajectory_path if with_sensor_rig else None
        ),
    )
    evidence = json.loads(
        output_path.with_suffix(".evidence.json").read_text(encoding="utf-8")
    )
    return evidence, captured


def test_dynamic_sensor_rig_drives_topdown_doa_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence, captured = _run_review(
        tmp_path, monkeypatch, with_sensor_rig=True
    )
    trajectory = _trajectory()
    expected_positions = np.asarray(
        [
            frame["world_from_rig"]["translation_m"]
            for frame in trajectory["frames"]
        ]
    )

    np.testing.assert_allclose(
        captured["topdown"]["listener_positions_m_by_frame"],
        expected_positions,
    )
    np.testing.assert_allclose(
        captured["compose"]["listener_positions_m_by_frame"],
        expected_positions,
    )
    assert len(captured["topdown"]["listener_yaws_deg_by_frame"]) == 75
    assert len(captured["compose"]["listener_yaws_deg_by_frame"]) == 75
    assert evidence["sensor_rig_trajectory"]["dynamic"] is True
    assert evidence["ue_camera_readback_binding"]["checked_frame_count"] == 75
    assert (
        evidence["ue_camera_readback_binding"][
            "checked_actual_readback_series_count"
        ]
        == 1
    )
    assert (
        evidence["coordinate_transform"]["listener_pose_mode"]
        == "sensor_rig_trajectory_v1"
    )


def test_ue_expected_pose_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    trajectory = _trajectory()
    records = _camera_readbacks(trajectory)
    records[37]["expected_pose_hash"] = "0" * 64
    metadata_path = tmp_path / "evidence.json"
    metadata_path.write_text(
        json.dumps(
            {
                "root_readback": {"camera": {"status": "pass"}},
                "runtime_readbacks": {"camera_root": records},
            }
        ),
        encoding="utf-8",
    )
    poses = review.m7_sensor_rig_pose_series(trajectory)

    with pytest.raises(
        review.SpearApartmentReviewError,
        match="differs from SensorRigTrajectory at frame 37",
    ):
        review._validate_ue_camera_readback_binding(
            metadata_path,
            expected_pose_hashes=tuple(
                frame["pose_hash"] for frame in trajectory["frames"]
            ),
            listener_positions_m=poses.positions_m,
            listener_yaws_deg=poses.yaws_deg,
        )


def test_missing_actual_camera_readback_series_fails_closed(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory()
    poses = review.m7_sensor_rig_pose_series(trajectory)
    metadata_path = tmp_path / "evidence.json"
    metadata_path.write_text(
        json.dumps(
            {
                "root_readback": {"camera": {"status": "pass"}},
                "frames": [
                    {"camera_state": {"pose_hash": frame["pose_hash"]}}
                    for frame in trajectory["frames"]
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        review.SpearApartmentReviewError,
        match="actual camera readback series is missing",
    ):
        review._validate_ue_camera_readback_binding(
            metadata_path,
            expected_pose_hashes=poses.pose_hashes,
            listener_positions_m=poses.positions_m,
            listener_yaws_deg=poses.yaws_deg,
        )


def test_wrong_actual_camera_transform_fails_even_with_matching_hash(
    tmp_path: Path,
) -> None:
    trajectory = _trajectory()
    poses = review.m7_sensor_rig_pose_series(trajectory)
    records = _camera_readbacks(trajectory)
    records[19]["location_cm"][0] += 1.0
    metadata_path = tmp_path / "evidence.json"
    metadata_path.write_text(
        json.dumps(
            {
                "root_readback": {"camera": {"status": "pass"}},
                "runtime_readbacks": {"camera_root": records},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        review.SpearApartmentReviewError,
        match="actual camera readback differs.*frame 19",
    ):
        review._validate_ue_camera_readback_binding(
            metadata_path,
            expected_pose_hashes=poses.pose_hashes,
            listener_positions_m=poses.positions_m,
            listener_yaws_deg=poses.yaws_deg,
        )


@pytest.mark.parametrize(
    ("axis", "wrong_angle_deg"),
    ((0, 45.0), (1, -30.0)),
    ids=("roll", "pitch"),
)
def test_wrong_actual_camera_roll_or_pitch_fails_closed(
    tmp_path: Path,
    axis: int,
    wrong_angle_deg: float,
) -> None:
    trajectory = _trajectory()
    poses = review.m7_sensor_rig_pose_series(trajectory)
    records = _camera_readbacks(trajectory)
    for record in records:
        record["rotation_deg"][axis] = wrong_angle_deg
    metadata_path = tmp_path / "evidence.json"
    metadata_path.write_text(
        json.dumps(
            {
                "root_readback": {"camera": {"status": "pass"}},
                "runtime_readbacks": {"camera_root": records},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        review.SpearApartmentReviewError,
        match="actual camera readback differs",
    ):
        review._validate_ue_camera_readback_binding(
            metadata_path,
            expected_pose_hashes=poses.pose_hashes,
            listener_positions_m=poses.positions_m,
            listener_yaws_deg=poses.yaws_deg,
        )


def test_failed_root_camera_gate_fails_closed(tmp_path: Path) -> None:
    trajectory = _trajectory()
    poses = review.m7_sensor_rig_pose_series(trajectory)
    metadata_path = tmp_path / "evidence.json"
    metadata_path.write_text(
        json.dumps(
            {
                "root_readback": {"camera": {"status": "fail"}},
                "runtime_readbacks": {
                    "camera_root": _camera_readbacks(trajectory)
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        review.SpearApartmentReviewError,
        match="root_readback.camera status did not pass",
    ):
        review._validate_ue_camera_readback_binding(
            metadata_path,
            expected_pose_hashes=poses.pose_hashes,
            listener_positions_m=poses.positions_m,
            listener_yaws_deg=poses.yaws_deg,
        )


def test_legacy_fixed_mic_remains_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence, captured = _run_review(
        tmp_path, monkeypatch, with_sensor_rig=False
    )

    assert captured["topdown"]["listener_positions_m_by_frame"] is None
    assert captured["topdown"]["listener_yaws_deg_by_frame"] is None
    assert captured["compose"]["listener_positions_m_by_frame"] is None
    assert captured["compose"]["listener_yaws_deg_by_frame"] is None
    assert "sensor_rig_trajectory" not in evidence
    assert (
        evidence["coordinate_transform"]["listener_pose_mode"]
        == "legacy_fixed_mic"
    )


def test_cli_forwards_optional_sensor_rig_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}
    paths = {
        name: tmp_path / name
        for name in (
            "spec.json",
            "ue.mp4",
            "visual.json",
            "audio.wav",
            "feasibility",
            "review.mp4",
            "rig.json",
        )
    }
    monkeypatch.setattr(
        review,
        "build_review",
        lambda **kwargs: captured.update(kwargs) or kwargs["output_path"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_spear_apartment_review.py",
            "--spec",
            str(paths["spec.json"]),
            "--ue-video",
            str(paths["ue.mp4"]),
            "--visual-metadata",
            str(paths["visual.json"]),
            "--audio",
            str(paths["audio.wav"]),
            "--feasibility-root",
            str(paths["feasibility"]),
            "--output",
            str(paths["review.mp4"]),
            "--sensor-rig-trajectory",
            str(paths["rig.json"]),
        ],
    )

    review.main()

    assert (
        captured["sensor_rig_trajectory_path"]
        == paths["rig.json"].resolve()
    )

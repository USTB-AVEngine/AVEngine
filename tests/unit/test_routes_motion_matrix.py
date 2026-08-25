from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from avengine.capture.mixed_capture import (
    locomotion_schedule_from_root_trajectory,
    trajectory_world_matrices,
)
from avengine.m6x.capture_adapter import (
    CaptureData,
    HUMAN_BEAGLE_CAPTURE_ADAPTER,
)
from avengine.routes.motion_matrix import (
    EPISODE_FRAME_COUNT,
    build_four_motion_master,
    motion_matrix_record,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def _anchors() -> dict:
    return json.loads(
        (REPOSITORY / "examples/m6x/fixed_apartment/anchor_library.json").read_text(
            encoding="utf-8"
        )
    )


def test_four_motion_master_is_continuous_and_covers_requested_matrix() -> None:
    master = build_four_motion_master(_anchors())
    assert master.frame_count == 4 * EPISODE_FRAME_COUNT
    assert [episode.episode_id for episode in master.episodes] == [
        "static_static",
        "human_moving_dog_static",
        "both_moving",
        "human_static_dog_moving",
    ]
    assert master.actor_root_paths["human0"].shape == (300, 3)
    assert master.actor_root_paths["dog0"].shape == (300, 3)
    for boundary in (75, 150, 225):
        for path in master.actor_root_paths.values():
            assert np.array_equal(path[boundary - 1], path[boundary])


def test_four_motion_locomotion_matches_episode_labels() -> None:
    master = build_four_motion_master(_anchors())
    schedules = {
        actor_id: locomotion_schedule_from_root_trajectory(
            path,
            action_sample_counts={"idle": 25, "walk": 16},
        )
        for actor_id, path in master.actor_root_paths.items()
    }
    for episode in master.episodes:
        start, end = episode.start_frame, episode.end_frame_exclusive
        expected = {
            "human0": "walk" if episode.human_motion == "moving" else "idle",
            "dog0": "walk" if episode.dog_motion == "moving" else "idle",
        }
        for actor_id, action_id in expected.items():
            assert {
                frame.action_id for frame in schedules[actor_id][start:end]
            } == {action_id}


def test_motion_matrix_record_stays_compact() -> None:
    record = motion_matrix_record(build_four_motion_master(_anchors()))
    assert record["frame_count"] == 300
    assert len(json.dumps(record)) < 2_000


def test_capture_adapter_closes_a_300_frame_motion_master() -> None:
    anchors = _anchors()
    trajectories = json.loads(
        (REPOSITORY / "examples/m6x/fixed_apartment/trajectory_templates.json").read_text(
            encoding="utf-8"
        )
    )
    master = build_four_motion_master(anchors)
    forwards = HUMAN_BEAGLE_CAPTURE_ADAPTER.materialize_actor_fallback_forwards_xz(
        trajectories, anchors
    )
    sample_counts = {
        "human0": {"idle": 25, "walk": 16},
        "dog0": {"idle": 25, "walk": 16},
    }
    schedules = {
        actor_id: locomotion_schedule_from_root_trajectory(
            path, action_sample_counts=sample_counts[actor_id]
        )
        for actor_id, path in master.actor_root_paths.items()
    }
    matrices = np.empty((300, 2, 4, 4), dtype=np.float64)
    matrices[:, 0] = trajectory_world_matrices(
        master.actor_root_paths["human0"],
        local_forward_axis=(0.0, 0.0, 1.0),
        fallback_forward_xz=forwards["human0"],
    )
    matrices[:, 1] = trajectory_world_matrices(
        master.actor_root_paths["dog0"],
        local_forward_axis=(1.0, 0.0, 0.0),
        fallback_forward_xz=forwards["dog0"],
    )
    records = []
    for frame_index in range(300):
        record = {}
        for actor_id, key in (("human0", "human"), ("dog0", "beagle")):
            state = schedules[actor_id][frame_index]
            record[key] = {
                "action_id": state.action_id,
                "action_time_ticks": state.action_frame_index * 3_200,
                "action_sample_index": state.action_sample_index,
                "action_phase": state.action_phase,
                "actor_root_position_m": master.actor_root_paths[actor_id][
                    frame_index
                ].tolist(),
            }
        records.append(record)
    capture = CaptureData(
        root=REPOSITORY,
        rgb=np.empty(0),
        semantic=np.empty(0),
        actor_world_matrices=matrices,
        anchor_positions_m=np.zeros((300, 3, 3), dtype=np.float64),
        records=tuple(records),
        evidence={
            "anchor_order": list(HUMAN_BEAGLE_CAPTURE_ADAPTER.capture_anchor_order),
            "runtime": {
                "human_action_sample_counts": sample_counts["human0"],
                "beagle_action_sample_counts": sample_counts["dog0"],
            },
        },
    )

    HUMAN_BEAGLE_CAPTURE_ADAPTER.validate_capture_locomotion(capture)
    HUMAN_BEAGLE_CAPTURE_ADAPTER.validate_capture_orientation(
        capture,
        actor_root_paths=master.actor_root_paths,
        actor_fallback_forwards_xz=forwards,
    )
    provisional = HUMAN_BEAGLE_CAPTURE_ADAPTER.provisional_source_paths(
        anchors, master.actor_root_paths
    )
    actual = HUMAN_BEAGLE_CAPTURE_ADAPTER.actual_source_paths(anchors, capture)
    assert {path.shape for path in provisional.values()} == {(300, 3)}
    assert {path.shape for path in actual.values()} == {(300, 3)}

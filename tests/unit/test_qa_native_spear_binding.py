from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from avengine.dataset.sensor_rig import m7_sensor_rig_pose_series
from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory


REPOSITORY = Path(__file__).resolve().parents[2]


def _load_tool(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BINDER = _load_tool(
    "qa_native_spear_binding_tool", "tools/qa/bind_native_spear_episode.py"
)
COMPILER = _load_tool(
    "qa_fact_table_compiler_tool", "tools/qa/compile_apartment_fact_tables.py"
)


def _sensor_rig() -> dict:
    return materialize_sensor_rig_trajectory(
        trajectory_id="native_binding_test_rotate_v1",
        program={
            "kind": "ROTATE_IN_PLACE",
            "position_m": [-0.7, 1.471, 0.65],
            "start_yaw_deg": 55.0,
            "end_yaw_deg": 145.0,
            "yaw_interpolation": "SHORTEST_ARC",
        },
    )


def _facts_and_readbacks(sensor_rig: dict) -> tuple[dict, dict]:
    series = m7_sensor_rig_pose_series(sensor_rig)
    frame_count = len(series.pose_hashes)
    roots = np.stack(
        [
            np.linspace(-4.0, -2.0, frame_count),
            np.full(frame_count, 0.4),
            np.linspace(-2.0, 0.0, frame_count),
        ],
        axis=1,
    )
    facing_yaw = np.full(frame_count, -135.0)
    facts = {
        "time": {"frame_count": frame_count},
        "listener": {
            "positions_m_by_frame": series.positions_m.tolist(),
            "orientations_wxyz_by_frame": series.orientations_wxyz.tolist(),
            "yaw_deg_by_frame": series.yaws_deg.tolist(),
        },
        "tracks": {
            "instances": {
                "source1": {
                    "root_position_m": roots.tolist(),
                    "facing_yaw_deg": facing_yaw.tolist(),
                }
            }
        },
    }
    camera = []
    actors = []
    for frame_index in range(frame_count):
        camera.append(
            {
                "frame_index": frame_index,
                "expected_pose_hash": series.pose_hashes[frame_index],
                "location_cm": BINDER._world_position_to_ue_cm(
                    series.positions_m[frame_index]
                ).tolist(),
                "rotation_deg": [
                    0.0,
                    0.0,
                    BINDER._world_yaw_to_ue_deg(series.yaws_deg[frame_index]),
                ],
            }
        )
        actors.append(
            {
                "frame_index": frame_index,
                "location_cm": BINDER._world_position_to_ue_cm(
                    roots[frame_index]
                ).tolist(),
                "rotation_deg": [
                    0.0,
                    0.0,
                    BINDER._world_yaw_to_ue_deg(facing_yaw[frame_index]),
                ],
            }
        )
    return facts, {
        "camera_root": camera,
        "actor_roots": {"source1_actor": actors},
    }


def test_native_runtime_alignment_binds_all_camera_and_actor_frames() -> None:
    sensor_rig = _sensor_rig()
    facts, runtime = _facts_and_readbacks(sensor_rig)

    result = BINDER._check_runtime_alignment(
        facts=facts,
        sensor_rig=sensor_rig,
        runtime_readbacks=runtime,
    )

    assert result["camera"] == {
        "checked_frame_count": 75,
        "unique_pose_hash_count": 75,
        "maximum_position_error_cm": 0.0,
        "maximum_yaw_error_deg": 0.0,
    }
    assert result["actors"]["source1_actor"]["checked_frame_count"] == 75


def test_native_runtime_alignment_rejects_pose_hash_drift() -> None:
    sensor_rig = _sensor_rig()
    facts, runtime = _facts_and_readbacks(sensor_rig)
    runtime["camera_root"][12]["expected_pose_hash"] = "0" * 64

    with pytest.raises(BINDER.NativeEpisodeBindingError, match="pose hash drift"):
        BINDER._check_runtime_alignment(
            facts=facts,
            sensor_rig=sensor_rig,
            runtime_readbacks=runtime,
        )


def test_dynamic_plan_listener_jobs_match_sensor_rig_frame_poses() -> None:
    sensor_rig = _sensor_rig()
    series = m7_sensor_rig_pose_series(sensor_rig)
    plan = {
        "jobs": [
            {
                "job_id": f"rir_{frame_index}",
                "listener_position_m": series.positions_m[frame_index].tolist(),
                "listener_orientation_wxyz": series.orientations_wxyz[
                    frame_index
                ].tolist(),
                "uses": [{"frame_index": frame_index}],
            }
            for frame_index in (0, 3, 30, 74)
        ]
    }

    assert (
        COMPILER._check_plan_listener_poses_against_sensor_rig(plan, sensor_rig)
        == 4
    )

    changed = deepcopy(plan)
    changed["jobs"][2]["listener_orientation_wxyz"] = [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(
        COMPILER.FactTableBatchError, match="orientation disagrees"
    ):
        COMPILER._check_plan_listener_poses_against_sensor_rig(changed, sensor_rig)

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/finalize_strict_two_human_dynamic_full75_canary.py"
SPEC = importlib.util.spec_from_file_location("dynamic_finalizer", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _write_float32_wav(path: Path, samples: np.ndarray) -> None:
    payload = np.asarray(samples, dtype="<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 2, 16000, 16000 * 8, 8, 32)
    riff_size = 4 + 8 + len(fmt) + 8 + len(payload)
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(payload))
        + payload
    )


def test_dynamic_expected_rir_counts_are_not_static_two_job_counts() -> None:
    assert TOOL.EXPECTED_ACOUSTICS == {
        "target_moves": {"unique": 76, "source1": 75, "source2": 1, "reuse": 74},
        "distractor_moves": {
            "unique": 76,
            "source1": 1,
            "source2": 75,
            "reuse": 74,
        },
        "both_move": {"unique": 150, "source1": 75, "source2": 75, "reuse": 0},
        "camera_pan_both_static": {
            "unique": 150,
            "source1": 75,
            "source2": 75,
            "reuse": 0,
        },
    }


def test_camera_pan_motion_contract_requires_static_actors_and_75_orientations() -> (
    None
):
    assert TOOL.EXPECTED_MOTION["camera_pan_both_static"] == {
        "action_counts": {
            "source1": {"idle": 75, "walk": 0},
            "source2": {"idle": 75, "walk": 0},
        },
        "interpolated_slots": [],
        "listener_orientation_count": 75,
    }


def test_both_move_requires_two_interpolated_walking_slots() -> None:
    assert TOOL.EXPECTED_MOTION["both_move"] == {
        "action_counts": {
            "source1": {"idle": 0, "walk": 75},
            "source2": {"idle": 0, "walk": 75},
        },
        "interpolated_slots": ["source1", "source2"],
        "listener_orientation_count": 1,
    }
    assert "equal_arc_interpolation_of_exact_native_human_polyline_v1" in (
        TOOL.INTERPOLATED_PATH_METHODS
    )


def test_float32_wav_contract_and_exact_silence(tmp_path: Path) -> None:
    samples = np.zeros((80_000, 2), dtype=np.float32)
    path = tmp_path / "silent.wav"
    _write_float32_wav(path, samples)

    observed, contract = TOOL._wav_float32(path)

    assert observed.shape == (80_000, 2)
    assert np.count_nonzero(observed) == 0
    assert contract["format_tag"] == 3
    assert contract["sample_count"] == 80_000
    assert contract["peak_absolute"] == 0.0


def test_float32_wav_rejects_wrong_sample_count(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    _write_float32_wav(path, np.zeros((79_999, 2), dtype=np.float32))

    with pytest.raises(RuntimeError, match="sample shape drift"):
        TOOL._wav_float32(path)


def _visibility_frame(
    frame_index: int, *, visible_pixels: int, visible_fraction: float
) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "visible_pixels": visible_pixels,
        "visible_fraction": visible_fraction,
        "target_bbox_xyxy_px": [100, 100, 300, 500],
    }


def test_dynamic_visibility_gate_is_fail_closed_per_frame() -> None:
    truth = {
        "schema": "avengine_qa_pixel_visibility_truth_v1",
        "status": "computed_modal_target_only_v1",
        "frame_indices": list(range(75)),
        "resolution_hw": [720, 1280],
        "per_instance": {
            "source1": {
                "frames": [
                    _visibility_frame(
                        index, visible_pixels=12_000, visible_fraction=0.9
                    )
                    for index in range(75)
                ]
            },
            "source2": {
                "frames": [
                    _visibility_frame(index, visible_pixels=8_000, visible_fraction=0.7)
                    for index in range(75)
                ]
            },
        },
    }
    assert TOOL._evaluate_visibility_gate(truth, [7, 50])["status"] == "pass"

    truth["per_instance"]["source1"]["frames"][7]["visible_pixels"] = 9_999
    truth["per_instance"]["source2"]["frames"][37]["visible_fraction"] = 0.49
    rejected = TOOL._evaluate_visibility_gate(truth, [7, 50])

    assert rejected["status"] == "fail"
    assert rejected["target_speech"]["failing_frame_count"] == 1
    assert rejected["target_speech"]["failures"][0]["frame_index"] == 7
    assert rejected["distractor_all_frames"]["failing_frame_count"] == 1
    assert rejected["distractor_all_frames"]["failures"][0]["frame_index"] == 37


def test_dynamic_visibility_window_excludes_target_failures_after_speech() -> None:
    truth = {
        "schema": "avengine_qa_pixel_visibility_truth_v1",
        "status": "computed_modal_target_only_v1",
        "frame_indices": list(range(75)),
        "resolution_hw": [720, 1280],
        "per_instance": {
            "source1": {
                "frames": [
                    _visibility_frame(
                        index,
                        visible_pixels=9_000 if index == 51 else 12_000,
                        visible_fraction=0.9,
                    )
                    for index in range(75)
                ]
            },
            "source2": {
                "frames": [
                    _visibility_frame(index, visible_pixels=8_000, visible_fraction=0.7)
                    for index in range(75)
                ]
            },
        },
    }

    result = TOOL._evaluate_visibility_gate(truth, [7, 50])

    assert result["status"] == "pass"
    assert result["target_speech"]["frame_count"] == 44


def _runtime_transform_fixture() -> tuple[dict[str, object], dict[str, object]]:
    frames = []
    normal = []
    for frame_index in range(75):
        camera_yaw = -142.0 - 6.0 * frame_index / 74.0
        camera = {
            "frame_index": frame_index,
            "expected_pose_hash": f"pose_{frame_index}",
            "location_cm": [-70.0, 65.0, 147.1],
            "rotation_deg": [0.0, 0.0, camera_yaw],
        }
        actors = {
            "source1_actor": {
                "frame_index": frame_index,
                "location_cm": [-202.0, -129.0, 40.0],
                "rotation_deg": [0.0, 0.0, -34.0],
            },
            "source2_actor": {
                "frame_index": frame_index,
                "location_cm": [-321.0, -55.0, 40.0],
                "rotation_deg": [0.0, 0.0, -64.0],
            },
        }
        normal.append({"camera": camera, "actors": actors})
        frames.append(
            {
                "camera_state": {
                    "pose_hash": f"pose_{frame_index}",
                    "ue_position_cm": [-70.0, 65.0, 147.1],
                    "ue_yaw_deg": camera_yaw,
                },
                "actor_states": [
                    {
                        "actor_id": "source1_actor",
                        "translation_ue_cm": [-202.0, -129.0, 40.0],
                        "actor_yaw_ue_deg": -34.0,
                    },
                    {
                        "actor_id": "source2_actor",
                        "translation_ue_cm": [-321.0, -55.0, 40.0],
                        "actor_yaw_ue_deg": -64.0,
                    },
                ],
            }
        )
    readbacks = {
        "normal": normal,
        "target_only": {
            "source1": json.loads(json.dumps(normal)),
            "source2": json.loads(json.dumps(normal)),
        },
    }
    return readbacks, {"frames": frames}


def test_runtime_transform_gate_closes_all_225_camera_readbacks() -> None:
    readbacks, plan = _runtime_transform_fixture()

    result = TOOL._validate_runtime_transform_readbacks(readbacks, plan)

    assert result == {
        "status": "pass_exact_all_normal_and_target_only_frames",
        "readback_pass_count": 3,
        "camera_readback_count": 225,
        "actor_readback_count": 450,
        "maximum_camera_location_error_cm": 0.0,
        "maximum_camera_rotation_error_deg": 0.0,
        "maximum_actor_location_error_cm": 0.0,
        "maximum_actor_rotation_error_deg": 0.0,
        "normal_distinct_camera_yaw_count": 75,
        "normal_camera_yaw_span_deg": pytest.approx(6.0),
    }


def test_runtime_transform_gate_rejects_camera_yaw_drift() -> None:
    readbacks, plan = _runtime_transform_fixture()
    readbacks["target_only"]["source2"][37]["camera"]["rotation_deg"][2] += 0.01

    with pytest.raises(
        RuntimeError, match="runtime camera/actor transform readback drift"
    ):
        TOOL._validate_runtime_transform_readbacks(readbacks, plan)

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


def _ground_contact_profile() -> dict[str, object]:
    return {
        "schema": "avengine_strict_two_human_ground_contact_release_profile_v1",
        "ue_length_unit": "centimeter",
        "bone_names": TOOL.GROUND_CONTACT_BONES,
        "clearance_interval_authority": {
            "derived_from_live_diagnostic": True,
            "artifact": "/evidence/live_ground_contact_diagnostic.json",
        },
        "expected_floor_hit_actor": "ApartmentFloorActor",
        "expected_floor_hit_components": ["ApartmentFloorComponent"],
        "support_anchor_clearance_interval_cm_by_action": {
            "idle": [1.0, 3.0],
            "walk": [0.5, 4.0],
        },
        "minimum_individual_anchor_clearance_cm": -0.5,
        "minimum_floor_normal_z": 0.99,
        "runtime_visual_ground_snap": {
            "schema": "ue_dynamic_ground_snap_v1",
            "target": "attached_visual_actor_root_component",
            "timeline_anchor_mutation_allowed": False,
            "emitter_or_rir_mutation_allowed": False,
            "maximum_abs_correction_cm": 15.0,
            "residual_tolerance_cm": 0.1,
        },
    }


def _ground_contact_readback(clearance_cm: float = 2.0) -> dict[str, object]:
    sides = {}
    for side, bone_names in TOOL.GROUND_CONTACT_BONES.items():
        anchors = {}
        for index, (anchor_kind, bone_name) in enumerate(bone_names.items()):
            observed_clearance = clearance_cm + index
            anchors[anchor_kind] = {
                "bone_name": bone_name,
                "bone_index": index,
                "world_position_ue_cm": [10.0, 20.0, 40.0 + observed_clearance],
                "bone_to_floor_clearance_cm": observed_clearance,
                "floor_trace": {
                    "status": "hit",
                    "profile_name": "BlockAll",
                    "trace_complex": True,
                    "hit_actor": "ApartmentFloorActor",
                    "hit_component": "ApartmentFloorComponent",
                    "hit_point_ue_cm": [10.0, 20.0, 40.0],
                    "hit_normal_ue": [0.0, 0.0, 1.0],
                },
            }
        sides[side] = {
            "status": "observed",
            "anchors": anchors,
            "minimum_bone_to_floor_clearance_cm": clearance_cm,
        }
    return {
        "schema": "avengine_native_live_ground_contact_readback_v1",
        "status": "pass_instrumented_measurement_only",
        "ue_length_unit": "centimeter",
        "runtime_visual_ground_snap": {
            "schema": "ue_dynamic_ground_snap_v1",
            "status": "passed",
            "target": "attached_visual_actor_root_component",
            "floor_trace": {
                "hit_actor": "ApartmentFloorActor",
                "hit_component": "ApartmentFloorComponent",
            },
            "applied_z_correction_cm": 1.75,
            "residual_clearance_cm": 0.02,
            "maximum_timeline_anchor_error_cm": 0.0,
            "timeline_anchor_mutated": False,
            "emitter_or_rir_mutated": False,
            "bounds_role": "action_only_not_release_evidence",
        },
        "sides": sides,
    }


def _ground_contact_case() -> tuple[dict[str, object], dict[str, object]]:
    profile = _ground_contact_profile()
    plan = {
        "actors": [
            {
                "actor_id": actor_id,
                "ground_contact_release_profile": json.loads(json.dumps(profile)),
            }
            for actor_id in ("source1_actor", "source2_actor")
        ],
        "frames": [
            {
                "actor_states": [
                    {"actor_id": "source1_actor", "action_id": "walk"},
                    {"actor_id": "source2_actor", "action_id": "walk"},
                ]
            }
            for _ in range(75)
        ],
    }
    assets = {
        "sampled_frames": [
            {
                "frame_index": frame_index,
                "per_instance": {
                    slot: {
                        "current_action": {"action_id": "walk"},
                        "live_ground_contact_readback": _ground_contact_readback(),
                    }
                    for slot in ("source1", "source2")
                },
            }
            for frame_index in (0, 37, 74)
        ]
    }
    return assets, plan


def test_ground_contact_release_passes_only_declared_live_interval() -> None:
    assets, plan = _ground_contact_case()

    result = TOOL._evaluate_ground_contact_release(assets, plan)

    assert result["status"] == "pass"
    assert result["release_authorized"] is True
    assert result["trace_count"] == 24
    assert result["runtime_visual_ground_snap_count"] == 6
    assert result["timeline_anchor_and_emitter_mutation_count"] == 0
    assert result["minimum_support_anchor_clearance_cm"] == 2.0
    assert result["maximum_support_anchor_clearance_cm"] == 2.0


def test_ground_contact_release_rejects_legacy_capture_without_live_fields() -> None:
    assets, plan = _ground_contact_case()
    for sample in assets["sampled_frames"]:
        for record in sample["per_instance"].values():
            record.pop("live_ground_contact_readback")

    result = TOOL._evaluate_ground_contact_release(assets, plan)

    assert result["status"] == "fail"
    assert result["release_authorized"] is False
    assert "live ground readback is missing" in result["first_blocker"]


def test_ground_contact_release_diagnoses_legacy_live_evidence_before_profile() -> None:
    assets, plan = _ground_contact_case()
    for sample in assets["sampled_frames"]:
        for record in sample["per_instance"].values():
            record.pop("live_ground_contact_readback")
    for actor in plan["actors"]:
        actor.pop("ground_contact_release_profile")

    result = TOOL._evaluate_ground_contact_release(assets, plan)

    assert result["status"] == "fail"
    assert result["release_authorized"] is False
    assert result["first_blocker"] == (
        "source1_actor live ground readback is missing at frame 0"
    )


def test_ground_contact_release_rejects_missing_profile_before_threshold_guess() -> (
    None
):
    assets, plan = _ground_contact_case()
    plan["actors"][0].pop("ground_contact_release_profile")

    result = TOOL._evaluate_ground_contact_release(assets, plan)

    assert result["status"] == "fail"
    assert result["release_authorized"] is False
    assert result["first_blocker"].endswith("release profile is missing")


def test_ground_contact_release_rejects_clearance_outside_declared_interval() -> None:
    assets, plan = _ground_contact_case()
    assets["sampled_frames"][1]["per_instance"]["source1"][
        "live_ground_contact_readback"
    ] = _ground_contact_readback(clearance_cm=5.0)

    result = TOOL._evaluate_ground_contact_release(assets, plan)

    assert result["status"] == "fail"
    assert "outside the declared interval" in result["first_blocker"]


def test_ground_contact_release_rejects_wrong_floor_object() -> None:
    assets, plan = _ground_contact_case()
    trace = assets["sampled_frames"][2]["per_instance"]["source2"][
        "live_ground_contact_readback"
    ]["sides"]["right"]["anchors"]["toe"]["floor_trace"]
    trace["hit_component"] = "ChairComponent"

    result = TOOL._evaluate_ground_contact_release(assets, plan)

    assert result["status"] == "fail"
    assert "undeclared floor object" in result["first_blocker"]

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

STAGING = Path(__file__).resolve().parents[2]
BUILDER_PATH = STAGING / "tools/qa/build_strict_two_human_motion_realism_receipt.py"
VALIDATOR_PATH = (
    STAGING / "tools/qa/validate_strict_two_human_motion_realism_receipt.py"
)


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _module("motion_realism_builder_test", BUILDER_PATH)
VALIDATOR = _module("motion_realism_validator_test", VALIDATOR_PATH)


CASES = [
    ("target_moves", "source1", 1.4735664530973638, [0, 26], 1.625),
    ("distractor_moves", "source2", 0.8501344921715555, [2, 17], 0.9375),
    ("both_move", "source1", 0.5526269018958885, [39, 49], 0.625),
    ("both_move", "source2", 0.5366364650287156, [62, 73], 0.6875),
]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fixture(
    root: Path,
    *,
    mechanism: str,
    slot: str,
    path_length_m: float,
    native_range: list[int],
    phase_cycles: float,
    release_profile: bool,
) -> Path:
    actor_id = f"{slot}_actor"
    other_slot = "source2" if slot == "source1" else "source1"
    other_actor = f"{other_slot}_actor"
    episode_id = f"test_{mechanism}_{slot}"
    native_intervals = native_range[1] - native_range[0]
    active_range = [0, native_intervals]
    roots: list[list[float]] = []
    states: list[dict[str, object]] = []
    for frame_index in range(BUILDER.FRAME_COUNT):
        if release_profile:
            active = frame_index <= native_intervals
            progress = min(frame_index, native_intervals) / native_intervals
        else:
            active = True
            progress = frame_index / (BUILDER.FRAME_COUNT - 1)
        root_point = [path_length_m * progress, 0.4, 0.0]
        roots.append(root_point)
        ticks = frame_index * round(
            BUILDER.TIMELINE_TICKS_PER_SECOND / BUILDER.FRAME_RATE_HZ
        )
        states.append(
            {
                "actor_id": actor_id,
                "action_id": "walk" if active else "idle",
                "ue_animation": "/Game/Test/Walking.Walking"
                if active
                else "/Game/Test/Standing_Idle.Standing_Idle",
                "action_phase": (
                    (ticks / BUILDER.ANIMATION_TICKS_PER_PHASE_CYCLE) % 1.0
                    if active
                    else 0.0
                ),
                "action_time_ticks": ticks if active else 0,
                "translation_m": root_point,
            }
        )
    frames = [
        {
            "frame_index": frame_index,
            "actor_states": [
                states[frame_index],
                {
                    "actor_id": other_actor,
                    "action_id": "idle",
                    "ue_animation": "/Game/Test/Standing_Idle.Standing_Idle",
                    "action_phase": 0.0,
                    "action_time_ticks": 0,
                    "translation_m": [2.0, 0.4, 0.0],
                },
            ],
        }
        for frame_index in range(BUILDER.FRAME_COUNT)
    ]
    suite = {
        "scenarios": [
            {
                "scenario_id": episode_id,
                "plan": {
                    "actors": [
                        {
                            "actor_id": actor_id,
                            "idle_animation": "/Game/Test/Standing_Idle.Standing_Idle",
                            "walking_animation": "/Game/Test/Walking.Walking",
                        },
                        {
                            "actor_id": other_actor,
                            "idle_animation": "/Game/Test/Standing_Idle.Standing_Idle",
                            "walking_animation": "/Game/Test/Walking.Walking",
                        },
                    ],
                    "frames": frames,
                },
            }
        ]
    }
    output_span = (BUILDER.FRAME_COUNT - 1) / BUILDER.FRAME_RATE_HZ
    timing = {
        "schema": "avengine_arc_length_bound_animation_timing_v1",
        "status": "pass",
        "path_length_m": path_length_m,
        "episode_span_seconds": output_span,
        "average_root_speed_m_per_second": path_length_m / output_span,
        "phase_cycle_count": phase_cycles,
        "path_provenance": {
            "method": "arc_length_interpolation_of_native_polyline_v1",
            "native_source_frame_indices_inclusive": native_range,
            "native_anchor_count": native_intervals + 1,
        },
    }
    root_application: dict[str, object] = {"animation_timing": {slot: timing}}
    if release_profile:
        native_duration = native_intervals / BUILDER.FRAME_RATE_HZ
        native_speed = path_length_m / native_duration
        play_length = (
            BUILDER.ANIMATION_TICKS_PER_PHASE_CYCLE / BUILDER.TIMELINE_TICKS_PER_SECOND
        )
        root_application["motion_realism_profiles"] = {
            slot: {
                "schema": BUILDER.PROFILE_SCHEMA,
                "status": "pass",
                "release_qualified": True,
                "slot_id": slot,
                "actor_id": actor_id,
                "native_rate_active_interval": {
                    "output_frame_range_inclusive": active_range,
                    "native_source_frame_range_inclusive": native_range,
                    "native_frame_rate_hz": 15,
                    "output_frame_rate_hz": 15,
                    "output_interval_count": native_intervals,
                    "global_time_stretch_applied": False,
                    "time_scale": 1.0,
                    "outside_action_id": "idle",
                    "outside_root_policy": "hold_boundary_root",
                },
                "root_speed": {
                    "authority": "retained_native_anchor_window_v1",
                    "native_average_speed_m_s": native_speed,
                    "output_active_average_speed_m_s": native_speed,
                    "maximum_relative_error": BUILDER.MAX_NATIVE_SPEED_RELATIVE_ERROR,
                },
                "walking_clip": {
                    "asset_path": "/Game/Test/Walking.Walking",
                    "play_length_readback_source": "live_uanimationasset_play_length_v1",
                    "live_play_length_seconds": play_length,
                    "live_play_rate": 1.0,
                    "timeline_ticks_per_second": BUILDER.TIMELINE_TICKS_PER_SECOND,
                    "animation_ticks_per_phase_cycle": BUILDER.ANIMATION_TICKS_PER_PHASE_CYCLE,
                    "canonical_cycles_per_second": 1.0 / play_length,
                },
                "foot_plant_sync": {
                    "schema": BUILDER.FOOT_PLANT_SCHEMA,
                    "status": "pass",
                    "contact_phase_authority_status": "pass",
                    "contact_phase_authority": "test_fixture_canonical_walking_contact_profile_v1",
                    "walking_asset_path": "/Game/Test/Walking.Walking",
                    "runtime_evidence_kind": "live_per_active_frame_foot_toe_floor_trace_v1",
                    "runtime_frame_indices": list(range(native_intervals + 1)),
                    "bones": BUILDER.REQUIRED_FOOT_BONES,
                    "ground_contact_release_gate_status": "pass",
                    "all_samples_pass": True,
                    "maximum_phase_error_cycles": 0.0,
                    "maximum_planted_foot_slip_m_per_frame": 0.01,
                },
            }
        }
    materialization = {
        "episode_id": episode_id,
        "mechanism": mechanism,
        "frame_count": BUILDER.FRAME_COUNT,
        "frame_rate_hz": int(BUILDER.FRAME_RATE_HZ),
        "suite_actor_root_application": root_application,
    }
    _write(root / "materialization_receipt.json", materialization)
    _write(root / "suite_execution_plan.json", suite)
    return root


class MotionRealismReceiptTests(unittest.TestCase):
    def test_legacy_full75_stretch_is_explicit_nonrelease_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            for mechanism, slot, path_length, native_range, phase_cycles in CASES:
                with self.subTest(mechanism=mechanism, slot=slot):
                    root = _fixture(
                        temporary_root / f"legacy_{mechanism}_{slot}",
                        mechanism=mechanism,
                        slot=slot,
                        path_length_m=path_length,
                        native_range=native_range,
                        phase_cycles=phase_cycles,
                        release_profile=False,
                    )
                    receipt = BUILDER.build_receipt(root)
                    moving = receipt["moving_slots"][0]
                    codes = [item["code"] for item in moving["blockers"]]
                    self.assertEqual(
                        receipt["status"], "reject_nonrelease_motion_realism_gate"
                    )
                    self.assertEqual(
                        receipt["release_classification"],
                        "nonrelease_pipeline_evidence_only",
                    )
                    self.assertEqual(
                        receipt["first_blocker"]["code"],
                        "missing_motion_realism_profile",
                    )
                    self.assertIn(
                        "global_time_stretch_of_short_native_anchor_window", codes
                    )
                    self.assertIn("output_speed_outside_native_rate_envelope", codes)
                    self.assertIn("output_cadence_outside_native_rate_envelope", codes)
                    self.assertIn("missing_live_foot_plant_sync_evidence", codes)
                    VALIDATOR.validate_receipt(receipt, materialization_root=root)

    def test_native_rate_active_interval_profile_passes_motion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mechanism, slot, path_length, native_range, phase_cycles = CASES[0]
            root = _fixture(
                Path(temporary) / "native_rate",
                mechanism=mechanism,
                slot=slot,
                path_length_m=path_length,
                native_range=native_range,
                phase_cycles=phase_cycles,
                release_profile=True,
            )
            receipt = BUILDER.build_receipt(root)
            self.assertEqual(receipt["status"], "pass_motion_realism_release_gate")
            self.assertIsNone(receipt["first_blocker"])
            self.assertEqual(receipt["moving_slots"][0]["blockers"], [])
            VALIDATOR.validate_receipt(receipt, materialization_root=root)

    def test_profile_cannot_declare_global_full75_stretch_as_native_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            mechanism, slot, path_length, native_range, phase_cycles = CASES[0]
            root = _fixture(
                tmp_path / "bad_interval",
                mechanism=mechanism,
                slot=slot,
                path_length_m=path_length,
                native_range=native_range,
                phase_cycles=phase_cycles,
                release_profile=True,
            )
            materialization = json.loads(
                (root / "materialization_receipt.json").read_text()
            )
            profile = materialization["suite_actor_root_application"][
                "motion_realism_profiles"
            ][slot]
            profile["native_rate_active_interval"]["output_frame_range_inclusive"] = [
                0,
                74,
            ]
            profile["native_rate_active_interval"]["output_interval_count"] = 74
            copy_root = tmp_path / "bad_interval_copy"
            _write(copy_root / "materialization_receipt.json", materialization)
            _write(
                copy_root / "suite_execution_plan.json",
                json.loads((root / "suite_execution_plan.json").read_text()),
            )
            receipt = BUILDER.build_receipt(copy_root)
            codes = [item["code"] for item in receipt["moving_slots"][0]["blockers"]]
            self.assertIn("native_rate_active_interval_contract_failed", codes)

    def test_idle_is_mandatory_outside_active_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            mechanism, slot, path_length, native_range, phase_cycles = CASES[0]
            root = _fixture(
                tmp_path / "bad_idle",
                mechanism=mechanism,
                slot=slot,
                path_length_m=path_length,
                native_range=native_range,
                phase_cycles=phase_cycles,
                release_profile=True,
            )
            suite = json.loads((root / "suite_execution_plan.json").read_text())
            state = suite["scenarios"][0]["plan"]["frames"][27]["actor_states"][0]
            state["action_id"] = "walk"
            state["ue_animation"] = "/Game/Test/Walking.Walking"
            copy_root = tmp_path / "bad_idle_copy"
            _write(
                copy_root / "materialization_receipt.json",
                json.loads((root / "materialization_receipt.json").read_text()),
            )
            _write(copy_root / "suite_execution_plan.json", suite)
            receipt = BUILDER.build_receipt(copy_root)
            self.assertIn(
                "non_idle_action_outside_active_interval",
                [item["code"] for item in receipt["moving_slots"][0]["blockers"]],
            )

    def test_live_foot_plant_evidence_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            mechanism, slot, path_length, native_range, phase_cycles = CASES[0]
            root = _fixture(
                tmp_path / "missing_feet",
                mechanism=mechanism,
                slot=slot,
                path_length_m=path_length,
                native_range=native_range,
                phase_cycles=phase_cycles,
                release_profile=True,
            )
            materialization = json.loads(
                (root / "materialization_receipt.json").read_text()
            )
            del materialization["suite_actor_root_application"][
                "motion_realism_profiles"
            ][slot]["foot_plant_sync"]
            copy_root = tmp_path / "missing_feet_copy"
            _write(copy_root / "materialization_receipt.json", materialization)
            _write(
                copy_root / "suite_execution_plan.json",
                json.loads((root / "suite_execution_plan.json").read_text()),
            )
            receipt = BUILDER.build_receipt(copy_root)
            self.assertIn(
                "missing_live_foot_plant_sync_evidence",
                [item["code"] for item in receipt["moving_slots"][0]["blockers"]],
            )

    def test_validator_rejects_tampered_arithmetic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mechanism, slot, path_length, native_range, phase_cycles = CASES[0]
            root = _fixture(
                Path(temporary) / "legacy",
                mechanism=mechanism,
                slot=slot,
                path_length_m=path_length,
                native_range=native_range,
                phase_cycles=phase_cycles,
                release_profile=False,
            )
            receipt = BUILDER.build_receipt(root)
            tampered = copy.deepcopy(receipt)
            tampered["moving_slots"][0]["native_rate_facts"][
                "global_time_stretch_factor"
            ] = 1.0
            with self.assertRaisesRegex(RuntimeError, "arithmetic drift"):
                VALIDATOR.validate_receipt(tampered)


if __name__ == "__main__":
    unittest.main()

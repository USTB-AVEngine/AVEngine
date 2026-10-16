from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

STAGING_ROOT = Path(__file__).resolve().parents[2]
AUDITOR_PATH = (
    STAGING_ROOT
    / "tools"
    / "qa"
    / "audit_strict_two_human_camera_pan_motion_realism.py"
)
VALIDATOR_PATH = (
    STAGING_ROOT
    / "tools"
    / "qa"
    / "validate_strict_two_human_camera_pan_motion_realism.py"
)


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDITOR = _module("camera_motion_auditor_test", AUDITOR_PATH)
VALIDATOR = _module("camera_motion_validator_test", VALIDATOR_PATH)


def _scenario(index: int, yaw_deg: float = 55.0) -> dict[str, object]:
    return {
        "scenario_id": f"scenario_{index:04d}",
        "plan": {
            "camera": {
                "habitat_position_m": [-0.7, 1.471, 0.65],
                "habitat_yaw_deg": yaw_deg,
                "horizontal_fov_deg": 105.0,
                "listener_id": "listener0",
                "ue_position_cm": [-70.0, 65.0, 147.1],
                "ue_yaw_deg": -90.0 - yaw_deg,
            },
            "frames": [
                {"frame_index": frame_index, "actor_states": []}
                for frame_index in range(75)
            ],
            "render": {
                "fps_den": 1,
                "fps_num": 15,
                "frame_count": 75,
                "ticks_per_frame": 3200,
            },
        },
        "render": {"frame_count": 75, "frame_rate_hz": 15},
    }


def _suite(count: int = 1000) -> dict[str, object]:
    return {
        "schema": AUDITOR.EXPECTED_SOURCE_SCHEMA,
        "scenarios": [_scenario(index) for index in range(count)],
    }


def _preflight(*, with_profile: bool = False) -> dict[str, object]:
    provenance: dict[str, object] = {
        "source_suite": "/authority/unique1000/suite_execution_plan.json",
        "midpoint_fresh_capture_episode": (
            "strict2h_dynamic_canary_02_distractor_moves_v2"
        ),
        "coordinate_contract": "UE_yaw_deg=-90-Habitat_yaw_deg",
    }
    if with_profile:
        provenance["approved_camera_motion_profile_id"] = "approved_test_profile_v1"
    return {
        "canaries": [
            {
                "episode_id": AUDITOR.EXPECTED_CANDIDATE_EPISODE,
                "candidate_revision": "camera_pan_v2_0589_right_target_yaw52_58_v1",
                "mechanism": "camera_pan_both_static",
                "target_side": "right",
                "camera": {
                    "yaw_path_deg": [
                        52.0 + 6.0 * frame_index / 74.0 for frame_index in range(75)
                    ],
                    "provenance": provenance,
                },
            }
        ]
    }


def _finalization() -> dict[str, object]:
    return {
        "status": "pass",
        "dynamic_full75_canary_pass": True,
        "capture": {
            "captured_frame_count": 75,
            "runtime": {
                "status": "pass",
                "transform_readbacks": {
                    "status": "pass_exact_all_normal_and_target_only_frames",
                    "camera_readback_count": 225,
                    "normal_camera_yaw_span_deg": 6.0,
                    "normal_distinct_camera_yaw_count": 75,
                    "maximum_camera_location_error_cm": 0.0,
                    "maximum_camera_rotation_error_deg": 0.0,
                },
            },
            "visibility_gate": {
                "status": "pass",
                "target_speech": {
                    "minimum_visible_pixels": 32617,
                    "minimum_visible_fraction": 1.0,
                },
                "distractor_all_frames": {
                    "minimum_visible_pixels": 16218,
                    "minimum_visible_fraction": 1.0,
                },
            },
        },
    }


def _visual_receipt() -> dict[str, object]:
    return {
        "review": {
            "findings": {
                "camera_pan": (
                    "pass_monotonic_visual_pan_consistent_with_runtime_yaw_minus142_to_minus148"
                ),
                "identity_continuity": "pass_left_female_right_male_continuous",
                "visible_floor_clearance_gap_beneath_both_characters": True,
            }
        }
    }


class CameraPanMotionRealismAuditTests(unittest.TestCase):
    def test_unique1000_static_inventory_rejects_release_but_preserves_machine_pass(
        self,
    ) -> None:
        receipt = AUDITOR.build_receipt(
            suite=_suite(),
            source_suite_path="/authority/unique1000/suite_execution_plan.json",
            preflight=_preflight(),
            finalization=_finalization(),
            visual_receipt=_visual_receipt(),
        )
        self.assertEqual(
            receipt["status"], "reject_release_missing_camera_motion_authority"
        )
        self.assertEqual(
            receipt["first_blocker"]["code"],
            "missing_native_or_approved_camera_motion_profile",
        )
        self.assertFalse(receipt["release_qualified"])
        self.assertEqual(receipt["formal_episode_count"], 0)
        self.assertEqual(
            receipt["source_authority"]["inventory"]["scenario_count"], 1000
        )
        self.assertEqual(
            receipt["source_authority"]["inventory"]["total_plan_frame_count"],
            75_000,
        )
        self.assertEqual(
            receipt["source_authority"]["inventory"]["dynamic_camera_inventory"][
                "scenario_count"
            ],
            0,
        )
        self.assertTrue(
            receipt["preserved_machine_evidence"]["dynamic_full75_canary_pass"]
        )
        VALIDATOR.validate_receipt(receipt)

    def test_current_pan_arithmetic_distinguishes_nominal_clip_and_sample_span(
        self,
    ) -> None:
        candidate = AUDITOR._candidate(_preflight(), 15.0)
        self.assertAlmostEqual(candidate["yaw_span_deg"], 6.0)
        self.assertEqual(candidate["nonzero_interframe_step_count"], 74)
        self.assertAlmostEqual(candidate["nominal_clip_duration_s"], 5.0)
        self.assertAlmostEqual(candidate["sampled_interval_duration_s"], 74 / 15)
        self.assertAlmostEqual(candidate["nominal_clip_angular_velocity_deg_s"], 1.2)
        self.assertAlmostEqual(
            candidate["interframe_slope_angular_velocity_deg_s"],
            6.0 / (74 / 15),
        )
        self.assertFalse(candidate["native_or_approved_motion_authority_present"])

    def test_cross_episode_static_yaws_are_never_interpreted_as_speed(self) -> None:
        suite = {
            "schema": AUDITOR.EXPECTED_SOURCE_SCHEMA,
            "scenarios": [_scenario(0, 52.0), _scenario(1, 58.0)],
        }
        inventory = AUDITOR.audit_inventory(suite)
        self.assertEqual(inventory["static_camera_inventory"]["episode_count"], 2)
        self.assertEqual(
            inventory["static_camera_inventory"]["unique_habitat_yaw_count"], 2
        )
        self.assertFalse(
            inventory["static_camera_inventory"][
                "cross_episode_pose_differences_are_motion_samples"
            ]
        )
        self.assertEqual(inventory["dynamic_camera_inventory"]["scenario_count"], 0)
        self.assertEqual(
            inventory["dynamic_camera_inventory"][
                "positive_angular_speed_sample_count"
            ],
            0,
        )

    def test_real_per_frame_yaw_path_produces_native_speed_and_duration_samples(
        self,
    ) -> None:
        scenario = _scenario(0)
        plan = scenario["plan"]
        assert isinstance(plan, dict)
        camera = plan["camera"]
        assert isinstance(camera, dict)
        camera["yaw_path_deg"] = [
            55.0 if frame_index < 10 else 55.0 + (frame_index - 9) * 2.0
            for frame_index in range(75)
        ]
        suite = {"schema": AUDITOR.EXPECTED_SOURCE_SCHEMA, "scenarios": [scenario]}
        inventory = AUDITOR.audit_inventory(suite)
        dynamic = inventory["dynamic_camera_inventory"]
        self.assertEqual(dynamic["scenario_count"], 1)
        self.assertEqual(dynamic["positive_angular_speed_sample_count"], 65)
        self.assertEqual(dynamic["continuous_pan_segment_count"], 1)
        self.assertAlmostEqual(
            dynamic["absolute_angular_speed_deg_s_distribution"]["median"], 30.0
        )
        self.assertAlmostEqual(
            dynamic["continuous_pan_duration_s_distribution"]["median"], 65 / 15
        )

    def test_fail_closed_receipt_refuses_to_ignore_an_approved_profile(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "fail-closed missing-authority path"):
            AUDITOR.build_receipt(
                suite=_suite(),
                source_suite_path="/authority/unique1000/suite_execution_plan.json",
                preflight=_preflight(with_profile=True),
                finalization=_finalization(),
                visual_receipt=_visual_receipt(),
            )

    def test_validator_rejects_invented_candidate_or_release_pass(self) -> None:
        receipt = AUDITOR.build_receipt(
            suite=_suite(),
            source_suite_path="/authority/unique1000/suite_execution_plan.json",
            preflight=_preflight(),
            finalization=_finalization(),
            visual_receipt=_visual_receipt(),
        )
        for mutation in ("candidate", "release"):
            with self.subTest(mutation=mutation):
                tampered = copy.deepcopy(receipt)
                if mutation == "candidate":
                    tampered["native_rate_candidate_search"]["candidate_count"] = 1
                else:
                    tampered["release_qualified"] = True
                with self.assertRaises(RuntimeError):
                    VALIDATOR.validate_receipt(tampered)


if __name__ == "__main__":
    unittest.main()

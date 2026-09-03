from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path

STAGING = Path(__file__).resolve().parents[2]
LOCAL_SOURCE_ROOT = STAGING / ".source_inputs"
AUTHORITATIVE_A_SOURCE_DIRECTORIES = {
    "target_moves": (
        STAGING / "tmp/lead_a_strict_two_human_full_episode_batch_v1/"
        "dynamic_target_moves_v2_materialized_v1"
    ),
    "distractor_moves": (
        STAGING / "tmp/lead_a_strict_two_human_full_episode_batch_v1/"
        "dynamic_distractor_moves_v2_materialized_v1"
    ),
    "both_move": (
        STAGING / "tmp/lead_a_strict_two_human_full_episode_batch_v1/"
        "dynamic_both_move_v1_materialized_v1"
    ),
}
SOURCE_INPUT_FILENAMES = (
    "materialization_receipt.json",
    "suite_execution_plan.json",
)
BUILDER_PATH = (
    STAGING / "tools/qa/build_strict_two_human_native_rate_dynamic_candidates.py"
)
VALIDATOR_PATH = (
    STAGING / "tools/qa/validate_strict_two_human_native_rate_dynamic_candidates.py"
)
LEGACY_BUILDER_PATH = (
    STAGING / "tools/qa/build_strict_two_human_motion_realism_receipt.py"
)


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _module("native_rate_dynamic_builder_test", BUILDER_PATH)
VALIDATOR = _module("native_rate_dynamic_validator_test", VALIDATOR_PATH)
LEGACY_BUILDER = _module("legacy_motion_builder_test", LEGACY_BUILDER_PATH)

EXPECTED_INTERVALS = {
    "target_moves": {"source1": [6, 32]},
    "distractor_moves": {"source2": [21, 36]},
    "both_move": {"source1": [14, 24], "source2": [14, 25]},
}
EXPECTED_NATIVE_SPEEDS = {
    ("target_moves", "source1"): 0.850134491,
    ("distractor_moves", "source2"): 0.850134492,
    ("both_move", "source1"): 0.828940353,
    ("both_move", "source2"): 0.731776998,
}


def _source_directory(mechanism: str) -> Path:
    local = LOCAL_SOURCE_ROOT / mechanism
    if local.is_dir():
        return local
    authoritative = AUTHORITATIVE_A_SOURCE_DIRECTORIES[mechanism]
    if authoritative.is_dir():
        return authoritative
    raise FileNotFoundError(
        f"no local or authoritative source directory for {mechanism}: "
        f"{local}, {authoritative}"
    )


def _materialize_minimal_source_fixture(destination: Path) -> None:
    for mechanism in BUILDER.CASE_ORDER:
        source = _source_directory(mechanism)
        case_destination = destination / mechanism
        case_destination.mkdir(parents=True, exist_ok=False)
        for filename in SOURCE_INPUT_FILENAMES:
            source_file = source / filename
            if not source_file.is_file():
                raise FileNotFoundError(
                    f"authoritative motion input is missing: {source_file}"
                )
            shutil.copy2(source_file, case_destination / filename)


class NativeRateDynamicCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._source_temporary = tempfile.TemporaryDirectory()
        cls.source_root = Path(cls._source_temporary.name).resolve()
        _materialize_minimal_source_fixture(cls.source_root)
        cls.candidates = BUILDER.build_all(cls.source_root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._source_temporary.cleanup()

    def test_source_fixture_is_temporary_and_minimal(self) -> None:
        self.assertNotIn(".source_inputs", self.source_root.parts)
        for mechanism in BUILDER.CASE_ORDER:
            case_files = {
                path.name for path in (self.source_root / mechanism).iterdir()
            }
            self.assertEqual(case_files, set(SOURCE_INPUT_FILENAMES))

    def test_three_candidates_are_cpu_pass_but_release_blocked(self) -> None:
        self.assertEqual(tuple(self.candidates), BUILDER.CASE_ORDER)
        for mechanism, (preflight, receipt) in self.candidates.items():
            with self.subTest(mechanism=mechanism):
                self.assertEqual(preflight["status"], "RELEASE_BLOCKED")
                self.assertEqual(receipt["status"], "RELEASE_BLOCKED")
                self.assertEqual(
                    preflight["cpu_candidate_preflight_status"],
                    "PASS_CPU_NATIVE_RATE_CANDIDATE",
                )
                self.assertFalse(preflight["release_qualified"])
                self.assertFalse(preflight["qualification_claim"])
                self.assertFalse(preflight["formal"])
                self.assertEqual(preflight["formal_episode_count"], 0)
                self.assertFalse(preflight["gpu_used"])
                self.assertFalse(preflight["gpu_launch_authorized"])
                self.assertEqual(
                    receipt["first_blocker"]["code"],
                    "fresh_native_pixels_not_verified",
                )
                blocker_codes = [item["code"] for item in receipt["release_blockers"]]
                self.assertEqual(
                    blocker_codes,
                    [
                        "fresh_native_pixels_not_verified",
                        "live_ground_contact_not_verified",
                        "live_foot_plant_sync_not_verified",
                        "live_walking_asset_readback_not_verified",
                        "retimed_exact_rir_not_built",
                    ],
                )
                VALIDATOR.validate_pair(
                    preflight,
                    receipt,
                    source_directory=self.source_root / mechanism,
                    replay=True,
                )

    def test_full75_has_exact_five_second_frame_coverage(self) -> None:
        for mechanism, (preflight, _) in self.candidates.items():
            with self.subTest(mechanism=mechanism):
                self.assertEqual(preflight["frame_count"], 75)
                self.assertEqual(preflight["frame_rate_hz"], 15)
                self.assertEqual(preflight["episode_duration_seconds"], 5.0)
                self.assertAlmostEqual(preflight["last_frame_pts_seconds"], 74 / 15)
                self.assertEqual(preflight["frame_coverage_end_seconds"], 5.0)
                self.assertEqual(
                    [frame["frame_index"] for frame in preflight["frames"]],
                    list(range(75)),
                )
                self.assertEqual(preflight["frames"][-1]["pts_ticks"], 74 * 3200)
                self.assertEqual(
                    preflight["frames"][-1]["frame_coverage_end_ticks"],
                    75 * 3200,
                )

    def test_native_windows_are_not_stretched_and_idle_holds_boundaries(self) -> None:
        for mechanism, expected_by_slot in EXPECTED_INTERVALS.items():
            preflight, _ = self.candidates[mechanism]
            for slot, expected_range in expected_by_slot.items():
                with self.subTest(mechanism=mechanism, slot=slot):
                    actor = preflight["actors"][slot]
                    interval = actor["native_rate_active_interval"]
                    self.assertEqual(
                        interval["output_frame_range_inclusive"], expected_range
                    )
                    start, end = expected_range
                    native_start, native_end = interval[
                        "native_source_frame_range_inclusive"
                    ]
                    self.assertEqual(end - start, native_end - native_start)
                    self.assertEqual(
                        interval["output_sample_count"],
                        interval["native_sample_count"],
                    )
                    self.assertEqual(interval["time_scale"], 1.0)
                    self.assertFalse(interval["global_time_stretch_applied"])
                    self.assertEqual(actor["action_id_path"][:start], ["idle"] * start)
                    self.assertEqual(
                        actor["action_id_path"][start : end + 1],
                        ["walk"] * (end - start + 1),
                    )
                    self.assertEqual(
                        actor["action_id_path"][end + 1 :],
                        ["idle"] * (74 - end),
                    )
                    first = actor["root_path_m"][start]
                    last = actor["root_path_m"][end]
                    self.assertTrue(
                        all(point == first for point in actor["root_path_m"][:start])
                    )
                    self.assertTrue(
                        all(point == last for point in actor["root_path_m"][end + 1 :])
                    )
                    self.assertEqual(actor["root_path_m"][start - 1], first)
                    self.assertEqual(actor["root_path_m"][end + 1], last)
                    self.assertEqual(
                        actor["trajectory_preflight"][
                            "pre_active_action_transition_root_step_m"
                        ],
                        0.0,
                    )
                    self.assertEqual(
                        actor["trajectory_preflight"][
                            "post_active_action_transition_root_step_m"
                        ],
                        0.0,
                    )

    def test_native_speed_phase_and_ticks_close_for_all_four_moving_slots(self) -> None:
        observed: dict[tuple[str, str], float] = {}
        for mechanism, expected_by_slot in EXPECTED_INTERVALS.items():
            preflight, _ = self.candidates[mechanism]
            for slot, (start, end) in expected_by_slot.items():
                actor = preflight["actors"][slot]
                trajectory = actor["trajectory_preflight"]
                speed = trajectory["active_average_speed_m_s"]
                observed[(mechanism, slot)] = speed
                self.assertAlmostEqual(
                    speed,
                    EXPECTED_NATIVE_SPEEDS[(mechanism, slot)],
                    places=6,
                )
                self.assertGreaterEqual(speed, 0.73)
                self.assertLessEqual(speed, 0.851)
                self.assertAlmostEqual(trajectory["active_phase_cadence_hz"], 0.9375)
                self.assertAlmostEqual(
                    trajectory["phase_advance_per_frame_cycles"], 0.0625
                )
                ticks = actor["action_time_ticks_path"][start : end + 1]
                self.assertTrue(
                    all(
                        current - previous == 3200
                        for previous, current in pairwise(ticks)
                    )
                )
                self.assertEqual(
                    actor["native_source_frame_index_path"][start : end + 1],
                    list(
                        range(
                            actor["native_rate_active_interval"][
                                "native_source_frame_range_inclusive"
                            ][0],
                            actor["native_rate_active_interval"][
                                "native_source_frame_range_inclusive"
                            ][1]
                            + 1,
                        )
                    ),
                )
                self.assertEqual(
                    trajectory["foot_plant_sync_status"],
                    "pending_live_runtime_evidence",
                )
                self.assertEqual(
                    trajectory["ground_contact_status"],
                    "pending_live_runtime_evidence",
                )
                self.assertEqual(
                    trajectory["skeletal_pose_transition_continuity_status"],
                    "pending_live_runtime_blend_readback",
                )
        self.assertEqual(set(observed), set(EXPECTED_NATIVE_SPEEDS))

    def test_active_interval_placement_preserves_mechanism_during_speech(self) -> None:
        target_active = set(range(6, 33))
        target_speech = set(range(7, 32))
        self.assertTrue(target_speech <= target_active)

        distractor_active = set(range(21, 37))
        distractor_speech = set(range(7, 51))
        self.assertTrue(distractor_active <= distractor_speech)

        both, _ = self.candidates["both_move"]
        source1_active = set(range(14, 25))
        source2_active = set(range(14, 26))
        both_speech = set(range(7, 32))
        self.assertEqual(len(source1_active & source2_active & both_speech), 11)
        self.assertEqual(
            both["mechanism_preflight"]["both_moving_overlap_frame_count"],
            11,
        )

    def test_audio_event_program_is_byte_semantically_unchanged(self) -> None:
        for mechanism, (preflight, _) in self.candidates.items():
            with self.subTest(mechanism=mechanism):
                source = json.loads(
                    (
                        self.source_root / mechanism / "materialization_receipt.json"
                    ).read_text(encoding="utf-8")
                )
                contract = preflight["audio_event_contract"]
                self.assertEqual(contract["audio_program"], source["audio_program"])
                self.assertFalse(contract["sound_event_content_and_timing_modified"])
                self.assertFalse(contract["source_activation_modified"])
                self.assertFalse(contract["existing_exact_rir_reuse_authorized"])
                self.assertTrue(contract["fresh_exact_rir_required"])

    def test_static_projection_keeps_side_depth_and_separation(self) -> None:
        expected_target_sides = {
            "target_moves": "left",
            "distractor_moves": "left",
            "both_move": "right",
        }
        for mechanism, (preflight, receipt) in self.candidates.items():
            with self.subTest(mechanism=mechanism):
                projection = preflight["projection_preflight"]
                self.assertEqual(projection["status"], "PASS_CPU_STATIC_PINHOLE_ONLY")
                self.assertEqual(
                    projection["target_side"], expected_target_sides[mechanism]
                )
                self.assertEqual(receipt["projection_preflight"], projection)
                sides = []
                for slot in ("source1", "source2"):
                    metrics = projection["actors"][slot]
                    sides.append(metrics["observed_side"])
                    self.assertEqual(
                        metrics["observed_side"],
                        metrics["expected_side_from_legacy_static_camera_projection"],
                    )
                    self.assertGreaterEqual(metrics["dead_zone_margin_fraction"], 0.05)
                    self.assertGreaterEqual(metrics["camera_depth_m_range"][0], 1.3)
                    self.assertLessEqual(metrics["camera_depth_m_range"][1], 6.5)
                self.assertNotEqual(sides[0], sides[1])
                self.assertGreaterEqual(
                    projection["minimum_actor_horizontal_separation_m"], 1.0
                )

    def test_legacy_slow_motion_materializations_remain_rejected(self) -> None:
        for mechanism in BUILDER.CASE_ORDER:
            with self.subTest(mechanism=mechanism):
                receipt = LEGACY_BUILDER.build_receipt(self.source_root / mechanism)
                self.assertEqual(
                    receipt["status"], "reject_nonrelease_motion_realism_gate"
                )
                self.assertEqual(
                    receipt["release_classification"],
                    "nonrelease_pipeline_evidence_only",
                )
                for slot in receipt["moving_slots"]:
                    codes = [item["code"] for item in slot["blockers"]]
                    self.assertEqual(slot["status"], "reject")
                    self.assertIn(
                        "global_time_stretch_of_short_native_anchor_window", codes
                    )
                    self.assertIn("output_speed_outside_native_rate_envelope", codes)
                    self.assertLess(
                        slot["native_rate_facts"][
                            "observed_full75_average_root_speed_m_s"
                        ],
                        0.31,
                    )

    def test_validator_rejects_time_stretch_audio_mutation_and_root_jump(self) -> None:
        preflight, receipt = self.candidates["target_moves"]

        stretched = copy.deepcopy(preflight)
        stretched["actors"]["source1"]["native_rate_active_interval"][
            "global_time_stretch_applied"
        ] = True
        with self.assertRaisesRegex(RuntimeError, "native-rate active interval"):
            VALIDATOR.validate_pair(stretched, receipt)

        audio_mutation = copy.deepcopy(preflight)
        audio_mutation["audio_event_contract"]["audio_program"][
            "target_speech_start_sample"
        ] += 1
        with self.assertRaisesRegex(RuntimeError, "altered the authoritative audio"):
            VALIDATOR.validate_pair(
                audio_mutation,
                receipt,
                source_directory=self.source_root / "target_moves",
            )

        teleported = copy.deepcopy(preflight)
        start = teleported["actors"]["source1"]["native_rate_active_interval"][
            "output_frame_range_inclusive"
        ][0]
        teleported["actors"]["source1"]["root_path_m"][start - 1][0] -= 0.1
        teleported["actors"]["source1"]["translation_ue_cm_path"][start - 1][0] -= 10.0
        teleported["frames"][start - 1]["actor_states"][0]["translation_m"][0] -= 0.1
        teleported["frames"][start - 1]["actor_states"][0]["translation_ue_cm"][0] -= (
            10.0
        )
        with self.assertRaisesRegex(RuntimeError, "boundary roots|position jump"):
            VALIDATOR.validate_pair(teleported, receipt)

    def test_validator_rejects_release_upgrade_or_missing_blocker(self) -> None:
        preflight, receipt = self.candidates["both_move"]
        upgraded = copy.deepcopy(preflight)
        upgraded["status"] = "PASS"
        upgraded["release_qualified"] = True
        with self.assertRaisesRegex(RuntimeError, "release status"):
            VALIDATOR.validate_pair(upgraded, receipt)

        missing = copy.deepcopy(receipt)
        missing["release_blockers"] = missing["release_blockers"][:-1]
        with self.assertRaisesRegex(RuntimeError, "blocker closure"):
            VALIDATOR.validate_pair(preflight, missing)

    def test_cross_runtime_replay_only_tolerates_ieee_float_tails(self) -> None:
        self.assertTrue(
            VALIDATOR._replay_equivalent(
                {"speed": 0.8501344921715553, "nested": [1, "left", False]},
                {"speed": 0.8501344921715556, "nested": [1, "left", False]},
            )
        )
        self.assertFalse(
            VALIDATOR._replay_equivalent(
                {"speed": 0.8501344921715553},
                {"speed": 0.8501344921815553},
            )
        )
        self.assertFalse(VALIDATOR._replay_equivalent({"value": 1}, {"value": 1.0}))
        self.assertFalse(VALIDATOR._replay_equivalent({"value": 1.0}, {"other": 1.0}))

    def test_writer_is_no_clobber_and_emits_exact_six_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            written = BUILDER.write_all(output, self.candidates)
            self.assertEqual(len(written), 6)
            self.assertTrue(all(path.is_file() for path in written))
            self.assertEqual(
                sorted(path.name for path in written),
                sorted(
                    name
                    for stem in BUILDER.CASE_OUTPUT_STEMS.values()
                    for name in (f"{stem}.json", f"{stem}_receipt.json")
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                BUILDER.write_all(output, self.candidates)


if __name__ == "__main__":
    unittest.main()

_RETAINED_TMP_WORKSPACE = Path(__file__).resolve().parents[2] / "tmp"
# Guarding on tmp/ existing was wrong: running the engine in a
# checkout creates tmp/spear_instance_*, which made this look
# mounted and sent 49 tests into a run without their data.  The
# evidence mount signature is a lead_* workspace.
if not any(_RETAINED_TMP_WORKSPACE.glob("lead_*")):
    raise unittest.SkipTest(
        "no lead_* evidence workspace under the repository tmp "
        "directory, so this checkout does not carry the retained "
        "strict-two-human evidence"
    )

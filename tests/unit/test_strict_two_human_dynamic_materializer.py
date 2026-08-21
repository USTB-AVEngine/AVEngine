from __future__ import annotations

import importlib.util
import json
import math
import wave
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/materialize_strict_two_human_dynamic_canary.py"
SPEC = importlib.util.spec_from_file_location("dynamic_materializer", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
LEGACY_BATCH_ROOT = REPOSITORY / "tmp/lead_a_strict_two_human_full_episode_batch_v1"
LEGACY_AUDIO_AVAILABLE = (
    TOOL.CONTROLLED_SOUND_CONTENT_REGISTRY.is_file()
    and (TOOL.BASE_AUDIO / "audio_program.json").is_file()
)


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for item in value.values() for key in _nested_keys(item)
        }
    if isinstance(value, list):
        return {key for item in value for key in _nested_keys(item)}
    return set()


def test_legacy_cli_arguments_remain_available() -> None:
    args = TOOL.parse_args(
        [
            "--preflight",
            "legacy_preflight.json",
            "--canary-index",
            "3",
            "--base-suite",
            "legacy_suite.json",
            "--audio-template",
            "legacy_audio",
            "--motion-candidate",
            "legacy_candidate.json",
            "--output",
            "legacy_output",
        ]
    )

    assert args.preflight == Path("legacy_preflight.json")
    assert args.canary_index == 3
    assert args.base_suite == Path("legacy_suite.json")
    assert args.motion_candidate == Path("legacy_candidate.json")
    assert args.planning_manifest is None


def _audio_row(*, female_target: bool) -> dict[str, object]:
    if female_target:
        target = {
            "content_id": "cremad_mti_v1",
            "identity_key": "F",
            "runtime_asset_id": "lead_b_rocketbox_adults_female_adult_01_original_v1",
            "runtime_revision": "native_runtime_ue_v1",
            "sound_asset_id": "speech_cremad_1002_mti_neu_v1",
            "speech_frame_window_inclusive": [7, 50],
            "speech_sample_count": 45912,
            "transcript": "Maybe tomorrow it will be cold.",
            "voice_id": "cremad_actor_1002",
        }
        distractor = {
            "identity_key": "M",
            "runtime_asset_id": "rocketbox_human_male_adult_01_m5_1_candidate",
            "runtime_revision": "native_runtime_ue_v3",
        }
    else:
        target = {
            "content_id": "cremad_ieo_v1",
            "identity_key": "M",
            "runtime_asset_id": "rocketbox_human_male_adult_01_m5_1_candidate",
            "runtime_revision": "native_runtime_ue_v3",
            "sound_asset_id": "speech_cremad_1001_ieo_neu_v1",
            "speech_frame_window_inclusive": [7, 31],
            "speech_sample_count": 25626,
            "transcript": "It's eleven o'clock.",
            "voice_id": "cremad_actor_1001",
        }
        distractor = {
            "identity_key": "F",
            "runtime_asset_id": "lead_b_rocketbox_adults_female_adult_01_original_v1",
            "runtime_revision": "native_runtime_ue_v1",
        }
    return {
        "episode_id": "dynamic_test_episode",
        "target": target,
        "distractor": distractor,
    }


@pytest.mark.skipif(
    not LEGACY_AUDIO_AVAILABLE,
    reason="legacy controlled-audio workspace artifacts are not mounted",
)
@pytest.mark.parametrize(
    "field",
    ["speech_sample_rate_hz", "speech_channel_count", "speech_audio_uri"],
)
def test_explicit_null_audio_metadata_is_rejected(field: str) -> None:
    row = _audio_row(female_target=False)
    row["target"][field] = None
    with pytest.raises(RuntimeError, match="metadata drift"):
        TOOL._controlled_target_sound(row)


@pytest.mark.skipif(
    not LEGACY_AUDIO_AVAILABLE,
    reason="legacy controlled-audio workspace artifacts are not mounted",
)
def test_audio_program_has_exact_declared_activity_window(tmp_path: Path) -> None:
    row = _audio_row(female_target=False)
    TOOL._copy_audio_contracts(TOOL.BASE_AUDIO, tmp_path, row)
    _, target_audio = TOOL._controlled_target_sound(row)
    result = TOOL._validate_audio_contracts(
        tmp_path,
        target_audio=target_audio,
        expected_speech_window=[7, 31],
    )
    program = json.loads(
        (tmp_path / "controlled_audio_program/audio_program.json").read_text()
    )

    assert result["status"] == "pass"
    assert result["speech_frame_window_inclusive"] == [7, 31]
    assert result["dry_bus_activity_checks"] == {
        "frame_6_silent": True,
        "frame_7_active": True,
        "frame_31_active": True,
        "frame_32_silent": True,
        "source2_all_zero": True,
    }
    assert program["events"][0]["start_sample"] == 7595
    assert program["events"][0]["end_sample_exclusive"] == 33221


@pytest.mark.skipif(
    not LEGACY_AUDIO_AVAILABLE,
    reason="legacy controlled-audio workspace artifacts are not mounted",
)
def test_audio_program_supports_female_static_target_and_silent_moving_source2(
    tmp_path: Path,
) -> None:
    row = _audio_row(female_target=True)
    TOOL._copy_audio_contracts(TOOL.BASE_AUDIO, tmp_path, row)
    _, target_audio = TOOL._controlled_target_sound(row)
    result = TOOL._validate_audio_contracts(
        tmp_path,
        target_audio=target_audio,
        expected_speech_window=[7, 50],
    )
    root = tmp_path / "controlled_audio_program"
    program = json.loads((root / "audio_program.json").read_text())
    endpoints = json.loads((root / "source_endpoint_registry.json").read_text())
    sounds = json.loads((root / "sound_asset_registry.json").read_text())
    bindings = {
        endpoint["binding"]["entity_instance_id"]: endpoint["binding"]
        for endpoint in endpoints["source_endpoints"]
    }

    assert result["speech_frame_window_inclusive"] == [7, 50]
    assert result["target_active_sample_count"] == 45912
    assert result["dry_bus_activity_checks"] == {
        "frame_6_silent": True,
        "frame_7_active": True,
        "frame_50_active": True,
        "frame_51_silent": True,
        "source2_all_zero": True,
    }
    assert program["events"][0]["sound_asset_id"] == "speech_cremad_1002_mti_neu_v1"
    assert program["events"][0]["start_sample"] == 7595
    assert program["events"][0]["end_sample_exclusive"] == 53507
    assert sounds["sound_assets"][0]["dry_audio"]["sample_count"] == 45912
    assert bindings["source1"]["entity_asset_id"].startswith(
        "lead_b_rocketbox_adults_female"
    )
    assert bindings["source2"]["entity_asset_id"].startswith("rocketbox_human_male")


def test_materializer_publishes_only_failure_receipt_on_error(tmp_path: Path) -> None:
    output = tmp_path / "failed_materialization"
    with pytest.raises(FileNotFoundError):
        TOOL.materialize(
            preflight_path=tmp_path / "missing_preflight.json",
            canary_index=1,
            base_suite_path=tmp_path / "missing_suite.json",
            audio_template=tmp_path / "missing_audio",
            output=output,
        )

    assert sorted(path.name for path in output.iterdir()) == ["FAILED.json"]
    failure = json.loads((output / "FAILED.json").read_text())
    assert failure["status"] == "failed"
    assert failure["formal"] is False
    assert failure["qualification_claim"] is False
    assert not list(tmp_path.glob(".failed_materialization.staging.*"))


def test_materializer_atomic_publish_does_not_replace_racing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "racing_materialization"
    original_publish = TOOL.atomic_publish_directory

    def create_destination_then_publish(policy, staging, destination):
        destination.mkdir()
        (destination / "concurrent_marker").write_text("preserve")
        return original_publish(policy, staging, destination)

    monkeypatch.setattr(
        TOOL, "atomic_publish_directory", create_destination_then_publish
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        TOOL.materialize(
            preflight_path=tmp_path / "missing_preflight.json",
            canary_index=1,
            base_suite_path=tmp_path / "missing_suite.json",
            audio_template=tmp_path / "missing_audio",
            output=output,
        )

    assert sorted(path.name for path in output.iterdir()) == ["concurrent_marker"]
    assert (output / "concurrent_marker").read_text() == "preserve"
    assert not list(tmp_path.glob(".racing_materialization.staging.*"))


@pytest.mark.parametrize("kind", ["dangling_leaf", "symlink_parent"])
def test_materializer_rejects_symlinked_output_path(tmp_path: Path, kind: str) -> None:
    outside = tmp_path / "outside"
    if kind == "dangling_leaf":
        target = outside / "materialized"
        output = tmp_path / "output_link"
        output.symlink_to(target, target_is_directory=True)
    else:
        outside.mkdir()
        parent = tmp_path / "linked_parent"
        parent.symlink_to(outside, target_is_directory=True)
        output = parent / "materialized"
        target = outside / "materialized"

    with pytest.raises(RuntimeError, match="symlinks"):
        TOOL.materialize(
            preflight_path=tmp_path / "missing_preflight.json",
            canary_index=1,
            base_suite_path=tmp_path / "missing_suite.json",
            audio_template=tmp_path / "missing_audio",
            output=output,
        )
    assert not target.exists()


def test_arc_length_interpolation_binds_phase_and_forward_to_motion() -> None:
    path_length_m = 1.4735664534507704
    roots = [
        [
            path_length_m * index / 74.0,
            0.4,
            -4.0 * path_length_m * index / 74.0,
        ]
        for index in range(75)
    ]
    actual_length_m = sum(
        math.hypot(current[0] - previous[0], current[2] - previous[2])
        for previous, current in pairwise(roots)
    )
    phases = [(1.625 * index / 74.0) % 1.0 for index in range(75)]
    norm = math.hypot(1.0, -4.0)
    forward = [1.0 / norm, 0.0, -4.0 / norm]
    yaw = math.degrees(math.atan2(forward[0], forward[2])) % 360.0
    role = {
        "path_provenance": {
            "method": "arc_length_interpolation_of_native_polyline_v1",
            "interior_output_roots_exact_native_frame_readbacks": False,
            "endpoints_exact_native_readbacks": True,
            "output_root_count": 75,
            "output_unique_root_count_at_1mm": 75,
        },
        "per_frame_action_phase": phases,
        "per_frame_anatomical_forward_habitat_world": [forward] * 75,
        "per_frame_tangent_yaw_habitat_deg": [yaw] * 75,
    }

    timing = TOOL._arc_length_animation_timing(role=role, roots=roots)

    assert timing is not None
    assert timing["status"] == "pass"
    assert timing["mode"] == "arc_length_preserving_native_stride_v1"
    assert timing["phase_cycle_count"] == pytest.approx(1.625)
    assert timing["path_length_m"] == pytest.approx(actual_length_m)
    assert timing["action_time_ticks_path"][-1] == 83_200
    assert timing["maximum_segment_length_delta_m"] < 1.0e-12
    assert timing["maximum_forward_angular_error_deg"] < 1.0e-5
    assert timing["claim_boundary"].startswith("interior roots")


def test_arc_length_interpolation_rejects_repeated_roots() -> None:
    roots = [[float(index), 0.4, 0.0] for index in range(75)]
    roots[20] = roots[19]
    role = {
        "path_provenance": {
            "method": "arc_length_interpolation_of_native_polyline_v1",
            "interior_output_roots_exact_native_frame_readbacks": False,
            "endpoints_exact_native_readbacks": True,
            "output_root_count": 75,
            "output_unique_root_count_at_1mm": 75,
        },
        "per_frame_action_phase": [(1.625 * index / 74.0) % 1.0 for index in range(75)],
        "per_frame_anatomical_forward_habitat_world": [[1.0, 0.0, 0.0]] * 75,
        "per_frame_tangent_yaw_habitat_deg": [90.0] * 75,
    }

    with pytest.raises(RuntimeError, match="move every frame"):
        TOOL._arc_length_animation_timing(role=role, roots=roots)


def test_camera_pan_acoustics_require_one_state_per_orientation_and_slot() -> None:
    assert TOOL.LEGACY_CAMERA_PAN_ACOUSTICS == {
        "motion_case": "source1_static_source2_static_camera_pan",
        "per_slot_distinct": {"source1": 75, "source2": 75},
        "unique": 150,
        "reuse": 0,
    }


def test_camera_pan_sensor_rig_applies_75_unique_orientations() -> None:
    yaw_path = [52.0 + 6.0 * frame_index / 74.0 for frame_index in range(75)]
    row = {
        "episode_id": "strict2h_dynamic_canary_04_camera_pan_both_static_v2",
        "camera": {
            "translation_m": [-0.7, 1.471, 0.65],
            "yaw_path_deg": yaw_path,
        },
    }

    rig = TOOL._sensor_rig(row)

    rotations = [
        tuple(frame["world_from_rig"]["rotation_xyzw"]) for frame in rig["frames"]
    ]
    observed_yaws = [
        math.degrees(2.0 * math.atan2(rotation[1], rotation[3]))
        for rotation in rotations
    ]
    assert len(rotations) == 75
    assert len(set(rotations)) == 75
    assert observed_yaws[0] == pytest.approx(52.0)
    assert observed_yaws[-1] == pytest.approx(58.0)
    assert all(current > previous for previous, current in pairwise(observed_yaws))


@pytest.mark.parametrize(
    (
        "candidate_name",
        "preflight_relative",
        "base_suite_relative",
        "canary_index",
        "expected_actions",
        "expected_unique_rirs",
    ),
    [
        (
            "native_strict_two_human_target_moves_native_rate_candidate_v1.json",
            "dynamic_target_moves_v2_cpu_candidate_v1/target_moves_v2_preflight.json",
            "dynamic_target_moves_v2_materialized_v1/suite_execution_plan.json",
            1,
            {"source1": {"idle": 48, "walk": 27}, "source2": {"idle": 75, "walk": 0}},
            28,
        ),
        (
            "native_strict_two_human_distractor_moves_native_rate_candidate_v1.json",
            "dynamic_distractor_moves_v2_geometry_v1/distractor_moves_v2_preflight.json",
            "dynamic_distractor_moves_v2_materialized_v1/suite_execution_plan.json",
            2,
            {"source1": {"idle": 75, "walk": 0}, "source2": {"idle": 59, "walk": 16}},
            17,
        ),
        (
            "native_strict_two_human_both_move_native_rate_candidate_v1.json",
            "dynamic_both_move_v1_adapter_v1/preflight.json",
            "dynamic_both_move_v1_materialized_v1/suite_execution_plan.json",
            3,
            {"source1": {"idle": 64, "walk": 11}, "source2": {"idle": 63, "walk": 12}},
            23,
        ),
    ],
)
def test_real_motion_candidate_is_consumed_and_published_frame_by_frame(
    tmp_path: Path,
    candidate_name: str,
    preflight_relative: str,
    base_suite_relative: str,
    canary_index: int,
    expected_actions: dict[str, dict[str, int]],
    expected_unique_rirs: int,
) -> None:
    batch_root = REPOSITORY / "tmp/lead_a_strict_two_human_full_episode_batch_v1"
    candidate_path = REPOSITORY / "examples/qa" / candidate_name
    preflight_path = batch_root / preflight_relative
    base_suite_path = batch_root / base_suite_relative
    output = tmp_path / candidate_path.stem
    if not (
        LEGACY_AUDIO_AVAILABLE
        and preflight_path.is_file()
        and base_suite_path.is_file()
    ):
        pytest.skip("legacy canary workspace artifacts are not mounted")
    preflight_document = json.loads(preflight_path.read_text())
    planning_row = next(
        row
        for row in preflight_document["canaries"]
        if row["execution_order"] == canary_index
    )
    planning_path = tmp_path / f"{candidate_path.stem}_planning.json"
    planning_path.write_text(json.dumps({"episodes": [planning_row]}))

    receipt_path = TOOL.materialize(
        preflight_path=preflight_path,
        canary_index=canary_index,
        base_suite_path=base_suite_path,
        audio_template=TOOL.BASE_AUDIO,
        output=output,
        motion_candidate_path=candidate_path,
        planning_manifest_path=planning_path,
        planning_episode_id=planning_row["episode_id"],
    )

    receipt = json.loads(receipt_path.read_text())
    profile = json.loads((output / "actor_motion_profile.json").read_text())
    suite = json.loads((output / "suite_execution_plan.json").read_text())
    candidate = json.loads(candidate_path.read_text())
    frames = suite["scenarios"][0]["plan"]["frames"]
    assert receipt["suite_actor_root_application"]["action_counts"] == expected_actions
    assert receipt["actor_motion_profile"]["derived_action_counts"] == expected_actions
    assert receipt["actor_motion_profile"]["legacy_root_motion_inference_used"] is False
    assert receipt["planning_episode_authority"] == {
        "status": "pass_absolute_manifest_unique_episode_and_semantic_binding",
        "path": str(planning_path.resolve()),
        "json_pointer": "/episodes/0",
        "episode_id": planning_row["episode_id"],
        "mechanism": planning_row["mechanism"],
    }
    assert (
        receipt["actor_motion_profile"]["profile_content_sha256"]
        == profile["profile_content_sha256"]
    )
    comparison = receipt["dynamic_acoustics"]["actor_motion_profile_comparison"]
    assert comparison["status"] == "pass_actual_plan_matches_profile_expectation"
    assert comparison["compared_counts"] == {
        "stride_frames": 1,
        "requested_pair_state_count": 150,
        "unique_rir_job_count": expected_unique_rirs,
    }
    assert (
        comparison["normalized_actual_plan_sha256"]
        == comparison["profile_expected_plan_sha256"]
    )
    assert len(frames) == len(profile["frames"]) == 75
    for index, (suite_frame, profile_frame, candidate_frame) in enumerate(
        zip(frames, profile["frames"], candidate["frames"], strict=True)
    ):
        assert (
            suite_frame["canonical_motion_profile_frame_sha256"]
            == profile_frame["canonical_frame_sha256"]
        )
        assert profile_frame["frame_index"] == index
        for suite_state, candidate_state in zip(
            suite_frame["actor_states"], candidate_frame["actor_states"], strict=True
        ):
            assert all(
                suite_state[key] == value for key, value in candidate_state.items()
            )


def test_motion_mechanism_without_candidate_fails_closed(tmp_path: Path) -> None:
    batch_root = REPOSITORY / "tmp/lead_a_strict_two_human_full_episode_batch_v1"
    if not (
        batch_root
        / "dynamic_target_moves_v2_cpu_candidate_v1/target_moves_v2_preflight.json"
    ).is_file():
        pytest.skip("legacy canary workspace artifacts are not mounted")
    output = tmp_path / "missing_motion_candidate"
    with pytest.raises(RuntimeError, match="requires --motion-candidate"):
        TOOL.materialize(
            preflight_path=(
                batch_root
                / "dynamic_target_moves_v2_cpu_candidate_v1/target_moves_v2_preflight.json"
            ),
            canary_index=1,
            base_suite_path=(
                batch_root
                / "dynamic_target_moves_v2_materialized_v1/suite_execution_plan.json"
            ),
            audio_template=TOOL.BASE_AUDIO,
            output=output,
        )

    failure = json.loads((output / "FAILED.json").read_text())
    assert failure["status"] == "failed"
    assert "requires --motion-candidate" in failure["error"]


def test_global_planning_row_without_native_motion_authority_writes_no_output(
    tmp_path: Path,
) -> None:
    batch_root = REPOSITORY / "tmp/lead_a_strict_two_human_full_episode_batch_v1"
    planning_path = batch_root / "cpu_plan_global100_v2/manifest.json"
    if not planning_path.is_file():
        pytest.skip("legacy global planning workspace artifact is not mounted")
    manifest = json.loads(planning_path.read_text())
    planning_row = next(
        row for row in manifest["episodes"] if row["mechanism"] == "target_moves"
    )
    output = tmp_path / "must_not_exist"

    with pytest.raises(RuntimeError, match="lacks a generic native motion authority"):
        TOOL.materialize(
            preflight_path=(
                batch_root
                / "dynamic_target_moves_v2_cpu_candidate_v1/target_moves_v2_preflight.json"
            ),
            canary_index=1,
            base_suite_path=(
                batch_root
                / "dynamic_target_moves_v2_materialized_v1/suite_execution_plan.json"
            ),
            audio_template=TOOL.BASE_AUDIO,
            output=output,
            planning_manifest_path=planning_path,
            planning_episode_id=planning_row["episode_id"],
        )

    assert not output.exists()


def test_planning_runtime_mismatch_fails_before_output(tmp_path: Path) -> None:
    batch_root = REPOSITORY / "tmp/lead_a_strict_two_human_full_episode_batch_v1"
    preflight_path = (
        batch_root
        / "dynamic_target_moves_v2_cpu_candidate_v1/target_moves_v2_preflight.json"
    )
    if not preflight_path.is_file():
        pytest.skip("legacy canary workspace artifacts are not mounted")
    row = json.loads(preflight_path.read_text())["canaries"][0]
    row["target"]["runtime_revision"] = "forged_revision"
    planning_path = tmp_path / "planning.json"
    planning_path.write_text(json.dumps({"episodes": [row]}))
    output = tmp_path / "must_not_exist"

    with pytest.raises(
        RuntimeError, match="planning target runtime_revision binding drift"
    ):
        TOOL.materialize(
            preflight_path=preflight_path,
            canary_index=1,
            base_suite_path=(
                batch_root
                / "dynamic_target_moves_v2_materialized_v1/suite_execution_plan.json"
            ),
            audio_template=TOOL.BASE_AUDIO,
            output=output,
            motion_candidate_path=(
                REPOSITORY
                / "examples/qa/native_strict_two_human_target_moves_native_rate_candidate_v1.json"
            ),
            planning_manifest_path=planning_path,
            planning_episode_id=row["episode_id"],
        )

    assert not output.exists()


def test_counterfactual_source_scenarios_bind_each_native_human_path(
    tmp_path: Path,
) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {"scenario_id": "native_target"},
                    {"scenario_id": "native_distractor"},
                ]
            }
        ),
        encoding="utf-8",
    )
    row = {
        "source_suite": str(suite_path),
        "native_source_scenario_ids": ["native_target", "native_distractor"],
        "target": {"path_provenance": {"native_source_scenario_id": "native_target"}},
        "distractor": {
            "path_provenance": {"native_source_scenario_id": "native_distractor"}
        },
    }

    resolved = TOOL._source_scenarios(row)

    assert resolved["source1"]["scenario_id"] == "native_target"
    assert resolved["source2"]["scenario_id"] == "native_distractor"


def test_equal_arc_native_human_method_binds_animation_phase() -> None:
    roots = [[0.01 * index, 0.4, -3.0] for index in range(75)]
    role = {
        "path_provenance": {
            "method": "equal_arc_interpolation_of_exact_native_human_polyline_v1",
            "interior_output_roots_exact_native_frame_readbacks": False,
            "endpoints_exact_native_readbacks": True,
            "output_root_count": 75,
            "output_unique_root_count_at_1mm": 75,
        },
        "per_frame_action_phase": [(0.625 * index / 74.0) % 1.0 for index in range(75)],
        "per_frame_anatomical_forward_habitat_world": [[1.0, 0.0, 0.0]] * 75,
        "per_frame_tangent_yaw_habitat_deg": [90.0] * 75,
    }

    timing = TOOL._arc_length_animation_timing(role=role, roots=roots)

    assert timing is not None
    assert timing["status"] == "pass"
    assert timing["phase_cycle_count"] == pytest.approx(0.625)
    assert timing["path_provenance"]["method"].startswith("equal_arc")


def test_planning_only_path_builds_profile_without_legacy_preflight_or_base_suite(
    tmp_path: Path,
) -> None:
    fixture_path = REPOSITORY / "tests/unit/test_actor_motion_profile.py"
    fixture_spec = importlib.util.spec_from_file_location(
        "planning_motion_profile_fixture", fixture_path
    )
    assert fixture_spec is not None and fixture_spec.loader is not None
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)

    manifest_path, _, row = fixture_module._write_planning_fixture(tmp_path)
    planning_binding = TOOL.bind_planning_episode(
        planning_manifest_path=manifest_path,
        episode_id=row["episode_id"],
    )
    profile, base_suite = TOOL._planning_materialization_authorities(planning_binding)
    declarations, frames, roots, emitters, semantics = (
        TOOL._actor_materialization_from_profile(profile)
    )

    assert profile["schema"] == "avengine_actor_motion_profile_v2"
    assert (
        profile["authorities"]["planning_episode"]["value"]["episode_id"]
        == row["episode_id"]
    )
    assert base_suite["scenarios"][0]["scenario_id"] == "PENDING_PLANNING_EPISODE"
    serialized_shell = json.dumps(base_suite, sort_keys=True).lower()
    assert "native_static" not in serialized_shell
    assert "native_moving" not in serialized_shell
    assert "fixture_native_source" not in serialized_shell
    assert base_suite["authority"]["qualification_claim"] is False
    assert [declaration["actor_id"] for declaration in declarations] == [
        "source1_actor",
        "source2_actor",
    ]
    assert len(frames) == 75
    assert frames[19]["actor_states"][0]["action_phase"] == 0.0
    assert frames[19]["actor_states"][0]["action_time_ticks"] == 19 * 3200
    assert roots["source1"] == row["target"]["root_path_m"]
    assert emitters["source1"][0] == pytest.approx([0.0, 1.97, 3.0])
    assert semantics["source1"]["native_rate_action_segments"][0][
        "output_frame_range_inclusive"
    ] == [0, 74]


def test_planning_audio_keeps_exact_samples_and_uses_semantic_m7_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_path = REPOSITORY / "tests/unit/test_actor_motion_profile.py"
    fixture_spec = importlib.util.spec_from_file_location(
        "planning_audio_fixture", fixture_path
    )
    assert fixture_spec is not None and fixture_spec.loader is not None
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)
    _, _, row = fixture_module._write_planning_fixture(tmp_path)
    audio_path = tmp_path / "fixture_speech.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(np.full(45_912, 4_000, dtype="<i2").tobytes())
    event = dict(row["audio_program"]["target_event"])
    monkeypatch.setattr(
        TOOL, "_planning_target_audio", lambda _row: (event, audio_path)
    )
    output = tmp_path / "materialized"
    output.mkdir()

    TOOL._copy_audio_contracts(
        tmp_path / "legacy_base_audio_must_not_exist",
        output,
        row,
        planning_mode=True,
    )
    result = TOOL._validate_planning_audio_contracts(output, expected_event=event)

    assert result["start_sample"] == 7_467
    assert result["end_sample_exclusive"] == 53_379
    assert result["target_active_sample_count"] == 45_912
    generated = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (output / "controlled_audio_program").glob("*.json")
    ]
    forbidden = {
        "file_sha256",
        "input_sha256",
        "audio_sha256",
        "sidecar_sha256",
        "byte_size",
        "audio_byte_size",
        "sample_width_bytes",
    }
    assert not (_nested_keys(generated) & forbidden)

    request = TOOL._acoustic_execution_request(
        output=output,
        episode_id=row["episode_id"],
        rir_plan={
            "requested_pair_state_count": 150,
            "unique_rir_job_count": 76,
            "cache_reuse_count": 74,
        },
        target_sound_asset_id=row["target"]["sound_asset_id"],
        target_audio=audio_path,
        planning_mode=True,
    )
    argv = request["binaural_render_argv"]
    assert "--semantic-source-endpoint-registry" in argv
    assert "--semantic-sound-content-registry" in argv
    assert "--semantic-audio-binding" in argv
    assert "--source-endpoint-registry" not in argv
    assert "--sound-asset-registry" not in argv
    assert "--sound-audio" not in argv
    assert "--semantic-no-file-evidence" in request["rir_render_argv"]
    legacy_request = TOOL._acoustic_execution_request(
        output=output,
        episode_id=row["episode_id"],
        rir_plan={
            "requested_pair_state_count": 150,
            "unique_rir_job_count": 76,
            "cache_reuse_count": 74,
        },
        target_sound_asset_id=row["target"]["sound_asset_id"],
        target_audio=audio_path,
        planning_mode=False,
    )
    assert "--semantic-no-file-evidence" not in legacy_request["rir_render_argv"]


def test_planning_audio_rejects_joint_event_and_target_uri_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_path = REPOSITORY / "tests/unit/test_actor_motion_profile.py"
    fixture_spec = importlib.util.spec_from_file_location(
        "planning_uri_fixture", fixture_path
    )
    assert fixture_spec is not None and fixture_spec.loader is not None
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)
    _, _, row = fixture_module._write_planning_fixture(tmp_path)
    forged = "semantic://forged-but-mutually-consistent"
    row["target"]["speech_audio_uri"] = forged
    row["audio_program"]["target_event"]["source_audio_uri"] = forged
    catalog = {
        "assets": [
            {
                "sound_asset_id": row["target"]["sound_asset_id"],
                "content": {
                    "speaker_id": row["target"]["voice_id"],
                    "statement_id": row["target"]["content_id"],
                    "transcript": "fixture transcript",
                },
                "audio": {
                    "uri": "semantic://fixture_speech",
                    "sample_count": row["target"]["speech_sample_count"],
                    "sample_rate_hz": 16_000,
                    "channel_count": 1,
                },
            }
        ]
    }
    row["target"]["transcript"] = "fixture transcript"
    catalog_path = tmp_path / "controlled_sound_catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    monkeypatch.setattr(TOOL, "CONTROLLED_SOUND_CONTENT_REGISTRY", catalog_path)
    monkeypatch.setattr(TOOL, "CONTROLLED_AUDIO_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="controlled sound metadata drift"):
        TOOL._planning_target_audio(row)

from __future__ import annotations

import importlib.util
import json
import math
from itertools import pairwise
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/materialize_strict_two_human_dynamic_canary.py"
SPEC = importlib.util.spec_from_file_location("dynamic_materializer", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


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

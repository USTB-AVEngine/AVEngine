from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/bind_native_paper_balance_episode.py"
TOOL_SPEC = importlib.util.spec_from_file_location(
    "bind_native_paper_balance_episode", TOOL_PATH
)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
BINDER = importlib.util.module_from_spec(TOOL_SPEC)
TOOL_SPEC.loader.exec_module(BINDER)


def _case(
    spec_id: str,
    question_type: str,
    selectors: dict,
    answer: str,
) -> tuple[dict, dict]:
    return (
        {
            "spec_id": spec_id,
            "question_type": question_type,
            "selectors": selectors,
        },
        {
            "spec_id": spec_id,
            "question_type": question_type,
            "status": "pass",
            "answer": {"value": answer},
        },
    )


def _manifest(scenario_type: str) -> dict:
    return {
        "authoritative_capture_request": {
            "scenario_type": scenario_type,
            "target_source_slot_id": "source2",
        },
        "ffprobe": {
            "streams": [
                {"codec_type": "video", "nb_frames": "75"},
                {"codec_type": "audio", "channels": 2},
            ]
        },
    }


def _visible_frame(index: int, *, x: float = 900.0) -> dict:
    return {
        "frame_index": index,
        "state": "visible_clear",
        "visible_pixels": 100,
        "target_centroid_xy_px": [x, 360.0],
    }


def _stationary_inputs() -> tuple[dict, dict, list, list, dict, dict]:
    sound_id = "speech_cremad_1001_ieo_neu_v1"
    facts = {
        "episode_id": "border_collie_human__paper_balance_stationary_first_v1",
        "sound_events": [
            {
                "event_id": "source2_speech_000",
                "source_slot_id": "source2",
                "sound_asset_id": sound_id,
                "start_frame": 7,
                "end_frame": 32,
            }
        ],
        "tracks": {
            "instances": {
                "source2": {"moving": [False] * 75},
            }
        },
    }
    truth = {
        "resolution_hw": [720, 1280],
        "per_instance": {
            "source1": {"frames": [_visible_frame(index) for index in range(75)]},
            "source2": {"frames": [_visible_frame(index) for index in range(75)]},
        },
    }
    pairs = [
        _case(
            "QS-001",
            "appearance_to_speaking",
            {"appearance_field": "breed_id", "appearance_value": "border_collie"},
            "no",
        ),
        _case("QS-002", "who_spoke_first", {}, "source2"),
        _case(
            "QS-003",
            "speaking_while_moving",
            {"sound_asset_id": sound_id},
            "no",
        ),
    ]
    return (
        facts,
        truth,
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        {"source2_speech_000": {"sound_asset_id": sound_id}},
        _manifest("paper_balance_stationary_first"),
    )


def _right_inputs() -> tuple[dict, dict, list, list, dict, dict]:
    sound_id = "speech_cremad_1005_tie_neu_v1"
    facts = {
        "episode_id": "border_collie_human__paper_balance_right_entry_v1",
        "sound_events": [
            {
                "event_id": "source2_speech_000",
                "source_slot_id": "source2",
                "sound_asset_id": sound_id,
                "start_frame": 30,
                "end_frame": 74,
                "statement_id": "cremad_tie_v1",
                "transcript": "That is exactly what happened.",
            }
        ],
    }
    source2 = [
        {
            "frame_index": index,
            "state": "out_of_view",
            "visible_pixels": 0,
            "target_centroid_xy_px": None,
        }
        for index in range(23)
    ]
    source2.extend(_visible_frame(index, x=1269.8785) for index in range(23, 75))
    truth = {
        "resolution_hw": [720, 1280],
        "per_instance": {
            "source1": {"frames": [_visible_frame(index) for index in range(75)]},
            "source2": {"frames": source2},
        },
    }
    pairs = [
        _case(
            "QS-001",
            "offscreen_to_onscreen",
            {"target_instance_id": "source2"},
            "right",
        ),
        _case(
            "QS-002",
            "appearance_to_spoken_content",
            {"appearance_field": "sex_or_gender_label", "appearance_value": "male"},
            "That is exactly what happened.",
        ),
    ]
    return (
        facts,
        truth,
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        {"source2_speech_000": {"sound_asset_id": sound_id}},
        _manifest("offscreen_to_onscreen"),
    )


def test_stationary_variant_accepts_all_three_exact_answer_strata() -> None:
    facts, truth, specs, evaluations, bindings, manifest = _stationary_inputs()
    result = BINDER._validate_variant(
        variant="stationary",
        facts=facts,
        pixel_truth=truth,
        specs=specs,
        evaluations=evaluations,
        event_sound_bindings=bindings,
        manifest=manifest,
    )
    assert result["appearance_to_speaking"] == "no"
    assert result["who_spoke_first"] == "source2"
    assert result["speaking_while_moving"] == "no"


def test_stationary_variant_rejects_nonstationary_speech_window() -> None:
    facts, truth, specs, evaluations, bindings, manifest = _stationary_inputs()
    facts["tracks"]["instances"]["source2"]["moving"][20] = True
    with pytest.raises(RuntimeError, match="not stationary"):
        BINDER._validate_variant(
            variant="stationary",
            facts=facts,
            pixel_truth=truth,
            specs=specs,
            evaluations=evaluations,
            event_sound_bindings=bindings,
            manifest=manifest,
        )


def test_right_entry_accepts_right_edge_and_second_transcript() -> None:
    facts, truth, specs, evaluations, bindings, manifest = _right_inputs()
    result = BINDER._validate_variant(
        variant="right_entry",
        facts=facts,
        pixel_truth=truth,
        specs=specs,
        evaluations=evaluations,
        event_sound_bindings=bindings,
        manifest=manifest,
    )
    assert result["offscreen_to_onscreen"] == "right"
    assert result["first_visible_frame"] == 23
    assert result["appearance_to_spoken_content"] == "That is exactly what happened."


def test_right_entry_rejects_center_dead_zone_and_early_visibility() -> None:
    facts, truth, specs, evaluations, bindings, manifest = _right_inputs()
    truth["per_instance"]["source2"]["frames"][23]["target_centroid_xy_px"] = [
        640.0,
        360.0,
    ]
    with pytest.raises(RuntimeError, match="dead-zone"):
        BINDER._validate_variant(
            variant="right_entry",
            facts=facts,
            pixel_truth=truth,
            specs=specs,
            evaluations=evaluations,
            event_sound_bindings=bindings,
            manifest=manifest,
        )

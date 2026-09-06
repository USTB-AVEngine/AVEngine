from __future__ import annotations

from copy import deepcopy

import json

import pytest

from avengine.qa.unified_catalog import (
    CATALOG,
    _candidates_qa_01,
    _candidates_qa_03,
    generate_unified_questions,
    get_requirements,
    normalize_episode_bundle,
    structural_baselines,
    distractors_equal_gold,
    _P8_BASE_GENERATORS,
    _P8_CANDIDATES,
    _p8_apply_distractor_gate,
    _p8_facts_for_candidate,
    _p8_form_candidate_values,
    _attach_p8_structure,
    VISIBILITY_STATES,
)
from avengine.qa.unified_scoring import score_unified_item


def _fixture() -> dict:
    frame_count = 40
    colors = ("blue", "pink", "green", "white")
    positions: dict[str, list[list[float]]] = {}
    for index, actor_id in enumerate(("a0", "a1", "a2", "a3")):
        rows: list[list[float]] = []
        for frame in range(frame_count):
            if actor_id == "a0":
                if frame <= 2:
                    point = [2.0, 0.0, -4.0]
                elif frame <= 5:
                    point = [1.0, 0.0, -3.0]
                elif frame == 6:
                    point = [1.0, 0.0, -3.0]
                else:
                    point = [-5.0, 0.0, -2.0]
            elif actor_id == "a1":
                point = [5.0, 0.0, -2.0] if frame < 12 else [2.0, 0.0, -1.0]
            elif actor_id == "a2":
                point = [0.0, 0.0, 5.0]
            else:
                point = [0.0, 0.0, -5.0]
            rows.append(point)
        positions[actor_id] = rows

    actors = {
        actor_id: {
            "asset_id": f"asset_{actor_id}",
            "species_id": "human",
            "display_label": f"{color} actor",
            "realized_attributes": {
                "coat_profile": {"value": color},
            },
        }
        for actor_id, color in zip(("a0", "a1", "a2", "a3"), colors)
    }
    frame_readbacks = {
        "clock": {
            "frame_count": frame_count,
            "frame_rate_hz": 10,
            "sample_rate_hz": 16000,
            "sample_count": 64000,
        },
        "actors": {
            actor_id: [
                {"frame_index": frame, "position_m": position}
                for frame, position in enumerate(path)
            ]
            for actor_id, path in positions.items()
        },
        "emitters": {
            actor_id: [
                {"frame_index": frame, "position_m": position}
                for frame, position in enumerate(path)
            ]
            for actor_id, path in positions.items()
        },
    }
    visibility: dict[str, dict] = {}
    for actor_id in actors:
        frames = []
        for frame in range(frame_count):
            state = "visible_clear"
            if actor_id == "a0":
                if frame == 8:
                    state = "fully_occluded"
                elif frame == 10:
                    state = "visible_occluded"
                elif frame == frame_count - 1:
                    state = "fully_occluded"
            if actor_id == "a3" and frame < 3:
                state = "out_of_view"
            record = {"frame_index": frame, "state": state}
            if actor_id == "a3" and frame == 3:
                record["target_centroid_xy_px"] = [90.0, 50.0]
            if actor_id == "a0" and frame == 10:
                record["occluder_instance_ids"] = ["table"]
            if actor_id == "a1" and frame == 12:
                record["state"] = "visible_occluded"
                record["occluder_instance_ids"] = ["chair"]
            frames.append(record)
        visibility[actor_id] = {
            "frames": frames,
            "semantic_id": 100 + int(actor_id[1:]),
            "state_counts": {},
        }
    pixel_truth = {
        "schema": "avengine_qa_pixel_visibility_truth_v1",
        "status": "computed_modal_target_only_v1",
        "resolution_hw": [100, 100],
        "per_instance": visibility,
    }
    events = [
        {
            "event_id": "e0",
            "actor_id": "a0",
            "sound_asset_id": "s0",
            "start_sample": 3200,
            "end_sample_exclusive": 9600,
            "sound_class": "bark",
            "event_segmentation_status": "reviewed",
        },
        {
            "event_id": "e1",
            "actor_id": "a1",
            "sound_asset_id": "s1",
            "start_sample": 16000,
            "end_sample_exclusive": 22400,
            "sound_class": "speech",
            "transcript": "hello world",
        },
        {
            "event_id": "e2",
            "actor_id": "a2",
            "sound_asset_id": "s2",
            "start_sample": 19200,
            "end_sample_exclusive": 25600,
            "sound_class": "laugh",
            "transcript": "ha ha",
            "event_segmentation_status": "reviewed",
        },
        {
            "event_id": "e3",
            "actor_id": "a3",
            "sound_asset_id": "s3",
            "start_sample": 32000,
            "end_sample_exclusive": 36800,
            "sound_class": "whistle",
            "transcript": "tweet",
            "event_segmentation_status": "reviewed",
        },
    ]
    return {
        "episode_id": "unified_fixture",
        "actors": actors,
        "frame_readbacks": {
            **frame_readbacks,
            "camera": [
                {"frame_index": frame, "position_m": [0.0, 0.0, 0.0], "yaw_deg": 0.0}
                for frame in range(frame_count)
            ],
        },
        "pixel_visibility_truth": pixel_truth,
        "appearance_review": {
            "status": "reviewed",
            "actors": {
                actor_id: {
                    "status": "reviewed",
                    "value": color,
                    "frame_refs": [0, 10, 20],
                }
                for actor_id, color in zip(
                    ("a0", "a1", "a2", "a3"),
                    colors,
                )
            },
        },
        "audio_program": {
            "timeline": {
                "time_base_hz": 48000,
                "sample_rate_hz": 16000,
                "frame_count": frame_count,
                "sample_count": 64000,
            },
            "events": events,
        },
        "audio_readback": {
            "channel_count": 2,
            "sample_rate_hz": 16000,
            "sample_count": 64000,
            "proof": "fixture_readback",
            "channel_order": ["left", "right"],
            "hrtf_id": "fixture_hrtf",
            "wet_tail_intervals": [
                {"event_id": "e0", "start_s": 0.2, "end_s": 0.61},
                {"event_id": "e1", "start_s": 1.0, "end_s": 1.41},
                {"event_id": "e2", "start_s": 1.2, "end_s": 1.61},
                {"event_id": "e3", "start_s": 2.0, "end_s": 2.31},
            ],
        },
        "occluder_registry": {
            "table": "table",
            "chair": "chair",
        },
    }


def test_catalog_is_complete_and_requirements_are_pre_capture() -> None:
    assert [item["qa_id"] for item in CATALOG] == [
        f"QA-{index:02d}" for index in range(1, 25)
    ]
    requirements = get_requirements("qa_13")
    assert requirements["qa_id"] == "QA-13"
    assert requirements["min_entities"] == 2
    assert requirements["events"]["post_sound_query"] is True
    for key in (
        "min_entities",
        "events",
        "appearance",
        "speech_content",
        "motion",
        "after_sound",
        "pixel_visibility",
        "occlusion",
        "entry",
        "distinct_sound_classes",
    ):
        assert key in requirements["potential_requirements"]
        assert key in requirements["evidence_requirements"]


def test_normalize_accepts_native_shapes_without_old_fact_wrapper() -> None:
    facts = normalize_episode_bundle(_fixture())
    assert facts["schema"] == "avengine_qa_unified_episode_facts_v1"
    assert facts["time"]["frame_count"] == 40
    assert set(facts["actors"]) == {"a0", "a1", "a2", "a3"}
    assert facts["audio"]["status"] == "pass"
    assert facts["listener"]["source"] == "frame_readbacks.camera"
    assert facts["events"][0]["start_frame"] == 2


def test_normalize_retains_explicit_sparse_pixel_frames_without_filling_gaps() -> None:
    raw = _fixture()
    for entry in raw["pixel_visibility_truth"]["per_instance"].values():
        entry["frames"] = [entry["frames"][0], entry["frames"][2]]
    facts = normalize_episode_bundle(raw)
    assert set(facts["visibility"]["a0"]) == {0, 2}
    assert set(facts["visibility"]["a3"]) == {0, 2}


def test_normalize_rejects_sparse_pixel_rows_without_explicit_frame_index() -> None:
    raw = _fixture()
    entry = raw["pixel_visibility_truth"]["per_instance"]["a0"]
    entry["frames"] = [entry["frames"][0], entry["frames"][2]]
    entry["frames"][0].pop("frame_index")
    facts = normalize_episode_bundle(raw)
    assert "a0" not in facts["visibility"]


def test_normalize_preserves_complete_legacy_ordinal_pixel_arrays() -> None:
    raw = _fixture()
    for entry in raw["pixel_visibility_truth"]["per_instance"].values():
        for row in entry["frames"]:
            row.pop("frame_index")
    facts = normalize_episode_bundle(raw)
    assert set(facts["visibility"]["a0"]) == set(range(40))


def test_qa07_does_not_treat_sparse_nonadjacent_frames_as_an_entry_transition() -> None:
    raw = _fixture()
    raw["pixel_visibility_truth"]["per_instance"]["a3"]["frames"] = [
        {"frame_index": 0, "state": "out_of_view"},
        {
            "frame_index": 2,
            "state": "visible_clear",
            "target_centroid_xy_px": [90.0, 50.0],
        },
    ]
    result = generate_unified_questions(raw, qa_ids=["QA-07"])
    assert result["counts"] == {"requested": 1, "valid": 0, "deferred": 1}
    assert result["deferred"][0]["code"] == "no_entry_transition"


def test_generation_restores_visibility_frame_keys_after_json_round_trip() -> None:
    raw = _fixture()
    for actor_id, state in zip(
        ("a0", "a1", "a2", "a3"),
        ("visible_clear", "visible_occluded", "fully_occluded", "out_of_view"),
    ):
        event = next(
            event
            for event in raw["audio_program"]["events"]
            if event["actor_id"] == actor_id
        )
        raw["pixel_visibility_truth"]["per_instance"][actor_id]["frames"][
            int(event["start_sample"] / 1600)
        ]["state"] = state
    normalized = normalize_episode_bundle(raw)
    round_tripped = json.loads(json.dumps(normalized))
    result = generate_unified_questions(round_tripped, qa_ids=["QA-08"])
    assert result["counts"] == {"requested": 1, "valid": 1, "deferred": 0}
    assert 0 <= result["items"][0]["evidence"]["query_frame"] < 40
def test_qa13_generated_open_form_uses_scorer_angle_convention() -> None:
    raw = _fixture()
    points = {
        "a0": [3.83, 0.0, -3.21],
        "a1": [-3.83, 0.0, -3.21],
        "a2": [3.83, 0.0, 3.21],
        "a3": [-3.83, 0.0, 3.21],
    }
    for actor_id, point in points.items():
        for stream in ("actors", "emitters"):
            for frame in raw["frame_readbacks"][stream][actor_id]:
                frame["position_m"] = list(point)
    result = generate_unified_questions(raw, qa_ids=["QA-13"])
    assert result["counts"] == {"requested": 1, "valid": 0, "deferred": 1}
    assert result["deferred"][0]["code"] == "target_unobservable_at_query"
    assert result["deferred"][0]["open_and_mcq_deferred"] is True
def test_ue_camera_basis_maps_forward_and_right_without_a_90_degree_bias() -> None:
    raw = _fixture()
    camera = raw["frame_readbacks"]["camera"]
    for record in camera:
        record["rotation_deg"] = [0.0, 0.0, -90.0]
    facts = normalize_episode_bundle(raw)
    basis = facts["listener"]["basis_m3"][0]
    assert basis["forward"] == pytest.approx([0.0, 0.0, -1.0])
    assert basis["right"] == pytest.approx([1.0, 0.0, 0.0])


def test_generation_reports_valid_rows_and_exact_missing_conditions() -> None:
    result = generate_unified_questions(_fixture(), seed="fixture")
    assert result["counts"]["requested"] == 24
    assert result["counts"]["valid"] + result["counts"]["deferred"] == 24
    by_id = {item["qa_id"]: item for item in result["items"]}
    if "QA-13" in by_id:
        assert by_id["QA-13"]["evidence"]["target_unobservable_at_query"] is False
        assert by_id["QA-13"]["form_status"]["mcq"]["status"] == "deferred"
    else:
        deferred = next(row for row in result["deferred"] if row["qa_id"] == "QA-13")
        assert deferred["code"] in {"target_unobservable_at_query", "post_sound_angle_not_separated"}
    assert by_id["QA-16"]["truth"]["value"] == "farther"
    assert by_id["QA-17"]["truth"]["value"] == "yes"
    assert by_id["QA-22"]["truth"]["value"] == [4, 4]
    assert by_id["QA-23"]["truth"]["value"] == [4]
    assert by_id["QA-24"]["truth"]["value"] == "fully_occluded"
    for option in by_id["QA-03"]["model_input"]["mcq"]["options"]:
        assert "value" not in option
        assert "source" not in option["label_en"].casefold()
def test_missing_pixel_truth_is_deferred_without_fabricating_states() -> None:
    raw = _fixture()
    raw.pop("pixel_visibility_truth")
    result = generate_unified_questions(raw, qa_ids=["QA-07", "QA-08", "QA-24"])
    assert result["counts"] == {"requested": 3, "valid": 0, "deferred": 3}
    assert {item["code"] for item in result["deferred"]} == {
        "missing_visibility_resolution",
        "no_event_visibility",
        "missing_pixel_visibility",
    }


def test_unresolved_audio_binding_cannot_become_a_false_negative() -> None:
    raw = _fixture()
    raw["audio_program"]["events"][0].pop("actor_id")
    result = generate_unified_questions(raw, qa_ids=["QA-03", "QA-23"])
    assert [item["qa_id"] for item in result["items"]] == ["QA-23"]
    assert result["deferred"][0]["code"] == "unresolved_event_attribution"


def test_non_speech_event_count_requires_explicit_segmentation_review() -> None:
    raw = _fixture()
    raw["audio_program"]["events"][0].pop("event_segmentation_status")
    result = generate_unified_questions(raw, qa_ids=["QA-23"])
    assert result["counts"] == {"requested": 1, "valid": 0, "deferred": 1}
    assert result["deferred"][0]["code"] == "event_segmentation_not_reviewed"


def test_post_sound_query_skips_wet_tail_and_programmed_events() -> None:
    raw = _fixture()
    raw["audio_readback"]["wet_tail_intervals"][0]["end_s"] = 0.9
    points = {
        "a0": [3.83, 0.0, -3.21],
        "a1": [-3.83, 0.0, -3.21],
        "a2": [3.83, 0.0, 3.21],
        "a3": [-3.83, 0.0, 3.21],
    }
    for actor_id, point in points.items():
        for stream in ("actors", "emitters"):
            for frame in raw["frame_readbacks"][stream][actor_id]:
                frame["position_m"] = list(point)
    # Contrast belongs to the actual legal post-sound query window (17+),
    # rather than to this competitor's unrelated earlier sound event.
    for frame in raw["frame_readbacks"]["actors"]["a0"]:
        if frame["frame_index"] >= 18:
            frame["position_m"][1] = 0.2
    result = generate_unified_questions(raw, qa_ids=["QA-13", "QA-17"])
    assert result["counts"]["valid"] == 1
    assert result["counts"]["deferred"] == 1
    assert result["deferred"][0]["qa_id"] == "QA-13"
    assert result["deferred"][0]["code"] == "target_unobservable_at_query"
    assert result["items"][0]["qa_id"] == "QA-17"
def test_qa13_does_not_treat_offscreen_competitor_as_a_different_band() -> None:
    facts = _two_actor_direction_facts(0.0, 90.0)
    facts["events"] = [event for event in facts["events"] if event["actor_id"] == "a0"]
    result = generate_unified_questions(facts, qa_ids=["QA-13"], seed="offscreen-competitor")
    assert result["counts"]["valid"] == 1
    item = result["items"][0]
    assert "open" in item["forms"] and "mcq" not in item["forms"]
    assert item["form_status"]["mcq"]["code"] == "candidate_value_missing"


def test_qa13_samples_distinct_legal_post_sound_query_frames() -> None:
    result = generate_unified_questions(_two_actor_direction_facts(32.0, -32.0),
        qa_ids=["QA-13"], seed="queries", items_per_type=4)
    assert len(result["items"]) == 4
    assert len({item["question_id"] for item in result["items"]}) == 4
    assert len({item["evidence"]["query_frame"] for item in result["items"]}) > 1
    for item in result["items"]:
        assert len(item["forms"]["mcq"]["options"]) == 3
        assert "independent sound event" in item["model_input"]["mcq"]["question_en"]
        assert f"video frame {item['evidence']['query_frame']}" in item["model_input"]["mcq"]["question_en"]


def test_missing_wet_tail_evidence_defers_temporal_queries() -> None:
    raw = _fixture()
    raw["audio_readback"].pop("wet_tail_intervals")
    result = generate_unified_questions(raw, qa_ids=["QA-13", "QA-16", "QA-17"])
    assert result["counts"] == {"requested": 3, "valid": 0, "deferred": 3}
    assert {item["code"] for item in result["deferred"]} == {
        "no_valid_post_sound_window"
    }


def test_occlusion_questions_emit_no_only_when_the_trigger_exists() -> None:
    raw = _fixture()
    full_frames = raw["pixel_visibility_truth"]["per_instance"]["a0"]["frames"]
    for frame in full_frames[9:]:
        frame["state"] = "fully_occluded"
    partial_frames = raw["pixel_visibility_truth"]["per_instance"]["a1"]["frames"]
    partial_frames[9]["state"] = "fully_occluded"
    partial_frames[10]["state"] = "visible_occluded"
    partial_frames[11]["state"] = "visible_clear"
    result = generate_unified_questions(raw, qa_ids=["QA-09", "QA-11"])
    assert result["counts"] == {"requested": 2, "valid": 2, "deferred": 0}
    by_id = {item["qa_id"]: item for item in result["items"]}
    assert by_id["QA-09"]["truth"]["value"] in {"yes", "no"}
    assert by_id["QA-11"]["truth"]["value"] in {"yes", "no"}


def test_occlusion_negative_answers_require_complete_visibility_coverage() -> None:
    raw = _fixture()
    raw["pixel_visibility_truth"]["per_instance"]["a0"]["frames"] = [
        {"frame_index": 8, "state": "fully_occluded"}
    ]
    qa09 = generate_unified_questions(raw, qa_ids=["QA-09"])
    assert qa09["counts"] == {"requested": 1, "valid": 0, "deferred": 1}
    assert qa09["deferred"][0]["code"] == "incomplete_visibility_for_negative"

    for actor_id, entry in raw["pixel_visibility_truth"]["per_instance"].items():
        if actor_id != "a0":
            for frame in entry["frames"]:
                frame["state"] = "visible_clear"
    raw["pixel_visibility_truth"]["per_instance"]["a0"]["frames"] = [
        {"frame_index": 10, "state": "visible_occluded"}
    ]
    qa11 = generate_unified_questions(raw, qa_ids=["QA-11"])
    assert qa11["counts"] == {"requested": 1, "valid": 0, "deferred": 1}
    assert qa11["deferred"][0]["code"] == "incomplete_visibility_for_negative"


def test_qa22_counts_observed_visible_entities_and_speakers_only() -> None:
    raw = _fixture()
    for actor_id, entry in raw["pixel_visibility_truth"]["per_instance"].items():
        if actor_id == "a0":
            entry["frames"] = [{"frame_index": 0, "state": "visible_clear"}]
        else:
            entry["frames"] = [
                {"frame_index": frame, "state": "out_of_view"}
                for frame in range(40)
            ]
    result = generate_unified_questions(raw, qa_ids=["QA-22"])
    assert result["counts"] == {"requested": 1, "valid": 1, "deferred": 0}
    item = result["items"][0]
    assert item["truth"]["value"] == [1, 1]
    assert item["evidence"]["appeared_actor_ids"] == ["a0"]
    assert item["evidence"]["speaking_actor_ids"] == ["a0"]


def test_qa22_defers_when_an_actor_has_unobserved_frames() -> None:
    raw = _fixture()
    raw["pixel_visibility_truth"]["per_instance"]["a1"]["frames"] = [
        {"frame_index": 0, "state": "out_of_view"}
    ]
    result = generate_unified_questions(raw, qa_ids=["QA-22"])
    assert result["counts"] == {"requested": 1, "valid": 0, "deferred": 1}
    assert result["deferred"][0]["code"] == "incomplete_visibility_for_entity_count"
    assert result["deferred"][0]["actor_ids"] == ["a1"]


def test_qa22_does_not_use_actor_roster_without_pixel_visibility() -> None:
    raw = _fixture()
    raw.pop("pixel_visibility_truth")
    result = generate_unified_questions(raw, qa_ids=["QA-22"])
    assert result["counts"] == {"requested": 1, "valid": 0, "deferred": 1}
    assert result["deferred"][0]["code"] == "missing_visibility_for_entity_count"


def test_qa10_keeps_open_form_when_only_one_real_occluder_is_registered() -> None:
    raw = _fixture()
    raw["pixel_visibility_truth"]["per_instance"]["a1"]["frames"][12].pop(
        "occluder_instance_ids"
    )
    raw["occluder_registry"].pop("chair")
    result = generate_unified_questions(raw, qa_ids=["QA-10"])
    assert result["counts"] == {"requested": 1, "valid": 1, "deferred": 0}
    item = result["items"][0]
    assert item["form_status"]["open"]["status"] == "pass"
    assert item["form_status"]["mcq"]["status"] == "deferred"
    assert "mcq" not in item["forms"]

def test_p8_candidate_enumeration_and_structural_baseline_are_explicit() -> None:
    facts = _fixture()
    normalized = normalize_episode_bundle(facts)
    actor_candidates = _candidates_qa_01(normalized)
    assert len(actor_candidates) == 4
    assert len({row["candidate_id"] for row in actor_candidates}) == 4
    fixed_candidates = _candidates_qa_03(normalized)
    assert len(fixed_candidates) == 1
    assert fixed_candidates[0]["kind"] == "semantic_fixed"

    baseline = structural_baselines(
        {"a": "blue", "b": "blue", "c": "green"},
        "c",
    )
    assert baseline["gold_is_majority"] is False
    assert baseline["gold_is_unique_minority"] is True
    assert baseline["candidate_value_multiplicity"] == {"blue": 2, "green": 1}


def test_p8_sampling_executes_frame_zero_window_and_rejects_invalid_frame() -> None:
    raw = _fixture()
    raw["sampling"] = {"query_frame_by_qa": {"QA-14": 0}}
    result = generate_unified_questions(raw, qa_ids=["QA-14"], seed="frame-zero")
    assert result["counts"] == {"requested": 1, "valid": 1, "deferred": 0}
    assert result["items"][0]["evidence"]["query_frame"] == 0

    raw = _fixture()
    raw["sampling"] = {"query_frame_by_qa": {"QA-14": 999}}
    result = generate_unified_questions(raw, qa_ids=["QA-14"], seed="bad-frame")
    assert result["counts"] == {"requested": 1, "valid": 0, "deferred": 1}
    assert result["deferred"][0]["code"] == "query_frame_out_of_range"

    raw = _fixture()
    raw["sampling"] = {
        "query_frame_by_qa": {
            "QA-14": {
                "policy": "uniform_in_legal_window",
                "window_frames": [0, 3],
            }
        }
    }
    result = generate_unified_questions(raw, qa_ids=["QA-14"], seed="window")
    assert result["counts"]["valid"] == 1
    assert result["items"][0]["evidence"]["query_frame"] in {0, 1, 2}


def test_p8_question_id_contains_target_event_and_query_window() -> None:
    raw = _fixture()
    for actor_id, state in zip(
        ("a0", "a1", "a2", "a3"),
        ("visible_clear", "visible_occluded", "fully_occluded", "out_of_view"),
    ):
        event = next(
            event
            for event in raw["audio_program"]["events"]
            if event["actor_id"] == actor_id
        )
        raw["pixel_visibility_truth"]["per_instance"][actor_id]["frames"][
            int(event["start_sample"] / 1600)
        ]["state"] = state
    result = generate_unified_questions(raw, qa_ids=["QA-08", "QA-14"])
    by_qa = {item["qa_id"]: item for item in result["items"]}
    assert "target_" in by_qa["QA-08"]["question_id"]
    assert "event_" in by_qa["QA-08"]["question_id"]
    assert "frame_" in by_qa["QA-08"]["question_id"]
    assert "frame_" in by_qa["QA-14"]["question_id"]
    assert "time_" in by_qa["QA-14"]["question_id"]


def test_p8_qa13_uses_its_actual_three_band_domain_without_legacy_sector_gate() -> None:
    result = generate_unified_questions(_two_actor_direction_facts(25.0, -25.0), qa_ids=["QA-13"], seed="fov")
    assert result["counts"]["valid"] == 1
    item = result["items"][0]
    assert item["evidence"]["target_unobservable_at_query"] is False
    assert len(item["forms"]["mcq"]["options"]) == 3
    assert "open" not in item["forms"]
    assert item["form_status"]["open"]["code"] == "open_numeric_candidate_gap_too_small"
    assert "independent sound event" in item["model_input"]["mcq"]["question_en"]


def test_p8_conditioned_qa13_defers_mcq_outside_in_view_domain() -> None:
    raw = _fixture()
    raw["sampling_policy"] = "conditioned_static_v2"
    points = {
        "a0": [3.83, 0.0, -3.21],
        "a1": [-3.83, 0.0, -3.21],
        "a2": [3.83, 0.0, 3.21],
        "a3": [-3.83, 0.0, 3.21],
    }
    for actor_id, point in points.items():
        for stream in ("actors", "emitters"):
            for frame in raw["frame_readbacks"][stream][actor_id]:
                frame["position_m"] = list(point)
    result = generate_unified_questions(raw, qa_ids=["QA-13"], seed="fov-out")
    assert result["counts"] == {"requested": 1, "valid": 0, "deferred": 1}
    assert result["deferred"][0]["code"] == "target_unobservable_at_query"
    assert result["deferred"][0]["open_and_mcq_deferred"] is True
def test_p8_question_wording_and_transcript_metrics_are_explicit() -> None:
    raw = _fixture()
    raw["sampling"] = {"query_time_s_by_qa": {"QA-18": 0.0}}
    raw["audio_readback"]["source_activity_intervals_samples"] = [
        {
            "event_id": event["event_id"],
            "start_sample": event["start_sample"],
            "end_sample_exclusive": event["end_sample_exclusive"],
        }
        for event in raw["audio_program"]["events"]
    ]
    result = generate_unified_questions(
        raw,
        qa_ids=["QA-12", "QA-17", "QA-18"],
        seed="wording",
    )
    by_qa = {item["qa_id"]: item for item in result["items"]}
    assert "video frame" in by_qa["QA-17"]["question"]["en"]
    assert "currently making a sound" in by_qa["QA-18"]["question"]["en"]
    qa12 = by_qa["QA-12"]
    assert qa12["evidence"]["transcript_attribution"]["match_required"] is True
    assert qa12["evidence"]["wer"]["metric"] == "word_error_rate"


def test_p8_qa22_uses_only_legal_speaking_count_options() -> None:
    raw = _fixture()
    for actor_id, entry in raw["pixel_visibility_truth"]["per_instance"].items():
        if actor_id in {"a0", "a1"}:
            entry["frames"] = [
                {"frame_index": frame, "state": "visible_clear"}
                for frame in range(40)
            ]
        else:
            entry["frames"] = [
                {"frame_index": frame, "state": "out_of_view"}
                for frame in range(40)
            ]
    result = generate_unified_questions(raw, qa_ids=["QA-22"])
    assert result["counts"]["valid"] == 1
    options = result["items"][0]["forms"]["mcq"]["options"]
    assert len(options) == 3
    assert {option["value"] for option in options} == {"2|0", "2|1", "2|2"}


def test_p8_multiple_items_have_unique_ids_and_list_coverage() -> None:
    raw = _fixture()
    raw["audio_program"]["events"] = raw["audio_program"]["events"][2:]
    result = generate_unified_questions(
        raw,
        qa_ids=["QA-01"],
        seed="many",
        items_per_type=2,
    )
    assert result["counts"]["valid"] == 2
    ids = [item["question_id"] for item in result["items"]]
    assert len(ids) == len(set(ids)) == 2
    assert len(result["coverage_by_qa"]["QA-01"]) == 2


def test_p8_private_structure_fields_do_not_enter_model_input() -> None:
    raw = _fixture()
    raw["audio_program"]["events"] = raw["audio_program"]["events"][2:]
    result = generate_unified_questions(raw, qa_ids=["QA-01"])
    model_input = result["items"][0]["model_input"]
    text = json.dumps(model_input, ensure_ascii=False)
    for private in ("candidate_id", "candidate_value_multiplicity", "gold_is_majority", "target_actor_id"):
        assert private not in text

def test_p8_selected_candidates_change_real_targets_and_pairs() -> None:
    raw = _fixture()
    raw["audio_program"]["events"] = raw["audio_program"]["events"][2:]
    target_ids = set()
    event_ids = set()
    pair_ids = set()
    for seed in (f"seed-{index}" for index in range(20)):
        qa01 = generate_unified_questions(raw, qa_ids=["QA-01"], seed=seed)
        qa02 = generate_unified_questions(raw, qa_ids=["QA-02"], seed=seed)
        qa14 = generate_unified_questions(raw, qa_ids=["QA-14"], seed=seed)
        target_ids.add(qa01["items"][0]["evidence"]["target_actor_id"])
        event_ids.add(qa02["items"][0]["evidence"]["event_id"])
        pair_ids.add(qa14["items"][0]["question_id"].rsplit("__", 1)[-1])
    assert len(target_ids) >= 2
    assert len(event_ids) >= 2
    assert len(pair_ids) >= 2


def test_p8_selected_candidates_drive_occlusion_targets() -> None:
    raw09 = _fixture()
    for frame in raw09["pixel_visibility_truth"]["per_instance"]["a1"]["frames"]:
        if frame["frame_index"] >= 12:
            frame["state"] = "fully_occluded"
    # Keep a separate fixture for QA-11: its a1 candidate must retain the
    # partial-occlusion trigger that QA-09 above intentionally replaces.
    raw11 = _fixture()
    raw11_frames = raw11["pixel_visibility_truth"]["per_instance"]["a1"]["frames"]
    raw11_frames[12]["state"] = "visible_occluded"
    for frame in raw11_frames[13:]:
        frame["state"] = "fully_occluded"
    reappearance_targets = set()
    clear_targets = set()
    for seed in (f"occlusion-{index}" for index in range(20)):
        reappearance = generate_unified_questions(raw09, qa_ids=["QA-09"], seed=seed)
        clear = generate_unified_questions(raw11, qa_ids=["QA-11"], seed=seed)
        reappearance_targets.add(
            reappearance["items"][0]["evidence"]["target_actor_id"]
        )
        clear_targets.add(clear["items"][0]["evidence"]["target_actor_id"])
    assert reappearance_targets >= {"a0", "a1"}
    assert clear_targets >= {"a0", "a1"}


def test_p8_fixed_semantics_keep_one_candidate_and_stable_identity() -> None:
    raw = _fixture()
    for qa_id in ("QA-03", "QA-22", "QA-23", "QA-24"):
        first = generate_unified_questions(raw, qa_ids=[qa_id], seed="fixed-0")
        assert first["candidate_counts"][qa_id] == 1
        ids = {
            generate_unified_questions(raw, qa_ids=[qa_id], seed=f"fixed-{index}")["items"][0]["question_id"]
            for index in range(5)
        }
        assert len(ids) == 1


def test_p8_structure_uses_entity_values_for_mcq_without_fabricating_missing() -> None:
    import avengine.qa.unified_catalog as catalog

    facts = normalize_episode_bundle(_fixture())
    candidate = _P8_CANDIDATES["QA-01"](facts)[0]
    candidate_facts = _p8_facts_for_candidate(facts, candidate, "structure")
    item = _P8_BASE_GENERATORS["QA-01"](candidate_facts, "structure")
    metadata = {
        "gold_actor": candidate["actor_id"],
        "form_candidate_values": _p8_form_candidate_values(
            "QA-01", item, candidate_facts
        ),
    }
    item = _attach_p8_structure(item, metadata)
    assert item["structure"]["mcq"]["candidate_value_multiplicity"] == {"yes": 4}
    assert item["structure"]["majority_refusal_applied"] is False

    qa02 = generate_unified_questions(_fixture(), qa_ids=["QA-02"])["items"][0]
    assert qa02["structure"]["mcq"]["candidate_value_multiplicity"] == {
        "blue": 1,
        "pink": 1,
        "green": 1,
        "white": 1,
    }


def test_p8_applicability_marks_animal_transcript_and_rigid_motion() -> None:
    animal = _fixture()
    for actor in animal["actors"].values():
        actor["species_id"] = "dog"
    animal_result = generate_unified_questions(animal, qa_ids=["QA-12"])
    assert animal_result["counts"] == {"requested": 1, "valid": 0, "deferred": 1}
    assert animal_result["deferred"][0]["code"] == "not_applicable_by_definition"

    rigid = _fixture()
    for actor in rigid["actors"].values():
        actor["species_id"] = "rigid_static_object"
    rigid_result = generate_unified_questions(rigid, qa_ids=["QA-06"])
    assert rigid_result["counts"] == {"requested": 1, "valid": 0, "deferred": 1}
    assert rigid_result["deferred"][0]["code"] == "not_applicable_by_definition"


def _two_actor_direction_facts(first_angle, second_angle):
    import math
    facts = normalize_episode_bundle(_fixture())
    ids = {"a0", "a1"}
    for field in ("actors", "visibility", "appearance_review"):
        facts[field] = {key: value for key, value in facts[field].items() if key in ids}
    facts["events"] = [e for e in facts["events"] if e["actor_id"] in ids]
    for actor_id, angle in zip(("a0", "a1"), (first_angle, second_angle)):
        point = [4*math.sin(math.radians(angle)), 0.0, -4*math.cos(math.radians(angle))]
        actor = facts["actors"][actor_id]
        actor["root_positions_m"] = [list(point) for _ in actor["root_positions_m"]]
        actor["emitter_positions_m"] = [list(point) for _ in actor["emitter_positions_m"]]
        for record in facts["visibility"][actor_id].values():
            record["state"] = "visible_clear"
    return facts


def test_event_pair_candidates_and_query_instants_do_not_duplicate_items():
    from avengine.qa.unified_catalog import _candidates_qa_05
    facts = normalize_episode_bundle(_fixture())
    assert len(_candidates_qa_05(facts)) == 6
    result = generate_unified_questions(facts, qa_ids=["QA-05", "QA-18"], items_per_type=3, seed="pairs")
    assert len({item["question_id"] for item in result["items"]}) == len(result["items"])
    pairs = [tuple(item["evidence"]["event_ids"]) for item in result["items"] if item["qa_id"] == "QA-05"]
    assert len(pairs) == len(set(pairs)) == 3
    qa18 = [item for item in result["items"] if item["qa_id"] == "QA-18"]
    assert len({item["evidence"]["query_frame"] for item in qa18}) == len(qa18)
    assert all("\\u" not in item["question"]["zh"] for item in qa18)


def test_quota_shortfall_is_reported_without_copying_a_pair():
    facts = _two_actor_direction_facts(32, -32)
    result = generate_unified_questions(facts, qa_ids=["QA-05"], items_per_type=3)
    assert len(result["items"]) == 1
    assert result["unmet_quota_by_qa"]["QA-05"] == {"requested":3,"valid":1,"missing":2,"code":"insufficient_candidates"}


def test_first_utterance_time_does_not_change_when_a_later_event_is_sampled():
    from copy import deepcopy
    facts = _two_actor_direction_facts(32, -32)
    repeated = deepcopy(facts["events"][0]);repeated.update(event_id="repeat", start_s=2.7, end_s=3.0, start_frame=27, end_frame=30)
    facts["events"].append(repeated)
    result = generate_unified_questions(facts, qa_ids=["QA-19"], items_per_type=3, seed="repeat")
    item = next(x for x in result["items"] if x["evidence"]["target_actor_id"] == "a0")
    assert item["evidence"]["first_event"]["event_id"] == "e0"
    assert item["truth"]["value"] == pytest.approx(.2)


def test_p8_form_candidate_values_follow_real_answer_domains_and_keep_missing() -> None:
    import avengine.qa.unified_catalog as catalog

    facts = normalize_episode_bundle(_fixture())
    domains = {
        "QA-06": {"moving", "still"},
        "QA-08": set(VISIBILITY_STATES),
        "QA-12": {"hello world", "ha ha", "tweet"},
        "QA-13": None,
        "QA-15": {"nearer", "farther"},
        "QA-16": {"nearer", "farther"},
        "QA-17": {"yes", "no"},
        "QA-19": None,
        "QA-21": {"speech", "bark", "laugh", "whistle"},
        "QA-24": set(VISIBILITY_STATES),
    }
    for qa_id, domain in domains.items():
        candidates = _P8_CANDIDATES[qa_id](facts)
        assert candidates, qa_id
        item = None
        # A light candidate may still fail its native-evidence predicate.
        # Check a candidate that actually emits itself, not a fallback event.
        for candidate in candidates:
            candidate_facts = _p8_facts_for_candidate(facts, candidate, "domains")
            try:
                emitted = _P8_BASE_GENERATORS[qa_id](candidate_facts, "domains")
            except catalog._Deferred:
                continue
            if catalog._p8_candidate_matches_item(emitted, candidate):
                item = emitted
                break
        if item is None:
            assert qa_id in {"QA-15", "QA-16", "QA-17"}
            continue
        values = _p8_form_candidate_values(qa_id, item, candidate_facts)
        assert set(values["open"]) == set(facts["actors"])
        assert all(value is None or value != "blue" for value in values["open"].values())
        if domain is not None:
            observed = {value for value in values["open"].values() if value is not None}
            assert observed <= domain, (qa_id, observed)
        if qa_id == "QA-13":
            observed = {value for value in values["open"].values() if value is not None}
            assert all(isinstance(value, float) for value in observed)
        if qa_id == "QA-19":
            assert all(
                value is None or isinstance(value, float)
                for value in values["open"].values()
            )
        if item.get("forms", {}).get("mcq"):
            observed_mcq = {
                value for value in values["mcq"].values() if value is not None
            }
            option_values = {
                str(option["value"])
                for option in item["forms"]["mcq"]["options"]
            }
            assert observed_mcq <= option_values


def test_p8_distractor_gate_is_per_form_and_preserves_exceptions() -> None:
    item = {
        "forms": {"open": {"answer_type": "closed_set"}, "mcq": {"options": []}},
        "model_input": {"open": {}, "mcq": {}},
        "form_status": {"open": {"status": "pass"}, "mcq": {"status": "pass"}},
    }
    candidate = {
        "gold_actor": "a0",
        "form_candidate_values": {
            "open": {"a0": "left", "a1": "left"},
            "mcq": {"a0": "band_0", "a1": "band_1"},
        },
    }
    _p8_apply_distractor_gate("QA-13", item, candidate)
    assert "open" not in item["forms"]
    assert "open" not in item["model_input"]
    assert item["form_status"]["open"]["code"] == "distractors_equal_gold"
    assert "mcq" in item["forms"]

    exempt = {
        "forms": {"open": {}, "mcq": {}},
        "model_input": {"open": {}, "mcq": {}},
        "form_status": {"open": {"status": "pass"}, "mcq": {"status": "pass"}},
    }
    _p8_apply_distractor_gate(
        "QA-01",
        exempt,
        {
            "gold_actor": "a0",
            "form_candidate_values": {
                "open": {"a0": "no", "a1": "no"},
                "mcq": {"a0": "no", "a1": "no"},
            },
        },
    )
    assert set(exempt["forms"]) == {"open", "mcq"}


def test_p8_qa14_enumerates_actor_pair_by_legal_query_frame() -> None:
    raw = _fixture()
    raw["sampling"] = {
        "query_frame_by_qa": {
            "QA-14": {
                "policy": "uniform_in_legal_window",
                "window_frames": [1, 4],
            }
        }
    }
    facts = normalize_episode_bundle(raw)
    candidates = _P8_CANDIDATES["QA-14"](facts)
    assert candidates
    assert {candidate["query_frame"] for candidate in candidates} <= {1, 2, 3}
    assert len({candidate["candidate_id"] for candidate in candidates}) == len(candidates)
    assert all(len(candidate["actor_ids"]) == 2 for candidate in candidates)


def test_p8_transition_and_occlusion_candidates_are_not_fixed_to_first_record() -> None:
    import avengine.qa.unified_catalog as catalog

    raw = _fixture()
    # QA-07: add a second real out-of-view -> visible transition on a2.
    a2 = raw["pixel_visibility_truth"]["per_instance"]["a2"]["frames"]
    a2[0]["state"] = "out_of_view"
    a2[1]["state"] = "visible_clear"
    a2[1]["target_centroid_xy_px"] = [10.0, 50.0]
    # QA-09: a0 reappears; a1 remains fully occluded after its trigger.
    a1 = raw["pixel_visibility_truth"]["per_instance"]["a1"]["frames"]
    for row in a1:
        if row["frame_index"] >= 12:
            row["state"] = "fully_occluded"
    # QA-11: a0 clears; a1 has a complete negative partial-occlusion record.
    a1[12]["state"] = "visible_occluded"
    for row in a1[13:]:
        row["state"] = "fully_occluded"

    facts = normalize_episode_bundle(raw)
    expected = {
        "QA-07": "actor_id",
        "QA-09": "actor_id",
        "QA-11": "actor_id",
        "QA-10": "actor_id",
    }
    for qa_id, field in expected.items():
        candidates = _P8_CANDIDATES[qa_id](facts)
        assert len(candidates) >= 2, qa_id
        assert len({candidate["candidate_id"] for candidate in candidates}) >= 2
        selected = set()
        for index in range(24):
            result = catalog.generate_unified_questions(
                raw, qa_ids=[qa_id], seed=f"transition-{qa_id}-{index}"
            )
            if result["items"]:
                selected.add(result["items"][0]["evidence"]["target_actor_id"])
        assert len(selected) >= 2, (qa_id, selected)


def test_p8_qa18_requires_source_activity_and_separates_wet_tail() -> None:
    raw = _fixture()
    missing = generate_unified_questions(raw, qa_ids=["QA-18"], seed="activity-missing")
    assert missing["counts"] == {"requested": 1, "valid": 0, "deferred": 1}
    assert missing["deferred"][0]["code"] == "missing_source_activity_readback"

    raw["audio_readback"]["source_activity_intervals_samples"] = [
        {
            "event_id": event["event_id"],
            "start_sample": event["start_sample"],
            "end_sample_exclusive": event["end_sample_exclusive"],
        }
        for event in raw["audio_program"]["events"]
    ]
    facts = normalize_episode_bundle(raw)
    assert facts["source_activity_evidence_present"] is True
    assert facts["source_activity_intervals_samples"]["e0"][0] == {
        "start_sample": 3200,
        "end_sample_exclusive": 9600,
    }

    event_records = _fixture()
    event_records["audio_readback"]["events"] = [
        {
            "event_id": event["event_id"],
            "source_activity_intervals_samples": [
                {
                    "start_sample": event["start_sample"],
                    "end_sample_exclusive": event["end_sample_exclusive"],
                }
            ],
        }
        for event in event_records["audio_program"]["events"]
    ]
    event_facts = normalize_episode_bundle(event_records)
    assert event_facts["source_activity_evidence_present"] is True
    assert event_facts["source_activity_intervals_samples"]["e1"][0] == {
        "start_sample": 16000,
        "end_sample_exclusive": 22400,
    }

    no_visual_review = json.loads(json.dumps(raw))
    no_visual_review["appearance_review"] = {}
    silent = generate_unified_questions(
        no_visual_review,
        qa_ids=["QA-18"],
        seed="activity-silent",
    )
    assert silent["counts"]["valid"] == 1
    assert silent["items"][0]["truth"]["value"] == "none"

    raw["sampling"] = {"query_time_s_by_qa": {"QA-18": 0.3}}
    wet = generate_unified_questions(raw, qa_ids=["QA-18"], seed="activity-wet")
    assert wet["counts"] == {"requested": 1, "valid": 0, "deferred": 1}
    assert wet["deferred"][0]["code"] == "query_inside_wet_tail"


def test_p8_open_and_mcq_use_distinct_angle_and_time_domains() -> None:
    directional = _two_actor_direction_facts(32.0, -32.0)
    qa13 = generate_unified_questions(
        directional,
        qa_ids=["QA-13"],
        seed="domain-angle",
    )
    assert qa13["items"]
    angle_item = qa13["items"][0]
    open_values = angle_item["structure"]["open"]["candidate_value_multiplicity"]
    mcq_values = angle_item["structure"]["mcq"]["candidate_value_multiplicity"]
    assert open_values
    assert all(
        value not in {"fov_band_0", "fov_band_1", "fov_band_2"}
        for value in open_values
    )
    assert set(mcq_values) <= {"fov_band_0", "fov_band_1", "fov_band_2"}

    qa19 = generate_unified_questions(
        _fixture(),
        qa_ids=["QA-19"],
        seed="domain-time",
    )
    assert qa19["items"]
    time_item = qa19["items"][0]
    time_open = time_item["structure"]["open"]["candidate_value_multiplicity"]
    time_mcq = time_item["structure"]["mcq"]["candidate_value_multiplicity"]
    assert all(isinstance(float(value), float) for value in time_open)
    assert set(time_mcq) <= {"band_0", "band_1", "band_2", "band_3"}


def test_counterfactual_properties_share_selected_anchor_and_include_silent_entities(monkeypatch):
    import avengine.qa.unified_catalog as catalog
    actors = ("speaker", "later_speaker", "silent")
    facts = {
        "actors": {actor: {} for actor in actors},
        "events": [
            {"event_id": "selected", "actor_id": "speaker", "start_s": 1.,
             "start_frame": 10, "end_frame": 20},
            {"event_id": "other", "actor_id": "later_speaker", "start_s": 5.,
             "start_frame": 50, "end_frame": 60}],
        "time": {"frame_count": 100},
        "_p8_candidate": {"event_id": "selected"},
    }
    calls = []
    def angle(_facts, actor, frame):
        calls.append((actor, frame))
        return 30. if frame == 10 else -30.
    monkeypatch.setattr(catalog, "_azimuth", angle)
    item = {"forms": {"open": {"present": True}, "mcq": {"options": [
        {"value": "left"}, {"value": "right"}]}},
        "evidence": {"event_id": "selected", "target_actor_id": "speaker"}}
    values = catalog._p8_form_candidate_values("QA-04", item, facts)
    assert values["open"] == {actor: "right" for actor in actors}
    assert set(calls) == {(actor, 10) for actor in actors}
    windows = []
    def motion(_facts, actor, first, last):
        windows.append((actor, first, last))
        return first == 10 and last == 20
    monkeypatch.setattr(catalog, "_stable_motion_window", motion)
    item["forms"]["mcq"]["options"] = [{"value": "moving"}, {"value": "still"}]
    values = catalog._p8_form_candidate_values("QA-06", item, facts)
    assert values["open"] == {actor: "moving" for actor in actors}
    assert set(windows) == {(actor, 10, 20) for actor in actors}


def test_whole_clip_visibility_candidates_do_not_weight_long_occlusions_more():
    facts = normalize_episode_bundle(_fixture())
    for qa_id in ("QA-09", "QA-11"):
        candidates = _P8_CANDIDATES[qa_id](facts)
        assert len(candidates) == len({row["actor_id"] for row in candidates})
    # Repeated fully-occluded frames still describe one whole-clip predicate.
    for frame in range(4, 9):
        facts["visibility"]["a0"][frame]["state"] = "fully_occluded"
    assert len(_P8_CANDIDATES["QA-09"](facts)) == 1


def test_repeated_speaker_statements_have_unambiguous_question_text():
    raw = _fixture()
    repeated = deepcopy(raw["audio_program"]["events"][1])
    repeated.update(event_id="e1_second", start_sample=48000,
                    end_sample_exclusive=54400, transcript="different words")
    raw["audio_program"]["events"].append(repeated)
    output = generate_unified_questions(raw, qa_ids=["QA-12"], items_per_type=10)
    rows = [row for row in output["items"] if row["evidence"]["target_actor_id"] == "a1"]
    assert len(rows) == 2
    assert {row["truth"]["value"] for row in rows} == {"hello world", "different words"}
    assert len({row["question"]["en"] for row in rows}) == 2
    assert all("spoken statement" in row["question"]["en"] for row in rows)


def test_post_sound_candidates_cannot_fall_back_to_a_different_event():
    import avengine.qa.unified_catalog as catalog
    facts = normalize_episode_bundle(_fixture())
    candidates = catalog._P8_CANDIDATES["QA-17"](facts)
    selected = next(row for row in candidates if row["event_id"] == "e2")
    candidate_facts = catalog._p8_facts_for_candidate(facts, selected, "selected-event")
    emitted = list(catalog._after_event_candidates(candidate_facts, qa_id="QA-17"))
    assert emitted
    assert all(event["event_id"] == "e2" and frame == selected["query_frame"]
               for event, frame, _evidence in emitted)


@pytest.mark.parametrize("angles", [(20., 25., -25.), (0., 20., 90.)])
def test_qa13_known_band_contrast_survives_majority_or_missing_other_value(angles):
    import math
    facts = _two_actor_direction_facts(angles[0], angles[1])
    facts["actors"]["a2"] = deepcopy(facts["actors"]["a1"])
    facts["actors"]["a2"]["actor_id"] = "a2"
    point = [5. * math.sin(math.radians(angles[2])), 0., -5. * math.cos(math.radians(angles[2]))]
    count = facts["time"]["frame_count"]
    for field in ("root_positions_m", "emitter_positions_m"):
        facts["actors"]["a2"][field] = [list(point) for _ in range(count)]
    facts["visibility"]["a2"] = deepcopy(facts["visibility"]["a1"])
    facts["appearance_review"]["a2"] = deepcopy(facts["appearance_review"]["a1"])
    facts["events"] = [event for event in facts["events"] if event["actor_id"] == "a0"]
    output = generate_unified_questions(facts, qa_ids=["QA-13"], seed="known-contrast")
    assert output["counts"]["valid"] == 1
    item = output["items"][0]
    assert "mcq" in item["forms"]
    assert item["form_status"]["mcq"]["status"] == "pass"

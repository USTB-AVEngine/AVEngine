"""Generation-time Gate A checks for both MCQ and Open forms."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "qa"))

from design_qa_v3_scene_batch import (  # noqa: E402
    GenerationConstraintError,
    audit_gatea_pair,
    balanced_binary_joint,
    build_answer,
    resolve_scene_render_context,
    validate_anchor_binding,
)


PARAMS = {"THETA_HALF": 30.0, "T_HALF": 1.0}


def program(slots):
    endpoints = ["ep1", "ep2"]
    rows = []
    for index, endpoint in enumerate(slots):
        rows.append({
            "event_id": f"event_{index}_{endpoint}",
            "source_endpoint_id": endpoint,
            "sound_asset_id": "same-dry-sound",
            "start_sample": 1000 + index * 10000,
            "end_sample_exclusive": 5800 + index * 10000,
            "linear_gain": 0.18,
        })
    return {"candidate_source_endpoint_ids": endpoints, "events": rows}


def answer(mcq, value, scoring):
    return {
        "mcq": {"stem": "same question", "options_space": ["a", "b"],
                "truth_option": mcq},
        "open": {"stem": "same open question", "truth_value": value,
                 "scoring": scoring},
    }


def test_card1_requires_strictly_disjoint_open_credit_regions():
    main, gate = program(["ep1", "ep2"]), program(["ep2", "ep1"])
    with pytest.raises(ValueError, match="open_gold_separated"):
        audit_gatea_pair(
            {"id": "card1F"}, main, gate,
            answer("left", 0.0, "circular_deg"),
            answer("right", 60.0, "circular_deg"), PARAMS)
    checks = audit_gatea_pair(
        {"id": "card1F"}, main, gate,
        answer("left", 0.0, "circular_deg"),
        answer("right", 60.001, "circular_deg"), PARAMS)
    assert checks["mcq_gold_flipped"]
    assert checks["open_gold_separated"]


def test_card8_uses_the_actual_time_scorer_threshold():
    main, gate = program(["ep1", "ep2"]), program(["ep2", "ep1"])
    with pytest.raises(ValueError, match="open_gold_separated"):
        audit_gatea_pair(
            {"id": "card8"}, main, gate,
            answer("band0", 0.5, "absolute_time"),
            answer("band1", 1.5, "absolute_time"), PARAMS)
    checks = audit_gatea_pair(
        {"id": "card8"}, main, gate,
        answer("band0", 0.5, "absolute_time"),
        answer("band1", 1.501, "absolute_time"), PARAMS)
    assert checks["open_separation"] == pytest.approx(1.001)


def test_gatea_rejects_non_slot_audio_mutation():
    main, gate = program(["ep1", "ep2"]), program(["ep2", "ep1"])
    gate = copy.deepcopy(gate)
    gate["events"][0]["linear_gain"] = 0.4
    with pytest.raises(ValueError, match="non_slot_event_fields_same"):
        audit_gatea_pair(
            {"id": "card9"}, main, gate,
            answer("black-and-white", "black-and-white", "closed_set"),
            answer("yellow", "yellow", "closed_set"), PARAMS)


def test_closed_set_gatea_flips_both_forms():
    checks = audit_gatea_pair(
        {"id": "card9"},
        program(["ep1", "ep2"]), program(["ep2", "ep1"]),
        answer("black-and-white", "black-and-white", "closed_set"),
        answer("yellow", "yellow", "closed_set"), PARAMS)
    assert checks["mcq_gold_flipped"]
    assert checks["open_gold_separated"]


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        (lambda gate: gate["events"].pop(), "event_count_same"),
        (lambda gate: gate.update(candidate_source_endpoint_ids=["x", "y"]),
         "candidate_endpoints_same"),
        (lambda gate: [event.update(source_endpoint_id=main)
                       for event, main in zip(
                           gate["events"], ["ep1", "ep2"])],
         "slot_sequence_changed"),
    ],
)
def test_each_gatea_structure_check_has_a_failing_control(mutation, failure):
    main, gate = program(["ep1", "ep2"]), program(["ep2", "ep1"])
    mutation(gate)
    with pytest.raises(GenerationConstraintError, match=failure):
        audit_gatea_pair(
            {"id": "card9"}, main, gate,
            answer("black-and-white", "black-and-white", "closed_set"),
            answer("yellow", "yellow", "closed_set"), PARAMS)


def test_card1_stems_keep_the_audio_referent_and_time_explicit():
    slot_coat = {"source1": "black-and-white", "source2": "yellow"}
    cell = {"answer_band": (-52.5, -17.5)}
    bands = [[-52.5, -17.5], [-17.5, 17.5], [17.5, 52.5]]
    for temporal, query_frame, phrase in (
            ("forward", 74, "At the end of the video"),
            ("backward", 22, "At 1.5 seconds on the video clock")):
        profile = {"id": f"card1-{temporal}", "temporal": temporal,
                   "answer_bands_deg": bands}
        main = build_answer(
            "azimuth_band", profile, cell, None, None, [], "source1",
            "source2", slot_coat, -30.0, query_frame, PARAMS)
        gate = build_answer(
            "azimuth_band", profile, cell, None, None, [], "source2",
            "source1", slot_coat, -30.0, query_frame, PARAMS)
        assert main["mcq"]["stem"] == gate["mcq"]["stem"]
        assert main["open"]["stem"] == gate["open"]["stem"]
        assert "dog that barked last" in main["mcq"]["stem"].lower()
        assert phrase in main["mcq"]["stem"]


def test_audit_rejects_a_changed_question_stem():
    main_answer = answer("black-and-white", "black-and-white", "closed_set")
    gate_answer = answer("yellow", "yellow", "closed_set")
    main_answer["mcq"].update(stem="same", options_space=["a", "b"])
    gate_answer["mcq"].update(stem="different", options_space=["a", "b"])
    main_answer["open"]["stem"] = gate_answer["open"]["stem"] = "same"
    with pytest.raises(GenerationConstraintError, match="mcq_stem_same"):
        audit_gatea_pair(
            {"id": "card9"}, program(["ep1", "ep2"]),
            program(["ep2", "ep1"]), main_answer, gate_answer, PARAMS)


def test_joint_allocator_covers_all_slot_coat_cells_for_six():
    rows = balanced_binary_joint(
        ["source1", "source2"], ["black-and-white", "yellow"], 6,
        "seed")
    assert set(rows) == {
        ("source1", "black-and-white"), ("source1", "yellow"),
        ("source2", "black-and-white"), ("source2", "yellow"),
    }


def test_unknown_anchor_binding_fails_instead_of_falling_through():
    with pytest.raises(GenerationConstraintError, match="unknown anchor_binding"):
        validate_anchor_binding(
            {"id": "bad", "anchor_binding": "first_callerr"},
            SimpleNamespace(), [], target_slot="source1", query_frame=0,
            answer={})


def test_render_context_requires_explicit_scene_map_and_transform():
    scene = SimpleNamespace(scene_id="new-room", render_config={})
    with pytest.raises(ValueError, match="no apartment fallback"):
        resolve_scene_render_context(scene)
    scene.render_config = {
        "native_map": "/Game/SPEAR/Scenes/debug_0000/Maps/debug_0000",
        "room_profile_id": "spear_debug_0000",
        "world_transform": "ue_xyz_cm_to_xzy_m_v1",
    }
    resolved = resolve_scene_render_context(scene)
    assert resolved["native_map"].endswith("/debug_0000")
    assert resolved["world_transform"]([100, 200, 300]) == [1.0, 3.0, 2.0]

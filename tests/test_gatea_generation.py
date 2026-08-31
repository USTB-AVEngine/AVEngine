"""Generation-time Gate A checks for both MCQ and Open forms."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "qa"))

from design_qa_v3_scene_batch import audit_gatea_pair  # noqa: E402


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
        "mcq": {"truth_option": mcq},
        "open": {"truth_value": value, "scoring": scoring},
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

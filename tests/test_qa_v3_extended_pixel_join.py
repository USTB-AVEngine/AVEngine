"""Tests for pixel joining of extended QA-v3 profiles."""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from join_qa_v3_extended_pixel import evaluate  # noqa: E402


def _pixel(states, frame):
    return {
        "authority": "test_pixel_authority",
        "per_instance": {
            slot: {"frames": [{"frame_index": frame, "state": state}]}
            for slot, state in states.items()
        },
    }


def _fact(profile_id):
    if profile_id == "card15a":
        truth = [4, 2]
    else:
        truth = "source1"
    return {
        "profile_id": profile_id,
        "point_id": "p",
        "mcq": {"truth_option": truth},
        "open": {"truth_value": truth},
    }


def test_card11_requires_three_visible_and_fourth_hidden():
    passed = evaluate(
        _fact("card11"),
        _pixel({
            "source1": "visible_clear",
            "source2": "visible_occluded",
            "source3": "visible_clear",
            "source4": "out_of_view",
        }, 30))
    assert passed["status"] == "pass"
    failed = evaluate(
        _fact("card11"),
        _pixel({
            "source1": "visible_clear",
            "source2": "visible_clear",
            "source3": "visible_clear",
            "source4": "visible_occluded",
        }, 30))
    assert failed["status"] == "pixel_rejected"
    assert failed["rejection_reasons"] == [
        "offscreen_candidate_is_visually_present"]


def test_card15a_requires_all_four_visible():
    result = evaluate(
        _fact("card15a"),
        _pixel({
            "source1": "visible_clear",
            "source2": "visible_occluded",
            "source3": "visible_clear",
            "source4": "visible_occluded",
        }, 30))
    assert result["status"] == "pass"
    assert result["bindings"]["distinct_callers"] == 2


def test_card16_binds_main_and_gatea_to_distinct_final_states():
    result = evaluate(
        _fact("card16"),
        _pixel({
            "source1": "visible_occluded",
            "source2": "out_of_view",
        }, 74))
    assert result["status"] == "pass"
    assert result["bindings"]["main_truth_option"] == "visible_occluded"
    assert result["bindings"]["gatea_truth_option"] == "out_of_view"
    rejected = evaluate(
        _fact("card16"),
        _pixel({
            "source1": "out_of_view",
            "source2": "out_of_view",
        }, 74))
    assert rejected["status"] == "pixel_rejected"

"""Tests for non-mutating QA-v3 historical candidate revalidation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from audit_qa_v3_prescale_candidates import audit_candidate  # noqa: E402


PARAMS = {"THETA_HALF": 30.0, "T_FULL": 0.5}


def _timeline(path, anchor_y, query_y):
    frames = []
    for index in range(75):
        t = index / 74
        y = anchor_y + (query_y - anchor_y) * t
        frames.append({
            "frame_index": index,
            "camera": {"translation_ue_cm": [0, 0, 147],
                       "yaw_ue_deg": 0},
            "actor_states": [
                {"source_slot_id": "source1",
                 "translation_ue_cm": [300, y, 0]},
                {"source_slot_id": "source2",
                 "translation_ue_cm": [300, -y, 0]},
            ],
        })
    path.write_text(json.dumps({"frames": frames}))


def _candidate(tmp_path, profile, fact, program=None):
    fact_path = tmp_path / f"{profile}_fact.json"
    timeline_path = tmp_path / f"{profile}_timeline.json"
    program_path = tmp_path / f"{profile}_program.json"
    fact_path.write_text(json.dumps(fact))
    _timeline(timeline_path, 0, 300)
    program_path.write_text(json.dumps(program or {"events": []}))
    return {
        "pilot_id": f"room__{profile}__001",
        "artifacts": {"fact": str(fact_path), "timeline": str(timeline_path),
                      "main_program": str(program_path)},
        "gateb": {},
    }


def test_old_card1_is_classified_for_audio_regeneration_and_rebalance(tmp_path):
    fact = {
        "target_slot": "source1", "anchor_frame": 0, "query_frame": 74,
        "audio": {"events": [
            {"role": "target_actor"}, {"role": "target_actor"},
            {"role": "non_target_actor"}]},
    }
    result = audit_candidate(_candidate(tmp_path, "card1F", fact), PARAMS)
    assert result["status"] == "media_regeneration_required"
    assert "regenerate_audio_and_fact" in result["actions"]
    assert "pool_joint_rebalance" in result["actions"]


def test_card8_old_metadata_can_be_rewritten_when_strict_regions_separate(tmp_path):
    fact = {
        "truth": {"first_onset_s": 0.5,
                  "non_target_first_onset_s": 1.6},
        "open": {},
    }
    candidate = _candidate(tmp_path, "card8", fact)
    candidate["gateb"] = {
        "audio_policy": "appearance_canonical_anchor_audio_must_be_identical"}
    result = audit_candidate(candidate, PARAMS)
    assert result["status"] == "metadata_or_pool_reselection_required"
    assert result["actions"] == ["rewrite_question_metadata"]


def test_card11_rejects_old_event_frame_and_out_of_view_policy(tmp_path):
    fact = {"pixel_acceptance": {"source4":
                                 "fully_occluded_or_out_of_view"}}
    program = {"events": [{"start_sample": 8000,
                            "end_sample_exclusive": 12800}]}
    result = audit_candidate(
        _candidate(tmp_path, "card11", fact, program), PARAMS)
    assert "regenerate_audio_and_fact" in result["actions"]
    assert "rerun_pixel_join" in result["actions"]


def test_demoted_and_future_profiles_are_not_reported_as_structure_pass(tmp_path):
    card15 = audit_candidate(_candidate(tmp_path, "card15a", {}), PARAMS)
    card17 = audit_candidate(_candidate(tmp_path, "card17", {}), PARAMS)
    assert card15["status"] == "demoted_from_main"
    assert card17["status"] == "future_extension"

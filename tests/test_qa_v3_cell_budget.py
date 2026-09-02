"""The scene batch manifest reports its joint allocation budget per room."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from design_qa_v3_scene_batch import cell_allocation, cell_budget_report  # noqa: E402


def _cell(index, *, slot, anchor, answer, moves_more, profile="card1F"):
    return {"profile": {"id": profile}, "cell_index": index, "target_slot": slot,
            "anchor_band": anchor, "answer_band": answer,
            "target_moves_more": moves_more, "target_coat": "yellow"}


def test_allocation_is_recorded_as_strings_without_losing_the_key():
    alloc = cell_allocation(_cell(0, slot="source1", anchor=(-52.5, -17.5),
                                  answer=[17.5, 52.5], moves_more=True))
    assert alloc == {"profile_id": "card1F", "target_slot": "source1",
                     "anchor_band": "-52.5,-17.5", "answer": "17.5,52.5",
                     "target_moves_more": True, "target_coat": "yellow"}
    assert cell_allocation(_cell(1, slot="source2", anchor=None, answer="closer",
                                 moves_more=False, profile="card5R"))["anchor_band"] is None


def test_budget_report_counts_filled_and_exhausted_per_joint_key():
    cells = [
        _cell(0, slot="source1", anchor=(-52.5, -17.5), answer=(17.5, 52.5), moves_more=True),
        _cell(1, slot="source1", anchor=(-52.5, -17.5), answer=(17.5, 52.5), moves_more=True),
        _cell(2, slot="source2", anchor=(17.5, 52.5), answer=(-52.5, -17.5), moves_more=False),
        _cell(3, slot="source2", anchor=(17.5, 52.5), answer=(-17.5, 17.5), moves_more=False),
    ]
    made = ["card1F_001", "card1F_003"]
    rejected = [{"point_id": "card1F_002", "reason": "no_candidate_within_attempt_budget"},
                {"point_id": "card1F_004", "reason": "camera_clearance_blocked"}]
    report = cell_budget_report(cells, made, rejected)
    card1f = report["card1F"]
    assert card1f["totals"] == {"requested": 4, "filled": 2, "exhausted": 2,
                               "keys": 3, "keys_unfilled": 1}
    first = card1f["cells"]["target_slot=source1|anchor_band=-52.5,-17.5|answer=17.5,52.5|target_moves_more=True"]
    assert first == {"requested": 2, "filled": 1,
                     "exhausted_by_reason": {"no_candidate_within_attempt_budget": 1}}
    last = card1f["cells"]["target_slot=source2|anchor_band=17.5,52.5|answer=-17.5,17.5|target_moves_more=False"]
    assert last["filled"] == 0 and last["exhausted_by_reason"] == {"camera_clearance_blocked": 1}
    assert "not backfilled" in card1f["boundary"]

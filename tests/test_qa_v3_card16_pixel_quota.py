"""Tests for post-pixel card16 gold-state quota selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from select_qa_v3_card16_pixel_quota import main, select  # noqa: E402


STATES = (
    "visible_clear", "visible_occluded", "fully_occluded", "out_of_view")


def _join(path, state, status="pass"):
    path.write_text(json.dumps({
        "profile_id": "card16",
        "status": status,
        "bindings": {"main_truth_option": state},
    }))
    return str(path)


def _records(tmp_path):
    records = []
    for room in ("room_a", "room_b"):
        for state in STATES:
            for index in range(2):
                pilot_id = f"{room}__{state}__{index}"
                records.append({
                    "pilot_id": pilot_id,
                    "room_id": room,
                    "pixel_join": _join(tmp_path / f"{pilot_id}.json", state),
                })
    return records


def test_selector_balances_states_and_round_robins_rooms(tmp_path):
    result = select(_records(tmp_path), per_state=2)
    assert result["status"] == "complete"
    assert result["selected_by_gold"] == {state: 2 for state in STATES}
    assert result["selected_by_room"] == {"room_a": 4, "room_b": 4}
    assert result["selected_count"] == 8
    assert "model" in result["selection_authority"]


def test_selector_reports_structural_shortfall_without_relaxing(tmp_path):
    records = _records(tmp_path)
    records = [record for record in records
               if "fully_occluded" not in record["pilot_id"]]
    result = select(records, per_state=2)
    assert result["status"] == "partial"
    assert result["shortfall_by_gold"]["fully_occluded"] == 2
    assert result["selected_by_gold"]["fully_occluded"] == 0


def test_rejected_pixel_join_is_not_selected_and_cli_is_no_clobber(tmp_path):
    records = _records(tmp_path)
    bad = tmp_path / "bad.json"
    records.append({
        "pilot_id": "room_c__bad",
        "room_id": "room_c",
        "pixel_join": _join(bad, "out_of_view", status="pixel_rejected"),
    })
    index = tmp_path / "index.json"
    output = tmp_path / "selection.json"
    index.write_text(json.dumps({"records": records}))
    assert main([
        "--index", str(index), "--per-state", "1", "--output", str(output),
    ]) == 0
    value = json.loads(output.read_text())
    assert value["pixel_rejected_count"] == 1
    assert value["selected_count"] == 4
    assert main([
        "--index", str(index), "--per-state", "1", "--output", str(output),
    ]) == 2


def test_duplicate_identity_and_invalid_quota_fail_closed(tmp_path):
    records = _records(tmp_path)[:2]
    records[1]["pilot_id"] = records[0]["pilot_id"]
    with pytest.raises(ValueError, match="unique"):
        select(records, per_state=1)
    with pytest.raises(ValueError, match="positive"):
        select([], per_state=0)

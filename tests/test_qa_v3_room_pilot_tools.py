"""Unit tests for room-centric QA-v3 pilot assembly/finalization helpers."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from assemble_qa_v3_room_pilot import _balanced_choice  # noqa: E402
from finalize_qa_v3_room_pilot import _card17_distinct  # noqa: E402
from materialize_qa_v3_dual_gateb import _swap_dynamic_states  # noqa: E402


def test_balanced_choice_round_robins_answer_strata():
    pool = []
    for label, count in [("left", 7), ("center", 2), ("right", 8)]:
        for index in range(count):
            pool.append({
                "mcq_truth_option": label,
                "geometry_signature": [label, index],
            })
    chosen = _balanced_choice(pool, 6)
    assert Counter(item["mcq_truth_option"] for item in chosen) == {
        "left": 2, "center": 2, "right": 2}


def test_route_swap_changes_dynamic_tracks_without_swapping_identity():
    timeline = {
        "frames": [{
            "actor_states": [
                {
                    "source_slot_id": "source1",
                    "asset_id": "asset_a",
                    "translation_ue_cm": [1.0, 0.0, 0.0],
                    "yaw_ue_deg": 10.0,
                    "action_id": "idle",
                    "action_phase": 0.0,
                },
                {
                    "source_slot_id": "source2",
                    "asset_id": "asset_b",
                    "translation_ue_cm": [2.0, 0.0, 0.0],
                    "yaw_ue_deg": 20.0,
                    "action_id": "walk",
                    "action_phase": 0.5,
                },
            ]
        }]
    }
    twin = _swap_dynamic_states(timeline)
    left, right = twin["frames"][0]["actor_states"]
    assert left["source_slot_id"] == "source1"
    assert left["asset_id"] == "asset_a"
    assert left["translation_ue_cm"] == [2.0, 0.0, 0.0]
    assert right["source_slot_id"] == "source2"
    assert right["asset_id"] == "asset_b"
    assert right["translation_ue_cm"] == [1.0, 0.0, 0.0]


def test_card17_runtime_distinct_check_uses_actual_readbacks(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    base = {
        "frames": [{
            "camera_pose": {"location_cm": [0, 0, 1], "rotation_deg": [0, 0, 0]},
            "actor_anchor_poses": {
                "source1": {"location_cm": [1, 0, 0]},
                "source2": {"location_cm": [2, 0, 0]},
            },
        }]
    }
    changed = json.loads(json.dumps(base))
    changed["frames"][0]["camera_pose"]["location_cm"][0] = 10
    (first / "frame_records.json").write_text(json.dumps(base))
    (second / "frame_records.json").write_text(json.dumps(changed))
    result = _card17_distinct(first, second)
    assert result["runtime_readbacks_differ"] is True

"""Run02 selection and visual-verification regression tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from design_qa_v3_scene_batch import materialize_derived_params  # noqa: E402
from select_qa_v3_run02_dev import (  # noqa: E402
    pixel_eligible,
    select_profile,
)
from verify_qa_v3_visual_batch import verify_batch  # noqa: E402


def _candidate(point_id, bits):
    answer, slot, source1_coat, target_coat, moves = bits
    return {
        "answer": answer,
        "first_caller_slot": None,
        "pixel_evidence": f"/pixel/{point_id}",
        "point_id": point_id,
        "profile_id": "cardX",
        "source1_coat": source1_coat,
        "source_design_root": "/design",
        "target_coat": target_coat,
        "target_moves_more": moves,
        "target_slot": slot,
    }


def test_selection_uses_coverage_objective_and_stable_tie_break():
    candidates = [
        _candidate("p001", ("a", "s1", "bw", "bw", False)),
        _candidate("p002", ("b", "s2", "y", "y", True)),
        _candidate("p003", ("a", "s1", "y", "y", True)),
        _candidate("p004", ("b", "s2", "bw", "bw", False)),
    ]
    chosen, diagnostics = select_profile(candidates, 2)
    assert [item["point_id"] for item in chosen] == ["p001", "p002"]
    assert diagnostics["squared_marginal_objective"] == 10
    assert diagnostics["marginals"]["answer"] == {"a": 1, "b": 1}


def test_selection_rejects_missing_binary_factor_value():
    candidates = [
        _candidate(f"p{i}", ("a" if i % 2 else "b", "s1", "bw", "y",
                             bool(i % 2)))
        for i in range(1, 7)
    ]
    with pytest.raises(ValueError, match="target_slot must expose both"):
        select_profile(candidates, 6)


def test_scene_batch_materializes_card8_derived_edges():
    params = {
        "BANDS_CARD8": [0.35, 1.1, 1.85, 2.6],
        "FIRST_MIN_S": 0.35,
        "GAP_MIN_S": 0.3,
        "T_HALF": 1.0,
        "T_FULL": 0.5,
        "CLIP_SECONDS": 5.0,
        "EVENT_SECONDS": 0.3,
    }
    effective = materialize_derived_params(params)
    assert params["BANDS_CARD8"] == [0.35, 1.1, 1.85, 2.6]
    assert effective["BANDS_CARD8"] == [
        0.35, 1.2875, 2.225, 3.1625, 4.1]
    assert "Derived before generation" in effective["BANDS_CARD8_note"]
    assert effective["CARD8_FIRST_CALL_SCORING"]["T_FULL"] == 0.5


def test_pixel_eligibility_applies_all_three_rules():
    truth = {
        "resolution_hw": [720, 1280],
        "per_instance": {
            "source1": {"frames": [{
                "visible_fraction": 0.6,
                "visible_pixels": 1001,
                "target_bbox_xyxy_px": [1, 1, 1279, 719],
            }]},
            "source2": {"frames": [{
                "visible_fraction": 0.7,
                "visible_pixels": 1200,
                "target_bbox_xyxy_px": [10, 10, 100, 100],
            }]},
        },
    }
    assert pixel_eligible(
        truth, minimum_visible_fraction=0.5, minimum_visible_pixels=1000,
        bbox_must_not_touch_frame_edge=True)
    truth["per_instance"]["source1"]["frames"][0][
        "target_bbox_xyxy_px"] = [0, 1, 1279, 719]
    assert not pixel_eligible(
        truth, minimum_visible_fraction=0.5, minimum_visible_pixels=1000,
        bbox_must_not_touch_frame_edge=True)


def _frame(index, actor_x, observed_x, animation_error):
    return {
        "frame_index": index,
        "camera": {
            "translation_ue_cm": [1.0, 2.0, 3.0],
            "yaw_ue_deg": 10.0,
        },
        "actor_states": [{
            "source_slot_id": "source1",
            "translation_ue_cm": [actor_x, 0.0, 0.0],
            "yaw_ue_deg": 20.0,
        }],
        "observed": {
            "camera_pose": {
                "location_cm": [1.0, 2.0, 3.0],
                "rotation_deg": [0.0, 0.0, 10.0],
            },
            "actor_anchor_poses": {
                "source1": {
                    "location_cm": [observed_x, 0.0, 0.0],
                    "rotation_deg": [0.0, 0.0, 20.0],
                },
            },
            "animation_readbacks": [{
                "source_slot_id": "source1",
                "observed_position_seconds": 1.0 + animation_error,
                "requested_position_seconds": 1.0,
                "absolute_error_seconds": animation_error,
            }],
        },
    }


def test_visual_verifier_writes_recomputable_batch_maxima(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "selected": [{"point_id": "cardX_001"}],
    }))
    visual = tmp_path / "visual"
    point = visual / "cardX_001"
    point.mkdir(parents=True)
    (point / "frame_records.json").write_text(json.dumps({
        "frames": [_frame(0, 0.0, 0.0, 0.0),
                   _frame(1, 1.0, 1.25, 0.000001)],
    }))
    (point / "research_receipt.json").write_text(json.dumps({
        "capture": {
            "completed_frame_count": 2,
            "frame_count": 2,
            "root_readback_summary": {
                "source1_actor": {"status": "pass"},
            },
            "animation_readback_summary": {"status": "pass"},
        },
    }))
    result = verify_batch(selection, visual, expected_frames=2)
    assert result["counts"] == {
        "selected_points": 1,
        "verified_points": 1,
        "verified_frames": 2,
        "failures": 0,
    }
    assert result["maxima"]["actor_position_error_cm"] == pytest.approx(0.25)
    assert result["maxima"]["animation_error_seconds"] == pytest.approx(1e-6)

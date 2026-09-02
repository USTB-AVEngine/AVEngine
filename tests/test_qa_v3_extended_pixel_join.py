"""Tests for pixel joining of extended QA-v3 profiles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import pytest

from join_qa_v3_extended_pixel import evaluate, main  # noqa: E402


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
            "source4": "fully_occluded",
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
    out_of_view = evaluate(
        _fact("card11"),
        _pixel({
            "source1": "visible_clear",
            "source2": "visible_clear",
            "source3": "visible_clear",
            "source4": "out_of_view",
        }, 30))
    assert out_of_view["status"] == "pixel_rejected"
    assert out_of_view["rejection_reasons"] == [
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


def _card16_pixel(source1_binding, source2_binding, source1_final, source2_final):
    return {
        "authority": "test_pixel_authority",
        "per_instance": {
            "source1": {"frames": [
                {"frame_index": 12, "state": source1_binding},
                {"frame_index": 74, "state": source1_final},
            ]},
            "source2": {"frames": [
                {"frame_index": 12, "state": source2_binding},
                {"frame_index": 74, "state": source2_final},
            ]},
        },
    }


def test_card16_binds_main_and_gatea_to_distinct_final_states():
    result = evaluate(
        _fact("card16"),
        _card16_pixel(
            "visible_clear", "visible_occluded",
            "visible_occluded", "out_of_view"))
    assert result["status"] == "pass"
    assert result["bindings"]["main_truth_option"] == "visible_occluded"
    assert result["bindings"]["gatea_truth_option"] == "out_of_view"
    rejected = evaluate(
        _fact("card16"),
        _card16_pixel(
            "fully_occluded", "visible_clear",
            "out_of_view", "out_of_view"))
    assert rejected["status"] == "pixel_rejected"
    assert rejected["rejection_reasons"] == [
        "main_first_caller_not_visible_at_binding_frame",
        "first_caller_and_counterfactual_have_same_final_state",
    ]


PIXEL_PARAMS = {
    "PIXEL_MIN_VISIBLE_FRACTION": 0.5,
    "PIXEL_MIN_VISIBLE_PIXELS": 1000,
    "PIXEL_BBOX_MUST_NOT_TOUCH_FRAME_EDGE": True,
    "PIXEL_THRESHOLD_STATUS": "placeholder_research",
}


def _card1_frame(index, *, pixels, fraction, bbox=(300, 300, 400, 400),
                 state="visible_occluded"):
    return {"frame_index": index, "state": state, "visible_pixels": pixels,
            "target_pixels": int(round(pixels / fraction)) if fraction else 0,
            "visible_fraction": fraction, "target_bbox_xyxy_px": list(bbox)}


def _card1_truth(main_anchor, main_query, gatea_anchor, gatea_query):
    return {
        "authority": "test_pixel_authority",
        "resolution_hw": [720, 1280],
        "per_instance": {
            "source2": {"frames": [main_anchor, main_query]},
            "source1": {"frames": [gatea_anchor, gatea_query]},
        },
    }


def _card1_fact(profile_id="card1F", anchor_frame=40, query_frame=74,
                with_thresholds=False):
    fact = {
        "profile_id": profile_id, "point_id": f"{profile_id}_002",
        "target_slot": "source2", "anchor_frame": anchor_frame,
        "query_frame": query_frame,
        "mcq": {"truth_option": "[17.5, 52.5)"},
        "open": {"truth_value": 40.496},
    }
    if with_thresholds:
        fact["pixel_acceptance"] = {"thresholds": {
            "min_visible_fraction": 0.5, "min_visible_pixels": 1000,
            "bbox_must_not_touch_frame_edge": True,
            "status": "placeholder_research"}}
    return fact


def test_card1_join_passes_when_both_referents_clear_both_frames():
    truth = _card1_truth(
        _card1_frame(40, pixels=9720, fraction=0.817),
        _card1_frame(74, pixels=8735, fraction=0.967),
        _card1_frame(40, pixels=1477, fraction=0.528),
        _card1_frame(74, pixels=1400, fraction=0.7))
    result = evaluate(_card1_fact(), truth, PIXEL_PARAMS)
    assert result["status"] == "pass"
    assert result["rejection_reasons"] == []
    bindings = result["bindings"]
    assert bindings["main_referent_slot"] == "source2"
    assert bindings["gatea_referent_slot"] == "source1"
    assert bindings["thresholds"]["min_visible_pixels"] == 1000
    assert bindings["threshold_status"] == "placeholder_research"
    assert bindings["evaluations"]["gatea"]["query_frame"]["visible_pixels"] == 1400
    assert bindings["evaluations"]["main"]["anchor_frame"]["requirement"] == \
        "referent_bindable_at_identity_anchor"
    assert "not accepted" in bindings["line_of_sight_role"]


def test_card1_join_rejects_gatea_referent_too_small_at_query_frame():
    """Kujiale card1F_002 的阳性拒绝:Gate A 指代者查询帧只有 198 像素。"""
    truth = _card1_truth(
        _card1_frame(40, pixels=9720, fraction=0.8170),
        _card1_frame(74, pixels=8735, fraction=0.9667),
        _card1_frame(40, pixels=1477, fraction=0.5279),
        _card1_frame(74, pixels=198, fraction=0.1049))
    result = evaluate(_card1_fact(), truth, PIXEL_PARAMS)
    assert result["status"] == "pixel_rejected"
    assert result["rejection_reasons"] == [
        "gatea_referent_query_frame_visible_fraction_below_threshold",
        "gatea_referent_query_frame_visible_pixels_below_threshold",
    ]
    gatea_query = result["bindings"]["evaluations"]["gatea"]["query_frame"]
    assert gatea_query["visible_pixels"] == 198
    assert gatea_query["passed"] is False
    assert result["bindings"]["evaluations"]["main"]["query_frame"]["passed"]


def test_card1_join_reports_anchor_frame_and_bbox_edge_failures_precisely():
    truth = _card1_truth(
        _card1_frame(40, pixels=0, fraction=0.0, state="fully_occluded"),
        _card1_frame(74, pixels=8735, fraction=0.9667),
        _card1_frame(40, pixels=1477, fraction=0.5279),
        _card1_frame(74, pixels=5000, fraction=0.9, bbox=(0, 300, 400, 400)))
    result = evaluate(_card1_fact("card1B", anchor_frame=40, query_frame=74),
                      truth, PIXEL_PARAMS)
    assert result["status"] == "pixel_rejected"
    assert result["rejection_reasons"] == [
        "main_referent_anchor_frame_not_visible_state",
        "main_referent_anchor_frame_visible_fraction_below_threshold",
        "main_referent_anchor_frame_visible_pixels_below_threshold",
        "gatea_referent_query_frame_bbox_touches_frame_edge",
    ]


def test_card1_join_rejects_missing_frames_and_needs_explicit_thresholds():
    truth = _card1_truth(
        _card1_frame(40, pixels=9720, fraction=0.817),
        _card1_frame(74, pixels=8735, fraction=0.967),
        _card1_frame(40, pixels=1477, fraction=0.528),
        _card1_frame(74, pixels=1400, fraction=0.7))
    with pytest.raises(ValueError, match="explicit thresholds"):
        evaluate(_card1_fact(), truth)
    with pytest.raises(ValueError, match="missing explicit pixel thresholds"):
        evaluate(_card1_fact(), truth, {"PIXEL_MIN_VISIBLE_FRACTION": 0.5})
    # thresholds recorded in the fact are honoured without params ...
    assert evaluate(_card1_fact(with_thresholds=True), truth)["status"] == "pass"
    # ... and a disagreeing params file is refused rather than silently chosen
    with pytest.raises(ValueError, match="differs"):
        evaluate(_card1_fact(with_thresholds=True), truth,
                 dict(PIXEL_PARAMS, PIXEL_MIN_VISIBLE_PIXELS=500))
    sparse = _card1_truth(
        _card1_frame(40, pixels=9720, fraction=0.817),
        _card1_frame(74, pixels=8735, fraction=0.967),
        _card1_frame(40, pixels=1477, fraction=0.528),
        _card1_frame(0, pixels=1400, fraction=0.7))
    result = evaluate(_card1_fact(), sparse, PIXEL_PARAMS)
    assert result["rejection_reasons"] == [
        "gatea_referent_query_frame_missing_in_pixel_truth"]


def test_card1_cli_records_params_and_refuses_without_thresholds(tmp_path):
    fact = tmp_path / "fact.json"
    pixel = tmp_path / "pixel.json"
    params = tmp_path / "params.json"
    fact.write_text(json.dumps(_card1_fact()))
    pixel.write_text(json.dumps(_card1_truth(
        _card1_frame(40, pixels=9720, fraction=0.817),
        _card1_frame(74, pixels=8735, fraction=0.967),
        _card1_frame(40, pixels=1477, fraction=0.528),
        _card1_frame(74, pixels=198, fraction=0.105))))
    params.write_text(json.dumps(PIXEL_PARAMS))
    refused = tmp_path / "refused.json"
    assert main(["--fact", str(fact), "--pixel-truth", str(pixel),
                 "--output", str(refused)]) == 2
    assert not refused.exists()
    output = tmp_path / "join.json"
    assert main(["--fact", str(fact), "--pixel-truth", str(pixel),
                 "--params", str(params), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["status"] == "pixel_rejected"
    assert result["inputs"]["params"]["path"] == str(params.resolve())
    assert len(result["inputs"]["params"]["sha256"]) == 64


def test_cli_binds_fact_and_pixel_inputs(tmp_path):
    fact = tmp_path / "fact.json"
    pixel = tmp_path / "pixel.json"
    output = tmp_path / "join.json"
    fact.write_text(json.dumps(_fact("card15a")))
    pixel.write_text(json.dumps(_pixel({
        "source1": "visible_clear",
        "source2": "visible_occluded",
        "source3": "visible_clear",
        "source4": "visible_occluded",
    }, 30)))
    assert main([
        "--fact", str(fact),
        "--pixel-truth", str(pixel),
        "--output", str(output),
    ]) == 0
    result = json.loads(output.read_text())
    assert result["inputs"]["fact"]["path"] == str(fact.resolve())
    assert result["inputs"]["pixel_truth"]["path"] == str(pixel.resolve())
    assert len(result["inputs"]["fact"]["sha256"]) == 64
    assert main([
        "--fact", str(fact),
        "--pixel-truth", str(pixel),
        "--output", str(output),
    ]) == 2

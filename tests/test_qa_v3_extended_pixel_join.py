"""Tests for pixel joining of extended QA-v3 profiles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import numpy as np
import pytest

from join_qa_v3_extended_pixel import (  # noqa: E402
    evaluate,
    main,
    occluder_statistics,
    visibility_timeline,
)
from qa_v3_pixel_thresholds import (  # noqa: E402
    card1_pixel_acceptance_block,
    pixel_policy_from_params,
    tier_for_frame,
)


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


# the historical threshold policy must now be named explicitly; the owner's
# tier policy is the default (see TIER_PARAMS below)
PIXEL_PARAMS = {
    "PIXEL_MIN_VISIBLE_FRACTION": 0.5,
    "PIXEL_MIN_VISIBLE_PIXELS": 1000,
    "PIXEL_BBOX_MUST_NOT_TOUCH_FRAME_EDGE": True,
    "PIXEL_THRESHOLD_STATUS": "placeholder_research",
    "PIXEL_ACCEPTANCE_POLICY": "both_frames_threshold_reject",
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
        fact["pixel_acceptance"] = {
            "thresholds": {
                "min_visible_fraction": 0.5, "min_visible_pixels": 1000,
                "bbox_must_not_touch_frame_edge": True,
                "status": "placeholder_research"},
            "acceptance_policy": {"policy": "both_frames_threshold_reject",
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


# 2026-09-02 semantics: nothing but a camera-side blockage rejects.
TIER_PARAMS = dict(PIXEL_PARAMS,
                   PIXEL_ACCEPTANCE_POLICY="camera_blockage_reject_then_tier",
                   PIXEL_CAMERA_BLOCKAGE_MAX_DISTANCE_M=1.5,
                   PIXEL_TIER_VISIBLE_FRACTION_EDGES=[0.5, 0.2],
                   PIXEL_TIER_REJECT_TIERS=[])
# 2026-09-03 owner rule: keep the question unless the referent is blocked
# completely at a declared frame.
OWNER_TIER_PARAMS = dict(TIER_PARAMS,
                         PIXEL_TIER_REJECT_TIERS=["hidden", "out_of_view"])


def test_tier_policy_is_the_default_and_the_legacy_policy_must_be_named():
    """Owner decision 2026-09-02: tiers by default, thresholds only on request."""
    implicit = {k: v for k, v in TIER_PARAMS.items()
                if k != "PIXEL_ACCEPTANCE_POLICY"}
    assert pixel_policy_from_params(implicit)["policy"] == \
        "camera_blockage_reject_then_tier"
    # the default still fails closed when the tier settings are missing
    with pytest.raises(ValueError, match="explicit"):
        pixel_policy_from_params({k: v for k, v in PIXEL_PARAMS.items()
                                  if k != "PIXEL_ACCEPTANCE_POLICY"})
    assert pixel_policy_from_params(PIXEL_PARAMS)["policy"] == \
        "both_frames_threshold_reject"
    truth = _truth_with_frames(
        _card1_frame(40, pixels=9720, fraction=0.817),
        _card1_frame(74, pixels=8735, fraction=0.967),
        _card1_frame(40, pixels=1477, fraction=0.528),
        _card1_frame(74, pixels=198, fraction=0.105))
    # a fact with thresholds but no policy and no params cannot pick a policy
    fact = _card1_fact(with_thresholds=True)
    del fact["pixel_acceptance"]["acceptance_policy"]
    with pytest.raises(ValueError, match="acceptance policy"):
        evaluate(fact, truth)
SENTINEL = 65504.0


def _depth_arrays(truth, occluders):
    """Tiny 4x4 depth frames: occluders[(slot, frame)] = (target_m, occluder_m,
    hidden_cells).  Footprint = first row; hidden cells sit in front of it."""
    frames = list(truth["frame_indices"])
    arrays = {"normal_depth_m": np.full((len(frames), 4, 4), 9.0, np.float32)}
    for slot in truth["per_instance"]:
        arrays[f"target_only_{slot}_depth_m"] = np.full(
            (len(frames), 4, 4), SENTINEL, np.float32)
    for (slot, frame), (target_m, occluder_m, hidden_cells) in occluders.items():
        k = frames.index(frame)
        arrays[f"target_only_{slot}_depth_m"][k, 0, :] = target_m
        arrays["normal_depth_m"][k, 0, :] = target_m
        arrays["normal_depth_m"][k, 0, :hidden_cells] = occluder_m
    return arrays


def _truth_with_frames(main_anchor, main_query, gatea_anchor, gatea_query):
    truth = _card1_truth(main_anchor, main_query, gatea_anchor, gatea_query)
    truth["frame_indices"] = [40, 74]
    truth["depth_comparison"] = {"target_only_background_depth_m": SENTINEL,
                                 "absolute_tolerance_m": 0.01,
                                 "relative_tolerance": 0.002}
    return truth


def test_tier_policy_keeps_far_occlusion_as_a_difficulty_tier():
    """card1F_002-like: Gate A referent 10% visible behind a far curtain."""
    truth = _truth_with_frames(
        _card1_frame(40, pixels=9720, fraction=0.817),
        _card1_frame(74, pixels=8735, fraction=0.967),
        _card1_frame(40, pixels=1477, fraction=0.528),
        _card1_frame(74, pixels=198, fraction=0.105))
    arrays = _depth_arrays(truth, {("source1", 74): (4.5, 4.2, 3)})
    result = evaluate(_card1_fact(), truth, TIER_PARAMS, arrays)
    assert result["status"] == "pass"
    assert result["rejection_reasons"] == []
    difficulty = result["bindings"]["difficulty"]
    assert difficulty["worst_tier"] == "heavy"
    assert difficulty["tiers"]["gatea"]["query_frame"] == "heavy"
    assert difficulty["tiers"]["main"]["anchor_frame"] == "light"
    assert difficulty["anchor_instant_hidden"] is False
    assert difficulty["referent_frames_below_placeholder_thresholds"] == 1
    gatea_query = result["bindings"]["evaluations"]["gatea"]["query_frame"]
    assert gatea_query["occluder"]["occluder_median_depth_m"] == pytest.approx(4.2)
    assert gatea_query["camera_side_blockage"] is False
    assert result["bindings"]["acceptance_policy"]["policy"] == \
        "camera_blockage_reject_then_tier"
    assert result["bindings"]["acceptance_policy"]["source"] == "params"


def test_tier_policy_rejects_only_camera_side_blockage():
    """card1B_010-like: a floor-lamp shade 0.3 m from the lens hides the dog."""
    truth = _truth_with_frames(
        _card1_frame(40, pixels=0, fraction=0.0, state="fully_occluded"),
        _card1_frame(74, pixels=8735, fraction=0.967),
        _card1_frame(40, pixels=1477, fraction=0.528),
        _card1_frame(74, pixels=0, fraction=0.0, state="fully_occluded"))
    arrays = _depth_arrays(truth, {("source2", 40): (5.5, 0.3, 4),
                                   ("source1", 74): (6.8, 4.1, 4)})
    result = evaluate(_card1_fact(), truth, TIER_PARAMS, arrays)
    assert result["status"] == "pixel_rejected"
    assert result["rejection_reasons"] == [
        "main_referent_anchor_frame_camera_side_blockage"]
    difficulty = result["bindings"]["difficulty"]
    assert difficulty["worst_tier"] == "hidden"
    assert difficulty["anchor_instant_hidden"] is True
    assert difficulty["query_instant_hidden"] is True
    # the far occluder (4.1 m) at the Gate A query frame stays a tier, not a reject
    assert result["bindings"]["evaluations"]["gatea"]["query_frame"][
        "camera_side_blockage"] is False
    # the historical policy on the same evidence rejects both hidden frames
    strict = evaluate(_card1_fact(), truth, PIXEL_PARAMS, arrays)
    assert strict["status"] == "pixel_rejected"
    assert "gatea_referent_query_frame_not_visible_state" in strict["rejection_reasons"]
    assert strict["bindings"]["difficulty"]["worst_tier"] == "hidden"


def test_tier_policy_rejects_only_a_completely_blocked_referent():
    """Owner 2026-09-03: a question survives unless the referent is 100% blocked.

    Same evidence as the far-curtain case: the Gate A referent keeps 198 pixels
    (10.5%) at the query frame, so it stays a heavy tier and the candidate is
    kept.  A referent with nothing visible at a declared frame is rejected,
    because that instant cannot be answered at all.
    """
    kept = _truth_with_frames(
        _card1_frame(40, pixels=9720, fraction=0.817),
        _card1_frame(74, pixels=8735, fraction=0.967),
        _card1_frame(40, pixels=1477, fraction=0.528),
        _card1_frame(74, pixels=198, fraction=0.105))
    arrays = _depth_arrays(kept, {("source1", 74): (4.5, 4.2, 3)})
    result = evaluate(_card1_fact(), kept, OWNER_TIER_PARAMS, arrays)
    assert result["status"] == "pass"
    assert result["rejection_reasons"] == []
    assert result["bindings"]["difficulty"]["worst_tier"] == "heavy"
    assert result["bindings"]["difficulty"]["reject_tiers"] == ["hidden", "out_of_view"]

    # nothing visible at the Gate A query frame, and the occluder is far from
    # the lens, so the 2026-09-02 policy kept it and this rule does not
    blocked = _truth_with_frames(
        _card1_frame(40, pixels=9720, fraction=0.817),
        _card1_frame(74, pixels=8735, fraction=0.967),
        _card1_frame(40, pixels=1477, fraction=0.528),
        _card1_frame(74, pixels=0, fraction=0.0, state="fully_occluded"))
    arrays = _depth_arrays(blocked, {("source1", 74): (4.5, 4.2, 4)})
    lenient = evaluate(_card1_fact(), blocked, TIER_PARAMS, arrays)
    assert lenient["status"] == "pass"
    strict = evaluate(_card1_fact(), blocked, OWNER_TIER_PARAMS, arrays)
    assert strict["status"] == "pixel_rejected"
    assert strict["rejection_reasons"] == ["gatea_referent_query_frame_hidden"]
    assert strict["bindings"]["difficulty"]["tiers"]["gatea"]["query_frame"] == "hidden"

    # a referent that walked out of frame at a declared instant rejects too
    gone = _truth_with_frames(
        _card1_frame(40, pixels=9720, fraction=0.817),
        _card1_frame(74, pixels=0, fraction=None, state="out_of_view"),
        _card1_frame(40, pixels=1477, fraction=0.528),
        _card1_frame(74, pixels=8735, fraction=0.967))
    arrays = _depth_arrays(gone, {})
    out = evaluate(_card1_fact(), gone, OWNER_TIER_PARAMS, arrays)
    assert out["status"] == "pixel_rejected"
    assert out["rejection_reasons"] == ["main_referent_query_frame_out_of_view"]


def test_reject_tiers_default_to_none_and_names_must_be_known_tiers():
    """Owner 2026-09-03 declined rejecting completely blocked referents, so the
    default list is empty: only a camera-side blockage rejects."""
    base = {k: v for k, v in OWNER_TIER_PARAMS.items()
            if k != "PIXEL_TIER_REJECT_TIERS"}
    assert pixel_policy_from_params(base)["reject_tiers"] == []
    with pytest.raises(ValueError, match="unknown tiers"):
        pixel_policy_from_params(dict(base, PIXEL_TIER_REJECT_TIERS=["invisible"]))
    with pytest.raises(ValueError, match="repeats"):
        pixel_policy_from_params(dict(base, PIXEL_TIER_REJECT_TIERS=["hidden", "hidden"]))
    with pytest.raises(ValueError, match="list of tier names"):
        pixel_policy_from_params(dict(base, PIXEL_TIER_REJECT_TIERS="hidden"))
    assert pixel_policy_from_params(OWNER_TIER_PARAMS)["reject_tiers"] == [
        "hidden", "out_of_view"]
    assert pixel_policy_from_params(TIER_PARAMS)["reject_tiers"] == []


def test_a_fact_designed_before_the_rule_cannot_be_re_judged_silently():
    """A pre-2026-09-03 fact carries no reject_tiers; applying the new rule to it
    must fail loudly instead of passing the candidate under the old list."""
    truth = _truth_with_frames(
        _card1_frame(40, pixels=9720, fraction=0.817),
        _card1_frame(74, pixels=8735, fraction=0.967),
        _card1_frame(40, pixels=1477, fraction=0.528),
        _card1_frame(74, pixels=0, fraction=0.0, state="fully_occluded"))
    arrays = _depth_arrays(truth, {("source1", 74): (4.5, 4.2, 4)})
    legacy = _card1_fact(with_thresholds=True)
    legacy["pixel_acceptance"]["acceptance_policy"] = {
        "policy": "camera_blockage_reject_then_tier",
        "status": "placeholder_research",
        "camera_blockage_max_distance_m": 1.5,
        "tier_visible_fraction_edges": [0.5, 0.2]}
    with pytest.raises(ValueError, match="designed before"):
        evaluate(legacy, truth, OWNER_TIER_PARAMS, arrays)
    # re-judging it under the list it was designed with reproduces the verdict
    kept = evaluate(legacy, truth, TIER_PARAMS, arrays)
    assert kept["status"] == "pass"
    assert kept["bindings"]["difficulty"]["reject_tiers"] == []
    # a fact that declares a different list than the params is also loud
    newer = _card1_fact(with_thresholds=True)
    newer["pixel_acceptance"]["acceptance_policy"] = dict(
        legacy["pixel_acceptance"]["acceptance_policy"], reject_tiers=["out_of_view"])
    with pytest.raises(ValueError, match="reject_tiers differs"):
        evaluate(newer, truth, OWNER_TIER_PARAMS, arrays)


def test_tier_policy_fails_closed_without_depth_arrays_and_on_policy_mismatch():
    truth = _truth_with_frames(
        _card1_frame(40, pixels=9720, fraction=0.817),
        _card1_frame(74, pixels=8735, fraction=0.967),
        _card1_frame(40, pixels=1477, fraction=0.528),
        _card1_frame(74, pixels=1400, fraction=0.7))
    with pytest.raises(ValueError, match="depth arrays"):
        evaluate(_card1_fact(), truth, TIER_PARAMS)
    fact = _card1_fact(with_thresholds=True)
    fact["pixel_acceptance"]["acceptance_policy"] = {
        "policy": "both_frames_threshold_reject"}
    arrays = _depth_arrays(truth, {})
    with pytest.raises(ValueError, match="acceptance policy policy differs"):
        evaluate(fact, truth, TIER_PARAMS, arrays)
    with pytest.raises(ValueError, match="explicit"):
        pixel_policy_from_params(dict(PIXEL_PARAMS,
                                      PIXEL_ACCEPTANCE_POLICY="camera_blockage_reject_then_tier"))
    with pytest.raises(ValueError, match="unknown PIXEL_ACCEPTANCE_POLICY"):
        pixel_policy_from_params(dict(PIXEL_PARAMS, PIXEL_ACCEPTANCE_POLICY="x"))
    block = card1_pixel_acceptance_block(
        TIER_PARAMS, target_slot="source2", other_slot="source1",
        anchor_frame=40, query_frame=74)
    assert block["acceptance_policy"]["camera_blockage_max_distance_m"] == 1.5
    assert tier_for_frame("visible_occluded", 0.49, [0.5, 0.2]) == "medium"
    assert tier_for_frame("visible_occluded", 0.1, [0.5, 0.2]) == "heavy"
    assert tier_for_frame("fully_occluded", 0.0, [0.5, 0.2]) == "hidden"
    assert tier_for_frame("out_of_view", None, [0.5, 0.2]) == "out_of_view"


def test_visibility_timeline_uses_every_captured_frame():
    truth = _truth_with_frames(
        _card1_frame(40, pixels=9720, fraction=0.817),
        _card1_frame(74, pixels=8735, fraction=0.967),
        _card1_frame(40, pixels=1477, fraction=0.528),
        _card1_frame(74, pixels=0, fraction=0.0, state="fully_occluded"))
    truth["per_instance"]["source1"]["frames"] += [
        _card1_frame(60, pixels=0, fraction=0.0, state="fully_occluded"),
        _card1_frame(70, pixels=0, fraction=0.0, state="fully_occluded"),
        _card1_frame(20, pixels=800, fraction=0.3),
    ]
    truth["frame_indices"] = [20, 40, 60, 70, 74]
    timeline = visibility_timeline(truth, "source1", 40, 74)
    assert timeline["captured_frame_indices"] == [20, 40, 60, 70, 74]
    assert timeline["visible_frame_count"] == 2
    assert timeline["visible_frame_fraction"] == pytest.approx(0.4)
    assert timeline["hidden_captured_frames_ending_at_query"] == 3
    assert timeline["nearest_visible_captured_frame_distance_to_anchor"] == 0
    assert occluder_statistics(None, truth, "source1", 74) is None


def test_cli_reads_depth_arrays_beside_the_truth(tmp_path):
    truth = _truth_with_frames(
        _card1_frame(40, pixels=9720, fraction=0.817),
        _card1_frame(74, pixels=8735, fraction=0.967),
        _card1_frame(40, pixels=1477, fraction=0.528),
        _card1_frame(74, pixels=198, fraction=0.105))
    arrays = _depth_arrays(truth, {("source1", 74): (4.5, 4.2, 3)})
    (tmp_path / "pixel_visibility_truth.json").write_text(json.dumps(truth))
    np.savez_compressed(tmp_path / "native_depth_and_object_ids.npz", **arrays)
    fact = tmp_path / "fact.json"; fact.write_text(json.dumps(_card1_fact()))
    params = tmp_path / "params.json"; params.write_text(json.dumps(TIER_PARAMS))
    output = tmp_path / "join.json"
    assert main(["--fact", str(fact),
                 "--pixel-truth", str(tmp_path / "pixel_visibility_truth.json"),
                 "--params", str(params), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["status"] == "pass"
    assert result["bindings"]["difficulty"]["worst_tier"] == "heavy"
    assert result["inputs"]["pixel_arrays"]["path"].endswith(
        "native_depth_and_object_ids.npz")


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


def test_visibility_timeline_declares_itself_sampled():
    from join_qa_v3_extended_pixel import visibility_timeline
    truth = {"per_instance": {"source1": {"frames": [
        {"frame_index": f, "state": "visible_clear" if f >= 20 else "fully_occluded",
         "visible_pixels": 500 if f >= 20 else 0}
        for f in range(0, 75, 5)] + [{"frame_index": 74, "state": "visible_clear",
                                     "visible_pixels": 300}]}}}
    timeline = visibility_timeline(truth, "source1", 40, 74)
    assert timeline["sampling"] == "captured_frames_only_not_every_clip_frame"
    assert timeline["capture_stride_frames"] == [4, 5]
    assert timeline["captured_frame_count"] == 16
    assert timeline["visible_frame_count"] == 12
    assert timeline["hidden_captured_frames_ending_at_query"] == 0

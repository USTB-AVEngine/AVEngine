#!/usr/bin/env python3
"""Join native pixel truth to pixel-dependent QA-v3 candidates.

Extended profiles (card11 / card15a / card16) bind their pixel states at fixed
frames.  Base cross-time profiles (card1F / card1B) are checked on **both
sides**: the main referent and the Gate A referent each have to be bindable at
the identity-anchor frame and identifiable at the visual-query frame.  The
thresholds are explicit placeholders that must travel through params, the fact
record and this join output; they are not human-calibrated admission values.
Line-of-sight screening is a search-time prefilter and is never accepted here
as pixel evidence.

Two acceptance policies exist for card1F / card1B.  The default (owner
decision 2026-09-02) is ``camera_blockage_reject_then_tier``: partially or
momentarily hidden dogs are difficulty tiers and a candidate is rejected only
when the occluder sits close to the lens (camera-side blockage), which needs
the captured depth arrays next to the pixel truth.  The historical
``both_frames_threshold_reject`` policy, which rejects any referent/frame below
the thresholds, stays selectable by name.  Every captured frame also feeds a
per-referent visibility timeline so tiers can later use whole-clip evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import sha256_file
from design_qa_v3_extended_profile import CARD11_BINDING_FRAME
from qa_v3_pixel_thresholds import (
    CARD1_FRAME_REQUIREMENTS,
    LINE_OF_SIGHT_ROLE,
    PIXEL_POLICY_THRESHOLD_REJECT,
    PIXEL_POLICY_TIER,
    PIXEL_THRESHOLD_KEYS,
    PIXEL_THRESHOLD_STATUS_DEFAULT,
    TIER_ORDER,
    pixel_policy_from_params,
    pixel_thresholds_from_params,
    tier_for_frame,
)

DEPTH_ARRAYS_FILENAME = "native_depth_and_object_ids.npz"
BLOCKAGE_TIERS = ("medium", "heavy", "hidden")


SUPPORTED = {"card11", "card15a", "card16", "card1F", "card1B"}
CARD1 = {"card1F", "card1B"}
VISIBLE = {"visible_clear", "visible_occluded"}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _state_by_frame(pixel_truth):
    result = {}
    for slot, record in pixel_truth.get("per_instance", {}).items():
        result[slot] = {
            int(frame["frame_index"]): str(frame["state"])
            for frame in record.get("frames", [])
        }
    return result


def _frames_by_index(pixel_truth):
    result = {}
    for slot, record in pixel_truth.get("per_instance", {}).items():
        result[slot] = {
            int(frame["frame_index"]): frame
            for frame in record.get("frames", [])
        }
    return result


def card1_pixel_thresholds(fact, params=None):
    """Resolve card1 thresholds from the fact and/or params; both must agree."""
    from_fact = (fact.get("pixel_acceptance") or {}).get("thresholds")
    from_params = pixel_thresholds_from_params(params) if params is not None else None
    if from_fact is None and from_params is None:
        raise ValueError(
            "card1 pixel join needs explicit thresholds: either the fact "
            "carries pixel_acceptance.thresholds or --params supplies "
            f"{list(PIXEL_THRESHOLD_KEYS)}")
    if from_fact is not None and from_params is not None:
        for key in ("min_visible_fraction", "min_visible_pixels",
                    "bbox_must_not_touch_frame_edge"):
            if from_fact.get(key) != from_params[key]:
                raise ValueError(
                    f"pixel threshold {key} differs between the fact "
                    f"({from_fact.get(key)!r}) and params ({from_params[key]!r})")
    chosen = dict(from_fact if from_fact is not None else from_params)
    for key in ("min_visible_fraction", "min_visible_pixels",
                "bbox_must_not_touch_frame_edge"):
        if key not in chosen:
            raise ValueError(f"pixel thresholds lack {key}")
    chosen.setdefault("status", PIXEL_THRESHOLD_STATUS_DEFAULT)
    chosen["source"] = ("fact_pixel_acceptance" if from_fact is not None
                        else "params")
    return chosen


def card1_pixel_policy(fact, params=None):
    """Resolve the acceptance policy from the fact and/or params (must agree)."""
    from_fact = (fact.get("pixel_acceptance") or {}).get("acceptance_policy")
    from_params = pixel_policy_from_params(params) if params is not None else None
    if from_fact is not None and from_params is not None:
        for key in ("policy", "camera_blockage_max_distance_m",
                    "tier_visible_fraction_edges"):
            if from_fact.get(key) != from_params.get(key):
                raise ValueError(
                    f"acceptance policy {key} differs between the fact "
                    f"({from_fact.get(key)!r}) and params "
                    f"({from_params.get(key)!r})")
        # 2026-09-03: which tiers reject is part of the policy, and the fact's
        # embedded policy wins below.  A fact designed before this key existed
        # would otherwise be re-judged under its own (empty) list while the
        # operator believes the params applied — the mismatch must be loud.
        fact_reject = from_fact.get("reject_tiers")
        params_reject = list(from_params.get("reject_tiers") or [])
        if fact_reject is None:
            if params_reject:
                raise ValueError(
                    "this fact was designed before PIXEL_TIER_REJECT_TIERS "
                    f"existed, so its effective list is [] while params ask for "
                    f"{params_reject}. Either regenerate the candidate with the "
                    "new params, or re-judge the existing evidence with "
                    "PIXEL_TIER_REJECT_TIERS: [] to reproduce the original "
                    "verdict; the join will not silently apply a rule the "
                    "candidate was not designed under")
        elif list(fact_reject) != params_reject:
            raise ValueError(
                f"acceptance policy reject_tiers differs between the fact "
                f"({list(fact_reject)!r}) and params ({params_reject!r})")
    if from_fact is None and from_params is None:
        raise ValueError(
            "card1 pixel join needs an acceptance policy: either the fact "
            "carries pixel_acceptance.acceptance_policy or --params names "
            "PIXEL_ACCEPTANCE_POLICY (default is the tier policy, which also "
            "needs PIXEL_CAMERA_BLOCKAGE_MAX_DISTANCE_M and "
            "PIXEL_TIER_VISIBLE_FRACTION_EDGES)")
    chosen = dict(from_fact if from_fact is not None else from_params)
    chosen["source"] = ("fact_pixel_acceptance" if from_fact is not None
                        else "params")
    return chosen


def load_depth_arrays(pixel_truth_path, arrays_path=None):
    """Depth arrays saved next to the pixel truth by the capture tool."""
    path = Path(arrays_path) if arrays_path else (
        Path(pixel_truth_path).parent / DEPTH_ARRAYS_FILENAME)
    if not path.is_file():
        return None, None
    with np.load(path) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    return arrays, path


def occluder_statistics(arrays, pixel_truth, slot, frame_index):
    """What hides the referent: median depth of the occluding pixels."""
    if arrays is None:
        return None
    frames = [int(value) for value in pixel_truth.get("frame_indices", [])]
    if frame_index not in frames:
        return None
    key = f"target_only_{slot}_depth_m"
    if key not in arrays or "normal_depth_m" not in arrays:
        return None
    position = frames.index(frame_index)
    target = np.asarray(arrays[key][position], dtype=np.float64)
    normal = np.asarray(arrays["normal_depth_m"][position], dtype=np.float64)
    comparison = pixel_truth.get("depth_comparison") or {}
    sentinel = float(comparison.get("target_only_background_depth_m", 65504.0))
    abs_tol = float(comparison.get("absolute_tolerance_m", 0.01))
    rel_tol = float(comparison.get("relative_tolerance", 0.002))
    footprint = target < sentinel * 0.5
    if not footprint.any():
        return {"footprint_pixels": 0, "hidden_pixels": 0,
                "hidden_fraction": None, "occluder_median_depth_m": None,
                "target_median_depth_m": None}
    hidden = footprint & (normal < target - (abs_tol + rel_tol * target))
    return {
        "footprint_pixels": int(footprint.sum()),
        "hidden_pixels": int(hidden.sum()),
        "hidden_fraction": float(hidden.sum() / footprint.sum()),
        "occluder_median_depth_m": (
            float(np.median(normal[hidden])) if hidden.any() else None),
        "target_median_depth_m": float(np.median(target[footprint])),
    }


def visibility_timeline(pixel_truth, slot, anchor_frame, query_frame):
    """Whole-clip visibility of one referent over every captured frame."""
    frames = sorted(
        (int(frame["frame_index"]), frame)
        for frame in (pixel_truth.get("per_instance", {}).get(slot, {})
                      .get("frames", [])))
    captured = [index for index, _ in frames]
    visible = [index for index, frame in frames
               if frame.get("state") in VISIBLE and (frame.get("visible_pixels") or 0) > 0]
    hidden_run = 0
    for index, frame in reversed(frames):
        if index > query_frame:
            continue
        if frame.get("state") in VISIBLE and (frame.get("visible_pixels") or 0) > 0:
            break
        hidden_run += 1
    nearest_visible_to_anchor = (
        min(abs(index - anchor_frame) for index in visible) if visible else None)
    strides = sorted({b - a for a, b in zip(captured, captured[1:])})
    return {
        "sampling": "captured_frames_only_not_every_clip_frame",
        "captured_frame_indices": captured,
        "captured_frame_count": len(captured),
        "capture_stride_frames": (strides[0] if len(strides) == 1 else strides) if strides else None,
        "visible_frame_count": len(visible),
        "visible_frame_fraction": (len(visible) / len(captured)
                                   if captured else None),
        "hidden_captured_frames_ending_at_query": hidden_run,
        "nearest_visible_captured_frame_distance_to_anchor": nearest_visible_to_anchor,
        "note": ("metrics are over the captured frames only (a sampled "
                 "timeline, typically every 5th frame); capture every frame "
                 "for a complete one"),
    }


def frame_pixel_conditions(frame, resolution_hw, thresholds):
    """Evaluate one referent at one frame against the explicit thresholds."""
    height, width = [int(value) for value in resolution_hw]
    state = frame.get("state")
    fraction = frame.get("visible_fraction")
    pixels = frame.get("visible_pixels")
    bbox = frame.get("target_bbox_xyxy_px")
    bbox_inside = None
    if bbox is not None:
        x0, y0, x1, y1 = [int(value) for value in bbox]
        bbox_inside = bool(x0 > 0 and y0 > 0 and x1 < width and y1 < height)
    conditions = {
        "state": state,
        "visible_fraction": fraction,
        "visible_pixels": pixels,
        "target_bbox_xyxy_px": bbox,
        "bbox_inside_frame": bbox_inside,
        "visible_state": state in VISIBLE,
        "visible_fraction_ok": (
            fraction is not None
            and float(fraction) >= float(thresholds["min_visible_fraction"])),
        "visible_pixels_ok": (
            pixels is not None
            and int(pixels) >= int(thresholds["min_visible_pixels"])),
        "bbox_ok": (
            bool(bbox_inside)
            if thresholds["bbox_must_not_touch_frame_edge"] else True),
    }
    failures = []
    if not conditions["visible_state"]:
        failures.append("not_visible_state")
    if not conditions["visible_fraction_ok"]:
        failures.append("visible_fraction_below_threshold")
    if not conditions["visible_pixels_ok"]:
        failures.append("visible_pixels_below_threshold")
    if not conditions["bbox_ok"]:
        failures.append("bbox_touches_frame_edge")
    conditions["failures"] = failures
    conditions["passed"] = not failures
    return conditions


def evaluate_card1(fact, pixel_truth, params=None, arrays=None):
    thresholds = card1_pixel_thresholds(fact, params)
    policy = card1_pixel_policy(fact, params)
    tier_policy = policy["policy"] == PIXEL_POLICY_TIER
    if tier_policy and arrays is None:
        raise ValueError(
            "the tier policy needs the captured depth arrays "
            f"({DEPTH_ARRAYS_FILENAME}) to tell camera-side blockage from "
            "scene occlusion; none were supplied")
    resolution = pixel_truth.get("resolution_hw")
    if resolution is None:
        raise ValueError("pixel truth lacks resolution_hw; bbox edge test "
                         "cannot run")
    edges = policy.get("tier_visible_fraction_edges") or [0.5, 0.2]
    reject_tiers = tuple(policy.get("reject_tiers") or ())
    blockage_distance = policy.get("camera_blockage_max_distance_m")
    target_slot = str(fact["target_slot"])
    other_slot = "source2" if target_slot == "source1" else "source1"
    referents = {"main": target_slot, "gatea": other_slot}
    frames = {"anchor_frame": int(fact["anchor_frame"]),
              "query_frame": int(fact["query_frame"])}
    by_index = _frames_by_index(pixel_truth)
    reasons = []
    evaluations = {}
    tiers = {}
    timelines = {}
    below_threshold_frames = 0
    frame_edge_cut = False
    for side, slot in referents.items():
        evaluations[side] = {"slot": slot}
        tiers[side] = {}
        timelines[side] = dict(visibility_timeline(
            pixel_truth, slot, frames["anchor_frame"], frames["query_frame"]),
            slot=slot)
        for role, index in frames.items():
            record = by_index.get(slot, {}).get(index)
            if record is None:
                reasons.append(f"{side}_referent_{role}_missing_in_pixel_truth")
                evaluations[side][role] = {
                    "frame_index": index, "passed": False,
                    "failures": ["missing_in_pixel_truth"]}
                tiers[side][role] = None
                continue
            conditions = frame_pixel_conditions(record, resolution, thresholds)
            conditions["frame_index"] = index
            conditions["requirement"] = CARD1_FRAME_REQUIREMENTS[role]
            tier = tier_for_frame(conditions["state"],
                                  conditions["visible_fraction"], edges)
            conditions["tier"] = tier
            conditions["occluder"] = occluder_statistics(
                arrays, pixel_truth, slot, index)
            occluder_depth = (conditions["occluder"] or {}).get(
                "occluder_median_depth_m")
            conditions["camera_side_blockage"] = bool(
                blockage_distance is not None
                and tier in BLOCKAGE_TIERS
                and occluder_depth is not None
                and occluder_depth <= blockage_distance)
            evaluations[side][role] = conditions
            tiers[side][role] = tier
            if not conditions["passed"]:
                below_threshold_frames += 1
            if conditions["bbox_inside_frame"] is False:
                frame_edge_cut = True
            if tier_policy:
                if conditions["camera_side_blockage"]:
                    reasons.append(f"{side}_referent_{role}_camera_side_blockage")
                # owner 2026-09-03:留下题目的底线是"不是百分百被挡住";完全看不见的
                # 那一帧没法答,所以这些档位拒题,其余照旧只记难度。
                if tier in reject_tiers:
                    reasons.append(f"{side}_referent_{role}_{tier}")
            else:
                for failure in conditions["failures"]:
                    reasons.append(f"{side}_referent_{role}_{failure}")
    assigned = [tier for per_side in tiers.values() for tier in per_side.values()
                if tier is not None]
    difficulty = {
        "tiers": tiers,
        "tier_order": list(TIER_ORDER),
        "worst_tier": (max(assigned, key=TIER_ORDER.index) if assigned else None),
        "anchor_instant_hidden": any(
            tiers[side].get("anchor_frame") in ("hidden", "out_of_view")
            for side in referents),
        "query_instant_hidden": any(
            tiers[side].get("query_frame") in ("hidden", "out_of_view")
            for side in referents),
        "frame_edge_cut": frame_edge_cut,
        "referent_frames_below_placeholder_thresholds": below_threshold_frames,
        "tier_visible_fraction_edges": edges,
        "reject_tiers": list(reject_tiers),
        "status": policy.get("status", PIXEL_THRESHOLD_STATUS_DEFAULT),
        "note": ("tiers are research placeholders derived from the two "
                 "declared frames; they are recorded under both policies and "
                 "gate nothing under the threshold policy"),
    }
    bindings = {
        "anchor_frame": frames["anchor_frame"],
        "query_frame": frames["query_frame"],
        "main_referent_slot": target_slot,
        "gatea_referent_slot": other_slot,
        "requirements": dict(CARD1_FRAME_REQUIREMENTS),
        "thresholds": thresholds,
        "threshold_status": thresholds["status"],
        "acceptance_policy": policy,
        "difficulty": difficulty,
        "visibility_timeline_sampled": timelines,
        "occluder_statistics_available": arrays is not None,
        "evaluations": evaluations,
        "line_of_sight_role": LINE_OF_SIGHT_ROLE,
        "mcq_truth_option": fact["mcq"]["truth_option"],
        "open_truth_value": fact["open"]["truth_value"],
    }
    return bindings, reasons


def evaluate(fact, pixel_truth, params=None, arrays=None):
    profile_id = fact.get("profile_id")
    if profile_id not in SUPPORTED:
        raise ValueError(f"unsupported pixel join profile: {profile_id!r}")
    states = _state_by_frame(pixel_truth)
    reasons = []
    bindings = {}
    if profile_id in CARD1:
        bindings, reasons = evaluate_card1(fact, pixel_truth, params, arrays)
    elif profile_id == "card11":
        frame = CARD11_BINDING_FRAME
        visible = [states.get(f"source{i}", {}).get(frame) for i in range(1, 4)]
        hidden = states.get("source4", {}).get(frame)
        if any(value not in VISIBLE for value in visible):
            reasons.append("one_of_three_visible_candidates_not_visible")
        if hidden != "fully_occluded":
            reasons.append("offscreen_candidate_is_visually_present")
        bindings = {
            "query_frame": frame,
            "visible_candidate_states": visible,
            "offscreen_source4_state": hidden,
            "mcq_truth_option": fact["mcq"]["truth_option"],
            "open_truth_value": fact["open"]["truth_value"],
        }
    elif profile_id == "card15a":
        frame = 30
        values = [states.get(f"source{i}", {}).get(frame) for i in range(1, 5)]
        if any(value not in VISIBLE for value in values):
            reasons.append("not_all_four_actors_visible")
        bindings = {
            "query_frame": frame,
            "actor_states": values,
            "in_scene_count": 4,
            "distinct_callers": fact["open"]["truth_value"][1],
            "mcq_truth_option": fact["mcq"]["truth_option"],
            "open_truth_value": fact["open"]["truth_value"],
        }
    else:
        binding_frame = 12
        frame = 74
        source1_binding = states.get("source1", {}).get(binding_frame)
        source2_binding = states.get("source2", {}).get(binding_frame)
        source1 = states.get("source1", {}).get(frame)
        source2 = states.get("source2", {}).get(frame)
        if source1_binding not in VISIBLE:
            reasons.append("main_first_caller_not_visible_at_binding_frame")
        if source2_binding not in VISIBLE:
            reasons.append("gatea_first_caller_not_visible_at_binding_frame")
        if source1 is None or source2 is None:
            reasons.append("missing_final_state")
        elif source1 == source2:
            reasons.append("first_caller_and_counterfactual_have_same_final_state")
        bindings = {
            "binding_frame": binding_frame,
            "query_frame": frame,
            "main_first_caller_slot": "source1",
            "main_binding_state": source1_binding,
            "main_truth_option": source1,
            "gatea_first_caller_slot": "source2",
            "gatea_binding_state": source2_binding,
            "gatea_truth_option": source2,
            "open_truth_value": source1,
        }
    passed = not reasons
    return {
        "schema": "qa_v3_extended_pixel_join_v1",
        "status": "pass" if passed else "pixel_rejected",
        "profile_id": profile_id,
        "point_id": fact.get("point_id"),
        "evidence_class": (
            "pixel_qualified_candidate" if passed else "pixel_rejected"),
        "qualification_claim": False,
        "pixel_authority": pixel_truth.get("authority"),
        "bindings": bindings,
        "rejection_reasons": reasons,
        "boundary": (
            "Native pixel join for one research candidate; does not establish "
            "single-modality resistance or dataset admission."),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fact", required=True, type=Path)
    parser.add_argument("--pixel-truth", required=True, type=Path)
    parser.add_argument("--params", type=Path, default=None,
                        help="explicit pixel thresholds for card1F/card1B; "
                             "required when the fact carries none")
    parser.add_argument("--pixel-arrays", type=Path, default=None,
                        help="depth arrays saved by the capture tool; defaults "
                             f"to {DEPTH_ARRAYS_FILENAME} beside the truth")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        print(f"refusing to overwrite: {args.output}", file=sys.stderr)
        return 2
    fact_path = args.fact.resolve()
    pixel_path = args.pixel_truth.resolve()
    params = _read(args.params.resolve()) if args.params else None
    arrays, arrays_path = load_depth_arrays(pixel_path, args.pixel_arrays)
    try:
        result = evaluate(_read(fact_path), _read(pixel_path), params, arrays)
    except ValueError as exc:
        print(f"pixel join refused: {exc}", file=sys.stderr)
        return 2
    result["inputs"] = {
        "fact": {
            "path": str(fact_path),
            "sha256": sha256_file(fact_path),
        },
        "pixel_truth": {
            "path": str(pixel_path),
            "sha256": sha256_file(pixel_path),
        },
    }
    if args.params:
        result["inputs"]["params"] = {
            "path": str(args.params.resolve()),
            "sha256": sha256_file(args.params.resolve()),
        }
    if arrays_path is not None:
        result["inputs"]["pixel_arrays"] = {
            "path": str(arrays_path.resolve()),
            "sha256": sha256_file(arrays_path.resolve()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "profile_id": result["profile_id"],
        "rejection_reasons": result["rejection_reasons"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

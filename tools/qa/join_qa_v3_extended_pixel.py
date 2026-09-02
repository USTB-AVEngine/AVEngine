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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from avengine.contracts.json_io import sha256_file
from design_qa_v3_extended_profile import CARD11_BINDING_FRAME
from qa_v3_pixel_thresholds import (
    CARD1_FRAME_REQUIREMENTS,
    LINE_OF_SIGHT_ROLE,
    PIXEL_THRESHOLD_KEYS,
    PIXEL_THRESHOLD_STATUS_DEFAULT,
    pixel_thresholds_from_params,
)


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


def evaluate_card1(fact, pixel_truth, params=None):
    thresholds = card1_pixel_thresholds(fact, params)
    resolution = pixel_truth.get("resolution_hw")
    if resolution is None:
        raise ValueError("pixel truth lacks resolution_hw; bbox edge test "
                         "cannot run")
    target_slot = str(fact["target_slot"])
    other_slot = "source2" if target_slot == "source1" else "source1"
    referents = {"main": target_slot, "gatea": other_slot}
    frames = {"anchor_frame": int(fact["anchor_frame"]),
              "query_frame": int(fact["query_frame"])}
    by_index = _frames_by_index(pixel_truth)
    reasons = []
    evaluations = {}
    for side, slot in referents.items():
        evaluations[side] = {"slot": slot}
        for role, index in frames.items():
            record = by_index.get(slot, {}).get(index)
            if record is None:
                reasons.append(f"{side}_referent_{role}_missing_in_pixel_truth")
                evaluations[side][role] = {
                    "frame_index": index, "passed": False,
                    "failures": ["missing_in_pixel_truth"]}
                continue
            conditions = frame_pixel_conditions(record, resolution, thresholds)
            conditions["frame_index"] = index
            conditions["requirement"] = CARD1_FRAME_REQUIREMENTS[role]
            evaluations[side][role] = conditions
            for failure in conditions["failures"]:
                reasons.append(f"{side}_referent_{role}_{failure}")
    bindings = {
        "anchor_frame": frames["anchor_frame"],
        "query_frame": frames["query_frame"],
        "main_referent_slot": target_slot,
        "gatea_referent_slot": other_slot,
        "requirements": dict(CARD1_FRAME_REQUIREMENTS),
        "thresholds": thresholds,
        "threshold_status": thresholds["status"],
        "evaluations": evaluations,
        "line_of_sight_role": LINE_OF_SIGHT_ROLE,
        "mcq_truth_option": fact["mcq"]["truth_option"],
        "open_truth_value": fact["open"]["truth_value"],
    }
    return bindings, reasons


def evaluate(fact, pixel_truth, params=None):
    profile_id = fact.get("profile_id")
    if profile_id not in SUPPORTED:
        raise ValueError(f"unsupported pixel join profile: {profile_id!r}")
    states = _state_by_frame(pixel_truth)
    reasons = []
    bindings = {}
    if profile_id in CARD1:
        bindings, reasons = evaluate_card1(fact, pixel_truth, params)
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
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        print(f"refusing to overwrite: {args.output}", file=sys.stderr)
        return 2
    fact_path = args.fact.resolve()
    pixel_path = args.pixel_truth.resolve()
    params = _read(args.params.resolve()) if args.params else None
    try:
        result = evaluate(_read(fact_path), _read(pixel_path), params)
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

#!/usr/bin/env python3
"""Join native pixel truth to pixel-dependent QA-v3 extended candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SUPPORTED = {"card11", "card15a", "card16"}
VISIBLE = {"visible_clear", "visible_occluded"}
HIDDEN = {"fully_occluded", "out_of_view"}


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


def evaluate(fact, pixel_truth):
    profile_id = fact.get("profile_id")
    if profile_id not in SUPPORTED:
        raise ValueError(f"unsupported pixel join profile: {profile_id!r}")
    states = _state_by_frame(pixel_truth)
    reasons = []
    bindings = {}
    if profile_id == "card11":
        frame = 30
        visible = [states.get(f"source{i}", {}).get(frame) for i in range(1, 4)]
        hidden = states.get("source4", {}).get(frame)
        if any(value not in VISIBLE for value in visible):
            reasons.append("one_of_three_visible_candidates_not_visible")
        if hidden not in HIDDEN:
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
        frame = 74
        source1 = states.get("source1", {}).get(frame)
        source2 = states.get("source2", {}).get(frame)
        if source1 is None or source2 is None:
            reasons.append("missing_final_state")
        elif source1 == source2:
            reasons.append("first_caller_and_counterfactual_have_same_final_state")
        bindings = {
            "query_frame": frame,
            "main_first_caller_slot": "source1",
            "main_truth_option": source1,
            "gatea_first_caller_slot": "source2",
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
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        print(f"refusing to overwrite: {args.output}", file=sys.stderr)
        return 2
    result = evaluate(_read(args.fact), _read(args.pixel_truth))
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

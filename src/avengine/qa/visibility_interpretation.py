"""Configured question-level visibility tolerance; raw renderer measurements stay intact."""
from __future__ import annotations
from copy import deepcopy
import math

QA_IDS = frozenset({"QA-07", "QA-09"})


def visibility_policy(facts):
    acceptance = (facts.get("sampling") or {}).get("acceptance_policy") or {}
    value = acceptance.get("visibility")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("acceptance_policy.visibility must be an object")
    result = dict(value)
    result["policy_id"] = str(acceptance.get("policy_id") or "configured_visibility_tolerance")
    for key in ("max_hidden_visible_fraction", "min_reappeared_visible_fraction",
                "max_edge_sliver_width_fraction"):
        number = result.get(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 <= number < 1:
            raise ValueError(f"visibility tolerance {key} must be a finite fraction in [0,1)")
    if result["max_hidden_visible_fraction"] >= result["min_reappeared_visible_fraction"]:
        raise ValueError("the clear-reappearance fraction must exceed the hidden fraction")
    return result


def interpret_rows(rows, *, resolution_hw, policy, qa_id):
    """Return a new keyed series. Tokens are QA interpretations, not new pixel truth."""
    output = deepcopy(rows)
    changed = 0
    hidden = False
    width = float(resolution_hw[1]) if resolution_hw and len(resolution_hw) == 2 else None
    for key in sorted(output, key=int):
        row = output[key]
        raw_state = row.get("raw_state", row.get("state"))
        row["state"] = raw_state
        row["raw_state"] = raw_state
        row["raw_in_fov"] = row.get("raw_in_fov", row.get("in_fov"))
        state = raw_state
        if qa_id == "QA-07" and width:
            box = row.get("target_bbox_xyxy_px")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                left, _, right, _ = map(float, box)
                touches_side = left <= 0 or right >= width
                if touches_side and 0 < right - left < width * policy["max_edge_sliver_width_fraction"]:
                    state = "out_of_view"
                    row["interpretation_reason"] = "edge_sliver_not_yet_clearly_in_frame"
        elif qa_id == "QA-09":
            fraction = row.get("visible_fraction")
            target, visible = row.get("target_pixels"), row.get("visible_pixels")
            if isinstance(target, (int, float)) and target > 0 and isinstance(visible, (int, float)):
                fraction = visible / target
            if raw_state == "fully_occluded":
                hidden = True
            if raw_state != "out_of_view" and isinstance(fraction, (int, float)) and math.isfinite(fraction) and 0 <= fraction <= 1:
                if fraction <= policy["max_hidden_visible_fraction"]:
                    hidden = True
                elif fraction >= policy["min_reappeared_visible_fraction"]:
                    hidden = False
                if hidden:
                    state = "fully_occluded"
                    row["interpretation_reason"] = "mostly_occluded_under_configured_tolerance"
        row["state"] = state
        row["in_fov"] = state != "out_of_view"
        row["interpretation_policy_id"] = policy["policy_id"]
        changed += int(state != raw_state)
    return output, {"qa_id": qa_id, "changed_frame_count": changed,
                    "policy": deepcopy(policy), "raw_measurements_preserved": True,
                    "claim_boundary": "Question-level tolerant interpretation, not strict pixel visibility."}


def prepare_facts_for_qa(facts, qa_id):
    policy = visibility_policy(facts)
    if policy is None or qa_id not in QA_IDS:
        return facts
    result = deepcopy(facts)
    summaries = {}
    resolution = (facts.get("visibility_meta") or {}).get("resolution_hw")
    for actor, rows in facts.get("visibility", {}).items():
        interpreted, report = interpret_rows(rows, resolution_hw=resolution, policy=policy, qa_id=qa_id)
        result["visibility"][actor] = interpreted
        summaries[actor] = report
    result["visibility_interpretation"] = {
        "qa_id": qa_id, "policy": deepcopy(policy), "actors": summaries,
        "claim_boundary": "Raw pixel counts are retained; states express the configured question tolerance."}
    return result

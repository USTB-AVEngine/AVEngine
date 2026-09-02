#!/usr/bin/env python3
"""Explicit placeholder pixel-answerability thresholds for QA-v3 base cards.

card1F / card1B need the main referent and the Gate A referent to be bindable
at the identity-anchor frame and identifiable at the visual-query frame.  The
thresholds below are research placeholders: they travel through params, the
fact record and the pixel join output verbatim, and they are not human
calibrated admission values.  Missing keys fail closed; nothing here supplies
an implicit default number.
"""

from __future__ import annotations

PIXEL_THRESHOLD_KEYS = (
    "PIXEL_MIN_VISIBLE_FRACTION",
    "PIXEL_MIN_VISIBLE_PIXELS",
    "PIXEL_BBOX_MUST_NOT_TOUCH_FRAME_EDGE",
)
PIXEL_THRESHOLD_STATUS_DEFAULT = "placeholder_research_not_human_calibrated"
CARD1_FRAME_REQUIREMENTS = {
    "anchor_frame": "referent_bindable_at_identity_anchor",
    "query_frame": "referent_identifiable_at_visual_query",
}
LINE_OF_SIGHT_ROLE = (
    "search-time prefilter only; not accepted as pixel answerability evidence")


def pixel_thresholds_from_params(params) -> dict:
    """Read the explicit placeholder thresholds; every key must be present."""
    missing = [key for key in PIXEL_THRESHOLD_KEYS if key not in params]
    if missing:
        raise ValueError(
            f"params missing explicit pixel thresholds {missing}; the card1 "
            "pixel acceptance cannot run on implicit defaults")
    fraction = float(params["PIXEL_MIN_VISIBLE_FRACTION"])
    pixels = int(params["PIXEL_MIN_VISIBLE_PIXELS"])
    if not 0.0 <= fraction <= 1.0 or pixels < 0:
        raise ValueError("pixel thresholds out of range")
    return {
        "min_visible_fraction": fraction,
        "min_visible_pixels": pixels,
        "bbox_must_not_touch_frame_edge": bool(
            params["PIXEL_BBOX_MUST_NOT_TOUCH_FRAME_EDGE"]),
        "status": str(params.get("PIXEL_THRESHOLD_STATUS",
                                 PIXEL_THRESHOLD_STATUS_DEFAULT)),
    }


def card1_pixel_acceptance_block(params, *, target_slot: str, other_slot: str,
                                 anchor_frame: int, query_frame: int) -> dict:
    """The fact-side declaration that the pixel join later verifies."""
    thresholds = pixel_thresholds_from_params(params)
    requirement = {
        "anchor_frame": {"frame_index": int(anchor_frame),
                         "must": CARD1_FRAME_REQUIREMENTS["anchor_frame"]},
        "query_frame": {"frame_index": int(query_frame),
                        "must": CARD1_FRAME_REQUIREMENTS["query_frame"]},
    }
    return {
        "policy": "card1_both_sides_native_pixel_join_v1",
        "status": thresholds["status"],
        "thresholds": thresholds,
        "referents": {
            "main": dict(requirement, referent_slot=target_slot),
            "gatea": dict(requirement, referent_slot=other_slot),
        },
        "line_of_sight_role": LINE_OF_SIGHT_ROLE,
        "join_tool": "tools/qa/join_qa_v3_extended_pixel.py",
        "pixel_status": "pending_native_pixel_join",
    }

"""Rule-based first-pass screening suggestions for room-area candidates.

This module consumes already-computed geometry facts. It does not calculate
floor area, infer room boundaries, modify source data, or write review verdicts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Suggestion = Literal[
    "usable_candidate",
    "discard_candidate",
    "crop_candidate",
    "scope_uncertain_hold",
    "unassessable_hold",
]

AREA_MIN_M2 = 10.0
LARGE_AREA_REVIEW_M2 = 50.0
MIN_COMPONENT_M2 = 0.5
MIN_LARGEST_COMPONENT_FRACTION = 0.5
MAX_ELONGATION = 5.0
MAX_MEDIAN_WIDTH_M = 0.8


@dataclass(frozen=True)
class ScreeningFacts:
    """Facts for one room-floor candidate; unknown values stay explicit."""

    area_m2: float | None
    area_assessable: bool = True
    scope_verified: bool = True
    unassigned_ground_upper_m2: float = 0.0
    room_body_present: bool | None = None
    stair_dominated: bool = False
    corridor_dominated: bool = False
    component_count_ge_min_area: int | None = None
    largest_component_fraction: float | None = None
    elongation_long_short: float | None = None
    largest_component_median_width_m: float | None = None
    owner_override: Literal["discard", "keep", "crop"] | None = None


@dataclass(frozen=True)
class ScreeningResult:
    suggestion: Suggestion
    reason_codes: tuple[str, ...]
    area_m2: float | None
    area_min_m2: float = AREA_MIN_M2
    large_area_review_m2: float = LARGE_AREA_REVIEW_M2


def _finite_nonnegative(value: float | None) -> bool:
    return value is not None and value >= 0


def screen_room(facts: ScreeningFacts) -> ScreeningResult:
    """Return a non-authoritative first-pass suggestion for one candidate.

    Decision order is deliberate: explicit owner feedback, data/scope quality,
    unsuitable dominant scene type, area minimum, shape, then large-area review.
    A probe-area value must never be passed as ``area_m2``; callers pass the
    semantic-floor polygon-union area after known blocker footprints.
    """
    area = facts.area_m2
    if facts.owner_override == "discard":
        return ScreeningResult("discard_candidate", ("owner_explicit_discard",), area)
    if facts.owner_override == "crop":
        return ScreeningResult("crop_candidate", ("owner_explicit_crop_review",), area)
    if facts.owner_override == "keep":
        return ScreeningResult("usable_candidate", ("owner_explicit_keep",), area)

    if not facts.area_assessable or not _finite_nonnegative(area):
        return ScreeningResult(
            "unassessable_hold", ("area_or_floor_unassessable",), area
        )

    upper = facts.unassigned_ground_upper_m2
    if not _finite_nonnegative(upper):
        return ScreeningResult(
            "unassessable_hold", ("invalid_unassigned_ground_upper_bound",), area
        )
    possible_upper = area + upper
    # Hold only when unresolved scope could change a project decision. The
    # upper bound is a sensitivity cue, never an area addition.
    if not facts.scope_verified and (
        (area < AREA_MIN_M2 <= possible_upper)
        or (area < LARGE_AREA_REVIEW_M2 <= possible_upper)
    ):
        return ScreeningResult(
            "scope_uncertain_hold",
            ("scope_unverified_may_change_area_decision",),
            area,
        )

    scene_unsuitable = (
        (facts.stair_dominated or facts.corridor_dominated)
        and facts.room_body_present is False
    )
    if scene_unsuitable:
        reasons = []
        if facts.stair_dominated:
            reasons.append("stair_dominated_without_room_body")
        if facts.corridor_dominated:
            reasons.append("corridor_dominated_without_room_body")
        return ScreeningResult("discard_candidate", tuple(reasons), area)

    if area < AREA_MIN_M2:
        return ScreeningResult("discard_candidate", ("area_below_10m2",), area)

    fragmented = (
        facts.component_count_ge_min_area is not None
        and facts.largest_component_fraction is not None
        and facts.component_count_ge_min_area >= 2
        and facts.largest_component_fraction < MIN_LARGEST_COMPONENT_FRACTION
    )
    narrow = (
        facts.elongation_long_short is not None
        and facts.largest_component_median_width_m is not None
        and facts.elongation_long_short >= MAX_ELONGATION
        and facts.largest_component_median_width_m < MAX_MEDIAN_WIDTH_M
    )
    if fragmented or narrow:
        reasons = []
        if fragmented:
            reasons.append("fragmented_without_dominant_component")
        if narrow:
            reasons.append("elongated_narrow_shape")
        return ScreeningResult("discard_candidate", tuple(reasons), area)

    if area >= LARGE_AREA_REVIEW_M2:
        return ScreeningResult(
            "crop_candidate", ("area_at_least_50m2_review_boundary",), area
        )

    return ScreeningResult("usable_candidate", ("passes_first_pass_geometry_screen",), area)

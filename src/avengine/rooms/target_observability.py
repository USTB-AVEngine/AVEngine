"""Small projection-only helpers for target camera candidate ordering.

The helpers consume existing conditioned-visibility screen fields and keep
insufficient proxy evidence as a lower-ranked candidate. They do not render,
accept questions, or reject scenes.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence


DEFAULT_MIN_PROJECTED_AREA_PX = 512
SCREEN_VISIBLE_STATES = frozenset(("visible_clear", "visible_occluded"))
CLAIM_BOUNDARY = (
    "projection-only target observability ordering; no native pixel pass, "
    "question acceptance, or scene rejection is produced"
)


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        return None
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in values):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def _resolution(value: Any) -> tuple[int, int] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        return None
    height, width = int(value[0]), int(value[1])
    return (height, width) if height > 0 and width > 0 else None


def _score_bbox(
    *,
    bbox: Any,
    resolution_hw: Any,
    border_marginal_samples: Any,
    state: Any,
    in_view: bool,
    min_projected_area_px: int,
    require_visible_state: bool,
) -> dict[str, Any]:
    parsed = _bbox(bbox)
    resolution = _resolution(resolution_hw)
    result = {
        "in_view": bool(in_view),
        "state": state,
        "projected_body_bbox_px": list(parsed) if parsed is not None else None,
        "projected_area_px": 0.0,
        "border_marginal_samples": (
            int(border_marginal_samples)
            if isinstance(border_marginal_samples, int)
            and not isinstance(border_marginal_samples, bool)
            else None
        ),
        "bbox_touches_frame_edge": False,
        "projected_observability_score": 0.0,
        "eligible": False,
        "reason": None,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if resolution is None:
        result["reason"] = "resolution_missing"
        return result
    if parsed is None:
        result["reason"] = "projected_body_bbox_missing_or_invalid"
        return result
    height, width = resolution
    x0, y0, x1, y1 = parsed
    area = (x1 - x0) * (y1 - y0)
    touches = (
        x0 <= -0.5
        or y0 <= -0.5
        or x1 >= width - 0.5
        or y1 >= height - 0.5
    )
    result["projected_area_px"] = area
    result["bbox_touches_frame_edge"] = touches
    if require_visible_state and state not in SCREEN_VISIBLE_STATES:
        result["reason"] = "state_not_visible"
        return result
    border = result["border_marginal_samples"]
    if border is None:
        result["reason"] = "border_evidence_missing"
        return result
    if border != 0:
        result["reason"] = "border_marginal"
        return result
    if touches:
        result["reason"] = "projected_bbox_touches_frame_edge"
        return result
    if not in_view:
        result["reason"] = "projection_out_of_view"
        return result
    if area < float(min_projected_area_px):
        result["reason"] = "projected_area_below_minimum"
        return result
    result["projected_observability_score"] = area / float(min_projected_area_px)
    result["eligible"] = True
    return result


def score_screen_visibility_frame(
    frame: Mapping[str, Any],
    resolution_hw: Sequence[int],
    *,
    min_projected_area_px: int = DEFAULT_MIN_PROJECTED_AREA_PX,
) -> dict[str, Any]:
    """Score one existing ScreenVisibility frame as a ranking hint."""

    if not isinstance(frame, Mapping):
        return _score_bbox(
            bbox=None,
            resolution_hw=resolution_hw,
            border_marginal_samples=None,
            state=None,
            in_view=False,
            min_projected_area_px=min_projected_area_px,
            require_visible_state=True,
        )
    if (
        isinstance(min_projected_area_px, bool)
        or not isinstance(min_projected_area_px, int)
        or min_projected_area_px < 1
    ):
        raise ValueError("min_projected_area_px must be a positive integer")
    state = frame.get("state")
    return _score_bbox(
        bbox=frame.get("projected_body_bbox_px"),
        resolution_hw=resolution_hw,
        border_marginal_samples=frame.get("border_marginal_samples"),
        state=state,
        in_view=state in SCREEN_VISIBLE_STATES,
        min_projected_area_px=int(min_projected_area_px),
        require_visible_state=True,
    )


def score_projected_target_point(
    *,
    camera: Any,
    root_m: Sequence[float],
    body: Any,
    policy: Any = None,
    min_projected_area_px: int = DEFAULT_MIN_PROJECTED_AREA_PX,
) -> dict[str, Any]:
    """Score one root point with conditioned_visibility's existing projection."""

    if (
        isinstance(min_projected_area_px, bool)
        or not isinstance(min_projected_area_px, int)
        or min_projected_area_px < 1
    ):
        raise ValueError("min_projected_area_px must be a positive integer")
    try:
        from avengine.rooms import conditioned_visibility as cv

        points = cv.body_sample_points(camera.position_m, root_m, body, policy)
        resolved_policy = cv.screen_policy(policy)
        projected = cv._project(camera, points, resolved_policy)
        depth = projected["depth_m"]
        columns = projected["column_px"]
        rows = projected["row_px"]
        in_frustum = projected["in_frustum"]
        near_border = projected["near_border"]
        valid = [
            index
            for index, value in enumerate(depth)
            if float(value) > resolved_policy.near_m
            and math.isfinite(float(columns[index]))
            and math.isfinite(float(rows[index]))
        ]
        if not valid:
            raise ValueError("no projected body samples")
        bbox = [
            min(float(columns[index]) for index in valid),
            min(float(rows[index]) for index in valid),
            max(float(columns[index]) for index in valid),
            max(float(rows[index]) for index in valid),
        ]
        border_count = sum(bool(near_border[index]) for index in valid)
        return _score_bbox(
            bbox=bbox,
            resolution_hw=camera.resolution_hw,
            border_marginal_samples=border_count,
            state=None,
            in_view=any(bool(in_frustum[index]) for index in valid),
            min_projected_area_px=int(min_projected_area_px),
            require_visible_state=False,
        )
    except (AttributeError, TypeError, ValueError, IndexError, OverflowError):
        return _score_bbox(
            bbox=None,
            resolution_hw=getattr(camera, "resolution_hw", None),
            border_marginal_samples=None,
            state=None,
            in_view=False,
            min_projected_area_px=int(min_projected_area_px),
            require_visible_state=False,
        ) | {"reason": "projected_body_bbox_unavailable"}


def rank_projected_point_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Sort existing point rows by score, then apply the caller's old cap."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("rows must be a sequence of mappings")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit must be a positive integer when supplied")
    indexed = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"rows[{index}] must be a mapping")
        indexed.append((index, row))
    indexed.sort(
        key=lambda item: (
            -float(item[1].get("projected_observability_score", 0.0) or 0.0),
            -float(item[1].get("projected_area_px", 0.0) or 0.0),
            item[0],
        )
    )
    if limit is not None:
        indexed = indexed[:limit]
    return [deepcopy(row) for _index, row in indexed]


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_MIN_PROJECTED_AREA_PX",
    "SCREEN_VISIBLE_STATES",
    "rank_projected_point_rows",
    "score_projected_target_point",
    "score_screen_visibility_frame",
]

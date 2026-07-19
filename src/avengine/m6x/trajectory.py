"""Materialize sparse M6.x anchor routes into exact visual-frame paths."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class M6XTrajectoryError(ValueError):
    """A sparse trajectory cannot be materialized deterministically."""


def _anchor_index(anchor_library: Mapping[str, Any]) -> dict[str, np.ndarray]:
    try:
        anchors = anchor_library["anchors"]
    except (KeyError, TypeError) as exc:
        raise M6XTrajectoryError("anchor library lacks anchors") from exc
    result: dict[str, np.ndarray] = {}
    for record in anchors:
        anchor_id = record.get("anchor_id")
        try:
            point = np.asarray(record.get("position_m"), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise M6XTrajectoryError(f"anchor {anchor_id!r} is invalid") from exc
        if (
            not isinstance(anchor_id, str)
            or not anchor_id
            or anchor_id in result
            or point.shape != (3,)
            or not np.all(np.isfinite(point))
        ):
            raise M6XTrajectoryError(f"anchor {anchor_id!r} is invalid")
        result[anchor_id] = point
    return result


def materialize_route(
    route: Mapping[str, Any],
    *,
    anchor_library: Mapping[str, Any],
    frame_count: int,
) -> np.ndarray:
    """Expand one hold or piecewise-linear route to ``[frame_count,3]``.

    Sparse anchor frame indices are inclusive.  Frames after the last anchor
    hold the final point, which makes a moving interval followed by a fixed
    tail explicit without adding a second runtime rule.
    """

    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
    ):
        raise M6XTrajectoryError("frame_count must be a positive integer")
    anchors = _anchor_index(anchor_library)
    try:
        anchor_ids = list(route["anchor_ids"])
        indices = [int(value) for value in route["anchor_frame_indices"]]
        interpolation = route["interpolation"]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise M6XTrajectoryError("route is malformed") from exc
    if (
        not anchor_ids
        or len(anchor_ids) != len(indices)
        or indices[0] != 0
        or indices != sorted(set(indices))
        or indices[-1] >= frame_count
        or any(anchor_id not in anchors for anchor_id in anchor_ids)
    ):
        raise M6XTrajectoryError("route anchors or frame indices are invalid")
    points = np.stack([anchors[anchor_id] for anchor_id in anchor_ids], axis=0)
    if interpolation == "hold":
        if len(points) != 1:
            raise M6XTrajectoryError("hold route must use exactly one anchor")
        return np.ascontiguousarray(np.repeat(points, frame_count, axis=0))
    if interpolation not in {"piecewise_linear", "navmesh_follow"}:
        raise M6XTrajectoryError(f"unsupported route interpolation: {interpolation!r}")
    if interpolation == "navmesh_follow":
        raise M6XTrajectoryError(
            "navmesh_follow requires a native PathFinder materializer; this fixed "
            "Apartment canary only accepts prequalified piecewise-linear routes"
        )
    if len(points) < 2:
        raise M6XTrajectoryError("piecewise_linear route needs at least two anchors")

    output = np.empty((frame_count, 3), dtype=np.float64)
    for segment in range(len(points) - 1):
        start = indices[segment]
        end = indices[segment + 1]
        span = end - start
        if np.array_equal(points[segment], points[segment + 1]):
            output[start : end + 1] = points[segment]
            continue
        for frame_index in range(start, end + 1):
            alpha = (frame_index - start) / span
            output[frame_index] = (
                points[segment] * (1.0 - alpha) + points[segment + 1] * alpha
            )
    output[indices[-1] :] = points[-1]
    return np.ascontiguousarray(output)


def materialize_template_route(
    template_set: Mapping[str, Any],
    *,
    template_id: str,
    route_id: str,
    anchor_library: Mapping[str, Any],
) -> np.ndarray:
    """Resolve one template/route pair and materialize its exact path."""

    templates = [
        item
        for item in template_set.get("templates", [])
        if item.get("template_id") == template_id
    ]
    if len(templates) != 1:
        raise M6XTrajectoryError(
            f"trajectory template {template_id!r} must resolve exactly once"
        )
    routes = [
        item
        for item in templates[0].get("routes", [])
        if item.get("route_id") == route_id
    ]
    if len(routes) != 1:
        raise M6XTrajectoryError(
            f"trajectory route {route_id!r} must resolve exactly once"
        )
    return materialize_route(
        routes[0],
        anchor_library=anchor_library,
        frame_count=int(template_set["frame_count"]),
    )


__all__ = [
    "M6XTrajectoryError",
    "materialize_route",
    "materialize_template_route",
]

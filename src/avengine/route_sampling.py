"""Waypoint-polyline routes: arc-length sampling shared by the whole chain.

One route is an ordered list of UE-cm waypoints.  Two waypoints describe the
straight chords the corridor bank has always used; three or more describe a
path that turns, which is what UE's navigation returns.  Everything downstream
- the visual timeline, the route bank, question design - measures a route the
same way through this module, so a route means the same thing everywhere:

    speed = arc_length / clip_seconds

That identity is why a natural walking speed no longer needs a straight line
long enough to fit the room diagonal: ask the planner for a longer polyline
instead.  Sampling is by planar (UE X/Y) arc length so the actor holds a
constant speed no matter how the polyline splits into segments; the vertical
component rides along with the segment parameter.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "planar_cumulative",
    "arc_length_cm",
    "sample_polyline",
    "resample_route",
    "max_turn_degrees",
    "segment_yaw_deg",
]


def planar_cumulative(points: Sequence[Sequence[float]]) -> list[float]:
    """Cumulative planar distance at each waypoint, starting at 0."""
    cumulative = [0.0]
    for first, second in zip(points[:-1], points[1:]):
        cumulative.append(
            cumulative[-1]
            + math.hypot(
                float(second[0]) - float(first[0]),
                float(second[1]) - float(first[1]),
            )
        )
    return cumulative


def arc_length_cm(points: Sequence[Sequence[float]]) -> float:
    """Total planar length of the route."""
    return planar_cumulative(points)[-1] if len(points) >= 2 else 0.0


def sample_polyline(
    points: Sequence[Sequence[float]],
    cumulative: Sequence[float],
    distance: float,
) -> tuple[list[float], int]:
    """Position at ``distance`` along the route, with the segment it sits on."""
    last_segment = max(len(points) - 2, 0)
    if distance <= 0.0:
        return [float(value) for value in points[0]], 0
    if distance >= cumulative[-1]:
        return [float(value) for value in points[-1]], last_segment
    for index in range(last_segment + 1):
        if distance <= cumulative[index + 1]:
            span = cumulative[index + 1] - cumulative[index]
            fraction = 0.0 if span <= 0.0 else (distance - cumulative[index]) / span
            first, second = points[index], points[index + 1]
            return (
                [
                    float(first[axis])
                    + (float(second[axis]) - float(first[axis])) * fraction
                    for axis in range(3)
                ],
                index,
            )
    return [float(value) for value in points[-1]], last_segment


def resample_route(
    points: Sequence[Sequence[float]], frame_count: int
) -> list[list[float]]:
    """Evenly spaced-by-arc-length positions, one per frame.

    Constant speed is the point: splitting the frames per segment instead
    would sprint along long segments and crawl along short ones.
    """
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if len(points) < 2:
        return [[float(value) for value in points[0]]] * frame_count
    cumulative = planar_cumulative(points)
    total = cumulative[-1]
    if frame_count == 1:
        return [[float(value) for value in points[0]]]
    return [
        sample_polyline(points, cumulative, total * index / (frame_count - 1))[0]
        for index in range(frame_count)
    ]


def segment_yaw_deg(
    first: Sequence[float], second: Sequence[float]
) -> float | None:
    """World yaw of one segment, or None when the two points coincide."""
    dx = float(second[0]) - float(first[0])
    dy = float(second[1]) - float(first[1])
    if math.hypot(dx, dy) <= 1.0e-9:
        return None
    return math.degrees(math.atan2(dy, dx))


def max_turn_degrees(points: Sequence[Sequence[float]]) -> float:
    """Sharpest heading change at any waypoint.

    A composition descriptor, not a legality gate: a hard turn reads as the
    actor pivoting in place because the walk cycle is authored straight ahead.
    """
    worst = 0.0
    for index in range(len(points) - 2):
        before = segment_yaw_deg(points[index], points[index + 1])
        after = segment_yaw_deg(points[index + 1], points[index + 2])
        if before is None or after is None:
            continue
        turn = abs((after - before + 180.0) % 360.0 - 180.0)
        worst = max(worst, turn)
    return worst

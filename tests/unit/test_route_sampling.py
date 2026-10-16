"""Arc-length route sampling shared by the timeline, the bank and question design."""

from __future__ import annotations

import math

import pytest

from avengine.route_sampling import (
    arc_length_cm,
    max_turn_degrees,
    planar_cumulative,
    resample_route,
    sample_polyline,
    segment_yaw_deg,
)

L_ROUTE = [[0.0, 0.0, 0.0], [300.0, 0.0, 0.0], [300.0, 200.0, 0.0]]
STRAIGHT = [[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]]


def test_cumulative_and_arc_length_ignore_height() -> None:
    climbing = [[0.0, 0.0, 0.0], [300.0, 0.0, 90.0]]
    assert planar_cumulative(L_ROUTE) == [0.0, 300.0, 500.0]
    assert arc_length_cm(climbing) == pytest.approx(300.0)
    assert arc_length_cm([[1.0, 2.0, 3.0]]) == 0.0


def test_resample_holds_constant_speed_through_the_turn() -> None:
    frames = resample_route(L_ROUTE, 75)
    cumulative = planar_cumulative(L_ROUTE)
    covered = []
    for position in frames:
        _, segment = sample_polyline(
            L_ROUTE, cumulative,
            min(max(0.0, _travelled_guess(position, L_ROUTE)), cumulative[-1]),
        )
        covered.append(_travelled_guess(position, L_ROUTE))
    steps = [covered[i + 1] - covered[i] for i in range(len(covered) - 1)]
    expected = 500.0 / 74
    assert max(steps) == pytest.approx(expected, abs=1e-9)
    assert min(steps) == pytest.approx(expected, abs=1e-9)


def _travelled_guess(position, route) -> float:
    """Arc length of a sampled point, found by walking the segments."""
    cumulative = planar_cumulative(route)
    for index in range(len(route) - 1):
        first, second = route[index], route[index + 1]
        span = math.dist(first[:2], second[:2])
        if span <= 0.0:
            continue
        along = ((position[0] - first[0]) * (second[0] - first[0])
                 + (position[1] - first[1]) * (second[1] - first[1])) / span
        if -1e-6 <= along <= span + 1e-6:
            off = math.dist(position[:2],
                            [first[0] + (second[0] - first[0]) * along / span,
                             first[1] + (second[1] - first[1]) * along / span])
            if off <= 1e-6:
                return cumulative[index] + along
    return cumulative[-1]


def test_resample_endpoints_are_exact() -> None:
    frames = resample_route(L_ROUTE, 75)
    assert frames[0] == L_ROUTE[0]
    assert frames[-1] == L_ROUTE[-1]
    assert len(frames) == 75


def test_resample_degenerate_inputs() -> None:
    assert resample_route([[1.0, 2.0, 3.0]], 4) == [[1.0, 2.0, 3.0]] * 4
    assert resample_route(L_ROUTE, 1) == [L_ROUTE[0]]
    with pytest.raises(ValueError):
        resample_route(L_ROUTE, 0)


def test_sample_polyline_clamps_and_reports_the_segment() -> None:
    cumulative = planar_cumulative(L_ROUTE)
    assert sample_polyline(L_ROUTE, cumulative, -1.0) == (L_ROUTE[0], 0)
    assert sample_polyline(L_ROUTE, cumulative, 10_000.0) == (L_ROUTE[-1], 1)
    position, segment = sample_polyline(L_ROUTE, cumulative, 400.0)
    assert segment == 1
    assert position == pytest.approx([300.0, 100.0, 0.0])


def test_segment_yaw_and_max_turn() -> None:
    assert segment_yaw_deg([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert segment_yaw_deg([0.0, 0.0], [0.0, 1.0]) == pytest.approx(90.0)
    assert segment_yaw_deg([2.0, 2.0], [2.0, 2.0]) is None
    assert max_turn_degrees(L_ROUTE) == pytest.approx(90.0)
    assert max_turn_degrees(STRAIGHT) == 0.0
    zigzag = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [100.0, 100.0, 0.0], [0.0, 100.0, 0.0]]
    assert max_turn_degrees(zigzag) == pytest.approx(90.0)


def test_speed_identity_is_arc_length_over_clip_seconds() -> None:
    """The whole point: a 6.9 m polyline walks at 1.38 m/s in a 5 s clip."""
    route = [[0.0, 0.0, 0.0], [400.0, 0.0, 0.0], [400.0, 290.0, 0.0]]
    assert arc_length_cm(route) / 100.0 / 5.0 == pytest.approx(1.38, abs=0.01)

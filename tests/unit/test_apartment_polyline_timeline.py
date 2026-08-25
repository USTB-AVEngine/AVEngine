"""Waypoint-polyline routes for the current-apartment visual timeline.

A route of two waypoints must stay bit-identical to the legacy straight-line
path; three or more waypoints are resampled by planar arc length so the actor
holds a constant speed and turns at the waypoints.
"""

from __future__ import annotations

import math

import pytest

from avengine.m5.current_apartment_visual import (
    CurrentApartmentVisualError,
    FRAME_COUNT,
    _finite_waypoints,
    _planar_cumulative,
    _sample_polyline,
    _timeline_state,
)

BINDING = {
    "actor_id": "source1_actor",
    "source_slot_id": "source1",
    "asset_id": "asset_v1",
    "revision": "rev_v1",
    "walk_phase_period_frames": 25,
    "ue_anatomical_forward_yaw_deg": 0.0,
}

STRAIGHT_START = [0.0, 0.0, 0.0]
STRAIGHT_END = [300.0, 0.0, 0.0]
L_ROUTE = [[0.0, 0.0, 0.0], [300.0, 0.0, 0.0], [300.0, 200.0, 0.0]]


def _states(waypoints, *, start=STRAIGHT_START, end=STRAIGHT_END, walk_start=0):
    return [
        _timeline_state(
            binding=BINDING,
            start=start,
            end=end,
            frame_index=index,
            walk_start_frame=walk_start,
            waypoints=waypoints,
        )
        for index in range(FRAME_COUNT)
    ]


def test_two_waypoints_match_the_legacy_straight_route_exactly() -> None:
    legacy = _states(None)
    routed = _states([STRAIGHT_START, STRAIGHT_END])
    assert routed == legacy
    assert all("route_geometry" not in state for state in routed)


def _travelled(state: dict, cumulative: list[float]) -> float:
    """Arc length covered by a sampled state, measured along the polyline."""
    segment = state["route_segment_index"]
    position = state["translation_ue_cm"]
    corner = L_ROUTE[segment]
    return cumulative[segment] + math.dist(position[:2], corner[:2])


def test_polyline_holds_constant_speed_along_the_route() -> None:
    states = _states(L_ROUTE)
    cumulative = _planar_cumulative(L_ROUTE)
    covered = [_travelled(state, cumulative) for state in states]
    steps = [covered[index + 1] - covered[index] for index in range(FRAME_COUNT - 1)]
    expected = 500.0 / (FRAME_COUNT - 1)
    assert max(steps) == pytest.approx(expected, abs=1.0e-9)
    assert min(steps) == pytest.approx(expected, abs=1.0e-9)


def test_only_the_corner_frame_shortens_the_straight_line_step() -> None:
    """Chord < arc exactly once: the frame that straddles the turn."""
    states = _states(L_ROUTE)
    chords = [
        math.dist(
            states[index]["translation_ue_cm"][:2],
            states[index + 1]["translation_ue_cm"][:2],
        )
        for index in range(FRAME_COUNT - 1)
    ]
    expected = 500.0 / (FRAME_COUNT - 1)
    short = [index for index, value in enumerate(chords)
             if value < expected - 1.0e-9]
    assert len(short) == 1
    assert chords[short[0]] < expected
    assert all(
        value == pytest.approx(expected, abs=1.0e-9)
        for index, value in enumerate(chords)
        if index != short[0]
    )


def test_polyline_endpoints_and_arc_length_are_exact() -> None:
    states = _states(L_ROUTE)
    assert states[0]["translation_ue_cm"] == L_ROUTE[0]
    assert states[FRAME_COUNT - 1]["translation_ue_cm"] == L_ROUTE[-1]
    assert states[0]["route_arc_length_ue_cm"] == pytest.approx(500.0)
    assert states[0]["route_waypoint_count"] == 3
    assert states[0]["route_geometry"] == "polyline"


def test_polyline_yaw_follows_the_segment_tangent() -> None:
    states = _states(L_ROUTE)
    yaws = [state["yaw_ue_deg"] for state in states]
    assert yaws[0] == pytest.approx(0.0)
    assert yaws[-1] == pytest.approx(90.0)
    changes = [
        index
        for index in range(1, FRAME_COUNT)
        if yaws[index] != yaws[index - 1]
    ]
    assert len(changes) == 1
    corner = changes[0]
    assert states[corner - 1]["route_segment_index"] == 0
    assert states[corner]["route_segment_index"] == 1


def test_static_polyline_stays_idle() -> None:
    point = [10.0, 20.0, 0.0]
    states = _states([point, point, point], start=point, end=point)
    assert {state["action_id"] for state in states} == {"idle"}
    assert all(state["translation_ue_cm"] == point for state in states)


def test_walk_start_frame_delays_the_polyline_departure() -> None:
    states = _states(L_ROUTE, walk_start=10)
    assert all(
        state["translation_ue_cm"] == L_ROUTE[0] for state in states[:11]
    )
    assert states[10]["action_id"] == "walk"
    assert states[9]["action_id"] == "idle"
    assert states[FRAME_COUNT - 1]["translation_ue_cm"] == L_ROUTE[-1]


def test_waypoint_validation_rejects_short_or_invalid_routes() -> None:
    assert _finite_waypoints(None, owner="route") is None
    with pytest.raises(CurrentApartmentVisualError):
        _finite_waypoints([[0.0, 0.0, 0.0]], owner="route")
    with pytest.raises(CurrentApartmentVisualError):
        _finite_waypoints("0,0,0", owner="route")
    with pytest.raises(CurrentApartmentVisualError):
        _finite_waypoints([[0.0, 0.0, 0.0], [1.0, float("nan"), 0.0]], owner="route")


def test_sample_polyline_clamps_outside_the_route() -> None:
    cumulative = _planar_cumulative(L_ROUTE)
    assert cumulative == [0.0, 300.0, 500.0]
    assert _sample_polyline(L_ROUTE, cumulative, -5.0)[0] == L_ROUTE[0]
    assert _sample_polyline(L_ROUTE, cumulative, 999.0)[0] == L_ROUTE[-1]
    midpoint, segment = _sample_polyline(L_ROUTE, cumulative, 400.0)
    assert segment == 1
    assert midpoint == pytest.approx([300.0, 100.0, 0.0])

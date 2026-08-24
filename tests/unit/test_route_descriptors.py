from __future__ import annotations

import numpy as np
import pytest

from avengine.m6x.route_descriptors import describe_route_pair, materialize_route


def test_l_route_uses_exact_equal_arc_samples_and_segment_tangents() -> None:
    route = materialize_route(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
        75,
        geometry="polyline",
    )
    assert route.positions_m.shape == (75, 3)
    steps = np.linalg.norm(np.diff(route.positions_m, axis=0), axis=1)
    assert steps == pytest.approx(np.full(74, 4.0 / 74.0), abs=1.0e-12)
    assert route.segment_boundaries[1] == {
        "waypoint_index": 1,
        "arc_distance_m": 2.0,
        "sample_position": 37.0,
        "lower_frame_index": 37,
        "upper_frame_index": 37,
        "exact_frame_index": 37,
    }
    assert route.tangent_yaw_degrees_by_frame[36] == 0.0
    assert route.tangent_yaw_degrees_by_frame[37] == 90.0


def test_endpoint_exchange_and_sub_60cm_are_retained_as_descriptors() -> None:
    forward = materialize_route(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        75,
        geometry="polyline",
    )
    reverse = materialize_route(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        75,
        geometry="polyline",
    )
    descriptor = describe_route_pair(
        forward,
        reverse,
        fps=15,
        listener_position_xy_m=[-1.0, -1.0],
        listener_yaw_degrees=0.0,
    )
    assert descriptor["endpoint_exchange"]["exact_endpoint_exchange"] is True
    assert descriptor["closest_continuous_plan"]["distance_m"] == pytest.approx(0.0)
    assert descriptor["proximity_reference"]["below_reference_frame_indices"]
    assert descriptor["proximity_reference"]["role"].endswith(
        "not_legality_gate"
    )
    assert descriptor["identity_continuity"] == {
        "source1_assignment_changes": False,
        "source2_assignment_changes": False,
        "endpoint_exchange_does_not_swap_source_or_actor_ids": True,
    }


def test_explicit_hold_has_75_positions_and_heading() -> None:
    hold = materialize_route(
        [[2.0, 3.0, 0.4]],
        75,
        geometry="hold",
        heading_yaw_degrees=-90.0,
    )
    assert hold.moving is False
    assert hold.positions_m.shape == (75, 3)
    assert np.all(hold.positions_m == [2.0, 3.0, 0.4])
    assert set(hold.tangent_yaw_degrees_by_frame) == {-90.0}


def test_continuous_closest_does_not_chord_a_fractional_corner() -> None:
    moving = materialize_route(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
        75,
        geometry="polyline",
    )
    # The corner is at normalized time 1/3, or frame position 24.666... .
    # The midpoint of frames 24/25 lies on the sampled diagonal chord but not
    # on either original L segment.
    chord_midpoint = 0.5 * (moving.positions_m[24] + moving.positions_m[25])
    hold = materialize_route(
        [chord_midpoint],
        75,
        geometry="hold",
        heading_yaw_degrees=0.0,
    )
    descriptor = describe_route_pair(moving, hold, fps=15)
    continuous = descriptor["closest_continuous_plan"]
    expected = min(float(chord_midpoint[1]), 1.0 - float(chord_midpoint[0]))
    assert expected > 1.0e-4
    assert continuous["distance_m"] == pytest.approx(expected, abs=1.0e-12)
    assert continuous["authority"] == (
        "original_piecewise_linear_normalized_arc_time"
    )
    assert continuous["normalized_time_interval"][1] == pytest.approx(1.0 / 3.0)

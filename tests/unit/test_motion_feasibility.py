from __future__ import annotations

import pytest

from avengine.qa.motion_feasibility import (
    MOTION_MODES,
    MotionFeasibilityError,
    camera_side,
    classify_motion_feasibility,
    describe_motion_geometry,
    evaluate_motion_pair,
    minimum_linear_path_separation,
)


def _opposite_side_paths():
    right = ((100.0, 100.0, 27.1), (400.0, 100.0, 27.1))
    left = ((100.0, -100.0, 27.1), (400.0, -100.0, 27.1))
    directed = (right, tuple(reversed(right)), left, tuple(reversed(left)))
    stands = tuple(sorted({*right, *left}))
    return directed, stands


def test_classifier_exhaustively_finds_all_four_motion_modes() -> None:
    directed, stands = _opposite_side_paths()

    result = classify_motion_feasibility(
        camera_position_xy=(0.0, 0.0),
        camera_yaw_degrees=0.0,
        directed_paths=directed,
        stand_points=stands,
        minimum_continuous_separation=60.0,
    )

    assert result.feasible_modes == MOTION_MODES
    assert result.all_modes_feasible is True
    assert result.feasible_pair_count == 4
    assert all(item.attempted_pair_count > 0 for item in result.modes)
    assert all(item.selected_pair is not None for item in result.modes)
    assert all(
        item.to_record()["candidate_pair_search_status"]
        == "stopped_after_first_feasible_pair"
        for item in result.modes
    )
    record = result.to_record()
    assert record["claim_boundary"].startswith("difficulty descriptors")
    assert record["candidate_filtering"] == "disabled"
    assert record["thresholds"]["minimum_continuous_separation"] == 60.0


def test_classifier_explains_same_side_failure_without_lowering_separation() -> None:
    first = ((100.0, 100.0), (400.0, 100.0))
    second = ((100.0, 200.0), (400.0, 200.0))
    directed = (first, tuple(reversed(first)), second, tuple(reversed(second)))
    stands = tuple(sorted({*first, *second}))

    result = classify_motion_feasibility(
        camera_position_xy=(0.0, 0.0),
        camera_yaw_degrees=0.0,
        directed_paths=directed,
        stand_points=stands,
    )

    assert result.feasible_mode_count == 0
    assert {item.failure_reason for item in result.modes} == {
        "no_pair_satisfies_side_margins"
    }
    assert all(item.side_gate_pass_pair_count == 0 for item in result.modes)
    assert all(item.separation_gate_pass_pair_count == 0 for item in result.modes)


def test_pair_evidence_accepts_explicit_future_anchor_fractions() -> None:
    right = ((100.0, 100.0), (400.0, 100.0))
    left = ((100.0, -100.0), (400.0, -100.0))

    evidence = evaluate_motion_pair(
        camera_position_xy=(0.0, 0.0),
        camera_yaw_degrees=0.0,
        source1_path=right,
        source2_path=left,
        first_anchor_fraction=0.2,
        second_anchor_fraction=0.8,
    )

    assert evidence.passed is True
    assert evidence.first_anchor_fraction == 0.2
    assert evidence.second_anchor_fraction == 0.8
    assert evidence.first_anchor_sides == ("right", "left")


def test_continuous_separation_catches_between_endpoint_crossing() -> None:
    first = ((-100.0, 0.0), (100.0, 0.0))
    second = ((0.0, -100.0), (0.0, 100.0))

    assert minimum_linear_path_separation(first, second) == pytest.approx(0.0)


def test_endpoint_exchange_is_retained_as_descriptor_with_source_identity() -> None:
    forward = ((100.0, 100.0, 27.1), (400.0, 100.0, 27.1))
    reverse = tuple(reversed(forward))
    classification = classify_motion_feasibility(
        camera_position_xy=(0.0, 0.0),
        camera_yaw_degrees=0.0,
        directed_paths=(forward, reverse),
        stand_points=forward,
        modes=("both_moving",),
    )

    descriptor = describe_motion_geometry(
        classification,
        camera_position_xyz=(0.0, 0.0, 147.1),
        camera_yaw_degrees=0.0,
    )["modes"]["both_moving"]["descriptor"]

    assert descriptor["frame_count"] == 75
    assert len(descriptor["frames"]) == 75
    assert descriptor["summary"]["endpoint_exchange"] is True
    assert descriptor["summary"]["continuous_crossing_xy"] is True
    assert descriptor["summary"]["difficulty_thresholds_do_not_filter_candidate"]
    assert descriptor["source_identity_binding"] == {
        "source1": "source1",
        "source2": "source2",
        "identity_preserved_through_crossing_or_swap": True,
    }
    assert descriptor["native_projection_visibility"]["status"].startswith("pending")


def test_inputs_fail_closed() -> None:
    assert camera_side((0.0, 0.0), 0.0, (1.0, 1.0)) == (
        "right",
        pytest.approx(2**-0.5),
    )
    with pytest.raises(MotionFeasibilityError, match="unknown motion mode"):
        classify_motion_feasibility(
            camera_position_xy=(0.0, 0.0),
            camera_yaw_degrees=0.0,
            directed_paths=(),
            stand_points=(),
            modes=("teleport",),
        )
    with pytest.raises(MotionFeasibilityError, match="must lie in"):
        evaluate_motion_pair(
            camera_position_xy=(0.0, 0.0),
            camera_yaw_degrees=0.0,
            source1_path=((0.0, 1.0), (1.0, 1.0)),
            source2_path=((0.0, -1.0), (1.0, -1.0)),
            first_anchor_fraction=1.5,
        )

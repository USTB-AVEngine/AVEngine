from __future__ import annotations

import json

import numpy as np
import pytest

from avengine.qa.motion_feasibility import (
    MOTION_MODES,
    MotionFeasibilityError,
    camera_side,
    classify_motion_feasibility,
    describe_motion_geometry,
    describe_source_pair_tracks,
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
    assert descriptor["track_provenance"] == (
        "synthesized_linear_between_declared_endpoints_not_runtime_track"
    )
    assert descriptor["runtime_track_claim"] is False
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


def test_explicit_frame_tracks_preserve_exchange_identity_and_raw_distances() -> None:
    source1 = np.asarray(((1.0, -1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)))
    source2 = np.asarray(((1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, -1.0, 0.0)))

    descriptor = describe_source_pair_tracks(
        source1,
        source2,
        source1_id="human_0",
        source2_id="dog_0",
        listener_positions_xyz=(0.0, 0.0, 0.0),
        listener_forward_xyz=(1.0, 0.0, 0.0),
        listener_right_xyz=(0.0, 1.0, 0.0),
        listener_up_xyz=(0.0, 0.0, 1.0),
        horizontal_axis_indices=(0, 1),
        vertical_axis_index=2,
        coordinate_space="apartment_ue_cm_xyz",
        track_authority="timeline_frame_actor_states",
        frame_indices=(10, 11, 12),
        pts_ticks=(0, 3200, 6400),
    )

    assert descriptor["track_provenance"] == (
        "caller_supplied_per_frame_positions_no_interpolation"
    )
    assert descriptor["runtime_track_claim"] is False
    assert descriptor["distance_series"]["pair_distance_xy"] == [2.0, 0.0, 2.0]
    assert descriptor["distance_series"]["pair_distance_horizontal"] == [
        2.0,
        0.0,
        2.0,
    ]
    assert descriptor["closest_frames"]["horizontal"] == {
        "frame_ordinal": 1,
        "frame_index": 11,
        "distance": 0.0,
        "tie_policy": "earliest_frame_ordinal",
        "tied_frame_ordinals": [1],
        "tied_frame_indices": [11],
    }
    swap = descriptor["observed_crossing_and_swap"]
    assert swap["observed_horizontal_coincidence_frame_indices"] == [11]
    assert swap["endpoint_exchange_horizontal"] is True
    assert swap["endpoint_exchange_xyz"] is True
    assert swap["endpoint_pairwise_listener_right_order_exchange"] is True
    assert swap["observed_pairwise_listener_right_order_swap_frame_pairs"] == [
        {
            "left_frame_ordinal": 0,
            "right_frame_ordinal": 2,
            "left_frame_index": 10,
            "right_frame_index": 12,
            "tied_frame_ordinals_between": [1],
        }
    ]
    assert swap["between_frame_crossing_status"] == "not_inferred"
    assert descriptor["source_identity_binding"] == {
        "source1_id": "human_0",
        "source2_id": "dog_0",
        "identity_preserved_through_crossing_or_swap": True,
    }
    assert descriptor["frames"][0]["source1"]["listener_local_azimuth_deg"] == (
        pytest.approx(-45.0)
    )
    assert descriptor["frames"][0]["source2"]["listener_local_azimuth_deg"] == (
        pytest.approx(45.0)
    )
    assert descriptor["frames"][0]["source1"]["visible_pixels"] is None
    assert descriptor["frames"][2]["pts_ticks"] == 6400
    assert descriptor["legality"]["status"] == "not_evaluated_by_track_descriptor"
    assert descriptor["difficulty"]["candidate_filtering"] == "disabled"
    assert descriptor["parameters"] == {
        "observed_coincidence_tolerance": 1.0e-6,
        "endpoint_exchange_tolerance": 1.0e-6,
        "side_center_tolerance": 1.0e-9,
        "basis_tolerance": 1.0e-6,
    }
    json.dumps(descriptor, sort_keys=True, allow_nan=False)


def test_explicit_tracks_support_habitat_y_up_with_listener_basis() -> None:
    descriptor = describe_source_pair_tracks(
        ((1.0, 1.0, 2.0), (2.0, 1.0, 3.0)),
        ((-1.0, 2.0, 2.0), (-2.0, 2.0, 3.0)),
        source1_id="mp3d_source_1",
        source2_id="mp3d_source_2",
        listener_positions_xyz=((0.0, 1.0, 0.0), (0.0, 1.0, 1.0)),
        listener_forward_xyz=(0.0, 0.0, 1.0),
        listener_right_xyz=(1.0, 0.0, 0.0),
        listener_up_xyz=(0.0, 1.0, 0.0),
        horizontal_axis_indices=(0, 2),
        vertical_axis_index=1,
        coordinate_space="habitat_xyz_y_up_m",
        track_authority="habitat_frame_records",
    )

    assert descriptor["axis_mapping"]["horizontal_axis_indices"] == [0, 2]
    assert descriptor["axis_mapping"]["vertical_axis_index"] == 1
    assert descriptor["axis_mapping"]["listener_basis_handedness"] == (
        "right_handed_forward_right_up"
    )
    assert descriptor["distance_series"]["pair_distance_horizontal"] == [
        2.0,
        4.0,
    ]
    assert descriptor["distance_series"]["pair_distance_xyz"] == pytest.approx(
        [5.0**0.5, 17.0**0.5]
    )
    assert descriptor["frames"][0]["source1"]["listener_local_azimuth_deg"] == (
        pytest.approx(26.5650511771)
    )
    assert descriptor["native_projection_visibility"] == {
        "status": "pending_same_renderer_native_execution",
        "projection_by_frame": None,
        "in_frame_by_frame": None,
        "occlusion_by_frame": None,
        "visible_pixels_by_frame": None,
    }


def test_seventy_five_frame_track_is_preserved_without_endpoint_interpolation() -> None:
    frame = np.arange(75, dtype=np.float64)
    source1 = np.stack((frame, np.square(frame % 7), np.zeros(75)), axis=1)
    source2 = np.stack((frame + 10.0, np.zeros(75), np.zeros(75)), axis=1)

    descriptor = describe_source_pair_tracks(
        source1,
        source2,
        source1_id="source1_actor",
        source2_id="source2_actor",
        listener_positions_xyz=(0.0, 0.0, 0.0),
        listener_forward_xyz=(1.0, 0.0, 0.0),
        listener_right_xyz=(0.0, 1.0, 0.0),
        listener_up_xyz=(0.0, 0.0, 1.0),
        horizontal_axis_indices=(0, 1),
        vertical_axis_index=2,
        coordinate_space="apartment_ue_cm_xyz",
        track_authority="unit_test_non_linear_frame_records",
        pts_ticks=tuple(index * 3200 for index in range(75)),
    )

    assert descriptor["frame_count"] == 75
    assert descriptor["frames"][37]["source1"]["position_xyz"] == [
        37.0,
        4.0,
        0.0,
    ]
    assert descriptor["frames"][37]["source1"]["position_xyz"] == (source1[37].tolist())
    assert descriptor["frames"][37]["source1"]["position_xyz"][1] != 8.0
    assert descriptor["frames"][74]["pts_ticks"] == 236800


def test_pairwise_listener_right_swap_handles_same_side_and_single_frame() -> None:
    common = {
        "source1_id": "source1",
        "source2_id": "source2",
        "listener_positions_xyz": (0.0, 0.0, 0.0),
        "listener_forward_xyz": (1.0, 0.0, 0.0),
        "listener_right_xyz": (0.0, 1.0, 0.0),
        "listener_up_xyz": (0.0, 0.0, 1.0),
        "horizontal_axis_indices": (0, 1),
        "vertical_axis_index": 2,
        "coordinate_space": "test_xyz",
        "track_authority": "unit_test_frames",
    }
    same_side_swap = describe_source_pair_tracks(
        ((1.0, 1.0, 0.0), (1.0, 2.0, 0.0), (1.0, 3.0, 0.0)),
        ((1.0, 3.0, 0.0), (1.0, 2.0, 0.0), (1.0, 1.0, 0.0)),
        **common,
    )
    swap = same_side_swap["observed_crossing_and_swap"]
    assert all(
        frame["source1"]["listener_side"] == "right"
        and frame["source2"]["listener_side"] == "right"
        for frame in same_side_swap["frames"]
    )
    assert swap["endpoint_pairwise_listener_right_order_exchange"] is True
    assert swap["observed_pairwise_listener_right_order_swap_frame_pairs"][0][
        "tied_frame_ordinals_between"
    ] == [1]

    single_frame = describe_source_pair_tracks(
        ((0.0, 0.0, 0.0),),
        ((0.0, 0.0, 0.0),),
        **common,
    )
    single_swap = single_frame["observed_crossing_and_swap"]
    assert single_swap["endpoint_exchange_horizontal"] is False
    assert single_swap["endpoint_exchange_xyz"] is False
    assert single_swap["endpoint_pairwise_listener_right_order_exchange"] is False
    assert single_frame["frames"][0]["source1"]["listener_angles_defined"] is False
    assert single_frame["frames"][0]["source1"]["listener_local_azimuth_deg"] is None
    assert single_frame["frames"][0]["source1"]["listener_local_elevation_deg"] is None


def test_explicit_track_descriptor_rejects_shape_axis_and_basis_ambiguity() -> None:
    common = {
        "source1_id": "source1",
        "source2_id": "source2",
        "listener_positions_xyz": (0.0, 0.0, 0.0),
        "listener_forward_xyz": (1.0, 0.0, 0.0),
        "listener_right_xyz": (0.0, 1.0, 0.0),
        "listener_up_xyz": (0.0, 0.0, 1.0),
        "horizontal_axis_indices": (0, 1),
        "vertical_axis_index": 2,
        "coordinate_space": "test_xyz",
        "track_authority": "unit_test_frames",
    }
    with pytest.raises(MotionFeasibilityError, match="equal.*shape"):
        describe_source_pair_tracks(
            ((0.0, 0.0, 0.0),),
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            **common,
        )
    with pytest.raises(MotionFeasibilityError, match="remaining axis"):
        describe_source_pair_tracks(
            ((0.0, 0.0, 0.0),),
            ((1.0, 0.0, 0.0),),
            **{**common, "vertical_axis_index": 1},
        )
    with pytest.raises(MotionFeasibilityError, match="orthonormal"):
        describe_source_pair_tracks(
            ((0.0, 0.0, 0.0),),
            ((1.0, 0.0, 0.0),),
            **{**common, "listener_right_xyz": (1.0, 0.0, 0.0)},
        )
    with pytest.raises(MotionFeasibilityError, match="real numeric values"):
        describe_source_pair_tracks(
            np.asarray(((0.0 + 1.0j, 0.0, 0.0),)),
            ((1.0, 0.0, 0.0),),
            **common,
        )
    with pytest.raises(MotionFeasibilityError, match="must not contain booleans"):
        describe_source_pair_tracks(
            ((True, 0.0, 0.0),),
            ((1.0, 0.0, 0.0),),
            **common,
        )
    with pytest.raises(MotionFeasibilityError, match="basis tolerance"):
        describe_source_pair_tracks(
            ((0.0, 0.0, 0.0),),
            ((1.0, 0.0, 0.0),),
            **{
                **common,
                "listener_forward_xyz": (0.0, 0.0, 0.0),
                "listener_right_xyz": (0.0, 0.0, 0.0),
                "listener_up_xyz": (0.0, 0.0, 0.0),
                "basis_tolerance": 2.0,
            },
        )
    with pytest.raises(MotionFeasibilityError, match="derived deltas"):
        describe_source_pair_tracks(
            ((1.0e308, 0.0, 0.0),),
            ((-1.0e308, 0.0, 0.0),),
            **common,
        )
    finite_extreme = describe_source_pair_tracks(
        ((1.0e308, 0.0, 0.0),),
        ((0.0, 0.0, 0.0),),
        **common,
    )
    assert finite_extreme["distance_series"]["pair_distance_xyz"] == [1.0e308]
    json.dumps(finite_extreme, allow_nan=False)


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

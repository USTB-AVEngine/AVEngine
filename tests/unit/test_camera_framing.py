from __future__ import annotations

from copy import deepcopy
import math

import pytest

from avengine.camera_framing import (
    CameraFramingError,
    evaluate_static_camera_candidate,
    project_world_aabb,
    solve_static_camera_candidates,
)
from avengine.sensor_rig_trajectory import (
    FRAME_COUNT,
    materialize_sensor_rig_trajectory,
    validate_sensor_rig_trajectory,
)


def _calibration(*, margin: float = 10.0) -> dict:
    return {
        "resolution_hw": [100, 200],
        "hfov_degrees": 90.0,
        "near_m": 0.1,
        "near_tolerance_m": 1.0e-6,
        "margins_px": {
            "left": margin,
            "right": margin,
            "top": margin,
            "bottom": margin,
        },
    }


def _raw_aabb(
    minimum: tuple[float, float, float] = (-0.5, 0.75, -5.0),
    maximum: tuple[float, float, float] = (0.5, 1.25, -4.0),
) -> dict:
    return {"minimum_m": list(minimum), "maximum_m": list(maximum)}


def _bound_aabb(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    *,
    covered_frame_indices: tuple[int, ...],
    action_id: str = "authored_action",
) -> dict:
    return {
        **_raw_aabb(minimum, maximum),
        "action_id": action_id,
        "bounds_authority": {
            "status": "pass",
            "authority_id": "bounds-authority-v1",
            "source": "live_skinned_component_bounds",
            "asset_id": "registered-asset-v1",
            "revision_id": "runtime-revision-v1",
            "action_scope": action_id,
        },
        "coordinate_chain": {
            "status": "pass",
            "authority_id": "coordinate-chain-v1",
            "from_frame": "component_local_m",
            "to_frame": "avengine_world_right_handed_y_up_m",
            "operations": ["component_to_world"],
        },
        "action_coverage": {
            "status": "pass",
            "authority_id": "action-coverage-v1",
            "action_id": action_id,
            "covered_frame_indices": list(covered_frame_indices),
        },
    }


def _frames(count: int = FRAME_COUNT) -> list[dict]:
    covered = tuple(range(count))
    return [
        {
            "frame_index": frame_index,
            "actor_aabbs": {
                "source2": _bound_aabb(
                    (1.0, 0.75, -6.0),
                    (1.5, 1.25, -5.0),
                    covered_frame_indices=covered,
                ),
                "source1": _bound_aabb(
                    (-1.5, 0.75, -5.0),
                    (-1.0, 1.25, -4.0),
                    covered_frame_indices=covered,
                ),
            },
        }
        for frame_index in range(count)
    ]


def _candidate(
    candidate_id: str,
    *,
    priority: float = 0.0,
    room_status: str = "pass",
    position_m: tuple[float, float, float] = (0.0, 1.0, 0.0),
    yaw_deg: float = 0.0,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "priority": priority,
        "position_m": list(position_m),
        "yaw_deg": yaw_deg,
        "room_gate": {
            "status": room_status,
            "authority_id": "room-gate-authority-v1",
            "hard_gates": {
                "navigable": {"status": room_status},
                "collision_clearance": {"status": room_status},
            },
        },
    }


def _solve(*, frames: list[dict] | None = None, candidates: list | None = None) -> dict:
    return solve_static_camera_candidates(
        frames=_frames() if frames is None else frames,
        candidates=[_candidate("camera")] if candidates is None else candidates,
        calibration=_calibration(),
        trajectory_id="generic_static_rig_v1",
        ordered_actor_ids=["source1", "source2"],
        minimum_order_gap_px=5.0,
    )


def test_projects_all_eight_corners_with_ordered_actor_gates() -> None:
    evidence = project_world_aabb(
        aabb_world_m=_raw_aabb(),
        camera_pose={"position_m": [0.0, 1.0, 0.0], "yaw_deg": 0.0},
        calibration=_calibration(),
    )

    assert evidence["corner_count"] == 8
    assert len(evidence["world_corners_m"]) == 8
    assert len(evidence["projected_corners_px"]) == 8
    assert evidence["gate_order"] == [
        "near_plane",
        "margins",
        "image_containment",
    ]
    assert [evidence["gates"][name]["status"] for name in evidence["gate_order"]] == [
        "pass",
        "pass",
        "pass",
    ]
    assert evidence["hard_gates_pass"]
    bbox = evidence["projected_bbox_px"]
    assert math.isclose(bbox["left"], 87.5, abs_tol=1.0e-12)
    assert math.isclose(bbox["right"], 112.5, abs_tol=1.0e-12)
    assert math.isclose(bbox["top"], 43.75, abs_tol=1.0e-12)
    assert math.isclose(bbox["bottom"], 56.25, abs_tol=1.0e-12)


def test_near_gate_records_tolerance_and_stops_later_projection_gates() -> None:
    evidence = project_world_aabb(
        aabb_world_m=_raw_aabb((-0.01, 0.99, -0.1000005), (0.01, 1.01, -0.1000004)),
        camera_pose={"position_m": [0.0, 1.0, 0.0], "yaw_deg": 0.0},
        calibration=_calibration(),
    )

    near = evidence["gates"]["near_plane"]
    assert near["status"] == "fail"
    assert near["configured_near_m"] == 0.1
    assert near["tolerance_m"] == 1.0e-6
    assert math.isclose(near["effective_threshold_m"], 0.100001)
    assert evidence["gates"]["margins"]["status"] == "not_evaluated"
    assert evidence["gates"]["image_containment"]["status"] == "not_evaluated"
    assert evidence["projected_bbox_px"] is None


def test_margin_gate_is_stricter_than_raw_image_containment() -> None:
    evidence = project_world_aabb(
        aabb_world_m=_raw_aabb((3.0, 0.75, -5.0), (3.5, 1.25, -4.0)),
        camera_pose={"position_m": [0.0, 1.0, 0.0], "yaw_deg": 0.0},
        calibration=_calibration(margin=20.0),
    )
    assert evidence["gates"]["near_plane"]["status"] == "pass"
    assert evidence["gates"]["margins"]["status"] == "fail"
    assert evidence["gates"]["image_containment"]["status"] == "pass"
    assert not evidence["hard_gates_pass"]


def test_partial_evaluator_enforces_full_bbox_left_right_order_and_gap() -> None:
    evaluation = evaluate_static_camera_candidate(
        frames=_frames(3),
        candidate=_candidate("camera"),
        calibration=_calibration(),
        ordered_actor_ids=["source1", "source2"],
        minimum_order_gap_px=5.0,
    )
    assert evaluation["selectable"]
    assert all(
        frame["full_bbox_order"]["status"] == "pass"
        for frame in evaluation["frame_evaluations"]
    )

    exchanged = evaluate_static_camera_candidate(
        frames=_frames(3),
        candidate=_candidate("camera"),
        calibration=_calibration(),
        ordered_actor_ids=["source2", "source1"],
        minimum_order_gap_px=5.0,
    )
    assert not exchanged["selectable"]
    assert all(
        frame["full_bbox_order"]["status"] == "fail"
        for frame in exchanged["frame_evaluations"]
    )


def test_one_frame_gap_failure_rejects_whole_candidate() -> None:
    frames = _frames(3)
    frames[1]["actor_aabbs"]["source2"].update(
        {"minimum_m": [-0.9, 0.75, -6.0], "maximum_m": [-0.4, 1.25, -5.0]}
    )
    evaluation = evaluate_static_camera_candidate(
        frames=frames,
        candidate=_candidate("camera"),
        calibration=_calibration(),
        ordered_actor_ids=["source1", "source2"],
        minimum_order_gap_px=5.0,
    )
    assert not evaluation["selectable"]
    statuses = [
        frame["full_bbox_order"]["status"] for frame in evaluation["frame_evaluations"]
    ]
    assert statuses == ["pass", "fail", "pass"]


def test_solver_is_input_order_independent_and_skips_failed_room_gate() -> None:
    candidates = [_candidate("zeta"), _candidate("alpha")]
    first = _solve(candidates=candidates)
    second = _solve(candidates=list(reversed(candidates)))
    assert first == second
    assert first["selected_candidate_id"] == "alpha"
    assert [item["candidate_id"] for item in first["candidate_evaluations"]] == [
        "alpha",
        "zeta",
    ]


def test_solver_requires_exact_canonical_sensor_rig_frame_closure() -> None:
    with pytest.raises(CameraFramingError, match="exactly match canonical"):
        _solve(frames=_frames(4))

    result = _solve()
    binding = result["sensor_rig_binding"]
    assert result["frame_indices"] == list(range(FRAME_COUNT))
    assert binding["source"] == "materialized_hold"
    assert validate_sensor_rig_trajectory(binding["trajectory"]) == []
    assert result["status"] == "pass_cpu_declared_bounds_framing"
    assert result["native_pixel_validation_status"] == "pending"
    assert result["qualification_claim"] is False
    assert result["formal_episode_count"] == 0
    assert result["claim_boundary"]["native_pixels_are_validated"] is False


def test_actor_bounds_authority_coordinate_and_action_coverage_fail_closed() -> None:
    mutations = (
        ("bounds_authority", None, "bounds_authority"),
        ("coordinate_chain", None, "coordinate_chain"),
        ("action_coverage", None, "action_coverage"),
    )
    for key, value, message in mutations:
        frames = _frames(1)
        if value is None:
            del frames[0]["actor_aabbs"]["source1"][key]
        with pytest.raises(CameraFramingError, match=message):
            evaluate_static_camera_candidate(
                frames=frames,
                candidate=_candidate("camera"),
                calibration=_calibration(),
                ordered_actor_ids=["source1", "source2"],
                minimum_order_gap_px=5.0,
            )

    frames = _frames(1)
    frames[0]["actor_aabbs"]["source1"]["bounds_authority"]["authority_id"] = ""
    with pytest.raises(CameraFramingError, match="authority_id must be non-empty"):
        evaluate_static_camera_candidate(
            frames=frames,
            candidate=_candidate("camera"),
            calibration=_calibration(),
            ordered_actor_ids=["source1", "source2"],
            minimum_order_gap_px=5.0,
        )

    frames = _frames(1)
    frames[0]["actor_aabbs"]["source1"]["coordinate_chain"]["to_frame"] = "other"
    with pytest.raises(CameraFramingError, match="canonical AVEngine world"):
        evaluate_static_camera_candidate(
            frames=frames,
            candidate=_candidate("camera"),
            calibration=_calibration(),
            ordered_actor_ids=["source1", "source2"],
            minimum_order_gap_px=5.0,
        )


def test_action_id_and_covered_frame_must_bind_actual_frame_state() -> None:
    frames = _frames(2)
    frames[1]["actor_aabbs"]["source1"]["action_id"] = "different_action"
    with pytest.raises(CameraFramingError, match="differs from action_coverage"):
        evaluate_static_camera_candidate(
            frames=frames,
            candidate=_candidate("camera"),
            calibration=_calibration(),
            ordered_actor_ids=["source1", "source2"],
            minimum_order_gap_px=5.0,
        )

    frames = _frames(2)
    frames[1]["actor_aabbs"]["source1"]["action_coverage"]["covered_frame_indices"] = [
        0
    ]
    with pytest.raises(CameraFramingError, match="not covered"):
        evaluate_static_camera_candidate(
            frames=frames,
            candidate=_candidate("camera"),
            calibration=_calibration(),
            ordered_actor_ids=["source1", "source2"],
            minimum_order_gap_px=5.0,
        )


def test_matching_provided_canonical_sensor_rig_is_accepted_without_mutation() -> None:
    trajectory = materialize_sensor_rig_trajectory(
        trajectory_id="generic_static_rig_v1",
        program={"kind": "HOLD", "position_m": [0.0, 1.0, 0.0], "yaw_deg": 0.0},
    )
    original = deepcopy(trajectory)
    result = solve_static_camera_candidates(
        frames=_frames(),
        candidates=[_candidate("camera")],
        calibration=_calibration(),
        trajectory_id="generic_static_rig_v1",
        ordered_actor_ids=["source1", "source2"],
        minimum_order_gap_px=5.0,
        sensor_rig_trajectory=trajectory,
    )
    assert result["sensor_rig_binding"]["source"] == "provided_validated"
    result["sensor_rig_binding"]["trajectory"]["trajectory_id"] = "changed"
    assert trajectory == original


def test_duck_typed_candidate_and_invalid_order_closure() -> None:
    class Candidate:
        candidate_id = "duck"
        priority = 0
        position_m = [0.0, 1.0, 0.0]
        yaw_deg = 0.0
        room_gate = {
            "status": "pass",
            "authority_id": "room-duck-v1",
            "hard_gates": {"navigable": {"passed": True}},
        }

    result = _solve(candidates=[Candidate()])
    assert result["selected_candidate_id"] == "duck"
    with pytest.raises(CameraFramingError, match="at least two unique"):
        solve_static_camera_candidates(
            frames=_frames(),
            candidates=[Candidate()],
            calibration=_calibration(),
            trajectory_id="generic_static_rig_v1",
            ordered_actor_ids=["source1"],
            minimum_order_gap_px=5.0,
        )


def test_room_gate_requires_structured_identity_and_every_declared_gate_passes() -> (
    None
):
    candidate = _candidate("missing-authority")
    candidate["room_gate"]["authority_id"] = ""
    with pytest.raises(CameraFramingError, match="authority_id must be non-empty"):
        _solve(candidates=[candidate])

    candidate = _candidate("failed-gate")
    candidate["room_gate"]["hard_gates"]["collision_clearance"] = {"status": "fail"}
    with pytest.raises(CameraFramingError, match="must explicitly pass"):
        _solve(candidates=[candidate])

    candidate = _candidate("missing-gates")
    candidate["room_gate"]["hard_gates"] = {}
    with pytest.raises(CameraFramingError, match="hard_gates must be non-empty"):
        _solve(candidates=[candidate])


def test_internal_candidate_shaped_record_cannot_bypass_public_validation() -> None:
    internal_shape = {
        "candidate_id": "forged",
        "priority": 0.0,
        "camera_pose": {"position_m": [0.0, 1.0, 0.0], "yaw_deg": 0.0},
        "room_gate_evidence": {"status": "pass"},
        "room_gate_pass": True,
    }
    with pytest.raises(CameraFramingError, match="missing room_gate"):
        evaluate_static_camera_candidate(
            frames=_frames(1),
            candidate=internal_shape,
            calibration=_calibration(),
            ordered_actor_ids=["source1", "source2"],
            minimum_order_gap_px=5.0,
        )

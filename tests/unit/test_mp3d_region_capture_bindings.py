"""CPU tests for post-hoc MP3D camera/route mini-pilot bindings."""

from pathlib import Path

import numpy as np

from avengine.m5.mp3d_region_capture_bindings import (
    CameraModel,
    project_case_source_centers,
    select_bijective_pairings,
)


def _camera() -> CameraModel:
    return CameraModel(
        placement_id="camera0",
        selection_order=1,
        m1_request_path=Path("/tmp/camera.json"),
        position_m=np.zeros(3, dtype=np.float64),
        world_from_camera_rotation=np.eye(3, dtype=np.float64),
        near_m=0.05,
        far_m=100.0,
        horizontal_tangent_limit=1.0,
        vertical_tangent_limit=0.75,
    )


def test_projection_retains_raw_centers_without_legality_filter() -> None:
    inside = np.tile(np.asarray([0.0, 0.0, -2.0]), (75, 1))
    outside = np.tile(np.asarray([3.0, 0.0, -2.0]), (75, 1))
    result = project_case_source_centers(
        _camera(), {"source1": inside, "source2": outside}
    )
    assert result["candidate_legality_effect"] == "none"
    assert result["whole_actor_visibility_authority"] is False
    assert result["source_summaries"]["source1"]["center_in_frustum_frame_count"] == 75
    assert result["source_summaries"]["source2"]["center_in_frustum_frame_count"] == 0
    assert result["at_least_one_center_frame_count"] == 75
    assert result["both_centers_frame_count"] == 0
    assert len(result["frames"]) == 75
    assert result["frames"][0]["sources"]["source2"]["horizontal_tangent"] == 1.5


def test_behind_camera_center_is_retained_with_null_tangent() -> None:
    behind = np.tile(np.asarray([0.0, 0.0, 1.0]), (75, 1))
    result = project_case_source_centers(
        _camera(), {"source1": behind, "source2": behind}
    )
    record = result["frames"][0]["sources"]["source1"]
    assert record["forward_depth_m"] == -1.0
    assert record["horizontal_tangent"] is None
    assert record["vertical_tangent"] is None
    assert record["center_in_frustum"] is False


def _pair(camera: str, family: str, score: tuple[int, int, int]):
    return {
        "placement_id": camera,
        "route_family_id": family,
        "m1_request_path": f"/{camera}.json",
        "descriptor_score": {
            "at_least_one_center_frame_count": score[0],
            "both_centers_frame_count": score[1],
            "source_center_in_frustum_count": score[2],
        },
    }


def test_assignment_is_bijective_and_uses_descriptors_only_for_scheduling() -> None:
    selected = select_bijective_pairings(
        [
            _pair("c0", "f0", (0, 0, 0)),
            _pair("c0", "f1", (10, 4, 14)),
            _pair("c1", "f0", (9, 3, 12)),
            _pair("c1", "f1", (0, 0, 0)),
        ],
        camera_ids=["c0", "c1"],
        route_family_ids=["f0", "f1"],
    )
    assert [(item["placement_id"], item["route_family_id"]) for item in selected] == [
        ("c0", "f1"),
        ("c1", "f0"),
    ]
    assert all("not candidate legality" in item["selection_role"] for item in selected)
    assert selected[0]["assignment_total_score"] == {
        "at_least_one_center_frame_count": 19,
        "both_centers_frame_count": 7,
        "source_center_in_frustum_count": 26,
    }

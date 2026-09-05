from __future__ import annotations

from pathlib import Path

from tools.rooms.plan_furnished_residential_episode import build_episode_plan
from tools.rooms.plan_furnished_camera_review import (
    _select_review_candidates,
    build_camera_review_plan,
)
from tests.unit.test_furniture_layout import _fixture
from avengine.rooms.furniture_layout import load_room_layout


def test_camera_review_expands_clock_and_holds_distinct_geometry_points(tmp_path: Path) -> None:
    layout = load_room_layout(_fixture(tmp_path / "room"))
    episode = build_episode_plan(layout, frame_count=75)
    review = build_camera_review_plan(
        episode,
        layout,
        candidate_count=4,
        hold_frames=6,
        grid_step_m=0.75,
    )
    assert review["planning_boundary"]["camera_review_only"] is True
    assert review["clock"]["frame_count"] == 24
    assert len(review["visual_plan"]["frames"]) == 24
    assert len(review["visual_plan"]["actors"]) == 2
    segments = review["visual_plan"]["camera_review"]["segments"]
    assert [(item["frame_start"], item["frame_end"]) for item in segments] == [
        (0, 5),
        (6, 11),
        (12, 17),
        (18, 23),
    ]
    assert len({item["geometry_point_id"] for item in review["visual_plan"]["camera_candidates"]}) == 4
    base = review["visual_plan"]["frames"][0]["actor_states"]
    for frame in review["visual_plan"]["frames"]:
        assert [item["actor_id"] for item in frame["actor_states"]] == [
            item["actor_id"] for item in base
        ]
        assert [item.get("translation_ue_cm") for item in frame["actor_states"]] == [
            item.get("translation_ue_cm") for item in base
        ]


def test_camera_review_uses_native_candidate_pool_metadata_and_fov_fields(tmp_path: Path) -> None:
    layout = load_room_layout(_fixture(tmp_path / "room"))
    episode = build_episode_plan(layout, frame_count=75)
    review = build_camera_review_plan(
        episode,
        layout,
        candidate_count=2,
        hold_frames=6,
        grid_step_m=0.5,
    )
    assert review["clock"]["frame_count"] == 12
    for candidate in review["visual_plan"]["camera_candidates"]:
        assert "forward_blender" in candidate
        assert "forward_ue" in candidate
        assert "review_score" in candidate
        assert "review_visibility_status" in candidate
        assert candidate["review_score_policy"].startswith("geometry_orientation_clearance")
    assert all(
        frame["camera_state"]["candidate_id"]
        == review["visual_plan"]["camera_review"]["segments"][frame["frame_index"] // 6]["candidate_id"]
        for frame in review["visual_plan"]["frames"]
    )
    assert review["visual_plan"]["camera_review"]["horizontal_fov_deg"] == 105.0
    assert review["visual_plan"]["camera_review"]["target_profile"] == "upper_body"
    assert all(
        candidate["horizontal_fov_deg"] == 105.0
        for candidate in review["visual_plan"]["camera_candidates"]
    )


def test_camera_review_keeps_a_near_target_geometry_point() -> None:
    scored = {
        "candidates": [
            {
                "candidate_id": "far",
                "geometry_point_id": "far",
                "position_authoring_m": [5.0, 5.0, 1.55],
                "review_score": 100.0,
            },
            {
                "candidate_id": "near",
                "geometry_point_id": "near",
                "position_authoring_m": [0.25, 0.25, 1.55],
                "review_score": -100.0,
            },
            {
                "candidate_id": "middle",
                "geometry_point_id": "middle",
                "position_authoring_m": [2.0, 2.0, 1.55],
                "review_score": 50.0,
            },
        ]
    }
    selected = _select_review_candidates(
        scored,
        candidate_count=2,
        target_position_m=[0.0, 0.0, 1.0],
        near_target_count=1,
    )
    assert {item["candidate_id"] for item in selected} == {"near", "far"}

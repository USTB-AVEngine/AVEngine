from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.rooms.furniture_layout import (
    FurnitureLayoutError,
    SeatCapacityError,
    build_seat_placements,
    clock_config,
    generate_camera_candidates,
    load_room_layout,
    score_camera_candidates,
)
from tools.rooms.plan_furnished_residential_episode import build_episode_plan


def _fixture(
    root: Path,
    *,
    room_id: str = "fixture_room",
    bounds: tuple[float, float, float, float] = (-4.0, -3.0, 4.0, 3.0),
    seat_count: int = 4,
    bad_objects: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    object_path = root / "object_semantics.json"
    anchors_path = root / "functional_anchors.json"
    objects = [
        {
            "object_id": "table",
            "category": "table",
            "bounds_xy_m": [-0.8, -0.6, 0.8, 0.6],
            "height_m": 0.8,
            "navigation_role": "ground_blocker",
            "static": True,
        }
    ]
    if bad_objects:
        objects = [{"object_id": "bad_table", "category": "table"}]
    seats = [
        {
            "anchor_id": f"seat_{index}",
            "position_m": [
                -2.2 + index * 1.45,
                -1.4,
                0.0,
            ],
            "facing_yaw_deg": 0.0,
            "support_height_m": 0.46,
        }
        for index in range(seat_count)
    ]
    object_path.write_text(json.dumps({"objects": objects}), encoding="utf-8")
    anchors_path.write_text(json.dumps({"seat_points": seats}), encoding="utf-8")
    manifest = {
        "kind": "fixture_room_handoff",
        "room_id": room_id,
        "status": "research_candidate",
        "envelope": {"bounds_xy_m": list(bounds)},
        "artifacts": {
            "object_semantics": object_path.name,
            "functional_anchors": anchors_path.name,
        },
        # Deliberately unusable: the planner must not consume review cameras.
        "review_cameras": [{"position_xyz_m": [float("nan"), 0, 0]}],
    }
    manifest_path = root / "room_manifest.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=True), encoding="utf-8")
    return manifest_path


def test_room_swap_uses_common_entry_and_changes_geometry_candidates(tmp_path: Path) -> None:
    room_a = load_room_layout(_fixture(tmp_path / "a"))
    room_b = load_room_layout(
        _fixture(
            tmp_path / "b",
            room_id="other_room",
            bounds=(8.0, 6.0, 16.0, 12.0),
        )
    )
    candidates_a = generate_camera_candidates(room_a)["candidates"]
    candidates_b = generate_camera_candidates(room_b)["candidates"]
    assert room_a["room_id"] != room_b["room_id"]
    assert candidates_a[0]["position_authoring_m"] != candidates_b[0]["position_authoring_m"]
    assert all(item["target_independent"] for item in candidates_a + candidates_b)
    assert room_a["review_cameras_used"] is False


def test_seat_shortage_is_explicit(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path / "room", seat_count=2)
    layout = load_room_layout(manifest)
    with pytest.raises(SeatCapacityError, match="requested 4") as exc_info:
        build_seat_placements(layout, seat_count=4)
    assert exc_info.value.available == 2


def test_bad_object_metadata_fails_before_camera_generation(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path / "bad", bad_objects=True)
    with pytest.raises(FurnitureLayoutError, match="bounds_xyz_m|bounds_xy_m"):
        load_room_layout(manifest)


def test_camera_pool_is_target_independent_and_scored_after_join(tmp_path: Path) -> None:
    layout = load_room_layout(_fixture(tmp_path / "room"))
    raw = generate_camera_candidates(layout)
    scored_a = score_camera_candidates(raw, target_position_m=[-3.0, -1.0, 1.0])
    scored_b = score_camera_candidates(
        raw,
        actor_positions_m=[[2.5, 1.5, 1.0]],
        question_context={"target_position_m": [2.5, 1.5, 1.0]},
    )
    raw_signature = [
        (item["candidate_id"], item["position_authoring_m"], item["yaw_deg"], item["pitch_deg"])
        for item in raw["candidates"]
    ]
    assert raw["generation"]["target_independent"] is True
    assert scored_a["generation"]["target_los_evaluated"] is False
    assert scored_b["generation"]["target_los_evaluated"] is False
    assert [
        (item["candidate_id"], item["position_authoring_m"], item["yaw_deg"], item["pitch_deg"])
        for item in raw["candidates"]
    ] == raw_signature
    assert [item["candidate_id"] for item in scored_a["candidates"]] != [
        item["candidate_id"] for item in scored_b["candidates"]
    ]


def test_pose_binding_offsets_from_seat_reference_and_150_clock(tmp_path: Path) -> None:
    layout = load_room_layout(_fixture(tmp_path / "room"))
    placement = build_seat_placements(
        layout,
        seat_count=4,
        actor_count=1,
        pose_bindings={
            "bindings": [
                {
                    "actor_id": "human0",
                    "seat_affordance_id": "seat_0",
                    "root_from_seat_m": [0.0, 0.0, 0.92],
                    "asset_id": "pose_asset",
                }
            ]
        },
    )
    actor = placement["actor_placements"][0]
    assert actor["placement_status"] == "bound"
    assert actor["seat_reference"]["reference_is_not_actor_root"] is True
    assert actor["root_position_authoring_m"][2] > actor["seat_reference"]["position_authoring_m"][2]
    assert actor["root_position_authoring_m"] != actor["seat_reference"]["position_authoring_m"]

    # The pose-agent import request is an asset pool.  Its reference actor yaw
    # must not become a new-room orientation; the room seat facing owns that.
    request_style = build_seat_placements(
        layout,
        seat_count=4,
        actor_count=1,
        pose_bindings={
            "assets": [
                {
                    "asset_id": "pose_pool_asset",
                    "animation_name": "Seated_Idle",
                    "seat_reference": {
                        "seat_anchor_id": "seat_0",
                        "reference_chair_yaw_degrees": 180.0,
                        "reference_actor_yaw_degrees": 170.0,
                        "root_offset_from_seat_anchor_blender_m": [0.0, 0.18, -0.01],
                    },
                }
            ]
        },
    )
    assert request_style["actor_placements"][0]["ue_animation"] == "Seated_Idle"
    assert request_style["actor_placements"][0]["pose_orientation_policy"].endswith(
        "reference_actor_yaw_ignored"
    )

    assert clock_config(frame_count=150)["ticks_per_frame"] == 3200
    plan = build_episode_plan(layout, frame_count=150)
    assert plan["clock"]["frame_count"] == 150
    assert len(plan["visual_plan"]["frames"]) == 150
    assert plan["visual_plan"]["frames"][149]["frame_index"] == 149

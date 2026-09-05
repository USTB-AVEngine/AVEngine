from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.rooms.furniture_layout import (
    FurnitureLayoutError,
    SeatCapacityError,
    authoring_to_habitat,
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


def test_camera_forward_conversion_matches_spear_blender_to_ue_convention(
    tmp_path: Path,
) -> None:
    layout = load_room_layout(_fixture(tmp_path / "room"))
    candidate_set = generate_camera_candidates(layout)
    candidate = next(
        item
        for item in candidate_set["candidates"]
        if item["yaw_deg"] == 300.0 and item["pitch_deg"] == 0.0
    )
    assert authoring_to_habitat([2.35, 0.65, 1.55]) == pytest.approx(
        [2.35, 1.55, -0.65]
    )
    assert candidate["forward_blender"] == pytest.approx(
        [0.5, -0.8660254037844386, 0.0]
    )
    assert candidate["forward_ue"] == pytest.approx(
        [0.5, 0.8660254037844386, 0.0]
    )
    assert candidate["ue_yaw_deg"] == pytest.approx(60.0)
    assert candidate["ue_pitch_deg"] == pytest.approx(0.0)


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
    zero_distance = score_camera_candidates(
        raw, target_position_m=raw["candidates"][0]["position_authoring_m"]
    )
    assert any(
        item["target_forward_ue"] == [0.0, 0.0, 0.0]
        for item in zero_distance["candidates"]
    )
    assert [
        (item["candidate_id"], item["position_authoring_m"], item["yaw_deg"], item["pitch_deg"])
        for item in raw["candidates"]
    ] == raw_signature
    assert [item["candidate_id"] for item in scored_a["candidates"]] != [
        item["candidate_id"] for item in scored_b["candidates"]
    ]


def test_seat_facing_corrects_an_opposite_declared_table_direction(
    tmp_path: Path,
) -> None:
    root = tmp_path / "orientation"
    root.mkdir()
    objects = {
        "objects": [
            {
                "object_id": "table",
                "category": "table",
                "center_xy_m": [0.0, 0.0],
                "size_xyz_m": [2.0, 1.0, 0.8],
                "seat_points": [
                    {
                        "anchor_id": "main_dining_seat_3",
                        "position_m": [1.0, -1.0, 0.0],
                        "facing_yaw_deg": -90.0,
                        "support_height_m": 0.46,
                    }
                ],
            }
        ]
    }
    (root / "objects.json").write_text(json.dumps(objects), encoding="utf-8")
    manifest = {
        "room_id": "orientation_fixture",
        "envelope": {"bounds_xy_m": [-2.0, -2.0, 2.0, 2.0]},
        "artifacts": {"objects": "objects.json"},
    }
    path = root / "room.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    layout = load_room_layout(path)
    seat = layout["seats"][0]
    assert seat["facing_yaw_deg"] == pytest.approx(-90.0)
    assert seat["facing_source"] == "declared_metadata"
    assert seat["facing_candidate_yaw_deg"] == pytest.approx(135.0)
    assert seat["facing_candidate_source"] == "furniture_center_geometry_candidate"


def test_overview_only_contains_camera_and_no_actor_states(tmp_path: Path) -> None:
    layout = load_room_layout(_fixture(tmp_path / "room"))
    plan = build_episode_plan(layout, frame_count=75, overview_only=True)
    assert plan["planning_boundary"]["overview_only"] is True
    assert plan["visual_plan"]["actors"] == []
    assert all(frame["actor_states"] == [] for frame in plan["visual_plan"]["frames"])
    assert plan["visual_plan"]["camera_selection"]["selection_mode"] == "overview_geometry_only"


def test_camera_scoring_reports_multi_target_framing_and_geometry_clearance(
    tmp_path: Path,
) -> None:
    layout = load_room_layout(_fixture(tmp_path / "room"))
    candidate_set = generate_camera_candidates(layout)
    targets = [[-2.0, -1.0, 0.5], [2.0, -1.0, 0.5]]
    bounds = [
        {"minimum_m": [x - 0.3, y - 0.3, 0.3], "maximum_m": [x + 0.3, y + 0.3, 1.7]}
        for x, y, _ in targets
    ]
    scored = score_camera_candidates(
        candidate_set,
        actor_positions_m=targets,
        target_bounds_m=bounds,
        obstacle_bounds_m=[item["bounds_xyz_m"] for item in layout["objects"]],
        room_bounds_xy_m=layout["geometry"]["bounds_xy_m"],
    )
    selected = scored["candidates"][0]
    assert scored["generation"]["target_geometry_framing_evaluated"] is True
    assert selected["target_count"] == 2
    assert "fully_framed_target_count" in selected
    assert "geometry_clearance_m" in selected
    assert scored["generation"]["target_los_evaluated"] is False


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
                    "animation": "/Game/Pose/Seated_Idle.Seated_Idle",
                    "blueprint": "/Game/Pose/BP_pose.BP_pose",
                    "skeletal_mesh": "/Game/Pose/pose.pose",
                    "emitter_offset_avengine_m": [0.1, 1.2, -0.2],
                    "animation_name": "Seated_Idle",
                    "seat_reference": {
                        "seat_anchor_id": "seat_0",
                        "reference_chair_yaw_degrees": 180.0,
                        "reference_actor_yaw_degrees": 170.0,
                        "seat_top_m": 0.53,
                        "root_offset_from_seat_anchor_blender_m": [0.0, 0.18, -0.01],
                    },
                }
            ]
        },
    )
    request_actor = request_style["actor_placements"][0]
    assert request_actor["ue_animation"] == "/Game/Pose/Seated_Idle.Seated_Idle"
    assert request_actor["blueprint_class_path"] == "/Game/Pose/BP_pose.BP_pose_C"
    assert request_actor["skeletal_mesh_path"] == "/Game/Pose/pose.pose"
    assert request_actor["emitter_local_ue_cm"] == pytest.approx([10.0, -20.0, 120.0])
    assert request_actor["root_position_authoring_m"][0] == pytest.approx(-2.2)
    assert request_actor["root_position_authoring_m"][1] == pytest.approx(-1.58)
    assert request_actor["root_position_authoring_m"][2] == pytest.approx(-0.08)
    assert request_actor["pose_orientation_policy"].endswith(
        "reference_actor_yaw_ignored"
    )

    candidate_set = generate_camera_candidates(layout)
    assert candidate_set["generation"]["yaw_candidates_deg"] == pytest.approx(
        list(range(0, 360, 30))
    )

    assert clock_config(frame_count=150)["ticks_per_frame"] == 3200
    plan = build_episode_plan(layout, frame_count=150)
    assert plan["clock"]["frame_count"] == 150
    assert len(plan["visual_plan"]["frames"]) == 150
    assert plan["visual_plan"]["frames"][149]["frame_index"] == 149

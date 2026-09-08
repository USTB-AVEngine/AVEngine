from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from avengine.rooms.furniture_layout import (
    FurnitureLayoutError,
    SeatCapacityError,
    authoring_to_habitat,
    build_seat_placements,
    camera_obstacle_bounds,
    clock_config,
    generate_camera_candidates,
    load_room_layout,
    score_camera_candidates,
)
from avengine.camera_pose import yaw_rotation_xyzw
from avengine.rooms.furnished_episode import (
    _actor_state,
    _overview_target_bounds,
    build_episode_plan,
    plan_furnished_residential_episode,
    reuse_camera_from_plan,
)


def _fixture(
    root: Path,
    *,
    room_id: str = "fixture_room",
    bounds: tuple[float, float, float, float] = (-4.0, -3.0, 4.0, 3.0),
    seat_count: int = 4,
    seat_facings: tuple[float, ...] | None = None,
    bad_objects: bool = False,
    furniture_assemblies: list[dict] | None = None,
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
            "facing_yaw_deg": (
                seat_facings[index]
                if seat_facings is not None and index < len(seat_facings)
                else 0.0
            ),
            "support_height_m": 0.46,
        }
        for index in range(seat_count)
    ]
    object_payload = {"objects": objects}
    if furniture_assemblies is not None:
        object_payload["furniture_assemblies"] = furniture_assemblies
    object_path.write_text(json.dumps(object_payload), encoding="utf-8")
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


def test_pose_root_closure_follows_actor_anatomical_forward(tmp_path: Path) -> None:
    manifest = _fixture(
        tmp_path / "closure",
        seat_facings=(0.0, 180.0, 90.0, -90.0),
    )
    layout = load_room_layout(manifest)
    placement = build_seat_placements(
        layout,
        seat_count=4,
        actor_count=4,
        pose_bindings={
            "assets": [
                {
                    "asset_id": f"pose_{index}",
                    "blueprint": f"/Game/Pose/BP_{index}.BP_{index}",
                    "skeletal_mesh": f"/Game/Pose/pose_{index}.pose_{index}",
                    "animation": "/Game/Pose/Seated_Idle.Seated_Idle",
                    "ue_anatomical_forward_yaw_deg": 90.0,
                    "seat_reference": {
                        "seat_anchor_id": f"seat_{index}",
                        "reference_chair_yaw_degrees": 0.0,
                        "seat_top_m": 0.46,
                        "root_offset_from_seat_anchor_blender_m": [0.0, -0.18, -0.01],
                    },
                }
                for index in range(4)
            ]
        },
    )
    for actor in placement["actor_placements"]:
        seat = actor["seat_reference"]["position_authoring_m"]
        root = actor["root_position_authoring_m"]
        theta = actor["seat_reference"]["facing_yaw_deg"]
        import math
        yaw = math.radians(theta + 90.0)
        anchor = [0.0, 0.18]
        world_anchor = [
            anchor[0] * math.cos(yaw) - anchor[1] * math.sin(yaw),
            anchor[0] * math.sin(yaw) + anchor[1] * math.cos(yaw),
        ]
        assert [root[0] + world_anchor[0], root[1] + world_anchor[1]] == pytest.approx(
            seat[:2]
        )
        assert actor["pose_seat_anchor_closure_error_m"] == pytest.approx(
            [0.0, 0.0, 0.0]
        )
        state = _actor_state(actor, frame_index=0, pts_ticks=0)
        assert state["actor_yaw_blender_deg"] == pytest.approx(theta + 90.0)
        assert state["actor_yaw_ue_deg"] == pytest.approx(
            ((-theta - 90.0 + 180.0) % 360.0) - 180.0
        )
        assert state["root_transform"]["rotation_xyzw"] == pytest.approx(
            yaw_rotation_xyzw(theta + 90.0)
        )
        assert state["rotation_xyzw"] == pytest.approx(
            yaw_rotation_xyzw(theta + 90.0)
        )


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


def test_camera_obstacle_bounds_excludes_floor_but_keeps_low_furniture(
    tmp_path: Path,
) -> None:
    layout = load_room_layout(_fixture(tmp_path / "room"))
    layout["objects"].extend(
        [
            {
                "object_id": "floor",
                "semantic_class": "floor",
                "navigation_role": "ground_blocker",
                "bounds_xyz_m": [[-4.0, -3.0, 0.0], [4.0, 3.0, 0.1]],
            },
            {
                "object_id": "ceiling",
                "semantic_class": "ceiling",
                "navigation_role": "ground_blocker",
                "bounds_xyz_m": [[-4.0, -3.0, 2.9], [4.0, 3.0, 3.0]],
            },
        ]
    )
    obstacles = camera_obstacle_bounds(layout)
    assert len(obstacles) == 1
    assert obstacles[0][0] == pytest.approx([-0.8, -0.6, 0.0])
    assert obstacles[0][1] == pytest.approx([0.8, 0.6, 0.8])


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
    assert request_actor["root_position_authoring_m"][0] == pytest.approx(-2.02)
    assert request_actor["root_position_authoring_m"][1] == pytest.approx(-1.4)
    assert request_actor["root_position_authoring_m"][2] == pytest.approx(-0.08)
    assert request_actor["pose_orientation_policy"].endswith(
        "reference_actor_yaw_ignored"
    )
    assert request_actor["pose_actor_yaw_blender_deg"] == pytest.approx(90.0)
    assert request_actor["pose_seat_anchor_closure_error_m"] == pytest.approx(
        [0.0, 0.0, 0.0]
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

    assert plan["scene"]["map_path_status"] == "not_declared"
    assert "no UE stage was launched" in plan["scene"]["claim_boundary"]



def test_v3_furniture_assemblies_drive_overview_targets(
    tmp_path: Path,
) -> None:
    assemblies = [
        {
            "object_id": "living_sofa_group",
            "kind": "living_group",
            "center_xy_m": [-1.5, 0.25],
        },
        {
            "object_id": "dining_table_group",
            "kind": "dining_group",
            "center_xy_m": [1.25, -0.5],
        },
    ]
    layout = load_room_layout(
        _fixture(
            tmp_path / "room",
            furniture_assemblies=assemblies,
        )
    )

    assert layout["furniture_assemblies"] == assemblies
    targets = _overview_target_bounds(layout)
    assert len(targets) == 2
    assert targets[0]["minimum_m"] == pytest.approx(
        [-1.8, -0.05, 0.35]
    )
    assert targets[0]["maximum_m"] == pytest.approx(
        [-1.2, 0.55, 1.70]
    )
    assert targets[1]["minimum_m"] == pytest.approx(
        [0.95, -0.8, 0.35]
    )
    assert targets[1]["maximum_m"] == pytest.approx(
        [1.55, -0.2, 1.70]
    )
    plan = build_episode_plan(
        layout, frame_count=75, overview_only=True
    )
    assert plan["visual_plan"]["camera_selection"]["target_count"] == 2


def test_furniture_assembly_center_must_be_finite_xy(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FurnitureLayoutError,
        match="furniture_assemblies.*center_xy_m",
    ):
        load_room_layout(
            _fixture(
                tmp_path / "bad-assembly",
                furniture_assemblies=[{
                    "object_id": "living_group",
                    "center_xy_m": [0.0, float("nan")],
                }],
            )
        )



def test_reuse_camera_from_plan_preserves_new_actor_states(tmp_path: Path) -> None:
    layout = load_room_layout(_fixture(tmp_path / "room"))
    plan = build_episode_plan(layout, frame_count=3)
    actor_states = copy.deepcopy(
        [frame["actor_states"] for frame in plan["visual_plan"]["frames"]]
    )
    source = tmp_path / "source_plan.json"
    source.write_text(
        json.dumps(
            {
                "visual_plan": {
                    "camera": {
                        "candidate_id": "native_selected_camera",
                        "position_authoring_m": [1.0, 2.0, 1.55],
                        "position_habitat_m": [1.0, 1.55, -2.0],
                        "position_ue_cm": [100.0, -200.0, 155.0],
                    }
                }
            }
        )
    )
    reused = reuse_camera_from_plan(plan, source.resolve())
    assert reused["visual_plan"]["camera"]["candidate_id"] == "native_selected_camera"
    assert reused["camera_reuse"]["camera_pose_only"] is True
    assert [frame["actor_states"] for frame in reused["visual_plan"]["frames"]] == actor_states
    assert all(
        frame["camera_state"]["frame_index"] == index
        for index, frame in enumerate(reused["visual_plan"]["frames"])
    )


def _camera_source(path: Path, camera: dict, frames=None) -> Path:
    visual = {"camera": camera}
    if frames is not None:
        visual["frames"] = frames
    path.write_text(json.dumps({"visual_plan": visual}))
    return path


def test_reuse_camera_rejects_dynamic_source_frames(tmp_path: Path) -> None:
    plan = build_episode_plan(load_room_layout(_fixture(tmp_path / "room")), frame_count=3)
    camera = {
        "position_authoring_m": [1.0, 2.0, 1.55],
        "position_habitat_m": [1.0, 1.55, -2.0],
        "position_ue_cm": [100.0, -200.0, 155.0],
    }
    moved = {**camera, "position_ue_cm": [101.0, -200.0, 155.0], "frame_index": 1}
    source = _camera_source(
        tmp_path / "dynamic.json",
        camera,
        [{"camera_state": {**camera, "frame_index": 0}}, {"camera_state": moved}],
    )
    with pytest.raises(FurnitureLayoutError, match="dynamic"):
        reuse_camera_from_plan(plan, source)


def test_reuse_camera_keeps_overview_selection_mode(tmp_path: Path) -> None:
    plan = build_episode_plan(
        load_room_layout(_fixture(tmp_path / "room")),
        frame_count=3,
        overview_only=True,
    )
    camera = {
        "position_authoring_m": [1.0, 2.0, 1.55],
        "position_habitat_m": [1.0, 1.55, -2.0],
        "position_ue_cm": [100.0, -200.0, 155.0],
    }
    source = _camera_source(tmp_path / "fixed.json", camera)
    reused = reuse_camera_from_plan(plan, source)
    assert reused["visual_plan"]["camera_selection"]["selection_mode"] == "overview_geometry_only"
    assert reused["camera_reuse"]["source_plan_path"] == str(source)


@pytest.mark.parametrize(
    "value",
    ([1.0, "2.0", 1.55], [1.0, float("nan"), 1.55], [1.0, True, 1.55]),
)
def test_reuse_camera_rejects_nonfinite_or_nonnumeric_positions(
    tmp_path: Path,
    value,
) -> None:
    plan = build_episode_plan(load_room_layout(_fixture(tmp_path / "room")), frame_count=3)
    camera = {
        "position_authoring_m": value,
        "position_habitat_m": [1.0, 1.55, -2.0],
        "position_ue_cm": [100.0, -200.0, 155.0],
    }
    source = _camera_source(tmp_path / "bad.json", camera)
    with pytest.raises(FurnitureLayoutError, match="invalid position_authoring_m"):
        reuse_camera_from_plan(plan, source)


def test_public_furnished_api_propagates_changed_seat_semantics(
    tmp_path: Path,
) -> None:
    room_path = _fixture(tmp_path / "room", seat_count=4)
    manifest = json.loads(room_path.read_text())
    anchors_path = room_path.parent / "functional_anchors.json"
    anchors = json.loads(anchors_path.read_text())
    anchors["seat_points"][0]["position_m"] = [-1.8, -1.4, 0.0]
    anchors["seat_points"][0]["support_height_m"] = 0.61
    anchors["seat_points"][0]["facing_yaw_deg"] = 45.0
    anchors_path.write_text(json.dumps(anchors))
    pose = tmp_path / "pose.json"
    pose.write_text(json.dumps({"bindings": [{
        "actor_id": "person0",
        "seat_affordance_id": "seat_0",
        "root_from_seat_m": [0.0, 0.2, -0.61],
        "ue_anatomical_forward_yaw_deg": 90.0,
    }]}))
    plan = plan_furnished_residential_episode(
        room=room_path,
        pose_bindings=pose,
        output=tmp_path / "plan",
        activity="seated",
        map_path="/Game/Rooms/Test",
        seat_count=1,
        actor_count=1,
        frame_count=3,
    )
    placement = plan["seat_layout"]["actor_placements"][0]
    assert placement["seat_reference"]["seat_surface_height_m"] == pytest.approx(0.61)
    assert placement["seat_reference"]["facing_yaw_deg"] == pytest.approx(45.0)
    assert plan["visual_plan"]["frames"][0]["actor_states"][0]["translation_ue_cm"] != [0, 0, 0]
    assert plan["visual_plan"]["frames"][0]["actor_states"][0]["actor_yaw_ue_deg"] == pytest.approx(-135.0)

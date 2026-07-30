from __future__ import annotations

from copy import deepcopy
import json
import math

import pytest

from avengine.contracts.json_io import canonical_json_sha256
from avengine.optional_backends.spear_visual import (
    BACKEND_ROLE,
    SpearVisualPlanError,
    actor_ue_yaw_degrees,
    build_spear_visual_plan,
    build_spear_visual_plan_from_files,
    camera_ue_yaw_degrees,
    habitat_point_to_apartment_ue_cm,
)
from avengine.sensor_rig_trajectory import (
    compute_sensor_rig_pose_hash,
    materialize_sensor_rig_trajectory,
)


DOG_ASSET = "beagle_asset"
HUMAN_ASSET = "human_asset"


def _quat_y(degrees: float) -> list[float]:
    half = math.radians(degrees) * 0.5
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def _state(actor_id: str, frame_index: int) -> dict:
    human = actor_id == "human0"
    return {
        "actor_id": actor_id,
        "root_transform": {
            "translation_m": [
                float(frame_index) / 100.0 if human else 1.0,
                0.25,
                -2.0 if human else -3.0,
            ],
            "rotation_xyzw": _quat_y(180.0 if human else 90.0),
            "scale": [1.0, 1.0, 1.0],
        },
        "action_id": "walk" if human else "idle",
        "action_time_ticks": frame_index * 3_200,
        "action_phase": (frame_index % 25) / 25.0,
        "pose_hash": "0" * 64,
        "contacts": {},
        "mouth_state": {"open_ratio": 0.0, "vocalizing": False},
    }


def _inputs() -> tuple[dict, dict, dict, dict, dict, dict]:
    timeline = {
        "schema": "avengine_authoritative_timeline_v2",
        "time_base_hz": 48_000,
        "duration_ticks": 240_000,
        "video": {
            "fps_num": 15,
            "fps_den": 1,
            "frame_count": 75,
            "ticks_per_frame": 3_200,
            "view_ids": ["view0"],
        },
        "audio": {
            "sample_rate_hz": 16_000,
            "sample_count": 80_000,
            "ticks_per_sample": 3,
            "channel_count": 4,
        },
        "actors": [
            {
                "actor_id": "dog0",
                "asset_id": DOG_ASSET,
                "template_id": "beagle",
                "body_plan_id": "quadruped_canine",
            },
            {
                "actor_id": "human0",
                "asset_id": HUMAN_ASSET,
                "template_id": "adult_human",
                "body_plan_id": "biped_human",
            },
        ],
        "frames": [
            {
                "frame_index": index,
                "pts_ticks": index * 3_200,
                "sample_start": 0,
                "sample_end": 1,
                "actor_states": [
                    _state("dog0", index),
                    _state("human0", index),
                ],
                "view_pose_hashes": {"view0": "1" * 64},
            }
            for index in range(75)
        ],
        "audio_events": [],
    }
    source_manifest = {
        "schema": "avengine_m6x_fixed_apartment_source_manifest_v1",
        "scenario_id": "S3",
        "variant_id": "A",
        "listener": {"listener_id": "listener0"},
        "sources": [
            {
                "source_endpoint_id": "dog_muzzle",
                "activation": "persistent_silent",
                "endpoint": {
                    "binding": {
                        "kind": "entity_anchor",
                        "entity_instance_id": "dog0",
                        "entity_asset_id": DOG_ASSET,
                    }
                },
                "trajectory": {
                    "frame_count": 75,
                    "positions_m": [[1.0, 0.6, -3.0] for _ in range(75)],
                },
            },
            {
                "source_endpoint_id": "human_mouth",
                "activation": "active",
                "endpoint": {
                    "binding": {
                        "kind": "entity_anchor",
                        "entity_instance_id": "human0",
                        "entity_asset_id": HUMAN_ASSET,
                    }
                },
                "trajectory": {
                    "frame_count": 75,
                    "positions_m": [[0.0, 1.7, -2.0] for _ in range(75)],
                },
            },
        ],
    }
    assessment = {"status": "present", "value": True}
    flags = {
        "schema": "avengine_m6_legacy_flag_report_v1",
        "clip_flags": {"steady_walk": assessment},
        "source_flags": {"dog_muzzle": {}, "human_mouth": {}},
        "pair_flags": [],
    }
    room_capsule = {
        "schema": "avengine_m6x_room_capsule_v1",
        "room_capsule_id": "apartment_fixed_v1",
        "revision": "v1",
        "source_scene_provenance": {
            "provider": "SPEAR_Unreal",
            "scene_id": "apartment_0000",
        },
        "room_registry_ref": {"room_id": "apartment_room_v1"},
        "camera_listener_rig": {"listener_id": "listener0"},
    }
    gate_record = {
        "status": "pass",
        "failed_frame_indices": [],
        "frames": [{"status": "pass"} for _ in range(75)],
    }
    qualification = {
        "schema": "avengine_m6x_fixed_apartment_native_qualification_v1",
        "status": "pass",
        "room_id": "apartment_room_v1",
        "listener": {
            "position_m": [-0.7, 1.471, 0.65],
            "yaw_deg": 55.0,
            "camera_hfov_degrees": 105.0,
        },
        "source_center_gate": {
            "status": "pass",
            "failed_source_frame_indices": {},
            "sources": {
                "dog_muzzle": deepcopy(gate_record),
                "human_mouth": deepcopy(gate_record),
            },
        },
    }
    bindings = {
        DOG_ASSET: {
            "blueprint_class_path": "/Game/AVEngine/BP_Beagle.BP_Beagle_C",
            "idle_animation": "Idle",
            "walking_animation": "Walking",
            "ue_anatomical_forward_yaw_deg": 0.0,
        },
        HUMAN_ASSET: {
            "blueprint_class_path": "/Game/AVEngine/BP_Human.BP_Human_C",
            "idle_animation": "Idle",
            "walking_animation": "Walking",
            "ue_anatomical_forward_yaw_deg": 90.0,
        },
    }
    return timeline, source_manifest, flags, room_capsule, qualification, bindings


def _build(
    values: tuple[dict, dict, dict, dict, dict, dict],
    *,
    sensor_rig_trajectory: dict | None = None,
) -> dict:
    timeline, source_manifest, flags, room_capsule, qualification, bindings = values
    return build_spear_visual_plan(
        timeline=timeline,
        source_manifest=source_manifest,
        flags=flags,
        room_capsule=room_capsule,
        qualification=qualification,
        actor_bindings=bindings,
        sensor_rig_trajectory=sensor_rig_trajectory,
    )


def test_legacy_apartment_coordinate_and_camera_yaw_formulas() -> None:
    assert habitat_point_to_apartment_ue_cm([1.25, 2.0, -3.0]) == (
        125.0,
        -300.0,
        200.0,
    )
    assert camera_ue_yaw_degrees(55.0) == -145.0


def test_actor_yaw_uses_body_forward_and_asset_local_forward() -> None:
    # Beagle local +X -> Habitat world -Z -> UE world -Y; UE asset +X is yaw 0.
    assert actor_ue_yaw_degrees(_quat_y(90.0), [1, 0, 0], 0.0) == pytest.approx(
        -90.0
    )
    # Human local +Z -> Habitat world -Z -> UE world -Y; UE asset +Y is yaw 90.
    assert actor_ue_yaw_degrees(_quat_y(180.0), [0, 0, 1], 90.0) == pytest.approx(
        -180.0
    )


def test_plan_preserves_timeline_state_and_room_authority() -> None:
    plan = _build(_inputs())
    assert plan["backend_role"] == BACKEND_ROLE == "comparison_visual"
    assert plan["authority"] == {
        "actor_state": "Timeline_v2",
        "room_identity_and_layout": "RoomCapsule",
        "source_logic": "source_manifest_and_flags",
        "source_center_placement": "room_qualification",
        "camera_listener_state": "room_qualification_static_compatibility",
        "backend_may_replan": False,
    }
    assert plan["room"]["room_capsule_id"] == "apartment_fixed_v1"
    assert plan["camera"]["ue_position_cm"] == pytest.approx(
        [-70.0, 65.0, 147.1]
    )
    assert plan["camera"]["ue_yaw_deg"] == -145.0
    assert plan["camera"]["default_pose_scope"] == "frame_zero_compatibility_only"
    assert len(plan["frames"]) == 75
    camera_states = [frame["camera_state"] for frame in plan["frames"]]
    assert len({state["pose_hash"] for state in camera_states}) == 1
    assert all(
        state["pose_hash"]
        == compute_sensor_rig_pose_hash(state["world_from_rig"])
        for state in camera_states
    )
    first = plan["frames"][0]["actor_states"]
    assert [state["actor_id"] for state in first] == ["dog0", "human0"]
    assert first[0]["action_id"] == "idle"
    assert first[0]["action_phase"] == 0.0
    assert first[0]["translation_m"] == [1.0, 0.25, -3.0]
    assert first[0]["translation_ue_cm"] == [100.0, -300.0, 25.0]
    assert first[0]["ue_animation"] == "Idle"
    assert first[0]["actor_yaw_ue_deg"] == pytest.approx(-90.0)
    assert first[1]["ue_animation"] == "Walking"
    assert first[1]["actor_yaw_ue_deg"] == pytest.approx(-180.0)


def test_explicit_sensor_rig_trajectory_drives_every_camera_frame() -> None:
    values = _inputs()
    trajectory = materialize_sensor_rig_trajectory(
        trajectory_id="spear_dynamic_camera_test_v1",
        program={
            "kind": "WAYPOINT_ROUTE",
            "waypoints": [
                {
                    "frame_index": 0,
                    "position_m": [-0.7, 1.471, 0.65],
                    "yaw_deg": 55.0,
                },
                {
                    "frame_index": 74,
                    "position_m": [0.8, 1.471, -1.25],
                    "yaw_deg": -35.0,
                },
            ],
            "interpolation": "LINEAR_POSITION_SHORTEST_YAW",
        },
    )
    for timeline_frame, trajectory_frame in zip(
        values[0]["frames"], trajectory["frames"]
    ):
        timeline_frame["view_pose_hashes"] = {
            "view0": trajectory_frame["pose_hash"]
        }

    plan = _build(values, sensor_rig_trajectory=trajectory)

    assert plan["authority"]["camera_listener_state"] == "SensorRigTrajectory_v1"
    assert plan["camera"]["trajectory_id"] == "spear_dynamic_camera_test_v1"
    assert plan["camera"]["timeline_pose_hash_crosscheck"] is True
    states = [frame["camera_state"] for frame in plan["frames"]]
    assert states[0]["habitat_position_m"] == [-0.7, 1.471, 0.65]
    assert states[-1]["habitat_position_m"] == [0.8, 1.471, -1.25]
    assert states[0]["ue_position_cm"] == pytest.approx([-70.0, 65.0, 147.1])
    assert states[-1]["ue_position_cm"] == pytest.approx([80.0, -125.0, 147.1])
    assert states[0]["ue_yaw_deg"] == pytest.approx(-145.0)
    assert states[-1]["ue_yaw_deg"] == pytest.approx(-55.0)
    assert len({state["pose_hash"] for state in states}) == 75
    assert all(
        state["pose_hash"]
        == compute_sensor_rig_pose_hash(state["world_from_rig"])
        for state in states
    )


def test_explicit_sensor_rig_trajectory_requires_matching_timeline_hash() -> None:
    values = _inputs()
    trajectory = materialize_sensor_rig_trajectory(
        trajectory_id="spear_hash_mismatch_test_v1",
        program={
            "kind": "ROTATE_IN_PLACE",
            "position_m": [-0.7, 1.471, 0.65],
            "start_yaw_deg": 55.0,
            "end_yaw_deg": -35.0,
            "yaw_interpolation": "SHORTEST_ARC",
        },
    )
    for timeline_frame, trajectory_frame in zip(
        values[0]["frames"], trajectory["frames"]
    ):
        timeline_frame["view_pose_hashes"] = {
            "view0": trajectory_frame["pose_hash"]
        }
    values[0]["frames"][37]["view_pose_hashes"]["view0"] = "f" * 64

    with pytest.raises(SpearVisualPlanError, match="frame 37.*differs"):
        _build(values, sensor_rig_trajectory=trajectory)


def test_declared_sensor_rig_sidecar_reference_is_required_and_hash_bound() -> None:
    values = _inputs()
    trajectory = materialize_sensor_rig_trajectory(
        trajectory_id="spear_manifest_bound_test_v1",
        program={
            "kind": "HOLD",
            "position_m": [-0.7, 1.471, 0.65],
            "yaw_deg": 55.0,
        },
    )
    for timeline_frame, trajectory_frame in zip(
        values[0]["frames"], trajectory["frames"]
    ):
        timeline_frame["view_pose_hashes"] = {
            "view0": trajectory_frame["pose_hash"]
        }
    values[1]["listener"]["sensor_rig_trajectory"] = {
        "trajectory_id": trajectory["trajectory_id"],
        "content_sha256": canonical_json_sha256(trajectory),
        "relative_path": "metadata/sensor_rig_trajectory.json",
    }

    plan = _build(values, sensor_rig_trajectory=trajectory)
    assert plan["camera"]["trajectory_id"] == trajectory["trajectory_id"]

    with pytest.raises(SpearVisualPlanError, match="no sidecar"):
        _build(values)
    values[1]["listener"]["sensor_rig_trajectory"]["content_sha256"] = "0" * 64
    with pytest.raises(SpearVisualPlanError, match="reference differs"):
        _build(values, sensor_rig_trajectory=trajectory)


def test_plan_preserves_authoritative_not_evaluated_flag_state() -> None:
    values = list(_inputs())
    values[2]["clip_flags"]["steady_walk"] = {
        "status": "not_evaluated",
        "value": None,
    }

    plan = _build(tuple(values))

    assert plan["source_logic"]["clip_flags"]["steady_walk"] == {
        "status": "not_evaluated",
        "value": None,
    }


def test_file_loader_emits_no_host_paths(tmp_path) -> None:
    values = _inputs()
    names = ("timeline", "source_manifest", "flags", "room_capsule", "qualification")
    paths = {}
    for name, value in zip(names, values[:5]):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[f"{name}_path"] = path
    plan = build_spear_visual_plan_from_files(
        **paths,
        actor_bindings=values[5],
    )
    assert str(tmp_path) not in json.dumps(plan, sort_keys=True)


def test_file_loader_accepts_sensor_rig_trajectory_sidecar(tmp_path) -> None:
    values = _inputs()
    trajectory = materialize_sensor_rig_trajectory(
        trajectory_id="spear_file_loader_dynamic_v1",
        program={
            "kind": "POLYLINE_MOVE",
            "path_points_m": [[-0.7, 1.471, 0.65], [0.0, 1.471, -0.5]],
            "position_interpolation": "ARC_LENGTH",
            "heading_policy": "FIXED_YAW",
            "initial_yaw_deg": 55.0,
        },
    )
    for timeline_frame, trajectory_frame in zip(
        values[0]["frames"], trajectory["frames"]
    ):
        timeline_frame["view_pose_hashes"] = {
            "view0": trajectory_frame["pose_hash"]
        }
    names = (
        "timeline",
        "source_manifest",
        "flags",
        "room_capsule",
        "qualification",
    )
    paths = {}
    for name, value in zip(names, values[:5]):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[f"{name}_path"] = path
    trajectory_path = tmp_path / "sensor_rig_trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")

    plan = build_spear_visual_plan_from_files(
        **paths,
        actor_bindings=values[5],
        sensor_rig_trajectory_path=trajectory_path,
    )

    assert plan["frames"][-1]["camera_state"]["pose_hash"] == trajectory[
        "frames"
    ][-1]["pose_hash"]
    assert str(tmp_path) not in json.dumps(plan, sort_keys=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("non_finite", "finite"),
        ("frame_count", "75 frames"),
        ("actor_closure", "actor closure"),
        ("missing_binding", "no SPEAR actor binding"),
        ("gate", "source-center gate"),
        ("source_actor", "does not resolve in Timeline"),
        ("quaternion", "unit quaternion"),
        ("action", "no Idle/Walking"),
    ],
)
def test_invalid_inputs_fail_closed(mutation: str, message: str) -> None:
    values = list(_inputs())
    if mutation == "non_finite":
        values[0]["frames"][0]["actor_states"][0]["root_transform"][
            "translation_m"
        ][0] = float("nan")
    elif mutation == "frame_count":
        values[0]["frames"].pop()
    elif mutation == "actor_closure":
        values[0]["frames"][0]["actor_states"].reverse()
    elif mutation == "missing_binding":
        del values[5][DOG_ASSET]
    elif mutation == "gate":
        values[4]["source_center_gate"]["sources"]["human_mouth"]["status"] = "fail"
    elif mutation == "source_actor":
        values[1]["sources"][0]["endpoint"]["binding"][
            "entity_instance_id"
        ] = "missing_actor"
    elif mutation == "quaternion":
        values[0]["frames"][0]["actor_states"][0]["root_transform"][
            "rotation_xyzw"
        ] = [0.0, 0.0, 0.0, 0.0]
    elif mutation == "action":
        values[0]["frames"][0]["actor_states"][0]["action_id"] = "run"
    with pytest.raises(SpearVisualPlanError, match=message):
        _build(tuple(values))

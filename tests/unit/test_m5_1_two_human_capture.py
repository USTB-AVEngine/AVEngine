from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from avengine.m5_1.two_human_capture import (
    TwoHumanCaptureError,
    _require_same_runtime_file,
    _safe_regular_path,
    _validate_camera_runtime_navigation,
    load_two_human_capture_authority,
    validate_two_human_authority_documents,
)
from avengine.runtime_profiles import load_source_asset_runtime_registry


EPISODE = "mp3d_two_human_0004"
ROOT = Path(__file__).resolve().parents[2]
MALE_ID = "rocketbox_human_male_adult_01_m5_1_candidate"
MALE_REVISION = "native_runtime_ue_v3"
FEMALE_ID = "lead_b_rocketbox_adults_female_adult_01_original_v1"
FEMALE_REVISION = "native_runtime_ue_v1"
RIG = {
    "translation_m": [0.0, 1.572447, -0.5],
    "rotation_xyzw": [0.0, 0.5, 0.0, 0.8660254037844386],
}
ROOTS = {
    "source1": [-4.6, 0.072447, -2.35],
    "source2": [-3.75, 0.072447, -3.35],
}
OFFSETS = {
    "source1": [0.0, 1.61, 0.0],
    "source2": [0.0, 1.569012451171875, 0.0],
}
ACTOR_ROTATIONS = {
    "source1": [0.0, 0.2, 0.0, 0.9797958971132712],
    "source2": [0.0, -0.1, 0.0, 0.99498743710662],
}


def _documents(tmp_path: Path) -> dict[str, dict[str, object]]:
    male_glb = tmp_path / "male.glb"
    female_glb = tmp_path / "female.glb"
    male_glb.write_bytes(b"male")
    female_glb.write_bytes(b"female")
    actors = [
        {
            "actor_id": "source1_actor",
            "asset_id": MALE_ID,
            "asset_revision": MALE_REVISION,
            "template_id": "rocketbox_human_male_adult_01",
            "body_plan_id": "biped_human",
            "emitter_offset_m": OFFSETS["source1"],
        },
        {
            "actor_id": "source2_actor",
            "asset_id": FEMALE_ID,
            "asset_revision": FEMALE_REVISION,
            "template_id": "rocketbox_adults_female_adult_01",
            "body_plan_id": "biped_human",
            "emitter_offset_m": OFFSETS["source2"],
        },
    ]
    frames = []
    rig_frames = []
    for index in range(75):
        rig_frame = {
            "frame_index": index,
            "pts_ticks": index * 3200,
            "world_from_rig": deepcopy(RIG),
        }
        rig_frames.append(rig_frame)
        frames.append(
            {
                "frame_index": index,
                "pts_ticks": index * 3200,
                "camera_state": deepcopy(rig_frame),
                "actor_states": [
                    {
                        "actor_id": "source1_actor",
                        "asset_id": MALE_ID,
                        "frame_index": index,
                        "action_id": "idle",
                        "action_phase": 0.0,
                        "action_time_ticks": index * 3200,
                        "translation_m": ROOTS["source1"],
                        "rotation_xyzw": ACTOR_ROTATIONS["source1"],
                    },
                    {
                        "actor_id": "source2_actor",
                        "asset_id": FEMALE_ID,
                        "frame_index": index,
                        "action_id": "idle",
                        "action_phase": 0.0,
                        "action_time_ticks": index * 3200,
                        "translation_m": ROOTS["source2"],
                        "rotation_xyzw": ACTOR_ROTATIONS["source2"],
                    },
                ],
            }
        )
    centers = {
        slot: [
            [ROOTS[slot][axis] + OFFSETS[slot][axis] for axis in range(3)]
            for _ in range(75)
        ]
        for slot in ("source1", "source2")
    }
    jobs = [
        {
            "job_id": f"rir_{job_index:06d}",
            "source_position_m": deepcopy(centers[slot][0]),
            "listener_position_m": deepcopy(RIG["translation_m"]),
            "listener_orientation_wxyz": [
                RIG["rotation_xyzw"][3],
                *RIG["rotation_xyzw"][:3],
            ],
            "uses": [
                {
                    "episode_id": EPISODE,
                    "frame_index": index,
                    "source_slot_id": slot,
                }
                for index in range(75)
            ],
        }
        for job_index, slot in enumerate(("source1", "source2"))
    ]
    plan = {
        "schema": "avengine_optional_spear_visual_plan_v1",
        "backend_role": "comparison_visual",
        "actors": actors,
        "camera": {
            "sensor_rig_trajectory_id": f"{EPISODE}__rig",
            "listener_id": "listener0",
        },
        "room": {
            "room_id": "habitat_mp3d_example_17DRP5sb8fy",
            "room_revision": "r1",
            "scene_id": "17DRP5sb8fy",
        },
        "render": {
            "fps_den": 1,
            "fps_num": 15,
            "frame_count": 75,
            "ticks_per_frame": 3200,
        },
        "qualification": {
            "qualification_claim": False,
            "formal_dataset_count": 0,
        },
        "frames": frames,
    }
    return {
        "atom": {
            "schema": "avengine_native_strict_two_human_mp3d_room_atom_request_v2",
            "episode_id": EPISODE,
            "room": {
                "room_id": "habitat_mp3d_example_17DRP5sb8fy",
                "room_revision": "r1",
                "scene_id": "17DRP5sb8fy",
            },
            "actor_framing": {
                "actor_bindings": [
                    {
                        "actor_id": "source1_actor",
                        "asset_id": MALE_ID,
                        "asset_revision": MALE_REVISION,
                        "source_asset_path": str(male_glb),
                        "skin_index": 0,
                        "action_name_by_action_id": {
                            "idle": "Standing_Idle",
                            "walk": "Walking",
                        },
                    },
                    {
                        "actor_id": "source2_actor",
                        "asset_id": FEMALE_ID,
                        "asset_revision": FEMALE_REVISION,
                        "source_asset_path": str(female_glb),
                        "skin_index": 0,
                        "action_name_by_action_id": {
                            "idle": "Standing_Idle",
                            "walk": "Walking",
                        },
                    },
                ]
            },
            "camera_framing": {"calibration": {"near_m": 0.05}},
            "qualification_claim": False,
            "formal_dataset_count": 0,
        },
        "suite": {
            "schema": "avengine_optional_spear_imported_glb_suite_v1",
            "backend_role": "comparison_visual",
            "qualification_claim": False,
            "formal_dataset_count": 0,
            "scenarios": [
                {
                    "schema": "avengine_optional_spear_imported_glb_scenario_v1",
                    "scenario_id": EPISODE,
                    "backend_role": "comparison_visual",
                    "render": {
                        "frame_count": 75,
                        "frame_rate_hz": 15,
                        "height": 720,
                        "width": 1280,
                        "horizontal_fov_deg": 90.0,
                    },
                    "plan": plan,
                }
            ],
        },
        "sensor_rig": {
            "schema": "avengine_sensor_rig_trajectory_v1",
            "trajectory_id": f"{EPISODE}__rig",
            "rig_id": "camera_rig_0",
            "listener_id": "listener0",
            "formal_view_id": "view0",
            "camera_listener_coupling": "rigid_colocated_cooriented",
            "time_base_hz": 48000,
            "ticks_per_frame": 3200,
            "coordinate_frame": "avengine_world_right_handed_y_up_m",
            "pose_model": "yaw_only_about_world_positive_y",
            "program": {
                "kind": "HOLD",
                "position_m": RIG["translation_m"],
                "yaw_deg": 60.0,
            },
            "rig_from_camera": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "rig_from_listener": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "frame_count": 75,
            "frame_rate_hz": 15,
            "duration_ticks": 240000,
            "frames": rig_frames,
        },
        "trajectory_bank": {
            "schema": "avengine_room_trajectory_bank_v2",
            "frame_count": 75,
            "frame_rate_hz": 15,
            "seconds_per_episode": 5.0,
            "source_slots": ["source1", "source2"],
            "episodes": [
                {
                    "episode_id": EPISODE,
                    "motion_case": "strict_two_human_static_mp3d",
                    "source_root_paths_m": {
                        slot: [ROOTS[slot] for _ in range(75)]
                        for slot in ("source1", "source2")
                    },
                    "source_center_paths_m": centers,
                }
            ],
        },
        "rir_plan": {
            "schema": "avengine_room_rir_job_plan_v2",
            "status": "planned_not_run",
            "claim_boundary": "planned semantic RIR jobs; RLR has not run",
            "producer_backend": "RLR Audio Propagation",
            "cache_artifact": "room impulse response (RIR)",
            "source_acoustic_profile": "omnidirectional_point_source_v1",
            "listener_pose_mode": "per_episode_frame",
            "dry_audio_independent": True,
            "slot_identity_affects_cache_key": False,
            "cache_key_fields": [
                "source_position_m",
                "listener_position_m",
                "listener_orientation_wxyz",
            ],
            "stride_frames": 1,
            "requested_pair_state_count": 150,
            "unique_rir_job_count": 2,
            "cache_reuse_count": 148,
            "unique_listener_pose_count": 1,
            "jobs": jobs,
        },
        "runtime_profiles": load_source_asset_runtime_registry(
            ROOT / "examples/runtime/source_asset_runtime_profiles.json"
        ),
        "room": {
            "room_id": "habitat_mp3d_example_17DRP5sb8fy",
            "room_kind": "habitat_native",
            "geometry_representation": "real_surface_mesh",
            "coordinate_system": {
                "handedness": "right",
                "up_axis": "+Y",
                "forward_axis": "-Z",
                "linear_unit": "meter",
                "quaternion_order": "xyzw",
            },
            "scene": {
                "scene_id": "/external/17DRP5sb8fy.glb",
                "navmesh_policy": "load_declared",
                "load_semantic_mesh": True,
                "enable_physics": True,
            },
        },
        "m1_request": {
            "room_id": "habitat_mp3d_example_17DRP5sb8fy",
            "primary_camera_rig": {
                "rig_id": "camera_rig_0",
                "view_id": "view0",
                "world_from_rig": deepcopy(RIG),
                "shared_calibration": {
                    "projection": "pinhole",
                    "resolution_hw": [720, 1280],
                    "hfov_degrees": 90.0,
                    "near_m": 0.05,
                    "far_m": 100.0,
                    "rig_from_sensor": {
                        "translation_m": [0.0, 0.0, 0.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                "modalities": [
                    {"modality": "rgb", "sensor_uuid": "rgb"},
                    {"modality": "depth", "sensor_uuid": "depth"},
                    {"modality": "semantic", "sensor_uuid": "semantic"},
                ],
            },
            "listener": {
                "listener_id": "listener0",
                "attached_to": "camera_rig_0",
                "rig_from_listener": {
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "sources": [
                {
                    "source_id": slot,
                    "world_from_source": {
                        "translation_m": deepcopy(centers[slot][0]),
                        "rotation_xyzw": ACTOR_ROTATIONS[slot],
                    },
                }
                for slot in ("source1", "source2")
            ],
        },
    }


def _validate(documents: dict[str, dict[str, object]]):
    return validate_two_human_authority_documents(**documents)


def test_authority_join_keeps_ue_comparison_and_frozen_habitat_state(
    tmp_path: Path,
) -> None:
    documents = _documents(tmp_path)
    authority = _validate(documents)
    assert authority.suite_visual_role == "comparison_visual"
    assert authority.episode_id == EPISODE
    assert [actor.package_stem for actor in authority.actors] == [
        "human0",
        "human1",
    ]
    assert [actor.walking_profile_sample_count for actor in authority.actors] == [
        None,
        19,
    ]
    assert [actor.semantic_id for actor in authority.actors] == [62000, 62001]
    assert authority.actor_frames[0][0].rotation_xyzw == (
        0.0,
        0.2,
        0.0,
        0.9797958971132712,
    )
    assert authority.actor_frames[0][-1].pts_ticks == 236800


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["suite"].update({"backend_role": "production_visual"}),
            "schemas/roles must remain exact",
        ),
        (
            lambda value: value["suite"].update({"schema": "foreign"}),
            "schemas/roles must remain exact",
        ),
        (
            lambda value: value["suite"].update({"formal_dataset_count": 1}),
            "Timeline or non-formal boundary drift",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["render"].update(
                {"frame_count": 74}
            ),
            "Timeline or non-formal boundary drift",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["render"].update(
                {"ticks_per_frame": 3199}
            ),
            "Timeline or non-formal boundary drift",
        ),
        (
            lambda value: value["sensor_rig"]["frames"][9]["world_from_rig"][
                "translation_m"
            ].__setitem__(0, 1.0),
            "camera and sensor rig differ",
        ),
        (
            lambda value: value["trajectory_bank"]["episodes"][0][
                "source_center_paths_m"
            ]["source2"][4].__setitem__(1, 9.0),
            "trajectory centers differ",
        ),
        (
            lambda value: value["trajectory_bank"].update(
                {"source_slots": ["source2", "source1"]}
            ),
            "trajectory bank identity or Timeline drift",
        ),
        (
            lambda value: value["trajectory_bank"]["episodes"][0].update(
                {"motion_case": "moving"}
            ),
            "trajectory motion case drift",
        ),
        (
            lambda value: value["rir_plan"]["jobs"][0].update(
                {"source_position_m": [1.0, 2.0, 3.0]}
            ),
            "RIR state differs",
        ),
        (
            lambda value: value["m1_request"]["primary_camera_rig"][
                "shared_calibration"
            ].update({"resolution_hw": [240, 320]}),
            "M1 camera does not match",
        ),
        (
            lambda value: value["m1_request"]["primary_camera_rig"][
                "shared_calibration"
            ].update({"projection": "orthographic"}),
            "M1 camera does not match",
        ),
        (
            lambda value: value["m1_request"]["primary_camera_rig"][
                "shared_calibration"
            ].update({"near_m": 0.1}),
            "M1 camera does not match",
        ),
        (
            lambda value: value["m1_request"]["primary_camera_rig"][
                "shared_calibration"
            ].update({"far_m": 99.0}),
            "M1 camera does not match",
        ),
        (
            lambda value: value["atom"]["camera_framing"]["calibration"].update(
                {"near_m": 0.04}
            ),
            "M1 camera does not match",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"action_time_ticks": 22401}),
            "off the 15 Hz grid",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"frame_index": 6}),
            "action_time_ticks is off|frame 7 action_time_ticks",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"asset_id": FEMALE_ID}),
            "action_time_ticks is off|frame 7 action_time_ticks",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"action_phase": 1.0}),
            "action_time_ticks is off|frame 7 action_time_ticks",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"action_id": "walk"}),
            "strict static actor action drift",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"action_phase": 0.5}),
            "strict static actor action drift",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"action_time_ticks": 25600}),
            "strict static actor action drift",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"translation_m": [-4.5, 0.072447, -2.35]}),
            "strict static root/rotation must remain frozen",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["frames"][7][
                "actor_states"
            ][0].update({"rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}),
            "strict static root/rotation must remain frozen",
        ),
        (
            lambda value: value["atom"]["actor_framing"]["actor_bindings"][1].update(
                {"skin_index": 1}
            ),
            "human skin/body/timeline binding drift",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["actors"][1].update(
                {"body_plan_id": "quadruped"}
            ),
            "human skin/body/timeline binding drift",
        ),
        (
            lambda value: value["rir_plan"]["jobs"][1]["uses"].__setitem__(
                74, deepcopy(value["rir_plan"]["jobs"][1]["uses"][73])
            ),
            "maps one use more than once",
        ),
        (
            lambda value: value["rir_plan"]["jobs"].append(
                deepcopy(value["rir_plan"]["jobs"][0])
            ),
            "job identity is invalid",
        ),
        (
            lambda value: value["rir_plan"]["jobs"][0].pop("job_id"),
            "job 0 fields are invalid",
        ),
        (
            lambda value: value["rir_plan"]["jobs"][0].update(
                {"listener_orientation_wxyz": [1.0, 0.0, 0.0, 1.0]}
            ),
            "must be unit normalized",
        ),
        (
            lambda value: value["m1_request"]["listener"].update(
                {"listener_id": "listener9"}
            ),
            "listener or camera rig identity drift",
        ),
        (
            lambda value: value["m1_request"]["sources"][1]["world_from_source"][
                "translation_m"
            ].__setitem__(0, 99.0),
            "source2 position differs",
        ),
        (
            lambda value: value["m1_request"]["sources"][1]["world_from_source"].update(
                {"rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}
            ),
            "source2 rotation differs",
        ),
        (
            lambda value: value["sensor_rig"].update({"listener_id": "listener9"}),
            "sensor rig identity or Timeline drift",
        ),
        (
            lambda value: value["sensor_rig"].update({"time_base_hz": 44100}),
            "sensor rig identity or Timeline drift",
        ),
        (
            lambda value: value["sensor_rig"]["program"].update({"yaw_deg": 59.0}),
            "HOLD program differs",
        ),
        (
            lambda value: value["sensor_rig"]["rig_from_listener"][
                "translation_m"
            ].__setitem__(0, 0.1),
            "offsets must remain identity",
        ),
        (
            lambda value: value["rir_plan"].update({"dry_audio_independent": False}),
            "full planning metadata is invalid",
        ),
        (
            lambda value: value["rir_plan"].update(
                {"cache_key_fields": ["source_position_m"]}
            ),
            "cache key fields are invalid",
        ),
        (
            lambda value: value["atom"]["actor_framing"]["actor_bindings"][1].update(
                {
                    "source_asset_path": value["atom"]["actor_framing"][
                        "actor_bindings"
                    ][0]["source_asset_path"]
                }
            ),
            "distinct semantics, assets, and source paths",
        ),
        (
            lambda value: value["suite"]["scenarios"][0]["plan"]["room"].update(
                {"scene_id": "foreign"}
            ),
            "room/revision/scene differs",
        ),
        (
            lambda value: value["room"]["coordinate_system"].update(
                {"forward_axis": "+Z"}
            ),
            "room coordinate/scene production semantics drift",
        ),
        (
            lambda value: value["room"].update(
                {"geometry_representation": "debug_aabb_proxy"}
            ),
            "room coordinate/scene production semantics drift",
        ),
        (
            lambda value: value["room"]["scene"].update(
                {"navmesh_policy": "recompute"}
            ),
            "room coordinate/scene production semantics drift",
        ),
        (
            lambda value: value["room"]["scene"].update({"load_semantic_mesh": False}),
            "room coordinate/scene production semantics drift",
        ),
        (
            lambda value: value["room"]["scene"].update({"enable_physics": False}),
            "room coordinate/scene production semantics drift",
        ),
    ],
)
def test_authority_join_rejects_cross_document_mutations(
    mutate, message: str, tmp_path: Path
) -> None:
    documents = _documents(tmp_path)
    mutate(documents)
    with pytest.raises(TwoHumanCaptureError, match=message):
        _validate(documents)


def test_authority_join_rejects_symlinked_source_glb(tmp_path: Path) -> None:
    documents = _documents(tmp_path)
    link = tmp_path / "female-link.glb"
    link.symlink_to(tmp_path / "female.glb")
    documents["atom"]["actor_framing"]["actor_bindings"][1]["source_asset_path"] = str(
        link
    )
    with pytest.raises(TwoHumanCaptureError, match="cannot contain symlinks"):
        _validate(documents)


def test_runtime_profile_registry_path_rejects_symlink(tmp_path: Path) -> None:
    registry = tmp_path / "profiles.json"
    registry.write_text("{}", encoding="utf-8")
    link = tmp_path / "profiles-link.json"
    link.symlink_to(registry)
    with pytest.raises(TwoHumanCaptureError, match="cannot contain symlinks"):
        _safe_regular_path(link, owner="runtime profile registry")


def test_runtime_samefile_allows_symlink_alias_and_rejects_other_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "scene.glb"
    target.write_bytes(b"scene")
    alias = tmp_path / "scene-alias.glb"
    alias.symlink_to(target)

    _require_same_runtime_file(alias, target, owner="scene")

    other = tmp_path / "other.glb"
    other.write_bytes(b"other")
    with pytest.raises(TwoHumanCaptureError, match="differs from resolved M1 runtime"):
        _require_same_runtime_file(alias, other, owner="scene")


def test_solver_and_render_agent_radii_are_independent_positive_values() -> None:
    _validate_camera_runtime_navigation(
        {"agent_height_m": 1.5, "agent_radius_m": 0.25},
        {"agent_height_m": 1.5, "agent_radius_m": 0.1},
    )
    for camera, navigation in (
        (
            {"agent_height_m": 1.5, "agent_radius_m": 0.0},
            {"agent_height_m": 1.5, "agent_radius_m": 0.1},
        ),
        (
            {"agent_height_m": 1.5, "agent_radius_m": 0.25},
            {"agent_height_m": 1.5, "agent_radius_m": -0.1},
        ),
    ):
        with pytest.raises(TwoHumanCaptureError, match="navigation values"):
            _validate_camera_runtime_navigation(camera, navigation)


def test_real_v4_authority_loader_with_projected_m1_request(tmp_path: Path) -> None:
    preflight = ROOT / "tmp/lead_a_strict_two_human_mp3d_room_v4/cpu_preflight_v1"
    atom = ROOT / "examples/qa/native_strict_two_human_mp3d_room_atom_v3.json"
    room = ROOT / "examples/m2/rooms/habitat_mp3d_articulated_review/room_manifest.json"
    required = [
        atom,
        room,
        preflight / "suite_execution_plan.json",
        preflight / "sensor_rig_trajectory.json",
        preflight / "trajectory_bank.json",
        preflight / "rir_job_plan.json",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("real v4 CPU authority is unavailable")
    suite = json.loads(required[2].read_text(encoding="utf-8"))
    rig = json.loads(required[3].read_text(encoding="utf-8"))
    bank = json.loads(required[4].read_text(encoding="utf-8"))
    scenario = suite["scenarios"][0]
    first_frame = scenario["plan"]["frames"][0]
    centers = bank["episodes"][0]["source_center_paths_m"]
    projected = {
        "schema": "avengine_m1_capture_request_v1",
        "request_id": "mp3d_two_human_projected_m1_v1",
        "room_id": "habitat_mp3d_example_17DRP5sb8fy",
        "seed": 17,
        "primary_camera_rig": {
            "rig_id": "camera_rig_0",
            "view_id": "view0",
            "world_from_rig": rig["frames"][0]["world_from_rig"],
            "shared_calibration": {
                "projection": "pinhole",
                "resolution_hw": [
                    scenario["render"]["height"],
                    scenario["render"]["width"],
                ],
                "hfov_degrees": scenario["render"]["horizontal_fov_deg"],
                "near_m": 0.05,
                "far_m": 100.0,
                "rig_from_sensor": {
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "modalities": [
                {"modality": "rgb", "sensor_uuid": "rig_rgb"},
                {"modality": "depth", "sensor_uuid": "rig_depth"},
                {"modality": "semantic", "sensor_uuid": "rig_semantic"},
            ],
        },
        "listener": {
            "listener_id": "listener0",
            "attached_to": "camera_rig_0",
            "rig_from_listener": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        "sources": [
            {
                "source_id": slot,
                "world_from_source": {
                    "translation_m": centers[slot][0],
                    "rotation_xyzw": first_frame["actor_states"][index][
                        "rotation_xyzw"
                    ],
                },
            }
            for index, slot in enumerate(("source1", "source2"))
        ],
        "qa_views": [
            {
                "qa_id": "navmesh_topdown",
                "kind": "topdown",
                "meters_per_pixel": 0.05,
                "height_m": 0.072447,
            }
        ],
    }
    request = tmp_path / "projected_m1.json"
    request.write_text(json.dumps(projected), encoding="utf-8")

    authority = load_two_human_capture_authority(
        atom_request_path=atom,
        suite_plan_path=required[2],
        sensor_rig_path=required[3],
        trajectory_bank_path=required[4],
        rir_plan_path=required[5],
        room_manifest_path=room,
        m1_request_path=request,
        runtime_root="/data/jzy/code/habitat-sim-AVEngine",
    )

    assert authority.episode_id == scenario["scenario_id"]
    assert authority.resolution_hw == (720, 1280)

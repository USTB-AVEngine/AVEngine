#!/usr/bin/env python3
"""Build a hash-free CPU preflight for the Skokloster strict M/F Episode."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FRAME_COUNT = 75
FPS = 15
TICKS_PER_FRAME = 3200
EPISODE_SAMPLES = 80000
SAMPLE_RATE_HZ = 16000
PACKAGED_MAP = (
    "/Game/MyAssets/Audioset/Scenes/skokloster_castle/Maps/skokloster_castle_strict"
)
PACKAGE_ID = "habitat_test_skokloster_castle_raw_research_v1_rlr_incompatible_filter_v2"
MALE_ASSET = "rocketbox_human_male_adult_01_m5_1_candidate"
FEMALE_ASSET = "lead_b_rocketbox_adults_female_adult_01_original_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    _require(
        not path.exists() and not path.is_symlink(), f"refusing to replace: {path}"
    )
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _vector(value: Any, *, length: int, owner: str) -> list[float]:
    _require(
        isinstance(value, list)
        and len(value) == length
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        ),
        f"{owner} must contain {length} finite numbers",
    )
    return [float(item) for item in value]


def _habitat_to_ue_cm(value: Sequence[float]) -> list[float]:
    return [100.0 * value[0], 100.0 * value[2], 100.0 * value[1]]


def _add(first: Sequence[float], second: Sequence[float]) -> list[float]:
    return [float(a + b) for a, b in zip(first, second, strict=True)]


def _validate_request(request: Mapping[str, Any]) -> None:
    _require(
        request.get("schema")
        == "avengine_native_strict_two_human_skokloster_room_atom_request_v1",
        "request schema drift",
    )
    _require(request.get("qualification_claim") is False, "qualification forbidden")
    _require(request.get("formal_dataset_count") == 0, "formal count must remain zero")
    _require(request.get("gpu_capture_authorized") is False, "GPU must remain blocked")
    room = request.get("room")
    _require(
        isinstance(room, Mapping)
        and room.get("room_id") == "habitat_test_skokloster_castle"
        and room.get("packaged_map") == PACKAGED_MAP,
        "room identity drift",
    )
    timeline = request.get("timeline")
    _require(
        isinstance(timeline, Mapping)
        and timeline.get("frame_count") == FRAME_COUNT
        and timeline.get("frame_rate_hz") == FPS
        and timeline.get("sample_rate_hz") == SAMPLE_RATE_HZ
        and timeline.get("sample_count") == EPISODE_SAMPLES
        and timeline.get("ticks_per_frame") == TICKS_PER_FRAME
        and timeline.get("ticks_per_sample") == 3
        and timeline.get("sparse_frame_indices") == [15],
        "timeline drift",
    )
    actors = request.get("actors")
    _require(
        isinstance(actors, list)
        and len(actors) == 2
        and [item.get("source_slot_id") for item in actors] == ["source1", "source2"]
        and [item.get("asset_id") for item in actors] == [MALE_ASSET, FEMALE_ASSET]
        and [item.get("role") for item in actors] == ["target", "distractor"],
        "request must bind one exact male target and female distractor",
    )
    audio = request.get("audio")
    _require(
        isinstance(audio, Mapping)
        and audio.get("target_sound_rights_status") == "review_required",
        "speech rights caveat must remain explicit",
    )


def _validate_external_evidence(
    *,
    request: Mapping[str, Any],
    search: Mapping[str, Any],
    rejection: Mapping[str, Any],
    runtime_profile: Mapping[str, Any],
    acoustic_profile: Mapping[str, Any],
    package: Mapping[str, Any],
    simulation: Mapping[str, Any],
    audio_program: Mapping[str, Any],
    audio_binding: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        rejection.get("schema")
        == "avengine_skokloster_strict_near_listener_cpu_rejection_v1"
        and rejection.get("status") == "rejected_cpu_geometry",
        "old near-listener rejection is missing",
    )
    _require(
        search.get("schema") == "avengine_skokloster_strict_listener_search_v1"
        and search.get("status") == "pass_cpu_preflight"
        and search.get("coupled_camera_listener_required") is True
        and search.get("gpu_capture_authorized") is False,
        "listener search evidence is invalid",
    )
    requirements = search.get("requirements")
    _require(
        isinstance(requirements, Mapping)
        and float(requirements.get("source_root_separation_m_observed", -1.0)) >= 1.3,
        "source root separation gate failed",
    )
    selected = search.get("selected")
    _require(
        isinstance(selected, Mapping)
        and selected.get("coupled_camera_listener") is True
        and int(selected.get("nav_island", -1)) == 0
        and float(selected.get("nav_clearance_m", -1.0)) >= 0.5,
        "selected camera/listener nav gate failed",
    )
    distances = _vector(
        selected.get("horizontal_source_distances_m"),
        length=2,
        owner="camera/source distances",
    )
    _require(all(2.2 <= value <= 3.5 for value in distances), "distance gate failed")
    projection = selected.get("projection")
    _require(
        isinstance(projection, Mapping)
        and float(projection.get("minimum_envelope_edge_margin_px", -1.0)) >= 48.0,
        "adult envelope margin gate failed",
    )
    mouths = projection.get("mouth_projections")
    _require(
        isinstance(mouths, list)
        and len(mouths) == 2
        and float(mouths[0]["x_px"]) <= 0.42 * 1280
        and float(mouths[1]["x_px"]) >= 0.58 * 1280,
        "mouth left/right gate failed",
    )
    los = selected.get("camera_to_mouth_line_of_sight")
    _require(
        isinstance(los, list)
        and len(los) == 2
        and all(item.get("clear") is True for item in los),
        "camera-to-mouth visibility failed",
    )
    enclosure = selected.get("enclosure_144")
    _require(
        isinstance(enclosure, Mapping)
        and enclosure.get("ray_count") == 144
        and enclosure.get("hit_ray_count") == 144
        and enclosure.get("escaped_ray_count") == 0
        and enclosure.get("probe_clearance_status") == "pass",
        "144-ray enclosure gate failed",
    )
    _require(
        runtime_profile.get("schema")
        == "avengine_skokloster_imported_room_runtime_profile_v1"
        and runtime_profile.get("status")
        == "packaged_room_object_readback_pass_visual_sparse_pending"
        and runtime_profile.get("visual", {}).get("packaged_runtime_map")
        == PACKAGED_MAP
        and runtime_profile.get("readiness", {}).get("packaged_mesh_readback")
        == "pass",
        "packaged room profile is not readback-closed",
    )
    _require(
        acoustic_profile.get("schema")
        == "avengine_skokloster_acoustic_research_profile_v1"
        and acoustic_profile.get("status") == "acoustic_research_ready"
        and acoustic_profile.get("profile_id")
        == "skokloster_rlr_numeric_cleanup_research_v2",
        "acoustic research profile drift",
    )
    _require(
        package.get("schema") == "avengine_acoustic_scene_package_v1"
        and package.get("package_id") == PACKAGE_ID
        and package.get("package_mode") == "research_candidate"
        and package.get("geometry", {}).get("triangle_count") == 999935,
        "RLR48 package drift",
    )
    _require(
        simulation.get("schema") == "avengine_rir_cache_simulation_request_v1"
        and simulation.get("simulation", {}).get("sample_rate_hz") == SAMPLE_RATE_HZ
        and simulation.get("simulation", {}).get("thread_count") == 1,
        "simulation request drift",
    )
    events = audio_program.get("events")
    _require(
        audio_program.get("schema") == "avengine_m6_audio_program_v1"
        and audio_program.get("mode") == "one_active_of_n"
        and isinstance(events, list)
        and len(events) == 1
        and events[0].get("source_endpoint_id")
        == request["audio"]["source1_endpoint_id"]
        and events[0].get("sound_asset_id") == "speech_cremad_1001_ieo_neu_v1"
        and events[0].get("start_sample") == 7467
        and events[0].get("end_sample_exclusive") == 33093
        and events[0].get("source_start_sample") == 0
        and events[0].get("source_end_sample_exclusive") == 25626
        and events[0].get("linear_gain") == 0.18
        and events[0].get("fade_samples") == 80,
        "canonical target AudioProgram drift",
    )
    _require(
        audio_binding.get("schema")
        == "avengine_native_strict_two_human_audio_binding_v1"
        and audio_binding.get("target_event_count") == 1
        and audio_binding.get("distractor_event_count") == 0
        and audio_binding.get("controlled_content", {}).get("source2") is None,
        "source2 must remain a silent persistent human",
    )
    return {
        "camera_listener_habitat_m": _vector(
            selected.get("camera_listener_habitat_m"),
            length=3,
            owner="coupled camera/listener",
        ),
        "camera_habitat_yaw_deg": float(selected["camera_habitat_yaw_deg"]),
        "listener_orientation_wxyz": _vector(
            selected.get("listener_orientation_wxyz"),
            length=4,
            owner="listener orientation",
        ),
        "listener_floor_habitat_m": _vector(
            selected.get("floor_habitat_m"),
            length=3,
            owner="listener floor",
        ),
        "nav_clearance_m": float(selected["nav_clearance_m"]),
        "source_distances_m": distances,
        "projection": dict(projection),
        "line_of_sight": [dict(item) for item in los],
        "enclosure": {
            "ray_count": enclosure["ray_count"],
            "hit_ray_count": enclosure["hit_ray_count"],
            "escaped_ray_count": enclosure["escaped_ray_count"],
            "probe_clearance_status": enclosure["probe_clearance_status"],
        },
    }


def _actor_declaration(actor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "actor_id": actor["actor_id"],
        "source_slot_id": actor["source_slot_id"],
        "asset_id": actor["asset_id"],
        "asset_revision": actor["asset_revision"],
        "template_id": actor["template_id"],
        "body_plan_id": "biped_human",
        "actor_scale": 1.0,
        "blueprint_class_path": actor["blueprint_class_path"],
        "skeletal_mesh_binding": "blueprint_component",
        "skeletal_mesh_path": actor["skeletal_mesh_path"],
        "skeleton_path": actor["skeleton_path"],
        "idle_animation": actor["idle_animation"],
        "walking_animation": actor["walking_animation"],
        "animation_paths_by_action_id": {
            "idle": actor["idle_animation"],
            "walk": actor["walking_animation"],
        },
        "emitter_anchor_id": "mouth",
        "emitter_offset_m": actor["emitter_offset_m"],
        "habitat_local_anatomical_forward_axis": [0.0, 0.0, 1.0],
        "ue_anatomical_forward_yaw_deg": actor["ue_anatomical_forward_yaw_deg"],
        "floor_contact_gate": False,
        "admission_state": "research",
    }


def _actor_state(
    actor: Mapping[str, Any], camera: Sequence[float], frame_index: int
) -> dict[str, Any]:
    root = _vector(actor["root_habitat_m"], length=3, owner="actor root")
    delta_x = camera[0] - root[0]
    delta_z = camera[2] - root[2]
    distance = math.hypot(delta_x, delta_z)
    _require(distance > 0.0, "actor and camera coincide")
    forward = [delta_x / distance, 0.0, delta_z / distance]
    habitat_yaw = math.degrees(math.atan2(forward[0], forward[2]))
    half = math.radians(habitat_yaw) / 2.0
    desired_ue_yaw = math.degrees(math.atan2(forward[2], forward[0]))
    actor_yaw_ue = desired_ue_yaw - float(actor["ue_anatomical_forward_yaw_deg"])
    return {
        "frame_index": frame_index,
        "actor_id": actor["actor_id"],
        "asset_id": actor["asset_id"],
        "blueprint_class_path": actor["blueprint_class_path"],
        "translation_m": root,
        "translation_ue_cm": _habitat_to_ue_cm(root),
        "rotation_xyzw": [0.0, math.sin(half), 0.0, math.cos(half)],
        "actor_yaw_ue_deg": actor_yaw_ue,
        "anatomical_forward_habitat_world": forward,
        "anatomical_forward_ue_world": [forward[0], forward[2], 0.0],
        "action_id": "idle",
        "action_phase": 0.0,
        "action_time_ticks": frame_index * TICKS_PER_FRAME,
        "ue_animation": actor["idle_animation"],
    }


def _build_documents(
    request: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    episode_id = str(request["episode_id"])
    camera = evidence["camera_listener_habitat_m"]
    yaw = float(evidence["camera_habitat_yaw_deg"])
    half = math.radians(yaw) / 2.0
    world_from_rig = {
        "translation_m": camera,
        "rotation_xyzw": [0.0, math.sin(half), 0.0, math.cos(half)],
    }
    camera_ue = _habitat_to_ue_cm(camera)
    camera_ue_yaw = -90.0 - yaw
    actors = [_actor_declaration(item) for item in request["actors"]]
    frames: list[dict[str, Any]] = []
    rig_frames: list[dict[str, Any]] = []
    for frame_index in range(FRAME_COUNT):
        pts = frame_index * TICKS_PER_FRAME
        camera_state = {
            "frame_index": frame_index,
            "pts_ticks": pts,
            "pose_id": f"{episode_id}__static_camera_listener_v1",
            "habitat_position_m": camera,
            "habitat_yaw_deg": yaw,
            "ue_position_cm": camera_ue,
            "ue_yaw_deg": camera_ue_yaw,
            "world_from_rig": world_from_rig,
        }
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": pts,
                "camera_state": camera_state,
                "actor_states": [
                    _actor_state(actor, camera, frame_index)
                    for actor in request["actors"]
                ],
            }
        )
        rig_frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": pts,
                "pose_id": camera_state["pose_id"],
                "world_from_rig": world_from_rig,
            }
        )

    plan = {
        "schema": "avengine_optional_spear_visual_plan_v1",
        "backend_role": "comparison_visual",
        "actors": actors,
        "frames": frames,
        "camera": {
            "dynamic": False,
            "listener_id": "listener0",
            "camera_listener_coupling": "rigid_colocated_cooriented",
            "habitat_position_m": camera,
            "habitat_yaw_deg": yaw,
            "ue_position_cm": camera_ue,
            "ue_yaw_deg": camera_ue_yaw,
            "horizontal_fov_deg": 105.0,
            "sensor_rig_trajectory_id": f"{episode_id}__sensor_rig",
        },
        "coordinate_contract": {
            "source_to_habitat": "H=(S.x,S.z,-S.y)",
            "habitat_to_ue_position": "U_cm=(100*H.x,100*H.z,100*H.y)",
            "camera_yaw": "UE_yaw_deg=-90-Habitat_yaw_deg",
        },
        "room": {
            "room_id": request["room"]["room_id"],
            "scene_id": request["room"]["scene_id"],
            "runtime_map": request["room"]["packaged_map"],
            "packaged_executable": request["room"]["packaged_executable"],
            "saved_surface_actor_tag": "avengine_skokloster_castle_surface",
        },
        "source_logic": {
            "scenario_id": episode_id,
            "target_source_slot_id": "source1",
            "distractor_source_slot_id": "source2",
            "sources": [
                {
                    "source_slot_id": "source1",
                    "entity_actor_id": "source1_actor",
                    "source_endpoint_id": request["audio"]["source1_endpoint_id"],
                    "sound_class": "human_speech",
                    "activation": "active_during_declared_event",
                },
                {
                    "source_slot_id": "source2",
                    "entity_actor_id": "source2_actor",
                    "source_endpoint_id": request["audio"]["source2_endpoint_id"],
                    "sound_class": "silent_human",
                    "activation": "persistent_silent_endpoint",
                },
            ],
        },
        "render": {
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": FPS,
            "ticks_per_frame": TICKS_PER_FRAME,
        },
        "qualification": {
            "status": "cpu_geometry_pass_fresh_spear_pixels_pending",
            "cpu_body_envelope_is_live_bbox_evidence": False,
            "qualification_claim": False,
            "formal_dataset_count": 0,
        },
    }
    scenario = {
        "schema": "avengine_optional_spear_skokloster_scenario_v1",
        "scenario_id": episode_id,
        "scenario_directory": episode_id,
        "variant_id": "skokloster_strict_two_human_static_v1",
        "backend_role": "comparison_visual",
        "native_scene": {
            "map": request["room"]["packaged_map"],
            "layout": "saved_packaged_room_actor_unchanged",
            "lighting": "packaged_map_unchanged",
        },
        "render": {
            "frame_count": FRAME_COUNT,
            "frame_rate_hz": FPS,
            "width": 1280,
            "height": 720,
            "horizontal_fov_deg": 105.0,
            "streaming_warmup_frames": 120,
            "camera_warmup_frames": 40,
        },
        "plan": plan,
        "authoritative_inputs": {
            "audio_program": request["audio"]["canonical_audio_program"],
            "source_endpoint_registry": request["audio"]["source_endpoint_registry"],
            "sound_asset_registry": request["audio"]["sound_asset_registry"],
        },
        "authoritative_capture_request": {
            "request_id": f"{episode_id}__native_capture",
            "episode_id": episode_id,
            "scenario_type": "strict_two_human_static_skokloster_research_probe",
            "target_source_slot_id": "source1",
            "fact_status": "pending_fresh_native_capture",
        },
    }
    suite = {
        "schema": "avengine_optional_spear_skokloster_suite_v1",
        "backend_role": "comparison_visual",
        "native_map": request["room"]["packaged_map"],
        "packaged_executable": request["room"]["packaged_executable"],
        "scenarios": [scenario],
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    rig = {
        "schema": "avengine_sensor_rig_trajectory_v1",
        "trajectory_id": f"{episode_id}__sensor_rig",
        "formal_view_id": "view0",
        "camera_listener_coupling": "rigid_colocated_cooriented",
        "coordinate_frame": "avengine_world_right_handed_y_up_m",
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FPS,
        "duration_ticks": FRAME_COUNT * TICKS_PER_FRAME,
        "frames": rig_frames,
    }

    roots = {
        actor["source_slot_id"]: _vector(
            actor["root_habitat_m"], length=3, owner="actor root"
        )
        for actor in request["actors"]
    }
    centers = {
        actor["source_slot_id"]: _add(
            roots[actor["source_slot_id"]], actor["emitter_offset_m"]
        )
        for actor in request["actors"]
    }
    trajectory = {
        "schema": "avengine_room_trajectory_bank_v2",
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FPS,
        "seconds_per_episode": 5.0,
        "episode_count": 1,
        "source_slots": ["source1", "source2"],
        "motion_case_counts": {
            "static_static": 1,
            "source1_moving_source2_static": 0,
            "source1_static_source2_moving": 0,
            "both_moving": 0,
        },
        "claim_boundary": "profile mouth centers; fresh native mouth readback pending",
        "episodes": [
            {
                "episode_id": episode_id,
                "motion_case": "strict_two_human_static_skokloster",
                "source_root_paths_m": {
                    slot: [position] * FRAME_COUNT for slot, position in roots.items()
                },
                "source_center_paths_m": {
                    slot: [position] * FRAME_COUNT for slot, position in centers.items()
                },
                "statistics": {
                    "target_source_slot_id": "source1",
                    "distractor_source_slot_id": "source2",
                    "native_recapture_required": True,
                },
            }
        ],
    }
    binding_report = {
        "schema": "avengine_asset_emitter_scenario_report_v1",
        "status": "pass",
        "method": "runtime_profile_root_plus_declared_mouth_offset",
        "profile_geometry_status": "pass",
        "native_readback_status": "pending_required",
        "claim_boundary": "profile-coordinate plan only; fresh f15 readback required",
        "scenario_count": 1,
        "scenarios": [
            {
                "trajectory_episode_id": episode_id,
                "output_episode_id": episode_id,
                "binding_report": {
                    "schema": "avengine_asset_emitter_binding_report_v1",
                    "status": "pass",
                    "episode_count": 1,
                    "listener_position_m": camera,
                    "target_world_emitter_at_sparse_frame_m": centers["source1"],
                    "native_readback_status": "pending_required",
                    "qualification_claim": False,
                    "bindings": [
                        {
                            "source_slot_id": actor["source_slot_id"],
                            "asset_id": actor["asset_id"],
                            "asset_revision": actor["asset_revision"],
                            "semantic_anchor_id": "mouth",
                            "emitter_offset_m": actor["emitter_offset_m"],
                            "offset_space": "final_scaled_asset_root",
                            "native_readback": "pending_required",
                        }
                        for actor in request["actors"]
                    ],
                },
            }
        ],
        "qualification_claim": False,
    }

    uses = {
        slot: [
            {
                "episode_id": episode_id,
                "source_slot_id": slot,
                "frame_index": frame_index,
            }
            for frame_index in range(FRAME_COUNT)
        ]
        for slot in ("source1", "source2")
    }
    rir_plan = {
        "schema": "avengine_room_rir_job_plan_v2",
        "status": "planned_not_run",
        "producer_backend": "RLR Audio Propagation",
        "source_acoustic_profile": "omnidirectional_point_source_v1",
        "listener_position_m": camera,
        "listener_orientation_wxyz": evidence["listener_orientation_wxyz"],
        "layout": "binaural",
        "requested_pair_state_count": 150,
        "unique_listener_pose_count": 1,
        "unique_rir_job_count": 2,
        "cache_reuse_count": 148,
        "jobs": [
            {
                "job_id": f"skokloster_{slot}_static_v1",
                "source_position_m": centers[slot],
                "uses": uses[slot],
            }
            for slot in ("source1", "source2")
        ],
        "claim_boundary": "two exact CPU RIR jobs planned but not run",
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    audio_plan = {
        "schema": "avengine_skokloster_strict_audio_program_binding_v1",
        "status": "validated_canonical_program_pending_exact_rir_render",
        "canonical_audio_program": request["audio"]["canonical_audio_program"],
        "canonical_audio_binding": request["audio"]["canonical_audio_binding"],
        "timeline": {
            "frame_count": FRAME_COUNT,
            "video_fps": FPS,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sample_count": EPISODE_SAMPLES,
            "ticks_per_frame": TICKS_PER_FRAME,
            "ticks_per_sample": 3,
        },
        "source1": {
            "role": "target",
            "sound_class": "human_speech",
            "source_endpoint_id": request["audio"]["source1_endpoint_id"],
            "sound_asset_id": "speech_cremad_1001_ieo_neu_v1",
            "start_sample": 7467,
            "end_sample_exclusive": 33093,
            "source_start_sample": 0,
            "source_end_sample_exclusive": 25626,
            "linear_gain": 0.18,
            "fade_samples": 80,
            "rights_status": request["audio"]["target_sound_rights_status"],
        },
        "source2": {
            "role": "distractor",
            "sound_class": "silent_human",
            "source_endpoint_id": request["audio"]["source2_endpoint_id"],
            "event_count": 0,
            "persistent_when_silent": True,
        },
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    return {
        "suite_execution_plan.json": suite,
        "sensor_rig_trajectory.json": rig,
        "trajectory_bank.json": trajectory,
        "asset_emitter_binding_report.json": binding_report,
        "rir_job_plan.json": rir_plan,
        "audio_program_binding.json": audio_plan,
    }


def _execution_plan(request: Mapping[str, Any], output: Path) -> dict[str, Any]:
    execution = request["execution"]
    repository = Path(execution["repository"])
    output_root = Path(execution["output_root"])
    rir_cache = output_root / "exact_rir_cache_v1"
    binaural = output_root / "binaural_v1"
    common_capture = [
        execution["python"],
        execution["capture_runner"],
        "--suite-plan",
        str(output / "suite_execution_plan.json"),
        "--scenario-id",
        request["episode_id"],
        "--audio-wav",
        str(binaural / "audio/binaural" / f"{request['episode_id']}__v00.wav"),
        "--spear-root",
        execution["spear_root"],
        "--spear-executable",
        request["room"]["packaged_executable"],
        "--output",
    ]
    return {
        "schema": "avengine_skokloster_strict_two_human_execution_plan_v1",
        "status": "cpu_ready_gpu_blocked",
        "attempt_id": output.name,
        "supersedes": [
            {
                "attempt_id": "cpu_preflight_v1",
                "status": "rejected_before_rir_execution",
                "reason": "noncanonical listener pose mode failed the real RIR validator",
            }
        ],
        "generated_preflight_root": str(output.resolve()),
        "runtime_output_root": str(output_root),
        "cpu_steps": [
            {
                "step_id": "render_two_exact_binaural_rirs",
                "status": "planned_not_run",
                "working_directory": str(repository),
                "environment": {
                    "AVENGINE_HABITAT_RUNTIME_ROOT": (
                        "/data/jzy/code/habitat-sim-AVEngine"
                    ),
                    "AVENGINE_SKOKLOSTER_RLR48_PACKAGE_ROOT": str(
                        Path(request["room"]["acoustic_package_manifest"]).parent
                    ),
                },
                "argv": [
                    execution["python"],
                    str(repository / "tools/m6x/render_rir_cache.py"),
                    "--rir-job-plan",
                    str(output / "rir_job_plan.json"),
                    "--acoustic-package-manifest",
                    request["room"]["acoustic_package_manifest"],
                    "--simulation-request",
                    request["room"]["simulation_request"],
                    "--hrtf",
                    execution["hrtf"],
                    "--output",
                    str(rir_cache),
                    "--layout",
                    "binaural",
                    "--batch-size",
                    "2",
                    "--thread-count",
                    str(execution["rir_thread_count"]),
                ],
                "expected": {
                    "compute_device": "CPU",
                    "selected_job_count": 2,
                    "full_plan_complete": True,
                    "layout": "binaural",
                },
            },
            {
                "step_id": "render_target_speech_silent_distractor_binaural",
                "status": "blocked_until_exact_rir_pass",
                "working_directory": str(repository),
                "argv": [
                    execution["python"],
                    str(repository / "tools/m7/render_asset_bound_binaural_batch.py"),
                    "--plan-root",
                    str(output),
                    "--rir-cache",
                    str(rir_cache),
                    "--audio-program",
                    request["audio"]["canonical_audio_program"],
                    "--source-endpoint-registry",
                    request["audio"]["source_endpoint_registry"],
                    "--sound-asset-registry",
                    request["audio"]["sound_asset_registry"],
                    "--source-endpoint-slot",
                    f"{request['audio']['source1_endpoint_id']}=source1",
                    "--source-endpoint-slot",
                    f"{request['audio']['source2_endpoint_id']}=source2",
                    "--sound-audio",
                    (
                        "speech_cremad_1001_ieo_neu_v1="
                        + request["audio"]["target_sound_path"]
                    ),
                    "--retain-stems",
                    "--output",
                    str(binaural),
                ],
                "expected": {
                    "target_event_count": 1,
                    "distractor_event_count": 0,
                    "sample_count": EPISODE_SAMPLES,
                    "channel_count": 2,
                },
            },
        ],
        "gpu_steps": [
            {
                "step_id": "fresh_sparse_f15",
                "status": "blocked_pending_explicit_gpu_authorization",
                "argv": common_capture
                + [
                    str(output_root / "native_sparse_f15_v1"),
                    "--rpc-port",
                    str(execution["rpc_port"]),
                    "--graphics-adapter",
                    str(execution["graphics_adapter"]),
                    "--frame-index",
                    "15",
                ],
                "required_live_gates": [
                    "target visible fraction >=0.8",
                    "distractor visible fraction >=0.5",
                    "each visible pixel count >=5000",
                    "bbox edge margin >=1px",
                    "normal RGB and metric depth",
                    "source1/source2 target-only metric depth from shared camera",
                ],
            },
            {
                "step_id": "full75_episode",
                "status": "blocked_until_f15_pixel_gate_pass",
                "argv": common_capture
                + [
                    str(output_root / "native_full75_v1"),
                    "--rpc-port",
                    str(execution["rpc_port"]),
                    "--graphics-adapter",
                    str(execution["graphics_adapter"]),
                ],
            },
        ],
        "gpu_capture_authorized": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def _preflight(
    request: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    gates = {
        "old_near_listener_rejected": "pass",
        "camera_listener_coupled": "pass",
        "single_nav_island": "pass",
        "listener_clearance_at_least_0_5m": "pass",
        "source_separation_at_least_1_3m": "pass",
        "camera_source_distance_2_2_to_3_5m": "pass",
        "adult_cylinder_envelope_margin_at_least_48px": "pass",
        "mouth_left_right_safe": "pass",
        "camera_to_both_mouths_clear": "pass",
        "enclosure_144_of_144": "pass",
        "packaged_room_object_readback": "pass",
        "rlr48_acoustic_research_package": "pass",
        "exact_two_rir_jobs": "planned_not_run",
        "target_audio_program_source2_silent": "pass",
        "target_sound_rights": request["audio"]["target_sound_rights_status"],
        "fresh_spear_pixel_bbox": "pending_required",
        "full75": "blocked_until_sparse_pixel_gate",
    }
    return {
        "schema": "avengine_skokloster_strict_two_human_cpu_preflight_v1",
        "status": "cpu_plan_pass_gpu_sparse_pending",
        "attempt_id": "cpu_preflight_v2",
        "supersedes": [
            {
                "attempt_id": "cpu_preflight_v1",
                "status": "rejected_before_rir_execution",
                "reason": "noncanonical listener pose mode failed the real RIR validator",
            }
        ],
        "episode_id": request["episode_id"],
        "camera_listener_habitat_m": evidence["camera_listener_habitat_m"],
        "listener_floor_habitat_m": evidence["listener_floor_habitat_m"],
        "camera_habitat_yaw_deg": evidence["camera_habitat_yaw_deg"],
        "listener_orientation_wxyz": evidence["listener_orientation_wxyz"],
        "nav_clearance_m": evidence["nav_clearance_m"],
        "source_distances_m": evidence["source_distances_m"],
        "cpu_projection": evidence["projection"],
        "cpu_projection_semantics": (
            "conservative root-cylinder geometry only; not live skeletal pixels or bbox"
        ),
        "line_of_sight": evidence["line_of_sight"],
        "enclosure": evidence["enclosure"],
        "gates": gates,
        "strict_pixel_thresholds": request["strict_pixel_gates"],
        "gpu_capture_authorized": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def build(args: argparse.Namespace) -> Path:
    request = _load(args.request.resolve())
    _validate_request(request)
    paths = {
        "search": args.listener_search
        or Path(request["room"]["listener_search_evidence"]),
        "rejection": args.near_rejection
        or Path(request["room"]["near_listener_rejection_evidence"]),
        "runtime": args.runtime_profile or Path(request["room"]["runtime_profile"]),
        "acoustic": args.acoustic_profile or Path(request["room"]["acoustic_profile"]),
        "package": args.package_manifest
        or Path(request["room"]["acoustic_package_manifest"]),
        "simulation": args.simulation_request
        or Path(request["room"]["simulation_request"]),
        "audio_program": args.audio_program
        or Path(request["audio"]["canonical_audio_program"]),
        "audio_binding": args.audio_binding
        or Path(request["audio"]["canonical_audio_binding"]),
    }
    loaded = {name: _load(path.resolve()) for name, path in paths.items()}
    evidence = _validate_external_evidence(
        request=request,
        search=loaded["search"],
        rejection=loaded["rejection"],
        runtime_profile=loaded["runtime"],
        acoustic_profile=loaded["acoustic"],
        package=loaded["package"],
        simulation=loaded["simulation"],
        audio_program=loaded["audio_program"],
        audio_binding=loaded["audio_binding"],
    )
    external_paths = [
        Path(request["room"]["packaged_executable"]),
        Path(request["execution"]["hrtf"]),
        Path(request["audio"]["target_sound_path"]),
        Path(request["audio"]["source_endpoint_registry"]),
        Path(request["audio"]["sound_asset_registry"]),
    ]
    _require(
        all(path.is_file() for path in external_paths), "external runtime input missing"
    )
    output = args.output.resolve()
    _require(not output.exists() and not output.is_symlink(), "output already exists")
    output.mkdir(parents=True)
    documents = _build_documents(request, evidence)
    documents["execution_plan.json"] = _execution_plan(request, output)
    documents["preflight.json"] = _preflight(request, evidence)
    for name, value in documents.items():
        _write(output / name, value)
    print(
        "SKOKLOSTER_STRICT_TWO_HUMAN_CPU_PREFLIGHT_OK "
        f"frames={FRAME_COUNT} rirs=2 output={output}",
        flush=True,
    )
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--listener-search", type=Path)
    parser.add_argument("--near-rejection", type=Path)
    parser.add_argument("--runtime-profile", type=Path)
    parser.add_argument("--acoustic-profile", type=Path)
    parser.add_argument("--package-manifest", type=Path)
    parser.add_argument("--simulation-request", type=Path)
    parser.add_argument("--audio-program", type=Path)
    parser.add_argument("--audio-binding", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

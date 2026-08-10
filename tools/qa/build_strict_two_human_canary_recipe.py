#!/usr/bin/env python3
"""Build one exact static two-human Apartment recipe and AudioProgram."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import write_json
from avengine.m6.audio_program import (
    bind_audio_program_hash,
    validate_audio_program,
)
from avengine.m6.registry import bind_content_hash
from avengine.m6.sources import (
    validate_sound_asset_registry,
    validate_source_endpoint_registry,
)
from avengine.optional_backends.spear_visual import (
    actor_ue_yaw_degrees,
    habitat_point_to_apartment_ue_cm,
)
from avengine.sensor_rig_trajectory import (
    materialize_sensor_rig_trajectory,
)

AUDIO_BUILDER_PATH = REPOSITORY / "tools/qa/build_native_controlled_audio_program.py"
AUDIO_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "native_controlled_audio_builder", AUDIO_BUILDER_PATH
)
if AUDIO_BUILDER_SPEC is None or AUDIO_BUILDER_SPEC.loader is None:
    raise RuntimeError(f"cannot import {AUDIO_BUILDER_PATH}")
AUDIO_BUILDER = importlib.util.module_from_spec(AUDIO_BUILDER_SPEC)
AUDIO_BUILDER_SPEC.loader.exec_module(AUDIO_BUILDER)

SCHEMA = "avengine_native_strict_two_human_canary_recipe_v1"
FRAME_COUNT = 75
FRAME_RATE_HZ = 15
TICKS_PER_FRAME = 3200
SAMPLE_RATE_HZ = 16000
SAMPLE_COUNT = 80000
TARGET_SPEECH_START_SAMPLE = 7467
TARGET_SPEECH_FIRST_FRAME = 7
TARGET_SPEECH_LAST_FRAME = 31
EPISODE_ID = "rocketbox_male_female__strict_two_human_canary_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def target_speech_sample_window(source_sample_count: int) -> tuple[int, int]:
    """Return the exact interval that intersects every declared speech frame."""

    _require(source_sample_count > 0, "target speech must contain samples")
    start_sample = TARGET_SPEECH_START_SAMPLE
    end_sample = start_sample + source_sample_count
    _require(end_sample <= SAMPLE_COUNT, "target speech does not fit timeline")
    first_frame = start_sample * FRAME_RATE_HZ // SAMPLE_RATE_HZ
    last_frame = (end_sample - 1) * FRAME_RATE_HZ // SAMPLE_RATE_HZ
    _require(
        first_frame == TARGET_SPEECH_FIRST_FRAME,
        "target speech first-frame contract drift",
    )
    _require(
        last_frame == TARGET_SPEECH_LAST_FRAME,
        "target speech last-frame contract drift",
    )
    return start_sample, end_sample


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset(registry: Mapping[str, Any], asset_id: str) -> Mapping[str, Any]:
    matches = [
        item for item in registry.get("assets", []) if item.get("asset_id") == asset_id
    ]
    _require(len(matches) == 1, f"runtime asset does not resolve once: {asset_id}")
    return matches[0]


def _profile_actor(actor_id: str, profile: Mapping[str, Any]) -> dict[str, Any]:
    timeline = profile["timeline"]
    unreal = profile["runtime_backends"]["spear_unreal"]
    return {
        "actor_id": actor_id,
        "asset_id": profile["asset_id"],
        "blueprint_class_path": unreal["blueprint_class_path"],
        "body_plan_id": timeline["body_plan_id"],
        "habitat_local_anatomical_forward_axis": timeline[
            "local_anatomical_forward_axis"
        ],
        "idle_animation": unreal["idle_animation"],
        "template_id": timeline["template_id"],
        "ue_anatomical_forward_yaw_deg": unreal[
            "ue_anatomical_forward_yaw_deg"
        ],
        "ue_component_frame_delta": deepcopy(unreal["ue_component_frame_delta"]),
        "walking_animation": unreal["walking_animation"],
    }


def _facing_camera_rotation(
    root_m: Sequence[float], camera_m: Sequence[float]
) -> tuple[list[float], list[float]]:
    dx = float(camera_m[0]) - float(root_m[0])
    dz = float(camera_m[2]) - float(root_m[2])
    norm = math.hypot(dx, dz)
    _require(norm > 1.0e-6, "actor root is colocated with the camera")
    forward = [dx / norm, 0.0, dz / norm]
    yaw = math.atan2(forward[0], forward[2])
    rotation = [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]
    return rotation, forward


def _screen_offset_fraction(
    point_m: Sequence[float],
    *,
    camera_m: Sequence[float],
    camera_forward: Sequence[float],
    horizontal_fov_deg: float,
) -> float:
    forward_xz = [float(camera_forward[0]), float(camera_forward[2])]
    norm = math.hypot(*forward_xz)
    _require(norm > 1.0e-6, "camera forward is invalid")
    forward_xz = [value / norm for value in forward_xz]
    right_xz = [-forward_xz[1], forward_xz[0]]
    delta = [
        float(point_m[0]) - float(camera_m[0]),
        float(point_m[2]) - float(camera_m[2]),
    ]
    depth = delta[0] * forward_xz[0] + delta[1] * forward_xz[1]
    _require(depth > 0.0, "planned actor is behind the camera")
    lateral = delta[0] * right_xz[0] + delta[1] * right_xz[1]
    return lateral / (depth * math.tan(math.radians(horizontal_fov_deg) / 2.0))


def build_static_actor_bundle(
    plan: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, Any]:
    actors_by_role = {item["role"]: item for item in plan["actors"]}
    _require(set(actors_by_role) == {"target", "distractor"}, "role closure failed")
    actor_specs = {
        "source1_actor": actors_by_role["target"],
        "source2_actor": actors_by_role["distractor"],
    }
    profiles = {
        actor_id: _asset(registry, spec["runtime_asset_id"])
        for actor_id, spec in actor_specs.items()
    }
    declarations = [
        _profile_actor(actor_id, profiles[actor_id])
        for actor_id in ("source1_actor", "source2_actor")
    ]
    declarations_by_id = {item["actor_id"]: item for item in declarations}
    camera_m = plan["camera_pose"]["translation_m"]
    camera_forward = [-math.sin(math.radians(55.0)), 0.0, -math.cos(math.radians(55.0))]
    states: list[dict[str, Any]] = []
    roots: dict[str, list[float]] = {}
    emitters: dict[str, list[float]] = {}
    projection: dict[str, float] = {}
    for actor_id in ("source1_actor", "source2_actor"):
        spec = actor_specs[actor_id]
        profile = profiles[actor_id]
        declaration = declarations_by_id[actor_id]
        slot = spec["source_slot_id"]
        root = [float(value) for value in spec["root_translation_m"]]
        rotation, forward_h = _facing_camera_rotation(root, camera_m)
        actor_yaw = actor_ue_yaw_degrees(
            rotation,
            declaration["habitat_local_anatomical_forward_axis"],
            declaration["ue_anatomical_forward_yaw_deg"],
        )
        forward_ue = [forward_h[0], forward_h[2], 0.0]
        anchor = next(
            item
            for item in profile["emitter_anchors"]
            if item["anchor_id"] == profile["default_emitter_anchor_id"]
        )
        offset = anchor["offset_m"]
        _require(
            offset[0] == 0.0 and offset[2] == 0.0,
            f"{slot} canary requires a root-vertical mouth offset",
        )
        emitter = [root[0], root[1] + float(offset[1]), root[2]]
        offset_fraction = _screen_offset_fraction(
            emitter,
            camera_m=camera_m,
            camera_forward=camera_forward,
            horizontal_fov_deg=105.0,
        )
        expected_sign = 1.0 if spec["expected_screen_side"] == "right" else -1.0
        _require(
            expected_sign * offset_fraction > 0.02,
            f"{slot} does not clear its planned screen-side dead zone",
        )
        roots[slot] = root
        emitters[slot] = emitter
        projection[slot] = offset_fraction
        states.append(
            {
                "action_id": "idle",
                "action_phase": 0.0,
                "action_time_ticks": 0,
                "actor_id": actor_id,
                "actor_yaw_ue_deg": actor_yaw,
                "anatomical_forward_habitat_world": forward_h,
                "anatomical_forward_ue_world": forward_ue,
                "asset_id": profile["asset_id"],
                "blueprint_class_path": declaration["blueprint_class_path"],
                "rotation_xyzw": rotation,
                "translation_m": root,
                "translation_ue_cm": list(habitat_point_to_apartment_ue_cm(root)),
                "ue_animation": declaration["idle_animation"],
            }
        )
    return {
        "declarations": declarations,
        "state_templates": states,
        "roots": roots,
        "emitters": emitters,
        "projection_offset_fraction": projection,
        "camera_forward_world": camera_forward,
    }


def _sound_record(
    controlled_record: Mapping[str, Any], *, rights_evidence_sha256: str
) -> dict[str, Any]:
    content = controlled_record["content"]
    audio = controlled_record["audio"]
    return {
        "sound_asset_id": controlled_record["sound_asset_id"],
        "revision": "v1",
        "semantic_sound_class": "human_speech",
        "taxonomy_path": ["human", "voice", "speech"],
        "instance_lineage_id": content["speaker_id"],
        "dry_audio": {
            "uri": audio["uri"],
            "sha256": audio["sha256"],
            "sample_rate_hz": audio["sample_rate_hz"],
            "channel_count": audio["channel_count"],
            "sample_count": audio["sample_count"],
        },
        "normalization_policy": {"mode": "preserve", "target_dbfs": None},
        "allowed_transforms": ["crop", "gain", "zero_pad"],
        "permitted_event_usage": ["one_active_of_n"],
        "tags": sorted(set(content["content_tags"] + [content["species"]])),
        "provenance": {
            "origin": "lead_b_controlled_sound_content_registry_v1",
            "license": None,
            "rights_status": "review_required",
            "rights_evidence_sha256": rights_evidence_sha256,
        },
        "admissibility": "research",
    }


def build_audio_contracts(
    *,
    plan: Mapping[str, Any],
    registry: Mapping[str, Any],
    registry_path: Path,
    controlled_registry: Mapping[str, Any],
    controlled_registry_path: Path,
) -> dict[str, Any]:
    actors = {item["role"]: item for item in plan["actors"]}
    target = actors["target"]
    distractor = actors["distractor"]
    target_profile = _asset(registry, target["runtime_asset_id"])
    distractor_profile = _asset(registry, distractor["runtime_asset_id"])
    runtime_sha = _sha256(registry_path)
    endpoints = bind_content_hash(
        {
            "schema": "avengine_m6_source_endpoint_registry_v1",
            "registry_id": "lead_d_strict_two_human_endpoints_v1",
            "revision": "v1",
            "source_endpoints": [
                {
                    "source_endpoint_id": "lead_d_source1_mouth",
                    "revision": "v1",
                    "binding": {
                        "kind": "entity_anchor",
                        "entity_instance_id": "source1",
                        "entity_asset_id": target_profile["asset_id"],
                        "entity_asset_revision": target_profile["revision"],
                        "emitter_anchor_id": "mouth",
                    },
                    "source_visibility_mode": "visible_entity",
                    "allowed_sound_class_ids": ["human_speech"],
                    "directivity_profile_id": "point_emitter_v1",
                    "persistent_when_silent": True,
                    "admission_state": "research",
                    "evidence_sha256": runtime_sha,
                },
                {
                    "source_endpoint_id": "lead_d_source2_mouth",
                    "revision": "v1",
                    "binding": {
                        "kind": "entity_anchor",
                        "entity_instance_id": "source2",
                        "entity_asset_id": distractor_profile["asset_id"],
                        "entity_asset_revision": distractor_profile["revision"],
                        "emitter_anchor_id": "mouth",
                    },
                    "source_visibility_mode": "visible_entity",
                    "allowed_sound_class_ids": ["human_speech"],
                    "directivity_profile_id": "point_emitter_v1",
                    "persistent_when_silent": True,
                    "admission_state": "research",
                    "evidence_sha256": runtime_sha,
                },
            ],
        }
    )
    speech = AUDIO_BUILDER._controlled_asset(
        controlled_registry, target["sound_asset_id"]
    )
    media_path = (
        controlled_registry_path.parent / "media" / f"{target['sound_asset_id']}.wav"
    )
    AUDIO_BUILDER._validate_wave(media_path, speech["audio"])
    sounds = bind_content_hash(
        {
            "schema": "avengine_m6_sound_asset_registry_v1",
            "registry_id": "lead_d_strict_two_human_sounds_v1",
            "revision": "v1",
            "sound_assets": [
                _sound_record(
                    speech, rights_evidence_sha256=_sha256(controlled_registry_path)
                )
            ],
        }
    )
    start_sample, end_sample = target_speech_sample_window(
        int(speech["audio"]["sample_count"])
    )
    program = bind_audio_program_hash(
        {
            "schema": "avengine_m6_audio_program_v1",
            "program_id": "lead_d_strict_two_human_canary_audio_v1",
            "revision": "v1",
            "mode": "one_active_of_n",
            "timeline": {
                "time_base_hz": 48000,
                "ticks_per_frame": TICKS_PER_FRAME,
                "video_fps": FRAME_RATE_HZ,
                "frame_count": FRAME_COUNT,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "ticks_per_sample": 3,
                "sample_count": SAMPLE_COUNT,
            },
            "candidate_source_endpoint_ids": [
                "lead_d_source1_mouth",
                "lead_d_source2_mouth",
            ],
            "events": [
                AUDIO_BUILDER._event(
                    event_id="source1_speech_000",
                    endpoint_id="lead_d_source1_mouth",
                    sound_id=target["sound_asset_id"],
                    start_sample=start_sample,
                    end_sample=end_sample,
                    source_start=0,
                    gain=0.18,
                )
            ],
            "source_specific_stems": True,
            "admission_state": "research",
        }
    )
    endpoint_errors = validate_source_endpoint_registry(endpoints)
    sound_errors = validate_sound_asset_registry(sounds)
    program_errors = validate_audio_program(
        program,
        source_endpoint_registry=endpoints,
        sound_asset_registry=sounds,
    )
    _require(not endpoint_errors, "; ".join(endpoint_errors))
    _require(not sound_errors, "; ".join(sound_errors))
    _require(not program_errors, "; ".join(program_errors))
    return {
        "source_endpoint_registry": endpoints,
        "sound_asset_registry": sounds,
        "audio_program": program,
        "media_path": media_path,
        "controlled_content": {
            "source1": {
                "sound_asset_id": target["sound_asset_id"],
                "statement_id": speech["content"]["statement_id"],
                "transcript": speech["content"]["transcript"],
                "language": speech["content"]["language"],
            },
            "source2": None,
        },
    }


def build_acoustic_binding_report(
    *,
    plan: Mapping[str, Any],
    registry: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind declared profile mouth offsets without claiming native eligibility."""

    return {
        "schema": "avengine_asset_emitter_scenario_report_v1",
        "status": "pass",
        "qualification_claim": False,
        "claim_boundary": (
            "profile-coordinate acoustic plan only; native frame-15 actor-root "
            "readback remains required before scene admission"
        ),
        "profile_geometry_status": "pass",
        "native_readback_status": "pending_required",
        "method": "runtime_profile_root_plus_declared_mouth_offset",
        "scenario_count": 1,
        "scenarios": [
            {
                "trajectory_episode_id": EPISODE_ID,
                "output_episode_id": EPISODE_ID,
                "binding_report": {
                    "schema": "avengine_asset_emitter_binding_report_v1",
                    "status": "pass",
                    "qualification_claim": False,
                    "native_readback_status": "pending_required",
                    "episode_count": 1,
                    "listener_position_m": plan["camera_pose"]["translation_m"],
                    "bindings": [
                        {
                            "source_slot_id": actor["source_slot_id"],
                            "asset_id": actor["runtime_asset_id"],
                            "asset_revision": actor["runtime_revision"],
                            "semantic_anchor_id": "mouth",
                            "emitter_offset_m": next(
                                item["offset_m"]
                                for item in _asset(
                                    registry, actor["runtime_asset_id"]
                                )["emitter_anchors"]
                                if item["anchor_id"] == "mouth"
                            ),
                            "offset_space": "final_scaled_asset_root",
                            "native_readback": "pending_required",
                        }
                        for actor in plan["actors"]
                    ],
                    "target_world_emitter_at_sparse_frame_m": bundle["emitters"][
                        "source1"
                    ],
                },
            }
        ],
    }


def build(
    *,
    plan_path: Path,
    cpu_preflight_path: Path,
    runtime_registry_path: Path,
    source_suite_path: Path,
    controlled_registry_path: Path,
    output: Path,
) -> dict[str, Path]:
    _require(not output.exists(), f"refusing to overwrite output: {output}")
    plan = _load(plan_path)
    cpu_preflight = _load(cpu_preflight_path)
    registry = _load(runtime_registry_path)
    source_suite = _load(source_suite_path)
    controlled_registry = _load(controlled_registry_path)
    _require(cpu_preflight.get("status") == "pass", "CPU preflight did not pass")
    _require(
        cpu_preflight.get("next_state") == "ready_for_exact_two_human_rir_plan",
        "CPU preflight state drift",
    )
    _require(len(source_suite.get("scenarios", [])) == 1, "template suite drift")
    bundle = build_static_actor_bundle(plan, registry)
    sensor_rig = materialize_sensor_rig_trajectory(
        trajectory_id=f"{EPISODE_ID}__sensor_rig",
        program={
            "kind": "HOLD",
            "position_m": plan["camera_pose"]["translation_m"],
            "yaw_deg": 55.0,
        },
    )
    _require(len(sensor_rig["frames"]) == FRAME_COUNT, "sensor rig frame drift")
    template_scenario = source_suite["scenarios"][0]
    template_frames = template_scenario["plan"]["frames"]
    _require(len(template_frames) == FRAME_COUNT, "template suite frame drift")
    frames = []
    for frame_index in range(FRAME_COUNT):
        camera_state = deepcopy(template_frames[frame_index]["camera_state"])
        camera_state["frame_index"] = frame_index
        camera_state["pts_ticks"] = frame_index * TICKS_PER_FRAME
        camera_state["pose_hash"] = sensor_rig["frames"][frame_index]["pose_hash"]
        camera_state["world_from_rig"] = deepcopy(
            sensor_rig["frames"][frame_index]["world_from_rig"]
        )
        actor_states = []
        for state_template in bundle["state_templates"]:
            state = deepcopy(state_template)
            state["action_time_ticks"] = frame_index * TICKS_PER_FRAME
            actor_states.append(state)
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * TICKS_PER_FRAME,
                "camera_state": camera_state,
                "actor_states": actor_states,
            }
        )

    scenario = deepcopy(template_scenario)
    scenario["scenario_id"] = EPISODE_ID
    scenario["scenario_directory"] = EPISODE_ID
    scenario["variant_id"] = "strict_two_human_static_canary"
    scenario["plan"]["actors"] = bundle["declarations"]
    scenario["plan"]["frames"] = frames
    scenario["plan"]["camera"] = {
        **deepcopy(template_scenario["plan"]["camera"]),
        "sensor_rig_trajectory_id": sensor_rig["trajectory_id"],
        "dynamic": False,
    }
    scenario["static_camera_upgrade"] = {
        "schema": sensor_rig["schema"],
        "trajectory_id": sensor_rig["trajectory_id"],
        "pose_hash": sensor_rig["frames"][0]["pose_hash"],
    }
    scenario["authoritative_capture_request"] = {
        "request_id": f"{EPISODE_ID}__native_capture",
        "episode_id": EPISODE_ID,
        "scenario_type": "strict_two_human_static_canary",
        "target_source_slot_id": "source1",
        "fact_path": "PENDING_NATIVE_CAPTURE",
        "fact_sha256": "PENDING_NATIVE_CAPTURE",
    }
    scenario["authoritative_inputs"] = {
        "source_endpoint_registry": "controlled_audio_program/source_endpoint_registry.json",
        "sound_asset_registry": "controlled_audio_program/sound_asset_registry.json",
        "audio_program": "controlled_audio_program/audio_program.json",
    }
    scenario["reuse_contract"] = {
        "camera_and_room": "retained stationary paper-balance Apartment seed",
        "actor_roots": "retained roots with both runtime profiles replaced by distinct adults",
        "actor_yaws": "recomputed independently from each human profile to face the camera",
        "audio": "new source1-only controlled AudioProgram and exact asset-bound RIR required",
    }
    suite = deepcopy(source_suite)
    suite["scenarios"] = [scenario]
    suite["camera_upgrade"] = {
        "schema": "avengine_static_spear_suite_camera_upgrade_v1",
        "source_suite": str(source_suite_path.resolve()),
        "sensor_rig_trajectory_id": sensor_rig["trajectory_id"],
        "qualification_claim": False,
    }

    root_paths = {
        slot: [deepcopy(point) for _ in range(FRAME_COUNT)]
        for slot, point in bundle["roots"].items()
    }
    emitter_paths = {
        slot: [deepcopy(point) for _ in range(FRAME_COUNT)]
        for slot, point in bundle["emitters"].items()
    }
    trajectory_bank = {
        "schema": "avengine_room_trajectory_bank_v2",
        "seed": 20260811,
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FRAME_RATE_HZ,
        "seconds_per_episode": 5,
        "source_slots": ["source1", "source2"],
        "episode_count": 1,
        "motion_case_counts": {"strict_two_human_static_canary": 1},
        "claim_boundary": "exact asset-bound two-human geometry pending native sparse capture",
        "path_semantics": {
            "source_center_paths_m": "asset-bound world mouth emitter points",
            "source_root_paths_m": "asset-bound actor roots",
        },
        "semantics": "two distinct original adult identities, target source1 speaking, source2 silent",
        "episodes": [
            {
                "episode_id": EPISODE_ID,
                "motion_case": "strict_two_human_static_canary",
                "source_center_paths_m": emitter_paths,
                "source_root_paths_m": root_paths,
                "statistics": {
                    "target_source_slot_id": "source1",
                    "distractor_source_slot_id": "source2",
                    "native_recapture_required": True,
                },
            }
        ],
    }
    binding_report = build_acoustic_binding_report(
        plan=plan,
        registry=registry,
        bundle=bundle,
    )
    audio = build_audio_contracts(
        plan=plan,
        registry=registry,
        registry_path=runtime_registry_path,
        controlled_registry=controlled_registry,
        controlled_registry_path=controlled_registry_path,
    )

    output.mkdir(parents=True)
    audio_root = output / "controlled_audio_program"
    audio_root.mkdir()
    paths = {
        "trajectory_bank": output / "trajectory_bank.json",
        "sensor_rig_trajectory": output / "sensor_rig_trajectory.json",
        "asset_emitter_binding_report": output / "asset_emitter_binding_report.json",
        "suite": output / "suite_execution_plan.pending_fact.json",
        "preflight": output / "preflight.json",
        "sparse_gate_request": output / "sparse_native_gate_request.json",
        "source_endpoint_registry": audio_root / "source_endpoint_registry.json",
        "sound_asset_registry": audio_root / "sound_asset_registry.json",
        "audio_program": audio_root / "audio_program.json",
        "controlled_audio_binding": audio_root / "controlled_audio_binding.json",
    }
    write_json(paths["trajectory_bank"], trajectory_bank)
    write_json(paths["sensor_rig_trajectory"], sensor_rig)
    write_json(paths["asset_emitter_binding_report"], binding_report)
    write_json(paths["suite"], suite)
    write_json(
        paths["preflight"],
        {
            "schema": "avengine_native_strict_two_human_recipe_preflight_v1",
            "status": "pass_pending_exact_rir_and_native_sparse",
            "qualification_claim": False,
            "episode_id": EPISODE_ID,
            "actor_ids": ["source1_actor", "source2_actor"],
            "actor_yaw_ue_deg": {
                state["actor_id"]: state["actor_yaw_ue_deg"]
                for state in bundle["state_templates"]
            },
            "projection_offset_fraction": bundle["projection_offset_fraction"],
            "target_event_count": 1,
            "distractor_event_count": 0,
            "exact_rir": "pending_required",
            "native_sparse": "pending_required",
            "formal_scene_count": 0,
        },
    )
    write_json(
        paths["sparse_gate_request"],
        {
            "schema": "avengine_native_strict_two_human_sparse_gate_request_v1",
            "status": "blocked_pending_exact_rir",
            "episode_id": EPISODE_ID,
            "frame_indices": [15],
            **plan["gpu_policy"],
        },
    )
    write_json(paths["source_endpoint_registry"], audio["source_endpoint_registry"])
    write_json(paths["sound_asset_registry"], audio["sound_asset_registry"])
    write_json(paths["audio_program"], audio["audio_program"])
    write_json(
        paths["controlled_audio_binding"],
        {
            "schema": "avengine_native_strict_two_human_audio_binding_v1",
            "status": "pass_pending_exact_rir_render",
            "source_endpoint_slots": {
                "lead_d_source1_mouth": "source1",
                "lead_d_source2_mouth": "source2",
            },
            "sound_audio_paths": {
                plan["actors"][0]["sound_asset_id"]: str(audio["media_path"].resolve())
            },
            "controlled_content": audio["controlled_content"],
            "target_event_count": 1,
            "distractor_event_count": 0,
        },
    )
    recipe = {
        "schema": SCHEMA,
        "status": "prepared_pending_exact_rir_and_native_sparse",
        "qualification_claim": False,
        "episode_id": EPISODE_ID,
        "scenario_type": "strict_two_human_static_canary",
        "target_source_slot_id": "source1",
        "inputs": {
            "plan": str(plan_path.resolve()),
            "cpu_preflight": str(cpu_preflight_path.resolve()),
            "runtime_registry": str(runtime_registry_path.resolve()),
            "source_suite_template": str(source_suite_path.resolve()),
            "controlled_sound_registry": str(controlled_registry_path.resolve()),
        },
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
    }
    paths["recipe"] = output / "recipe.json"
    write_json(paths["recipe"], recipe)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPOSITORY / "examples/qa/native_strict_two_human_canary_v1.json",
    )
    parser.add_argument("--cpu-preflight", type=Path, required=True)
    parser.add_argument(
        "--runtime-registry",
        type=Path,
        default=REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json",
    )
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--controlled-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = build(
        plan_path=args.plan.resolve(),
        cpu_preflight_path=args.cpu_preflight.resolve(),
        runtime_registry_path=args.runtime_registry.resolve(),
        source_suite_path=args.source_suite.resolve(),
        controlled_registry_path=args.controlled_registry.resolve(),
        output=args.output.resolve(),
    )
    print(f"STRICT_TWO_HUMAN_RECIPE_OK recipe={paths['recipe']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

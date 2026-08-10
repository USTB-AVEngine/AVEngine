#!/usr/bin/env python3
"""Prepare and validate rows 2-8 of the strict two-human CPU acoustic batch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import canonical_json_sha256, write_json
from avengine.m6.audio_program import bind_audio_program_hash, validate_audio_program
from avengine.m6.registry import bind_content_hash
from avengine.m6.sources import (
    validate_sound_asset_registry,
    validate_source_endpoint_registry,
)
from avengine.m7.sensor_rig import m7_sensor_rig_binding
from avengine.optional_backends.spear_visual import (
    actor_ue_yaw_degrees,
    camera_ue_yaw_degrees,
    habitat_point_to_apartment_ue_cm,
)
from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory

CANARY_PATH = REPOSITORY / "tools/qa/build_strict_two_human_canary_recipe.py"
CANARY_SPEC = importlib.util.spec_from_file_location("strict_canary_recipe", CANARY_PATH)
if CANARY_SPEC is None or CANARY_SPEC.loader is None:
    raise RuntimeError(f"cannot import {CANARY_PATH}")
CANARY = importlib.util.module_from_spec(CANARY_SPEC)
CANARY_SPEC.loader.exec_module(CANARY)

PREFLIGHT_PATH = REPOSITORY / "tools/qa/build_strict_two_human_expansion_preflight.py"
PREFLIGHT_SPEC = importlib.util.spec_from_file_location(
    "strict_expansion_preflight", PREFLIGHT_PATH
)
if PREFLIGHT_SPEC is None or PREFLIGHT_SPEC.loader is None:
    raise RuntimeError(f"cannot import {PREFLIGHT_PATH}")
PREFLIGHT = importlib.util.module_from_spec(PREFLIGHT_SPEC)
PREFLIGHT_SPEC.loader.exec_module(PREFLIGHT)

SCHEMA = "avengine_native_strict_two_human_expansion_acoustic_batch_v1"
ROW_SCHEMA = "avengine_native_strict_two_human_expansion_acoustic_recipe_v1"
DELIVERY_SCHEMA = "avengine_native_strict_two_human_expansion_cpu_delivery_v1"
FRAME_COUNT = 75
FRAME_RATE_HZ = 15
TICKS_PER_FRAME = 3200
SAMPLE_RATE_HZ = 16000
SAMPLE_COUNT = 80000
UE_IMPORT_ASSET_ID_BY_ORIGINAL_IDENTITY = {
    "rocketbox_adults_male_adult_01": "rocketbox_male_adult_01",
    "rocketbox_adults_female_adult_01": "rocketbox_female_adult_01",
    "rocketbox_professions_construction_male_01": (
        "rocketbox_construction_male_01"
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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


def _record(role: str, path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"{role} input is missing: {path}")
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _wave_header(path: Path) -> dict[str, int]:
    with path.open("rb") as stream:
        _require(stream.read(4) == b"RIFF", f"invalid WAV RIFF header: {path}")
        stream.read(4)
        _require(stream.read(4) == b"WAVE", f"invalid WAV format header: {path}")
        fmt: tuple[int, int, int, int] | None = None
        data_size: int | None = None
        while True:
            chunk_id = stream.read(4)
            if not chunk_id:
                break
            chunk_size_raw = stream.read(4)
            _require(len(chunk_size_raw) == 4, f"truncated WAV chunk: {path}")
            chunk_size = struct.unpack("<I", chunk_size_raw)[0]
            payload = stream.read(chunk_size)
            _require(len(payload) == chunk_size, f"truncated WAV payload: {path}")
            if chunk_size % 2:
                stream.read(1)
            if chunk_id == b"fmt ":
                _require(chunk_size >= 16, f"short WAV fmt chunk: {path}")
                format_tag, channels, sample_rate, _, _, bits = struct.unpack(
                    "<HHIIHH", payload[:16]
                )
                fmt = (format_tag, channels, sample_rate, bits)
            elif chunk_id == b"data":
                data_size = chunk_size
        _require(fmt is not None and data_size is not None, f"WAV chunks missing: {path}")
        format_tag, channels, sample_rate, bits = fmt
        _require(format_tag in {1, 3}, f"unsupported WAV encoding: {format_tag}")
        bytes_per_frame = channels * bits // 8
        _require(bytes_per_frame > 0, f"invalid WAV frame width: {path}")
        _require(
            data_size % bytes_per_frame == 0,
            f"WAV data does not contain whole frames: {path}",
        )
        return {
            "format_tag": format_tag,
            "channel_count": channels,
            "sample_rate_hz": sample_rate,
            "sample_count": data_size // bytes_per_frame,
        }


def _angle_delta_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _speech_window(
    *, start_sample: int, source_sample_count: int
) -> tuple[int, int, list[int]]:
    _require(start_sample >= 0, "speech start sample is negative")
    _require(source_sample_count > 0, "target dry speech is empty")
    end_sample = start_sample + source_sample_count
    _require(end_sample <= SAMPLE_COUNT, "target dry speech exceeds five seconds")
    first_frame = start_sample * FRAME_RATE_HZ // SAMPLE_RATE_HZ
    last_frame = (end_sample - 1) * FRAME_RATE_HZ // SAMPLE_RATE_HZ
    return start_sample, end_sample, [first_frame, last_frame]


def _runtime_actor(
    *,
    actor_id: str,
    identity: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    profile = CANARY._asset(registry, identity["runtime_asset_id"])
    manifest_path = Path(identity["ue_import_manifest"])
    _require(manifest_path.is_file(), f"{actor_id} UE import manifest is missing")
    manifest = _load(manifest_path)
    original_identity_id = str(identity["original_identity_id"])
    _require(
        manifest.get("asset_id")
        == UE_IMPORT_ASSET_ID_BY_ORIGINAL_IDENTITY.get(original_identity_id),
        f"{actor_id} UE import asset identity mismatch",
    )
    observed_base_avatar_id = manifest.get("base_avatar_id")
    if observed_base_avatar_id is not None:
        _require(
            observed_base_avatar_id == original_identity_id,
            f"{actor_id} UE import base-avatar identity mismatch",
        )
    else:
        _require(
            original_identity_id == "rocketbox_adults_male_adult_01",
            f"{actor_id} only the legacy male lineage may omit base_avatar_id",
        )
        normalization_path = Path(manifest["source_manifest"])
        _require(
            normalization_path.is_file(),
            f"{actor_id} legacy male normalization manifest is missing",
        )
        normalization = _load(normalization_path)
        _require(
            normalization.get("asset_id") == "rocketbox_male_adult_01"
            and normalization.get("tag")
            == "rocketbox_male_adult_01_original_ue_v3"
            and normalization.get("automatic_checks", {}).get("overall")
            == "passed",
            f"{actor_id} legacy male normalization lineage mismatch",
        )
    _require(
        manifest.get("reload_verification", {}).get("status") == "passed",
        f"{actor_id} UE reload did not pass",
    )
    declaration = CANARY._profile_actor(
        actor_id,
        profile,
        manifest["content"],
        manifest_path,
    )
    return profile, declaration


def _actor_bundle(
    *, row: Mapping[str, Any], plan: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, Any]:
    catalog = plan["approved_identity_catalog"]
    camera = row["camera_pose"]
    camera_m = camera["translation_m"]
    thresholds = plan["projection_and_native_thresholds"]
    declarations: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    roots: dict[str, list[float]] = {}
    emitters: dict[str, list[float]] = {}
    projections: dict[str, list[float]] = {}
    vertical_envelopes: dict[str, dict[str, float]] = {}
    profiles: dict[str, Mapping[str, Any]] = {}
    identities: dict[str, Mapping[str, Any]] = {}
    for actor in row["actors"]:
        slot = str(actor["source_slot_id"])
        actor_id = f"{slot}_actor"
        identity = catalog[actor["identity_key"]]
        profile, declaration = _runtime_actor(
            actor_id=actor_id,
            identity=identity,
            registry=registry,
        )
        root = [float(value) for value in actor["root_translation_m"]]
        rotation, forward_h = CANARY._facing_camera_rotation(root, camera_m)
        observed_yaw = actor_ue_yaw_degrees(
            rotation,
            declaration["habitat_local_anatomical_forward_axis"],
            declaration["ue_anatomical_forward_yaw_deg"],
        )
        _require(
            _angle_delta_deg(observed_yaw, float(actor["actor_yaw_ue_deg"]))
            <= float(thresholds["actor_facing_yaw_tolerance_deg"]),
            f"{row['row_id']}/{slot} actor yaw drift",
        )
        mouth_offset = [float(value) for value in identity["mouth_offset_from_root_m"]]
        _require(
            mouth_offset[0] == 0.0 and mouth_offset[2] == 0.0,
            f"{row['row_id']}/{slot} requires vertical mouth offset",
        )
        emitter = [root[i] + mouth_offset[i] for i in range(3)]
        _, _, x_fraction, y_fraction = PREFLIGHT._project(
            camera,
            emitter,
            float(thresholds["horizontal_fov_deg"]),
            thresholds["resolution_hw"],
        )
        states.append(
            {
                "action_id": "idle",
                "action_phase": 0.0,
                "action_time_ticks": 0,
                "actor_id": actor_id,
                "actor_yaw_ue_deg": observed_yaw,
                "anatomical_forward_habitat_world": forward_h,
                "anatomical_forward_ue_world": [forward_h[0], forward_h[2], 0.0],
                "asset_id": profile["asset_id"],
                "blueprint_class_path": declaration["blueprint_class_path"],
                "rotation_xyzw": rotation,
                "translation_m": root,
                "translation_ue_cm": list(habitat_point_to_apartment_ue_cm(root)),
                "ue_animation": declaration["idle_animation"],
            }
        )
        declarations.append(declaration)
        roots[slot] = root
        emitters[slot] = emitter
        projections[slot] = [x_fraction, y_fraction]
        envelope = thresholds["conservative_actor_vertical_envelope_from_root_m"]
        _, _, _, bottom_fraction = PREFLIGHT._project(
            camera,
            [root[0], root[1] + float(envelope[0]), root[2]],
            float(thresholds["horizontal_fov_deg"]),
            thresholds["resolution_hw"],
        )
        _, _, _, top_fraction = PREFLIGHT._project(
            camera,
            [root[0], root[1] + float(envelope[1]), root[2]],
            float(thresholds["horizontal_fov_deg"]),
            thresholds["resolution_hw"],
        )
        vertical_envelopes[slot] = {
            "top_y_fraction": top_fraction,
            "bottom_y_fraction": bottom_fraction,
        }
        profiles[slot] = profile
        identities[slot] = identity
    return {
        "declarations": declarations,
        "states": states,
        "roots": roots,
        "emitters": emitters,
        "projections": projections,
        "vertical_envelopes": vertical_envelopes,
        "profiles": profiles,
        "identities": identities,
    }


def _audio_contracts(
    *,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    bundle: Mapping[str, Any],
    registry_path: Path,
    controlled_registry: Mapping[str, Any],
    controlled_registry_path: Path,
) -> dict[str, Any]:
    target_identity = bundle["identities"]["source1"]
    target_profile = bundle["profiles"]["source1"]
    distractor_profile = bundle["profiles"]["source2"]
    speech = CANARY.AUDIO_BUILDER._controlled_asset(
        controlled_registry, target_identity["sound_asset_id"]
    )
    _require(
        speech["content"]["transcript"] == target_identity["transcript"],
        f"{row['row_id']} transcript drift",
    )
    _require(
        speech.get("listening_review", {}).get("state") == "pending",
        f"{row['row_id']} listening boundary drift",
    )
    media_path = (
        controlled_registry_path.parent
        / "media"
        / f"{target_identity['sound_asset_id']}.wav"
    )
    CANARY.AUDIO_BUILDER._validate_wave(media_path, speech["audio"])
    start_sample, end_sample, frame_window = _speech_window(
        start_sample=int(plan["timeline"]["target_speech_start_sample"]),
        source_sample_count=int(speech["audio"]["sample_count"]),
    )
    _require(
        frame_window == target_identity["expected_speech_frame_window_inclusive"],
        f"{row['row_id']} full dry speech frame window drift",
    )
    _require(frame_window[0] <= 15 <= frame_window[1], f"{row['row_id']} f15 is silent")

    runtime_sha = _sha256(registry_path)
    endpoints = bind_content_hash(
        {
            "schema": "avengine_m6_source_endpoint_registry_v1",
            "registry_id": f"{row['row_id']}__endpoints_v1",
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
    sounds = bind_content_hash(
        {
            "schema": "avengine_m6_sound_asset_registry_v1",
            "registry_id": f"{row['row_id']}__sounds_v1",
            "revision": "v1",
            "sound_assets": [
                CANARY._sound_record(
                    speech,
                    rights_evidence_sha256=_sha256(controlled_registry_path),
                )
            ],
        }
    )
    program = bind_audio_program_hash(
        {
            "schema": "avengine_m6_audio_program_v1",
            "program_id": f"{row['episode_id']}__audio_v1",
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
                CANARY.AUDIO_BUILDER._event(
                    event_id="source1_speech_000",
                    endpoint_id="lead_d_source1_mouth",
                    sound_id=target_identity["sound_asset_id"],
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
    errors = [
        *validate_source_endpoint_registry(endpoints),
        *validate_sound_asset_registry(sounds),
        *validate_audio_program(
            program,
            source_endpoint_registry=endpoints,
            sound_asset_registry=sounds,
        ),
    ]
    _require(not errors, "; ".join(errors))
    return {
        "source_endpoint_registry": endpoints,
        "sound_asset_registry": sounds,
        "audio_program": program,
        "media_path": media_path,
        "sound_asset_id": target_identity["sound_asset_id"],
        "transcript": target_identity["transcript"],
        "event_sample_window": [start_sample, end_sample],
        "event_frame_window_inclusive": frame_window,
        "source_sample_count": speech["audio"]["sample_count"],
    }


def _row_recipe(
    *,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_path: Path,
    cpu_preflight_path: Path,
    registry: Mapping[str, Any],
    registry_path: Path,
    source_suite: Mapping[str, Any],
    source_suite_path: Path,
    controlled_registry: Mapping[str, Any],
    controlled_registry_path: Path,
    output: Path,
) -> dict[str, Any]:
    _require(not output.exists(), f"refusing to overwrite row output: {output}")
    bundle = _actor_bundle(row=row, plan=plan, registry=registry)
    camera = row["camera_pose"]
    sensor_rig = materialize_sensor_rig_trajectory(
        trajectory_id=f"{row['episode_id']}__sensor_rig",
        program={
            "kind": "HOLD",
            "position_m": camera["translation_m"],
            "yaw_deg": camera["habitat_yaw_deg"],
        },
    )
    _require(len(sensor_rig["frames"]) == FRAME_COUNT, "sensor rig frame drift")
    _require(len(source_suite.get("scenarios", [])) == 1, "source suite must contain one template")
    template = source_suite["scenarios"][0]
    template_frames = template["plan"]["frames"]
    _require(len(template_frames) == FRAME_COUNT, "source suite frame drift")
    frames: list[dict[str, Any]] = []
    ue_position = list(habitat_point_to_apartment_ue_cm(camera["translation_m"]))
    ue_yaw = camera_ue_yaw_degrees(float(camera["habitat_yaw_deg"]))
    for frame_index in range(FRAME_COUNT):
        camera_state = deepcopy(template_frames[frame_index]["camera_state"])
        camera_state.update(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * TICKS_PER_FRAME,
                "habitat_position_m": deepcopy(camera["translation_m"]),
                "habitat_yaw_deg": float(camera["habitat_yaw_deg"]),
                "ue_position_cm": ue_position,
                "ue_yaw_deg": ue_yaw,
                "world_from_rig": deepcopy(
                    sensor_rig["frames"][frame_index]["world_from_rig"]
                ),
                "pose_hash": sensor_rig["frames"][frame_index]["pose_hash"],
            }
        )
        actor_states = []
        for state_template in bundle["states"]:
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
    scenario = deepcopy(template)
    scenario["scenario_id"] = row["episode_id"]
    scenario["scenario_directory"] = row["episode_id"]
    scenario["variant_id"] = "strict_two_human_expansion_static"
    scenario["plan"]["actors"] = bundle["declarations"]
    scenario["plan"]["frames"] = frames
    scenario["plan"]["camera"] = {
        **deepcopy(template["plan"]["camera"]),
        "habitat_position_m": deepcopy(camera["translation_m"]),
        "habitat_yaw_deg": float(camera["habitat_yaw_deg"]),
        "ue_position_cm": ue_position,
        "ue_yaw_deg": ue_yaw,
        "sensor_rig_trajectory_id": sensor_rig["trajectory_id"],
        "dynamic": False,
    }
    scenario["static_camera_upgrade"] = {
        "schema": sensor_rig["schema"],
        "trajectory_id": sensor_rig["trajectory_id"],
        "pose_hash": sensor_rig["frames"][0]["pose_hash"],
    }
    live_expectations = {
        declaration["actor_id"].removesuffix("_actor"): declaration[
            "runtime_asset_expectation"
        ]
        for declaration in bundle["declarations"]
    }
    scenario["authoritative_capture_request"] = {
        "request_id": f"{row['episode_id']}__native_capture",
        "episode_id": row["episode_id"],
        "scenario_type": "strict_two_human_expansion_static",
        "target_source_slot_id": "source1",
        "fact_path": "PENDING_NATIVE_CAPTURE",
        "fact_sha256": "PENDING_NATIVE_CAPTURE",
        "runtime_asset_expectations": live_expectations,
    }
    scenario["authoritative_inputs"] = {
        "source_endpoint_registry": "controlled_audio_program/source_endpoint_registry.json",
        "sound_asset_registry": "controlled_audio_program/sound_asset_registry.json",
        "audio_program": "controlled_audio_program/audio_program.json",
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
        "motion_case_counts": {"strict_two_human_expansion_static": 1},
        "claim_boundary": "exact profile mouth geometry pending native sparse capture",
        "path_semantics": {
            "source_center_paths_m": "asset-bound world mouth emitter points",
            "source_root_paths_m": "asset-bound actor roots",
        },
        "semantics": "distinct original adult target speaking and source2 silent",
        "episodes": [
            {
                "episode_id": row["episode_id"],
                "motion_case": "strict_two_human_expansion_static",
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
    binding_report = {
        "schema": "avengine_asset_emitter_scenario_report_v1",
        "status": "pass",
        "qualification_claim": False,
        "claim_boundary": "profile-coordinate acoustic plan pending f15 native actor-root readback",
        "method": "runtime_profile_root_plus_declared_mouth_offset",
        "native_readback_status": "pending_required",
        "scenario_count": 1,
        "scenarios": [
            {
                "trajectory_episode_id": row["episode_id"],
                "output_episode_id": row["episode_id"],
                "binding_report": {
                    "schema": "avengine_asset_emitter_binding_report_v1",
                    "status": "pass",
                    "qualification_claim": False,
                    "native_readback_status": "pending_required",
                    "episode_count": 1,
                    "listener_position_m": camera["translation_m"],
                    "bindings": [
                        {
                            "source_slot_id": slot,
                            "asset_id": bundle["profiles"][slot]["asset_id"],
                            "asset_revision": bundle["profiles"][slot]["revision"],
                            "semantic_anchor_id": "mouth",
                            "emitter_offset_m": bundle["identities"][slot][
                                "mouth_offset_from_root_m"
                            ],
                            "offset_space": "final_scaled_asset_root",
                            "native_readback": "pending_required",
                        }
                        for slot in ("source1", "source2")
                    ],
                    "target_world_emitter_at_sparse_frame_m": bundle["emitters"][
                        "source1"
                    ],
                },
            }
        ],
    }
    audio = _audio_contracts(
        row=row,
        plan=plan,
        bundle=bundle,
        registry_path=registry_path,
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
            "schema": "avengine_native_strict_two_human_expansion_row_preflight_v1",
            "status": "pass_pending_exact_rir_cache_binaural_and_native_sparse",
            "qualification_claim": False,
            "row_id": row["row_id"],
            "episode_id": row["episode_id"],
            "identity_pair": row["identity_pair"],
            "target_expected_screen_side": row["target_expected_screen_side"],
            "projection_xy_fraction": bundle["projections"],
            "conservative_vertical_envelope_fraction": bundle[
                "vertical_envelopes"
            ],
            "target_event_count": 1,
            "distractor_event_count": 0,
            "target_sound_asset_id": audio["sound_asset_id"],
            "target_transcript": audio["transcript"],
            "target_event_sample_window": audio["event_sample_window"],
            "target_event_frame_window_inclusive": audio[
                "event_frame_window_inclusive"
            ],
            "f15_target_speaking": True,
            "exact_rir": "pending_required_per_row",
            "native_sparse": "pending_required",
            "formal_scene_count": 0,
        },
    )
    write_json(
        paths["sparse_gate_request"],
        {
            "schema": "avengine_native_strict_two_human_sparse_gate_request_v1",
            "status": "blocked_pending_exact_rir_cache_binaural",
            "row_id": row["row_id"],
            "episode_id": row["episode_id"],
            "frame_indices": [15],
            "suite_plan": str(paths["suite"].resolve()),
            "runtime_asset_expectations": live_expectations,
            "target_only_actor_map": plan["target_only_actor_map"],
            "required_live_gates": [
                "stable_actor_tag",
                "exact_blueprint_class",
                "skeletal_mesh",
                "skeleton",
                "standing_idle",
                "native_actor_root_plus_declared_profile_offset",
            ],
            "projection_and_native_thresholds": plan[
                "projection_and_native_thresholds"
            ],
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
                audio["sound_asset_id"]: str(audio["media_path"].resolve())
            },
            "target_sound_asset_id": audio["sound_asset_id"],
            "target_transcript": audio["transcript"],
            "target_event_sample_window": audio["event_sample_window"],
            "target_event_frame_window_inclusive": audio[
                "event_frame_window_inclusive"
            ],
            "target_event_count": 1,
            "distractor_event_count": 0,
        },
    )
    recipe = {
        "schema": ROW_SCHEMA,
        "status": "prepared_pending_exact_rir_cache_binaural_and_native_sparse",
        "qualification_claim": False,
        "row_id": row["row_id"],
        "episode_id": row["episode_id"],
        "target_source_slot_id": "source1",
        "inputs": {
            "plan": _record("strict8_plan", plan_path),
            "cpu_preflight": _record("strict8_cpu_preflight", cpu_preflight_path),
            "runtime_registry": _record("runtime_registry", registry_path),
            "source_suite_template": _record(
                "source_suite_template", source_suite_path
            ),
            "controlled_sound_registry": _record(
                "controlled_sound_registry", controlled_registry_path
            ),
        },
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
    }
    recipe["recipe_identity_sha256"] = canonical_json_sha256(
        {
            "schema": ROW_SCHEMA,
            "row_id": row["row_id"],
            "episode_id": row["episode_id"],
            "inputs": recipe["inputs"],
            "trajectory_bank": canonical_json_sha256(trajectory_bank),
            "sensor_rig_trajectory": canonical_json_sha256(sensor_rig),
            "asset_emitter_binding_report": canonical_json_sha256(binding_report),
            "audio_program_content_sha256": audio["audio_program"][
                "program_content_sha256"
            ],
        }
    )
    paths["recipe"] = output / "recipe.json"
    write_json(paths["recipe"], recipe)
    return {
        "row_id": row["row_id"],
        "episode_id": row["episode_id"],
        "recipe": str(paths["recipe"].resolve()),
        "recipe_root": str(output.resolve()),
        "recipe_identity_sha256": recipe["recipe_identity_sha256"],
        "target_sound_asset_id": audio["sound_asset_id"],
        "target_event_frame_window_inclusive": audio[
            "event_frame_window_inclusive"
        ],
        "source1_emitter_m": bundle["emitters"]["source1"],
        "source2_emitter_m": bundle["emitters"]["source2"],
        "status": "prepared_pending_exact_rir_cache_binaural_and_native_sparse",
    }


def prepare(
    *,
    plan_path: Path,
    cpu_preflight_path: Path,
    registry_path: Path,
    source_suite_path: Path,
    controlled_registry_path: Path,
    output: Path,
) -> Path:
    _require(not output.exists(), f"refusing to overwrite batch output: {output}")
    plan = _load(plan_path)
    cpu_preflight = _load(cpu_preflight_path)
    registry = _load(registry_path)
    source_suite = _load(source_suite_path)
    controlled_registry = _load(controlled_registry_path)
    errors = PREFLIGHT.validate_plan(plan, registry)
    _require(not errors, "strict8 plan failed: " + "; ".join(errors))
    _require(
        cpu_preflight.get("status")
        == "pass_cpu_plan_pending_exact_rir_and_seven_sparse_native_gates",
        "strict8 CPU preflight status mismatch",
    )
    _require(cpu_preflight.get("plan_id") == plan["plan_id"], "strict8 plan ID drift")
    _require(
        cpu_preflight.get("plan_record", {}).get("sha256") == _sha256(plan_path),
        "strict8 CPU preflight does not bind this exact plan",
    )
    _require(cpu_preflight.get("row_count") == 8, "strict8 CPU preflight row drift")
    _require(
        cpu_preflight.get("left_target_count") == 4
        and cpu_preflight.get("right_target_count") == 4,
        "strict8 CPU preflight side balance drift",
    )
    _require(
        cpu_preflight.get("camera_translation_cluster_count") == 8
        and float(cpu_preflight.get("minimum_camera_translation_separation_m", 0.0))
        >= float(
            plan["projection_and_native_thresholds"][
                "minimum_camera_translation_cluster_separation_m"
            ]
        ),
        "strict8 CPU preflight camera independence drift",
    )
    _require(
        cpu_preflight.get("native_occupied_floor_point_count") == 21
        and len(cpu_preflight.get("occupied_floor_point_evidence", [])) == 21,
        "strict8 CPU preflight provenance closure drift",
    )
    _require(
        cpu_preflight.get("formal_scene_count") == 0
        and cpu_preflight.get("qualification_claim") is False
        and cpu_preflight.get("gpu_or_rir_executed") is False,
        "strict8 CPU preflight claim boundary drift",
    )
    expected_preflight_statuses = [
        "pass_existing_sparse_canary",
        *[
            "pass_cpu_geometry_pending_exact_rir_and_native_sparse"
            for _ in range(7)
        ],
    ]
    _require(
        [record.get("status") for record in cpu_preflight.get("rows", [])]
        == expected_preflight_statuses,
        "strict8 CPU preflight per-row state drift",
    )
    rows = plan["rows"]
    _require(rows[0]["status"] == "pass_existing_sparse_canary", "row1 canary drift")
    selected = rows[1:]
    _require(len(selected) == 7, "exactly rows2-8 must be prepared")
    output.mkdir(parents=True)
    row_records = [
        _row_recipe(
            row=row,
            plan=plan,
            plan_path=plan_path,
            cpu_preflight_path=cpu_preflight_path,
            registry=registry,
            registry_path=registry_path,
            source_suite=source_suite,
            source_suite_path=source_suite_path,
            controlled_registry=controlled_registry,
            controlled_registry_path=controlled_registry_path,
            output=output / row["row_id"] / "recipe_v1",
        )
        for row in selected
    ]
    _require(
        len({record["episode_id"] for record in row_records}) == 7,
        "row episode IDs are not unique",
    )
    manifest = {
        "schema": SCHEMA,
        "status": "prepared_cpu_pending_per_row_rir_cache_binaural",
        "claim_boundary": "rows2-8 CPU recipes only; no GPU capture or formal scene is claimed",
        "row_count": 7,
        "retained_row1_canary": {
            "row_id": rows[0]["row_id"],
            "episode_id": rows[0]["episode_id"],
            "status": rows[0]["status"],
        },
        "rows": row_records,
        "cross_row_rir_reuse_allowed": False,
        "gpu_executed": False,
        "formal_scene_count": 0,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def finalize(*, batch_root: Path, output: Path) -> Path:
    _require(not output.exists(), f"refusing to overwrite delivery output: {output}")
    manifest = _load(batch_root / "manifest.json")
    _require(manifest.get("schema") == SCHEMA, "batch manifest schema mismatch")
    _require(manifest.get("row_count") == 7, "batch manifest row count mismatch")
    global_states: set[str] = set()
    rows: list[dict[str, Any]] = []
    output.mkdir(parents=True)
    for row in manifest["rows"]:
        row_root = batch_root / row["row_id"]
        recipe_root = row_root / "recipe_v1"
        plan_root = row_root / "exact_rir_plan_v1"
        cache_root = row_root / "rir_cache_v1"
        binaural_root = row_root / "binaural_v1"
        recipe = _load(recipe_root / "recipe.json")
        _require(
            recipe.get("recipe_identity_sha256")
            == row.get("recipe_identity_sha256"),
            f"{row['row_id']} recipe identity drift",
        )
        for record in recipe.get("inputs", {}).values():
            input_path = Path(record["path"])
            _require(
                input_path.is_file()
                and input_path.stat().st_size == record["size_bytes"]
                and _sha256(input_path) == record["sha256"],
                f"{row['row_id']} recipe input drift: {record.get('role')}",
            )
        trajectory = _load(recipe_root / "trajectory_bank.json")
        sensor_rig = _load(recipe_root / "sensor_rig_trajectory.json")
        audio_program = _load(
            recipe_root / "controlled_audio_program/audio_program.json"
        )
        endpoints = _load(
            recipe_root / "controlled_audio_program/source_endpoint_registry.json"
        )
        sound_registry = _load(
            recipe_root / "controlled_audio_program/sound_asset_registry.json"
        )
        audio_binding = _load(
            recipe_root / "controlled_audio_program/controlled_audio_binding.json"
        )
        plan_delivery = _load(plan_root / "delivery.json")
        rir_plan = _load(plan_root / "rir_job_plan.json")
        sound_assignments = _load(plan_root / "sound_assignments.json")
        cache_receipt = _load(cache_root / "receipt.json")
        cache_request = _load(cache_root / "request.json")
        cache_index = _load(cache_root / "index.json")
        binaural_delivery = _load(binaural_root / "delivery.json")
        samples = _load(binaural_root / "samples.json")
        recipe_preflight = _load(recipe_root / "preflight.json")
        blocked_request = _load(recipe_root / "sparse_native_gate_request.json")
        _require(plan_delivery.get("status") == "pass", f"{row['row_id']} RIR plan failed")
        _require(plan_delivery.get("unique_rir_job_count") == 2, f"{row['row_id']} requires two exact RIRs")
        _require(plan_delivery.get("research_only") is True, f"{row['row_id']} RIR plan left research boundary")
        _require(plan_delivery.get("sound_pair_counts") == {"human_speech|silent_human": 1}, f"{row['row_id']} sound-pair metadata drift")
        _require(
            sound_assignments.get("ordered_pair_counts")
            == {"human_speech|silent_human": 1}
            and sound_assignments.get("assignments")
            == [
                {
                    "episode_id": row["episode_id"],
                    "source_classes": {
                        "source1": "human_speech",
                        "source2": "silent_human",
                    },
                }
            ],
            f"{row['row_id']} exact sound assignment drift",
        )
        forbidden_metadata = json.dumps(
            {
                "delivery": plan_delivery,
                "assignments": sound_assignments,
                "audio_binding": audio_binding,
            },
            sort_keys=True,
        ).lower()
        _require(
            not any(
                token in forbidden_metadata
                for token in ("dog_bark", "cat_meow", "dog|", "cat|")
            ),
            f"{row['row_id']} inherited animal sound metadata",
        )
        sensor_binding = m7_sensor_rig_binding(sensor_rig)
        declared_sensor_binding = plan_delivery.get("sensor_rig_trajectory", {})
        _require(
            declared_sensor_binding.get("relative_path")
            == "sensor_rig_trajectory.json"
            and all(
                declared_sensor_binding.get(key) == sensor_binding[key]
                for key in (
                    "trajectory_id",
                    "content_sha256",
                    "first_pose_hash",
                    "last_pose_hash",
                )
            ),
            f"{row['row_id']} RIR plan sensor binding drift",
        )
        jobs = rir_plan.get("jobs", [])
        _require(len(jobs) == 2, f"{row['row_id']} RIR job count drift")
        episode = trajectory["episodes"][0]
        expected_emitter_by_slot = {
            slot: points[15]
            for slot, points in episode["source_center_paths_m"].items()
        }
        jobs_by_slot: dict[str, Mapping[str, Any]] = {}
        for job in jobs:
            uses = job.get("uses", [])
            use_slots = {use.get("source_slot_id") for use in uses}
            _require(
                len(use_slots) == 1,
                f"{row['row_id']} RIR job mixes source slots",
            )
            slot = str(next(iter(use_slots)))
            _require(slot not in jobs_by_slot, f"{row['row_id']} duplicate RIR slot")
            _require(
                all(use.get("episode_id") == row["episode_id"] for use in uses)
                and [use.get("frame_index") for use in uses]
                == list(range(0, FRAME_COUNT, 3)),
                f"{row['row_id']} RIR uses do not bind the exact row",
            )
            _require(
                max(
                    abs(float(job["source_position_m"][axis]) - float(expected_emitter_by_slot[slot][axis]))
                    for axis in range(3)
                )
                <= 1e-9,
                f"{row['row_id']} RIR source position drift",
            )
            jobs_by_slot[slot] = job
        _require(
            set(jobs_by_slot) == {"source1", "source2"},
            f"{row['row_id']} exact two-endpoint RIR closure failed",
        )
        states = {job["acoustic_state_sha256"] for job in jobs}
        _require(len(states) == 2, f"{row['row_id']} RIR states are not distinct")
        _require(not global_states.intersection(states), f"{row['row_id']} reused another row RIR state")
        global_states.update(states)
        _require(cache_receipt.get("status") == "pass", f"{row['row_id']} RIR cache failed")
        _require(cache_receipt.get("compute_device") == "CPU", f"{row['row_id']} RIR cache was not CPU")
        _require(cache_receipt.get("full_plan_complete") is True, f"{row['row_id']} RIR cache incomplete")
        _require(cache_receipt.get("selected_job_count") == 2, f"{row['row_id']} RIR cache count drift")
        _require(
            cache_request.get("plan", {}).get("path")
            == str((plan_root / "rir_job_plan.json").resolve())
            and cache_request.get("plan", {}).get("sha256")
            == _sha256(plan_root / "rir_job_plan.json"),
            f"{row['row_id']} RIR cache does not bind the exact plan",
        )
        _require(
            cache_request.get("request_identity_sha256")
            == cache_receipt.get("request_identity_sha256")
            == cache_index.get("request_identity_sha256"),
            f"{row['row_id']} RIR cache request identity drift",
        )
        _require(
            {
                entry.get("acoustic_state_sha256")
                for entry in cache_index.get("entries", [])
            }
            == states,
            f"{row['row_id']} RIR cache state coverage drift",
        )
        _require(binaural_delivery.get("status") == "pass", f"{row['row_id']} binaural failed")
        _require(binaural_delivery.get("sample_count") == 1, f"{row['row_id']} binaural sample count drift")
        _require(binaural_delivery.get("both_sources_active") is False, f"{row['row_id']} distractor became active")
        sample_rows = samples.get("samples", [])
        _require(len(sample_rows) == 1, f"{row['row_id']} samples row count drift")
        sample = sample_rows[0]
        _require(sample.get("episode_id") == row["episode_id"], f"{row['row_id']} episode drift")
        _require(
            binaural_delivery.get("sensor_rig_trajectory", {}).get(
                "content_sha256"
            )
            == sensor_binding["content_sha256"],
            f"{row['row_id']} binaural sensor binding drift",
        )
        _require(
            sample.get("acoustic_selection_binding_sha256")
            == cache_receipt.get("acoustic_selection_binding_sha256"),
            f"{row['row_id']} binaural/cache selection drift",
        )
        _require(
            sample.get("audio_program_binding", {})
            .get("audio_program_ref", {})
            .get("program_content_sha256")
            == audio_program["program_content_sha256"],
            f"{row['row_id']} binaural audio-program drift",
        )
        endpoint_assets = {
            endpoint["binding"]["entity_instance_id"]: endpoint["binding"][
                "entity_asset_id"
            ]
            for endpoint in endpoints["source_endpoints"]
        }
        _require(
            sample.get("asset_ids_by_source_slot") == endpoint_assets,
            f"{row['row_id']} binaural runtime-asset binding drift",
        )
        _require(
            [asset["sound_asset_id"] for asset in sound_registry["sound_assets"]]
            == [row["target_sound_asset_id"]]
            and audio_binding.get("target_sound_asset_id")
            == row["target_sound_asset_id"],
            f"{row['row_id']} target sound binding drift",
        )
        audio = sample["audio"]
        _require(audio.get("channel_count") == 2, f"{row['row_id']} output is not binaural")
        _require(audio.get("sample_rate_hz") == SAMPLE_RATE_HZ, f"{row['row_id']} sample rate drift")
        _require(audio.get("sample_count") == SAMPLE_COUNT, f"{row['row_id']} is not five seconds")
        stems = audio["stems"]
        _require(stems["source1"]["peak_absolute"] > 0.0, f"{row['row_id']} target stem is silent")
        _require(stems["source2"]["peak_absolute"] == 0.0, f"{row['row_id']} distractor stem is active")
        activity = sample["source_activity_summary"]
        _require(activity["active_source_slots"] == ["source1"], f"{row['row_id']} active slots drift")
        _require(activity["silent_source_slots"] == ["source2"], f"{row['row_id']} silent slots drift")
        _require(activity["active_sample_count_by_source_slot"]["source1"] == recipe_preflight["target_event_sample_window"][1] - recipe_preflight["target_event_sample_window"][0], f"{row['row_id']} active sample duration drift")
        mixture = audio["mixture"]
        mixture_path = binaural_root / "audio/binaural" / mixture["path"]
        _require(
            mixture_path.is_file()
            and _sha256(mixture_path) == mixture["audio_sha256"],
            f"{row['row_id']} binaural WAV artifact drift",
        )
        wav_header = _wave_header(mixture_path)
        _require(
            wav_header["channel_count"] == 2
            and wav_header["sample_rate_hz"] == SAMPLE_RATE_HZ
            and wav_header["sample_count"] == SAMPLE_COUNT,
            f"{row['row_id']} binaural WAV header drift",
        )
        ready_request = deepcopy(blocked_request)
        ready_request["status"] = "ready_for_native_sparse"
        ready_request["audio_wav"] = str(mixture_path.resolve())
        ready_request["audio_record"] = {
            "path": str(mixture_path.resolve()),
            "sha256": mixture["audio_sha256"],
            "channel_count": 2,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sample_count": SAMPLE_COUNT,
        }
        ready_request["recipe_identity_sha256"] = row[
            "recipe_identity_sha256"
        ]
        ready_request["cpu_acoustic_evidence"] = {
            "exact_rir_plan": _record(
                "exact_rir_plan", plan_root / "rir_job_plan.json"
            ),
            "rir_cache": _record("rir_cache_receipt", cache_root / "receipt.json"),
            "binaural_delivery": _record(
                "binaural_delivery", binaural_root / "delivery.json"
            ),
        }
        row_output = output / row["row_id"]
        row_output.mkdir()
        ready_path = row_output / "sparse_native_gate_request.ready.json"
        write_json(ready_path, ready_request)
        rows.append(
            {
                "row_id": row["row_id"],
                "episode_id": row["episode_id"],
                "target_sound_asset_id": row["target_sound_asset_id"],
                "target_event_frame_window_inclusive": row[
                    "target_event_frame_window_inclusive"
                ],
                "exact_rir_job_count": 2,
                "binaural_sample_count": 80000,
                "source1_peak_absolute": stems["source1"]["peak_absolute"],
                "source2_peak_absolute": stems["source2"]["peak_absolute"],
                "ready_capture_request": str(ready_path.resolve()),
                "status": "pass_cpu_ready_for_f15_sparse",
            }
        )
    _require(len(global_states) == 14, "cross-row exact RIR independence failed")
    delivery = {
        "schema": DELIVERY_SCHEMA,
        "status": "pass_cpu_rows2_to8_ready_for_sequential_f15_sparse",
        "claim_boundary": "CPU acoustics and requests only; no rows2-8 native capture or formal scene claimed",
        "row_count": 7,
        "exact_rir_job_count": 14,
        "cross_row_rir_reuse_count": 0,
        "rows": rows,
        "retained_row1_canary": manifest["retained_row1_canary"],
        "gpu_executed": False,
        "formal_scene_count": 0,
    }
    delivery_path = output / "delivery.json"
    write_json(delivery_path, delivery)
    return delivery_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--plan",
        type=Path,
        default=REPOSITORY / "examples/qa/native_strict_two_human_expansion_v1.json",
    )
    prepare_parser.add_argument("--cpu-preflight", type=Path, required=True)
    prepare_parser.add_argument(
        "--runtime-registry",
        type=Path,
        default=REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json",
    )
    prepare_parser.add_argument("--source-suite", type=Path, required=True)
    prepare_parser.add_argument("--controlled-registry", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--batch-root", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare(
            plan_path=args.plan.resolve(),
            cpu_preflight_path=args.cpu_preflight.resolve(),
            registry_path=args.runtime_registry.resolve(),
            source_suite_path=args.source_suite.resolve(),
            controlled_registry_path=args.controlled_registry.resolve(),
            output=args.output.resolve(),
        )
    else:
        result = finalize(
            batch_root=args.batch_root.resolve(),
            output=args.output.resolve(),
        )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

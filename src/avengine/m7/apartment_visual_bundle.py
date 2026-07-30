"""Build generic source1/source2 Timeline records for Apartment UE rendering."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256
from avengine.m5_1.mixed_capture import trajectory_world_matrices
from avengine.m6.audio_program import (
    AudioProgramError,
    CompiledAudioProgram,
    compile_audio_program,
)
from avengine.optional_backends.spear_replicacad_execution import (
    _rotation_matrix_to_xyzw,
)
from avengine.runtime_profiles import (
    load_default_source_asset_runtime_registry,
    resolve_source_asset_alias,
    source_timeline_profiles,
)
from avengine.m7.sensor_rig import (
    M7SensorRigError,
    resolve_m7_sensor_rig_trajectory,
)
from avengine.sensor_rig_trajectory import validate_sensor_rig_trajectory


FRAME_COUNT = 75
FRAME_RATE_HZ = 15
SAMPLE_RATE_HZ = 16_000
SAMPLE_COUNT = 80_000
TICKS_PER_FRAME = 3_200
SOURCE_SLOTS = ("source1", "source2")
VOCAL_SOUND_CLASSES = frozenset(
    {"animal_vocalization", "human_speech"}
)

DEFAULT_SOURCE_ASSET_RUNTIME_REGISTRY = (
    load_default_source_asset_runtime_registry()
)
ASSET_VISUAL_PROFILES: Mapping[str, Mapping[str, Any]] = (
    source_timeline_profiles(DEFAULT_SOURCE_ASSET_RUNTIME_REGISTRY)
)

# Compatibility names are resolved from editable registry aliases. They do not
# define the supported asset set and may be changed without editing Python.
HUMAN_ASSET_ID = str(
    resolve_source_asset_alias(
        DEFAULT_SOURCE_ASSET_RUNTIME_REGISTRY, "legacy_human"
    )["asset_id"]
)
BORDER_COLLIE_ASSET_ID = str(
    resolve_source_asset_alias(
        DEFAULT_SOURCE_ASSET_RUNTIME_REGISTRY, "current_generated_dog"
    )["asset_id"]
)
CAT_ASSET_ID = str(
    resolve_source_asset_alias(
        DEFAULT_SOURCE_ASSET_RUNTIME_REGISTRY, "current_generated_cat"
    )["asset_id"]
)


class ApartmentVisualBundleError(ValueError):
    """An asset-bound route cannot become an exact two-actor Timeline."""


def _compile_program_projection(
    materialized_audio_program: Mapping[str, Any],
    endpoint_to_source_slot: Mapping[str, str],
) -> tuple[CompiledAudioProgram, dict[str, str]]:
    if not isinstance(materialized_audio_program, Mapping):
        raise ApartmentVisualBundleError(
            "materialized_audio_program must be an M6 AudioProgram object"
        )
    timeline = materialized_audio_program.get("timeline")
    frame_count = (
        timeline.get("frame_count") if isinstance(timeline, Mapping) else None
    )
    sample_count = (
        timeline.get("sample_count") if isinstance(timeline, Mapping) else None
    )
    if frame_count != FRAME_COUNT or sample_count != SAMPLE_COUNT:
        raise ApartmentVisualBundleError(
            "materialized_audio_program must use exactly 75 frames and 80000 samples"
        )
    try:
        compiled = compile_audio_program(materialized_audio_program)
    except AudioProgramError as error:
        raise ApartmentVisualBundleError(
            f"materialized_audio_program is invalid: {error}"
        ) from error

    if len(compiled.candidate_source_endpoint_ids) != len(SOURCE_SLOTS):
        raise ApartmentVisualBundleError(
            "materialized_audio_program must declare exactly two candidate endpoints"
        )
    if not isinstance(endpoint_to_source_slot, Mapping):
        raise ApartmentVisualBundleError(
            "endpoint_to_source_slot must be an endpoint-to-source-slot mapping"
        )
    mapping = dict(endpoint_to_source_slot)
    candidates = set(compiled.candidate_source_endpoint_ids)
    mapped_slots = list(mapping.values())
    if (
        set(mapping) != candidates
        or any(slot not in SOURCE_SLOTS for slot in mapped_slots)
        or len(set(mapped_slots)) != len(mapping)
        or set(mapped_slots) != set(SOURCE_SLOTS)
    ):
        raise ApartmentVisualBundleError(
            "endpoint_to_source_slot must be a bijection from the two program "
            "candidate endpoints onto source1/source2"
        )
    return compiled, mapping


def _optional_program_projection(
    materialized_audio_program: Mapping[str, Any] | None,
    endpoint_to_source_slot: Mapping[str, str] | None,
) -> tuple[CompiledAudioProgram, dict[str, str]] | None:
    if materialized_audio_program is None and endpoint_to_source_slot is None:
        return None
    if materialized_audio_program is None or endpoint_to_source_slot is None:
        raise ApartmentVisualBundleError(
            "materialized_audio_program and endpoint_to_source_slot must be "
            "provided together"
        )
    return _compile_program_projection(
        materialized_audio_program, endpoint_to_source_slot
    )


def _source_activity_by_frame(
    compiled: CompiledAudioProgram,
    endpoint_to_source_slot: Mapping[str, str],
    *,
    active_event_ids: frozenset[str] | None = None,
) -> dict[str, np.ndarray]:
    activity = {
        slot: np.zeros(FRAME_COUNT, dtype=np.bool_) for slot in SOURCE_SLOTS
    }
    for frame_index in range(FRAME_COUNT):
        current = compiled.current_event_by_source(frame_index)
        for endpoint_id in compiled.candidate_source_endpoint_ids:
            slot = endpoint_to_source_slot[endpoint_id]
            event = current[endpoint_id]
            activity[slot][frame_index] = (
                event is not None
                and (
                    active_event_ids is None
                    or event in active_event_ids
                )
            )
    return activity


def _program_event_semantics(
    program_projection: tuple[CompiledAudioProgram, dict[str, str]] | None,
    semantic_sound_class_by_event_id: Mapping[str, str] | None,
) -> dict[str, str]:
    if program_projection is None:
        if semantic_sound_class_by_event_id is not None:
            raise ApartmentVisualBundleError(
                "event sound semantics require a materialized AudioProgram"
            )
        return {}
    if not isinstance(semantic_sound_class_by_event_id, Mapping):
        raise ApartmentVisualBundleError(
            "AudioProgram visual projection requires event sound semantics"
        )
    semantics = dict(semantic_sound_class_by_event_id)
    event_ids = {event.event_id for event in program_projection[0].events}
    if (
        set(semantics) != event_ids
        or any(
            not isinstance(sound_class, str) or not sound_class
            for sound_class in semantics.values()
        )
    ):
        raise ApartmentVisualBundleError(
            "event sound semantics must exactly cover the AudioProgram events"
        )
    return semantics


def program_source_activity_by_frame(
    materialized_audio_program: Mapping[str, Any],
    endpoint_to_source_slot: Mapping[str, str],
) -> dict[str, np.ndarray]:
    """Project M6 event windows to source-slot activity at each frame start."""

    compiled, mapping = _compile_program_projection(
        materialized_audio_program, endpoint_to_source_slot
    )
    return _source_activity_by_frame(compiled, mapping)


def binding_assets_by_episode(
    report: Mapping[str, Any],
    *,
    source_profiles: Mapping[str, Mapping[str, Any]] = ASSET_VISUAL_PROFILES,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Return exact source-slot binding records keyed by output episode ID."""

    if report.get("status") != "pass" or not isinstance(report.get("scenarios"), list):
        raise ApartmentVisualBundleError("asset-emitter binding report is invalid")
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    for scenario in report["scenarios"]:
        if not isinstance(scenario, Mapping):
            raise ApartmentVisualBundleError("asset-emitter scenario is invalid")
        episode_id = scenario.get("output_episode_id")
        binding_report = scenario.get("binding_report")
        bindings = (
            binding_report.get("bindings")
            if isinstance(binding_report, Mapping)
            else None
        )
        if not isinstance(episode_id, str) or not isinstance(bindings, list):
            raise ApartmentVisualBundleError("asset-emitter scenario is incomplete")
        by_slot: dict[str, Mapping[str, Any]] = {}
        for binding in bindings:
            slot = binding.get("source_slot_id") if isinstance(binding, Mapping) else None
            asset_id = binding.get("asset_id") if isinstance(binding, Mapping) else None
            if (
                slot not in {"source1", "source2"}
                or slot in by_slot
                or asset_id not in source_profiles
            ):
                raise ApartmentVisualBundleError(
                    f"episode {episode_id!r} has an unsupported source binding"
                )
            declared_revision = binding.get("asset_revision")
            if (
                declared_revision is not None
                and declared_revision != source_profiles[asset_id]["revision"]
            ):
                raise ApartmentVisualBundleError(
                    f"episode {episode_id!r} source asset revision differs "
                    "from the selected runtime registry"
                )
            by_slot[slot] = deepcopy(dict(binding))
        if set(by_slot) != {"source1", "source2"} or episode_id in result:
            raise ApartmentVisualBundleError(
                f"episode {episode_id!r} does not close over two source slots"
            )
        result[episode_id] = by_slot
    return result


def _root_paths(episode: Mapping[str, Any]) -> dict[str, np.ndarray]:
    raw = episode.get("source_root_paths_m")
    if not isinstance(raw, Mapping) or set(raw) != {"source1", "source2"}:
        raise ApartmentVisualBundleError("episode root paths do not close over source slots")
    result = {}
    for slot in ("source1", "source2"):
        points = np.asarray(raw[slot], dtype=np.float64)
        if points.shape != (FRAME_COUNT, 3) or not np.all(np.isfinite(points)):
            raise ApartmentVisualBundleError(f"{slot} root path must be finite [75,3]")
        result[slot] = np.ascontiguousarray(points)
    return result


def _fallback_forward(
    root_path: np.ndarray, listener_position_m: Sequence[float]
) -> tuple[float, float]:
    listener = np.asarray(listener_position_m, dtype=np.float64)
    delta = listener[[0, 2]] - root_path[0, [0, 2]]
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-12:
        return (0.0, -1.0)
    return (float(delta[0] / norm), float(delta[1] / norm))


def build_timeline(
    *,
    episode: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, Any]],
    listener_position_m: Sequence[float],
    listener_yaw_deg: float = 0.0,
    sensor_rig_trajectory: Mapping[str, Any] | None = None,
    source_profiles: Mapping[str, Mapping[str, Any]] = ASSET_VISUAL_PROFILES,
    materialized_audio_program: Mapping[str, Any] | None = None,
    endpoint_to_source_slot: Mapping[str, str] | None = None,
    semantic_sound_class_by_event_id: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build one exact Timeline with frame-bound rig poses and actor headings."""

    try:
        rig_trajectory = resolve_m7_sensor_rig_trajectory(
            sensor_rig_trajectory=sensor_rig_trajectory,
            listener_position_m=listener_position_m,
            listener_yaw_deg=listener_yaw_deg,
        )
    except M7SensorRigError as error:
        raise ApartmentVisualBundleError(str(error)) from error
    rig_frames = rig_trajectory["frames"]
    effective_listener_position = rig_frames[0]["world_from_rig"][
        "translation_m"
    ]
    program_projection = _optional_program_projection(
        materialized_audio_program, endpoint_to_source_slot
    )
    event_semantics = _program_event_semantics(
        program_projection,
        semantic_sound_class_by_event_id,
    )
    program_activity = (
        None
        if program_projection is None
        else _source_activity_by_frame(
            *program_projection,
            active_event_ids=frozenset(
                event_id
                for event_id, sound_class in event_semantics.items()
                if sound_class in VOCAL_SOUND_CLASSES
            ),
        )
    )
    root_paths = _root_paths(episode)
    statistics = episode.get("statistics")
    if not isinstance(statistics, Mapping):
        raise ApartmentVisualBundleError("episode statistics are missing")
    actors = []
    matrices: dict[str, np.ndarray] = {}
    headings: dict[str, np.ndarray] = {}
    for slot in ("source1", "source2"):
        asset_id = bindings[slot]["asset_id"]
        try:
            profile = source_profiles[asset_id]
        except KeyError as error:
            raise ApartmentVisualBundleError(
                f"{slot} selects unregistered source asset {asset_id!r}"
            ) from error
        actor_id = f"{slot}_actor"
        actors.append(
            {
                "actor_id": actor_id,
                "asset_id": asset_id,
                "template_id": profile["template_id"],
                "body_plan_id": profile["body_plan_id"],
            }
        )
        axis = np.asarray(profile["local_anatomical_forward_axis"], dtype=np.float64)
        matrix = trajectory_world_matrices(
            root_paths[slot],
            local_forward_axis=axis,
            fallback_forward_xz=_fallback_forward(
                root_paths[slot], effective_listener_position
            ),
        )
        matrices[slot] = matrix
        world = np.einsum("nij,j->ni", matrix[:, :3, :3], axis)
        horizontal = world[:, (0, 2)]
        horizontal /= np.linalg.norm(horizontal, axis=1)[:, None]
        headings[slot] = np.ascontiguousarray(horizontal)

    frames = []
    for frame_index in range(FRAME_COUNT):
        actor_states = []
        for slot in ("source1", "source2"):
            asset_id = bindings[slot]["asset_id"]
            profile = source_profiles[asset_id]
            slot_stats = statistics.get(slot)
            motion = slot_stats.get("motion") if isinstance(slot_stats, Mapping) else None
            if motion not in {"static", "moving"}:
                raise ApartmentVisualBundleError(f"{slot} motion is invalid")
            moving = motion == "moving"
            period = int(profile["walk_phase_period_frames"])
            actor_states.append(
                {
                    "actor_id": f"{slot}_actor",
                    "action_id": (
                        profile["walking_action_id"]
                        if moving
                        else profile["idle_action_id"]
                    ),
                    "action_phase": (frame_index % period) / period if moving else 0.0,
                    "action_time_ticks": frame_index * TICKS_PER_FRAME,
                    "contacts": {},
                    "mouth_state": {
                        "open_ratio": 0.0,
                        "vocalizing": (
                            False
                            if program_activity is None
                            else bool(program_activity[slot][frame_index])
                        ),
                    },
                    "root_transform": {
                        "translation_m": root_paths[slot][frame_index].tolist(),
                        "rotation_xyzw": _rotation_matrix_to_xyzw(
                            matrices[slot][frame_index, :3, :3]
                        ),
                        "scale": [1.0, 1.0, 1.0],
                    },
                }
            )
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * TICKS_PER_FRAME,
                "sample_start": int(round(frame_index * SAMPLE_RATE_HZ / FRAME_RATE_HZ)),
                "sample_end": int(
                    round((frame_index + 1) * SAMPLE_RATE_HZ / FRAME_RATE_HZ)
                ),
                "actor_states": actor_states,
                "view_pose_hashes": {
                    "view0": rig_frames[frame_index]["pose_hash"]
                },
            }
        )
    audio_events = (
        [
            {
                "event_id": f"{slot}_full_duration_vocalization",
                "actor_id": f"{slot}_actor",
                "emitter_bone": bindings[slot]["semantic_anchor_id"],
                "event_type": "vocalization",
                "start_sample": 0,
                "end_sample": SAMPLE_COUNT,
                "semantic_sync_required": False,
            }
            for slot in SOURCE_SLOTS
        ]
        if program_projection is None
        else [
            {
                "event_id": event.event_id,
                "actor_id": (
                    f"{program_projection[1][event.source_endpoint_id]}_actor"
                ),
                "emitter_bone": bindings[
                    program_projection[1][event.source_endpoint_id]
                ]["semantic_anchor_id"],
                "event_type": (
                    "vocalization"
                    if event_semantics[event.event_id]
                    in VOCAL_SOUND_CLASSES
                    else "other"
                ),
                "start_sample": event.start_sample,
                "end_sample": event.end_sample_exclusive,
                "semantic_sync_required": True,
            }
            for event in program_projection[0].events
        ]
    )
    return (
        {
            "schema": "avengine_authoritative_timeline_v2",
            "time_base_hz": 48_000,
            "duration_ticks": FRAME_COUNT * TICKS_PER_FRAME,
            "video": {
                "fps_num": FRAME_RATE_HZ,
                "fps_den": 1,
                "frame_count": FRAME_COUNT,
                "ticks_per_frame": TICKS_PER_FRAME,
                "view_ids": ["view0"],
            },
            "audio": {
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "sample_count": SAMPLE_COUNT,
                "channel_count": 2,
                "ticks_per_sample": 3,
            },
            "actors": actors,
            "audio_events": audio_events,
            "frames": frames,
        },
        headings,
    )


def build_source_manifest(
    *,
    episode_id: str,
    episode: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, Any]],
    source_profiles: Mapping[str, Mapping[str, Any]] = ASSET_VISUAL_PROFILES,
    materialized_audio_program: Mapping[str, Any] | None = None,
    endpoint_to_source_slot: Mapping[str, str] | None = None,
    audio_program_variant_id: str | None = None,
    semantic_sound_class_by_event_id: Mapping[str, str] | None = None,
    sensor_rig_trajectory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    program_projection = _optional_program_projection(
        materialized_audio_program, endpoint_to_source_slot
    )
    event_semantics = _program_event_semantics(
        program_projection,
        semantic_sound_class_by_event_id,
    )
    if program_projection is None:
        if audio_program_variant_id is not None:
            raise ApartmentVisualBundleError(
                "audio_program_variant_id requires a materialized AudioProgram"
            )
        manifest_variant_id = "A"
    elif audio_program_variant_id not in {"A", "B"}:
        raise ApartmentVisualBundleError(
            "AudioProgram visual projection requires variant A or B"
        )
    else:
        manifest_variant_id = audio_program_variant_id
    slot_to_program_endpoint = (
        {}
        if program_projection is None
        else {
            slot: endpoint_id
            for endpoint_id, slot in program_projection[1].items()
        }
    )
    centers = episode.get("source_center_paths_m")
    if not isinstance(centers, Mapping) or set(centers) != {"source1", "source2"}:
        raise ApartmentVisualBundleError("episode source centers are incomplete")
    sources = []
    for slot in ("source1", "source2"):
        points = np.asarray(centers[slot], dtype=np.float64)
        if points.shape != (FRAME_COUNT, 3) or not np.all(np.isfinite(points)):
            raise ApartmentVisualBundleError(f"{slot} centers must be finite [75,3]")
        asset_id = bindings[slot]["asset_id"]
        try:
            profile = source_profiles[asset_id]
        except KeyError as error:
            raise ApartmentVisualBundleError(
                f"{slot} selects unregistered source asset {asset_id!r}"
            ) from error
        endpoint_id = f"{slot}_emitter"
        program_endpoint_id = slot_to_program_endpoint.get(slot)
        sources.append(
            {
                "source_endpoint_id": endpoint_id,
                "source_slot_id": slot,
                "activation": (
                    "active"
                    if program_projection is None
                    or program_endpoint_id
                    in program_projection[0].active_source_endpoint_ids
                    else "persistent_silent"
                ),
                "visible_asset": {
                    "asset_id": asset_id,
                    "revision": profile["revision"],
                    "display_label": profile["display_label"],
                    "identity": deepcopy(dict(profile["identity"])),
                    "realized_attributes": deepcopy(
                        dict(profile["realized_attributes"])
                    ),
                },
                "endpoint": {
                    "source_endpoint_id": endpoint_id,
                    "revision": "m7_asset_bound_v1",
                    "admission_state": "research",
                    "persistent_when_silent": True,
                    "directivity_profile_id": "point_emitter_v1",
                    "source_visibility_mode": "visible_entity",
                    "binding": {
                        "kind": "entity_anchor",
                        "entity_instance_id": f"{slot}_actor",
                        "entity_asset_id": asset_id,
                        "entity_asset_revision": profile["revision"],
                        "emitter_anchor_id": bindings[slot]["semantic_anchor_id"],
                    },
                },
                "trajectory": {
                    "frame_count": FRAME_COUNT,
                    "position_authority": "asset_bound_trajectory_bank_source_center_path",
                    "positions_m": points.tolist(),
                },
            }
        )
    purpose = (
        "two_replaceable_sources_both_active"
        if program_projection is None
        else f"two_replaceable_sources_{program_projection[0].mode}"
    )
    events = (
        [
            {
                "event_id": f"{slot}_full_duration_vocalization",
                "source_endpoint_id": f"{slot}_emitter",
                "start_sample": 0,
                "end_sample_exclusive": SAMPLE_COUNT,
                "start_tick": 0,
                "end_tick_exclusive": SAMPLE_COUNT * 3,
            }
            for slot in SOURCE_SLOTS
        ]
        if program_projection is None
        else [
            {
                **deepcopy(dict(event)),
                "source_endpoint_id": (
                    f"{program_projection[1][event['source_endpoint_id']]}_emitter"
                ),
                "audio_program_source_endpoint_id": event["source_endpoint_id"],
                "semantic_sound_class": event_semantics[event["event_id"]],
                "event_type": (
                    "vocalization"
                    if event_semantics[event["event_id"]]
                    in VOCAL_SOUND_CLASSES
                    else "other"
                ),
            }
            for event in materialized_audio_program["events"]
        ]
    )
    listener = {
        "listener_id": "listener0",
        "camera_listener_colocated": True,
        "camera_listener_cooriented": True,
        "audio_visibility_policy": "360_degree_no_camera_fov_cutoff",
    }
    if sensor_rig_trajectory is not None:
        trajectory_errors = validate_sensor_rig_trajectory(
            sensor_rig_trajectory
        )
        if trajectory_errors:
            raise ApartmentVisualBundleError(
                "sensor_rig_trajectory is invalid: "
                + "; ".join(trajectory_errors)
            )
        listener["sensor_rig_trajectory"] = {
            "trajectory_id": sensor_rig_trajectory["trajectory_id"],
            "content_sha256": canonical_json_sha256(
                sensor_rig_trajectory
            ),
            "relative_path": "metadata/sensor_rig_trajectory.json",
        }
    result = {
        "schema": "avengine_m7_asset_bound_apartment_source_manifest_v1",
        "scenario_id": episode_id,
        "variant_id": manifest_variant_id,
        "purpose": purpose,
        "listener": listener,
        "room_policy": "fixed_scene_instance_no_furniture_mutation",
        "sources": sources,
        "events": events,
        "stem_policy": {
            "independent_binaural_stem_per_candidate_source": True,
            "mixture_is_exact_stem_sum": True,
            "normalization": False,
            "limiting": False,
        },
    }
    if program_projection is not None:
        result["audio_program"] = {
            "program_id": program_projection[0].program_id,
            "revision": program_projection[0].revision,
            "mode": program_projection[0].mode,
            "variant_id": manifest_variant_id,
            "program_content_sha256": materialized_audio_program[
                "program_content_sha256"
            ],
            "endpoint_to_source_slot": deepcopy(dict(program_projection[1])),
        }
    return result


def build_flags() -> dict[str, Any]:
    assessment = {
        "status": "not_evaluated",
        "value": None,
        "reason": "Batch visual compilation does not recompute semantic flags.",
        "reason_code": "batch_visual_flag_gate_not_run",
        "evidence": [],
    }
    return {
        "schema": "avengine_m5_1_flag_report_v1",
        "source_flags": {
            f"{slot}_emitter": {"both_sources_active": deepcopy(assessment)}
            for slot in ("source1", "source2")
        },
        "clip_flags": {"both_sources_active": deepcopy(assessment)},
    }


def build_qualification(
    *,
    template: Mapping[str, Any],
    episode_ids: Sequence[str],
    episodes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result = deepcopy(dict(template))
    result["schema"] = "avengine_m7_asset_bound_apartment_qualification_v1"
    result["status"] = "pass"
    result["runtime_backend"] = "asset_bound_trajectory_bank_and_cached_native_rlr"
    result["claim_boundary"] = "source-center placement only; no body-volume claim"
    result["source_center_gate"] = {
        "schema": "avengine_m6x_source_center_obstacle_gate_v2",
        "status": "pass",
        "semantics": "selected asset-bound source-center gate",
        "full_body_collision_claim": False,
        "failed_source_frame_indices": {},
        "sources": {
            f"{slot}_emitter": {
                "status": "pass",
                "failed_frame_indices": [],
                "frame_count": len(episode_ids) * FRAME_COUNT,
                "source_slot_id": slot,
                "authority": "asset-bound trajectory bank center gate",
            }
            for slot in ("source1", "source2")
        },
        "selected_episode_statistics": {
            episode_id: deepcopy(dict(episodes[episode_id]["statistics"]))
            for episode_id in episode_ids
        },
    }
    return result


__all__ = [
    "ASSET_VISUAL_PROFILES",
    "ApartmentVisualBundleError",
    "DEFAULT_SOURCE_ASSET_RUNTIME_REGISTRY",
    "binding_assets_by_episode",
    "build_flags",
    "build_qualification",
    "build_source_manifest",
    "build_timeline",
    "program_source_activity_by_frame",
    "resolve_m7_sensor_rig_trajectory",
]

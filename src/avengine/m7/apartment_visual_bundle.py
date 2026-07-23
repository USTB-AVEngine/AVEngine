"""Build generic source1/source2 Timeline records for Apartment UE rendering."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.m5_1.mixed_capture import trajectory_world_matrices
from avengine.optional_backends.spear_replicacad_execution import (
    _rotation_matrix_to_xyzw,
)


FRAME_COUNT = 75
FRAME_RATE_HZ = 15
SAMPLE_RATE_HZ = 16_000
SAMPLE_COUNT = 80_000
TICKS_PER_FRAME = 3_200

HUMAN_ASSET_ID = "rocketbox_human_male_adult_01_m5_1_candidate"
BORDER_COLLIE_ASSET_ID = (
    "generated_border_collie_black_white_medium_standard_adult_research_v1"
)
CAT_ASSET_ID = "generated_abyssinian_ruddy_medium_standard_adult_research_v1"

ASSET_VISUAL_PROFILES: Mapping[str, Mapping[str, Any]] = {
    HUMAN_ASSET_ID: {
        "template_id": "rocketbox_human_male_adult_01",
        "body_plan_id": "biped_human",
        "local_anatomical_forward_axis": (0.0, 0.0, 1.0),
        "walk_phase_period_frames": 16,
        "display_label": "Human",
    },
    BORDER_COLLIE_ASSET_ID: {
        "template_id": "generated_border_collie_target_native_v1",
        "body_plan_id": "quadruped_canine",
        "local_anatomical_forward_axis": (1.0, 0.0, 0.0),
        "walk_phase_period_frames": 25,
        "display_label": "Border Collie",
    },
    CAT_ASSET_ID: {
        "template_id": "generated_abyssinian_target_native_v1",
        "body_plan_id": "quadruped_mammal_felid_v1",
        "local_anatomical_forward_axis": (1.0, 0.0, 0.0),
        "walk_phase_period_frames": 25,
        "display_label": "Abyssinian",
    },
}


class ApartmentVisualBundleError(ValueError):
    """An asset-bound route cannot become an exact two-actor Timeline."""


def binding_assets_by_episode(
    report: Mapping[str, Any],
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
                or asset_id not in ASSET_VISUAL_PROFILES
            ):
                raise ApartmentVisualBundleError(
                    f"episode {episode_id!r} has an unsupported source binding"
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
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build one exact 75-frame two-actor Timeline and explicit headings."""

    root_paths = _root_paths(episode)
    statistics = episode.get("statistics")
    if not isinstance(statistics, Mapping):
        raise ApartmentVisualBundleError("episode statistics are missing")
    actors = []
    matrices: dict[str, np.ndarray] = {}
    headings: dict[str, np.ndarray] = {}
    for slot in ("source1", "source2"):
        asset_id = bindings[slot]["asset_id"]
        profile = ASSET_VISUAL_PROFILES[asset_id]
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
                root_paths[slot], listener_position_m
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
            profile = ASSET_VISUAL_PROFILES[asset_id]
            slot_stats = statistics.get(slot)
            motion = slot_stats.get("motion") if isinstance(slot_stats, Mapping) else None
            if motion not in {"static", "moving"}:
                raise ApartmentVisualBundleError(f"{slot} motion is invalid")
            moving = motion == "moving"
            period = int(profile["walk_phase_period_frames"])
            actor_states.append(
                {
                    "actor_id": f"{slot}_actor",
                    "action_id": "walk" if moving else "idle",
                    "action_phase": (frame_index % period) / period if moving else 0.0,
                    "action_time_ticks": frame_index * TICKS_PER_FRAME,
                    "contacts": {},
                    "mouth_state": {"open_ratio": 0.0, "vocalizing": False},
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
                "view_pose_hashes": {},
            }
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
            "audio_events": [
                {
                    "event_id": f"{slot}_full_duration_vocalization",
                    "actor_id": f"{slot}_actor",
                    "emitter_bone": bindings[slot]["semantic_anchor_id"],
                    "event_type": "vocalization",
                    "start_sample": 0,
                    "end_sample": SAMPLE_COUNT,
                    "semantic_sync_required": False,
                }
                for slot in ("source1", "source2")
            ],
            "frames": frames,
        },
        headings,
    )


def build_source_manifest(
    *,
    episode_id: str,
    episode: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    centers = episode.get("source_center_paths_m")
    if not isinstance(centers, Mapping) or set(centers) != {"source1", "source2"}:
        raise ApartmentVisualBundleError("episode source centers are incomplete")
    sources = []
    for slot in ("source1", "source2"):
        points = np.asarray(centers[slot], dtype=np.float64)
        if points.shape != (FRAME_COUNT, 3) or not np.all(np.isfinite(points)):
            raise ApartmentVisualBundleError(f"{slot} centers must be finite [75,3]")
        asset_id = bindings[slot]["asset_id"]
        endpoint_id = f"{slot}_emitter"
        sources.append(
            {
                "source_endpoint_id": endpoint_id,
                "source_slot_id": slot,
                "activation": "active",
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
                        "entity_asset_revision": "asset_bound_research_v1",
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
    return {
        "schema": "avengine_m7_asset_bound_apartment_source_manifest_v1",
        "scenario_id": episode_id,
        "variant_id": "A",
        "purpose": "two_replaceable_sources_both_active",
        "listener": {
            "listener_id": "listener0",
            "camera_listener_colocated": True,
            "camera_listener_cooriented": True,
            "audio_visibility_policy": "360_degree_no_camera_fov_cutoff",
        },
        "room_policy": "fixed_scene_instance_no_furniture_mutation",
        "sources": sources,
        "events": [
            {
                "event_id": f"{slot}_full_duration_vocalization",
                "source_endpoint_id": f"{slot}_emitter",
                "start_sample": 0,
                "end_sample_exclusive": SAMPLE_COUNT,
                "start_tick": 0,
                "end_tick_exclusive": SAMPLE_COUNT * 3,
            }
            for slot in ("source1", "source2")
        ],
        "stem_policy": {
            "independent_binaural_stem_per_candidate_source": True,
            "mixture_is_exact_stem_sum": True,
            "normalization": False,
            "limiting": False,
        },
    }


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
    "binding_assets_by_episode",
    "build_flags",
    "build_qualification",
    "build_source_manifest",
    "build_timeline",
]

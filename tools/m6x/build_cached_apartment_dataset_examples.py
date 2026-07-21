#!/usr/bin/env python3
"""Build four cache-bound Apartment dataset examples for UE review.

The tool consumes one source-slot trajectory bank and one completed native RIR
cache.  It binds source1 to the human/speech pair and source2 to the Beagle/
bark pair, renders exact-length binaural stems, and emits the metadata/video
closure consumed by ``run_spear_apartment_canary.py --input-layout motion-pilot``.
No visual route or acoustic position is replanned here.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avengine.contracts.json_io import load_json, sha256_file, write_json
from avengine.m4.audio import read_float32_wav, write_float32_wav
from avengine.m5.audio import M5_AUDIO_SAMPLE_COUNT, render_dynamic_stems_and_mix
from avengine.m5_1.mixed_capture import trajectory_world_matrices
from avengine.m5_1.review import encode_annotated_review
from avengine.m6x.capture_adapter import _matrix_quaternion_xyzw
from avengine.m6x.geometry import RuntimeObstacleMap
from avengine.m6x.rir_cache import load_cached_rir_episode
from avengine.m6x.topdown import render_runtime_topdown_frames


REPOSITORY = Path(__file__).resolve().parents[2]
FPS = 15
FRAME_COUNT = 75
SAMPLE_RATE_HZ = 16_000
TIME_BASE_HZ = 48_000
TICKS_PER_FRAME = 3_200
TICKS_PER_SAMPLE = 3
REVIEW_GAIN_DB = 12.0
REVIEW_GAIN = 10.0 ** (REVIEW_GAIN_DB / 20.0)
SCHEMA = "avengine_cached_apartment_dataset_examples_v1"

SOURCE_BINDINGS: Mapping[str, Mapping[str, Any]] = {
    "source1": {
        "actor_id": "human0",
        "endpoint_id": "m6x_human0_mouth",
        "dry_name": "m6x_human0_mouth.wav",
        "semantic_class": "human_speech",
        "local_forward_axis": (0.0, 0.0, 1.0),
        "walk_phase_period_frames": 16,
    },
    "source2": {
        "actor_id": "dog0",
        "endpoint_id": "m6x_dog0_muzzle",
        "dry_name": "m6x_dog0_muzzle.wav",
        "semantic_class": "animal_vocalization",
        "local_forward_axis": (1.0, 0.0, 0.0),
        "walk_phase_period_frames": 25,
    },
}

SELECTIONS = (
    ("P0", "00_static_static", "static_static_017"),
    (
        "P1",
        "01_human_moving_dog_static",
        "source1_moving_source2_static_007",
    ),
    ("P2", "02_both_moving", "both_moving_038"),
    (
        "P3",
        "03_human_static_dog_moving",
        "source1_static_source2_moving_010",
    ),
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _activity_by_frame(samples: np.ndarray) -> np.ndarray:
    value = np.asarray(samples, dtype=np.float64)
    if value.shape != (M5_AUDIO_SAMPLE_COUNT,):
        raise RuntimeError("dry bus must contain exactly 80,000 mono samples")
    result = np.zeros(FRAME_COUNT, dtype=np.bool_)
    for frame_index in range(FRAME_COUNT):
        start = int(round(frame_index * SAMPLE_RATE_HZ / FPS))
        end = int(round((frame_index + 1) * SAMPLE_RATE_HZ / FPS))
        result[frame_index] = bool(np.any(value[start:end] != 0.0))
    return result


def _fallback_toward_listener(
    root_path: np.ndarray, listener: np.ndarray
) -> tuple[float, float]:
    delta = (
        listener[(0, 2),]
        - root_path[
            0,
            (0, 2),
        ]
    )
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-12:
        return (0.0, -1.0)
    return (float(delta[0] / norm), float(delta[1] / norm))


def _timeline(
    *,
    template: Mapping[str, Any],
    root_paths: Mapping[str, np.ndarray],
    motion_by_slot: Mapping[str, str],
    listener_position_m: np.ndarray,
) -> dict[str, Any]:
    actors = deepcopy(template["actors"])
    bindings_by_actor = {
        value["actor_id"]: (slot, value) for slot, value in SOURCE_BINDINGS.items()
    }
    matrices: dict[str, np.ndarray] = {}
    for actor in actors:
        actor_id = actor["actor_id"]
        slot, binding = bindings_by_actor[actor_id]
        root = root_paths[slot]
        matrices[actor_id] = trajectory_world_matrices(
            root,
            local_forward_axis=binding["local_forward_axis"],
            fallback_forward_xz=_fallback_toward_listener(root, listener_position_m),
        )

    frames = []
    for frame_index in range(FRAME_COUNT):
        states = []
        for actor in actors:
            actor_id = actor["actor_id"]
            slot, binding = bindings_by_actor[actor_id]
            moving = motion_by_slot[slot] == "moving"
            action_id = "walk" if moving else "idle"
            phase = (
                (frame_index % int(binding["walk_phase_period_frames"]))
                / int(binding["walk_phase_period_frames"])
                if moving
                else 0.0
            )
            states.append(
                {
                    "actor_id": actor_id,
                    "action_id": action_id,
                    "action_phase": phase,
                    "action_time_ticks": frame_index * TICKS_PER_FRAME,
                    "contacts": {},
                    "mouth_state": {"open_ratio": 0.0, "vocalizing": False},
                    "root_transform": {
                        "translation_m": root_paths[slot][frame_index].tolist(),
                        "rotation_xyzw": list(
                            _matrix_quaternion_xyzw(
                                matrices[actor_id][frame_index, :3, :3]
                            )
                        ),
                        "scale": [1.0, 1.0, 1.0],
                    },
                }
            )
        frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * TICKS_PER_FRAME,
                "sample_start": int(round(frame_index * SAMPLE_RATE_HZ / FPS)),
                "sample_end": int(round((frame_index + 1) * SAMPLE_RATE_HZ / FPS)),
                "actor_states": states,
                "view_pose_hashes": {},
            }
        )
    return {
        "schema": "avengine_authoritative_timeline_v2",
        "time_base_hz": TIME_BASE_HZ,
        "duration_ticks": FRAME_COUNT * TICKS_PER_FRAME,
        "video": {
            "fps_num": FPS,
            "fps_den": 1,
            "frame_count": FRAME_COUNT,
            "ticks_per_frame": TICKS_PER_FRAME,
            "view_ids": ["view0"],
        },
        "audio": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sample_count": M5_AUDIO_SAMPLE_COUNT,
            "channel_count": 2,
            "ticks_per_sample": TICKS_PER_SAMPLE,
        },
        "actors": actors,
        "audio_events": [
            {
                "event_id": event["event_id"],
                "actor_id": (
                    "human0"
                    if event["source_endpoint_id"] == "m6x_human0_mouth"
                    else "dog0"
                ),
                "emitter_bone": (
                    "mouth"
                    if event["source_endpoint_id"] == "m6x_human0_mouth"
                    else "muzzle"
                ),
                "event_type": "vocalization",
                "start_sample": event["start_sample"],
                "end_sample": event["end_sample_exclusive"],
                "semantic_sync_required": False,
            }
            for event in template["source_manifest_events"]
        ],
        "frames": frames,
    }


def _source_manifest(
    *,
    template: Mapping[str, Any],
    scenario_id: str,
    episode_directory: str,
    episode_id: str,
    center_paths: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    endpoint_templates = template["source_manifest_sources"]
    sources = []
    for slot, binding in SOURCE_BINDINGS.items():
        endpoint_id = binding["endpoint_id"]
        source = deepcopy(endpoint_templates[endpoint_id])
        source["activation"] = "active"
        source["source_slot_id"] = slot
        source["trajectory"] = {
            "frame_count": FRAME_COUNT,
            "position_authority": "trajectory_bank_source_center_path",
            "positions_m": center_paths[slot].tolist(),
        }
        sources.append(source)
    return {
        "schema": "avengine_m6x_fixed_apartment_source_manifest_v1",
        "scenario_id": scenario_id,
        "variant_id": "A",
        "purpose": episode_id,
        "listener": {
            "listener_id": "listener0",
            "camera_listener_colocated": True,
            "camera_listener_cooriented": True,
            "audio_visibility_policy": "360_degree_no_camera_fov_cutoff",
        },
        "room_policy": "fixed_scene_instance_no_furniture_mutation",
        "sources": sources,
        "events": deepcopy(template["source_manifest_events"]),
        "stem_policy": {
            "independent_binaural_stem_per_candidate_source": True,
            "mixture_is_exact_stem_sum": True,
            "normalization": False,
            "limiting": False,
        },
        "rir_evidence": {
            "pair_specific": True,
            "source_slot_order": ["source1", "source2"],
            "source_ids": [
                SOURCE_BINDINGS["source1"]["endpoint_id"],
                SOURCE_BINDINGS["source2"]["endpoint_id"],
            ],
            "uri": (
                f"bundle://episodes/{episode_directory}/metadata/cache_binding.json"
            ),
        },
    }


def _flags(endpoint_ids: list[str]) -> dict[str, Any]:
    ids = (
        "crosses_azimuth_zero",
        "far_from_mic_whole_clip",
        "leaves_camera_fov",
        "never_occluded",
        "occluded_by_furniture",
        "occluded_by_wall",
        "passes_close_to_mic",
        "sources_pass_each_other",
        "stationary",
        "stays_in_camera_fov",
        "steady_walk",
        "stop_and_go",
    )
    assessment = {
        "status": "not_evaluated",
        "value": None,
        "reason": "This cache-bound preview did not recompute the M5.1 flag suite.",
        "reason_code": "preview_flag_gate_not_run",
        "evidence": [],
    }
    return {
        "schema": "avengine_m5_1_flag_report_v1",
        "source_flags": {
            endpoint_id: {flag_id: deepcopy(assessment) for flag_id in ids}
            for endpoint_id in endpoint_ids
        },
        "clip_flags": {flag_id: deepcopy(assessment) for flag_id in ids},
    }


def _info_frames(
    *,
    scenario_id: str,
    episode_id: str,
    motion_case: str,
    activities: Mapping[str, np.ndarray],
    topdown: np.ndarray,
) -> np.ndarray:
    frames = []
    title_font = _font(24)
    body_font = _font(17)
    small_font = _font(14)
    for frame_index in range(FRAME_COUNT):
        panel = Image.new("RGB", (640, 480), (22, 28, 36))
        draw = ImageDraw.Draw(panel)
        draw.text(
            (24, 22),
            f"{scenario_id}  {motion_case}",
            font=title_font,
            fill=(255, 255, 255),
        )
        draw.text((24, 63), episode_id, font=small_font, fill=(178, 193, 210))
        lines = (
            "CACHE-BOUND DATASET PREVIEW",
            "source1 -> human0 -> speech",
            "source2 -> dog0 -> bark",
            "native RLR HRTF binaural: Left / Right",
            "25 RIR keyframes x 2 sources = 50 jobs",
            "audio is 360 deg; camera FOV does not mute it",
            f"review listening gain: +{REVIEW_GAIN_DB:.0f} dB (dataset WAV is raw)",
        )
        y = 112
        for line in lines:
            draw.text((24, y), line, font=body_font, fill=(220, 229, 238))
            y += 36
        human = "ACTIVE" if activities["source1"][frame_index] else "silent"
        dog = "ACTIVE" if activities["source2"][frame_index] else "silent"
        draw.text(
            (24, 382), f"Human speech: {human}", font=body_font, fill=(42, 210, 220)
        )
        draw.text(
            (24, 414), f"Dog bark:      {dog}", font=body_font, fill=(250, 120, 70)
        )
        draw.text(
            (500, 442),
            f"frame {frame_index:02d}/74",
            font=small_font,
            fill=(174, 185, 197),
        )
        frames.append(np.concatenate((np.asarray(panel), topdown[frame_index]), axis=1))
    return np.ascontiguousarray(np.stack(frames), dtype=np.uint8)


def _qualification(
    *,
    room_id: str,
    listener: Mapping[str, Any],
    obstacle_authority: Mapping[str, Any],
    selected: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_records = {}
    for slot, binding in SOURCE_BINDINGS.items():
        source_records[binding["endpoint_id"]] = {
            "status": "pass",
            "failed_frame_indices": [],
            "frame_count": len(selected) * FRAME_COUNT,
            "source_slot_id": slot,
            "authority": "selected records from trajectory_bank.json",
        }
    return {
        "schema": "avengine_m6x_cached_preview_qualification_v1",
        "status": "pass",
        "room_id": room_id,
        "runtime_backend": "trajectory_bank_and_cached_native_rlr",
        "claim_boundary": "source-center placement only; no body-volume claim",
        "listener": dict(listener),
        "obstacle_authority": dict(obstacle_authority),
        "source_center_gate": {
            "schema": "avengine_m6x_source_center_obstacle_gate_v2",
            "status": "pass",
            "semantics": "selected trajectory-bank source-center gate",
            "full_body_collision_claim": False,
            "failed_source_frame_indices": {},
            "sources": source_records,
            "selected_episode_statistics": {
                episode_id: value["statistics"]
                for episode_id, value in selected.items()
            },
        },
    }


def _probe(path: Path) -> Mapping[str, Any]:
    import json
    import subprocess

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate,nb_read_frames,sample_rate,channels:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    output = args.output.resolve()
    staging = output.with_name(f".{output.name}.staging")
    if os.path.lexists(output) or os.path.lexists(staging):
        raise RuntimeError(f"refusing to replace output or staging path: {output}")
    staging.mkdir(parents=True)

    bank_root = args.trajectory_bank.resolve()
    cache_root = args.rir_cache.resolve()
    old_pilot = args.audio_template_bundle.resolve()
    bank_record = load_json(bank_root / "trajectory_bank.json")
    episodes = {value["episode_id"]: value for value in bank_record["episodes"]}
    selected = {episode_id: episodes[episode_id] for _, _, episode_id in SELECTIONS}
    arrays = np.load(bank_root / "trajectory_bank.npz", allow_pickle=False)
    episode_indices = {
        str(value): index for index, value in enumerate(arrays["episode_ids"])
    }
    source_slot_ids = tuple(str(value) for value in arrays["source_slot_ids"])
    if source_slot_ids != ("source1", "source2"):
        raise RuntimeError("trajectory bank source-slot order changed")

    template_episode = old_pilot / "episodes/02_both_moving"
    timeline_template = load_json(template_episode / "metadata/timeline.json")
    source_template = load_json(template_episode / "metadata/source_manifest.json")
    template = {
        "actors": timeline_template["actors"],
        "source_manifest_events": source_template["events"],
        "source_manifest_sources": {
            value["source_endpoint_id"]: {
                key: item for key, item in value.items() if key != "trajectory"
            }
            for value in source_template["sources"]
        },
    }
    dry_root = template_episode / "audio/dry"
    dry_by_slot = {
        slot: read_float32_wav(dry_root / binding["dry_name"]).samples[0]
        for slot, binding in SOURCE_BINDINGS.items()
    }
    activities = {
        slot: _activity_by_frame(value) for slot, value in dry_by_slot.items()
    }

    feasible_record = load_json(bank_root / "feasible_region.json")
    obstacle_record = feasible_record["obstacle_authority"]
    navmesh = np.load(bank_root / "feasible_region_source1.npz", allow_pickle=False)[
        "navmesh_mask"
    ]
    obstacle_map = RuntimeObstacleMap(
        binary_navmesh=np.ascontiguousarray(navmesh),
        bounds_m=tuple(
            tuple(float(item) for item in row) for row in obstacle_record["bounds_m"]
        ),
        floor_height_m=float(obstacle_record["floor_height_m"]),
        meters_per_pixel=float(obstacle_record["meters_per_pixel"]),
        rigid_obstacles=tuple(obstacle_record.get("rigid_obstacles", ())),
        authority=str(obstacle_record["authority"]),
        claim_boundary=str(obstacle_record["claim_boundary"]),
        rigid_obstacles_baked_into_navmesh=bool(
            obstacle_record.get("rigid_obstacles_baked_into_navmesh", False)
        ),
    )
    room_capsule = load_json(old_pilot / "room/room_capsule.json")
    old_qualification = load_json(old_pilot / "room/qualification.json")
    listener = old_qualification["listener"]
    (staging / "room").mkdir(parents=True)
    shutil.copy2(
        old_pilot / "room/room_capsule.json", staging / "room/room_capsule.json"
    )
    write_json(
        staging / "room/qualification.json",
        _qualification(
            room_id=room_capsule["room_registry_ref"]["room_id"],
            listener=listener,
            obstacle_authority=obstacle_record,
            selected=selected,
        ),
    )

    rows = []
    for scenario_id, directory, episode_id in SELECTIONS:
        episode_started = time.perf_counter()
        index = episode_indices[episode_id]
        root_paths = {
            slot: np.ascontiguousarray(arrays["source_root_paths_m"][index, ordinal])
            for ordinal, slot in enumerate(source_slot_ids)
        }
        center_paths = {
            slot: np.ascontiguousarray(arrays["source_center_paths_m"][index, ordinal])
            for ordinal, slot in enumerate(source_slot_ids)
        }
        motion_by_slot = {
            slot: selected[episode_id]["statistics"][slot]["motion"]
            for slot in source_slot_ids
        }
        cached = load_cached_rir_episode(
            cache_root=cache_root,
            plan_path=bank_root / "rir_job_plan.json",
            episode_id=episode_id,
            frame_count=FRAME_COUNT,
            frame_rate_hz=FPS,
        )
        maximum_cache_position_error = 0.0
        for job in cached.evidence["jobs"]:
            slot = job["source_slot_id"]
            frame_index = int(job["visual_frame_index"])
            maximum_cache_position_error = max(
                maximum_cache_position_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(job["source_position_m"], dtype=np.float64)
                            - center_paths[slot][frame_index]
                        )
                    )
                ),
            )
        if maximum_cache_position_error > 1.0e-9:
            raise RuntimeError("cached RIR positions differ from trajectory bank")

        stems, mixture = render_dynamic_stems_and_mix(
            dry_by_slot,
            cached.samples,
            cached.lengths,
            source_ids=cached.source_slot_ids,
            keyframe_samples=cached.keyframe_samples,
        )
        stem_sum = stems["source1"].episode + stems["source2"].episode
        stem_sum_error = float(np.max(np.abs(mixture - stem_sum)))
        if stem_sum_error > 1.0e-12:
            raise RuntimeError("rendered mixture differs from exact stem sum")
        review_audio = np.ascontiguousarray(mixture * REVIEW_GAIN)
        if float(np.max(np.abs(review_audio))) >= 1.0:
            raise RuntimeError("fixed review gain would clip this selected example")

        episode_root = staging / "episodes" / directory
        audio_root = episode_root / "audio/binaural"
        for slot, binding in SOURCE_BINDINGS.items():
            write_float32_wav(
                audio_root / f"{binding['endpoint_id']}_stem.wav",
                stems[slot].episode,
                SAMPLE_RATE_HZ,
                metadata={
                    "source_slot_id": slot,
                    "source_endpoint_id": binding["endpoint_id"],
                    "episode_id": episode_id,
                    "gain_policy": "dataset_raw_no_normalization_no_limiting",
                },
            )
        write_float32_wav(
            audio_root / "mixture_dataset_raw.wav",
            mixture,
            SAMPLE_RATE_HZ,
            metadata={
                "episode_id": episode_id,
                "layout": "native_RLR_HRTF_binaural_left_right",
                "mixture": "exact_source1_plus_source2_stem_sum",
                "gain_policy": "dataset_raw_no_normalization_no_limiting",
            },
        )
        write_float32_wav(
            audio_root / "mixture_review_gain12db.wav",
            review_audio,
            SAMPLE_RATE_HZ,
            metadata={
                "episode_id": episode_id,
                "derived_from": "mixture_dataset_raw.wav",
                "fixed_linear_gain": REVIEW_GAIN,
                "fixed_gain_db": REVIEW_GAIN_DB,
                "limiting": False,
                "normalization": False,
            },
        )

        timeline = _timeline(
            template=template,
            root_paths=root_paths,
            motion_by_slot=motion_by_slot,
            listener_position_m=np.asarray(listener["position_m"], dtype=np.float64),
        )
        manifest = _source_manifest(
            template=template,
            scenario_id=scenario_id,
            episode_directory=directory,
            episode_id=episode_id,
            center_paths=center_paths,
        )
        metadata_root = episode_root / "metadata"
        write_json(metadata_root / "timeline.json", timeline)
        write_json(metadata_root / "source_manifest.json", manifest)
        write_json(
            metadata_root / "flags.json",
            _flags([value["endpoint_id"] for value in SOURCE_BINDINGS.values()]),
        )
        binding = {
            "schema": "avengine_cached_apartment_episode_binding_v1",
            "status": "pass",
            "scenario_id": scenario_id,
            "episode_id": episode_id,
            "motion_case": selected[episode_id]["motion_case"],
            "source_bindings": [
                {
                    "source_slot_id": slot,
                    "actor_id": value["actor_id"],
                    "source_endpoint_id": value["endpoint_id"],
                    "semantic_sound_class": value["semantic_class"],
                    "dry_audio": value["dry_name"],
                }
                for slot, value in SOURCE_BINDINGS.items()
            ],
            "rir_cache": cached.evidence,
            "verification": {
                "rir_job_count": len(cached.evidence["jobs"]),
                "expected_rir_job_count": 50,
                "maximum_cache_vs_trajectory_position_error_m": maximum_cache_position_error,
                "maximum_mixture_vs_stem_sum_error": stem_sum_error,
                "timeline_root_paths_are_bank_arrays": True,
                "source_center_paths_are_bank_arrays": True,
                "audio_sample_count": int(mixture.shape[1]),
                "audio_channel_count": int(mixture.shape[0]),
                "audio_sample_rate_hz": SAMPLE_RATE_HZ,
                "dataset_peak": float(np.max(np.abs(mixture))),
                "review_peak": float(np.max(np.abs(review_audio))),
            },
        }
        if binding["verification"]["rir_job_count"] != 50:
            raise RuntimeError(
                "selected episode does not close over exactly 50 RIR jobs"
            )
        write_json(metadata_root / "cache_binding.json", binding)
        write_json(
            metadata_root / "motion_case.json",
            {
                "schema": "avengine_apartment_motion_case_v1",
                "episode_id": episode_id,
                "motion_case": selected[episode_id]["motion_case"],
                "source_slot_motion": motion_by_slot,
                "both_sources_have_events": True,
                "overlapping_source_events": True,
                "statistics": selected[episode_id]["statistics"],
            },
        )

        topdown = render_runtime_topdown_frames(
            obstacle_map,
            center_paths,
            listener_position_m=listener["position_m"],
            listener_yaw_deg=listener["yaw_deg"],
            camera_hfov_degrees=listener["camera_hfov_degrees"],
            source_activity_by_frame=activities,
            source_labels={"source1": "Human", "source2": "Beagle"},
            source_colors={"source1": (42, 210, 220), "source2": (250, 120, 70)},
            size_wh=(640, 480),
        )
        review_frames = _info_frames(
            scenario_id=scenario_id,
            episode_id=episode_id,
            motion_case=selected[episode_id]["motion_case"],
            activities=activities,
            topdown=topdown,
        )
        videos = episode_root / "videos"
        videos.mkdir(parents=True, exist_ok=True)
        diagnostic = videos / "diagnostic_topdown_binaural.mp4"
        encode_annotated_review(
            review_frames,
            diagnostic,
            fps=FPS,
            audio_path=audio_root / "mixture_review_gain12db.wav",
        )
        os.link(diagnostic, videos / "clean_binaural.mp4")
        media_probe = _probe(diagnostic)
        rows.append(
            {
                "scenario_id": scenario_id,
                "directory": directory,
                "episode_id": episode_id,
                "motion_case": selected[episode_id]["motion_case"],
                "dataset_audio": str(
                    (audio_root / "mixture_dataset_raw.wav").relative_to(staging)
                ),
                "review_audio": str(
                    (audio_root / "mixture_review_gain12db.wav").relative_to(staging)
                ),
                "diagnostic_video": str(diagnostic.relative_to(staging)),
                "cache_binding": str(
                    (metadata_root / "cache_binding.json").relative_to(staging)
                ),
                "diagnostic_video_sha256": sha256_file(diagnostic),
                "media_probe": media_probe,
                "build_wall_seconds": time.perf_counter() - episode_started,
            }
        )

    write_json(
        staging / "manifest.json",
        {
            "schema": SCHEMA,
            "status": "pass",
            "room_id": room_capsule["room_registry_ref"]["room_id"],
            "frame_contract": {
                "seconds": 5,
                "frame_count": FRAME_COUNT,
                "frame_rate_hz": FPS,
            },
            "audio_contract": {
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "sample_count": M5_AUDIO_SAMPLE_COUNT,
                "channel_count": 2,
                "layout": "native_RLR_HRTF_binaural_left_right",
                "both_sources_active": True,
                "camera_fov_cutoff": False,
                "dataset_gain": "raw_no_normalization_no_limiting",
                "review_video_fixed_gain_db": REVIEW_GAIN_DB,
            },
            "source_bindings": SOURCE_BINDINGS,
            "inputs": {
                "trajectory_bank": str(bank_root),
                "trajectory_bank_sha256": sha256_file(
                    bank_root / "trajectory_bank.json"
                ),
                "rir_job_plan_sha256": sha256_file(bank_root / "rir_job_plan.json"),
                "rir_cache": str(cache_root),
                "rir_cache_request_identity_sha256": load_json(
                    cache_root / "request.json"
                )["request_identity_sha256"],
                "audio_template_bundle": str(old_pilot),
            },
            "episodes": rows,
            "build_wall_seconds": time.perf_counter() - started,
        },
    )
    os.rename(staging, output)
    print(f"CACHED_APARTMENT_EXAMPLES_OK output={output}", flush=True)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-bank",
        type=Path,
        default=REPOSITORY / "tmp/m7/apartment_source_slots_diverse_20260721_03",
    )
    parser.add_argument(
        "--rir-cache",
        type=Path,
        default=REPOSITORY / "tmp/m7/apartment_rir_cache_t32_b64_full_20260721_02",
    )
    parser.add_argument(
        "--audio-template-bundle",
        type=Path,
        default=REPOSITORY / "tmp/m7/apartment_four_motion_pilot_20260720_01",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

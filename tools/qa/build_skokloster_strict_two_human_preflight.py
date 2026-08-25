#!/usr/bin/env python3
# HISTORICAL TOOL (single-repo closure, 2026-08-21): this script built or
# validates retained strict-two-human evidence recorded against the
# pre-closure transition environment (sibling Habitat fork, sound-spaces,
# SPEAR-lead-b, and multi-repo SPEAR checkouts). The hard-coded absolute
# paths below are a frozen historical record, not current inputs. The current
# production chain runs on the installed runtime prefix and external data
# roots under /data/avengine_external; do not use this tool for new work.
"""Build a file-evidence-free CPU preflight for the Skokloster strict M/F Episode."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from avengine.timeline.audio_program import bind_audio_program_hash
from avengine.acoustics.rir_cache import validate_semantic_rir_job_plan
from avengine.dataset.sensor_rig import validate_m7_rir_listener_alignment
from avengine.sensor_rig_trajectory import (
    materialize_sensor_rig_trajectory,
    validate_sensor_rig_trajectory,
)

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
REMOTE_REPOSITORY = Path("/data/jzy/code/AVEngine-lead-a")
HABITAT_RUNTIME_ROOT = "/data/jzy/code/habitat-sim-AVEngine"
SOUNDSPACES_ROOT = "/data/jzy/code/sound-spaces"
SKOKLOSTER_RLR48_PACKAGE_ROOT = Path(
    "/tmp/skokloster_strict_room_atom_run/clean_package"
)
HABITAT_PYTHON = Path("/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin/python")
HABITAT_PATH = (
    "/data/jzy/miniconda3/envs/avengine-habitat-runtime/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
HABITAT_EDITABLE_BUILD = (
    "/data/jzy/code/habitat-sim-AVEngine/build/cp312-cp312-linux_x86_64"
)
REQUEST_SCHEMA = "avengine_native_strict_two_human_skokloster_room_atom_request_v1"
REQUEST_SCHEMA_V2 = "avengine_native_strict_two_human_skokloster_room_atom_request_v2"
LEGACY_RIR_EXECUTION_MODE = "legacy"
SEMANTIC_RIR_EXECUTION_MODE = "semantic_no_file_evidence"
SEMANTIC_REQUEST_ID = "skokloster_castle_strict_two_human_static_semantic_v2"
SEMANTIC_EPISODE_ID = "skokloster_castle_male_female_static_semantic_0001"
SEMANTIC_OUTPUT_ROOT = REMOTE_REPOSITORY / "tmp/lead_a_skokloster_strict_two_human_v2"
SEMANTIC_PREFLIGHT_ROOT = SEMANTIC_OUTPUT_ROOT / "cpu_preflight_v1"


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


def validate_rir_execution_environment(environment: Mapping[str, Any]) -> None:
    """Reject a Skokloster RIR plan that is not bound to the native CPU runtime."""

    required = {
        "AVENGINE_HABITAT_RUNTIME_ROOT": HABITAT_RUNTIME_ROOT,
        "AVENGINE_SOUNDSPACES_ROOT": SOUNDSPACES_ROOT,
        "AVENGINE_SKOKLOSTER_RLR48_PACKAGE_ROOT": str(SKOKLOSTER_RLR48_PACKAGE_ROOT),
        "PATH": HABITAT_PATH,
        "PYTHONPATH": str(REMOTE_REPOSITORY / "src"),
        "SKBUILD_EDITABLE_SKIP": HABITAT_EDITABLE_BUILD,
        "NUMBA_DISABLE_JIT": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    for name, expected in required.items():
        value = environment.get(name)
        _require(isinstance(value, str), f"RIR execution environment lacks {name}")
        _require(value == expected, f"RIR execution environment drifted {name}")


def validate_rir_runtime_binding(
    python_executable: str | Path,
    environment: Mapping[str, Any],
) -> None:
    """Bind RLR to the reviewed Habitat interpreter and fail on substitutions."""

    _require(
        Path(python_executable) == HABITAT_PYTHON,
        "RIR runtime interpreter differs from the authoritative Habitat Python",
    )
    validate_rir_execution_environment(environment)


def probe_rir_runtime(output: Path) -> Path:
    """Import the native RLR stack in order and prove CUDA stayed uninitialized."""

    validate_rir_runtime_binding(sys.executable, os.environ)
    _require(not output.exists(), f"refusing to replace runtime probe: {output}")
    numpy = importlib.import_module("numpy")
    quaternion = importlib.import_module("quaternion")
    habitat_sim = importlib.import_module("habitat_sim")
    avengine = importlib.import_module("avengine")
    driver = importlib.import_module("numba.cuda.cudadrv.driver")

    avengine_path = Path(avengine.__file__).resolve()
    expected_avengine_path = (REMOTE_REPOSITORY / "src/avengine/__init__.py").resolve()
    _require(
        avengine_path == expected_avengine_path,
        "runtime probe imported avengine from an unreviewed checkout",
    )
    cuda_initialized = bool(driver.driver.is_initialized)
    _require(not cuda_initialized, "runtime probe initialized CUDA unexpectedly")
    receipt = {
        "schema": "avengine_skokloster_rir_runtime_probe_v1",
        "status": "pass",
        "python_executable": str(Path(sys.executable).resolve()),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "quaternion": getattr(quaternion, "__version__", None),
            "habitat_sim": getattr(habitat_sim, "__version__", None),
        },
        "avengine_source": str(avengine_path),
        "import_order": ["numpy", "quaternion", "habitat_sim", "avengine"],
        "environment": {
            name: os.environ[name]
            for name in (
                "PATH",
                "PYTHONPATH",
                "SKBUILD_EDITABLE_SKIP",
                "NUMBA_DISABLE_JIT",
                "CUDA_VISIBLE_DEVICES",
                "AVENGINE_HABITAT_RUNTIME_ROOT",
                "AVENGINE_SOUNDSPACES_ROOT",
                "AVENGINE_SKOKLOSTER_RLR48_PACKAGE_ROOT",
            )
        },
        "compute_device": "CPU",
        "gpu_required": False,
        "cuda_initialized": cuda_initialized,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, receipt)
    print(
        f"SKOKLOSTER_RIR_RUNTIME_PROBE_OK output={output} cuda_initialized=false",
        flush=True,
    )
    return output


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


def _semantic_regular_file(value: Any, *, owner: str) -> Path:
    raw = Path(str(value))
    _require(raw.is_absolute() and ".." not in raw.parts, f"{owner} path is invalid")
    _require(
        not any(candidate.is_symlink() for candidate in (raw, *raw.parents))
        and raw.is_file()
        and raw.resolve(strict=True) == raw,
        f"{owner} must be an absolute regular file without symlink components",
    )
    return raw


def _semantic_fresh_path(path: Path, *, owner: str) -> Path:
    _require(path.is_absolute() and ".." not in path.parts, f"{owner} path is invalid")
    _require(
        not any(candidate.is_symlink() for candidate in (path, *path.parents))
        and not path.exists(),
        f"{owner} must be absent without symlink components",
    )
    return path


def _validate_request(request: Mapping[str, Any]) -> None:
    _require(
        request.get("schema") in {REQUEST_SCHEMA, REQUEST_SCHEMA_V2},
        "request schema drift",
    )
    if request.get("schema") == REQUEST_SCHEMA_V2:
        _require(
            request.get("request_id") == SEMANTIC_REQUEST_ID
            and request.get("episode_id") == SEMANTIC_EPISODE_ID,
            "v2 semantic request or episode identity drift",
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
    if request.get("schema") == REQUEST_SCHEMA_V2:
        _require(
            audio.get("source1_endpoint_id") == "lead_d_source1_mouth"
            and audio.get("source2_endpoint_id") == "lead_d_source2_mouth"
            and audio["source1_endpoint_id"] != audio["source2_endpoint_id"],
            "v2 semantic source endpoint identity drift",
        )

    execution = request.get("execution")
    _require(isinstance(execution, Mapping), "execution contract is missing")
    if request.get("schema") == REQUEST_SCHEMA_V2:
        _require(
            execution.get("output_root") == str(SEMANTIC_OUTPUT_ROOT),
            "v2 semantic output root drift",
        )
    if request.get("schema") == REQUEST_SCHEMA_V2:
        _require(
            "rir_execution_mode" in execution,
            "v2 request must explicitly select semantic RIR execution",
        )
        rir_execution_mode = execution["rir_execution_mode"]
        _require(
            rir_execution_mode == SEMANTIC_RIR_EXECUTION_MODE,
            "v2 request must select semantic no-file-evidence execution",
        )
    else:
        rir_execution_mode = execution.get(
            "rir_execution_mode", LEGACY_RIR_EXECUTION_MODE
        )
        _require(
            rir_execution_mode == LEGACY_RIR_EXECUTION_MODE,
            "v1 request may only use legacy RIR execution",
        )
    if rir_execution_mode == SEMANTIC_RIR_EXECUTION_MODE:
        _require(
            request.get("schema") == REQUEST_SCHEMA_V2,
            "semantic RIR execution requires the v2 request shape",
        )
        _semantic_regular_file(
            room.get("acoustic_package_manifest"),
            owner="semantic acoustic package manifest",
        )
        _semantic_regular_file(
            room.get("simulation_request"), owner="semantic simulation request"
        )
        _semantic_regular_file(execution.get("hrtf"), owner="semantic HRTF")


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
    if (
        request["execution"].get("rir_execution_mode", LEGACY_RIR_EXECUTION_MODE)
        == SEMANTIC_RIR_EXECUTION_MODE
    ):
        effective = simulation.get("simulation")
        _require(isinstance(effective, Mapping), "semantic simulation is missing")
        _require(
            effective.get("channel_layout") == {"type": "binaural", "channel_count": 2}
            and effective.get("temporal_coherence") is False,
            "semantic simulation must be binaural and noncoherent",
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


def _semantic_audio_documents(request: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    episode_id = str(request["episode_id"])
    content_id = "speech_cremad_1001_ieo_neu_v1"
    target_sound = _semantic_regular_file(
        request["audio"]["target_sound_path"], owner="semantic target sound"
    )
    with wave.open(str(target_sound), "rb") as source_wave:
        metadata = {
            "sample_rate_hz": source_wave.getframerate(),
            "channel_count": source_wave.getnchannels(),
            "sample_count": source_wave.getnframes(),
        }
        pcm_width = source_wave.getsampwidth()
        compression = source_wave.getcomptype()
    _require(
        metadata == {"sample_rate_hz": 16000, "channel_count": 1, "sample_count": 25626}
        and pcm_width == 2
        and compression == "NONE",
        "semantic target sound PCM structure drift",
    )
    endpoints = {
        str(request["audio"]["source1_endpoint_id"]): "source1",
        str(request["audio"]["source2_endpoint_id"]): "source2",
    }
    program = bind_audio_program_hash(
        {
            "schema": "avengine_semantic_audio_program_v1",
            "program_id": f"{episode_id}__semantic_audio_v1",
            "revision": "planning_v1",
            "mode": "one_active_of_n",
            "timeline": {
                "time_base_hz": 48000,
                "ticks_per_frame": 3200,
                "video_fps": 15,
                "frame_count": 75,
                "sample_rate_hz": 16000,
                "ticks_per_sample": 3,
                "sample_count": 80000,
            },
            "candidate_source_endpoint_ids": sorted(endpoints),
            "events": [
                {
                    "event_id": f"{episode_id}__target_speech",
                    "source_endpoint_id": request["audio"]["source1_endpoint_id"],
                    "content_id": content_id,
                    "start_tick": 7467 * 3,
                    "end_tick_exclusive": 33093 * 3,
                    "start_sample": 7467,
                    "end_sample_exclusive": 33093,
                    "source_start_sample": 0,
                    "source_end_sample_exclusive": 25626,
                    "source_sample_rate_hz": metadata["sample_rate_hz"],
                    "source_channel_count": metadata["channel_count"],
                    "source_sample_count": metadata["sample_count"],
                    "linear_gain": 0.18,
                    "fade_samples": 80,
                    "render_source_stem": True,
                }
            ],
            "source_specific_stems": True,
            "admission_state": "research",
        }
    )
    return {
        "semantic_source_endpoint_registry.json": {
            "schema": "avengine_semantic_source_endpoint_registry_v1",
            "registry_id": f"{episode_id}__semantic_endpoints",
            "revision": "planning_v1",
            "source_endpoint_ids": endpoints,
        },
        "semantic_sound_content_registry.json": {
            "schema": "avengine_semantic_sound_content_registry_v1",
            "registry_id": f"{episode_id}__semantic_sound_content",
            "revision": "planning_v1",
            "contents": [
                {
                    "content_id": content_id,
                    "sound_asset_id": content_id,
                    "voice_id": "cremad_1001",
                    "source_audio_uri": f"semantic://{content_id}",
                    **metadata,
                }
            ],
        },
        "semantic_audio_program.json": program,
        "semantic_audio_binding.json": {
            "schema": "avengine_semantic_audio_binding_v1",
            "episode_id": episode_id,
            "variant_id": "A",
            "content_bindings": {
                content_id: {
                    "content_id": content_id,
                    "path": str(target_sound),
                    **metadata,
                }
            },
        },
    }


def _build_documents(
    request: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    episode_id = str(request["episode_id"])
    semantic_rir = (
        request["execution"].get("rir_execution_mode", LEGACY_RIR_EXECUTION_MODE)
        == SEMANTIC_RIR_EXECUTION_MODE
    )
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

    sensor_rig_trajectory_id = f"{episode_id}__sensor_rig_v3"
    rig = materialize_sensor_rig_trajectory(
        trajectory_id=sensor_rig_trajectory_id,
        program={"kind": "HOLD", "position_m": camera, "yaw_deg": yaw},
    )
    _require(
        not validate_sensor_rig_trajectory(rig),
        "canonical SensorRigTrajectory validator rejected the static hold",
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
            "sensor_rig_trajectory_id": sensor_rig_trajectory_id,
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
    authoritative_inputs = (
        {
            "audio_program": str(
                SEMANTIC_PREFLIGHT_ROOT / "semantic_audio_program.json"
            ),
            "source_endpoint_registry": str(
                SEMANTIC_PREFLIGHT_ROOT / "semantic_source_endpoint_registry.json"
            ),
            "sound_content_registry": str(
                SEMANTIC_PREFLIGHT_ROOT / "semantic_sound_content_registry.json"
            ),
            "audio_binding": str(
                SEMANTIC_PREFLIGHT_ROOT / "semantic_audio_binding.json"
            ),
        }
        if semantic_rir
        else {
            "audio_program": request["audio"]["canonical_audio_program"],
            "source_endpoint_registry": request["audio"]["source_endpoint_registry"],
            "sound_asset_registry": request["audio"]["sound_asset_registry"],
        }
    )
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
        "authoritative_inputs": authoritative_inputs,
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
    if semantic_rir:
        rir_plan = {
            "schema": "avengine_room_rir_job_plan_v2",
            "status": "planned_not_run",
            "listener_pose_mode": "fixed",
            "listener_position_m": camera,
            "listener_orientation_wxyz": evidence["listener_orientation_wxyz"],
            "cache_key_fields": [
                "source_position_m",
                "listener_position_m",
                "listener_orientation_wxyz",
            ],
            "stride_frames": 1,
            "requested_pair_state_count": 150,
            "unique_rir_job_count": 2,
            "jobs": [
                {
                    "job_id": f"skokloster_{slot}_static_semantic_v1",
                    "source_position_m": centers[slot],
                    "uses": uses[slot],
                }
                for slot in ("source1", "source2")
            ],
            "claim_boundary": "fresh semantic CPU RIR plan; no file evidence",
            "producer_backend": "RLR Audio Propagation",
            "cache_artifact": "room impulse response (RIR)",
            "source_acoustic_profile": "omnidirectional_point_source_v1",
            "slot_identity_affects_cache_key": False,
            "dry_audio_independent": True,
            "unique_listener_pose_count": 1,
            "cache_reuse_count": 148,
        }
        validate_semantic_rir_job_plan(rir_plan)
    else:
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
    alignment = validate_m7_rir_listener_alignment(
        rir_job_plan=rir_plan,
        sensor_rig_trajectory=rig,
    )
    _require(
        alignment.get("listener_pose_mode") == "fixed"
        and alignment.get("checked_use_count") == 150,
        "canonical SensorRigTrajectory does not align with all RIR uses",
    )
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
    documents = {
        "suite_execution_plan.json": suite,
        "sensor_rig_trajectory.json": rig,
        "trajectory_bank.json": trajectory,
        "asset_emitter_binding_report.json": binding_report,
        "rir_job_plan.json": rir_plan,
    }
    if semantic_rir:
        documents.update(_semantic_audio_documents(request))
    else:
        documents["audio_program_binding.json"] = audio_plan
    return documents


def _execution_plan(request: Mapping[str, Any], output: Path) -> dict[str, Any]:
    execution = request["execution"]
    repository = Path(execution["repository"])
    _require(repository == REMOTE_REPOSITORY, "execution repository drift")
    output_root = Path(execution["output_root"])
    semantic_rir = (
        execution.get("rir_execution_mode", LEGACY_RIR_EXECUTION_MODE)
        == SEMANTIC_RIR_EXECUTION_MODE
    )
    runtime_probe = output / "rir_runtime_probe.json"
    rir_cache = output_root / (
        "semantic_exact_rir_cache_v1" if semantic_rir else "exact_rir_cache_v3"
    )
    binaural = output_root / ("semantic_binaural_v1" if semantic_rir else "binaural_v4")
    rir_environment = {
        "AVENGINE_HABITAT_RUNTIME_ROOT": HABITAT_RUNTIME_ROOT,
        "AVENGINE_SOUNDSPACES_ROOT": SOUNDSPACES_ROOT,
        "AVENGINE_SKOKLOSTER_RLR48_PACKAGE_ROOT": str(SKOKLOSTER_RLR48_PACKAGE_ROOT),
        "PATH": HABITAT_PATH,
        "PYTHONPATH": str(repository / "src"),
        "SKBUILD_EDITABLE_SKIP": HABITAT_EDITABLE_BUILD,
        "NUMBA_DISABLE_JIT": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    validate_rir_runtime_binding(HABITAT_PYTHON, rir_environment)
    rir_argv = [
        str(HABITAT_PYTHON),
        str(repository / "tools/acoustics/render_rir_cache.py"),
        "--rir-job-plan",
        str(output / "rir_job_plan.json"),
    ]
    if semantic_rir:
        rir_argv.extend(
            [
                "--semantic-no-file-evidence",
                "--acoustic-package-manifest",
                request["room"]["acoustic_package_manifest"],
                "--simulation-request",
                request["room"]["simulation_request"],
                "--hrtf",
                execution["hrtf"],
            ]
        )
    else:
        rir_argv.extend(
            [
                "--acoustic-package-manifest",
                request["room"]["acoustic_package_manifest"],
                "--simulation-request",
                request["room"]["simulation_request"],
                "--hrtf",
                execution["hrtf"],
            ]
        )
    rir_argv.extend(
        [
            "--output",
            str(rir_cache),
            "--layout",
            "binaural",
            "--batch-size",
            "2",
            "--thread-count",
            str(execution["rir_thread_count"]),
        ]
    )
    rir_step = {
        "step_id": "render_two_exact_binaural_rirs",
        **(
            {"attempt_id": "semantic_exact_rir_cache_v1"}
            if semantic_rir
            else {
                "attempt_id": "exact_rir_cache_v3",
                "supersedes_failed_attempts": ["exact_rir_cache_v1"],
                "prior_valid_cache_not_reusable_for_plan_path": str(
                    output_root / "exact_rir_cache_v2"
                ),
            }
        ),
        "status": "planned_not_run",
        "working_directory": str(repository),
        "environment": rir_environment,
        "argv": rir_argv,
        "expected": {
            "compute_device": "CPU",
            "selected_job_count": 2,
            "full_plan_complete": True,
            "layout": "binaural",
        },
    }
    supersedes = (
        []
        if semantic_rir
        else [
            {
                "attempt_id": "cpu_preflight_v1",
                "status": "rejected_before_rir_execution",
                "reason": "noncanonical listener pose mode failed the real RIR validator",
            },
            {
                "attempt_id": "cpu_preflight_v2",
                "status": "rejected_during_rir_execution",
                "reason": (
                    "repository Python lacked numpy-quaternion; exact_rir_cache_v1 "
                    "failed before rendering any job"
                ),
                "failed_cache": str(output_root / "exact_rir_cache_v1"),
                "rendered_job_count": 0,
            },
            {
                "attempt_id": "cpu_preflight_v3",
                "status": "audio_plan_rejected_after_rir_pass",
                "reason": (
                    "handwritten sensor rig did not satisfy the authoritative M7 "
                    "SensorRigTrajectory validator"
                ),
                "retained_rir_cache": str(output_root / "exact_rir_cache_v2"),
                "retained_rir_job_count": 2,
                "failed_audio_attempt": "binaural_v2",
            },
        ]
    )
    if semantic_rir:
        m7_argv = [
            execution["python"],
            str(repository / "tools/dataset/render_asset_bound_binaural_batch.py"),
            "--plan-root",
            str(output),
            "--rir-cache",
            str(rir_cache),
            "--audio-program",
            str(output / "semantic_audio_program.json"),
            "--audio-program-variant",
            "A",
            "--semantic-source-endpoint-registry",
            str(output / "semantic_source_endpoint_registry.json"),
            "--semantic-sound-content-registry",
            str(output / "semantic_sound_content_registry.json"),
            "--semantic-audio-binding",
            str(output / "semantic_audio_binding.json"),
            "--variants-per-episode",
            "1",
            "--retain-stems",
            "--output",
            str(binaural),
        ]
    else:
        m7_argv = [
            execution["python"],
            str(repository / "tools/dataset/render_asset_bound_binaural_batch.py"),
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
            "speech_cremad_1001_ieo_neu_v1=" + request["audio"]["target_sound_path"],
            "--retain-stems",
            "--output",
            str(binaural),
        ]
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
        "schema": (
            "avengine_skokloster_strict_two_human_execution_plan_v2"
            if semantic_rir
            else "avengine_skokloster_strict_two_human_execution_plan_v1"
        ),
        "status": "cpu_ready_gpu_blocked",
        "attempt_id": output.name,
        "supersedes": supersedes,
        "generated_preflight_root": str(output.resolve()),
        "runtime_output_root": str(output_root),
        "cpu_steps": [
            {
                "step_id": "probe_authoritative_habitat_rir_runtime",
                "status": "planned_not_run",
                "working_directory": str(repository),
                "environment": rir_environment,
                "argv": [
                    str(HABITAT_PYTHON),
                    str(
                        repository
                        / "tools/qa/build_skokloster_strict_two_human_preflight.py"
                    ),
                    "--runtime-probe-output",
                    str(runtime_probe),
                ],
                "expected": {
                    "receipt": str(runtime_probe),
                    "status": "pass",
                    "python": "3.12.13",
                    "numpy": "2.3.5",
                    "quaternion": "2024.0.13",
                    "habitat_sim": "0.3.3",
                    "avengine_source": str(repository / "src/avengine/__init__.py"),
                    "compute_device": "CPU",
                    "cuda_initialized": False,
                    "qualification_claim": False,
                },
            },
            rir_step,
            {
                "step_id": "render_target_speech_silent_distractor_binaural",
                "status": "blocked_until_exact_rir_pass",
                "working_directory": str(repository),
                "argv": m7_argv,
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
    request: Mapping[str, Any], evidence: Mapping[str, Any], attempt_id: str
) -> dict[str, Any]:
    semantic_rir = (
        request["execution"].get("rir_execution_mode", LEGACY_RIR_EXECUTION_MODE)
        == SEMANTIC_RIR_EXECUTION_MODE
    )
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
        "schema": (
            "avengine_skokloster_strict_two_human_cpu_preflight_v2"
            if semantic_rir
            else "avengine_skokloster_strict_two_human_cpu_preflight_v1"
        ),
        "status": "cpu_plan_pass_gpu_sparse_pending",
        "attempt_id": attempt_id,
        "supersedes": []
        if semantic_rir
        else [
            {
                "attempt_id": "cpu_preflight_v1",
                "status": "rejected_before_rir_execution",
                "reason": "noncanonical listener pose mode failed the real RIR validator",
            },
            {
                "attempt_id": "cpu_preflight_v2",
                "status": "rejected_during_rir_execution",
                "reason": (
                    "repository Python lacked numpy-quaternion; no RIR job rendered"
                ),
                "rendered_job_count": 0,
            },
            {
                "attempt_id": "cpu_preflight_v3",
                "status": "audio_plan_rejected_after_rir_pass",
                "reason": (
                    "handwritten sensor rig did not satisfy the authoritative M7 "
                    "SensorRigTrajectory validator"
                ),
                "retained_rir_job_count": 2,
                "failed_audio_attempt": "binaural_v2",
            },
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


def _validate_semantic_selected_paths(
    request: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    declared_semantic_paths = {
        "package": Path(request["room"]["acoustic_package_manifest"]),
        "simulation": Path(request["room"]["simulation_request"]),
    }
    for name, declared in declared_semantic_paths.items():
        selected = _semantic_regular_file(paths[name], owner=f"semantic {name} input")
        _require(
            selected == declared,
            f"semantic {name} override differs from the declared request path",
        )


def build(args: argparse.Namespace) -> Path:
    request = _load(args.request.resolve())
    _validate_request(request)
    semantic_rir = (
        request["execution"].get("rir_execution_mode", LEGACY_RIR_EXECUTION_MODE)
        == SEMANTIC_RIR_EXECUTION_MODE
    )
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
    if semantic_rir:
        _validate_semantic_selected_paths(request, paths)
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
    raw_output = args.output
    if semantic_rir:
        _require(
            raw_output == SEMANTIC_PREFLIGHT_ROOT,
            "v2 semantic preflight output path drift",
        )
        _semantic_fresh_path(
            SEMANTIC_OUTPUT_ROOT, owner="semantic execution output root"
        )
        output = raw_output
    else:
        output = raw_output.resolve()
        _require(
            not output.exists() and not output.is_symlink(), "output already exists"
        )
    output.mkdir(parents=True)
    documents = _build_documents(request, evidence)
    documents["execution_plan.json"] = _execution_plan(request, output)
    documents["preflight.json"] = _preflight(request, evidence, output.name)
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
    parser.add_argument("--runtime-probe-output", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--listener-search", type=Path)
    parser.add_argument("--near-rejection", type=Path)
    parser.add_argument("--runtime-profile", type=Path)
    parser.add_argument("--acoustic-profile", type=Path)
    parser.add_argument("--package-manifest", type=Path)
    parser.add_argument("--simulation-request", type=Path)
    parser.add_argument("--audio-program", type=Path)
    parser.add_argument("--audio-binding", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runtime_probe_output is not None:
        _require(args.request is None, "runtime probe may not also build a request")
        _require(args.output is None, "runtime probe may not also build an output")
        probe_rir_runtime(args.runtime_probe_output.resolve())
        return 0
    _require(args.request is not None, "--request is required when building")
    _require(args.output is not None, "--output is required when building")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

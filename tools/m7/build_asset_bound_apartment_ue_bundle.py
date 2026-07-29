#!/usr/bin/env python3
"""Materialize M7 source1/source2 routes as one reusable Apartment UE bundle."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import load_json, sha256_file, write_json
from avengine.m6x.geometry import RuntimeObstacleMap
from avengine.m6x.topdown import render_runtime_topdown_frames
from avengine.m7.apartment_visual_bundle import (
    FRAME_COUNT,
    FRAME_RATE_HZ,
    binding_assets_by_episode,
    build_flags,
    build_qualification,
    build_source_manifest,
    build_timeline,
    program_source_activity_by_frame,
)
from avengine.optional_backends.spear_apartment import (
    build_rawvideo_encode_command,
)
from avengine.runtime_profiles import (
    build_exact_asset_bound_runtime_binding,
    default_source_asset_runtime_registry_path,
    load_source_asset_runtime_registry,
    resolve_source_asset_runtime_profile,
    source_timeline_profiles,
)


SCHEMA = "avengine_m7_asset_bound_apartment_ue_input_bundle_v1"
SOURCE_SLOTS = ("source1", "source2")
_AUDIO_PROGRAM_SAMPLE_FIELDS = frozenset(
    {
        "audio_program_binding",
        "audio_program_instance_path",
        "audio_program_instance_sha256",
    }
)


def _obstacle_map(root: Path) -> RuntimeObstacleMap:
    record = load_json(root / "feasible_region.json")["obstacle_authority"]
    navmesh = np.load(
        root / "feasible_region_source1.npz", allow_pickle=False
    )["navmesh_mask"]
    return RuntimeObstacleMap(
        binary_navmesh=np.ascontiguousarray(navmesh),
        bounds_m=tuple(tuple(float(value) for value in row) for row in record["bounds_m"]),
        floor_height_m=float(record["floor_height_m"]),
        meters_per_pixel=float(record["meters_per_pixel"]),
        rigid_obstacles=tuple(record.get("rigid_obstacles", ())),
        authority=str(record["authority"]),
        claim_boundary=str(record["claim_boundary"]),
        rigid_obstacles_baked_into_navmesh=bool(
            record.get("rigid_obstacles_baked_into_navmesh", False)
        ),
    )


def _samples(batch_root: Path) -> dict[str, Mapping[str, Any]]:
    record = load_json(batch_root / "samples.json")
    rows = record.get("samples")
    if record.get("status") != "pass" or not isinstance(rows, list):
        raise RuntimeError("audio batch samples are invalid")
    program_states: list[bool] = []
    result = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("audio batch sample is invalid")
        present = _AUDIO_PROGRAM_SAMPLE_FIELDS.intersection(row)
        if present and present != _AUDIO_PROGRAM_SAMPLE_FIELDS:
            raise RuntimeError("audio batch has an incomplete AudioProgram sample")
        program_states.append(bool(present))
        if row.get("variant_index") != 0:
            continue
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or episode_id in result:
            raise RuntimeError("audio batch v00 episode IDs are invalid")
        result[episode_id] = row
    if any(program_states) and not all(program_states):
        raise RuntimeError("legacy and AudioProgram samples may not be mixed")
    if any(program_states) and any(
        row.get("variant_index") != 0 for row in rows
    ):
        raise RuntimeError(
            "AudioProgram visual bundles require only v00 samples"
        )
    return result


def _assert_sample_asset_alignment(
    *,
    episode_id: str,
    episode_bindings: Mapping[str, Mapping[str, Any]],
    sample: Mapping[str, Any],
) -> None:
    visual_assets = {
        slot: episode_bindings[slot]["asset_id"] for slot in SOURCE_SLOTS
    }
    if sample.get("asset_ids_by_source_slot") != visual_assets:
        raise RuntimeError(
            f"visual and audio asset bindings differ for {episode_id}"
        )


def _sample_audio_program_projection(
    *,
    sample: Mapping[str, Any],
    batch_root: Path,
) -> tuple[
    Mapping[str, Any] | None,
    Mapping[str, str] | None,
    Mapping[str, Any] | None,
    str | None,
    Mapping[str, str] | None,
]:
    binding = sample.get("audio_program_binding")
    relative = sample.get("audio_program_instance_path")
    expected_sha256 = sample.get("audio_program_instance_sha256")
    if binding is None and relative is None and expected_sha256 is None:
        return None, None, None, None, None
    if (
        not isinstance(binding, Mapping)
        or not isinstance(relative, str)
        or not relative
        or not isinstance(expected_sha256, str)
    ):
        raise RuntimeError("audio sample has an incomplete AudioProgram binding")
    root = batch_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError("AudioProgram instance escapes the audio batch root") from error
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise RuntimeError("AudioProgram instance is missing or changed")
    instance = load_json(path)
    program = instance.get("materialized_audio_program")
    mapping = binding.get("source_endpoint_to_source_slot")
    variant_id = binding.get("variant_id")
    mapped_events = instance.get("mapped_events")
    sound_semantics = instance.get("sound_asset_semantics")
    if (
        instance.get("schema") != "avengine_m7_m6_audio_program_instance_v1"
        or instance.get("status") != "pass"
        or instance.get("audio_program_binding") != dict(binding)
        or not isinstance(program, Mapping)
        or not isinstance(mapping, Mapping)
        or variant_id not in {"A", "B"}
        or not isinstance(mapped_events, list)
        or not isinstance(sound_semantics, Mapping)
        or program.get("program_content_sha256")
        != binding.get("materialized_program_content_sha256")
    ):
        raise RuntimeError("AudioProgram instance differs from its sample binding")
    events = program.get("events")
    if not isinstance(events, list) or len(mapped_events) != len(events):
        raise RuntimeError("AudioProgram mapped event semantics are incomplete")
    event_semantics: dict[str, str] = {}
    used_sound_ids: set[str] = set()
    for event, mapped_event in zip(events, mapped_events, strict=True):
        if not isinstance(event, Mapping) or not isinstance(mapped_event, Mapping):
            raise RuntimeError("AudioProgram mapped event semantics are invalid")
        event_id = event.get("event_id")
        sound_id = event.get("sound_asset_id")
        endpoint_id = event.get("source_endpoint_id")
        sound_class = mapped_event.get("semantic_sound_class")
        expected = {
            **dict(event),
            "source_slot_id": mapping.get(endpoint_id),
            "semantic_sound_class": sound_class,
        }
        if (
            not isinstance(event_id, str)
            or not isinstance(sound_id, str)
            or endpoint_id not in mapping
            or not isinstance(sound_class, str)
            or not sound_class
            or dict(mapped_event) != expected
            or sound_semantics.get(sound_id) != sound_class
            or event_id in event_semantics
        ):
            raise RuntimeError("AudioProgram mapped event semantics are invalid")
        event_semantics[event_id] = sound_class
        used_sound_ids.add(sound_id)
    if set(sound_semantics) != used_sound_ids:
        raise RuntimeError("AudioProgram sound semantics differ from its events")
    return program, dict(mapping), instance, str(variant_id), event_semantics


def _resolve_exact_episode_runtime_bindings(
    *,
    episode_bindings: Mapping[str, Mapping[str, Any]],
    source_registry: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Freeze exact runtime closures for generated assets already carrying one."""

    result: dict[str, dict[str, Any]] = {}
    emitter_fields = (
        "source_slot_id",
        "asset_id",
        "asset_revision",
        "semantic_anchor_id",
        "emitter_offset_m",
        "local_anatomical_forward_axis",
        "offset_space",
    )
    for slot in SOURCE_SLOTS:
        binding = episode_bindings[slot]
        asset_id = binding["asset_id"]
        record = resolve_source_asset_runtime_profile(
            source_registry,
            asset_id,
            binding.get("asset_revision"),
        )
        if not isinstance(record.get("asset_bound_lineage"), Mapping):
            continue
        exact = build_exact_asset_bound_runtime_binding(
            source_registry,
            source_slot_id=slot,
            asset_id=asset_id,
            revision=binding.get("asset_revision"),
            anchor_id=binding.get("semantic_anchor_id"),
        )
        emitter = exact["emitter"]
        mismatches = [
            field
            for field in emitter_fields
            if binding.get(field) != emitter.get(field)
        ]
        if mismatches:
            raise RuntimeError(
                f"exact runtime emitter binding differs for {slot}: "
                f"{', '.join(mismatches)}"
            )
        result[slot] = exact
    return result


def _encode_diagnostic(
    frames: np.ndarray,
    *,
    audio_path: Path,
    output_path: Path,
    video_encoder: str,
    encoder_gpu: int | None,
) -> None:
    """Stream RGB frames through FFmpeg and add the exact five-second WAV."""

    if (
        frames.shape != (FRAME_COUNT, 480, 640, 3)
        or frames.dtype != np.uint8
    ):
        raise RuntimeError("diagnostic Topdown frames differ from 75x480x640 RGB8")
    visual = output_path.with_name(f".{output_path.stem}.visual.mp4")
    command = build_rawvideo_encode_command(
        output_path=visual,
        video_encoder=video_encoder,
        encoder_gpu=encoder_gpu,
        width=640,
        height=480,
        frame_rate_hz=FRAME_RATE_HZ,
        frame_count=FRAME_COUNT,
        pixel_format="rgb24",
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        process.stdin.write(np.ascontiguousarray(frames).tobytes(order="C"))
        process.stdin.close()
        stderr = b"" if process.stderr is None else process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                "diagnostic rawvideo encode failed: "
                + stderr.decode("utf-8", errors="replace").strip()
            )
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(visual),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-ar",
                "16000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        visual.unlink(missing_ok=True)


def _build_episode(
    *,
    ordinal: int,
    episode_id: str,
    episode: Mapping[str, Any],
    episode_bindings: Mapping[str, Mapping[str, Any]],
    source_profiles: Mapping[str, Mapping[str, Any]],
    listener: Mapping[str, Any],
    obstacle_map: RuntimeObstacleMap,
    sample: Mapping[str, Any],
    batch_root: Path,
    staging: Path,
    video_encoder: str,
    encoder_gpu: int | None,
    variants_per_episode: int,
    runtime_bindings_by_source_slot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one independent episode directory for sequential or process use."""

    episode_started = time.perf_counter()
    _assert_sample_asset_alignment(
        episode_id=episode_id,
        episode_bindings=episode_bindings,
        sample=sample,
    )
    (
        materialized_audio_program,
        endpoint_to_source_slot,
        program_instance,
        audio_program_variant_id,
        event_sound_classes,
    ) = _sample_audio_program_projection(sample=sample, batch_root=batch_root)
    timeline, headings = build_timeline(
        episode=episode,
        bindings=episode_bindings,
        listener_position_m=listener["position_m"],
        source_profiles=source_profiles,
        materialized_audio_program=materialized_audio_program,
        endpoint_to_source_slot=endpoint_to_source_slot,
        semantic_sound_class_by_event_id=event_sound_classes,
    )
    manifest = build_source_manifest(
        episode_id=episode_id,
        episode=episode,
        bindings=episode_bindings,
        source_profiles=source_profiles,
        materialized_audio_program=materialized_audio_program,
        endpoint_to_source_slot=endpoint_to_source_slot,
        audio_program_variant_id=audio_program_variant_id,
        semantic_sound_class_by_event_id=event_sound_classes,
    )
    episodes_root = staging / "episodes"
    episodes_root.mkdir(exist_ok=True)
    final_episode_root = episodes_root / episode_id
    episode_root = episodes_root / f".{episode_id}.staging.{os.getpid()}"
    if final_episode_root.exists() or episode_root.exists():
        raise RuntimeError(f"episode output already exists: {episode_id}")
    metadata = episode_root / "metadata"
    videos = episode_root / "videos"
    metadata.mkdir(parents=True)
    videos.mkdir()
    try:
        write_json(metadata / "timeline.json", timeline)
        write_json(metadata / "source_manifest.json", manifest)
        write_json(metadata / "flags.json", build_flags())
        centers = {
            slot: np.asarray(
                episode["source_center_paths_m"][slot], dtype=np.float64
            )
            for slot in SOURCE_SLOTS
        }
        activity = (
            {
                slot: np.ones(FRAME_COUNT, dtype=np.bool_)
                for slot in SOURCE_SLOTS
            }
            if materialized_audio_program is None
            else program_source_activity_by_frame(
                materialized_audio_program,
                endpoint_to_source_slot,
            )
        )
        topdown = render_runtime_topdown_frames(
            obstacle_map,
            centers,
            listener_position_m=listener["position_m"],
            listener_yaw_deg=listener["yaw_deg"],
            camera_hfov_degrees=listener["camera_hfov_degrees"],
            source_activity_by_frame=activity,
            source_heading_xz_by_frame=headings,
            source_labels={
                slot: (
                    f"{slot}: "
                    f"{source_profiles[episode_bindings[slot]['asset_id']]['display_label']}"
                )
                for slot in SOURCE_SLOTS
            },
            source_colors={
                "source1": (42, 210, 220),
                "source2": (250, 120, 70),
            },
            size_wh=(640, 480),
        )
        mixture = sample["audio"]["mixture"]
        mixture_path = batch_root / "audio/binaural" / mixture["path"]
        if (
            not mixture_path.is_file()
            or sha256_file(mixture_path) != mixture["audio_sha256"]
        ):
            raise RuntimeError(f"batch mixture changed for {episode_id}")
        diagnostic = videos / "diagnostic_topdown_binaural.mp4"
        _encode_diagnostic(
            topdown,
            audio_path=mixture_path,
            output_path=diagnostic,
            video_encoder=video_encoder,
            encoder_gpu=encoder_gpu,
        )
        os.link(diagnostic, videos / "clean_binaural.mp4")
        batch_binding = {
            "schema": "avengine_m7_asset_bound_apartment_episode_binding_v1",
            "status": "pass",
            "episode_id": episode_id,
            "visual_render_reuse_count": variants_per_episode,
            "v00_sample_id": sample["sample_id"],
            "v00_mixture": dict(mixture),
            "asset_ids_by_source_slot": sample["asset_ids_by_source_slot"],
            "source_center_gate_status": "pass",
            "runtime_bindings_by_source_slot": dict(
                runtime_bindings_by_source_slot
            ),
        }
        if program_instance is not None:
            batch_binding.update(
                {
                    "audio_program_binding": sample["audio_program_binding"],
                    "audio_program_instance_path": sample[
                        "audio_program_instance_path"
                    ],
                    "audio_program_instance_sha256": sample[
                        "audio_program_instance_sha256"
                    ],
                }
            )
        write_json(metadata / "batch_binding.json", batch_binding)
        row = {
            "episode_ordinal": ordinal,
            "episode_id": episode_id,
            "motion_case": episode["motion_case"],
            "asset_ids_by_source_slot": sample["asset_ids_by_source_slot"],
            "v00_sample_id": sample["sample_id"],
            "build_wall_seconds": time.perf_counter() - episode_started,
        }
        write_json(
            metadata / "build_record.json",
            {
                "schema": "avengine_m7_apartment_ue_input_episode_build_v1",
                "status": "pass",
                "diagnostic_sha256": sha256_file(diagnostic),
                "row": row,
            },
        )
        os.rename(episode_root, final_episode_root)
        return row
    except BaseException:
        shutil.rmtree(episode_root, ignore_errors=True)
        raise


def _load_completed_episode(
    *,
    staging: Path,
    episode_id: str,
    ordinal: int,
    sample: Mapping[str, Any],
    batch_root: Path,
    runtime_bindings_by_source_slot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    episode_root = staging / "episodes" / episode_id
    record_path = episode_root / "metadata/build_record.json"
    if not record_path.is_file():
        if episode_root.exists():
            shutil.rmtree(episode_root)
        return None
    record = load_json(record_path)
    row = record.get("row")
    diagnostic = episode_root / "videos/diagnostic_topdown_binaural.mp4"
    clean = episode_root / "videos/clean_binaural.mp4"
    batch_binding_path = episode_root / "metadata/batch_binding.json"
    batch_binding = (
        load_json(batch_binding_path) if batch_binding_path.is_file() else {}
    )
    program_projection = _sample_audio_program_projection(
        sample=sample,
        batch_root=batch_root,
    )
    has_audio_program = program_projection[0] is not None
    program_binding_differs = (
        any(
            field in batch_binding
            for field in (
                "audio_program_binding",
                "audio_program_instance_path",
                "audio_program_instance_sha256",
            )
        )
        if not has_audio_program
        else (
            batch_binding.get("audio_program_binding")
            != sample.get("audio_program_binding")
            or batch_binding.get("audio_program_instance_path")
            != sample.get("audio_program_instance_path")
            or batch_binding.get("audio_program_instance_sha256")
            != sample.get("audio_program_instance_sha256")
        )
    )
    if (
        record.get("status") != "pass"
        or not isinstance(row, Mapping)
        or row.get("episode_id") != episode_id
        or row.get("v00_sample_id") != sample["sample_id"]
        or not diagnostic.is_file()
        or not clean.is_file()
        or diagnostic.stat().st_ino != clean.stat().st_ino
        or sha256_file(diagnostic) != record.get("diagnostic_sha256")
        or batch_binding.get("runtime_bindings_by_source_slot", {})
        != dict(runtime_bindings_by_source_slot)
        or program_binding_differs
        or any(
            not (episode_root / relative).is_file()
            for relative in (
                "metadata/timeline.json",
                "metadata/source_manifest.json",
                "metadata/flags.json",
                "metadata/batch_binding.json",
            )
        )
    ):
        raise RuntimeError(f"completed episode changed: {episode_id}")
    result = dict(row)
    result["episode_ordinal"] = ordinal
    return result


def build_bundle(
    *,
    plan_root: Path,
    feasibility_root: Path,
    batch_root: Path,
    room_template_bundle: Path,
    source_asset_registry: Path | None,
    episode_ids: Sequence[str] | None,
    video_encoder: str,
    encoder_gpu: int | None,
    workers: int,
    resume: bool,
    output: Path,
) -> Path:
    started = time.perf_counter()
    plan_root = plan_root.resolve()
    feasibility_root = feasibility_root.resolve()
    batch_root = batch_root.resolve()
    room_template_bundle = room_template_bundle.resolve()
    source_asset_registry = (
        default_source_asset_runtime_registry_path()
        if source_asset_registry is None
        else source_asset_registry.resolve()
    )
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output}")
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 32
    ):
        raise ValueError("workers must be an integer between 1 and 32")
    staging = output.with_name(f".{output.name}.staging")
    if staging.is_symlink():
        raise FileExistsError(f"refusing symlink staging output: {staging}")
    if staging.exists():
        if not resume or not staging.is_dir():
            raise FileExistsError(f"refusing to replace staging output: {staging}")
    elif resume:
        raise FileNotFoundError(f"resume staging output is missing: {staging}")

    bank = load_json(plan_root / "trajectory_bank.json")
    raw_episodes = bank.get("episodes")
    if not isinstance(raw_episodes, list):
        raise RuntimeError("trajectory bank episodes are invalid")
    episodes = {
        value["episode_id"]: value
        for value in raw_episodes
        if isinstance(value, Mapping) and isinstance(value.get("episode_id"), str)
    }
    source_registry = load_source_asset_runtime_registry(source_asset_registry)
    source_profiles = source_timeline_profiles(source_registry)
    bindings = binding_assets_by_episode(
        load_json(plan_root / "asset_emitter_binding_report.json"),
        source_profiles=source_profiles,
    )
    exact_runtime_bindings = {
        episode_id: _resolve_exact_episode_runtime_bindings(
            episode_bindings=episode_bindings,
            source_registry=source_registry,
        )
        for episode_id, episode_bindings in bindings.items()
    }
    if set(episodes) != set(bindings):
        raise RuntimeError("trajectory bank and asset binding episode sets differ")
    selected = tuple(episode_ids) if episode_ids is not None else tuple(sorted(episodes))
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(episodes):
        raise RuntimeError("episode selection is empty, repeated, or unknown")
    samples = _samples(batch_root)
    if not set(selected) <= set(samples):
        raise RuntimeError("selected episodes lack v00 audio samples")
    batch_delivery = load_json(batch_root / "delivery.json")
    variants_per_episode = batch_delivery.get("variants_per_episode")
    if (
        batch_delivery.get("status") != "pass"
        or isinstance(variants_per_episode, bool)
        or not isinstance(variants_per_episode, int)
        or variants_per_episode < 1
        or batch_delivery.get("episode_count") != len(episodes)
    ):
        raise RuntimeError("audio batch episode/variant declaration is invalid")
    program_flags = {
        sample.get("audio_program_binding") is not None
        for sample in samples.values()
    }
    if len(program_flags) != 1:
        raise RuntimeError("legacy and AudioProgram samples may not be mixed")
    if True in program_flags and variants_per_episode != 1:
        raise RuntimeError(
            "AudioProgram visual bundles require one program instance per episode"
        )

    qualification_template = load_json(
        room_template_bundle / "room/qualification.json"
    )
    listener = qualification_template["listener"]
    obstacle_map = _obstacle_map(feasibility_root)
    staging.mkdir(parents=True, exist_ok=resume)
    try:
        room_root = staging / "room"
        room_root.mkdir(exist_ok=True)
        shutil.copy2(
            room_template_bundle / "room/room_capsule.json",
            room_root / "room_capsule.json",
        )
        write_json(
            room_root / "qualification.json",
            build_qualification(
                template=qualification_template,
                episode_ids=selected,
                episodes=episodes,
            ),
        )
        completed_rows = []
        task_arguments = []
        for ordinal, episode_id in enumerate(selected):
            for partial in (staging / "episodes").glob(
                f".{episode_id}.staging.*"
            ):
                shutil.rmtree(partial)
            completed = (
                _load_completed_episode(
                    staging=staging,
                    episode_id=episode_id,
                    ordinal=ordinal,
                    sample=samples[episode_id],
                    batch_root=batch_root,
                    runtime_bindings_by_source_slot=exact_runtime_bindings[
                        episode_id
                    ],
                )
                if resume
                else None
            )
            if completed is not None:
                completed_rows.append(completed)
                continue
            task_arguments.append(
                {
                    "ordinal": ordinal,
                    "episode_id": episode_id,
                    "episode": episodes[episode_id],
                    "episode_bindings": bindings[episode_id],
                    "source_profiles": source_profiles,
                    "listener": listener,
                    "obstacle_map": obstacle_map,
                    "sample": samples[episode_id],
                    "batch_root": batch_root,
                    "staging": staging,
                    "video_encoder": video_encoder,
                    "encoder_gpu": encoder_gpu,
                    "variants_per_episode": variants_per_episode,
                    "runtime_bindings_by_source_slot": exact_runtime_bindings[
                        episode_id
                    ],
                }
            )
        if workers == 1:
            new_rows = [_build_episode(**values) for values in task_arguments]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(_build_episode, **values)
                    for values in task_arguments
                ]
                new_rows = [future.result() for future in futures]
        episode_rows = completed_rows + new_rows
        episode_rows.sort(key=lambda value: value["episode_ordinal"])
        write_json(
            staging / "manifest.json",
            {
                "schema": SCHEMA,
                "status": "pass",
                "research_only": True,
                "qualification_claim": False,
                "episode_count": len(selected),
                "episode_ids": list(selected),
                "visual_render_policy": (
                    "one_UE_render_per_episode_then_index_declared_audio_variants"
                ),
                "audio_variants_per_visual_episode": variants_per_episode,
                "scene_copy_count": 0,
                "frame_count": FRAME_COUNT,
                "frame_rate_hz": FRAME_RATE_HZ,
                "diagnostic_video_encoder": video_encoder,
                "diagnostic_encoder_gpu": encoder_gpu,
                "diagnostic_topdown_layout": "topdown_only_640x480",
                "episode_workers": workers,
                "source_asset_runtime_registry": {
                    "path": str(source_asset_registry),
                    "registry_id": source_registry["registry_id"],
                    "revision": source_registry["revision"],
                    "selected_asset_ids": sorted(
                        {
                            binding["asset_id"]
                            for episode_id in selected
                            for binding in bindings[episode_id].values()
                        }
                    ),
                },
                "resumed_episode_count": len(completed_rows),
                "new_episode_count": len(new_rows),
                "episodes": episode_rows,
                "inputs": {
                    "plan_root": str(plan_root),
                    "batch_root": str(batch_root),
                    "feasibility_root": str(feasibility_root),
                    "room_template_bundle": str(room_template_bundle),
                    "source_asset_registry": str(source_asset_registry),
                },
                "build_wall_seconds": time.perf_counter() - started,
            },
        )
        (staging / "failure.json").unlink(missing_ok=True)
        os.rename(staging, output)
    except BaseException as exc:
        if staging.exists():
            write_json(
                staging / "failure.json",
                {
                    "schema": "avengine_m7_apartment_ue_input_failure_v1",
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "completed_episode_count": len(
                        tuple(
                            (staging / "episodes").glob(
                                "*/metadata/build_record.json"
                            )
                        )
                    ),
                },
            )
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--feasibility-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--room-template-bundle", type=Path, required=True)
    parser.add_argument(
        "--source-asset-registry",
        type=Path,
        default=default_source_asset_runtime_registry_path(),
        help=(
            "Source asset runtime registry; changing it selects available "
            "animal/human geometry, emitter and Timeline profiles."
        ),
    )
    parser.add_argument("--episode-id", action="append")
    parser.add_argument(
        "--video-encoder", choices=("libx264", "h264_nvenc"), default="libx264"
    )
    parser.add_argument("--encoder-gpu", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume verified completed episodes from the fixed staging directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = build_bundle(
        plan_root=args.plan_root,
        feasibility_root=args.feasibility_root,
        batch_root=args.batch_root,
        room_template_bundle=args.room_template_bundle,
        source_asset_registry=args.source_asset_registry,
        episode_ids=args.episode_id,
        video_encoder=args.video_encoder,
        encoder_gpu=args.encoder_gpu,
        workers=args.workers,
        resume=args.resume,
        output=args.output,
    )
    print(f"ASSET_BOUND_APARTMENT_UE_BUNDLE_OK output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

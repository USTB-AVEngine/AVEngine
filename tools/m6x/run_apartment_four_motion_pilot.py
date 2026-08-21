#!/usr/bin/env python3
"""Run one shared Apartment capture for the four human/dog motion cases.

The pilot loads the room and articulated assets once, captures one continuous
300-frame master, renders one two-source binaural RIR grid, and slices the
result into four independent 5-second samples.  Every sample retains both a
clean RGB+binaural video and an RGB+Topdown+binaural review video together with
Timeline/source/flag metadata.  Wall-clock timing is intentionally part of the
pilot output because this runner is also a throughput probe for a later
800/100/100 training split.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image

from avengine.contracts.json_io import load_json, write_json
from avengine.m3.runtime import load_compiled_acoustic_scene
from avengine.m4.runtime import M4SimulationConfig
from avengine.m5_1.acoustics import (
    build_strided_review_keyframes,
    render_research_review_binaural_rir_sequence,
    research_review_trajectory_record,
)
from avengine.m6.audio_program import materialize_audio_program_variant
from avengine.m6x.apartment import listener_orientation_wxyz, qualify_fixed_apartment
from avengine.m6x.articulated_anchor_profile import (
    AnchorProfileSpec,
    compile_articulated_anchor_profile,
    materialize_articulated_anchor_paths,
)
from avengine.m6x.canary import (
    FPS,
    RIR_STRIDE_FRAMES,
    _asset_bindings,
    _fixed_acoustic_identity,
    _render_variant,
    _scenario_grid_and_sequence,
    _validated_acoustic_metadata,
    _validated_inputs,
    _write_review_index,
    _write_scenario_rir_evidence,
)
from avengine.m6x.capture_adapter import HUMAN_BEAGLE_CAPTURE_ADAPTER
from avengine.m6x.motion_matrix import (
    EPISODE_FRAME_COUNT,
    build_four_motion_master,
    motion_matrix_record,
)
from avengine.m6x.topdown import render_runtime_topdown_frames
from avengine.m6x.visual_profile import (
    load_review_visual_profile,
    validate_profile_capture_request,
    validate_realized_review_profile,
)


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_IDS = ("m6x_dog0_muzzle", "m6x_human0_mouth")
PILOT_SCHEMA = "avengine_apartment_four_motion_pilot_v1"
TIMING_SCHEMA = "avengine_apartment_four_motion_timing_v1"
REQUIRED_VIDEO_NAMES = (
    "videos/clean_binaural.mp4",
    "videos/diagnostic_topdown_binaural.mp4",
)


def _elapsed(started_at: float) -> float:
    value = time.perf_counter() - started_at
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"invalid monotonic wall-clock duration: {value}")
    return value


def _tree_size_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _episode_route_statistics(
    actor_root_paths: Mapping[str, np.ndarray], *, start: int, end: int
) -> dict[str, Any]:
    duration_seconds = (end - start) / FPS
    result: dict[str, Any] = {}
    for actor_id, full_path in sorted(actor_root_paths.items()):
        path = np.asarray(full_path[start:end], dtype=np.float64)
        steps = np.linalg.norm(np.diff(path[:, (0, 2)], axis=0), axis=1)
        distance = float(np.sum(steps))
        result[actor_id] = {
            "horizontal_distance_m": distance,
            "mean_episode_speed_m_s": distance / duration_seconds,
            "moving_step_mean_speed_m_s": (
                float(np.mean(steps[steps > 1.0e-12]) * FPS)
                if np.any(steps > 1.0e-12)
                else 0.0
            ),
        }
    return result


def _resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "config_root": args.config_root.resolve(),
        "runtime_root": args.runtime_root.resolve(),
        "room_manifest": args.room_manifest.resolve(),
        "m1_request": args.m1_request.resolve(),
        "room_registry": args.room_registry.resolve(),
        "entity_registry": args.entity_registry.resolve(),
        "endpoint_registry": args.endpoint_registry.resolve(),
        "sound_registry": args.sound_registry.resolve(),
        "human_runtime_glb": args.human_runtime_glb.resolve(),
        "animal_manifest": args.animal_manifest.resolve(),
        "animal_request": args.animal_request.resolve(),
        "beagle_audio": args.beagle_audio.resolve(),
        "acoustic_package_manifest": args.acoustic_package_manifest.resolve(),
        "m4_request": args.m4_request.resolve(),
        "hrtf": args.hrtf.resolve(),
        "review_visual_profile": args.review_visual_profile.resolve(),
        "exterior_proxy_glb": args.exterior_proxy_glb.resolve(),
    }


def run(args: argparse.Namespace) -> Path:
    run_started = time.perf_counter()
    phase_wall_seconds: dict[str, float] = {}
    episode_timings: dict[str, Any] = {}
    paths = _resolve_inputs(args)
    output = args.output.resolve()
    staging = output.with_name(f".{output.name}.staging")
    if os.path.lexists(output) or os.path.lexists(staging):
        raise RuntimeError(f"refusing to replace output or staging path: {output}")
    staging.mkdir(parents=True)

    try:
        phase_started = time.perf_counter()
        values = _validated_inputs(
            config_root=paths["config_root"],
            room_registry_path=paths["room_registry"],
            entity_registry_path=paths["entity_registry"],
            endpoint_registry_path=paths["endpoint_registry"],
            sound_registry_path=paths["sound_registry"],
        )
        visual_profile = load_review_visual_profile(paths["review_visual_profile"])
        validate_profile_capture_request(visual_profile, load_json(paths["m1_request"]))
        master = build_four_motion_master(values["anchors"])
        actor_root_paths = dict(master.actor_root_paths)
        actor_forwards = (
            HUMAN_BEAGLE_CAPTURE_ADAPTER.materialize_actor_fallback_forwards_xz(
                values["trajectories"], values["anchors"]
            )
        )
        scenario_template = deepcopy(
            next(
                scenario
                for scenario in values["suite"]["scenarios"]
                if scenario["scenario_id"] == "S4"
            )
        )
        program_ref = scenario_template["audio_program_ref"]
        base_program = values["programs"][
            (program_ref["program_id"], program_ref["revision"])
        ]
        program = materialize_audio_program_variant(
            base_program,
            "A",
            source_endpoint_registry=values["endpoints"],
            sound_asset_registry=values["sounds"],
        )
        if tuple(program["candidate_source_endpoint_ids"]) != SOURCE_IDS:
            raise RuntimeError(
                "the four-motion pilot requires canonical human/dog sources"
            )
        matrix_metadata = motion_matrix_record(master)
        for episode in matrix_metadata["episodes"]:
            episode["route_statistics"] = _episode_route_statistics(
                actor_root_paths,
                start=episode["start_frame"],
                end=episode["end_frame_exclusive"],
            )
        write_json(staging / "motion_matrix.json", matrix_metadata)
        phase_wall_seconds["input_and_route_planning"] = _elapsed(phase_started)

        phase_started = time.perf_counter()
        provisional_all = HUMAN_BEAGLE_CAPTURE_ADAPTER.provisional_source_paths(
            values["anchors"], actor_root_paths
        )
        provisional = {
            source_id: provisional_all[source_id] for source_id in SOURCE_IDS
        }
        preflight = qualify_fixed_apartment(
            room_manifest_path=paths["room_manifest"],
            m1_request_path=paths["m1_request"],
            anchor_library=values["anchors"],
            source_center_trajectories_m=provisional,
            runtime_root=paths["runtime_root"],
            minimum_navmesh_clearance_m=0.02,
        )
        if preflight.record["status"] != "pass":
            raise RuntimeError("four-motion root/source-center preflight failed")
        phase_wall_seconds["source_center_preflight"] = _elapsed(phase_started)

        phase_started = time.perf_counter()
        capture = HUMAN_BEAGLE_CAPTURE_ADAPTER.capture(
            room_manifest_path=paths["room_manifest"],
            m1_request_path=paths["m1_request"],
            provider_assets={
                "human_runtime_glb_path": paths["human_runtime_glb"],
                "animal_manifest_path": paths["animal_manifest"],
                "animal_request_path": paths["animal_request"],
                "review_visual_profile_path": paths["review_visual_profile"],
                "exterior_proxy_glb_path": paths["exterior_proxy_glb"],
            },
            actor_root_paths=actor_root_paths,
            actor_fallback_forwards_xz=actor_forwards,
            output_dir=staging / "shared/master_capture",
            runtime_root=paths["runtime_root"],
            route_provenance={
                "route_family": "apartment_four_motion_master_300_v1",
                "source": "motion_matrix.json",
                "placement_semantics": "source_center_only",
                "room_loaded_once": True,
                "episode_frame_count": EPISODE_FRAME_COUNT,
            },
        )
        validate_realized_review_profile(
            capture.evidence,
            profile=visual_profile,
            exterior_proxy_glb_path=paths["exterior_proxy_glb"],
        )
        phase_wall_seconds["habitat_capture_300_frames"] = _elapsed(phase_started)

        phase_started = time.perf_counter()
        actual_all = HUMAN_BEAGLE_CAPTURE_ADAPTER.actual_source_paths(
            values["anchors"], capture
        )
        source_paths = {source_id: actual_all[source_id] for source_id in SOURCE_IDS}
        anchor_profile = compile_articulated_anchor_profile(
            actor_world_matrices=capture.actor_world_matrices,
            frame_records=capture.records,
            specs=(
                AnchorProfileSpec(
                    source_endpoint_id="m6x_human0_mouth",
                    actor_id="human0",
                    asset_id=HUMAN_BEAGLE_CAPTURE_ADAPTER.actor_binding(
                        "human0"
                    ).asset_id,
                    record_key="human",
                    anchor_id="mouth",
                    anchor_record_key="mouth_emitter_anchor_m",
                    capture_matrix_index=0,
                    local_anatomical_forward_axis=(0.0, 0.0, 1.0),
                    action_sample_counts=capture.evidence["runtime"][
                        "human_action_sample_counts"
                    ],
                ),
                AnchorProfileSpec(
                    source_endpoint_id="m6x_dog0_muzzle",
                    actor_id="dog0",
                    asset_id=HUMAN_BEAGLE_CAPTURE_ADAPTER.actor_binding(
                        "dog0"
                    ).asset_id,
                    record_key="beagle",
                    anchor_id="muzzle",
                    anchor_record_key="mouth_emitter_anchor_m",
                    capture_matrix_index=1,
                    local_anatomical_forward_axis=(1.0, 0.0, 0.0),
                    action_sample_counts=capture.evidence["runtime"][
                        "beagle_action_sample_counts"
                    ],
                ),
            ),
        )
        profiled_paths = materialize_articulated_anchor_paths(
            anchor_profile,
            actor_root_paths=actor_root_paths,
            actor_fallback_forwards_xz=actor_forwards,
        )
        profile_errors = {
            source_id: float(
                np.max(
                    np.linalg.norm(
                        profiled_paths[source_id] - source_paths[source_id], axis=1
                    )
                )
            )
            for source_id in SOURCE_IDS
        }
        if max(profile_errors.values()) > 2.0e-5:
            raise RuntimeError(
                f"articulated anchor profile reconstruction failed: {profile_errors}"
            )
        write_json(staging / "shared/articulated_anchor_profile.json", anchor_profile)
        write_json(
            staging / "shared/articulated_anchor_profile_verification.json",
            {
                "schema": "avengine_articulated_anchor_profile_verification_v1",
                "status": "pass",
                "visual_observation_calls_for_reuse": 0,
                "maximum_reconstruction_error_m": profile_errors,
                "maximum_allowed_error_m": 2.0e-5,
            },
        )
        qualification = qualify_fixed_apartment(
            room_manifest_path=paths["room_manifest"],
            m1_request_path=paths["m1_request"],
            anchor_library=values["anchors"],
            source_center_trajectories_m=source_paths,
            runtime_root=paths["runtime_root"],
            minimum_navmesh_clearance_m=0.02,
        )
        if qualification.record["status"] != "pass":
            raise RuntimeError("captured human/dog source-center qualification failed")
        write_json(staging / "room/qualification.json", qualification.record)
        shutil.copy2(
            paths["config_root"] / "room_capsule.json",
            staging / "room/room_capsule.json",
        )
        listener_position = qualification.record["listener"]["position_m"]
        listener_yaw = float(qualification.record["listener"]["yaw_deg"])
        listener_orientation = listener_orientation_wxyz(listener_yaw)
        overview = render_runtime_topdown_frames(
            qualification.obstacle_map,
            {
                source_id: trajectory[:1]
                for source_id, trajectory in source_paths.items()
            },
            listener_position_m=listener_position,
            listener_yaw_deg=listener_yaw,
            camera_hfov_degrees=qualification.record["listener"]["camera_hfov_degrees"],
            source_labels={
                "m6x_dog0_muzzle": "Beagle",
                "m6x_human0_mouth": "Human",
            },
        )[0]
        Image.fromarray(overview, mode="RGB").save(
            staging / "room/runtime_obstacle_map.png"
        )
        phase_wall_seconds["captured_source_center_qualification"] = _elapsed(
            phase_started
        )

        phase_started = time.perf_counter()
        master_grid = build_strided_review_keyframes(
            source_paths,
            visual_frame_rate_hz=FPS,
            rir_stride_frames=RIR_STRIDE_FRAMES,
            listener_position_m=listener_position,
            listener_orientation_wxyz=listener_orientation,
        )
        scene = load_compiled_acoustic_scene(
            paths["acoustic_package_manifest"],
            allow_nonpassing_research_qa=True,
        )
        acoustic_identity = _fixed_acoustic_identity(
            scene,
            room_capsule=values["room_capsule"],
            room_registry=values["room_registry"],
        )
        simulation = M4SimulationConfig.from_mapping(
            load_json(paths["m4_request"])["simulation"]
        )
        master_sequence = render_research_review_binaural_rir_sequence(
            scene,
            simulation,
            grid=master_grid,
            hrtf_file_path=str(paths["hrtf"]),
        )
        master_sequence = replace(
            master_sequence,
            metadata=_validated_acoustic_metadata(
                master_sequence.metadata,
                scene=scene,
                simulation=simulation,
                hrtf_file_path=paths["hrtf"],
                acoustic_identity=acoustic_identity,
            ),
        )
        acoustics_root = staging / "shared/acoustics"
        acoustics_root.mkdir(parents=True)
        np.save(
            acoustics_root / "samples.npy", master_sequence.samples, allow_pickle=False
        )
        np.save(
            acoustics_root / "lengths.npy", master_sequence.lengths, allow_pickle=False
        )
        write_json(acoustics_root / "metadata.json", master_sequence.metadata)
        write_json(
            acoustics_root / "trajectory.json",
            research_review_trajectory_record(master_grid),
        )
        phase_wall_seconds["rlr_binaural_rir_300_frames"] = _elapsed(phase_started)

        bindings = _asset_bindings(
            values["sounds"],
            repository_root=REPOSITORY,
            external_sound_asset_paths={
                "dog_beagle_v2_scheduled_dry": paths["beagle_audio"]
            },
        )
        rows: list[dict[str, Any]] = []
        for index, episode in enumerate(master.episodes):
            episode_started = time.perf_counter()
            scenario = deepcopy(scenario_template)
            scenario["scenario_id"] = f"P{index}"
            scenario["purpose"] = episode.episode_id
            scenario["capture_frame_window"] = {
                "start_frame": episode.start_frame,
                "end_frame_exclusive": episode.end_frame_exclusive,
            }
            scenario["motion"] = {
                "human0": episode.human_motion,
                "dog0": episode.dog_motion,
            }
            for binding in scenario["source_bindings"]:
                binding["trajectory_template_id"] = (
                    "apartment_four_motion_master_300_v1"
                )
                binding["trajectory_route_id"] = episode.episode_id

            phase_started = time.perf_counter()
            grid, sequence, trajectories = _scenario_grid_and_sequence(
                master_grid,
                master_sequence,
                source_paths=source_paths,
                candidate_source_ids=SOURCE_IDS,
                start_frame=episode.start_frame,
                end_frame_exclusive=episode.end_frame_exclusive,
                listener_position_m=listener_position,
                listener_orientation=listener_orientation,
            )
            episode_root = staging / "episodes" / f"{index:02d}_{episode.episode_id}"
            rir_metadata_path = _write_scenario_rir_evidence(
                episode_root,
                scenario_id=scenario["scenario_id"],
                grid=grid,
                sequence=sequence,
            )
            slice_seconds = _elapsed(phase_started)

            phase_started = time.perf_counter()
            row = _render_variant(
                variant_root=episode_root,
                scenario=scenario,
                variant_id="A",
                program=program,
                capture=capture,
                window_start=episode.start_frame,
                rgb=np.ascontiguousarray(
                    capture.rgb[episode.start_frame : episode.end_frame_exclusive]
                ),
                semantic=np.ascontiguousarray(
                    capture.semantic[episode.start_frame : episode.end_frame_exclusive]
                ),
                grid=grid,
                sequence=sequence,
                trajectories=trajectories,
                qualification=qualification,
                listener_position_m=listener_position,
                listener_yaw_deg=listener_yaw,
                listener_orientation=listener_orientation,
                camera_hfov_degrees=float(
                    qualification.record["listener"]["camera_hfov_degrees"]
                ),
                endpoints=values["endpoints"],
                sounds=values["sounds"],
                asset_bindings=bindings,
                rir_metadata_path=rir_metadata_path,
                rir_bundle_uri=(
                    "bundle://"
                    + rir_metadata_path.parent.relative_to(staging).as_posix()
                ),
                visual_profile=visual_profile,
            )
            render_seconds = _elapsed(phase_started)
            for required_name in REQUIRED_VIDEO_NAMES:
                if not (episode_root / required_name).is_file():
                    raise RuntimeError(
                        f"episode {episode.episode_id} lacks {required_name}"
                    )
            route_statistics = _episode_route_statistics(
                actor_root_paths,
                start=episode.start_frame,
                end=episode.end_frame_exclusive,
            )
            write_json(
                episode_root / "metadata/motion_case.json",
                {
                    "schema": "avengine_apartment_motion_case_v1",
                    "episode_id": episode.episode_id,
                    "motion": scenario["motion"],
                    "route_statistics": route_statistics,
                    "both_sources_have_events": True,
                    "overlapping_source_events": True,
                },
            )
            episode_timings[episode.episode_id] = {
                "rir_slice_and_evidence_wall_seconds": slice_seconds,
                "audio_video_labels_render_wall_seconds": render_seconds,
                "episode_pipeline_wall_seconds": _elapsed(episode_started),
                "retained_size_bytes_before_shared_cleanup": _tree_size_bytes(
                    episode_root
                ),
            }
            row["purpose"] = episode.episode_id
            rows.append(row)

        phase_started = time.perf_counter()
        review_index = _write_review_index(
            staging,
            rows,
            listener_position_m=listener_position,
            listener_yaw_deg=listener_yaw,
        )
        write_json(
            staging / "pilot_manifest.json",
            {
                "schema": PILOT_SCHEMA,
                "status": "pass",
                "room_id": qualification.record["room_id"],
                "room_loaded_once_for_master_capture": True,
                "scene_asset_copied_per_episode": False,
                "frame_contract": {
                    "episode_count": 4,
                    "seconds_per_episode": 5,
                    "frames_per_episode": EPISODE_FRAME_COUNT,
                    "frame_rate_hz": FPS,
                    "master_frame_count": master.frame_count,
                },
                "audio_contract": {
                    "layout": "binaural",
                    "channel_count": 2,
                    "sample_rate_hz": 16_000,
                    "sources": list(SOURCE_IDS),
                    "both_sources_have_events_in_every_episode": True,
                    "events_overlap_in_every_episode": True,
                    "audio_program_id": program["program_id"],
                },
                "required_outputs_per_episode": [
                    *REQUIRED_VIDEO_NAMES,
                    "metadata/timeline.json",
                    "metadata/source_manifest.json",
                    "metadata/flags.json",
                    "metadata/audio_program.json",
                ],
                "motion_matrix": matrix_metadata["episodes"],
                "retention": {
                    "keep_dense_master": bool(args.keep_dense_master),
                    "policy": (
                        "retain reusable RGB/semantic/RIR master arrays"
                        if args.keep_dense_master
                        else "retain final media/labels and per-episode RIR evidence; "
                        "discard dense reusable master arrays after rendering"
                    ),
                },
                "input_paths": {
                    key: str(value) for key, value in sorted(paths.items())
                },
            },
        )
        phase_wall_seconds["review_index_and_manifest"] = _elapsed(phase_started)

        dense_size_before_cleanup = sum(
            _tree_size_bytes(path)
            for path in (
                staging / "shared/master_capture",
                staging / "shared/acoustics",
            )
            if path.exists()
        )
        phase_started = time.perf_counter()
        if not args.keep_dense_master:
            for path in (
                staging / "shared/master_capture/arrays",
                staging / "shared/master_capture/runtime",
                staging / "shared/acoustics",
            ):
                if path.exists():
                    shutil.rmtree(path)
            write_json(
                staging / "shared/master_capture/RETENTION.json",
                {
                    "schema": "avengine_transient_master_retention_v1",
                    "status": "intentionally_not_reusable",
                    "reason": (
                        "dense capture and shared RIR arrays were transient inputs to "
                        "the four retained 5-second pilot samples"
                    ),
                    "scene_asset_was_not_copied_per_episode": True,
                },
            )
        phase_wall_seconds["dense_intermediate_cleanup"] = _elapsed(phase_started)

        total_seconds = _elapsed(run_started)
        episode_render_total = sum(
            value["episode_pipeline_wall_seconds"] for value in episode_timings.values()
        )
        timing = {
            "schema": TIMING_SCHEMA,
            "status": "pass",
            "clock": "time.perf_counter",
            "measurement_scope": (
                "fresh route preflight, one 300-frame Habitat capture, source-center "
                "qualification, one 300-frame/two-source RLR render, four audio/video/"
                "label renders, readback gates, and cleanup"
            ),
            "phase_wall_seconds": phase_wall_seconds,
            "episode_timings": episode_timings,
            "run_total_wall_seconds": total_seconds,
            "throughput": {
                "habitat_capture_frames_per_second": (
                    master.frame_count
                    / phase_wall_seconds["habitat_capture_300_frames"]
                ),
                "fresh_pipeline_episodes_per_hour": 4 * 3600.0 / total_seconds,
                "naive_serial_1000_episode_projection_hours": (
                    total_seconds * 1000.0 / (4 * 3600.0)
                ),
                "post_capture_episode_render_episodes_per_hour": (
                    4 * 3600.0 / episode_render_total
                ),
                "projection_warning": (
                    "The 1000-sample figure is a linear four-sample pilot projection, "
                    "not a production benchmark; shared startup and batch scheduling "
                    "can change it."
                ),
            },
            "storage": {
                "dense_shared_bytes_before_cleanup": dense_size_before_cleanup,
                "dense_shared_retained": bool(args.keep_dense_master),
                "retained_pilot_bytes_before_timing_json": _tree_size_bytes(staging),
                "scene_copies_per_episode": 0,
            },
        }
        write_json(staging / "timing.json", timing)
        os.rename(staging, output)
    except Exception:
        write_json(
            staging / "FAILED_TIMING.json",
            {
                "schema": TIMING_SCHEMA,
                "status": "fail",
                "phase_wall_seconds": phase_wall_seconds,
                "episode_timings": episode_timings,
                "run_total_wall_seconds": _elapsed(run_started),
            },
        )
        raise

    print(
        "APARTMENT_FOUR_MOTION_PILOT_OK "
        f"output={output} review={output / review_index.name} "
        f"timing={output / 'timing.json'}",
        flush=True,
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-root",
        type=Path,
        default=REPOSITORY / "examples/m6x/fixed_apartment",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--room-manifest",
        type=Path,
        default=REPOSITORY / "tmp/m1/legacy_apartment_package/room_manifest.json",
    )
    parser.add_argument(
        "--m1-request",
        type=Path,
        default=REPOSITORY
        / "examples/m6x/fixed_apartment/m1_capture_request_review_720p.json",
    )
    parser.add_argument(
        "--room-registry",
        type=Path,
        default=REPOSITORY / "examples/m6/rooms/room_registry.json",
    )
    parser.add_argument(
        "--entity-registry",
        type=Path,
        default=REPOSITORY / "examples/m6/registries/entity_assets_v1.json",
    )
    parser.add_argument(
        "--endpoint-registry",
        type=Path,
        default=REPOSITORY / "examples/m6/registries/source_endpoints_v1.json",
    )
    parser.add_argument(
        "--sound-registry",
        type=Path,
        default=REPOSITORY / "examples/m6/registries/sound_assets_v1.json",
    )
    parser.add_argument(
        "--human-runtime-glb",
        type=Path,
        default=REPOSITORY.parent
        / "AVEngine/external/SPEAR/tmp/rocketbox_native_runtime_ue_v3/"
        "rocketbox_male_adult_01_original_ue_v3/runtime.glb",
    )
    parser.add_argument(
        "--animal-manifest",
        type=Path,
        default=REPOSITORY
        / "tmp/m2/rocketbox_beagle_m2_canary_v7_world_contact_r5/asset_manifest.json",
    )
    parser.add_argument(
        "--animal-request",
        type=Path,
        default=REPOSITORY
        / "tmp/m2/rocketbox_beagle_m2_formal_request_v7_world_contact_r5.json",
    )
    parser.add_argument(
        "--beagle-audio",
        type=Path,
        default=REPOSITORY.parent
        / "AVEngine/external/SPEAR/tmp/animal_audio_event_audit_v1/"
        "dog_beagle_v2_scheduled_dry.wav",
    )
    parser.add_argument(
        "--acoustic-package-manifest",
        type=Path,
        default=REPOSITORY / "tmp/m3/root_ue_package_current_20260718_02/manifest.json",
    )
    parser.add_argument(
        "--m4-request",
        type=Path,
        default=REPOSITORY
        / "examples/m4/blender_custom/multi_source_canary_request.json",
    )
    parser.add_argument(
        "--hrtf",
        type=Path,
        default=Path("/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa"),
    )
    parser.add_argument(
        "--review-visual-profile",
        type=Path,
        default=REPOSITORY / "examples/m6x/fixed_apartment/review_visual_profile.json",
    )
    parser.add_argument(
        "--exterior-proxy-glb",
        type=Path,
        default=REPOSITORY / "tmp/m6x/assets/approaching_storm_4k_exterior_v3/"
        "approaching_storm_4k_exterior.glb",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--keep-dense-master",
        action="store_true",
        help="Keep the reusable RGB/semantic master arrays and shared RIR grid.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

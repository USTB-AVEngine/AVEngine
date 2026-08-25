#!/usr/bin/env python3
"""Build one hash-bound MP3D room-evaluation listening review.

This tool does not render Habitat observations or RLR impulse responses.  It
joins one completed room-evaluation binaural delivery to a retained Habitat
RGB/Topdown capture, after proving that both sides use the same complete
SensorRigTrajectory and that every displayed source position is the exact
same-frame position retained by the RIR plan.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)
from avengine.spatial_audio.audio import read_float32_wav
from avengine.timeline.metrics import listener_local_source_geometry
from avengine.capture.review import (
    SourceOverlayTrack,
    compose_annotated_frames,
    encode_annotated_review,
)
from avengine.acoustics.rir_cache import validate_rir_job_plan
from avengine.m6x.room_feasibility import (
    RIR_JOB_PLAN_SCHEMA,
    TRAJECTORY_BANK_SCHEMA,
)
from avengine.m7.room_evaluation import (
    ROOM_EVALUATION_PLAN_SCHEMA,
    ROOM_SOUND_ASSIGNMENTS_SCHEMA,
)
from avengine.m7.sensor_rig import (
    m7_sensor_rig_binding,
    m7_sensor_rig_pose_series,
    validate_m7_rir_listener_alignment,
)


SCHEMA = "avengine_m7_mp3d_room_evaluation_review_v1"
FRAME_COUNT = 75
FRAME_RATE_HZ = 15
AUDIO_SAMPLE_RATE_HZ = 16_000
AUDIO_SAMPLE_COUNT = 80_000
SOURCE_SLOTS = ("source1", "source2")
EXPECTED_ROOM_REF = {
    "registry_id": "avengine_m6_representative_rooms_v1",
    "room_id": "habitat_mp3d_example_17DRP5sb8fy",
    "revision": "raw_v1_plus_declared_proxy_v2_research",
}
EXPECTED_BINDING_ID = "mp3d_17DRP5sb8fy_soundspaces2_v1"
EXPECTED_PROFILE_ID = "soundspaces2_mp3d_public_materials_v1"
EXPECTED_SCENE_ID = "17DRP5sb8fy"
REVIEW_FILENAME = "mp3d_soundspaces2_room_evaluation_binaural_doa_review.mp4"


class MP3DRoomEvaluationReviewError(RuntimeError):
    """A retained visual/audio pair cannot form an exact MP3D review."""


@dataclass(frozen=True)
class MP3DRoomEvaluationReviewInputs:
    """Validated arrays, identities and geometry for one review episode."""

    episode_id: str
    sample_id: str
    source_classes: Mapping[str, str]
    source_positions_m: Mapping[str, np.ndarray]
    listener_positions_m: np.ndarray
    listener_orientations_wxyz: np.ndarray
    listener_yaws_deg: np.ndarray
    pose_hashes: tuple[str, ...]
    frame_geometry: tuple[Mapping[str, Any], ...]
    rgb: np.ndarray
    topdown_rgb: np.ndarray
    audio_path: Path
    acoustic_selection_binding: Mapping[str, Any]
    acoustic_selection_binding_sha256: str
    sensor_rig_binding: Mapping[str, Any]
    evidence_inputs: Mapping[str, Any]
    alignment: Mapping[str, Any]


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MP3DRoomEvaluationReviewError(f"required input is missing: {path}")
    return {
        "path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _within_file(root: Path, relative: Any, *, owner: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise MP3DRoomEvaluationReviewError(f"{owner} must be a relative path")
    try:
        path = (root / relative).resolve(strict=True)
    except OSError as exc:
        raise MP3DRoomEvaluationReviewError(f"{owner} is missing") from exc
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise MP3DRoomEvaluationReviewError(f"{owner} escapes its delivery root") from exc
    if not path.is_file():
        raise MP3DRoomEvaluationReviewError(f"{owner} is not a file")
    return path


def _selection_binding(
    delivery: Mapping[str, Any],
    samples: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    binding = delivery.get("acoustic_selection_binding")
    samples_binding = samples.get("acoustic_selection_binding")
    if (
        not isinstance(binding, Mapping)
        or dict(binding) != samples_binding
        or binding.get("schema")
        != "avengine_rir_cache_acoustic_selection_binding_v1"
        or binding.get("selection_mode") != "registry"
        or binding.get("registry_selection_applied") is not True
        or binding.get("room_ref") != EXPECTED_ROOM_REF
        or binding.get("binding_id") != EXPECTED_BINDING_ID
        or not isinstance(binding.get("profile_ref"), Mapping)
        or binding["profile_ref"].get("profile_id") != EXPECTED_PROFILE_ID
    ):
        raise MP3DRoomEvaluationReviewError(
            "audio is not registry-bound to MP3D 17DRP5sb8fy SoundSpaces2"
        )
    normalized = dict(binding)
    declared = normalized.get("binding_content_sha256")
    actual = canonical_json_sha256(
        {
            key: value
            for key, value in normalized.items()
            if key != "binding_content_sha256"
        }
    )
    if (
        declared != actual
        or delivery.get("acoustic_selection_binding_sha256") != actual
        or samples.get("acoustic_selection_binding_sha256") != actual
    ):
        raise MP3DRoomEvaluationReviewError(
            "audio acoustic-selection binding hash differs"
        )
    return normalized, actual


def _one_audio_sample(
    audio_root: Path,
    binding: Mapping[str, Any],
    binding_sha256: str,
    delivery: Mapping[str, Any],
    samples: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Path, Path]:
    raw_samples = samples.get("samples")
    output_closure = delivery.get("output_closure")
    closure_files = (
        output_closure.get("files")
        if isinstance(output_closure, Mapping)
        else None
    )
    if (
        delivery.get("schema") != "avengine_room_evaluation_binaural_batch_v1"
        or delivery.get("status") != "pass"
        or delivery.get("research_only") is not True
        or delivery.get("qualification_claim") is not False
        or delivery.get("sample_count") != 1
        or delivery.get("source_slots") != list(SOURCE_SLOTS)
        or delivery.get("both_sources_active") is not True
        or delivery.get("layout") != "native_RLR_HRTF_binaural_left_right"
        or delivery.get("mixture_is_exact_persisted_source_stem_sum") is not True
        or samples.get("schema")
        != "avengine_room_evaluation_binaural_samples_v1"
        or samples.get("status") != "pass"
        or samples.get("sample_count") != 1
        or not isinstance(raw_samples, list)
        or len(raw_samples) != 1
        or not isinstance(closure_files, Mapping)
        or not isinstance(closure_files.get("samples.json"), Mapping)
        or closure_files["samples.json"].get("sha256")
        != sha256_file(audio_root / "samples.json")
    ):
        raise MP3DRoomEvaluationReviewError(
            "audio delivery/samples closure is not one passing binaural episode"
        )
    sample = raw_samples[0]
    if not isinstance(sample, Mapping):
        raise MP3DRoomEvaluationReviewError("audio sample record is malformed")
    classes = sample.get("source_classes")
    if (
        not isinstance(sample.get("sample_id"), str)
        or not sample["sample_id"]
        or not isinstance(sample.get("episode_id"), str)
        or not sample["episode_id"]
        or sample.get("both_sources_active") is not True
        or not isinstance(classes, Mapping)
        or set(classes) != set(SOURCE_SLOTS)
        or any(not isinstance(classes[slot], str) or not classes[slot] for slot in SOURCE_SLOTS)
        or classes["source1"] == classes["source2"]
        or sample.get("audio_sample_rate_hz") != AUDIO_SAMPLE_RATE_HZ
        or sample.get("audio_sample_count") != AUDIO_SAMPLE_COUNT
        or sample.get("audio_channel_count") != 2
        or sample.get("mixture_is_exact_persisted_source_stem_sum") is not True
        or sample.get("acoustic_selection_binding_sha256") != binding_sha256
    ):
        raise MP3DRoomEvaluationReviewError("audio sample contract differs")
    audio_path = _within_file(
        audio_root, sample.get("audio_path"), owner="sample mixture WAV"
    )
    sidecar_path = _within_file(
        audio_root,
        sample.get("audio_sidecar_path"),
        owner="sample mixture WAVE sidecar",
    )
    if (
        sample.get("audio_sha256") != sha256_file(audio_path)
        or sample.get("audio_sidecar_sha256") != sha256_file(sidecar_path)
    ):
        raise MP3DRoomEvaluationReviewError("sample WAVE or sidecar hash differs")
    decoded = read_float32_wav(
        audio_path, sidecar_path=sidecar_path, verify_sidecar=True
    )
    sidecar = decoded.sidecar
    metadata = sidecar.get("metadata") if isinstance(sidecar, Mapping) else None
    if (
        decoded.sample_rate_hz != AUDIO_SAMPLE_RATE_HZ
        or decoded.samples.shape != (2, AUDIO_SAMPLE_COUNT)
        or not isinstance(metadata, Mapping)
        or metadata.get("role") != "room_evaluation_binaural_mixture"
        or metadata.get("sample_id") != sample["sample_id"]
        or metadata.get("episode_id") != sample["episode_id"]
        or metadata.get("source_classes") != dict(classes)
        or metadata.get("acoustic_selection_binding_sha256") != binding_sha256
        or metadata.get("normalization") is not False
        or metadata.get("limiting") is not False
    ):
        raise MP3DRoomEvaluationReviewError(
            "decoded binaural WAVE/sidecar contract differs"
        )
    if dict(binding) != delivery["acoustic_selection_binding"]:
        raise AssertionError("selection binding changed during audio validation")
    return sample, audio_path, sidecar_path


def _plan_inputs(
    plan_root: Path,
    *,
    episode_id: str,
    source_classes: Mapping[str, str],
    audio_input_closure: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    dict[str, Any],
]:
    paths = {
        name: plan_root / name
        for name in (
            "delivery.json",
            "trajectory_bank.json",
            "sound_assignments.json",
            "rir_job_plan.json",
            "sensor_rig_trajectory.json",
        )
    }
    if any(not path.is_file() for path in paths.values()):
        raise MP3DRoomEvaluationReviewError("plan closure is incomplete")
    plan_delivery = load_json(paths["delivery.json"])
    bank = load_json(paths["trajectory_bank.json"])
    assignments = load_json(paths["sound_assignments.json"])
    rir_plan = load_json(paths["rir_job_plan.json"])
    sensor_rig = load_json(paths["sensor_rig_trajectory.json"])
    binding = m7_sensor_rig_binding(sensor_rig)
    poses = m7_sensor_rig_pose_series(sensor_rig)
    if (
        plan_delivery.get("schema") != ROOM_EVALUATION_PLAN_SCHEMA
        or plan_delivery.get("status") != "pass"
        or plan_delivery.get("research_only") is not True
        or plan_delivery.get("qualification_claim") is not False
        or plan_delivery.get("episode_count") != 1
        or plan_delivery.get("frame_count") != FRAME_COUNT
        or plan_delivery.get("frame_rate_hz") != FRAME_RATE_HZ
        or plan_delivery.get("listener_pose_mode")
        != "sensor_rig_trajectory_v1"
        or not isinstance(plan_delivery.get("sensor_rig_trajectory"), Mapping)
        or plan_delivery["sensor_rig_trajectory"].get("trajectory_id")
        != binding["trajectory_id"]
        or plan_delivery["sensor_rig_trajectory"].get("content_sha256")
        != binding["content_sha256"]
    ):
        raise MP3DRoomEvaluationReviewError(
            "room-evaluation plan does not bind one complete dynamic SensorRigTrajectory"
        )
    closure_binding = audio_input_closure.get("sensor_rig_trajectory")
    closure_files = audio_input_closure.get("files")
    if (
        not isinstance(closure_binding, Mapping)
        or closure_binding.get("trajectory_id") != binding["trajectory_id"]
        or closure_binding.get("content_sha256") != binding["content_sha256"]
        or not isinstance(closure_files, Mapping)
        or any(
            not isinstance(closure_files.get(name), Mapping)
            or closure_files[name].get("sha256") != sha256_file(path)
            for name, path in paths.items()
        )
    ):
        raise MP3DRoomEvaluationReviewError(
            "audio input closure differs from the current room-evaluation plan"
        )
    episodes = bank.get("episodes")
    if (
        bank.get("schema") != TRAJECTORY_BANK_SCHEMA
        or bank.get("episode_count") != 1
        or bank.get("frame_count") != FRAME_COUNT
        or bank.get("frame_rate_hz") != FRAME_RATE_HZ
        or bank.get("source_slots") != list(SOURCE_SLOTS)
        or not isinstance(episodes, list)
        or len(episodes) != 1
        or not isinstance(episodes[0], Mapping)
        or episodes[0].get("episode_id") != episode_id
    ):
        raise MP3DRoomEvaluationReviewError(
            "trajectory bank differs from the one audio episode"
        )
    raw_paths = episodes[0].get("source_center_paths_m")
    if not isinstance(raw_paths, Mapping) or set(raw_paths) != set(SOURCE_SLOTS):
        raise MP3DRoomEvaluationReviewError(
            "trajectory bank lacks source1/source2 center paths"
        )
    source_positions: dict[str, np.ndarray] = {}
    for slot in SOURCE_SLOTS:
        try:
            points = np.asarray(raw_paths[slot], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MP3DRoomEvaluationReviewError(
                f"trajectory bank {slot} path is invalid"
            ) from exc
        if points.shape != (FRAME_COUNT, 3) or not np.all(np.isfinite(points)):
            raise MP3DRoomEvaluationReviewError(
                f"trajectory bank {slot} path is invalid"
            )
        source_positions[slot] = np.ascontiguousarray(points)
    raw_assignments = assignments.get("assignments")
    if (
        assignments.get("schema") != ROOM_SOUND_ASSIGNMENTS_SCHEMA
        or assignments.get("status") != "pass"
        or assignments.get("episode_count") != 1
        or assignments.get("both_sources_active") is not True
        or not isinstance(raw_assignments, list)
        or len(raw_assignments) != 1
        or raw_assignments[0].get("episode_id") != episode_id
        or raw_assignments[0].get("source_classes") != dict(source_classes)
    ):
        raise MP3DRoomEvaluationReviewError(
            "plan sound assignment differs from the delivered audio sample"
        )
    if rir_plan.get("schema") != RIR_JOB_PLAN_SCHEMA:
        raise MP3DRoomEvaluationReviewError("RIR plan schema differs")
    jobs = validate_rir_job_plan(rir_plan)
    try:
        listener_alignment = validate_m7_rir_listener_alignment(
            rir_job_plan=rir_plan,
            sensor_rig_trajectory=sensor_rig,
        )
    except ValueError as exc:
        raise MP3DRoomEvaluationReviewError(str(exc)) from exc
    positions_by_use: dict[tuple[str, int], tuple[float, ...]] = {}
    for job in jobs:
        source_position = tuple(float(value) for value in job["source_position_m"])
        for use in job["uses"]:
            if use["episode_id"] != episode_id:
                raise MP3DRoomEvaluationReviewError(
                    "RIR plan contains a different episode"
                )
            key = (str(use["source_slot_id"]), int(use["frame_index"]))
            if key in positions_by_use:
                raise MP3DRoomEvaluationReviewError(
                    "RIR plan repeats an episode source frame"
                )
            positions_by_use[key] = source_position
    expected_uses = {
        (slot, frame_index)
        for slot in SOURCE_SLOTS
        for frame_index in range(FRAME_COUNT)
    }
    if set(positions_by_use) != expected_uses:
        raise MP3DRoomEvaluationReviewError(
            "RIR plan does not cover every displayed source frame"
        )
    for slot, frame_index in sorted(expected_uses):
        if not np.array_equal(
            np.asarray(positions_by_use[(slot, frame_index)], dtype=np.float64),
            source_positions[slot][frame_index],
        ):
            raise MP3DRoomEvaluationReviewError(
                f"RIR plan {slot} position differs at frame {frame_index}"
            )
    return (
        sensor_rig,
        source_positions,
        poses.positions_m,
        poses.orientations_wxyz,
        poses.yaws_deg,
        poses.pose_hashes,
        {
            "status": "pass",
            "sensor_rig_trajectory": dict(binding),
            "rir_listener_alignment": listener_alignment,
            "source_position_use_count": len(positions_by_use),
            "all_source_positions_match_trajectory_bank": True,
            "plan_files": {
                name: _file_record(path) for name, path in paths.items()
            },
        },
    )


def _visual_inputs(
    visual_root: Path,
    *,
    plan_sensor_rig: Mapping[str, Any],
    sensor_rig_binding: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    paths = {
        "summary": visual_root / "summary.json",
        "sensor_rig_trajectory": visual_root / "sensor_rig_trajectory.json",
        "rgb": visual_root / "rgb.npy",
        "topdown_rgb": visual_root / "topdown_rgb.npy",
    }
    if any(not path.is_file() for path in paths.values()):
        raise MP3DRoomEvaluationReviewError("visual capture closure is incomplete")
    summary = load_json(paths["summary"])
    visual_sensor_rig = load_json(paths["sensor_rig_trajectory"])
    visual_binding = m7_sensor_rig_binding(visual_sensor_rig)
    if (
        visual_sensor_rig != plan_sensor_rig
        or visual_binding != sensor_rig_binding
    ):
        raise MP3DRoomEvaluationReviewError(
            "visual capture SensorRigTrajectory content differs from the plan"
        )
    scene = summary.get("scene")
    trajectory = summary.get("trajectory")
    visual = summary.get("visual")
    scene_id = scene.get("scene_id") if isinstance(scene, Mapping) else None
    scene_path = Path(scene_id) if isinstance(scene_id, str) and scene_id else None
    scene_identity_matches = (
        scene_id == EXPECTED_SCENE_ID
        or (
            scene_path is not None
            and scene_path.name == f"{EXPECTED_SCENE_ID}.glb"
        )
    )
    sensor_artifact = (
        summary.get("artifacts", {}).get("sensor_rig_trajectory")
        if isinstance(summary.get("artifacts"), Mapping)
        else None
    )
    if (
        summary.get("status") not in {"pass", "visual_pass_audio_blocked"}
        or not isinstance(scene, Mapping)
        or scene.get("room_id") != EXPECTED_ROOM_REF["room_id"]
        or not scene_identity_matches
        or not isinstance(trajectory, Mapping)
        or trajectory.get("trajectory_id") != sensor_rig_binding["trajectory_id"]
        or trajectory.get("frame_count") != FRAME_COUNT
        or not isinstance(visual, Mapping)
        or not isinstance(sensor_artifact, Mapping)
        or sensor_artifact.get("sha256")
        != sha256_file(paths["sensor_rig_trajectory"])
    ):
        raise MP3DRoomEvaluationReviewError(
            "retained visual capture is not the MP3D 17DRP5sb8fy SensorRig capture"
        )
    try:
        rgb = np.load(paths["rgb"], allow_pickle=False)
        topdown = np.load(paths["topdown_rgb"], allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise MP3DRoomEvaluationReviewError(
            "visual RGB/Topdown arrays cannot be read"
        ) from exc
    if (
        rgb.dtype != np.uint8
        or topdown.dtype != np.uint8
        or rgb.ndim != 4
        or topdown.ndim != 4
        or rgb.shape[0] != FRAME_COUNT
        or topdown.shape[0] != FRAME_COUNT
        or rgb.shape[-1] != 3
        or topdown.shape[-1] != 3
        or visual.get("rgb_shape") != list(rgb.shape)
        or visual.get("topdown_shape") != list(topdown.shape)
        or visual.get("topdown_is_qa_only") is not True
    ):
        raise MP3DRoomEvaluationReviewError(
            "visual RGB/Topdown arrays differ from their retained summary"
        )
    return (
        np.ascontiguousarray(rgb),
        np.ascontiguousarray(topdown),
        {
            "status": "pass",
            "scene_id": EXPECTED_SCENE_ID,
            "retained_scene_path": scene_id,
            "room_id": EXPECTED_ROOM_REF["room_id"],
            "sensor_rig_content_matches_plan": True,
            "topdown_is_qa_only": True,
            "files": {
                name: _file_record(path) for name, path in paths.items()
            },
        },
    )


def _geometry_by_frame(
    *,
    source_positions_m: Mapping[str, np.ndarray],
    listener_positions_m: np.ndarray,
    listener_orientations_wxyz: np.ndarray,
    pose_hashes: tuple[str, ...],
) -> tuple[Mapping[str, Any], ...]:
    frames = []
    for frame_index in range(FRAME_COUNT):
        sources = {}
        for slot in SOURCE_SLOTS:
            geometry = listener_local_source_geometry(
                source_positions_m[slot][frame_index],
                listener_positions_m[frame_index],
                listener_orientations_wxyz[frame_index],
            )
            sources[slot] = {
                "position_m": source_positions_m[slot][frame_index].tolist(),
                **geometry,
            }
        frames.append(
            {
                "frame_index": frame_index,
                "pose_hash": pose_hashes[frame_index],
                "listener_position_m": listener_positions_m[
                    frame_index
                ].tolist(),
                "listener_orientation_wxyz": listener_orientations_wxyz[
                    frame_index
                ].tolist(),
                "sources": sources,
            }
        )
    return tuple(frames)


def load_review_inputs(
    *,
    audio_root: str | Path,
    visual_capture_root: str | Path,
    plan_root: str | Path,
) -> MP3DRoomEvaluationReviewInputs:
    """Load and fail-closed validate one formal MP3D review join."""

    audio = Path(audio_root).resolve(strict=True)
    visual = Path(visual_capture_root).resolve(strict=True)
    plan = Path(plan_root).resolve(strict=True)
    audio_delivery_path = audio / "delivery.json"
    audio_samples_path = audio / "samples.json"
    audio_delivery = load_json(audio_delivery_path)
    audio_samples = load_json(audio_samples_path)
    binding, binding_sha256 = _selection_binding(
        audio_delivery, audio_samples
    )
    sample, audio_path, sidecar_path = _one_audio_sample(
        audio,
        binding,
        binding_sha256,
        audio_delivery,
        audio_samples,
    )
    delivery_input_closure = audio_delivery.get("input_closure")
    samples_input_closure = audio_samples.get("input_closure")
    if (
        not isinstance(delivery_input_closure, Mapping)
        or delivery_input_closure != samples_input_closure
        or delivery_input_closure.get("acoustic_selection_binding") != binding
        or delivery_input_closure.get("acoustic_selection_binding_sha256")
        != binding_sha256
    ):
        raise MP3DRoomEvaluationReviewError(
            "audio input closure differs across delivery and samples"
        )
    (
        plan_sensor_rig,
        source_positions,
        listener_positions,
        listener_orientations,
        listener_yaws,
        pose_hashes,
        plan_evidence,
    ) = _plan_inputs(
        plan,
        episode_id=str(sample["episode_id"]),
        source_classes=sample["source_classes"],
        audio_input_closure=delivery_input_closure,
    )
    sensor_binding = m7_sensor_rig_binding(plan_sensor_rig)
    rgb, topdown, visual_evidence = _visual_inputs(
        visual,
        plan_sensor_rig=plan_sensor_rig,
        sensor_rig_binding=sensor_binding,
    )
    geometry = _geometry_by_frame(
        source_positions_m=source_positions,
        listener_positions_m=listener_positions,
        listener_orientations_wxyz=listener_orientations,
        pose_hashes=pose_hashes,
    )
    return MP3DRoomEvaluationReviewInputs(
        episode_id=str(sample["episode_id"]),
        sample_id=str(sample["sample_id"]),
        source_classes=dict(sample["source_classes"]),
        source_positions_m=source_positions,
        listener_positions_m=listener_positions,
        listener_orientations_wxyz=listener_orientations,
        listener_yaws_deg=listener_yaws,
        pose_hashes=pose_hashes,
        frame_geometry=geometry,
        rgb=rgb,
        topdown_rgb=topdown,
        audio_path=audio_path,
        acoustic_selection_binding=binding,
        acoustic_selection_binding_sha256=binding_sha256,
        sensor_rig_binding=sensor_binding,
        evidence_inputs={
            "audio": {
                "status": "pass",
                "delivery": _file_record(audio_delivery_path),
                "samples": _file_record(audio_samples_path),
                "mixture_wav": _file_record(audio_path),
                "mixture_wav_sidecar": _file_record(sidecar_path),
            },
            "plan": plan_evidence,
            "visual_capture": visual_evidence,
        },
        alignment={
            "schema": "avengine_m7_mp3d_review_cross_modal_alignment_v1",
            "status": "pass",
            "room_ref_matches_visual_scene": True,
            "audio_plan_sensor_rig_binding_matches": True,
            "visual_plan_sensor_rig_content_matches": True,
            "source_positions_from_plan_trajectory_bank": True,
            "source_positions_match_every_rir_plan_use": True,
            "doa_listener_pose_source": "plan SensorRigTrajectory same frame",
            "doa_source_position_source": (
                "plan trajectory_bank source_center_paths_m same frame"
            ),
            "checked_frame_count": FRAME_COUNT,
            "checked_source_frame_count": FRAME_COUNT * len(SOURCE_SLOTS),
        },
    )


def _validate_media_report(
    report: Mapping[str, Any], *, expected_path: Path
) -> dict[str, Any]:
    streams = report.get("ffprobe", {}).get("streams")
    video_streams = (
        [item for item in streams if item.get("codec_type") == "video"]
        if isinstance(streams, list)
        else []
    )
    audio_streams = (
        [item for item in streams if item.get("codec_type") == "audio"]
        if isinstance(streams, list)
        else []
    )
    if (
        report.get("audio_muxed") is not True
        or report.get("frame_count") != FRAME_COUNT
        or report.get("frame_rate_hz") != FRAME_RATE_HZ
        or len(video_streams) != 1
        or len(audio_streams) != 1
        or video_streams[0].get("codec_name") != "h264"
        or audio_streams[0].get("codec_name") != "aac"
        or int(audio_streams[0].get("sample_rate", 0))
        != AUDIO_SAMPLE_RATE_HZ
        or int(audio_streams[0].get("channels", 0)) != 2
        or not expected_path.is_file()
        or report.get("sha256") != sha256_file(expected_path)
    ):
        raise MP3DRoomEvaluationReviewError(
            "encoded review media readback differs"
        )
    result = dict(report)
    result["path"] = expected_path.name
    return result


def build_review(
    *,
    audio_root: str | Path,
    visual_capture_root: str | Path,
    plan_root: str | Path,
    output: str | Path,
) -> Path:
    """Validate, compose and encode one immutable review directory."""

    inputs = load_review_inputs(
        audio_root=audio_root,
        visual_capture_root=visual_capture_root,
        plan_root=plan_root,
    )
    destination = Path(output).expanduser().absolute()
    if os.path.lexists(destination):
        raise MP3DRoomEvaluationReviewError(
            f"refusing to replace review output: {destination}"
        )
    destination.mkdir(parents=True)
    try:
        colors = {
            "source1": (42, 210, 220),
            "source2": (250, 120, 70),
        }
        tracks = tuple(
            SourceOverlayTrack(
                source_id=slot,
                label=f"{slot}: {inputs.source_classes[slot]}",
                asset_class="generic point source",
                sound_class=inputs.source_classes[slot],
                color_rgb=colors[slot],
                positions_m=inputs.source_positions_m[slot],
                current_event_by_frame=("room_evaluation_audio",) * FRAME_COUNT,
                active_by_frame=(True,) * FRAME_COUNT,
                true_flags=("registry-bound", "plan-bound"),
            )
            for slot in SOURCE_SLOTS
        )
        diagnostics = tuple(
            (
                "geometry GT | "
                f"s1 az={frame['sources']['source1']['azimuth_deg']:+.1f}deg "
                f"d={frame['sources']['source1']['distance_m']:.2f}m | "
                f"s2 az={frame['sources']['source2']['azimuth_deg']:+.1f}deg "
                f"d={frame['sources']['source2']['distance_m']:.2f}m"
            )
            for frame in inputs.frame_geometry
        )
        frames = compose_annotated_frames(
            main_rgb=inputs.rgb,
            topdown_rgb=inputs.topdown_rgb,
            tracks=tracks,
            clip_id=inputs.episode_id,
            room_id=EXPECTED_ROOM_REF["room_id"],
            review_stage_label="M7 MP3D SoundSpaces2/RLR room evaluation",
            listener_position_m=inputs.listener_positions_m[0],
            listener_yaw_deg=float(inputs.listener_yaws_deg[0]),
            listener_positions_m_by_frame=inputs.listener_positions_m,
            listener_yaws_deg_by_frame=inputs.listener_yaws_deg,
            aggregate_true_flags=(
                "research-only",
                "registry-bound-acoustics",
                "same-frame-listener",
                "same-frame-source-positions",
            ),
            audio_diagnostic_by_frame=diagnostics,
            center_gate_pass=True,
            fps=FRAME_RATE_HZ,
        )
        video_path = destination / REVIEW_FILENAME
        media = _validate_media_report(
            encode_annotated_review(
                frames,
                video_path,
                fps=FRAME_RATE_HZ,
                audio_path=inputs.audio_path,
            ),
            expected_path=video_path,
        )
        evidence = {
            "schema": SCHEMA,
            "status": "pass",
            "research_only": True,
            "qualification_claim": False,
            "claim_boundary": (
                "Hash-bound MP3D/SoundSpaces2 room-evaluation listening review; "
                "geometry-derived DOA is ground truth, not model prediction; "
                "Topdown is QA-only; no room, material, episode, or dataset "
                "admission claim."
            ),
            "episode_id": inputs.episode_id,
            "sample_id": inputs.sample_id,
            "source_classes": dict(inputs.source_classes),
            "acoustic_selection_binding": dict(
                inputs.acoustic_selection_binding
            ),
            "acoustic_selection_binding_sha256": (
                inputs.acoustic_selection_binding_sha256
            ),
            "sensor_rig_trajectory": dict(inputs.sensor_rig_binding),
            "inputs": dict(inputs.evidence_inputs),
            "cross_modal_alignment": dict(inputs.alignment),
            "geometry_convention": {
                "coordinate_frame": (
                    "listener_x_right_y_up_negative_z_forward"
                ),
                "azimuth": "positive_right_degrees",
                "elevation": "positive_up_degrees",
                "distance": "euclidean_metres",
            },
            "frame_geometry": list(inputs.frame_geometry),
            "media": media,
            "producer": _file_record(Path(__file__).resolve()),
        }
        write_json(destination / "evidence.json", evidence)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--visual-capture-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = build_review(
        audio_root=args.audio_root,
        visual_capture_root=args.visual_capture_root,
        plan_root=args.plan_root,
        output=args.output,
    )
    print(f"MP3D_ROOM_EVALUATION_REVIEW_OK output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

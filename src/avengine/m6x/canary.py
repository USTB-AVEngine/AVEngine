"""End-to-end fixed SPEAR Apartment S0--S5 research canary.

The runner deliberately reuses the already exported Apartment package and the
existing M5/M5.1 visual/audio kernels.  It adds only the fixed-room source
logic required by M6.x: native obstacle qualification, named source programs,
360-degree binaural rendering, exact stems, Timeline/flags, and a diagnostic
Topdown that consumes the same obstacle snapshot as the center-point gate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from html import escape
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)
from avengine.m3.runtime import (
    CompiledAcousticScene,
    _verify_upload_report,
    load_compiled_acoustic_scene,
)
from avengine.m4.audio import read_float32_wav, write_float32_wav
from avengine.m4.runtime import M4SimulationConfig
from avengine.m5.acoustics import DynamicRIRSequence
from avengine.m5.timeline import json_schema_errors as m5_json_schema_errors
from avengine.m5.video import encode_h264_base_video, mux_binaural_wav
from avengine.m5_1.acoustics import (
    ResearchReviewKeyframeGrid,
    build_strided_review_keyframes,
    render_research_review_binaural_audio,
    render_research_review_binaural_rir_sequence,
    research_review_trajectory_record,
)
from avengine.m5_1.delivery import binaural_frame_diagnostics
from avengine.m5_1.dry_audio import DryAudioClipSpec, assemble_dry_audio_buses
from avengine.m5_1.mixed_capture import capture_human_beagle_paths
from avengine.m5_1.review import (
    SourceOverlayTrack,
    compose_annotated_frames,
    encode_annotated_review,
)
from avengine.m6.audio_program import (
    compile_audio_program,
    materialize_audio_program_variant,
    validate_audio_program,
)
from avengine.m6.flags import evaluate_legacy_flags
from avengine.m6.entities import validate_entity_asset_registry
from avengine.m6.sources import (
    validate_sound_asset_registry,
    validate_source_endpoint_registry,
)
from avengine.m6x.apartment import (
    FixedApartmentQualification,
    listener_orientation_wxyz,
    qualify_fixed_apartment,
)
from avengine.m6x.contracts import (
    validate_anchor_library,
    validate_room_capsule,
    validate_scenario_suite,
    validate_trajectory_template_set,
)
from avengine.m6x.topdown import render_runtime_topdown_frames
from avengine.m6x.trajectory import materialize_template_route


CANARY_SCHEMA = "avengine_m6x_fixed_apartment_canary_v1"
CAPTURE_SCHEMA = "avengine_m5_1_human_beagle_capture_v1"
CAPTURE_ANCHOR_ORDER = (
    "human0.head",
    "human0.mouth_emitter",
    "dog0.mouth_emitter",
)
FRAME_COUNT = 75
MASTER_FRAME_COUNT = 270
FPS = 15
SAMPLE_RATE_HZ = 16_000
TIME_BASE_HZ = 48_000
RIR_STRIDE_FRAMES = 3


class M6XCanaryError(RuntimeError):
    """The M6.x canary failed before a reviewable delivery was complete."""


@dataclass(frozen=True)
class CaptureData:
    root: Path
    rgb: np.ndarray
    semantic: np.ndarray
    actor_world_matrices: np.ndarray
    anchor_positions_m: np.ndarray
    records: tuple[Mapping[str, Any], ...]
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class M6XCanaryResult:
    output_dir: Path
    review_index: Path
    bundle_manifest: Path
    videos: tuple[Path, ...]


def _validate_capture_reuse_contract(
    evidence: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
) -> None:
    """Bind a retained capture to the current fixed-Apartment request."""

    room_manifest_path = Path(room_manifest_path).resolve()
    m1_request_path = Path(m1_request_path).resolve()
    room_manifest = load_json(room_manifest_path)
    request = load_json(m1_request_path)
    errors: list[str] = []

    if evidence.get("schema") != CAPTURE_SCHEMA:
        errors.append(f"schema must be {CAPTURE_SCHEMA!r}")
    if evidence.get("status") != "pass":
        errors.append("status must be 'pass'")
    if evidence.get("frame_count") != MASTER_FRAME_COUNT:
        errors.append(f"frame_count must be {MASTER_FRAME_COUNT}")
    if evidence.get("frame_rate_hz") != FPS:
        errors.append(f"frame_rate_hz must be {FPS}")
    if evidence.get("time_base_hz") != TIME_BASE_HZ:
        errors.append(f"time_base_hz must be {TIME_BASE_HZ}")
    if evidence.get("anchor_order") != list(CAPTURE_ANCHOR_ORDER):
        errors.append(f"anchor_order must be {list(CAPTURE_ANCHOR_ORDER)!r}")

    request_room_id = request.get("room_id")
    manifest_room_id = room_manifest.get("room_id")
    if not isinstance(request_room_id, str) or request_room_id != manifest_room_id:
        errors.append("current request room_id differs from its room manifest")

    inputs = evidence.get("inputs")
    if not isinstance(inputs, Mapping):
        errors.append("inputs must identify the retained capture inputs")
    else:
        for key, expected_path in (
            ("room_manifest", room_manifest_path),
            ("m1_request", m1_request_path),
        ):
            record = inputs.get(key)
            if not isinstance(record, Mapping):
                errors.append(f"inputs.{key} is missing")
            elif record.get("sha256") != sha256_file(expected_path):
                errors.append(f"inputs.{key} differs from the current Apartment input")

    rig = request.get("primary_camera_rig")
    if not isinstance(rig, Mapping):
        errors.append("current Apartment request has no primary_camera_rig")
    else:
        world_from_rig = rig.get("world_from_rig")
        calibration = rig.get("shared_calibration")
        if not isinstance(world_from_rig, Mapping) or not isinstance(
            calibration, Mapping
        ):
            errors.append("current Apartment camera contract is incomplete")
        else:
            expected_camera = {
                "position_m": world_from_rig.get("translation_m"),
                "rotation_xyzw": world_from_rig.get("rotation_xyzw"),
                "horizontal_fov_deg": calibration.get("hfov_degrees"),
                "legacy_camera_contract_required": True,
            }
            camera = evidence.get("camera")
            if not isinstance(camera, Mapping) or any(
                camera.get(key) != value for key, value in expected_camera.items()
            ):
                errors.append("camera differs from the current fixed-Apartment request")

    ticks_per_frame = TIME_BASE_HZ // FPS
    if len(records) != MASTER_FRAME_COUNT:
        errors.append(f"frame_readback must contain {MASTER_FRAME_COUNT} records")
    elif any(
        not isinstance(record, Mapping)
        or record.get("frame_index") != index
        or record.get("pts_ticks") != index * ticks_per_frame
        for index, record in enumerate(records)
    ):
        errors.append("frame_readback frame_index/pts_ticks do not match the time base")

    if errors:
        raise M6XCanaryError("capture reuse contract failed: " + "; ".join(errors))


def _load_capture(
    path: str | Path,
    *,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
) -> CaptureData:
    root = Path(path).resolve()
    required = {
        "rgb": root / "arrays/rgb.npy",
        "semantic": root / "arrays/semantic.npy",
        "actor": root / "arrays/actor_world_matrices.npy",
        "anchors": root / "arrays/anchor_positions_m.npy",
        "records": root / "frame_readback.json",
        "evidence": root / "evidence.json",
    }
    missing = [str(value) for value in required.values() if not value.is_file()]
    if missing:
        raise M6XCanaryError(f"capture is incomplete: {missing}")
    records = json.loads(required["records"].read_text(encoding="utf-8"))
    evidence = load_json(required["evidence"])
    if not isinstance(records, list):
        raise M6XCanaryError("capture frame_readback must be a JSON array")
    _validate_capture_reuse_contract(
        evidence,
        records,
        room_manifest_path=room_manifest_path,
        m1_request_path=m1_request_path,
    )
    rgb = np.load(required["rgb"], allow_pickle=False)
    semantic = np.load(required["semantic"], allow_pickle=False)
    actor = np.load(required["actor"], allow_pickle=False)
    anchors = np.load(required["anchors"], allow_pickle=False)
    if (
        rgb.shape != (MASTER_FRAME_COUNT, 240, 320, 3)
        or rgb.dtype != np.uint8
        or semantic.shape != (MASTER_FRAME_COUNT, 240, 320)
        or actor.shape != (MASTER_FRAME_COUNT, 2, 4, 4)
        or anchors.shape != (MASTER_FRAME_COUNT, len(CAPTURE_ANCHOR_ORDER), 3)
    ):
        raise M6XCanaryError(
            "capture arrays/evidence differ from the 270-frame contract"
        )
    return CaptureData(
        root=root,
        rgb=np.ascontiguousarray(rgb),
        semantic=np.ascontiguousarray(semantic),
        actor_world_matrices=np.ascontiguousarray(actor),
        anchor_positions_m=np.ascontiguousarray(anchors),
        records=tuple(records),
        evidence=evidence,
    )


def _capture_data(result: Any) -> CaptureData:
    return CaptureData(
        root=result.output_dir,
        rgb=result.rgb,
        semantic=result.semantic,
        actor_world_matrices=result.actor_world_matrices,
        anchor_positions_m=result.anchor_positions_m,
        records=result.records,
        evidence=result.evidence,
    )


def _validated_inputs(
    *,
    config_root: Path,
    room_registry_path: Path,
    entity_registry_path: Path,
    endpoint_registry_path: Path,
    sound_registry_path: Path,
) -> dict[str, Any]:
    values = {
        "room_capsule": load_json(config_root / "room_capsule.json"),
        "anchors": load_json(config_root / "anchor_library.json"),
        "trajectories": load_json(config_root / "trajectory_templates.json"),
        "suite": load_json(config_root / "scenario_suite.json"),
        "room_registry": load_json(room_registry_path),
        "entities": load_json(entity_registry_path),
        "endpoints": load_json(endpoint_registry_path),
        "sounds": load_json(sound_registry_path),
    }
    errors: list[str] = []
    errors.extend(
        f"room capsule: {item}"
        for item in validate_room_capsule(
            values["room_capsule"], room_registry=values["room_registry"]
        )
    )
    errors.extend(
        f"anchors: {item}"
        for item in validate_anchor_library(
            values["anchors"], room_capsule=values["room_capsule"]
        )
    )
    errors.extend(
        f"trajectories: {item}"
        for item in validate_trajectory_template_set(
            values["trajectories"],
            anchor_library=values["anchors"],
            room_capsule=values["room_capsule"],
        )
    )
    programs = [
        load_json(path)
        for path in sorted((config_root / "audio_programs").glob("*.json"))
    ]
    errors.extend(
        f"scenario suite: {item}"
        for item in validate_scenario_suite(
            values["suite"],
            room_capsule=values["room_capsule"],
            anchor_library=values["anchors"],
            trajectory_templates=values["trajectories"],
            audio_programs=programs,
        )
    )
    for owner, validator in (
        ("entity registry", validate_entity_asset_registry),
        ("endpoint registry", validate_source_endpoint_registry),
        ("sound registry", validate_sound_asset_registry),
    ):
        registry = {
            "entity registry": values["entities"],
            "endpoint registry": values["endpoints"],
            "sound registry": values["sounds"],
        }[owner]
        errors.extend(f"{owner}: {item}" for item in validator(registry))
    for program in programs:
        errors.extend(
            f"AudioProgram {program.get('program_id')}: {item}"
            for item in validate_audio_program(
                program,
                source_endpoint_registry=values["endpoints"],
                sound_asset_registry=values["sounds"],
            )
        )
    if errors:
        raise M6XCanaryError("; ".join(errors))
    values["programs"] = {
        (program["program_id"], program["revision"]): program for program in programs
    }
    return values


def _anchor_positions(anchor_library: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        item["anchor_id"]: np.asarray(item["position_m"], dtype=np.float64)
        for item in anchor_library["anchors"]
    }


def _master_root_paths(
    trajectory_templates: Mapping[str, Any], anchor_library: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    human = materialize_template_route(
        trajectory_templates,
        template_id="human_master_motion_270",
        route_id="human_master_route",
        anchor_library=anchor_library,
    )
    dog = materialize_template_route(
        trajectory_templates,
        template_id="dog_master_motion_270",
        route_id="dog_master_route",
        anchor_library=anchor_library,
    )
    if human.shape != (MASTER_FRAME_COUNT, 3) or dog.shape != (MASTER_FRAME_COUNT, 3):
        raise M6XCanaryError("master human/dog routes must contain exactly 270 points")
    return human, dog


def _static_path(point: Sequence[float]) -> np.ndarray:
    value = np.asarray(point, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise M6XCanaryError("static source position is invalid")
    return np.ascontiguousarray(np.repeat(value[None, :], MASTER_FRAME_COUNT, axis=0))


def _provisional_source_paths(
    anchor_library: Mapping[str, Any], human: np.ndarray, dog: np.ndarray
) -> dict[str, np.ndarray]:
    anchors = _anchor_positions(anchor_library)
    return {
        "m6x_dog0_muzzle": dog,
        "m6x_human0_mouth": human,
        "m6x_marker_front_speaker": _static_path(anchors["marker_front"]),
        "m6x_marker_rear_speaker": _static_path(anchors["marker_rear"]),
        "m6x_world_los_speaker": _static_path(anchors["world_los"]),
        "m6x_world_nlos_speaker": _static_path(anchors["world_nlos"]),
    }


def _actual_source_paths(
    anchor_library: Mapping[str, Any], capture: CaptureData
) -> dict[str, np.ndarray]:
    anchors = _anchor_positions(anchor_library)
    anchor_order = capture.evidence.get("anchor_order")
    if (
        not isinstance(anchor_order, list)
        or len(anchor_order) != len(set(anchor_order))
        or not all(isinstance(item, str) for item in anchor_order)
    ):
        raise M6XCanaryError("capture anchor_order must contain unique string names")
    anchor_indices = {name: index for index, name in enumerate(anchor_order)}
    required = ("human0.mouth_emitter", "dog0.mouth_emitter")
    missing = [name for name in required if name not in anchor_indices]
    if missing:
        raise M6XCanaryError(f"capture is missing emitter anchors: {missing}")
    result = {
        "m6x_dog0_muzzle": np.ascontiguousarray(
            capture.anchor_positions_m[:, anchor_indices["dog0.mouth_emitter"], :]
        ),
        "m6x_human0_mouth": np.ascontiguousarray(
            capture.anchor_positions_m[:, anchor_indices["human0.mouth_emitter"], :]
        ),
        "m6x_marker_front_speaker": _static_path(anchors["marker_front"]),
        "m6x_marker_rear_speaker": _static_path(anchors["marker_rear"]),
        "m6x_world_los_speaker": _static_path(anchors["world_los"]),
        "m6x_world_nlos_speaker": _static_path(anchors["world_nlos"]),
    }
    return dict(sorted(result.items()))


def _scenario_grid_and_sequence(
    master_grid: ResearchReviewKeyframeGrid,
    master_sequence: DynamicRIRSequence,
    *,
    source_paths: Mapping[str, np.ndarray],
    candidate_source_ids: Sequence[str],
    start_frame: int,
    end_frame_exclusive: int,
    listener_position_m: Sequence[float],
    listener_orientation: Sequence[float],
) -> tuple[ResearchReviewKeyframeGrid, DynamicRIRSequence, dict[str, np.ndarray]]:
    source_ids = tuple(candidate_source_ids)
    if source_ids != tuple(sorted(set(source_ids))):
        raise M6XCanaryError("scenario source IDs must use canonical order")
    if (
        end_frame_exclusive - start_frame != FRAME_COUNT
        or start_frame % RIR_STRIDE_FRAMES != 0
    ):
        raise M6XCanaryError("scenario windows must be aligned 75-frame slices")
    trajectories = {
        source_id: np.ascontiguousarray(
            source_paths[source_id][start_frame:end_frame_exclusive]
        )
        for source_id in source_ids
    }
    grid = build_strided_review_keyframes(
        trajectories,
        visual_frame_rate_hz=FPS,
        rir_stride_frames=RIR_STRIDE_FRAMES,
        listener_position_m=listener_position_m,
        listener_orientation_wxyz=listener_orientation,
    )
    retained_master_indices = [
        index
        for index, visual_index in enumerate(master_grid.visual_frame_indices)
        if start_frame <= visual_index < end_frame_exclusive
    ]
    rebased_visual = tuple(
        master_grid.visual_frame_indices[index] - start_frame
        for index in retained_master_indices
    )
    if rebased_visual != grid.visual_frame_indices:
        raise M6XCanaryError("master RIR grid cannot be sliced onto scenario frames")
    source_indices = [master_sequence.source_ids.index(item) for item in source_ids]
    samples = np.ascontiguousarray(
        master_sequence.samples[
            np.ix_(
                np.asarray(retained_master_indices, dtype=np.int64),
                np.asarray(source_indices, dtype=np.int64),
                np.arange(master_sequence.samples.shape[2]),
                np.arange(master_sequence.samples.shape[3]),
            )
        ]
    )
    lengths = np.ascontiguousarray(
        master_sequence.lengths[
            np.ix_(
                np.asarray(retained_master_indices, dtype=np.int64),
                np.asarray(source_indices, dtype=np.int64),
            )
        ]
    )
    trajectory = research_review_trajectory_record(grid)
    trajectory_hash = canonical_json_sha256(trajectory)
    metadata = {
        "schema": "avengine_m6x_sliced_dynamic_rir_v1",
        "source_ids": list(source_ids),
        "trajectory": trajectory,
        "trajectory_sha256": trajectory_hash,
        "parent_master_trajectory_sha256": master_sequence.trajectory_sha256,
        "parent_keyframe_indices": retained_master_indices,
        "layout_type": master_sequence.layout_type,
        "layout_id": master_sequence.layout_id,
        "channel_labels": list(master_sequence.channel_labels),
        "sample_rate_hz": master_sequence.sample_rate_hz,
    }
    acoustic_identity = master_sequence.metadata.get("m6x_acoustic_identity")
    if isinstance(acoustic_identity, Mapping):
        metadata["m6x_acoustic_identity"] = dict(acoustic_identity)
    sequence = DynamicRIRSequence(
        samples=samples,
        lengths=lengths,
        source_ids=source_ids,
        keyframe_ticks=tuple(frame.tick for frame in grid.keyframes),
        keyframe_samples=tuple(frame.sample_index for frame in grid.keyframes),
        sample_rate_hz=master_sequence.sample_rate_hz,
        layout_type=master_sequence.layout_type,
        layout_id=master_sequence.layout_id,
        channel_labels=master_sequence.channel_labels,
        trajectory_sha256=trajectory_hash,
        metadata=metadata,
    )
    return grid, sequence, trajectories


def _fixed_acoustic_identity(
    scene: CompiledAcousticScene,
    *,
    room_capsule: Mapping[str, Any],
    room_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check the selected M3 scene against the fixed RoomCapsule.

    This is a semantic room/package check, not a new release lock.  Exact
    geometry and material realization is checked separately against the RLR
    upload receipt already retained by the renderer.
    """

    room_reference = room_capsule["room_registry_ref"]
    matching_rooms = [
        record
        for record in room_registry.get("records", ())
        if isinstance(record, Mapping)
        and record.get("room_id") == room_reference["room_id"]
        and record.get("revision") == room_reference["room_revision"]
    ]
    if len(matching_rooms) != 1:
        raise M6XCanaryError(
            "fixed acoustic scene cannot resolve exactly one RoomCapsule room record"
        )
    room_record = matching_rooms[0]
    acoustic_reference = room_capsule["acoustic_package_ref"]
    matching_representations = [
        record
        for record in room_record.get("acoustic_representations", ())
        if isinstance(record, Mapping)
        and record.get("representation_id")
        == acoustic_reference["acoustic_representation_id"]
        and record.get("resource_id") == acoustic_reference["resource_id"]
    ]
    if len(matching_representations) != 1:
        raise M6XCanaryError(
            "fixed acoustic scene cannot resolve its RoomCapsule representation"
        )
    representation = matching_representations[0]
    manifest = scene.manifest
    observed_room_id = manifest.get("source_room", {}).get("room_id")
    observed_geometry = manifest.get("geometry", {}).get("representation")
    observed_materials = manifest.get("materials", {}).get("material_semantics")
    if observed_room_id != room_record["room_id"]:
        raise M6XCanaryError(
            "acoustic package source_room differs from the fixed RoomCapsule"
        )
    if observed_geometry != representation.get("geometry_kind"):
        raise M6XCanaryError(
            "acoustic package geometry differs from the RoomCapsule representation"
        )
    if observed_materials != room_capsule["acoustic_material_status"]:
        raise M6XCanaryError(
            "acoustic package material semantics differ from the RoomCapsule"
        )
    return {
        "room_capsule_id": room_capsule["room_capsule_id"],
        "room_capsule_revision": room_capsule["revision"],
        "room_id": room_record["room_id"],
        "room_revision": room_record["revision"],
        "acoustic_representation_id": representation["representation_id"],
        "acoustic_resource_id": representation["resource_id"],
        "validated_against_package_id": scene.package_id,
        "geometry_representation": observed_geometry,
        "material_semantics": observed_materials,
        "validation_basis": "semantic_room_binding_and_exact_native_upload_receipt",
    }


def _validated_acoustic_metadata(
    metadata: Mapping[str, Any],
    *,
    scene: CompiledAcousticScene,
    simulation: M4SimulationConfig,
    hrtf_file_path: str | Path,
    acoustic_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify retained RIRs came from the selected scene/configuration/HRTF."""

    upload_report = metadata.get("upload_report")
    if not isinstance(upload_report, Mapping):
        raise M6XCanaryError("RIR metadata lacks the native acoustic upload receipt")
    try:
        _verify_upload_report(scene, upload_report)
    except (KeyError, TypeError, ValueError) as exc:
        raise M6XCanaryError(
            f"RIR acoustic geometry/material receipt differs: {exc}"
        ) from exc

    expected_claim = {
        "package_mode": scene.manifest.get("package_mode"),
        "material_semantics": scene.manifest.get("materials", {}).get(
            "material_semantics"
        ),
        "material_qualification_claim": scene.manifest.get("materials", {}).get(
            "qualification_claim"
        ),
        "qa_status_by_report": {
            name: report.get("status")
            for name, report in sorted(scene.qa_reports.items())
        },
    }
    if metadata.get("scene_claim_boundary") != expected_claim:
        raise M6XCanaryError("RIR scene claim differs from the selected acoustic scene")

    expected_configuration = simulation.to_dict()
    expected_configuration.pop("channel_layout")
    expected_configuration.pop("speed_of_sound_m_s")
    observed_configuration = metadata.get("runtime", {}).get("configuration_readback")
    if observed_configuration != expected_configuration:
        raise M6XCanaryError("RIR runtime configuration differs from the M4 template")

    hrtf_path = Path(hrtf_file_path).resolve()
    if not hrtf_path.is_file():
        raise M6XCanaryError(f"binaural HRTF is missing: {hrtf_path}")
    if metadata.get("hrtf", {}).get("sha256") != sha256_file(hrtf_path):
        raise M6XCanaryError("RIR HRTF differs from the selected binaural decoder")

    result = dict(metadata)
    result["m6x_acoustic_identity"] = dict(acoustic_identity)
    return result


def _load_retained_master_sequence(
    path: str | Path,
    *,
    grid: ResearchReviewKeyframeGrid,
    scene: CompiledAcousticScene,
    simulation: M4SimulationConfig,
    hrtf_file_path: str | Path,
    acoustic_identity: Mapping[str, Any],
) -> DynamicRIRSequence:
    """Load a completed master RIR render for a matching trajectory grid."""

    root = Path(path).resolve()
    required = {
        "samples": root / "samples.npy",
        "lengths": root / "lengths.npy",
        "metadata": root / "metadata.json",
    }
    missing = [str(value) for value in required.values() if not value.is_file()]
    if missing:
        raise M6XCanaryError(f"retained acoustics are incomplete: {missing}")

    samples = np.load(required["samples"], allow_pickle=False)
    lengths = np.load(required["lengths"], allow_pickle=False)
    metadata = load_json(required["metadata"])
    source_ids = tuple(metadata.get("source_ids", ()))
    trajectory = research_review_trajectory_record(grid)
    if source_ids != grid.source_ids or metadata.get("trajectory") != trajectory:
        raise M6XCanaryError(
            "retained acoustics were rendered for a different source trajectory"
        )
    expected_prefix = (len(grid.keyframes), len(grid.source_ids), 2)
    if samples.ndim != 4 or samples.shape[:3] != expected_prefix:
        raise M6XCanaryError(
            f"retained RIR samples have invalid shape: {samples.shape}"
        )
    if lengths.shape != expected_prefix[:2]:
        raise M6XCanaryError(
            f"retained RIR lengths have invalid shape: {lengths.shape}"
        )
    if (
        not np.all(np.isfinite(samples))
        or np.any(lengths <= 0)
        or np.any(lengths > samples.shape[-1])
    ):
        raise M6XCanaryError("retained RIR arrays contain invalid values or lengths")
    if (
        metadata.get("sample_rate_hz") != SAMPLE_RATE_HZ
        or metadata.get("layout_type") != "binaural"
        or tuple(metadata.get("channel_labels", ())) != ("left", "right")
    ):
        raise M6XCanaryError("retained acoustics do not satisfy the binaural contract")
    metadata = _validated_acoustic_metadata(
        metadata,
        scene=scene,
        simulation=simulation,
        hrtf_file_path=hrtf_file_path,
        acoustic_identity=acoustic_identity,
    )

    return DynamicRIRSequence(
        samples=np.ascontiguousarray(samples),
        lengths=np.ascontiguousarray(lengths),
        source_ids=source_ids,
        keyframe_ticks=tuple(frame.tick for frame in grid.keyframes),
        keyframe_samples=tuple(frame.sample_index for frame in grid.keyframes),
        sample_rate_hz=SAMPLE_RATE_HZ,
        layout_type=str(metadata["layout_type"]),
        layout_id=str(metadata["layout_id"]),
        channel_labels=tuple(metadata["channel_labels"]),
        trajectory_sha256=str(metadata["trajectory_sha256"]),
        metadata=metadata,
    )


def _float32_stems_and_exact_mixture(
    stems: Mapping[str, Any], source_ids: Sequence[str]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Quantize persisted stems first, then sum those exact float32 values."""

    retained: dict[str, np.ndarray] = {}
    mixture = np.zeros((2, 80_000), dtype=np.float32)
    for source_id in source_ids:
        try:
            episode = np.asarray(stems[source_id].episode, dtype=np.float32)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise M6XCanaryError(f"invalid binaural stem for {source_id}") from exc
        if episode.shape != mixture.shape or not np.all(np.isfinite(episode)):
            raise M6XCanaryError(
                f"binaural stem for {source_id} must be finite [2,80000]"
            )
        retained[source_id] = np.ascontiguousarray(episode)
        np.add(mixture, episode, out=mixture)
    return retained, np.ascontiguousarray(mixture)


def _write_scenario_rir_evidence(
    scenario_root: Path,
    *,
    scenario_id: str,
    grid: ResearchReviewKeyframeGrid,
    sequence: DynamicRIRSequence,
) -> Path:
    """Retain the actual two-source scenario slice used by every variant."""

    root = scenario_root / "audio/rir_evidence"
    root.mkdir(parents=True)
    np.save(root / "samples.npy", sequence.samples, allow_pickle=False)
    np.save(root / "lengths.npy", sequence.lengths, allow_pickle=False)
    write_json(root / "trajectory.json", research_review_trajectory_record(grid))
    metadata = {
        "schema": "avengine_m6x_scenario_rir_evidence_v1",
        "status": "pass",
        "scenario_id": scenario_id,
        "source_ids": list(sequence.source_ids),
        "layout_type": sequence.layout_type,
        "layout_id": sequence.layout_id,
        "channel_labels": list(sequence.channel_labels),
        "sample_rate_hz": sequence.sample_rate_hz,
        "samples_shape": list(sequence.samples.shape),
        "lengths_shape": list(sequence.lengths.shape),
        "keyframe_count": len(sequence.keyframe_ticks),
        "sequence_metadata": dict(sequence.metadata),
    }
    write_json(root / "metadata.json", metadata)
    return root / "metadata.json"


def _entity_instances(
    program: Mapping[str, Any], endpoints: Mapping[str, Any]
) -> dict[str, Any]:
    endpoint_records = _endpoint_index(endpoints)
    source_records = []
    entities: dict[str, dict[str, Any]] = {}
    for source_id in program["candidate_source_endpoint_ids"]:
        endpoint = endpoint_records[source_id]
        binding = dict(endpoint["binding"])
        source_records.append(
            {
                "source_endpoint_id": source_id,
                "binding": binding,
                "source_visibility_mode": endpoint["source_visibility_mode"],
            }
        )
        if binding["kind"] == "entity_anchor":
            entity_id = binding["entity_instance_id"]
            entities[entity_id] = {
                "entity_instance_id": entity_id,
                "entity_asset_id": binding["entity_asset_id"],
                "entity_asset_revision": binding["entity_asset_revision"],
                "emitter_anchor_id": binding["emitter_anchor_id"],
            }
    return {
        "schema": "avengine_m6x_entity_instances_v1",
        "entity_instances": [entities[key] for key in sorted(entities)],
        "source_endpoints": source_records,
    }


def _event_window_labels(program: Mapping[str, Any]) -> tuple[str, ...]:
    if not program["events"]:
        return ("no active event (exact silent negative)",)
    return tuple(
        (
            f"{event['source_endpoint_id']}: "
            f"{event['start_sample'] / SAMPLE_RATE_HZ:.2f}–"
            f"{event['end_sample_exclusive'] / SAMPLE_RATE_HZ:.2f}s "
            f"({event['event_id']})"
        )
        for event in program["events"]
    )


def _spatial_state_labels(
    scenario: Mapping[str, Any], qualification: FixedApartmentQualification
) -> tuple[str, ...]:
    qualified = {
        item["anchor_id"]: item
        for item in qualification.anchor_qualification["records"]
    }
    labels = []
    for binding in scenario["source_bindings"]:
        source_id = binding["source_endpoint_id"]
        record = qualified[binding["spawn_anchor_id"]]
        labels.append(
            f"{source_id}: {record['observed_listener_relative_sector']}, "
            f"camera {record['observed_camera_fov']}, "
            f"acoustic {record['observed_acoustic_path']}"
        )
    return tuple(labels)


def _sound_index(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {item["sound_asset_id"]: item for item in registry["sound_assets"]}


def _endpoint_index(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {item["source_endpoint_id"]: item for item in registry["source_endpoints"]}


def _asset_bindings(
    sounds: Mapping[str, Any],
    *,
    repository_root: Path,
    beagle_audio_path: Path,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for sound_id, record in _sound_index(sounds).items():
        uri = str(record["dry_audio"]["uri"])
        if uri.startswith("repo://"):
            path = (repository_root / uri.removeprefix("repo://")).resolve()
        elif sound_id == "dog_beagle_v2_scheduled_dry":
            path = beagle_audio_path.resolve()
        else:
            continue
        if not path.is_file():
            raise M6XCanaryError(f"dry audio is missing for {sound_id}: {path}")
        expected = str(record["dry_audio"]["sha256"])
        if sha256_file(path) != expected:
            raise M6XCanaryError(f"dry audio differs from registry for {sound_id}")
        result[sound_id] = {"path": str(path), "sha256": expected}
    return result


def _dry_event_mappings(
    program: Mapping[str, Any], sounds: Mapping[str, Any]
) -> list[dict[str, Any]]:
    sound_records = _sound_index(sounds)
    result: list[dict[str, Any]] = []
    for event in program["events"]:
        dry = sound_records[event["sound_asset_id"]]["dry_audio"]
        result.append(
            {
                "event_id": event["event_id"],
                "source_id": event["source_endpoint_id"],
                "start_sample": event["start_sample"],
                "end_sample_exclusive": event["end_sample_exclusive"],
                "dry_asset_id": event["sound_asset_id"],
                "dry_asset_sha256": dry["sha256"],
                "dry_clip_start_sample": event["source_start_sample"],
                "dry_clip_end_sample_exclusive": event["source_end_sample_exclusive"],
                "linear_gain": event["linear_gain"],
                "fade_samples": event["fade_samples"],
            }
        )
    return result


def _write_audio(
    path: Path,
    samples: np.ndarray,
    *,
    role: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = write_float32_wav(
        path,
        samples,
        SAMPLE_RATE_HZ,
        channel_axis=0,
        metadata={"role": role, "qualification_claim": False, **dict(metadata)},
    )
    decoded = read_float32_wav(
        artifact.audio_path,
        sidecar_path=artifact.sidecar_path,
        verify_sidecar=True,
    )
    expected = np.asarray(samples, dtype="<f4")
    if decoded.samples.shape != expected.shape or not np.array_equal(
        decoded.samples, expected
    ):
        raise M6XCanaryError(f"float32 WAVE readback differs: {path}")
    return {
        "path": str(path),
        "sidecar_path": str(artifact.sidecar_path),
        "channel_count": decoded.channel_count,
        "sample_rate_hz": decoded.sample_rate_hz,
        "sample_count": decoded.frame_count,
        "peak_absolute": float(np.max(np.abs(decoded.samples))),
    }


def _program_frame_state(
    program: Mapping[str, Any], source_ids: Sequence[str]
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str | None, ...]]]:
    compiled = compile_audio_program(program)
    activity = {
        source_id: np.zeros(FRAME_COUNT, dtype=np.bool_) for source_id in source_ids
    }
    events = {source_id: [None] * FRAME_COUNT for source_id in source_ids}
    for frame_index in range(FRAME_COUNT):
        current = compiled.current_event_by_source(frame_index)
        for source_id in source_ids:
            event = current[source_id]
            activity[source_id][frame_index] = event is not None
            events[source_id][frame_index] = event
    return activity, {source_id: tuple(values) for source_id, values in events.items()}


def _semantic_centroids(semantic: np.ndarray, semantic_id: int) -> np.ndarray:
    result = np.full((semantic.shape[0], 2), np.nan, dtype=np.float64)
    for frame_index, frame in enumerate(semantic):
        y, x = np.nonzero(frame == semantic_id)
        if x.size:
            result[frame_index] = (float(np.mean(x)), float(np.mean(y)))
    return result


def _matrix_quaternion_xyzw(matrix: np.ndarray) -> list[float]:
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise M6XCanaryError("actor rotation matrix is invalid")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        values = np.asarray(
            [
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            values = np.asarray(
                [
                    0.25 * s,
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s,
                    (rotation[2, 1] - rotation[1, 2]) / s,
                ]
            )
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            values = np.asarray(
                [
                    (rotation[0, 1] + rotation[1, 0]) / s,
                    0.25 * s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                    (rotation[0, 2] - rotation[2, 0]) / s,
                ]
            )
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            values = np.asarray(
                [
                    (rotation[0, 2] + rotation[2, 0]) / s,
                    (rotation[1, 2] + rotation[2, 1]) / s,
                    0.25 * s,
                    (rotation[1, 0] - rotation[0, 1]) / s,
                ]
            )
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-12 or not math.isfinite(norm):
        raise M6XCanaryError("actor rotation quaternion is invalid")
    values /= norm
    if values[3] < 0.0:
        values *= -1.0
    return [0.0 if abs(float(value)) < 1.0e-15 else float(value) for value in values]


def _endpoint_actor_id(endpoint: Mapping[str, Any]) -> str:
    binding = endpoint["binding"]
    if binding["kind"] == "entity_anchor":
        return str(binding["entity_instance_id"])
    return str(binding["world_point_id"])


def _timeline(
    *,
    capture: CaptureData,
    window_start: int,
    program: Mapping[str, Any],
    trajectories: Mapping[str, np.ndarray],
    endpoints: Mapping[str, Any],
    sounds: Mapping[str, Any],
    listener_record: Mapping[str, Any],
) -> dict[str, Any]:
    endpoint_records = _endpoint_index(endpoints)
    sound_records = _sound_index(sounds)
    candidate_ids = tuple(program["candidate_source_endpoint_ids"])
    candidate_actor = {
        source_id: _endpoint_actor_id(endpoint_records[source_id])
        for source_id in candidate_ids
    }
    actor_ids = tuple(sorted({"dog0", "human0", *candidate_actor.values()}))
    asset_by_actor = {
        "dog0": (
            "rocketbox_dog_beagle_01_m2_v7_world_contact_candidate",
            "rocketbox_dog_beagle_01",
            "quadruped_canine",
        ),
        "human0": (
            "rocketbox_human_male_adult_01_m5_1_candidate",
            "rocketbox_human_male_adult_01",
            "biped_human",
        ),
        "marker_front": (
            "legacy_apartment_source_marker_front_v1",
            "rigid_source_marker",
            "rigid_object",
        ),
        "marker_rear": (
            "legacy_apartment_source_marker_rear_v1",
            "rigid_source_marker",
            "rigid_object",
        ),
        "spear_apartment_los_point": (
            "world_source_point",
            "world_source_point",
            "environmental_source",
        ),
        "spear_apartment_nlos_point": (
            "world_source_point",
            "world_source_point",
            "environmental_source",
        ),
    }
    actors = [
        {
            "actor_id": actor_id,
            "asset_id": asset_by_actor[actor_id][0],
            "template_id": asset_by_actor[actor_id][1],
            "body_plan_id": asset_by_actor[actor_id][2],
        }
        for actor_id in actor_ids
    ]
    emitter_hash = {
        source_id: canonical_json_sha256(trajectories[source_id].tolist())
        for source_id in candidate_ids
    }
    audio_events = []
    for event in program["events"]:
        endpoint = endpoint_records[event["source_endpoint_id"]]
        binding = endpoint["binding"]
        audio_events.append(
            {
                "event_id": event["event_id"],
                "actor_id": candidate_actor[event["source_endpoint_id"]],
                "event_type": (
                    "vocalization"
                    if sound_records[event["sound_asset_id"]]["semantic_sound_class"]
                    in {"animal_vocalization", "human_speech"}
                    else "other"
                ),
                "start_sample": event["start_sample"],
                "end_sample": event["end_sample_exclusive"],
                "emitter_bone": binding["emitter_anchor_id"],
                "emitter_path_sha256": emitter_hash[event["source_endpoint_id"]],
                "audio_asset_sha256": sound_records[event["sound_asset_id"]][
                    "dry_audio"
                ]["sha256"],
                "semantic_sync_required": True,
            }
        )
    camera_hash = canonical_json_sha256(dict(listener_record))
    events_by_actor: dict[str, list[Mapping[str, Any]]] = {
        actor_id: [] for actor_id in actor_ids
    }
    for event in audio_events:
        events_by_actor[event["actor_id"]].append(event)
    endpoint_by_actor = {
        actor_id: source_id for source_id, actor_id in candidate_actor.items()
    }
    frames = []
    for local_frame in range(FRAME_COUNT):
        master_frame = window_start + local_frame
        sample_start = (3200 * local_frame + 1) // 3
        sample_end = (3200 * (local_frame + 1) + 1) // 3
        states = []
        for actor_id in actor_ids:
            if actor_id == "human0":
                matrix = capture.actor_world_matrices[master_frame, 0]
                position = matrix[:3, 3]
                rotation = _matrix_quaternion_xyzw(matrix[:3, :3])
                action_id = "walk"
                action_phase = (master_frame % 16) / 16.0
                pose_hash = str(capture.records[master_frame]["human"]["pose_sha256"])
            elif actor_id == "dog0":
                matrix = capture.actor_world_matrices[master_frame, 1]
                position = matrix[:3, 3]
                rotation = _matrix_quaternion_xyzw(matrix[:3, :3])
                action_id = "walk"
                action_phase = (master_frame % 45) / 45.0
                pose_hash = str(
                    capture.records[master_frame]["beagle"]["readback"]["state_sha256"]
                )
            else:
                source_id = endpoint_by_actor[actor_id]
                position = trajectories[source_id][local_frame]
                rotation = [0.0, 0.0, 0.0, 1.0]
                action_id = "static"
                action_phase = 0.0
                pose_hash = canonical_json_sha256(
                    {"actor_id": actor_id, "position_m": position.tolist()}
                )
            vocalizing = any(
                event["start_sample"] <= sample_start < event["end_sample"]
                for event in events_by_actor[actor_id]
            )
            states.append(
                {
                    "actor_id": actor_id,
                    "root_transform": {
                        "translation_m": position.tolist(),
                        "rotation_xyzw": rotation,
                        "scale": [1.0, 1.0, 1.0],
                    },
                    "action_id": action_id,
                    "action_time_ticks": local_frame * 3200,
                    "action_phase": action_phase,
                    "pose_hash": pose_hash,
                    "contacts": {},
                    "mouth_state": {
                        "open_ratio": 0.0,
                        "vocalizing": vocalizing,
                    },
                }
            )
        frames.append(
            {
                "frame_index": local_frame,
                "pts_ticks": local_frame * 3200,
                "sample_start": sample_start,
                "sample_end": sample_end,
                "actor_states": states,
                "view_pose_hashes": {"view0": camera_hash},
            }
        )
    result = {
        "schema": "avengine_authoritative_timeline_v2",
        "time_base_hz": 48_000,
        "duration_ticks": 240_000,
        "video": {
            "fps_num": 15,
            "fps_den": 1,
            "frame_count": 75,
            "ticks_per_frame": 3200,
            "view_ids": ["view0"],
        },
        "audio": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sample_count": 80_000,
            "ticks_per_sample": 3,
            "channel_count": 2,
        },
        "actors": actors,
        "frames": frames,
        "audio_events": audio_events,
    }
    errors = m5_json_schema_errors(result, "avengine_authoritative_timeline_v2")
    if errors:
        raise M6XCanaryError("Timeline v2: " + "; ".join(errors))
    return result


def _visibility_facts(
    *,
    source_ids: Sequence[str],
    semantic: np.ndarray,
    endpoint_records: Mapping[str, Mapping[str, Any]],
    anchor_qualification: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    semantic_id = {
        "m6x_human0_mouth": 220,
        "m6x_dog0_muzzle": 221,
        "m6x_marker_front_speaker": 101,
        "m6x_marker_rear_speaker": 102,
    }
    anchor_by_endpoint = {
        "m6x_marker_front_speaker": "marker_front",
        "m6x_marker_rear_speaker": "marker_rear",
        "m6x_world_los_speaker": "world_los",
        "m6x_world_nlos_speaker": "world_nlos",
    }
    qualified = {
        item["anchor_id"]: item for item in anchor_qualification.get("records", [])
    }
    result: dict[str, dict[str, Any]] = {}
    for source_id in source_ids:
        if source_id in semantic_id:
            visible = [
                bool(np.any(frame == semantic_id[source_id])) for frame in semantic
            ]
            occlusion = ["clear" if value else None for value in visible]
        else:
            anchor = qualified[anchor_by_endpoint[source_id]]
            in_fov = anchor["observed_camera_fov"] == "in_fov"
            visible = [in_fov] * FRAME_COUNT
            path = anchor["observed_acoustic_path"]
            occlusion = ["clear" if path == "los" else "wall"] * FRAME_COUNT
        result[source_id] = {
            "in_fov_by_frame": visible,
            "occlusion_by_frame": occlusion,
        }
    return result


def _source_manifest(
    *,
    scenario: Mapping[str, Any],
    variant_id: str,
    program: Mapping[str, Any],
    trajectories: Mapping[str, np.ndarray],
    endpoints: Mapping[str, Any],
    sounds: Mapping[str, Any],
) -> dict[str, Any]:
    endpoint_records = _endpoint_index(endpoints)
    sound_records = _sound_index(sounds)
    compiled = compile_audio_program(program)
    return {
        "schema": "avengine_m6x_fixed_apartment_source_manifest_v1",
        "scenario_id": scenario["scenario_id"],
        "purpose": scenario["purpose"],
        "variant_id": variant_id,
        "room_policy": "fixed_scene_instance_no_furniture_mutation",
        "listener": {
            "listener_id": "listener0",
            "camera_listener_colocated": True,
            "camera_listener_cooriented": True,
            "audio_visibility_policy": "360_degree_no_camera_fov_cutoff",
        },
        "sources": [
            {
                "source_endpoint_id": source_id,
                "endpoint": endpoint_records[source_id],
                "trajectory": {
                    "position_authority": (
                        "captured_articulated_emitter_link_world_transform"
                        if source_id in {"m6x_human0_mouth", "m6x_dog0_muzzle"}
                        else "fixed_anchor_world_position"
                    ),
                    "frame_count": FRAME_COUNT,
                    "positions_m": trajectories[source_id].tolist(),
                },
                "activation": (
                    "active"
                    if source_id in compiled.active_source_endpoint_ids
                    else "persistent_silent"
                ),
            }
            for source_id in compiled.candidate_source_endpoint_ids
        ],
        "events": [
            {
                **dict(event),
                "sound_asset": sound_records[event["sound_asset_id"]],
            }
            for event in program["events"]
        ],
        "stem_policy": {
            "independent_binaural_stem_per_candidate_source": True,
            "mixture_is_exact_stem_sum": True,
            "normalization": False,
            "limiting": False,
        },
    }


def _track_class(source_id: str) -> tuple[str, str, tuple[int, int, int], str]:
    mapping = {
        "m6x_dog0_muzzle": (
            "Beagle",
            "animal_vocalization",
            (250, 120, 70),
            "dog",
        ),
        "m6x_human0_mouth": (
            "Human",
            "human_speech",
            (42, 210, 220),
            "human",
        ),
        "m6x_marker_front_speaker": (
            "Front marker",
            "test_signal",
            (255, 90, 70),
            "rigid_object",
        ),
        "m6x_marker_rear_speaker": (
            "Rear marker",
            "test_signal",
            (89, 156, 255),
            "rigid_object",
        ),
        "m6x_world_los_speaker": (
            "LOS source",
            "test_signal",
            (120, 220, 112),
            "world_point",
        ),
        "m6x_world_nlos_speaker": (
            "NLOS source",
            "test_signal",
            (167, 121, 255),
            "world_point",
        ),
    }
    return mapping[source_id]


def _scenario_tracks(
    *,
    source_ids: Sequence[str],
    trajectories: Mapping[str, np.ndarray],
    semantic: np.ndarray,
    activity: Mapping[str, np.ndarray],
    events: Mapping[str, tuple[str | None, ...]],
    gate: Mapping[str, Any],
    window_start: int,
) -> tuple[SourceOverlayTrack, ...]:
    semantic_ids = {
        "m6x_dog0_muzzle": 221,
        "m6x_human0_mouth": 220,
        "m6x_marker_front_speaker": 101,
        "m6x_marker_rear_speaker": 102,
    }
    tracks = []
    for source_id in source_ids:
        label, sound_class, color, asset_class = _track_class(source_id)
        source_gate = gate["sources"][source_id]
        nav_clearance = np.asarray(
            [
                item["navmesh_clearance_m"]
                for item in source_gate["frames"][
                    window_start : window_start + FRAME_COUNT
                ]
            ],
            dtype=np.float64,
        )
        main_marker = (
            None
            if source_id not in semantic_ids
            else _semantic_centroids(semantic, semantic_ids[source_id])
        )
        tracks.append(
            SourceOverlayTrack(
                source_id=source_id,
                label=label,
                asset_class=asset_class,
                sound_class=sound_class,
                color_rgb=color,
                positions_m=trajectories[source_id],
                current_event_by_frame=events[source_id],
                active_by_frame=tuple(bool(value) for value in activity[source_id]),
                true_flags=(),
                center_clearance_m=nav_clearance,
                main_marker_xy=main_marker,
            )
        )
    return tuple(tracks)


def _variant_name(scenario_id: str, variant_id: str) -> str:
    if scenario_id == "S1":
        return "A_front" if variant_id == "A" else "B_rear"
    if scenario_id == "S2" and variant_id == "silent_negative":
        return "silent_negative"
    return variant_id


def _render_variant(
    *,
    variant_root: Path,
    scenario: Mapping[str, Any],
    variant_id: str,
    program: Mapping[str, Any],
    capture: CaptureData,
    window_start: int,
    rgb: np.ndarray,
    semantic: np.ndarray,
    grid: ResearchReviewKeyframeGrid,
    sequence: DynamicRIRSequence,
    trajectories: Mapping[str, np.ndarray],
    qualification: FixedApartmentQualification,
    listener_position_m: Sequence[float],
    listener_yaw_deg: float,
    listener_orientation: Sequence[float],
    camera_hfov_degrees: float,
    endpoints: Mapping[str, Any],
    sounds: Mapping[str, Any],
    asset_bindings: Mapping[str, Any],
    rir_metadata_path: Path,
    rir_bundle_uri: str,
) -> dict[str, Any]:
    source_ids = tuple(program["candidate_source_endpoint_ids"])
    clip = DryAudioClipSpec.from_values(
        frame_count=FRAME_COUNT,
        fps_numerator=FPS,
        sample_rate_hz=SAMPLE_RATE_HZ,
    )
    dry = assemble_dry_audio_buses(
        _dry_event_mappings(program, sounds),
        source_ids=source_ids,
        clip=clip,
        asset_bindings=asset_bindings,
    )
    rendered_stems, _rendered_mixture = render_research_review_binaural_audio(
        dry.buses, sequence, grid=grid
    )
    stems, mixture = _float32_stems_and_exact_mixture(rendered_stems, source_ids)
    compiled = compile_audio_program(program)
    silent_variant = not compiled.active_source_endpoint_ids
    if silent_variant:
        if any(np.any(bus) for bus in dry.buses.values()) or np.any(mixture):
            raise M6XCanaryError("silent-negative audio is not exact zero")
    else:
        peak = float(np.max(np.abs(mixture)))
        if not 1.0e-9 < peak < 1.0:
            raise M6XCanaryError(
                f"active binaural mixture must be audible and unclipped; peak={peak}"
            )

    audio_records = {"dry": {}, "binaural_stems": {}}
    for source_id in source_ids:
        audio_records["dry"][source_id] = _write_audio(
            variant_root / "audio/dry" / f"{source_id}.wav",
            dry.buses[source_id][None, :],
            role="m6x_exact_dry_source_bus",
            metadata={
                "scenario_id": scenario["scenario_id"],
                "variant_id": variant_id,
                "source_endpoint_id": source_id,
            },
        )
        audio_records["binaural_stems"][source_id] = _write_audio(
            variant_root / "audio/binaural" / f"{source_id}_stem.wav",
            stems[source_id],
            role="m6x_dynamic_binaural_source_stem",
            metadata={
                "scenario_id": scenario["scenario_id"],
                "variant_id": variant_id,
                "source_endpoint_id": source_id,
                "trajectory_sha256": sequence.trajectory_sha256,
            },
        )
    mixture_path = variant_root / "audio/binaural/mixture.wav"
    audio_records["mixture"] = _write_audio(
        mixture_path,
        mixture,
        role="m6x_binaural_stem_sum_mixture",
        metadata={
            "scenario_id": scenario["scenario_id"],
            "variant_id": variant_id,
            "canonical_source_endpoint_order": list(source_ids),
            "normalization": False,
            "limiting": False,
        },
    )

    clean_base = variant_root / "videos/clean_video_only.mp4"
    clean_path = variant_root / "videos/clean_binaural.mp4"
    encode_h264_base_video(rgb, clean_base)
    mux_binaural_wav(clean_base, mixture_path, clean_path)
    clean_base.unlink()

    activity, event_state = _program_frame_state(program, source_ids)
    labels = {source_id: _track_class(source_id)[0] for source_id in source_ids}
    colors = {source_id: _track_class(source_id)[2] for source_id in source_ids}
    topdown = render_runtime_topdown_frames(
        qualification.obstacle_map,
        trajectories,
        listener_position_m=listener_position_m,
        listener_yaw_deg=listener_yaw_deg,
        camera_hfov_degrees=camera_hfov_degrees,
        source_activity_by_frame=activity,
        source_labels=labels,
        source_colors=colors,
    )
    visibility = _visibility_facts(
        source_ids=source_ids,
        semantic=semantic,
        endpoint_records=_endpoint_index(endpoints),
        anchor_qualification=qualification.anchor_qualification,
    )
    flags = evaluate_legacy_flags(
        observer_position_m=listener_position_m,
        observer_yaw_deg=listener_yaw_deg,
        fps=FPS,
        positions_by_source={
            source_id: positions.tolist()
            for source_id, positions in trajectories.items()
        },
        visibility_facts_by_source=visibility,
        evidence_uri="bundle://room/qualification.json",
    )
    diagnostic_labels, diagnostic_records = binaural_frame_diagnostics(mixture, clip)
    tracks = _scenario_tracks(
        source_ids=source_ids,
        trajectories=trajectories,
        semantic=semantic,
        activity=activity,
        events=event_state,
        gate=qualification.source_center_gate,
        window_start=window_start,
    )
    annotated = compose_annotated_frames(
        main_rgb=rgb,
        topdown_rgb=topdown,
        tracks=tracks,
        clip_id=f"{scenario['scenario_id']}_{variant_id}",
        room_id="spear_apartment_0000_habitat_fixed_v1",
        review_stage_label="M6.x",
        listener_position_m=listener_position_m,
        listener_yaw_deg=listener_yaw_deg,
        aggregate_true_flags=tuple(
            sorted(
                flag_id
                for flag_id, value in flags["clip_flags"].items()
                if value["status"] == "present"
            )
        ),
        audio_diagnostic_by_frame=diagnostic_labels,
        center_gate_pass=qualification.source_center_gate["status"] == "pass",
        fps=FPS,
    )
    diagnostic_path = variant_root / "videos/diagnostic_topdown_binaural.mp4"
    encode_annotated_review(
        annotated, diagnostic_path, fps=FPS, audio_path=mixture_path
    )

    source_manifest = _source_manifest(
        scenario=scenario,
        variant_id=variant_id,
        program=program,
        trajectories=trajectories,
        endpoints=endpoints,
        sounds=sounds,
    )
    source_manifest["rir_evidence"] = {
        "uri": rir_bundle_uri,
        "source_ids": list(sequence.source_ids),
        "pair_specific": True,
    }
    timeline = _timeline(
        capture=capture,
        window_start=window_start,
        program=program,
        trajectories=trajectories,
        endpoints=endpoints,
        sounds=sounds,
        listener_record={
            "position_m": list(listener_position_m),
            "orientation_wxyz": list(listener_orientation),
        },
    )
    metadata_root = variant_root / "metadata"
    write_json(
        metadata_root / "request.json",
        {
            "schema": "avengine_m6x_scenario_variant_request_v1",
            "room_capsule_id": "spear_apartment_0000_habitat_fixed_v1",
            "scenario": dict(scenario),
            "variant_id": variant_id,
            "audio_program_id": program["program_id"],
            "audio_program_revision": program["revision"],
        },
    )
    write_json(
        metadata_root / "entity_instances.json", _entity_instances(program, endpoints)
    )
    write_json(metadata_root / "audio_program.json", program)
    write_json(metadata_root / "timeline.json", timeline)
    write_json(metadata_root / "source_manifest.json", source_manifest)
    write_json(metadata_root / "flags.json", flags)
    write_json(
        metadata_root / "binaural_frame_diagnostics.json",
        {
            "schema": "avengine_m6x_binaural_frame_diagnostics_v1",
            "metric_boundary": "review-only ILD and broadband cross-correlation ITD",
            "records": list(diagnostic_records),
        },
    )
    write_json(
        metadata_root / "provenance.json",
        {
            "schema": "avengine_m6x_scenario_provenance_v1",
            "room_policy": "fixed_existing_scene_no_furniture_mutation",
            "visual_capture": "bundle://shared/master_capture",
            "rir_evidence": rir_bundle_uri,
            "runtime_backend": qualification.record["runtime_backend"],
            "placement_semantics": "source_center_only",
            "topdown_obstacle_authority": qualification.record["obstacle_authority"][
                "authority"
            ],
        },
    )
    status = {
        "schema": "avengine_m6x_variant_status_v1",
        "status": "pass",
        "scenario_id": scenario["scenario_id"],
        "variant_id": variant_id,
        "source_center_gate": "pass",
        "body_volume_checked": False,
        "audio_visibility_policy": "360_degree_no_camera_fov_cutoff",
        "silent_negative_exact_zero": silent_variant,
        "clean_video": str(clean_path.relative_to(variant_root)),
        "diagnostic_video": str(diagnostic_path.relative_to(variant_root)),
        "mixture_peak_absolute": float(np.max(np.abs(mixture))),
        "rir_evidence": rir_bundle_uri,
        "checks": {
            "visual": "pass",
            "obstacles": "pass",
            "acoustic": "pass",
            "timeline": "pass",
            "flags": "pass",
            "media_readback": "pass",
        },
    }
    write_json(metadata_root / "final_status.json", status)
    return {
        "scenario_id": scenario["scenario_id"],
        "purpose": scenario["purpose"],
        "variant_id": variant_id,
        "variant_name": variant_root.name,
        "status": "pass",
        "clean_video": clean_path,
        "diagnostic_video": diagnostic_path,
        "mixture": mixture_path,
        "stems": tuple(
            variant_root / "audio/binaural" / f"{source_id}_stem.wav"
            for source_id in source_ids
        ),
        "audio_records": audio_records,
        "source_ids": source_ids,
        "event_windows": _event_window_labels(program),
        "spatial_states": _spatial_state_labels(scenario, qualification),
        "checks": "visual / obstacles / acoustic / timeline / flags / media: pass",
        "rir_metadata": rir_metadata_path,
    }


def _write_review_index(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    listener_position_m: Sequence[float],
    listener_yaw_deg: float,
) -> Path:
    table_rows = []
    for row in rows:
        clean = Path(row["clean_video"]).relative_to(output).as_posix()
        diagnostic = Path(row["diagnostic_video"]).relative_to(output).as_posix()
        mixture = Path(row["mixture"]).relative_to(output).as_posix()
        stems = "<br>".join(
            f'<a href="{escape(Path(path).relative_to(output).as_posix())}">{escape(Path(path).name)}</a>'
            for path in row["stems"]
        )
        endpoints = "<br>".join(escape(str(value)) for value in row["source_ids"])
        events = "<br>".join(escape(str(value)) for value in row["event_windows"])
        spatial = "<br>".join(escape(str(value)) for value in row["spatial_states"])
        rir = Path(row["rir_metadata"]).relative_to(output).as_posix()
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['scenario_id']))}</td>"
            f"<td>{escape(str(row['purpose']))}</td>"
            f"<td>{escape(str(row['variant_name']))}</td>"
            f"<td>{endpoints}</td>"
            f"<td>{events}</td>"
            f"<td>{spatial}</td>"
            f"<td>{escape(str(row['status']))}</td>"
            f"<td>{escape(str(row['checks']))}</td>"
            f'<td><a href="{escape(clean)}">clean</a></td>'
            f'<td><a href="{escape(diagnostic)}">diagnostic + Topdown</a></td>'
            f'<td><a href="{escape(mixture)}">mixture.wav</a></td>'
            f"<td>{stems}</td>"
            f'<td><a href="{escape(rir)}">RIR evidence</a></td>'
            "</tr>"
        )
    listener_text = escape(str([float(value) for value in listener_position_m]))
    listener_yaw_text = escape(f"{listener_yaw_deg:g}")
    html = (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>AVEngine M6.x fixed Apartment review</title>
<style>
body{font-family:system-ui,sans-serif;margin:2rem;background:#f6f7f9;color:#20242a}
h1{margin-bottom:.35rem}.summary{background:white;padding:1rem;border-radius:.6rem;margin-bottom:1rem}
table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #ccd1d8;padding:.55rem;text-align:left;vertical-align:top}
th{background:#e9edf2}a{color:#075bb5}code{background:#eef1f5;padding:.1rem .25rem}
</style></head><body>
<h1>AVEngine M6.x — fixed SPEAR Apartment</h1>
<div class="summary">
Room: <code>spear_apartment_0000_habitat_fixed_v1</code><br>
Listener: one co-located/co-oriented camera + microphone rig at
<code>"""
        + listener_text
        + """</code>, yaw <code>"""
        + listener_yaw_text
        + """°</code><br>
Clip: 5 s / 75 frames / 15 fps / 16 kHz binaural<br>
Audio: 360° propagation; never cut off by camera FOV<br>
Placement gate: source center only; no body-volume claim<br>
Obstacle authority: live Habitat navmesh + every loaded rigid collision OBB
</div>
<table><thead><tr><th>Scenario</th><th>Purpose</th><th>Variant</th><th>Endpoints</th><th>Active event windows</th><th>Expected/observed spatial state</th><th>Status</th><th>Checks</th><th>Clean video</th><th>Diagnostic</th><th>Mixture</th><th>Independent stems</th><th>Scenario RIR</th></tr></thead>
<tbody>"""
        + "\n".join(table_rows)
        + """</tbody></table>
</body></html>
"""
    )
    path = output / "REVIEW_INDEX.html"
    path.write_text(html, encoding="utf-8", newline="\n")
    return path


def run_fixed_apartment_canary(
    *,
    config_root: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    room_registry_path: str | Path,
    entity_registry_path: str | Path,
    endpoint_registry_path: str | Path,
    sound_registry_path: str | Path,
    human_runtime_glb_path: str | Path,
    beagle_animal_manifest_path: str | Path,
    beagle_m2_request_path: str | Path,
    beagle_audio_path: str | Path,
    acoustic_package_manifest_path: str | Path,
    m4_request_path: str | Path,
    hrtf_file_path: str | Path,
    output_dir: str | Path,
    runtime_root: str | Path | None = None,
    capture_dir: str | Path | None = None,
    acoustics_dir: str | Path | None = None,
) -> M6XCanaryResult:
    """Build all fixed-room S0--S5 visual, acoustic, and metadata outputs."""

    repository_root = Path(__file__).resolve().parents[3]
    config_root = Path(config_root).resolve()
    room_manifest_path = Path(room_manifest_path).resolve()
    m1_request_path = Path(m1_request_path).resolve()
    output = Path(output_dir).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if os.path.lexists(output) or os.path.lexists(staging):
        raise M6XCanaryError(f"refusing to replace output or staging path: {output}")
    values = _validated_inputs(
        config_root=config_root,
        room_registry_path=Path(room_registry_path).resolve(),
        entity_registry_path=Path(entity_registry_path).resolve(),
        endpoint_registry_path=Path(endpoint_registry_path).resolve(),
        sound_registry_path=Path(sound_registry_path).resolve(),
    )
    human_root, dog_root = _master_root_paths(values["trajectories"], values["anchors"])
    provisional_paths = _provisional_source_paths(
        values["anchors"], human_root, dog_root
    )
    preflight = qualify_fixed_apartment(
        room_manifest_path=room_manifest_path,
        m1_request_path=m1_request_path,
        anchor_library=values["anchors"],
        source_center_trajectories_m=provisional_paths,
        runtime_root=runtime_root,
        minimum_navmesh_clearance_m=0.02,
    )
    if preflight.record["status"] != "pass":
        raise M6XCanaryError("fixed Apartment anchor/route preflight failed")

    staging.mkdir(parents=True)
    try:
        if capture_dir is None:
            capture_result = capture_human_beagle_paths(
                room_manifest_path=room_manifest_path,
                m1_request_path=m1_request_path,
                human_runtime_glb_path=human_runtime_glb_path,
                beagle_animal_manifest_path=beagle_animal_manifest_path,
                beagle_m2_request_path=beagle_m2_request_path,
                human_root_path_m=human_root,
                beagle_root_path_m=dog_root,
                output_dir=staging / "shared/master_capture",
                runtime_root=runtime_root,
                route_provenance={
                    "route_family": "m6x_fixed_apartment_master_270",
                    "source": "examples/m6x/fixed_apartment/trajectory_templates.json",
                    "placement_semantics": "source_center_only",
                },
                require_legacy_camera=True,
            )
            capture = _capture_data(capture_result)
            _validate_capture_reuse_contract(
                capture.evidence,
                capture.records,
                room_manifest_path=room_manifest_path,
                m1_request_path=m1_request_path,
            )
        else:
            source_capture = _load_capture(
                capture_dir,
                room_manifest_path=room_manifest_path,
                m1_request_path=m1_request_path,
            )
            shutil.copytree(source_capture.root, staging / "shared/master_capture")
            capture = _load_capture(
                staging / "shared/master_capture",
                room_manifest_path=room_manifest_path,
                m1_request_path=m1_request_path,
            )

        source_paths = _actual_source_paths(values["anchors"], capture)
        qualification = qualify_fixed_apartment(
            room_manifest_path=room_manifest_path,
            m1_request_path=m1_request_path,
            anchor_library=values["anchors"],
            source_center_trajectories_m=source_paths,
            runtime_root=runtime_root,
            minimum_navmesh_clearance_m=0.02,
        )
        if qualification.record["status"] != "pass":
            raise M6XCanaryError(
                "captured emitter source-center or anchor qualification failed"
            )
        expected_room_id = values["room_capsule"]["room_registry_ref"]["room_id"]
        if qualification.record.get("room_id") != expected_room_id:
            raise M6XCanaryError(
                "visual qualification room differs from the fixed RoomCapsule"
            )
        write_json(staging / "room/qualification.json", qualification.record)
        overview_paths = {
            source_id: path[:1] for source_id, path in source_paths.items()
        }
        overview = render_runtime_topdown_frames(
            qualification.obstacle_map,
            overview_paths,
            listener_position_m=qualification.record["listener"]["position_m"],
            listener_yaw_deg=qualification.record["listener"]["yaw_deg"],
            camera_hfov_degrees=qualification.record["listener"]["camera_hfov_degrees"],
            source_labels={
                source_id: _track_class(source_id)[0] for source_id in overview_paths
            },
        )[0]
        from PIL import Image

        Image.fromarray(overview, mode="RGB").save(
            staging / "room/runtime_obstacle_map.png"
        )

        listener_position = qualification.record["listener"]["position_m"]
        listener_yaw = float(qualification.record["listener"]["yaw_deg"])
        listener_orientation = listener_orientation_wxyz(listener_yaw)
        master_grid = build_strided_review_keyframes(
            source_paths,
            visual_frame_rate_hz=FPS,
            rir_stride_frames=RIR_STRIDE_FRAMES,
            listener_position_m=listener_position,
            listener_orientation_wxyz=listener_orientation,
        )
        scene = load_compiled_acoustic_scene(
            acoustic_package_manifest_path,
            allow_nonpassing_research_qa=True,
        )
        acoustic_identity = _fixed_acoustic_identity(
            scene,
            room_capsule=values["room_capsule"],
            room_registry=values["room_registry"],
        )
        m4_request = load_json(m4_request_path)
        simulation = M4SimulationConfig.from_mapping(m4_request["simulation"])
        resolved_hrtf = Path(hrtf_file_path).resolve()
        acoustics_root = staging / "shared/acoustics"
        if acoustics_dir is None:
            master_sequence = render_research_review_binaural_rir_sequence(
                scene,
                simulation,
                grid=master_grid,
                hrtf_file_path=str(resolved_hrtf),
            )
            master_sequence = replace(
                master_sequence,
                metadata=_validated_acoustic_metadata(
                    master_sequence.metadata,
                    scene=scene,
                    simulation=simulation,
                    hrtf_file_path=resolved_hrtf,
                    acoustic_identity=acoustic_identity,
                ),
            )
            acoustics_root.mkdir(parents=True)
            np.save(
                acoustics_root / "samples.npy",
                master_sequence.samples,
                allow_pickle=False,
            )
            np.save(
                acoustics_root / "lengths.npy",
                master_sequence.lengths,
                allow_pickle=False,
            )
            write_json(acoustics_root / "metadata.json", master_sequence.metadata)
            write_json(
                acoustics_root / "trajectory.json",
                research_review_trajectory_record(master_grid),
            )
        else:
            retained_root = Path(acoustics_dir).resolve()
            master_sequence = _load_retained_master_sequence(
                retained_root,
                grid=master_grid,
                scene=scene,
                simulation=simulation,
                hrtf_file_path=resolved_hrtf,
                acoustic_identity=acoustic_identity,
            )
            shutil.copytree(retained_root, acoustics_root)
            write_json(acoustics_root / "metadata.json", master_sequence.metadata)

        bindings = _asset_bindings(
            values["sounds"],
            repository_root=repository_root,
            beagle_audio_path=Path(beagle_audio_path),
        )
        rows: list[dict[str, Any]] = []
        for scenario in values["suite"]["scenarios"]:
            window = scenario["capture_frame_window"]
            start = int(window["start_frame"])
            end = int(window["end_frame_exclusive"])
            candidate_ids = tuple(
                item["source_endpoint_id"] for item in scenario["source_bindings"]
            )
            grid, sequence, trajectories = _scenario_grid_and_sequence(
                master_grid,
                master_sequence,
                source_paths=source_paths,
                candidate_source_ids=candidate_ids,
                start_frame=start,
                end_frame_exclusive=end,
                listener_position_m=listener_position,
                listener_orientation=listener_orientation,
            )
            rgb = np.ascontiguousarray(capture.rgb[start:end])
            semantic = np.ascontiguousarray(capture.semantic[start:end])
            reference = scenario["audio_program_ref"]
            base_program = values["programs"][
                (reference["program_id"], reference["revision"])
            ]
            render_programs: list[tuple[str, Mapping[str, Any]]] = []
            for variant_id in scenario["audio_variants"]:
                render_programs.append(
                    (
                        variant_id,
                        materialize_audio_program_variant(
                            base_program,
                            variant_id,
                            source_endpoint_registry=values["endpoints"],
                            sound_asset_registry=values["sounds"],
                        ),
                    )
                )
            if scenario["scenario_id"] == "S2":
                silent_ref = scenario["silent_negative_program_ref"]
                render_programs.append(
                    (
                        "silent_negative",
                        values["programs"][
                            (silent_ref["program_id"], silent_ref["revision"])
                        ],
                    )
                )
            scenario_name = f"{scenario['scenario_id']}_{scenario['purpose']}"
            scenario_root = staging / "scenarios" / scenario_name
            rir_metadata_path = _write_scenario_rir_evidence(
                scenario_root,
                scenario_id=scenario["scenario_id"],
                grid=grid,
                sequence=sequence,
            )
            rir_bundle_uri = (
                "bundle://" + rir_metadata_path.parent.relative_to(staging).as_posix()
            )
            for variant_id, program in render_programs:
                variant_root = (
                    scenario_root
                    / "variants"
                    / _variant_name(scenario["scenario_id"], variant_id)
                )
                rows.append(
                    _render_variant(
                        variant_root=variant_root,
                        scenario=scenario,
                        variant_id=variant_id,
                        program=program,
                        capture=capture,
                        window_start=start,
                        rgb=rgb,
                        semantic=semantic,
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
                        rir_bundle_uri=rir_bundle_uri,
                    )
                )

        index = _write_review_index(
            staging,
            rows,
            listener_position_m=listener_position,
            listener_yaw_deg=listener_yaw,
        )
        manifest = {
            "schema": CANARY_SCHEMA,
            "status": "pass",
            "room_capsule_id": values["room_capsule"]["room_capsule_id"],
            "scenario_suite_id": values["suite"]["scenario_suite_id"],
            "clip": {
                "frame_count": FRAME_COUNT,
                "frame_rate_hz": FPS,
                "duration_seconds": 5,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "audio_layout": "binaural_left_right",
            },
            "obstacle_policy": {
                "authority": (
                    "live_habitat_navmesh_plus_all_loaded_rigid_collision_obbs"
                ),
                "placement_semantics": "source_center_only",
                "body_volume_checked": False,
                "topdown_uses_same_runtime_snapshot": True,
            },
            "audio_visibility_policy": "360_degree_no_camera_fov_cutoff",
            "acoustic_identity": acoustic_identity,
            "variants": [
                {
                    "scenario_id": row["scenario_id"],
                    "purpose": row["purpose"],
                    "variant_id": row["variant_id"],
                    "status": row["status"],
                    "clean_video": Path(row["clean_video"])
                    .relative_to(staging)
                    .as_posix(),
                    "diagnostic_video": Path(row["diagnostic_video"])
                    .relative_to(staging)
                    .as_posix(),
                    "mixture": Path(row["mixture"]).relative_to(staging).as_posix(),
                    "stems": [
                        Path(path).relative_to(staging).as_posix()
                        for path in row["stems"]
                    ],
                    "rir_evidence": Path(row["rir_metadata"])
                    .parent.relative_to(staging)
                    .as_posix(),
                }
                for row in rows
            ],
            "review_index": index.relative_to(staging).as_posix(),
            "claim_boundary": (
                "research canary with placeholder acoustic materials; no room, "
                "material, asset, episode, or dataset admission claim"
            ),
        }
        bundle_path = staging / "bundle_manifest.json"
        write_json(bundle_path, manifest)
        os.replace(staging, output)
    except Exception:
        # Retain long-running native outputs for diagnosis/reuse instead of
        # silently deleting a completed visual capture or RIR render.
        raise

    videos = tuple(
        sorted(
            path
            for path in output.glob("scenarios/*/variants/*/videos/*.mp4")
            if path.is_file()
        )
    )
    return M6XCanaryResult(
        output_dir=output,
        review_index=output / "REVIEW_INDEX.html",
        bundle_manifest=output / "bundle_manifest.json",
        videos=videos,
    )


__all__ = [
    "CANARY_SCHEMA",
    "M6XCanaryError",
    "M6XCanaryResult",
    "run_fixed_apartment_canary",
]

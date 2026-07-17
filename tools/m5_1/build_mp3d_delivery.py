#!/usr/bin/env python3
"""Build the 18-second MP3D human/Beagle annotated binaural review."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.m4.audio import read_float32_wav, write_float32_wav
from avengine.m5_1.acoustics import (
    build_strided_review_keyframes,
    render_research_review_binaural_audio,
)
from avengine.m5_1.delivery import (
    M51DeliveryError,
    actual_emitter_trajectory_record,
    binaural_frame_diagnostics,
    declared_audio_asset_bindings,
    executable_event_mappings,
    load_retained_binaural_sequence,
    verify_audio_program_receipts,
)
from avengine.m5_1.dry_audio import DryAudioClipSpec, assemble_dry_audio_buses
from avengine.m5_1.mp3d_capture import (
    derive_mp3d_route_paths,
    load_mp3d_route_manifest,
)
from avengine.m5_1.mp3d_delivery import (
    MP3D_DELIVERY_SCHEMA,
    build_mp3d_overlay_tracks,
    listener_yaw_degrees,
    load_real_mp3d_navmesh_qa,
    render_mp3d_topdown_frames,
    source_program_reuse_record,
)
from avengine.m5_1.review import compose_annotated_frames, encode_annotated_review
from avengine.m5_1.source_contracts import load_source_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--acoustics-dir", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--route-manifest", required=True, type=Path)
    parser.add_argument("--m1-request", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--human-gain", type=float, default=0.18)
    parser.add_argument("--beagle-gain", type=float, default=0.18)
    parser.add_argument("--fade-samples", type=int, default=80)
    return parser.parse_args(argv)


def _absolute_file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _capture_array(
    capture: Path, capture_evidence: Mapping[str, Any], role: str
) -> np.ndarray:
    records = capture_evidence.get("array_artifacts")
    record = records.get(role) if isinstance(records, Mapping) else None
    relative = record.get("path") if isinstance(record, Mapping) else None
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise M51DeliveryError(f"capture array record is invalid: {role}")
    path = (capture / relative).resolve()
    try:
        path.relative_to(capture)
    except ValueError as exc:
        raise M51DeliveryError(f"capture array escapes bundle: {role}") from exc
    if (
        not path.is_file()
        or path.stat().st_size != record.get("byte_size")
        or sha256_file(path) != record.get("sha256")
    ):
        raise M51DeliveryError(f"capture array bytes changed: {role}")
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != record.get("shape") or array.dtype.str != record.get("dtype"):
        raise M51DeliveryError(f"capture array metadata changed: {role}")
    return np.ascontiguousarray(array)


def _audio_record(
    path: Path,
    samples: np.ndarray,
    *,
    metadata: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    artifact = write_float32_wav(
        path,
        samples,
        16_000,
        channel_axis=0,
        metadata=metadata,
    )
    decoded = read_float32_wav(
        artifact.audio_path,
        sidecar_path=artifact.sidecar_path,
        verify_sidecar=True,
    )
    if decoded.frame_count != 288_000 or not np.array_equal(
        decoded.samples, np.asarray(samples, dtype="<f4")
    ):
        raise M51DeliveryError("MP3D float32 WAVE differs on readback")
    return {
        "audio": file_record(artifact.audio_path, relative_to=root),
        "sidecar": file_record(artifact.sidecar_path, relative_to=root),
        "sample_rate_hz": decoded.sample_rate_hz,
        "sample_count": decoded.frame_count,
        "channel_count": decoded.channel_count,
        "peak_absolute": float(np.max(np.abs(decoded.samples))),
    }


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(path, size) if path.is_file() else ImageFont.load_default()


def _contact_sheet(frames: np.ndarray, path: Path) -> dict[str, Any]:
    indices = (0, 63, 75, 90, 120, 150, 170, 269)
    sheet = Image.new("RGB", (1280, 960), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for slot, frame_index in enumerate(indices):
        image = Image.fromarray(frames[frame_index], mode="RGB").resize((640, 240))
        x = (slot % 2) * 640
        y = (slot // 2) * 240
        sheet.paste(image, (x, y))
        draw.text(
            (x + 6, y + 218),
            f"frame {frame_index:03d}",
            fill=(255, 255, 255),
            font=_font(14),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=92, optimize=True)
    with Image.open(path) as readback:
        readback.load()
        if readback.mode != "RGB" or readback.size != (1280, 960):
            raise M51DeliveryError("MP3D delivery contact sheet readback differs")
    return file_record(path, relative_to=path.parents[1])


def _qa_view(request: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = [
        item
        for item in request.get("qa_views", [])
        if isinstance(item, Mapping) and item.get("kind") == "topdown"
    ]
    if len(candidates) != 1:
        raise M51DeliveryError("M1 request must declare one Topdown QA view")
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    capture = args.capture_dir.resolve()
    acoustics = args.acoustics_dir.resolve()
    source_path = args.source_manifest.resolve()
    route_path = args.route_manifest.resolve()
    request_path = args.m1_request.resolve()
    output = args.output_dir.resolve()
    staging = output.with_name(f".{output.name}.staging")
    if os.path.lexists(output) or os.path.lexists(staging):
        raise M51DeliveryError(f"refusing to replace delivery output: {output}")
    required = (
        capture / "evidence.json",
        capture / "mp3d_gate_evidence.json",
        acoustics / "evidence.json",
        source_path,
        route_path,
        request_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise M51DeliveryError(f"MP3D delivery input is missing: {missing}")

    capture_evidence = load_json(required[0])
    gate_evidence = load_json(required[1])
    acoustic_evidence = load_json(required[2])
    if (
        capture_evidence.get("status") != "pass"
        or capture_evidence.get("qualification_claim") is not False
        or capture_evidence.get("frame_count") != 270
        or capture_evidence.get("frame_rate_hz") != 15
    ):
        raise M51DeliveryError("MP3D capture evidence is not the bounded 18-second pass")
    if (
        gate_evidence.get("status") != "pass"
        or gate_evidence.get("qualification_claim") is not False
        or gate_evidence.get("gate_count") != 14
        or gate_evidence.get("passed_gate_count") != 14
    ):
        raise M51DeliveryError("MP3D gate evidence is not the expected 14/14 pass")
    acoustic_gate = acoustic_evidence.get("acoustic_package_gate")
    if (
        acoustic_evidence.get("status") != "pass"
        or acoustic_evidence.get("qualification_claim") is not False
        or not isinstance(acoustic_gate, Mapping)
        or acoustic_gate.get("package_mode") != "research_candidate"
        or acoustic_gate.get("material_semantics") != "research_placeholder"
        or acoustic_gate.get("qualification_claim")
        != "unqualified_research_placeholder"
    ):
        raise M51DeliveryError("MP3D RIR evidence is not an unqualified research pass")
    source = load_source_manifest(source_path)
    route = load_mp3d_route_manifest(route_path)
    request = load_json(request_path)
    if request.get("request_id") != route["request_id"] or request.get("room_id") != route["room_id"]:
        raise M51DeliveryError("M1 request identity differs from MP3D route")

    rgb = _capture_array(capture, capture_evidence, "rgb")
    semantic = _capture_array(capture, capture_evidence, "semantic")
    anchors = _capture_array(capture, capture_evidence, "anchor_positions_m")
    matrices = _capture_array(capture, capture_evidence, "actor_world_matrices")
    route_paths = derive_mp3d_route_paths(route)
    actor_centers = matrices[:, :, :3, 3]
    if not np.array_equal(actor_centers[:, 0], route_paths.human) or not np.array_equal(
        actor_centers[:, 1], route_paths.beagle
    ):
        raise M51DeliveryError("captured MP3D actor roots differ from route paths")

    clip_mapping = source["clip"]
    clip = DryAudioClipSpec.from_values(
        frame_count=int(clip_mapping["frame_count"]),
        fps_numerator=int(clip_mapping["fps_num"]),
        fps_denominator=int(clip_mapping["fps_den"]),
        sample_rate_hz=int(clip_mapping["sample_rate_hz"]),
    )
    if clip.frame_count != 270 or clip.sample_count != 288_000:
        raise M51DeliveryError("reused source program is not exact 18-second/16-kHz")
    rig_transform = request["primary_camera_rig"]["world_from_rig"]
    listener = np.asarray(rig_transform["translation_m"], dtype=np.float64)
    rotation_xyzw = np.asarray(rig_transform["rotation_xyzw"], dtype=np.float64)
    listener_yaw = listener_yaw_degrees(rotation_xyzw)
    x, y, z, w = rotation_xyzw / np.linalg.norm(rotation_xyzw)
    orientation_wxyz = (float(w), float(x), float(y), float(z))
    grid = build_strided_review_keyframes(
        {"source0": anchors[:, 1, :], "source1": anchors[:, 2, :]},
        visual_frame_rate_hz=15,
        rir_stride_frames=3,
        listener_position_m=listener,
        listener_orientation_wxyz=orientation_wxyz,
    )
    sequence = load_retained_binaural_sequence(acoustics, grid=grid)
    event_mappings = executable_event_mappings(
        source,
        gain_by_source={"source0": args.human_gain, "source1": args.beagle_gain},
        fade_samples=args.fade_samples,
    )
    dry = assemble_dry_audio_buses(
        event_mappings,
        source_ids=grid.source_ids,
        clip=clip,
        asset_bindings=declared_audio_asset_bindings(source),
    )
    verify_audio_program_receipts(source, dry.placement_receipts)
    stems, mixture = render_research_review_binaural_audio(dry.buses, sequence, grid=grid)
    mixture_peak = float(np.max(np.abs(mixture)))
    if not 0.0 < mixture_peak < 1.0:
        raise M51DeliveryError(
            f"MP3D binaural mixture must be audible and unclipped; peak={mixture_peak}"
        )

    qa_view = _qa_view(request)
    pathfinder_record = gate_evidence.get("pathfinder")
    if not isinstance(pathfinder_record, Mapping):
        raise M51DeliveryError("MP3D gate evidence lacks PathFinder record")
    navmesh_record = pathfinder_record.get("declared_navmesh")
    if not isinstance(navmesh_record, Mapping):
        raise M51DeliveryError("MP3D gate evidence lacks declared navmesh")
    navmesh = load_real_mp3d_navmesh_qa(
        navmesh_record=navmesh_record,
        actor_center_paths_m={"human0": route_paths.human, "dog0": route_paths.beagle},
        meters_per_pixel=float(qa_view["meters_per_pixel"]),
        height_m=float(qa_view["height_m"]),
        maximum_y_delta_m=float(route["pathfinder_gate"]["maximum_y_delta_m"]),
    )
    declared_bounds = np.asarray(pathfinder_record["bounds_m"], dtype=np.float64)
    if not np.allclose(np.asarray(navmesh.bounds_m), declared_bounds, atol=1.0e-6, rtol=0.0):
        raise M51DeliveryError("Topdown PathFinder bounds differ from gate evidence")
    shared_islands = np.unique(
        np.concatenate((navmesh.island_id["human0"], navmesh.island_id["dog0"]))
    )
    if shared_islands.tolist() != [int(pathfinder_record["shared_island_id"])]:
        raise M51DeliveryError("Topdown shared island differs from gate evidence")

    staging.mkdir(parents=True)
    try:
        audio_records: dict[str, Any] = {"dry": {}, "binaural_stems": {}}
        for source_id in grid.source_ids:
            audio_records["dry"][source_id] = _audio_record(
                staging / "audio/dry" / f"{source_id}.wav",
                dry.buses[source_id][None, :],
                metadata={
                    "role": "m5_1_mp3d_scheduled_dry_source_bus",
                    "source_id": source_id,
                    "assembly_content_sha256": dry.assembly_content_sha256,
                    "qualification_claim": False,
                },
                root=staging,
            )
            audio_records["binaural_stems"][source_id] = _audio_record(
                staging / "audio/binaural" / f"{source_id}_stem.wav",
                stems[source_id].episode,
                metadata={
                    "role": "m5_1_mp3d_dynamic_binaural_stem",
                    "source_id": source_id,
                    "trajectory_sha256": sequence.trajectory_sha256,
                    "partition_error": stems[source_id].maximum_partition_error,
                    "qualification_claim": False,
                },
                root=staging,
            )
        mixture_path = staging / "audio/binaural/mixture.wav"
        audio_records["binaural_mixture"] = _audio_record(
            mixture_path,
            mixture,
            metadata={
                "role": "m5_1_mp3d_dynamic_binaural_mixture",
                "canonical_source_order": list(grid.source_ids),
                "trajectory_sha256": sequence.trajectory_sha256,
                "qualification_claim": False,
                "normalization": False,
                "limiting": False,
            },
            root=staging,
        )
        actual_trajectory = actual_emitter_trajectory_record(
            anchors,
            capture_evidence_sha256=sha256_file(capture / "evidence.json"),
        )
        reuse_record = source_program_reuse_record(source)
        write_json(staging / "actual_emitter_trajectories.json", actual_trajectory)
        write_json(staging / "source_program_reuse.json", reuse_record)
        write_json(staging / "dry_audio_assembly.json", dry.metadata())
        np.save(staging / "navmesh_binary_map.npy", navmesh.binary_map, allow_pickle=False)
        diagnostic_labels, diagnostic_records = binaural_frame_diagnostics(mixture, clip)
        write_json(
            staging / "binaural_frame_diagnostics.json",
            {
                "schema": "avengine_m5_1_binaural_frame_diagnostics_v1",
                "metric_boundary": (
                    "Review-only frame-local ILD and broadband cross-correlation ITD; "
                    "not an HRTF qualification metric"
                ),
                "records": list(diagnostic_records),
            },
        )
        write_json(
            staging / "navmesh_center_diagnostics.json",
            {
                "schema": "avengine_m5_1_mp3d_navmesh_center_diagnostics_v1",
                "center_navigation_semantics": "actor_root_center_only",
                "full_articulated_mesh_clearance_claim": False,
                "navmesh": navmesh.navmesh_record,
                "bounds_m": [list(value) for value in navmesh.bounds_m],
                "shared_island_id": int(shared_islands[0]),
                "actors": {
                    actor_id: {
                        "all_frames_navigable": bool(np.all(navmesh.navigable[actor_id])),
                        "minimum_navmesh_edge_clearance_m": float(
                            np.min(navmesh.clearance_m[actor_id])
                        ),
                        "maximum_navmesh_edge_clearance_m": float(
                            np.max(navmesh.clearance_m[actor_id])
                        ),
                        "per_frame_clearance_m": navmesh.clearance_m[actor_id].tolist(),
                    }
                    for actor_id in ("human0", "dog0")
                },
            },
        )
        tracks = build_mp3d_overlay_tracks(
            source,
            anchor_positions_m=anchors,
            semantic_frames=semantic,
            clearance_m=navmesh.clearance_m,
            gate_evidence=gate_evidence,
        )
        topdown = render_mp3d_topdown_frames(
            navmesh_binary_map=navmesh.binary_map,
            navmesh_bounds_m=navmesh.bounds_m,
            actor_center_paths_m={"human0": route_paths.human, "dog0": route_paths.beagle},
            listener_position_m=listener,
            listener_yaw_deg=listener_yaw,
            clearance_m=navmesh.clearance_m,
            shared_island_id=int(shared_islands[0]),
        )
        annotated = compose_annotated_frames(
            main_rgb=rgb,
            topdown_rgb=topdown,
            tracks=tracks,
            clip_id="mp3d_human_beagle_source_program_clip",
            room_id=str(route["room_id"]),
            listener_position_m=listener,
            listener_yaw_deg=listener_yaw,
            aggregate_true_flags=(
                "centers_navigable",
                "shared_navmesh_island",
                "visible_all_frames",
                "simultaneous_events",
            ),
            audio_diagnostic_by_frame=diagnostic_labels,
            center_gate_pass=True,
            fps=15,
        )
        contact_sheet = _contact_sheet(
            annotated, staging / "qa/annotated_contact_sheet.jpg"
        )
        review_path = staging / "videos/mp3d_human_beagle_annotated_binaural.mp4"
        review_report = dict(
            encode_annotated_review(
                annotated,
                review_path,
                fps=15,
                audio_path=mixture_path,
            )
        )
        review_report["path"] = str(review_path.relative_to(staging))
        center_separation = np.linalg.norm(route_paths.human - route_paths.beagle, axis=1)
        evidence: dict[str, Any] = {
            "schema": MP3D_DELIVERY_SCHEMA,
            "status": "pass",
            "research_only": True,
            "qualification_claim": False,
            "claim_boundary": (
                "M5.1 MP3D annotated research review. Pathfinder gates actor root "
                "centers only; unqualified research-placeholder acoustics and room "
                "materials; no full-mesh, asset, room, material, episode, or dataset "
                "admission claim."
            ),
            "inputs": {
                "capture_evidence": _absolute_file_record(capture / "evidence.json"),
                "mp3d_gate_evidence": _absolute_file_record(
                    capture / "mp3d_gate_evidence.json"
                ),
                "acoustic_evidence": _absolute_file_record(acoustics / "evidence.json"),
                "source_manifest": _absolute_file_record(source_path),
                "route_manifest": _absolute_file_record(route_path),
                "m1_request": _absolute_file_record(request_path),
            },
            "timeline": {
                "frame_count": 270,
                "frame_rate_hz": 15,
                "duration_seconds": 18,
                "audio_sample_rate_hz": 16_000,
                "audio_sample_count": 288_000,
            },
            "source_ids": list(grid.source_ids),
            "source_contract_reuse": file_record(
                staging / "source_program_reuse.json", relative_to=staging
            ),
            "legacy_spatial_flags_applied": False,
            "simultaneous_event_overlaps": reuse_record["event_overlap_windows"],
            "center_navigation_gate": {
                "status": "pass",
                "semantics": "actor_root_center_only",
                "full_articulated_mesh_clearance_claim": False,
                "shared_island_id": int(shared_islands[0]),
                "minimum_human_clearance_to_navmesh_edge_m": float(
                    np.min(navmesh.clearance_m["human0"])
                ),
                "minimum_dog_clearance_to_navmesh_edge_m": float(
                    np.min(navmesh.clearance_m["dog0"])
                ),
                "minimum_human_dog_center_distance_m": float(
                    np.min(center_separation)
                ),
                "diagnostics": file_record(
                    staging / "navmesh_center_diagnostics.json", relative_to=staging
                ),
                "raw_navmesh_binary_map": file_record(
                    staging / "navmesh_binary_map.npy", relative_to=staging
                ),
            },
            "actual_emitter_trajectories": file_record(
                staging / "actual_emitter_trajectories.json", relative_to=staging
            ),
            "dry_audio_assembly": file_record(
                staging / "dry_audio_assembly.json", relative_to=staging
            ),
            "dynamic_binaural": {
                "acoustic_package_gate": acoustic_gate,
                "trajectory_sha256": sequence.trajectory_sha256,
                "rir_keyframe_count": len(sequence.keyframe_samples),
                "rir_rate_hz": 5,
                "layout_id": sequence.layout_id,
                "channel_labels": list(sequence.channel_labels),
                "mixture_peak_absolute": mixture_peak,
                "normalization": False,
                "limiting": False,
                "audio_records": audio_records,
                "frame_diagnostics": file_record(
                    staging / "binaural_frame_diagnostics.json", relative_to=staging
                ),
            },
            "review_media": {
                "annotated_mp3d": review_report,
                "contact_sheet": contact_sheet,
                "topdown_is_qa_only": True,
                "topdown_authority": "Habitat PathFinder.get_topdown_view",
            },
        }
        evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
        write_json(staging / "evidence.json", evidence)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(output / "evidence.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

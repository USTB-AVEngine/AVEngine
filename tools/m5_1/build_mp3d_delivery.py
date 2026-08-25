#!/usr/bin/env python3
"""Build an 18-second Habitat-native human/Beagle annotated binaural review."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.spatial_audio.audio import read_float32_wav, write_float32_wav
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
    source_actor_binding_record,
    source_binding_entries,
    verify_source_event_audio_activity,
    verify_audio_program_receipts,
)
from avengine.m5_1.dry_audio import DryAudioClipSpec, assemble_dry_audio_buses
from avengine.m5_1.mp3d_capture import (
    derive_mp3d_route_paths,
    load_mp3d_route_manifest,
)
from avengine.m5_1.replicacad_capture import (
    derive_replicacad_route_paths,
    load_replicacad_route_manifest,
)
from avengine.m5_1.mp3d_delivery import (
    MP3D_DELIVERY_SCHEMA,
    build_mp3d_overlay_tracks,
    listener_yaw_degrees,
    load_real_mp3d_navmesh_qa,
    render_mp3d_topdown_frames,
    source_program_reuse_record,
    validate_room_visual_gate,
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
    parser.add_argument(
        "--room-family",
        choices=("mp3d", "replicacad"),
        default="mp3d",
        help="Select the room-specific route/gate adapter (default: mp3d)",
    )
    parser.add_argument(
        "--replicacad-root",
        type=Path,
        help="Required with --room-family replicacad; resolves portable navmesh records",
    )
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


def _repository_file_record(path: Path) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repository)
    except ValueError as exc:
        raise M51DeliveryError(
            f"ReplicaCAD delivery input escapes AVENGINE_REPOSITORY_ROOT: {resolved}"
        ) from exc
    return {
        "root_id": "AVENGINE_REPOSITORY_ROOT",
        "relative_path": relative.as_posix(),
        "byte_size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _runtime_navmesh_record(
    record: Mapping[str, Any],
    *,
    room_family: str,
    replicacad_root: Path | None,
) -> dict[str, Any]:
    if room_family == "mp3d":
        return dict(record)
    if replicacad_root is None:
        raise M51DeliveryError(
            "--replicacad-root is required for a ReplicaCAD delivery"
        )
    if record.get("root_id") != "AVENGINE_REPLICACAD_ROOT":
        raise M51DeliveryError("ReplicaCAD navmesh record has the wrong root_id")
    relative = record.get("relative_path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise M51DeliveryError("ReplicaCAD navmesh relative_path is invalid")
    root = replicacad_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise M51DeliveryError("ReplicaCAD navmesh escapes its external root") from exc
    return {
        "path": str(path),
        "byte_size": record.get("byte_size"),
        "sha256": record.get("sha256"),
    }


def _clearance_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "query_count": int(array.size),
        "minimum_navmesh_edge_clearance_m": float(np.min(array)),
        "maximum_navmesh_edge_clearance_m": float(np.max(array)),
        "failed_frame_indices": [],
        "clearance_sequence_sha256": canonical_json_sha256(array.tolist()),
    }


def _portable_audio_locator(raw: str) -> dict[str, str]:
    parsed = urlparse(raw)
    if parsed.scheme not in ("", "file"):
        raise M51DeliveryError(f"unsupported local audio locator scheme: {parsed.scheme}")
    path = Path(unquote(parsed.path) if parsed.scheme == "file" else raw).resolve()
    parts = path.parts
    if "LibriTTS" in parts:
        index = parts.index("LibriTTS")
        relative = Path(*parts[index + 1 :]).as_posix()
        return {"root_id": "AVENGINE_LIBRITTS_ROOT", "relative_path": relative}
    if "AVEngine" in parts:
        index = parts.index("AVEngine")
        relative = Path(*parts[index + 1 :]).as_posix()
        return {"root_id": "AVENGINE_LEGACY_ROOT", "relative_path": relative}
    raise M51DeliveryError(
        f"local audio path lacks a declared portable root class: {path.name}"
    )


def _portable_source_program_audio(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    sources: list[dict[str, Any]] = []
    for source in result.get("sources", []):
        copied = dict(source)
        provenance = dict(copied.get("audio_provenance", {}))
        assets: list[dict[str, Any]] = []
        for asset in provenance.get("audio_assets", []):
            item = dict(asset)
            uri = item.pop("uri", None)
            if not isinstance(uri, str):
                raise M51DeliveryError("source-program audio asset lacks a URI")
            item["locator"] = _portable_audio_locator(uri)
            assets.append(item)
        provenance["audio_assets"] = assets
        copied["audio_provenance"] = provenance
        sources.append(copied)
    result["sources"] = sources
    result["portable_audio_roots"] = {
        "AVENGINE_LIBRITTS_ROOT": "external licensed speech corpus root",
        "AVENGINE_LEGACY_ROOT": "external legacy AVEngine workspace root",
    }
    result.pop("record_content_sha256", None)
    result["record_content_sha256"] = canonical_json_sha256(result)
    return result


def _portable_dry_audio_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    receipts: list[dict[str, Any]] = []
    for receipt in result.get("placement_receipts", []):
        copied = dict(receipt)
        asset = dict(copied.get("dry_asset", {}))
        raw_path = asset.pop("path", None)
        if not isinstance(raw_path, str):
            raise M51DeliveryError("dry-audio receipt lacks its local source path")
        asset["locator"] = _portable_audio_locator(raw_path)
        copied["dry_asset"] = asset
        receipts.append(copied)
    result["placement_receipts"] = receipts
    result["portable_audio_roots"] = {
        "AVENGINE_LIBRITTS_ROOT": "external licensed speech corpus root",
        "AVENGINE_LEGACY_ROOT": "external legacy AVEngine workspace root",
    }
    result.pop("assembly_content_sha256", None)
    result["assembly_content_sha256"] = canonical_json_sha256(result)
    return result


def _verified_evidence_content_sha256(
    document: Mapping[str, Any], *, owner: str
) -> str:
    content = dict(document)
    declared = content.pop("evidence_content_sha256", None)
    if not isinstance(declared, str) or canonical_json_sha256(content) != declared:
        raise M51DeliveryError(f"{owner} evidence content identity differs")
    return declared


def _record_sha256(record: Any, *, owner: str) -> str:
    value = record.get("sha256") if isinstance(record, Mapping) else None
    if not isinstance(value, str):
        raise M51DeliveryError(f"{owner} lacks a file-content identity")
    return value


def _validate_capture_route_request_identity(
    capture_evidence: Mapping[str, Any],
    *,
    capture_evidence_sha256: str,
    route: Mapping[str, Any],
    route_sha256: str,
    request: Mapping[str, Any],
    request_sha256: str,
    room_family: str,
) -> str:
    capture_content_sha256 = _verified_evidence_content_sha256(
        capture_evidence, owner=f"{room_family} capture"
    )
    inputs = capture_evidence.get("inputs")
    route_provenance = (
        inputs.get("route_provenance") if isinstance(inputs, Mapping) else None
    )
    embedded_route = (
        route_provenance.get("route_manifest")
        if isinstance(route_provenance, Mapping)
        else None
    )
    embedded_request = inputs.get("m1_request") if isinstance(inputs, Mapping) else None
    if _record_sha256(embedded_route, owner="capture route") != route_sha256:
        raise M51DeliveryError("capture route-manifest bytes differ from delivery input")
    if _record_sha256(embedded_request, owner="capture request") != request_sha256:
        raise M51DeliveryError("capture M1-request bytes differ from delivery input")
    if (
        not isinstance(route_provenance, Mapping)
        or route_provenance.get("route_id") != route.get("route_id")
        or request.get("request_id") != route.get("request_id")
        or request.get("room_id") != route.get("room_id")
    ):
        raise M51DeliveryError("capture/route/request logical identity differs")
    for field in ("route_id", "request_id", "room_id"):
        captured = capture_evidence.get(field)
        placement = route.get("placement_gate")
        legacy_identity_adapter = room_family == "mp3d" or (
            room_family == "replicacad"
            and (
                not isinstance(placement, Mapping)
                or placement.get("require_rigid_object_center_clearance") is not True
            )
        )
        if captured is None and legacy_identity_adapter:
            continue
        expected = route.get(field)
        if captured != expected:
            raise M51DeliveryError(f"capture {field} differs from route")
    if not isinstance(capture_evidence_sha256, str):
        raise M51DeliveryError("capture evidence file identity is invalid")
    return capture_content_sha256


def _validate_gate_capture_identity(
    gate_evidence: Mapping[str, Any],
    *,
    capture_evidence_sha256: str,
    capture_content_sha256: str,
    route_sha256: str,
    request_sha256: str,
    room_family: str,
) -> None:
    _verified_evidence_content_sha256(
        gate_evidence, owner=f"{room_family} visual gate"
    )
    mixed = gate_evidence.get("mixed_capture")
    evidence_record = mixed.get("evidence") if isinstance(mixed, Mapping) else None
    if _record_sha256(evidence_record, owner="gate mixed capture") != capture_evidence_sha256:
        raise M51DeliveryError("visual gate binds a different capture evidence file")
    declared_capture_content = (
        evidence_record.get("evidence_content_sha256")
        if isinstance(evidence_record, Mapping)
        and evidence_record.get("evidence_content_sha256") is not None
        else mixed.get("evidence_content_sha256")
        if isinstance(mixed, Mapping)
        else None
    )
    if (
        declared_capture_content is not None
        and declared_capture_content != capture_content_sha256
    ):
        raise M51DeliveryError("visual gate binds different capture evidence content")
    if room_family == "mp3d":
        inputs = gate_evidence.get("inputs")
        if not isinstance(inputs, Mapping):
            raise M51DeliveryError("MP3D gate lacks route/request inputs")
        if _record_sha256(inputs.get("route_manifest"), owner="gate route") != route_sha256:
            raise M51DeliveryError("MP3D gate route-manifest bytes differ")
        if _record_sha256(inputs.get("m1_request"), owner="gate request") != request_sha256:
            raise M51DeliveryError("MP3D gate M1-request bytes differ")
    else:
        if _record_sha256(gate_evidence.get("route_manifest"), owner="gate route") != route_sha256:
            raise M51DeliveryError("ReplicaCAD gate route-manifest bytes differ")
        if _record_sha256(gate_evidence.get("m1_request"), owner="gate request") != request_sha256:
            raise M51DeliveryError("ReplicaCAD gate M1-request bytes differ")


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


def _persistent_claim_boundary(
    frames: np.ndarray, *, room_family: str
) -> np.ndarray:
    """Burn the review-only claim boundary into every standalone frame."""

    first = "RESEARCH ONLY | UNQUALIFIED ACOUSTICS | ROOT-CENTER CLEARANCE ONLY"
    second = (
        "ACOUSTIC GEOMETRY: STAGE SURFACE ONLY | BEAGLE DRY-AUDIO RIGHTS: UNRESOLVED"
        if room_family == "replicacad"
        else "BEAGLE DRY-AUDIO RIGHTS: UNRESOLVED / SEE RETAINED PROVENANCE"
    )
    output: list[np.ndarray] = []
    for raw in np.asarray(frames):
        image = Image.fromarray(raw, mode="RGB")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        top = 122
        draw.rectangle((0, top, image.width, top + 42), fill=(0, 0, 0, 218))
        draw.text((8, top + 2), first, fill=(255, 214, 72, 255), font=_font(13))
        draw.text((8, top + 21), second, fill=(255, 255, 255, 255), font=_font(12))
        output.append(
            np.asarray(
                Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"),
                dtype=np.uint8,
            )
        )
    return np.ascontiguousarray(np.stack(output, axis=0))


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
    room_family = str(args.room_family)
    gate_name = (
        "mp3d_gate_evidence.json"
        if room_family == "mp3d"
        else "replicacad_gate_evidence.json"
    )
    room_label = "MP3D" if room_family == "mp3d" else "ReplicaCAD"
    review_stem = (
        "mp3d_human_beagle_annotated_binaural.mp4"
        if room_family == "mp3d"
        else "replicacad_human_beagle_annotated_binaural.mp4"
    )
    delivery_schema = (
        MP3D_DELIVERY_SCHEMA
        if room_family == "mp3d"
        else "avengine_m5_1_replicacad_delivery_v1"
    )
    staging = output.with_name(f".{output.name}.staging")
    if os.path.lexists(output) or os.path.lexists(staging):
        raise M51DeliveryError(f"refusing to replace delivery output: {output}")
    required = (
        capture / "evidence.json",
        capture / gate_name,
        acoustics / "evidence.json",
        source_path,
        route_path,
        request_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise M51DeliveryError(f"{room_label} delivery input is missing: {missing}")

    capture_evidence = load_json(required[0])
    gate_evidence = load_json(required[1])
    acoustic_evidence = load_json(required[2])
    if (
        capture_evidence.get("status") != "pass"
        or capture_evidence.get("qualification_claim") is not False
        or capture_evidence.get("frame_count") != 270
        or capture_evidence.get("frame_rate_hz") != 15
    ):
        raise M51DeliveryError(
            f"{room_label} capture evidence is not the bounded 18-second pass"
        )
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
        raise M51DeliveryError(
            f"{room_label} RIR evidence is not an unqualified research pass"
        )
    source = load_source_manifest(source_path)
    if room_family == "mp3d":
        route = load_mp3d_route_manifest(route_path)
        route_paths = derive_mp3d_route_paths(route)
    else:
        route = load_replicacad_route_manifest(route_path)
        route_paths = derive_replicacad_route_paths(route)
    request = load_json(request_path)
    if request.get("request_id") != route["request_id"] or request.get("room_id") != route["room_id"]:
        raise M51DeliveryError(f"M1 request identity differs from {room_label} route")
    validate_room_visual_gate(
        gate_evidence,
        route,
        room_family=room_family,
    )
    capture_evidence_sha256 = sha256_file(capture / "evidence.json")
    route_sha256 = sha256_file(route_path)
    request_sha256 = sha256_file(request_path)
    capture_content_sha256 = _validate_capture_route_request_identity(
        capture_evidence,
        capture_evidence_sha256=capture_evidence_sha256,
        route=route,
        route_sha256=route_sha256,
        request=request,
        request_sha256=request_sha256,
        room_family=room_family,
    )
    _validate_gate_capture_identity(
        gate_evidence,
        capture_evidence_sha256=capture_evidence_sha256,
        capture_content_sha256=capture_content_sha256,
        route_sha256=route_sha256,
        request_sha256=request_sha256,
        room_family=room_family,
    )
    bindings = source_actor_binding_record(
        source,
        route,
        capture_evidence,
        room_family=room_family,
    )
    binding_entries = source_binding_entries(bindings)

    rgb = _capture_array(capture, capture_evidence, "rgb")
    semantic = _capture_array(capture, capture_evidence, "semantic")
    anchors = _capture_array(capture, capture_evidence, "anchor_positions_m")
    matrices = _capture_array(capture, capture_evidence, "actor_world_matrices")
    actor_centers = matrices[:, :, :3, 3]
    if not np.array_equal(actor_centers[:, 0], route_paths.human) or not np.array_equal(
        actor_centers[:, 1], route_paths.beagle
    ):
        raise M51DeliveryError(
            f"captured {room_label} actor roots differ from route paths"
        )

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
        {
            source_id: anchors[:, int(binding["emitter_anchor_index"]), :]
            for source_id, binding in binding_entries.items()
        },
        visual_frame_rate_hz=15,
        rir_stride_frames=3,
        listener_position_m=listener,
        listener_orientation_wxyz=orientation_wxyz,
    )
    sequence = load_retained_binaural_sequence(
        acoustics,
        grid=grid,
        expected_room_id=str(route["room_id"]),
        expected_route_id=str(route["route_id"]),
        expected_request_id=str(route["request_id"]),
        expected_capture_evidence_sha256=capture_evidence_sha256,
        expected_capture_content_sha256=capture_content_sha256,
        expected_source_manifest_sha256=sha256_file(source_path),
        expected_source_binding_sha256=str(bindings["record_content_sha256"]),
    )
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
    event_audio_activity = verify_source_event_audio_activity(
        source,
        clip,
        dry_buses=dry.buses,
        binaural_stems={
            source_id: stems[source_id].episode for source_id in grid.source_ids
        },
    )
    mixture_peak = float(np.max(np.abs(mixture)))
    if not 0.0 < mixture_peak < 1.0:
        raise M51DeliveryError(
            f"{room_label} binaural mixture must be audible and unclipped; peak={mixture_peak}"
        )

    qa_view = _qa_view(request)
    pathfinder_record = gate_evidence.get("pathfinder")
    if not isinstance(pathfinder_record, Mapping):
        raise M51DeliveryError(f"{room_label} gate evidence lacks PathFinder record")
    navmesh_record = pathfinder_record.get("declared_navmesh")
    if not isinstance(navmesh_record, Mapping):
        raise M51DeliveryError(f"{room_label} gate evidence lacks declared navmesh")
    runtime_navmesh_record = _runtime_navmesh_record(
        navmesh_record,
        room_family=room_family,
        replicacad_root=args.replicacad_root,
    )
    navmesh = load_real_mp3d_navmesh_qa(
        navmesh_record=runtime_navmesh_record,
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
        audio_role_prefix = f"m5_1_{room_family}"
        dry_metadata = dry.metadata()
        if room_family == "replicacad":
            dry_metadata = _portable_dry_audio_metadata(dry_metadata)
        retained_assembly_sha256 = str(dry_metadata["assembly_content_sha256"])
        audio_records: dict[str, Any] = {"dry": {}, "binaural_stems": {}}
        for source_id in grid.source_ids:
            audio_records["dry"][source_id] = _audio_record(
                staging / "audio/dry" / f"{source_id}.wav",
                dry.buses[source_id][None, :],
                metadata={
                    "role": f"{audio_role_prefix}_scheduled_dry_source_bus",
                    "source_id": source_id,
                    "assembly_content_sha256": retained_assembly_sha256,
                    "qualification_claim": False,
                },
                root=staging,
            )
            audio_records["binaural_stems"][source_id] = _audio_record(
                staging / "audio/binaural" / f"{source_id}_stem.wav",
                stems[source_id].episode,
                metadata={
                    "role": f"{audio_role_prefix}_dynamic_binaural_stem",
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
                "role": f"{audio_role_prefix}_dynamic_binaural_mixture",
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
            capture_evidence_sha256=capture_evidence_sha256,
            source_actor_bindings=bindings,
        )
        reuse_record = source_program_reuse_record(source)
        if room_family == "replicacad":
            reuse_record.pop("record_content_sha256", None)
            reuse_record["schema"] = (
                "avengine_m5_1_habitat_native_source_program_reuse_v1"
            )
            reuse_record["room_family"] = "replicacad"
            reuse_record["mp3d_visual_provenance_authority"] = (
                "not_applicable; authenticated ReplicaCAD mixed capture evidence"
            )
            reuse_record["mp3d_spatial_authorities"] = []
            reuse_record["habitat_native_spatial_authorities"] = [
                "captured articulated emitter-link world transforms",
                "M1 co-located camera/listener transform",
                "real ReplicaCAD apt_0 PathFinder gate evidence",
                "captured semantic visibility",
            ]
            reuse_record = _portable_source_program_audio(reuse_record)
        write_json(staging / "actual_emitter_trajectories.json", actual_trajectory)
        write_json(staging / "source_actor_bindings.json", bindings)
        write_json(staging / "source_program_reuse.json", reuse_record)
        write_json(staging / "dry_audio_assembly.json", dry_metadata)
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
                "schema": (
                    "avengine_m5_1_mp3d_navmesh_center_diagnostics_v1"
                    if room_family == "mp3d"
                    else "avengine_m5_1_replicacad_navmesh_center_diagnostics_v1"
                ),
                "center_navigation_semantics": "actor_root_center_only",
                "full_articulated_mesh_clearance_claim": False,
                "navmesh": (
                    navmesh.navmesh_record
                    if room_family == "mp3d"
                    else dict(navmesh_record)
                ),
                "bounds_m": [list(value) for value in navmesh.bounds_m],
                "shared_island_id": int(shared_islands[0]),
                "actors": {
                    actor_id: {
                        "all_frames_navigable": bool(np.all(navmesh.navigable[actor_id])),
                        **_clearance_summary(navmesh.clearance_m[actor_id]),
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
            source_actor_bindings=bindings,
        )
        topdown = render_mp3d_topdown_frames(
            navmesh_binary_map=navmesh.binary_map,
            navmesh_bounds_m=navmesh.bounds_m,
            actor_center_paths_m={"human0": route_paths.human, "dog0": route_paths.beagle},
            listener_position_m=listener,
            listener_yaw_deg=listener_yaw,
            camera_hfov_degrees=float(
                request["primary_camera_rig"]["shared_calibration"]["hfov_degrees"]
            ),
            clearance_m=navmesh.clearance_m,
            shared_island_id=int(shared_islands[0]),
            source_actor_bindings=bindings,
        )
        annotated = compose_annotated_frames(
            main_rgb=rgb,
            topdown_rgb=topdown,
            tracks=tracks,
            clip_id=f"{room_family}_human_beagle_source_program_clip",
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
        annotated = _persistent_claim_boundary(annotated, room_family=room_family)
        contact_sheet = _contact_sheet(
            annotated, staging / "qa/annotated_contact_sheet.jpg"
        )
        review_path = staging / "videos" / review_stem
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
            "schema": delivery_schema,
            "status": "pass",
            "research_only": True,
            "qualification_claim": False,
            "claim_boundary": (
                f"M5.1 {room_label} annotated research review. Pathfinder gates actor root "
                "centers only; unqualified research-placeholder acoustics and room "
                "materials; "
                + (
                    "ReplicaCAD acoustics use stage-surface geometry only; "
                    if room_family == "replicacad"
                    else ""
                )
                +
                "Beagle dry-audio rights remain unresolved and follow retained source "
                "provenance; no full-mesh, "
                "asset, room, material, episode, or dataset "
                "admission claim."
            ),
            "inputs": {
                "capture_evidence": (
                    _absolute_file_record(capture / "evidence.json")
                    if room_family == "mp3d"
                    else _repository_file_record(capture / "evidence.json")
                ),
                f"{room_family}_gate_evidence": (
                    _absolute_file_record(capture / gate_name)
                    if room_family == "mp3d"
                    else _repository_file_record(capture / gate_name)
                ),
                "acoustic_evidence": (
                    _absolute_file_record(acoustics / "evidence.json")
                    if room_family == "mp3d"
                    else _repository_file_record(acoustics / "evidence.json")
                ),
                "source_manifest": (
                    _absolute_file_record(source_path)
                    if room_family == "mp3d"
                    else _repository_file_record(source_path)
                ),
                "route_manifest": (
                    _absolute_file_record(route_path)
                    if room_family == "mp3d"
                    else _repository_file_record(route_path)
                ),
                "m1_request": (
                    _absolute_file_record(request_path)
                    if room_family == "mp3d"
                    else _repository_file_record(request_path)
                ),
            },
            "timeline": {
                "frame_count": 270,
                "frame_rate_hz": 15,
                "duration_seconds": 18,
                "audio_sample_rate_hz": 16_000,
                "audio_sample_count": 288_000,
            },
            "source_ids": list(grid.source_ids),
            "source_actor_bindings": file_record(
                staging / "source_actor_bindings.json", relative_to=staging
            ),
            "source_actor_binding_content_sha256": bindings[
                "record_content_sha256"
            ],
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
                "source_event_audio_activity": event_audio_activity,
                "frame_diagnostics": file_record(
                    staging / "binaural_frame_diagnostics.json", relative_to=staging
                ),
            },
            "review_media": {
                f"annotated_{room_family}": review_report,
                "contact_sheet": contact_sheet,
                "topdown_is_qa_only": True,
                "topdown_authority": "Habitat PathFinder.get_topdown_view",
                "persistent_claim_boundary": [
                    "RESEARCH ONLY",
                    "UNQUALIFIED ACOUSTICS",
                    "ROOT-CENTER CLEARANCE ONLY",
                    *(
                        ["ACOUSTIC GEOMETRY: STAGE SURFACE ONLY"]
                        if room_family == "replicacad"
                        else []
                    ),
                    "BEAGLE DRY-AUDIO RIGHTS: UNRESOLVED / SEE RETAINED PROVENANCE",
                ],
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

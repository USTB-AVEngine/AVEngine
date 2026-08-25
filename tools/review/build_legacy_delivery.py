#!/usr/bin/env python3
"""Build the final annotated 18-second M5.1 legacy comparison delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
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
from avengine.spatial_audio.audio import read_float32_wav, write_float32_wav
from avengine.capture.acoustics import (
    build_strided_review_keyframes,
    render_research_review_binaural_audio,
)
from avengine.capture.delivery import (
    DELIVERY_SCHEMA,
    M51DeliveryError,
    actual_emitter_trajectory_record,
    binaural_frame_diagnostics,
    build_legacy_overlay_tracks,
    declared_audio_asset_bindings,
    executable_event_mappings,
    load_retained_binaural_sequence,
    verify_audio_program_receipts,
)
from avengine.capture.dry_audio import DryAudioClipSpec, assemble_dry_audio_buses
from avengine.capture.review import compose_annotated_frames, encode_annotated_review
from avengine.capture.source_contracts import load_source_manifest
from avengine.capture.topdown import render_legacy_topdown_frames


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--acoustics-dir", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--route-manifest", required=True, type=Path)
    parser.add_argument("--old-review-video", required=True, type=Path)
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
        raise M51DeliveryError("authoritative float32 WAVE differs on readback")
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
    indices = (0, 63, 90, 107, 147, 180, 188, 269)
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
    return file_record(path, relative_to=path.parents[1])


def _comparison_video(
    old_review: Path,
    new_review: Path,
    output: Path,
) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise M51DeliveryError("ffmpeg and ffprobe are required")
    if not old_review.is_file() or not new_review.is_file():
        raise M51DeliveryError("old/new comparison video is missing")
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    filters = (
        f"[0:v]crop=640:480:0:0,drawbox=x=0:y=438:w=640:h=42:color=black@0.7:t=fill,"
        f"drawtext=fontfile={font}:text='OLD UE':x=12:y=446:fontsize=24:fontcolor=white[old];"
        f"[1:v]crop=640:480:0:0,drawbox=x=0:y=438:w=640:h=42:color=black@0.7:t=fill,"
        f"drawtext=fontfile={font}:text='NEW HABITAT':x=12:y=446:fontsize=24:fontcolor=white[new];"
        f"[1:v]crop=640:480:640:0,drawbox=x=0:y=438:w=640:h=42:color=black@0.7:t=fill,"
        f"drawtext=fontfile={font}:text='COMMON TOPDOWN QA':x=12:y=446:fontsize=24:fontcolor=white[top];"
        "[old][new][top]hstack=inputs=3[v]"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(old_review),
        "-i",
        str(new_review),
        "-filter_complex",
        filters,
        "-map",
        "[v]",
        "-map",
        "1:a:0",
        "-frames:v",
        "270",
        "-t",
        "18",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "15",
        "-bf",
        "0",
        "-threads",
        "1",
        "-c:a",
        "aac",
        "-ar",
        "16000",
        "-ac",
        "2",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(output),
    ]
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    if run.returncode != 0:
        output.unlink(missing_ok=True)
        raise M51DeliveryError(f"engine comparison encoding failed: {run.stderr.strip()}")
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,nb_read_frames,sample_rate,channels:format=duration",
            "-of",
            "json",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise M51DeliveryError(f"comparison ffprobe failed: {probe.stderr.strip()}")
    payload = json.loads(probe.stdout)
    video = next(
        (stream for stream in payload["streams"] if stream.get("codec_type") == "video"),
        {},
    )
    audio = next(
        (stream for stream in payload["streams"] if stream.get("codec_type") == "audio"),
        {},
    )
    if (
        video.get("codec_name") != "h264"
        or int(video.get("width", 0)) != 1920
        or int(video.get("height", 0)) != 480
        or int(video.get("nb_read_frames", 0)) != 270
        or audio.get("codec_name") != "aac"
        or int(audio.get("sample_rate", 0)) != 16_000
        or int(audio.get("channels", 0)) != 2
    ):
        raise M51DeliveryError("engine comparison readback differs")
    return {
        "path": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "byte_size": output.stat().st_size,
        "frame_count": 270,
        "frame_rate_hz": 15,
        "duration_seconds": float(payload["format"]["duration"]),
        "width": 1920,
        "height": 480,
        "audio_sample_rate_hz": 16_000,
        "audio_channel_count": 2,
        "panels": ["old_ue_main", "new_habitat_main", "new_topdown_qa"],
        "ffprobe": payload,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    capture = args.capture_dir.resolve()
    acoustics = args.acoustics_dir.resolve()
    source_path = args.source_manifest.resolve()
    route_path = args.route_manifest.resolve()
    old_review = args.old_review_video.resolve()
    output = args.output_dir.resolve()
    staging = output.with_name(f".{output.name}.staging")
    if os.path.lexists(output) or os.path.lexists(staging):
        raise M51DeliveryError(f"refusing to replace delivery output: {output}")
    required = (
        capture / "arrays/rgb.npy",
        capture / "arrays/semantic.npy",
        capture / "arrays/anchor_positions_m.npy",
        capture / "evidence.json",
        acoustics / "evidence.json",
        source_path,
        route_path,
        old_review,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise M51DeliveryError(f"delivery input is missing: {missing}")
    source = load_source_manifest(source_path)
    route = load_json(route_path)
    rgb = np.load(required[0], allow_pickle=False)
    semantic = np.load(required[1], allow_pickle=False)
    anchors = np.load(required[2], allow_pickle=False)
    clip_mapping = source["clip"]
    clip = DryAudioClipSpec.from_values(
        frame_count=int(clip_mapping["frame_count"]),
        fps_numerator=int(clip_mapping["fps_num"]),
        fps_denominator=int(clip_mapping["fps_den"]),
        sample_rate_hz=int(clip_mapping["sample_rate_hz"]),
    )
    if clip.sample_count != int(clip_mapping["sample_count"]) or clip.sample_count != 288_000:
        raise M51DeliveryError("source manifest clip does not describe exact 18-second audio")
    observer = source["observer"]
    listener = np.asarray(observer["position_m"], dtype=np.float64)
    half_yaw = math.radians(float(observer["yaw_deg"])) / 2.0
    orientation_wxyz = (math.cos(half_yaw), 0.0, math.sin(half_yaw), 0.0)
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
    stems, mixture = render_research_review_binaural_audio(
        dry.buses,
        sequence,
        grid=grid,
    )
    mixture_peak = float(np.max(np.abs(mixture)))
    if not 0.0 < mixture_peak < 1.0:
        raise M51DeliveryError(
            f"binaural review mixture must be audible and unclipped; peak={mixture_peak}"
        )

    staging.mkdir(parents=True)
    try:
        audio_records: dict[str, Any] = {"dry": {}, "binaural_stems": {}}
        for source_id in grid.source_ids:
            audio_records["dry"][source_id] = _audio_record(
                staging / "audio/dry" / f"{source_id}.wav",
                dry.buses[source_id][None, :],
                metadata={
                    "role": "m5_1_scheduled_dry_source_bus",
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
                    "role": "m5_1_dynamic_binaural_stem",
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
                "role": "m5_1_dynamic_binaural_mixture",
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
        write_json(staging / "actual_emitter_trajectories.json", actual_trajectory)
        write_json(staging / "dry_audio_assembly.json", dry.metadata())
        diagnostic_labels, diagnostic_records = binaural_frame_diagnostics(
            mixture, clip
        )
        write_json(
            staging / "binaural_frame_diagnostics.json",
            {
                "schema": "avengine_m5_1_binaural_frame_diagnostics_v1",
                "metric_boundary": (
                    "Review-only frame-local ILD and broadband cross-correlation "
                    "ITD; not an HRTF qualification metric"
                ),
                "records": list(diagnostic_records),
            },
        )
        tracks = build_legacy_overlay_tracks(
            source,
            route,
            anchor_positions_m=anchors,
            semantic_frames=semantic,
        )
        topdown = render_legacy_topdown_frames(route)
        aggregate_flags = tuple(
            sorted(
                flag_id
                for flag_id, assessment in source["clip_flags"].items()
                if assessment["status"] == "present" and assessment["value"] is True
            )
        )
        center_gate_pass = all(
            int(route["gates"][gate]["collision_count"]) == 0
            for gate in ("human_center_point_aabb", "dog_center_point_aabb")
        )
        annotated = compose_annotated_frames(
            main_rgb=rgb,
            topdown_rgb=topdown,
            tracks=tracks,
            clip_id=str(source["clip"]["clip_id"]),
            room_id="legacy_ue_apartment_0000",
            listener_position_m=listener,
            listener_yaw_deg=float(observer["yaw_deg"]),
            aggregate_true_flags=aggregate_flags,
            audio_diagnostic_by_frame=diagnostic_labels,
            center_gate_pass=center_gate_pass,
            fps=15,
        )
        contact_sheet = _contact_sheet(
            annotated, staging / "qa/annotated_contact_sheet.jpg"
        )
        review_path = staging / "videos/legacy_apartment_habitat_annotated_binaural.mp4"
        review_report = dict(
            encode_annotated_review(
                annotated,
                review_path,
                fps=15,
                audio_path=mixture_path,
            )
        )
        review_report["path"] = str(review_path.relative_to(staging))
        comparison_path = staging / "videos/legacy_apartment_ue_vs_habitat.mp4"
        comparison_report = _comparison_video(
            old_review, review_path, comparison_path
        )
        comparison_report["path"] = str(comparison_path.relative_to(staging))
        evidence: dict[str, Any] = {
            "schema": DELIVERY_SCHEMA,
            "status": "pass",
            "research_only": True,
            "qualification_claim": False,
            "claim_boundary": (
                "M5.1 annotated engine-comparison review using unqualified "
                "research-placeholder room materials; no asset, room, material, "
                "episode, or dataset admission claim"
            ),
            "inputs": {
                "capture_evidence": _absolute_file_record(capture / "evidence.json"),
                "acoustic_evidence": _absolute_file_record(acoustics / "evidence.json"),
                "source_manifest": _absolute_file_record(source_path),
                "route_manifest": _absolute_file_record(route_path),
                "old_review_video": _absolute_file_record(old_review),
            },
            "timeline": {
                "frame_count": 270,
                "frame_rate_hz": 15,
                "duration_seconds": 18,
                "audio_sample_rate_hz": 16_000,
                "audio_sample_count": 288_000,
            },
            "source_ids": list(grid.source_ids),
            "simultaneous_event_overlaps": source["relationships"][0][
                "event_overlap_windows"
            ],
            "point_collision_gate": {
                "status": "pass" if center_gate_pass else "fail",
                "human_minimum_clearance_m": route["gates"][
                    "human_center_point_aabb"
                ]["closest"]["clearance_m"],
                "dog_minimum_clearance_m": route["gates"][
                    "dog_center_point_aabb"
                ]["closest"]["clearance_m"],
                "minimum_human_dog_center_distance_m": route["gates"]
                ["inter_source_center_separation"]["minimum_observed_m"],
            },
            "actual_emitter_trajectories": file_record(
                staging / "actual_emitter_trajectories.json", relative_to=staging
            ),
            "dry_audio_assembly": file_record(
                staging / "dry_audio_assembly.json", relative_to=staging
            ),
            "dynamic_binaural": {
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
                    staging / "binaural_frame_diagnostics.json",
                    relative_to=staging,
                ),
            },
            "review_media": {
                "annotated_habitat": review_report,
                "direct_engine_comparison": comparison_report,
                "contact_sheet": contact_sheet,
                "topdown_is_qa_only": True,
            },
        }
        if not center_gate_pass:
            raise M51DeliveryError("legacy center-point collision gate failed")
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

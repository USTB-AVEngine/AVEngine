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
from PIL import Image, ImageDraw

from avengine.contracts.json_io import load_json, sha256_file, write_json
from avengine.m6x.geometry import RuntimeObstacleMap
from avengine.m6x.topdown import render_runtime_topdown_frames
from avengine.m7.apartment_visual_bundle import (
    ASSET_VISUAL_PROFILES,
    FRAME_COUNT,
    FRAME_RATE_HZ,
    binding_assets_by_episode,
    build_flags,
    build_qualification,
    build_source_manifest,
    build_timeline,
)
from avengine.optional_backends.spear_apartment import (
    build_rawvideo_encode_command,
)


SCHEMA = "avengine_m7_asset_bound_apartment_ue_input_bundle_v1"
SOURCE_SLOTS = ("source1", "source2")


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
    result = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("variant_index") != 0:
            continue
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or episode_id in result:
            raise RuntimeError("audio batch v00 episode IDs are invalid")
        result[episode_id] = row
    return result


def _info_topdown_frames(
    *, episode_id: str, motion_case: str, bindings: Mapping[str, Mapping[str, Any]], topdown: np.ndarray
) -> np.ndarray:
    frames = []
    labels = {
        slot: ASSET_VISUAL_PROFILES[bindings[slot]["asset_id"]]["display_label"]
        for slot in SOURCE_SLOTS
    }
    for frame_index in range(FRAME_COUNT):
        panel = Image.new("RGB", (640, 480), (22, 28, 36))
        draw = ImageDraw.Draw(panel)
        draw.text((24, 24), "M7 APARTMENT ASSET-BOUND EPISODE", fill=(255, 255, 255))
        draw.text((24, 58), episode_id, fill=(180, 198, 216))
        draw.text((24, 98), f"motion: {motion_case}", fill=(230, 230, 230))
        draw.text((24, 132), f"source1: {labels['source1']}", fill=(42, 210, 220))
        draw.text((24, 166), f"source2: {labels['source2']}", fill=(250, 120, 70))
        draw.text((24, 216), "both sources active; audio is 360 degrees", fill=(225, 230, 238))
        draw.text((24, 250), "right: authoritative source-center Topdown", fill=(225, 230, 238))
        draw.text((24, 284), "visual route rendered once; audio variants reused", fill=(225, 230, 238))
        draw.text((24, 440), f"frame {frame_index:02d}/74", fill=(174, 185, 197))
        frames.append(np.concatenate((np.asarray(panel), topdown[frame_index]), axis=1))
    return np.ascontiguousarray(np.stack(frames), dtype=np.uint8)


def _encode_diagnostic(
    frames: np.ndarray,
    *,
    audio_path: Path,
    output_path: Path,
    video_encoder: str,
    encoder_gpu: int | None,
) -> None:
    """Stream RGB frames through FFmpeg and add the exact five-second WAV."""

    visual = output_path.with_name(f".{output_path.stem}.visual.mp4")
    command = build_rawvideo_encode_command(
        output_path=visual,
        video_encoder=video_encoder,
        encoder_gpu=encoder_gpu,
        width=1280,
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
    listener: Mapping[str, Any],
    obstacle_map: RuntimeObstacleMap,
    sample: Mapping[str, Any],
    batch_root: Path,
    staging: Path,
    video_encoder: str,
    encoder_gpu: int | None,
) -> dict[str, Any]:
    """Build one independent episode directory for sequential or process use."""

    episode_started = time.perf_counter()
    timeline, headings = build_timeline(
        episode=episode,
        bindings=episode_bindings,
        listener_position_m=listener["position_m"],
    )
    manifest = build_source_manifest(
        episode_id=episode_id,
        episode=episode,
        bindings=episode_bindings,
    )
    episode_root = staging / "episodes" / episode_id
    metadata = episode_root / "metadata"
    videos = episode_root / "videos"
    metadata.mkdir(parents=True)
    videos.mkdir()
    write_json(metadata / "timeline.json", timeline)
    write_json(metadata / "source_manifest.json", manifest)
    write_json(metadata / "flags.json", build_flags())

    centers = {
        slot: np.asarray(episode["source_center_paths_m"][slot], dtype=np.float64)
        for slot in SOURCE_SLOTS
    }
    activity = {slot: np.ones(FRAME_COUNT, dtype=np.bool_) for slot in SOURCE_SLOTS}
    topdown = render_runtime_topdown_frames(
        obstacle_map,
        centers,
        listener_position_m=listener["position_m"],
        listener_yaw_deg=listener["yaw_deg"],
        camera_hfov_degrees=listener["camera_hfov_degrees"],
        source_activity_by_frame=activity,
        source_heading_xz_by_frame=headings,
        source_labels={slot: slot for slot in SOURCE_SLOTS},
        source_colors={"source1": (42, 210, 220), "source2": (250, 120, 70)},
        size_wh=(640, 480),
    )
    review_frames = _info_topdown_frames(
        episode_id=episode_id,
        motion_case=str(episode["motion_case"]),
        bindings=episode_bindings,
        topdown=topdown,
    )
    mixture = sample["audio"]["mixture"]
    mixture_path = batch_root / "audio/binaural" / mixture["path"]
    if not mixture_path.is_file() or sha256_file(mixture_path) != mixture["audio_sha256"]:
        raise RuntimeError(f"batch mixture changed for {episode_id}")
    diagnostic = videos / "diagnostic_topdown_binaural.mp4"
    _encode_diagnostic(
        review_frames,
        audio_path=mixture_path,
        output_path=diagnostic,
        video_encoder=video_encoder,
        encoder_gpu=encoder_gpu,
    )
    os.link(diagnostic, videos / "clean_binaural.mp4")
    write_json(
        metadata / "batch_binding.json",
        {
            "schema": "avengine_m7_asset_bound_apartment_episode_binding_v1",
            "status": "pass",
            "episode_id": episode_id,
            "visual_render_reuse_count": 10,
            "v00_sample_id": sample["sample_id"],
            "v00_mixture": dict(mixture),
            "asset_ids_by_source_slot": sample["asset_ids_by_source_slot"],
            "source_center_gate_status": "pass",
        },
    )
    return {
        "episode_ordinal": ordinal,
        "episode_id": episode_id,
        "motion_case": episode["motion_case"],
        "asset_ids_by_source_slot": sample["asset_ids_by_source_slot"],
        "v00_sample_id": sample["sample_id"],
        "build_wall_seconds": time.perf_counter() - episode_started,
    }


def build_bundle(
    *,
    plan_root: Path,
    feasibility_root: Path,
    batch_root: Path,
    room_template_bundle: Path,
    episode_ids: Sequence[str] | None,
    video_encoder: str,
    encoder_gpu: int | None,
    workers: int,
    output: Path,
) -> Path:
    started = time.perf_counter()
    plan_root = plan_root.resolve()
    feasibility_root = feasibility_root.resolve()
    batch_root = batch_root.resolve()
    room_template_bundle = room_template_bundle.resolve()
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output}")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise ValueError("workers must be an integer between 1 and 32")
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"refusing to replace staging output: {staging}")

    bank = load_json(plan_root / "trajectory_bank.json")
    raw_episodes = bank.get("episodes")
    if not isinstance(raw_episodes, list):
        raise RuntimeError("trajectory bank episodes are invalid")
    episodes = {
        value["episode_id"]: value
        for value in raw_episodes
        if isinstance(value, Mapping) and isinstance(value.get("episode_id"), str)
    }
    bindings = binding_assets_by_episode(
        load_json(plan_root / "asset_emitter_binding_report.json")
    )
    if set(episodes) != set(bindings):
        raise RuntimeError("trajectory bank and asset binding episode sets differ")
    selected = tuple(episode_ids) if episode_ids is not None else tuple(sorted(episodes))
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(episodes):
        raise RuntimeError("episode selection is empty, repeated, or unknown")
    samples = _samples(batch_root)
    if not set(selected) <= set(samples):
        raise RuntimeError("selected episodes lack v00 audio samples")

    qualification_template = load_json(
        room_template_bundle / "room/qualification.json"
    )
    listener = qualification_template["listener"]
    obstacle_map = _obstacle_map(feasibility_root)
    staging.mkdir(parents=True)
    try:
        room_root = staging / "room"
        room_root.mkdir()
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
        task_arguments = [
            {
                "ordinal": ordinal,
                "episode_id": episode_id,
                "episode": episodes[episode_id],
                "episode_bindings": bindings[episode_id],
                "listener": listener,
                "obstacle_map": obstacle_map,
                "sample": samples[episode_id],
                "batch_root": batch_root,
                "staging": staging,
                "video_encoder": video_encoder,
                "encoder_gpu": encoder_gpu,
            }
            for ordinal, episode_id in enumerate(selected)
        ]
        if workers == 1:
            episode_rows = [_build_episode(**values) for values in task_arguments]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(_build_episode, **values)
                    for values in task_arguments
                ]
                episode_rows = [future.result() for future in futures]
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
                "visual_render_policy": "one_UE_render_per_episode_then_reuse_for_10_audio_variants",
                "scene_copy_count": 0,
                "frame_count": FRAME_COUNT,
                "frame_rate_hz": FRAME_RATE_HZ,
                "diagnostic_video_encoder": video_encoder,
                "diagnostic_encoder_gpu": encoder_gpu,
                "episode_workers": workers,
                "episodes": episode_rows,
                "inputs": {
                    "plan_root": str(plan_root),
                    "batch_root": str(batch_root),
                    "feasibility_root": str(feasibility_root),
                    "room_template_bundle": str(room_template_bundle),
                },
                "build_wall_seconds": time.perf_counter() - started,
            },
        )
        os.rename(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--feasibility-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--room-template-bundle", type=Path, required=True)
    parser.add_argument("--episode-id", action="append")
    parser.add_argument(
        "--video-encoder", choices=("libx264", "h264_nvenc"), default="libx264"
    )
    parser.add_argument("--encoder-gpu", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = build_bundle(
        plan_root=args.plan_root,
        feasibility_root=args.feasibility_root,
        batch_root=args.batch_root,
        room_template_bundle=args.room_template_bundle,
        episode_ids=args.episode_id,
        video_encoder=args.video_encoder,
        encoder_gpu=args.encoder_gpu,
        workers=args.workers,
        output=args.output,
    )
    print(f"ASSET_BOUND_APARTMENT_UE_BUNDLE_OK output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

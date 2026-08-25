#!/usr/bin/env python3
"""Build AVEngine Timeline, Topdown and binaural audio for a residential room."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.acoustics.runtime import load_compiled_acoustic_scene  # noqa: E402
from avengine.spatial_audio.audio import read_float32_wav, write_float32_wav  # noqa: E402
from avengine.spatial_audio.runtime import M4SimulationConfig  # noqa: E402
from avengine.capture.acoustics import (  # noqa: E402
    build_strided_review_keyframes,
    render_research_review_binaural_audio,
    render_research_review_binaural_rir_sequence,
)
from avengine.m6x.apartment import listener_orientation_wxyz  # noqa: E402
from avengine.acoustics.rir_cache import (  # noqa: E402
    load_semantic_acoustic_scene,
    render_semantic_rir_cache,
)
from avengine.m6x.room_feasibility import (  # noqa: E402
    TrajectoryBank,
    TrajectoryEpisode,
    build_rir_job_plan,
)
from avengine.acoustics.semantic_rir_cache import SemanticRIRCacheSession  # noqa: E402
from avengine.m7.asset_bound_audio import (  # noqa: E402
    float32_stems_and_exact_mix,
    render_asset_bound_binaural,
)
from avengine.optional_backends.residential_episode import (  # noqa: E402
    DOG_SOURCE_ID,
    FRAME_COUNT,
    FPS,
    HUMAN_SOURCE_ID,
    build_residential_source_episode,
    object_footprint_rectangles_xy,
)


WIDTH = 640
HEIGHT = 480
SOURCE_COLORS = {
    "dog0": (40, 210, 220),
    "human0": (255, 126, 76),
}
ROLE_STYLE = {
    "ground_blocker": ((204, 118, 46), (242, 167, 85)),
    "walkable_floor_covering": ((39, 125, 113), (67, 199, 174)),
    "elevated_object": ((80, 104, 130), (132, 164, 196)),
}

SEMANTIC_RIR_EPISODE_ID = "m6z_residential_full75"
SEMANTIC_SOURCE_SLOT_BY_SOURCE_ID = {
    HUMAN_SOURCE_ID: "source1",
    DOG_SOURCE_ID: "source2",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


class _TopdownTransform:
    def __init__(self, polygon: Sequence[Sequence[float]]) -> None:
        points = np.asarray(polygon, dtype=np.float64)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        extent = maximum - minimum
        self.minimum = minimum
        self.scale = min((WIDTH - 150) / extent[0], (HEIGHT - 100) / extent[1])
        self.offset = np.asarray(
            [
                75.0 + (WIDTH - 150 - extent[0] * self.scale) / 2.0,
                62.0 + (HEIGHT - 100 - extent[1] * self.scale) / 2.0,
            ]
        )

    def point(self, xy: Sequence[float]) -> tuple[int, int]:
        value = (
            np.asarray(xy, dtype=np.float64) - self.minimum
        ) * self.scale + self.offset
        return int(round(float(value[0]))), int(round(float(HEIGHT - value[1])))


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    direction_xy: Sequence[float],
    *,
    color: tuple[int, int, int],
    length: float,
    width: int = 3,
) -> None:
    direction = np.asarray(direction_xy, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-9:
        return
    direction /= norm
    # Image Y is downward.
    screen = np.asarray([direction[0], -direction[1]])
    end = np.asarray(origin, dtype=np.float64) + screen * length
    left = end - screen * 7 + np.asarray([-screen[1], screen[0]]) * 5
    right = end - screen * 7 - np.asarray([-screen[1], screen[0]]) * 5
    draw.line([origin, tuple(end)], fill=color, width=width)
    draw.polygon([tuple(end), tuple(left), tuple(right)], fill=color)


def _render_topdown_frames(episode: Mapping[str, Any], output: Path) -> Path:
    frames_dir = output / "topdown_frames"
    frames_dir.mkdir(parents=True)
    transform = _TopdownTransform(episode["room_polygon_xy_m"])
    polygon_px = [transform.point(point) for point in episode["room_polygon_xy_m"]]
    camera_plan = episode["visual_plan"]["camera"]
    camera_xyz = [
        camera_plan["habitat_position_m"][0],
        camera_plan["habitat_position_m"][2],
        camera_plan["habitat_position_m"][1],
    ]
    camera_yaw = float(camera_plan["ue_yaw_deg"])
    camera_direction = np.asarray(
        [math.cos(math.radians(camera_yaw)), math.sin(math.radians(camera_yaw))]
    )
    camera_px = transform.point(camera_xyz[:2])
    hfov = float(camera_plan["horizontal_fov_deg"])
    fov_distance_m = 4.5
    fov_points = [camera_px]
    for angle in (camera_yaw - hfov / 2.0, camera_yaw + hfov / 2.0):
        endpoint = np.asarray(camera_xyz[:2]) + fov_distance_m * np.asarray(
            [math.cos(math.radians(angle)), math.sin(math.radians(angle))]
        )
        fov_points.append(transform.point(endpoint))

    routes = episode["routes_xyz_m"]
    activity = episode["source_activity_by_frame"]
    source_by_actor = {"dog0": DOG_SOURCE_ID, "human0": HUMAN_SOURCE_ID}
    title_font = _font(18)
    small_font = _font(12)
    for frame_index in range(FRAME_COUNT):
        image = Image.new("RGB", (WIDTH, HEIGHT), (18, 23, 29))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.polygon(
            polygon_px, fill=(45, 54, 63, 255), outline=(205, 215, 225, 255), width=3
        )
        for item in episode["objects"]:
            role = item.get("navigation_role", "ground_blocker")
            style = ROLE_STYLE.get(role, ROLE_STYLE["ground_blocker"])
            if role == "ground_blocker":
                rectangles = object_footprint_rectangles_xy(item)
            else:
                bounds = item["bounds_xyz_m"]
                rectangles = [[bounds[0][:2], bounds[1][:2]]]
            for rectangle in rectangles:
                box = [transform.point(rectangle[0]), transform.point(rectangle[1])]
                xyxy = (
                    min(box[0][0], box[1][0]),
                    min(box[0][1], box[1][1]),
                    max(box[0][0], box[1][0]),
                    max(box[0][1], box[1][1]),
                )
                draw.rectangle(
                    xyxy,
                    fill=(*style[0], 70),
                    outline=(*style[1], 170),
                    width=1,
                )

        draw.polygon(fov_points, fill=(245, 220, 92, 38), outline=(245, 220, 92, 190))
        mic_radius = max(13, int(round(0.25 * transform.scale)))
        draw.ellipse(
            [
                camera_px[0] - mic_radius,
                camera_px[1] - mic_radius,
                camera_px[0] + mic_radius,
                camera_px[1] + mic_radius,
            ],
            outline=(255, 220, 91, 245),
            width=3,
        )
        _draw_arrow(draw, camera_px, camera_direction, color=(255, 220, 91), length=38)
        draw.text(
            (camera_px[0] + 10, camera_px[1] + 8),
            "camera + 360° mic",
            font=small_font,
            fill=(255, 236, 151),
        )

        for actor_id in ("dog0", "human0"):
            color = SOURCE_COLORS[actor_id]
            route = routes[actor_id]
            path_px = [transform.point(point[:2]) for point in route]
            draw.line(path_px, fill=(*color, 135), width=3)
            current = path_px[frame_index]
            active = bool(activity[source_by_actor[actor_id]][frame_index])
            radius = 10 if active else 7
            if active:
                draw.ellipse(
                    [
                        current[0] - 15,
                        current[1] - 15,
                        current[0] + 15,
                        current[1] + 15,
                    ],
                    outline=(*color, 200),
                    width=3,
                )
            draw.ellipse(
                [
                    current[0] - radius,
                    current[1] - radius,
                    current[0] + radius,
                    current[1] + radius,
                ],
                fill=(*color, 255),
                outline=(255, 255, 255, 245),
                width=2,
            )
            if frame_index + 1 < FRAME_COUNT:
                delta = np.asarray(route[frame_index + 1][:2]) - np.asarray(
                    route[frame_index][:2]
                )
            else:
                delta = np.asarray(route[frame_index][:2]) - np.asarray(
                    route[frame_index - 1][:2]
                )
            _draw_arrow(draw, current, delta, color=color, length=25)
            label = f"{actor_id} / {'ACTIVE' if active else 'silent'}"
            draw.text(
                (current[0] + 11, current[1] - 18),
                label,
                font=small_font,
                fill=(*color, 255),
            )

        draw.rectangle((0, 0, WIDTH, 48), fill=(10, 13, 17, 235))
        draw.text(
            (14, 8),
            f"{episode['scene']['scene_id']}  human + Beagle",
            font=title_font,
            fill=(245, 247, 250),
        )
        draw.text(
            (14, 30),
            f"frame {frame_index:02d}/74  source-center gate: PASS",
            font=small_font,
            fill=(170, 225, 178),
        )
        draw.rectangle((0, HEIGHT - 34, WIDTH, HEIGHT), fill=(10, 13, 17, 225))
        draw.text(
            (12, HEIGHT - 27),
            "orange=blocking  teal=rug/floor  blue=overhead  audio has no camera-FOV cutoff",
            font=small_font,
            fill=(220, 226, 232),
        )
        image.save(frames_dir / f"frame_{frame_index:04d}.png")

    video = output / "topdown_only.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-frames:v",
            str(FRAME_COUNT),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FPS),
            "-y",
            str(video),
        ],
        check=True,
    )
    return video


def _render_audio(
    *,
    episode: Mapping[str, Any],
    dry_root: Path,
    acoustic_manifest: Path,
    simulation_request: Path,
    hrtf: Path,
    output: Path,
) -> dict[str, Any]:
    proxy = episode.get("acoustic_proxy")
    if not isinstance(proxy, Mapping):
        raise RuntimeError("episode lacks an explicit acoustic proxy declaration")
    translation = np.asarray(
        proxy.get("coordinate_translation_habitat_m", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise RuntimeError("acoustic proxy translation must contain three numbers")

    trajectories = {
        source_id: (np.asarray(points, dtype=np.float64) + translation).tolist()
        for source_id, points in episode["source_trajectories_habitat_m"].items()
    }
    listener = (
        np.asarray(
            episode["visual_plan"]["camera"]["habitat_position_m"], dtype=np.float64
        )
        + translation
    )
    listener_yaw = float(episode["visual_plan"]["camera"]["habitat_yaw_deg"])
    grid = build_strided_review_keyframes(
        trajectories,
        visual_frame_rate_hz=FPS,
        rir_stride_frames=int(proxy.get("rir_stride_frames", 3)),
        listener_position_m=listener,
        listener_orientation_wxyz=listener_orientation_wxyz(listener_yaw),
    )
    scene = load_compiled_acoustic_scene(
        acoustic_manifest, allow_nonpassing_research_qa=True
    )
    request = _load(simulation_request)
    simulation = M4SimulationConfig.from_mapping(request["simulation"])
    sequence = render_research_review_binaural_rir_sequence(
        scene, simulation, grid=grid, hrtf_file_path=str(hrtf.resolve())
    )

    dry_paths = {
        DOG_SOURCE_ID: dry_root / "m6x_dog0_muzzle.wav",
        HUMAN_SOURCE_ID: dry_root / "m6x_human0_mouth.wav",
    }
    dry = {}
    for source_id, path in dry_paths.items():
        value = read_float32_wav(path)
        if value.sample_rate_hz != 16_000 or value.samples.shape != (1, 80_000):
            raise RuntimeError(f"unexpected dry bus contract: {path}")
        dry[source_id] = np.asarray(value.samples[0], dtype=np.float64)
    stems, mixture = render_research_review_binaural_audio(dry, sequence, grid=grid)
    peak = float(np.max(np.abs(mixture)))
    if not 1.0e-8 < peak < 1.0:
        raise RuntimeError(
            f"binaural preview must be audible and unclipped, peak={peak}"
        )

    audio_dir = output / "audio"
    for source_id in grid.source_ids:
        write_float32_wav(
            audio_dir / f"{source_id}_stem.wav",
            stems[source_id].episode,
            16_000,
            metadata={
                "role": "residential_review_binaural_stem",
                "source_id": source_id,
                "acoustic_proxy": dict(proxy),
                "qualification_claim": False,
            },
        )
    mixture_path = audio_dir / "mixture.wav"
    write_float32_wav(
        mixture_path,
        mixture,
        16_000,
        metadata={
            "role": "residential_review_binaural_mixture",
            "acoustic_proxy": dict(proxy),
            "qualification_claim": False,
        },
    )
    evidence = {
        "status": "pass",
        "source_ids": list(grid.source_ids),
        "layout": "binaural_left_right",
        "sample_rate_hz": 16_000,
        "sample_count": 80_000,
        "rir_keyframe_count": len(grid.keyframes),
        "rir_stride_frames": grid.rir_stride_frames,
        "mixture_peak": peak,
        "mixture": str(mixture_path),
        "proxy": dict(proxy),
        "claim_boundary": (
            "directional review rendered in the declared generic acoustic proxy; "
            "not a claim about the visual room's exact materials or RT60"
        ),
    }
    _write(output / "audio_evidence.json", evidence)
    return evidence


def _build_semantic_rir_job_plan(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt one M6z full75 episode to M6x's exact semantic RIR plan."""

    source_roots = {
        "source1": np.asarray(episode["routes_habitat_m"]["human0"], dtype=np.float64),
        "source2": np.asarray(episode["routes_habitat_m"]["dog0"], dtype=np.float64),
    }
    source_centers = {
        "source1": np.asarray(
            episode["source_trajectories_habitat_m"][HUMAN_SOURCE_ID],
            dtype=np.float64,
        ),
        "source2": np.asarray(
            episode["source_trajectories_habitat_m"][DOG_SOURCE_ID],
            dtype=np.float64,
        ),
    }
    bank = TrajectoryBank(
        episodes=(
            TrajectoryEpisode(
                episode_id=SEMANTIC_RIR_EPISODE_ID,
                motion_case="both_moving",
                source_root_paths_m=source_roots,
                source_center_paths_m=source_centers,
                statistics={},
            ),
        ),
        frame_count=FRAME_COUNT,
        frame_rate_hz=FPS,
        seed=0,
    )
    listener = episode["qualification"]["listener"]
    return build_rir_job_plan(
        bank,
        listener_position_m=listener["position_m"],
        listener_orientation_wxyz=listener["orientation_wxyz"],
        stride_frames=1,
    )


def _render_semantic_cached_audio(
    *,
    episode: Mapping[str, Any],
    dry_root: Path,
    acoustic_manifest: Path,
    simulation_request: Path,
    hrtf: Path,
    rir_batch_size: int,
    output: Path,
) -> dict[str, Any]:
    """Render one fresh full75 RLR cache, then assemble the M6z audio files."""

    plan = _build_semantic_rir_job_plan(episode)
    plan_path = output / "rir_job_plan.json"
    _write(plan_path, plan)
    request = _load(simulation_request)
    simulation = M4SimulationConfig.from_mapping(request["simulation"])
    scene = load_semantic_acoustic_scene(acoustic_manifest)
    cache = render_semantic_rir_cache(
        plan_path=plan_path,
        scene=scene,
        simulation_request_path=simulation_request,
        simulation=simulation,
        output=output / "rir_cache",
        layout_type="binaural",
        hrtf_file_path=hrtf,
        batch_size=rir_batch_size,
        coordinate_translation_m=(0.0, 0.0, 0.0),
    )
    cached = SemanticRIRCacheSession(
        cache_root=cache.output,
        plan_path=plan_path,
        expected_episode_id=SEMANTIC_RIR_EPISODE_ID,
        frame_count=FRAME_COUNT,
        frame_rate_hz=FPS,
    ).load_episode(SEMANTIC_RIR_EPISODE_ID)

    dry_paths = {
        DOG_SOURCE_ID: dry_root / "m6x_dog0_muzzle.wav",
        HUMAN_SOURCE_ID: dry_root / "m6x_human0_mouth.wav",
    }
    dry: dict[str, np.ndarray] = {}
    for source_id, dry_path in dry_paths.items():
        value = read_float32_wav(dry_path)
        if value.sample_rate_hz != 16_000 or value.samples.shape != (1, 80_000):
            raise RuntimeError(f"unexpected dry bus contract: {dry_path}")
        dry[source_id] = np.asarray(value.samples[0], dtype=np.float64)
    dry_by_slot = {
        slot: dry[source_id]
        for source_id, slot in SEMANTIC_SOURCE_SLOT_BY_SOURCE_ID.items()
    }
    stems, _ = render_asset_bound_binaural(
        dry_by_slot,
        rir_samples=cached.samples,
        rir_lengths=cached.lengths,
        source_ids=cached.source_slot_ids,
        keyframe_samples=cached.keyframe_samples,
    )
    stored_stems, mixture = float32_stems_and_exact_mix(
        stems, source_ids=cached.source_slot_ids
    )
    peak = float(np.max(np.abs(mixture)))
    if not 1.0e-8 < peak < 1.0:
        raise RuntimeError(
            f"binaural preview must be audible and unclipped, peak={peak}"
        )

    audio_dir = output / "audio"
    for source_id, source_slot_id in SEMANTIC_SOURCE_SLOT_BY_SOURCE_ID.items():
        write_float32_wav(
            audio_dir / f"{source_id}_stem.wav",
            stored_stems[source_slot_id],
            16_000,
            metadata={
                "role": "residential_semantic_cached_rlr_binaural_stem",
                "source_id": source_id,
                "source_slot_id": source_slot_id,
                "qualification_claim": False,
            },
        )
    mixture_path = audio_dir / "mixture.wav"
    write_float32_wav(
        mixture_path,
        mixture,
        16_000,
        metadata={
            "role": "residential_semantic_cached_rlr_binaural_mixture",
            "source_slots": list(cached.source_slot_ids),
            "qualification_claim": False,
        },
    )
    evidence = {
        "status": "pass",
        "audio_mode": "semantic_cached_rlr",
        "source_ids": [DOG_SOURCE_ID, HUMAN_SOURCE_ID],
        "source_slots_by_source_id": dict(SEMANTIC_SOURCE_SLOT_BY_SOURCE_ID),
        "layout": "binaural_left_right",
        "sample_rate_hz": 16_000,
        "sample_count": 80_000,
        "rir_keyframe_count": len(cached.visual_frame_indices),
        "rir_stride_frames": 1,
        "requested_pair_state_count": plan["requested_pair_state_count"],
        "unique_rir_job_count": plan["unique_rir_job_count"],
        "mixture_peak": peak,
        "mixture": str(mixture_path),
        "rir_job_plan": str(plan_path),
        "rir_cache": {
            "path": str(cache.output),
            "native_execution": cache.receipt.get("native_execution"),
            "native_realized_job_count": cache.receipt.get("native_realized_job_count"),
            "native_simulate_owned_call_count": cache.receipt.get(
                "native_simulate_owned_call_count"
            ),
        },
        "acoustic_scene": {
            "manifest_path": str(scene.manifest_path),
            "package_id": scene.package_id,
        },
        "simulation_request": {
            "path": str(simulation_request.resolve()),
            "request_id": request.get("request_id"),
        },
        "coordinate_translation_habitat_m": [0.0, 0.0, 0.0],
        "claim_boundary": (
            "native CPU RLR RIRs use the selected InteriorAgent/Kujiale USD "
            "geometry and declared SoundSpaces-style semantic material constraints; "
            "research candidate only, not a physical material calibration claim"
        ),
    }
    _write(output / "audio_evidence.json", evidence)
    return evidence


def _probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate,nb_read_frames,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    episode = build_residential_source_episode(
        scene_metadata=_load(args.scene_metadata), profile=_load(args.profile)
    )
    records = {
        "timeline.json": episode["timeline"],
        "source_manifest.json": episode["source_manifest"],
        "flags.json": episode["flags"],
        "room_capsule.json": episode["room_capsule"],
        "qualification.json": episode["qualification"],
        "visual_plan.json": episode["visual_plan"],
        "episode_plan.json": episode,
    }
    for name, value in records.items():
        _write(output / name, value)
    topdown = _render_topdown_frames(episode, output)
    if args.audio_mode == "review_proxy":
        audio = _render_audio(
            episode=episode,
            dry_root=args.dry_root,
            acoustic_manifest=args.acoustic_manifest,
            simulation_request=args.simulation_request,
            hrtf=args.hrtf,
            output=output,
        )
    elif args.audio_mode == "semantic_cached_rlr":
        semantic_inputs = {
            "--semantic-acoustic-manifest": args.semantic_acoustic_manifest,
            "--semantic-simulation-request": args.semantic_simulation_request,
            "--semantic-hrtf": args.semantic_hrtf,
        }
        missing = [name for name, value in semantic_inputs.items() if value is None]
        if missing:
            raise RuntimeError("semantic_cached_rlr requires " + ", ".join(missing))
        audio = _render_semantic_cached_audio(
            episode=episode,
            dry_root=args.dry_root,
            acoustic_manifest=args.semantic_acoustic_manifest,
            simulation_request=args.semantic_simulation_request,
            hrtf=args.semantic_hrtf,
            rir_batch_size=args.semantic_rir_batch_size,
            output=output,
        )
    else:
        raise RuntimeError(f"unsupported audio mode: {args.audio_mode}")
    evidence = {
        "status": "pass",
        "schema": "avengine_optional_residential_episode_build_evidence_v1",
        "backend_role": episode["visual_plan"]["backend_role"],
        "scene": episode["scene"],
        "clock": episode["clock"],
        "route_speeds_mps": {
            actor_id: value["mean_speed_mps"]
            for actor_id, value in episode["route_metrics"].items()
        },
        "source_center_gate": "pass",
        "simultaneous_source_audio": True,
        "topdown": str(topdown),
        "topdown_probe": _probe(topdown),
        "audio": audio,
        "next_stage": (
            f"SPEAR_UE_{episode['visual_plan']['backend_role']}_render_and_mux"
        ),
    }
    _write(output / "evidence.json", evidence)
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-metadata", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--dry-root", type=Path, required=True)
    parser.add_argument(
        "--acoustic-manifest",
        type=Path,
        default=REPOSITORY
        / "tmp/m3/formal_20260717_01/compile/low_absorption/manifest.json",
    )
    parser.add_argument(
        "--simulation-request",
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
        "--audio-mode",
        choices=("review_proxy", "semantic_cached_rlr"),
        default="review_proxy",
    )
    parser.add_argument("--semantic-acoustic-manifest", type=Path)
    parser.add_argument("--semantic-simulation-request", type=Path)
    parser.add_argument("--semantic-hrtf", type=Path)
    parser.add_argument(
        "--semantic-rir-batch-size",
        type=int,
        default=8,
        help="native RLR sources per semantic cache batch",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    result = build(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

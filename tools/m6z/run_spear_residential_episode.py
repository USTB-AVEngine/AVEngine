#!/usr/bin/env python3
"""Render one AVEngine residential human+Beagle episode through SPEAR/UE."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import cv2


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "tools/m6y"))

from avengine.optional_backends.spear_apartment import (  # noqa: E402
    ANIMATION_TOLERANCE_SECONDS,
    FRAME_COUNT,
    FPS,
    HEIGHT,
    WIDTH,
    build_png_encode_command,
    summarize_actor_bounds,
    summarize_root_readbacks,
)
from run_spear_apartment_canary import (  # noqa: E402
    _actor_bounds_readback,
    _actor_readback,
    _apply_actor_state,
    _apply_camera,
    _destroy_runtime_actors,
    _read_frame,
    _spawn_camera,
    _spawn_runtime_actors,
)
from run_spear_kujiale_canary import (  # noqa: E402
    _configure_spear,
    _spawn_review_lights,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _light_plan(episode: Mapping[str, Any]) -> dict[str, Any]:
    lights = []
    for raw in episode.get("review_lights", []):
        position = raw["position_xyz_m"]
        lights.append(
            {
                "light_id": raw["light_id"],
                "position_m": list(position),
                "position_ue_cm": [100.0 * float(item) for item in position],
                "intensity_lumens": float(raw["intensity_lumens"]),
                "attenuation_radius_cm": 100.0 * float(raw["attenuation_radius_m"]),
                "temperature_kelvin": float(raw["temperature_kelvin"]),
                "source_radius_cm": 100.0 * float(raw.get("source_radius_m", 0.0)),
                "soft_source_radius_cm": 100.0 * float(raw.get("soft_source_radius_m", 0.0)),
            }
        )
    return {"review_lights": lights}


def _probe(
    path: Path, *, width: int, height: int, expect_audio: bool
) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate,nb_read_frames,sample_rate,channels:format=duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    value = json.loads(result.stdout)
    video = [item for item in value["streams"] if item["codec_type"] == "video"]
    audio = [item for item in value["streams"] if item["codec_type"] == "audio"]
    if len(video) != 1 or len(audio) != int(expect_audio):
        raise RuntimeError(f"media stream closure failed: {path}")
    v = video[0]
    if (
        int(v["width"]) != width
        or int(v["height"]) != height
        or v["avg_frame_rate"] != f"{FPS}/1"
        or int(v["nb_read_frames"]) != FRAME_COUNT
    ):
        raise RuntimeError(f"video readback failed: {v}")
    if expect_audio and (
        int(audio[0]["channels"]) != 2 or int(audio[0]["sample_rate"]) != 16_000
    ):
        raise RuntimeError(f"audio readback failed: {audio[0]}")
    duration = float(value["format"]["duration"])
    if not math.isfinite(duration) or abs(duration - 5.0) > 1.0 / FPS:
        raise RuntimeError(f"duration readback failed: {duration}")
    return {
        "status": "pass",
        "path": str(path),
        "width": width,
        "height": height,
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FPS,
        "duration_seconds": duration,
        "audio": "binaural_left_right" if expect_audio else None,
    }


def _mux_clean(video: Path, audio: Path, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video), "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0", "-frames:v", str(FRAME_COUNT),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(output),
        ],
        check=True,
    )


def _mux_topdown(video: Path, topdown: Path, audio: Path, output: Path) -> None:
    graph = (
        "[0:v]scale=640:360:flags=lanczos,pad=640:480:0:60:color=black[ue];"
        "[1:v]scale=640:480:flags=lanczos[top];"
        "[ue][top]hstack=inputs=2[video]"
    )
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video), "-i", str(topdown), "-i", str(audio),
            "-filter_complex", graph, "-map", "[video]", "-map", "2:a:0",
            "-frames:v", str(FRAME_COUNT), "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(output),
        ],
        check=True,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    episode_root = args.episode_root.expanduser().resolve()
    episode = _load(episode_root / "episode_plan.json")
    plan = episode["visual_plan"]
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    frames_dir = output / "frames"
    frames_dir.mkdir()
    _write(output / "visual_plan.json", plan)

    config_args = argparse.Namespace(**vars(args))
    config_plan = {"map_path": episode["scene"]["map_path"]}
    instance = _configure_spear(config_args, config_plan)
    game = instance.get_game()
    runtimes: dict[str, dict[str, Any]] = {}
    light_records: list[dict[str, Any]] = []
    stage_actor_count = 0
    actor_readbacks = {"dog0": [], "human0": []}
    animation_readbacks = {"dog0": [], "human0": []}
    actor_bounds = {"dog0": [], "human0": []}
    camera_readbacks = []
    try:
        with instance.begin_frame():
            camera, capture = _spawn_camera(game)
            capture.set_property_value(
                property_name="FOVAngle",
                property_value=float(plan["camera"]["horizontal_fov_deg"]),
            )
            observed_fov = float(capture.get_property_value(property_name="FOVAngle"))
            if abs(observed_fov - float(plan["camera"]["horizontal_fov_deg"])) > 1.0e-4:
                raise RuntimeError(f"camera HFOV readback failed: {observed_fov}")
            _apply_camera(camera, plan["camera"])
            runtimes = _spawn_runtime_actors(
                game, {"plan": plan}, args.spear_root.expanduser().resolve()
            )
            for state in plan["frames"][0]["actor_states"]:
                _apply_actor_state(runtimes[state["actor_id"]], state, 0)
            light_records = _spawn_review_lights(game, _light_plan(episode))
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(bPaused=False)
        with instance.end_frame():
            pass
        instance.step(num_frames=args.streaming_warmup_frames)

        with instance.begin_frame():
            stage_actor_count = len(
                game.unreal_service.find_actors_by_class(uclass="AUsdStageActor")
            )
        with instance.end_frame():
            pass
        if stage_actor_count != args.expected_stage_actor_count:
            raise RuntimeError(
                f"expected {args.expected_stage_actor_count} UsdStageActor(s), got {stage_actor_count}"
            )

        for frame_index, frame in enumerate(plan["frames"]):
            with instance.begin_frame():
                for state in frame["actor_states"]:
                    actor_id = state["actor_id"]
                    root, animation = _apply_actor_state(
                        runtimes[actor_id], state, frame_index
                    )
                    actor_readbacks[actor_id].append(root)
                    animation_readbacks[actor_id].append(animation)
                _apply_camera(camera, plan["camera"])
                camera_readbacks.append(_actor_readback(camera, frame_index))
            with instance.end_frame():
                image = _read_frame(capture).copy()
                for actor_id, runtime in runtimes.items():
                    actor_bounds[actor_id].append(
                        _actor_bounds_readback(runtime["visual_actor"], frame_index)
                    )
            frame_path = frames_dir / f"frame_{frame_index:04d}.png"
            if image.shape[:2] != (HEIGHT, WIDTH) or not cv2.imwrite(str(frame_path), image):
                raise RuntimeError(f"could not write frame: {frame_path}")
            if frame_index % FPS == 0:
                print(f"[residential:{episode['scene']['scene_id']}] frame {frame_index:02d}/74", flush=True)
    finally:
        if runtimes:
            try:
                _destroy_runtime_actors(instance, runtimes)
            except Exception as exc:
                print(f"warning: actor cleanup failed: {exc}", file=sys.stderr)
        instance.close(force=True)

    root_gate = summarize_root_readbacks(
        expected_frames=plan["frames"],
        actor_readbacks=actor_readbacks,
        camera_readbacks=camera_readbacks,
        camera_position_cm=plan["camera"]["ue_position_cm"],
        camera_yaw_deg=plan["camera"]["ue_yaw_deg"],
    )
    bounds_gate = summarize_actor_bounds(
        expected_frames=plan["frames"],
        actor_declarations=plan["actors"],
        actor_bounds=actor_bounds,
    )
    animation_gate = {}
    for actor_id, records in animation_readbacks.items():
        maximum = max(item["absolute_error_seconds"] for item in records)
        if maximum > ANIMATION_TOLERANCE_SECONDS:
            raise RuntimeError(f"{actor_id} animation phase readback failed")
        animation_gate[actor_id] = {
            "status": "pass",
            "action_ids": sorted({item["action_id"] for item in records}),
            "maximum_absolute_error_seconds": maximum,
        }

    visual = output / "ue_visual_only.mp4"
    subprocess.run(
        build_png_encode_command(
            frames_pattern=frames_dir / "frame_%04d.png", output_path=visual
        ),
        check=True,
    )
    audio = episode_root / "audio/mixture.wav"
    topdown = episode_root / "topdown_only.mp4"
    clean = output / "ue_clean_binaural.mp4"
    combined = output / "ue_topdown_binaural.mp4"
    _mux_clean(visual, audio, clean)
    _mux_topdown(visual, topdown, audio, combined)
    media = {
        "ue_visual_only": _probe(visual, width=1280, height=720, expect_audio=False),
        "ue_clean_binaural": _probe(clean, width=1280, height=720, expect_audio=True),
        "ue_topdown_binaural": _probe(combined, width=1280, height=480, expect_audio=True),
    }
    if not args.keep_frames:
        shutil.rmtree(frames_dir)
    evidence = {
        "schema": "avengine_optional_spear_residential_episode_evidence_v1",
        "status": "pass",
        "backend_role": "comparison_visual",
        "scene": episode["scene"],
        "stage_actor_count": stage_actor_count,
        "runtime_review_lights": light_records,
        "root_readback": root_gate,
        "animation_phase_readback": animation_gate,
        "visual_bounds_readback": bounds_gate,
        "media": media,
        "authority": {
            "ue_pixels": "optional room comparison visual",
            "timeline_source_logic_audio_topdown_metadata": "AVEngine",
            "backend_replanned_route": False,
            "audio_camera_fov_cutoff": False,
            "source_center_gate": "center_only_not_body_volume",
        },
        "audio_claim_boundary": episode["acoustic_proxy"],
    }
    _write(output / "evidence.json", evidence)
    print(combined, flush=True)
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--spear-root", type=Path, required=True)
    parser.add_argument("--uproject", type=Path, required=True)
    parser.add_argument("--unreal-editor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rpc-port", type=int, default=39379)
    parser.add_argument("--graphics-adapter", type=int, default=0)
    parser.add_argument("--streaming-warmup-frames", type=int, default=180)
    parser.add_argument("--expected-stage-actor-count", type=int, default=1)
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

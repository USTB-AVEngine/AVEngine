#!/usr/bin/env python3
"""Render one HM3D moving-source episode: pose, FOA, first-person video, binaural.

This chains the four single-purpose tools in the one order their contract
allows, because the pose file is a shared decision rather than a parameter.
``choose_listener_pose`` ranks candidate listener poses; the ambisonic audio
pass auditions them in order and writes ``accepted_index`` back into the pose
file; the video pass then renders the entry the audio pass accepted; and the
binaural pass reuses the accepted listener and takes its head orientation from
the video's own camera manifest. Run out of order, the picture and the sound
answer two different questions about the same episode.

The chain deliberately spans two interpreters. The pose and video passes run
under this process's own python, which activates AVEngine's installed Habitat
prefix from the ``--runtime-*`` arguments. The audio passes import a
habitat_sim built with the SoundSpaces audio sensor, which lives in a separate
environment - ``--audio-python`` names its interpreter. Collapsing the two
into one environment is exactly the confusion that made the visual runtime
look broken once before.

Every step's artifacts stay in its own subdirectory, and the receipt names the
exact files downstream review should read. The final deliverable is a muxed
first-person mp4 with binaural audio - the thing a human auditions - alongside
the FOA wav that training consumes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def run(step: str, argv: list[str], log_dir: Path) -> None:
    """Run one stage, teeing its output to a per-step log.

    The child's stdout is replayed onto ours so the Studio task log stays one
    chronological story, but it is also kept per step: when a five-minute chain
    fails at step four, the step's own log is the thing to read.
    """

    log_path = log_dir / f"{step}.log"
    print(f"=== {step}: {' '.join(argv)}", flush=True)
    with log_path.open("wb") as log_file:
        completed = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
        )
        log_file.write(completed.stdout)
    sys.stdout.buffer.write(completed.stdout)
    sys.stdout.flush()
    if completed.returncode != 0:
        raise SystemExit(f"{step} failed with exit code {completed.returncode}")


def the_only(pattern: str, directory: Path) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one {pattern} in {directory}, found "
            f"{[m.name for m in matches]}"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-prefix", required=True)
    parser.add_argument("--magnum-site", required=True)
    parser.add_argument("--rlr-sdk-root", required=True)
    parser.add_argument(
        "--audio-python",
        required=True,
        type=Path,
        help="interpreter of the SoundSpaces-audio habitat environment",
    )
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--materials-json", required=True, type=Path)
    parser.add_argument("--hrtf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--episode-id")
    parser.add_argument("--motion-case", default="source1_moving_source2_static")
    parser.add_argument("--slot", default="source1")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--listener-height-m", type=float, default=1.5)
    parser.add_argument("--minimum-range-m", type=float, default=2.0)
    parser.add_argument("--maximum-range-m", type=float, default=6.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--aim-open", action="store_true")
    parser.add_argument("--overhead-m", type=float)
    parser.add_argument("--place-at-emitter", action="store_true")
    parser.add_argument("--frame-rate-hz", type=float, default=15.0)
    args = parser.parse_args()

    bank = args.bank.resolve()
    if bank.is_dir():
        # A route-bank task writes one bank file per detected floor. A single
        # file is unambiguous and accepted; several floors are a real choice
        # the caller has to make, so the error names them instead of picking.
        candidates = sorted(bank.glob("*.bank.json"))
        if len(candidates) == 1:
            bank = candidates[0]
        else:
            names = [candidate.name for candidate in candidates]
            raise SystemExit(
                f"--bank {bank} is a directory holding {len(candidates)} bank "
                f"files {names}; name the floor's file explicitly"
            )
    args.bank = bank

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists (fresh/no-clobber): {output}")
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()

    python = sys.executable
    audio_python = str(args.audio_python)
    runtime = [
        "--runtime-prefix", args.runtime_prefix,
        "--magnum-site", args.magnum_site,
        "--rlr-sdk-root", args.rlr_sdk_root,
    ]
    episode = ["--episode-index", str(args.episode_index)]
    if args.episode_id:
        episode = ["--episode-id", args.episode_id]
    episode += ["--motion-case", args.motion_case, "--slot", args.slot]

    pose_dir = output / "listener_pose"
    pose_dir.mkdir()
    pose = pose_dir / "pose.json"
    run(
        "choose_listener_pose",
        [
            python, str(REPOSITORY / "tools/scene/choose_listener_pose.py"),
            *runtime, *episode,
            "--bank", str(args.bank),
            "--output", str(pose),
            "--listener-height-m", str(args.listener_height_m),
            "--minimum-range-m", str(args.minimum_range_m),
            "--maximum-range-m", str(args.maximum_range_m),
            "--seed", str(args.seed),
        ],
        logs,
    )

    foa_dir = output / "audio_foa"
    run(
        "render_foa",
        [
            audio_python, str(REPOSITORY / "tools/audio/render_moving_source.py"),
            *episode,
            "--bank", str(args.bank),
            "--output-dir", str(foa_dir),
            "--layout", "ambisonics",
            "--listener-pose", str(pose),
            "--dataset-config", str(args.dataset_config),
            "--scene-id", args.scene_id,
            "--materials-json", str(args.materials_json),
            "--frame-stride", str(args.frame_stride),
            "--seed", str(args.seed),
        ],
        logs,
    )
    foa_report = foa_dir / "render_report.json"
    if not foa_report.is_file():
        raise SystemExit("the ambisonic pass wrote no render_report.json")

    video_dir = output / "video"
    video_argv = [
        python, str(REPOSITORY / "tools/visual/render_moving_source_video.py"),
        *runtime,
        "--bank", str(args.bank),
        "--acoustic-report", str(foa_report),
        "--listener-pose", str(pose),
        "--asset-dir", str(args.asset_dir),
        "--output-dir", str(video_dir),
        "--width", str(args.width),
        "--height", str(args.height),
    ]
    if args.aim_open:
        video_argv.append("--aim-open")
    if args.overhead_m is not None:
        video_argv += ["--overhead-m", str(args.overhead_m)]
    if args.place_at_emitter:
        video_argv.append("--place-at-emitter")
    run("render_video", video_argv, logs)
    video_manifest = video_dir / "video_manifest.json"
    if not video_manifest.is_file():
        raise SystemExit("the video pass wrote no video_manifest.json")

    binaural_dir = output / "audio_binaural"
    run(
        "render_binaural",
        [
            audio_python, str(REPOSITORY / "tools/audio/render_moving_source.py"),
            *episode,
            "--bank", str(args.bank),
            "--output-dir", str(binaural_dir),
            "--layout", "binaural",
            "--from-report", str(foa_report),
            "--video-manifest", str(video_manifest),
            "--hrtf", str(args.hrtf),
            "--dataset-config", str(args.dataset_config),
            "--scene-id", args.scene_id,
            "--materials-json", str(args.materials_json),
            "--frame-stride", str(args.frame_stride),
            "--seed", str(args.seed),
        ],
        logs,
    )

    foa_wav = the_only("moving_source.ambisonic.wav", foa_dir)
    binaural_wav = the_only("moving_source.binaural.wav", binaural_dir)
    first_frame = the_only("frame_0000.png", video_dir)

    manifest = json.loads(video_manifest.read_text(encoding="utf-8"))
    frame_rate = float(
        manifest.get("frame_rate_hz")
        or manifest.get("frame_rate")
        or args.frame_rate_hz
    )
    deliverable = output / "episode_binaural.mp4"
    run(
        "mux",
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-framerate", str(frame_rate),
            "-i", str(video_dir / "frame_%04d.png"),
            "-i", str(binaural_wav),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(deliverable),
        ],
        logs,
    )

    receipt = {
        "schema": "avengine_hm3d_episode_receipt_v1",
        "bank": str(args.bank.resolve()),
        "scene_id": args.scene_id,
        "motion_case": args.motion_case,
        "slot": args.slot,
        "episode_index": args.episode_index,
        "episode_id": args.episode_id,
        "listener_pose": str(pose),
        "foa_report": str(foa_report),
        "foa_wav": str(foa_wav),
        "binaural_wav": str(binaural_wav),
        "video_manifest": str(video_manifest),
        "first_frame": str(first_frame),
        "deliverable_mp4": str(deliverable),
        "frame_rate_hz": frame_rate,
        "acceptance_note": (
            "acceptance is the per-frame error_deg records inside foa_report "
            "and the human verdict on deliverable_mp4, never this chain's "
            "exit code"
        ),
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"receipt": str(output / "receipt.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

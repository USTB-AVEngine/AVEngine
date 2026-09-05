#!/usr/bin/env python3
"""Studio end-to-end MP3D chain: author route → capture → dynamic audio → clip.

One queue task drives the engine's own CLI verbs in sequence, each into a
fresh subdirectory of --output. Any step failure fails the whole task
(fail-closed); steps.json records the argv and status of every step.
Research-only throughout: the formal dataset denominator stays 0.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def run_step(name: str, argv: list[str], steps: list[dict], steps_path: Path) -> None:
    record = {"step": name, "argv": [str(item) for item in argv], "status": "running"}
    steps.append(record)
    steps_path.write_text(
        json.dumps({"steps": steps}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[studio-e2e] step {name}: {' '.join(str(item) for item in argv)}", flush=True)
    launch_env = dict(os.environ)
    current_source = str(REPOSITORY / "src")
    existing = launch_env.get("PYTHONPATH", "")
    launch_env["PYTHONPATH"] = (
        current_source + (os.pathsep + existing if existing else "")
    )
    record["cwd"] = str(REPOSITORY)
    record["avengine_source"] = current_source
    completed = subprocess.run(
        [str(item) for item in argv],
        check=False,
        cwd=str(REPOSITORY),
        env=launch_env,
    )
    record["status"] = "pass" if completed.returncode == 0 else "fail"
    record["returncode"] = completed.returncode
    steps_path.write_text(
        json.dumps({"steps": steps}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise SystemExit(f"step {name} failed with returncode {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--camera-selection", choices=("framing", "lateral_sweep"), default="lateral_sweep"
    )
    parser.add_argument("--source-animal-manifest", required=True, type=Path)
    parser.add_argument("--source-m2-request", required=True, type=Path)
    parser.add_argument("--room-manifest", type=Path)
    parser.add_argument("--simulation-request", type=Path)
    parser.add_argument("--package-manifest", type=Path)
    parser.add_argument("--audio-program", type=Path)
    parser.add_argument("--source-endpoint-registry", type=Path)
    parser.add_argument("--sound-asset-registry", type=Path)
    parser.add_argument("--sound-asset-map", type=Path)
    parser.add_argument(
        "--sound-asset-path",
        action="append",
        default=[],
        metavar="SOUND_ID=PATH",
    )
    parser.add_argument("--beagle-audio", type=Path)
    parser.add_argument("--hrtf", type=Path)
    parser.add_argument("--hrtf-license", type=Path)
    parser.add_argument("--runtime-prefix", required=True, type=Path)
    parser.add_argument("--mp3d-root", required=True, type=Path)
    parser.add_argument("--magnum-python-site", required=True, type=Path)
    parser.add_argument("--rlr-sdk-root", required=True, type=Path)
    parser.add_argument("--rir-stride-frames", type=int, default=3)
    parser.add_argument(
        "--layouts",
        default="binaural",
        help="comma-separated output layouts; end-to-end review requires binaural",
    )
    parser.add_argument("--execution-variant")
    parser.add_argument("--variant", default="A")
    parser.add_argument("--author-only", action="store_true",
                        help="stop after route authoring (fast preview for the canvas)")
    parser.add_argument(
        "--review-root",
        type=Path,
        default=Path("/data/avengine_external/review"),
        help="the route author verb requires its output to be an immediate "
        "fresh child of the external review root",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.author_only:
        required_audio = {
            "room_manifest": args.room_manifest,
            "simulation_request": args.simulation_request,
            "package_manifest": args.package_manifest,
            "audio_program": args.audio_program,
            "source_endpoint_registry": args.source_endpoint_registry,
            "sound_asset_registry": args.sound_asset_registry,
        }
        missing = [
            name for name, value in required_audio.items() if value is None
        ]
        if missing:
            parser.error(
                "end-to-end mode requires: " + ", ".join(missing)
            )

    if args.author_only:
        layouts = ("binaural",)
    else:
        layouts = tuple(
            item.strip() for item in args.layouts.split(",") if item.strip()
        )
        if (
            not layouts
            or len(layouts) != len(set(layouts))
            or any(item not in {"binaural", "ambisonics"} for item in layouts)
        ):
            parser.error(
                "--layouts must contain unique binaural/ambisonics values"
            )
        if "binaural" not in layouts:
            parser.error("end-to-end review requires binaural in --layouts")
        if args.hrtf is None:
            parser.error("end-to-end binaural audio requires --hrtf")

    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"output already exists (fresh/no-clobber): {output}")
    output.mkdir(parents=True)
    steps: list[dict] = []
    steps_path = output / "steps.json"
    python = sys.executable
    cli = [python, "-m", "avengine.cli"]
    # the route author verb resolves the RLR SDK via the environment only
    os.environ["AVENGINE_RLR_SDK_ROOT"] = str(args.rlr_sdk_root.resolve())

    # The authored route is a review product; the small route JSONs are then
    # mirrored into the task output so the Studio can serve them as artifacts.
    route_dir = args.review_root.resolve() / f"studio_{output.parent.parent.name}_route"
    run_step(
        "author_route",
        cli
        + [
            "m5",
            "author-current-mp3d-two-beagle-route",
            "--source-animal-manifest", args.source_animal_manifest,
            "--source-m2-request", args.source_m2_request,
            "--runtime-prefix", args.runtime_prefix,
            "--mp3d-root", args.mp3d_root,
            "--magnum-python-site", args.magnum_python_site,
            "--seed", str(args.seed),
            "--camera-selection", args.camera_selection,
            "--output", route_dir,
        ],
        steps,
        steps_path,
    )
    route_mirror = output / "route"
    route_mirror.mkdir(parents=True, exist_ok=True)
    for json_path in sorted(route_dir.glob("*.json")):
        shutil.copy2(json_path, route_mirror / json_path.name)
    if args.author_only:
        print(json.dumps({"status": "pass", "route": str(route_dir)}, ensure_ascii=False))
        return 0

    m1_request = route_dir / "research_m1_request_480p.json"
    if not m1_request.is_file():
        m1_request = route_dir / "research_m1_request.json"
    m2_request = route_dir / "primary_m2_request.json"

    capture_dir = output / "capture"
    run_step(
        "capture_visual",
        cli
        + [
            "m5",
            "capture-current-visual",
            "--animal-manifest", args.source_animal_manifest,
            "--m2-request", m2_request,
            "--room-manifest", args.room_manifest,
            "--m1-request", m1_request,
            "--runtime-prefix", args.runtime_prefix,
            "--mp3d-root", args.mp3d_root,
            "--magnum-python-site", args.magnum_python_site,
            "--rlr-sdk-root", args.rlr_sdk_root,
            "--output", capture_dir,
        ],
        steps,
        steps_path,
    )

    audio_dir = output / "audio"
    audio_argv = cli + [
        "m5",
        "render-current-mp3d-dynamic-audio",
        "--visual-capture-dir", capture_dir,
        "--m1-request", route_dir / "research_m1_request.json",
        "--simulation-request", args.simulation_request,
        "--package-manifest", args.package_manifest,
        "--audio-program", args.audio_program,
        "--source-endpoint-registry", args.source_endpoint_registry,
        "--sound-asset-registry", args.sound_asset_registry,
        "--layouts", ",".join(layouts),
        "--runtime-prefix", args.runtime_prefix,
        "--rlr-sdk-root", args.rlr_sdk_root,
        "--magnum-python-site", args.magnum_python_site,
        "--rir-stride-frames", str(args.rir_stride_frames),
        "--variant", args.variant,
        "--output", audio_dir,
    ]
    if args.hrtf is not None:
        audio_argv += ["--hrtf", args.hrtf]
    if args.sound_asset_map is not None:
        audio_argv += ["--sound-asset-map", args.sound_asset_map]
    for assignment in args.sound_asset_path:
        audio_argv += ["--sound-asset-path", assignment]
    if args.beagle_audio is not None:
        audio_argv += ["--beagle-audio", args.beagle_audio]
    if args.hrtf_license is not None:
        audio_argv += ["--hrtf-license", args.hrtf_license]
    if args.execution_variant is not None:
        audio_argv += ["--execution-variant", args.execution_variant]
    run_step("dynamic_audio", audio_argv, steps, steps_path)

    clip_path = output / "clip" / "mp3d_dynamic_binaural.mp4"
    clip_path.parent.mkdir(parents=True)
    run_step(
        "review_clip",
        [
            python,
            str(REPOSITORY / "tools/review/build_current_mp3d_dynamic_review_clip.py"),
            "--visual-capture-dir", capture_dir,
            "--mixture-wav", audio_dir / "audio" / "binaural" / "mixture.wav",
            "--output", clip_path,
        ],
        steps,
        steps_path,
    )

    print(
        json.dumps(
            {
                "status": "pass",
                "route": str(route_dir),
                "capture": str(capture_dir),
                "audio": str(audio_dir),
                "clip": str(clip_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

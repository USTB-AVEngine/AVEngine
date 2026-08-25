#!/usr/bin/env python3
"""Studio end-to-end Apartment chain: author timeline → UE capture → audio → clip.

Drives the engine's own CLI verbs and tools in sequence. The camera must
stay at the fixed-apartment M1 review pose: the dynamic-audio step
cross-checks the capture camera against the M1 listener authority at 1e-6
and fails closed on any drift, so only the four actor endpoints are freely
draggable in v1. Research-only; the formal dataset denominator stays 0.
"""

from __future__ import annotations

import argparse
import json
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
    print(f"[studio-e2e] step {name}", flush=True)
    completed = subprocess.run([str(item) for item in argv], check=False)
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
    parser.add_argument("--actor-selection", required=True, type=Path)
    parser.add_argument("--source-asset-registry", required=True, type=Path)
    parser.add_argument("--camera-position-ue-cm", nargs=3, type=float, required=True)
    parser.add_argument("--camera-yaw-deg", type=float, required=True)
    parser.add_argument("--human-start-ue-cm", nargs=3, type=float, required=True)
    parser.add_argument("--human-end-ue-cm", nargs=3, type=float, required=True)
    parser.add_argument("--beagle-start-ue-cm", nargs=3, type=float, required=True)
    parser.add_argument("--beagle-end-ue-cm", nargs=3, type=float, required=True)
    parser.add_argument("--closure-report", required=True, type=Path)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--spear-executable", required=True, type=Path)
    parser.add_argument("--rpc-port", type=int, default=39511)
    parser.add_argument("--graphics-adapter", type=int)
    parser.add_argument(
        "--spear-ext-dir",
        type=Path,
        help="directory holding the external avengine_spear_ext native "
        "extension build (added to PYTHONPATH for the UE capture step)",
    )
    parser.add_argument("--m1-request", required=True, type=Path)
    parser.add_argument("--simulation-request", required=True, type=Path)
    parser.add_argument("--package-manifest", required=True, type=Path)
    parser.add_argument("--audio-program", required=True, type=Path)
    parser.add_argument("--source-endpoint-registry", required=True, type=Path)
    parser.add_argument("--sound-asset-registry", required=True, type=Path)
    parser.add_argument("--beagle-audio", required=True, type=Path)
    parser.add_argument("--hrtf", required=True, type=Path)
    parser.add_argument("--hrtf-license", type=Path)
    parser.add_argument("--runtime-prefix", required=True, type=Path)
    parser.add_argument("--rlr-sdk-root", required=True, type=Path)
    parser.add_argument("--magnum-python-site", required=True, type=Path)
    parser.add_argument("--rir-stride-frames", type=int, default=3)
    parser.add_argument("--variant", default="A")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    steps: list[dict] = []
    steps_path = output / "steps.json"
    python = sys.executable
    cli = [python, "-m", "avengine.cli"]
    if args.spear_ext_dir is not None:
        import os

        ext_dir = str(args.spear_ext_dir.resolve())
        existing = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = (
            ext_dir + (os.pathsep + existing if existing else "")
        )

    def cm(values: list[float]) -> list[str]:
        return [str(value) for value in values]

    # the author verb writes the timeline JSON file at --output itself
    timeline_path = output / "timeline.json"
    run_step(
        "author_timeline",
        cli
        + [
            "m5", "author-current-apartment-visual-timeline",
            "--actor-selection", args.actor_selection,
            "--source-asset-registry", args.source_asset_registry,
            "--camera-position-ue-cm", *cm(args.camera_position_ue_cm),
            "--camera-yaw-deg", str(args.camera_yaw_deg),
            "--human-start-ue-cm", *cm(args.human_start_ue_cm),
            "--human-end-ue-cm", *cm(args.human_end_ue_cm),
            "--beagle-start-ue-cm", *cm(args.beagle_start_ue_cm),
            "--beagle-end-ue-cm", *cm(args.beagle_end_ue_cm),
            "--output", timeline_path,
        ],
        steps,
        steps_path,
    )

    capture_dir = output / "capture"
    capture_argv = cli + [
        "m5", "capture-current-apartment-visual",
        "--actor-selection", args.actor_selection,
        "--source-asset-registry", args.source_asset_registry,
        "--timeline", timeline_path,
        "--closure-report", args.closure_report,
        "--stage-root", args.stage_root,
        "--spear-executable", args.spear_executable,
        "--rpc-port", str(args.rpc_port),
        "--output", capture_dir,
    ]
    if args.graphics_adapter is not None:
        capture_argv += ["--graphics-adapter", str(args.graphics_adapter)]
    run_step("capture_visual", capture_argv, steps, steps_path)

    audio_dir = output / "audio"
    audio_argv = [
        python,
        str(REPOSITORY / "tools/m7/render_current_apartment_dynamic_audio.py"),
        "--visual-capture-dir", capture_dir,
        "--m1-request", args.m1_request,
        "--simulation-request", args.simulation_request,
        "--package-manifest", args.package_manifest,
        "--audio-program", args.audio_program,
        "--source-endpoint-registry", args.source_endpoint_registry,
        "--sound-asset-registry", args.sound_asset_registry,
        "--beagle-audio", args.beagle_audio,
        "--hrtf", args.hrtf,
        "--runtime-prefix", args.runtime_prefix,
        "--rlr-sdk-root", args.rlr_sdk_root,
        "--magnum-python-site", args.magnum_python_site,
        "--rir-stride-frames", str(args.rir_stride_frames),
        "--variant", args.variant,
        "--output", audio_dir,
    ]
    if args.hrtf_license is not None:
        audio_argv += ["--hrtf-license", args.hrtf_license]
    run_step("dynamic_audio", audio_argv, steps, steps_path)

    clip_path = output / "clip" / "apartment_dynamic_binaural.mp4"
    clip_path.parent.mkdir(parents=True)
    run_step(
        "review_clip",
        [
            python,
            str(REPOSITORY / "tools/review/build_current_mp3d_dynamic_review_clip.py"),
            "--visual-capture-dir", capture_dir,
            "--mixture-wav", audio_dir / "audio" / "binaural" / "mixture.wav",
            "--channel-order", "bgr",
            "--output", clip_path,
        ],
        steps,
        steps_path,
    )

    print(
        json.dumps(
            {
                "status": "pass",
                "timeline": str(timeline_path),
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

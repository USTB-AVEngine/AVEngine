#!/usr/bin/env python3
"""Plan and render one static-seated furnished room as a Studio research task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.rooms.furnished_episode import plan_furnished_residential_episode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--pose-bindings", type=Path, required=True)
    parser.add_argument("--pose-request", type=Path)
    parser.add_argument("--activity", required=True)
    parser.add_argument("--map-path", required=True)
    parser.add_argument("--camera-source-plan", type=Path)
    parser.add_argument("--seat-count", type=int, default=4)
    parser.add_argument("--actor-count", type=int, default=4)
    parser.add_argument("--frame-count", type=int, default=75)
    parser.add_argument("--frame-rate-hz", type=float, default=15.0)
    parser.add_argument("--sample-rate-hz", type=int, default=16_000)
    parser.add_argument("--grid-step-m", type=float, default=2.0)
    parser.add_argument("--camera-height-m", type=float, default=1.55)
    parser.add_argument("--spear-ext-dir", type=Path)
    parser.add_argument("--uproject", type=Path, required=True)
    parser.add_argument("--unreal-editor", type=Path, required=True)
    parser.add_argument("--rpc-port", type=int, default=39379)
    parser.add_argument("--graphics-adapter", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--exposure-bias-ev", type=float)
    parser.add_argument("--streaming-warmup-frames", type=int, default=180)
    parser.add_argument("--native-multimodal", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _runner_command(
    args: argparse.Namespace,
    *,
    plan_root: Path,
    capture_root: Path,
) -> list[str]:
    runner = (
        Path(__file__).resolve().parents[1]
        / "rooms"
        / "run_spear_residential_episode.py"
    )
    command = [
        sys.executable,
        str(runner),
        "--episode-root",
        str(plan_root),
        "--uproject",
        str(args.uproject),
        "--unreal-editor",
        str(args.unreal_editor),
        "--output",
        str(capture_root),
        "--rpc-port",
        str(args.rpc_port),
        "--graphics-adapter",
        str(args.graphics_adapter),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--streaming-warmup-frames",
        str(args.streaming_warmup_frames),
        "--visual-only-research",
    ]
    if args.spear_ext_dir is not None:
        command += ["--spear-ext-dir", str(args.spear_ext_dir)]
    if args.exposure_bias_ev is not None:
        command += ["--exposure-bias-ev", str(args.exposure_bias_ev)]
    if args.native_multimodal:
        command.append("--native-multimodal")
    if args.keep_frames:
        command.append("--keep-frames")
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    output.mkdir(parents=True)

    plan_root = output / "plan"
    capture_root = output / "capture"
    plan_furnished_residential_episode(
        room=args.room,
        asset_root=args.asset_root,
        pose_bindings=args.pose_bindings,
        pose_request=args.pose_request,
        output=plan_root,
        activity=args.activity,
        map_path=args.map_path,
        seat_count=args.seat_count,
        actor_count=args.actor_count,
        frame_count=args.frame_count,
        frame_rate_hz=args.frame_rate_hz,
        sample_rate_hz=args.sample_rate_hz,
        grid_step_m=args.grid_step_m,
        camera_height_m=args.camera_height_m,
        camera_source_plan=args.camera_source_plan,
    )
    command = _runner_command(
        args,
        plan_root=plan_root,
        capture_root=capture_root,
    )
    subprocess.run(command, check=True)

    receipt = {
        "status": "research_only",
        "activity": "seated",
        "plan_root": str(plan_root),
        "capture_root": str(capture_root),
        "executor": "tools/rooms/run_spear_residential_episode.py",
        "qualification_claim": False,
    }
    (output / "studio_seated_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

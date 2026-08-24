#!/usr/bin/env python3
"""Capture one MP3D two-actor case with the native visual runner.

The command is intentionally one-case-at-a-time and refuses to start when the
explicitly selected GPU has a compute process.  It never auto-selects a GPU
and never invokes RLR.  Use ``--compare-to`` after a second fresh output to
perform the A/B evidence comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from avengine.m5.current_visual_multi_actor_native import (
    NativeMultiActorVisualError,
    capture_mp3d_region_case,
    compare_capture_outputs,
)


def _gpu_compute_pids(gpu_device_id: int) -> list[int]:
    if (
        isinstance(gpu_device_id, bool)
        or not isinstance(gpu_device_id, int)
        or gpu_device_id < 0
    ):
        raise ValueError("gpu_device_id must be a nonnegative integer")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_device_id}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"cannot establish GPU{gpu_device_id} compute occupancy: {exc}"
        ) from exc
    pids: list[int] = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if value:
            try:
                pids.append(int(value))
            except ValueError as exc:
                raise RuntimeError(
                    f"nvidia-smi returned an invalid GPU{gpu_device_id} PID: {value}"
                ) from exc
    return pids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--room-manifest", type=Path)
    parser.add_argument(
        "--m1-request",
        type=Path,
        help="validated post-hoc camera binding; does not alter actor routes",
    )
    parser.add_argument("--runtime-prefix", type=Path)
    parser.add_argument("--mp3d-root", type=Path)
    parser.add_argument("--magnum-python-site", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rlr-sdk-root", type=Path)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="compare an existing fresh capture against --output; do not capture",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.compare_to is not None:
            result = compare_capture_outputs(args.compare_to, args.output)
        else:
            required = (
                args.room_manifest,
                args.runtime_prefix,
                args.mp3d_root,
                args.magnum_python_site,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "--room-manifest, --runtime-prefix, --mp3d-root and "
                    "--magnum-python-site are required for capture"
                )
            pids = _gpu_compute_pids(args.gpu_device_id)
            if pids:
                raise RuntimeError(
                    f"GPU{args.gpu_device_id} is occupied; refusing native capture "
                    f"and not switching GPUs (compute PIDs: {pids})"
                )
            result = capture_mp3d_region_case(
                case_manifest_path=args.case_manifest,
                room_manifest_path=args.room_manifest,
                m1_request_path=args.m1_request,
                runtime_prefix=args.runtime_prefix,
                mp3d_root=args.mp3d_root,
                magnum_python_site=args.magnum_python_site,
                output_directory=args.output,
                rlr_sdk_root=args.rlr_sdk_root,
                gpu_device_id=args.gpu_device_id,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (NativeMultiActorVisualError, OSError, RuntimeError, ValueError) as exc:
        print(f"capture_mp3d_region_case: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

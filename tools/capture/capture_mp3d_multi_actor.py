#!/usr/bin/env python3
"""Capture one explicit N-actor MP3D case through the installed Habitat runtime.

The case and actor tracks must come from the current CPU planning chain.  The
native output is research-only and records RGB/depth/semantic arrays together
with actor root, joint, and emitter readback.  It does not run RLR audio or
object-ID/target-only capture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from avengine.capture.mp3d_multi_actor import (
    MP3DMultiActorCaptureError,
    capture_mp3d_multi_actor,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--room-manifest", type=Path, required=True)
    parser.add_argument("--m1-request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime-prefix",
        type=Path,
        help="Non-Git installed Habitat runtime prefix",
    )
    parser.add_argument(
        "--mp3d-root",
        type=Path,
        help="External MP3D data root containing scene_datasets",
    )
    parser.add_argument(
        "--magnum-python-site",
        type=Path,
        help="External Corrade/Magnum Python site",
    )
    parser.add_argument("--rlr-sdk-root", type=Path, help="Declared SDK needed by adapter-linked runtime builds")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = capture_mp3d_multi_actor(
            case_manifest_path=args.case_manifest,
            room_manifest_path=args.room_manifest,
            m1_request_path=args.m1_request,
            runtime_prefix=args.runtime_prefix,
            rlr_sdk_root=args.rlr_sdk_root,
            mp3d_root=args.mp3d_root,
            magnum_python_site=args.magnum_python_site,
            output_directory=args.output,
            gpu_device_id=args.gpu_device_id,
        )
    except (MP3DMultiActorCaptureError, ImportError, OSError, RuntimeError, ValueError) as exc:
        _parser().error(str(exc))
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "artifact_role": receipt["artifact_role"],
                "frame_count": receipt["capture"]["frame_count"],
                "actor_count": len(receipt["actors"]),
                "output": str(args.output.resolve()),
                "object_id": receipt["object_id"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build one camera/listener-coherent M1 request at an arbitrary room pose."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from avengine.camera_pose import apply_camera_listener_pose
from avengine.contracts.json_io import load_json, write_json
from avengine.m1.contracts import validate_capture_request, validate_room_manifest


def build_request(args: argparse.Namespace) -> Path:
    source = args.base_request.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output}")
    request = load_json(source)
    room_id = request.get("room_id")
    if args.room_manifest is not None:
        room = load_json(args.room_manifest.resolve())
        room_errors = validate_room_manifest(room)
        if room_errors:
            raise RuntimeError("; ".join(room_errors))
        room_id = room.get("room_id")
    errors = validate_capture_request(request, room_id=room_id)
    if errors:
        raise RuntimeError("; ".join(errors))
    result = apply_camera_listener_pose(
        request,
        request_id=args.request_id,
        position_m=args.position_m,
        yaw_deg=args.yaw_deg,
        horizontal_fov_deg=args.horizontal_fov_deg,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-request", type=Path, required=True)
    parser.add_argument("--room-manifest", type=Path)
    parser.add_argument("--request-id", required=True)
    parser.add_argument(
        "--position-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        required=True,
    )
    parser.add_argument("--yaw-deg", type=float, required=True)
    parser.add_argument("--horizontal-fov-deg", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    output = build_request(parse_args(argv))
    print(f"CAMERA_POSE_REQUEST_OK output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

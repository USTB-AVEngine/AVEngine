#!/usr/bin/env python3
"""Prepare a fixed-camera native four-member cross-time state group."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.dataset.binding_group_motion import prepare_fixed_camera_state_group
from avengine.dataset.binding_group_native import BindingNativeError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-episode-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-static-group-root", type=Path)
    parser.add_argument("--resample-early-audio", action="store_true")
    parser.add_argument("--route-seed-offset", type=int, help="fresh route draw while retaining unchanged static media")
    parser.add_argument("--rpc-port", type=int)
    parser.add_argument("--graphics-adapter", type=int)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--world-id", required=True)
    args = parser.parse_args(argv)
    try:
        summary = prepare_fixed_camera_state_group(
            base_episode_root=args.base_episode_root, output_root=args.output_root,
            group_id=args.group_id, world_id=args.world_id,
            reuse_static_group_root=args.reuse_static_group_root,
            rpc_port=args.rpc_port, graphics_adapter=args.graphics_adapter,
            resample_early_audio=args.resample_early_audio,
            route_seed_offset=args.route_seed_offset)
    except (BindingNativeError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

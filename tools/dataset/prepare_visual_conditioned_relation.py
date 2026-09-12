#!/usr/bin/env python3
"""Prepare one native visual-conditioned relation binding group."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.dataset.binding_group_native import (
    BindingNativeError,
    prepare_visual_conditioned_relation_group,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-request", type=Path, required=True)
    parser.add_argument("--first-visual-capture-root", type=Path, required=True)
    parser.add_argument("--second-visual-capture-root", type=Path)
    parser.add_argument("--sound-pool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-asset-id", action="append", dest="source_asset_ids")
    parser.add_argument("--room-id")
    parser.add_argument("--rpc-port", type=int,
                        help="per-instance port a lease selected; omitted keeps the request")
    parser.add_argument("--graphics-adapter", type=int,
                        help="per-instance device a lease selected; omitted keeps the request")
    parser.add_argument("--group-id", default="visual_conditioned_relation_mp3d_group_v1")
    parser.add_argument("--world-id", default="world_mp3d_relation_0001")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--qa-id", action="append", dest="qa_ids")
    parser.add_argument("--reference-time-s", type=int, default=0)
    parser.add_argument("--appearance-value", action="append", required=True)
    parser.add_argument("--window-start-s", type=int, default=4)
    parser.add_argument("--window-end-s", type=int, default=6)
    parser.add_argument("--start-time-s", action="append", type=float)
    args = parser.parse_args(argv)
    try:
        summary = prepare_visual_conditioned_relation_group(
            base_request_path=args.base_request,
            first_visual_capture_root=args.first_visual_capture_root,
            second_visual_capture_root=args.second_visual_capture_root,
            sound_pool=args.sound_pool,
            output_root=args.output_root,
            source_asset_ids=(
                tuple(args.source_asset_ids)
                if args.source_asset_ids is not None else None
            ),
            room_id=args.room_id,
            rpc_port=args.rpc_port,
            graphics_adapter=args.graphics_adapter,
            group_id=args.group_id,
            world_id=args.world_id,
            qa_ids=tuple(args.qa_ids) if args.qa_ids else None,
            seed=args.seed,
            reference_time_s=args.reference_time_s,
            appearance_values=tuple(args.appearance_value),
            window_s=(args.window_start_s, args.window_end_s),
            start_times_s=(
                tuple(args.start_time_s) if args.start_time_s is not None else None
            ),
        )
    except (BindingNativeError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": summary["status"],
        "group_spec": summary["group_spec"],
        "plan_equivalence": summary["plan_equivalence"],
        "native_readback_equivalence": summary["native_readback_equivalence"],
        "shared_audio_by_column": summary["shared_audio_by_column"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

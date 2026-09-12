#!/usr/bin/env python3
"""Prepare one native HM3D cross-event physical-identity group."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.dataset.binding_group_identity import (
    IdentityNativeError, continue_identity_group_from_plans, prepare_identity_group,
    repair_identity_group_v1, rerender_identity_group_audio, resume_identity_group,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-episode-root", type=Path,
                        help="base episode root for a fresh identity group")
    parser.add_argument("--continue-from-layouts", type=Path,
                        help="reuse an already selected v0/v1 CPU layout root")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--group-id", default="identity_hm3d_group_v1")
    parser.add_argument("--world-id", default="world_hm3d_identity_0001")
    parser.add_argument("--graphics-adapter", type=int)
    parser.add_argument("--rpc-port", type=int)
    parser.add_argument("--resume-from", type=Path,
                        help="finish audio and assembly from an existing visual run")
    parser.add_argument("--repair-v1-from", type=Path,
                        help="re-capture only v1 after a bounded pose repair")
    parser.add_argument("--rerender-audio-from", type=Path,
                        help="freshly render all four audio members from completed captures")
    parser.add_argument("--v1-visual-from", type=Path,
                        help="corrected v1 visual repair root for --rerender-audio-from")
    parser.add_argument("--source-context-policy", choices=("independent_states",),
                        default="independent_states",
                        help="native audio context policy for --rerender-audio-from")
    args = parser.parse_args(argv)
    try:
        selected_modes = sum(value is not None for value in (
            args.continue_from_layouts, args.resume_from, args.repair_v1_from,
            args.rerender_audio_from,
        ))
        if selected_modes > 1:
            parser.error("continue/resume/repair/rerender modes are mutually exclusive")
        if args.continue_from_layouts is not None and args.base_episode_root is not None:
            parser.error("--continue-from-layouts cannot be combined with --base-episode-root")
        if (args.rerender_audio_from is None) != (args.v1_visual_from is None):
            parser.error("--rerender-audio-from requires --v1-visual-from")
        if args.continue_from_layouts is not None:
            summary = continue_identity_group_from_plans(
                layout_root=args.continue_from_layouts,
                request_path=args.request,
                output_root=args.output_root,
                group_id=args.group_id, world_id=args.world_id,
                graphics_adapter=args.graphics_adapter, rpc_port=args.rpc_port,
            )
        elif args.rerender_audio_from is not None:
            summary = rerender_identity_group_audio(
                source_root=args.rerender_audio_from,
                v1_visual_root=args.v1_visual_from,
                output_root=args.output_root,
                group_id=args.group_id, world_id=args.world_id,
                source_context_policy=args.source_context_policy,
            )
        elif args.repair_v1_from is not None:
            summary = repair_identity_group_v1(
                source_root=args.repair_v1_from, output_root=args.output_root,
                group_id=args.group_id, world_id=args.world_id,
            )
        elif args.resume_from is not None:
            summary = resume_identity_group(
                source_root=args.resume_from, output_root=args.output_root,
                group_id=args.group_id, world_id=args.world_id,
            )
        else:
            if args.base_episode_root is None:
                parser.error("--base-episode-root is required for a fresh identity group")
            summary = prepare_identity_group(
                base_episode_root=args.base_episode_root,
                request_path=args.request,
                output_root=args.output_root,
                group_id=args.group_id,
                world_id=args.world_id,
                graphics_adapter=args.graphics_adapter,
                rpc_port=args.rpc_port,
            )
    except (IdentityNativeError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": summary["status"],
        "group_spec": summary["group_spec"],
        "assembled": summary["assembled"],
        "sound_selection": summary.get("sound_selection"),
        "early_probe": summary.get("early_probe"),
        "routes": summary.get("routes"),
        "pcm_by_column": summary["pcm_by_column"],
        "question_truth_by_member": summary["question_truth_by_member"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""重放已渲染 episode 的登记外观复核，按房间给出"登记值 × 判定"混淆表。

对每个渲染根目录下的每个 episode，读回保留的帧、实例掩码与像素可见性真值，
用当前分类器重新判一遍每个出场角色的登记外观值，并额外把同类别的其它登记值
也试一遍，量出误收。只读输入，不重渲、不写进 episode 树。

    python tools/qa/replay_appearance_review.py \
        --render-root /path/to/render_v1 --render-root /path/to/other_render \
        --out /path/to/out_dir --frame-stride 10 --exclude kjl__QA-10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", action="append", required=True, type=Path,
                        help="渲染根目录，可重复给多个")
    parser.add_argument("--out", required=True, type=Path, help="产物目录")
    parser.add_argument("--frame-stride", type=int, default=10,
                        help="每隔多少帧复核一次（交付链用 1，重放默认 10）")
    parser.add_argument("--cross-acceptance-frames", type=int, default=2,
                        help="每个角色做几帧误收探测")
    parser.add_argument("--exclude", action="append", default=[],
                        help="episode 名或路径里出现该子串就跳过")
    parser.add_argument("--minimum-visible-frames", type=int, default=30,
                        help="混淆表只统计可见像素帧不少于该值的角色")
    parser.add_argument("--asset-registry", type=Path, default=None,
                        help="source asset runtime registry，用来解析资产登记的第二色")
    parser.add_argument("--label", default="replay", help="产物文件名前缀")
    args = parser.parse_args(argv)

    from avengine.rooms.appearance_replay import (
        confusion, load_asset_registry, replay_render_roots,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    episodes = []
    lines_path = args.out / f"{args.label}_episodes.jsonl"
    with lines_path.open("w", encoding="utf-8") as handle:
        for episode in replay_render_roots(
            args.render_root,
            frame_stride=args.frame_stride,
            cross_acceptance_frames=args.cross_acceptance_frames,
            exclude=args.exclude,
            asset_registry=load_asset_registry(args.asset_registry),
        ):
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
            handle.flush()
            episodes.append(episode)
            mark = episode.get("skipped") or episode.get("failed") or (
                f"{sum(1 for a in episode.get('actors', {}).values() if a.get('status') == 'reviewed')}"
                f"/{len(episode.get('actors', {}))} reviewed"
            )
            print(f"{episode.get('episode')}: {mark}", flush=True)
    table = confusion(episodes, minimum_visible_frames=args.minimum_visible_frames)
    table["render_roots"] = [str(root) for root in args.render_root]
    table["frame_stride"] = int(args.frame_stride)
    table["excluded"] = list(args.exclude)
    table["asset_registry"] = str(args.asset_registry) if args.asset_registry else None
    table["episode_records"] = str(lines_path)
    summary_path = args.out / f"{args.label}_confusion.json"
    summary_path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(table["cohort"], ensure_ascii=False))
    print(json.dumps(table["by_room"], ensure_ascii=False))
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

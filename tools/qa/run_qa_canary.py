#!/usr/bin/env python3
"""运行 QA Canary 验收并输出证据到磁盘。

用法::

    python tools/qa/run_qa_canary.py --output-dir /tmp/qa_canary_evidence

输出结构::

    <output-dir>/
      canary_evidence.json          # 整体证据清单
      c1_fully_visible/
        episode.json                # 完整 Episode 文档
        qa_results.json             # 问题验证结果
        normal_semantics.npz        # 正常语义分割数组 (N,H,W)
        target_only_semantics.npz   # 目标专用语义数组
        visibility_overlay.npz      # 状态变化叠加
      c2_partial_occlusion/  ...
      c3_fully_occluded/     ...
      c4_out_of_view_enter/  ...
      c5_camera_motion_reappear/ ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from avengine.qa.canary import (
    CANARY_BUILDERS,
    build_all_canaries,
    verify_canary,
    _CANARY_QUESTIONS,
)


def _write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def run_qa_canary(
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """运行全部五个 QA canary 并将证据写入磁盘。

    Args:
        output_directory: 输出目录路径。
        overwrite: 是否覆盖已有目录。

    Returns:
        最终输出目录的 Path。

    Raises:
        FileExistsError: 输出目录已存在且 ``overwrite=False``。
    """
    destination = Path(output_directory).resolve()
    staging = Path(
        temp_dir := os.path.join(
            str(destination.parent),
            f".{destination.name}.staging-{os.getpid()}",
        )
    ).resolve()

    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"输出目录已存在: {destination}\n"
                f"使用 --overwrite 强制覆盖"
            )
        shutil.rmtree(destination)

    try:
        staging.mkdir(parents=True, exist_ok=False)

        # 构建全部 canary
        canaries = build_all_canaries()

        evidence_summary: dict[str, Any] = {
            "schema": "avengine_qa_canary_evidence_v1",
            "canary_count": len(canaries),
            "all_passed": all(c["all_passed"] for c in canaries),
            "canaries": [],
        }

        for canary in canaries:
            canary_id: str = canary["canary_id"]
            doc: dict = canary["episode"]
            qa_results: list[dict] = canary["qa_results"]
            normal_sem: np.ndarray | None = canary.get("normal_semantics")
            target_only_sem: np.ndarray | None = canary.get("target_only_semantics")

            canary_dir = staging / canary_id
            canary_dir.mkdir(parents=True, exist_ok=True)

            # 写入 Episode 文档
            _write_json(canary_dir / "episode.json", doc)

            # 写入 QA 结果
            _write_json(canary_dir / "qa_results.json", {
                "canary_id": canary_id,
                "all_passed": canary["all_passed"],
                "results": qa_results,
            })

            # 写入合成语义分割数组 (NPZ)
            if normal_sem is not None:
                np.savez_compressed(
                    canary_dir / "normal_semantics.npz",
                    normal_semantics=normal_sem,
                )
            if target_only_sem is not None:
                np.savez_compressed(
                    canary_dir / "target_only_semantics.npz",
                    target_only_semantics=target_only_sem,
                )

            # 提取可见性叠加数据
            vis_frames = doc["facts"]["visibility_facts"]["per_frame"]
            visibility_states = [
                f["actor_visibility"].get("dog_01", {}).get("visibility_state", "unknown")
                for f in vis_frames
            ]

            # 可见性状态变化叠加（每帧的可见性状态）
            overlay_info = {
                "description": "每帧目标可见性状态变化",
                "frame_count": len(visibility_states),
                "per_frame_states": visibility_states,
                "events": doc["facts"]["events"],
            }
            _write_json(canary_dir / "visibility_overlay.json", overlay_info)

            evidence_summary["canaries"].append({
                "canary_id": canary_id,
                "all_passed": canary["all_passed"],
                "qa_count": len(qa_results),
                "qa_passed": sum(1 for r in qa_results if r["passed"]),
                "episode_path": f"{canary_id}/episode.json",
                "qa_results_path": f"{canary_id}/qa_results.json",
                "visibility_overlay_path": f"{canary_id}/visibility_overlay.json",
            })

        # 写入整体证据清单
        _write_json(staging / "canary_evidence.json", evidence_summary)

        # 原子重命名
        os.rename(staging, destination)

    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # 打印摘要
    print(json.dumps({
        "status": "pass" if evidence_summary["all_passed"] else "fail",
        "destination": str(destination),
        "canary_count": evidence_summary["canary_count"],
        "all_passed": evidence_summary["all_passed"],
        "canaries": [
            {
                "canary_id": c["canary_id"],
                "passed": c["all_passed"],
                "qa_passed": f"{c['qa_passed']}/{c['qa_count']}",
            }
            for c in evidence_summary["canaries"]
        ],
    }, ensure_ascii=False, indent=2, sort_keys=True))

    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="证据输出目录",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="覆盖已有输出目录",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_qa_canary(args.output_dir, overwrite=args.overwrite)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

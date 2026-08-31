#!/usr/bin/env python3
"""Sequential UE capture runner for a qa-v3 design batch (stage two).

逐点调用 `avengine m5 capture-current-apartment-visual`,每点独立日志。
续跑语义(batch2d 的 b007 漏渲教训制度化):
  - 完成点 = 输出点目录里 research_receipt.json 存在且 frame_count==75
    → 跳过(幂等续跑);
  - **半成品**(目录在但无合格 receipt)→ 立即停下报人,绝不自动清理
    再跑——上次事故正是"清理半成品"与"跳过已存在"顺序交错漏掉一个点;
  - 任一点捕获失败 → 停(不跳过继续)。
输出根 fresh:首次运行创建;续跑时必须显式传 --resume 才接受已存在的
输出根。全链 research_candidate。

用法:
  run_qa_v3_capture_batch.py --inputs-root DESIGN --output-root CAPS \
      [--resume] [--points p1,p2] --spear-ext PATH --python BIN \
      --closure-report P --stage-root P --spear-executable P \
      [--graphics-adapter 1] [--rpc-port 30180]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def point_state(out_dir: Path) -> str:
    if not out_dir.exists():
        return "missing"
    receipt = out_dir / "research_receipt.json"
    if receipt.is_file():
        try:
            doc = json.loads(receipt.read_text())
            if doc.get("capture", {}).get("animation_readback_summary",
                                          {}).get("frame_count") == 75:
                return "complete"
        except (json.JSONDecodeError, OSError):
            pass
    return "partial"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--points", default=None,
                        help="逗号分隔的点子集;默认设计批全部点")
    parser.add_argument("--python", required=True)
    parser.add_argument("--spear-ext", required=True)
    parser.add_argument("--closure-report", required=True)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--spear-executable", required=True)
    parser.add_argument("--graphics-adapter", default="1")
    parser.add_argument("--rpc-port", default="30180")
    args = parser.parse_args(argv)

    if args.output_root.exists() and not args.resume:
        print(f"output root exists; pass --resume to continue: {args.output_root}",
              file=sys.stderr)
        return 2
    args.output_root.mkdir(parents=True, exist_ok=True)

    points = (args.points.split(",") if args.points else
              sorted(d.name for d in args.inputs_root.iterdir()
                     if d.is_dir() and (d / "timeline.json").is_file()))
    done = skipped = 0
    for pid in points:
        out_dir = args.output_root / pid
        state = point_state(out_dir)
        if state == "complete":
            skipped += 1
            continue
        if state == "partial":
            print(f"FAIL: {pid} is a partial capture at {out_dir} — refusing to "
                  f"clean or overwrite automatically; inspect and remove manually "
                  f"(b007-lesson guard)", file=sys.stderr)
            return 1
        pdir = args.inputs_root / pid
        cmd = [args.python, "-m", "avengine.cli", "m5",
               "capture-current-apartment-visual",
               "--actor-selection", str(pdir / "actor_selection.json"),
               "--source-asset-registry",
               str(REPO / "examples/runtime/source_asset_runtime_profiles.json"),
               "--timeline", str(pdir / "timeline.json"),
               "--closure-report", args.closure_report,
               "--stage-root", args.stage_root,
               "--spear-executable", args.spear_executable,
               "--rpc-port", args.rpc_port,
               "--graphics-adapter", args.graphics_adapter,
               "--output", str(out_dir)]
        env = dict(os.environ,
                   PYTHONPATH=f"{args.spear_ext}:{REPO / 'src'}")
        log_path = args.output_root / f"{pid}.log"
        started = datetime.now(timezone.utc).isoformat()
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=str(REPO), env=env)
        if proc.returncode != 0 or point_state(out_dir) != "complete":
            print(f"FAIL: capture of {pid} failed (exit {proc.returncode}); "
                  f"log: {log_path}", file=sys.stderr)
            return 1
        done += 1
        print(f"ok {pid} started={started} log={log_path.name}")
    print(f"captured={done} skipped_complete={skipped} total={len(points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

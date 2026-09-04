#!/usr/bin/env python3
"""Sequential UE capture runner for a qa-v3 design batch (stage two).

逐点调用 `avengine m5 capture-current-apartment-visual`,每点独立日志。
续跑语义(batch2d 的 b007 漏渲教训制度化):
  - 完成点 = 成功 receipt、实际 RGB 数组与逐帧读回匹配当前 timeline
    → 跳过(幂等续跑);
  - **半成品**(目录在但无合格 receipt)→ 立即停下报人,绝不自动清理
    再跑——上次事故正是"清理半成品"与"跳过已存在"顺序交错漏掉一个点;
  - 任一点捕获失败 → 停(不跳过继续)。
输出根 fresh:首次运行创建;续跑时必须显式传 --resume 才接受已存在的
输出根。全链 research_candidate。

用法:
  run_qa_v3_capture_batch.py --inputs-root DESIGN --output-root CAPS \
      [--resume] [--points p1,p2] [--intervention-file relative.json] \
      --spear-ext PATH --python BIN --closure-report P --stage-root P \
      --spear-executable P [--graphics-adapter 1] [--rpc-port 30180]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))
from verify_qa_v3_visual_batch import verify_point


def _resolve_description_path(raw: object, *, description_path: Path,
                               field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"intervention description field {field!r} must be a non-empty path")
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = description_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(
            f"intervention description field {field!r} is missing: {path}")
    return path


def resolve_capture_inputs(point_dir: Path, intervention_file: str | Path | None = None):
    """Resolve the selection/timeline for one point.

    With no description the historical main inputs are used.  When a
    description is supplied, its ``actor_selection`` and ``timeline`` fields
    are resolved relative to that description file, so the same runner can
    capture any declared visual intervention without knowing its profile name.
    """
    point_dir = Path(point_dir).resolve()
    if intervention_file is None:
        return point_dir / "actor_selection.json", point_dir / "timeline.json", None
    description = Path(intervention_file).expanduser()
    if not description.is_absolute():
        description = point_dir / description
    description = description.resolve()
    if not description.is_file():
        raise ValueError(f"intervention description is missing: {description}")
    try:
        payload = json.loads(description.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid intervention description {description}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"intervention description must be an object: {description}")
    selection = _resolve_description_path(
        payload.get("actor_selection"),
        description_path=description, field="actor_selection")
    timeline = _resolve_description_path(
        payload.get("timeline"), description_path=description, field="timeline")
    return selection, timeline, description


def _receipt_input_path(receipt_path: Path, receipt: dict, key: str) -> Path:
    inputs = receipt.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("capture receipt has no inputs")
    raw = inputs.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"capture receipt has no {key} input")
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = receipt_path.parent / path
    return path.resolve()


def _check_receipt_inputs(out_dir: Path, *, selection_path: Path,
                          timeline_path: Path) -> None:
    receipt_path = out_dir / "research_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "actor_selection": Path(selection_path).resolve(),
        "timeline": Path(timeline_path).resolve(),
    }
    for key, wanted in expected.items():
        actual = _receipt_input_path(receipt_path, receipt, key)
        if actual != wanted:
            raise ValueError(
                f"capture receipt {key} input {actual} differs from requested {wanted}")


def point_state(out_dir: Path, *, timeline_path: Path | None = None,
                selection_path: Path | None = None) -> str:
    if not out_dir.exists():
        return "missing"
    try:
        verify_point(out_dir.name, out_dir, timeline_path=timeline_path)
        if selection_path is not None:
            if timeline_path is None:
                raise ValueError("selection_path requires timeline_path")
            _check_receipt_inputs(
                out_dir, selection_path=selection_path, timeline_path=timeline_path)
    except (OSError, EOFError, ValueError, TypeError, KeyError, IndexError,
            json.JSONDecodeError):
        return "partial"
    return "complete"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--points", default=None,
                        help="逗号分隔的点子集;默认设计批全部点")
    parser.add_argument(
        "--intervention-file", default=None,
        help=("逐点干预描述相对路径; description 的 actor_selection 和 "
              "timeline 相对 description 文件解析"),
    )
    parser.add_argument("--python", required=True)
    parser.add_argument("--spear-ext", required=True)
    parser.add_argument("--source-asset-registry", default=str(REPO / "examples/runtime/source_asset_runtime_profiles.json"))
    parser.add_argument("--capture-warmup-config")
    parser.add_argument("--closure-report", required=True)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--spear-executable", required=True)
    parser.add_argument("--graphics-adapter", default="1")
    parser.add_argument("--rpc-port", default="30180")
    args = parser.parse_args(argv)
    args.inputs_root = args.inputs_root.resolve()
    args.output_root = args.output_root.resolve()

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
        pdir = args.inputs_root / pid
        try:
            selection_path, timeline_path, _ = resolve_capture_inputs(
                pdir, args.intervention_file)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"FAIL: {pid} capture input resolution failed: {exc}",
                  file=sys.stderr)
            return 1
        state = point_state(
            out_dir, timeline_path=timeline_path, selection_path=selection_path)
        if state == "complete":
            skipped += 1
            continue
        if state == "partial":
            print(f"FAIL: {pid} is a partial capture at {out_dir} — refusing to "
                  f"clean or overwrite automatically; inspect and remove manually "
                  f"(b007-lesson guard)", file=sys.stderr)
            return 1
        cmd = [args.python, "-m", "avengine.cli", "m5",
               "capture-current-apartment-visual",
               "--actor-selection", str(selection_path),
               "--source-asset-registry",
               args.source_asset_registry,
               "--timeline", str(timeline_path),
               "--closure-report", args.closure_report,
               "--stage-root", args.stage_root,
               "--spear-executable", args.spear_executable,
               "--rpc-port", args.rpc_port,
               "--graphics-adapter", args.graphics_adapter,
               "--output", str(out_dir)]
        if args.capture_warmup_config:
            cmd.extend(["--capture-warmup-config", args.capture_warmup_config])
        env = dict(os.environ,
                   PYTHONPATH=f"{REPO / 'src'}:{args.spear_ext}")
        log_path = args.output_root / f"{pid}.log"
        started = datetime.now(timezone.utc).isoformat()
        with open(log_path, "x") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=str(REPO), env=env)
        if proc.returncode != 0 or point_state(
                out_dir, timeline_path=timeline_path,
                selection_path=selection_path) != "complete":
            print(f"FAIL: capture of {pid} failed (exit {proc.returncode}); "
                  f"log: {log_path}", file=sys.stderr)
            return 1
        done += 1
        print(f"ok {pid} started={started} log={log_path.name}")
    print(f"captured={done} skipped_complete={skipped} total={len(points)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

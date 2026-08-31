#!/usr/bin/env python3
"""Sequential dynamic-audio runner for a qa-v3 design batch (stage two).

对设计批每个点位逐点调用 render_current_apartment_dynamic_audio.py。
固定依赖路径从 --config JSON 读(一个批一份,进 manifest 链);每点渲染
主 program(必要时含 Gate A program,--variants main,gateA)。

续跑语义与捕获调度器一致(b007 教训):
  - 完成点 = receipt 存在且 audio/binaural/mixture.wav 在盘上 → 跳过;
  - 半成品 → 拒绝清理,立即失败报人;
  - 任一点失败 → 停。
输出根 fresh,续跑需显式 --resume。全链 research_candidate。

config JSON 必备键:
  python, repo, m1_request, simulation_request, package_manifest,
  source_endpoint_registry, sound_asset_registry, beagle_audio, hrtf,
  runtime_prefix, rlr_sdk_root, magnum_python_site, source_asset_registry
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REQUIRED_CONFIG_KEYS = (
    "python", "repo", "m1_request", "simulation_request", "package_manifest",
    "source_endpoint_registry", "sound_asset_registry", "beagle_audio",
    "hrtf", "runtime_prefix", "rlr_sdk_root", "magnum_python_site",
    "source_asset_registry",
)


def point_state(out_dir: Path) -> str:
    if not out_dir.exists():
        return "missing"
    if (out_dir / "research_receipt.json").is_file() and \
            (out_dir / "audio" / "binaural" / "mixture.wav").is_file():
        return "complete"
    return "partial"


def program_path(programs_dir: Path, pid: str, variant: str) -> Path:
    suffix = "_rand_v1.json" if variant == "main" else f"_{variant}_rand_v1.json"
    matches = sorted(programs_dir.glob(f"qa_v3_*_{pid}{suffix}"))
    if len(matches) != 1:
        raise SystemExit(
            f"FAIL: expected exactly one {variant} program for {pid}, "
            f"found {len(matches)} in {programs_dir}")
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs-root", required=True, type=Path)
    parser.add_argument("--captures-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--variants", default="main",
                        help="逗号分隔:main[,gateA]")
    parser.add_argument("--points", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    cfg = json.loads(args.config.read_text())
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        print(f"config missing keys: {missing}", file=sys.stderr)
        return 2
    if args.output_root.exists() and not args.resume:
        print(f"output root exists; pass --resume to continue: {args.output_root}",
              file=sys.stderr)
        return 2
    args.output_root.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    points = (args.points.split(",") if args.points else
              sorted(d.name for d in args.inputs_root.iterdir()
                     if d.is_dir() and (d / "timeline.json").is_file()))
    programs_dir = args.inputs_root / "programs"
    repo = Path(cfg["repo"])
    done = skipped = 0
    for pid in points:
        cap_dir = args.captures_root / pid
        if not (cap_dir / "research_receipt.json").is_file():
            print(f"FAIL: capture for {pid} not complete at {cap_dir}",
                  file=sys.stderr)
            return 1
        for variant in variants:
            out_dir = args.output_root / (pid if variant == "main"
                                          else f"{pid}_{variant}")
            state = point_state(out_dir)
            if state == "complete":
                skipped += 1
                continue
            if state == "partial":
                print(f"FAIL: {out_dir} is a partial render — refusing to clean "
                      f"or overwrite automatically; inspect and remove manually "
                      f"(b007-lesson guard)", file=sys.stderr)
                return 1
            prog = program_path(programs_dir, pid, variant)
            cmd = [cfg["python"],
                   str(repo / "tools/dataset/render_current_apartment_dynamic_audio.py"),
                   "--visual-capture-dir", str(cap_dir),
                   "--m1-request", cfg["m1_request"],
                   "--simulation-request", cfg["simulation_request"],
                   "--package-manifest", cfg["package_manifest"],
                   "--audio-program", str(prog),
                   "--source-endpoint-registry", cfg["source_endpoint_registry"],
                   "--sound-asset-registry", cfg["sound_asset_registry"],
                   "--beagle-audio", cfg["beagle_audio"],
                   "--hrtf", cfg["hrtf"],
                   "--runtime-prefix", cfg["runtime_prefix"],
                   "--rlr-sdk-root", cfg["rlr_sdk_root"],
                   "--magnum-python-site", cfg["magnum_python_site"],
                   "--actor-selection", str(args.inputs_root / pid /
                                            "actor_selection.json"),
                   "--source-asset-registry", cfg["source_asset_registry"],
                   "--output", str(out_dir)]
            log_path = args.output_root / f"{out_dir.name}.log"
            with open(log_path, "w") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      cwd=str(repo))
            if proc.returncode != 0 or point_state(out_dir) != "complete":
                print(f"FAIL: audio render {out_dir.name} failed "
                      f"(exit {proc.returncode}); log: {log_path}",
                      file=sys.stderr)
                return 1
            done += 1
            print(f"ok {out_dir.name} log={log_path.name}")
    print(f"rendered={done} skipped_complete={skipped} "
          f"points={len(points)} variants={variants}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

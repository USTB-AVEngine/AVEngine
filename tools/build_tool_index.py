#!/usr/bin/env python3
"""Generate the capability index: what each tool does, grouped by capability.

Since 2026-08-25 the tools tree itself is laid out by capability (the stage
directories m1..m7/m6x/m6y/m6z are gone), so the index simply follows the
directory layout and each tool's own docstring.  Regenerate after adding,
moving or re-describing a tool:

    python tools/build_tool_index.py [--check]

--check exits non-zero when the checked-in index is stale (wired into the
unit suite, so a stale index fails the regression run).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
INDEX_PATH = REPOSITORY / "docs" / "TOOL_INDEX.md"

# Directory -> (title, subtitle). Order is the order of the index.
DIRECTORIES: list[tuple[str, str, str]] = [
    ("assets", "资产生成与装配", "从图像/网格到可运行资产：生成、修复、绑骨、动作、验收、打包、变体"),
    ("rooms", "房间", "房间引入、制备、审计、资格金丝雀（Habitat 与 SPEAR/UE 两条腿都在这里）"),
    ("scene", "场景放置", "基于真实场景表面规划并核验实体放置"),
    ("acoustics", "声学", "声学场景包、材质、RIR 缓存与计划、声学核验"),
    ("routes", "相机与路径", "相机机位、可行域、路径库、轨迹选择、发声锚点"),
    ("capture", "episode 捕获", "演员级 episode 的视觉捕获与动作试点"),
    ("visual", "视觉回放", "在 AVEngine 原生视觉路径中回放、渲染与核验放置结果"),
    ("timeline", "时间线", "权威时间线、音频程序（src/avengine/timeline，工具暂无）"),
    ("audio", "空间音频", "双耳/FOA 渲染与混音（src/avengine/spatial_audio，工具暂无）"),
    ("qa", "出题与认证", "题型设计、出题、闸门核验、held-out 划分、评测与打分"),
    ("dataset", "数据集装配", "训练/评测数据的规模化装配、重组、吞吐批与验证"),
    ("review", "审阅", "审阅页、交付片段、对比与评审证据"),
    ("registry", "注册表与发布", "注册表发布与核验"),
    ("release", "发布", "发布包组装"),
    ("studio", "Studio", "审阅与任务网页台"),
    ("ue", "UE 工程", "UE 编辑器内脚本：导入、修复、建图、导出"),
    ("blender", "Blender 工程", "Blender 内运行的资产处理脚本"),
    ("motion", "动作", "动作重定向与接触相位"),
]


def summary(path: Path) -> str:
    """First docstring line, or the first comment line, or an empty string."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        doc = None
    if doc:
        first = doc.strip().splitlines()[0].strip()
        return first.rstrip(".")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:5]:
        if line.startswith("#") and not line.startswith("#!"):
            return line.lstrip("# ").strip()
    return ""


def build() -> str:
    tools_root = REPOSITORY / "tools"
    seen: set[str] = set()
    lines: list[str] = []
    lines.append("# 工具能力索引")
    lines.append("")
    lines.append("> 由 `tools/build_tool_index.py` 从每个工具自己的 docstring 生成，")
    lines.append("> 改完工具重新运行即可刷新（`--check` 已入单测回归，索引过期会红）。")
    lines.append("")
    lines.append("自 2026-08-25 起，目录本身就是能力分组（阶段目录 m1…m7 已移除），")
    lines.append("本表按目录列出每个工具做什么。")
    total = 0
    body: list[str] = []
    for dirname, title, subtitle in DIRECTORIES:
        directory = tools_root / dirname
        if not directory.is_dir():
            continue
        entries = sorted(
            p for p in directory.iterdir()
            if p.suffix in {".py", ".sh"} and p.name != "__init__.py"
        )
        if not entries:
            continue
        seen.add(dirname)
        body.append("")
        body.append(f"## {title}（`tools/{dirname}/`）")
        body.append("")
        body.append(f"*{subtitle}*")
        body.append("")
        body.append("| 工具 | 做什么 |")
        body.append("|---|---|")
        for p in entries:
            total += 1
            body.append(f"| `tools/{dirname}/{p.name}` | {summary(p)} |")
    unknown = sorted(
        d.name for d in tools_root.iterdir()
        if d.is_dir() and d.name not in seen and d.name != "__pycache__"
        and any(f.suffix in {".py", ".sh"} for f in d.iterdir())
    )
    if unknown:
        raise SystemExit(
            f"tools/ has directories missing from build_tool_index.DIRECTORIES: {unknown}"
        )
    lines.append(f"当前共 {total} 个工具脚本。")
    lines.extend(body)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = build()
    if args.check:
        if not INDEX_PATH.exists() or INDEX_PATH.read_text(encoding="utf-8") != content:
            print("docs/TOOL_INDEX.md is stale; run: python tools/build_tool_index.py",
                  file=sys.stderr)
            return 1
        return 0
    INDEX_PATH.write_text(content, encoding="utf-8")
    print(f"wrote {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

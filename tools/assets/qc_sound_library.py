#!/usr/bin/env python3
"""Check every clip in the dry-sound library and say what is wrong, in words.

Run this after dropping new material into the library. Each clip gets a
report written beside it (clip.qc.json) which the sound-library page
reads, so whoever collects the material sees the verdict without asking
anyone. Exit code is nonzero when any clip is unusable, so this can gate
a batch.

Beyond the per-clip checks it does the one thing a single clip cannot
see: byte-identical copies of the same audio filed under several event
classes. That is not a crime - one recording legitimately serves both
"alarm beep" and "alarm clock" - but the right shape is one file whose
sidecar declares both classes. Two copies mean the question miner can
pick physically identical audio for two different sound sources in one
room, and "which of them is making the sound" stops having an answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.assets.sound_qc import write_clip_qc  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-root",
        type=Path,
        default=Path("/data/avengine_external/assets/sound_library_v1"),
    )
    parser.add_argument(
        "--quiet", action="store_true", help="only print the summary"
    )
    args = parser.parse_args()

    root = args.library_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"library root does not exist: {root}")

    clips = sorted(root.rglob("*.wav"))
    if not clips:
        print(json.dumps({"library_root": str(root), "clips": 0}))
        return 0

    verdicts: Counter[str] = Counter()
    by_digest: dict[str, list[str]] = defaultdict(list)
    flagged: list[tuple[str, dict]] = []

    for clip in clips:
        relative = clip.relative_to(root).as_posix()
        by_digest[hashlib.sha256(clip.read_bytes()).hexdigest()].append(relative)
        report = write_clip_qc(clip)
        verdicts[report["verdict"]] += 1
        if report["findings"]:
            flagged.append((relative, report))

    duplicates = {
        digest: paths for digest, paths in by_digest.items() if len(paths) > 1
    }

    if not args.quiet:
        for relative, report in flagged:
            mark = "✗" if report["verdict"] == "fail" else "!"
            print(f"{mark} {relative}")
            for finding in report["findings"]:
                print(f"    - {finding['reason_zh']}")
        if duplicates:
            print("\n同一段音频被复制到了多个事件类目录下:")
            for paths in sorted(duplicates.values(), key=len, reverse=True):
                classes = sorted({p.split("/")[0] for p in paths})
                print(f"    {paths[0]} 也出现在 {classes}")
            print(
                "  建议:每段音频只留一份,在旁车 json 的 event_classes 里写上它服务的"
                "所有事件类,不要复制文件。"
            )

    unique = len(by_digest)
    print(
        f"\n共 {len(clips)} 个文件 / {unique} 段不同音频 · "
        f"合格 {verdicts['pass']} · 有提醒 {verdicts['warn']} · 不可用 {verdicts['fail']} · "
        f"重复占用 {len(clips) - unique} 个文件"
    )
    return 1 if verdicts["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

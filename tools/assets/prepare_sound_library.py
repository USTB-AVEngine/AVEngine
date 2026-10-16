#!/usr/bin/env python3
"""Convert the collected dry clips into the form the pipeline consumes.

The collector was promised they would never hand-edit audio: this is the
script that keeps that promise. It reads the library as delivered, skips
whatever QC judged unusable, and writes 16 kHz mono copies into a
separate tree - the originals are never touched, so a mistake here costs
a re-run and nothing else.

Run qc_sound_library.py first; clips without a QC report are skipped
rather than prepared blind.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.assets.sound_prepare import (  # noqa: E402
    TARGET_RATE_HZ,
    prepare_library,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-root",
        type=Path,
        default=Path("/data/avengine_external/assets/sound_library_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/avengine_external/assets/sound_library_v1_prepared"),
    )
    parser.add_argument("--target-rate-hz", type=int, default=TARGET_RATE_HZ)
    parser.add_argument(
        "--reject-warned",
        action="store_true",
        help="also skip clips QC merely warned about (default: keep them)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.library_root.is_dir():
        raise SystemExit(f"library root does not exist: {args.library_root}")

    report = prepare_library(
        args.library_root.resolve(),
        args.output_root.resolve(),
        target_rate_hz=args.target_rate_hz,
        accept_warn=not args.reject_warned,
    )

    manifest = args.output_root / "prepared_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not args.quiet:
        for clip in report["clips"]:
            if clip["status"] in ("skipped", "failed"):
                print(f"✗ {clip['source']}\n    - {clip['reason_zh']}")
    counts = report["counts"]
    print(
        f"\n处理完成 {counts.get('prepared', 0)} 条 · "
        f"重复未重做 {counts.get('alias', 0)} 条 · "
        f"跳过 {counts.get('skipped', 0)} 条 · "
        f"失败 {counts.get('failed', 0)} 条\n"
        f"产物在 {args.output_root}(原始素材未改动),清单 {manifest}"
    )
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export one completed QA-v3 pipeline run without model-specific fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.dataset_export import DatasetExportError, export_dataset, probe_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", required=True, type=Path)
    parser.add_argument("--released-items", type=Path)
    parser.add_argument(
        "--layout", action="append", dest="layouts",
        help="requested layout name or comma-separated names; repeatable",
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    pipeline_root = args.pipeline_root.expanduser().resolve()
    released = (
        args.released_items.expanduser().resolve()
        if args.released_items is not None
        else pipeline_root / "questions/released_items.json"
    )
    try:
        manifest = export_dataset(
            pipeline_root=pipeline_root,
            released_items_path=released,
            output_root=args.output_root,
            layouts=args.layouts,
            video_probe=lambda path: probe_video(path, ffprobe=args.ffprobe),
        )
    except DatasetExportError as exc:
        print(f"dataset export failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": manifest["status"],
        "output": str(args.output_root.expanduser().resolve()),
        "counts": manifest["counts"],
        "layouts": manifest["layouts"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

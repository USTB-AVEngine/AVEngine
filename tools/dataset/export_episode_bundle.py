#!/usr/bin/env python3
"""Publish a shared-room Episode/QA reference bundle.

The request file is the only authoring input. Media and evidence remain at
their declared paths; this command writes JSON/JSONL indexes and storage
statistics without copying those files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.dataset.episode_export import (  # noqa: E402
    EpisodeExportError,
    export_episode_bundle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--gzip-json",
        action="store_true",
        help="compress JSON/JSONL sidecars as .gz; keep manifest.json readable",
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    try:
        manifest = export_episode_bundle(
            request_path=args.request,
            output_root=args.output_root,
            gzip_json=args.gzip_json,
            ffprobe=args.ffprobe,
        )
    except EpisodeExportError as exc:
        print(f"episode export failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

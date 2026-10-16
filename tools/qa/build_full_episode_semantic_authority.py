#!/usr/bin/env python3
"""Build an approved full-Episode semantic authority without overwriting output."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from avengine.contracts.json_io import load_json
from avengine.qa.full_episode_semantic_authority import (
    build_full_episode_semantic_authority,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _publish_json_exclusive(value: object, output_path: Path) -> None:
    """Fully persist JSON before atomically linking it at an unused final path."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, output_path)
        directory_descriptor = os.open(
            output_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    request_path = args.request.resolve(strict=True)
    output_raw = args.output.absolute()
    if output_raw.is_symlink() or os.path.lexists(output_raw):
        raise FileExistsError(f"refusing to overwrite output: {output_raw}")
    output_path = output_raw.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    result = build_full_episode_semantic_authority(
        load_json(request_path), authority_path=output_path
    )
    _publish_json_exclusive(result, output_path)
    print(
        f"FULL_EPISODE_SEMANTIC_AUTHORITY_OK episodes={result['episode_count']} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

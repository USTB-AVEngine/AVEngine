#!/usr/bin/env python3
"""Build one validation batch from an explicit request."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from avengine.contracts.json_io import load_json, write_json
from avengine.qa.full_episode_validation_batch import (
    build_full_episode_validation_batch,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    request_path = args.request.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    result = build_full_episode_validation_batch(load_json(request_path))
    write_json(output_path, result)
    print(
        "FULL_EPISODE_VALIDATION_BATCH_OK "
        f"episodes={result['episode_count']} "
        f"output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

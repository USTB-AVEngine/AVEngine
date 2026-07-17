#!/usr/bin/env python3
"""Bake one positive uniform skin-ancestor scale into GLB payload data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from avengine.m2.similarity import SimilarityBakeError, bake_uniform_skin_ancestor_scale


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    for label, path in (("output", args.output), ("report", args.report)):
        if path.exists() or path.is_symlink():
            parser.error(f"refusing to replace {label}: {path}")
    source_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    if len({source_path, output_path, report_path}) != 3:
        parser.error("input, output, and report paths must differ")
    output_created = False
    try:
        report = bake_uniform_skin_ancestor_scale(source_path, output_path)
        output_created = True
        _write_exclusive(
            report_path,
            (
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
        )
    except (SimilarityBakeError, OSError) as exc:
        message = str(exc)
        if output_created:
            try:
                output_path.unlink()
            except OSError as cleanup_exc:
                message += f"; failed to clean newly created output: {cleanup_exc}"
        parser.error(message)
    print(
        json.dumps(
            {"status": "pass", "output": report["output"], "report": str(report_path)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

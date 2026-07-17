#!/usr/bin/env python3
"""Create a Habitat-native, root-local GLB research candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from avengine.m2.rebase import (
    RebaseError,
    rebase_skin_root,
    rebase_skin_root_preserving_local_tr,
)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--preserve-local-tr",
        action="store_true",
        help=(
            "retain non-root STEP/LINEAR translations for the research-only "
            "local-TR v2 runtime"
        ),
    )
    args = parser.parse_args(argv)
    for label, path in (("output", args.output), ("report", args.report)):
        if path.exists() or path.is_symlink():
            parser.error(f"refusing to replace {label}: {path}")
    source = args.input.resolve()
    output = args.output.resolve()
    report_path = args.report.resolve()
    if len({source, output, report_path}) != 3:
        parser.error("input, output, and report paths must differ")
    output_created = False
    try:
        rebase = (
            rebase_skin_root_preserving_local_tr
            if args.preserve_local_tr
            else rebase_skin_root
        )
        report = rebase(source, output)
        output_created = True
        _write_exclusive(
            report_path,
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    except (OSError, RebaseError) as exc:
        message = str(exc)
        if output_created:
            try:
                output.unlink()
            except OSError as cleanup_exc:
                message += f"; failed to clean newly created output: {cleanup_exc}"
        parser.error(message)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": report["output"],
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

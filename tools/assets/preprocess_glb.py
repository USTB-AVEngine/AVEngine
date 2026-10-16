#!/usr/bin/env python3
"""Select GLB actions and strip provably unweighted controller roots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from avengine.assets.preprocess import GlbPreprocessError, preprocess_glb


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


def _action_mapping(value: str) -> tuple[str, str]:
    source, separator, target = value.partition("=")
    if not separator or not source.strip() or not target.strip():
        raise argparse.ArgumentTypeError(
            "action mappings must use exact SOURCE=OUTPUT names"
        )
    return source, target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select exact actions and remove only disconnected skin-root branches "
            "proven to have zero mesh weight."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--action",
        type=_action_mapping,
        action="append",
        required=True,
        metavar="SOURCE=OUTPUT",
        help="exact action selection/rename; repeat in desired output order",
    )
    parser.add_argument(
        "--prune-zero-weight-leaves",
        action="store_true",
        help=(
            "also remove terminal skin joints proven to have exactly zero weight "
            "and no descendants or scene payload"
        ),
    )
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
        report = preprocess_glb(
            source_path,
            output_path,
            action_map=args.action,
            prune_zero_weight_leaves=args.prune_zero_weight_leaves,
        )
        output_created = True
        _write_exclusive(
            report_path,
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    except (GlbPreprocessError, OSError) as exc:
        message = str(exc)
        if output_created:
            try:
                output_path.unlink()
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

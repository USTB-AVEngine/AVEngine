#!/usr/bin/env python3
"""Apply explicit action durations without changing sampled pose values."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from avengine.assets.timing import ActionTimingError, retime_glb_actions


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


def _duration(value: str) -> tuple[str, float]:
    name, separator, raw = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("duration must use ACTION=SECONDS")
    try:
        seconds = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("duration seconds must be numeric") from exc
    return name, seconds


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--duration", type=_duration, action="append", required=True)
    args = parser.parse_args(argv)
    for label, path in (("output", args.output), ("report", args.report)):
        if path.exists() or path.is_symlink():
            parser.error(f"refusing to replace {label}: {path}")
    source_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    if len({source_path, output_path, report_path}) != 3:
        parser.error("input, output, and report paths must differ")
    durations = dict(args.duration)
    if len(durations) != len(args.duration):
        parser.error("each action may be retimed only once")
    output_created = False
    try:
        report = retime_glb_actions(
            source_path, output_path, durations_seconds=durations
        )
        output_created = True
        _write_exclusive(
            report_path,
            (
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
        )
    except (ActionTimingError, OSError) as exc:
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

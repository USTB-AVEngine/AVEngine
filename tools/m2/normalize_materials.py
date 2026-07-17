#!/usr/bin/env python3
"""Normalize GLB PBR materials without modifying geometry or animation data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from avengine.m2.materials import (
    MaterialNormalizationError,
    normalize_glb_materials,
    validate_material_normalization_report,
)


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Set non-metallic matte PBR factors while proving every non-material "
            "GLB section unchanged."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--force-opaque",
        action="store_true",
        help="also set alphaMode=OPAQUE and baseColorFactor alpha=1",
    )
    args = parser.parse_args(argv)

    for label, path in (("output", args.output), ("report", args.report)):
        if path.exists() or path.is_symlink():
            parser.error(f"refusing to replace {label}: {path}")

    resolved = {
        "input": args.input.resolve(),
        "output": args.output.resolve(),
        "report": args.report.resolve(),
    }
    if len(set(resolved.values())) != 3:
        parser.error("input, output, and report paths must all differ")
    output_created = False
    try:
        report = normalize_glb_materials(
            resolved["input"],
            resolved["output"],
            force_opaque=args.force_opaque,
        )
        output_created = True
        validate_material_normalization_report(report, verify_files=True)
        _write_json_exclusive(resolved["report"], report)
    except (MaterialNormalizationError, OSError) as exc:
        message = str(exc)
        if output_created:
            try:
                resolved["output"].unlink()
            except OSError as cleanup_exc:
                message += f"; failed to clean newly created output: {cleanup_exc}"
        parser.error(message)

    print(
        json.dumps(
            {
                "status": report["status"],
                "output": report["output"],
                "report": str(resolved["report"]),
                "report_content_sha256": report["report_content_sha256"],
                "material_count": report["material_count"],
                "qualification_state": report["qualification_state"],
                "qualification_claim": report["qualification_claim"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

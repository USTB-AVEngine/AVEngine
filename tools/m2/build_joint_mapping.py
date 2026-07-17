#!/usr/bin/env python3
"""Emit the exact Habitat joint mapping bound to a rebase report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from avengine.contracts.json_io import load_json, sha256_file
from avengine.m2.glb import load_glb
from avengine.m2.habitat import build_habitat_asset_mapping_from_rebase_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-glb", type=Path, required=True)
    parser.add_argument("--rebase-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"refusing to replace joint mapping: {output}")
    document = load_glb(args.visual_glb)
    mapping = build_habitat_asset_mapping_from_rebase_report(
        document, load_json(args.rebase_report)
    )
    value = mapping.joint_mapping_data()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if load_json(output) != value:
        raise ValueError("joint mapping readback differs")
    print(
        json.dumps(
            {"status": "pass", "output": str(output), "sha256": sha256_file(output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

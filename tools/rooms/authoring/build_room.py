#!/usr/bin/env python3
# Generic Blender authoring entry point for room resource builders.
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

DEFAULT_BLENDER = Path("/data/jzy/blender/blender-4.5.13-linux-x64/blender")
DEFAULT_BUILDER = Path(__file__).with_name("semantic_household_builder.py")

def main() -> int:
    parser = argparse.ArgumentParser(description="Build one room resource with Blender.")
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--builder", type=Path, default=DEFAULT_BUILDER)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    blender = args.blender.expanduser().resolve(strict=True)
    builder = args.builder.expanduser().resolve(strict=True)
    spec = args.spec.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise SystemExit("refusing to replace existing output: " + str(output))
    command = [str(blender), "--background", "--python", str(builder), "--", "--spec", str(spec), "--output-root", str(output)]
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export one existing AVEngine room .blend to USD without exporting its camera."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import bpy

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    argv = __import__("sys").argv
    args = parser.parse_args(argv[argv.index("--") + 1:] if "--" in argv else [])
    blend = args.blend.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit("refusing to replace existing USD output: " + str(output))
    builder_path = Path(__file__).with_name("semantic_household_builder.py")
    spec = importlib.util.spec_from_file_location("avengine_room_builder", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import local room builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    record = module.export_static_usd(bpy.context.scene, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = output.with_suffix(".export.json")
    report.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Bind a compiled Timeline-v2 visual plan to the imported MP3D UE scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avengine.optional_backends.spear_mp3d import build_mp3d_execution_plan


def _mapping(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-plan", type=Path, required=True)
    parser.add_argument("--ue-import-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-output-gain", type=float, default=0.72)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_mp3d_execution_plan(
        visual_plan=_mapping(args.visual_plan.resolve()),
        ue_import_manifest=_mapping(args.ue_import_manifest.resolve()),
        output_gain=args.fixed_output_gain,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "SPEAR_MP3D_EXECUTION_OK "
        f"frames={plan['render']['frame_count']} "
        f"scene_meshes={plan['scene']['spawned_scene_mesh_actor_count']} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

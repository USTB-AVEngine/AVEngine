#!/usr/bin/env python3
"""Prepare the complete ReplicaCAD scene request for the optional UE backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avengine.optional_backends.spear_replicacad import build_replicacad_scene_plan
from avengine.optional_backends.spear_replicacad_execution import (
    assert_apt0_execution_request,
    build_replicacad_execution_request,
)
from avengine.optional_backends.spear_replicacad_glb import (
    prepare_replicacad_source_glbs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicacad-root", type=Path, required=True)
    parser.add_argument("--scene-id", default="apt_0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--content-root",
        default="/Game/AVEngine/Optional/ReplicaCAD/apt_0",
    )
    parser.add_argument(
        "--prepared-glb-dir",
        type=Path,
        help=(
            "Bake source glTF node transforms into geometry for deterministic "
            "UE asset import; required before the editor import stage."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.replicacad_root.resolve()
    plan = build_replicacad_scene_plan(
        root / "replicaCAD.scene_dataset_config.json",
        root / "configs" / "scenes" / f"{args.scene_id}.scene_instance.json",
    )
    request = build_replicacad_execution_request(plan, content_root=args.content_root)
    if args.scene_id == "apt_0":
        assert_apt0_execution_request(request)
    if args.prepared_glb_dir is not None:
        request = prepare_replicacad_source_glbs(
            request, args.prepared_glb_dir.resolve()
        )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(request, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts = request["counts"]
    print(
        "SPEAR_REPLICACAD_REQUEST_OK "
        f"logical_instances={counts['logical_instance_count']} "
        f"source_glbs={counts['source_glb_count']} "
        f"static_mesh_assets={counts['expected_imported_static_mesh_asset_count']} "
        f"runtime_mesh_actors={counts['expected_runtime_mesh_actor_count']} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

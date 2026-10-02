#!/usr/bin/env python3
"""Build an external-input inventory for HM3D room-screening runs."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from tools.rooms.emit_hm3d_room_manifest import inventory_rooms
from avengine.rooms.habitat_capture import prepare_installed_habitat_runtime

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "tmp/room_screening/input_inventory.json"
HOUSE_PATTERN = re.compile(r"^hm3d_([a-zA-Z0-9]+)_([0-9]{5})_([A-Za-z0-9]+)$")


def load_houses(houses_file: Path | None, repeated: list[str] | None) -> list[str]:
    houses = list(repeated or [])
    if houses_file:
        houses.extend(
            line.strip() for line in houses_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    result = list(dict.fromkeys(houses))
    if not result:
        raise ValueError("provide --houses-file or at least one --house")
    invalid = [house for house in result if not HOUSE_PATTERN.fullmatch(house)]
    if invalid:
        raise ValueError(f"invalid HM3D house IDs: {invalid}")
    return result


def scalar_settings(settings) -> dict:
    result = {}
    for name in dir(settings):
        if name.startswith("_"):
            continue
        try:
            value = getattr(settings, name)
        except Exception:
            continue
        if callable(value):
            continue
        if isinstance(value, (bool, int, float, str)):
            if isinstance(value, float) and not math.isfinite(value):
                value = str(value)
            result[name] = value
    return result


def build_inventory(houses: list[str], dataset_root: Path, habitat_sim) -> dict:
    scene_rows, region_rows = [], []
    for house in houses:
        match = HOUSE_PATTERN.fullmatch(house)
        assert match is not None
        split, index, scene_id = match.groups()
        scene_dir = dataset_root / split / f"{index}-{scene_id}"
        semantic_glb = scene_dir / f"{scene_id}.semantic.glb"
        semantic_txt = scene_dir / f"{scene_id}.semantic.txt"
        navmesh_path = scene_dir / f"{scene_id}.basis.navmesh"
        issue_codes = []
        navmesh_settings = None
        navmesh_loaded = False
        if navmesh_path.is_file():
            pathfinder = habitat_sim.PathFinder()
            navmesh_loaded = bool(pathfinder.load_nav_mesh(str(navmesh_path)))
            if navmesh_loaded:
                navmesh_settings = scalar_settings(pathfinder.nav_mesh_settings)
            else:
                issue_codes.append("navmesh_load_failed")
        else:
            issue_codes.append("navmesh_missing")
        has_semantics = semantic_glb.is_file() and semantic_txt.is_file()
        room_doc = None
        if has_semantics:
            try:
                room_doc = inventory_rooms(scene_dir, scene_id)
            except Exception as error:
                issue_codes.append(f"semantic_inventory_failed:{type(error).__name__}")
        else:
            issue_codes.append("annotation_missing")
        complete = navmesh_loaded and room_doc is not None
        scene_rows.append({
            "house": house,
            "split": split,
            "input_status": "complete" if complete else "unassessable",
            "issue_codes": issue_codes,
            "navmesh": {
                "loaded": navmesh_loaded,
                "navmesh_settings": navmesh_settings or {},
            },
            "source_room_count": len(room_doc.get("rooms", [])) if room_doc else 0,
        })
        for room in (room_doc or {}).get("rooms", []):
            region_rows.append({
                "house": house,
                "region_id": room["region_id"],
                "bbox_xz_m": room.get("bbox_xz_m"),
                "floor_y_m": room.get("floor_y_m"),
                "source_floor_area_m2": room.get("floor_area_m2"),
                "source_top_categories": room.get("top_categories", []),
            })
        status = "ready" if complete else f"unassessable ({','.join(issue_codes)})"
        print(f"{house}: {len((room_doc or {}).get('rooms', []))} semantic regions; {status}", flush=True)
    return {
        "schema_version": "room_screening_input_inventory_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_note": "Generated from explicitly supplied HM3D files and semantic region annotations.",
        "scene_count": len(scene_rows),
        "region_count": len(region_rows),
        "scenes": scene_rows,
        "regions": region_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--houses-file", type=Path, help="UTF-8 file with one HM3D house ID per line")
    parser.add_argument("--house", action="append", help="one HM3D house ID; repeatable")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="external HM3D root containing train/, val/, etc.")
    parser.add_argument("--runtime-prefix", default=os.environ.get("AVENGINE_HABITAT_RUNTIME_PREFIX"))
    parser.add_argument("--magnum-python-site", default=os.environ.get("AVENGINE_HABITAT_MAGNUM_PYTHON_SITE"))
    parser.add_argument("--mp3d-root", default=os.environ.get("AVENGINE_MP3D_ROOT"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="output JSON; default is repository tmp/room_screening/")
    args = parser.parse_args()
    if not (args.runtime_prefix and args.magnum_python_site and args.mp3d_root):
        parser.error("supply Habitat runtime, Magnum Python site, and MP3D root explicitly or via AVENGINE_* environment variables")
    try:
        houses = load_houses(args.houses_file, args.house)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        parser.error(f"HM3D dataset root not found: {dataset_root}")
    runtime = prepare_installed_habitat_runtime(
        runtime_prefix=args.runtime_prefix,
        magnum_python_site=args.magnum_python_site,
        mp3d_root=args.mp3d_root,
        allow_mp3d_environment=False,
    )
    result = build_inventory(houses, dataset_root, runtime.habitat_sim)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"Wrote {len(result['scenes'])} scenes / {len(result['regions'])} regions to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract a compact room polygon and top-level bounds from InteriorAgent USD.

Run this optional tool in an environment with Pixar USD.  It writes only
simple scene facts used for source-center placement and Topdown review; it does
not copy meshes, materials or textures from the external dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.optional_backends.interioragent_kujiale import (  # noqa: E402
    load_room_metadata,
)
from avengine.optional_backends.residential_episode import (  # noqa: E402
    SCENE_METADATA_SCHEMA,
    classify_object_bounds,
)

try:
    from pxr import Usd, UsdGeom
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("Pixar USD Python bindings are required") from exc


def _bounds(cache: Any, prim: Any) -> list[list[float]] | None:
    aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    minimum = aligned.GetMin()
    maximum = aligned.GetMax()
    values = [
        [float(minimum[index]) for index in range(3)],
        [float(maximum[index]) for index in range(3)],
    ]
    if not all(math.isfinite(item) for point in values for item in point):
        return None
    if any(values[1][axis] < values[0][axis] for axis in range(3)):
        return None
    return values


def extract(
    *,
    source: Path,
    rooms_path: Path,
    room_type: str,
    room_scope: str,
    scene_id: str,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rooms = load_room_metadata(rooms_path)
    matches = [room for room in rooms if room["room_type"] == room_type]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {room_type!r} polygon, got {len(matches)}")

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"could not open USD stage: {source}")
    scope_path = f"/Root/Meshes/{room_scope}"
    scope = stage.GetPrimAtPath(scope_path)
    if not scope or not scope.IsValid():
        raise RuntimeError(f"room scope is absent: {scope_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
        ignoreVisibility=False,
    )
    objects = []
    for child in scope.GetChildren():
        bounds = _bounds(cache, child)
        if bounds is None:
            continue
        objects.append(
            {
                "object_id": child.GetName(),
                "prim_path": str(child.GetPath()),
                "bounds_xyz_m": bounds,
                "navigation_role": classify_object_bounds(bounds),
            }
        )
    objects.sort(key=lambda item: item["object_id"].encode("utf-8"))
    role_counts: dict[str, int] = {}
    for item in objects:
        role = item["navigation_role"]
        role_counts[role] = role_counts.get(role, 0) + 1
    return {
        "schema": SCENE_METADATA_SCHEMA,
        "dataset_id": "spatialverse/InteriorAgent",
        "scene_id": scene_id,
        "room_id": f"{scene_id}_{room_scope}",
        "room_type": room_type,
        "room_scope": room_scope,
        "room_polygon_xy_m": matches[0]["polygon_xy_m"],
        "floor_z_m": 0.0,
        "objects": objects,
        "object_role_counts": role_counts,
        "source_reference": str(source),
        "claim_boundary": (
            "external InteriorAgent/Kujiale research scene; metadata contains "
            "only a room polygon and object bounds, not dataset geometry or textures"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rooms", type=Path, required=True)
    parser.add_argument("--room-type", default="living room")
    parser.add_argument("--room-scope", default="livingroom_491")
    parser.add_argument("--scene-id", default="kujiale_0020")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.replace:
        raise FileExistsError(f"refusing to replace output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = extract(
        source=args.source,
        rooms_path=args.rooms,
        room_type=args.room_type,
        room_scope=args.room_scope,
        scene_id=args.scene_id,
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(result["object_role_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

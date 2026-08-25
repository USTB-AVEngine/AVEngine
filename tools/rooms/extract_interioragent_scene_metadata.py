#!/usr/bin/env python3
"""Extract a room polygon and navigation footprints from InteriorAgent USD.

Run this optional tool in an environment with Pixar USD.  It writes only
simple scene facts used for source-center placement and Topdown review.  A
ground blocker is represented by descendant-mesh XY rectangles instead of one
potentially over-large parent AABB; no mesh, material or texture is copied.
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
    from pxr import Gf, Usd, UsdGeom
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


FOOTPRINT_CELL_SIZE_M = 0.05


def _contains_xy(
    point: tuple[float, float], triangle: tuple[tuple[float, float], ...]
) -> bool:
    a, b, c = triangle
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (
        a[1] - c[1]
    )
    if abs(denominator) <= 1.0e-12:
        return False
    u = ((b[1] - c[1]) * (point[0] - c[0]) + (c[0] - b[0]) * (
        point[1] - c[1]
    )) / denominator
    v = ((c[1] - a[1]) * (point[0] - c[0]) + (a[0] - c[0]) * (
        point[1] - c[1]
    )) / denominator
    return u >= -1.0e-9 and v >= -1.0e-9 and u + v <= 1.0 + 1.0e-9


def _mark_segment_cells(
    cells: set[tuple[int, int]], a: tuple[float, float], b: tuple[float, float]
) -> None:
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    step_count = max(1, int(math.ceil(length / (FOOTPRINT_CELL_SIZE_M / 2.0))))
    for index in range(step_count + 1):
        fraction = index / step_count
        x = a[0] + (b[0] - a[0]) * fraction
        y = a[1] + (b[1] - a[1]) * fraction
        cells.add(
            (
                math.floor(x / FOOTPRINT_CELL_SIZE_M),
                math.floor(y / FOOTPRINT_CELL_SIZE_M),
            )
        )


def _mesh_footprint_cells(
    prim: Any, xform_cache: Any, *, floor_z_m: float
) -> set[tuple[int, int]]:
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
    counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
    indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
    if not points or not counts or not indices:
        return set()
    transform = xform_cache.GetLocalToWorldTransform(prim)
    world = [transform.Transform(Gf.Vec3d(*point)) for point in points]
    cells: set[tuple[int, int]] = set()
    offset = 0
    for count in counts:
        count = int(count)
        face = [int(value) for value in indices[offset : offset + count]]
        offset += count
        if count < 3 or any(index < 0 or index >= len(world) for index in face):
            continue
        for triangle_index in range(1, count - 1):
            vertices = (
                world[face[0]],
                world[face[triangle_index]],
                world[face[triangle_index + 1]],
            )
            z_values = [float(value[2]) for value in vertices]
            if max(z_values) <= floor_z_m + 0.10:
                continue
            # The user-selected placement policy checks only the source root
            # center, not a human/animal body capsule.  Keep a narrow slice
            # just above floor coverings so table tops and hanging lights do
            # not become full-height blockers while legs and cabinet bases do.
            if min(z_values) >= floor_z_m + 0.15:
                continue
            triangle = tuple((float(value[0]), float(value[1])) for value in vertices)
            for start, end in zip(triangle, triangle[1:] + triangle[:1], strict=True):
                _mark_segment_cells(cells, start, end)
            minimum_x = min(value[0] for value in triangle)
            maximum_x = max(value[0] for value in triangle)
            minimum_y = min(value[1] for value in triangle)
            maximum_y = max(value[1] for value in triangle)
            first_x = math.ceil(minimum_x / FOOTPRINT_CELL_SIZE_M - 0.5)
            last_x = math.floor(maximum_x / FOOTPRINT_CELL_SIZE_M - 0.5)
            first_y = math.ceil(minimum_y / FOOTPRINT_CELL_SIZE_M - 0.5)
            last_y = math.floor(maximum_y / FOOTPRINT_CELL_SIZE_M - 0.5)
            for cell_y in range(first_y, last_y + 1):
                for cell_x in range(first_x, last_x + 1):
                    center = (
                        (cell_x + 0.5) * FOOTPRINT_CELL_SIZE_M,
                        (cell_y + 0.5) * FOOTPRINT_CELL_SIZE_M,
                    )
                    if _contains_xy(center, triangle):
                        cells.add((cell_x, cell_y))
    return cells


def _cells_to_rectangles(
    cells: set[tuple[int, int]],
) -> list[list[list[float]]]:
    row_cells: dict[int, list[int]] = {}
    for cell_x, cell_y in cells:
        row_cells.setdefault(cell_y, []).append(cell_x)
    row_runs: dict[int, set[tuple[int, int]]] = {}
    for cell_y, raw_x in row_cells.items():
        values = sorted(set(raw_x))
        runs: set[tuple[int, int]] = set()
        start = previous = values[0]
        for current in values[1:]:
            if current != previous + 1:
                runs.add((start, previous))
                start = current
            previous = current
        runs.add((start, previous))
        row_runs[cell_y] = runs

    cell_rectangles: list[tuple[int, int, int, int]] = []
    active: dict[tuple[int, int], int] = {}
    previous_y: int | None = None
    for cell_y in sorted(row_runs):
        if previous_y is None or cell_y != previous_y + 1:
            for (start_x, end_x), start_y in active.items():
                cell_rectangles.append((start_x, end_x, start_y, previous_y))
            active.clear()
        current_runs = row_runs[cell_y]
        for run in tuple(active):
            if run not in current_runs:
                start_y = active.pop(run)
                cell_rectangles.append((run[0], run[1], start_y, previous_y))
        for run in current_runs:
            active.setdefault(run, cell_y)
        previous_y = cell_y
    for (start_x, end_x), start_y in active.items():
        cell_rectangles.append((start_x, end_x, start_y, previous_y))

    rectangles = [
        [
            [start_x * FOOTPRINT_CELL_SIZE_M, start_y * FOOTPRINT_CELL_SIZE_M],
            [
                (end_x + 1) * FOOTPRINT_CELL_SIZE_M,
                (end_y + 1) * FOOTPRINT_CELL_SIZE_M,
            ],
        ]
        for start_x, end_x, start_y, end_y in cell_rectangles
    ]
    return sorted(rectangles, key=lambda value: tuple(item for point in value for item in point))


def _navigation_footprint_parts(
    prim: Any, xform_cache: Any, *, floor_z_m: float
) -> list[list[list[float]]]:
    """Rasterize actual descendant Mesh triangles into compact XY rectangles."""

    cells: set[tuple[int, int]] = set()
    for descendant in Usd.PrimRange(prim):
        if descendant.IsA(UsdGeom.Mesh):
            cells.update(
                _mesh_footprint_cells(
                    descendant, xform_cache, floor_z_m=floor_z_m
                )
            )
    return _cells_to_rectangles(cells)


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
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    floor_z_m = 0.0
    objects = []
    for child in scope.GetChildren():
        bounds = _bounds(cache, child)
        if bounds is None:
            continue
        role = classify_object_bounds(bounds, floor_z_m=floor_z_m)
        record = {
            "object_id": child.GetName(),
            "prim_path": str(child.GetPath()),
            "bounds_xyz_m": bounds,
            "navigation_role": role,
        }
        if role == "ground_blocker":
            parts = _navigation_footprint_parts(
                child, xform_cache, floor_z_m=floor_z_m
            )
            if parts:
                record["footprint_parts_xy_m"] = parts
                record["footprint_basis"] = (
                    "descendant_mesh_projected_triangle_grid_0.05m"
                )
            else:
                record["footprint_parts_xy_m"] = [
                    [bounds[0][:2], bounds[1][:2]]
                ]
                record["footprint_basis"] = (
                    "fallback_top_level_bounds_empty_low_slice"
                )
        objects.append(record)
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
        "floor_z_m": floor_z_m,
        "objects": objects,
        "object_role_counts": role_counts,
        "source_reference": str(source),
        "claim_boundary": (
            "external InteriorAgent/Kujiale research scene; metadata contains "
            "only a room polygon, object bounds and descendant-mesh XY navigation "
            "footprints, not dataset geometry or textures"
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

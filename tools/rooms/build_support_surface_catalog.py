#!/usr/bin/env python3
"""Fit a room's wall and ceiling support surfaces from its own render surface.

A doorbell belongs on a wall and a smoke alarm on a ceiling, and until a room
says where its walls and ceilings are, a request that draws one of those assets
has nowhere to put it.  The two UE rooms had surfaces fitted from a single
depth readback; a Habitat room had none at all.  This reads them out of the
triangles every room package already carries and every part of the chain
already ray-traces, so the four families are on one route and a new room needs
no capture to join it.

A surface is a group of triangles that share a normal and a plane.  Walls are
the vertical groups, clipped to the height band a device is mounted in above
the navigable floor beneath them; ceilings are the horizontal groups a room
height above that floor.  Every surface names the triangle indices it was fitted
on, so the placement planner's own cross-check runs against the same geometry.
Nothing here is specific to a room, a map or an asset.

usage:
  build_support_surface_catalog.py --room-id ROOM --room-catalog CATALOG.json
                                   --asset-geometry MEASUREMENTS.json
                                   --output CATALOG_OUT.json
                                   [--surfaces-per-kind 6] [--max-triangles 4000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.capture.qa_plan_adapters import load_planning_resources_for_room  # noqa: E402
from avengine.rooms.conditioned_sampler import declared_floor_heights_m  # noqa: E402
from avengine.rooms.qa_episode import read_json  # noqa: E402

CATALOG_SCHEMA = "avengine_support_surface_catalog_v1"
GEOMETRY_DERIVATION = "room_render_surface_coplanar_cluster_v1"
# A mounted device lives between these heights above the floor it belongs to,
# and a room's ceiling is between these heights above the same floor. Both are
# statements about rooms people walk around in, not about any one map.
WALL_BAND_M = (0.9, 2.1)
CEILING_BAND_M = (2.2, 3.5)
# Grouping resolution: two triangles are on the same surface when their normals
# and their plane offsets agree to within these steps.
NORMAL_STEP = 0.05
OFFSET_STEP_M = 0.02
PLANE_TOLERANCE_M = 0.02
DEFAULT_PLACEMENT_CONFIG = {
    "normal_tolerance_deg": 8.0,
    "plane_tolerance_m": 0.05,
    "min_inter_instance_gap_m": 0.0,
    "candidate_search": {"grid_step_m": 0.25, "max_candidates": 64, "edge_margin_m": 0.05},
}


def _triangle_frames(vertices, triangles):
    corners = vertices[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > 1.0e-9
    unit = np.zeros_like(normals)
    unit[usable] = normals[usable] / lengths[usable, None]
    return unit, lengths / 2.0, corners.mean(axis=1), usable


def _clusters(unit, areas, centres, keep):
    """Coplanar, same-normal groups of the kept triangles, largest area first."""
    offsets = np.einsum("ij,ij->i", unit, centres)
    keys = np.column_stack([
        np.round(unit / NORMAL_STEP).astype(np.int64),
        np.round(offsets / OFFSET_STEP_M).astype(np.int64),
    ])
    grouped: dict[tuple, list[int]] = {}
    for index in np.flatnonzero(keep):
        grouped.setdefault(tuple(int(value) for value in keys[index]), []).append(int(index))
    return sorted(grouped.values(), key=lambda rows: -float(areas[rows].sum()))


def _plane_basis(normal):
    helper = np.asarray([0.0, 1.0, 0.0]) if abs(float(normal[1])) < 0.8 \
        else np.asarray([1.0, 0.0, 0.0])
    basis_u = np.cross(helper, normal)
    basis_u = basis_u / np.linalg.norm(basis_u)
    return basis_u, np.cross(normal, basis_u)


def _floor_below(height, floors, fallback):
    below = [value for value in floors if value <= height - 0.1]
    return max(below) if below else fallback


def _floor_for_band(height, floors, fallback, band):
    """The floor this surface belongs to, judged by the band it has to sit in.

    Taking the nearest floor below is wrong in a house with more than one
    level: a stair landing half way up counts as a floor, and it would make a
    ground-floor ceiling look like something a quarter of a metre above a
    landing instead of a room height above the ground. Any declared floor that
    puts this surface in the band is the floor it belongs to.
    """
    candidates = [value for value in list(floors) + [fallback]
                  if band[0] <= height - value <= band[1]]
    return max(candidates) if candidates else None


def fit_surface(vertices, triangles, rows, areas, unit, centres, *, surface_id,
                surface_kind, geometry_id, floors, mesh_bottom, max_triangles,
                band=None):
    """One support surface from one coplanar cluster, or ``None`` when it is unusable."""
    rows = np.asarray(sorted(rows, key=lambda index: -float(areas[index])), dtype=np.int64)
    weights = areas[rows]
    normal = (unit[rows] * weights[:, None]).sum(axis=0)
    normal = normal / np.linalg.norm(normal)
    origin = (centres[rows] * weights[:, None]).sum(axis=0) / weights.sum()
    basis_u, basis_v = _plane_basis(normal)
    # A triangle whose normal is merely close to the cluster's can still have a
    # vertex well off the fitted plane when it is metres long. The surface is
    # what lies on the plane, so those are dropped rather than widening the
    # tolerance the placement planner is later asked to trust.
    deviation = np.abs((vertices[triangles[rows]] - origin) @ normal).max(axis=1)
    rows = rows[deviation <= PLANE_TOLERANCE_M][:max_triangles]
    if not len(rows):
        return None
    floor = _floor_for_band(
        float(origin[1]), floors, mesh_bottom,
        (WALL_BAND_M[0] - 0.6, WALL_BAND_M[1] + 0.6) if band is not None else CEILING_BAND_M)
    if floor is None:
        floor = _floor_below(float(origin[1]), floors, mesh_bottom)
    bounds_v = None
    if band is not None and abs(float(basis_v[1])) > 0.5:
        low = (floor + band[0] - float(origin[1])) / float(basis_v[1])
        high = (floor + band[1] - float(origin[1])) / float(basis_v[1])
        bounds_v = sorted((low, high))
        inside = [index for index in rows
                  if np.all(bounds_v[0] <= (vertices[triangles[index]] - origin) @ basis_v)
                  and np.all((vertices[triangles[index]] - origin) @ basis_v <= bounds_v[1])]
        if not inside:
            return None
        rows = np.asarray(inside, dtype=np.int64)
    points = vertices[triangles[rows]].reshape(-1, 3) - origin
    along_u = points @ basis_u
    along_v = points @ basis_v
    if bounds_v is None:
        bounds_v = [float(along_v.min()), float(along_v.max())]
    else:
        bounds_v = [max(bounds_v[0], float(along_v.min())),
                    min(bounds_v[1], float(along_v.max()))]
    if bounds_v[0] >= bounds_v[1]:
        return None
    return {
        "surface_id": surface_id,
        "surface_kind": surface_kind,
        "origin_m": [float(value) for value in origin],
        "normal_m": [float(value) for value in normal],
        "basis_u_m": [float(value) for value in basis_u],
        "basis_v_m": [float(value) for value in basis_v],
        "bounds_u_m": [float(along_u.min()), float(along_u.max())],
        "bounds_v_m": [float(value) for value in bounds_v],
        "geometry_ref": {"authority": "visual_geometry", "geometry_id": geometry_id,
                         "triangle_indices": [int(value) for value in rows]},
        "fit": {
            "triangle_count": int(len(rows)),
            "retained_area_m2": float(areas[rows].sum()),
            "plane_tolerance_m": PLANE_TOLERANCE_M,
            "max_plane_deviation_m": float(np.abs(points @ normal).max()),
            "floor_below_m": float(floor),
            "height_above_floor_m": float(origin[1] - floor),
            "mounting_band_m": None if band is None else [floor + band[0], floor + band[1]],
        },
    }


def _smallest_registered_footprint(measurements, surface_kind):
    """The smallest in-plane extent any registered asset of this kind needs.

    A surface too small to hold the smallest thing that could be mounted on it
    is not a support surface, and keeping it would crowd out the real ones.
    The size comes from the measured assets, so it is a fact about the library
    rather than a number chosen for one room.
    """
    smallest = None
    for value in measurements.values():
        if not isinstance(value, dict) or value.get("support_kind") != surface_kind:
            continue
        low = value.get("bounds_min_m")
        high = value.get("bounds_max_m")
        normal = value.get("plane_normal_m")
        if low is None or high is None or normal is None:
            continue
        axes = [index for index in range(3) if abs(float(normal[index])) < 0.5]
        extent = max(float(high[index]) - float(low[index]) for index in axes)
        smallest = extent if smallest is None else min(smallest, extent)
    return smallest or 0.0


def build_catalog(room_id, request, *, asset_geometry, surfaces_per_kind, max_triangles):
    space, mesh, layout, resolution = load_planning_resources_for_room(room_id, request)
    room = dict(resolution.planning_room)
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    unit, areas, centres, usable = _triangle_frames(vertices, triangles)
    floors = list(declared_floor_heights_m(room, space))
    mesh_bottom = float(vertices[:, 1].min())
    geometry_id = f"{room_id}_render_surface"
    heights = centres[:, 1]
    # A ceiling is horizontal and a room height above the floor beneath it; a
    # wall is vertical and reaches into the band a device is mounted in.
    wall_band = (WALL_BAND_M[0] - 0.6, WALL_BAND_M[1] + 0.6)
    in_ceiling_band = np.asarray([
        _floor_for_band(float(value), floors, mesh_bottom, CEILING_BAND_M) is not None
        for value in heights])
    in_wall_band = np.asarray([
        _floor_for_band(float(value), floors, mesh_bottom, wall_band) is not None
        for value in heights])
    ceiling_keep = usable & (np.abs(unit[:, 1]) > 0.9) & in_ceiling_band
    wall_keep = usable & (np.abs(unit[:, 1]) < 0.2) & in_wall_band

    measurements = read_json(asset_geometry)["asset_visual_geometry_measurements"]
    margin = float(DEFAULT_PLACEMENT_CONFIG["candidate_search"]["edge_margin_m"])
    surfaces = []
    for kind, keep, band in (("wall", wall_keep, WALL_BAND_M), ("ceiling", ceiling_keep, None)):
        needed = _smallest_registered_footprint(measurements, kind) + 2.0 * margin
        fitted = []
        for cluster in _clusters(unit, areas, centres, keep):
            surface = fit_surface(
                vertices, triangles, cluster, areas, unit, centres,
                surface_id="pending", surface_kind=kind, geometry_id=geometry_id,
                floors=floors, mesh_bottom=mesh_bottom,
                max_triangles=max_triangles, band=band,
            )
            if surface is None:
                continue
            if (surface["bounds_u_m"][1] - surface["bounds_u_m"][0] < needed
                    or surface["bounds_v_m"][1] - surface["bounds_v_m"][0] < needed):
                continue
            fitted.append(surface)
            if len(fitted) >= surfaces_per_kind * 3:
                break
        fitted.sort(key=lambda row: -row["fit"]["retained_area_m2"])
        for index, surface in enumerate(fitted[:surfaces_per_kind], start=1):
            surface["surface_id"] = f"{room_id}_mesh_{kind}_{index:02d}"
            surface["room_id"] = room_id
            surface["fit"]["smallest_registered_footprint_m"] = needed
            surfaces.append(surface)
    if not surfaces:
        raise SystemExit(f"{room_id}: no wall or ceiling surface could be fitted")

    return _catalog(room_id, mesh, surfaces, floors, asset_geometry)


def _catalog(room_id, mesh, surfaces, floors, asset_geometry):
    measurements = read_json(asset_geometry)["asset_visual_geometry_measurements"]
    geometry_id = f"{room_id}_render_surface"
    return {
        "schema": CATALOG_SCHEMA,
        "room": {"room_id": room_id,
                 "coordinate_frame": {"handedness": "right", "linear_unit": "meter",
                                      "up_axis": "+Y"}},
        "layout": {"support_surfaces": surfaces},
        "visual_geometry": {
            "authority": "visual_geometry", "geometry_id": geometry_id,
            "source_ref": str(mesh.source["vertices"]),
            "vertices_path": str(mesh.source["vertices"]),
            "triangles_path": str(mesh.source["triangles"]),
            "coordinate_frame": {"handedness": "right", "linear_unit": "meter",
                                 "up_axis": "+Y"},
        },
        "asset_visual_geometry_measurements": measurements,
        "asset_geometry_catalog_ref": str(Path(asset_geometry).resolve()),
        "placement_config": DEFAULT_PLACEMENT_CONFIG,
        "measurement_contract": {
            "geometry_derivation": GEOMETRY_DERIVATION,
            "normal_step": NORMAL_STEP,
            "offset_step_m": OFFSET_STEP_M,
            "plane_tolerance_m": PLANE_TOLERANCE_M,
            "wall_band_above_floor_m": list(WALL_BAND_M),
            "ceiling_band_above_floor_m": list(CEILING_BAND_M),
            "declared_floor_heights_m": floors,
            "note": ("surfaces are coplanar same-normal clusters of the room package's "
                     "own render surface, so the named triangle indices are on the very "
                     "geometry the chain ray-traces"),
        },
        "claim_boundary": ("support surface fitting only; it makes no native visual "
                           "support, room collision or dataset admission claim"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--room-catalog", required=True, type=Path)
    parser.add_argument("--asset-geometry", required=True, type=Path,
                        help="an avengine geometry measurement catalog for the source assets")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--surfaces-per-kind", type=int, default=6)
    parser.add_argument("--max-triangles", type=int, default=4000)
    parser.add_argument("--request", type=Path,
                        help="an existing request whose runtime path bindings should be used")
    arguments = parser.parse_args()

    request = read_json(arguments.request) if arguments.request else {}
    request = dict(request)
    request["room_catalog"] = str(arguments.room_catalog.resolve())
    catalog = build_catalog(
        arguments.room_id, request, asset_geometry=arguments.asset_geometry,
        surfaces_per_kind=arguments.surfaces_per_kind,
        max_triangles=arguments.max_triangles,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "room_id": arguments.room_id,
        "output": str(arguments.output.resolve()),
        "surfaces": [
            {"surface_id": row["surface_id"], "surface_kind": row["surface_kind"],
             "origin_m": [round(value, 3) for value in row["origin_m"]],
             "normal_m": [round(value, 3) for value in row["normal_m"]],
             "height_above_floor_m": round(row["fit"]["height_above_floor_m"], 3),
             "retained_area_m2": round(row["fit"]["retained_area_m2"], 3),
             "triangle_count": row["fit"]["triangle_count"]}
            for row in catalog["layout"]["support_surfaces"]
        ],
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

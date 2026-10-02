"""Adapter from shared room_screening measurement to selection decisions.

No semantic GLB loader, furniture footprint algorithm or navmesh triangulator
is duplicated here. Scope, floor eligibility and placement stay in selection.
"""

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import shapely
from shapely.geometry import Polygon
from tools.rooms.room_screening.geometry import (
    load_semantic_ground_and_instances,
    parse_semantic_annotations,
    shape_preserving_projected_footprint,
    union_projected_polygons,
)
from tools.rooms.room_screening.compute_semantic_region_candidates import (
    navmesh_triangles,
)
from .geometry import floor_windows


def load_scene(scene_dir):
    sid = Path(scene_dir).name.split("-", 1)[1]
    result = load_semantic_ground_and_instances(Path(scene_dir), sid)
    ann, _, regions = parse_semantic_annotations(
        Path(scene_dir) / f"{sid}.semantic.txt"
    )
    return SimpleNamespace(
        ground=result[0],
        instances=result[2],
        annotations=ann,
        regions=regions,
        diagnostics=dict(
            measurement_backend="room_screening.geometry",
            face_count=result[8],
            unmapped_faces=result[4],
            unmapped_colours=result[5],
            semantic_hashes=result[6],
            semantic_glb=str(Path(scene_dir) / f"{sid}.semantic.glb"),
        ),
    )


def region_geometry(mesh, rid, p):
    members = {i: d for i, d in mesh.annotations.items() if d["region_id"] == rid}
    layers = floor_windows(mesh.ground.get(rid, []), p)
    bbox = None
    if layers:
        b = union_projected_polygons([l["geometry"] for l in layers]).bounds
        bbox = [[b[0], b[1]], [b[2], b[3]]]
    return layers, members, bbox


def layer_furniture(mesh, rid, floor_y, agent_height):
    """Smy shape-preserving, body-height clipped footprints; no hull bridging.

    Same-region and unassigned instances contribute known occupancy. Other
    regions are not silently subtracted. Heights, centres and IDs seed splits.
    """
    result = []
    for instance in mesh.instances:
        if instance["role"] != "blocker" or instance["region_id"] not in (rid, -1):
            continue
        tri = instance["triangles"]
        g = shape_preserving_projected_footprint(tri, floor_y, agent_height)
        if g.is_empty:
            continue
        result.append(
            dict(
                instance_id=instance["instance_id"],
                category=instance["category"],
                geometry=g,
                centre_xz_m=[float(g.centroid.x), float(g.centroid.y)],
                height_range_m=[float(tri[:, :, 1].min()), float(tri[:, :, 1].max())],
                footprint_method="room_screening.shape_preserving; navmesh agent body-height slab",
            )
        )
    return result


def navigation_geometry(nav_polygons, nav_heights, floor_y, scope, p, main_support):
    """Measure shared continuous triangles on the native graph's main component.

    Grid support is a connectivity proxy; navmesh/ground triangles own area.
    Keeping both measurements exposes boundary inflation by snap/grid cells.
    Largest projected scan fragment is reported but cannot replace native
    connectivity: tiny semantic scan cracks can split polygon components.
    """
    selected = [
        g
        for g, y in zip(nav_polygons, nav_heights)
        if abs(y - floor_y) <= p["floor_height_separation_m"]
    ]
    nav = (
        union_projected_polygons(selected).intersection(scope)
        if selected
        else Polygon()
    )
    parts = [
        g
        for g in (nav.geoms if hasattr(nav, "geoms") else [nav])
        if g.geom_type == "Polygon"
    ]
    largest = max(parts, key=lambda g: g.area, default=Polygon())
    main = nav.intersection(main_support)
    return dict(
        nav_main_area_m2=float(main.area),
        nav_triangle_intersection_area_m2=float(nav.area),
        nav_largest_projected_polygon_area_m2=float(largest.area),
        nav_polygon=shapely.geometry.mapping(nav),
        nav_main_polygon=shapely.geometry.mapping(main),
        nav_area_method="room_screening.navmesh_triangles continuous intersection; native geodesic grid main-component support",
    )

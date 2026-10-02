"""HM3D semantic-instance geometry helpers for provisional furniture audits."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

GROUND_CATEGORIES = {
    "floor", "carpet", "rug", "flooring", "floor mat", "mat", "doormat",
    "shower floor", "bathroom floor", "bath floor",
    "bath mat", "bathmat", "bathroom mat", "shower mat",
    "bathroom rug", "bath carpet", "bathroom carpet",
}
BLOCKER_TERMS = (
    "table", "chair", "armchair", "sofa", "couch", "bed", "sunbed", "lounger",
    "recliner", "chaise", "daybed", "cabinet", "shelf",
    "bookshelf", "bookcase", "shelving", "bedframe", "nightstand", "wardrobe",
    "dresser", "desk", "stool", "seat", "bench", "ottoman",
    "piano", "appliance", "refrigerator", "fridge", "oven", "stove",
    "sink", "washbasin", "toilet", "bathtub", "shower", "counter", "countertop", "worktop", "plant", "lamp",
    "fireplace", "washer", "dryer", "microwave", "basket", "hamper",
    "chest", "rack", "cart", "island", "vanity", "heater", "radiator",
    "boiler", "furnace", "playpen", "ladder", "dispenser", "trashcan",
    "trash", "bin", "crate", "box", "stand", "machine", "urinal", "bidet",
    "speaker stand", "vacuum", "ironing board", "flower stand", "flowerpot",
)
STRUCTURAL_CATEGORIES = {
    "wall", "ceiling", "floor", "flooring", "carpet", "rug", "floor mat", "mat", "doormat",
    "door", "window", "door/window", "door/window frame", "door frame", "window frame",
    "room", "region", "background", "stairs", "staircase", "stair step", "handrail",
    "railing", "stairs railing", "balustrade", "parapet", "pillar", "beam", "support beam",
    "partition", "window glass", "sliding glass door", "garage door", "closet door",
}


def classify_category(category: str) -> str:
    """Map a source label to blocker, ground, structural, or review-required."""
    name = " ".join(category.strip().lower().split())
    if name in GROUND_CATEGORIES:
        return "ground"
    if name in STRUCTURAL_CATEGORIES or name.startswith(("wall ", "ceiling ")):
        return "structural"
    words = set(re.findall(r"[a-z]+", name))
    if any(term in words for term in BLOCKER_TERMS):
        return "blocker"
    return "review"


def parse_semantic_labels(path: Path) -> dict[int, dict]:
    """Parse the HM3D semantic TXT without splitting quoted category names."""
    labels: dict[int, dict] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                instance_id = int(row[0].strip())
                colour_text = row[1].strip().lstrip("#").upper()
                region_id = int(row[3].strip())
            except (ValueError, IndexError):
                continue
            if len(colour_text) != 6 or any(ch not in "0123456789ABCDEF" for ch in colour_text):
                continue
            rgb = tuple(int(colour_text[i:i + 2], 16) for i in (0, 2, 4))
            category = row[2].strip().lower()
            labels[(rgb[0] << 16) | (rgb[1] << 8) | rgb[2]] = {
                "instance_id": instance_id,
                "category": category,
                "region_id": region_id,
                "role": classify_category(category),
                "rgb": colour_text,
            }
    return labels


def _clip_triangle_to_y_slab(triangle: np.ndarray, low: float, high: float) -> np.ndarray:
    """Return vertices of a triangle clipped to a closed horizontal y slab."""
    polygon = [np.asarray(point, dtype=np.float64) for point in triangle]
    for bound, keep_above in ((low, True), (high, False)):
        if not polygon:
            break
        clipped = []
        previous = polygon[-1]
        previous_inside = previous[1] >= bound if keep_above else previous[1] <= bound
        for current in polygon:
            current_inside = current[1] >= bound if keep_above else current[1] <= bound
            if current_inside != previous_inside:
                dy = current[1] - previous[1]
                if abs(dy) > 1e-15:
                    fraction = (bound - previous[1]) / dy
                    clipped.append(previous + fraction * (current - previous))
            if current_inside:
                clipped.append(current)
            previous = current
            previous_inside = current_inside
        polygon = clipped
    if not polygon:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(polygon, dtype=np.float64)


def conservative_projected_footprint(
    triangles: np.ndarray,
    floor_y: float,
    agent_height: float,
) -> Polygon:
    """Clip an instance mesh to agent body height, project to XZ, and hull it.

    A convex hull is intentionally conservative for incomplete scanned meshes.
    This is an audit proxy, not an exact solid collision footprint.
    """
    if triangles is None or len(triangles) == 0:
        return Polygon()
    low = float(floor_y)
    high = low + float(agent_height)
    tri = np.asarray(triangles, dtype=np.float64).reshape((-1, 3, 3))
    relevant = (tri[:, :, 1].max(axis=1) >= low) & (tri[:, :, 1].min(axis=1) <= high)
    points = []
    for triangle in tri[relevant]:
        clipped = _clip_triangle_to_y_slab(triangle, low, high)
        if len(clipped):
            points.extend(clipped[:, (0, 2)])
    if len(points) < 3:
        return Polygon()
    footprint = MultiPoint(np.asarray(points)).convex_hull
    return footprint if footprint.geom_type == "Polygon" else Polygon()


def shape_preserving_projected_footprint(
    triangles: np.ndarray,
    floor_y: float,
    agent_height: float,
) -> Polygon:
    """Estimate an instance footprint without bridging every concavity.

    The former convex hull can turn a concave object into a large solid
    footprint and incorrectly remove real floor. Here, projected surface
    triangles retain concavities; enclosed holes are filled because a table
    top or furniture body still blocks the person's volume. The convex
    footprint remains available separately as an uncertainty envelope; this
    function alone is not a collision-ground-truth claim.
    """
    if triangles is None or len(triangles) == 0:
        return Polygon()
    low = float(floor_y)
    high = low + float(agent_height)
    tri = np.asarray(triangles, dtype=np.float64).reshape((-1, 3, 3))
    relevant = (tri[:, :, 1].max(axis=1) >= low) & (tri[:, :, 1].min(axis=1) <= high)
    projected_faces = []
    for triangle in tri[relevant]:
        clipped = _clip_triangle_to_y_slab(triangle, low, high)
        if len(clipped):
            polygon = Polygon(clipped[:, (0, 2)])
            if polygon.is_valid and polygon.area > 1e-10:
                projected_faces.append(polygon)
    if not projected_faces:
        return Polygon()
    surface = unary_union(projected_faces)
    components = list(surface.geoms) if hasattr(surface, "geoms") else [surface]
    filled = []
    for component in components:
        if component.geom_type != "Polygon" or component.is_empty:
            continue
        # Fill interior scan holes (e.g. a tabletop kept as its perimeter),
        # while preserving the object's actual concave exterior boundary.
        polygon = Polygon(component.exterior)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.area > 1e-10:
            filled.append(polygon)
    if not filled:
        return Polygon()
    result = unary_union(filled)
    return result if result.geom_type in {"Polygon", "MultiPolygon"} else Polygon()


def projected_unmapped_surface_union(
    triangles: np.ndarray,
    floor_y: float,
    agent_height: float,
    query_bounds: tuple[float, float, float, float] | None = None,
):
    """Project unmatched faces locally, without treating them as one object.

    Unknown colors do not identify object instances. Keeping each face local
    avoids taking one convex hull around all unannotated faces in a scene. This
    geometry only flags manual review; it is never subtracted as a known object.
    """
    from shapely.ops import unary_union

    if triangles is None or len(triangles) == 0:
        return Polygon()
    tri = np.asarray(triangles, dtype=np.float64).reshape((-1, 3, 3))
    low, high = float(floor_y), float(floor_y) + float(agent_height)
    relevant = (tri[:, :, 1].max(axis=1) >= low) & (tri[:, :, 1].min(axis=1) <= high)
    if query_bounds is not None:
        qxmin, qzmin, qxmax, qzmax = query_bounds
        relevant &= (tri[:, :, 0].max(axis=1) >= qxmin) & (tri[:, :, 0].min(axis=1) <= qxmax)
        relevant &= (tri[:, :, 2].max(axis=1) >= qzmin) & (tri[:, :, 2].min(axis=1) <= qzmax)
    polygons = []
    for triangle in tri[relevant]:
        clipped = _clip_triangle_to_y_slab(triangle, low, high)
        if len(clipped) < 3:
            continue
        polygon = MultiPoint(clipped[:, (0, 2)]).convex_hull
        if polygon.geom_type == "Polygon" and polygon.area > 1e-10:
            polygons.append(polygon)
    return unary_union(polygons) if polygons else Polygon()


def load_semantic_ground_and_instances(scene_dir: Path, scene_id: str):
    """Load floor faces and per-colour semantic object triangles in Habitat axes."""
    # Keep geometry-only unit tests independent of AVEngine's optional runtime
    # dependencies; import its GLB loader only when processing a real scene.
    from avengine.acoustics.gltf import (
        extract_triangle_scene_document,
        load_glb_bytes,
        triangle_vertex_colours,
    )
    from avengine.acoustics.semantic import _linear_to_srgb_bytes

    semantic_glb = scene_dir / f"{scene_id}.semantic.glb"
    semantic_txt = scene_dir / f"{scene_id}.semantic.txt"
    labels = parse_semantic_labels(semantic_txt)
    glb_bytes = semantic_glb.read_bytes()
    semantic_hashes = {
        "semantic_glb_sha256": hashlib.sha256(glb_bytes).hexdigest(),
        "semantic_txt_sha256": hashlib.sha256(semantic_txt.read_bytes()).hexdigest(),
    }
    document = load_glb_bytes(glb_bytes, source_path=str(semantic_glb))
    scene = extract_triangle_scene_document(document)
    linear, _mixed = triangle_vertex_colours(document, scene)
    face_colours = _linear_to_srgb_bytes(linear).astype(np.int64)
    colour_codes = (face_colours[:, 0] << 16) | (face_colours[:, 1] << 8) | face_colours[:, 2]
    corners = scene.vertices[scene.triangles.astype(np.int64)].astype(np.float64)
    corners = np.stack((corners[..., 0], corners[..., 2], -corners[..., 1]), axis=-1)

    unique_codes, inverse, counts = np.unique(colour_codes, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)))
    ground_faces = defaultdict(list)
    all_regions = set()
    instance_groups = defaultdict(list)
    instance_metadata = {}
    unmapped_groups = []
    unmapped_face_count = 0
    unmapped_colour_count = 0
    palette_codes = np.asarray(sorted(labels), dtype=np.int64)
    palette = np.asarray([
        ((code >> 16) & 255, (code >> 8) & 255, code & 255)
        for code in palette_codes
    ], dtype=np.int64)
    for group_index, code in enumerate(unique_codes):
        first, last = int(starts[group_index]), int(starts[group_index + 1])
        group_triangles = corners[order[first:last]]
        label = labels.get(int(code))
        if label is None and int(code) != 0 and len(palette):
            colour = np.asarray(((int(code) >> 16) & 255, (int(code) >> 8) & 255, int(code) & 255), dtype=np.int64)
            distances = np.max(np.abs(palette - colour), axis=1)
            nearest = int(np.argmin(distances))
            # Match the ±2 channel tolerance already used by AVEngine's HM3D
            # semantic loader for color-space rounding drift.
            if int(distances[nearest]) <= 2:
                label = labels[int(palette_codes[nearest])]
        if label is None:
            unmapped_face_count += last - first
            unmapped_colour_count += 1
            unmapped_groups.append(group_triangles)
            continue
        region_id = label["region_id"]
        if region_id >= 0:
            all_regions.add(region_id)
        if label["role"] == "ground":
            if region_id < 0:
                continue
            for face in group_triangles:
                poly = Polygon(face[:, (0, 2)])
                if poly.is_empty or poly.area <= 1e-10:
                    continue
                ground_faces[region_id].append({
                    "polygon": poly,
                    "ys": float(face[:, 1].mean()),
                    "category": label["category"],
                })
            continue
        if label["role"] not in {"blocker", "review"}:
            continue
        key = ("instance", label["instance_id"])
        instance_metadata[key] = label
        instance_groups[key].append(group_triangles)
    instances = []
    for key, chunks in instance_groups.items():
        metadata = instance_metadata[key]
        instances.append({
            **metadata,
            "triangles": np.concatenate(chunks, axis=0),
            "face_count": int(sum(len(chunk) for chunk in chunks)),
        })
    unmapped_triangles = (
        np.concatenate(unmapped_groups, axis=0)
        if unmapped_groups else np.empty((0, 3, 3), dtype=np.float64)
    )
    return (
        ground_faces,
        all_regions,
        instances,
        unmapped_triangles,
        unmapped_face_count,
        unmapped_colour_count,
        semantic_hashes,
        len(labels),
        len(scene.triangles),
    )

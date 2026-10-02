"""Decision geometry only; shared screening owns semantic and area measurements."""

from pathlib import Path
import numpy as np
import trimesh
import shapely
from shapely.geometry import Polygon, GeometryCollection
from avengine.acoustics.gltf import load_glb_bytes, extract_triangle_scene_document
from tools.rooms.room_screening.geometry import (
    parse_semantic_annotations as hm3d_annotations,
    union_projected_polygons,
)


def canonical(vertices):
    return np.stack((vertices[..., 0], vertices[..., 2], -vertices[..., 1]), axis=-1)


def read_glb(path):
    document = load_glb_bytes(Path(path).read_bytes(), source_path=str(path))
    return document, extract_triangle_scene_document(document)


def projected_union(corners, parameters):
    """Thin compatibility wrapper around the shared projection union."""
    if not len(corners):
        return GeometryCollection()
    xz = np.asarray(corners)[..., [0, 2]]
    polygons = shapely.polygons(xz)
    selected = polygons[
        shapely.area(polygons) > parameters["projection_min_triangle_area_m2"]
    ]
    return union_projected_polygons(selected, parameters["projection_precision_m"])


def floor_windows(faces, p):
    """Peel area-weighted bounded height windows from smy's measured floor faces.

    Height classification remains a decision rule, independent of face-count
    median/chained provisional levels. Union measurement belongs to screening.
    """
    if not faces:
        return []
    heights = np.asarray([f["ys"] for f in faces])
    weights = np.asarray([f["polygon"].area for f in faces])
    remaining = np.argsort(heights, kind="stable")
    out = []
    while len(remaining):
        h = heights[remaining]
        ends = np.searchsorted(h, h + p["floor_height_separation_m"], side="right")
        cumulative = np.r_[0.0, np.cumsum(weights[remaining])]
        start = int(np.argmax(cumulative[ends] - cumulative[np.arange(len(h))]))
        end = int(ends[start])
        chosen = remaining[start:end]
        g = union_projected_polygons(
            [faces[i]["polygon"] for i in chosen], p["projection_precision_m"]
        )
        if not g.is_empty:
            w = weights[chosen]
            mi = min(int(np.searchsorted(np.cumsum(w), w.sum() / 2)), len(chosen) - 1)
            out.append(
                dict(
                    floor_y_m=float(heights[chosen[mi]]),
                    geometry=g,
                    height_range_m=[
                        float(heights[chosen].min()),
                        float(heights[chosen].max()),
                    ],
                    face_count=len(chosen),
                    projected_face_area_sum_m2=float(w.sum()),
                    height_method="maximum projected-area bounded window; weighted median",
                )
            )
        remaining = np.concatenate([remaining[:start], remaining[end:]])
    return sorted(out, key=lambda f: f["floor_y_m"])


def floor_layers(corners, parameters):
    """Synthetic-fixture adapter; production consumes shared ground faces."""
    faces = [
        dict(polygon=Polygon(tri[:, [0, 2]]), ys=float(tri[:, 1].mean()))
        for tri in corners
    ]
    return floor_windows(
        [
            f
            for f in faces
            if f["polygon"].area > parameters["projection_min_triangle_area_m2"]
        ],
        parameters,
    )


def dominant_layer(floors, minimum_fraction):
    if not floors:
        return None, 0.0
    areas = [f["metrics"]["floor_area_m2"] for f in floors]
    index = int(np.argmax(areas))
    fraction = areas[index] / sum(areas) if sum(areas) else 0.0
    return (index if fraction >= minimum_fraction else None), fraction


def short_side(geometry):
    if geometry.is_empty:
        return 0.0
    r = geometry.minimum_rotated_rectangle
    if r.geom_type != "Polygon":
        return 0.0
    p = np.array(r.exterior.coords)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).min())


def collision_mesh(scene_dir):
    scene_dir = Path(scene_dir)
    sid = scene_dir.name.split("-", 1)[-1]
    path = scene_dir / f"{sid}.glb"
    _, scene = read_glb(path)
    return trimesh.Trimesh(
        canonical(scene.vertices), scene.triangles, process=False
    ), str(path)


# Source: https://github.com/niessner/Matterport/blob/master/data_organization.md
MP3D_TYPES = {
    "a": "bathroom",
    "b": "bedroom",
    "c": "storage",
    "d": "dining",
    "e": "entryway",
    "f": "living",
    "g": "garage",
    "h": "corridor",
    "i": "library",
    "j": "laundry",
    "k": "kitchen",
    "l": "living",
    "m": "meeting",
    "n": "living",
    "o": "office",
    "p": "outdoor",
    "r": "recreation",
    "s": "stairs",
    "t": "bathroom",
    "u": "storage",
    "v": "living",
    "w": "gym",
    "x": "outdoor",
    "y": "outdoor",
    "z": "unknown",
    "Z": "scan_junk",
    "-": "unknown",
}


def infer_type(members, native_label=None):
    cats = {d["category"] for d in members.values()}
    if native_label is not None:
        return MP3D_TYPES.get(native_label, "unknown"), [
            "mp3d_region_label:" + native_label
        ]
    explicit = [
        ("stairs", {"stairs", "stair", "staircase", "steps"}),
        ("corridor", {"hallway", "corridor"}),
        ("garage", {"garage", "car", "garage door"}),
        ("storage", {"storage room", "closet room"}),
        ("outdoor", {"outdoor", "yard", "grass", "driveway", "patio"}),
    ]
    # A residential anchor plus stairs describes a mixed region, not a pure stairwell.
    anchors = cats & {
        "bed",
        "mattress",
        "sofa",
        "couch",
        "dining table",
        "desk",
        "stove",
        "oven",
    }
    for name, markers in explicit:
        found = cats & markers
        if found and not anchors:
            return name, sorted(found)
    for name, markers in [
        ("bedroom", {"bed", "mattress"}),
        ("kitchen", {"stove", "oven", "fridge", "refrigerator"}),
        ("living", {"sofa", "couch"}),
        ("bathroom", {"toilet", "bathtub", "shower"}),
        ("dining", {"dining table", "table"}),
        ("office", {"desk"}),
    ]:
        found = cats & markers
        if found:
            return name, sorted(found)
    return "unknown", []

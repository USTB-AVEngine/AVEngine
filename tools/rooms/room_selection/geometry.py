"""Semantic-instance geometry in Habitat's X,Y-up,Z metre frame."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import shapely
import trimesh
from shapely.geometry import GeometryCollection, Polygon

from avengine.acoustics.gltf import (
    extract_triangle_scene_document,
    load_glb_bytes,
    triangle_vertex_colours,
)
from avengine.acoustics.semantic import _linear_to_srgb_bytes


@dataclass
class SemanticMesh:
    vertices: np.ndarray
    faces: np.ndarray
    face_instances: np.ndarray
    instances: dict
    regions: dict
    diagnostics: dict

    @cached_property
    def corners(self):
        return self.vertices[self.faces]


def canonical(vertices):
    """Same source Z-up to Habitat Y-up rotation as the acoustic compiler."""
    return np.stack((vertices[..., 0], vertices[..., 2], -vertices[..., 1]), axis=-1)


def read_glb(path):
    document = load_glb_bytes(Path(path).read_bytes(), source_path=str(path))
    scene = extract_triangle_scene_document(document)
    return document, scene


def hm3d_annotations(path):
    instances, colours, regions = {}, {}, {}
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for row in csv.reader(stream, skipinitialspace=True):
            if len(row) < 4 or not row[0].strip().isdigit():
                continue
            iid, colour, category, region = (
                int(row[0]),
                row[1].strip(),
                row[2].strip().lower(),
                int(row[3]),
            )
            packed = int(colour, 16)
            if packed in colours and colours[packed] != iid:
                previous = colours[packed]
                colours[packed] = None
                for affected in [region] + (
                    [instances[previous]["region_id"]] if previous is not None else []
                ):
                    regions.setdefault(affected, {}).setdefault(
                        "ambiguous_colours", []
                    ).append(colour)
            else:
                colours[packed] = iid
            instances[iid] = dict(instance_id=iid, category=category, region_id=region)
            regions.setdefault(region, {})
    return instances, colours, regions


def load_hm3d(scene_dir):
    scene_dir = Path(scene_dir)
    sid = scene_dir.name.split("-", 1)[1]
    instances, colours, regions = hm3d_annotations(scene_dir / f"{sid}.semantic.txt")
    doc, mesh = read_glb(scene_dir / f"{sid}.semantic.glb")
    linear, mixed = triangle_vertex_colours(doc, mesh)
    rgb = _linear_to_srgb_bytes(linear).astype(np.int32)
    packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    keys = np.array(sorted(k for k, v in colours.items() if v is not None))
    values = np.array([colours[k] for k in keys], dtype=np.int32)
    positions = np.searchsorted(keys, packed)
    matched = (positions < len(keys)) & (
        keys[np.minimum(positions, len(keys) - 1)] == packed
    )
    ids = np.full(len(packed), -1, dtype=np.int32)
    ids[matched] = values[positions[matched]]
    return SemanticMesh(
        canonical(mesh.vertices),
        mesh.triangles.astype(np.int32),
        ids,
        instances,
        regions,
        dict(
            face_count=len(ids),
            unmapped_faces=int((~matched).sum()),
            mixed_vertex_colour_faces=int(mixed),
            semantic_glb=str(scene_dir / f"{sid}.semantic.glb"),
            annotation=str(scene_dir / f"{sid}.semantic.txt"),
        ),
    )


def mp3d_house(path):
    categories, objects, regions = {}, {}, {}
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "C":
            categories[int(f[1])] = f[5].replace("_", " ").lower()
        elif f[0] == "O":
            objects[int(f[1])] = (int(f[2]), int(f[3]))
        elif f[0] == "R":
            lo = np.array(f[9:12], float)
            hi = np.array(f[12:15], float)
            regions[int(f[1])] = dict(
                level_index=int(f[2]),
                region_label=f[5],
                bbox_xz_m=[
                    [float(lo[0]), float(-hi[1])],
                    [float(hi[0]), float(-lo[1])],
                ],
                floor_y_m=float(lo[2]),
            )
    instances = {
        iid: dict(instance_id=iid, region_id=r, category=categories.get(c, "unknown"))
        for iid, (r, c) in objects.items()
    }
    return instances, regions


def load_mp3d(scene_dir):
    from plyfile import PlyData

    scene_dir = Path(scene_dir)
    sid = scene_dir.name
    instances, regions = mp3d_house(scene_dir / f"{sid}.house")
    # Every downloaded semantic face has exactly three vertex indices. mmap
    # avoids copying the PLY payload; the identity column is preserved verbatim.
    ply = PlyData.read(
        str(scene_dir / f"{sid}_semantic.ply"),
        known_list_len={"face": {"vertex_indices": 3}},
    )
    v = ply["vertex"].data
    vertices = canonical(np.column_stack([v["x"], v["y"], v["z"]]))
    faces = np.asarray(ply["face"].data["vertex_indices"], dtype=np.int32)
    ids = np.asarray(ply["face"].data["object_id"], dtype=np.int32)
    mapped = np.isin(ids, list(instances))
    return SemanticMesh(
        vertices,
        faces,
        ids,
        instances,
        regions,
        dict(
            face_count=len(ids),
            unmapped_faces=int((~mapped).sum()),
            house=str(scene_dir / f"{sid}.house"),
            semantic_ply=str(scene_dir / f"{sid}_semantic.ply"),
            coordinate_transform="x,y,z -> x,z,-y",
        ),
    )


def projected_union(corners, parameters):
    """Union all non-degenerate projected triangles, including disconnected parts."""
    if not len(corners):
        return GeometryCollection()
    xz = corners[..., [0, 2]]
    u, v = xz[:, 1] - xz[:, 0], xz[:, 2] - xz[:, 0]
    area = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2
    xz = xz[area > parameters["projection_min_triangle_area_m2"]]
    if not len(xz):
        return GeometryCollection()
    polygons = shapely.polygons(xz)
    return shapely.union_all(polygons, grid_size=parameters["projection_precision_m"])


def floor_layers(corners, parameters):
    """Bound each cluster's height range; no transitive staircase bridging."""
    if not len(corners):
        return []
    heights = corners[:, :, 1].mean(axis=1)
    order = np.argsort(heights, kind="stable")
    out = []
    start = 0
    gap = parameters["floor_height_separation_m"]
    while start < len(order):
        end = start + 1
        while end < len(order) and heights[order[end]] - heights[order[start]] <= gap:
            end += 1
        chosen = order[start:end]
        g = projected_union(corners[chosen], parameters)
        if not g.is_empty:
            out.append(
                dict(
                    floor_y_m=float(np.median(heights[chosen])),
                    geometry=g,
                    height_range_m=[
                        float(heights[chosen].min()),
                        float(heights[chosen].max()),
                    ],
                    face_count=len(chosen),
                )
            )
        start = end
    return out


def short_side(geometry):
    if geometry.is_empty:
        return 0.0
    r = geometry.minimum_rotated_rectangle
    if r.geom_type != "Polygon":
        return 0.0
    p = np.array(r.exterior.coords)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).min())


def region_geometry(mesh, region, parameters):
    members = {iid: d for iid, d in mesh.instances.items() if d["region_id"] == region}
    floor_ids = [
        iid
        for iid, d in members.items()
        if d["category"] in parameters["floor_categories"]
    ]
    corners = mesh.corners
    floors = floor_layers(corners[np.isin(mesh.face_instances, floor_ids)], parameters)
    furniture = []
    # Group once rather than rescanning all triangles once per object.
    if not hasattr(mesh, "instance_order"):
        mesh.instance_order = np.argsort(mesh.face_instances, kind="stable")
    order = mesh.instance_order
    values = mesh.face_instances[order]
    for iid, d in members.items():
        if d["category"] not in parameters["furniture_categories"]:
            continue
        a, b = np.searchsorted(values, [iid, iid + 1])
        part = corners[order[a:b]]
        if not len(part):
            continue
        g = projected_union(part, parameters)
        if g.is_empty:
            continue
        furniture.append(
            dict(
                instance_id=iid,
                category=d["category"],
                geometry=g,
                centre_xz_m=[float(g.centroid.x), float(g.centroid.y)],
                height_range_m=[float(part[:, :, 1].min()), float(part[:, :, 1].max())],
            )
        )
    all_ids = list(members)
    part = corners[np.isin(mesh.face_instances, all_ids)]
    bbox = None
    if len(part):
        xz = part[..., [0, 2]]
        bbox = [xz.min(axis=(0, 1)).tolist(), xz.max(axis=(0, 1)).tolist()]
    return floors, furniture, members, bbox


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

"""Rank HM3D semantic categories by surface area, which is what acoustics sees.

The acoustic material lookup matches on category name, and the database shipped
with SoundSpaces carries 64 labels written for MP3D while HM3D's vocabulary is
far finer. Deciding which HM3D categories are worth mapping by counting
instances would be the wrong measure: a room holds one ceiling and forty
ornaments, and the ceiling is what the sound bounces off. So this weighs every
category by the triangle area carrying it.

Areas come out of the semantic glb rather than from the annotation text, because
the text says which instance is which category and says nothing about how much
of the room each instance is. Node transforms are applied while walking the
scene graph - skipping them leaves every mesh in its own local frame and the
areas are then wrong by whatever scale each node carries.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import struct
from pathlib import Path

import numpy as np

COMPONENT = {
    5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2),
    5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4),
}
COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load_glb(path: Path):
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise SystemExit(f"{path} is not a glb")
    json_length = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20 : 20 + json_length].decode("utf-8"))
    offset = 20 + json_length
    buffers = []
    while offset < len(data):
        length, kind = struct.unpack("<II", data[offset : offset + 8])
        chunk = data[offset + 8 : offset + 8 + length]
        if kind == 0x004E4942:
            buffers.append(chunk)
        offset += 8 + length
    return gltf, (buffers[0] if buffers else b"")


def accessor(gltf, blob, index):
    item = gltf["accessors"][index]
    fmt, size = COMPONENT[item["componentType"]]
    per = COUNT[item["type"]]
    view = gltf["bufferViews"][item["bufferView"]]
    start = view.get("byteOffset", 0) + item.get("byteOffset", 0)
    stride = view.get("byteStride") or size * per
    dtype = {"b": "<i1", "B": "<u1", "h": "<i2",
             "H": "<u2", "I": "<u4", "f": "<f4"}[fmt]
    if stride == size * per:
        flat = np.frombuffer(blob, dtype=dtype, count=item["count"] * per,
                             offset=start)
        return flat.reshape(item["count"], per).astype(
            np.float64 if fmt == "f" else np.int64
        )
    # Interleaved buffer view: take the rows out with a strided view rather than
    # unpacking each one, which on a 400k-vertex mesh is the whole runtime.
    raw = np.frombuffer(blob, dtype=np.uint8, offset=start,
                        count=stride * item["count"])
    picked = raw.reshape(item["count"], stride)[:, : size * per]
    return np.ascontiguousarray(picked).view(dtype).reshape(
        item["count"], per
    ).astype(np.float64 if fmt == "f" else np.int64)


def node_matrix(node):
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=float).reshape(4, 4).T
    matrix = np.eye(4)
    if "scale" in node:
        matrix[:3, :3] = np.diag(node["scale"])
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        matrix[:3, :3] = rotation @ matrix[:3, :3]
    if "translation" in node:
        matrix[:3, 3] = node["translation"]
    return matrix


def linear_to_srgb(x):
    """HM3D stores the instance colour linear; the annotation text lists sRGB.

    Hashing the linear bytes matches nothing at all - measured, zero of every
    face in two scenes - and the transfer curve is the whole difference. Getting
    this wrong is silent: every face simply falls out as unannotated.
    """

    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def category_by_colour(semantic_txt: Path):
    """Read the colour-to-category listing, skipping lines HM3D got wrong.

    One line in the released train split - 474,c,"radiator",11 in
    00546-nS8T59Aw3sf - carries a one-character colour, and int(h, 16) on it
    took this tool down 105 scenes into a 145 scene sweep. A malformed line
    costs one instance; refusing the file costs the scene.
    """

    mapping = {}
    skipped = 0
    for line in semantic_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(",", 3)
        if len(parts) < 3 or not parts[0].strip().isdigit():
            continue
        colour = parts[1].strip().upper()
        name = parts[2].strip().strip('"')
        if len(colour) != 6 or any(
            character not in "0123456789ABCDEF" for character in colour
        ):
            skipped += 1
            continue
        mapping[colour] = name
    if skipped:
        print(f"    skipped {skipped} malformed annotation line(s) in {semantic_txt.name}")
    return mapping


def scene_areas(glb: Path, semantic_txt: Path, tolerance: int = 2):
    gltf, blob = load_glb(glb)
    colours = category_by_colour(semantic_txt)
    reference_names = list(colours.values())
    reference_rgb = np.array(
        [[int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)] for h in colours],
        dtype=int,
    ) if colours else np.zeros((0, 3), dtype=int)
    areas = collections.Counter()
    unmatched = collections.Counter()

    def walk(index, parent):
        node = gltf["nodes"][index]
        world = parent @ node_matrix(node)
        if "mesh" in node:
            for primitive in gltf["meshes"][node["mesh"]]["primitives"]:
                attributes = primitive.get("attributes", {})
                if "POSITION" not in attributes or "COLOR_0" not in attributes:
                    continue
                points = accessor(gltf, blob, attributes["POSITION"])
                points = (world[:3, :3] @ points.T).T + world[:3, 3]
                raw = accessor(gltf, blob, attributes["COLOR_0"])
                byte = np.rint(
                    linear_to_srgb(raw[:, :3].astype(float) / 65535.0) * 255.0
                ).astype(int)
                if "indices" in primitive:
                    faces = accessor(gltf, blob, primitive["indices"]).reshape(-1, 3)
                else:
                    faces = np.arange(len(points)).reshape(-1, 3)
                a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
                face_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
                keys = byte[faces[:, 0]]
                # Resolve each distinct colour once, and allow a channel or two
                # of rounding: the transfer curve leaves a handful of values one
                # step off, and dropping those faces would quietly lose whole
                # instances.
                distinct = {tuple(v) for v in np.unique(keys, axis=0)}
                resolved = {}
                for value in distinct:
                    hexcode = "%02X%02X%02X" % value
                    name = colours.get(hexcode)
                    if name is None:
                        distances = np.abs(reference_rgb - np.asarray(value)).max(axis=1)
                        nearest = int(np.argmin(distances))
                        if distances[nearest] <= tolerance:
                            name = reference_names[nearest]
                    if name is None:
                        unmatched[hexcode] += 1
                    resolved[value] = name
                # Group by name and sum once per name instead of per face.
                order = {value: index for index, value in enumerate(distinct)}
                codes = np.array(
                    [order[tuple(row)] for row in np.unique(keys, axis=0)]
                )
                lookup = np.zeros(len(distinct), dtype=np.int64)
                names = [resolved[value] for value in distinct]
                index_of = {value: i for i, value in enumerate(distinct)}
                face_codes = np.fromiter(
                    (index_of[tuple(row)] for row in keys),
                    dtype=np.int64,
                    count=len(keys),
                )
                sums = np.zeros(len(distinct), dtype=np.float64)
                np.add.at(sums, face_codes, face_area)
                for position, name in enumerate(names):
                    if name is not None and sums[position] > 0:
                        areas[name] += float(sums[position])
                del codes, lookup
        for child in node.get("children", []):
            walk(child, world)

    for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        walk(root, np.eye(4))
    return areas, unmatched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hm3d-split-root", required=True, type=Path)
    parser.add_argument("--materials-json", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--scenes", type=int, default=0, help="0 for all")
    parser.add_argument("--top", type=int, default=60)
    args = parser.parse_args()

    labels = set()
    for material in json.loads(args.materials_json.read_text(encoding="utf-8"))[
        "materials"
    ]:
        labels.update(material.get("labels", []))

    total = collections.Counter()
    per_scene = 0
    for scene_dir in sorted(args.hm3d_split_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        stem = scene_dir.name.split("-", 1)[-1]
        glb = scene_dir / f"{stem}.semantic.glb"
        txt = scene_dir / f"{stem}.semantic.txt"
        if not glb.is_file() or not txt.is_file():
            continue
        areas, unmatched = scene_areas(glb, txt)
        total.update(areas)
        per_scene += 1
        print(f"  {scene_dir.name:<24} {len(areas):>4} categories  "
              f"{sum(areas.values()):9.1f} m2"
              + (f"  ({sum(unmatched.values())} faces with no annotation)"
                 if unmatched else ""))
        if args.scenes and per_scene >= args.scenes:
            break

    grand = sum(total.values())
    print(f"\n{per_scene} scenes, {len(total)} distinct categories, "
          f"{grand:.0f} m2 of annotated surface")
    print(f"\n{'category':<28} {'m2':>10} {'share':>7}  in the database?")
    rows = []
    covered = 0.0
    for name, area in total.most_common(args.top):
        matched = name in labels
        covered += area if matched else 0.0
        rows.append({"category": name, "area_m2": round(area, 1),
                     "share": round(area / grand, 5), "in_database": matched})
        print(f"{name:<28} {area:10.1f} {100 * area / grand:6.2f}%  "
              f"{'yes' if matched else 'NO'}")
    matched_all = sum(a for n, a in total.items() if n in labels)
    print(f"\narea already covered by the database: "
          f"{100 * matched_all / grand:.1f}%")
    if args.report:
        args.report.write_text(
            json.dumps(
                {
                    "schema": "avengine_hm3d_semantic_area_v1",
                    "scenes": per_scene,
                    "distinct_categories": len(total),
                    "annotated_area_m2": round(grand, 1),
                    "area_share_covered_by_database": round(matched_all / grand, 5),
                    "database_labels": sorted(labels),
                    "categories": [
                        {"category": n, "area_m2": round(a, 2),
                         "share": round(a / grand, 6), "in_database": n in labels}
                        for n, a in total.most_common()
                    ],
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

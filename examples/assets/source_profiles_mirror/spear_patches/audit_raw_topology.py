#!/usr/bin/env python3
"""Weld by position, then count boundary edges and connected components.

The five-view renders cannot answer "is this box open": the material is
doubleSided false, so a closed box seen from behind renders exactly like an
open one - you look through the culled rear face at the inside of the front.
And any per-edge measurement on a glTF has to weld first, because glTF splits
vertices at every UV and normal seam.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np


def load_glb(path: Path):
    data = path.read_bytes()
    json_len = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20 : 20 + json_len].decode("utf-8"))
    bin_offset = 20 + json_len + 8
    views = gltf["bufferViews"]
    accessors = gltf["accessors"]

    def read(accessor_index):
        accessor = accessors[accessor_index]
        view = views[accessor["bufferView"]]
        start = bin_offset + view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        dtype = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[
            accessor["componentType"]
        ]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
        count = accessor["count"] * width
        flat = np.frombuffer(data, dtype=dtype, count=count, offset=start)
        return flat.reshape(-1, width) if width > 1 else flat

    primitive = gltf["meshes"][0]["primitives"][0]
    return read(primitive["attributes"]["POSITION"]).astype(np.float64), read(
        primitive["indices"]
    ).astype(np.int64)


def components(vertex_count: int, edges: np.ndarray) -> int:
    parent = np.arange(vertex_count)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in edges:
        a, b = find(left), find(right)
        if a != b:
            parent[a] = b
    return len(np.unique([find(i) for i in range(vertex_count)]))


def audit(path: Path, quantum: float = 1.0e-5) -> dict:
    positions, indices = load_glb(path)
    keys = np.round(positions / quantum).astype(np.int64)
    _, welded, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    welded_positions = positions[welded]
    faces = inverse[indices].reshape(-1, 3)
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]

    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    extent = welded_positions.max(0) - welded_positions.min(0)
    return {
        "welded_vertices": int(len(welded_positions)),
        "faces": int(len(faces)),
        "boundary_edges": int((counts == 1).sum()),
        "manifold_edges": int((counts == 2).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "components": components(len(welded_positions), unique_edges),
        "extent_xyz": [round(float(value), 4) for value in extent],
        "aspect_longest_over_shortest": round(
            float(extent.max() / max(extent.min(), 1e-9)), 2
        ),
    }


def main() -> int:
    root = Path(sys.argv[1])
    for directory in sorted(root.iterdir()):
        glb = directory / "pixal_raw_1024.glb"
        if not glb.is_file():
            continue
        report = audit(glb)
        name = directory.name.replace("audio_playback_", "").replace("_product_view", "")
        print(
            f"{name:<38} faces={report['faces']:>7} bnd={report['boundary_edges']:>6} "
            f"nonmf={report['nonmanifold_edges']:>5} comp={report['components']:>4} "
            f"extent={report['extent_xyz']} aspect={report['aspect_longest_over_shortest']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

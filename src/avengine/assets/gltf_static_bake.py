"""Bake static glTF scene transforms while preserving original material bytes."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def _matrix(node):
    if "matrix" in node:
        return np.asarray(node["matrix"], float).reshape(4, 4).T
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    norm = x*x + y*y + z*z + w*w
    if norm < 1e-15:
        raise ValueError("zero glTF rotation quaternion")
    k = 2 / norm
    rotation = np.array([
        [1-k*(y*y+z*z), k*(x*y-z*w), k*(x*z+y*w)],
        [k*(x*y+z*w), 1-k*(x*x+z*z), k*(y*z-x*w)],
        [k*(x*z-y*w), k*(y*z+x*w), 1-k*(x*x+y*y)],
    ])
    result = np.eye(4)
    result[:3, :3] = rotation @ np.diag(node.get("scale", [1, 1, 1]))
    result[:3, 3] = node.get("translation", [0, 0, 0])
    return result


def bake_static_glb(source: Path, destination: Path) -> dict:
    """Resolve every active parent transform into a separate mesh's accessors.

    UE's mesh-only importer can discard outer physical-correction nodes. This
    copies transformed vertex attributes into new accessors and leaves UVs,
    textures, material definitions, and the original source file untouched.
    """
    original = source.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", original)
    if magic != b"glTF" or version != 2 or total != len(original):
        raise ValueError("not a complete binary glTF 2 file")
    offset = 12
    document = None
    binary = None
    while offset < len(original):
        length, kind = struct.unpack_from("<II", original, offset)
        chunk = original[offset+8:offset+8+length]
        if kind == 0x4E4F534A:
            document = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = bytearray(chunk)
        offset += 8 + length
    if document is None or binary is None:
        raise ValueError("static import requires embedded JSON and BIN")
    if document.get("animations") or document.get("skins"):
        raise ValueError("static baking cannot silently discard animation/skin")
    old_nodes = document["nodes"]
    old_meshes = document["meshes"]
    nodes, meshes, matrices, bounds = [], [], [], []

    def read_accessor(index):
        accessor = document["accessors"][index]
        if accessor["componentType"] != 5126 or "sparse" in accessor:
            raise ValueError("transformed attributes must be dense float32")
        view = document["bufferViews"][accessor["bufferView"]]
        if view.get("buffer", 0) != 0:
            raise ValueError("external attribute buffers are not supported")
        width = {"VEC3": 3, "VEC4": 4}[accessor["type"]]
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        stride = view.get("byteStride", width*4)
        return np.ndarray((accessor["count"], width), dtype="<f4",
                          buffer=binary, offset=start, strides=(stride, 4)).copy()

    def append_accessor(array, original_index):
        array = np.ascontiguousarray(array, dtype="<f4")
        padding = (-len(binary)) % 4
        binary.extend(b"\0" * padding)
        start = len(binary)
        binary.extend(array.tobytes())
        document["bufferViews"].append({
            "buffer": 0, "byteOffset": start, "byteLength": array.nbytes,
            "target": 34962,
        })
        accessor = deepcopy(document["accessors"][original_index])
        accessor.update(bufferView=len(document["bufferViews"])-1,
                        byteOffset=0, min=array.min(axis=0).tolist(),
                        max=array.max(axis=0).tolist())
        document["accessors"].append(accessor)
        return len(document["accessors"])-1

    def visit(index, parent, ancestors):
        if index in ancestors:
            raise ValueError("cyclic glTF node graph")
        node = old_nodes[index]
        world = parent @ _matrix(node)
        if "mesh" in node:
            mesh = deepcopy(old_meshes[node["mesh"]])
            for primitive in mesh["primitives"]:
                if primitive.get("targets"):
                    raise ValueError("static morph targets need explicit baking")
                attributes = primitive["attributes"]
                for semantic in ["POSITION", "NORMAL", "TANGENT"]:
                    if semantic not in attributes:
                        continue
                    old_index = attributes[semantic]
                    values = read_accessor(old_index)
                    if semantic == "POSITION":
                        values = values @ world[:3, :3].T + world[:3, 3]
                        bounds.append(np.stack([values.min(axis=0), values.max(axis=0)]))
                    else:
                        transformed = (values[:, :3] @ np.linalg.inv(world[:3, :3])
                                       if semantic == "NORMAL" else values[:, :3] @ world[:3, :3].T)
                        transformed /= np.maximum(np.linalg.norm(
                            transformed, axis=1, keepdims=True), 1e-15)
                        values[:, :3] = transformed
                        if semantic == "TANGENT":
                            values[:, 3] *= np.sign(np.linalg.det(world[:3, :3]))
                    attributes[semantic] = append_accessor(values, old_index)
            meshes.append(mesh)
            nodes.append({"name": node.get("name", f"mesh_{index}"),
                          "mesh": len(meshes)-1})
            matrices.append(world.tolist())
        for child in node.get("children", []):
            visit(child, world, ancestors | {index})

    scene = document["scenes"][document.get("scene", 0)]
    for node in scene.get("nodes", []):
        visit(node, np.eye(4), set())
    if not meshes:
        raise ValueError("active static scene has no mesh")
    document.update(nodes=nodes, meshes=meshes,
                    scenes=[{"nodes": list(range(len(nodes)))}], scene=0)
    document["buffers"] = [{"byteLength": len(binary)}]
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 4)
    binary.extend(b"\0" * ((-len(binary)) % 4))
    payload = (struct.pack("<4sII", b"glTF", 2, 28+len(encoded)+len(binary))
               + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
               + struct.pack("<II", len(binary), 0x004E4942) + binary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    lo = np.asarray(bounds)[:, 0].min(axis=0)
    hi = np.asarray(bounds)[:, 1].max(axis=0)
    return {
        "source": str(source), "source_sha256": hashlib.sha256(original).hexdigest(),
        "derived": str(destination),
        "derived_sha256": hashlib.sha256(payload).hexdigest(),
        "baked_mesh_count": len(meshes), "baked_node_matrices": matrices,
        "bounds_min_m": lo.tolist(), "bounds_max_m": hi.tolist(),
        "dimensions_m": (hi-lo).tolist(), "material_and_image_definitions_preserved": True,
    }

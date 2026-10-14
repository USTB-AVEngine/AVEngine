"""Static scene corrections must survive mesh-only UE import staging."""
import json
from pathlib import Path
import struct

import numpy as np

from avengine.assets.gltf_static_bake import bake_static_glb


def _file(path, nodes):
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype="<f4")
    image = b"texture-image-preserved"
    binary = positions.tobytes() + image
    value = {
        "asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteLength": positions.nbytes},
                        {"buffer": 0, "byteOffset": positions.nbytes,
                         "byteLength": len(image)}],
        "accessors": [{"bufferView": 0, "componentType": 5126,
                       "count": 3, "type": "VEC3"}],
        "images": [{"bufferView": 1, "mimeType": "image/png"}],
        "materials": [{"name": "owner-approved-color", "doubleSided": True}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "nodes": nodes, "scenes": [{"nodes": [0]}], "scene": 0,
    }
    encoded = json.dumps(value).encode()
    encoded += b" " * ((-len(encoded)) % 4)
    binary += b"\0" * ((-len(binary)) % 4)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 28+len(encoded)+len(binary))
                     + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
                     + struct.pack("<II", len(binary), 0x004E4942) + binary)
    return value, positions, image


def _read(path):
    data = path.read_bytes()
    size = struct.unpack_from("<I", data, 12)[0]
    value = json.loads(data[20:20+size])
    binary = data[28+size:]
    vertices = []
    for mesh in value["meshes"]:
        accessor = value["accessors"][mesh["primitives"][0]["attributes"]["POSITION"]]
        view = value["bufferViews"][accessor["bufferView"]]
        vertices.append(np.frombuffer(binary, "<f4", count=accessor["count"]*3,
                                      offset=view["byteOffset"]).reshape(-1, 3))
    return value, binary, vertices


def test_parent_scale_rotation_and_translation_are_baked_without_texture_changes(tmp_path):
    source, destination = tmp_path/"original.glb", tmp_path/"derived.glb"
    value, positions, image = _file(source, [
        {"translation": [2, 3, 4], "scale": [2, 1, 3], "children": [1]},
        {"rotation": [0, np.sqrt(.5), 0, np.sqrt(.5)], "mesh": 0},
    ])
    original = source.read_bytes()
    receipt = bake_static_glb(source, destination)
    derived, binary, meshes = _read(destination)
    # Parent nonuniform scale follows the child rotation; reversing them is wrong.
    expected = np.array([[2, 3, 4], [2, 3, 1], [4, 4, 4]], float)
    np.testing.assert_allclose(meshes[0], expected, atol=1e-6)
    assert derived["materials"] == value["materials"]
    assert derived["images"] == value["images"]
    assert binary[positions.nbytes:positions.nbytes+len(image)] == image
    assert source.read_bytes() == original
    assert receipt["material_and_image_definitions_preserved"]


def test_instanced_mesh_gets_separate_world_coordinates(tmp_path):
    source, destination = tmp_path/"original.glb", tmp_path/"derived.glb"
    _, positions, _ = _file(source, [
        {"children": [1, 2]}, {"mesh": 0, "translation": [1, 0, 0]},
        {"mesh": 0, "translation": [-2, 0, 0]},
    ])
    bake_static_glb(source, destination)
    value, _, meshes = _read(destination)
    assert len(value["meshes"]) == 2
    np.testing.assert_allclose(meshes[0], positions+[1, 0, 0])
    np.testing.assert_allclose(meshes[1], positions+[-2, 0, 0])

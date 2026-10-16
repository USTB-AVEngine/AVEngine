from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from avengine.acoustics.gltf import GltfError, extract_triangle_scene


_INDEX_DTYPE = {
    5121: np.dtype("u1"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
}


def _glb_bytes(
    *,
    index_component_type: int = 5123,
    nodes: list[dict] | None = None,
    required_extensions: list[str] | None = None,
    node_extension: bool = False,
    sparse_position: bool = False,
    positions: np.ndarray | None = None,
) -> bytes:
    if positions is None:
        positions = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype="<f4",
        )
    positions = np.ascontiguousarray(positions, dtype="<f4")
    indices = np.asarray([0, 1, 2], dtype=_INDEX_DTYPE[index_component_type])
    position_bytes = positions.tobytes()
    index_offset = (len(position_bytes) + 3) & ~3
    binary = position_bytes + b"\x00" * (index_offset - len(position_bytes))
    binary += indices.tobytes()
    declared_binary_length = len(binary)
    binary += b"\x00" * ((-len(binary)) % 4)
    if nodes is None:
        nodes = [{"mesh": 0}]
    if node_extension:
        nodes[0]["extensions"] = {
            "EXT_mesh_gpu_instancing": {"attributes": {"TRANSLATION": 0}}
        }
    position_accessor: dict = {
        "bufferView": 0,
        "componentType": 5126,
        "count": len(positions),
        "type": "VEC3",
    }
    if sparse_position:
        position_accessor["sparse"] = {"count": 1}
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "materials": [{"name": "TestSurface"}],
        "accessors": [
            position_accessor,
            {
                "bufferView": 1,
                "componentType": index_component_type,
                "count": 3,
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes)},
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": len(indices.tobytes()),
            },
        ],
        "buffers": [{"byteLength": declared_binary_length}],
    }
    if required_extensions is not None:
        document["extensionsRequired"] = required_extensions
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def _write_glb(tmp_path: Path, payload: bytes) -> Path:
    path = tmp_path / "fixture.glb"
    path.write_bytes(payload)
    return path


@pytest.mark.parametrize("component_type", [5121, 5123, 5125])
def test_triangle_extractor_accepts_all_gltf_unsigned_index_types(
    tmp_path: Path, component_type: int
) -> None:
    scene = extract_triangle_scene(
        _write_glb(tmp_path, _glb_bytes(index_component_type=component_type))
    )

    assert scene.vertices.dtype.str == "<f4"
    assert scene.triangles.dtype.str == "<u4"
    assert scene.triangles.tolist() == [[0, 1, 2]]
    assert scene.triangle_source_material_names == ("TestSurface",)


def test_triangle_extractor_expands_trs_matrix_instances_and_preserves_winding(
    tmp_path: Path,
) -> None:
    nodes = [
        {"mesh": 0, "translation": [1.0, 0.0, 0.0], "scale": [-1.0, 1.0, 1.0]},
        {
            "mesh": 0,
            "matrix": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                2.0,
                0.0,
                1.0,
            ],
        },
    ]
    scene = extract_triangle_scene(_write_glb(tmp_path, _glb_bytes(nodes=nodes)))

    assert scene.vertices.shape == (6, 3)
    assert scene.triangles.shape == (2, 3)
    assert scene.source_node_instance_count == 2
    assert len(scene.objects) == 2
    first = scene.vertices[scene.triangles[0]].astype(float)
    assert np.cross(first[1] - first[0], first[2] - first[0])[2] > 0
    assert np.allclose(scene.vertices[3:].min(axis=0), [0.0, 2.0, 0.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"required_extensions": ["KHR_draco_mesh_compression"]}, "required GLB extensions"),
        ({"node_extension": True}, "instance-affecting extensions"),
        ({"sparse_position": True}, "sparse storage"),
    ],
)
def test_triangle_extractor_fails_closed_on_unreplayed_geometry_semantics(
    tmp_path: Path, kwargs: dict, message: str
) -> None:
    with pytest.raises(GltfError, match=message):
        extract_triangle_scene(_write_glb(tmp_path, _glb_bytes(**kwargs)))


def test_triangle_extractor_rejects_float32_overflow_after_transform(
    tmp_path: Path,
) -> None:
    positions = np.asarray(
        [[3.0e38, 0.0, 0.0], [3.0e38, 1.0, 0.0], [3.0e38, 0.0, 1.0]],
        dtype="<f4",
    )
    nodes = [{"mesh": 0, "scale": [2.0, 2.0, 2.0]}]

    with pytest.raises(GltfError, match="overflow float32"):
        extract_triangle_scene(
            _write_glb(tmp_path, _glb_bytes(nodes=nodes, positions=positions))
        )

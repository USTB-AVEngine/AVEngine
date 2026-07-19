from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest

from avengine.optional_backends.spear_replicacad_glb import (
    ReplicaCADGLBError,
    normalize_replicacad_glb,
    prepare_replicacad_source_glbs,
)


def _build_fixture(
    path: Path,
    *,
    shared_mesh_node: bool = False,
    scale: tuple[float, float, float] = (2, 3, 4),
) -> bytes:
    positions = (0, 0, 0, 1, 0, 0, 0, 1, 0)
    normals = (0, 0, 1) * 3
    tangents = (1, 0, 0, 1) * 3
    binary = struct.pack("<9f9f12f", *positions, *normals, *tangents)
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0] + ([1] if shared_mesh_node else [])}],
        "nodes": [
            {
                "name": "triangle",
                "mesh": 0,
                "translation": [1, 2, 3],
                "scale": list(scale),
            }
        ]
        + ([{"name": "duplicate", "mesh": 0}] if shared_mesh_node else []),
        "meshes": [
            {
                "name": "triangle",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "TANGENT": 2},
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [{"name": "retained_pbr"}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 36},
            {"buffer": 0, "byteOffset": 72, "byteLength": 48},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [1, 1, 0],
            },
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4"},
        ],
        "buffers": [{"byteLength": len(binary)}],
    }
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 4)
    binary += b"\0" * ((-len(binary)) % 4)
    payload = b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(encoded) + 8 + len(binary)),
            struct.pack("<II", len(encoded), 0x4E4F534A),
            encoded,
            struct.pack("<II", len(binary), 0x004E4942),
            binary,
        )
    )
    path.write_bytes(payload)
    return payload


def _read(path: Path) -> tuple[dict, bytes]:
    payload = path.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20 : 20 + json_length].decode().rstrip(" \0"))
    binary_offset = 20 + json_length + 8
    return document, payload[binary_offset:]


def test_bakes_complete_node_transform_without_changing_source(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    original = _build_fixture(source)
    destination = tmp_path / "prepared.glb"

    evidence = normalize_replicacad_glb(source, destination)

    assert source.read_bytes() == original
    assert evidence["mesh_count"] == 1
    assert evidence["transformed_accessor_count"] == 3
    assert evidence["position_bounds_gltf_m"] == {
        "minimum": [1.0, 2.0, 3.0],
        "maximum": [3.0, 5.0, 3.0],
    }
    document, binary = _read(destination)
    assert document["nodes"] == [{"name": "triangle", "mesh": 0}]
    assert document["materials"] == [{"name": "retained_pbr"}]
    assert struct.unpack_from("<9f", binary, 0) == pytest.approx(
        (1, 2, 3, 3, 2, 3, 1, 5, 3)
    )
    assert struct.unpack_from("<9f", binary, 36) == pytest.approx((0, 0, 1) * 3)
    assert struct.unpack_from("<12f", binary, 72) == pytest.approx((1, 0, 0, 1) * 3)


def test_rejects_one_mesh_referenced_by_multiple_nodes(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    _build_fixture(source, shared_mesh_node=True)

    with pytest.raises(ReplicaCADGLBError, match="exactly one mesh node"):
        normalize_replicacad_glb(source, tmp_path / "prepared.glb")


def test_rejects_reflection_without_silently_breaking_triangle_winding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.glb"
    _build_fixture(source, scale=(-1, 1, 1))

    with pytest.raises(ReplicaCADGLBError, match="winding reversal"):
        normalize_replicacad_glb(source, tmp_path / "prepared.glb")


def test_prepares_every_request_source_and_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.glb"
    _build_fixture(source)
    request = {
        "pbr_import": {
            "source_meshes": [
                {
                    "mesh_source_id": "mesh_source_000",
                    "source_glb_path": str(source),
                    "source_inventory": {"mesh_count": 1},
                }
            ]
        }
    }
    output = tmp_path / "prepared"

    result = prepare_replicacad_source_glbs(request, output)

    source_record = result["pbr_import"]["source_meshes"][0]
    assert Path(source_record["editor_import_source_glb_path"]).is_file()
    assert result["glb_preparation"]["source_glb_count"] == 1
    assert "editor_import_source_glb_path" not in request["pbr_import"]["source_meshes"][0]
    with pytest.raises(FileExistsError):
        prepare_replicacad_source_glbs(request, output)

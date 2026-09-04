from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from avengine.assets.mesh_quality import (
    MeshQualityError,
    load_glb_mesh,
    measure_mesh_quality,
)


GLB_JSON = 0x4E4F534A
GLB_BIN = 0x004E4942


def _build_glb(document: dict, binary: bytes) -> bytes:
    json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary += b"\x00" * ((-len(binary)) % 4)
    chunks = (
        struct.pack("<II", len(json_bytes), GLB_JSON)
        + json_bytes
        + struct.pack("<II", len(binary), GLB_BIN)
        + binary
    )
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (10.0, 0.0, 0.0),
            (11.0, 0.0, 0.0),
            (10.0, 1.0, 0.0),
            (20.0, 0.0, 0.0),
            (20.0 + 1.0e-8, 0.0, 0.0),
            (20.0, 1.0e-8, 0.0),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 1, 2], [1, 3, 2], [4, 5, 6], [7, 8, 9]],
        dtype=np.int32,
    )
    return vertices, faces


def test_measurement_is_unclassified_and_chunk_invariant() -> None:
    vertices, faces = _fixture()
    one = measure_mesh_quality(
        vertices,
        faces,
        tiny_area_threshold=1.0e-12,
        small_component_max_faces=1,
        chunk_size=1,
    )
    many = measure_mesh_quality(
        vertices,
        faces,
        tiny_area_threshold=1.0e-12,
        small_component_max_faces=1,
        chunk_size=100,
    )
    assert one["status"] == "measured_unclassified"
    assert {
        key: value for key, value in one["metrics"].items() if key != "chunk_size"
    } == {
        key: value for key, value in many["metrics"].items() if key != "chunk_size"
    }
    assert one["metrics"]["vertices"] == 10
    assert one["metrics"]["faces"] == 4
    assert one["metrics"]["tiny_face_count"] == 1
    assert one["metrics"]["connected_component_count"] == 3
    assert one["metrics"]["small_component_count"] == 2
    assert one["metrics"]["small_component_faces"] == 2
    assert one["metrics"]["largest_component_faces"] == 2
    assert one["metrics"]["largest_component_fraction"] == 0.5
    assert one["metrics"]["face_component_consistent"] is True
    assert one["mutation"] == {"input_modified": False, "components_deleted": False}


def test_explicit_policy_classifies_without_global_thresholds(tmp_path: Path) -> None:
    vertices, faces = _fixture()
    policy = {
        "schema": "avengine_mesh_quality_policy_v1",
        "asset_category": "unit_fixture",
        "measurement": {
            "tiny_face_area_threshold": 1.0e-12,
            "small_component_max_faces": 1,
        },
        "limits": {
            "max_tiny_faces": 1,
            "max_small_component_count": 2,
            "max_small_component_faces": 2,
            "min_largest_component_fraction": 0.5,
            "require_support_plane": False,
        },
    }
    report = measure_mesh_quality(vertices, faces, quality_policy=policy)
    assert report["status"] == "pass"
    assert all(report["policy_checks"].values())

    review = dict(policy)
    review["limits"] = dict(policy["limits"])
    review["limits"]["max_tiny_faces"] = 0
    report = measure_mesh_quality(vertices, faces, quality_policy=review)
    assert report["status"] == "review_required"
    assert report["policy_checks"]["max_tiny_faces"] is False

    support = tmp_path / "level.json"
    support.write_text("{}", encoding="utf-8")
    support_report = measure_mesh_quality(
        vertices,
        faces,
        support_plane_path=support,
        quality_policy=policy,
    )
    assert support_report["support_plane"]["present"] is True
    assert support_report["policy_checks"]["require_support_plane"] is False


def test_policy_rejects_unknown_limit() -> None:
    vertices, faces = _fixture()
    with pytest.raises(MeshQualityError, match="unknown limits"):
        measure_mesh_quality(
            vertices,
            faces,
            quality_policy={
                "schema": "avengine_mesh_quality_policy_v1",
                "asset_category": "fixture",
                "limits": {"max_magic": 1},
            },
        )


def test_load_glb_mesh_decodes_uint_indices_and_node_transform(tmp_path: Path) -> None:
    positions = struct.pack(
        "<fffffffff",
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    indices = struct.pack("<HHH", 0, 1, 2)
    binary = positions + indices
    document = {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["EXT_texture_webp"],
        "extensionsRequired": ["EXT_texture_webp"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "translation": [2.0, 3.0, 4.0]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
    }
    path = tmp_path / "triangle.glb"
    path.write_bytes(_build_glb(document, binary))
    geometry = load_glb_mesh(path)
    np.testing.assert_allclose(
        geometry.vertices,
        np.asarray([(2.0, 3.0, 4.0), (3.0, 3.0, 4.0), (2.0, 4.0, 4.0)]),
    )
    np.testing.assert_array_equal(geometry.faces, np.asarray([[0, 1, 2]], dtype=np.int32))
    assert geometry.primitive_count == 1

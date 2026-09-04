from __future__ import annotations

import hashlib
import importlib
import json
import struct
from pathlib import Path

import pytest


IMPORTER = importlib.import_module("tools.ue.import_controlled_humans_editor")


def _write_glb(path: Path) -> None:
    document = {
        "asset": {"version": "2.0"},
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "TEXCOORD_0": 1,
                            "JOINTS_0": 2,
                            "WEIGHTS_0": 3,
                        }
                    }
                ]
            }
        ],
        "skins": [{"joints": [0, 1]}],
        "animations": [{"name": "Walk"}, {"name": "Idle"}],
        "materials": [{"name": "violet_body"}],
        "images": [{"name": "violet_color", "mimeType": "image/png", "bufferView": 0}],
        "textures": [{"source": 0}],
        "buffers": [{"byteLength": 0}],
        "nodes": [
            {"mesh": 0, "skin": 0},
            {
                "name": "SkeletonRoot",
                "scale": [1.0, 1.0, 1.0],
                "translation": [0.0, 0.0, 0.0],
            },
        ],
        "scenes": [{"nodes": [0, 1]}],
        "scene": 0,
    }
    raw = json.dumps(document, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 4)
    payload = b"glTF" + struct.pack("<II", 2, 20 + len(raw)) + struct.pack(
        "<II", len(raw), IMPORTER._GLTF_JSON_CHUNK
    ) + raw
    path.write_bytes(payload)


def test_glb_readback_uses_catalog_values(tmp_path: Path) -> None:
    glb = tmp_path / "violet.glb"
    _write_glb(glb)
    contract = {
        "expected_primitive_count": 1,
        "required_primitive_attributes": [
            "POSITION",
            "TEXCOORD_0",
            "JOINTS_0",
            "WEIGHTS_0",
        ],
        "expected_bone_count": 2,
        "required_animation_names": ["Walk", "Idle"],
        "expected_material_names": ["violet_body"],
        "expected_image_names": ["violet_color"],
        "expected_texture_count": 1,
        "expected_skeleton_family": "SkeletonRoot",
    }

    observed = IMPORTER._read_glb_contract(glb, contract)
    assert observed["joint_count"] == 2
    assert observed["animation_names"] == ["Idle", "Walk"]
    assert observed["material_names"] == ["violet_body"]
    assert observed["image_names"] == ["violet_color"]


def test_source_manifest_reconciles_runtime_bytes_without_a_static_hash(
    tmp_path: Path,
) -> None:
    glb = tmp_path / "violet.glb"
    _write_glb(glb)
    manifest = tmp_path / "normalization_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "source-violet-v1",
                "tag": "human_top_violet_v1",
                "asset_id": "human_violet",
                "runtime_glb": {
                    "filename": glb.name,
                    "sha256": hashlib.sha256(glb.read_bytes()).hexdigest(),
                    "size_bytes": glb.stat().st_size,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    contract = {
        "tag": "human_top_violet_v1",
        "asset_id": "human_violet",
        "source_manifest_schema": "source-violet-v1",
        "normalization_schema": None,
        "actor_scale": 1.0,
        "requires_in_place_actions": False,
    }

    observed = IMPORTER._validate_source_manifest(manifest, glb, contract)
    assert observed["asset_id"] == "human_violet"


def test_editor_script_is_importable_without_project_extension() -> None:
    source = Path(IMPORTER.__file__).read_text(encoding="utf-8")
    assert "import spear" not in source
    assert "from avengine.assets import controlled_humans" in source
    assert "_delete_directories" not in source
    assert IMPORTER.unreal is None or hasattr(IMPORTER.unreal, "AssetImportTask")


def test_glb_rejects_missing_catalog_material(tmp_path: Path) -> None:
    glb = tmp_path / "violet.glb"
    _write_glb(glb)
    contract = {
        "expected_primitive_count": 1,
        "required_primitive_attributes": ["POSITION", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"],
        "expected_bone_count": 2,
        "required_animation_names": ["Walk", "Idle"],
        "expected_material_names": ["another_body"],
        "expected_image_names": ["violet_color"],
        "expected_texture_count": 1,
        "expected_skeleton_family": "SkeletonRoot",
    }
    with pytest.raises(RuntimeError, match="materials differ"):
        IMPORTER._read_glb_contract(glb, contract)


def test_import_manifest_is_readable_json_and_does_not_replace_existing(tmp_path):
    manifest = tmp_path / "saved" / "ue_import_manifest.json"
    payload = {"asset_id": "human_violet", "animations": ["idle", "walk"]}
    IMPORTER._write_json_no_replace(manifest, payload)
    assert json.loads(manifest.read_text()) == payload
    assert manifest.read_bytes().endswith(b"\n")
    original = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="refusing to replace"):
        IMPORTER._write_json_no_replace(manifest, {"asset_id": "other"})
    assert manifest.read_bytes() == original

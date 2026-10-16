from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any

import pytest

from avengine.assets.glb import load_glb
from avengine.assets.glb_write import build_glb
from tools.assets import add_missing_uv0 as add_uv_cli


_CORE_TEXTURE_FIELDS = [
    ("pbrMetallicRoughness", "baseColorTexture"),
    ("pbrMetallicRoughness", "metallicRoughnessTexture"),
    (None, "normalTexture"),
    (None, "occlusionTexture"),
    (None, "emissiveTexture"),
]

_EXTENSION_TEXTURE_FIELDS = [
    ("KHR_materials_anisotropy", "anisotropyTexture"),
    ("KHR_materials_clearcoat", "clearcoatTexture"),
    ("KHR_materials_clearcoat", "clearcoatRoughnessTexture"),
    ("KHR_materials_clearcoat", "clearcoatNormalTexture"),
    ("KHR_materials_diffuse_transmission", "diffuseTransmissionTexture"),
    ("KHR_materials_diffuse_transmission", "diffuseTransmissionColorTexture"),
    ("KHR_materials_iridescence", "iridescenceTexture"),
    ("KHR_materials_iridescence", "iridescenceThicknessTexture"),
    ("KHR_materials_pbrSpecularGlossiness", "diffuseTexture"),
    ("KHR_materials_pbrSpecularGlossiness", "specularGlossinessTexture"),
    ("KHR_materials_sheen", "sheenColorTexture"),
    ("KHR_materials_sheen", "sheenRoughnessTexture"),
    ("KHR_materials_specular", "specularTexture"),
    ("KHR_materials_specular", "specularColorTexture"),
    ("KHR_materials_transmission", "transmissionTexture"),
    ("KHR_materials_volume", "thicknessTexture"),
]


def _write_fixture(
    tmp_path: Path,
    *,
    materials: list[dict[str, Any]] | None = None,
    primitive_material: int | None = 0,
    name: str = "source.glb",
) -> Path:
    binary = struct.pack(
        "<9f",
        -0.5,
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    primitive: dict[str, Any] = {"attributes": {"POSITION": 0}}
    if primitive_material is not None:
        primitive["material"] = primitive_material
    document: dict[str, Any] = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
            }
        ],
        "meshes": [{"primitives": [primitive]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    if materials is not None:
        document["materials"] = materials
    if any(
        "Texture" in json.dumps(material, sort_keys=True)
        for material in materials or []
    ):
        document["textures"] = [{}]
    source = tmp_path / name
    source.write_bytes(build_glb(document, binary))
    return source


def _core_material(
    container: str | None, field: str, texture_info: dict[str, Any]
) -> dict[str, Any]:
    if container is None:
        return {field: texture_info}
    return {container: {field: texture_info}}


def _extension_material(
    extension: str, field: str, texture_info: dict[str, Any]
) -> dict[str, Any]:
    return {"extensions": {extension: {field: texture_info}}}


def _cli_arguments(source: Path, output: Path, report: Path) -> list[str]:
    return [
        "--input",
        str(source),
        "--output",
        str(output),
        "--report",
        str(report),
    ]


@pytest.mark.parametrize(("container", "field"), _CORE_TEXTURE_FIELDS)
def test_add_missing_uv0_rejects_every_core_texture_using_uv0(
    tmp_path: Path, container: str | None, field: str
) -> None:
    material = _core_material(container, field, {"index": 0})
    source = _write_fixture(tmp_path, materials=[material])

    with pytest.raises(ValueError, match="would consume it"):
        add_uv_cli.augment(source, tmp_path / "output.glb")

    assert not (tmp_path / "output.glb").exists()


@pytest.mark.parametrize(("extension", "field"), _EXTENSION_TEXTURE_FIELDS)
def test_add_missing_uv0_rejects_known_extension_texture_using_uv0(
    tmp_path: Path, extension: str, field: str
) -> None:
    material = _extension_material(extension, field, {"index": 0})
    source = _write_fixture(tmp_path, materials=[material])

    with pytest.raises(ValueError, match="would consume it"):
        add_uv_cli.augment(source, tmp_path / "output.glb")


def test_add_missing_uv0_rejects_unknown_material_extension(tmp_path: Path) -> None:
    source = _write_fixture(
        tmp_path,
        materials=[{"extensions": {"EXT_future_material": {}}}],
    )

    with pytest.raises(ValueError, match="unknown material extension"):
        add_uv_cli.augment(source, tmp_path / "output.glb")


def test_add_missing_uv0_rejects_texture_transform_overriding_to_uv0(
    tmp_path: Path,
) -> None:
    texture_info = {
        "index": 0,
        "texCoord": 1,
        "extensions": {"KHR_texture_transform": {"texCoord": 0}},
    }
    source = _write_fixture(
        tmp_path,
        materials=[
            _core_material("pbrMetallicRoughness", "baseColorTexture", texture_info)
        ],
    )

    with pytest.raises(ValueError, match="would consume it"):
        add_uv_cli.augment(source, tmp_path / "output.glb")


def test_add_missing_uv0_audits_unreferenced_materials(tmp_path: Path) -> None:
    source = _write_fixture(
        tmp_path,
        materials=[
            {"pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0]}},
            _core_material("pbrMetallicRoughness", "baseColorTexture", {"index": 0}),
        ],
    )

    with pytest.raises(ValueError, match=r"would consume it: materials\[1\]"):
        add_uv_cli.augment(source, tmp_path / "output.glb")


def test_add_missing_uv0_allows_texture_explicitly_using_another_set(
    tmp_path: Path,
) -> None:
    material = _extension_material(
        "KHR_materials_clearcoat", "clearcoatTexture", {"index": 0, "texCoord": 1}
    )
    source = _write_fixture(tmp_path, materials=[material])
    output = tmp_path / "output.glb"

    report = add_uv_cli.augment(source, output)

    primitive = load_glb(output).json["meshes"][0]["primitives"][0]
    assert "TEXCOORD_0" in primitive["attributes"]
    assert report["policy"]["texture_using_synthesized_texcoord_0_allowed"] is False


@pytest.mark.parametrize("occupied", ["output", "report"])
@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_add_missing_uv0_cli_preflights_both_paired_outputs(
    tmp_path: Path, occupied: str, kind: str
) -> None:
    source = _write_fixture(
        tmp_path,
        materials=[{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1]}}],
    )
    output = tmp_path / "output.glb"
    report = tmp_path / "report.json"
    path = output if occupied == "output" else report
    if kind == "file":
        path.write_bytes(b"sentinel")
    else:
        path.symlink_to(tmp_path / f"dangling-{occupied}")

    with pytest.raises(SystemExit):
        add_uv_cli.main(_cli_arguments(source, output, report))

    counterpart = report if occupied == "output" else output
    assert not counterpart.exists()
    assert not counterpart.is_symlink()
    if kind == "file":
        assert path.read_bytes() == b"sentinel"
    else:
        assert path.is_symlink()


def test_add_missing_uv0_cli_cleans_output_when_report_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_fixture(
        tmp_path,
        materials=[{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1]}}],
    )
    output = tmp_path / "output.glb"
    report = tmp_path / "report.json"
    real_write = add_uv_cli._write_exclusive
    calls = 0

    def fail_second_write(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected report failure")
        real_write(path, payload)

    monkeypatch.setattr(add_uv_cli, "_write_exclusive", fail_second_write)
    with pytest.raises(SystemExit):
        add_uv_cli.main(_cli_arguments(source, output, report))

    assert calls == 2
    assert not output.exists()
    assert not report.exists()

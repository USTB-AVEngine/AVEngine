from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from avengine.contracts.json_io import sha256_file
from avengine.contracts.json_io import canonical_json_sha256
from avengine.m2.glb import load_glb
from avengine.m2.glb_write import build_glb
from avengine.m2.materials import (
    MaterialNormalizationError,
    load_and_validate_material_normalization_report,
    normalize_glb_materials,
    validate_material_normalization_report,
)
from tools.m2 import normalize_materials as cli


UNCHANGED_SECTIONS = (
    "buffers",
    "accessors",
    "meshes",
    "skins",
    "animations",
    "textures",
    "images",
    "samplers",
)


def _document() -> dict[str, Any]:
    return {
        "asset": {"version": "2.0", "generator": "material-normalizer-test"},
        "extensionsUsed": ["KHR_materials_specular"],
        "buffers": [{"byteLength": 4}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 4}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5121,
                "count": 4,
                "type": "SCALAR",
            }
        ],
        "nodes": [{"name": "root", "mesh": 0}],
        "meshes": [
            {
                "name": "animal",
                "primitives": [
                    {"attributes": {"_TEST": 0}, "material": 0},
                    {"attributes": {"_TEST": 0}, "material": 1},
                ],
            }
        ],
        "skins": [{"name": "skin", "joints": [0]}],
        "animations": [{"name": "Idle", "channels": [], "samplers": []}],
        "samplers": [{"magFilter": 9729, "minFilter": 9729}],
        "images": [{"name": "embedded", "bufferView": 0, "mimeType": "image/png"}],
        "textures": [{"sampler": 0, "source": 0}],
        "materials": [
            {
                "name": "glossy_blend",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.2, 0.3, 0.4, 0.4],
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 0},
                    "metallicFactor": 0.8,
                    "roughnessFactor": 0.1,
                },
                "alphaMode": "BLEND",
                "emissiveFactor": [0.8, 0.6, 0.4],
                "emissiveTexture": {"index": 0},
                "doubleSided": True,
                "extensions": {
                    "KHR_materials_specular": {
                        "specularFactor": 0.9,
                        "specularColorFactor": [-0.5, 0.4, 1.5],
                        "specularTexture": {"index": 0},
                    }
                },
                "extras": {"retained": "yes"},
            },
            {
                "name": "implicit_pbr_mask",
                "alphaMode": "MASK",
                "alphaCutoff": 0.4,
                "doubleSided": False,
            },
        ],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }


def _source(tmp_path: Path, document: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "source.glb"
    path.write_bytes(build_glb(_document() if document is None else document, b"DATA"))
    return path


def test_normalizer_changes_only_bounded_material_fields(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "normalized.glb"
    before = load_glb(source)

    report = normalize_glb_materials(source, output)
    after = load_glb(output)
    materials = after.json["materials"]

    assert materials[0]["pbrMetallicRoughness"]["metallicFactor"] == 0.0
    assert materials[0]["pbrMetallicRoughness"]["roughnessFactor"] == 0.72
    assert "metallicRoughnessTexture" not in materials[0]["pbrMetallicRoughness"]
    assert materials[0]["pbrMetallicRoughness"]["baseColorFactor"] == [
        0.2,
        0.3,
        0.4,
        0.4,
    ]
    assert materials[0]["alphaMode"] == "BLEND"
    assert materials[0]["emissiveFactor"] == [0.0, 0.0, 0.0]
    assert "emissiveTexture" not in materials[0]
    specular = materials[0]["extensions"]["KHR_materials_specular"]
    assert specular["specularFactor"] == 0.25
    assert specular["specularColorFactor"] == [0.0, 0.4, 1.0]
    assert specular["specularTexture"] == {"index": 0}
    assert materials[0]["extras"] == {"retained": "yes"}

    assert materials[1]["pbrMetallicRoughness"] == {
        "metallicFactor": 0.0,
        "roughnessFactor": 1.0,
    }
    assert materials[1]["extensions"]["KHR_materials_specular"] == {
        "specularFactor": 0.25,
        "specularColorFactor": [1.0, 1.0, 1.0],
    }
    assert materials[1]["alphaMode"] == "MASK"
    assert materials[1]["alphaCutoff"] == 0.4
    assert before.binary == after.binary == b"DATA"
    for section in UNCHANGED_SECTIONS:
        assert before.json.get(section) == after.json.get(section)
    assert {key: value for key, value in before.json.items() if key != "materials"} == {
        key: value for key, value in after.json.items() if key != "materials"
    }

    assert report["status"] == "pass"
    assert report["qualification_state"] == "research_candidate"
    assert report["qualification_claim"] is False
    assert report["invariants"]["binary_chunk"]["unchanged"] is True
    assert report["invariants"]["all_non_material_control_json"]["unchanged"] is True
    assert report["invariants"]["only_material_control_json_changed"] is True
    assert report["policy"]["emissive_factor"] == [0.0, 0.0, 0.0]
    assert report["policy"]["emissive_texture"] == "removed"
    assert all(
        item["unchanged"]
        for item in report["invariants"]["required_json_sections"].values()
    )


def test_force_opaque_is_explicit_and_preserves_rgb(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "opaque.glb"

    report = normalize_glb_materials(source, output, force_opaque=True)
    materials = load_glb(output).json["materials"]

    assert report["policy"]["force_opaque"] is True
    assert materials[0]["alphaMode"] == "OPAQUE"
    assert materials[0]["pbrMetallicRoughness"]["baseColorFactor"] == [
        0.2,
        0.3,
        0.4,
        1.0,
    ]
    assert materials[1]["alphaMode"] == "OPAQUE"
    assert materials[1]["pbrMetallicRoughness"]["baseColorFactor"] == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert all(material["emissiveFactor"] == [0.0, 0.0, 0.0] for material in materials)
    assert all("emissiveTexture" not in material for material in materials)


def test_default_mode_does_not_add_or_change_alpha_declarations(
    tmp_path: Path,
) -> None:
    document = _document()
    document["materials"][1].pop("alphaMode")
    source = _source(tmp_path, document)
    output = tmp_path / "normalized.glb"

    normalize_glb_materials(source, output)
    materials = load_glb(output).json["materials"]

    assert materials[0]["alphaMode"] == "BLEND"
    assert materials[0]["pbrMetallicRoughness"]["baseColorFactor"][3] == 0.4
    assert "alphaMode" not in materials[1]
    assert "baseColorFactor" not in materials[1]["pbrMetallicRoughness"]


def test_report_contains_per_material_before_after_and_file_hashes(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "normalized.glb"
    report = normalize_glb_materials(source, output)

    assert report["source"]["sha256"] == sha256_file(source)
    assert report["output"]["sha256"] == sha256_file(output)
    assert report["material_count"] == 2
    assert len(report["materials"]) == 2
    assert all("before" in item and "after" in item for item in report["materials"])
    assert report["materials"][0]["before"] == _document()["materials"][0]
    assert report["materials"][0]["after"] == load_glb(output).json["materials"][0]
    validate_material_normalization_report(report)


def test_report_hash_and_file_closure_fail_on_tampering(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "normalized.glb"
    report = normalize_glb_materials(source, output)

    tampered = deepcopy(report)
    tampered["materials"][0]["after"]["name"] = "edited"
    with pytest.raises(MaterialNormalizationError, match="report_content_sha256"):
        validate_material_normalization_report(tampered)

    tampered["report_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in tampered.items()
            if key != "report_content_sha256"
        }
    )
    with pytest.raises(MaterialNormalizationError, match="before/after records"):
        validate_material_normalization_report(tampered)

    output.write_bytes(output.read_bytes() + b"tamper")
    with pytest.raises(MaterialNormalizationError, match="byte_size mismatch"):
        validate_material_normalization_report(report)


def test_output_is_exclusive_and_source_is_never_overwritten(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "existing.glb"
    output.write_bytes(b"keep-me")
    source_before = source.read_bytes()

    with pytest.raises(MaterialNormalizationError, match="already exists"):
        normalize_glb_materials(source, output)
    assert output.read_bytes() == b"keep-me"
    assert source.read_bytes() == source_before

    with pytest.raises(MaterialNormalizationError, match="must not overwrite"):
        normalize_glb_materials(source, source)
    assert source.read_bytes() == source_before


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.pop("materials"), "materials must be an array"),
        (
            lambda value: value["materials"][0].update(pbrMetallicRoughness="bad"),
            "pbrMetallicRoughness must be an object",
        ),
        (
            lambda value: value["materials"][0].update(pbrMetallicRoughness=None),
            "pbrMetallicRoughness must be an object",
        ),
        (
            lambda value: value["materials"][0]["pbrMetallicRoughness"].update(
                roughnessFactor=1.2
            ),
            "roughnessFactor must be <= 1",
        ),
        (
            lambda value: value["materials"][0].update(alphaMode="TRANSPARENT"),
            "alphaMode must be OPAQUE",
        ),
        (
            lambda value: value["materials"][0]["extensions"][
                "KHR_materials_specular"
            ].update(specularColorFactor=[0.2, 0.3]),
            "exactly three finite numbers",
        ),
        (
            lambda value: value["materials"][0].update(emissiveFactor=[0.2, 0.3]),
            "emissiveFactor must contain exactly 3 finite numbers",
        ),
        (
            lambda value: value["materials"][0].update(emissiveTexture=None),
            "emissiveTexture must be a textureInfo object",
        ),
        (
            lambda value: value["materials"][0]["extensions"].update(
                KHR_materials_specular=None
            ),
            "KHR_materials_specular must be an object",
        ),
        (
            lambda value: value.update(
                extensionsUsed=[
                    "KHR_materials_specular",
                    "KHR_materials_specular",
                ]
            ),
            "extensionsUsed must not contain duplicates",
        ),
        (
            lambda value: value["materials"][0]["extensions"].update(
                KHR_materials_transmission={"transmissionFactor": 0.5}
            ),
            "unsupported material extensions",
        ),
    ],
)
def test_malformed_or_ambiguous_materials_fail_closed(
    tmp_path: Path, mutate, message: str
) -> None:
    document = _document()
    mutate(document)
    source = _source(tmp_path, document)

    with pytest.raises(MaterialNormalizationError, match=message):
        normalize_glb_materials(source, tmp_path / "normalized.glb")


def test_missing_root_specular_declaration_is_added_and_bound(
    tmp_path: Path,
) -> None:
    document = _document()
    document.pop("extensionsUsed")
    for material in document["materials"]:
        material.pop("extensions", None)
    source = _source(tmp_path, document)
    output = tmp_path / "normalized.glb"

    report = normalize_glb_materials(source, output)
    normalized = load_glb(output).json

    assert normalized["extensionsUsed"] == ["KHR_materials_specular"]
    assert all(
        material["extensions"]["KHR_materials_specular"]["specularFactor"] == 0.25
        for material in normalized["materials"]
    )
    validate_material_normalization_report(report)


def test_cli_writes_exclusive_hash_bound_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "normalized.glb"
    report_path = tmp_path / "normalization.json"

    assert (
        cli.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--report",
                str(report_path),
                "--force-opaque",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    report = load_and_validate_material_normalization_report(report_path)
    assert summary["status"] == "pass"
    assert summary["report_content_sha256"] == report["report_content_sha256"]
    assert summary["qualification_claim"] is False
    original_report = report_path.read_bytes()

    second_output = tmp_path / "must_not_be_created.glb"
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "--input",
                str(source),
                "--output",
                str(second_output),
                "--report",
                str(report_path),
            ]
        )
    assert raised.value.code == 2
    assert not second_output.exists()
    assert report_path.read_bytes() == original_report


@pytest.mark.parametrize("occupied", ["output", "report"])
@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_cli_preflights_both_paired_outputs(
    tmp_path: Path, occupied: str, kind: str
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "normalized.glb"
    report = tmp_path / "normalization.json"
    path = output if occupied == "output" else report
    if kind == "file":
        path.write_bytes(b"sentinel")
    else:
        path.symlink_to(tmp_path / f"dangling-{occupied}")

    with pytest.raises(SystemExit):
        cli.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--report",
                str(report),
            ]
        )

    counterpart = report if occupied == "output" else output
    assert not counterpart.exists()
    assert not counterpart.is_symlink()
    if kind == "file":
        assert path.read_bytes() == b"sentinel"
    else:
        assert path.is_symlink()


def test_cli_rolls_back_output_when_report_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "normalized.glb"
    report = tmp_path / "normalization.json"

    def fail_report(_path: Path, _value: object) -> None:
        raise OSError("injected report failure")

    monkeypatch.setattr(cli, "_write_json_exclusive", fail_report)
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--report",
                str(report),
            ]
        )

    assert not output.exists()
    assert not report.exists()


def test_real_horse_force_opaque_preserves_animation_and_mesh_json(
    tmp_path: Path,
) -> None:
    source = Path("assets/mesh_library/quaternius_farm/Horse.glb")
    output = tmp_path / "horse_opaque.glb"
    before = load_glb(source)

    report = normalize_glb_materials(source, output, force_opaque=True)
    after = load_glb(output)

    assert report["material_count"] >= 1
    assert all(
        material["alphaMode"] == "OPAQUE" for material in after.json["materials"]
    )
    assert all(
        material["pbrMetallicRoughness"]["baseColorFactor"][3] == 1.0
        for material in after.json["materials"]
    )
    assert "KHR_materials_specular" in after.json["extensionsUsed"]
    assert all(
        material["extensions"]["KHR_materials_specular"]["specularFactor"] == 0.25
        for material in after.json["materials"]
    )
    for section in UNCHANGED_SECTIONS:
        assert before.json.get(section) == after.json.get(section)
    validate_material_normalization_report(report)

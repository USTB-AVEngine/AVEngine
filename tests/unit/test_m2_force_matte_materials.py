from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from avengine.m2.glb import load_glb
from avengine.m2.glb_write import build_glb
from tools.m2 import force_matte_materials as cli


def _document() -> dict[str, Any]:
    return {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 4}],
        "materials": [
            {
                "name": "unsafe",
                "alphaMode": "BLEND",
                "emissiveFactor": [0.7, 0.4, 0.2],
                "emissiveTexture": {"index": 0},
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.6, 0.4, 0.2, 0.25],
                    "metallicFactor": 0.8,
                    "roughnessFactor": 0.1,
                    "metallicRoughnessTexture": {"index": 0},
                },
                "extensions": {
                    "KHR_materials_specular": {
                        "specularFactor": 1.0,
                        "specularTexture": {"index": 0},
                    }
                },
            }
        ],
    }


def _source(tmp_path: Path, document: dict[str, Any] | None = None) -> Path:
    source = tmp_path / "source.glb"
    source.write_bytes(
        build_glb(_document() if document is None else document, b"DATA")
    )
    return source


def _arguments(source: Path, output: Path, report: Path) -> list[str]:
    return [
        "--input",
        str(source),
        "--output",
        str(output),
        "--report",
        str(report),
    ]


def test_force_matte_enforces_complete_material_policy(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "matte.glb"
    report_path = tmp_path / "report.json"

    assert cli.main(_arguments(source, output, report_path)) == 0

    before = load_glb(source)
    after = load_glb(output)
    material = after.json["materials"][0]
    pbr = material["pbrMetallicRoughness"]
    assert before.binary == after.binary == b"DATA"
    assert after.json["extensionsUsed"] == ["KHR_materials_specular"]
    assert material["alphaMode"] == "OPAQUE"
    assert material["emissiveFactor"] == [0.0, 0.0, 0.0]
    assert "emissiveTexture" not in material
    assert pbr["baseColorFactor"] == [0.6, 0.4, 0.2, 1.0]
    assert pbr["metallicFactor"] == 0.0
    assert pbr["roughnessFactor"] == 1.0
    assert "metallicRoughnessTexture" not in pbr
    assert material["extensions"] == {
        "KHR_materials_specular": {
            "specularFactor": 0.0,
            "specularColorFactor": [1.0, 1.0, 1.0],
        }
    }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "avengine_m2_force_matte_materials_v2"
    assert report["status"] == "pass"
    assert report["material_policy_complete"] is True
    assert report["qualification_claim"] is False


def test_force_matte_rejects_unsupported_material_extension(tmp_path: Path) -> None:
    document = _document()
    document["materials"][0]["extensions"]["KHR_materials_transmission"] = {
        "transmissionFactor": 0.5
    }
    source = _source(tmp_path, document)
    output = tmp_path / "matte.glb"
    report = tmp_path / "report.json"

    with pytest.raises(SystemExit):
        cli.main(_arguments(source, output, report))

    assert not output.exists()
    assert not report.exists()


@pytest.mark.parametrize("occupied", ["output", "report"])
@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_force_matte_preflights_both_paired_outputs(
    tmp_path: Path, occupied: str, kind: str
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "matte.glb"
    report = tmp_path / "report.json"
    path = output if occupied == "output" else report
    if kind == "file":
        path.write_bytes(b"sentinel")
    else:
        path.symlink_to(tmp_path / f"dangling-{occupied}")

    with pytest.raises(SystemExit):
        cli.main(_arguments(source, output, report))

    counterpart = report if occupied == "output" else output
    assert not counterpart.exists()
    assert not counterpart.is_symlink()
    if kind == "file":
        assert path.read_bytes() == b"sentinel"
    else:
        assert path.is_symlink()


def test_force_matte_rolls_back_output_when_report_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "matte.glb"
    report = tmp_path / "report.json"
    real_write = cli._write
    calls = 0

    def fail_second_write(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected report failure")
        real_write(path, payload)

    monkeypatch.setattr(cli, "_write", fail_second_write)
    with pytest.raises(SystemExit):
        cli.main(_arguments(source, output, report))

    assert calls == 2
    assert not output.exists()
    assert not report.exists()

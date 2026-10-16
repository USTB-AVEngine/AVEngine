"""Unit tests for tools/ue/assemble_package_stage.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "assemble_package_stage",
    REPOSITORY / "tools/ue/assemble_package_stage.py",
)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def _write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _source_slice(tmp_path: Path) -> Path:
    root = tmp_path / "slice"
    _write(root / "SpearSim/SpearSim.uproject", b"{}")
    _write(root / "SpearSim/Config/DefaultGame.ini")
    _write(root / "plugins/SpContent/SpContent.uplugin", b"{}")
    _write(root / "PROVENANCE.md")
    return root


def _closure_report(tmp_path: Path, *, complete: bool = True) -> Path:
    primary = _write(tmp_path / "inputs/game/BP_demo.uasset", b"bp")
    sidecar = _write(tmp_path / "inputs/game/BP_demo.uexp", b"exp")
    camera = _write(tmp_path / "inputs/sp/BP_CameraSensor.uasset", b"cam")
    report = {
        "variants": {
            "test_variant": {
                "mapping_complete": complete,
                "physical_mappings": [
                    {
                        "package": "/Game/MyAssets/Audioset/Blueprints/gate_demo/BP_demo",
                        "source_file": str(primary),
                        "source_sidecars": [str(sidecar)],
                        "status": "unique_authorized_external_input",
                    },
                    {
                        "package": "/SpContent/Blueprints/BP_CameraSensor",
                        "source_file": str(camera),
                        "source_sidecars": [],
                        "status": "unique_authorized_external_input",
                    },
                ],
            }
        }
    }
    path = tmp_path / "closure.json"
    path.write_text(json.dumps(report))
    return path


def _args(tmp_path: Path, report: Path, stage: Path) -> object:
    return tool.parse_args(
        [
            "--closure-report", str(report),
            "--variant-name", "test_variant",
            "--source-slice", str(_source_slice(tmp_path)),
            "--stage-root", str(stage),
            "--source-commit", "deadbeef",
        ]
    )


def test_assembles_stage_layout(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    result = tool.assemble_stage(_args(tmp_path, _closure_report(tmp_path), stage))
    assert (stage / "SpearSim/SpearSim.uproject").is_file()
    assert (stage / "plugins/SpContent/SpContent.uplugin").is_file()
    assert (
        stage / "SpearSim/Content/MyAssets/Audioset/Blueprints/gate_demo/BP_demo.uasset"
    ).read_bytes() == b"bp"
    assert (
        stage / "SpearSim/Content/MyAssets/Audioset/Blueprints/gate_demo/BP_demo.uexp"
    ).read_bytes() == b"exp"
    assert (
        stage / "plugins/SpContent/Content/Blueprints/BP_CameraSensor.uasset"
    ).read_bytes() == b"cam"
    uproject = json.loads((stage / "SpearSim/SpearSim.uproject").read_text())
    assert {"Name": "SpContent", "Enabled": True} in uproject.get("Plugins", [])
    provenance = json.loads((stage / "STAGE_PROVENANCE.json").read_text())
    assert provenance["closure_variant"] == "test_variant"
    assert provenance["avengine_source_commit"] == "deadbeef"
    assert provenance["research_only"] is True
    assert provenance["qualification_claim"] is False
    assert result["content_files_copied"] == 3
    assert "BuildCookRun" in result["command"][1]


def test_refuses_existing_stage(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    with pytest.raises(tool.StageAssemblyError, match="refusing to replace"):
        tool.assemble_stage(_args(tmp_path, _closure_report(tmp_path), stage))


def test_refuses_incomplete_variant(tmp_path: Path) -> None:
    report = _closure_report(tmp_path, complete=False)
    with pytest.raises(tool.StageAssemblyError, match="not mapping_complete"):
        tool.assemble_stage(_args(tmp_path, report, tmp_path / "stage"))


def test_refuses_missing_source_file(tmp_path: Path) -> None:
    report_path = _closure_report(tmp_path)
    report = json.loads(report_path.read_text())
    report["variants"]["test_variant"]["physical_mappings"][0]["source_file"] = str(
        tmp_path / "inputs/game/absent.uasset"
    )
    report_path.write_text(json.dumps(report))
    with pytest.raises(tool.StageAssemblyError, match="missing source file"):
        tool.assemble_stage(_args(tmp_path, report_path, tmp_path / "stage"))

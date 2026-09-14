from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from avengine.runtime_profiles import (
    load_default_source_asset_runtime_registry,
    load_source_asset_runtime_registry,
    validate_new_source_asset_runtime_profile,
    validate_source_asset_runtime_registry,
)
from tools.registry.register_source_runtime_asset import (
    SourceRuntimeAssetRegistrationError,
    merge_source_runtime_asset,
    register_source_runtime_asset,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/registry/register_source_runtime_asset.py"
PYTHONPATH = (
    f"{ROOT / 'src'}:"
    "/data/jzy/tmp/wt-grok-pilot46-round2/tmp/native_python_addons_v1"
)


def _asset_fixture() -> tuple[dict, dict]:
    registry = load_default_source_asset_runtime_registry()
    source = next(
        item
        for item in registry["assets"]
        if item["asset_id"]
        == "generated_air_conditioner_wall_split_white_research_v1"
    )
    asset = deepcopy(source)
    asset.update(
        {
            "asset_id": "generated_test_wall_source_runtime_asset_v1",
            "revision": "test_registration_v1",
            "display_label": "Test wall source",
            "display_label_zh": "测试壁挂声源",
        }
    )
    asset["runtime_backends"]["spear_unreal"][
        "static_mesh_object_path"
    ] = "/Game/AVEngine/Test/SM_test_wall_source.SM_test_wall_source"
    return registry, asset


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _cli_env() -> dict[str, str]:
    env = dict(__import__("os").environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = PYTHONPATH
    return env


def test_merge_function_accepts_any_supported_appearance_value() -> None:
    registry, asset = _asset_fixture()
    asset["realized_attributes"]["finish"] = "silver"
    asset["realized_attributes"]["body_color"] = "white"
    assert validate_new_source_asset_runtime_profile(asset) == []

    before = deepcopy(registry)
    merged = merge_source_runtime_asset(registry, asset)
    assert len(merged["assets"]) == len(registry["assets"]) + 1
    assert merged["assets"][-1]["asset_id"] == asset["asset_id"]
    assert registry == before
    assert validate_source_asset_runtime_registry(merged) == []


def test_function_rejects_duplicate_without_mutating_input() -> None:
    registry, asset = _asset_fixture()
    asset["asset_id"] = registry["assets"][0]["asset_id"]
    before = deepcopy(registry)
    with pytest.raises(SourceRuntimeAssetRegistrationError, match="duplicate"):
        merge_source_runtime_asset(registry, asset)
    assert registry == before


def test_cli_writes_fresh_merged_registry_and_preserves_inputs(tmp_path: Path) -> None:
    registry, asset = _asset_fixture()
    registry_path = tmp_path / "registry.json"
    asset_path = tmp_path / "asset.json"
    output_path = tmp_path / "derived" / "registry.json"
    _write_json(registry_path, registry)
    _write_json(asset_path, asset)
    registry_before = registry_path.read_bytes()
    asset_before = asset_path.read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--registry",
            str(registry_path),
            "--asset",
            str(asset_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "SOURCE_RUNTIME_ASSET_REGISTRATION_OK" in result.stdout
    assert registry_path.read_bytes() == registry_before
    assert asset_path.read_bytes() == asset_before
    merged = load_source_asset_runtime_registry(output_path)
    assert len(merged["assets"]) == len(registry["assets"]) + 1
    assert merged["assets"][-1]["asset_id"] == asset["asset_id"]


def test_cli_rejects_duplicate_and_leaves_fresh_output_absent(tmp_path: Path) -> None:
    registry, asset = _asset_fixture()
    asset["asset_id"] = registry["assets"][0]["asset_id"]
    registry_path = tmp_path / "registry.json"
    asset_path = tmp_path / "asset.json"
    output_path = tmp_path / "derived.json"
    _write_json(registry_path, registry)
    _write_json(asset_path, asset)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--registry",
            str(registry_path),
            "--asset",
            str(asset_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "duplicate" in result.stderr
    assert not output_path.exists()


def test_cli_rejects_unimplemented_only_appearance_value(tmp_path: Path) -> None:
    registry, asset = _asset_fixture()
    asset["realized_attributes"].pop("body_color")
    # standard_seal_point is observed now; a value with no colour-family
    # predicate at all is what the registration check has to refuse.
    asset["realized_attributes"]["finish"] = "iridescent_teal_flake"
    registry_path = tmp_path / "registry.json"
    asset_path = tmp_path / "asset.json"
    output_path = tmp_path / "derived.json"
    _write_json(registry_path, registry)
    _write_json(asset_path, asset)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--registry",
            str(registry_path),
            "--asset",
            str(asset_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "native RGB appearance vocabulary" in result.stderr
    assert not output_path.exists()


def test_cli_refuses_existing_output_without_overwriting(tmp_path: Path) -> None:
    registry, asset = _asset_fixture()
    registry_path = tmp_path / "registry.json"
    asset_path = tmp_path / "asset.json"
    output_path = tmp_path / "derived.json"
    _write_json(registry_path, registry)
    _write_json(asset_path, asset)
    output_path.write_text("sentinel\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--registry",
            str(registry_path),
            "--asset",
            str(asset_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "fresh output already exists" in result.stderr
    assert output_path.read_text(encoding="utf-8") == "sentinel\n"

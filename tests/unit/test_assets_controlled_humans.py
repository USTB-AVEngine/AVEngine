from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from avengine.assets import controlled_humans as controlled


def _write_catalog(
    tmp_path: Path,
    *,
    data_root_env: str | None = None,
    include_required: bool = True,
) -> tuple[Path, Path]:
    producer = tmp_path / "producer"
    producer.mkdir(parents=True)
    source = producer / "source.fbx"
    manifest = producer / "normalization_manifest.json"
    glb = producer / "runtime.glb"
    source.write_bytes(b"source")
    manifest.write_text('{"schema": "source"}\n', encoding="utf-8")
    glb.write_bytes(b"glb")

    def entry(tag: str, color: str) -> dict:
        value: dict = {
            "tag": tag,
            "asset_id": f"asset_{color}",
            "top_color": color,
            "rgb": [11, 22, 33],
            "source_tag": f"source_{color}",
            "variant_id": f"variant_{color}",
            "producer_root": str(producer),
            "source_asset": {"path": source.name},
            "normalization_manifest": {
                "path": manifest.name,
                "size_bytes": manifest.stat().st_size,
            },
            "runtime_glb": {
                "path": glb.name,
                "sha256": hashlib.sha256(glb.read_bytes()).hexdigest(),
            },
        }
        if include_required:
            value.update(
                {
                    "expected_bone_count": 2,
                    "material_names": [f"{color}_body"],
                    "image_names": [f"{color}_color"],
                    "animation_names": ["Walk", "Idle"],
                    "preview_animation_name": "Walk",
                    "required_primitive_attributes": ["POSITION"],
                    "expected_primitive_count": 1,
                    "expected_texture_count": 1,
                    "actor_scale": 1.0,
                    "ue_manifest_relative_path": (
                        f"tmp/controlled_human_ue_import_v1/{tag}/manifest.json"
                    ),
                    "ue_manifest_schema": "avengine_controlled_human_ue_import_v1",
                }
            )
        return value

    if data_root_env:
        for value in (entry("human_violet_v1", "violet"), entry("human_amber_v1", "amber")):
            value["producer_root"] = "."
        catalog_value = {
            "schema": controlled.CATALOG_SCHEMA,
            "artifact_root_env": data_root_env,
            "entries": [
                entry("human_violet_v1", "violet"),
                entry("human_amber_v1", "amber"),
            ],
        }
    else:
        catalog_value = {
            "schema": controlled.CATALOG_SCHEMA,
            "entries": [
                entry("human_violet_v1", "violet"),
                entry("human_amber_v1", "amber"),
            ],
        }
    catalog = tmp_path / "controlled_humans.json"
    catalog.write_text(
        json.dumps(catalog_value, indent=2) + "\n",
        encoding="utf-8",
    )
    return catalog, producer


def test_catalog_drives_arbitrary_tags_and_import_properties(tmp_path: Path) -> None:
    catalog, producer = _write_catalog(tmp_path)

    document = controlled.load_catalog(catalog_path=catalog)
    assert [entry["tag"] for entry in document["entries"]] == [
        "human_violet_v1",
        "human_amber_v1",
    ]
    entry = controlled.entry_for_tag("human_violet_v1", catalog_path=catalog)
    assert entry["top_color"] == "violet"

    contract = controlled.resolve_import_contract(
        "human_violet_v1", catalog_path=catalog
    )
    assert contract["expected_material_names"] == ["violet_body"]
    assert contract["expected_image_names"] == ["violet_color"]
    assert contract["required_animation_names"] == ["Walk", "Idle"]
    assert contract["expected_bone_count"] == 2
    assert contract["runtime_glb_path"] == str(producer / "runtime.glb")
    assert contract["preview_animation_name"] == "Walk"


def test_external_root_is_selected_by_catalog_environment(tmp_path: Path, monkeypatch) -> None:
    catalog, producer = _write_catalog(
        tmp_path,
        data_root_env="CONTROLLED_HUMAN_TEST_ROOT",
    )
    monkeypatch.setenv("CONTROLLED_HUMAN_TEST_ROOT", str(tmp_path / "producer"))
    contract = controlled.resolve_import_contract(
        "human_violet_v1",
        catalog_path=catalog,
    )
    assert contract["producer_root"] == str(producer.resolve())


def test_described_artifacts_are_checked_only_when_requested(tmp_path: Path) -> None:
    catalog, producer = _write_catalog(tmp_path)
    entry = controlled.entry_for_tag("human_violet_v1", catalog_path=catalog)
    document = controlled.load_catalog(catalog_path=catalog)

    observed = controlled.validate_artifacts(
        entry,
        document,
        catalog_path=catalog,
    )
    assert observed["runtime_glb"][1]["size_bytes"] == 3
    assert observed["normalization_manifest"][1]["size_bytes"] == 21

    (producer / "runtime.glb").write_bytes(b"changed")
    with pytest.raises(controlled.ControlledHumanError, match="hash differs"):
        controlled.validate_artifacts(entry, document, catalog_path=catalog)


def test_artifact_paths_cannot_escape_or_follow_a_symlink(tmp_path: Path) -> None:
    catalog, producer = _write_catalog(tmp_path)
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["entries"][0]["runtime_glb"] = {"path": "../outside.glb"}
    catalog.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(controlled.ControlledHumanError, match="unsafe"):
        controlled.load_catalog(catalog_path=catalog)

    outside = tmp_path / "outside.glb"
    outside.write_bytes(b"outside")
    link = producer / "link.glb"
    link.symlink_to(outside)
    payload["entries"][0]["runtime_glb"] = {"path": link.name}
    catalog.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    entry = controlled.entry_for_tag("human_violet_v1", catalog_path=catalog)
    document = controlled.load_catalog(catalog_path=catalog)
    with pytest.raises(controlled.ControlledHumanError, match="symbolic link"):
        controlled.validate_artifacts(entry, document, catalog_path=catalog)


def test_required_import_properties_have_no_python_fallback(tmp_path: Path) -> None:
    catalog, _producer = _write_catalog(tmp_path, include_required=False)
    with pytest.raises(controlled.ControlledHumanError, match="must be declared"):
        controlled.load_importer_contracts(catalog_path=catalog)

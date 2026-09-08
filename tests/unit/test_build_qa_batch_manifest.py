"""Manifest builder writes this worktree catalog and full path_bindings."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem + "_h3_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builder():
    return _load("tools/dataset/build_qa_batch_manifest.py")


@pytest.fixture
def controller():
    return _load("tools/studio/run_qa_episode.py")


def test_declared_catalog_path_is_preserved_without_directory_heuristics(builder, tmp_path, monkeypatch):
    production = builder.production_room_catalog_path()
    monkeypatch.chdir(tmp_path)
    wt_catalog = tmp_path / "wt-production" / "catalog.json"
    wt_catalog.parent.mkdir()
    wt_catalog.write_text(json.dumps({"rooms": []}) + "\n", encoding="utf-8")
    assert builder.resolve_request_room_catalog(str(wt_catalog)) == wt_catalog.resolve()
    assert builder.resolve_request_room_catalog("examples/rooms/packages/catalog.json") == production
    assert builder.resolve_request_room_catalog(None) == production
    assert builder.resolve_request_room_catalog("") == production
    other = tmp_path / "custom_catalog.json"
    other.write_text("{}\n", encoding="utf-8")
    assert builder.resolve_request_room_catalog(str(other)) == other.resolve()
    explicit = tmp_path / "explicit.json"
    explicit.write_text("{}\n", encoding="utf-8")
    assert builder.resolve_request_room_catalog(str(wt_catalog), explicit=explicit) == explicit.resolve()


def test_catalog_resolution_does_not_follow_cwd(builder, tmp_path, monkeypatch):
    decoy = tmp_path / "examples" / "rooms" / "packages"
    decoy.mkdir(parents=True)
    (decoy / "catalog.json").write_text(json.dumps({"rooms": []}) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    resolved = builder.resolve_request_room_catalog("examples/rooms/packages/catalog.json")
    assert resolved == builder.production_room_catalog_path()
    assert resolved != (decoy / "catalog.json").resolve()


def test_stamp_request_writes_absolute_catalog_and_full_bindings(builder):
    catalog = json.loads(builder.production_room_catalog_path().read_text(encoding="utf-8"))
    bindings = builder.catalog_path_bindings(catalog)
    assert "AVENGINE_MULTI_HOME_AUTHORING_ROOT" in bindings
    request = {
        "episode_id": "demo",
        "room_catalog": "/data/jzy/tmp/wt-multi-home-activity-integration/examples/rooms/packages/catalog.json",
        "runtime": {"graphics_adapter": 0, "path_bindings": {"AVENGINE_CUSTOM": "/custom"}},
    }
    builder.stamp_request_catalog(
        request, catalog_path=builder.production_room_catalog_path(), path_bindings=bindings)
    assert request["room_catalog"] == str(builder.production_room_catalog_path())
    assert request["runtime"]["path_bindings"]["AVENGINE_CUSTOM"] == "/custom"
    assert request["runtime"]["path_bindings"]["AVENGINE_MULTI_HOME_ROOT"] == bindings["AVENGINE_MULTI_HOME_ROOT"]
    assert request["runtime"]["path_bindings"]["AVENGINE_MULTI_HOME_AUTHORING_ROOT"] == (
        bindings["AVENGINE_MULTI_HOME_AUTHORING_ROOT"])
    assert request["runtime"]["graphics_adapter"] == 0


def test_request_path_bindings_override_catalog(controller):
    catalog = {"path_bindings": {"AVENGINE_A": "/catalog", "AVENGINE_B": "/catalog-b"}}
    request = {"runtime": {"graphics_adapter": 1, "path_bindings": {"AVENGINE_A": "/request"}}}
    runtime = controller.request_package_runtime(request, catalog)
    assert runtime["path_bindings"]["AVENGINE_A"] == "/request"
    assert runtime["path_bindings"]["AVENGINE_B"] == "/catalog-b"
    assert runtime["graphics_adapter"] == 1


def test_request_without_bindings_uses_catalog(controller):
    catalog = {"path_bindings": {"AVENGINE_A": "/catalog"}}
    runtime = controller.request_package_runtime({"runtime": {"rpc_port": 1}}, catalog)
    assert runtime["path_bindings"] == {"AVENGINE_A": "/catalog"}
    assert runtime["rpc_port"] == 1

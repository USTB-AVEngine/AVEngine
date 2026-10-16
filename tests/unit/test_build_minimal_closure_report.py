"""Unit tests for tools/ue/build_minimal_closure_report.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "build_minimal_closure_report",
    REPOSITORY / "tools/ue/build_minimal_closure_report.py",
)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


MAP_PACKAGE = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"
CAMERA_PACKAGE = "/SpContent/Blueprints/BP_CameraSensor"
ACTOR_BP = "/Game/MyAssets/Audioset/Blueprints/gate_demo/BP_gate_demo"
ACTOR_MESH = "/Game/MyAssets/Audioset/Meshes/gate_demo/runtime"
ACTOR_IDLE = "/Game/MyAssets/Audioset/Meshes/gate_demo/Standing_Idle"
ACTOR_WALK = "/Game/MyAssets/Audioset/Meshes/gate_demo/Walking"


def _write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _graph(tmp_path: Path, *, edges: list[tuple[str, str]]) -> Path:
    packages = sorted(
        {MAP_PACKAGE, CAMERA_PACKAGE, ACTOR_BP, ACTOR_MESH, ACTOR_IDLE, ACTOR_WALK}
        | {edge[0] for edge in edges}
        | {edge[1] for edge in edges if edge[1].startswith(("/Game/", "/SpContent/"))}
    )
    path = tmp_path / "dependency_graph.json"
    path.write_text(
        json.dumps(
            {
                "kind": "asset_registry_dependency_export",
                "packages": packages,
                "edges": [
                    {"from_package": a, "to_package": b} for a, b in edges
                ],
            }
        )
    )
    return path


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "asset_id": "demo_actor",
                        "runtime_backends": {
                            "spear_unreal": {
                                "blueprint_class_path": f"{ACTOR_BP}.BP_gate_demo_C",
                                "idle_animation": f"{ACTOR_IDLE}.Standing_Idle",
                                "walking_animation": f"{ACTOR_WALK}.Walking",
                            }
                        },
                    }
                ]
            }
        )
    )
    return path


def _source_root(tmp_path: Path, name: str = "root") -> Path:
    root = tmp_path / name
    game = root / "cpp/unreal_projects/SpearSim/Content"
    _write(game / "SPEAR/Scenes/apartment_0000/Maps/apartment_0000.umap")
    _write(game / "MyAssets/Audioset/Blueprints/gate_demo/BP_gate_demo.uasset")
    _write(game / "MyAssets/Audioset/Meshes/gate_demo/runtime.uasset")
    _write(game / "MyAssets/Audioset/Meshes/gate_demo/runtime.uexp")
    _write(game / "MyAssets/Audioset/Meshes/gate_demo/Standing_Idle.uasset")
    _write(game / "MyAssets/Audioset/Meshes/gate_demo/Walking.uasset")
    _write(root / "cpp/unreal_plugins/SpContent/Content/Blueprints/BP_CameraSensor.uasset")
    return root


def _args(tmp_path: Path, graph: Path, registry: Path, roots: list[Path]) -> object:
    argv = [
        "--dependency-graph", str(graph),
        "--source-asset-registry", str(registry),
        "--variant-name", "test_variant",
        "--output", str(tmp_path / "out/report.json"),
    ]
    for root in roots:
        argv += ["--source-root", str(root)]
    return tool.parse_args(argv)


def test_closure_report_maps_reachable_content(tmp_path: Path) -> None:
    graph = _graph(
        tmp_path,
        edges=[
            (ACTOR_BP, ACTOR_MESH),
            (ACTOR_BP, "/Script/Engine"),
            (MAP_PACKAGE, "/Script/NavigationSystem"),
        ],
    )
    args = _args(tmp_path, graph, _registry(tmp_path), [_source_root(tmp_path)])
    report = tool.build_report(args)
    variant = report["variants"]["test_variant"]
    packages = {entry["package"] for entry in variant["physical_mappings"]}
    assert packages == {
        MAP_PACKAGE, CAMERA_PACKAGE, ACTOR_BP, ACTOR_MESH, ACTOR_IDLE, ACTOR_WALK,
    }
    assert variant["mapping_complete"] is True
    assert variant["non_content_terminal_dependencies"] == 2
    mesh = next(
        entry for entry in variant["physical_mappings"] if entry["package"] == ACTOR_MESH
    )
    assert mesh["source_sidecars"] and mesh["source_sidecars"][0].endswith("runtime.uexp")
    assert all(
        entry["status"] == "unique_authorized_external_input"
        for entry in variant["physical_mappings"]
    )
    map_entry = next(
        entry for entry in variant["physical_mappings"] if entry["package"] == MAP_PACKAGE
    )
    assert map_entry["source_file"].endswith(".umap")


def test_missing_source_fails_closed(tmp_path: Path) -> None:
    graph = _graph(tmp_path, edges=[(ACTOR_BP, ACTOR_MESH)])
    root = _source_root(tmp_path)
    (root / "cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Meshes/gate_demo/runtime.uasset").unlink()
    args = _args(tmp_path, graph, _registry(tmp_path), [root])
    with pytest.raises(tool.ClosureReportError, match="no authorized source file"):
        tool.build_report(args)


def test_conflicting_duplicate_sources_fail_closed(tmp_path: Path) -> None:
    graph = _graph(tmp_path, edges=[(ACTOR_BP, ACTOR_MESH)])
    root_a = _source_root(tmp_path, "root_a")
    root_b = _source_root(tmp_path, "root_b")
    conflicting = (
        root_b
        / "cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Meshes/gate_demo/runtime.uasset"
    )
    conflicting.write_bytes(b"different")
    args = _args(tmp_path, graph, _registry(tmp_path), [root_a, root_b])
    with pytest.raises(tool.ClosureReportError, match="ambiguous authorized sources"):
        tool.build_report(args)


def test_byte_identical_duplicate_sources_resolve_to_first_root(tmp_path: Path) -> None:
    graph = _graph(tmp_path, edges=[(ACTOR_BP, ACTOR_MESH)])
    root_a = _source_root(tmp_path, "root_a")
    root_b = _source_root(tmp_path, "root_b")
    args = _args(tmp_path, graph, _registry(tmp_path), [root_a, root_b])
    report = tool.build_report(args)
    variant = report["variants"]["test_variant"]
    assert all(
        entry["source_file"].startswith(str(root_a))
        for entry in variant["physical_mappings"]
    )


def test_unknown_seed_asset_fails_closed(tmp_path: Path) -> None:
    graph = _graph(tmp_path, edges=[])
    args = _args(tmp_path, graph, _registry(tmp_path), [_source_root(tmp_path)])
    args.asset_id = ["missing_actor"]
    with pytest.raises(tool.ClosureReportError, match="unknown registry asset"):
        tool.build_report(args)

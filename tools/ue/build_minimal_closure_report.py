#!/usr/bin/env python3
"""Build a minimal-closure report for the current Apartment visual stage.

Replays the retained 20260820 analysis procedure as a maintained tool: BFS the
in-editor asset-registry dependency export from explicit seed packages (room
map + camera blueprint + selected source-asset actors), then map every
reachable content package to exactly one authorized external input file.

The output matches the report shape consumed by
``avengine.m5.current_apartment_visual._closure_mappings``: a ``variants``
mapping whose complete variant carries ``physical_mappings`` entries with
``status == "unique_authorized_external_input"``. The tool is fail-closed: an
unresolvable or ambiguous package aborts without writing output, and the
output path refuses to replace an existing file.

Package-to-file rule (unchanged from the retained report):
  /Game/<rel>      -> <root>/cpp/unreal_projects/SpearSim/Content/<rel>.uasset|.umap
  /SpContent/<rel> -> <root>/cpp/unreal_plugins/SpContent/Content/<rel>.uasset|.umap
  /Engine, /Script and other mount roots are non-content terminal dependencies
  and are never copied into a game-content closure.
Sidecars with the same stem (.uexp/.ubulk/.uptnl/.m.ubulk) are recorded and
travel with their primary file.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

DEFAULT_ROOM_MAP = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"
DEFAULT_CAMERA_PACKAGE = "/SpContent/Blueprints/BP_CameraSensor"
CONTENT_ROOTS = ("/Game/", "/SpContent/")
SIDECAR_SUFFIXES = (".uexp", ".ubulk", ".uptnl", ".m.ubulk")
PRIMARY_SUFFIXES = (".uasset", ".umap")


class ClosureReportError(RuntimeError):
    """Raised when the closure cannot be built fail-closed."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-graph", required=True, type=Path)
    parser.add_argument("--source-asset-registry", required=True, type=Path)
    parser.add_argument(
        "--asset-id",
        action="append",
        default=None,
        help="registry asset to seed; repeatable; default: every asset with a spear_unreal backend",
    )
    parser.add_argument("--room-map", default=DEFAULT_ROOM_MAP)
    parser.add_argument("--camera-package", default=DEFAULT_CAMERA_PACKAGE)
    parser.add_argument(
        "--source-root",
        action="append",
        required=True,
        type=Path,
        help="authorized external input root, repeatable, highest priority first",
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        help="optional assembled stage to verify byte identity against",
    )
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _load_mapping(path: Path, *, owner: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ClosureReportError(f"cannot read {owner}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ClosureReportError(f"{owner} is not a JSON object: {path}")
    return value


def _package_of_object_path(value: str, *, owner: str) -> str:
    package = value.split(".", 1)[0]
    if not package.startswith(CONTENT_ROOTS):
        raise ClosureReportError(f"{owner} is not a content object path: {value}")
    return package


def select_seed_records(
    registry: dict[str, Any],
    *,
    asset_ids: list[str] | None,
    room_map: str,
    camera_package: str,
) -> list[dict[str, str]]:
    assets = registry.get("assets")
    if not isinstance(assets, list):
        raise ClosureReportError("registry has no assets list")
    by_id = {a.get("asset_id"): a for a in assets if isinstance(a, dict)}
    selected = list(by_id) if asset_ids is None else asset_ids
    seeds = [
        {"origin": "current room runtime profile", "package": room_map, "role": "apartment_map"},
        {
            "origin": "current apartment runner",
            "package": camera_package,
            "role": "camera_blueprint",
        },
    ]
    for asset_id in selected:
        asset = by_id.get(asset_id)
        if asset is None:
            raise ClosureReportError(f"unknown registry asset: {asset_id}")
        backend = (asset.get("runtime_backends") or {}).get("spear_unreal")
        if not isinstance(backend, dict):
            if asset_ids is None:
                continue
            raise ClosureReportError(f"asset has no spear_unreal backend: {asset_id}")
        blueprint = backend.get("blueprint_class_path")
        if not isinstance(blueprint, str):
            raise ClosureReportError(f"asset has no blueprint_class_path: {asset_id}")
        seeds.append(
            {
                "object_path": blueprint,
                "origin": f"{asset_id} runtime profile",
                "package": _package_of_object_path(blueprint, owner=asset_id),
                "role": "actor_blueprint",
            }
        )
    return seeds


def reachable_closure(
    graph: dict[str, Any], seed_packages: list[str]
) -> tuple[list[str], Counter, int]:
    packages = graph.get("packages")
    edges = graph.get("edges")
    if not isinstance(packages, list) or not isinstance(edges, list):
        raise ClosureReportError("dependency graph is missing packages or edges")
    known = set(packages)
    absent = [seed for seed in seed_packages if seed not in known]
    if absent:
        raise ClosureReportError(f"seed packages absent from graph: {absent}")
    forward: dict[str, list[str]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        forward.setdefault(str(edge.get("from_package")), []).append(
            str(edge.get("to_package"))
        )
    seen: set[str] = set()
    non_content: set[str] = set()
    queue = deque(seed_packages)
    while queue:
        package = queue.popleft()
        if package in seen:
            continue
        seen.add(package)
        for target in forward.get(package, ()):
            if target.startswith(CONTENT_ROOTS):
                if target not in seen:
                    queue.append(target)
            else:
                non_content.add(target)
    content = sorted(p for p in seen if p.startswith(CONTENT_ROOTS))
    class_counts = Counter(
        "game_content" if p.startswith("/Game/") else "spcontent" for p in content
    )
    return content, class_counts, len(non_content)


def _candidate_files(package: str, root: Path) -> list[Path]:
    if package.startswith("/Game/"):
        base = root / "cpp/unreal_projects/SpearSim/Content" / package.removeprefix("/Game/")
    else:
        base = root / "cpp/unreal_plugins/SpContent/Content" / package.removeprefix(
            "/SpContent/"
        )
    return [base.with_suffix(suffix) for suffix in PRIMARY_SUFFIXES]


def map_package(
    package: str, source_roots: list[Path], *, stage_root: Path | None
) -> dict[str, Any]:
    hits: list[Path] = []
    for root in source_roots:
        for candidate in _candidate_files(package, root):
            if candidate.is_file() and not candidate.is_symlink():
                hits.append(candidate)
    if not hits:
        raise ClosureReportError(f"no authorized source file for package {package}")
    primary = hits[0]
    for other in hits[1:]:
        if other.suffix != primary.suffix or not filecmp.cmp(
            primary, other, shallow=False
        ):
            raise ClosureReportError(
                f"ambiguous authorized sources for package {package}: {primary} vs {other}"
            )
    sidecars = sorted(
        str(primary.with_name(primary.stem + suffix))
        for suffix in SIDECAR_SUFFIXES
        if primary.with_name(primary.stem + suffix).is_file()
    )
    entry: dict[str, Any] = {
        "content_class": (
            "authorized_external_game_content"
            if package.startswith("/Game/")
            else "authorized_external_spcontent"
        ),
        "package": package,
        "source_file": str(primary),
        "source_file_regular": True,
        "source_sidecars": sidecars,
        "source_stage_byte_identical": False,
        "stage_file": None,
        "status": "unique_authorized_external_input",
    }
    if stage_root is not None:
        if package.startswith("/Game/"):
            staged = (
                stage_root
                / "SpearSim/Content"
                / (package.removeprefix("/Game/") + primary.suffix)
            )
        else:
            staged = (
                stage_root
                / "plugins/SpContent/Content"
                / (package.removeprefix("/SpContent/") + primary.suffix)
            )
        entry["stage_file"] = str(staged)
        if not staged.is_file() or staged.is_symlink():
            raise ClosureReportError(f"stage is missing closure package {package}: {staged}")
        if not filecmp.cmp(primary, staged, shallow=False):
            raise ClosureReportError(f"stage file differs from source for {package}")
        entry["source_stage_byte_identical"] = True
    return entry


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    graph = _load_mapping(args.dependency_graph, owner="dependency graph")
    if graph.get("kind") != "asset_registry_dependency_export":
        raise ClosureReportError("dependency graph has an unexpected kind")
    registry = _load_mapping(args.source_asset_registry, owner="source asset registry")
    source_roots = []
    for root in args.source_root:
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise ClosureReportError(f"--source-root is not a directory: {root}")
        source_roots.append(resolved)
    stage_root = None
    if args.stage_root is not None:
        stage_root = args.stage_root.expanduser().resolve()
        if not stage_root.is_dir():
            raise ClosureReportError(f"--stage-root is not a directory: {args.stage_root}")

    seeds = select_seed_records(
        registry,
        asset_ids=args.asset_id,
        room_map=args.room_map,
        camera_package=args.camera_package,
    )
    seed_packages = sorted({seed["package"] for seed in seeds})
    content, class_counts, non_content_count = reachable_closure(graph, seed_packages)
    mappings = [
        map_package(package, source_roots, stage_root=stage_root) for package in content
    ]
    variant = {
        "absent_seed_packages": 0,
        "mapping_complete": True,
        "name": args.variant_name,
        "non_content_terminal_dependencies": non_content_count,
        "physical_mapping_status_counts": dict(
            Counter(entry["status"] for entry in mappings)
        ),
        "physical_mappings": mappings,
        "reachable_class_counts": dict(class_counts),
        "reachable_content_package_count": len(content),
        "seed_records": seeds,
    }
    return {
        "claim_boundary": (
            "Authorized-input closure mapping for the current Apartment visual stage; "
            "research only, no dataset admission and no byte lock beyond the recorded files."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dependency_graph": str(args.dependency_graph.resolve()),
            "source_asset_registry": str(args.source_asset_registry.resolve()),
            "source_roots": [str(root) for root in source_roots],
            "stage_root": None if stage_root is None else str(stage_root),
        },
        "mapping_rule": {
            "/Game/<relative_package>": "authorized_external_game_content_input/<relative_package>.uasset or .umap",
            "/SpContent/<relative_package>": "authorized_external_spcontent_input/<relative_package>.uasset or .umap",
            "non_content": (
                "/Engine is UE installation content; /Script is a code/module dependency; "
                "other mount roots are UE plugin content or modules and are not copied into "
                "a game-content closure"
            ),
            "sidecars": "same package stem with .uexp/.ubulk/.uptnl/.m.ubulk if present",
        },
        "report_kind": "avengine_minimal_closure_report",
        "status": "pass",
        "variants": {args.variant_name: variant},
    }


def main() -> int:
    args = parse_args()
    output = args.output.expanduser()
    if output.exists() or output.is_symlink():
        raise ClosureReportError(f"refusing to replace existing output: {output}")
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    variant = report["variants"][args.variant_name]
    print(
        json.dumps(
            {
                "output": str(output),
                "content_packages": variant["reachable_content_package_count"],
                "non_content_terminal_dependencies": variant[
                    "non_content_terminal_dependencies"
                ],
                "seeds": len(variant["seed_records"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ClosureReportError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)

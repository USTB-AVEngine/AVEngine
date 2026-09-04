"""Export real Unreal package dependencies for declared mounted content roots.

Set AVENGINE_ASSET_DEPENDENCY_REQUEST to a JSON object containing mount_roots
and output. Run in the Unreal Editor after saving imported content. The
result is consumed by build_minimal_closure_report.py in this repository.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import unreal


def main():
    request_value = os.environ.get("AVENGINE_ASSET_DEPENDENCY_REQUEST")
    if not request_value:
        raise RuntimeError("AVENGINE_ASSET_DEPENDENCY_REQUEST is required")
    request_path = Path(request_value).resolve()
    request = json.loads(request_path.read_text())
    output = Path(request["output"]).expanduser()
    if not output.is_absolute():
        output = request_path.parent / output
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to replace dependency export: {output}")
    roots = request["mount_roots"]
    if not isinstance(roots, list) or not roots or any(
        not isinstance(root, str) or not root.startswith("/") for root in roots
    ):
        raise RuntimeError("mount_roots must be a nonempty list of Unreal package roots")
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    registry.scan_paths_synchronous(roots, force_rescan=True)
    registry.wait_for_completion()
    packages = set()
    for root in roots:
        for asset in registry.get_assets_by_path(root, recursive=True):
            packages.add(str(asset.package_name))
    options = unreal.AssetRegistryDependencyOptions(
        include_soft_package_references=True, include_hard_package_references=True,
        include_searchable_names=False, include_soft_management_references=False,
        include_hard_management_references=False,
    )
    edges = []
    for package in sorted(packages):
        for dependency in sorted(str(value) for value in registry.get_dependencies(package, options)):
            edges.append({"from_package": package, "to_package": dependency})
    result = {"kind": "asset_registry_dependency_export", "packages": sorted(packages),
              "edges": edges, "mount_roots": roots, "producer": str(Path(__file__).resolve())}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    unreal.log(f"AVENGINE_ASSET_DEPENDENCY_EXPORT_OK packages={len(packages)} edges={len(edges)} output={output}")


if __name__ == "__main__":
    main()

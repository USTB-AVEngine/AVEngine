"""Import declared rigid source GLBs into fresh Unreal content directories.

Set AVENGINE_STATIC_SOURCE_REQUEST to a JSON object with output and assets.
Each asset declares asset_id, source_glb and destination (/Game package folder).
For a separate-process reload, declare verify_manifest and a fresh output.
Run through UnrealEditor -run=pythonscript; all project code comes from AVEngine.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys

import unreal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_spear_3d_front_sample_editor import _assets_by_class, _import_glb


def _vector(vector):
    values = [float(vector.x), float(vector.y), float(vector.z)]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("imported static mesh has non-finite bounds")
    return values


def _inspect(destination):
    assets = _assets_by_class(destination)
    mesh_paths = assets.get("StaticMesh", [])
    if not mesh_paths:
        raise RuntimeError(f"no StaticMesh imported into {destination}")
    meshes = []
    for path in mesh_paths:
        mesh = unreal.load_asset(path)
        if not isinstance(mesh, unreal.StaticMesh):
            raise RuntimeError(f"could not reload StaticMesh: {path}")
        box = mesh.get_bounding_box()
        lower, upper = _vector(box.min), _vector(box.max)
        if any(hi <= lo for lo, hi in zip(lower, upper)):
            raise RuntimeError(f"degenerate static mesh bounds: {path}")
        materials = []
        for slot in mesh.get_editor_property("static_materials"):
            material = slot.get_editor_property("material_interface")
            materials.append(str(material.get_path_name()) if material else None)
        meshes.append({"object_path": path, "minimum_cm": lower,
                       "maximum_cm": upper, "materials": materials})
    return {"assets_by_class": assets, "meshes": meshes}


def _write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    value = os.environ.get("AVENGINE_STATIC_SOURCE_REQUEST")
    if not value:
        raise RuntimeError("AVENGINE_STATIC_SOURCE_REQUEST is required")
    request_path = Path(value).expanduser().resolve()
    request = json.loads(request_path.read_text())
    def resource(value):
        path = Path(value).expanduser()
        return (request_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
    output = resource(request["output"])
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to replace output: {output}")
    verify = request.get("verify_manifest")
    if verify:
        source = resource(verify)
        imported = json.loads(source.read_text())
        observed = []
        for entry in imported["assets"]:
            current = _inspect(entry["destination"])
            if current != entry["content"]:
                raise RuntimeError(f"saved static content differs after reload: {entry['asset_id']}")
            observed.append({"asset_id": entry["asset_id"], "destination": entry["destination"],
                             "content": current})
        _write_new(output, {"status": "pass", "verification_of": str(source), "assets": observed})
        unreal.log(f"AVENGINE_STATIC_SOURCE_RELOAD_OK output={output}")
        return
    assets = request.get("assets")
    if not isinstance(assets, list) or not assets:
        raise RuntimeError("assets must be a nonempty list")
    prepared, identities, directories = [], set(), set()
    for entry in assets:
        asset_id, destination = entry["asset_id"], entry["destination"]
        if not isinstance(asset_id, str) or not asset_id or asset_id in identities:
            raise RuntimeError("asset IDs must be nonempty and unique")
        if not isinstance(destination, str) or not re.fullmatch(r"/Game/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+", destination):
            raise RuntimeError(f"invalid content directory: {destination}")
        if any(destination == d or destination.startswith(d + "/") or d.startswith(destination + "/") for d in directories):
            raise RuntimeError("import directories must not overlap")
        if unreal.EditorAssetLibrary.does_directory_exist(destination):
            raise RuntimeError(f"refusing to replace Unreal directory: {destination}")
        glb = resource(entry["source_glb"])
        if not glb.is_file() or glb.suffix.lower() != ".glb":
            raise RuntimeError(f"source GLB is missing: {glb}")
        identities.add(asset_id); directories.add(destination)
        prepared.append((asset_id, glb, destination))
    records = []
    for asset_id, glb, destination in prepared:
        try:
            _import_glb(glb, destination)
            records.append({"asset_id": asset_id, "source_glb": str(glb),
                            "destination": destination, "content": _inspect(destination)})
        except BaseException:
            unreal.log_error(f"Static source import failed; partial content retained: {destination}")
            raise
    _write_new(output, {"status": "pass", "research_only": True,
                        "producer": str(Path(__file__).resolve()), "assets": records})
    unreal.log(f"AVENGINE_STATIC_SOURCE_IMPORT_OK output={output}")


if __name__ == "__main__":
    main()

"""Small QA room-package validation and lossless legacy catalog wrapping."""
from __future__ import annotations

from copy import deepcopy
import json
import os
import re
from string import Template
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "avengine_qa_room_package_v1"
RENDERERS = {"apartment": "ue_spear", "kujiale": "ue_spear", "authored": "ue_spear",
             "mp3d": "habitat", "hm3d": "habitat"}
REQUIRED = ("room_id", "family", "renderer", "visual_scene", "acoustic_package",
            "walkable_space", "floor_reference", "static_geometry", "semantics",
            "coordinate_frame", "subrooms")


def room_package_errors(package: Mapping[str, Any]) -> list[str]:
    """Report missing facts without fabricating defaults or judging feasibility."""
    errors = []
    if package.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    for key in REQUIRED:
        if key not in package or package[key] is None or package[key] == "":
            errors.append(f"missing {key}")
    if package.get("family") not in RENDERERS:
        errors.append("family is not a production room family")
    elif package.get("renderer") != RENDERERS[package["family"]]:
        errors.append("renderer conflicts with the production family route")
    visual = package.get("visual_scene")
    if not isinstance(visual, Mapping):
        errors.append("visual_scene must be a mapping")
    else:
        keys = ("map_path", "uproject") if package.get("renderer") == "ue_spear" else (
            "scene_glb", "dataset_config", "navmesh")
        for key in keys:
            if not isinstance(visual.get(key), str) or not visual[key]:
                errors.append(f"missing visual_scene.{key}")
        if str(visual.get("scene_glb", "")).endswith(".basis.glb"):
            errors.append("Habitat scene_glb must be a non-basis GLB")
    walkable = package.get("walkable_space")
    if not isinstance(walkable, Mapping) or walkable.get("kind") not in {
            "furniture_grid", "walkable_grid", "route_bank", "habitat_navmesh"}:
        errors.append("walkable_space.kind is missing or unsupported")
    elif not walkable.get("path"):
        errors.append("missing walkable_space.path")
    for key in ("static_geometry", "semantics", "coordinate_frame"):
        if not isinstance(package.get(key), Mapping) or not package[key]:
            errors.append(f"{key} must declare its source")
    coordinate = package.get("coordinate_frame", {})
    if isinstance(coordinate, Mapping):
        for key in ("linear_unit", "up_axis", "handedness", "world_transform"):
            if not coordinate.get(key):
                errors.append(f"missing coordinate_frame.{key}")
    floor = package.get("floor_reference")
    if floor is not None and not (isinstance(floor, str) and floor or
                                  isinstance(floor, Mapping) and floor.get("path")):
        errors.append("floor_reference must reference a measured artifact")
    if not isinstance(package.get("subrooms"), list):
        errors.append("subrooms must be a list (empty when the map has no subdivisions)")
    for field, path in missing_filesystem_paths(package):
        errors.append(f"missing path {field}: {path}")
    return errors


def _is_ue_or_usd_virtual_path(value: str) -> bool:
    return value.startswith("/Game") or value.startswith("/Root")


def _is_absolute_filesystem_path(value: str) -> bool:
    if not isinstance(value, str) or not value.startswith("/") or _is_ue_or_usd_virtual_path(value):
        return False
    if value.startswith("${"):
        return False
    return True


def missing_filesystem_paths(package: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return (field, path) for absolute non-/Game non-/Root paths that are missing."""
    missing: list[tuple[str, str]] = []

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).endswith("_template"):
                    continue
                name = f"{prefix}.{key}" if prefix else str(key)
                walk(item, name)
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{prefix}[{index}]")
            return
        if isinstance(value, str) and _is_absolute_filesystem_path(value):
            if not Path(value).exists():
                missing.append((prefix, value))

    walk(package, "")
    return missing


def validate_room_package(package: Mapping[str, Any]) -> dict:
    errors = room_package_errors(package)
    if errors:
        raise ValueError("RoomPackage: " + "; ".join(errors))
    return deepcopy(dict(package))


def renderer_for_room(package: Mapping[str, Any]) -> str:
    """Dispatch only on the declared family/renderer, never an adapter-name switch."""
    family, renderer = package.get("family"), package.get("renderer")
    if family not in RENDERERS or RENDERERS[family] != renderer:
        raise ValueError(f"invalid production family/renderer: {family!r}/{renderer!r}")
    return str(renderer)


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(os.path.expandvars(str(path))).expanduser().read_text())


def resolve_room_package_paths(package: Mapping[str, Any], *, runtime: Mapping[str, Any] | None = None) -> dict:
    """Expand configured package roots once, retaining original template metadata."""
    runtime = runtime or {}
    bindings = dict(os.environ)
    if runtime.get("mp3d_root"):
        bindings["AVENGINE_MP3D_ROOT"] = str(runtime["mp3d_root"])
    bindings.update({str(k): str(v) for k, v in runtime.get("path_bindings", {}).items()})
    def expand(value, key=""):
        if isinstance(value, Mapping):
            return {k: expand(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v, key) for v in value]
        if isinstance(value, str) and not key.endswith("_template"):
            result = Template(value).safe_substitute(bindings)
            missing = re.findall(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", result)
            if missing:
                raise ValueError("RoomPackage missing configured path roots: " + ", ".join(sorted(set(missing))))
            return result
        return deepcopy(value)
    return expand(package)


def package_from_catalog_entry(entry: Mapping[str, Any], *,
                                runtime: Mapping[str, Any] | None = None) -> dict:
    """Wrap old catalog metadata, preserving any unmeasured fields as missing.

    Explicit packages are strict. Legacy drafts do not retroactively invalidate
    old requests; their validation errors remain visible until P3 supplies the
    measured package. The old entry itself is retained unchanged.
    """
    declared = entry.get("room_package")
    if declared is not None:
        package = _load_json(declared) if isinstance(declared, (str, Path)) else declared
        return resolve_room_package_paths(validate_room_package(package), runtime=runtime)
    if entry.get("schema") == SCHEMA:
        return resolve_room_package_paths(validate_room_package(entry), runtime=runtime)
    native = entry.get("native_room_adapter") == "avengine_native_spear_apartment_qa_room_v1"
    family = entry.get("family", "apartment" if native else "authored")
    renderer = entry.get("renderer", RENDERERS.get(family))
    runtime = runtime or {}
    package = {
        "schema": SCHEMA, "room_id": entry.get("room_id"), "family": family, "renderer": renderer,
        "visual_scene": {"map_path": entry.get("map_path"), "uproject": runtime.get("uproject")},
        "acoustic_package": entry.get("acoustic_package"),
        "walkable_space": {"kind": "route_bank" if native else "furniture_grid",
                           "path": entry.get("route_bank") if native else entry.get("manifest")},
        "floor_reference": entry.get("floor_reference"),
        "static_geometry": deepcopy(entry.get("static_geometry")),
        "semantics": deepcopy(entry.get("semantics")),
        "coordinate_frame": {
            "linear_unit": "centimeter", "up_axis": "+Z", "handedness": "left",
            "world_transform": "ue_xyz_cm_to_xzy_m_v1"},
        "subrooms": deepcopy(entry.get("subrooms", [])),
        "legacy_catalog_entry": deepcopy(dict(entry)),
    }
    if renderer == "habitat":
        source = _load_json(entry["room_manifest"]) if entry.get("room_manifest") else {}
        scene = source.get("scene", {})
        package.update(
            visual_scene={"scene_glb": scene.get("scene_id"),
                          "dataset_config": scene.get("dataset_config_path"),
                          "navmesh": scene.get("navmesh_path")},
            walkable_space={"kind": "habitat_navmesh", "path": scene.get("navmesh_path")},
            semantics=deepcopy(source.get("semantics")),
            coordinate_frame={**deepcopy(source.get("coordinate_system", {})),
                              "world_transform": "identity_meter_y_up_right"})
    package["validation_errors"] = room_package_errors(package)
    return package

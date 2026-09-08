"""Small QA room-package validation and lossless legacy catalog wrapping."""
from __future__ import annotations

from copy import deepcopy
import json
import re
from string import Template
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "avengine_qa_room_package_v1"
RENDERERS = {"apartment": "ue_spear", "kujiale": "ue_spear", "authored": "ue_spear",
             "mp3d": "habitat", "hm3d": "habitat"}
REQUIRED = ("room_id", "family", "renderer", "visual_scene", "acoustic_package",
            "walkable_space", "floor_reference", "static_geometry", "semantics",
            "coordinate_frame", "subrooms")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_REPO_RELATIVE_PREFIXES = (
    "tmp/", "examples/", "assets/", "tools/", "src/", "external/", "envs/", "docs/",
)
_PATH_SUFFIXES = {
    ".json", ".npy", ".npz", ".glb", ".gltf", ".usd", ".usda", ".usdc", ".usdz",
    ".uproject", ".umap", ".png", ".jpg", ".jpeg", ".wav", ".mp4", ".txt",
    ".navmesh", ".ply", ".obj", ".ini", ".cfg", ".xml", ".yaml", ".yml",
}


def room_package_errors(
    package: Mapping[str, Any],
    *,
    relative_roots: Sequence[str | Path] | None = None,
) -> list[str]:
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
    for field, path in missing_filesystem_paths(package, relative_roots=relative_roots):
        errors.append(f"missing path {field}: {path}")
    return errors


def _is_ue_or_usd_virtual_path(value: str) -> bool:
    return value.startswith("/Game") or value.startswith("/Root")


def _is_unexpanded_template(value: str) -> bool:
    return isinstance(value, str) and "${" in value


def _is_absolute_filesystem_path(value: str) -> bool:
    if not isinstance(value, str) or not value.startswith("/") or _is_ue_or_usd_virtual_path(value):
        return False
    if _is_unexpanded_template(value):
        return False
    return True


def _normalized_relative(value: str) -> str:
    return value.replace("\\", "/")


def _is_repo_relative_path(value: str) -> bool:
    if not isinstance(value, str) or not value or value.startswith("/") or _is_ue_or_usd_virtual_path(value):
        return False
    if _is_unexpanded_template(value):
        return False
    return _normalized_relative(value).startswith(_REPO_RELATIVE_PREFIXES)


def _looks_like_relative_filesystem_path(
    value: str,
    *,
    include_filename: bool = False,
) -> bool:
    if _is_repo_relative_path(value):
        return True
    if not isinstance(value, str) or not value or value.startswith("/") or _is_ue_or_usd_virtual_path(value):
        return False
    if _is_unexpanded_template(value):
        return False
    normalized = _normalized_relative(value)
    name = Path(value).name.lower()
    if name.endswith(".scene_dataset_config.json"):
        return True
    suffix = Path(value).suffix.lower()
    return suffix in _PATH_SUFFIXES and (include_filename or "/" in normalized)


def _relative_roots(
    relative_roots: Sequence[str | Path] | None,
    *,
    declared: str | Path | None = None,
) -> list[Path]:
    roots: list[Path] = [REPOSITORY_ROOT.resolve()]
    seen = {REPOSITORY_ROOT.resolve()}
    extra: list[Path] = []
    if relative_roots:
        extra.extend(Path(item) for item in relative_roots)
    if declared is not None:
        path = Path(declared)
        extra.append(path.parent if path.is_absolute() else REPOSITORY_ROOT / path.parent)
    for item in extra:
        root = item if item.is_absolute() else REPOSITORY_ROOT / item
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            roots.append(resolved)
    return roots


def _resolve_relative_filesystem_path(value: str, roots: Sequence[Path]) -> str:
    """Rebase a declared relative path against the repository/package roots."""
    if _is_repo_relative_path(value):
        return str((REPOSITORY_ROOT / value).resolve())
    candidates = [Path(root) / value for root in roots[1:]]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((Path(roots[-1]) / value).resolve())


def _relative_roots_for_declared(declared: Any) -> list[Path]:
    if isinstance(declared, (str, Path)):
        return _relative_roots(None, declared=declared)
    return _relative_roots(None)


def missing_filesystem_paths(
    package: Mapping[str, Any],
    *,
    relative_roots: Sequence[str | Path] | None = None,
) -> list[tuple[str, str]]:
    """Return (field, path) for filesystem paths that do not exist.

    Call this after ``resolve_room_package_paths``. Unexpanded ``${VAR}``
    templates are not filesystem paths. Absolute non-/Game non-/Root paths are
    always checked. Relative ``tmp/`` / ``examples/`` paths are resolved against
    the repository root; other relative path-like strings are checked when
    extra roots (catalog directory) are supplied.
    """
    check_all_relative = relative_roots is not None
    roots = _relative_roots(relative_roots)
    missing: list[tuple[str, str]] = []

    def exists_relative(value: str) -> bool:
        return any((Path(root) / value).exists() for root in roots)

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
        if not isinstance(value, str):
            return
        if _is_absolute_filesystem_path(value):
            if not Path(value).exists():
                missing.append((prefix, value))
            return
        should_check = _is_repo_relative_path(value)
        if check_all_relative:
            should_check = should_check or _looks_like_relative_filesystem_path(value, include_filename=True)
        if should_check and not exists_relative(value):
            missing.append((prefix, value))

    walk(package, "")
    return missing


def validate_room_package(
    package: Mapping[str, Any],
    *,
    relative_roots: Sequence[str | Path] | None = None,
) -> dict:
    errors = room_package_errors(package, relative_roots=relative_roots)
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
    text = str(path)
    if _is_unexpanded_template(text):
        raise ValueError(f"unexpanded path template is not allowed: {text!r}")
    return json.loads(Path(text).expanduser().read_text(encoding="utf-8"))


def configured_path_bindings(runtime: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Return the explicit path roots used to expand a package.

    Shell ``AVENGINE_*`` variables are not a source. Callers must pass
    ``runtime.path_bindings`` (and optional ``mp3d_root``).
    """
    if runtime is None:
        runtime = {}
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime must be a mapping")
    bindings: dict[str, str] = {}
    if runtime.get("mp3d_root"):
        bindings["AVENGINE_MP3D_ROOT"] = str(runtime["mp3d_root"])
    raw = runtime.get("path_bindings") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("runtime.path_bindings must be a mapping")
    bindings.update({str(key): str(value) for key, value in raw.items()})
    return bindings


def resolve_room_package_paths(
    package: Mapping[str, Any],
    *,
    runtime: Mapping[str, Any] | None = None,
    relative_roots: Sequence[str | Path] | None = None,
) -> dict:
    """Expand configured roots and rebase relative filesystem paths."""
    bindings = configured_path_bindings(runtime)
    roots = _relative_roots(relative_roots) if relative_roots is not None else None
    missing: set[str] = set()

    def expand(value, key=""):
        if isinstance(value, Mapping):
            return {k: expand(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v, key) for v in value]
        if isinstance(value, str) and not key.endswith("_template"):
            result = Template(value).safe_substitute(bindings)
            missing.update(re.findall(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", result))
            if missing:
                # Keep building the complete object so every undeclared root
                # across every field appears in the final diagnostic.
                return result
            if roots is not None and _looks_like_relative_filesystem_path(
                result, include_filename=True
            ):
                return _resolve_relative_filesystem_path(result, roots)
            return result
        return deepcopy(value)

    resolved = expand(package)
    if missing:
        raise ValueError(
            "RoomPackage missing configured path roots: "
            + ", ".join(sorted(missing))
        )
    return resolved


def resolve_catalog_room_package_path(
    declared: str | Path,
    *,
    catalog_path: str | Path | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> Path:
    """Resolve a catalog room_package path against declared roots only."""
    text = Template(str(declared)).safe_substitute(configured_path_bindings(runtime))
    missing = re.findall(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", text)
    if missing:
        raise ValueError(
            "RoomPackage missing configured path roots: "
            + ", ".join(sorted(set(missing)))
        )
    text = str(Path(text).expanduser())
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    if catalog_path is None:
        raise ValueError(f"relative room_package path requires catalog_path: {declared!r}")
    catalog_file = Path(catalog_path).expanduser()
    if not catalog_file.is_absolute():
        catalog_file = REPOSITORY_ROOT / catalog_file
    catalog_dir = catalog_file.resolve().parent
    candidates = [catalog_dir / path]
    if path.name:
        candidates.append(catalog_dir / path.name)
    if _is_repo_relative_path(text):
        candidates.append(REPOSITORY_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def write_room_package_plan_snapshot(
    plan_dir: str | Path,
    package: Mapping[str, Any],
    *,
    path_bindings: Mapping[str, Any],
    catalog_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write the expanded room package and the bindings used to expand it."""
    plan_dir = Path(plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    package_path = plan_dir / "room_package.json"
    bindings_path = plan_dir / "path_bindings.json"
    bindings = {str(key): str(value) for key, value in dict(path_bindings).items()}
    record: dict[str, Any] = {"path_bindings": bindings}
    if catalog_path is not None:
        record["catalog_path"] = str(Path(catalog_path).expanduser().resolve())
    package_path.write_text(
        json.dumps(dict(package), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    bindings_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"room_package": package_path, "path_bindings": bindings_path}


def package_from_catalog_entry(entry: Mapping[str, Any], *,
                                runtime: Mapping[str, Any] | None = None,
                                catalog_path: str | Path | None = None) -> dict:
    """Wrap old catalog metadata, preserving any unmeasured fields as missing.

    Explicit packages are strict. Legacy drafts do not retroactively invalidate
    old requests; their validation errors remain visible until P3 supplies the
    measured package. The old entry itself is retained unchanged.

    Relative ``room_package`` paths are resolved against the catalog file
    directory, not the process cwd. Path existence runs after template
    expansion so ``${AVENGINE_...}`` values are real filesystem paths.
    """
    declared = entry.get("room_package")
    if declared is not None:
        if isinstance(declared, (str, Path)):
            declared_path = resolve_catalog_room_package_path(
                declared, catalog_path=catalog_path, runtime=runtime)
            package = _load_json(declared_path)
            relative_roots = _relative_roots_for_declared(declared_path)
        else:
            package = declared
            relative_roots = _relative_roots_for_declared(None)
        return validate_room_package(
            resolve_room_package_paths(
                package, runtime=runtime, relative_roots=relative_roots
            ),
            relative_roots=relative_roots,
        )
    if entry.get("schema") == SCHEMA:
        relative_roots = (
            _relative_roots([Path(catalog_path).expanduser().resolve().parent])
            if catalog_path is not None else _relative_roots(None)
        )
        return validate_room_package(
            resolve_room_package_paths(
                entry, runtime=runtime, relative_roots=relative_roots
            ),
            relative_roots=relative_roots,
        )
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
        source = {}
        if entry.get("room_manifest"):
            source = _load_json(resolve_catalog_room_package_path(
                entry["room_manifest"], catalog_path=catalog_path, runtime=runtime))
        scene = source.get("scene", {})
        package.update(
            visual_scene={"scene_glb": scene.get("scene_id"),
                          "dataset_config": scene.get("dataset_config_path"),
                          "navmesh": scene.get("navmesh_path")},
            walkable_space={"kind": "habitat_navmesh", "path": scene.get("navmesh_path")},
            semantics=deepcopy(source.get("semantics")),
            coordinate_frame={**deepcopy(source.get("coordinate_system", {})),
                              "world_transform": "identity_meter_y_up_right"})
    relative_roots = (
        _relative_roots([Path(catalog_path).expanduser().resolve().parent])
        if catalog_path is not None else _relative_roots(None)
    )
    package = resolve_room_package_paths(
        package, runtime=runtime, relative_roots=relative_roots
    )
    package["validation_errors"] = room_package_errors(
        package, relative_roots=relative_roots
    )
    return package

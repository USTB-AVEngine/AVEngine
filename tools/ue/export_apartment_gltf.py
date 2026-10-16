"""Export the legacy SPEAR apartment as real UE render-surface geometry.

This file runs inside Unreal Editor 5.5 through SPEAR's
``tools/run_editor_script.py``. It intentionally exports StaticMesh render LOD0
from the loaded editor world. It never reads actor bounds as geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import unreal


MAP_ASSET = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"
SPEAR_MAP_PACKAGE = Path(
    "cpp/unreal_projects/SpearSim/Content/SPEAR/Scenes/"
    "apartment_0000/Maps/apartment_0000.umap"
)
SPEAR_PROJECT_CONTENT = Path("cpp/unreal_projects/SpearSim/Content")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--spear-root",
        required=True,
        help="Clean SPEAR checkout whose loaded apartment source is being exported",
    )
    parser.add_argument("--texture-size", type=int, default=512)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_spear_source_snapshot(root: Path, *, capture_phase: str) -> dict:
    """Bind the export to the clean SPEAR commit, project, and map bytes."""

    if not root.is_dir():
        raise FileNotFoundError(f"SPEAR checkout does not exist: {root}")

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SPEAR git {' '.join(arguments)} failed: {result.stderr.strip()}"
            )
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError(f"SPEAR HEAD is not a lowercase full commit: {commit!r}")
    tracked_status = git("status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise RuntimeError("SPEAR tracked worktree must be clean before UE export")
    map_package = (root / SPEAR_MAP_PACKAGE).resolve()
    if not map_package.is_file():
        raise FileNotFoundError(
            f"SPEAR apartment map package is missing: {map_package}"
        )
    expected_project_dir = (root / "cpp" / "unreal_projects" / "SpearSim").resolve()
    return {
        "schema": "avengine_spear_source_snapshot_v1",
        "capture_phase": capture_phase,
        "repository_root": str(root),
        "actual_project_dir": str(expected_project_dir),
        "commit": commit,
        "tracked_worktree_dirty": False,
        "map_asset": MAP_ASSET,
        "map_package_path": str(map_package),
        "map_package_sha256": sha256(map_package),
    }


def selected_project_package_records(
    root: Path, asset_object_paths: set[str]
) -> tuple[list[dict], list[str]]:
    """Bind every directly selected /Game mesh/material package to Git bytes."""

    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Unable to enumerate tracked SPEAR project assets")
    tracked = {value.decode("utf-8") for value in result.stdout.split(b"\0") if value}
    packages: dict[str, set[str]] = {}
    engine_references: list[str] = []
    for object_path in sorted(asset_object_paths):
        package_name = object_path.split(".", 1)[0]
        if package_name.startswith("/Engine/"):
            engine_references.append(object_path)
            continue
        if not package_name.startswith("/Game/"):
            raise RuntimeError(f"Unsupported UE asset root in export: {object_path}")
        packages.setdefault(package_name, set()).add(object_path)

    records: list[dict] = []
    for package_name, object_paths in sorted(packages.items()):
        relative = (
            SPEAR_PROJECT_CONTENT / f"{package_name.removeprefix('/Game/')}.uasset"
        )
        repository_relative = relative.as_posix()
        package_path = (root / relative).resolve()
        if repository_relative not in tracked:
            raise RuntimeError(
                f"Selected UE package is not tracked by SPEAR Git: {package_name}"
            )
        if not package_path.is_file():
            raise FileNotFoundError(
                f"Selected UE package file is missing: {package_path}"
            )
        records.append(
            {
                "package_name": package_name,
                "asset_object_paths": sorted(object_paths),
                "repository_relative_path": repository_relative,
                "resolved_path": str(package_path),
                "byte_size": package_path.stat().st_size,
                "sha256": sha256(package_path),
                "git_tracked": True,
            }
        )
    return records, sorted(engine_references)


def dirty_package_names() -> dict[str, list[str]]:
    utilities = unreal.EditorLoadingAndSavingUtils
    return {
        "content": sorted(
            str(package.get_path_name())
            for package in utilities.get_dirty_content_packages()
        ),
        "maps": sorted(
            str(package.get_path_name())
            for package in utilities.get_dirty_map_packages()
        ),
    }


def require_no_dirty_packages(label: str) -> dict[str, list[str]]:
    dirty = dirty_package_names()
    if dirty["content"] or dirty["maps"]:
        raise RuntimeError(f"{label} has unsaved UE packages: {dirty}")
    return dirty


def vector(value) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def rotator(value) -> list[float]:
    return [float(value.roll), float(value.pitch), float(value.yaw)]


def safe_path(value) -> str | None:
    return value.get_path_name() if value is not None else None


def set_option(options, name: str, value, applied: dict, warnings: list[str]) -> None:
    try:
        options.set_editor_property(name, value)
        applied[name] = str(options.get_editor_property(name))
    except Exception as error:
        warnings.append(f"Unable to set {name}: {type(error).__name__}: {error}")


def message_list(messages, name: str) -> list[str]:
    if messages is None:
        raise RuntimeError("GLTFExporter did not return a readable message object")
    try:
        return [str(value) for value in messages.get_editor_property(name)]
    except Exception as property_error:
        try:
            return [str(value) for value in getattr(messages, name)]
        except Exception as attribute_error:
            raise RuntimeError(
                f"Unable to read GLTFExporter {name} messages: "
                f"property={type(property_error).__name__}: {property_error}; "
                f"attribute={type(attribute_error).__name__}: {attribute_error}"
            ) from attribute_error


def actor_record(actor, components) -> dict:
    transform = actor.get_actor_transform()
    component_records = []
    for component in components:
        mesh = component.get_editor_property("static_mesh")
        materials = []
        try:
            materials = [safe_path(value) for value in component.get_materials()]
        except Exception:
            pass
        component_transform = component.get_world_transform()
        component_records.append(
            {
                "component_name": component.get_name(),
                "static_mesh_asset": safe_path(mesh),
                "material_assets": materials,
                "world_transform_ue": {
                    "translation_cm": vector(component_transform.translation),
                    "rotation_roll_pitch_yaw_deg": rotator(
                        component_transform.rotation.rotator()
                    ),
                    "scale_xyz": vector(component_transform.scale3d),
                },
            }
        )
    return {
        "actor_name": actor.get_name(),
        "actor_label": actor.get_actor_label(),
        "actor_class": actor.get_class().get_name(),
        "relevant_for_level_bounds": bool(
            actor.get_editor_property("relevant_for_level_bounds")
        ),
        "world_transform_ue": {
            "translation_cm": vector(transform.translation),
            "rotation_roll_pitch_yaw_deg": rotator(transform.rotation.rotator()),
            "scale_xyz": vector(transform.scale3d),
        },
        "static_mesh_components": component_records,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve()
    spear_root = Path(args.spear_root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.unlink(missing_ok=True)

    actual_project_dir = Path(unreal.Paths.project_dir()).resolve()
    expected_project_dir = (
        spear_root / "cpp" / "unreal_projects" / "SpearSim"
    ).resolve()
    if actual_project_dir != expected_project_dir:
        raise RuntimeError(
            "UE is running a different project than --spear-root: "
            f"actual={actual_project_dir}, expected={expected_project_dir}"
        )
    dirty_before_reload = require_no_dirty_packages("pre-export editor state")
    source_snapshot = capture_spear_source_snapshot(
        spear_root, capture_phase="before_ue_gltf_export"
    )

    editor = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    world = unreal.EditorLoadingAndSavingUtils.load_map(MAP_ASSET)
    if world is None:
        raise RuntimeError(f"Unable to reload source map from disk: {MAP_ASSET}")
    world = editor.get_editor_world()
    loaded_world = world.get_path_name()
    expected_world_prefix = MAP_ASSET + "."
    if not loaded_world.startswith(expected_world_prefix):
        raise RuntimeError(f"Loaded editor world {loaded_world!r} is not {MAP_ASSET!r}")
    dirty_after_reload = require_no_dirty_packages("reloaded source map")
    engine_version = unreal.SystemLibrary.get_engine_version()
    if not str(engine_version).startswith("5.5."):
        raise RuntimeError(
            f"M1 legacy export requires Unreal Engine 5.5.x: {engine_version}"
        )
    plugin_descriptor_path = (
        Path(unreal.Paths.engine_plugins_dir())
        / "Enterprise"
        / "GLTFExporter"
        / "GLTFExporter.uplugin"
    ).resolve()
    if not plugin_descriptor_path.is_file():
        raise FileNotFoundError(
            f"GLTFExporter descriptor is missing: {plugin_descriptor_path}"
        )
    with plugin_descriptor_path.open("r", encoding="utf-8") as handle:
        plugin_descriptor = json.load(handle)
    all_actors = list(actor_subsystem.get_all_level_actors())
    selected = []
    actor_records = []
    unique_mesh_assets: set[str] = set()
    selected_asset_object_paths: set[str] = set()
    static_mesh_component_count = 0
    for actor in all_actors:
        components = [
            component
            for component in actor.get_components_by_class(unreal.StaticMeshComponent)
            if component.get_editor_property("static_mesh") is not None
        ]
        if not components:
            continue
        if not bool(actor.get_editor_property("relevant_for_level_bounds")):
            continue
        selected.append(actor)
        actor_records.append(actor_record(actor, components))
        static_mesh_component_count += len(components)
        for component in components:
            mesh_path = component.get_editor_property("static_mesh").get_path_name()
            unique_mesh_assets.add(mesh_path)
            selected_asset_object_paths.add(mesh_path)
            selected_asset_object_paths.update(
                safe_path(material)
                for material in component.get_materials()
                if material is not None
            )

    if not selected:
        raise RuntimeError(
            "No relevant StaticMesh actors were found in the editor world"
        )
    project_package_records, engine_asset_references = selected_project_package_records(
        spear_root, selected_asset_object_paths
    )

    options = unreal.GLTFExportOptions()
    applied_options: dict[str, str] = {}
    option_warnings: list[str] = []
    size = unreal.GLTFMaterialBakeSize(
        x=args.texture_size, y=args.texture_size, auto_detect=False
    )
    option_values = {
        "export_uniform_scale": 0.01,
        "export_preview_mesh": False,
        "skip_near_default_values": False,
        "include_copyright_notice": True,
        "export_proxy_materials": True,
        "use_importer_material_mapping": True,
        "export_unlit_materials": True,
        "export_clear_coat_materials": True,
        "export_cloth_materials": True,
        "export_thin_translucent_materials": True,
        "export_specular_glossiness_materials": True,
        "export_emissive_strength": True,
        "bake_material_inputs": unreal.GLTFMaterialBakeMode.USE_MESH_DATA,
        "default_material_bake_size": size,
        "default_level_of_detail": 0,
        "export_source_model": False,
        "export_vertex_colors": False,
        "export_vertex_skin_weights": False,
        "make_skinned_meshes_root": False,
        "use_mesh_quantization": False,
        "export_level_sequences": False,
        "export_animation_sequences": False,
        "texture_image_format": unreal.GLTFTextureImageFormat.PNG,
        "export_texture_transforms": True,
        "adjust_normalmaps": True,
        "export_hidden_in_game": False,
        "export_lights": False,
        "export_cameras": False,
        "export_material_variants": unreal.GLTFMaterialVariantMode.NONE,
    }
    for name, value in option_values.items():
        set_option(options, name, value, applied_options, option_warnings)
    critical_options = {
        "export_uniform_scale",
        "bake_material_inputs",
        "default_material_bake_size",
        "default_level_of_detail",
        "export_source_model",
        "export_lights",
        "export_cameras",
    }
    missing_critical_options = sorted(critical_options - set(applied_options))
    if missing_critical_options or option_warnings:
        raise RuntimeError(
            "Unable to apply all required GLTF export options: "
            f"missing={missing_critical_options}, warnings={option_warnings}"
        )

    output.unlink(missing_ok=True)
    unreal.log(
        f"AVEngine M1: exporting {len(selected)} actors / "
        f"{static_mesh_component_count} StaticMesh components to {output}"
    )
    result = unreal.GLTFExporter.export_to_gltf(
        world, str(output), options, set(selected)
    )
    messages = None
    if isinstance(result, tuple):
        success = bool(result[0])
        if len(result) > 1:
            messages = result[1]
    elif result is None:
        success = False
    elif isinstance(result, bool):
        # Some engine versions may expose only the native return value. Keep
        # failing closed because that form cannot prove the exporter emitted
        # no error messages.
        success = result
    else:
        # UE 5.5's generated Python signature returns GLTFExportMessages on
        # success and None on failure; the native bool is consumed by the
        # wrapper as the success predicate.
        success = True
        messages = result
    if not success or not output.is_file():
        raise RuntimeError(f"GLTFExporter failed: result={result!r}")

    export_messages = {
        "suggestions": message_list(messages, "suggestions"),
        "warnings": message_list(messages, "warnings"),
        "errors": message_list(messages, "errors"),
    }
    dirty_after_export = require_no_dirty_packages("post-export editor state")
    source_snapshot_after_export = capture_spear_source_snapshot(
        spear_root, capture_phase="after_ue_gltf_export"
    )
    before_identity = {
        key: value for key, value in source_snapshot.items() if key != "capture_phase"
    }
    after_identity = {
        key: value
        for key, value in source_snapshot_after_export.items()
        if key != "capture_phase"
    }
    if before_identity != after_identity:
        raise RuntimeError("SPEAR source checkout or map changed during UE export")
    report = {
        "schema": "avengine_legacy_ue_apartment_export_v1",
        "status": "pass" if not export_messages["errors"] else "fail",
        "source_map_asset": MAP_ASSET,
        "source_snapshot": source_snapshot,
        "source_snapshot_after_export": source_snapshot_after_export,
        "actual_project_dir": str(actual_project_dir),
        "dirty_packages": {
            "before_reload": dirty_before_reload,
            "after_reload": dirty_after_reload,
            "after_export": dirty_after_export,
        },
        "loaded_editor_world": loaded_world,
        "engine_version": engine_version,
        "gltf_exporter_plugin": {
            "descriptor_path": str(plugin_descriptor_path),
            "descriptor_sha256": sha256(plugin_descriptor_path),
            "version": plugin_descriptor.get("Version"),
            "version_name": plugin_descriptor.get("VersionName"),
        },
        "geometry_source": "UE StaticMesh render data LOD0",
        "geometry_representation": "real_surface_mesh",
        "uses_actor_bounds_as_geometry": False,
        "coordinate_conversion": "UE (X,Y,Z) cm -> glTF (X,Z,Y) * 0.01 m",
        "selected_actor_count": len(selected),
        "static_mesh_component_count": static_mesh_component_count,
        "unique_static_mesh_asset_count": len(unique_mesh_assets),
        "unique_static_mesh_assets": sorted(unique_mesh_assets),
        "selected_project_asset_package_count": len(project_package_records),
        "selected_project_asset_packages": project_package_records,
        "selected_engine_asset_references": engine_asset_references,
        "export_options": applied_options,
        "option_warnings": option_warnings,
        "export_messages": export_messages,
        "actors": sorted(actor_records, key=lambda value: value["actor_name"]),
        "output": {
            "path": str(output),
            "byte_size": output.stat().st_size,
            "sha256": sha256(output),
        },
    }
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if export_messages["errors"]:
        raise RuntimeError(
            f"GLTFExporter reported errors; failing manifest written to {manifest}"
        )
    unreal.log(f"AVEngine M1: export complete, manifest={manifest}")


if __name__ == "__main__":
    main()

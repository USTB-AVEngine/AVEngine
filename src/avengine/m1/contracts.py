from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    resolve_declared_path,
)
from avengine.contracts.transforms import transform_error, validate_transform


ROOM_SCHEMA = "avengine_room_package_v1"
CAPTURE_SCHEMA = "avengine_m1_capture_request_v1"
EVIDENCE_SCHEMA = "avengine_m1_visual_evidence_v1"
STATUS_VALUES = {"pass", "fail", "blocked", "not_run"}
ROOM_KINDS = {
    "habitat_native",
    "blender_custom",
    "legacy_ue_real_surface_export",
}
MODALITIES = {"rgb", "depth", "semantic"}
IDENTITY_TRANSFORM = {
    "translation_m": [0.0, 0.0, 0.0],
    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
}

_SCHEMA_FILES = {
    ROOM_SCHEMA: "room_package_v1.schema.json",
    CAPTURE_SCHEMA: "m1_capture_request_v1.schema.json",
}


def _json_schema_errors(value: Any, schema_name: str) -> list[str]:
    filename = _SCHEMA_FILES[schema_name]
    source_path = Path(__file__).resolve().parents[3] / "schemas" / filename
    installed_path = Path(sys.prefix) / "share" / "avengine" / "schemas" / filename
    schema_path = source_path if source_path.is_file() else installed_path
    schema = load_json(schema_path)
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"JSON Schema {location}: {error.message}")
    return errors


class ContractError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class ValidatedM1Inputs:
    room_path: Path
    request_path: Path
    room: dict[str, Any]
    request: dict[str, Any]


def _resolved_room_asset_paths(
    inputs: ValidatedM1Inputs, runtime_root: str | Path
) -> dict[str, Path]:
    environment = dict(os.environ)
    environment["AVENGINE_HABITAT_RUNTIME_ROOT"] = str(Path(runtime_root).resolve())
    return {
        asset["role"]: resolve_declared_path(
            asset["path"],
            manifest_dir=inputs.room_path.parent,
            environment=environment,
        )
        for asset in inputs.room["assets"]
    }


def _resolved_graph_path(value: Any, *, base_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _dataset_search_directories(
    dataset: dict[str, Any],
    dataset_path: Path,
    section_name: str,
    environment: dict[str, str],
) -> list[Path]:
    raw = dataset.get(section_name, {}).get("paths", {}).get(".json", [])
    if not isinstance(raw, list):
        return []
    paths: list[Path] = []
    for value in raw:
        if not isinstance(value, str):
            continue
        paths.append(
            resolve_declared_path(
                value,
                manifest_dir=dataset_path.parent,
                environment=environment,
            )
        )
    return paths


def _require_unique_dataset_file(
    *,
    dataset: dict[str, Any],
    dataset_path: Path,
    section_name: str,
    expected_path: Path,
    environment: dict[str, str],
    errors: list[str],
) -> None:
    directories = _dataset_search_directories(
        dataset, dataset_path, section_name, environment
    )
    candidates = [
        (directory / expected_path.name).resolve()
        for directory in directories
        if (directory / expected_path.name).is_file()
    ]
    if len(candidates) != 1 or candidates[0] != expected_path:
        errors.append(
            f"dataset {section_name} search paths must resolve exactly one "
            f"{expected_path.name} at the declared asset path"
        )


def _require_unique_dataset_handle_candidate(
    *,
    dataset: dict[str, Any],
    dataset_path: Path,
    section_name: str,
    handle_query: str,
    expected_path: Path,
    environment: dict[str, str],
    errors: list[str],
) -> None:
    directories = _dataset_search_directories(
        dataset, dataset_path, section_name, environment
    )
    candidates = sorted(
        path.resolve()
        for directory in directories
        if directory.is_dir()
        for path in directory.glob("*.json")
        if handle_query.lower() in str(path.resolve()).lower()
    )
    if candidates != [expected_path]:
        errors.append(
            f"dataset {section_name} handle query {handle_query!r} must have "
            "exactly one declared candidate"
        )


def _config_handle(path: Path, suffix: str) -> str:
    if not path.name.endswith(suffix):
        return path.stem
    return path.name[: -len(suffix)]


def validate_scene_asset_graph(
    inputs: ValidatedM1Inputs, runtime_root: str | Path
) -> list[str]:
    """Replay Habitat's config search graph to the hashed room assets."""

    errors: list[str] = []
    environment = dict(os.environ)
    environment["AVENGINE_HABITAT_RUNTIME_ROOT"] = str(Path(runtime_root).resolve())
    try:
        roles = _resolved_room_asset_paths(inputs, runtime_root)
        scene = inputs.room["scene"]
        dataset_path = resolve_declared_path(
            scene["dataset_config_path"],
            manifest_dir=inputs.room_path.parent,
            environment=environment,
        )
        navmesh_path = resolve_declared_path(
            scene["navmesh_path"],
            manifest_dir=inputs.room_path.parent,
            environment=environment,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        return [f"unable to resolve scene asset graph: {type(error).__name__}: {error}"]

    if dataset_path != roles.get("scene_dataset_config"):
        errors.append(
            "scene dataset config does not resolve to its declared asset role"
        )
    if navmesh_path != roles.get("navmesh"):
        errors.append("scene navmesh does not resolve to its declared asset role")
    if scene.get("navmesh_policy") != "load_declared":
        errors.append(
            "M1 capture requires navmesh_policy='load_declared'; use "
            "build-navmesh before capture"
        )

    try:
        dataset = load_json(dataset_path)
    except (OSError, TypeError, ValueError) as error:
        errors.append(
            f"unable to load scene dataset config: {type(error).__name__}: {error}"
        )
        return errors

    if scene["scene_id_kind"] == "path":
        try:
            scene_path = resolve_declared_path(
                scene["scene_id"],
                manifest_dir=inputs.room_path.parent,
                environment=environment,
            )
        except (OSError, TypeError, ValueError) as error:
            errors.append(
                f"unable to resolve path scene_id: {type(error).__name__}: {error}"
            )
        else:
            if scene_path != roles.get("render_surface_mesh"):
                errors.append(
                    "path scene_id does not resolve to the render_surface_mesh asset"
                )
            stage_section = dataset.get("stages", {})
            stage_patterns = stage_section.get("paths", {}).get(".glb", [])
            try:
                relative_scene = scene_path.relative_to(dataset_path.parent)
            except ValueError:
                relative_scene = None
            if (
                relative_scene is None
                or not isinstance(stage_patterns, list)
                or not any(
                    isinstance(pattern, str) and relative_scene.match(pattern)
                    for pattern in stage_patterns
                )
            ):
                errors.append(
                    "path scene_id is not selected by the dataset stage search paths"
                )

            defaults = stage_section.get("default_attributes", {})
            if not isinstance(defaults, dict):
                defaults = {}
            config_name = scene_path.stem

            def default_asset(key: str) -> Path | None:
                raw = defaults.get(key)
                if not isinstance(raw, str):
                    return None
                expanded = raw.replace("%%CONFIG_NAME_AS_ASSET_FILENAME%%", config_name)
                return _resolved_graph_path(expanded, base_dir=scene_path.parent)

            nav_from_stage = default_asset("nav_asset")
            if nav_from_stage != navmesh_path:
                errors.append(
                    "path stage nav_asset does not resolve to the declared navmesh"
                )
            if scene.get("load_semantic_mesh"):
                if default_asset("semantic_asset") != roles.get(
                    "semantic_surface_mesh"
                ):
                    errors.append(
                        "path stage semantic_asset does not resolve to "
                        "semantic_surface_mesh"
                    )
                if default_asset("semantic_descriptor_filename") != roles.get(
                    "semantic_descriptor"
                ):
                    errors.append(
                        "path stage semantic descriptor does not resolve to "
                        "semantic_descriptor"
                    )
        return errors

    scene_id = scene["scene_id"]
    stage_path = roles.get("stage_config")
    instance_path = roles.get("scene_instance")
    render_path = roles.get("render_surface_mesh")
    if stage_path is None or instance_path is None or render_path is None:
        errors.append("handle scene is missing stage, instance, or render asset roles")
        return errors
    try:
        stage = load_json(stage_path)
        instance = load_json(instance_path)
    except (OSError, TypeError, ValueError) as error:
        errors.append(
            f"unable to load handle scene graph: {type(error).__name__}: {error}"
        )
        return errors

    try:
        navmesh_declared = dataset.get("navmesh_instances", {}).get(scene_id)
        navmesh_from_dataset = (
            resolve_declared_path(
                navmesh_declared,
                manifest_dir=dataset_path.parent,
                environment=environment,
            )
            if isinstance(navmesh_declared, str)
            else None
        )
        render_from_stage = resolve_declared_path(
            stage["render_asset"],
            manifest_dir=stage_path.parent,
            environment=environment,
        )
        collision_from_stage = resolve_declared_path(
            stage["collision_asset"],
            manifest_dir=stage_path.parent,
            environment=environment,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        errors.append(
            f"unable to resolve handle scene graph links: {type(error).__name__}: {error}"
        )
        return errors

    if stage_path.name != f"{scene_id}.stage_config.json":
        errors.append("stage_config filename does not bind the requested scene handle")
    if instance_path.name != f"{scene_id}.scene_instance.json":
        errors.append(
            "scene_instance filename does not bind the requested scene handle"
        )
    _require_unique_dataset_file(
        dataset=dataset,
        dataset_path=dataset_path,
        section_name="stages",
        expected_path=stage_path,
        environment=environment,
        errors=errors,
    )
    _require_unique_dataset_handle_candidate(
        dataset=dataset,
        dataset_path=dataset_path,
        section_name="stages",
        handle_query=scene_id,
        expected_path=stage_path,
        environment=environment,
        errors=errors,
    )
    _require_unique_dataset_file(
        dataset=dataset,
        dataset_path=dataset_path,
        section_name="scene_instances",
        expected_path=instance_path,
        environment=environment,
        errors=errors,
    )
    _require_unique_dataset_handle_candidate(
        dataset=dataset,
        dataset_path=dataset_path,
        section_name="scene_instances",
        handle_query=scene_id,
        expected_path=instance_path,
        environment=environment,
        errors=errors,
    )
    if instance.get("stage_instance", {}).get("template_name") != scene_id:
        errors.append("scene_instance does not select the requested stage handle")
    if navmesh_from_dataset != navmesh_path:
        errors.append(
            "dataset-config navmesh mapping does not match the declared navmesh"
        )
    if render_from_stage != render_path:
        errors.append("stage render_asset does not resolve to render_surface_mesh")
    if collision_from_stage != render_path:
        errors.append("stage collision_asset does not resolve to render_surface_mesh")

    object_instances = instance.get("object_instances", [])
    if not isinstance(object_instances, list):
        errors.append("scene_instance.object_instances must be an array")
        object_instances = []
    if len(object_instances) != len(inputs.request["sources"]):
        errors.append(
            "scene_instance must contain exactly one marker object per M1 source"
        )
    semantic_ids: set[int] = set()
    for source in inputs.request["sources"]:
        source_id = source["source_id"]
        config_path = roles.get(f"object_config_{source_id}")
        object_path = roles.get(f"semantic_object_{source_id}")
        if config_path is None or object_path is None:
            errors.append(f"handle scene is missing object asset roles for {source_id}")
            continue
        _require_unique_dataset_file(
            dataset=dataset,
            dataset_path=dataset_path,
            section_name="objects",
            expected_path=config_path,
            environment=environment,
            errors=errors,
        )
        template_name = _config_handle(config_path, ".object_config.json")
        _require_unique_dataset_handle_candidate(
            dataset=dataset,
            dataset_path=dataset_path,
            section_name="objects",
            handle_query=template_name,
            expected_path=config_path,
            environment=environment,
            errors=errors,
        )
        try:
            object_config = load_json(config_path)
            render_from_object = _resolved_graph_path(
                object_config.get("render_asset"), base_dir=config_path.parent
            )
            collision_from_object = _resolved_graph_path(
                object_config.get("collision_asset"), base_dir=config_path.parent
            )
        except (OSError, TypeError, ValueError) as error:
            errors.append(
                f"unable to load object graph for {source_id}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        if render_from_object != object_path:
            errors.append(
                f"{source_id} object render_asset does not resolve to its declared mesh"
            )
        if collision_from_object != object_path:
            errors.append(
                f"{source_id} object collision_asset does not resolve to its declared mesh"
            )
        semantic_id = object_config.get("semantic_id")
        if not isinstance(semantic_id, int) or isinstance(semantic_id, bool):
            errors.append(f"{source_id} object semantic_id must be an integer")
        elif semantic_id in semantic_ids:
            errors.append("source marker semantic IDs must be pairwise distinct")
        else:
            semantic_ids.add(semantic_id)

        matching_instances = [
            item
            for item in object_instances
            if isinstance(item, dict) and item.get("template_name") == template_name
        ]
        if len(matching_instances) != 1:
            errors.append(f"scene_instance must select {template_name!r} exactly once")
            continue
        instance_object = matching_instances[0]
        if instance_object.get("motion_type") != "STATIC":
            errors.append(f"{source_id} marker object must be STATIC")
        expected_transform = source["world_from_source"]
        if not np.allclose(
            np.asarray(instance_object.get("translation"), dtype=np.float64),
            np.asarray(expected_transform["translation_m"], dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        ):
            errors.append(
                f"{source_id} scene marker translation does not match world_from_source"
            )
        raw_rotation = instance_object.get("rotation")
        if not (
            isinstance(raw_rotation, list)
            and len(raw_rotation) == 4
            and np.allclose(
                np.asarray(
                    [
                        raw_rotation[1],
                        raw_rotation[2],
                        raw_rotation[3],
                        raw_rotation[0],
                    ],
                    dtype=np.float64,
                ),
                np.asarray(expected_transform["rotation_xyzw"], dtype=np.float64),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            errors.append(
                f"{source_id} scene marker rotation does not match world_from_source"
            )

    lighting_path = roles.get("lighting_config")
    if lighting_path is None:
        errors.append("handle scene is missing the lighting_config asset role")
    else:
        _require_unique_dataset_file(
            dataset=dataset,
            dataset_path=dataset_path,
            section_name="light_setups",
            expected_path=lighting_path,
            environment=environment,
            errors=errors,
        )
        lighting_handle = _config_handle(lighting_path, ".lighting_config.json")
        _require_unique_dataset_handle_candidate(
            dataset=dataset,
            dataset_path=dataset_path,
            section_name="light_setups",
            handle_query=lighting_handle,
            expected_path=lighting_path,
            environment=environment,
            errors=errors,
        )
        if instance.get("default_lighting") != lighting_handle:
            errors.append(
                "scene_instance default_lighting does not select lighting_config"
            )
        try:
            lighting = load_json(lighting_path)
        except (OSError, TypeError, ValueError) as error:
            errors.append(
                f"unable to load lighting config: {type(error).__name__}: {error}"
            )
        else:
            if not isinstance(lighting.get("lights"), dict) or not lighting["lights"]:
                errors.append("lighting_config must define at least one light")
    return errors


def validate_recorded_scene_asset_graph(
    inputs: ValidatedM1Inputs,
    runtime_root: str | Path,
    snapshot: Any,
) -> list[str]:
    """Validate a captured Simulator metadata snapshot against declared assets."""

    if not isinstance(snapshot, dict):
        return ["loaded Habitat scene graph snapshot must be an object"]
    errors: list[str] = []
    roles = _resolved_room_asset_paths(inputs, runtime_root)
    scene = inputs.room["scene"]
    expected_dataset = roles.get("scene_dataset_config")
    expected_navmesh = roles.get("navmesh")
    expected_stage_handle = (
        roles.get("render_surface_mesh")
        if scene["scene_id_kind"] == "path"
        else roles.get("stage_config")
    )
    expected_scene_handle = (
        roles.get("render_surface_mesh")
        if scene["scene_id_kind"] == "path"
        else roles.get("scene_instance")
    )

    def expected_string(path: Path | None) -> str | None:
        return str(path) if path is not None else None

    if snapshot.get("active_dataset") != expected_string(expected_dataset):
        errors.append("loaded Habitat active_dataset differs from scene_dataset_config")
    expected_scene_name = (
        roles["render_surface_mesh"].stem
        if scene["scene_id_kind"] == "path"
        else scene["scene_id"]
    )
    if snapshot.get("current_scene") != expected_scene_name:
        errors.append("loaded Habitat current scene name differs from scene_id")
    if snapshot.get("scene_handle_matches") != [expected_string(expected_scene_handle)]:
        errors.append(
            "loaded Habitat scene handle is missing, ambiguous, or unexpected"
        )
    if snapshot.get("stage_template_matches") != [
        expected_string(expected_stage_handle)
    ]:
        errors.append(
            "loaded Habitat stage template is missing, ambiguous, or unexpected"
        )

    stage = snapshot.get("stage")
    if not isinstance(stage, dict):
        errors.append("loaded Habitat stage initialization template is missing")
        stage = {}
    expected_render = expected_string(roles.get("render_surface_mesh"))
    if stage.get("handle") != expected_string(expected_stage_handle):
        errors.append("loaded Habitat stage handle differs from the declared stage")
    if stage.get("render_asset") != expected_render:
        errors.append("loaded Habitat render asset differs from render_surface_mesh")
    if stage.get("collision_asset") != expected_render:
        errors.append("loaded Habitat collision asset differs from render_surface_mesh")

    navmesh = snapshot.get("navmesh")
    if not isinstance(navmesh, dict):
        errors.append("loaded Habitat navmesh record is missing")
        navmesh = {}
    if navmesh.get("declared_path") != expected_string(expected_navmesh):
        errors.append("loaded Habitat navmesh record differs from declared navmesh")
    if navmesh.get("explicit_load_succeeded") is not True:
        errors.append("declared navmesh was not explicitly loaded into Pathfinder")
    if navmesh.get("active_fingerprint") != navmesh.get("declared_fingerprint"):
        errors.append("active Pathfinder data differs from the declared navmesh file")
    requested_settings = navmesh.get("requested_agent_settings")
    declared_settings = navmesh.get("declared_fingerprint", {}).get("settings")
    navigation = inputs.room.get("navigation", {})
    expected_requested = {
        "agent_height": float(navigation.get("agent_height_m", 1.5)),
        "agent_radius": float(navigation.get("agent_radius_m", 0.2)),
        "include_static_objects": bool(navigation.get("include_static_objects", False)),
    }
    if requested_settings != expected_requested:
        errors.append("recorded navmesh agent settings differ from the room contract")
    if not isinstance(declared_settings, dict):
        errors.append("declared navmesh settings are missing from the load snapshot")
    else:
        for key, expected in expected_requested.items():
            actual = declared_settings.get(key)
            if isinstance(expected, bool):
                matches = actual is expected
            else:
                try:
                    matches = math.isclose(
                        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-6
                    )
                except (TypeError, ValueError):
                    matches = False
            if not matches:
                errors.append(
                    f"declared navmesh {key} differs from the room navigation contract"
                )

    if scene["scene_id_kind"] == "path":
        if stage.get("navmesh_asset") != expected_string(expected_navmesh):
            errors.append("loaded path stage navmesh differs from declared navmesh")
        if stage.get("semantic_asset") != expected_string(
            roles.get("semantic_surface_mesh")
        ):
            errors.append("loaded path stage semantic mesh differs from declared asset")
        if stage.get("semantic_descriptor") != expected_string(
            roles.get("semantic_descriptor")
        ):
            errors.append(
                "loaded path stage semantic descriptor differs from declared asset"
            )
        return errors

    expected_objects: dict[str, dict[str, Any]] = {}
    for source in inputs.request["sources"]:
        source_id = source["source_id"]
        config_path = roles.get(f"object_config_{source_id}")
        object_path = roles.get(f"semantic_object_{source_id}")
        if config_path is None or object_path is None:
            continue
        config = load_json(config_path)
        expected_objects[source_id] = {
            "source_id": source_id,
            "creation_config": str(config_path),
            "render_asset": str(object_path),
            "collision_asset": str(object_path),
            "semantic_id": config.get("semantic_id"),
            "translation_m": source["world_from_source"]["translation_m"],
            "rotation_xyzw": source["world_from_source"]["rotation_xyzw"],
        }
    template_matches = snapshot.get("object_template_matches")
    if not isinstance(template_matches, dict):
        errors.append("loaded Habitat object template match record is missing")
        template_matches = {}
    for source_id, expected in expected_objects.items():
        if template_matches.get(source_id) != [expected["creation_config"]]:
            errors.append(
                f"loaded Habitat object template for {source_id} is ambiguous or unexpected"
            )

    actual_objects = snapshot.get("objects")
    if not isinstance(actual_objects, list):
        errors.append("loaded Habitat object instance record is missing")
        actual_objects = []
    indexed_objects = {
        item.get("source_id"): item for item in actual_objects if isinstance(item, dict)
    }
    if set(indexed_objects) != set(expected_objects) or len(actual_objects) != len(
        expected_objects
    ):
        errors.append("loaded Habitat object instances do not exactly match M1 sources")
    for source_id, expected in expected_objects.items():
        actual = indexed_objects.get(source_id, {})
        for key in (
            "creation_config",
            "render_asset",
            "collision_asset",
            "semantic_id",
        ):
            if actual.get(key) != expected[key]:
                errors.append(
                    f"loaded {source_id} object {key} differs from declaration"
                )
        for key in ("translation_m", "rotation_xyzw"):
            try:
                matches = np.allclose(
                    np.asarray(actual.get(key), dtype=np.float64),
                    np.asarray(expected[key], dtype=np.float64),
                    rtol=0.0,
                    atol=1e-6,
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                errors.append(
                    f"loaded {source_id} object {key} differs from source pose"
                )

    lighting_path = roles.get("lighting_config")
    lighting = snapshot.get("lighting")
    if not isinstance(lighting, dict):
        errors.append("loaded Habitat lighting record is missing")
        lighting = {}
    if lighting.get("template_matches") != [expected_string(lighting_path)]:
        errors.append("loaded Habitat lighting template is ambiguous or unexpected")
    if lighting.get("selected_handle") != expected_string(lighting_path):
        errors.append("loaded Habitat stage selected a different lighting template")
    if lighting.get("selected_template_exists") is not True:
        errors.append("loaded Habitat selected lighting template is not registered")
    if lighting.get("current_setup_matches_selected") is not True:
        errors.append("loaded Habitat current light setup differs from stage selection")
    if lighting_path is not None:
        lighting_config = load_json(lighting_path)
        expected_count = len(lighting_config.get("lights", {}))
        if lighting.get("current_light_count") != expected_count:
            errors.append("loaded Habitat light count differs from lighting_config")
    return errors


def validate_loaded_scene_asset_graph(
    inputs: ValidatedM1Inputs,
    runtime_root: str | Path,
    simulator: Any,
    *,
    declared_navmesh_loaded: bool,
) -> tuple[list[str], dict[str, Any]]:
    """Read the final Simulator templates and bind them to declared assets."""

    roles = _resolved_room_asset_paths(inputs, runtime_root)
    scene = inputs.room["scene"]

    def normalized_path(value: Any, *, base_dir: Path | None = None) -> str | None:
        if not isinstance(value, str) or not value:
            return None
        path = Path(value)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        return str(path.resolve())

    def matching_paths(
        handles: list[str], expected: Path | None, *, handle_query: str
    ) -> list[str]:
        if expected is None:
            return []
        return sorted(
            str(Path(handle).resolve())
            for handle in handles
            if handle_query.lower() in handle.lower()
        )

    expected_stage_handle = (
        roles.get("render_surface_mesh")
        if scene["scene_id_kind"] == "path"
        else roles.get("stage_config")
    )
    expected_scene_handle = (
        roles.get("render_surface_mesh")
        if scene["scene_id_kind"] == "path"
        else roles.get("scene_instance")
    )
    mediator = simulator.metadata_mediator
    stage_manager = mediator.stage_template_manager
    scene_handles = list(mediator.get_scene_handles())
    stage_handles = list(stage_manager.get_template_handles())
    stage = simulator.get_stage_initialization_template()
    stage_record: dict[str, Any] | None = None
    if stage is not None:
        stage_dir = Path(stage.file_directory).resolve()
        stage_record = {
            "handle": normalized_path(stage.handle, base_dir=stage_dir),
            "render_asset": normalized_path(stage.render_asset_fullpath),
            "collision_asset": normalized_path(stage.collision_asset_fullpath),
            "navmesh_asset": normalized_path(
                stage.navmesh_asset_handle, base_dir=stage_dir
            ),
            "semantic_asset": normalized_path(stage.semantic_asset_fullpath),
            "semantic_descriptor": normalized_path(stage.house_fq_filename),
        }

    def pathfinder_fingerprint(pathfinder: Any) -> dict[str, Any]:
        settings = pathfinder.nav_mesh_settings
        vertices = np.ascontiguousarray(
            np.asarray(
                [
                    [float(component) for component in value]
                    for value in pathfinder.build_navmesh_vertices()
                ],
                dtype="<f4",
            ).reshape(-1, 3)
        )
        indices = np.ascontiguousarray(
            np.asarray(pathfinder.build_navmesh_vertex_indices(), dtype="<i4").reshape(
                -1
            )
        )
        bounds = pathfinder.get_bounds()
        boolean_settings = {
            "filter_low_hanging_obstacles",
            "filter_ledge_spans",
            "filter_walkable_low_height_spans",
            "include_static_objects",
        }
        setting_names = (
            "agent_height",
            "agent_radius",
            "agent_max_climb",
            "agent_max_slope",
            "cell_size",
            "cell_height",
            "filter_low_hanging_obstacles",
            "filter_ledge_spans",
            "filter_walkable_low_height_spans",
            "include_static_objects",
            "region_min_size",
            "region_merge_size",
            "edge_max_len",
            "edge_max_error",
            "verts_per_poly",
            "detail_sample_dist",
            "detail_sample_max_error",
        )
        core = {
            "schema": "avengine_pathfinder_fingerprint_v1",
            "settings": {
                key: (
                    bool(getattr(settings, key))
                    if key in boolean_settings
                    else float(getattr(settings, key))
                )
                for key in setting_names
            },
            "vertices": {
                "dtype": vertices.dtype.str,
                "shape": list(vertices.shape),
                "sha256": hashlib.sha256(vertices.tobytes(order="C")).hexdigest(),
            },
            "indices": {
                "dtype": indices.dtype.str,
                "shape": list(indices.shape),
                "sha256": hashlib.sha256(indices.tobytes(order="C")).hexdigest(),
            },
        }
        return {
            **core,
            "fingerprint_sha256": canonical_json_sha256(core),
            "diagnostics": {
                "navigable_area_m2": float(pathfinder.navigable_area),
                "island_count": int(pathfinder.num_islands),
                "bounds": [
                    [float(component) for component in bounds[0]],
                    [float(component) for component in bounds[1]],
                ],
            },
        }

    declared_pathfinder = simulator.pathfinder.__class__()
    declared_pathfinder_loaded = bool(
        declared_pathfinder.load_nav_mesh(str(roles["navmesh"]))
    )
    navigation = inputs.room.get("navigation", {})
    snapshot: dict[str, Any] = {
        "active_dataset": normalized_path(simulator.active_dataset),
        "current_scene": str(simulator.curr_scene_name),
        "scene_handle_matches": matching_paths(
            scene_handles,
            expected_scene_handle,
            handle_query=(
                str(roles["render_surface_mesh"])
                if scene["scene_id_kind"] == "path"
                else scene["scene_id"]
            ),
        ),
        "stage_template_matches": matching_paths(
            stage_handles,
            expected_stage_handle,
            handle_query=(
                str(roles["render_surface_mesh"])
                if scene["scene_id_kind"] == "path"
                else scene["scene_id"]
            ),
        ),
        "stage": stage_record,
        "navmesh": {
            "declared_path": str(roles["navmesh"]),
            "explicit_load_succeeded": bool(
                declared_navmesh_loaded and declared_pathfinder_loaded
            ),
            "requested_agent_settings": {
                "agent_height": float(navigation.get("agent_height_m", 1.5)),
                "agent_radius": float(navigation.get("agent_radius_m", 0.2)),
                "include_static_objects": bool(
                    navigation.get("include_static_objects", False)
                ),
            },
            "active_fingerprint": pathfinder_fingerprint(simulator.pathfinder),
            "declared_fingerprint": (
                pathfinder_fingerprint(declared_pathfinder)
                if declared_pathfinder_loaded
                else None
            ),
        },
        "object_template_matches": {},
        "objects": [],
        "lighting": {"template_matches": [], "current_light_count": 0},
    }

    if scene["scene_id_kind"] == "handle":
        object_manager = mediator.object_template_manager
        object_handles = list(object_manager.get_template_handles())
        expected_by_config: dict[str, str] = {}
        for source in inputs.request["sources"]:
            source_id = source["source_id"]
            config_path = roles.get(f"object_config_{source_id}")
            if config_path is None:
                continue
            expected_by_config[str(config_path)] = source_id
            snapshot["object_template_matches"][source_id] = matching_paths(
                object_handles,
                config_path,
                handle_query=_config_handle(config_path, ".object_config.json"),
            )

        object_records: list[dict[str, Any]] = []
        rigid_manager = simulator.get_rigid_object_manager()
        for object_handle in rigid_manager.get_object_handles():
            obj = rigid_manager.get_object_by_handle(object_handle)
            attributes = obj.creation_attributes
            config_path = normalized_path(
                attributes.handle, base_dir=Path(attributes.file_directory).resolve()
            )
            rotation = obj.rotation
            object_records.append(
                {
                    "source_id": expected_by_config.get(config_path),
                    "creation_config": config_path,
                    "render_asset": normalized_path(attributes.render_asset_fullpath),
                    "collision_asset": normalized_path(
                        attributes.collision_asset_fullpath
                    ),
                    "semantic_id": int(obj.semantic_id),
                    "translation_m": [float(value) for value in obj.translation],
                    "rotation_xyzw": [
                        float(rotation.vector.x),
                        float(rotation.vector.y),
                        float(rotation.vector.z),
                        float(rotation.scalar),
                    ],
                }
            )
        snapshot["objects"] = sorted(
            object_records, key=lambda item: str(item.get("source_id"))
        )

        lighting_path = roles.get("lighting_config")
        lighting_manager = mediator.lighting_template_manager
        raw_light_key = stage.get("light_setup_key") if stage is not None else None
        selected_light_path = normalized_path(raw_light_key)
        light_template_exists = bool(
            isinstance(raw_light_key, str)
            and lighting_manager.get_library_has_handle(raw_light_key)
        )
        current_matches_selected = False
        if light_template_exists:
            current_matches_selected = bool(
                simulator.get_current_light_setup()
                == simulator.get_light_setup(raw_light_key)
            )
        snapshot["lighting"] = {
            "template_matches": matching_paths(
                list(lighting_manager.get_template_handles()),
                lighting_path,
                handle_query=(
                    _config_handle(lighting_path, ".lighting_config.json")
                    if lighting_path is not None
                    else ""
                ),
            ),
            "selected_handle": selected_light_path,
            "selected_template_exists": light_template_exists,
            "current_setup_matches_selected": current_matches_selected,
            "current_light_count": len(simulator.get_current_light_setup()),
        }

    errors = validate_recorded_scene_asset_graph(inputs, runtime_root, snapshot)
    return errors, snapshot


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _required_string(value: Any, name: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{name} must be a non-empty string")


def _reject_extra_keys(
    value: dict[str, Any], allowed: set[str], name: str, errors: list[str]
) -> None:
    extra = sorted(set(value) - allowed)
    if extra:
        errors.append(f"{name} has unsupported fields: {extra}")


def _require_vec3(value: Any, name: str, errors: list[str]) -> bool:
    valid = (
        isinstance(value, list)
        and len(value) == 3
        and all(_is_number(item) for item in value)
    )
    if not valid:
        errors.append(f"{name} must contain three finite numbers")
    return valid


def _check_coordinate_system(value: Any, errors: list[str]) -> None:
    expected = {
        "handedness": "right",
        "up_axis": "+Y",
        "forward_axis": "-Z",
        "linear_unit": "meter",
        "quaternion_order": "xyzw",
    }
    if not isinstance(value, dict):
        errors.append("coordinate_system must be an object")
        return
    _reject_extra_keys(value, set(expected), "coordinate_system", errors)
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            errors.append(
                f"coordinate_system.{key} must be {expected_value!r}, "
                f"got {value.get(key)!r}"
            )


def validate_room_manifest(room: dict[str, Any]) -> list[str]:
    errors: list[str] = _json_schema_errors(room, ROOM_SCHEMA)
    _reject_extra_keys(
        room,
        {
            "schema",
            "room_id",
            "room_kind",
            "geometry_representation",
            "coordinate_system",
            "scene",
            "assets",
            "semantics",
            "navigation",
            "openings",
            "connectivity_pairs",
            "ray_checks",
            "acoustics",
            "provenance",
            "surface_audit",
        },
        "room manifest",
        errors,
    )
    if room.get("schema") != ROOM_SCHEMA:
        errors.append(f"schema must be {ROOM_SCHEMA!r}")
    _required_string(room.get("room_id"), "room_id", errors)

    room_kind = room.get("room_kind")
    if room_kind not in ROOM_KINDS:
        errors.append(f"room_kind must be one of {sorted(ROOM_KINDS)}")

    representation = room.get("geometry_representation")
    if representation not in {"real_surface_mesh", "debug_aabb_proxy"}:
        errors.append(
            "geometry_representation must be 'real_surface_mesh' or 'debug_aabb_proxy'"
        )
    if room_kind in {"blender_custom", "legacy_ue_real_surface_export"}:
        if representation != "real_surface_mesh":
            errors.append(f"{room_kind} cannot use a debug AABB proxy")

    _check_coordinate_system(room.get("coordinate_system"), errors)

    scene = room.get("scene")
    if not isinstance(scene, dict):
        errors.append("scene must be an object")
    else:
        _reject_extra_keys(
            scene,
            {
                "scene_id_kind",
                "scene_id",
                "dataset_config_path",
                "navmesh_path",
                "navmesh_policy",
                "load_semantic_mesh",
                "enable_physics",
            },
            "scene",
            errors,
        )
        if scene.get("scene_id_kind") not in {"path", "handle"}:
            errors.append("scene.scene_id_kind must be 'path' or 'handle'")
        _required_string(scene.get("scene_id"), "scene.scene_id", errors)
        _required_string(
            scene.get("dataset_config_path"), "scene.dataset_config_path", errors
        )
        if scene.get("navmesh_policy") not in {
            "load_declared",
            "recompute_if_missing",
        }:
            errors.append(
                "scene.navmesh_policy must be 'load_declared' or 'recompute_if_missing'"
            )
        _required_string(scene.get("navmesh_path"), "scene.navmesh_path", errors)
        for boolean_field in ("load_semantic_mesh", "enable_physics"):
            if not isinstance(scene.get(boolean_field), bool):
                errors.append(f"scene.{boolean_field} must be a boolean")

    assets = room.get("assets")
    if not isinstance(assets, list) or not assets:
        errors.append("assets must be a non-empty array")
    else:
        roles: set[str] = set()
        for index, asset in enumerate(assets):
            prefix = f"assets[{index}]"
            if not isinstance(asset, dict):
                errors.append(f"{prefix} must be an object")
                continue
            _reject_extra_keys(
                asset,
                {"role", "path", "license", "redistribution"},
                prefix,
                errors,
            )
            _required_string(asset.get("role"), f"{prefix}.role", errors)
            _required_string(asset.get("path"), f"{prefix}.path", errors)
            role = asset.get("role")
            if isinstance(role, str):
                if role in roles:
                    errors.append(f"asset role is duplicated: {role}")
                roles.add(role)

        required_roles = {"render_surface_mesh", "scene_dataset_config"}
        if room_kind == "habitat_native":
            required_roles.add("navmesh")
            if isinstance(scene, dict) and scene.get("load_semantic_mesh") is True:
                required_roles.update(
                    {"semantic_surface_mesh", "semantic_descriptor"}
                )
        elif room_kind == "blender_custom":
            required_roles.update(
                {
                    "blender_build_report",
                    "stage_config",
                    "scene_instance",
                    "lighting_config",
                    "navmesh",
                }
            )
        elif room_kind == "legacy_ue_real_surface_export":
            required_roles.update(
                {
                    "ue_export_manifest",
                    "real_surface_mesh_audit",
                    "legacy_source_map_package",
                    "stage_config",
                    "scene_instance",
                    "lighting_config",
                    "navmesh",
                }
            )
        missing_roles = sorted(required_roles - roles)
        if missing_roles:
            errors.append(f"assets are missing required roles: {missing_roles}")
        role_paths = {
            asset.get("role"): asset.get("path")
            for asset in assets
            if isinstance(asset, dict)
        }
        if isinstance(scene, dict):
            if scene.get("dataset_config_path") != role_paths.get(
                "scene_dataset_config"
            ):
                errors.append(
                    "scene.dataset_config_path must equal the scene_dataset_config "
                    "asset path"
                )
            if scene.get("navmesh_path") != role_paths.get("navmesh"):
                errors.append("scene.navmesh_path must equal the navmesh asset path")
            if scene.get("scene_id_kind") == "path" and scene.get(
                "scene_id"
            ) != role_paths.get("render_surface_mesh"):
                errors.append(
                    "path scene.scene_id must equal the render_surface_mesh asset path"
                )

    semantics = room.get("semantics")
    if not isinstance(semantics, dict):
        errors.append("semantics must be an object")
    else:
        _reject_extra_keys(
            semantics, {"interpretation", "id_to_label"}, "semantics", errors
        )
        _required_string(
            semantics.get("interpretation"), "semantics.interpretation", errors
        )
        if "id_to_label" in semantics and not isinstance(
            semantics.get("id_to_label"), dict
        ):
            errors.append("semantics.id_to_label must be an object")

    navigation = room.get("navigation")
    if not isinstance(navigation, dict):
        errors.append("navigation must be an object")
    else:
        _reject_extra_keys(
            navigation,
            {"agent_height_m", "agent_radius_m", "include_static_objects"},
            "navigation",
            errors,
        )
        for field in ("agent_height_m", "agent_radius_m"):
            if not _is_number(navigation.get(field)) or navigation.get(field, 0) <= 0:
                errors.append(f"navigation.{field} must be positive")
        if not isinstance(navigation.get("include_static_objects"), bool):
            errors.append("navigation.include_static_objects must be a boolean")

    openings = room.get("openings")
    if not isinstance(openings, list):
        errors.append("openings must be an array")
        openings = []
    opening_ids: set[str] = set()
    for index, opening in enumerate(openings):
        prefix = f"openings[{index}]"
        if not isinstance(opening, dict):
            errors.append(f"{prefix} must be an object")
            continue
        _reject_extra_keys(
            opening, {"opening_id", "kind", "description"}, prefix, errors
        )
        opening_id = opening.get("opening_id")
        _required_string(opening_id, f"{prefix}.opening_id", errors)
        if isinstance(opening_id, str):
            if opening_id in opening_ids:
                errors.append(f"duplicate opening_id: {opening_id}")
            opening_ids.add(opening_id)
        if opening.get("kind") not in {"door", "window", "archway", "other"}:
            errors.append(f"{prefix}.kind is invalid")
        _required_string(opening.get("description"), f"{prefix}.description", errors)

    provenance = room.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
    else:
        _required_string(provenance.get("source"), "provenance.source", errors)
        _required_string(
            provenance.get("source_revision"), "provenance.source_revision", errors
        )

    acoustics = room.get("acoustics")
    if not isinstance(acoustics, dict) or acoustics.get("status") != "deferred_to_m3":
        errors.append("acoustics.status must explicitly be 'deferred_to_m3'")
    elif isinstance(acoustics, dict):
        _reject_extra_keys(acoustics, {"status", "reason"}, "acoustics", errors)
        _required_string(acoustics.get("reason"), "acoustics.reason", errors)

    connectivity_pairs = room.get("connectivity_pairs", [])
    if not isinstance(connectivity_pairs, list):
        errors.append("connectivity_pairs must be an array")
    else:
        pair_ids: set[str] = set()
        for index, pair in enumerate(connectivity_pairs):
            prefix = f"connectivity_pairs[{index}]"
            if not isinstance(pair, dict):
                errors.append(f"{prefix} must be an object")
                continue
            _reject_extra_keys(pair, {"pair_id", "start_m", "end_m"}, prefix, errors)
            _required_string(pair.get("pair_id"), f"{prefix}.pair_id", errors)
            pair_id = pair.get("pair_id")
            if isinstance(pair_id, str):
                if pair_id in pair_ids:
                    errors.append(f"duplicate connectivity pair: {pair_id}")
                pair_ids.add(pair_id)
            for endpoint in ("start_m", "end_m"):
                _require_vec3(pair.get(endpoint), f"{prefix}.{endpoint}", errors)

    if not connectivity_pairs:
        errors.append("M1 room must declare at least one connectivity pair")

    ray_checks = room.get("ray_checks", [])
    if not isinstance(ray_checks, list):
        errors.append("ray_checks must be an array")
    else:
        ray_ids: set[str] = set()
        for index, check in enumerate(ray_checks):
            prefix = f"ray_checks[{index}]"
            if not isinstance(check, dict):
                errors.append(f"{prefix} must be an object")
                continue
            _reject_extra_keys(
                check,
                {"check_id", "origin_m", "direction", "expectation", "distance_m"},
                prefix,
                errors,
            )
            check_id = check.get("check_id")
            _required_string(check_id, f"{prefix}.check_id", errors)
            if isinstance(check_id, str):
                if check_id in ray_ids:
                    errors.append(f"duplicate ray check: {check_id}")
                ray_ids.add(check_id)
            if check.get("expectation") not in {"clear_until_m", "hit_within_m"}:
                errors.append(
                    f"{prefix}.expectation must be clear_until_m or hit_within_m"
                )
            if (
                not _is_number(check.get("distance_m"))
                or check.get("distance_m", 0) <= 0
            ):
                errors.append(f"{prefix}.distance_m must be positive")
            _require_vec3(check.get("origin_m"), f"{prefix}.origin_m", errors)
            if _require_vec3(check.get("direction"), f"{prefix}.direction", errors):
                norm = float(
                    np.linalg.norm(np.asarray(check["direction"], dtype=float))
                )
                if abs(norm - 1.0) > 1e-6:
                    errors.append(f"{prefix}.direction must be unit length")

    surface_audit = room.get("surface_audit")
    if room_kind in {"blender_custom", "legacy_ue_real_surface_export"}:
        if not isinstance(surface_audit, dict):
            errors.append(f"{room_kind} requires surface_audit evidence")
        else:
            if surface_audit.get("aabb_proxy") is not False:
                errors.append("surface_audit.aabb_proxy must be false")
            _required_string(
                surface_audit.get("method"), "surface_audit.method", errors
            )

    if room_kind == "blender_custom":
        if not openings:
            errors.append("blender_custom must declare modeled openings")
        expectations = {
            check.get("expectation") for check in ray_checks if isinstance(check, dict)
        }
        if expectations != {"clear_until_m", "hit_within_m"}:
            errors.append(
                "blender_custom requires clear-opening and solid-control ray checks"
            )

    if room_kind == "legacy_ue_real_surface_export" and isinstance(surface_audit, dict):
        triangles = surface_audit.get("triangle_count")
        if (
            not isinstance(triangles, int)
            or isinstance(triangles, bool)
            or triangles <= 252
        ):
            errors.append("legacy surface_audit.triangle_count must exceed 252")
        if surface_audit.get("real_surface_gate_status") != "pass":
            errors.append(
                "legacy surface_audit.real_surface_gate_status must be 'pass'"
            )
        mesh_sha256 = surface_audit.get("mesh_sha256")
        if (
            not isinstance(mesh_sha256, str)
            or len(mesh_sha256) != 64
            or any(character not in "0123456789abcdef" for character in mesh_sha256)
        ):
            errors.append("legacy surface_audit.mesh_sha256 must be lowercase SHA-256")

    return errors


def validate_capture_request(
    request: dict[str, Any], *, room_id: str | None = None
) -> list[str]:
    errors: list[str] = _json_schema_errors(request, CAPTURE_SCHEMA)
    _reject_extra_keys(
        request,
        {
            "schema",
            "request_id",
            "room_id",
            "seed",
            "primary_camera_rig",
            "listener",
            "sources",
            "qa_views",
        },
        "capture request",
        errors,
    )
    if request.get("schema") != CAPTURE_SCHEMA:
        errors.append(f"schema must be {CAPTURE_SCHEMA!r}")
    _required_string(request.get("request_id"), "request_id", errors)
    _required_string(request.get("room_id"), "room_id", errors)
    if room_id is not None and request.get("room_id") != room_id:
        errors.append(
            f"request room_id {request.get('room_id')!r} does not match {room_id!r}"
        )
    if not isinstance(request.get("seed"), int) or isinstance(
        request.get("seed"), bool
    ):
        errors.append("seed must be an integer")

    rig = request.get("primary_camera_rig")
    if not isinstance(rig, dict):
        errors.append("primary_camera_rig must be one object")
        return errors

    _reject_extra_keys(
        rig,
        {"rig_id", "view_id", "world_from_rig", "shared_calibration", "modalities"},
        "primary_camera_rig",
        errors,
    )
    if rig.get("rig_id") != "camera_rig_0":
        errors.append("primary_camera_rig.rig_id must be 'camera_rig_0'")
    if rig.get("view_id") != "view0":
        errors.append("primary_camera_rig.view_id must be 'view0'")
    errors.extend(
        validate_transform(
            rig.get("world_from_rig"), name="primary_camera_rig.world_from_rig"
        )
    )

    calibration = rig.get("shared_calibration")
    if not isinstance(calibration, dict):
        errors.append("primary_camera_rig.shared_calibration must be an object")
        calibration = {}
    else:
        _reject_extra_keys(
            calibration,
            {
                "projection",
                "resolution_hw",
                "hfov_degrees",
                "near_m",
                "far_m",
                "rig_from_sensor",
            },
            "primary_camera_rig.shared_calibration",
            errors,
        )
    resolution = calibration.get("resolution_hw")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in resolution
        )
    ):
        errors.append(
            "shared_calibration.resolution_hw must contain two positive integers"
        )
    if calibration.get("projection") != "pinhole":
        errors.append("shared_calibration.projection must be 'pinhole'")
    for key in ("hfov_degrees", "near_m", "far_m"):
        if not _is_number(calibration.get(key)) or calibration.get(key, 0) <= 0:
            errors.append(f"shared_calibration.{key} must be positive")
    if _is_number(calibration.get("hfov_degrees")) and not (
        0.0 < float(calibration["hfov_degrees"]) < 180.0
    ):
        errors.append("shared_calibration.hfov_degrees must be smaller than 180")
    if (
        _is_number(calibration.get("near_m"))
        and _is_number(calibration.get("far_m"))
        and calibration["near_m"] >= calibration["far_m"]
    ):
        errors.append("shared_calibration.near_m must be smaller than far_m")
    errors.extend(
        validate_transform(
            calibration.get("rig_from_sensor"),
            name="primary_camera_rig.shared_calibration.rig_from_sensor",
        )
    )
    if isinstance(calibration.get("rig_from_sensor"), dict):
        try:
            identity_error = transform_error(
                calibration["rig_from_sensor"], IDENTITY_TRANSFORM
            )
            if identity_error > 1e-9:
                errors.append(
                    "rig_from_sensor must be identity because world_from_rig is the "
                    "formal camera/listener viewpoint"
                )
        except (KeyError, TypeError, ValueError):
            pass

    modalities = rig.get("modalities")
    if not isinstance(modalities, list):
        errors.append("primary_camera_rig.modalities must be an array")
        modalities = []
    actual_modalities: set[str] = set()
    uuids: set[str] = set()
    for index, modality in enumerate(modalities):
        prefix = f"primary_camera_rig.modalities[{index}]"
        if not isinstance(modality, dict):
            errors.append(f"{prefix} must be an object")
            continue
        extra_modality_fields = sorted(set(modality) - {"modality", "sensor_uuid"})
        if extra_modality_fields:
            errors.append(
                f"{prefix} has modality-specific calibration fields: "
                f"{extra_modality_fields}"
            )
        name = modality.get("modality")
        uuid = modality.get("sensor_uuid")
        if name not in MODALITIES:
            errors.append(f"{prefix}.modality must be one of {sorted(MODALITIES)}")
        elif name in actual_modalities:
            errors.append(f"duplicate modality: {name}")
        if isinstance(name, str):
            actual_modalities.add(name)
        _required_string(uuid, f"{prefix}.sensor_uuid", errors)
        if isinstance(uuid, str) and uuid in uuids:
            errors.append(f"duplicate sensor_uuid: {uuid}")
        if isinstance(uuid, str):
            uuids.add(uuid)
    if actual_modalities != MODALITIES:
        errors.append(
            f"modalities must be exactly {sorted(MODALITIES)}, got "
            f"{sorted(actual_modalities)}"
        )

    listener = request.get("listener")
    if not isinstance(listener, dict):
        errors.append("listener must be one object")
    else:
        _reject_extra_keys(
            listener,
            {"listener_id", "attached_to", "rig_from_listener"},
            "listener",
            errors,
        )
        if listener.get("listener_id") != "listener0":
            errors.append("listener.listener_id must be 'listener0'")
        if listener.get("attached_to") != rig.get("rig_id"):
            errors.append("listener.attached_to must name primary_camera_rig.rig_id")
        errors.extend(
            validate_transform(
                listener.get("rig_from_listener"), name="listener.rig_from_listener"
            )
        )
        if not errors or (
            isinstance(listener.get("rig_from_listener"), dict)
            and isinstance(calibration.get("rig_from_sensor"), dict)
        ):
            try:
                error = transform_error(
                    listener["rig_from_listener"], calibration["rig_from_sensor"]
                )
                if error > 1e-9:
                    errors.append(
                        "listener must be co-located and co-oriented with the three "
                        f"visual modalities (transform error {error})"
                    )
            except (KeyError, TypeError, ValueError):
                pass

    sources = request.get("sources")
    if not isinstance(sources, list):
        errors.append("sources must be an array independent of sensor modalities")
    else:
        if len(sources) < 2:
            errors.append(
                "M1 capture requires at least two independently named sources"
            )
        source_ids: set[str] = set()
        for index, source in enumerate(sources):
            prefix = f"sources[{index}]"
            if not isinstance(source, dict):
                errors.append(f"{prefix} must be an object")
                continue
            _reject_extra_keys(
                source, {"source_id", "world_from_source"}, prefix, errors
            )
            source_id = source.get("source_id")
            _required_string(source_id, f"{prefix}.source_id", errors)
            if isinstance(source_id, str) and source_id in source_ids:
                errors.append(f"duplicate source_id: {source_id}")
            if isinstance(source_id, str):
                source_ids.add(source_id)
            errors.extend(
                validate_transform(
                    source.get("world_from_source"),
                    name=f"{prefix}.world_from_source",
                )
            )
        valid_source_transforms = [
            source["world_from_source"]
            for source in sources
            if isinstance(source, dict)
            and isinstance(source.get("world_from_source"), dict)
            and not validate_transform(
                source["world_from_source"], name="source.world_from_source"
            )
        ]
        for left_index, left in enumerate(valid_source_transforms):
            for right in valid_source_transforms[left_index + 1 :]:
                if transform_error(left, right) <= 1e-9:
                    errors.append(
                        "M1 source world transforms must be pairwise distinct"
                    )
                    break
            else:
                continue
            break

    qa_views = request.get("qa_views", [])
    if not isinstance(qa_views, list):
        errors.append("qa_views must be an array")
    else:
        if not qa_views:
            errors.append("M1 capture requires at least one QA-only topdown view")
        qa_ids: set[str] = set()
        for index, view in enumerate(qa_views):
            prefix = f"qa_views[{index}]"
            if not isinstance(view, dict):
                errors.append(f"{prefix} must be an object")
                continue
            _reject_extra_keys(
                view,
                {"qa_id", "kind", "meters_per_pixel", "height_m"},
                prefix,
                errors,
            )
            if view.get("kind") != "topdown":
                errors.append(f"{prefix}.kind must be 'topdown'")
            if "view_id" in view:
                errors.append(
                    f"{prefix} is QA-only and must not declare a formal view_id"
                )
            qa_id = view.get("qa_id")
            _required_string(qa_id, f"{prefix}.qa_id", errors)
            if isinstance(qa_id, str):
                if qa_id in qa_ids:
                    errors.append(f"duplicate qa_id: {qa_id}")
                qa_ids.add(qa_id)
            if (
                not _is_number(view.get("meters_per_pixel"))
                or view.get("meters_per_pixel", 0) <= 0
            ):
                errors.append(f"{prefix}.meters_per_pixel must be positive")
            if not _is_number(view.get("height_m")):
                errors.append(f"{prefix}.height_m must be finite")

    return errors


def load_and_validate_inputs(
    room_path: str | Path, request_path: str | Path
) -> ValidatedM1Inputs:
    resolved_room = Path(room_path).resolve()
    resolved_request = Path(request_path).resolve()
    room = load_json(resolved_room)
    request = load_json(resolved_request)
    errors = [f"room: {error}" for error in validate_room_manifest(room)]
    errors.extend(
        f"request: {error}"
        for error in validate_capture_request(request, room_id=room.get("room_id"))
    )
    if errors:
        raise ContractError(errors)
    return ValidatedM1Inputs(
        room_path=resolved_room,
        request_path=resolved_request,
        room=room,
        request=request,
    )


def aggregate_status(checks: list[dict[str, Any]]) -> str:
    required = [check for check in checks if check.get("required", True)]
    if not required:
        raise ValueError("At least one required check is needed to aggregate status")
    statuses = [check.get("status") for check in required]
    if any(status not in STATUS_VALUES for status in statuses):
        raise ValueError("Every required check must use the AVEngine status vocabulary")
    if "fail" in statuses:
        return "fail"
    if "blocked" in statuses:
        return "blocked"
    if "not_run" in statuses:
        return "not_run"
    return "pass"

"""Import one catalog-described controlled human with generic Unreal APIs.

Run this file with UnrealEditor's PythonScriptPlugin. The catalog and source
assets are AVEngine runtime inputs; the importer has no dependency on a
SPEAR/Python extension and does not encode a colour or tag allow-list.


Adapted from the retained Eastforward/SPEAR asset tooling at commit
7b4d2cd3 (2026-09-04). See docs/provenance/UPSTREAM_ADAPTATIONS.md;
the retained SPEAR MIT notice is in LICENSES/SPEAR-MIT.txt.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:  # Unreal is present only inside the editor process.
    import unreal  # type: ignore
except ImportError:  # pragma: no cover - exercised by ordinary Python tests
    unreal = None  # type: ignore[assignment]

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
_GLTF_JSON_CHUNK = 0x4E4F534A
_WEBP_EXTENSION = "EXT_texture_webp"
_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _require_unreal() -> Any:
    require(unreal is not None, "this importer must run inside UnrealEditor")
    return unreal


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _repository_root() -> Path:
    value = _env("AVENGINE_REPOSITORY_ROOT")
    root = Path(value).expanduser() if value else SCRIPT_ROOT
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def _path(value: str | Path, *, base: Path | None = None) -> Path:
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = (base or Path.cwd()) / result
    return result.resolve()


def _direct_file(value: str | Path, description: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        require(
            not cursor.is_symlink(),
            f"{description} contains a symbolic link: {cursor}",
        )
    result = raw.resolve()
    require(
        result.is_file(),
        f"{description} is not a direct regular file: {result}",
    )
    return result


def _load_json(value: str | Path, description: str) -> dict[str, Any]:
    path = _direct_file(value, description)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {description}: {path}") from error
    require(isinstance(result, dict), f"{description} must contain one object")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _selected_configuration() -> dict[str, Any]:
    root = _repository_root()
    require(
        _TAG_PATTERN.fullmatch(_env("AVENGINE_CONTROLLED_HUMAN_TAG") or "")
        is not None,
        "AVENGINE_CONTROLLED_HUMAN_TAG must be a non-empty safe identifier",
    )
    tag = _env("AVENGINE_CONTROLLED_HUMAN_TAG")
    assert tag is not None

    source_root = str(root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    import avengine
    avengine_file = Path(avengine.__file__).resolve()
    require(
        Path(source_root).resolve() in avengine_file.parents,
        f"AVEngine import escaped the selected repository root: {avengine_file}",
    )
    from avengine.assets import controlled_humans

    catalog_value = _env("AVENGINE_CONTROLLED_HUMAN_CATALOG")
    contracts = controlled_humans.load_importer_contracts(
        root,
        catalog_value,
        validate_artifacts=False,
    )
    try:
        contract = contracts[tag]
    except KeyError as error:
        raise RuntimeError(f"controlled human tag is not described: {tag!r}") from error

    source_glb_value = _env("AVENGINE_CONTROLLED_HUMAN_SOURCE_GLB")
    source_manifest_value = _env("AVENGINE_CONTROLLED_HUMAN_SOURCE_MANIFEST")
    require(
        source_glb_value or contract.get("runtime_glb_path"),
        "source GLB is not configured by the catalog or environment",
    )
    require(
        source_manifest_value or contract.get("source_manifest_path"),
        "source manifest is not configured by the catalog or environment",
    )
    source_glb = _direct_file(
        source_glb_value or str(contract["runtime_glb_path"]),
        "controlled human runtime GLB",
    )
    source_manifest = _direct_file(
        source_manifest_value or str(contract["source_manifest_path"]),
        "controlled human source manifest",
    )
    if contract.get("artifact_records"):
        catalog = controlled_humans.load_catalog(root, catalog_value)
        controlled_humans.validate_artifacts(
            contract["raw_entry"],
            catalog,
            catalog_path=contract["catalog_path"],
            repo_root=root,
        )
        described_glb = contract.get("runtime_glb_path")
        described_manifest = contract.get("source_manifest_path")
        if described_glb is not None:
            require(
                source_glb == Path(described_glb),
                "source GLB differs from the catalog-described artifact",
            )
        if described_manifest is not None:
            require(
                source_manifest == Path(described_manifest),
                "source manifest differs from the catalog-described artifact",
            )

    ue_manifest_value = _env("AVENGINE_CONTROLLED_HUMAN_UE_MANIFEST")
    if ue_manifest_value:
        ue_manifest = _path(ue_manifest_value)
    else:
        relative = Path(str(contract["ue_manifest_relative_path"]))
        ue_manifest = _path(relative, base=root) if not relative.is_absolute() else relative
    mesh_dir = _env("AVENGINE_CONTROLLED_HUMAN_MESH_DIR")
    blueprint_dir = _env("AVENGINE_CONTROLLED_HUMAN_BLUEPRINT_DIR")
    mesh_dir = mesh_dir or f"/Game/AVEngine/ControlledHumans/{tag}/Meshes"
    blueprint_dir = blueprint_dir or f"/Game/AVEngine/ControlledHumans/{tag}/Blueprints"
    for name, value in (("mesh directory", mesh_dir), ("Blueprint directory", blueprint_dir)):
        require(value.startswith("/Game/"), f"{name} must be an Unreal /Game path")
    blueprint_name = _env("AVENGINE_CONTROLLED_HUMAN_BLUEPRINT_NAME")
    blueprint_name = blueprint_name or f"BP_controlled_{tag}"
    require(
        _TAG_PATTERN.fullmatch(blueprint_name) is not None,
        "Blueprint name must be a safe Unreal identifier",
    )
    return {
        "root": root,
        "tag": tag,
        "contract": contract,
        "source_glb": source_glb,
        "source_manifest": source_manifest,
        "ue_manifest": ue_manifest,
        "mesh_dir": mesh_dir,
        "blueprint_dir": blueprint_dir,
        "blueprint_name": blueprint_name,
        "verify_only": _env("AVENGINE_CONTROLLED_HUMAN_VERIFY_ONLY") == "1",
    }


def _list(value: object, description: str) -> list[Any]:
    require(isinstance(value, list), f"{description} must be a list")
    return value


def _manifest_range(value: object, description: str) -> list[float]:
    values = _list(value, description)
    require(
        len(values) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in values),
        f"{description} must contain two numbers",
    )
    return [float(values[0]), float(values[1])]


def _validate_source_manifest(
    source_manifest_path: Path,
    source_glb: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _load_json(source_manifest_path, "controlled human source manifest")
    expected_schema = contract.get("source_manifest_schema")
    if expected_schema is not None:
        require(
            manifest.get("schema") == expected_schema,
            f"source manifest schema differs from catalog: {manifest.get('schema')!r}",
        )
    if manifest.get("tag") is not None:
        require(
            manifest["tag"] == contract["tag"],
            "source manifest tag differs from selected catalog entry",
        )
    if manifest.get("asset_id") is not None:
        require(
            manifest["asset_id"] == contract["asset_id"],
            "source manifest asset_id differs from selected catalog entry",
        )
    usage_scope = contract.get("usage_scope")
    if usage_scope is not None:
        require(
            manifest.get("usage_scope") == usage_scope,
            "source manifest usage_scope differs from catalog",
        )

    runtime = manifest.get("runtime_glb")
    if isinstance(runtime, Mapping):
        filename = runtime.get("filename")
        if filename is not None:
            require(filename == source_glb.name, "source manifest GLB filename differs")
        recorded_size = runtime.get("size_bytes")
        if recorded_size is not None:
            require(
                recorded_size == source_glb.stat().st_size,
                "source manifest GLB size differs from source bytes",
            )
        recorded_hash = runtime.get("sha256")
        if recorded_hash is not None:
            require(
                recorded_hash == _sha256(source_glb),
                "source manifest GLB description is stale",
            )

    normalization = manifest.get("normalization")
    expected_normalization_schema = contract.get("normalization_schema")
    if isinstance(normalization, Mapping):
        if expected_normalization_schema is not None:
            require(
                normalization.get("schema") == expected_normalization_schema,
                "source normalization schema differs from catalog",
            )
        normalized_count = normalization.get("normalized_joint_count")
        if normalized_count is not None:
            require(
                normalized_count == contract["expected_bone_count"],
                "source normalization joint count differs from catalog",
            )
        if normalization.get("static_wrapper_translation_zeroed") is not None:
            require(
                normalization["static_wrapper_translation_zeroed"] is True,
                "source normalization wrapper is not grounded",
            )
        if contract.get("requires_in_place_actions"):
            in_place = normalization.get("in_place_actions")
            require(
                isinstance(in_place, list) and in_place,
                "source normalization does not declare in-place actions",
            )
            root_motion = normalization.get("root_motion", {})
            require(isinstance(root_motion, Mapping), "source normalization lacks root motion")
            for action_name in in_place:
                require(
                    isinstance(action_name, str) and action_name,
                    "source in-place action name is invalid",
                )
                action_motion = root_motion.get(action_name, {})
                require(
                    isinstance(action_motion, Mapping),
                    f"source normalization lacks root motion for {action_name}",
                )
                horizontal = action_motion.get("maximum_horizontal_deviation_after_m")
                vertical = action_motion.get("maximum_vertical_world_error_m")
                if horizontal is not None:
                    require(
                        float(horizontal) < 1.0e-6,
                        f"{action_name} retains horizontal root motion",
                    )
                if vertical is not None:
                    require(
                        float(vertical) < 1.0e-6,
                        f"{action_name} is not vertically grounded",
                    )

    expected_qa = manifest.get("expected_ue_qa")
    if isinstance(expected_qa, Mapping):
        for name in ("height_range_cm", "bottom_range_cm"):
            expected = contract.get(name)
            if expected is not None and expected_qa.get(name) is not None:
                require(
                    _manifest_range(expected_qa[name], f"source expected_ue_qa.{name}")
                    == [float(item) for item in expected],
                    f"source expected_ue_qa.{name} differs from catalog",
                )
        if expected_qa.get("actor_scale") is not None:
            require(
                float(expected_qa["actor_scale"]) == float(contract["actor_scale"]),
                "source actor scale differs from catalog",
            )
    if manifest.get("automatic_checks", {}).get("overall") is not None:
        require(
            manifest["automatic_checks"]["overall"] == "passed",
            "source automatic checks did not pass",
        )
    return manifest


def _read_glb_contract(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = path.read_bytes()
    require(len(payload) >= 20 and payload[:4] == b"glTF", "source is not a GLB 2.0 file")
    version, declared_length = struct.unpack_from("<II", payload, 4)
    json_length, chunk_type = struct.unpack_from("<II", payload, 12)
    require(
        version == 2
        and declared_length == len(payload)
        and chunk_type == _GLTF_JSON_CHUNK
        and 20 + json_length <= len(payload),
        "GLB header is invalid",
    )
    try:
        document = json.loads(payload[20 : 20 + json_length].rstrip(b" \x00").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("GLB JSON chunk is invalid") from error
    require(isinstance(document, dict), "GLB JSON chunk must be an object")

    meshes = _list(document.get("meshes"), "GLB meshes")
    require(len(meshes) == 1 and isinstance(meshes[0], Mapping), "GLB must contain one mesh")
    primitives = _list(meshes[0].get("primitives"), "GLB mesh primitives")
    expected_primitive_count = contract.get("expected_primitive_count")
    if expected_primitive_count is not None:
        require(
            len(primitives) == int(expected_primitive_count),
            "GLB primitive count differs from catalog",
        )
    require(primitives, "GLB mesh has no primitives")
    required_attributes = set(contract["required_primitive_attributes"])
    for primitive in primitives:
        require(isinstance(primitive, Mapping), "GLB primitive is not an object")
        attributes = primitive.get("attributes")
        require(
            isinstance(attributes, Mapping)
            and required_attributes.issubset(attributes),
            "GLB primitive lacks required geometry or skin attributes",
        )

    skins = _list(document.get("skins"), "GLB skins")
    require(len(skins) == 1 and isinstance(skins[0], Mapping), "GLB must contain one skin")
    joints = _list(skins[0].get("joints"), "GLB skin joints")
    expected_bones = contract.get("expected_bone_count")
    if expected_bones is not None:
        require(len(joints) == int(expected_bones), "GLB skin joint count differs from catalog")

    animations = _list(document.get("animations"), "GLB animations")
    animation_names = [item.get("name") for item in animations if isinstance(item, Mapping)]
    require(
        len(animation_names) == len(animations) and all(isinstance(name, str) for name in animation_names),
        "GLB animation names are invalid",
    )
    expected_animations = set(contract.get("required_animation_names") or ())
    if expected_animations:
        require(
            set(animation_names) == expected_animations
            and len(animation_names) == len(expected_animations),
            f"GLB animations differ: {sorted(animation_names)}",
        )

    materials = _list(document.get("materials"), "GLB materials")
    material_names = [item.get("name") for item in materials if isinstance(item, Mapping)]
    require(
        len(material_names) == len(materials) and all(isinstance(name, str) for name in material_names),
        "GLB material names are invalid",
    )
    expected_materials = set(contract.get("expected_material_names") or ())
    if expected_materials:
        require(
            set(material_names) == expected_materials
            and len(material_names) == len(expected_materials),
            f"GLB materials differ: {sorted(material_names)}",
        )

    images = _list(document.get("images"), "GLB images")
    image_names = [item.get("name") for item in images if isinstance(item, Mapping)]
    require(
        len(image_names) == len(images) and all(isinstance(name, str) for name in image_names),
        "GLB image names are invalid",
    )
    expected_images = set(contract.get("expected_image_names") or ())
    if expected_images:
        require(
            set(image_names) == expected_images
            and len(image_names) == len(expected_images),
            f"GLB images differ: {sorted(image_names)}",
        )

    textures = _list(document.get("textures"), "GLB textures")
    expected_texture_count = contract.get("expected_texture_count")
    if expected_texture_count is not None:
        require(
            len(textures) == int(expected_texture_count),
            "GLB texture count differs from catalog",
        )
    require(_WEBP_EXTENSION not in document.get("extensionsRequired", []), "GLB requires WebP")
    buffers = _list(document.get("buffers"), "GLB buffers")
    require(len(buffers) == 1 and "uri" not in buffers[0], "GLB must use one embedded buffer")
    for image in images:
        extensions = image.get("extensions", {}) if isinstance(image, Mapping) else {}
        view = image.get("bufferView") if isinstance(image, Mapping) else None
        require(
            isinstance(image, Mapping)
            and image.get("mimeType") == "image/png"
            and isinstance(view, int)
            and "uri" not in image
            and isinstance(extensions, Mapping)
            and _WEBP_EXTENSION not in extensions,
            "GLB images must be embedded core PNG images",
        )

    nodes = _list(document.get("nodes"), "GLB nodes")
    scenes = _list(document.get("scenes"), "GLB scenes")
    scene_index = document.get("scene", 0)
    require(
        isinstance(scene_index, int) and 0 <= scene_index < len(scenes)
        and isinstance(scenes[scene_index], Mapping),
        "GLB scene index is invalid",
    )
    roots = scenes[scene_index].get("nodes", [])
    mesh_nodes = [
        index
        for index, node in enumerate(nodes)
        if isinstance(node, Mapping) and node.get("mesh") == 0 and node.get("skin") == 0
    ]
    require(len(mesh_nodes) == 1 and mesh_nodes[0] in roots, "GLB skinned mesh is not a scene root")
    family = contract.get("expected_skeleton_family")
    if family:
        armatures = [
            index
            for index, node in enumerate(nodes)
            if isinstance(node, Mapping) and node.get("name") == family
        ]
        require(len(armatures) == 1 and armatures[0] in roots, "GLB skeleton wrapper is not a scene root")
        armature = nodes[armatures[0]]
        require(
            armature.get("scale") == [1.0, 1.0, 1.0]
            and armature.get("translation") == [0.0, 0.0, 0.0],
            "GLB skeleton wrapper is not normalized",
        )
    return {
        "mesh_count": len(meshes),
        "primitive_count": len(primitives),
        "skin_count": len(skins),
        "joint_count": len(joints),
        "animation_names": sorted(animation_names),
        "material_names": sorted(material_names),
        "image_names": sorted(image_names),
        "texture_count": len(textures),
    }


def _asset_record(path: str) -> dict[str, str]:
    ue = _require_unreal()
    data = ue.EditorAssetLibrary.find_asset_data(asset_path=path)
    require(data is not None, f"could not read imported asset data: {path}")
    class_path = data.get_editor_property("asset_class_path")
    class_name = str(class_path.get_editor_property("asset_name"))
    package_path = str(data.get_editor_property("package_path"))
    asset_name = str(data.get_editor_property("asset_name"))
    return {
        "asset_name": asset_name,
        "class_name": class_name,
        "object_path": posixpath.join(package_path, f"{asset_name}.{asset_name}"),
    }


def _collect_imported_assets(mesh_dir: str) -> dict[str, Any]:
    ue = _require_unreal()
    result: dict[str, Any] = {
        "skeletal_mesh": [],
        "skeleton": [],
        "animations": {},
        "materials": [],
        "textures": [],
        "other": [],
    }
    paths = ue.EditorAssetLibrary.list_assets(
        directory_path=mesh_dir,
        recursive=True,
        include_folder=False,
    )
    for path in paths:
        record = _asset_record(str(path))
        class_name = record["class_name"]
        if class_name == "SkeletalMesh":
            result["skeletal_mesh"].append(record)
        elif class_name == "Skeleton":
            result["skeleton"].append(record)
        elif class_name == "AnimSequence":
            name = record["asset_name"]
            require(name not in result["animations"], f"duplicate animation asset: {name}")
            result["animations"][name] = record
        elif class_name in {"Material", "MaterialInstanceConstant"}:
            result["materials"].append(record)
        elif class_name == "Texture2D":
            result["textures"].append(record)
        else:
            result["other"].append(record)
    return result


def _load_asset(record: Mapping[str, Any]) -> Any:
    ue = _require_unreal()
    asset = ue.load_asset(name=record["object_path"])
    require(asset is not None, f"could not load UE asset: {record['object_path']}")
    return asset


def _component_from_blueprint(blueprint: Any) -> Any:
    ue = _require_unreal()
    generated_class = blueprint.generated_class()
    require(generated_class is not None, "Blueprint has no generated class")
    default_object = ue.get_default_object(generated_class)
    require(default_object is not None, "Blueprint generated class has no default object")
    for name in ("skeletal_mesh_component", "SkeletalMeshComponent"):
        try:
            component = default_object.get_editor_property(name)
        except Exception:
            continue
        if component is not None:
            return component
    # Some UE versions expose the component only through the SCS. This branch
    # still uses generic Unreal APIs and avoids a project-specific extension.
    try:
        scs = blueprint.get_editor_property("simple_construction_script")
        root_nodes = scs.get_editor_property("root_nodes")
        for node in root_nodes:
            component = node.get_editor_property("component_template")
            if isinstance(component, ue.SkeletalMeshComponent):
                return component
    except Exception:
        pass
    raise RuntimeError("Blueprint has no SkeletalMeshComponent")


def _set_editor_property(object_: Any, name: str, value: Any, *, required: bool = True) -> bool:
    try:
        object_.set_editor_property(name=name, value=value)
    except Exception as error:
        if required:
            raise RuntimeError(f"could not set Unreal property {name}: {error}") from error
        return False
    return True


def _configure_component(component: Any, mesh: Any, walking: Any) -> None:
    ue = _require_unreal()
    if hasattr(component, "set_animation_mode"):
        component.set_animation_mode(
            animation_mode=ue.AnimationMode.ANIMATION_SINGLE_NODE
        )
    if hasattr(component, "set_skeletal_mesh_asset"):
        component.set_skeletal_mesh_asset(new_mesh=mesh)
    else:
        _set_editor_property(component, "skeletal_mesh", mesh)
    play_data = ue.SingleAnimationPlayData(
        anim_to_play=walking,
        saved_position=0.0,
        saved_play_rate=1.0,
    )
    _set_editor_property(component, "animation_data", play_data)
    tick_option = getattr(
        ue.VisibilityBasedAnimTickOption,
        "ALWAYS_TICK_POSE_AND_REFRESH_BONES",
        None,
    )
    if tick_option is not None:
        _set_editor_property(
            component,
            "visibility_based_anim_tick_option",
            tick_option,
            required=False,
        )


def _create_blueprint(
    assets: Mapping[str, Any],
    *,
    blueprint_dir: str,
    blueprint_name: str,
    preferred_animation_name: str,
) -> tuple[str, Any]:
    ue = _require_unreal()
    blueprint_path = posixpath.join(blueprint_dir, f"{blueprint_name}.{blueprint_name}")
    require(
        not ue.EditorAssetLibrary.does_asset_exist(blueprint_path),
        f"refusing to replace existing Blueprint: {blueprint_path}",
    )
    factory = ue.BlueprintFactory()
    _set_editor_property(factory, "parent_class", ue.SkeletalMeshActor)
    asset_tools = ue.AssetToolsHelpers.get_asset_tools()
    blueprint = asset_tools.create_asset(
        asset_name=blueprint_name,
        package_path=blueprint_dir,
        asset_class=ue.Blueprint,
        factory=factory,
    )
    require(blueprint is not None, "could not create controlled-human Blueprint")
    component = _component_from_blueprint(blueprint)
    animation_names = assets["animations"]
    preferred_animation = preferred_animation_name
    require(
        preferred_animation in animation_names,
        f"catalog preview animation was not imported: {preferred_animation}",
    )
    _configure_component(
        component,
        _load_asset(assets["skeletal_mesh"][0]),
        _load_asset(animation_names[preferred_animation]),
    )
    subsystem = ue.get_editor_subsystem(ue.EditorAssetSubsystem)
    subsystem.save_loaded_asset(asset_to_save=blueprint)
    return blueprint_path, component


def _validate_runtime_assets(
    assets: Mapping[str, Any],
    component: Any,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    ue = _require_unreal()
    expected_animations = set(contract["required_animation_names"])
    expected_materials = set(contract["expected_material_names"])
    expected_bone_count = int(contract["expected_bone_count"])
    expected_texture_count = int(contract["expected_texture_count"])
    require(len(assets["skeletal_mesh"]) == 1, "UE import must create one SkeletalMesh")
    require(len(assets["skeleton"]) == 1, "UE import must create one Skeleton")
    require(
        set(assets["animations"]) == expected_animations,
        f"UE animations differ: {sorted(assets['animations'])}",
    )
    require(
        len(assets["materials"]) == len(expected_materials),
        "UE import material count differs from catalog",
    )
    require(
        len(assets["textures"]) == expected_texture_count,
        "UE import texture count differs from catalog",
    )
    mesh = _load_asset(assets["skeletal_mesh"][0])
    skeleton = _load_asset(assets["skeleton"][0])
    mesh_skeleton = mesh.get_editor_property("skeleton")
    require(
        mesh_skeleton is not None
        and mesh_skeleton.get_path_name() == skeleton.get_path_name(),
        "SkeletalMesh references the wrong Skeleton",
    )
    for name, record in assets["animations"].items():
        animation_skeleton = _load_asset(record).get_editor_property("skeleton")
        require(
            animation_skeleton is not None
            and animation_skeleton.get_path_name() == skeleton.get_path_name(),
            f"animation references the wrong Skeleton: {name}",
        )
    require(
        int(component.get_num_bones()) == expected_bone_count,
        f"UE imported {component.get_num_bones()} bones, expected {expected_bone_count}",
    )
    slots = mesh.get_editor_property("materials")
    require(len(slots) == len(expected_materials), "SkeletalMesh material slot count differs")
    slot_names = {str(slot.material_slot_name) for slot in slots}
    require(slot_names == expected_materials, f"SkeletalMesh material slots differ: {slot_names}")
    require(
        all(slot.material_interface is not None for slot in slots),
        "SkeletalMesh contains a null material slot",
    )

    bounds = mesh.get_imported_bounds()
    height_cm = 2.0 * float(bounds.box_extent.z)
    bottom_cm = float(bounds.origin.z - bounds.box_extent.z)
    top_cm = float(bounds.origin.z + bounds.box_extent.z)
    height_range = contract.get("height_range_cm")
    if height_range is not None:
        height_range = [float(value) for value in height_range]
        require(
            height_range[0] <= height_cm <= height_range[1],
            f"SkeletalMesh height {height_cm} cm is outside {height_range}",
        )
    bottom_range = contract.get("bottom_range_cm")
    if bottom_range is not None:
        bottom_range = [float(value) for value in bottom_range]
        require(
            bottom_range[0] <= bottom_cm <= bottom_range[1],
            f"SkeletalMesh bottom {bottom_cm} cm is outside {bottom_range}",
        )
    authored = contract.get("authored_height_cm")
    delta = abs(height_cm - float(authored)) if authored is not None else None
    tolerance = max(3.0, float(authored) * 0.02) if authored is not None else None
    if delta is not None and tolerance is not None:
        require(delta <= tolerance, f"SkeletalMesh height changed from authored {authored} cm")
    return {
        "bone_count": int(component.get_num_bones()),
        "actor_scale": float(contract["actor_scale"]),
        "bounds": {
            "origin_cm": [
                float(bounds.origin.x),
                float(bounds.origin.y),
                float(bounds.origin.z),
            ],
            "box_extent_cm": [
                float(bounds.box_extent.x),
                float(bounds.box_extent.y),
                float(bounds.box_extent.z),
            ],
            "height_cm": height_cm,
            "bottom_cm": bottom_cm,
            "top_cm": top_cm,
            "height_range_cm": height_range,
            "bottom_range_cm": bottom_range,
        },
        "material_slots": [
            {
                "slot_name": str(slot.material_slot_name),
                "material_path": slot.material_interface.get_path_name(),
            }
            for slot in slots
        ],
        "animation_names": sorted(assets["animations"]),
    }


def _content_record(
    assets: Mapping[str, Any],
    blueprint_path: str,
    *,
    mesh_dir: str,
    blueprint_dir: str,
) -> dict[str, Any]:
    return {
        "mesh_directory": mesh_dir,
        "blueprint_directory": blueprint_dir,
        "skeletal_mesh": assets["skeletal_mesh"][0]["object_path"],
        "skeleton": assets["skeleton"][0]["object_path"],
        "animations": {
            name: assets["animations"][name]["object_path"]
            for name in sorted(assets["animations"])
        },
        "materials": sorted(record["object_path"] for record in assets["materials"]),
        "textures": sorted(record["object_path"] for record in assets["textures"]),
        "blueprint": blueprint_path,
    }


def _write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        require(not path.exists() and not path.is_symlink(), f"refusing to replace UE manifest: {path}")
        os.link(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _build_manifest(
    configuration: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    glb_contract: Mapping[str, Any],
    assets: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    blueprint_path: str,
) -> dict[str, Any]:
    contract = configuration["contract"]
    source_glb = configuration["source_glb"]
    source_manifest_path = configuration["source_manifest"]
    return {
        "schema": str(contract["ue_manifest_schema"] or "avengine_controlled_human_ue_import_v1"),
        "generated_at": _utc_now(),
        "tag": configuration["tag"],
        "asset_id": contract["asset_id"],
        "usage_scope": contract.get("usage_scope"),
        "formal_registration_authorized": False,
        "catalog": {
            "path": contract["catalog_path"],
            "schema": contract["catalog_schema"],
            "top_color": contract.get("top_color"),
            "rgb": contract.get("rgb"),
            "source_tag": contract.get("source_tag"),
            "variant_id": contract.get("variant_id"),
            "producer_root": contract.get("producer_root"),
        },
        "source_glb": str(source_glb),
        "source_glb_sha256": _sha256(source_glb),
        "source_glb_size_bytes": source_glb.stat().st_size,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_manifest_schema": source_manifest.get("schema"),
        "glb_contract": dict(glb_contract),
        "content": _content_record(
            assets,
            blueprint_path,
            mesh_dir=configuration["mesh_dir"],
            blueprint_dir=configuration["blueprint_dir"],
        ),
        "runtime_contract": dict(runtime_contract),
        "reload_verification": {"status": "not_run"},
        "claim_boundary": (
            "generic Unreal editor import for a catalog-described research "
            "candidate; no formal dataset or registry admission"
        ),
    }


def _save_directory(directory: str) -> None:
    ue = _require_unreal()
    subsystem = ue.get_editor_subsystem(ue.EditorAssetSubsystem)
    try:
        subsystem.save_directory(
            directory_path=directory,
            only_if_is_dirty=False,
            recursive=True,
        )
    except TypeError:
        subsystem.save_directory(directory, only_if_is_dirty=False, recursive=True)


def _wait_for_assets() -> None:
    ue = _require_unreal()
    try:
        ue.AssetRegistryHelpers.get_asset_registry().wait_for_completion()
    except AttributeError:
        pass


def _verify_existing(configuration: Mapping[str, Any]) -> None:
    ue = _require_unreal()
    manifest = _load_json(configuration["ue_manifest"], "controlled human UE manifest")
    contract = configuration["contract"]
    require(manifest.get("schema") == contract["ue_manifest_schema"], "UE manifest schema differs")
    require(manifest.get("tag") == configuration["tag"], "UE manifest tag differs")
    require(manifest.get("asset_id") == contract["asset_id"], "UE manifest asset differs")
    require(manifest.get("formal_registration_authorized") is False, "UE manifest authorizes registration")
    require(
        _path(manifest.get("source_glb", "")) == configuration["source_glb"],
        "UE manifest source GLB path differs",
    )
    require(
        _path(manifest.get("source_manifest", "")) == configuration["source_manifest"],
        "UE manifest source manifest path differs",
    )
    recorded_glb_hash = manifest.get("source_glb_sha256")
    if recorded_glb_hash is not None:
        require(recorded_glb_hash == _sha256(configuration["source_glb"]), "source GLB changed after import")
    recorded_manifest_hash = manifest.get("source_manifest_sha256")
    if recorded_manifest_hash is not None:
        require(
            recorded_manifest_hash == _sha256(configuration["source_manifest"]),
            "source manifest changed after import",
        )
    _validate_source_manifest(
        configuration["source_manifest"],
        configuration["source_glb"],
        contract,
    )
    glb_contract = _read_glb_contract(configuration["source_glb"], contract)
    require(manifest.get("glb_contract") == glb_contract, "GLB contract changed after import")
    content = manifest.get("content")
    require(isinstance(content, Mapping), "UE manifest has no content record")
    require(
        content.get("mesh_directory") == configuration["mesh_dir"]
        and content.get("blueprint_directory") == configuration["blueprint_dir"],
        "UE content escaped its isolated directories",
    )
    assets = _collect_imported_assets(configuration["mesh_dir"])
    blueprint = ue.load_asset(name=content.get("blueprint"))
    require(blueprint is not None, "could not reload controlled-human Blueprint")
    component = _component_from_blueprint(blueprint)
    runtime_contract = _validate_runtime_assets(assets, component, contract)
    require(
        manifest.get("runtime_contract") == runtime_contract,
        "UE runtime readback differs from import manifest",
    )
    _log(f"CONTROLLED_HUMAN_IMPORT_VERIFY_OK tag={configuration['tag']} manifest={configuration['ue_manifest']}")


def _log(message: str) -> None:
    if unreal is not None:
        logger = getattr(unreal, "log", None)
        if logger is not None:
            logger(message)
            return
    print(message)


def main() -> None:
    configuration = _selected_configuration()
    ue = _require_unreal()
    if configuration["verify_only"]:
        _verify_existing(configuration)
        return

    ue_manifest = configuration["ue_manifest"]
    require(
        not ue_manifest.exists() and not ue_manifest.is_symlink(),
        f"refusing to replace existing UE manifest: {ue_manifest}",
    )
    for directory in (configuration["mesh_dir"], configuration["blueprint_dir"]):
        require(
            not ue.EditorAssetLibrary.does_directory_exist(directory_path=directory),
            f"refusing to replace existing Unreal directory: {directory}",
        )

    _validate_source_manifest(
        configuration["source_manifest"],
        configuration["source_glb"],
        configuration["contract"],
    )
    glb_contract = _read_glb_contract(configuration["source_glb"], configuration["contract"])
    try:
        for directory in (configuration["mesh_dir"], configuration["blueprint_dir"]):
            require(
                ue.EditorAssetLibrary.make_directory(directory_path=directory),
                f"could not create Unreal directory: {directory}",
            )

        task = ue.AssetImportTask()
        _set_editor_property(task, "automated", True)
        _set_editor_property(task, "destination_path", configuration["mesh_dir"])
        _set_editor_property(task, "filename", str(configuration["source_glb"]))
        _set_editor_property(task, "replace_existing", False)
        _set_editor_property(task, "replace_existing_settings", False, required=False)
        _set_editor_property(task, "save", False)
        ue.AssetToolsHelpers.get_asset_tools().import_asset_tasks(import_tasks=[task])
        _wait_for_assets()
        _save_directory(configuration["mesh_dir"])
        assets = _collect_imported_assets(configuration["mesh_dir"])
        blueprint_path, component = _create_blueprint(
            assets,
            blueprint_dir=configuration["blueprint_dir"],
            blueprint_name=configuration["blueprint_name"],
            preferred_animation_name=configuration["contract"]["preview_animation_name"],
        )
        runtime_contract = _validate_runtime_assets(
            assets,
            component,
            configuration["contract"],
        )
        manifest = _build_manifest(
            configuration,
            _load_json(configuration["source_manifest"], "controlled human source manifest"),
            glb_contract,
            assets,
            runtime_contract,
            blueprint_path,
        )
        _write_json_no_replace(ue_manifest, manifest)
    except BaseException as error:
        _log(
            f"CONTROLLED_HUMAN_IMPORT_FAILED tag={configuration['tag']} "
            f"partial_mesh_directory={configuration['mesh_dir']} "
            f"partial_blueprint_directory={configuration['blueprint_dir']} "
            f"manifest={ue_manifest} error={error}"
        )
        raise

    _log(
        f"CONTROLLED_HUMAN_IMPORT_OK tag={configuration['tag']} "
        f"blueprint={blueprint_path} manifest={ue_manifest}"
    )


if __name__ == "__main__":  # pragma: no cover - Unreal invokes this entrypoint
    main()

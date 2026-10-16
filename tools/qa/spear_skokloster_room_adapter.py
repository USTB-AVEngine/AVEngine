#!/usr/bin/env python3
"""Generic imported-GLB runtime contract specialized to Skokloster's one mesh."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "avengine_spear_imported_glb_room_adapter_v1"
IMPORT_SCHEMA = "avengine_skokloster_ue_import_result_v1"
ROOM_ID = "habitat_test_skokloster_castle"
SCENE_ID = "skokloster-castle"
CONTENT_PREFIX = "/Game/MyAssets/Audioset/Scenes/skokloster_castle/"
ENTRY_MAP = "/Engine/Maps/Entry"
CAMERA_BLUEPRINT = "/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C"
RGB_COMPONENT = "DefaultSceneRoot.final_tone_curve_hdr_"
DEPTH_COMPONENT = "DefaultSceneRoot.sp_depth_meters_"
OBJECT_ID_COMPONENT = "DefaultSceneRoot.sp_object_ids_uint8_"
EXPECTED_STATIC_MESH_COUNT = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def validate_import_result(value: Mapping[str, Any]) -> list[str]:
    _require(value.get("schema") == IMPORT_SCHEMA, "Skokloster import schema drift")
    _require(value.get("status") == "pass", "Skokloster editor verification failed")
    _require(
        value.get("mode") == "fresh_editor_verify_only"
        and value.get("reload_verification") == "pass",
        "Skokloster import lacks fresh-editor reload evidence",
    )
    _require(
        value.get("content_root") == "/Game/MyAssets/Audioset/Scenes/skokloster_castle",
        "Skokloster content root drift",
    )
    scene = value.get("scene_content")
    _require(isinstance(scene, Mapping), "Skokloster result lacks scene_content")
    paths = scene.get("static_meshes")
    _require(
        isinstance(paths, list)
        and scene.get("static_mesh_count") == EXPECTED_STATIC_MESH_COUNT
        and len(paths) == EXPECTED_STATIC_MESH_COUNT
        and len(set(paths)) == EXPECTED_STATIC_MESH_COUNT,
        "Skokloster must reload exactly one unique StaticMesh",
    )
    _require(
        all(
            isinstance(path, str)
            and path.startswith(CONTENT_PREFIX)
            and "." in path.rsplit("/", 1)[-1]
            for path in paths
        ),
        "Skokloster cooked mesh object path is outside the isolated root",
    )
    return list(paths)


def build_room_adapter_record(
    import_result: Mapping[str, Any], *, import_result_path: str
) -> dict[str, Any]:
    mesh_paths = validate_import_result(import_result)
    return {
        "schema": SCHEMA,
        "adapter_kind": "spear_imported_glb",
        "room_id": ROOM_ID,
        "scene_id": SCENE_ID,
        "entry_map": ENTRY_MAP,
        "ue_import_result": import_result_path,
        "expected_static_mesh_count": EXPECTED_STATIC_MESH_COUNT,
        "static_mesh_object_paths": mesh_paths,
        "coordinate_contract": {
            "raw_source": "legacy Habitat test-scene POSITION, Z up and +Y front",
            "source_to_habitat": "H=(S.x,S.z,-S.y)",
            "prepared_glb": "canonical glTF metres, +Y up, -Z forward",
            "habitat_to_ue_cm": "U_cm=(100*H.x,100*H.z,100*H.y)",
            "runtime_room_actor_transform": "identity",
        },
        "spawn_policy": {
            "actor_class": "AStaticMeshActor",
            "spawn_collision_handling": "AlwaysSpawn",
            "component_mobility": "Movable",
            "component_collision": "NoCollision",
            "cast_shadow": True,
            "fresh_cooked_load_and_component_readback_required": True,
        },
        "camera_contract": {
            "blueprint_class_path": CAMERA_BLUEPRINT,
            "one_camera_actor_for_all_passes": True,
            "components": {
                "normal_rgb": RGB_COMPONENT,
                "normal_metric_depth": DEPTH_COMPONENT,
                "normal_object_ids": OBJECT_ID_COMPONENT,
                "source1_target_only_metric_depth": DEPTH_COMPONENT,
                "source2_target_only_metric_depth": DEPTH_COMPONENT,
            },
            "pass_order": ["normal", "source1_target_only", "source2_target_only"],
            "target_only_policy": "PRM_UseShowOnlyList on the shared depth component",
        },
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "claim_boundary": (
            "room/camera contract only; fresh cooked load/readback and strict pixels "
            "remain pending until packaged SPEAR execution"
        ),
    }


def validate_room_adapter(value: Mapping[str, Any]) -> None:
    _require(value.get("schema") == SCHEMA, "room adapter schema drift")
    _require(value.get("room_id") == ROOM_ID, "room identity drift")
    _require(value.get("entry_map") == ENTRY_MAP, "imported room must use Entry map")
    paths = value.get("static_mesh_object_paths")
    _require(
        isinstance(paths, list) and len(paths) == 1 and len(set(paths)) == 1,
        "room adapter must declare exactly one unique mesh path",
    )
    coordinate = value.get("coordinate_contract")
    _require(
        isinstance(coordinate, Mapping)
        and coordinate.get("source_to_habitat") == "H=(S.x,S.z,-S.y)"
        and coordinate.get("runtime_room_actor_transform") == "identity",
        "room adapter coordinate chain drift",
    )
    camera = value.get("camera_contract")
    components = camera.get("components") if isinstance(camera, Mapping) else None
    _require(
        isinstance(camera, Mapping)
        and camera.get("blueprint_class_path") == CAMERA_BLUEPRINT
        and camera.get("one_camera_actor_for_all_passes") is True
        and isinstance(components, Mapping)
        and components.get("normal_metric_depth") == DEPTH_COMPONENT
        and components.get("source1_target_only_metric_depth") == DEPTH_COMPONENT
        and components.get("source2_target_only_metric_depth") == DEPTH_COMPONENT,
        "three capture passes must share one BP_CameraSensor depth component",
    )


def _set_collision_disabled(component: Any) -> None:
    try:
        component.SetCollisionEnabled(NewType="NoCollision")
    except (AttributeError, RuntimeError):
        component.set_property_value(
            property_name="CollisionEnabled", property_value="NoCollision"
        )


def _static_mesh_handle(component: Any) -> tuple[int, str]:
    try:
        value = component.GetStaticMesh(as_handle=True)
        method = "UStaticMeshComponent.GetStaticMesh"
    except (AttributeError, RuntimeError):
        value = component.get_property_value(property_name="StaticMesh", as_handle=True)
        method = "UStaticMeshComponent.StaticMesh_property"
    _require(not isinstance(value, bool) and int(value) > 0, "invalid mesh readback")
    return int(value), method


def spawn_scene_meshes_with_readback(
    game: Any, adapter: Mapping[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    validate_room_adapter(adapter)
    object_path = adapter["static_mesh_object_paths"][0]
    expected_handle = int(
        game.unreal_service.load_object(
            uclass="UStaticMesh", name=object_path, as_handle=True
        )
    )
    _require(expected_handle > 0, f"could not load cooked mesh: {object_path}")
    mesh = game.get_unreal_object(uobject=expected_handle)
    actor = game.unreal_service.spawn_actor(
        uclass="AStaticMeshActor",
        location={"X": 0.0, "Y": 0.0, "Z": 0.0},
        spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
    )
    _require(actor is not None, "could not spawn Skokloster room actor")
    component = game.unreal_service.get_component_by_class(
        actor=actor, uclass="UStaticMeshComponent"
    )
    component.SetMobility(NewMobility="Movable")
    component.SetStaticMesh(NewMesh=mesh)
    component.SetCastShadow(NewCastShadow=True)
    _set_collision_disabled(component)
    stable_name = "AVEngine/ImportedGLB/skokloster_castle/surface_000"
    game.unreal_service.set_stable_name_for_actor(actor=actor, stable_name=stable_name)
    observed_handle, method = _static_mesh_handle(component)
    _require(observed_handle == expected_handle, "live cooked mesh readback differs")
    return [actor], {
        "schema": "avengine_spear_imported_glb_live_readback_v1",
        "status": "pass",
        "scene_id": SCENE_ID,
        "entry_map": ENTRY_MAP,
        "expected_static_mesh_count": 1,
        "spawned_static_mesh_count": 1,
        "all_expected_handles_match_components": True,
        "unique_loaded_object_handle_count": 1,
        "unique_component_mesh_handle_count": 1,
        "meshes": [
            {
                "mesh_index": 0,
                "object_path": object_path,
                "stable_actor_name": stable_name,
                "expected_object_handle": expected_handle,
                "observed_component_mesh_handle": observed_handle,
                "readback_method": method,
                "status": "pass",
            }
        ],
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def destroy_scene_meshes(instance: Any, actors: Sequence[Any]) -> None:
    with instance.begin_frame():
        for actor in actors:
            actor.K2_DestroyActor()
    with instance.end_frame():
        pass


__all__ = [
    "CAMERA_BLUEPRINT",
    "DEPTH_COMPONENT",
    "ENTRY_MAP",
    "EXPECTED_STATIC_MESH_COUNT",
    "OBJECT_ID_COMPONENT",
    "RGB_COMPONENT",
    "SCHEMA",
    "build_room_adapter_record",
    "destroy_scene_meshes",
    "load_json_object",
    "spawn_scene_meshes_with_readback",
    "validate_import_result",
    "validate_room_adapter",
]

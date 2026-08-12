#!/usr/bin/env python3
"""Runtime adapter for reload-verified GLB scenes imported into cooked SPEAR.

The adapter deliberately owns only room materialization.  Episode actors and
the multimodal ``BP_CameraSensor`` remain owned by the existing native pixel
capture path.  This keeps the imported-room contract reusable without making
the visual backend a second geometry, trajectory, or audio authority.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from avengine.qa.spear_unreal_capabilities import read_handle_capability

SCHEMA = "avengine_spear_imported_glb_room_adapter_v1"
IMPORT_SCHEMA = "avengine_mp3d_ue_import_result_v1"
ENTRY_MAP = "/Engine/Maps/Entry"
CAMERA_BLUEPRINT = "/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C"
RGB_COMPONENT = "DefaultSceneRoot.final_tone_curve_hdr_"
DEPTH_COMPONENT = "DefaultSceneRoot.sp_depth_meters_"
OBJECT_ID_COMPONENT = "DefaultSceneRoot.sp_object_ids_uint8_"
EXPECTED_STATIC_MESH_COUNT = 71


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json_object(path: Path, *, owner: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{owner} JSON root must be an object: {path}")
    return value


def validate_mp3d_import_manifest(value: Mapping[str, Any]) -> list[str]:
    """Return the exact reload-verified cooked UStaticMesh object closure."""

    _require(value.get("schema") == IMPORT_SCHEMA, "MP3D import schema drift")
    _require(value.get("status") == "passed", "MP3D UE import did not pass")
    _require(value.get("scene_id") == "17DRP5sb8fy", "MP3D scene identity drift")
    reload_value = value.get("reload_verification")
    _require(
        isinstance(reload_value, Mapping)
        and reload_value.get("status") == "passed"
        and reload_value.get("process") == "second_ue_editor_process",
        "MP3D import lacks a fresh-process reload verification",
    )
    scene = value.get("scene_content")
    _require(isinstance(scene, Mapping), "MP3D import lacks scene_content")
    paths = scene.get("static_meshes")
    _require(isinstance(paths, list), "MP3D import lacks static_meshes")
    _require(
        len(paths) == EXPECTED_STATIC_MESH_COUNT
        and scene.get("static_mesh_count") == EXPECTED_STATIC_MESH_COUNT,
        "MP3D imported static-mesh count is not exactly 71",
    )
    _require(
        len(set(paths)) == EXPECTED_STATIC_MESH_COUNT
        and all(
            isinstance(path, str)
            and path.startswith("/Game/MyAssets/Audioset/Scenes/mp3d_17DRP5sb8fy/")
            and "." in path.rsplit("/", 1)[-1]
            for path in paths
        ),
        "MP3D static-mesh object paths are invalid or duplicated",
    )
    counts = scene.get("class_counts")
    _require(
        isinstance(counts, Mapping)
        and counts.get("StaticMesh") == EXPECTED_STATIC_MESH_COUNT,
        "MP3D class-count closure differs from 71 UStaticMesh objects",
    )
    return list(paths)


def build_room_adapter_record(
    import_manifest: Mapping[str, Any],
    *,
    execution_manifest_path: str,
) -> dict[str, Any]:
    mesh_paths = validate_mp3d_import_manifest(import_manifest)
    return {
        "schema": SCHEMA,
        "adapter_kind": "spear_imported_glb",
        "room_id": "habitat_mp3d_example_17DRP5sb8fy",
        "scene_id": "17DRP5sb8fy",
        "entry_map": ENTRY_MAP,
        "ue_import_manifest": execution_manifest_path,
        "expected_static_mesh_count": EXPECTED_STATIC_MESH_COUNT,
        "static_mesh_object_paths": mesh_paths,
        "coordinate_contract": {
            "source_axis_description": import_manifest["coordinate_contract"][
                "source_axis_description"
            ],
            "source_to_habitat": import_manifest["coordinate_contract"][
                "source_to_canonical"
            ],
            "habitat_to_ue_cm": "U_cm=(100*H.x,100*H.z,100*H.y)",
        },
        "spawn_policy": {
            "actor_class": "AStaticMeshActor",
            "spawn_collision_handling": "AlwaysSpawn",
            "component_mobility": "Movable",
            "component_collision": "NoCollision",
            "cast_shadow": True,
            "fresh_cooked_load_and_component_readback_required": True,
        },
        "review_lighting": {
            "directional_key": {
                "yaw_deg": -45.0,
                "pitch_deg": -50.0,
                "intensity_lux": 10.0,
            },
            "skylight_intensity": 0.35,
            "provenance": "retained passed MP3D Entry-map runtime evidence",
            "claim_boundary": "review lighting; not Matterport illumination truth",
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
            "runtime room/camera adapter only; fresh cooked load/readback and pixels "
            "remain pending until this record is executed in packaged SPEAR"
        ),
    }


def validate_room_adapter(value: Mapping[str, Any]) -> None:
    _require(value.get("schema") == SCHEMA, "room adapter schema drift")
    _require(value.get("adapter_kind") == "spear_imported_glb", "wrong adapter kind")
    _require(value.get("entry_map") == ENTRY_MAP, "imported GLB must use Entry map")
    paths = value.get("static_mesh_object_paths")
    _require(
        isinstance(paths, list)
        and len(paths) == EXPECTED_STATIC_MESH_COUNT
        and len(set(paths)) == EXPECTED_STATIC_MESH_COUNT,
        "room adapter must declare exactly 71 unique mesh paths",
    )
    camera = value.get("camera_contract")
    _require(isinstance(camera, Mapping), "room adapter lacks camera contract")
    components = camera.get("components")
    _require(
        camera.get("blueprint_class_path") == CAMERA_BLUEPRINT
        and camera.get("one_camera_actor_for_all_passes") is True
        and isinstance(components, Mapping)
        and components.get("normal_metric_depth") == DEPTH_COMPONENT
        and components.get("source1_target_only_metric_depth") == DEPTH_COMPONENT
        and components.get("source2_target_only_metric_depth") == DEPTH_COMPONENT,
        "normal and target-only passes do not share BP_CameraSensor metric depth",
    )
    coordinate = value.get("coordinate_contract")
    _require(
        isinstance(coordinate, Mapping)
        and coordinate.get("source_axis_description") == "Matterport raw GLB Z-up"
        and coordinate.get("source_to_habitat") == "H=(S.x,S.z,-S.y)"
        and coordinate.get("habitat_to_ue_cm") == "U_cm=(100*H.x,100*H.z,100*H.y)",
        "room adapter coordinate chain is not raw MP3D Z-up to Habitat to UE",
    )
    lighting = value.get("review_lighting")
    _require(
        isinstance(lighting, Mapping)
        and isinstance(lighting.get("directional_key"), Mapping)
        and lighting.get("skylight_intensity") == 0.35,
        "room adapter lacks the retained MP3D Entry-map review-lighting profile",
    )


def _set_collision_disabled(component: Any) -> None:
    try:
        component.SetCollisionEnabled(NewType="NoCollision")
    except (AttributeError, RuntimeError):
        component.set_property_value(
            property_name="CollisionEnabled", property_value="NoCollision"
        )


def _static_mesh_handle(component: Any) -> tuple[int, str]:
    evidence = read_handle_capability(
        component,
        owner="live UStaticMeshComponent.StaticMesh",
        getter_name="GetStaticMesh",
        property_name="StaticMesh",
        getter_kwargs={"as_handle": True},
        property_kwargs={"as_handle": True},
    )
    method = {
        "callable_getter": "UStaticMeshComponent.GetStaticMesh",
        "property_readback": "UStaticMeshComponent.StaticMesh_property",
    }[str(evidence["strategy"])]
    return int(evidence["handle"]), method


def spawn_scene_meshes_with_readback(
    game: Any, adapter: Mapping[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Fresh-load, spawn, and read back every declared cooked mesh."""

    validate_room_adapter(adapter)
    actors: list[Any] = []
    records: list[dict[str, Any]] = []
    for index, object_path in enumerate(adapter["static_mesh_object_paths"]):
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
        _require(actor is not None, f"could not spawn mesh actor {index}")
        component = game.unreal_service.get_component_by_class(
            actor=actor, uclass="UStaticMeshComponent"
        )
        component.SetMobility(NewMobility="Movable")
        component.SetStaticMesh(NewMesh=mesh)
        component.SetCastShadow(NewCastShadow=True)
        _set_collision_disabled(component)
        stable_name = f"AVEngine/ImportedGLB/17DRP5sb8fy/mesh_{index:03d}"
        game.unreal_service.set_stable_name_for_actor(
            actor=actor, stable_name=stable_name
        )
        observed_handle, method = _static_mesh_handle(component)
        _require(
            observed_handle == expected_handle,
            f"live cooked mesh readback differs at index {index}",
        )
        actors.append(actor)
        records.append(
            {
                "mesh_index": index,
                "object_path": object_path,
                "stable_actor_name": stable_name,
                "expected_object_handle": expected_handle,
                "observed_component_mesh_handle": observed_handle,
                "readback_method": method,
                "status": "pass",
            }
        )
    _require(
        len(actors) == EXPECTED_STATIC_MESH_COUNT,
        "spawned imported-room mesh closure is not 71",
    )
    expected_handles = [item["expected_object_handle"] for item in records]
    observed_handles = [item["observed_component_mesh_handle"] for item in records]
    _require(
        len(set(expected_handles)) == EXPECTED_STATIC_MESH_COUNT
        and len(set(observed_handles)) == EXPECTED_STATIC_MESH_COUNT,
        "71 declared cooked mesh paths did not resolve to 71 unique live objects",
    )
    return actors, {
        "schema": "avengine_spear_imported_glb_live_readback_v1",
        "status": "pass",
        "scene_id": adapter["scene_id"],
        "entry_map": adapter["entry_map"],
        "expected_static_mesh_count": EXPECTED_STATIC_MESH_COUNT,
        "spawned_static_mesh_count": len(actors),
        "all_expected_handles_match_components": True,
        "unique_loaded_object_handle_count": len(set(expected_handles)),
        "unique_component_mesh_handle_count": len(set(observed_handles)),
        "meshes": records,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def spawn_review_lighting(
    game: Any, spear_root: Path, profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Reuse the retained MP3D Entry-map sky/key-light implementation."""

    import sys

    sys.path.insert(0, str(spear_root / "examples"))
    from render_in_gpurir_room import spawn_directional_light, spawn_sky

    sky = spawn_sky(game=game)
    key = profile["directional_key"]
    light = spawn_directional_light(
        game=game,
        yaw_deg=key["yaw_deg"],
        pitch_deg=key["pitch_deg"],
        intensity_lux=key["intensity_lux"],
    )
    component = game.unreal_service.get_component_by_class(
        actor=light, uclass="UDirectionalLightComponent"
    )
    component.SetCastShadows(bNewValue=True)
    skylight = sky.get("ASkyLight")
    sky_readback = None
    if skylight is not None:
        sky_component = game.unreal_service.get_component_by_class(
            actor=skylight, uclass="USkyLightComponent"
        )
        sky_component.SetIntensity(NewIntensity=profile["skylight_intensity"])
        sky_readback = float(sky_component.get_property_value("Intensity"))
    return {
        "status": "pass",
        "directional_key": dict(key),
        "directional_intensity_readback_lux": float(
            component.get_property_value("Intensity")
        ),
        "skylight_intensity_readback": sky_readback,
        "claim_boundary": "review lighting only; not Matterport illumination truth",
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
    "spawn_review_lighting",
    "spawn_scene_meshes_with_readback",
    "validate_mp3d_import_manifest",
    "validate_room_adapter",
]

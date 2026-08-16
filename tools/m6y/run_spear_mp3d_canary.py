#!/usr/bin/env python3
"""Render the retained 270-frame MP3D route through packaged SPEAR.

The 18-second M5.1 route is older than, and incompatible with, the frozen
75-frame Timeline-v2 schema.  This runner therefore consumes an explicit
M5.1 compatibility plan; it never relabels the route as Timeline v2.  Habitat
PathFinder, captured actor roots, source programs, animated emitter links,
binaural audio and Topdown remain authoritative.  UE only renders the already
imported 71-mesh MP3D scene and the bound human/Beagle visuals.

``--spear-root`` supplies the external UE runtime, project, and retained assets.
The AVEngine-owned host/game client and launch settings stay in this repository;
the selected lighting and rig helpers are AVEngine-local.
All plans, frames and evidence are written under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.backends.spear_ue import client as spear_client
from avengine.backends.spear_ue.launch import parallel_instance_settings
from avengine.backends.spear_ue.lighting import spawn_directional_light, spawn_sky
from avengine.backends.spear_ue.rig_direction import select_skeletal_mesh_component
from avengine.optional_backends.spear_mp3d import (
    DOG_BP_CLASS_PATH,
    HUMAN_BP_CLASS_PATH,
    M5_1_EXECUTION_SCHEMA,
    M5_1_FPS,
    M5_1_FRAME_COUNT,
    MP3D_ROOM_ID,
    build_m5_1_mp3d_execution_plan,
    render_color_fidelity_qa,
)
from avengine.optional_backends.spear_apartment import (
    apply_ue_component_frame_delta,
    summarize_actor_bounds,
)


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_SPEAR_ROOT = REPOSITORY.parent / "AVEngine/external/SPEAR"
DEFAULT_ROUTE = REPOSITORY / "examples/m5_1/mp3d_articulated_review/route_manifest.json"
DEFAULT_CAPTURE = (
    REPOSITORY / "tmp/m5_1/mp3d_mixed_heading_lighting_20260718_01/evidence.json"
)
DEFAULT_FRAME_READBACK = (
    REPOSITORY / "tmp/m5_1/mp3d_mixed_heading_lighting_20260718_01/frame_readback.json"
)
DEFAULT_NAVMESH_GATE = (
    REPOSITORY
    / "tmp/m5_1/mp3d_mixed_heading_lighting_20260718_01/mp3d_gate_evidence.json"
)
DEFAULT_DELIVERY = REPOSITORY / "tmp/m5_1/mp3d_delivery_heading_lighting_20260718_02"
DEFAULT_SOURCE_PROGRAM = DEFAULT_DELIVERY / "source_program_reuse.json"
DEFAULT_EMITTERS = DEFAULT_DELIVERY / "actual_emitter_trajectories.json"
DEFAULT_HABITAT_REVIEW = (
    DEFAULT_DELIVERY / "videos/mp3d_human_beagle_annotated_binaural.mp4"
)
DEFAULT_ROOM_REGISTRY = REPOSITORY / "examples/m6/rooms/room_registry.json"
DEFAULT_ROOM_QUALIFICATION = (
    REPOSITORY / "examples/m6/rooms/qualification/mp3d_17DRP5sb8fy_raw.json"
)
DEFAULT_HABITAT_RGB = (
    REPOSITORY / "tmp/m5_1/mp3d_mixed_heading_lighting_20260718_01/arrays/rgb.npy"
)
CAMERA_BLUEPRINT = "/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C"
CAPTURE_COMPONENT_NAME = "DefaultSceneRoot.final_tone_curve_hdr_"
ENTRY_MAP = "/Engine/Maps/Entry"
EVIDENCE_SCHEMA = "avengine_optional_spear_mp3d_runtime_evidence_v1"
SMOKE_SCHEMA = "avengine_optional_spear_mp3d_runtime_smoke_v1"
INITIALIZE_CLIENT_MAX_TIME_SECONDS = 600.0
CLIENT_INTERNAL_TIMEOUT_SECONDS = 60.0
POSITION_TOLERANCE_CM = 0.02
ANGLE_TOLERANCE_DEG = 0.02
ANIMATION_TOLERANCE_SECONDS = 1.0e-4


def _load_json(path: Path, *, owner: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load {owner}: {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "byte_size": resolved.stat().st_size,
    }


def _default_old_paths(spear_root: Path) -> tuple[Path, Path]:
    import_result = (
        spear_root / "tmp/mp3d_ue_comparison_20260718_01/import/ue_import_result.json"
    )
    human_manifest = (
        spear_root / "tmp/rocketbox_native_ue_import_v3/"
        "rocketbox_male_adult_01_original_ue_v3/ue_import_manifest.json"
    )
    return import_result, human_manifest


def build_execution_plan(args: argparse.Namespace) -> dict[str, Any]:
    spear_root = args.spear_root.resolve()
    default_import, default_human = _default_old_paths(spear_root)
    import_path = (args.ue_import_manifest or default_import).resolve()
    human_path = (args.human_ue_manifest or default_human).resolve()
    frame_readback = _load_json(args.frame_readback.resolve(), owner="frame readback")
    if not isinstance(frame_readback, list):
        raise RuntimeError("frame readback JSON root must be a list")
    mappings = {
        "route_manifest": args.route_manifest,
        "capture_evidence": args.capture_evidence,
        "navmesh_gate": args.navmesh_gate,
        "source_program": args.source_program,
        "emitter_trajectories": args.emitter_trajectories,
        "room_registry": args.room_registry,
        "raw_room_qualification": args.room_qualification,
        "ue_import_manifest": import_path,
        "ue_material_color_result": args.ue_material_color_result,
        "human_ue_manifest": human_path,
    }
    loaded: dict[str, Mapping[str, Any]] = {}
    for name, path in mappings.items():
        value = _load_json(path.resolve(), owner=name)
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{name} JSON root must be an object")
        loaded[name] = value
    return build_m5_1_mp3d_execution_plan(
        route_manifest=loaded["route_manifest"],
        capture_evidence=loaded["capture_evidence"],
        frame_readback=frame_readback,
        navmesh_gate=loaded["navmesh_gate"],
        source_program=loaded["source_program"],
        emitter_trajectories=loaded["emitter_trajectories"],
        room_registry=loaded["room_registry"],
        raw_room_qualification=loaded["raw_room_qualification"],
        ue_import_manifest=loaded["ue_import_manifest"],
        ue_material_color_result=loaded["ue_material_color_result"],
        human_ue_manifest=loaded["human_ue_manifest"],
        output_gain=args.fixed_output_gain,
    )


def _struct_components(value: Any, names: Sequence[str]) -> list[float]:
    current = value
    expected = [name.casefold() for name in names]
    for _ in range(4):
        if not isinstance(current, Mapping):
            break
        lowered = {str(key).casefold(): item for key, item in current.items()}
        if all(name in lowered for name in expected):
            result = [float(lowered[name]) for name in expected]
            if all(math.isfinite(item) for item in result):
                return result
            break
        if "returnvalue" in lowered and isinstance(lowered["returnvalue"], Mapping):
            current = lowered["returnvalue"]
        elif len(current) == 1 and isinstance(next(iter(current.values())), Mapping):
            current = next(iter(current.values()))
        else:
            break
    raise RuntimeError(f"could not decode Unreal struct {names}: {value!r}")


def _actor_readback(actor: Any, frame_index: int) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "location_cm": _struct_components(
            actor.K2_GetActorLocation(as_dict=True), ("x", "y", "z")
        ),
        "rotation_deg": _struct_components(
            actor.K2_GetActorRotation(as_dict=True), ("roll", "pitch", "yaw")
        ),
    }


def _actor_bounds_readback(actor: Any, frame_index: int) -> dict[str, Any]:
    value = actor.GetActorBounds(
        bOnlyCollidingComponents=False,
        bIncludeFromChildActors=True,
        as_dict=True,
    )
    if not isinstance(value, Mapping):
        raise RuntimeError(f"could not read actor bounds: {value}")
    lowered = {str(key).casefold(): item for key, item in value.items()}
    origin = _struct_components(lowered.get("origin"), ("x", "y", "z"))
    extent = _struct_components(lowered.get("boxextent"), ("x", "y", "z"))
    if any(item <= 0.0 for item in extent):
        raise RuntimeError(f"actor bounds are degenerate: {value}")
    return {
        "frame_index": frame_index,
        "origin_cm": origin,
        "extent_cm": extent,
        "minimum_cm": [origin[axis] - extent[axis] for axis in range(3)],
        "maximum_cm": [origin[axis] + extent[axis] for axis in range(3)],
    }


def _spawn_camera(
    game: Any, *, width: int, height: int, hfov: float
) -> tuple[Any, Any]:
    camera_class = game.unreal_service.load_class(
        uclass="AActor", name=CAMERA_BLUEPRINT
    )
    camera = game.unreal_service.spawn_actor(uclass=camera_class)
    capture = game.unreal_service.get_component_by_name(
        actor=camera,
        component_name=CAPTURE_COMPONENT_NAME,
        uclass="USpSceneCaptureComponent2D",
    )
    viewport = game.rendering_service.get_current_viewport_desc()
    game.rendering_service.align_camera_with_viewport(
        camera_sensor=camera,
        camera_components=[capture],
        viewport_desc=viewport,
        widths=width,
        heights=height,
    )
    capture.Initialize()
    capture.initialize_sp_funcs()
    capture.set_property_value(property_name="FOVAngle", property_value=hfov)
    observed = float(capture.get_property_value(property_name="FOVAngle"))
    if abs(observed - hfov) > 1.0e-4:
        raise RuntimeError(f"camera HFOV readback {observed} != {hfov}")
    return camera, capture


def _read_frame(capture: Any) -> np.ndarray:
    frame = capture.read_pixels()["arrays"]["data"][:, :, [0, 1, 2]]
    return np.asarray(frame).copy()


def _apply_camera(camera: Any, plan: Mapping[str, Any]) -> None:
    position = plan["camera"]["ue_position_cm"]
    camera.K2_SetActorLocationAndRotation(
        NewLocation={"X": position[0], "Y": position[1], "Z": position[2]},
        NewRotation={
            "Roll": 0.0,
            "Pitch": 0.0,
            "Yaw": plan["camera"]["ue_yaw_deg"],
        },
        bSweep=False,
        bTeleport=True,
    )


def _set_collision_disabled(component: Any) -> None:
    try:
        component.SetCollisionEnabled(NewType="NoCollision")
    except Exception:
        component.set_property_value(
            property_name="CollisionEnabled", property_value="NoCollision"
        )


def _spawn_scene_meshes(game: Any, mesh_paths: Sequence[str]) -> list[Any]:
    actors = []
    for index, mesh_path in enumerate(mesh_paths):
        mesh = game.unreal_service.load_object(uclass="UStaticMesh", name=mesh_path)
        actor = game.unreal_service.spawn_actor(
            uclass="AStaticMeshActor",
            location={"X": 0.0, "Y": 0.0, "Z": 0.0},
            spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
        )
        component = game.unreal_service.get_component_by_class(
            actor=actor, uclass="UStaticMeshComponent"
        )
        component.SetMobility(NewMobility="Movable")
        component.SetStaticMesh(NewMesh=mesh)
        component.SetCastShadow(NewCastShadow=True)
        _set_collision_disabled(component)
        game.unreal_service.set_stable_name_for_actor(
            actor=actor, stable_name=f"AVEngine/MP3D/mesh_{index:03d}"
        )
        actors.append(actor)
    if len(actors) != 71:
        raise RuntimeError(f"spawned scene mesh count {len(actors)} != 71")
    return actors


def _spawn_lighting(game: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    profile = plan["exposure_and_lighting"]
    key = profile["directional_key"]
    sky = spawn_sky(game=game)
    light = spawn_directional_light(
        game=game,
        yaw_deg=key["yaw_deg"],
        pitch_deg=key["pitch_deg"],
        intensity_lux=key["intensity_lux"],
    )
    light_component = game.unreal_service.get_component_by_class(
        actor=light, uclass="UDirectionalLightComponent"
    )
    light_component.SetCastShadows(bNewValue=True)
    skylight = sky.get("ASkyLight")
    sky_readback = None
    if skylight is not None:
        component = game.unreal_service.get_component_by_class(
            actor=skylight, uclass="USkyLightComponent"
        )
        component.SetIntensity(NewIntensity=profile["skylight_intensity"])
        try:
            component.SetCastShadows(bNewValue=False)
        except Exception:
            pass
        sky_readback = float(component.get_property_value("Intensity"))
    return {
        "status": "pass",
        "directional_key": {
            **key,
            "intensity_readback_lux": float(
                light_component.get_property_value("Intensity")
            ),
            "cast_shadows_readback": bool(
                light_component.get_property_value("CastShadows")
            ),
        },
        "skylight_intensity_requested": profile["skylight_intensity"],
        "skylight_intensity_readback": sky_readback,
        "spawned_sky_actor_classes": sorted(sky),
        "claim_boundary": profile["claim_boundary"],
    }


def _skeletal_component(game: Any, actor: Any) -> Any:
    component = select_skeletal_mesh_component(
        unreal_service=game.unreal_service, actor=actor
    )
    if component is None:
        raise RuntimeError("spawned actor has no populated SkeletalMeshComponent")
    return component


def _spawn_runtime_actors(
    game: Any, spear_root: Path, plan: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    # Keep this argument for ReplicaCAD and QA callers that supply the
    # external UE runtime boundary. It is no longer a Python import root.
    first_states = {
        state["actor_id"]: state for state in plan["frames"][0]["actor_states"]
    }
    result: dict[str, dict[str, Any]] = {}
    for declaration in plan["actors"]:
        actor_id = declaration["actor_id"]
        expected_class = (
            HUMAN_BP_CLASS_PATH if actor_id == "human0" else DOG_BP_CLASS_PATH
        )
        if declaration["blueprint_class_path"] != expected_class:
            raise RuntimeError(f"unexpected UE class for {actor_id}")
        state = first_states[actor_id]
        position = state["translation_ue_cm"]
        anchor = game.unreal_service.spawn_actor(
            uclass="AActor",
            spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
        )
        anchor_root = game.unreal_service.create_scene_component_for_actor(
            owner=anchor,
            scene_component_name=f"{actor_id}_timeline_anchor_root",
            uclass="USceneComponent",
        )
        anchor_root.SetMobility(NewMobility="Movable")
        anchor.SetActorEnableCollision(bNewActorEnableCollision=False)
        anchor.SetActorTickEnabled(bEnabled=True)
        anchor.K2_SetActorLocationAndRotation(
            NewLocation={"X": position[0], "Y": position[1], "Z": position[2]},
            NewRotation={
                "Roll": 0.0,
                "Pitch": 0.0,
                "Yaw": state["actor_yaw_ue_deg"],
            },
            bSweep=False,
            bTeleport=True,
        )
        blueprint = game.unreal_service.load_class(
            uclass="AActor", name=declaration["blueprint_class_path"]
        )
        visual_actor = game.unreal_service.spawn_actor(
            uclass=blueprint,
            location={"X": position[0], "Y": position[1], "Z": position[2]},
            spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
        )
        visual_actor.SetActorEnableCollision(bNewActorEnableCollision=False)
        visual_actor.SetActorTickEnabled(bEnabled=True)
        visual_actor.SetActorScale3D(NewScale3D={"X": 1.0, "Y": 1.0, "Z": 1.0})
        visual_root = visual_actor.K2_GetRootComponent()
        visual_root.SetMobility(NewMobility="Movable")
        attached = visual_root.K2_AttachToComponent(
            Parent=anchor_root,
            SocketName="None",
            LocationRule="SnapToTarget",
            RotationRule="SnapToTarget",
            ScaleRule="KeepWorld",
            bWeldSimulatedBodies=False,
        )
        if attached is not True:
            raise RuntimeError(f"could not attach {actor_id} visual root to anchor")
        if visual_root.GetAttachParent(as_handle=True) != anchor_root.uobject:
            raise RuntimeError(f"{actor_id} visual root attached to the wrong parent")
        component_frame = apply_ue_component_frame_delta(visual_root, declaration)
        component = _skeletal_component(game, visual_actor)
        component.SetComponentTickEnabled(bEnabled=True)
        component.SetCastShadow(NewCastShadow=True)
        animation = game.unreal_service.load_object(
            uclass="UAnimationAsset", name=declaration["walking_animation"]
        )
        animation_length = float(animation.GetPlayLength())
        if not math.isfinite(animation_length) or animation_length <= 0.0:
            raise RuntimeError(f"{actor_id} Walking animation has invalid length")
        component.PlayAnimation(NewAnimToPlay=animation, bLooping=True)
        component.Stop()
        result[actor_id] = {
            "anchor": anchor,
            "anchor_root": anchor_root,
            "visual_actor": visual_actor,
            "visual_root": visual_root,
            "component": component,
            "animation": animation,
            "animation_length_seconds": animation_length,
            "component_frame": component_frame,
            "hierarchy": {
                "status": "pass",
                "timeline_root_owner": "hidden_anchor_actor",
                "asset_frame_owner": "attached_visual_actor_root",
                "visual_parent_readback_matches_anchor": True,
            },
        }
    return result


def _wrap_angle_error(observed: float, expected: float) -> float:
    return abs((float(observed) - float(expected) + 180.0) % 360.0 - 180.0)


def _apply_actor_state(
    runtime: Mapping[str, Any], state: Mapping[str, Any], frame_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    requested_position = float(state["animation_position_seconds"])
    if requested_position > runtime["animation_length_seconds"] + 1.0e-6:
        raise RuntimeError(
            f"animation is shorter than requested state at frame {frame_index}"
        )
    component = runtime["component"]
    component.SetPosition(InPos=requested_position, bFireNotifies=False)
    observed_position = float(component.GetPosition())
    error = abs(observed_position - requested_position)
    if error > ANIMATION_TOLERANCE_SECONDS:
        raise RuntimeError(f"animation readback failed at frame {frame_index}: {error}")
    position = state["translation_ue_cm"]
    anchor = runtime["anchor"]
    anchor.K2_SetActorLocationAndRotation(
        NewLocation={"X": position[0], "Y": position[1], "Z": position[2]},
        NewRotation={
            "Roll": 0.0,
            "Pitch": 0.0,
            "Yaw": state["actor_yaw_ue_deg"],
        },
        bSweep=False,
        bTeleport=True,
    )
    return _actor_readback(anchor, frame_index), {
        "frame_index": frame_index,
        "requested_position_seconds": requested_position,
        "readback_position_seconds": observed_position,
        "absolute_error_seconds": error,
    }


def _grade_frame(frame_bgr: np.ndarray, gain: float) -> np.ndarray:
    graded = np.clip(frame_bgr.astype(np.float32) * float(gain), 0.0, 255.0)
    return np.rint(graded).astype(np.uint8)


class _LuminanceAccumulator:
    def __init__(self, profile: Mapping[str, Any]) -> None:
        qa = profile["qa"]
        self.saturation_threshold = float(qa["luminance_saturation_threshold"])
        self.nonblack_threshold = float(qa["nonblack_luminance_threshold"])
        self.count = 0
        self.total = 0.0
        self.saturated = 0
        self.nonblack = 0
        self.histogram = np.zeros(4096, dtype=np.int64)

    def add_bgr(self, frame: np.ndarray) -> None:
        array = frame.astype(np.float64) / 255.0
        luminance = (
            0.2126 * array[..., 2] + 0.7152 * array[..., 1] + 0.0722 * array[..., 0]
        )
        self.count += int(luminance.size)
        self.total += float(np.sum(luminance))
        self.saturated += int(np.count_nonzero(luminance >= self.saturation_threshold))
        self.nonblack += int(np.count_nonzero(luminance >= self.nonblack_threshold))
        bins = np.minimum((luminance * 4095.0).astype(np.int64), 4095)
        self.histogram += np.bincount(bins.ravel(), minlength=4096)

    def result(self, profile: Mapping[str, Any]) -> dict[str, Any]:
        if self.count <= 0:
            raise RuntimeError("luminance QA received no pixels")
        target = int(math.ceil(0.95 * self.count))
        p95_bin = int(np.searchsorted(np.cumsum(self.histogram), target))
        mean = self.total / self.count
        p95 = p95_bin / 4095.0
        saturated = self.saturated / self.count
        nonblack = self.nonblack / self.count
        qa = profile["qa"]
        passed = (
            saturated <= qa["maximum_saturated_fraction"]
            and nonblack >= qa["minimum_nonblack_fraction"]
            and mean >= qa["minimum_mean_luminance"]
            and p95 >= qa["minimum_p95_luminance"]
            and mean <= qa["maximum_mean_luminance"]
            and p95 <= qa["maximum_p95_luminance"]
        )
        return {
            "status": "pass" if passed else "fail",
            "pixel_count": self.count,
            "mean_luminance": mean,
            "p95_luminance_histogram_4096_bins": p95,
            "saturated_fraction": saturated,
            "nonblack_fraction": nonblack,
            "nonblack_luminance_threshold": self.nonblack_threshold,
            "thresholds": dict(qa),
        }


def _root_gate(
    expected_frames: Sequence[Mapping[str, Any]],
    actor_readbacks: Mapping[str, Sequence[Mapping[str, Any]]],
    camera_readbacks: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for actor_id in ("human0", "dog0"):
        positions = []
        yaws = []
        roll_pitch = []
        records = actor_readbacks[actor_id]
        if len(records) != len(expected_frames):
            raise RuntimeError(f"{actor_id} readback count differs")
        for frame, observed in zip(expected_frames, records):
            if observed.get("frame_index") != frame.get("frame_index"):
                raise RuntimeError(f"{actor_id} readback frame order differs")
            state = next(
                item for item in frame["actor_states"] if item["actor_id"] == actor_id
            )
            positions.append(
                max(
                    abs(float(left) - float(right))
                    for left, right in zip(
                        observed["location_cm"], state["translation_ue_cm"]
                    )
                )
            )
            rotation = observed["rotation_deg"]
            yaws.append(_wrap_angle_error(rotation[2], state["actor_yaw_ue_deg"]))
            roll_pitch.append(max(abs(rotation[0]), abs(rotation[1])))
        maximum_position = max(positions)
        maximum_yaw = max(yaws)
        maximum_roll_pitch = max(roll_pitch)
        if (
            maximum_position > POSITION_TOLERANCE_CM
            or maximum_yaw > ANGLE_TOLERANCE_DEG
            or maximum_roll_pitch > ANGLE_TOLERANCE_DEG
        ):
            raise RuntimeError(f"{actor_id} root readback gate failed")
        result[actor_id] = {
            "status": "pass",
            "frame_count": len(records),
            "maximum_absolute_position_error_cm": maximum_position,
            "maximum_absolute_yaw_error_deg": maximum_yaw,
            "maximum_absolute_roll_or_pitch_deg": maximum_roll_pitch,
        }

    if len(camera_readbacks) != len(expected_frames):
        raise RuntimeError("camera readback count differs")
    for frame, observed in zip(expected_frames, camera_readbacks):
        if observed.get("frame_index") != frame.get("frame_index"):
            raise RuntimeError("camera readback frame order differs")
    expected_camera = plan["camera"]["ue_position_cm"]
    expected_yaw = plan["camera"]["ue_yaw_deg"]
    position_errors = [
        max(
            abs(float(left) - float(right))
            for left, right in zip(record["location_cm"], expected_camera)
        )
        for record in camera_readbacks
    ]
    yaw_errors = [
        _wrap_angle_error(record["rotation_deg"][2], expected_yaw)
        for record in camera_readbacks
    ]
    camera_roll_pitch = [
        max(abs(record["rotation_deg"][0]), abs(record["rotation_deg"][1]))
        for record in camera_readbacks
    ]
    if (
        max(position_errors) > POSITION_TOLERANCE_CM
        or max(yaw_errors) > ANGLE_TOLERANCE_DEG
        or max(camera_roll_pitch) > ANGLE_TOLERANCE_DEG
    ):
        raise RuntimeError("camera readback gate failed")
    result["camera"] = {
        "status": "pass",
        "frame_count": len(camera_readbacks),
        "maximum_absolute_position_error_cm": max(position_errors),
        "maximum_absolute_yaw_error_deg": max(yaw_errors),
        "maximum_absolute_roll_or_pitch_deg": max(camera_roll_pitch),
    }
    return result


def _audio_packet_sha256(path: Path) -> str:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-c",
            "copy",
            "-f",
            "streamhash",
            "-hash",
            "sha256",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    prefix = "0,a,SHA256="
    if len(lines) != 1 or not lines[0].startswith(prefix):
        raise RuntimeError(f"unexpected audio streamhash for {path}: {lines}")
    return lines[0][len(prefix) :].lower()


def _probe_media(
    path: Path,
    *,
    width: int,
    height: int,
    expect_audio: bool,
) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            (
                "stream=codec_type,codec_name,width,height,avg_frame_rate,"
                "nb_read_frames,sample_rate,channels,duration:format=duration"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    videos = [item for item in payload["streams"] if item["codec_type"] == "video"]
    audio = [item for item in payload["streams"] if item["codec_type"] == "audio"]
    if len(videos) != 1 or len(audio) != int(expect_audio):
        raise RuntimeError(f"media stream closure failed: {path}")
    video = videos[0]
    if (
        video.get("codec_name") != "h264"
        or int(video.get("width", -1)) != width
        or int(video.get("height", -1)) != height
        or video.get("avg_frame_rate") != f"{M5_1_FPS}/1"
        or int(video.get("nb_read_frames", -1)) != M5_1_FRAME_COUNT
    ):
        raise RuntimeError(f"video readback failed: {video}")
    duration = float(payload["format"]["duration"])
    if abs(duration - M5_1_FRAME_COUNT / M5_1_FPS) > 1.0 / M5_1_FPS:
        raise RuntimeError(f"media duration changed: {duration}")
    if expect_audio and (
        audio[0].get("codec_name") != "aac"
        or int(audio[0].get("channels", -1)) != 2
        or int(audio[0].get("sample_rate", -1)) != 16_000
    ):
        raise RuntimeError(f"binaural stream readback failed: {audio[0]}")
    return {
        "status": "pass",
        "path": path.name,
        "width": width,
        "height": height,
        "frame_count": M5_1_FRAME_COUNT,
        "frame_rate_hz": M5_1_FPS,
        "duration_seconds": duration,
        "audio_packet_sha256": _audio_packet_sha256(path) if expect_audio else None,
    }


def _encode_media(
    *, frames_root: Path, habitat_review: Path, output_root: Path
) -> dict[str, Path]:
    visual = output_root / "mp3d_spear_visual_only.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(M5_1_FPS),
            "-i",
            str(frames_root / "frame_%04d.png"),
            "-frames:v",
            str(M5_1_FRAME_COUNT),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(visual),
        ],
        check=True,
    )
    clean = output_root / "mp3d_spear_clean_binaural.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(visual),
            "-i",
            str(habitat_review),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(clean),
        ],
        check=True,
    )
    topdown = output_root / "mp3d_spear_topdown_binaural.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(visual),
            "-i",
            str(habitat_review),
            "-filter_complex",
            (
                "[0:v]scale=640:360:flags=lanczos,pad=640:480:0:60:black[ue];"
                "[1:v]crop=640:480:640:0[top];[ue][top]hstack=inputs=2[out]"
            ),
            "-map",
            "[out]",
            "-map",
            "1:a:0",
            "-frames:v",
            str(M5_1_FRAME_COUNT),
            "-r",
            str(M5_1_FPS),
            "-vsync",
            "cfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-c:a",
            "copy",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(topdown),
        ],
        check=True,
    )
    triptych = output_root / "mp3d_spear_habitat_topdown_triptych_binaural.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(visual),
            "-i",
            str(habitat_review),
            "-filter_complex",
            (
                "[0:v]scale=640:360:flags=lanczos,pad=640:480:0:60:black[ue];"
                "[1:v]setsar=1[hab];[ue][hab]hstack=inputs=2[out]"
            ),
            "-map",
            "[out]",
            "-map",
            "1:a:0",
            "-frames:v",
            str(M5_1_FRAME_COUNT),
            "-r",
            str(M5_1_FPS),
            "-vsync",
            "cfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-c:a",
            "copy",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            str(triptych),
        ],
        check=True,
    )
    return {"visual": visual, "clean": clean, "topdown": topdown, "triptych": triptych}


def _cleanup_failed_constructor(*, executable: Path, temporary_directory: Path) -> None:
    try:
        import psutil
    except ImportError:
        return
    config_suffix = str(temporary_directory / "config.yaml")
    executable_text = str(executable.resolve())
    matched = []
    for process in psutil.process_iter(("pid", "cmdline")):
        try:
            command = process.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        joined = " ".join(str(item) for item in command)
        if executable_text not in joined and "SpearSim" not in joined:
            continue
        if not any(
            str(item).startswith("-sp-config-file=")
            and str(item).endswith(config_suffix)
            for item in command
        ):
            continue
        try:
            matched.extend(process.children(recursive=True))
            matched.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for process in reversed(matched):
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(matched, timeout=10.0)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def _configure_instance(
    args: argparse.Namespace, plan: Mapping[str, Any]
) -> tuple[Any, Path]:
    spear_root = args.spear_root.resolve()
    editor = args.unreal_editor.resolve()
    project = args.ue_project.resolve()
    if not editor.is_file() or not os.access(editor, os.X_OK):
        raise RuntimeError(f"UnrealEditor is missing or not executable: {editor}")
    if project.suffix != ".uproject" or not project.is_file():
        raise RuntimeError(f"isolated SPEAR uproject is missing: {project}")
    old_project = (
        spear_root / "cpp/unreal_projects/SpearSim/SpearSim.uproject"
    ).resolve()
    if project == old_project or old_project.parent in project.parents:
        raise RuntimeError(
            "refusing to render MP3D through the old dirty SPEAR project"
        )
    settings = parallel_instance_settings(
        args.rpc_port, graphics_adapter=args.graphics_adapter
    )
    config = spear_client.get_config(user_config_files=[])
    config.defrost()
    config.SPEAR.LAUNCH_MODE = "editor"
    config.SPEAR.INSTANCE.EDITOR_EXECUTABLE = str(editor)
    config.SPEAR.INSTANCE.EDITOR_UPROJECT = str(project)
    config.SPEAR.INSTANCE.EDITOR_LAUNCH_MODE = "game"
    config.SPEAR.INSTANCE.INITIALIZE_CLIENT_MAX_TIME_SECONDS = (
        INITIALIZE_CLIENT_MAX_TIME_SECONDS
    )
    config.SPEAR.INSTANCE.CLIENT_INTERNAL_TIMEOUT_SECONDS = (
        CLIENT_INTERNAL_TIMEOUT_SECONDS
    )
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = ENTRY_MAP
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.FIXED_DELTA_TIME = 1.0 / M5_1_FPS
    config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT = settings["rpc_port"]
    config.SPEAR.INSTANCE.TEMP_DIR = settings["temp_dir"]
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.log = settings["log"]
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.renderoffscreen = None
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.execcmds = ",".join(
        plan["exposure_and_lighting"]["console_commands"]
    )
    config.SP_CORE.SHARED_MEMORY_INITIAL_UNIQUE_ID = settings[
        "shared_memory_initial_unique_id"
    ]
    if settings["graphics_adapter"] is not None:
        config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.graphicsadapter = settings[
            "graphics_adapter"
        ]
    vulkan_icd = os.environ.get("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    if Path(vulkan_icd).is_file():
        config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = vulkan_icd
    config.freeze()
    spear_client.configure_system(config=config)
    try:
        instance = spear_client.Instance(config=config)
    except BaseException:
        _cleanup_failed_constructor(
            executable=editor, temporary_directory=Path(settings["temp_dir"])
        )
        raise
    return instance, spear_root


def run(args: argparse.Namespace) -> Path:
    output_root = args.output_dir.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to replace output directory: {output_root}")
    output_root.mkdir(parents=True)
    plan = build_execution_plan(args)
    if plan.get("schema") != M5_1_EXECUTION_SCHEMA:
        raise RuntimeError("compiled MP3D plan schema changed")
    plan_path = output_root / "execution_plan.json"
    _write_json(plan_path, plan)
    if args.dry_run:
        print(f"SPEAR_MP3D_DRY_RUN_OK plan={plan_path}", flush=True)
        return plan_path

    material_result = _load_json(
        args.ue_material_color_result.resolve(), owner="UE material-color result"
    )
    result_project = material_result.get("project_file")
    if (
        not isinstance(result_project, str)
        or Path(result_project).resolve() != args.ue_project.resolve()
    ):
        raise RuntimeError(
            "MP3D material-color reload result does not belong to the selected "
            "isolated UE project"
        )
    habitat_rgb = np.load(args.habitat_rgb_array.resolve(), mmap_mode="r")
    if (
        habitat_rgb.shape != (M5_1_FRAME_COUNT, 240, 320, 3)
        or habitat_rgb.dtype != np.uint8
    ):
        raise RuntimeError(
            f"unexpected Habitat MP3D RGB authority: shape={habitat_rgb.shape} "
            f"dtype={habitat_rgb.dtype}"
        )

    frames_root = output_root / "frames"
    frames_root.mkdir()
    instance, spear_root = _configure_instance(args, plan)
    game = instance.get_game()
    actor_readbacks = {"human0": [], "dog0": []}
    animation_readbacks = {"human0": [], "dog0": []}
    actor_bounds = {"human0": [], "dog0": []}
    camera_readbacks: list[dict[str, Any]] = []
    luminance = _LuminanceAccumulator(plan["exposure_and_lighting"])
    ue_color_frames: list[np.ndarray] = []
    habitat_color_frames: list[np.ndarray] = []
    lighting: dict[str, Any]
    smoke_index = args.smoke_frame_index
    capture_indices = (
        [smoke_index] if smoke_index is not None else list(range(M5_1_FRAME_COUNT))
    )
    try:
        with instance.begin_frame():
            room_actors = _spawn_scene_meshes(
                game, plan["scene"]["static_mesh_object_paths"]
            )
            lighting = _spawn_lighting(game, plan)
            camera, capture = _spawn_camera(
                game=game,
                width=plan["render"]["width"],
                height=plan["render"]["height"],
                hfov=plan["camera"]["horizontal_fov_deg"],
            )
            runtimes = _spawn_runtime_actors(game, spear_root, plan)
            _apply_camera(camera, plan)
            for state in plan["frames"][0]["actor_states"]:
                _apply_actor_state(runtimes[state["actor_id"]], state, 0)
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                bPaused=False
            )
        with instance.end_frame():
            pass
        if len(room_actors) != 71:
            raise RuntimeError("runtime MP3D scene mesh closure changed")
        instance.step(num_frames=plan["render"]["streaming_warmup_frames"])
        with instance.begin_frame():
            _apply_camera(camera, plan)
        with instance.end_frame():
            pass
        instance.step(num_frames=plan["render"]["camera_warmup_frames"])

        import cv2

        for output_index, frame_index in enumerate(capture_indices):
            frame_plan = plan["frames"][frame_index]
            with instance.begin_frame():
                for state in frame_plan["actor_states"]:
                    actor_id = state["actor_id"]
                    root, animation = _apply_actor_state(
                        runtimes[actor_id], state, frame_index
                    )
                    actor_readbacks[actor_id].append(root)
                    animation_readbacks[actor_id].append(animation)
                _apply_camera(camera, plan)
                camera_readbacks.append(_actor_readback(camera, frame_index))
            with instance.end_frame():
                raw = _read_frame(capture)
                for actor_id, runtime in runtimes.items():
                    actor_bounds[actor_id].append(
                        _actor_bounds_readback(runtime["visual_actor"], frame_index)
                    )
                if raw.shape != (
                    plan["render"]["height"],
                    plan["render"]["width"],
                    3,
                ):
                    raise RuntimeError(f"unexpected captured frame shape: {raw.shape}")
                frame = _grade_frame(
                    raw, plan["exposure_and_lighting"]["fixed_output_gain"]
                )
                luminance.add_bgr(frame)
                ue_color_frames.append(
                    cv2.resize(
                        frame[:, :, ::-1],
                        (320, 240),
                        interpolation=cv2.INTER_AREA,
                    )
                )
                habitat_color_frames.append(np.asarray(habitat_rgb[frame_index]).copy())
                path = frames_root / f"frame_{output_index:04d}.png"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"could not write frame: {path}")
            if smoke_index is None and frame_index % M5_1_FPS == 0:
                print(
                    f"[spear-mp3d] frame {frame_index:03d}/{M5_1_FRAME_COUNT - 1}",
                    flush=True,
                )
    finally:
        instance.close(force=True)

    luminance_qa = luminance.result(plan["exposure_and_lighting"])
    if luminance_qa["status"] != "pass":
        raise RuntimeError(f"MP3D fixed-exposure QA failed: {luminance_qa}")
    color_qa = render_color_fidelity_qa(
        np.stack(ue_color_frames),
        np.stack(habitat_color_frames),
        plan["exposure_and_lighting"],
    )
    if color_qa["status"] != "pass":
        raise RuntimeError(f"MP3D rendered color-fidelity QA failed: {color_qa}")
    captured_frame_plans = [plan["frames"][index] for index in capture_indices]
    bounds_gate = summarize_actor_bounds(
        expected_frames=captured_frame_plans,
        actor_declarations=plan["actors"],
        actor_bounds=actor_bounds,
    )
    runtime_hierarchy = {
        actor_id: {
            "asset_local_frame": runtime["component_frame"],
            "runtime_hierarchy": runtime["hierarchy"],
        }
        for actor_id, runtime in runtimes.items()
    }
    if smoke_index is not None:
        smoke = {
            "schema": SMOKE_SCHEMA,
            "status": "pass",
            "backend_role": "comparison_visual",
            "room_id": MP3D_ROOM_ID,
            "frame_index": smoke_index,
            "frame_path": str((frames_root / "frame_0000.png").resolve()),
            "lighting": lighting,
            "luminance_qa": luminance_qa,
            "color_fidelity_qa": color_qa,
            "material_color_contract": plan["scene"]["material_color_contract"],
            "visual_bounds_readback": bounds_gate,
            "runtime_actor_hierarchy": runtime_hierarchy,
            "clock": plan["clock"],
            "claim_boundary": "single-frame runtime smoke; not a 270-frame delivery",
        }
        smoke_path = output_root / "smoke_evidence.json"
        _write_json(smoke_path, smoke)
        print(
            f"SPEAR_MP3D_SMOKE_OK frame={smoke['frame_path']} evidence={smoke_path}",
            flush=True,
        )
        return smoke_path

    root_gate = _root_gate(plan["frames"], actor_readbacks, camera_readbacks, plan)
    animation_gate = {}
    for actor_id, records in animation_readbacks.items():
        maximum = max(item["absolute_error_seconds"] for item in records)
        if len(records) != M5_1_FRAME_COUNT or maximum > ANIMATION_TOLERANCE_SECONDS:
            raise RuntimeError(f"{actor_id} animation-phase closure failed")
        animation_gate[actor_id] = {
            "status": "pass",
            "frame_count": len(records),
            "maximum_absolute_error_seconds": maximum,
            "tolerance_seconds": ANIMATION_TOLERANCE_SECONDS,
        }

    habitat_review = args.habitat_review.resolve()
    media_paths = _encode_media(
        frames_root=frames_root,
        habitat_review=habitat_review,
        output_root=output_root,
    )
    media = {
        "ue_visual_only": _probe_media(
            media_paths["visual"], width=1280, height=720, expect_audio=False
        ),
        "ue_clean_binaural": _probe_media(
            media_paths["clean"], width=1280, height=720, expect_audio=True
        ),
        "ue_topdown_binaural": _probe_media(
            media_paths["topdown"], width=1280, height=480, expect_audio=True
        ),
        "ue_habitat_topdown_triptych_binaural": _probe_media(
            media_paths["triptych"], width=1920, height=480, expect_audio=True
        ),
    }
    source_audio_hash = _audio_packet_sha256(habitat_review)
    for name, record in media.items():
        if (
            name != "ue_visual_only"
            and record["audio_packet_sha256"] != source_audio_hash
        ):
            raise RuntimeError(f"{name} changed the authoritative binaural packets")

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "status": "pass",
        "backend_role": "comparison_visual",
        "room_id": MP3D_ROOM_ID,
        "route_id": plan["route_id"],
        "clock": plan["clock"],
        "authority": plan["authority"],
        "runtime": {
            "map": ENTRY_MAP,
            "rpc_port": args.rpc_port,
            "graphics_adapter": args.graphics_adapter,
            "spawned_scene_mesh_count": 71,
            "scene_mesh_collision": "NoCollision",
            "resolution": [1280, 720],
            "frame_rate_hz": M5_1_FPS,
            "streaming_warmup_frames": plan["render"]["streaming_warmup_frames"],
            "camera_warmup_frames": plan["render"]["camera_warmup_frames"],
            "auto_exposure_console_commands_requested": plan["exposure_and_lighting"][
                "console_commands"
            ],
            "fixed_output_gain": plan["exposure_and_lighting"]["fixed_output_gain"],
            "lighting": lighting,
        },
        "readback": {
            "root_and_camera": root_gate,
            "animation_phase": animation_gate,
            "visual_bounds": bounds_gate,
            "runtime_actor_hierarchy": runtime_hierarchy,
        },
        "exposure_qa": luminance_qa,
        "color_fidelity_qa": color_qa,
        "media": media,
        "audio_authority": {
            "source_video": _file_record(habitat_review),
            "audio_packet_sha256": source_audio_hash,
            "semantics": "Habitat-native two-channel binaural; no camera-FOV cutoff",
        },
        "inputs": {
            "execution_plan": _file_record(plan_path),
            "route_manifest": _file_record(args.route_manifest.resolve()),
            "capture_evidence": _file_record(args.capture_evidence.resolve()),
            "frame_readback": _file_record(args.frame_readback.resolve()),
            "navmesh_gate": _file_record(args.navmesh_gate.resolve()),
            "source_program": _file_record(args.source_program.resolve()),
            "emitter_trajectories": _file_record(args.emitter_trajectories.resolve()),
            "ue_material_color_result": _file_record(
                args.ue_material_color_result.resolve()
            ),
            "habitat_rgb_array": _file_record(args.habitat_rgb_array.resolve()),
        },
        "claim_boundary": plan["claim_boundary"],
    }
    evidence_path = output_root / "evidence.json"
    _write_json(evidence_path, evidence)
    if not args.keep_frames:
        shutil.rmtree(frames_root)
    print(
        f"SPEAR_MP3D_CANARY_OK video={media_paths['topdown']} evidence={evidence_path}",
        flush=True,
    )
    return evidence_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spear-root", type=Path, default=DEFAULT_SPEAR_ROOT)
    parser.add_argument("--unreal-editor", type=Path)
    parser.add_argument("--ue-project", type=Path)
    parser.add_argument("--route-manifest", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--capture-evidence", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--frame-readback", type=Path, default=DEFAULT_FRAME_READBACK)
    parser.add_argument("--navmesh-gate", type=Path, default=DEFAULT_NAVMESH_GATE)
    parser.add_argument("--source-program", type=Path, default=DEFAULT_SOURCE_PROGRAM)
    parser.add_argument("--emitter-trajectories", type=Path, default=DEFAULT_EMITTERS)
    parser.add_argument("--habitat-review", type=Path, default=DEFAULT_HABITAT_REVIEW)
    parser.add_argument("--room-registry", type=Path, default=DEFAULT_ROOM_REGISTRY)
    parser.add_argument(
        "--room-qualification", type=Path, default=DEFAULT_ROOM_QUALIFICATION
    )
    parser.add_argument("--ue-import-manifest", type=Path)
    parser.add_argument("--ue-material-color-result", type=Path, required=True)
    parser.add_argument("--human-ue-manifest", type=Path)
    parser.add_argument("--habitat-rgb-array", type=Path, default=DEFAULT_HABITAT_RGB)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rpc-port", type=int, default=39331)
    parser.add_argument("--graphics-adapter", type=int)
    parser.add_argument("--fixed-output-gain", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument(
        "--smoke-frame-index",
        type=int,
        help="Render one selected authority frame; this is never formal evidence.",
    )
    args = parser.parse_args(argv)
    if not 1024 <= args.rpc_port <= 65535:
        parser.error("--rpc-port must be in [1024,65535]")
    if args.graphics_adapter is not None and args.graphics_adapter < 0:
        parser.error("--graphics-adapter must be non-negative")
    if (
        args.smoke_frame_index is not None
        and not 0 <= args.smoke_frame_index < M5_1_FRAME_COUNT
    ):
        parser.error("--smoke-frame-index must be in [0,269]")
    if not 0.1 <= args.fixed_output_gain <= 1.0:
        parser.error("--fixed-output-gain must be in [0.1,1.0]")
    if not args.dry_run and (args.unreal_editor is None or args.ue_project is None):
        parser.error("--unreal-editor and --ue-project are required outside --dry-run")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

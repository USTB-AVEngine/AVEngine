#!/usr/bin/env python3
"""Render M6.x S0/S3/S4 through the native SPEAR Apartment map.

This is an optional comparison-visual runner.  It reads the Habitat-native
Timeline/protocol bundle, teleports the already-imported UE actors to those
exact roots, samples the declared animation phase, and captures native UE
pixels.  It never replans a route or creates a second audio/flag authority.

``--dry-run`` needs only the AVEngine Python package.  A real render must run
inside ``spear-env`` and receives the old SPEAR checkout through
``--spear-root``; that checkout is imported read-only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.optional_backends.spear_apartment import (
    ANIMATION_TOLERANCE_SECONDS,
    CAMERA_WARMUP_FRAMES,
    FPS,
    FRAME_COUNT,
    HEIGHT,
    NATIVE_APARTMENT_MAP,
    STREAMING_WARMUP_FRAMES,
    WIDTH,
    animation_position_seconds,
    apply_ue_component_frame_delta,
    build_clean_binaural_mux_command,
    build_native_apartment_motion_pilot_suite,
    build_native_apartment_asset_bound_suite,
    build_native_apartment_suite,
    build_png_encode_command,
    build_rawvideo_encode_command,
    build_topdown_visual_command,
    load_apartment_lighting_profile,
    summarize_actor_bounds,
    summarize_anatomical_forward_readbacks,
    summarize_root_readbacks,
)


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = REPOSITORY / "tmp/m6x/fixed_apartment_canary_20260720_02"
DEFAULT_SPEAR_ROOT = REPOSITORY.parent / "AVEngine/external/SPEAR"
DEFAULT_LIGHTING_PROFILES = (
    REPOSITORY / "examples/m6y/spear_apartment_lighting_profiles.json"
)
CAMERA_BLUEPRINT = "/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C"
CAPTURE_COMPONENT_NAME = "DefaultSceneRoot.final_tone_curve_hdr_"
EVIDENCE_SCHEMA = "avengine_optional_spear_apartment_runtime_evidence_v2"
TIMING_SCHEMA = "avengine_apartment_runtime_timing_v1"
REQUIRED_SAMPLE_OUTPUTS = (
    "ue_visual_only.mp4",
    "ue_topdown_visual_only.mp4",
    "ue_clean_binaural.mp4",
    "ue_topdown_binaural.mp4",
)
INITIALIZE_CLIENT_MAX_TIME_SECONDS = 600.0
CLIENT_INTERNAL_TIMEOUT_SECONDS = 60.0
ANATOMICAL_FORWARD_SAMPLE_FRAMES = (0, FRAME_COUNT // 2, FRAME_COUNT - 1)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _elapsed_seconds(started_at: float) -> float:
    """Return a JSON-safe monotonic wall-clock duration."""

    elapsed = time.perf_counter() - started_at
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError(f"invalid wall-clock duration: {elapsed}")
    return elapsed


def _struct_components(value: Any, names: Sequence[str]) -> list[float]:
    expected = [name.casefold() for name in names]
    current = value
    for _ in range(3):
        if not isinstance(current, Mapping):
            break
        lowered = {str(key).casefold(): item for key, item in current.items()}
        if all(name in lowered for name in expected):
            result = [float(lowered[name]) for name in expected]
            if not all(math.isfinite(item) for item in result):
                break
            return result
        if "returnvalue" in lowered and isinstance(lowered["returnvalue"], Mapping):
            current = lowered["returnvalue"]
            continue
        if len(current) == 1:
            candidate = next(iter(current.values()))
            if isinstance(candidate, Mapping):
                current = candidate
                continue
        break
    raise RuntimeError(f"could not read Unreal struct components {expected}: {value}")


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


def _spawn_camera(game: Any) -> tuple[Any, Any]:
    """Spawn a SceneCapture without altering native-map lights or geometry."""

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
        widths=WIDTH,
        heights=HEIGHT,
    )
    capture.Initialize()
    capture.initialize_sp_funcs()
    capture.set_property_value(property_name="FOVAngle", property_value=105.0)
    observed_fov = float(capture.get_property_value(property_name="FOVAngle"))
    if abs(observed_fov - 105.0) > 1.0e-4:
        raise RuntimeError(f"camera HFOV readback {observed_fov} != 105")
    return camera, capture


def _spawn_generated_lights(
    game: Any, lighting_profile: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Spawn and read back the profile's visual-only soft point lights."""

    records = []
    for light in lighting_profile["generated_lights"]:
        position = light["position_ue_cm"]
        actor = game.unreal_service.spawn_actor(
            uclass="APointLight",
            location={"X": position[0], "Y": position[1], "Z": position[2]},
        )
        if actor is None:
            raise RuntimeError(f"could not spawn generated light {light['light_id']}")
        actor.K2_GetRootComponent().SetMobility(NewMobility="Movable")
        component = game.unreal_service.get_component_by_class(
            actor=actor, uclass="UPointLightComponent"
        )
        component.SetIntensity(NewIntensity=light["intensity_lumens"])
        component.SetAttenuationRadius(NewRadius=light["attenuation_radius_cm"])
        component.SetCastShadows(bNewValue=light["cast_shadows"])
        component.set_property_value(
            property_name="SourceRadius", property_value=light["source_radius_cm"]
        )
        component.set_property_value(
            property_name="SoftSourceRadius",
            property_value=light["soft_source_radius_cm"],
        )
        component.set_property_value(
            property_name="bUseTemperature", property_value=True
        )
        component.set_property_value(
            property_name="Temperature",
            property_value=light["temperature_kelvin"],
        )
        record = {
            **light,
            "location_readback_cm": _struct_components(
                actor.K2_GetActorLocation(as_dict=True), ("x", "y", "z")
            ),
            "intensity_readback_lumens": float(
                component.get_property_value(property_name="Intensity")
            ),
            "attenuation_readback_cm": float(
                component.get_property_value(property_name="AttenuationRadius")
            ),
            "temperature_readback_kelvin": float(
                component.get_property_value(property_name="Temperature")
            ),
            "cast_shadows_readback": bool(
                component.get_property_value(property_name="CastShadows")
            ),
        }
        if (
            max(
                abs(record["location_readback_cm"][axis] - position[axis])
                for axis in range(3)
            )
            > 1.0e-3
        ):
            raise RuntimeError(f"generated light {light['light_id']} moved")
        if (
            abs(record["intensity_readback_lumens"] - light["intensity_lumens"])
            > 1.0e-3
        ):
            raise RuntimeError(f"generated light {light['light_id']} intensity drifted")
        records.append(record)
    return records


def _read_frame(capture: Any) -> Any:
    return capture.read_pixels()["arrays"]["data"][:, :, [0, 1, 2]]


def _load_skeletal_component(game: Any, actor: Any, spear_root: Path) -> Any:
    spike_dir = spear_root / "tools/spike_rlr"
    sys.path.insert(0, str(spike_dir))
    from rig_direction_check import select_skeletal_mesh_component

    component = select_skeletal_mesh_component(
        unreal_service=game.unreal_service, actor=actor
    )
    if component is None:
        raise RuntimeError("spawned actor has no populated SkeletalMeshComponent")
    return component


def _sample_anatomical_forward(
    game: Any,
    actor: Any,
    frame_index: int,
    *,
    explicit_quadruped_bones: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read the rendered skeleton's semantic forward inside an active frame."""

    from rig_direction_check import (
        sample_body_basis_in_frame,
        sample_body_bone_position_in_frame,
    )

    diagnostics: list[dict[str, Any]] = []
    if explicit_quadruped_bones is None:
        basis = sample_body_basis_in_frame(
            actor,
            unreal_service=game.unreal_service,
            diagnostics=diagnostics,
        )
    else:
        required = {"rear", "front", "body", "left_foot", "right_foot"}
        if set(explicit_quadruped_bones) != required:
            raise RuntimeError(
                "explicit quadruped basis must define exactly " f"{sorted(required)}"
            )
        positions = {}
        for role, bone_name in explicit_quadruped_bones.items():
            position = sample_body_bone_position_in_frame(
                actor,
                bone_name,
                unreal_service=game.unreal_service,
                diagnostics=diagnostics,
            )
            if position is None:
                basis = None
                break
            positions[role] = position
        else:
            # UE world Z is authoritative up. Project the package-declared
            # torso-to-muzzle vector onto the floor plane instead of deriving
            # up from a morphology-dependent AABB or paw/body proportions.
            # This works for both compact cats and longer generated dogs.
            rear = np.asarray(positions["rear"], dtype=np.float64)
            front = np.asarray(positions["front"], dtype=np.float64)
            raw_forward = front - rear
            forward_xy = raw_forward[:2]
            forward_xy_norm = float(np.linalg.norm(forward_xy))
            if forward_xy_norm <= 1.0e-6:
                raise RuntimeError(
                    "explicit quadruped torso-to-muzzle direction has no "
                    "horizontal component"
                )
            forward = np.asarray(
                [forward_xy[0] / forward_xy_norm, forward_xy[1] / forward_xy_norm, 0.0],
                dtype=np.float64,
            )
            left_foot = np.asarray(positions["left_foot"], dtype=np.float64)
            right_foot = np.asarray(positions["right_foot"], dtype=np.float64)
            paw_delta_xy = (right_foot - left_foot)[:2]
            paw_delta_norm = float(np.linalg.norm(paw_delta_xy))
            paw_alignment = None
            if paw_delta_norm > 1.0e-6:
                right = np.asarray([-forward[1], forward[0]], dtype=np.float64)
                paw_alignment = float(
                    np.dot(right, paw_delta_xy / paw_delta_norm)
                )
            feet_center_z = 0.5 * (left_foot[2] + right_foot[2])
            basis = {
                "basis_kind": "asset_bound_quadruped_world_z_head_projection_v1",
                "up_vector_ue": [0.0, 0.0, 1.0],
                "right_vector_ue": [-float(forward[1]), float(forward[0]), 0.0],
                "forward_vector_ue": forward.tolist(),
                "forward_yaw_ue_deg": float(
                    np.degrees(np.arctan2(forward[1], forward[0]))
                ),
                "raw_torso_to_muzzle_vector_ue_cm": raw_forward.tolist(),
                "body_height_above_paired_paws_cm": float(
                    positions["body"][2] - feet_center_z
                ),
                "anatomical_right_alignment": paw_alignment,
            }
            basis["bone_names"] = dict(explicit_quadruped_bones)
            basis["positions_ue_cm"] = {
                role: [float(value) for value in position]
                for role, position in positions.items()
            }
    if basis is None:
        raise RuntimeError(
            "could not derive rendered anatomical forward at frame "
            f"{frame_index}: {diagnostics}"
        )
    record = {
        "frame_index": frame_index,
        "basis_kind": basis["basis_kind"],
        "forward_vector_ue": basis["forward_vector_ue"],
        "forward_yaw_ue_deg": basis["forward_yaw_ue_deg"],
        "bone_names": basis["bone_names"],
    }
    if "positions_ue_cm" in basis:
        record["positions_ue_cm"] = basis["positions_ue_cm"]
    return record


def _spawn_runtime_actors(
    game: Any, scenario: Mapping[str, Any], spear_root: Path
) -> dict[str, dict[str, Any]]:
    plan = scenario["plan"]
    first_states = {
        item["actor_id"]: item for item in plan["frames"][0]["actor_states"]
    }
    runtimes: dict[str, dict[str, Any]] = {}
    for declaration in plan["actors"]:
        actor_id = declaration["actor_id"]
        state = first_states[actor_id]
        position = state["translation_ue_cm"]

        # Keep the authoritative Timeline transform on a deliberately empty
        # anchor.  Imported animal Blueprints may use their skeletal mesh as
        # the actor root, so applying an asset-local correction directly to
        # that actor is overwritten by the next Timeline teleport.  A child
        # visual actor gives the two transforms distinct owners:
        #
        #   Timeline/source center -> anchor world transform
        #   imported asset frame   -> visual child relative transform
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
        observed_parent = visual_root.GetAttachParent(as_handle=True)
        if observed_parent != anchor_root.uobject:
            raise RuntimeError(f"{actor_id} visual root attached to the wrong parent")

        component = _load_skeletal_component(game, visual_actor, spear_root)
        component_frame_correction = apply_ue_component_frame_delta(
            visual_root, declaration
        )
        component.SetComponentTickEnabled(bEnabled=True)
        component.SetCastShadow(NewCastShadow=True)
        component.set_property_value(
            property_name="GlobalAnimRateScale", property_value=1.0
        )
        animation_paths = {
            "idle": declaration["idle_animation"],
            "walk": declaration["walking_animation"],
        }
        animations = {
            path: game.unreal_service.load_object(uclass="UAnimationAsset", name=path)
            for path in animation_paths.values()
        }
        lengths = {
            path: float(asset.GetPlayLength()) for path, asset in animations.items()
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in lengths.values()):
            raise RuntimeError(f"{actor_id} has an invalid UE animation length")
        runtimes[actor_id] = {
            "anchor": anchor,
            "anchor_root": anchor_root,
            "visual_actor": visual_actor,
            "visual_root": visual_root,
            "component": component,
            "animations": animations,
            "lengths": lengths,
            "current_animation": None,
            "component_frame_correction": component_frame_correction,
            "anatomical_basis_bones": declaration.get(
                "ue_anatomical_basis_bones"
            ),
            "hierarchy": {
                "status": "pass",
                "timeline_root_owner": "hidden_anchor_actor",
                "asset_frame_owner": "attached_visual_actor_root",
                "visual_parent_readback_matches_anchor": True,
            },
        }
    return runtimes


def _assert_suite_actor_binding_closure(suite: Mapping[str, Any]) -> None:
    reference_actor_ids: tuple[str, ...] | None = None
    binding_by_asset: dict[str, tuple[Any, ...]] = {}
    for scenario in suite["scenarios"]:
        declarations = scenario["plan"]["actors"]
        actor_ids = tuple(value["actor_id"] for value in declarations)
        if reference_actor_ids is None:
            reference_actor_ids = actor_ids
        elif actor_ids != reference_actor_ids:
            raise RuntimeError("Apartment UE actor-slot closure differs")
        for value in declarations:
            binding = (
                value["blueprint_class_path"],
                value["idle_animation"],
                value["walking_animation"],
                value["ue_component_frame_delta"],
                value.get("ue_anatomical_basis_bones"),
            )
            previous = binding_by_asset.setdefault(value["asset_id"], binding)
            if previous != binding:
                raise RuntimeError(
                    f"UE binding changes for asset {value['asset_id']!r}"
                )


def _apply_camera(camera: Any, camera_plan: Mapping[str, Any]) -> None:
    position = camera_plan["ue_position_cm"]
    camera.K2_SetActorLocationAndRotation(
        NewLocation={"X": position[0], "Y": position[1], "Z": position[2]},
        NewRotation={
            "Roll": 0.0,
            "Pitch": 0.0,
            "Yaw": camera_plan["ue_yaw_deg"],
        },
        bSweep=False,
        bTeleport=True,
    )


def _apply_actor_state(
    runtime: dict[str, Any], state: Mapping[str, Any], frame_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    anchor = runtime["anchor"]
    component = runtime["component"]
    animation_path = state["ue_animation"]
    if animation_path not in runtime["animations"]:
        raise RuntimeError(f"unloaded animation requested: {animation_path}")
    if runtime["current_animation"] != animation_path:
        component.PlayAnimation(
            NewAnimToPlay=runtime["animations"][animation_path], bLooping=True
        )
        runtime["current_animation"] = animation_path
    # Automatic UE advancement would be a second clock.  Stop and sample the
    # exact normalized Timeline phase for every formal frame.
    component.Stop()
    requested_seconds = animation_position_seconds(
        state["action_phase"], runtime["lengths"][animation_path]
    )
    component.SetPosition(InPos=requested_seconds, bFireNotifies=False)
    observed_seconds = float(component.GetPosition())
    animation_error = abs(observed_seconds - requested_seconds)
    if animation_error > ANIMATION_TOLERANCE_SECONDS:
        raise RuntimeError(
            f"animation phase readback failed at frame {frame_index}: "
            f"{observed_seconds} != {requested_seconds}"
        )
    position = state["translation_ue_cm"]
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
        "action_id": state["action_id"],
        "animation_path": animation_path,
        "requested_position_seconds": requested_seconds,
        "observed_position_seconds": observed_seconds,
        "absolute_error_seconds": animation_error,
    }


def _probe_media(
    path: Path,
    *,
    expected_width: int,
    expected_height: int,
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
    video = [
        value
        for value in payload.get("streams", ())
        if value.get("codec_type") == "video"
    ]
    audio = [
        value
        for value in payload.get("streams", ())
        if value.get("codec_type") == "audio"
    ]
    if len(video) != 1 or len(audio) != int(expect_audio):
        raise RuntimeError(f"media stream closure failed for {path}")
    stream = video[0]
    if (
        stream.get("codec_name") != "h264"
        or int(stream.get("width", -1)) != expected_width
        or int(stream.get("height", -1)) != expected_height
        or stream.get("avg_frame_rate") != f"{FPS}/1"
        or int(stream.get("nb_read_frames", -1)) != FRAME_COUNT
    ):
        raise RuntimeError(f"video readback failed for {path}: {stream}")
    if expect_audio and (
        audio[0].get("codec_name") != "aac"
        or int(audio[0].get("channels", -1)) != 2
        or int(audio[0].get("sample_rate", -1)) != 16_000
    ):
        raise RuntimeError(f"binaural readback failed for {path}: {audio[0]}")
    duration = float(payload.get("format", {}).get("duration", "nan"))
    if not math.isfinite(duration) or abs(duration - FRAME_COUNT / FPS) > 1.0 / FPS:
        raise RuntimeError(f"media duration readback failed for {path}: {duration}")
    return {
        "status": "pass",
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "width": expected_width,
        "height": expected_height,
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FPS,
        "duration_seconds": duration,
        "audio": (
            {
                "channels": 2,
                "sample_rate_hz": 16_000,
                "semantics": "authoritative Habitat-native binaural stream",
            }
            if expect_audio
            else None
        ),
        "audio_packet_sha256": (_audio_packet_sha256(path) if expect_audio else None),
    }


def _audio_packet_sha256(path: Path) -> str:
    """Hash the encoded audio packets without decoding or re-encoding them."""

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


def _resolve_bundle_path(
    bundle_root: Path, scenario: Mapping[str, Any], key: str
) -> Path:
    path = (bundle_root / scenario["authoritative_inputs"][key]).resolve()
    try:
        path.relative_to(bundle_root)
    except ValueError as exc:
        raise RuntimeError(f"scenario media path escapes bundle: {path}") from exc
    if not path.is_file():
        raise RuntimeError(f"scenario media is missing: {path}")
    return path


def _render_scenario(
    *,
    instance: Any,
    game: Any,
    camera: Any,
    capture: Any,
    runtimes: Mapping[str, dict[str, Any]],
    scenario: Mapping[str, Any],
    bundle_root: Path,
    output_root: Path,
    keep_frames: bool,
    video_encoder: str,
    encoder_gpu: int | None,
) -> dict[str, Any]:
    import cv2

    scenario_started = time.perf_counter()
    phase_wall_seconds: dict[str, float] = {}
    scenario_id = scenario["scenario_id"]
    scenario_root = output_root / scenario_id
    scenario_root.mkdir()
    frames_root = scenario_root / "frames"
    if keep_frames:
        frames_root.mkdir()
    ue_video = scenario_root / "ue_visual_only.mp4"
    plan = scenario["plan"]
    camera_plan = plan["camera"]

    # A short view-specific warmup follows the one suite-wide streaming warmup.
    phase_started = time.perf_counter()
    with instance.begin_frame():
        for state in plan["frames"][0]["actor_states"]:
            _apply_actor_state(runtimes[state["actor_id"]], state, 0)
        _apply_camera(camera, camera_plan)
    with instance.end_frame():
        pass
    instance.step(num_frames=CAMERA_WARMUP_FRAMES)
    phase_wall_seconds["camera_warmup"] = _elapsed_seconds(phase_started)

    phase_started = time.perf_counter()
    rawvideo_process = None
    if not keep_frames:
        rawvideo_process = subprocess.Popen(
            build_rawvideo_encode_command(
                output_path=ue_video,
                video_encoder=video_encoder,
                encoder_gpu=encoder_gpu,
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    actor_readbacks = {actor_id: [] for actor_id in runtimes}
    animation_readbacks = {actor_id: [] for actor_id in runtimes}
    actor_bounds = {actor_id: [] for actor_id in runtimes}
    visual_forward_readbacks = {actor_id: [] for actor_id in runtimes}
    camera_readbacks = []
    try:
        for frame_index, frame in enumerate(plan["frames"]):
            with instance.begin_frame():
                for state in frame["actor_states"]:
                    actor_id = state["actor_id"]
                    root_record, animation_record = _apply_actor_state(
                        runtimes[actor_id], state, frame_index
                    )
                    actor_readbacks[actor_id].append(root_record)
                    animation_readbacks[actor_id].append(animation_record)
                    if frame_index in ANATOMICAL_FORWARD_SAMPLE_FRAMES:
                        visual_forward_readbacks[actor_id].append(
                            _sample_anatomical_forward(
                                game,
                                runtimes[actor_id]["visual_actor"],
                                frame_index,
                                explicit_quadruped_bones=runtimes[actor_id].get(
                                    "anatomical_basis_bones"
                                ),
                            )
                        )
                _apply_camera(camera, camera_plan)
                camera_readbacks.append(_actor_readback(camera, frame_index))
            with instance.end_frame():
                image = _read_frame(capture).copy()
                for actor_id, runtime in runtimes.items():
                    actor_bounds[actor_id].append(
                        _actor_bounds_readback(runtime["visual_actor"], frame_index)
                    )
                if image.shape[:2] != (HEIGHT, WIDTH):
                    raise RuntimeError(
                        f"unexpected UE frame shape: {image.shape}"
                    )
                if keep_frames:
                    frame_path = frames_root / f"frame_{frame_index:04d}.png"
                    if not cv2.imwrite(str(frame_path), image):
                        raise RuntimeError(f"could not write UE frame: {frame_path}")
                else:
                    if rawvideo_process is None or rawvideo_process.stdin is None:
                        raise RuntimeError("rawvideo encoder stdin is unavailable")
                    rawvideo_process.stdin.write(image.tobytes(order="C"))
            if frame_index % FPS == 0:
                print(
                    f"[spear-apartment:{scenario_id}] frame "
                    f"{frame_index:02d}/{FRAME_COUNT - 1}",
                    flush=True,
                )
        if rawvideo_process is not None:
            assert rawvideo_process.stdin is not None
            rawvideo_process.stdin.close()
            stderr = (
                b""
                if rawvideo_process.stderr is None
                else rawvideo_process.stderr.read()
            )
            return_code = rawvideo_process.wait()
            if return_code != 0:
                raise RuntimeError(
                    "rawvideo FFmpeg encode failed: "
                    + stderr.decode("utf-8", errors="replace").strip()
                )
    except BaseException:
        if rawvideo_process is not None and rawvideo_process.poll() is None:
            rawvideo_process.kill()
            rawvideo_process.wait()
        raise
    capture_phase = (
        "visual_capture_and_png_write"
        if keep_frames
        else "visual_capture_and_rawvideo_encode"
    )
    phase_wall_seconds[capture_phase] = _elapsed_seconds(phase_started)

    phase_started = time.perf_counter()
    _write_json(
        scenario_root / "runtime_readbacks.json",
        {
            "actor_roots": actor_readbacks,
            "camera_root": camera_readbacks,
            "animation_phase": animation_readbacks,
            "visual_bounds": actor_bounds,
            "visual_anatomical_forward": visual_forward_readbacks,
        },
    )

    root_gate = summarize_root_readbacks(
        expected_frames=plan["frames"],
        actor_readbacks=actor_readbacks,
        camera_readbacks=camera_readbacks,
        camera_position_cm=camera_plan["ue_position_cm"],
        camera_yaw_deg=camera_plan["ue_yaw_deg"],
    )
    animation_gate = {}
    for actor_id, records in animation_readbacks.items():
        maximum_error = max(value["absolute_error_seconds"] for value in records)
        if maximum_error > ANIMATION_TOLERANCE_SECONDS:
            raise RuntimeError(f"{actor_id} animation phase gate failed")
        animation_gate[actor_id] = {
            "status": "pass",
            "maximum_absolute_error_seconds": maximum_error,
            "tolerance_seconds": ANIMATION_TOLERANCE_SECONDS,
            "action_ids": sorted({value["action_id"] for value in records}),
        }
    bounds_gate = summarize_actor_bounds(
        expected_frames=plan["frames"],
        actor_declarations=plan["actors"],
        actor_bounds=actor_bounds,
    )
    anatomical_forward_gate = summarize_anatomical_forward_readbacks(
        expected_frames=plan["frames"],
        visual_forward_readbacks=visual_forward_readbacks,
    )
    phase_wall_seconds["runtime_readback_and_gate"] = _elapsed_seconds(phase_started)

    phase_started = time.perf_counter()
    if keep_frames:
        subprocess.run(
            build_png_encode_command(
                frames_pattern=frames_root / "frame_%04d.png",
                output_path=ue_video,
                video_encoder=video_encoder,
                encoder_gpu=encoder_gpu,
            ),
            check=True,
        )
    phase_wall_seconds["rgb_video_encode"] = _elapsed_seconds(phase_started)

    authoritative_clean = _resolve_bundle_path(
        bundle_root, scenario, "authoritative_clean_binaural"
    )
    authoritative_diagnostic = _resolve_bundle_path(
        bundle_root, scenario, "authoritative_diagnostic_topdown"
    )
    clean_video = scenario_root / "ue_clean_binaural.mp4"
    phase_started = time.perf_counter()
    subprocess.run(
        build_clean_binaural_mux_command(
            ue_video_path=ue_video,
            authoritative_clean_path=authoritative_clean,
            output_path=clean_video,
        ),
        check=True,
    )
    phase_wall_seconds["rgb_binaural_mux"] = _elapsed_seconds(phase_started)

    topdown_visual = scenario_root / "ue_topdown_visual_only.mp4"
    phase_started = time.perf_counter()
    subprocess.run(
        build_topdown_visual_command(
            ue_video_path=ue_video,
            authoritative_diagnostic_path=authoritative_diagnostic,
            output_path=topdown_visual,
            video_encoder=video_encoder,
            encoder_gpu=encoder_gpu,
        ),
        check=True,
    )
    phase_wall_seconds["rgb_topdown_video_compose"] = _elapsed_seconds(phase_started)

    topdown_video = scenario_root / "ue_topdown_binaural.mp4"
    phase_started = time.perf_counter()
    subprocess.run(
        build_clean_binaural_mux_command(
            ue_video_path=topdown_visual,
            authoritative_clean_path=authoritative_clean,
            output_path=topdown_video,
        ),
        check=True,
    )
    phase_wall_seconds["topdown_binaural_mux"] = _elapsed_seconds(phase_started)
    phase_started = time.perf_counter()
    media = {
        "ue_visual_only": _probe_media(
            ue_video,
            expected_width=WIDTH,
            expected_height=HEIGHT,
            expect_audio=False,
        ),
        "ue_topdown_visual_only": _probe_media(
            topdown_visual,
            expected_width=1280,
            expected_height=480,
            expect_audio=False,
        ),
        "ue_clean_binaural": _probe_media(
            clean_video,
            expected_width=WIDTH,
            expected_height=HEIGHT,
            expect_audio=True,
        ),
        "ue_topdown_binaural": _probe_media(
            topdown_video,
            expected_width=1280,
            expected_height=480,
            expect_audio=True,
        ),
    }
    expected_audio_hashes = {
        "ue_clean_binaural": _audio_packet_sha256(authoritative_clean),
        "ue_topdown_binaural": _audio_packet_sha256(authoritative_clean),
    }
    for media_id, expected_hash in expected_audio_hashes.items():
        if media[media_id]["audio_packet_sha256"] != expected_hash:
            raise RuntimeError(
                f"{media_id} did not preserve the authoritative audio packets"
            )
    phase_wall_seconds["media_readback"] = _elapsed_seconds(phase_started)

    phase_started = time.perf_counter()
    if keep_frames:
        # ``--keep-frames`` is a diagnostic request, so retain the PNGs.
        pass
    elif frames_root.exists():
        shutil.rmtree(frames_root)
    phase_wall_seconds["frame_cleanup"] = _elapsed_seconds(phase_started)
    record = {
        "status": "pass",
        "scenario_id": scenario_id,
        "root_readback": root_gate,
        "animation_phase_readback": animation_gate,
        "asset_local_component_frame": {
            actor_id: runtime["component_frame_correction"]
            for actor_id, runtime in runtimes.items()
        },
        "runtime_actor_hierarchy": {
            actor_id: runtime["hierarchy"] for actor_id, runtime in runtimes.items()
        },
        "visual_bounds_readback": bounds_gate,
        "visual_anatomical_forward_readback": anatomical_forward_gate,
        "media": media,
        "audio_authority": {
            "status": "pass",
            "packet_copy_verified": True,
            "source_packet_sha256": expected_audio_hashes,
            "camera_fov_cutoff": False,
        },
        "authority": {
            "ue_pixels": "native SPEAR Apartment comparison visual",
            "audio": "copied from Habitat-native scenario",
            "topdown": "right panel copied from Habitat-native diagnostic",
            "source_logic_flags_metadata": "unchanged Habitat-native inputs",
        },
        "timing": {
            "schema": TIMING_SCHEMA,
            "clock": "time.perf_counter",
            "scope": "scenario render after actor setup and before actor teardown",
            "phase_wall_seconds": phase_wall_seconds,
            "video_encoder": video_encoder,
            "encoder_gpu": encoder_gpu,
            "frame_transport": (
                "png_sequence_then_encode"
                if keep_frames
                else "raw_bgr24_ffmpeg_pipe"
            ),
            "render_total_wall_seconds": _elapsed_seconds(scenario_started),
        },
    }
    _write_json(scenario_root / "evidence.json", record)
    return record


def _destroy_runtime_actors(
    instance: Any, runtimes: Mapping[str, dict[str, Any]]
) -> None:
    """Destroy one scenario's actors in child-before-parent order.

    UE's single-node animation instance is mutable state owned by the spawned
    SkeletalMeshComponent.  Reusing it across independent S0/S3/S4 episodes
    can make a later PlayAnimation transition dereference stale state.  Each
    formal scenario therefore owns a fresh runtime actor hierarchy.
    """

    with instance.begin_frame():
        for runtime in runtimes.values():
            runtime["visual_actor"].K2_DestroyActor()
        for runtime in runtimes.values():
            runtime["anchor"].K2_DestroyActor()
    with instance.end_frame():
        pass


def _configure_instance(args: argparse.Namespace) -> tuple[Any, Path]:
    spear_root = args.spear_root.resolve()
    executable = (
        spear_root
        / "cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh"
    )
    if not executable.is_file():
        raise RuntimeError(f"cooked SPEAR executable is missing: {executable}")
    examples = spear_root / "examples"
    if not examples.is_dir():
        raise RuntimeError(f"SPEAR examples directory is missing: {examples}")
    sys.path.insert(0, str(examples))
    from render_in_apartment import APARTMENT_MAP, parallel_instance_settings

    if APARTMENT_MAP != NATIVE_APARTMENT_MAP:
        raise RuntimeError(
            f"SPEAR Apartment map changed: {APARTMENT_MAP} != {NATIVE_APARTMENT_MAP}"
        )
    import spear

    settings = parallel_instance_settings(
        args.rpc_port, graphics_adapter=args.graphics_adapter
    )
    config = spear.get_config(user_config_files=[])
    config.defrost()
    config.SPEAR.LAUNCH_MODE = "game"
    config.SPEAR.INSTANCE.GAME_EXECUTABLE = str(executable)
    # The native Apartment is large and lives on comparatively slow storage.
    # A cold launch has already exceeded the upstream 120-second default while
    # legitimately loading this exact map, so the optional runner gives only
    # initialization (not individual RPC calls) a bounded ten-minute window.
    config.SPEAR.INSTANCE.INITIALIZE_CLIENT_MAX_TIME_SECONDS = (
        INITIALIZE_CLIENT_MAX_TIME_SECONDS
    )
    # A cold 1280x720 native frame can compile PSOs for longer than SPEAR's
    # interactive 2-second RPC default.  Keep the timeout finite but large
    # enough for that first real frame; subsequent readbacks remain checked.
    config.SPEAR.INSTANCE.CLIENT_INTERNAL_TIMEOUT_SECONDS = (
        CLIENT_INTERNAL_TIMEOUT_SECONDS
    )
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = NATIVE_APARTMENT_MAP
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.FIXED_DELTA_TIME = 1.0 / FPS
    config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT = settings["rpc_port"]
    config.SPEAR.INSTANCE.TEMP_DIR = settings["temp_dir"]
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.log = settings["log"]
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.renderoffscreen = None
    config.SP_CORE.SHARED_MEMORY_INITIAL_UNIQUE_ID = settings[
        "shared_memory_initial_unique_id"
    ]
    if settings["graphics_adapter"] is not None:
        config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.graphicsadapter = settings[
            "graphics_adapter"
        ]
    config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = "/etc/vulkan/icd.d/nvidia_icd.json"
    config.freeze()
    spear.configure_system(config=config)
    try:
        instance = spear.Instance(config=config)
    except BaseException:
        # Instance.__init__ launches UE before it waits for RPC.  If its
        # constructor itself fails, no Instance object exists for close().
        # Kill only the uniquely configured process tree for this RPC worker;
        # never use a broad pkill that could affect another agent's renderer.
        _cleanup_failed_constructor(
            executable=executable,
            temporary_directory=Path(settings["temp_dir"]),
        )
        raise
    return instance, spear_root


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
        joined = " ".join(str(value) for value in command)
        if executable_text not in joined and "SpearSim" not in joined:
            continue
        if not any(
            str(value).startswith("-sp-config-file=")
            and str(value).endswith(config_suffix)
            for value in command
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


def run(args: argparse.Namespace) -> Path:
    run_started = time.perf_counter()
    phase_wall_seconds: dict[str, float] = {}
    phase_started = time.perf_counter()
    bundle_root = args.bundle_root.resolve()
    if args.input_layout == "motion-pilot":
        default_scenarios: tuple[str, ...] | None = ("P0", "P1", "P2", "P3")
    elif args.input_layout == "m6x-canary":
        default_scenarios = ("S0", "S3", "S4")
    else:
        default_scenarios = None
    scenarios = (
        tuple(args.scenario)
        if args.scenario
        else default_scenarios
    )
    lighting_profile = load_apartment_lighting_profile(
        args.lighting_profiles, args.lighting_profile
    )
    suite_builder = {
        "motion-pilot": build_native_apartment_motion_pilot_suite,
        "m6x-canary": build_native_apartment_suite,
        "asset-bound-batch": build_native_apartment_asset_bound_suite,
    }[args.input_layout]
    suite = suite_builder(
        bundle_root, scenario_ids=scenarios, lighting_profile=lighting_profile
    )
    phase_wall_seconds["plan_compile"] = _elapsed_seconds(phase_started)
    _assert_suite_actor_binding_closure(suite)
    encoder_gpu = args.encoder_gpu
    if args.video_encoder == "h264_nvenc" and encoder_gpu is None:
        encoder_gpu = args.graphics_adapter

    output_root = args.output_dir.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output_root}")
    output_root.mkdir(parents=True)
    plan_path = output_root / "suite_execution_plan.json"
    _write_json(plan_path, suite)
    if args.dry_run:
        timing_path = output_root / "timing.json"
        _write_json(
            timing_path,
            {
                "schema": TIMING_SCHEMA,
                "status": "dry_run",
                "clock": "time.perf_counter",
                "phase_wall_seconds": phase_wall_seconds,
                "run_total_wall_seconds": _elapsed_seconds(run_started),
            },
        )
        print(f"SPEAR_APARTMENT_DRY_RUN_OK plan={plan_path}", flush=True)
        return plan_path

    phase_started = time.perf_counter()
    instance, spear_root = _configure_instance(args)
    phase_wall_seconds["runtime_initialize"] = _elapsed_seconds(phase_started)
    game = instance.get_game()
    scenario_records = []
    startup_evidence: dict[str, Any] = {}
    light_records: list[dict[str, Any]] = []
    runtime_close_seconds = 0.0
    try:
        phase_started = time.perf_counter()
        first = suite["scenarios"][0]
        with instance.begin_frame():
            camera, capture = _spawn_camera(game)
            light_records = _spawn_generated_lights(game, lighting_profile)
            _apply_camera(camera, first["plan"]["camera"])
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                bPaused=False
            )
        with instance.end_frame():
            pass
        phase_wall_seconds["shared_scene_setup"] = _elapsed_seconds(phase_started)
        # One suite-wide native texture/virtual-texture warmup.  The camera is
        # already at the authoritative listener pose before this begins.
        phase_started = time.perf_counter()
        instance.step(num_frames=STREAMING_WARMUP_FRAMES)
        phase_wall_seconds["shared_streaming_warmup"] = _elapsed_seconds(phase_started)
        for scenario in suite["scenarios"]:
            episode_started = time.perf_counter()
            phase_started = time.perf_counter()
            with instance.begin_frame():
                runtimes = _spawn_runtime_actors(game, scenario, spear_root)
                _apply_camera(camera, scenario["plan"]["camera"])
                for state in scenario["plan"]["frames"][0]["actor_states"]:
                    _apply_actor_state(runtimes[state["actor_id"]], state, 0)
            with instance.end_frame():
                pass
            actor_setup_seconds = _elapsed_seconds(phase_started)
            startup_evidence[scenario["scenario_id"]] = {
                actor_id: {
                    "asset_local_frame": runtime["component_frame_correction"],
                    "runtime_hierarchy": runtime["hierarchy"],
                }
                for actor_id, runtime in runtimes.items()
            }
            _write_json(
                output_root / "component_frame_startup_evidence.json",
                startup_evidence,
            )
            record: dict[str, Any] | None = None
            actor_teardown_seconds = 0.0
            try:
                record = _render_scenario(
                    instance=instance,
                    game=game,
                    camera=camera,
                    capture=capture,
                    runtimes=runtimes,
                    scenario=scenario,
                    bundle_root=bundle_root,
                    output_root=output_root,
                    keep_frames=args.keep_frames,
                    video_encoder=args.video_encoder,
                    encoder_gpu=encoder_gpu,
                )
            finally:
                phase_started = time.perf_counter()
                _destroy_runtime_actors(instance, runtimes)
                actor_teardown_seconds = _elapsed_seconds(phase_started)
            if record is None:
                raise RuntimeError("scenario render returned no evidence")
            record["timing"]["actor_setup_wall_seconds"] = actor_setup_seconds
            record["timing"]["actor_teardown_wall_seconds"] = actor_teardown_seconds
            record["timing"]["episode_total_wall_seconds"] = _elapsed_seconds(
                episode_started
            )
            _write_json(output_root / scenario["scenario_id"] / "evidence.json", record)
            scenario_records.append(record)
    finally:
        phase_started = time.perf_counter()
        instance.close(force=True)
        runtime_close_seconds = _elapsed_seconds(phase_started)
    phase_wall_seconds["runtime_close"] = runtime_close_seconds

    timing = {
        "schema": TIMING_SCHEMA,
        "status": "pass",
        "clock": "time.perf_counter",
        "measurement_scope": (
            "plan compilation, one native runtime launch, shared warmup, all "
            "requested episodes, media encoding/readback, and runtime close"
        ),
        "excluded_reused_costs": [
            (
                "Habitat/RLR binaural rendering and source metadata assembly "
                "already present in bundle_root"
            )
        ],
        "required_sample_outputs": list(REQUIRED_SAMPLE_OUTPUTS),
        "video_encoder": args.video_encoder,
        "encoder_gpu": encoder_gpu,
        "phase_wall_seconds": phase_wall_seconds,
        "scenario_timings": {
            record["scenario_id"]: record["timing"] for record in scenario_records
        },
        "run_total_wall_seconds": _elapsed_seconds(run_started),
    }
    timing_path = output_root / "timing.json"
    _write_json(timing_path, timing)

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "status": "pass",
        "backend_role": "comparison_visual",
        "native_map": NATIVE_APARTMENT_MAP,
        "lighting_profile": lighting_profile,
        "runtime_generated_lights": light_records,
        "native_scene_policy": {
            "map_geometry_mutated": False,
            "map_lighting_mutated": False,
            "native_map_lighting_retained": True,
            "additional_lights_spawned": bool(light_records),
            "additional_hdri_or_window_proxy_spawned": False,
            "actor_collision_enabled": False,
        },
        "runtime": {
            "rpc_port": args.rpc_port,
            "graphics_adapter": args.graphics_adapter,
            "resolution": [WIDTH, HEIGHT],
            "frame_rate_hz": FPS,
            "streaming_warmup_frames": STREAMING_WARMUP_FRAMES,
            "camera_warmup_frames_per_scenario": CAMERA_WARMUP_FRAMES,
        },
        "authority": suite["authority"],
        "scenarios": scenario_records,
        "timing": timing,
    }
    evidence_path = output_root / "evidence.json"
    _write_json(evidence_path, evidence)
    print(
        "SPEAR_APARTMENT_CANARY_OK "
        f"output={output_root} evidence={evidence_path} timing={timing_path}",
        flush=True,
    )
    return evidence_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--input-layout",
        choices=("m6x-canary", "motion-pilot", "asset-bound-batch"),
        default="m6x-canary",
        help=(
            "Input bundle layout; motion-pilot consumes P0--P3, while "
            "asset-bound-batch consumes generic source1/source2 episode IDs."
        ),
    )
    parser.add_argument("--spear-root", type=Path, default=DEFAULT_SPEAR_ROOT)
    parser.add_argument(
        "--lighting-profiles", type=Path, default=DEFAULT_LIGHTING_PROFILES
    )
    parser.add_argument(
        "--lighting-profile",
        help="Profile id; omitted uses default_profile_id from the JSON file.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        action="append",
        help="Repeat to render a subset; defaults depend on --input-layout.",
    )
    parser.add_argument("--rpc-port", type=int, default=39311)
    parser.add_argument("--graphics-adapter", type=int)
    parser.add_argument(
        "--video-encoder",
        choices=("libx264", "h264_nvenc"),
        default="libx264",
        help="H.264 encoder; h264_nvenc greatly reduces batch finalization time.",
    )
    parser.add_argument(
        "--encoder-gpu",
        type=int,
        help="NVENC GPU index; defaults to --graphics-adapter when available.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args(argv)
    if not 1024 <= args.rpc_port <= 65535:
        parser.error("--rpc-port must be in [1024,65535]")
    if args.graphics_adapter is not None and args.graphics_adapter < 0:
        parser.error("--graphics-adapter must be non-negative")
    if args.encoder_gpu is not None and args.encoder_gpu < 0:
        parser.error("--encoder-gpu must be non-negative")
    selected = set(args.scenario or ())
    allowed = {
        "motion-pilot": {"P0", "P1", "P2", "P3"},
        "m6x-canary": {"S0", "S3", "S4"},
    }.get(args.input_layout)
    if allowed is not None and selected - allowed:
        parser.error("--scenario values do not match --input-layout")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

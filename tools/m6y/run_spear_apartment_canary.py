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
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

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
    build_native_apartment_suite,
    build_png_encode_command,
    build_topdown_binaural_command,
    summarize_actor_bounds,
    summarize_root_readbacks,
)


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = REPOSITORY / "tmp/m6x/fixed_apartment_canary_20260720_02"
DEFAULT_SPEAR_ROOT = REPOSITORY.parent / "AVEngine/external/SPEAR"
CAMERA_BLUEPRINT = "/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C"
CAPTURE_COMPONENT_NAME = "DefaultSceneRoot.final_tone_curve_hdr_"
EVIDENCE_SCHEMA = "avengine_optional_spear_apartment_runtime_evidence_v1"
INITIALIZE_CLIENT_MAX_TIME_SECONDS = 600.0
CLIENT_INTERNAL_TIMEOUT_SECONDS = 60.0


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
        visual_actor.SetActorScale3D(
            NewScale3D={"X": 1.0, "Y": 1.0, "Z": 1.0}
        )
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
            path: game.unreal_service.load_object(
                uclass="UAnimationAsset", name=path
            )
            for path in animation_paths.values()
        }
        lengths = {path: float(asset.GetPlayLength()) for path, asset in animations.items()}
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
            "hierarchy": {
                "status": "pass",
                "timeline_root_owner": "hidden_anchor_actor",
                "asset_frame_owner": "attached_visual_actor_root",
                "visual_parent_readback_matches_anchor": True,
            },
        }
    return runtimes


def _assert_suite_actor_binding_closure(suite: Mapping[str, Any]) -> None:
    declarations = suite["scenarios"][0]["plan"]["actors"]
    reference = [
        (
            value["actor_id"],
            value["asset_id"],
            value["blueprint_class_path"],
            value["idle_animation"],
            value["walking_animation"],
            value["ue_component_frame_delta"],
        )
        for value in declarations
    ]
    for scenario in suite["scenarios"][1:]:
        current = [
            (
                value["actor_id"],
                value["asset_id"],
                value["blueprint_class_path"],
                value["idle_animation"],
                value["walking_animation"],
                value["ue_component_frame_delta"],
            )
            for value in scenario["plan"]["actors"]
        ]
        if current != reference:
            raise RuntimeError("S0/S3/S4 UE actor binding closure differs")


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
        value for value in payload.get("streams", ()) if value.get("codec_type") == "video"
    ]
    audio = [
        value for value in payload.get("streams", ()) if value.get("codec_type") == "audio"
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
        "audio_packet_sha256": (
            _audio_packet_sha256(path) if expect_audio else None
        ),
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
    camera: Any,
    capture: Any,
    runtimes: Mapping[str, dict[str, Any]],
    scenario: Mapping[str, Any],
    bundle_root: Path,
    output_root: Path,
    keep_frames: bool,
) -> dict[str, Any]:
    import cv2

    scenario_id = scenario["scenario_id"]
    scenario_root = output_root / scenario_id
    scenario_root.mkdir()
    frames_root = scenario_root / "frames"
    frames_root.mkdir()
    plan = scenario["plan"]
    camera_plan = plan["camera"]

    # A short view-specific warmup follows the one suite-wide streaming warmup.
    with instance.begin_frame():
        for state in plan["frames"][0]["actor_states"]:
            _apply_actor_state(runtimes[state["actor_id"]], state, 0)
        _apply_camera(camera, camera_plan)
    with instance.end_frame():
        pass
    instance.step(num_frames=CAMERA_WARMUP_FRAMES)

    actor_readbacks = {actor_id: [] for actor_id in runtimes}
    animation_readbacks = {actor_id: [] for actor_id in runtimes}
    actor_bounds = {actor_id: [] for actor_id in runtimes}
    camera_readbacks = []
    for frame_index, frame in enumerate(plan["frames"]):
        with instance.begin_frame():
            for state in frame["actor_states"]:
                actor_id = state["actor_id"]
                root_record, animation_record = _apply_actor_state(
                    runtimes[actor_id], state, frame_index
                )
                actor_readbacks[actor_id].append(root_record)
                animation_readbacks[actor_id].append(animation_record)
            _apply_camera(camera, camera_plan)
            camera_readbacks.append(_actor_readback(camera, frame_index))
        with instance.end_frame():
            image = _read_frame(capture).copy()
            for actor_id, runtime in runtimes.items():
                actor_bounds[actor_id].append(
                    _actor_bounds_readback(runtime["visual_actor"], frame_index)
                )
            frame_path = frames_root / f"frame_{frame_index:04d}.png"
            if image.shape[:2] != (HEIGHT, WIDTH) or not cv2.imwrite(
                str(frame_path), image
            ):
                raise RuntimeError(f"could not write UE frame: {frame_path}")
        if frame_index % FPS == 0:
            print(
                f"[spear-apartment:{scenario_id}] frame "
                f"{frame_index:02d}/{FRAME_COUNT - 1}",
                flush=True,
            )

    _write_json(
        scenario_root / "runtime_readbacks.json",
        {
            "actor_roots": actor_readbacks,
            "camera_root": camera_readbacks,
            "animation_phase": animation_readbacks,
            "visual_bounds": actor_bounds,
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

    ue_video = scenario_root / "ue_visual_only.mp4"
    subprocess.run(
        build_png_encode_command(
            frames_pattern=frames_root / "frame_%04d.png", output_path=ue_video
        ),
        check=True,
    )
    authoritative_clean = _resolve_bundle_path(
        bundle_root, scenario, "authoritative_clean_binaural"
    )
    authoritative_diagnostic = _resolve_bundle_path(
        bundle_root, scenario, "authoritative_diagnostic_topdown"
    )
    clean_video = scenario_root / "ue_clean_binaural.mp4"
    subprocess.run(
        build_clean_binaural_mux_command(
            ue_video_path=ue_video,
            authoritative_clean_path=authoritative_clean,
            output_path=clean_video,
        ),
        check=True,
    )
    topdown_video = scenario_root / "ue_topdown_binaural.mp4"
    subprocess.run(
        build_topdown_binaural_command(
            ue_video_path=ue_video,
            authoritative_diagnostic_path=authoritative_diagnostic,
            output_path=topdown_video,
        ),
        check=True,
    )
    media = {
        "ue_visual_only": _probe_media(
            ue_video,
            expected_width=WIDTH,
            expected_height=HEIGHT,
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
        "ue_topdown_binaural": _audio_packet_sha256(authoritative_diagnostic),
    }
    for media_id, expected_hash in expected_audio_hashes.items():
        if media[media_id]["audio_packet_sha256"] != expected_hash:
            raise RuntimeError(
                f"{media_id} did not preserve the authoritative audio packets"
            )
    if not keep_frames:
        shutil.rmtree(frames_root)
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
    }
    _write_json(scenario_root / "evidence.json", record)
    return record


def _destroy_runtime_actors(instance: Any, runtimes: Mapping[str, dict[str, Any]]) -> None:
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
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = (
        NATIVE_APARTMENT_MAP
    )
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
    config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = (
        "/etc/vulkan/icd.d/nvidia_icd.json"
    )
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


def _cleanup_failed_constructor(
    *, executable: Path, temporary_directory: Path
) -> None:
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
    bundle_root = args.bundle_root.resolve()
    scenarios = tuple(args.scenario or ("S0", "S3", "S4"))
    suite = build_native_apartment_suite(
        bundle_root, scenario_ids=scenarios
    )
    _assert_suite_actor_binding_closure(suite)

    output_root = args.output_dir.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output_root}")
    output_root.mkdir(parents=True)
    plan_path = output_root / "suite_execution_plan.json"
    _write_json(plan_path, suite)
    if args.dry_run:
        print(f"SPEAR_APARTMENT_DRY_RUN_OK plan={plan_path}", flush=True)
        return plan_path

    instance, spear_root = _configure_instance(args)
    game = instance.get_game()
    scenario_records = []
    startup_evidence: dict[str, Any] = {}
    try:
        first = suite["scenarios"][0]
        with instance.begin_frame():
            camera, capture = _spawn_camera(game)
            _apply_camera(camera, first["plan"]["camera"])
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                bPaused=False
            )
        with instance.end_frame():
            pass
        # One suite-wide native texture/virtual-texture warmup.  The camera is
        # already at the authoritative listener pose before this begins.
        instance.step(num_frames=STREAMING_WARMUP_FRAMES)
        for scenario in suite["scenarios"]:
            with instance.begin_frame():
                runtimes = _spawn_runtime_actors(game, scenario, spear_root)
                _apply_camera(camera, scenario["plan"]["camera"])
                for state in scenario["plan"]["frames"][0]["actor_states"]:
                    _apply_actor_state(runtimes[state["actor_id"]], state, 0)
            with instance.end_frame():
                pass
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
            try:
                scenario_records.append(
                    _render_scenario(
                        instance=instance,
                        camera=camera,
                        capture=capture,
                        runtimes=runtimes,
                        scenario=scenario,
                        bundle_root=bundle_root,
                        output_root=output_root,
                        keep_frames=args.keep_frames,
                    )
                )
            finally:
                _destroy_runtime_actors(instance, runtimes)
    finally:
        instance.close(force=True)

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "status": "pass",
        "backend_role": "comparison_visual",
        "native_map": NATIVE_APARTMENT_MAP,
        "native_scene_policy": {
            "map_geometry_mutated": False,
            "map_lighting_mutated": False,
            "additional_lights_spawned": False,
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
    }
    evidence_path = output_root / "evidence.json"
    _write_json(evidence_path, evidence)
    print(
        "SPEAR_APARTMENT_CANARY_OK "
        f"output={output_root} evidence={evidence_path}",
        flush=True,
    )
    return evidence_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--spear-root", type=Path, default=DEFAULT_SPEAR_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=("S0", "S3", "S4"),
        help="Repeat to render a subset; default is S0, S3, and S4.",
    )
    parser.add_argument("--rpc-port", type=int, default=39311)
    parser.add_argument("--graphics-adapter", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args(argv)
    if not 1024 <= args.rpc_port <= 65535:
        parser.error("--rpc-port must be in [1024,65535]")
    if args.graphics_adapter is not None and args.graphics_adapter < 0:
        parser.error("--graphics-adapter must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

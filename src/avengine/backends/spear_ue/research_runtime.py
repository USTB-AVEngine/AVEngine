"""Small shared primitives for external SPEAR visual research runs.

The functions here deliberately own only packaged-game launch, SceneCapture
life-cycle, and one frame transaction. They are factored from the current
Apartment runner's tested host/game path, without importing its M6/M7 bundle,
audio, qualification, or media-composition layers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from avengine.backends.spear_ue import client as _spear_client
from avengine.backends.spear_ue.launch import (
    parallel_instance_settings,
    validate_current_production_spear_executable,
)
from avengine.backends.spear_ue.rig_direction import select_skeletal_mesh_component


T = TypeVar("T")
_DEFAULT_VULKAN_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"


class SpearResearchRuntimeError(RuntimeError):
    """A minimal external SPEAR visual transaction cannot proceed safely."""


def cleanup_failed_constructor(*, executable: Path, temporary_directory: Path) -> None:
    """Terminate only the process tree configured for this failed launch."""

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
    _gone, alive = psutil.wait_procs(matched, timeout=10.0)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def launch_external_game_instance(
    *,
    spear_executable: Path,
    native_map: str,
    frame_rate_hz: int,
    rpc_port: int,
    graphics_adapter: int | None,
    initialize_client_max_time_seconds: float = 600.0,
    client_internal_timeout_seconds: float = 60.0,
    spear_client_module: Any = _spear_client,
) -> Any:
    """Launch one explicit non-Git packaged game with isolated IPC settings."""

    executable = validate_current_production_spear_executable(spear_executable)
    if not isinstance(native_map, str) or not native_map.startswith("/Game/"):
        raise SpearResearchRuntimeError("native_map must be a nonempty /Game path")
    if (
        isinstance(frame_rate_hz, bool)
        or not isinstance(frame_rate_hz, int)
        or frame_rate_hz < 1
    ):
        raise SpearResearchRuntimeError("frame_rate_hz must be a positive integer")
    if (
        not math.isfinite(initialize_client_max_time_seconds)
        or initialize_client_max_time_seconds <= 0.0
        or not math.isfinite(client_internal_timeout_seconds)
        or client_internal_timeout_seconds <= 0.0
    ):
        raise SpearResearchRuntimeError("SPEAR client timeouts must be positive")
    settings = parallel_instance_settings(rpc_port, graphics_adapter=graphics_adapter)
    config = spear_client_module.get_config(user_config_files=[])
    config.defrost()
    config.SPEAR.LAUNCH_MODE = "game"
    config.SPEAR.INSTANCE.GAME_EXECUTABLE = str(executable)
    config.SPEAR.INSTANCE.INITIALIZE_CLIENT_MAX_TIME_SECONDS = (
        float(initialize_client_max_time_seconds)
    )
    config.SPEAR.INSTANCE.CLIENT_INTERNAL_TIMEOUT_SECONDS = float(
        client_internal_timeout_seconds
    )
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = native_map
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.FIXED_DELTA_TIME = (
        1.0 / frame_rate_hz
    )
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
    config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = _DEFAULT_VULKAN_ICD
    config.freeze()
    spear_client_module.configure_system(config=config)
    try:
        return spear_client_module.Instance(config=config)
    except BaseException:
        cleanup_failed_constructor(
            executable=executable,
            temporary_directory=Path(settings["temp_dir"]),
        )
        raise


def spawn_scene_capture(
    game: Any,
    *,
    camera_blueprint: str,
    component_name: str,
    width: int,
    height: int,
    hfov_degrees: float,
) -> tuple[Any, Any]:
    """Spawn and initialize one RGB SceneCapture actor without audio setup."""

    if width < 1 or height < 1 or not 0.0 < hfov_degrees < 180.0:
        raise SpearResearchRuntimeError("SceneCapture dimensions or HFOV are invalid")
    camera_class = game.unreal_service.load_class(
        uclass="AActor", name=camera_blueprint
    )
    camera = game.unreal_service.spawn_actor(uclass=camera_class)
    capture = game.unreal_service.get_component_by_name(
        actor=camera,
        component_name=component_name,
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
    capture.set_property_value(property_name="FOVAngle", property_value=hfov_degrees)
    observed = float(capture.get_property_value(property_name="FOVAngle"))
    if abs(observed - hfov_degrees) > 1.0e-4:
        raise SpearResearchRuntimeError("SceneCapture HFOV readback differs")
    return camera, capture


def close_scene_capture(
    *, instance: Any, game: Any, camera: Any | None, capture: Any | None
) -> None:
    """Release SceneCapture shared-memory state before closing an Instance."""

    if camera is None and capture is None:
        return
    with instance.begin_frame():
        pass
    with instance.end_frame():
        try:
            if capture is not None:
                capture.terminate_sp_funcs()
        finally:
            try:
                if capture is not None:
                    capture.Terminate()
            finally:
                if camera is not None:
                    game.unreal_service.destroy_actor(actor=camera)


def read_rgb_bgr(capture: Any) -> np.ndarray:
    """Copy the current RGB plane before a later frame reuses shared memory."""

    pixels = np.asarray(capture.read_pixels()["arrays"]["data"][:, :, [0, 1, 2]])
    return np.ascontiguousarray(pixels.copy())


def _read_unreal_pose_vector(
    value: Any, *, keys: tuple[str, str, str], owner: str
) -> list[float]:
    """Parse one Unreal as_dict vector without trusting wrapper objects.

    Depending on the SPEAR/Python bridge version, ``as_dict`` may return the
    struct directly, under a case-insensitive ``ReturnValue`` key, or inside
    one other mapping wrapper.  Keep the bounded unwrapping behavior used by
    the older M6Y runner while still rejecting non-numeric and non-finite
    components.
    """

    expected = [key.casefold() for key in keys]
    current = value
    for _ in range(3):
        if not isinstance(current, Mapping):
            break
        lowered = {str(key).casefold(): item for key, item in current.items()}
        if all(key in lowered for key in expected):
            result: list[float] = []
            for key, original_key in zip(expected, keys):
                component = lowered[key]
                if isinstance(component, bool) or not isinstance(component, Real):
                    raise SpearResearchRuntimeError(
                        "%s readback component %s must be finite: %r"
                        % (owner, original_key, value)
                    )
                number = float(component)
                if not math.isfinite(number):
                    raise SpearResearchRuntimeError(
                        "%s readback component %s must be finite: %r"
                        % (owner, original_key, value)
                    )
                result.append(number)
            return result
        if "returnvalue" in lowered and isinstance(
            lowered["returnvalue"], Mapping
        ):
            current = lowered["returnvalue"]
            continue
        if len(current) == 1:
            candidate = next(iter(current.values()))
            if isinstance(candidate, Mapping):
                current = candidate
                continue
        break
    raise SpearResearchRuntimeError(
        "%s readback is missing components %s: %r" % (owner, keys, value)
    )


def read_actor_pose(actor: Any) -> dict[str, list[float]]:
    """Read a neutral Unreal actor pose using explicit as_dict calls.

    The returned vectors are world-space centimetres and degrees in Unreal's
    [Roll, Pitch, Yaw] order. Keeping this helper actor-neutral lets the
    same end-frame transaction read the camera and Timeline anchor actors.
    """

    actor_name = type(actor).__name__
    try:
        location = actor.K2_GetActorLocation(as_dict=True)
        rotation = actor.K2_GetActorRotation(as_dict=True)
    except Exception as error:
        raise SpearResearchRuntimeError(
            "%s pose readback failed: %s" % (actor_name, error)
        ) from error
    return {
        "location_cm": _read_unreal_pose_vector(
            location, keys=("X", "Y", "Z"), owner="%s location" % actor_name
        ),
        "rotation_deg": _read_unreal_pose_vector(
            rotation,
            keys=("Roll", "Pitch", "Yaw"),
            owner="%s rotation" % actor_name,
        ),
    }


def run_frame_transaction(
    instance: Any,
    *,
    apply: Callable[[], None],
    readback: Callable[[], T],
) -> T:
    """Apply UE mutations and consume its matching readback in one transaction."""

    with instance.begin_frame():
        apply()
    with instance.end_frame():
        return readback()


def warm_scene_capture_until_stable(
    instance: Any,
    capture: Any,
    *,
    maximum_frames: int = 60,
    minimum_frames: int = 4,
    required_consecutive_stable_frames: int = 3,
    mean_absolute_change_threshold: float = 0.8,
) -> dict[str, Any]:
    """Discard real SceneCapture frames until exposure and streaming settle."""

    counts = (maximum_frames, minimum_frames, required_consecutive_stable_frames)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in counts
    ):
        raise SpearResearchRuntimeError(
            "SceneCapture warmup frame counts must be positive integers"
        )
    if minimum_frames > maximum_frames:
        raise SpearResearchRuntimeError(
            "SceneCapture warmup minimum exceeds its maximum"
        )
    if (
        not math.isfinite(mean_absolute_change_threshold)
        or mean_absolute_change_threshold < 0.0
    ):
        raise SpearResearchRuntimeError(
            "SceneCapture warmup threshold must be finite and non-negative"
        )

    previous: np.ndarray | None = None
    consecutive_stable = 0
    for discarded_frame_count in range(1, maximum_frames + 1):
        current = run_frame_transaction(
            instance,
            apply=lambda: None,
            readback=lambda: read_rgb_bgr(capture),
        )
        if previous is not None:
            if current.shape != previous.shape:
                raise SpearResearchRuntimeError(
                    "SceneCapture warmup frame shape changed"
                )
            change = float(
                np.mean(
                    np.abs(
                        current.astype(np.int16, copy=False)
                        - previous.astype(np.int16, copy=False)
                    )
                )
            )
            if not math.isfinite(change):
                raise SpearResearchRuntimeError(
                    "SceneCapture warmup produced a non-finite change"
                )
            if (
                discarded_frame_count >= minimum_frames
                and change <= mean_absolute_change_threshold
            ):
                consecutive_stable += 1
            else:
                consecutive_stable = 0
            if consecutive_stable >= required_consecutive_stable_frames:
                return {
                    "status": "pass",
                    "discarded_frame_count": discarded_frame_count,
                    "minimum_frame_count": minimum_frames,
                    "maximum_frame_count": maximum_frames,
                    "required_consecutive_stable_frames": (
                        required_consecutive_stable_frames
                    ),
                    "mean_absolute_change_threshold": (
                        mean_absolute_change_threshold
                    ),
                    "final_mean_absolute_change": change,
                    "formal_frame_zero_follows_warmup": True,
                }
        previous = current
    raise SpearResearchRuntimeError(
        "SceneCapture did not stabilize before the warmup limit"
    )


def spawn_attached_visual_actor(
    game: Any,
    *,
    actor_id: str,
    blueprint_class_path: str,
    position_ue_cm: list[float],
    yaw_ue_degrees: float,
) -> dict[str, Any]:
    """Create the current safe Timeline-anchor / visual-child hierarchy."""

    if len(position_ue_cm) != 3 or not all(
        math.isfinite(value) for value in position_ue_cm
    ):
        raise SpearResearchRuntimeError(
            "actor position must contain three finite values"
        )
    if not math.isfinite(yaw_ue_degrees):
        raise SpearResearchRuntimeError("actor yaw must be finite")
    anchor = game.unreal_service.spawn_actor(
        uclass="AActor",
        spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
    )
    anchor_root = game.unreal_service.create_scene_component_for_actor(
        owner=anchor,
        scene_component_name=f"{actor_id}_research_timeline_anchor_root",
        uclass="USceneComponent",
    )
    anchor_root.SetMobility(NewMobility="Movable")
    anchor.SetActorEnableCollision(bNewActorEnableCollision=False)
    anchor.SetActorTickEnabled(bEnabled=True)
    anchor.K2_SetActorLocationAndRotation(
        NewLocation={
            "X": position_ue_cm[0],
            "Y": position_ue_cm[1],
            "Z": position_ue_cm[2],
        },
        NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": yaw_ue_degrees},
        bSweep=False,
        bTeleport=True,
    )
    blueprint = game.unreal_service.load_class(
        uclass="AActor", name=blueprint_class_path
    )
    visual_actor = game.unreal_service.spawn_actor(
        uclass=blueprint,
        location={
            "X": position_ue_cm[0],
            "Y": position_ue_cm[1],
            "Z": position_ue_cm[2],
        },
        spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
    )
    visual_actor.SetActorEnableCollision(bNewActorEnableCollision=False)
    visual_actor.SetActorTickEnabled(bEnabled=True)
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
    if (
        attached is not True
        or visual_root.GetAttachParent(as_handle=True) != anchor_root.uobject
    ):
        raise SpearResearchRuntimeError(
            "visual actor did not attach to its timeline anchor"
        )
    component = select_skeletal_mesh_component(
        unreal_service=game.unreal_service, actor=visual_actor
    )
    if component is None:
        raise SpearResearchRuntimeError(
            "spawned visual actor has no SkeletalMeshComponent"
        )
    component.SetComponentTickEnabled(bEnabled=True)
    component.SetCastShadow(NewCastShadow=True)
    return {
        "anchor": anchor,
        "anchor_root": anchor_root,
        "visual_actor": visual_actor,
        "visual_root": visual_root,
        "component": component,
    }


__all__ = [
    "SpearResearchRuntimeError",
    "cleanup_failed_constructor",
    "close_scene_capture",
    "launch_external_game_instance",
    "read_actor_pose",
    "read_rgb_bgr",
    "run_frame_transaction",
    "spawn_attached_visual_actor",
    "spawn_scene_capture",
    "warm_scene_capture_until_stable",
]

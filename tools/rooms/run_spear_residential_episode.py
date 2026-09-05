#!/usr/bin/env python3
"""Render one AVEngine residential visual episode through SPEAR/UE."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "tools/rooms"))

# The SPEAR client probes for its native extension at import time and freezes
# the answer in a module-level flag, so the extension directory has to be on
# sys.path before any avengine import below - argparse in main() is too late.
# This pre-scan mirrors the --spear-ext-dir argument argparse also declares.
if "--spear-ext-dir" in sys.argv:
    _spear_ext_dir = sys.argv[sys.argv.index("--spear-ext-dir") + 1]
    sys.path.insert(0, _spear_ext_dir)
from avengine.episode_clock import EpisodeClock
from avengine.backends.spear_ue.research_runtime import (
    apply_capture_exposure, attach_emitter_component, read_scene_component_pose,
)
from avengine.qa.pixel_visibility import compile_depth_pixel_visibility_truth  # noqa: E402
from avengine.optional_backends.residential_episode import TICKS_PER_FRAME  # noqa: E402

from avengine.optional_backends.spear_apartment import (  # noqa: E402
    ANIMATION_TOLERANCE_SECONDS,
    FRAME_COUNT,
    FPS,
    HEIGHT,
    WIDTH,
    build_png_encode_command,
    summarize_actor_bounds,
    summarize_root_readbacks,
)
from run_spear_apartment_canary import (  # noqa: E402
    CAMERA_BLUEPRINT,
    CAPTURE_COMPONENT_NAME,
    _actor_bounds_readback,
    _actor_readback,
    _apply_actor_state,
    _apply_camera,
    _apply_camera_state_and_readback,
    _destroy_runtime_actors,
    _read_frame,
    _spawn_camera,
    _spawn_runtime_actors,
)
from run_spear_kujiale_canary import (  # noqa: E402
    _configure_spear,
    _spawn_review_lights,
)


OBJECT_IDS_COMPONENT = "DefaultSceneRoot.sp_object_ids_uint8_"
DEPTH_COMPONENT = "DefaultSceneRoot.sp_depth_meters_"
TARGET_ONLY_BACKGROUND_DEPTH_M = 65504.0
DEPTH_ABSOLUTE_TOLERANCE_M = 0.01
DEPTH_RELATIVE_TOLERANCE = 0.002


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _is_overview_plan(episode: Mapping[str, Any]) -> bool:
    """Return true only for the explicit zero-actor room overview mode."""

    plan = episode.get("visual_plan")
    boundary = episode.get("planning_boundary")
    selection = plan.get("camera_selection") if isinstance(plan, Mapping) else None
    return (
        isinstance(boundary, Mapping)
        and boundary.get("overview_only") is True
        and isinstance(selection, Mapping)
        and selection.get("selection_mode") == "overview_geometry_only"
    )


def _resolve_plan_clock(episode: Mapping[str, Any]) -> EpisodeClock:
    """Use a declared clock; retained plans keep the original 75/15/16k default."""
    plan = episode["visual_plan"]
    raw = plan.get("clock", episode.get("clock"))
    clock = (EpisodeClock.from_mapping(raw) if raw is not None else
             EpisodeClock.from_values(frame_count=FRAME_COUNT, frame_rate_hz=FPS,
                                      sample_rate_hz=16000, compatibility="legacy_residential"))
    _require(clock.frame_rate_hz.denominator == 1 and 48000 % int(clock.frame_rate_hz) == 0,
             "residential clock must have an integral frame rate and 48k ticks per frame")
    _require(clock.sample_rate_hz == 16000, "residential audio route currently requires 16 kHz")
    frames = plan.get("frames")
    _require(isinstance(frames, list) and len(frames) == clock.frame_count,
             "visual frames differ from the declared clock")
    declarations = plan.get("actors")
    overview_only = _is_overview_plan(episode)
    _require(isinstance(declarations, list), "visual actors are missing")
    ids = [x.get("actor_id") if isinstance(x, Mapping) else None for x in declarations]
    _require(
        all(isinstance(x, str) and bool(x) for x in ids)
        and len(set(ids)) == len(ids)
        and (overview_only or bool(ids)),
        "visual actor IDs must be distinct nonempty strings",
    )
    if overview_only:
        _require(not ids, "overview geometry plans must have zero actors")
    ticks_per_frame = 48000 // int(clock.frame_rate_hz)
    for index, frame in enumerate(frames):
        _require(isinstance(frame, Mapping) and frame.get("frame_index") == index,
                 "visual frame order differs from the clock")
        if raw is not None or "pts_ticks" in frame:
            _require(frame.get("pts_ticks") == index * ticks_per_frame,
                     "visual frame ticks differ from the clock")
        states = frame.get("actor_states", [])
        _require(isinstance(states, list), "per-frame actor states must be a list")
        state_ids = [x.get("actor_id") for x in states if isinstance(x, Mapping)]
        _require(len(state_ids) == len(ids) and set(state_ids) == set(ids),
                 "per-frame actor closure differs from declarations")
        if overview_only:
            _require(not state_ids, "overview geometry frames must have zero actor states")
    return clock


def _apply_camera_for_frame(
    camera: Any,
    plan: Mapping[str, Any],
    frame_index: int,
    *,
    readback: bool,
) -> dict[str, Any] | None:
    """Apply one declared frame camera, preserving legacy fixed-camera plans."""

    frame = plan["frames"][frame_index]
    _require(isinstance(frame, Mapping), "visual plan frame is invalid")
    camera_state = frame.get("camera_state")
    if isinstance(camera_state, Mapping):
        if readback:
            return _apply_camera_state_and_readback(camera, camera_state, frame_index)
        _apply_camera(camera, camera_state)
        return None
    _apply_camera(camera, plan["camera"])
    if readback:
        return _actor_readback(camera, frame_index)
    return None


def _raw_object_ids(component: Any) -> np.ndarray:
    bgra = component.read_pixels()["arrays"]["data"].copy()
    return np.ascontiguousarray(bgra).view(np.uint32).reshape(
        bgra.shape[:2]
    ) & np.uint32(0x00FFFFFF)


def _depth_native(component: Any) -> np.ndarray:
    return component.read_pixels()["arrays"]["data"][:, :, 0].copy()


def _rgb_bgr(component: Any) -> np.ndarray:
    return component.read_pixels()["arrays"]["data"][:, :, :3].copy()


def _spawn_multimodal_camera(
    game: Any,
    *,
    horizontal_fov_deg: float,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> tuple[Any, dict[str, Any]]:
    """Spawn one BP_CameraSensor whose three passes share one actor pose."""

    game.segmentation_service.initialize()
    camera_class = game.unreal_service.load_class(
        uclass="AActor", name=CAMERA_BLUEPRINT
    )
    camera = game.unreal_service.spawn_actor(uclass=camera_class)
    components = {
        "rgb": game.unreal_service.get_component_by_name(
            actor=camera,
            component_name=CAPTURE_COMPONENT_NAME,
            uclass="USpSceneCaptureComponent2D",
        ),
        "object_ids": game.unreal_service.get_component_by_name(
            actor=camera,
            component_name=OBJECT_IDS_COMPONENT,
            uclass="USpSceneCaptureComponent2D",
        ),
        "depth": game.unreal_service.get_component_by_name(
            actor=camera,
            component_name=DEPTH_COMPONENT,
            uclass="USpSceneCaptureComponent2D",
        ),
    }
    viewport = game.rendering_service.get_current_viewport_desc()
    game.rendering_service.align_camera_with_viewport(
        camera_sensor=camera,
        camera_components=list(components.values()),
        viewport_desc=viewport,
        widths=width,
        heights=height,
    )
    for component in components.values():
        component.Initialize()
        component.initialize_sp_funcs()
        component.set_property_value(
            property_name="FOVAngle", property_value=horizontal_fov_deg
        )
        observed = float(component.get_property_value(property_name="FOVAngle"))
        _require(
            abs(observed - horizontal_fov_deg) <= 1.0e-4,
            "multimodal camera FOV readback drift",
        )
    return camera, components


def _derive_native_pixel_masks(
    *,
    normal_depths: list[np.ndarray],
    target_depths_by_actor: Mapping[str, list[np.ndarray]],
    semantic_ids_by_actor: Mapping[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Derive modal target visibility from same-camera normal/target depth."""

    _require(normal_depths, "normal depth pass is empty")
    _require(
        set(target_depths_by_actor) == set(semantic_ids_by_actor),
        "target actor IDs differ from semantic IDs",
    )
    frame_count = len(normal_depths)
    height, width = normal_depths[0].shape
    modal = np.zeros((frame_count, height, width), dtype=np.uint8)
    target_masks = {
        actor_id: np.zeros((frame_count, height, width), dtype=np.uint8)
        for actor_id in semantic_ids_by_actor
    }
    for frame_index, normal_value in enumerate(normal_depths):
        normal = np.asarray(normal_value, dtype=np.float32)
        best_residual = np.full((height, width), np.inf, dtype=np.float32)
        for actor_id in sorted(semantic_ids_by_actor):
            values = target_depths_by_actor[actor_id]
            _require(
                len(values) == frame_count,
                "target depth frame count differs from normal pass",
            )
            target = np.asarray(values[frame_index], dtype=np.float32)
            _require(target.shape == (height, width), "target depth shape drift")
            footprint = target < TARGET_ONLY_BACKGROUND_DEPTH_M
            target_masks[actor_id][frame_index][footprint] = semantic_ids_by_actor[
                actor_id
            ]
            residual = np.abs(normal - target)
            tolerance = DEPTH_ABSOLUTE_TOLERANCE_M + DEPTH_RELATIVE_TOLERANCE * target
            visible = footprint & (residual <= tolerance)
            wins = visible & (residual < best_residual)
            modal[frame_index][wins] = semantic_ids_by_actor[actor_id]
            best_residual[wins] = residual[wins]
    return modal, target_masks


def _maximum_multimodal_readback_drift(
    normal_readbacks: list[Mapping[str, Any]],
    target_readbacks: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, float]:
    maximum_location_cm = 0.0
    maximum_rotation_deg = 0.0
    for records in target_readbacks.values():
        _require(
            len(records) == len(normal_readbacks),
            "target-only runtime readback count differs from normal pass",
        )
        for frame_index, (normal, target) in enumerate(
            zip(normal_readbacks, records, strict=True)
        ):
            for owner in ("camera", "actors"):
                normal_value = normal[owner]
                target_value = target[owner]
                if owner == "actors":
                    _require(
                        set(normal_value) == set(target_value),
                        "target-only actor IDs differ from normal pass",
                    )
                    pairs = zip(
                        (normal_value[key] for key in sorted(normal_value)),
                        (target_value[key] for key in sorted(target_value)),
                        strict=True,
                    )
                else:
                    pairs = ((normal_value, target_value),)
                for left, right in pairs:
                    _require(
                        left.get("frame_index") == frame_index
                        and right.get("frame_index") == frame_index,
                        "target-only frame index drift",
                    )
                    maximum_location_cm = max(
                        maximum_location_cm,
                        max(
                            abs(float(a) - float(b))
                            for a, b in zip(
                                left["location_cm"], right["location_cm"], strict=True
                            )
                        ),
                    )
                    maximum_rotation_deg = max(
                        maximum_rotation_deg,
                        max(
                            abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)
                            for a, b in zip(
                                left["rotation_deg"],
                                right["rotation_deg"],
                                strict=True,
                            )
                        ),
                    )
    _require(maximum_location_cm <= 1.0e-4, "target pass location drift")
    _require(maximum_rotation_deg <= 1.0e-4, "target pass rotation drift")
    return {
        "maximum_location_drift_cm": maximum_location_cm,
        "maximum_rotation_drift_deg": maximum_rotation_deg,
    }


def _finalize_overview_native_pixel_artifacts(
    *,
    output: Path,
    normal_depths: list[np.ndarray],
    normal_object_ids: list[np.ndarray],
    normal_readbacks: list[Mapping[str, Any]],
    frame_count: int,
) -> dict[str, Any]:
    """Persist normal scene depth/readbacks for a zero-actor room overview."""

    _require(
        len(normal_depths) == len(normal_object_ids) == len(normal_readbacks) == frame_count,
        "overview native-pixel frame count drift",
    )
    _require(normal_depths, "overview native-pixel capture has no frames")
    height, width = normal_depths[0].shape
    _require(
        all(
            array.shape == (height, width)
            and np.issubdtype(array.dtype, np.floating)
            and np.isfinite(array).all()
            and (array > 0.0).all()
            for array in normal_depths
        ),
        "overview metric-depth arrays are invalid",
    )
    _require(
        all(array.shape == (height, width) and array.dtype == np.uint32 for array in normal_object_ids),
        "overview object-ID arrays are invalid",
    )
    depth_path = output / "metric_depth_native.npz"
    object_ids_path = output / "normal_object_ids_uint32.npz"
    readbacks_path = output / "native_pixel_runtime_readbacks.json"
    np.savez_compressed(depth_path, normal_depth_m=np.stack(normal_depths))
    np.savez_compressed(object_ids_path, normal_object_ids=np.stack(normal_object_ids))
    _write(
        readbacks_path,
        {
            "schema": "avengine_overview_native_metric_runtime_readbacks_v1",
            "status": "pass",
            "overview_only": True,
            "normal": normal_readbacks,
            "target_only": {},
            "actor_ids": [],
        },
    )
    return {
        "status": "pass",
        "overview_only": True,
        "actor_ids": [],
        "frame_count": frame_count,
        "resolution_hw": [height, width],
        "normal_object_id_dtype": str(np.stack(normal_object_ids).dtype),
        "artifacts": {
            "metric_depth": str(depth_path),
            "normal_object_ids": str(object_ids_path),
            "runtime_readbacks": str(readbacks_path),
        },
    }


def _finalize_native_pixel_artifacts(
    *,
    output: Path,
    episode: Mapping[str, Any],
    normal_depths: list[np.ndarray],
    normal_object_ids: list[np.ndarray],
    target_depths_by_actor: Mapping[str, list[np.ndarray]],
    normal_readbacks: list[Mapping[str, Any]],
    target_readbacks: Mapping[str, list[Mapping[str, Any]]],
    camera_pose_ids: list[str] | None = None,
    frame_count: int = FRAME_COUNT,
) -> dict[str, Any]:
    """Persist one completed same-camera native-pixel capture without hashes."""

    plan = episode.get("visual_plan")
    _require(isinstance(plan, Mapping), "episode visual plan is missing")
    actors = plan.get("actors")
    frames = plan.get("frames")
    _require(
        isinstance(actors, list) and isinstance(frames, list),
        "episode native-pixel authorities are incomplete",
    )
    actor_ids = [
        str(actor.get("actor_id")) for actor in actors if isinstance(actor, Mapping)
    ]
    if not actor_ids:
        _require(_is_overview_plan(episode), "empty native-pixel actor closure is overview-only")
        return _finalize_overview_native_pixel_artifacts(
            output=output,
            normal_depths=normal_depths,
            normal_object_ids=normal_object_ids,
            normal_readbacks=normal_readbacks,
            frame_count=frame_count,
        )
    _require(
        len(actor_ids) == len(actors) and len(set(actor_ids)) == len(actor_ids),
        "native-pixel capture requires distinct actor IDs",
    )
    _require(
        len(frames) == frame_count,
        "native-pixel visual-plan frame authority differs from the declared clock",
    )
    _require(
        len(normal_depths)
        == len(normal_object_ids)
        == len(normal_readbacks)
        == frame_count,
        "normal native-pixel frame count drift",
    )
    _require(
        set(target_depths_by_actor) == set(target_readbacks) == set(actor_ids),
        "target-only actor authority drift",
    )
    semantic_ids = {actor_id: index + 1 for index, actor_id in enumerate(actor_ids)}
    height, width = normal_depths[0].shape
    _require(
        all(
            array.shape == (height, width)
            and np.issubdtype(array.dtype, np.floating)
            and np.isfinite(array).all()
            and (array > 0.0).all()
            for array in normal_depths
        ),
        "normal metric-depth arrays are invalid",
    )
    _require(
        all(
            len(arrays) == frame_count
            and all(
                array.shape == (height, width)
                and np.issubdtype(array.dtype, np.floating)
                and np.isfinite(array).all()
                and (array > 0.0).all()
                for array in arrays
            )
            for arrays in target_depths_by_actor.values()
        ),
        "target-only metric-depth arrays are invalid",
    )
    _require(
        all(
            array.shape == (height, width) and array.dtype == np.uint32
            for array in normal_object_ids
        ),
        "normal object-ID arrays are invalid",
    )
    if camera_pose_ids is None:
        timeline = episode.get("timeline")
        _require(isinstance(timeline, Mapping), "episode timeline is missing")
        timeline_frames = timeline.get("frames")
        _require(
            isinstance(timeline_frames, list) and len(timeline_frames) == frame_count,
            "native-pixel timeline frame authority differs from the declared clock",
        )
        camera_pose_ids = []
        for frame_index, frame in enumerate(timeline_frames):
            _require(isinstance(frame, Mapping), "timeline frame is invalid")
            poses = frame.get("view_pose_hashes")
            _require(
                isinstance(poses, Mapping) and isinstance(poses.get("view0"), str),
                f"timeline frame {frame_index} lacks view0 pose hash",
            )
            camera_pose_ids.append(poses["view0"])
    else:
        _require(
            len(camera_pose_ids) == frame_count
            and all(isinstance(value, str) and value for value in camera_pose_ids),
            "explicit camera pose IDs must contain 75 non-empty strings",
        )
        camera_pose_ids = list(camera_pose_ids)
    common_context = {
        "renderer_backend": str(episode.get("renderer_backend", "spear_unreal_native_kujiale")),
        "rgb_renderer_backend": str(episode.get("renderer_backend", "spear_unreal_native_kujiale")),
        "camera_contract_id": "avengine_kujiale_native_spear_bp_camera_sensor_v1",
        "semantic_id_namespace": "avengine_kujiale_native_metric_depth_instances_v1",
        "resolution_hw": [height, width],
        "frame_indices": list(range(frame_count)),
        "camera_pose_ids": camera_pose_ids,
    }
    truth = compile_depth_pixel_visibility_truth(
        normal_depth_m_frames=normal_depths,
        target_only_depth_m_frames_by_instance=target_depths_by_actor,
        semantic_ids_by_instance=semantic_ids,
        normal_context={"pass_kind": "modal_scene", **common_context},
        target_only_contexts_by_instance={
            actor_id: {
                "pass_kind": "target_only",
                "target_instance_id": actor_id,
                **common_context,
            }
            for actor_id in actor_ids
        },
        target_only_background_depth_m=TARGET_ONLY_BACKGROUND_DEPTH_M,
        absolute_tolerance_m=DEPTH_ABSOLUTE_TOLERANCE_M,
        relative_tolerance=DEPTH_RELATIVE_TOLERANCE,
    )
    modal_masks, target_masks = _derive_native_pixel_masks(
        normal_depths=normal_depths,
        target_depths_by_actor=target_depths_by_actor,
        semantic_ids_by_actor=semantic_ids,
    )
    alignment = _maximum_multimodal_readback_drift(normal_readbacks, target_readbacks)
    normal_depth_array = np.stack(normal_depths)
    normal_object_ids_array = np.stack(normal_object_ids)
    target_depth_arrays = {
        actor_id: np.stack(target_depths_by_actor[actor_id]) for actor_id in actor_ids
    }
    depth_path = output / "metric_depth_native.npz"
    object_ids_path = output / "normal_object_ids_uint32.npz"
    masks_path = output / "native_pixel_masks_depth_authority_v1.npz"
    truth_path = output / "pixel_visibility_truth.json"
    readbacks_path = output / "native_pixel_runtime_readbacks.json"
    np.savez_compressed(
        depth_path,
        normal_depth_m=normal_depth_array,
        **{
            f"target_only_{actor_id}_depth_m": target_depth_arrays[actor_id]
            for actor_id in actor_ids
        },
    )
    np.savez_compressed(object_ids_path, normal_object_ids=normal_object_ids_array)
    np.savez_compressed(
        masks_path,
        depth_derived_modal_semantic=modal_masks,
        **{
            f"modal_visible_{actor_id}": modal_masks == semantic_ids[actor_id]
            for actor_id in actor_ids
        },
        **{f"target_only_{actor_id}": target_masks[actor_id] for actor_id in actor_ids},
    )
    _write(truth_path, truth)
    _write(
        readbacks_path,
        {
            "schema": "avengine_kujiale_native_multimodal_runtime_readbacks_v1",
            "status": "pass",
            "normal": normal_readbacks,
            "target_only": target_readbacks,
            "alignment": alignment,
        },
    )
    return {
        "status": "pass",
        "authority": truth["authority"],
        "semantic_ids_by_actor": semantic_ids,
        "frame_count": frame_count,
        "resolution_hw": [height, width],
        "normal_object_id_dtype": str(normal_object_ids_array.dtype),
        "alignment": alignment,
        "artifacts": {
            "metric_depth": str(depth_path),
            "normal_object_ids": str(object_ids_path),
            "pixel_masks": str(masks_path),
            "pixel_truth": str(truth_path),
            "runtime_readbacks": str(readbacks_path),
        },
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _audio_claim_boundary(episode_root: Path, episode: Mapping[str, Any]) -> Any:
    """Prefer the actual semantic-cache claim while retaining legacy evidence."""

    audio_evidence_path = episode_root / "audio_evidence.json"
    if not audio_evidence_path.exists():
        return episode["acoustic_proxy"]
    audio_evidence = _load(audio_evidence_path)
    audio_mode = audio_evidence.get("audio_mode")
    if audio_mode is None or audio_mode == "review_proxy":
        return episode["acoustic_proxy"]
    if audio_mode != "semantic_cached_rlr":
        raise RuntimeError(f"unsupported audio_mode in audio evidence: {audio_mode!r}")
    claim_boundary = audio_evidence.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise RuntimeError("semantic cached RLR audio evidence lacks a claim_boundary")
    return claim_boundary


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _research_root_readback_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Select ordinary numeric root-readback results for a plain receipt."""

    allowed = (
        "status",
        "maximum_position_error_cm",
        "maximum_yaw_error_deg",
        "per_frame_camera_state",
    )

    def one(record: Mapping[str, Any]) -> dict[str, Any]:
        return {key: record[key] for key in allowed if key in record}

    if "status" in value:
        return one(value)
    return {
        owner: one(record)
        for owner, record in value.items()
        if isinstance(record, Mapping)
    }


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _summarize_camera_full_rotation(
    *,
    plan: Mapping[str, Any],
    readbacks: list[Mapping[str, Any]],
    tolerance_deg: float = 1.0e-4,
    frame_count: int = FRAME_COUNT,
) -> dict[str, Any]:
    """Check optional roll/pitch as well as yaw against UE readback.

    The shared Apartment helper historically checks only yaw because its
    camera plans are level.  Residential look-at cameras can have a real
    downward pitch; silently dropping it changes framing even when position
    and yaw still pass.
    """

    frames = plan.get("frames")
    _require(
        isinstance(frames, list) and len(frames) == len(readbacks) == frame_count,
        "full camera rotation readback requires complete plan and runtime frames",
    )
    maximum = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    per_frame_states = all(
        isinstance(frame, Mapping) and isinstance(frame.get("camera_state"), Mapping)
        for frame in frames
    )
    for frame_index, (frame, observed) in enumerate(
        zip(frames, readbacks, strict=True)
    ):
        expected = frame["camera_state"] if per_frame_states else plan["camera"]
        observed_rotation = observed.get("rotation_deg")
        _require(
            isinstance(observed_rotation, list) and len(observed_rotation) == 3,
            f"camera rotation readback is invalid at frame {frame_index}",
        )
        expected_rotation = (
            float(expected.get("ue_roll_deg", 0.0)),
            float(expected.get("ue_pitch_deg", 0.0)),
            float(expected["ue_yaw_deg"]),
        )
        for axis_index, axis_name in enumerate(("roll", "pitch", "yaw")):
            error = abs(
                _wrap_degrees(
                    float(observed_rotation[axis_index])
                    - expected_rotation[axis_index]
                )
            )
            maximum[axis_name] = max(maximum[axis_name], error)
    _require(
        max(maximum.values()) <= tolerance_deg,
        f"UE camera full rotation readback drifted: {maximum}",
    )
    return {
        "status": "pass",
        "per_frame_camera_state": per_frame_states,
        "maximum_roll_error_deg": maximum["roll"],
        "maximum_pitch_error_deg": maximum["pitch"],
        "maximum_yaw_error_deg": maximum["yaw"],
    }


def _light_plan(episode: Mapping[str, Any]) -> dict[str, Any]:
    lights = []
    for raw in episode.get("review_lights", []):
        position = raw["position_xyz_m"]
        record = {
            "light_id": raw["light_id"],
            "position_m": list(position),
            "position_ue_cm": [100.0 * float(item) for item in position],
            "intensity_lumens": float(raw["intensity_lumens"]),
            "attenuation_radius_cm": 100.0 * float(raw["attenuation_radius_m"]),
            "temperature_kelvin": float(raw["temperature_kelvin"]),
            "source_radius_cm": 100.0 * float(raw.get("source_radius_m", 0.0)),
            "soft_source_radius_cm": 100.0
            * float(raw.get("soft_source_radius_m", 0.0)),
        }
        source_prim = raw.get("source_prim")
        if isinstance(source_prim, str) and source_prim:
            record["source_prim"] = source_prim
        lights.append(record)
    return {"review_lights": lights}


def _probe(
    path: Path, *, width: int, height: int, expect_audio: bool,
    frame_count: int = FRAME_COUNT, frame_rate_hz: int = FPS,
) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate,nb_read_frames,sample_rate,channels:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    video = [item for item in value["streams"] if item["codec_type"] == "video"]
    audio = [item for item in value["streams"] if item["codec_type"] == "audio"]
    if len(video) != 1 or len(audio) != int(expect_audio):
        raise RuntimeError(f"media stream closure failed: {path}")
    v = video[0]
    if (
        int(v["width"]) != width
        or int(v["height"]) != height
        or v["avg_frame_rate"] != f"{frame_rate_hz}/1"
        or int(v["nb_read_frames"]) != frame_count
    ):
        raise RuntimeError(f"video readback failed: {v}")
    if expect_audio and (
        int(audio[0]["channels"]) != 2 or int(audio[0]["sample_rate"]) != 16_000
    ):
        raise RuntimeError(f"audio readback failed: {audio[0]}")
    duration = float(value["format"]["duration"])
    if not math.isfinite(duration) or abs(duration - frame_count / frame_rate_hz) > 1.0 / frame_rate_hz:
        raise RuntimeError(f"duration readback failed: {duration}")
    return {
        "status": "pass",
        "path": str(path),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "frame_rate_hz": frame_rate_hz,
        "duration_seconds": duration,
        "audio": "binaural_left_right" if expect_audio else None,
    }


def _mux_clean(video: Path, audio: Path, output: Path, *, frame_count: int = FRAME_COUNT) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-frames:v",
            str(frame_count),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def _mux_topdown(video: Path, topdown: Path, audio: Path, output: Path, *, frame_count: int = FRAME_COUNT) -> None:
    graph = (
        "[0:v]scale=640:360:flags=lanczos,pad=640:480:0:60:color=black[ue];"
        "[1:v]scale=640:480:flags=lanczos[top];"
        "[ue][top]hstack=inputs=2[video]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-i",
            str(topdown),
            "-i",
            str(audio),
            "-filter_complex",
            graph,
            "-map",
            "[video]",
            "-map",
            "2:a:0",
            "-frames:v",
            str(frame_count),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import cv2

    episode_root = args.episode_root.expanduser().resolve()
    episode = _load(episode_root / "episode_plan.json")
    plan = episode["visual_plan"]
    clock = _resolve_plan_clock(episode)
    frame_count = clock.frame_count
    frame_rate_hz = int(clock.frame_rate_hz)
    ticks_per_frame = 48000 // frame_rate_hz
    raw_actors = plan.get("actors")
    _require(isinstance(raw_actors, list), "visual plan actors are missing")
    actor_ids = [
        str(actor.get("actor_id")) for actor in raw_actors if isinstance(actor, Mapping)
    ]
    overview_only = _is_overview_plan(episode)
    _require(
        len(set(actor_ids)) == len(actor_ids)
        and (overview_only or bool(actor_ids)),
        "residential capture requires distinct plan actor IDs",
    )
    if overview_only:
        _require(not actor_ids, "overview geometry plans must have zero actors")
    native_multimodal = bool(getattr(args, "native_multimodal", False))
    visual_only_research = bool(getattr(args, "visual_only_research", False))
    if overview_only:
        _require(
            visual_only_research,
            "overview geometry capture requires --visual-only-research",
        )
    capture_width = int(getattr(args, "width", WIDTH))
    capture_height = int(getattr(args, "height", HEIGHT))
    _require(
        capture_width > 0
        and capture_height > 0
        and capture_width % 2 == 0
        and capture_height % 2 == 0,
        "capture width and height must be positive even integers",
    )
    exposure_bias_ev = getattr(args, "exposure_bias_ev", None)
    if exposure_bias_ev is None:
        exposure_bias_ev = plan["camera"].get("exposure_bias_ev")
    if exposure_bias_ev is not None:
        _require(
            not isinstance(exposure_bias_ev, bool)
            and isinstance(exposure_bias_ev, (int, float))
            and math.isfinite(exposure_bias_ev),
            "camera exposure bias must be finite",
        )
    exposure_readback: dict[str, Any] = {"status": "not_requested"}
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    frames_dir = output / "frames"
    frames_dir.mkdir()
    _write(output / "visual_plan.json", plan)

    config_args = argparse.Namespace(**vars(args))
    config_args.frame_rate_hz = frame_rate_hz
    config_plan = {"map_path": episode["scene"]["map_path"]}
    instance = _configure_spear(config_args, config_plan)
    game = instance.get_game()
    runtimes: dict[str, dict[str, Any]] = {}
    light_records: list[dict[str, Any]] = []
    stage_actor_count = 0
    actor_readbacks = {actor_id: [] for actor_id in actor_ids}
    animation_readbacks = {actor_id: [] for actor_id in actor_ids}
    actor_bounds = {actor_id: [] for actor_id in actor_ids}
    camera_readbacks: list[dict[str, Any]] = []
    emitter_components: dict[str, Any] = {}
    emitter_readbacks: dict[str, list[dict[str, Any]]] = {}
    normal_depths: list[np.ndarray] = []
    normal_object_ids: list[np.ndarray] = []
    normal_multimodal_readbacks: list[Mapping[str, Any]] = []
    target_depths_by_actor: dict[str, list[np.ndarray]] = {}
    target_readbacks: dict[str, list[Mapping[str, Any]]] = {}
    components: dict[str, Any] | None = None
    try:
        with instance.begin_frame():
            if native_multimodal:
                camera, components = _spawn_multimodal_camera(
                    game,
                    horizontal_fov_deg=float(plan["camera"]["horizontal_fov_deg"]),
                    width=capture_width,
                    height=capture_height,
                )
                capture = components["rgb"]
            else:
                camera, capture = _spawn_camera(
                    game,
                    width=capture_width,
                    height=capture_height,
                    hfov_degrees=float(plan["camera"]["horizontal_fov_deg"]),
                )
            exposure_readback = apply_capture_exposure(capture, bias_ev=exposure_bias_ev)
            capture.set_property_value(
                property_name="FOVAngle",
                property_value=float(plan["camera"]["horizontal_fov_deg"]),
            )
            observed_fov = float(capture.get_property_value(property_name="FOVAngle"))
            if abs(observed_fov - float(plan["camera"]["horizontal_fov_deg"])) > 1.0e-4:
                raise RuntimeError(f"camera HFOV readback failed: {observed_fov}")
            _apply_camera(camera, plan["camera"])
            if actor_ids:
                runtimes = _spawn_runtime_actors(game, {"plan": plan})
                for declaration in plan["actors"]:
                    offset = declaration.get("emitter_local_ue_cm")
                    if offset is not None:
                        actor_id = declaration["actor_id"]
                        emitter_components[actor_id] = attach_emitter_component(
                            game, actor_id=actor_id,
                            anchor_root=runtimes[actor_id]["anchor"].K2_GetRootComponent(),
                            emitter_local_ue_cm=offset)
                        emitter_readbacks[actor_id] = []
                for state in plan["frames"][0]["actor_states"]:
                    _apply_actor_state(runtimes[state["actor_id"]], state, 0)
            light_records = _spawn_review_lights(game, _light_plan(episode))
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                bPaused=False
            )
        with instance.end_frame():
            pass
        instance.step(num_frames=args.streaming_warmup_frames)

        if native_multimodal:
            _require(components is not None, "native multimodal components are missing")
            with instance.begin_frame():
                game.segmentation_service.initialize()
                components["depth"].PrimitiveRenderMode = "PRM_RenderScenePrimitives"
                components["depth"].ShowOnlyActors = []
                for state in plan["frames"][0]["actor_states"]:
                    _apply_actor_state(runtimes[state["actor_id"]], state, 0)
                _apply_camera_for_frame(camera, plan, 0, readback=False)
            with instance.end_frame():
                pass
            instance.step(num_frames=2)

        with instance.begin_frame():
            stage_actor_count = len(
                game.unreal_service.find_actors_by_class(uclass="AUsdStageActor")
            )
        with instance.end_frame():
            pass
        if stage_actor_count != args.expected_stage_actor_count:
            raise RuntimeError(
                f"expected {args.expected_stage_actor_count} UsdStageActor(s), got {stage_actor_count}"
            )

        for frame_index, frame in enumerate(plan["frames"]):
            native_frame_readback: dict[str, Any] | None = None
            with instance.begin_frame():
                native_actor_readbacks: dict[str, Any] = {}
                for state in frame["actor_states"]:
                    actor_id = state["actor_id"]
                    root, animation = _apply_actor_state(
                        runtimes[actor_id], state, frame_index
                    )
                    actor_readbacks[actor_id].append(root)
                    animation_readbacks[actor_id].append(animation)
                    if native_multimodal:
                        native_actor_readbacks[actor_id] = root
                camera_readback = _apply_camera_for_frame(
                    camera, plan, frame_index, readback=True
                )
                assert camera_readback is not None
                camera_readbacks.append(camera_readback)
                for actor_id, component in emitter_components.items():
                    emitter_readbacks[actor_id].append({
                        "frame_index": frame_index, **read_scene_component_pose(component)})
                if native_multimodal:
                    native_frame_readback = {
                        "camera": camera_readback,
                        "actors": native_actor_readbacks,
                    }
            with instance.end_frame():
                image = (
                    _rgb_bgr(components["rgb"]).copy()
                    if native_multimodal
                    else _read_frame(capture).copy()
                )
                if native_multimodal:
                    normal_depths.append(_depth_native(components["depth"]))
                    normal_object_ids.append(_raw_object_ids(components["object_ids"]))
                for actor_id, runtime in runtimes.items():
                    actor_bounds[actor_id].append(
                        _actor_bounds_readback(runtime["visual_actor"], frame_index)
                    )
            if native_multimodal:
                _require(
                    native_frame_readback is not None, "native readback is missing"
                )
                normal_multimodal_readbacks.append(native_frame_readback)
            frame_path = frames_dir / f"frame_{frame_index:04d}.png"
            if image.shape[:2] != (capture_height, capture_width) or not cv2.imwrite(
                str(frame_path), image
            ):
                raise RuntimeError(f"could not write frame: {frame_path}")
            if frame_index % frame_rate_hz == 0:
                print(
                    f"[residential:{episode['scene']['scene_id']}] frame {frame_index:02d}/{frame_count - 1}",
                    flush=True,
                )

        if native_multimodal:
            _require(components is not None, "native multimodal components are missing")
            for target_actor_id in actor_ids:
                target_depths_by_actor[target_actor_id] = []
                target_readbacks[target_actor_id] = []
                with instance.begin_frame():
                    manager = game.segmentation_service.proxy_component_manager
                    manager.SetAllowedActors(
                        AllowedActors=[runtimes[target_actor_id]["visual_actor"]]
                    )
                    game.segmentation_service.initialize()
                    components["depth"].PrimitiveRenderMode = "PRM_UseShowOnlyList"
                    components["depth"].ShowOnlyActors = [
                        runtimes[target_actor_id]["visual_actor"]
                    ]
                    for state in plan["frames"][0]["actor_states"]:
                        _apply_actor_state(runtimes[state["actor_id"]], state, 0)
                    _apply_camera_for_frame(camera, plan, 0, readback=False)
                with instance.end_frame():
                    pass
                instance.step(num_frames=2)

                for frame_index, frame in enumerate(plan["frames"]):
                    target_frame_readback: dict[str, Any] = {"actors": {}}
                    with instance.begin_frame():
                        for state in frame["actor_states"]:
                            actor_id = state["actor_id"]
                            root, _ = _apply_actor_state(
                                runtimes[actor_id], state, frame_index
                            )
                            target_frame_readback["actors"][actor_id] = root
                        camera_readback = _apply_camera_for_frame(
                            camera, plan, frame_index, readback=True
                        )
                        assert camera_readback is not None
                        target_frame_readback["camera"] = camera_readback
                    with instance.end_frame():
                        target_depth = _depth_native(components["depth"])
                    target_depths_by_actor[target_actor_id].append(target_depth)
                    target_readbacks[target_actor_id].append(target_frame_readback)
    finally:
        if runtimes:
            try:
                _destroy_runtime_actors(instance, runtimes)
            except Exception as exc:
                print(f"warning: actor cleanup failed: {exc}", file=sys.stderr)
        instance.close(force=True)

    native_pixel: dict[str, Any] | None = None
    if native_multimodal:
        native_pixel = _finalize_native_pixel_artifacts(
            output=output,
            episode=episode,
            frame_count=frame_count,
            normal_depths=normal_depths,
            normal_object_ids=normal_object_ids,
            target_depths_by_actor=target_depths_by_actor,
            normal_readbacks=normal_multimodal_readbacks,
            target_readbacks=target_readbacks,
            camera_pose_ids=(
                [f"current_visual_frame_{index:04d}" for index in range(frame_count)]
                if visual_only_research
                else None
            ),
        )

    root_gate = summarize_root_readbacks(
        expected_frames=plan["frames"],
        actor_readbacks=actor_readbacks,
        camera_readbacks=camera_readbacks,
        camera_position_cm=plan["camera"]["ue_position_cm"],
        camera_yaw_deg=plan["camera"]["ue_yaw_deg"],
        frame_count=frame_count,
    )
    bounds_gate = summarize_actor_bounds(
        expected_frames=plan["frames"],
        actor_declarations=plan["actors"],
        actor_bounds=actor_bounds,
    )
    animation_gate = {}
    for actor_id, records in animation_readbacks.items():
        maximum = max(item["absolute_error_seconds"] for item in records)
        if maximum > ANIMATION_TOLERANCE_SECONDS:
            raise RuntimeError(f"{actor_id} animation phase readback failed")
        animation_gate[actor_id] = {
            "status": "pass",
            "action_ids": sorted({item["action_id"] for item in records}),
            "maximum_absolute_error_seconds": maximum,
        }
    camera_rotation_gate = _summarize_camera_full_rotation(
        plan=plan,
        readbacks=camera_readbacks,
        frame_count=frame_count,
    )

    _write(output / "frame_readbacks.json", {
        "clock": clock.to_dict(), "camera": camera_readbacks,
        "actors": actor_readbacks, "animations": animation_readbacks,
        "bounds": actor_bounds, "emitters": emitter_readbacks,
    })
    visual = output / "ue_visual_only.mp4"
    subprocess.run(
        build_png_encode_command(
            frames_pattern=frames_dir / "frame_%04d.png", output_path=visual,
            frame_count=frame_count, frame_rate_hz=frame_rate_hz,
        ),
        check=True,
    )
    if visual_only_research:
        visual_probe = _probe(
            visual,
            width=capture_width,
            height=capture_height,
            expect_audio=False,
            frame_count=frame_count, frame_rate_hz=frame_rate_hz,
        )
        if not args.keep_frames:
            shutil.rmtree(frames_dir)
        receipt = {
            "status": "research_only",
            "research_only": True,
            "episode_counted": False,
            "formal_dataset_count": 0,
            "qualification": False,
            "qualification_claim": False,
            "clock": {
                "frame_count": frame_count,
                "frame_rate_hz": frame_rate_hz,
                "ticks_per_frame": ticks_per_frame,
            },
            "backend_role": episode["visual_plan"]["backend_role"],
            "scene": episode["scene"],
            "stage_actor_count": stage_actor_count,
            "runtime_review_lights": light_records,
            "capture_exposure_readback": exposure_readback,
            "visual_lighting": episode["visual_lighting"],
            "root_readback": _research_root_readback_summary(root_gate),
            "camera_full_rotation_readback": camera_rotation_gate,
            "animation_phase_readback": animation_gate,
            "visual_bounds_readback": bounds_gate,
            "media": {"ue_visual_only": visual_probe},
            "audio": {"status": "not_requested"},
            "rlr": {"status": "not_requested"},
        }
        if native_pixel is not None:
            receipt["native_pixel"] = native_pixel
        _write(output / "research_receipt.json", receipt)
        print(visual, flush=True)
        return receipt

    audio = episode_root / "audio/mixture.wav"
    topdown = episode_root / "topdown_only.mp4"
    clean = output / "ue_clean_binaural.mp4"
    combined = output / "ue_topdown_binaural.mp4"
    _mux_clean(visual, audio, clean, frame_count=frame_count)
    _mux_topdown(visual, topdown, audio, combined, frame_count=frame_count)
    media = {
        "ue_visual_only": _probe(
            visual,
            width=capture_width,
            height=capture_height,
            expect_audio=False,
            frame_count=frame_count, frame_rate_hz=frame_rate_hz,
        ),
        "ue_clean_binaural": _probe(
            clean,
            width=capture_width,
            height=capture_height,
            expect_audio=True,
            frame_count=frame_count, frame_rate_hz=frame_rate_hz,
        ),
        "ue_topdown_binaural": _probe(
            combined, width=1280, height=480, expect_audio=True,
            frame_count=frame_count, frame_rate_hz=frame_rate_hz
        ),
    }
    if not args.keep_frames:
        shutil.rmtree(frames_dir)
    evidence = {
        "schema": "avengine_optional_spear_residential_episode_evidence_v1",
        "status": "pass",
        "backend_role": episode["visual_plan"]["backend_role"],
        "scene": episode["scene"],
        "stage_actor_count": stage_actor_count,
        "runtime_review_lights": light_records,
            "capture_exposure_readback": exposure_readback,
        "visual_lighting": episode["visual_lighting"],
        "root_readback": root_gate,
        "camera_full_rotation_readback": camera_rotation_gate,
        "animation_phase_readback": animation_gate,
        "visual_bounds_readback": bounds_gate,
        "media": media,
        "authority": {
            "ue_pixels": (f"optional room {episode['visual_plan']['backend_role']}"),
            "timeline_source_logic_audio_topdown_metadata": "AVEngine",
            "backend_replanned_route": False,
            "audio_camera_fov_cutoff": False,
            "source_center_gate": "center_only_not_body_volume",
        },
        "audio_claim_boundary": _audio_claim_boundary(episode_root, episode),
    }
    if native_pixel is not None:
        evidence.update(
            {
                "status_scope": "native_capture_execution",
                "research_only": True,
                "qualification_claim": False,
                "formal_dataset_count": 0,
                "episode_promotion": False,
                "native_pixel": native_pixel,
            }
        )
    _write(output / "evidence.json", evidence)
    print(combined, flush=True)
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument(
        "--spear-ext-dir",
        type=Path,
        help=(
            "directory holding AVEngine's compiled avengine_spear_ext; the "
            "SPEAR client refuses to start without it, so a Studio task must "
            "be able to name it explicitly instead of inheriting a shell"
        ),
    )
    parser.add_argument("--uproject", type=Path, required=True)
    parser.add_argument("--unreal-editor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rpc-port", type=int, default=39379)
    parser.add_argument("--graphics-adapter", type=int, default=0)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--exposure-bias-ev", type=float,
                        help="optional native RGB exposure compensation in stops")
    parser.add_argument("--streaming-warmup-frames", type=int, default=180)
    parser.add_argument("--expected-stage-actor-count", type=int, default=1)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument(
        "--native-multimodal",
        action="store_true",
        help="capture native RGB/depth/object-ID and two target-only depth passes",
    )
    parser.add_argument(
        "--visual-only-research",
        action="store_true",
        help="encode current UE pixels without reading, claiming or muxing audio",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.spear_ext_dir is not None and not args.spear_ext_dir.is_dir():
        raise SystemExit(f"--spear-ext-dir is not a directory: {args.spear_ext_dir}")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

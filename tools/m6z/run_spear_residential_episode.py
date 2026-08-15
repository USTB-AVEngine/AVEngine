#!/usr/bin/env python3
"""Render one AVEngine residential human+Beagle episode through SPEAR/UE."""

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
sys.path.insert(0, str(REPOSITORY / "tools/m6y"))
from avengine.qa.pixel_visibility import compile_depth_pixel_visibility_truth  # noqa: E402

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
    game: Any, *, horizontal_fov_deg: float
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
        widths=WIDTH,
        heights=HEIGHT,
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


def _finalize_native_pixel_artifacts(
    *,
    output: Path,
    episode: Mapping[str, Any],
    normal_depths: list[np.ndarray],
    normal_object_ids: list[np.ndarray],
    target_depths_by_actor: Mapping[str, list[np.ndarray]],
    normal_readbacks: list[Mapping[str, Any]],
    target_readbacks: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Persist one completed same-camera native-pixel capture without hashes."""

    plan = episode.get("visual_plan")
    timeline = episode.get("timeline")
    _require(isinstance(plan, Mapping), "episode visual plan is missing")
    _require(isinstance(timeline, Mapping), "episode timeline is missing")
    actors = plan.get("actors")
    frames = plan.get("frames")
    timeline_frames = timeline.get("frames")
    _require(
        isinstance(actors, list)
        and isinstance(frames, list)
        and isinstance(timeline_frames, list),
        "episode native-pixel authorities are incomplete",
    )
    actor_ids = [
        str(actor.get("actor_id")) for actor in actors if isinstance(actor, Mapping)
    ]
    _require(
        len(actor_ids) == 2 and len(set(actor_ids)) == 2,
        "native-pixel capture requires exactly two distinct actor IDs",
    )
    _require(
        len(frames) == FRAME_COUNT == len(timeline_frames),
        "native-pixel frame authority is not full75",
    )
    _require(
        len(normal_depths)
        == len(normal_object_ids)
        == len(normal_readbacks)
        == FRAME_COUNT,
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
            len(arrays) == FRAME_COUNT
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
    camera_pose_ids = []
    for frame_index, frame in enumerate(timeline_frames):
        _require(isinstance(frame, Mapping), "timeline frame is invalid")
        poses = frame.get("view_pose_hashes")
        _require(
            isinstance(poses, Mapping) and isinstance(poses.get("view0"), str),
            f"timeline frame {frame_index} lacks view0 pose hash",
        )
        camera_pose_ids.append(poses["view0"])
    common_context = {
        "renderer_backend": "spear_unreal_native_kujiale",
        "rgb_renderer_backend": "spear_unreal_native_kujiale",
        "camera_contract_id": "avengine_kujiale_native_spear_bp_camera_sensor_v1",
        "semantic_id_namespace": "avengine_kujiale_native_metric_depth_instances_v1",
        "resolution_hw": [height, width],
        "frame_indices": list(range(FRAME_COUNT)),
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
        "frame_count": FRAME_COUNT,
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


def _light_plan(episode: Mapping[str, Any]) -> dict[str, Any]:
    lights = []
    for raw in episode.get("review_lights", []):
        position = raw["position_xyz_m"]
        lights.append(
            {
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
        )
    return {"review_lights": lights}


def _probe(
    path: Path, *, width: int, height: int, expect_audio: bool
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
        or v["avg_frame_rate"] != f"{FPS}/1"
        or int(v["nb_read_frames"]) != FRAME_COUNT
    ):
        raise RuntimeError(f"video readback failed: {v}")
    if expect_audio and (
        int(audio[0]["channels"]) != 2 or int(audio[0]["sample_rate"]) != 16_000
    ):
        raise RuntimeError(f"audio readback failed: {audio[0]}")
    duration = float(value["format"]["duration"])
    if not math.isfinite(duration) or abs(duration - 5.0) > 1.0 / FPS:
        raise RuntimeError(f"duration readback failed: {duration}")
    return {
        "status": "pass",
        "path": str(path),
        "width": width,
        "height": height,
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FPS,
        "duration_seconds": duration,
        "audio": "binaural_left_right" if expect_audio else None,
    }


def _mux_clean(video: Path, audio: Path, output: Path) -> None:
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
            str(FRAME_COUNT),
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


def _mux_topdown(video: Path, topdown: Path, audio: Path, output: Path) -> None:
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
            str(FRAME_COUNT),
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
    raw_actors = plan.get("actors")
    _require(isinstance(raw_actors, list), "visual plan actors are missing")
    actor_ids = [
        str(actor.get("actor_id")) for actor in raw_actors if isinstance(actor, Mapping)
    ]
    _require(
        len(actor_ids) == 2 and len(set(actor_ids)) == 2,
        "residential capture requires exactly two distinct plan actor IDs",
    )
    native_multimodal = bool(getattr(args, "native_multimodal", False))
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    frames_dir = output / "frames"
    frames_dir.mkdir()
    _write(output / "visual_plan.json", plan)

    config_args = argparse.Namespace(**vars(args))
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
                )
                capture = components["rgb"]
            else:
                camera, capture = _spawn_camera(game)
            capture.set_property_value(
                property_name="FOVAngle",
                property_value=float(plan["camera"]["horizontal_fov_deg"]),
            )
            observed_fov = float(capture.get_property_value(property_name="FOVAngle"))
            if abs(observed_fov - float(plan["camera"]["horizontal_fov_deg"])) > 1.0e-4:
                raise RuntimeError(f"camera HFOV readback failed: {observed_fov}")
            _apply_camera(camera, plan["camera"])
            runtimes = _spawn_runtime_actors(
                game, {"plan": plan}, args.spear_root.expanduser().resolve()
            )
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
            if image.shape[:2] != (HEIGHT, WIDTH) or not cv2.imwrite(
                str(frame_path), image
            ):
                raise RuntimeError(f"could not write frame: {frame_path}")
            if frame_index % FPS == 0:
                print(
                    f"[residential:{episode['scene']['scene_id']}] frame {frame_index:02d}/74",
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
            normal_depths=normal_depths,
            normal_object_ids=normal_object_ids,
            target_depths_by_actor=target_depths_by_actor,
            normal_readbacks=normal_multimodal_readbacks,
            target_readbacks=target_readbacks,
        )

    root_gate = summarize_root_readbacks(
        expected_frames=plan["frames"],
        actor_readbacks=actor_readbacks,
        camera_readbacks=camera_readbacks,
        camera_position_cm=plan["camera"]["ue_position_cm"],
        camera_yaw_deg=plan["camera"]["ue_yaw_deg"],
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

    visual = output / "ue_visual_only.mp4"
    subprocess.run(
        build_png_encode_command(
            frames_pattern=frames_dir / "frame_%04d.png", output_path=visual
        ),
        check=True,
    )
    audio = episode_root / "audio/mixture.wav"
    topdown = episode_root / "topdown_only.mp4"
    clean = output / "ue_clean_binaural.mp4"
    combined = output / "ue_topdown_binaural.mp4"
    _mux_clean(visual, audio, clean)
    _mux_topdown(visual, topdown, audio, combined)
    media = {
        "ue_visual_only": _probe(visual, width=1280, height=720, expect_audio=False),
        "ue_clean_binaural": _probe(clean, width=1280, height=720, expect_audio=True),
        "ue_topdown_binaural": _probe(
            combined, width=1280, height=480, expect_audio=True
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
        "visual_lighting": episode["visual_lighting"],
        "root_readback": root_gate,
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
    parser.add_argument("--spear-root", type=Path, required=True)
    parser.add_argument("--uproject", type=Path, required=True)
    parser.add_argument("--unreal-editor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rpc-port", type=int, default=39379)
    parser.add_argument("--graphics-adapter", type=int, default=0)
    parser.add_argument("--streaming-warmup-frames", type=int, default=180)
    parser.add_argument("--expected-stage-actor-count", type=int, default=1)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument(
        "--native-multimodal",
        action="store_true",
        help="capture native RGB/depth/object-ID and two target-only depth passes",
    )
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

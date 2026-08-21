#!/usr/bin/env python3
"""Capture an imported-MP3D strict two-human SPEAR review Episode.

This is intentionally a visual research probe.  It fresh-loads and reads back
the complete cooked room, then captures normal RGB/metric depth and one
target-only metric-depth pass for each human through one ``BP_CameraSensor``.
It does not render audio and it never increments a formal-data denominator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
TOOLS_QA = REPOSITORY / "tools/qa"
if str(TOOLS_QA) not in sys.path:
    sys.path.insert(0, str(TOOLS_QA))
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.spear_unreal_capabilities import (  # noqa: E402
    live_handle,
    set_numeric_property_with_readback,
)

from spear_imported_glb_room_adapter import (  # noqa: E402
    CAMERA_BLUEPRINT,
    DEPTH_COMPONENT,
    ENTRY_MAP,
    EXPECTED_STATIC_MESH_COUNT,
    OBJECT_ID_COMPONENT,
    RGB_COMPONENT,
    destroy_scene_meshes,
    load_json_object,
    spawn_review_lighting,
    spawn_scene_meshes_with_readback,
    validate_room_adapter,
)

SCHEMA = "avengine_spear_imported_glb_strict_two_human_capture_v1"
SUITE_SCHEMA = "avengine_optional_spear_imported_glb_suite_v1"
FRAME_COUNT = 75
FPS = 15
WIDTH = 1280
HEIGHT = 720
TARGET_ONLY_BACKGROUND_DEPTH_M = 65504.0
CAPTURE_PHASES = (
    "preconnect",
    "post-entry",
    "mesh",
    "lighting",
    "camera",
    "actor",
    "capture",
    "artifact_finalize",
    "complete",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


class CapturePhaseJournal:
    """Append-only phase and failure evidence for the diagnostic capture."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.current_phase = "before_output_materialization"
        self.sequence = 0

    def enter(self, phase: str) -> Path:
        _require(self.output.is_dir(), "capture output is not materialized")
        _require(
            phase in CAPTURE_PHASES,
            f"unknown capture phase: {phase}",
        )
        self.current_phase = phase
        path = self.output / (f"capture_phase_{self.sequence:02d}_{phase}.json")
        _write_json_exclusive(
            path,
            {
                "schema": "avengine_mp3d_f15_capture_phase_v1",
                "status": "entered",
                "phase": phase,
                "sequence": self.sequence,
                "recorded_at_utc": _utc_now(),
                "qualification_claim": False,
                "formal_dataset_count": 0,
            },
        )
        self.sequence += 1
        return path

    def record_failure(self, exc: BaseException) -> Path | None:
        if not self.output.is_dir():
            return None
        path = self.output / "capture_failure.json"
        _write_json_exclusive(
            path,
            {
                "schema": "avengine_mp3d_f15_capture_failure_v1",
                "status": "failed",
                "phase": self.current_phase,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "recorded_at_utc": _utc_now(),
                "qualification_claim": False,
                "formal_dataset_count": 0,
            },
        )
        return path


def _load_native_capture_backend() -> ModuleType:
    path = TOOLS_QA / "capture_spear_native_pixel_episode.py"
    _require(path.is_file(), f"existing native capture backend is missing: {path}")
    name = "avengine_mp3d_existing_native_capture_backend"
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _vector3(value: Any, *, owner: str) -> list[float]:
    _require(
        isinstance(value, list)
        and len(value) == 3
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value
        ),
        f"{owner} must contain three finite numbers",
    )
    return [float(item) for item in value]


def _habitat_to_ue_cm(position: Sequence[float]) -> list[float]:
    return [100.0 * position[0], 100.0 * position[2], 100.0 * position[1]]


def _maximum_error(left: Sequence[float], right: Sequence[float]) -> float:
    _require(len(left) == len(right), "vector length drift")
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def validate_capture_contract(
    suite: Mapping[str, Any],
    *,
    scenario_id: str,
    room_adapter: Mapping[str, Any],
    requested_frame_indices: Sequence[int] | None,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    """Validate the full CPU-visible contract before starting packaged SPEAR."""

    validate_room_adapter(room_adapter)
    _require(suite.get("schema") == SUITE_SCHEMA, "suite schema drift")
    _require(suite.get("native_map") == ENTRY_MAP, "suite must use the Entry map")
    _require(suite.get("qualification_claim") is False, "formal claim forbidden")
    _require(suite.get("formal_dataset_count") == 0, "formal count must remain zero")
    scenarios = suite.get("scenarios")
    _require(isinstance(scenarios, list), "suite scenarios are missing")
    matches = [item for item in scenarios if item.get("scenario_id") == scenario_id]
    _require(len(matches) == 1, "scenario selection is not unique")
    scenario = matches[0]
    plan = scenario.get("plan")
    _require(isinstance(plan, Mapping), "scenario plan is missing")
    room = plan.get("room")
    _require(
        isinstance(room, Mapping)
        and room.get("runtime_map") == ENTRY_MAP
        and room.get("scene_id") == "17DRP5sb8fy",
        "scenario does not bind the expected imported MP3D room",
    )
    nested_adapter = room.get("room_adapter")
    _require(
        isinstance(nested_adapter, Mapping)
        and nested_adapter.get("static_mesh_object_paths")
        == room_adapter.get("static_mesh_object_paths")
        and nested_adapter.get("camera_contract")
        == room_adapter.get("camera_contract"),
        "suite and execution room-adapter bindings differ",
    )
    declarations = plan.get("actors")
    _require(
        isinstance(declarations, list)
        and [item.get("actor_id") for item in declarations]
        == ["source1_actor", "source2_actor"]
        and all(item.get("body_plan_id") == "biped_human" for item in declarations)
        and "male_adult_01" in declarations[0].get("template_id", "")
        and "female_adult_01" in declarations[1].get("template_id", ""),
        "capture requires one distinct male and one distinct female human",
    )
    _require(
        all(
            isinstance(item.get("emitter_offset_m"), list)
            and len(item["emitter_offset_m"]) == 3
            for item in declarations
        ),
        "declared mouth-proxy offsets are missing",
    )
    all_frames = plan.get("frames")
    _require(
        isinstance(all_frames, list)
        and len(all_frames) == FRAME_COUNT
        and [item.get("frame_index") for item in all_frames]
        == list(range(FRAME_COUNT)),
        "scenario is not one complete ordered 75-frame Episode",
    )
    camera = plan.get("camera")
    _require(isinstance(camera, Mapping), "camera/listener contract is missing")
    habitat_camera = _vector3(
        camera.get("habitat_position_m"), owner="camera Habitat position"
    )
    ue_camera = _vector3(camera.get("ue_position_cm"), owner="camera UE position")
    _require(
        _maximum_error(ue_camera, _habitat_to_ue_cm(habitat_camera)) <= 1.0e-6,
        "camera/listener Habitat-to-UE position conversion drift",
    )
    _require(
        abs(
            float(camera.get("ue_yaw_deg"))
            - (-90.0 - float(camera.get("habitat_yaw_deg")))
        )
        <= 1.0e-9,
        "camera/listener Habitat-to-UE yaw conversion drift",
    )
    _require(
        float(camera.get("horizontal_fov_deg")) == 90.0
        and camera.get("listener_id") == "listener0",
        "camera/listener FOV or identity drift",
    )
    declaration_ids = [item["actor_id"] for item in declarations]
    roots_by_actor: dict[str, list[float]] = {}
    pose_ids = set()
    for frame in all_frames:
        camera_state = frame.get("camera_state")
        _require(isinstance(camera_state, Mapping), "frame camera state is missing")
        _require(
            _maximum_error(camera_state["habitat_position_m"], habitat_camera) <= 1.0e-9
            and _maximum_error(camera_state["ue_position_cm"], ue_camera) <= 1.0e-6
            and float(camera_state["habitat_yaw_deg"])
            == float(camera["habitat_yaw_deg"])
            and float(camera_state["ue_yaw_deg"]) == float(camera["ue_yaw_deg"]),
            "camera and colocated listener are not static across the Episode",
        )
        pose_ids.add(camera_state.get("pose_hash"))
        states = frame.get("actor_states")
        _require(
            isinstance(states, list)
            and [item.get("actor_id") for item in states] == declaration_ids,
            "frame actor closure or order drift",
        )
        for state in states:
            actor_id = state["actor_id"]
            root = _vector3(state["translation_m"], owner=f"{actor_id} root")
            _require(
                _maximum_error(state["translation_ue_cm"], _habitat_to_ue_cm(root))
                <= 1.0e-6,
                f"{actor_id} Habitat-to-UE root conversion drift",
            )
            if actor_id not in roots_by_actor:
                roots_by_actor[actor_id] = root
            _require(
                _maximum_error(root, roots_by_actor[actor_id]) <= 1.0e-9
                and state.get("action_id") == "idle",
                f"{actor_id} must be a static idle strict probe",
            )
    _require(len(pose_ids) == 1 and None not in pose_ids, "camera pose identity drift")
    indices = (
        list(range(FRAME_COUNT))
        if requested_frame_indices is None
        else [int(value) for value in requested_frame_indices]
    )
    _require(
        indices
        and len(indices) == len(set(indices))
        and all(0 <= index < FRAME_COUNT for index in indices),
        "selected frame indices must be unique values in [0,74]",
    )
    return scenario, [all_frames[index] for index in indices]


def _unreal_handle(value: Any, *, owner: str) -> int:
    return live_handle(value, owner=owner)


def _set_camera_hfov(
    camera: Any,
    components: Mapping[str, Any],
    hfov_deg: float,
) -> dict[str, Any]:
    """Set HFOV on the exact named scene-capture components already in use.

    ``BP_CameraSensor`` is a multimodal scene-capture actor; it does not promise
    exactly one ``UCameraComponent``.  The native pixel runner has already
    resolved the RGB, metric-depth, and object-ID components by stable name, so
    those exact live objects are the only authoritative HFOV targets.
    """

    capability = set_numeric_property_with_readback(
        components,
        owner="named camera components",
        property_name="FOVAngle",
        requested_value=float(hfov_deg),
        required_names=("rgb", "depth", "object_ids"),
        tolerance=1.0e-6,
        require_distinct_handles=True,
    )
    return {
        "status": "pass",
        "camera_actor_handle": _unreal_handle(camera, owner="BP_CameraSensor"),
        "component_handles": capability["component_handles"],
        "requested_horizontal_fov_deg": float(hfov_deg),
        "observed_horizontal_fov_deg_by_component": capability["observed_by_component"],
        "write_method": "named_USpSceneCaptureComponent2D.FOVAngle_property",
        "capability_readback": capability,
    }


def _camera_pass_identity(
    camera: Any,
    components: Mapping[str, Any],
    *,
    pass_id: str,
) -> dict[str, Any]:
    return {
        "pass_id": pass_id,
        "camera_actor_handle": _unreal_handle(camera, owner="BP_CameraSensor"),
        "rgb_component_handle": _unreal_handle(
            components["rgb"], owner="RGB component"
        ),
        "metric_depth_component_handle": _unreal_handle(
            components["depth"], owner="metric-depth component"
        ),
        "object_id_component_handle": _unreal_handle(
            components["object_ids"], owner="object-ID component"
        ),
    }


def _assert_shared_camera(pass_identities: Sequence[Mapping[str, Any]]) -> None:
    _require(
        [item["pass_id"] for item in pass_identities]
        == ["normal", "source1_target_only", "source2_target_only"],
        "capture pass order drift",
    )
    for key in (
        "camera_actor_handle",
        "rgb_component_handle",
        "metric_depth_component_handle",
        "object_id_component_handle",
    ):
        _require(
            len({int(item[key]) for item in pass_identities}) == 1,
            f"{key} differs across capture passes",
        )


def _mouth_proxy_projection(
    scenario: Mapping[str, Any], frame: Mapping[str, Any]
) -> dict[str, list[float]]:
    plan = scenario["plan"]
    camera = plan["camera"]
    camera_ue = _vector3(camera["ue_position_cm"], owner="camera UE position")
    yaw = math.radians(float(camera["ue_yaw_deg"]))
    forward = [math.cos(yaw), math.sin(yaw), 0.0]
    right = [-math.sin(yaw), math.cos(yaw), 0.0]
    focal = (
        0.5 * WIDTH / math.tan(math.radians(float(camera["horizontal_fov_deg"])) / 2.0)
    )
    principal = [(WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0]
    declarations = {item["actor_id"]: item for item in plan["actors"]}
    result: dict[str, list[float]] = {}
    for state in frame["actor_states"]:
        actor_id = state["actor_id"]
        instance_id = actor_id.removesuffix("_actor")
        root = _vector3(state["translation_m"], owner=f"{actor_id} root")
        offset = _vector3(
            declarations[actor_id]["emitter_offset_m"],
            owner=f"{actor_id} emitter offset",
        )
        mouth_h = [root[index] + offset[index] for index in range(3)]
        mouth_ue = _habitat_to_ue_cm(mouth_h)
        relative = [mouth_ue[index] - camera_ue[index] for index in range(3)]
        depth = sum(relative[index] * forward[index] for index in range(3))
        horizontal = sum(relative[index] * right[index] for index in range(3))
        _require(depth > 0.0, f"{instance_id} mouth proxy is behind the camera")
        result[instance_id] = [
            principal[0] + focal * horizontal / depth,
            principal[1] - focal * relative[2] / depth,
        ]
    _require(
        result["source1"][0] < result["source2"][0],
        "planned mouth proxies are not left/right separated",
    )
    return result


def _target_bbox(mask: Any) -> dict[str, Any]:
    import numpy as np

    rows, columns = np.nonzero(mask)
    _require(rows.size > 0, "target-only human has no rendered pixels")
    x0 = int(columns.min())
    y0 = int(rows.min())
    x1 = int(columns.max())
    y1 = int(rows.max())
    return {
        "xyxy_inclusive_px": [x0, y0, x1, y1],
        "center_uv": [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
        "pixel_count": int(rows.size),
        "touches_frame_boundary": x0 == 0
        or y0 == 0
        or x1 == WIDTH - 1
        or y1 == HEIGHT - 1,
    }


def _build_live_review(
    *,
    scenario: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
    target_depths: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    per_frame = []
    for capture_index, frame in enumerate(frames):
        mouths = _mouth_proxy_projection(scenario, frame)
        per_source = {}
        for instance_id in ("source1", "source2"):
            target = target_depths[instance_id][capture_index]
            bbox = _target_bbox(target < TARGET_ONLY_BACKGROUND_DEPTH_M)
            x0, y0, x1, y1 = bbox["xyxy_inclusive_px"]
            u, v = mouths[instance_id]
            bbox["declared_mouth_proxy_uv"] = [u, v]
            bbox["declared_mouth_proxy_inside_body_bbox"] = (
                x0 <= u <= x1 and y0 <= v <= y1
            )
            bbox["mouth_authority"] = (
                "root-plus-declared-offset proxy; no live bone/socket claim"
            )
            per_source[instance_id] = bbox
        left_right = (
            per_source["source1"]["center_uv"][0]
            < per_source["source2"]["center_uv"][0]
        )
        per_frame.append(
            {
                "frame_index": int(frame["frame_index"]),
                "source1_body_left_of_source2_body": left_right,
                "per_source": per_source,
            }
        )
    bbox_gate = all(
        item["source1_body_left_of_source2_body"]
        and all(
            source["pixel_count"] > 0 and not source["touches_frame_boundary"]
            for source in item["per_source"].values()
        )
        for item in per_frame
    )
    proxy_inside = all(
        source["declared_mouth_proxy_inside_body_bbox"]
        for item in per_frame
        for source in item["per_source"].values()
    )
    return {
        "status": (
            "automated_bbox_pass_manual_mouth_review_pending"
            if bbox_gate
            else "automated_bbox_review_failed"
        ),
        "automated_full_body_bbox_gate": bbox_gate,
        "all_declared_mouth_proxies_inside_live_body_bbox": proxy_inside,
        "live_mouth_bone_or_socket_status": "pending_not_available",
        "manual_sparse_f15_visual_review_required": True,
        "per_frame": per_frame,
    }


def _write_review_overlays(
    *,
    output: Path,
    rgb_frames: Sequence[Any],
    review: Mapping[str, Any],
) -> list[str]:
    import cv2

    paths = []
    colors = {"source1": (255, 128, 0), "source2": (0, 128, 255)}
    for capture_index, (rgb, record) in enumerate(zip(rgb_frames, review["per_frame"])):
        canvas = rgb.copy()
        for instance_id in ("source1", "source2"):
            source = record["per_source"][instance_id]
            x0, y0, x1, y1 = source["xyxy_inclusive_px"]
            u, v = source["declared_mouth_proxy_uv"]
            color = colors[instance_id]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            cv2.drawMarker(
                canvas,
                (round(u), round(v)),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=24,
                thickness=2,
            )
            cv2.putText(
                canvas,
                instance_id,
                (x0, max(18, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        path = output / (
            f"review_{capture_index:06d}_formal_{record['frame_index']:02d}.png"
        )
        _require(cv2.imwrite(str(path), canvas), f"could not write {path}")
        paths.append(str(path.resolve()))
    return paths


def _run_impl(args: argparse.Namespace, journal: CapturePhaseJournal) -> Path:
    import cv2
    import numpy as np

    suite = load_json_object(args.suite_plan.resolve(), owner="suite plan")
    room_adapter = load_json_object(args.room_adapter.resolve(), owner="room adapter")
    scenario, frames = validate_capture_contract(
        suite,
        scenario_id=args.scenario_id,
        room_adapter=room_adapter,
        requested_frame_indices=args.frame_index,
    )
    _require(not args.output.exists(), f"refusing to replace output: {args.output}")
    args.output.mkdir(parents=True)
    rgb_directory = args.output / "rgb_frames"
    rgb_directory.mkdir()
    journal.enter("preconnect")

    native = _load_native_capture_backend()
    runner = native.RUNNER
    spike = native.SPIKE
    configure_args = argparse.Namespace(
        spear_executable=(
            args.spear_root
            / "cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh"
        ),
        rpc_port=args.rpc_port,
        graphics_adapter=args.graphics_adapter,
    )
    instance = runner._configure_instance(
        configure_args, native_map=ENTRY_MAP
    )
    game = instance.get_game()
    journal.enter("post-entry")
    room_actors: list[Any] = []
    runtimes: dict[str, Any] = {}
    camera = None
    components: dict[str, Any] = {}
    room_readback: dict[str, Any] = {}
    lighting_readback: dict[str, Any] = {}
    fov_readback: dict[str, Any] = {}
    normal_rgbs: list[Any] = []
    normal_depths: list[Any] = []
    normal_object_ids: list[Any] = []
    normal_readbacks: list[dict[str, Any]] = []
    target_depths: dict[str, list[Any]] = {}
    target_readbacks: dict[str, list[dict[str, Any]]] = {}
    pass_identities: list[dict[str, Any]] = []
    try:
        journal.enter("mesh")
        with instance.begin_frame():
            room_actors, room_readback = spawn_scene_meshes_with_readback(
                game, room_adapter
            )
            journal.enter("lighting")
            lighting_readback = spawn_review_lighting(
                game, room_adapter["review_lighting"]
            )
            journal.enter("camera")
            camera, components = spike._spawn_multimodal_camera(game)
            _require(
                set(components) >= {"rgb", "depth", "object_ids"},
                "BP_CameraSensor multimodal component closure drift",
            )
            fov_readback = _set_camera_hfov(
                camera,
                components,
                float(scenario["plan"]["camera"]["horizontal_fov_deg"]),
            )
            journal.enter("actor")
            runtimes = runner._spawn_runtime_actors(game, scenario)
            spike._apply_exact_frame(camera=camera, runtimes=runtimes, frame=frames[0])
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                bPaused=False
            )
        with instance.end_frame():
            pass
        instance.step(num_frames=args.warmup_frames)

        journal.enter("capture")
        with instance.begin_frame():
            game.segmentation_service.initialize()
            components["depth"].PrimitiveRenderMode = "PRM_RenderScenePrimitives"
            components["depth"].ShowOnlyActors = []
            spike._apply_exact_frame(camera=camera, runtimes=runtimes, frame=frames[0])
            pass_identities.append(
                _camera_pass_identity(camera, components, pass_id="normal")
            )
        with instance.end_frame():
            pass
        instance.step(num_frames=2)

        for capture_index, frame in enumerate(frames):
            with instance.begin_frame():
                readback = spike._apply_exact_frame(
                    camera=camera, runtimes=runtimes, frame=frame
                )
            with instance.end_frame():
                rgb = native._rgb_bgr(components["rgb"])
                depth = native._depth_native(components["depth"])
                object_ids = native._raw_object_ids(components["object_ids"])
            path = rgb_directory / f"frame_{capture_index:06d}.png"
            _require(cv2.imwrite(str(path), rgb), f"could not write {path}")
            normal_rgbs.append(rgb)
            normal_depths.append(depth)
            normal_object_ids.append(object_ids)
            normal_readbacks.append(readback)

        for instance_id in ("source1", "source2"):
            actor_id = f"{instance_id}_actor"
            with instance.begin_frame():
                components["depth"].PrimitiveRenderMode = "PRM_UseShowOnlyList"
                components["depth"].ShowOnlyActors = [
                    runtimes[actor_id]["visual_actor"]
                ]
                spike._apply_exact_frame(
                    camera=camera, runtimes=runtimes, frame=frames[0]
                )
                pass_identities.append(
                    _camera_pass_identity(
                        camera, components, pass_id=f"{instance_id}_target_only"
                    )
                )
            with instance.end_frame():
                pass
            instance.step(num_frames=2)
            target_depths[instance_id] = []
            target_readbacks[instance_id] = []
            for frame in frames:
                with instance.begin_frame():
                    readback = spike._apply_exact_frame(
                        camera=camera, runtimes=runtimes, frame=frame
                    )
                with instance.end_frame():
                    depth = native._depth_native(components["depth"])
                target_depths[instance_id].append(depth)
                target_readbacks[instance_id].append(readback)
        _assert_shared_camera(pass_identities)
    finally:
        if runtimes:
            try:
                runner._destroy_runtime_actors(instance, runtimes)
            except Exception:  # noqa: BLE001, S110
                pass
        if room_actors:
            try:
                destroy_scene_meshes(instance, room_actors)
            except Exception:  # noqa: BLE001, S110
                pass
        instance.close(force=True)

    journal.enter("artifact_finalize")
    _require(
        room_readback.get("spawned_static_mesh_count") == EXPECTED_STATIC_MESH_COUNT
        and room_readback.get("all_expected_handles_match_components") is True,
        "fresh packaged room readback did not close 71 unique meshes",
    )
    alignment = native._maximum_readback_drift(normal_readbacks, target_readbacks)
    review = _build_live_review(
        scenario=scenario, frames=frames, target_depths=target_depths
    )
    overlay_paths = _write_review_overlays(
        output=args.output, rgb_frames=normal_rgbs, review=review
    )
    frame_indices = [int(frame["frame_index"]) for frame in frames]
    depth_path = args.output / "metric_depth_native.npz"
    np.savez_compressed(
        depth_path,
        normal_depth_m=np.stack(normal_depths),
        target_only_source1_depth_m=np.stack(target_depths["source1"]),
        target_only_source2_depth_m=np.stack(target_depths["source2"]),
    )
    object_id_path = args.output / "normal_object_ids_uint32.npz"
    np.savez_compressed(object_id_path, normal_object_ids=np.stack(normal_object_ids))
    room_readback_path = args.output / "room_live_readback.json"
    _write_json(room_readback_path, room_readback)
    runtime_readback_path = args.output / "runtime_readbacks.json"
    _write_json(
        runtime_readback_path,
        {
            "schema": "avengine_spear_shared_camera_runtime_readbacks_v1",
            "normal": normal_readbacks,
            "target_only": target_readbacks,
        },
    )
    manifest = {
        "schema": SCHEMA,
        "status": "capture_pass_review_pending",
        "scenario_id": args.scenario_id,
        "native_map": ENTRY_MAP,
        "room": {
            "scene_id": "17DRP5sb8fy",
            "fresh_cooked_mesh_readback_status": "pass",
            "spawned_static_mesh_count": EXPECTED_STATIC_MESH_COUNT,
            "lighting_readback": lighting_readback,
        },
        "coordinate_and_listener_contract": {
            "status": "pass",
            "raw_room_source_axis": "Matterport raw GLB Z-up",
            "raw_to_habitat": "H=(S.x,S.z,-S.y)",
            "habitat_to_ue_cm": "U_cm=(100*H.x,100*H.z,100*H.y)",
            "camera_listener_coupling": "rigid_colocated_cooriented",
            "habitat_position_m": scenario["plan"]["camera"]["habitat_position_m"],
            "ue_position_cm": scenario["plan"]["camera"]["ue_position_cm"],
        },
        "frame_contract": {
            "captured_frame_count": len(frames),
            "formal_episode_frame_count": FRAME_COUNT,
            "frame_rate_hz": FPS,
            "captured_frame_indices": frame_indices,
            "resolution_hw": [HEIGHT, WIDTH],
        },
        "camera_contract": {
            "status": "pass",
            "blueprint_class_path": CAMERA_BLUEPRINT,
            "component_paths": {
                "rgb": RGB_COMPONENT,
                "metric_depth": DEPTH_COMPONENT,
                "object_ids": OBJECT_ID_COMPONENT,
            },
            "same_actor_and_components_across_all_three_passes": True,
            "pass_identities": pass_identities,
            "hfov_readback": fov_readback,
            "runtime_alignment": alignment,
        },
        "live_review": review,
        "artifacts": {
            "rgb_frames": str(rgb_directory.resolve()),
            "metric_depth": str(depth_path.resolve()),
            "normal_object_ids": str(object_id_path.resolve()),
            "room_live_readback": str(room_readback_path.resolve()),
            "runtime_readbacks": str(runtime_readback_path.resolve()),
            "review_overlays": overlay_paths,
        },
        "gpu_f15_review_ready": args.frame_index == [15],
        "gpu_full75_allowed": False,
        "gpu_full75_blocker": (
            "manual f15 M/F full-body and declared-mouth-proxy review plus exact "
            "2/2 CPU RIR completion required"
        ),
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    manifest_path = args.output / "manifest.json"
    _write_json(manifest_path, manifest)
    journal.enter("complete")
    print(
        "SPEAR_IMPORTED_MP3D_STRICT_TWO_HUMAN_CAPTURE_OK "
        f"output={args.output} frames={len(frames)} meshes=71 formal=0",
        flush=True,
    )
    return manifest_path


def run(args: argparse.Namespace) -> Path:
    journal = CapturePhaseJournal(args.output)
    try:
        return _run_impl(args, journal)
    except BaseException as exc:
        journal.record_failure(exc)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-plan", required=True, type=Path)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--room-adapter", required=True, type=Path)
    parser.add_argument("--spear-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rpc-port", type=int, default=39631)
    parser.add_argument("--graphics-adapter", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=120)
    parser.add_argument(
        "--frame-index",
        action="append",
        type=int,
        help="Capture only this formal frame; repeat for a sparse request.",
    )
    args = parser.parse_args(argv)
    if not 1024 <= args.rpc_port <= 65535:
        parser.error("--rpc-port must be in [1024,65535]")
    if args.graphics_adapter < 0 or args.warmup_frames < 0:
        parser.error("GPU and warmup values must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

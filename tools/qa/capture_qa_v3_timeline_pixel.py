#!/usr/bin/env python3
"""Capture native pixel visibility for one current QA-v3 visual timeline.

This is a thin adapter between the current ``actor_selection + timeline``
authoring path and the retained same-camera normal/target-only depth authority.
It is a research probe: it does not admit a question, render audio, or create a
formal dataset episode.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))


def _preload_spear_extension(argv: Sequence[str]) -> Path | None:
    """Put an explicitly declared host SDK on sys.path before client import."""

    positions = [index for index, value in enumerate(argv) if value == "--spear-ext"]
    if not positions:
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RuntimeError("--spear-ext must be supplied exactly once with a path")
    path = Path(argv[positions[0] + 1]).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"--spear-ext is not a directory: {path}")
    sys.path.insert(1, str(path))
    return path


_BOOTSTRAP_SPEAR_EXT = _preload_spear_extension(sys.argv[1:])

from avengine.backends.spear_ue.research_runtime import read_actor_pose  # noqa: E402
from avengine.qa.pixel_visibility import (  # noqa: E402
    PIXEL_VISIBILITY_DEPTH_AUTHORITY,
    compile_depth_pixel_visibility_truth,
)
from avengine.timeline import current_apartment_visual as VISUAL  # noqa: E402

SPIKE_PATH = REPOSITORY / "tools/qa/spike_spear_native_pixel_visibility.py"
SPIKE_SPEC = importlib.util.spec_from_file_location(
    "qa_v3_native_pixel_helpers", SPIKE_PATH
)
if SPIKE_SPEC is None or SPIKE_SPEC.loader is None:
    raise RuntimeError(f"cannot import {SPIKE_PATH}")
SPIKE = importlib.util.module_from_spec(SPIKE_SPEC)
SPIKE_SPEC.loader.exec_module(SPIKE)

SCHEMA = "qa_v3_current_timeline_native_pixel_probe_v1"
TARGET_ONLY_BACKGROUND_DEPTH_M = 65504.0
ABSOLUTE_TOLERANCE_M = 0.01
RELATIVE_TOLERANCE = 0.002


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _selected_indices(
    values: Sequence[int] | None, *, frame_count: int = VISUAL.FRAME_COUNT,
) -> list[int]:
    _require(
        isinstance(frame_count, int) and not isinstance(frame_count, bool)
        and frame_count > 0,
        "timeline frame_count must be a positive integer",
    )
    if values is None:
        selected = sorted({0, frame_count // 2, frame_count - 1})
    else:
        selected = [int(value) for value in values]
    _require(
        selected
        and len(selected) == len(set(selected))
        and selected == sorted(selected)
        and all(0 <= value < frame_count for value in selected),
        f"--frame-index values must be unique, increasing values in [0,{frame_count - 1}]",
    )
    return selected


def _camera_settings(render: Mapping[str, Any]) -> dict[str, Any]:
    height, width = render["resolution_hw"]
    hfov = float(render["hfov_degrees"])
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in (height, width)) and 0.0 < hfov < 180.0,
        "timeline camera resolution and HFOV must be valid",
    )
    return {"height": height, "width": width, "hfov_deg": hfov}


def _pose_id(frame: Mapping[str, Any]) -> str:
    camera = frame["camera"]
    values = [*camera["translation_ue_cm"], camera["yaw_ue_deg"]]
    return "frame=%d;ue_pose=%s" % (
        int(frame["frame_index"]),
        ",".join(format(float(value), ".12g") for value in values),
    )


def _apply_frame(
    *,
    camera: Any,
    runtimes: Mapping[str, Mapping[str, Any]],
    frame: Mapping[str, Any],
) -> dict[str, Any]:
    camera_state = frame["camera"]
    camera.K2_SetActorLocationAndRotation(
        NewLocation={
            "X": camera_state["translation_ue_cm"][0],
            "Y": camera_state["translation_ue_cm"][1],
            "Z": camera_state["translation_ue_cm"][2],
        },
        NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": camera_state["yaw_ue_deg"]},
        bSweep=False,
        bTeleport=True,
    )
    animations = {}
    for state in frame["actor_states"]:
        slot = str(state["source_slot_id"])
        animations[slot] = VISUAL._apply_runtime_state(
            runtimes[slot], state=state, frame_index=int(frame["frame_index"])
        )
    return {
        "frame_index": int(frame["frame_index"]),
        "declared_camera_pose_id": _pose_id(frame),
        "camera": read_actor_pose(camera),
        "actors": {
            slot: read_actor_pose(runtimes[slot]["anchor"])
            for slot in runtimes
        },
        "animations": animations,
    }


def _maximum_pass_drift(
    normal: Sequence[Mapping[str, Any]],
    target: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    maximum_location_cm = 0.0
    maximum_rotation_deg = 0.0
    for pass_name, records in target.items():
        _require(len(records) == len(normal), f"{pass_name} readback length differs")
        for left, right in zip(normal, records):
            _require(
                left["declared_camera_pose_id"] == right["declared_camera_pose_id"],
                f"{pass_name} replayed a different declared camera pose",
            )
            actor_slots = tuple(left["actors"])
            for owner in ("camera", *actor_slots):
                left_pose = left["camera"] if owner == "camera" else left["actors"][owner]
                right_pose = right["camera"] if owner == "camera" else right["actors"][owner]
                maximum_location_cm = max(
                    maximum_location_cm,
                    max(
                        abs(float(a) - float(b))
                        for a, b in zip(left_pose["location_cm"], right_pose["location_cm"])
                    ),
                )
                maximum_rotation_deg = max(
                    maximum_rotation_deg,
                    max(
                        abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)
                        for a, b in zip(left_pose["rotation_deg"], right_pose["rotation_deg"])
                    ),
                )
    _require(maximum_location_cm <= 1.0e-4, "target-only pass location drift")
    _require(maximum_rotation_deg <= 1.0e-4, "target-only pass rotation drift")
    return {
        "maximum_location_drift_cm": maximum_location_cm,
        "maximum_rotation_drift_deg": maximum_rotation_deg,
    }


def _close_multimodal_camera(
    *, instance: Any, game: Any, camera: Any, components: Mapping[str, Any]
) -> None:
    """Release all shared-memory capture components before closing SPEAR."""

    if camera is None and not components:
        return
    with instance.begin_frame():
        pass
    with instance.end_frame():
        for component in components.values():
            try:
                component.terminate_sp_funcs()
            finally:
                component.Terminate()
        if camera is not None:
            game.unreal_service.destroy_actor(actor=camera)


def run(args: argparse.Namespace) -> Path:
    from PIL import Image

    selection_file, bindings, authorization = VISUAL._selection_bindings(
        actor_selection_path=args.actor_selection,
        source_asset_registry_path=args.source_asset_registry,
    )
    timeline_file, timeline = VISUAL._load_timeline(
        timeline_path=args.timeline,
        bindings=bindings,
        asset_authorization=authorization,
    )
    _require(authorization == "verified_internal", "pixel probe requires verified assets")
    frame_count, frame_rate_hz, _ticks = VISUAL._timeline_render_clock(timeline)
    render = timeline["render"]
    camera_settings = _camera_settings(render)
    native_map = VISUAL.resolve_native_map(timeline, args.native_map)
    closure_file, mappings = VISUAL._closure_mappings(
        closure_report_path=args.closure_report,
        bindings=bindings,
        native_map=native_map,
    )
    stage, executable = VISUAL._validate_stage(
        stage_root=args.stage_root,
        spear_executable=args.spear_executable,
        closure_mappings=mappings,
    )
    indices = _selected_indices(args.frame_index, frame_count=frame_count)
    frames = [timeline["frames"][index] for index in indices]
    output = VISUAL._new_external_output_directory(args.output, owner="pixel output")
    rgb_directory = output / "rgb_frames"
    rgb_directory.mkdir()

    instance = VISUAL.launch_external_game_instance(
        spear_executable=executable,
        native_map=native_map,
        frame_rate_hz=frame_rate_hz,
        rpc_port=args.rpc_port,
        graphics_adapter=args.graphics_adapter,
    )
    game = instance.get_game()
    camera = None
    components: dict[str, Any] = {}
    runtimes: dict[str, dict[str, Any]] = {}
    normal_depths: list[np.ndarray] = []
    normal_object_ids: list[np.ndarray] = []
    normal_readbacks: list[dict[str, Any]] = []
    target_depths: dict[str, list[np.ndarray]] = {slot: [] for slot in bindings}
    target_readbacks: dict[str, list[dict[str, Any]]] = {slot: [] for slot in bindings}
    descriptors: list[dict[str, Any]] = []
    stable_names: dict[str, str] = {}
    try:
        with instance.begin_frame():
            camera, components = SPIKE._spawn_multimodal_camera(game, **camera_settings)
            runtimes = VISUAL._spawn_runtime_actors(
                game=game, bindings=bindings, initial_frame=timeline["frames"][0]
            )
            for slot, runtime in runtimes.items():
                stable_name = f"qa_v3_timeline_{slot}"
                game.unreal_service.set_stable_name_for_actor(
                    actor=runtime["visual_actor"], stable_name=stable_name
                )
                stable_names[slot] = stable_name
            _apply_frame(camera=camera, runtimes=runtimes, frame=frames[0])
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(bPaused=False)
        with instance.end_frame():
            pass
        instance.step(num_frames=args.warmup_frames)
        capture_warmup = VISUAL.warm_scene_capture_until_stable(
            instance, components["rgb"], config_path=args.capture_warmup_config)

        with instance.begin_frame():
            game.segmentation_service.initialize()
            components["depth"].PrimitiveRenderMode = "PRM_RenderScenePrimitives"
            components["depth"].ShowOnlyActors = []
            _apply_frame(camera=camera, runtimes=runtimes, frame=frames[0])
        with instance.end_frame():
            pass
        instance.step(num_frames=2)

        with instance.begin_frame():
            descriptors = [
                dict(value)
                for value in game.segmentation_service.get_mesh_proxy_geometry_descs(
                    include_debug_info=False, as_global=True
                )
            ]
        with instance.end_frame():
            pass
        for slot, stable_name in stable_names.items():
            _require(
                SPIKE._descriptor_indices_for_stable_actor(descriptors, stable_name),
                f"SPEAR proxy descriptors lack {slot}/{stable_name}",
            )

        for capture_index, frame in enumerate(frames):
            with instance.begin_frame():
                readback = _apply_frame(camera=camera, runtimes=runtimes, frame=frame)
            with instance.end_frame():
                capture = SPIKE._capture_buffers(
                    game=game, components=components, include_rgb_depth=True
                )
            Image.fromarray(
                np.ascontiguousarray(capture["rgb"][:, :, ::-1]), mode="RGB"
            ).save(
                rgb_directory / f"frame_{int(frame['frame_index']):06d}.png"
            )
            normal_depths.append(capture["depth_m"])
            normal_object_ids.append(capture["raw_object_ids_uint32"])
            normal_readbacks.append(readback)

        for slot in bindings:
            with instance.begin_frame():
                manager = game.segmentation_service.proxy_component_manager
                manager.SetAllowedActors(AllowedActors=[runtimes[slot]["visual_actor"]])
                game.segmentation_service.initialize()
                components["depth"].PrimitiveRenderMode = "PRM_UseShowOnlyList"
                components["depth"].ShowOnlyActors = [runtimes[slot]["visual_actor"]]
                _apply_frame(camera=camera, runtimes=runtimes, frame=frames[0])
            with instance.end_frame():
                pass
            instance.step(num_frames=2)
            for frame in frames:
                with instance.begin_frame():
                    readback = _apply_frame(camera=camera, runtimes=runtimes, frame=frame)
                with instance.end_frame():
                    capture = SPIKE._capture_buffers(
                        game=game, components=components, include_rgb_depth=True
                    )
                target_depths[slot].append(capture["depth_m"])
                target_readbacks[slot].append(readback)
    finally:
        try:
            if camera is not None or components:
                _close_multimodal_camera(
                    instance=instance, game=game, camera=camera, components=components
                )
        finally:
            if runtimes:
                try:
                    VISUAL._destroy_runtime_actors(instance, runtimes)
                except Exception:
                    pass
            instance.close(force=True)

    pose_ids = [_pose_id(frame) for frame in frames]
    context = {
        "renderer_backend": f"spear_unreal_native:{native_map}",
        "rgb_renderer_backend": f"spear_unreal_native:{native_map}",
        "camera_contract_id": (
            f"qa_v3_bp_camera_sensor_{camera_settings['width']}x"
            f"{camera_settings['height']}_hfov{camera_settings['hfov_deg']:g}_v1"),
        "semantic_id_namespace": "qa_v3_same_camera_target_only_depth_instances_v1",
        "resolution_hw": list(render["resolution_hw"]),
        "frame_indices": indices,
        "frame_rate_hz": float(render["frame_rate_hz"]),
        "hfov_degrees": camera_settings["hfov_deg"],
        "camera_pose_ids": pose_ids,
    }
    semantic_ids = {slot: index for index, slot in enumerate(bindings, start=1)}
    truth = compile_depth_pixel_visibility_truth(
        normal_depth_m_frames=normal_depths,
        target_only_depth_m_frames_by_instance=target_depths,
        semantic_ids_by_instance=semantic_ids,
        normal_context={"pass_kind": "modal_scene", **context},
        target_only_contexts_by_instance={
            slot: {"pass_kind": "target_only", "target_instance_id": slot, **context}
            for slot in semantic_ids
        },
        target_only_background_depth_m=TARGET_ONLY_BACKGROUND_DEPTH_M,
        absolute_tolerance_m=ABSOLUTE_TOLERANCE_M,
        relative_tolerance=RELATIVE_TOLERANCE,
    )
    _require(
        truth["authority"] == PIXEL_VISIBILITY_DEPTH_AUTHORITY,
        "pixel compiler returned the wrong authority",
    )
    alignment = _maximum_pass_drift(normal_readbacks, target_readbacks)
    arrays = {
        "normal_depth_m": np.stack(normal_depths),
        "normal_object_ids_uint32": np.stack(normal_object_ids),
    }
    arrays.update({
        f"target_only_{slot}_depth_m": np.stack(depths)
        for slot, depths in target_depths.items()})
    np.savez_compressed(output / "native_depth_and_object_ids.npz", **arrays)
    _write_json(output / "pixel_visibility_truth.json", truth)
    _write_json(
        output / "runtime_readbacks.json",
        {"normal": normal_readbacks, "target_only": target_readbacks, "alignment": alignment},
    )
    _write_json(
        output / "evidence.json",
        {
            "schema": SCHEMA,
            "status": "pass",
            "research_only": True,
            "episode_counted": False,
            "qualification_claim": False,
            "claim_boundary": (
                "selected-frame native pixel evidence for one QA-v3 timeline; "
                "does not render audio or admit a question"
            ),
            "inputs": {
                "actor_selection": str(selection_file),
                "timeline": str(timeline_file),
                "spear_ext": str(args.spear_ext.resolve()),
                "closure_report": str(closure_file),
                "stage_root": str(stage),
                "spear_executable": str(executable),
            },
            "native_map": native_map,
            "render": dict(render),
            "frame_indices": indices,
            "runtime_alignment": alignment,
            "capture_warmup": capture_warmup,
            "pixel_visibility": truth,
            "artifacts": {
                "rgb_frames": "rgb_frames",
                "arrays": "native_depth_and_object_ids.npz",
                "truth": "pixel_visibility_truth.json",
                "readbacks": "runtime_readbacks.json",
            },
        },
    )
    print(f"QA_V3_TIMELINE_PIXEL_OK output={output}", flush=True)
    return output / "evidence.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-selection", required=True, type=Path)
    parser.add_argument("--source-asset-registry", required=True, type=Path)
    parser.add_argument("--timeline", required=True, type=Path)
    parser.add_argument("--closure-report", required=True, type=Path)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--spear-executable", required=True, type=Path)
    parser.add_argument(
        "--spear-ext",
        required=True,
        type=Path,
        help="declared installed AVEngine SPEAR host SDK directory",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-map")
    parser.add_argument("--frame-index", type=int, action="append")
    parser.add_argument("--rpc-port", type=int, default=39541)
    parser.add_argument("--graphics-adapter", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=40)
    parser.add_argument("--capture-warmup-config", type=Path)
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

#!/usr/bin/env python3
"""Capture one real SPEAR RGB/depth/modal/target-only visibility spike.

The spike replays one frame from a retained native Apartment suite plan.  RGB,
metric depth and modal object IDs come from the same ``BP_CameraSensor`` and
camera pose.  Target-only masks are produced by the same SPEAR object-ID
renderer after restricting its proxy manager to one target actor and replaying
the exact declared camera, actor-root and animation state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.pixel_visibility import (  # noqa: E402
    compile_depth_pixel_visibility_truth,
)


RUNNER_PATH = REPOSITORY / "tools/m6y/run_spear_apartment_canary.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "lead_a_spear_apartment_runner", RUNNER_PATH
)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise RuntimeError(f"cannot import {RUNNER_PATH}")
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


OBJECT_IDS_COMPONENT = "DefaultSceneRoot.sp_object_ids_uint8_"
DEPTH_COMPONENT = "DefaultSceneRoot.sp_depth_meters_"
SCHEMA = "avengine_qa_native_spear_pixel_visibility_spike_v1"
TARGET_ONLY_BACKGROUND_DEPTH_M = 65504.0
DEPTH_ABSOLUTE_TOLERANCE_M = 0.01
DEPTH_RELATIVE_TOLERANCE = 0.002


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _spawn_multimodal_camera(game: Any) -> tuple[Any, dict[str, Any]]:
    game.segmentation_service.initialize()
    camera_class = game.unreal_service.load_class(
        uclass="AActor", name=RUNNER.CAMERA_BLUEPRINT
    )
    camera = game.unreal_service.spawn_actor(uclass=camera_class)
    components = {
        "rgb": game.unreal_service.get_component_by_name(
            actor=camera,
            component_name=RUNNER.CAPTURE_COMPONENT_NAME,
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
        widths=RUNNER.WIDTH,
        heights=RUNNER.HEIGHT,
    )
    for component in components.values():
        component.Initialize()
        component.initialize_sp_funcs()
        component.set_property_value(property_name="FOVAngle", property_value=105.0)
        _require(
            abs(float(component.get_property_value(property_name="FOVAngle")) - 105.0)
            <= 1.0e-4,
            "multimodal camera FOV readback drift",
        )
    return camera, components


def _apply_exact_frame(
    *,
    camera: Any,
    runtimes: Mapping[str, Mapping[str, Any]],
    frame: Mapping[str, Any],
) -> dict[str, Any]:
    actor_readbacks: dict[str, Any] = {}
    for state in frame["actor_states"]:
        root, _animation = RUNNER._apply_actor_state(
            runtimes[state["actor_id"]], state, int(frame["frame_index"])
        )
        actor_readbacks[state["actor_id"]] = root
    camera_readback = RUNNER._apply_camera_state_and_readback(
        camera, frame["camera_state"], int(frame["frame_index"])
    )
    return {"camera": camera_readback, "actors": actor_readbacks}


def _capture_buffers(
    *,
    game: Any,
    components: Mapping[str, Any],
    include_rgb_depth: bool,
) -> dict[str, Any]:
    raw_object_ids = components["object_ids"].read_pixels()["arrays"]["data"].copy()
    # Decode in-process instead of calling get_segmentation_data().  This
    # cooked SPEAR build applies UE's default material to dynamically spawned
    # skeletal proxies because M_Emissive lacks bUsedWithSkeletalMesh.  Those
    # fallback pixels are valid renderer evidence but are not registered raw
    # IDs; upstream's diagnostic path opens a blocking cv2 window for them.
    raw_ids = (
        np.ascontiguousarray(raw_object_ids)
        .view(np.uint32)
        .reshape(raw_object_ids.shape[:2])
        & np.uint32(0x00FFFFFF)
    )
    descriptors = game.segmentation_service.get_mesh_proxy_geometry_descs(
        include_debug_info=False,
        as_global=True,
    )
    result = {
        "raw_object_ids_uint32": np.asarray(raw_ids, dtype=np.uint32),
        "descriptors": descriptors,
    }
    if include_rgb_depth:
        result["rgb"] = components["rgb"].read_pixels()["arrays"]["data"][
            :, :, [0, 1, 2]
        ].copy()
        result["depth_m"] = components["depth"].read_pixels()["arrays"]["data"][
            :, :, 0
        ].astype(np.float32)
    return result


def _descriptor_indices_for_stable_actor(
    descriptors: Sequence[Mapping[str, Any]], stable_name: str
) -> list[int]:
    return [
        index
        for index, descriptor in enumerate(descriptors)
        if descriptor.get("actorStableName") == stable_name
    ]


def _descriptor_raw_ids_for_stable_actor(
    descriptors: Sequence[Mapping[str, Any]],
    stable_name: str,
) -> list[int]:
    return [
        int(descriptors[index]["rawId"])
        for index in _descriptor_indices_for_stable_actor(descriptors, stable_name)
    ]


def _depth_visible_mask(
    *,
    modal_depth_m: np.ndarray,
    target_depth_m: np.ndarray,
    target_footprint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return same-camera target visibility and its per-pixel depth residual."""

    target_valid = (
        target_footprint
        & np.isfinite(target_depth_m)
        & (target_depth_m > 0.0)
    )
    modal_valid = np.isfinite(modal_depth_m) & (modal_depth_m > 0.0)
    residual_m = np.abs(modal_depth_m - target_depth_m)
    # The two passes use the same component, pose and formal animation sample.
    # Keep a small absolute allowance for float render-target quantization plus
    # a depth-proportional allowance for rasterized edge pixels.
    tolerance_m = (
        DEPTH_ABSOLUTE_TOLERANCE_M
        + DEPTH_RELATIVE_TOLERANCE * np.maximum(target_depth_m, 0.0)
    )
    visible = target_valid & modal_valid & (residual_m <= tolerance_m)
    return visible, residual_m


def _camera_intrinsics(*, width: int, height: int, hfov_deg: float) -> dict[str, Any]:
    fx = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx
    vfov = math.degrees(2.0 * math.atan(0.5 * height / fy))
    return {
        "model": "pinhole_square_pixels",
        "width": width,
        "height": height,
        "horizontal_fov_deg": hfov_deg,
        "vertical_fov_deg": vfov,
        "fx_px": fx,
        "fy_px": fy,
        "cx_px": (width - 1) / 2.0,
        "cy_px": (height - 1) / 2.0,
        "principal_point_convention": "zero_based_pixel_centers",
    }


def run(args: argparse.Namespace) -> Path:
    import cv2

    suite = json.loads(args.suite_plan.read_text(encoding="utf-8"))
    scenarios = {
        scenario["scenario_id"]: scenario for scenario in suite["scenarios"]
    }
    _require(args.scenario_id in scenarios, "scenario is absent from suite plan")
    scenario = scenarios[args.scenario_id]
    frames = scenario["plan"]["frames"]
    _require(0 <= args.frame_index < len(frames), "frame index is out of range")
    frame = frames[args.frame_index]
    args.output.mkdir(parents=True)
    configure_args = argparse.Namespace(
        spear_root=args.spear_root,
        rpc_port=args.rpc_port,
        graphics_adapter=args.graphics_adapter,
    )
    instance, spear_root = RUNNER._configure_instance(
        configure_args, native_map=str(suite["native_map"])
    )
    game = instance.get_game()
    runtimes: dict[str, Any] = {}
    components: dict[str, Any] = {}
    camera = None
    try:
        with instance.begin_frame():
            camera, components = _spawn_multimodal_camera(game)
            runtimes = RUNNER._spawn_runtime_actors(game, scenario, spear_root)
            stable_names = {}
            for actor_id, runtime in runtimes.items():
                stable_name = f"lead_a_native_{actor_id}"
                game.unreal_service.set_stable_name_for_actor(
                    actor=runtime["visual_actor"], stable_name=stable_name
                )
                stable_names[actor_id.removesuffix("_actor")] = stable_name
            _apply_exact_frame(camera=camera, runtimes=runtimes, frame=frame)
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                bPaused=False
            )
        with instance.end_frame():
            pass

        instance.step(num_frames=args.warmup_frames)
        # Rebuild object-ID proxies after the two dynamic actors exist.
        with instance.begin_frame():
            game.segmentation_service.initialize()
            _apply_exact_frame(
                camera=camera, runtimes=runtimes, frame=frame
            )
        with instance.end_frame():
            pass
        # Proxy components are registered and rendered on subsequent engine
        # frames.  Two frames match the canonical SPEAR MCP example; reapply
        # the formal state afterwards so the captured frame remains exact.
        instance.step(num_frames=2)
        with instance.begin_frame():
            normal_readback = _apply_exact_frame(
                camera=camera, runtimes=runtimes, frame=frame
            )
        with instance.end_frame():
            normal = _capture_buffers(
                game=game, components=components, include_rgb_depth=True
            )

        semantic_ids = {"source1": 1, "source2": 2}
        modal_descriptor_indices: dict[str, list[int]] = {}
        modal_descriptor_raw_ids: dict[str, list[int]] = {}
        direct_modal_pixel_counts: dict[str, int] = {}
        for instance_id, stable_name in stable_names.items():
            indices = _descriptor_indices_for_stable_actor(
                normal["descriptors"], stable_name
            )
            _require(
                indices,
                f"SPEAR proxy descriptors lack {instance_id}/{stable_name}",
            )
            raw_ids = _descriptor_raw_ids_for_stable_actor(
                normal["descriptors"], stable_name
            )
            modal_descriptor_indices[instance_id] = indices
            modal_descriptor_raw_ids[instance_id] = raw_ids
            direct_modal_pixel_counts[instance_id] = int(
                np.count_nonzero(
                    np.isin(
                        normal["raw_object_ids_uint32"],
                        np.asarray(raw_ids, dtype=np.uint32),
                    )
                )
            )

        target_only_masks: dict[str, np.ndarray] = {}
        target_only_depths: dict[str, np.ndarray] = {}
        target_only_raw_ids: dict[str, np.ndarray] = {}
        target_only_object_id_foregrounds: dict[str, np.ndarray] = {}
        target_readbacks: dict[str, Any] = {}
        for instance_id, stable_name in stable_names.items():
            actor_id = f"{instance_id}_actor"
            with instance.begin_frame():
                manager = game.segmentation_service.proxy_component_manager
                manager.SetAllowedActors(
                    AllowedActors=[runtimes[actor_id]["visual_actor"]]
                )
                game.segmentation_service.initialize()
                components["depth"].PrimitiveRenderMode = "PRM_UseShowOnlyList"
                components["depth"].ShowOnlyActors = [
                    runtimes[actor_id]["visual_actor"]
                ]
                _apply_exact_frame(
                    camera=camera, runtimes=runtimes, frame=frame
                )
            with instance.end_frame():
                pass
            instance.step(num_frames=2)
            with instance.begin_frame():
                target_readbacks[instance_id] = _apply_exact_frame(
                    camera=camera, runtimes=runtimes, frame=frame
                )
            with instance.end_frame():
                target_capture = _capture_buffers(
                    game=game, components=components, include_rgb_depth=True
                )
            target_raw = target_capture["raw_object_ids_uint32"]
            object_id_foreground = target_raw != 0
            object_id_pixels = int(np.count_nonzero(object_id_foreground))
            _require(
                object_id_pixels < object_id_foreground.size // 2,
                f"{instance_id} target-only object-ID foreground is implausible: "
                f"{object_id_pixels}/{object_id_foreground.size}",
            )
            target_only_raw_ids[instance_id] = target_raw
            target_only_depths[instance_id] = target_capture["depth_m"]
            target_only_object_id_foregrounds[instance_id] = object_id_foreground
            target_footprint = (
                target_only_depths[instance_id] < TARGET_ONLY_BACKGROUND_DEPTH_M
            )
            target_only_masks[instance_id] = np.where(
                target_footprint, semantic_ids[instance_id], 0
            ).astype(np.int32)

        # The cooked object-ID material cannot encode IDs on the dynamically
        # spawned skeletal meshes, but the exact same native camera provides a
        # stronger fallback: normal metric depth plus per-target show-only
        # metric depth.  Equality within render-target tolerance labels modal
        # target pixels; the target-only object-ID foreground is its footprint.
        modal_mask = np.zeros((RUNNER.HEIGHT, RUNNER.WIDTH), dtype=np.int32)
        best_residual_m = np.full(
            (RUNNER.HEIGHT, RUNNER.WIDTH), np.inf, dtype=np.float32
        )
        depth_visible_masks: dict[str, np.ndarray] = {}
        depth_residuals: dict[str, np.ndarray] = {}
        for instance_id in sorted(stable_names):
            footprint = target_only_masks[instance_id] == semantic_ids[instance_id]
            visible, residual_m = _depth_visible_mask(
                modal_depth_m=normal["depth_m"],
                target_depth_m=target_only_depths[instance_id],
                target_footprint=footprint,
            )
            depth_visible_masks[instance_id] = visible
            depth_residuals[instance_id] = residual_m
            wins = visible & (residual_m < best_residual_m)
            modal_mask[wins] = semantic_ids[instance_id]
            best_residual_m[wins] = residual_m[wins]

        contexts = {
            "renderer_backend": "spear_unreal_native_apartment",
            "rgb_renderer_backend": "spear_unreal_native_apartment",
            "camera_contract_id": "lead_a_native_spear_bp_camera_sensor_v1",
            "semantic_id_namespace": (
                "lead_a_native_spear_depth_equal_target_only_instances_v1"
            ),
            "resolution_hw": [RUNNER.HEIGHT, RUNNER.WIDTH],
            "frame_indices": [args.frame_index],
            "camera_pose_ids": [frame["camera_state"]["pose_hash"]],
        }
        truth = compile_depth_pixel_visibility_truth(
            normal_depth_m_frames=[normal["depth_m"]],
            target_only_depth_m_frames_by_instance={
                instance_id: [depth]
                for instance_id, depth in target_only_depths.items()
            },
            semantic_ids_by_instance=semantic_ids,
            normal_context={"pass_kind": "modal_scene", **contexts},
            target_only_contexts_by_instance={
                instance_id: {
                    "pass_kind": "target_only",
                    "target_instance_id": instance_id,
                    **contexts,
                }
                for instance_id in target_only_depths
            },
            target_only_background_depth_m=TARGET_ONLY_BACKGROUND_DEPTH_M,
            absolute_tolerance_m=DEPTH_ABSOLUTE_TOLERANCE_M,
            relative_tolerance=DEPTH_RELATIVE_TOLERANCE,
        )

        np.savez_compressed(
            args.output / "native_modal_target_only_depth.npz",
            rgb_bgr_uint8=normal["rgb"],
            depth_m_float32=normal["depth_m"],
            normal_object_ids_uint32=normal["raw_object_ids_uint32"],
            modal_semantic_int32=modal_mask,
            target_only_source1_int32=target_only_masks["source1"],
            target_only_source2_int32=target_only_masks["source2"],
            target_only_source1_depth_m_float32=target_only_depths["source1"],
            target_only_source2_depth_m_float32=target_only_depths["source2"],
            target_only_source1_object_ids_uint32=target_only_raw_ids["source1"],
            target_only_source2_object_ids_uint32=target_only_raw_ids["source2"],
            source1_depth_residual_m_float32=depth_residuals["source1"],
            source2_depth_residual_m_float32=depth_residuals["source2"],
        )
        cv2.imwrite(str(args.output / "rgb.png"), normal["rgb"])
        modal_vis = np.zeros((RUNNER.HEIGHT, RUNNER.WIDTH, 3), dtype=np.uint8)
        modal_vis[modal_mask == 1] = (64, 220, 96)
        modal_vis[modal_mask == 2] = (240, 160, 48)
        cv2.imwrite(str(args.output / "modal_targets.png"), modal_vis)
        depth = normal["depth_m"]
        finite = depth[np.isfinite(depth) & (depth > 0.0)]
        _require(finite.size > 0, "native depth has no finite positive pixels")
        depth_vis = np.clip(depth, 0.0, float(np.percentile(finite, 99)))
        depth_vis = cv2.applyColorMap(
            np.asarray(255.0 * depth_vis / max(float(depth_vis.max()), 1.0e-9), dtype=np.uint8),
            cv2.COLORMAP_VIRIDIS,
        )
        cv2.imwrite(str(args.output / "depth_m_visualized.png"), depth_vis)
        _write_json(args.output / "pixel_visibility_truth.json", truth)
        evidence = {
            "schema": SCHEMA,
            "status": "pass",
            "qualification_claim": False,
            "claim_boundary": (
                "one-frame native SPEAR technical spike; proves same-renderer mask "
                "feasibility but does not yet bind a 75-frame Fact table"
            ),
            "scenario_id": args.scenario_id,
            "frame_index": args.frame_index,
            "pts_ticks": frame["pts_ticks"],
            "camera_pose_hash": frame["camera_state"]["pose_hash"],
            "camera_intrinsics": _camera_intrinsics(
                width=RUNNER.WIDTH, height=RUNNER.HEIGHT, hfov_deg=105.0
            ),
            "normal_runtime_readback": normal_readback,
            "target_only_runtime_readbacks": target_readbacks,
            "modal_descriptor_indices": modal_descriptor_indices,
            "modal_descriptor_raw_ids": modal_descriptor_raw_ids,
            "direct_modal_pixel_counts": direct_modal_pixel_counts,
            "visibility_derivation": {
                "method": (
                    "same_camera_normal_vs_target_only_metric_depth_v1"
                ),
                "target_footprint": (
                    "target_only_depth_below_65504m_background_sentinel"
                ),
                "depth_tolerance_m": (
                    f"{DEPTH_ABSOLUTE_TOLERANCE_M} + "
                    f"{DEPTH_RELATIVE_TOLERANCE} * target_depth_m"
                ),
                "direct_skeletal_object_ids_usable": all(
                    count > 0 for count in direct_modal_pixel_counts.values()
                ),
                "cooked_renderer_limitation": (
                    "M_Emissive is not compiled with bUsedWithSkeletalMesh; "
                    "dynamic skeletal proxies fall back to non-registered "
                    "material pixels, so direct per-instance raw IDs are not "
                    "the modal authority in this spike"
                ),
                "per_instance": {
                    instance_id: {
                        "target_pixels": int(
                            np.count_nonzero(
                                target_only_masks[instance_id]
                                == semantic_ids[instance_id]
                            )
                        ),
                        "modal_visible_pixels": int(
                            np.count_nonzero(modal_mask == semantic_ids[instance_id])
                        ),
                        "target_only_object_id_foreground_pixels": int(
                            np.count_nonzero(
                                target_only_object_id_foregrounds[instance_id]
                            )
                        ),
                    }
                    for instance_id in sorted(stable_names)
                },
            },
            "pixel_visibility": truth,
            "depth": {
                "units": "meters",
                "shape_hw": list(depth.shape),
                "finite_positive_fraction": float(
                    np.mean(np.isfinite(depth) & (depth > 0.0))
                ),
                "minimum_finite_positive_m": float(finite.min()),
                "maximum_finite_positive_m": float(finite.max()),
            },
            "artifacts": {
                "arrays": "native_modal_target_only_depth.npz",
                "rgb": "rgb.png",
                "modal_targets": "modal_targets.png",
                "depth_visualization": "depth_m_visualized.png",
                "pixel_visibility_truth": "pixel_visibility_truth.json",
            },
        }
        _write_json(args.output / "evidence.json", evidence)
        print(f"SPEAR_NATIVE_PIXEL_SPIKE_OK output={args.output}", flush=True)
        return args.output / "evidence.json"
    finally:
        if runtimes:
            try:
                RUNNER._destroy_runtime_actors(instance, runtimes)
            except Exception:
                pass
        instance.close(force=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-plan", required=True, type=Path)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--spear-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rpc-port", type=int, default=39471)
    parser.add_argument("--graphics-adapter", type=int, default=2)
    parser.add_argument("--warmup-frames", type=int, default=40)
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

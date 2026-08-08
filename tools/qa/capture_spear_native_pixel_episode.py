#!/usr/bin/env python3
"""Capture a full native SPEAR RGB/depth/pixel-truth Episode.

The tool replays every formal frame from an existing native Apartment suite
plan.  A normal pass records RGB, metric depth and raw object IDs.  One
show-only pass per controlled source records target-only metric depth from the
same ``BP_CameraSensor``.  The QA pixel compiler derives visibility from exact
normal-vs-target depth agreement and verifies renderer/camera/frame identity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.qa.pixel_visibility import (  # noqa: E402
    PIXEL_VISIBILITY_DEPTH_AUTHORITY,
    compile_depth_pixel_visibility_truth,
)


SPIKE_PATH = REPOSITORY / "tools/qa/spike_spear_native_pixel_visibility.py"
SPIKE_SPEC = importlib.util.spec_from_file_location(
    "lead_a_native_pixel_spike", SPIKE_PATH
)
if SPIKE_SPEC is None or SPIKE_SPEC.loader is None:
    raise RuntimeError(f"cannot import {SPIKE_PATH}")
SPIKE = importlib.util.module_from_spec(SPIKE_SPEC)
SPIKE_SPEC.loader.exec_module(SPIKE)
RUNNER = SPIKE.RUNNER

SCHEMA = "avengine_qa_native_spear_pixel_episode_v1"
ABSOLUTE_TOLERANCE_M = 0.01
RELATIVE_TOLERANCE = 0.002
TARGET_ONLY_BACKGROUND_DEPTH_M = 65504.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_artifact_record(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"artifact file is missing: {path}")
    return {
        "kind": "file",
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _directory_artifact_record(path: Path) -> dict[str, Any]:
    _require(path.is_dir(), f"artifact directory is missing: {path}")
    inventory = [
        {
            "relative_path": str(file.relative_to(path)),
            "size_bytes": file.stat().st_size,
            "sha256": _sha256(file),
        }
        for file in sorted(item for item in path.rglob("*") if item.is_file())
    ]
    _require(inventory, f"artifact directory is empty: {path}")
    return {
        "kind": "directory",
        "path": str(path.resolve()),
        "file_count": len(inventory),
        "total_size_bytes": sum(item["size_bytes"] for item in inventory),
        "inventory": inventory,
        "inventory_root_sha256": _canonical_json_sha256(inventory),
    }


def _raw_object_ids(component: Any) -> np.ndarray:
    bgra = component.read_pixels()["arrays"]["data"].copy()
    return (
        np.ascontiguousarray(bgra)
        .view(np.uint32)
        .reshape(bgra.shape[:2])
        & np.uint32(0x00FFFFFF)
    )


def _depth_native(component: Any) -> np.ndarray:
    return component.read_pixels()["arrays"]["data"][:, :, 0].copy()


def _rgb_bgr(component: Any) -> np.ndarray:
    return component.read_pixels()["arrays"]["data"][:, :, :3].copy()


def _safe_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "rawId",
        "actorStableName",
        "actorName",
        "componentStableName",
        "componentName",
        "materialName",
    ):
        value = descriptor.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def _descriptor_raw_ids(
    descriptors: Sequence[Mapping[str, Any]], stable_name: str
) -> list[int]:
    return [
        int(descriptor["rawId"])
        for descriptor in descriptors
        if descriptor.get("actorStableName") == stable_name
    ]


def _derive_masks(
    *,
    normal_depths: Sequence[np.ndarray],
    target_depths_by_instance: Mapping[str, Sequence[np.ndarray]],
    semantic_ids: Mapping[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frame_count = len(normal_depths)
    height, width = normal_depths[0].shape
    modal = np.zeros((frame_count, height, width), dtype=np.uint8)
    target_masks = {
        instance_id: np.zeros((frame_count, height, width), dtype=np.uint8)
        for instance_id in semantic_ids
    }
    for frame_index, normal_value in enumerate(normal_depths):
        normal = np.asarray(normal_value, dtype=np.float32)
        best_residual = np.full((height, width), np.inf, dtype=np.float32)
        for instance_id in sorted(semantic_ids):
            target = np.asarray(
                target_depths_by_instance[instance_id][frame_index],
                dtype=np.float32,
            )
            footprint = target < TARGET_ONLY_BACKGROUND_DEPTH_M
            target_masks[instance_id][frame_index][footprint] = semantic_ids[
                instance_id
            ]
            residual = np.abs(normal - target)
            tolerance = ABSOLUTE_TOLERANCE_M + RELATIVE_TOLERANCE * target
            visible = footprint & (residual <= tolerance)
            wins = visible & (residual < best_residual)
            modal[frame_index][wins] = semantic_ids[instance_id]
            best_residual[wins] = residual[wins]
    return modal, target_masks


def _maximum_readback_drift(
    normal_readbacks: Sequence[Mapping[str, Any]],
    target_readbacks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    maximum_location_cm = 0.0
    maximum_rotation_deg = 0.0
    for readbacks in target_readbacks.values():
        _require(
            len(readbacks) == len(normal_readbacks),
            "target-only runtime readback count differs from normal pass",
        )
        for normal, target in zip(normal_readbacks, readbacks):
            _require(
                normal["camera"]["expected_pose_hash"]
                == target["camera"]["expected_pose_hash"],
                "target-only camera pose hash differs from normal pass",
            )
            for owner in ["camera", *sorted(normal["actors"])]:
                normal_value = normal[owner] if owner == "camera" else normal["actors"][owner]
                target_value = target[owner] if owner == "camera" else target["actors"][owner]
                location = max(
                    abs(float(left) - float(right))
                    for left, right in zip(
                        normal_value["location_cm"], target_value["location_cm"]
                    )
                )
                rotation = max(
                    abs(
                        ((float(left) - float(right) + 180.0) % 360.0) - 180.0
                    )
                    for left, right in zip(
                        normal_value["rotation_deg"], target_value["rotation_deg"]
                    )
                )
                maximum_location_cm = max(maximum_location_cm, location)
                maximum_rotation_deg = max(maximum_rotation_deg, rotation)
    _require(maximum_location_cm <= 1.0e-4, "target pass location drift")
    _require(maximum_rotation_deg <= 1.0e-4, "target pass rotation drift")
    return {
        "normal_frame_count": len(normal_readbacks),
        "target_pass_count": len(target_readbacks),
        "maximum_location_drift_cm": maximum_location_cm,
        "maximum_rotation_drift_deg": maximum_rotation_deg,
    }


def _run_checked(command: Sequence[str]) -> None:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed: "
            + " ".join(command)
            + "\n"
            + completed.stdout
            + completed.stderr
        )


def _ffprobe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def run(args: argparse.Namespace) -> Path:
    import cv2

    suite = json.loads(args.suite_plan.read_text(encoding="utf-8"))
    scenarios = {item["scenario_id"]: item for item in suite["scenarios"]}
    _require(args.scenario_id in scenarios, "scenario is absent from suite plan")
    scenario = scenarios[args.scenario_id]
    all_frames = scenario["plan"]["frames"]
    _require(
        len(all_frames) == RUNNER.FRAME_COUNT,
        "scenario is not a 75-frame Episode",
    )
    selected_indices = (
        list(range(RUNNER.FRAME_COUNT))
        if args.frame_index is None
        else list(args.frame_index)
    )
    _require(
        selected_indices
        and len(selected_indices) == len(set(selected_indices))
        and all(0 <= index < RUNNER.FRAME_COUNT for index in selected_indices),
        "selected formal frame indices must be unique values in [0,74]",
    )
    frames = [all_frames[index] for index in selected_indices]
    _require(args.audio_wav.is_file(), "authoritative binaural WAV is missing")
    args.output.mkdir(parents=True)
    rgb_directory = args.output / "rgb_frames"
    rgb_directory.mkdir()

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
    normal_depths: list[np.ndarray] = []
    normal_object_ids: list[np.ndarray] = []
    normal_readbacks: list[dict[str, Any]] = []
    target_depths: dict[str, list[np.ndarray]] = {}
    target_readbacks: dict[str, list[dict[str, Any]]] = {}
    target_object_id_foreground_counts: dict[str, list[int]] = {}
    descriptors: list[dict[str, Any]] = []
    stable_names: dict[str, str] = {}
    direct_modal_pixel_counts: dict[str, list[int]] = {}
    try:
        with instance.begin_frame():
            camera, components = SPIKE._spawn_multimodal_camera(game)
            runtimes = RUNNER._spawn_runtime_actors(game, scenario, spear_root)
            for actor_id, runtime in runtimes.items():
                stable_name = f"lead_a_native_{actor_id}"
                game.unreal_service.set_stable_name_for_actor(
                    actor=runtime["visual_actor"], stable_name=stable_name
                )
                stable_names[actor_id.removesuffix("_actor")] = stable_name
            SPIKE._apply_exact_frame(
                camera=camera, runtimes=runtimes, frame=frames[0]
            )
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                bPaused=False
            )
        with instance.end_frame():
            pass
        instance.step(num_frames=args.warmup_frames)

        with instance.begin_frame():
            game.segmentation_service.initialize()
            components["depth"].PrimitiveRenderMode = "PRM_RenderScenePrimitives"
            components["depth"].ShowOnlyActors = []
            SPIKE._apply_exact_frame(
                camera=camera, runtimes=runtimes, frame=frames[0]
            )
        with instance.end_frame():
            pass
        instance.step(num_frames=2)

        with instance.begin_frame():
            raw_descriptors = (
                game.segmentation_service.get_mesh_proxy_geometry_descs(
                    include_debug_info=False, as_global=True
                )
            )
        with instance.end_frame():
            pass
        descriptors = [_safe_descriptor(item) for item in raw_descriptors]
        raw_ids_by_instance = {
            instance_id: _descriptor_raw_ids(raw_descriptors, stable_name)
            for instance_id, stable_name in stable_names.items()
        }
        _require(
            all(raw_ids_by_instance.values()),
            "dynamic source proxy descriptors are missing",
        )
        direct_modal_pixel_counts = {
            instance_id: [] for instance_id in stable_names
        }

        for capture_index, frame in enumerate(frames):
            with instance.begin_frame():
                readback = SPIKE._apply_exact_frame(
                    camera=camera, runtimes=runtimes, frame=frame
                )
            with instance.end_frame():
                rgb = _rgb_bgr(components["rgb"])
                depth = _depth_native(components["depth"])
                raw_ids = _raw_object_ids(components["object_ids"])
            _require(
                cv2.imwrite(
                    str(rgb_directory / f"frame_{capture_index:06d}.png"), rgb
                ),
                f"could not write RGB frame {frame['frame_index']}",
            )
            normal_depths.append(depth)
            normal_object_ids.append(raw_ids)
            normal_readbacks.append(readback)
            for instance_id, expected_raw_ids in raw_ids_by_instance.items():
                direct_modal_pixel_counts[instance_id].append(
                    int(
                        np.count_nonzero(
                            np.isin(
                                raw_ids,
                                np.asarray(expected_raw_ids, dtype=np.uint32),
                            )
                        )
                    )
                )

        for instance_id in sorted(stable_names):
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
                SPIKE._apply_exact_frame(
                    camera=camera, runtimes=runtimes, frame=frames[0]
                )
            with instance.end_frame():
                pass
            instance.step(num_frames=2)
            target_depths[instance_id] = []
            target_readbacks[instance_id] = []
            target_object_id_foreground_counts[instance_id] = []
            for frame in frames:
                with instance.begin_frame():
                    readback = SPIKE._apply_exact_frame(
                        camera=camera, runtimes=runtimes, frame=frame
                    )
                with instance.end_frame():
                    depth = _depth_native(components["depth"])
                    raw_ids = _raw_object_ids(components["object_ids"])
                target_depths[instance_id].append(depth)
                target_readbacks[instance_id].append(readback)
                target_object_id_foreground_counts[instance_id].append(
                    int(np.count_nonzero(raw_ids != 0))
                )
    finally:
        if runtimes:
            try:
                RUNNER._destroy_runtime_actors(instance, runtimes)
            except Exception:
                pass
        instance.close(force=True)

    frame_indices = [int(frame["frame_index"]) for frame in frames]
    camera_pose_ids = [frame["camera_state"]["pose_hash"] for frame in frames]
    semantic_ids = {"source1": 1, "source2": 2}
    _require(set(target_depths) == set(semantic_ids), "unexpected source slots")
    common_context = {
        "renderer_backend": "spear_unreal_native_apartment",
        "rgb_renderer_backend": "spear_unreal_native_apartment",
        "camera_contract_id": "lead_a_native_spear_bp_camera_sensor_v1",
        "semantic_id_namespace": "lead_a_native_spear_metric_depth_instances_v1",
        "resolution_hw": [RUNNER.HEIGHT, RUNNER.WIDTH],
        "frame_indices": frame_indices,
        "camera_pose_ids": camera_pose_ids,
    }
    truth = compile_depth_pixel_visibility_truth(
        normal_depth_m_frames=normal_depths,
        target_only_depth_m_frames_by_instance=target_depths,
        semantic_ids_by_instance=semantic_ids,
        normal_context={"pass_kind": "modal_scene", **common_context},
        target_only_contexts_by_instance={
            instance_id: {
                "pass_kind": "target_only",
                "target_instance_id": instance_id,
                **common_context,
            }
            for instance_id in semantic_ids
        },
        target_only_background_depth_m=TARGET_ONLY_BACKGROUND_DEPTH_M,
        absolute_tolerance_m=ABSOLUTE_TOLERANCE_M,
        relative_tolerance=RELATIVE_TOLERANCE,
    )
    _require(
        truth["authority"] == PIXEL_VISIBILITY_DEPTH_AUTHORITY,
        "pixel compiler returned the wrong authority",
    )
    modal_masks, target_masks = _derive_masks(
        normal_depths=normal_depths,
        target_depths_by_instance=target_depths,
        semantic_ids=semantic_ids,
    )
    alignment = _maximum_readback_drift(normal_readbacks, target_readbacks)

    normal_depth_array = np.stack(normal_depths)
    normal_object_id_array = np.stack(normal_object_ids)
    target_depth_arrays = {
        instance_id: np.stack(values)
        for instance_id, values in target_depths.items()
    }
    depth_path = args.output / "metric_depth_native.npz"
    np.savez_compressed(
        depth_path,
        normal_depth_m=normal_depth_array,
        target_only_source1_depth_m=target_depth_arrays["source1"],
        target_only_source2_depth_m=target_depth_arrays["source2"],
    )
    object_id_path = args.output / "normal_object_ids_uint32.npz"
    np.savez_compressed(object_id_path, normal_object_ids=normal_object_id_array)
    mask_path = args.output / "native_pixel_masks_depth_authority_v1.npz"
    np.savez_compressed(
        mask_path,
        depth_derived_modal_semantic=modal_masks,
        modal_visible_source1=modal_masks == semantic_ids["source1"],
        modal_visible_source2=modal_masks == semantic_ids["source2"],
        target_only_source1=target_masks["source1"],
        target_only_source2=target_masks["source2"],
    )
    truth_path = args.output / "pixel_visibility_truth.json"
    _write_json(truth_path, truth)
    readbacks_path = args.output / "runtime_readbacks.json"
    _write_json(
        readbacks_path,
        {
            "schema": "avengine_native_spear_multimodal_runtime_readbacks_v1",
            "normal": normal_readbacks,
            "target_only": target_readbacks,
        },
    )
    descriptor_path = args.output / "normal_object_id_descriptors.json"
    _write_json(
        descriptor_path,
        {
            "schema": "avengine_native_spear_object_id_descriptors_v1",
            "descriptors": descriptors,
        },
    )

    visual_path = args.output / "native_rgb_visual_only.mp4"
    muxed_path = args.output / "native_rgb_binaural.mp4"
    _run_checked(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(RUNNER.FPS),
            "-start_number",
            "0",
            "-i",
            str(rgb_directory / "frame_%06d.png"),
            "-frames:v",
            str(len(frames)),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(RUNNER.FPS),
            str(visual_path),
        ]
    )
    _run_checked(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(visual_path),
            "-i",
            str(args.audio_wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "2",
            "-shortest",
            str(muxed_path),
        ]
    )
    video_probe = _ffprobe(muxed_path)
    intrinsics = SPIKE._camera_intrinsics(
        width=RUNNER.WIDTH, height=RUNNER.HEIGHT, hfov_deg=105.0
    )
    depth_statistics = {
        "dtype": str(normal_depth_array.dtype),
        "shape_nhw": list(normal_depth_array.shape),
        "units": "meters",
        "finite_positive_fraction": float(
            np.mean(np.isfinite(normal_depth_array) & (normal_depth_array > 0.0))
        ),
        "minimum_m": float(normal_depth_array.min()),
        "maximum_m": float(normal_depth_array.max()),
        "per_frame_minimum_m": [
            float(np.min(frame)) for frame in normal_depth_array
        ],
        "per_frame_maximum_m": [
            float(np.max(frame)) for frame in normal_depth_array
        ],
    }
    artifact_paths = {
        "rgb_frames": rgb_directory,
        "native_rgb_visual_only": visual_path,
        "native_rgb_binaural": muxed_path,
        "metric_depth": depth_path,
        "normal_object_ids": object_id_path,
        "pixel_masks": mask_path,
        "pixel_visibility_truth": truth_path,
        "runtime_readbacks": readbacks_path,
        "object_id_descriptors": descriptor_path,
    }
    artifact_records = {
        name: (
            _directory_artifact_record(path)
            if path.is_dir()
            else _file_artifact_record(path)
        )
        for name, path in artifact_paths.items()
    }
    manifest = {
        "schema": SCHEMA,
        "status": "pass",
        "benchmark_qualification_claim": False,
        "native_pixel_fact_binding_claim": True,
        "claim_boundary": (
            "native SPEAR Apartment RGB/depth/pixel evidence bound to an existing "
            "controlled Episode; this optional comparison render is not a release "
            "qualification claim"
        ),
        "scenario_id": args.scenario_id,
        "authoritative_capture_request": scenario.get(
            "authoritative_capture_request"
        ),
        "static_camera_upgrade": scenario.get("static_camera_upgrade"),
        "native_map": suite["native_map"],
        "frame_contract": {
            "frame_count": len(frames),
            "formal_episode_frame_count": len(all_frames),
            "captured_frame_indices": frame_indices,
            "frame_rate_hz": RUNNER.FPS,
            "resolution_hw": [RUNNER.HEIGHT, RUNNER.WIDTH],
            "camera_pose_ids": camera_pose_ids,
        },
        "camera_intrinsics": intrinsics,
        "runtime_alignment": alignment,
        "pixel_visibility": {
            "authority": truth["authority"],
            "mask_array_contract": {
                "depth_derived_modal_semantic": (
                    "uint8 semantic IDs derived from metric depth; not direct "
                    "raw skeletal object IDs"
                ),
                "modal_visible_source1": "boolean depth-derived modal mask",
                "modal_visible_source2": "boolean depth-derived modal mask",
                "target_only_source1": "uint8 semantic-ID target footprint",
                "target_only_source2": "uint8 semantic-ID target footprint",
            },
            "per_instance_state_counts": {
                instance_id: entry["state_counts"]
                for instance_id, entry in truth["per_instance"].items()
            },
        },
        "direct_object_id_limitation": {
            "dynamic_skeletal_raw_ids_usable": all(
                any(count > 0 for count in counts)
                for counts in direct_modal_pixel_counts.values()
            ),
            "per_instance_direct_modal_pixel_counts": direct_modal_pixel_counts,
            "reason": (
                "cooked M_Emissive lacks bUsedWithSkeletalMesh; static environment "
                "object IDs remain retained for occluder analysis"
            ),
            "target_only_object_id_foreground_counts": (
                target_object_id_foreground_counts
            ),
        },
        "metric_depth": depth_statistics,
        "audio": {
            "authoritative_wav": str(args.audio_wav.resolve()),
            "sha256": _sha256(args.audio_wav),
        },
        "artifacts": {
            name: str(path.resolve()) for name, path in artifact_paths.items()
        },
        "artifact_records": artifact_records,
        "sha256": {
            name: record["sha256"]
            for name, record in artifact_records.items()
            if record["kind"] == "file"
        },
        "ffprobe": video_probe,
    }
    manifest_path = args.output / "manifest.json"
    _write_json(manifest_path, manifest)
    print(f"SPEAR_NATIVE_PIXEL_EPISODE_OK output={args.output}", flush=True)
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-plan", required=True, type=Path)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--audio-wav", required=True, type=Path)
    parser.add_argument("--spear-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rpc-port", type=int, default=39474)
    parser.add_argument("--graphics-adapter", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=40)
    parser.add_argument(
        "--frame-index",
        action="append",
        type=int,
        help="Capture only this formal Episode frame; repeat for a sparse capture.",
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

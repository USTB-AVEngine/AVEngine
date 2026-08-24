#!/usr/bin/env python3
"""Render one lightweight native Habitat camera-pose probe.

This is deliberately not formal M1 release evidence.  It validates the room
and capture-request contracts, loads the declared Habitat scene and NavMesh,
places the co-located camera/listener rig at the requested transform, renders
the RGB/depth/semantic sensors once, and records the floor snap beneath the
camera.  It skips repository-wide provenance scans so interactive camera
configuration checks do not inherit release-audit latency.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from avengine.contracts.json_io import file_record, load_json, write_json
from avengine.m1.contracts import load_and_validate_inputs
from avengine.m1.evidence import array_sha256
from avengine.m1.habitat_capture import (
    _make_configuration,
    _numpy_quaternion,
    prepare_installed_habitat_runtime,
)


def _finite_xyz(value: Any) -> list[float]:
    return [float(value[index]) for index in range(3)]


def run_probe(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    inputs = load_and_validate_inputs(args.room, args.request)
    request = inputs.request
    rig = request["primary_camera_rig"]
    transform = rig["world_from_rig"]

    runtime = prepare_installed_habitat_runtime(
        runtime_prefix=args.runtime_prefix,
        magnum_python_site=args.magnum_python_site,
        rlr_sdk_root=args.rlr_sdk_root,
        pbr_asset_root=args.pbr_asset_root,
        mp3d_root=args.mp3d_root,
        allow_mp3d_environment=args.mp3d_root is not None,
    )
    habitat_sim = runtime.habitat_sim
    configuration, modality_to_uuid, _, resolved_scene = _make_configuration(
        inputs,
        None,
        output,
        mp3d_root=runtime.mp3d_root,
        include_audio_sensor=False,
        physics_config_path=runtime.physics_config_path,
    )
    state = habitat_sim.AgentState()
    state.position = np.asarray(transform["translation_m"], dtype=np.float64)
    state.rotation = _numpy_quaternion(transform["rotation_xyzw"], runtime.quaternion)

    with habitat_sim.Simulator(configuration) as simulator:
        navmesh = resolved_scene["navmesh"]
        if navmesh is None or not Path(navmesh).is_file():
            raise FileNotFoundError("camera pose probe requires the declared NavMesh")
        if not simulator.pathfinder.load_nav_mesh(str(navmesh)):
            raise RuntimeError("Habitat failed to load the declared NavMesh")

        camera_position = np.asarray(transform["translation_m"], dtype=np.float64)
        if (args.expected_floor_point_m is None) != (
            args.expected_floor_island_id is None
        ):
            raise ValueError(
                "expected floor point and island ID must be provided together"
            )
        expected_floor_point = (
            None
            if args.expected_floor_point_m is None
            else np.asarray(args.expected_floor_point_m, dtype=np.float64)
        )
        if expected_floor_point is not None and (
            expected_floor_point.shape != (3,)
            or not np.all(np.isfinite(expected_floor_point))
        ):
            raise ValueError("expected floor point must contain three finite numbers")
        floor_probe = (
            camera_position if expected_floor_point is None else expected_floor_point
        )
        floor_point = np.asarray(
            simulator.pathfinder.snap_point(floor_probe), dtype=np.float64
        )
        if floor_point.shape != (3,) or not np.isfinite(floor_point).all():
            raise RuntimeError("camera position has no finite NavMesh floor point")
        horizontal_snap_distance = float(
            np.linalg.norm(floor_point[[0, 2]] - camera_position[[0, 2]])
        )
        floor_y_error = (
            0.0
            if expected_floor_point is None
            else abs(float(floor_point[1] - expected_floor_point[1]))
        )
        floor_island_id = int(simulator.pathfinder.get_island(floor_point))
        floor_island_passed = (
            True
            if args.expected_floor_island_id is None
            else floor_island_id == int(args.expected_floor_island_id)
        )
        expected_floor_passed = bool(
            expected_floor_point is None
            or (
                simulator.pathfinder.is_navigable(
                    expected_floor_point, float(args.max_floor_y_error_m)
                )
                and floor_y_error <= float(args.max_floor_y_error_m)
                and floor_island_passed
            )
        )
        eye_height = float(camera_position[1] - floor_point[1])
        navigation_passed = (
            horizontal_snap_distance <= float(args.max_horizontal_snap_m)
            and float(args.min_eye_height_m)
            <= eye_height
            <= float(args.max_eye_height_m)
            and expected_floor_passed
        )

        simulator.seed(int(request["seed"]))
        simulator.initialize_agent(0, state)
        sensor_uuids = [
            modality_to_uuid[modality] for modality in ("rgb", "depth", "semantic")
        ]
        wrappers = [simulator.sensors[uuid] for uuid in sensor_uuids]
        observations = simulator.render_sensors(wrappers)

    rgb = np.ascontiguousarray(observations[modality_to_uuid["rgb"]])
    depth = np.ascontiguousarray(observations[modality_to_uuid["depth"]])
    semantic = np.ascontiguousarray(observations[modality_to_uuid["semantic"]])
    rgb_path = output / "rgb.png"
    depth_path = output / "depth.npy"
    semantic_path = output / "semantic.npy"
    Image.fromarray(rgb[:, :, :3].astype(np.uint8), mode="RGB").save(rgb_path)
    np.save(depth_path, depth, allow_pickle=False)
    np.save(semantic_path, semantic, allow_pickle=False)

    receipt = {
        "schema": "avengine_native_camera_pose_probe_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "native_interactive_probe_not_release_evidence",
        "status": "pass" if navigation_passed else "fail",
        "room_id": inputs.room["room_id"],
        "request_id": request["request_id"],
        "installed_runtime_inputs": {
            "runtime_prefix": str(runtime.prefix),
            "magnum_python_site": str(runtime.magnum_python_site),
            "rlr_sdk_root": (
                None if args.rlr_sdk_root is None else str(args.rlr_sdk_root.resolve())
            ),
            "pbr_asset_root": (
                None if runtime.pbr_asset_root is None else str(runtime.pbr_asset_root)
            ),
            "mp3d_root": (
                None if runtime.mp3d_root is None else str(runtime.mp3d_root)
            ),
            "physics_config_path": str(runtime.physics_config_path),
            "include_audio_sensor": False,
        },
        "camera_listener_rig": {
            "rig_id": rig["rig_id"],
            "view_id": rig["view_id"],
            "world_from_rig": transform,
            "listener_co_located": request["listener"]["rig_from_listener"]
            == {
                "translation_m": [0, 0, 0],
                "rotation_xyzw": [0, 0, 0, 1],
            },
        },
        "navigation": {
            "floor_authority": (
                "camera_eye_nearest_navmesh"
                if expected_floor_point is None
                else "explicit_builder_expected_floor_point"
            ),
            "camera_position_m": _finite_xyz(camera_position),
            "expected_floor_point_m": (
                None
                if expected_floor_point is None
                else _finite_xyz(expected_floor_point)
            ),
            "floor_point_m": _finite_xyz(floor_point),
            "horizontal_snap_distance_m": horizontal_snap_distance,
            "floor_y_error_m": floor_y_error,
            "floor_island_id": floor_island_id,
            "expected_floor_island_id": args.expected_floor_island_id,
            "expected_floor_passed": expected_floor_passed,
            "eye_height_above_navmesh_m": eye_height,
            "thresholds": {
                "max_horizontal_snap_m": float(args.max_horizontal_snap_m),
                "max_floor_y_error_m": float(args.max_floor_y_error_m),
                "min_eye_height_m": float(args.min_eye_height_m),
                "max_eye_height_m": float(args.max_eye_height_m),
            },
            "passed": navigation_passed,
        },
        "observations": {
            "rgb": {
                "shape": list(rgb.shape),
                "dtype": str(rgb.dtype),
                "raw_array_sha256": array_sha256(modality_to_uuid["rgb"], rgb),
                "artifact": file_record(rgb_path, relative_to=output),
            },
            "depth": {
                "shape": list(depth.shape),
                "dtype": str(depth.dtype),
                "finite_fraction": float(np.isfinite(depth).mean()),
                "raw_array_sha256": array_sha256(modality_to_uuid["depth"], depth),
                "artifact": file_record(depth_path, relative_to=output),
            },
            "semantic": {
                "shape": list(semantic.shape),
                "dtype": str(semantic.dtype),
                "raw_array_sha256": array_sha256(
                    modality_to_uuid["semantic"], semantic
                ),
                "artifact": file_record(semantic_path, relative_to=output),
            },
        },
    }
    if not all(
        math.isfinite(value)
        for value in (
            horizontal_snap_distance,
            eye_height,
            receipt["observations"]["depth"]["finite_fraction"],
        )
    ):
        raise RuntimeError("camera pose probe produced a non-finite measurement")
    receipt_path = output / "receipt.json"
    write_json(receipt_path, receipt)
    return receipt_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--runtime-prefix", type=Path, required=True)
    parser.add_argument("--magnum-python-site", type=Path)
    parser.add_argument("--rlr-sdk-root", type=Path)
    parser.add_argument("--pbr-asset-root", type=Path)
    parser.add_argument("--mp3d-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-horizontal-snap-m", type=float, default=0.15)
    parser.add_argument("--expected-floor-point-m", nargs=3, type=float)
    parser.add_argument("--expected-floor-island-id", type=int)
    parser.add_argument("--max-floor-y-error-m", type=float, default=0.25)
    parser.add_argument("--min-eye-height-m", type=float, default=0.8)
    parser.add_argument("--max-eye-height-m", type=float, default=2.2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    receipt = run_probe(parse_args(argv))
    if load_json(receipt).get("status") != "pass":
        print(f"NATIVE_CAMERA_POSE_PROBE_FAIL receipt={receipt}", flush=True)
        return 2
    print(f"NATIVE_CAMERA_POSE_PROBE_OK receipt={receipt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

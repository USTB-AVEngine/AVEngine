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

from avengine.contracts.json_io import file_record, write_json
from avengine.m1.contracts import load_and_validate_inputs
from avengine.m1.evidence import array_sha256
from avengine.m1.habitat_capture import (
    _import_habitat,
    _make_configuration,
    _numpy_quaternion,
)


def _finite_xyz(value: Any) -> list[float]:
    return [float(value[index]) for index in range(3)]


def run_probe(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.mkdir(parents=True)
    inputs = load_and_validate_inputs(args.room, args.request)
    runtime = args.runtime_root.resolve()
    request = inputs.request
    rig = request["primary_camera_rig"]
    transform = rig["world_from_rig"]

    qt, habitat_sim, _, _ = _import_habitat()
    configuration, modality_to_uuid, _, resolved_scene = _make_configuration(
        inputs, runtime, output
    )
    state = habitat_sim.AgentState()
    state.position = np.asarray(transform["translation_m"], dtype=np.float64)
    state.rotation = _numpy_quaternion(transform["rotation_xyzw"], qt)

    with habitat_sim.Simulator(configuration) as simulator:
        navmesh = resolved_scene["navmesh"]
        if navmesh is None or not Path(navmesh).is_file():
            raise FileNotFoundError("camera pose probe requires the declared NavMesh")
        if not simulator.pathfinder.load_nav_mesh(str(navmesh)):
            raise RuntimeError("Habitat failed to load the declared NavMesh")

        camera_position = np.asarray(transform["translation_m"], dtype=np.float64)
        floor_point = np.asarray(
            simulator.pathfinder.snap_point(camera_position), dtype=np.float64
        )
        if floor_point.shape != (3,) or not np.isfinite(floor_point).all():
            raise RuntimeError("camera position has no finite NavMesh floor point")
        horizontal_snap_distance = float(
            np.linalg.norm(floor_point[[0, 2]] - camera_position[[0, 2]])
        )
        eye_height = float(camera_position[1] - floor_point[1])
        navigation_passed = (
            horizontal_snap_distance <= float(args.max_horizontal_snap_m)
            and float(args.min_eye_height_m)
            <= eye_height
            <= float(args.max_eye_height_m)
        )

        simulator.seed(int(request["seed"]))
        simulator.initialize_agent(0, state)
        sensor_uuids = [
            modality_to_uuid[modality]
            for modality in ("rgb", "depth", "semantic")
        ]
        wrappers = [simulator.sensors[uuid] for uuid in sensor_uuids]
        observations = simulator.render_sensors(wrappers)

    rgb = np.ascontiguousarray(observations[modality_to_uuid["rgb"]])
    depth = np.ascontiguousarray(observations[modality_to_uuid["depth"]])
    semantic = np.ascontiguousarray(
        observations[modality_to_uuid["semantic"]]
    )
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
            "camera_position_m": _finite_xyz(camera_position),
            "floor_point_m": _finite_xyz(floor_point),
            "horizontal_snap_distance_m": horizontal_snap_distance,
            "eye_height_above_navmesh_m": eye_height,
            "thresholds": {
                "max_horizontal_snap_m": float(args.max_horizontal_snap_m),
                "min_eye_height_m": float(args.min_eye_height_m),
                "max_eye_height_m": float(args.max_eye_height_m),
            },
            "passed": navigation_passed,
        },
        "observations": {
            "rgb": {
                "shape": list(rgb.shape),
                "dtype": str(rgb.dtype),
                "raw_array_sha256": array_sha256(
                    modality_to_uuid["rgb"], rgb
                ),
                "artifact": file_record(rgb_path, relative_to=output),
            },
            "depth": {
                "shape": list(depth.shape),
                "dtype": str(depth.dtype),
                "finite_fraction": float(np.isfinite(depth).mean()),
                "raw_array_sha256": array_sha256(
                    modality_to_uuid["depth"], depth
                ),
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
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-horizontal-snap-m", type=float, default=0.15)
    parser.add_argument("--min-eye-height-m", type=float, default=0.8)
    parser.add_argument("--max-eye-height-m", type=float, default=2.2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    receipt = run_probe(parse_args(argv))
    print(f"NATIVE_CAMERA_POSE_PROBE_OK receipt={receipt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

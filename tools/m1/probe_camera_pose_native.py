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

from avengine.contracts.json_io import load_json, write_json
from avengine.m1.contracts import load_and_validate_inputs
from avengine.m1.evidence import save_observations
from avengine.m1.habitat_capture import (
    _make_configuration,
    _numpy_quaternion,
    _save_topdown,
    prepare_installed_habitat_runtime,
)


def _finite_xyz(value: Any) -> list[float]:
    return [float(value[index]) for index in range(3)]


def _intended_floor(
    args: argparse.Namespace,
    request: dict[str, Any],
    camera_position: np.ndarray,
) -> tuple[np.ndarray | None, int | None, str, bool]:
    explicit_point = args.expected_floor_point_m
    expected_island_id = args.expected_floor_island_id
    topdown_views = [
        view for view in request.get("qa_views", []) if view.get("kind") == "topdown"
    ]
    if len(topdown_views) > 1:
        raise ValueError("native camera probe accepts at most one Topdown QA view")
    topdown_required = len(topdown_views) == 1
    if explicit_point is not None:
        if expected_island_id is None:
            raise ValueError(
                "explicit expected floor point requires an expected island ID"
            )
        expected_floor_point = np.asarray(explicit_point, dtype=np.float64)
        authority = "explicit_builder_expected_floor_point"
    elif topdown_required:
        expected_floor_point = np.asarray(
            [
                camera_position[0],
                float(topdown_views[0]["height_m"]),
                camera_position[2],
            ],
            dtype=np.float64,
        )
        authority = "request_topdown_height_m_intended_floor"
    else:
        if expected_island_id is not None:
            raise ValueError(
                "expected island ID requires an explicit point or Topdown QA view"
            )
        expected_floor_point = None
        authority = "camera_eye_nearest_navmesh"
    if expected_floor_point is not None and (
        expected_floor_point.shape != (3,)
        or not np.all(np.isfinite(expected_floor_point))
    ):
        raise ValueError("expected floor point must contain three finite numbers")
    return (
        expected_floor_point,
        expected_island_id,
        authority,
        topdown_required,
    )


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
        allow_mp3d_environment=False,
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
        (
            expected_floor_point,
            expected_floor_island_id,
            floor_authority,
            topdown_required,
        ) = _intended_floor(args, request, camera_position)
        floor_probe = (
            camera_position if expected_floor_point is None else expected_floor_point
        )
        floor_point = np.asarray(
            simulator.pathfinder.snap_point(floor_probe), dtype=np.float64
        )
        if floor_point.shape != (3,) or not np.isfinite(floor_point).all():
            raise RuntimeError("camera position has no finite NavMesh floor point")
        horizontal_snap_distance = float(
            np.linalg.norm(floor_point[[0, 2]] - floor_probe[[0, 2]])
        )
        floor_y_error = (
            0.0
            if expected_floor_point is None
            else abs(float(floor_point[1] - expected_floor_point[1]))
        )
        floor_island_id = int(simulator.pathfinder.get_island(floor_point))
        floor_island_passed = (
            True
            if expected_floor_island_id is None
            else floor_island_id == int(expected_floor_island_id)
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
        topdown_reports, topdown_checks = _save_topdown(simulator, inputs, output)

    observation_records = save_observations(observations, modality_to_uuid, output)
    topdown_passed = bool(
        not topdown_required
        or (
            len(topdown_reports) == 1
            and topdown_checks
            and all(check.get("status") == "pass" for check in topdown_checks)
        )
    )

    receipt = {
        "schema": "avengine_native_camera_pose_probe_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "native_interactive_probe_not_release_evidence",
        "status": "pass" if navigation_passed and topdown_passed else "fail",
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
            "floor_authority": floor_authority,
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
            "expected_floor_island_id": expected_floor_island_id,
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
        "observations": observation_records,
        "topdown": {
            "required": topdown_required,
            "reports": topdown_reports,
            "checks": topdown_checks,
            "passed": topdown_passed,
        },
    }
    if not all(
        math.isfinite(value)
        for value in (
            horizontal_snap_distance,
            eye_height,
            observation_records["depth"]["statistics"]["finite_fraction"],
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
    parser.add_argument(
        "--magnum-site",
        "--magnum-python-site",
        dest="magnum_python_site",
        type=Path,
    )
    parser.add_argument("--rlr-sdk-root", type=Path)
    parser.add_argument("--mp3d-root", type=Path)
    parser.add_argument(
        "--pbr-root", "--pbr-asset-root", dest="pbr_asset_root", type=Path
    )
    parser.add_argument("--expected-floor-point-m", nargs=3, type=float)
    parser.add_argument(
        "--expected-island-id",
        "--expected-floor-island-id",
        dest="expected_floor_island_id",
        type=int,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-horizontal-snap-m", type=float, default=0.15)
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

#!/usr/bin/env python3
"""Export a fresh MP3D camera-floor sample bank from live Habitat PathFinder."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from avengine.contracts.json_io import load_json, write_json
from avengine.m1.contracts import (
    load_and_validate_inputs,
    validate_loaded_scene_asset_graph,
    validate_scene_asset_graph,
)
from avengine.m1.habitat_capture import (
    _make_configuration,
    prepare_installed_habitat_runtime,
)
from avengine.m5.current_visual import _current_mp3d_room_error
from avengine.m5.mp3d_floor_samples import (
    MP3DFloorSampleError,
    export_live_mp3d_floor_samples,
)
from avengine.security import PathPolicyError, WorkspacePathPolicy, atomic_publish_directory


class MP3DFloorExportError(ValueError):
    """Explicit runtime or retained route inputs cannot produce a native export."""


def _required_file(path: Path, *, owner: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise MP3DFloorExportError(f"{owner} cannot be resolved: {path}") from exc
    if not resolved.is_file():
        raise MP3DFloorExportError(f"{owner} is not a file: {resolved}")
    return resolved


def _required_directory(path: Path, *, owner: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise MP3DFloorExportError(f"{owner} cannot be resolved: {path}") from exc
    if not resolved.is_dir():
        raise MP3DFloorExportError(f"{owner} is not a directory: {resolved}")
    return resolved


def _optional_directory(path: Path | None, *, owner: str) -> Path | None:
    return None if path is None else _required_directory(path, owner=owner)


def _fresh_output(path: Path) -> tuple[Path, WorkspacePathPolicy]:
    raw = path.expanduser()
    if not raw.is_absolute():
        raise MP3DFloorExportError("--output must be an absolute path")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise MP3DFloorExportError(f"--output parent does not exist: {raw.parent}") from exc
    if not parent.is_dir():
        raise MP3DFloorExportError(f"--output parent is not a directory: {parent}")
    output = parent / raw.name
    if os.path.lexists(output):
        raise MP3DFloorExportError(f"refusing to replace output: {output}")
    try:
        policy = WorkspacePathPolicy.from_roots((parent,))
    except PathPolicyError as exc:
        raise MP3DFloorExportError(str(exc)) from exc
    return output, policy


def _actor_routes(value: Any, *, expected_room_id: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MP3DFloorExportError("route explanation must be an object")
    if value.get("research_only") is not True or value.get("episode_counted") is not False:
        raise MP3DFloorExportError(
            "route explanation must remain research_only and episode_counted=false"
        )
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get(
        "new_request_room_id"
    ) != expected_room_id:
        raise MP3DFloorExportError("route explanation room differs from selected MP3D room")
    native = value.get("native_pathfinder")
    if not isinstance(native, Mapping) or native.get(
        "all_paths_share_single_island"
    ) is not True:
        raise MP3DFloorExportError(
            "route explanation lacks retained single-island PathFinder evidence"
        )
    realization = value.get("current_visual_realization")
    routes = (
        realization.get("actor_skin_root_paths_m")
        if isinstance(realization, Mapping)
        else None
    )
    if not isinstance(routes, Mapping) or not routes:
        raise MP3DFloorExportError("route explanation lacks actor_skin_root_paths_m")
    return routes


def run(args: argparse.Namespace) -> Path:
    room_manifest = _required_file(args.room_manifest, owner="--room-manifest")
    base_request = _required_file(args.base_m1_request, owner="--base-m1-request")
    route_path = _required_file(args.route_explanation, owner="--route-explanation")
    runtime_prefix = _required_directory(args.runtime_prefix, owner="--runtime-prefix")
    magnum_site = _required_directory(args.magnum_site, owner="--magnum-site")
    rlr_sdk_root = _required_directory(args.rlr_sdk_root, owner="--rlr-sdk-root")
    mp3d_root = _required_directory(args.mp3d_root, owner="--mp3d-root")
    pbr_root = _optional_directory(args.pbr_root, owner="--pbr-root")
    output, policy = _fresh_output(args.output)

    try:
        inputs = load_and_validate_inputs(room_manifest, base_request)
    except (OSError, ValueError) as exc:
        raise MP3DFloorExportError(f"MP3D M1 inputs are invalid: {exc}") from exc
    room_route_error = _current_mp3d_room_error(inputs)
    if room_route_error is not None:
        raise MP3DFloorExportError(room_route_error)
    room_id = str(inputs.room["room_id"])
    routes = _actor_routes(load_json(route_path), expected_room_id=room_id)
    try:
        runtime = prepare_installed_habitat_runtime(
            runtime_prefix=runtime_prefix,
            mp3d_root=mp3d_root,
            pbr_asset_root=pbr_root,
            magnum_python_site=magnum_site,
            rlr_sdk_root=rlr_sdk_root,
            allow_mp3d_environment=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise MP3DFloorExportError(f"installed Habitat runtime is unavailable: {exc}") from exc
    graph_errors = validate_scene_asset_graph(inputs, None, mp3d_root=runtime.mp3d_root)
    if graph_errors:
        raise MP3DFloorExportError(
            "MP3D scene graph failed before Simulator: " + "; ".join(graph_errors)
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        configuration, _modalities, _listener_uuid, resolved_scene = _make_configuration(
            inputs,
            None,
            staging / "runtime_scratch",
            mp3d_root=runtime.mp3d_root,
            include_audio_sensor=False,
            physics_config_path=runtime.physics_config_path,
        )
        with runtime.habitat_sim.Simulator(configuration) as simulator:
            raw_navmesh_path = resolved_scene.get("navmesh")
            if not isinstance(raw_navmesh_path, (str, os.PathLike)):
                raise MP3DFloorExportError("declared MP3D navmesh is missing")
            navmesh_path = Path(raw_navmesh_path)
            if not navmesh_path.is_file():
                raise MP3DFloorExportError("declared MP3D navmesh is missing")
            navmesh_loaded = bool(simulator.pathfinder.load_nav_mesh(str(navmesh_path)))
            if not navmesh_loaded or not bool(simulator.pathfinder.is_loaded):
                raise MP3DFloorExportError("Simulator could not load the declared MP3D navmesh")
            loaded_errors, loaded_graph = validate_loaded_scene_asset_graph(
                inputs,
                None,
                simulator,
                declared_navmesh_loaded=True,
                mp3d_root=runtime.mp3d_root,
            )
            if loaded_errors:
                raise MP3DFloorExportError(
                    "fresh Simulator room graph differs from M1 declaration: "
                    + "; ".join(loaded_errors)
                )
            try:
                document = export_live_mp3d_floor_samples(
                    simulator.pathfinder,
                    room_id=room_id,
                    actor_routes_m=routes,
                    meters_per_pixel=args.meters_per_pixel,
                    maximum_sample_count=args.maximum_sample_count,
                    minimum_sample_count=args.minimum_sample_count,
                    candidate_scan_multiplier=args.candidate_scan_multiplier,
                    maximum_horizontal_snap_error_m=(
                        args.maximum_horizontal_snap_error_m
                    ),
                    maximum_vertical_snap_error_m=args.maximum_vertical_snap_error_m,
                    minimum_clearance_m=args.minimum_clearance_m,
                    maximum_route_floor_height_span_m=(
                        args.maximum_route_floor_height_span_m
                    ),
                )
            except MP3DFloorSampleError as exc:
                raise MP3DFloorExportError(str(exc)) from exc

        document.update(
            {
                "claim_boundary": (
                    "Live Habitat PathFinder navmesh floor export only. Native "
                    "RGB/depth/semantic pixels, first-person review, audio, and "
                    "four-point mini-pilot remain pending."
                ),
                "runtime_inputs": {
                    "runtime_prefix": str(runtime.prefix),
                    "magnum_python_site": str(runtime.magnum_python_site),
                    "rlr_sdk_root": str(rlr_sdk_root),
                    "mp3d_root": str(runtime.mp3d_root),
                    "pbr_asset_root": (
                        None if runtime.pbr_asset_root is None else str(runtime.pbr_asset_root)
                    ),
                    "room_manifest": str(room_manifest),
                    "base_m1_request": str(base_request),
                    "route_explanation": str(route_path),
                    "declared_scene": loaded_graph,
                },
            }
        )
        write_json(staging / "camera_floor_samples.json", document)
        (staging / "README.md").write_text(
            "# MP3D native camera-floor samples\n\n"
            "This fresh output was exported from the declared MP3D navmesh through "
            "the live installed Habitat PathFinder. Every selected XYZ has a snap, "
            "island and clearance readback in camera_floor_samples.json. Native "
            "pixels, first-person review and the four-point mini-pilot remain pending.\n",
            encoding="utf-8",
        )
        try:
            atomic_publish_directory(policy, staging, output)
        except (FileExistsError, PathPolicyError) as exc:
            raise MP3DFloorExportError(str(exc)) from exc
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-manifest", required=True, type=Path)
    parser.add_argument("--base-m1-request", required=True, type=Path)
    parser.add_argument("--route-explanation", required=True, type=Path)
    parser.add_argument("--runtime-prefix", required=True, type=Path)
    parser.add_argument("--magnum-site", required=True, type=Path)
    parser.add_argument("--rlr-sdk-root", required=True, type=Path)
    parser.add_argument("--mp3d-root", required=True, type=Path)
    parser.add_argument("--pbr-root", type=Path)
    parser.add_argument("--meters-per-pixel", type=float, default=0.25)
    parser.add_argument("--maximum-sample-count", type=int, default=128)
    parser.add_argument("--minimum-sample-count", type=int, default=4)
    parser.add_argument("--candidate-scan-multiplier", type=int, default=16)
    parser.add_argument("--maximum-horizontal-snap-error-m", type=float, default=0.10)
    parser.add_argument("--maximum-vertical-snap-error-m", type=float, default=0.25)
    parser.add_argument("--minimum-clearance-m", type=float, default=0.05)
    parser.add_argument("--maximum-route-floor-height-span-m", type=float, default=0.50)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

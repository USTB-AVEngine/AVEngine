#!/usr/bin/env python3
"""Build a fresh CPU-only MP3D viewpoint candidate bank.

The command consumes ordinary JSON exported from prior Habitat-native route and
navmesh work.  It never imports or launches Habitat, and it never claims native
pixels or first-person certification.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from avengine.contracts.json_io import load_json, write_json
from avengine.m1.contracts import load_and_validate_inputs
from avengine.m5.current_visual import _current_mp3d_room_error
from avengine.m5.mp3d_viewpoints import select_mp3d_viewpoints
from avengine.security import (
    PathPolicyError,
    WorkspacePathPolicy,
    atomic_publish_directory,
)


class MP3DViewpointBuildError(ValueError):
    """The explicit retained inputs cannot produce one fresh candidate bank."""


def _required_file(path: Path, *, owner: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"{owner} cannot be resolved: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{owner} is not a file: {resolved}")
    return resolved


def _fresh_output(path: Path) -> tuple[Path, WorkspacePathPolicy]:
    raw = path.expanduser()
    if not raw.is_absolute():
        raise MP3DViewpointBuildError("--output must be an absolute path")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise MP3DViewpointBuildError(
            f"--output parent does not exist: {raw.parent}"
        ) from exc
    if not parent.is_dir():
        raise MP3DViewpointBuildError(f"--output parent is not a directory: {parent}")
    output = parent / raw.name
    if os.path.lexists(output):
        raise MP3DViewpointBuildError(f"refusing to replace output: {output}")
    try:
        policy = WorkspacePathPolicy.from_roots((parent,))
    except PathPolicyError as exc:
        raise MP3DViewpointBuildError(str(exc)) from exc
    return output, policy


def _route_authority(
    value: Any, *, expected_room_id: str
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise MP3DViewpointBuildError("route explanation must be an object")
    if (
        value.get("research_only") is not True
        or value.get("episode_counted") is not False
    ):
        raise MP3DViewpointBuildError(
            "route explanation must remain research_only and episode_counted=false"
        )
    provenance = value.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("new_request_room_id") != expected_room_id
    ):
        raise MP3DViewpointBuildError(
            "route explanation room differs from the selected MP3D room"
        )
    native = value.get("native_pathfinder")
    if (
        not isinstance(native, Mapping)
        or native.get("all_paths_share_single_island") is not True
    ):
        raise MP3DViewpointBuildError(
            "route explanation lacks retained single-island native PathFinder evidence"
        )
    realization = value.get("current_visual_realization")
    routes = (
        realization.get("actor_skin_root_paths_m")
        if isinstance(realization, Mapping)
        else None
    )
    if not isinstance(routes, Mapping) or not routes:
        raise MP3DViewpointBuildError("route explanation lacks actor_skin_root_paths_m")
    evidence = {
        "research_only": True,
        "episode_counted": False,
        "room_id": expected_room_id,
        "all_paths_share_single_island": True,
        "island_id": native.get("island_id"),
        "candidate_floor_navigation_reexecuted": False,
    }
    return evidence, routes


def _floor_authority(value: Any, *, expected_room_id: str) -> tuple[str, Any]:
    if not isinstance(value, Mapping):
        raise MP3DViewpointBuildError("camera-floor sample input must be an object")
    if value.get("room_id") != expected_room_id:
        raise MP3DViewpointBuildError(
            "camera-floor sample room differs from the selected MP3D room"
        )
    authority = value.get("authority")
    if not isinstance(authority, str) or not authority.strip():
        raise MP3DViewpointBuildError(
            "camera-floor sample input must name its retained authority"
        )
    samples = value.get("camera_floor_samples_m")
    if not isinstance(samples, list):
        raise MP3DViewpointBuildError(
            "camera-floor sample input lacks camera_floor_samples_m"
        )
    return authority.strip(), samples


def run(args: argparse.Namespace) -> Path:
    room_manifest = _required_file(args.room_manifest, owner="--room-manifest")
    base_request = _required_file(args.base_m1_request, owner="--base-m1-request")
    route_path = _required_file(args.route_explanation, owner="--route-explanation")
    floor_path = _required_file(
        args.camera_floor_samples, owner="--camera-floor-samples"
    )
    output, policy = _fresh_output(args.output)
    try:
        inputs = load_and_validate_inputs(room_manifest, base_request)
    except (OSError, ValueError) as exc:
        raise MP3DViewpointBuildError(f"MP3D M1 inputs are invalid: {exc}") from exc
    room_route_error = _current_mp3d_room_error(inputs)
    if room_route_error is not None:
        raise MP3DViewpointBuildError(room_route_error)
    room_id = str(inputs.room["room_id"])
    route_evidence, actor_routes = _route_authority(
        load_json(route_path), expected_room_id=room_id
    )
    floor_authority, floor_samples = _floor_authority(
        load_json(floor_path), expected_room_id=room_id
    )
    try:
        result = select_mp3d_viewpoints(
            room_id=room_id,
            room_navigation=inputs.room["navigation"],
            base_m1_request=inputs.request,
            camera_floor_samples_m=floor_samples,
            actor_routes_m=actor_routes,
            selection_count=args.selection_count,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MP3DViewpointBuildError(str(exc)) from exc

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    requests = staging / "requests"
    requests.mkdir()
    request_paths: dict[str, str] = {}
    for viewpoint in result.viewpoints:
        relative = Path("requests") / f"{viewpoint.viewpoint_id}.json"
        write_json(staging / relative, viewpoint.m1_request)
        request_paths[viewpoint.viewpoint_id] = relative.as_posix()

    record = result.to_record()
    record.update(
        {
            "inputs": {
                "room_manifest": str(room_manifest),
                "base_m1_request": str(base_request),
                "route_explanation": str(route_path),
                "camera_floor_samples": str(floor_path),
            },
            "retained_route_native_authority": route_evidence,
            "camera_floor_sample_declared_authority": floor_authority,
            "m1_capture_request_paths": request_paths,
        }
    )
    write_json(staging / "viewpoint_bank.json", record)
    (staging / "README.md").write_text(
        "# MP3D viewpoint candidates\n\n"
        "This is a CPU-only analytic candidate prefilter. It consumed retained "
        "Habitat/PathFinder-derived inputs but did not launch Habitat itself. "
        "Candidate-floor readback, native RGB/depth/semantic pixels, Topdown and "
        "first-person review, per-view occlusion checks, and every four-point "
        "mini-pilot remain pending. These candidates are research-only, are not "
        "certified viewpoints, and count as zero formal episodes.\n",
        encoding="utf-8",
    )
    try:
        atomic_publish_directory(policy, staging, output)
    except (FileExistsError, PathPolicyError) as exc:
        raise MP3DViewpointBuildError(str(exc)) from exc
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-manifest", required=True, type=Path)
    parser.add_argument("--base-m1-request", required=True, type=Path)
    parser.add_argument("--route-explanation", required=True, type=Path)
    parser.add_argument("--camera-floor-samples", required=True, type=Path)
    parser.add_argument("--selection-count", type=int, default=6)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

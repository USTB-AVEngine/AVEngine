#!/usr/bin/env python3
"""Build the complete target-agnostic MP3D camera placement catalog."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from avengine.camera_placement import (
    DEFAULT_YAW_DEGREES,
    CameraPlacementError,
    camera_height_variants,
    iter_declared_floor_camera_placements,
)
from avengine.contracts.json_io import load_json, write_json
from avengine.m1.contracts import load_and_validate_inputs
from avengine.m5.current_visual import _current_mp3d_room_error
from avengine.security import (
    PathPolicyError,
    WorkspacePathPolicy,
    atomic_publish_directory,
)


class MP3DCameraCatalogError(ValueError):
    """Declared MP3D floor authority cannot produce one camera catalog."""


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
        raise MP3DCameraCatalogError("--output must be an absolute path")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise MP3DCameraCatalogError(
            f"--output parent does not exist: {raw.parent}"
        ) from exc
    output = parent / raw.name
    if os.path.lexists(output):
        raise MP3DCameraCatalogError(f"refusing to replace output: {output}")
    try:
        policy = WorkspacePathPolicy.from_roots((parent,))
    except PathPolicyError as exc:
        raise MP3DCameraCatalogError(str(exc)) from exc
    return output, policy


def _floor_authority(value: Any, *, expected_room_id: str) -> tuple[str, list[Any]]:
    if not isinstance(value, Mapping):
        raise MP3DCameraCatalogError("camera floor export must be an object")
    if (
        value.get("room_id") != expected_room_id
        or value.get("research_only") is not True
        or value.get("episode_counted") is not False
        or value.get("native_habitat_floor_export") is not True
        or value.get("coordinate_space") != "habitat_world_xyz_m"
    ):
        raise MP3DCameraCatalogError(
            "camera floor export does not match one native research room"
        )
    authority = value.get("authority")
    samples = value.get("camera_floor_samples_m")
    if (
        not isinstance(authority, str)
        or not authority.strip()
        or not isinstance(samples, list)
    ):
        raise MP3DCameraCatalogError(
            "camera floor export lacks authority or camera_floor_samples_m"
        )
    return authority.strip(), samples


def run(args: argparse.Namespace) -> Path:
    room_path = _required_file(args.room_manifest, owner="--room-manifest")
    request_path = _required_file(args.base_m1_request, owner="--base-m1-request")
    floor_path = _required_file(
        args.camera_floor_samples, owner="--camera-floor-samples"
    )
    output, policy = _fresh_output(args.output)
    try:
        inputs = load_and_validate_inputs(room_path, request_path)
    except (OSError, ValueError) as exc:
        raise MP3DCameraCatalogError(f"MP3D M1 inputs are invalid: {exc}") from exc
    room_error = _current_mp3d_room_error(inputs)
    if room_error is not None:
        raise MP3DCameraCatalogError(room_error)
    room_id = str(inputs.room["room_id"])
    authority, floor_samples = _floor_authority(
        load_json(floor_path), expected_room_id=room_id
    )
    navigation = inputs.room.get("navigation")
    if not isinstance(navigation, Mapping):
        raise MP3DCameraCatalogError("room manifest lacks navigation")
    try:
        heights = camera_height_variants(navigation["agent_height_m"])
        placements = tuple(
            iter_declared_floor_camera_placements(
                floor_samples,
                camera_center_heights_m=heights,
                yaw_degrees=DEFAULT_YAW_DEGREES,
            )
        )
    except (KeyError, CameraPlacementError) as exc:
        raise MP3DCameraCatalogError(str(exc)) from exc
    if not placements:
        raise MP3DCameraCatalogError("declared floor authority produced no placements")
    request_id = inputs.request.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise MP3DCameraCatalogError("base M1 request_id must be nonempty")
    records = []
    for index, placement in enumerate(placements, start=1):
        records.append(
            {
                "placement_id": f"mp3d_camera_placement_{index:05d}",
                **placement.to_record(),
                "room_id": room_id,
                "base_m1_request_id": request_id,
                "semantic_room_label": None,
                "semantic_target": None,
                "source_path_join_status": "pending",
                "native_dynamic_visibility_status": "pending",
                "difficulty_status": "not_evaluated",
            }
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        write_json(
            staging / "camera_placement_catalog.json",
            {
                "artifact_status": "complete_target_agnostic_camera_placement_catalog",
                "research_only": True,
                "episode_counted": False,
                "formal_release_status": "not_run",
                "candidate_filtering": "none",
                "studio_used": False,
                "room_id": room_id,
                "inputs": {
                    "room_manifest": str(room_path),
                    "base_m1_request": str(request_path),
                    "camera_floor_samples": str(floor_path),
                },
                "floor_position_authority": authority,
                "camera_height_variants_m": {
                    height_id: height for height_id, height in heights
                },
                "yaw_degrees": list(DEFAULT_YAW_DEGREES),
                "input_floor_sample_count": len(floor_samples),
                "canonical_floor_sample_count": len(
                    {tuple(float(item) for item in point) for point in floor_samples}
                ),
                "placement_count": len(records),
                "placements": records,
                "claim_boundary": (
                    "native-exported floor positions crossed with generic camera "
                    "heights and 30-degree yaw only; semantic targets, source paths, "
                    "visibility, difficulty, ranking, native capture and certification "
                    "remain downstream"
                ),
            },
        )
        (staging / "README.md").write_text(
            "# MP3D target-agnostic camera catalog\n\n"
            "This catalog preserves every declared native floor position crossed "
            "with low/base/high camera height and twelve 30-degree yaw values. "
            "It performs no semantic, source-path, visibility, difficulty, or "
            "ranking filter and does not certify a viewpoint.\n",
            encoding="utf-8",
        )
        try:
            atomic_publish_directory(policy, staging, output)
        except (FileExistsError, PathPolicyError) as exc:
            raise MP3DCameraCatalogError(str(exc)) from exc
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room-manifest", required=True, type=Path)
    parser.add_argument("--base-m1-request", required=True, type=Path)
    parser.add_argument("--camera-floor-samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

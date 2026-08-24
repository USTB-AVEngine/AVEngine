#!/usr/bin/env python3
"""Describe retained MP3D actor tracks from every native-probed camera.

This CPU-only command consumes existing caller-authored 75-frame actor-root
tracks verbatim. It does not interpolate endpoints, launch Habitat, infer
visibility, rank cameras, or filter candidates by difficulty descriptors.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import load_json, write_json
from avengine.m2.habitat_capture import quaternion_xyzw_to_matrix
from avengine.qa.motion_feasibility import (
    MotionFeasibilityError,
    describe_source_pair_tracks,
)
from avengine.security import (
    PathPolicyError,
    WorkspacePathPolicy,
    atomic_publish_directory,
)


FRAME_COUNT = 75


class MP3DTrackDescriptorError(ValueError):
    """Retained MP3D inputs cannot produce honest raw track descriptors."""


def _required_file(path: Path, *, owner: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"{owner} cannot be resolved: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{owner} is not a file: {resolved}")
    return resolved


def _required_directory(path: Path, *, owner: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"{owner} cannot be resolved: {path}") from exc
    if not resolved.is_dir():
        raise FileNotFoundError(f"{owner} is not a directory: {resolved}")
    return resolved


def _fresh_output(path: Path) -> tuple[Path, WorkspacePathPolicy]:
    raw = path.expanduser()
    if not raw.is_absolute():
        raise MP3DTrackDescriptorError("--output must be an absolute path")
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as exc:
        raise MP3DTrackDescriptorError(
            f"--output parent does not exist: {raw.parent}"
        ) from exc
    output = parent / raw.name
    if os.path.lexists(output):
        raise MP3DTrackDescriptorError(f"refusing to replace output: {output}")
    try:
        policy = WorkspacePathPolicy.from_roots((parent,))
    except PathPolicyError as exc:
        raise MP3DTrackDescriptorError(str(exc)) from exc
    return output, policy


def _mapping(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MP3DTrackDescriptorError(f"{owner} must be an object")
    return value


def _bank_child(bank_root: Path, value: Any, *, owner: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MP3DTrackDescriptorError(f"{owner} must be a relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MP3DTrackDescriptorError(f"{owner} must stay under the bank root")
    resolved = _required_file(bank_root / relative, owner=owner)
    if not resolved.is_relative_to(bank_root):
        raise MP3DTrackDescriptorError(f"{owner} escapes the bank root")
    return resolved


def _actor_tracks(
    route: Any, *, expected_room_id: str
) -> tuple[tuple[str, str], tuple[np.ndarray, np.ndarray], Mapping[str, Any]]:
    value = _mapping(route, owner="route explanation")
    if (
        value.get("research_only") is not True
        or value.get("episode_counted") is not False
    ):
        raise MP3DTrackDescriptorError(
            "route explanation must remain research_only and episode_counted=false"
        )
    provenance = _mapping(value.get("provenance"), owner="route provenance")
    if provenance.get("new_request_room_id") != expected_room_id:
        raise MP3DTrackDescriptorError("route explanation room differs from bank")
    realization = _mapping(
        value.get("current_visual_realization"), owner="current visual realization"
    )
    actor_ids = realization.get("actor_ids")
    routes = _mapping(
        realization.get("actor_skin_root_paths_m"), owner="actor skin-root paths"
    )
    if (
        not isinstance(actor_ids, Sequence)
        or isinstance(actor_ids, (str, bytes))
        or len(actor_ids) != 2
        or any(not isinstance(item, str) or not item for item in actor_ids)
        or len(set(actor_ids)) != 2
        or set(routes) != set(actor_ids)
    ):
        raise MP3DTrackDescriptorError(
            "current visual realization must bind exactly two distinct actor IDs"
        )
    normalized = []
    for actor_id in actor_ids:
        try:
            track = np.asarray(routes[actor_id], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MP3DTrackDescriptorError(
                f"{actor_id} skin-root track must be finite [75,3]"
            ) from exc
        if track.shape != (FRAME_COUNT, 3) or not np.all(np.isfinite(track)):
            raise MP3DTrackDescriptorError(
                f"{actor_id} skin-root track must be finite [75,3]"
            )
        normalized.append(np.ascontiguousarray(track))
    authority = {
        "field": "current_visual_realization.actor_skin_root_paths_m",
        "semantics": "visual_actor_skin_root_not_audio_emitter",
        "frame_count": FRAME_COUNT,
        "actor_ids": list(actor_ids),
        "caller_positions_consumed_without_interpolation": True,
    }
    return (actor_ids[0], actor_ids[1]), (normalized[0], normalized[1]), authority


def _camera_basis(request: Mapping[str, Any]) -> tuple[list[float], ...]:
    rig = _mapping(request.get("primary_camera_rig"), owner="primary camera rig")
    transform = _mapping(rig.get("world_from_rig"), owner="world_from_rig")
    position = np.asarray(transform.get("translation_m"), dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise MP3DTrackDescriptorError("camera position must be finite XYZ")
    try:
        rotation = quaternion_xyzw_to_matrix(transform.get("rotation_xyzw"))
    except (TypeError, ValueError) as exc:
        raise MP3DTrackDescriptorError(f"camera quaternion is invalid: {exc}") from exc
    right = rotation @ np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    up = rotation @ np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    forward = rotation @ np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    return position.tolist(), forward.tolist(), right.tolist(), up.tolist()


def _probe_evidence(
    receipt: Any,
    *,
    expected_room_id: str,
    expected_request_id: str,
    expected_transform: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(receipt, owner="native probe receipt")
    navigation = _mapping(value.get("navigation"), owner="probe navigation")
    rig = _mapping(value.get("camera_listener_rig"), owner="probe camera/listener")
    topdown = _mapping(value.get("topdown"), owner="probe Topdown")
    observations = _mapping(value.get("observations"), owner="probe observations")
    reports = topdown.get("reports")
    checks = topdown.get("checks")
    topdown_required = topdown.get("required")
    if topdown_required is None:
        topdown_required = isinstance(reports, list) and len(reports) == 1
    topdown_passed = topdown.get("passed")
    if topdown_passed is None:
        topdown_passed = bool(
            topdown_required
            and isinstance(reports, list)
            and len(reports) == 1
            and isinstance(checks, list)
            and checks
            and all(
                isinstance(check, Mapping) and check.get("status") == "pass"
                for check in checks
            )
        )
    if (
        value.get("status") != "pass"
        or value.get("room_id") != expected_room_id
        or value.get("request_id") != expected_request_id
        or navigation.get("passed") is not True
        or navigation.get("expected_floor_passed") is not True
        or rig.get("listener_co_located") is not True
        or rig.get("world_from_rig") != expected_transform
        or topdown_required is not True
        or topdown_passed is not True
        or set(observations) != {"rgb", "depth", "semantic"}
    ):
        raise MP3DTrackDescriptorError(
            "native probe does not match one passing co-located viewpoint request"
        )
    return {
        "status": "pass",
        "navigation": dict(navigation),
        "topdown": {
            **dict(topdown),
            "required": True,
            "passed": True,
            "summary_fields_derived_from_reports_checks": (
                "required" not in topdown or "passed" not in topdown
            ),
        },
        "observations": dict(observations),
    }


def run(args: argparse.Namespace) -> Path:
    bank_path = _required_file(args.viewpoint_bank, owner="--viewpoint-bank")
    route_path = _required_file(args.route_explanation, owner="--route-explanation")
    probe_root = _required_directory(args.probe_root, owner="--probe-root")
    output, policy = _fresh_output(args.output)
    bank = _mapping(load_json(bank_path), owner="viewpoint bank")
    if (
        bank.get("research_only") is not True
        or bank.get("episode_counted") is not False
    ):
        raise MP3DTrackDescriptorError(
            "viewpoint bank must remain research_only and episode_counted=false"
        )
    room_id = bank.get("room_id")
    if not isinstance(room_id, str) or not room_id:
        raise MP3DTrackDescriptorError("viewpoint bank must name one room_id")
    viewpoint_records = bank.get("viewpoints")
    request_paths = _mapping(
        bank.get("m1_capture_request_paths"), owner="M1 capture request paths"
    )
    if not isinstance(viewpoint_records, list) or not viewpoint_records:
        raise MP3DTrackDescriptorError("viewpoint bank has no viewpoints")
    viewpoint_ids = []
    for index, record in enumerate(viewpoint_records):
        viewpoint = _mapping(record, owner=f"viewpoints[{index}]")
        viewpoint_id = viewpoint.get("viewpoint_id")
        if (
            not isinstance(viewpoint_id, str)
            or not viewpoint_id
            or Path(viewpoint_id).name != viewpoint_id
        ):
            raise MP3DTrackDescriptorError("viewpoint_id must be a safe basename")
        viewpoint_ids.append(viewpoint_id)
    if len(set(viewpoint_ids)) != len(viewpoint_ids) or set(request_paths) != set(
        viewpoint_ids
    ):
        raise MP3DTrackDescriptorError(
            "viewpoint records and M1 request paths must form one exact bijection"
        )
    actor_ids, actor_tracks, track_authority = _actor_tracks(
        load_json(route_path), expected_room_id=room_id
    )

    descriptors = []
    bank_root = bank_path.parent.resolve()
    for viewpoint_id in viewpoint_ids:
        request_path = _bank_child(
            bank_root,
            request_paths[viewpoint_id],
            owner=f"M1 request for {viewpoint_id}",
        )
        request = _mapping(load_json(request_path), owner=f"request {viewpoint_id}")
        if request.get("room_id") != room_id:
            raise MP3DTrackDescriptorError(f"{viewpoint_id} request room differs")
        transform = _mapping(
            _mapping(
                request.get("primary_camera_rig"), owner=f"{viewpoint_id} rig"
            ).get("world_from_rig"),
            owner=f"{viewpoint_id} world_from_rig",
        )
        listener = _mapping(request.get("listener"), owner=f"{viewpoint_id} listener")
        if listener.get("rig_from_listener") != {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }:
            raise MP3DTrackDescriptorError(
                f"{viewpoint_id} listener is not co-located with its camera rig"
            )
        probe_path = _required_file(
            probe_root / viewpoint_id / "receipt.json",
            owner=f"native probe receipt for {viewpoint_id}",
        )
        native = _probe_evidence(
            load_json(probe_path),
            expected_room_id=room_id,
            expected_request_id=str(request.get("request_id")),
            expected_transform=transform,
        )
        position, forward, right, up = _camera_basis(request)
        try:
            descriptor = describe_source_pair_tracks(
                actor_tracks[0],
                actor_tracks[1],
                source1_id=actor_ids[0],
                source2_id=actor_ids[1],
                listener_positions_xyz=position,
                listener_forward_xyz=forward,
                listener_right_xyz=right,
                listener_up_xyz=up,
                horizontal_axis_indices=(0, 2),
                vertical_axis_index=1,
                coordinate_space="habitat_world_xyz_m_y_up",
                track_authority=track_authority["field"],
                frame_indices=tuple(range(FRAME_COUNT)),
            )
        except MotionFeasibilityError as exc:
            raise MP3DTrackDescriptorError(
                f"{viewpoint_id} track descriptor failed: {exc}"
            ) from exc
        descriptors.append(
            {
                "viewpoint_id": viewpoint_id,
                "m1_request": str(request_path),
                "native_probe_receipt": str(probe_path),
                "camera_listener": {
                    "world_from_rig": dict(transform),
                    "listener_co_located": True,
                    "forward_world_xyz": forward,
                    "right_world_xyz": right,
                    "up_world_xyz": up,
                },
                "native_static_probe": native,
                "track_descriptor": descriptor,
            }
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        write_json(
            staging / "viewpoint_raw_track_descriptors.json",
            {
                "artifact_status": "explicit_actor_root_tracks_described",
                "research_only": True,
                "episode_counted": False,
                "formal_release_status": "not_run",
                "candidate_filtering": "none",
                "raw_legal_candidate_catalog_preserved": True,
                "room_id": room_id,
                "inputs": {
                    "viewpoint_bank": str(bank_path),
                    "route_explanation": str(route_path),
                    "native_probe_root": str(probe_root),
                },
                "track_authority": track_authority,
                "viewpoint_count": len(descriptors),
                "viewpoints": descriptors,
                "claim_boundary": (
                    "exact retained visual actor-root tracks and static native probe "
                    "references only; audio-emitter paths, dynamic actor pixels, "
                    "projection, in-frame, occlusion, visible pixels, collision, "
                    "question difficulty and mini-pilot certification remain separate"
                ),
            },
        )
        (staging / "README.md").write_text(
            "# MP3D raw viewpoint tracks\n\n"
            "All 75 caller-supplied actor-root frames are retained without "
            "interpolation. This catalog applies no side, distance, exchange, "
            "visibility, composition, or difficulty filter. Native dynamic pixels, "
            "audio emitter tracks, RIRs and mini-pilots remain pending.\n",
            encoding="utf-8",
        )
        try:
            atomic_publish_directory(policy, staging, output)
        except (FileExistsError, PathPolicyError) as exc:
            raise MP3DTrackDescriptorError(str(exc)) from exc
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewpoint-bank", required=True, type=Path)
    parser.add_argument("--route-explanation", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

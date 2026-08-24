#!/usr/bin/env python3
"""Describe retained MP3D actor tracks from every native-probed camera.

This CPU-only command consumes existing caller-authored 75-frame actor-root
tracks verbatim. It does not interpolate endpoints, launch Habitat, infer
visibility, rank cameras, or filter candidates by difficulty descriptors.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import load_json, write_json
from avengine.m2.habitat_capture import quaternion_xyzw_to_matrix
from avengine.m5.mp3d_raw_clues import (
    MP3DRawClueError,
    dynamic_capture_request_path,
    load_dynamic_capture_raw_clues,
)
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


def _viewpoint_join_input(
    value: Mapping[str, Any], *, input_kind: str
) -> tuple[str, list[dict[str, Any]], Mapping[str, Any]]:
    if (
        value.get("research_only") is not True
        or value.get("episode_counted") is not False
    ):
        raise MP3DTrackDescriptorError(
            "viewpoint input must remain research_only and episode_counted=false"
        )
    room_id = value.get("room_id")
    if not isinstance(room_id, str) or not room_id:
        raise MP3DTrackDescriptorError("viewpoint input must name one room_id")
    request_paths = _mapping(
        value.get("m1_capture_request_paths"), owner="M1 capture request paths"
    )
    normalized: list[dict[str, Any]] = []
    if input_kind == "region_viewpoint_banks":
        region_banks = value.get("region_banks")
        if not isinstance(region_banks, list) or not region_banks:
            raise MP3DTrackDescriptorError("region viewpoint input has no banks")
        for region_index, region_value in enumerate(region_banks):
            region = _mapping(region_value, owner=f"region_banks[{region_index}]")
            candidates = region.get("selected_candidates")
            if not isinstance(candidates, list) or not candidates:
                raise MP3DTrackDescriptorError(
                    f"region_banks[{region_index}] has no selected candidates"
                )
            for candidate_index, candidate_value in enumerate(candidates):
                candidate = _mapping(
                    candidate_value,
                    owner=(
                        f"region_banks[{region_index}].selected_candidates"
                        f"[{candidate_index}]"
                    ),
                )
                normalized.append(
                    {
                        "viewpoint_id": candidate.get("placement_id"),
                        "input_record": deepcopy(dict(candidate)),
                        "input_parent_record": deepcopy(dict(region)),
                    }
                )
    elif input_kind == "probe_subset":
        candidates = value.get("selected_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise MP3DTrackDescriptorError("probe subset has no selected candidates")
        normalized = [
            {
                "viewpoint_id": _mapping(
                    item, owner=f"selected_candidates[{index}]"
                ).get("placement_id"),
                "input_record": deepcopy(
                    dict(_mapping(item, owner=f"selected_candidates[{index}]"))
                ),
                "input_parent_record": None,
            }
            for index, item in enumerate(candidates)
        ]
    else:
        records = value.get("viewpoints")
        if not isinstance(records, list) or not records:
            raise MP3DTrackDescriptorError("viewpoint bank has no viewpoints")
        normalized = []
        for index, item in enumerate(records):
            record = _mapping(item, owner=f"viewpoints[{index}]")
            normalized.append(
                {
                    "viewpoint_id": record.get(
                        "viewpoint_id", record.get("placement_id")
                    ),
                    "input_record": deepcopy(dict(record)),
                    "input_parent_record": None,
                }
            )
    ids = []
    for index, record in enumerate(normalized):
        viewpoint_id = record["viewpoint_id"]
        if (
            not isinstance(viewpoint_id, str)
            or not viewpoint_id
            or Path(viewpoint_id).name != viewpoint_id
        ):
            raise MP3DTrackDescriptorError(
                f"viewpoint input record {index} has no safe placement/viewpoint ID"
            )
        ids.append(viewpoint_id)
        declared = record["input_record"].get("m1_capture_request_path")
        if declared is not None and declared != request_paths.get(viewpoint_id):
            raise MP3DTrackDescriptorError(
                f"{viewpoint_id} input record and M1 path mapping differ"
            )
    if len(set(ids)) != len(ids) or set(request_paths) != set(ids):
        raise MP3DTrackDescriptorError(
            "input records and M1 request paths must form one exact bijection"
        )
    return room_id, normalized, request_paths


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


def _capture_artifact(root: Path, value: Any, *, owner: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MP3DTrackDescriptorError(f"{owner} must be a relative artifact path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MP3DTrackDescriptorError(f"{owner} must stay under its capture root")
    path = _required_file(root / relative, owner=owner)
    if not path.is_relative_to(root):
        raise MP3DTrackDescriptorError(f"{owner} escapes its capture root")
    return path


def _dynamic_capture_roots(paths: Sequence[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for index, path in enumerate(paths):
        root = _required_directory(path, owner=f"--dynamic-capture-dir[{index}]")
        try:
            request_path = dynamic_capture_request_path(root)
        except MP3DRawClueError as exc:
            raise MP3DTrackDescriptorError(str(exc)) from exc
        request_key = str(request_path)
        if request_key in result:
            raise MP3DTrackDescriptorError(
                "dynamic capture directories repeat one M1 request"
            )
        result[request_key] = root
    return result


def run(args: argparse.Namespace) -> Path:
    if args.probe_subset is not None:
        input_kind = "probe_subset"
        input_path = _required_file(args.probe_subset, owner="--probe-subset")
    elif args.region_viewpoint_banks is not None:
        input_kind = "region_viewpoint_banks"
        input_path = _required_file(
            args.region_viewpoint_banks, owner="--region-viewpoint-banks"
        )
    else:
        input_kind = "viewpoint_bank"
        input_path = _required_file(args.viewpoint_bank, owner="--viewpoint-bank")
    probe_root = _required_directory(args.probe_root, owner="--probe-root")
    output, policy = _fresh_output(args.output)
    input_document = _mapping(load_json(input_path), owner="viewpoint input")
    room_id, viewpoint_records, request_paths = _viewpoint_join_input(
        input_document, input_kind=input_kind
    )
    if args.route_explanation is None:
        route_path = None
        actor_ids = None
        actor_tracks = None
        track_authority: Mapping[str, Any] = {
            "status": "pending_no_route_explanation",
            "field": None,
        }
    else:
        route_path = _required_file(args.route_explanation, owner="--route-explanation")
        actor_ids, actor_tracks, track_authority = _actor_tracks(
            load_json(route_path), expected_room_id=room_id
        )
    dynamic_by_request = _dynamic_capture_roots(args.dynamic_capture_dir)

    descriptors = []
    bank_root = input_path.parent.resolve()
    expected_request_keys: set[str] = set()
    for viewpoint_record in viewpoint_records:
        viewpoint_id = str(viewpoint_record["viewpoint_id"])
        request_path = _bank_child(
            bank_root,
            request_paths[viewpoint_id],
            owner=f"M1 request for {viewpoint_id}",
        )
        request = _mapping(load_json(request_path), owner=f"request {viewpoint_id}")
        if request.get("room_id") != room_id:
            raise MP3DTrackDescriptorError(f"{viewpoint_id} request room differs")
        expected_request_keys.add(str(request_path))
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
        native["receipt_record"] = deepcopy(dict(load_json(probe_path)))
        position, forward, right, up = _camera_basis(request)
        descriptor = None
        if actor_ids is not None and actor_tracks is not None:
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
                    track_authority=str(track_authority["field"]),
                    frame_indices=tuple(range(FRAME_COUNT)),
                )
            except MotionFeasibilityError as exc:
                raise MP3DTrackDescriptorError(
                    f"{viewpoint_id} track descriptor failed: {exc}"
                ) from exc
        dynamic_root = dynamic_by_request.get(str(request_path))
        runtime_descriptor = None
        dynamic_raw_clues: Mapping[str, Any]
        if dynamic_root is not None:
            try:
                dynamic = load_dynamic_capture_raw_clues(
                    dynamic_root,
                    request=request,
                    request_path=request_path,
                    expected_room_id=room_id,
                )
            except MP3DRawClueError as exc:
                raise MP3DTrackDescriptorError(
                    f"{viewpoint_id} dynamic raw-clue join failed: {exc}"
                ) from exc
            dynamic_raw_clues = deepcopy(dict(dynamic.raw_clues))
            if dynamic.has_current_pose_authority:
                runtime_track = deepcopy(
                    dynamic_raw_clues["pair_tracks"][
                        "actual_emitter_relative_to_actual_listener"
                    ]
                )
            else:
                try:
                    runtime_track = describe_source_pair_tracks(
                        dynamic.source_positions[:, 0, :],
                        dynamic.source_positions[:, 1, :],
                        source1_id=dynamic.source_ids[0],
                        source2_id=dynamic.source_ids[1],
                        listener_positions_xyz=position,
                        listener_forward_xyz=forward,
                        listener_right_xyz=right,
                        listener_up_xyz=up,
                        horizontal_axis_indices=(0, 2),
                        vertical_axis_index=1,
                        coordinate_space="habitat_world_xyz_m_y_up",
                        track_authority="dynamic_capture.arrays/source_positions_m.npy",
                        frame_indices=tuple(range(FRAME_COUNT)),
                    )
                except MotionFeasibilityError as exc:
                    raise MP3DTrackDescriptorError(
                        f"{viewpoint_id} runtime emitter track failed: {exc}"
                    ) from exc
            visible_pixels = dynamic.visible_pixels
            for frame_index, frame in enumerate(runtime_track["frames"]):
                frame["source1"]["visible_pixels"] = int(visible_pixels[frame_index, 0])
                frame["source2"]["visible_pixels"] = int(visible_pixels[frame_index, 1])
            if not dynamic.has_current_pose_authority:
                runtime_track["native_projection_visibility"] = {
                    "status": "native_visible_pixels_recorded_projection_occlusion_pending",
                    "projection_by_frame": None,
                    "in_frame_by_frame": None,
                    "occlusion_by_frame": None,
                    "visible_pixels_by_frame": {
                        dynamic.source_ids[0]: visible_pixels[:, 0].tolist(),
                        dynamic.source_ids[1]: visible_pixels[:, 1].tolist(),
                    },
                }
            runtime_descriptor = {
                "runtime_binding_status": (
                    "pass_current_post_render_authority"
                    if dynamic.has_current_pose_authority
                    else "pass_legacy_pending_current_pose_authority"
                ),
                "capture_root": str(dynamic.root),
                "research_receipt": str(dynamic.receipt_path),
                "research_receipt_record": deepcopy(dict(dynamic.receipt)),
                "actor_ids": list(dynamic.actor_ids),
                "source_ids": list(dynamic.source_ids),
                "track_descriptor": runtime_track,
            }
        else:
            dynamic_raw_clues = {
                "status": "pending_no_dynamic_capture",
                "has_current_post_render_pose_authority": False,
                "frames": None,
                "pair_tracks": None,
                "motion_timing": None,
                "planned_continuous_polyline": None,
                "audio_timing": {
                    "status": "pending_no_audio_program_or_native_audio_receipt",
                    "audio_program": None,
                    "audio_receipt": None,
                    "source_event_timing_by_frame": None,
                    "p1_or_rir_identity": None,
                },
                "p1_amodal_visibility": {
                    "status": "pending_no_p1_or_amodal_receipt",
                    "occlusion_by_frame": None,
                    "off_screen_by_frame": None,
                },
                "candidate_filtering": "none",
                "candidate_retained": True,
            }
        descriptors.append(
            {
                "viewpoint_id": viewpoint_id,
                "input_kind": input_kind,
                "input_record": deepcopy(viewpoint_record["input_record"]),
                "input_parent_record": deepcopy(
                    viewpoint_record["input_parent_record"]
                ),
                "m1_request": str(request_path),
                "m1_request_record": deepcopy(dict(request)),
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
                "runtime_emitter_track": runtime_descriptor,
                "dynamic_raw_clues": dynamic_raw_clues,
                "candidate_retained_by_bridge": True,
                "bridge_candidate_filtering": "none",
            }
        )
    if not set(dynamic_by_request).issubset(expected_request_keys):
        raise MP3DTrackDescriptorError(
            "a dynamic capture does not belong to any input candidate"
        )
    if len(descriptors) != len(viewpoint_records):
        raise AssertionError("raw-clue bridge changed the input candidate count")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        write_json(
            staging / "viewpoint_raw_track_descriptors.json",
            {
                "artifact_status": (
                    "input_candidates_joined_to_static_and_dynamic_raw_clues"
                    if dynamic_by_request
                    else "input_candidates_joined_to_static_receipts"
                ),
                "research_only": True,
                "episode_counted": False,
                "formal_release_status": "not_run",
                "candidate_filtering": "none",
                "raw_legal_candidate_catalog_preserved": True,
                "input_records_preserved_without_bridge_filtering": True,
                "input_kind": input_kind,
                "room_id": room_id,
                "inputs": {
                    "viewpoint_input": str(input_path),
                    "route_explanation": (
                        None if route_path is None else str(route_path)
                    ),
                    "native_probe_root": str(probe_root),
                    "dynamic_capture_directories": [
                        str(item.resolve()) for item in args.dynamic_capture_dir
                    ],
                },
                "input_document": deepcopy(dict(input_document)),
                "track_authority": track_authority,
                "input_candidate_count": len(viewpoint_records),
                "output_viewpoint_count": len(descriptors),
                "input_count_equals_output_count": True,
                "viewpoint_count": len(descriptors),
                "viewpoints": descriptors,
                "claim_boundary": (
                    "every input candidate record is retained and joined to its static "
                    "receipt plus optional dynamic raw clues; modal pixels do not infer "
                    "occlusion or off-screen, and audio/P1/RIR/mini-pilot certification "
                    "remain separate pending evidence"
                ),
            },
        )
        (staging / "README.md").write_text(
            "# MP3D raw viewpoint tracks\n\n"
            "All 75 caller-supplied actor-root frames are retained without "
            "interpolation. This catalog applies no side, distance, exchange, "
            "visibility, composition, or difficulty filter. Native dynamic pixels, "
            "emitter tracks and visible-pixel arrays are attached only when explicitly "
            "provided; RIRs and mini-pilots remain pending.\n",
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--viewpoint-bank", type=Path)
    source.add_argument("--probe-subset", type=Path)
    source.add_argument("--region-viewpoint-banks", type=Path)
    parser.add_argument(
        "--route-explanation",
        type=Path,
        help=(
            "optional legacy planned actor-root authority; current dynamic captures "
            "carry their own planned/actual roots"
        ),
    )
    parser.add_argument("--probe-root", required=True, type=Path)
    parser.add_argument(
        "--dynamic-capture-dir",
        action="append",
        default=[],
        type=Path,
        help="repeat once per viewpoint to bind runtime emitter/visibility arrays",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

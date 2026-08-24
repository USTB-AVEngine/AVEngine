"""CPU research actor-track materializer for MP3D region route plans.

The output is deliberately not an M2 capture request.  It preserves the
formal M2 package/base-request authority used for calibration and action/contact
loading, while allowing two independent actor tracks and all-idle static tracks
without weakening the existing single-asset M2 contract.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import load_json, sha256_file
from avengine.contracts.transforms import normalized_quaternion_xyzw
from avengine.m2.contracts import (
    ValidatedM2Inputs,
    compute_applied_state_hash,
    compute_pose_hash,
    load_and_validate_inputs,
)
from avengine.m2.habitat_capture import (
    load_runtime_asset_bundle,
    quaternion_xyzw_to_matrix,
    compile_frame_applications,
)
from avengine.m2.kinematics import _matrix_rotation_to_quaternion
from avengine.security.path_policy import WorkspacePathPolicy, atomic_publish_directory


FRAME_COUNT = 75
FRAME_RATE_HZ = 15
TICKS_PER_FRAME = 3_200
MOTION_CASES = (
    "static_static",
    "s1_moving_s2_static",
    "s1_static_s2_moving",
    "both_moving",
)
ACTOR_AXES = (
    {
        "axis_index": 0,
        "actor_id": "actor0",
        "source_id": "source1",
        "semantic_id": 210,
        "emitter_link_name": "beagle Xtra Mouth",
    },
    {
        "axis_index": 1,
        "actor_id": "actor1",
        "source_id": "source2",
        "semantic_id": 211,
        "emitter_link_name": "beagle Xtra Mouth",
    },
)
_MOVING_SOURCES = {
    "s1_moving_s2_static": {"source1"},
    "s1_static_s2_moving": {"source2"},
    "both_moving": {"source1", "source2"},
    "static_static": set(),
}
_TANGENT_EPSILON_M = 1.0e-10


class MP3DRegionActorTrackError(ValueError):
    """The CPU region actor-track materializer cannot proceed truthfully."""


@dataclass(frozen=True)
class MaterializerCalibration:
    asset_path: Path
    base_request_path: Path
    inputs: ValidatedM2Inputs
    bundle: Any
    asset_manifest_sha256: str
    asset_id: str
    contact_by_action_tick: Mapping[str, Mapping[int, tuple[dict[str, Any], ...]]]
    base_first_rotation_xyzw: tuple[float, float, float, float]
    forward_axis: str
    up_axis: str
    base_compiled_frame_count: int


@dataclass(frozen=True)
class RouteCase:
    region_instance_id: str
    route_family_id: str
    motion_case: str
    camera_binding: Mapping[str, Any]
    source_positions_m: Mapping[str, np.ndarray]
    route_descriptors: Mapping[str, Any]


def validate_shared_asset_calibration(
    asset_manifest_paths: Sequence[str | Path],
    base_m2_request_paths: Sequence[str | Path],
) -> None:
    """Reject mixed assets/calibrations in this bounded two-Beagle materializer."""

    if len(asset_manifest_paths) != 2 or len(base_m2_request_paths) != 2:
        raise MP3DRegionActorTrackError(
            "exactly two actor asset/request references are required"
        )
    asset_paths = tuple(
        Path(path).expanduser().resolve() for path in asset_manifest_paths
    )
    request_paths = tuple(
        Path(path).expanduser().resolve() for path in base_m2_request_paths
    )
    if len(set(asset_paths)) != 1 or len(set(request_paths)) != 1:
        raise MP3DRegionActorTrackError(
            "different asset manifests or base M2 calibrations are not supported "
            "by the bounded Beagle materializer"
        )


def _error(message: str) -> MP3DRegionActorTrackError:
    return MP3DRegionActorTrackError(message)


def _finite_xyz(value: Any, *, owner: str) -> np.ndarray:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise _error(f"{owner} must be a finite three-vector")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise _error(f"{owner} must be a finite three-vector")
    return result


def _finite_route(value: Any, *, owner: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (FRAME_COUNT, 3) or not np.all(np.isfinite(result)):
        raise _error(f"{owner} must be finite [{FRAME_COUNT},3]")
    return np.ascontiguousarray(result)


def _write_no_replace(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise _error(f"refusing to replace existing artifact: {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _matrix_from_transform(transform: Mapping[str, Any]) -> np.ndarray:
    translation = _finite_xyz(transform.get("translation_m"), owner="translation_m")
    quaternion = normalized_quaternion_xyzw(transform.get("rotation_xyzw"))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_xyzw_to_matrix(quaternion)
    matrix[:3, 3] = translation
    return matrix


def _transform_json(matrix: np.ndarray) -> dict[str, list[float]]:
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise _error("transform matrix must be finite [4,4]")
    return {
        "translation_m": [float(value) for value in matrix[:3, 3]],
        "rotation_xyzw": [
            float(value) for value in _matrix_rotation_to_quaternion(matrix[:3, :3])
        ],
    }


def _load_contact_phases(
    *,
    bundle: Any,
    asset: Mapping[str, Any],
) -> Mapping[str, Mapping[int, tuple[dict[str, Any], ...]]]:
    file_records = asset.get("files")
    record = next(
        (
            item
            for item in file_records
            if isinstance(item, Mapping) and item.get("role") == "contact_phases"
        ),
        None,
    )
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise _error("qualified asset lacks a contact_phases file record")
    path = (Path(bundle.paths_by_role["visual"]).parent / record["path"]).resolve()
    if not path.is_file():
        raise _error(f"contact_phases file is missing: {path}")
    if sha256_file(path) != record["sha256"]:
        raise _error("contact_phases bytes differ from the qualified asset manifest")
    document = load_json(path)
    if document.get("schema") != "avengine_m2_contact_phases_v1":
        raise _error("contact_phases schema is not the M2 contact-phase authority")
    visual_record = bundle.records_by_role["visual"]
    if document.get("source_glb_sha256") != visual_record["sha256"]:
        raise _error("contact_phases source visual differs from the asset visual")
    coordinate = document.get("coordinate_system")
    asset_coordinate = asset.get("coordinate_system")
    if coordinate != asset_coordinate:
        raise _error("contact_phases coordinate system differs from the asset")
    contact_order = asset.get("contacts", {}).get("contact_order")
    if document.get("contact_order") != contact_order:
        raise _error("contact_phases contact order differs from the asset")
    raw_actions = document.get("actions")
    if not isinstance(raw_actions, list):
        raise _error("contact_phases actions must be a list")
    result: dict[str, Mapping[int, tuple[dict[str, Any], ...]]] = {}
    for action in raw_actions:
        if not isinstance(action, Mapping):
            raise _error("contact_phases action must be an object")
        action_id = action.get("semantic_action_id")
        frames = action.get("frames")
        if not isinstance(action_id, str) or not isinstance(frames, list):
            raise _error("contact_phases action lacks semantic_action_id/frames")
        by_tick: dict[int, tuple[dict[str, Any], ...]] = {}
        for frame in frames:
            if not isinstance(frame, Mapping):
                raise _error(f"contact_phases {action_id} frame must be an object")
            tick = frame.get("sample_tick")
            contacts = frame.get("contacts")
            if (
                isinstance(tick, bool)
                or not isinstance(tick, int)
                or not isinstance(contacts, list)
            ):
                raise _error(f"contact_phases {action_id} frame is malformed")
            if tick in by_tick:
                raise _error(f"contact_phases {action_id} repeats sample tick {tick}")
            normalized: list[dict[str, Any]] = []
            for contact in contacts:
                if (
                    not isinstance(contact, Mapping)
                    or contact.get("contact_id") not in contact_order
                    or not isinstance(contact.get("in_contact"), bool)
                ):
                    raise _error(f"contact_phases {action_id} has an invalid contact")
                normalized.append(
                    {
                        "contact_id": str(contact["contact_id"]),
                        "in_contact": bool(contact["in_contact"]),
                    }
                )
            if [item["contact_id"] for item in normalized] != list(contact_order):
                raise _error(
                    f"contact_phases {action_id} contact order is not canonical"
                )
            by_tick[tick] = tuple(normalized)
        result[action_id] = by_tick
    if set(result) != {"idle", "walk"}:
        raise _error("contact_phases must contain exactly idle and walk")
    for action_id in ("idle", "walk"):
        clip = bundle.action_sets_by_role[bundle.action_roles_by_id[action_id]].action(
            action_id
        )
        if set(clip.sample_ticks) != set(result[action_id]):
            raise _error(
                f"contact_phases {action_id} ticks differ from baked action samples"
            )
    return result


def load_materializer_calibration(
    *,
    asset_manifest_path: str | Path,
    base_m2_request_path: str | Path,
) -> MaterializerCalibration:
    asset_path = Path(asset_manifest_path).expanduser().resolve(strict=True)
    base_request_path = Path(base_m2_request_path).expanduser().resolve(strict=True)
    try:
        inputs = load_and_validate_inputs(asset_path, base_request_path)
        bundle = load_runtime_asset_bundle(inputs)
        base_frames = compile_frame_applications(inputs, bundle)
    except Exception as exc:
        raise _error(
            f"qualified base M2 package/request validation failed: {exc}"
        ) from exc
    asset = inputs.asset
    coordinate = asset.get("coordinate_system")
    if not isinstance(coordinate, Mapping):
        raise _error("asset lacks coordinate system")
    if (
        coordinate.get("forward_axis") != "-Z"
        or coordinate.get("up_axis") != "+Y"
        or coordinate.get("quaternion_order") != "xyzw"
        or coordinate.get("linear_unit") != "meter"
    ):
        raise _error(
            "materializer only supports the qualified Beagle -Z/+Y meter frame"
        )
    states = inputs.request.get("states")
    if not isinstance(states, list) or len(states) != FRAME_COUNT:
        raise _error("base M2 request must contain 75 states")
    base_rotation = normalized_quaternion_xyzw(
        states[0]["root_transform"]["rotation_xyzw"]
    )
    return MaterializerCalibration(
        asset_path=asset_path,
        base_request_path=base_request_path,
        inputs=inputs,
        bundle=bundle,
        asset_manifest_sha256=sha256_file(asset_path),
        asset_id=str(asset["asset_id"]),
        contact_by_action_tick=_load_contact_phases(bundle=bundle, asset=asset),
        base_first_rotation_xyzw=tuple(float(value) for value in base_rotation),
        forward_axis="-Z",
        up_axis="+Y",
        base_compiled_frame_count=len(base_frames),
    )


def load_region_cases(
    route_bank_path: str | Path,
    *,
    region_instance_id: str,
) -> tuple[RouteCase, ...]:
    bank_path = Path(route_bank_path).expanduser().resolve(strict=True)
    document = load_json(bank_path)
    families = [
        family
        for family in document.get("families", [])
        if isinstance(family, Mapping)
        and family.get("region_instance_id") == region_instance_id
    ]
    if len(families) != 4:
        raise _error(
            f"{region_instance_id} must have exactly four r4 route families, got {len(families)}"
        )
    cases: list[RouteCase] = []
    for family in sorted(
        families,
        key=lambda item: int(item["camera_binding"]["selection_order"]),
    ):
        family_id = str(family["route_family_id"])
        camera_binding = deepcopy(dict(family["camera_binding"]))
        raw_cases = family.get("cases")
        if not isinstance(raw_cases, Mapping):
            raise _error(f"{family_id} lacks cases")
        for motion_case in MOTION_CASES:
            raw_case = raw_cases.get(motion_case)
            if not isinstance(raw_case, Mapping):
                raise _error(f"{family_id} lacks {motion_case}")
            source1 = _finite_route(
                raw_case.get("source1_positions_m"),
                owner=f"{family_id}/{motion_case}/source1_positions_m",
            )
            source2 = _finite_route(
                raw_case.get("source2_positions_m"),
                owner=f"{family_id}/{motion_case}/source2_positions_m",
            )
            if (
                raw_case.get("frame_count") != FRAME_COUNT
                or raw_case.get("frame_rate_hz") != FRAME_RATE_HZ
            ):
                raise _error(f"{family_id}/{motion_case} clock is not 75@15")
            cases.append(
                RouteCase(
                    region_instance_id=region_instance_id,
                    route_family_id=family_id,
                    motion_case=motion_case,
                    camera_binding=camera_binding,
                    source_positions_m={"source1": source1, "source2": source2},
                    route_descriptors=deepcopy(dict(raw_case["route_descriptors"])),
                )
            )
    return tuple(cases)


def _moving_tangents(
    positions: np.ndarray,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    raw = np.zeros_like(positions)
    sources: list[dict[str, Any]] = []
    for index in range(FRAME_COUNT):
        if index == 0:
            tangent = positions[1] - positions[0]
            source = "forward_difference"
        elif index == FRAME_COUNT - 1:
            tangent = positions[-1] - positions[-2]
            source = "backward_difference"
        else:
            tangent = positions[index + 1] - positions[index - 1]
            source = "central_difference"
        tangent[1] = 0.0
        raw[index] = tangent
        sources.append({"frame_index": index, "source": source})
    for index, tangent in enumerate(raw):
        norm = float(np.linalg.norm(tangent[[0, 2]]))
        if norm > _TANGENT_EPSILON_M:
            continue
        candidates = [
            candidate
            for candidate, other in enumerate(raw)
            if candidate != index
            and float(np.linalg.norm(other[[0, 2]])) > _TANGENT_EPSILON_M
        ]
        if not candidates:
            raise _error("moving route has no non-zero tangent")
        nearest = min(
            candidates, key=lambda candidate: (abs(candidate - index), candidate)
        )
        raw[index] = raw[nearest]
        sources[index] = {
            "frame_index": index,
            "source": "nearest_nonzero_segment",
            "reference_frame_index": nearest,
        }
    return raw, tuple(sources)


def _tangent_quaternion_xyzw(tangent: np.ndarray) -> tuple[float, float, float, float]:
    direction = np.asarray(tangent, dtype=np.float64).copy()
    direction[1] = 0.0
    norm = float(np.linalg.norm(direction[[0, 2]]))
    if norm <= _TANGENT_EPSILON_M:
        raise _error("tangent has no horizontal direction")
    forward = direction / norm
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    back = -forward
    right = np.cross(up, back)
    right /= float(np.linalg.norm(right))
    rotation = np.column_stack((right, up, back))
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-8):
        raise _error("tangent rotation is not right-handed")
    quaternion = _matrix_rotation_to_quaternion(rotation)
    return tuple(float(value) for value in quaternion)


def _track_rotations(
    positions: np.ndarray,
    *,
    moving: bool,
    base_rotation: Sequence[float],
) -> tuple[tuple[tuple[float, float, float, float], ...], tuple[dict[str, Any], ...]]:
    if not moving:
        rotation = tuple(
            float(value) for value in normalized_quaternion_xyzw(base_rotation)
        )
        return tuple(rotation for _ in range(FRAME_COUNT)), tuple(
            {"frame_index": index, "source": "base_grounded_m2_first_state_rotation"}
            for index in range(FRAME_COUNT)
        )
    tangents, sources = _moving_tangents(positions)
    return tuple(_tangent_quaternion_xyzw(tangent) for tangent in tangents), sources


def _clip_sample(
    calibration: MaterializerCalibration,
    *,
    action_id: str,
    action_time_ticks: int,
) -> tuple[
    int, int, tuple[tuple[float, float, float, float], ...], tuple[dict[str, Any], ...]
]:
    clip = calibration.bundle.action_sets_by_role[
        calibration.bundle.action_roles_by_id[action_id]
    ].action(action_id)
    effective_tick = action_time_ticks % clip.loop_duration_ticks
    try:
        sample_index = clip.sample_ticks.index(effective_tick)
    except ValueError as exc:
        raise _error(f"{action_id} action tick {effective_tick} is not baked") from exc
    rotations = tuple(
        tuple(float(component) for component in quaternion)
        for quaternion in clip.rotations_xyzw[sample_index]
    )
    contacts = calibration.contact_by_action_tick[action_id].get(effective_tick)
    if contacts is None:
        raise _error(f"{action_id} has no contact authority at tick {effective_tick}")
    return effective_tick, sample_index, rotations, contacts


def _build_actor_track(
    *,
    case: RouteCase,
    calibration: MaterializerCalibration,
    actor_axis: Mapping[str, Any],
    route_positions: np.ndarray,
    motion_case: str,
) -> dict[str, Any]:
    source_id = str(actor_axis["source_id"])
    moving = source_id in _MOVING_SOURCES[motion_case]
    rotations, rotation_sources = _track_rotations(
        route_positions,
        moving=moving,
        base_rotation=calibration.base_first_rotation_xyzw,
    )
    base_state = calibration.inputs.request["states"][0]
    base_mouth = deepcopy(base_state["mouth_state"])
    states: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    action_id = "walk" if moving else "idle"
    for index in range(FRAME_COUNT):
        action_time_ticks = index * TICKS_PER_FRAME
        effective_tick, sample_index, joints, contacts = _clip_sample(
            calibration,
            action_id=action_id,
            action_time_ticks=action_time_ticks,
        )
        root_transform = {
            "translation_m": route_positions[index].tolist(),
            "rotation_xyzw": list(rotations[index]),
        }
        state = {
            "frame_index": index,
            "pts_ticks": action_time_ticks,
            "action_id": action_id,
            "action_time_ticks": action_time_ticks,
            "root_transform": root_transform,
            "joint_states": [
                {
                    "joint_id": joint_id,
                    "rotation_xyzw": list(rotation),
                }
                for joint_id, rotation in zip(
                    calibration.bundle.joint_mapping["runtime_joint_order"],
                    joints,
                    strict=True,
                )
            ],
            "contact_states": [dict(contact) for contact in contacts],
            "mouth_state": deepcopy(base_mouth),
        }
        state["pose_hash"] = compute_pose_hash(calibration.inputs.asset, state)
        state["applied_state_hash"] = compute_applied_state_hash(
            calibration.inputs.asset,
            state,
            asset_manifest_sha256=calibration.asset_manifest_sha256,
        )
        actor_matrix = _matrix_from_transform(root_transform)
        skin_matrix = actor_matrix @ np.asarray(
            calibration.bundle.actor_from_skin_root,
            dtype=np.float64,
        )
        states.append(
            {
                **state,
                "effective_action_tick": effective_tick,
                "action_sample_index": sample_index,
            }
        )
        frames.append(
            {
                "frame_index": index,
                "pts_ticks": action_time_ticks,
                "action_id": action_id,
                "action_time_ticks": action_time_ticks,
                "action_sample_index": sample_index,
                "route_source_center_m": route_positions[index].tolist(),
                "planned_world_from_actor": _transform_json(actor_matrix),
                "derived_world_from_skin_root": _transform_json(skin_matrix),
                "orientation_source": rotation_sources[index],
                "emitter_actual_position_m": None,
                "emitter_status": "pending_native_execution",
                "paw_support_contact": None,
                "paw_support_status": "pending_native_execution",
                "articulated_collision": None,
                "object_id_modal": None,
            }
        )
    return {
        "artifact_kind": "m5_research_actor_track",
        "format_version": 1,
        "research_only": True,
        "formal_m2_request": False,
        "episode_counted": False,
        "receipt": {
            "status": "research_only",
            "formal_m2_request": False,
            "episode_counted": False,
        },
        "actor_id": str(actor_axis["actor_id"]),
        "source_id": source_id,
        "semantic_id": int(actor_axis["semantic_id"]),
        "emitter_link_name": str(actor_axis["emitter_link_name"]),
        "asset_id": calibration.asset_id,
        "asset_manifest_path": str(calibration.asset_path),
        "asset_manifest_sha256": calibration.asset_manifest_sha256,
        "base_m2_request_path": str(calibration.base_request_path),
        "base_m2_request_id": calibration.inputs.request["request_id"],
        "base_m2_validation": {
            "status": "pass",
            "loader": "avengine.m2.contracts.load_and_validate_inputs",
            "compiler": "avengine.m2.habitat_capture.compile_frame_applications",
            "compiled_frame_count": calibration.base_compiled_frame_count,
        },
        "route_source_center_plan": {
            "authority": "r4 region source-center route plan",
            "region_instance_id": case.region_instance_id,
            "route_family_id": case.route_family_id,
            "motion_case": motion_case,
            "source_id": source_id,
            "positions_m": route_positions.tolist(),
            "root_translation_calibration": (
                "zero route-point-to-world_from_actor offset; exact-asset "
                "research calibration reference only"
            ),
        },
        "coordinate_system": {
            "forward_axis": calibration.forward_axis,
            "up_axis": calibration.up_axis,
            "quaternion_order": "xyzw",
            "linear_unit": "meter",
        },
        "native_pending": {
            "support_contact": None,
            "emitter_actual_position_m": None,
            "articulated_collision": None,
            "object_id": None,
            "native_execution": None,
            "rlr": None,
        },
        "states": states,
        "frames": frames,
    }


def _camera_reference(
    route_bank_path: Path,
    document: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    raw_path = binding.get("m1_capture_request_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise _error("camera binding lacks m1_capture_request_path")
    candidates = [route_bank_path.parent / raw_path]
    inputs = document.get("inputs")
    if isinstance(inputs, Mapping) and isinstance(
        inputs.get("region_viewpoint_bank"), str
    ):
        candidates.append(
            Path(inputs["region_viewpoint_bank"]).resolve().parent / raw_path
        )
    resolved = next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()), None
    )
    return {
        "declared_path": raw_path,
        "resolved_path": None if resolved is None else str(resolved),
        "status": "resolved"
        if resolved is not None
        else "declared_reference_not_present_in_r4_artifact",
        "binding": deepcopy(dict(binding)),
    }


def _case_manifest(
    *,
    case: RouteCase,
    calibration: MaterializerCalibration,
    route_bank_path: Path,
    document: Mapping[str, Any],
    track_paths: Mapping[str, str],
) -> dict[str, Any]:
    camera = _camera_reference(route_bank_path, document, case.camera_binding)
    return {
        "artifact_kind": "m5_research_two_actor_region_case",
        "format_version": 1,
        "research_only": True,
        "formal_m2_request": False,
        "episode_counted": False,
        "receipt": {
            "status": "research_only",
            "formal_m2_request": False,
            "episode_counted": False,
        },
        "region_instance_id": case.region_instance_id,
        "route_family_id": case.route_family_id,
        "motion_case": case.motion_case,
        "camera_m1_reference": camera,
        "route_source_center_plan": {
            "authority": "r4 region source-center route plan",
            "camera_participation_in_root_generation": False,
            "source1_positions_m": case.source_positions_m["source1"].tolist(),
            "source2_positions_m": case.source_positions_m["source2"].tolist(),
            "route_descriptors": deepcopy(dict(case.route_descriptors)),
        },
        "actors": [
            {
                **dict(ACTOR_AXES[index]),
                "asset_id": calibration.asset_id,
                "asset_manifest_path": str(calibration.asset_path),
                "base_m2_request_path": str(calibration.base_request_path),
                "actor_track_path": track_paths[axis["actor_id"]],
            }
            for index, axis in enumerate(ACTOR_AXES)
        ],
        "identity_exchange": deepcopy(
            case.route_descriptors.get("identity_continuity", {})
        ),
        "native_pending": {
            "support_contact": None,
            "emitter_actual_position_m": None,
            "articulated_collision": None,
            "object_id": None,
            "native_execution": None,
            "rlr": None,
        },
    }


def materialize_region_actor_tracks(
    *,
    route_bank_path: str | Path,
    region_instance_id: str,
    asset_manifest_path: str | Path,
    base_m2_request_path: str | Path,
    output_root: str | Path,
) -> Path:
    validate_shared_asset_calibration(
        (asset_manifest_path, asset_manifest_path),
        (base_m2_request_path, base_m2_request_path),
    )
    route_path = Path(route_bank_path).expanduser().resolve(strict=True)
    output_unresolved = Path(output_root).expanduser()
    if output_unresolved.is_absolute():
        output_candidate = output_unresolved
    else:
        output_candidate = Path.cwd() / output_unresolved
    if output_candidate.exists() or output_candidate.is_symlink():
        raise _error(f"output root exists; refusing to replace: {output_candidate}")
    parent = output_candidate.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    policy = WorkspacePathPolicy.from_roots((parent,))
    destination = policy.resolve_output(
        output_candidate, owner="region actor-track output"
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
    try:
        document = load_json(route_path)
        calibration = load_materializer_calibration(
            asset_manifest_path=asset_manifest_path,
            base_m2_request_path=base_m2_request_path,
        )
        cases = load_region_cases(route_path, region_instance_id=region_instance_id)
        case_records: list[dict[str, Any]] = []
        for case in cases:
            case_id = (
                f"{case.region_instance_id.replace(':', '_')}_{case.route_family_id}_"
            )
            case_id += case.motion_case
            case_dir = staging / "cases" / case_id
            actor_paths: dict[str, str] = {}
            for axis in ACTOR_AXES:
                actor_dir = case_dir / str(axis["actor_id"])
                actor_dir.mkdir(parents=True, exist_ok=True)
                track = _build_actor_track(
                    case=case,
                    calibration=calibration,
                    actor_axis=axis,
                    route_positions=case.source_positions_m[str(axis["source_id"])],
                    motion_case=case.motion_case,
                )
                relative = (
                    Path("cases") / case_id / str(axis["actor_id"]) / "actor_track.json"
                )
                _write_no_replace(actor_dir / "actor_track.json", track)
                actor_paths[str(axis["actor_id"])] = relative.as_posix()
            manifest = _case_manifest(
                case=case,
                calibration=calibration,
                route_bank_path=route_path,
                document=document,
                track_paths=actor_paths,
            )
            _write_no_replace(case_dir / "case_manifest.json", manifest)
            case_records.append(
                {
                    "case_id": case_id,
                    "region_instance_id": case.region_instance_id,
                    "route_family_id": case.route_family_id,
                    "motion_case": case.motion_case,
                    "case_manifest": f"cases/{case_id}/case_manifest.json",
                    "actor_track_paths": actor_paths,
                }
            )
        pilot = {
            "artifact_kind": "m5_research_region_actor_track_pilot",
            "format_version": 1,
            "research_only": True,
            "formal_m2_request": False,
            "episode_counted": False,
            "receipt": {
                "status": "research_only",
                "formal_m2_request": False,
                "episode_counted": False,
            },
            "region_instance_id": region_instance_id,
            "route_bank_path": str(route_path),
            "asset_manifest_path": str(calibration.asset_path),
            "base_m2_request_path": str(calibration.base_request_path),
            "case_count": len(case_records),
            "actor_track_count": len(case_records) * 2,
            "motion_cases": list(MOTION_CASES),
            "cases": case_records,
            "native_pending": {
                "support_contact": None,
                "emitter_actual_position_m": None,
                "articulated_collision": None,
                "object_id": None,
                "native_execution": None,
                "rlr": None,
            },
        }
        _write_no_replace(staging / "pilot_manifest.json", pilot)
        return atomic_publish_directory(policy, staging, destination)
    except BaseException:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise


__all__ = [
    "ACTOR_AXES",
    "FRAME_COUNT",
    "FRAME_RATE_HZ",
    "MOTION_CASES",
    "MP3DRegionActorTrackError",
    "MaterializerCalibration",
    "RouteCase",
    "load_materializer_calibration",
    "load_region_cases",
    "materialize_region_actor_tracks",
    "validate_shared_asset_calibration",
]

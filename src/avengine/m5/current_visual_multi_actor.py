"""Validated CPU apply plans for two independent MP3D actor tracks.

This is the pre-native part of the multi-actor visual route.  It consumes the
M5 research-only actor-track materializer output, validates each actor against
its own exact M2 asset/action bundle, and returns immutable frame plans.  It
does not create a Simulator, render pixels, inspect visibility, or run audio.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from avengine.contracts.json_io import load_json, sha256_file
from avengine.contracts.transforms import normalized_quaternion_xyzw
from avengine.m2.contracts import (
    ValidatedM2Inputs,
    compute_applied_state_hash,
    compute_pose_hash,
    load_and_validate_inputs as load_m2_inputs,
)
from avengine.m2.habitat_capture import (
    compile_frame_applications,
    load_runtime_asset_bundle,
    quaternion_xyzw_to_matrix,
)


FRAME_COUNT = 75
FRAME_RATE_HZ = 15
TICKS_PER_FRAME = 3_200
CURRENT_ACTOR_IDS = ("actor0", "actor1")
CURRENT_SOURCE_IDS = ("source1", "source2")
MOTION_CASES = {
    "static_static": {"source1": "idle", "source2": "idle"},
    "s1_moving_s2_static": {"source1": "walk", "source2": "idle"},
    "s1_static_s2_moving": {"source1": "idle", "source2": "walk"},
    "both_moving": {"source1": "walk", "source2": "walk"},
}
PENDING_KEYS = (
    "support_contact",
    "emitter_actual_position_m",
    "articulated_collision",
    "object_id",
    "native_execution",
    "rlr",
)


class CurrentVisualMultiActorError(RuntimeError):
    """A research apply plan cannot be trusted."""


def _fail(message: str) -> CurrentVisualMultiActorError:
    return CurrentVisualMultiActorError(message)


def _regular_file(path: Path, *, owner: str, reject_symlink: bool = True) -> Path:
    if reject_symlink and path.is_symlink():
        raise _fail(f"{owner} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise _fail(f"{owner} is missing: {path}") from exc
    if not resolved.is_file():
        raise _fail(f"{owner} is not a regular file: {resolved}")
    return resolved


def _safe_relative_file(manifest_path: Path, raw: Any, *, owner: str) -> Path:
    """Resolve materializer paths without allowing case-directory escape.

    The materializer stores actor paths relative to the artifact root (under
    ``cases/``), while small fixtures may store them relative to the case
    directory. Both roots are explicit; ``..`` never broadens either root.
    """

    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise _fail(f"{owner} must be a relative path")
    relative = Path(raw)
    if relative.parts and relative.parts[0] == "cases":
        root = manifest_path.parents[2]
    else:
        root = manifest_path.parent
    root = root.resolve(strict=True)
    candidate = root / relative
    resolved = _regular_file(candidate, owner=owner)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise _fail(f"{owner} escapes its declared artifact root") from exc
    return resolved


def _finite_vector(value: Any, *, shape: tuple[int, ...], owner: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise _fail(f"{owner} must be finite with shape {shape}")
    return np.ascontiguousarray(array)


def _matrix_from_payload(value: Any, *, owner: str) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise _fail(f"{owner} must be a transform object")
    translation = _finite_vector(
        value.get("translation_m"), shape=(3,), owner=f"{owner}.translation_m"
    )
    try:
        quaternion = normalized_quaternion_xyzw(value.get("rotation_xyzw"))
        rotation = quaternion_xyzw_to_matrix(quaternion)
    except (TypeError, ValueError) as exc:
        raise _fail(f"{owner}.rotation_xyzw is invalid") from exc
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _rigid_matrix(value: Any, *, owner: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise _fail(f"{owner} must be a finite [4,4] matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise _fail(f"{owner} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise _fail(f"{owner} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise _fail(f"{owner} rotation determinant is not +1")
    return np.ascontiguousarray(matrix)


def _matrix_tuple(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(component) for component in row) for row in value)


def _vector_tuple(value: Any, *, owner: str) -> tuple[float, float, float]:
    array = _finite_vector(value, shape=(3,), owner=owner)
    return tuple(float(item) for item in array)  # type: ignore[return-value]


def _pending_nulls(value: Any, *, owner: str) -> None:
    if not isinstance(value, Mapping) or any(
        key not in value or value[key] is not None for key in PENDING_KEYS
    ):
        raise _fail(f"{owner} must keep all native outputs pending/null")


@dataclass(frozen=True)
class ActorFrameApplyPlan:
    frame_index: int
    pts_ticks: int
    action_id: str
    action_sample_index: int
    route_source_center_m: tuple[float, float, float]
    planned_world_from_actor: tuple[tuple[float, ...], ...]
    derived_world_from_skin_root: tuple[tuple[float, ...], ...]
    joint_states: tuple[tuple[str, tuple[float, float, float, float]], ...]


@dataclass(frozen=True)
class ActorTrackApplyPlan:
    actor_id: str
    source_id: str
    semantic_id: int
    emitter_link_name: str
    asset_id: str
    frames: tuple[ActorFrameApplyPlan, ...]


@dataclass(frozen=True)
class MultiActorApplyPlan:
    case_id: str
    region_instance_id: str
    route_family_id: str
    motion_case: str
    camera_m1_request_path: str
    actor_tracks: tuple[ActorTrackApplyPlan, ActorTrackApplyPlan]
    frame_count: int = FRAME_COUNT
    frame_rate_hz: int = FRAME_RATE_HZ
    ticks_per_frame: int = TICKS_PER_FRAME


@dataclass(frozen=True)
class ActorTrackInput:
    actor_id: str
    source_id: str
    semantic_id: int
    emitter_link_name: str
    track_path: Path
    track: Mapping[str, Any]
    m2_inputs: ValidatedM2Inputs
    bundle: Any


@dataclass(frozen=True)
class MultiActorCaseInput:
    manifest_path: Path
    manifest: Mapping[str, Any]
    actor_tracks: tuple[ActorTrackInput, ActorTrackInput]
    camera_m1_request_path: Path


def _expected_action(case_motion: str, source_id: str) -> str:
    try:
        return MOTION_CASES[case_motion][source_id]
    except KeyError as exc:
        raise _fail(
            f"unsupported motion/source axis: {case_motion}/{source_id}"
        ) from exc


def _contact_order(m2_inputs: ValidatedM2Inputs) -> tuple[str, ...]:
    contacts = m2_inputs.asset.get("contacts")
    order = contacts.get("contact_order") if isinstance(contacts, Mapping) else None
    if (
        not isinstance(order, list)
        or not order
        or not all(isinstance(item, str) and item for item in order)
    ):
        raise _fail("asset contact_order is missing or invalid")
    return tuple(order)


def _clip_for(bundle: Any, action_id: str) -> Any:
    try:
        role = bundle.action_roles_by_id[action_id]
        return bundle.action_sets_by_role[role].action(action_id)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise _fail(f"runtime bundle lacks validated action {action_id!r}") from exc


def _validate_track(
    *,
    region_instance_id: str,
    route_family_id: str,
    case_motion: str,
    actor_record: Mapping[str, Any],
    manifest_path: Path,
) -> ActorTrackInput:
    track_path = _safe_relative_file(
        manifest_path, actor_record.get("actor_track_path"), owner="actor track"
    )
    track = load_json(track_path)
    if not isinstance(track, Mapping):
        raise _fail(f"actor track is not an object: {track_path}")
    if (
        track.get("research_only") is not True
        or track.get("formal_m2_request") is not False
        or track.get("episode_counted") is not False
        or track.get("receipt")
        != {
            "status": "research_only",
            "formal_m2_request": False,
            "episode_counted": False,
        }
    ):
        raise _fail(f"track is not research-only: {track_path}")
    _pending_nulls(track.get("native_pending"), owner=f"track {track_path}")

    actor_id = actor_record.get("actor_id")
    source_id = actor_record.get("source_id")
    semantic_id = actor_record.get("semantic_id")
    emitter_link_name = actor_record.get("emitter_link_name")
    if not isinstance(actor_id, str) or not isinstance(source_id, str):
        raise _fail(f"track identity is not textual: {track_path}")
    if not isinstance(semantic_id, int) or isinstance(semantic_id, bool):
        raise _fail(f"semantic_id is not an integer: {track_path}")
    if not isinstance(emitter_link_name, str) or not emitter_link_name:
        raise _fail(f"emitter_link_name is invalid: {track_path}")
    if any(
        track.get(key) != actor_record.get(key)
        for key in ("actor_id", "source_id", "semantic_id", "emitter_link_name")
    ):
        raise _fail(f"track identity differs from case manifest: {track_path}")

    asset_raw = track.get("asset_manifest_path")
    request_raw = track.get("base_m2_request_path")
    if not isinstance(asset_raw, str) or not isinstance(request_raw, str):
        raise _fail(f"track lacks absolute asset/base-request paths: {track_path}")
    asset_path = _regular_file(Path(asset_raw), owner="actor asset manifest")
    request_path = _regular_file(Path(request_raw), owner="actor base M2 request")
    try:
        m2_inputs = load_m2_inputs(asset_path, request_path)
        bundle = load_runtime_asset_bundle(m2_inputs)
        compile_frame_applications(m2_inputs, bundle)
    except Exception as exc:
        raise _fail(f"base M2 validation failed for {track_path}: {exc}") from exc
    if track.get("asset_id") != m2_inputs.asset.get("asset_id"):
        raise _fail(f"track asset_id differs from base M2 asset: {track_path}")
    declared_sha = track.get("asset_manifest_sha256")
    if not isinstance(declared_sha, str) or declared_sha != sha256_file(asset_path):
        raise _fail(f"track asset manifest hash differs from exact file: {track_path}")

    route = track.get("route_source_center_plan")
    states = track.get("states")
    frames = track.get("frames")
    if (
        not isinstance(route, Mapping)
        or not isinstance(states, list)
        or not isinstance(frames, list)
    ):
        raise _fail(f"track route/states/frames are missing: {track_path}")
    if (
        route.get("region_instance_id") != region_instance_id
        or route.get("route_family_id") != route_family_id
        or route.get("motion_case") != case_motion
        or route.get("source_id") != source_id
    ):
        raise _fail(f"track route identity differs from case: {track_path}")
    if route.get("camera_participation_in_root_generation", False) is not False:
        raise _fail(f"camera participates in route root generation: {track_path}")
    if route.get("route_point_to_actor_root_offset_m") != [0.0, 0.0, 0.0]:
        raise _fail(
            f"track route->actor calibration is not explicit zero: {track_path}"
        )
    positions = _finite_vector(
        route.get("positions_m"), shape=(FRAME_COUNT, 3), owner=f"{track_path} route"
    )
    expected_action = _expected_action(case_motion, source_id)
    displacement = np.linalg.norm(positions - positions[0], axis=1)
    if expected_action == "idle" and float(np.max(displacement)) > 1.0e-8:
        raise _fail(f"idle track has moving route points: {track_path}")
    if expected_action == "walk" and float(np.max(displacement)) <= 1.0e-8:
        raise _fail(f"walk track has no route displacement: {track_path}")
    runtime_order = tuple(bundle.joint_mapping["runtime_joint_order"])
    contact_order = _contact_order(m2_inputs)
    clip = _clip_for(bundle, expected_action)
    expected_asset_hash = str(declared_sha)
    if len(states) != FRAME_COUNT or len(frames) != FRAME_COUNT:
        raise _fail(f"track does not contain exactly 75 states/frames: {track_path}")
    actor_from_skin_root = _rigid_matrix(
        bundle.actor_from_skin_root, owner=f"{track_path} actor_from_skin_root"
    )
    for index, (state, frame) in enumerate(zip(states, frames, strict=True)):
        if not isinstance(state, Mapping) or not isinstance(frame, Mapping):
            raise _fail(f"track frame {index} is not an object: {track_path}")
        expected_ticks = index * TICKS_PER_FRAME
        if any(
            item.get("frame_index") != index or item.get("pts_ticks") != expected_ticks
            for item in (state, frame)
        ):
            raise _fail(f"track frame/PTS axis is invalid at {track_path}:{index}")
        if any(item.get("action_id") != expected_action for item in (state, frame)):
            raise _fail(f"track action axis is invalid at {track_path}:{index}")
        if state.get("action_time_ticks") != expected_ticks:
            raise _fail(f"track action clock is invalid at {track_path}:{index}")
        effective_tick = expected_ticks % int(clip.loop_duration_ticks)
        try:
            sample_index = clip.sample_ticks.index(effective_tick)
        except ValueError as exc:
            raise _fail(
                f"track action tick is not baked at {track_path}:{index}"
            ) from exc
        if state.get("effective_action_tick") != effective_tick:
            raise _fail(f"track effective action tick differs at {track_path}:{index}")
        if (
            state.get("action_sample_index") != sample_index
            or frame.get("action_sample_index") != sample_index
        ):
            raise _fail(f"track action sample differs at {track_path}:{index}")
        root = _matrix_from_payload(state.get("root_transform"), owner="track root")
        planned = _matrix_from_payload(
            frame.get("planned_world_from_actor"), owner="planned actor"
        )
        if not np.allclose(root, planned, rtol=0.0, atol=1.0e-7):
            raise _fail(f"root/planned actor mismatch at {track_path}:{index}")
        if not np.allclose(root[:3, 3], positions[index], rtol=0.0, atol=1.0e-8):
            raise _fail(f"route/planned actor mismatch at {track_path}:{index}")
        derived = _matrix_from_payload(
            frame.get("derived_world_from_skin_root"), owner="derived skin root"
        )
        if not np.allclose(derived, root @ actor_from_skin_root, rtol=0.0, atol=1.0e-7):
            raise _fail(f"planned skin-root mismatch at {track_path}:{index}")
        if frame.get("route_source_center_m") != list(
            positions[index]
        ) and not np.allclose(
            _finite_vector(
                frame.get("route_source_center_m"),
                shape=(3,),
                owner="frame route source center",
            ),
            positions[index],
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise _fail(f"frame route source mismatch at {track_path}:{index}")
        if any(
            frame.get(key) is not None
            for key in (
                "emitter_actual_position_m",
                "paw_support_contact",
                "articulated_collision",
                "object_id_modal",
            )
        ):
            raise _fail(f"native frame evidence is not pending at {track_path}:{index}")
        joints = state.get("joint_states")
        contacts = state.get("contact_states")
        if (
            not isinstance(joints, list)
            or tuple(
                item.get("joint_id") for item in joints if isinstance(item, Mapping)
            )
            != runtime_order
        ):
            raise _fail(f"track joint axis differs at {track_path}:{index}")
        if (
            not isinstance(contacts, list)
            or tuple(
                item.get("contact_id") for item in contacts if isinstance(item, Mapping)
            )
            != contact_order
        ):
            raise _fail(f"track contact axis differs at {track_path}:{index}")
        baked = np.asarray(clip.rotations_xyzw[sample_index], dtype=np.float64)
        declared = np.asarray(
            [item.get("rotation_xyzw") for item in joints], dtype=np.float64
        )
        if declared.shape != baked.shape or not np.allclose(
            declared, baked, rtol=0.0, atol=1.0e-9
        ):
            raise _fail(f"track joint sample differs at {track_path}:{index}")
        if compute_pose_hash(m2_inputs.asset, state) != state.get("pose_hash"):
            raise _fail(f"track pose hash differs at {track_path}:{index}")
        if compute_applied_state_hash(
            m2_inputs.asset, state, asset_manifest_sha256=expected_asset_hash
        ) != state.get("applied_state_hash"):
            raise _fail(f"track applied-state hash differs at {track_path}:{index}")
    return ActorTrackInput(
        actor_id=actor_id,
        source_id=source_id,
        semantic_id=semantic_id,
        emitter_link_name=emitter_link_name,
        track_path=track_path,
        track=track,
        m2_inputs=m2_inputs,
        bundle=bundle,
    )


def load_multi_actor_case(case_manifest_path: str | Path) -> MultiActorCaseInput:
    """Load and validate one materialized two-actor case without native runtime."""

    manifest_path = _regular_file(Path(case_manifest_path), owner="case manifest")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise _fail("case manifest is not an object")
    if (
        manifest.get("research_only") is not True
        or manifest.get("formal_m2_request") is not False
        or manifest.get("episode_counted") is not False
        or manifest.get("receipt")
        != {
            "status": "research_only",
            "formal_m2_request": False,
            "episode_counted": False,
        }
    ):
        raise _fail("case manifest is not research-only")
    for key in ("region_instance_id", "route_family_id", "motion_case"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise _fail(f"case manifest lacks {key}")
    motion_case = manifest["motion_case"]
    if motion_case not in MOTION_CASES:
        raise _fail(f"unsupported case motion: {motion_case}")
    _pending_nulls(manifest.get("native_pending"), owner="case manifest")
    camera = manifest.get("camera_m1_reference")
    if not isinstance(camera, Mapping) or camera.get("status") != "resolved":
        raise _fail("case M1 camera reference is not resolved")
    camera_path = camera.get("resolved_path")
    if not isinstance(camera_path, str):
        raise _fail("case M1 camera reference lacks resolved_path")
    camera_file = _regular_file(Path(camera_path), owner="camera M1 request")
    route = manifest.get("route_source_center_plan")
    if (
        not isinstance(route, Mapping)
        or route.get("camera_participation_in_root_generation") is not False
    ):
        raise _fail("case route plan does not prove camera independence")
    actors = manifest.get("actors")
    if not isinstance(actors, list) or len(actors) != 2:
        raise _fail("case manifest must contain exactly two actors")
    expected_sources = dict(zip(CURRENT_ACTOR_IDS, CURRENT_SOURCE_IDS, strict=True))
    for record in actors:
        if not isinstance(record, Mapping):
            raise _fail("case actor records must be objects")
        if record.get("actor_id") not in expected_sources:
            raise _fail("case actor axis must remain actor0/actor1")
        if record.get("source_id") != expected_sources[record["actor_id"]]:
            raise _fail("case actor/source cross-wire is not allowed")
    tracks = tuple(
        _validate_track(
            region_instance_id=manifest["region_instance_id"],
            route_family_id=manifest["route_family_id"],
            case_motion=motion_case,
            actor_record=record,
            manifest_path=manifest_path,
        )
        for record in actors
    )
    if tuple(item.actor_id for item in tracks) != CURRENT_ACTOR_IDS:
        raise _fail("case actor axis must remain actor0/actor1")
    if tuple(item.source_id for item in tracks) != CURRENT_SOURCE_IDS:
        raise _fail("case source axis must remain actor0/source1, actor1/source2")
    if len({item.semantic_id for item in tracks}) != 2:
        raise _fail("case semantic IDs must be unique")
    return MultiActorCaseInput(
        manifest_path=manifest_path,
        manifest=manifest,
        actor_tracks=(tracks[0], tracks[1]),
        camera_m1_request_path=camera_file,
    )


def _build_track_apply_plan(track_input: ActorTrackInput) -> ActorTrackApplyPlan:
    """Convert a validated track into immutable native apply values."""

    track = track_input.track
    route = track["route_source_center_plan"]
    positions = _finite_vector(
        route["positions_m"], shape=(FRAME_COUNT, 3), owner="track route"
    )
    actor_from_skin_root = _rigid_matrix(
        track_input.bundle.actor_from_skin_root, owner="actor_from_skin_root"
    )
    frames: list[ActorFrameApplyPlan] = []
    for index, (state, frame) in enumerate(
        zip(track["states"], track["frames"], strict=True)
    ):
        if (
            state.get("frame_index") != index
            or frame.get("frame_index") != index
            or state.get("pts_ticks") != index * TICKS_PER_FRAME
            or frame.get("pts_ticks") != index * TICKS_PER_FRAME
            or state.get("pts_ticks") != frame.get("pts_ticks")
            or state.get("action_id") != frame.get("action_id")
            or state.get("action_sample_index") != frame.get("action_sample_index")
        ):
            raise _fail(f"track frame clock/action mismatch at frame {index}")
        root = _matrix_from_payload(state["root_transform"], owner="root_transform")
        planned = _matrix_from_payload(
            frame["planned_world_from_actor"], owner="planned_world_from_actor"
        )
        derived = _matrix_from_payload(
            frame["derived_world_from_skin_root"],
            owner="derived_world_from_skin_root",
        )
        if not np.allclose(root, planned, rtol=0.0, atol=1.0e-7):
            raise _fail(f"root/planned actor mismatch at frame {index}")
        if not np.allclose(root[:3, 3], positions[index], rtol=0.0, atol=1.0e-8):
            raise _fail(f"route/planned actor mismatch at frame {index}")
        if not np.allclose(derived, root @ actor_from_skin_root, rtol=0.0, atol=1.0e-7):
            raise _fail(f"planned skin-root mismatch at frame {index}")
        frame_route = _finite_vector(
            frame["route_source_center_m"],
            shape=(3,),
            owner="route_source_center_m",
        )
        if not np.allclose(frame_route, positions[index], rtol=0.0, atol=1.0e-8):
            raise _fail(f"frame route source mismatch at frame {index}")
        joints: list[tuple[str, tuple[float, float, float, float]]] = []
        for joint in state["joint_states"]:
            if not isinstance(joint, Mapping):
                raise _fail(f"joint state {index} is not an object")
            rotation = normalized_quaternion_xyzw(joint["rotation_xyzw"])
            if not np.all(np.isfinite(rotation)):
                raise _fail(f"joint rotation {index} is not finite")
            joints.append(
                (
                    str(joint["joint_id"]),
                    tuple(float(component) for component in rotation),
                )
            )
        frames.append(
            ActorFrameApplyPlan(
                frame_index=index,
                pts_ticks=int(state["pts_ticks"]),
                action_id=str(state["action_id"]),
                action_sample_index=int(state["action_sample_index"]),
                route_source_center_m=_vector_tuple(
                    frame["route_source_center_m"], owner="route_source_center_m"
                ),
                planned_world_from_actor=_matrix_tuple(planned),
                derived_world_from_skin_root=_matrix_tuple(derived),
                joint_states=tuple(joints),
            )
        )
    return ActorTrackApplyPlan(
        actor_id=track_input.actor_id,
        source_id=track_input.source_id,
        semantic_id=track_input.semantic_id,
        emitter_link_name=track_input.emitter_link_name,
        asset_id=str(track_input.m2_inputs.asset["asset_id"]),
        frames=tuple(frames),
    )


def build_apply_plan(case: MultiActorCaseInput) -> MultiActorApplyPlan:
    """Build two independent 75-frame plans; camera data never enters them."""

    left, right = (_build_track_apply_plan(item) for item in case.actor_tracks)
    for index, (left_frame, right_frame) in enumerate(
        zip(left.frames, right.frames, strict=True)
    ):
        if (
            left_frame.frame_index != right_frame.frame_index
            or left_frame.pts_ticks != right_frame.pts_ticks
        ):
            raise _fail(f"actor clocks differ at frame {index}")
    return MultiActorApplyPlan(
        case_id=(f"{case.manifest['route_family_id']}::{case.manifest['motion_case']}"),
        region_instance_id=str(case.manifest["region_instance_id"]),
        route_family_id=str(case.manifest["route_family_id"]),
        motion_case=str(case.manifest["motion_case"]),
        camera_m1_request_path=str(case.camera_m1_request_path),
        actor_tracks=(left, right),
    )


def _contact_value(contact: Any, name: str) -> Any:
    if isinstance(contact, Mapping) and name in contact:
        return contact[name]
    if hasattr(contact, name):
        return getattr(contact, name)
    raise _fail(f"contact lacks required field {name}")


def _contact_int(contact: Any, name: str) -> int:
    value = _contact_value(contact, name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise _fail(f"contact {name} must be an integer")
    return int(value)


def _contact_float(contact: Any, name: str) -> float:
    try:
        value = float(_contact_value(contact, name))
    except (TypeError, ValueError) as exc:
        raise _fail(f"contact {name} must be numeric") from exc
    if not math.isfinite(value):
        raise _fail(f"contact {name} must be finite")
    return value


def _contact_vector(contact: Any, name: str) -> tuple[float, float, float]:
    value = _contact_value(contact, name)
    if isinstance(value, Mapping):
        value = [value.get(axis) for axis in ("x", "y", "z")]
    array = _finite_vector(value, shape=(3,), owner=f"contact {name}")
    return tuple(float(item) for item in array)  # type: ignore[return-value]


def _serialize_contact(contact: Any) -> dict[str, Any]:
    is_active = _contact_value(contact, "is_active")
    if not isinstance(is_active, (bool, np.bool_)):
        raise _fail("contact is_active must be a bool")
    record = {
        "object_id_a": _contact_int(contact, "object_id_a"),
        "object_id_b": _contact_int(contact, "object_id_b"),
        "link_id_a": _contact_int(contact, "link_id_a"),
        "link_id_b": _contact_int(contact, "link_id_b"),
        "position_on_a_in_ws": list(_contact_vector(contact, "position_on_a_in_ws")),
        "position_on_b_in_ws": list(_contact_vector(contact, "position_on_b_in_ws")),
        "contact_normal_on_b_in_ws": list(
            _contact_vector(contact, "contact_normal_on_b_in_ws")
        ),
        "contact_distance": _contact_float(contact, "contact_distance"),
        "is_active": bool(is_active),
    }
    if (
        isinstance(contact, Mapping)
        and "normal_force" in contact
        or hasattr(contact, "normal_force")
    ):
        record["normal_force"] = _contact_float(contact, "normal_force")
    return record


@dataclass(frozen=True)
class CollisionFrameSummary:
    actor_actor_contacts: tuple[dict[str, Any], ...]
    actor_stage_contacts: tuple[dict[str, Any], ...]
    actor_other_contacts: tuple[dict[str, Any], ...]
    unrelated_contacts: tuple[dict[str, Any], ...]
    actor_pair_collision_free: bool


def classify_collision_contacts(
    contacts: Iterable[Any],
    *,
    runtime_actor_object_ids: Iterable[int],
    stage_object_id: int = 0,
) -> CollisionFrameSummary:
    """Classify raw ContactPointData without inferring support or legality."""

    actor_ids = set()
    for value in runtime_actor_object_ids:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise _fail("runtime actor object IDs must be integers")
        actor_ids.add(int(value))
    if len(actor_ids) != 2:
        raise _fail("exactly two runtime actor object IDs are required")
    if isinstance(stage_object_id, bool) or not isinstance(
        stage_object_id, (int, np.integer)
    ):
        raise _fail("stage_object_id must be an integer")
    stage_object_id = int(stage_object_id)
    actor_actor: list[dict[str, Any]] = []
    actor_stage: list[dict[str, Any]] = []
    actor_other: list[dict[str, Any]] = []
    unrelated: list[dict[str, Any]] = []
    for contact in contacts:
        record = _serialize_contact(contact)
        object_a = record["object_id_a"]
        object_b = record["object_id_b"]
        has_a = object_a in actor_ids
        has_b = object_b in actor_ids
        if has_a and has_b:
            record["scope"] = "actor_pair"
            actor_actor.append(record)
        elif has_a or has_b:
            record["scope"] = "actor_stage"
            other_id = object_b if has_a else object_a
            if other_id == stage_object_id:
                record["stage_contact_recorded_unclassified"] = True
                actor_stage.append(record)
            else:
                record["scope"] = "actor_other"
                actor_other.append(record)
        else:
            record["scope"] = "unrelated"
            unrelated.append(record)
    return CollisionFrameSummary(
        actor_actor_contacts=tuple(actor_actor),
        actor_stage_contacts=tuple(actor_stage),
        actor_other_contacts=tuple(actor_other),
        unrelated_contacts=tuple(unrelated),
        actor_pair_collision_free=not actor_actor,
    )


__all__ = [
    "ActorFrameApplyPlan",
    "ActorTrackApplyPlan",
    "CollisionFrameSummary",
    "CurrentVisualMultiActorError",
    "MultiActorApplyPlan",
    "MultiActorCaseInput",
    "ActorTrackInput",
    "build_apply_plan",
    "classify_collision_contacts",
    "load_multi_actor_case",
]

"""CPU materialization of MP3D region routes into Habitat actor apply tracks.

This module is the boundary between a planned MP3D route and a future native
Habitat capture.  It loads each explicitly supplied M2 animal package and its
base request, samples the package's baked action loops at the requested clock,
and writes root/joint apply targets.  Route points remain PathFinder actor-root
centres; emitter positions and all native readback are intentionally pending.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.assets.contracts import (
    ValidatedM2Inputs,
    load_and_validate_inputs as load_m2_inputs,
)
from avengine.assets.habitat_capture import (
    RuntimeAssetBundle,
    load_runtime_asset_bundle,
)
from avengine.camera_pose import CameraPoseError
from avengine.contracts.transforms import normalized_quaternion_xyzw
from avengine.rooms.contracts import (
    ContractError,
    load_and_validate_inputs as load_m1_inputs,
)
from avengine.routes.trajectory import (
    M6XTrajectoryError,
    resample_polyline_by_arc_length,
)
from avengine.timeline.current_mp3d_dynamic_audio import (
    CurrentMP3DDynamicAudioError,
    _resolve_visual_clock,
)


ACTOR_TRACK_SCHEMA = "avengine_mp3d_region_actor_track_v1"
CASE_SCHEMA = "avengine_mp3d_region_actor_track_case_v1"
RECEIPT_SCHEMA = "avengine_mp3d_region_actor_track_materialization_v1"
_SLOT_RE = re.compile(r"source([1-9][0-9]*)\Z")
_EPSILON_M = 1.0e-9


class MP3DRegionActorTrackError(ValueError):
    """A planned route cannot become a truthful Habitat apply track."""


def _read_json(path: str | Path, *, owner: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise MP3DRegionActorTrackError(f"{owner} must be a regular file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MP3DRegionActorTrackError(f"cannot read {owner}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MP3DRegionActorTrackError(f"{owner} must be a JSON object")
    return deepcopy(dict(value))


def _finite_vector(value: Any, *, owner: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise MP3DRegionActorTrackError(f"{owner} must be a finite 3-vector")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DRegionActorTrackError(f"{owner} must be a finite 3-vector") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise MP3DRegionActorTrackError(f"{owner} must be a finite 3-vector")
    return np.ascontiguousarray(result)


def _slot_index(value: Any, *, owner: str) -> int:
    if not isinstance(value, str):
        raise MP3DRegionActorTrackError(f"{owner} must be sourceN")
    match = _SLOT_RE.fullmatch(value)
    if match is None:
        raise MP3DRegionActorTrackError(f"{owner} must be sourceN")
    return int(match.group(1))


def _positive_int(value: Any, *, owner: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MP3DRegionActorTrackError(f"{owner} must be an integer >= {minimum}")
    return int(value)


def _fresh_output(path: str | Path) -> Path:
    output = Path(path).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise MP3DRegionActorTrackError(
            f"refusing to replace actor-track output: {output}"
        )
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_actor_specs(
    value: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...], Path | None]:
    if isinstance(value, (str, Path)):
        source_path = Path(value).expanduser().resolve()
        document = _read_json(source_path, owner="actor track configuration")
    else:
        source_path = None
        if isinstance(value, Mapping):
            document = deepcopy(dict(value))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            document = {"actors": deepcopy(list(value))}
        else:
            raise MP3DRegionActorTrackError(
                "actor track configuration must be an actors list or object"
            )
    actors = document.get("actors")
    if not isinstance(actors, list) or len(actors) < 2:
        raise MP3DRegionActorTrackError(
            "actor track configuration must contain at least two actors"
        )
    indexed: list[tuple[int, Mapping[str, Any]]] = []
    semantic_ids: set[int] = set()
    endpoint_ids: set[str] = set()
    actor_ids: set[str] = set()
    for ordinal, actor in enumerate(actors):
        if not isinstance(actor, Mapping):
            raise MP3DRegionActorTrackError(f"actors[{ordinal}] must be an object")
        slot = actor.get("source_slot_id")
        index = _slot_index(slot, owner=f"actors[{ordinal}].source_slot_id")
        actor_id = actor.get("actor_id")
        endpoint_id = actor.get("source_endpoint_id")
        if not isinstance(actor_id, str) or not actor_id:
            raise MP3DRegionActorTrackError(f"actors[{ordinal}].actor_id is required")
        if not isinstance(endpoint_id, str) or not endpoint_id:
            raise MP3DRegionActorTrackError(
                f"actors[{ordinal}].source_endpoint_id is required"
            )
        if actor_id in actor_ids or endpoint_id in endpoint_ids:
            raise MP3DRegionActorTrackError(
                "actor_id and source_endpoint_id must be unique"
            )
        actor_ids.add(actor_id)
        endpoint_ids.add(endpoint_id)
        semantic_id = actor.get("semantic_id")
        if (
            isinstance(semantic_id, bool)
            or not isinstance(semantic_id, int)
            or semantic_id < 0
            or semantic_id in semantic_ids
        ):
            raise MP3DRegionActorTrackError(
                f"actors[{ordinal}].semantic_id must be a unique nonnegative integer"
            )
        semantic_ids.add(semantic_id)
        for key in (
            "asset_id",
            "asset_revision",
            "asset_manifest_path",
            "base_m2_request_path",
            "emitter_anchor_id",
        ):
            if not isinstance(actor.get(key), str) or not actor[key]:
                raise MP3DRegionActorTrackError(
                    f"actors[{ordinal}].{key} is required"
                )
        offset = _finite_vector(
            actor.get("route_to_actor_root_offset_m"),
            owner=f"actors[{ordinal}].route_to_actor_root_offset_m",
        )
        if not np.all(np.isfinite(offset)):
            raise MP3DRegionActorTrackError(
                f"actors[{ordinal}].route_to_actor_root_offset_m is invalid"
            )
        indexed.append((index, actor))
    indexed.sort(key=lambda item: item[0])
    if tuple(index for index, _actor in indexed) != tuple(
        range(1, len(indexed) + 1)
    ):
        raise MP3DRegionActorTrackError(
            "actor track slots must be the contiguous source1..sourceN sequence"
        )
    return document, tuple(actor for _index, actor in indexed), source_path


def _load_planned_inputs(
    *,
    region_plan_path: str | Path,
    planned_timeline_path: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    plan = _read_json(region_plan_path, owner="region plan")
    if (
        plan.get("artifact_kind") != "mp3d_region_source_route_plan"
        or plan.get("research_only") is not True
        or plan.get("episode_counted") is not False
    ):
        raise MP3DRegionActorTrackError(
            "region plan must be the research-only MP3D source-route plan"
        )
    timeline = _read_json(planned_timeline_path, owner="planned timeline")
    if (
        timeline.get("artifact_role") != "planned_timeline_not_native_capture"
        or timeline.get("research_only") is not True
        or timeline.get("episode_counted") is not False
    ):
        raise MP3DRegionActorTrackError(
            "planned timeline is not explicitly marked as non-native planned data"
        )
    plan_region = timeline.get("region")
    if not isinstance(plan_region, Mapping):
        raise MP3DRegionActorTrackError("planned timeline has no region identity")
    plan_family_id = timeline.get("route_family_id")
    motion_case = timeline.get("motion_case")
    if not isinstance(plan_family_id, str) or not isinstance(motion_case, str):
        raise MP3DRegionActorTrackError(
            "planned timeline lacks route_family_id or motion_case"
        )
    region_index = plan_region.get("region_index")
    regions = [
        item
        for item in plan.get("regions", [])
        if isinstance(item, Mapping) and item.get("region_index") == region_index
    ]
    if len(regions) != 1 or regions[0].get("region_instance_id") != plan_region.get(
        "region_instance_id"
    ):
        raise MP3DRegionActorTrackError(
            "planned timeline region does not resolve in the supplied region plan"
        )
    families = [
        item
        for item in regions[0].get("route_families", [])
        if isinstance(item, Mapping) and item.get("route_family_id") == plan_family_id
    ]
    if len(families) != 1 or motion_case not in (families[0].get("cases") or {}):
        raise MP3DRegionActorTrackError(
            "planned timeline route family/case does not resolve in the region plan"
        )
    try:
        room_inputs = load_m1_inputs(room_manifest_path, m1_request_path)
    except (OSError, TypeError, ValueError, ContractError) as exc:
        raise MP3DRegionActorTrackError(f"M1 inputs are invalid: {exc}") from exc
    room_id = room_inputs.room.get("room_id")
    house_id = plan.get("house_id")
    if (
        not isinstance(room_id, str)
        or not isinstance(house_id, str)
        or not room_id.endswith(house_id)
    ):
        raise MP3DRegionActorTrackError(
            f"M1 room_id {room_id!r} does not identify plan house {house_id!r}"
        )
    if timeline.get("room", {}).get("room_id") != room_id:
        raise MP3DRegionActorTrackError(
            "planned timeline room_id differs from the supplied M1 request"
        )
    return plan, timeline, room_inputs, families[0]


def _resolve_clock(
    timeline: Mapping[str, Any],
    *,
    frame_count: int | None,
    frame_rate_hz: int | float | None,
    time_base_hz: int | None,
    ticks_per_frame: int | None,
) -> dict[str, int | float]:
    render = timeline.get("render")
    if not isinstance(render, Mapping):
        raise MP3DRegionActorTrackError("planned timeline has no render clock")
    base_time = time_base_hz if time_base_hz is not None else render.get("time_base_hz")
    if base_time is None:
        base_time = 48_000
    base_frames = render.get("frame_count")
    base_rate = render.get("frame_rate_hz")
    requested_frames = frame_count if frame_count is not None else base_frames
    requested_rate = frame_rate_hz if frame_rate_hz is not None else base_rate
    requested_ticks = ticks_per_frame
    if requested_ticks is None and frame_count is None and frame_rate_hz is None:
        requested_ticks = render.get("ticks_per_frame")
    try:
        clock = _resolve_visual_clock(
            frame_count=requested_frames,
            frame_rate_hz=requested_rate,
            ticks_per_frame=requested_ticks,
            time_base_hz=base_time,
        )
    except (CurrentMP3DDynamicAudioError, TypeError, ValueError) as exc:
        raise MP3DRegionActorTrackError(
            f"planned/current actor clock is invalid: {exc}"
        ) from exc
    if int(clock["frame_count"]) < 2:
        raise MP3DRegionActorTrackError("actor tracks need at least two frames")
    return clock


def _timeline_actor_records(
    timeline: Mapping[str, Any],
    actors: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    timeline_actors = timeline.get("actors")
    if not isinstance(timeline_actors, list):
        raise MP3DRegionActorTrackError("planned timeline has no actor records")
    by_slot = {
        item.get("source_slot_id"): item
        for item in timeline_actors
        if isinstance(item, Mapping)
    }
    records: list[Mapping[str, Any]] = []
    for actor in actors:
        slot = actor["source_slot_id"]
        record = by_slot.get(slot)
        if not isinstance(record, Mapping):
            raise MP3DRegionActorTrackError(
                f"planned timeline has no actor slot {slot!r}"
            )
        identity_keys = (
            ("actor_id", "actor_id"),
            ("source_endpoint_id", "source_endpoint_id"),
            ("asset_id", "asset_id"),
            ("revision", "asset_revision"),
        )
        for timeline_key, actor_key in identity_keys:
            if record.get(timeline_key) != actor.get(actor_key):
                raise MP3DRegionActorTrackError(
                    f"actor slot {slot} {timeline_key} differs between plan "
                    "and package mapping"
                )
        records.append(record)
    if len(timeline_actors) != len(actors):
        raise MP3DRegionActorTrackError(
            "planned timeline actor count differs from explicit package mappings"
        )
    return tuple(records)


def _resample_route_points(
    points: np.ndarray,
    *,
    target_frame_count: int,
    owner: str,
) -> np.ndarray:
    moving = bool(
        np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > _EPSILON_M)
    )
    if target_frame_count == len(points):
        materialized = points.copy()
    elif not moving:
        materialized = np.repeat(points[:1], target_frame_count, axis=0)
    else:
        try:
            materialized = resample_polyline_by_arc_length(
                points,
                target_frame_count,
                owner=owner,
            )
        except M6XTrajectoryError as exc:
            raise MP3DRegionActorTrackError(str(exc)) from exc
    return np.ascontiguousarray(materialized, dtype=np.float64)


def _route_positions_from_plan(
    family: Mapping[str, Any],
    *,
    motion_case: str,
    actors: Sequence[Mapping[str, Any]],
    target_frame_count: int,
) -> dict[str, np.ndarray]:
    cases = family.get("cases")
    case = cases.get(motion_case) if isinstance(cases, Mapping) else None
    if not isinstance(case, Mapping):
        raise MP3DRegionActorTrackError(
            f"region plan route family has no {motion_case!r} case"
        )
    result: dict[str, np.ndarray] = {}
    for actor in actors:
        slot = str(actor["source_slot_id"])
        key = f"{slot}_positions_m"
        try:
            points = np.asarray(case[key], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise MP3DRegionActorTrackError(
                f"region plan case lacks numeric {key} route positions"
            ) from exc
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise MP3DRegionActorTrackError(
                f"region plan {key} must contain at least two 3D points"
            )
        if not np.all(np.isfinite(points)):
            raise MP3DRegionActorTrackError(
                f"region plan {key} contains non-finite points"
            )
        result[slot] = _resample_route_points(
            np.ascontiguousarray(points, dtype=np.float64),
            target_frame_count=target_frame_count,
            owner=f"region plan {key}",
        )
    return result


def _planned_positions(
    timeline: Mapping[str, Any],
    actor_records: Sequence[Mapping[str, Any]],
    *,
    target_frame_count: int,
) -> dict[str, np.ndarray]:
    frames = timeline.get("frames")
    render = timeline.get("render")
    if not isinstance(frames, list) or not isinstance(render, Mapping):
        raise MP3DRegionActorTrackError("planned timeline frames/render are missing")
    declared_count = render.get("frame_count")
    if declared_count != len(frames):
        raise MP3DRegionActorTrackError(
            "planned timeline render.frame_count differs from frames"
        )
    by_slot: dict[str, list[np.ndarray]] = {
        str(record["source_slot_id"]): [] for record in actor_records
    }
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or frame.get("frame_index") != index:
            raise MP3DRegionActorTrackError(
                "planned timeline frame indices must be contiguous"
            )
        states = frame.get("actor_states")
        if not isinstance(states, list):
            raise MP3DRegionActorTrackError(
                f"planned timeline frame {index} has no actor states"
            )
        state_by_slot = {
            state.get("source_slot_id"): state
            for state in states
            if isinstance(state, Mapping)
        }
        if set(state_by_slot) != set(by_slot):
            raise MP3DRegionActorTrackError(
                f"planned timeline frame {index} actor slots differ"
            )
        for slot in by_slot:
            state = state_by_slot[slot]
            if "planned_route_center_m" not in state:
                raise MP3DRegionActorTrackError(
                    "planned timeline must carry planned_route_center_m, not an "
                    "unlabelled source/emitter position"
                )
            by_slot[slot].append(
                _finite_vector(
                    state["planned_route_center_m"],
                    owner=f"planned frame {index} {slot} route center",
                )
            )
    result: dict[str, np.ndarray] = {}
    for slot, values in by_slot.items():
        points = np.ascontiguousarray(np.stack(values), dtype=np.float64)
        result[slot] = _resample_route_points(
            points,
            target_frame_count=target_frame_count,
            owner=f"planned {slot} route center",
        )
    return result


def _m1_source_order(
    m1_request: Mapping[str, Any], endpoint_ids: Sequence[str], first_positions: Mapping[str, np.ndarray]
) -> None:
    sources = m1_request.get("sources")
    if not isinstance(sources, list) or [item.get("source_id") for item in sources] != list(
        endpoint_ids
    ):
        raise MP3DRegionActorTrackError(
            "M1 source order must equal explicit actor endpoint order"
        )
    for source, endpoint_id in zip(sources, endpoint_ids, strict=True):
        transform = source.get("world_from_source") if isinstance(source, Mapping) else None
        if not isinstance(transform, Mapping):
            raise MP3DRegionActorTrackError(
                f"M1 source {endpoint_id!r} has no world transform"
            )
        declared = _finite_vector(
            transform.get("translation_m"), owner=f"M1 source {endpoint_id} position"
        )
        if not np.allclose(declared, first_positions[endpoint_id], rtol=0.0, atol=1.0e-8):
            raise MP3DRegionActorTrackError(
                f"M1 source {endpoint_id!r} does not start at the planned route center"
            )


def _base_rotation(inputs: ValidatedM2Inputs) -> list[float]:
    states = inputs.request.get("states")
    if not isinstance(states, list) or not states:
        raise MP3DRegionActorTrackError("base M2 request has no calibration states")
    transform = states[0].get("root_transform")
    if not isinstance(transform, Mapping):
        raise MP3DRegionActorTrackError("base M2 request state has no root transform")
    try:
        return [float(value) for value in normalized_quaternion_xyzw(transform.get("rotation_xyzw"))]
    except (TypeError, ValueError) as exc:
        raise MP3DRegionActorTrackError("base M2 root rotation is invalid") from exc


def _tangent_quaternion(points: np.ndarray, index: int, fallback: Sequence[float]) -> tuple[list[float], str]:
    if len(points) < 2:
        return list(fallback), "base_m2_request_static_rotation"
    if index == 0:
        tangent = points[1] - points[0]
    elif index == len(points) - 1:
        tangent = points[-1] - points[-2]
    else:
        tangent = points[index + 1] - points[index - 1]
    horizontal = np.asarray([float(tangent[0]), float(tangent[2])], dtype=np.float64)
    if float(np.linalg.norm(horizontal)) <= _EPSILON_M:
        return list(fallback), "base_m2_request_rotation_for_zero_tangent"
    # M2 Beagle packages declare local forward=-Z. Rotate -Z onto the XZ route tangent.
    yaw = math.atan2(-float(horizontal[0]), -float(horizontal[1]))
    return [0.0, math.sin(yaw * 0.5), 0.0, math.cos(yaw * 0.5)], "route_tangent_from_asset_forward_minus_z"


def _package_for_actor(
    actor: Mapping[str, Any],
    *,
    cache: dict[tuple[Path, Path], tuple[ValidatedM2Inputs, RuntimeAssetBundle]],
) -> tuple[ValidatedM2Inputs, RuntimeAssetBundle]:
    asset_path = Path(actor["asset_manifest_path"]).expanduser().resolve()
    request_path = Path(actor["base_m2_request_path"]).expanduser().resolve()
    key = (asset_path, request_path)
    if key in cache:
        return cache[key]
    try:
        inputs = load_m2_inputs(asset_path, request_path)
        bundle = load_runtime_asset_bundle(inputs)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise MP3DRegionActorTrackError(
            f"actor package/request is not a usable current M2 Habitat package: {exc}"
        ) from exc
    if (
        inputs.asset.get("asset_id") != actor.get("asset_id")
        or actor.get("asset_revision") is None
    ):
        raise MP3DRegionActorTrackError(
            f"actor {actor['actor_id']!r} package identity differs from its explicit mapping"
        )
    coordinate = inputs.asset.get("coordinate_system")
    if not isinstance(coordinate, Mapping) or coordinate != {
        "forward_axis": "-Z",
        "handedness": "right",
        "linear_unit": "meter",
        "quaternion_order": "xyzw",
        "up_axis": "+Y",
    }:
        raise MP3DRegionActorTrackError(
            "actor package coordinate system must be the current Habitat -Z/+Y meter frame"
        )
    cache[key] = (inputs, bundle)
    return inputs, bundle


def _anchor_for_actor(
    actor: Mapping[str, Any], inputs: ValidatedM2Inputs
) -> Mapping[str, Any]:
    anchors = inputs.asset.get("anchors")
    if not isinstance(anchors, list):
        raise MP3DRegionActorTrackError("M2 asset has no anchors")
    matches = [
        item
        for item in anchors
        if isinstance(item, Mapping) and item.get("anchor_id") == actor["emitter_anchor_id"]
    ]
    if len(matches) != 1:
        raise MP3DRegionActorTrackError(
            f"actor {actor['actor_id']!r} emitter anchor {actor['emitter_anchor_id']!r} "
            f"is not unique in the M2 package"
        )
    return matches[0]


def _track_for_actor(
    actor: Mapping[str, Any],
    *,
    timeline: Mapping[str, Any],
    positions_by_slot: Mapping[str, np.ndarray],
    clock: Mapping[str, int | float],
    inputs: ValidatedM2Inputs,
    bundle: RuntimeAssetBundle,
) -> dict[str, Any]:
    slot = str(actor["source_slot_id"])
    points = positions_by_slot[slot]
    offset = _finite_vector(
        actor["route_to_actor_root_offset_m"],
        owner=f"actor {actor['actor_id']} route_to_actor_root_offset_m",
    )
    root_points = points + offset[None, :]
    moving = bool(np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > _EPSILON_M))
    action_id = "walk" if moving else "idle"
    try:
        clip = bundle.action_sets_by_role[
            bundle.action_roles_by_id[action_id]
        ].action(action_id)
    except (KeyError, ValueError) as exc:
        raise MP3DRegionActorTrackError(
            f"M2 package lacks baked {action_id} action for {actor['actor_id']!r}"
        ) from exc
    base_rotation = _base_rotation(inputs)
    anchor = _anchor_for_actor(actor, inputs)
    anchor_joint_id = anchor.get("joint_id")
    joint_order = tuple(bundle.joint_mapping["runtime_joint_order"])
    if not isinstance(anchor_joint_id, str) or anchor_joint_id not in joint_order:
        raise MP3DRegionActorTrackError(
            f"actor {actor['actor_id']!r} emitter anchor joint is absent from the "
            "M2 runtime joint order"
        )
    actor_from_skin_root = np.asarray(bundle.actor_from_skin_root, dtype=np.float64)
    if actor_from_skin_root.shape != (4, 4) or not np.all(np.isfinite(actor_from_skin_root)):
        raise MP3DRegionActorTrackError("M2 actor_from_skin_root mapping is invalid")
    frames: list[dict[str, Any]] = []
    tick_step = int(clock["ticks_per_frame"])
    for index, root_position in enumerate(root_points):
        action_time_ticks = index * tick_step
        effective_tick = action_time_ticks % int(clip.loop_duration_ticks)
        try:
            sample_index = clip.sample_ticks.index(effective_tick)
        except ValueError as exc:
            raise MP3DRegionActorTrackError(
                f"clock tick {effective_tick} for {actor['actor_id']!r} is not on the "
                "package baked action sample grid; use a compatible frame rate/tick step"
            ) from exc
        rotations = clip.rotations_xyzw[sample_index]
        if len(rotations) != len(joint_order):
            raise MP3DRegionActorTrackError(
                f"M2 {action_id} joint target count differs from runtime order"
            )
        rotation, rotation_source = (
            _tangent_quaternion(points, index, base_rotation)
            if moving
            else (list(base_rotation), "base_m2_request_static_rotation")
        )
        world_from_actor = np.eye(4, dtype=np.float64)
        world_from_actor[:3, :3] = np.asarray(
            _quaternion_to_matrix(rotation), dtype=np.float64
        )
        world_from_actor[:3, 3] = root_position
        world_from_skin_root = world_from_actor @ actor_from_skin_root
        frames.append(
            {
                "frame_index": index,
                "pts_ticks": index * tick_step,
                "action_id": action_id,
                "action_time_ticks": action_time_ticks,
                "effective_action_tick": effective_tick,
                "action_sample_index": sample_index,
                "planned_route_center_m": points[index].tolist(),
                "planned_world_from_actor": _transform_record(world_from_actor),
                "planned_world_from_skin_root": _transform_record(world_from_skin_root),
                "root_rotation_source": rotation_source,
                "joint_targets": [
                    {
                        "joint_id": joint_id,
                        "rotation_xyzw": [float(component) for component in quaternion],
                    }
                    for joint_id, quaternion in zip(joint_order, rotations, strict=True)
                ],
                "native_pending": {
                    "emitter_world_position_m": None,
                    "support_contact": None,
                    "articulated_collision": None,
                    "object_id": None,
                },
            }
        )
    return {
        "schema": ACTOR_TRACK_SCHEMA,
        "artifact_role": "planned_habitat_actor_apply_track",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "native_observed": False,
        "actor_id": actor["actor_id"],
        "source_slot_id": slot,
        "source_endpoint_id": actor["source_endpoint_id"],
        "semantic_id": actor["semantic_id"],
        "asset": {
            "asset_id": inputs.asset["asset_id"],
            "revision": actor.get("asset_revision"),
            "asset_manifest_path": str(Path(actor["asset_manifest_path"]).expanduser().resolve()),
            "base_m2_request_path": str(Path(actor["base_m2_request_path"]).expanduser().resolve()),
            "base_m2_request_id": inputs.request.get("request_id"),
            "package_admission_state": inputs.asset.get("admission_state"),
            "runtime_joint_order": list(joint_order),
            "runtime_roles": {
                role: str(path)
                for role, path in sorted(bundle.paths_by_role.items())
            },
            "actions": {
                action_id: {
                    "role": bundle.action_roles_by_id[action_id],
                    "source_action_name": bundle.action_sets_by_role[
                        bundle.action_roles_by_id[action_id]
                    ].action(action_id).source_action_name,
                    "sample_count": bundle.action_sets_by_role[
                        bundle.action_roles_by_id[action_id]
                    ].action(action_id).sample_count,
                    "loop_duration_ticks": bundle.action_sets_by_role[
                        bundle.action_roles_by_id[action_id]
                    ].action(action_id).loop_duration_ticks,
                }
                for action_id in sorted(bundle.action_roles_by_id)
            },
        },
        "emitter": {
            "anchor_id": actor["emitter_anchor_id"],
            "joint_id": anchor.get("joint_id"),
            "position_authority": "pending_native_emitter_link_readback",
            "planned_route_center_is_not_emitter_position": True,
        },
        "clock": dict(clock),
        "route_source_center_plan": {
            "authority": "planned MP3D region route center",
            "region_instance_id": timeline["region"]["region_instance_id"],
            "route_family_id": timeline["route_family_id"],
            "motion_case": timeline["motion_case"],
            "source_id": actor["source_endpoint_id"],
            "positions_m": points.tolist(),
            "route_to_actor_root_offset_m": offset.tolist(),
            "position_semantics": "PathFinder route center; no emitter offset/readback",
        },
        "native_pending": {
            "emitter_world_position_m": None,
            "support_contact": None,
            "articulated_collision": None,
            "object_id": None,
            "native_execution": None,
            "rlr": None,
        },
        "frames": frames,
    }


def _quaternion_to_matrix(value: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalized_quaternion_xyzw(value)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> list[float]:
    """Convert an orthonormal rotation matrix without dropping skin-root rotation."""

    rotation = np.asarray(matrix[:3, :3], dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            ],
            dtype=np.float64,
        )
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quaternion = np.asarray(
            [
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quaternion = np.asarray(
            [
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    quaternion = normalized_quaternion_xyzw(quaternion)
    if quaternion[3] < 0.0 or (
        quaternion[3] == 0.0
        and next((float(value) for value in quaternion[:3] if value != 0.0), 0.0) < 0.0
    ):
        quaternion = -quaternion
    return [float(value) for value in quaternion]


def _transform_record(matrix: np.ndarray) -> dict[str, list[float]]:
    return {
        "translation_m": [float(value) for value in matrix[:3, 3]],
        "rotation_xyzw": _matrix_to_quaternion_xyzw(matrix),
    }


def materialize_region_actor_tracks(
    *,
    region_plan_path: str | Path,
    planned_timeline_path: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    actor_config: Sequence[Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    output_directory: str | Path,
    frame_count: int | None = None,
    frame_rate_hz: int | float | None = None,
    time_base_hz: int | None = None,
    ticks_per_frame: int | None = None,
) -> dict[str, Any]:
    """Materialize explicit current M2 actor packages for one planned case."""

    plan, timeline, room_inputs, _family = _load_planned_inputs(
        region_plan_path=region_plan_path,
        planned_timeline_path=planned_timeline_path,
        room_manifest_path=room_manifest_path,
        m1_request_path=m1_request_path,
    )
    _selection_document, actors, actor_config_path = _load_actor_specs(actor_config)
    timeline_actor_records = _timeline_actor_records(timeline, actors)
    endpoint_ids = tuple(str(actor["source_endpoint_id"]) for actor in actors)
    clock = _resolve_clock(
        timeline,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        time_base_hz=time_base_hz,
        ticks_per_frame=ticks_per_frame,
    )
    timeline_frame_count = int(timeline["render"]["frame_count"])
    timeline_positions = _planned_positions(
        timeline,
        timeline_actor_records,
        target_frame_count=timeline_frame_count,
    )
    plan_positions_at_timeline_clock = _route_positions_from_plan(
        _family,
        motion_case=str(timeline["motion_case"]),
        actors=actors,
        target_frame_count=timeline_frame_count,
    )
    for slot in timeline_positions:
        if not np.allclose(
            timeline_positions[slot],
            plan_positions_at_timeline_clock[slot],
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise MP3DRegionActorTrackError(
                f"planned timeline {slot} route differs from the region plan"
            )
    positions_by_slot = _route_positions_from_plan(
        _family,
        motion_case=str(timeline["motion_case"]),
        actors=actors,
        target_frame_count=int(clock["frame_count"]),
    )
    positions_by_endpoint = {
        endpoint_id: positions_by_slot[str(actor["source_slot_id"])]
        for actor, endpoint_id in zip(actors, endpoint_ids, strict=True)
    }
    _m1_source_order(room_inputs.request, endpoint_ids, {
        endpoint_id: positions[0]
        for endpoint_id, positions in positions_by_endpoint.items()
    })
    package_cache: dict[
        tuple[Path, Path], tuple[ValidatedM2Inputs, RuntimeAssetBundle]
    ] = {}
    tracks: list[dict[str, Any]] = []
    for actor in actors:
        inputs, bundle = _package_for_actor(actor, cache=package_cache)
        tracks.append(
            _track_for_actor(
                actor,
                timeline=timeline,
                positions_by_slot=positions_by_slot,
                clock=clock,
                inputs=inputs,
                bundle=bundle,
            )
        )

    output = _fresh_output(output_directory)
    _write_json(output / "m1_capture_request.json", room_inputs.request)
    if actor_config_path is not None:
        _write_json(output / "actor_config.json", _selection_document)
    track_records: list[dict[str, Any]] = []
    for track in tracks:
        track_path = Path("tracks") / f"{track['source_slot_id']}.json"
        _write_json(output / track_path, track)
        track_records.append(
            {
                "actor_id": track["actor_id"],
                "source_slot_id": track["source_slot_id"],
                "source_endpoint_id": track["source_endpoint_id"],
                "track_path": track_path.as_posix(),
            }
        )
    case = {
        "schema": CASE_SCHEMA,
        "artifact_role": "planned_habitat_actor_apply_case",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "native_observed": False,
        "region": {
            "house_id": plan["house_id"],
            "region_index": timeline["region"]["region_index"],
            "region_instance_id": timeline["region"]["region_instance_id"],
        },
        "route_family_id": timeline["route_family_id"],
        "motion_case": timeline["motion_case"],
        "clock": dict(clock),
        "m1_request_path": str(Path(m1_request_path).expanduser().resolve()),
        "planned_timeline_path": str(Path(planned_timeline_path).expanduser().resolve()),
        "actor_tracks": track_records,
        "native_pending": {
            "capture": None,
            "emitter_readback": None,
            "support_contact": None,
            "collision": None,
            "object_id": None,
            "rlr": None,
        },
        "audio_consumption": (
            "requires observed native emitter trajectories; planned route centers "
            "must not be passed to current dynamic audio"
        ),
    }
    _write_json(output / "case_manifest.json", case)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "artifact_role": "planned_actor_tracks_not_native_capture",
        "status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "native_observed": False,
        "claim_boundary": (
            "CPU M2 package/action validation and apply-target planning only; "
            "no native Habitat execution, emitter readback, collision, pixels, "
            "or RLR audio"
        ),
        "inputs": {
            "region_plan": str(Path(region_plan_path).expanduser().resolve()),
            "planned_timeline": str(Path(planned_timeline_path).expanduser().resolve()),
            "room_manifest": str(Path(room_manifest_path).expanduser().resolve()),
            "m1_request": str(Path(m1_request_path).expanduser().resolve()),
            "actor_config": (
                None if actor_config_path is None else str(actor_config_path)
            ),
        },
        "region_instance_id": timeline["region"]["region_instance_id"],
        "route_family_id": timeline["route_family_id"],
        "motion_case": timeline["motion_case"],
        "clock": dict(clock),
        "actors": track_records,
        "native_capture": {"status": "not_run", "observed_frame_records": None},
        "artifacts": {
            "m1_capture_request": "m1_capture_request.json",
            "case_manifest": "case_manifest.json",
            "actor_tracks": [record["track_path"] for record in track_records],
        },
        "downstream": {
            "native_capture": "must consume case_manifest and write observed outputs",
            "dynamic_audio": (
                "blocked until native emitter readback; route centers are not "
                "audio source positions"
            ),
        },
    }
    _write_json(output / "research_receipt.json", receipt)
    return receipt


__all__ = [
    "ACTOR_TRACK_SCHEMA",
    "CASE_SCHEMA",
    "MP3DRegionActorTrackError",
    "RECEIPT_SCHEMA",
    "materialize_region_actor_tracks",
]

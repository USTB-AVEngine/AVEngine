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
from avengine.assets.habitat_static_assets import (
    HabitatStaticAssetError,
    HabitatAssetBinding,
    load_habitat_asset_bindings,
)
from avengine.camera_pose import CameraPoseError
from avengine.contracts.transforms import normalized_quaternion_xyzw
from avengine.capture.neutral_readback import validate_clock
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
    allow_research_candidate: bool = False,
) -> tuple[ValidatedM2Inputs, RuntimeAssetBundle]:
    asset_path = Path(actor["asset_manifest_path"]).expanduser().resolve()
    request_path = Path(actor["base_m2_request_path"]).expanduser().resolve()
    key = (asset_path, request_path)
    if key in cache:
        return cache[key]
    try:
        inputs = load_m2_inputs(
            asset_path, request_path, allow_research_candidate=allow_research_candidate
        )
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
            "joint_from_anchor": anchor["joint_from_anchor"],
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


def _transform_matrix(value: Any, *, owner: str) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise MP3DRegionActorTrackError(f"{owner} must be a transform object")
    translation = _finite_vector(
        value.get("translation_m"), owner=f"{owner}.translation_m"
    )
    try:
        rotation = np.asarray(
            normalized_quaternion_xyzw(value.get("rotation_xyzw")), dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise MP3DRegionActorTrackError(f"{owner}.rotation_xyzw is invalid") from exc
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quaternion_to_matrix(rotation)
    matrix[:3, 3] = translation
    return matrix


def materialize_habitat_rigid_track(
    *,
    actor: Mapping[str, Any],
    m1_request: Mapping[str, Any],
    clock: Mapping[str, int | float],
    habitat_binding: Mapping[str, Any],
    floor_height_m: float | None = None,
) -> dict[str, Any]:
    """Materialize a static GLB source with a declared resting pose.

    A rigid source has no route or action samples. Its source endpoint denotes
    the registered emitter position; when world_from_object is omitted the
    object origin is reconstructed from that endpoint and the registered
    emitter offset. The resulting track still uses the current case schema so
    the Habitat capture can keep actor and object readback in one aligned
    stream.
    """

    entity_class = actor.get("entity_class", "rigid_object")
    if entity_class not in {"rigid_object", "rigid_static_object"}:
        raise MP3DRegionActorTrackError(
            "materialize_habitat_rigid_track requires a rigid_object actor"
        )
    for key in (
        "actor_id",
        "source_slot_id",
        "source_endpoint_id",
        "semantic_id",
        "asset_id",
    ):
        if key not in actor or not isinstance(actor[key], (str, int)):
            raise MP3DRegionActorTrackError(f"rigid actor {key} is required")
    if isinstance(actor["semantic_id"], bool) or not isinstance(actor["semantic_id"], int):
        raise MP3DRegionActorTrackError("rigid actor semantic_id must be an integer")
    if not isinstance(habitat_binding, Mapping):
        raise MP3DRegionActorTrackError("rigid actor Habitat binding is required")
    glb_path = habitat_binding.get("glb_path") or habitat_binding.get("glb_relative_path")
    if not isinstance(glb_path, str) or not glb_path:
        raise MP3DRegionActorTrackError("rigid actor Habitat binding has no GLB path")
    emitter = habitat_binding.get("emitter")
    if not isinstance(emitter, Mapping):
        raise MP3DRegionActorTrackError("rigid actor Habitat binding has no emitter")
    emitter_offset = _finite_vector(
        emitter.get("offset_m", emitter.get("translation_m", [0.0, 0.0, 0.0])),
        owner="rigid actor Habitat emitter offset",
    )
    resting = habitat_binding.get("resting_pose")
    if not isinstance(resting, Mapping):
        raise MP3DRegionActorTrackError("rigid actor Habitat binding has no resting_pose")
    attachment = resting.get("attachment_surface")
    if attachment != "floor":
        raise MP3DRegionActorTrackError(
            "P4 rigid materialization currently supports floor resting_pose only"
        )
    base_plane_offset = resting.get("base_plane_offset_m", 0.0)
    try:
        base_plane_offset = float(base_plane_offset)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DRegionActorTrackError(
            "rigid actor resting_pose.base_plane_offset_m is invalid"
        ) from exc
    if not np.isfinite(base_plane_offset):
        raise MP3DRegionActorTrackError(
            "rigid actor resting_pose.base_plane_offset_m is invalid"
        )
    source_records = m1_request.get("sources")
    if not isinstance(source_records, list):
        raise MP3DRegionActorTrackError("M1 request has no sources")
    source_matches = [
        item
        for item in source_records
        if isinstance(item, Mapping)
        and item.get("source_id") == actor["source_endpoint_id"]
    ]
    if len(source_matches) != 1:
        raise MP3DRegionActorTrackError(
            f"rigid actor source endpoint {actor['source_endpoint_id']!r} is not unique"
        )
    source_transform = _transform_matrix(
        source_matches[0].get("world_from_source"),
        owner=f"rigid actor {actor['actor_id']} source transform",
    )
    world_from_object_raw = actor.get("world_from_object")
    if world_from_object_raw is None:
        world_from_object = source_transform.copy()
        world_from_object[:3, 3] = source_transform[:3, 3] - (
            source_transform[:3, :3] @ emitter_offset
        )
    else:
        world_from_object = _transform_matrix(
            world_from_object_raw,
            owner=f"rigid actor {actor['actor_id']} object transform",
        )
    if floor_height_m is not None:
        try:
            floor_height = float(floor_height_m)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MP3DRegionActorTrackError("floor_height_m is invalid") from exc
        if not np.isfinite(floor_height):
            raise MP3DRegionActorTrackError("floor_height_m is invalid")
        world_from_object[1, 3] = floor_height - base_plane_offset
    transform = _transform_record(world_from_object)
    frames: list[dict[str, Any]] = []
    frame_count = clock.get("frame_count")
    ticks_per_frame = clock.get("ticks_per_frame")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 2
        or isinstance(ticks_per_frame, bool)
        or not isinstance(ticks_per_frame, int)
        or ticks_per_frame <= 0
    ):
        raise MP3DRegionActorTrackError("rigid actor clock is invalid")
    for index in range(frame_count):
        frames.append(
            {
                "frame_index": index,
                "pts_ticks": index * ticks_per_frame,
                "action_id": "static",
                "action_time_ticks": 0,
                "effective_action_tick": 0,
                "action_sample_index": 0,
                "planned_route_center_m": None,
                "planned_object_origin_m": transform["translation_m"],
                "planned_world_from_object": transform,
                "planned_world_from_skin_root": transform,
                "root_rotation_source": "habitat_rigid_binding_resting_pose",
                "joint_targets": [],
                "native_pending": {
                    "emitter_world_position_m": None,
                    "support_contact": None,
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
        "entity_class": "rigid_object",
        "actor_id": actor["actor_id"],
        "source_slot_id": actor["source_slot_id"],
        "source_endpoint_id": actor["source_endpoint_id"],
        "semantic_id": actor["semantic_id"],
        "asset": {
            "asset_id": actor["asset_id"],
            "revision": actor.get("asset_revision"),
            "entity_class": "rigid_object",
            "habitat_binding": deepcopy(dict(habitat_binding)),
        },
        "emitter": {
            "anchor_id": emitter.get("anchor_id"),
            "offset_m": emitter_offset.tolist(),
            "offset_space": emitter.get(
                "offset_space", "final_scaled_asset_root"
            ),
            "position_authority": "pending_native_object_readback",
            "planned_route_center_is_not_emitter_position": True,
        },
        "clock": dict(clock),
        "route_source_center_plan": {
            "authority": "static object binding",
            "source_id": actor["source_endpoint_id"],
            "positions_m": None,
            "position_semantics": "static object origin; emitter is read back from binding offset",
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


def _common_root_transform(value: Any, *, owner: str) -> tuple[np.ndarray, list[float]]:
    if not isinstance(value, Mapping):
        raise MP3DRegionActorTrackError(f"{owner} must be an object")
    scale_value = value.get("scale", [1.0, 1.0, 1.0])
    try:
        scale = np.asarray(scale_value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DRegionActorTrackError(f"{owner}.scale is invalid") from exc
    if (
        scale.shape != (3,)
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
        or not np.allclose(scale, np.ones(3), rtol=0.0, atol=1.0e-9)
    ):
        raise MP3DRegionActorTrackError(
            f"{owner}.scale must be the unit scale for the current Habitat materializer"
        )
    return _transform_matrix(value, owner=owner), [float(item) for item in scale]


def _common_camera(
    value: Any,
    *,
    owner: str,
) -> tuple[dict[str, Any], np.ndarray]:
    if not isinstance(value, Mapping):
        raise MP3DRegionActorTrackError(f"{owner} must be an object")
    position = _finite_vector(value.get("position_m"), owner=f"{owner}.position_m")
    basis = value.get("basis")
    if not isinstance(basis, Mapping):
        raise MP3DRegionActorTrackError(f"{owner}.basis is required")
    forward = _finite_vector(basis.get("forward"), owner=f"{owner}.basis.forward")
    right = _finite_vector(basis.get("right"), owner=f"{owner}.basis.right")
    up = _finite_vector(basis.get("up"), owner=f"{owner}.basis.up")
    axes = np.column_stack((right, up, -forward))
    if (
        not np.allclose(axes.T @ axes, np.eye(3), rtol=0.0, atol=1.0e-5)
        or not np.isclose(np.linalg.det(axes), 1.0, rtol=0.0, atol=1.0e-5)
    ):
        raise MP3DRegionActorTrackError(
            f"{owner}.basis must be orthonormal and right handed"
        )
    fov = value.get("horizontal_fov_deg")
    if (
        isinstance(fov, bool)
        or not isinstance(fov, (int, float))
        or not np.isfinite(float(fov))
        or not 0.0 < float(fov) < 180.0
    ):
        raise MP3DRegionActorTrackError(
            f"{owner}.horizontal_fov_deg must be between 0 and 180"
        )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = axes
    matrix[:3, 3] = position
    normalized = {
        "position_m": position.tolist(),
        "basis": {
            "forward": forward.tolist(),
            "right": right.tolist(),
            "up": up.tolist(),
        },
        "horizontal_fov_deg": float(fov),
    }
    return normalized, matrix


def _common_camera_records(
    visual_plan: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    declared = visual_plan.get("camera")
    if not isinstance(declared, Mapping):
        raise MP3DRegionActorTrackError("common visual_plan.camera is required")
    first, first_matrix = _common_camera(declared, owner="visual_plan.camera")
    for index, frame in enumerate(frames):
        state = frame.get("camera_state")
        current, current_matrix = _common_camera(
            state, owner=f"visual_plan.frames[{index}].camera_state"
        )
        if (
            not np.allclose(
                current_matrix, first_matrix, rtol=0.0, atol=1.0e-8
            )
            or current["horizontal_fov_deg"] != first["horizontal_fov_deg"]
        ):
            raise MP3DRegionActorTrackError(
                "common Habitat materialization requires one static camera"
            )
    return first, first_matrix


def _common_actor_slots(
    actors: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], tuple[str, ...]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    slots: list[str] = []
    for index, actor in enumerate(actors):
        if not isinstance(actor, Mapping):
            raise MP3DRegionActorTrackError(f"visual_plan.actors[{index}] must be an object")
        actor_id = actor.get("actor_id")
        if not isinstance(actor_id, str) or not actor_id:
            raise MP3DRegionActorTrackError(
                f"visual_plan.actors[{index}].actor_id is required"
            )
        if actor_id in by_id:
            raise MP3DRegionActorTrackError(f"common plan repeats actor_id {actor_id!r}")
        slot = actor.get("source_slot_id")
        if not isinstance(slot, str) or not _SLOT_RE.fullmatch(slot):
            slot = actor_id if isinstance(actor_id, str) else ""
        if not isinstance(slot, str) or not _SLOT_RE.fullmatch(slot):
            slot = f"source{index + 1}"
        if slot in slots:
            raise MP3DRegionActorTrackError(f"common plan repeats source slot {slot!r}")
        by_id[actor_id] = actor
        slots.append(slot)
    expected = tuple(f"source{index}" for index in range(1, len(slots) + 1))
    if tuple(slots) != expected:
        raise MP3DRegionActorTrackError(
            "common plan source slots must be contiguous source1..sourceN"
        )
    return by_id, tuple(slots)


def _common_actor_slot(actor: Mapping[str, Any], *, index: int) -> str:
    slot = actor.get("source_slot_id")
    actor_id = actor.get("actor_id")
    if isinstance(slot, str) and _SLOT_RE.fullmatch(slot):
        return slot
    if isinstance(actor_id, str) and _SLOT_RE.fullmatch(actor_id):
        return actor_id
    return f"source{index + 1}"


def _common_endpoint(actor: Mapping[str, Any], *, slot: str) -> str:
    emitter = actor.get("emitter_binding")
    candidates = (
        actor.get("source_endpoint_id"),
        emitter.get("source_endpoint_id")
        if isinstance(emitter, Mapping)
        else None,
        emitter.get("endpoint_id") if isinstance(emitter, Mapping) else None,
    )
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return f"{slot}_emitter"


def _common_emitter(
    actor: Mapping[str, Any],
    *,
    binding: HabitatAssetBinding,
    owner: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Map a common semantic emitter to the package-native M2 anchor."""

    declared = actor.get("emitter_binding")
    if not isinstance(declared, Mapping):
        raise MP3DRegionActorTrackError(f"{owner}.emitter_binding is required")
    semantic_anchor = declared.get("semantic_anchor_id")
    if semantic_anchor is None:
        semantic_anchor = declared.get("anchor_id")
    if not isinstance(semantic_anchor, str) or not semantic_anchor:
        raise MP3DRegionActorTrackError(f"{owner}.emitter_binding has no semantic anchor id")

    bound_emitter = binding.emitter
    bound_semantic = bound_emitter.get("semantic_anchor_id")
    if bound_semantic is None:
        bound_semantic = bound_emitter.get("anchor_id")
    if isinstance(bound_semantic, str) and bound_semantic != semantic_anchor:
        raise MP3DRegionActorTrackError(
            f"{owner}.emitter_binding semantic anchor differs from Habitat binding"
        )
    native_anchor = bound_emitter.get("anchor_id")
    if not isinstance(native_anchor, str) or not native_anchor:
        raise MP3DRegionActorTrackError(
            f"{owner} Habitat binding has no native M2 anchor id"
        )
    declared_native = declared.get("native_anchor_id")
    if declared_native is None and declared.get("semantic_anchor_id") is None:
        declared_native = declared.get("anchor_id")
    if declared_native is not None and declared_native != native_anchor:
        raise MP3DRegionActorTrackError(
            f"{owner}.emitter_binding native anchor differs from Habitat binding"
        )

    native_offset = bound_emitter.get("offset_m")
    if native_offset is None:
        native_offset = [0.0, 0.0, 0.0]
    native_offset_array = _finite_vector(
        native_offset, owner=f"{owner}.emitter_binding.native_offset_m"
    )
    root_offset = bound_emitter.get("root_offset_m")
    if root_offset is None:
        root_offset = declared.get("emitter_offset_m")
    if root_offset is None:
        root_offset = declared.get("offset_m")
    if root_offset is None:
        root_offset = native_offset
    root_offset_array = _finite_vector(
        root_offset, owner=f"{owner}.emitter_binding.root_offset_m"
    )

    normalized = deepcopy(dict(declared))
    normalized["semantic_anchor_id"] = semantic_anchor
    normalized["native_anchor_id"] = native_anchor
    normalized["emitter_offset_m"] = root_offset_array.tolist()
    normalized["offset_space"] = "final_scaled_asset_root"
    normalized["native_offset_m"] = native_offset_array.tolist()
    normalized["native_offset_space"] = str(
        bound_emitter.get("offset_space", "joint_local")
    )
    if isinstance(bound_emitter.get("joint_from_anchor"), Mapping):
        normalized["native_joint_from_anchor"] = deepcopy(
            dict(bound_emitter["joint_from_anchor"])
        )
    return normalized, root_offset_array


def _common_actor_state(
    frame: Mapping[str, Any],
    *,
    actor_id: str,
    frame_index: int,
) -> Mapping[str, Any]:
    states = frame.get("actor_states")
    if not isinstance(states, list):
        raise MP3DRegionActorTrackError(
            f"common frame {frame_index} has no actor_states"
        )
    matches = [
        state
        for state in states
        if isinstance(state, Mapping) and state.get("actor_id") == actor_id
    ]
    if len(matches) != 1:
        raise MP3DRegionActorTrackError(
            f"common frame {frame_index} must have one state for {actor_id!r}"
        )
    return matches[0]


def _common_action_sample(
    bundle: RuntimeAssetBundle,
    state: Mapping[str, Any],
    *,
    actor_id: str,
    frame_index: int,
) -> tuple[str, int, int, np.ndarray, float, bool]:
    action_id = state.get("action_id")
    if not isinstance(action_id, str) or action_id not in bundle.action_roles_by_id:
        raise MP3DRegionActorTrackError(
            f"common actor {actor_id!r} frame {frame_index} has no validated action"
        )
    action_time_ticks = state.get("action_time_ticks")
    if (
        isinstance(action_time_ticks, bool)
        or not isinstance(action_time_ticks, int)
        or action_time_ticks < 0
    ):
        raise MP3DRegionActorTrackError(
            f"common actor {actor_id!r} frame {frame_index} action_time_ticks is invalid"
        )
    phase = state.get("action_phase")
    if (
        isinstance(phase, bool)
        or not isinstance(phase, (int, float))
        or not np.isfinite(float(phase))
        or not 0.0 <= float(phase) <= 1.0
    ):
        raise MP3DRegionActorTrackError(
            f"common actor {actor_id!r} frame {frame_index} action_phase is invalid"
        )
    moving = state.get("moving")
    if not isinstance(moving, bool):
        raise MP3DRegionActorTrackError(
            f"common actor {actor_id!r} frame {frame_index} moving is invalid"
        )
    role = bundle.action_roles_by_id[action_id]
    clip = bundle.action_sets_by_role[role].action(action_id)
    effective_tick = int(action_time_ticks) % int(clip.loop_duration_ticks)
    if not math.isclose(float(phase), effective_tick / int(clip.loop_duration_ticks), abs_tol=1.0e-8):
        raise MP3DRegionActorTrackError(
            f"common actor {actor_id!r} frame {frame_index} normalized phase disagrees with its action tick")
    try:
        sample_index = clip.sample_ticks.index(effective_tick)
    except ValueError as exc:
        raise MP3DRegionActorTrackError(
            f"common actor {actor_id!r} frame {frame_index} action tick "
            "is not on the baked action sample grid"
        ) from exc
    rotations = np.asarray(clip.rotations_xyzw[sample_index], dtype=np.float64)
    if rotations.ndim != 2 or rotations.shape[1] != 4:
        raise MP3DRegionActorTrackError(
            f"common actor {actor_id!r} frame {frame_index} action pose is invalid"
        )
    return (
        action_id,
        int(action_time_ticks),
        int(sample_index),
        rotations,
        float(phase),
        moving,
    )


def _common_articulated_track(
    *,
    actor: Mapping[str, Any],
    binding: HabitatAssetBinding,
    frames: Sequence[Mapping[str, Any]],
    clock: Mapping[str, int | float],
    slot: str,
    endpoint_id: str,
    semantic_id: int,
    allow_research_candidate: bool = False,
) -> dict[str, Any]:
    if binding.asset_manifest_path is None or binding.base_m2_request_path is None:
        raise MP3DRegionActorTrackError(
            f"actor {actor.get('actor_id')!r} Habitat binding lacks M2 package/request paths"
        )
    anchor_binding, _offset = _common_emitter(
        actor,
        binding=binding,
        owner=f"actor {actor.get('actor_id')!r}",
    )
    semantic_anchor_id = str(anchor_binding["semantic_anchor_id"])
    native_anchor_id = str(anchor_binding["native_anchor_id"])
    actor_spec = {
        "actor_id": actor["actor_id"],
        "asset_id": actor["asset_id"],
        "asset_revision": actor.get("asset_revision") or binding.revision,
        "asset_manifest_path": str(binding.asset_manifest_path),
        "base_m2_request_path": str(binding.base_m2_request_path),
        "emitter_anchor_id": native_anchor_id,
    }
    if not isinstance(actor_spec["asset_revision"], str):
        raise MP3DRegionActorTrackError(
            f"actor {actor.get('actor_id')!r} has no asset revision"
        )
    inputs, bundle = _package_for_actor(
        actor_spec, cache={}, allow_research_candidate=allow_research_candidate
    )
    anchor = _anchor_for_actor(actor_spec, inputs)
    bound_joint_id = binding.emitter.get("joint_id")
    if (
        isinstance(bound_joint_id, str)
        and bound_joint_id
        and bound_joint_id != anchor.get("joint_id")
    ):
        raise MP3DRegionActorTrackError(
            f"actor {actor.get('actor_id')!r} native emitter joint differs from M2 anchor"
        )
    joint_order = tuple(bundle.joint_mapping["runtime_joint_order"])
    actor_from_skin_root = np.asarray(bundle.actor_from_skin_root, dtype=np.float64)
    if (
        actor_from_skin_root.shape != (4, 4)
        or not np.all(np.isfinite(actor_from_skin_root))
    ):
        raise MP3DRegionActorTrackError(
            f"actor {actor.get('actor_id')!r} actor_from_skin_root mapping is invalid"
        )
    output_frames: list[dict[str, Any]] = []
    roots: list[list[float]] = []
    for frame_index, frame in enumerate(frames):
        state = _common_actor_state(
            frame,
            actor_id=str(actor["actor_id"]),
            frame_index=frame_index,
        )
        root, scale = _common_root_transform(
            state.get("root_transform"),
            owner=f"actor {actor['actor_id']!r} frame {frame_index}.root_transform",
        )
        (
            action_id,
            action_time_ticks,
            sample_index,
            rotations,
            phase,
            moving,
        ) = _common_action_sample(
            bundle,
            state,
            actor_id=str(actor["actor_id"]),
            frame_index=frame_index,
        )
        if rotations.shape != (len(joint_order), 4):
            raise MP3DRegionActorTrackError(
                f"common actor {actor['actor_id']!r} frame {frame_index} "
                "joint target count differs from runtime order"
            )
        transform = _transform_record(root)
        skin_root = root @ actor_from_skin_root
        if not np.all(np.isfinite(skin_root)):
            raise MP3DRegionActorTrackError(
                f"common actor {actor['actor_id']!r} frame {frame_index} "
                "skin-root transform is not finite"
            )
        skin_root_transform = _transform_record(skin_root)
        roots.append(transform["translation_m"])
        output_frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * int(clock["ticks_per_frame"]),
                "action_id": action_id,
                "action_time_ticks": action_time_ticks,
                "effective_action_tick": action_time_ticks
                % int(
                    bundle.action_sets_by_role[
                        bundle.action_roles_by_id[action_id]
                    ].action(action_id).loop_duration_ticks
                ),
                "action_sample_index": sample_index,
                "action_phase": phase,
                "moving": moving,
                "planned_route_center_m": transform["translation_m"],
                "planned_world_from_actor": transform,
                "planned_world_from_skin_root": skin_root_transform,
                "planned_scale": scale,
                "root_rotation_source": "common_plan_root_transform",
                "joint_targets": [
                    {
                        "joint_id": joint_id,
                        "rotation_xyzw": [
                            float(component) for component in quaternion
                        ],
                    }
                    for joint_id, quaternion in zip(
                        joint_order, rotations, strict=True
                    )
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
        "entity_class": str(actor.get("entity_class", binding.entity_class)),
        "actor_id": actor["actor_id"],
        "source_slot_id": slot,
        "source_endpoint_id": endpoint_id,
        "semantic_id": semantic_id,
        "asset": {
            "asset_id": inputs.asset["asset_id"],
            "revision": actor.get("asset_revision") or binding.revision,
            "asset_manifest_path": str(inputs.asset_path),
            "base_m2_request_path": str(inputs.request_path),
            "base_m2_request_id": inputs.request.get("request_id"),
            "package_admission_state": inputs.asset.get("admission_state"),
            "runtime_joint_order": list(joint_order),
            "actor_from_skin_root": actor_from_skin_root.tolist(),
            "runtime_roles": {
                role: str(path)
                for role, path in sorted(bundle.paths_by_role.items())
            },
            "habitat_binding": binding.to_dict(),
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
            "anchor_id": native_anchor_id,
            "semantic_anchor_id": semantic_anchor_id,
            "native_anchor_id": native_anchor_id,
            "joint_id": anchor.get("joint_id"),
            "joint_from_anchor": anchor["joint_from_anchor"],
            "offset_m": anchor_binding["emitter_offset_m"],
            "offset_space": anchor_binding.get(
                "offset_space", "final_scaled_asset_root"
            ),
            "native_offset_m": anchor_binding["native_offset_m"],
            "native_offset_space": anchor_binding["native_offset_space"],
            "position_authority": "pending_native_emitter_link_readback",
            "planned_route_center_is_not_emitter_position": True,
        },
        "clock": dict(clock),
        "route_source_center_plan": {
            "authority": "common visual plan root transform",
            "source_id": endpoint_id,
            "positions_m": roots,
            "position_semantics": "common plan actor-root positions; no route replanning or resampling",
            "root_transform_semantics": "planned_world_from_actor is authoritative actor root; skin root is actor root composed with actor_from_skin_root",
            "actor_from_skin_root": actor_from_skin_root.tolist(),
        },
        "native_pending": {
            "emitter_world_position_m": None,
            "support_contact": None,
            "articulated_collision": None,
            "object_id": None,
            "native_execution": None,
            "rlr": None,
        },
        "frames": output_frames,
    }


def _common_rigid_track(
    *,
    actor: Mapping[str, Any],
    binding: HabitatAssetBinding,
    frames: Sequence[Mapping[str, Any]],
    clock: Mapping[str, int | float],
    slot: str,
    endpoint_id: str,
    semantic_id: int,
) -> dict[str, Any]:
    emitter_binding, _offset = _common_emitter(
        actor,
        binding=binding,
        owner=f"actor {actor.get('actor_id')!r}",
    )
    output_frames: list[dict[str, Any]] = []
    positions: list[list[float]] = []
    for frame_index, frame in enumerate(frames):
        state = _common_actor_state(
            frame,
            actor_id=str(actor["actor_id"]),
            frame_index=frame_index,
        )
        root, scale = _common_root_transform(
            state.get("root_transform"),
            owner=f"actor {actor['actor_id']!r} frame {frame_index}.root_transform",
        )
        moving = state.get("moving")
        if moving is not False:
            raise MP3DRegionActorTrackError(
                f"rigid actor {actor['actor_id']!r} must declare moving=false"
            )
        action_id = state.get("action_id", "static")
        if action_id != "static":
            raise MP3DRegionActorTrackError(
                f"rigid actor {actor['actor_id']!r} must use action_id='static'"
            )
        action_time_ticks = state.get("action_time_ticks", 0)
        if (
            isinstance(action_time_ticks, bool)
            or not isinstance(action_time_ticks, int)
            or action_time_ticks < 0
        ):
            raise MP3DRegionActorTrackError(
                f"rigid actor {actor['actor_id']!r} action_time_ticks is invalid"
            )
        phase = state.get("action_phase", 0.0)
        if (
            isinstance(phase, bool)
            or not isinstance(phase, (int, float))
            or not np.isfinite(float(phase))
            or not 0.0 <= float(phase) <= 1.0
        ):
            raise MP3DRegionActorTrackError(
                f"rigid actor {actor['actor_id']!r} action_phase is invalid"
            )
        transform = _transform_record(root)
        if output_frames:
            first_root = _transform_matrix(output_frames[0]["planned_world_from_object"], owner="static first root")
            if not np.allclose(root, first_root, rtol=0.0, atol=1.0e-9):
                raise MP3DRegionActorTrackError(f"rigid actor {actor['actor_id']!r} root moves despite its static declaration")
        positions.append(transform["translation_m"])
        output_frames.append(
            {
                "frame_index": frame_index,
                "pts_ticks": frame_index * int(clock["ticks_per_frame"]),
                "action_id": "static",
                "action_time_ticks": int(action_time_ticks),
                "effective_action_tick": 0,
                "action_sample_index": 0,
                "action_phase": float(phase),
                "moving": False,
                "planned_route_center_m": None,
                "planned_object_origin_m": transform["translation_m"],
                "planned_world_from_object": transform,
                "planned_world_from_skin_root": transform,
                "planned_scale": scale,
                "root_rotation_source": "common_plan_root_transform",
                "joint_targets": [],
                "native_pending": {
                    "emitter_world_position_m": None,
                    "support_contact": None,
                    "articulated_collision": None,
                    "object_id": None,
                },
            }
        )
    habitat_binding = binding.to_dict()
    return {
        "schema": ACTOR_TRACK_SCHEMA,
        "artifact_role": "planned_habitat_actor_apply_track",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "native_observed": False,
        "entity_class": "rigid_object",
        "actor_id": actor["actor_id"],
        "source_slot_id": slot,
        "source_endpoint_id": endpoint_id,
        "semantic_id": semantic_id,
        "asset": {
            "asset_id": actor["asset_id"],
            "revision": actor.get("asset_revision") or binding.revision,
            "entity_class": "rigid_object",
            "habitat_binding": habitat_binding,
        },
        "emitter": {
            "anchor_id": emitter_binding["semantic_anchor_id"],
            "offset_m": emitter_binding["emitter_offset_m"],
            "offset_space": emitter_binding.get(
                "offset_space", "final_scaled_asset_root"
            ),
            "position_authority": "pending_native_object_readback",
            "planned_route_center_is_not_emitter_position": True,
        },
        "clock": dict(clock),
        "route_source_center_plan": {
            "authority": "common visual plan object transform",
            "source_id": endpoint_id,
            "positions_m": positions,
            "position_semantics": "common plan object origins; no route replanning",
        },
        "native_pending": {
            "emitter_world_position_m": None,
            "support_contact": None,
            "articulated_collision": None,
            "object_id": None,
            "native_execution": None,
            "rlr": None,
        },
        "frames": output_frames,
    }


def _common_plan_m1_request(
    *,
    room: Mapping[str, Any],
    camera: Mapping[str, Any],
    camera_matrix: np.ndarray,
    sources: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    base_m1_request: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if base_m1_request is None:
        request: dict[str, Any] = {
            "schema": "avengine_m1_capture_request_v1",
            "request_id": str(
                plan.get("episode_id") or plan.get("plan_id") or "common_plan_habitat"
            ),
            "room_id": str(room["room_id"]),
            "seed": int(plan.get("seed", 0)) % (2 ** 31),
            "primary_camera_rig": {
                "rig_id": "camera_rig_0",
                "view_id": "view0",
                "shared_calibration": {
                    "projection": "pinhole",
                    "resolution_hw": [240, 320],
                    "hfov_degrees": float(camera["horizontal_fov_deg"]),
                    "near_m": 0.05,
                    "far_m": 100.0,
                    "rig_from_sensor": {
                        "translation_m": [0.0, 0.0, 0.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                "modalities": [
                    {"modality": "rgb", "sensor_uuid": "rig_rgb"},
                    {"modality": "depth", "sensor_uuid": "rig_depth"},
                    {"modality": "semantic", "sensor_uuid": "rig_semantic"},
                ],
            },
            "listener": {
                "listener_id": "listener0",
                "attached_to": "camera_rig_0",
                "rig_from_listener": {
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "qa_views": [],
        }
        navigation = plan.get("room_capabilities", {}).get("evidence_refs", {}).get("navigation", {})
        primary = plan.get("visual_plan", {}).get("camera", {})
        floor = navigation.get("floor_height_m")
        if primary.get("height_above_floor_m") is not None:
            floor = float(primary["position_m"][1]) - float(primary["height_above_floor_m"])
        if floor is None or not np.isfinite(float(floor)):
            raise MP3DRegionActorTrackError("common plan without a base M1 request needs its measured navigation floor")
        request["qa_views"] = [{"qa_id": "navmesh_topdown", "kind": "topdown",
                               "meters_per_pixel": float(navigation.get("resolution_m", 0.05)),
                               "height_m": float(floor)}]
    else:
        request = deepcopy(dict(base_m1_request))
    rig = request.get("primary_camera_rig")
    if not isinstance(rig, Mapping):
        raise MP3DRegionActorTrackError("base M1 request has no primary_camera_rig")
    rig = deepcopy(dict(rig))
    calibration = rig.get("shared_calibration")
    if not isinstance(calibration, Mapping):
        raise MP3DRegionActorTrackError(
            "base M1 request has no shared_camera calibration"
        )
    calibration = deepcopy(dict(calibration))
    # The neutral pose names the co-located optical camera and listener.
    # Existing M1 requires identity sensor extrinsics; never copy an old pose here.
    sensor = {"translation_m": [0., 0., 0.], "rotation_xyzw": [0., 0., 0., 1.]}
    calibration["rig_from_sensor"] = deepcopy(sensor)
    rig["world_from_rig"] = _transform_record(camera_matrix)
    if isinstance(request.get("listener"), Mapping):
        request["listener"] = {**deepcopy(request["listener"]), "rig_from_listener": deepcopy(sensor)}
    calibration["hfov_degrees"] = float(camera["horizontal_fov_deg"])
    resolution = plan.get("visual_plan", {}).get("camera", {}).get("resolution_hw")
    if resolution is not None:
        if not isinstance(resolution, list) or len(resolution) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in resolution):
            raise MP3DRegionActorTrackError("common camera resolution must be positive integer [height,width]")
        calibration["resolution_hw"] = list(resolution)
    rig["shared_calibration"] = calibration
    request["primary_camera_rig"] = rig
    request["room_id"] = str(room["room_id"])
    if plan.get("episode_id") or plan.get("plan_id"):
        request["request_id"] = str(plan.get("episode_id") or plan.get("plan_id"))
    if isinstance(plan.get("seed"), int) and not isinstance(plan.get("seed"), bool):
        request["seed"] = int(plan["seed"]) % (2 ** 31)
    request["sources"] = deepcopy(list(sources))
    return request


def _common_source_transform(
    root: np.ndarray,
    emitter_offset: np.ndarray,
) -> dict[str, Any]:
    position = root[:3, 3] + root[:3, :3] @ emitter_offset
    return {
        "translation_m": [float(value) for value in position],
        "rotation_xyzw": _matrix_to_quaternion_xyzw(root),
    }


def materialize_common_plan_habitat(
    *,
    plan: Mapping[str, Any],
    room_manifest: str | Path,
    runtime_registry: str | Path,
    output: str | Path,
    habitat_binding_delta: str | Path | None = None,
    base_m1_request: str | Path | Mapping[str, Any] | None = None,
    allow_research_candidate: bool = False,
) -> dict[str, Any]:
    """Materialize one renderer-neutral plan without changing its trajectories."""

    if not isinstance(plan, Mapping):
        raise MP3DRegionActorTrackError("common plan must be an object")
    clock_raw = plan.get("clock")
    try:
        clock = validate_clock(clock_raw)
    except (TypeError, ValueError) as exc:
        raise MP3DRegionActorTrackError(f"common plan clock is invalid: {exc}") from exc
    visual = plan.get("visual_plan")
    if not isinstance(visual, Mapping):
        raise MP3DRegionActorTrackError("common plan has no visual_plan")
    actors_raw = visual.get("actors")
    frames_raw = visual.get("frames")
    if (
        not isinstance(actors_raw, list)
        or len(actors_raw) < 2
        or not isinstance(frames_raw, list)
        or len(frames_raw) != int(clock["frame_count"])
    ):
        raise MP3DRegionActorTrackError(
            "common visual plan must contain at least two actors and one frame per clock frame"
        )
    actors_by_id, slots = _common_actor_slots(actors_raw)
    actor_ids = tuple(actor["actor_id"] for actor in actors_raw)
    for index, frame in enumerate(frames_raw):
        if not isinstance(frame, Mapping):
            raise MP3DRegionActorTrackError(
                f"common visual_plan.frames[{index}] must be an object"
            )
        if frame.get("frame_index", index) != index:
            raise MP3DRegionActorTrackError(
                f"common visual_plan frame indices are not contiguous at {index}"
            )
        states = frame.get("actor_states")
        if not isinstance(states, list) or len(states) != len(actor_ids):
            raise MP3DRegionActorTrackError(
                f"common frame {index} actor state count differs from actors"
            )
        if [state.get("actor_id") for state in states if isinstance(state, Mapping)] != list(
            actor_ids
        ):
            raise MP3DRegionActorTrackError(
                f"common frame {index} actor order differs from visual_plan.actors"
            )
        pts_ticks = frame.get("pts_ticks", index * int(clock["ticks_per_frame"]))
        if pts_ticks != index * int(clock["ticks_per_frame"]):
            raise MP3DRegionActorTrackError(
                f"common frame {index} pts_ticks differs from clock"
            )
    try:
        room = _read_json(room_manifest, owner="common plan room manifest")
    except (OSError, TypeError, ValueError) as exc:
        raise MP3DRegionActorTrackError(f"common plan room manifest is invalid: {exc}") from exc
    room_id = room.get("room_id")
    if not isinstance(room_id, str) or not room_id:
        raise MP3DRegionActorTrackError("room manifest has no room_id")
    camera, camera_matrix = _common_camera_records(
        visual,
        [frame for frame in frames_raw if isinstance(frame, Mapping)],
    )
    asset_ids = [str(actor.get("asset_id")) for actor in actors_raw]
    if any(not value for value in asset_ids):
        raise MP3DRegionActorTrackError("common actors must carry asset_id")
    try:
        bindings = load_habitat_asset_bindings(
            tuple(dict.fromkeys(asset_ids)),
            runtime_registry_path=runtime_registry,
            binding_delta_path=habitat_binding_delta,
        )
    except (HabitatStaticAssetError, TypeError, ValueError) as exc:
        raise MP3DRegionActorTrackError(
            f"common plan Habitat bindings are invalid: {exc}"
        ) from exc
    semantic_ids: list[int] = []
    actor_records: list[dict[str, Any]] = []
    tracks: list[dict[str, Any]] = []
    roots_by_actor: dict[str, list[np.ndarray]] = {}
    for index, actor in enumerate(actors_raw):
        actor_id = str(actor["actor_id"])
        asset_id = str(actor["asset_id"])
        binding = bindings[asset_id]
        declared_class = actor.get("entity_class", binding.entity_class)
        if not isinstance(declared_class, str) or not declared_class:
            raise MP3DRegionActorTrackError(
                f"common actor {actor_id!r} has no entity_class"
            )
        if (
            declared_class in {"rigid_object", "rigid_static_object"}
            and binding.normalized_entity_class != "rigid_object"
        ) or (
            declared_class not in {"rigid_object", "rigid_static_object"}
            and binding.normalized_entity_class == "rigid_object"
        ):
            raise MP3DRegionActorTrackError(
                f"common actor {actor_id!r} entity_class differs from Habitat binding"
            )
        revision = actor.get("asset_revision")
        if revision is not None and revision != binding.revision:
            raise MP3DRegionActorTrackError(
                f"common actor {actor_id!r} asset_revision differs from Habitat binding"
            )
        semantic_id = actor.get("semantic_id", 210 + index)
        if (
            isinstance(semantic_id, bool)
            or not isinstance(semantic_id, int)
            or semantic_id <= 0
            or semantic_id in semantic_ids
        ):
            raise MP3DRegionActorTrackError(
                f"common actor {actor_id!r} semantic_id must be unique and positive"
            )
        semantic_ids.append(int(semantic_id))
        slot = _common_actor_slot(actor, index=index)
        endpoint_id = _common_endpoint(actor, slot=slot)
        actor_copy = deepcopy(dict(actor))
        actor_copy.setdefault("asset_revision", binding.revision)
        actor_copy["entity_class"] = declared_class
        if binding.normalized_entity_class == "rigid_object":
            track = _common_rigid_track(
                actor=actor_copy,
                binding=binding,
                frames=frames_raw,
                clock=clock,
                slot=slot,
                endpoint_id=endpoint_id,
                semantic_id=int(semantic_id),
            )
        else:
            track = _common_articulated_track(
                actor=actor_copy,
                binding=binding,
                frames=frames_raw,
                clock=clock,
                slot=slot,
                endpoint_id=endpoint_id,
                semantic_id=int(semantic_id),
                allow_research_candidate=allow_research_candidate,
            )
        tracks.append(track)
        roots_by_actor[actor_id] = [
            _common_root_transform(
                _common_actor_state(
                    frame,
                    actor_id=actor_id,
                    frame_index=frame_index,
                ).get("root_transform"),
                owner=f"actor {actor_id!r} frame {frame_index}.root_transform",
            )[0]
            for frame_index, frame in enumerate(frames_raw)
        ]
        actor_records.append(
            {
                "actor_id": actor_id,
                "source_slot_id": slot,
                "source_endpoint_id": endpoint_id,
                "asset_id": asset_id,
                "asset_revision": actor_copy.get("asset_revision"),
                "semantic_id": int(semantic_id),
                "entity_class": declared_class,
            }
        )
    source_records: list[dict[str, Any]] = []
    for actor, track in zip(actors_raw, tracks, strict=True):
        actor_id = str(actor["actor_id"])
        offset = np.asarray(track["emitter"]["offset_m"], dtype=np.float64)
        source_records.append(
            {
                "source_id": track["source_endpoint_id"],
                "world_from_source": _common_source_transform(
                    roots_by_actor[actor_id][0], offset
                ),
            }
        )
    base_request: Mapping[str, Any] | None
    if base_m1_request is None:
        base_request = None
    elif isinstance(base_m1_request, Mapping):
        base_request = base_m1_request
    else:
        try:
            base_request = _read_json(base_m1_request, owner="base M1 request")
        except (OSError, TypeError, ValueError) as exc:
            raise MP3DRegionActorTrackError(f"base M1 request is invalid: {exc}") from exc
    request = _common_plan_m1_request(
        room=room,
        camera=camera,
        camera_matrix=camera_matrix,
        sources=source_records,
        plan=plan,
        base_m1_request=base_request,
    )
    output_path = _fresh_output(output)
    _write_json(output_path / "common_plan.json", deepcopy(dict(plan)))
    _write_json(output_path / "m1_capture_request.json", request)
    track_records: list[dict[str, Any]] = []
    for track in tracks:
        track_path = Path("tracks") / f"{track['source_slot_id']}.json"
        _write_json(output_path / track_path, track)
        track_records.append(
            {
                "actor_id": track["actor_id"],
                "source_slot_id": track["source_slot_id"],
                "source_endpoint_id": track["source_endpoint_id"],
                "semantic_id": track["semantic_id"],
                "entity_class": track["entity_class"],
                "track_path": track_path.as_posix(),
            }
        )
    house_id = room_id.rsplit("_", 1)[-1]
    region = visual.get("region")
    if not isinstance(region, Mapping):
        region = plan.get("region")
    if not isinstance(region, Mapping):
        region = {}
    region_index = region.get("region_index", 0)
    if isinstance(region_index, bool) or not isinstance(region_index, int) or region_index < 0:
        raise MP3DRegionActorTrackError("common plan region_index is invalid")
    region_instance_id = region.get("region_instance_id")
    if not isinstance(region_instance_id, str) or not region_instance_id:
        region_instance_id = f"{house_id}:region:{region_index:03d}"
    case = {
        "schema": CASE_SCHEMA,
        "artifact_role": "planned_habitat_actor_apply_case",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "native_observed": False,
        "region": {
            "house_id": house_id,
            "region_index": region_index,
            "region_instance_id": region_instance_id,
        },
        "route_family_id": str(plan.get("route_family_id", "common_plan")),
        "motion_case": str(plan.get("motion_case", "common_plan")),
        "clock": dict(clock),
        "m1_request_path": str((output_path / "m1_capture_request.json").resolve()),
        "common_plan_path": str((output_path / "common_plan.json").resolve()),
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
            "requires observed native emitter trajectories; common-plan routes "
            "were not replanned"
        ),
    }
    _write_json(output_path / "case_manifest.json", case)
    try:
        from avengine.capture.mp3d_multi_actor import (
            _load_case_and_m1,
            _resolve_case_track_paths,
        )

        validated_room_inputs = load_m1_inputs(
            room_manifest,
            output_path / "m1_capture_request.json",
        )
        _resolved_case, _resolved_tracks = _resolve_case_track_paths(
            output_path / "case_manifest.json",
            case,
        )
        _load_case_and_m1(
            case_manifest_path=output_path / "case_manifest.json",
            room_manifest_path=validated_room_inputs.room_path,
            m1_request_path=validated_room_inputs.request_path,
        )
    except (ContractError, OSError, TypeError, ValueError) as exc:
        raise MP3DRegionActorTrackError(
            f"common plan output failed capture input validation: {exc}"
        ) from exc
    checks: list[dict[str, Any]] = []
    for actor, track in zip(actors_raw, tracks, strict=True):
        actor_id = str(actor["actor_id"])
        state_roots = roots_by_actor[actor_id]
        track_roots = [
            _transform_matrix(
                frame.get("planned_world_from_actor")
                or frame.get("planned_world_from_object")
                or frame.get("planned_world_from_skin_root"),
                owner=f"materialized {track['source_slot_id']} frame {index}",
            )
            for index, frame in enumerate(track["frames"])
        ]
        max_error = max(
            float(np.max(np.abs(left - right)))
            for left, right in zip(state_roots, track_roots, strict=True)
        )
        checks.append(
            {
                "actor_id": actor_id,
                "source_slot_id": track["source_slot_id"],
                "root_max_error": max_error,
                "root_preserved": max_error <= 1.0e-9,
            }
        )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "artifact_role": "planned_common_habitat_materialization",
        "simulator_seed": {"input_seed": int(plan.get("seed", 0)),
                           "habitat_seed": int(plan.get("seed", 0)) % (2 ** 31),
                           "mapping": "modulo_2_to_31_for_habitat_signed_int"},
        "status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "claim_boundary": (
            "Common renderer-neutral plan materialized into M1/case/tracks; "
            "no native capture or audio was run by this function."
        ),
        "clock": dict(clock),
        "camera": camera,
        "actors": actor_records,
        "checks": {
            "route_replanned": False,
            "frame_count_preserved": True,
            "root_checks": checks,
            "camera_static_validated": True,
            "capture_input_validation": "pass",
        },
        "inputs": {
            "room_manifest": str(Path(room_manifest).expanduser().resolve()),
            "runtime_registry": str(Path(runtime_registry).expanduser().resolve()),
            "habitat_binding_delta": (
                None
                if habitat_binding_delta is None
                else str(Path(habitat_binding_delta).expanduser().resolve())
            ),
            "base_m1_request": (
                None
                if base_m1_request is None or isinstance(base_m1_request, Mapping)
                else str(Path(base_m1_request).expanduser().resolve())
            ),
        },
        "artifacts": {
            "common_plan": "common_plan.json",
            "m1_capture_request": "m1_capture_request.json",
            "case_manifest": "case_manifest.json",
            "actor_tracks": [item["track_path"] for item in track_records],
        },
    }
    _write_json(output_path / "research_receipt.json", receipt)
    return receipt
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
    "materialize_common_plan_habitat",
    "materialize_habitat_rigid_track",
    "materialize_region_actor_tracks",
]

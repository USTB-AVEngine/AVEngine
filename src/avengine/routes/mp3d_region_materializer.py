"""Materialize one MP3D region route case for current AVEngine inputs.

The region planner intentionally stops at CPU route geometry.  This module
bridges one selected case to a current, room-backed M1 request and explicitly
planned capture/audio inputs without starting Habitat, RLR, or a native
capture.  Actor assets
and source endpoints are explicit inputs; no historical human/beagle mapping is
silently substituted.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.camera_pose import CameraPoseError, apply_camera_listener_pose
from avengine.registry.sources import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
)
from avengine.rooms.contracts import (
    ContractError,
    load_and_validate_inputs as load_m1_inputs,
    validate_capture_request,
)
from avengine.rooms.mp3d_regions import MP3DHouseFloorPlan, parse_mp3d_house
from avengine.routes.trajectory import (
    M6XTrajectoryError,
    resample_polyline_by_arc_length,
)
from avengine.timeline.audio_program import validate_audio_program
from avengine.timeline.current_mp3d_dynamic_audio import (
    CurrentMP3DDynamicAudioError,
    _program_clock_binding,
    _resolve_visual_clock,
)


MATERIALIZATION_ARTIFACT_KIND = "mp3d_region_case_materialization"
MATERIALIZATION_SCHEMA = "avengine_mp3d_region_case_materialization_v1"
DEFAULT_TIME_BASE_HZ = 48_000
SOURCE_SLOT_PATTERN = re.compile(r"source([1-9][0-9]*)\Z")


class MP3DRegionMaterializationError(ValueError):
    """A selected region case cannot become a current AVEngine input."""


def _json_object(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MP3DRegionMaterializationError(f"{owner} must be a JSON object")
    return deepcopy(dict(value))


def _read_json(path: str | Path, *, owner: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise MP3DRegionMaterializationError(
            f"{owner} must be a regular file: {resolved}"
        )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MP3DRegionMaterializationError(
            f"cannot read {owner}: {resolved}: {exc}"
        ) from exc
    return _json_object(value, owner=owner)


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool):
        raise MP3DRegionMaterializationError(f"{owner} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DRegionMaterializationError(f"{owner} must be finite") from exc
    if not math.isfinite(result):
        raise MP3DRegionMaterializationError(f"{owner} must be finite")
    return result


def _positive_int(value: Any, *, owner: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MP3DRegionMaterializationError(
            f"{owner} must be an integer >= {minimum}"
        )
    return int(value)


def _point(value: Any, *, owner: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise MP3DRegionMaterializationError(f"{owner} must be a finite 3-vector")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DRegionMaterializationError(
            f"{owner} must be a finite 3-vector"
        ) from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise MP3DRegionMaterializationError(f"{owner} must be a finite 3-vector")
    return np.ascontiguousarray(result)


def _source_slot_index(value: Any, *, owner: str) -> int:
    if not isinstance(value, str):
        raise MP3DRegionMaterializationError(f"{owner} must be sourceN")
    match = SOURCE_SLOT_PATTERN.fullmatch(value)
    if match is None:
        raise MP3DRegionMaterializationError(f"{owner} must be sourceN")
    return int(match.group(1))


def _fresh_output(path: str | Path) -> Path:
    output = Path(path).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise MP3DRegionMaterializationError(
            f"refusing to replace materialization output: {output}"
        )
    output.mkdir(parents=True)
    return output


def _load_region_plan(
    value: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, Mapping):
        plan = _json_object(value, owner="region plan")
        source_path = None
    else:
        source_path = Path(value).expanduser().resolve()
        plan = _read_json(source_path, owner="region plan")
    if plan.get("artifact_kind") != "mp3d_region_source_route_plan":
        raise MP3DRegionMaterializationError(
            "region plan artifact_kind must be 'mp3d_region_source_route_plan'"
        )
    if plan.get("research_only") is not True or plan.get("episode_counted") is not False:
        raise MP3DRegionMaterializationError(
            "region plan must remain research_only and episode_counted=false"
        )
    house_id = plan.get("house_id")
    if not isinstance(house_id, str) or not house_id:
        raise MP3DRegionMaterializationError("region plan house_id is required")
    regions = plan.get("regions")
    if not isinstance(regions, list) or not regions:
        raise MP3DRegionMaterializationError("region plan must contain regions")
    return plan, source_path


def _resolve_plan_path(
    raw: Any,
    *,
    owner: str,
    plan_path: Path | None,
) -> Path:
    if not isinstance(raw, str) or not raw:
        raise MP3DRegionMaterializationError(f"region plan input {owner} is required")
    path = Path(raw)
    if not path.is_absolute() and plan_path is not None:
        path = plan_path.parent / path
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise MP3DRegionMaterializationError(f"region plan {owner} is missing: {path}")
    return path


def _select_region_case(
    plan: Mapping[str, Any],
    *,
    region_index: int,
    route_family_id: str | None,
    motion_case: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    regions = [
        item
        for item in plan["regions"]
        if isinstance(item, Mapping) and item.get("region_index") == region_index
    ]
    if len(regions) != 1:
        raise MP3DRegionMaterializationError(
            f"region_index {region_index} does not identify one planned region"
        )
    region = regions[0]
    families = region.get("route_families")
    if not isinstance(families, list) or not families:
        raise MP3DRegionMaterializationError(
            f"region {region_index} has no route families"
        )
    if route_family_id is None:
        if len(families) != 1:
            raise MP3DRegionMaterializationError(
                "route_family_id is required when a region has multiple families"
            )
        family = families[0]
    else:
        matches = [
            item
            for item in families
            if isinstance(item, Mapping)
            and item.get("route_family_id") == route_family_id
        ]
        if len(matches) != 1:
            raise MP3DRegionMaterializationError(
                f"route_family_id {route_family_id!r} is not unique in region {region_index}"
            )
        family = matches[0]
    cases = family.get("cases")
    if not isinstance(cases, Mapping):
        raise MP3DRegionMaterializationError("selected route family has no cases")
    case = cases.get(motion_case)
    if not isinstance(case, Mapping):
        raise MP3DRegionMaterializationError(
            f"motion_case {motion_case!r} is absent from selected route family"
        )
    if case.get("route_family_id") != family.get("route_family_id"):
        raise MP3DRegionMaterializationError(
            "selected case route_family_id differs from its route family"
        )
    if family.get("region_index") != region_index:
        raise MP3DRegionMaterializationError(
            "selected route family region_index differs from its region"
        )
    return region, family, case


def _load_authoritative_region(
    plan: Mapping[str, Any],
    *,
    plan_path: Path | None,
    region: Mapping[str, Any],
) -> tuple[MP3DHouseFloorPlan, Any]:
    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping):
        raise MP3DRegionMaterializationError(
            "region plan must retain its inputs.house path"
        )
    house_path = _resolve_plan_path(
        inputs.get("house"), owner="inputs.house", plan_path=plan_path
    )
    try:
        house = parse_mp3d_house(house_path)
    except (OSError, ValueError) as exc:
        raise MP3DRegionMaterializationError(
            f"cannot parse authoritative MP3D house: {exc}"
        ) from exc
    if house.house_id != plan.get("house_id"):
        raise MP3DRegionMaterializationError(
            f"parsed house_id {house.house_id!r} differs from plan {plan.get('house_id')!r}"
        )
    region_index = region.get("region_index")
    actual = house.by_region_index.get(region_index)
    if actual is None:
        raise MP3DRegionMaterializationError(
            f"authoritative house lacks region_index {region_index}"
        )
    if actual.region_instance_id != region.get("region_instance_id"):
        raise MP3DRegionMaterializationError(
            "planned region_instance_id differs from the authoritative .house region"
        )
    return house, actual


def _load_actor_selection(path: str | Path) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    selection = _read_json(path, owner="actor selection")
    if selection.get("research_only") is False:
        raise MP3DRegionMaterializationError(
            "actor selection may not claim formal status"
        )
    actors = selection.get("actors")
    if not isinstance(actors, list) or len(actors) < 2:
        raise MP3DRegionMaterializationError(
            "actor selection must contain at least two actors"
        )
    indexed: list[tuple[int, Mapping[str, Any]]] = []
    for index, actor in enumerate(actors):
        if not isinstance(actor, Mapping):
            raise MP3DRegionMaterializationError(
                f"actor selection actors[{index}] must be an object"
            )
        slot_index = _source_slot_index(
            actor.get("source_slot_id"), owner=f"actors[{index}].source_slot_id"
        )
        if not isinstance(actor.get("asset_id"), str) or not actor["asset_id"]:
            raise MP3DRegionMaterializationError(
                f"actors[{index}].asset_id is required"
            )
        if not isinstance(actor.get("revision"), str) or not actor["revision"]:
            raise MP3DRegionMaterializationError(
                f"actors[{index}].revision is required"
            )
        identity = actor.get("entity_instance_id") or actor.get(
            "legacy_timeline_actor_id"
        )
        explicit_endpoint = actor.get("source_endpoint_id")
        if identity is None and not isinstance(explicit_endpoint, str):
            raise MP3DRegionMaterializationError(
                f"actors[{index}] needs entity_instance_id, legacy_timeline_actor_id, "
                "or an explicit source_endpoint_id"
            )
        indexed.append((slot_index, actor))
    indexed.sort(key=lambda item: item[0])
    expected = tuple(range(1, len(indexed) + 1))
    actual = tuple(item[0] for item in indexed)
    if actual != expected:
        raise MP3DRegionMaterializationError(
            "actor selection slots must be the contiguous source1..sourceN sequence"
        )
    return selection, tuple(item[1] for item in indexed)


def _resolve_endpoints(
    actors: Sequence[Mapping[str, Any]],
    endpoint_registry: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    records = endpoint_registry.get("source_endpoints")
    if not isinstance(records, list):
        raise MP3DRegionMaterializationError(
            "source endpoint registry must contain source_endpoints"
        )
    resolved_ids: list[str] = []
    resolved_records: list[Mapping[str, Any]] = []
    for index, actor in enumerate(actors):
        explicit = actor.get("source_endpoint_id")
        if explicit is not None:
            if not isinstance(explicit, str) or not explicit:
                raise MP3DRegionMaterializationError(
                    f"actor {index} source_endpoint_id must be a nonempty string"
                )
            matches = [
                item
                for item in records
                if isinstance(item, Mapping)
                and item.get("source_endpoint_id") == explicit
            ]
        else:
            instance = actor.get("entity_instance_id") or actor.get(
                "legacy_timeline_actor_id"
            )
            matches = []
            for item in records:
                if not isinstance(item, Mapping):
                    continue
                binding = item.get("binding")
                if not isinstance(binding, Mapping):
                    continue
                if (
                    binding.get("kind") == "entity_anchor"
                    and binding.get("entity_instance_id") == instance
                    and binding.get("entity_asset_id") == actor.get("asset_id")
                    and binding.get("entity_asset_revision") == actor.get("revision")
                ):
                    matches.append(item)
        if len(matches) != 1:
            identity = (
                explicit
                or actor.get("entity_instance_id")
                or actor.get("legacy_timeline_actor_id")
            )
            raise MP3DRegionMaterializationError(
                f"actor slot {actor.get('source_slot_id')} has {len(matches)} "
                f"source endpoint matches for {identity!r}; provide an explicit "
                "asset-to-endpoint mapping"
            )
        endpoint_id = matches[0].get("source_endpoint_id")
        if not isinstance(endpoint_id, str) or not endpoint_id:
            raise MP3DRegionMaterializationError(
                f"source endpoint for actor slot {actor.get('source_slot_id')} has no ID"
            )
        if endpoint_id in resolved_ids:
            raise MP3DRegionMaterializationError(
                f"source endpoint {endpoint_id!r} is assigned to multiple actors"
            )
        resolved_ids.append(endpoint_id)
        resolved_records.append(matches[0])
    return tuple(resolved_ids), tuple(resolved_records)


def _resolve_clock(
    plan: Mapping[str, Any],
    *,
    frame_count: int | None,
    frame_rate_hz: int | float | None,
    time_base_hz: int,
    ticks_per_frame: int | None,
) -> dict[str, int | float]:
    parameters = plan.get("parameters")
    if not isinstance(parameters, Mapping):
        raise MP3DRegionMaterializationError("region plan parameters are required")
    requested_frames = (
        frame_count if frame_count is not None else parameters.get("frame_count")
    )
    requested_rate = (
        frame_rate_hz
        if frame_rate_hz is not None
        else parameters.get("frame_rate_hz")
    )
    try:
        resolved = _resolve_visual_clock(
            frame_count=requested_frames,
            frame_rate_hz=requested_rate,
            ticks_per_frame=ticks_per_frame,
            time_base_hz=time_base_hz,
        )
    except (CurrentMP3DDynamicAudioError, TypeError, ValueError) as exc:
        raise MP3DRegionMaterializationError(
            "selected region clock is incompatible with the current dynamic "
            "audio clock; pass an explicit compatible frame_count/frame_rate_hz/"
            f"ticks_per_frame (details: {exc})"
        ) from exc
    if int(resolved["frame_count"]) < 2:
        raise MP3DRegionMaterializationError(
            "materialized region capture needs at least two frames"
        )
    return resolved


def _resample_positions(
    case: Mapping[str, Any],
    *,
    source_slots: Sequence[str],
    target_frame_count: int,
    region: Any,
    maximum_y_delta_m: float,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for slot in source_slots:
        key = f"{slot}_positions_m"
        raw = case.get(key)
        try:
            points = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MP3DRegionMaterializationError(
                f"selected case {key} is not numeric"
            ) from exc
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise MP3DRegionMaterializationError(
                f"selected case {key} must contain at least two [x,y,z] points"
            )
        if not np.all(np.isfinite(points)):
            raise MP3DRegionMaterializationError(
                f"selected case {key} contains non-finite points"
            )
        if any(
            not region.contains(point, y_tolerance_m=maximum_y_delta_m)
            for point in points
        ):
            raise MP3DRegionMaterializationError(
                f"selected case {key} leaves its authoritative region"
            )
        moving = bool(np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > 1.0e-9))
        if target_frame_count == len(points):
            materialized = points.copy()
        elif not moving:
            materialized = np.repeat(points[:1], target_frame_count, axis=0)
        else:
            try:
                materialized = resample_polyline_by_arc_length(
                    points,
                    target_frame_count,
                    owner=f"selected case {key}",
                )
            except M6XTrajectoryError as exc:
                raise MP3DRegionMaterializationError(str(exc)) from exc
        if any(
            not region.contains(point, y_tolerance_m=maximum_y_delta_m)
            for point in materialized
        ):
            raise MP3DRegionMaterializationError(
                f"resampled case {key} leaves its authoritative region"
            )
        result[slot] = np.ascontiguousarray(materialized, dtype=np.float64)
    for left_index, left in enumerate(source_slots):
        for right in source_slots[left_index + 1 :]:
            separation = np.linalg.norm(
                result[left][:, (0, 2)] - result[right][:, (0, 2)], axis=1
            )
            if np.any(separation <= 1.0e-9):
                raise MP3DRegionMaterializationError(
                    f"materialized source routes {left}/{right} coincide"
                )
    return result


def _camera_binding(
    family: Mapping[str, Any],
    *,
    region: Any,
    maximum_y_delta_m: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    raw = family.get("camera_binding")
    if not isinstance(raw, Mapping):
        raise MP3DRegionMaterializationError(
            "selected route family has no camera binding"
        )
    position = _point(raw.get("position_m"), owner="camera_binding.position_m")
    yaw = _finite_number(raw.get("yaw_deg"), owner="camera_binding.yaw_deg")
    floor = raw.get("floor_position_m")
    if floor is not None:
        floor_point = _point(floor, owner="camera_binding.floor_position_m")
        if not region.contains(floor_point, y_tolerance_m=maximum_y_delta_m):
            raise MP3DRegionMaterializationError(
                "camera_binding.floor_position_m is outside its authoritative region"
            )
    return position, yaw, dict(raw)


def _materialized_request(
    base_request: Mapping[str, Any],
    *,
    room_id: str,
    endpoint_ids: Sequence[str],
    source_positions: Mapping[str, np.ndarray],
    camera_position: Sequence[float],
    camera_yaw_deg: float,
    request_id: str | None,
) -> dict[str, Any]:
    generated_id = request_id or (
        f"{base_request['request_id']}_mp3d_region_case"
    )
    try:
        request = apply_camera_listener_pose(
            base_request,
            request_id=generated_id,
            position_m=camera_position,
            yaw_deg=camera_yaw_deg,
        )
    except (CameraPoseError, ContractError, KeyError, TypeError, ValueError) as exc:
        raise MP3DRegionMaterializationError(
            f"cannot apply selected region camera to the M1 request: {exc}"
        ) from exc
    request["room_id"] = room_id
    request["sources"] = [
        {
            "source_id": endpoint_id,
            "world_from_source": {
                "translation_m": source_positions[endpoint_id][0].tolist(),
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
        for endpoint_id in endpoint_ids
    ]
    errors = validate_capture_request(request, room_id=room_id)
    if errors:
        raise MP3DRegionMaterializationError(
            "materialized M1 capture request is invalid: " + "; ".join(errors)
        )
    return request


def _timeline_record(
    *,
    room_id: str,
    house_id: str,
    region: Mapping[str, Any],
    family: Mapping[str, Any],
    motion_case: str,
    actors: Sequence[Mapping[str, Any]],
    endpoint_ids: Sequence[str],
    positions_by_endpoint: Mapping[str, np.ndarray],
    camera_position: Sequence[float],
    camera_yaw_deg: float,
    clock: Mapping[str, int | float],
) -> dict[str, Any]:
    count = int(clock["frame_count"])
    actor_records = []
    for actor, endpoint_id in zip(actors, endpoint_ids, strict=True):
        actor_records.append(
            {
                "source_slot_id": actor["source_slot_id"],
                "actor_id": (
                    actor.get("entity_instance_id")
                    or actor.get("legacy_timeline_actor_id")
                    or f"{actor['source_slot_id']}_actor"
                ),
                "asset_id": actor["asset_id"],
                "revision": actor["revision"],
                "source_endpoint_id": endpoint_id,
            }
        )
    frames = []
    for index in range(count):
        states = []
        for actor_record, endpoint_id in zip(actor_records, endpoint_ids, strict=True):
            points = positions_by_endpoint[endpoint_id]
            moving = bool(
                np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > 1.0e-9)
            )
            states.append(
                {
                    **actor_record,
                    "planned_route_center_m": points[index].tolist(),
                    "position_semantics": "planner_route_center_not_emitter_readback",
                    "action_id": "walk" if moving else "idle",
                    "action_phase": (
                        float(index) / float(count - 1) if moving else 0.0
                    ),
                }
            )
        frames.append(
            {
                "frame_index": index,
                "pts_ticks": index * int(clock["ticks_per_frame"]),
                "planned_camera_pose": {
                    "position_m": list(camera_position),
                    "yaw_deg": float(camera_yaw_deg),
                },
                "actor_states": states,
            }
        )
    return {
        "schema": "avengine_mp3d_region_planned_timeline_v1",
        "kind": "mp3d_region_case_planned_timeline",
        "artifact_role": "planned_timeline_not_native_capture",
        "status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "claim_boundary": (
            "CPU region-case materialization only; native capture, visual "
            "readback, RLR audio, and formal admission remain unrun"
        ),
        "room": {"room_id": room_id, "house_id": house_id},
        "region": {
            "region_index": region["region_index"],
            "region_instance_id": region["region_instance_id"],
            "category_code": region.get("category_code"),
            "category_name": region.get("category_name"),
        },
        "route_family_id": family["route_family_id"],
        "motion_case": motion_case,
        "render": {
            "frame_count": count,
            "frame_rate_hz": clock["frame_rate_hz"],
            "ticks_per_frame": clock["ticks_per_frame"],
            "time_base_hz": clock["time_base_hz"],
            "sample_rate_hz": clock["sample_rate_hz"],
            "sample_count": clock["sample_count"],
        },
        "camera": {
            "position_m": list(camera_position),
            "yaw_deg": float(camera_yaw_deg),
            "m1_rig_id": "camera_rig_0",
            "m1_view_id": "view0",
        },
        "actors": actor_records,
        "source_endpoint_ids": list(endpoint_ids),
        "frames": frames,
    }


def _planned_frame_records(
    timeline: Mapping[str, Any],
    *,
    endpoint_ids: Sequence[str],
    positions_by_endpoint: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    render = timeline["render"]
    frames = []
    for index in range(int(render["frame_count"])):
        frames.append(
            {
                "frame_index": index,
                "pts_ticks": index * int(render["ticks_per_frame"]),
                "planned_source_positions_m": [
                    positions_by_endpoint[endpoint_id][index].tolist()
                    for endpoint_id in endpoint_ids
                ],
                "planned_camera_pose": {
                    "position_m": list(timeline["camera"]["position_m"]),
                    "yaw_deg": timeline["camera"]["yaw_deg"],
                },
            }
        )
    return {
        "artifact_role": "planned_frame_records_not_observed_capture",
        "render": dict(render),
        "frames": frames,
    }


def _validate_audio_program_input(
    path: str | Path,
    *,
    endpoint_ids: Sequence[str],
    clock: Mapping[str, int | float],
    endpoint_registry: Mapping[str, Any],
    sound_asset_registry_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    program = _read_json(path, owner="audio program")
    candidates = program.get("candidate_source_endpoint_ids")
    if candidates != list(endpoint_ids):
        raise MP3DRegionMaterializationError(
            "audio program candidate endpoints must equal the materialized "
            f"actor endpoint order {list(endpoint_ids)!r}"
        )
    try:
        _program_clock_binding(program, clock)
    except CurrentMP3DDynamicAudioError as exc:
        raise MP3DRegionMaterializationError(
            f"audio program clock does not match materialized capture clock: {exc}"
        ) from exc
    sounds = None
    if sound_asset_registry_path is not None:
        try:
            sounds = load_sound_asset_registry(sound_asset_registry_path)
        except (OSError, ValueError, TypeError) as exc:
            raise MP3DRegionMaterializationError(
                f"cannot load sound asset registry: {exc}"
            ) from exc
        errors = validate_audio_program(
            program,
            source_endpoint_registry=endpoint_registry,
            sound_asset_registry=sounds,
        )
        if errors:
            raise MP3DRegionMaterializationError(
                "audio program does not validate against supplied registries: "
                + "; ".join(errors)
            )
    return program, sounds


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize_region_case(
    region_plan: Mapping[str, Any] | str | Path,
    *,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    actor_selection_path: str | Path,
    source_endpoint_registry_path: str | Path,
    output_directory: str | Path,
    region_index: int,
    motion_case: str,
    route_family_id: str | None = None,
    frame_count: int | None = None,
    frame_rate_hz: int | float | None = None,
    time_base_hz: int = DEFAULT_TIME_BASE_HZ,
    ticks_per_frame: int | None = None,
    request_id: str | None = None,
    audio_program_path: str | Path | None = None,
    sound_asset_registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write one bounded region case as an M1 request and planned inputs.

    ``planned_frame_records.json`` is deliberately not a capture.  Native
    execution must later write observed ``frame_records.json`` before any
    current audio consumer is invoked.  ``frame_count`` and the clock fields
    describe the planned output.  When
    they differ from the planner's sparse route clock, moving routes are
    resampled along their existing polyline; static routes are held exactly.
    Actor identities and endpoint IDs are always resolved from explicit input
    files.  An audio program is optional because the region planner does not
    choose sounds; if supplied, its endpoint and clock declarations are
    checked before it is copied into the materialization directory.
    """

    if isinstance(region_index, bool) or not isinstance(region_index, int):
        raise MP3DRegionMaterializationError("region_index must be an integer")
    if not isinstance(motion_case, str) or not motion_case:
        raise MP3DRegionMaterializationError("motion_case must be nonempty")
    if isinstance(time_base_hz, bool) or not isinstance(time_base_hz, int):
        raise MP3DRegionMaterializationError("time_base_hz must be an integer")
    if time_base_hz < 1:
        raise MP3DRegionMaterializationError("time_base_hz must be positive")

    plan, plan_path = _load_region_plan(region_plan)
    region, family, case = _select_region_case(
        plan,
        region_index=region_index,
        route_family_id=route_family_id,
        motion_case=motion_case,
    )
    house_plan, authoritative_region = _load_authoritative_region(
        plan, plan_path=plan_path, region=region
    )
    try:
        room_inputs = load_m1_inputs(room_manifest_path, m1_request_path)
    except (OSError, ValueError, TypeError, ContractError) as exc:
        raise MP3DRegionMaterializationError(
            f"current M1 room/request inputs are invalid: {exc}"
        ) from exc
    room_id = room_inputs.room.get("room_id")
    scene = room_inputs.room.get("scene")
    scene_id = scene.get("scene_id") if isinstance(scene, Mapping) else None
    room_matches_house = (
        isinstance(room_id, str)
        and (
            room_id.endswith(house_plan.house_id)
            or (isinstance(scene_id, str) and house_plan.house_id in scene_id)
        )
    )
    if not room_matches_house:
        raise MP3DRegionMaterializationError(
            "M1 room does not identify the region plan's house: "
            f"room_id={room_id!r}, scene_id={scene_id!r}, "
            f"house_id={house_plan.house_id!r}"
        )
    try:
        endpoint_registry = load_source_endpoint_registry(
            source_endpoint_registry_path
        )
    except (OSError, ValueError, TypeError) as exc:
        raise MP3DRegionMaterializationError(
            f"source endpoint registry is invalid: {exc}"
        ) from exc
    selection, actors = _load_actor_selection(actor_selection_path)
    endpoint_ids, _ = _resolve_endpoints(actors, endpoint_registry)
    source_slots = tuple(str(actor["source_slot_id"]) for actor in actors)
    plan_slots = tuple(
        slot
        for slot in source_slots
        if isinstance(case.get(f"{slot}_positions_m"), list)
    )
    if plan_slots != source_slots:
        raise MP3DRegionMaterializationError(
            "selected region case does not provide one route for every actor "
            f"slot {list(source_slots)!r}; its route planner currently exposes "
            f"{list(plan_slots)!r}"
        )
    clock = _resolve_clock(
        plan,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        time_base_hz=time_base_hz,
        ticks_per_frame=ticks_per_frame,
    )
    parameters = plan.get("parameters") or {}
    maximum_y_delta = _finite_number(
        parameters.get("maximum_y_delta_m", 0.3),
        owner="parameters.maximum_y_delta_m",
    )
    if maximum_y_delta < 0.0:
        raise MP3DRegionMaterializationError(
            "parameters.maximum_y_delta_m must be nonnegative"
        )
    positions_by_slot = _resample_positions(
        case,
        source_slots=source_slots,
        target_frame_count=int(clock["frame_count"]),
        region=authoritative_region,
        maximum_y_delta_m=maximum_y_delta,
    )
    positions_by_endpoint = {
        endpoint_id: positions_by_slot[slot]
        for slot, endpoint_id in zip(source_slots, endpoint_ids, strict=True)
    }
    camera_position, camera_yaw_deg, camera_binding = _camera_binding(
        family,
        region=authoritative_region,
        maximum_y_delta_m=maximum_y_delta,
    )
    generated_request_id = request_id or (
        f"{room_inputs.request['request_id']}_region_{region_index}_{motion_case}"
    )
    request = _materialized_request(
        room_inputs.request,
        room_id=room_id,
        endpoint_ids=endpoint_ids,
        source_positions=positions_by_endpoint,
        camera_position=camera_position,
        camera_yaw_deg=camera_yaw_deg,
        request_id=generated_request_id,
    )
    timeline = _timeline_record(
        room_id=room_id,
        house_id=house_plan.house_id,
        region=region,
        family=family,
        motion_case=motion_case,
        actors=actors,
        endpoint_ids=endpoint_ids,
        positions_by_endpoint=positions_by_endpoint,
        camera_position=camera_position,
        camera_yaw_deg=camera_yaw_deg,
        clock=clock,
    )
    planned_frame_records = _planned_frame_records(
        timeline,
        endpoint_ids=endpoint_ids,
        positions_by_endpoint=positions_by_endpoint,
    )
    program = None
    sounds = None
    if audio_program_path is not None:
        program, sounds = _validate_audio_program_input(
            audio_program_path,
            endpoint_ids=endpoint_ids,
            clock=clock,
            endpoint_registry=endpoint_registry,
            sound_asset_registry_path=sound_asset_registry_path,
        )

    output = _fresh_output(output_directory)
    _write_json(output / "actor_selection.json", selection)
    _write_json(output / "source_endpoints.json", endpoint_registry)
    _write_json(output / "m1_capture_request.json", request)
    _write_json(output / "planned_timeline.json", timeline)
    _write_json(output / "planned_frame_records.json", planned_frame_records)
    if program is not None:
        _write_json(output / "audio_program.json", program)
    if sounds is not None:
        _write_json(output / "sound_assets.json", sounds)

    plan_input_path = (
        str(plan_path) if plan_path is not None else "<in-memory-region-plan>"
    )
    receipt: dict[str, Any] = {
        "schema": MATERIALIZATION_SCHEMA,
        "artifact_kind": MATERIALIZATION_ARTIFACT_KIND,
        "status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "qualification_claim": False,
        "claim_boundary": (
            "CPU region-case materialization only; no native visual capture, "
            "RLR render, or formal dataset admission"
        ),
        "inputs": {
            "region_plan": plan_input_path,
            "room_manifest": str(Path(room_manifest_path).expanduser().resolve()),
            "base_m1_request": str(Path(m1_request_path).expanduser().resolve()),
            "actor_selection": str(Path(actor_selection_path).expanduser().resolve()),
            "source_endpoint_registry": str(
                Path(source_endpoint_registry_path).expanduser().resolve()
            ),
            "audio_program": (
                None
                if audio_program_path is None
                else str(Path(audio_program_path).expanduser().resolve())
            ),
            "sound_asset_registry": (
                None
                if sound_asset_registry_path is None
                else str(Path(sound_asset_registry_path).expanduser().resolve())
            ),
        },
        "room": {"room_id": room_id, "house_id": house_plan.house_id},
        "region": {
            "region_index": region["region_index"],
            "region_instance_id": region["region_instance_id"],
            "category_code": region.get("category_code"),
            "category_name": region.get("category_name"),
        },
        "route": {
            "route_family_id": family["route_family_id"],
            "motion_case": motion_case,
            "planner_frame_count": case.get("frame_count"),
            "planner_frame_rate_hz": case.get("frame_rate_hz"),
            "resampling": "static_hold_or_arc_length_polyline_to_requested_clock",
            "position_semantics": (
                "PathFinder route center only; no actor mesh/emitter offset or "
                "runtime readback"
            ),
            "camera_binding": camera_binding,
        },
        "actors": [
            {
                "source_slot_id": actor["source_slot_id"],
                "asset_id": actor["asset_id"],
                "revision": actor["revision"],
                "source_endpoint_id": endpoint_id,
                "entity_instance_id": (
                    actor.get("entity_instance_id")
                    or actor.get("legacy_timeline_actor_id")
                ),
                "position_semantics": "planned_route_center_not_emitter_readback",
            }
            for actor, endpoint_id in zip(actors, endpoint_ids, strict=True)
        ],
        "planned_clock": {
            "frame_count": clock["frame_count"],
            "frame_rate_hz": clock["frame_rate_hz"],
            "ticks_per_frame": clock["ticks_per_frame"],
            "time_base_hz": clock["time_base_hz"],
            "sample_rate_hz": clock["sample_rate_hz"],
            "sample_count": clock["sample_count"],
            "source": "region planner output resampled for a future capture",
        },
        "native_capture": {
            "status": "not_run",
            "native_capture_started": False,
            "gpu_started": False,
        },
        "audio": {
            "status": (
                "planned_program_clock_and_endpoint_bound"
                if program is not None
                else "requires_explicit_audio_program"
            ),
            "source_endpoint_ids": list(endpoint_ids),
            "program": "audio_program.json" if program is not None else None,
            "sound_assets": "sound_assets.json" if sounds is not None else None,
        },
        "artifacts": {
            "actor_selection": "actor_selection.json",
            "source_endpoints": "source_endpoints.json",
            "m1_capture_request": "m1_capture_request.json",
            "planned_timeline": "planned_timeline.json",
            "planned_frame_records": "planned_frame_records.json",
            **({"audio_program": "audio_program.json"} if program is not None else {}),
            **({"sound_assets": "sound_assets.json"} if sounds is not None else {}),
        },
        "downstream": {
            "current_m1_parser": "m1_capture_request.json",
            "current_dynamic_audio_capture_parser": (
                "planned_frame_records.json is not a capture; native capture must "
                "write observed frame_records.json before audio consumption"
            ),
            "native_capture": "not_run; requires a room-specific actor materializer",
            "rlr_audio": (
                "program endpoint/clock binding prepared"
                if program is not None
                else "not prepared; supply audio_program_path"
            ),
        },
    }
    _write_json(output / "research_receipt.json", receipt)
    return receipt


__all__ = [
    "DEFAULT_TIME_BASE_HZ",
    "MATERIALIZATION_ARTIFACT_KIND",
    "MATERIALIZATION_SCHEMA",
    "MP3DRegionMaterializationError",
    "materialize_region_case",
]

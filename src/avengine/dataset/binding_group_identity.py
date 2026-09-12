"""Native HM3D cross-event physical-identity producer.

The group changes the physical actor that produces the second event while
keeping the camera, event locations, clips, and room fixed. Visual plans are
renderer-neutral and are materialized through the existing Habitat adapter;
all four audio members are rendered independently by RLR before PCM equality
is checked. Outputs remain research-only until human and formal admission.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence
import wave

import numpy as np

from avengine.assets.mp3d_region_actor_tracks import materialize_common_plan_habitat
from avengine.capture.qa_plan_adapters import load_planning_resources
from avengine.dataset import binding_group_native as native
from avengine.dataset.binding_group_motion import native_common_endpoint_paths
from avengine.qa.answerability import line_of_sight
from avengine.qa.binding_groups import assemble_binding_dataset
from avengine.qa.binding_questions import generate_binding_question
from avengine.routes.trajectory import resample_polyline_by_arc_length
from avengine.runtime_profiles import (
    build_asset_emitter_binding,
    load_source_asset_runtime_registry,
    resolve_source_asset_runtime_profile,
)

FAMILY = "cross_event_identity"
DEFAULT_QUERY = {"event_numbers": [1, 2]}


class IdentityNativeError(native.BindingNativeError):
    """A cross-event identity group cannot be realized from the supplied inputs."""


def _load(path: str | Path) -> dict[str, Any]:
    try:
        return native._load(Path(path).expanduser().resolve())
    except native.BindingNativeError as exc:
        raise IdentityNativeError(str(exc)) from exc


def _write(path: str | Path, value: Any) -> Path:
    try:
        return native._write(Path(path).expanduser().resolve(), value)
    except native.BindingNativeError as exc:
        raise IdentityNativeError(str(exc)) from exc


def _file(value: Any, *, base: Path, owner: str) -> Path:
    try:
        return native._file(value, base=base, owner=owner)
    except native.BindingNativeError as exc:
        raise IdentityNativeError(str(exc)) from exc


def _mapping(value: Any, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise IdentityNativeError(f"{owner} must be an object")
    return dict(value)


def _config(request: Mapping[str, Any]) -> dict[str, Any]:
    raw = request.get("binding_identity")
    if not isinstance(raw, Mapping):
        raise IdentityNativeError(
            "request must declare binding_identity timing, route and visibility settings"
        )
    raw = dict(raw)
    if (
        "minimum_silence_after_emission_s" not in raw
        and "minimum_silence_after_wet_tail_s" in raw
    ):
        # Read old diagnostic plans without losing compatibility, but expose
        # the emission-based meaning in all new planning code and receipts.
        raw["minimum_silence_after_emission_s"] = raw["minimum_silence_after_wet_tail_s"]
    required = (
        "sound_identity_field",
        "event_start_s",
        "minimum_silence_after_emission_s",
        "post_motion_silence_s",
        "walk_speed_range_mps",
        "minimum_motion_s",
        "end_hold_s",
        "minimum_entity_separation_m",
        "same_floor_tolerance_m",
        "path_length_range_m",
        "route_retry_budget",
        "visibility_margin_deg",
        "minimum_bearing_change_deg",
        "angle_tolerance_deg",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise IdentityNativeError(f"binding_identity missing configuration: {missing}")
    value = deepcopy(dict(raw))
    speeds = value["walk_speed_range_mps"]
    lengths = value["path_length_range_m"]
    if (
        not isinstance(speeds, Sequence)
        or isinstance(speeds, (str, bytes))
        or len(speeds) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in speeds)
        or not 0 < float(speeds[0]) <= float(speeds[1])
    ):
        raise IdentityNativeError("binding_identity.walk_speed_range_mps must be two ordered positive numbers")
    if (
        not isinstance(lengths, Sequence)
        or isinstance(lengths, (str, bytes))
        or len(lengths) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in lengths)
        or not 0 < float(lengths[0]) <= float(lengths[1])
    ):
        raise IdentityNativeError("binding_identity.path_length_range_m must be two ordered positive numbers")
    nonnegative = (
        "event_start_s",
        "minimum_silence_after_emission_s",
        "post_motion_silence_s",
        "minimum_motion_s",
        "end_hold_s",
        "minimum_entity_separation_m",
        "same_floor_tolerance_m",
        "visibility_margin_deg",
        "minimum_bearing_change_deg",
        "angle_tolerance_deg",
    )
    for key in nonnegative:
        number = value[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) < 0
        ):
            raise IdentityNativeError(f"binding_identity.{key} must be finite and nonnegative")
    budget = value["route_retry_budget"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise IdentityNativeError("binding_identity.route_retry_budget must be a positive integer")
    if not isinstance(value["sound_identity_field"], str) or not value["sound_identity_field"].strip():
        raise IdentityNativeError("binding_identity.sound_identity_field must be nonempty")
    target = value.get("target_position_m")
    if target is not None:
        if (
            not isinstance(target, Sequence)
            or isinstance(target, (str, bytes))
            or len(target) != 3
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in target)
        ):
            raise IdentityNativeError("binding_identity.target_position_m must be three finite numbers")
    return value


def _validate_request(request: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    clock = _mapping(plan.get("clock"), "plan.clock")
    for key in ("frame_count", "frame_rate_hz", "sample_rate_hz", "sample_count", "time_base_hz", "ticks_per_frame"):
        if key not in clock:
            raise IdentityNativeError(f"plan.clock lacks {key}")
    for key in ("frame_count", "frame_rate_hz", "sample_rate_hz"):
        if key not in request or not math.isclose(float(request[key]), float(clock[key]), abs_tol=1e-9):
            raise IdentityNativeError(f"request {key} differs from the base plan clock")
    if request.get("camera", {}).get("motion") != "static":
        raise IdentityNativeError("cross-event identity requires a static camera")
    assets = request.get("source_asset_ids")
    if (
        not isinstance(assets, Sequence)
        or isinstance(assets, (str, bytes))
        or len(assets) != 2
        or len(set(assets)) != 2
        or any(not isinstance(item, str) or not item.strip() for item in assets)
    ):
        raise IdentityNativeError("cross-event identity requires two distinct source_asset_ids")
    entities = request.get("entities")
    if not isinstance(entities, Mapping) or int(entities.get("total_count", -1)) != 2:
        raise IdentityNativeError("cross-event identity requires entities.total_count=2")
    reserve = request.get("profile", {}).get("reserve_tail_s")
    if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not math.isfinite(float(reserve)) or float(reserve) < 0:
        raise IdentityNativeError("request.profile.reserve_tail_s must be finite and nonnegative")
    if not isinstance(request.get("post_assembly_convolution_gain"), (int, float)):
        raise IdentityNativeError("request must declare post_assembly_convolution_gain")
    _config(request)


def _registry(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    path = _file(request.get("source_registry"), base=native.REPOSITORY, owner="source registry")
    try:
        registry = load_source_asset_runtime_registry(path)
    except Exception as exc:
        raise IdentityNativeError(f"source registry is invalid: {path}: {exc}") from exc
    try:
        index = {
            str(asset_id): resolve_source_asset_runtime_profile(registry, str(asset_id))
            for asset_id in request["source_asset_ids"]
        }
    except Exception as exc:
        raise IdentityNativeError(f"selected source asset is not registered: {exc}") from exc
    for asset_id, record in index.items():
        if record.get("entity_class") != "articulated_human":
            raise IdentityNativeError(f"identity first group requires articulated_human assets: {asset_id}")
    return registry, index


def _sound_path(sound: Mapping[str, Any], pool_path: Path) -> Path:
    return _file(sound.get("path") or sound.get("prepared"), base=pool_path.parent, owner="prepared sound")


def _motion_count_options(path_length_m: float, config: Mapping[str, Any], fps: float) -> list[int]:
    speed_lo, speed_hi = map(float, config["walk_speed_range_mps"])
    minimum_steps = max(
        1,
        int(math.ceil(float(path_length_m) * fps / speed_hi - 1.0e-9)),
        int(math.ceil(float(config["minimum_motion_s"]) * fps - 1.0e-9)),
    )
    maximum_steps = int(math.floor(float(path_length_m) * fps / speed_lo + 1.0e-9))
    if minimum_steps > maximum_steps:
        return []
    return list(range(minimum_steps, maximum_steps + 1))


def _sample_native_topologies(
    plan: Mapping[str, Any], request: Mapping[str, Any], *, count: int = 8,
) -> list[dict[str, Any]]:
    """Sample bounded legal PathFinder geometries once for pair filtering."""
    config = _config(request)
    if _geometry_pool_specs(request):
        report = identity_geometry_pool_report(plan, request)
        if report["status"] != "compatible":
            raise IdentityGeometryQueryRequired(
                "declared identity geometry pools serve no legal common endpoint "
                "for the planner placement during sound-pair pre-screening",
                report=report,
            )
        return _raw_native_polyline_candidates(
            plan, request,
            pool_path=str(report["selected_pool"]["resolved_path"]),
        )
    base_offset = int(config.get("route_seed_offset", 910))
    dummy_first = {"sample_count": 1, "sample_rate_hz": int(request["sample_rate_hz"])}
    dummy_second = {"sample_count": 1, "sample_rate_hz": int(request["sample_rate_hz"])}
    candidates = []
    rejection_counts: dict[str, int] = {}
    for index in range(int(count)):
        candidate_request = deepcopy(dict(request))
        candidate_binding = deepcopy(dict(candidate_request["binding_identity"]))
        candidate_binding.pop("target_position_m", None)
        candidate_binding.pop("motion_counts_by_actor", None)
        candidate_binding["route_seed_offset"] = base_offset + index * 1009
        candidate_request["binding_identity"] = candidate_binding
        try:
            route, _tracks = _select_topology(
                plan, candidate_request, None, dummy_first, dummy_second
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {str(exc)}"
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        candidates.append({
            "target_m": deepcopy(route["target_m"]),
            "route_records": deepcopy(route["route_records"]),
            "route_seed_offset": int(candidate_binding["route_seed_offset"]),
            "preflight": deepcopy(route.get("preflight")),
        })
    if not candidates:
        raise IdentityNativeError(
            "bounded native PathFinder sampling found no legal identity topology "
            f"candidates; candidate_attempts={int(count)}; "
            f"rejection_count={sum(rejection_counts.values())}; "
            f"rejection_reasons={rejection_counts}"
        )
    return candidates




def _raw_native_polyline_candidates(
    plan: Mapping[str, Any], request: Mapping[str, Any],
    *, pool_path: str | None = None,
) -> list[dict[str, Any]]:
    """Load common-endpoint polylines queried from UE Recast.

    These are sparse native waypoints. Their route shape stays authoritative,
    while the identity clock later samples each selected polyline by arc
    length to a legal integer frame count. The result deliberately carries no
    ``native_frame_counts`` claim because the query itself has no video clock.
    """
    config = _config(request)
    query_value = pool_path if pool_path is not None else config.get(
        "native_polyline_query_path"
    )
    if not isinstance(query_value, str) or not query_value.strip():
        return []
    query_path = _file(
        query_value, base=native.REPOSITORY,
        owner="native UE polyline query",
    )
    query = _load(query_path)
    if query.get("status") != "pass":
        raise IdentityNativeError(
            f"native UE polyline query is not a passing receipt: {query_path}"
        )
    rows = query.get("candidate_pairs")
    if not isinstance(rows, list):
        raise IdentityNativeError(
            f"native UE polyline query lacks candidate_pairs: {query_path}"
        )
    frames = plan.get("visual_plan", {}).get("frames")
    if not isinstance(frames, list) or not frames or not isinstance(frames[0], Mapping):
        raise IdentityNativeError("identity plan lacks a first frame for native polyline validation")
    states = frames[0].get("actor_states")
    if not isinstance(states, list):
        raise IdentityNativeError("identity plan first frame lacks actor states")
    starts = {
        str(state["actor_id"]): np.asarray(
            state["root_transform"]["translation_m"], dtype=float,
        )
        for state in states
        if isinstance(state, Mapping)
        and state.get("actor_id") in {"source1", "source2"}
    }
    if set(starts) != {"source1", "source2"}:
        raise IdentityNativeError("native polyline validation requires source1/source2 starts")
    lower, upper = map(float, config["path_length_range_m"])
    candidates: list[dict[str, Any]] = []
    for candidate_index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            continue
        try:
            target = np.asarray(raw["target_m"], dtype=float)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            continue
        paths_value = raw.get("paths")
        if not isinstance(paths_value, Mapping):
            continue
        paths: dict[str, np.ndarray] = {}
        route_records: dict[str, dict[str, Any]] = {}
        valid = True
        for actor_id in ("source1", "source2"):
            try:
                path = np.asarray(paths_value[actor_id], dtype=float)
            except (KeyError, TypeError, ValueError, OverflowError):
                valid = False
                break
            if (
                path.ndim != 2
                or path.shape[1] != 3
                or len(path) < 2
                or not np.all(np.isfinite(path))
                or not np.allclose(path[0], starts[actor_id], atol=1.0e-4, rtol=0.0)
                or not np.allclose(path[-1], target, atol=1.0e-5, rtol=0.0)
            ):
                valid = False
                break
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if not lower <= length <= upper:
                valid = False
                break
            if not _motion_count_options(
                length, config, float(plan["clock"]["frame_rate_hz"])
            ):
                valid = False
                break
            paths[actor_id] = path
            route_records[actor_id] = {
                "pathfinder_polyline_m": path.tolist(),
                "path_length_m": length,
                "authority": "native_spear_ue_recast_common_endpoint_query",
                "source_query_path": str(query_path),
                "query_candidate_index": int(candidate_index),
                "native_waypoint_count": int(len(path)),
                "native_timing_preserved": False,
            }
        if not valid:
            continue
        if np.linalg.norm(paths["source1"][-1] - paths["source2"][-1]) > 1.0e-5:
            continue
        candidates.append({
            "target_m": target.tolist(),
            "paths": paths,
            "route_records": route_records,
            "route_seed_offset": int(config.get("route_seed_offset", 910)) + int(candidate_index),
            "native_polyline_source": str(query_path),
            "native_polyline_query_candidate_index": int(candidate_index),
            "native_timing_preserved": False,
        })
    if not candidates:
        raise IdentityNativeError(
            "native UE polyline query has no common-endpoint paths within the declared path range"
        )
    return candidates

IDENTITY_GEOMETRY_POOL_SCHEMAS = frozenset({
    "avengine_binding_identity_apartment_common_endpoint_query_v1",
})
IDENTITY_GEOMETRY_QUERY_REQUEST_SCHEMA = (
    "avengine_identity_native_geometry_query_request_v1"
)
IDENTITY_GEOMETRY_QUERY_AUTHORIZATION = "identity_native_geometry_query_granted_v1"


class IdentityGeometryQueryRequired(IdentityNativeError):
    """No declared real geometry pool serves this plan's actual start positions.

    The plan itself stays authoritative: its placement and camera were accepted
    by the ordinary planner under the declared profile, so the remedy is a new
    native geometry query at those exact starts, never a relocated actor or a
    hand-written polyline. ``report`` carries the per-pool rejection reasons and
    the executable query request.
    """

    def __init__(self, message: str, *, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = deepcopy(dict(report))


def _identity_start_positions(plan: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The plan's own first-frame source placement, in plan coordinates."""
    frames = plan.get("visual_plan", {}).get("frames")
    if not isinstance(frames, list) or not frames or not isinstance(frames[0], Mapping):
        raise IdentityNativeError(
            "identity plan lacks a first frame for geometry pool matching"
        )
    states = frames[0].get("actor_states")
    if not isinstance(states, list):
        raise IdentityNativeError("identity plan first frame lacks actor states")
    starts: dict[str, np.ndarray] = {}
    for state in states:
        if not isinstance(state, Mapping):
            continue
        actor_id = state.get("actor_id")
        if actor_id not in {"source1", "source2"}:
            continue
        point = np.asarray(
            _mapping(state.get("root_transform"), f"{actor_id} root_transform")[
                "translation_m"
            ],
            dtype=float,
        )
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise IdentityNativeError(f"{actor_id} start position is not finite")
        starts[str(actor_id)] = point
    if set(starts) != {"source1", "source2"}:
        raise IdentityNativeError(
            "identity geometry matching requires source1 and source2 starts"
        )
    return starts


def _geometry_pool_specs(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Declared real geometry candidate pools, in the order they are tried.

    ``native_geometry_pools`` is the configurable list. The single-path
    ``native_polyline_query_path`` stays accepted so plans written before the
    list existed keep resolving to the same pool.
    """
    config = _config(request)
    specs: list[dict[str, Any]] = []
    declared = config.get("native_geometry_pools")
    if declared is not None:
        if (
            not isinstance(declared, Sequence)
            or isinstance(declared, (str, bytes))
        ):
            raise IdentityNativeError(
                "binding_identity.native_geometry_pools must be a list of pool declarations"
            )
        for index, row in enumerate(declared):
            if isinstance(row, str):
                row = {"path": row}
            if not isinstance(row, Mapping):
                raise IdentityNativeError(
                    f"binding_identity.native_geometry_pools[{index}] must be a path or an object"
                )
            path = row.get("path")
            if not isinstance(path, str) or not path.strip():
                raise IdentityNativeError(
                    f"binding_identity.native_geometry_pools[{index}] lacks a path"
                )
            specs.append({
                "declared_index": int(index),
                "path": path,
                "room_id": row.get("room_id"),
                "kind": str(row.get("kind") or "native_common_endpoint_query"),
                "origin": "binding_identity.native_geometry_pools",
            })
    legacy = config.get("native_polyline_query_path")
    if isinstance(legacy, str) and legacy.strip():
        resolved = {str(Path(spec["path"]).expanduser()) for spec in specs}
        if str(Path(legacy).expanduser()) not in resolved:
            specs.append({
                "declared_index": len(specs),
                "path": legacy,
                "room_id": None,
                "kind": "native_common_endpoint_query",
                "origin": "binding_identity.native_polyline_query_path",
            })
    return specs


def _geometry_pool_room_ids(pool: Mapping[str, Any]) -> set[str]:
    """Room identities the pool receipt itself claims, from declared fields."""
    names: set[str] = set()
    for key in ("room_id", "scene_id", "room_family"):
        value = pool.get(key)
        if isinstance(value, str) and value.strip():
            names.add(value.strip())
    scene = pool.get("scene")
    if isinstance(scene, Mapping):
        for key in ("room_id", "scene_id"):
            value = scene.get(key)
            if isinstance(value, str) and value.strip():
                names.add(value.strip())
    manifest = pool.get("source_manifest") or pool.get("room_manifest")
    if isinstance(manifest, str) and manifest.strip():
        names.add(Path(manifest).parent.name)
    return names


def _geometry_pool_compatibility(
    spec: Mapping[str, Any],
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Check one declared pool against this plan without altering the plan.

    Room, coordinate unit, floor height, declared starts and the current
    camera-bearing conditions are all read from real data; nothing here
    relocates an actor or edits a route.
    """
    row: dict[str, Any] = {
        "declared_index": int(spec.get("declared_index", 0)),
        "declared_path": str(spec["path"]),
        "origin": str(spec.get("origin") or "binding_identity"),
        "kind": str(spec.get("kind") or "native_common_endpoint_query"),
        "status": "incompatible",
        "reasons": [],
        "candidate_count": 0,
    }
    try:
        path = _file(spec["path"], base=native.REPOSITORY, owner="identity geometry pool")
    except IdentityNativeError as exc:
        row["reasons"].append(f"pool_path_unreadable: {exc}")
        return row
    row["resolved_path"] = str(path)
    try:
        pool = _load(path)
    except IdentityNativeError as exc:
        row["reasons"].append(f"pool_unreadable: {exc}")
        return row
    schema = pool.get("schema")
    row["pool_schema"] = schema
    if schema not in IDENTITY_GEOMETRY_POOL_SCHEMAS:
        row["reasons"].append(f"pool_schema_not_accepted: {schema!r}")
    if pool.get("status") != "pass":
        row["reasons"].append(f"pool_status_not_pass: {pool.get('status')!r}")
    scene = plan.get("scene") if isinstance(plan.get("scene"), Mapping) else {}
    plan_rooms = {
        str(value).strip()
        for value in (scene.get("room_id"), scene.get("scene_id"), request.get("room_id"))
        if isinstance(value, str) and value.strip()
    }
    pool_rooms = _geometry_pool_room_ids(pool)
    row["plan_room_ids"] = sorted(plan_rooms)
    row["pool_room_ids"] = sorted(pool_rooms)
    declared_room = spec.get("room_id")
    if isinstance(declared_room, str) and declared_room.strip():
        if declared_room.strip() not in plan_rooms:
            row["reasons"].append(
                f"declared_pool_room_is_not_the_plan_room: {declared_room!r}"
            )
    if pool_rooms and plan_rooms and not (pool_rooms & plan_rooms):
        row["reasons"].append("pool_room_identity_does_not_match_the_plan_room")
    starts = _identity_start_positions(plan)
    row["plan_starts_m"] = {key: value.tolist() for key, value in starts.items()}
    fixed = pool.get("fixed_starts_m")
    if isinstance(fixed, Sequence) and not isinstance(fixed, (str, bytes)):
        try:
            declared_starts = np.asarray(fixed, dtype=float)
        except (TypeError, ValueError, OverflowError):
            declared_starts = None
        if declared_starts is not None and declared_starts.ndim == 2 and declared_starts.shape[1] == 3:
            row["pool_fixed_starts_m"] = declared_starts.tolist()
            row["start_match_distances_m"] = {
                actor_id: float(
                    np.min(np.linalg.norm(declared_starts - point[None, :], axis=1))
                )
                for actor_id, point in starts.items()
            }
    if row["reasons"]:
        return row
    try:
        candidates = _raw_native_polyline_candidates(
            plan, _request_with_single_pool(request, str(path)),
        )
    except IdentityNativeError as exc:
        row["reasons"].append(f"pool_has_no_legal_common_endpoint_for_these_starts: {exc}")
        return row
    row["candidate_count"] = len(candidates)
    row["candidate_indices_sample"] = [
        int(item["native_polyline_query_candidate_index"]) for item in candidates[:16]
    ]
    row["candidate_target_sample_m"] = [
        [round(float(value), 6) for value in item["target_m"]] for item in candidates[:4]
    ]
    row["status"] = "compatible"
    return row


def _request_with_single_pool(
    request: Mapping[str, Any], path: str,
) -> dict[str, Any]:
    """A request copy that points the existing filter at exactly one pool."""
    scoped = deepcopy(dict(request))
    config = deepcopy(dict(_config(request)))
    config.pop("native_geometry_pools", None)
    config["native_polyline_query_path"] = str(path)
    scoped["binding_identity"] = config
    return scoped


def _route_bank_identity_capability(
    plan: Mapping[str, Any], request: Mapping[str, Any], space: Any,
) -> dict[str, Any]:
    """Measure whether the retained route bank can serve identity at all.

    A common endpoint only exists where two retained routes share a vertex,
    because a walker never leaves the single route its start sits on. This
    reports that structure so a blocked group says which geometry is missing
    instead of repeating a per-attempt rejection string.
    """
    row: dict[str, Any] = {"available": False}
    try:
        bank = space.route_bank()
    except Exception as exc:  # a non-route-bank space simply has none
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    if not bank:
        return row
    row["available"] = True
    row["route_count"] = int(len(bank))
    row["route_authority"] = str(space.metadata.get("route_authority", ""))
    vertices = []
    for route in bank:
        points = np.asarray(route["points_m"], dtype=float)
        vertices.append({tuple(np.round(point, 6)) for point in points})
    shared = 0
    shared_examples: list[list[str]] = []
    for first in range(len(bank)):
        for second in range(first + 1, len(bank)):
            if vertices[first] & vertices[second]:
                shared += 1
                if len(shared_examples) < 4:
                    shared_examples.append(
                        [str(bank[first]["route_id"]), str(bank[second]["route_id"])]
                    )
    row["route_pairs_sharing_a_vertex"] = int(shared)
    row["route_pairs_sharing_a_vertex_examples"] = shared_examples
    row["distinct_route_pairs"] = int(len(bank) * (len(bank) - 1) // 2)
    starts = _identity_start_positions(plan)
    row["plan_starts_on_a_shared_route"] = bool(
        any(
            all(
                min(
                    float(np.linalg.norm(np.asarray(vertex, dtype=float) - point))
                    for vertex in vertex_set
                ) <= 1.0e-5
                for point in starts.values()
            )
            for vertex_set in vertices
        )
    )
    row["identity_requires"] = (
        "two retained routes sharing one vertex, or one route walked inward "
        "from both of its ends"
    )
    return row


def identity_geometry_query_request(
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    reason: str,
    pools: Sequence[Mapping[str, Any]] = (),
    requested_common_endpoints: int = 1200,
) -> dict[str, Any]:
    """Build the native geometry query this plan actually needs.

    The request is executable data, not a launch: it names the exact starts the
    ordinary planner produced, the declared filters the answer has to satisfy,
    and the native session it should share. ``execute_identity_geometry_query``
    refuses to run it without an explicit authorization value.
    """
    config = _config(request)
    starts = _identity_start_positions(plan)
    resources = plan.get("resources") if isinstance(plan.get("resources"), Mapping) else {}
    package = (
        resources.get("room_package")
        if isinstance(resources.get("room_package"), Mapping) else {}
    )
    scene = plan.get("scene") if isinstance(plan.get("scene"), Mapping) else {}
    clock = _mapping(plan.get("clock"), "plan.clock")
    return {
        "schema": IDENTITY_GEOMETRY_QUERY_REQUEST_SCHEMA,
        "status": "requested",
        "reason": str(reason),
        "task_family": FAMILY,
        "episode_id": request.get("episode_id"),
        "group_id": request.get("group_id"),
        "member_role": request.get("member_role"),
        "room": {
            "room_id": scene.get("room_id") or request.get("room_id"),
            "scene_id": scene.get("scene_id"),
            "room_family": package.get("family"),
            "renderer": package.get("renderer") or resources.get("backend"),
            "map_path": resources.get("map_path"),
            "room_manifest": resources.get("room_manifest"),
            "floor_reference": deepcopy(package.get("floor_reference")),
            "coordinate_frame": deepcopy(
                plan.get("coordinate_frame") or package.get("coordinate_frame")
            ),
        },
        "fixed_starts_m": [
            starts["source1"].tolist(), starts["source2"].tolist(),
        ],
        "fixed_start_actor_ids": ["source1", "source2"],
        "fixed_start_authority": (
            "ordinary planner placement in this plan's first frame; the query "
            "must not move it"
        ),
        "requested_common_endpoints": int(requested_common_endpoints),
        "declared_filters": {
            "frame_rate_hz": float(clock["frame_rate_hz"]),
            "frame_count": int(clock["frame_count"]),
            "path_length_range_m": [float(value) for value in config["path_length_range_m"]],
            "walk_speed_range_mps": [float(value) for value in config["walk_speed_range_mps"]],
            "minimum_motion_s": float(config["minimum_motion_s"]),
            "minimum_entity_separation_m": float(config["minimum_entity_separation_m"]),
            "same_floor_tolerance_m": float(config["same_floor_tolerance_m"]),
            "visibility_margin_deg": float(config["visibility_margin_deg"]),
            "minimum_bearing_change_deg": float(config["minimum_bearing_change_deg"]),
        },
        "native_calls": {
            "convention": (
                "tools/routes/build_apartment_route_bank.py:_query_routes with "
                "repeated common endpoints"
            ),
            "navigation_data_actor": "RecastNavMesh-Default",
            "steps": [
                "navigation.get_random_points(navigation_data, num_points=requested_common_endpoints)",
                "navigation.find_paths(start_points=[start1]*n + [start2]*n, end_points=targets*2)",
                "write both polylines per target into candidate_pairs[].paths",
            ],
            "receipt_schema": sorted(IDENTITY_GEOMETRY_POOL_SCHEMAS)[0],
        },
        "native_resources": {
            "renderer": package.get("renderer") or resources.get("backend"),
            "execution": "gpu",
            "kind": "gpu_native_visual",
            "runtime_context": "renderer_native",
            "stage": "capture",
            "preferred_session": "identity_probe_capture",
            "session_sharing": (
                "reuse the identity_probe_capture SpearSim session for this map; "
                "launch nothing ahead of a capture that has not run"
            ),
            "rpc_port": (request.get("runtime") or {}).get("rpc_port"),
            "graphics_adapter": (request.get("runtime") or {}).get("graphics_adapter"),
        },
        "budget_claim": {
            "extra_visual_launches_if_shared": 0,
            "extra_visual_launches_if_standalone": 1,
            "extra_rlr_contexts": 0,
            "authorization_required": IDENTITY_GEOMETRY_QUERY_AUTHORIZATION,
            "started_here": 0,
        },
        "execution_entrypoint": (
            "avengine.dataset.binding_group_identity.execute_identity_geometry_query"
        ),
        "recovery_entrypoint": (
            "avengine.dataset.binding_group_identity.adopt_identity_geometry_query_receipt"
        ),
        "rejected_pools": [deepcopy(dict(row)) for row in pools],
    }


def identity_geometry_pool_report(
    plan: Mapping[str, Any], request: Mapping[str, Any],
) -> dict[str, Any]:
    """Filter every declared real geometry pool through the ordinary entry point.

    Returns the selected pool when one serves this plan's own starts, and
    otherwise the executable query request plus the reason each pool was
    rejected. No pool is edited, no start is moved.
    """
    specs = _geometry_pool_specs(request)
    rows = [_geometry_pool_compatibility(spec, plan, request) for spec in specs]
    selected = next((row for row in rows if row["status"] == "compatible"), None)
    report: dict[str, Any] = {
        "schema": "avengine_identity_geometry_pool_report_v1",
        "declared_pool_count": len(specs),
        "pools": rows,
        "plan_starts_m": {
            key: value.tolist()
            for key, value in _identity_start_positions(plan).items()
        },
        "selected_pool": deepcopy(selected) if selected is not None else None,
        "status": "compatible" if selected is not None else "requires_native_geometry_query",
    }
    if selected is None:
        report["query_request"] = identity_geometry_query_request(
            plan, request,
            reason=(
                "no declared geometry pool has a legal common endpoint for the "
                "planner placement in this plan"
            ),
            pools=rows,
        )
    return report


def execute_identity_geometry_query(
    query_request: Mapping[str, Any],
    output_path: str | Path,
    *,
    authorization: str | None = None,
    session: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a granted identity geometry query, or refuse and say so.

    The controller owns the native budget, so an ungranted call returns a
    ``refused`` receipt with the launch count it did not spend instead of
    starting a renderer. A granted call is executed by the native session
    handed in through ``session``; this function never launches its own.
    """
    if not isinstance(query_request, Mapping):
        raise IdentityNativeError("identity geometry query request must be an object")
    if query_request.get("schema") != IDENTITY_GEOMETRY_QUERY_REQUEST_SCHEMA:
        raise IdentityNativeError(
            "identity geometry query request schema is not "
            f"{IDENTITY_GEOMETRY_QUERY_REQUEST_SCHEMA}"
        )
    target = Path(output_path).expanduser().resolve()
    if authorization != IDENTITY_GEOMETRY_QUERY_AUTHORIZATION:
        return {
            "status": "refused",
            "reason": "native_geometry_query_not_authorized",
            "required_authorization": IDENTITY_GEOMETRY_QUERY_AUTHORIZATION,
            "native_visual_worlds_created": 0,
            "native_acoustic_contexts_created": 0,
            "requested_output": str(target),
            "query_request": deepcopy(dict(query_request)),
        }
    if not isinstance(session, Mapping) or not callable(session.get("query")):
        raise IdentityNativeError(
            "an authorized identity geometry query needs session['query'], a "
            "callable on an already-open native session; this entry point does "
            "not launch a renderer of its own"
        )
    receipt = session["query"](deepcopy(dict(query_request)))
    if not isinstance(receipt, Mapping):
        raise IdentityNativeError("identity geometry query session returned no receipt")
    written = _write(target, dict(receipt))
    return {
        "status": "pass",
        "receipt_path": str(written),
        "native_visual_worlds_created": int(
            session.get("native_visual_worlds_created", 0)
        ),
        "native_acoustic_contexts_created": 0,
        "shared_session": bool(session.get("shared", False)),
    }


def adopt_identity_geometry_query_receipt(
    receipt_path: str | Path,
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Adopt an existing query receipt as this plan's geometry pool.

    This is the recovery side: a receipt written by an earlier granted launch,
    or by a previous attempt, is validated against the current plan and turned
    into candidates without any new native work. The returned request copy is
    what the ordinary topology entry point consumes.
    """
    path = _file(receipt_path, base=native.REPOSITORY, owner="identity geometry receipt")
    row = _geometry_pool_compatibility(
        {
            "declared_index": 0,
            "path": str(path),
            "kind": "native_common_endpoint_query",
            "origin": "adopt_identity_geometry_query_receipt",
        },
        plan, request,
    )
    if row["status"] != "compatible":
        raise IdentityNativeError(
            "adopted identity geometry receipt does not serve this plan: "
            f"{row['reasons']}"
        )
    config = deepcopy(dict(_config(request)))
    declared = list(config.get("native_geometry_pools") or [])
    declared.insert(0, {"path": str(path), "kind": "native_common_endpoint_query"})
    config["native_geometry_pools"] = declared
    adopted = deepcopy(dict(request))
    adopted["binding_identity"] = config
    return {
        "status": "pass",
        "pool": row,
        "request": adopted,
        "native_visual_worlds_created": 0,
        "native_acoustic_contexts_created": 0,
    }


def _schedule_for_geometry(
    geometry: Mapping[str, Any], request: Mapping[str, Any],
    first_sound: Mapping[str, Any], second_sound: Mapping[str, Any],
) -> dict[str, Any] | None:
    config = _config(request)
    clock = _mapping(request, "request")
    fps = float(request["frame_rate_hz"])
    sr = int(request["sample_rate_hz"])
    stride = int(request.get("rir_stride", 3))
    first_dry_end_s = float(config["event_start_s"]) + int(first_sound["sample_count"]) / sr
    first_support_key_frame = int(math.ceil(first_dry_end_s * fps / stride - 1.0e-9)) * stride
    motion_start = first_support_key_frame + int(math.ceil(float(config["minimum_silence_after_emission_s"]) * fps - 1.0e-9))
    duration_s = float(request.get("duration_seconds", int(request["frame_count"]) / float(request["frame_rate_hz"])))
    latest_event2_start = int(math.floor((duration_s - float(request["profile"]["reserve_tail_s"]) - int(second_sound["sample_count"]) / sr) * fps + 1.0e-9))
    route_records = geometry.get("route_records")
    if not isinstance(route_records, Mapping):
        return None
    count_options: dict[str, list[int]] = {}
    native_counts = geometry.get("native_frame_counts")
    for actor_id in ("source1", "source2"):
        row = route_records.get(actor_id)
        if not isinstance(row, Mapping):
            return None
        if isinstance(native_counts, Mapping):
            try:
                selected_count = int(native_counts[actor_id])
            except (KeyError, TypeError, ValueError, OverflowError):
                return None
            if selected_count < 2:
                return None
            count_options[actor_id] = [selected_count]
        else:
            options = _motion_count_options(float(row["path_length_m"]), config, fps)
            if not options:
                return None
            count_options[actor_id] = options
    valid_count_pairs = []
    post_frames = int(math.ceil(float(config["post_motion_silence_s"]) * fps - 1.0e-9))
    for first_steps in count_options["source1"]:
        for second_steps in count_options["source2"]:
            counts = (first_steps + 1, second_steps + 1)
            raw_event2_frame = motion_start + max(counts) + post_frames
            event2_frame = int(math.ceil(raw_event2_frame / stride - 1.0e-9)) * stride
            if event2_frame <= latest_event2_start and event2_frame < int(clock["frame_count"]):
                valid_count_pairs.append({
                    "source1": counts[0], "source2": counts[1],
                    "event2_frame": event2_frame,
                })
    if not valid_count_pairs:
        return None
    return {
        "first_dry_end_s": first_dry_end_s,
        "first_support_key_frame": first_support_key_frame,
        "motion_start_frame": motion_start,
        "latest_event2_start_frame": latest_event2_start,
        "count_options": count_options,
        "valid_count_pairs": valid_count_pairs,
        "rir_stride_frames": stride,
    }


def _select_sound_pair(
    request: Mapping[str, Any], actor_assets: Sequence[str],
    asset_records: Mapping[str, Mapping[str, Any]], *,
    plan: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pool_path = _file(request.get("sound_pool"), base=native.REPOSITORY, owner="current sound pool")
    pool = _load(pool_path)
    raw_sounds = pool.get("sounds") if isinstance(pool.get("sounds"), list) else []
    config = _config(request)
    identity_field = str(config["sound_identity_field"])
    from avengine.rooms.conditioned_sampler import sound_matches
    eligible: list[dict[str, Any]] = []
    for raw in raw_sounds:
        if not isinstance(raw, Mapping):
            continue
        sound = deepcopy(dict(raw))
        if sound.get("sound_class") not in {"speech", "speech_playback"}:
            continue
        identity = sound.get(identity_field)
        if not isinstance(identity, str) or not identity.strip():
            continue
        if int(sound.get("sample_rate_hz", -1)) != int(request["sample_rate_hz"]):
            continue
        count = sound.get("sample_count")
        start = sound.get("audible_start_sample")
        end = sound.get("audible_end_sample_exclusive")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (count, start, end)):
            continue
        if count <= 0 or not 0 <= start < end <= count:
            continue
        if not all(asset in sound.get("compatible_asset_ids", []) for asset in actor_assets):
            continue
        if not all(sound_matches(asset_records[asset], sound) for asset in actor_assets):
            continue
        _sound_path(sound, pool_path)
        eligible.append(sound)
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sound in eligible:
        by_identity[str(sound[identity_field])].append(sound)
    pairs: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for identity, sounds in sorted(by_identity.items()):
        sounds.sort(key=lambda item: str(item["sound_asset_id"]))
        for first_index, first in enumerate(sounds):
            for second in sounds[first_index + 1 :]:
                if first.get("transcript") == second.get("transcript"):
                    continue
                pairs.append((first, second, identity))
    if not pairs:
        raise IdentityNativeError(
            "current sound pool has no two distinct complete clips sharing the configured sound identity"
        )
    if plan is None:
        raise IdentityNativeError(
            "sound selection requires a base plan for native feasible-window filtering"
        )
    # Sample legal native geometry once, then evaluate every complete pair
    # against its integer clock window.  This keeps pair choice uniform over
    # the feasible set without replanning the room hundreds of times.
    geometries = _sample_native_topologies(plan, request)
    feasible: list[tuple[dict[str, Any], dict[str, Any], str, list[tuple[dict[str, Any], dict[str, Any]]]]] = []
    timing_rejected = 0
    for first, second, identity in pairs:
        options = []
        for geometry in geometries:
            schedule = _schedule_for_geometry(geometry, request, first, second)
            if schedule is not None:
                options.append((geometry, schedule))
        if options:
            feasible.append((first, second, identity, options))
        else:
            timing_rejected += 1
    if not feasible:
        raise IdentityNativeError(
            f"no complete same-identity sound pair fits the declared native feasible window; candidates={len(pairs)} geometries={len(geometries)}"
        )
    uniform_seed = int(request.get("seed", 0)) + 1701
    uniform_rng = np.random.default_rng(uniform_seed)
    selected_index = int(uniform_rng.integers(len(feasible)))
    first, second, identity, geometry_options = feasible[selected_index]
    geometry_index = int(uniform_rng.integers(len(geometry_options)))
    geometry, schedule = geometry_options[geometry_index]
    valid_pair_index = int(uniform_rng.integers(len(schedule["valid_count_pairs"])))
    selected_counts = schedule["valid_count_pairs"][valid_pair_index]
    binding = request.get("binding_identity")
    if not isinstance(binding, dict):
        raise IdentityNativeError("request.binding_identity must be mutable for selected native topology")
    binding["target_position_m"] = deepcopy(geometry["target_m"])
    binding["route_seed_offset"] = int(geometry["route_seed_offset"])
    binding["motion_counts_by_actor"] = {
        "source1": int(selected_counts["source1"]),
        "source2": int(selected_counts["source2"]),
    }
    feasible_ids = [
        [str(item[0]["sound_asset_id"]), str(item[1]["sound_asset_id"])]
        for item in feasible
    ]
    feasible_geometry_counts = [len(item[3]) for item in feasible]
    return first, second, {
        "pool_path": str(pool_path),
        "sound_identity_field": identity_field,
        "sound_identity_value": identity,
        "eligible_sound_count": len(eligible),
        "candidate_pair_count": len(pairs),
        "timing_rejected_pair_count": timing_rejected,
        "feasible_pair_count": len(feasible),
        "feasible_pair_ids": feasible_ids,
        "feasible_geometry_count": len(geometries),
        "feasible_geometry_options_per_pair": feasible_geometry_counts,
        "selected_geometry": {
            "target_m": deepcopy(geometry["target_m"]),
            "route_seed_offset": int(geometry["route_seed_offset"]),
            "motion_schedule": deepcopy(schedule),
            "selected_motion_counts_by_actor": {
                "source1": int(selected_counts["source1"]),
                "source2": int(selected_counts["source2"]),
            },
        },
        "selected": [str(first["sound_asset_id"]), str(second["sound_asset_id"])],
        "selected_durations_s": [
            int(first["sample_count"]) / int(first["sample_rate_hz"]),
            int(second["sample_count"]) / int(second["sample_rate_hz"]),
        ],
        "uniform_seed": uniform_seed,
        "selection": "seeded_uniform_from_native_feasible_complete_pairs",
        "feasible_window": {
            "event_start_s": float(config["event_start_s"]),
            "minimum_silence_after_emission_s": float(config["minimum_silence_after_emission_s"]),
            "rir_stride_frames": int(request.get("rir_stride", 3)),
            "reserve_tail_s": float(request["profile"]["reserve_tail_s"]),
            "full_clip_required": True,
            "native_pathfinder_required": True,
        },
    }


def _first_event_actor(request: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    binding = request.get("binding_identity")
    if not isinstance(binding, Mapping):
        raise IdentityNativeError("request.binding_identity is required for first-event actor selection")
    explicit = binding.get("first_event_actor_id")
    if explicit is not None:
        actor_id = str(explicit)
        if actor_id not in {"source1", "source2"}:
            raise IdentityNativeError("binding_identity.first_event_actor_id must be source1 or source2")
        return actor_id, {"actor_id": actor_id, "selection": "explicit_request"}
    seed = int(request.get("seed", 0)) + 3101
    actor_id = ("source1", "source2")[int(np.random.default_rng(seed).integers(2))]
    if isinstance(binding, dict):
        binding["first_event_actor_id"] = actor_id
    return actor_id, {
        "actor_id": actor_id,
        "selection": "seeded_uniform_two_compatible_actors",
        "uniform_seed": seed,
    }

def _event(sound: Mapping[str, Any], *, event_id: str, actor_id: str, start_s: float, clock: Mapping[str, Any]) -> dict[str, Any]:
    sr = int(clock["sample_rate_hz"])
    tb = int(clock["time_base_hz"])
    start = int(round(float(start_s) * sr))
    duration = int(sound["sample_count"])
    end = start + duration
    if start < 0 or end > int(clock["sample_count"]):
        raise IdentityNativeError(f"{event_id} does not fit the episode clock")
    row = deepcopy(dict(sound))
    row.update(
        event_id=event_id,
        actor_id=actor_id,
        source_endpoint_id=f"{actor_id}_mouth",
        voice_binding_actor_id=actor_id,
        start_sample=start,
        end_sample=end,
        end_sample_exclusive=end,
        start_s=start / sr,
        end_s=end / sr,
        start_tick=int(round(start * tb / sr)),
        end_tick=int(round(end * tb / sr)),
        end_tick_exclusive=int(round(end * tb / sr)),
        source_start_sample=0,
        source_end_sample_exclusive=duration,
        linear_gain=1.0,
        event_unit="independent_source_playback_onset",
        planned_audible_interval_samples=[
            start + int(sound["audible_start_sample"]),
            start + int(sound["audible_end_sample_exclusive"]),
        ],
    )
    return row


def _actor_declarations(registry: Mapping[str, Any], actor_assets: Sequence[str]) -> list[dict[str, Any]]:
    actors = []
    for index, asset_id in enumerate(actor_assets, start=1):
        actor_id = f"source{index}"
        record = resolve_source_asset_runtime_profile(registry, str(asset_id))
        timeline = deepcopy(dict(record["timeline"]))
        emitter = build_asset_emitter_binding(
            registry,
            source_slot_id=actor_id,
            asset_id=str(asset_id),
            anchor_id=str(record.get("default_emitter_anchor_id") or "mouth"),
        )
        actors.append({
            "actor_id": actor_id,
            "asset_id": str(asset_id),
            "asset_revision": str(record["revision"]),
            "entity_class": str(record["entity_class"]),
            "identity": deepcopy(dict(record.get("identity") or {})),
            "realized_attributes": deepcopy(dict(record.get("realized_attributes") or {})),
            "display_label": str(record.get("display_label") or asset_id),
            "emitter_binding": emitter,
            "timeline": timeline,
            "motion_model": "articulated",
            "native_binding_status": "pending_executor",
        })
    return actors


def _rotation(heading: float) -> tuple[list[float], np.ndarray]:
    c, s = math.cos(heading), math.sin(heading)
    return [0.0, math.sin(heading / 2.0), 0.0, math.cos(heading / 2.0)], np.asarray(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float
    )


def _actor_states(actor: Mapping[str, Any], initial: Mapping[str, Any], points: np.ndarray, moving_start: int, moving_end: int, clock: Mapping[str, Any]) -> list[dict[str, Any]]:
    timeline = actor.get("timeline")
    if not isinstance(timeline, Mapping) or not timeline.get("walking_action_id") or not timeline.get("idle_action_id"):
        raise IdentityNativeError(f"{actor.get('actor_id')} lacks registered native idle/walk actions")
    ticks = int(clock["ticks_per_frame"])
    period = int(timeline["walk_phase_period_frames"]) * ticks
    axis = np.asarray(timeline.get("local_anatomical_forward_axis", [0.0, 0.0, 1.0]), dtype=float)
    anatomical_yaw = math.atan2(-axis[2], axis[0])
    initial_q = np.asarray(initial["root_transform"]["rotation_xyzw"], dtype=float)
    heading = 2.0 * math.atan2(float(initial_q[1]), float(initial_q[3]))
    motion_tick = 0
    offset = np.asarray(actor["emitter_binding"]["emitter_offset_m"], dtype=float)
    result: list[dict[str, Any]] = []
    for frame_index, point in enumerate(points):
        moving = moving_start <= frame_index < moving_end and frame_index + 1 < len(points) and np.linalg.norm(points[frame_index + 1] - point) * float(clock["frame_rate_hz"]) > 0.05
        if moving:
            direction = points[frame_index + 1] - point
            heading = math.atan2(-float(direction[2]), float(direction[0])) - anatomical_yaw
            action_id = str(timeline["walking_action_id"])
            action_time_ticks = motion_tick
            action_phase = (motion_tick % period) / period
            motion_tick += ticks
        else:
            action_id = str(timeline["idle_action_id"])
            action_time_ticks = 0
            action_phase = 0.0
        quaternion, matrix = _rotation(heading)
        state = deepcopy(dict(initial))
        state.update(
            actor_id=str(actor["actor_id"]),
            frame_index=frame_index,
            action_id=action_id,
            action_phase=float(action_phase),
            action_time_ticks=int(action_time_ticks),
            moving=bool(moving),
            planned_emitter_m=(point + matrix @ offset).tolist(),
        )
        state["root_transform"] = {
            "translation_m": [float(value) for value in point],
            "rotation_xyzw": quaternion,
            "scale": deepcopy(initial["root_transform"].get("scale", [1.0, 1.0, 1.0])),
        }
        result.append(state)
    return result


def _camera_visible(camera: Mapping[str, Any], point: Sequence[float], mesh: Any, margin_deg: float) -> tuple[bool, dict[str, Any]]:
    origin = np.asarray(camera["position_m"], dtype=float)
    delta = np.asarray(point, dtype=float) - origin
    forward = np.asarray(camera["basis"]["forward"], dtype=float)
    right = np.asarray(camera["basis"]["right"], dtype=float)
    depth = float(delta @ forward)
    angle = math.degrees(math.atan2(float(delta @ right), depth))
    los = line_of_sight(mesh, origin, np.asarray(point, dtype=float))
    ok = depth > 0.0 and abs(angle) < float(camera["horizontal_fov_deg"]) / 2.0 - margin_deg and los == "clear"
    return ok, {"depth_m": depth, "bearing_deg": angle, "line_of_sight": los}


def _min_separation(points: np.ndarray, other: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(np.asarray(points, dtype=float) - np.asarray(other, dtype=float), axis=1)))


def _build_frames(plan: Mapping[str, Any], actors: Sequence[Mapping[str, Any]], tracks: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    clock = plan["clock"]
    camera = deepcopy(plan["visual_plan"]["camera"])
    frames = []
    for frame_index in range(int(clock["frame_count"])):
        states = [deepcopy(tracks[actor["actor_id"]][frame_index]) for actor in actors]
        frame = {
            "frame_index": frame_index,
            "pts_ticks": frame_index * int(clock["ticks_per_frame"]),
            "actor_states": states,
            "camera_state": {**deepcopy(camera), "frame_index": frame_index},
        }
        frames.append(frame)
    return frames


def _base_identity_plan(base_plan: Mapping[str, Any], request: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    plan = deepcopy(dict(base_plan))
    actors = _actor_declarations(registry, request["source_asset_ids"])
    plan["visual_plan"] = deepcopy(dict(plan.get("visual_plan") or {}))
    plan["visual_plan"]["actors"] = actors
    plan["visual_plan"]["camera"] = deepcopy(base_plan["visual_plan"]["camera"])
    plan["request"] = deepcopy(dict(request))
    plan["resources"] = deepcopy(dict(base_plan.get("resources") or {}))
    plan["scene"] = deepcopy(dict(base_plan.get("scene") or {}))
    plan["scene"]["room_id"] = str(request.get("room_id") or plan["scene"].get("room_id"))
    plan["clock"] = deepcopy(dict(base_plan["clock"]))
    plan["condition_profile"] = deepcopy(dict(base_plan.get("condition_profile") or {}))
    plan["condition_profile"]["speech_motion"] = "binding_identity_intervention"
    plan["status"] = "research_candidate"
    plan["qualification_claim"] = False
    plan["formal_dataset_registration_authorized"] = False
    return plan


def _navigation_authority(plan: Mapping[str, Any]) -> str:
    capabilities = plan.get("room_capabilities")
    if isinstance(capabilities, Mapping):
        evidence_refs = capabilities.get("evidence_refs")
        if isinstance(evidence_refs, Mapping):
            navigation = evidence_refs.get("navigation")
            if isinstance(navigation, Mapping):
                authority = navigation.get("authority")
                if isinstance(authority, str) and authority.strip():
                    return authority.strip()
    raise IdentityNativeError(
        "identity plan lacks declared room_capabilities.evidence_refs.navigation.authority"
    )


def _with_audio(plan: Mapping[str, Any], request: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value = deepcopy(dict(plan))
    value["request"] = deepcopy(dict(request))
    value["audio_events"] = [deepcopy(dict(event)) for event in events]
    value["voice_bindings"] = [
        {
            **deepcopy(dict(event)),
            "actor_id": str(event.get("actor_id") or "source1"),
            "source_endpoint_id": f"{event.get('actor_id') or 'source1'}_mouth",
        }
        for event in events
    ]
    value["visual_plan"] = deepcopy(dict(plan["visual_plan"]))
    navigation_authority = _navigation_authority(plan)
    value["planned_conditions"] = {
        "authority": f"{navigation_authority}_and_explicit_identity_intervention",
        "camera_motion": "static",
        "audio_event_schedule": [
            {
                "event_id": event["event_id"],
                "start_sample": event["start_sample"],
                "end_sample_exclusive": event["end_sample_exclusive"],
            }
            for event in events
        ],
    }
    value["activity_plan"] = {"authority": navigation_authority, "actors": []}
    value["camera_condition_sampling"] = deepcopy(value["planned_conditions"])
    return value


def _materialize_visual(base_root: Path, output: Path, plan: Mapping[str, Any], request: Mapping[str, Any]) -> Path:
    if output.exists() or output.is_symlink():
        raise IdentityNativeError(f"refusing existing identity visual root: {output}")
    output.mkdir(parents=True)
    (output / "plan").mkdir()
    for name in (
        "room_package.json",
        "path_bindings.json",
        "room_layout.json",
        "navigation.npz",
        "habitat_room_manifest.json",
    ):
        source = base_root / "plan" / name
        if source.is_file():
            shutil.copy2(source, output / "plan" / name)
    native._write(output / "request.json", request)
    native._write(output / "plan/episode_plan.json", plan)
    native._write(output / "plan/audio_events.json", plan.get("audio_events", []))
    native._write(output / "plan/voice_bindings.json", plan.get("voice_bindings", []))
    family = native.room_family_from_plan(plan)
    if family in {"kujiale", "apartment"}:
        package_path = output / "plan/room_package.json"
        if not package_path.is_file():
            raise IdentityNativeError(
                f"base UE plan lacks room_package.json: {package_path}"
            )
        return output
    if family not in {"hm3d", "mp3d"}:
        raise IdentityNativeError(f"identity first group does not support room family: {family}")
    manifest = output / "plan/habitat_room_manifest.json"
    if not manifest.is_file():
        raise IdentityNativeError(f"base plan lacks Habitat room manifest: {manifest}")
    base_m1 = base_root / "plan/habitat_execution/m1_capture_request.json"
    if not base_m1.is_file():
        raise IdentityNativeError(f"base plan lacks m1 capture request: {base_m1}")
    try:
        materialize_common_plan_habitat(
            plan=plan,
            room_manifest=manifest,
            runtime_registry=request["source_registry"],
            output=output / "plan/habitat_execution",
            base_m1_request=base_m1,
            allow_research_candidate=bool(request.get("allow_research_candidate_assets", False)),
        )
    except Exception as exc:
        raise IdentityNativeError(f"Habitat identity plan materialization failed: {exc}") from exc
    return output


def _capture(request: Mapping[str, Any], visual_root: Path, label: str) -> dict[str, Any]:
    try:
        return native.capture_visual_plan(request, visual_root, label=label)
    except Exception as exc:
        raise IdentityNativeError(str(exc)) from exc


def _endpoint_bindings(
    capture: Mapping[str, Any], plan: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    try:
        return native._neutral_endpoint_bindings(
            capture["neutral_readback"], plan=plan,
        )
    except Exception as exc:
        raise IdentityNativeError(f"native endpoint readback is unavailable: {exc}") from exc


def _assignment_kwargs(assignment: str, targets: Mapping[str, str], endpoints: Mapping[str, str]) -> dict[str, Any]:
    # The native helper accepts repeated event targets while preserving one
    # authoritative endpoint per persistent actor.
    return {
        "assignment_targets": {assignment: targets},
        "expected_event_count": 2,
        "endpoint_by_actor": endpoints,
        "require_authoritative_endpoints": True,
    }


def _probe_audio(plan: Mapping[str, Any], request: Mapping[str, Any], capture: Mapping[str, Any], output: Path) -> tuple[dict[str, Any], float]:
    endpoints = _endpoint_bindings(capture, plan)
    first_event = plan["audio_events"][0]
    target = {str(first_event["event_id"]): str(first_event.get("actor_id") or "source1")}
    params = {
        "assignment_targets": {"a0": target},
        "expected_event_count": 1,
        "endpoint_by_actor": endpoints,
        "require_authoritative_endpoints": True,
    }
    assigned, rebound = native.build_audio_assignment_plan(plan, request, "a0", **params)
    variant_root = native.materialize_audio_variant(
        capture, output / "variant", assigned, rebound, member_id="probe_a0"
    )
    result = native.finalize_audio_assignment(variant_root, rebound)
    report = _load(result["audio_report"])
    tails = report.get("wet_tail_intervals")
    tail_end = None
    if isinstance(tails, list):
        for row in tails:
            if isinstance(row, Mapping) and row.get("event_id") == plan["audio_events"][0]["event_id"]:
                value = row.get("end_sample_exclusive")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    tail_end = max(float(tail_end or 0.0), float(value) / float(plan["clock"]["sample_rate_hz"]))
    if tail_end is None:
        for row in report.get("events", []):
            if isinstance(row, Mapping) and row.get("event_id") == plan["audio_events"][0]["event_id"]:
                value = row.get("wet_tail_end_sample_original", row.get("wet_tail_end_sample"))
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    tail_end = float(value) / float(plan["clock"]["sample_rate_hz"])
    if tail_end is None:
        raise IdentityNativeError("early RLR report has no measured wet-tail end for event_001")
    return result, float(tail_end)




def _camera_visibility_actor_ids(
    plan: Mapping[str, Any], request: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return actor IDs required by the request's anchor-visibility profile."""
    actors = plan.get("visual_plan", {}).get("actors")
    if not isinstance(actors, list):
        return ("source1", "source2")
    profile = request.get("profile")
    indices = profile.get("anchor_indices") if isinstance(profile, Mapping) else None
    if indices is None:
        plan_profile = plan.get("condition_profile")
        if not isinstance(plan_profile, Mapping):
            planning = plan.get("planning_result")
            plan_profile = planning.get("condition_profile") if isinstance(planning, Mapping) else None
        indices = plan_profile.get("anchor_indices") if isinstance(plan_profile, Mapping) else None
    if isinstance(indices, Sequence) and not isinstance(indices, (str, bytes)):
        selected = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                continue
            if 0 <= index < len(actors):
                actor_id = actors[index].get("actor_id")
                if isinstance(actor_id, str) and actor_id.strip():
                    selected.append(actor_id.strip())
        if selected:
            return tuple(dict.fromkeys(selected))
    return tuple(
        str(actor.get("actor_id"))
        for actor in actors
        if isinstance(actor, Mapping)
        and isinstance(actor.get("actor_id"), str)
        and actor.get("actor_id")
    ) or ("source1", "source2")


def _anchor_actor_ids(plan: Mapping[str, Any], request: Mapping[str, Any]) -> set[str]:
    return set(_camera_visibility_actor_ids(plan, request))


def _visibility_margin_for_actor(
    plan: Mapping[str, Any], request: Mapping[str, Any],
    actor_id: str, default_margin_deg: float,
) -> float:
    """Use the declared in-FOV anchor gate without adding edge padding."""
    return 0.0 if actor_id in _anchor_actor_ids(plan, request) else float(default_margin_deg)


def _select_route_bank_topology(
    plan: Mapping[str, Any], request: Mapping[str, Any],
    first_sound: Mapping[str, Any], second_sound: Mapping[str, Any],
    space: Any, mesh: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select an apartment topology from unchanged native route-bank points."""
    config = _config(request)
    camera = plan["visual_plan"]["camera"]
    actors = plan["visual_plan"]["actors"]
    visible_actor_ids = _camera_visibility_actor_ids(plan, request)
    frames = plan["visual_plan"]["frames"]
    initial_states = {
        str(state["actor_id"]): state for state in frames[0]["actor_states"]
    }
    starts = {
        actor_id: np.asarray(
            initial_states[actor_id]["root_transform"]["translation_m"],
            dtype=float,
        )
        for actor_id in ("source1", "source2")
    }
    fps = float(plan["clock"]["frame_rate_hz"])
    count = int(plan["clock"]["frame_count"])
    stride = int(request.get("rir_stride", 3))
    if stride < 1:
        raise IdentityNativeError("request.rir_stride must be positive")
    first_dry_end_s = float(config["event_start_s"]) + int(
        first_sound["sample_count"]
    ) / float(plan["clock"]["sample_rate_hz"])
    first_support_key_frame = int(
        math.ceil(first_dry_end_s * fps / stride - 1.0e-9)
    ) * stride
    motion_start = first_support_key_frame + int(
        math.ceil(float(config["minimum_silence_after_emission_s"]) * fps - 1.0e-9)
    )
    motion_start = max(1, motion_start)
    latest_event2_start = int(
        math.floor(
            (float(plan["clock"]["duration_seconds"])
             - float(request["profile"]["reserve_tail_s"])
             - int(second_sound["sample_count"])
             / int(second_sound["sample_rate_hz"]))
            * fps + 1.0e-9
        )
    )
    try:
        candidates = native_common_endpoint_paths(
            space, starts,
            frame_rate_hz=fps,
            minimum_motion_s=float(config["minimum_motion_s"]),
            path_length_range_m=config["path_length_range_m"],
            walk_speed_range_mps=config["walk_speed_range_mps"],
            minimum_entity_separation_m=float(config["minimum_entity_separation_m"]),
            same_floor_tolerance_m=float(config["same_floor_tolerance_m"]),
        )
    except Exception as exc:
        raise IdentityNativeError(f"native route-bank topology query failed: {exc}") from exc
    if not candidates:
        raise IdentityNativeError(
            "native route bank has no pair of unchanged paths ending at one common endpoint"
        )
    configured_target = config.get("target_position_m")
    configured_counts = config.get("motion_counts_by_actor")
    rng = np.random.default_rng(
        int(request.get("seed", 0)) + int(config.get("route_seed_offset", 910))
    )
    order = np.arange(len(candidates), dtype=int)
    rng.shuffle(order)
    for candidate_index in order:
        candidate = candidates[int(candidate_index)]
        target = np.asarray(candidate["target_m"], dtype=float)
        if (
            configured_target is not None
            and not np.allclose(target, np.asarray(configured_target, dtype=float), atol=1.0e-5, rtol=0.0)
        ):
            continue
        native_counts = candidate.get("native_frame_counts")
        if not isinstance(native_counts, Mapping):
            continue
        try:
            motion_counts = {
                actor_id: int(native_counts[actor_id])
                for actor_id in ("source1", "source2")
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if isinstance(configured_counts, Mapping):
            if any(
                int(configured_counts.get(actor_id, -1)) != motion_counts[actor_id]
                for actor_id in ("source1", "source2")
            ):
                continue
        if any(
            motion_start + motion_counts[actor_id] >= count
            for actor_id in ("source1", "source2")
        ):
            continue
        raw_event2_frame = motion_start + max(motion_counts.values()) + int(
            math.ceil(float(config["post_motion_silence_s"]) * fps - 1.0e-9)
        )
        event2_frame = int(
            math.ceil(raw_event2_frame / stride - 1.0e-9)
        ) * stride
        if event2_frame > latest_event2_start or event2_frame >= count:
            continue
        paths = candidate.get("paths")
        if not isinstance(paths, Mapping):
            continue
        tracks: dict[str, list[dict[str, Any]]] = {}
        route_records = deepcopy(dict(candidate.get("route_records") or {}))
        valid = True
        for actor in actors:
            actor_id = str(actor["actor_id"])
            path = np.asarray(paths.get(actor_id), dtype=float)
            if path.ndim != 2 or path.shape[1] != 3 or len(path) != motion_counts[actor_id]:
                valid = False
                break
            all_points = np.repeat(starts[actor_id][None, :], count, axis=0)
            all_points[motion_start:motion_start + len(path)] = path
            all_points[motion_start + len(path):] = path[-1]
            moving_end = motion_start + len(path) - 1
            tracks[actor_id] = _actor_states(
                actor, initial_states[actor_id], all_points,
                motion_start, moving_end, plan["clock"],
            )
            row = route_records.get(actor_id)
            if not isinstance(row, dict):
                valid = False
                break
            row["scheduled_frame_interval"] = [motion_start, moving_end]
            row["native_frame_count"] = len(path)
        if not valid:
            continue
        v0_tracks = {
            "source1": tracks["source1"],
            "source2": _actor_states(
                actors[1], initial_states["source2"],
                np.repeat(starts["source2"][None, :], count, axis=0),
                count, count, plan["clock"],
            ),
        }
        v1_tracks = {
            "source1": _actor_states(
                actors[0], initial_states["source1"],
                np.repeat(starts["source1"][None, :], count, axis=0),
                count, count, plan["clock"],
            ),
            "source2": tracks["source2"],
        }
        final_idle_rotation = deepcopy(
            v0_tracks["source1"][-1]["root_transform"]["rotation_xyzw"]
        )
        source2_motion_end = int(
            route_records["source2"]["scheduled_frame_interval"][1]
        )
        for state in v1_tracks["source2"]:
            if (
                int(state.get("frame_index", -1)) >= source2_motion_end
                and not bool(state.get("moving"))
            ):
                state["root_transform"]["rotation_xyzw"] = deepcopy(final_idle_rotation)
        for variant_tracks, static_actor, moving_actor in (
            (v0_tracks, "source2", "source1"),
            (v1_tracks, "source1", "source2"),
        ):
            moving_points = np.asarray(
                [row["planned_emitter_m"] for row in variant_tracks[moving_actor]],
                dtype=float,
            )
            static_points = np.asarray(
                [row["planned_emitter_m"] for row in variant_tracks[static_actor]],
                dtype=float,
            )
            if _min_separation(moving_points, static_points) < float(
                config["minimum_entity_separation_m"]
            ):
                valid = False
                break
            for actor_id in visible_actor_ids:
                for state in variant_tracks[actor_id]:
                    okay, _detail = _camera_visible(
                        camera, state["planned_emitter_m"], mesh,
                        _visibility_margin_for_actor(
                            plan, request, actor_id,
                            float(config["visibility_margin_deg"]),
                        ),
                    )
                    if not okay:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            moving_bearings = [
                _camera_visible(
                    camera,
                    variant_tracks[moving_actor][index]["planned_emitter_m"],
                    mesh, 0.0,
                )[1]["bearing_deg"]
                for index in (0, count - 1)
            ]
            if abs(
                ((moving_bearings[1] - moving_bearings[0] + 180.0) % 360.0) - 180.0
            ) < float(config["minimum_bearing_change_deg"]):
                valid = False
                break
        if not valid:
            continue
        return {
            "target_m": target.tolist(),
            "motion_start_frame": motion_start,
            "event2_frame": event2_frame,
            "event2_start_s": event2_frame / fps,
            "event2_start_frame": event2_frame,
            "motion_guard": {
                "basis": "first_dry_end_then_next_rir_key_plus_declared_gap",
                "first_dry_end_s": first_dry_end_s,
                "first_support_key_frame": first_support_key_frame,
                "declared_gap_s": float(config["minimum_silence_after_emission_s"]),
                "measured_wet_tail_end_s": None,
                "rir_stride": stride,
            },
            "motion_counts": [motion_counts["source1"], motion_counts["source2"]],
            "speeds_mps": [
                float(
                    route_records[actor_id].get(
                        "speed_mps",
                        float(route_records[actor_id]["path_length_m"])
                        * fps / max(motion_counts[actor_id] - 1, 1),
                    )
                )
                for actor_id in ("source1", "source2")
            ],
            "route_records": route_records,
            "native_frame_counts": motion_counts,
            "native_timing_preserved": True,
            "native_paths": {
                actor_id: np.asarray(paths[actor_id], dtype=float).tolist()
                for actor_id in ("source1", "source2")
            },
            "native_navigation_authority": deepcopy(space.metadata),
            "preflight": {
                "camera": "coarse_emitter_los_and_fov_pass",
                "minimum_entity_separation_m": float(config["minimum_entity_separation_m"]),
                "attempt": int(candidate_index),
                "native_route_bank": True,
            },
        }, {"v0": v0_tracks, "v1": v1_tracks}
    raise IdentityNativeError(
        "native route-bank paths failed timing, visibility or identity separation filters"
    )





def _validate_polyline_frame_speeds(
    sampled: np.ndarray, frame_rate_hz: float,
    speed_range_mps: Sequence[float], *, owner: str,
) -> dict[str, Any]:
    """Validate each adjacent sampled-frame displacement, including turns."""
    points = np.asarray(sampled, dtype=float)
    step_speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) * float(
        frame_rate_hz
    )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 2
        or len(step_speeds) != len(points) - 1
        or not np.all(np.isfinite(step_speeds))
        or np.min(step_speeds) < float(speed_range_mps[0]) - 1.0e-5
        or np.max(step_speeds) > float(speed_range_mps[1]) + 1.0e-5
    ):
        raise IdentityNativeError(
            f"{owner} has an adjacent-frame speed outside the declared range"
        )
    return {
        "sampling_policy": "arc_length_over_native_recast_polyline",
        "sampled_frame_count": int(len(points)),
        "native_timing_preserved": False,
        "minimum_step_speed_mps": float(np.min(step_speeds)),
        "maximum_step_speed_mps": float(np.max(step_speeds)),
        "speed_mps": float(np.mean(step_speeds)),
    }


def _sample_native_polyline(
    path: np.ndarray, frame_count: int, frame_rate_hz: float,
    speed_range_mps: Sequence[float], *, owner: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample a native polyline and validate every adjacent-frame speed."""
    try:
        sampled = resample_polyline_by_arc_length(
            path, int(frame_count), owner=owner,
        )
    except Exception as exc:
        raise IdentityNativeError(f"{owner} cannot be sampled: {exc}") from exc
    return sampled, _validate_polyline_frame_speeds(
        sampled, frame_rate_hz, speed_range_mps, owner=owner,
    )


def _select_raw_native_polyline_topology(
    plan: Mapping[str, Any], request: Mapping[str, Any],
    first_sound: Mapping[str, Any], second_sound: Mapping[str, Any],
    space: Any, mesh: Any, *, pool_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Schedule one native Recast polyline at a legal integer frame count."""
    config = _config(request)
    candidates = _raw_native_polyline_candidates(plan, request, pool_path=pool_path)
    camera = plan["visual_plan"]["camera"]
    actors = plan["visual_plan"]["actors"]
    visible_actor_ids = _camera_visibility_actor_ids(plan, request)
    frames = plan["visual_plan"]["frames"]
    initial_states = {
        str(state["actor_id"]): state for state in frames[0]["actor_states"]
    }
    starts = {
        actor_id: np.asarray(
            initial_states[actor_id]["root_transform"]["translation_m"],
            dtype=float,
        )
        for actor_id in ("source1", "source2")
    }
    fps = float(plan["clock"]["frame_rate_hz"])
    count = int(plan["clock"]["frame_count"])
    stride = int(request.get("rir_stride", 3))
    if stride < 1:
        raise IdentityNativeError("request.rir_stride must be positive")
    first_dry_end_s = float(config["event_start_s"]) + int(
        first_sound["sample_count"]
    ) / float(plan["clock"]["sample_rate_hz"])
    first_support_key_frame = int(
        math.ceil(first_dry_end_s * fps / stride - 1.0e-9)
    ) * stride
    motion_start = first_support_key_frame + int(
        math.ceil(float(config["minimum_silence_after_emission_s"]) * fps - 1.0e-9)
    )
    motion_start = max(1, motion_start)
    latest_event2_start = int(
        math.floor(
            (float(plan["clock"]["duration_seconds"])
             - float(request["profile"]["reserve_tail_s"])
             - int(second_sound["sample_count"])
             / int(second_sound["sample_rate_hz"]))
            * fps + 1.0e-9
        )
    )
    configured_target = config.get("target_position_m")
    configured_counts = config.get("motion_counts_by_actor")
    rng = np.random.default_rng(
        int(request.get("seed", 0)) + int(config.get("route_seed_offset", 910))
    )
    order = np.arange(len(candidates), dtype=int)
    rng.shuffle(order)
    for candidate_index in order:
        candidate = candidates[int(candidate_index)]
        target = np.asarray(candidate["target_m"], dtype=float)
        if (
            configured_target is not None
            and not np.allclose(
                target, np.asarray(configured_target, dtype=float),
                atol=1.0e-5, rtol=0.0,
            )
        ):
            continue
        raw_paths = candidate.get("paths")
        if not isinstance(raw_paths, Mapping):
            continue
        selected_counts: dict[str, int] = {}
        sampled_paths: dict[str, np.ndarray] = {}
        route_records = deepcopy(dict(candidate.get("route_records") or {}))
        valid = True
        for actor_id in ("source1", "source2"):
            row = route_records.get(actor_id)
            if not isinstance(row, dict):
                valid = False
                break
            path = np.asarray(raw_paths.get(actor_id), dtype=float)
            if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
                valid = False
                break
            options = _motion_count_options(
                float(row["path_length_m"]), config, fps
            )
            if not options:
                valid = False
                break
            if isinstance(configured_counts, Mapping):
                try:
                    frame_count = int(configured_counts[actor_id])
                except (KeyError, TypeError, ValueError, OverflowError):
                    valid = False
                    break
                if frame_count < 2 or frame_count - 1 not in options:
                    valid = False
                    break
            else:
                frame_count = int(options[int(rng.integers(len(options)))])
            if frame_count < 2 or motion_start + frame_count >= count:
                valid = False
                break
            try:
                sampled, speed_summary = _sample_native_polyline(
                    path, frame_count, fps, config["walk_speed_range_mps"],
                    owner=f"native Recast polyline {actor_id}",
                )
            except IdentityNativeError:
                valid = False
                break
            selected_counts[actor_id] = frame_count
            sampled_paths[actor_id] = sampled
            row["sampled_path_m"] = sampled.tolist()
            row.update(speed_summary)
        if not valid:
            continue
        raw_event2_frame = motion_start + max(selected_counts.values()) + int(
            math.ceil(float(config["post_motion_silence_s"]) * fps - 1.0e-9)
        )
        event2_frame = int(
            math.ceil(raw_event2_frame / stride - 1.0e-9)
        ) * stride
        if event2_frame > latest_event2_start or event2_frame >= count:
            continue
        tracks: dict[str, list[dict[str, Any]]] = {}
        for actor in actors:
            actor_id = str(actor["actor_id"])
            path = sampled_paths[actor_id]
            all_points = np.repeat(starts[actor_id][None, :], count, axis=0)
            all_points[motion_start:motion_start + len(path)] = path
            all_points[motion_start + len(path):] = path[-1]
            moving_end = motion_start + len(path) - 1
            tracks[actor_id] = _actor_states(
                actor, initial_states[actor_id], all_points,
                motion_start, moving_end, plan["clock"],
            )
            route_records[actor_id]["scheduled_frame_interval"] = [
                motion_start, moving_end,
            ]
        v0_tracks = {
            "source1": tracks["source1"],
            "source2": _actor_states(
                actors[1], initial_states["source2"],
                np.repeat(starts["source2"][None, :], count, axis=0),
                count, count, plan["clock"],
            ),
        }
        v1_tracks = {
            "source1": _actor_states(
                actors[0], initial_states["source1"],
                np.repeat(starts["source1"][None, :], count, axis=0),
                count, count, plan["clock"],
            ),
            "source2": tracks["source2"],
        }
        final_idle_rotation = deepcopy(
            v0_tracks["source1"][-1]["root_transform"]["rotation_xyzw"]
        )
        source2_motion_end = int(
            route_records["source2"]["scheduled_frame_interval"][1]
        )
        for state in v1_tracks["source2"]:
            if (
                int(state.get("frame_index", -1)) >= source2_motion_end
                and not bool(state.get("moving"))
            ):
                state["root_transform"]["rotation_xyzw"] = deepcopy(final_idle_rotation)
        for variant_tracks, static_actor, moving_actor in (
            (v0_tracks, "source2", "source1"),
            (v1_tracks, "source1", "source2"),
        ):
            moving_points = np.asarray(
                [row["planned_emitter_m"] for row in variant_tracks[moving_actor]],
                dtype=float,
            )
            static_points = np.asarray(
                [row["planned_emitter_m"] for row in variant_tracks[static_actor]],
                dtype=float,
            )
            if _min_separation(moving_points, static_points) < float(
                config["minimum_entity_separation_m"]
            ):
                valid = False
                break
            for actor_id in visible_actor_ids:
                for state in variant_tracks[actor_id]:
                    okay, _detail = _camera_visible(
                        camera, state["planned_emitter_m"], mesh,
                        _visibility_margin_for_actor(
                            plan, request, actor_id,
                            float(config["visibility_margin_deg"]),
                        ),
                    )
                    if not okay:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            moving_bearings = [
                _camera_visible(
                    camera,
                    variant_tracks[moving_actor][index]["planned_emitter_m"],
                    mesh, 0.0,
                )[1]["bearing_deg"]
                for index in (0, count - 1)
            ]
            if abs(
                ((moving_bearings[1] - moving_bearings[0] + 180.0) % 360.0) - 180.0
            ) < float(config["minimum_bearing_change_deg"]):
                valid = False
                break
        if not valid:
            continue
        query_path = str(candidate["native_polyline_source"])
        return {
            "target_m": target.tolist(),
            "motion_start_frame": motion_start,
            "event2_frame": event2_frame,
            "event2_start_s": event2_frame / fps,
            "event2_start_frame": event2_frame,
            "motion_guard": {
                "basis": "first_dry_end_then_next_rir_key_plus_declared_gap",
                "first_dry_end_s": first_dry_end_s,
                "first_support_key_frame": first_support_key_frame,
                "declared_gap_s": float(config["minimum_silence_after_emission_s"]),
                "measured_wet_tail_end_s": None,
                "rir_stride_frames": stride,
            },
            "motion_counts": [
                selected_counts["source1"], selected_counts["source2"],
            ],
            "speeds_mps": [
                float(route_records[actor_id]["speed_mps"])
                for actor_id in ("source1", "source2")
            ],
            "route_records": route_records,
            "native_polyline_source": query_path,
            "base_layout_navigation_authority": str(
                space.metadata.get("authority")
            ),
            "base_layout_navigation_metadata": deepcopy(space.metadata),
            "motion_query_authority": "native_spear_ue_recast_common_endpoint_query",
            "motion_query_source": query_path,
            "native_polyline_query_candidate_index": int(
                candidate.get("native_polyline_query_candidate_index", -1)
            ),
            "native_polyline_sampling": "arc_length_by_declared_speed_and_integer_frame_count",
            "native_timing_preserved": False,
            "native_navigation_authority": {
                **deepcopy(dict(space.metadata)),
                "query_source": query_path,
                "query_authority": "native_spear_ue_recast_common_endpoint_query",
            },
            "preflight": {
                "camera": "coarse_emitter_los_and_fov_pass",
                "minimum_entity_separation_m": float(config["minimum_entity_separation_m"]),
                "attempt": int(candidate_index),
                "native_polyline_query": True,
                "step_speed_checked": True,
            },
        }, {"v0": v0_tracks, "v1": v1_tracks}
    raise IdentityNativeError(
        "native Recast polylines failed fixed-FPS timing, visibility or identity filters"
    )

def _select_topology(plan: Mapping[str, Any], request: Mapping[str, Any], tail_end_s: float, first_sound: Mapping[str, Any], second_sound: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _config(request)
    space, mesh, _layout = load_planning_resources(plan["resources"], request)
    room_package = plan.get("resources", {}).get("room_package", {})
    walkable_space = room_package.get("walkable_space", {}) if isinstance(room_package, Mapping) else {}
    if _geometry_pool_specs(request):
        report = identity_geometry_pool_report(plan, request)
        if report["status"] != "compatible":
            raise IdentityGeometryQueryRequired(
                "declared identity geometry pools have no legal common endpoint "
                "for this plan own placement; a native geometry query at those "
                "exact starts is required",
                report=report,
            )
        return _select_raw_native_polyline_topology(
            plan, request,
            first_sound, second_sound, space, mesh,
            pool_path=str(report["selected_pool"]["resolved_path"]),
        )
    if isinstance(walkable_space, Mapping) and walkable_space.get("kind") == "route_bank":
        try:
            return _select_route_bank_topology(
                plan, request, first_sound, second_sound, space, mesh,
            )
        except IdentityNativeError as exc:
            report = identity_geometry_pool_report(plan, request)
            report["route_bank"] = _route_bank_identity_capability(plan, request, space)
            report["route_bank_rejection"] = str(exc)
            report["query_request"] = identity_geometry_query_request(
                plan, request,
                reason=(
                    "retained route bank has no common endpoint for the planner "
                    f"placement in this plan: {exc}"
                ),
                pools=report["pools"],
            )
            report["status"] = "requires_native_geometry_query"
            raise IdentityGeometryQueryRequired(
                "retained route bank cannot serve this identity placement; a "
                "native geometry query at the exact planner starts is required",
                report=report,
            ) from exc
    camera = plan["visual_plan"]["camera"]
    actors = plan["visual_plan"]["actors"]
    frames = plan["visual_plan"]["frames"]
    initial_states = {str(state["actor_id"]): state for state in frames[0]["actor_states"]}
    starts = {actor_id: np.asarray(initial_states[actor_id]["root_transform"]["translation_m"], dtype=float) for actor_id in ("source1", "source2")}
    floor = float(np.mean([starts["source1"][1], starts["source2"][1]]))
    fps = float(plan["clock"]["frame_rate_hz"])
    count = int(plan["clock"]["frame_count"])
    rir_stride = int(request.get("rir_stride", 3))
    if rir_stride < 1:
        raise IdentityNativeError("request.rir_stride must be positive")
    first_dry_end_s = float(config["event_start_s"]) + int(first_sound["sample_count"]) / float(plan["clock"]["sample_rate_hz"])
    first_support_key_frame = int(math.ceil(first_dry_end_s * fps / rir_stride - 1.0e-9)) * rir_stride
    motion_start = first_support_key_frame + int(math.ceil(float(config["minimum_silence_after_emission_s"]) * fps - 1.0e-9))
    motion_start = max(1, motion_start)
    latest_event2_start = int(math.floor((float(plan["clock"]["duration_seconds"]) - float(request["profile"]["reserve_tail_s"]) - int(second_sound["sample_count"]) / int(second_sound["sample_rate_hz"])) * fps + 1.0e-9))
    if motion_start >= latest_event2_start:
        raise IdentityNativeError(f"first emission guard leaves no motion window before event2: start={motion_start/fps:.3f}s latest_event2={latest_event2_start/fps:.3f}s")
    rng = np.random.default_rng(int(request.get("seed", 0)) + int(config.get("route_seed_offset", 910)))
    min_length, max_length = map(float, config["path_length_range_m"])
    min_sep = float(config["minimum_entity_separation_m"])
    floor_tol = float(config["same_floor_tolerance_m"])
    speed_lo, speed_hi = map(float, config["walk_speed_range_mps"])
    retry = int(config["route_retry_budget"])
    configured_target = config.get("target_position_m")
    for attempt in range(retry):
        if attempt == 0 and configured_target is not None:
            target = np.asarray(configured_target, dtype=float)
        else:
            try:
                target = np.asarray(space.sample_navigable(rng), dtype=float)
            except Exception:
                continue
        if abs(float(target[1]) - floor) > floor_tol:
            continue
        paths = []
        for actor_id in ("source1", "source2"):
            try:
                path = space.shortest_path(starts[actor_id], target)
            except Exception:
                path = None
            if path is None:
                break
            path = np.asarray(path, dtype=float)
            if len(path) < 2:
                break
            if np.max(np.abs(path[:, 1] - floor)) > floor_tol:
                break
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if not min_length <= length <= max_length:
                break
            if not all(space.is_navigable(point) for point in path):
                break
            paths.append(path)
        if len(paths) != 2:
            continue
        lengths = [
            float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            for path in paths
        ]
        overrides = config.get("motion_counts_by_actor")
        motion_counts = []
        speeds = []
        for actor_id, length in zip(("source1", "source2"), lengths, strict=True):
            options = _motion_count_options(length, config, fps)
            if not options:
                motion_counts = []
                break
            if isinstance(overrides, Mapping) and actor_id in overrides:
                try:
                    selected_count = int(overrides[actor_id])
                except (TypeError, ValueError, OverflowError):
                    motion_counts = []
                    break
                selected_steps = selected_count - 1
                if selected_steps not in options:
                    motion_counts = []
                    break
            else:
                selected_steps = int(rng.choice(options))
                selected_count = selected_steps + 1
            motion_counts.append(selected_count)
            speeds.append(length * fps / selected_steps)
        if len(motion_counts) != 2:
            continue
        motion_end = motion_start + max(motion_counts)
        event2_frame = motion_end + int(math.ceil(float(config["post_motion_silence_s"]) * fps - 1.0e-9))
        event2_frame = int(math.ceil(event2_frame / rir_stride - 1.0e-9)) * rir_stride
        if event2_frame > latest_event2_start or event2_frame >= count:
            continue
        tracks = {}
        route_records = {}
        for actor_index, actor in enumerate(actors):
            actor_id = str(actor["actor_id"])
            moving_path = resample_polyline_by_arc_length(paths[actor_index], motion_counts[actor_index])
            all_points = np.repeat(starts[actor_id][None, :], count, axis=0)
            all_points[motion_start : motion_start + len(moving_path)] = moving_path
            all_points[motion_start + len(moving_path) :] = moving_path[-1]
            moving_end = motion_start + len(moving_path) - 1
            tracks[actor_id] = _actor_states(actor, initial_states[actor_id], all_points, motion_start, moving_end, plan["clock"])
            route_records[actor_id] = {
                "pathfinder_polyline_m": paths[actor_index].tolist(),
                "path_length_m": float(np.linalg.norm(np.diff(paths[actor_index], axis=0), axis=1).sum()),
                "speed_mps": speeds[actor_index],
                "moving_frame_interval": [motion_start, moving_end],
                "all_centers_native_navigable": True,
                "authority": space.metadata.get("authority"),
            }
        v0_tracks = {
            "source1": tracks["source1"],
            "source2": _actor_states(actors[1], initial_states["source2"], np.repeat(starts["source2"][None, :], count, axis=0), count, count, plan["clock"]),
        }
        v1_tracks = {
            "source1": _actor_states(actors[0], initial_states["source1"], np.repeat(starts["source1"][None, :], count, axis=0), count, count, plan["clock"]),
            "source2": tracks["source2"],
        }
        # The controlled blue/green assets share the same rig and nominal
        # mouth anchor.  Their shortest paths approach the common target from
        # opposite directions, so the final idle yaw would otherwise move the
        # native mouth readback by a few centimetres.  Align the post-motion
        # idle pose with v0's moving actor; this preserves the persistent actor
        # IDs and route positions while making the audio-shared event endpoint
        # physically identical for the native RLR comparison.
        final_idle_rotation = deepcopy(v0_tracks["source1"][-1]["root_transform"]["rotation_xyzw"])
        source2_motion_end = int(route_records["source2"]["moving_frame_interval"][1])
        for state in v1_tracks["source2"]:
            if (int(state.get("frame_index", -1)) >= source2_motion_end
                    and not bool(state.get("moving"))):
                state["root_transform"]["rotation_xyzw"] = deepcopy(final_idle_rotation)
        valid = True
        for variant_tracks, static_actor, moving_actor in (
            (v0_tracks, "source2", "source1"),
            (v1_tracks, "source1", "source2"),
        ):
            moving_points = np.asarray([row["planned_emitter_m"] for row in variant_tracks[moving_actor]], dtype=float)
            static_points = np.asarray([row["planned_emitter_m"] for row in variant_tracks[static_actor]], dtype=float)
            if _min_separation(moving_points, static_points) < min_sep:
                valid = False
                break
            for actor_id in ("source1", "source2"):
                states = variant_tracks[actor_id]
                for state in states:
                    okay, _detail = _camera_visible(camera, state["planned_emitter_m"], mesh, float(config["visibility_margin_deg"]))
                    if not okay:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            moving_bearings = [_camera_visible(camera, variant_tracks[moving_actor][index]["planned_emitter_m"], mesh, 0.0)[1]["bearing_deg"] for index in (0, count - 1)]
            if abs(((moving_bearings[1] - moving_bearings[0] + 180.0) % 360.0) - 180.0) < float(config["minimum_bearing_change_deg"]):
                valid = False
                break
        if valid:
            return {
                "target_m": target.tolist(),
                "motion_start_frame": motion_start,
                "event2_frame": event2_frame,
                "event2_start_s": event2_frame / fps,
                "event2_start_frame": event2_frame,
                "motion_guard": {
                    "basis": "first_dry_end_then_next_rir_key_plus_declared_gap",
                    "first_dry_end_s": first_dry_end_s,
                    "first_support_key_frame": first_support_key_frame,
                    "declared_gap_s": float(config["minimum_silence_after_emission_s"]),
                    "measured_wet_tail_end_s": None if tail_end_s is None else float(tail_end_s),
                    "rir_stride_frames": rir_stride,
                },
                "motion_counts": motion_counts,
                "speeds_mps": speeds,
                "route_records": route_records,
                "native_navigation_authority": space.metadata,
                "preflight": {"camera": "coarse_emitter_los_and_fov_pass", "minimum_entity_separation_m": min_sep, "attempt": attempt},
            }, {"v0": v0_tracks, "v1": v1_tracks}
    raise IdentityNativeError("bounded native PathFinder sampling found no visible collision-separated L/P-to-R topology within the declared conditions")


def _ensure_audio_tools() -> None:
    """Load the task-local soundfile reader after native RLR finalization.

    The native helper installs an ``info``-only compatibility module when the
    Habitat prefix lacks python-soundfile.  The assembler needs the complete
    reader, which is retained inside AVEngine's task-local addon directory.
    """
    addon_path = (native.REPOSITORY / "tmp/native_python_addons_v1").resolve()
    if addon_path.is_dir() and str(addon_path) not in sys.path:
        sys.path.insert(0, str(addon_path))
    loaded = sys.modules.get("soundfile")
    if loaded is not None and not hasattr(loaded, "read"):
        sys.modules.pop("soundfile", None)
    try:
        import soundfile as sf
    except ImportError as exc:
        raise IdentityNativeError(
            "strict assembly requires the bundled soundfile reader"
        ) from exc
    if not hasattr(sf, "read"):
        raise IdentityNativeError("bundled soundfile reader lacks read()")


def _pcm_signature(path: str | Path) -> dict[str, Any]:
    """Read the raw WAV payload for PCM and IEEE-float delivery files."""
    value = Path(path).expanduser().resolve()
    fmt: tuple[int, int, int, int] | None = None
    payload: bytes | None = None
    with value.open("rb") as stream:
        if stream.read(4) != b"RIFF":
            raise IdentityNativeError(f"WAV is not RIFF: {value}")
        stream.read(4)
        if stream.read(4) != b"WAVE":
            raise IdentityNativeError(f"WAV is not WAVE: {value}")
        while True:
            chunk_id = stream.read(4)
            if not chunk_id:
                break
            if len(chunk_id) != 4:
                raise IdentityNativeError(f"WAV has truncated chunk header: {value}")
            size_bytes = stream.read(4)
            if len(size_bytes) != 4:
                raise IdentityNativeError(f"WAV has truncated chunk size: {value}")
            size = int.from_bytes(size_bytes, "little")
            chunk = stream.read(size)
            if len(chunk) != size:
                raise IdentityNativeError(f"WAV has truncated chunk payload: {value}")
            if chunk_id == b"fmt ":
                if size < 16:
                    raise IdentityNativeError(f"WAV fmt chunk is too short: {value}")
                fmt = (
                    int.from_bytes(chunk[0:2], "little"),
                    int.from_bytes(chunk[2:4], "little"),
                    int.from_bytes(chunk[4:8], "little"),
                    int.from_bytes(chunk[12:14], "little"),
                )
            elif chunk_id == b"data":
                payload = chunk
            if size % 2:
                stream.read(1)
    if fmt is None or payload is None:
        raise IdentityNativeError(f"WAV lacks fmt/data chunks: {value}")
    audio_format, channels, sample_rate, block_align = fmt
    if channels < 1 or sample_rate < 1 or block_align < 1:
        raise IdentityNativeError(f"WAV header is invalid: {value}")
    if audio_format not in (1, 3, 65534):
        raise IdentityNativeError(
            f"WAV format {audio_format} is unsupported for PCM equality: {value}"
        )
    if len(payload) % block_align:
        raise IdentityNativeError(f"WAV data is not frame aligned: {value}")
    return {
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "sample_width": block_align // channels,
        "frame_count": len(payload) // block_align,
        "payload": payload,
    }


def _pcm_equal(left: str | Path, right: str | Path) -> dict[str, Any]:
    a, b = _pcm_signature(left), _pcm_signature(right)
    same = all(a[key] == b[key] for key in ("channels", "sample_rate_hz", "sample_width", "frame_count", "payload"))
    return {"status": "pass" if same else "fail", "same": same, "left": str(Path(left).resolve()), "right": str(Path(right).resolve()), "channels": a["channels"], "sample_rate_hz": a["sample_rate_hz"], "frame_count": a["frame_count"]}


def _normalize_resume_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the paths expected by group assembly from a delivery result.

    Existing v0 members are read from qa_delivery ``result.json`` files,
    whose lossless WAV field is named ``lossless_stereo_wav``. Fresh members
    returned by ``finalize_audio_assignment`` already carry the shorter
    ``audio`` alias, so resume normalizes only the legacy shape here.
    """
    normalized = dict(result)
    if "audio" not in normalized:
        wav = normalized.get("lossless_stereo_wav")
        if not wav:
            raise IdentityNativeError(
                "resume delivery result lacks audio/lossless_stereo_wav"
            )
        normalized["audio"] = str(Path(wav).expanduser().resolve())
    return normalized


def _variant_targets(
    visual_id: str, assignment: str, *, first_event_actor_id: str = "source1",
) -> dict[str, str]:
    if first_event_actor_id not in {"source1", "source2"}:
        raise IdentityNativeError("first_event_actor_id must be source1 or source2")
    other_actor = "source2" if first_event_actor_id == "source1" else "source1"
    if visual_id == "v0" and assignment == "a0":
        return {"event_001": first_event_actor_id, "event_002": "source1"}
    if visual_id == "v0" and assignment == "a1":
        return {"event_001": other_actor, "event_002": "source1"}
    if visual_id == "v1" and assignment == "a0":
        return {"event_001": first_event_actor_id, "event_002": "source2"}
    if visual_id == "v1" and assignment == "a1":
        return {"event_001": other_actor, "event_002": "source2"}
    raise IdentityNativeError(f"unknown identity assignment {visual_id}/{assignment}")


def _group_spec(group_id: str, world_id: str, request: Mapping[str, Any], room_family: str, room_id: str, variants: Mapping[str, Mapping[str, Any]], questions: Mapping[str, Mapping[str, Any]], routes: Mapping[str, Any]) -> dict[str, Any]:
    members = []
    for member_id in ("v0_a0", "v0_a1", "v1_a0", "v1_a1"):
        member = variants[member_id]
        members.append({
            "member_id": member_id,
            "facts_path": str(Path(member["facts"]).resolve()),
            "video_path": str(Path(member["visual_video"]).resolve()),
            "audio_path": str(Path(member["audio"]).resolve()),
            "interventions": {
                "visual_variant": member_id.split("_", 1)[0],
                "audio_assignment": member_id.split("_", 1)[1],
                "persistent_actor_ids": ["source1", "source2"],
                "route_topology": routes,
            },
        })
    spec = {
        "schema": "avengine_binding_group_spec_v1",
        "request": deepcopy(dict(request)),
        "groups": [{
            "group_id": group_id,
            "world_id": world_id,
            "task_family": FAMILY,
            "room_family": room_family,
            "room_id": room_id,
            "split": "pilot",
            "request": deepcopy(dict(request)),
            "query": deepcopy(DEFAULT_QUERY),
            "angle_tolerance_deg": float(_config(request)["angle_tolerance_deg"]),
            "members": members,
            "comparisons": [
                {"members": ["v0_a0", "v0_a1"], "shared_modality": "video", "answer_relation": "different", "kind": "necessity"},
                {"members": ["v1_a0", "v1_a1"], "shared_modality": "video", "answer_relation": "different", "kind": "necessity"},
                {
                    "members": ["v0_a0", "v1_a0"],
                    "shared_modality": "audio",
                    "answer_relation": "different",
                    "kind": "necessity",
                    "allow_audio_reassignment": True,
                },
                {
                    "members": ["v0_a1", "v1_a1"],
                    "shared_modality": "audio",
                    "answer_relation": "different",
                    "kind": "necessity",
                    "allow_audio_reassignment": True,
                },
            ],
        }],
    }
    spec["groups"][0]["question_truth_by_member"] = {
        member_id: questions[member_id]["forms"]["open"]["truth"] for member_id in questions
    }
    return spec


def prepare_identity_group(*, base_episode_root: str | Path, request_path: str | Path, output_root: str | Path, group_id: str = "identity_hm3d_group_v1", world_id: str = "world_hm3d_identity_0001", graphics_adapter: int | None = None, rpc_port: int | None = None) -> dict[str, Any]:
    base_root = Path(base_episode_root).expanduser().resolve()
    request_file = Path(request_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise IdentityNativeError(f"refusing existing identity output root: {output}")
    base_plan_path = base_root / "plan/episode_plan.json"
    if not base_plan_path.is_file() or not (base_root / "plan").is_dir():
        raise IdentityNativeError("base episode root must contain plan/episode_plan.json and plan/")
    base_plan = _load(base_plan_path)
    request = _load(request_file)
    if graphics_adapter is not None:
        request.setdefault("runtime", {})["graphics_adapter"] = int(graphics_adapter)
    if rpc_port is not None:
        request.setdefault("runtime", {})["rpc_port"] = int(rpc_port)
    _validate_request(request, base_plan)
    registry, asset_index = _registry(request)
    if not isinstance(group_id, str) or not group_id.strip() or not isinstance(world_id, str) or not world_id.strip():
        raise IdentityNativeError("group_id and world_id must be nonempty")
    output.mkdir(parents=True)
    (output / "probe").mkdir()
    (output / "visual").mkdir()
    (output / "variants").mkdir()
    (output / "requests").mkdir()
    provenance = {
        "schema": "avengine_binding_group_identity_provenance_v1",
        "status": "running",
        "repository": str(native.REPOSITORY.resolve()),
        "base_episode_root": str(base_root),
        "base_plan": str(base_plan_path),
        "request": str(request_file),
        "group_id": group_id,
        "world_id": world_id,
        "task_family": FAMILY,
        "source_asset_ids": list(request["source_asset_ids"]),
        "runtime": {"graphics_adapter": request.get("runtime", {}).get("graphics_adapter"), "rpc_port": request.get("runtime", {}).get("rpc_port")},
        "claim_boundary": "research_only native media and controlled identity relations; no human/model/formal admission claim",
    }
    _write(output / "provenance_started.json", provenance)
    try:
        base = _base_identity_plan(base_plan, request, registry)
        base["visual_plan"]["frames"] = deepcopy(base_plan["visual_plan"]["frames"])
        for frame in base["visual_plan"]["frames"]:
            frame["camera_state"] = {**deepcopy(base["visual_plan"]["camera"]), "frame_index": int(frame["frame_index"])}
            for actor_index, actor in enumerate(base["visual_plan"]["actors"]):
                initial = base_plan["visual_plan"]["frames"][0]["actor_states"][actor_index]
                state = deepcopy(initial)
                state.update(actor_id=actor["actor_id"], frame_index=int(frame["frame_index"]), action_id=actor["timeline"]["idle_action_id"], action_phase=0.0, action_time_ticks=0, moving=False)
                point = np.asarray(initial["root_transform"]["translation_m"], dtype=float)
                quaternion, matrix = _rotation(2.0 * math.atan2(float(np.asarray(initial["root_transform"]["rotation_xyzw"])[1]), float(np.asarray(initial["root_transform"]["rotation_xyzw"])[3])))
                state["planned_emitter_m"] = (point + matrix @ np.asarray(actor["emitter_binding"]["emitter_offset_m"], dtype=float)).tolist()
                state["root_transform"]["rotation_xyzw"] = quaternion
                frame["actor_states"][actor_index] = state
        first_sound, second_sound, sound_selection = _select_sound_pair(
            request, request["source_asset_ids"], asset_index, plan=base
        )
        first_event_actor_id, anchor_selection = _first_event_actor(request)
        sound_selection["first_event_actor"] = anchor_selection
        clock = base["clock"]
        first_event = _event(
            first_sound, event_id="event_001", actor_id=first_event_actor_id,
            start_s=float(_config(request)["event_start_s"]), clock=clock,
        )
        probe_request = deepcopy(request)
        probe_request["episode_id"] = f"{group_id}_probe"
        probe_request["entities"] = {**dict(request["entities"]), "silent_count": 1}
        probe_plan = _with_audio(base, probe_request, [first_event])
        _write(output / "requests/probe_request.json", probe_request)
        probe_visual_root = _materialize_visual(base_root, output / "probe/visual", probe_plan, probe_request)
        probe_capture = _capture(probe_request, probe_visual_root, "probe")
        _write(output / "probe/capture.json", probe_capture)
        probe_result, tail_end_s = _probe_audio(probe_plan, probe_request, probe_capture, output / "probe")
        _write(output / "probe/audio.json", probe_result)
        routes, tracks = _select_topology(base, request, tail_end_s, first_sound, second_sound)
        event2 = _event(second_sound, event_id="event_002", actor_id="source1", start_s=float(routes["event2_start_s"]), clock=clock)
        events = [first_event, event2]
        plans = {}
        captures = {}
        for visual_id in ("v0", "v1"):
            variant_request = deepcopy(request)
            variant_request["episode_id"] = f"{group_id}_{visual_id}"
            variant_plan = _with_audio(base, variant_request, events)
            variant_tracks = tracks[visual_id]
            variant_plan["visual_plan"]["frames"] = _build_frames(variant_plan, variant_plan["visual_plan"]["actors"], variant_tracks)
            variant_plan["identity_intervention"] = {
                "variant": visual_id,
                "persistent_actor_ids": ["source1", "source2"],
                "topology": "source1_L_to_R" if visual_id == "v0" else "source2_P_to_R",
                "camera_motion": "static",
                "movement_after_measured_wet_tail_s": tail_end_s,
                "event2_start_s": routes["event2_start_s"],
            }
            plans[visual_id] = variant_plan
            _write(output / f"requests/{visual_id}_request.json", variant_request)
            visual_root = _materialize_visual(base_root, output / f"visual/{visual_id}", variant_plan, variant_request)
            captures[visual_id] = _capture(variant_request, visual_root, visual_id)
            _write(output / f"visual/{visual_id}_capture.json", captures[visual_id])
        variants: dict[str, dict[str, Any]] = {}
        questions: dict[str, dict[str, Any]] = {}
        pcm_checks: dict[str, dict[str, Any]] = {}
        for visual_id in ("v0", "v1"):
            endpoints = _endpoint_bindings(captures[visual_id], plans[visual_id])
            for assignment in ("a0", "a1"):
                member_id = f"{visual_id}_{assignment}"
                targets = _variant_targets(
                    visual_id, assignment,
                    first_event_actor_id=str(
                        plans[visual_id].get("audio_events", [{}])[0].get("actor_id") or "source1"
                    ),
                )
                params = _assignment_kwargs(assignment, targets, endpoints)
                assigned, rebound = native.build_audio_assignment_plan(
                    plans[visual_id], plans[visual_id]["request"], assignment,
                    **params,
                )
                variant_root = native.materialize_audio_variant(
                    captures[visual_id], output / "variants" / member_id, assigned, rebound, member_id=member_id
                )
                # Every member is rendered independently; no report reuse is passed here.
                variants[member_id] = native.finalize_audio_assignment(variant_root, rebound)
                facts = _load(variants[member_id]["facts"])
                questions[member_id] = generate_binding_question(facts, FAMILY, DEFAULT_QUERY, seed=f"{group_id}:{member_id}")
                _write(output / f"variants/{member_id}/binding_question.json", questions[member_id])
        for assignment in ("a0", "a1"):
            pcm_checks[assignment] = _pcm_equal(variants[f"v0_{assignment}"]["audio"], variants[f"v1_{assignment}"]["audio"])
            if not pcm_checks[assignment]["same"]:
                raise IdentityNativeError(f"independently rendered shared audio column {assignment} has different PCM")
        group = _group_spec(
            group_id,
            world_id,
            request,
            native.room_family_from_plan(plans["v0"]),
            str(plans["v0"]["scene"]["room_id"]),
            variants,
            questions,
            routes,
        )
        spec_path = _write(output / "group_spec.json", group)
        _ensure_audio_tools()
        assembled = assemble_binding_dataset(group, input_base=output, output=output / "assembled", seed=group_id, verify_media=True)
        summary = {
            **provenance,
            "status": "pass",
            "room_family": native.room_family_from_plan(plans["v0"]),
            "room_id": plans["v0"]["scene"]["room_id"],
            "sound_selection": sound_selection,
            "early_probe": {"capture": probe_capture, "audio": probe_result, "measured_wet_tail_end_s": tail_end_s},
            "routes": routes,
            "captured": captures,
            "variants": variants,
            "pcm_by_column": pcm_checks,
            "group_spec": str(spec_path),
            "assembled": str(output / "assembled/binding_groups.json"),
            "assembled_group_count": assembled["group_count"],
            "assembled_sample_count": assembled["sample_count"],
            "validation": assembled["validation"],
            "question_truth_by_member": {member_id: item["forms"]["open"]["truth"] for member_id, item in questions.items()},
            "claim_boundary": "research_only native media and cross-event physical identity relation; human/model/formal admission not run",
        }
        _write(output / "summary.json", summary)
        return summary
    except Exception as exc:
        try:
            _write(output / "failure.json", {**provenance, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
        raise


def _motion_settled_frame(plan: Mapping[str, Any]) -> int:
    frames = plan.get("visual_plan", {}).get("frames", [])
    moving = [
        int(state.get("frame_index", index))
        for index, frame in enumerate(frames)
        for state in frame.get("actor_states", [])
        if isinstance(state, Mapping) and bool(state.get("moving"))
    ]
    return max(moving, default=-1) + 1


def _align_event2_to_rir_boundary(
    plan: Mapping[str, Any], *, settled_frame: int, rir_stride_frames: int,
) -> dict[str, int | float]:
    """Move event_002 to the first RIR key after all actors settle."""
    clock = plan.get("clock")
    if not isinstance(clock, Mapping):
        raise IdentityNativeError("identity plan lacks a clock for event2 alignment")
    fps = float(clock["frame_rate_hz"])
    sr = int(clock["sample_rate_hz"])
    ticks_per_sample = int(clock["time_base_hz"]) // sr
    stride = int(rir_stride_frames)
    if stride < 1:
        raise IdentityNativeError("RIR stride must be positive for event2 alignment")
    key_frame = ((int(settled_frame) + stride - 1) // stride) * stride
    start_sample = int(math.ceil(key_frame * sr / fps - 1.0e-12))
    events = plan.get("audio_events")
    bindings = plan.get("voice_bindings")
    if not isinstance(events, list) or not isinstance(bindings, list):
        raise IdentityNativeError("identity plan lacks event/binding lists for event2 alignment")
    event_id = "event_002"
    source_event = next(
        (row for row in events if isinstance(row, Mapping) and row.get("event_id") == event_id),
        None,
    )
    if not isinstance(source_event, Mapping):
        raise IdentityNativeError("identity plan lacks event_002 for event2 alignment")
    old_start = int(source_event.get("start_sample", 0))
    duration = int(source_event["sample_count"])
    end_sample = start_sample + duration
    if end_sample > int(clock["sample_count"]):
        raise IdentityNativeError("settled event_002 does not fit the episode clock")
    old_planned = source_event.get("planned_audible_interval_samples")
    new_values = {
        "start_sample": start_sample,
        "end_sample": end_sample,
        "end_sample_exclusive": end_sample,
        "start_s": start_sample / sr,
        "end_s": end_sample / sr,
        "start_tick": start_sample * ticks_per_sample,
        "end_tick": end_sample * ticks_per_sample,
        "end_tick_exclusive": end_sample * ticks_per_sample,
    }
    if isinstance(old_planned, Sequence) and not isinstance(old_planned, (str, bytes)) and len(old_planned) == 2:
        new_values["planned_audible_interval_samples"] = [
            start_sample + int(old_planned[0]) - old_start,
            start_sample + int(old_planned[1]) - old_start,
        ]
    for collection in (events, bindings):
        matches = [
            row for row in collection
            if isinstance(row, Mapping) and row.get("event_id") == event_id
        ]
        if len(matches) != 1:
            raise IdentityNativeError("identity plan event_002 declarations are not unique")
        matches[0].update(new_values)
    return {
        "settled_frame": int(settled_frame),
        "rir_key_frame": int(key_frame),
        "start_sample": int(start_sample),
        "end_sample_exclusive": int(end_sample),
    }


def rerender_identity_group_audio(
    *, source_root: str | Path, v1_visual_root: str | Path,
    output_root: str | Path, group_id: str, world_id: str,
    source_context_policy: str = "independent_states",
) -> dict[str, Any]:
    """Re-render all four audio members from completed v0/v1 captures.

    This path is deliberately audio-only: it reuses the completed v0 capture
    and the corrected v1 capture, but renders every member afresh under an
    explicit independent-source context before strict PCM comparison.
    """
    if source_context_policy != "independent_states":
        raise IdentityNativeError(
            "identity audio rerender requires source_context_policy=independent_states"
        )
    source = Path(source_root).expanduser().resolve()
    v1_source = Path(v1_visual_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise IdentityNativeError(f"refusing existing identity audio rerender root: {output}")
    plans = {
        "v0": _load(source / "visual/v0/plan/episode_plan.json"),
        "v1": _load(v1_source / "visual/v1/plan/episode_plan.json"),
    }
    requests = {
        "v0": _load(source / "requests/v0_request.json"),
        "v1": _load(v1_source / "requests/v1_request.json"),
    }
    captures = {
        "v0": _load(source / "visual/v0_capture.json"),
        "v1": _load(v1_source / "visual/v1_capture.json"),
    }
    _validate_request(requests["v0"], plans["v0"])
    settled_frame = max(_motion_settled_frame(plans["v0"]), _motion_settled_frame(plans["v1"]))
    rir_stride_frames = max(
        int(requests[visual_id].get("rir_stride", 3))
        for visual_id in ("v0", "v1")
    )
    event2_alignment = {}
    for visual_id in ("v0", "v1"):
        capture_path = Path(captures[visual_id]["capture"]).expanduser().resolve()
        if not capture_path.is_dir():
            raise IdentityNativeError(
                f"identity audio rerender capture is unavailable: {capture_path}"
            )
        requests[visual_id]["source_context_policy"] = source_context_policy
        plans[visual_id]["request"] = deepcopy(requests[visual_id])
        event2_alignment[visual_id] = _align_event2_to_rir_boundary(
            plans[visual_id], settled_frame=settled_frame,
            rir_stride_frames=rir_stride_frames,
        )
        intervention = deepcopy(dict(plans[visual_id].get("identity_intervention") or {}))
        intervention["event2_start_s"] = float(
            event2_alignment[visual_id]["start_sample"]
        ) / float(plans[visual_id]["clock"]["sample_rate_hz"])
        plans[visual_id]["identity_intervention"] = intervention

    output.mkdir(parents=True)
    (output / "variants").mkdir()
    (output / "requests").mkdir()
    _write(output / "requests/v0_request.json", requests["v0"])
    _write(output / "requests/v1_request.json", requests["v1"])
    _write(output / "provenance_audio_rerender.json", {
        "schema": "avengine_binding_group_identity_audio_rerender_provenance_v1",
        "status": "running",
        "source_root": str(source),
        "v1_visual_root": str(v1_source),
        "group_id": group_id,
        "world_id": world_id,
        "task_family": FAMILY,
        "source_context_policy": source_context_policy,
        "reused_visual_captures": [
            str(Path(captures[visual_id]["capture"]).resolve())
            for visual_id in ("v0", "v1")
        ],
        "audio_render_policy": "all_four_members_independent_RLR",
        "event2_alignment": event2_alignment,
    })
    try:
        variants: dict[str, dict[str, Any]] = {}
        questions: dict[str, dict[str, Any]] = {}
        for visual_id in ("v0", "v1"):
            endpoints = _endpoint_bindings(captures[visual_id], plans[visual_id])
            for assignment in ("a0", "a1"):
                member_id = f"{visual_id}_{assignment}"
                plan = deepcopy(plans[visual_id])
                first_event_actor_id = str(
                    plan.get("audio_events", [{}])[0].get("actor_id") or "source1"
                )
                assigned, rebound = native.build_audio_assignment_plan(
                    plan, requests[visual_id], assignment,
                    **_assignment_kwargs(
                        assignment,
                        _variant_targets(
                            visual_id, assignment,
                            first_event_actor_id=first_event_actor_id,
                        ),
                        endpoints,
                    ),
                )
                variant_root = native.materialize_audio_variant(
                    captures[visual_id], output / "variants" / member_id,
                    assigned, rebound, member_id=member_id,
                )
                variants[member_id] = native.finalize_audio_assignment(
                    variant_root, rebound
                )
                facts = _load(variants[member_id]["facts"])
                questions[member_id] = generate_binding_question(
                    facts, FAMILY, DEFAULT_QUERY, seed=f"{group_id}:{member_id}"
                )
                _write(
                    output / f"variants/{member_id}/binding_question.json",
                    questions[member_id],
                )
        pcm_by_column = {}
        for assignment in ("a0", "a1"):
            pcm_by_column[assignment] = _pcm_equal(
                variants[f"v0_{assignment}"]["audio"],
                variants[f"v1_{assignment}"]["audio"],
            )
            if not pcm_by_column[assignment]["same"]:
                raise IdentityNativeError(
                    f"independent-source audio column {assignment} differs in actual PCM"
                )
        routes = {
            "source": "completed_v0_v1_capture_audio_rerender",
            "source_root": str(source),
            "v1_visual_root": str(v1_source),
            "source_context_policy": source_context_policy,
            "event2_alignment": event2_alignment,
            "visual_variants": {
                visual_id: deepcopy(plans[visual_id].get("identity_intervention"))
                for visual_id in ("v0", "v1")
            },
        }
        spec = _group_spec(
            group_id, world_id, requests["v0"],
            native.room_family_from_plan(plans["v0"]),
            str(plans["v0"]["scene"]["room_id"]),
            variants, questions, routes,
        )
        spec_path = _write(output / "group_spec.json", spec)
        _ensure_audio_tools()
        assembled = assemble_binding_dataset(
            spec, input_base=output, output=output / "assembled",
            seed=group_id, verify_media=True,
        )
        summary = {
            "schema": "avengine_binding_group_identity_summary_v1",
            "status": "pass",
            "group_id": group_id,
            "world_id": world_id,
            "task_family": FAMILY,
            "source_root": str(source),
            "v1_visual_root": str(v1_source),
            "source_context_policy": source_context_policy,
            "event2_alignment": event2_alignment,
            "group_spec": str(spec_path),
            "assembled": str(output / "assembled/binding_groups.json"),
            "assembled_group_count": assembled["group_count"],
            "assembled_sample_count": assembled["sample_count"],
            "validation": assembled["validation"],
            "variants": variants,
            "pcm_by_column": pcm_by_column,
            "question_truth_by_member": {
                member_id: item["forms"]["open"]["truth"]
                for member_id, item in questions.items()
            },
            "claim_boundary": "research_only native media and cross-event physical identity relation; human/model/formal admission not run",
        }
        _write(output / "summary.json", summary)
        return summary
    except Exception as exc:
        try:
            _write(output / "failure.json", {
                "schema": "avengine_binding_group_identity_audio_rerender_provenance_v1",
                "status": "fail", "source_root": str(source),
                "v1_visual_root": str(v1_source), "group_id": group_id,
                "world_id": world_id, "source_context_policy": source_context_policy,
                "error": f"{type(exc).__name__}: {exc}",
            })
        except Exception:
            pass
        raise


def repair_identity_group_v1(*, source_root: str | Path, output_root: str | Path, group_id: str, world_id: str) -> dict[str, Any]:
    """Re-capture only v1 after correcting its pre-motion idle pose.

    The source root must contain a completed identity run with valid v0
    capture/audio.  Its v0 artifacts are read-only; only v1 visual capture,
    v1 audio assignments, and a fresh assembled group are produced here.
    """
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise IdentityNativeError(f"refusing existing identity v1 repair root: {output}")
    v0_plan = _load(source / "visual/v0/plan/episode_plan.json")
    v1_plan = _load(source / "visual/v1/plan/episode_plan.json")
    v0_request = _load(source / "requests/v0_request.json")
    v1_request = _load(source / "requests/v1_request.json")
    _validate_request(v0_request, v0_plan)
    v0_capture = _load(source / "visual/v0_capture.json")
    if not Path(v0_capture["capture"]).expanduser().is_dir():
        raise IdentityNativeError("v1 repair requires the completed v0 native capture")
    v0_results = {
        member_id: _normalize_resume_result(
            _load(source / f"variants/{member_id}/delivery/result.json")
        )
        for member_id in ("v0_a0", "v0_a1")
    }

    def _actor_state(frame: Mapping[str, Any], actor_id: str) -> dict[str, Any]:
        for state in frame.get("actor_states", []):
            if isinstance(state, Mapping) and state.get("actor_id") == actor_id:
                return dict(state)
        raise IdentityNativeError(f"identity v1 repair frame lacks actor {actor_id}")

    v0_frames = v0_plan["visual_plan"]["frames"]
    v1_frames = v1_plan["visual_plan"]["frames"]
    source2_states = [_actor_state(frame, "source2") for frame in v1_frames]
    moving_frames = [
        int(state.get("frame_index", index))
        for index, state in enumerate(source2_states)
        if bool(state.get("moving"))
    ]
    if not moving_frames:
        raise IdentityNativeError("v1 repair plan has no source2 motion interval")
    first_moving = min(moving_frames)
    motion_end = max(moving_frames) + 1
    final_rotation = deepcopy(_actor_state(v0_frames[-1], "source1")["root_transform"]["rotation_xyzw"])
    # Restore only the pre-motion source2 idle states from the unchanged v0
    # source2 track; retain v1's legal walk path and align settled idle yaw.
    for index, frame in enumerate(v1_frames):
        state = _actor_state(frame, "source2")
        if index < first_moving:
            restored = _actor_state(v0_frames[index], "source2")
            state["root_transform"]["rotation_xyzw"] = deepcopy(
                restored["root_transform"]["rotation_xyzw"]
            )
        elif index >= motion_end and not bool(state.get("moving")):
            state["root_transform"]["rotation_xyzw"] = deepcopy(final_rotation)
        for candidate in frame.get("actor_states", []):
            if isinstance(candidate, dict) and candidate.get("actor_id") == "source2":
                candidate.clear()
                candidate.update(state)
                break
    v1_plan["request"] = deepcopy(v1_request)
    intervention = deepcopy(dict(v1_plan.get("identity_intervention") or {}))
    intervention["pose_repair"] = {
        "reference": "v0/source1_final_idle_pose",
        "pre_motion_source2_rotation": "restored_from_v0_source2",
        "first_moving_frame": first_moving,
        "post_motion_idle_start_frame": motion_end,
        "source2_idle_action_phase": 0.0,
    }
    v1_plan["identity_intervention"] = intervention

    output.mkdir(parents=True)
    (output / "visual").mkdir()
    (output / "variants").mkdir()
    (output / "requests").mkdir()
    _write(output / "provenance_repair.json", {
        "schema": "avengine_binding_group_identity_v1_repair_provenance_v1",
        "status": "running",
        "source_root": str(source),
        "group_id": group_id,
        "world_id": world_id,
        "task_family": FAMILY,
        "reused_v0_capture": str(Path(v0_capture["capture"]).resolve()),
        "reused_v0_audio_members": [str(source / f"variants/{member}/delivery/result.json") for member in ("v0_a0", "v0_a1")],
        "repair": intervention["pose_repair"],
    })
    try:
        _write(output / "requests/v0_request.json", v0_request)
        _write(output / "requests/v1_request.json", v1_request)
        visual_root = _materialize_visual(
            source / "visual/v1", output / "visual/v1", v1_plan, v1_request
        )
        v1_capture = _capture(v1_request, visual_root, "v1_repair")
        _write(output / "visual/v1_capture.json", v1_capture)
        variants: dict[str, dict[str, Any]] = dict(v0_results)
        questions: dict[str, dict[str, Any]] = {}
        for member_id in ("v0_a0", "v0_a1"):
            facts = _load(variants[member_id]["facts"])
            questions[member_id] = generate_binding_question(
                facts, FAMILY, DEFAULT_QUERY, seed=f"{group_id}:{member_id}"
            )
        endpoints = _endpoint_bindings(v1_capture, v1_plan)
        for assignment in ("a0", "a1"):
            member_id = f"v1_{assignment}"
            targets = _variant_targets(
                "v1", assignment,
                first_event_actor_id=str(
                    v1_plan.get("audio_events", [{}])[0].get("actor_id") or "source1"
                ),
            )
            assigned, rebound = native.build_audio_assignment_plan(
                v1_plan, v1_plan["request"], assignment,
                **_assignment_kwargs(assignment, targets, endpoints),
            )
            variant_root = native.materialize_audio_variant(
                v1_capture, output / "variants" / member_id,
                assigned, rebound, member_id=member_id,
            )
            variants[member_id] = native.finalize_audio_assignment(variant_root, rebound)
            facts = _load(variants[member_id]["facts"])
            questions[member_id] = generate_binding_question(
                facts, FAMILY, DEFAULT_QUERY, seed=f"{group_id}:{member_id}"
            )
            _write(output / f"variants/{member_id}/binding_question.json", questions[member_id])
        pcm_by_column = {}
        for assignment in ("a0", "a1"):
            pcm_by_column[assignment] = _pcm_equal(
                variants[f"v0_{assignment}"]["audio"],
                variants[f"v1_{assignment}"]["audio"],
            )
            if not pcm_by_column[assignment]["same"]:
                raise IdentityNativeError(
                    f"repaired shared audio column {assignment} differs in actual PCM"
                )
        routes = {
            "source": "v1_pose_repair_from_completed_group",
            "source_root": str(source),
            "visual_variants": {
                visual_id: deepcopy(plans.get("identity_intervention"))
                for visual_id, plans in (("v0", v0_plan), ("v1", v1_plan))
            },
        }
        spec = _group_spec(
            group_id, world_id, v0_request,
            native.room_family_from_plan(v0_plan),
            str(v0_plan["scene"]["room_id"]),
            variants, questions, routes,
        )
        spec_path = _write(output / "group_spec.json", spec)
        _ensure_audio_tools()
        assembled = assemble_binding_dataset(
            spec, input_base=output, output=output / "assembled",
            seed=group_id, verify_media=True,
        )
        summary = {
            "schema": "avengine_binding_group_identity_summary_v1",
            "status": "pass",
            "group_id": group_id,
            "world_id": world_id,
            "task_family": FAMILY,
            "source_root": str(source),
            "group_spec": str(spec_path),
            "assembled": str(output / "assembled/binding_groups.json"),
            "assembled_group_count": assembled["group_count"],
            "assembled_sample_count": assembled["sample_count"],
            "validation": assembled["validation"],
            "variants": variants,
            "pcm_by_column": pcm_by_column,
            "question_truth_by_member": {
                member_id: item["forms"]["open"]["truth"]
                for member_id, item in questions.items()
            },
            "repair": intervention["pose_repair"],
            "reused_v0_capture": str(Path(v0_capture["capture"]).resolve()),
            "claim_boundary": "research_only native media and cross-event physical identity relation; human/model/formal admission not run",
        }
        _write(output / "summary.json", summary)
        return summary
    except Exception as exc:
        try:
            _write(output / "failure.json", {
                "schema": "avengine_binding_group_identity_v1_repair_provenance_v1",
                "status": "fail", "source_root": str(source),
                "group_id": group_id, "world_id": world_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
        except Exception:
            pass
        raise



def _verify_selected_identity_plans(
    request: Mapping[str, Any], plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify a CPU-selected v0/v1 pair before native continuation."""
    if set(plans) != {"v0", "v1"}:
        raise IdentityNativeError("selected identity continuation requires v0 and v1 plans")
    config = _config(request)
    first_actor = config.get("first_event_actor_id")
    if first_actor not in {"source1", "source2"}:
        raise IdentityNativeError(
            "selected identity continuation requires an explicit first_event_actor_id"
        )
    common_request_keys = (
        "room_id", "source_asset_ids", "frame_count", "frame_rate_hz",
        "sample_rate_hz", "source_context_policy",
    )
    for visual_id, plan in plans.items():
        plan_request = plan.get("request")
        if not isinstance(plan_request, Mapping):
            raise IdentityNativeError(f"selected {visual_id} plan lacks its source request")
        for key in common_request_keys:
            if plan_request.get(key) != request.get(key):
                raise IdentityNativeError(
                    f"selected {visual_id} plan differs from continuation request: {key}"
                )
        plan_config = _config(plan_request)
        for key, value in config.items():
            if plan_config.get(key) != value:
                raise IdentityNativeError(
                    f"selected {visual_id} plan differs from continuation binding_identity: {key}"
                )
    v0, v1 = plans["v0"], plans["v1"]
    if native.room_family_from_plan(v0) != native.room_family_from_plan(v1):
        raise IdentityNativeError("selected identity plans use different room families")
    if v0.get("visual_plan", {}).get("camera") != v1.get("visual_plan", {}).get("camera"):
        raise IdentityNativeError("selected identity plans do not share one camera")
    if v0.get("clock") != v1.get("clock"):
        raise IdentityNativeError("selected identity plans do not share one clock")
    target = config.get("target_position_m")
    counts = config.get("motion_counts_by_actor")
    if not isinstance(target, Sequence) or isinstance(target, (str, bytes)) or len(target) != 3:
        raise IdentityNativeError("selected identity continuation requires target_position_m")
    if not isinstance(counts, Mapping):
        raise IdentityNativeError("selected identity continuation requires motion_counts_by_actor")
    frame_rate = float(v0["clock"]["frame_rate_hz"])
    event_rows: dict[str, Mapping[str, Any]] = {}
    for visual_id, plan in plans.items():
        events = plan.get("audio_events")
        if not isinstance(events, list) or len(events) != 2:
            raise IdentityNativeError(f"selected {visual_id} plan must contain two audio events")
        by_id = {
            str(row.get("event_id")): row
            for row in events if isinstance(row, Mapping) and row.get("event_id")
        }
        if set(by_id) != {"event_001", "event_002"}:
            raise IdentityNativeError(f"selected {visual_id} plan has invalid event IDs")
        first = by_id["event_001"]
        if str(first.get("actor_id")) != str(first_actor):
            raise IdentityNativeError(
                f"selected {visual_id} event_001 actor does not match first_event_actor_id"
            )
        event_rows[visual_id] = by_id["event_002"]
        if visual_id == "v0":
            continue
        if (
            by_id["event_002"].get("start_sample")
            != event_rows["v0"].get("start_sample")
            or by_id["event_002"].get("sample_count")
            != event_rows["v0"].get("sample_count")
        ):
            raise IdentityNativeError("selected v0/v1 plans differ in event_002 timing or clip")
    fps = float(v0["clock"]["frame_rate_hz"])
    stride = int(request.get("rir_stride", 3))
    post_frames = int(math.ceil(float(config["post_motion_silence_s"]) * fps - 1.0e-9))
    endpoint_frames: dict[str, int] = {}
    route_records: dict[str, Any] = {}
    for visual_id, moving_actor in (("v0", "source1"), ("v1", "source2")):
        frames = plans[visual_id].get("visual_plan", {}).get("frames")
        if not isinstance(frames, list):
            raise IdentityNativeError(f"selected {visual_id} plan lacks visual frames")
        states = [
            frame.get("actor_states", [])[0]
            for frame in frames
            if isinstance(frame, Mapping) and isinstance(frame.get("actor_states"), list)
        ]
        actor_states = {
            str(state.get("actor_id")): state
            for frame in frames
            if isinstance(frame, Mapping)
            for state in frame.get("actor_states", [])
            if isinstance(state, Mapping) and state.get("actor_id")
        }
        ordered_states = [
            next(
                state for state in frame.get("actor_states", [])
                if isinstance(state, Mapping) and state.get("actor_id") == moving_actor
            )
            for frame in frames
            if isinstance(frame, Mapping)
        ]
        moving_indices = [
            int(state.get("frame_index", index))
            for index, state in enumerate(ordered_states)
            if bool(state.get("moving"))
        ]
        if not moving_indices:
            raise IdentityNativeError(f"selected {visual_id} has no moving {moving_actor} states")
        endpoint_frame = max(moving_indices) + 1
        if endpoint_frame >= len(ordered_states):
            raise IdentityNativeError(f"selected {visual_id} movement has no settled endpoint")
        endpoint = ordered_states[endpoint_frame].get("root_transform", {}).get("translation_m")
        if (
            not isinstance(endpoint, Sequence)
            or isinstance(endpoint, (str, bytes))
            or len(endpoint) != 3
            or not np.allclose(np.asarray(endpoint, dtype=float), np.asarray(target, dtype=float), atol=1.0e-5, rtol=0.0)
        ):
            raise IdentityNativeError(
                f"selected {visual_id} settled endpoint differs from target_position_m"
            )
        selected_count = int(counts.get(moving_actor, -1))
        path_point_count = endpoint_frame - min(moving_indices) + 1
        if selected_count != path_point_count:
            raise IdentityNativeError(
                f"selected {visual_id} motion count differs for {moving_actor}: {selected_count} != {path_point_count}"
            )
        endpoint_frames[visual_id] = endpoint_frame
        route_records[moving_actor] = {
            "moving_frame_interval": [min(moving_indices), endpoint_frame],
            "native_motion_point_count": path_point_count,
            "settled_endpoint_m": [float(value) for value in endpoint],
        }
    event2_frame = int(round(float(event_rows["v0"]["start_sample"]) / int(request["sample_rate_hz"]) * fps))
    minimum_event2_frame = max(endpoint_frames.values()) + post_frames
    if event2_frame < minimum_event2_frame:
        raise IdentityNativeError(
            f"selected event_002 precedes settled motion: {event2_frame} < {minimum_event2_frame}"
        )
    if stride < 1 or event2_frame % stride:
        raise IdentityNativeError("selected event_002 does not start on an RIR keyframe")
    authority = _navigation_authority(v0)
    motion_query_authority = None
    motion_query_sources: list[str] = []
    for plan in plans.values():
        intervention = plan.get("identity_intervention")
        if isinstance(intervention, Mapping) and intervention.get("native_polyline_source"):
            motion_query_authority = str(
                intervention.get("motion_query_authority")
                or "native_spear_ue_recast_common_endpoint_query"
            )
            source = str(intervention["native_polyline_source"])
            if source not in motion_query_sources:
                motion_query_sources.append(source)
        plan_conditions = plan.get("planned_conditions")
        if isinstance(plan_conditions, dict):
            plan_conditions["authority"] = f"{authority}_and_explicit_identity_intervention"
        activity = plan.get("activity_plan")
        if isinstance(activity, dict):
            activity["authority"] = authority
    return {
        "navigation_authority": authority,
        "base_layout_navigation_authority": authority,
        "motion_query_authority": motion_query_authority,
        "motion_query_sources": motion_query_sources,
        "first_event_actor_id": str(first_actor),
        "target_position_m": [float(value) for value in target],
        "motion_counts_by_actor": {str(key): int(value) for key, value in counts.items()},
        "event2_start_frame": event2_frame,
        "event2_start_sample": int(event_rows["v0"]["start_sample"]),
        "settled_endpoint_frames": endpoint_frames,
        "route_records": route_records,
    }


def continue_identity_group_from_plans(
    *, layout_root: str | Path, request_path: str | Path,
    output_root: str | Path, group_id: str, world_id: str,
    graphics_adapter: int | None = None, rpc_port: int | None = None,
) -> dict[str, Any]:
    """Capture and render a previously selected CPU identity pair.

    The selected v0/v1 plans are treated as authoritative geometry. This
    continuation skips planning and the obsolete wet-tail probe, then captures
    both visuals and renders all four audio assignments independently.
    """
    layouts = Path(layout_root).expanduser().resolve()
    request_file = Path(request_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise IdentityNativeError(f"refusing existing identity continuation root: {output}")
    request = _load(request_file)
    if graphics_adapter is not None:
        request.setdefault("runtime", {})["graphics_adapter"] = int(graphics_adapter)
    if rpc_port is not None:
        request.setdefault("runtime", {})["rpc_port"] = int(rpc_port)
    plans = {
        visual_id: _load(layouts / visual_id / "plan/episode_plan.json")
        for visual_id in ("v0", "v1")
    }
    _validate_request(request, plans["v0"])
    selected = _verify_selected_identity_plans(request, plans)
    output.mkdir(parents=True)
    (output / "visual").mkdir()
    (output / "variants").mkdir()
    (output / "requests").mkdir()
    provenance = {
        "schema": "avengine_binding_group_identity_continuation_provenance_v1",
        "status": "running",
        "repository": str(native.REPOSITORY.resolve()),
        "selected_layout_root": str(layouts),
        "request": str(request_file),
        "group_id": group_id,
        "world_id": world_id,
        "task_family": FAMILY,
        "room_family": native.room_family_from_plan(plans["v0"]),
        "source_context_policy": request.get("source_context_policy"),
        "selected_geometry": selected,
        "claim_boundary": "research_only native media and controlled identity relations; no human/model/formal admission claim",
    }
    _write(output / "provenance_continuation.json", provenance)
    try:
        requests: dict[str, dict[str, Any]] = {}
        captures: dict[str, dict[str, Any]] = {}
        for visual_id in ("v0", "v1"):
            variant_request = deepcopy(request)
            variant_request["episode_id"] = f"{group_id}_{visual_id}"
            requests[visual_id] = variant_request
            plan = plans[visual_id]
            plan["request"] = deepcopy(variant_request)
            _write(output / f"requests/{visual_id}_request.json", variant_request)
            visual_root = _materialize_visual(
                layouts / visual_id, output / f"visual/{visual_id}", plan, variant_request
            )
            captures[visual_id] = _capture(variant_request, visual_root, visual_id)
            _write(output / f"visual/{visual_id}_capture.json", captures[visual_id])
        variants: dict[str, dict[str, Any]] = {}
        questions: dict[str, dict[str, Any]] = {}
        for visual_id in ("v0", "v1"):
            endpoints = _endpoint_bindings(captures[visual_id], plans[visual_id])
            for assignment in ("a0", "a1"):
                member_id = f"{visual_id}_{assignment}"
                targets = _variant_targets(
                    visual_id, assignment,
                    first_event_actor_id=str(selected["first_event_actor_id"]),
                )
                assigned, rebound = native.build_audio_assignment_plan(
                    plans[visual_id], requests[visual_id], assignment,
                    **_assignment_kwargs(assignment, targets, endpoints),
                )
                variant_root = native.materialize_audio_variant(
                    captures[visual_id], output / "variants" / member_id,
                    assigned, rebound, member_id=member_id,
                )
                variants[member_id] = native.finalize_audio_assignment(variant_root, rebound)
                facts = _load(variants[member_id]["facts"])
                questions[member_id] = generate_binding_question(
                    facts, FAMILY, DEFAULT_QUERY, seed=f"{group_id}:{member_id}"
                )
                _write(output / f"variants/{member_id}/binding_question.json", questions[member_id])
        pcm_by_column: dict[str, dict[str, Any]] = {}
        for assignment in ("a0", "a1"):
            pcm_by_column[assignment] = _pcm_equal(
                variants[f"v0_{assignment}"]["audio"],
                variants[f"v1_{assignment}"]["audio"],
            )
            if not pcm_by_column[assignment]["same"]:
                raise IdentityNativeError(
                    f"continued identity audio column {assignment} differs in actual PCM"
                )
        routes = {
            "source": "cpu_selected_identity_layouts",
            "navigation_authority": selected["navigation_authority"],
            "selected_geometry": selected,
            "visual_variants": {
                visual_id: deepcopy(plans[visual_id].get("identity_intervention"))
                for visual_id in ("v0", "v1")
            },
        }
        spec = _group_spec(
            group_id, world_id, request,
            native.room_family_from_plan(plans["v0"]),
            str(plans["v0"]["scene"]["room_id"]),
            variants, questions, routes,
        )
        spec_path = _write(output / "group_spec.json", spec)
        _ensure_audio_tools()
        assembled = assemble_binding_dataset(
            spec, input_base=output, output=output / "assembled", seed=group_id,
            verify_media=True,
        )
        summary = {
            "schema": "avengine_binding_group_identity_summary_v1",
            "status": "pass",
            "group_id": group_id,
            "world_id": world_id,
            "task_family": FAMILY,
            "source_root": str(layouts),
            "request": str(request_file),
            "room_family": native.room_family_from_plan(plans["v0"]),
            "source_context_policy": request.get("source_context_policy"),
            "selected_geometry": selected,
            "group_spec": str(spec_path),
            "assembled": str(output / "assembled/binding_groups.json"),
            "assembled_group_count": assembled["group_count"],
            "assembled_sample_count": assembled["sample_count"],
            "validation": assembled["validation"],
            "captured": captures,
            "variants": variants,
            "pcm_by_column": pcm_by_column,
            "question_truth_by_member": {
                member_id: item["forms"]["open"]["truth"]
                for member_id, item in questions.items()
            },
            "claim_boundary": "research_only native media and cross-event physical identity relation; human/model/formal admission not run",
        }
        _write(output / "summary.json", summary)
        return summary
    except Exception as exc:
        try:
            _write(output / "failure.json", {
                **provenance, "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            })
        except Exception:
            pass
        raise

def resume_identity_group(*, source_root: str | Path, output_root: str | Path, group_id: str, world_id: str) -> dict[str, Any]:
    """Finish audio/assembly from completed identity visual captures.

    Completed member outputs and both native visual captures stay read-only.
    Only missing audio members are independently rendered in a fresh root.
    """
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise IdentityNativeError(f"refusing existing identity resume root: {output}")
    request = _load(source / "requests/v0_request.json")
    plans = {
        visual_id: _load(source / f"visual/{visual_id}/plan/episode_plan.json")
        for visual_id in ("v0", "v1")
    }
    captures = {
        visual_id: _load(source / f"visual/{visual_id}_capture.json")
        for visual_id in ("v0", "v1")
    }
    _validate_request(request, plans["v0"])
    output.mkdir(parents=True)
    (output / "variants").mkdir()
    (output / "requests").mkdir()
    (output / "visual").mkdir()
    for visual_id, plan in plans.items():
        (output / "visual" / visual_id / "plan").mkdir(parents=True)
        _write(output / f"visual/{visual_id}/plan/episode_plan.json", plan)
        _write(output / f"visual/{visual_id}_capture.json", captures[visual_id])
        _write(output / f"requests/{visual_id}_request.json", plan["request"])
    reused_members: dict[str, str] = {}
    _write(output / "provenance_resume.json", {
        "schema": "avengine_binding_group_identity_resume_provenance_v1",
        "status": "running",
        "source_root": str(source),
        "group_id": group_id,
        "world_id": world_id,
        "task_family": FAMILY,
        "reused_visual_captures": [str(Path(captures[v]["capture"]).resolve()) for v in ("v0", "v1")],
        "audio_render_policy": "reuse_completed_members_and_independently_render_missing_members",
    })
    try:
        variants: dict[str, dict[str, Any]] = {}
        questions: dict[str, dict[str, Any]] = {}
        for visual_id in ("v0", "v1"):
            endpoints = _endpoint_bindings(captures[visual_id], plans[visual_id])
            for assignment in ("a0", "a1"):
                member_id = f"{visual_id}_{assignment}"
                result_path = source / f"variants/{member_id}/delivery/result.json"
                reused = result_path.is_file()
                if reused:
                    retained_result = _load(result_path)
                    variants[member_id] = _normalize_resume_result(retained_result)
                    reused_members[member_id] = str(result_path)
                    destination = output / f"variants/{member_id}/delivery"
                    destination.mkdir(parents=True)
                    _write(destination / "result.json", retained_result)
                else:
                    targets = _variant_targets(
                    visual_id, assignment,
                    first_event_actor_id=str(
                        plans[visual_id].get("audio_events", [{}])[0].get("actor_id") or "source1"
                    ),
                )
                    assigned, rebound = native.build_audio_assignment_plan(
                        plans[visual_id], plans[visual_id]["request"], assignment,
                        **_assignment_kwargs(assignment, targets, endpoints),
                    )
                    variant_root = native.materialize_audio_variant(
                        captures[visual_id], output / "variants" / member_id,
                        assigned, rebound, member_id=member_id,
                    )
                    variants[member_id] = native.finalize_audio_assignment(variant_root, rebound)
                facts = _load(variants[member_id]["facts"])
                questions[member_id] = generate_binding_question(
                    facts, FAMILY, DEFAULT_QUERY, seed=f"{group_id}:{member_id}"
                )
                _write(output / f"variants/{member_id}/binding_question.json", questions[member_id])
        pcm_by_column = {}
        for assignment in ("a0", "a1"):
            pcm_by_column[assignment] = _pcm_equal(
                variants[f"v0_{assignment}"]["audio"],
                variants[f"v1_{assignment}"]["audio"],
            )
            if not pcm_by_column[assignment]["same"]:
                raise IdentityNativeError(
                    f"resume audio column {assignment} differs in actual PCM"
                )
        route_metadata = {
            "source": "completed_identity_plan_readback",
            "visual_variants": {
                visual_id: deepcopy(plans[visual_id].get("identity_intervention"))
                for visual_id in ("v0", "v1")
            },
        }
        spec = _group_spec(
            group_id, world_id, request,
            native.room_family_from_plan(plans["v0"]),
            str(plans["v0"]["scene"]["room_id"]),
            variants, questions, route_metadata,
        )
        spec_path = _write(output / "group_spec.json", spec)
        _ensure_audio_tools()
        assembled = assemble_binding_dataset(
            spec, input_base=output, output=output / "assembled",
            seed=group_id, verify_media=True,
        )
        summary = {
            "schema": "avengine_binding_group_identity_summary_v1",
            "status": "pass",
            "group_id": group_id,
            "world_id": world_id,
            "task_family": FAMILY,
            "source_root": str(source),
            "group_spec": str(spec_path),
            "assembled": str(output / "assembled/binding_groups.json"),
            "assembled_group_count": assembled["group_count"],
            "assembled_sample_count": assembled["sample_count"],
            "validation": assembled["validation"],
            "variants": variants,
            "reused_completed_members": reused_members,
            "pcm_by_column": pcm_by_column,
            "question_truth_by_member": {
                member_id: item["forms"]["open"]["truth"]
                for member_id, item in questions.items()
            },
            "claim_boundary": "research_only native media and cross-event physical identity relation; human/model/formal admission not run",
        }
        _write(output / "summary.json", summary)
        return summary
    except Exception as exc:
        try:
            _write(output / "failure.json", {
                "schema": "avengine_binding_group_identity_resume_provenance_v1",
                "status": "fail", "source_root": str(source),
                "group_id": group_id, "world_id": world_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
        except Exception:
            pass
        raise



# ---------------------------------------------------------------------------
# Resumable per-unit identity recipe
#
# The legacy prepare_identity_group function remains available as a whole-group
# compatibility entry point. These runners expose the same algorithm as
# ordinary plan/capture/audio units so the scheduler can account for the
# internal probe and resume each boundary independently.
# ---------------------------------------------------------------------------

IDENTITY_INTERNAL_UNIT_IDS = frozenset({
    "identity_probe_plan",
    "identity_probe_capture",
    "identity_probe_audio",
    "identity_topology",
})


def _identity_done(results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return native._result_rows(results)


def _identity_result(
    done: Mapping[str, Mapping[str, Any]], unit_id: str
) -> dict[str, Any]:
    row = done.get(unit_id)
    if row is None:
        raise IdentityNativeError(
            f"cross_event_identity is missing a passing result for {unit_id}"
        )
    return dict(row)


def _identity_member_request(
    context: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    requests = context.get("member_requests")
    if not isinstance(requests, Mapping):
        raise IdentityNativeError("cross_event_identity context has no member_requests")
    member_ids = list(item.get("member_request_ids") or ())
    if not member_ids:
        member_ids = list(
            (context.get("contract") or {}).get("member_request_ids") or ()
        )
    for member_id in member_ids:
        request = requests.get(str(member_id))
        if isinstance(request, Mapping):
            return deepcopy(dict(request))
    raise IdentityNativeError(
        f"cross_event_identity has no request for stage members {member_ids}"
    )


def _identity_retained_root(
    context: Mapping[str, Any], key: str
) -> Path | None:
    roots = context.get("identity_retained_roots") or {}
    if not isinstance(roots, Mapping):
        return None
    value = roots.get(key)
    aliases = {
        "identity_probe_plan": ("identity_probe", "probe"),
        "identity_probe_capture": ("identity_probe", "probe"),
        "identity_probe_audio": ("identity_probe_variant", "identity_probe_audio"),
    }
    if value is None:
        for alias in aliases.get(key, ()):
            value = roots.get(alias)
            if value is not None:
                break
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _identity_base_root(context: Mapping[str, Any]) -> Path:
    value = context.get("identity_base_episode_root")
    if value is None:
        value = (context.get("identity_retained_roots") or {}).get(
            "identity_base"
        )
    if value is None:
        raise IdentityNativeError(
            "cross_event_identity needs an explicit identity_base_episode_root "
            "with plan/episode_plan.json; no planner is started implicitly"
        )
    root = Path(value).expanduser().resolve()
    if not (root / "plan/episode_plan.json").is_file():
        raise IdentityNativeError(
            f"cross_event_identity base episode root lacks plan/episode_plan.json: {root}"
        )
    return root


def _identity_plan_path(
    row: Mapping[str, Any], *, unit_id: str
) -> Path:
    facts = row.get("facts") or {}
    outputs = row.get("outputs") or {}
    value = facts.get("episode_plan_path") or outputs.get("episode_plan")
    if not isinstance(value, str) or not value.strip():
        raise IdentityNativeError(f"{unit_id} result has no episode_plan path")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise IdentityNativeError(f"{unit_id} episode_plan is unavailable: {path}")
    return path


def _identity_capture_output(
    row: Mapping[str, Any], *, unit_id: str
) -> dict[str, Any]:
    outputs = row.get("outputs") or {}
    capture = outputs.get("capture")
    neutral = outputs.get("neutral_readback")
    if (
        not isinstance(capture, str)
        or not Path(capture).expanduser().resolve().is_dir()
        or not isinstance(neutral, str)
        or not Path(neutral).expanduser().resolve().is_file()
    ):
        raise IdentityNativeError(
            f"{unit_id} result has no readable capture and neutral_readback"
        )
    return {
        "capture": str(Path(capture).expanduser().resolve()),
        "neutral_readback": str(Path(neutral).expanduser().resolve()),
        "visual_video": outputs.get("visual_video"),
    }


def _identity_request_path(
    row: Mapping[str, Any], *, unit_id: str
) -> Path:
    value = (row.get("outputs") or {}).get("request_path")
    if not isinstance(value, str) or not Path(value).expanduser().resolve().is_file():
        raise IdentityNativeError(f"{unit_id} result has no readable request path")
    return Path(value).expanduser().resolve()


def _identity_sound_pair(
    context: Mapping[str, Any], probe_plan: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = probe_plan.get("identity_probe")
    if isinstance(metadata, Mapping):
        first = metadata.get("first_sound")
        second = metadata.get("second_sound")
        if isinstance(first, Mapping) and isinstance(second, Mapping):
            return deepcopy(dict(first)), deepcopy(dict(second))
    pair = context.get("identity_sound_pair")
    if isinstance(pair, Mapping):
        first = pair.get("first_sound", pair.get("first"))
        second = pair.get("second_sound", pair.get("second"))
        if isinstance(first, Mapping) and isinstance(second, Mapping):
            return deepcopy(dict(first)), deepcopy(dict(second))
    roots = context.get("identity_retained_roots") or {}
    public_root = roots.get("v0") if isinstance(roots, Mapping) else None
    if public_root is not None:
        plan_path = Path(public_root).expanduser().resolve() / "plan/episode_plan.json"
        if plan_path.is_file():
            public_plan = _load(plan_path)
            events = public_plan.get("audio_events")
            if isinstance(events, list) and len(events) >= 2:
                first = next(
                    (row for row in events if isinstance(row, Mapping)
                     and row.get("event_id") == "event_001"),
                    None,
                )
                second = next(
                    (row for row in events if isinstance(row, Mapping)
                     and row.get("event_id") == "event_002"),
                    None,
                )
                if isinstance(first, Mapping) and isinstance(second, Mapping):
                    return deepcopy(dict(first)), deepcopy(dict(second))
    raise IdentityNativeError(
        "cross_event_identity probe has no selected first/second sound pair"
    )


def _identity_static_base_plan(
    base_plan: Mapping[str, Any],
    request: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _base_identity_plan(base_plan, request, registry)
    source_frames = base_plan.get("visual_plan", {}).get("frames")
    actors = plan.get("visual_plan", {}).get("actors")
    if not isinstance(source_frames, list) or not isinstance(actors, list):
        raise IdentityNativeError(
            "cross_event_identity base plan lacks visual frames and actors"
        )
    plan["visual_plan"]["frames"] = deepcopy(source_frames)
    for frame in plan["visual_plan"]["frames"]:
        if not isinstance(frame, Mapping):
            continue
        frame["camera_state"] = {
            **deepcopy(plan["visual_plan"]["camera"]),
            "frame_index": int(frame.get("frame_index", 0)),
        }
        states = frame.get("actor_states")
        if not isinstance(states, list):
            continue
        for actor_index, actor in enumerate(actors):
            if actor_index >= len(states) or not isinstance(states[actor_index], Mapping):
                continue
            initial = base_plan["visual_plan"]["frames"][0]["actor_states"][actor_index]
            state = deepcopy(initial)
            state.update(
                actor_id=actor["actor_id"],
                frame_index=int(frame.get("frame_index", 0)),
                action_id=actor["timeline"]["idle_action_id"],
                action_phase=0.0,
                action_time_ticks=0,
                moving=False,
            )
            point = np.asarray(
                initial["root_transform"]["translation_m"], dtype=float
            )
            rotation = np.asarray(
                initial["root_transform"].get(
                    "rotation_xyzw", [0.0, 0.0, 0.0, 1.0]
                ),
                dtype=float,
            )
            quaternion, matrix = _rotation(
                2.0 * math.atan2(float(rotation[1]), float(rotation[3]))
            )
            state["planned_emitter_m"] = (
                point
                + matrix @ np.asarray(
                    actor["emitter_binding"]["emitter_offset_m"], dtype=float
                )
            ).tolist()
            state["root_transform"]["rotation_xyzw"] = quaternion
            states[actor_index] = state
    return plan


def identity_group_stage_context(
    *args: Any,
    identity_base_episode_root: str | Path | None = None,
    identity_sound_pair: Mapping[str, Any] | None = None,
    identity_topology: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build native group context while retaining identity-only source hints."""
    retained = kwargs.pop("retained_visual_roots", None)
    retained = dict(retained or {}) if isinstance(retained, Mapping) else {}
    public_keys = {"v0", "v1", "v0_capture", "v1_capture"}
    public_retained = {
        str(key): value for key, value in retained.items() if key in public_keys
    }
    context = native.group_stage_context(
        *args,
        retained_visual_roots=public_retained or None,
        **kwargs,
    )
    context["identity_retained_roots"] = {
        str(key): str(Path(value).expanduser().resolve())
        for key, value in retained.items()
    }
    if identity_base_episode_root is not None:
        context["identity_base_episode_root"] = str(
            Path(identity_base_episode_root).expanduser().resolve()
        )
    if identity_sound_pair is not None:
        context["identity_sound_pair"] = deepcopy(dict(identity_sound_pair))
    if identity_topology is not None:
        context["identity_retained_topology"] = deepcopy(dict(identity_topology))
    return context


def load_group_stage_results(
    output_root: str | Path, group_id: str
) -> list[dict[str, Any]]:
    return native.load_group_stage_results(output_root, group_id)


def _identity_plan_only_bootstrap(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    request: Mapping[str, Any],
    unit_root: Path,
) -> tuple[Path, Path, dict[str, Any] | None]:
    """Resolve a base plan or create it through the ordinary plan-only path."""
    try:
        base_root = _identity_base_root(context)
        return (
            base_root,
            base_root / "plan/episode_plan.json",
            None,
        )
    except IdentityNativeError:
        bootstrap_request_path = _write(
            unit_root / "base_plan_request.json", request
        )
        planned = native.plan_visual_variant(
            bootstrap_request_path,
            unit_root / "base_plan",
            label=f"{item['unit_id']}_base_plan",
            log=unit_root / f"{item['unit_id']}.base_plan.log",
        )
        base_root = Path(planned["output"]).expanduser().resolve()
        base_plan_path = Path(planned["plan"]).expanduser().resolve()
        return (
            base_root,
            base_plan_path,
            {
                "source": "ordinary_plan_only",
                "entrypoint": "avengine.dataset.binding_group_native.plan_visual_variant",
                "plan_path": str(base_plan_path),
                "base_episode_root": str(base_root),
                "native_visual_worlds_created": 0,
                "native_acoustic_contexts_created": 0,
            },
        )


def _run_identity_probe_plan_unit(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del output_root, results, lease
    retained = _identity_retained_root(context, "identity_probe_plan")
    if retained is not None:
        plan_path = retained / "plan/episode_plan.json"
        request_path = retained / "request.json"
        if not plan_path.is_file() or not request_path.is_file():
            raise IdentityNativeError(
                f"identity probe retained root lacks plan/request: {retained}"
            )
        plan = _load(plan_path)
        request = _load(request_path)
        first_sound, second_sound = _identity_sound_pair(context, plan)
        return native._stage_result(
            item,
            status="pass",
            facts={
                "episode_plan_path": str(plan_path.resolve()),
                "renderer": native._plan_renderer(plan),
                "clock": deepcopy(plan.get("clock")),
                "identity_probe": {
                    "first_sound": first_sound,
                    "second_sound": second_sound,
                    "retained": True,
                },
            },
            outputs={
                "episode_plan": str(plan_path.resolve()),
                "request_path": str(request_path.resolve()),
                "base_plan_source": str(retained.resolve()),
                "first_sound": first_sound,
                "second_sound": second_sound,
                "identity_probe": True,
                "internal_only": True,
            },
        )
    request = _identity_member_request(context, item)
    base_root, base_plan_path, base_plan_bootstrap = _identity_plan_only_bootstrap(
        item, context, request, unit_root
    )
    base_plan = _load(base_plan_path)
    _validate_request(request, base_plan)
    registry, asset_index = _registry(request)
    base = _identity_static_base_plan(base_plan, request, registry)
    first_sound, second_sound, selection = _select_sound_pair(
        request,
        request["source_asset_ids"],
        asset_index,
        plan=base,
    )
    first_actor, actor_selection = _first_event_actor(request)
    selection["first_event_actor"] = actor_selection
    first_event = _event(
        first_sound,
        event_id="event_001",
        actor_id=first_actor,
        start_s=float(_config(request)["event_start_s"]),
        clock=base["clock"],
    )
    probe_request = deepcopy(request)
    probe_request["episode_id"] = f"{context['group_id']}_probe"
    probe_request["entities"] = {
        **dict(request["entities"]),
        "silent_count": 1,
    }
    probe_plan = _with_audio(base, probe_request, [first_event])
    probe_plan["identity_probe"] = {
        "first_sound": deepcopy(first_sound),
        "second_sound": deepcopy(second_sound),
        "sound_selection": deepcopy(selection),
        "first_event_actor": deepcopy(actor_selection),
    }
    # A fresh group has no retained base episode, so the plan-only bootstrap
    # above already wrote into this attempt root. The dispatcher has refused a
    # pre-existing attempt root before calling us, so tolerating it here cannot
    # reuse another attempt's output.
    unit_root.mkdir(parents=True, exist_ok=True)
    request_path = _write(unit_root / "request.json", probe_request)
    base_path = _write(unit_root / "base_plan.json", base)
    plan_path = _write(unit_root / "plan/episode_plan.json", probe_plan)
    _write(unit_root / "first_sound.json", first_sound)
    _write(unit_root / "second_sound.json", second_sound)
    return native._stage_result(
        item,
        status="pass",
        facts={
            "episode_plan_path": str(plan_path),
            "renderer": native._plan_renderer(probe_plan),
            "clock": deepcopy(probe_plan.get("clock")),
            "identity_probe": {
                "first_sound": first_sound,
                "second_sound": second_sound,
                "selection": selection,
                "first_event_actor": actor_selection,
            },
        },
        outputs={
            "episode_plan": str(plan_path),
            "request_path": str(request_path),
            "base_plan": str(base_path),
            "base_plan_source": str(base_root),
            "base_plan_bootstrap": deepcopy(base_plan_bootstrap),
            "first_sound": first_sound,
            "second_sound": second_sound,
            "first_event_actor_id": first_actor,
            "identity_probe": True,
            "internal_only": True,
        },
    )


def _run_identity_probe_capture_unit(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del output_root
    done = _identity_done(results)
    plan_row = _identity_result(done, "identity_probe_plan")
    plan_path = _identity_plan_path(plan_row, unit_id="identity_probe_plan")
    request_path = _identity_request_path(
        plan_row, unit_id="identity_probe_plan"
    )
    plan = _load(plan_path)
    request = _load(request_path)
    # The internal probe has no member_request_ids, so the normal runner's
    # context request fallback cannot carry the lease into this recipe path.
    # Bind the allocator's per-instance runtime before materializing the visual
    # tree; otherwise capture_visual_plan silently falls back to adapter 0 and
    # its default RPC port even when the broker granted another device/port.
    instance_runtime = native.resolve_instance_runtime(item, lease=lease)
    if lease is not None and instance_runtime.get("graphics_adapter") is None:
        raise IdentityNativeError(
            "identity probe capture received a lease without graphics_adapter"
        )
    if (
        lease is not None
        and native.room_family_from_plan(plan) in {"apartment", "kujiale"}
        and instance_runtime.get("rpc_port") is None
    ):
        raise IdentityNativeError(
            "identity probe capture received a lease without rpc_port"
        )
    if instance_runtime.get("graphics_adapter") is not None:
        request.setdefault("runtime", {})
        request["runtime"] = {
            **dict(request.get("runtime") or {}),
            "graphics_adapter": int(instance_runtime["graphics_adapter"]),
        }
    if instance_runtime.get("rpc_port") is not None:
        request.setdefault("runtime", {})
        request["runtime"] = {
            **dict(request.get("runtime") or {}),
            "rpc_port": int(instance_runtime["rpc_port"]),
        }
    retained = _identity_retained_root(context, "identity_probe_capture")
    if retained is not None:
        capture_root = retained / "capture"
        if not capture_root.is_dir():
            capture_root = retained
        neutral = capture_root / "neutral_readback.json"
        if not neutral.is_file():
            raise IdentityNativeError(
                f"identity probe retained root lacks neutral_readback.json: {capture_root}"
            )
        captured = {
            "output": str(retained.resolve()),
            "plan": str(plan_path.resolve()),
            "capture": str(capture_root.resolve()),
            "neutral_readback": str(neutral.resolve()),
            "frame_readbacks": str(
                (
                    capture_root / "frame_readbacks.json"
                    if (capture_root / "frame_readbacks.json").is_file()
                    else capture_root / "frame_records.json"
                ).resolve()
            ),
            "visual_video": None,
        }
        native_worlds = 0
    else:
        base_root = Path(
            (plan_row.get("outputs") or {}).get("base_plan_source")
            or context.get("identity_base_episode_root")
            or _identity_base_root(context)
        ).expanduser().resolve()
        visual_root = _materialize_visual(
            base_root, unit_root / "visual", plan, request
        )
        command = native._qa_module().capture_command(dict(request), visual_root)
        command_runtime = {}
        if lease is not None:
            flags = {"graphics_adapter": "--graphics-adapter"}
            if native.room_family_from_plan(plan) in {"apartment", "kujiale"}:
                flags["rpc_port"] = "--rpc-port"
            for key, flag in flags.items():
                expected = instance_runtime[key]
                try:
                    actual = int(command[command.index(flag) + 1])
                except (ValueError, IndexError):
                    raise IdentityNativeError(
                        f"identity probe capture command lacks leased {flag}"
                    )
                if expected is None or actual != int(expected):
                    raise IdentityNativeError(
                        f"identity probe capture command {flag}={actual} "
                        f"differs from lease value {expected}"
                    )
                command_runtime[key] = actual
        _write(unit_root / "capture_launch_preflight.json", {
            "status": "pass",
            "argv": command,
            "instance_runtime": deepcopy(instance_runtime),
            "command_runtime": command_runtime,
            "lease_checked": lease is not None,
        })
        captured = _capture(request, visual_root, "identity_probe")
        native_worlds = 1
    frame_count = native._captured_frame_count(
        Path(captured["capture"]), plan
    )
    unit_root.mkdir(parents=True, exist_ok=True)
    capture_path = _write(
        unit_root / "capture.json",
        captured,
    )
    return native._stage_result(
        item,
        status="pass",
        facts={
            "capture_receipt_path": str(
                (Path(captured["capture"]) / "research_receipt.json").resolve()
            ),
            "captured_frame_count": frame_count,
        },
        outputs={
            "capture": str(Path(captured["capture"]).resolve()),
            "capture_root": str(Path(captured["output"]).resolve()),
            "episode_plan": str(plan_path.resolve()),
            "request_path": str(request_path.resolve()),
            "neutral_readback": str(
                Path(captured["neutral_readback"]).resolve()
            ),
            "frame_readbacks": str(
                Path(captured["frame_readbacks"]).resolve()
            ),
            "visual_video": captured.get("visual_video"),
            "capture_receipt": str(capture_path.resolve()),
            "native_visual_worlds_created": native_worlds,
            "instance_runtime": deepcopy(instance_runtime),
            "internal_only": True,
            "member_request_ids": [],
        },
    )


def _identity_probe_tail(
    facts: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], float]:
    audio = facts.get("audio") if isinstance(facts.get("audio"), Mapping) else {}
    tails = audio.get("wet_tail_intervals")
    if not isinstance(tails, list) or not tails:
        raise IdentityNativeError(
            "identity probe audio facts have no measured wet_tail_intervals"
        )
    ends = []
    for row in tails:
        if isinstance(row, Mapping):
            value = row.get("end_s")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                ends.append(float(value))
    if not ends:
        raise IdentityNativeError(
            "identity probe audio facts have no finite wet-tail end_s"
        )
    return deepcopy(tails), max(ends)


def _identity_probe_audio_unit(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del output_root, lease
    done = _identity_done(results)
    plan_row = _identity_result(done, "identity_probe_plan")
    capture_row = _identity_result(done, "identity_probe_capture")
    plan_path = _identity_plan_path(plan_row, unit_id="identity_probe_plan")
    request_path = _identity_request_path(
        plan_row, unit_id="identity_probe_plan"
    )
    plan = _load(plan_path)
    request = _load(request_path)
    capture = _identity_capture_output(
        capture_row, unit_id="identity_probe_capture"
    )
    retained = _identity_retained_root(context, "identity_probe_audio")
    if retained is not None:
        facts_path = retained / "delivery/facts.json"
        report_path = retained / "delivery/research_report.json"
        if not facts_path.is_file() or not report_path.is_file():
            raise IdentityNativeError(
                f"identity probe retained audio root lacks facts/report: {retained}"
            )
        facts = _load(facts_path)
        tails, tail_end = _identity_probe_tail(facts)
        result_doc = (
            _load(retained / "delivery/result.json")
            if (retained / "delivery/result.json").is_file()
            else {}
        )
        audio_path = (
            (facts.get("audio") or {}).get("path")
            or result_doc.get("lossless_stereo_wav")
        )
        if not isinstance(audio_path, str) or not Path(audio_path).is_file():
            raise IdentityNativeError(
                f"identity probe retained audio has no readable mixture: {retained}"
            )
        visual_video = result_doc.get("visual_video")
        if not isinstance(visual_video, str):
            candidate = retained / "delivery/visual_rgb.mp4"
            visual_video = str(candidate.resolve()) if candidate.is_file() else None
        native_contexts = 0
        probe_result_path = (
            retained.parent / "audio.json"
            if (retained.parent / "audio.json").is_file()
            else None
        )
    else:
        result_doc, tail_end = _probe_audio(
            plan, request, capture, unit_root / "probe"
        )
        facts_path = Path(result_doc["facts"]).expanduser().resolve()
        report_path = Path(result_doc["audio_report"]).expanduser().resolve()
        facts = _load(facts_path)
        tails, _ = _identity_probe_tail(facts)
        audio_path = str(Path(result_doc["audio"]).expanduser().resolve())
        visual_video = result_doc.get("visual_video")
        native_contexts = 1
        probe_result_path = None
    unit_root.mkdir(parents=True, exist_ok=True)
    return native._stage_result(
        item,
        status="pass",
        facts={
            "facts_path": str(facts_path.resolve()),
            "audio_report_path": str(report_path.resolve()),
            "wet_tail_intervals": tails,
            "measured_wet_tail_end_s": tail_end,
        },
        outputs={
            "audio": str(Path(audio_path).resolve()),
            "audio_report": str(report_path.resolve()),
            "facts": str(facts_path.resolve()),
            "visual_video": visual_video,
            "capture": capture["capture"],
            "neutral_readback": capture["neutral_readback"],
            "episode_plan": str(plan_path.resolve()),
            "request_path": str(request_path.resolve()),
            "first_sound": deepcopy(
                (plan.get("identity_probe") or {}).get(
                    "first_sound",
                    (plan.get("audio_events") or [{}])[0],
                )
            ),
            "second_sound": deepcopy(
                (plan.get("identity_probe") or {}).get("second_sound")
            ),
            "measured_wet_tail_end_s": tail_end,
            "native_acoustic_contexts_created": native_contexts,
            "native_visual_worlds_created": 0,
            "internal_only": True,
            "member_request_ids": [],
            "probe_result_path": (
                None if probe_result_path is None
                else str(probe_result_path.resolve())
            ),
            "shared_audio_column": {
                "reused": False,
                "internal_probe": True,
            },
        },
    )


def _run_identity_topology_unit(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del output_root, lease
    done = _identity_done(results)
    plan_row = _identity_result(done, "identity_probe_plan")
    probe_audio = _identity_result(done, "identity_probe_audio")
    probe_plan = _load(
        _identity_plan_path(plan_row, unit_id="identity_probe_plan")
    )
    probe_request = _load(
        _identity_request_path(plan_row, unit_id="identity_probe_plan")
    )
    facts = probe_audio.get("facts") or {}
    tail_end = facts.get("measured_wet_tail_end_s")
    if isinstance(tail_end, bool) or not isinstance(tail_end, (int, float)):
        raise IdentityNativeError(
            "identity topology requires the probe's measured wet-tail end"
        )
    first_sound, second_sound = _identity_sound_pair(context, probe_plan)
    retained_topology = context.get("identity_retained_topology")
    topology_base_path = (plan_row.get("outputs") or {}).get("base_plan")
    tracks = None
    if isinstance(retained_topology, Mapping):
        topology = deepcopy(dict(retained_topology))
        topology.setdefault("motion_guard", {})
        topology["motion_guard"] = dict(topology["motion_guard"])
        topology["motion_guard"]["measured_wet_tail_end_s"] = float(tail_end)
    else:
        base_path = topology_base_path
        base_plan = (
            _load(Path(base_path))
            if isinstance(base_path, str) and Path(base_path).is_file()
            else deepcopy(probe_plan)
        )
        topology, tracks = _select_topology(
            base_plan,
            probe_request,
            float(tail_end),
            first_sound,
            second_sound,
        )
    unit_root.mkdir(parents=True)
    topology_path = _write(unit_root / "topology.json", topology)
    probe_request_path = _identity_request_path(
        plan_row, unit_id="identity_probe_plan"
    )
    selected_request = deepcopy(probe_request)
    selected_request_path = _write(
        unit_root / "selected_request.json", selected_request
    )
    first_path = _write(unit_root / "first_sound.json", first_sound)
    second_path = _write(unit_root / "second_sound.json", second_sound)
    tracks_path = None
    if tracks is not None:
        tracks_path = _write(unit_root / "tracks.json", tracks)
    return native._stage_result(
        item,
        status="pass",
        facts={
            "episode_plan_path": str(topology_path.resolve()),
            "renderer": "identity_topology",
            "clock": deepcopy(probe_plan.get("clock")),
            "measured_wet_tail_end_s": float(tail_end),
            "event2_start_frame": topology.get("event2_start_frame"),
        },
        outputs={
            "topology_path": str(topology_path.resolve()),
            "base_plan_path": (
                None
                if not isinstance(topology_base_path, str)
                else str(Path(topology_base_path).expanduser().resolve())
            ),
            "selected_request_path": str(selected_request_path.resolve()),
            "probe_request_path": str(probe_request_path.resolve()),
            "first_sound_path": str(first_path.resolve()),
            "second_sound_path": str(second_path.resolve()),
            "tracks_path": (
                None if tracks_path is None else str(tracks_path.resolve())
            ),
            "topology": deepcopy(topology),
            "measured_wet_tail_end_s": float(tail_end),
            "internal_only": True,
        },
    )


def _identity_public_plan_from_topology(
    unit_id: str,
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    done = _identity_done(results)
    topology_row = _identity_result(done, "identity_topology")
    topology_outputs = topology_row.get("outputs") or {}
    topology_path = topology_outputs.get("topology_path")
    if not isinstance(topology_path, str):
        raise IdentityNativeError("identity public plan has no topology output")
    topology = _load(Path(topology_path))
    measured_tail = topology_outputs.get("measured_wet_tail_end_s")
    if not isinstance(measured_tail, (int, float)):
        raise IdentityNativeError(
            "identity public plan has no measured probe wet-tail end"
        )
    retained = _identity_retained_root(context, unit_id)
    if retained is not None:
        plan_path = retained / "plan/episode_plan.json"
        request_path = retained / "request.json"
        plan = _load(plan_path)
        request = _load(request_path)
        _validate_request(request, plan)
        intervention = plan.get("identity_intervention")
        observed_tail = (
            intervention.get("movement_after_measured_wet_tail_s")
            if isinstance(intervention, Mapping)
            else None
        )
        if (
            not isinstance(observed_tail, (int, float))
            or not math.isclose(
                float(observed_tail), float(measured_tail),
                abs_tol=1.0e-6, rel_tol=0.0,
            )
        ):
            raise IdentityNativeError(
                f"{unit_id} plan does not consume the measured probe wet tail: "
                f"{observed_tail!r} != {measured_tail!r}"
            )
        if unit_id == "v1":
            previous = _identity_result(done, "v0")
            previous_plan = _load(
                _identity_plan_path(previous, unit_id="v0")
            )
            _verify_selected_identity_plans(
                _identity_member_request(context, item),
                {"v0": previous_plan, "v1": plan},
            )
        return native._stage_result(
            item,
            status="pass",
            facts={
                "episode_plan_path": str(plan_path.resolve()),
                "renderer": native._plan_renderer(plan),
                "clock": deepcopy(plan.get("clock")),
                "measured_wet_tail_end_s": float(measured_tail),
            },
            outputs={
                "episode_plan": str(plan_path.resolve()),
                "request_path": str(request_path.resolve()),
                "topology_path": str(Path(topology_path).resolve()),
                "source_asset_ids": list(plan.get("request", {}).get(
                    "source_asset_ids", ()
                )),
                "measured_wet_tail_end_s": float(measured_tail),
                "identity_intervention": deepcopy(
                    plan.get("identity_intervention")
                ),
                "reused_retained_plan": str(retained.resolve()),
            },
        )
    base_path = topology_outputs.get("base_plan_path")
    if not isinstance(base_path, str):
        base_path = topology_outputs.get("base_plan")
    if not isinstance(base_path, str) or not Path(base_path).is_file():
        raise IdentityNativeError(
            "identity public plan needs topology base_plan_path"
        )
    base_plan = _load(Path(base_path))
    selected_request_path = topology_outputs.get("selected_request_path")
    if not isinstance(selected_request_path, str):
        raise IdentityNativeError("identity topology has no selected request")
    selected_request = _load(Path(selected_request_path))
    first_path = topology_outputs.get("first_sound_path")
    second_path = topology_outputs.get("second_sound_path")
    if not isinstance(first_path, str) or not isinstance(second_path, str):
        raise IdentityNativeError("identity topology has no selected sound paths")
    first_sound = _load(Path(first_path))
    second_sound = _load(Path(second_path))
    request = _identity_member_request(context, item)
    _validate_request(request, base_plan)
    variant_request = deepcopy(request)
    variant_request["binding_identity"] = deepcopy(
        selected_request["binding_identity"]
    )
    visual_id = unit_id
    variant_request["episode_id"] = f"{context['group_id']}_{visual_id}"
    first_actor = str(
        topology.get("first_event_actor_id")
        or (selected_request.get("binding_identity") or {}).get(
            "first_event_actor_id"
        )
        or "source1"
    )
    event1 = _event(
        first_sound,
        event_id="event_001",
        actor_id=first_actor,
        start_s=float(_config(variant_request)["event_start_s"]),
        clock=base_plan["clock"],
    )
    event2 = _event(
        second_sound,
        event_id="event_002",
        actor_id="source1",
        start_s=float(topology["event2_start_s"]),
        clock=base_plan["clock"],
    )
    plan = _with_audio(base_plan, variant_request, [event1, event2])
    alignment = _align_event2_to_rir_boundary(
        plan,
        settled_frame=int(topology["event2_frame"]),
        rir_stride_frames=int(variant_request.get("rir_stride", 3)),
    )
    tracks_path = topology_outputs.get("tracks_path")
    if not isinstance(tracks_path, str) or not Path(tracks_path).is_file():
        raise IdentityNativeError(
            "identity public plan needs tracks from _select_topology unless a retained plan is supplied"
        )
    tracks = _load(Path(tracks_path))
    variant_tracks = tracks.get(visual_id)
    if not isinstance(variant_tracks, Mapping):
        raise IdentityNativeError(
            f"identity topology has no tracks for {visual_id}"
        )
    plan["visual_plan"]["frames"] = _build_frames(
        plan,
        plan["visual_plan"]["actors"],
        variant_tracks,
    )
    plan["identity_intervention"] = {
        "variant": visual_id,
        "persistent_actor_ids": ["source1", "source2"],
        "topology": (
            "source1_L_to_R" if visual_id == "v0" else "source2_P_to_R"
        ),
        "camera_motion": "static",
        "movement_after_measured_wet_tail_s": float(measured_tail),
        "event2_start_s": alignment["start_sample"]
        / float(plan["clock"]["sample_rate_hz"]),
        "event2_start_frame": int(
            alignment["start_sample"]
            / float(plan["clock"]["sample_rate_hz"])
            * float(plan["clock"]["frame_rate_hz"])
        ),
        "event2_rir_alignment": alignment,
    }
    probe_plan_row = _identity_result(done, "identity_probe_plan")
    base_root_value = (probe_plan_row.get("outputs") or {}).get("base_plan_source")
    if not isinstance(base_root_value, str):
        raise IdentityNativeError("identity public plan lacks its probe base plan root")
    visual_root = _materialize_visual(
        Path(base_root_value).expanduser().resolve(),
        unit_root / "episode", plan, variant_request,
    )
    request_path = visual_root / "request.json"
    plan_path = visual_root / "plan/episode_plan.json"
    # Resolve the real producer inputs during the CPU plan stage, before a
    # capture attempt spends its native allowance.
    command = native._qa_module().capture_command(dict(variant_request), visual_root)
    _write(unit_root / "materialization_preflight.json", {
        "status": "pass", "argv": command,
        "plan_path": str(plan_path.resolve()),
        "base_root": str(Path(base_root_value).expanduser().resolve()),
    })
    _write(unit_root / "topology.json", topology)
    return native._stage_result(
        item,
        status="pass",
        facts={
            "episode_plan_path": str(plan_path.resolve()),
            "renderer": native._plan_renderer(plan),
            "clock": deepcopy(plan.get("clock")),
            "measured_wet_tail_end_s": float(measured_tail),
            "event2_start_frame": alignment["rir_key_frame"],
        },
        outputs={
            "episode_plan": str(plan_path.resolve()),
            "request_path": str(request_path.resolve()),
            "topology_path": str(Path(topology_path).resolve()),
            "source_asset_ids": list(
                variant_request["source_asset_ids"]
            ),
            "measured_wet_tail_end_s": float(measured_tail),
            "event2_alignment": alignment,
            "identity_intervention": deepcopy(
                plan["identity_intervention"]
            ),
        },
    )


def _run_identity_visual_plan_unit(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unit_id = str(item.get("unit_id") or "")
    if unit_id == "identity_probe_plan":
        return _run_identity_probe_plan_unit(
            item, context, unit_root, output_root=output_root,
            results=results, lease=lease,
        )
    if unit_id == "identity_topology":
        return _run_identity_topology_unit(
            item, context, unit_root, output_root=output_root,
            results=results, lease=lease,
        )
    if unit_id in {"v0", "v1"}:
        return _identity_public_plan_from_topology(
            unit_id, item, context, unit_root, results
        )
    raise IdentityNativeError(
        f"cross_event_identity visual_plan runner cannot handle {unit_id!r}"
    )


def _run_identity_visual_capture_unit(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unit_id = str(item.get("unit_id") or "")
    if unit_id == "identity_probe_capture":
        return _run_identity_probe_capture_unit(
            item, context, unit_root, output_root=output_root,
            results=results, lease=lease,
        )
    if unit_id not in {"v0_capture", "v1_capture"}:
        raise IdentityNativeError(
            f"cross_event_identity visual_capture runner cannot handle {unit_id!r}"
        )
    return native.run_visual_capture_unit(
        item,
        context,
        unit_root,
        output_root=output_root,
        results=results,
        lease=lease,
    )



def _identity_retained_audio_result(
    root: Path,
    *,
    unit_id: str,
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    checked = native.verify_materialized_audio_root(root)
    facts_path = root / "delivery/facts.json"
    report_path = root / "delivery/research_report.json"
    if not facts_path.is_file() or not report_path.is_file():
        raise IdentityNativeError(
            f"{unit_id} retained audio root lacks facts/report: {root}"
        )
    facts = _load(facts_path)
    result_path = root / "delivery/result.json"
    result_doc = (
        _normalize_resume_result(_load(result_path))
        if result_path.is_file() else {}
    )
    audio_path = result_doc.get("audio") or (facts.get("audio") or {}).get("path")
    if not isinstance(audio_path, str) or not Path(audio_path).is_file():
        raise IdentityNativeError(
            f"{unit_id} retained audio has no readable mixture: {root}"
        )
    visual_video = result_doc.get("visual_video")
    if not isinstance(visual_video, str):
        visual_video = (facts.get("source_paths") or {}).get("video")
    if not isinstance(visual_video, str) or not Path(visual_video).is_file():
        candidate = root / "delivery/visual_rgb.mp4"
        visual_video = str(candidate.resolve()) if candidate.is_file() else None
    if not isinstance(visual_video, str) or not Path(visual_video).is_file():
        raise IdentityNativeError(
            f"{unit_id} retained audio has no readable visual_video: {root}"
        )
    ancillary = result_doc.get("ancillary_audio_outputs")
    if ancillary is None and isinstance(result_doc.get("result"), Mapping):
        ancillary = result_doc["result"].get("ancillary_audio_outputs")
    return {
        "facts": str(facts_path.resolve()),
        "audio_report": str(report_path.resolve()),
        "audio": str(Path(audio_path).resolve()),
        "visual_video": str(Path(visual_video).resolve()),
        "questions": result_doc.get("questions_path"),
        "preview": result_doc.get("preview"),
        "declared_audio_delivery": result_doc.get("declared_audio_delivery"),
        "delivered_audio_layouts": result_doc.get(
            "delivered_audio_layouts",
            {"status": "pass", "undelivered_attached_view_layouts": []},
        ),
        "ancillary_audio_outputs": deepcopy(ancillary),
        "materialized_precheck": checked,
        "capture": capture["capture"],
    }


def _identity_audio_stage_result_from_retained(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_id: str,
    capture_unit_id: str,
    capture: Mapping[str, Any],
) -> dict[str, Any] | None:
    retained = _identity_retained_root(context, unit_id)
    if retained is None:
        return None
    result = _identity_retained_audio_result(
        retained, unit_id=unit_id, capture=capture
    )
    facts = _load(Path(result["facts"]))
    audio = facts.get("audio") if isinstance(facts.get("audio"), Mapping) else {}
    tails = audio.get("wet_tail_intervals")
    if not isinstance(tails, list) or not tails:
        raise IdentityNativeError(
            f"{unit_id} retained audio facts have no wet_tail_intervals"
        )
    assignment = unit_id.rsplit("_", 1)[-1]
    return native._stage_result(
        item,
        status="pass",
        facts={
            "facts_path": result["facts"],
            "audio_report_path": result["audio_report"],
            "wet_tail_intervals": deepcopy(tails),
        },
        outputs={
            "variant_root": str(retained),
            "audio": result["audio"],
            "audio_report": result["audio_report"],
            "questions": result["questions"],
            "visual_video": result["visual_video"],
            "preview": result["preview"],
            "capture": capture["capture"],
            "neutral_readback": capture["neutral_readback"],
            "episode_plan": str(
                (retained / "plan/episode_plan.json").resolve()
            ),
            "request_path": str((retained / "request.json").resolve()),
            "assignment_plan_path": str(
                (retained / "plan/episode_plan.json").resolve()
            ),
            "assignment_request_path": str(
                (retained / "request.json").resolve()
            ),
            "assignment_column": assignment,
            "visual_unit_id": capture_unit_id,
            "member_request_id": (
                list(item.get("member_request_ids") or ())[0]
                if item.get("member_request_ids") else None
            ),
            "member_request_ids": list(item.get("member_request_ids") or ()),
            "declared_audio_delivery": result["declared_audio_delivery"],
            "delivered_audio_layouts": result["delivered_audio_layouts"],
            "ancillary_audio_outputs": result["ancillary_audio_outputs"],
            "shared_audio_column": {
                "reused": False,
                "identity_independent_render": True,
            },
            "identity_retained_audio_root": str(retained),
            "native_visual_worlds_created": 0,
            "native_acoustic_contexts_created": 0,
        },
    )


def _run_identity_audio_unit(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unit_id = str(item.get("unit_id") or "")
    if unit_id == "identity_probe_audio":
        return _identity_probe_audio_unit(
            item, context, unit_root, output_root=output_root,
            results=results, lease=lease,
        )
    if unit_id not in {"v0_a0", "v0_a1", "v1_a0", "v1_a1"}:
        raise IdentityNativeError(
            f"cross_event_identity audio runner cannot handle {unit_id!r}"
        )
    unit_spec = native._unit_row(context, unit_id)
    capture_unit_id = str(unit_spec.get("visual_unit_id") or "")
    if not capture_unit_id:
        raise IdentityNativeError(f"{unit_id} declares no visual unit")
    done = _identity_done(results)
    capture_row = _identity_result(done, capture_unit_id)
    capture = _identity_capture_output(
        capture_row, unit_id=capture_unit_id
    )
    plan_path = _identity_plan_path(
        _identity_result(done, capture_unit_id.replace("_capture", "")),
        unit_id=capture_unit_id.replace("_capture", ""),
    )
    request_path = _identity_request_path(
        _identity_result(done, capture_unit_id.replace("_capture", "")),
        unit_id=capture_unit_id.replace("_capture", ""),
    )
    plan = _load(plan_path)
    capture_request = _load(request_path)
    retained_result = _identity_audio_stage_result_from_retained(
        item, context, unit_id, capture_unit_id, capture
    )
    if retained_result is not None:
        return retained_result
    member_request_id, member_request = native._member_request_for_audio_unit(
        context, item, unit_spec
    )
    request, audio_view_fields = native._apply_member_audio_view_fields(
        capture_request, member_request, label=unit_id
    )
    visual_id = unit_id.split("_", 1)[0]
    assignment = unit_id.rsplit("_", 1)[-1]
    first_event_actor = str(
        (plan.get("audio_events") or [{}])[0].get("actor_id") or "source1"
    )
    targets = _variant_targets(
        visual_id, assignment,
        first_event_actor_id=first_event_actor,
    )
    endpoints = _endpoint_bindings(capture, plan)
    assigned, rebound = native.build_audio_assignment_plan(
        plan,
        request,
        assignment,
        **_assignment_kwargs(assignment, targets, endpoints),
    )
    unit_root.mkdir(parents=True)
    assignment_plan_path = _write(
        unit_root / "assignment_plan.json", assigned
    )
    assignment_request_path = _write(
        unit_root / "assignment_request.json", rebound
    )
    variant_root = native.materialize_audio_variant(
        capture,
        unit_root / "episode",
        assigned,
        rebound,
        member_id=unit_id,
    )
    finalized = native.finalize_audio_assignment(
        variant_root,
        rebound,
        shared_visual_root=native.shared_visual_evidence_root(
            output_root, context["group_id"]
        ),
    )
    visual_video = finalized.get("visual_video") or capture.get("visual_video")
    if (
        not isinstance(visual_video, str)
        or not Path(visual_video).expanduser().resolve().is_file()
    ):
        raise IdentityNativeError(
            f"{unit_id} audio finalization published no readable visual_video"
        )
    facts_path = Path(finalized["facts"]).expanduser().resolve()
    facts = _load(facts_path)
    audio_facts = facts.get("audio") if isinstance(facts.get("audio"), Mapping) else {}
    tails = audio_facts.get("wet_tail_intervals")
    if not isinstance(tails, list) or not tails:
        raise IdentityNativeError(
            f"{unit_id} delivery published no measured wet_tail_intervals"
        )
    finalized_result = finalized.get("result") or {}
    ancillary = finalized_result.get("ancillary_audio_outputs")
    declared_delivery = finalized.get(
        "declared_audio_delivery",
        native.declared_audio_delivery(rebound),
    )
    delivered_layouts = finalized.get(
        "delivered_audio_layouts",
        {"status": "pass", "undelivered_attached_view_layouts": []},
    )
    outputs = {
        "variant_root": str(variant_root),
        "audio": finalized["audio"],
        "audio_report": finalized["audio_report"],
        "questions": finalized["questions"],
        "visual_video": str(Path(visual_video).resolve()),
        "preview": finalized.get("preview"),
        "capture": capture["capture"],
        "neutral_readback": capture["neutral_readback"],
        "episode_plan": str(plan_path),
        "request_path": str(request_path),
        "assignment_plan_path": str(assignment_plan_path),
        "assignment_request_path": str(assignment_request_path),
        "assignment_column": assignment,
        "visual_unit_id": capture_unit_id,
        "member_request_id": member_request_id,
        "member_request_ids": list(item.get("member_request_ids") or ()),
        "audio_view_fields": audio_view_fields,
        "declared_audio_delivery": declared_delivery,
        "delivered_audio_layouts": delivered_layouts,
        "ancillary_audio_outputs": deepcopy(ancillary),
        "shared_audio_column": {
            "reused": False,
            "identity_independent_render": True,
        },
        "native_visual_worlds_created": 0,
        "native_acoustic_contexts_created": 1,
    }
    facts_out = {
        "facts_path": str(facts_path),
        "audio_report_path": str(
            Path(finalized["audio_report"]).resolve()
        ),
        "wet_tail_intervals": deepcopy(tails),
    }
    if delivered_layouts.get("status") != "pass":
        return native._stage_result(
            item,
            status="blocked",
            facts=facts_out,
            outputs=outputs,
            reason=delivered_layouts.get("reason"),
        )
    return native._stage_result(
        item, status="pass", facts=facts_out, outputs=outputs
    )


def _identity_public_audio_rows(
    context: Mapping[str, Any],
    done: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[tuple[str, str, str]],
]:
    variants: dict[str, dict[str, Any]] = {}
    plans: dict[str, dict[str, Any]] = {}
    member_units: list[tuple[str, str, str]] = []
    for audio_unit_id in ("v0_a0", "v0_a1", "v1_a0", "v1_a1"):
        row = _identity_result(done, audio_unit_id)
        facts = row.get("facts") or {}
        outputs = row.get("outputs") or {}
        plan_path = outputs.get("assignment_plan_path")
        if (
            not isinstance(plan_path, str)
            or not Path(plan_path).expanduser().resolve().is_file()
        ):
            raise IdentityNativeError(
                f"{audio_unit_id} has no readable assignment plan"
            )
        for key in ("facts_path", "audio_report_path"):
            if (
                not isinstance(facts.get(key), str)
                or not Path(facts[key]).expanduser().resolve().is_file()
            ):
                raise IdentityNativeError(
                    f"{audio_unit_id} facts lack readable {key}"
                )
        for key in ("audio", "visual_video"):
            if (
                not isinstance(outputs.get(key), str)
                or not Path(outputs[key]).expanduser().resolve().is_file()
            ):
                raise IdentityNativeError(
                    f"{audio_unit_id} outputs lack readable {key}"
                )
        plans[audio_unit_id] = _load(Path(plan_path))
        visual_unit_id = str(outputs.get("visual_unit_id") or "")
        if visual_unit_id not in {"v0_capture", "v1_capture"}:
            raise IdentityNativeError(
                f"{audio_unit_id} has no valid visual_unit_id"
            )
        variants[audio_unit_id] = {
            "facts": str(Path(facts["facts_path"]).resolve()),
            "audio": str(Path(outputs["audio"]).resolve()),
            "visual_video": str(Path(outputs["visual_video"]).resolve()),
            "audio_report": str(Path(facts["audio_report_path"]).resolve()),
            "visual_capture_root": str(
                Path(outputs["capture"]).resolve()
            ),
        }
        member_units.append(
            (audio_unit_id, visual_unit_id, audio_unit_id)
        )
    return variants, plans, member_units



def identity_shared_audio_input_equivalence(
    left: Mapping[str, Any], right: Mapping[str, Any],
    left_plan: Mapping[str, Any], right_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare actual acoustic inputs on every emitting/interpolation frame."""
    if native.audio_shared_content_signature(left_plan) != native.audio_shared_content_signature(right_plan):
        raise IdentityNativeError("identity canonical audio has different sounds or event timing")
    reports = [_load(Path(v["audio_report"])) for v in (left, right)]
    readbacks = [_load(Path(v["visual_capture_root"]) / "neutral_readback.json") for v in (left, right)]
    if readbacks[0]["clock"] != readbacks[1]["clock"] or readbacks[0]["camera"] != readbacks[1]["camera"]:
        raise IdentityNativeError("identity canonical audio has different native camera/clock")
    for key in ("clock", "gain_application"):
        if reports[0][key] != reports[1][key]:
            raise IdentityNativeError(f"identity canonical audio differs in {key}")
    for key in ("package_manifest", "hrtf", "simulation_request", "audio_render_config", "dry_assets"):
        if reports[0]["inputs"][key] != reports[1]["inputs"][key]:
            raise IdentityNativeError(f"identity canonical audio differs in actual input {key}")
    events = [{e["event_id"]: e for e in p["audio_events"]} for p in (left_plan, right_plan)]
    if set(events[0]) != set(events[1]):
        raise IdentityNativeError("identity canonical event IDs differ")
    layouts = set(reports[0]["outputs_by_layout"])
    if layouts != set(reports[1]["outputs_by_layout"]):
        raise IdentityNativeError("identity canonical audio layouts differ")
    proof = {
        "status": "pass", "source_reports": [left["audio_report"], right["audio_report"]],
        "source_readbacks": [str(Path(v["visual_capture_root"]) / "neutral_readback.json") for v in (left, right)],
        "checked": ["sounds_and_event_slices", "native_camera_and_clock", "room_and_material_package",
                    "hrtf", "simulation_and_gain", "emitter_positions_at_all_event_frames",
                    "RIR_interpolation_support_poses"],
        "layouts": {},
    }
    for layout in sorted(layouts):
        caches = [r["audio"]["layout_delivery"][layout]["cache"] for r in reports]
        requests = [_load(Path(c["path"]) / "request.json") for c in caches]
        for key in ("acoustic_scene", "acoustic_selection_binding", "output", "runtime_policy"):
            if requests[0][key] != requests[1][key]:
                raise IdentityNativeError(f"identity canonical RIR {layout} differs in {key}")
        if requests[0]["simulation"]["effective"] != requests[1]["simulation"]["effective"]:
            raise IdentityNativeError("identity canonical RIR simulation differs")
        plans = [_load(Path(c["plan_path"])) for c in caches]
        indices = [_load(Path(c["index_path"])) for c in caches]
        uses = []
        for plan, index in zip(plans, indices):
            by_id = {e["job_id"]: e for e in index["entries"]}
            uses.append({
                (u["source_slot_id"], int(u["frame_index"])): by_id[j["job_id"]]
                for j in plan["jobs"] for u in j["uses"]
            })
        grids = [r["dynamic_rir"]["by_layout"][layout]["keyframe_samples"] for r in reports]
        if grids[0] != grids[1]:
            raise IdentityNativeError("identity canonical RIR keyframe clocks differ")
        grid = np.asarray(grids[0], dtype=np.int64)
        clock = reports[0]["clock"]
        fps, rate = float(clock["frame_rate_hz"]), int(clock["sample_rate_hz"])
        event_proofs = []
        for event_id in sorted(events[0], key=lambda eid: events[0][eid]["start_sample"]):
            es = [e[event_id] for e in events]
            start, end = int(es[0]["start_sample"]), int(es[0]["end_sample_exclusive"])
            first = max(0, int(np.searchsorted(grid, start, side="right")) - 1)
            last = min(len(grid) - 1, int(np.searchsorted(grid, end - 1, side="left")))
            support = []
            max_ir = 0
            for ki in range(first, last + 1):
                frame = int(round(int(grid[ki]) * fps / rate))
                pair = []
                for i in (0, 1):
                    endpoint = es[i]["source_endpoint_id"]
                    slot = "source" + str(reports[i]["sources"]["source_ids"].index(endpoint) + 1)
                    entry = uses[i][(slot, frame)]
                    native_point = readbacks[i]["entities"][es[i]["actor_id"]][frame]["emitter"]
                    if entry["source_position_m"] != native_point:
                        raise IdentityNativeError(f"RIR pose does not match native emitter: {event_id}/{frame}")
                    pair.append(entry)
                for key in ("acoustic_state_sha256", "source_position_m", "listener_position_m", "listener_orientation_wxyz"):
                    if pair[0][key] != pair[1][key]:
                        raise IdentityNativeError(f"identity canonical event {event_id} frame {frame} differs in {key}")
                max_ir = max(max_ir, int(pair[0]["sample_count"]))
                support.append({
                    "frame_index": frame,
                    "acoustic_state": pair[0]["acoustic_state_sha256"],
                    "source_position_m": pair[0]["source_position_m"],
                    "ir_bytes_equal": pair[0]["ir_sha256"] == pair[1]["ir_sha256"],
                })
            first_frame = int(math.floor(start * fps / rate))
            last_frame = min(len(readbacks[0]["camera"]) - 1, int(math.ceil(end * fps / rate)))
            for frame in range(first_frame, last_frame + 1):
                points = [readbacks[i]["entities"][es[i]["actor_id"]][frame]["emitter"] for i in (0, 1)]
                if points[0] != points[1]:
                    raise IdentityNativeError(f"native emission positions differ: {event_id}/{frame}")
            event_proofs.append({
                "event_id": event_id, "start_sample": start, "end_sample_exclusive": end,
                "canonical_full_support_end_sample_exclusive": end + max_ir - 1,
                "source_endpoint_id": es[0]["source_endpoint_id"],
                "target_endpoint_id": es[1]["source_endpoint_id"],
                "target_actor_id": es[1]["actor_id"],
                "support": support,
            })
        # Source stems can be rebound losslessly from the canonical mixture
        # only when the complete impulse-response tails do not overlap.
        for a, b in zip(event_proofs, event_proofs[1:]):
            if a["canonical_full_support_end_sample_exclusive"] > b["start_sample"]:
                raise IdentityNativeError("canonical event tails overlap; event-isolated stems are required")
        proof["layouts"][layout] = {"events": event_proofs, "cache_receipts": [c["receipt_path"] for c in caches]}
    return proof


def _canonical_identity_audio_member(
    source: Mapping[str, Any], target: Mapping[str, Any],
    source_plan: Mapping[str, Any], target_plan: Mapping[str, Any],
    *, root: Path, member_id: str, shared_visual_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse a measured canonical column after physical input equivalence."""
    from avengine.spatial_audio.audio import read_float32_wav, write_float32_wav, sha256_file
    proof = identity_shared_audio_input_equivalence(source, target, source_plan, target_plan)
    proof_path = _write(root / "input_equivalence.json", proof)
    original = _load(Path(source["audio_report"]))
    target_report = _load(Path(target["audio_report"]))
    report = deepcopy(original)
    outputs_by_layout = {}
    for layout, layout_proof in proof["layouts"].items():
        mixture_path = original["outputs_by_layout"][layout]["mixture"]
        wav = read_float32_wav(mixture_path)
        stems = {endpoint: np.zeros_like(wav.samples) for endpoint in target_report["sources"]["source_ids"]}
        for event in layout_proof["events"]:
            lo = int(event["start_sample"])
            hi = min(wav.frame_count, int(event["canonical_full_support_end_sample_exclusive"]))
            stems[event["target_endpoint_id"]][:, lo:hi] += wav.samples[:, lo:hi]
        if not np.array_equal(sum(stems.values()), wav.samples):
            raise IdentityNativeError("canonical event stem rebinding does not reconstruct exact PCM")
        paths = {}
        for endpoint, samples in stems.items():
            path = root / "stems" / layout / f"{endpoint}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_float32_wav(path, samples, wav.sample_rate_hz)
            paths[endpoint] = str(path.resolve())
        outputs_by_layout[layout] = {"mixture": mixture_path, "stems": paths}
        report["audio"]["layout_delivery"][layout]["stems"] = {
            endpoint: {"path": path, "peak_dbfs": (
                20.0 * math.log10(float(np.max(np.abs(stems[endpoint]))))
                if np.any(stems[endpoint]) else None
            )} for endpoint, path in paths.items()
        }
        # Keep the producer's mixture descriptor (path, peak, activity), not
        # just its path: the attached-layout reader consumes that structure.
    report["outputs_by_layout"] = outputs_by_layout
    report["stems"] = outputs_by_layout["binaural"]["stems"]
    report["stem_records"] = {
        endpoint: {"path": path, "peak_dbfs": (
            20.0 * math.log10(float(np.max(np.abs(read_float32_wav(path).samples))))
            if np.any(read_float32_wav(path).samples) else None
        )} for endpoint, path in report["stems"].items()
    }
    report["audio"]["stems"] = deepcopy(report["stems"])
    report["audio"]["stem_records"] = deepcopy(report["stem_records"])
    report["canonical_source_outputs"] = deepcopy(report.get("outputs") or {})
    report["outputs"] = {
        str(path): sha256_file(path)
        for layout in outputs_by_layout.values()
        for path in [layout["mixture"], *layout["stems"].values()]
    }
    target_events = {e["event_id"]: e for e in target_plan["audio_events"]}
    def bind_events(value):
        if isinstance(value, dict):
            eid = value.get("event_id")
            if eid in target_events:
                event = target_events[eid]
                for key in ("actor_id", "voice_binding_actor_id"):
                    if key in value:
                        value[key] = event["actor_id"]
                if "source_endpoint_id" in value:
                    value["source_endpoint_id"] = event["source_endpoint_id"]
                if "output_stem" in value:
                    value["output_stem"] = report["stems"][event["source_endpoint_id"]]
            for child in value.values():
                bind_events(child)
        elif isinstance(value, list):
            for child in value:
                bind_events(child)
    bind_events(report)
    for key in ("audio_program", "audio_program_metadata", "audio_program_path", "audio_program_record"):
        if key in target_report:
            report[key] = deepcopy(target_report[key])
    report["canonical_audio_reuse"] = {
        "status": "pass", "input_equivalence": str(proof_path.resolve()),
        "canonical_source_report": source["audio_report"],
        "independent_target_report": target["audio_report"],
        "assignment_readback": str(Path(target["visual_capture_root"]) / "neutral_readback.json"),
        "native_contexts_created": 0,
        "source_stems": "lossless partition by disjoint full impulse-response support",
    }
    peak_stems = {k: float(np.max(np.abs(read_float32_wav(p).samples))) for k, p in report["stems"].items()}
    qa_gain = report.get("qa", {}).get("event_clock_and_gain", {})
    if "peak_abs_by_stream" in qa_gain:
        qa_gain["peak_abs_by_stream"]["stems"] = peak_stems
    if "peak_dbfs_by_stream" in qa_gain:
        qa_gain["peak_dbfs_by_stream"]["stems"] = {k:(20.0*math.log10(v) if v else None) for k,v in peak_stems.items()}
    report_path = _write(root / "canonical_audio_report.json", report)
    request = deepcopy(target_plan["request"])
    materialized = native.materialize_audio_variant(
        {"capture": target["visual_capture_root"]}, root / "episode",
        target_plan, request, member_id=member_id,
    )
    finalized = native.finalize_audio_assignment(
        materialized, request, audio_report=report_path,
        shared_visual_root=shared_visual_root,
    )
    if finalized["delivered_audio_layouts"]["status"] != "pass":
        raise IdentityNativeError("canonical identity finalization lacks a requested audio layout")
    variant = {
        "facts": finalized["facts"], "audio": finalized["audio"],
        "visual_video": finalized["visual_video"], "audio_report": finalized["audio_report"],
        "visual_capture_root": target["visual_capture_root"],
    }
    evidence = {
        "member_id": member_id, "facts_path": finalized["facts"],
        "audio_path": finalized["audio"], "audio_report_path": finalized["audio_report"],
        "declared_audio_delivery": finalized["declared_audio_delivery"],
        "delivered_audio_layouts": finalized["delivered_audio_layouts"],
        "ancillary_audio_outputs": deepcopy(finalized["result"].get("ancillary_audio_outputs", [])),
        "canonical_reuse": report["canonical_audio_reuse"],
    }
    return variant, evidence


def _run_identity_assembly_unit(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    unit_root: Path,
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del lease
    contract = context["contract"]
    done = _identity_done(results)
    visual_ids = sorted(str(value) for value in contract["visual_units"])
    if visual_ids != ["v0_capture", "v1_capture"]:
        raise IdentityNativeError(
            f"cross_event_identity assembly needs v0_capture/v1_capture, got {visual_ids}"
        )
    visual: dict[str, dict[str, Any]] = {}
    plans: dict[str, str] = {}
    readbacks: dict[str, str] = {}
    requests: dict[str, dict[str, Any]] = {}
    for visual_id in visual_ids:
        row = _identity_result(done, visual_id)
        outputs = row.get("outputs") or {}
        for key in ("capture", "episode_plan", "neutral_readback", "request_path"):
            value = outputs.get(key)
            if (
                not isinstance(value, str)
                or not Path(value).expanduser().resolve().exists()
            ):
                raise IdentityNativeError(
                    f"{visual_id} output lacks readable {key}"
                )
        visual[visual_id] = {
            "capture": str(Path(outputs["capture"]).resolve()),
            "visual_video": outputs.get("visual_video"),
            "neutral_readback": str(
                Path(outputs["neutral_readback"]).resolve()
            ),
        }
        plans[visual_id] = str(Path(outputs["episode_plan"]).resolve())
        readbacks[visual_id] = str(
            Path(outputs["neutral_readback"]).resolve()
        )
        requests[visual_id] = _load(Path(outputs["request_path"]))
    checks = []
    checks.append({
        "check": "planned_world",
        "units": visual_ids,
        **native.compare_group_visual_plans(
            plans["v0_capture"], plans["v1_capture"], contract=contract
        ),
    })
    checks.append({
        "check": "native_readback",
        "units": visual_ids,
        **native.compare_group_native_visuals(
            {"neutral_readback": readbacks["v0_capture"]},
            {"neutral_readback": readbacks["v1_capture"]},
            contract=contract,
            left_unit_id="v0_capture",
            right_unit_id="v1_capture",
        ),
    })
    left_endpoints = _endpoint_bindings(
        {"neutral_readback": readbacks["v0_capture"]},
        _load(Path(plans["v0_capture"])),
    )
    right_endpoints = _endpoint_bindings(
        {"neutral_readback": readbacks["v1_capture"]},
        _load(Path(plans["v1_capture"])),
    )
    if left_endpoints != right_endpoints:
        raise IdentityNativeError(
            "identity visual variants expose different native source endpoints"
        )
    checks.append({
        "check": "native_source_endpoints",
        "units": visual_ids,
        "status": "pass",
        "same": True,
    })
    variants, assignment_plans, member_units = _identity_public_audio_rows(
        context, done
    )
    audio_stage_evidence = []
    for assignment in ("a0", "a1"):
        source_id, target_id = f"v0_{assignment}", f"v1_{assignment}"
        if not _pcm_equal(variants[source_id]["audio"], variants[target_id]["audio"])["same"]:
            variants[target_id], evidence = _canonical_identity_audio_member(
                variants[source_id], variants[target_id],
                assignment_plans[source_id], assignment_plans[target_id],
                root=unit_root / "canonical_audio" / target_id,
                member_id=target_id,
                shared_visual_root=native.shared_visual_evidence_root(output_root, context["group_id"]),
            )
            evidence["group_id"] = context["group_id"]
            audio_stage_evidence.append(evidence)
    corrected_evidence = {row["member_id"]: row for row in audio_stage_evidence}
    audio_stage_evidence = []
    for member_id in ("v0_a0", "v0_a1", "v1_a0", "v1_a1"):
        original = _identity_result(done, member_id)
        outputs = original.get("outputs") or {}
        variant = variants[member_id]
        evidence = corrected_evidence.get(member_id) or {
            "group_id": context["group_id"], "member_id": member_id,
            "audio_path": variant["audio"], "facts_path": variant["facts"],
            "audio_report_path": variant["audio_report"],
            "declared_audio_delivery": deepcopy(outputs.get("declared_audio_delivery") or {}),
            "delivered_audio_layouts": deepcopy(outputs.get("delivered_audio_layouts") or {}),
            "ancillary_audio_outputs": deepcopy(outputs.get("ancillary_audio_outputs") or []),
        }
        evidence["work_item_id"] = item["work_item_id"]
        evidence["source_work_item_id"] = original["work_item_id"]
        evidence["episode_id"] = None
        audio_stage_evidence.append(evidence)
    pcm_by_column = {}
    for assignment in ("a0", "a1"):
        pcm = _pcm_equal(
            variants[f"v0_{assignment}"]["audio"],
            variants[f"v1_{assignment}"]["audio"],
        )
        pcm_by_column[assignment] = pcm
        if not pcm["same"]:
            raise IdentityNativeError(
                f"identity shared audio column {assignment} differs in actual PCM"
            )
    questions = {}
    for member_id, variant in variants.items():
        facts = _load(Path(variant["facts"]))
        questions[member_id] = generate_binding_question(
            facts, FAMILY, DEFAULT_QUERY, seed=f"{context['group_id']}:{member_id}"
        )
    topology_row = _identity_result(done, "identity_topology")
    topology = (topology_row.get("outputs") or {}).get("topology")
    if not isinstance(topology, Mapping):
        topology_path = (topology_row.get("outputs") or {}).get("topology_path")
        if not isinstance(topology_path, str):
            raise IdentityNativeError(
                "identity assembly has no topology result"
            )
        topology = _load(Path(topology_path))
    anchor_request = requests["v0_capture"]
    anchor_plan = _load(Path(plans["v0_capture"]))
    routes = {
        "source": "identity_topology_stage",
        "topology": deepcopy(dict(topology)),
        "measured_probe_wet_tail_end_s": (
            (topology_row.get("outputs") or {}).get(
                "measured_wet_tail_end_s"
            )
        ),
        "visual_variants": {
            visual_id: deepcopy(
                _load(Path(plans[visual_id])).get("identity_intervention")
            )
            for visual_id in visual_ids
        },
    }
    spec = _group_spec(
        context["group_id"],
        context["world_id"],
        anchor_request,
        native.room_family_from_plan(anchor_plan),
        str(anchor_plan["scene"]["room_id"]),
        variants,
        questions,
        routes,
    )
    public_ids = {
        str(row.get("member_id"))
        for row in spec["groups"][0]["members"]
    }
    if public_ids != {"v0_a0", "v0_a1", "v1_a0", "v1_a1"}:
        raise IdentityNativeError(
            "identity assembly leaked an internal probe into public members"
        )
    unit_root.mkdir(parents=True, exist_ok=True)
    spec_path = _write(unit_root / "group_spec.json", spec)
    _ensure_audio_tools()
    assembled = assemble_binding_dataset(
        spec,
        input_base=native.REPOSITORY,
        output=unit_root / "assembled",
        seed=f"{context['group_id']}-identity-assembly",
        verify_media=True,
    )
    group_validation = (
        assembled.get("groups", [{}])[0].get("validation")
        if assembled.get("groups")
        else None
    )
    if not isinstance(group_validation, Mapping) or group_validation.get(
        "status"
    ) != "pass":
        raise IdentityNativeError(
            "identity assembly did not pass media/answer validation"
        )
    return native._stage_result(
        item,
        status="pass",
        facts={
            "group_spec_path": str(spec_path.resolve()),
            "assembled_path": str((unit_root / "assembled").resolve()),
            "validation": {
                "status": "pass",
                "query": deepcopy(DEFAULT_QUERY),
                "internal_probe_excluded": True,
                "controlled_world_checks": checks,
                "pcm_by_column": pcm_by_column,
                "group_count": assembled.get("group_count"),
                "world_count": assembled.get("world_count"),
                "sample_count": assembled.get("sample_count"),
                "media_validation": assembled.get("validation"),
                "group_validation": deepcopy(group_validation),
                "public_payload_check": deepcopy(
                    assembled.get("public_payload_check")
                ),
            },
        },
        outputs={
            "assembled_root": str((unit_root / "assembled").resolve()),
            "audio_stage_evidence": audio_stage_evidence,
            "group_spec": str(spec_path.resolve()),
            "query": deepcopy(DEFAULT_QUERY),
            "internal_probe_excluded": True,
            "pcm_by_column": pcm_by_column,
            "member_sample_ids": {
                str(row.get("member_id")): row.get("sample_id")
                for group in assembled.get("groups", [])
                for row in group.get("members", [])
            },
            "member_units": [list(row) for row in member_units],
            "world_id": context["world_id"],
            "world_id_source": context["world_id_source"],
            "native_visual_worlds_created": sum(
                int((done[visual_id].get("outputs") or {}).get(
                    "native_visual_worlds_created"
                ) or 0)
                for visual_id in visual_ids
            ),
        },
    )


def _recipe_audio_attempt_root(
    previous_attempt_root: str | Path,
    item: Mapping[str, Any],
    lineage: Mapping[str, Any] | None = None,
) -> Path:
    """Find the materialized audio root inside one of this recipe's attempts.

    The generic recovery in binding_group_native accepts ``episode`` or the
    attempt root itself. This recipe materializes its audio under
    ``variants/<unit_id>`` instead, so the unit id resolves the member before
    the shared implementation reads it. Nothing is copied or written here.
    """
    previous = Path(previous_attempt_root).expanduser().resolve()
    unit_ids = []
    for source in (item, lineage or {}):
        if isinstance(source, Mapping):
            for key in ("unit_id", "member_request_id", "request_id"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    unit_ids.append(value.strip())
    candidates = [previous / "episode"]
    candidates.extend(previous / "variants" / unit_id for unit_id in unit_ids)
    candidates.append(previous)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "request.json").is_file():
            return candidate
    raise IdentityNativeError(
        "previous audio attempt has no materialized audio root; looked at "
        + ", ".join(str(path) for path in candidates)
    )


def recover_rendered_audio_attempt(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    output_root: str | Path,
    previous_attempt_root: str | Path,
    lineage: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """CPU-finalize one interrupted audio attempt of this recipe.

    The ordinary runner looks this name up on the recipe module, so the module
    has to carry it or an interrupted member of this family silently gets no
    recovery at all. Only the attempt layout differs from the shared
    implementation, which then does the finalize without launching RLR.
    """
    return native.recover_rendered_audio_attempt(
        item,
        context,
        output_root=output_root,
        previous_attempt_root=_recipe_audio_attempt_root(
            previous_attempt_root, item, lineage,
        ),
        lineage=lineage,
        results=results,
        lease=lease,
    )


IDENTITY_STAGE_RUNNERS = {
    "visual_plan": _run_identity_visual_plan_unit,
    "visual_capture": _run_identity_visual_capture_unit,
    "audio": _run_identity_audio_unit,
    "assembly": _run_identity_assembly_unit,
}
STAGE_RUNNERS = IDENTITY_STAGE_RUNNERS


def run_identity_group_stage_work_item(
    item: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    output_root: str | Path,
    results: Sequence[Mapping[str, Any]] = (),
    lease: Mapping[str, Any] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run identity units through native's fresh/resume/save dispatcher."""
    return native.run_group_stage_work_item(
        item,
        context,
        output_root=output_root,
        results=results,
        lease=lease,
        resume=resume,
        stage_runners=IDENTITY_STAGE_RUNNERS,
    )


__all__ = ["FAMILY", "IdentityNativeError", "recover_rendered_audio_attempt", "IdentityGeometryQueryRequired", "IDENTITY_GEOMETRY_POOL_SCHEMAS", "IDENTITY_GEOMETRY_QUERY_REQUEST_SCHEMA", "IDENTITY_GEOMETRY_QUERY_AUTHORIZATION", "identity_geometry_pool_report", "identity_geometry_query_request", "execute_identity_geometry_query", "adopt_identity_geometry_query_receipt", "prepare_identity_group", "resume_identity_group", "identity_group_stage_context", "load_group_stage_results", "run_identity_group_stage_work_item", "IDENTITY_STAGE_RUNNERS", "STAGE_RUNNERS", "IDENTITY_INTERNAL_UNIT_IDS", "_align_event2_to_rir_boundary", "_select_sound_pair", "_probe_audio", "_select_topology", "_verify_selected_identity_plans"]

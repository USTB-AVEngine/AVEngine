#!/usr/bin/env python3
"""Build a fail-closed CPU plan for 100 independent strict two-human Episodes.

The builder never starts SPEAR, RLR, or a GPU process. It loads a frozen global
assignment and recomputes every balance, motion, geometry, and native actor-root
readback gate with the Python standard library. Mirroring, changing speech, or
changing an answer-option order never creates a new Episode identity.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_REQUEST = (
    REPOSITORY / "examples/qa/native_strict_two_human_full_episode_batch_v1.json"
)
OUTPUT_SCHEMA = "avengine_native_strict_two_human_full_episode_candidate_manifest_v1"
FROZEN_ASSIGNMENT_SCHEMA = "avengine_strict2h_full75_global_assignment_audit_v1"
ASSIGNMENT_VALIDATION_SCHEMA = (
    "avengine_strict2h_full75_global_assignment_validation_v1"
)
FRAME_COUNT = 75
TICKS_PER_FRAME = 3200


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@functools.lru_cache(maxsize=512)
def _load_cached(path: Path) -> dict[str, Any]:
    return _load(path)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY / path).resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _round_point(values: Sequence[float], digits: int = 9) -> list[float]:
    return [round(float(value), digits) for value in values]


def _distance_xz(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def _motion_metrics(path: Sequence[Sequence[float]]) -> dict[str, Any]:
    _require(len(path) == FRAME_COUNT, "motion path is not full75")
    horizontal_path_length_m = sum(
        _distance_xz(previous, current)
        for previous, current in itertools.pairwise(path)
    )
    maximum_displacement_from_start_m = max(
        _distance_xz(path[0], point) for point in path
    )
    unique_root_positions_at_1mm = len(
        {(round(float(point[0]), 3), round(float(point[2]), 3)) for point in path}
    )
    return {
        "horizontal_path_length_m": horizontal_path_length_m,
        "maximum_displacement_from_start_m": maximum_displacement_from_start_m,
        "unique_root_positions_at_1mm": unique_root_positions_at_1mm,
    }


def _actor_motion_passes(
    path: Sequence[Sequence[float]],
    *,
    expected_moving: bool,
    contract: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    metrics = _motion_metrics(path)
    if expected_moving:
        passed = (
            metrics["horizontal_path_length_m"]
            >= float(contract["minimum_moving_horizontal_path_length_m"])
            and metrics["maximum_displacement_from_start_m"]
            >= float(contract["minimum_moving_maximum_displacement_from_start_m"])
            and metrics["unique_root_positions_at_1mm"]
            >= int(contract["minimum_moving_unique_root_positions_at_1mm"])
        )
    else:
        passed = metrics["maximum_displacement_from_start_m"] <= float(
            contract["maximum_static_horizontal_drift_m"]
        )
    return passed, {
        **metrics,
        "expected_moving": expected_moving,
        "status": "pass" if passed else "fail",
    }


def _mechanism_motion_preflight(
    *,
    mechanism: str,
    target_path: Sequence[Sequence[float]],
    distractor_path: Sequence[Sequence[float]],
    camera_yaw_path: Sequence[float],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    target_expected_moving = mechanism in {"target_moves", "both_move"}
    distractor_expected_moving = mechanism in {"distractor_moves", "both_move"}
    target_pass, target = _actor_motion_passes(
        target_path, expected_moving=target_expected_moving, contract=contract
    )
    distractor_pass, distractor = _actor_motion_passes(
        distractor_path, expected_moving=distractor_expected_moving, contract=contract
    )
    camera_pan_degrees = abs(float(camera_yaw_path[-1]) - float(camera_yaw_path[0]))
    camera_expected_moving = mechanism == "camera_pan_both_static"
    camera_pass = (
        camera_pan_degrees >= float(contract["minimum_camera_pan_total_degrees"])
        if camera_expected_moving
        else camera_pan_degrees <= 1.0e-9
    )
    passed = target_pass and distractor_pass and camera_pass
    return {
        "status": "pass" if passed else "fail",
        "mechanism": mechanism,
        "target": target,
        "distractor": distractor,
        "camera": {
            "expected_moving": camera_expected_moving,
            "total_pan_degrees": camera_pan_degrees,
            "status": "pass" if camera_pass else "fail",
        },
    }


def _dynamic_rir_state_budget(
    *,
    target_path: Sequence[Sequence[float]],
    distractor_path: Sequence[Sequence[float]],
    camera_yaw_path: Sequence[float],
) -> dict[str, Any]:
    """Count exact stride-1 source/listener states before RLR materialization.

    The listener position is fixed within each current Apartment candidate. A
    constant per-identity emitter offset cannot change within-slot state
    equality, while the geometry gate keeps the two actors spatially distinct.
    Therefore root position plus listener yaw is an exact count proxy for the
    later source-position/listener-pose cache keys.
    """

    _require(len(camera_yaw_path) == FRAME_COUNT, "camera yaw path is not full75")
    per_source: dict[str, int] = {}
    for source_slot, path in (
        ("source1", target_path),
        ("source2", distractor_path),
    ):
        _require(len(path) == FRAME_COUNT, f"{source_slot} RIR path is not full75")
        per_source[source_slot] = len(
            {
                (
                    tuple(float(value) for value in point),
                    float(camera_yaw_path[frame_index]),
                )
                for frame_index, point in enumerate(path)
            }
        )
    return {
        "status": "pass_exact_stride1_state_count_pending_rir_materialization",
        "frame_stride": 1,
        "requested_source_frame_uses": FRAME_COUNT * 2,
        "expected_unique_rir_state_count": sum(per_source.values()),
        "expected_unique_rir_state_count_by_source_slot": per_source,
        "cache_key_proxy": "source_root_position_plus_listener_yaw_v1",
        "claim_boundary": (
            "exact count for state equality; native RLR jobs and emitter/world "
            "bindings remain pending per-Episode materialization"
        ),
    }


def _storage_budget_summary(
    *,
    resource_budget: Mapping[str, Any],
    episode_count: int,
    exact_rir_state_count: int,
) -> dict[str, Any]:
    capture_bytes_per_episode = int(
        resource_budget["capture_media_reference_bytes_per_full75_episode"]
    )
    capture_budget_gb = float(
        resource_budget["capture_media_budget_gb_for_100_episodes"]
    )
    reference_rir_cache_bytes = int(
        resource_budget["empirical_rir_reference_cache_bytes"]
    )
    reference_rir_state_count = int(
        resource_budget["empirical_rir_reference_state_count"]
    )
    rir_budget_gb = float(
        resource_budget["empirical_rir_cache_budget_gb_for_9080_states"]
    )
    minimum_workspace_gb = float(
        resource_budget["minimum_capture_plus_rir_workspace_gb"]
    )
    _require(episode_count == 100, "storage reference is scoped to exactly 100 Episodes")
    _require(capture_bytes_per_episode > 0, "capture media reference must be positive")
    _require(reference_rir_cache_bytes > 0, "RIR cache reference must be positive")
    _require(reference_rir_state_count > 0, "RIR state reference must be positive")
    _require(exact_rir_state_count == 9080, "storage budget expects 9080 exact RIR states")

    capture_extrapolated_gb = capture_bytes_per_episode * episode_count / 1.0e9
    rir_bytes_per_state = reference_rir_cache_bytes / reference_rir_state_count
    rir_extrapolated_gb = rir_bytes_per_state * exact_rir_state_count / 1.0e9
    _require(
        capture_budget_gb >= capture_extrapolated_gb,
        "capture-media-only budget is below the empirical extrapolation",
    )
    _require(
        rir_budget_gb >= rir_extrapolated_gb,
        "empirical RIR budget is below the measured-rate extrapolation",
    )
    _require(
        minimum_workspace_gb >= capture_budget_gb + rir_budget_gb,
        "minimum workspace omits capture media or RIR cache budget",
    )
    exclusions = resource_budget["workspace_budget_exclusions"]
    _require(
        isinstance(exclusions, list) and len(exclusions) >= 1,
        "workspace budget exclusions must be explicit",
    )
    return {
        "schema": "avengine_strict2h_full75_storage_budget_v1",
        "status": "pass_planning_floor_not_all_intermediates",
        "capture_media_only": {
            "reference_bytes_per_episode": capture_bytes_per_episode,
            "episode_count": episode_count,
            "empirical_extrapolation_decimal_gb": capture_extrapolated_gb,
            "budget_decimal_gb": capture_budget_gb,
            "scope": resource_budget["capture_media_budget_scope"],
        },
        "rir_cache_empirical_budget": {
            "reference_cache_bytes": reference_rir_cache_bytes,
            "reference_state_count": reference_rir_state_count,
            "reference_bytes_per_state": rir_bytes_per_state,
            "planned_exact_state_count": exact_rir_state_count,
            "empirical_extrapolation_decimal_gb": rir_extrapolated_gb,
            "budget_decimal_gb": rir_budget_gb,
            "claim_boundary": (
                "budget estimate from one measured 76-state cache; room, material, "
                "compression, and implementation differences can change realized size"
            ),
        },
        "minimum_capture_plus_rir_workspace_decimal_gb": minimum_workspace_gb,
        "excluded_from_minimum": list(exclusions),
        "claim_boundary": (
            "planning floor for capture media plus exact RIR caches only; excluded "
            "intermediates and failed attempts require additional workspace"
        ),
    }


def _camera_yaw_deg(
    camera: Sequence[float], paths: Sequence[Sequence[Sequence[float]]]
) -> float:
    anchors = [path[index] for path in paths for index in (0, 37, 74)]
    mean_x = sum(float(point[0]) for point in anchors) / len(anchors)
    mean_z = sum(float(point[2]) for point in anchors) / len(anchors)
    return (
        math.degrees(
            math.atan2(-(mean_x - float(camera[0])), -(mean_z - float(camera[2])))
        )
        % 360.0
    )


def _camera_rotation_xyzw(yaw_deg: float) -> list[float]:
    half = math.radians(yaw_deg) / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def _project(
    *,
    camera: Sequence[float],
    yaw_deg: float,
    point: Sequence[float],
    horizontal_fov_deg: float,
    resolution_hw: Sequence[int],
) -> tuple[float, float, float]:
    yaw = math.radians(yaw_deg)
    forward = (-math.sin(yaw), -math.cos(yaw))
    right = (-forward[1], forward[0])
    dx = float(point[0]) - float(camera[0])
    dz = float(point[2]) - float(camera[2])
    depth = dx * forward[0] + dz * forward[1]
    lateral = dx * right[0] + dz * right[1]
    if depth <= 0.0:
        return depth, float("inf"), float("inf")
    tan_horizontal = math.tan(math.radians(horizontal_fov_deg) / 2.0)
    tan_vertical = tan_horizontal * float(resolution_hw[0]) / float(resolution_hw[1])
    x_fraction = 0.5 + lateral / (2.0 * depth * tan_horizontal)
    y_fraction = 0.5 - (
        (float(point[1]) - float(camera[1])) / (2.0 * depth * tan_vertical)
    )
    return depth, x_fraction, y_fraction


def _geometry_metrics(
    *,
    request: Mapping[str, Any],
    camera: Sequence[float],
    yaw_deg: float,
    target_path: Sequence[Sequence[float]],
    distractor_path: Sequence[Sequence[float]],
    target_side: str,
    camera_pan: bool,
) -> dict[str, float] | None:
    contract = request["geometry_contract"]
    hfov = float(contract["horizontal_fov_deg"])
    resolution = contract["resolution_hw"]
    safe_x = [float(value) for value in contract["safe_x_fraction_open_interval"]]
    safe_y = [
        float(value) for value in contract["safe_vertical_fraction_open_interval"]
    ]
    envelope = [
        float(value) for value in contract["human_vertical_envelope_from_root_m"]
    ]
    dead_zone = float(contract["screen_side_dead_zone_fraction"])
    minimum_separation = float(contract["minimum_projected_x_separation_fraction"])
    minimum_depth = float(contract["minimum_camera_depth_m"])
    maximum_depth = float(contract["maximum_camera_depth_m"])
    actor_separation = float(contract["minimum_actor_horizontal_separation_m"])
    pan = float(contract["camera_pan_total_degrees"])
    minimum_x_separation = 1.0
    minimum_actor_distance = float("inf")
    minimum_depth_observed = float("inf")
    maximum_depth_observed = 0.0
    target_x_values: list[float] = []
    distractor_x_values: list[float] = []
    for frame_index in range(FRAME_COUNT):
        frame_yaw = yaw_deg
        if camera_pan:
            frame_yaw += (float(frame_index) / 74.0 - 0.5) * pan
        projected_x: list[float] = []
        for point in (target_path[frame_index], distractor_path[frame_index]):
            depth, x_fraction, y_root = _project(
                camera=camera,
                yaw_deg=frame_yaw,
                point=point,
                horizontal_fov_deg=hfov,
                resolution_hw=resolution,
            )
            if not minimum_depth < depth < maximum_depth:
                return None
            projected_x.append(x_fraction)
            minimum_depth_observed = min(minimum_depth_observed, depth)
            maximum_depth_observed = max(maximum_depth_observed, depth)
            if not safe_x[0] < x_fraction < safe_x[1]:
                return None
            for height in envelope:
                _, _, y_fraction = _project(
                    camera=camera,
                    yaw_deg=frame_yaw,
                    point=[point[0], float(point[1]) + height, point[2]],
                    horizontal_fov_deg=hfov,
                    resolution_hw=resolution,
                )
                if not safe_y[0] < y_fraction < safe_y[1]:
                    return None
            if not safe_y[0] < y_root < safe_y[1]:
                return None
        target_x, distractor_x = projected_x
        if target_side == "left":
            if not (target_x < 0.5 - dead_zone and distractor_x > 0.5 + dead_zone):
                return None
        else:
            if not (target_x > 0.5 + dead_zone and distractor_x < 0.5 - dead_zone):
                return None
        separation = abs(target_x - distractor_x)
        if separation < minimum_separation:
            return None
        distance = _distance_xz(target_path[frame_index], distractor_path[frame_index])
        if distance < actor_separation:
            return None
        minimum_x_separation = min(minimum_x_separation, separation)
        minimum_actor_distance = min(minimum_actor_distance, distance)
        target_x_values.append(target_x)
        distractor_x_values.append(distractor_x)
    return {
        "minimum_projected_x_separation_fraction": minimum_x_separation,
        "minimum_actor_horizontal_separation_m": minimum_actor_distance,
        "minimum_camera_depth_m": minimum_depth_observed,
        "maximum_camera_depth_m": maximum_depth_observed,
        "target_x_fraction_minimum": min(target_x_values),
        "target_x_fraction_maximum": max(target_x_values),
        "distractor_x_fraction_minimum": min(distractor_x_values),
        "distractor_x_fraction_maximum": max(distractor_x_values),
    }


def _source_inventory(
    suite: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[float, float, float], dict[str, Any]],
    dict[str, Mapping[str, Any]],
]:
    trajectories: list[dict[str, Any]] = []
    point_records: dict[tuple[float, float, float], dict[str, Any]] = {}
    scenario_by_id: dict[str, Mapping[str, Any]] = {}
    for scenario in suite["scenarios"]:
        scenario_id = str(scenario["scenario_id"])
        scenario_by_id[scenario_id] = scenario
        paths: dict[str, list[list[float]]] = {
            "source1_actor": [],
            "source2_actor": [],
        }
        states: dict[str, list[dict[str, Any]]] = {
            "source1_actor": [],
            "source2_actor": [],
        }
        frames = scenario["plan"]["frames"]
        _require(len(frames) == FRAME_COUNT, f"{scenario_id}: source frame drift")
        for frame in frames:
            frame_index = int(frame["frame_index"])
            _require(
                frame_index == len(paths["source1_actor"]),
                f"{scenario_id}: frame order drift",
            )
            actors = {str(item["actor_id"]): item for item in frame["actor_states"]}
            _require(
                set(actors) == {"source1_actor", "source2_actor"},
                f"{scenario_id}: source actor closure drift",
            )
            for actor_id in sorted(actors):
                state = actors[actor_id]
                point = _round_point(state["translation_m"])
                paths[actor_id].append(point)
                states[actor_id].append(deepcopy(state))
                key = tuple(round(value, 6) for value in point)
                point_records.setdefault(
                    key,
                    {
                        "scenario_id": scenario_id,
                        "frame_index": frame_index,
                        "actor_id": actor_id,
                        "point_m": point,
                    },
                )
        trajectories.append(
            {
                "scenario_id": scenario_id,
                "paths": paths,
                "states": states,
            }
        )
    return trajectories, point_records, scenario_by_id


def _camera_candidates(
    point_records: Mapping[tuple[float, float, float], Mapping[str, Any]],
    *,
    cell_size: float,
) -> dict[str, list[dict[str, Any]]]:
    by_cell: dict[tuple[int, int], list[tuple[float, dict[str, Any]]]] = {}
    for record in point_records.values():
        point = record["point_m"]
        cell = (
            math.floor(float(point[0]) / cell_size),
            math.floor(float(point[2]) / cell_size),
        )
        center = ((cell[0] + 0.5) * cell_size, (cell[1] + 0.5) * cell_size)
        distance = (float(point[0]) - center[0]) ** 2 + (
            float(point[2]) - center[1]
        ) ** 2
        by_cell.setdefault(cell, []).append((distance, dict(record)))
    representatives = [
        {
            **min(records, key=lambda item: (item[0], item[1]["scenario_id"]))[1],
            "cell": cell,
        }
        for cell, records in by_cell.items()
    ]
    representatives.sort(key=lambda item: (item["point_m"][2], item["point_m"][0]))
    _require(len(representatives) >= 100, "fewer than 100 native camera clusters")
    strata: dict[str, list[dict[str, Any]]] = {
        f"stratum_{index + 1:02d}": [] for index in range(5)
    }
    for rank, record in enumerate(representatives):
        stratum_index = min(4, rank * 5 // len(representatives))
        strata[f"stratum_{stratum_index + 1:02d}"].append(record)
    return strata


def _camera_cluster_id(record: Mapping[str, Any]) -> str:
    cell_x, cell_z = record["cell"]
    return f"apartment_grid075_x{int(cell_x):+03d}_z{int(cell_z):+03d}"


def _readback_world_point(record: Mapping[str, Any], suite_root: Path) -> list[float]:
    path = suite_root / str(record["scenario_id"]) / "runtime_readbacks.json"
    value = _load_cached(path)
    matches = [
        item
        for item in value["actor_roots"][str(record["actor_id"])]
        if int(item["frame_index"]) == int(record["frame_index"])
    ]
    _require(len(matches) == 1, f"native readback did not resolve once: {record}")
    location_cm = [float(value) for value in matches[0]["location_cm"]]
    return [location_cm[0] / 100.0, location_cm[2] / 100.0, location_cm[1] / 100.0]


def _validate_provenance(
    *,
    camera_record: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    target_actor_id: str,
    distractor_actor_id: str,
    target_frame_map: Sequence[int],
    distractor_frame_map: Sequence[int],
    suite_root: Path,
) -> dict[str, Any]:
    maximum_drift = 0.0
    checked = 0
    camera_observed = _readback_world_point(camera_record, suite_root)
    maximum_drift = max(
        maximum_drift,
        max(
            abs(camera_observed[index] - float(camera_record["point_m"][index]))
            for index in range(3)
        ),
    )
    checked += 1
    for actor_id, frame_map in (
        (target_actor_id, target_frame_map),
        (distractor_actor_id, distractor_frame_map),
    ):
        unique_indices = sorted({int(index) for index in frame_map})
        for frame_index in unique_indices:
            state = trajectory["states"][actor_id][frame_index]
            record = {
                "scenario_id": trajectory["scenario_id"],
                "frame_index": frame_index,
                "actor_id": actor_id,
                "point_m": state["translation_m"],
            }
            observed = _readback_world_point(record, suite_root)
            maximum_drift = max(
                maximum_drift,
                max(
                    abs(observed[index] - float(state["translation_m"][index]))
                    for index in range(3)
                ),
            )
            checked += 1
    _require(maximum_drift <= 1.0e-6, "native root/readback provenance drift")
    return {
        "status": "pass_exact_native_root_readbacks",
        "checked_record_count": checked,
        "maximum_location_drift_m": maximum_drift,
    }


def _frame_maps(mechanism: str, hold_frame: int) -> tuple[list[int], list[int]]:
    moving = list(range(FRAME_COUNT))
    held = [hold_frame] * FRAME_COUNT
    if mechanism in {"both_static", "camera_pan_both_static"}:
        return held, held
    if mechanism == "target_moves":
        return moving, held
    if mechanism == "distractor_moves":
        return held, moving
    if mechanism == "both_move":
        return moving, moving
    raise ValueError(f"unsupported mechanism: {mechanism}")


def _path(
    trajectory: Mapping[str, Any], actor_id: str, frame_map: Sequence[int]
) -> list[list[float]]:
    return [
        _round_point(trajectory["paths"][actor_id][int(frame_index)])
        for frame_index in frame_map
    ]


def _identity_metadata(
    strict: Mapping[str, Any], sounds: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    assets = {str(item["sound_asset_id"]): item for item in sounds["assets"]}
    result: dict[str, dict[str, Any]] = {}
    for key, identity in strict["approved_identity_catalog"].items():
        sound_id = str(identity["sound_asset_id"])
        _require(sound_id in assets, f"controlled sound missing: {sound_id}")
        sound = assets[sound_id]
        _require(
            sound["content"]["transcript"] == identity["transcript"],
            f"{key}: transcript drift",
        )
        result[key] = {
            "identity_key": key,
            "identity_id": identity["original_identity_id"],
            "runtime_asset_id": identity["runtime_asset_id"],
            "runtime_revision": identity["runtime_revision"],
            "sound_asset_id": sound_id,
            "voice_id": sound["content"]["speaker_id"],
            "content_id": sound["content"]["statement_id"],
            "transcript": sound["content"]["transcript"],
            "speech_sample_count": int(sound["audio"]["sample_count"]),
            "speech_frame_window_inclusive": identity[
                "expected_speech_frame_window_inclusive"
            ],
            "listening_review": sound["listening_review"]["state"],
            "rights_status": sound["license"]["rights_status"],
        }
    return result


def _candidate_cases(
    trajectory: Mapping[str, Any], mechanism: str
) -> list[tuple[str, str, int]]:
    cases: list[tuple[str, str, int]] = []
    for hold in (0, 15, 37, 60, 74):
        cases.extend(
            [
                ("source1_actor", "source2_actor", hold),
                ("source2_actor", "source1_actor", hold),
            ]
        )
    if mechanism == "both_move":
        return cases[:2]
    return cases


def _validate_frozen_assignment_structure(
    request: Mapping[str, Any], assignment: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Validate the frozen global assignment without a solver dependency.

    The assignment is allowed to choose any feasible mechanism/stratum pairing.
    Balance is enforced globally and within each execution batch instead of by
    the old per-row mechanism-to-stratum formula, which excluded valid graphs.
    """

    _require(
        assignment.get("schema") == FROZEN_ASSIGNMENT_SCHEMA,
        "frozen assignment schema drift",
    )
    _require(
        assignment.get("status") == "pass_cpu_graph_assignment_not_native_execution",
        "frozen assignment status drift",
    )
    _require(
        assignment.get("formal_episode_count") == 0,
        "frozen assignment formal count must remain zero",
    )
    _require(
        assignment.get("qualification_claim") is False,
        "frozen assignment qualification claim forbidden",
    )
    episode_count = int(request["output_contract"]["episode_count"])
    batch_size = int(request["output_contract"]["batch_size"])
    batch_count = int(request["output_contract"]["batch_count"])
    rows_value = assignment.get("rows")
    _require(isinstance(rows_value, list), "frozen assignment rows missing")
    rows = [dict(row) for row in rows_value]
    _require(
        int(assignment.get("episode_count", -1)) == episode_count,
        "frozen assignment episode count drift",
    )
    _require(len(rows) == episode_count, "frozen assignment row count drift")
    _require(
        episode_count == batch_size * batch_count,
        "request batch dimensions do not close",
    )
    _require(
        sorted(int(row["episode_index"]) for row in rows)
        == list(range(1, episode_count + 1)),
        "frozen assignment episode index closure failed",
    )
    rows.sort(key=lambda row: int(row["episode_index"]))

    mechanisms = list(request["mechanism_schedule"])
    ordered_pairs = list(request["balance_contract"]["ordered_identity_pairs"])
    expected_strata = set(request["balance_contract"]["spatial_stratum_counts"])
    required_fields = {
        "episode_index",
        "batch_id",
        "mechanism",
        "target_side",
        "identity_pair",
        "stratum_id",
        "native_source_scenario_id",
        "camera_cluster_id",
        "target_actor_id",
        "distractor_actor_id",
        "hold_frame",
    }
    for zero, row in enumerate(rows):
        _require(
            required_fields <= set(row),
            f"assignment row {zero + 1}: fields missing",
        )
        _require(
            int(row["episode_index"]) == zero + 1,
            f"assignment row {zero + 1}: index drift",
        )
        _require(
            row["batch_id"] == f"batch_{zero // batch_size + 1:02d}",
            f"assignment row {zero + 1}: batch drift",
        )
        _require(
            row["mechanism"] == mechanisms[zero % len(mechanisms)],
            f"assignment row {zero + 1}: mechanism slot drift",
        )
        _require(
            row["target_side"] == ("left" if zero % 2 == 0 else "right"),
            f"assignment row {zero + 1}: side slot drift",
        )
        _require(
            row["identity_pair"] == ordered_pairs[zero % len(ordered_pairs)],
            f"assignment row {zero + 1}: identity-pair slot drift",
        )
        _require(
            row["stratum_id"] in expected_strata,
            f"assignment row {zero + 1}: unknown stratum",
        )
        _require(
            {row["target_actor_id"], row["distractor_actor_id"]}
            == {"source1_actor", "source2_actor"},
            f"assignment row {zero + 1}: actor closure drift",
        )

    _require(
        Counter(row["mechanism"] for row in rows)
        == Counter(request["balance_contract"]["mechanism_counts"]),
        "frozen assignment mechanism balance drift",
    )
    _require(
        Counter(row["target_side"] for row in rows)
        == Counter(request["balance_contract"]["target_side_counts"]),
        "frozen assignment side balance drift",
    )
    _require(
        Counter(row["stratum_id"] for row in rows)
        == Counter(request["balance_contract"]["spatial_stratum_counts"]),
        "frozen assignment spatial balance drift",
    )
    _require(
        len({str(row["native_source_scenario_id"]) for row in rows})
        == episode_count,
        "frozen assignment source reuse",
    )
    _require(
        len({str(row["camera_cluster_id"]) for row in rows}) == episode_count,
        "frozen assignment camera reuse",
    )
    expected_mechanism_side = Counter(
        {(mechanism, side): 1 for mechanism in mechanisms for side in ("left", "right")}
    )
    expected_batch_strata = Counter({stratum: 2 for stratum in expected_strata})
    for batch_number in range(1, batch_count + 1):
        batch_id = f"batch_{batch_number:02d}"
        batch_rows = [row for row in rows if row["batch_id"] == batch_id]
        _require(len(batch_rows) == batch_size, f"{batch_id}: row count drift")
        _require(
            Counter((row["mechanism"], row["target_side"]) for row in batch_rows)
            == expected_mechanism_side,
            f"{batch_id}: each mechanism must have one left and one right row",
        )
        _require(
            Counter(row["stratum_id"] for row in batch_rows)
            == expected_batch_strata,
            f"{batch_id}: stratum balance drift",
        )
    return rows


def _resolve_frozen_assignment(
    *,
    request: Mapping[str, Any],
    assignment: Mapping[str, Any],
    suite_path: Path,
    trajectories: Sequence[dict[str, Any]],
    camera_strata: Mapping[str, Sequence[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _validate_frozen_assignment_structure(request, assignment)
    _require(
        Path(str(assignment["source_suite"])).resolve() == suite_path.resolve(),
        "frozen assignment source suite drift",
    )
    trajectory_by_id = {
        str(trajectory["scenario_id"]): trajectory for trajectory in trajectories
    }
    camera_by_id: dict[str, dict[str, Any]] = {}
    camera_stratum: dict[str, str] = {}
    for stratum_id, records in camera_strata.items():
        for record in records:
            cluster_id = _camera_cluster_id(record)
            _require(cluster_id not in camera_by_id, "camera cluster collision")
            camera_by_id[cluster_id] = record
            camera_stratum[cluster_id] = stratum_id

    resolved: list[dict[str, Any]] = []
    moving_path_lengths: list[float] = []
    minimum_actor_separation = float("inf")
    maximum_provenance_drift = 0.0
    suite_root = suite_path.parent
    for row in rows:
        episode_number = int(row["episode_index"])
        scenario_id = str(row["native_source_scenario_id"])
        cluster_id = str(row["camera_cluster_id"])
        _require(
            scenario_id in trajectory_by_id,
            f"assignment row {episode_number}: unknown source scenario",
        )
        _require(
            cluster_id in camera_by_id,
            f"assignment row {episode_number}: unknown camera cluster",
        )
        trajectory = trajectory_by_id[scenario_id]
        camera_record = camera_by_id[cluster_id]
        _require(
            camera_stratum[cluster_id] == row["stratum_id"],
            f"assignment row {episode_number}: camera stratum drift",
        )
        _require(
            str(camera_record["scenario_id"]) != scenario_id,
            f"assignment row {episode_number}: camera/source scenario alias",
        )
        target_actor_id = str(row["target_actor_id"])
        distractor_actor_id = str(row["distractor_actor_id"])
        hold_frame = int(row["hold_frame"])
        mechanism = str(row["mechanism"])
        choice = (target_actor_id, distractor_actor_id, hold_frame)
        _require(
            choice in _candidate_cases(trajectory, mechanism),
            f"assignment row {episode_number}: candidate-case drift",
        )
        target_map, distractor_map = _frame_maps(mechanism, hold_frame)
        target_path = _path(trajectory, target_actor_id, target_map)
        distractor_path = _path(trajectory, distractor_actor_id, distractor_map)
        camera = [
            float(camera_record["point_m"][0]),
            float(request["geometry_contract"]["camera_height_m"]),
            float(camera_record["point_m"][2]),
        ]
        yaw = _camera_yaw_deg(camera, [target_path, distractor_path])
        pan = float(request["geometry_contract"]["camera_pan_total_degrees"])
        camera_yaw_path = [
            yaw
            + (
                (frame_index / 74.0 - 0.5) * pan
                if mechanism == "camera_pan_both_static"
                else 0.0
            )
            for frame_index in range(FRAME_COUNT)
        ]
        motion = _mechanism_motion_preflight(
            mechanism=mechanism,
            target_path=target_path,
            distractor_path=distractor_path,
            camera_yaw_path=camera_yaw_path,
            contract=request["motion_contract"],
        )
        _require(
            motion["status"] == "pass",
            f"assignment row {episode_number}: motion failure",
        )
        for role in ("target", "distractor"):
            if motion[role]["expected_moving"]:
                moving_path_lengths.append(
                    float(motion[role]["horizontal_path_length_m"])
                )
        geometry = _geometry_metrics(
            request=request,
            camera=camera,
            yaw_deg=yaw,
            target_path=target_path,
            distractor_path=distractor_path,
            target_side=str(row["target_side"]),
            camera_pan=mechanism == "camera_pan_both_static",
        )
        _require(
            geometry is not None,
            f"assignment row {episode_number}: geometry failure",
        )
        minimum_actor_separation = min(
            minimum_actor_separation,
            float(geometry["minimum_actor_horizontal_separation_m"]),
        )
        provenance = _validate_provenance(
            camera_record=camera_record,
            trajectory=trajectory,
            target_actor_id=target_actor_id,
            distractor_actor_id=distractor_actor_id,
            target_frame_map=target_map,
            distractor_frame_map=distractor_map,
            suite_root=suite_root,
        )
        _require(
            provenance["status"] == "pass_exact_native_root_readbacks",
            f"assignment row {episode_number}: provenance failure",
        )
        maximum_provenance_drift = max(
            maximum_provenance_drift,
            float(provenance["maximum_location_drift_m"]),
        )
        resolved.append(
            {
                "assignment": row,
                "camera_record": camera_record,
                "camera": camera,
                "cluster_id": cluster_id,
                "trajectory": trajectory,
                "target_actor_id": target_actor_id,
                "distractor_actor_id": distractor_actor_id,
                "target_map": target_map,
                "distractor_map": distractor_map,
                "target_path": target_path,
                "distractor_path": distractor_path,
                "yaw": yaw,
                "camera_yaw_path": camera_yaw_path,
                "metrics": geometry,
                "motion_preflight": motion,
                "provenance": provenance,
            }
        )

    _require(moving_path_lengths, "frozen assignment contains no moving actor paths")
    validation = {
        "schema": ASSIGNMENT_VALIDATION_SCHEMA,
        "status": "pass_exact_cpu_assignment_not_native_execution",
        "assignment_mode": "frozen_global_assignment_pure_stdlib_revalidation",
        "solver_required_at_builder_runtime": False,
        "fixed_mechanism_stratum_cross_quota_required": False,
        "episode_count": len(rows),
        "batch_count": int(request["output_contract"]["batch_count"]),
        "unique_source_scenario_count": len(
            {str(row["native_source_scenario_id"]) for row in rows}
        ),
        "unique_camera_cluster_count": len(
            {str(row["camera_cluster_id"]) for row in rows}
        ),
        "mechanism_counts": dict(
            sorted(Counter(row["mechanism"] for row in rows).items())
        ),
        "target_side_counts": dict(
            sorted(Counter(row["target_side"] for row in rows).items())
        ),
        "stratum_counts": dict(
            sorted(Counter(row["stratum_id"] for row in rows).items())
        ),
        "minimum_moving_path_length_m": min(moving_path_lengths),
        "minimum_actor_horizontal_separation_m": minimum_actor_separation,
        "maximum_native_source_root_readback_drift_m": maximum_provenance_drift,
        "formal_episode_count": 0,
        "qualification_claim": False,
        "remaining_gates": [
            "dynamic suite materialization with human runtime states",
            "exact acoustic plan and binaural render",
            "native per-frame human and camera readbacks",
            "body/scene intersection audit because capture collision is disabled",
            "machine visibility/depth/mask gates",
            "human visual and audio review",
        ],
    }
    return resolved, validation


def _question(index: int, side: str, episode_id: str) -> dict[str, Any]:
    normal_order = ((index // 2) % 2) == 0
    options = ["Left", "Right"] if normal_order else ["Right", "Left"]
    answer = side.title()
    return {
        "question_id": f"{episode_id}__speaker_side",
        "prompt": "Which visible person is speaking?",
        "options": options,
        "correct_index": options.index(answer),
        "answer": answer,
        "option_order_id": "left_right" if normal_order else "right_left",
    }


def _canary_plan(
    request: Mapping[str, Any], publication: Mapping[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in publication["rows"][:4]:
        row_index = int(row["row_index"])
        if row_index == 1:
            suite = (
                REPOSITORY
                / "tmp/lead_d_strict_two_human_canary_v1/final_gate_v1/suite_execution_plan.json"
            )
            audio = REPOSITORY / (
                "tmp/lead_d_strict_two_human_canary_v1/binaural_v5/audio/binaural/"
                "rocketbox_male_female__strict_two_human_canary_v1__v00.wav"
            )
            acoustic_evidence = {
                "exact_rir_plan": str(
                    (
                        REPOSITORY
                        / "tmp/lead_d_strict_two_human_canary_v1/exact_rir_plan_v3/rir_job_plan.json"
                    ).resolve()
                ),
                "rir_cache": str(
                    (
                        REPOSITORY
                        / "tmp/lead_d_strict_two_human_canary_v1/exact_rir_cache_v4/receipt.json"
                    ).resolve()
                ),
                "binaural_delivery": str(
                    (
                        REPOSITORY
                        / "tmp/lead_d_strict_two_human_canary_v1/binaural_v5/delivery.json"
                    ).resolve()
                ),
            }
        else:
            gate = _load(_resolve(row["cpu_gate"]))
            suite = Path(gate["suite_plan"])
            audio = Path(gate["audio_wav"])
            acoustic_evidence = {
                key: value["path"]
                for key, value in gate["cpu_acoustic_evidence"].items()
            }
        _require(suite.is_file(), f"canary suite missing: {suite}")
        _require(audio.is_file(), f"canary audio missing: {audio}")
        output = REPOSITORY / (
            "tmp/lead_a_strict_two_human_full_episode_batch_v1/full75_canaries/"
            f"{row['row_id']}"
        )
        rows.append(
            {
                "canary_index": row_index,
                "row_id": row["row_id"],
                "episode_id": row["episode_id"],
                "target_identity_key": row["target_identity_key"],
                "distractor_identity_key": row["distractor_identity_key"],
                "target_side": row["target_side"],
                "speech_frame_window_inclusive": row["speech_frame_window_inclusive"],
                "suite_plan": str(suite.resolve()),
                "audio_wav": str(audio.resolve()),
                "acoustic_evidence": acoustic_evidence,
                "output_root": str(output.resolve()),
                "capture_argv": [
                    "/data/jzy/miniconda3/envs/spear-env/bin/python",
                    "tools/qa/capture_spear_native_pixel_episode.py",
                    "--suite-plan",
                    str(suite.resolve()),
                    "--scenario-id",
                    row["episode_id"],
                    "--audio-wav",
                    str(audio.resolve()),
                    "--spear-root",
                    "/data/jzy/code/SPEAR-lead-b",
                    "--output",
                    str(output.resolve()),
                    "--rpc-port",
                    str(39610 + row_index),
                    "--graphics-adapter",
                    "1",
                ],
            }
        )
    return {
        "schema": "avengine_native_strict_two_human_full75_canary_plan_v1",
        "status": "ready_pending_gpu1_idle_gate",
        "claim_boundary": "Four existing sparse-passed rows promoted to full75 only after fresh complete native capture and finalization.",
        "full_batch_authorized": False,
        "gpu_policy": request["gpu_policy"],
        "canaries": rows,
    }


def _dynamic_mechanism_canary_plan(
    request: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    mechanisms = [
        "target_moves",
        "distractor_moves",
        "both_move",
        "camera_pan_both_static",
    ]
    pilot_rows = [item for item in episodes if int(item["episode_index"]) <= 20]
    selected = [
        next(item for item in pilot_rows if item["mechanism"] == mechanism)
        for mechanism in mechanisms
    ]
    _require(
        all(item["motion_preflight"]["status"] == "pass" for item in selected),
        "dynamic canary motion preflight failed",
    )
    return {
        "schema": "avengine_native_strict_two_human_dynamic_full75_canary_plan_v1",
        "status": "pass_cpu_selection_pending_suite_acoustics_and_gpu1",
        "static_pipeline_canary_pass_count": 4,
        "dynamic_mechanism_canary_pass_count": 0,
        "dynamic_mechanism_canary_required_count": 4,
        "single_room_mechanism_pilot_authorized": False,
        "formal_episode_count": 0,
        "qualification_claim": False,
        "gpu_policy": request["gpu_policy"],
        "canaries": [
            {
                "execution_order": order,
                "episode_index": row["episode_index"],
                "episode_id": row["episode_id"],
                "mechanism": row["mechanism"],
                "target_identity_key": row["target"]["identity_key"],
                "distractor_identity_key": row["distractor"]["identity_key"],
                "target_side": row["target"]["side"],
                "speech_frame_window_inclusive": row["target"][
                    "speech_frame_window_inclusive"
                ],
                "motion_preflight": row["motion_preflight"],
                "projection_preflight": row["projection_preflight"],
                "native_source_scenario_id": row["native_source_scenario_id"],
                "camera_cluster_id": row["camera_cluster_id"],
                "suite_plan": "PENDING_DYNAMIC_SUITE_MATERIALIZATION",
                "audio_wav": "PENDING_EXACT_RIR_AND_BINAURAL_RENDER",
                "exact_rir_plan": "PENDING_DYNAMIC_EXACT_RIR_PLAN",
                "capture_output": str(
                    (
                        REPOSITORY
                        / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_full75_canaries"
                        / row["episode_id"]
                    ).resolve()
                ),
                "rpc_port": 39700 + order,
                "physical_gpu_index": 1,
                "graphics_adapter_argument": 1,
                "status": "cpu_preflight_pass_pending_suite_acoustics_and_gpu1",
            }
            for order, row in enumerate(selected, start=1)
        ],
    }


def build(request_path: Path, output: Path) -> dict[str, Path]:
    _require(not output.exists(), f"refusing to overwrite output: {output}")
    request = _load(request_path)
    _require(
        request.get("schema")
        == "avengine_native_strict_two_human_full_episode_batch_request_v1",
        "request schema drift",
    )
    _require(request.get("formal_episode_count") == 0, "formal count must remain zero")
    _require(request.get("qualification_claim") is False, "qualification forbidden")
    timeline = request["timeline"]
    _require(
        timeline
        == {
            "frame_count": 75,
            "frame_rate_hz": 15,
            "duration_seconds": 5,
            "sample_rate_hz": 16000,
            "sample_count": 80000,
            "target_speech_start_sample": 7467,
        },
        "timeline contract drift",
    )
    gpu = request["gpu_policy"]
    _require(
        gpu["physical_gpu_index"] == 1
        and gpu["graphics_adapter_argument"] == 1
        and gpu["forbidden_physical_gpu_indices"] == [0, 3]
        and gpu["cpu_builder_must_not_launch_gpu"] is True,
        "GPU1-only policy drift",
    )
    strict = _load(_resolve(request["inputs"]["strict_sparse_contract"]))
    publication = _load(_resolve(request["inputs"]["strict_sparse_publication"]))
    sounds = _load(_resolve(request["inputs"]["controlled_sound_registry"]))
    identities = _identity_metadata(strict, sounds)
    suite_path = _resolve(request["inputs"]["native_floor_point_suite"])
    suite = _load(suite_path)
    _require(
        len(suite["scenarios"]) == 1000,
        "native source suite must contain 1000 Episodes",
    )
    _require(
        suite["native_map"] == "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000",
        "native map drift",
    )
    trajectories, points, _ = _source_inventory(suite)
    cell_size = float(request["independence_contract"]["camera_cluster_cell_size_m"])
    camera_strata = _camera_candidates(points, cell_size=cell_size)
    assignment_path = _resolve(request["inputs"]["frozen_global_assignment"])
    assignment = _load(assignment_path)
    resolved_assignment, assignment_validation = _resolve_frozen_assignment(
        request=request,
        assignment=assignment,
        suite_path=suite_path,
        trajectories=trajectories,
        camera_strata=camera_strata,
    )

    episodes: list[dict[str, Any]] = []
    used_source_scenarios: set[str] = set()
    used_camera_clusters: set[str] = set()
    used_dedup_keys: set[str] = set()
    for found in resolved_assignment:
        row = found["assignment"]
        episode_number = int(row["episode_index"])
        index = episode_number - 1
        batch_number = index // int(request["output_contract"]["batch_size"]) + 1
        mechanism = str(row["mechanism"])
        target_side = str(row["target_side"])
        pair = str(row["identity_pair"])
        target_key, distractor_key = pair.split("/")
        stratum_id = str(row["stratum_id"])
        episode_id = f"strict2h_full75_{episode_number:04d}_v1"
        target = identities[target_key]
        distractor = identities[distractor_key]
        target_binding = (
            f"{found['trajectory']['scenario_id']}/{found['target_actor_id']}/"
            f"{'full75' if len(set(found['target_map'])) > 1 else 'hold_f' + str(found['target_map'][0])}"
        )
        distractor_binding = (
            f"{found['trajectory']['scenario_id']}/{found['distractor_actor_id']}/"
            f"{'full75' if len(set(found['distractor_map'])) > 1 else 'hold_f' + str(found['distractor_map'][0])}"
        )
        dedup_key = {
            "room_id": request["native_room_scope"]["room_id"],
            "camera_cluster_id": found["cluster_id"],
            "native_source_scenario_id": found["trajectory"]["scenario_id"],
            "target_path_binding": target_binding,
            "distractor_path_binding": distractor_binding,
            "mechanism": mechanism,
        }
        dedup_key_text = "|".join(
            str(dedup_key[key])
            for key in request["independence_contract"]["dedup_key_fields"]
        )
        _require(dedup_key_text not in used_dedup_keys, "duplicate Episode key")
        used_dedup_keys.add(dedup_key_text)
        used_source_scenarios.add(str(found["trajectory"]["scenario_id"]))
        used_camera_clusters.add(str(found["cluster_id"]))
        camera_yaw_path = found["camera_yaw_path"]
        motion_preflight = found["motion_preflight"]
        dynamic_rir_state_budget = _dynamic_rir_state_budget(
            target_path=found["target_path"],
            distractor_path=found["distractor_path"],
            camera_yaw_path=camera_yaw_path,
        )
        question = _question(index, target_side, episode_id)
        episodes.append(
            {
                "episode_index": episode_number,
                "episode_id": episode_id,
                "batch_id": f"batch_{batch_number:02d}",
                "phase": (
                    "single_room_mechanism_pilot_20"
                    if episode_number <= 20
                    else "interim_single_room_candidate_not_final_multi_room_100"
                ),
                "status": "cpu_feasible_pending_exact_rir_sparse_and_full75_native",
                "formal": False,
                "qualification_claim": False,
                "room_id": request["native_room_scope"]["room_id"],
                "scene_id": request["native_room_scope"]["scene_id"],
                "room_region_id": stratum_id,
                "renderer_backend": request["native_room_scope"]["renderer_backend"],
                "mechanism": mechanism,
                "native_source_scenario_id": found["trajectory"]["scenario_id"],
                "camera_cluster_id": found["cluster_id"],
                "camera_pose_id": f"{episode_id}__camera_f000",
                "camera": {
                    "translation_m": _round_point(found["camera"]),
                    "rotation_xyzw": _camera_rotation_xyzw(found["yaw"]),
                    "habitat_yaw_deg": found["yaw"],
                    "yaw_path_deg": camera_yaw_path,
                    "horizontal_fov_deg": request["geometry_contract"][
                        "horizontal_fov_deg"
                    ],
                    "provenance": {
                        "scenario_id": found["camera_record"]["scenario_id"],
                        "frame_index": found["camera_record"]["frame_index"],
                        "actor_id": found["camera_record"]["actor_id"],
                    },
                },
                "target": {
                    **target,
                    "source_slot_id": "source1",
                    "side": target_side,
                    "voice_policy": "speaking",
                    "path_binding": target_binding,
                    "source_actor_id": found["target_actor_id"],
                    "frame_index_map": found["target_map"],
                    "root_path_m": found["target_path"],
                },
                "distractor": {
                    "identity_key": distractor["identity_key"],
                    "identity_id": distractor["identity_id"],
                    "runtime_asset_id": distractor["runtime_asset_id"],
                    "runtime_revision": distractor["runtime_revision"],
                    "source_slot_id": "source2",
                    "side": "right" if target_side == "left" else "left",
                    "voice_policy": "silent",
                    "voice_id": None,
                    "content_id": None,
                    "path_binding": distractor_binding,
                    "source_actor_id": found["distractor_actor_id"],
                    "frame_index_map": found["distractor_map"],
                    "root_path_m": found["distractor_path"],
                },
                "timeline": request["timeline"],
                "audio_program": {
                    "mode": "one_active_of_n",
                    "active_source_slots": ["source1"],
                    "silent_source_slots": ["source2"],
                    "target_event": {
                        "sound_asset_id": target["sound_asset_id"],
                        "voice_id": target["voice_id"],
                        "content_id": target["content_id"],
                        "start_sample": request["timeline"][
                            "target_speech_start_sample"
                        ],
                        "end_sample_exclusive": request["timeline"][
                            "target_speech_start_sample"
                        ]
                        + target["speech_sample_count"],
                    },
                },
                "dynamic_rir_state_budget": dynamic_rir_state_budget,
                "question": question,
                "projection_preflight": found["metrics"],
                "motion_preflight": motion_preflight,
                "native_root_provenance": found["provenance"],
                "dedup_key": dedup_key,
                "dedup_key_text": dedup_key_text,
                "artifact_refs": {
                    "rgb_video": "PENDING_FULL75_NATIVE_CAPTURE",
                    "binaural_audio": "PENDING_EXACT_RIR_AND_RENDER",
                    "metric_depth": "PENDING_FULL75_NATIVE_CAPTURE",
                    "target_mask": "PENDING_FULL75_NATIVE_CAPTURE",
                    "distractor_mask": "PENDING_FULL75_NATIVE_CAPTURE",
                    "runtime_readbacks": "PENDING_FULL75_NATIVE_CAPTURE",
                },
                "review": {
                    "sparse_native": "pending",
                    "full75_machine": "pending",
                    "human_visual": "pending",
                    "human_audio": "pending",
                    "rights": "pending",
                },
            }
        )

    mechanism_counts = Counter(item["mechanism"] for item in episodes)
    side_counts = Counter(item["target"]["side"] for item in episodes)
    region_counts = Counter(item["room_region_id"] for item in episodes)
    pair_counts = Counter(
        f"{item['target']['identity_key']}/{item['distractor']['identity_key']}"
        for item in episodes
    )
    answer_index_counts = Counter(
        item["question"]["correct_index"] for item in episodes
    )
    _require(
        mechanism_counts == Counter(request["balance_contract"]["mechanism_counts"]),
        "mechanism balance drift",
    )
    _require(
        side_counts == Counter(request["balance_contract"]["target_side_counts"]),
        "side balance drift",
    )
    _require(
        region_counts == Counter(request["balance_contract"]["spatial_stratum_counts"]),
        "spatial balance drift",
    )
    _require(
        answer_index_counts == Counter({0: 50, 1: 50}), "answer-index balance drift"
    )
    _require(len(used_source_scenarios) == 100, "source Episode independence drift")
    _require(len(used_camera_clusters) == 100, "camera-cluster independence drift")
    _require(len(used_dedup_keys) == 100, "dedup-key independence drift")
    rir_state_counts = Counter(
        {
            mechanism: sum(
                item["dynamic_rir_state_budget"]["expected_unique_rir_state_count"]
                for item in episodes
                if item["mechanism"] == mechanism
            )
            for mechanism in request["mechanism_schedule"]
        }
    )
    exact_rir_state_count_required = sum(rir_state_counts.values())
    requested_source_frame_uses = sum(
        item["dynamic_rir_state_budget"]["requested_source_frame_uses"]
        for item in episodes
    )
    _require(
        exact_rir_state_count_required > len(episodes) * 2,
        "dynamic RIR budget collapsed to the obsolete two-state estimate",
    )
    storage_budget = _storage_budget_summary(
        resource_budget=request["resource_budget"],
        episode_count=len(episodes),
        exact_rir_state_count=exact_rir_state_count_required,
    )

    output.mkdir(parents=True)
    manifest = {
        "schema": OUTPUT_SCHEMA,
        "status": "pass_interim_single_room_cpu_feasibility_not_final_multi_room_100",
        "claim_boundary": request["claim_boundary"],
        "request_id": request["request_id"],
        "assignment_mode": assignment_validation["assignment_mode"],
        "frozen_assignment": str(assignment_path),
        "assignment_validation_status": assignment_validation["status"],
        "episode_count": 100,
        "single_room_mechanism_pilot_count": 20,
        "interim_single_room_candidate_count": 100,
        "final_multi_room_episode_count": 0,
        "formal_episode_count": 0,
        "qualification_claim": False,
        "ready_room_count": request["native_room_scope"]["ready_room_count"],
        "room_scope_boundary": request["native_room_scope"]["boundary"],
        "timeline": request["timeline"],
        "gpu_policy": request["gpu_policy"],
        "episodes": episodes,
    }
    summary = {
        "schema": "avengine_native_strict_two_human_full_episode_batch_summary_v1",
        "status": "pass_interim_single_room_cpu_plan",
        "assignment_mode": assignment_validation["assignment_mode"],
        "assignment_validation_status": assignment_validation["status"],
        "solver_required_at_builder_runtime": False,
        "fixed_mechanism_stratum_cross_quota_required": False,
        "episode_count": len(episodes),
        "batch_count": 10,
        "batch_size": 10,
        "static_full75_pipeline_canary_count": 4,
        "dynamic_full75_mechanism_canary_count": 4,
        "full75_gate_canary_count_total": 8,
        "mechanism_counts": dict(sorted(mechanism_counts.items())),
        "target_side_counts": dict(sorted(side_counts.items())),
        "ordered_identity_pair_counts": dict(sorted(pair_counts.items())),
        "spatial_stratum_counts": dict(sorted(region_counts.items())),
        "answer_index_counts": {
            str(key): value for key, value in sorted(answer_index_counts.items())
        },
        "unique_native_source_scenario_count": len(used_source_scenarios),
        "unique_camera_cluster_count": len(used_camera_clusters),
        "unique_dedup_key_count": len(used_dedup_keys),
        "exact_rir_state_count_required": exact_rir_state_count_required,
        "exact_rir_state_count_by_mechanism": dict(sorted(rir_state_counts.items())),
        "requested_source_frame_uses": requested_source_frame_uses,
        "rir_budget_claim_boundary": (
            "stride-1 exact state-count budget; native RLR jobs remain pending "
            "per-Episode materialization"
        ),
        "native_render_pass_count_required": 300,
        "native_rendered_frame_count_required": 22500,
        "capture_media_only_estimated_storage_gb": storage_budget[
            "capture_media_only"
        ]["budget_decimal_gb"],
        "empirical_rir_cache_estimated_storage_gb": storage_budget[
            "rir_cache_empirical_budget"
        ]["budget_decimal_gb"],
        "minimum_workspace_storage_gb": storage_budget[
            "minimum_capture_plus_rir_workspace_decimal_gb"
        ],
        "storage_budget": storage_budget,
        "ready_room_count": request["native_room_scope"]["ready_room_count"],
        "room_scope_boundary": request["native_room_scope"]["boundary"],
        "formal_episode_count": 0,
        "qualification_claim": False,
        "batch_launch_authorized": False,
        "single_room_mechanism_pilot_count": 20,
        "final_multi_room_episode_count": 0,
        "final_required_ready_room_count": 3,
        "next_gate": "materialize_and_pass_four_dynamic_full75_mechanism_canaries_then_limit_execution_to_first_20_until_three_real_rooms_are_ready",
    }
    dedup = {
        "schema": "avengine_native_strict_two_human_full_episode_dedup_audit_v1",
        "status": "pass",
        "episode_count": 100,
        "unique_camera_cluster_count": len(used_camera_clusters),
        "unique_native_source_scenario_count": len(used_source_scenarios),
        "unique_episode_key_count": len(used_dedup_keys),
        "mirror_variants_counted_as_independent": 0,
        "voice_replacements_counted_as_independent": 0,
        "content_replacements_counted_as_independent": 0,
        "same_camera_variants_counted_as_independent": 0,
        "keys": [item["dedup_key"] for item in episodes],
    }
    paths = {
        "manifest": output / "manifest.json",
        "summary": output / "summary.json",
        "dedup_audit": output / "dedup_audit.json",
        "assignment_validation": output / "assignment_validation.json",
        "canary_plan": output / "canary_plan.json",
        "dynamic_mechanism_canary_plan": output / "dynamic_mechanism_canary_plan.json",
    }
    _write(paths["manifest"], manifest)
    _write(paths["summary"], summary)
    _write(paths["dedup_audit"], dedup)
    _write(paths["assignment_validation"], assignment_validation)
    _write(paths["canary_plan"], _canary_plan(request, publication))
    _write(
        paths["dynamic_mechanism_canary_plan"],
        _dynamic_mechanism_canary_plan(request, episodes),
    )
    for batch_number in range(1, 11):
        rows = [
            item for item in episodes if item["batch_id"] == f"batch_{batch_number:02d}"
        ]
        _require(len(rows) == 10, f"batch {batch_number}: row count drift")
        _write(
            output / f"batches/batch_{batch_number:02d}.json",
            {
                "schema": "avengine_native_strict_two_human_full_episode_execution_batch_v1",
                "status": (
                    "blocked_pending_full75_canaries"
                    if batch_number <= 2
                    else "blocked_pending_three_real_rooms"
                ),
                "batch_id": f"batch_{batch_number:02d}",
                "episode_count": 10,
                "gpu_policy": request["gpu_policy"],
                "episodes": rows,
            },
        )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = build(args.request.resolve(), args.output.resolve())
    print(
        "STRICT_TWO_HUMAN_FULL_EPISODE_CPU_PLAN_OK "
        f"manifest={paths['manifest']} summary={paths['summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

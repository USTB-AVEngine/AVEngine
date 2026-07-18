"""Stable public access to the existing M5.1 source/event/flag semantics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from avengine.contracts.json_io import canonical_json_sha256
from avengine.m5_1.source_contracts import (
    ALL_FLAG_IDS,
    AND_AGGREGATED_FLAGS,
    OR_AGGREGATED_FLAGS,
    PAIR_FLAG_IDS,
    SOURCE_FLAG_IDS,
)
from avengine.m5_1.orientation import habitat_basis_from_yaw_degrees
from avengine.m6.registry import (
    FLAG_DEFINITION_REGISTRY_SCHEMA,
    M6RegistryError,
    json_schema_errors,
    load_validated_document,
    registry_semantic_errors,
)


LEGACY_THRESHOLDS: Mapping[str, float | int] = MappingProxyType(
    {
        "passes_close_to_mic_m": 1.0,
        "far_from_mic_whole_clip_m": 4.0,
        "stationary_mean_speed_mps": 0.1,
        "steady_walk_min_mean_speed_mps": 0.15,
        "steady_walk_max_speed_cv": 0.4,
        "stop_and_go_stop_speed_mps": 0.05,
        "stop_and_go_min_stopped_frames": 3,
        "stop_and_go_min_moving_frames": 3,
        "sources_pass_each_other_m": 0.5,
    }
)

STATUS_VALUE = MappingProxyType(
    {"present": True, "absent": False, "not_evaluated": None}
)


@dataclass(frozen=True)
class LegacyFlagDefinition:
    flag_id: str
    scope: str
    clip_aggregation: str
    evaluator_kind: str
    required_fact_ids: tuple[str, ...]
    threshold_ids: tuple[str, ...]
    definition: str


@dataclass(frozen=True)
class LegacyFlagAccess:
    registry_id: str
    revision: str
    source_flag_ids: tuple[str, ...]
    pair_flag_ids: tuple[str, ...]
    thresholds: Mapping[str, float | int]
    definitions: tuple[LegacyFlagDefinition, ...]


def validate_legacy_flag_registry(value: Any) -> list[str]:
    errors = json_schema_errors(value, FLAG_DEFINITION_REGISTRY_SCHEMA)
    errors.extend(
        registry_semantic_errors(
            value,
            records_field="flags",
            record_id_field="flag_id",
            record_revision_field="definition_revision",
            require_sorted=False,
        )
    )
    if not isinstance(value, Mapping):
        return errors
    thresholds = value.get("thresholds")
    if thresholds != dict(LEGACY_THRESHOLDS):
        errors.append("thresholds must exactly equal the frozen M5.1 v1 thresholds")
    flags = value.get("flags")
    if not isinstance(flags, list):
        return errors
    observed_ids = tuple(
        item.get("flag_id") for item in flags if isinstance(item, Mapping)
    )
    if observed_ids != ALL_FLAG_IDS:
        errors.append("flags must exactly preserve M5.1 v1 ID membership and order")
    for index, item in enumerate(flags):
        if not isinstance(item, Mapping):
            continue
        flag_id = item.get("flag_id")
        if flag_id not in ALL_FLAG_IDS:
            continue
        expected_scope = "pairwise" if flag_id in PAIR_FLAG_IDS else "per_source"
        expected_aggregation = "or" if flag_id in OR_AGGREGATED_FLAGS else "and"
        if item.get("scope") != expected_scope:
            errors.append(f"flags[{index}].scope must remain {expected_scope}")
        if item.get("clip_aggregation") != expected_aggregation:
            errors.append(
                f"flags[{index}].clip_aggregation must remain {expected_aggregation}"
            )
        if item.get("definition_revision") != "m5_1_v1":
            errors.append(f"flags[{index}].definition_revision must remain m5_1_v1")
        threshold_ids = item.get("threshold_ids", [])
        if any(threshold_id not in LEGACY_THRESHOLDS for threshold_id in threshold_ids):
            errors.append(f"flags[{index}].threshold_ids contains an unknown threshold")
    return errors


def load_legacy_flag_registry(path: str | Path) -> dict[str, Any]:
    return load_validated_document(path, validator=validate_legacy_flag_registry)


def legacy_flag_access(value: Mapping[str, Any]) -> LegacyFlagAccess:
    errors = validate_legacy_flag_registry(value)
    if errors:
        raise M6RegistryError(errors)
    definitions = tuple(
        LegacyFlagDefinition(
            flag_id=item["flag_id"],
            scope=item["scope"],
            clip_aggregation=item["clip_aggregation"],
            evaluator_kind=item["evaluator_kind"],
            required_fact_ids=tuple(item["required_fact_ids"]),
            threshold_ids=tuple(item["threshold_ids"]),
            definition=item["definition"],
        )
        for item in value["flags"]
    )
    return LegacyFlagAccess(
        registry_id=value["registry_id"],
        revision=value["revision"],
        source_flag_ids=SOURCE_FLAG_IDS,
        pair_flag_ids=PAIR_FLAG_IDS,
        thresholds=MappingProxyType(dict(value["thresholds"])),
        definitions=definitions,
    )


def aggregate_legacy_status(flag_id: str, statuses: Iterable[str]) -> str:
    """Expose the exact M5.1 tri-state OR/AND aggregation as a stable API."""

    if flag_id not in ALL_FLAG_IDS:
        raise KeyError(f"unknown M5.1 flag ID: {flag_id}")
    values = tuple(statuses)
    if not values or any(value not in STATUS_VALUE for value in values):
        raise ValueError("statuses must be a non-empty sequence of M5.1 states")
    if flag_id in OR_AGGREGATED_FLAGS:
        if "present" in values:
            return "present"
        if all(value == "absent" for value in values):
            return "absent"
        return "not_evaluated"
    if "absent" in values:
        return "absent"
    if all(value == "present" for value in values):
        return "present"
    return "not_evaluated"


def provider_assessment(
    *,
    flag_id: str,
    scope: str,
    status: str | None,
    reason_code: str,
    reason: str,
    evidence: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Adapt provider facts without coercing missing facts to ``absent``."""

    if flag_id not in ALL_FLAG_IDS:
        raise KeyError(f"unknown M5.1 flag ID: {flag_id}")
    expected_scope = "pairwise" if flag_id in PAIR_FLAG_IDS else "per_source"
    if scope != expected_scope:
        raise ValueError(f"{flag_id} requires scope={expected_scope!r}")
    normalized_status = "not_evaluated" if status is None else status
    if normalized_status not in STATUS_VALUE:
        raise ValueError("status must be present, absent, not_evaluated, or None")
    evidence_items = [dict(item) for item in evidence]
    if not evidence_items:
        raise ValueError("at least one evidence record is required")
    kinds = {item.get("kind") for item in evidence_items}
    missing_kinds = {"missing_dependency", "manual_review"}
    if normalized_status == "not_evaluated" and not kinds.intersection(missing_kinds):
        raise ValueError("not_evaluated requires missing_dependency or manual_review evidence")
    if normalized_status != "not_evaluated" and kinds <= missing_kinds:
        raise ValueError("evaluated status cannot rely only on missing/manual evidence")
    return {
        "scope": scope,
        "status": normalized_status,
        "value": STATUS_VALUE[normalized_status],
        "reason_code": reason_code,
        "reason": reason,
        "evidence": evidence_items,
    }


_VISIBILITY_FLAG_IDS = (
    "occluded_by_furniture",
    "occluded_by_wall",
    "never_occluded",
    "leaves_camera_fov",
    "stays_in_camera_fov",
)
_TRAJECTORY_FLAG_IDS = tuple(
    flag_id for flag_id in SOURCE_FLAG_IDS if flag_id not in _VISIBILITY_FLAG_IDS
)


def _metric_assessment(
    *,
    scope: str,
    present: bool,
    evidence_id: str,
    evidence_uri: str,
    evidence_sha256: str,
    metric: str,
    value: float | int,
    unit: str,
    threshold: float | int | None = None,
    comparison: str | None = None,
) -> dict[str, Any]:
    measurement: dict[str, Any] = {"metric": metric, "value": value, "unit": unit}
    if threshold is not None:
        measurement["threshold"] = threshold
    if comparison is not None:
        measurement["comparison"] = comparison
    status = "present" if present else "absent"
    return {
        "scope": scope,
        "status": status,
        "value": present,
        "reason_code": (
            "trajectory_metric_satisfies_definition"
            if present
            else "trajectory_metric_does_not_satisfy_definition"
        ),
        "reason": "Assessment recomputed by the M6 adapter using frozen M5.1 v1 semantics.",
        "evidence": [
            {
                "evidence_id": evidence_id,
                "kind": "metric",
                "uri": evidence_uri,
                "sha256": evidence_sha256,
                "summary": f"Recomputed {metric} from supplied authoritative facts.",
                "measurement": measurement,
            }
        ],
    }


def _missing_assessment(
    *, scope: str, evidence_id: str, evidence_uri: str, evidence_sha256: str
) -> dict[str, Any]:
    return provider_assessment(
        flag_id=(
            "sources_pass_each_other" if scope == "pairwise" else "occluded_by_wall"
        ),
        scope=scope,
        status=None,
        reason_code="required_verifier_input_missing",
        reason="The room/source provider did not supply the required evaluated fact.",
        evidence=[
            {
                "evidence_id": evidence_id,
                "kind": "missing_dependency",
                "uri": evidence_uri,
                "sha256": evidence_sha256,
                "summary": "Missing provider fact is retained as not_evaluated, never absent.",
            }
        ],
    )


def _visibility_statuses(
    facts: Mapping[str, Any] | None, *, frame_count: int
) -> dict[str, bool | None]:
    if facts is None:
        return {flag_id: None for flag_id in _VISIBILITY_FLAG_IDS}
    in_fov = facts.get("in_fov_by_frame")
    occlusion = facts.get("occlusion_by_frame")
    if not (
        isinstance(in_fov, Sequence)
        and not isinstance(in_fov, (str, bytes))
        and isinstance(occlusion, Sequence)
        and not isinstance(occlusion, (str, bytes))
        and len(in_fov) == frame_count
        and len(occlusion) == frame_count
    ):
        raise ValueError(
            "visibility facts require frame-aligned in_fov_by_frame and "
            "occlusion_by_frame sequences"
        )
    if any(item not in {True, False, None} for item in in_fov):
        raise ValueError("in_fov_by_frame values must be true, false, or null")
    allowed_occlusion = {"clear", "furniture", "wall", "other", None}
    if any(item not in allowed_occlusion for item in occlusion):
        raise ValueError(
            "occlusion_by_frame values must be clear, furniture, wall, other, or null"
        )

    def or_fact(values: Sequence[bool | None]) -> bool | None:
        if True in values:
            return True
        if all(value is False for value in values):
            return False
        return None

    furniture = or_fact(
        [None if item is None else item == "furniture" for item in occlusion]
    )
    wall = or_fact([None if item is None else item == "wall" for item in occlusion])
    if any(item in {"furniture", "wall", "other"} for item in occlusion):
        never_occluded: bool | None = False
    elif all(item == "clear" for item in occlusion):
        never_occluded = True
    else:
        never_occluded = None
    leaves = or_fact([None if item is None else not item for item in in_fov])
    if False in in_fov:
        stays: bool | None = False
    elif all(item is True for item in in_fov):
        stays = True
    else:
        stays = None
    return {
        "occluded_by_furniture": furniture,
        "occluded_by_wall": wall,
        "never_occluded": never_occluded,
        "leaves_camera_fov": leaves,
        "stays_in_camera_fov": stays,
    }


def evaluate_legacy_flags(
    *,
    observer_position_m: Sequence[float],
    observer_yaw_deg: float,
    fps: float,
    positions_by_source: Mapping[str, Sequence[Sequence[float]]],
    visibility_facts_by_source: Mapping[str, Mapping[str, Any]] | None = None,
    evidence_uri: str = "memory://m6/legacy-flag-facts",
) -> dict[str, Any]:
    """Evaluate the frozen M5.1 trajectory flags and optional visibility facts.

    Inputs are compact provider facts, not a replacement dense episode schema.
    All sources must have one synchronized Y-up Habitat-world trajectory.
    Visibility uses explicit provider facts; if absent, the five dependent
    flags remain ``not_evaluated`` with missing-dependency evidence.
    """

    if (
        len(observer_position_m) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in observer_position_m
        )
    ):
        raise ValueError("observer_position_m must contain three finite numbers")
    if (
        isinstance(observer_yaw_deg, bool)
        or not isinstance(observer_yaw_deg, (int, float))
        or not math.isfinite(float(observer_yaw_deg))
    ):
        raise ValueError("observer_yaw_deg must be finite")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
    ):
        raise ValueError("fps must be a positive finite number")
    source_ids = tuple(positions_by_source)
    if not source_ids or source_ids != tuple(sorted(set(source_ids))):
        raise ValueError("positions_by_source must use unique canonical source ID order")

    normalized: dict[str, tuple[tuple[float, float, float], ...]] = {}
    frame_count: int | None = None
    for source_id, raw_positions in positions_by_source.items():
        positions: list[tuple[float, float, float]] = []
        for raw in raw_positions:
            if (
                len(raw) != 3
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in raw
                )
            ):
                raise ValueError(f"{source_id} trajectory positions must be finite vec3")
            positions.append(tuple(float(item) for item in raw))
        if len(positions) < 2:
            raise ValueError(f"{source_id} trajectory requires at least two frames")
        if frame_count is None:
            frame_count = len(positions)
        elif len(positions) != frame_count:
            raise ValueError("source trajectories must have equal synchronized frame counts")
        normalized[source_id] = tuple(positions)
    assert frame_count is not None
    if visibility_facts_by_source is not None:
        unknown_sources = set(visibility_facts_by_source) - set(source_ids)
        if unknown_sources:
            raise ValueError(
                f"visibility facts reference unknown sources: {sorted(unknown_sources)}"
            )

    evidence_payload = {
        "observer_position_m": list(observer_position_m),
        "observer_yaw_deg": observer_yaw_deg,
        "fps": fps,
        "positions_by_source": positions_by_source,
        "visibility_facts_by_source": visibility_facts_by_source,
    }
    evidence_sha256 = canonical_json_sha256(evidence_payload)
    observer = tuple(float(item) for item in observer_position_m)
    right_xz = habitat_basis_from_yaw_degrees(float(observer_yaw_deg)).right_xz
    source_flags: dict[str, dict[str, Any]] = {}

    for source_id in source_ids:
        positions = normalized[source_id]
        distances = [math.dist(position, observer) for position in positions]
        speeds = [
            math.dist(left, right) * float(fps)
            for left, right in zip(positions, positions[1:])
        ]
        mean_speed = sum(speeds) / len(speeds)
        speed_std = math.sqrt(
            sum((speed - mean_speed) ** 2 for speed in speeds) / len(speeds)
        )
        speed_cv = speed_std / max(mean_speed, 1.0e-6)
        lateral = [
            (position[0] - observer[0]) * right_xz[0]
            + (position[2] - observer[2]) * right_xz[1]
            for position in positions
        ]
        stop_threshold = float(LEGACY_THRESHOLDS["stop_and_go_stop_speed_mps"])
        stopped = sum(speed < stop_threshold for speed in speeds)
        moving = sum(speed >= stop_threshold for speed in speeds)
        expected = {
            "crosses_azimuth_zero": min(lateral) < 0.0 < max(lateral),
            "passes_close_to_mic": min(distances)
            < float(LEGACY_THRESHOLDS["passes_close_to_mic_m"]),
            "far_from_mic_whole_clip": min(distances)
            > float(LEGACY_THRESHOLDS["far_from_mic_whole_clip_m"]),
            "stationary": mean_speed
            < float(LEGACY_THRESHOLDS["stationary_mean_speed_mps"]),
            "steady_walk": len(speeds) >= 3
            and mean_speed
            >= float(LEGACY_THRESHOLDS["steady_walk_min_mean_speed_mps"])
            and speed_cv < float(LEGACY_THRESHOLDS["steady_walk_max_speed_cv"]),
            "stop_and_go": stopped
            >= int(LEGACY_THRESHOLDS["stop_and_go_min_stopped_frames"])
            and moving
            >= int(LEGACY_THRESHOLDS["stop_and_go_min_moving_frames"]),
        }
        metric_specs: Mapping[str, tuple[str, float | int, str, str | None, str | None]] = {
            "crosses_azimuth_zero": ("camera_local_lateral_span_m", max(lateral) - min(lateral), "m", None, None),
            "passes_close_to_mic": ("minimum_listener_distance_m", min(distances), "m", "lt", "passes_close_to_mic_m"),
            "far_from_mic_whole_clip": ("minimum_listener_distance_m", min(distances), "m", "gt", "far_from_mic_whole_clip_m"),
            "stationary": ("mean_speed_mps", mean_speed, "m/s", "lt", "stationary_mean_speed_mps"),
            "steady_walk": ("speed_coefficient_of_variation", speed_cv, "ratio", "lt", "steady_walk_max_speed_cv"),
            "stop_and_go": ("stopped_frame_count", stopped, "frames", "gte", "stop_and_go_min_stopped_frames"),
        }
        assessments: dict[str, Any] = {}
        visibility = _visibility_statuses(
            None
            if visibility_facts_by_source is None
            else visibility_facts_by_source.get(source_id),
            frame_count=frame_count,
        )
        for flag_id in _VISIBILITY_FLAG_IDS:
            status = visibility[flag_id]
            if status is None:
                assessments[flag_id] = _missing_assessment(
                    scope="per_source",
                    evidence_id=f"{source_id}_{flag_id}",
                    evidence_uri=evidence_uri,
                    evidence_sha256=evidence_sha256,
                )
            else:
                assessments[flag_id] = provider_assessment(
                    flag_id=flag_id,
                    scope="per_source",
                    status="present" if status else "absent",
                    reason_code="provider_visibility_fact_evaluated",
                    reason="Assessment derived from explicit frame-aligned room-provider facts.",
                    evidence=[
                        {
                            "evidence_id": f"{source_id}_{flag_id}",
                            "kind": "frame_interval",
                            "uri": evidence_uri,
                            "sha256": evidence_sha256,
                            "summary": "Frame-aligned FOV/occlusion provider facts were evaluated.",
                        }
                    ],
                )
        for flag_id in _TRAJECTORY_FLAG_IDS:
            metric, value, unit, comparison, threshold_id = metric_specs[flag_id]
            assessments[flag_id] = _metric_assessment(
                scope="per_source",
                present=expected[flag_id],
                evidence_id=f"{source_id}_{flag_id}",
                evidence_uri=evidence_uri,
                evidence_sha256=evidence_sha256,
                metric=metric,
                value=value,
                unit=unit,
                comparison=comparison,
                threshold=(
                    None if threshold_id is None else LEGACY_THRESHOLDS[threshold_id]
                ),
            )
        source_flags[source_id] = {
            flag_id: assessments[flag_id] for flag_id in SOURCE_FLAG_IDS
        }

    pair_flags: list[dict[str, Any]] = []
    for left_id, right_id in combinations(source_ids, 2):
        minimum = min(
            math.hypot(left[0] - right[0], left[2] - right[2])
            for left, right in zip(normalized[left_id], normalized[right_id])
        )
        pair_flags.append(
            {
                "source_ids": [left_id, right_id],
                "flags": {
                    "sources_pass_each_other": _metric_assessment(
                        scope="pairwise",
                        present=minimum
                        < float(LEGACY_THRESHOLDS["sources_pass_each_other_m"]),
                        evidence_id=f"{left_id}_{right_id}_sources_pass_each_other",
                        evidence_uri=evidence_uri,
                        evidence_sha256=evidence_sha256,
                        metric="minimum_horizontal_source_separation_m",
                        value=minimum,
                        unit="m",
                        comparison="lt",
                        threshold=LEGACY_THRESHOLDS["sources_pass_each_other_m"],
                    )
                },
            }
        )

    clip_flags: dict[str, Any] = {}
    for flag_id in ALL_FLAG_IDS:
        if flag_id in PAIR_FLAG_IDS:
            statuses = [item["flags"][flag_id]["status"] for item in pair_flags]
            # A single-source provider has no pair assessment.  It is not a
            # valid M5.1 relationship domain and must remain not evaluated.
            aggregate = (
                "not_evaluated"
                if not statuses
                else aggregate_legacy_status(flag_id, statuses)
            )
        else:
            statuses = [source_flags[source_id][flag_id]["status"] for source_id in source_ids]
            aggregate = aggregate_legacy_status(flag_id, statuses)
        missing = aggregate == "not_evaluated"
        clip_flags[flag_id] = {
            "scope": "clip",
            "status": aggregate,
            "value": STATUS_VALUE[aggregate],
            "reason_code": (
                "aggregate_contains_not_evaluated"
                if missing
                else "legacy_source_pair_aggregate"
            ),
            "reason": "Clip status uses the frozen M5.1 OR/AND aggregation rule.",
            "evidence": [
                {
                    "evidence_id": f"clip_{flag_id}",
                    "kind": "missing_dependency" if missing else "artifact",
                    "uri": evidence_uri,
                    "sha256": evidence_sha256,
                    "summary": "Aggregate recomputed from source/pair assessments.",
                }
            ],
        }
    return {
        "schema": "avengine_m6_legacy_flag_report_v1",
        "definition_registry_id": "legacy_m5_1_source_event_flags",
        "definition_revision": "m5_1_v1",
        "coordinate_frame": "avengine_world_right_handed_y_up_m",
        "source_flags": source_flags,
        "pair_flags": pair_flags,
        "clip_flags": clip_flags,
        "facts_content_sha256": evidence_sha256,
    }

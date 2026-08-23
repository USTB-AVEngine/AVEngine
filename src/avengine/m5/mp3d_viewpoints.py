"""CPU-only MP3D viewpoint candidate selection from retained native authority.

The selector deliberately stops before Habitat rendering.  It consumes floor
samples exported from a declared Habitat navmesh plus actor routes that were
already checked by the native PathFinder, then performs deterministic analytic
frustum and diversity scoring.  Every returned viewpoint remains a candidate:
native floor readback, occlusion, RGB pixels, first-person review, and the
four-point mini-pilot are separate later steps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
import re
from typing import Any

import numpy as np

from avengine.camera_pose import (
    CameraPoseError,
    apply_camera_listener_pose,
    yaw_rotation_xyzw,
)
from avengine.m2.habitat_capture import quaternion_xyzw_to_matrix
from avengine.m5.current_mp3d_route import (
    CurrentMP3DRouteError,
    _camera_frustum_predicate,
    _camera_yaw_toward,
)


class MP3DViewpointPrefilterError(ValueError):
    """Retained MP3D authority cannot produce enough distinct candidates."""


@dataclass(frozen=True)
class MP3DViewpointCandidate:
    viewpoint_id: str
    camera_position_m: tuple[float, float, float]
    camera_yaw_degrees: float
    camera_height_role: str
    focus_position_m: tuple[float, float, float]
    analytic_scores: Mapping[str, Any]
    route_subset: Mapping[str, Any]
    m1_request: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "viewpoint_id": self.viewpoint_id,
            "camera": {
                "habitat_translation_m": list(self.camera_position_m),
                "habitat_yaw_degrees": self.camera_yaw_degrees,
                "height_role": self.camera_height_role,
                "focus_position_m": list(self.focus_position_m),
            },
            "analytic_scores": dict(self.analytic_scores),
            "route_subset": deepcopy(dict(self.route_subset)),
            "candidate_floor_navigation_status": "pending_native_habitat_readback",
            "native_habitat_pixel_status": "pending",
            "first_person_certification_status": "pending",
            "four_point_mini_pilot_status": "pending",
        }


@dataclass(frozen=True)
class MP3DViewpointPrefilterResult:
    room_id: str
    viewpoints: tuple[MP3DViewpointCandidate, ...]
    pairwise_deduplication: tuple[Mapping[str, Any], ...]
    thresholds: Mapping[str, float]
    input_counts: Mapping[str, int]

    def to_record(self) -> dict[str, Any]:
        return {
            "artifact_status": "cpu_offline_candidate_prefilter_complete",
            "certification_status": "pending_native_habitat",
            "research_only": True,
            "episode_counted": False,
            "formal_release_status": "not_run",
            "studio_used": False,
            "backend_route": "mp3d_habitat_sim_only",
            "selection_execution": "cpu_offline_no_habitat_process",
            "retained_native_authority_status": "consumed_not_reexecuted",
            "native_candidate_navigation_validation_status": "pending",
            "native_habitat_pixel_status": "pending",
            "first_person_certification_status": "pending",
            "four_point_mini_pilot_status": "pending",
            "claim_boundary": (
                "Analytic candidate prefilter from retained Habitat-navmesh floor "
                "samples and native-PathFinder actor routes only. This builder does "
                "not prove candidate navigability, scene occlusion, RGB composition, "
                "first-person quality, audio, or mini-pilot certification."
            ),
            "room_id": self.room_id,
            "viewpoint_count": len(self.viewpoints),
            "selection_thresholds": dict(self.thresholds),
            "input_counts": dict(self.input_counts),
            "pairwise_deduplication": [
                dict(item) for item in self.pairwise_deduplication
            ],
            "viewpoints": [item.to_record() for item in self.viewpoints],
        }


@dataclass(frozen=True)
class _Candidate:
    camera_position_m: tuple[float, float, float]
    camera_yaw_degrees: float
    camera_height_role: str
    focus_position_m: tuple[float, float, float]
    analytic_scores: Mapping[str, Any]
    route_subset: Mapping[str, Any]
    coverage_signature: frozenset[str]
    path_signature: frozenset[str]
    sort_key: tuple[Any, ...]


def _finite_number(value: Any, *, owner: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MP3DViewpointPrefilterError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise MP3DViewpointPrefilterError(f"{owner} must be finite and >= {minimum}")
    return result


def _finite_points(value: Any, *, owner: str, minimum_count: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DViewpointPrefilterError(
            f"{owner} must contain finite XYZ points"
        ) from exc
    if (
        result.ndim != 2
        or result.shape[1:] != (3,)
        or len(result) < minimum_count
        or not np.all(np.isfinite(result))
    ):
        raise MP3DViewpointPrefilterError(
            f"{owner} must contain at least {minimum_count} finite XYZ points"
        )
    return np.ascontiguousarray(result)


def _canonical_floor_samples(value: Any) -> np.ndarray:
    points = _finite_points(value, owner="camera floor samples", minimum_count=1)
    unique = sorted({tuple(float(item) for item in point) for point in points})
    return np.asarray(unique, dtype=np.float64)


def _canonical_actor_routes(value: Any) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or not value:
        raise MP3DViewpointPrefilterError("actor routes must be a nonempty mapping")
    result: dict[str, np.ndarray] = {}
    for raw_actor_id in sorted(value, key=lambda item: str(item).encode("utf-8")):
        if not isinstance(raw_actor_id, str) or not raw_actor_id.strip():
            raise MP3DViewpointPrefilterError(
                "actor route IDs must be nonempty strings"
            )
        result[raw_actor_id] = _finite_points(
            value[raw_actor_id],
            owner=f"actor route {raw_actor_id!r}",
            minimum_count=2,
        )
    return result


def _height_variants(agent_height_m: float) -> tuple[tuple[str, float], ...]:
    base = _finite_number(agent_height_m, owner="navigation agent height", minimum=0.5)
    values = (
        ("base", base),
        ("low", max(0.5, base * 0.8)),
        ("high", base * 1.2),
    )
    result: list[tuple[str, float]] = []
    for role, height in values:
        if not any(abs(height - previous) <= 1.0e-6 for _name, previous in result):
            result.append((role, float(height)))
    return tuple(result)


def _route_focus_anchors(
    routes: Mapping[str, np.ndarray], *, probe_height_m: float
) -> tuple[tuple[float, float, float], ...]:
    offset = np.asarray([0.0, probe_height_m, 0.0], dtype=np.float64)
    all_points = np.concatenate([points + offset for points in routes.values()], axis=0)
    pending = [np.mean(all_points, axis=0)]
    for points in routes.values():
        probes = points + offset
        indices = sorted(
            {
                0,
                (len(probes) - 1) // 4,
                (len(probes) - 1) // 2,
                3 * (len(probes) - 1) // 4,
                len(probes) - 1,
            }
        )
        pending.extend(probes[index] for index in indices)
    ordered = sorted(
        (tuple(float(item) for item in point) for point in pending),
        key=lambda point: (point[0], point[2], point[1]),
    )
    result: list[tuple[float, float, float]] = []
    for point in ordered:
        if not any(math.dist(point, previous) <= 1.0e-6 for previous in result):
            result.append(point)
    return tuple(result)


def _visible_ranges(mask: np.ndarray) -> list[list[int]]:
    flags = np.asarray(mask, dtype=np.bool_)
    result: list[list[int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        if start is not None and (not flag or index == len(flags) - 1):
            end = index if flag and index == len(flags) - 1 else index - 1
            result.append([start, end])
            start = None
    return result


def _jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _circular_yaw_difference(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _analytic_request(
    base_request: Mapping[str, Any],
    *,
    position_m: Sequence[float],
    yaw_degrees: float,
) -> dict[str, Any]:
    try:
        rig = deepcopy(base_request["primary_camera_rig"])
        rig["world_from_rig"]["translation_m"] = list(position_m)
        rig["world_from_rig"]["rotation_xyzw"] = yaw_rotation_xyzw(yaw_degrees)
    except (KeyError, TypeError) as exc:
        raise MP3DViewpointPrefilterError("base M1 request lacks a camera rig") from exc
    return {"primary_camera_rig": rig}


def _lateral_metrics(
    request: Mapping[str, Any],
    route_probes: Mapping[str, np.ndarray],
    visible_masks: Mapping[str, np.ndarray],
) -> tuple[float, float]:
    rig = request["primary_camera_rig"]
    origin = np.asarray(rig["world_from_rig"]["translation_m"], dtype=np.float64)
    rotation = quaternion_xyzw_to_matrix(rig["world_from_rig"]["rotation_xyzw"])
    right = (rotation @ np.asarray([1.0, 0.0, 0.0], dtype=np.float64))[[0, 2]]
    right /= np.linalg.norm(right)
    values: list[float] = []
    actor_medians: list[float] = []
    for actor_id, points in route_probes.items():
        visible = points[np.asarray(visible_masks[actor_id], dtype=np.bool_)]
        deltas = visible[:, (0, 2)] - origin[[0, 2]]
        distances = np.linalg.norm(deltas, axis=1)
        lateral = (deltas @ right) / distances
        values.extend(float(item) for item in lateral)
        actor_medians.append(float(np.median(lateral)))
    span = max(values) - min(values) if values else 0.0
    separation = (
        max(actor_medians) - min(actor_medians) if len(actor_medians) > 1 else 0.0
    )
    return float(span), float(separation)


def _pairwise_record(
    left: _Candidate,
    right: _Candidate,
    *,
    minimum_position_m: float,
    minimum_yaw_degrees: float,
    minimum_coverage_jaccard_distance: float,
    minimum_path_jaccard_distance: float,
) -> dict[str, Any]:
    position = math.dist(left.camera_position_m, right.camera_position_m)
    yaw = _circular_yaw_difference(left.camera_yaw_degrees, right.camera_yaw_degrees)
    coverage_similarity = _jaccard_similarity(
        left.coverage_signature, right.coverage_signature
    )
    path_similarity = _jaccard_similarity(left.path_signature, right.path_signature)
    coverage_distance = 1.0 - coverage_similarity
    path_distance = 1.0 - path_similarity
    duplicate = (
        position < minimum_position_m
        and yaw < minimum_yaw_degrees
        and coverage_distance < minimum_coverage_jaccard_distance
        and path_distance < minimum_path_jaccard_distance
    )
    return {
        "position_distance_m": position,
        "circular_yaw_difference_degrees": yaw,
        "coverage_jaccard_similarity": coverage_similarity,
        "coverage_jaccard_distance": coverage_distance,
        "path_jaccard_similarity": path_similarity,
        "path_jaccard_distance": path_distance,
        "near_duplicate": duplicate,
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def select_mp3d_viewpoints(
    *,
    room_id: str,
    room_navigation: Mapping[str, Any],
    base_m1_request: Mapping[str, Any],
    camera_floor_samples_m: Sequence[Sequence[Real]],
    actor_routes_m: Mapping[str, Sequence[Sequence[Real]]],
    selection_count: int = 6,
    actor_probe_height_m: Real = 0.35,
    minimum_actor_route_fraction: Real = 0.35,
    minimum_pairwise_position_m: Real = 0.75,
    minimum_pairwise_yaw_degrees: Real = 15.0,
    minimum_pairwise_coverage_jaccard_distance: Real = 0.10,
    minimum_pairwise_path_jaccard_distance: Real = 0.10,
) -> MP3DViewpointPrefilterResult:
    """Select 4--8 deterministic MP3D candidates without launching Habitat."""

    if not isinstance(room_id, str) or not room_id.strip():
        raise MP3DViewpointPrefilterError("room_id must be nonempty")
    if isinstance(selection_count, bool) or not isinstance(selection_count, int):
        raise MP3DViewpointPrefilterError("selection_count must be an integer")
    if not 4 <= selection_count <= 8:
        raise MP3DViewpointPrefilterError("selection_count must lie within 4--8")
    if not isinstance(room_navigation, Mapping):
        raise MP3DViewpointPrefilterError("room navigation must be an object")
    try:
        agent_height = room_navigation["agent_height_m"]
    except KeyError as exc:
        raise MP3DViewpointPrefilterError(
            "room navigation lacks agent_height_m"
        ) from exc
    probe_height = _finite_number(
        actor_probe_height_m, owner="actor probe height", minimum=0.0
    )
    minimum_actor_fraction = _finite_number(
        minimum_actor_route_fraction,
        owner="minimum actor route fraction",
        minimum=0.0,
    )
    thresholds = {
        "minimum_actor_route_fraction": minimum_actor_fraction,
        "minimum_pairwise_position_m": _finite_number(
            minimum_pairwise_position_m,
            owner="minimum pairwise position",
            minimum=0.0,
        ),
        "minimum_pairwise_yaw_degrees": _finite_number(
            minimum_pairwise_yaw_degrees,
            owner="minimum pairwise yaw",
            minimum=0.0,
        ),
        "minimum_pairwise_coverage_jaccard_distance": _finite_number(
            minimum_pairwise_coverage_jaccard_distance,
            owner="minimum coverage Jaccard distance",
            minimum=0.0,
        ),
        "minimum_pairwise_path_jaccard_distance": _finite_number(
            minimum_pairwise_path_jaccard_distance,
            owner="minimum path Jaccard distance",
            minimum=0.0,
        ),
    }
    if minimum_actor_fraction > 1.0 or any(
        thresholds[name] > 1.0
        for name in (
            "minimum_pairwise_coverage_jaccard_distance",
            "minimum_pairwise_path_jaccard_distance",
        )
    ):
        raise MP3DViewpointPrefilterError("fraction thresholds must not exceed 1")
    if thresholds["minimum_pairwise_yaw_degrees"] > 180.0:
        raise MP3DViewpointPrefilterError("pairwise yaw threshold must not exceed 180")

    floor_samples = _canonical_floor_samples(camera_floor_samples_m)
    routes = _canonical_actor_routes(actor_routes_m)
    route_probes = {
        actor_id: points + np.asarray([0.0, probe_height, 0.0])
        for actor_id, points in routes.items()
    }
    focus_anchors = _route_focus_anchors(routes, probe_height_m=probe_height)
    height_variants = _height_variants(agent_height)
    floor_probe_points = floor_samples + np.asarray([0.0, probe_height, 0.0])
    actor_point_count = sum(len(points) for points in routes.values())
    actor_segment_count = sum(len(points) - 1 for points in routes.values())
    height_priority = {"base": 0, "low": 1, "high": 2}
    pending: list[_Candidate] = []

    for floor_point in floor_samples:
        for height_role, height in height_variants:
            camera_position = np.asarray(floor_point, dtype=np.float64).copy()
            camera_position[1] += height
            for focus in focus_anchors:
                try:
                    yaw = _camera_yaw_toward(
                        camera_position, np.asarray(focus, dtype=np.float64)
                    )
                except (CurrentMP3DRouteError, ValueError):
                    continue
                analytic_request = _analytic_request(
                    base_m1_request,
                    position_m=camera_position,
                    yaw_degrees=yaw,
                )
                visible = _camera_frustum_predicate(analytic_request)
                visible_masks = {
                    actor_id: np.asarray(
                        [visible(point) for point in points], dtype=np.bool_
                    )
                    for actor_id, points in route_probes.items()
                }
                actor_fractions = {
                    actor_id: float(np.mean(mask))
                    for actor_id, mask in visible_masks.items()
                }
                if min(actor_fractions.values()) + 1.0e-12 < minimum_actor_fraction:
                    continue

                coverage_signature: set[str] = set()
                path_signature: set[str] = set()
                route_subset: dict[str, Any] = {}
                for actor_id, mask in visible_masks.items():
                    visible_indices = np.flatnonzero(mask)
                    segments = np.flatnonzero(mask[:-1] & mask[1:])
                    coverage_signature.update(
                        f"{actor_id}:frame:{int(index)}" for index in visible_indices
                    )
                    path_signature.update(
                        f"{actor_id}:segment:{int(index)}" for index in segments
                    )
                    route_subset[actor_id] = {
                        "route_frame_count": len(mask),
                        "visible_frame_count": int(np.count_nonzero(mask)),
                        "visible_frame_ranges_inclusive": _visible_ranges(mask),
                        "visible_segment_indices": [int(item) for item in segments],
                    }

                visible_route_fraction = len(coverage_signature) / actor_point_count
                visible_path_fraction = (
                    len(path_signature) / actor_segment_count
                    if actor_segment_count
                    else 0.0
                )
                visible_area_fraction = float(
                    np.mean([visible(point) for point in floor_probe_points])
                )
                lateral_span, actor_lateral_separation = _lateral_metrics(
                    analytic_request, route_probes, visible_masks
                )
                route_center = np.mean(
                    np.concatenate(list(route_probes.values()), axis=0), axis=0
                )
                distance = float(
                    np.linalg.norm(camera_position[[0, 2]] - route_center[[0, 2]])
                )
                scores = {
                    "visible_navmesh_sample_fraction": visible_area_fraction,
                    "visible_actor_route_fraction": visible_route_fraction,
                    "visible_actor_route_fraction_by_actor": actor_fractions,
                    "visible_route_segment_fraction": visible_path_fraction,
                    "lateral_span_normalized": lateral_span,
                    "actor_median_lateral_separation_normalized": (
                        actor_lateral_separation
                    ),
                    "route_center_distance_m": distance,
                    "analytic_occlusion_status": "not_evaluated",
                }
                sort_key: tuple[Any, ...] = (
                    -min(actor_fractions.values()),
                    -visible_route_fraction,
                    -visible_path_fraction,
                    -visible_area_fraction,
                    -actor_lateral_separation,
                    -lateral_span,
                    abs(distance - 4.5),
                    height_priority[height_role],
                    float(camera_position[0]),
                    float(camera_position[1]),
                    float(camera_position[2]),
                    float(yaw),
                    focus,
                )
                pending.append(
                    _Candidate(
                        camera_position_m=tuple(
                            float(item) for item in camera_position
                        ),
                        camera_yaw_degrees=float(yaw),
                        camera_height_role=height_role,
                        focus_position_m=focus,
                        analytic_scores=scores,
                        route_subset=route_subset,
                        coverage_signature=frozenset(coverage_signature),
                        path_signature=frozenset(path_signature),
                        sort_key=sort_key,
                    )
                )

    pending.sort(key=lambda item: item.sort_key)
    selected: list[_Candidate] = []
    for candidate in pending:
        if any(
            _pairwise_record(
                candidate,
                retained,
                minimum_position_m=thresholds["minimum_pairwise_position_m"],
                minimum_yaw_degrees=thresholds["minimum_pairwise_yaw_degrees"],
                minimum_coverage_jaccard_distance=thresholds[
                    "minimum_pairwise_coverage_jaccard_distance"
                ],
                minimum_path_jaccard_distance=thresholds[
                    "minimum_pairwise_path_jaccard_distance"
                ],
            )["near_duplicate"]
            for retained in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == selection_count:
            break
    if len(selected) != selection_count:
        raise MP3DViewpointPrefilterError(
            "MP3D authority produced only "
            f"{len(selected)} distinct analytic candidates; need {selection_count}"
        )

    prefix = _slug(room_id)
    viewpoints: list[MP3DViewpointCandidate] = []
    base_request_id = base_m1_request.get("request_id")
    if not isinstance(base_request_id, str) or not base_request_id:
        raise MP3DViewpointPrefilterError("base M1 request_id must be nonempty")
    for index, candidate in enumerate(selected, start=1):
        viewpoint_id = f"mp3d_{prefix}_view_{index:02d}"
        try:
            request = apply_camera_listener_pose(
                base_m1_request,
                request_id=f"{base_request_id}_{viewpoint_id}",
                position_m=candidate.camera_position_m,
                yaw_deg=candidate.camera_yaw_degrees,
            )
        except (CameraPoseError, KeyError, TypeError, ValueError) as exc:
            raise MP3DViewpointPrefilterError(
                f"could not author M1 request for {viewpoint_id}: {exc}"
            ) from exc
        if request["primary_camera_rig"].get("view_id") != "view0":
            raise MP3DViewpointPrefilterError(
                "MP3D viewpoint authoring changed the formal view0 identity"
            )
        viewpoints.append(
            MP3DViewpointCandidate(
                viewpoint_id=viewpoint_id,
                camera_position_m=candidate.camera_position_m,
                camera_yaw_degrees=candidate.camera_yaw_degrees,
                camera_height_role=candidate.camera_height_role,
                focus_position_m=candidate.focus_position_m,
                analytic_scores=candidate.analytic_scores,
                route_subset=candidate.route_subset,
                m1_request=request,
            )
        )

    pairwise = []
    for left_index, left in enumerate(selected):
        for right_index in range(left_index + 1, len(selected)):
            record = _pairwise_record(
                left,
                selected[right_index],
                minimum_position_m=thresholds["minimum_pairwise_position_m"],
                minimum_yaw_degrees=thresholds["minimum_pairwise_yaw_degrees"],
                minimum_coverage_jaccard_distance=thresholds[
                    "minimum_pairwise_coverage_jaccard_distance"
                ],
                minimum_path_jaccard_distance=thresholds[
                    "minimum_pairwise_path_jaccard_distance"
                ],
            )
            record.update(
                {
                    "left_viewpoint_id": viewpoints[left_index].viewpoint_id,
                    "right_viewpoint_id": viewpoints[right_index].viewpoint_id,
                }
            )
            pairwise.append(record)

    return MP3DViewpointPrefilterResult(
        room_id=room_id,
        viewpoints=tuple(viewpoints),
        pairwise_deduplication=tuple(pairwise),
        thresholds=thresholds,
        input_counts={
            "camera_floor_samples": len(floor_samples),
            "actor_routes": len(routes),
            "actor_route_points": actor_point_count,
            "focus_anchors": len(focus_anchors),
            "analytic_candidates_before_deduplication": len(pending),
        },
    )


__all__ = [
    "MP3DViewpointCandidate",
    "MP3DViewpointPrefilterError",
    "MP3DViewpointPrefilterResult",
    "select_mp3d_viewpoints",
]

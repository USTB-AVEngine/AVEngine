"""Post-hoc MP3D camera descriptors and continuous native-probe ranking.

This module joins a complete legal camera-placement catalog with one retained
two-source emitter track.  The join never changes placement legality.  It
records analytic centre-frustum and listener-local geometry for every camera,
then chooses a bounded native-probe subset with a continuous coverage/novelty
score.  Semantic room names, visibility thresholds, side margins and source
separation are deliberately absent from both enumeration and ranking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import numpy as np

from avengine.camera_pose import yaw_rotation_xyzw
from avengine.m2.habitat_capture import quaternion_xyzw_to_matrix
from avengine.m2.timeline import FRAME_COUNT
from avengine.qa.fact_table import (
    center_frustum_track,
    listener_local_spherical_track,
)


class MP3DProbeSubsetError(ValueError):
    """A complete camera catalog cannot be joined or ranked safely."""


_CATALOG_ARTIFACT_STATUS = "complete_target_agnostic_camera_placement_catalog"
_CATALOG_FLOOR_AUTHORITY = "live_habitat_pathfinder_declared_navmesh_export"
_PLACEMENT_STATUS = "declared_floor_position_crossed_with_height_and_yaw"
_CATALOG_AUTHORITY_VERIFIED = "verified_target_agnostic_native_floor_catalog"


@dataclass(frozen=True)
class MP3DProbeCandidateSummary:
    """Compact ranking state for one catalog placement."""

    placement: Mapping[str, Any]
    source_ids: tuple[str, str]
    per_source_in_frustum_fraction: tuple[float, float]
    per_source_in_frustum_frame_count: tuple[int, int]
    both_in_frustum_frame_count: int
    either_in_frustum_frame_count: int
    coverage_signature: frozenset[int]
    mean_source_distance_m: float
    coverage_preference_score: float
    proximity_preference_score: float
    analytic_quality_score: float
    catalog_authority_status: str

    @property
    def placement_id(self) -> str:
        return str(self.placement["placement_id"])

    @property
    def position_m(self) -> tuple[float, float, float]:
        return _finite_xyz(self.placement["position_m"], owner="placement position")

    @property
    def height_above_floor_m(self) -> float:
        floor = _finite_xyz(self.placement["floor_position_m"], owner="floor position")
        return float(self.position_m[1] - floor[1])

    @property
    def yaw_deg(self) -> float:
        return _finite_number(self.placement["yaw_deg"], owner="placement yaw")

    def ranking_record(self) -> dict[str, Any]:
        return {
            "per_source_in_frustum_fraction": dict(
                zip(
                    self.source_ids,
                    self.per_source_in_frustum_fraction,
                    strict=True,
                )
            ),
            "per_source_in_frustum_frame_count": dict(
                zip(
                    self.source_ids,
                    self.per_source_in_frustum_frame_count,
                    strict=True,
                )
            ),
            "both_in_frustum_frame_count": self.both_in_frustum_frame_count,
            "either_in_frustum_frame_count": self.either_in_frustum_frame_count,
            "coverage_preference_score": self.coverage_preference_score,
            "mean_source_distance_m": self.mean_source_distance_m,
            "proximity_preference_score": self.proximity_preference_score,
            "analytic_quality_score": self.analytic_quality_score,
        }


@dataclass(frozen=True)
class MP3DProbeSubsetAnalysis:
    """Complete compact analysis plus the selected probe records."""

    room_id: str
    source_ids: tuple[str, str]
    frame_count: int
    candidates: tuple[MP3DProbeCandidateSummary, ...]
    selected: tuple[Mapping[str, Any], ...]
    horizontal_scene_diagonal_m: float
    camera_height_span_m: float
    catalog_authority_verification: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        positions = {
            tuple(float(item) for item in record["camera"]["floor_position_m"])
            for record in self.selected
        }
        yaws = {float(record["camera"]["yaw_deg"]) for record in self.selected}
        heights = {str(record["camera"]["height_id"]) for record in self.selected}
        return {
            "schema": "avengine_mp3d_native_probe_subset_v1",
            "artifact_status": "complete_cpu_posthoc_descriptors_and_probe_subset",
            "research_only": True,
            "episode_counted": False,
            "formal_release_status": "not_run",
            "studio_used": False,
            "backend_route": "mp3d_habitat_sim_native_probe_pending",
            "room_id": self.room_id,
            "source_ids_in_array_axis_1_order": list(self.source_ids),
            "frame_count": self.frame_count,
            "input_legal_placement_count": len(self.candidates),
            "full_descriptor_count": len(self.candidates),
            "selected_probe_count": len(self.selected),
            "selected_unique_floor_position_count": len(positions),
            "selected_unique_yaw_count": len(yaws),
            "selected_height_ids": sorted(heights),
            "candidate_legality_effect": "none",
            "candidate_filtering": "none_on_input_legal_catalog",
            "difficulty_status": "descriptors_recorded_not_gated",
            "input_catalog_authority_verification": deepcopy(
                dict(self.catalog_authority_verification)
            ),
            "selection_method": (
                "greedy_continuous_coverage_quality_plus_pose_yaw_height_and_"
                "coverage_profile_novelty"
            ),
            "selection_hard_gates": [],
            "selection_weights": {
                "coverage_preference_min_source_fraction": 0.75,
                "coverage_preference_mean_source_fraction": 0.25,
                "analytic_quality_coverage": 0.90,
                "analytic_quality_proximity": 0.10,
                "selection_quality": 0.70,
                "selection_novelty": 0.30,
                "novelty_horizontal_position": 0.65,
                "novelty_yaw": 0.20,
                "novelty_coverage_profile": 0.10,
                "novelty_camera_height": 0.05,
            },
            "continuous_scales": {
                "horizontal_scene_diagonal_m": self.horizontal_scene_diagonal_m,
                "horizontal_position_tanh_scale_fraction_of_scene_diagonal": 0.25,
                "source_proximity_rational_scale_m": 4.0,
                "camera_height_span_m": self.camera_height_span_m,
            },
            "selected_candidates": [deepcopy(dict(item)) for item in self.selected],
            "native_probe_status": "pending",
            "native_occlusion_and_semantic_pixel_status": "pending",
            "claim_boundary": (
                "Every input legal placement is retained with post-hoc analytic "
                "descriptors. The requested bounded subset contains only CPU-ranked "
                "native-probe candidates; no native navigation readback, pixels, occlusion, "
                "audio, RIR, mini-pilot, certification or formal Episode is claimed."
            ),
        }


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MP3DProbeSubsetError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MP3DProbeSubsetError(f"{owner} must be a finite number")
    return result


def _finite_xyz(value: Any, *, owner: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MP3DProbeSubsetError(f"{owner} must contain three finite numbers")
    if len(value) != 3:
        raise MP3DProbeSubsetError(f"{owner} must contain three finite numbers")
    result = tuple(
        _finite_number(item, owner=f"{owner}[{index}]")
        for index, item in enumerate(value)
    )
    return result[0], result[1], result[2]


def _source_positions(value: Any) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MP3DProbeSubsetError("source positions must be finite [75,2,3]") from exc
    if result.shape != (FRAME_COUNT, 2, 3) or not np.all(np.isfinite(result)):
        raise MP3DProbeSubsetError("source positions must be finite [75,2,3]")
    return np.ascontiguousarray(result)


def _source_ids(base_request: Mapping[str, Any]) -> tuple[str, str]:
    sources = base_request.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise MP3DProbeSubsetError("base M1 request must contain two sources")
    ids = tuple(
        item.get("source_id") if isinstance(item, Mapping) else None for item in sources
    )
    if (
        len(ids) != 2
        or any(not isinstance(item, str) or not item for item in ids)
        or len(set(ids)) != 2
    ):
        raise MP3DProbeSubsetError("base M1 request must contain two unique source IDs")
    return ids  # type: ignore[return-value]


def _catalog_placements(
    catalog: Mapping[str, Any], *, base_request: Mapping[str, Any]
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    if catalog.get("artifact_status") != _CATALOG_ARTIFACT_STATUS:
        raise MP3DProbeSubsetError(
            "camera catalog artifact_status is not the target-agnostic catalog"
        )
    if catalog.get("floor_position_authority") != _CATALOG_FLOOR_AUTHORITY:
        raise MP3DProbeSubsetError(
            "camera catalog lacks the live Habitat PathFinder floor authority"
        )
    if (
        catalog.get("research_only") is not True
        or catalog.get("episode_counted") is not False
        or catalog.get("candidate_filtering") != "none"
        or catalog.get("studio_used") is not False
    ):
        raise MP3DProbeSubsetError(
            "camera catalog must be one unfiltered research-only legal catalog"
        )
    room_id = catalog.get("room_id")
    request_id = base_request.get("request_id")
    if (
        not isinstance(room_id, str)
        or not room_id
        or room_id != base_request.get("room_id")
        or not isinstance(request_id, str)
        or not request_id
    ):
        raise MP3DProbeSubsetError("camera catalog and base M1 request differ")
    raw = catalog.get("placements")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise MP3DProbeSubsetError("camera catalog placements must be nonempty")
    if catalog.get("placement_count") != len(raw):
        raise MP3DProbeSubsetError("camera catalog placement_count is inconsistent")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise MP3DProbeSubsetError(f"placement[{index}] must be an object")
        record = deepcopy(dict(item))
        placement_id = record.get("placement_id")
        if (
            not isinstance(placement_id, str)
            or not placement_id
            or placement_id in seen
        ):
            raise MP3DProbeSubsetError("placement IDs must be nonempty and unique")
        seen.add(placement_id)
        _finite_xyz(record.get("position_m"), owner=f"{placement_id} position")
        _finite_xyz(
            record.get("floor_position_m"), owner=f"{placement_id} floor position"
        )
        _finite_number(record.get("yaw_deg"), owner=f"{placement_id} yaw")
        if record.get("placement_status") != _PLACEMENT_STATUS:
            raise MP3DProbeSubsetError(
                f"{placement_id} placement_status is not a verified legal placement"
            )
        if (
            record.get("room_id") != room_id
            or record.get("base_m1_request_id") != request_id
        ):
            raise MP3DProbeSubsetError(
                f"{placement_id} differs from catalog room or base request"
            )
        records.append(record)
    ordered = tuple(
        sorted(records, key=lambda item: item["placement_id"].encode("utf-8"))
    )
    return ordered, {
        "status": _CATALOG_AUTHORITY_VERIFIED,
        "artifact_status": _CATALOG_ARTIFACT_STATUS,
        "floor_position_authority": _CATALOG_FLOOR_AUTHORITY,
        "required_placement_status": _PLACEMENT_STATUS,
        "all_placements_match_required_status": True,
        "verified_placement_count": len(ordered),
    }


def _orientation_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    x, y, z, w = yaw_rotation_xyzw(yaw_deg)
    return float(w), float(x), float(y), float(z)


def _center_geometry(
    source_positions: np.ndarray,
    *,
    camera_position_m: tuple[float, float, float],
    yaw_deg: float,
    hfov_degrees: float,
    resolution_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rotation_xyzw = yaw_rotation_xyzw(yaw_deg)
    rotation = quaternion_xyzw_to_matrix(rotation_xyzw)
    listener_wxyz = _orientation_wxyz(yaw_deg)
    origin = np.asarray(camera_position_m, dtype=np.float64)
    height, width = resolution_hw
    tan_half_h = math.tan(math.radians(hfov_degrees) * 0.5)
    tan_half_v = tan_half_h * float(height) / float(width)
    masks: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for source_index in range(2):
        positions = source_positions[:, source_index, :]
        spherical = listener_local_spherical_track(
            positions,
            origin,
            listener_wxyz,
        )
        frustum = center_frustum_track(
            positions,
            origin,
            listener_wxyz,
            hfov_degrees=hfov_degrees,
            resolution_hw=resolution_hw,
        )
        local = (positions - origin) @ rotation
        depth = -local[:, 2]
        valid_depth = depth > 0.0
        normalized_x: list[float | None] = []
        normalized_y: list[float | None] = []
        for frame_index in range(FRAME_COUNT):
            if not valid_depth[frame_index]:
                normalized_x.append(None)
                normalized_y.append(None)
                continue
            x_value = float(local[frame_index, 0] / (depth[frame_index] * tan_half_h))
            y_value = float(local[frame_index, 1] / (depth[frame_index] * tan_half_v))
            normalized_x.append(x_value if math.isfinite(x_value) else None)
            normalized_y.append(y_value if math.isfinite(y_value) else None)
        mask = np.asarray(frustum["in_frustum"], dtype=np.bool_)
        masks.append(mask)
        distance = np.asarray(spherical["distance_m"], dtype=np.float64)
        distances.append(distance)
        records.append(
            {
                "listener_local_azimuth_deg": spherical["azimuth_deg"],
                "listener_local_elevation_deg": spherical["elevation_deg"],
                "listener_distance_m": spherical["distance_m"],
                "camera_forward_depth_m": [float(item) for item in depth],
                "normalized_pinhole_x_right": normalized_x,
                "normalized_pinhole_y_up": normalized_y,
                "center_in_frustum": frustum["in_frustum"],
                "center_in_frustum_frame_count": frustum["in_frustum_frame_count"],
                "center_in_frustum_fraction": float(np.mean(mask)),
                "always_outside_center_frustum": frustum["always_outside_frustum"],
                "always_inside_center_frustum": frustum["always_inside_frustum"],
                "center_frustum_events": frustum["events"],
                "native_occlusion_status": "pending",
                "native_semantic_visible_pixels": None,
            }
        )
    return np.stack(masks, axis=1), np.stack(distances, axis=1), records


def _calibration(base_request: Mapping[str, Any]) -> tuple[float, tuple[int, int]]:
    try:
        calibration = base_request["primary_camera_rig"]["shared_calibration"]
        hfov = _finite_number(calibration["hfov_degrees"], owner="camera hfov")
        raw_resolution = calibration["resolution_hw"]
        height, width = int(raw_resolution[0]), int(raw_resolution[1])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise MP3DProbeSubsetError("base M1 request lacks pinhole calibration") from exc
    if not 0.0 < hfov < 180.0 or height <= 0 or width <= 0:
        raise MP3DProbeSubsetError("base M1 pinhole calibration is invalid")
    return hfov, (height, width)


def analyze_mp3d_probe_candidates(
    *,
    catalog: Mapping[str, Any],
    base_request: Mapping[str, Any],
    source_positions_m: Any,
    selection_count: int = 12,
) -> MP3DProbeSubsetAnalysis:
    """Describe all legal placements compactly and rank a probe-only subset."""

    if isinstance(selection_count, bool) or not isinstance(selection_count, int):
        raise MP3DProbeSubsetError("selection_count must be an integer")
    source_positions = _source_positions(source_positions_m)
    source_ids = _source_ids(base_request)
    placements, catalog_authority_verification = _catalog_placements(
        catalog, base_request=base_request
    )
    if not 1 <= selection_count <= len(placements):
        raise MP3DProbeSubsetError(
            "selection_count must be positive and no larger than the catalog"
        )
    hfov, resolution = _calibration(base_request)
    candidates: list[MP3DProbeCandidateSummary] = []
    for placement in placements:
        position = _finite_xyz(placement["position_m"], owner="placement position")
        yaw = _finite_number(placement["yaw_deg"], owner="placement yaw")
        masks, distances, _records = _center_geometry(
            source_positions,
            camera_position_m=position,
            yaw_deg=yaw,
            hfov_degrees=hfov,
            resolution_hw=resolution,
        )
        fractions = np.mean(masks, axis=0)
        counts = np.count_nonzero(masks, axis=0)
        coverage_preference = float(
            0.75 * np.min(fractions) + 0.25 * np.mean(fractions)
        )
        mean_distance = float(np.mean(distances))
        proximity = float(1.0 / (1.0 + mean_distance / 4.0))
        quality = float(0.90 * coverage_preference + 0.10 * proximity)
        candidates.append(
            MP3DProbeCandidateSummary(
                placement=placement,
                source_ids=source_ids,
                per_source_in_frustum_fraction=(
                    float(fractions[0]),
                    float(fractions[1]),
                ),
                per_source_in_frustum_frame_count=(int(counts[0]), int(counts[1])),
                both_in_frustum_frame_count=int(
                    np.count_nonzero(np.all(masks, axis=1))
                ),
                either_in_frustum_frame_count=int(
                    np.count_nonzero(np.any(masks, axis=1))
                ),
                coverage_signature=frozenset(
                    int(item) for item in np.flatnonzero(masks.reshape(-1))
                ),
                mean_source_distance_m=mean_distance,
                coverage_preference_score=coverage_preference,
                proximity_preference_score=proximity,
                analytic_quality_score=quality,
                catalog_authority_status=_CATALOG_AUTHORITY_VERIFIED,
            )
        )

    position_xz = np.asarray(
        [[item.position_m[0], item.position_m[2]] for item in candidates],
        dtype=np.float64,
    )
    scene_diagonal = float(
        np.linalg.norm(np.max(position_xz, axis=0) - np.min(position_xz, axis=0))
    )
    camera_heights = np.asarray(
        [item.height_above_floor_m for item in candidates], dtype=np.float64
    )
    height_span = float(np.max(camera_heights) - np.min(camera_heights))
    selected = _select_continuous_probe_subset(
        candidates,
        selection_count=selection_count,
        horizontal_scene_diagonal_m=scene_diagonal,
        camera_height_span_m=height_span,
    )
    return MP3DProbeSubsetAnalysis(
        room_id=str(catalog["room_id"]),
        source_ids=source_ids,
        frame_count=FRAME_COUNT,
        candidates=tuple(candidates),
        selected=selected,
        horizontal_scene_diagonal_m=scene_diagonal,
        camera_height_span_m=height_span,
        catalog_authority_verification=catalog_authority_verification,
    )


def _jaccard_distance(left: frozenset[int], right: frozenset[int]) -> float:
    union = left | right
    if not union:
        return 0.0
    return float(1.0 - len(left & right) / len(union))


def _circular_yaw_distance(left: float, right: float) -> float:
    return float(abs((left - right + 180.0) % 360.0 - 180.0))


def _pair_novelty(
    candidate: MP3DProbeCandidateSummary,
    selected: MP3DProbeCandidateSummary,
    *,
    horizontal_scene_diagonal_m: float,
    camera_height_span_m: float,
) -> dict[str, Any]:
    left = candidate.position_m
    right = selected.position_m
    horizontal_distance = math.dist((left[0], left[2]), (right[0], right[2]))
    position_scale = max(horizontal_scene_diagonal_m * 0.25, 1.0e-9)
    position_novelty = float(math.tanh(horizontal_distance / position_scale))
    yaw_difference = _circular_yaw_distance(candidate.yaw_deg, selected.yaw_deg)
    yaw_novelty = float(yaw_difference / 180.0)
    profile_novelty = _jaccard_distance(
        candidate.coverage_signature, selected.coverage_signature
    )
    height_difference = abs(
        candidate.height_above_floor_m - selected.height_above_floor_m
    )
    height_novelty = float(
        min(height_difference / camera_height_span_m, 1.0)
        if camera_height_span_m > 1.0e-12
        else 0.0
    )
    combined = float(
        0.65 * position_novelty
        + 0.20 * yaw_novelty
        + 0.10 * profile_novelty
        + 0.05 * height_novelty
    )
    return {
        "nearest_previously_selected_placement_id": selected.placement_id,
        "horizontal_position_distance_m": horizontal_distance,
        "horizontal_position_novelty": position_novelty,
        "circular_yaw_difference_deg": yaw_difference,
        "yaw_novelty": yaw_novelty,
        "coverage_profile_jaccard_novelty": profile_novelty,
        "camera_height_difference_m": height_difference,
        "camera_height_novelty": height_novelty,
        "combined_novelty": combined,
    }


def _select_continuous_probe_subset(
    candidates: Sequence[MP3DProbeCandidateSummary],
    *,
    selection_count: int,
    horizontal_scene_diagonal_m: float,
    camera_height_span_m: float,
) -> tuple[Mapping[str, Any], ...]:
    height_priority = {"base": 0, "low": 1, "high": 2}
    selected_candidates: list[MP3DProbeCandidateSummary] = []
    selected_records: list[dict[str, Any]] = []
    remaining = {item.placement_id: item for item in candidates}
    for selection_order in range(1, selection_count + 1):
        ranked: list[
            tuple[tuple[Any, ...], MP3DProbeCandidateSummary, dict[str, Any]]
        ] = []
        for candidate in remaining.values():
            if not selected_candidates:
                novelty = {
                    "nearest_previously_selected_placement_id": None,
                    "horizontal_position_distance_m": None,
                    "horizontal_position_novelty": 1.0,
                    "circular_yaw_difference_deg": None,
                    "yaw_novelty": 1.0,
                    "coverage_profile_jaccard_novelty": 1.0,
                    "camera_height_difference_m": None,
                    "camera_height_novelty": 1.0,
                    "combined_novelty": 1.0,
                }
            else:
                pair_records = [
                    _pair_novelty(
                        candidate,
                        retained,
                        horizontal_scene_diagonal_m=horizontal_scene_diagonal_m,
                        camera_height_span_m=camera_height_span_m,
                    )
                    for retained in selected_candidates
                ]
                novelty = min(
                    pair_records,
                    key=lambda item: (
                        item["combined_novelty"],
                        str(item["nearest_previously_selected_placement_id"]).encode(
                            "utf-8"
                        ),
                    ),
                )
            score = float(
                0.70 * candidate.analytic_quality_score
                + 0.30 * float(novelty["combined_novelty"])
            )
            height_id = str(candidate.placement.get("height_id", ""))
            sort_key = (
                -score,
                -candidate.analytic_quality_score,
                -candidate.coverage_preference_score,
                -float(novelty["combined_novelty"]),
                height_priority.get(height_id, 3),
                candidate.placement_id.encode("utf-8"),
            )
            ranked.append((sort_key, candidate, {**novelty, "selection_score": score}))
        _key, winner, selection = min(ranked, key=lambda item: item[0])
        placement = winner.placement
        selected_records.append(
            {
                "selection_order": selection_order,
                "placement_id": winner.placement_id,
                "camera": {
                    "position_m": list(winner.position_m),
                    "floor_position_m": list(
                        _finite_xyz(
                            placement["floor_position_m"], owner="floor position"
                        )
                    ),
                    "yaw_deg": winner.yaw_deg,
                    "height_id": placement.get("height_id"),
                    "floor_sample_index": placement.get("floor_sample_index"),
                },
                "ranking": {
                    **winner.ranking_record(),
                    **selection,
                },
                "selection_reason": (
                    "highest remaining continuous analytic quality plus novelty "
                    "score at this greedy round; no visibility or diversity gate"
                ),
                "input_placement_legality_preserved": True,
                "input_catalog_authority_status": winner.catalog_authority_status,
                "native_probe_status": "pending",
            }
        )
        selected_candidates.append(winner)
        del remaining[winner.placement_id]
    return tuple(selected_records)


def build_placement_descriptor(
    candidate: MP3DProbeCandidateSummary,
    *,
    base_request: Mapping[str, Any],
    source_positions_m: Any,
) -> dict[str, Any]:
    """Return the full columnar per-frame descriptor for one legal placement."""

    if candidate.catalog_authority_status != _CATALOG_AUTHORITY_VERIFIED:
        raise MP3DProbeSubsetError(
            "placement descriptor requires verified native-floor catalog authority"
        )
    source_positions = _source_positions(source_positions_m)
    hfov, resolution = _calibration(base_request)
    masks, _distances, records = _center_geometry(
        source_positions,
        camera_position_m=candidate.position_m,
        yaw_deg=candidate.yaw_deg,
        hfov_degrees=hfov,
        resolution_hw=resolution,
    )
    per_source = {
        source_id: record
        for source_id, record in zip(candidate.source_ids, records, strict=True)
    }
    return {
        "placement": deepcopy(dict(candidate.placement)),
        "candidate_legality": {
            "input_status": candidate.placement.get("placement_status"),
            "catalog_authority_status": candidate.catalog_authority_status,
            "descriptor_effect": "none",
            "preserved": True,
        },
        "camera_orientation_xyzw": yaw_rotation_xyzw(candidate.yaw_deg),
        "frame_count": FRAME_COUNT,
        "per_source": per_source,
        "joint_center_frustum": {
            "both_in_frustum": [bool(item) for item in np.all(masks, axis=1)],
            "either_in_frustum": [bool(item) for item in np.any(masks, axis=1)],
            "both_in_frustum_frame_count": int(np.count_nonzero(np.all(masks, axis=1))),
            "either_in_frustum_frame_count": int(
                np.count_nonzero(np.any(masks, axis=1))
            ),
        },
        "ranking_inputs": candidate.ranking_record(),
        "semantic_room_or_target_used_for_generation_or_ranking": False,
        "difficulty_status": "raw_descriptors_recorded_not_gated",
        "native_occlusion_and_semantic_pixel_status": "pending",
    }


__all__ = [
    "MP3DProbeCandidateSummary",
    "MP3DProbeSubsetAnalysis",
    "MP3DProbeSubsetError",
    "analyze_mp3d_probe_candidates",
    "build_placement_descriptor",
]

"""Target-agnostic per-region MP3D CPU viewpoint-bank selection.

The input camera catalog keeps its original legality.  This module consumes an
additive static ``.house`` membership sidecar, uses only unique membership for
each region's quota, and chooses distinct floor positions with continuous pose
and boundary preferences.  Region categories and source routes are never
selection inputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
from statistics import median
from typing import Any


_SIDECAR_STATUS = "complete_additive_mp3d_camera_region_membership_sidecar"
_CATALOG_STATUS = "complete_target_agnostic_camera_placement_catalog"
_CATALOG_FLOOR_AUTHORITY = "live_habitat_pathfinder_declared_navmesh_export"
_PLACEMENT_STATUS = "declared_floor_position_crossed_with_height_and_yaw"
_MEMBERSHIP_STATUSES = ("unique", "boundary", "unknown", "ambiguous")


class MP3DRegionViewpointError(ValueError):
    """The sidecar and catalog cannot produce truthful per-region banks."""


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MP3DRegionViewpointError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MP3DRegionViewpointError(f"{owner} must be a finite number")
    return result


def _finite_xyz(value: Any, *, owner: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MP3DRegionViewpointError(f"{owner} must contain three finite numbers")
    if len(value) != 3:
        raise MP3DRegionViewpointError(f"{owner} must contain three finite numbers")
    result = tuple(
        _finite_number(item, owner=f"{owner}[{index}]")
        for index, item in enumerate(value)
    )
    return result[0], result[1], result[2]


def _integer(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MP3DRegionViewpointError(f"{owner} must be an integer")
    return value


def _string(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise MP3DRegionViewpointError(f"{owner} must be a nonempty string")
    return value


@dataclass(frozen=True)
class MP3DRegionViewCandidate:
    placement: Mapping[str, Any]
    membership: Mapping[str, Any]

    @property
    def placement_id(self) -> str:
        return str(self.placement["placement_id"])

    @property
    def floor_sample_index(self) -> int:
        return int(self.placement["floor_sample_index"])

    @property
    def floor_position_m(self) -> tuple[float, float, float]:
        return _finite_xyz(
            self.placement["floor_position_m"], owner="candidate floor position"
        )

    @property
    def position_m(self) -> tuple[float, float, float]:
        return _finite_xyz(self.placement["position_m"], owner="candidate position")

    @property
    def yaw_deg(self) -> float:
        return _finite_number(self.placement["yaw_deg"], owner="candidate yaw")

    @property
    def height_id(self) -> str:
        return str(self.placement["height_id"])

    @property
    def height_above_floor_m(self) -> float:
        return self.position_m[1] - self.floor_position_m[1]

    @property
    def boundary_distance_m(self) -> float:
        return _finite_number(
            self.membership["distance_to_nearest_polygon_boundary_m"],
            owner="candidate boundary distance",
        )


@dataclass(frozen=True)
class MP3DRegionViewpointAnalysis:
    room_id: str
    views_per_region: int
    input_floor_sample_count: int
    input_placement_count: int
    floor_membership_status_counts: Mapping[str, int]
    placement_membership_status_counts: Mapping[str, int]
    region_banks: tuple[Mapping[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        selected = [
            selection
            for region in self.region_banks
            for selection in region["selected_candidates"]
        ]
        yaw_counts = Counter(str(float(item["camera"]["yaw_deg"])) for item in selected)
        height_counts = Counter(str(item["camera"]["height_id"]) for item in selected)
        return {
            "schema": "avengine_mp3d_region_viewpoint_banks_v1",
            "artifact_status": "complete_cpu_mp3d_region_viewpoint_banks",
            "research_only": True,
            "episode_counted": False,
            "formal_release_status": "not_run",
            "studio_used": False,
            "backend_route": "mp3d_habitat_sim_native_capture_pending",
            "room_id": self.room_id,
            "input_floor_sample_count": self.input_floor_sample_count,
            "input_placement_count": self.input_placement_count,
            "input_floor_membership_status_counts": {
                status: int(self.floor_membership_status_counts.get(status, 0))
                for status in _MEMBERSHIP_STATUSES
            },
            "input_placement_membership_status_counts": {
                status: int(self.placement_membership_status_counts.get(status, 0))
                for status in _MEMBERSHIP_STATUSES
            },
            "input_nonunique_memberships_retained": True,
            "input_candidate_legality_effect": "none",
            "input_catalog_filtering_effect": "none",
            "region_quota_membership": "unique_only",
            "region_quota_distinct_floor_positions_required": True,
            "views_per_region": self.views_per_region,
            "region_bank_count": len(self.region_banks),
            "selected_viewpoint_count": len(selected),
            "selected_unique_floor_position_count": len(
                {
                    (
                        item["region_instance_id"],
                        tuple(item["camera"]["floor_position_m"]),
                    )
                    for item in selected
                }
            ),
            "selected_yaw_distribution": dict(sorted(yaw_counts.items())),
            "selected_height_id_distribution": dict(sorted(height_counts.items())),
            "selection_method": (
                "per_region_greedy_continuous_boundary_and_central_height_"
                "quality_plus_floor_position_yaw_and_height_novelty"
            ),
            "selection_hard_numeric_thresholds": [],
            "selection_weights": {
                "quality_boundary_distance": 0.90,
                "quality_central_height": 0.10,
                "selection_quality": 0.40,
                "selection_novelty": 0.60,
                "novelty_floor_position": 0.70,
                "novelty_circular_yaw": 0.20,
                "novelty_height": 0.10,
            },
            "selection_inputs_used": [
                "floor_position_m",
                "yaw_deg",
                "height_above_floor_m",
                "distance_to_nearest_polygon_boundary_m",
            ],
            "selection_inputs_not_used": [
                "region_category_code_or_name",
                "semantic_target",
                "source_or_actor_route",
                "source_coverage_or_visibility",
                "source_identity_or_type",
                "60cm_separation",
                "side_margin",
                "crossing_or_exchange",
            ],
            "region_banks": [deepcopy(dict(item)) for item in self.region_banks],
            "native_capture_status": "pending",
            "native_viewpoint_certification_status": "pending",
            "source_path_join_status": "pending",
            "audio_rir_status": "pending",
            "four_point_mini_pilot_status": "pending",
            "claim_boundary": (
                "Forty CPU-selected camera requests span four distinct legal floor "
                "positions per static uniquely-bound region. Category names and the "
                "current two-source route are not ranking inputs. No native pixels, "
                "camera/listener readback, actor path, audio/RIR, mini-pilot, "
                "certified viewpoint or formal Episode is claimed."
            ),
        }


def _status_counter(
    records: Sequence[Mapping[str, Any]], *, owner: str
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for index, record in enumerate(records):
        status = record.get("membership_status")
        if status not in _MEMBERSHIP_STATUSES:
            raise MP3DRegionViewpointError(
                f"{owner}[{index}] has invalid membership_status"
            )
        counter[str(status)] += 1
    return counter


def _validate_top_level(
    sidecar: Mapping[str, Any],
    catalog: Mapping[str, Any],
    base_request: Mapping[str, Any],
) -> tuple[str, str, frozenset[tuple[str, float]]]:
    room_id = _string(sidecar.get("room_id"), owner="sidecar room_id")
    request_id = _string(base_request.get("request_id"), owner="base request_id")
    if base_request.get("room_id") != room_id or catalog.get("room_id") != room_id:
        raise MP3DRegionViewpointError(
            "sidecar, catalog and base M1 request room IDs differ"
        )
    if (
        sidecar.get("artifact_status") != _SIDECAR_STATUS
        or sidecar.get("research_only") is not True
        or sidecar.get("episode_counted") is not False
        or sidecar.get("formal_release_status") != "not_run"
        or sidecar.get("candidate_filtering") != "none"
        or sidecar.get("candidate_legality_effect") != "none"
        or sidecar.get("boundary_descriptor_is_legality_gate") is not False
        or sidecar.get("placements_per_floor_sample") != 36
        or sidecar.get("studio_used") is not False
    ):
        raise MP3DRegionViewpointError(
            "camera region sidecar is not the additive unfiltered research artifact"
        )
    if (
        catalog.get("artifact_status") != _CATALOG_STATUS
        or catalog.get("floor_position_authority") != _CATALOG_FLOOR_AUTHORITY
        or catalog.get("research_only") is not True
        or catalog.get("episode_counted") is not False
        or catalog.get("formal_release_status") != "not_run"
        or catalog.get("candidate_filtering") != "none"
        or catalog.get("studio_used") is not False
    ):
        raise MP3DRegionViewpointError(
            "camera catalog is not the complete unfiltered native-floor artifact"
        )
    raw_heights = catalog.get("camera_height_variants_m")
    raw_yaws = catalog.get("yaw_degrees")
    if (
        not isinstance(raw_heights, Mapping)
        or not isinstance(raw_yaws, Sequence)
        or isinstance(raw_yaws, (str, bytes))
    ):
        raise MP3DRegionViewpointError(
            "camera catalog lacks declared height/yaw variants"
        )
    height_ids = tuple(
        _string(value, owner="catalog camera height ID") for value in raw_heights
    )
    for height_id in height_ids:
        _finite_number(
            raw_heights[height_id], owner=f"catalog camera height {height_id}"
        )
    yaw_values = tuple(
        _finite_number(value, owner="catalog yaw value") for value in raw_yaws
    )
    expected_combinations = frozenset(
        (height_id, yaw) for height_id in height_ids for yaw in yaw_values
    )
    if (
        len(height_ids) != 3
        or len(yaw_values) != 12
        or len(expected_combinations) != 36
    ):
        raise MP3DRegionViewpointError(
            "camera catalog must declare three heights and twelve unique yaw values"
        )
    return room_id, request_id, expected_combinations


def _region_records(sidecar: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = sidecar.get("regions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise MP3DRegionViewpointError("sidecar regions must be nonempty")
    result: dict[str, dict[str, Any]] = {}
    seen_indices: set[int] = set()
    for ordinal, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise MP3DRegionViewpointError(f"regions[{ordinal}] must be an object")
        record = deepcopy(dict(value))
        region_id = _string(
            record.get("region_instance_id"),
            owner=f"regions[{ordinal}] region_instance_id",
        )
        region_index = _integer(
            record.get("region_index"), owner=f"regions[{ordinal}] region_index"
        )
        category_code = record.get("category_code")
        category_name = record.get("category_name")
        if (
            region_id in result
            or region_index in seen_indices
            or not isinstance(category_code, str)
            or len(category_code) != 1
            or (category_name is not None and not isinstance(category_name, str))
        ):
            raise MP3DRegionViewpointError(
                "region IDs/indices must be unique and category labels well formed"
            )
        seen_indices.add(region_index)
        result[region_id] = record
    if sidecar.get("region_instance_count") != len(result):
        raise MP3DRegionViewpointError("sidecar region_instance_count drift")
    return result


def _floor_records(
    sidecar: Mapping[str, Any], regions: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[int, dict[str, Any]], Counter[str]]:
    raw = sidecar.get("floor_sample_memberships")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise MP3DRegionViewpointError(
            "sidecar floor_sample_memberships must be nonempty"
        )
    result: dict[int, dict[str, Any]] = {}
    seen_placement_ids: set[str] = set()
    for ordinal, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise MP3DRegionViewpointError(
                f"floor_sample_memberships[{ordinal}] must be an object"
            )
        record = deepcopy(dict(value))
        floor_index = _integer(
            record.get("floor_sample_index"),
            owner=f"floor_sample_memberships[{ordinal}] floor_sample_index",
        )
        if floor_index in result:
            raise MP3DRegionViewpointError("floor_sample_index values must be unique")
        _finite_xyz(
            record.get("floor_position_m"), owner=f"floor sample {floor_index} position"
        )
        distance = _finite_number(
            record.get("distance_to_nearest_polygon_boundary_m"),
            owner=f"floor sample {floor_index} boundary distance",
        )
        if distance < 0.0 or not isinstance(
            record.get("near_boundary_descriptor"), bool
        ):
            raise MP3DRegionViewpointError(
                f"floor sample {floor_index} has invalid boundary descriptors"
            )
        status = record.get("membership_status")
        member_ids = record.get("member_region_instance_ids")
        if (
            status not in _MEMBERSHIP_STATUSES
            or not isinstance(member_ids, Sequence)
            or isinstance(member_ids, (str, bytes))
            or any(item not in regions for item in member_ids)
            or len(set(member_ids)) != len(member_ids)
        ):
            raise MP3DRegionViewpointError(
                f"floor sample {floor_index} has invalid region membership"
            )
        primary = record.get("primary_region_instance_id")
        if status == "unique" and (len(member_ids) != 1 or primary != member_ids[0]):
            raise MP3DRegionViewpointError(
                f"floor sample {floor_index} unique membership is inconsistent"
            )
        if status == "unknown" and (member_ids or primary is not None):
            raise MP3DRegionViewpointError(
                f"floor sample {floor_index} unknown membership was coerced"
            )
        if status == "ambiguous" and len(member_ids) < 2:
            raise MP3DRegionViewpointError(
                f"floor sample {floor_index} ambiguous membership lost member IDs"
            )
        placement_ids = record.get("placement_ids")
        if (
            not isinstance(placement_ids, Sequence)
            or isinstance(placement_ids, (str, bytes))
            or len(placement_ids) != 36
            or record.get("placement_count") != 36
            or any(not isinstance(item, str) or not item for item in placement_ids)
            or len(set(placement_ids)) != 36
            or seen_placement_ids.intersection(placement_ids)
        ):
            raise MP3DRegionViewpointError(
                f"floor sample {floor_index} must retain 36 unique placement IDs"
            )
        seen_placement_ids.update(str(item) for item in placement_ids)
        result[floor_index] = record
    if set(result) != set(range(len(result))):
        raise MP3DRegionViewpointError(
            "floor_sample_index values must be contiguous from zero"
        )
    if sidecar.get("floor_sample_count") != len(result):
        raise MP3DRegionViewpointError("sidecar floor_sample_count drift")
    counter = _status_counter(tuple(result.values()), owner="floor memberships")
    declared = sidecar.get("floor_membership_status_counts")
    if not isinstance(declared, Mapping) or any(
        declared.get(status, 0) != counter[status] for status in _MEMBERSHIP_STATUSES
    ):
        raise MP3DRegionViewpointError("sidecar floor membership status counts drift")
    return result, counter


def _catalog_records(
    catalog: Mapping[str, Any],
    *,
    request_id: str,
    expected_combinations: frozenset[tuple[str, float]],
) -> dict[str, dict[str, Any]]:
    raw = catalog.get("placements")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise MP3DRegionViewpointError("camera catalog placements must be nonempty")
    result: dict[str, dict[str, Any]] = {}
    combinations_by_floor: dict[int, set[tuple[str, float]]] = defaultdict(set)
    for ordinal, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise MP3DRegionViewpointError(
                f"catalog placements[{ordinal}] must be an object"
            )
        record = deepcopy(dict(value))
        placement_id = _string(
            record.get("placement_id"),
            owner=f"catalog placements[{ordinal}] placement_id",
        )
        if placement_id in result:
            raise MP3DRegionViewpointError("catalog placement IDs must be unique")
        floor_index = _integer(
            record.get("floor_sample_index"), owner=f"{placement_id} floor index"
        )
        _finite_xyz(record.get("floor_position_m"), owner=f"{placement_id} floor")
        _finite_xyz(record.get("position_m"), owner=f"{placement_id} position")
        _finite_number(record.get("yaw_deg"), owner=f"{placement_id} yaw")
        if (
            record.get("placement_status") != _PLACEMENT_STATUS
            or record.get("base_m1_request_id") != request_id
            or not isinstance(record.get("height_id"), str)
            or not record.get("height_id")
        ):
            raise MP3DRegionViewpointError(
                f"{placement_id} lacks matching legal catalog/request authority"
            )
        combination = (
            str(record["height_id"]),
            _finite_number(record["yaw_deg"], owner=f"{placement_id} yaw"),
        )
        if (
            combination not in expected_combinations
            or combination in combinations_by_floor[floor_index]
        ):
            raise MP3DRegionViewpointError(
                f"{placement_id} duplicates or leaves the declared height/yaw grid"
            )
        combinations_by_floor[floor_index].add(combination)
        result[placement_id] = record
    if catalog.get("placement_count") != len(result):
        raise MP3DRegionViewpointError("camera catalog placement_count drift")
    if any(
        combinations != expected_combinations
        for combinations in combinations_by_floor.values()
    ):
        raise MP3DRegionViewpointError(
            "each catalog floor sample must retain all 36 height/yaw placements"
        )
    return result


def _joined_candidates(
    sidecar: Mapping[str, Any],
    floor_records: Mapping[int, Mapping[str, Any]],
    catalog_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[MP3DRegionViewCandidate]], Counter[str]]:
    raw = sidecar.get("placement_memberships")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise MP3DRegionViewpointError("sidecar placement_memberships must be nonempty")
    seen: set[str] = set()
    eligible: dict[str, list[MP3DRegionViewCandidate]] = defaultdict(list)
    sidecar_records: list[Mapping[str, Any]] = []
    for ordinal, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise MP3DRegionViewpointError(
                f"placement_memberships[{ordinal}] must be an object"
            )
        membership = deepcopy(dict(value))
        placement_id = _string(
            membership.get("placement_id"),
            owner=f"placement_memberships[{ordinal}] placement_id",
        )
        if placement_id in seen or placement_id not in catalog_records:
            raise MP3DRegionViewpointError(
                "sidecar/catalog placement ID drift or duplication"
            )
        seen.add(placement_id)
        catalog = catalog_records[placement_id]
        floor_index = _integer(
            membership.get("floor_sample_index"), owner=f"{placement_id} floor index"
        )
        if floor_index not in floor_records:
            raise MP3DRegionViewpointError(
                f"{placement_id} references an unknown floor sample"
            )
        floor = floor_records[floor_index]
        exact_fields = (
            "floor_sample_index",
            "floor_position_m",
            "position_m",
            "yaw_deg",
            "height_id",
        )
        if any(membership.get(field) != catalog.get(field) for field in exact_fields):
            raise MP3DRegionViewpointError(
                f"{placement_id} sidecar/catalog pose or coordinate drift"
            )
        if (
            membership.get("floor_position_m") != floor.get("floor_position_m")
            or membership.get("membership_status") != floor.get("membership_status")
            or membership.get("member_region_instance_ids")
            != floor.get("member_region_instance_ids")
            or membership.get("primary_region_instance_id")
            != floor.get("primary_region_instance_id")
            or membership.get("candidate_legality_effect") != "none"
            or placement_id not in floor.get("placement_ids", ())
        ):
            raise MP3DRegionViewpointError(
                f"{placement_id} placement/floor membership drift"
            )
        sidecar_records.append(membership)
        if membership.get("membership_status") == "unique":
            region_id = str(membership["primary_region_instance_id"])
            eligible[region_id].append(
                MP3DRegionViewCandidate(placement=catalog, membership=membership)
            )
    if seen != set(catalog_records):
        raise MP3DRegionViewpointError("sidecar and catalog placement ID sets differ")
    if sidecar.get("placement_count") != len(sidecar_records):
        raise MP3DRegionViewpointError("sidecar placement_count drift")
    counter = _status_counter(sidecar_records, owner="placement memberships")
    declared = sidecar.get("placement_membership_status_counts")
    if not isinstance(declared, Mapping) or any(
        declared.get(status, 0) != counter[status] for status in _MEMBERSHIP_STATUSES
    ):
        raise MP3DRegionViewpointError(
            "sidecar placement membership status counts drift"
        )
    return eligible, counter


def _circular_yaw_distance(left: float, right: float) -> float:
    return float(abs((left - right + 180.0) % 360.0 - 180.0))


def _candidate_quality(
    candidate: MP3DRegionViewCandidate,
    *,
    boundary_scale_m: float,
    height_min_m: float,
    height_max_m: float,
) -> dict[str, float]:
    boundary = candidate.boundary_distance_m
    boundary_preference = float(boundary / (boundary + boundary_scale_m))
    height_center = (height_min_m + height_max_m) * 0.5
    height_half_span = (height_max_m - height_min_m) * 0.5
    central_height = float(
        1.0
        - min(
            abs(candidate.height_above_floor_m - height_center)
            / max(height_half_span, 1.0e-12),
            1.0,
        )
    )
    quality = float(0.90 * boundary_preference + 0.10 * central_height)
    return {
        "distance_to_nearest_polygon_boundary_m": boundary,
        "boundary_distance_rational_scale_m": boundary_scale_m,
        "boundary_distance_preference": boundary_preference,
        "height_above_floor_m": candidate.height_above_floor_m,
        "central_height_preference": central_height,
        "continuous_quality_preference": quality,
    }


def _pair_novelty(
    candidate: MP3DRegionViewCandidate,
    selected: MP3DRegionViewCandidate,
    *,
    position_scale_m: float,
    height_span_m: float,
) -> dict[str, Any]:
    distance = math.dist(
        (candidate.floor_position_m[0], candidate.floor_position_m[2]),
        (selected.floor_position_m[0], selected.floor_position_m[2]),
    )
    position_novelty = float(math.tanh(distance / position_scale_m))
    yaw_difference = _circular_yaw_distance(candidate.yaw_deg, selected.yaw_deg)
    yaw_novelty = float(yaw_difference / 180.0)
    height_difference = abs(
        candidate.height_above_floor_m - selected.height_above_floor_m
    )
    height_novelty = float(
        min(height_difference / height_span_m, 1.0) if height_span_m > 1.0e-12 else 0.0
    )
    combined = float(
        0.70 * position_novelty + 0.20 * yaw_novelty + 0.10 * height_novelty
    )
    return {
        "nearest_previously_selected_placement_id": selected.placement_id,
        "horizontal_floor_position_distance_m": distance,
        "horizontal_floor_position_novelty": position_novelty,
        "circular_yaw_difference_deg": yaw_difference,
        "circular_yaw_novelty": yaw_novelty,
        "height_difference_m": height_difference,
        "height_novelty": height_novelty,
        "continuous_novelty": combined,
    }


def _select_region(
    region: Mapping[str, Any],
    candidates: Sequence[MP3DRegionViewCandidate],
    *,
    views_per_region: int,
) -> dict[str, Any]:
    by_floor: dict[int, list[MP3DRegionViewCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_floor[candidate.floor_sample_index].append(candidate)
    region_id = str(region["region_instance_id"])
    if len(by_floor) < views_per_region:
        raise MP3DRegionViewpointError(
            f"region {region_id} has only {len(by_floor)} unique floor positions; "
            f"needs {views_per_region} without copying a position"
        )
    floor_positions = tuple(
        candidate_group[0].floor_position_m
        for _index, candidate_group in sorted(by_floor.items())
    )
    min_x = min(point[0] for point in floor_positions)
    max_x = max(point[0] for point in floor_positions)
    min_z = min(point[2] for point in floor_positions)
    max_z = max(point[2] for point in floor_positions)
    scene_diagonal = math.dist((min_x, min_z), (max_x, max_z))
    position_scale = max(scene_diagonal * 0.25, 1.0e-9)
    boundary_values = tuple(
        candidate_group[0].boundary_distance_m for candidate_group in by_floor.values()
    )
    boundary_scale = max(float(median(boundary_values)), 1.0e-9)
    heights = tuple(candidate.height_above_floor_m for candidate in candidates)
    height_min = min(heights)
    height_max = max(heights)
    height_span = height_max - height_min

    selected: list[MP3DRegionViewCandidate] = []
    selected_records = []
    used_floors: set[int] = set()
    for selection_order in range(1, views_per_region + 1):
        ranked = []
        for candidate in candidates:
            if candidate.floor_sample_index in used_floors:
                continue
            quality = _candidate_quality(
                candidate,
                boundary_scale_m=boundary_scale,
                height_min_m=height_min,
                height_max_m=height_max,
            )
            if not selected:
                novelty = {
                    "nearest_previously_selected_placement_id": None,
                    "horizontal_floor_position_distance_m": None,
                    "horizontal_floor_position_novelty": 1.0,
                    "circular_yaw_difference_deg": None,
                    "circular_yaw_novelty": 1.0,
                    "height_difference_m": None,
                    "height_novelty": 1.0,
                    "continuous_novelty": 1.0,
                }
            else:
                novelty = min(
                    (
                        _pair_novelty(
                            candidate,
                            retained,
                            position_scale_m=position_scale,
                            height_span_m=height_span,
                        )
                        for retained in selected
                    ),
                    key=lambda item: (
                        item["continuous_novelty"],
                        str(item["nearest_previously_selected_placement_id"]).encode(
                            "utf-8"
                        ),
                    ),
                )
            score = float(
                0.40 * quality["continuous_quality_preference"]
                + 0.60 * novelty["continuous_novelty"]
            )
            sort_key = (
                -score,
                -quality["continuous_quality_preference"],
                -novelty["continuous_novelty"],
                candidate.placement_id.encode("utf-8"),
            )
            ranked.append((sort_key, candidate, quality, novelty, score))
        _key, winner, quality, novelty, score = min(ranked, key=lambda item: item[0])
        selected.append(winner)
        used_floors.add(winner.floor_sample_index)
        selected_records.append(
            {
                "selection_order": selection_order,
                "placement_id": winner.placement_id,
                "region_instance_id": region_id,
                "region_index": region["region_index"],
                "region_category_code": region["category_code"],
                "region_category_name": region.get("category_name"),
                "camera": {
                    "floor_sample_index": winner.floor_sample_index,
                    "floor_position_m": list(winner.floor_position_m),
                    "position_m": list(winner.position_m),
                    "yaw_deg": winner.yaw_deg,
                    "height_id": winner.height_id,
                },
                "near_boundary_descriptor": winner.membership[
                    "near_boundary_descriptor"
                ],
                "ranking": {
                    **quality,
                    **novelty,
                    "selection_score": score,
                },
                "selection_reason": (
                    "highest remaining continuous boundary/height quality plus "
                    "floor-position/yaw/height novelty score while satisfying the "
                    "region quota's distinct-floor-position requirement; no source, "
                    "visibility, category or numeric-threshold gate"
                ),
                "region_category_used_for_generation_or_ranking": False,
                "source_route_or_visibility_used_for_ranking": False,
                "input_placement_legality_preserved": True,
                "native_capture_status": "pending",
            }
        )
    return {
        "region_instance_id": region_id,
        "region_index": region["region_index"],
        "category_code": region["category_code"],
        "category_name": region.get("category_name"),
        "category_mapping_status": region.get("category_mapping_status"),
        "eligible_unique_floor_position_count": len(by_floor),
        "eligible_unique_placement_count": len(candidates),
        "selected_viewpoint_count": len(selected_records),
        "selected_unique_floor_position_count": len(used_floors),
        "selected_unique_yaw_count": len(
            {item["camera"]["yaw_deg"] for item in selected_records}
        ),
        "selected_height_id_distribution": dict(
            sorted(
                Counter(
                    item["camera"]["height_id"] for item in selected_records
                ).items()
            )
        ),
        "continuous_scales": {
            "horizontal_floor_position_diagonal_m": scene_diagonal,
            "horizontal_floor_position_tanh_scale_m": position_scale,
            "boundary_distance_rational_scale_m": boundary_scale,
            "camera_height_span_m": height_span,
        },
        "selected_candidates": selected_records,
    }


def analyze_mp3d_region_viewpoints(
    *,
    sidecar: Mapping[str, Any],
    catalog: Mapping[str, Any],
    base_request: Mapping[str, Any],
    views_per_region: int = 4,
) -> MP3DRegionViewpointAnalysis:
    """Build deterministic per-region banks without changing input legality."""

    if (
        isinstance(views_per_region, bool)
        or not isinstance(views_per_region, int)
        or views_per_region <= 0
    ):
        raise MP3DRegionViewpointError("views_per_region must be a positive integer")
    room_id, request_id, expected_combinations = _validate_top_level(
        sidecar, catalog, base_request
    )
    regions = _region_records(sidecar)
    floors, floor_counts = _floor_records(sidecar, regions)
    catalog_records = _catalog_records(
        catalog,
        request_id=request_id,
        expected_combinations=expected_combinations,
    )
    eligible, placement_counts = _joined_candidates(sidecar, floors, catalog_records)
    if set(eligible) != set(regions):
        missing = sorted(set(regions) - set(eligible))
        if missing:
            raise MP3DRegionViewpointError(
                "regions lack uniquely-bound camera candidates: " + ", ".join(missing)
            )
    banks = tuple(
        _select_region(
            region,
            tuple(
                sorted(
                    eligible[region_id],
                    key=lambda item: item.placement_id.encode("utf-8"),
                )
            ),
            views_per_region=views_per_region,
        )
        for region_id, region in sorted(
            regions.items(), key=lambda item: item[0].encode("utf-8")
        )
    )
    return MP3DRegionViewpointAnalysis(
        room_id=room_id,
        views_per_region=views_per_region,
        input_floor_sample_count=len(floors),
        input_placement_count=len(catalog_records),
        floor_membership_status_counts=floor_counts,
        placement_membership_status_counts=placement_counts,
        region_banks=banks,
    )


__all__ = [
    "MP3DRegionViewpointAnalysis",
    "MP3DRegionViewpointError",
    "analyze_mp3d_region_viewpoints",
]

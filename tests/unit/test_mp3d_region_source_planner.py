from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from avengine.m5.mp3d_regions import MP3DFloorPolygon, MP3DHouseFloorPlan, MP3DRegion
from avengine.m5.mp3d_region_source_planner import (
    FRAME_COUNT,
    MOTION_CASES,
    MP3DRegionSourcePlannerError,
    build_region_source_route_bank,
)


@dataclass
class _Query:
    requested_start: np.ndarray | None = None
    requested_end: np.ndarray | None = None
    points: list[np.ndarray] | None = None
    geodesic_distance: float = 0.0


class _PathFinder:
    def get_bounds(self) -> np.ndarray:
        return np.asarray([[0.0, -1.0, 0.0], [20.0, 1.0, 8.0]])

    def is_navigable(self, point: np.ndarray, maximum_y_delta: float) -> bool:
        value = np.asarray(point, dtype=np.float64)
        return (
            value.shape == (3,)
            and abs(float(value[1])) <= maximum_y_delta
            and 0.0 <= value[0] <= 20.0
            and 0.0 <= value[2] <= 8.0
        )

    def snap_point(self, point: np.ndarray) -> np.ndarray:
        value = np.asarray(point, dtype=np.float64).copy()
        value[1] = 0.0
        return value

    def get_island(self, point: np.ndarray) -> int:
        del point
        return 1

    def distance_to_closest_obstacle(
        self, point: np.ndarray, maximum_search_radius: float
    ) -> float:
        del point
        return min(1.0, float(maximum_search_radius))

    def find_path(self, query: _Query) -> bool:
        start = np.asarray(query.requested_start, dtype=np.float64)
        end = np.asarray(query.requested_end, dtype=np.float64)
        midpoint = (start + end) * 0.5
        query.points = [start, midpoint, end]
        query.geodesic_distance = float(np.linalg.norm(end[[0, 2]] - start[[0, 2]]))
        return query.geodesic_distance > 1.0e-7


class _FailingPathFinder(_PathFinder):
    def find_path(self, query: _Query) -> bool:
        del query
        return False


def _region(index: int) -> MP3DRegion:
    row, column = divmod(index, 5)
    x = column * 3.0
    z = row * 3.0
    polygon = MP3DFloorPolygon(
        surface_index=index,
        region_index=index,
        vertices_habitat_xyz_m=(
            (x, 0.0, z),
            (x + 2.0, 0.0, z),
            (x + 2.0, 0.0, z + 2.0),
            (x, 0.0, z + 2.0),
        ),
    )
    return MP3DRegion(
        region_index=index,
        region_instance_id=f"17DRP5sb8fy:region:{index:03d}",
        level_index=0,
        category_code="b",
        category_name="bedroom",
        habitat_bbox_min_xyz_m=(x, 0.0, z),
        habitat_bbox_max_xyz_m=(x + 2.0, 0.0, z + 2.0),
        floor_polygons=(polygon,),
    )


def _inputs() -> tuple[dict, dict, MP3DHouseFloorPlan]:
    region_banks = []
    memberships = []
    placement_number = 0
    for index in range(10):
        region_id = f"17DRP5sb8fy:region:{index:03d}"
        selected = []
        row, column = divmod(index, 5)
        base_x = column * 3.0
        base_z = row * 3.0
        for order in range(1, 5):
            placement_number += 1
            placement_id = f"placement_{placement_number:03d}"
            floor_position = [base_x + 0.25 * order, 0.0, base_z + 0.25 * order]
            selected.append(
                {
                    "selection_order": order,
                    "placement_id": placement_id,
                    "region_instance_id": region_id,
                    "camera": {
                        "floor_sample_index": placement_number,
                        "floor_position_m": floor_position,
                        "position_m": [floor_position[0], 1.5, floor_position[2]],
                        "yaw_deg": float(order * 30),
                        "height_id": "base",
                    },
                    "m1_capture_request_path": (
                        f"requests/{region_id}/"
                        f"mp3d_camera_placement_{placement_number:03d}.json"
                    ),
                }
            )
            memberships.append(
                {
                    "placement_id": placement_id,
                    "membership_status": "unique",
                    "primary_region_instance_id": region_id,
                }
            )
        region_banks.append(
            {
                "region_instance_id": region_id,
                "region_index": index,
                "selected_candidates": selected,
            }
        )
    bank = {"region_banks": region_banks}
    sidecar = {"placement_memberships": memberships}
    plan = MP3DHouseFloorPlan(
        house_id="17DRP5sb8fy",
        house_name="test",
        house_label="test",
        declared_counts={},
        parsed_panorama_count=0,
        parsed_portal_count=0,
        regions=tuple(_region(index) for index in range(10)),
    )
    return bank, sidecar, plan


def _build(pathfinder: object | None = None, **kwargs: object) -> dict:
    bank, sidecar, plan = _inputs()
    finder = _PathFinder() if pathfinder is None else pathfinder
    return build_region_source_route_bank(
        bank=bank,
        sidecar=sidecar,
        house_plan=plan,
        pathfinder=finder,
        shortest_path_factory=_Query,
        sample_spacing_m=0.5,
        seed=1234,
        **kwargs,
    )


def test_builds_ten_regions_forty_bindings_one_hundred_sixty_cases() -> None:
    record = _build()
    assert record["camera_binding_count"] == 40
    assert record["route_family_count"] == 40
    assert record["case_count"] == 160
    assert record["frame_count"] == FRAME_COUNT
    assert record["motion_cases"] == list(MOTION_CASES)
    assert record["candidate_filtering"] == "none"
    assert record["camera_route_generation_decoupled"] is True
    assert record["exchange_case_count"] >= 1
    assert len(record["regions"]) == 10
    family_ids = [family["route_family_id"] for family in record["families"]]
    assert len(family_ids) == len(set(family_ids)) == 40
    assert all(item["camera_binding_count"] == 4 for item in record["regions"])
    assert all(item["route_family_count"] == 4 for item in record["regions"])
    assert all(len(family["cases"]) == 4 for family in record["families"])
    assert all(
        case["frame_count"] == FRAME_COUNT
        for family in record["families"]
        for case in family["cases"].values()
    )


def test_camera_pose_changes_do_not_change_route_generation() -> None:
    bank, sidecar, plan = _inputs()
    first = build_region_source_route_bank(
        bank=bank,
        sidecar=sidecar,
        house_plan=plan,
        pathfinder=_PathFinder(),
        shortest_path_factory=_Query,
        sample_spacing_m=0.5,
        seed=99,
    )
    changed = {
        **bank,
        "region_banks": [
            {
                **region,
                "selected_candidates": [
                    {
                        **candidate,
                        "camera": {
                            **candidate["camera"],
                            "position_m": [99.0, 1.5, 99.0],
                            "yaw_deg": 271.0,
                        },
                    }
                    for candidate in region["selected_candidates"]
                ],
            }
            for region in bank["region_banks"]
        ],
    }
    second = build_region_source_route_bank(
        bank=changed,
        sidecar=sidecar,
        house_plan=plan,
        pathfinder=_PathFinder(),
        shortest_path_factory=_Query,
        sample_spacing_m=0.5,
        seed=99,
    )
    for left, right in zip(first["families"], second["families"], strict=True):
        for motion_case in MOTION_CASES:
            assert (
                left["cases"][motion_case]["source1_positions_m"]
                == right["cases"][motion_case]["source1_positions_m"]
            )
            assert (
                left["cases"][motion_case]["source2_positions_m"]
                == right["cases"][motion_case]["source2_positions_m"]
            )


def test_nonexchange_families_do_not_reuse_route_geometry() -> None:
    record = _build()
    for region_index in range(10):
        families = [
            family
            for family in record["families"]
            if family["region_index"] == region_index
        ]
        assert len(families) == 4

        def canonical_signature(
            case: dict, key: str
        ) -> tuple[tuple[float, float], ...]:
            forward = tuple(
                (round(float(point[0]), 6), round(float(point[2]), 6))
                for point in case[key]
            )
            return min(forward, tuple(reversed(forward)))

        exchange_family = next(
            family
            for family in families
            if family["route_family_id"].endswith("_route_family_01")
        )
        exchange_case = exchange_family["cases"]["both_moving"]
        exchange_source1 = canonical_signature(exchange_case, "source1_waypoints_m")
        exchange_source2 = canonical_signature(exchange_case, "source2_waypoints_m")
        assert exchange_source1 == exchange_source2

        used_paths = {exchange_source1}
        for family in families:
            if family is exchange_family:
                continue
            case = family["cases"]["both_moving"]
            signatures = {
                canonical_signature(case, "source1_waypoints_m"),
                canonical_signature(case, "source2_waypoints_m"),
            }
            assert len(signatures) == 2
            assert used_paths.isdisjoint(signatures)
            used_paths.update(signatures)
        assert len(used_paths) == 7


def test_exchange_and_sub_sixty_distance_are_descriptors_not_filters() -> None:
    record = _build()
    exchange_cases = [
        case
        for family in record["families"]
        for case in family["cases"].values()
        if case["route_descriptors"]["endpoint_exchange"]["exact_endpoint_exchange"]
    ]
    assert exchange_cases
    exchange = exchange_cases[0]
    assert exchange["route_descriptors"]["identity_continuity"][
        "endpoint_exchange_does_not_swap_source_or_actor_ids"
    ]
    assert exchange["route_descriptors"]["proximity_reference"]["role"].endswith(
        "not_legality_gate"
    )
    assert exchange["candidate_filtering"] == "none"
    assert exchange["legality_effect"] == "none"


def test_native_pathfinder_failure_does_not_fallback_to_straight_route() -> None:
    with pytest.raises(
        MP3DRegionSourcePlannerError,
        match="cannot produce four PathFinder families",
    ):
        _build(pathfinder=_FailingPathFinder())


def test_visibility_and_collision_remain_pending() -> None:
    record = _build()
    case = record["families"][0]["cases"]["both_moving"]
    assert case["native_projection_visibility"]["status"].startswith("pending")
    assert case["collision_status"].startswith("pending")
    assert case["audio_rir_status"] == "pending"

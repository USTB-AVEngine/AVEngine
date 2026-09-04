from __future__ import annotations

import numpy as np
import pytest

from avengine.rooms.mp3d_regions import (
    MP3DFloorPolygon,
    MP3DHouseFloorPlan,
    MP3DRegion,
)
from avengine.routes.mp3d_region_planner import (
    MP3DRegionRouteError,
    build_region_route_plan,
)
from avengine.routes.mp3d_region_views import select_region_cameras


class _Query:
    def __init__(self) -> None:
        self.requested_start = None
        self.requested_end = None
        self.points = ()
        self.geodesic_distance = 0.0


class _PathFinder:
    is_loaded = True

    def is_navigable(self, point, maximum_y_delta=0.3) -> bool:
        del maximum_y_delta
        point = np.asarray(point, dtype=float)
        return point.shape == (3,) and np.all(np.isfinite(point)) and np.all(
            np.abs(point[[0, 2]]) <= 2.0
        )

    def snap_point(self, point):
        point = np.asarray(point, dtype=float).copy()
        point[0] = np.clip(point[0], -2.0, 2.0)
        point[2] = np.clip(point[2], -2.0, 2.0)
        return point

    def get_island(self, point) -> int:
        del point
        return 7

    def find_path(self, query: _Query) -> bool:
        start = np.asarray(query.requested_start, dtype=float)
        end = np.asarray(query.requested_end, dtype=float)
        query.points = (start, end)
        query.geodesic_distance = float(np.linalg.norm(end - start))
        return query.geodesic_distance > 1.0e-7


def _plan() -> MP3DHouseFloorPlan:
    polygon = MP3DFloorPolygon(
        surface_index=0,
        region_index=0,
        vertices_habitat_xyz_m=(
            (-2.0, 0.0, -2.0),
            (2.0, 0.0, -2.0),
            (2.0, 0.0, 2.0),
            (-2.0, 0.0, 2.0),
        ),
    )
    region = MP3DRegion(
        region_index=0,
        region_instance_id="fixture:region:000",
        level_index=0,
        category_code="l",
        category_name="living room",
        habitat_bbox_min_xyz_m=(-2.0, 0.0, -2.0),
        habitat_bbox_max_xyz_m=(2.0, 0.1, 2.0),
        floor_polygons=(polygon,),
    )
    return MP3DHouseFloorPlan(
        house_id="fixture",
        house_name="-",
        house_label="-",
        declared_counts={},
        parsed_portal_count=0,
        parsed_panorama_count=0,
        regions=(region,),
    )


def _camera_plan() -> dict:
    return select_region_cameras(
        _plan(),
        {
            "placement_memberships": [
                {
                    "placement_id": "camera0",
                    "primary_region_instance_id": "fixture:region:000",
                    "membership_status": "unique",
                    "floor_sample_index": 0,
                    "floor_position_m": [-1.5, 0.0, -1.5],
                    "position_m": [-1.5, 1.5, -1.5],
                    "yaw_deg": 0.0,
                    "height_id": "base",
                    "distance_to_nearest_polygon_boundary_m": 0.5,
                }
            ]
        },
        region_indices=[0],
        cameras_per_region=1,
    )


def test_route_plan_uses_realistic_small_request_without_fixed_counts() -> None:
    result = build_region_route_plan(
        _plan(),
        _camera_plan(),
        _PathFinder(),
        _Query,
        region_indices=[0],
        route_families_per_region=2,
        motion_cases=("static_static", "both_moving"),
        frame_count=5,
        frame_rate_hz=7,
        sample_spacing_m=1.0,
        maximum_candidate_points=16,
        maximum_route_attempts=40,
        seed=9,
    )

    assert result["region_count"] == 1
    assert result["route_family_count"] == 2
    assert result["case_count"] == 4
    assert result["parameters"]["frame_count"] == 5
    assert result["parameters"]["frame_rate_hz"] == 7
    families = result["regions"][0]["route_families"]
    assert len(families) == 2
    for family in families:
        static = family["cases"]["static_static"]
        moving = family["cases"]["both_moving"]
        assert static["frame_count"] == 5
        assert static["frame_rate_hz"] == 7
        assert static["source1_positions_m"] == [static["source1_positions_m"][0]] * 5
        assert len(moving["source1_positions_m"]) == 5
        assert moving["source1_positions_m"] != static["source1_positions_m"]
        assert family["camera_inputs_used_for_route_generation"] is False


def test_route_plan_rejects_small_attempt_budget() -> None:
    with pytest.raises(MP3DRegionRouteError, match="bounded attempts"):
        build_region_route_plan(
            _plan(),
            _camera_plan(),
            _PathFinder(),
            _Query,
            region_indices=[0],
            route_families_per_region=2,
            motion_cases=("both_moving",),
            frame_count=3,
            maximum_candidate_points=9,
            maximum_route_attempts=1,
            seed=2,
        )


def test_unknown_motion_case_is_rejected() -> None:
    with pytest.raises(MP3DRegionRouteError, match="unsupported motion_case"):
        build_region_route_plan(
            _plan(),
            _camera_plan(),
            _PathFinder(),
            _Query,
            region_indices=[0],
            motion_cases=("made_up",),
        )

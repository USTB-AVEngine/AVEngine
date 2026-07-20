from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from avengine.m6x.feasibility_topdown import render_feasibility_topdown
from avengine.m6x.geometry import RuntimeObstacleMap
from avengine.m6x.raster_pathfinder import (
    RasterShortestPath,
    build_polygon_raster_obstacle_map,
)
from avengine.m6x.room_feasibility import (
    MOTION_CASES,
    RoomFeasibilityCompiler,
    TrajectoryBankBuilder,
    build_rir_job_plan,
    evaluate_trajectory_coverage,
)


class _PathFinder:
    is_loaded = True

    def __init__(self) -> None:
        self.binary = np.ones((60, 60), dtype=np.uint8)
        self.find_path_calls = 0

    def get_topdown_view(self, meters_per_pixel, floor_height):
        assert meters_per_pixel == 0.1
        assert floor_height == 0.2
        return self.binary.copy()

    def get_bounds(self):
        return np.asarray(((0.0, 0.0, 0.0), (6.0, 3.0, 6.0)))

    def snap_point(self, point):
        value = np.asarray(point, dtype=np.float64).copy()
        value[0] = np.clip(value[0], 0.0, 6.0)
        value[1] = 0.2
        value[2] = np.clip(value[2], 0.0, 6.0)
        return value

    def is_navigable(self, point, maximum_y_delta):
        x, _y, z = np.asarray(point, dtype=np.float64)
        return 0.0 <= x <= 6.0 and 0.0 <= z <= 6.0

    def distance_to_closest_obstacle(self, point, maximum_search_radius):
        x, _y, z = np.asarray(point, dtype=np.float64)
        return float(min(x, 6.0 - x, z, 6.0 - z))

    def find_path(self, query):
        self.find_path_calls += 1
        start = np.asarray(query.requested_start, dtype=np.float64)
        end = np.asarray(query.requested_end, dtype=np.float64)
        query.points = [start, end]
        query.geodesic_distance = float(np.linalg.norm(end - start))
        return True


def _obstacle_map(pathfinder: _PathFinder, *, with_blocker: bool) -> RuntimeObstacleMap:
    obstacles = ()
    if with_blocker:
        obstacles = (
            {
                "object_id": 1,
                "handle": "table",
                "obstacle_role": "ground_blocker",
                "blocks_source_center": True,
                "footprint_xz_m": [
                    [2.5, 2.5],
                    [3.5, 2.5],
                    [3.5, 3.5],
                    [2.5, 3.5],
                ],
                "world_obb": {
                    "center_m": [3.0, 0.7, 3.0],
                    "axes_xyz": np.eye(3).tolist(),
                    "half_extents_m": [0.5, 0.5, 0.5],
                },
            },
        )
    return RuntimeObstacleMap(
        binary_navmesh=pathfinder.binary.copy(),
        bounds_m=((0.0, 0.0, 0.0), (6.0, 3.0, 6.0)),
        floor_height_m=0.2,
        meters_per_pixel=0.1,
        rigid_obstacles=obstacles,
        _pathfinder=pathfinder,
    )


def test_compiles_complete_region_components_samples_and_height_specific_obb() -> None:
    pathfinder = _PathFinder()
    obstacle_map = _obstacle_map(pathfinder, with_blocker=True)
    compiler = RoomFeasibilityCompiler(obstacle_map)
    low = compiler.compile(
        source_center_height_m=0.7,
        minimum_navmesh_clearance_m=0.0,
        sample_spacing_m=0.5,
    )
    high = compiler.compile(
        source_center_height_m=2.0,
        minimum_navmesh_clearance_m=0.0,
        sample_spacing_m=0.5,
    )
    assert low.feasible_mask.shape == (60, 60)
    assert np.count_nonzero(low.feasible_mask) < np.count_nonzero(high.feasible_mask)
    assert len(low.components) == 1
    assert len(low.sample_pixels_rc) > 100
    assert low.summary()["claim_boundary"].startswith("source center only")
    assert low.summary()["approximate_feasible_area_m2"] > 30.0


def test_builds_all_four_motion_cases_deterministically_and_plans_rir_cache() -> None:
    pathfinder = _PathFinder()
    obstacle_map = _obstacle_map(pathfinder, with_blocker=False)
    compiler = RoomFeasibilityCompiler(obstacle_map)
    regions = {
        "human0": compiler.compile(
            source_center_height_m=1.8,
            minimum_navmesh_clearance_m=0.0,
            sample_spacing_m=0.5,
        ),
        "dog0": compiler.compile(
            source_center_height_m=0.7,
            minimum_navmesh_clearance_m=0.0,
            sample_spacing_m=0.5,
        ),
    }

    def materialize(roots):
        return {
            "human_source": roots["human0"] + np.asarray((0.0, 1.6, 0.0)),
            "dog_source": roots["dog0"] + np.asarray((0.0, 0.4, 0.0)),
        }

    def build():
        return TrajectoryBankBuilder(
            pathfinder=pathfinder,
            obstacle_map=obstacle_map,
            region_by_actor=regions,
            shortest_path_factory=SimpleNamespace,
            source_path_materializer=materialize,
        ).build(
            episodes_per_motion_case=1,
            frame_count=15,
            frame_rate_hz=15,
            seed=17,
            minimum_route_distance_m=1.0,
            maximum_route_distance_m=5.0,
            minimum_pair_separation_m=0.1,
        )

    first = build()
    second = build()
    assert [episode.motion_case for episode in first.episodes] == list(MOTION_CASES)
    assert first.record() == second.record()
    for episode in first.episodes:
        human_distance = episode.statistics["human0"]["geodesic_distance_m"]
        dog_distance = episode.statistics["dog0"]["geodesic_distance_m"]
        assert (human_distance > 0.0) == (
            "human_moving" in episode.motion_case
            or episode.motion_case == "both_moving"
        )
        assert (dog_distance > 0.0) == (
            "dog_moving" in episode.motion_case or episode.motion_case == "both_moving"
        )

    plan = build_rir_job_plan(
        first,
        listener_position_m=(1.0, 1.5, 1.0),
        listener_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        stride_frames=3,
    )
    assert plan["status"] == "planned_not_run"
    assert plan["requested_pair_state_count"] == 4 * 2 * 5
    assert plan["unique_rir_job_count"] < plan["requested_pair_state_count"]

    coverage = evaluate_trajectory_coverage(
        regions,
        first,
        minimum_half_meter_fraction=0.0,
        maximum_gap_m=10.0,
    )
    assert coverage.record["status"] == "pass"
    assert coverage.record["component_count"] == 1
    assert coverage.record["trajectory_seed_pixel_count"] > 0
    assert np.all(
        np.isfinite(
            coverage.distance_to_trajectory_m[
                regions["human0"].feasible_mask & regions["dog0"].feasible_mask
            ]
        )
    )

    overview = render_feasibility_topdown(
        regions,
        first,
        source_to_actor={"human_source": "human0", "dog_source": "dog0"},
        trajectory_coverage=coverage,
        listener_position_m=(1.0, 1.5, 1.0),
        listener_yaw_deg=0.0,
        camera_hfov_degrees=90.0,
        size_wh=(1000, 800),
    )
    assert overview.shape == (800, 1000, 3)
    assert overview.dtype == np.uint8
    assert overview.flags.c_contiguous


def test_polygon_raster_pathfinder_routes_around_declared_footprint() -> None:
    obstacle = {
        "object_id": "wall0",
        "handle": "wall0",
        "obstacle_role": "ground_blocker",
        "blocks_source_center": True,
        "footprint_xz_m": [
            [2.5, 0.0],
            [3.5, 0.0],
            [3.5, 4.5],
            [2.5, 4.5],
        ],
        "world_obb": {
            "center_m": [3.0, 1.0, 2.25],
            "axes_xyz": np.eye(3).tolist(),
            "half_extents_m": [0.5, 1.0, 2.25],
        },
    }
    pathfinder, obstacle_map = build_polygon_raster_obstacle_map(
        polygon_xz_m=[[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]],
        rigid_obstacles=[obstacle],
        floor_height_m=0.0,
        meters_per_pixel=0.1,
        padding_m=0.1,
        authority="synthetic_polygon_test",
    )
    query = RasterShortestPath(
        requested_start=np.asarray([1.0, 0.0, 1.0]),
        requested_end=np.asarray([5.0, 0.0, 1.0]),
    )
    assert pathfinder.find_path(query)
    assert query.points is not None
    assert query.geodesic_distance > 7.0
    assert all(
        not (2.5 <= point[0] <= 3.5 and 0.0 <= point[2] <= 4.5)
        for point in query.points
    )
    assert obstacle_map.summary()["authority"] == "synthetic_polygon_test"
    assert obstacle_map.summary()["rigid_obstacles_baked_into_navmesh"] is True
    region = RoomFeasibilityCompiler(obstacle_map).compile(
        source_center_height_m=1.0,
        minimum_navmesh_clearance_m=0.0,
        sample_spacing_m=0.5,
    )
    assert len(region.components) == 1


def test_reversed_route_library_avoids_repeating_shortest_path_queries() -> None:
    pathfinder = _PathFinder()
    obstacle_map = _obstacle_map(pathfinder, with_blocker=False)
    compiler = RoomFeasibilityCompiler(obstacle_map)
    regions = {
        actor_id: compiler.compile(
            source_center_height_m=height,
            minimum_navmesh_clearance_m=0.0,
            sample_spacing_m=0.5,
        )
        for actor_id, height in (("human0", 1.6), ("dog0", 0.45))
    }
    bank = TrajectoryBankBuilder(
        pathfinder=pathfinder,
        obstacle_map=obstacle_map,
        region_by_actor=regions,
        shortest_path_factory=SimpleNamespace,
    ).build(
        episodes_per_motion_case=2,
        frame_count=15,
        frame_rate_hz=15,
        seed=29,
        minimum_route_distance_m=1.0,
        maximum_route_distance_m=5.0,
        minimum_pair_separation_m=0.0,
        reuse_reversed_moving_paths=True,
    )
    assert len(bank.episodes) == 8
    # Each actor needs two independent routes in its four-entry
    # forward/reverse library.  Later motion cases reuse those entries.
    assert pathfinder.find_path_calls == 4

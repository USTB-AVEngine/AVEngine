from __future__ import annotations

import numpy as np
import pytest

from avengine.camera_placement import (
    DEFAULT_YAW_DEGREES,
    CameraPlacementError,
    iter_feasible_camera_placements,
)
from avengine.m6x.geometry import RuntimeObstacleMap


def _obstacle_map() -> RuntimeObstacleMap:
    return RuntimeObstacleMap(
        binary_navmesh=np.ones((4, 4), dtype=np.uint8),
        bounds_m=((0.0, 0.0, 0.0), (4.0, 3.0, 4.0)),
        floor_height_m=0.0,
        meters_per_pixel=1.0,
        rigid_obstacles=(),
        authority="unit_test_room_geometry",
    )


def test_iterator_crosses_floor_feasible_centers_with_twelve_stable_yaws() -> None:
    floor_mask = np.zeros((4, 4), dtype=np.bool_)
    floor_mask[1, 1] = True
    floor_mask[2, 3] = True

    placements = tuple(
        iter_feasible_camera_placements(
            _obstacle_map(),
            camera_center_heights_m=(("high", 1.8), ("low", 0.8)),
            minimum_navmesh_clearance_m=0.0,
            minimum_rigid_clearance_m=0.0,
            sample_spacing_m=1.0,
            floor_feasible_mask=floor_mask,
        )
    )

    assert len(placements) == 2 * 2 * 12
    assert tuple(sorted({item.yaw_deg for item in placements})) == (DEFAULT_YAW_DEGREES)
    assert placements[0].height_id == "low"
    assert placements[0].position_m == (1.5, 0.8, 1.5)
    assert placements[0].raster_pixel_rc == (1, 1)
    assert placements[0].yaw_deg == 0.0
    assert placements[-1].height_id == "high"
    assert placements[-1].position_m == (3.5, 1.8, 2.5)
    assert placements[-1].yaw_deg == 330.0
    assert placements[0].to_record()["placement_status"] == (
        "camera_center_feasible_only"
    )


def test_iterator_is_order_independent_and_exactly_deduplicates_yaws() -> None:
    first = tuple(
        iter_feasible_camera_placements(
            _obstacle_map(),
            camera_center_heights_m=(("high", 1.8), ("low", 0.8)),
            minimum_navmesh_clearance_m=0.0,
            sample_spacing_m=2.0,
            yaw_degrees=(330.0, 0.0, 30.0),
        )
    )
    second = tuple(
        iter_feasible_camera_placements(
            _obstacle_map(),
            camera_center_heights_m=(("low", 0.8), ("high", 1.8)),
            minimum_navmesh_clearance_m=0.0,
            sample_spacing_m=2.0,
            yaw_degrees=(30.0, 330.0, 0.0),
        )
    )

    assert first == second
    with pytest.raises(CameraPlacementError, match="duplicates after wrapping"):
        tuple(
            iter_feasible_camera_placements(
                _obstacle_map(),
                camera_center_heights_m=(("base", 1.5),),
                yaw_degrees=(0.0, 360.0),
            )
        )


def test_iterator_rejects_invalid_floor_mask_and_does_not_accept_targets() -> None:
    invalid = np.ones((4, 4), dtype=np.bool_)
    obstacle_map = _obstacle_map()
    obstacle_map.binary_navmesh[0, 0] = 0

    with pytest.raises(CameraPlacementError, match="subset"):
        tuple(
            iter_feasible_camera_placements(
                obstacle_map,
                camera_center_heights_m=(("base", 1.5),),
                floor_feasible_mask=invalid,
            )
        )
    with pytest.raises(TypeError):
        tuple(
            iter_feasible_camera_placements(
                obstacle_map,
                camera_center_heights_m=(("base", 1.5),),
                semantic_targets={},  # type: ignore[call-arg]
            )
        )

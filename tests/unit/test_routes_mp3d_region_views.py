from __future__ import annotations

from avengine.rooms.mp3d_regions import (
    MP3DFloorPolygon,
    MP3DHouseFloorPlan,
    MP3DRegion,
)
from avengine.routes.mp3d_region_views import (
    MP3DRegionViewError,
    select_region_cameras,
)


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


def _sidecar() -> dict:
    return {
        "placement_memberships": [
            {
                "placement_id": "p0",
                "primary_region_instance_id": "fixture:region:000",
                "membership_status": "unique",
                "floor_sample_index": 0,
                "floor_position_m": [-1.5, 0.0, -1.0],
                "position_m": [-1.5, 1.5, -1.0],
                "yaw_deg": 0.0,
                "height_id": "base",
                "distance_to_nearest_polygon_boundary_m": 0.5,
            },
            {
                "placement_id": "p1",
                "primary_region_instance_id": "fixture:region:000",
                "membership_status": "unique",
                "floor_sample_index": 1,
                "floor_position_m": [1.5, 0.0, 1.0],
                "position_m": [1.5, 1.5, 1.0],
                "yaw_deg": 90.0,
                "height_id": "base",
                "distance_to_nearest_polygon_boundary_m": 0.5,
            },
            {
                "placement_id": "boundary",
                "primary_region_instance_id": "fixture:region:000",
                "membership_status": "boundary",
                "floor_sample_index": 2,
                "floor_position_m": [2.0, 0.0, 0.0],
                "position_m": [2.0, 1.5, 0.0],
                "yaw_deg": 180.0,
                "height_id": "base",
                "distance_to_nearest_polygon_boundary_m": 0.0,
            },
        ]
    }


def test_select_region_cameras_is_seeded_and_parameterized() -> None:
    plan = _plan()
    result = select_region_cameras(
        plan,
        _sidecar(),
        region_indices=[0],
        cameras_per_region=2,
        seed=11,
    )

    region = result["regions"][0]
    assert result["region_count"] == 1
    assert result["cameras_per_region"] == 2
    assert region["camera_count"] == 2
    assert {item["placement_id"] for item in region["cameras"]} <= {"p0", "p1"}
    assert len(
        {item["floor_sample_index"] for item in region["cameras"]}
    ) == 2
    assert result == select_region_cameras(
        plan,
        _sidecar(),
        region_indices=[0],
        cameras_per_region=2,
        seed=11,
    )


def test_select_region_cameras_can_choose_one_without_four_view_assumption() -> None:
    result = select_region_cameras(
        _plan(),
        _sidecar(),
        region_indices=[0],
        cameras_per_region=1,
        seed=3,
        require_unique_floor=False,
    )
    assert result["regions"][0]["camera_count"] == 1


def test_select_region_cameras_rejects_unknown_region() -> None:
    try:
        select_region_cameras(_plan(), _sidecar(), region_indices=[4])
    except MP3DRegionViewError as exc:
        assert "unknown region" in str(exc)
    else:
        raise AssertionError("unknown region should fail")

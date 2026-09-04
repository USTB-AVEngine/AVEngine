from __future__ import annotations

from pathlib import Path

import pytest

from avengine.rooms.mp3d_regions import (
    MP3DRegionError,
    bind_floor_points_to_regions,
    parse_mp3d_house,
)


def _minimal_house() -> str:
    lines = [
        "ASCII 1.1",
        # H counts: images panoramas vertices surfaces segments objects
        # categories regions portals levels, followed by unused house fields.
        "H - - 0 0 4 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "L 0 0 - 0 0 0 -1 -1 -1 1 1 1 0 0 0 0 0",
        "R 0 0 0 0 l 0 0 0 -1 -0.1 -1 1 0.1 1 0 0 0 0 0",
        "S 0 0 0 F 0 0 0 0 0 1 -1 -1 -1 1 1 1 0 0 0 0 0",
        "V 0 0 F -1 -1 0 0 0 1 0 0 0 0",
        "V 1 0 F 1 -1 0 0 0 1 0 0 0 0",
        "V 2 0 F 1 1 0 0 0 1 0 0 0 0",
        "V 3 0 F -1 1 0 0 0 1 0 0 0 0",
    ]
    return "\n".join(lines) + "\n"


def test_parse_house_builds_transformed_floor_region(tmp_path: Path) -> None:
    path = tmp_path / "tiny.house"
    path.write_text(_minimal_house(), encoding="utf-8")

    plan = parse_mp3d_house(path)

    assert plan.house_id == "tiny"
    assert len(plan.regions) == 1
    region = plan.regions[0]
    assert region.region_instance_id == "tiny:region:000"
    assert region.category_name == "living room"
    assert len(region.floor_polygons) == 1
    assert region.floor_polygons[0].floor_elevation_m == pytest.approx(0.0)
    assert region.contains((0.0, 0.0, 0.0))
    assert not region.contains((2.0, 0.0, 0.0))


def test_floor_membership_is_descriptive_and_marks_boundary(tmp_path: Path) -> None:
    path = tmp_path / "tiny.house"
    path.write_text(_minimal_house(), encoding="utf-8")
    plan = parse_mp3d_house(path)

    membership = bind_floor_points_to_regions(
        [(-0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        , plan,
        boundary_descriptor_threshold_m=0.2,
    )

    assert [item.membership_status for item in membership] == [
        "unique",
        "boundary",
        "unknown",
    ]
    assert membership[0].member_region_indices == (0,)
    assert membership[1].on_polygon_boundary is True
    assert membership[1].near_boundary is True
    assert membership[2].member_region_indices == ()


@pytest.mark.parametrize(
    "contents",
    ["", "BINARY 1.1\n", "ASCII 1.1\nH - - 0\n"],
)
def test_parse_house_rejects_truncated_or_wrong_header(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "bad.house"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(MP3DRegionError):
        parse_mp3d_house(path)

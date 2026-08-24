from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.contracts.json_io import load_json
from avengine.m3.compiler import _KNOWN_SOURCE_TRANSFORMS
from avengine.m5.mp3d_regions import (
    MP3D_Z_UP_Y_FRONT_TO_HABITAT_MATRIX_ROW_MAJOR,
    MP3DRegionError,
    bind_floor_points_to_regions,
    mp3d_z_up_y_front_to_habitat,
    parse_mp3d_house,
)
from tools.m5 import bind_mp3d_camera_regions as binder


ROOM_ID = "habitat_mp3d_unknown_scene"
YAWS = tuple(float(value) for value in range(0, 360, 30))
HEIGHTS = {"low": 1.2, "base": 1.5, "high": 1.8}


def _house_lines(*, category1: str = "q") -> list[str]:
    return [
        "ASCII 1.1",
        "H - - 0 0 8 2 0 0 0 2 0 1 0 0 0 0 0 -2 -2 0 3 3 3 0 0 0 0 0",
        "L 0 2 - 0 0 0 -2 -2 0 3 3 3 0 0 0 0 0",
        "R 0 0 0 0 b 0 0 0 -1 -1 0 1 1 2 2 0 0 0 0",
        f"R 1 0 0 0 {category1} 1.5 1.5 0 0.5 0.5 0 2.5 2.5 2 2 0 0 0 0",
        "S 0 0 0 F 0 0 0 0 0 1 -1 -1 0 1 1 0 0 0 0 0 0",
        "S 1 1 0 F 1.5 1.5 0 0 0 1 0.5 0.5 0 2.5 2.5 0 0 0 0 0 0",
        "V 0 0 F -1 -1 0 0 0 1 0 0 0",
        "V 1 0 F 1 -1 0 0 0 1 0 0 0",
        "V 2 0 F 1 1 0 0 0 1 0 0 0",
        "V 3 0 F -1 1 0 0 0 1 0 0 0",
        "V 4 1 F 0.5 0.5 0 0 0 1 0 0 0",
        "V 5 1 F 2.5 0.5 0 0 0 1 0 0 0",
        "V 6 1 F 2.5 2.5 0 0 0 1 0 0 0",
        "V 7 1 F 0.5 2.5 0 0 0 1 0 0 0",
    ]


def _write_house(path: Path, *, category1: str = "q") -> Path:
    path.write_text(
        "\n".join(_house_lines(category1=category1)) + "\n", encoding="utf-8"
    )
    return path


def _samples() -> list[list[float]]:
    # Canonical order is intentionally different from this input order.
    return [
        [3.0, 0.1, 3.0],  # unknown
        [1.0, 0.1, 0.0],  # exact boundary of region 0
        [0.75, 0.1, -0.75],  # overlap, retained ambiguous
        [0.0, 0.1, 0.0],  # unique region 0
    ]


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path, list[list[float]]]:
    floor_path = tmp_path / "camera_floor_samples.json"
    catalog_path = tmp_path / "camera_placement_catalog.json"
    raw_samples = _samples()
    canonical = sorted({tuple(point) for point in raw_samples})
    floor_path.write_text(
        json.dumps(
            {
                "authority": "live_habitat_pathfinder_declared_navmesh_export",
                "camera_floor_samples_m": raw_samples,
                "coordinate_space": "habitat_world_xyz_m",
                "episode_counted": False,
                "native_habitat_floor_export": True,
                "research_only": True,
                "room_id": ROOM_ID,
            }
        ),
        encoding="utf-8",
    )
    placements = []
    for floor_index, floor in enumerate(canonical):
        for height_id, height in HEIGHTS.items():
            for yaw in YAWS:
                placements.append(
                    {
                        "placement_id": f"placement_{len(placements):04d}",
                        "floor_sample_index": floor_index,
                        "floor_position_m": list(floor),
                        "height_id": height_id,
                        "position_m": [floor[0], floor[1] + height, floor[2]],
                        "yaw_deg": yaw,
                        "placement_status": (
                            "declared_floor_position_crossed_with_height_and_yaw"
                        ),
                    }
                )
    catalog_path.write_text(
        json.dumps(
            {
                "artifact_status": (
                    "complete_target_agnostic_camera_placement_catalog"
                ),
                "camera_height_variants_m": HEIGHTS,
                "candidate_filtering": "none",
                "canonical_floor_sample_count": len(canonical),
                "episode_counted": False,
                "floor_position_authority": (
                    "live_habitat_pathfinder_declared_navmesh_export"
                ),
                "input_floor_sample_count": len(raw_samples),
                "placement_count": len(placements),
                "placements": placements,
                "research_only": True,
                "room_id": ROOM_ID,
                "yaw_degrees": YAWS,
            }
        ),
        encoding="utf-8",
    )
    return floor_path, catalog_path, [list(point) for point in canonical]


def _args(tmp_path: Path):
    house = _write_house(tmp_path / "scene.house")
    floor, catalog, canonical = _write_artifacts(tmp_path)
    output = tmp_path / "region_sidecar"
    return (
        binder.parse_args(
            [
                "--house",
                str(house),
                "--camera-floor-samples",
                str(floor),
                "--camera-catalog",
                str(catalog),
                "--output",
                str(output),
            ]
        ),
        output,
        floor,
        catalog,
        canonical,
    )


def test_registered_coordinate_transform_is_x_z_negative_y() -> None:
    registered_matrix, _source = _KNOWN_SOURCE_TRANSFORMS[
        "mp3d_z_up_y_front_to_habitat"
    ]
    assert tuple(registered_matrix) == (MP3D_Z_UP_Y_FRONT_TO_HABITAT_MATRIX_ROW_MAJOR)
    assert mp3d_z_up_y_front_to_habitat([1.25, -2.5, 3.75]) == (
        1.25,
        3.75,
        2.5,
    )


def test_parser_preserves_unknown_region_category(tmp_path: Path) -> None:
    plan = parse_mp3d_house(_write_house(tmp_path / "scene.house"))

    assert len(plan.regions) == 2
    assert plan.regions[0].category_name == "bedroom"
    assert plan.regions[1].category_code == "q"
    assert plan.regions[1].category_name is None
    assert plan.regions[1].label_record()["category_mapping_status"] == (
        "unknown_label_code"
    )


def test_parser_distinguishes_panorama_and_portal_p_records(tmp_path: Path) -> None:
    lines = _house_lines()[:]
    lines[1] = "H - - 0 1 8 2 0 0 0 2 1 1 0 0 0 0 0 -2 -2 0 3 3 3 0 0 0 0 0"
    lines.extend(
        [
            "P panorama_name 0 0 0 0 0 1 0 0 0 0 0",
            "P 0 0 1 D -1 -1 0 1 1 2 0 0 0 0",
        ]
    )
    path = tmp_path / "scene.house"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plan = parse_mp3d_house(path)

    assert plan.declared_counts["panoramas"] == 1
    assert plan.declared_counts["portals"] == 1
    assert plan.parsed_panorama_count == 1
    assert plan.parsed_portal_count == 1


def test_membership_retains_unique_boundary_unknown_and_ambiguous(
    tmp_path: Path,
) -> None:
    plan = parse_mp3d_house(_write_house(tmp_path / "scene.house"))
    memberships = bind_floor_points_to_regions(
        [
            [0.0, 0.1, 0.0],
            [1.0, 0.1, 0.0],
            [3.0, 0.1, 3.0],
            [0.75, 0.1, -0.75],
        ],
        plan,
    )

    assert [item.membership_status for item in memberships] == [
        "unique",
        "boundary",
        "unknown",
        "ambiguous",
    ]
    assert memberships[0].member_region_indices == (0,)
    assert memberships[1].member_region_indices == (0,)
    assert memberships[2].member_region_indices == ()
    assert memberships[3].member_region_indices == (0, 1)


def test_near_boundary_is_descriptor_not_membership_gate(tmp_path: Path) -> None:
    plan = parse_mp3d_house(_write_house(tmp_path / "scene.house"))
    membership = bind_floor_points_to_regions([[0.97, 0.1, 0.0]], plan)[0]

    assert membership.membership_status == "unique"
    assert membership.near_boundary is True
    assert membership.nearest_polygon_boundary_distance_m == pytest.approx(0.03)


def test_region_bbox_y_prevents_cross_level_horizontal_binding(
    tmp_path: Path,
) -> None:
    plan = parse_mp3d_house(_write_house(tmp_path / "scene.house"))
    membership = bind_floor_points_to_regions([[0.0, 4.0, 0.0]], plan)[0]

    assert membership.membership_status == "unknown"
    assert membership.member_region_indices == ()


def test_bad_or_nonfinite_floor_polygon_fails_closed(tmp_path: Path) -> None:
    nonfinite = _house_lines()
    nonfinite[7] = "V 0 0 F nan -1 0 0 0 1 0 0 0"
    path = tmp_path / "nonfinite.house"
    path.write_text("\n".join(nonfinite) + "\n", encoding="utf-8")
    with pytest.raises(MP3DRegionError, match="finite number"):
        parse_mp3d_house(path)

    self_intersecting = _house_lines()
    self_intersecting[7:11] = [
        "V 0 0 F -1 -1 0 0 0 1 0 0 0",
        "V 1 0 F 1 1 0 0 0 1 0 0 0",
        "V 2 0 F 1 -1 0 0 0 1 0 0 0",
        "V 3 0 F -1 1 0 0 0 1 0 0 0",
    ]
    path = tmp_path / "self_intersecting.house"
    path.write_text("\n".join(self_intersecting) + "\n", encoding="utf-8")
    with pytest.raises(MP3DRegionError, match="zero-area|self-intersecting"):
        parse_mp3d_house(path)


def test_builder_publishes_additive_unfiltered_per_record_sidecar(
    tmp_path: Path,
) -> None:
    args, output, _floor, catalog, canonical = _args(tmp_path)
    catalog_before = catalog.read_bytes()

    assert binder.run(args) == output
    assert catalog.read_bytes() == catalog_before
    record = load_json(output / "camera_region_sidecar.json")

    assert record["candidate_filtering"] == "none"
    assert record["candidate_legality_effect"] == "none"
    assert record["boundary_descriptor_is_legality_gate"] is False
    assert record["floor_sample_count"] == 4
    assert record["placement_count"] == 4 * 36
    assert len(record["floor_sample_memberships"]) == 4
    assert len(record["placement_memberships"]) == 4 * 36
    assert all(
        item["height_id"] in HEIGHTS and item["yaw_deg"] in YAWS
        for item in record["placement_memberships"]
    )
    assert record["floor_membership_status_counts"] == {
        "ambiguous": 1,
        "boundary": 1,
        "unique": 1,
        "unknown": 1,
    }
    assert record["placement_membership_status_counts"] == {
        "ambiguous": 36,
        "boundary": 36,
        "unique": 36,
        "unknown": 36,
    }
    for floor_index, floor_record in enumerate(record["floor_sample_memberships"]):
        assert floor_record["floor_sample_index"] == floor_index
        assert floor_record["floor_position_m"] == canonical[floor_index]
        assert floor_record["placement_count"] == 36
        assert len(floor_record["placement_ids"]) == 36
    ambiguous = next(
        item
        for item in record["floor_sample_memberships"]
        if item["membership_status"] == "ambiguous"
    )
    assert ambiguous["member_region_indices"] == [0, 1]
    assert [item["category_code"] for item in ambiguous["member_regions"]] == [
        "b",
        "q",
    ]
    assert ambiguous["primary_region_index"] is None
    unknown = next(
        item
        for item in record["floor_sample_memberships"]
        if item["membership_status"] == "unknown"
    )
    assert unknown["member_region_indices"] == []
    assert unknown["primary_region_index"] is None
    assert (output / "README.md").is_file()

    with pytest.raises(
        binder.MP3DCameraRegionBindingError, match="refusing to replace output"
    ):
        binder.run(args)


@pytest.mark.parametrize("drift", ["index", "coordinate"])
def test_builder_rejects_floor_index_or_coordinate_drift(
    tmp_path: Path, drift: str
) -> None:
    args, output, _floor, catalog, _canonical = _args(tmp_path)
    value = json.loads(catalog.read_text(encoding="utf-8"))
    if drift == "index":
        value["placements"][0]["floor_sample_index"] = 1
    else:
        value["placements"][0]["floor_position_m"][0] += 1.0e-6
    catalog.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        binder.MP3DCameraRegionBindingError,
        match="floor_sample_index/floor_position_m drift",
    ):
        binder.run(args)
    assert not output.exists()


def test_builder_rejects_missing_one_of_36_placements(tmp_path: Path) -> None:
    args, output, _floor, catalog, _canonical = _args(tmp_path)
    value = json.loads(catalog.read_text(encoding="utf-8"))
    value["placements"].pop()
    value["placement_count"] -= 1
    catalog.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        binder.MP3DCameraRegionBindingError,
        match="exactly 36 placements per floor sample",
    ):
        binder.run(args)
    assert not output.exists()

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from avengine.contracts.json_io import load_json
from avengine.m1.contracts import validate_capture_request
from avengine.m5.mp3d_region_viewpoints import (
    MP3DRegionViewpointError,
    analyze_mp3d_region_viewpoints,
)
from tools.m5 import build_mp3d_region_viewpoint_banks as builder


REPOSITORY = Path(__file__).resolve().parents[2]
REQUEST = REPOSITORY / "examples/m1/requests/habitat_mp3d_example.json"
ROOM_ID = "habitat_mp3d_example_17DRP5sb8fy"
REQUEST_ID = "m1_habitat_mp3d_single_view_v1"
YAWS = tuple(float(value) for value in range(0, 360, 30))
HEIGHTS = {"low": 1.2, "base": 1.5, "high": 1.8}


def _base_request() -> dict:
    return load_json(REQUEST)


def _member(region_index: int, *, unknown_category: bool = False) -> dict:
    return {
        "category_code": "q" if unknown_category else "b",
        "category_mapping_status": (
            "unknown_label_code" if unknown_category else "known_label_code"
        ),
        "category_name": None if unknown_category else "bedroom",
        "distance_to_region_polygon_boundary_m": 0.2,
        "floor_elevation_delta_m": 0.0,
        "level_index": 0,
        "region_index": region_index,
        "region_instance_id": f"synthetic:region:{region_index:03d}",
    }


def _fixture(*, positions_per_region: int = 4) -> tuple[dict, dict]:
    regions = [
        {
            **_member(0),
            "floor_polygon_count": 1,
            "floor_sample_count": positions_per_region,
            "placement_count": positions_per_region * 36,
        },
        {
            **_member(1, unknown_category=True),
            "floor_polygon_count": 1,
            "floor_sample_count": positions_per_region,
            "placement_count": positions_per_region * 36,
        },
    ]
    floor_specs: list[tuple[list[float], str, list[int], float]] = []
    for region_index in range(2):
        for position_index in range(positions_per_region):
            floor_specs.append(
                (
                    [float(position_index), 0.1, float(region_index * 10)],
                    "unique",
                    [region_index],
                    0.01 + 0.1 * position_index,
                )
            )
    floor_specs.extend(
        [
            ([20.0, 0.1, 0.0], "boundary", [0], 0.0),
            ([21.0, 0.1, 0.0], "unknown", [], 0.4),
            ([22.0, 0.1, 0.0], "ambiguous", [0, 1], 0.0),
        ]
    )

    floor_records = []
    placement_memberships = []
    catalog_placements = []
    placement_ordinal = 0
    for floor_index, (floor, status, member_indices, distance) in enumerate(
        floor_specs
    ):
        member_records = [
            _member(index, unknown_category=index == 1) for index in member_indices
        ]
        member_ids = [item["region_instance_id"] for item in member_records]
        primary = member_records[0] if len(member_records) == 1 else None
        placement_ids = []
        for height_id, height in HEIGHTS.items():
            for yaw in YAWS:
                placement_id = f"placement_{placement_ordinal:05d}"
                placement_ordinal += 1
                placement_ids.append(placement_id)
                position = [floor[0], floor[1] + height, floor[2]]
                catalog_placements.append(
                    {
                        "base_m1_request_id": REQUEST_ID,
                        "floor_position_m": list(floor),
                        "floor_sample_index": floor_index,
                        "height_id": height_id,
                        "placement_id": placement_id,
                        "placement_status": (
                            "declared_floor_position_crossed_with_height_and_yaw"
                        ),
                        "position_m": list(position),
                        "room_id": ROOM_ID,
                        "yaw_deg": yaw,
                    }
                )
                placement_memberships.append(
                    {
                        "candidate_legality_effect": "none",
                        "distance_to_nearest_polygon_boundary_m": distance,
                        "floor_position_m": list(floor),
                        "floor_sample_index": floor_index,
                        "height_id": height_id,
                        "member_region_indices": member_indices,
                        "member_region_instance_ids": member_ids,
                        "member_regions": member_records,
                        "membership_status": status,
                        "near_boundary_descriptor": distance <= 0.05,
                        "on_polygon_boundary": status in {"boundary", "ambiguous"},
                        "placement_id": placement_id,
                        "position_m": list(position),
                        "primary_region_category_code": (
                            primary["category_code"] if primary else None
                        ),
                        "primary_region_category_name": (
                            primary["category_name"] if primary else None
                        ),
                        "primary_region_index": (
                            primary["region_index"] if primary else None
                        ),
                        "primary_region_instance_id": (
                            primary["region_instance_id"] if primary else None
                        ),
                        "yaw_deg": yaw,
                    }
                )
        floor_records.append(
            {
                "distance_to_nearest_polygon_boundary_m": distance,
                "floor_position_m": list(floor),
                "floor_sample_index": floor_index,
                "member_region_indices": member_indices,
                "member_region_instance_ids": member_ids,
                "member_regions": member_records,
                "membership_status": status,
                "near_boundary_descriptor": distance <= 0.05,
                "on_polygon_boundary": status in {"boundary", "ambiguous"},
                "placement_count": 36,
                "placement_ids": placement_ids,
                "primary_region_category_code": (
                    primary["category_code"] if primary else None
                ),
                "primary_region_category_name": (
                    primary["category_name"] if primary else None
                ),
                "primary_region_index": primary["region_index"] if primary else None,
                "primary_region_instance_id": (
                    primary["region_instance_id"] if primary else None
                ),
            }
        )
    floor_counts = Counter(item["membership_status"] for item in floor_records)
    placement_counts = Counter(
        item["membership_status"] for item in placement_memberships
    )
    sidecar = {
        "artifact_status": ("complete_additive_mp3d_camera_region_membership_sidecar"),
        "boundary_descriptor_is_legality_gate": False,
        "candidate_filtering": "none",
        "candidate_legality_effect": "none",
        "episode_counted": False,
        "formal_release_status": "not_run",
        "floor_membership_status_counts": {
            status: floor_counts[status]
            for status in ("unique", "boundary", "unknown", "ambiguous")
        },
        "floor_sample_count": len(floor_records),
        "floor_sample_memberships": floor_records,
        "placement_count": len(placement_memberships),
        "placement_membership_status_counts": {
            status: placement_counts[status]
            for status in ("unique", "boundary", "unknown", "ambiguous")
        },
        "placement_memberships": placement_memberships,
        "placements_per_floor_sample": 36,
        "region_instance_count": 2,
        "regions": regions,
        "research_only": True,
        "room_id": ROOM_ID,
        "studio_used": False,
    }
    catalog = {
        "artifact_status": "complete_target_agnostic_camera_placement_catalog",
        "camera_height_variants_m": HEIGHTS,
        "candidate_filtering": "none",
        "episode_counted": False,
        "formal_release_status": "not_run",
        "floor_position_authority": ("live_habitat_pathfinder_declared_navmesh_export"),
        "placement_count": len(catalog_placements),
        "placements": catalog_placements,
        "research_only": True,
        "room_id": ROOM_ID,
        "studio_used": False,
        "yaw_degrees": YAWS,
    }
    return sidecar, catalog


def _selected_signature(record: dict) -> list[tuple[str, str, dict]]:
    return [
        (
            bank["region_instance_id"],
            selected["placement_id"],
            selected["ranking"],
        )
        for bank in record["region_banks"]
        for selected in bank["selected_candidates"]
    ]


def test_selects_four_distinct_floor_positions_per_region() -> None:
    sidecar, catalog = _fixture()
    record = analyze_mp3d_region_viewpoints(
        sidecar=sidecar,
        catalog=catalog,
        base_request=_base_request(),
    ).to_record()

    assert record["region_bank_count"] == 2
    assert record["selected_viewpoint_count"] == 8
    assert record["selected_unique_floor_position_count"] == 8
    assert record["selection_hard_numeric_thresholds"] == []
    assert record["input_floor_membership_status_counts"] == {
        "ambiguous": 1,
        "boundary": 1,
        "unique": 8,
        "unknown": 1,
    }
    for bank in record["region_banks"]:
        selected = bank["selected_candidates"]
        assert bank["selected_viewpoint_count"] == 4
        assert bank["selected_unique_floor_position_count"] == 4
        assert (
            len({tuple(item["camera"]["floor_position_m"]) for item in selected}) == 4
        )
        assert all(
            item["region_category_used_for_generation_or_ranking"] is False
            and item["source_route_or_visibility_used_for_ranking"] is False
            and item["input_placement_legality_preserved"] is True
            for item in selected
        )


def test_selection_is_deterministic_and_independent_of_category_names() -> None:
    sidecar, catalog = _fixture()
    first = analyze_mp3d_region_viewpoints(
        sidecar=sidecar,
        catalog=catalog,
        base_request=_base_request(),
    ).to_record()
    sidecar["regions"] = list(reversed(sidecar["regions"]))
    sidecar["floor_sample_memberships"] = list(
        reversed(sidecar["floor_sample_memberships"])
    )
    sidecar["placement_memberships"] = list(reversed(sidecar["placement_memberships"]))
    catalog["placements"] = list(reversed(catalog["placements"]))
    sidecar["regions"][0]["category_code"] = "Z"
    sidecar["regions"][0]["category_name"] = "unseen posthoc category"
    second = analyze_mp3d_region_viewpoints(
        sidecar=sidecar,
        catalog=catalog,
        base_request=_base_request(),
    ).to_record()

    assert _selected_signature(first) == _selected_signature(second)


def test_all_near_boundary_unique_positions_remain_selectable() -> None:
    sidecar, catalog = _fixture()
    for floor in sidecar["floor_sample_memberships"]:
        if floor["membership_status"] == "unique":
            floor["distance_to_nearest_polygon_boundary_m"] = 0.001
            floor["near_boundary_descriptor"] = True
    for placement in sidecar["placement_memberships"]:
        if placement["membership_status"] == "unique":
            placement["distance_to_nearest_polygon_boundary_m"] = 0.001
            placement["near_boundary_descriptor"] = True

    record = analyze_mp3d_region_viewpoints(
        sidecar=sidecar,
        catalog=catalog,
        base_request=_base_request(),
    ).to_record()

    assert record["selected_viewpoint_count"] == 8
    assert all(
        selected["near_boundary_descriptor"] is True
        for bank in record["region_banks"]
        for selected in bank["selected_candidates"]
    )


def test_insufficient_unique_positions_fails_without_copying() -> None:
    sidecar, catalog = _fixture(positions_per_region=3)

    with pytest.raises(
        MP3DRegionViewpointError,
        match="only 3 unique floor positions; needs 4 without copying",
    ):
        analyze_mp3d_region_viewpoints(
            sidecar=sidecar,
            catalog=catalog,
            base_request=_base_request(),
        )


@pytest.mark.parametrize("drift", ["id", "coordinate"])
def test_sidecar_catalog_id_or_coordinate_drift_fails(drift: str) -> None:
    sidecar, catalog = _fixture()
    if drift == "id":
        sidecar["placement_memberships"][0]["placement_id"] = "missing_placement"
    else:
        sidecar["placement_memberships"][0]["position_m"][0] += 1.0e-6

    with pytest.raises(MP3DRegionViewpointError, match="drift"):
        analyze_mp3d_region_viewpoints(
            sidecar=sidecar,
            catalog=catalog,
            base_request=_base_request(),
        )


def _builder_args(tmp_path: Path):
    sidecar, catalog = _fixture()
    sidecar_path = tmp_path / "sidecar.json"
    catalog_path = tmp_path / "catalog.json"
    request_path = tmp_path / "base_request.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    request_path.write_text(json.dumps(_base_request()), encoding="utf-8")
    output = tmp_path / "region-banks"
    return (
        builder.parse_args(
            [
                "--camera-region-sidecar",
                str(sidecar_path),
                "--camera-catalog",
                str(catalog_path),
                "--base-m1-request",
                str(request_path),
                "--output",
                str(output),
            ]
        ),
        output,
    )


def test_builder_writes_independent_valid_m1_requests(tmp_path: Path) -> None:
    args, output = _builder_args(tmp_path)

    assert builder.run(args) == output
    record = load_json(output / "region_viewpoint_banks.json")

    assert record["selected_viewpoint_count"] == 8
    assert len(record["m1_capture_request_paths"]) == 8
    request_ids = set()
    for bank in record["region_banks"]:
        for selected in bank["selected_candidates"]:
            request = load_json(output / selected["m1_capture_request_path"])
            assert validate_capture_request(request, room_id=ROOM_ID) == []
            assert request["request_id"] not in request_ids
            request_ids.add(request["request_id"])
            assert (
                request["primary_camera_rig"]["world_from_rig"]["translation_m"]
                == selected["camera"]["position_m"]
            )
            assert request["qa_views"] == [
                {
                    "height_m": selected["camera"]["floor_position_m"][1],
                    "kind": "topdown",
                    "meters_per_pixel": 0.05,
                    "qa_id": "navmesh_topdown",
                }
            ]
    assert (output / "README.md").is_file()

    with pytest.raises(
        builder.MP3DRegionViewpointBuildError, match="refusing to replace output"
    ):
        builder.run(args)

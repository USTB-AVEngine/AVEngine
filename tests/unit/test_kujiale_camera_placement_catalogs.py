from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avengine.camera_placement import DEFAULT_YAW_DEGREES
from avengine.contracts.json_io import load_json
from tools.m6z import build_kujiale_camera_placement_catalogs as builder


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _inputs(tmp_path: Path, *, room_type: str = "unknown room kind"):
    stage = tmp_path / "kujiale_test_full_home.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    metadata = _write_json(
        tmp_path / "metadata.json",
        {
            "schema": "avengine_optional_residential_scene_metadata_v1",
            "dataset_id": "spatialverse/InteriorAgent",
            "scene_id": "kujiale_test",
            "room_id": "kujiale_test_room_007",
            "room_scope": "room_007",
            "room_type": room_type,
            "room_polygon_xy_m": [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 4.0],
                [0.0, 4.0],
            ],
            "floor_z_m": 0.0,
            "source_reference": str(stage),
            "object_role_counts": {"ground_blocker": 1},
            "objects": [
                {
                    "object_id": "low_table",
                    "navigation_role": "ground_blocker",
                    "bounds_xyz_m": [[1.0, 1.0, 0.0], [2.0, 2.0, 0.5]],
                    "footprint_parts_xy_m": [[[1.0, 1.0], [2.0, 2.0]]],
                    "footprint_basis": "unit_test_mesh_footprint",
                }
            ],
        },
    )
    source = _write_json(
        tmp_path / "source.json",
        {
            "schema": "avengine_optional_residential_source_profile_v1",
            "dataset_id": "spatialverse/InteriorAgent",
            "scene_id": "kujiale_test",
            "map_path": "/Game/AVEngine/Test/kujiale_test_full_home_v1",
        },
    )
    visual = _write_json(
        tmp_path / "visual.json",
        {
            "schema_version": "avengine_optional_interioragent_kujiale_profile_v1",
            "dataset_id": "spatialverse/InteriorAgent",
            "scene_id": "kujiale_test",
            "map_path": "/Game/AVEngine/Test/kujiale_test_full_home_v1",
            "source_stage_name": stage.name,
            "selected_scopes": ["room_007", "wall", "ceiling", "floor"],
        },
    )
    return metadata, source, visual


def _args(
    tmp_path: Path,
    *,
    room_type: str = "unknown room kind",
    output_name: str = "catalog",
):
    metadata, source, visual = _inputs(tmp_path, room_type=room_type)
    output = tmp_path / output_name
    args = builder.parse_args(
        [
            "--scene-metadata",
            str(metadata),
            "--source-profile",
            str(source),
            "--visual-profile",
            str(visual),
            "--meters-per-pixel",
            "0.5",
            "--camera-spacing-m",
            "0.5",
            "--output",
            str(output),
        ]
    )
    return args, output, metadata, source, visual


def test_complete_catalog_crosses_free_space_height_and_twelve_yaws(
    tmp_path: Path,
) -> None:
    args, output, _metadata, _source, _visual = _args(tmp_path)
    assert builder.run(args) == output
    delivery = load_json(output / "delivery.json")
    catalog = load_json(
        output / "room_007" / "camera_placement_catalog.json"
    )

    assert delivery["room_instance_count"] == 1
    assert delivery["candidate_filtering"] == "none"
    assert delivery["studio_used"] is False
    assert catalog["candidate_filtering"] == "none"
    assert catalog["room_type_annotation"] == "unknown room kind"
    assert catalog["room_polygon_role"] == "geometric_region_boundary_only"
    assert catalog["yaw_degrees"] == list(DEFAULT_YAW_DEGREES)
    assert set(catalog["camera_height_variants_above_floor_m"]) == {
        "low",
        "base",
        "high",
    }
    assert catalog["placement_count"] == (
        catalog["sampled_floor_position_count"] * 3 * 12
    )
    assert delivery["total_placement_count"] == catalog["placement_count"]
    assert all(
        item["semantic_target"] is None
        and item["source_path_join_status"] == "pending"
        and item["difficulty_status"] == "not_evaluated"
        for item in catalog["placements"]
    )
    # The low table ends at 0.5 m, below every camera eye. Its declared ground
    # footprint still makes that floor position invalid for camera placement.
    assert all(
        not (
            1.0 <= item["position_m"][0] <= 2.0
            and 1.0 <= item["position_m"][2] <= 2.0
        )
        for item in catalog["placements"]
    )
    with np.load(output / "room_007" / "camera_free_space_raster.npz") as raster:
        assert int(np.count_nonzero(raster["free_space_mask"])) == catalog[
            "free_space_raster_pixel_count"
        ]
        assert len(raster["sampled_pixels_rc"]) == catalog[
            "sampled_floor_position_count"
        ]

    with pytest.raises(
        builder.KujialeCameraCatalogError, match="refusing to replace output"
    ):
        builder.run(args)


def test_room_type_annotation_does_not_change_generated_placements(
    tmp_path: Path,
) -> None:
    args, first_output, metadata, _source, _visual = _args(
        tmp_path, room_type="kitchen", output_name="first"
    )
    builder.run(args)
    first = load_json(first_output / "room_007" / "camera_placement_catalog.json")

    value = json.loads(metadata.read_text(encoding="utf-8"))
    value["room_type"] = "never seen before"
    metadata.write_text(json.dumps(value), encoding="utf-8")
    args.output = tmp_path / "second"
    builder.run(args)
    second = load_json(
        tmp_path / "second" / "room_007" / "camera_placement_catalog.json"
    )

    assert first["room_type_annotation"] == "kitchen"
    assert second["room_type_annotation"] == "never seen before"
    assert first["placements"] == [
        {**item, "room_type_annotation": "kitchen"}
        for item in second["placements"]
    ]


def test_current_profile_map_mismatch_fails_before_output(tmp_path: Path) -> None:
    args, output, _metadata, source, _visual = _args(tmp_path)
    value = json.loads(source.read_text(encoding="utf-8"))
    value["map_path"] = "/Game/AVEngine/Test/kujiale_test_live_v2"
    source.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(RuntimeError, match="map_path differ"):
        builder.run(args)
    assert not output.exists()


def test_multiroom_bundle_requires_unique_room_identity(tmp_path: Path) -> None:
    args, output, metadata, _source, _visual = _args(tmp_path)
    args.scene_metadata = [metadata, metadata]
    with pytest.raises(
        builder.KujialeCameraCatalogError, match="room_id and room_scope.*unique"
    ):
        builder.run(args)
    assert not output.exists()

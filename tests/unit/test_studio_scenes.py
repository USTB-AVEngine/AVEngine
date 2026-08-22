from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from avengine.studio.scenes import (
    StudioSceneError,
    list_scene_bundles,
    load_draft_obstacle_grid,
    load_scene_bundle,
    scene_file_path,
)
from avengine.studio.validation import check_points


def make_bundle(scenes_root: Path, room_id: str, *, with_grid: bool = True) -> dict:
    bundle_dir = scenes_root / room_id
    bundle_dir.mkdir(parents=True)
    positions = np.asarray(
        [[0, 0, 0], [4, 0, 0], [4, 0, 4], [0, 0, 4]], dtype=np.float32
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    material_ids = np.asarray([0, 0], dtype=np.uint32)
    (bundle_dir / "mesh_positions.bin").write_bytes(positions.tobytes())
    (bundle_dir / "mesh_indices.bin").write_bytes(triangles.tobytes())
    (bundle_dir / "mesh_material_ids.bin").write_bytes(material_ids.tobytes())
    bundle: dict = {
        "schema": "avengine_studio_scene_bundle_v1",
        "room_id": room_id,
        "display_name": room_id,
        "mesh": {
            "positions": {"file": "mesh_positions.bin", "dtype": "float32", "count": 4},
            "indices": {"file": "mesh_indices.bin", "dtype": "uint32", "count": 6},
            "triangle_material_ids": {
                "file": "mesh_material_ids.bin", "dtype": "uint32", "count": 2
            },
            "materials": [{"id": 0, "category": "floor", "color": "#8d99ae"}],
            "bounds_m": [[0, 0, 0], [4, 0, 4]],
        },
    }
    if with_grid:
        # 10x10 grid at 0.5 m/px covering x,z in [0,5): walkable everywhere
        # except the column band x in [2.0, 2.5) (col 4) so a ray across it
        # is blocked and a point inside it is rejected.
        grid = np.ones((10, 10), dtype=np.uint8)
        grid[:, 4] = 0
        bundle["obstacle_map"] = {
            "floor_height_m": 0.0,
            "meters_per_pixel": 0.5,
            "bounds_m": [[0.0, 0.0, 0.0], [5.0, 1.0, 5.0]],
            "grid_shape": [10, 10],
            "grid_order": "row_is_z_col_is_x_from_lower_bounds",
            "navmesh_grid_packbits_b64": base64.b64encode(
                np.packbits(grid.reshape(-1)).tobytes()
            ).decode("ascii"),
            "rigid_obstacles": [],
        }
    (bundle_dir / "bundle.json").write_text(
        json.dumps(bundle), encoding="utf-8"
    )
    return bundle


def test_list_and_load_bundles(tmp_path: Path) -> None:
    make_bundle(tmp_path, "room_a")
    (tmp_path / "not_a_bundle").mkdir()
    scenes = list_scene_bundles(tmp_path)
    assert [scene["room_id"] for scene in scenes] == ["room_a"]
    assert scenes[0]["has_obstacle_map"] is True
    bundle = load_scene_bundle(tmp_path, "room_a")
    assert bundle["mesh"]["materials"][0]["category"] == "floor"
    with pytest.raises(StudioSceneError):
        load_scene_bundle(tmp_path, "missing")


def test_scene_file_path_is_whitelisted(tmp_path: Path) -> None:
    make_bundle(tmp_path, "room_a")
    path = scene_file_path(tmp_path, "room_a", "mesh_positions.bin")
    assert path.name == "mesh_positions.bin"
    with pytest.raises(StudioSceneError):
        scene_file_path(tmp_path, "room_a", "../secrets.txt")
    with pytest.raises(StudioSceneError):
        scene_file_path(tmp_path, "room_a", "bundle.json.tmp")


def test_draft_grid_roundtrip_and_walkability(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path, "room_a")
    grid = load_draft_obstacle_grid(bundle)
    assert grid is not None
    assert grid.grid.shape == (10, 10)
    # far from the blocked band: walkable; deep inside it with radius 1
    # neighborhood still blocked only if all neighbors blocked, so use the
    # exact center column with neighbors also blocked via radius_cells=0
    assert grid.is_walkable(0.75, 0.75) is True
    assert grid.is_walkable(2.25, 2.25, radius_cells=0) is False
    assert grid.is_walkable(99.0, 99.0) is False


def test_check_points_flags_blocked_and_obb(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path, "room_a")
    bundle["obstacle_map"]["rigid_obstacles"] = [
        {
            "object_id": 7,
            "handle": "sofa",
            "blocks_source_center": True,
            "world_obb": {
                "center_m": [1.0, 0.5, 1.0],
                "axes_xyz": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "half_extents_m": [0.4, 0.5, 0.4],
            },
        }
    ]
    grid = load_draft_obstacle_grid(bundle)
    result = check_points(
        grid,
        [
            {"label": "free", "position_m": [3.75, 0.0, 0.75]},
            {"label": "inside_sofa", "position_m": [1.0, 0.4, 1.0]},
            {"label": "outside", "position_m": [40.0, 0.0, 40.0]},
        ],
    )
    by_label = {record["label"]: record for record in result["points"]}
    assert by_label["free"]["ok"] is True
    assert by_label["inside_sofa"]["ok"] is False
    assert by_label["outside"]["ok"] is False
    assert result["all_ok"] is False

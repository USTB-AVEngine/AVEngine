"""Walkable-floor grid product: writer, reader, rasterisation, conversion.

合成房间(方房 + 中央岛台)证明:岛台格不可走;存下的净空永远不高于真实
距离(保守);带边距抽点永远合法;栅格边界当障碍;篡改数组即拒绝载入;
导航网格随机点栅格化按最少点数判可走,场景自己的导航点所在格补为可走;
可行区栅格按声明的坐标约定转换。不引用任何真实房间。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from walkable_grid import (  # noqa: E402
    WalkableGrid,
    clearance_from_mask,
    write_walkable_grid,
)
from build_qa_v3_walkable_grid import (  # noqa: E402
    grid_from_feasible_region,
    grid_from_navmesh_points,
    stamp_scene_points,
)

ROOM_CM = 1400.0
CELL_CM = 10.0
ISLAND = (-100.0, -100.0, 100.0, 100.0)      # a 2 m furniture island in the middle


def room_grid(root, scene_id="synth_room"):
    n = int(ROOM_CM / CELL_CM)
    origin = (-ROOM_CM / 2.0, -ROOM_CM / 2.0)
    walkable = np.ones((n, n), dtype=bool)
    for row in range(n):
        for col in range(n):
            x = origin[0] + (col + 0.5) * CELL_CM
            y = origin[1] + (row + 0.5) * CELL_CM
            if ISLAND[0] <= x <= ISLAND[2] and ISLAND[1] <= y <= ISLAND[3]:
                walkable[row, col] = False
    write_walkable_grid(root, scene_id=scene_id, cell_cm=CELL_CM, origin_xy_cm=origin,
                        walkable=walkable, source={"kind": "feasible_region_mask", "test": True})
    return WalkableGrid.load(root)


# ---------------------------------------------------------------------------
# grid product
# ---------------------------------------------------------------------------

def test_grid_marks_the_island_and_keeps_clearance_conservative(tmp_path):
    grid = room_grid(tmp_path / "grid")
    assert not grid.is_walkable((0.0, 0.0))                 # island centre
    assert grid.is_walkable((400.0, 400.0), margin_cm=30.0)
    # 25 cm from the island edge: the stored clearance never exceeds the truth
    point = (ISLAND[2] + 25.0, 0.0)
    assert grid.is_walkable(point)
    assert grid.clearance_at(point) <= 25.0
    assert grid.clearance_at(point) >= 25.0 - 1.5 * CELL_CM - CELL_CM
    # outside the raster is never walkable and the border counts as an obstacle
    assert not grid.is_walkable((ROOM_CM, 0.0))
    assert not grid.is_walkable((ROOM_CM / 2.0 - 5.0, 0.0), margin_cm=20.0)
    rng = np.random.default_rng(3)
    for _ in range(200):
        assert grid.is_walkable(grid.sample_xy(rng, margin_cm=30.0), margin_cm=30.0)
    ok, detail = grid.route_ok([(400.0, 0.0), (200.0, 0.0), (0.0, 0.0)], margin_cm=30.0)
    assert not ok and detail["worst_frame"] == 2 and detail["min_clearance_cm"] == -1.0
    assert grid.identity["scene_id"] == "synth_room" and grid.identity["cell_cm"] == CELL_CM
    assert abs(grid.walkable_area_m2 - (14.0 * 14.0 - 2.0 * 2.0)) < 0.5
    # a tampered arrays file is refused
    arrays = tmp_path / "grid" / "walkable_grid.npz"
    arrays.write_bytes(arrays.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="digest"):
        WalkableGrid.load(tmp_path / "grid")


def test_rasterisation_min_points_and_scene_point_stamping():
    rng = np.random.default_rng(0)
    # a dense cloud over a 2 m x 1 m rectangle plus one stray point
    cloud = np.column_stack([rng.uniform(0.0, 200.0, 5000), rng.uniform(0.0, 100.0, 5000)])
    points = np.vstack([cloud, [[500.0, 500.0]]])
    xyz = np.column_stack([points, np.full(len(points), 30.0)])
    walkable, origin, counts = grid_from_navmesh_points(xyz, 10.0, min_points=2)
    assert origin == (-10.0, -10.0)
    assert counts.sum() == len(points)
    dense_cells = walkable[1:11, 1:21]
    assert dense_cells.all()                 # 200 cells, 25 points each on average
    stray = (int((500.0 - origin[1]) // 10.0), int((500.0 - origin[0]) // 10.0))
    assert counts[stray] == 1 and not walkable[stray]      # one point is below the minimum
    report = stamp_scene_points(walkable, origin, 10.0, [(500.0, 500.0), (-500.0, 0.0)])
    assert report == {"cells_added_from_scene_points": 1, "scene_points_outside_raster": 1}
    assert walkable[stray]                   # a scene point makes its cell walkable
    clearance = clearance_from_mask(walkable, 10.0)
    assert clearance[~walkable].max() == 0.0
    assert clearance[5, 10] > clearance[1, 1]     # interior beats the edge


def test_feasible_region_conversion_follows_the_declared_contract(tmp_path):
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 1:5] = 1
    np.savez(tmp_path / "region.npz", feasible_mask=mask)
    (tmp_path / "region.json").write_text(json.dumps({"source1": {
        "schema": "avengine_room_feasible_region_v1", "mask_shape_hw": [4, 6],
        "bounds_m": [[-1.0, 0.0, -2.0], [2.0, 0.0, 0.0]], "pixel_size_x_m": 0.5,
        "pixel_size_z_m": 0.5, "feasible_pixel_count": 8,
        "claim_boundary": "source centre only"}}))
    config = {"metadata": str(tmp_path / "region.json"), "metadata_key": "source1",
              "arrays": str(tmp_path / "region.npz"), "mask_key": "feasible_mask",
              "coordinate_contract": "habitat_xz_m_to_ue_xy_cm_v1"}
    walkable, origin, cell_cm, facts = grid_from_feasible_region(config)
    assert origin == (-100.0, -200.0) and cell_cm == 50.0
    assert walkable.shape == (4, 6) and walkable.sum() == 8
    assert facts["kind"] == "feasible_region_mask"
    with pytest.raises(RuntimeError, match="contract"):
        grid_from_feasible_region(dict(config, coordinate_contract="other"))

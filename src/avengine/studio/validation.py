"""Draft placement checks against a serialized scene bundle.

These checks answer in milliseconds while the user drags markers. They are
deliberately labeled draft: the grid is a rasterized navmesh snapshot and
the OBB clearances reuse the engine's own point_to_world_obb_clearance, but
the native placement gates inside the render chain remain the authority.
"""

from __future__ import annotations

import math

from avengine.m6x.geometry import point_to_world_obb_clearance
from avengine.studio.scenes import DraftObstacleGrid

DRAFT_CLAIM = (
    "draft Studio preview check; the native placement gates in the render "
    "chain remain the authority"
)


def check_points(
    grid: DraftObstacleGrid,
    points: list[dict],
    *,
    minimum_rigid_clearance_m: float = 0.0,
) -> dict:
    """Check labeled world-meter points: [{"label": ..., "position_m": [x,y,z]}]."""

    records = []
    all_ok = True
    for entry in points:
        label = str(entry.get("label", "point"))
        position = entry.get("position_m")
        if (
            not isinstance(position, (list, tuple))
            or len(position) != 3
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in position)
        ):
            records.append({"label": label, "ok": False, "reason": "invalid point"})
            all_ok = False
            continue
        x_m, y_m, z_m = (float(v) for v in position)
        walkable = grid.is_walkable(x_m, z_m)
        reason = None if walkable else "outside the walkable navmesh"

        rigid_clearance = math.inf
        inside_rigid = False
        nearest = None
        for obstacle in grid.rigid_obstacles:
            if obstacle.get("blocks_source_center", True) is False:
                continue
            clearance, inside = point_to_world_obb_clearance(
                [x_m, y_m, z_m], obstacle
            )
            if clearance < rigid_clearance or inside:
                rigid_clearance = clearance
                inside_rigid = inside_rigid or inside
                nearest = obstacle.get("handle")
        rigid_ok = not inside_rigid and (
            math.isinf(rigid_clearance)
            or rigid_clearance >= minimum_rigid_clearance_m
        )
        if walkable and not rigid_ok:
            reason = f"too close to rigid obstacle {nearest}"
        ok = walkable and rigid_ok
        all_ok = all_ok and ok
        records.append(
            {
                "label": label,
                "ok": ok,
                "walkable": walkable,
                "rigid_clearance_m": None
                if math.isinf(rigid_clearance)
                else round(rigid_clearance, 4),
                "reason": reason,
            }
        )
    return {"claim": DRAFT_CLAIM, "all_ok": all_ok, "points": records}

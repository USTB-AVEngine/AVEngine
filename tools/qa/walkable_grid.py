#!/usr/bin/env python3
"""Per-scene walkable-floor grid: where an actor may stand, and how far the
nearest obstacle is.

Why
---
Until 2026-09-03 every route the solver could use came from a fixed bank made
upstream (UE navigation paths for the apartment, Studio trajectories for
Kujiale).  When the bank had no route with the geometry a question needs, the
solver reported an exhausted budget: on Kujiale the card1 family filled 1 of
18 cells per round because the 200 moving routes there are short (median
1.88 m between anchor and query frame) and few.  The room-level fix is to let
the solver design a route for the pose it has chosen, which needs one input
the bank never exposed: the room's walkable floor.

What this is
------------
A raster over the scene's normalised horizontal plane (UE ``(x, y)``
centimetres, the same plane as ``scene_sampler``).  Each cell says whether an
actor's centre may stand there.  A second array stores, per walkable cell, a
conservative clearance: the distance from any point in the cell to the
nearest non-walkable cell, so a route can be required to keep a margin from
furniture and walls.  Cell ``(row, col)`` covers
``x in [origin_x + col*cell, origin_x + (col+1)*cell)`` and likewise ``y``
with ``row``.

Sources (recorded in the product, never inferred from a room id)
----------------------------------------------------------------
* ``ue_navmesh_random_points``: random points drawn from the stage's own
  RecastNavMesh, rasterised.  The navmesh is already shrunk by the agent
  radius, so cells are walkable for an agent of that size.
* ``feasible_region_mask``: an existing ``avengine_room_feasible_region_v1``
  raster (the Kujiale route bank's own feasibility map), re-expressed on this
  schema through the declared coordinate contract.

Boundary
--------
The grid is two-dimensional: it knows nothing about heights, so a table top
is "not walkable" only if the source treated it so.  The margin is a research
placeholder.  Pixel truth on rendered candidates remains the authority for
"the actor did not walk through furniture".
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA = "qa_v3_walkable_grid_v1"
INDEX_NAME = "walkable_grid.json"
ARRAYS_NAME = "walkable_grid.npz"
# Clearance is measured centre-to-centre by the distance transform; a point
# anywhere in a cell and the far corner of an obstacle cell add up to at most
# sqrt(2) cells.  1.5 cells keeps the stored value conservative.
CLEARANCE_SLACK_CELLS = 1.5
SOURCE_KINDS = ("ue_navmesh_random_points", "feasible_region_mask")


def rasterize_points(points_xy_cm: np.ndarray, cell_cm: float, *,
                     padding_cells: int = 1) -> tuple[np.ndarray, tuple[float, float]]:
    """Count points per cell on a grid that covers them with a padding ring.

    Returns (counts[rows, cols], origin_xy_cm).  Rows index y, columns x."""
    points = np.asarray(points_xy_cm, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] == 0:
        raise ValueError("no points to rasterise")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    cell = float(cell_cm)
    if not math.isfinite(cell) or cell <= 0.0:
        raise ValueError("cell_cm must be positive")
    pad = int(padding_cells)
    if pad < 0:
        raise ValueError("padding_cells must be non-negative")
    origin = (math.floor(points[:, 0].min() / cell) * cell - pad * cell,
              math.floor(points[:, 1].min() / cell) * cell - pad * cell)
    cols = int(math.floor((points[:, 0].max() - origin[0]) / cell)) + 1 + pad
    rows = int(math.floor((points[:, 1].max() - origin[1]) / cell)) + 1 + pad
    col_index = np.floor((points[:, 0] - origin[0]) / cell).astype(np.int64)
    row_index = np.floor((points[:, 1] - origin[1]) / cell).astype(np.int64)
    counts = np.zeros((rows, cols), dtype=np.int64)
    np.add.at(counts, (row_index, col_index), 1)
    return counts, (float(origin[0]), float(origin[1]))


def clearance_from_mask(walkable: np.ndarray, cell_cm: float) -> np.ndarray:
    """Conservative clearance in centimetres for every cell (0 where not walkable).

    The grid border counts as an obstacle, so a route can never leave the
    raster while satisfying a positive margin."""
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(walkable, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("walkable mask must be two-dimensional")
    padded = np.pad(mask, 1, constant_values=False)
    centre_distance_cells = distance_transform_edt(padded)[1:-1, 1:-1]
    clearance = (centre_distance_cells - CLEARANCE_SLACK_CELLS) * float(cell_cm)
    clearance = np.where(mask, np.maximum(clearance, 0.0), 0.0)
    return clearance.astype(np.float32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_walkable_grid(output_dir: Path | str, *, scene_id: str, cell_cm: float,
                        origin_xy_cm: Sequence[float], walkable: np.ndarray,
                        source: dict, code: dict | None = None,
                        validation: dict | None = None,
                        extra_arrays: dict[str, np.ndarray] | None = None) -> Path:
    """Write the product (index JSON + arrays).  Refuses to overwrite."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / INDEX_NAME
    arrays_path = output / ARRAYS_NAME
    if index_path.exists() or arrays_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing grid in {output}")
    if source.get("kind") not in SOURCE_KINDS:
        raise ValueError(f"source.kind must be one of {SOURCE_KINDS}")
    mask = np.asarray(walkable, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("walkable mask must be a non-empty 2-D mask with walkable cells")
    clearance = clearance_from_mask(mask, cell_cm)
    arrays = {"walkable": mask, "clearance_cm": clearance}
    for key, value in (extra_arrays or {}).items():
        if key in arrays:
            raise ValueError(f"extra array {key!r} collides with a required array")
        arrays[key] = np.asarray(value)
    np.savez_compressed(arrays_path, **arrays)
    cell = float(cell_cm)
    index = {
        "schema": SCHEMA,
        "scene_id": str(scene_id),
        "cell_cm": cell,
        "origin_xy_cm": [float(origin_xy_cm[0]), float(origin_xy_cm[1])],
        "shape_hw": [int(mask.shape[0]), int(mask.shape[1])],
        "axes": {"rows": "ue_y_cm", "cols": "ue_x_cm"},
        "arrays": {"path": ARRAYS_NAME, "sha256": _sha256_file(arrays_path),
                   "walkable": "bool[rows, cols]",
                   "clearance_cm": ("float32[rows, cols]; conservative distance from any "
                                    "point of the cell to the nearest non-walkable cell "
                                    "or the grid border; 0 where not walkable")},
        "clearance_slack_cells": CLEARANCE_SLACK_CELLS,
        "walkable_cells": int(mask.sum()),
        "walkable_area_m2": float(mask.sum()) * (cell / 100.0) ** 2,
        "source": source,
        "code": code,
        "validation": validation,
        "status": "research_candidate",
        "boundary": ("two-dimensional walkability of an actor centre; not a collision "
                     "volume, not pixel truth; the margin used by consumers is a "
                     "research placeholder"),
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1))
    return output


@dataclass
class WalkableGrid:
    scene_id: str
    cell_cm: float
    origin_xy_cm: tuple[float, float]
    walkable: np.ndarray
    clearance_cm: np.ndarray
    index: dict = field(default_factory=dict)
    path: str = ""

    @classmethod
    def load(cls, location: str | Path) -> "WalkableGrid":
        root = Path(location)
        index_path = root / INDEX_NAME if root.is_dir() else root
        if not index_path.is_file():
            raise FileNotFoundError(f"walkable grid index not found: {index_path}")
        index = json.loads(index_path.read_text())
        if index.get("schema") != SCHEMA:
            raise ValueError(f"{index_path}: schema {index.get('schema')!r} is not {SCHEMA}")
        arrays_path = index_path.parent / index["arrays"]["path"]
        if _sha256_file(arrays_path) != index["arrays"]["sha256"]:
            raise ValueError(f"{arrays_path}: arrays do not match the index digest")
        with np.load(arrays_path) as arrays:
            walkable = np.asarray(arrays["walkable"], dtype=bool)
            clearance = np.asarray(arrays["clearance_cm"], dtype=np.float32)
        shape = tuple(int(v) for v in index["shape_hw"])
        if walkable.shape != shape or clearance.shape != shape:
            raise ValueError(f"{arrays_path}: array shapes differ from the index")
        if (clearance[~walkable] != 0).any():
            raise ValueError(f"{arrays_path}: clearance is non-zero off the walkable mask")
        return cls(scene_id=str(index["scene_id"]), cell_cm=float(index["cell_cm"]),
                   origin_xy_cm=(float(index["origin_xy_cm"][0]), float(index["origin_xy_cm"][1])),
                   walkable=walkable, clearance_cm=clearance, index=index,
                   path=str(index_path.parent.resolve()))

    # -- geometry ---------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.walkable.shape

    @property
    def bounds_xy_cm(self) -> tuple[tuple[float, float], tuple[float, float]]:
        rows, cols = self.shape
        return ((self.origin_xy_cm[0], self.origin_xy_cm[1]),
                (self.origin_xy_cm[0] + cols * self.cell_cm,
                 self.origin_xy_cm[1] + rows * self.cell_cm))

    @property
    def walkable_area_m2(self) -> float:
        return float(self.walkable.sum()) * (self.cell_cm / 100.0) ** 2

    @property
    def identity(self) -> dict:
        return {"schema": SCHEMA, "scene_id": self.scene_id, "path": self.path,
                "cell_cm": self.cell_cm, "shape_hw": list(self.shape),
                "source_kind": (self.index.get("source") or {}).get("kind"),
                "arrays_sha256": (self.index.get("arrays") or {}).get("sha256"),
                "walkable_area_m2": round(self.walkable_area_m2, 3)}

    def cell_of(self, xy) -> tuple[int, int] | None:
        x, y = float(xy[0]), float(xy[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        col = math.floor((x - self.origin_xy_cm[0]) / self.cell_cm)
        row = math.floor((y - self.origin_xy_cm[1]) / self.cell_cm)
        rows, cols = self.shape
        if 0 <= row < rows and 0 <= col < cols:
            return row, col
        return None

    def clearance_at(self, xy) -> float:
        """Conservative clearance at a point; 0 outside the grid or off the mask."""
        cell = self.cell_of(xy)
        if cell is None:
            return 0.0
        return float(self.clearance_cm[cell])

    def is_walkable(self, xy, margin_cm: float = 0.0) -> bool:
        cell = self.cell_of(xy)
        if cell is None or not self.walkable[cell]:
            return False
        return self.clearance_cm[cell] >= float(margin_cm)

    def route_clearance(self, samples: Sequence[Sequence[float]]) -> tuple[float, int]:
        """Minimum clearance over a route and the index where it occurs."""
        values = [self.clearance_at(xy) if self.is_walkable(xy) else -1.0 for xy in samples]
        index = int(np.argmin(values))
        return float(values[index]), index

    def route_ok(self, samples: Sequence[Sequence[float]], margin_cm: float) -> tuple[bool, dict]:
        """Whether every sample is walkable with the margin; detail names the worst frame."""
        worst, index = self.route_clearance(samples)
        ok = worst >= float(margin_cm)
        return ok, {"min_clearance_cm": round(worst, 2), "worst_frame": index,
                    "margin_cm": float(margin_cm)}

    def cells_with_clearance(self, margin_cm: float) -> np.ndarray:
        return np.flatnonzero(self.walkable.ravel() & (self.clearance_cm.ravel() >= float(margin_cm)))

    def sample_xy(self, rng, margin_cm: float = 0.0) -> tuple[float, float]:
        """Uniform draw over the cells that keep the margin, jittered inside the cell."""
        flat = self.cells_with_clearance(margin_cm)
        if flat.size == 0:
            raise ValueError(f"{self.scene_id}: no cell keeps a {margin_cm} cm clearance")
        chosen = int(flat[int(rng.integers(flat.size))])
        row, col = divmod(chosen, self.shape[1])
        x = self.origin_xy_cm[0] + (col + float(rng.random())) * self.cell_cm
        y = self.origin_xy_cm[1] + (row + float(rng.random())) * self.cell_cm
        return (x, y)

    def fraction_inside(self, points: Sequence[Sequence[float]], margin_cm: float = 0.0) -> float:
        points = list(points)
        if not points:
            return float("nan")
        return sum(1 for xy in points if self.is_walkable(xy, margin_cm)) / len(points)


def grid_from_config(config: Any) -> WalkableGrid:
    """Scene-config entry: a path string or {"path": ...}."""
    if isinstance(config, str):
        return WalkableGrid.load(config)
    if isinstance(config, dict) and "path" in config:
        return WalkableGrid.load(config["path"])
    raise ValueError("walkable_grid must be a path or an object with a path")

"""Scene bundle discovery and loading for the Studio 3D editor.

A bundle directory (built by tools/studio/build_studio_scene_bundle.py)
holds bundle.json plus raw little-endian mesh buffers. Bundles live under
the configured scenes root; the server serves them read-only.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BUNDLE_SCHEMA = "avengine_studio_scene_bundle_v1"

_ACTOR_FILE_PATTERN = re.compile(r"^actor_([a-z0-9_]+)\.glb$")

# Only these bundle files may be served over HTTP. "textured.glb" resolves
# to the external dataset file recorded in bundle.json (read-only).
SERVABLE_BUNDLE_FILES = frozenset(
    {
        "bundle.json",
        "mesh_positions.bin",
        "mesh_indices.bin",
        "mesh_material_ids.bin",
        "reference_frame.png",
        "textured.glb",
        "composition.json",
    }
)


class StudioSceneError(ValueError):
    """Raised for unknown scenes or malformed bundles."""


@dataclass(frozen=True)
class DraftObstacleGrid:
    """Deserialized draft navmesh grid for millisecond placement checks."""

    grid: np.ndarray  # uint8 [rows, cols], 1 = walkable
    bounds_m: np.ndarray  # float64 [2, 3]
    floor_height_m: float
    meters_per_pixel: float
    rigid_obstacles: tuple[dict, ...]

    def cell_for_point(self, x_m: float, z_m: float) -> tuple[int, int]:
        row = int((z_m - self.bounds_m[0][2]) / self.meters_per_pixel)
        col = int((x_m - self.bounds_m[0][0]) / self.meters_per_pixel)
        return row, col

    def is_walkable(self, x_m: float, z_m: float, *, radius_cells: int = 1) -> bool:
        rows, cols = self.grid.shape
        row, col = self.cell_for_point(x_m, z_m)
        if not (0 <= row < rows and 0 <= col < cols):
            return False
        window = self.grid[
            max(0, row - radius_cells) : min(rows, row + radius_cells + 1),
            max(0, col - radius_cells) : min(cols, col + radius_cells + 1),
        ]
        return bool(np.any(window))


def list_scene_bundles(scenes_root: str | Path) -> list[dict]:
    root = Path(scenes_root)
    scenes: list[dict] = []
    if not root.is_dir():
        return scenes
    for child in sorted(root.iterdir()):
        bundle_path = child / "bundle.json"
        if not bundle_path.is_file():
            continue
        try:
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if bundle.get("schema") != BUNDLE_SCHEMA:
            continue
        scenes.append(
            {
                "room_id": bundle.get("room_id", child.name),
                "display_name": bundle.get("display_name", child.name),
                "bundle_dir": str(child),
                "triangle_count": bundle.get("mesh", {})
                .get("triangle_material_ids", {})
                .get("count"),
                "has_obstacle_map": "obstacle_map" in bundle,
                "has_textured_mesh": "textured_mesh" in bundle,
                "has_reference_frame": "reference_frame" in bundle,
                "authoring_mode": bundle.get("authoring", {}).get("mode"),
            }
        )
    return scenes


def load_scene_bundle(scenes_root: str | Path, room_id: str) -> dict:
    for scene in list_scene_bundles(scenes_root):
        if scene["room_id"] == room_id:
            bundle_path = Path(scene["bundle_dir"]) / "bundle.json"
            return json.loads(bundle_path.read_text(encoding="utf-8"))
    raise StudioSceneError(f"unknown scene: {room_id}")


def scene_file_path(scenes_root: str | Path, room_id: str, file_name: str) -> Path:
    actor_match = _ACTOR_FILE_PATTERN.match(file_name)
    if file_name not in SERVABLE_BUNDLE_FILES and actor_match is None:
        raise StudioSceneError(f"file {file_name!r} is not a servable bundle file")
    for scene in list_scene_bundles(scenes_root):
        if scene["room_id"] == room_id:
            bundle_dir = Path(scene["bundle_dir"])
            if file_name == "textured.glb" or actor_match is not None:
                bundle = json.loads(
                    (bundle_dir / "bundle.json").read_text(encoding="utf-8")
                )
                if actor_match is not None:
                    record = bundle.get("actor_models", {}).get(actor_match.group(1))
                    source = (record or {}).get("source_path")
                    label = f"actor model {actor_match.group(1)!r}"
                else:
                    source = bundle.get("textured_mesh", {}).get("source_path")
                    label = "textured mesh"
                if not source or not Path(source).is_file():
                    raise StudioSceneError(f"scene {room_id} declares no {label}")
                return Path(source)
            path = bundle_dir / file_name
            if not path.is_file():
                raise StudioSceneError(f"bundle file missing: {path}")
            return path
    raise StudioSceneError(f"unknown scene: {room_id}")


def scene_dataset_file_path(
    scenes_root: str | Path, room_id: str, relative: str
) -> Path:
    """Resolve a composition object glb under the bundle's declared dataset root."""

    for scene in list_scene_bundles(scenes_root):
        if scene["room_id"] == room_id:
            bundle = json.loads(
                (Path(scene["bundle_dir"]) / "bundle.json").read_text(encoding="utf-8")
            )
            root_value = bundle.get("composition", {}).get("dataset_root")
            if not root_value:
                raise StudioSceneError(f"scene {room_id} declares no composition")
            root = Path(root_value).resolve()
            target = (root / relative).resolve()
            if not str(target).startswith(str(root) + "/"):
                raise StudioSceneError("dataset path escapes the declared root")
            if target.suffix != ".glb" or not target.is_file():
                raise StudioSceneError(f"no such dataset glb: {relative}")
            return target
    raise StudioSceneError(f"unknown scene: {room_id}")


def load_draft_obstacle_grid(bundle: dict) -> DraftObstacleGrid | None:
    payload = bundle.get("obstacle_map")
    if not payload:
        return None
    rows, cols = (int(item) for item in payload["grid_shape"])
    packed = np.frombuffer(
        base64.b64decode(payload["navmesh_grid_packbits_b64"]), dtype=np.uint8
    )
    grid = np.unpackbits(packed)[: rows * cols].reshape(rows, cols)
    return DraftObstacleGrid(
        grid=grid,
        bounds_m=np.asarray(payload["bounds_m"], dtype=np.float64),
        floor_height_m=float(payload["floor_height_m"]),
        meters_per_pixel=float(payload["meters_per_pixel"]),
        rigid_obstacles=tuple(payload.get("rigid_obstacles", ())),
    )

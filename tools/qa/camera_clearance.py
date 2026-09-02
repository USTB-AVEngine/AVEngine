"""Camera clearance table: cube-ring depth geometry and the solver-side reader.

The table is produced once per scene by build_qa_v3_camera_clearance_table.py
(four 90-degree actor-free depth faces per solver camera point, plus per-yaw
clearance summaries).  This module holds the pure-numpy geometry shared by
the builder and its consumers, and the reader the solver uses during search:

  * is this camera point, at this height, facing this yaw, clear of nearby
    furniture in the band where a floor-standing target would appear?
  * which yaws at a point are clear at all (so hopeless points are skipped)?
  * what is the first obstacle along an arbitrary sight line (for later
    visibility prediction)?

Conventions: UE world frame, X forward, Y right, Z up, yaw positive from +X
towards +Y (scene_sampler.bearing_deg reference).  The engine's depth buffer
(sp_depth_meters_) is the radial Euclidean distance from the camera to the
surface, not planar depth: on 2026-09-02 re-projecting stored faces as radial
reproduced direct renders to 0.2-0.4 % median error at every yaw, whereas the
planar reading left 15-22 % errors that grew with the yaw offset from a face
axis.  Faces therefore store radial metres; no-hit pixels carry the sentinel
65504.  Every threshold is a research placeholder until the human calibration
study reports.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA = "qa_v3_camera_clearance_table_v1"
NO_HIT_M = 65504.0
FACE_YAWS_DEG = (0.0, 90.0, 180.0, 270.0)
FACE_HFOV_DEG = 90.0
DEFAULT_METRIC = "target_band_blocked_column_fraction"


class CameraClearanceError(ValueError):
    """The table is missing, inconsistent, or does not cover what was asked."""


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def ndc_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre NDC grids: x runs -1 (left) to +1 (right), y -1 (top) to +1 (bottom)."""
    xs = (np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0
    ys = (np.arange(height, dtype=np.float64) + 0.5) / height * 2.0 - 1.0
    return np.meshgrid(xs, ys)


def face_tan_v(width: int, height: int) -> float:
    """tan(vertical half FOV) of a 90-degree face with the given aspect."""
    return math.tan(math.radians(FACE_HFOV_DEG / 2.0)) * height / width


def face_ray_directions(width: int, height: int, face_yaw_deg: float) -> np.ndarray:
    """Unit world direction (x, y, z) for every pixel of one 90-degree face."""
    x_ndc, y_ndc = ndc_grid(width, height)
    forward = np.ones_like(x_ndc)
    right = x_ndc * math.tan(math.radians(FACE_HFOV_DEG / 2.0))
    down = y_ndc * face_tan_v(width, height)
    norm = np.sqrt(forward ** 2 + right ** 2 + down ** 2)
    yaw = math.radians(face_yaw_deg)
    world_x = (forward * math.cos(yaw) - right * math.sin(yaw)) / norm
    world_y = (forward * math.sin(yaw) + right * math.cos(yaw)) / norm
    world_z = -down / norm
    return np.stack([world_x, world_y, world_z], axis=-1)


def clean_depth(depth: np.ndarray) -> np.ndarray:
    """Planar depth with no-hit pixels (non-finite, <= 0, > 1 km) set to the sentinel."""
    out = np.asarray(depth, dtype=np.float32).copy()
    bad = ~np.isfinite(out) | (out <= 0.0) | (out > 1000.0)
    out[bad] = NO_HIT_M
    return out


def sample_cube_radial(faces_radial: np.ndarray, theta_deg: np.ndarray,
                       elev_deg: np.ndarray) -> np.ndarray:
    """Radial distance along world directions (azimuth theta, elevation elev).

    faces_radial: (4, H, W) radial depth, face k looks along world yaw 90k.
    Nearest-pixel sampling; directions outside the vertical coverage of the
    ring return NaN, no-hit pixels return the sentinel."""
    if faces_radial.ndim != 3 or faces_radial.shape[0] != 4:
        raise CameraClearanceError("faces must be shaped (4, H, W)")
    height, width = faces_radial.shape[1:]
    theta = np.asarray(theta_deg, dtype=np.float64)
    elev = np.deg2rad(np.asarray(elev_deg, dtype=np.float64))
    face = np.mod(np.rint(theta / 90.0), 4).astype(np.int64)
    beta = np.deg2rad(theta - 90.0 * face)
    x_ndc = np.tan(beta) / math.tan(math.radians(FACE_HFOV_DEG / 2.0))
    y_ndc = -np.tan(elev) / np.cos(beta) / face_tan_v(width, height)
    inside = np.abs(y_ndc) <= 1.0
    col = np.clip(np.floor((x_ndc + 1.0) / 2.0 * width), 0, width - 1).astype(np.int64)
    row = np.clip(np.floor((y_ndc + 1.0) / 2.0 * height), 0, height - 1).astype(np.int64)
    radial = faces_radial[face, row, col].astype(np.float64)
    return np.where(inside, radial, np.nan)


class VirtualCamera:
    """Pinhole camera at the production contract used for re-projection."""

    def __init__(self, hfov_deg: float, width: int, height: int):
        self.hfov_deg = float(hfov_deg)
        self.width = int(width)
        self.height = int(height)
        x_ndc, y_ndc = ndc_grid(self.width, self.height)
        tan_h = math.tan(math.radians(self.hfov_deg / 2.0))
        tan_v = tan_h * self.height / self.width
        right = x_ndc * tan_h
        down = y_ndc * tan_v
        self.alpha_deg = np.degrees(np.arctan(right))
        self.elev_deg = np.degrees(-np.arctan(down / np.sqrt(1.0 + right ** 2)))
        self.cos_axis = 1.0 / np.sqrt(1.0 + right ** 2 + down ** 2)

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def reproject_depth(self, faces_radial: np.ndarray, yaw_deg: float) -> np.ndarray:
        """Radial depth image this camera would record at the given world yaw.

        Same convention as a direct engine render, so any statistic computed
        on a direct depth frame can be computed on this image unchanged."""
        radial = sample_cube_radial(faces_radial, yaw_deg + self.alpha_deg, self.elev_deg)
        return radial.astype(np.float32)


def band_column_medians(depth: np.ndarray, rows: tuple[int, int]) -> np.ndarray:
    """Per-column median of the finite, positive, non-sentinel band pixels."""
    band = depth[rows[0]:rows[1], :]
    valid = np.isfinite(band) & (band > 0.0) & (band < NO_HIT_M)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.where(valid, band, np.nan), axis=0)


def blocked_column_fraction(depth: np.ndarray, rows: tuple[int, int], near_m: float) -> float:
    """Share of columns whose median band depth is closer than near_m.

    Same definition as preflight_camera_clearance_depth._blocked_columns (the
    metric validated 16/16 against pixel truth on 2026-09-02): a column counts
    as blocked when the median of its finite, positive band pixels is closer
    than near_m.  No-hit pixels carry no geometry and are left out."""
    median = band_column_medians(depth, rows)
    return float((np.isfinite(median) & (median < near_m)).mean())


def min_pool(depth: np.ndarray, factor: int) -> np.ndarray:
    """Conservative downsample: keep the nearest depth in each factor x factor block."""
    if factor <= 1:
        return depth
    height, width = depth.shape
    if height % factor or width % factor:
        raise CameraClearanceError(f"depth shape {depth.shape} is not divisible by {factor}")
    view = depth.reshape(height // factor, factor, width // factor, factor)
    return view.min(axis=(1, 3))


def point_key(xy: Sequence[float]) -> str:
    """Stable lookup key for a solver camera point (UE cm, 0.1 cm rounding)."""
    return f"{round(float(xy[0]), 1):.1f},{round(float(xy[1]), 1):.1f}"


def yaw_bin_index(yaw_deg: float, yaw_step_deg: float) -> int:
    count = int(round(360.0 / yaw_step_deg))
    return int(np.rint((yaw_deg % 360.0) / yaw_step_deg)) % count


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClearanceRule:
    """Which stored summary decides 'camera clear' and where the cut sits."""
    target_height_m: float
    near_m: float
    blocked_fraction_max: float
    metric: str = DEFAULT_METRIC

    def as_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "target_height_m": self.target_height_m,
                "near_m": self.near_m, "blocked_fraction_max": self.blocked_fraction_max,
                "status": "placeholder_research_not_human_calibrated"}


def _close_index(values: np.ndarray, wanted: float, *, what: str, tol: float) -> int:
    diff = np.abs(np.asarray(values, dtype=np.float64) - float(wanted))
    index = int(np.argmin(diff))
    if diff[index] > tol:
        raise CameraClearanceError(
            f"{what} {wanted} is not in the table grid {np.asarray(values).tolist()}")
    return index


class CameraClearanceTable:
    """Reader for one scene's table directory (camera_clearance_table.json + summaries.npz)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        index_path = self.root / "camera_clearance_table.json"
        if not index_path.is_file():
            raise CameraClearanceError(f"camera clearance table has no index: {index_path}")
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        if self.index.get("schema") != SCHEMA:
            raise CameraClearanceError(
                f"unexpected clearance table schema {self.index.get('schema')!r}")
        summaries_path = self.root / self.index["summaries"]["path"]
        if not summaries_path.is_file():
            raise CameraClearanceError(f"clearance table lacks {summaries_path}")
        with np.load(summaries_path) as data:
            self.target = np.asarray(data["target_band_blocked_column_fraction"], np.float32)
            self.eye = np.asarray(data["eye_band_blocked_column_fraction"], np.float32)
            self.points_xy = np.asarray(data["points_xy_cm"], np.float64)
            self.heights_m = np.asarray(data["camera_heights_m"], np.float64)
            self.yaws_deg = np.asarray(data["yaws_deg"], np.float64)
            self.nears_m = np.asarray(data["nears_m"], np.float64)
            self.target_heights_m = np.asarray(data["target_heights_m"], np.float64)
        if self.target.shape != (len(self.points_xy), len(self.heights_m),
                                 len(self.target_heights_m), len(self.nears_m),
                                 len(self.yaws_deg)):
            raise CameraClearanceError("clearance summaries have inconsistent shapes")
        self.yaw_step_deg = float(self.index["summaries"]["yaw_step_deg"])
        if len(self.yaws_deg) != int(round(360.0 / self.yaw_step_deg)):
            raise CameraClearanceError("yaw grid does not match yaw_step_deg")
        keys = self.index["points"]["keys"]
        if len(keys) != len(self.points_xy):
            raise CameraClearanceError("point keys and summaries disagree on point count")
        self._key_to_index = {key: i for i, key in enumerate(keys)}
        self.scene_id = str(self.index["scene_id"])
        self._faces_cache: dict[str, dict[tuple[int, int], np.ndarray]] = {}
        self._shard_of: dict[tuple[int, int], str] | None = None

    @classmethod
    def load(cls, path: str | Path) -> "CameraClearanceTable":
        root = Path(path)
        if root.is_file():
            root = root.parent
        return cls(root)

    # -- identity ----------------------------------------------------------
    @property
    def identity(self) -> dict[str, Any]:
        return {"path": str(self.root), "schema": self.index["schema"],
                "scene_id": self.scene_id,
                "code_revision": (self.index.get("code") or {}).get("revision"),
                "points": int(len(self.points_xy)),
                "camera_heights_m": self.heights_m.tolist(),
                "yaw_step_deg": self.yaw_step_deg,
                "stage": self.index.get("stage")}

    # -- indices -----------------------------------------------------------
    def point_index(self, xy: Sequence[float]) -> int:
        key = point_key(xy)
        try:
            return self._key_to_index[key]
        except KeyError:
            raise CameraClearanceError(
                f"camera point {key} is not covered by the clearance table "
                f"({self.scene_id}, {len(self.points_xy)} points)") from None

    def height_index(self, camera_height_m: float) -> int:
        return _close_index(self.heights_m, camera_height_m, what="camera height", tol=2.0e-3)

    def has_height(self, camera_height_m: float) -> bool:
        return bool(np.any(np.abs(self.heights_m - float(camera_height_m)) <= 2.0e-3))

    def rule_indices(self, rule: ClearanceRule) -> tuple[int, int]:
        if rule.metric != DEFAULT_METRIC:
            raise CameraClearanceError(f"unsupported clearance metric {rule.metric!r}")
        return (_close_index(self.target_heights_m, rule.target_height_m,
                             what="target height", tol=1.0e-6),
                _close_index(self.nears_m, rule.near_m, what="near distance", tol=1.0e-6))

    def missing_points(self, points: Sequence[Sequence[float]]) -> list[str]:
        return [point_key(xy) for xy in points if point_key(xy) not in self._key_to_index]

    # -- lookups -----------------------------------------------------------
    def blocked_fraction(self, xy: Sequence[float], camera_height_m: float,
                         yaw_deg: float, rule: ClearanceRule) -> float:
        pi = self.point_index(xy)
        hi = self.height_index(camera_height_m)
        ti, ni = self.rule_indices(rule)
        value = self.target[pi, hi, ti, ni, yaw_bin_index(yaw_deg, self.yaw_step_deg)]
        if not np.isfinite(value):
            raise CameraClearanceError(
                f"clearance summary is undefined at {point_key(xy)} h={camera_height_m}")
        return float(value)

    def is_clear(self, xy: Sequence[float], camera_height_m: float, yaw_deg: float,
                 rule: ClearanceRule) -> bool:
        return self.blocked_fraction(xy, camera_height_m, yaw_deg, rule) <= rule.blocked_fraction_max

    def clear_yaw_mask(self, xy: Sequence[float], camera_height_m: float,
                       rule: ClearanceRule) -> np.ndarray:
        pi = self.point_index(xy)
        hi = self.height_index(camera_height_m)
        ti, ni = self.rule_indices(rule)
        return self.target[pi, hi, ti, ni, :] <= rule.blocked_fraction_max

    def points_with_clear_yaw(self, camera_height_m: float,
                              rule: ClearanceRule) -> np.ndarray:
        """Boolean per table point: at least one yaw is clear at this height."""
        hi = self.height_index(camera_height_m)
        ti, ni = self.rule_indices(rule)
        return (self.target[:, hi, ti, ni, :] <= rule.blocked_fraction_max).any(axis=1)

    # -- sight lines ---------------------------------------------------------
    def _shard_map(self) -> dict[tuple[int, int], str]:
        if self._shard_of is None:
            mapping: dict[tuple[int, int], str] = {}
            for shard in self.index["faces"]["shards"]:
                with np.load(self.root / shard["path"]) as data:
                    for pi, hi in zip(data["point_index"], data["height_index"]):
                        mapping[(int(pi), int(hi))] = shard["path"]
            self._shard_of = mapping
        return self._shard_of

    def faces(self, xy: Sequence[float], camera_height_m: float) -> np.ndarray:
        """Stored (4, H, W) radial faces for a point/height (float32, sentinel kept)."""
        pi = self.point_index(xy)
        hi = self.height_index(camera_height_m)
        path = self._shard_map().get((pi, hi))
        if path is None:
            raise CameraClearanceError(f"no stored faces for {point_key(xy)} h={camera_height_m}")
        cache = self._faces_cache.get(path)
        if cache is None:
            cache = {}
            with np.load(self.root / path) as data:
                radial = data["radial_m"]
                for i, (p, h) in enumerate(zip(data["point_index"], data["height_index"])):
                    cache[(int(p), int(h))] = radial[i].astype(np.float32)
            self._faces_cache[path] = cache
        return cache[(pi, hi)]

    def first_obstacle_m(self, xy: Sequence[float], camera_height_m: float,
                         azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
        """Radial distance to the first scene surface along world sight lines."""
        return sample_cube_radial(self.faces(xy, camera_height_m),
                                  np.asarray(azimuth_deg, np.float64),
                                  np.asarray(elevation_deg, np.float64))


def rule_from_params(params: dict, table: CameraClearanceTable | None = None) -> ClearanceRule:
    """The clearance rule is declared in params; missing keys fail closed."""
    keys = ("CAMERA_CLEARANCE_TARGET_HEIGHT_M", "CAMERA_CLEARANCE_NEAR_M",
            "CAMERA_CLEARANCE_BLOCKED_FRACTION_MAX")
    missing = [k for k in keys if k not in params]
    if missing:
        raise CameraClearanceError(f"params lack camera clearance keys {missing}")
    rule = ClearanceRule(target_height_m=float(params[keys[0]]),
                         near_m=float(params[keys[1]]),
                         blocked_fraction_max=float(params[keys[2]]))
    if not 0.0 <= rule.blocked_fraction_max <= 1.0:
        raise CameraClearanceError("CAMERA_CLEARANCE_BLOCKED_FRACTION_MAX must lie in [0,1]")
    if table is not None:
        table.rule_indices(rule)
    return rule


def fallback_heights_from_params(params: dict) -> list[float]:
    raw = params.get("CAMERA_HEIGHT_FALLBACK_M") or []
    if isinstance(raw, (int, float)):
        raw = [raw]
    heights = [float(v) for v in raw]
    if any(not math.isfinite(h) or h <= 0.0 for h in heights):
        raise CameraClearanceError("CAMERA_HEIGHT_FALLBACK_M must be positive numbers")
    return heights

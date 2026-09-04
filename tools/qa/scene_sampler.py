#!/usr/bin/env python3
"""Scene-agnostic candidate search for qa-v3 question types.

取代"手挖 38 条走廊路线 + 写死相机"的房间专用做法。这里只有题型约束与
场景输入两样东西:场景给导航路线和可站点,题型声明语义角色、时间关系、
可见性与答案空间,求解器要么给出合格计划,要么给出**明确的拒绝原因**。
不出现房间 ID、世界坐标、路线名或固定 yaw。

关键手法:答案带不靠枚举相机朝向撞出来,而是**解出来**。给定相机位置 C
与目标片尾位置 P,目标方位 = bearing(P−C) − yaw,所以"答案落在带
[lo,hi)"等价于 yaw ∈ (bearing−hi, bearing−lo]。于是任何场景、任何路线
都能构造出任一答案带,固定机位造成的死区(run01 里视锥右半边零可行)
随之消失。相机与听者位姿由 avengine.camera_pose 同一份结果产生,UE yaw
换算封在那里,调用方不重复实现。

场景输入(SceneInputs)全部来自场景自身:
  - routes:导航系统给的路线,每条按请求时钟重采样位置;
  - stand_points:可站点(路线路点即导航可达点,去重后即可);
  - camera_points:相机候选位置(同上,取相机高度);
  - line_of_sight:可选的占用栅格视线筛查;没有就如实声明未筛,
    交给渲染后的像素真值复核。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from camera_clearance import (  # noqa: E402
    CameraClearanceTable,
    fallback_heights_from_params,
    rule_from_params,
    yaw_bin_index,
)
from route_synthesis import (  # noqa: E402
    ENABLED_KEY as ROUTE_SYNTHESIS_ENABLED_KEY,
    PointSpec,
    RouteSynthesizer,
    SynthesisSettings,
)
from walkable_grid import WalkableGrid, grid_from_config  # noqa: E402
from score_open_answers import angle_credit_radius  # noqa: E402
from qa_v3_arc import Arc, arc_overlap_width_deg  # noqa: E402
from floor_reference import (
    MATCH_TOLERANCE_CM as FLOOR_MATCH_TOLERANCE_CM,
    FloorReference,
    FloorReferenceError,
    floor_reference_from_config,
)

FRAME_COUNT = 75
DEFAULT_VIDEO_FPS = 15.0
QUERY_VISIBILITY_VALUES = ("any", "visible", "out_of_view")


def _positive_integer(value, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{owner} must be a positive integer")
    return value


def _positive_frame_count(value, *, owner="FRAME_COUNT") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f"{owner} must be an integer >= 2")
    return int(value)


def _positive_fps(value, *, owner="VIDEO_FPS") -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner} must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{owner} must be a finite positive number")
    return value


def frame_count_from_params(params: dict | None, *, default: int = FRAME_COUNT) -> int:
    """Read the requested visual clock while retaining old direct-call defaults."""
    value = default if not params or "FRAME_COUNT" not in params else params["FRAME_COUNT"]
    return _positive_frame_count(value)


def video_fps_from_params(params: dict | None, *, default: float = DEFAULT_VIDEO_FPS) -> float:
    value = default if not params or "VIDEO_FPS" not in params else params["VIDEO_FPS"]
    return _positive_fps(value)


def validate_frame_clock(params: dict | None, *, require_clip_seconds: bool = False) -> dict:
    """Validate the frame/sample clock shared by geometry, audio and timeline.

    A 150-frame @15fps point is a ten-second point; silently retaining a
    five-second sample clock would change event placement and motion speed.
    Legacy unit callers may omit the clock fields and use the 75/15 defaults.
    """
    params = params or {}
    frame_count = frame_count_from_params(params)
    frame_rate_hz = video_fps_from_params(params)
    clip_raw = params.get("CLIP_SECONDS")
    if clip_raw is None:
        if require_clip_seconds:
            raise ValueError("params missing CLIP_SECONDS")
        clip_seconds = frame_count / frame_rate_hz
    else:
        try:
            clip_seconds = float(clip_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("CLIP_SECONDS must be a finite positive number") from exc
        if not math.isfinite(clip_seconds) or clip_seconds <= 0.0:
            raise ValueError("CLIP_SECONDS must be a finite positive number")
        expected = frame_count / frame_rate_hz
        if not math.isclose(clip_seconds, expected, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f"FRAME_COUNT={frame_count} at VIDEO_FPS={frame_rate_hz:g} "
                f"requires CLIP_SECONDS={expected:g}, got {clip_seconds:g}")
    result = {"frame_count": frame_count, "frame_rate_hz": frame_rate_hz,
              "clip_seconds": clip_seconds}
    if "SAMPLE_RATE_HZ" in params:
        sample_rate = _positive_integer(params["SAMPLE_RATE_HZ"], owner="SAMPLE_RATE_HZ")
        result["sample_rate_hz"] = sample_rate
        exact_samples = clip_seconds * sample_rate
        sample_count = round(exact_samples)
        if not math.isclose(exact_samples, sample_count, rel_tol=0.0, abs_tol=1.0e-7):
            raise ValueError("CLIP_SECONDS*SAMPLE_RATE_HZ must be an integer sample count")
        if "SAMPLE_COUNT" in params and _positive_integer(
                params["SAMPLE_COUNT"], owner="SAMPLE_COUNT") != sample_count:
            raise ValueError(
                f"SAMPLE_COUNT={params['SAMPLE_COUNT']} disagrees with "
                f"CLIP_SECONDS*SAMPLE_RATE_HZ={sample_count}")
        result["sample_count"] = sample_count
    if "TIME_BASE_HZ" in params and "TICKS_PER_FRAME" in params:
        time_base = _positive_integer(params["TIME_BASE_HZ"], owner="TIME_BASE_HZ")
        ticks_per_frame = _positive_integer(params["TICKS_PER_FRAME"], owner="TICKS_PER_FRAME")
        expected_ticks = time_base / frame_rate_hz
        if not math.isclose(expected_ticks, ticks_per_frame, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f"TICKS_PER_FRAME={ticks_per_frame} disagrees with "
                f"TIME_BASE_HZ/VIDEO_FPS={expected_ticks:g}")
        result["time_base_hz"] = time_base
        result["ticks_per_frame"] = ticks_per_frame
    if "TIME_BASE_HZ" in params and "SAMPLE_RATE_HZ" in params and "TICKS_PER_SAMPLE" in params:
        ticks_per_sample = _positive_integer(params["TICKS_PER_SAMPLE"], owner="TICKS_PER_SAMPLE")
        expected_ticks = _positive_integer(params["TIME_BASE_HZ"], owner="TIME_BASE_HZ") / sample_rate
        if ticks_per_sample <= 0 or not math.isclose(expected_ticks, ticks_per_sample,
                                                       rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f"TICKS_PER_SAMPLE={ticks_per_sample} disagrees with "
                f"TIME_BASE_HZ/SAMPLE_RATE_HZ={expected_ticks:g}")
        result["ticks_per_sample"] = ticks_per_sample
    return result


def frame_time_seconds(frame: int, params: dict | None) -> float:
    clock = validate_frame_clock(params)
    frame = require_frame(frame, clock["frame_count"])
    return frame / clock["frame_rate_hz"]


def _resample_route_samples(samples, frame_count: int) -> list[tuple[float, float]]:
    """Resample a bank route to the declared visual frame clock."""
    frame_count = _positive_frame_count(frame_count)
    source = list(samples)
    if len(source) < 2:
        raise ValueError("route samples must contain at least two points")
    if any(len(point) < 2 for point in source):
        raise ValueError("route samples must contain two-dimensional points")
    if len(source) == frame_count:
        return [(float(point[0]), float(point[1])) for point in source]
    scale = (len(source) - 1) / float(frame_count - 1)
    result = []
    for index in range(frame_count):
        position = index * scale
        lower = min(int(math.floor(position)), len(source) - 1)
        upper = min(lower + 1, len(source) - 1)
        fraction = position - lower
        result.append((
            float(source[lower][0])
            + (float(source[upper][0]) - float(source[lower][0])) * fraction,
            float(source[lower][1])
            + (float(source[upper][1]) - float(source[lower][1])) * fraction,
        ))
    return result


def _route_path_length_m(samples) -> float:
    return sum(math.dist(a, b) for a, b in zip(samples, samples[1:])) / 100.0


def bearing_deg(origin: Sequence[float], point: Sequence[float]) -> float:
    """世界系下 origin→point 的方位角(度),与 UE yaw 同一参照。"""
    return math.degrees(math.atan2(point[1] - origin[1], point[0] - origin[0]))


def relative_azimuth_deg(camera_xy, camera_yaw_deg: float, point_xy) -> float:
    """右为正的相对方位;与生产侧 side_of 的叉积符号一致。"""
    delta = bearing_deg(camera_xy, point_xy) - camera_yaw_deg
    return (delta + 180.0) % 360.0 - 180.0


def circular_gap_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def open_angle_gold_regions_disjoint(
        main_deg: float, gatea_deg: float, theta_half: float) -> bool:
    """Whether two angular golds have disjoint wide-credit regions.

    The argument is the active credit radius: THETA_FULL under strict scoring,
    THETA_HALF for legacy two-tier scoring. Two credited regions must not overlap.
    """
    return circular_gap_deg(main_deg, gatea_deg) > 2.0 * float(theta_half)


def open_angle_candidate_scores_zero(
        candidate_deg: float, truth_deg: float, theta_half: float) -> bool:
    """Whether the production angle scorer awards zero to one candidate.

    The argument is the active radius supplied by angle_credit_radius, so the
    search uses the same credited region as the selected scoring policy.
    """
    return circular_gap_deg(candidate_deg, truth_deg) > float(theta_half)


def yaw_interval_for_band(camera_xy, point_xy, band_lo: float,
                          band_hi: float) -> tuple[float, float]:
    """解出使 point 的相对方位落进 [band_lo, band_hi) 的 yaw 区间。

    这是本模块的关键:答案带是解出来的,不是枚举撞出来的。
    """
    bearing = bearing_deg(camera_xy, point_xy)
    return (bearing - band_hi, bearing - band_lo)


PRODUCTION_SCENE_SCHEMA = "avengine_qa_v3_production_scene_v1"
PRODUCTION_SCENE_DIR = "examples/qa/scenes"
PRODUCTION_SCENE_REQUIRED = (
    "schema", "scene_id", "scene_asset_id", "backend", "route_domain",
    "route_bank", "camera_base_request", "walkable_grid",
)
PRODUCTION_SCENE_RENDER_REQUIRED = ("floor_reference", "camera_clearance_table")


def resolve_production_scene_config(path, *, repo_root=None,
                                    historical_reproduction=False):
    """Resolve an explicit scene input, including a run's recorded copy.

    Resource identity and measured-floor checks validate the scene. Its storage
    directory cannot do that: the room scheduler writes a complete input copy
    outside the example catalog before invoking the same generator.
    The two keyword arguments remain accepted for historical callers.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"scene config does not exist: {path}")
    return path


def resolve_scene_resource_paths(config: dict, base_dir: str | Path) -> dict:
    """Resolve file references relative to their declaring configuration."""
    if not isinstance(config, dict):
        raise ValueError("scene config must be a JSON object")
    base = Path(base_dir).resolve()

    def resource(value):
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"scene resource path must be non-empty: {value!r}")
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else base / path).resolve())

    resolved = dict(config)
    for key in ("route_bank", "camera_base_request", "camera_clearance_table"):
        if resolved.get(key) is not None:
            resolved[key] = resource(resolved[key])
    for key in ("floor_reference", "walkable_grid"):
        value = resolved.get(key)
        if isinstance(value, dict):
            resolved[key] = dict(value, path=resource(value.get("path")))
        elif value is not None:
            resolved[key] = resource(value)
    value = resolved.get("line_of_sight_grid")
    if isinstance(value, dict):
        resolved["line_of_sight_grid"] = dict(value)
        for field in ("arrays", "metadata"):
            if value.get(field) is not None:
                resolved["line_of_sight_grid"][field] = resource(value[field])
    return resolved


def read_scene_config(path: str | Path) -> dict:
    """Read a scene without making its resource paths depend on the caller cwd."""
    source = resolve_production_scene_config(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    return resolve_scene_resource_paths(value, source.parent)


def require_production_scene_config(config, path) -> None:
    """Validate a complete scene declaration and its room identity.

    The default catalog has one named file per room. A run may record that
    declaration elsewhere without changing its meaning; measured-floor and
    resource checks in ``load_scene`` still apply to the recorded copy.
    """

    path = Path(path)
    missing = [k for k in PRODUCTION_SCENE_REQUIRED if k not in config]
    if missing:
        raise ValueError(f"{path.name}: production scene config missing {missing}")
    if config["schema"] != PRODUCTION_SCENE_SCHEMA:
        raise ValueError(
            f"{path.name}: schema is {config['schema']!r}, "
            f"expected {PRODUCTION_SCENE_SCHEMA!r}")
    if path.stem != str(config["scene_id"]):
        raise ValueError(
            f"{path.name}: file name and scene_id {config['scene_id']!r} disagree; "
            "one room, one file, named after it")
    if config.get("render"):
        absent = [k for k in PRODUCTION_SCENE_RENDER_REQUIRED if k not in config]
        if absent:
            raise ValueError(
                f"{path.name}: carries render facts but not {absent}; render "
                "facts are only checkable against a measured floor and a "
                "clearance table for this room")


def camera_half_fov(scene) -> float:
    """Return the physical camera-cone half angle from the scene calibration.

    VISUAL_FOV_MARGIN_DEG is a conservative design allowance for answer bands
    and safety screening. It is deliberately absent here: a declared
    visible/out_of_view query describes the actual camera cone.
    """
    try:
        hfov = float(scene.hfov_deg)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("scene must declare a numeric hfov_deg") from exc
    if not math.isfinite(hfov) or not (0.0 < hfov <= 180.0):
        raise ValueError(f"scene declares an invalid hfov_deg {hfov!r}")
    return hfov / 2.0


def effective_half_fov(scene, params) -> float:
    margin = float(params.get("VISUAL_FOV_MARGIN_DEG", 0.0))
    half_fov = camera_half_fov(scene)
    if not math.isfinite(margin) or margin < 0.0 or margin >= half_fov:
        raise ValueError("VISUAL_FOV_MARGIN_DEG must lie in [0, half_fov)")
    return half_fov - margin


ANSWER_DOMAINS = ("camera_cone", "full_circle", "rear_cone", "front_back")


AngleBand = tuple[float, float] | Arc


def answer_domain_arcs(domain: str, scene, params) -> tuple[AngleBand, ...]:
    """The arcs a declared answer domain covers, in engine-frame degrees.

    camera_cone keeps the historical numeric tuple.  The other domains carry
    one explicit Arc, because the whole circle and the rear cone are connected
    sets even when their canonical spelling crosses the seam.
    """
    if domain not in ANSWER_DOMAINS:
        raise ValueError(
            f"unknown answer_domain {domain!r}; expected one of {ANSWER_DOMAINS}")
    if domain == "front_back":
        half = effective_half_fov(scene, params)
        return (
            Arc(start_deg=-half, sweep_deg=2.0 * half),
            Arc(start_deg=180.0 - half, sweep_deg=2.0 * half),
        )
    if domain == "camera_cone":
        half = effective_half_fov(scene, params)
        return ((-half, half),)
    if domain == "full_circle":
        # Anchor the public sector order at the signed -180 boundary. Arc
        # normalizes that endpoint to +180, but the legacy serializer restores
        # the unambiguous [-180, 180] spelling.
        return (Arc(start_deg=180.0, sweep_deg=360.0),)
    half = effective_half_fov(scene, params)
    # In the canonical (-180, 180] representation the rear cone is split at
    # the seam, but on the circle it is one directed arc.
    return (Arc(start_deg=180.0 - half, sweep_deg=2.0 * half),)


def front_back_labels(profile) -> tuple[str, str]:
    """Return the two configured labels for the distinct front/back domains."""
    shape = profile.get("answer_shape") or {}
    if not isinstance(shape, dict):
        raise ValueError(f"{profile.get('id')}: answer_shape must be an object")
    labels = shape.get("binary_labels", shape.get("labels", ("front", "back")))
    if (not isinstance(labels, (list, tuple)) or len(labels) != 2
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != 2):
        raise ValueError(
            f"{profile.get('id')}: front/back binary labels must be two distinct non-empty strings")
    return str(labels[0]), str(labels[1])


def front_back_domain_arcs(profile, scene, params) -> tuple[Arc, Arc]:
    """Resolve distinct front and back answer regions from profile geometry."""
    shape = profile.get("answer_shape") or {}
    if not isinstance(shape, dict):
        raise ValueError(f"{profile.get('id')}: answer_shape must be an object")
    half = effective_half_fov(scene, params)
    front = band_to_arc(shape.get("front_arc", (-half, half)))
    back = band_to_arc(
        shape.get("back_arc", Arc(start_deg=180.0 - half, sweep_deg=2.0 * half)))
    if front.width_deg <= 0.0 or back.width_deg <= 0.0:
        raise ValueError(f"{profile.get('id')}: front/back regions must have positive width")
    if arc_overlap_width_deg(front, back) > 1.0e-9:
        raise ValueError(f"{profile.get('id')}: front/back regions overlap")
    return front, back


def band_to_arc(band: AngleBand | dict) -> Arc:
    """Resolve a band into the Arc that gives its direction and width.

    Tuple/list bands are the existing JSON boundary representation.  Their
    order is meaningful: (162, -162) is the short forward seam crossing, not
    the 324-degree complement.  New callers may keep an Arc all the way
    through solving when the signed sweep itself matters.
    """
    if isinstance(band, Arc):
        return band
    if isinstance(band, dict):
        return Arc.from_dict(band)
    try:
        lo, hi = (float(value) for value in band)
    except (TypeError, ValueError):
        raise ValueError(f"angle band must be an Arc or a two-value pair: {band!r}")
    return Arc.from_forward_bounds(lo, hi)


def band_to_bounds(band: AngleBand | dict) -> tuple[float, float]:
    """Serialize an Arc band at the old [lo, hi) boundary.

    The ordered pair preserves a seam crossing through hi < lo.  A full circle
    retains the historical [-180, 180] spelling.
    """
    if isinstance(band, (list, tuple)) and len(band) == 2:
        # Preserve already serialized endpoints, including an explicit -180.
        return float(band[0]), float(band[1])
    arc = band_to_arc(band)
    if arc.width_deg >= 360.0:
        return (-180.0, 180.0)
    return (arc.start_deg, arc.end_deg)


def band_width_deg(band: AngleBand | dict) -> float:
    """Return the circular width of a tuple or Arc band."""
    return band_to_arc(band).width_deg


def bands_equivalent(first: AngleBand | dict, second: AngleBand | dict,
                     *, tolerance: float = 1.0e-9) -> bool:
    """Compare two directed bands without collapsing a seam crossing."""
    first_arc, second_arc = band_to_arc(first), band_to_arc(second)
    if first_arc.width_deg >= 360.0 or second_arc.width_deg >= 360.0:
        return (first_arc.width_deg >= 360.0
                and second_arc.width_deg >= 360.0)
    return (circular_gap_deg(first_arc.start_deg, second_arc.start_deg)
            <= tolerance
            and abs(first_arc.sweep_deg - second_arc.sweep_deg) <= tolerance)


def _angle_in_band(value: float, band: AngleBand | dict) -> bool:
    """Containment for a directed band, including one crossing the seam."""
    value = float(value)
    if not math.isfinite(value):
        return False
    arc = band_to_arc(band)
    if not arc.contains(value):
        return False
    # Preserve the historical half-open upper edge for non-full bands.
    if arc.width_deg >= 360.0:
        return True
    return not math.isclose(circular_gap_deg(value, arc.end_deg), 0.0,
                            abs_tol=1e-9)

def band_contains(value: float, band: AngleBand | dict) -> bool:
    """Public containment helper for generated facts and solver consumers."""
    return _angle_in_band(value, band)


def query_bound_for_domain(domain: str | None, scene, params) -> float:
    """Return the maximum query bearing allowed by an answer domain.

    ``None`` and ``camera_cone`` retain the old trusted visual cone.  The two
    F2 domains deliberately allow the complete azimuth circle; their answer
    bands still decide whether a candidate is front, side, or behind.  No room
    geometry or hand-written rear angle enters this mapping.
    """
    if domain is None or domain == "camera_cone":
        return effective_half_fov(scene, params)
    if domain in ("full_circle", "rear_cone", "front_back"):
        return 180.0
    raise ValueError(
        f"unknown query answer_domain {domain!r}; expected one of {ANSWER_DOMAINS}")


def resolve_query_visibility(
    domain: str | None,
    *,
    query_visibility: str | None = None,
    query_requires_visibility: bool | None = None,
) -> str:
    """Resolve tri-state query visibility while accepting the legacy bool."""
    if query_visibility is not None:
        if not isinstance(query_visibility, str):
            raise ValueError(
                "query_visibility must be one of "
                f"{QUERY_VISIBILITY_VALUES}")
        state = query_visibility.strip().lower()
        if state not in QUERY_VISIBILITY_VALUES:
            raise ValueError(
                "query_visibility must be one of "
                f"{QUERY_VISIBILITY_VALUES}, got {query_visibility!r}")
        if query_requires_visibility is not None:
            if not isinstance(query_requires_visibility, bool):
                raise ValueError("query_requires_visibility must be a boolean")
            if query_requires_visibility and state != "visible":
                raise ValueError(
                    "query_requires_visibility=True conflicts with "
                    f"query_visibility={state!r}")
            if not query_requires_visibility and state == "visible":
                raise ValueError(
                    "query_requires_visibility=False conflicts with "
                    "query_visibility='visible'")
        return state
    if query_requires_visibility is None:
        return "visible" if domain in (None, "camera_cone") else "any"
    if not isinstance(query_requires_visibility, bool):
        raise ValueError("query_requires_visibility must be a boolean")
    return "visible" if query_requires_visibility else "any"


def query_visibility_matches(
    azimuth_deg: float, visibility: str, half_fov_deg: float,
) -> bool:
    """Return whether one emitter bearing satisfies a camera-cone policy."""
    if visibility not in QUERY_VISIBILITY_VALUES:
        raise ValueError(
            f"query visibility must be one of {QUERY_VISIBILITY_VALUES}")
    inside = abs(float(azimuth_deg)) <= float(half_fov_deg)
    if visibility == "visible":
        return inside
    if visibility == "out_of_view":
        return not inside
    return True


def resolve_query_geometry(
    scene,
    params,
    *,
    query_domain: str | None = None,
    answer_domain: str | None = None,
    query_bound_deg: float | None = None,
    query_requires_visibility: bool | None = None,
    query_visibility: str | None = None,
) -> tuple[float, bool]:
    """Resolve the declared query domain into a bearing bound and visibility rule.

    ``answer_domain`` is accepted as a profile-shaped alias for callers that
    forward the declaration directly.  Explicit ``query_bound_deg`` and
    ``query_requires_visibility`` are useful for a profile whose answer space
    and visual requirement are intentionally different.  Omitting everything
    keeps the historical camera-cone, visible-query behavior.
    """
    if (query_domain is not None and answer_domain is not None
            and query_domain != answer_domain):
        raise ValueError(
            f"query_domain {query_domain!r} conflicts with answer_domain "
            f"{answer_domain!r}")
    domain = query_domain if query_domain is not None else answer_domain
    default_bound = query_bound_for_domain(domain, scene, params)
    bound = default_bound if query_bound_deg is None else float(query_bound_deg)
    if not math.isfinite(bound) or not 0.0 <= bound <= 180.0:
        raise ValueError("query_bound_deg must be finite and lie in [0, 180]")
    visibility = resolve_query_visibility(
        domain,
        query_visibility=query_visibility,
        query_requires_visibility=query_requires_visibility,
    )
    return bound, visibility == "visible"


def derive_answer_arcs(profile, scene, params) -> list[Arc]:
    """Derive equal-width directed Arc bands for a declared answer domain.

    The rear cone is connected on the circle even though its canonical numeric
    spelling crosses the seam.  It therefore enters this calculation as one
    Arc and is divided into one equal sweep per band.  A seam band such as
    162 -> 198 remains an Arc; only the legacy serializer turns it into the
    ordered pair (162, -162).
    """
    domain = profile.get("answer_domain")
    if domain is None:
        raise ValueError(f"{profile.get('id')}: no answer_domain to derive from")
    shape = profile.get("answer_shape") or {}
    if not isinstance(shape, dict):
        raise ValueError(f"{profile.get('id')}: answer_shape must be an object")
    if domain == "front_back":
        return list(front_back_domain_arcs(profile, scene, params))
    count = shape.get("equal_bands", 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(
            f"{profile.get('id')}: answer_shape.equal_bands must be >= 1 to "
            f"derive bands for domain {domain!r}")
    domains = [band_to_arc(region) for region in answer_domain_arcs(
        domain, scene, params)]
    if len(domains) != 1:
        raise ValueError(
            f"{domain!r} must resolve to one connected Arc before equal-width "
            "bands can be derived")
    domain_arc = domains[0]
    step = domain_arc.sweep_deg / count
    bands = [Arc(start_deg=domain_arc.start_deg + i * step,
                 sweep_deg=step) for i in range(count)]
    indices = shape.get("band_indices")
    if indices is not None:
        if (not isinstance(indices, list) or not indices
                or any(isinstance(i, bool) or not isinstance(i, int)
                       or not 0 <= i < len(bands) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError(f"{profile.get('id')}: band_indices must select distinct bands")
        bands = [bands[i] for i in indices]
    return bands


def derive_answer_bands(profile, scene, params) -> list[AngleBand]:
    """Derive bands with the existing numeric boundary where it is stable.

    camera_cone and full_circle retain numeric ordered pairs for current
    generator/extended callers.  rear_cone returns Arc objects because its
    seam band cannot be represented safely by an ordinary increasing pair.
    Use band_to_bounds at a JSON/publication boundary for the Arc form.
    """
    arcs = derive_answer_arcs(profile, scene, params)
    domain = profile.get("answer_domain")
    if domain in ("rear_cone", "front_back"):
        return arcs
    if domain == "full_circle":
        count = len(arcs)
        step = 360.0 / count
        all_bands = [(-180.0 + i * step, -180.0 + (i + 1) * step)
                     for i in range(count)]
        indices = (profile.get("answer_shape") or {}).get("band_indices")
        return (all_bands if indices is None
                else [all_bands[i] for i in indices])
    return [band_to_bounds(arc) for arc in arcs]

def audit_answer_bands(scene, params, profiles) -> dict:
    """Reconcile every profile's answer bands against this scene's camera.

    A profile that declares answer_domain is derived from the room's camera and
    compared as directed Arcs.  A seam pair and its Arc spelling therefore
    describe the same wedge without silently becoming its complement.  Legacy
    numeric profiles retain their old audit regime.
    """
    half = effective_half_fov(scene, params)
    derived, legacy = {}, {}
    for profile in profiles:
        pid = profile.get("id")
        declared = profile.get("answer_bands_deg")
        if profile.get("answer_domain") is not None:
            bands = derive_answer_arcs(profile, scene, params)
            if declared is not None:
                same = (len(declared) == len(bands) and all(
                    bands_equivalent(declared_band, derived_band)
                    for declared_band, derived_band
                    in zip(declared, bands, strict=True)))
                if not same:
                    declared_view = [list(band_to_bounds(b)) for b in declared]
                    derived_view = [list(band_to_bounds(b)) for b in bands]
                    raise ValueError(
                        f"{pid}: answer_bands_deg {declared_view} "
                        f"disagrees with what domain "
                        f"{profile['answer_domain']!r} derives for this scene, "
                        f"{derived_view}. Drop the hand-written "
                        "table -- the domain is the input now")
            output_bands = derive_answer_bands(profile, scene, params)
            derived[pid] = [list(band_to_bounds(band))
                            for band in output_bands]
            continue
        if not declared:
            continue
        camera_arc = Arc.from_bounds(-half, half)
        declared_width = 0.0
        unreachable = 0.0
        for band in declared:
            arc = band_to_arc(band)
            width = band_width_deg(arc)
            declared_width += width
            reachable = arc_overlap_width_deg(arc, camera_arc)
            unreachable += max(0.0, width - reachable)
        legacy[pid] = {
            "declared_total_deg": round(declared_width, 6),
            "unreachable_deg": round(unreachable, 6),
        }
    return {"usable_half_angle_deg": half,
            "derived_from_domain": derived,
            "legacy_declared_degrees": legacy}

def materialize_answer_domains(scene, params, profiles):
    """Return executable profiles and the audit of their declared answer domains."""
    audit = audit_answer_bands(scene, params, profiles)
    effective = []
    for profile in profiles:
        item = dict(profile)
        bands = audit["derived_from_domain"].get(profile.get("id"))
        if bands is not None:
            item["answer_bands_deg"] = bands
        effective.append(item)
    return effective, audit


def interior_answer_band(band_lo: float, band_hi: float, params):
    margin = float(params.get("ANSWER_BAND_INTERIOR_MARGIN_DEG", 0.0))
    lo, hi = float(band_lo), float(band_hi)
    width = hi - lo if hi >= lo else (hi - lo) % 360.0
    if (
        not math.isfinite(margin) or margin < 0.0
        or not math.isfinite(width) or 2.0 * margin >= width
    ):
        raise ValueError("ANSWER_BAND_INTERIOR_MARGIN_DEG does not fit the band")
    # Keep a directed seam-crossing band directed; yaw_interval_for_band and
    # _angle_in_band both understand the resulting ordered pair.
    return lo + margin, hi - margin


@dataclass
class Route:
    route_id: str
    samples_xy: list[tuple[float, float]]      # samples in the declared visual clock
    implied_speed_mps: float
    # None for a bank route; a synthesized route carries its design record so
    # the fact can say where the trajectory came from (see route_synthesis.py).
    provenance: dict | None = None
    frame_rate_hz: float = DEFAULT_VIDEO_FPS
    duration_seconds: float | None = None

    def at(self, frame: int) -> tuple[float, float]:
        if not self.samples_xy:
            raise ValueError(f"route {self.route_id!r} has no samples")
        return self.samples_xy[max(0, min(len(self.samples_xy) - 1, int(frame)))]

    @property
    def source(self) -> str:
        return "bank" if self.provenance is None else str(self.provenance.get("source"))

    @property
    def source_record(self) -> dict:
        if self.provenance is None:
            return {"source": "bank", "route_id": self.route_id}
        return dict(self.provenance, route_id=self.route_id)

    def shifted(self, idle_frames: int) -> "Route":
        """静→走:前 idle_frames 帧停在起点,其后沿用原速度(平移式)。"""
        if idle_frames <= 0:
            return self
        frame_count = len(self.samples_xy)
        if not 1 <= idle_frames <= frame_count - 2:
            raise ValueError(
                f"idle_frames must be in [1, {frame_count - 2}], got {idle_frames}")
        shifted = ([self.samples_xy[0]] * idle_frames
                   + self.samples_xy[:frame_count - idle_frames])
        return Route(
            f"{self.route_id}+idle{idle_frames}", shifted,
            self.implied_speed_mps, provenance=self.provenance,
            frame_rate_hz=self.frame_rate_hz,
            duration_seconds=self.duration_seconds)

    def paused(self, start_frame: int, end_frame: int) -> "Route":
        """Freeze one inclusive window, then resume the delayed route."""
        frame_count = len(self.samples_xy)
        if not 0 <= start_frame <= end_frame < frame_count:
            raise ValueError(
                f"invalid pause window {start_frame}..{end_frame}")
        delay = end_frame - start_frame
        samples = []
        for frame in range(frame_count):
            if frame < start_frame:
                samples.append(self.samples_xy[frame])
            elif frame <= end_frame:
                samples.append(self.samples_xy[start_frame])
            else:
                samples.append(self.samples_xy[frame - delay])
        return Route(
            f"{self.route_id}+pause{start_frame}-{end_frame}",
            samples, self.implied_speed_mps, provenance=self.provenance,
            frame_rate_hz=self.frame_rate_hz,
            duration_seconds=self.duration_seconds)

    @property
    def displacement_cm(self) -> float:
        return math.dist(self.samples_xy[0], self.samples_xy[-1])


@dataclass
class SceneInputs:
    scene_id: str
    backend: str
    routes: list[Route]
    stand_points: list[tuple[float, float]]
    camera_points: list[tuple[float, float]]
    camera_height_m: float
    hfov_deg: float
    line_of_sight: Callable[[Sequence[float], Sequence[float]], bool] | None = None
    provenance: dict = field(default_factory=dict)
    render_config: dict = field(default_factory=dict)
    clearance: CameraClearanceTable | None = None
    walkable: WalkableGrid | None = None
    floor: FloorReference | None = None

    @property
    def line_of_sight_screened(self) -> bool:
        return self.line_of_sight is not None

    @property
    def route_synthesis_available(self) -> bool:
        return self.walkable is not None

    @property
    def camera_clearance_screened(self) -> bool:
        return self.clearance is not None


# 路线库适配器按**库自己的 schema** 分派,不按房间 ID。每个后端声明自己的
# 水平面约定:UE 的地平面是 (x, y)、单位厘米;habitat 的是 (x, z)、单位米。
# 这是后端坐标约定,不是房间特例。
BANK_ADAPTERS: dict[str, str] = {
    "avengine_apartment_route_bank_v1": "ue_route_bank",
    "avengine_room_trajectory_bank_v2": "habitat_trajectory_bank",
}


def _routes_from_ue_route_bank(
        bank: dict, limit: int | None, *, frame_count: int,
        frame_rate_hz: float) -> list[Route]:
    routes: list[Route] = []
    for entry in bank.get("routes", []):
        samples = entry.get("samples_ue_cm")
        if not samples or len(samples) < 2:
            continue
        source_fps = _positive_fps(entry.get("frame_rate_hz", DEFAULT_VIDEO_FPS))
        source_duration = float(entry.get(
            "duration_seconds", bank.get("clip_seconds", len(samples) / source_fps)))
        if not math.isfinite(source_duration) or source_duration <= 0.0:
            continue
        source_xy = [(float(point[0]), float(point[1])) for point in samples]
        adapted = _resample_route_samples(source_xy, frame_count)
        source_speed = float(entry.get("implied_speed_mps", 0.0))
        if not math.isfinite(source_speed) or source_speed < 0.0:
            continue
        target_duration = frame_count / frame_rate_hz
        target_speed = (source_speed * source_duration / target_duration
                        if target_duration > 0.0 else source_speed)
        routes.append(Route(
            str(entry["route_id"]), adapted, target_speed,
            provenance={
                "source": "bank",
                "clock_adapted": len(samples) != frame_count,
                "source_frame_count": len(samples),
                "target_frame_count": frame_count,
                "source_frame_rate_hz": source_fps,
                "target_frame_rate_hz": frame_rate_hz,
                "source_duration_seconds": source_duration,
                "target_duration_seconds": target_duration,
                "source_camera_points": [
                    list(source_xy[0]), list(source_xy[-1]),
                    list(source_xy[len(source_xy) // 2])],
            },
            frame_rate_hz=frame_rate_hz,
            duration_seconds=target_duration))
        if limit and len(routes) >= limit:
            break
    return routes


def _routes_from_habitat_trajectory_bank(
        bank: dict, limit: int | None, *, frame_count: int,
        frame_rate_hz: float) -> list[Route]:
    """habitat 逐槽位路径 → 统一 Route(米→厘米,取 (x, z) 水平面)。

    每个 episode 的每个槽位都是一条独立可用的路线:采样器只关心"一条
    75 帧的可行轨迹",谁跟谁配对由题型求解阶段重新决定。
    """
    routes: list[Route] = []
    for episode in bank.get("episodes", []):
        paths = episode.get("source_center_paths_m") or {}
        for slot, path in sorted(paths.items()):
            if not path or len(path) < 2:
                continue
            source_xy_m = [
                (float(point[0]), float(point[2] if len(point) >= 3 else point[1]))
                for point in path
            ]
            adapted_m = _resample_route_samples(source_xy_m, frame_count)
            xy = [(float(point[0]) * 100.0, float(point[1]) * 100.0)
                  for point in adapted_m]
            source_duration = float(episode.get(
                "duration_seconds", bank.get(
                    "seconds_per_episode", len(path) / DEFAULT_VIDEO_FPS)))
            if not math.isfinite(source_duration) or source_duration <= 0.0:
                continue
            span = _route_path_length_m(xy)
            routes.append(Route(
                f"{episode['episode_id']}:{slot}", xy,
                span / (frame_count / frame_rate_hz),
                provenance={
                    "source": "bank",
                    "clock_adapted": len(path) != frame_count,
                    "source_frame_count": len(path),
                    "target_frame_count": frame_count,
                    "source_frame_rate_hz": DEFAULT_VIDEO_FPS,
                    "target_frame_rate_hz": frame_rate_hz,
                    "source_duration_seconds": source_duration,
                    "target_duration_seconds": frame_count / frame_rate_hz,
                    "source_camera_points": [
                        [100.0 * source_xy_m[0][0], 100.0 * source_xy_m[0][1]],
                        [100.0 * source_xy_m[-1][0], 100.0 * source_xy_m[-1][1]],
                        [100.0 * source_xy_m[len(source_xy_m) // 2][0],
                         100.0 * source_xy_m[len(source_xy_m) // 2][1]]],
                },
                frame_rate_hz=frame_rate_hz,
                duration_seconds=frame_count / frame_rate_hz))
            if limit and len(routes) >= limit:
                return routes
    return routes


def routes_from_bank(
        bank: dict, limit: int | None = None, *, frame_count: int = FRAME_COUNT,
        frame_rate_hz: float = DEFAULT_VIDEO_FPS) -> list[Route]:
    frame_count = _positive_frame_count(frame_count)
    frame_rate_hz = _positive_fps(frame_rate_hz)
    schema = str(bank.get("schema", ""))
    adapter = BANK_ADAPTERS.get(schema)
    if adapter is None:
        raise ValueError(
            f"no route-bank adapter for schema {schema!r}; add one adapter "
            "keyed by the bank schema (never by a room id)")
    if adapter == "ue_route_bank":
        return _routes_from_ue_route_bank(
            bank, limit, frame_count=frame_count, frame_rate_hz=frame_rate_hz)
    return _routes_from_habitat_trajectory_bank(
        bank, limit, frame_count=frame_count, frame_rate_hz=frame_rate_hz)


def adapt_scene_clock(scene: SceneInputs, params: dict) -> SceneInputs:
    """Adapt route samples to the explicitly declared frame/sample clock.

    Route-bank geometry is resampled over the whole declared clip.  The speed
    metadata is scaled with the source and target durations, while all solver
    geometry and later facts consume the resulting samples directly.
    """
    clock = validate_frame_clock(params, require_clip_seconds=True)
    target_count = clock["frame_count"]
    target_rate = clock["frame_rate_hz"]
    target_duration = clock["clip_seconds"]
    if not scene.routes:
        raise ValueError(f"{scene.scene_id}: cannot adapt an empty route set")
    adapted_routes = []
    for route in scene.routes:
        source_count = len(route.samples_xy)
        if source_count < 2:
            raise ValueError(f"route {route.route_id!r} has fewer than two samples")
        samples = _resample_route_samples(route.samples_xy, target_count)
        source_rate = _positive_fps(getattr(route, "frame_rate_hz", DEFAULT_VIDEO_FPS))
        source_duration = getattr(route, "duration_seconds", None)
        if source_duration is None:
            source_duration = source_count / source_rate
        source_duration = float(source_duration)
        if not math.isfinite(source_duration) or source_duration <= 0.0:
            raise ValueError(f"route {route.route_id!r} has invalid duration")
        speed = float(route.implied_speed_mps)
        if not math.isfinite(speed) or speed < 0.0:
            raise ValueError(f"route {route.route_id!r} has invalid implied speed")
        target_speed = speed * source_duration / target_duration
        provenance = dict(route.provenance or {"source": "bank"})
        provenance.update({
            "clock_adapted": source_count != target_count,
            "source_frame_count": source_count,
            "target_frame_count": target_count,
            "source_frame_rate_hz": source_rate,
            "target_frame_rate_hz": target_rate,
            "source_duration_seconds": source_duration,
            "target_duration_seconds": target_duration,
        })
        adapted_routes.append(replace(
            route, samples_xy=samples, implied_speed_mps=target_speed,
            provenance=provenance, frame_rate_hz=target_rate,
            duration_seconds=target_duration))
    provenance = dict(scene.provenance)
    provenance["frame_clock"] = clock
    provenance["route_clock_adaptation"] = {
        "source_frame_counts": sorted({len(route.samples_xy) for route in scene.routes}),
        "target_frame_count": target_count,
        "target_frame_rate_hz": target_rate,
        "target_duration_seconds": target_duration,
    }
    return replace(scene, routes=adapted_routes, provenance=provenance)


def ensure_scene_clock(scene: SceneInputs, params: dict) -> SceneInputs:
    """Return a scene whose route samples match the params clock."""
    clock = validate_frame_clock(params)
    if scene.routes and all(
            len(route.samples_xy) == clock["frame_count"]
            and math.isclose(route.frame_rate_hz, clock["frame_rate_hz"],
                             rel_tol=0.0, abs_tol=1.0e-9)
            and (route.duration_seconds is None or math.isclose(
                route.duration_seconds, clock["clip_seconds"],
                rel_tol=0.0, abs_tol=1.0e-9))
            for route in scene.routes):
        return scene
    return adapt_scene_clock(scene, dict(params, CLIP_SECONDS=clock["clip_seconds"]))


def scene_frame_count(scene: SceneInputs, params: dict | None = None) -> int:
    """Resolve and validate one frame count for all routes in a scene."""
    default = (len(scene.routes[0].samples_xy)
               if getattr(scene, "routes", None) else FRAME_COUNT)
    target = frame_count_from_params(params, default=default)
    lengths = {len(route.samples_xy) for route in scene.routes}
    if lengths and lengths != {target}:
        raise ValueError(
            f"{scene.scene_id}: route sample counts {sorted(lengths)} do not "
            f"match requested FRAME_COUNT={target}; call ensure_scene_clock")
    return target


def _require_frame(frame, frame_count: int, *, name="frame") -> int:
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise ValueError(f"{name} must be an integer frame index")
    if not 0 <= frame < frame_count:
        raise ValueError(f"{name} {frame} outside 0..{frame_count - 1}")
    return int(frame)


def require_frame(frame, frame_count: int, *, name="frame") -> int:
    """Public frame-index validation for sibling generators."""
    return _require_frame(frame, frame_count, name=name)


def line_of_sight_from_feasible_grid(config: dict):
    """Build a conservative 2D LOS proxy from a declared feasible raster.

    The sampler's normalized horizontal plane is UE ``(x,y)`` centimetres.
    The supported grid contract was authored in Habitat ``(x,z)`` metres;
    route-bank adaptation already maps those axes to the normalized plane.
    This is a pre-render obstacle screen only.  It cannot see height-dependent
    occlusion and never replaces native pixel truth.
    """

    required = ("metadata", "metadata_key", "arrays", "mask_key", "coordinate_contract")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"line_of_sight_grid missing keys: {missing}")
    if config["coordinate_contract"] != "habitat_xz_m_to_ue_xy_cm_v1":
        raise ValueError("unsupported line_of_sight_grid coordinate contract")
    metadata_doc = json.loads(Path(config["metadata"]).read_text())
    metadata = metadata_doc.get(config["metadata_key"])
    if not isinstance(metadata, dict) or metadata.get("schema") != "avengine_room_feasible_region_v1":
        raise ValueError("line_of_sight_grid metadata has the wrong schema")
    with np.load(Path(config["arrays"])) as arrays:
        if config["mask_key"] not in arrays.files:
            raise ValueError("line_of_sight_grid mask key is absent")
        mask = np.asarray(arrays[config["mask_key"]], dtype=bool).copy()
    expected_shape = tuple(int(value) for value in metadata["mask_shape_hw"])
    if mask.shape != expected_shape or mask.ndim != 2:
        raise ValueError("line_of_sight_grid mask shape differs from metadata")
    bounds = metadata["bounds_m"]
    minimum_x_m = float(bounds[0][0])
    minimum_z_m = float(bounds[0][2])
    pixel_x_m = float(metadata["pixel_size_x_m"])
    pixel_z_m = float(metadata["pixel_size_z_m"])
    if pixel_x_m <= 0.0 or pixel_z_m <= 0.0:
        raise ValueError("line_of_sight_grid pixel sizes must be positive")

    def grid_coordinates(point):
        column = (float(point[0]) / 100.0 - minimum_x_m) / pixel_x_m
        row = (float(point[1]) / 100.0 - minimum_z_m) / pixel_z_m
        return row, column

    def line_of_sight(origin, target):
        row0, column0 = grid_coordinates(origin)
        row1, column1 = grid_coordinates(target)
        span = max(abs(row1 - row0), abs(column1 - column0))
        # Quarter-pixel sampling is conservative around diagonal obstacle
        # corners without inventing a room-specific dilation radius.
        sample_count = max(2, int(math.ceil(span * 4.0)) + 1)
        rows = np.floor(np.linspace(row0, row1, sample_count)).astype(np.int64)
        columns = np.floor(np.linspace(column0, column1, sample_count)).astype(np.int64)
        inside = (
            (rows >= 0) & (rows < mask.shape[0])
            & (columns >= 0) & (columns < mask.shape[1])
        )
        return bool(np.all(inside) and np.all(mask[rows, columns]))

    return line_of_sight


def load_scene(config: dict, *, route_limit: int | None = None,
               frame_count: int = FRAME_COUNT,
               frame_rate_hz: float = DEFAULT_VIDEO_FPS) -> SceneInputs:
    """从场景配置载入场景无关输入。

    config 只允许指向场景自身的产物(导航路线库、相机基准请求),不允许
    携带逐点坐标或手填路线;缺项即拒绝载入,而不是回退到某个房间的默认。
    """
    required = ("scene_id", "backend", "route_bank", "camera_base_request")
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"scene config missing keys: {missing}")
    frame_count = _positive_frame_count(frame_count)
    frame_rate_hz = _positive_fps(frame_rate_hz)
    bank = json.loads(Path(config["route_bank"]).read_text())
    routes = routes_from_bank(
        bank, route_limit, frame_count=frame_count, frame_rate_hz=frame_rate_hz)
    if not routes:
        raise ValueError(
            f"{config['scene_id']}: route bank yielded no usable "
            f"{frame_count}-frame routes")
    # 可站点与相机候选都取导航可达点(路线端点与路点),去重后使用:
    # 这些点由场景导航系统背书,不是手填坐标。
    points: set[tuple[float, float]] = set()
    for route in routes:
        source_points = (route.provenance or {}).get("source_camera_points")
        if isinstance(source_points, list) and len(source_points) == 3:
            points.update(_round_xy(point) for point in source_points)
        else:
            points.add(_round_xy(route.samples_xy[0]))
            points.add(_round_xy(route.samples_xy[-1]))
            points.add(_round_xy(route.samples_xy[frame_count // 2]))
    ordered = sorted(points)
    base_request = json.loads(Path(config["camera_base_request"]).read_text())
    height = float(config.get("camera_height_m")
                   or _listener_height_m(base_request))
    hfov = scene_hfov_deg(config, base_request=base_request)
    render_config = config.get("render") or {}
    if not isinstance(render_config, dict):
        raise ValueError(f"{config['scene_id']}: render must be an object")
    floor = None
    floor_config = config.get("floor_reference")
    if floor_config is not None:
        # 地板参照是房间的属性:引擎里量出来的地板高度,必须是这个房间的。
        try:
            floor = floor_reference_from_config(floor_config)
        except FloorReferenceError as error:
            raise ValueError(f"{config['scene_id']}: {error}") from error
        if floor.scene_id != str(config["scene_id"]):
            raise ValueError(
                f"{config['scene_id']}: floor reference belongs to {floor.scene_id!r}")
    if render_config:
        # 要渲染的房间必须先量过地板(2026-09-03 Apartment 的 ground_z_ue_cm 手写成 0,
        # 实际地板在 +27 cm,所有渲染的狗都陷进地板)。配置里写的 ground_z_ue_cm 必须
        # 等于实测值,不然拒绝载入,而不是带着错的地面继续设计。
        if floor is None:
            raise ValueError(
                f"{config['scene_id']}: render facts require a measured floor_reference; "
                f"render.ground_z_ue_cm may not be a hand-written constant")
        declared = render_config.get("ground_z_ue_cm")
        if declared is None or not floor.matches(float(declared)):
            raise ValueError(
                f"{config['scene_id']}: render.ground_z_ue_cm={declared} disagrees with the "
                f"measured floor {floor.ground_z_ue_cm} cm (tolerance "
                f"{FLOOR_MATCH_TOLERANCE_CM} cm) in {floor.root}")
    los_config = config.get("line_of_sight_grid")
    line_of_sight = None if los_config is None else line_of_sight_from_feasible_grid(los_config)
    clearance = None
    table_path = config.get("camera_clearance_table")
    if table_path is not None:
        # 机位净空表是房间的属性:必须是这个房间、覆盖求解器会抽到的每个相机点、
        # 含场景相机高度。缺任何一项都拒绝载入,不猜。
        clearance = CameraClearanceTable.load(table_path)
        if clearance.scene_id != str(config["scene_id"]):
            raise ValueError(
                f"{config['scene_id']}: camera clearance table belongs to "
                f"{clearance.scene_id!r}")
        missing = clearance.missing_points(ordered)
        if missing:
            raise ValueError(
                f"{config['scene_id']}: camera clearance table does not cover "
                f"{len(missing)} of {len(ordered)} camera points (first: "
                f"{missing[:3]})")
        if not clearance.has_height(height):
            raise ValueError(
                f"{config['scene_id']}: camera clearance table lacks the scene "
                f"camera height {height} m (has {clearance.heights_m.tolist()})")
        table_ground = clearance.ground_z_ue_cm
        if table_ground is not None and floor is not None and not floor.matches(table_ground):
            # 表按绝对 z 渲:地板量错时渲的表,相机离地高度就是错的,必须重渲。
            raise ValueError(
                f"{config['scene_id']}: camera clearance table was rendered from ground z "
                f"{table_ground} cm but the measured floor is {floor.ground_z_ue_cm} cm; "
                f"re-render the table at the measured floor")
    walkable = None
    grid_config = config.get("walkable_grid")
    if grid_config is not None:
        # 可走栅格也是房间的属性:必须是这个房间的,且求解器会抽到的每个相机点
        # 都得落在可走格内(相机点本来就是导航可达点)。
        walkable = grid_from_config(grid_config)
        if walkable.scene_id != str(config["scene_id"]):
            raise ValueError(
                f"{config['scene_id']}: walkable grid belongs to {walkable.scene_id!r}")
        outside = [xy for xy in ordered if not walkable.is_walkable(xy)]
        if outside:
            raise ValueError(
                f"{config['scene_id']}: walkable grid does not contain {len(outside)} of "
                f"{len(ordered)} navigable points (first: {outside[:3]})")
    return SceneInputs(
        scene_id=str(config["scene_id"]), backend=str(config["backend"]),
        routes=routes, stand_points=ordered, camera_points=ordered,
        camera_height_m=height, hfov_deg=hfov,
        line_of_sight=line_of_sight,
        provenance={"route_bank": config["route_bank"],
                    "route_bank_schema": bank.get("schema"),
                    "route_bank_source": bank.get("source"),
                    "bank_adapter": BANK_ADAPTERS.get(str(bank.get("schema"))),
                    "camera_base_request": config["camera_base_request"],
                    "routes_loaded": len(routes),
                    "line_of_sight_grid": los_config,
                    "camera_clearance_table": (
                        clearance.identity if clearance is not None else None),
                    "walkable_grid": (
                        walkable.identity if walkable is not None else None),
                    "floor_reference": (
                        floor.identity if floor is not None else None),
                    "navigable_points": len(ordered),
                    "frame_clock": {
                        "frame_count": frame_count,
                        "frame_rate_hz": frame_rate_hz,
                        "duration_seconds": frame_count / frame_rate_hz,
                    }},
        render_config=dict(render_config),
        clearance=clearance,
        walkable=walkable,
        floor=floor,
    )


def _round_xy(xy) -> tuple[float, float]:
    return (round(float(xy[0]), 1), round(float(xy[1]), 1))


def _listener_height_m(request: dict) -> float:
    rig = request["primary_camera_rig"]["world_from_rig"]["translation_m"]
    listener = request["listener"]["rig_from_listener"]["translation_m"]
    return float(rig[1]) + float(listener[1])


def scene_hfov_deg(config: dict, *, base_request: dict | None = None) -> float:
    """场景相机的水平视场角:场景配置的 hfov_deg 优先,否则读相机基准请求。

    这是所有工具(求解器、预检、净空表)取相机视场角的唯一入口,任何地方
    都不得再把 105 写死。缺失或不合理即拒绝,不回退到默认值。
    """
    value = config.get("hfov_deg")
    if value is None:
        if base_request is None:
            path = config.get("camera_base_request")
            if path is None:
                raise ValueError("scene config declares neither hfov_deg nor "
                                 "camera_base_request")
            base_request = json.loads(Path(path).read_text())
        value = _hfov_deg(base_request)
    hfov = float(value)
    if not math.isfinite(hfov) or not (0.0 < hfov < 180.0):
        raise ValueError(f"{config.get('scene_id')}: implausible hfov {hfov}")
    return hfov


def _hfov_deg(request: dict) -> float:
    """相机水平视场:从 M1 请求的标定块读,字段名按仓库既有约定。"""
    rig = request.get("primary_camera_rig", {})
    for holder in (rig.get("shared_calibration") or {}, rig):
        for key in ("hfov_degrees", "horizontal_fov_deg",
                    "horizontal_field_of_view_deg"):
            if key in holder:
                return float(holder[key])
    for sensor in rig.get("sensors") or []:
        for key in ("hfov_degrees", "horizontal_fov_deg"):
            if key in sensor:
                return float(sensor[key])
    raise ValueError("camera base request declares no horizontal FOV")


@dataclass
class Rejection:
    reason: str
    detail: str = ""


@dataclass
class PointPlan:
    scene_id: str
    profile_id: str
    camera_xy: tuple[float, float]
    camera_ue_yaw_deg: float
    target_route: Route          # 已套用静→走平移的路线(求解器视角)
    base_route: Route            # 未平移的原路线(创作时间线用)
    other_route: Route
    idle_frames: int
    anchor_frame: int
    query_frame: int
    answer_cell: dict
    checks: dict
    # None means "scene camera height"; set when the clearance table sent the
    # camera to a fallback height at this pose.
    camera_height_m: float | None = None
    camera_clearance: dict | None = None


class RejectionLedger:
    """拒绝台账 + 搜索成本计数。

    拒绝是正常结果,但必须说得出为什么;而"配额填满了"也不等于"通过率
    100%" —— 所以这里同时记录评估过多少个候选组合、耗尽预算多少次,
    让分母可见。
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.first_details: dict[str, str] = {}
        self.combinations_evaluated = 0     # 抽过多少个 相机×路线×转折帧
        self.stand_points_evaluated = 0     # 为第二角色查过多少个可站点
        self.budget_exhausted = 0           # 有多少次求解把尝试预算用光
        # 路线来源的分母:多少次尝试抽的是库路线、多少次是合成;合成设计了
        # 多少条、建成多少条、没建成的原因。合成关掉时这些全是零。
        self.synthesis: dict = {
            "bank_attempts": 0, "synthesized_attempts": 0,
            "target_designs": 0, "target_built": 0,
            "other_designs": 0, "other_built": 0,
            "design_rejections": {}}

    def add(self, rejection: Rejection) -> None:
        self.counts[rejection.reason] = self.counts.get(rejection.reason, 0) + 1
        self.first_details.setdefault(rejection.reason, rejection.detail)

    def note_combination(self) -> None:
        self.combinations_evaluated += 1

    def note_design(self, role: str, designs: int, built: int,
                    reason: str | None = None) -> None:
        self.synthesis[f"{role}_designs"] += int(designs)
        self.synthesis[f"{role}_built"] += int(built)
        if reason:
            key = f"{role}:{reason}"
            rejections = self.synthesis["design_rejections"]
            rejections[key] = rejections.get(key, 0) + 1

    def absorb(self, other: "RejectionLedger") -> None:
        """Fold another ledger's counters into this one (per-profile → total)."""
        for reason, count in other.counts.items():
            self.counts[reason] = self.counts.get(reason, 0) + count
            self.first_details.setdefault(reason, other.first_details.get(reason, ""))
        self.combinations_evaluated += other.combinations_evaluated
        self.stand_points_evaluated += other.stand_points_evaluated
        self.budget_exhausted += other.budget_exhausted
        for key, value in other.synthesis.items():
            if key == "design_rejections":
                for reason, count in value.items():
                    self.synthesis[key][reason] = self.synthesis[key].get(reason, 0) + count
            else:
                self.synthesis[key] += value

    def summary(self) -> dict:
        total = sum(self.counts.values())
        return {"total": total,
                "combinations_evaluated": self.combinations_evaluated,
                "stand_points_evaluated": self.stand_points_evaluated,
                "budget_exhausted": self.budget_exhausted,
                "by_reason": dict(sorted(self.counts.items(),
                                         key=lambda kv: -kv[1])),
                "route_synthesis": json.loads(json.dumps(self.synthesis)),
                "first_example": self.first_details}


def require_camera_clearance(scene: SceneInputs, params: dict) -> None:
    """Fail closed before any search when clearance screening cannot happen.

    CAMERA_CLEARANCE_REQUIRED in params means: a scene without a table must
    not be searched at all.  When a table is present, the rule keys and every
    fallback height must exist in the table, so a later attempt never fails
    half-way with an unscreened candidate."""
    if scene.clearance is None:
        if params.get("CAMERA_CLEARANCE_REQUIRED"):
            raise ValueError(
                f"{scene.scene_id}: CAMERA_CLEARANCE_REQUIRED but the scene config "
                "declares no camera_clearance_table")
        return
    rule_from_params(params, scene.clearance)
    for height in fallback_heights_from_params(params):
        if not scene.clearance.has_height(height):
            raise ValueError(
                f"{scene.scene_id}: CAMERA_HEIGHT_FALLBACK_M {height} is not in the "
                f"camera clearance table heights {scene.clearance.heights_m.tolist()}")


def screen_camera_clearance(scene: SceneInputs, params: dict, camera, yaw: float,
                            ledger: "RejectionLedger | None") -> dict | None:
    """Decide, from the scene's clearance table, whether this pose may be used.

    Returns the camera height to use and the evidence, or None (with a ledger
    entry) when the view is blocked at the scene height and at every fallback
    height declared in params.  Fallback heights are tried in the declared
    order; the decision is purely geometric (no answer is consulted)."""
    if scene.clearance is None:
        return {"screened": False, "camera_height_m": float(scene.camera_height_m),
                "fallback_used": False}
    rule = rule_from_params(params, scene.clearance)
    heights = [float(scene.camera_height_m)] + fallback_heights_from_params(params)
    tried = []
    for height in heights:
        fraction = scene.clearance.blocked_fraction(camera, height, yaw, rule)
        tried.append({"camera_height_m": height, "blocked_fraction": fraction})
        if fraction <= rule.blocked_fraction_max:
            return {"screened": True, "camera_height_m": height,
                    "blocked_fraction": fraction,
                    "fallback_used": height != float(scene.camera_height_m),
                    "tried": tried, "rule": rule.as_dict()}
    if ledger is not None:
        ledger.add(Rejection(
            "camera_clearance_blocked",
            f"blocked fractions {[round(t['blocked_fraction'], 3) for t in tried]} at "
            f"heights {heights} exceed {rule.blocked_fraction_max}"))
    return None


def target_route_pool(scene: SceneInputs, params: dict) -> list[Route]:
    """Routes a dual-motion solver may draw its target from.

    Kujiale's trajectory bank is half static routes, and every draw of one is
    a wasted attempt in solvers that reject a static target anyway (about
    half of all card1 attempts there on 2026-09-02).  With
    ROUTE_PREFILTER_STATIC_TARGETS set, routes whose end-to-end displacement
    is at most ROUTE_MIN_DISPLACEMENT_CM (default: the solvers' own 1e-6 cm)
    leave the pool before sampling.  The constraint is unchanged and the
    per-attempt rejection stays as a backstop; the flag is off by default so
    existing batches replay unchanged.  Solvers that need a still actor (card6R)
    keep drawing from the full bank."""
    if not params.get("ROUTE_PREFILTER_STATIC_TARGETS"):
        return scene.routes
    threshold = float(params.get("ROUTE_MIN_DISPLACEMENT_CM", 1.0e-6))
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("ROUTE_MIN_DISPLACEMENT_CM must be a finite non-negative number")
    pool = [route for route in scene.routes if route.displacement_cm > threshold]
    if not pool:
        raise ValueError(f"{scene.scene_id}: no route moves more than {threshold} cm")
    return pool


def route_pool_report(scene: SceneInputs, params: dict) -> dict:
    """What the batch manifest records about the route pool."""
    enabled = bool(params.get("ROUTE_PREFILTER_STATIC_TARGETS"))
    threshold = float(params.get("ROUTE_MIN_DISPLACEMENT_CM", 1.0e-6)) if enabled else None
    moving = sum(1 for route in scene.routes
                 if route.displacement_cm > (threshold if enabled else 1.0e-6))
    return {"routes_loaded": len(scene.routes), "moving_routes": moving,
            "static_routes": len(scene.routes) - moving,
            "static_targets_prefiltered": enabled,
            "min_displacement_cm": threshold}


def sample_clear_yaw(scene: SceneInputs, params: dict, camera, lo_yaw: float,
                     hi_yaw: float, rng, ledger: "RejectionLedger | None"):
    """Draw a camera yaw inside [lo_yaw, hi_yaw) that the clearance table calls clear.

    Without a table this is a plain uniform draw (unchanged behaviour).  With a
    table the draw is uniform over the clear 2-degree bins of the interval at
    the scene camera height; only when no bin is clear are the fallback heights
    tried, so a raised camera is used just where the room forces it.  When no
    height has a clear bin the attempt is rejected as camera_clearance_blocked.
    Returns (yaw, clearance) or None.  The decision is geometric: it never looks
    at an answer."""
    lo, hi = float(lo_yaw), float(hi_yaw)
    width = (hi - lo) if hi > lo else (hi - lo) + 360.0
    if scene.clearance is None:
        yaw = lo + width * rng.random()
        return (yaw, {"screened": False, "camera_height_m": float(scene.camera_height_m),
                      "fallback_used": False, "yaw_sampling": "uniform_in_interval"})
    rule = rule_from_params(params, scene.clearance)
    table = scene.clearance
    step = table.yaw_step_deg
    centres = table.yaws_deg
    inside = ((centres - lo) % 360.0) < width
    if not inside.any():                       # interval narrower than one bin
        inside = np.zeros_like(inside)
        inside[yaw_bin_index(lo + width / 2.0, step)] = True
    heights = [float(scene.camera_height_m)] + fallback_heights_from_params(params)
    tried = []
    for height in heights:
        mask = table.clear_yaw_mask(camera, height, rule) & inside
        tried.append({"camera_height_m": height, "clear_bins": int(mask.sum()),
                      "bins_in_interval": int(inside.sum())})
        if mask.any():
            bins = np.flatnonzero(mask)
            centre = float(centres[bins[int(rng.integers(len(bins)))]])
            # uniform inside the bin, then clipped to the interval
            yaw = centre + (rng.random() - 0.5) * step
            offset = (yaw - lo) % 360.0
            if offset >= width:
                yaw = centre
            fraction = table.blocked_fraction(camera, height, yaw, rule)
            return (float(yaw), {"screened": True, "camera_height_m": height,
                                 "blocked_fraction": fraction,
                                 "fallback_used": height != float(scene.camera_height_m),
                                 "tried": tried, "rule": rule.as_dict(),
                                 "yaw_sampling": "clear_bins_within_interval"})
    if ledger is not None:
        ledger.add(Rejection(
            "camera_clearance_blocked",
            f"no clear yaw bin in [{lo:.1f},{hi:.1f}) at heights {heights}"))
    return None


def require_route_synthesis(scene: SceneInputs, params: dict) -> None:
    """Fail closed before any search when route synthesis cannot happen.

    ROUTE_SYNTHESIS_ENABLED in params means the solver may design routes; a
    scene without a walkable grid must then not be searched at all, and every
    synthesis key must be present and sane, so no later attempt fails
    half-way with an unscreened designed route."""
    settings = SynthesisSettings.from_params(params)
    if settings is None:
        return
    if scene.walkable is None:
        raise ValueError(
            f"{scene.scene_id}: {ROUTE_SYNTHESIS_ENABLED_KEY} but the scene config "
            "declares no walkable_grid")
    if scene.walkable.cells_with_clearance(settings.margin_cm).size == 0:
        raise ValueError(
            f"{scene.scene_id}: no walkable cell keeps the {settings.margin_cm} cm margin")


def route_synthesizer(scene: SceneInputs, params: dict) -> RouteSynthesizer | None:
    """The scene's synthesizer, or None when synthesis is off in params."""
    settings = SynthesisSettings.from_params(params)
    if settings is None:
        return None
    if scene.walkable is None:
        raise ValueError(
            f"{scene.scene_id}: {ROUTE_SYNTHESIS_ENABLED_KEY} but the scene config "
            "declares no walkable_grid")
    return RouteSynthesizer(
        scene.walkable, settings, frame_rate_hz=settings.frame_rate_hz,
        frame_count=settings.frame_count)


def attempt_budgets(synth: RouteSynthesizer | None, max_attempts: int) -> tuple[int, int]:
    """(bank attempts, total attempts) for one solve.

    Bank first, and the bank keeps the whole budget the profile declares, so a
    scene that could fill a cell from recorded routes behaves exactly as it
    did before synthesis existed.  Designed routes get their own extra budget
    (ROUTE_SYNTHESIS_ATTEMPTS) after the bank budget is spent.  Without a
    synthesizer every attempt is a bank attempt."""
    bank = int(max_attempts)
    if synth is None:
        return bank, bank
    return bank, bank + int(synth.settings.synthesized_attempts)


def route_synthesis_report(scene: SceneInputs, params: dict) -> dict:
    """What a batch manifest records about route synthesis for this scene."""
    settings = SynthesisSettings.from_params(params)
    return {"enabled": settings is not None,
            "settings": settings.as_dict() if settings is not None else None,
            "walkable_grid": scene.provenance.get("walkable_grid"),
            "order": "bank_first_then_synthesize" if settings is not None else "bank_only"}


def _synthesis_pose(scene: SceneInputs, params: dict, rng, ledger: "RejectionLedger | None"):
    """Camera point and a clear yaw over the whole circle: the pose a designed
    route is built for.  Same clearance screen as every other draw."""
    camera = scene.camera_points[int(rng.integers(len(scene.camera_points)))]
    picked = sample_clear_yaw(scene, params, camera, -180.0, 180.0, rng, ledger)
    if picked is None:
        return None
    yaw, clearance = picked
    return camera, yaw, clearance


DESIGN_EDGE_MARGIN_DEG = 0.25   # keep designed azimuths off band and FOV edges


def _design_band(band, half_fov: float, *,
                 bound_deg: float | None = None) -> tuple[float, float]:
    """Interior of a band for a synthesized point.

    A directed Arc is kept in an unwrapped linear form for PointSpec.  The
    solver subsequently checks the generated point with band_to_arc, so a
    design interval such as 162 -> 198 remains the rear seam band.
    """
    bound = float(half_fov) if bound_deg is None else float(bound_deg)
    if band is None:
        lo, hi = -bound, bound
    elif isinstance(band, (Arc, dict)):
        arc = band_to_arc(band)
        if arc.width_deg >= 360.0:
            lo, hi = -bound, bound
        else:
            endpoint = arc.start_deg + arc.sweep_deg
            lo, hi = sorted((arc.start_deg, endpoint))
    else:
        raw_lo, raw_hi = (float(value) for value in band)
        if raw_hi < raw_lo:
            arc = band_to_arc((raw_lo, raw_hi))
            endpoint = arc.start_deg + arc.sweep_deg
            lo, hi = sorted((arc.start_deg, endpoint))
        else:
            lo, hi = max(raw_lo, -bound), min(raw_hi, bound)
    if hi - lo <= 2.0 * DESIGN_EDGE_MARGIN_DEG:
        return lo, hi
    return lo + DESIGN_EDGE_MARGIN_DEG, hi - DESIGN_EDGE_MARGIN_DEG

def _distance_floor_cm(params: dict) -> float:
    return float(params.get("MIN_CAMERA_DISTANCE_CM", 100.0))


def _design_target(synth: RouteSynthesizer, rng, ledger: "RejectionLedger",
                   camera, yaw: float, specs, idle: int, min_gap_deg: float | None = None):
    """Design the target's base route; records the design counters and, on
    failure, the reason as this attempt's rejection.  ``min_gap_deg`` is the
    azimuth the target must sweep between the two designed frames (the
    solver's own zero-score rule, applied at design time instead of after)."""
    before = dict(synth.counters, rejected=dict(synth.counters["rejected"]))
    route, reason = synth.design_many(rng, camera, yaw, specs, idle_frames=idle, role="target",
                                      min_gap_between_points_deg=min_gap_deg)
    ledger.note_design("target", synth.counters["designs"] - before["designs"],
                       synth.counters["built"] - before["built"], reason)
    if route is None:
        ledger.add(Rejection(reason, "designed target route could not be built"))
    return route


def _synthesize_other(synth: RouteSynthesizer, rng, ledger: "RejectionLedger",
                      camera, yaw: float, specs, acceptable, tries: int | None = None):
    """Design a second-actor route and hold it to the same checks as a bank
    candidate (``acceptable`` is the solver's own per-candidate predicate)."""
    for _ in range(int(tries or synth.settings.other_design_tries)):
        ledger.stand_points_evaluated += 1
        route, reason = synth.design(rng, camera, yaw, specs, idle_frames=0, role="other")
        ledger.note_design("other", 1, 0 if route is None else 1,
                           reason if route is None else None)
        if route is None:
            continue
        if acceptable(route):
            return route
        ledger.note_design("other", 0, 0, "other_failed_solver_checks")
    return None


def _route_sources(target_base: Route, other: Route) -> dict:
    return {"target": target_base.source, "other": other.source}


def solve_forward_cross_time(scene: SceneInputs, params: dict, *,
                             answer_band: AngleBand,
                             answer_bands: Sequence[AngleBand],
                             anchor_frame: int, idle_choices: Iterable[int],
                             rng, ledger: RejectionLedger,
                             anchor_band: tuple[float, float] | None = None,
                             target_moves_more: bool | None = None,
                             max_attempts: int = 4000,
                             query_domain: str | None = None,
                             answer_domain: str | None = None,
                             query_bound_deg: float | None = None,
                             query_requires_visibility: bool | None = None,
                             query_visibility: str | None = None,
                             secondary_anchor_bound_deg: float | None = None,
                             secondary_query_bound_deg: float | None = None) -> PointPlan | Rejection:
    """①F 正向错时:音频锚在前,视觉查询在后(查询帧=片尾)。

    约束(全部由题型声明,与房间无关):
      - 目标片尾方位落进指定答案带(解 yaw,不枚举);
      - 正式 Open 评分器下直接复述锚定角度必须得零;
      - 若分配了锚定带,锚定方位必须落进该带,供条件配平;
      - 锚定时刻两角色方位分离 >= MIN_AZIMUTH_SEP(锚可绑定);
      - 锚定时刻与查询时刻目标都在视锥内;
      - 另一角色不得与目标同答案带(否则选项无区分度);
      - 有视线筛查就用,没有就如实记未筛。
    """
    frame_count = scene_frame_count(scene, params)
    anchor_frame = _require_frame(anchor_frame, frame_count, name="anchor_frame")
    query_bound, query_visible = resolve_query_geometry(
        scene, params, query_domain=query_domain, answer_domain=answer_domain,
        query_bound_deg=query_bound_deg,
        query_requires_visibility=query_requires_visibility,
        query_visibility=query_visibility)
    half_fov = effective_half_fov(scene, params)
    visibility_half_fov = camera_half_fov(scene)
    visibility = resolve_query_visibility(
        query_domain if query_domain is not None else answer_domain,
        query_visibility=query_visibility,
        query_requires_visibility=query_requires_visibility,
    )
    theta_full = float(params["THETA_FULL"])
    theta_half = angle_credit_radius(params)
    min_sep = float(params["MIN_AZIMUTH_SEP"])
    band_lo, band_hi = band_to_bounds(answer_band)
    pool = target_route_pool(scene, params)
    n_routes, n_cams = len(pool), len(scene.camera_points)
    synth = route_synthesizer(scene, params)
    bank_attempts, total_attempts = attempt_budgets(synth, max_attempts)
    solve_lo, solve_hi = interior_answer_band(band_lo, band_hi, params)
    for attempt in range(1, total_attempts + 1):
        ledger.note_combination()
        if synth is None or attempt <= bank_attempts:
            ledger.synthesis["bank_attempts"] += 1
            route = pool[int(rng.integers(n_routes))]
            idle = int(rng.choice(list(idle_choices)))
            moved = route.shifted(idle)
            if moved.displacement_cm <= 1.0e-6:
                ledger.add(Rejection("target_route_static_for_dual_motion"))
                continue
            camera = scene.camera_points[int(rng.integers(n_cams))]
            end_xy = moved.at(frame_count - 1)
            if _too_close(camera, end_xy, params):
                ledger.add(Rejection("camera_too_close_to_target"))
                continue
            lo_yaw, hi_yaw = yaw_interval_for_band(camera, end_xy, solve_lo, solve_hi)
            picked = sample_clear_yaw(scene, params, camera, lo_yaw, hi_yaw, rng, ledger)
            if picked is None:
                continue
            yaw, clearance = picked
        else:
            # 库里抽不到就当场设计:先定机位与净空朝向,再把锚帧、查询帧的位置
            # 直接画进分配的方位带与距离范围,铺一条匀速直线。下面的每条检查
            # 对合成路线原样执行,不因来源放宽。
            ledger.synthesis["synthesized_attempts"] += 1
            pose = _synthesis_pose(scene, params, rng, ledger)
            if pose is None:
                continue
            camera, yaw, clearance = pose
            idle = int(rng.choice(list(idle_choices)))
            floor_cm = _distance_floor_cm(params)
            far_cm = synth.settings.max_camera_distance_cm
            route = _design_target(synth, rng, ledger, camera, yaw, [
                PointSpec(anchor_frame, *_design_band(anchor_band, half_fov),
                          _anchor_min_distance_cm(params) or floor_cm, far_cm),
                PointSpec(frame_count - 1,
                          *_design_band(answer_band, half_fov,
                                        bound_deg=query_bound),
                          floor_cm, far_cm)], idle,
                min_gap_deg=(theta_half if anchor_band is not None else theta_full)
                + DESIGN_EDGE_MARGIN_DEG)
            if route is None:
                continue
            moved = route.shifted(idle)
            if moved.displacement_cm <= 1.0e-6:
                ledger.add(Rejection("target_route_static_for_dual_motion"))
                continue
            end_xy = moved.at(frame_count - 1)
            if _too_close(camera, end_xy, params):
                ledger.add(Rejection("camera_too_close_to_target"))
                continue
        az_end = relative_azimuth_deg(camera, yaw, end_xy)
        if not _angle_in_band(az_end, (band_lo, band_hi)):
            ledger.add(Rejection("band_solution_out_of_band",
                                 f"az={az_end:.2f} band={band_lo},{band_hi}"))
            continue
        if abs(az_end) > query_bound:
            ledger.add(Rejection("answer_band_outside_fov",
                                 "the declared band lies outside the camera "
                                 "field of view"))
            continue
        if not query_visibility_matches(az_end, visibility, visibility_half_fov):
            ledger.add(Rejection(
                "target_query_visibility_mismatch",
                f"az={az_end:.2f} policy={visibility} "
                f"half_fov={visibility_half_fov:.2f}"))
            continue
        anchor_xy = moved.at(anchor_frame)
        if _too_close_at_anchor(camera, anchor_xy, params):
            ledger.add(Rejection("camera_too_close_to_target_at_anchor"))
            continue
        az_anchor = relative_azimuth_deg(camera, yaw, anchor_xy)
        if abs(az_anchor) > half_fov:
            ledger.add(Rejection("target_outside_fov_at_anchor"))
            continue
        if anchor_band is not None and not _angle_in_band(az_anchor, anchor_band):
            ledger.add(Rejection(
                "anchor_outside_allocated_band",
                f"az={az_anchor:.2f} band={band_to_bounds(anchor_band)}"))
            continue
        gap = circular_gap_deg(az_anchor, az_end)
        if anchor_band is not None:
            if not open_angle_candidate_scores_zero(
                    az_anchor, az_end, theta_half):
                ledger.add(Rejection(
                    "anchor_angle_scores_nonzero_at_query",
                    f"gap {gap:.1f} <= widest credited radius "
                    f"THETA_HALF {theta_half}"))
                continue
        elif gap <= theta_full:
            # Compatibility for diagnostic callers that have not allocated an
            # anchor stratum. Production card1 cells always pass anchor_band.
            ledger.add(Rejection(
                "insufficient_azimuth_travel_after_anchor",
                f"gap {gap:.1f} <= THETA_FULL {theta_full}"))
            continue
        other_route = _pick_other_route(
            scene, moved, camera, yaw, az_anchor, az_end, band_lo, band_hi,
            answer_bands, min_sep, half_fov, theta_half, params,
            anchor_frame, frame_count - 1, rng, ledger,
            target_moves_more=target_moves_more,
            anchor_bound_deg=half_fov, query_bound_deg=query_bound,
            other_anchor_bound_deg=secondary_anchor_bound_deg,
            other_query_bound_deg=secondary_query_bound_deg)
        if other_route is None:
            continue
        other_anchor_xy = other_route.at(anchor_frame)
        other_answer_xy = other_route.at(frame_count - 1)
        other_answer_az = relative_azimuth_deg(camera, yaw, other_answer_xy)
        if scene.line_of_sight is not None:
            if not scene.line_of_sight(camera, anchor_xy):
                ledger.add(Rejection("target_occluded_at_anchor_frame"))
                continue
            if query_visible and not scene.line_of_sight(camera, end_xy):
                ledger.add(Rejection("target_occluded_at_query_frame"))
                continue
            if (not scene.line_of_sight(camera, other_anchor_xy)
                    or (query_visible
                        and not scene.line_of_sight(camera, other_answer_xy))):
                ledger.add(Rejection("other_actor_occluded"))
                continue
        return PointPlan(
            scene_id=scene.scene_id, profile_id="card1F", camera_xy=camera,
            camera_ue_yaw_deg=yaw, camera_height_m=clearance["camera_height_m"],
            camera_clearance=clearance, target_route=moved, base_route=route,
            other_route=other_route, idle_frames=idle, anchor_frame=anchor_frame,
            query_frame=frame_count - 1,
            answer_cell={"kind": "azimuth_band", "band": [band_lo, band_hi],
                         "value_deg": az_end},
            checks={"az_anchor_deg": az_anchor, "az_end_deg": az_end,
                    "azimuth_travel_deg": circular_gap_deg(az_anchor, az_end),
                    "anchor_open_score": (
                        0.0 if gap > theta_half
                        else 0.5 if gap > theta_full else 1.0),
                    "anchor_open_zero_score_min_gap_deg": theta_half,
                    "allocated_anchor_band": (
                        list(anchor_band) if anchor_band is not None else None),
                    "anchor_camera_distance_cm": math.hypot(
                        anchor_xy[0] - camera[0], anchor_xy[1] - camera[1]),
                    "anchor_camera_distance_min_cm": _anchor_min_distance_cm(params),
                    "anchor_separation_deg": circular_gap_deg(
                        az_anchor, relative_azimuth_deg(
                            camera, yaw, other_anchor_xy)),
                    "gatea_answer_azimuth_deg": other_answer_az,
                    "gatea_open_gold_separation_deg": circular_gap_deg(
                        az_end, other_answer_az),
                    "gatea_open_min_separation_deg": 2.0 * theta_half,
                    "line_of_sight_screened": scene.line_of_sight_screened,
                    "route_sources": _route_sources(route, other_route),
                    "query_bound_deg": query_bound,
                    "query_requires_visibility": query_visible,
                    "query_visibility": visibility,
                    "search_attempts": attempt},
        )
    ledger.budget_exhausted += 1
    return Rejection("no_candidate_within_attempt_budget",
                     f"{total_attempts} attempts ({bank_attempts} bank, "
                     f"{total_attempts - bank_attempts} designed)")


def solve_backward_cross_time(scene: SceneInputs, params: dict, *,
                              answer_band: AngleBand,
                              answer_bands: Sequence[AngleBand],
                              anchor_frame: int, query_frame: int,
                              idle_choices: Iterable[int], rng,
                              ledger: RejectionLedger,
                              anchor_band: tuple[float, float] | None = None,
                              target_moves_more: bool | None = None,
                              max_attempts: int = 4000,
                              query_domain: str | None = None,
                              answer_domain: str | None = None,
                              query_bound_deg: float | None = None,
                              query_requires_visibility: bool | None = None,
                              query_visibility: str | None = None,
                              secondary_anchor_bound_deg: float | None = None,
                              secondary_query_bound_deg: float | None = None) -> PointPlan | Rejection:
    """①B 反向错时:视觉查询在前,音频锚在后(末段发声确定身份)。

    与 ①F 的差别只在时间方向,几何约束同构:查询帧的目标状态必须可观察,
    锚定时刻两角色可分辨,且**查询时刻附近不得有直接泄露答案的音频**
    (由 AudioProgram profile 保证,这里只声明并记录该要求)。
    """
    frame_count = scene_frame_count(scene, params)
    anchor_frame = _require_frame(anchor_frame, frame_count, name="anchor_frame")
    query_frame = _require_frame(query_frame, frame_count, name="query_frame")
    query_bound, query_visible = resolve_query_geometry(
        scene, params, query_domain=query_domain, answer_domain=answer_domain,
        query_bound_deg=query_bound_deg,
        query_requires_visibility=query_requires_visibility,
        query_visibility=query_visibility)
    half_fov = effective_half_fov(scene, params)
    visibility_half_fov = camera_half_fov(scene)
    visibility = resolve_query_visibility(
        query_domain if query_domain is not None else answer_domain,
        query_visibility=query_visibility,
        query_requires_visibility=query_requires_visibility,
    )
    theta_full = float(params["THETA_FULL"])
    theta_half = angle_credit_radius(params)
    min_sep = float(params["MIN_AZIMUTH_SEP"])
    band_lo, band_hi = band_to_bounds(answer_band)
    pool = target_route_pool(scene, params)
    n_routes, n_cams = len(pool), len(scene.camera_points)
    synth = route_synthesizer(scene, params)
    bank_attempts, total_attempts = attempt_budgets(synth, max_attempts)
    solve_lo, solve_hi = interior_answer_band(band_lo, band_hi, params)
    for attempt in range(1, total_attempts + 1):
        ledger.note_combination()
        if synth is None or attempt <= bank_attempts:
            ledger.synthesis["bank_attempts"] += 1
            route = pool[int(rng.integers(n_routes))]
            idle = int(rng.choice(list(idle_choices)))
            moved = route.shifted(idle)
            if moved.displacement_cm <= 1.0e-6:
                ledger.add(Rejection("target_route_static_for_dual_motion"))
                continue
            camera = scene.camera_points[int(rng.integers(n_cams))]
            query_xy = moved.at(query_frame)
            if _too_close(camera, query_xy, params):
                ledger.add(Rejection("camera_too_close_to_target"))
                continue
            lo_yaw, hi_yaw = yaw_interval_for_band(camera, query_xy, solve_lo, solve_hi)
            picked = sample_clear_yaw(scene, params, camera, lo_yaw, hi_yaw, rng, ledger)
            if picked is None:
                continue
            yaw, clearance = picked
        else:
            ledger.synthesis["synthesized_attempts"] += 1
            pose = _synthesis_pose(scene, params, rng, ledger)
            if pose is None:
                continue
            camera, yaw, clearance = pose
            idle = int(rng.choice(list(idle_choices)))
            floor_cm = _distance_floor_cm(params)
            far_cm = synth.settings.max_camera_distance_cm
            route = _design_target(synth, rng, ledger, camera, yaw, [
                PointSpec(anchor_frame, *_design_band(anchor_band, half_fov),
                          _anchor_min_distance_cm(params) or floor_cm, far_cm),
                PointSpec(query_frame,
                          *_design_band(answer_band, half_fov,
                                        bound_deg=query_bound),
                          floor_cm, far_cm)], idle,
                min_gap_deg=(theta_half if anchor_band is not None else theta_full)
                + DESIGN_EDGE_MARGIN_DEG)
            if route is None:
                continue
            moved = route.shifted(idle)
            if moved.displacement_cm <= 1.0e-6:
                ledger.add(Rejection("target_route_static_for_dual_motion"))
                continue
            query_xy = moved.at(query_frame)
            if _too_close(camera, query_xy, params):
                ledger.add(Rejection("camera_too_close_to_target"))
                continue
        az_query = relative_azimuth_deg(camera, yaw, query_xy)
        if (not _angle_in_band(az_query, (band_lo, band_hi))
                or abs(az_query) > query_bound):
            ledger.add(Rejection("answer_band_outside_fov"))
            continue
        if not query_visibility_matches(az_query, visibility, visibility_half_fov):
            ledger.add(Rejection(
                "target_query_visibility_mismatch",
                f"az={az_query:.2f} policy={visibility} "
                f"half_fov={visibility_half_fov:.2f}"))
            continue
        anchor_xy = moved.at(anchor_frame)
        if _too_close_at_anchor(camera, anchor_xy, params):
            ledger.add(Rejection("camera_too_close_to_target_at_anchor"))
            continue
        az_anchor = relative_azimuth_deg(camera, yaw, anchor_xy)
        if abs(az_anchor) > half_fov:
            ledger.add(Rejection("target_outside_fov_at_anchor"))
            continue
        if anchor_band is not None and not _angle_in_band(az_anchor, anchor_band):
            ledger.add(Rejection(
                "anchor_outside_allocated_band",
                f"az={az_anchor:.2f} band={anchor_band[0]},{anchor_band[1]}"))
            continue
        # 反向错时同样要求"查询时刻的状态不能在锚定时刻直接读到":
        # 目标必须在两个时刻之间移动够多,否则听完再看当前帧即可作答。
        gap = circular_gap_deg(az_anchor, az_query)
        if anchor_band is not None:
            if not open_angle_candidate_scores_zero(
                    az_anchor, az_query, theta_half):
                ledger.add(Rejection(
                    "anchor_angle_scores_nonzero_at_query",
                    f"gap {gap:.1f} <= widest credited radius "
                    f"THETA_HALF {theta_half}"))
                continue
        elif gap <= theta_full:
            ledger.add(Rejection(
                "insufficient_azimuth_travel_between_frames",
                f"gap {gap:.1f} <= THETA_FULL {theta_full}"))
            continue
        other_route = _pick_other_route(
            scene, moved, camera, yaw, az_anchor, az_query, band_lo, band_hi,
            answer_bands, min_sep, half_fov, theta_half, params,
            anchor_frame, query_frame, rng, ledger,
            target_moves_more=target_moves_more,
            anchor_bound_deg=half_fov, query_bound_deg=query_bound,
            other_anchor_bound_deg=secondary_anchor_bound_deg,
            other_query_bound_deg=secondary_query_bound_deg)
        if other_route is None:
            continue
        other_anchor_xy = other_route.at(anchor_frame)
        other_answer_xy = other_route.at(query_frame)
        other_answer_az = relative_azimuth_deg(camera, yaw, other_answer_xy)
        if scene.line_of_sight is not None:
            if not scene.line_of_sight(camera, anchor_xy):
                ledger.add(Rejection("target_occluded_at_anchor_frame"))
                continue
            if query_visible and not scene.line_of_sight(camera, query_xy):
                ledger.add(Rejection("target_occluded_at_query_frame"))
                continue
            if (not scene.line_of_sight(camera, other_anchor_xy)
                    or (query_visible
                        and not scene.line_of_sight(camera, other_answer_xy))):
                ledger.add(Rejection("other_actor_occluded"))
                continue
        return PointPlan(
            scene_id=scene.scene_id, profile_id="card1B", camera_xy=camera,
            camera_ue_yaw_deg=yaw, camera_height_m=clearance["camera_height_m"],
            camera_clearance=clearance, target_route=moved, base_route=route,
            other_route=other_route, idle_frames=idle, anchor_frame=anchor_frame,
            query_frame=query_frame,
            answer_cell={"kind": "azimuth_band", "band": [band_lo, band_hi],
                         "value_deg": az_query},
            checks={"az_anchor_deg": az_anchor, "az_query_deg": az_query,
                    "azimuth_travel_deg": circular_gap_deg(az_anchor, az_query),
                    "anchor_open_score": (
                        0.0 if gap > theta_half
                        else 0.5 if gap > theta_full else 1.0),
                    "anchor_open_zero_score_min_gap_deg": theta_half,
                    "allocated_anchor_band": (
                        list(anchor_band) if anchor_band is not None else None),
                    "anchor_camera_distance_cm": math.hypot(
                        anchor_xy[0] - camera[0], anchor_xy[1] - camera[1]),
                    "anchor_camera_distance_min_cm": _anchor_min_distance_cm(params),
                    "gatea_answer_azimuth_deg": other_answer_az,
                    "gatea_open_gold_separation_deg": circular_gap_deg(
                        az_query, other_answer_az),
                    "gatea_open_min_separation_deg": 2.0 * theta_half,
                    "line_of_sight_screened": scene.line_of_sight_screened,
                    "requires_silence_near_query": True,
                    "route_sources": _route_sources(route, other_route),
                    "query_bound_deg": query_bound,
                    "query_requires_visibility": query_visible,
                    "query_visibility": visibility,
                    "search_attempts": attempt},
        )
    ledger.budget_exhausted += 1
    return Rejection("no_candidate_within_attempt_budget",
                     f"{total_attempts} attempts ({bank_attempts} bank, "
                     f"{total_attempts - bank_attempts} designed)")


def _too_close(camera, point, params) -> bool:
    min_cm = float(params.get("MIN_CAMERA_DISTANCE_CM", 100.0))
    return math.hypot(point[0] - camera[0], point[1] - camera[1]) < min_cm


def _anchor_min_distance_cm(params):
    """Explicit anchor-instant camera distance floor; None keeps the old
    query-frame-only behaviour so historical params stay reproducible."""
    value = params.get("MIN_CAMERA_DISTANCE_ANCHOR_CM")
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("MIN_CAMERA_DISTANCE_ANCHOR_CM must be a finite "
                         "non-negative centimetre value")
    return value


def _too_close_at_anchor(camera, point, params) -> bool:
    """The old floor only looked at the query frame; a dog barking 0.7 m from
    the lens at the anchor instant fell below the frame's bottom edge and was
    never screened.  This floor is opt-in via MIN_CAMERA_DISTANCE_ANCHOR_CM."""
    min_cm = _anchor_min_distance_cm(params)
    if min_cm is None:
        return False
    return math.hypot(point[0] - camera[0], point[1] - camera[1]) < min_cm


def _pick_other_route(scene, target_route, camera, yaw, az_anchor, az_answer,
                      band_lo, band_hi, answer_bands, min_sep, half_fov,
                      theta_half, params, anchor_frame, query_frame, rng,
                      ledger, target_moves_more=None, *,
                      anchor_bound_deg=None, query_bound_deg=None,
                      other_anchor_bound_deg=None, other_query_bound_deg=None):
    """Pick a moving Gate-A actor for both MCQ and Open card1 forms.

    Bank routes are tried first (64 random draws).  When the solver may
    synthesize routes, a designed second-actor route is tried next and held
    to exactly the same per-candidate checks (``acceptable``)."""
    anchor_bound = half_fov if anchor_bound_deg is None else float(anchor_bound_deg)
    query_bound = half_fov if query_bound_deg is None else float(query_bound_deg)
    other_anchor_bound = (
        anchor_bound if other_anchor_bound_deg is None
        else float(other_anchor_bound_deg))
    other_query_bound = (
        query_bound if other_query_bound_deg is None
        else float(other_query_bound_deg))
    if (not math.isfinite(anchor_bound) or not 0.0 <= anchor_bound <= 180.0
            or not math.isfinite(other_anchor_bound)
            or not 0.0 <= other_anchor_bound <= 180.0
            or not math.isfinite(query_bound) or not 0.0 <= query_bound <= 180.0
            or not math.isfinite(other_query_bound)
            or not 0.0 <= other_query_bound <= 180.0):
        raise ValueError("actor bearing bounds must be finite and lie in [0, 180]")
    flags = {"open_overlap": False, "outside_answer_space": False,
             "motion_rank_mismatch": False, "static_route": False}

    def acceptable(route) -> bool:
        if route.route_id == target_route.route_id:
            return False
        if route.displacement_cm <= 1.0e-6:
            flags["static_route"] = True
            return False
        if target_moves_more is not None:
            observed = target_route.displacement_cm > route.displacement_cm
            if math.isclose(target_route.displacement_cm,
                            route.displacement_cm, abs_tol=1e-6) or \
                    observed != target_moves_more:
                flags["motion_rank_mismatch"] = True
                return False
        anchor_xy = route.at(anchor_frame)
        answer_xy = route.at(query_frame)
        az_other_anchor = relative_azimuth_deg(camera, yaw, anchor_xy)
        az_other_answer = relative_azimuth_deg(camera, yaw, answer_xy)
        if (abs(az_other_anchor) > other_anchor_bound
                or abs(az_other_answer) > other_query_bound):
            return False
        if circular_gap_deg(az_other_anchor, az_anchor) < min_sep:
            return False
        if not any(_angle_in_band(az_other_answer, band)
                   for band in answer_bands):
            flags["outside_answer_space"] = True
            return False
        if _angle_in_band(az_other_answer, (band_lo, band_hi)):
            return False
        if not open_angle_gold_regions_disjoint(
                az_answer, az_other_answer, theta_half):
            flags["open_overlap"] = True
            return False
        if _too_close(camera, anchor_xy, params) or \
                _too_close(camera, answer_xy, params):
            return False
        return True

    order = rng.permutation(len(scene.routes))
    for index in order[:64]:
        ledger.stand_points_evaluated += 1
        route = scene.routes[int(index)]
        if acceptable(route):
            return route
    synth = route_synthesizer(scene, params)
    if synth is not None:
        # 对照狗的答案带从声明的其他带里选一条,锚帧位置在视场内;分离、
        # 距离底线、Open 金标不重叠、运动量排序都由上面的 acceptable 复核。
        other_bands = [band for band in answer_bands
                       if not bands_equivalent(band, (band_lo, band_hi))]
        if other_bands:
            chosen = other_bands[int(rng.integers(len(other_bands)))]
            floor_cm = _distance_floor_cm(params)
            far_cm = synth.settings.max_camera_distance_cm
            # the draws avoid what acceptable() would reject anyway: within
            # MIN_AZIMUTH_SEP of the target at the anchor instant, within the
            # Open gold radius of the target's answer at the query instant
            separation = (az_anchor, min_sep + DESIGN_EDGE_MARGIN_DEG)
            gold = (az_answer, 2.0 * theta_half + DESIGN_EDGE_MARGIN_DEG)
            answer_spec = PointSpec(query_frame, *_design_band(
                chosen, half_fov, bound_deg=other_query_bound),
                                    floor_cm, far_cm,
                                    exclusions=((gold, separation) if anchor_frame == query_frame
                                                else (gold,)))
            if anchor_frame == query_frame:
                specs = [answer_spec]
            else:
                specs = [PointSpec(anchor_frame, *_design_band(
                    None, half_fov, bound_deg=other_anchor_bound),
                                   floor_cm, far_cm, exclusions=(separation,)), answer_spec]
            route = _synthesize_other(synth, rng, ledger, camera, yaw, specs, acceptable)
            if route is not None:
                return route
    if flags["motion_rank_mismatch"]:
        ledger.add(Rejection(
            "no_second_route_for_allocated_motion_rank",
            "moving secondary routes existed, but none satisfied the "
            "allocated target_moves_more relation together with the spatial "
            "question constraints"))
    elif flags["outside_answer_space"]:
        ledger.add(Rejection(
            "no_second_actor_in_declared_mcq_space",
            "candidate actors were visible and separated, but at least one "
            "fell outside every declared MCQ answer band and no valid Gate A "
            "actor remained"))
    elif flags["open_overlap"]:
        ledger.add(Rejection(
            "no_second_actor_with_disjoint_open_gold",
            "candidate actors existed outside the main MCQ band, but none "
            f"was more than {2.0 * theta_half:.1f} degrees from the main Open "
            "gold"))
    elif flags["static_route"]:
        ledger.add(Rejection(
            "no_moving_second_actor",
            "secondary routes were available, but only static routes survived "
            "the other dual-motion constraints"))
    else:
        ledger.add(Rejection(
            "no_separable_second_actor",
            "no navigable stand point clears the anchor-instant azimuth "
            "separation while staying out of the answer band and inside "
            "the field of view"))
    return None



def solve_instant_azimuth(scene: SceneInputs, params: dict, *,
                          answer_band: AngleBand,
                          answer_bands: Sequence[AngleBand],
                          query_frame: int, profile_id: str,
                          idle_choices: Iterable[int], rng,
                          ledger: RejectionLedger,
                          target_moves_more: bool | None = None,
                          max_attempts: int = 4000,
                          open_half_width_deg: float | None = None,
                          query_domain: str | None = None,
                          answer_domain: str | None = None,
                          query_bound_deg: float | None = None,
                          query_requires_visibility: bool | None = None,
                          query_visibility: str | None = None,
                          secondary_anchor_bound_deg: float | None = None,
                          secondary_query_bound_deg: float | None = None,
                          candidate_validator: Callable[[PointPlan], Rejection | None] | None = None):
    """Immediate-DoA control: bind the caller and its visual azimuth together."""
    frame_count = scene_frame_count(scene, params)
    query_frame = _require_frame(query_frame, frame_count, name="query_frame")
    band_lo, band_hi = band_to_bounds(answer_band)
    solve_lo, solve_hi = interior_answer_band(band_lo, band_hi, params)
    query_bound, query_visible = resolve_query_geometry(
        scene, params, query_domain=query_domain, answer_domain=answer_domain,
        query_bound_deg=query_bound_deg,
        query_requires_visibility=query_requires_visibility,
        query_visibility=query_visibility)
    half_fov = effective_half_fov(scene, params)
    visibility_half_fov = camera_half_fov(scene)
    visibility = resolve_query_visibility(
        query_domain if query_domain is not None else answer_domain,
        query_visibility=query_visibility,
        query_requires_visibility=query_requires_visibility,
    )
    min_sep = float(params["MIN_AZIMUTH_SEP"])
    theta_half = (angle_credit_radius(params) if open_half_width_deg is None
                  else float(open_half_width_deg))
    pool = target_route_pool(scene, params)
    n_routes, n_cams = len(pool), len(scene.camera_points)
    synth = route_synthesizer(scene, params)
    bank_attempts, total_attempts = attempt_budgets(synth, max_attempts)
    for attempt in range(1, total_attempts + 1):
        ledger.note_combination()
        if synth is None or attempt <= bank_attempts:
            ledger.synthesis["bank_attempts"] += 1
            route = pool[int(rng.integers(n_routes))]
            idle = int(rng.choice(list(idle_choices)))
            moved = route.shifted(idle)
            if moved.displacement_cm <= 1.0e-6:
                ledger.add(Rejection("target_route_static_for_dual_motion"))
                continue
            answer_xy = moved.at(query_frame)
            camera = scene.camera_points[int(rng.integers(n_cams))]
            if _too_close(camera, answer_xy, params):
                ledger.add(Rejection("camera_too_close_to_target"))
                continue
            lo_yaw, hi_yaw = yaw_interval_for_band(
                camera, answer_xy, solve_lo, solve_hi)
            picked = sample_clear_yaw(scene, params, camera, lo_yaw, hi_yaw, rng, ledger)
            if picked is None:
                continue
            yaw, clearance = picked
        else:
            ledger.synthesis["synthesized_attempts"] += 1
            pose = _synthesis_pose(scene, params, rng, ledger)
            if pose is None:
                continue
            camera, yaw, clearance = pose
            idle = int(rng.choice(list(idle_choices)))
            route = _design_target(synth, rng, ledger, camera, yaw, [
                PointSpec(query_frame,
                          *_design_band(answer_band, half_fov,
                                        bound_deg=query_bound),
                          _distance_floor_cm(params),
                          synth.settings.max_camera_distance_cm)], idle)
            if route is None:
                continue
            moved = route.shifted(idle)
            if moved.displacement_cm <= 1.0e-6:
                ledger.add(Rejection("target_route_static_for_dual_motion"))
                continue
            answer_xy = moved.at(query_frame)
            if _too_close(camera, answer_xy, params):
                ledger.add(Rejection("camera_too_close_to_target"))
                continue
        azimuth = relative_azimuth_deg(camera, yaw, answer_xy)
        if abs(azimuth) > query_bound:
            ledger.add(Rejection("answer_band_outside_fov"))
            continue
        if not query_visibility_matches(azimuth, visibility, visibility_half_fov):
            ledger.add(Rejection(
                "target_query_visibility_mismatch",
                f"az={azimuth:.2f} policy={visibility} "
                f"half_fov={visibility_half_fov:.2f}"))
            continue
        other = _pick_other_route(
            scene, moved, camera, yaw, azimuth, azimuth,
            band_lo, band_hi, answer_bands, min_sep, half_fov,
            theta_half, params, query_frame, query_frame, rng, ledger,
            target_moves_more=target_moves_more,
            anchor_bound_deg=query_bound, query_bound_deg=query_bound,
            other_anchor_bound_deg=secondary_anchor_bound_deg,
            other_query_bound_deg=secondary_query_bound_deg)
        if other is None:
            continue
        other_azimuth = relative_azimuth_deg(
            camera, yaw, other.at(query_frame))
        if scene.line_of_sight is not None and query_visible:
            if not scene.line_of_sight(camera, answer_xy):
                ledger.add(Rejection("target_occluded_at_binding_instant"))
                continue
            if not scene.line_of_sight(camera, other.at(query_frame)):
                ledger.add(Rejection("other_actor_occluded"))
                continue
        plan = PointPlan(
            scene_id=scene.scene_id, profile_id=profile_id,
            camera_xy=camera, camera_ue_yaw_deg=yaw,
            camera_height_m=clearance["camera_height_m"], camera_clearance=clearance,
            target_route=moved, base_route=route, other_route=other,
            idle_frames=idle, anchor_frame=query_frame,
            query_frame=query_frame,
            answer_cell={"kind": "instant_azimuth_band",
                         "band": [band_lo, band_hi],
                         "value_deg": azimuth},
            checks={
                "binding_azimuth_deg": round(azimuth, 3),
                "gatea_answer_azimuth_deg": round(other_azimuth, 3),
                "gatea_open_gold_separation_deg": circular_gap_deg(
                    azimuth, other_azimuth),
                "gatea_open_min_separation_deg": 2.0 * theta_half,
                "line_of_sight_screened": scene.line_of_sight_screened,
                "route_sources": _route_sources(route, other),
                "query_bound_deg": query_bound,
                "query_requires_visibility": query_visible,
                "query_visibility": visibility,
                "search_attempts": attempt,
            },
        )
        if candidate_validator is not None:
            rejection = candidate_validator(plan)
            if rejection is not None:
                if not isinstance(rejection, Rejection):
                    raise TypeError("candidate validator must return a Rejection or None")
                ledger.add(rejection)
                continue
        return plan
    ledger.budget_exhausted += 1
    return Rejection("no_candidate_within_attempt_budget",
                     f"{total_attempts} attempts ({bank_attempts} bank, "
                     f"{total_attempts - bank_attempts} designed)")

def solve_instant_distance_order(scene: SceneInputs, params: dict, *,
                                 query_frame: int, profile_id: str,
                                 idle_choices: Iterable[int], rng,
                                 ledger: RejectionLedger,
                                 target_moves_more: bool | None = None,
                                 min_distance_gap_cm: float = 50.0,
                                 max_attempts: int = 4000):
    """Visual control: the allocated target is measurably closer at one frame."""
    frame_count = scene_frame_count(scene, params)
    query_frame = _require_frame(query_frame, frame_count, name="query_frame")
    half_fov = effective_half_fov(scene, params)
    min_sep = float(params["MIN_AZIMUTH_SEP"])
    min_gap = float(min_distance_gap_cm)
    pool = target_route_pool(scene, params)
    n_routes, n_cams = len(pool), len(scene.camera_points)
    synth = route_synthesizer(scene, params)
    bank_attempts, total_attempts = attempt_budgets(synth, max_attempts)
    for attempt in range(1, total_attempts + 1):
        ledger.note_combination()
        if synth is None or attempt <= bank_attempts:
            ledger.synthesis["bank_attempts"] += 1
            route = pool[int(rng.integers(n_routes))]
            idle = int(rng.choice(list(idle_choices)))
            moved = route.shifted(idle)
            if moved.displacement_cm <= 1.0e-6:
                ledger.add(Rejection("target_route_static_for_dual_motion"))
                continue
            camera = scene.camera_points[int(rng.integers(n_cams))]
            target_xy = moved.at(query_frame)
            target_distance = math.dist(camera, target_xy)
            if _too_close(camera, target_xy, params):
                ledger.add(Rejection("camera_too_close_to_target"))
                continue
            picked = sample_clear_yaw(scene, params, camera, -180.0, 180.0, rng, ledger)
            if picked is None:
                continue
            yaw, clearance = picked
        else:
            ledger.synthesis["synthesized_attempts"] += 1
            pose = _synthesis_pose(scene, params, rng, ledger)
            if pose is None:
                continue
            camera, yaw, clearance = pose
            idle = int(rng.choice(list(idle_choices)))
            route = _design_target(synth, rng, ledger, camera, yaw, [
                PointSpec(query_frame, *_design_band(None, half_fov),
                          _distance_floor_cm(params), synth.settings.max_camera_distance_cm)],
                idle)
            if route is None:
                continue
            moved = route.shifted(idle)
            if moved.displacement_cm <= 1.0e-6:
                ledger.add(Rejection("target_route_static_for_dual_motion"))
                continue
            target_xy = moved.at(query_frame)
            target_distance = math.dist(camera, target_xy)
            if _too_close(camera, target_xy, params):
                ledger.add(Rejection("camera_too_close_to_target"))
                continue
        target_azimuth = relative_azimuth_deg(camera, yaw, target_xy)
        if abs(target_azimuth) > half_fov:
            ledger.add(Rejection("target_outside_fov_at_binding_instant"))
            continue

        def acceptable_other(candidate):
            """The second actor's checks; the farther distance when they pass."""
            if candidate.route_id == route.route_id:
                return None
            if candidate.displacement_cm <= 1.0e-6:
                return None
            if target_moves_more is not None:
                observed = moved.displacement_cm > candidate.displacement_cm
                if (math.isclose(moved.displacement_cm,
                                 candidate.displacement_cm, abs_tol=1e-6)
                        or observed != target_moves_more):
                    return None
            other_xy = candidate.at(query_frame)
            distance = math.dist(camera, other_xy)
            if distance - target_distance < min_gap:
                return None
            if _too_close(camera, other_xy, params):
                return None
            other_azimuth = relative_azimuth_deg(camera, yaw, other_xy)
            if abs(other_azimuth) > half_fov:
                return None
            if circular_gap_deg(target_azimuth, other_azimuth) < min_sep:
                return None
            return distance

        other = None
        other_distance = None
        for index in rng.permutation(n_routes)[:64]:
            ledger.stand_points_evaluated += 1
            candidate = scene.routes[int(index)]
            distance = acceptable_other(candidate)
            if distance is not None:
                other, other_distance = candidate, distance
                break
        if other is None and synth is not None:
            far_cm = synth.settings.max_camera_distance_cm
            if target_distance + min_gap < far_cm:
                designed = _synthesize_other(
                    synth, rng, ledger, camera, yaw,
                    [PointSpec(query_frame, *_design_band(None, half_fov),
                               target_distance + min_gap, far_cm,
                               exclusions=((target_azimuth, min_sep + DESIGN_EDGE_MARGIN_DEG),))],
                    lambda candidate: acceptable_other(candidate) is not None)
                if designed is not None:
                    other, other_distance = designed, acceptable_other(designed)
        if other is None:
            ledger.add(Rejection(
                "no_farther_second_actor",
                f"no moving second route is at least {min_gap:.1f} cm farther "
                "while satisfying view, separation and motion-rank constraints"))
            continue
        if scene.line_of_sight is not None:
            if not scene.line_of_sight(camera, target_xy):
                ledger.add(Rejection("target_occluded_at_binding_instant"))
                continue
            if not scene.line_of_sight(camera, other.at(query_frame)):
                ledger.add(Rejection("other_actor_occluded"))
                continue
        return PointPlan(
            scene_id=scene.scene_id, profile_id=profile_id,
            camera_xy=camera, camera_ue_yaw_deg=yaw,
            camera_height_m=clearance["camera_height_m"], camera_clearance=clearance,
            target_route=moved, base_route=route, other_route=other,
            idle_frames=idle, anchor_frame=query_frame,
            query_frame=query_frame,
            answer_cell={"kind": "distance_at_query",
                         "target_distance_cm": target_distance,
                         "other_distance_cm": other_distance},
            checks={
                "distance_gap_cm": other_distance - target_distance,
                "minimum_distance_gap_cm": min_gap,
                "line_of_sight_screened": scene.line_of_sight_screened,
                "route_sources": _route_sources(route, other),
                "search_attempts": attempt,
            },
        )
    ledger.budget_exhausted += 1
    return Rejection("no_candidate_within_attempt_budget",
                     f"{total_attempts} attempts ({bank_attempts} bank, "
                     f"{total_attempts - bank_attempts} designed)")




def solve_distance_change_pair(scene: SceneInputs, params: dict, *,
                               start_frame: int, end_frame: int,
                               target_relation: str, profile_id: str,
                               idle_choices: Iterable[int], rng,
                               ledger: RejectionLedger,
                               target_moves_more: bool | None = None,
                               min_change_cm: float = 50.0,
                               max_attempts: int = 4000):
    """Find opposite target/distractor distance trends over one time window."""
    frame_count = scene_frame_count(scene, params)
    if not 0 <= start_frame < end_frame < frame_count:
        raise ValueError(
            f"invalid distance window {start_frame}..{end_frame}")
    if target_relation not in ("closer", "farther"):
        raise ValueError(f"unknown distance relation {target_relation!r}")
    min_change = float(min_change_cm)
    half_fov = effective_half_fov(scene, params)
    min_sep = float(params["MIN_AZIMUTH_SEP"])
    pool = target_route_pool(scene, params)
    n_routes, n_cams = len(pool), len(scene.camera_points)

    def relation(delta):
        if delta <= -min_change:
            return "closer"
        if delta >= min_change:
            return "farther"
        return None

    synth = route_synthesizer(scene, params)
    bank_attempts, total_attempts = attempt_budgets(synth, max_attempts)

    def screen_target(camera, moved):
        """Distance floor at both frames and the allocated relation; None with
        a ledger entry when violated (same for bank and designed routes)."""
        target_start = moved.at(start_frame)
        target_end = moved.at(end_frame)
        if (_too_close(camera, target_start, params)
                or _too_close(camera, target_end, params)):
            ledger.add(Rejection("camera_too_close_to_target"))
            return None
        target_delta = math.dist(camera, target_end) - math.dist(camera, target_start)
        if relation(target_delta) != target_relation:
            ledger.add(Rejection(
                "target_distance_relation_mismatch",
                f"delta {target_delta:.1f} cm does not satisfy "
                f"{target_relation} by {min_change:.1f} cm"))
            return None
        return target_delta

    for attempt in range(1, total_attempts + 1):
        ledger.note_combination()
        if synth is None or attempt <= bank_attempts:
            ledger.synthesis["bank_attempts"] += 1
            route = pool[int(rng.integers(n_routes))]
            idle = int(rng.choice(list(idle_choices)))
            moved = route.shifted(idle)
            if moved.displacement_cm <= 1.0e-6:
                ledger.add(Rejection("target_route_static_for_dual_motion"))
                continue
            camera = scene.camera_points[int(rng.integers(n_cams))]
            target_delta = screen_target(camera, moved)
            if target_delta is None:
                continue
            picked = sample_clear_yaw(scene, params, camera, -180.0, 180.0, rng, ledger)
            if picked is None:
                continue
            yaw, clearance = picked
        else:
            ledger.synthesis["synthesized_attempts"] += 1
            pose = _synthesis_pose(scene, params, rng, ledger)
            if pose is None:
                continue
            camera, yaw, clearance = pose
            idle = int(rng.choice(list(idle_choices)))
            floor_cm = _distance_floor_cm(params)
            far_cm = synth.settings.max_camera_distance_cm
            route = _design_target(synth, rng, ledger, camera, yaw, [
                PointSpec(start_frame, *_design_band(None, half_fov), floor_cm, far_cm),
                PointSpec(end_frame, *_design_band(None, half_fov), floor_cm, far_cm)], idle)
            if route is None:
                continue
            moved = route.shifted(idle)
            if moved.displacement_cm <= 1.0e-6:
                ledger.add(Rejection("target_route_static_for_dual_motion"))
                continue
            target_delta = screen_target(camera, moved)
            if target_delta is None:
                continue
        target_azimuths = [
            relative_azimuth_deg(camera, yaw, moved.at(frame))
            for frame in (start_frame, end_frame)]
        if any(abs(value) > half_fov for value in target_azimuths):
            ledger.add(Rejection("target_outside_fov_at_relation_frames"))
            continue
        expected_other = "farther" if target_relation == "closer" else "closer"

        def acceptable_other(candidate):
            """The second actor's checks; its distance change when they pass."""
            if candidate.route_id == route.route_id:
                return None
            if candidate.displacement_cm <= 1.0e-6:
                return None
            if target_moves_more is not None:
                observed = moved.displacement_cm > candidate.displacement_cm
                if (math.isclose(moved.displacement_cm,
                                 candidate.displacement_cm, abs_tol=1e-6)
                        or observed != target_moves_more):
                    return None
            other_start = candidate.at(start_frame)
            other_end = candidate.at(end_frame)
            if (_too_close(camera, other_start, params)
                    or _too_close(camera, other_end, params)):
                return None
            delta = (math.dist(camera, other_end)
                     - math.dist(camera, other_start))
            if relation(delta) != expected_other:
                return None
            other_azimuths = [
                relative_azimuth_deg(camera, yaw, candidate.at(frame))
                for frame in (start_frame, end_frame)]
            if any(abs(value) > half_fov for value in other_azimuths):
                return None
            if any(circular_gap_deg(target, distractor) < min_sep
                   for target, distractor in zip(
                       target_azimuths, other_azimuths)):
                return None
            return delta

        other = None
        other_delta = None
        for index in rng.permutation(n_routes)[:64]:
            ledger.stand_points_evaluated += 1
            candidate = scene.routes[int(index)]
            delta = acceptable_other(candidate)
            if delta is not None:
                other, other_delta = candidate, delta
                break
        if other is None and synth is not None:
            floor_cm = _distance_floor_cm(params)
            far_cm = synth.settings.max_camera_distance_cm
            designed = _synthesize_other(
                synth, rng, ledger, camera, yaw,
                [PointSpec(start_frame, *_design_band(None, half_fov), floor_cm, far_cm,
                           exclusions=((target_azimuths[0], min_sep + DESIGN_EDGE_MARGIN_DEG),)),
                 PointSpec(end_frame, *_design_band(None, half_fov), floor_cm, far_cm,
                           exclusions=((target_azimuths[1], min_sep + DESIGN_EDGE_MARGIN_DEG),))],
                lambda candidate: acceptable_other(candidate) is not None)
            if designed is not None:
                other, other_delta = designed, acceptable_other(designed)
        if other is None:
            ledger.add(Rejection(
                "no_opposite_distance_trend_actor",
                f"no second moving route changes distance as {expected_other} "
                "while satisfying view, separation and motion-rank constraints"))
            continue
        if scene.line_of_sight is not None:
            if not all(scene.line_of_sight(camera, moved.at(frame))
                       for frame in (start_frame, end_frame)):
                ledger.add(Rejection("target_occluded_at_relation_frames"))
                continue
            if not all(scene.line_of_sight(camera, other.at(frame))
                       for frame in (start_frame, end_frame)):
                ledger.add(Rejection("other_actor_occluded"))
                continue
        return PointPlan(
            scene_id=scene.scene_id, profile_id=profile_id,
            camera_xy=camera, camera_ue_yaw_deg=yaw,
            camera_height_m=clearance["camera_height_m"], camera_clearance=clearance,
            target_route=moved, base_route=route, other_route=other,
            idle_frames=idle, anchor_frame=start_frame,
            query_frame=end_frame,
            answer_cell={"kind": "distance_change",
                         "relation": target_relation,
                         "target_delta_cm": target_delta,
                         "other_delta_cm": other_delta},
            checks={
                "distance_window_frames": [start_frame, end_frame],
                "target_distance_delta_cm": target_delta,
                "other_distance_delta_cm": other_delta,
                "minimum_distance_change_cm": min_change,
                "line_of_sight_screened": scene.line_of_sight_screened,
                "route_sources": _route_sources(route, other),
                "search_attempts": attempt,
            },
        )
    ledger.budget_exhausted += 1
    return Rejection("no_candidate_within_attempt_budget",
                     f"{total_attempts} attempts ({bank_attempts} bank, "
                     f"{total_attempts - bank_attempts} designed)")


def solve_motion_state_pair(scene: SceneInputs, params: dict, *,
                            start_frame: int, end_frame: int,
                            target_state: str, profile_id: str,
                            idle_choices: Iterable[int], rng,
                            ledger: RejectionLedger,
                            min_motion_cm: float = 10.0,
                            max_attempts: int = 4000):
    """Find opposite moving/still roles over one declared frame window."""
    frame_count = scene_frame_count(scene, params)
    if not 0 <= start_frame < end_frame < frame_count:
        raise ValueError(f"invalid motion window {start_frame}..{end_frame}")
    if target_state not in ("moving", "still"):
        raise ValueError(f"unknown motion state {target_state!r}")
    opposite = "still" if target_state == "moving" else "moving"
    half_fov = effective_half_fov(scene, params)
    min_sep = float(params["MIN_AZIMUTH_SEP"])
    minimum = float(min_motion_cm)
    n_routes, n_cams = len(scene.routes), len(scene.camera_points)

    def apply_state(route, state):
        return route if state == "moving" else route.paused(
            start_frame, end_frame)

    def window_displacement(route):
        return math.dist(route.at(start_frame), route.at(end_frame))

    def state_matches(route, state):
        displacement = window_displacement(route)
        return (displacement >= minimum if state == "moving"
                else displacement <= 1.0e-6)

    synth = route_synthesizer(scene, params)
    bank_attempts, total_attempts = attempt_budgets(synth, max_attempts)

    def screen_target(shifted):
        """Full-clip motion and the allocated window state; None with a ledger
        entry when violated (same for bank and designed routes)."""
        target = apply_state(shifted, target_state)
        if target.displacement_cm <= 1.0e-6:
            ledger.add(Rejection("target_route_static_over_full_clip"))
            return None
        if not state_matches(target, target_state):
            ledger.add(Rejection("target_motion_state_mismatch"))
            return None
        return target

    for attempt in range(1, total_attempts + 1):
        ledger.note_combination()
        if synth is None or attempt <= bank_attempts:
            ledger.synthesis["bank_attempts"] += 1
            base = scene.routes[int(rng.integers(n_routes))]
            idle = int(rng.choice(list(idle_choices)))
            target = screen_target(base.shifted(idle))
            if target is None:
                continue
            camera = scene.camera_points[int(rng.integers(n_cams))]
            target_points = [target.at(frame)
                             for frame in (start_frame, end_frame)]
            if any(_too_close(camera, point, params) for point in target_points):
                ledger.add(Rejection("camera_too_close_to_target"))
                continue
            picked = sample_clear_yaw(scene, params, camera, -180.0, 180.0, rng, ledger)
            if picked is None:
                continue
            yaw, clearance = picked
        else:
            ledger.synthesis["synthesized_attempts"] += 1
            pose = _synthesis_pose(scene, params, rng, ledger)
            if pose is None:
                continue
            camera, yaw, clearance = pose
            idle = int(rng.choice(list(idle_choices)))
            base = _design_target(synth, rng, ledger, camera, yaw, [
                PointSpec(start_frame, *_design_band(None, half_fov),
                          _distance_floor_cm(params), synth.settings.max_camera_distance_cm)],
                idle)
            if base is None:
                continue
            target = screen_target(base.shifted(idle))
            if target is None:
                continue
            target_points = [target.at(frame)
                             for frame in (start_frame, end_frame)]
            if any(_too_close(camera, point, params) for point in target_points):
                ledger.add(Rejection("camera_too_close_to_target"))
                continue
        target_azimuths = [
            relative_azimuth_deg(camera, yaw, point)
            for point in target_points]
        if any(abs(value) > half_fov for value in target_azimuths):
            ledger.add(Rejection("target_outside_fov_at_motion_frames"))
            continue

        def acceptable_other(candidate_base):
            """The second actor's checks; the state-applied route when they pass."""
            if candidate_base.route_id == base.route_id:
                return None
            candidate = apply_state(candidate_base, opposite)
            if candidate.displacement_cm <= 1.0e-6:
                return None
            if not state_matches(candidate, opposite):
                return None
            other_points = [candidate.at(frame)
                            for frame in (start_frame, end_frame)]
            if any(_too_close(camera, point, params)
                   for point in other_points):
                return None
            other_azimuths = [
                relative_azimuth_deg(camera, yaw, point)
                for point in other_points]
            if any(abs(value) > half_fov for value in other_azimuths):
                return None
            if any(circular_gap_deg(target_az, other_az) < min_sep
                   for target_az, other_az in zip(
                       target_azimuths, other_azimuths)):
                return None
            return candidate

        other = None
        for index in rng.permutation(n_routes)[:64]:
            ledger.stand_points_evaluated += 1
            other = acceptable_other(scene.routes[int(index)])
            if other is not None:
                break
        if other is None and synth is not None:
            designed = _synthesize_other(
                synth, rng, ledger, camera, yaw,
                [PointSpec(start_frame, *_design_band(None, half_fov),
                           _distance_floor_cm(params), synth.settings.max_camera_distance_cm,
                           exclusions=((target_azimuths[0], min_sep + DESIGN_EDGE_MARGIN_DEG),))],
                lambda candidate_base: acceptable_other(candidate_base) is not None)
            if designed is not None:
                other = acceptable_other(designed)
        if other is None:
            ledger.add(Rejection(
                "no_opposite_motion_state_actor",
                f"no second route is {opposite} in frames "
                f"{start_frame}..{end_frame} while satisfying view/separation"))
            continue
        if scene.line_of_sight is not None:
            if not all(scene.line_of_sight(camera, target.at(frame))
                       for frame in (start_frame, end_frame)):
                ledger.add(Rejection("target_occluded_at_motion_frames"))
                continue
            if not all(scene.line_of_sight(camera, other.at(frame))
                       for frame in (start_frame, end_frame)):
                ledger.add(Rejection("other_actor_occluded"))
                continue
        return PointPlan(
            scene_id=scene.scene_id, profile_id=profile_id,
            camera_xy=camera, camera_ue_yaw_deg=yaw,
            camera_height_m=clearance["camera_height_m"], camera_clearance=clearance,
            target_route=target, base_route=base, other_route=other,
            idle_frames=idle, anchor_frame=start_frame,
            query_frame=end_frame,
            answer_cell={"kind": "motion_state",
                         "state": target_state,
                         "target_window_displacement_cm":
                             window_displacement(target),
                         "other_window_displacement_cm":
                             window_displacement(other)},
            checks={
                "motion_window_frames": [start_frame, end_frame],
                "minimum_motion_cm": minimum,
                "target_state": target_state,
                "other_state": opposite,
                "uses_solved_route_samples_directly": True,
                "line_of_sight_screened": scene.line_of_sight_screened,
                "route_sources": _route_sources(base, other),
                "search_attempts": attempt,
            },
        )
    ledger.budget_exhausted += 1
    return Rejection("no_candidate_within_attempt_budget",
                     f"{total_attempts} attempts ({bank_attempts} bank, "
                     f"{total_attempts - bank_attempts} designed)")
def solve_instant_binding(scene: SceneInputs, params: dict, *,
                          instants: Sequence[int], profile_id: str,
                          idle_choices: Iterable[int], rng,
                          ledger: RejectionLedger,
                          target_moves_more: bool | None = None,
                          max_attempts: int = 4000):
    """⑦/⑨ 这类**即时绑定**题的布局:不要求错时,只要求在关键时刻可绑定。

    关键时刻(⑦ 是查询帧,⑨ 是两次首叫所在帧)上,两个角色都必须在
    视锥内、且方位分离达标 —— 否则"哪一只在叫 / 谁先叫"没法由音频方向
    绑到画面里的个体。刻意不施加锚后角位移那条约束:那是错时族的题眼,
    对即时绑定题是多余的限制。
    """
    frame_count = scene_frame_count(scene, params)
    instants = tuple(_require_frame(frame, frame_count, name="binding_frame")
                     for frame in instants)
    if not instants:
        raise ValueError("binding instants must not be empty")
    half_fov = effective_half_fov(scene, params)
    min_sep = float(params["MIN_AZIMUTH_SEP"])
    pool = target_route_pool(scene, params)
    n_routes, n_cams = len(pool), len(scene.camera_points)
    synth = route_synthesizer(scene, params)
    bank_attempts, total_attempts = attempt_budgets(synth, max_attempts)
    for attempt in range(1, total_attempts + 1):
        ledger.note_combination()
        if synth is None or attempt <= bank_attempts:
            ledger.synthesis["bank_attempts"] += 1
            route = pool[int(rng.integers(n_routes))]
            idle = int(rng.choice(list(idle_choices)))
            moved = route.shifted(idle)
            if moved.displacement_cm <= 1.0e-6:
                ledger.add(Rejection("target_route_static_for_dual_motion"))
                continue
            camera = scene.camera_points[int(rng.integers(n_cams))]
            if _too_close(camera, moved.at(instants[0]), params):
                ledger.add(Rejection("camera_too_close_to_target"))
                continue
            picked = sample_clear_yaw(scene, params, camera, -180.0, 180.0, rng, ledger)
            if picked is None:
                continue
            yaw, clearance = picked
        else:
            ledger.synthesis["synthesized_attempts"] += 1
            pose = _synthesis_pose(scene, params, rng, ledger)
            if pose is None:
                continue
            camera, yaw, clearance = pose
            idle = int(rng.choice(list(idle_choices)))
            route = _design_target(synth, rng, ledger, camera, yaw, [
                PointSpec(int(instants[0]), *_design_band(None, half_fov),
                          _distance_floor_cm(params), synth.settings.max_camera_distance_cm)],
                idle)
            if route is None:
                continue
            moved = route.shifted(idle)
            if moved.displacement_cm <= 1.0e-6:
                ledger.add(Rejection("target_route_static_for_dual_motion"))
                continue
            if _too_close(camera, moved.at(instants[0]), params):
                ledger.add(Rejection("camera_too_close_to_target"))
                continue
        azimuths = [relative_azimuth_deg(camera, yaw, moved.at(f))
                    for f in instants]
        if any(abs(a) > half_fov for a in azimuths):
            ledger.add(Rejection("target_outside_fov_at_binding_instant"))
            continue

        def acceptable_other(candidate) -> bool:
            if candidate.route_id == route.route_id:
                return False
            if candidate.displacement_cm <= 1.0e-6:
                return False
            if target_moves_more is not None:
                observed = moved.displacement_cm > candidate.displacement_cm
                if math.isclose(moved.displacement_cm,
                                candidate.displacement_cm, abs_tol=1e-6) or \
                        observed != target_moves_more:
                    return False
            other_azimuths = [relative_azimuth_deg(
                camera, yaw, candidate.at(frame)) for frame in instants]
            if any(abs(value) > half_fov for value in other_azimuths):
                return False
            if any(circular_gap_deg(other, target) < min_sep
                   for other, target in zip(other_azimuths, azimuths)):
                return False
            if any(_too_close(camera, candidate.at(frame), params)
                   for frame in instants):
                return False
            return True

        other = None
        order = rng.permutation(len(scene.routes))
        for index in order[:64]:
            ledger.stand_points_evaluated += 1
            candidate = scene.routes[int(index)]
            if acceptable_other(candidate):
                other = candidate
                break
        if other is None and synth is not None:
            other = _synthesize_other(
                synth, rng, ledger, camera, yaw,
                [PointSpec(int(instants[0]), *_design_band(None, half_fov),
                           _distance_floor_cm(params), synth.settings.max_camera_distance_cm,
                           exclusions=((azimuths[0], min_sep + DESIGN_EDGE_MARGIN_DEG),))],
                acceptable_other)
        if other is None:
            ledger.add(Rejection("no_separable_second_actor",
                                 "no stand point stays in view and separated "
                                 "at every binding instant"))
            continue
        other_azimuths = [relative_azimuth_deg(camera, yaw, other.at(frame))
                          for frame in instants]
        if scene.line_of_sight is not None:
            if not all(scene.line_of_sight(camera, moved.at(f))
                       for f in instants):
                ledger.add(Rejection("target_occluded_at_binding_instant"))
                continue
            if not all(scene.line_of_sight(camera, other.at(frame))
                       for frame in instants):
                ledger.add(Rejection("other_actor_occluded"))
                continue
        return PointPlan(
            scene_id=scene.scene_id, profile_id=profile_id, camera_xy=camera,
            camera_ue_yaw_deg=yaw, camera_height_m=clearance["camera_height_m"],
            camera_clearance=clearance, target_route=moved, base_route=route,
            other_route=other, idle_frames=idle,
            anchor_frame=int(instants[-1]), query_frame=int(instants[0]),
            answer_cell={"kind": "binding_instants",
                         "instants": [int(f) for f in instants]},
            checks={"binding_azimuths_deg": [round(a, 2) for a in azimuths],
                    "min_separation_deg": round(
                        min(circular_gap_deg(other_az, target_az)
                            for other_az, target_az in zip(
                                other_azimuths, azimuths)), 2),
                    "line_of_sight_screened": scene.line_of_sight_screened,
                    "route_sources": _route_sources(route, other),
                    "search_attempts": attempt},
        )
    ledger.budget_exhausted += 1
    return Rejection("no_candidate_within_attempt_budget",
                     f"{total_attempts} attempts ({bank_attempts} bank, "
                     f"{total_attempts - bank_attempts} designed)")

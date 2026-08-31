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
  - routes:导航系统给的路线,每条自带 75 帧重采样位置;
  - stand_points:可站点(路线路点即导航可达点,去重后即可);
  - camera_points:相机候选位置(同上,取相机高度);
  - line_of_sight:可选的占用栅格视线筛查;没有就如实声明未筛,
    交给渲染后的像素真值复核。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

FRAME_COUNT = 75


def bearing_deg(origin: Sequence[float], point: Sequence[float]) -> float:
    """世界系下 origin→point 的方位角(度),与 UE yaw 同一参照。"""
    return math.degrees(math.atan2(point[1] - origin[1], point[0] - origin[0]))


def relative_azimuth_deg(camera_xy, camera_yaw_deg: float, point_xy) -> float:
    """右为正的相对方位;与生产侧 side_of 的叉积符号一致。"""
    delta = bearing_deg(camera_xy, point_xy) - camera_yaw_deg
    return (delta + 180.0) % 360.0 - 180.0


def circular_gap_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def yaw_interval_for_band(camera_xy, point_xy, band_lo: float,
                          band_hi: float) -> tuple[float, float]:
    """解出使 point 的相对方位落进 [band_lo, band_hi) 的 yaw 区间。

    这是本模块的关键:答案带是解出来的,不是枚举撞出来的。
    """
    bearing = bearing_deg(camera_xy, point_xy)
    return (bearing - band_hi, bearing - band_lo)


@dataclass
class Route:
    route_id: str
    samples_xy: list[tuple[float, float]]      # 75 帧位置
    implied_speed_mps: float

    def at(self, frame: int) -> tuple[float, float]:
        return self.samples_xy[max(0, min(FRAME_COUNT - 1, frame))]

    def shifted(self, idle_frames: int) -> "Route":
        """静→走:前 idle_frames 帧停在起点,其后沿用原速度(平移式)。"""
        if idle_frames <= 0:
            return self
        shifted = ([self.samples_xy[0]] * idle_frames
                   + self.samples_xy[:FRAME_COUNT - idle_frames])
        return Route(f"{self.route_id}+idle{idle_frames}", shifted,
                     self.implied_speed_mps)


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

    @property
    def line_of_sight_screened(self) -> bool:
        return self.line_of_sight is not None


# 路线库适配器按**库自己的 schema** 分派,不按房间 ID。每个后端声明自己的
# 水平面约定:UE 的地平面是 (x, y)、单位厘米;habitat 的是 (x, z)、单位米。
# 这是后端坐标约定,不是房间特例。
BANK_ADAPTERS: dict[str, str] = {
    "avengine_apartment_route_bank_v1": "ue_route_bank",
    "avengine_room_trajectory_bank_v2": "habitat_trajectory_bank",
}


def _routes_from_ue_route_bank(bank: dict, limit: int | None) -> list[Route]:
    routes: list[Route] = []
    for entry in bank.get("routes", []):
        samples = entry.get("samples_ue_cm")
        if not samples or len(samples) != FRAME_COUNT:
            continue
        routes.append(Route(str(entry["route_id"]),
                            [(float(p[0]), float(p[1])) for p in samples],
                            float(entry.get("implied_speed_mps", 0.0))))
        if limit and len(routes) >= limit:
            break
    return routes


def _routes_from_habitat_trajectory_bank(bank: dict,
                                         limit: int | None) -> list[Route]:
    """habitat 逐槽位路径 → 统一 Route(米→厘米,取 (x, z) 水平面)。

    每个 episode 的每个槽位都是一条独立可用的路线:采样器只关心"一条
    75 帧的可行轨迹",谁跟谁配对由题型求解阶段重新决定。
    """
    routes: list[Route] = []
    for episode in bank.get("episodes", []):
        paths = episode.get("source_center_paths_m") or {}
        for slot, path in sorted(paths.items()):
            if not path or len(path) != FRAME_COUNT:
                continue
            xy = [(float(p[0]) * 100.0, float(p[2]) * 100.0) for p in path]
            span = math.dist(xy[0], xy[-1]) / 100.0
            routes.append(Route(f"{episode['episode_id']}:{slot}", xy,
                                span / 5.0))
            if limit and len(routes) >= limit:
                return routes
    return routes


def routes_from_bank(bank: dict, limit: int | None = None) -> list[Route]:
    schema = str(bank.get("schema", ""))
    adapter = BANK_ADAPTERS.get(schema)
    if adapter is None:
        raise ValueError(
            f"no route-bank adapter for schema {schema!r}; add one adapter "
            "keyed by the bank schema (never by a room id)")
    if adapter == "ue_route_bank":
        return _routes_from_ue_route_bank(bank, limit)
    return _routes_from_habitat_trajectory_bank(bank, limit)


def load_scene(config: dict, *, route_limit: int | None = None) -> SceneInputs:
    """从场景配置载入场景无关输入。

    config 只允许指向场景自身的产物(导航路线库、相机基准请求),不允许
    携带逐点坐标或手填路线;缺项即拒绝载入,而不是回退到某个房间的默认。
    """
    required = ("scene_id", "backend", "route_bank", "camera_base_request")
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"scene config missing keys: {missing}")
    bank = json.loads(Path(config["route_bank"]).read_text())
    routes = routes_from_bank(bank, route_limit)
    if not routes:
        raise ValueError(f"{config['scene_id']}: route bank yielded no usable "
                         "75-frame routes")
    # 可站点与相机候选都取导航可达点(路线端点与路点),去重后使用:
    # 这些点由场景导航系统背书,不是手填坐标。
    points: set[tuple[float, float]] = set()
    for route in routes:
        points.add(_round_xy(route.samples_xy[0]))
        points.add(_round_xy(route.samples_xy[-1]))
        points.add(_round_xy(route.samples_xy[FRAME_COUNT // 2]))
    ordered = sorted(points)
    base_request = json.loads(Path(config["camera_base_request"]).read_text())
    height = float(config.get("camera_height_m")
                   or _listener_height_m(base_request))
    hfov = float(config.get("hfov_deg") or _hfov_deg(base_request))
    if not (0.0 < hfov < 180.0):
        raise ValueError(f"{config['scene_id']}: implausible hfov {hfov}")
    return SceneInputs(
        scene_id=str(config["scene_id"]), backend=str(config["backend"]),
        routes=routes, stand_points=ordered, camera_points=ordered,
        camera_height_m=height, hfov_deg=hfov,
        provenance={"route_bank": config["route_bank"],
                    "route_bank_schema": bank.get("schema"),
                    "route_bank_source": bank.get("source"),
                    "bank_adapter": BANK_ADAPTERS.get(str(bank.get("schema"))),
                    "camera_base_request": config["camera_base_request"],
                    "routes_loaded": len(routes),
                    "navigable_points": len(ordered)},
    )


def _round_xy(xy) -> tuple[float, float]:
    return (round(float(xy[0]), 1), round(float(xy[1]), 1))


def _listener_height_m(request: dict) -> float:
    rig = request["primary_camera_rig"]["world_from_rig"]["translation_m"]
    listener = request["listener"]["rig_from_listener"]["translation_m"]
    return float(rig[1]) + float(listener[1])


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
    target_route: Route
    other_point: tuple[float, float]
    idle_frames: int
    anchor_frame: int
    query_frame: int
    answer_cell: dict
    checks: dict


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

    def add(self, rejection: Rejection) -> None:
        self.counts[rejection.reason] = self.counts.get(rejection.reason, 0) + 1
        self.first_details.setdefault(rejection.reason, rejection.detail)

    def note_combination(self) -> None:
        self.combinations_evaluated += 1

    def summary(self) -> dict:
        total = sum(self.counts.values())
        return {"total": total,
                "combinations_evaluated": self.combinations_evaluated,
                "stand_points_evaluated": self.stand_points_evaluated,
                "budget_exhausted": self.budget_exhausted,
                "by_reason": dict(sorted(self.counts.items(),
                                         key=lambda kv: -kv[1])),
                "first_example": self.first_details}


def solve_forward_cross_time(scene: SceneInputs, params: dict, *,
                             answer_band: tuple[float, float],
                             anchor_frame: int, idle_choices: Iterable[int],
                             rng, ledger: RejectionLedger,
                             max_attempts: int = 4000) -> PointPlan | Rejection:
    """①F 正向错时:音频锚在前,视觉查询在后(查询帧=片尾)。

    约束(全部由题型声明,与房间无关):
      - 目标片尾方位落进指定答案带(解 yaw,不枚举);
      - 锚定时刻到查询时刻目标角位移 > THETA_FULL(否则锚时即可读答案);
      - 锚定时刻两角色方位分离 >= MIN_AZIMUTH_SEP(锚可绑定);
      - 锚定时刻与查询时刻目标都在视锥内;
      - 另一角色不得与目标同答案带(否则选项无区分度);
      - 有视线筛查就用,没有就如实记未筛。
    """
    half_fov = scene.hfov_deg / 2.0
    theta_full = float(params["THETA_FULL"])
    min_sep = float(params["MIN_AZIMUTH_SEP"])
    band_lo, band_hi = answer_band
    n_routes, n_cams = len(scene.routes), len(scene.camera_points)
    for attempt in range(1, max_attempts + 1):
        ledger.note_combination()
        route = scene.routes[int(rng.integers(n_routes))]
        idle = int(rng.choice(list(idle_choices)))
        moved = route.shifted(idle)
        camera = scene.camera_points[int(rng.integers(n_cams))]
        end_xy = moved.at(FRAME_COUNT - 1)
        if _too_close(camera, end_xy, params):
            ledger.add(Rejection("camera_too_close_to_target"))
            continue
        lo_yaw, hi_yaw = yaw_interval_for_band(camera, end_xy, band_lo, band_hi)
        yaw = float(lo_yaw + (hi_yaw - lo_yaw) * rng.random())
        az_end = relative_azimuth_deg(camera, yaw, end_xy)
        if not (band_lo <= az_end < band_hi):
            ledger.add(Rejection("band_solution_out_of_band",
                                 f"az={az_end:.2f} band={band_lo},{band_hi}"))
            continue
        if abs(az_end) > half_fov:
            ledger.add(Rejection("answer_band_outside_fov",
                                 "the declared band lies outside the camera "
                                 "field of view"))
            continue
        anchor_xy = moved.at(anchor_frame)
        az_anchor = relative_azimuth_deg(camera, yaw, anchor_xy)
        if abs(az_anchor) > half_fov:
            ledger.add(Rejection("target_outside_fov_at_anchor"))
            continue
        if circular_gap_deg(az_anchor, az_end) <= theta_full:
            ledger.add(Rejection("insufficient_azimuth_travel_after_anchor",
                                 f"{circular_gap_deg(az_anchor, az_end):.1f} "
                                 f"<= THETA_FULL {theta_full}"))
            continue
        other = _pick_other_point(scene, camera, yaw, az_anchor, az_end,
                                  band_lo, band_hi, min_sep, half_fov, rng,
                                  ledger)
        if other is None:
            continue
        if scene.line_of_sight is not None:
            if not scene.line_of_sight(camera, end_xy):
                ledger.add(Rejection("target_occluded_at_query_frame"))
                continue
            if not scene.line_of_sight(camera, other):
                ledger.add(Rejection("other_actor_occluded"))
                continue
        return PointPlan(
            scene_id=scene.scene_id, profile_id="card1F", camera_xy=camera,
            camera_ue_yaw_deg=yaw, target_route=moved, other_point=other,
            idle_frames=idle, anchor_frame=anchor_frame,
            query_frame=FRAME_COUNT - 1,
            answer_cell={"kind": "azimuth_band", "band": [band_lo, band_hi],
                         "value_deg": az_end},
            checks={"az_anchor_deg": az_anchor, "az_end_deg": az_end,
                    "azimuth_travel_deg": circular_gap_deg(az_anchor, az_end),
                    "anchor_separation_deg": circular_gap_deg(
                        az_anchor, relative_azimuth_deg(camera, yaw, other)),
                    "line_of_sight_screened": scene.line_of_sight_screened,
                    "search_attempts": attempt},
        )
    ledger.budget_exhausted += 1
    return Rejection("no_candidate_within_attempt_budget",
                     f"{max_attempts} attempts")


def solve_backward_cross_time(scene: SceneInputs, params: dict, *,
                              answer_band: tuple[float, float],
                              anchor_frame: int, query_frame: int,
                              idle_choices: Iterable[int], rng,
                              ledger: RejectionLedger,
                              max_attempts: int = 4000) -> PointPlan | Rejection:
    """①B 反向错时:视觉查询在前,音频锚在后(末段发声确定身份)。

    与 ①F 的差别只在时间方向,几何约束同构:查询帧的目标状态必须可观察,
    锚定时刻两角色可分辨,且**查询时刻附近不得有直接泄露答案的音频**
    (由 AudioProgram profile 保证,这里只声明并记录该要求)。
    """
    half_fov = scene.hfov_deg / 2.0
    theta_full = float(params["THETA_FULL"])
    min_sep = float(params["MIN_AZIMUTH_SEP"])
    band_lo, band_hi = answer_band
    n_routes, n_cams = len(scene.routes), len(scene.camera_points)
    for attempt in range(1, max_attempts + 1):
        ledger.note_combination()
        route = scene.routes[int(rng.integers(n_routes))]
        idle = int(rng.choice(list(idle_choices)))
        moved = route.shifted(idle)
        camera = scene.camera_points[int(rng.integers(n_cams))]
        query_xy = moved.at(query_frame)
        if _too_close(camera, query_xy, params):
            ledger.add(Rejection("camera_too_close_to_target"))
            continue
        lo_yaw, hi_yaw = yaw_interval_for_band(camera, query_xy, band_lo, band_hi)
        yaw = float(lo_yaw + (hi_yaw - lo_yaw) * rng.random())
        az_query = relative_azimuth_deg(camera, yaw, query_xy)
        if not (band_lo <= az_query < band_hi) or abs(az_query) > half_fov:
            ledger.add(Rejection("answer_band_outside_fov"))
            continue
        anchor_xy = moved.at(anchor_frame)
        az_anchor = relative_azimuth_deg(camera, yaw, anchor_xy)
        if abs(az_anchor) > half_fov:
            ledger.add(Rejection("target_outside_fov_at_anchor"))
            continue
        # 反向错时同样要求"查询时刻的状态不能在锚定时刻直接读到":
        # 目标必须在两个时刻之间移动够多,否则听完再看当前帧即可作答。
        if circular_gap_deg(az_anchor, az_query) <= theta_full:
            ledger.add(Rejection("insufficient_azimuth_travel_between_frames",
                                 f"{circular_gap_deg(az_anchor, az_query):.1f}"))
            continue
        other = _pick_other_point(scene, camera, yaw, az_anchor, az_query,
                                  band_lo, band_hi, min_sep, half_fov, rng,
                                  ledger)
        if other is None:
            continue
        if scene.line_of_sight is not None and not scene.line_of_sight(
                camera, query_xy):
            ledger.add(Rejection("target_occluded_at_query_frame"))
            continue
        return PointPlan(
            scene_id=scene.scene_id, profile_id="card1B", camera_xy=camera,
            camera_ue_yaw_deg=yaw, target_route=moved, other_point=other,
            idle_frames=idle, anchor_frame=anchor_frame,
            query_frame=query_frame,
            answer_cell={"kind": "azimuth_band", "band": [band_lo, band_hi],
                         "value_deg": az_query},
            checks={"az_anchor_deg": az_anchor, "az_query_deg": az_query,
                    "azimuth_travel_deg": circular_gap_deg(az_anchor, az_query),
                    "line_of_sight_screened": scene.line_of_sight_screened,
                    "requires_silence_near_query": True,
                    "search_attempts": attempt},
        )
    ledger.budget_exhausted += 1
    return Rejection("no_candidate_within_attempt_budget",
                     f"{max_attempts} attempts")


def _too_close(camera, point, params) -> bool:
    min_cm = float(params.get("MIN_CAMERA_DISTANCE_CM", 100.0))
    return math.hypot(point[0] - camera[0], point[1] - camera[1]) < min_cm


def _pick_other_point(scene, camera, yaw, az_anchor, az_answer, band_lo,
                      band_hi, min_sep, half_fov, rng, ledger):
    """另一角色:锚定时刻方位分离达标、在视锥内、且不与答案同带。"""
    order = rng.permutation(len(scene.stand_points))
    for index in order[:64]:
        ledger.stand_points_evaluated += 1
        candidate = scene.stand_points[int(index)]
        az = relative_azimuth_deg(camera, yaw, candidate)
        if abs(az) > half_fov:
            continue
        if circular_gap_deg(az, az_anchor) < min_sep:
            continue
        if band_lo <= az < band_hi:
            continue
        if _too_close(camera, candidate, {"MIN_CAMERA_DISTANCE_CM": 100.0}):
            continue
        return candidate
    ledger.add(Rejection("no_separable_second_actor",
                         "no navigable stand point clears the anchor-instant "
                         "azimuth separation while staying out of the answer "
                         "band and inside the field of view"))
    return None

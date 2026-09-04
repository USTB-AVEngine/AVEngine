#!/usr/bin/env python3
"""圆上的弧:起点加带符号扫角。有序的 [lo, hi] 表示不了它。

为什么需要一个专门的表示
------------------------
2026-09-03 加查询窗口的时候,方位真值从一个数变成了一个区间,而那个区间是用
``min(samples), max(samples)`` 记的。那是把圆上的量当直线读:样本 175, 178, -179,
-177, -175 的真实扫角是 10 度,线性读法给 358 度。同一天 Gate A 的区间分离判据也是
端点相减,于是 [172, 178] 对 [-178, -172] 这一对(圆上真实间隙 4 度)算出 344 度、
越过 2*THETA_HALF = 60 的阈值、**被判为已分离**。pilot 在真实代码路径上复现了它:
``open_separation = 344.0`` 配 ``open_gold_separated = True``。那是认证这道门的假通过,
因为两个金标实际上叠在一起,一个完全不听音频的模型两边都答对还算"通过必要性认证"。

三处当时都改成了跨 ±180 就抛,理由是产生错的认证数据比抛异常危险得多,而且当时
F2 的答案形式还没定,不该猜一个表示。owner 2026-09-03 把答案范围放开到整圈之后,
这个表示成了必需品,本模块就是它。

不变量(pilot 2026-09-04 提的四条,与最终选哪种表示无关)
--------------------------------------------------------
1. 楔形与它的补集必须不同。有序 ``[lo, hi]`` 做不到:``[170, -170)`` 排序后就是
   ``[-170, 170)``,一个是身后 20 度,一个是身前 340 度,数值一模一样。
2. 扫角保号,且允许超过 180 度,不许折回。方向本身是答案量的一部分。
3. 两个弧的分离按圆上集合算:各向外扩 ``theta_half`` 之后在圆上不相交。端点相减在
   跨界时跟这件事完全不是一回事。
4. 编码解码往返同时保住起点与带符号扫角;用例必须含跨 ±180 与扫角 > 180;补集不得判等。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

SCHEMA = "avengine_qa_v3_arc_v1"


def normalize_deg(value: float) -> float:
    """把一个角度折进 (-180, 180]。"""

    folded = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if folded == -180.0 else folded


def signed_delta_deg(frm: float, to: float) -> float:
    """从 frm 走到 to 的最短带符号转角,落在 (-180, 180]。"""

    return normalize_deg(float(to) - float(frm))


@dataclass(frozen=True)
class Arc:
    """从 ``start_deg`` 起、扫过 ``sweep_deg`` 的弧。

    ``sweep_deg`` 保号(正是朝方位增大的方向)且可以超过 360 度——一个候选真的可以在
    查询窗口里扫过 200 度,那种情况下把它折回 160 度就丢掉了信息。
    """

    start_deg: float
    sweep_deg: float

    def __post_init__(self) -> None:
        for name in ("start_deg", "sweep_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"Arc.{name} must be finite, got {value!r}")
        object.__setattr__(self, "start_deg", normalize_deg(self.start_deg))
        object.__setattr__(self, "sweep_deg", float(self.sweep_deg))

    # ── 构造 ────────────────────────────────────────────────────────────────
    @classmethod
    def from_samples(cls, samples) -> "Arc":
        """从一串逐帧方位样本还原弧,用累积的最短转角解绕。

        这是取代 ``min/max`` 的那一步:样本 175, 178, -179, -177, -175 得到
        start=175, sweep=10,而不是一个 358 度的区间。
        """

        values = [float(v) for v in samples]
        if not values:
            raise ValueError("Arc.from_samples needs at least one sample")
        unwrapped = [values[0]]
        for value in values[1:]:
            unwrapped.append(unwrapped[-1] + signed_delta_deg(unwrapped[-1], value))
        lo, hi = min(unwrapped), max(unwrapped)
        return cls(start_deg=lo, sweep_deg=hi - lo)

    @classmethod
    def from_bounds(cls, lo: float, hi: float) -> "Arc":
        """把一对本来就不跨界的边界当成弧,方便跟旧的 [lo, hi) 表示互通。"""

        lo, hi = float(lo), float(hi)
        if hi < lo:
            raise ValueError(
                f"from_bounds({lo}, {hi}): hi < lo is ambiguous on a circle; "
                "use Arc(start_deg=..., sweep_deg=...) and say which way it goes")
        return cls(start_deg=lo, sweep_deg=hi - lo)

    # ── 性质 ────────────────────────────────────────────────────────────────
    @property
    def end_deg(self) -> float:
        return normalize_deg(self.start_deg + self.sweep_deg)

    @property
    def width_deg(self) -> float:
        return abs(self.sweep_deg)

    @property
    def wraps(self) -> bool:
        """这条弧是否跨过 ±180。"""

        if self.width_deg >= 360.0:
            return True
        lo = self.start_deg
        hi = lo + self.sweep_deg
        return not (-180.0 <= min(lo, hi) and max(lo, hi) <= 180.0)

    def as_dict(self) -> dict:
        return {"schema": SCHEMA,
                "start_deg": self.start_deg,
                "sweep_deg": self.sweep_deg}

    @classmethod
    def from_dict(cls, payload) -> "Arc":
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"expected {SCHEMA}, got {payload.get('schema')!r}")
        return cls(start_deg=float(payload["start_deg"]),
                   sweep_deg=float(payload["sweep_deg"]))

    def contains(self, azimuth_deg: float) -> bool:
        """这个方位是否落在弧上(含两端)。"""

        if self.width_deg >= 360.0:
            return True
        offset = signed_delta_deg(self.start_deg, azimuth_deg)
        if self.sweep_deg >= 0.0:
            if offset < -1e-9:
                offset += 360.0
            return -1e-9 <= offset <= self.sweep_deg + 1e-9
        if offset > 1e-9:
            offset -= 360.0
        return self.sweep_deg - 1e-9 <= offset <= 1e-9

    def dilated(self, margin_deg: float) -> "Arc":
        """向两端各外扩 margin_deg。"""

        margin = float(margin_deg)
        if margin < 0.0:
            raise ValueError("dilation margin must be >= 0")
        sign = 1.0 if self.sweep_deg >= 0.0 else -1.0
        return Arc(start_deg=self.start_deg - sign * margin,
                   sweep_deg=self.sweep_deg + sign * 2.0 * margin)


def _covered(arc: Arc) -> list[tuple[float, float]]:
    """把弧摊成 [-540, 540) 上一到两段不跨界的闭区间,方便做集合运算。"""

    lo = arc.start_deg
    hi = lo + arc.sweep_deg
    lo, hi = min(lo, hi), max(lo, hi)
    if hi - lo >= 360.0:
        return [(-540.0, 540.0)]
    return [(lo + k, hi + k) for k in (-360.0, 0.0, 360.0)]


def arcs_intersect(first: Arc, second: Arc, *, tolerance: float = 1e-9) -> bool:
    """两条弧在圆上是否相交。"""

    for a_lo, a_hi in _covered(first):
        for b_lo, b_hi in _covered(second):
            if min(a_hi, b_hi) - max(a_lo, b_lo) > -tolerance:
                return True
    return False


def wide_credit_regions_disjoint(first: Arc, second: Arc,
                                 theta_half_deg: float) -> bool:
    """两个金标各外扩 theta_half 之后在圆上是否互不相交。

    这是取代端点相减的那一步。Gate A 要求两个金标的宽信区域分开,而"分开"是圆上的
    集合性质:[172, 178] 与 [-178, -172] 端点差 344 度,圆上却只隔 4 度,外扩 30 度
    之后重重叠在一起。
    """

    margin = float(theta_half_deg)
    if margin < 0.0:
        raise ValueError("theta_half_deg must be >= 0")
    return not arcs_intersect(first.dilated(margin), second.dilated(margin))


def circular_gap_deg(first: Arc, second: Arc) -> float:
    """Smallest angular gap between two arcs on the circle; 0 if they touch.

    This is the number a fact should record.  The linear endpoint arithmetic it
    replaces reported 344 degrees for a pair 4 degrees apart, so the recorded
    evidence was as wrong as the verdict drawn from it.
    """

    if arcs_intersect(first, second):
        return 0.0
    best = 360.0
    for a_lo, a_hi in _covered(first)[1:2]:
        for b_lo, b_hi in _covered(second):
            best = min(best, abs(b_lo - a_hi), abs(a_lo - b_hi))
    return min(best, 360.0 - best) if best > 180.0 else best

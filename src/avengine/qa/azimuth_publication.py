"""Every azimuth a question publishes follows DCASE: front 0 degrees, left positive.

The engine measures bearings right positive, and its acoustic checks read that
sign (the expected ILD/ITD of a listener-local bearing), so the engine frame is
never flipped. What a question states, answers and offers is converted here,
once. Owner decision 2026-09-03: Spatial-Omni tells the model that positive
values are to the left, and a bank that says otherwise contradicts it.

Private evidence keeps engine-frame bearings; forms that publish an angle say
which convention they use, so the scorer reads a direction word correctly.
"""
from __future__ import annotations

#: The value unified_scoring.score_angle understands for "positive is left".
PUBLISHED_CONVENTION = "left_positive"
#: Marker for evidence fields that stay in the engine frame.
EVIDENCE_CONVENTION = "engine_right_positive"

CONVENTION_EN = ("front is 0°, left is positive and right is negative, range [-180°, 180°): "
                 "+90° is directly left, -90° directly right, ±180° directly behind")
CONVENTION_ZH = "正前方为0°，左侧为正、右侧为负，范围[-180°，180°)：+90°为正左，-90°为正右，±180°为正后"


def publish_azimuth_deg(engine_deg: float) -> float:
    """An engine-frame (right positive) bearing as published, in [-180, 180)."""
    return (-float(engine_deg) + 180.0) % 360.0 - 180.0


def _deg(value: float) -> str:
    text = f"{value:+.1f}".rstrip("0").rstrip(".")
    return "0" if text in ("+0", "-0") else text


def sector_ranges(sectors, *, width_deg: float = 45.0) -> tuple[str, str]:
    """State each named sector's published range, derived from the sector table.

    ``sectors`` is the engine-ordered table (index i centred on i*width, right
    positive); the published centre is its negative. Listed in increasing
    published angle, front first, so the text reads front, then leftward.
    """
    half = width_deg / 2.0
    rows = []
    for index, (_value, label_en, label_zh) in enumerate(sectors):
        centre = publish_azimuth_deg(index * width_deg)
        low, high = centre - half, centre + half
        rows.append((centre if centre >= -half else centre + 360.0, label_en, label_zh, low, high))
    rows.sort()
    # The unit is stated once and never after a number: these are definitions, and the
    # public display rule rounds any decimal followed by a degree unit to a whole degree.
    en, zh = [], []
    for _order, label_en, label_zh, low, high in rows:
        if low < -180.0 or high > 180.0:  # "behind" straddles ±180
            lo, hi = (low + 360.0, high) if low < -180.0 else (low, high - 360.0)
            en.append(f"{label_en} {_deg(lo)} to 180 or -180 to {_deg(hi)}")
            zh.append(f"{label_zh}{_deg(lo)}～180及-180～{_deg(hi)}")
        else:
            en.append(f"{label_en} {_deg(low)} to {_deg(high)}")
            zh.append(f"{label_zh}{_deg(low)}～{_deg(high)}")
    return ("Directions are horizontal angles in degrees from the way you are facing "
            "(front 0, left positive, right negative): " + "; ".join(en) + ".",
            "方向按水平角度划分（单位：度），以你此刻的朝向为0，左侧为正、右侧为负：" + "；".join(zh) + "。")

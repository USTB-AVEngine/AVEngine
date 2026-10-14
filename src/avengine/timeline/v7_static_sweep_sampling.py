"""Camera sampling for static 30-second v7 sweeps; admission gates are unchanged."""
from __future__ import annotations

import math


def full_heading_static_camera_knots(camera_yaw_deg: float) -> list[dict[str, float]]:
    """Visit four headings, retaining the original front holds in both windows."""
    yaw = float(camera_yaw_deg)
    if not math.isfinite(yaw):
        raise ValueError("camera yaw must be finite")
    samples = ((0, 0), (2, 0), (5, 90), (8, 180), (11, 270),
               (13, 360), (15, 360), (17, 360), (20, 450),
               (23, 540), (26, 630), (28, 720), (30, 720))
    return [{"time_s": time_s, "yaw_deg": yaw + offset}
            for time_s, offset in samples]

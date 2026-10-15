"""Shared side-plane boundary crossing seeds and bounded entry timing."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
REVISION = "joint-boundary-crossing-v1"
SIDES = {"left": -1, "right": 1}

@dataclass(frozen=True)
class EntryEnvelope:
    """Conservative upright body envelope, including pose and root uncertainty."""

    radius_m: float = .4
    height_m: float = 1.9
    hfov_deg: float = 85.
    aspect: float = 16. / 9.
    edge_margin_deg: float = 2.
    pixel_margin_px: float = 8.
    minimum_camera_distance_m: float = 1.5
    minimum_stable_s: float = .6

    def __post_init__(self):
        values = (self.radius_m, self.height_m, self.hfov_deg, self.aspect,
                  self.edge_margin_deg, self.pixel_margin_px,
                  self.minimum_camera_distance_m, self.minimum_stable_s)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("entry envelope needs positive finite dimensions")
        if self.hfov_deg / 2 + self.edge_margin_deg >= 89.:
            raise ValueError("entry field of view has no usable side plane")

def tangent_entry_anchors(camera_m, endpoint_m, side: str, *,
                          final_azimuth_deg: float = 26.,
                          envelope: EntryEnvelope | None = None) -> dict:
    """Invert the side-frustum condition to obtain yaw and a straight entry.

    Camera coordinates are right/up/forward, with right = forward x world-up.
    The target translates towards the centre at constant camera depth. The
    start's entire envelope lies beyond the chosen side plane. At the end the
    entire envelope lies within both horizontal and vertical frustum planes.
    Room support, obstruction and body-to-body spacing are certified by core.
    """
    env = envelope or EntryEnvelope()
    if side not in SIDES:
        raise ValueError("entry side must be left or right")
    camera = np.asarray(camera_m, float)
    endpoint = np.asarray(endpoint_m, float)
    if camera.shape != (3,) or endpoint.shape != (3,) or not np.isfinite([camera, endpoint]).all():
        raise ValueError("entry camera and endpoint must be finite XYZ positions")
    if not 0 < final_azimuth_deg < env.hfov_deg / 2 - env.edge_margin_deg:
        raise ValueError("entry endpoint needs a noncentral inside azimuth")
    delta = endpoint - camera
    radius = float(np.linalg.norm(delta[[0, 2]]))
    if radius < env.minimum_camera_distance_m:
        raise ValueError("entry endpoint violates camera standoff")
    sign = SIDES[side]
    bearing = math.atan2(float(delta[0]), float(-delta[2]))
    yaw = bearing - sign * math.radians(final_azimuth_deg)
    forward = np.array([math.sin(yaw), 0., -math.cos(yaw)])
    right = np.cross(forward, [0., 1., 0.])
    depth = float(delta @ forward)
    lateral = float(delta @ right)
    body_radius = env.radius_m
    h_inside = math.tan(math.radians(env.hfov_deg / 2 - env.edge_margin_deg))
    vfov_half = math.atan(math.tan(math.radians(env.hfov_deg / 2)) / env.aspect)
    v_inside = math.tan(vfov_half - math.radians(env.edge_margin_deg))
    near_depth = depth - body_radius
    vertical = max(abs(float(delta[1])), abs(float(delta[1] + env.height_m)))
    if near_depth <= 0 or abs(lateral) + body_radius > near_depth * h_inside:
        raise ValueError("entry endpoint envelope is not wholly inside horizontally")
    if vertical > near_depth * v_inside:
        raise ValueError("entry endpoint envelope is not wholly inside vertically")
    outside_lateral = (depth + body_radius) * math.tan(math.radians(env.hfov_deg / 2 + env.edge_margin_deg)) + body_radius
    length = outside_lateral - sign * lateral
    start = endpoint + sign * right * length
    inward = -sign * right
    return {
        "revision": REVISION, "branch": side,
        "listener_m": camera.tolist(),
        "camera_yaw_deg": (math.degrees(yaw) + 180.) % 360. - 180.,
        "camera_forward": forward.tolist(), "camera_right": right.tolist(),
        "start_m": start.tolist(), "endpoint_m": endpoint.tolist(),
        "inward_unit_xz": inward[[0, 2]].tolist(), "planning_length_m": length,
        "depth_m": depth, "final_azimuth_deg": sign * final_azimuth_deg,
        "body_radius_m": body_radius, "body_height_m": env.height_m,
        "edge_margin_deg": env.edge_margin_deg,
        "start_outside_plane_margin_m": (depth + body_radius) * (
            math.tan(math.radians(env.hfov_deg / 2 + env.edge_margin_deg)) -
            math.tan(math.radians(env.hfov_deg / 2))),
        "horizontal_endpoint_plane_margin_m": near_depth * math.tan(math.radians(env.hfov_deg / 2)) - abs(lateral) - body_radius,
        "vertical_endpoint_plane_margin_m": near_depth * math.tan(vfov_half) - vertical,
        "capture_camera_moves": False,
        "evidence_kind": "analytic_body_envelope_prediction",
        "native_pixel_truth_verified": False,
    }

def entry_timing(length_m: float, speed_m_s: float, *, stop_s: float = 7.6,
                 gait_extension_reserve_m: float = .65) -> dict:
    """Put the completed entry away from half-overlapping crop boundaries.

    An entry that stops just after t=5 can leave a crop containing only a slow
    deceleration sliver. Solve the start from length instead: the requested
    finish is near t=7.6, leaving room for complete-step fitting and a stable
    first-window entry witness, and real locomotion inside the t=5 crop.
    """
    if not math.isfinite(length_m) or length_m <= 0 or not .8 <= speed_m_s <= 1.4:
        raise ValueError("entry timing needs positive length and legal human speed")
    if not math.isfinite(stop_s) or not math.isfinite(gait_extension_reserve_m) or gait_extension_reserve_m < 0:
        raise ValueError("entry timing needs finite stop and nonnegative extension reserve")
    start = stop_s-length_m/speed_m_s-.25
    latest = stop_s+.02+gait_extension_reserve_m/speed_m_s+1/30.
    if start < .75 or latest > 9.:
        raise ValueError("entry route cannot fit an outside pause and stable first-window finish")
    return {"start_s": start, "requested_interval_s": [start, stop_s+.02],
            "nominal_stop_s": stop_s, "latest_reserved_stop_s": latest,
            "complete_step_time_is_revalidated_natively": True}

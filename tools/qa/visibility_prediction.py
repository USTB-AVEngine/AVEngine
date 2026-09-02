"""Predict actor visibility from the camera clearance table, before rendering.

The table stores, for every solver camera point, how far the first scene
surface is in every direction.  An actor's route is known frame by frame, so
for any frame we can ask: along the sight lines from the camera to points on
the actor's body, does the scene get in the way before the ray reaches the
body?  The share of body samples that are reachable is the predicted visible
fraction.  Another actor standing in the way is handled with a cylinder test.

This is a prediction used to budget difficulty tiers and to record what the
solver expected; the native pixel truth remains the acceptance authority.
Body dimensions are research placeholders supplied by the caller (a dog is
0.5 m tall and 0.8 m long here); the asset policy will own them later.

Units follow the table: positions in UE centimetres, heights in metres.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from camera_clearance import NO_HIT_M, CameraClearanceTable

DEFAULT_BODY_M = {"height_m": 0.5, "length_m": 0.8}
# body samples: three heights (legs, body, head) x three lateral offsets
BODY_HEIGHT_FRACTIONS = (0.2, 0.6, 1.0)
BODY_LATERAL_FRACTIONS = (-0.5, 0.0, 0.5)
REACH_MARGIN_M = 0.10
TIER_EDGES_DEFAULT = (0.5, 0.2)
TIERS = ("light", "medium", "heavy", "hidden", "out_of_view", "unknown")


def body_from_params(params: Mapping[str, Any], key: str = "PREDICTION_BODY_M") -> dict:
    """Body dimensions come from params (placeholder) and are recorded as such."""
    body = params.get(key)
    if body is None:
        return dict(DEFAULT_BODY_M, status="placeholder_default")
    height = float(body["height_m"])
    length = float(body["length_m"])
    if not (height > 0.0 and length > 0.0):
        raise ValueError(f"{key} dimensions must be positive")
    return {"height_m": height, "length_m": length,
            "status": str(body.get("status", "placeholder_from_params"))}


def body_samples_m(camera_xy_m: Sequence[float], actor_xy_m: Sequence[float],
                   body: Mapping[str, float]) -> np.ndarray:
    """World (x, y, z_above_floor) of the body sample points, metres.

    Lateral offsets run perpendicular to the sight line so the silhouette
    width does not depend on the actor's heading (which the plan may not
    know yet)."""
    dx = actor_xy_m[0] - camera_xy_m[0]
    dy = actor_xy_m[1] - camera_xy_m[1]
    norm = math.hypot(dx, dy)
    if norm <= 1.0e-9:
        raise ValueError("actor stands on the camera")
    perp = (-dy / norm, dx / norm)
    points = []
    for hf in BODY_HEIGHT_FRACTIONS:
        for lf in BODY_LATERAL_FRACTIONS:
            offset = lf * float(body["length_m"])
            points.append((actor_xy_m[0] + perp[0] * offset,
                           actor_xy_m[1] + perp[1] * offset,
                           hf * float(body["height_m"])))
    return np.asarray(points, dtype=np.float64)


def sight_lines(camera_xyz_m: Sequence[float], points_m: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuth (deg, UE yaw reference), elevation (deg) and distance (m) to each point."""
    dx = points_m[:, 0] - camera_xyz_m[0]
    dy = points_m[:, 1] - camera_xyz_m[1]
    dz = points_m[:, 2] - camera_xyz_m[2]
    horizontal = np.hypot(dx, dy)
    azimuth = np.degrees(np.arctan2(dy, dx))
    elevation = np.degrees(np.arctan2(dz, horizontal))
    distance = np.sqrt(horizontal ** 2 + dz ** 2)
    return azimuth, elevation, distance


def actor_cylinder_blocks(camera_xyz_m: Sequence[float], points_m: np.ndarray,
                          other_xy_m: Sequence[float], other_body: Mapping[str, float]
                          ) -> np.ndarray:
    """Does another actor (a vertical cylinder) cut a ray before its target point?"""
    radius = 0.5 * float(other_body["length_m"])
    top = float(other_body["height_m"])
    cam = np.asarray(camera_xyz_m, dtype=np.float64)
    seg = points_m - cam
    seg_xy = seg[:, :2]
    length_xy = np.linalg.norm(seg_xy, axis=1)
    to_other = np.asarray(other_xy_m, dtype=np.float64) - cam[:2]
    along = (seg_xy @ to_other) / np.maximum(length_xy ** 2, 1.0e-12)   # fraction of segment
    within = (along > 0.0) & (along < 1.0)
    closest = cam[:2] + seg_xy * along[:, None]
    lateral = np.linalg.norm(closest - np.asarray(other_xy_m), axis=1)
    height_at = cam[2] + seg[:, 2] * along
    return within & (lateral <= radius) & (height_at >= 0.0) & (height_at <= top)


def predict_point_visibility(table: CameraClearanceTable, *, camera_xy_cm: Sequence[float],
                             camera_height_m: float, ground_z_cm: float,
                             actor_xy_cm: Sequence[float], body: Mapping[str, float],
                             others: Sequence[tuple[Sequence[float], Mapping[str, float]]] = (),
                             margin_m: float = REACH_MARGIN_M) -> dict[str, Any]:
    """Predicted visible fraction of one actor at one instant.

    Each body sample is reachable when the first scene surface along its
    sight line is at least as far as the sample (minus a margin), and no
    other actor's cylinder cuts the ray first.  Samples whose direction falls
    outside the stored ring return unknown and are left out of the fraction."""
    camera_xyz_m = (camera_xy_cm[0] / 100.0, camera_xy_cm[1] / 100.0, float(camera_height_m))
    points = body_samples_m(camera_xyz_m[:2],
                            (actor_xy_cm[0] / 100.0, actor_xy_cm[1] / 100.0), body)
    # heights are relative to the floor; the camera height is too, so z=0 is the floor
    azimuth, elevation, distance = sight_lines(camera_xyz_m, points)
    first = table.first_obstacle_m(camera_xy_cm, camera_height_m, azimuth, elevation)
    known = np.isfinite(first)
    reachable = known & ((first >= NO_HIT_M) | (first >= distance - margin_m))
    by_actor = np.zeros(len(points), dtype=bool)
    for other_xy_cm, other_body in others:
        by_actor |= actor_cylinder_blocks(
            camera_xyz_m, points, (other_xy_cm[0] / 100.0, other_xy_cm[1] / 100.0), other_body)
    visible = reachable & ~by_actor
    n_known = int(known.sum())
    return {
        "predicted_visible_fraction": (float(visible.sum() / n_known) if n_known else None),
        "known_fraction": float(n_known / len(points)),
        "samples": int(len(points)),
        "blocked_by_scene": int((known & ~reachable).sum()),
        "blocked_by_actor": int((reachable & by_actor).sum()),
        "distance_m": float(np.median(distance)),
        "margin_m": float(margin_m),
        "ground_z_ue_cm": float(ground_z_cm),
    }


def predicted_tier(in_fov: bool, fraction: float | None,
                   edges: Sequence[float] = TIER_EDGES_DEFAULT) -> str:
    """Same ladder as the pixel join's tier policy, applied to a prediction."""
    if not in_fov:
        return "out_of_view"
    if fraction is None:
        return "unknown"
    if fraction <= 0.0:
        return "hidden"
    if fraction >= float(edges[0]):
        return "light"
    if fraction >= float(edges[1]):
        return "medium"
    return "heavy"


def relative_azimuth(camera_xy: Sequence[float], camera_yaw_deg: float,
                     point_xy: Sequence[float]) -> float:
    bearing = math.degrees(math.atan2(point_xy[1] - camera_xy[1], point_xy[0] - camera_xy[0]))
    return (bearing - camera_yaw_deg + 180.0) % 360.0 - 180.0


def predict_timeline(table: CameraClearanceTable, *, camera_xy_cm: Sequence[float],
                     camera_height_m: float, camera_yaw_deg: float, hfov_deg: float,
                     ground_z_cm: float, routes_by_slot: Mapping[str, Sequence[Sequence[float]]],
                     bodies_by_slot: Mapping[str, Mapping[str, float]],
                     frames: Sequence[int] | None = None,
                     edges: Sequence[float] = TIER_EDGES_DEFAULT) -> dict[str, Any]:
    """Per-slot, per-frame predicted visibility along whole routes.

    routes_by_slot maps a slot to its per-frame (x, y) in UE cm.  Every other
    slot is treated as a potential occluder at the same frame."""
    slots = list(routes_by_slot)
    n_frames = min(len(route) for route in routes_by_slot.values())
    frame_list = list(range(n_frames)) if frames is None else [int(f) for f in frames]
    half_fov = float(hfov_deg) / 2.0
    out: dict[str, Any] = {"frames": frame_list, "slots": {}}
    for slot in slots:
        rows = []
        for frame in frame_list:
            actor = routes_by_slot[slot][frame]
            others = [(routes_by_slot[other][frame], bodies_by_slot[other])
                      for other in slots if other != slot]
            azimuth = relative_azimuth(camera_xy_cm, camera_yaw_deg, actor)
            in_fov = abs(azimuth) <= half_fov
            prediction = predict_point_visibility(
                table, camera_xy_cm=camera_xy_cm, camera_height_m=camera_height_m,
                ground_z_cm=ground_z_cm, actor_xy_cm=actor, body=bodies_by_slot[slot],
                others=others)
            fraction = prediction["predicted_visible_fraction"]
            rows.append({"frame": frame, "relative_azimuth_deg": round(azimuth, 3),
                         "in_fov": in_fov,
                         "predicted_visible_fraction": fraction,
                         "known_fraction": prediction["known_fraction"],
                         "blocked_by_scene": prediction["blocked_by_scene"],
                         "blocked_by_actor": prediction["blocked_by_actor"],
                         "tier": predicted_tier(in_fov, fraction, edges)})
        out["slots"][slot] = {"per_frame": rows, "body": dict(bodies_by_slot[slot])}
    out["tier_edges"] = [float(e) for e in edges]
    return out


def timeline_statistics(per_frame: Sequence[Mapping[str, Any]], *,
                        instants: Mapping[str, int], window_frames: int = 2,
                        visible_min_fraction: float = 0.2) -> dict[str, Any]:
    """Whole-clip statistics the tier rules (still to be fixed with human data) will use:
    how much of the clip the actor is visible, whether it shows around each
    named instant, and how long it was hidden right before each instant."""
    frames = [row["frame"] for row in per_frame]
    visible = {row["frame"]: (row["in_fov"] and row["predicted_visible_fraction"] is not None
                              and row["predicted_visible_fraction"] >= visible_min_fraction)
               for row in per_frame}
    n_known = sum(1 for row in per_frame if row["predicted_visible_fraction"] is not None
                  or not row["in_fov"])
    visible_frames = sum(1 for value in visible.values() if value)
    stats: dict[str, Any] = {
        "frames_evaluated": len(frames),
        "visible_frames_fraction": (visible_frames / len(frames)) if frames else None,
        "known_frames": n_known,
        "visible_min_fraction": visible_min_fraction,
        "window_frames": int(window_frames),
        "visible_near_instant": {},
        "hidden_frames_before_instant": {},
    }
    ordered = sorted(frames)
    for name, instant in instants.items():
        near = [f for f in ordered if abs(f - int(instant)) <= window_frames]
        stats["visible_near_instant"][name] = any(visible[f] for f in near) if near else None
        hidden = 0
        for f in reversed([f for f in ordered if f <= int(instant)]):
            if visible[f]:
                break
            hidden += 1
        stats["hidden_frames_before_instant"][name] = hidden
    longest, run = 0, 0
    for f in ordered:
        run = 0 if visible[f] else run + 1
        longest = max(longest, run)
    stats["longest_hidden_streak_frames"] = longest
    stats["never_visible"] = visible_frames == 0
    return stats

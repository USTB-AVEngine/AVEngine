#!/usr/bin/env python3
"""Routes designed by the solver for the pose it has chosen.

Why
---
The bank-only solver drew a route, then solved the camera yaw so that the
route's key frame fell into the requested answer band.  When no bank route
had the needed shape, the cell's budget ran out.  On Kujiale the card1 family
filled about 1 of 18 cells per round (2026-09-02): the 200 moving routes there
travel a median 1.88 m between anchor and query frame, and only 5 percent of
random (camera, route) pairs keep both instants 2.5 m from the lens while
sweeping more than 30 degrees.  Relaxing the distance floor to 2.0 m gave
5/36.  The shortage is in the bank, not in the constraints.

What this does
--------------
Given the camera pose, it places the target at the question's key frames
directly inside the declared azimuth bands and distance ranges and walks it
there at a constant speed drawn from a declared range:

* two designed frames: the later point is drawn (azimuth, distance), the
  speed is drawn, and the earlier point's distance is *solved* on its own
  azimuth ray so that the straight leg between them has exactly that speed
  (the quadratic of the 2026-09-03 proposal); the legs before the first and
  after the last designed frame keep the speed and may turn by up to a
  declared angle, the way a navigation path turns at a corner;
* one designed frame: the point is drawn, then a heading and a speed; the
  incoming leg may turn at the designed frame.

Every frame the actor actually occupies (after the solver's idle shift) must
lie in the scene's walkable grid with a clearance margin.  The result is an
ordinary ``Route``: the solver applies the same idle shift, the same checks
and the same rejection rules to it as to a bank route; only the provenance
differs and is recorded.

The owner's ruling (2026-09-03): a synthesized route is a legitimate
trajectory provided it is treated exactly like a bank route.  Nothing here
relaxes a question-type constraint; a designed route that fails any solver
check is rejected like any other candidate.

Boundary
--------
Piecewise-straight paths are less natural than recorded ones.  The walkable
grid is two-dimensional, so pixel truth on rendered candidates remains the
authority for "the actor did not walk through furniture".  Speed range,
margin, turn limit and the extra attempt budget are research placeholders.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence

from walkable_grid import WalkableGrid

FRAME_COUNT = 75
FRAME_RATE_HZ = 15.0

ENABLED_KEY = "ROUTE_SYNTHESIS_ENABLED"
SPEED_KEY = "ROUTE_SYNTHESIS_SPEED_MPS_RANGE"
MARGIN_KEY = "ROUTE_SYNTHESIS_WALKABLE_MARGIN_M"
MAX_DISTANCE_KEY = "ROUTE_SYNTHESIS_MAX_CAMERA_DISTANCE_CM"
ATTEMPTS_KEY = "ROUTE_SYNTHESIS_ATTEMPTS"
DESIGN_TRIES_KEY = "ROUTE_SYNTHESIS_DESIGN_TRIES"
MAX_TURN_KEY = "ROUTE_SYNTHESIS_MAX_TURN_DEG"

REASON_SPEED = "synthesis_no_distance_solves_speed"
REASON_WALKABLE = "synthesis_route_outside_walkable"
REASON_SPEC = "synthesis_infeasible_spec"


@dataclass(frozen=True)
class SynthesisSettings:
    speed_min_mps: float
    speed_max_mps: float
    margin_cm: float
    max_camera_distance_cm: float
    synthesized_attempts: int
    design_tries: int
    max_turn_deg: float

    @classmethod
    def from_params(cls, params: dict) -> "SynthesisSettings | None":
        """None when synthesis is off; otherwise every key must be present and sane."""
        if not params.get(ENABLED_KEY):
            return None
        missing = [key for key in (SPEED_KEY, MARGIN_KEY, MAX_DISTANCE_KEY, ATTEMPTS_KEY)
                   if key not in params]
        if missing:
            raise ValueError(f"{ENABLED_KEY} is set but params lack {missing}")
        speed = params[SPEED_KEY]
        if (not isinstance(speed, (list, tuple)) or len(speed) != 2
                or not all(math.isfinite(float(v)) for v in speed)
                or not (0.0 < float(speed[0]) <= float(speed[1]))):
            raise ValueError(f"{SPEED_KEY} must be [min, max] metres per second, 0 < min <= max")
        margin_m = float(params[MARGIN_KEY])
        if not math.isfinite(margin_m) or margin_m < 0.0:
            raise ValueError(f"{MARGIN_KEY} must be a finite non-negative metre value")
        max_distance = float(params[MAX_DISTANCE_KEY])
        if not math.isfinite(max_distance) or max_distance <= 0.0:
            raise ValueError(f"{MAX_DISTANCE_KEY} must be a positive centimetre value")
        attempts = int(params[ATTEMPTS_KEY])
        if attempts < 0:
            raise ValueError(f"{ATTEMPTS_KEY} must be non-negative")
        tries = int(params.get(DESIGN_TRIES_KEY, 8))
        if tries < 1:
            raise ValueError(f"{DESIGN_TRIES_KEY} must be at least one")
        turn = float(params.get(MAX_TURN_KEY, 90.0))
        if not math.isfinite(turn) or not (0.0 <= turn <= 180.0):
            raise ValueError(f"{MAX_TURN_KEY} must lie in [0, 180]")
        return cls(speed_min_mps=float(speed[0]), speed_max_mps=float(speed[1]),
                   margin_cm=margin_m * 100.0, max_camera_distance_cm=max_distance,
                   synthesized_attempts=attempts, design_tries=tries, max_turn_deg=turn)

    def as_dict(self) -> dict:
        return {"speed_mps_range": [self.speed_min_mps, self.speed_max_mps],
                "walkable_margin_cm": self.margin_cm,
                "max_camera_distance_cm": self.max_camera_distance_cm,
                "synthesized_attempts_after_bank": self.synthesized_attempts,
                "design_tries_per_attempt": self.design_tries,
                "max_turn_deg": self.max_turn_deg}


@dataclass(frozen=True)
class PointSpec:
    """Where the actor must be at one frame, relative to the camera pose.

    ``exclusions`` are (centre, half-width) azimuth windows the draw must avoid,
    e.g. "at least MIN_AZIMUTH_SEP from the target's azimuth"; they turn a
    rejection the solver would make anyway into a constraint on the draw."""
    frame: int
    azimuth_lo_deg: float
    azimuth_hi_deg: float
    distance_lo_cm: float
    distance_hi_cm: float
    exclusions: tuple = ()

    def feasible(self) -> bool:
        return (0 <= self.frame < FRAME_COUNT
                and self.azimuth_lo_deg < self.azimuth_hi_deg
                and 0.0 < self.distance_lo_cm <= self.distance_hi_cm)

    def azimuth_intervals(self, extra_exclusions=()) -> list[tuple[float, float]]:
        return subtract_windows((self.azimuth_lo_deg, self.azimuth_hi_deg),
                                list(self.exclusions) + list(extra_exclusions))


def subtract_windows(interval, windows) -> list[tuple[float, float]]:
    """The parts of [lo, hi) outside every (centre, half-width) window."""
    pieces = [(float(interval[0]), float(interval[1]))]
    for centre, half in windows:
        lo_w, hi_w = float(centre) - float(half), float(centre) + float(half)
        kept = []
        for lo, hi in pieces:
            if hi <= lo_w or lo >= hi_w:
                kept.append((lo, hi))
                continue
            if lo < lo_w:
                kept.append((lo, lo_w))
            if hi > hi_w:
                kept.append((hi_w, hi))
        pieces = kept
    return [(lo, hi) for lo, hi in pieces if hi - lo > 1e-9]


def draw_from_intervals(rng, intervals) -> float | None:
    """Uniform draw over a union of disjoint intervals (by length)."""
    lengths = [hi - lo for lo, hi in intervals]
    total = sum(lengths)
    if total <= 0.0:
        return None
    u = float(rng.random()) * total
    for (lo, hi), length in zip(intervals, lengths):
        if u <= length:
            return lo + u
        u -= length
    return intervals[-1][1]


def _unit(bearing_deg: float) -> tuple[float, float]:
    radians = math.radians(bearing_deg)
    return math.cos(radians), math.sin(radians)


def _point_from_pose(camera_xy, camera_yaw_deg: float, azimuth_deg: float,
                     distance_cm: float) -> tuple[float, float]:
    ux, uy = _unit(float(camera_yaw_deg) + float(azimuth_deg))
    return (float(camera_xy[0]) + distance_cm * ux, float(camera_xy[1]) + distance_cm * uy)


def solve_ray_distance(camera_xy, bearing_deg: float, target_xy, chord_cm: float) -> list[float]:
    """Distances d along the ray from the camera at ``bearing_deg`` whose point
    lies exactly ``chord_cm`` from ``target_xy`` (0, 1 or 2 non-negative roots)."""
    ux, uy = _unit(bearing_deg)
    wx = float(target_xy[0]) - float(camera_xy[0])
    wy = float(target_xy[1]) - float(camera_xy[1])
    along = ux * wx + uy * wy
    discriminant = along * along - (wx * wx + wy * wy - chord_cm * chord_cm)
    if discriminant < 0.0:
        return []
    root = math.sqrt(discriminant)
    return sorted({d for d in (along - root, along + root) if d >= 0.0})


def leg_positions(anchor_frame: int, anchor_xy, heading_deg: float, step_cm: float,
                  frames) -> list[tuple[float, float]]:
    """Positions on a straight leg through ``anchor_xy`` at ``anchor_frame``."""
    ux, uy = _unit(heading_deg)
    return [(float(anchor_xy[0]) + step_cm * (t - anchor_frame) * ux,
             float(anchor_xy[1]) + step_cm * (t - anchor_frame) * uy) for t in frames]


def polyline_positions(designed: Sequence[tuple[int, tuple[float, float]]], step_cm: float,
                       pre_heading_deg: float, post_heading_deg: float,
                       idle_frames: int) -> list[tuple[float, float]]:
    """Base-route samples: base[k] is the actor's position at clip time k + idle.

    Between two designed frames the actor walks the straight leg joining them;
    before the first it arrives along ``pre_heading_deg`` and after the last it
    leaves along ``post_heading_deg``, all at ``step_cm`` per frame."""
    designed = sorted(designed, key=lambda item: item[0])
    (first_frame, first_xy), (last_frame, last_xy) = designed[0], designed[-1]
    pre = _unit(pre_heading_deg)
    post = _unit(post_heading_deg)
    samples = []
    for k in range(FRAME_COUNT):
        t = k + int(idle_frames)
        if t <= first_frame:
            samples.append((first_xy[0] + step_cm * (t - first_frame) * pre[0],
                            first_xy[1] + step_cm * (t - first_frame) * pre[1]))
        elif t >= last_frame:
            samples.append((last_xy[0] + step_cm * (t - last_frame) * post[0],
                            last_xy[1] + step_cm * (t - last_frame) * post[1]))
        else:
            span = float(last_frame - first_frame)
            samples.append((first_xy[0] + (last_xy[0] - first_xy[0]) * (t - first_frame) / span,
                            first_xy[1] + (last_xy[1] - first_xy[1]) * (t - first_frame) / span))
    return samples


def _route_id(design: dict) -> str:
    digest = hashlib.sha1(json.dumps(design, sort_keys=True).encode()).hexdigest()
    return f"synth:{digest[:12]}"


class RouteSynthesizer:
    """Designs constant-speed, piecewise-straight routes on one scene's walkable grid."""

    SPEED_DRAWS = 6          # speed draws per design before giving up on the solve

    def __init__(self, grid: WalkableGrid, settings: SynthesisSettings,
                 frame_rate_hz: float = FRAME_RATE_HZ) -> None:
        self.grid = grid
        self.settings = settings
        self.frame_rate_hz = float(frame_rate_hz)
        self.counters: dict = {"designs": 0, "built": 0, "rejected": {}}

    # -- helpers -----------------------------------------------------------

    def _reject(self, reason: str):
        self.counters["rejected"][reason] = self.counters["rejected"].get(reason, 0) + 1
        return None, reason

    def _draw_speed(self, rng) -> float:
        low, high = self.settings.speed_min_mps, self.settings.speed_max_mps
        return low + (high - low) * float(rng.random())

    def _turn_candidates(self, rng) -> list[float]:
        """Turn angles to try for a leg, straight included, in a random order."""
        limit = self.settings.max_turn_deg
        steps = 3
        values = [0.0]
        for k in range(1, steps + 1):
            values.extend([limit * k / steps, -limit * k / steps])
        order = rng.permutation(len(values))
        return [values[int(i)] for i in order]

    def _leg_ok(self, anchor_frame, anchor_xy, heading, step_cm, frames) -> bool:
        if not frames:
            return True
        ok, _ = self.grid.route_ok(leg_positions(anchor_frame, anchor_xy, heading, step_cm, frames),
                                   self.settings.margin_cm)
        return ok

    def _finish(self, samples, design: dict, role: str, idle_frames: int):
        from scene_sampler import Route  # local import: scene_sampler imports this module

        # only frames the actor occupies after the idle shift are checked:
        # base[k] for k <= 74 - idle (the shifted route holds base[0] earlier)
        occupied = samples[:FRAME_COUNT - int(idle_frames)]
        ok, detail = self.grid.route_ok(occupied, self.settings.margin_cm)
        if not ok:
            return self._reject(REASON_WALKABLE)
        displacement_cm = math.dist(samples[0], samples[-1])
        design = dict(design, min_clearance_cm=detail["min_clearance_cm"],
                      worst_frame=detail["worst_frame"], margin_cm=self.settings.margin_cm,
                      checked_frames=len(occupied))
        provenance = {"source": "synthesized", "role": role, "design": design,
                      "grid": {"scene_id": self.grid.scene_id,
                               "arrays_sha256": self.grid.identity["arrays_sha256"]}}
        self.counters["built"] += 1
        # bank identity: implied speed is the end-to-end span over the 5 s clip
        route = Route(_route_id(design), samples, displacement_cm / 100.0 / 5.0,
                      provenance=provenance)
        return route, None

    def _point_record(self, camera_xy, yaw, frame, azimuth, distance, solved):
        return {"frame": int(frame), "azimuth_deg": round(azimuth, 4),
                "distance_cm": round(distance, 3), "solved": bool(solved),
                "xy_cm": [round(v, 3) for v in _point_from_pose(camera_xy, yaw, azimuth, distance)]}

    def _legs(self, rng, designed, step_cm, heading, idle_frames):
        """Find turn angles for the incoming and outgoing legs that stay walkable."""
        (first_frame, first_xy) = designed[0]
        (last_frame, last_xy) = designed[-1]
        pre_frames = list(range(int(idle_frames), first_frame))
        post_frames = list(range(last_frame + 1, FRAME_COUNT))
        turn_in = None
        for turn in self._turn_candidates(rng):
            if self._leg_ok(first_frame, first_xy, heading + turn, step_cm, pre_frames):
                turn_in = turn
                break
        if turn_in is None:
            return None
        turn_out = None
        for turn in self._turn_candidates(rng):
            if self._leg_ok(last_frame, last_xy, heading + turn, step_cm, post_frames):
                turn_out = turn
                break
        if turn_out is None:
            return None
        return turn_in, turn_out

    # -- public ------------------------------------------------------------

    def design(self, rng, camera_xy, camera_yaw_deg: float, specs: Sequence[PointSpec], *,
               idle_frames: int, role: str, min_gap_between_points_deg: float | None = None):
        """One design attempt.  Returns (route, None) or (None, reason).

        Two specs: the later point is drawn (its azimuth from the part of its
        band that leaves room in the earlier band once ``min_gap_between_points_deg``
        is honoured), the speed is drawn, the earlier azimuth is drawn from
        what remains of its band, and the earlier distance is solved so that
        the leg has that speed.  One spec: the point is drawn, then a heading.
        Incoming and outgoing legs try a few turn angles and keep the first
        that stays walkable."""
        self.counters["designs"] += 1
        specs = sorted(specs, key=lambda spec: spec.frame)
        if not specs or len(specs) > 2 or not all(spec.feasible() for spec in specs):
            return self._reject(REASON_SPEC)
        if len(specs) == 2 and specs[0].frame == specs[1].frame:
            return self._reject(REASON_SPEC)
        yaw = float(camera_yaw_deg)
        design = {"camera_xy_cm": [round(float(camera_xy[0]), 3), round(float(camera_xy[1]), 3)],
                  "camera_yaw_deg": round(yaw, 4), "idle_frames": int(idle_frames),
                  "shape": "constant_speed_polyline"}
        if len(specs) == 1:
            spec = specs[0]
            azimuth = draw_from_intervals(rng, spec.azimuth_intervals())
            if azimuth is None:
                return self._reject(REASON_SPEC)
            distance = spec.distance_lo_cm + (spec.distance_hi_cm - spec.distance_lo_cm) * float(rng.random())
            point = self._point_record(camera_xy, yaw, spec.frame, azimuth, distance, False)
            if not self.grid.is_walkable(point["xy_cm"], self.settings.margin_cm):
                return self._reject(REASON_WALKABLE)
            speed_mps = self._draw_speed(rng)
            step_cm = speed_mps * 100.0 / self.frame_rate_hz
            base = 360.0 * float(rng.random())
            for k in rng.permutation(8):
                heading = (base + 45.0 * int(k)) % 360.0
                legs = self._legs(rng, [(spec.frame, tuple(point["xy_cm"]))], step_cm, heading,
                                  idle_frames)
                if legs is not None:
                    turn_in, turn_out = legs
                    design.update(points=[point], speed_mps=round(speed_mps, 4),
                                  heading_deg=round(heading, 4), turn_in_deg=round(turn_in, 4),
                                  turn_out_deg=round(turn_out, 4))
                    samples = polyline_positions([(spec.frame, tuple(point["xy_cm"]))], step_cm,
                                                 heading + turn_in, heading + turn_out, idle_frames)
                    return self._finish(samples, design, role, idle_frames)
            return self._reject(REASON_WALKABLE)
        early, late = specs
        gap = float(min_gap_between_points_deg or 0.0)
        early_intervals = early.azimuth_intervals()
        if not early_intervals:
            return self._reject(REASON_SPEC)
        if gap > 0.0:
            # A late azimuth q is usable when some early azimuth lies at least
            # `gap` away: for an early piece [lo, hi] that excludes only the
            # window (hi - gap, lo + gap), which is empty once the piece is wider
            # than twice the gap.  Union over pieces, then the late band's own
            # exclusions.
            allowed = []
            for lo, hi in early_intervals:
                half = gap - (hi - lo) / 2.0
                windows = [((lo + hi) / 2.0, half)] if half > 0.0 else []
                allowed.extend(subtract_windows((late.azimuth_lo_deg, late.azimuth_hi_deg), windows))
            late_intervals = _intersect(
                late.azimuth_intervals(),
                _union_within((late.azimuth_lo_deg, late.azimuth_hi_deg), allowed))
        else:
            late_intervals = late.azimuth_intervals()
        late_azimuth = draw_from_intervals(rng, late_intervals)
        if late_azimuth is None:
            return self._reject(REASON_SPEC)
        late_distance = late.distance_lo_cm + (late.distance_hi_cm - late.distance_lo_cm) * float(rng.random())
        late_point = self._point_record(camera_xy, yaw, late.frame, late_azimuth, late_distance, False)
        if not self.grid.is_walkable(late_point["xy_cm"], self.settings.margin_cm):
            return self._reject(REASON_WALKABLE)
        early_choices = early.azimuth_intervals([(late_azimuth, gap)] if gap > 0.0 else [])
        early_azimuth = draw_from_intervals(rng, early_choices)
        if early_azimuth is None:
            return self._reject(REASON_SPEC)
        early_point = None
        for _ in range(self.SPEED_DRAWS):
            speed_mps = self._draw_speed(rng)
            chord_cm = speed_mps * 100.0 / self.frame_rate_hz * (late.frame - early.frame)
            roots = [d for d in solve_ray_distance(camera_xy, yaw + early_azimuth,
                                                   late_point["xy_cm"], chord_cm)
                     if early.distance_lo_cm <= d <= early.distance_hi_cm]
            if roots:
                distance = roots[int(rng.integers(len(roots)))] if len(roots) > 1 else roots[0]
                early_point = self._point_record(camera_xy, yaw, early.frame, early_azimuth,
                                                 distance, True)
                break
        if early_point is None:
            return self._reject(REASON_SPEED)
        # the recorded (rounded) points are what the route passes through; the
        # leg speed is recomputed from them so speed and geometry agree exactly
        leg_cm = math.dist(early_point["xy_cm"], late_point["xy_cm"])
        step_cm = leg_cm / (late.frame - early.frame)
        speed_mps = step_cm * self.frame_rate_hz / 100.0
        if not (self.settings.speed_min_mps - 1e-6 <= speed_mps <= self.settings.speed_max_mps + 1e-6):
            return self._reject(REASON_SPEED)
        heading = math.degrees(math.atan2(late_point["xy_cm"][1] - early_point["xy_cm"][1],
                                          late_point["xy_cm"][0] - early_point["xy_cm"][0]))
        mid_frames = list(range(early.frame, late.frame + 1))
        if not self._leg_ok(early.frame, early_point["xy_cm"], heading, step_cm, mid_frames):
            return self._reject(REASON_WALKABLE)
        designed = [(early.frame, tuple(early_point["xy_cm"])), (late.frame, tuple(late_point["xy_cm"]))]
        legs = self._legs(rng, designed, step_cm, heading, idle_frames)
        if legs is None:
            return self._reject(REASON_WALKABLE)
        turn_in, turn_out = legs
        design.update(points=[early_point, late_point], speed_mps=round(speed_mps, 4),
                      heading_deg=round(heading, 4), turn_in_deg=round(turn_in, 4),
                      turn_out_deg=round(turn_out, 4),
                      min_gap_between_points_deg=gap if gap > 0.0 else None)
        samples = polyline_positions(designed, step_cm, heading + turn_in, heading + turn_out,
                                     idle_frames)
        return self._finish(samples, design, role, idle_frames)

    def design_many(self, rng, camera_xy, camera_yaw_deg: float, specs: Sequence[PointSpec], *,
                    idle_frames: int, role: str, tries: int | None = None,
                    min_gap_between_points_deg: float | None = None):
        """Repeat ``design`` up to ``tries`` times; returns (route, last_reason)."""
        reason = REASON_SPEC
        for _ in range(int(tries or self.settings.design_tries)):
            route, reason = self.design(rng, camera_xy, camera_yaw_deg, specs,
                                        idle_frames=idle_frames, role=role,
                                        min_gap_between_points_deg=min_gap_between_points_deg)
            if route is not None:
                return route, None
        return None, reason

    def report(self) -> dict:
        return {"settings": self.settings.as_dict(), "grid": self.grid.identity,
                "counters": json.loads(json.dumps(self.counters))}


def _union_within(interval, pieces):
    """Pieces clipped to interval, merged where they touch or overlap."""
    lo0, hi0 = interval
    clipped = sorted((max(lo0, lo), min(hi0, hi)) for lo, hi in pieces if min(hi0, hi) > max(lo0, lo))
    merged: list[tuple[float, float]] = []
    for lo, hi in clipped:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _intersect(a, b):
    """Intersection of two lists of disjoint intervals."""
    out = []
    for lo_a, hi_a in a:
        for lo_b, hi_b in b:
            lo, hi = max(lo_a, lo_b), min(hi_a, hi_b)
            if hi - lo > 1e-9:
                out.append((lo, hi))
    return sorted(out)

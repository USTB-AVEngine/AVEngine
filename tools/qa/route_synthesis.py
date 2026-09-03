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
Given the camera pose, it draws the target's positions at the question's key
frames directly inside the declared azimuth bands and distance ranges,
connects them with a straight constant-speed path, fills all 75 frames, and
keeps the whole path inside the scene's walkable grid with a clearance margin.
Speed must fall in a declared range.  The result is an ordinary ``Route``: the
solver applies the same idle shift, the same checks and the same rejection
rules to it as to a bank route; only the provenance differs and is recorded.

The owner's ruling (2026-09-03): a synthesized route is a legitimate
trajectory provided it is treated exactly like a bank route.  Nothing here
relaxes a question-type constraint; a designed route that fails any solver
check is rejected like any other candidate.

Boundary
--------
Straight paths are less natural than recorded ones.  The walkable grid is
two-dimensional, so pixel truth on rendered candidates remains the authority
for "the actor did not walk through furniture".  Speed range, margin and the
bank-first attempt budget are research placeholders.
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
BANK_ATTEMPTS_KEY = "ROUTE_BANK_ATTEMPTS_BEFORE_SYNTHESIS"
DESIGN_TRIES_KEY = "ROUTE_SYNTHESIS_DESIGN_TRIES"

REASON_SPEED = "synthesis_speed_out_of_range"
REASON_WALKABLE = "synthesis_route_outside_walkable"
REASON_SPEC = "synthesis_infeasible_spec"


@dataclass(frozen=True)
class SynthesisSettings:
    speed_min_mps: float
    speed_max_mps: float
    margin_cm: float
    max_camera_distance_cm: float
    bank_attempts: int
    design_tries: int

    @classmethod
    def from_params(cls, params: dict) -> "SynthesisSettings | None":
        """None when synthesis is off; otherwise every key must be present and sane."""
        if not params.get(ENABLED_KEY):
            return None
        missing = [key for key in (SPEED_KEY, MARGIN_KEY, MAX_DISTANCE_KEY, BANK_ATTEMPTS_KEY)
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
        bank_attempts = int(params[BANK_ATTEMPTS_KEY])
        if bank_attempts < 0:
            raise ValueError(f"{BANK_ATTEMPTS_KEY} must be non-negative")
        tries = int(params.get(DESIGN_TRIES_KEY, 8))
        if tries < 1:
            raise ValueError(f"{DESIGN_TRIES_KEY} must be at least one")
        return cls(speed_min_mps=float(speed[0]), speed_max_mps=float(speed[1]),
                   margin_cm=margin_m * 100.0, max_camera_distance_cm=max_distance,
                   bank_attempts=bank_attempts, design_tries=tries)

    def as_dict(self) -> dict:
        return {"speed_mps_range": [self.speed_min_mps, self.speed_max_mps],
                "walkable_margin_cm": self.margin_cm,
                "max_camera_distance_cm": self.max_camera_distance_cm,
                "bank_attempts_before_synthesis": self.bank_attempts,
                "design_tries_per_attempt": self.design_tries}


@dataclass(frozen=True)
class PointSpec:
    """Where the actor must be at one frame, relative to the camera pose."""
    frame: int
    azimuth_lo_deg: float
    azimuth_hi_deg: float
    distance_lo_cm: float
    distance_hi_cm: float

    def feasible(self) -> bool:
        return (0 <= self.frame < FRAME_COUNT
                and self.azimuth_lo_deg < self.azimuth_hi_deg
                and 0.0 < self.distance_lo_cm <= self.distance_hi_cm)


def _point_from_pose(camera_xy, camera_yaw_deg: float, azimuth_deg: float,
                     distance_cm: float) -> tuple[float, float]:
    bearing = math.radians(float(camera_yaw_deg) + float(azimuth_deg))
    return (float(camera_xy[0]) + distance_cm * math.cos(bearing),
            float(camera_xy[1]) + distance_cm * math.sin(bearing))


def line_samples(frame_a: int, point_a, frame_b: int, point_b, idle_frames: int) -> list[tuple[float, float]]:
    """Base-route samples whose idle-shifted version passes point_a at frame_a
    and point_b at frame_b: base[k] lies on the line at "time" k + idle."""
    if frame_a == frame_b:
        raise ValueError("the two designed frames must differ")
    span = float(frame_b - frame_a)
    dx = (float(point_b[0]) - float(point_a[0])) / span
    dy = (float(point_b[1]) - float(point_a[1])) / span
    return [(float(point_a[0]) + dx * (k + idle_frames - frame_a),
             float(point_a[1]) + dy * (k + idle_frames - frame_a))
            for k in range(FRAME_COUNT)]


def heading_samples(frame: int, point, heading_deg: float, step_cm: float,
                    idle_frames: int) -> list[tuple[float, float]]:
    ux, uy = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    return [(float(point[0]) + ux * step_cm * (k + idle_frames - frame),
             float(point[1]) + uy * step_cm * (k + idle_frames - frame))
            for k in range(FRAME_COUNT)]


def _route_id(design: dict) -> str:
    digest = hashlib.sha1(json.dumps(design, sort_keys=True).encode()).hexdigest()
    return f"synth:{digest[:12]}"


class RouteSynthesizer:
    """Designs straight constant-speed routes on one scene's walkable grid."""

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

    def _speed_ok(self, step_cm: float) -> bool:
        speed_mps = step_cm * self.frame_rate_hz / 100.0
        return self.settings.speed_min_mps <= speed_mps <= self.settings.speed_max_mps

    def _finish(self, samples, design: dict, role: str):
        from scene_sampler import Route  # local import: scene_sampler imports this module

        ok, detail = self.grid.route_ok(samples, self.settings.margin_cm)
        if not ok:
            return self._reject(REASON_WALKABLE)
        displacement_cm = math.dist(samples[0], samples[-1])
        design = dict(design, min_clearance_cm=detail["min_clearance_cm"],
                      worst_frame=detail["worst_frame"], margin_cm=self.settings.margin_cm)
        provenance = {"source": "synthesized", "role": role, "design": design,
                      "grid": {"scene_id": self.grid.scene_id,
                               "arrays_sha256": self.grid.identity["arrays_sha256"]}}
        self.counters["built"] += 1
        # bank identity: implied speed is the end-to-end span over the 5 s clip
        route = Route(_route_id(design), samples, displacement_cm / 100.0 / 5.0,
                      provenance=provenance)
        return route, None

    # -- public ------------------------------------------------------------

    def design(self, rng, camera_xy, camera_yaw_deg: float, specs: Sequence[PointSpec], *,
               idle_frames: int, role: str):
        """One design attempt.  Returns (route, None) or (None, reason).

        One spec: the position at that frame is drawn in the spec, then a
        uniform heading and a uniform speed in the declared range.  Two specs:
        both positions are drawn and the speed follows from their distance and
        the frame gap; it must fall in the declared range.  Every base sample
        must keep the walkable margin."""
        self.counters["designs"] += 1
        specs = list(specs)
        if not specs or len(specs) > 2 or not all(spec.feasible() for spec in specs):
            return self._reject(REASON_SPEC)
        if len(specs) == 2 and specs[0].frame == specs[1].frame:
            return self._reject(REASON_SPEC)
        points = []
        for spec in specs:
            azimuth = spec.azimuth_lo_deg + (spec.azimuth_hi_deg - spec.azimuth_lo_deg) * float(rng.random())
            distance = spec.distance_lo_cm + (spec.distance_hi_cm - spec.distance_lo_cm) * float(rng.random())
            points.append({"frame": int(spec.frame), "azimuth_deg": round(azimuth, 4),
                           "distance_cm": round(distance, 3),
                           "xy_cm": [round(v, 3) for v in _point_from_pose(
                               camera_xy, camera_yaw_deg, azimuth, distance)]})
        design = {"camera_xy_cm": [round(float(camera_xy[0]), 3), round(float(camera_xy[1]), 3)],
                  "camera_yaw_deg": round(float(camera_yaw_deg), 4), "points": points,
                  "idle_frames": int(idle_frames), "shape": "straight_constant_speed"}
        if len(points) == 1:
            heading = 360.0 * float(rng.random())
            speed_mps = (self.settings.speed_min_mps
                         + (self.settings.speed_max_mps - self.settings.speed_min_mps) * float(rng.random()))
            step_cm = speed_mps * 100.0 / self.frame_rate_hz
            design.update(heading_deg=round(heading, 4), speed_mps=round(speed_mps, 4))
            samples = heading_samples(points[0]["frame"], points[0]["xy_cm"], heading, step_cm,
                                      int(idle_frames))
        else:
            a, b = points
            step_cm = math.dist(a["xy_cm"], b["xy_cm"]) / abs(b["frame"] - a["frame"])
            if not self._speed_ok(step_cm):
                return self._reject(REASON_SPEED)
            design.update(speed_mps=round(step_cm * self.frame_rate_hz / 100.0, 4))
            samples = line_samples(a["frame"], a["xy_cm"], b["frame"], b["xy_cm"], int(idle_frames))
        return self._finish(samples, design, role)

    def design_many(self, rng, camera_xy, camera_yaw_deg: float, specs: Sequence[PointSpec], *,
                    idle_frames: int, role: str, tries: int | None = None):
        """Repeat ``design`` up to ``tries`` times; returns (route, last_reason)."""
        reason = REASON_SPEC
        for _ in range(int(tries or self.settings.design_tries)):
            route, reason = self.design(rng, camera_xy, camera_yaw_deg, specs,
                                        idle_frames=idle_frames, role=role)
            if route is not None:
                return route, None
        return None, reason

    def report(self) -> dict:
        return {"settings": self.settings.as_dict(), "grid": self.grid.identity,
                "counters": json.loads(json.dumps(self.counters))}

"""Public joint-construction records. World metres, +Y up, camera looks -Z at yaw 0.

Azimuth is image-right positive (NOT the audio left-positive convention).
Planning certificates never replace the unchanged native QA truth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol
import math
import numpy as np


@dataclass(frozen=True)
class CameraPose:
    position_m: tuple[float, float, float]
    yaw_deg: float
    camera_id: str = "camera"
    hfov_deg: float = 85.0
    width: int = 640
    height: int = 360

    def __post_init__(self):
        if (len(self.position_m) != 3 or not np.isfinite(self.position_m).all()
                or not math.isfinite(self.yaw_deg) or not 0 < self.hfov_deg < 180
                or not isinstance(self.width, int) or not isinstance(self.height, int)
                or self.width < 1 or self.height < 1):
            raise ValueError("invalid fixed camera geometry")
        object.__setattr__(self, "position_m", tuple(map(float, self.position_m)))

    @property
    def forward(self):
        a = math.radians(self.yaw_deg)
        return np.array([math.sin(a), 0., -math.cos(a)])

    @property
    def right(self):
        return np.cross(self.forward, [0., 1., 0.])

    @property
    def focal_px(self):
        return self.width / (2 * math.tan(math.radians(self.hfov_deg) / 2))


@dataclass(frozen=True)
class BodySpec:
    species: str = "human"
    radius_m: float = .35
    height_m: float = 1.8
    margin_m: float = .08

    def __post_init__(self):
        if self.species not in {"human", "dog", "cat", "slide"}:
            raise ValueError("unsupported species")
        if not np.isfinite([self.radius_m, self.height_m, self.margin_m]).all() or min(self.radius_m, self.height_m) <= 0 or self.margin_m < 0:
            raise ValueError("invalid body envelope")

    @classmethod
    def for_species(cls, species, **overrides):
        radius, height = {"human": (.35, 1.8), "dog": (.25, .75), "cat": (.20, .45), "slide": (.20, .5)}[species]
        return cls(**dict(dict(species=species, radius_m=radius, height_m=height), **overrides))


@dataclass(frozen=True)
class FamilyRequest:
    family: str
    branch: str
    target_ids: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ActorPlan:
    entity_id: str
    positions_m: np.ndarray
    body: BodySpec = field(default_factory=BodySpec)
    headings_rad: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SoundEvent:
    entity_id: str
    sound_id: str
    sound_class: str
    start_s: float
    end_s: float
    # The renderer must enforce zero output after this declared wet support.
    wet_end_s: float | None = None


@dataclass
class ScenePlan:
    camera: CameraPose
    actors: dict[str, ActorPlan]
    sounds: tuple[SoundEvent, ...] = ()
    frame_rate_hz: float = 30.
    duration_s: float = 30.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # pass / fail / unknown
    reason: str = ""
    margin: float | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in {"pass", "fail", "unknown"}:
            raise ValueError("invalid planning check status")


@dataclass
class Certificate:
    checks: tuple[CheckResult, ...]
    authority: str = "planning_geometry_only"

    @property
    def passed(self):
        return bool(self.checks) and all(c.status == "pass" for c in self.checks)


@dataclass
class SolveResult:
    status: str  # certified / unavailable / budget_exhausted
    plan: ScenePlan | None = None
    certificate: Certificate | None = None
    reasons: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


class FamilyPlugin(Protocol):
    """Family modules own answer constraints; the core owns shared geometry.

    candidates must construct for request.branch, not select a branch afterwards.
    verify returns checks for ALL requested family/branch conditions. No truth writes.
    """
    def candidates(self, context: Any, request: FamilyRequest) -> Iterable[ScenePlan]: ...
    def verify(self, context: Any, request: FamilyRequest, plan: ScenePlan) -> Iterable[CheckResult]: ...

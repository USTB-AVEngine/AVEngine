"""Solve fixed-camera visibility crossings and occlusion sequences for QA candidates.

``conditioned_sampler`` can already say "the anchor stays inside the field of
view for its whole audible window" (``anchor_visibility``) and "one ray from the
camera to the emitter and to the body proxy is blocked"
(``anchor_line_of_sight``).  Neither statement is the question QA-07 and
QA-09..QA-11 ask.  A frustum test on one body point cannot express an
``out_of_view`` to ``visible`` crossing, and a blocked ray is a different
measurement from a ``fully_occluded`` pixel state: the ray can be blocked while
a shoulder still shows, and it can be clear while the visible footprint is a
handful of pixels.  This module closes that gap on the planning side and states
where its own authority stops.

Three tiers, named so that no caller can mistake one for another:

``frustum``
    Camera pose, body-sample projection and the pixel column of every sample.
    Needs no scene geometry.  Refutes a requirement when the subject can never
    leave or enter the frustum in the clip.

``frustum_and_ray``
    The same samples plus exact retained-triangle rays through
    :func:`avengine.qa.answerability.line_of_sight`, actor cylinders and, where
    the room declares them, static occluder boxes.  Ranks candidates and
    attributes occlusion to a named instance.

``native_pixel``
    :func:`avengine.qa.pixel_visibility.compile_pixel_visibility_truth` or its
    metric-depth twin, produced by the room's own renderer, plus
    :func:`avengine.rooms.qa_evidence.derive_actor_occluders` for occluder
    identity.  This tier is the only answer authority.

The consequence is deliberate and is enforced by :data:`SCREEN_VERDICTS`: the
screen never returns a positive verdict.  A screened candidate is
``consistent`` (worth spending a render on), ``refuted`` (not worth one) or
``undetermined``, and the pixel state is decided afterwards by
:func:`accept_native_visibility`.

The screen's own error was measured, not assumed.  Twelve instances over five
retained native captures (two MP3D, two HM3D, one UE/SPEAR Kujiale; 1800
compared frames, 2026-09-10) gave these numbers, and they are what set the
boundary:

* One direction held everywhere.  An in-frustum body sample means the
  target-only footprint is not empty, and in 1800 frames the renderer never
  once reported ``out_of_view`` where the screen had a sample inside the image
  (0 violations).  That single implication is the only thing the screen
  refutes.
* The opposite direction did not hold.  The nine-sample hull under-covers a
  silhouette, so 112 frames predicted ``out_of_view`` were frames the renderer
  saw the target in.  Those are a pruning cost, so such a frame refutes
  nothing.
* The occlusion split depends on which geometry is screened.  All five captures
  declare ``static_geometry.source = acoustic_package_arrays`` - the acoustic
  proxy, which AGENTS.md deliberately keeps separate from real visual geometry
  - and the screen under-predicted occlusion on 152 frames, 150 of them the one
  UE/SPEAR capture where the proxy missed the occluder for a clip the renderer
  called ``fully_occluded`` end to end.  So an occlusion requirement screened
  against anything but the geometry that renders is left ``undetermined``
  rather than refused (see :data:`GEOMETRY_AUTHORITIES`); refusing on that
  basis would discard exactly the candidates QA-09 to QA-11 need.
* Whole-state agreement was 0.44 overall (787 of 1800), 0.62 on MP3D, 0.44 on
  HM3D and 0.25 on the proxy-limited Kujiale capture.  That is a ranking
  signal, not an acceptance criterion.

Pruning is therefore a throughput decision with a recorded error rate, and
:func:`compare_screen_to_witness` re-measures all four counts on any new
episode rather than trusting these.

Room dispatch runs off the RoomPackage capability report and the declared
renderer, never off a room identifier: the two Habitat rooms reach the paired
modal/target-only semantic pass, the two UE/SPEAR rooms reach the metric-depth
pass, and a new room that declares the same dimensions arrives with the same
facilities.  A subject that cannot walk is not a reason to refuse a whole
group: only a requirement that needs the *subject's own* frustum crossing is
inapplicable to a fixed device, and every occlusion requirement stays open
because an occluder or the scene supplies the change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from avengine.dataset.source_capabilities import (
    CAPABILITY_STATES,
    STATE_AVAILABLE,
    STATE_EVIDENCE_MISSING,
    STATE_NOT_APPLICABLE,
    STATE_NOT_IMPLEMENTED,
)
from avengine.qa import unified_catalog as catalog
from avengine.qa.answerability import MeshHandle, line_of_sight
from avengine.qa.pixel_visibility import (
    PIXEL_VISIBILITY_AUTHORITIES,
    PIXEL_VISIBILITY_AUTHORITY,
    PIXEL_VISIBILITY_DEPTH_AUTHORITY,
    PIXEL_VISIBILITY_STATES,
    PixelVisibilityError,
    bind_pixel_visibility_truth,
)
from avengine.qa.unified_catalog import VISIBILITY_STATES, VISIBLE_STATES

SCHEMA = "avengine_qa_conditioned_visibility_v1"

# The shared planning frame.  Every position this module reads is metres,
# +Y up, right handed, exactly like ``capture.neutral_readback.COORDINATE_FRAME``
# and ``qa_plan_adapters._package_mesh``.
COORDINATE_FRAME = {"linear_unit": "meter", "up_axis": "+Y", "handedness": "right"}

SCREEN_TIERS = ("frustum", "frustum_and_ray", "native_pixel")

# Array storage does not determine geometric authority. A room may use the
# same audited render surfaces for acoustics and visibility. Legacy proxy
# arrays remain conservative unless the room explicitly declares otherwise.
GEOMETRY_AUTHORITIES = ("visual_mesh", "acoustic_proxy_mesh", "unknown")
_ACOUSTIC_PROXY_SOURCES = frozenset({"acoustic_package_arrays"})


def geometry_authority_for_package(package: Mapping[str, Any] | None) -> str:
    """Read which geometry a room package's static arrays actually represent."""

    geometry = package.get("static_geometry") if isinstance(package, Mapping) else None
    if not isinstance(geometry, Mapping):
        return "unknown"
    if geometry.get("render_surface") is True:
        return "visual_mesh"
    source = geometry.get("source")
    if isinstance(source, str) and source in _ACOUSTIC_PROXY_SOURCES:
        return "acoustic_proxy_mesh"
    representation = geometry.get("representation")
    if isinstance(representation, str) and representation == "visual_surface_mesh":
        return "visual_mesh"
    return "unknown"

# A screen verdict is never a pass.  ``consistent`` means "nothing measured
# here contradicts the requirement, so this candidate is worth rendering".
SCREEN_VERDICTS = ("consistent", "undetermined", "refuted")

CONFIRMATION_AUTHORITY = "native_pixel_witness"

# Entity classes that hold still.  Same set as
# ``conditioned_sampler.RIGID`` - a fixed device is a legal subject and a
# legal occluder, it simply cannot supply its own frustum crossing.
IMMOBILE_ENTITY_CLASSES = frozenset({"rigid_object", "rigid_static_object"})

REQUIREMENT_KINDS = (
    "out_of_view_to_visible",
    "visibility_state",
    "fully_occluded_then_visible",
    "fully_occluded_without_return",
    "visible_occluded_to_visible_clear",
    "registered_occluder_visible",
)

# What each requirement needs to *change* inside one fixed-camera clip.  An
# empty tuple means a persistent state satisfies it, which is why a bolted-down
# device permanently hidden behind a counter is a legal QA-09 ``no``.
_REQUIRED_DYNAMICS: dict[str, tuple[str, ...]] = {
    "out_of_view_to_visible": ("subject_frustum_crossing",),
    "visibility_state": (),
    "fully_occluded_then_visible": ("subject_or_occluder_motion",),
    "fully_occluded_without_return": (),
    "visible_occluded_to_visible_clear": ("subject_or_occluder_motion",),
    "registered_occluder_visible": (),
}

ENTRY_SIDES = ("left", "right")

# The planning knobs this module implements, keyed exactly as
# ``generation_conditions.KNOB_GAPS`` names them, so P02-R1 can replace two
# static gap entries with a live capability query.
_KNOB_IMPLEMENTATIONS: dict[str, dict[str, Any]] = {
    "visibility_transition": {
        "values": ("out_of_view_to_visible",),
        "requirement_kinds": ("out_of_view_to_visible",),
        "solver": "avengine.rooms.conditioned_visibility.solve_visibility_candidates",
        "screen": "avengine.rooms.conditioned_visibility.screen_visibility_series",
        "acceptance": "avengine.rooms.conditioned_visibility.accept_native_visibility",
        "note": "the crossing is solved over the whole clip, not over the audible "
        "window, and the entry side is the sign of the predicted centroid column "
        "offset under the judge's own dead zone",
    },
    "pixel_occlusion_transition": {
        "values": (
            "fully_occluded",
            "fully_occluded_then_visible",
            "fully_occluded_without_return",
            "registered_occluder_visible",
            "visible_occluded_to_visible_clear",
        ),
        "requirement_kinds": (
            "visibility_state",
            "fully_occluded_then_visible",
            "fully_occluded_without_return",
            "registered_occluder_visible",
            "visible_occluded_to_visible_clear",
        ),
        "solver": "avengine.rooms.conditioned_visibility.solve_visibility_candidates",
        "screen": "avengine.rooms.conditioned_visibility.screen_visibility_series",
        "acceptance": "avengine.rooms.conditioned_visibility.accept_native_visibility",
        "note": "a multi-sample body screen replaces the single emitter/body ray; "
        "the screen can only refute a full occlusion, never confirm one, so the "
        "pixel state stays the answer authority",
    },
}

# ``pixel_occlusion_transition`` knob value -> requirement kind (plus the state
# a bare ``fully_occluded`` value asks for).
_TRANSITION_TO_REQUIREMENT: dict[str, tuple[str, str | None]] = {
    "fully_occluded": ("visibility_state", "fully_occluded"),
    "fully_occluded_then_visible": ("fully_occluded_then_visible", None),
    "fully_occluded_without_return": ("fully_occluded_without_return", None),
    "registered_occluder_visible": ("registered_occluder_visible", None),
    "visible_occluded_to_visible_clear": ("visible_occluded_to_visible_clear", None),
}


class ConditionedVisibilityError(ValueError):
    """A visibility requirement, camera, track or witness is unusable."""


# ---------------------------------------------------------------------------
# Small validators.  Every one names the field, because "invalid input" and
# "this room does not declare that dimension" lead to different fixes.
# ---------------------------------------------------------------------------


def _text(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConditionedVisibilityError(f"{owner} must be a non-empty string")
    return value


def _finite(value: Any, *, owner: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ConditionedVisibilityError(f"{owner} must be a real number") from error
    if not math.isfinite(number):
        raise ConditionedVisibilityError(f"{owner} must be finite")
    return number


def _positive(value: Any, *, owner: str) -> float:
    number = _finite(value, owner=owner)
    if number <= 0.0:
        raise ConditionedVisibilityError(f"{owner} must be positive")
    return number


def _index(value: Any, *, owner: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ConditionedVisibilityError(f"{owner} must be an integer")
    number = int(value)
    if number < minimum:
        raise ConditionedVisibilityError(f"{owner} must be at least {minimum}")
    return number


def _vector3(value: Any, *, owner: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ConditionedVisibilityError(f"{owner} must be a finite 3-vector")
    return array


def _resolution(value: Any, *, owner: str) -> tuple[int, int]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise ConditionedVisibilityError(f"{owner} must be [height, width]")
    height = _index(value[0], owner=f"{owner}[0]", minimum=1)
    width = _index(value[1], owner=f"{owner}[1]", minimum=1)
    return height, width


# ---------------------------------------------------------------------------
# Screen policy: the body-sample grid and the margins that make a call
# marginal rather than decisive.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreenPolicy:
    """How densely a body is sampled and when a screened call is marginal.

    The default grid is the one the retained QA-v3 predictor was validated
    with (``tools/qa/visibility_prediction.py``): three heights at 0.2/0.6/1.0
    of the body height and three lateral offsets spanning the body *width*, not
    its length.  On 59 captured Apartment candidates (948 in-view frames,
    2026-09-02) narrowing the lateral span from +-0.4 m to +-0.2 m moved tier
    agreement from 0.695 to 0.753 and below-half agreement from 820 to 887 of
    948.  ``dense_5x5_v1`` keeps the same span and only refines the grid; it
    does not change the tier ladder or claim a stronger authority.
    """

    policy_id: str = "validated_3x3_v1"
    height_fractions: tuple[float, ...] = (0.2, 0.6, 1.0)
    lateral_fractions: tuple[float, ...] = (-0.5, 0.0, 0.5)
    near_m: float = 0.1
    # A sample whose predicted column/row sits this close to an image border
    # cannot decide in-view versus out-of-view at screen resolution.
    edge_margin_px: float = 2.0
    # A blocked ray is re-cast this much short of the sample so a grazing hit
    # on the body's own surface is not read as an occluder.
    ray_backoff_m: float = 0.05
    # A crossing counts as an entry only once the body reaches this fraction of
    # the image width inside the entry edge. A body grazing the edge as a
    # sliver is not an entry: one QA-07 capture (2026-09-12) entered 4 to 12 px
    # after a consistent screen and was refused by the pixel judge.
    entry_depth_fraction: float = 0.05

    def __post_init__(self) -> None:
        _text(self.policy_id, owner="policy_id")
        if not self.height_fractions or not self.lateral_fractions:
            raise ConditionedVisibilityError("screen policy needs body samples")
        for value in self.height_fractions:
            number = _finite(value, owner="height_fractions[]")
            if not 0.0 < number <= 1.0:
                raise ConditionedVisibilityError(
                    "height_fractions must lie in (0, 1]"
                )
        for value in self.lateral_fractions:
            _finite(value, owner="lateral_fractions[]")
        _positive(self.near_m, owner="near_m")
        if _finite(self.edge_margin_px, owner="edge_margin_px") < 0.0:
            raise ConditionedVisibilityError("edge_margin_px must not be negative")
        if _finite(self.ray_backoff_m, owner="ray_backoff_m") < 0.0:
            raise ConditionedVisibilityError("ray_backoff_m must not be negative")
        depth = _finite(self.entry_depth_fraction, owner="entry_depth_fraction")
        if not 0.0 <= depth < 0.5:
            raise ConditionedVisibilityError("entry_depth_fraction must lie in [0, 0.5)")

    @property
    def sample_count(self) -> int:
        return len(self.height_fractions) * len(self.lateral_fractions)

    def as_report(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "height_fractions": list(self.height_fractions),
            "lateral_fractions": list(self.lateral_fractions),
            "sample_count": self.sample_count,
            "near_m": self.near_m,
            "edge_margin_px": self.edge_margin_px,
            "ray_backoff_m": self.ray_backoff_m,
            "entry_depth_fraction": self.entry_depth_fraction,
            "lateral_span": "body_width_perpendicular_to_the_sight_line",
            "validated_against": "tools/qa/validate_visibility_prediction.py",
        }


VALIDATED_SCREEN_POLICY = ScreenPolicy()
DENSE_SCREEN_POLICY = ScreenPolicy(
    policy_id="dense_5x5_v1",
    height_fractions=(0.1, 0.3, 0.55, 0.8, 1.0),
    lateral_fractions=(-0.5, -0.25, 0.0, 0.25, 0.5),
)
SCREEN_POLICIES = {
    VALIDATED_SCREEN_POLICY.policy_id: VALIDATED_SCREEN_POLICY,
    DENSE_SCREEN_POLICY.policy_id: DENSE_SCREEN_POLICY,
}


def screen_policy(policy: ScreenPolicy | str | None) -> ScreenPolicy:
    """Resolve a named policy, an explicit policy, or the validated default."""

    if policy is None:
        return VALIDATED_SCREEN_POLICY
    if isinstance(policy, ScreenPolicy):
        return policy
    name = _text(policy, owner="screen policy id")
    try:
        return SCREEN_POLICIES[name]
    except KeyError as error:
        raise ConditionedVisibilityError(
            f"unknown screen policy {name!r}; registered policies are "
            f"{sorted(SCREEN_POLICIES)}"
        ) from error


# ---------------------------------------------------------------------------
# Body proxy.  The runtime source registry declares emitter anchor heights but
# no body extent, so the vertical reach can be derived from a registered
# anchor while the lateral half-width stays an explicit caller value with its
# provenance recorded.
# ---------------------------------------------------------------------------

BODY_HEIGHT_SOURCES = (
    "caller_declared",
    "registered_emitter_anchor_height",
    "placeholder_default",
)

# Same placeholders as ``tools/qa/visibility_prediction.DEFAULT_BODY_M``.
PLACEHOLDER_BODY_HEIGHT_M = 0.5
PLACEHOLDER_BODY_WIDTH_M = 0.4


@dataclass(frozen=True)
class BodyProxy:
    """The vertical reach and lateral width the screen sweeps for one subject."""

    height_m: float
    width_m: float = PLACEHOLDER_BODY_WIDTH_M
    height_source: str = "caller_declared"
    width_source: str = "placeholder_default"

    def __post_init__(self) -> None:
        _positive(self.height_m, owner="body height_m")
        _positive(self.width_m, owner="body width_m")
        if self.height_source not in BODY_HEIGHT_SOURCES:
            raise ConditionedVisibilityError(
                f"body height_source must be one of {list(BODY_HEIGHT_SOURCES)}"
            )
        _text(self.width_source, owner="body width_source")

    def as_report(self) -> dict[str, Any]:
        return {
            "height_m": self.height_m,
            "width_m": self.width_m,
            "height_source": self.height_source,
            "width_source": self.width_source,
            "claim_boundary": (
                "a screening proxy for the silhouette; it is not the asset's "
                "reviewed geometry and never overrides a rendered footprint"
            ),
        }


@dataclass(frozen=True)
class ActorBodyEnvelope:
    """Per-frame full-body points supplied by registered asset geometry/poses.

    vertices_m is in the same world frame as ActorTrack.positions_m.
    The caller derives it from the registered visual asset and its allowed
    animation poses; this class does not infer a silhouette from an emitter
    anchor or claim containment between supplied frames.
    """

    vertices_m: np.ndarray
    source_ref: str
    pose_coverage: str = "caller_declared_actual_or_allowed_poses"

    def __post_init__(self) -> None:
        array = np.asarray(self.vertices_m, dtype=float)
        if (
            array.ndim != 3
            or array.shape[0] == 0
            or array.shape[1] == 0
            or array.shape[2] != 3
        ):
            raise ConditionedVisibilityError(
                "body envelope vertices_m must be [frame, vertex, 3]"
            )
        if not np.all(np.isfinite(array)):
            raise ConditionedVisibilityError(
                "body envelope vertices_m must be finite"
            )
        array = np.ascontiguousarray(array)
        array.setflags(write=False)
        object.__setattr__(self, "vertices_m", array)
        _text(self.source_ref, owner="body envelope source_ref")
        _text(self.pose_coverage, owner="body envelope pose_coverage")

    @property
    def frame_count(self) -> int:
        return int(self.vertices_m.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices_m.shape[1])

    def as_report(self) -> dict[str, Any]:
        return {
            "source_ref": self.source_ref,
            "pose_coverage": self.pose_coverage,
            "frame_count": self.frame_count,
            "vertices_per_frame": self.vertex_count,
            "authority": "caller_supplied_full_skinned_body_envelope",
            "claim_boundary": (
                "complete supplied mesh points constrain the screened frustum at "
                "the supplied poses; no continuous containment or pixel state is "
                "claimed here"
            ),
        }


def body_proxy_from_emitter_anchor(
    emitter_offset_m: Sequence[float] | None,
    *,
    width_m: float | None = None,
) -> BodyProxy:
    """Derive a body proxy from the registered emitter anchor offset.

    The runtime source registry declares each asset's emitter anchor offset in
    its final scaled root space, and that anchor height is the only registered
    vertical dimension available at planning time.  It is used as the sampled
    reach exactly as registered: a mouth anchor sits below a human's crown and
    a muzzle anchor sits near a dog's, so any single factor from anchor to
    crown would be a per-species guess written into production Python.  The
    reach is therefore conservative at the top of the silhouette, which only
    ever makes the screen more willing to predict ``out_of_view`` or
    ``fully_occluded`` - the two states it is not allowed to confirm anyway.
    A missing or non-positive height falls back to the recorded placeholder,
    and the returned proxy says which of the two happened.
    """

    height: float | None = None
    if emitter_offset_m is not None:
        offset = _vector3(emitter_offset_m, owner="emitter_offset_m")
        if float(offset[1]) > 0.0:
            height = float(offset[1])
    if height is None:
        return BodyProxy(
            height_m=PLACEHOLDER_BODY_HEIGHT_M,
            width_m=PLACEHOLDER_BODY_WIDTH_M if width_m is None else float(width_m),
            height_source="placeholder_default",
            width_source="placeholder_default" if width_m is None else "caller_declared",
        )
    return BodyProxy(
        height_m=height,
        width_m=PLACEHOLDER_BODY_WIDTH_M if width_m is None else float(width_m),
        height_source="registered_emitter_anchor_height",
        width_source="placeholder_default" if width_m is None else "caller_declared",
    )


# ---------------------------------------------------------------------------
# Camera pose and actor tracks.  Both accept exactly the shapes
# ``conditioned_sampler`` already builds, so the sampler hands its own camera
# candidate and its own ``paths``/``bodies`` arrays straight over.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraPose:
    """One fixed camera candidate in the shared planning frame."""

    candidate_id: str
    position_m: tuple[float, float, float]
    forward: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    horizontal_fov_deg: float
    resolution_hw: tuple[int, int]

    def __post_init__(self) -> None:
        _text(self.candidate_id, owner="camera candidate_id")
        _vector3(self.position_m, owner="camera position_m")
        for name in ("forward", "right", "up"):
            axis = _vector3(getattr(self, name), owner=f"camera {name}")
            if not math.isclose(float(np.linalg.norm(axis)), 1.0, abs_tol=1.0e-6):
                raise ConditionedVisibilityError(
                    f"camera {name} must be a unit vector"
                )
        fov = _positive(self.horizontal_fov_deg, owner="horizontal_fov_deg")
        if fov >= 180.0:
            raise ConditionedVisibilityError("horizontal_fov_deg must be below 180")
        _resolution(self.resolution_hw, owner="camera resolution_hw")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CameraPose":
        """Read the camera dict ``select_camera_and_schedule`` returns."""

        if not isinstance(value, Mapping):
            raise ConditionedVisibilityError("camera must be a mapping")
        motion = value.get("motion", "static")
        if motion != "static":
            raise ConditionedVisibilityError(
                "conditioned visibility solves fixed cameras only; the request "
                f"declares camera motion {motion!r}"
            )
        basis = value.get("basis")
        if not isinstance(basis, Mapping):
            raise ConditionedVisibilityError("camera basis must be a mapping")
        height, width = _resolution(
            value.get("resolution_hw"), owner="camera resolution_hw"
        )
        return cls(
            candidate_id=_text(value.get("candidate_id"), owner="candidate_id"),
            position_m=tuple(
                float(item)
                for item in _vector3(value.get("position_m"), owner="position_m")
            ),
            forward=tuple(
                float(item) for item in _vector3(basis.get("forward"), owner="forward")
            ),
            right=tuple(
                float(item) for item in _vector3(basis.get("right"), owner="right")
            ),
            up=tuple(float(item) for item in _vector3(basis.get("up"), owner="up")),
            horizontal_fov_deg=_positive(
                value.get("horizontal_fov_deg"), owner="horizontal_fov_deg"
            ),
            resolution_hw=(height, width),
        )

    @property
    def dead_zone_px(self) -> float:
        """The judge's own entry-side dead zone: ``max(1, width * 0.02)``."""

        return max(1.0, float(self.resolution_hw[1]) * 0.02)

    @property
    def center_column_px(self) -> float:
        """The judge's own image centre column: ``(width - 1) / 2``."""

        return (float(self.resolution_hw[1]) - 1.0) / 2.0

    def as_report(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "position_m": list(self.position_m),
            "basis": {
                "forward": list(self.forward),
                "right": list(self.right),
                "up": list(self.up),
            },
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "resolution_hw": list(self.resolution_hw),
            "motion": "static",
        }


@dataclass(frozen=True)
class ActorTrack:
    """One instance's per-frame root position plus what it can do."""

    instance_id: str
    positions_m: np.ndarray
    body: BodyProxy
    entity_class: str = "articulated_human"
    registered_occluder: bool = True
    body_envelope: ActorBodyEnvelope | None = None

    def __post_init__(self) -> None:
        _text(self.instance_id, owner="instance_id")
        array = np.asarray(self.positions_m, dtype=float)
        if array.ndim != 2 or array.shape[1] != 3 or not len(array):
            raise ConditionedVisibilityError(
                f"{self.instance_id}: positions_m must be [frame, 3]"
            )
        if not np.all(np.isfinite(array)):
            raise ConditionedVisibilityError(
                f"{self.instance_id}: positions_m must be finite"
            )
        object.__setattr__(self, "positions_m", array)
        if not isinstance(self.body, BodyProxy):
            raise ConditionedVisibilityError(
                f"{self.instance_id}: body must be a BodyProxy"
            )
        if self.body_envelope is not None:
            if not isinstance(self.body_envelope, ActorBodyEnvelope):
                raise ConditionedVisibilityError(
                    f"{self.instance_id}: body_envelope must be an ActorBodyEnvelope"
                )
            if self.body_envelope.frame_count != int(array.shape[0]):
                raise ConditionedVisibilityError(
                    f"{self.instance_id}: body envelope and root positions "
                    "must share one frame clock"
                )
        _text(self.entity_class, owner="entity_class")

    @property
    def frame_count(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def can_move(self) -> bool:
        """Whether this instance's own root position changes in the clip."""

        if self.entity_class in IMMOBILE_ENTITY_CLASSES:
            return False
        if self.frame_count < 2:
            return False
        return bool(
            np.any(
                np.linalg.norm(np.diff(self.positions_m, axis=0), axis=1) > 1.0e-6
            )
        )


@dataclass(frozen=True)
class StaticOccluder:
    """A registered static object as an axis-aligned box in the shared frame.

    Room layouts store furniture boxes in their own authoring frame, so the
    caller converts once through :func:`static_occluders_from_layout` and this
    class only ever holds shared-frame metres.
    """

    occluder_id: str
    minimum_m: tuple[float, float, float]
    maximum_m: tuple[float, float, float]
    semantic_class: str | None = None

    def __post_init__(self) -> None:
        _text(self.occluder_id, owner="occluder_id")
        low = _vector3(self.minimum_m, owner=f"{self.occluder_id}.minimum_m")
        high = _vector3(self.maximum_m, owner=f"{self.occluder_id}.maximum_m")
        if np.any(low >= high):
            raise ConditionedVisibilityError(
                f"{self.occluder_id}: minimum_m must be strictly below maximum_m"
            )


# The two authoring-to-shared transforms this repository already declares.
# ``qa_plan_adapters`` applies the first to a retained furniture/native room
# mesh; the second is for a layout that already stores the shared frame.
AUTHORING_TO_SHARED_TRANSFORMS = {
    "authoring_xyz_m_to_xz_negative_y_m": lambda p: np.column_stack(
        (p[:, 0], p[:, 2], -p[:, 1])
    ),
    "identity_shared_meter_y_up": lambda p: p,
}


def static_occluders_from_layout(
    layout: Mapping[str, Any] | None,
    *,
    authoring_to_shared: str | None,
    include_movable: bool = False,
) -> tuple[tuple[StaticOccluder, ...], dict[str, Any]]:
    """Convert a room layout's declared objects into shared-frame boxes.

    The transform must be named explicitly.  A layout whose frame the caller
    cannot name returns no occluders and the reason why, because guessing an
    up axis silently moves every box and the resulting occluder identity would
    look plausible and be wrong.
    """

    objects = layout.get("objects") if isinstance(layout, Mapping) else None
    if not isinstance(objects, list) or not objects:
        return (), {
            "state": STATE_EVIDENCE_MISSING,
            "reason": "the room layout declares no object list, so static "
            "occluder identity has no planning-time source",
            "object_count": 0,
        }
    if authoring_to_shared is None:
        return (), {
            "state": STATE_EVIDENCE_MISSING,
            "reason": "the layout object frame was not named; refusing to guess "
            "an authoring-to-shared transform",
            "object_count": len(objects),
        }
    try:
        transform = AUTHORING_TO_SHARED_TRANSFORMS[authoring_to_shared]
    except KeyError as error:
        raise ConditionedVisibilityError(
            f"unknown authoring_to_shared transform {authoring_to_shared!r}; "
            f"declared transforms are {sorted(AUTHORING_TO_SHARED_TRANSFORMS)}"
        ) from error
    occluders: list[StaticOccluder] = []
    skipped: list[dict[str, Any]] = []
    for record in objects:
        if not isinstance(record, Mapping):
            continue
        object_id = record.get("object_id")
        bounds = record.get("bounds_xyz_m")
        if not isinstance(object_id, str) or not object_id:
            continue
        if not include_movable and not bool(record.get("static", True)):
            skipped.append({"object_id": object_id, "reason": "declared movable"})
            continue
        if (
            not isinstance(bounds, Sequence)
            or len(bounds) != 2
            or any(not isinstance(item, Sequence) or len(item) != 3 for item in bounds)
        ):
            skipped.append({"object_id": object_id, "reason": "no bounds_xyz_m"})
            continue
        corners = np.asarray(
            [
                [bounds[i][0], bounds[j][1], bounds[k][2]]
                for i in (0, 1)
                for j in (0, 1)
                for k in (0, 1)
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(corners)):
            skipped.append({"object_id": object_id, "reason": "non-finite bounds"})
            continue
        shared = np.asarray(transform(corners), dtype=float)
        low = shared.min(axis=0)
        high = shared.max(axis=0)
        if np.any(low >= high):
            skipped.append({"object_id": object_id, "reason": "degenerate box"})
            continue
        occluders.append(
            StaticOccluder(
                occluder_id=object_id,
                minimum_m=tuple(float(item) for item in low),
                maximum_m=tuple(float(item) for item in high),
                semantic_class=(
                    str(record["semantic_class"])
                    if isinstance(record.get("semantic_class"), str)
                    else None
                ),
            )
        )
    report = {
        "state": STATE_AVAILABLE if occluders else STATE_EVIDENCE_MISSING,
        "reason": None
        if occluders
        else "no declared object survived box conversion",
        "object_count": len(objects),
        "occluder_count": len(occluders),
        "authoring_to_shared": authoring_to_shared,
        "skipped": skipped,
    }
    return tuple(occluders), report


# ---------------------------------------------------------------------------
# Requirements.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisibilityRequirement:
    """One pixel-visibility statement a candidate has to be able to support."""

    kind: str
    subject: str
    state: str | None = None
    # Some questions accept any of several pixel states rather than one. QA-02
    # and QA-20 are answerable whenever the emitter is visible at all, clear or
    # partly occluded, and a single ``state`` cannot say that: two separate
    # requirements would mean both states at once. ``allowed_states`` is that
    # disjunction, and ``states`` below is what every judge reads, so a caller
    # that states one exact state behaves exactly as it did.
    allowed_states: tuple[str, ...] = ()
    side: str | None = None
    min_state_frames: int = 1
    min_sustain_frames: int = 2
    require_complete_coverage: bool = False
    require_publishable_window: bool = False
    occluder_subject: str | None = None
    observation_windows: tuple[tuple[int, int], ...] = ()
    public_time_precision: int = 0
    source_condition_key: str | None = None
    qa_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in REQUIREMENT_KINDS:
            raise ConditionedVisibilityError(
                f"unknown visibility requirement kind {self.kind!r}; supported "
                f"kinds are {list(REQUIREMENT_KINDS)}"
            )
        _text(self.subject, owner="requirement subject")
        object.__setattr__(
            self, "allowed_states",
            tuple(str(value) for value in (self.allowed_states or ())),
        )
        if self.kind == "visibility_state":
            if self.state is not None and self.allowed_states:
                raise ConditionedVisibilityError(
                    "a visibility_state requirement states one exact state or a set "
                    "of accepted states, not both"
                )
            named = (self.state,) if self.state is not None else self.allowed_states
            if not named:
                raise ConditionedVisibilityError(
                    "a visibility_state requirement needs a state from "
                    f"{list(VISIBILITY_STATES)}"
                )
            unknown = [value for value in named if value not in VISIBILITY_STATES]
            if unknown:
                raise ConditionedVisibilityError(
                    "a visibility_state requirement needs a state from "
                    f"{list(VISIBILITY_STATES)}; got {unknown}"
                )
        elif self.state is not None or self.allowed_states:
            raise ConditionedVisibilityError(
                f"{self.kind} does not take an explicit state"
            )
        if self.kind == "out_of_view_to_visible":
            if self.side is not None and self.side not in ENTRY_SIDES:
                raise ConditionedVisibilityError(
                    f"entry side must be one of {list(ENTRY_SIDES)}"
                )
        elif self.side is not None:
            raise ConditionedVisibilityError(f"{self.kind} does not take a side")
        _index(self.min_state_frames, owner="min_state_frames", minimum=1)
        _index(self.min_sustain_frames, owner="min_sustain_frames", minimum=1)
        _index(self.public_time_precision, owner="public_time_precision", minimum=0)
        if self.occluder_subject is not None:
            _text(self.occluder_subject, owner="occluder_subject")
        for window in self.observation_windows:
            if (
                not isinstance(window, Sequence)
                or len(window) != 2
                or _index(window[0], owner="observation window start") >= _index(
                    window[1], owner="observation window end", minimum=1
                )
            ):
                raise ConditionedVisibilityError(
                    "observation windows must be half-open [start, end) frames"
                )

    @property
    def states(self) -> tuple[str, ...]:
        """Every pixel state that satisfies this requirement."""
        if self.state is not None:
            return (self.state,)
        return self.allowed_states

    @property
    def required_dynamics(self) -> tuple[str, ...]:
        return _REQUIRED_DYNAMICS[self.kind]

    def as_report(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "state": self.state,
            "allowed_states": list(self.allowed_states),
            "side": self.side,
            "min_state_frames": self.min_state_frames,
            "min_sustain_frames": self.min_sustain_frames,
            "require_complete_coverage": self.require_complete_coverage,
            "require_publishable_window": self.require_publishable_window,
            "occluder_subject": self.occluder_subject,
            "observation_windows": [list(window) for window in self.observation_windows],
            "public_time_precision": self.public_time_precision,
            "source_condition_key": self.source_condition_key,
            "qa_id": self.qa_id,
            "required_dynamics": list(self.required_dynamics),
            "confirmation_authority": CONFIRMATION_AUTHORITY,
        }


def requirements_from_planning(
    planning: Mapping[str, Any],
    *,
    subject: str,
    evidence: Mapping[str, Any] | None = None,
    observation_windows: Sequence[Sequence[int]] = (),
    public_time_precision: int = 0,
    qa_id: str | None = None,
    source_condition_key: str | None = None,
) -> tuple[VisibilityRequirement, ...]:
    """Translate one compiled condition's planning knobs into requirements.

    ``planning`` is the ``planning`` mapping of a
    :class:`avengine.qa.generation_conditions.Condition`; ``evidence`` is that
    condition's ``evidence`` mapping.  ``anchor_visibility`` and
    ``anchor_line_of_sight`` alone stay what they are in the sampler - a
    frustum condition and a ray proxy - and only become a pixel-state
    requirement when the condition's own evidence asks for a pixel state.
    """

    if not isinstance(planning, Mapping):
        raise ConditionedVisibilityError("planning knobs must be a mapping")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    windows = tuple(
        (int(window[0]), int(window[1])) for window in observation_windows
    )
    common = {
        "subject": subject,
        "observation_windows": windows,
        "public_time_precision": int(public_time_precision),
        "qa_id": qa_id,
        "source_condition_key": source_condition_key,
    }
    requirements: list[VisibilityRequirement] = []

    transition = planning.get("visibility_transition")
    if transition is not None:
        if transition != "out_of_view_to_visible":
            raise ConditionedVisibilityError(
                f"unsupported visibility_transition {transition!r}; this module "
                "implements out_of_view_to_visible"
            )
        side = planning.get("entry_side")
        requirements.append(
            VisibilityRequirement(
                kind="out_of_view_to_visible",
                side=None if side is None else _text(side, owner="entry_side"),
                require_publishable_window=True,
                **common,
            )
        )

    occlusion = planning.get("pixel_occlusion_transition", planning.get("registered_occluder_transition"))
    if occlusion is not None:
        try:
            kind, state = _TRANSITION_TO_REQUIREMENT[str(occlusion)]
        except KeyError as error:
            raise ConditionedVisibilityError(
                f"unsupported pixel_occlusion_transition {occlusion!r}; this "
                f"module implements {sorted(_TRANSITION_TO_REQUIREMENT)}"
            ) from error
        requirements.append(
            VisibilityRequirement(
                kind=kind,
                state=state,
                require_publishable_window=kind == "registered_occluder_visible",
                require_complete_coverage=kind == "fully_occluded_without_return",
                **common,
            )
        )

    accepted_states = evidence.get("visibility_state_in")
    if (
        accepted_states is not None
        and not isinstance(accepted_states, (str, bytes))
        and isinstance(accepted_states, Sequence)
        and not any(item.kind == "visibility_state" for item in requirements)
    ):
        named = [str(value) for value in accepted_states]
        unknown = [value for value in named if value not in VISIBILITY_STATES]
        if not named or unknown:
            raise ConditionedVisibilityError(
                "condition evidence names unknown visibility states "
                f"{unknown or named!r}"
            )
        requirements.append(
            VisibilityRequirement(
                kind="visibility_state", allowed_states=tuple(sorted(set(named))),
                **common,
            )
        )

    wanted_state = evidence.get("visibility_state") or evidence.get(
        "final_visibility_state"
    )
    if wanted_state is not None and not any(
        item.kind == "visibility_state" for item in requirements
    ):
        if wanted_state not in VISIBILITY_STATES:
            raise ConditionedVisibilityError(
                f"condition evidence names an unknown visibility state {wanted_state!r}"
            )
        requirements.append(
            VisibilityRequirement(
                kind="visibility_state", state=str(wanted_state), **common
            )
        )

    previous = evidence.get("previous_state")
    current = evidence.get("current_state")
    if previous == "visible_occluded" and current == "visible_clear":
        requirements.append(
            VisibilityRequirement(
                kind="visible_occluded_to_visible_clear", **common
            )
        )

    if planning.get("require_complete_visibility_coverage") and requirements:
        requirements = [
            replace(item, require_complete_coverage=True) for item in requirements
        ]
    return tuple(requirements)


def requirements_from_conditions(
    conditions: Sequence[Mapping[str, Any]],
    *,
    observation_windows_by_subject: Mapping[str, Sequence[Sequence[int]]] | None = None,
    public_time_precision: int = 0,
    qa_id: str | None = None,
) -> tuple[VisibilityRequirement, ...]:
    """Collect requirements from a compiled condition list.

    Accepts the dicts ``generation_conditions.CompiledConditions.to_dict()``
    produces, so a caller hands the compiled result straight over without
    unpacking dataclasses.  Conditions of other kinds are ignored, and a
    condition whose subject is ``None`` cannot carry a per-instance pixel
    requirement and is skipped with no side effect.
    """

    if isinstance(conditions, Mapping):
        conditions = conditions.get("conditions") or ()
    windows_by_subject = dict(observation_windows_by_subject or {})
    collected: list[VisibilityRequirement] = []
    seen: set[tuple[Any, ...]] = set()
    for condition in conditions or ():
        if not isinstance(condition, Mapping):
            continue
        subject = condition.get("subject")
        if not isinstance(subject, str) or not subject:
            continue
        planning = condition.get("planning")
        if not isinstance(planning, Mapping):
            continue
        for item in requirements_from_planning(
            planning,
            subject=subject,
            evidence=condition.get("evidence"),
            observation_windows=windows_by_subject.get(subject, ()),
            public_time_precision=public_time_precision,
            qa_id=qa_id or condition.get("qa_id"),
            source_condition_key=condition.get("key"),
        ):
            identity = (
                item.kind,
                item.subject,
                item.state,
                item.allowed_states,
                item.side,
                item.require_complete_coverage,
                # Two statements about the same state in two different windows
                # are two statements. Leaving the window out of the identity
                # silently dropped the second one, which is exactly the shape
                # a QA-25 AV pair (visible sub-window, hidden sub-window of one
                # event) has.
                item.observation_windows,
            )
            if identity in seen:
                continue
            seen.add(identity)
            collected.append(item)
    return tuple(collected)


# ---------------------------------------------------------------------------
# Before the routes are drawn: what a requirement needs the scene to do.
# ---------------------------------------------------------------------------

# The dynamics vocabulary of ``_REQUIRED_DYNAMICS``, published so a route
# sampler can branch on it by name instead of on a requirement kind.
ROUTE_DYNAMICS = ("subject_frustum_crossing", "subject_or_occluder_motion")


@dataclass(frozen=True)
class RouteDynamicsRequirement:
    """What one instance's route has to do for a pixel requirement to be reachable.

    :func:`solve_visibility_candidates` already refuses a requirement whose
    change can never happen under a fixed camera, and says exactly why.  That
    verdict arrives *after* the routes are drawn, which is too late to act on:
    a still route plus an ``out_of_view_to_visible`` requirement has no camera
    pose left to find, so every retry spends its search and refuses.  This is
    the same statement published *before* the routes are drawn, so the caller
    that draws them can satisfy it.

    ``subject_must_move`` is the strict form: only this instance's own root
    motion can produce the change.  ``any_mover_satisfies`` is the loose form:
    a moving occluder does just as well, so the subject may stay put.
    """

    instance_id: str
    dynamic: str
    subject_must_move: bool
    any_mover_satisfies: bool
    requirement_kinds: tuple[str, ...]
    qa_ids: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        _text(self.instance_id, owner="instance_id")
        if self.dynamic not in ROUTE_DYNAMICS:
            raise ConditionedVisibilityError(
                f"unknown route dynamic {self.dynamic!r}; supported dynamics "
                f"are {list(ROUTE_DYNAMICS)}"
            )

    def as_report(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "dynamic": self.dynamic,
            "subject_must_move": self.subject_must_move,
            "any_mover_satisfies": self.any_mover_satisfies,
            "requirement_kinds": list(self.requirement_kinds),
            "qa_ids": list(self.qa_ids),
            "reason": self.reason,
        }


_ROUTE_DYNAMIC_REASON = {
    "subject_frustum_crossing": (
        "the camera is fixed, so only this instance's own root motion can carry "
        "it across the frustum boundary; a still route makes the requirement "
        "unreachable rather than merely hard to find"
    ),
    "subject_or_occluder_motion": (
        "the camera is fixed, so the occlusion state can only change if this "
        "instance or some other instance moves; a scene where nothing moves "
        "holds one occlusion state for the whole clip"
    ),
}


def route_dynamics_requirements(
    requirements: Sequence[VisibilityRequirement],
) -> tuple[RouteDynamicsRequirement, ...]:
    """What the routes must do before these requirements can be screened.

    One row per (subject, dynamic).  A requirement whose
    :attr:`VisibilityRequirement.required_dynamics` is empty is satisfied by a
    persistent state and contributes no row, which is why a bolted-down device
    permanently hidden behind a counter stays a legal QA-09 ``no``.
    """

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for requirement in requirements:
        if not isinstance(requirement, VisibilityRequirement):
            raise ConditionedVisibilityError(
                "route_dynamics_requirements needs VisibilityRequirement instances"
            )
        for dynamic in requirement.required_dynamics:
            row = grouped.setdefault(
                (requirement.subject, dynamic),
                {"kinds": [], "qa_ids": []},
            )
            if requirement.kind not in row["kinds"]:
                row["kinds"].append(requirement.kind)
            if requirement.qa_id and requirement.qa_id not in row["qa_ids"]:
                row["qa_ids"].append(str(requirement.qa_id))
    return tuple(
        RouteDynamicsRequirement(
            instance_id=subject,
            dynamic=dynamic,
            subject_must_move=dynamic == "subject_frustum_crossing",
            any_mover_satisfies=dynamic == "subject_or_occluder_motion",
            requirement_kinds=tuple(row["kinds"]),
            qa_ids=tuple(row["qa_ids"]),
            reason=_ROUTE_DYNAMIC_REASON[dynamic],
        )
        for (subject, dynamic), row in sorted(grouped.items())
    )


def route_motion_plan(
    requirements: Sequence[VisibilityRequirement],
    *,
    mobile_instance_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Who has to walk so these requirements are reachable at all.

    ``mobile_instance_ids`` are the instances whose routes the caller is able
    to make move — an immobile device belongs outside that list, and this
    function then reports the requirement as unsatisfiable by name rather than
    letting the caller draw a still route and discover it one refusal at a
    time.

    The returned ``must_move`` is a set of instance ids, which is exactly what
    a route sampler needs.  Nothing here invents a distance or a speed: how far
    the instance walks stays the route sampler's own policy, and whether the
    resulting track actually produces the state stays
    :func:`solve_visibility_candidates`'s screen.
    """

    mobile = [str(item) for item in mobile_instance_ids]
    mobile_set = set(mobile)
    rows = route_dynamics_requirements(requirements)
    must_move: list[str] = []
    unsatisfiable: list[dict[str, Any]] = []
    for row in rows:
        if row.subject_must_move:
            if row.instance_id in mobile_set:
                if row.instance_id not in must_move:
                    must_move.append(row.instance_id)
            else:
                unsatisfiable.append(
                    {
                        **row.as_report(),
                        "why": (
                            f"{row.instance_id} cannot move, and no other "
                            "instance can stand in for its own frustum crossing"
                        ),
                    }
                )
            continue
        # A moving occluder does just as well, so prefer the subject when it
        # can walk and fall back to any other mover.
        if row.instance_id in mobile_set:
            if row.instance_id not in must_move:
                must_move.append(row.instance_id)
            continue
        others = [item for item in mobile if item != row.instance_id]
        if others:
            if others[0] not in must_move:
                must_move.append(others[0])
            continue
        unsatisfiable.append(
            {
                **row.as_report(),
                "why": (
                    "neither the subject nor any other instance can move, so no "
                    "occlusion state can change inside this fixed-camera clip"
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "record": "route_motion_plan",
        "rows": [row.as_report() for row in rows],
        "must_move": tuple(must_move),
        "unsatisfiable": unsatisfiable,
        "mobile_instance_ids": tuple(mobile),
        "claim_boundary": (
            "a precondition on the routes, not a solution. Satisfying it makes "
            "the requirement reachable; whether any camera pose actually "
            "supports it is still decided by solve_visibility_candidates, and "
            "the pixel state is still decided by the renderer."
        ),
    }


# ---------------------------------------------------------------------------
# Tier one: projection.
# ---------------------------------------------------------------------------


def body_sample_points(
    camera_position_m: Sequence[float],
    root_m: Sequence[float],
    body: BodyProxy,
    policy: ScreenPolicy | None = None,
) -> np.ndarray:
    """World sample points on one body at one instant.

    The lateral offsets run perpendicular to the horizontal sight line so the
    silhouette does not depend on a heading the plan may not have fixed yet,
    and the heights run from the declared body height above the root.
    """

    resolved = screen_policy(policy)
    camera = _vector3(camera_position_m, owner="camera_position_m")
    root = _vector3(root_m, owner="root_m")
    delta = root - camera
    horizontal = math.hypot(float(delta[0]), float(delta[2]))
    if horizontal <= 1.0e-9:
        raise ConditionedVisibilityError(
            "the subject stands on the camera axis; no lateral span exists"
        )
    perpendicular = np.array(
        [-float(delta[2]) / horizontal, 0.0, float(delta[0]) / horizontal]
    )
    points = []
    for height_fraction in resolved.height_fractions:
        for lateral_fraction in resolved.lateral_fractions:
            offset = perpendicular * (lateral_fraction * body.width_m)
            points.append(
                [
                    float(root[0] + offset[0]),
                    float(root[1] + height_fraction * body.height_m),
                    float(root[2] + offset[2]),
                ]
            )
    return np.asarray(points, dtype=float)


def _project(
    camera: CameraPose, points_m: np.ndarray, policy: ScreenPolicy
) -> dict[str, np.ndarray]:
    """Project sample points onto the image plane of one fixed camera."""

    origin = np.asarray(camera.position_m, dtype=float)
    forward = np.asarray(camera.forward, dtype=float)
    right = np.asarray(camera.right, dtype=float)
    up = np.asarray(camera.up, dtype=float)
    height, width = camera.resolution_hw
    tangent_h = math.tan(math.radians(camera.horizontal_fov_deg) / 2.0)
    tangent_v = tangent_h / (float(width) / float(height))

    delta = points_m - origin
    depth = delta @ forward
    lateral = delta @ right
    vertical = delta @ up
    ahead = depth > policy.near_m
    safe_depth = np.where(ahead, depth, np.nan)
    column = camera.center_column_px + (lateral / safe_depth) / tangent_h * (
        float(width) / 2.0
    )
    row = (float(height) - 1.0) / 2.0 - (vertical / safe_depth) / tangent_v * (
        float(height) / 2.0
    )
    inside = (
        ahead
        & (np.abs(lateral) <= depth * tangent_h)
        & (np.abs(vertical) <= depth * tangent_v)
    )
    # A sample this close to a border cannot decide in-view at screen
    # resolution; the frame verdict downgrades instead of guessing.
    margin = policy.edge_margin_px
    near_border = np.zeros_like(inside)
    with np.errstate(invalid="ignore"):
        near_border = ahead & (
            (np.abs(column - -0.5) <= margin)
            | (np.abs(column - (float(width) - 0.5)) <= margin)
            | (np.abs(row - -0.5) <= margin)
            | (np.abs(row - (float(height) - 0.5)) <= margin)
        )
    return {
        "depth_m": depth,
        "column_px": column,
        "row_px": row,
        "in_frustum": inside,
        "near_border": near_border,
        "lateral_m": lateral,
    }



def _envelope_projection_summary(
    camera: CameraPose,
    projected: Mapping[str, np.ndarray],
    policy: ScreenPolicy,
) -> dict[str, Any]:
    """Summarise a full-body projection without reducing it to emitter samples."""

    height, width = camera.resolution_hw
    depth = projected["depth_m"]
    columns = projected["column_px"]
    rows = projected["row_px"]
    valid = (
        (depth > policy.near_m)
        & np.isfinite(columns)
        & np.isfinite(rows)
    )
    if not np.any(valid):
        return {
            "projected_bbox_px": None,
            "in_view": False,
            "outside_side": None,
            "valid_points": 0,
        }
    valid_columns = columns[valid]
    valid_rows = rows[valid]
    minimum_column = float(np.min(valid_columns))
    maximum_column = float(np.max(valid_columns))
    minimum_row = float(np.min(valid_rows))
    maximum_row = float(np.max(valid_rows))
    image_min_column = -0.5
    image_max_column = float(width) - 0.5
    image_min_row = -0.5
    image_max_row = float(height) - 0.5
    bbox_intersects = (
        maximum_column >= image_min_column
        and minimum_column <= image_max_column
        and maximum_row >= image_min_row
        and minimum_row <= image_max_row
    )
    outside_side = None
    if maximum_column < image_min_column:
        outside_side = "left"
    elif minimum_column > image_max_column:
        outside_side = "right"
    return {
        "projected_bbox_px": [
            minimum_column,
            minimum_row,
            maximum_column,
            maximum_row,
        ],
        # A projected envelope that overlaps the image is conservatively
        # considered in view even when no sampled vertex lands strictly inside:
        # a triangle edge can cross a frustum boundary between vertices.
        "in_view": bool(np.any(projected["in_frustum"]) or bbox_intersects),
        "outside_side": outside_side,
        "valid_points": int(np.count_nonzero(valid)),
    }


# ---------------------------------------------------------------------------
# Tier two: rays, actor cylinders and static boxes.
# ---------------------------------------------------------------------------


def _segment_hits_box(
    origin: np.ndarray,
    target: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> bool:
    """Slab test for a finite segment against an axis-aligned box."""

    direction = target - origin
    low, high = 0.0, 1.0
    for axis in range(3):
        component = float(direction[axis])
        start = float(origin[axis])
        if abs(component) < 1.0e-12:
            if start < float(minimum[axis]) or start > float(maximum[axis]):
                return False
            continue
        first = (float(minimum[axis]) - start) / component
        second = (float(maximum[axis]) - start) / component
        if first > second:
            first, second = second, first
        low = max(low, first)
        high = min(high, second)
        if low > high:
            return False
    return True


def _cylinder_blocks(
    origin: np.ndarray,
    points_m: np.ndarray,
    other_position_m: np.ndarray,
    other_body: BodyProxy,
) -> np.ndarray:
    """Whether another instance's vertical cylinder cuts each sight line.

    The radius is half the declared body width; the same treatment
    ``tools/qa/visibility_prediction.actor_cylinder_blocks`` applies, kept here
    in the shared metre frame instead of UE centimetres.
    """

    radius = 0.5 * float(other_body.width_m)
    top = float(other_position_m[1]) + float(other_body.height_m)
    base = float(other_position_m[1])
    segment = points_m - origin
    horizontal = segment[:, [0, 2]]
    length = np.linalg.norm(horizontal, axis=1)
    to_other = other_position_m[[0, 2]] - origin[[0, 2]]
    along = (horizontal @ to_other) / np.maximum(length**2, 1.0e-12)
    within = (along > 0.0) & (along < 1.0)
    closest = origin[[0, 2]] + horizontal * along[:, None]
    lateral = np.linalg.norm(closest - other_position_m[[0, 2]], axis=1)
    height_at = float(origin[1]) + segment[:, 1] * along
    return within & (lateral <= radius) & (height_at >= base) & (height_at <= top)


_UNRESOLVED_STATIC = "unresolved_static_geometry"


def _blocked_by_scene(
    mesh: Any,
    origin: np.ndarray,
    points_m: np.ndarray,
    policy: ScreenPolicy,
    cache: dict[tuple[Any, ...], str],
) -> tuple[np.ndarray, np.ndarray]:
    """Cast one ray per sample and report blocked plus measured mask."""

    blocked = np.zeros(len(points_m), dtype=bool)
    measured = np.zeros(len(points_m), dtype=bool)
    if mesh is None:
        return blocked, measured
    for index, point in enumerate(points_m):
        direction = point - origin
        distance = float(np.linalg.norm(direction))
        if distance <= policy.ray_backoff_m:
            measured[index] = True
            continue
        endpoint = point - direction / distance * policy.ray_backoff_m
        key = (
            round(float(origin[0]), 4),
            round(float(origin[1]), 4),
            round(float(origin[2]), 4),
            round(float(endpoint[0]), 4),
            round(float(endpoint[1]), 4),
            round(float(endpoint[2]), 4),
        )
        verdict = cache.get(key)
        if verdict is None:
            verdict = line_of_sight(mesh, origin, endpoint)
            cache[key] = verdict
        if verdict == "unmeasured":
            continue
        measured[index] = True
        blocked[index] = verdict == "blocked"
    return blocked, measured


def narrow_mesh_to_segments(mesh: Any, endpoints: np.ndarray) -> Any:
    """Keep only the triangles a set of camera-to-sample segments can hit.

    Every segment lies inside the union bounding box of its endpoints, so the
    whole-room bounding-box scan ``line_of_sight`` performs per ray runs once
    per screened series instead of once per sample.  Same reduction
    ``binding_group_motion._visible_path`` already applies to its emitter
    segments; dropping a triangle that cannot intersect any segment does not
    change a single ray verdict.
    """

    if not isinstance(mesh, MeshHandle):
        return mesh
    points = np.asarray(endpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ConditionedVisibilityError("mesh narrowing needs [point, 3] endpoints")
    low = points.min(axis=0) - 1.0e-4
    high = points.max(axis=0) + 1.0e-4
    relevant = np.all(mesh.maximum >= low, axis=1) & np.all(mesh.minimum <= high, axis=1)
    return MeshHandle(mesh.vertices, mesh.triangles[relevant], mesh.source)


def _frame_screen(
    *,
    camera: CameraPose,
    subject: ActorTrack,
    others: Sequence[ActorTrack],
    static_occluders: Sequence[StaticOccluder],
    mesh: Any,
    frame: int,
    policy: ScreenPolicy,
    ray_cache: dict[tuple[Any, ...], str],
    cast_rays: bool,
    geometry_authority: str,
) -> dict[str, Any]:
    """Screen one instance at one frame into a four-state prediction."""

    origin = np.asarray(camera.position_m, dtype=float)
    root = subject.positions_m[frame]
    body_envelope = subject.body_envelope
    points = (
        body_envelope.vertices_m[frame]
        if body_envelope is not None
        else body_sample_points(origin, root, subject.body, policy)
    )
    projected = _project(camera, points, policy)
    in_frustum = projected["in_frustum"]
    in_count = int(np.count_nonzero(in_frustum))
    marginal_border = int(np.count_nonzero(projected["near_border"]))
    envelope_summary = (
        _envelope_projection_summary(camera, projected, policy)
        if body_envelope is not None
        else None
    )
    in_view = (
        bool(envelope_summary["in_view"])
        if envelope_summary is not None
        else bool(in_count > 0)
    )
    record: dict[str, Any] = {
        "frame_index": int(frame),
        "samples": int(len(points)),
        "in_frustum_samples": in_count,
        "border_marginal_samples": marginal_border,
        "tier": "frustum",
        # The frustum tier alone settles in-view versus out-of-view; only the
        # occlusion split below it needs geometry.
        "in_view": in_view,
        "body_geometry": (
            "full_body_envelope"
            if body_envelope is not None
            else "emitter_anchor_proxy"
        ),
        "body_geometry_source": (
            None if body_envelope is None else body_envelope.source_ref
        ),
        "projected_body_bbox_px": (
            None
            if envelope_summary is None
            else envelope_summary["projected_bbox_px"]
        ),
        "outside_side": (
            None if envelope_summary is None else envelope_summary["outside_side"]
        ),
        "occluder_ids": [],
        "occluder_attribution": {},
        "blocked_samples": None,
        "scene_measured_samples": None,
        "visible_samples": None,
        "unknown_samples": None,
        "predicted_column_px": None,
        "side": None,
        "side_beyond_dead_zone": None,
        "state": None,
        "decisive": False,
        "refutes": [],
    }
    if in_count:
        columns = projected["column_px"][in_frustum]
        centroid = float(np.nanmean(columns))
        offset = centroid - camera.center_column_px
        record["predicted_column_px"] = centroid
        record["side"] = "right" if offset > 0.0 else "left"
        record["side_beyond_dead_zone"] = bool(abs(offset) > camera.dead_zone_px)
        record["predicted_column_offset_px"] = offset
    if not in_view:
        # The proxy's nine samples can under-cover a silhouette.  A supplied
        # full-body envelope additionally treats an image-overlapping projected
        # envelope as in view, so this branch means the complete screened
        # envelope is outside the image at this frame.
        record["state"] = "out_of_view"
        record["decisive"] = False
        record["refutes"] = []
        record["reason"] = (
            "the screened body geometry has no projected envelope overlap with "
            "the image; this is a planning prediction, not a pixel proof"
        )
        return record
    if not cast_rays:
        # A real point on the body projecting inside the frame does mean the
        # target-only footprint is not empty; that is the screen's one sound
        # direction and it needs no geometry.
        record["refutes"] = ["out_of_view"]
        record["reason"] = (
            "full body envelope establishes frustum occupancy; the occlusion "
            "split was not measured at this frame"
            if body_envelope is not None
            else "frustum tier only; the occlusion split was not measured at this frame"
        )
        return record

    record["tier"] = "frustum_and_ray"
    scene_blocked, measured = _blocked_by_scene(mesh, origin, points, policy, ray_cache)
    actor_blocked = np.zeros(len(points), dtype=bool)
    attribution: dict[str, int] = {}
    for other in others:
        if other.instance_id == subject.instance_id:
            continue
        hits = _cylinder_blocks(
            origin, points, other.positions_m[frame], other.body
        ) & in_frustum
        if np.any(hits):
            actor_blocked |= hits
            attribution[other.instance_id] = int(np.count_nonzero(hits))
    static_hits = np.zeros(len(points), dtype=bool)
    for occluder in static_occluders:
        minimum = np.asarray(occluder.minimum_m, dtype=float)
        maximum = np.asarray(occluder.maximum_m, dtype=float)
        hits = np.asarray(
            [
                bool(
                    in_frustum[index]
                    and _segment_hits_box(origin, points[index], minimum, maximum)
                )
                for index in range(len(points))
            ]
        )
        if np.any(hits):
            static_hits |= hits
            attribution[occluder.occluder_id] = int(np.count_nonzero(hits))

    # An occluding instance is measured analytically, so actor occlusion is
    # screened even when the room supplies no static mesh.  Scene occlusion is
    # only known where the ray was actually measured.
    blocked = (actor_blocked | (scene_blocked & measured)) & in_frustum
    unexplained = int(
        np.count_nonzero(scene_blocked & measured & in_frustum & ~static_hits & ~actor_blocked)
    )
    if unexplained:
        attribution[_UNRESOLVED_STATIC] = unexplained
    visible = in_frustum & ~blocked & measured
    unknown = in_frustum & ~blocked & ~measured
    blocked_count = int(np.count_nonzero(blocked))
    visible_count = int(np.count_nonzero(visible))
    unknown_count = int(np.count_nonzero(unknown))
    record["blocked_samples"] = blocked_count
    record["scene_measured_samples"] = int(np.count_nonzero(measured & in_frustum))
    record["visible_samples"] = visible_count
    record["unknown_samples"] = unknown_count
    record["occluder_attribution"] = dict(sorted(attribution.items()))
    record["occluder_ids"] = sorted(
        key for key in attribution if key != _UNRESOLVED_STATIC
    )

    record["geometry_authority"] = geometry_authority
    if blocked_count == in_count:
        # Decided without the mesh when instances do all the blocking, and
        # still only a prediction: a sparse sample set cannot prove an empty
        # footprint.
        record["state"] = "fully_occluded"
        record["decisive"] = False
        record["refutes"] = ["out_of_view"]
        record["reason"] = (
            "every in-frustum body sample is blocked; only the native pixel pass "
            "can prove that no pixel of the silhouette survives"
        )
        return record
    if unknown_count:
        record["state"] = None
        record["refutes"] = ["out_of_view"]
        record["reason"] = (
            f"{unknown_count} in-frustum samples have no measured scene ray, so the "
            "occlusion split is unknown at this frame"
        )
        return record
    if blocked_count:
        record["state"] = "visible_occluded"
        record["decisive"] = visible_count >= 2 and blocked_count >= 2
    else:
        record["state"] = "visible_clear"
        # One hidden pixel is enough to make the true state visible_occluded,
        # so a fully clear screen is never decisive about being clear.
        record["decisive"] = False
    # A visible sample rules out an empty footprint whatever the geometry is.
    # Ruling out a full occlusion needs the geometry that actually renders, so
    # a proxy mesh may only rank.
    record["refutes"] = (
        ["fully_occluded", "out_of_view"]
        if geometry_authority == "visual_mesh"
        else ["out_of_view"]
    )
    return record


def screen_visibility_series(
    *,
    camera: CameraPose | Mapping[str, Any],
    tracks: Sequence[ActorTrack],
    subject: str,
    mesh: Any = None,
    static_occluders: Sequence[StaticOccluder] = (),
    policy: ScreenPolicy | str | None = None,
    frames: Sequence[int] | None = None,
    cast_rays: bool = True,
    ray_cache: dict[tuple[Any, ...], str] | None = None,
    geometry_authority: str = "unknown",
) -> dict[str, Any]:
    """Screen one subject over the clip under one fixed camera.

    The result is a per-frame four-state *prediction* with an explicit
    ``decisive`` flag and the set of pixel states each frame refutes.  It is
    never an answer: see the module docstring and :data:`SCREEN_VERDICTS`.
    """

    resolved_camera = (
        camera if isinstance(camera, CameraPose) else CameraPose.from_mapping(camera)
    )
    resolved_policy = screen_policy(policy)
    if geometry_authority not in GEOMETRY_AUTHORITIES:
        raise ConditionedVisibilityError(
            f"geometry_authority must be one of {list(GEOMETRY_AUTHORITIES)}"
        )
    by_id = {track.instance_id: track for track in tracks}
    if len(by_id) != len(tracks):
        raise ConditionedVisibilityError("actor tracks repeat an instance id")
    if subject not in by_id:
        raise ConditionedVisibilityError(
            f"subject {subject!r} has no track; tracks are {sorted(by_id)}"
        )
    frame_counts = {track.frame_count for track in tracks}
    if len(frame_counts) != 1:
        raise ConditionedVisibilityError(
            "every actor track must share one frame clock; observed counts "
            f"{sorted(frame_counts)}"
        )
    frame_count = frame_counts.pop()
    selected = (
        list(range(frame_count))
        if frames is None
        else [_index(frame, owner="frame") for frame in frames]
    )
    if any(frame >= frame_count for frame in selected):
        raise ConditionedVisibilityError("a requested frame is outside the clock")
    cache = {} if ray_cache is None else ray_cache
    target = by_id[subject]
    others = [track for track in tracks if track.instance_id != subject]
    # Full envelopes are used for the frustum crossing only.  Sending every
    # skinned vertex through the room ray screen would turn a planning
    # envelope into an accidental renderer-sized workload; occlusion remains
    # undetermined when only this envelope is supplied.
    effective_cast_rays = cast_rays and target.body_envelope is None
    narrowed = mesh
    narrowing: dict[str, Any] | None = None
    if effective_cast_rays and isinstance(mesh, MeshHandle):
        endpoints = np.concatenate(
            [np.asarray(resolved_camera.position_m, dtype=float)[None, :]]
            + [
                body_sample_points(
                    resolved_camera.position_m,
                    target.positions_m[frame],
                    target.body,
                    resolved_policy,
                )
                for frame in selected
            ]
        )
        narrowed = narrow_mesh_to_segments(mesh, endpoints)
        narrowing = {
            "triangles_declared": int(mesh.triangles.shape[0]),
            "triangles_retained": int(narrowed.triangles.shape[0]),
            "policy": "union_bounding_box_of_camera_and_every_body_sample",
        }
    records = [
        _frame_screen(
            camera=resolved_camera,
            subject=target,
            others=others,
            static_occluders=tuple(static_occluders),
            mesh=narrowed,
            frame=frame,
            policy=resolved_policy,
            ray_cache=cache,
            cast_rays=effective_cast_rays,
            geometry_authority=geometry_authority,
        )
        for frame in selected
    ]
    counts = {state: 0 for state in PIXEL_VISIBILITY_STATES}
    counts["unmeasured"] = 0
    in_view_frames = 0
    for record in records:
        state = record.get("state")
        counts[state if state in counts else "unmeasured"] += 1
        in_view_frames += int(bool(record.get("in_view")))
    return {
        "schema": SCHEMA,
        "record": "screen_series",
        "tier": (
            "frustum_and_ray"
            if (effective_cast_rays and mesh is not None)
            else "frustum"
        ),
        "authority": "screening_prediction_never_an_answer_v1",
        "confirmation_authority": CONFIRMATION_AUTHORITY,
        "geometry_authority": geometry_authority,
        "sound_direction": (
            "an in-frustum body sample rules out out_of_view; nothing else here "
            "refutes, and an occlusion state is only refutable against the "
            "geometry that actually renders"
        ),
        "camera": resolved_camera.as_report(),
        "subject": subject,
        "subject_body": target.body.as_report(),
        "subject_body_envelope": (
            None
            if target.body_envelope is None
            else target.body_envelope.as_report()
        ),
        "body_geometry": (
            "full_body_envelope"
            if target.body_envelope is not None
            else "emitter_anchor_proxy"
        ),
        "subject_can_move": target.can_move,
        "entity_class": target.entity_class,
        "policy": resolved_policy.as_report(),
        "frame_count": frame_count,
        "frames_screened": selected,
        "static_occluder_count": len(tuple(static_occluders)),
        "static_geometry_source": None if mesh is None else _mesh_source(mesh),
        "static_geometry_narrowing": narrowing,
        "predicted_state_counts": counts,
        "in_view_frames": in_view_frames,
        "out_of_view_frames": len(records) - in_view_frames,
        "frames": records,
    }


def _mesh_source(mesh: Any) -> Any:
    source = getattr(mesh, "source", None)
    if source is None and isinstance(mesh, Mapping):
        source = {
            key: str(mesh.get(key))
            for key in ("vertices", "triangles")
            if mesh.get(key) is not None
        }
    return source


# ---------------------------------------------------------------------------
# Runs and windows over a screened or witnessed state series.
# ---------------------------------------------------------------------------


def _runs(states: Sequence[tuple[int, Any]]) -> list[dict[str, Any]]:
    """Maximal runs of one value over consecutive frame indices."""

    runs: list[dict[str, Any]] = []
    for frame, value in states:
        if (
            runs
            and runs[-1]["value"] == value
            and runs[-1]["end"] == frame
        ):
            runs[-1]["end"] = frame + 1
            continue
        runs.append({"value": value, "start": int(frame), "end": int(frame) + 1})
    return runs


def _facts_view(
    *,
    states_by_instance: Mapping[str, Mapping[int, Mapping[str, Any]]],
    frame_count: int,
    frame_rate_hz: float,
    precision: int,
    occluder_evidence: Mapping[str, Any] | None = None,
    visibility_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The smallest facts shape the catalog's own visibility predicates read.

    Building this view is what lets acceptance call
    ``unified_catalog._entry_transition_window``,
    ``_occlusion_interval_window``, ``_visibility_is_complete`` and
    ``_display_time_bounds`` instead of restating them.  Measuring something
    adjacent to what the judge measures produces reports that look reasonable
    and prove nothing.
    """

    view: dict[str, Any] = {
        "visibility": {
            instance: dict(frames) for instance, frames in states_by_instance.items()
        },
        "time": {
            "frame_count": int(frame_count),
            "frame_rate_hz": float(frame_rate_hz),
        },
        "sampling": {"time_display_precision": int(precision)},
    }
    if occluder_evidence is not None:
        view["occluder_evidence"] = occluder_evidence
    if visibility_meta is not None:
        # Same field the catalog's own entry-side check reads.
        view["visibility_meta"] = dict(visibility_meta)
    return view


def subject_state_windows(
    series: Mapping[str, Any],
    windows: Mapping[str, Sequence[int]] | Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Which screened states hold inside each named sub-window of one clip.

    ``series`` is one :func:`screen_visibility_series` result and ``windows``
    are half-open ``[start, end)`` frame ranges, either named (``{"visible":
    [30, 45], "query": [60, 61]}``) or positional.  The result is per window:
    the frames it covers, the count of each screened state inside it, the
    states no frame inside it can hold, and whether every frame inside it was
    measured at all.

    This exists so a caller that needs two different states inside one audible
    event — a visible anchor frame and a hidden query frame, say — can read the
    windows back from one screen instead of screening the clip twice and
    hoping the two runs agree.  Everything it returns is a screened
    prediction: the states are the same non-authoritative four-state
    prediction the rest of this tier produces, so a window that "holds"
    ``fully_occluded`` here is worth rendering, never proven.
    """

    if isinstance(windows, Mapping):
        items = [(str(key), value) for key, value in windows.items()]
    else:
        items = [(str(index), value) for index, value in enumerate(windows)]
    by_frame = {
        int(record["frame_index"]): record for record in series.get("frames", ())
    }
    rows: dict[str, Any] = {}
    for name, window in items:
        if (
            not isinstance(window, Sequence)
            or isinstance(window, (str, bytes))
            or len(window) != 2
        ):
            raise ConditionedVisibilityError(
                f"window {name!r} must be a half-open [start, end) frame pair"
            )
        start = _index(window[0], owner=f"window {name} start")
        end = _index(window[1], owner=f"window {name} end", minimum=1)
        if start >= end:
            raise ConditionedVisibilityError(
                f"window {name!r} must be a half-open [start, end) frame pair"
            )
        frames = [frame for frame in range(start, end) if frame in by_frame]
        counts = {state: 0 for state in VISIBILITY_STATES}
        unmeasured = 0
        refuted: set[str] = set(VISIBILITY_STATES)
        for frame in frames:
            record = by_frame[frame]
            state = record.get("state")
            if state is None:
                unmeasured += 1
                refuted = set()
                continue
            counts[str(state)] += 1
            refuted &= set(record.get("refutes") or ())
        rows[name] = {
            "window": [start, end],
            "frames_screened": len(frames),
            "frames_requested": end - start,
            "frames_unmeasured": unmeasured,
            "state_counts": counts,
            "states_refuted_throughout": sorted(refuted),
            "complete": len(frames) == end - start and unmeasured == 0,
        }
    return {
        "schema": SCHEMA,
        "record": "subject_state_windows",
        "subject": series.get("subject"),
        "tier": series.get("tier"),
        "confirmation_authority": CONFIRMATION_AUTHORITY,
        "windows": rows,
        "claim_boundary": (
            "screened per-window state counts. A state counted here is worth "
            "rendering, not observed; the pixel truth is still the authority."
        ),
    }


def publishable_window(
    facts_view: Mapping[str, Any], window: Sequence[int]
) -> dict[str, Any]:
    """Whether one frame window survives the judge's inward time rounding."""

    try:
        bounds = catalog._display_time_bounds(facts_view, [int(window[0]), int(window[1])])
    except catalog._Deferred as deferred:  # pragma: no cover - defensive
        return {
            "frames": [int(window[0]), int(window[1])],
            "publishable": False,
            "reason": getattr(deferred, "reason", "deferred"),
        }
    return {
        "frames": [int(window[0]), int(window[1])],
        "publishable": bounds is not None,
        "public_seconds": None if bounds is None else [bounds[0], bounds[1]],
        "reason": None
        if bounds is not None
        else "the interval is shorter than one unit of the public time precision",
    }


def first_publishable_extent(
    facts_view: Mapping[str, Any],
    start_frame: int,
    *,
    frame_count: int,
) -> dict[str, Any]:
    """The shortest window from ``start_frame`` that a question may publish.

    QA-07 and QA-10 print a public interval, so a two-frame transition is
    legal for the judge's state test and still unusable as a question.  At
    15 fps and whole-second precision the first publishable extent is around
    16 frames, which is why an entry crossing has to leave that much clip
    behind it.
    """

    for end in range(int(start_frame) + 1, int(frame_count) + 1):
        result = publishable_window(facts_view, [int(start_frame), end])
        if result["publishable"]:
            return {
                "start_frame": int(start_frame),
                "end_frame": end,
                "frames_needed": end - int(start_frame),
                "public_seconds": result["public_seconds"],
                "publishable": True,
            }
    return {
        "start_frame": int(start_frame),
        "end_frame": None,
        "frames_needed": None,
        "public_seconds": None,
        "publishable": False,
        "reason": "no window starting at this frame fits inside the clip at the "
        "configured public time precision",
    }


# ---------------------------------------------------------------------------
# Candidate solving.
# ---------------------------------------------------------------------------


def _dynamics_state(
    requirement: VisibilityRequirement,
    *,
    subject: ActorTrack,
    others: Sequence[ActorTrack],
    static_occluders: Sequence[StaticOccluder],
) -> dict[str, Any]:
    """Whether a fixed camera and these tracks could ever move this state.

    A device that cannot walk only blocks the requirement that needs the
    subject's own frustum crossing.  Every occlusion requirement stays open,
    because a walking occluder or a permanent piece of scene supplies the
    change; refusing the whole group because the target is bolted down is the
    accounting mistake this function exists to prevent.
    """

    for dynamic in requirement.required_dynamics:
        if dynamic == "subject_frustum_crossing" and not subject.can_move:
            return {
                "state": STATE_NOT_APPLICABLE,
                "reason": "a fixed camera and a subject whose root never moves "
                f"({subject.entity_class}) cannot produce an out_of_view to "
                "visible crossing; occlusion requirements are unaffected",
            }
        if dynamic == "subject_or_occluder_motion":
            movers = [item.instance_id for item in others if item.can_move]
            if not subject.can_move and not movers:
                return {
                    "state": STATE_NOT_APPLICABLE,
                    "reason": "neither the subject nor any other instance moves, "
                    "so no occlusion state can change inside this fixed-camera clip",
                }
    if requirement.kind == "registered_occluder_visible" and not others and not static_occluders:
        return {
            "state": STATE_EVIDENCE_MISSING,
            "reason": "no other instance and no declared static object could be "
            "named as the occluder",
        }
    return {"state": STATE_AVAILABLE, "reason": None}


# Requirement kinds whose refutation rests on the occlusion split rather than
# on the frustum, so a proxy mesh may only leave them undetermined.
_OCCLUSION_REFUTATION_KINDS = frozenset(
    {
        "fully_occluded_then_visible",
        "fully_occluded_without_return",
        "visible_occluded_to_visible_clear",
        "registered_occluder_visible",
    }
)


def _entry_depth_px(record: Mapping[str, Any], side: Any, image_width: float) -> float:
    """How far inside the image the screened body reaches at one frame.

    With a full-body envelope this is the projected width lying inside the
    image; with the emitter proxy it is the centroid's distance from the entry
    edge.  A body that only grazes the edge scores a few pixels either way.
    """

    bbox = record.get("projected_body_bbox_px")
    if (
        isinstance(bbox, Sequence)
        and len(bbox) == 4
        and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in bbox)
    ):
        # How far the body's inner edge has come in from the edge it entered
        # through. A person fully inside near the left edge scores their full
        # extent; a sliver at that edge scores a few pixels.
        if side == "left":
            return max(0.0, min(float(bbox[2]), image_width - 0.5) + 0.5)
        return max(0.0, (image_width - 0.5) - max(float(bbox[0]), -0.5))
    column = record.get("predicted_column_px")
    if column is None or image_width <= 0.0:
        return 0.0
    if side == "left":
        return float(column) + 0.5
    return (image_width - 0.5) - float(column)


def _evaluate_requirement(
    requirement: VisibilityRequirement,
    *,
    series: Mapping[str, Any],
    facts_view: Mapping[str, Any],
    frame_count: int,
    geometry_authority: str = "unknown",
) -> dict[str, Any]:
    """Screen one requirement against one screened state series."""

    records = list(series["frames"])
    by_frame = {int(record["frame_index"]): record for record in records}
    states = [
        (int(record["frame_index"]), record.get("state")) for record in records
    ]
    runs = _runs(states)
    # In-view versus out-of-view is a frustum fact and never waits for a ray,
    # so the crossing requirement reads its own runs.
    view_runs = _runs(
        [(int(record["frame_index"]), bool(record.get("in_view"))) for record in records]
    )
    result: dict[str, Any] = {
        "kind": requirement.kind,
        "subject": requirement.subject,
        "verdict": "undetermined",
        "reason": None,
        "tier": series["tier"],
        "confirmation_authority": CONFIRMATION_AUTHORITY,
        "screen_runs": runs,
        "margin": 0.0,
    }
    unmeasured = sum(1 for _frame, state in states if state is None)
    if unmeasured == len(states):
        result["reason"] = "no frame of this candidate could be screened"
        return result

    if requirement.kind == "out_of_view_to_visible":
        camera_report = series.get("camera") if isinstance(series.get("camera"), Mapping) else {}
        resolution = camera_report.get("resolution_hw") or (0, 0)
        image_width = float(resolution[1]) if len(resolution) == 2 else 0.0
        policy_report = series.get("policy") if isinstance(series.get("policy"), Mapping) else {}
        depth_px = float(policy_report.get("entry_depth_fraction", 0.0) or 0.0) * image_width
        crossings: list[dict[str, Any]] = []
        for previous, current in zip(view_runs, view_runs[1:]):
            if previous["value"] is not False or current["value"] is not True:
                continue
            entry = int(current["start"])
            side = by_frame[entry].get("side")
            beyond = bool(by_frame[entry].get("side_beyond_dead_zone"))
            sustain = 0
            deep_sustain = 0
            deep_streak = 0
            for frame in range(entry, int(current["end"])):
                record = by_frame[frame]
                if record.get("side") != side or not record.get("side_beyond_dead_zone"):
                    break
                sustain += 1
                if _entry_depth_px(record, side, image_width) >= depth_px:
                    deep_streak += 1
                    deep_sustain = max(deep_sustain, deep_streak)
                else:
                    deep_streak = 0
            extent = first_publishable_extent(
                facts_view, entry, frame_count=frame_count
            )
            crossings.append(
                {
                    "entry_frame": entry,
                    "side": side,
                    "side_beyond_dead_zone": beyond,
                    "initial_outside_side": by_frame[previous["start"]].get(
                        "outside_side"
                    ),
                    "same_side_sustain_frames": sustain,
                    "deep_sustain_frames": deep_sustain,
                    "entry_depth_px_required": depth_px,
                    "publishable_extent": extent,
                    "out_of_view_run": [previous["start"], previous["end"]],
                    "visible_run": [current["start"], current["end"]],
                }
            )
        result["crossings"] = crossings
        if not crossings:
            result["verdict"] = "refuted"
            result["reason"] = (
                "the subject never leaves and re-enters the frustum in this clip; "
                "an exit at the end of the clip is not an entry"
            )
            return result
        wanted = [
            item
            for item in crossings
            if requirement.side is None
            or (
                item["side"] == requirement.side
                and (
                    series.get("body_geometry") != "full_body_envelope"
                    or item["initial_outside_side"] == requirement.side
                )
            )
        ]
        if not wanted:
            result["verdict"] = "refuted"
            result["reason"] = (
                f"every screened crossing enters from the other side; this branch "
                f"needs {requirement.side}"
            )
            return result
        usable = [
            item
            for item in wanted
            if item["side_beyond_dead_zone"]
            and item["same_side_sustain_frames"] >= requirement.min_sustain_frames
            and item["deep_sustain_frames"] >= requirement.min_sustain_frames
        ]
        publishable = [
            item
            for item in usable
            if not requirement.require_publishable_window
            or (
                item["publishable_extent"]["publishable"]
                and item["same_side_sustain_frames"]
                >= item["publishable_extent"]["frames_needed"]
            )
        ]
        if not usable:
            result["verdict"] = "refuted"
            result["reason"] = (
                "no crossing holds one entry side beyond the judge's dead zone and at "
                f"least {depth_px:.0f} px inside the entry edge for "
                f"{requirement.min_sustain_frames} consecutive frames"
            )
            return result
        if not publishable:
            needed = [
                item["publishable_extent"]["frames_needed"]
                for item in usable
                if item["publishable_extent"]["frames_needed"] is not None
            ]
            shortfall = (
                f"it needs {min(needed)} frames of same-side visibility after the "
                "crossing"
                if needed
                else "no window starting at the crossing fits inside the clip, so it "
                "needs an earlier crossing"
            )
            result["verdict"] = "refuted"
            result["reason"] = (
                "the entry interval cannot be printed at the configured public "
                f"time precision; {shortfall}"
            )
            return result
        best = max(publishable, key=lambda item: item["same_side_sustain_frames"])
        result["verdict"] = "consistent"
        result["selected"] = best
        result["margin"] = float(best["same_side_sustain_frames"])
        return result

    if requirement.kind == "visibility_state":
        wanted = set(requirement.states)
        matching = [frame for frame, state in states if state in wanted]
        refuting = [
            frame
            for frame, _state in states
            if wanted <= set(by_frame[frame].get("refutes") or ())
        ]
        windows = requirement.observation_windows
        if windows:
            inside = [
                frame
                for frame in matching
                if any(start <= frame < end for start, end in windows)
            ]
        else:
            inside = matching
        result["frames_in_state"] = matching
        result["frames_in_state_inside_observation"] = inside
        result["frames_refuting_state"] = refuting
        if len(inside) >= requirement.min_state_frames:
            result["verdict"] = "consistent"
            result["margin"] = float(len(inside))
            return result
        if refuting and len(refuting) == len(states):
            result["verdict"] = "refuted"
            result["reason"] = (
                f"every screened frame refutes {wanted}"
            )
            return result
        result["reason"] = (
            f"fewer than {requirement.min_state_frames} screened frames reach "
            f"{wanted} inside the observation windows"
        )
        return result

    if requirement.kind in {
        "fully_occluded_then_visible",
        "fully_occluded_without_return",
    }:
        hidden = [item for item in runs if item["value"] == "fully_occluded"]
        result["fully_occluded_runs"] = hidden
        if not hidden:
            visible_everywhere = all(
                state in VISIBLE_STATES for _frame, state in states if state is not None
            )
            result["verdict"] = "refuted" if visible_everywhere else "undetermined"
            result["reason"] = (
                "no screened frame hides every body sample, so no full occlusion "
                "is reachable from these tracks under this camera"
            )
            return result
        long_enough = [
            item
            for item in hidden
            if item["end"] - item["start"] >= requirement.min_state_frames
        ]
        if not long_enough:
            result["reason"] = (
                "the screened full-occlusion runs are shorter than "
                f"{requirement.min_state_frames} frames"
            )
            return result
        if requirement.kind == "fully_occluded_then_visible":
            returns = [
                item
                for item in long_enough
                if any(
                    other["value"] in VISIBLE_STATES
                    and other["start"] >= item["end"]
                    and other["end"] - other["start"] >= requirement.min_state_frames
                    for other in runs
                )
            ]
            result["returning_runs"] = returns
            if not returns:
                result["verdict"] = "refuted"
                result["reason"] = (
                    "the subject never becomes visible again after the screened "
                    "full occlusion; a terminal hide answers the negative branch, "
                    "not this one"
                )
                return result
            result["verdict"] = "consistent"
            result["margin"] = float(
                max(item["end"] - item["start"] for item in returns)
            )
            return result
        terminal = [item for item in long_enough if item["end"] >= frame_count]
        result["terminal_runs"] = terminal
        if not terminal:
            result["verdict"] = "refuted"
            result["reason"] = (
                "every screened full occlusion is followed by a visible run, so a "
                "negative reappearance answer would be wrong here"
            )
            return result
        if requirement.require_complete_coverage and unmeasured:
            result["reason"] = (
                "a negative reappearance answer needs an explicit state at every "
                f"frame; {unmeasured} screened frames are unmeasured"
            )
            return result
        result["verdict"] = "consistent"
        result["margin"] = float(
            max(item["end"] - item["start"] for item in terminal)
        )
        return result

    if requirement.kind == "visible_occluded_to_visible_clear":
        transitions = [
            {"from": previous, "to": current}
            for previous, current in zip(runs, runs[1:])
            if previous["value"] == "visible_occluded"
            and current["value"] == "visible_clear"
            and current["end"] - current["start"] >= requirement.min_state_frames
        ]
        result["transitions"] = transitions
        if transitions:
            result["verdict"] = "consistent"
            result["margin"] = float(
                max(item["to"]["end"] - item["to"]["start"] for item in transitions)
            )
            return result
        if not any(item["value"] == "visible_occluded" for item in runs):
            result["verdict"] = "refuted"
            result["reason"] = (
                "no screened frame partially hides the subject, so it cannot become "
                "clear from a partial occlusion"
            )
            return result
        result["reason"] = (
            "the screened partial occlusions are never followed by a clear run"
        )
        return result

    if requirement.kind == "registered_occluder_visible":
        candidates: list[dict[str, Any]] = []
        for frame, state in states:
            if state not in {"visible_occluded", "fully_occluded"}:
                continue
            record = by_frame[frame]
            attribution = record.get("occluder_attribution") or {}
            named = [key for key in attribution if key != _UNRESOLVED_STATIC]
            unresolved = int(attribution.get(_UNRESOLVED_STATIC, 0))
            if len(named) != 1 or unresolved:
                continue
            if (
                requirement.occluder_subject is not None
                and named[0] != requirement.occluder_subject
            ):
                continue
            candidates.append(
                {
                    "frame_index": frame,
                    "occluder_id": named[0],
                    "blocked_samples": record.get("blocked_samples"),
                    "state": state,
                }
            )
        result["single_occluder_frames"] = candidates
        if not candidates:
            result["verdict"] = "refuted" if not any(
                state in {"visible_occluded", "fully_occluded"}
                for _frame, state in states
            ) else "undetermined"
            result["reason"] = (
                "no screened occluded frame is explained by exactly one registered "
                "instance, so the pixel pass would have no unique occluder to name"
            )
            return result
        grouped = _runs(
            [(item["frame_index"], item["occluder_id"]) for item in candidates]
        )
        stable = [
            item
            for item in grouped
            if item["end"] - item["start"] >= requirement.min_state_frames
        ]
        publishable = [
            item
            for item in stable
            if not requirement.require_publishable_window
            or publishable_window(facts_view, [item["start"], item["end"]])[
                "publishable"
            ]
        ]
        result["stable_occluder_runs"] = stable
        if not stable:
            result["reason"] = (
                "the single-occluder frames are not consecutive enough to form a "
                "stable occlusion interval"
            )
            return result
        if not publishable:
            result["verdict"] = "refuted"
            result["reason"] = (
                "the stable occlusion interval cannot be printed at the configured "
                "public time precision"
            )
            return result
        best = max(publishable, key=lambda item: item["end"] - item["start"])
        result["verdict"] = "consistent"
        result["selected"] = best
        result["margin"] = float(best["end"] - best["start"])
        return result

    raise ConditionedVisibilityError(  # pragma: no cover - guarded by the dataclass
        f"no screen is implemented for requirement kind {requirement.kind!r}"
    )


def _downgrade_proxy_refutation(
    result: dict[str, Any], requirement: VisibilityRequirement, geometry_authority: str
) -> dict[str, Any]:
    """A proxy mesh may rank an occlusion requirement, never refuse it.

    Measured on one retained UE/SPEAR capture the acoustic proxy missed the
    occluder for every frame the renderer called ``fully_occluded``, so a
    refusal drawn from that geometry would throw away exactly the candidates
    the question needs.
    """

    if (
        result["verdict"] != "refuted"
        or geometry_authority == "visual_mesh"
        or requirement.kind not in _OCCLUSION_REFUTATION_KINDS
    ):
        return result
    result["verdict"] = "undetermined"
    result["screen_refutation_withheld"] = {
        "geometry_authority": geometry_authority,
        "would_have_refuted": result.get("reason"),
    }
    result["reason"] = (
        "the screening geometry is not the geometry that renders "
        f"({geometry_authority}), so this occlusion requirement is left "
        "undetermined instead of refused"
    )
    return result


def _observation_retention(
    requirement: VisibilityRequirement,
    evaluation: Mapping[str, Any],
    *,
    frame_count: int,
) -> dict[str, Any]:
    """How much of P02's legal query window this candidate leaves usable."""

    windows = requirement.observation_windows
    total = sum(end - start for start, end in windows)
    if not windows:
        return {
            "declared": False,
            "observation_frames_total": 0,
            "observation_frames_retained": 0,
            "reason": "the caller declared no legal integer query window for this "
            "subject, so retention is not measured",
        }
    consumed: list[tuple[int, int]] = []
    selected = evaluation.get("selected")
    if isinstance(selected, Mapping):
        if "entry_frame" in selected:
            extent = selected.get("publishable_extent") or {}
            end = extent.get("end_frame") or int(selected["entry_frame"]) + 1
            consumed.append((int(selected["entry_frame"]), int(end)))
        elif "start" in selected and "end" in selected:
            consumed.append((int(selected["start"]), int(selected["end"])))
    retained = 0
    for start, end in windows:
        for frame in range(start, min(end, frame_count)):
            if any(low <= frame < high for low, high in consumed):
                continue
            retained += 1
    return {
        "declared": True,
        "observation_frames_total": total,
        "observation_frames_retained": retained,
        "frames_consumed_by_transition": [list(item) for item in consumed],
    }


def solve_visibility_candidates(
    requirements: Sequence[VisibilityRequirement],
    *,
    camera_poses: Sequence[CameraPose | Mapping[str, Any]],
    tracks: Sequence[ActorTrack],
    frame_rate_hz: float,
    mesh: Any = None,
    static_occluders: Sequence[StaticOccluder] = (),
    policy: ScreenPolicy | str | None = None,
    geometry_authority: str = "unknown",
    public_time_precision: int = 0,
    max_candidates: int | None = None,
    max_ray_poses: int | None = None,
    frame_budget: int | None = None,
) -> dict[str, Any]:
    """Rank fixed-camera candidates that could support these requirements.

    The two tiers run in order: every pose is classified at the ``frustum``
    tier first, and only the poses that survive that refutation cast static
    geometry rays.  Nothing here renders anything, so no returned candidate is
    a pass; the caller renders the ranked candidates and settles the answer
    with :func:`accept_native_visibility`.
    """

    resolved_policy = screen_policy(policy)
    requirement_list = list(requirements)
    if not requirement_list:
        raise ConditionedVisibilityError("solve_visibility_candidates needs a requirement")
    for requirement in requirement_list:
        if not isinstance(requirement, VisibilityRequirement):
            raise ConditionedVisibilityError(
                "requirements must be VisibilityRequirement instances"
            )
    poses = [
        pose if isinstance(pose, CameraPose) else CameraPose.from_mapping(pose)
        for pose in camera_poses
    ]
    if not poses:
        raise ConditionedVisibilityError("solve_visibility_candidates needs a camera pose")
    track_list = list(tracks)
    by_id = {track.instance_id: track for track in track_list}
    if len(by_id) != len(track_list):
        raise ConditionedVisibilityError("actor tracks repeat an instance id")
    frame_counts = {track.frame_count for track in track_list}
    if len(frame_counts) != 1:
        raise ConditionedVisibilityError(
            f"actor tracks disagree on the frame clock: {sorted(frame_counts)}"
        )
    frame_count = frame_counts.pop()
    rate = _positive(frame_rate_hz, owner="frame_rate_hz")
    if max_candidates is not None:
        max_candidates = _index(max_candidates, owner="max_candidates", minimum=1)
    if max_ray_poses is not None:
        max_ray_poses = _index(max_ray_poses, owner="max_ray_poses", minimum=1)
    if frame_budget is not None:
        frame_budget = _index(
            frame_budget, owner="frame_budget", minimum=frame_count
        )
    # max_candidates is also an upper bound on expensive ray/full-frame
    # evaluations. Slicing only the ranked result would still pay for every
    # ray pose, which defeats the caller's CPU budget.
    expensive_pose_budget = max_ray_poses
    if max_candidates is not None:
        expensive_pose_budget = (
            max_candidates
            if expensive_pose_budget is None
            else min(expensive_pose_budget, max_candidates)
        )

    subjects = sorted({requirement.subject for requirement in requirement_list})
    missing = [subject for subject in subjects if subject not in by_id]
    if missing:
        raise ConditionedVisibilityError(
            f"these requirement subjects have no track: {missing}"
        )

    # Per-requirement applicability, before any pose is screened.  A subject
    # that cannot walk removes exactly the requirement that needs its own
    # crossing, never the group.
    applicability: list[dict[str, Any]] = []
    for requirement in requirement_list:
        subject = by_id[requirement.subject]
        others = [item for item in track_list if item.instance_id != subject.instance_id]
        verdict = _dynamics_state(
            requirement,
            subject=subject,
            others=others,
            static_occluders=tuple(static_occluders),
        )
        applicability.append(
            {
                # ``state`` stays the pixel state this requirement asks for.
                # The planner verdict belongs to a different vocabulary and
                # used to overwrite it, so a ``fully_occluded`` requirement was
                # reported back as the capability word ``available``.
                **requirement.as_report(),
                "capability_state": verdict["state"],
                "capability_reason": verdict["reason"],
                "reason": verdict["reason"],
                "subject_can_move": subject.can_move,
                "entity_class": subject.entity_class,
                "mobile_other_instances": [
                    item.instance_id for item in others if item.can_move
                ],
            }
        )
    solvable = [
        requirement
        for requirement, row in zip(requirement_list, applicability)
        if row["capability_state"] == STATE_AVAILABLE
    ]

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "record": "candidate_solution",
        "status": "screened_no_pixels_rendered",
        "tier": "frustum_and_ray" if mesh is not None else "frustum",
        "confirmation_authority": CONFIRMATION_AUTHORITY,
        "claim_boundary": (
            "screened candidate ranking under a fixed camera. No candidate here "
            "is a proven pixel state, an answerable question or a dataset sample."
        ),
        "coordinate_frame": dict(COORDINATE_FRAME),
        "policy": resolved_policy.as_report(),
        "geometry_authority": geometry_authority,
        "pruning_boundary": (
            "screened verdicts rank and prune. A refuted candidate is one this "
            "screen will not spend a render on, not a proven impossibility: the "
            "only sound direction is that an in-frustum body sample rules out "
            "out_of_view, so discarding a candidate costs throughput and never "
            "licenses a pixel claim."
        ),
        "frame_count": frame_count,
        "frame_rate_hz": rate,
        "public_time_precision": int(public_time_precision),
        "requirements": applicability,
        "requirement_group_state": (
            STATE_AVAILABLE
            if solvable
            else (
                STATE_NOT_APPLICABLE
                if all(
                    row["capability_state"] == STATE_NOT_APPLICABLE
                    for row in applicability
                )
                else STATE_EVIDENCE_MISSING
            )
        ),
        "camera_pose_count": len(poses),
        "static_occluder_count": len(tuple(static_occluders)),
        "static_geometry_source": None if mesh is None else _mesh_source(mesh),
        "candidates": [],
        "refuted": [],
        "stages": {
            "poses_screened_frustum": 0,
            "poses_surviving_frustum": 0,
            "poses_screened_with_rays": 0,
            "poses_consistent": 0,
        },
        "budgets": {
            "max_candidates": max_candidates,
            "max_ray_poses": max_ray_poses,
            "expensive_pose_budget": expensive_pose_budget,
            "frame_budget": frame_budget,
            "required_frame_budget": frame_count,
            "frame_evaluations_per_ray_pose": frame_count,
        },
    }
    if not solvable:
        report["status"] = "no_solvable_requirement"
        return report

    ray_cache: dict[tuple[Any, ...], str] = {}
    frustum_survivors: list[tuple[CameraPose, dict[str, Any]]] = []
    for pose in poses:
        report["stages"]["poses_screened_frustum"] += 1
        series_by_subject = {
            subject: screen_visibility_series(
                camera=pose,
                tracks=track_list,
                subject=subject,
                mesh=None,
                policy=resolved_policy,
                cast_rays=False,
                geometry_authority=geometry_authority,
            )
            for subject in sorted({item.subject for item in solvable})
        }
        refusal = _frustum_refusal(solvable, series_by_subject, frame_count=frame_count)
        if refusal is not None:
            report["refuted"].append(
                {
                    "candidate_id": pose.candidate_id,
                    "tier": "frustum",
                    "kind": refusal["kind"],
                    "reason": refusal["reason"],
                }
            )
            continue
        report["stages"]["poses_surviving_frustum"] += 1
        frustum_survivors.append((pose, series_by_subject))
        if (
            expensive_pose_budget is not None
            and len(frustum_survivors) >= expensive_pose_budget
        ):
            break

    for pose, _frustum in frustum_survivors:
        report["stages"]["poses_screened_with_rays"] += 1
        evaluations: list[dict[str, Any]] = []
        series_by_subject: dict[str, Any] = {}
        for subject in sorted({item.subject for item in solvable}):
            series_by_subject[subject] = screen_visibility_series(
                camera=pose,
                tracks=track_list,
                subject=subject,
                mesh=mesh,
                static_occluders=static_occluders,
                policy=resolved_policy,
                cast_rays=True,
                ray_cache=ray_cache,
                geometry_authority=geometry_authority,
            )
        states_by_instance = {
            subject: {
                int(record["frame_index"]): {
                    "frame_index": int(record["frame_index"]),
                    "state": record.get("state"),
                }
                for record in series["frames"]
                if record.get("state") is not None
            }
            for subject, series in series_by_subject.items()
        }
        facts_view = _facts_view(
            states_by_instance=states_by_instance,
            frame_count=frame_count,
            frame_rate_hz=rate,
            precision=int(public_time_precision),
        )
        worst = "consistent"
        for requirement in solvable:
            evaluation = _evaluate_requirement(
                requirement,
                series=series_by_subject[requirement.subject],
                facts_view=facts_view,
                frame_count=frame_count,
                geometry_authority=geometry_authority,
            )
            evaluation = _downgrade_proxy_refutation(
                evaluation, requirement, geometry_authority
            )
            evaluation["observation"] = _observation_retention(
                requirement, evaluation, frame_count=frame_count
            )
            evaluations.append(evaluation)
            if evaluation["verdict"] == "refuted":
                worst = "refuted"
            elif evaluation["verdict"] == "undetermined" and worst != "refuted":
                worst = "undetermined"
        candidate = {
            "candidate_id": pose.candidate_id,
            "camera": pose.as_report(),
            "verdict": worst,
            "tier": "frustum_and_ray" if mesh is not None else "frustum",
            "requirement_screens": evaluations,
            "margin": float(
                min((item["margin"] for item in evaluations), default=0.0)
            ),
            "predicted_state_counts": {
                subject: series["predicted_state_counts"]
                for subject, series in series_by_subject.items()
            },
            "observation_frames_retained": min(
                (
                    item["observation"]["observation_frames_retained"]
                    for item in evaluations
                    if item["observation"]["declared"]
                ),
                default=None,
            ),
        }
        if worst == "refuted":
            report["refuted"].append(
                {
                    "candidate_id": pose.candidate_id,
                    "tier": candidate["tier"],
                    "kind": next(
                        item["kind"]
                        for item in evaluations
                        if item["verdict"] == "refuted"
                    ),
                    "reason": next(
                        item["reason"]
                        for item in evaluations
                        if item["verdict"] == "refuted"
                    ),
                }
            )
            continue
        if worst == "consistent":
            report["stages"]["poses_consistent"] += 1
        report["candidates"].append(candidate)

    report["candidates"].sort(
        key=lambda item: (
            0 if item["verdict"] == "consistent" else 1,
            -item["margin"],
            item["candidate_id"],
        )
    )
    if max_candidates is not None:
        report["candidates"] = report["candidates"][: int(max_candidates)]
    if not report["candidates"]:
        report["status"] = "no_candidate_survived_screening"
    return report


def _frustum_refusal(
    requirements: Sequence[VisibilityRequirement],
    series_by_subject: Mapping[str, Mapping[str, Any]],
    *,
    frame_count: int,
) -> dict[str, Any] | None:
    """Cheap tier-one refutation: what the frustum alone already rules out."""

    for requirement in requirements:
        series = series_by_subject[requirement.subject]
        records = list(series["frames"])
        in_view = [bool(record.get("in_view")) for record in records]
        if requirement.kind == "out_of_view_to_visible":
            crossed = any(
                not in_view[index] and in_view[index + 1]
                for index in range(len(in_view) - 1)
            )
            if not crossed:
                return {
                    "kind": requirement.kind,
                    "reason": "the frustum tier finds no frame where the subject "
                    "goes from having no in-image body sample to having one",
                }
            continue
        if requirement.kind == "visibility_state" and set(requirement.states) == {"out_of_view"}:
            if not any(not visible for visible in in_view):
                return {
                    "kind": requirement.kind,
                    "reason": "the subject keeps at least one in-image body sample "
                    "at every frame, so it is never out of view",
                }
            continue
        if requirement.kind in {
            "fully_occluded_then_visible",
            "fully_occluded_without_return",
            "visible_occluded_to_visible_clear",
            "registered_occluder_visible",
        } or (
            requirement.kind == "visibility_state"
            and set(requirement.states) <= {"fully_occluded", "visible_occluded", "visible_clear"}
            and requirement.states
        ):
            if not any(in_view):
                return {
                    "kind": requirement.kind,
                    "reason": "the subject never puts a body sample inside the "
                    "image, so it has no in-view footprint to occlude",
                }
    return None


# ---------------------------------------------------------------------------
# Provider capability dispatch.
# ---------------------------------------------------------------------------

VISIBILITY_FACILITIES = (
    "frustum_screen",
    "ray_occlusion_screen",
    "actor_occluder_identity_screen",
    "static_occluder_identity_screen",
    "native_pixel_witness",
    "native_actor_occluder_witness",
    "native_static_occluder_witness",
)

# renderer -> how that renderer produces the paired modal/target-only pass and
# which pixel-truth authority the result carries.
NATIVE_WITNESS_ROUTES = {
    "habitat": {
        "witness_mode": "habitat_modal_target_only_semantic_v1",
        "authority": PIXEL_VISIBILITY_AUTHORITY,
        "producer": "avengine.capture.mp3d_multi_actor",
        "compiler": "avengine.qa.pixel_visibility.compile_pixel_visibility_truth",
        "static_object_ids": False,
    },
    "ue_spear": {
        "witness_mode": "spear_modal_target_only_metric_depth_v1",
        "authority": PIXEL_VISIBILITY_DEPTH_AUTHORITY,
        "producer": "tools/rooms/run_spear_residential_episode.py",
        "compiler": "avengine.qa.pixel_visibility.compile_depth_pixel_visibility_truth",
        "static_object_ids": True,
    },
}

NATIVE_MASK_ARTIFACTS = (
    "pixel_visibility_truth.json",
    "native_pixel_masks_depth_authority_v1.npz",
)


def provider_visibility_capabilities(
    resolution: Any,
    *,
    layout: Mapping[str, Any] | None = None,
    authoring_to_shared: str | None = None,
    other_instance_count: int = 0,
    package: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report which visibility facilities one resolved room supplies.

    ``resolution`` is a :class:`avengine.rooms.room_providers.RoomResolution`
    or the mapping its ``as_report`` produces.  Every decision reads the
    declared renderer and the RoomPackage capability dimensions, so a room that
    is added by registering its package arrives with the same facilities and no
    identifier ever appears in a branch here.
    """

    route, capabilities, room_id, resolved_package = _resolution_facts(resolution)
    if package is None:
        package = resolved_package
    renderer = route.get("renderer")
    dimensions = capabilities.get("dimensions") if isinstance(capabilities, Mapping) else None
    dimensions = dimensions if isinstance(dimensions, Mapping) else {}

    def dimension_status(name: str) -> str:
        entry = dimensions.get(name)
        return str(entry.get("status")) if isinstance(entry, Mapping) else "not_run"

    static_geometry = dimension_status("static_geometry")
    semantics = dimension_status("semantics")
    visual_scene = dimension_status("visual_scene")

    facilities: dict[str, dict[str, Any]] = {
        "frustum_screen": {
            "state": STATE_AVAILABLE,
            "tier": "frustum",
            "reason": None,
            "entrypoint": "avengine.rooms.conditioned_visibility.screen_visibility_series",
            "needs": ["camera_pose", "actor_tracks"],
        }
    }
    authority = geometry_authority_for_package(package)
    facilities["ray_occlusion_screen"] = {
        "state": STATE_AVAILABLE if static_geometry == "pass" else STATE_EVIDENCE_MISSING,
        "tier": "frustum_and_ray",
        "reason": None
        if static_geometry == "pass"
        else f"the room package static_geometry dimension is {static_geometry}",
        "entrypoint": "avengine.qa.answerability.line_of_sight",
        "needs": ["static_geometry.vertices", "static_geometry.triangles"],
        "geometry_authority": authority,
        "may_refute_an_occlusion_state": authority == "visual_mesh",
        "geometry_note": None
        if authority == "visual_mesh"
        else "these arrays are not the geometry that renders, so the ray tier "
        "ranks occlusion candidates and never refuses one",
    }
    facilities["actor_occluder_identity_screen"] = {
        "state": STATE_AVAILABLE
        if int(other_instance_count) > 0
        else STATE_EVIDENCE_MISSING,
        "tier": "frustum_and_ray",
        "reason": None
        if int(other_instance_count) > 0
        else "the episode declares no other instance that could occlude the subject",
        "entrypoint": "avengine.rooms.conditioned_visibility.screen_visibility_series",
        "needs": ["other_actor_tracks"],
    }
    boxes, box_report = static_occluders_from_layout(
        layout, authoring_to_shared=authoring_to_shared
    )
    facilities["static_occluder_identity_screen"] = {
        "state": box_report["state"],
        "tier": "frustum_and_ray",
        "reason": box_report["reason"],
        "entrypoint": "avengine.rooms.conditioned_visibility.static_occluders_from_layout",
        "needs": ["room_layout.objects[].bounds_xyz_m", "authoring_to_shared"],
        "occluder_count": len(boxes),
    }

    witness = NATIVE_WITNESS_ROUTES.get(str(renderer))
    if witness is None:
        facilities["native_pixel_witness"] = {
            "state": STATE_NOT_IMPLEMENTED,
            "tier": "native_pixel",
            "reason": f"renderer {renderer!r} has no paired modal/target-only pass; "
            f"registered renderers are {sorted(NATIVE_WITNESS_ROUTES)}",
            "needs": [],
        }
        facilities["native_actor_occluder_witness"] = dict(
            facilities["native_pixel_witness"]
        )
        facilities["native_static_occluder_witness"] = dict(
            facilities["native_pixel_witness"]
        )
    else:
        blockers = [
            f"visual_scene={visual_scene}" if visual_scene != "pass" else None,
            f"semantics={semantics}" if semantics != "pass" else None,
        ]
        blockers = [item for item in blockers if item]
        facilities["native_pixel_witness"] = {
            "state": STATE_AVAILABLE if not blockers else STATE_EVIDENCE_MISSING,
            "tier": "native_pixel",
            "reason": None
            if not blockers
            else "the room package does not resolve " + ", ".join(blockers),
            "witness_mode": witness["witness_mode"],
            "authority": witness["authority"],
            "producer": witness["producer"],
            "compiler": witness["compiler"],
            "retained_artifacts": list(NATIVE_MASK_ARTIFACTS),
            "needs": ["visual_scene", "semantics"],
        }
        facilities["native_actor_occluder_witness"] = {
            "state": facilities["native_pixel_witness"]["state"],
            "tier": "native_pixel",
            "reason": facilities["native_pixel_witness"]["reason"],
            "entrypoint": "avengine.rooms.qa_evidence.derive_actor_occluders",
            "authority": "intersection_of_native_depth_modal_and_target_only_masks",
            "needs": list(NATIVE_MASK_ARTIFACTS),
        }
        facilities["native_static_occluder_witness"] = (
            {
                "state": facilities["native_pixel_witness"]["state"],
                "tier": "native_pixel",
                "reason": facilities["native_pixel_witness"]["reason"],
                "entrypoint": "tools/qa/derive_native_occluder_evidence.py",
                "authority": "same_renderer_same_camera_occluded_target_footprint_"
                "normal_static_object_ids_v1",
                "needs": ["normal_object_ids", "object_id_descriptors"],
            }
            if witness["static_object_ids"]
            else {
                "state": STATE_NOT_IMPLEMENTED,
                "tier": "native_pixel",
                "reason": "this renderer's capture does not emit per-object ids for "
                "the modal pass, so a static occluder cannot be named from pixels; "
                "an instance occluder still resolves through the actor witness",
                "needs": [],
            }
        )

    states = [entry["state"] for entry in facilities.values()]
    return {
        "schema": SCHEMA,
        "record": "provider_capabilities",
        "room_id": room_id,
        "family": route.get("family"),
        "renderer": renderer,
        "dispatch": "declared_renderer_and_room_package_dimensions",
        "status": STATE_AVAILABLE
        if all(state == STATE_AVAILABLE for state in states)
        else "partial",
        "facilities": facilities,
        "geometry_authority": authority,
        "screen_can_confirm_a_pixel_state": False,
        "confirmation_authority": CONFIRMATION_AUTHORITY,
    }


def _resolution_facts(
    resolution: Any,
) -> tuple[dict[str, Any], Mapping[str, Any], Any, Mapping[str, Any] | None]:
    """Read route, capabilities and package off a RoomResolution or its report."""

    if isinstance(resolution, Mapping):
        route = resolution.get("route")
        capabilities = resolution.get("capabilities")
        room_id = resolution.get("room_id")
        if not isinstance(route, Mapping):
            raise ConditionedVisibilityError(
                "a room resolution mapping must carry its route; an unrouted room "
                "has no visibility facilities to report"
            )
        package = resolution.get("package")
        return (
            dict(route),
            capabilities or {},
            room_id,
            package if isinstance(package, Mapping) else None,
        )
    route = getattr(resolution, "route", None)
    if route is None:
        raise ConditionedVisibilityError(
            "this room did not resolve to a route, so no renderer can be dispatched"
        )
    package = getattr(resolution, "package", None)
    return (
        {
            "family": getattr(route, "family", None),
            "renderer": getattr(route, "renderer", None),
            "walkable_kind": getattr(route, "walkable_kind", None),
            "production_family": getattr(route, "production_family", None),
        },
        getattr(resolution, "capabilities", None) or {},
        getattr(resolution, "room_id", None),
        package if isinstance(package, Mapping) else None,
    )


def visibility_knob_support() -> dict[str, Any]:
    """The planning knobs this module implements, for a live capability query.

    ``generation_conditions.KNOB_GAPS`` currently carries
    ``visibility_transition`` and ``pixel_occlusion_transition`` as static
    entries naming ``conditioned_sampler.select_camera_and_schedule``.  Once
    the sampler routes those two knobs through
    :func:`solve_visibility_candidates`, that module can read this report
    instead of hard-coding the gap.
    """

    return {
        "schema": SCHEMA,
        "record": "knob_support",
        "implemented": {
            knob: {**detail, "values": list(detail["values"]),
                   "requirement_kinds": list(detail["requirement_kinds"])}
            for knob, detail in _KNOB_IMPLEMENTATIONS.items()
        },
        "requirement_kinds": list(REQUIREMENT_KINDS),
        "screen_tiers": list(SCREEN_TIERS),
        "screen_verdicts": list(SCREEN_VERDICTS),
        "confirmation_authority": CONFIRMATION_AUTHORITY,
        "claim_boundary": (
            "planning-side knobs and a screened candidate ranking. Implementing a "
            "knob here is not evidence that any episode reached the requested "
            "pixel state."
        ),
    }


# ---------------------------------------------------------------------------
# Tier three: native pixel acceptance.
# ---------------------------------------------------------------------------


def native_acceptance_plan(
    requirements: Sequence[VisibilityRequirement],
    *,
    capabilities: Mapping[str, Any],
) -> dict[str, Any]:
    """State what the renderer must produce before a requirement can pass."""

    facilities = capabilities.get("facilities") if isinstance(capabilities, Mapping) else None
    facilities = facilities if isinstance(facilities, Mapping) else {}
    witness = facilities.get("native_pixel_witness") or {}
    rows: list[dict[str, Any]] = []
    for requirement in requirements:
        needs = list(NATIVE_MASK_ARTIFACTS)
        checks = ["bind_pixel_visibility_truth"]
        blockers: list[str] = []
        if str(witness.get("state")) != STATE_AVAILABLE:
            blockers.append(
                f"native_pixel_witness={witness.get('state')}: {witness.get('reason')}"
            )
        if requirement.require_complete_coverage:
            checks.append("unified_catalog._visibility_is_complete")
        if requirement.kind == "out_of_view_to_visible":
            checks.append("unified_catalog._entry_transition_window")
        if requirement.kind == "registered_occluder_visible":
            checks.append("unified_catalog._occlusion_interval_window")
            actor_witness = facilities.get("native_actor_occluder_witness") or {}
            static_witness = facilities.get("native_static_occluder_witness") or {}
            available = [
                name
                for name, entry in (
                    ("actor", actor_witness),
                    ("static_object", static_witness),
                )
                if str(entry.get("state")) == STATE_AVAILABLE
            ]
            needs.append("actor_occluders.json")
            if not available:
                blockers.append(
                    "no occluder witness is available on this route, so no pixel "
                    "occluder can be named"
                )
            rows.append(
                {
                    **requirement.as_report(),
                    "required_artifacts": needs,
                    "required_checks": checks,
                    "occluder_witnesses_available": available,
                    "blockers": blockers,
                    "capability_state": (
                        STATE_AVAILABLE if not blockers else STATE_EVIDENCE_MISSING
                    ),
                }
            )
            continue
        rows.append(
            {
                **requirement.as_report(),
                "required_artifacts": needs,
                "required_checks": checks,
                "blockers": blockers,
                "capability_state": (
                    STATE_AVAILABLE if not blockers else STATE_EVIDENCE_MISSING
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "record": "native_acceptance_plan",
        "witness_mode": witness.get("witness_mode"),
        "authority": witness.get("authority"),
        "compiler": witness.get("compiler"),
        "retained_artifacts": list(NATIVE_MASK_ARTIFACTS),
        "requirements": rows,
        # Plan-level readiness. Each row keeps its own pixel ``state``; the
        # per-row planner verdict is ``capability_state``.
        "state": STATE_AVAILABLE
        if rows and all(row["capability_state"] == STATE_AVAILABLE for row in rows)
        else STATE_EVIDENCE_MISSING,
    }


ACCEPTANCE_STATUSES = ("pass", "fail", "not_run")


def _acceptance_row(
    requirement: VisibilityRequirement,
    status: str,
    *,
    reason: str = "",
    **extra: Any,
) -> dict[str, Any]:
    if status not in ACCEPTANCE_STATUSES:
        raise ConditionedVisibilityError(f"unknown acceptance status {status!r}")
    row: dict[str, Any] = {
        "kind": requirement.kind,
        "subject": requirement.subject,
        "state": requirement.state,
        "allowed_states": list(requirement.allowed_states),
        "side": requirement.side,
        "status": status,
        "judge_source": "avengine/qa/unified_catalog.py",
    }
    if reason:
        row["reason"] = reason
    row.update(extra)
    return row


def accept_native_visibility(
    requirements: Sequence[VisibilityRequirement],
    *,
    pixel_truth: Mapping[str, Any],
    frame_rate_hz: float,
    frame_count: int | None = None,
    resolution_hw: Sequence[int] | None = None,
    camera_pose_ids: Sequence[str] | None = None,
    actor_by_instance: Mapping[str, str] | None = None,
    occluder_evidence: Mapping[str, Any] | None = None,
    occluder_registry: Mapping[str, Any] | None = None,
    appearance_reviewed: Sequence[str] | None = None,
    public_time_precision: int = 0,
    screen: Mapping[str, Any] | None = None,
    acceptance_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Settle each requirement against the renderer's own pixel truth.

    Every visibility predicate here is the catalog's own: the entry window
    comes from ``unified_catalog._entry_transition_window``, the occlusion
    interval from ``_occlusion_interval_window``, completeness from
    ``_visibility_is_complete`` and the public range from
    ``_display_time_bounds``.  A target that has in-view frames but no reviewed
    appearance label is reported as missing evidence, never as an absent
    target.
    """

    if not isinstance(pixel_truth, Mapping):
        raise ConditionedVisibilityError("pixel_truth must be a mapping")
    rate = _positive(frame_rate_hz, owner="frame_rate_hz")
    mapping = dict(actor_by_instance or {})
    subjects = {
        requirement.subject: mapping.get(requirement.subject, requirement.subject)
        for requirement in requirements
    }
    # Binding covers the complete captured episode when the caller supplies
    # the plan-to-capture actor mapping.  Judging remains scoped to the
    # requested requirement subjects below; truth-derived IDs never expand
    # the plan's expected actor set.
    instance_ids = sorted(set(subjects.values()))
    expected_instance_ids = (
        sorted(set(mapping.values())) if mapping else instance_ids
    )
    declared_frames = pixel_truth.get("frame_indices")
    if frame_count is None:
        if not isinstance(declared_frames, Sequence) or not declared_frames:
            raise ConditionedVisibilityError(
                "pixel truth declares no frame_indices, so frame_count must be given"
            )
        frame_count = len(declared_frames)
    frame_count = _index(frame_count, owner="frame_count", minimum=1)

    binding: dict[str, Any] = {"status": "not_run", "reason": None}
    if resolution_hw is not None:
        try:
            bind_pixel_visibility_truth(
                pixel_truth,
                expected_instance_ids=expected_instance_ids,
                expected_frame_count=frame_count,
                expected_resolution_hw=list(_resolution(resolution_hw, owner="resolution_hw")),
                expected_camera_pose_ids=(
                    None if camera_pose_ids is None else list(camera_pose_ids)
                ),
            )
            binding = {"status": "pass", "reason": None}
        except PixelVisibilityError as error:
            binding = {"status": "fail", "reason": str(error)}
    else:
        binding = {
            "status": "not_run",
            "reason": "no camera resolution was supplied, so the cross-artifact "
            "binding of this pixel truth was not checked here",
        }

    indexed = catalog._visibility_index(
        pixel_truth, actor_ids=instance_ids, frame_count=frame_count
    )
    truth_resolution = pixel_truth.get("resolution_hw")
    if resolution_hw is not None:
        truth_resolution = list(_resolution(resolution_hw, owner="resolution_hw"))
    meta = (
        {"resolution_hw": list(truth_resolution)}
        if isinstance(truth_resolution, Sequence)
        and not isinstance(truth_resolution, (str, bytes))
        and len(truth_resolution) == 2
        else None
    )
    facts_view = _facts_view(
        states_by_instance=indexed,
        frame_count=frame_count,
        frame_rate_hz=rate,
        precision=int(public_time_precision),
        occluder_evidence=occluder_evidence,
        visibility_meta=meta,
    )
    reviewed = None if appearance_reviewed is None else set(appearance_reviewed)
    from avengine.qa.visibility_interpretation import prepare_facts_for_qa, visibility_policy
    if acceptance_policy is not None:
        facts_view.setdefault("sampling", {})["acceptance_policy"] = dict(acceptance_policy)
    base_facts_view = facts_view
    tolerant_policy = visibility_policy(base_facts_view)

    rows: list[dict[str, Any]] = []
    for requirement in requirements:
        facts_view = prepare_facts_for_qa(base_facts_view, requirement.qa_id)
        instance = subjects[requirement.subject]
        frames = facts_view["visibility"].get(instance)
        if not frames:
            rows.append(
                _acceptance_row(
                    requirement,
                    "not_run",
                    reason=f"the pixel truth carries no usable visibility series for "
                    f"{instance!r}",
                    instance_id=instance,
                )
            )
            continue
        series = [frames[index] for index in sorted(frames)]
        observed = sorted({str(record.get("state")) for record in series})
        in_view = [
            int(record["frame_index"])
            for record in series
            if record.get("state") in VISIBLE_STATES
        ]
        if reviewed is not None and instance not in reviewed and in_view:
            rows.append(
                _acceptance_row(
                    requirement,
                    "not_run",
                    reason="the subject has in-view frames but no reviewed "
                    "appearance label; a missing label is missing evidence, not an "
                    "absent target",
                    instance_id=instance,
                    observed_states=observed,
                    in_view_frames=len(in_view),
                    capability_state=STATE_EVIDENCE_MISSING,
                )
            )
            continue
        rows.append(
            _judge_requirement(
                replace(requirement, occluder_subject=mapping.get(
                    requirement.occluder_subject, requirement.occluder_subject))
                if requirement.occluder_subject is not None else requirement,
                instance=instance,
                series=series,
                facts_view=facts_view,
                frame_count=frame_count,
                occluder_registry=occluder_registry,
                observed=observed,
            )
        )

    statuses = [row["status"] for row in rows]
    aggregate = (
        "fail"
        if "fail" in statuses or binding["status"] == "fail"
        else ("incomplete" if "not_run" in statuses or not rows else "pass")
    )
    result = {
        "schema": SCHEMA,
        "record": "native_acceptance",
        "tier": "native_pixel",
        "status": aggregate,
        "pixel_truth_schema": pixel_truth.get("schema"),
        "pixel_truth_authority": pixel_truth.get("authority"),
        "pixel_truth_authority_registered": pixel_truth.get("authority")
        in PIXEL_VISIBILITY_AUTHORITIES,
        "binding": binding,
        "frame_count": frame_count,
        "frame_rate_hz": rate,
        "public_time_precision": int(public_time_precision),
        "occluder_evidence_present": occluder_evidence is not None,
        "requirements": rows,
        "judge_source": "avengine/qa/unified_catalog.py",
        "claim_boundary": (
            "the requested pixel states are proven or refuted on this rendered "
            "episode. It is not human answerability, dataset admission or paper "
            "admission."
        ),
    }
    if tolerant_policy is not None and any(item.qa_id in {"QA-07", "QA-09"} for item in requirements):
        result["acceptance_policy"] = dict(acceptance_policy)
        result["tier"] = "native_measurement_with_question_tolerance"
        result["claim_boundary"] = (
            "Native measurements are unchanged. Requested questions are judged with "
            "the configured observability tolerance, not strict zero-pixel visibility.")
    if screen is not None:
        result["screen_agreement"] = compare_screen_to_witness(
            screen, pixel_truth, instance_ids=instance_ids
        )
    return result


def _judge_requirement(
    requirement: VisibilityRequirement,
    *,
    instance: str,
    series: Sequence[Mapping[str, Any]],
    facts_view: Mapping[str, Any],
    frame_count: int,
    occluder_registry: Mapping[str, Any] | None,
    observed: Sequence[str],
) -> dict[str, Any]:
    """One requirement against the pixel states, using the catalog's judges."""

    complete = catalog._visibility_is_complete(facts_view, instance)
    common = {
        "instance_id": instance,
        "observed_states": list(observed),
        "visibility_complete": complete,
    }
    if requirement.require_complete_coverage and not complete:
        return _acceptance_row(
            requirement,
            "not_run",
            reason="this requirement needs an explicit pixel state at every frame; "
            "the truth is incomplete, so a negative answer cannot be observed",
            **common,
        )

    if requirement.kind == "out_of_view_to_visible":
        meta = facts_view.get("visibility_meta")
        meta_resolution = (
            meta.get("resolution_hw") if isinstance(meta, Mapping) else None
        )
        width = (
            float(meta_resolution[1])
            if isinstance(meta_resolution, Sequence)
            and not isinstance(meta_resolution, (str, bytes))
            and len(meta_resolution) == 2
            else None
        )
        if width is None:
            return _acceptance_row(
                requirement,
                "not_run",
                reason="the pixel truth carries no image width, so an entry side "
                "cannot be measured",
                **common,
            )
        center = (width - 1.0) / 2.0
        dead_zone = max(1.0, width * 0.02)
        found: list[dict[str, Any]] = []
        for previous, current in zip(series, series[1:]):
            if int(current.get("frame_index", -1)) != int(
                previous.get("frame_index", -2)
            ) + 1:
                continue
            if previous.get("state") != "out_of_view" or current.get(
                "state"
            ) not in VISIBLE_STATES:
                continue
            centroid = current.get("target_centroid_xy_px")
            if not isinstance(centroid, Sequence) or len(centroid) != 2:
                continue
            offset = float(centroid[0]) - center
            if abs(offset) <= dead_zone:
                continue
            entry = int(current["frame_index"])
            window = catalog._entry_transition_window(
                facts_view, instance, entry, center=center, dead_zone=dead_zone
            )
            if window is None:
                continue
            display = publishable_window(facts_view, window)
            found.append(
                {
                    "entry_frame": entry,
                    "side": "right" if offset > 0.0 else "left",
                    "centroid_offset_px": offset,
                    "window_frames": list(window),
                    "public_seconds": display.get("public_seconds"),
                    "publishable": display["publishable"],
                }
            )
        if not found:
            return _acceptance_row(
                requirement,
                "fail",
                reason="no consecutive out_of_view to visible pixel pair with a "
                "sustained entry side exists in this episode",
                measured={"entry_candidates": []},
                **common,
            )
        wanted = [
            item
            for item in found
            if requirement.side is None or item["side"] == requirement.side
        ]
        accepted_observed_side = False
        if not wanted:
            policy = (facts_view.get("sampling") or {}).get("acceptance_policy") or {}
            allowed = (policy.get("accept_observed_branches") or {}).get(requirement.qa_id, ())
            observed = [item for item in found if item["side"] in allowed]
            if observed:
                # The pixels answer the question with the other side. The
                # requested side was a preference, and the policy keeps a
                # measured legal branch as measured.
                wanted = observed
                accepted_observed_side = True
        if not wanted:
            return _acceptance_row(
                requirement,
                "fail",
                reason=f"the measured entry sides are "
                f"{sorted({item['side'] for item in found})}; this branch needs "
                f"{requirement.side}",
                measured={"entry_candidates": found},
                **common,
            )
        publishable = [item for item in wanted if item["publishable"]]
        if requirement.require_publishable_window and not publishable:
            return _acceptance_row(
                requirement,
                "fail",
                reason="the entry interval is too short to print at the configured "
                "public time precision",
                measured={"entry_candidates": wanted},
                **common,
            )
        return _acceptance_row(
            requirement,
            "pass",
            measured={
                "entry_candidates": wanted,
                "selected": (publishable or wanted)[0],
                "observed_side": (publishable or wanted)[0]["side"],
                "planned_side": requirement.side,
                "accepted_observed_branch": accepted_observed_side,
            },
            **common,
        )

    if requirement.kind == "visibility_state":
        matching = [
            int(record["frame_index"])
            for record in series
            if record.get("state") in set(requirement.states)
        ]
        windows = requirement.observation_windows
        inside = (
            [
                frame
                for frame in matching
                if any(start <= frame < end for start, end in windows)
            ]
            if windows
            else matching
        )
        if len(inside) >= requirement.min_state_frames:
            return _acceptance_row(
                requirement,
                "pass",
                measured={"frames_in_state": inside},
                **common,
            )
        return _acceptance_row(
            requirement,
            "fail",
            reason=f"only {len(inside)} frames reach {' or '.join(requirement.states)} inside the "
            f"declared observation windows; {requirement.min_state_frames} are needed",
            measured={"frames_in_state": matching, "inside_observation": inside},
            **common,
        )

    if requirement.kind in {
        "fully_occluded_then_visible",
        "fully_occluded_without_return",
    }:
        hidden = [
            int(record["frame_index"])
            for record in series
            if record.get("state") == "fully_occluded"
        ]
        returned = [
            int(record["frame_index"])
            for record in series
            if record.get("state") in VISIBLE_STATES
            and hidden
            and int(record["frame_index"]) > min(hidden)
        ]
        if not hidden:
            return _acceptance_row(
                requirement,
                "fail",
                reason="no frame of this episode carries a fully_occluded pixel "
                "state; a blocked ray is a different measurement",
                measured={"fully_occluded_frames": []},
                **common,
            )
        wants_return = requirement.kind == "fully_occluded_then_visible"
        policy = (facts_view.get("sampling") or {}).get("acceptance_policy") or {}
        allowed_answers = (policy.get("accept_observed_branches") or {}).get(requirement.qa_id, ())
        observed_answer = "yes" if returned else "no"
        accepts_observed = observed_answer in allowed_answers
        seen_before = [
            int(record["frame_index"])
            for record in series
            if record.get("state") in VISIBLE_STATES
            and int(record["frame_index"]) < min(hidden)
        ]
        if not returned and not seen_before:
            # "Did X reappear after being hidden?" presupposes X was seen. A
            # target hidden from its first occluded frame to the end and never
            # visible before gives the viewer nothing to answer about.
            return _acceptance_row(
                requirement,
                "fail",
                reason="the target is never visible before its full occlusion, so a "
                "viewer cannot know whom a negative reappearance question is about",
                measured={
                    "fully_occluded_frames": hidden,
                    "visible_before_full_occlusion": [],
                    "observed_answer": observed_answer,
                },
                **common,
            )
        if bool(returned) == wants_return or accepts_observed:
            return _acceptance_row(
                requirement,
                "pass",
                measured={
                    "fully_occluded_frames": hidden,
                    "visible_after_full_occlusion": returned,
                    "visible_before_full_occlusion": seen_before,
                    "observed_answer": observed_answer,
                    "planned_answer": "yes" if wants_return else "no",
                    "accepted_observed_branch": accepts_observed,
                },
                **common,
            )
        return _acceptance_row(
            requirement,
            "fail",
            reason="the measured reappearance answer is "
            f"{'yes' if returned else 'no'}; this branch needs "
            f"{'yes' if wants_return else 'no'}",
            measured={
                "fully_occluded_frames": hidden,
                "visible_after_full_occlusion": returned,
            },
            **common,
        )

    if requirement.kind == "visible_occluded_to_visible_clear":
        transitions = [
            {
                "previous_frame": int(previous["frame_index"]),
                "frame": int(current["frame_index"]),
            }
            for previous, current in zip(series, series[1:])
            if int(current.get("frame_index", -1))
            == int(previous.get("frame_index", -2)) + 1
            and previous.get("state") == "visible_occluded"
            and current.get("state") == "visible_clear"
        ]
        if transitions:
            return _acceptance_row(
                requirement,
                "pass",
                measured={"transitions": transitions},
                **common,
            )
        return _acceptance_row(
            requirement,
            "fail",
            reason="no consecutive visible_occluded to visible_clear pixel pair "
            "exists in this episode",
            measured={"transitions": []},
            **common,
        )

    if requirement.kind == "registered_occluder_visible":
        frames: list[dict[str, Any]] = []
        for record in series:
            frame = int(record["frame_index"])
            if record.get("state") not in {"visible_occluded", "fully_occluded"}:
                continue
            ids = catalog._occluder_ids(facts_view, instance, frame)
            if len(ids) != 1:
                continue
            if (
                requirement.occluder_subject is not None
                and ids[0] != requirement.occluder_subject
            ):
                continue
            window = catalog._occlusion_interval_window(facts_view, instance, frame)
            if window is None:
                continue
            display = publishable_window(facts_view, window)
            frames.append(
                {
                    "frame_index": frame,
                    "occluder_instance_id": ids[0],
                    "window_frames": list(window),
                    "public_seconds": display.get("public_seconds"),
                    "publishable": display["publishable"],
                    "registered": (
                        None
                        if occluder_registry is None
                        else ids[0] in occluder_registry
                    ),
                }
            )
        if not frames:
            return _acceptance_row(
                requirement,
                "fail",
                reason="no occluded frame resolves to exactly one occluder instance "
                "over a stable interval",
                measured={"single_occluder_frames": []},
                **common,
            )
        unregistered = [
            item["occluder_instance_id"] for item in frames if item["registered"] is False
        ]
        if unregistered:
            return _acceptance_row(
                requirement,
                "not_run",
                reason="the named pixel occluders are not in the supplied registry, "
                "so no option list can be built",
                measured={
                    "single_occluder_frames": frames,
                    "unregistered_occluder_ids": sorted(set(unregistered)),
                },
                capability_state=STATE_EVIDENCE_MISSING,
                **common,
            )
        publishable = [item for item in frames if item["publishable"]]
        if requirement.require_publishable_window and not publishable:
            return _acceptance_row(
                requirement,
                "fail",
                reason="the stable occlusion interval is too short to print at the "
                "configured public time precision",
                measured={"single_occluder_frames": frames},
                **common,
            )
        return _acceptance_row(
            requirement,
            "pass",
            measured={
                "single_occluder_frames": frames,
                "selected": (publishable or frames)[0],
            },
            **common,
        )

    raise ConditionedVisibilityError(  # pragma: no cover - guarded by the dataclass
        f"no acceptance judge is implemented for {requirement.kind!r}"
    )


def compare_screen_to_witness(
    screen: Mapping[str, Any],
    pixel_truth: Mapping[str, Any],
    *,
    instance_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score one screened series against the rendered pixel states.

    Four counts are separated because they have four different causes and only
    the first one would break a claim this module makes:

    ``in_view_direction_violations``
        The screen put a body sample inside the image and the renderer reported
        ``out_of_view``.  The one sound direction of the screen says this
        cannot happen: a real point on the body projecting inside the frame
        means the target-only footprint is not empty.

    ``out_of_view_prediction_errors``
        The screen found no sample inside the image and the renderer saw the
        target anyway.  Expected: the nine-sample hull under-covers a
        silhouette, so this is a pruning cost, not a contradiction.

    ``occlusion_underprediction`` / ``occlusion_overprediction``
        The screened occlusion split disagreed.  Expected wherever the
        screening geometry is an acoustic proxy rather than the room's visual
        mesh, which is why ``geometry_authority`` travels with every series.

    ``unsound_refutations`` counts the frames whose recorded ``refutes`` set
    contained the state the renderer then produced, whatever the cause.
    """

    if not isinstance(screen, Mapping) or screen.get("record") != "screen_series":
        raise ConditionedVisibilityError("screen must be a screen_series record")
    subject = screen.get("subject")
    per_instance = pixel_truth.get("per_instance")
    if not isinstance(per_instance, Mapping):
        raise ConditionedVisibilityError("pixel truth carries no per_instance record")
    lookup = str(subject)
    if lookup not in per_instance and instance_ids:
        matches = [item for item in instance_ids if item in per_instance]
        lookup = matches[0] if len(matches) == 1 else lookup
    entry = per_instance.get(lookup)
    if not isinstance(entry, Mapping) or not isinstance(entry.get("frames"), Sequence):
        return {
            "status": "not_run",
            "reason": f"the pixel truth has no frames for {lookup!r}",
            "subject": subject,
        }
    witnessed = {
        int(record["frame_index"]): str(record.get("state"))
        for record in entry["frames"]
        if isinstance(record, Mapping) and isinstance(record.get("frame_index"), int)
    }
    matrix: dict[str, dict[str, int]] = {}
    unsound: list[dict[str, Any]] = []
    in_view_violations: list[dict[str, Any]] = []
    out_of_view_errors = 0
    under_predicted = 0
    over_predicted = 0
    compared = 0
    agreed = 0
    for record in screen.get("frames") or ():
        frame = int(record["frame_index"])
        truth = witnessed.get(frame)
        predicted = record.get("state")
        if truth is None:
            continue
        compared += 1
        agreed += int(truth == predicted)
        matrix.setdefault(truth, {})
        key = predicted if predicted is not None else "unmeasured"
        matrix[truth][key] = matrix[truth].get(key, 0) + 1
        if bool(record.get("in_view")) and truth == "out_of_view":
            in_view_violations.append(
                {
                    "frame_index": frame,
                    "in_frustum_samples": record.get("in_frustum_samples"),
                    "visible_samples": record.get("visible_samples"),
                }
            )
        if not record.get("in_view") and truth != "out_of_view":
            out_of_view_errors += 1
        if truth == "fully_occluded" and predicted in VISIBLE_STATES:
            under_predicted += 1
        if predicted == "fully_occluded" and truth in VISIBLE_STATES:
            over_predicted += 1
        if truth in (record.get("refutes") or ()):
            unsound.append(
                {
                    "frame_index": frame,
                    "witnessed": truth,
                    "screen_predicted": predicted,
                    "screen_refuted": list(record.get("refutes") or ()),
                }
            )
    return {
        "status": "measured" if compared else "not_run",
        "subject": subject,
        "pixel_instance_id": lookup,
        "geometry_authority": screen.get("geometry_authority"),
        "frames_compared": compared,
        "frames_agreed": agreed,
        "agreement": (agreed / compared) if compared else None,
        "confusion_witnessed_to_screened": matrix,
        "in_view_direction_violations": in_view_violations,
        "in_view_direction_violation_count": len(in_view_violations),
        "out_of_view_prediction_errors": out_of_view_errors,
        "occlusion_underprediction": under_predicted,
        "occlusion_overprediction": over_predicted,
        "unsound_refutations": unsound,
        "unsound_refutation_count": len(unsound),
        "policy": screen.get("policy"),
        "claim_boundary": (
            "a positive control for the screen against one rendered episode. "
            "Agreement here does not license the screen to confirm a pixel state."
        ),
    }

"""Solve where the sound sits and where the movement sits, in one ten-second clip.

``generation_conditions`` compiles a QA type and its key branch into named
requirements.  Three of the requirements it compiles have no solver, which is
why it reports them as ``interface_not_implemented``: ``competitor_motion``,
``target_moved_after_sound`` and ``distance_trend_during_event``.  All three
are statements about *time* - which frames a body is allowed to move in,
relative to which audible window - and none of them can be answered by looking
at a route or a sound on its own.  This module is that solver.  It consumes
the compiled conditions, the cropped activity coordinates a segment carries,
and the episode clock, and returns concrete frame intervals plus the reason a
combination was refused.

Three different motion requirements
-----------------------------------

The previous wave collapsed every post-sound question into the
``cross_time_state`` recipe, which moves the actor only after the measured wet
tail.  That silently deleted the legal candidates in which the actor moves
*during* the tail, and it is not what the questions say:

``during_audible_window``
    ``QA-06`` on its ``moving`` branch and ``QA-15``.  The target's complete
    audible window has to sit inside one motion state, because the shipped
    judge - ``unified_catalog._stable_motion_window`` - reads every frame from
    the event start frame through the event end frame and refuses a window
    whose motion state changes.  Movement is required to overlap the sound.

``post_sound_query_only``
    ``QA-13``, ``QA-16`` and ``QA-17`` outside a core recipe.  What has to be
    after the measured wet tail is the *query moment*, not the movement.  An
    actor that starts walking while the reverberation is still audible is a
    perfectly good candidate: the answer interval starts at the anchor event's
    end frame, and the tail only constrains where the question may point.
    Forcing the motion after the tail here throws away legal episodes and
    changes the question.

``after_wet_tail``
    The ``cross_time_state`` core recipe only.  It captures a static early
    pass, measures the tail from the real binaural readback, and *then* opens
    the motion window at ``ceil(max(wet_tail_end_s) * fps) + 1``.  This is the
    one place where the movement itself is placed after the tail.

Which of the three applies is never re-derived from a QA id here.  It is read
back out of the compiled conditions, from the ``motion_during_event`` and
``motion_window_placement`` entries that ``generation_conditions`` produced, so
a change to the question's meaning travels through one definition instead of
two that drift apart.

Two clocks, and a tail that is not known yet
--------------------------------------------

Planning happens before any acoustic render, so the measured wet tail does not
exist yet.  A plan therefore carries the *reserved* tail - the caller's three
second budget - and says so: ``tail_basis`` is ``planned_reserve_s`` and
``requires_recheck_after_measurement`` is true.  After the render,
:func:`requery_after_measured_tail` replaces the reservation with the measured
interval and reports whether the query frame moved.  It did move, the answer
truth is recomputed; a query moment that changes and a truth that does not is
the failure mode this split exists to prevent.

What this module does not do
----------------------------

It does not sample the navigation mesh, choose a camera, or select a sound.
Route geometry belongs to ``conditioned_sampler``, and choosing which recording
an episode uses is a decision that has to be recorded with its policy and its
eligible count rather than made implicitly by whichever candidate happens to be
shortest.  :func:`solve_motion_windows` takes the candidates it is given and
answers, per candidate, whether the timing works and what path length that
would need.  It also does not certify a delivered episode: proving a condition
on real facts is ``generation_conditions.check_conditions``, and this module
calls that vocabulary rather than growing a second one.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from avengine.dataset.source_capabilities import (
    STATE_AVAILABLE,
    STATE_EVIDENCE_MISSING,
    STATE_NOT_APPLICABLE,
    locomotion_capability,
    segment_budget,
    source_family,
)
from avengine.qa import generation_conditions as gc
from avengine.qa import unified_catalog as catalog

__all__ = [
    "ConditionedMotionError",
    "EpisodeClock",
    "MotionBudget",
    "SoundCandidate",
    "MotionRequirement",
    "EventPlacement",
    "MotionSolution",
    "MOTION_SEMANTICS",
    "MOVING_FLAG_CONVENTIONS",
    "COMPETITOR_MOTION_VALUES",
    "motion_semantics",
    "resolve_anchor",
    "resolve_distance_trend_criterion",
    "solve_motion_windows",
    "requery_after_measured_tail",
    "moving_flags_from_path",
    "build_motion_trajectory",
    "static_trajectory",
    "distance_trend",
    "post_sound_distance_stability",
    "feasible_sound_candidates",
    "assert_static_camera",
    "verify_motion_candidate",
    "solver_signature",
]

SOLVER_NAME = "avengine.rooms.conditioned_motion"
SOLVER_VERSION = "p03_joint_motion_v1"

# The three requirements above, plus the case where a question says nothing
# about the target's own locomotion at all.  ``unconstrained`` is not a fourth
# recipe: it records that no compiled condition placed the movement, so a
# solver must not invent a placement to make coverage look better.
MOTION_SEMANTICS = (
    "during_audible_window",
    "post_sound_query_only",
    "after_wet_tail",
    "unconstrained",
)

# How a per-frame boolean ``moving`` track is derived from a per-frame path.
# ``conditioned_sampler.sample_routes`` repeats the final difference, so the
# last frame inherits the previous frame's state; ``binding_group_motion``
# appends a zero, so the final frame is always still.  A candidate validated
# under one convention and rendered under the other disagrees on exactly one
# frame - the last - which is enough to flip a ``QA-17`` answer whose query
# frame is the final frame.  The convention is therefore explicit and recorded.
MOVING_FLAG_CONVENTIONS = ("forward_difference_hold_last", "forward_difference_last_still")
DEFAULT_MOVING_FLAG_CONVENTION = "forward_difference_hold_last"

# ``generation_conditions`` plans this knob for the competitor role.  ``any``
# exists so an existing caller that does not care keeps its current behaviour;
# it is never chosen here to paper over an unsatisfiable ``still``/``moving``.
COMPETITOR_MOTION_VALUES = ("still", "moving", "any")

DEFAULT_MOVING_THRESHOLD_MPS = 0.05
DEFAULT_WALK_SPEED_RANGE_MPS = (0.5, 0.8)
DEFAULT_RESERVE_TAIL_S = 3.0
DEFAULT_MIN_GAP_BETWEEN_AUDIBLE_WINDOWS_S = 0.5

# The endpoint-only judge that ships today accepts a walk that goes away and
# comes back as long as the two ends differ by the margin.  ``QA-15`` asks
# whether the source got nearer or farther *while it was sounding*, so a
# planner that only checks the endpoints will happily build a candidate whose
# honest answer is "both".  Until P08 lands the catalog's own definition this
# is the planning-side criterion, and it is deliberately stricter than the
# shipped judge: anything it accepts, the endpoint judge also accepts.
DEFAULT_DISTANCE_TREND_CRITERION: dict[str, Any] = {
    "net_change_at_least_m": gc.DISTANCE_MARGIN_M,
    "max_reversal_m": 0.05,
    "min_frames": 2,
    "measured_over": "complete_audible_window_frames",
    "definition": (
        "the signed listener distance must change by at least the margin between the "
        "first and last frame of the audible window, and no backward excursion inside "
        "that window may exceed max_reversal_m"
    ),
}
# Where P08 is expected to publish the agreed criterion.  Both an attribute and
# a callable are accepted so this does not dictate the shape of someone else's
# module; whichever exists wins, and the source is recorded on every candidate.
DISTANCE_TREND_CRITERION_HOOKS = (
    ("avengine/qa/unified_catalog.py:distance_trend_criterion", "distance_trend_criterion"),
    ("avengine/qa/unified_catalog.py:DISTANCE_TREND_CRITERION", "DISTANCE_TREND_CRITERION"),
)

# Refusal codes.  A refusal names one measurable cause; "no candidate" without
# a cause is what made the previous batch impossible to argue with.
REJECTION_CODES = (
    "audio_does_not_fit_reserved_tail",
    "audible_window_missing_activity_measurement",
    "audible_window_outside_clip",
    "activity_intervals_outside_prepared_segment",
    "motion_window_shorter_than_two_frames",
    "motion_window_outside_clip",
    "motion_window_collides_with_end_hold",
    "no_legal_query_frame_after_reserved_tail",
    "no_displayable_integer_query_window",
    "target_cannot_self_locomote",
    "no_locomotion_capable_competitor",
    "anchor_role_not_declared",
    "competitor_shares_target_answer",
    "required_path_length_exceeds_available",
    "speed_outside_declared_range",
    "unsatisfiable_motion_semantics",
)


class ConditionedMotionError(ValueError):
    """A motion request is malformed or contradicts the compiled conditions."""


def _positive(value: Any, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConditionedMotionError(f"{owner} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ConditionedMotionError(f"{owner} must be finite and positive")
    return number


def _nonnegative(value: Any, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConditionedMotionError(f"{owner} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ConditionedMotionError(f"{owner} must be finite and nonnegative")
    return number


def _exact_int(value: Any, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConditionedMotionError(f"{owner} must be an integer")
    return int(value)


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class EpisodeClock:
    """The one clock every interval in a solution is expressed in.

    Samples and frames are both kept because the audio side counts samples and
    the motion side counts frames, and a solver that converts silently is a
    solver whose off-by-one lands in a rendered episode.
    """

    frame_count: int
    frame_rate_hz: float
    sample_rate_hz: int

    def __post_init__(self) -> None:
        if _exact_int(self.frame_count, "frame_count") < 2:
            raise ConditionedMotionError("an episode needs at least two video frames")
        _positive(self.frame_rate_hz, "frame_rate_hz")
        if _exact_int(self.sample_rate_hz, "sample_rate_hz") <= 0:
            raise ConditionedMotionError("sample_rate_hz must be positive")

    @classmethod
    def from_mapping(cls, clock: Mapping[str, Any]) -> "EpisodeClock":
        """Accept the ``plan['clock']`` mapping the rest of the pipeline passes."""

        if not isinstance(clock, Mapping):
            raise ConditionedMotionError("clock must be a mapping")
        missing = [key for key in ("frame_count", "frame_rate_hz", "sample_rate_hz")
                   if key not in clock]
        if missing:
            raise ConditionedMotionError(f"clock is missing {missing}")
        return cls(
            frame_count=_exact_int(clock["frame_count"], "clock.frame_count"),
            frame_rate_hz=float(clock["frame_rate_hz"]),
            sample_rate_hz=_exact_int(clock["sample_rate_hz"], "clock.sample_rate_hz"),
        )

    @property
    def duration_s(self) -> float:
        return float(self.frame_count) / float(self.frame_rate_hz)

    def frame_of_time(self, seconds: float) -> int:
        """The frame index a moment falls in, clamped to the clip."""

        raw = int(math.floor(float(seconds) * float(self.frame_rate_hz)))
        return max(0, min(self.frame_count - 1, raw))

    def first_frame_after(self, seconds: float) -> int:
        """The first frame whose own time is strictly after ``seconds``.

        This is the frame rule ``_silent_after`` uses: a query frame at exactly
        the tail end is inside the tail, not after it.
        """

        raw = math.floor(float(seconds) * float(self.frame_rate_hz)) + 1
        return int(max(0, raw))

    def frame_of_sample(self, sample: int) -> int:
        return int(math.floor(int(sample) * float(self.frame_rate_hz) / float(self.sample_rate_hz)))

    def frame_ceil_of_sample(self, sample: int) -> int:
        return int(math.ceil(int(sample) * float(self.frame_rate_hz) / float(self.sample_rate_hz)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_count": int(self.frame_count),
            "frame_rate_hz": float(self.frame_rate_hz),
            "sample_rate_hz": int(self.sample_rate_hz),
            "duration_s": self.duration_s,
        }


@dataclass(frozen=True)
class MotionBudget:
    """The caller's declared limits.  Nothing here has a hidden default cap.

    ``reserve_tail_s`` is the owner's current three second terminal reserve and
    ``walk_speed_range_mps`` the declared legal speed band.  ``max_clip_s`` is
    deliberately optional and defaults to ``None``: the five second value that
    lives in the old sound-pool configuration is a historical filter, not a
    universal upper bound on a cropped segment, and this solver never applies
    it unless the caller states it.
    """

    reserve_tail_s: float = DEFAULT_RESERVE_TAIL_S
    walk_speed_range_mps: tuple[float, float] = DEFAULT_WALK_SPEED_RANGE_MPS
    moving_threshold_mps: float = DEFAULT_MOVING_THRESHOLD_MPS
    min_gap_between_audible_windows_s: float = DEFAULT_MIN_GAP_BETWEEN_AUDIBLE_WINDOWS_S
    earliest_start_s: float = 0.0
    end_hold_s: float = 0.0
    anchor_pre_silence_s: float = gc.ANCHOR_PRE_SILENCE_S
    max_clip_s: float | None = None
    # How much movement an "any movement" question needs before it counts.  The
    # default of zero means one inter-frame step, which is what the predicate
    # literally asks for; a caller that wants a walk a viewer can see states a
    # duration rather than having one invented here.
    minimum_motion_s: float = 0.0
    public_time_precision: int = gc.DEFAULT_PUBLIC_TIME_PRECISION
    moving_flag_convention: str = DEFAULT_MOVING_FLAG_CONVENTION

    def __post_init__(self) -> None:
        _nonnegative(self.reserve_tail_s, "reserve_tail_s")
        _nonnegative(self.earliest_start_s, "earliest_start_s")
        _nonnegative(self.end_hold_s, "end_hold_s")
        _nonnegative(self.anchor_pre_silence_s, "anchor_pre_silence_s")
        _nonnegative(self.min_gap_between_audible_windows_s,
                     "min_gap_between_audible_windows_s")
        _nonnegative(self.minimum_motion_s, "minimum_motion_s")
        _positive(self.moving_threshold_mps, "moving_threshold_mps")
        speeds = self.walk_speed_range_mps
        if len(speeds) != 2:
            raise ConditionedMotionError("walk_speed_range_mps must be [min, max]")
        low, high = _positive(speeds[0], "walk_speed_range_mps[0]"), _positive(
            speeds[1], "walk_speed_range_mps[1]")
        if low > high:
            raise ConditionedMotionError("walk_speed_range_mps must be ordered")
        if low <= self.moving_threshold_mps:
            raise ConditionedMotionError(
                "the slowest declared walk must exceed the moving threshold, or a walking "
                "actor reads as still")
        if self.max_clip_s is not None:
            _positive(self.max_clip_s, "max_clip_s")
        if (isinstance(self.public_time_precision, bool)
                or not isinstance(self.public_time_precision, int)
                or not 0 <= self.public_time_precision <= 9):
            raise ConditionedMotionError(
                "public_time_precision must be an integer from 0 through 9")
        if self.moving_flag_convention not in MOVING_FLAG_CONVENTIONS:
            raise ConditionedMotionError(
                f"moving_flag_convention must be one of {MOVING_FLAG_CONVENTIONS}")

    @classmethod
    def from_profile(
        cls, profile: Mapping[str, Any] | None = None, **overrides: Any
    ) -> "MotionBudget":
        """Read the fields a ``conditioned_sampler`` profile already carries.

        An explicit keyword wins over the profile, so a caller that states a
        speed or a clock does not have that request overwritten by a route
        default - the exact substitution the owner refused.
        """

        profile = dict(profile or {})
        values: dict[str, Any] = {}
        if "reserve_tail_s" in profile:
            values["reserve_tail_s"] = float(profile["reserve_tail_s"])
        if "min_gap_between_audible_windows_s" in profile:
            values["min_gap_between_audible_windows_s"] = float(
                profile["min_gap_between_audible_windows_s"])
        if "walk_speed_range_mps" in profile:
            speeds = profile["walk_speed_range_mps"]
            values["walk_speed_range_mps"] = (float(speeds[0]), float(speeds[1]))
        for name in ("end_hold_s", "earliest_start_s", "max_clip_s",
                     "minimum_motion_s", "moving_threshold_mps",
                     "public_time_precision", "moving_flag_convention"):
            if name in profile:
                values[name] = profile[name]
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reserve_tail_s": float(self.reserve_tail_s),
            "walk_speed_range_mps": [float(v) for v in self.walk_speed_range_mps],
            "moving_threshold_mps": float(self.moving_threshold_mps),
            "min_gap_between_audible_windows_s": float(self.min_gap_between_audible_windows_s),
            "earliest_start_s": float(self.earliest_start_s),
            "end_hold_s": float(self.end_hold_s),
            "anchor_pre_silence_s": float(self.anchor_pre_silence_s),
            "max_clip_s": None if self.max_clip_s is None else float(self.max_clip_s),
            "minimum_motion_s": float(self.minimum_motion_s),
            "public_time_precision": int(self.public_time_precision),
            "moving_flag_convention": self.moving_flag_convention,
            "max_clip_s_note": (
                "None means the caller stated no clip cap; the historical five second "
                "sound-pool filter is not applied here"
            ),
        }


@dataclass(frozen=True)
class SoundCandidate:
    """One prepared segment, in the coordinates the segment itself is written in.

    Built from the row ``sound_segments.pool_row`` emits.  Two things about
    that row are load-bearing and are checked rather than assumed:

    * ``source_activity_intervals_samples`` in a *pool row* is in prepared
      segment samples, while the same key in the *plan record* it was built
      from is in whole-recording samples at the analysis rate.  Handing the
      record to this constructor would silently place the audible window
      somewhere in the middle of a ten second clip, so intervals that escape
      the prepared segment are refused with that collision named.
    * The internal gaps are kept, not closed.  The owner authorised natural
      pauses inside a sounding segment, so the audible window runs from the
      first activity start to the last activity end and the pauses stay inside
      it.  ``max_internal_silence_s`` is carried so a consumer can see them.
    """

    sound_asset_id: str
    sample_rate_hz: int
    sample_count: int
    audible_start_sample: int
    audible_end_sample_exclusive: int
    activity_intervals_samples: tuple[tuple[int, int], ...] = ()
    activity_basis: str = "event_bounding_box"
    max_internal_silence_s: float | None = None
    activity_coverage: float | None = None
    source_origin: str | None = None
    source_crop_start_sample: int | None = None
    source_crop_end_sample_exclusive: int | None = None
    processing_identity: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if _exact_int(self.sample_rate_hz, "sample_rate_hz") <= 0:
            raise ConditionedMotionError("sound sample_rate_hz must be positive")
        if _exact_int(self.sample_count, "sample_count") <= 0:
            raise ConditionedMotionError("sound sample_count must be positive")
        start = _exact_int(self.audible_start_sample, "audible_start_sample")
        end = _exact_int(self.audible_end_sample_exclusive, "audible_end_sample_exclusive")
        if not 0 <= start < end <= self.sample_count:
            raise ConditionedMotionError(
                f"audible window [{start}, {end}) escapes the prepared segment of "
                f"{self.sample_count} samples")
        for first, last in self.activity_intervals_samples:
            if not 0 <= first < last <= self.sample_count:
                raise ConditionedMotionError(
                    f"activity interval [{first}, {last}) escapes the prepared segment of "
                    f"{self.sample_count} samples; a plan record's "
                    "source_activity_intervals_samples is in whole-recording samples at "
                    "the analysis rate, a pool row's is in prepared segment samples - "
                    "pass the pool row")

    @property
    def duration_s(self) -> float:
        return float(self.sample_count) / float(self.sample_rate_hz)

    @property
    def audible_duration_s(self) -> float:
        return float(self.audible_end_sample_exclusive - self.audible_start_sample) / float(
            self.sample_rate_hz)

    @property
    def has_measured_activity(self) -> bool:
        return self.activity_basis == "measured_activity_intervals"

    @classmethod
    def from_pool_row(cls, row: Mapping[str, Any]) -> "SoundCandidate":
        """Read one candidate out of a sound pool row, old shape or new."""

        if not isinstance(row, Mapping):
            raise ConditionedMotionError("a sound candidate must be a mapping")
        for key in ("sound_asset_id", "sample_rate_hz", "sample_count"):
            if key not in row:
                raise ConditionedMotionError(f"sound pool row is missing {key!r}")
        intervals_raw = row.get("source_activity_intervals_samples")
        intervals: tuple[tuple[int, int], ...] = ()
        basis = "event_bounding_box"
        if isinstance(intervals_raw, Sequence) and not isinstance(intervals_raw, (str, bytes)):
            collected = []
            for pair in intervals_raw:
                if not isinstance(pair, Sequence) or len(pair) != 2:
                    raise ConditionedMotionError(
                        "each activity interval must be a [start, end) pair")
                collected.append((int(pair[0]), int(pair[1])))
            if collected:
                intervals = tuple(collected)
                basis = "measured_activity_intervals"
        start = row.get("audible_start_sample")
        end = row.get("audible_end_sample_exclusive")
        if start is None or end is None:
            if not intervals:
                raise ConditionedMotionError(
                    f"{row['sound_asset_id']!r} carries neither an audible window nor "
                    "measured activity intervals; a sample count is not an audible window")
            start, end = intervals[0][0], intervals[-1][1]
        return cls(
            sound_asset_id=str(row["sound_asset_id"]),
            sample_rate_hz=int(row["sample_rate_hz"]),
            sample_count=int(row["sample_count"]),
            audible_start_sample=int(start),
            audible_end_sample_exclusive=int(end),
            activity_intervals_samples=intervals,
            activity_basis=basis,
            max_internal_silence_s=(
                None if row.get("max_internal_silence_s") is None
                else float(row["max_internal_silence_s"])),
            activity_coverage=(
                None if row.get("activity_coverage") is None
                else float(row["activity_coverage"])),
            source_origin=row.get("source_origin") or row.get("path"),
            source_crop_start_sample=(
                None if row.get("source_crop_start_sample") is None
                else int(row["source_crop_start_sample"])),
            source_crop_end_sample_exclusive=(
                None if row.get("source_crop_end_sample_exclusive") is None
                else int(row["source_crop_end_sample_exclusive"])),
            processing_identity={
                key: row[key]
                for key in ("linear_gain", "normalization_applied", "source_sha256",
                            "source_rate_hz", "crop_authorization", "selection_authorized")
                if key in row
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sound_asset_id": self.sound_asset_id,
            "sample_rate_hz": int(self.sample_rate_hz),
            "sample_count": int(self.sample_count),
            "duration_s": self.duration_s,
            "audible_start_sample": int(self.audible_start_sample),
            "audible_end_sample_exclusive": int(self.audible_end_sample_exclusive),
            "audible_duration_s": self.audible_duration_s,
            "activity_intervals_samples": [list(pair) for pair in self.activity_intervals_samples],
            "activity_basis": self.activity_basis,
            "max_internal_silence_s": self.max_internal_silence_s,
            "activity_coverage": self.activity_coverage,
            "source_origin": self.source_origin,
            "source_crop_start_sample": self.source_crop_start_sample,
            "source_crop_end_sample_exclusive": self.source_crop_end_sample_exclusive,
            "processing_identity": dict(self.processing_identity),
        }


# --------------------------------------------------------------------------- semantics


def _conditions_of(compiled: Any) -> tuple[Any, ...]:
    conditions = getattr(compiled, "conditions", None)
    if conditions is None:
        raise ConditionedMotionError(
            "expected a generation_conditions.CompiledConditions; a plain mapping does "
            "not carry the compiled predicates this solver reads")
    return tuple(conditions)


def _condition(compiled: Any, key: str) -> Any | None:
    """Find a compiled condition by key, or failing that by kind.

    A key names one question's phrasing and a kind names what the predicate is.
    Reading only the key meant QA-25's hidden_motion_changed - which is a
    motion_during_event in every respect - was invisible here, so the AV
    subset fell through to post_sound_query_only and the solver demanded a
    whole second after the wet tail that the question never asks for.
    """
    for item in _conditions_of(compiled):
        if item.key == key:
            return item
    for item in _conditions_of(compiled):
        if item.kind == key:
            return item
    return None


def _conditions_of_kind(compiled: Any, kind: str) -> list[Any]:
    return [item for item in _conditions_of(compiled) if item.kind == kind]


def motion_semantics(compiled: Any) -> dict[str, Any]:
    """Read which of the three motion requirements a compiled set carries.

    The reading is done on the compiled conditions rather than on the QA id,
    so the "one placement per meaning" decision lives in
    ``generation_conditions`` alone.  A caller that needs to know *why* gets
    the exact condition keys that produced the answer.
    """

    qa_id = getattr(compiled, "qa_id", None)
    branch = getattr(compiled, "branch", None)
    during = _condition(compiled, "motion_during_event")
    placements = [item for item in _conditions_of_kind(compiled, "motion_window_placement")]
    placement_values = {str(item.detail.get("placement")) for item in placements}
    after_sound = _condition(compiled, "motion_after_sound")
    tail = _condition(compiled, "wet_tail_readback")

    if gc.MOTION_AFTER_TAIL in placement_values:
        semantics = "after_wet_tail"
        because = ["motion_window_placement=" + gc.MOTION_AFTER_TAIL]
    elif during is not None and bool(during.detail.get("moving")):
        semantics = "during_audible_window"
        because = ["motion_during_event(moving=True)"]
        if gc.MOTION_DURING_EVENT in placement_values:
            because.append("motion_window_placement=" + gc.MOTION_DURING_EVENT)
    elif after_sound is not None or tail is not None:
        semantics = "post_sound_query_only"
        because = [key for key in ("post_sound_silent_window", "motion_after_sound",
                                   "wet_tail_readback")
                   if _condition(compiled, key) is not None]
    elif during is not None:
        # QA-06 on its ``still`` branch: the audible window still has to sit in
        # one motion state, that state is simply "not moving".
        semantics = "during_audible_window"
        because = ["motion_during_event(moving=False)"]
    else:
        semantics = "unconstrained"
        because = []

    target_moves: bool | None = None
    if during is not None:
        target_moves = bool(during.detail.get("moving"))
    if after_sound is not None and "any_motion_in_interval" in after_sound.detail:
        target_moves = bool(after_sound.detail["any_motion_in_interval"])
    if target_moves is None:
        # QA-16 never says "it moved" in words: it asks whether the source got
        # nearer or farther after the sound, and states that as the
        # ``target_moved_after_sound`` planning knob on its distance condition.
        # Reading only the two detail flags left QA-16's target unconstrained
        # and its distance question unanswerable.
        for item in _conditions_of(compiled):
            value = item.planning.get("target_moved_after_sound")
            if value is not None:
                target_moves = bool(value)

    competitor = None
    # Role motion can be requested by a distance or motion condition; read the
    # shared planning field without requiring an extra readback predicate.
    for item in _conditions_of(compiled):
        value = item.planning.get("competitor_motion")
        if value is not None:
            competitor = str(value)
    if competitor is not None and competitor not in COMPETITOR_MOTION_VALUES:
        raise ConditionedMotionError(
            f"compiled competitor_motion {competitor!r} is not one of {COMPETITOR_MOTION_VALUES}")

    predicate = None
    for item in _conditions_of_kind(compiled, "answer_distinguishable"):
        if item.detail.get("predicate"):
            predicate = str(item.detail["predicate"])
    trend = _condition(compiled, "distance_net_change")

    # Whether the actors have to stand still while the sound plays is a
    # separate statement from where the movement goes, and only some recipes
    # make it.  ``QA-13`` and ``QA-17`` outside a core recipe do not: they
    # constrain the query moment, not the body, and forcing them still would
    # delete legal candidates.
    speech_motion = None
    for item in _conditions_of(compiled):
        value = item.planning.get("speech_motion")
        if value is not None:
            speech_motion = str(value)

    return {
        "qa_id": qa_id,
        "branch": branch,
        "semantics": semantics,
        "because": because,
        "target_moves": target_moves,
        "competitor_motion": competitor,
        "distinguish_predicate": predicate,
        "requires_measured_wet_tail": tail is not None,
        "requires_source_activity_measurement": bool(
            _condition(compiled, "source_activity_readback") is not None),
        "distance_trend_sign": None if trend is None else str(trend.detail.get("sign")),
        "distance_margin_m": None if trend is None else float(trend.detail.get("margin_m")),
        "stable_after_sound": _condition(compiled, "distance_stable_after_sound") is not None,
        "requires_anchor_role": _condition(compiled, "target_is_anchor") is not None,
        "speech_motion": speech_motion,
        "distractor_gate_applies": bool(
            qa_id is not None and qa_id not in gc.DISTRACTOR_GATE_EXEMPT_QA_IDS),
        "source": f"{gc.__name__}.CompiledConditions",
        "note": (
            "post_sound_query_only places the query after the measured wet tail and "
            "leaves the movement free to start inside it; only after_wet_tail moves the "
            "movement itself"
        ),
    }


def assert_static_camera(camera: Any) -> dict[str, Any]:
    """Refuse a moving camera, and refuse to answer without being told.

    Every question this solver serves is asked of a fixed rig, and the whole
    bearing and distance vocabulary is listener-relative.  A caller that states
    a camera gets it checked here rather than discovering the conflict after a
    render; a caller that states none is told so, instead of having ``static``
    assumed on its behalf.
    """

    if camera is None:
        return {"camera_motion": None, "state": STATE_EVIDENCE_MISSING,
                "reason": "no camera was stated, so the fixed-rig invariant is unchecked"}
    if not isinstance(camera, Mapping):
        raise ConditionedMotionError("camera must be a mapping")
    motion = camera.get("motion", camera.get("camera_motion"))
    if motion is None:
        return {"camera_motion": None, "state": STATE_EVIDENCE_MISSING,
                "reason": "the stated camera declares no motion mode"}
    if str(motion) != "static":
        raise ConditionedMotionError(
            f"these questions are asked of a fixed rig; this camera declares "
            f"{motion!r} motion")
    return {"camera_motion": "static", "state": STATE_AVAILABLE, "reason": ""}


def resolve_anchor(compiled: Any) -> dict[str, Any]:
    """Say which instance holds the anchor role, and refuse to guess.

    The anchor is a declared property of one entity instance.  It is not "the
    first programmed event", and taking the first event is how a non-anchor
    target came to be reported as conditioned when the sampler had never
    intersected its event mask with its moving mask.
    """

    subjects = tuple(getattr(compiled, "subjects", ()) or ())
    declared = [item for item in subjects if getattr(item, "is_anchor", False)]
    targets = [item for item in subjects if item.role == "target"]
    required = _condition(compiled, "target_is_anchor") is not None
    if declared:
        anchor = declared[0]
        return {
            "entity_instance_id": anchor.entity_instance_id,
            "state": STATE_AVAILABLE,
            "required_by_question": required,
            "declared_anchor_count": len(declared),
            "reason": "",
        }
    return {
        "entity_instance_id": None,
        "state": STATE_EVIDENCE_MISSING if required else STATE_NOT_APPLICABLE,
        "required_by_question": required,
        "declared_anchor_count": 0,
        "reason": (
            f"{[item.entity_instance_id for item in targets]} declare no anchor role; the "
            "sampler conditions only anchor indices, and the first programmed event is a "
            "different statement"
            if required else
            "this question does not require the target to hold the anchor role"
        ),
    }


def resolve_distance_trend_criterion(override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Use the catalog's criterion when it exists, and say where it came from.

    ``QA-15`` asks about a trend, and the shipped judge compares two endpoints.
    P08 owns the catalog definition; until it publishes one this returns the
    planning-side criterion above, marked with its own source so nobody reads
    it as the catalog's answer.
    """

    if override is not None:
        criterion = dict(DEFAULT_DISTANCE_TREND_CRITERION)
        criterion.update(dict(override))
        criterion["criterion_source"] = "caller_override"
        return criterion
    for location, attribute in DISTANCE_TREND_CRITERION_HOOKS:
        value = getattr(catalog, attribute, None)
        if value is None:
            continue
        resolved = value() if callable(value) else value
        if isinstance(resolved, Mapping):
            criterion = dict(DEFAULT_DISTANCE_TREND_CRITERION)
            criterion.update(dict(resolved))
            criterion["criterion_source"] = location
            return criterion
    criterion = dict(DEFAULT_DISTANCE_TREND_CRITERION)
    criterion["criterion_source"] = (
        f"{SOLVER_NAME}.DEFAULT_DISTANCE_TREND_CRITERION "
        "(planning default; the catalog publishes no criterion yet)"
    )
    return criterion


# --------------------------------------------------------------------------- solving


@dataclass(frozen=True)
class EventPlacement:
    """Where one instance's programmed sound sits inside the episode clock."""

    entity_instance_id: str
    sound_asset_id: str
    start_sample: int
    end_sample_exclusive: int
    audible_start_sample: int
    audible_end_sample_exclusive: int
    start_frame: int
    end_frame: int
    audible_start_frame: int
    audible_end_frame: int
    activity_basis: str
    internal_silence_kept_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_instance_id": self.entity_instance_id,
            "sound_asset_id": self.sound_asset_id,
            "start_sample": int(self.start_sample),
            "end_sample_exclusive": int(self.end_sample_exclusive),
            "planned_audible_interval_samples": [
                int(self.audible_start_sample), int(self.audible_end_sample_exclusive)],
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "audible_frame_window": [int(self.audible_start_frame), int(self.audible_end_frame)],
            "activity_basis": self.activity_basis,
            "internal_silence_kept_s": self.internal_silence_kept_s,
            "frame_window_note": (
                "start_frame/end_frame are what unified_catalog._stable_motion_window reads; "
                "the audible frame window is the measured sound inside them"
            ),
        }


@dataclass(frozen=True)
class MotionRequirement:
    """What one instance's body has to do, in frames, and why."""

    entity_instance_id: str
    role: str
    is_anchor: bool
    semantics: str
    must_move: bool
    moving_frames: tuple[int, int] | None
    permitted_moving_frames: tuple[int, int] | None
    still_frames: tuple[tuple[int, int], ...]
    locomotion_state: str
    locomotion_reason: str
    required_path_length_range_m: tuple[float, float] | None
    required_moving_steps: int | None
    motion_window_policy: str
    predicate: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_instance_id": self.entity_instance_id,
            "role": self.role,
            "is_anchor": bool(self.is_anchor),
            "semantics": self.semantics,
            "must_move": bool(self.must_move),
            "moving_frames": None if self.moving_frames is None else list(self.moving_frames),
            "permitted_moving_frames": (
                None if self.permitted_moving_frames is None
                else list(self.permitted_moving_frames)),
            "still_frames": [list(pair) for pair in self.still_frames],
            "locomotion_state": self.locomotion_state,
            "locomotion_reason": self.locomotion_reason,
            "required_path_length_range_m": (
                None if self.required_path_length_range_m is None
                else list(self.required_path_length_range_m)),
            "required_moving_steps": self.required_moving_steps,
            "motion_window_policy": self.motion_window_policy,
            "predicate": self.predicate,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MotionSolution:
    """One joint layout of sound and movement, or the reasons there is none."""

    qa_id: str
    branch: str | None
    status: str
    semantics: dict[str, Any]
    clock: EpisodeClock
    budget: MotionBudget
    placements: tuple[EventPlacement, ...]
    requirements: tuple[MotionRequirement, ...]
    query_window: dict[str, Any]
    anchor: dict[str, Any]
    rejections: tuple[dict[str, Any], ...]
    camera: dict[str, Any] = field(default_factory=dict)
    qualifications: tuple[dict[str, Any], ...] = ()
    distance_trend: dict[str, Any] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.status == "solved"

    def requirement_for(self, entity_instance_id: str) -> MotionRequirement | None:
        for item in self.requirements:
            if item.entity_instance_id == entity_instance_id:
                return item
        return None

    def placement_for(self, entity_instance_id: str) -> EventPlacement | None:
        for item in self.placements:
            if item.entity_instance_id == entity_instance_id:
                return item
        return None

    def sampler_profile(self) -> dict[str, Any]:
        """The knobs a planner can apply straight away, motion side only.

        Deliberately narrow: it carries what this solver decided and nothing
        it merely passed through.  In particular a question that says nothing
        about the body while the sound plays - ``QA-13``, and ``QA-17``
        outside a core recipe - gets no ``speech_motion`` here, because
        writing ``all_still`` for it would quietly forbid a legal candidate.
        """

        profile: dict[str, Any] = {}
        semantics = self.semantics["semantics"]
        target = next((item for item in self.requirements if item.role == "target"), None)
        if semantics == "during_audible_window" and target is not None:
            profile["speech_motion"] = "speaker_moving" if target.must_move else "all_still"
            profile["motion_window_placement"] = gc.MOTION_DURING_EVENT
        elif semantics == "after_wet_tail":
            profile["speech_motion"] = "all_still"
            profile["motion_window_placement"] = gc.MOTION_AFTER_TAIL
        elif semantics == "post_sound_query_only":
            profile["motion_window_placement"] = "free_after_anchor_event_end"
        if self.semantics.get("competitor_motion") is not None:
            profile["competitor_motion"] = self.semantics["competitor_motion"]
        if (semantics in {"post_sound_query_only", "after_wet_tail"}
                and self.semantics.get("target_moves") is not None):
            profile["target_moved_after_sound"] = bool(self.semantics["target_moves"])
        if self.semantics.get("distance_trend_sign") is not None:
            profile["distance_trend_during_event"] = self.branch
        return profile

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": solver_signature(),
            "qa_id": self.qa_id,
            "branch": self.branch,
            "status": self.status,
            "semantics": dict(self.semantics),
            "clock": self.clock.to_dict(),
            "budget": self.budget.to_dict(),
            "placements": [item.to_dict() for item in self.placements],
            "requirements": [item.to_dict() for item in self.requirements],
            "query_window": dict(self.query_window),
            "anchor": dict(self.anchor),
            "camera": dict(self.camera),
            "rejections": [dict(row) for row in self.rejections],
            "qualifications": [dict(row) for row in self.qualifications],
            "distance_trend": dict(self.distance_trend),
            "motion_sampler_profile": self.sampler_profile(),
        }


def _reject(code: str, subject: str | None, detail: str, **extra: Any) -> dict[str, Any]:
    if code not in REJECTION_CODES:
        raise ConditionedMotionError(f"unknown rejection code: {code!r}")
    row = {"code": code, "subject": subject, "detail": detail}
    row.update(extra)
    return row


def _registry_record(registry: Any, asset_id: str | None) -> Mapping[str, Any] | None:
    if registry is None or asset_id is None:
        return None
    assets = registry.get("assets") if isinstance(registry, Mapping) else None
    if not isinstance(assets, Sequence):
        return None
    for record in assets:
        if isinstance(record, Mapping) and record.get("asset_id") == asset_id:
            return record
    return None


def _locomotion(registry: Any, subject: Any) -> tuple[str, str, str | None]:
    """Resolve one subject's locomotion capability through the shared vocabulary."""

    record = _registry_record(registry, getattr(subject, "asset_id", None))
    if record is None:
        source_class = getattr(subject, "source_class", None)
        if source_class == "rigid_static_object":
            return (STATE_NOT_APPLICABLE,
                    "a registered device is a static source and never walks itself",
                    "device")
        return (STATE_EVIDENCE_MISSING,
                "no registry record resolves this instance's asset, so its locomotion "
                "capability is unknown", None)
    capability = locomotion_capability(record)
    return capability["state"], capability["reason"], source_family(record)


def _audio_span_budget(clock: EpisodeClock, budget: MotionBudget) -> dict[str, Any]:
    """Reuse the shared dry-program budget instead of restating the arithmetic."""

    return segment_budget(
        episode_s=clock.duration_s,
        reserve_tail_s=budget.reserve_tail_s,
        earliest_start_s=budget.earliest_start_s,
    )


def _place_event(
    *,
    entity_instance_id: str,
    sound: SoundCandidate,
    start_s: float,
    clock: EpisodeClock,
) -> EventPlacement:
    start_sample = int(round(start_s * clock.sample_rate_hz))
    end_sample = start_sample + int(sound.sample_count)
    audible_start = start_sample + int(sound.audible_start_sample)
    audible_end = start_sample + int(sound.audible_end_sample_exclusive)
    return EventPlacement(
        entity_instance_id=entity_instance_id,
        sound_asset_id=sound.sound_asset_id,
        start_sample=start_sample,
        end_sample_exclusive=end_sample,
        audible_start_sample=audible_start,
        audible_end_sample_exclusive=audible_end,
        start_frame=clock.frame_of_sample(start_sample),
        end_frame=min(clock.frame_count, clock.frame_ceil_of_sample(end_sample)),
        audible_start_frame=clock.frame_of_sample(audible_start),
        audible_end_frame=min(clock.frame_count, clock.frame_ceil_of_sample(audible_end)),
        activity_basis=sound.activity_basis,
        internal_silence_kept_s=sound.max_internal_silence_s,
    )


def _path_length_range(
    steps: int, clock: EpisodeClock, budget: MotionBudget
) -> tuple[float, float]:
    """How long a walkable path a run of ``steps`` moving frames needs.

    A frame reads as moving because the position at the *next* frame differs,
    so ``n`` moving frames consume ``n`` inter-frame steps and the walk occupies
    ``n + 1`` positions.  Getting this off by one produces a candidate whose
    final audible frame reads still and whose ``QA-06`` answer therefore
    refuses to be stable - which is exactly what the first run of this solver
    did.  Every step has to fall inside the declared speed band, so the length
    is bounded on both sides; returning the band rather than one number lets
    the route sampler keep choosing a real navigation path.
    """

    seconds = max(1, int(steps)) / float(clock.frame_rate_hz)
    low, high = budget.walk_speed_range_mps
    return (low * seconds, high * seconds)


def solve_motion_windows(
    compiled: Any,
    *,
    clock: EpisodeClock | Mapping[str, Any],
    budget: MotionBudget | None = None,
    sounds: Mapping[str, SoundCandidate | Mapping[str, Any]],
    registry: Any = None,
    camera: Mapping[str, Any] | None = None,
    event_start_s: Mapping[str, float] | None = None,
    other_event_windows_s: Sequence[Sequence[float]] = (),
    measured_wet_tail_s: Mapping[str, float] | float | None = None,
    available_path_length_m: Mapping[str, float] | None = None,
    distance_trend_criterion: Mapping[str, Any] | None = None,
) -> MotionSolution:
    """Lay one question's sound and movement out on the episode clock.

    ``compiled`` is one ``generation_conditions.CompiledConditions``.  ``sounds``
    maps each participating ``entity_instance_id`` to the prepared segment it
    will emit - the caller chooses which recording, records why, and passes it
    here; this function never reaches for the shortest candidate to make a
    layout fit.  ``event_start_s`` says where the caller's own event scheduler
    put each program, because ordering several audible windows against the
    minimum gap belongs to ``conditioned_sampler.schedule_legal_events`` and
    not to a second scheduler here; omitting it places every program at the
    budget's earliest start.  One instance gets one program here;
    ``other_event_windows_s`` carries every further ``[start_s, end_s)`` the
    caller already scheduled - including a second program on the same instance -
    so the post-sound window is closed by all of them and not only by the ones
    this call happens to place.  ``measured_wet_tail_s`` is absent during planning
    and present on the second pass, and the solution says which of the two it
    used.

    The result is a plan, never a verdict on a delivered episode.  Every
    refusal names one measurable cause, and a solution can be refused for
    several at once so the first cause does not hide the rest.
    """

    clock = clock if isinstance(clock, EpisodeClock) else EpisodeClock.from_mapping(clock)
    budget = budget or MotionBudget()
    camera_state = assert_static_camera(camera)
    meaning = motion_semantics(compiled)
    anchor = resolve_anchor(compiled)
    qa_id = str(getattr(compiled, "qa_id"))
    branch = getattr(compiled, "branch", None)
    subjects = tuple(getattr(compiled, "subjects", ()) or ())
    if not subjects:
        raise ConditionedMotionError("compiled conditions name no subjects")

    rejections: list[dict[str, Any]] = []
    qualifications: list[dict[str, Any]] = []
    span = _audio_span_budget(clock, budget)

    resolved_sounds: dict[str, SoundCandidate] = {}
    for key, value in dict(sounds).items():
        resolved_sounds[str(key)] = (
            value if isinstance(value, SoundCandidate) else SoundCandidate.from_pool_row(value))

    target = next((item for item in subjects if item.role == "target"), None)
    if target is None:
        raise ConditionedMotionError("compiled conditions name no target subject")
    if target.entity_instance_id not in resolved_sounds:
        raise ConditionedMotionError(
            f"no sound candidate was supplied for the target {target.entity_instance_id!r}")

    # ---- 1. the anchor event, and whether it fits in front of the reserved tail
    placements: list[EventPlacement] = []
    anchor_placement: EventPlacement | None = None
    for subject in subjects:
        sound = resolved_sounds.get(subject.entity_instance_id)
        if sound is None:
            continue
        if sound.sample_rate_hz != clock.sample_rate_hz:
            raise ConditionedMotionError(
                f"{sound.sound_asset_id!r} is at {sound.sample_rate_hz} Hz but the episode "
                f"clock runs at {clock.sample_rate_hz} Hz")
        if budget.max_clip_s is not None and sound.duration_s > budget.max_clip_s + 1e-9:
            rejections.append(_reject(
                "audio_does_not_fit_reserved_tail", subject.entity_instance_id,
                f"{sound.sound_asset_id!r} runs {sound.duration_s:.3f} s and the caller "
                f"declared a {budget.max_clip_s:.3f} s clip cap",
                sound_asset_id=sound.sound_asset_id))
            continue
        if sound.duration_s > span["max_single_program_span_s"] + 1e-9:
            rejections.append(_reject(
                "audio_does_not_fit_reserved_tail", subject.entity_instance_id,
                f"{sound.sound_asset_id!r} runs {sound.duration_s:.3f} s and only "
                f"{span['max_single_program_span_s']:.3f} s is left after the "
                f"{budget.reserve_tail_s:.3f} s reserved tail",
                sound_asset_id=sound.sound_asset_id,
                max_single_program_span_s=span["max_single_program_span_s"]))
            continue
        if (meaning["requires_source_activity_measurement"]
                and not sound.has_measured_activity):
            rejections.append(_reject(
                "audible_window_missing_activity_measurement", subject.entity_instance_id,
                f"{sound.sound_asset_id!r} carries only an event bounding box; this "
                "question compiles require_source_activity_measurement, and a guard-padded "
                "span is not a measured active interval",
                sound_asset_id=sound.sound_asset_id, activity_basis=sound.activity_basis))
            continue
        placement = _place_event(
            entity_instance_id=subject.entity_instance_id, sound=sound,
            start_s=float((event_start_s or {}).get(
                subject.entity_instance_id, budget.earliest_start_s)),
            clock=clock)
        if placement.end_frame > clock.frame_count:
            rejections.append(_reject(
                "audible_window_outside_clip", subject.entity_instance_id,
                f"the placed event ends at frame {placement.end_frame} of "
                f"{clock.frame_count}", sound_asset_id=sound.sound_asset_id))
            continue
        placements.append(placement)
        if subject.entity_instance_id == target.entity_instance_id:
            anchor_placement = placement
        if sound.max_internal_silence_s:
            qualifications.append({
                "subject": subject.entity_instance_id,
                "qualification": "audible_window_contains_natural_pauses",
                "max_internal_silence_s": sound.max_internal_silence_s,
                "detail": (
                    "the owner authorised natural pauses inside a sounding segment, so the "
                    "audible window spans them and the motion predicate is judged over the "
                    "whole span rather than over each active island"),
            })

    if anchor_placement is None:
        return MotionSolution(
            qa_id=qa_id, branch=branch, status="rejected", semantics=meaning, clock=clock,
            budget=budget, placements=tuple(placements), requirements=(),
            query_window={"status": "not_derived",
                          "reason": "the target's event could not be placed"},
            anchor=anchor, rejections=tuple(rejections), camera=camera_state,
            qualifications=tuple(qualifications))

    # ---- 2. the tail, planned or measured
    tail_end_s = _tail_end_for(anchor_placement, clock, budget, measured_wet_tail_s)
    query = _derive_query_window(qa_id, meaning, anchor_placement, clock, budget, tail_end_s,
                                 others=placements,
                                 other_windows_s=other_event_windows_s)
    if meaning["semantics"] in {"post_sound_query_only", "after_wet_tail"}:
        if query["status"] == "empty":
            rejections.append(_reject(
                "no_legal_query_frame_after_reserved_tail", target.entity_instance_id,
                query["reason"], **{k: query[k] for k in ("tail_end_s", "tail_basis")}))
        elif query["status"] == "not_displayable":
            rejections.append(_reject(
                "no_displayable_integer_query_window", target.entity_instance_id,
                query["reason"], **{k: query[k] for k in ("legal_frames", "tail_end_s")}))

    # ---- 3. who among the competitors carries the differing answer
    duty, duty_rejections = _competitor_duty(meaning, subjects, registry)
    rejections.extend(duty_rejections)

    # ---- 4. per-subject motion requirements
    requirements: list[MotionRequirement] = []
    for subject in subjects:
        requirement, subject_rejections = _requirement_for_subject(
            subject=subject, compiled=compiled, meaning=meaning, anchor=anchor,
            duty=duty.get(subject.entity_instance_id),
            placement=next((p for p in placements
                            if p.entity_instance_id == subject.entity_instance_id), None),
            anchor_placement=anchor_placement, clock=clock, budget=budget,
            registry=registry, tail_end_s=tail_end_s, query=query,
            available_path_length_m=(available_path_length_m or {}).get(
                subject.entity_instance_id))
        requirements.append(requirement)
        rejections.extend(subject_rejections)

    if anchor["state"] == STATE_EVIDENCE_MISSING:
        rejections.append(_reject(
            "anchor_role_not_declared", target.entity_instance_id, anchor["reason"]))

    if meaning["distractor_gate_applies"] and meaning["competitor_motion"] in (None, "any"):
        # Not a refusal: whether a branch compiles a distinguishability
        # condition is ``generation_conditions``' decision, and inventing one
        # here would be a second definition of the same requirement.  It is
        # recorded so a candidate that the catalog's gate later defers is not a
        # surprise, and so the missing compile has one named call site.
        qualifications.append({
            "subject": target.entity_instance_id,
            "qualification": "no_compiled_competitor_separation",
            "detail": (
                f"{qa_id}/{branch} is subject to unified_catalog._p8_apply_distractor_gate "
                "but compiles no answer_distinguishable condition, so this solver places no "
                "competitor motion; if the branch needs one it belongs in "
                "avengine/qa/generation_conditions.py beside the existing ones"),
        })

    # ---- 5. the competitor must not share the target's answer
    rejections.extend(_competitor_rejections(meaning, requirements))

    trend: dict[str, Any] = {}
    if meaning["distance_trend_sign"] is not None:
        trend = dict(resolve_distance_trend_criterion(distance_trend_criterion))
        trend["expected_sign"] = meaning["distance_trend_sign"]
        trend["judged_over_frames"] = [
            anchor_placement.start_frame, anchor_placement.end_frame]
        trend["verified_here"] = False
        trend["verified_by"] = f"{SOLVER_NAME}.verify_motion_candidate on a real trajectory"

    status = "rejected" if rejections else "solved"
    return MotionSolution(
        qa_id=qa_id, branch=branch, status=status, semantics=meaning, clock=clock,
        budget=budget, placements=tuple(placements), requirements=tuple(requirements),
        query_window=query, anchor=anchor, rejections=tuple(rejections),
        camera=camera_state, qualifications=tuple(qualifications), distance_trend=trend)


def _tail_end_for(
    placement: EventPlacement,
    clock: EpisodeClock,
    budget: MotionBudget,
    measured_wet_tail_s: Mapping[str, float] | float | None,
) -> dict[str, Any]:
    """The moment the anchor's reverberation stops, measured or reserved."""

    event_end_s = placement.end_sample_exclusive / float(clock.sample_rate_hz)
    if measured_wet_tail_s is None:
        return {
            "tail_end_s": event_end_s + budget.reserve_tail_s,
            "tail_basis": "planned_reserve_s",
            "reserve_tail_s": budget.reserve_tail_s,
            "event_end_s": event_end_s,
            "requires_recheck_after_measurement": True,
            "note": (
                "no render has happened, so the reserved tail stands in for the measured "
                "one; requery_after_measured_tail replaces it and reports whether the "
                "query frame moved"),
        }
    if isinstance(measured_wet_tail_s, Mapping):
        candidates = [float(v) for v in measured_wet_tail_s.values()]
        value = max(candidates) if candidates else event_end_s
    else:
        value = float(measured_wet_tail_s)
    if not math.isfinite(value):
        raise ConditionedMotionError("a measured wet tail end must be finite")
    return {
        "tail_end_s": value,
        "tail_basis": "measured_wet_tail_intervals",
        "reserve_tail_s": budget.reserve_tail_s,
        "event_end_s": event_end_s,
        "requires_recheck_after_measurement": False,
        "note": "measured from the delivered binaural readback",
    }


def _blocking_event_start_s(
    placement: EventPlacement,
    others: Sequence[EventPlacement],
    clock: EpisodeClock,
    other_windows_s: Sequence[Sequence[float]] = (),
) -> float | None:
    """The moment another programmed event closes this one's post-sound window.

    ``unified_catalog._silent_after`` refuses a query frame once some other
    event has started and outlasts the anchor, so the window does not run to
    the end of the clip whenever a second program follows.  On the retained
    episode this is the difference between the whole clip and eight frames,
    and a planner that ignored it would hand the question a moment the judge
    refuses.
    """

    anchor_end_s = placement.end_sample_exclusive / float(clock.sample_rate_hz)
    starts = [
        other.start_sample / float(clock.sample_rate_hz)
        for other in others
        if other.entity_instance_id != placement.entity_instance_id
        and other.end_sample_exclusive / float(clock.sample_rate_hz) > anchor_end_s
    ]
    anchor_start_s = placement.start_sample / float(clock.sample_rate_hz)
    for window in other_windows_s or ():
        start, end = float(window[0]), float(window[1])
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ConditionedMotionError("an other-event window must be finite")
        if abs(start - anchor_start_s) < 1e-9 and abs(end - anchor_end_s) < 1e-9:
            continue  # the anchor itself, restated by a caller listing every event
        if end > anchor_end_s:
            starts.append(start)
    return min(starts) if starts else None


def _derive_query_window(
    qa_id: str,
    meaning: Mapping[str, Any],
    placement: EventPlacement,
    clock: EpisodeClock,
    budget: MotionBudget,
    tail: Mapping[str, Any],
    others: Sequence[EventPlacement] = (),
    other_windows_s: Sequence[Sequence[float]] = (),
) -> dict[str, Any]:
    """Which frames the question may point at, and whether it can say so.

    A post-sound question needs two separate things and the retained batch
    failed the second one thirteen times: a frame that is provably after the
    tail, and a public interval that survives being quantized inward to whole
    seconds.  Both are reported, because "there is a legal frame but the
    question cannot state an interval" is a different refusal from "there is
    no legal frame".
    """

    if meaning["semantics"] not in {"post_sound_query_only", "after_wet_tail"}:
        return {
            "status": "not_required",
            "reason": f"{qa_id} does not ask about a moment after the sound",
            "tail_end_s": tail["tail_end_s"],
            "tail_basis": tail["tail_basis"],
            "requires_recheck_after_measurement": tail["requires_recheck_after_measurement"],
        }
    first = max(placement.end_frame, clock.first_frame_after(tail["tail_end_s"]))
    last = clock.frame_count - int(math.ceil(budget.end_hold_s * clock.frame_rate_hz))
    blocked_from_s = _blocking_event_start_s(placement, others, clock, other_windows_s)
    if blocked_from_s is not None:
        # A frame is legal while its own time is strictly before the other
        # event's start, which is the comparison _silent_after makes.
        last = min(last, int(math.ceil(blocked_from_s * clock.frame_rate_hz - 1e-9)))
    if first >= last:
        return {
            "status": "empty",
            "reason": (
                f"the anchor ends at frame {placement.end_frame} and its tail at "
                f"{tail['tail_end_s']:.3f} s, leaving no frame before {last} of "
                f"{clock.frame_count}"
                + ("" if blocked_from_s is None else
                   f"; another programmed event starts at {blocked_from_s:.3f} s and "
                   "outlasts this one")),
            "legal_frames": [],
            "blocking_event_start_s": blocked_from_s,
            "tail_end_s": tail["tail_end_s"],
            "tail_basis": tail["tail_basis"],
            "requires_recheck_after_measurement": tail["requires_recheck_after_measurement"],
        }
    display = gc.integer_second_window(
        first / clock.frame_rate_hz, (last - 1) / clock.frame_rate_hz,
        precision=budget.public_time_precision)
    if display is None:
        return {
            "status": "not_displayable",
            "reason": (
                f"frames [{first}, {last}) are legal but at precision "
                f"{budget.public_time_precision} the interval "
                f"[{first / clock.frame_rate_hz:.3f}, {(last - 1) / clock.frame_rate_hz:.3f}] s "
                "does not cross two display marks"),
            "legal_frames": [first, last],
            "blocking_event_start_s": blocked_from_s,
            "tail_end_s": tail["tail_end_s"],
            "tail_basis": tail["tail_basis"],
            "requires_recheck_after_measurement": tail["requires_recheck_after_measurement"],
        }
    return {
        "status": "available",
        "reason": "",
        "legal_frames": [first, last],
        "earliest_query_frame": first,
        "latest_query_frame_exclusive": last,
        "public_interval_s": [display[0], display[1]],
        "public_time_precision": budget.public_time_precision,
        "tail_end_s": tail["tail_end_s"],
        "tail_basis": tail["tail_basis"],
        "requires_recheck_after_measurement": tail["requires_recheck_after_measurement"],
        "blocking_event_start_s": blocked_from_s,
        "authority": gc.PUBLIC_WINDOW_AUTHORITY.get(qa_id, "unspecified"),
    }


COMPETITOR_DUTIES = ("moving", "still", "unconstrained", "unconstrained_incapable")


def _competitor_duty(
    meaning: Mapping[str, Any], subjects: Sequence[Any], registry: Any
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Decide which competitors carry the differing answer, over the whole set.

    ``unified_catalog._p8_apply_distractor_gate`` defers a candidate when
    *every* real entity shares the gold answer, so what a branch needs is at
    least one competitor whose answer differs - not every competitor.  Reading
    it as "every competitor" is what made a mixed group unusable: one
    registered device in the room cannot walk, and demanding that it walk threw
    away the legitimate human slot standing next to it.  A device that cannot
    move is therefore left unconstrained with that reason recorded, and the
    refusal fires only when no competitor in the room can move at all.

    ``still`` is applied to every competitor.  That is stronger than the gate
    needs, and it is what ``generation_conditions`` compiles, so it is left
    alone rather than relaxed here into a second definition.
    """

    wanted = meaning.get("competitor_motion")
    competitors = [item for item in subjects if item.role == "competitor"]
    duty: dict[str, str] = {}
    rejections: list[dict[str, Any]] = []
    if wanted in (None, "any"):
        return ({item.entity_instance_id: "unconstrained" for item in competitors}, rejections)
    if wanted == "still":
        return ({item.entity_instance_id: "still" for item in competitors}, rejections)
    capable: list[str] = []
    for item in competitors:
        state, reason, family = _locomotion(registry, item)
        if state == STATE_AVAILABLE:
            capable.append(item.entity_instance_id)
            # The catalog gate needs one competitor with a different answer.
            # Keeping every capable competitor moving over-constrains mixed
            # groups and makes extra distractors answerable only by accident.
            duty[item.entity_instance_id] = "unconstrained"
        else:
            duty[item.entity_instance_id] = "unconstrained_incapable"
    if not capable:
        rejections.append(_reject(
            "no_locomotion_capable_competitor", None,
            "this branch needs a competitor whose answer differs by moving, and none of "
            f"{[item.entity_instance_id for item in competitors]} has an available "
            "locomotion capability",
            competitors=[item.entity_instance_id for item in competitors]))
    else:
        # Subject order is the compiled order, so this selection is
        # deterministic while leaving other capable competitors free for
        # ordinary scene variation.
        duty[capable[0]] = "moving"
    return duty, rejections


def _merge_windows(windows: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Collapse touching still windows so a reader sees one interval, not three."""

    ordered = sorted((int(a), int(b)) for a, b in windows if int(b) > int(a))
    merged: list[list[int]] = []
    for first, last in ordered:
        if merged and first <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], last)
        else:
            merged.append([first, last])
    return tuple((int(a), int(b)) for a, b in merged)


def _choose_motion_window(
    permitted: tuple[int, int], steps: int, clock: EpisodeClock
) -> tuple[int, int] | None:
    """Pick the concrete run inside a permitted range, earliest first.

    A question that asks whether *any* movement happened does not name a
    window, so one has to be chosen.  Choosing the earliest run of the declared
    minimum length keeps the choice stated and reproducible; the full permitted
    range travels beside it so a route sampler that needs a different placement
    can take one without re-deriving the rule.
    """

    first, last = int(permitted[0]), min(int(permitted[1]), clock.frame_count)
    if last - first < int(steps):
        return None
    return (first, first + int(steps))


def _requirement_for_subject(
    *,
    subject: Any,
    compiled: Any,
    meaning: Mapping[str, Any],
    anchor: Mapping[str, Any],
    duty: str | None,
    placement: EventPlacement | None,
    anchor_placement: EventPlacement,
    clock: EpisodeClock,
    budget: MotionBudget,
    registry: Any,
    tail_end_s: Mapping[str, Any],
    query: Mapping[str, Any],
    available_path_length_m: float | None,
) -> tuple[MotionRequirement, list[dict[str, Any]]]:
    """Turn one subject's role and the question's meaning into frame intervals.

    Every window here is judged over the *anchor event's* frames, for every
    role.  ``unified_catalog`` evaluates the distractor gate by asking each
    actor the same question about the same anchor event, so comparing a
    competitor over its own shorter event would be a comparison of two
    different questions that happens to produce a number.
    """

    rejections: list[dict[str, Any]] = []
    instance_id = subject.entity_instance_id
    is_target = subject.role == "target"
    locomotion_state, locomotion_reason, family = _locomotion(registry, subject)
    semantics = meaning["semantics"]
    predicate = meaning["distinguish_predicate"]
    # The audible window every role is judged over.  A target that carries its
    # own placement uses it; a competitor is always judged over the anchor's.
    judged = placement if (is_target and placement is not None) else anchor_placement
    event_window = (int(judged.start_frame), int(judged.end_frame))

    def unconstrained(reason: str) -> tuple[MotionRequirement, list[dict[str, Any]]]:
        return (MotionRequirement(
            entity_instance_id=instance_id, role=subject.role,
            is_anchor=bool(getattr(subject, "is_anchor", False)),
            semantics="unconstrained", must_move=False, moving_frames=None,
            permitted_moving_frames=None, still_frames=(),
            locomotion_state=locomotion_state, locomotion_reason=locomotion_reason,
            required_path_length_range_m=None, required_moving_steps=None,
            motion_window_policy="none", predicate=predicate, reason=reason), rejections)

    if is_target:
        if meaning["target_moves"] is None and semantics != "during_audible_window":
            # QA-13 asks where the source was after the sound.  It says nothing
            # about whether the body moved, so nothing here places it.
            return unconstrained(
                "no compiled condition states whether this target moves; the question "
                "constrains the query moment, not the body")
        must_move = bool(meaning["target_moves"])
    else:
        if duty in (None, "unconstrained"):
            return unconstrained(
                "no compiled condition separates this competitor's motion from the "
                "target's, so it is left unconstrained rather than randomised")
        if duty == "unconstrained_incapable":
            return unconstrained(
                f"this branch asks a competitor to move and {instance_id} cannot: "
                f"{locomotion_reason}. Another competitor in this room carries the "
                "differing answer, and this instance stays available for every role that "
                "is not its own locomotion")
        must_move = duty == "moving"

    if is_target and must_move and locomotion_state != STATE_AVAILABLE:
        rejections.append(_reject(
            "target_cannot_self_locomote", instance_id,
            f"{instance_id} must move for this branch but its locomotion capability is "
            f"{locomotion_state}: {locomotion_reason}",
            source_family=family, locomotion_state=locomotion_state))

    moving_frames: tuple[int, int] | None = None
    permitted: tuple[int, int] | None = None
    steps: int | None = None
    policy = "none"
    still: list[tuple[int, int]] = []
    reason = ""

    if semantics == "during_audible_window":
        if must_move:
            moving_frames = event_window
            permitted = (0, int(clock.frame_count))
            steps = event_window[1] - event_window[0]
            policy = "complete_audible_window_of_the_anchor_event"
            reason = (
                "the complete audible window has to sit in one moving run, because "
                "unified_catalog._stable_motion_window refuses a window whose motion "
                "state changes on any frame of the event span")
        else:
            still.append(event_window)
            policy = "complete_audible_window_of_the_anchor_event"
            reason = "the complete audible window has to sit in one still run"
    elif semantics in {"post_sound_query_only", "after_wet_tail"}:
        if semantics == "after_wet_tail":
            open_at = int(math.ceil(tail_end_s["tail_end_s"] * clock.frame_rate_hz)) + 1
            placement_reason = (
                "the cross_time_state recipe opens the motion window at "
                "ceil(max(wet_tail_end_s) * fps) + 1, after the tail measured from the "
                "early binaural readback")
        else:
            open_at = int(anchor_placement.end_frame)
            placement_reason = (
                "movement may begin as soon as the anchor event ends, including while "
                "the measured wet tail is still audible; only the query moment has to "
                "be after that tail")
        last = clock.frame_count - int(math.ceil(budget.end_hold_s * clock.frame_rate_hz))
        permitted = (open_at, int(clock.frame_count))
        # Standing still while the sound plays is its own compiled statement.
        if meaning.get("speech_motion") == "all_still":
            still.append((0, min(open_at, clock.frame_count)))
        if must_move:
            steps = max(1, int(math.ceil(budget.minimum_motion_s * clock.frame_rate_hz)))
            chosen = _choose_motion_window((open_at, last), steps, clock)
            policy = "earliest_run_of_the_declared_minimum_motion_inside_the_permitted_range"
            if budget.minimum_motion_s <= 0:
                rejections.append(_reject(
                    "unsatisfiable_motion_semantics", instance_id,
                    "this branch needs movement after the sound but the caller declared "
                    "no minimum_motion_s, so the only window this solver could choose is "
                    "the single inter-frame step the predicate literally asks for; state "
                    "a duration rather than have one invented here",
                    minimum_motion_s=float(budget.minimum_motion_s),
                    required_moving_steps=steps))
            if chosen is None:
                rejections.append(_reject(
                    "motion_window_outside_clip", instance_id,
                    f"the permitted motion range [{open_at}, {last}) cannot hold the "
                    f"{steps} moving frames the declared minimum_motion_s of "
                    f"{budget.minimum_motion_s:.3f} s asks for",
                    permitted_moving_frames=[open_at, last], required_moving_steps=steps))
            else:
                moving_frames = chosen
            reason = placement_reason
        else:
            end = int(query.get("latest_query_frame_exclusive") or clock.frame_count)
            still.append((int(anchor_placement.end_frame), min(end, clock.frame_count)))
            policy = "anchor_end_through_the_latest_legal_query_frame"
            reason = (
                "this branch answers no, so the interval the question reads - from the "
                "anchor end frame through the query frame - has to stay still")
    else:
        reason = "no compiled condition places this instance's motion"

    path_range: tuple[float, float] | None = None
    if moving_frames is not None:
        first, last_frame = moving_frames
        if last_frame - first < 1:
            rejections.append(_reject(
                "motion_window_shorter_than_two_frames", instance_id,
                f"the motion window [{first}, {last_frame}) holds no frame, so no "
                "inter-frame displacement can exist",
                moving_frames=[first, last_frame]))
        if first < 0 or last_frame > clock.frame_count:
            rejections.append(_reject(
                "motion_window_outside_clip", instance_id,
                f"the motion window [{first}, {last_frame}) escapes the "
                f"{clock.frame_count} frame clip; the walk cannot be moved outside the "
                "Episode",
                moving_frames=[first, last_frame], frame_count=clock.frame_count))
        hold_first = clock.frame_count - int(math.ceil(budget.end_hold_s * clock.frame_rate_hz))
        if budget.end_hold_s > 0 and last_frame > hold_first:
            rejections.append(_reject(
                "motion_window_collides_with_end_hold", instance_id,
                f"the motion window ends at {last_frame} but the declared "
                f"{budget.end_hold_s:.3f} s end hold starts at frame {hold_first}",
                moving_frames=[first, last_frame], end_hold_first_frame=hold_first))
        if last_frame - first >= 1 and 0 <= first and last_frame <= clock.frame_count:
            steps = last_frame - first
            path_range = _path_length_range(steps, clock, budget)
            if (available_path_length_m is not None
                    and available_path_length_m + 1e-9 < path_range[0]):
                rejections.append(_reject(
                    "required_path_length_exceeds_available", instance_id,
                    f"the motion window needs at least {path_range[0]:.3f} m at the "
                    f"declared {budget.walk_speed_range_mps[0]:.3f} m/s floor, and the "
                    f"caller offers {available_path_length_m:.3f} m",
                    required_path_length_range_m=list(path_range),
                    available_path_length_m=float(available_path_length_m)))

    return (MotionRequirement(
        entity_instance_id=instance_id, role=subject.role,
        is_anchor=bool(getattr(subject, "is_anchor", False)), semantics=semantics,
        must_move=bool(must_move), moving_frames=moving_frames,
        permitted_moving_frames=permitted, still_frames=_merge_windows(still),
        locomotion_state=locomotion_state, locomotion_reason=locomotion_reason,
        required_path_length_range_m=path_range, required_moving_steps=steps,
        motion_window_policy=policy, predicate=predicate, reason=reason),
        rejections)


def _competitor_rejections(
    meaning: Mapping[str, Any], requirements: Sequence[MotionRequirement]
) -> list[dict[str, Any]]:
    """Refuse a layout in which the competitor would answer the same as the target.

    The compiled ``answer_distinguishable`` condition names the predicate the
    two roles must differ on.  Under ``speech_motion=speaker_moving`` the
    shipped sampler gives every non-anchor articulated actor a random moving
    flag, which is how 140 of 148 retained ``QA-06`` candidates ended up
    sharing one answer.  Deciding it here, at planning time, is the point.
    """

    wanted = meaning.get("competitor_motion")
    if wanted in (None, "any"):
        return []
    target = next((item for item in requirements if item.role == "target"), None)
    if target is None:
        return []
    rejections: list[dict[str, Any]] = []
    for item in requirements:
        if item.role != "competitor" or item.semantics == "unconstrained":
            continue
        if bool(item.must_move) == bool(target.must_move):
            rejections.append(_reject(
                "competitor_shares_target_answer", item.entity_instance_id,
                f"the compiled condition asks the competitor to be {wanted!r} on predicate "
                f"{meaning.get('distinguish_predicate')!r}, but this layout has both roles "
                f"{'moving' if target.must_move else 'still'}",
                predicate=meaning.get("distinguish_predicate"),
                target=target.entity_instance_id))
    return rejections


def requery_after_measured_tail(
    solution: MotionSolution,
    measured_wet_tail_s: Mapping[str, float] | float,
    *,
    query_frame: int | None = None,
) -> dict[str, Any]:
    """Replace the reserved tail with the measured one and say what changed.

    A plan is made before there is any reverberation to measure.  When the
    render comes back the tail is a real number, and it can be shorter or
    longer than the reservation.  If the earliest legal query frame moves, the
    answer truth computed at the old frame is a statement about a moment the
    question no longer asks about, so this reports ``truth_recompute_required``
    rather than quietly keeping the old value.
    """

    if not isinstance(solution, MotionSolution):
        raise ConditionedMotionError("requery_after_measured_tail needs a MotionSolution")
    target_ids = {item.entity_instance_id for item in solution.requirements
                  if item.role == "target"}
    anchor_placement = next(
        (item for item in solution.placements if item.entity_instance_id in target_ids),
        solution.placements[0] if solution.placements else None)
    if anchor_placement is None:
        raise ConditionedMotionError("this solution placed no event to requery")
    tail = _tail_end_for(anchor_placement, solution.clock, solution.budget, measured_wet_tail_s)
    updated = _derive_query_window(
        solution.qa_id, solution.semantics, anchor_placement, solution.clock,
        solution.budget, tail, others=solution.placements)
    before = solution.query_window
    moved = (before.get("earliest_query_frame") != updated.get("earliest_query_frame")
             or before.get("status") != updated.get("status"))
    result = {
        "qa_id": solution.qa_id,
        "branch": solution.branch,
        "planned_query_window": dict(before),
        "measured_query_window": dict(updated),
        "earliest_query_frame_moved": bool(moved),
        "tail_basis": tail["tail_basis"],
        "measured_tail_end_s": tail["tail_end_s"],
        "planned_tail_end_s": before.get("tail_end_s"),
        "still_satisfiable": updated.get("status") in {"available", "not_required"},
    }
    if query_frame is not None:
        frame = _exact_int(query_frame, "query_frame")
        legal = updated.get("legal_frames") or []
        inside = bool(legal) and legal[0] <= frame < legal[1]
        result["query_frame"] = frame
        result["query_frame_still_legal"] = inside
        result["truth_recompute_required"] = bool(moved or not inside)
        if not inside:
            result["reason"] = (
                f"frame {frame} is outside the measured legal window {legal}; the query "
                "moment changes, so the answer truth has to be recomputed at the new frame")
    else:
        result["truth_recompute_required"] = bool(moved)
    if solution.semantics["semantics"] == "after_wet_tail":
        first_motion = int(math.ceil(
            tail["tail_end_s"] * solution.clock.frame_rate_hz)) + 1
        result["measured_first_motion_frame"] = first_motion
        planned = [item.moving_frames[0] for item in solution.requirements
                   if item.moving_frames is not None]
        result["planned_first_motion_frame"] = min(planned) if planned else None
        result["motion_window_must_move"] = bool(
            planned and min(planned) != first_motion)
    return result


# --------------------------------------------------------------------------- trajectory


def moving_flags_from_path(
    path: Any,
    *,
    frame_rate_hz: float,
    threshold_mps: float = DEFAULT_MOVING_THRESHOLD_MPS,
    convention: str = DEFAULT_MOVING_FLAG_CONVENTION,
) -> np.ndarray:
    """Derive the per-frame ``moving`` track the facts and the judges read.

    The threshold and the difference convention both belong to the caller,
    because the two producers in this repository disagree about the final
    frame and the judge reads that frame like any other.
    """

    if convention not in MOVING_FLAG_CONVENTIONS:
        raise ConditionedMotionError(f"unknown moving flag convention: {convention!r}")
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        raise ConditionedMotionError("a path needs at least two frames of positions")
    fps = _positive(frame_rate_hz, "frame_rate_hz")
    delta = np.diff(points, axis=0)
    tail = delta[-1:] if convention == "forward_difference_hold_last" else np.zeros_like(delta[-1:])
    delta = np.concatenate([delta, tail], axis=0)
    return np.linalg.norm(delta, axis=1) * fps > float(threshold_mps)


def static_trajectory(point: Any, frame_count: int) -> np.ndarray:
    """Hold one position for the whole clip: a device, or a still competitor."""

    position = np.asarray(point, dtype=float).reshape(-1)
    if position.size != 3:
        raise ConditionedMotionError("a position must be three coordinates")
    return np.repeat(position[None], _exact_int(frame_count, "frame_count"), axis=0)


def build_motion_trajectory(
    *,
    polyline_m: Any,
    moving_frames: Sequence[int],
    clock: EpisodeClock,
    budget: MotionBudget,
    start_point_m: Any = None,
) -> dict[str, Any]:
    """Walk a caller-supplied navigation polyline so one frame window reads moving.

    ``moving_frames`` names the frames that have to read as moving, and a frame
    reads as moving because the *next* frame's position differs, so the walk
    occupies one position more than the window has frames.  The polyline comes
    from the route sampler, which owns the navigation mesh; what happens here
    is the time mapping.  The walk is resampled by arc length so every
    inter-frame step is the same distance, the actor holds its start before the
    window and its end after it, and the resulting speed is checked against the
    declared band rather than accepted because the geometry happened to fit.
    Nothing is time-stretched to make a speed legal.
    """

    poly = np.asarray(polyline_m, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 2 or poly.shape[1] != 3:
        raise ConditionedMotionError("a polyline needs at least two three-dimensional points")
    first, last = (_exact_int(moving_frames[0], "moving_frames[0]"),
                   _exact_int(moving_frames[1], "moving_frames[1]"))
    if not 0 <= first < last <= clock.frame_count:
        raise ConditionedMotionError(
            f"the motion window [{first}, {last}) escapes the {clock.frame_count} frame clip")
    steps = last - first
    points = steps + 1
    stop = first + points
    clipped = False
    if stop > clock.frame_count:
        # The run reaches the end of the clip.  Under the hold-last convention
        # the final frame inherits the previous difference and still reads as
        # moving, so one fewer position is enough; under the other convention
        # the final frame is still by definition and the window is impossible.
        if budget.moving_flag_convention != "forward_difference_hold_last":
            raise ConditionedMotionError(
                f"a motion window ending at frame {last} cannot read as moving under the "
                f"{budget.moving_flag_convention!r} convention, which makes the final "
                "frame still by definition")
        points, stop, clipped = clock.frame_count - first, clock.frame_count, True
    if points < 2:
        raise ConditionedMotionError("a motion window needs at least two positions")
    from avengine.routes.trajectory import resample_polyline_by_arc_length

    sampled = resample_polyline_by_arc_length(poly, points)
    start = np.asarray(start_point_m, dtype=float) if start_point_m is not None else sampled[0]
    path = np.repeat(np.asarray(start, dtype=float)[None], clock.frame_count, axis=0)
    path[first:stop] = sampled
    path[stop:] = sampled[-1]
    step_lengths = np.linalg.norm(np.diff(sampled, axis=0), axis=1)
    speeds = step_lengths * clock.frame_rate_hz
    walking = speeds[speeds > budget.moving_threshold_mps]
    low, high = budget.walk_speed_range_mps
    within = bool(len(walking)) and bool(
        walking.min() >= low - 1e-6 and walking.max() <= high + 1e-6)
    flags = moving_flags_from_path(
        path, frame_rate_hz=clock.frame_rate_hz,
        threshold_mps=budget.moving_threshold_mps,
        convention=budget.moving_flag_convention)
    return {
        "path_m": path,
        "moving_flags": flags,
        "moving_frames": [first, last],
        "position_frames": [first, stop],
        "clipped_to_clip_end": clipped,
        "required_moving_steps": steps,
        "path_length_m": float(step_lengths.sum()),
        "step_speeds_mps": [float(v) for v in speeds],
        "minimum_step_speed_mps": float(walking.min()) if len(walking) else 0.0,
        "maximum_step_speed_mps": float(walking.max()) if len(walking) else 0.0,
        "speed_within_declared_range": within,
        "declared_walk_speed_range_mps": [float(low), float(high)],
        "moving_frames_realized": [int(i) for i in np.flatnonzero(flags)][:512],
        "window_reads_moving": bool(flags[first:last].all()),
        "time_stretch_applied": False,
        "coordinate_interpolation": "arc_length_resample_of_caller_polyline",
        "camera_motion": "static",
    }


def distance_trend(
    positions_m: Any,
    listener_position_m: Any,
    frames: Sequence[int],
    *,
    criterion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure the signed listener-distance change over one frame window.

    Returns both readings on purpose: the endpoint difference the shipped judge
    compares, and whether the walk actually held that direction across the
    window.  A candidate that passes the first and fails the second is the one
    whose honest answer to "did it get nearer or farther" is "both".
    """

    resolved = dict(criterion or resolve_distance_trend_criterion())
    points = np.asarray(positions_m, dtype=float)
    listener = np.asarray(listener_position_m, dtype=float).reshape(-1)
    if listener.size != 3:
        raise ConditionedMotionError("a listener position must be three coordinates")
    first, last = int(frames[0]), int(frames[1])
    if not 0 <= first < last <= points.shape[0]:
        raise ConditionedMotionError(
            f"the trend window [{first}, {last}) escapes {points.shape[0]} frames of positions")
    window = points[first:last]
    distances = np.linalg.norm(window - listener, axis=1)
    if distances.size < int(resolved.get("min_frames", 2)):
        raise ConditionedMotionError("a distance trend needs at least two frames")
    net = float(distances[-1] - distances[0])
    steps = np.diff(distances)
    sign = "negative" if net < 0 else "positive"
    backward = steps if net < 0 else -steps
    # The largest excursion against the overall direction, measured as a run of
    # consecutive backward steps rather than as a single step, because ten
    # small steps the wrong way are the same problem as one large one.
    excursion = 0.0
    accumulated = 0.0
    for value in backward:
        accumulated = max(0.0, accumulated + float(value))
        excursion = max(excursion, accumulated)
    return {
        "frames": [first, last],
        "first_distance_m": float(distances[0]),
        "last_distance_m": float(distances[-1]),
        "net_change_m": net,
        "sign": sign,
        "abs_net_change_m": abs(net),
        "max_reversal_m": float(excursion),
        "monotone_within_tolerance": bool(
            excursion <= float(resolved.get("max_reversal_m", 0.0)) + 1e-9),
        "meets_margin": bool(abs(net) >= float(resolved.get("net_change_at_least_m", 0.0)) - 1e-9),
        "criterion": resolved,
        "endpoint_only_judge_would_accept": bool(
            abs(net) >= float(resolved.get("net_change_at_least_m", 0.0)) - 1e-9),
    }


# --------------------------------------------------------------------------- verifying


def post_sound_distance_stability(
    positions_m: Any,
    listener_position_m: Any,
    *,
    anchor_end_frame: int,
    windows: Sequence[Sequence[int]],
    frame_rate_hz: float,
    margin_m: float = gc.DISTANCE_MARGIN_M,
    precision: int = gc.DEFAULT_PUBLIC_TIME_PRECISION,
) -> dict[str, Any]:
    """Measure what ``QA-16`` actually claims, the way its readback measures it.

    ``generation_conditions._check_distance_stable_after_sound`` classifies
    every frame of the legal post-tail window as nearer, farther or
    under_margin against the distance at the anchor event's end frame, splits
    the window into maximal constant runs, and needs one run that holds a real
    trend *and* survives being published at the stated precision.  The two
    numbers that look like the answer and are not: a net change between two
    endpoints, and a frame count that never becomes a whole-second interval.

    The run splitting and the display quantisation come from that module's own
    helpers rather than from a second copy here, so a change to the judged
    definition cannot leave this planner agreeing with a rule nobody uses any
    more.  Both are private names today; promoting them is a one-line change in
    ``generation_conditions`` and is recorded as a cross-owner request rather
    than made here.
    """

    points = np.asarray(positions_m, dtype=float)
    listener = np.asarray(listener_position_m, dtype=float).reshape(-1)
    if listener.size != 3:
        raise ConditionedMotionError("a listener position must be three coordinates")
    anchor = _exact_int(anchor_end_frame, "anchor_end_frame")
    if not 0 <= anchor < points.shape[0]:
        raise ConditionedMotionError(
            f"the anchor end frame {anchor} escapes {points.shape[0]} frames of positions")
    reference = float(np.linalg.norm(points[anchor] - listener))
    margin = float(margin_m)

    def value_for(frame: int) -> Any:
        if not 0 <= frame < points.shape[0]:
            return None
        delta = float(np.linalg.norm(points[frame] - listener)) - reference
        if abs(delta) < margin:
            return "under_margin"
        return "nearer" if delta < 0 else "farther"

    runs = gc._stable_runs([list(map(int, window)) for window in windows], value_for)
    for run in runs:
        run["public_s"] = gc._displayable(run, float(frame_rate_hz), precision)
    usable = [run for run in runs
              if run["value"] in {"nearer", "farther"} and run["public_s"] is not None]
    return {
        "anchor_end_frame": anchor,
        "reference_distance_m": reference,
        "reference": "source distance at the anchor event end frame",
        "margin_m": margin,
        "public_time_precision": int(precision),
        "windows": [list(map(int, window)) for window in windows],
        "stable_runs": runs,
        "usable_runs": usable,
        "publishable": bool(usable),
        "selected_run": usable[0] if usable else None,
        "judge": "avengine/qa/generation_conditions.py:_check_distance_stable_after_sound",
    }


def feasible_sound_candidates(
    compiled: Any,
    *,
    clock: EpisodeClock | Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any] | SoundCandidate],
    budget: MotionBudget | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Solve once per candidate recording and report, without choosing one.

    Choosing which recording an episode emits is a decision that has to travel
    with its policy and its eligible count, so this returns every candidate's
    verdict and leaves the choice to the caller.  Silently keeping whichever
    clip is shortest is how a layout comes to "fit" without anyone deciding
    that it should.
    """

    clock = clock if isinstance(clock, EpisodeClock) else EpisodeClock.from_mapping(clock)
    target = next((item for item in getattr(compiled, "subjects", ())
                   if item.role == "target"), None)
    if target is None:
        raise ConditionedMotionError("compiled conditions name no target subject")
    others = dict(kwargs.pop("sounds", {}) or {})
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        resolved = (candidate if isinstance(candidate, SoundCandidate)
                    else SoundCandidate.from_pool_row(candidate))
        solution = solve_motion_windows(
            compiled, clock=clock, budget=budget,
            sounds={**others, target.entity_instance_id: resolved}, **kwargs)
        rows.append({
            "sound_asset_id": resolved.sound_asset_id,
            "duration_s": resolved.duration_s,
            "audible_duration_s": resolved.audible_duration_s,
            "activity_basis": resolved.activity_basis,
            "status": solution.status,
            "rejection_codes": [row["code"] for row in solution.rejections],
            "solution": solution,
        })
    feasible = [row for row in rows if row["status"] == "solved"]
    return {
        "qa_id": getattr(compiled, "qa_id", None),
        "branch": getattr(compiled, "branch", None),
        "considered": len(rows),
        "eligible_count": len(feasible),
        "candidates": rows,
        "selection": "none_made_here",
        "selection_note": (
            "the caller picks from the feasible rows and records its own policy; this "
            "function does not prefer the shortest, the longest or the first"
        ),
    }


def verify_motion_candidate(
    solution: MotionSolution,
    *,
    positions_m: Mapping[str, Any],
    root_positions_m: Mapping[str, Any] | None = None,
    listener_position_m: Any = None,
    moving_flags: Mapping[str, Any] | None = None,
    query_frame: int | None = None,
    distance_trend_criterion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check a real trajectory against the layout this solver produced.

    ``positions_m`` maps each entity instance to the per-frame positions the
    *distance* questions are judged on: pass the planned emitter positions,
    because an actor's root and its mouth are up to a body radius apart and
    ``QA-15`` judges a 0.2 m margin.  ``root_positions_m`` is the track the
    *locomotion* questions are judged on.  The two are separate on purpose: an
    emitter rides an offset that rotates with the body, so its per-frame step
    is not the walk speed, and checking a declared 0.5-0.8 m/s band against an
    emitter track reports speeds the actor never walked at.  Omitting the root
    track falls back to ``positions_m`` for both, and the report says so.

    This checks a plan, not a delivered episode.  Proving the same statements
    on rendered facts is ``generation_conditions.check_conditions``, which
    calls the catalog's own predicates; the two are kept apart so a green plan
    is never mistaken for native evidence.
    """

    if not isinstance(solution, MotionSolution):
        raise ConditionedMotionError("verify_motion_candidate needs a MotionSolution")
    clock, budget = solution.clock, solution.budget
    checks: list[dict[str, Any]] = []
    tracks: dict[str, np.ndarray] = {}

    def read(source: Mapping[str, Any], label: str) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for instance_id, raw in dict(source).items():
            points = np.asarray(raw, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ConditionedMotionError(
                    f"{instance_id!r} {label} must be a (frames, 3) array")
            if points.shape[0] != clock.frame_count:
                raise ConditionedMotionError(
                    f"{instance_id!r} has {points.shape[0]} frames of {label} but the "
                    f"clock declares {clock.frame_count}")
            result[str(instance_id)] = points
        return result

    tracks = read(positions_m, "positions")
    roots = read(root_positions_m, "root positions") if root_positions_m else dict(tracks)
    missing_roots = [key for key in tracks if key not in roots]
    if missing_roots:
        raise ConditionedMotionError(
            f"root positions are missing for {missing_roots}; pass every instance or none")

    flags: dict[str, np.ndarray] = {}
    for instance_id, points in roots.items():
        supplied = (moving_flags or {}).get(instance_id)
        if supplied is not None:
            value = np.asarray(supplied, dtype=bool)
            if value.shape[0] != clock.frame_count:
                raise ConditionedMotionError(
                    f"{instance_id!r} moving flags do not match the clock")
            flags[instance_id] = value
        else:
            flags[instance_id] = moving_flags_from_path(
                points, frame_rate_hz=clock.frame_rate_hz,
                threshold_mps=budget.moving_threshold_mps,
                convention=budget.moving_flag_convention)

    for requirement in solution.requirements:
        instance_id = requirement.entity_instance_id
        track = flags.get(instance_id)
        if track is None:
            checks.append(_check(instance_id, "trajectory_supplied", "not_run",
                                 "no trajectory was supplied for this instance"))
            continue
        for first, last in requirement.still_frames:
            first, last = max(0, int(first)), min(clock.frame_count, int(last))
            if last <= first:
                continue
            moved = [int(i) for i in range(first, last) if bool(track[i])]
            checks.append(_check(
                instance_id, "still_window", "pass" if not moved else "fail",
                f"frames [{first}, {last}) must hold still",
                measured={"moving_frames_inside": moved[:16],
                          "moving_frame_count": len(moved), "window": [first, last]}))
        if requirement.moving_frames is not None:
            first, last = requirement.moving_frames
            first, last = max(0, int(first)), min(clock.frame_count, int(last))
            still_inside = [int(i) for i in range(first, last) if not bool(track[i])]
            if requirement.semantics == "during_audible_window":
                # The judge refuses a window whose motion state changes at all,
                # so every frame of the event span has to be moving.
                checks.append(_check(
                    instance_id, "motion_during_event",
                    "pass" if not still_inside else "fail",
                    "every frame of the audible window must be moving, because "
                    "_stable_motion_window refuses a window whose state changes",
                    measured={"still_frames_inside": still_inside[:16],
                              "still_frame_count": len(still_inside),
                              "window": [first, last]}))
            else:
                any_moving = [int(i) for i in range(first, last) if bool(track[i])]
                checks.append(_check(
                    instance_id, "motion_after_sound",
                    "pass" if any_moving else "fail",
                    "the question asks whether any movement happened in this interval, not "
                    "whether every frame moved",
                    measured={"moving_frame_count": len(any_moving),
                              "first_moving_frame": any_moving[0] if any_moving else None,
                              "window": [first, last]}))
            stop = min(clock.frame_count, last + 1)
            speeds = np.linalg.norm(
                np.diff(roots[instance_id][first:stop], axis=0), axis=1) * clock.frame_rate_hz
            walking = speeds[speeds > budget.moving_threshold_mps]
            low, high = budget.walk_speed_range_mps
            within = bool(len(walking)) and bool(
                walking.min() >= low - 1e-6 and walking.max() <= high + 1e-6)
            checks.append(_check(
                instance_id, "walk_speed_within_declared_range",
                "pass" if within else "fail",
                f"every walking step must fall inside the declared [{low}, {high}] m/s band",
                measured={"minimum_step_speed_mps": float(walking.min()) if len(walking) else 0.0,
                          "maximum_step_speed_mps": float(walking.max()) if len(walking) else 0.0,
                          "walking_step_count": int(len(walking))}))

    # The competitor's answer, on the question's own predicate.
    target = next((item for item in solution.requirements if item.role == "target"), None)
    if target is not None and solution.semantics.get("competitor_motion") not in (None, "any"):
        for requirement in solution.requirements:
            if requirement.role != "competitor" or requirement.semantics == "unconstrained":
                continue
            if requirement.entity_instance_id not in flags:
                continue
            same = _predicate_value(
                solution, requirement, flags, tracks, listener_position_m,
                distance_trend_criterion) == _predicate_value(
                    solution, target, flags, tracks, listener_position_m,
                    distance_trend_criterion)
            checks.append(_check(
                requirement.entity_instance_id, "competitor_answer_differs",
                "fail" if same else "pass",
                f"the competitor must not answer the same as the target on "
                f"{solution.semantics.get('distinguish_predicate')!r}",
                measured={"predicate": solution.semantics.get("distinguish_predicate"),
                          "target": target.entity_instance_id}))

    trend_report: dict[str, Any] = {}
    if solution.semantics.get("distance_trend_sign") is not None and target is not None:
        if listener_position_m is None:
            checks.append(_check(
                target.entity_instance_id, "distance_trend", "not_run",
                "a distance trend needs the fixed listener position"))
        else:
            placement = solution.placement_for(target.entity_instance_id)
            window = ([placement.start_frame, placement.end_frame] if placement is not None
                      else [0, clock.frame_count])
            trend_report = distance_trend(
                tracks[target.entity_instance_id], listener_position_m, window,
                criterion=distance_trend_criterion or solution.distance_trend or None)
            expected = solution.semantics["distance_trend_sign"]
            ok = (trend_report["sign"] == expected and trend_report["meets_margin"]
                  and trend_report["monotone_within_tolerance"])
            checks.append(_check(
                target.entity_instance_id, "distance_trend", "pass" if ok else "fail",
                f"the listener distance must change {expected} by at least the margin and "
                "hold that direction across the audible window",
                measured=trend_report))

    if solution.semantics.get("stable_after_sound") and target is not None:
        if listener_position_m is None:
            checks.append(_check(target.entity_instance_id, "distance_stable_after_sound",
                                 "not_run", "needs the fixed listener position"))
        else:
            placement = solution.placement_for(target.entity_instance_id)
            anchor_end = placement.end_frame if placement is not None else 0
            legal = solution.query_window.get("legal_frames")
            windows = ([legal] if legal else [[anchor_end, clock.frame_count]])
            report = post_sound_distance_stability(
                tracks[target.entity_instance_id], listener_position_m,
                anchor_end_frame=anchor_end, windows=windows,
                frame_rate_hz=clock.frame_rate_hz,
                precision=budget.public_time_precision)
            checks.append(_check(
                target.entity_instance_id, "distance_stable_after_sound",
                "pass" if report["publishable"] else "fail",
                "one run inside the legal post-sound window has to hold a single trend at "
                f"the {report['margin_m']} m margin and still be publishable at precision "
                f"{report['public_time_precision']}",
                measured=report))

    if query_frame is not None and solution.query_window.get("status") == "available":
        legal = solution.query_window["legal_frames"]
        inside = legal[0] <= int(query_frame) < legal[1]
        checks.append(_check(
            None, "query_frame_after_wet_tail", "pass" if inside else "fail",
            f"the query frame must fall inside the legal window {legal}",
            measured={"query_frame": int(query_frame), "legal_frames": legal,
                      "tail_basis": solution.query_window.get("tail_basis")}))

    failed = [row for row in checks if row["status"] == "fail"]
    not_run = [row for row in checks if row["status"] == "not_run"]
    return {
        "solver": solver_signature(),
        "qa_id": solution.qa_id,
        "branch": solution.branch,
        "semantics": solution.semantics["semantics"],
        "status": "fail" if failed else ("incomplete" if not_run else "pass"),
        "checks": checks,
        "failed_checks": [row["check"] for row in failed],
        "not_run_checks": [row["check"] for row in not_run],
        "moving_flag_convention": budget.moving_flag_convention,
        "moving_threshold_mps": budget.moving_threshold_mps,
        "distance_position_basis": "caller_supplied positions_m",
        "locomotion_position_basis": (
            "caller_supplied root_positions_m" if root_positions_m
            else "positions_m reused, because no separate root track was supplied"),
        "distance_trend": trend_report,
        "solution_qualifications": [dict(row) for row in solution.qualifications],
        "claim_boundary": (
            "Checks a planned trajectory against the compiled conditions. It is not a "
            "render, not native evidence, and not a substitute for "
            "generation_conditions.check_conditions on delivered facts."
        ),
    }


def _check(subject: str | None, name: str, status: str, detail: str,
           measured: Any = None) -> dict[str, Any]:
    if status not in ("pass", "fail", "not_run"):
        raise ConditionedMotionError(f"unknown check status: {status!r}")
    row: dict[str, Any] = {"subject": subject, "check": name, "status": status,
                           "detail": detail}
    if measured is not None:
        row["measured"] = measured
    return row


def _predicate_value(
    solution: MotionSolution,
    requirement: MotionRequirement,
    flags: Mapping[str, np.ndarray],
    tracks: Mapping[str, np.ndarray],
    listener_position_m: Any,
    criterion: Mapping[str, Any] | None,
) -> Any:
    """Evaluate the question's own distinguishing predicate for one instance.

    Both roles are read over the anchor event's frames, because that is what
    ``unified_catalog._p8_apply_distractor_gate`` does: it asks every actor the
    same question about the same anchor event.  Reading a competitor over its
    own shorter event would compare two different questions and produce a
    number that looks like agreement.
    """

    predicate = solution.semantics.get("distinguish_predicate")
    track = flags[requirement.entity_instance_id]
    target_id = next((item.entity_instance_id for item in solution.requirements
                      if item.role == "target"), None)
    placement = solution.placement_for(target_id) if target_id else None
    if placement is None and solution.placements:
        placement = solution.placements[0]
    if placement is None:
        return None
    if predicate == "distance_trend":
        if listener_position_m is None:
            return None
        report = distance_trend(
            tracks[requirement.entity_instance_id], listener_position_m,
            [placement.start_frame, placement.end_frame], criterion=criterion)
        if not report["meets_margin"]:
            return "no_trend"
        return report["sign"]
    if predicate == "motion_after_sound":
        return any(bool(track[i]) for i in range(placement.end_frame, len(track)))
    return all(bool(track[i]) for i in range(placement.start_frame, placement.end_frame))


def solver_signature() -> dict[str, Any]:
    """What this solver is, what it accepts and what it leaves to its caller.

    ``conditioned_sampler`` reads this to declare the knobs it can now satisfy,
    so the capability declaration follows the implementation instead of being
    edited by hand when a gap is closed.
    """

    return {
        "name": SOLVER_NAME,
        "version": SOLVER_VERSION,
        "solves_knobs": [
            "competitor_motion",
            "target_moved_after_sound",
            "distance_trend_during_event",
            "motion_window_placement",
        ],
        "entry_points": {
            "solve_motion_windows": (
                "compiled, *, clock, budget=None, sounds, registry=None, "
                "event_start_s=None, other_event_windows_s=(), "
                "measured_wet_tail_s=None, available_path_length_m=None, "
                "distance_trend_criterion=None -> MotionSolution"),
            "build_motion_trajectory": (
                "*, polyline_m, moving_frames, clock, budget, start_point_m=None -> dict"),
            "verify_motion_candidate": (
                "solution, *, positions_m, root_positions_m=None, "
                "listener_position_m=None, moving_flags=None, query_frame=None, "
                "distance_trend_criterion=None -> dict"),
            "requery_after_measured_tail": (
                "solution, measured_wet_tail_s, *, query_frame=None -> dict"),
            "feasible_sound_candidates": (
                "compiled, *, clock, candidates, budget=None, **solve_kwargs -> dict"),
            "post_sound_distance_stability": (
                "positions_m, listener_position_m, *, anchor_end_frame, windows, "
                "frame_rate_hz, margin_m, precision -> dict"),
            "assert_static_camera": "camera -> dict",
        },
        "motion_semantics": list(MOTION_SEMANTICS),
        "left_to_caller": [
            "navigation mesh sampling and the route polyline",
            "camera placement, which stays static",
            "choosing which recording an episode emits, with its policy and eligible count",
            "proving conditions on delivered facts, which is check_conditions",
        ],
        "distance_trend_criterion": resolve_distance_trend_criterion(),
        "moving_flag_conventions": list(MOVING_FLAG_CONVENTIONS),
        "rejection_codes": list(REJECTION_CODES),
    }

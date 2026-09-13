"""Compile QA-01..QA-25 and their key branches into actual generation conditions.

The unified catalog states, per QA type, which requirement families a question
needs.  It does not say which scene a producer has to build, and it carries no
representation of the answer branch, so "was the source moving while it made
that sound" and "was it standing still" resolve to the same catalog row.  This
module closes that gap.  One QA type, one key branch, the named target and
competitor instances and one stated event selection compile to concrete
planning knobs for :mod:`avengine.rooms.conditioned_sampler`, and to the
matching readback predicates that prove the same statement on real facts.

Two requirement families in this catalog are opposites and must never be
merged into one "make it move" switch:

* ``QA-06`` on its ``moving`` branch and ``QA-15`` need the target's own sound
  activity to overlap real root displacement, so the complete audible window
  has to lie inside a moving run.  ``conditioned_sampler`` expresses that as
  ``speech_motion="speaker_moving"``, whose event mask is intersected with the
  anchor's per-frame moving mask.
* ``QA-13``, ``QA-16`` and ``QA-17`` need movement that starts strictly after
  the measured wet tail of the anchor event, so that same audible window has
  to lie inside a still run.  ``binding_group_motion`` expresses that by
  measuring the early binaural tail first and opening the motion window at
  ``ceil(max(wet_end_s) * fps) + 1``.

Asking for both on one entity instance and one event is refused here, at
planning time, before any capture or acoustic stage is scheduled.

Applicability uses the four-state vocabulary already shared by
``avengine.dataset.source_capabilities`` and ``avengine.qa.batch_coverage``:
an unsatisfiable request is reported as semantically inapplicable, as a
missing planner interface, or as missing evidence, and those three are never
collapsed into one another.  Aggregation over several candidates keeps a
per-candidate reason so one inapplicable device candidate cannot swallow a
legitimate human candidate in the same episode.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import copy
import math
from typing import Any, Callable

from avengine.dataset.source_capabilities import (
    CAPABILITY_STATES,
    STATE_AVAILABLE,
    STATE_EVIDENCE_MISSING,
    STATE_NOT_APPLICABLE,
    STATE_NOT_IMPLEMENTED,
    locomotion_capability,
    source_family,
)
from avengine.qa import unified_catalog as catalog
from avengine.qa.unified_catalog import QA_IDS, VISIBILITY_STATES, VISIBLE_STATES

APPLICABILITY_STATES = CAPABILITY_STATES

# Roles a compiled condition can attach to.  A role is a property of one entity
# instance inside one request; it is not a property of the asset, and the
# target's anchor role is not the same statement as "the first event".
ROLES = ("target", "competitor")

# The key answer branches.  Changing the branch changes the necessary
# conditions, which is exactly why they are named here rather than left for a
# generator to discover after an episode has already been rendered.
BRANCHES: dict[str, tuple[str, ...]] = {
    "QA-05": ("overlap", "disjoint"),
    "QA-06": ("moving", "still"),
    "QA-07": ("left", "right"),
    "QA-08": VISIBILITY_STATES,
    "QA-09": ("yes", "no"),
    "QA-15": ("nearer", "farther"),
    "QA-17": ("yes", "no"),
    "QA-20": ("visible_candidate", "none_of_them"),
    "QA-24": VISIBILITY_STATES,
    "QA-25": ("A", "V", "AV"),
}

CONDITION_KINDS = (
    "role_assignment",
    "event_selection",
    "required_modality",
    "candidate_domain",
    "appearance_reference",
    "speech_content",
    "sound_class_variety",
    "source_activity_readback",
    "wet_tail_readback",
    "motion_during_event",
    "motion_after_sound",
    "motion_window_placement",
    "distance_net_change",
    "distance_stable_after_sound",
    "distance_margin_at_time",
    "entry_transition",
    "occlusion_transition",
    "visibility_state",
    "visibility_coverage_complete",
    "occluder_identity",
    "legal_integer_query_window",
    "event_statistics_window",
    "bearing_reference",
    "answer_distinguishable",
)

# Which layer owns each planning key this module can emit.  Only the two
# capability layers describe a generator interface that a planner build either
# has or has not; the other layers are demands addressed to the request, the
# asset and sound selection, the measurement readback or the publication step,
# and reporting them as sampler gaps would paint a permanently red picture of
# work that is not the sampler's to do.
PLANNING_LAYERS = (
    "sampler_profile",
    "solver",
    "request",
    "selection",
    "readback",
    "publication",
    "derived",
)
CAPABILITY_LAYERS = ("sampler_profile", "solver")

PLANNING_KEY_LAYER: dict[str, str] = {
    # Keys that go into ``profile`` for conditioned_sampler.resolve_condition_profile.
    "anchor_count": "sampler_profile",
    "anchor_line_of_sight": "sampler_profile",
    "anchor_visibility": "sampler_profile",
    "competitor_motion": "sampler_profile",
    "competitor_visibility": "sampler_profile",
    "distance_range_m": "sampler_profile",
    "event_relation": "sampler_profile",
    "min_gap_between_audible_windows_s": "sampler_profile",
    "minimum_overlap_s": "sampler_profile",
    "reserve_tail_s": "sampler_profile",
    "retry_budget_within_profile": "sampler_profile",
    "separation_bin_deg": "sampler_profile",
    # Mapped so a declaration naming it hits the measured-trap check below
    # rather than the "this compiler never emits it" check; nothing emits it.
    "separation_floor_deg": "sampler_profile",
    "separation_target_policy": "sampler_profile",
    "speech_motion": "sampler_profile",
    # Which declared instance must own the earliest audible window. The sampler
    # enumerates every feasible actor order and used to draw one uniformly, so a
    # question whose gold answer is the first speaker was answered by whoever the
    # draw picked.
    "first_speaker_instance_id": "sampler_profile",
    # Goals a trajectory or visibility solver has to reach.
    "anchor_median_plane_offset_deg": "solver",
    "distance_trend_during_event": "solver",
    "entry_side": "solver",
    "pixel_occlusion_transition": "solver",
    "pixel_occlusion_partial_transition": "solver",
    "require_distance_margin_m": "solver",
    "registered_occluder_transition": "solver",
    "target_moved_after_sound": "solver",
    "visibility_transition": "solver",
    # Statements about the request rather than about a planner interface.
    "event_selector": "request",
    "event_statistics_window": "request",
    "min_entity_instances": "request",
    "required_modalities": "request",
    "target_event_ordinal": "request",
    # Constraints on which assets and sounds may be chosen.
    "distinct_sound_classes": "selection",
    "require_distinct_candidate_labels": "selection",
    "require_locomotion_capable_target": "selection",
    "require_transcript_bound_sound": "selection",
    "require_unique_reviewed_appearance": "selection",
    # Measurements that have to be produced and recorded.
    "require_complete_visibility_coverage": "readback",
    "require_measured_wet_tail": "readback",
    "require_source_activity_measurement": "readback",
    "require_wet_tail_complement_window": "readback",
    # How the question is published.
    "min_display_units": "publication",
    "public_time_precision": "publication",
    # Carried only so two opposite recipes collide during compilation.
    "motion_window_placement": "derived",
}

# Measured on 2026-09-10 against the real ``resolve_condition_profile``: the
# sampler takes this key without complaining and then drops it, because the
# default ``separation_bin_deg`` is a mapping and the floor is read from that
# mapping instead.  It is also the wrong quantity for a left/right question,
# which asks how far the source sits from the median plane rather than how far
# two sources sit from each other.  Do not map a question onto it.
ACCEPTED_BUT_IGNORED_KNOBS: dict[str, str] = {
    "separation_floor_deg": (
        "conditioned_sampler.resolve_condition_profile reads the floor from the "
        "separation_bin_deg mapping, so a profile-level separation_floor_deg is "
        "accepted and then ignored; it is inter-source separation, not a "
        "median-plane offset"
    ),
}

# Knobs this module needs that no planner accepts yet.  Each names the exact
# place that has to change, so the gap is actionable instead of decorative.
KNOB_GAPS: dict[str, tuple[str, str]] = {
}

# The knobs the checked-in sampler was measured to honour on 2026-09-10.  This
# is the conservative fallback used when no planner declares its capabilities,
# so behaviour is unchanged for a caller that passes nothing.
BASELINE_SAMPLER_KNOBS = frozenset({
    "anchor_count",
    "anchor_line_of_sight",
    "anchor_visibility",
    "competitor_visibility",
    "distance_range_m",
    "event_relation",
    "min_gap_between_audible_windows_s",
    "minimum_overlap_s",
    "reserve_tail_s",
    "retry_budget_within_profile",
    "separation_bin_deg",
    "separation_target_policy",
    "speech_motion",
})

# Room-family routes a declaration may name.  A solver can land on one route
# before the others, and a knob supported only elsewhere is still a gap here.
BACKENDS = ("spear_unreal", "spear_usd", "habitat")

# Marker a declaration sets when its planner genuinely dropped a baseline knob,
# so the measured fallback is replaced instead of merged.
_REPLACES_BASELINE = "replaces the measured baseline"

# Whether a missing knob makes its condition unreachable, or only unguaranteed.
#
# ``guarantee_required`` means the condition essentially never occurs unless a
# planner is told to produce it, so the compiled state has to say the interface
# is missing.  ``verify_only`` means the condition does occur without being
# asked for and is measurable afterwards, so planning may proceed and the
# missing knob is recorded as a yield cost rather than a blocker.  The basis is
# the retained 148-member batch and this module's own measurements, not a guess;
# an unlisted capability key is treated as ``guarantee_required`` so a new knob
# has to be classified deliberately.
GOAL_ENFORCEMENT: dict[str, tuple[str, str]] = {
    "visibility_transition": (
        "guarantee_required",
        "conditioned_sampler selects a camera and route whose body-proxy series "
        "contains the requested out_of_view to visible crossing",
    ),
    "pixel_occlusion_transition": (
        "guarantee_required",
        "conditioned_visibility screens the multi-sample pixel requirement and leaves "
        "native pixels as the acceptance authority",
    ),
    "distance_trend_during_event": (
        "guarantee_required",
        "conditioned_motion.solve_motion_windows supplies a signed distance-trend "
        "requirement and conditioned_sampler rejects poses that miss its margin",
    ),
    "competitor_motion": (
        "verify_only",
        "speech_motion=speaker_moving is accepted today, so the target's own state is "
        "requestable; only the competitor's state is left to chance, which cost 140 of "
        "148 QA-06 candidates a distinct answer and left both candidates moving in 61 of "
        "100 diagnostic seeds. That is a retry yield, not an unreachable condition",
    ),
    "anchor_median_plane_offset_deg": (
        "guarantee_required",
        "conditioned_sampler filters each camera candidate by the listener-relative "
        "anchor angle before scheduling the event",
    ),
    "entry_side": (
        "verify_only",
        "which side a crossing happens on is free once the crossing exists; the blocking "
        "capability is visibility_transition, and the side is read from the pixels",
    ),
    "require_distance_margin_m": (
        "verify_only",
        "QA-14 produced items in the retained batch: two placed sources usually differ by "
        "more than the judged 0.2 m, and the gap is measured at the query frame",
    ),
    "target_moved_after_sound": (
        "verify_only",
        "QA-16 produced items in the retained batch, and this module measured a "
        "distinguishable QA-17 yes answer on retained event_001, so post-tail movement "
        "occurs without a dedicated knob",
    ),
}
GOAL_ENFORCEMENTS = ("guarantee_required", "verify_only")


def goal_enforcement(key: str) -> tuple[str, str]:
    """Return whether a missing knob blocks its condition, and why we know."""

    if planning_layer(key) not in CAPABILITY_LAYERS:
        raise GenerationConditionError(
            f"{key!r} is not a planner interface key; it belongs to the "
            f"{planning_layer(key)!r} layer"
        )
    if key in ACCEPTED_BUT_IGNORED_KNOBS:
        return ("guarantee_required", ACCEPTED_BUT_IGNORED_KNOBS[key])
    return GOAL_ENFORCEMENT.get(
        key,
        (
            "guarantee_required",
            "not classified against a measured record yet, so it is treated as blocking "
            "until someone measures whether it occurs unrequested",
        ),
    )


# A declaration is about an interface, never about an outcome.  These words in a
# declaration mean the caller is trying to assert evidence or a met condition,
# which no planner can assert on the compiler's behalf.
_FORBIDDEN_DECLARATION_KEYS = frozenset({
    "state",
    "states",
    "available",
    "evidence",
    "condition_met",
    "conditions_met",
    "native_evidence",
    "produced",
    "verified",
})

_CATALOG_PATH = "avengine/qa/unified_catalog.py"

MOTION_DURING_EVENT = "during_target_audible_window"
MOTION_AFTER_TAIL = "after_measured_wet_tail"

# The judged margin in avengine/qa/unified_catalog.py QA-15 and QA-16.
DISTANCE_MARGIN_M = 0.2
# unified_catalog._event_start_side_window refuses an onset closer than this to
# the median plane, so a left/right question needs at least this offset.
QA04_SIDE_DEAD_ZONE_DEG = 5.0
# The minimum silence the catalog demands before an anchor event.
ANCHOR_PRE_SILENCE_S = 0.1
# Public intervals are whole seconds unless an episode declares otherwise. The
# planner keeps an additional 0.2 s of room so inward quantization still leaves
# one complete public second.
DEFAULT_PUBLIC_TIME_PRECISION = 0
PUBLIC_WINDOW_MIN_S = 1.2

# The event selection each question implies when a caller states none.  A
# question that counts over the clip is not a question about one audible window,
# so the default is part of the question rather than a planner convenience.
DEFAULT_EVENT_SELECTORS: dict[str, str] = {
    "QA-22": "whole_clip",
    "QA-23": "whole_clip",
}

# unified_catalog._P8_DISTRACTOR_GATE_EXEMPT: these four are not refused when
# every real entity shares the gold answer, so no distinguishability condition
# is compiled for them.  For every other type the gate is real and defers the
# affected answer form.
DISTRACTOR_GATE_EXEMPT_QA_IDS = frozenset({"QA-05", "QA-18", "QA-22", "QA-23"})

# How many named instances one question is actually about.  ``production_spec``
# derives ``target_instance_ids`` as every speaking instance, so a single-target
# question arrives naming several candidates.  Taking the first one and dropping
# the rest is the accounting mistake that hid a legitimate human candidate
# behind an inapplicable device candidate, so a single-target question fans out
# into one compiled result per candidate instead.
SUBJECT_SCOPES = ("single_target", "target_pair", "candidate_set")
QUESTION_SUBJECT_SCOPE: dict[str, str] = {
    "QA-03": "candidate_set",
    "QA-05": "target_pair",
    "QA-14": "target_pair",
    "QA-18": "candidate_set",
    "QA-20": "candidate_set",
    "QA-22": "candidate_set",
    "QA-23": "candidate_set",
}


def subject_scope(qa_id: str) -> str:
    """Return whether one question is about one instance, a pair or the set."""

    return QUESTION_SUBJECT_SCOPE.get(_qa_id(qa_id), "single_target")

TAIL_DEPENDENT_QA_IDS = frozenset({"QA-13", "QA-16", "QA-17"})
MOTION_DURING_EVENT_QA_IDS = frozenset({"QA-06", "QA-15"})
PUBLIC_WINDOW_QA_IDS = frozenset(
    {"QA-07", "QA-13", "QA-14", "QA-16", "QA-17", "QA-18", "QA-25"}
)
# Only these QA types ask about an entity's own locomotion, so only they refuse
# a device target.  A device still holds every other role.
SELF_MOTION_QA_IDS = frozenset({"QA-06", "QA-07", "QA-15", "QA-16", "QA-17"})
SPEECH_CONTENT_QA_IDS = frozenset({"QA-12"})


# The judges this module calls instead of writing its own approximations, with
# the parameters it passes.  ``avengine.qa.unified_catalog`` and
# ``avengine.qa.angular_questions`` belong to another owner, so a renamed or
# resignatured helper has to fail here rather than quietly change what a
# condition means.  Keep this in step with the calls below.
CATALOG_JUDGES: dict[str, tuple[str, ...]] = {
    "_anchor_pre_silence": ("facts", "event", "minimum_seconds"),
    "_azimuth": ("facts", "actor_id", "frame"),
    "_bound_events": ("facts",),
    "_canonical_qa_id": ("value",),
    "_derived_legal_query_windows": ("facts", "qa_id", "event"),
    "_display_time_bounds": ("facts", "window"),
    "_distance_at": ("facts", "actor_id", "frame"),
    "_entry_transition_window": ("facts", "actor_id", "entry_frame", "center", "dead_zone"),
    "_event_frame": ("event", "key"),
    "_event_start_side_window": ("facts", "event"),
    "_motion_at": ("facts", "actor_id", "frame"),
    "_occluder_ids": ("facts", "actor_id", "frame"),
    "_reviewed_appearances": ("facts",),
    "_silent_after": ("facts", "event", "query_frame"),
    "_source_activity_for_event": ("facts", "event_id"),
    "_source_activity_present": ("facts",),
    "_stable_motion_window": ("facts", "actor_id", "start_frame", "end_frame"),
    "_time_display_precision": ("facts",),
    "_visibility_is_complete": ("facts", "actor_id"),
}
ANGULAR_JUDGES: dict[str, tuple[str, ...]] = {
    "_hidden_motion_changed": ("facts", "actor_id", "anchor", "query"),
    "_visual_bearing": ("facts", "actor_id", "frame"),
    "_whole_second_frames": ("facts",),
}


class GenerationConditionError(ValueError):
    """A condition request is malformed or contradicts the unified catalog."""


# --------------------------------------------------------------------------- types


@dataclass(frozen=True)
class GeneratorCapabilities:
    """What one planner build declares it accepts, with where that came from.

    This describes an interface and nothing else.  It cannot say that native
    evidence exists and it cannot say a condition was met: those two are read
    back from a delivered episode by :func:`check_conditions`, never asserted
    by a caller.  A declaration that tries to carry a state or evidence word is
    refused, because "the caller wrote available" is not an implementation.

    ``knobs`` may be a plain set of names, or a mapping from name to the room
    family routes that support it, so a solver that lands on the UE route first
    still reports a gap for the Habitat route.
    """

    source: str
    version: str
    knobs: frozenset[str] = frozenset()
    knob_backends: dict[str, tuple[str, ...]] = field(default_factory=dict)
    backends: tuple[str, ...] = BACKENDS
    declared_at: str = ""
    # Which declaration contributed each knob, so a merged set still names the
    # planner that actually accepts it rather than a combined label.
    knob_source: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.source}@{self.version}"

    def declared_by(self, knob: str) -> str:
        return self.knob_source.get(knob, self.label)

    def supports(self, knob: str, *, backend: str | None = None) -> bool:
        if knob not in self.knobs:
            return False
        allowed = self.knob_backends.get(knob)
        if allowed is None or backend is None:
            return True
        return backend in allowed

    def unsupported_reason(self, knob: str, *, backend: str | None = None) -> str:
        if knob not in self.knobs:
            known = KNOB_GAPS.get(knob)
            if known is not None:
                location, detail = known
                return f"{location}: {detail}"
            return (
                f"no planner declares the {knob!r} knob; {self.label} declares "
                f"{sorted(self.knobs)}"
            )
        allowed = self.knob_backends.get(knob) or ()
        return (
            f"{self.declared_by(knob)} supports {knob!r} only on {list(allowed)}, not on "
            f"the {backend!r} route"
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "source": self.source,
            "version": self.version,
            "knobs": sorted(self.knobs),
            "backends": list(self.backends),
        }
        if self.knob_backends:
            value["knob_backends"] = {
                key: list(values) for key, values in sorted(self.knob_backends.items())
            }
        if self.declared_at:
            value["declared_at"] = self.declared_at
        if self.knob_source:
            value["knob_source"] = dict(sorted(self.knob_source.items()))
        return value


_BASELINE_LABEL = "avengine.rooms.conditioned_sampler@measured_baseline_20260910"
BASELINE_CAPABILITIES = GeneratorCapabilities(
    source="avengine.rooms.conditioned_sampler",
    version="measured_baseline_20260910",
    knobs=BASELINE_SAMPLER_KNOBS,
    declared_at=(
        "compiled-in fallback, measured by calling resolve_condition_profile with each "
        "knob set to a legal non-default value and checking the value survives"
    ),
    knob_source={knob: _BASELINE_LABEL for knob in sorted(BASELINE_SAMPLER_KNOBS)},
)


def resolve_generator_capabilities(value: Any = None) -> GeneratorCapabilities:
    """Resolve one or more planner declarations, on top of the measured fallback.

    ``value`` may be ``None`` (use the fallback alone, which keeps today's
    behaviour), a :class:`GeneratorCapabilities`, a mapping, any object or
    module exposing ``describe_generator_capabilities()``, or a sequence of
    those.  Declarations add up, because the planners are layered: the sampler
    declares its profile knobs and a trajectory or visibility solver declares
    its own goals, and neither should erase the other.  A declaration that
    genuinely removed a knob sets ``replaces_baseline`` so the measured fallback
    is dropped instead of merged.
    """

    if value is None:
        return BASELINE_CAPABILITIES
    if isinstance(value, GeneratorCapabilities):
        return value
    if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        return _merge_capabilities([_one_declaration(item) for item in value])
    return _merge_capabilities([_one_declaration(value)])


# Probe values used to measure a declaration rather than trust it.  A knob whose
# legal values the planner publishes is probed from that vocabulary, so a new
# value set needs no edit here.
_SCALAR_PROBES: dict[str, Any] = {
    "anchor_count": 1,
    "anchor_median_plane_offset_deg": 5.0,
    "distance_range_m": [1.0, 4.0],
    "min_gap_between_audible_windows_s": 0.75,
    "minimum_overlap_s": 0.4,
    "reserve_tail_s": 2.5,
    "retry_budget_within_profile": 50,
    "separation_bin_deg": [30.0, 60.0],
    "separation_target_policy": "any_legal_in_bin",
    # The measurement request declares no instances, so its actors carry the
    # positional slot names; source1 is a value the resolved profile can carry.
    "first_speaker_instance_id": "source1",
}


def measure_sampler_capabilities(
    registry: Mapping[str, Any],
    *,
    sampler: Any = None,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure which profile keys a sampler build really reads and honours.

    A declaration says what a planner accepts; this checks it.  For every knob
    the planner declares, a legal non-default value is pushed through
    ``resolve_condition_profile`` and the resolved profile is read back, so a
    knob that is advertised but dropped shows up as ``accepted_but_ignored``
    rather than being taken on trust.  Pure CPU: it resolves a profile and
    renders nothing.
    """

    if sampler is None:
        from avengine.rooms import conditioned_sampler as sampler
    resolve = getattr(sampler, "resolve_condition_profile", None)
    if not callable(resolve):
        raise GenerationConditionError(
            f"{getattr(sampler, '__name__', sampler)!r} has no resolve_condition_profile "
            f"to measure"
        )
    # This planner's own declaration, not the merged set: measuring the merged
    # set would probe the baseline's knobs against a planner that never claimed
    # them and report a shrug as a failure.
    declared = _one_declaration(sampler)
    enum_values = getattr(sampler, "ENUM_KNOB_VALUES", {}) or {}
    # A knob a solver consumes is still a knob the profile has to carry, and the
    # planner publishes its legal values in the same shape.  Reading only
    # ENUM_KNOB_VALUES reported every solver knob as unprobeable, so a knob that
    # was advertised and dropped would have looked the same as one that works.
    solver_values = getattr(sampler, "SOLVER_KNOB_VALUES", {}) or {}
    base = dict(
        request
        or {
            "seed": 7,
            "camera": {"motion": "static"},
            "entities": {"total_count": 2, "silent_count": 0, "min_articulated_count": 2},
            "profile": {},
        }
    )
    baseline = resolve(copy.deepcopy(base), registry)
    outcomes: dict[str, dict[str, Any]] = {}
    for knob in sorted(declared.knobs):
        if knob in enum_values or knob in solver_values:
            published = enum_values.get(knob) or solver_values.get(knob)
            choices = [value for value in published if value != baseline.get(knob)]
            probe = choices[0] if choices else baseline.get(knob)
        elif knob in _SCALAR_PROBES:
            probe = _SCALAR_PROBES[knob]
        else:
            outcomes[knob] = {
                "outcome": "not_probed",
                "reason": "this planner declares the knob but publishes no legal values "
                "for it, so the declaration cannot be checked; publish it in "
                "ENUM_KNOB_VALUES or give it a scalar probe",
            }
            continue
        attempt = copy.deepcopy(base)
        attempt.setdefault("profile", {})[knob] = probe
        try:
            resolved = resolve(attempt, registry)
        except Exception as error:  # the planner refuses its own declared value
            outcomes[knob] = {
                "outcome": "rejected",
                "probe": probe,
                "error": f"{type(error).__name__}: {error}",
            }
            continue
        if knob not in resolved:
            outcomes[knob] = {"outcome": "not_read", "probe": probe}
            continue
        got = resolved[knob]
        carried = got == probe or (
            isinstance(probe, (list, tuple)) and list(got or []) == list(probe)
        )
        if carried:
            outcomes[knob] = {"outcome": "honoured", "probe": probe}
        elif got == baseline.get(knob):
            outcomes[knob] = {"outcome": "accepted_but_ignored", "probe": probe, "resolved": got}
        else:
            outcomes[knob] = {"outcome": "honoured_transformed", "probe": probe, "resolved": got}
    good = {"honoured", "honoured_transformed"}
    unverified = sorted(knob for knob, row in outcomes.items() if row["outcome"] not in good)
    return {
        "declared": declared.to_dict(),
        "outcomes": outcomes,
        "verified_knobs": sorted(knob for knob, row in outcomes.items() if row["outcome"] in good),
        "unverified_knobs": unverified,
        "agrees_with_declaration": not unverified,
        "note": "a declared knob that is not honoured here is advertised but not read; "
        "the declaration is the claim and this is the check",
    }


def _merge_capabilities(
    declarations: Sequence[GeneratorCapabilities],
) -> GeneratorCapabilities:
    replaces = any(item.declared_at == _REPLACES_BASELINE for item in declarations)
    parts = list(declarations) if replaces else [BASELINE_CAPABILITIES, *declarations]
    if len(parts) == 1:
        return parts[0]
    knobs: set[str] = set()
    knob_backends: dict[str, tuple[str, ...]] = {}
    knob_source: dict[str, str] = {}
    routes: set[str] = set()
    for part in parts:
        knobs |= set(part.knobs)
        knob_backends.update(part.knob_backends)
        knob_source.update(part.knob_source or {knob: part.label for knob in part.knobs})
        routes |= set(part.backends)
    # A planner that supersedes the measured baseline for the same source is
    # listed once, so a merged label stays readable.
    labels: list[tuple[str, str]] = []
    for part in parts:
        if part.source in {source for source, _ in labels}:
            labels = [
                (source, part.version if source == part.source else version)
                for source, version in labels
            ]
            continue
        labels.append((part.source, part.version))
    return GeneratorCapabilities(
        source=" + ".join(source for source, _ in labels),
        version=" + ".join(version for _, version in labels),
        knobs=frozenset(knobs),
        knob_backends=knob_backends,
        backends=tuple(sorted(routes)),
        declared_at="merged declarations, listed per knob in knob_source",
        knob_source=knob_source,
    )


def _one_declaration(value: Any) -> GeneratorCapabilities:
    if isinstance(value, GeneratorCapabilities):
        return value
    describe = getattr(value, "describe_generator_capabilities", None)
    if callable(describe):
        return _one_declaration(describe())
    if not isinstance(value, Mapping):
        raise GenerationConditionError(
            "generator capabilities must be a mapping, a GeneratorCapabilities, or an "
            "object exposing describe_generator_capabilities()"
        )
    data = dict(value)
    forbidden = sorted(set(data) & _FORBIDDEN_DECLARATION_KEYS)
    if forbidden:
        raise GenerationConditionError(
            f"a capability declaration describes an interface, not an outcome; remove "
            f"{forbidden}. Whether native evidence exists and whether a condition was met "
            f"are read back from a delivered episode, not declared here"
        )
    known = {
        "source",
        "version",
        "knobs",
        "knob_backends",
        "backends",
        "declared_at",
        "replaces_baseline",
        # Provenance a merged declaration writes out; accepted back so a saved
        # condition set round-trips without losing which planner owns a knob.
        "knob_source",
    }
    unknown = sorted(set(data) - known)
    if unknown:
        raise GenerationConditionError(
            f"unknown capability declaration keys {unknown}; expected {sorted(known)}"
        )
    source = data.get("source")
    version = data.get("version")
    if not isinstance(source, str) or not source.strip():
        raise GenerationConditionError("a capability declaration must name its source")
    if not isinstance(version, str) or not version.strip():
        raise GenerationConditionError(
            f"{source} must state a capability version so a stale declaration is visible"
        )
    raw_knobs = data.get("knobs") or ()
    knob_backends: dict[str, tuple[str, ...]] = {}
    if isinstance(raw_knobs, Mapping):
        names = list(raw_knobs)
        for name, routes in raw_knobs.items():
            if routes is None:
                continue
            if isinstance(routes, str) or not isinstance(routes, Sequence):
                raise GenerationConditionError(
                    f"{source} declares {name!r} with a route list, not {routes!r}"
                )
            knob_backends[str(name)] = tuple(str(route) for route in routes)
    elif isinstance(raw_knobs, Sequence) and not isinstance(raw_knobs, (str, bytes)):
        names = [str(name) for name in raw_knobs]
    elif isinstance(raw_knobs, (set, frozenset)):
        names = [str(name) for name in raw_knobs]
    else:
        raise GenerationConditionError(f"{source} must declare knobs as a list or a mapping")
    names = [str(name) for name in names]
    ignored = sorted(name for name in names if name in ACCEPTED_BUT_IGNORED_KNOBS)
    if ignored:
        raise GenerationConditionError(
            f"{source} declares support for {ignored}, which the checked-in sampler was "
            f"measured to accept and then ignore: {ACCEPTED_BUT_IGNORED_KNOBS[ignored[0]]}"
        )
    unmapped = sorted(name for name in names if name not in PLANNING_KEY_LAYER)
    if unmapped:
        raise GenerationConditionError(
            f"{source} declares knobs this compiler never emits: {unmapped}; the compiler "
            f"only asks for {sorted(PLANNING_KEY_LAYER)}"
        )
    wrong_layer = sorted(
        name for name in names if PLANNING_KEY_LAYER[name] not in CAPABILITY_LAYERS
    )
    if wrong_layer:
        raise GenerationConditionError(
            f"{source} declares support for keys that are not a planner interface: "
            f"{wrong_layer}. Those belong to the "
            f"{sorted({PLANNING_KEY_LAYER[name] for name in wrong_layer})} layer"
        )
    routes = data.get("backends")
    if routes is None:
        resolved_routes = BACKENDS
    elif isinstance(routes, str) or not isinstance(routes, (Sequence, set, frozenset)):
        raise GenerationConditionError(f"{source} must declare backends as a list")
    else:
        resolved_routes = tuple(str(route) for route in routes)
    for name, allowed in knob_backends.items():
        outside = sorted(set(allowed) - set(resolved_routes))
        if outside:
            raise GenerationConditionError(
                f"{source} restricts {name!r} to routes it does not declare: {outside}"
            )
    declared_at = data.get("declared_at") or ""
    if not isinstance(declared_at, str):
        raise GenerationConditionError(f"{source} declared_at must be text")
    if data.get("replaces_baseline"):
        declared_at = _REPLACES_BASELINE
    label = f"{source.strip()}@{version.strip()}"
    supplied = data.get("knob_source")
    if supplied is not None and not isinstance(supplied, Mapping):
        raise GenerationConditionError(f"{source} knob_source must be a mapping")
    knob_source = {knob: label for knob in sorted(names)}
    for knob, owner in (supplied or {}).items():
        if knob in knob_source:
            knob_source[str(knob)] = str(owner)
    return GeneratorCapabilities(
        source=source.strip(),
        version=version.strip(),
        knobs=frozenset(names),
        knob_backends=knob_backends,
        backends=resolved_routes,
        declared_at=declared_at,
        knob_source=knob_source,
    )


@dataclass(frozen=True)
class ConditionSubject:
    """One entity instance in one role, with the five identities kept apart.

    ``entity_instance_id`` is the identity a role and a motion requirement
    attach to.  ``asset_id`` says which registered asset the instance realizes,
    and two instances may share it.  ``sound_asset_id``, ``event_id`` and
    ``sound_identity_id`` describe the emission rather than the body, and a
    question about one of them is not a question about the instance.
    """

    entity_instance_id: str
    role: str
    asset_id: str | None = None
    sound_asset_id: str | None = None
    event_id: str | None = None
    sound_identity_id: str | None = None
    source_class: str | None = None
    is_anchor: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.entity_instance_id, str) or not self.entity_instance_id:
            raise GenerationConditionError("entity_instance_id must be a non-empty string")
        if self.role not in ROLES:
            raise GenerationConditionError(f"role must be one of {ROLES}: {self.role!r}")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "entity_instance_id": self.entity_instance_id,
            "role": self.role,
            "is_anchor": self.is_anchor,
        }
        for key in ("asset_id", "sound_asset_id", "event_id", "sound_identity_id", "source_class"):
            if getattr(self, key) is not None:
                value[key] = getattr(self, key)
        return value


@dataclass(frozen=True)
class Condition:
    """One named requirement, the knob that satisfies it and the proof it needs."""

    key: str
    kind: str
    subject: str | None
    detail: dict[str, Any] = field(default_factory=dict)
    planning: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    state: str = STATE_AVAILABLE
    reason: str = ""
    # Set when the condition asks whether a legal query frame exists anywhere in
    # the clip, rather than what holds at one already-chosen frame.
    search_frames: bool = False
    # Which declaration cleared this condition's planner knobs, if any.  Present
    # so a reader can tell "a planner says it accepts this" apart from "this was
    # actually planned, rendered and measured".
    support: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in CONDITION_KINDS:
            raise GenerationConditionError(f"unknown condition kind: {self.kind!r}")
        if self.state not in APPLICABILITY_STATES:
            raise GenerationConditionError(f"unknown applicability state: {self.state!r}")

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"key": self.key, "kind": self.kind, "state": self.state}
        if self.subject is not None:
            value["subject"] = self.subject
        for name in ("detail", "planning", "evidence"):
            payload = getattr(self, name)
            if payload:
                value[name] = dict(payload)
        if self.reason:
            value["reason"] = self.reason
        if self.search_frames:
            value["search_frames"] = True
        if self.support:
            value["support"] = dict(self.support)
        return value


@dataclass(frozen=True)
class CompiledConditions:
    """Everything one (qa_id, branch, target selection) needs, and its verdict."""

    qa_id: str
    branch: str | None
    subjects: tuple[ConditionSubject, ...]
    event: dict[str, Any]
    conditions: tuple[Condition, ...]
    state: str
    reason: str
    conflicts: tuple[dict[str, Any], ...] = ()
    required_modalities: tuple[str, ...] = ()
    capabilities: GeneratorCapabilities = BASELINE_CAPABILITIES
    backend: str | None = None
    public_time_precision: int = DEFAULT_PUBLIC_TIME_PRECISION
    task_family: str | None = None

    @property
    def target_instance_ids(self) -> tuple[str, ...]:
        return tuple(s.entity_instance_id for s in self.subjects if s.role == "target")

    def planning_conditions(self) -> tuple[Condition, ...]:
        """Conditions a planner has to satisfy before a capture is worth running."""

        return tuple(item for item in self.conditions if item.planning)

    def evidence_conditions(self) -> tuple[Condition, ...]:
        """Conditions a readback has to prove on the delivered facts."""

        return tuple(item for item in self.conditions if item.evidence)

    def sampler_profile(self) -> dict[str, Any]:
        """The conditioned-sampler profile keys implied by these conditions.

        Every key whose owning layer is ``sampler_profile`` is carried, so a key
        the sampler really accepts is not dropped on the way out.  Keys owned by
        another layer are available from :meth:`planning_by_layer`.
        """

        profile: dict[str, Any] = {}
        for item in self.conditions:
            for knob, value in item.planning.items():
                if value is None:
                    continue
                if planning_layer(knob) == "sampler_profile":
                    profile[knob] = value
        return profile

    def planning_by_layer(self) -> dict[str, dict[str, Any]]:
        """Group every planning key by the layer that has to act on it."""

        grouped: dict[str, dict[str, Any]] = {}
        for item in self.conditions:
            for knob, value in item.planning.items():
                if value is None:
                    continue
                grouped.setdefault(planning_layer(knob), {})[knob] = value
        return {layer: grouped[layer] for layer in PLANNING_LAYERS if layer in grouped}

    def solver_goals(self) -> dict[str, Any]:
        """The goals a trajectory or visibility solver has to reach."""

        return self.planning_by_layer().get("solver", {})

    def gaps(self) -> tuple[dict[str, Any], ...]:
        """Every condition that is not currently satisfiable, with its reason."""

        return tuple(
            {
                "key": item.key,
                "kind": item.kind,
                "subject": item.subject,
                "state": item.state,
                "reason": item.reason,
            }
            for item in self.conditions
            if item.state != STATE_AVAILABLE
        )

    def state_layers(self) -> dict[str, Any]:
        """Keep three different questions apart in the report.

        ``planning_support`` says whether some planner declares the knobs this
        condition set needs.  ``native_evidence`` and ``condition_met`` are only
        answerable by reading a delivered episode, so this compiler reports them
        as not checked and :func:`check_conditions` fills them in.
        """

        return {
            "planning_support": self.state,
            "planning_support_declared_by": (
                f"{self.capabilities.source}@{self.capabilities.version}"
            ),
            "native_evidence": "not_checked",
            "condition_met": "not_checked",
            "note": "a declaration is an interface claim; it is not evidence that an "
            "episode was planned, rendered or measured",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "qa_id": self.qa_id,
            "branch": self.branch,
            "state": self.state,
            "reason": self.reason,
            "task_family": self.task_family,
            "backend": self.backend,
            "public_time_precision": self.public_time_precision,
            "subjects": [s.to_dict() for s in self.subjects],
            "event": dict(self.event),
            "required_modalities": list(self.required_modalities),
            "conditions": [item.to_dict() for item in self.conditions],
            "sampler_profile": self.sampler_profile(),
            "planning_by_layer": self.planning_by_layer(),
            "solver_goals": self.solver_goals(),
            "capabilities": self.capabilities.to_dict(),
            "state_layers": self.state_layers(),
            "conflicts": [dict(row) for row in self.conflicts],
            "gaps": [dict(row) for row in self.gaps()],
        }


# --------------------------------------------------------------------------- helpers


def branches_for(qa_id: str) -> tuple[str, ...]:
    """Return the key answer branches of one QA type, empty when it has none."""

    return BRANCHES.get(_qa_id(qa_id), ())


def _qa_id(value: Any) -> str:
    try:
        resolved = catalog._canonical_qa_id(value)
    except catalog.UnifiedQAError as error:
        raise GenerationConditionError(str(error)) from error
    return resolved


def _check_branch(qa_id: str, branch: Any) -> str | None:
    allowed = BRANCHES.get(qa_id, ())
    if branch is None:
        return None
    if not allowed:
        raise GenerationConditionError(f"{qa_id} has no key branches; got {branch!r}")
    if branch not in allowed:
        raise GenerationConditionError(f"{qa_id} branch must be one of {allowed}: {branch!r}")
    return str(branch)


def integer_second_window(
    start_s: float, end_s: float, *, precision: int = DEFAULT_PUBLIC_TIME_PRECISION
) -> tuple[float, float] | None:
    """Quantize a proven interval inward the way the catalog publishes it.

    This mirrors ``unified_catalog._display_time_bounds``: the public bounds are
    pulled inside the proven interval, so at whole-second precision an interval
    has to cross two second marks before it can be stated at all.  Thirteen of
    the retained QA-17 candidates failed for exactly this reason, which is a
    planning condition rather than a rendering accident.
    """

    if not isinstance(precision, int) or isinstance(precision, bool) or not 0 <= precision <= 9:
        raise GenerationConditionError("public time precision must be an integer from 0 through 9")
    if not math.isfinite(start_s) or not math.isfinite(end_s):
        raise GenerationConditionError("interval bounds must be finite")
    scale = float(10**precision)
    lower = math.ceil(start_s * scale - 1.0e-9)
    upper = math.floor(end_s * scale + 1.0e-9)
    if upper <= lower:
        return None
    return lower / scale, upper / scale


def _subject_ids(subjects: Sequence[ConditionSubject], role: str) -> tuple[str, ...]:
    return tuple(s.entity_instance_id for s in subjects if s.role == role)


def _instance_records(instances: Any) -> dict[str, dict[str, Any]]:
    """Accept P01 ``EntityInstanceSpec`` objects or plain mappings alike."""

    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(instances or ()):
        if hasattr(item, "to_dict"):
            row = dict(item.to_dict())
        elif isinstance(item, Mapping):
            row = dict(item)
        else:
            raise GenerationConditionError(f"instances[{index}] must be a mapping or spec")
        key = row.get("instance_id") or row.get("entity_instance_id")
        if not isinstance(key, str) or not key:
            raise GenerationConditionError(f"instances[{index}] needs an instance_id")
        result[key] = row
    return result


def _registry_record(registry: Any, asset_id: str | None) -> Mapping[str, Any] | None:
    if asset_id is None or not isinstance(registry, Mapping):
        return None
    assets = registry.get("assets")
    if not isinstance(assets, Sequence):
        return None
    for row in assets:
        if isinstance(row, Mapping) and str(row.get("asset_id")) == asset_id:
            return row
    return None


def _locomotion_state(record: Mapping[str, Any] | None, source_class: str | None) -> tuple[str, str]:
    """Resolve whether this subject may be asked about its own locomotion."""

    if record is not None:
        capability = locomotion_capability(record)
        if capability["state"] == STATE_AVAILABLE:
            return STATE_AVAILABLE, ""
        return capability["state"], capability["reason"]
    if source_class == "rigid_static_object":
        return (
            STATE_NOT_APPLICABLE,
            "a registered static device never walks by itself, so it cannot be a self-motion target",
        )
    if source_class in {"articulated_human", "articulated_animal"}:
        return STATE_AVAILABLE, ""
    return (
        STATE_EVIDENCE_MISSING,
        "no registry record and no source_class, so locomotion cannot be resolved",
    )


def planning_layer(key: str) -> str:
    """Return which layer owns one planning key."""

    try:
        return PLANNING_KEY_LAYER[key]
    except KeyError:
        raise GenerationConditionError(
            f"the planning key {key!r} has no owning layer; add it to PLANNING_KEY_LAYER "
            f"so it cannot be silently mistaken for a sampler gap"
        ) from None


def _apply_capability_states(
    conditions: Sequence[Condition],
    capabilities: GeneratorCapabilities,
    backend: str | None,
) -> list[Condition]:
    """Set each condition's planning state from the planner's own declaration.

    A condition already blocked for a semantic reason keeps that reason: a
    device that cannot walk stays inapplicable whatever a sampler declares.
    """

    resolved: list[Condition] = []
    for item in conditions:
        interface_keys = [
            key for key in item.planning if planning_layer(key) in CAPABILITY_LAYERS
        ]
        if item.state != STATE_AVAILABLE or not interface_keys:
            resolved.append(item)
            continue
        missing = [
            key for key in interface_keys if not capabilities.supports(key, backend=backend)
        ]
        blocking = [key for key in missing if goal_enforcement(key)[0] == "guarantee_required"]
        unguaranteed = [key for key in missing if key not in blocking]
        support: dict[str, Any] = {
            "knobs": sorted(interface_keys),
            "declared_by": capabilities.source,
            "version": capabilities.version,
            "backend": backend,
            "means": "this planner accepts these knobs; it is not evidence that anything "
            "was planned, rendered or measured",
        }
        support["declared_by_knob"] = {
            key: capabilities.declared_by(key)
            for key in sorted(interface_keys)
            if capabilities.supports(key, backend=backend)
        }
        if unguaranteed:
            support["not_guaranteed"] = {
                key: goal_enforcement(key)[1] for key in sorted(unguaranteed)
            }
            support["not_guaranteed_effect"] = (
                "planning may proceed and the readback decides; expect retries rather "
                "than a refusal"
            )
        if blocking:
            gap = "; ".join(
                f"{capabilities.unsupported_reason(key, backend=backend)} "
                f"[{goal_enforcement(key)[1]}]"
                for key in blocking
            )
            resolved.append(
                replace(
                    item,
                    state=STATE_NOT_IMPLEMENTED,
                    reason=f"{gap}; {item.reason}" if item.reason else gap,
                    support=support,
                )
            )
            continue
        resolved.append(replace(item, support=support))
    return resolved


# --------------------------------------------------------------------------- builders


def _modality_condition(qa_id: str, branch: str | None) -> Condition:
    requirements = catalog.get_requirements(qa_id)
    modalities = tuple(requirements["potential_requirements"].get("required_modalities") or ())
    subsets = requirements.get("subsets")
    if branch and isinstance(subsets, Mapping) and branch in subsets:
        modalities = tuple(subsets[branch].get("required_modalities") or modalities)
    return Condition(
        key="required_modalities",
        kind="required_modality",
        subject=None,
        detail={"modalities": list(modalities)},
        planning={"required_modalities": list(modalities)},
        evidence={"delivered_modalities_include": list(modalities)},
    )


def _domain_condition(qa_id: str) -> Condition:
    requirements = catalog.get_requirements(qa_id)["potential_requirements"]
    minimum = int(requirements.get("min_entities") or 1)
    return Condition(
        key="candidate_domain",
        kind="candidate_domain",
        subject=None,
        detail={"min_entities": minimum},
        planning={"min_entity_instances": minimum},
        evidence={"distinct_labelled_candidates_at_least": minimum},
    )


def _event_condition(qa_id: str, selector: Mapping[str, Any], target: str | None) -> Condition:
    kind = str(selector.get("kind", "target_audible_window"))
    detail = {"selector": dict(selector), "target_instance_id": target}
    reason = ""
    state = STATE_AVAILABLE
    if qa_id in {"QA-22", "QA-23"} and kind != "whole_clip":
        state = STATE_NOT_APPLICABLE
        reason = f"{qa_id} counts over the whole clip; a {kind} selector states a different question"
    if qa_id in TAIL_DEPENDENT_QA_IDS and kind == "whole_clip":
        state = STATE_NOT_APPLICABLE
        reason = f"{qa_id} anchors on one event; a whole_clip selector has no anchor to be after"
    if kind == "target_audible_window":
        detail["note"] = (
            "the audible window of the named target instance; this is not the first "
            "programmed event, and an ordinal has to be written down when the question "
            "means a specific ordinal"
        )
    return Condition(
        key="event_selection",
        kind="event_selection",
        subject=target,
        detail=detail,
        planning={"event_selector": dict(selector)},
        evidence={"bound_event_matches_selector": True},
        state=state,
        reason=reason,
    )


def _activity_condition(subject: str | None) -> Condition:
    return Condition(
        key="source_activity_readback",
        kind="source_activity_readback",
        subject=subject,
        detail={
            "note": "a nonzero sample or a guard-padded event span is not proof that a "
            "source was sounding throughout; the measured active interval is the authority"
        },
        planning={"require_source_activity_measurement": True},
        evidence={"facts_key": "source_activity_intervals_samples", "must_cover_event": True},
    )


def _tail_condition(subject: str | None) -> Condition:
    return Condition(
        key="wet_tail_readback",
        kind="wet_tail_readback",
        subject=subject,
        detail={"anchor_pre_silence_s": ANCHOR_PRE_SILENCE_S},
        planning={"require_measured_wet_tail": True},
        evidence={"facts_key": "audio.wet_tail_intervals", "per_event": True},
    )


# Which measurement owns the legal query interval of each question.  Checking a
# post-sound window for a question whose interval comes from an entry transition
# would report a number about the wrong thing.
PUBLIC_WINDOW_AUTHORITY: dict[str, str] = {
    "QA-13": "derived_post_sound_window",
    "QA-16": "derived_post_sound_window",
    "QA-17": "derived_post_sound_window",
    "QA-18": "wet_tail_complement_window",
    "QA-07": "entry_transition_window",
    "QA-14": "stable_distance_window",
    "QA-25": "whole_second_query_frame",
}


def _public_window_condition(qa_id: str, precision: int) -> Condition:
    authority = PUBLIC_WINDOW_AUTHORITY.get(qa_id, "unspecified")
    return Condition(
        key="legal_integer_query_window",
        kind="legal_integer_query_window",
        subject=None,
        detail={
            "public_time_precision": precision,
            "window_authority": authority,
            "qa_id": qa_id,
            "minimum_interval_s": PUBLIC_WINDOW_MIN_S,
            "planning_margin_s": 0.2,
            "note": "the public bounds are quantized inward, so the proven interval has "
            "to cross two display marks before the question can state it",
        },
        planning={"public_time_precision": precision, "min_display_units": 1},
        evidence={"display_bounds_exist": True},
    )


def _motion_during_condition(
    subject: ConditionSubject,
    *,
    moving: bool,
    qa_id: str,
    distinguish_predicate: str = "motion_during_event",
) -> list[Condition]:
    """The target's whole audible window has to sit in one motion state."""

    knob_value = "speaker_moving" if moving else "all_still"
    conditions = [
        Condition(
            key="motion_during_event",
            kind="motion_during_event",
            subject=subject.entity_instance_id,
            detail={
                "moving": moving,
                "stable_over": "complete_audible_window",
                "judge": "avengine/qa/unified_catalog.py:_stable_motion_window",
                "moving_threshold_mps": 0.05,
            },
            planning={"speech_motion": knob_value},
            evidence={
                "actors_key": "moving",
                "constant_over_event_frames": True,
                "expected_value": moving,
            },
        ),
    ]
    if moving:
        # Only the moving branch needs displacement inside the audible window.
        # Demanding a placement for the still branch would contradict itself.
        conditions.append(
            Condition(
                key="motion_window_placement",
                kind="motion_window_placement",
                subject=subject.entity_instance_id,
                detail={"placement": MOTION_DURING_EVENT},
                planning={"motion_window_placement": MOTION_DURING_EVENT},
                evidence={"motion_run_intersects_event_frames": True},
            )
        )
        # The anchor is the only index whose event mask is intersected with its
        # own moving mask, so a non-anchor target is not actually conditioned.
        conditions.append(
            Condition(
                key="target_is_anchor",
                kind="role_assignment",
                subject=subject.entity_instance_id,
                detail={
                    "requires_anchor_role": True,
                    "why": "conditioned_sampler intersects the event mask with the moving "
                    "mask only for anchor indices",
                },
                planning={"anchor_count": 1},
                evidence={"anchor_role_recorded": True},
                state=STATE_AVAILABLE if subject.is_anchor else STATE_EVIDENCE_MISSING,
                reason=""
                if subject.is_anchor
                else f"{subject.entity_instance_id} is not declared as an anchor, so its "
                "audible window is not constrained to its moving run",
            )
        )
    # The distractor gate reads both branches, so both branches need a
    # competitor whose answer differs.  Only the moving branch used to state
    # one, which is why the still branch kept losing every form to
    # ``distractors_equal_gold``: a still target beside still competitors gives
    # every candidate the same answer.
    conditions.append(
        Condition(
            key="answer_distinguishable",
            kind="answer_distinguishable",
            subject=subject.entity_instance_id,
            detail={
                "predicate": distinguish_predicate,
                "must_differ_from_role": "competitor",
                "judge": f"{_CATALOG_PATH}:_p8_apply_distractor_gate",
                "why": "in the retained batch 137 of 148 QA-06 candidates and 125 of 148 "
                "QA-17 candidates were deferred as distractors_equal_gold; the moving "
                "branch stated a still competitor, the still branch stated nothing at all",
                "requires_every_competitor_articulated": not moving,
            },
            planning={"competitor_motion": "still" if moving else "moving"},
            evidence={"competitor_predicate_differs": True},
        )
    )
    return conditions


def _post_tail_motion_conditions(
    subject: ConditionSubject, qa_id: str, task_family: str | None = None
) -> list[Condition]:
    """What a post-sound question needs, and what its core recipe adds.

    The question itself only needs a query moment that is provably after the
    event and after that event's measured wet tail; movement inside the tail
    still counts towards the answer, because the answer interval starts at the
    anchor end frame.  A ``cross_time_state`` core member is stricter, because
    the recipe captures a static early pass, measures the tail from the real
    binaural readback and only then opens the motion window.  Those are two
    different statements and only the second one places the motion.
    """

    conditions = [
        _tail_condition(subject.entity_instance_id),
        Condition(
            key="post_sound_silent_window",
            kind="motion_after_sound",
            subject=subject.entity_instance_id,
            detail={
                "query_after_event_end": True,
                "no_other_event_overlaps": True,
                "query_after_measured_wet_tail": True,
                "judge": f"{_CATALOG_PATH}:_silent_after",
            },
            planning={"min_gap_between_audible_windows_s": 0.5},
            evidence={"derived_legal_window_nonempty": True},
        ),
    ]
    if task_family == "cross_time_state":
        conditions.append(
            Condition(
                key="motion_window_placement",
                kind="motion_window_placement",
                subject=subject.entity_instance_id,
                detail={
                    "placement": MOTION_AFTER_TAIL,
                    "first_motion_frame": "ceil(max(wet_tail_end_s) * fps) + 1",
                    "authority": "avengine/dataset/binding_group_motion.py, measured early "
                    "binaural wet tail",
                    "scope": "the cross_time_state recipe, not QA-13/16/17 in general",
                },
                planning={
                    "motion_window_placement": MOTION_AFTER_TAIL,
                    "speech_motion": "all_still",
                },
                evidence={"motion_run_starts_after_wet_tail": True},
            )
        )
    return conditions


# --------------------------------------------------------------------------- handlers

_HANDLERS: dict[str, Callable[..., list[Condition]]] = {}


def _handler(*qa_ids: str) -> Callable[[Callable[..., list[Condition]]], Callable[..., list[Condition]]]:
    def register(function: Callable[..., list[Condition]]) -> Callable[..., list[Condition]]:
        for qa_id in qa_ids:
            _HANDLERS[qa_id] = function
        return function

    return register


def _target(subjects: Sequence[ConditionSubject]) -> ConditionSubject:
    for item in subjects:
        if item.role == "target":
            return item
    raise GenerationConditionError("a compiled condition set needs one target subject")


@_handler("QA-01", "QA-02", "QA-19", "QA-21")
def _appearance_bound(*, qa_id, branch, subjects, **_: Any) -> list[Condition]:
    target = _target(subjects)
    result = [
        Condition(
            key="appearance_reference",
            kind="appearance_reference",
            subject=target.entity_instance_id,
            detail={"unique_target_value": True, "reviewed": True},
            planning={"require_unique_reviewed_appearance": True},
            evidence={"actors_key": "appearance", "field_and_value": True},
        ),
        _activity_condition(target.entity_instance_id),
    ]
    if qa_id == "QA-02":
        result.append(
            Condition(
                key="visibility_state",
                kind="visibility_state",
                subject=target.entity_instance_id,
                detail={"allowed_states": sorted(VISIBLE_STATES), "at": "event_frame"},
                planning={"anchor_visibility": "in_fov"},
                evidence={"visibility_state_in": sorted(VISIBLE_STATES)},
            )
        )
    if qa_id == "QA-21":
        result.append(
            Condition(
                key="sound_class_variety",
                kind="sound_class_variety",
                subject=None,
                detail={"distinct_sound_classes": 2, "explicit_class_required": True},
                planning={"distinct_sound_classes": 2},
                evidence={"events_carry_sound_class_explicit": True},
            )
        )
    if qa_id == "QA-19":
        result.append(
            Condition(
                key="first_event_of_target",
                kind="event_selection",
                subject=target.entity_instance_id,
                detail={
                    "first_event_of_target": True,
                    "note": "this narrows the target audible window to its own first "
                    "emission; it does not restate the request event selector",
                },
                planning={"target_event_ordinal": 1},
                evidence={"target_first_event_identified": True},
            )
        )
    return result


@_handler("QA-03", "QA-18", "QA-24")
def _multi_candidate(*, qa_id, branch, subjects, precision, **_: Any) -> list[Condition]:
    target = _target(subjects)
    result = [
        Condition(
            key="labelled_candidates",
            kind="appearance_reference",
            subject=None,
            detail={"candidate_labels": True, "unique_labels": True},
            planning={"require_distinct_candidate_labels": True},
            evidence={"actor_labels_unique": True},
        ),
        _activity_condition(None),
    ]
    if qa_id in {"QA-03", "QA-24"}:
        result.append(
            Condition(
                key="unique_earliest_speaker",
                kind="event_selection",
                subject=None,
                detail={
                    "unique_earliest": True,
                    "earliest_instance": target.entity_instance_id,
                    "why": "the gold answer of this question is the entity that spoke "
                    "first, so the scheduler has to put that entity first rather than "
                    "draw an actor order uniformly and hope",
                },
                planning={
                    "event_relation": "sequential",
                    "first_speaker_instance_id": target.entity_instance_id,
                },
                evidence={"single_earliest_event_actor": True},
            )
        )
    if qa_id == "QA-18":
        result.append(_tail_condition(None))
        result.append(
            Condition(
                key="active_at_query_time",
                kind="event_selection",
                subject=None,
                detail={
                    "query_time": True,
                    "window": "wet-tail complement, so a silent answer is not just a tail",
                },
                planning={"require_wet_tail_complement_window": True},
                evidence={"query_time_in_legal_window": True},
            )
        )
        result.append(_public_window_condition(qa_id, precision))
    if qa_id == "QA-24":
        target = _target(subjects)
        result.append(
            Condition(
                key="visibility_state",
                kind="visibility_state",
                subject=target.entity_instance_id,
                detail={"state": branch, "at": "final_frame", "four_state": True},
                planning=_visibility_planning(branch),
                evidence={"final_visibility_state": branch, "state_space": list(VISIBILITY_STATES)},
                **_visibility_state_verdict(branch),
            )
        )
    return result


def _visibility_planning(branch: str | None) -> dict[str, Any]:
    if branch == "out_of_view":
        return {"anchor_visibility": "off_screen"}
    if branch == "visible_clear":
        return {"anchor_visibility": "in_fov", "anchor_line_of_sight": "clear"}
    if branch == "visible_occluded":
        return {"anchor_visibility": "in_fov", "anchor_line_of_sight": "occluded"}
    if branch == "fully_occluded":
        return {"anchor_visibility": "in_fov", "pixel_occlusion_transition": "fully_occluded"}
    if branch == "out_of_view":
        return {"anchor_visibility": "off_screen"}
    return {}


def _visibility_state_verdict(branch: str | None) -> dict[str, str]:
    if branch == "visible_occluded":
        return {
            "state": STATE_AVAILABLE,
            "reason": "anchor_line_of_sight=occluded is a ray proxy; the pixel state is "
            "still the answer authority",
        }
    return {}


@_handler("QA-04")
def _speaker_side(*, subjects, precision, **_: Any) -> list[Condition]:
    target = _target(subjects)
    return [
        _activity_condition(target.entity_instance_id),
        Condition(
            key="listener_relative_sector",
            kind="bearing_reference",
            subject=target.entity_instance_id,
            detail={
                "reference": "listener_relative_azimuth",
                "convention": "front 0, right positive, [-180, 180)",
                "front_dead_zone_deg": QA04_SIDE_DEAD_ZONE_DEG,
                "stable_frames_from_onset": 2,
                "judge": f"{_CATALOG_PATH}:_event_start_side_window",
                "note": "the sign convention is only flipped at the publication edge, and "
                "this is the offset from the listener median plane, not the angle between "
                "two sources",
            },
            planning={"anchor_median_plane_offset_deg": QA04_SIDE_DEAD_ZONE_DEG},
            evidence={
                "listener_status": "pass",
                "stable_side_window_from_onset": True,
                "public_interval_displayable": True,
            },
        ),
    ]


@_handler("QA-05")
def _overlap(*, branch, subjects, **_: Any) -> list[Condition]:
    overlap = branch != "disjoint"
    return [
        Condition(
            key="event_interval_relation",
            kind="event_selection",
            subject=None,
            detail={"pairwise_intervals": True, "overlap": overlap},
            planning={
                "event_relation": "overlap" if overlap else "sequential",
                "minimum_overlap_s": 0.3 if overlap else None,
            },
            evidence={"measured_interval_overlap": overlap},
        ),
        _activity_condition(None),
    ]


@_handler("QA-06")
def _speaking_while_moving(*, qa_id, branch, subjects, **_: Any) -> list[Condition]:
    target = _target(subjects)
    moving = branch == "moving"
    result = [_activity_condition(target.entity_instance_id)]
    result.extend(_motion_during_condition(target, moving=moving, qa_id=qa_id))
    return result


@_handler("QA-07")
def _entry_side(*, branch, subjects, precision, **_: Any) -> list[Condition]:
    # The public interval of this question is the entry transition itself, so it
    # is measured by the entry condition rather than by the shared window check.
    target = _target(subjects)
    return [
        Condition(
            key="entry_transition",
            kind="entry_transition",
            subject=target.entity_instance_id,
            detail={
                "transition": "out_of_view_to_visible",
                "consecutive_frames": True,
                "side": branch,
                "side_dead_zone_px": "max(1, width * 0.02)",
                "note": "the retained batch has no out_of_view state at all, so this is a "
                "route and camera condition rather than a scoring threshold",
            },
            planning={"visibility_transition": "out_of_view_to_visible", "entry_side": branch},
            evidence={
                "previous_state": "out_of_view",
                "current_state_in": sorted(VISIBLE_STATES),
                "centroid_offset_sign": "positive" if branch == "right" else "negative",
                "public_interval_displayable": True,
            },
        ),
    ]


@_handler("QA-08")
def _visibility_while_speaking(*, branch, subjects, **_: Any) -> list[Condition]:
    target = _target(subjects)
    return [
        _activity_condition(target.entity_instance_id),
        Condition(
            key="visibility_state",
            kind="visibility_state",
            subject=target.entity_instance_id,
            detail={"state": branch, "at": "event_start_visibility_window", "four_state": True},
            planning=_visibility_planning(branch),
            evidence={"visibility_state": branch, "state_space": list(VISIBILITY_STATES)},
            **_visibility_state_verdict(branch),
        ),
    ]


@_handler("QA-09")
def _reappearance(*, branch, subjects, **_: Any) -> list[Condition]:
    target = _target(subjects)
    result = [
        Condition(
            key="occlusion_transition",
            kind="occlusion_transition",
            subject=target.entity_instance_id,
            detail={
                "requires_state": "fully_occluded",
                "then_visible": branch == "yes",
                "note": "a blocked line-of-sight ray and a partial occlusion are both "
                "different measurements and cannot stand in for a fully_occluded pixel state",
            },
            planning={"pixel_occlusion_transition": "fully_occluded_then_visible"
                      if branch == "yes" else "fully_occluded_without_return"},
            evidence={
                "fully_occluded_frames_present": True,
                "visible_after_full_occlusion": branch == "yes",
            },
        )
    ]
    if branch == "no":
        result.append(
            Condition(
                key="visibility_coverage_complete",
                kind="visibility_coverage_complete",
                subject=target.entity_instance_id,
                detail={
                    "why": "a negative reappearance answer is only observable when every "
                    "frame carries an explicit visibility state"
                },
                planning={"require_complete_visibility_coverage": True},
                evidence={"judge": "avengine/qa/unified_catalog.py:_visibility_is_complete"},
            )
        )
    return result


@_handler("QA-10")
def _occluder(*, subjects, **_: Any) -> list[Condition]:
    target = _target(subjects)
    return [
        Condition(
            key="occluder_identity",
            kind="occluder_identity",
            subject=target.entity_instance_id,
            detail={"unique_registered_occluder": True},
            planning={
                "registered_occluder_transition": "registered_occluder_visible",
                # The second body must be allowed onto the target's bearing.
                "separation_bin_deg": {"bins": [[0.0, 180.0]], "floor_deg": 0.0},
            },
            evidence={"occluder_ids_present": True, "occluder_registry_resolves": True},
        )
    ]


@_handler("QA-11")
def _became_clear(*, subjects, **_: Any) -> list[Condition]:
    target = _target(subjects)
    return [
        Condition(
            key="occlusion_transition",
            kind="occlusion_transition",
            subject=target.entity_instance_id,
            detail={"transition": "visible_occluded_to_visible_clear"},
            planning={"pixel_occlusion_partial_transition": "visible_occluded_to_visible_clear"},
            evidence={"previous_state": "visible_occluded", "current_state": "visible_clear"},
            reason="the camera-first path constructs a partial view followed by a clear view; "
            "native pixels remain the answer authority",
        )
    ]


@_handler("QA-12")
def _spoken_content(*, subjects, registry, **_: Any) -> list[Condition]:
    target = _target(subjects)
    record = _registry_record(registry, target.asset_id)
    state, reason = STATE_AVAILABLE, ""
    if record is not None and source_family(record) == "device":
        state = STATE_AVAILABLE
        reason = "a playback device can carry a transcript, so this stays applicable"
    return [
        Condition(
            key="transcript",
            kind="speech_content",
            subject=target.entity_instance_id,
            detail={"transcript": True, "unique_target_statement": True},
            planning={"require_transcript_bound_sound": True},
            evidence={"events_key": "transcript", "unique_per_target": True},
            state=state,
            reason=reason,
        ),
        _activity_condition(target.entity_instance_id),
    ]


@_handler("QA-13")
def _post_sound_azimuth(*, subjects, precision, task_family=None, **_: Any) -> list[Condition]:
    target = _target(subjects)
    result = list(_post_tail_motion_conditions(target, "QA-13", task_family))
    result.append(
        Condition(
            key="post_sound_bearing",
            kind="bearing_reference",
            subject=target.entity_instance_id,
            detail={"reference": "listener_relative_azimuth", "public_unit": "integer_degree"},
            planning={},
            evidence={"azimuth_derivable_at_query": True},
        )
    )
    result.append(_public_window_condition("QA-13", precision))
    return result


@_handler("QA-14")
def _distance_comparison(*, subjects, precision, **_: Any) -> list[Condition]:
    targets = _subject_ids(subjects, "target")
    return [
        Condition(
            key="two_unique_targets",
            kind="appearance_reference",
            subject=None,
            detail={"two_unique_targets": True, "targets": list(targets)},
            planning={"require_unique_reviewed_appearance": True, "min_entity_instances": 2},
            evidence={"two_distinct_reviewed_appearances": True},
        ),
        Condition(
            key="distance_margin_at_time",
            kind="distance_margin_at_time",
            subject=None,
            detail={"margin_m": DISTANCE_MARGIN_M, "at": "query_frame",
                    "targets": list(targets)},
            planning={"distance_range_m": None, "require_distance_margin_m": DISTANCE_MARGIN_M},
            evidence={
                "listener_relative_distance_gap_at_least_m": DISTANCE_MARGIN_M,
                "public_interval_displayable": True,
            },
        ),
        Condition(
            key="targets_observable",
            kind="visibility_state",
            subject=None,
            detail={"allowed_states": sorted(VISIBLE_STATES), "at": "query_frame"},
            planning={"competitor_visibility": "in_fov"},
            evidence={"both_targets_visibility_state_in": sorted(VISIBLE_STATES)},
        ),
    ]


@_handler("QA-15")
def _distance_trend_during(*, qa_id, branch, subjects, **_: Any) -> list[Condition]:
    target = _target(subjects)
    result = [_activity_condition(target.entity_instance_id)]
    result.extend(
        _motion_during_condition(
            target, moving=True, qa_id=qa_id, distinguish_predicate="distance_trend"
        )
    )
    result.append(
        Condition(
            key="distance_net_change",
            kind="distance_net_change",
            subject=target.entity_instance_id,
            detail={
                "margin_m": DISTANCE_MARGIN_M,
                "sign": "negative" if branch == "nearer" else "positive",
                "measured_between": "event start frame and min(frame_count - 1, max(start + 1, end - 1))",
                "judge_is_endpoint_only": True,
                "note": "the shipped judge compares two endpoints, so a 0.2 m net change "
                "is not a proof of a monotone trend across the interval",
            },
            planning={"distance_trend_during_event": branch},
            evidence={
                "net_change_at_least_m": DISTANCE_MARGIN_M,
                "expected_sign": "negative" if branch == "nearer" else "positive",
                "monotonic_over_interval": "reported_separately",
            },
        )
    )
    return result


@_handler("QA-16")
def _distance_trend_after(*, branch, subjects, precision, task_family=None, **_: Any) -> list[Condition]:
    target = _target(subjects)
    result = list(_post_tail_motion_conditions(target, "QA-16", task_family))
    result.append(
        Condition(
            key="distance_stable_after_sound",
            kind="distance_stable_after_sound",
            subject=target.entity_instance_id,
            detail={
                "margin_m": DISTANCE_MARGIN_M,
                "stable_over_whole_window": True,
                "reference": "source distance at the anchor event end frame",
            },
            planning={"target_moved_after_sound": True, "competitor_motion": "still"},
            evidence={
                "every_frame_same_trend": True,
                "every_frame_margin_at_least_m": DISTANCE_MARGIN_M,
            },
        )
    )
    result.append(_public_window_condition("QA-16", precision))
    return result


@_handler("QA-17")
def _motion_after_sound(*, branch, subjects, precision, task_family=None, **_: Any) -> list[Condition]:
    target = _target(subjects)
    moved = branch == "yes"
    result = list(_post_tail_motion_conditions(target, "QA-17", task_family))
    result.append(
        Condition(
            key="motion_after_sound",
            kind="motion_after_sound",
            subject=target.entity_instance_id,
            detail={
                "any_motion_in_interval": moved,
                "interval": "anchor event end frame through the query frame",
                "note": "the question asks whether any movement occurred; it does not "
                "require every frame in the interval to be moving",
            },
            planning={"target_moved_after_sound": moved},
            evidence={"any_moving_frame_in_interval": moved, "stable_answer_over_window": True},
        )
    )
    result.append(
        Condition(
            key="answer_distinguishable",
            kind="answer_distinguishable",
            subject=target.entity_instance_id,
            detail={
                "predicate": "motion_after_sound",
                "must_differ_from_role": "competitor",
                "why": "the retained cross-time state groups moved both objects after the "
                "tail, so QA-17 read yes for both candidates",
            },
            planning={"competitor_motion": "still" if moved else "moving"},
            evidence={"competitor_predicate_differs": True},
        )
    )
    result.append(_public_window_condition("QA-17", precision))
    return result


@_handler("QA-20")
def _visible_candidate_or_none(*, branch, subjects, **_: Any) -> list[Condition]:
    target = _target(subjects)
    hidden = branch == "none_of_them"
    return [
        _activity_condition(target.entity_instance_id),
        Condition(
            key="visibility_state",
            kind="visibility_state",
            subject=target.entity_instance_id,
            detail={
                "emitter_among_visible_candidates": not hidden,
                "at": "anchor_event_frame",
                "note": "the none_of_them branch is the off-screen emitter case, so the "
                "answer domain is derived from the camera rather than limited to the frustum",
            },
            planning={"anchor_visibility": "off_screen" if hidden else "in_fov"},
            evidence={
                "visibility_state_in": ["out_of_view", "fully_occluded"]
                if hidden
                else sorted(VISIBLE_STATES)
            },
        ),
        Condition(
            key="visible_candidate_domain",
            kind="candidate_domain",
            subject=None,
            detail={"min_visible_candidates": 2, "include_none_option": hidden},
            planning={"competitor_visibility": "in_fov", "min_entity_instances": 2},
            evidence={"visible_candidate_frame_labelled": True},
        ),
        Condition(
            key="target_event_is_attributable",
            kind="event_selection",
            subject=target.entity_instance_id,
            detail={
                "first_event_of_target": True,
                "why": "this question asks which visible candidate the listener heard, so "
                "the event it points at has to be the target's own and has to be "
                "attributable to it. It used to carry no event statement at all and "
                "borrowed the ordering QA-24 happened to request when they shared a world.",
                "not_earliest_overall": "being first among all speakers is stronger than "
                "this question needs, and would contradict the QA-05 overlap branch that "
                "twelve shipped episodes pair it with",
            },
            planning={"target_event_ordinal": 1},
            evidence={"unique_onset_or_unique_sound_class": True},
        ),
    ]


@_handler("QA-22", "QA-23")
def _counts(*, qa_id, subjects, **_: Any) -> list[Condition]:
    result = [
        Condition(
            key="whole_clip_statistics",
            kind="event_statistics_window",
            subject=None,
            detail={"window": "whole_clip", "counts": "entities and speaking individuals"
                    if qa_id == "QA-22" else "raw event ids"},
            planning={"event_statistics_window": "whole_clip"},
            evidence={"events_enumerated": True, "event_ids_distinct": True},
        ),
        _activity_condition(None),
    ]
    if qa_id == "QA-23":
        result.append(
            Condition(
                key="event_definition",
                kind="event_statistics_window",
                subject=None,
                detail={
                    "event_definition": "one programmed emission of one instance",
                    "note": "a repeated emission of one instance counts separately, which is "
                    "what event_relation=repeat produces",
                },
                planning={"event_relation": "repeat"},
                evidence={"event_segmentation_proven": True},
            )
        )
    return result


@_handler("QA-25")
def _continuous_bearing(*, branch, subjects, precision, **_: Any) -> list[Condition]:
    target = _target(subjects)
    result: list[Condition] = []
    if branch == "A":
        result.append(_activity_condition(target.entity_instance_id))
        result.append(
            Condition(
                key="bearing_reference",
                kind="bearing_reference",
                subject=target.entity_instance_id,
                detail={
                    "subset": "A",
                    "target": "audible_event_emitter",
                    "onset_identifies_event": True,
                    "note": "simultaneous onsets need a unique sound class, otherwise the "
                    "listener cannot tell which emitter the question means",
                },
                planning={"event_relation": "sequential"},
                evidence={"unique_onset_or_unique_sound_class": True},
            )
        )
    elif branch == "V":
        result.append(
            Condition(
                key="bearing_reference",
                kind="bearing_reference",
                subject=target.entity_instance_id,
                detail={
                    "subset": "V",
                    "target": "visible_pixel_centroid",
                    "public_camera_calibration": True,
                    "query_at_whole_second": True,
                },
                planning={"anchor_visibility": "in_fov", "public_time_precision": precision},
                evidence={
                    "camera_calibration_published": True,
                    "visual_bearing_derivable": True,
                },
            )
        )
    else:
        result.append(_activity_condition(target.entity_instance_id))
        result.append(_tail_condition(target.entity_instance_id))
        result.append(
            Condition(
                key="bearing_reference",
                kind="bearing_reference",
                subject=target.entity_instance_id,
                detail={
                    "subset": "AV",
                    "target": "visually_anchored_hidden_emitter",
                    "min_query_sources": 2,
                    "anchor_frame_isolated": True,
                    "rival_sound_asset_must_differ": True,
                },
                planning={"anchor_count": 1, "min_entity_instances": 2, "event_relation": "overlap"},
                evidence={
                    "isolated_visible_anchor_frame": True,
                    "no_competing_wet_tail_at_anchor": True,
                    "rival_active_with_other_sound_asset": True,
                },
            )
        )
        result.append(
            Condition(
                key="visibility_state",
                kind="visibility_state",
                subject=target.entity_instance_id,
                detail={
                    "state_at_query_in": ["out_of_view", "fully_occluded"],
                    "why": "the AV subset asks for a bearing the pixels cannot supply at the "
                    "query moment, so the target has to be hidden there",
                },
                planning={
                    # The whole-clip crossing is not the statement this question
                    # makes. What it needs is one audible event that carries a
                    # visible identification frame and, later inside that same
                    # event, a hidden frame to be asked at. anchor_visibility
                    # says that per event; visibility_transition could only say
                    # it about the clip, and a uniform in_fov mask - the default
                    # - actively contradicted the hidden query frame.
                    "anchor_visibility": "visible_then_hidden",
                },
                evidence={"query_visibility_state_in": ["out_of_view", "fully_occluded"]},
                search_frames=True,
            )
        )
        result.append(
            Condition(
                key="hidden_motion_changed",
                kind="motion_during_event",
                subject=target.entity_instance_id,
                detail={
                    # The target moves during its own audible window: that is what
                    # "changed while hidden" means. Leaving this unsaid made
                    # conditioned_motion._meaning fall through to
                    # post_sound_query_only, because the AV subset also reads the
                    # wet tail, and the solver then demanded a whole second after
                    # the tail that this question never asks for.
                    "moving": True,
                    "hidden_change_required": True,
                    "between": "last visible frame and the query frame",
                    "why": "without a change while hidden the visual extrapolation from the "
                    "last visible frame already answers the question",
                },
                planning={"speech_motion": "speaker_moving",
                          "motion_window_placement": MOTION_DURING_EVENT},
                evidence={"judge": "avengine/qa/angular_questions.py:_hidden_motion_changed"},
            )
        )
    result.append(_public_window_condition("QA-25", precision))
    return result


# --------------------------------------------------------------------------- compile


def _resolve_subjects(
    qa_id: str,
    target: Any,
    instances: Any,
    registry: Any,
) -> tuple[tuple[ConditionSubject, ...], dict[str, Any], str | None]:
    """Build the subject list from a P01 qa_target plus the request instances."""

    if hasattr(target, "to_dict"):
        data = dict(target.to_dict())
    elif isinstance(target, Mapping):
        data = dict(target)
    else:
        raise GenerationConditionError("target must be a QaTargetSpec or a mapping")
    target_ids = list(data.get("target_instance_ids") or data.get("target_instances") or ())
    if not target_ids:
        raise GenerationConditionError(f"{qa_id} target must name its entity instances")
    records = _instance_records(instances)
    anchors = set(data.get("anchor_instance_ids") or ())
    subjects: list[ConditionSubject] = []
    for instance_id in target_ids:
        row = records.get(instance_id, {})
        subjects.append(
            ConditionSubject(
                entity_instance_id=instance_id,
                role="target",
                asset_id=row.get("asset_id"),
                source_class=row.get("source_class"),
                sound_asset_id=row.get("sound_asset_id"),
                sound_identity_id=row.get("sound_identity_id"),
                is_anchor=instance_id in anchors or row.get("role") == "anchor" or not anchors,
            )
        )
    for instance_id, row in records.items():
        if instance_id in target_ids:
            continue
        subjects.append(
            ConditionSubject(
                entity_instance_id=instance_id,
                role="competitor",
                asset_id=row.get("asset_id"),
                source_class=row.get("source_class"),
                sound_asset_id=row.get("sound_asset_id"),
                sound_identity_id=row.get("sound_identity_id"),
            )
        )
    selector = data.get("event") or {"kind": "target_audible_window"}
    if isinstance(selector, str):
        selector = {"kind": selector}
    return tuple(subjects), dict(selector), data.get("branch")


def compile_generation_conditions(
    target: Any,
    *,
    branch: Any = None,
    instances: Any = (),
    registry: Any = None,
    public_time_precision: int = DEFAULT_PUBLIC_TIME_PRECISION,
    task_family: str | None = None,
    capabilities: Any = None,
    backend: str | None = None,
) -> CompiledConditions:
    """Compile one QA type and one key branch into actual generation conditions.

    ``target`` is a P01 ``QaTargetSpec`` or the same mapping shape, so a config
    that already names ``qa_id``, ``target_instance_ids`` and ``event`` needs no
    second request vocabulary.  ``branch`` may also travel on that mapping under
    the ``branch`` key; an explicit argument wins.

    ``capabilities`` is the planner's own declaration of the knobs it accepts,
    resolved by :func:`resolve_generator_capabilities`; passing nothing keeps the
    measured fallback, so an existing caller sees no change.  ``backend`` names
    the room family route, because a solver may support a knob on one route only.
    """

    if hasattr(target, "qa_id"):
        qa_id = _qa_id(target.qa_id)
    elif isinstance(target, Mapping):
        qa_id = _qa_id(target.get("qa_id"))
    else:
        raise GenerationConditionError("target must carry a qa_id")
    subjects, selector, declared_branch = _resolve_subjects(qa_id, target, instances, registry)
    scope = subject_scope(qa_id)
    named = _subject_ids(subjects, "target")
    if scope == "single_target" and len(named) != 1:
        raise GenerationConditionError(
            f"{qa_id} asks about one instance at a time but names {list(named)}; call "
            "compile_target_candidates so every candidate keeps its own result"
        )
    if scope == "target_pair" and len(named) != 2:
        raise GenerationConditionError(
            f"{qa_id} compares two named instances but names {list(named)}; call "
            "compile_target_candidates to enumerate the pairs"
        )
    resolved_branch = _check_branch(qa_id, branch if branch is not None else declared_branch)

    conditions: list[Condition] = [
        _modality_condition(qa_id, resolved_branch),
        _domain_condition(qa_id),
        _event_condition(qa_id, selector, _target(subjects).entity_instance_id),
        Condition(
            key="role_assignment",
            kind="role_assignment",
            subject=None,
            detail={
                "targets": list(_subject_ids(subjects, "target")),
                "competitors": list(_subject_ids(subjects, "competitor")),
                "note": "roles attach to entity_instance_id; the anchor role is a separate "
                "statement from the first programmed event",
            },
            planning={"anchor_count": max(1, len(_subject_ids(subjects, "target")))},
            evidence={"roles_recorded_per_instance": True},
        ),
    ]

    handler = _HANDLERS.get(qa_id)
    if handler is not None:
        conditions.extend(
            handler(
                qa_id=qa_id,
                branch=resolved_branch,
                subjects=subjects,
                registry=registry,
                precision=public_time_precision,
                task_family=task_family,
            )
        )

    # A device holds every role except its own locomotion.
    if qa_id in SELF_MOTION_QA_IDS:
        subject = _target(subjects)
        record = _registry_record(registry, subject.asset_id)
        state, reason = _locomotion_state(record, subject.source_class)
        conditions.append(
            Condition(
                key="target_self_locomotion",
                kind="role_assignment",
                subject=subject.entity_instance_id,
                detail={"requires_self_locomotion": True, "asset_id": subject.asset_id},
                planning={"require_locomotion_capable_target": True},
                evidence={"locomotion_capability": "available"},
                state=state,
                reason=reason,
            )
        )

    resolved_capabilities = resolve_generator_capabilities(capabilities)
    if backend is not None and backend not in BACKENDS:
        raise GenerationConditionError(
            f"unknown room family route {backend!r}; the known routes are {list(BACKENDS)}"
        )
    if backend is not None and backend not in resolved_capabilities.backends:
        raise GenerationConditionError(
            f"{resolved_capabilities.label} does not declare the {backend!r} route; it "
            f"declares {list(resolved_capabilities.backends)}"
        )
    conditions = _apply_capability_states(conditions, resolved_capabilities, backend)

    conflicts = _conflicts_within(conditions, task_family=task_family, qa_id=qa_id)
    state, reason = _verdict(conditions, conflicts)
    requirements = catalog.get_requirements(qa_id)["potential_requirements"]
    return CompiledConditions(
        qa_id=qa_id,
        branch=resolved_branch,
        subjects=subjects,
        event=selector,
        conditions=tuple(conditions),
        state=state,
        reason=reason,
        conflicts=tuple(conflicts),
        required_modalities=tuple(requirements.get("required_modalities") or ()),
        capabilities=resolved_capabilities,
        backend=backend,
        public_time_precision=public_time_precision,
        task_family=task_family,
    )


def compile_target_candidates(
    target: Any,
    *,
    branch: Any = None,
    instances: Any = (),
    registry: Any = None,
    public_time_precision: int = DEFAULT_PUBLIC_TIME_PRECISION,
    task_family: str | None = None,
    capabilities: Any = None,
    backend: str | None = None,
) -> list[CompiledConditions]:
    """Compile one qa_target into one result per candidate it actually names.

    A single-target question yields one result per named instance, a pairwise
    question one result per unordered pair, and a set question one result for
    the whole named set.  Nothing is dropped, so a later summary can report why
    each candidate was refused instead of reporting one verdict for the type.
    """

    if hasattr(target, "to_dict"):
        data = dict(target.to_dict())
    elif isinstance(target, Mapping):
        data = dict(target)
    else:
        raise GenerationConditionError("target must be a QaTargetSpec or a mapping")
    qa_id = _qa_id(data.get("qa_id"))
    named = [
        str(value)
        for value in (data.get("target_instance_ids") or data.get("target_instances") or ())
    ]
    if not named:
        raise GenerationConditionError(f"{qa_id} target must name its entity instances")
    scope = subject_scope(qa_id)
    if scope == "candidate_set":
        selections: list[list[str]] = [named]
    elif scope == "target_pair":
        selections = [
            [named[first], named[second]]
            for first in range(len(named))
            for second in range(first + 1, len(named))
        ] or [named]
    else:
        selections = [[value] for value in named]
    results: list[CompiledConditions] = []
    for selection in selections:
        row = dict(data)
        row["target_instance_ids"] = selection
        row.pop("target_instances", None)
        results.append(
            compile_generation_conditions(
                row,
                branch=branch,
                instances=instances,
                registry=registry,
                public_time_precision=public_time_precision,
                task_family=task_family,
                capabilities=capabilities,
                backend=backend,
            )
        )
    return results


def _conflicts_within(
    conditions: Sequence[Condition], *, task_family: str | None, qa_id: str
) -> list[dict[str, Any]]:
    """Refuse contradictory knob demands before an expensive stage is scheduled."""

    conflicts: list[dict[str, Any]] = []
    demands: dict[tuple[str | None, str], dict[Any, list[str]]] = {}
    for item in conditions:
        for knob, value in item.planning.items():
            if value is None:
                continue
            key = (item.subject, knob)
            demands.setdefault(key, {}).setdefault(_hashable(value), []).append(item.key)
    for (subject, knob), values in sorted(demands.items(), key=lambda row: (str(row[0][0]), row[0][1])):
        if len(values) > 1:
            conflicts.append(
                {
                    "reason": "contradictory_planning_knob",
                    "subject": subject,
                    "knob": knob,
                    "values": {str(value): keys for value, keys in values.items()},
                }
            )
    if task_family in {"cross_time_state"} and qa_id in MOTION_DURING_EVENT_QA_IDS:
        wants_motion = any(
            item.planning.get("motion_window_placement") == MOTION_DURING_EVENT
            for item in conditions
        )
        if wants_motion:
            conflicts.append(
                {
                    "reason": "recipe_places_motion_after_the_measured_tail",
                    "task_family": task_family,
                    "knob": "motion_window_placement",
                    "detail": (
                        "a cross_time_state member opens its motion window at "
                        "ceil(max(wet_tail_end_s) * fps) + 1, so no audible window of that "
                        f"instance can lie inside a moving run; {qa_id} needs the opposite"
                    ),
                }
            )
    return conflicts


def _hashable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    return value


def _verdict(conditions: Sequence[Condition], conflicts: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    if conflicts:
        return STATE_NOT_APPLICABLE, "; ".join(
            str(row.get("reason")) for row in conflicts
        )
    for state in (STATE_NOT_APPLICABLE, STATE_NOT_IMPLEMENTED, STATE_EVIDENCE_MISSING):
        blocked = [item for item in conditions if item.state == state]
        if blocked:
            return state, "; ".join(f"{item.key}: {item.reason}" for item in blocked)
    return STATE_AVAILABLE, ""


def compile_request_conditions(
    request: Any,
    *,
    public_time_precision: int = DEFAULT_PUBLIC_TIME_PRECISION,
    registry: Any = None,
    branches: Mapping[str, str] | None = None,
    capabilities: Any = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Compile every qa_target of one P01 production request.

    Cross-target conflicts are reported for the request as a whole, because two
    targets can each be satisfiable and still demand opposite scene conditions
    from the one episode that has to carry both.
    """

    if hasattr(request, "qa_targets"):
        targets = list(request.qa_targets)
        instances = list(getattr(request, "instances", ()) or ())
        task_family = getattr(request, "task_family", None)
        request_id = getattr(request, "request_id", None)
    elif isinstance(request, Mapping):
        targets = list(request.get("qa_targets") or ())
        instances = list(request.get("instances") or ())
        task_family = request.get("task_family")
        request_id = request.get("request_id")
    else:
        raise GenerationConditionError("request must be a ProductionRequest or a mapping")

    compiled: list[CompiledConditions] = []
    for target in targets:
        qa_id = _qa_id(
            getattr(target, "qa_id", None) if not isinstance(target, Mapping) else target.get("qa_id")
        )
        wanted = (branches or {}).get(qa_id)
        # A branch map may name one branch or several; naming none compiles the
        # branch-agnostic set so a caller can see the shared conditions.
        selected = list(wanted) if isinstance(wanted, (list, tuple)) else [wanted]
        for branch in selected:
            compiled.extend(
                compile_target_candidates(
                    target,
                    branch=branch,
                    instances=instances,
                    registry=registry,
                    public_time_precision=public_time_precision,
                    task_family=task_family,
                    capabilities=capabilities,
                    backend=backend,
                )
            )
    resolved_capabilities = resolve_generator_capabilities(capabilities)
    return {
        "request_id": request_id,
        "task_family": task_family,
        "public_time_precision": public_time_precision,
        "backend": backend,
        "capabilities": resolved_capabilities.to_dict(),
        "compiled": [item.to_dict() for item in compiled],
        "request_conflicts": reject_conflicts(compiled),
        "report": condition_report(compiled),
    }


def reject_conflicts(compiled: Sequence[CompiledConditions]) -> list[dict[str, Any]]:
    """Report knob demands that two compiled sets cannot both get from one episode."""

    demands: dict[tuple[str | None, str], dict[Any, list[str]]] = {}
    for entry in compiled:
        label = f"{entry.qa_id}:{entry.branch or '-'}"
        for item in entry.conditions:
            for knob, value in item.planning.items():
                if value is None or planning_layer(knob) not in {
                    "sampler_profile",
                    "solver",
                    "derived",
                }:
                    continue
                demands.setdefault((item.subject, knob), {}).setdefault(
                    _hashable(value), []
                ).append(f"{label}/{item.key}")
    conflicts: list[dict[str, Any]] = []
    for (subject, knob), values in sorted(demands.items(), key=lambda row: (str(row[0][0]), row[0][1])):
        if len(values) > 1:
            conflicts.append(
                {
                    "reason": "contradictory_planning_knob",
                    "subject": subject,
                    "knob": knob,
                    "values": {str(value): keys for value, keys in values.items()},
                }
            )
    return conflicts


def restore_compiled_conditions(payload: Mapping[str, Any]) -> CompiledConditions:
    """Rebuild a compiled condition set from its own ``to_dict`` output.

    A runner persists conditions beside a work item and has to get the same
    conditions back after a restart, without recompiling against a source tree
    that may have moved on.  ``to_dict`` then ``restore`` then ``to_dict`` is
    the identity, which is what the resume path needs.
    """

    if not isinstance(payload, Mapping):
        raise GenerationConditionError("a saved condition set must be a mapping")
    try:
        qa_id = _qa_id(payload["qa_id"])
        subjects = tuple(
            ConditionSubject(
                entity_instance_id=str(row["entity_instance_id"]),
                role=str(row["role"]),
                asset_id=row.get("asset_id"),
                sound_asset_id=row.get("sound_asset_id"),
                event_id=row.get("event_id"),
                sound_identity_id=row.get("sound_identity_id"),
                source_class=row.get("source_class"),
                is_anchor=bool(row.get("is_anchor", False)),
            )
            for row in payload["subjects"]
        )
        conditions = tuple(
            Condition(
                key=str(row["key"]),
                kind=str(row["kind"]),
                subject=row.get("subject"),
                detail=dict(row.get("detail") or {}),
                planning=dict(row.get("planning") or {}),
                evidence=dict(row.get("evidence") or {}),
                state=str(row.get("state", STATE_AVAILABLE)),
                reason=str(row.get("reason", "")),
                search_frames=bool(row.get("search_frames", False)),
                support=dict(row.get("support") or {}),
            )
            for row in payload["conditions"]
        )
    except (KeyError, TypeError) as error:
        raise GenerationConditionError(f"a saved condition set is incomplete: {error}") from error
    declared = payload.get("capabilities")
    return CompiledConditions(
        qa_id=qa_id,
        branch=payload.get("branch"),
        subjects=subjects,
        event=dict(payload.get("event") or {}),
        conditions=conditions,
        state=str(payload.get("state", STATE_AVAILABLE)),
        reason=str(payload.get("reason", "")),
        conflicts=tuple(dict(row) for row in payload.get("conflicts") or ()),
        required_modalities=tuple(payload.get("required_modalities") or ()),
        capabilities=_one_declaration(declared) if declared else BASELINE_CAPABILITIES,
        backend=payload.get("backend"),
        public_time_precision=int(
            payload.get("public_time_precision", DEFAULT_PUBLIC_TIME_PRECISION)
        ),
        task_family=payload.get("task_family"),
    )


def unsupported_by_layer(compiled: Sequence[CompiledConditions]) -> dict[str, Any]:
    """Group every unmet planning key by the layer that has to implement it.

    A reader can then see how much of the remaining work is a sampler knob, how
    much is a trajectory or visibility solver, and how much is not a planner
    question at all.
    """

    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in compiled:
        for item in entry.conditions:
            for key in item.planning:
                layer = planning_layer(key)
                if layer not in CAPABILITY_LAYERS:
                    continue
                if entry.capabilities.supports(key, backend=entry.backend):
                    continue
                enforcement, basis = goal_enforcement(key)
                label = f"{entry.qa_id}:{entry.branch or '-'}"
                row = rows.setdefault(layer, {}).setdefault(
                    key, {"enforcement": enforcement, "basis": basis, "wanted_by": []}
                )
                if label not in row["wanted_by"]:
                    row["wanted_by"].append(label)
    for keys in rows.values():
        for row in keys.values():
            row["wanted_by"].sort()
    return {
        layer: {key: keys[key] for key in sorted(keys)}
        for layer, keys in sorted(rows.items())
    }


def condition_report(compiled: Sequence[CompiledConditions]) -> dict[str, Any]:
    """Aggregate per candidate, keeping the reason for every blocked candidate.

    A mixed set never collapses: one inapplicable device candidate and one
    blocked human candidate in the same episode stay two separate rows, so a
    summary cannot remove a QA type from the denominator on the strength of the
    inapplicable one.
    """

    per_qa: dict[str, dict[str, Any]] = {}
    for entry in compiled:
        row = per_qa.setdefault(
            entry.qa_id,
            {"states": {}, "candidates": [], "branches_seen": []},
        )
        row["states"][entry.state] = row["states"].get(entry.state, 0) + 1
        row["candidates"].append(
            {
                "branch": entry.branch,
                "targets": list(entry.target_instance_ids),
                "state": entry.state,
                "reason": entry.reason,
                "conflicts": [dict(item) for item in entry.conflicts],
            }
        )
        if entry.branch is not None and entry.branch not in row["branches_seen"]:
            row["branches_seen"].append(entry.branch)
    for qa_id, row in per_qa.items():
        expected = list(branches_for(qa_id))
        row["branches_expected"] = expected
        row["branches_missing"] = [item for item in expected if item not in row["branches_seen"]]
        row["any_available"] = row["states"].get(STATE_AVAILABLE, 0) > 0
    return {
        "qa_ids": dict(sorted(per_qa.items())),
        "states": list(APPLICABILITY_STATES),
        "note": "a per-candidate state is never merged into another candidate's state",
    }


def compile_catalog_matrix(
    *,
    instances: Any = (),
    registry: Any = None,
    public_time_precision: int = DEFAULT_PUBLIC_TIME_PRECISION,
    task_family: str | None = None,
    capabilities: Any = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Compile every QA type and every key branch for one entity selection.

    This is the traceable applicability pass: each of the twenty-five types
    yields a compiled result rather than a candidate id list, and a type with
    key branches yields one result per branch.
    """

    records = _instance_records(instances)
    if not records:
        raise GenerationConditionError("a catalog matrix needs at least one entity instance")
    # Each QA type gets the event selection its own question implies, so a matrix
    # row is refused for a real reason rather than for a placeholder selector.
    selectors = {qa_id: DEFAULT_EVENT_SELECTORS.get(qa_id, "target_audible_window")
                 for qa_id in QA_IDS}
    everyone = list(records)
    rows: list[CompiledConditions] = []
    for qa_id in QA_IDS:
        scope = subject_scope(qa_id)
        if scope == "single_target":
            targets = everyone[:1]
        elif scope == "target_pair":
            targets = everyone[:2]
        else:
            targets = everyone
        if scope == "target_pair" and len(targets) < 2:
            raise GenerationConditionError(
                f"{qa_id} compares two instances; this selection names only {targets}"
            )
        for branch in branches_for(qa_id) or (None,):
            rows.extend(
                compile_target_candidates(
                    {
                        "qa_id": qa_id,
                        "target_instance_ids": targets,
                        "event": {"kind": selectors[qa_id]},
                    },
                    branch=branch,
                    instances=instances,
                    registry=registry,
                    public_time_precision=public_time_precision,
                    task_family=task_family,
                    capabilities=capabilities,
                    backend=backend,
                )
            )
    resolved_capabilities = resolve_generator_capabilities(capabilities)
    return {
        "task_family": task_family,
        "public_time_precision": public_time_precision,
        "backend": backend,
        "capabilities": resolved_capabilities.to_dict(),
        "compiled": [item.to_dict() for item in rows],
        "report": condition_report(rows),
        "unsupported_by_layer": unsupported_by_layer(rows),
    }


# --------------------------------------------------------------------- evidence side

# Per-check verdicts.  The aggregate adds ``incomplete`` for the case where
# nothing contradicts the conditions but some of them could not be measured
# here; that is not the same as a proven pass.
CHECK_STATUSES = ("pass", "fail", "not_run")
AGGREGATE_STATUSES = ("pass", "fail", "incomplete")

# The readback side deliberately calls the catalog's own predicates rather than
# reimplementing them.  A check that measures something adjacent to what the
# judge measures produces reports that look reasonable and prove nothing.
_JUDGE_SOURCE = _CATALOG_PATH


def _row(
    condition: Condition, status: str, *, measured: Any = None, reason: str = "", **extra: Any
) -> dict[str, Any]:
    if status not in CHECK_STATUSES:
        raise GenerationConditionError(f"unknown check status: {status!r}")
    row: dict[str, Any] = {
        "key": condition.key,
        "kind": condition.kind,
        "subject": condition.subject,
        "status": status,
    }
    if measured is not None:
        row["measured"] = measured
    if reason:
        row["reason"] = reason
    row.update(extra)
    return row


def _resolve_actor(
    facts: Mapping[str, Any],
    subject: ConditionSubject,
    actor_by_instance: Mapping[str, str] | None,
) -> str | None:
    """Map one entity instance onto the actor id used inside these facts."""

    actors = facts.get("actors")
    if not isinstance(actors, Mapping):
        return None
    if actor_by_instance and subject.entity_instance_id in actor_by_instance:
        candidate = actor_by_instance[subject.entity_instance_id]
        return candidate if candidate in actors else None
    if subject.entity_instance_id in actors:
        return subject.entity_instance_id
    if subject.asset_id is not None:
        matches = [
            actor_id
            for actor_id, row in actors.items()
            if isinstance(row, Mapping) and str(row.get("asset_id")) == subject.asset_id
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _target_events(facts: Mapping[str, Any], actor_id: str, selector: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every event of one actor the stated selector admits, earliest first."""

    try:
        events = catalog._bound_events(facts)
    except catalog._Deferred:
        return []
    own = sorted(
        (event for event in events if event.get("actor_id") == actor_id),
        key=lambda event: (float(event["start_s"]), str(event["event_id"])),
    )
    kind = str(selector.get("kind", "target_audible_window"))
    if kind == "event_id":
        wanted = selector.get("event_id")
        return [event for event in own if str(event.get("event_id")) == str(wanted)]
    if kind == "event_ordinal":
        ordinal = int(selector.get("ordinal") or 1)
        return own[ordinal - 1 : ordinal]
    if kind == "whole_clip":
        return []
    return own


def _event_frames(facts: Mapping[str, Any], event: Mapping[str, Any]) -> tuple[int, int]:
    count = int(facts["time"]["frame_count"])
    start = max(0, catalog._event_frame(event, "start_frame"))
    end = min(count, catalog._event_frame(event, "end_frame"))
    return start, end


def _moving_runs(facts: Mapping[str, Any], actor_id: str) -> list[list[int]]:
    row = facts.get("actors", {}).get(actor_id, {})
    flags = row.get("moving") if isinstance(row, Mapping) else None
    if not isinstance(flags, Sequence):
        return []
    runs: list[list[int]] = []
    for index, value in enumerate(flags):
        if not bool(value):
            continue
        if runs and runs[-1][1] == index:
            runs[-1][1] = index + 1
        else:
            runs.append([index, index + 1])
    return runs


def _max_wet_tail_end_s(facts: Mapping[str, Any], event_id: Any) -> float | None:
    audio = facts.get("audio")
    tails = audio.get("wet_tail_intervals") if isinstance(audio, Mapping) else None
    if not isinstance(tails, Sequence):
        return None
    ends = [
        float(row["end_s"])
        for row in tails
        if isinstance(row, Mapping) and row.get("event_id") == event_id and "end_s" in row
    ]
    return max(ends) if ends else None


def _legal_post_tail_windows(
    facts: Mapping[str, Any], event: Mapping[str, Any], qa_id: str
) -> list[list[int]] | None:
    """Reuse the catalog's own derivation of the silent post-sound windows."""

    reference = qa_id if qa_id in TAIL_DEPENDENT_QA_IDS else "QA-17"
    try:
        return catalog._derived_legal_query_windows(facts, reference, event=event)
    except catalog._Deferred:
        return None


def _stable_runs(
    windows: Sequence[Sequence[int]],
    value_for_frame: Callable[[int], Any],
    *,
    min_frames: int = 2,
) -> list[dict[str, Any]]:
    """Split each legal window into its maximal runs of one constant answer.

    ``unified_catalog._stable_frame_window`` searches for the longest interval
    inside one legal window that contains the sampled query frame and holds a
    single answer.  The maximal constant runs are exactly the intervals that
    search can return, so a caller can decide feasibility without having to
    guess which frame the sampler will draw.
    """

    runs: list[dict[str, Any]] = []
    for start, end in windows:
        current: dict[str, Any] | None = None
        for frame in range(int(start), int(end)):
            value = value_for_frame(frame)
            if current is not None and current["value"] == value:
                current["end"] = frame + 1
                continue
            if current is not None and current["end"] - current["start"] >= min_frames:
                runs.append(current)
            current = {"start": frame, "end": frame + 1, "value": value}
        if current is not None and current["end"] - current["start"] >= min_frames:
            runs.append(current)
    return runs


def _displayable(run: Mapping[str, Any], fps: float, precision: int) -> list[float] | None:
    bounds = integer_second_window(int(run["start"]) / fps, int(run["end"]) / fps, precision=precision)
    return None if bounds is None else [bounds[0], bounds[1]]


def _any_motion_between(facts: Mapping[str, Any], actor_id: str, first: int, last: int) -> bool | None:
    try:
        return any(
            catalog._motion_at(facts, actor_id, frame) for frame in range(max(0, first), last + 1)
        )
    except catalog._Deferred:
        return None


def _distances_over(facts: Mapping[str, Any], actor_id: str, frames: Sequence[int]) -> list[float] | None:
    try:
        return [catalog._distance_at(facts, actor_id, int(frame)) for frame in frames]
    except catalog._Deferred:
        return None


def _appearance_candidate_count(facts: Mapping[str, Any]) -> int:
    try:
        return len(catalog._reviewed_appearances(facts))
    except catalog._Deferred:
        return 0


def _visibility_state_at(facts: Mapping[str, Any], actor_id: str, frame: int) -> str | None:
    frames = facts.get("visibility", {}).get(actor_id)
    if not isinstance(frames, Mapping):
        return None
    record = frames.get(frame, frames.get(str(frame)))
    if not isinstance(record, Mapping):
        return None
    state = record.get("state")
    return str(state) if isinstance(state, str) else None


def _visibility_series(facts: Mapping[str, Any], actor_id: str) -> list[Mapping[str, Any]]:
    frames = facts.get("visibility", {}).get(actor_id)
    if not isinstance(frames, Mapping):
        return []
    def order(key: Any) -> int:
        try:
            return int(key)
        except (TypeError, ValueError):
            return -1
    return [frames[key] for key in sorted(frames, key=order) if isinstance(frames[key], Mapping)]


# --------------------------------------------------------------------- per-kind checks


def _check_motion_during_event(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    actor_id = actors.get(condition.subject)
    if actor_id is None:
        return _row(condition, "not_run", reason="the target instance has no actor in these facts")
    if condition.detail.get("hidden_change_required"):
        return _check_hidden_motion_changed(condition, facts, actor_id)
    if event is None:
        return _row(condition, "not_run", reason="no bound event of the target was selected")
    start, end = _event_frames(facts, event)
    expected = condition.evidence.get("expected_value")
    try:
        actual = catalog._stable_motion_window(facts, actor_id, start, end)
    except catalog._Deferred as error:
        return _row(
            condition,
            "fail",
            measured={"event_frames": [start, end], "judge_code": error.code},
            reason=f"{_JUDGE_SOURCE}:_stable_motion_window refused this window: {error.code}",
        )
    runs = _moving_runs(facts, actor_id)
    measured = {
        "event_frames": [start, end],
        "moving_constant_value": bool(actual),
        "moving_runs": runs,
    }
    if expected is None:
        return _row(condition, "pass", measured=measured)
    status = "pass" if bool(actual) == bool(expected) else "fail"
    return _row(
        condition,
        status,
        measured=measured,
        reason=""
        if status == "pass"
        else f"the target is {'moving' if actual else 'still'} across its audible window; "
        f"this branch needs {'moving' if expected else 'still'}",
    )


def _check_hidden_motion_changed(
    condition: Condition, facts: Mapping[str, Any], actor_id: str
) -> dict[str, Any]:
    """The AV subset needs a real change while the target is not visible.

    Without one, extrapolating the last visible motion already answers the
    question and the audio adds nothing, so this calls the same judge the
    angular-question candidate search uses.
    """

    from avengine.qa import angular_questions

    hidden = {"out_of_view", "fully_occluded"}
    series = _visibility_series(facts, actor_id)
    frames = [int(row.get("frame_index", -1)) for row in series]
    states = {int(row.get("frame_index", -1)): row.get("state") for row in series}
    pairs: list[dict[str, Any]] = []
    for frame in frames:
        if states.get(frame) not in hidden:
            continue
        visible_before = [
            index for index in frames if index < frame and states.get(index) in VISIBLE_STATES
        ]
        if not visible_before:
            continue
        anchor = max(visible_before)
        changed = angular_questions._hidden_motion_changed(facts, actor_id, anchor, frame)
        pairs.append({"anchor_frame": anchor, "query_frame": frame, "changed": bool(changed)})
    measured = {
        "observed_states_in_episode": sorted({str(value) for value in states.values() if value}),
        "hidden_query_pairs": pairs,
    }
    qualifying = [row for row in pairs if row["changed"]]
    if not pairs:
        return _row(
            condition,
            "fail",
            measured=measured,
            reason="the target is never hidden after being visible, so there is no hidden "
            "interval to change during",
        )
    return _row(
        condition,
        "pass" if qualifying else "fail",
        measured=measured if not qualifying else {**measured, "selected": qualifying[0]},
        reason=""
        if qualifying
        else "every hidden interval continues the last visible motion exactly, so a visual "
        "extrapolation already answers the question",
    )


def _check_motion_after_sound(
    condition, facts, actors, event, *, precision=DEFAULT_PUBLIC_TIME_PRECISION, **_: Any
) -> dict[str, Any]:
    actor_id = actors.get(condition.subject)
    if actor_id is None:
        return _row(condition, "not_run", reason="the target instance has no actor in these facts")
    if event is None:
        return _row(condition, "not_run", reason="no anchor event of the target was selected")
    windows = _legal_post_tail_windows(facts, event, "QA-17")
    if not windows:
        return _row(
            condition,
            "fail",
            measured={"legal_post_tail_windows": windows},
            reason="no silent post-sound window survives the measured wet tail and the "
            "other programmed events",
        )
    anchor_end = catalog._event_frame(event, "end_frame")
    fps = float(facts["time"]["frame_rate_hz"])
    if condition.evidence.get("derived_legal_window_nonempty"):
        return _row(
            condition,
            "pass",
            measured={
                "legal_post_tail_windows": [[int(a), int(b)] for a, b in windows],
                "anchor_end_frame": int(anchor_end),
                "wet_tail_end_s": _max_wet_tail_end_s(facts, event.get("event_id")),
            },
        )
    unavailable = False

    def value_for(frame: int) -> Any:
        nonlocal unavailable
        value = _any_motion_between(facts, actor_id, anchor_end, frame)
        if value is None:
            unavailable = True
            return None
        return "yes" if value else "no"

    runs = _stable_runs(windows, value_for)
    if unavailable:
        return _row(condition, "not_run", reason="per-frame motion readback is unavailable")
    for run in runs:
        run["public_s"] = _displayable(run, fps, precision)
    measured = {
        "legal_post_tail_windows": [[int(a), int(b)] for a, b in windows],
        "anchor_end_frame": int(anchor_end),
        "wet_tail_end_s": _max_wet_tail_end_s(facts, event.get("event_id")),
        "public_time_precision": precision,
        "stable_runs": runs,
    }
    if not runs:
        return _row(
            condition,
            "fail",
            measured=measured,
            reason="the answer never holds for two consecutive legal frames, so no stable "
            "question exists",
        )
    expected = condition.detail.get("any_motion_in_interval")
    wanted = None if expected is None else ("yes" if expected else "no")
    matching = [run for run in runs if wanted is None or run["value"] == wanted]
    if not matching:
        observed = sorted({str(run["value"]) for run in runs})
        return _row(
            condition,
            "fail",
            measured=measured,
            reason=f"the stable answers available here are {observed}; this branch needs {wanted}",
        )
    publishable = [run for run in matching if run["public_s"] is not None]
    if not publishable:
        return _row(
            condition,
            "fail",
            measured=measured,
            reason="the runs that carry this answer are too short to state at the public "
            "time precision",
        )
    return _row(condition, "pass", measured={**measured, "selected_run": publishable[0]})


def _check_motion_window_placement(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    actor_id = actors.get(condition.subject)
    if actor_id is None:
        return _row(condition, "not_run", reason="the target instance has no actor in these facts")
    runs = _moving_runs(facts, actor_id)
    placement = condition.detail.get("placement")
    fps = float(facts["time"]["frame_rate_hz"])
    if placement == MOTION_DURING_EVENT:
        if event is None:
            return _row(condition, "not_run", reason="no bound event of the target was selected")
        start, end = _event_frames(facts, event)
        overlap = [run for run in runs if run[0] < end and run[1] > start]
        measured = {"event_frames": [start, end], "moving_runs": runs, "overlapping_runs": overlap}
        return _row(
            condition,
            "pass" if overlap else "fail",
            measured=measured,
            reason=""
            if overlap
            else "no moving run intersects the audible window, so sound activity and root "
            "displacement do not overlap",
        )
    if placement == MOTION_AFTER_TAIL:
        if event is None:
            return _row(condition, "not_run", reason="no anchor event of the target was selected")
        tail_end = _max_wet_tail_end_s(facts, event.get("event_id"))
        if tail_end is None:
            return _row(
                condition,
                "not_run",
                reason="this event has no measured wet-tail interval, so the window cannot be placed",
            )
        first_allowed = math.ceil(tail_end * fps) + 1
        offenders = [run for run in runs if run[0] < first_allowed]
        measured = {
            "wet_tail_end_s": tail_end,
            "first_allowed_motion_frame": first_allowed,
            "moving_runs": runs,
            "runs_before_allowed": offenders,
        }
        if not runs:
            return _row(
                condition,
                "fail",
                measured=measured,
                reason="the target never moves, so no post-tail state change exists",
            )
        return _row(
            condition,
            "pass" if not offenders else "fail",
            measured=measured,
            reason=""
            if not offenders
            else "movement starts inside the measured wet tail, so the post-sound question "
            "is not silent",
        )
    return _row(condition, "not_run", reason=f"unknown motion window placement {placement!r}")


def _check_distance_net_change(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    actor_id = actors.get(condition.subject)
    if actor_id is None:
        return _row(condition, "not_run", reason="the target instance has no actor in these facts")
    if event is None:
        return _row(condition, "not_run", reason="no bound event of the target was selected")
    count = int(facts["time"]["frame_count"])
    start = max(0, catalog._event_frame(event, "start_frame"))
    end = min(count - 1, max(start + 1, catalog._event_frame(event, "end_frame") - 1))
    endpoints = _distances_over(facts, actor_id, [start, end])
    if endpoints is None:
        return _row(condition, "not_run", reason="listener-relative distance is not derivable")
    delta = endpoints[1] - endpoints[0]
    margin = float(condition.detail.get("margin_m") or DISTANCE_MARGIN_M)
    expected_sign = condition.detail.get("sign")
    series = _distances_over(facts, actor_id, range(start, end + 1)) or []
    steps = [b - a for a, b in zip(series, series[1:])]
    monotone = bool(steps) and (
        all(step <= 1.0e-9 for step in steps) or all(step >= -1.0e-9 for step in steps)
    )
    measured = {
        "endpoint_frames": [start, end],
        "distance_start_m": endpoints[0],
        "distance_end_m": endpoints[1],
        "endpoint_delta_m": delta,
        "margin_m": margin,
        # Reported next to the judged value, never in place of it: the shipped
        # judge only compares two endpoints, so monotonicity is extra evidence.
        "monotonic_over_interval": monotone,
        "judge_is_endpoint_only": True,
    }
    if abs(delta) < margin:
        return _row(
            condition,
            "fail",
            measured=measured,
            reason=f"the net listener distance changes by {delta:+.4f} m, under the judged "
            f"{margin} m margin",
        )
    actual_sign = "negative" if delta < 0 else "positive"
    status = "pass" if expected_sign in (None, actual_sign) else "fail"
    return _row(
        condition,
        status,
        measured=measured,
        reason=""
        if status == "pass"
        else f"the net change is {actual_sign}; this branch needs {expected_sign}",
    )


def _check_distance_stable_after_sound(
    condition, facts, actors, event, *, precision=DEFAULT_PUBLIC_TIME_PRECISION, **_: Any
) -> dict[str, Any]:
    actor_id = actors.get(condition.subject)
    if actor_id is None or event is None:
        return _row(condition, "not_run", reason="no resolved target and anchor event")
    windows = _legal_post_tail_windows(facts, event, "QA-16")
    if not windows:
        return _row(condition, "fail", reason="no silent post-sound window survives the wet tail")
    anchor_frame = catalog._event_frame(event, "end_frame")
    reference = _distances_over(facts, actor_id, [anchor_frame])
    if reference is None:
        return _row(condition, "not_run", reason="anchor distance is not derivable")
    margin = float(condition.detail.get("margin_m") or DISTANCE_MARGIN_M)
    fps = float(facts["time"]["frame_rate_hz"])
    unavailable = False

    def value_for(frame: int) -> Any:
        nonlocal unavailable
        values = _distances_over(facts, actor_id, [frame])
        if values is None:
            unavailable = True
            return None
        delta = values[0] - reference[0]
        if abs(delta) < margin:
            return "under_margin"
        return "nearer" if delta < 0 else "farther"

    runs = _stable_runs(windows, value_for)
    if unavailable:
        return _row(condition, "not_run", reason="post-sound distance is not derivable")
    for run in runs:
        run["public_s"] = _displayable(run, fps, precision)
    measured = {
        "anchor_frame": int(anchor_frame),
        "anchor_distance_m": reference[0],
        "margin_m": margin,
        "public_time_precision": precision,
        "stable_runs": runs,
    }
    usable = [
        run for run in runs if run["value"] in {"nearer", "farther"} and run["public_s"] is not None
    ]
    if not usable:
        return _row(
            condition,
            "fail",
            measured=measured,
            reason="no publishable run holds one distance trend at the judged margin across "
            "the whole interval",
        )
    return _row(condition, "pass", measured={**measured, "selected_run": usable[0]})


def _check_visibility_state(condition, facts, actors, event, *, query_frame=None, **_: Any) -> dict[str, Any]:
    actor_id = actors.get(condition.subject)
    if actor_id is None:
        return _row(condition, "not_run", reason="the subject has no actor in these facts")
    allowed = condition.evidence.get("visibility_state_in") or condition.evidence.get(
        "query_visibility_state_in"
    )
    wanted = condition.evidence.get("visibility_state") or condition.evidence.get(
        "final_visibility_state"
    )
    at = condition.detail.get("at")
    frame = query_frame
    if at == "final_frame":
        frame = int(facts["time"]["frame_count"]) - 1
    elif frame is None and event is not None:
        frame = max(0, catalog._event_frame(event, "start_frame"))
    observed_states = sorted({
        str(row.get("state")) for row in _visibility_series(facts, actor_id) if row.get("state")
    })
    if condition.search_frames and query_frame is None and allowed:
        hits = [
            int(row.get("frame_index", -1))
            for row in _visibility_series(facts, actor_id)
            if row.get("state") in set(allowed)
        ]
        measured = {
            "searched_whole_clip": True,
            "frames_in_allowed_states": hits,
            "observed_states_in_episode": observed_states,
        }
        return _row(
            condition,
            "pass" if hits else "fail",
            measured=measured,
            reason=""
            if hits
            else f"no frame of this episode puts the target in {sorted(allowed)}, so no "
            "legal query moment exists",
        )
    if frame is None:
        return _row(
            condition,
            "not_run",
            measured={"observed_states": observed_states},
            reason="no query frame is available for this visibility question",
        )
    state = _visibility_state_at(facts, actor_id, int(frame))
    measured = {"frame": int(frame), "state": state, "observed_states_in_episode": observed_states}
    if state is None:
        return _row(
            condition, "not_run", measured=measured, reason="no pixel visibility state at this frame"
        )
    if wanted is not None:
        status = "pass" if state == wanted else "fail"
        return _row(
            condition,
            status,
            measured=measured,
            reason="" if status == "pass" else f"the pixel state is {state}; this branch needs {wanted}",
        )
    if allowed:
        status = "pass" if state in set(allowed) else "fail"
        return _row(
            condition,
            status,
            measured=measured,
            reason="" if status == "pass" else f"the pixel state is {state}, outside {sorted(allowed)}",
        )
    return _row(condition, "pass", measured=measured)


def _check_entry_transition(
    condition, facts, actors, event, *, precision=DEFAULT_PUBLIC_TIME_PRECISION, **_: Any
) -> dict[str, Any]:
    actor_id = actors.get(condition.subject)
    if actor_id is None:
        return _row(condition, "not_run", reason="the subject has no actor in these facts")
    meta = facts.get("visibility_meta") or {}
    resolution = meta.get("resolution_hw") if isinstance(meta, Mapping) else None
    series = _visibility_series(facts, actor_id)
    observed = sorted({str(row.get("state")) for row in series if row.get("state")})
    if not isinstance(resolution, Sequence) or len(resolution) != 2:
        return _row(
            condition,
            "not_run",
            measured={"observed_states": observed},
            reason="pixel-truth resolution is absent, so an entry side cannot be measured",
        )
    width = float(resolution[1])
    center = (width - 1.0) / 2.0
    dead_zone = max(1.0, width * 0.02)
    wanted = condition.detail.get("side")
    found: list[dict[str, Any]] = []
    for previous, current in zip(series, series[1:]):
        if int(current.get("frame_index", -1)) != int(previous.get("frame_index", -2)) + 1:
            continue
        if previous.get("state") != "out_of_view" or current.get("state") not in VISIBLE_STATES:
            continue
        centroid = current.get("target_centroid_xy_px")
        if not isinstance(centroid, Sequence) or len(centroid) != 2:
            continue
        offset = float(centroid[0]) - center
        if abs(offset) <= dead_zone:
            continue
        entry_frame = int(current["frame_index"])
        window = catalog._entry_transition_window(
            facts, actor_id, entry_frame, center=center, dead_zone=dead_zone
        )
        public = None
        if window is not None:
            fps = float(facts["time"]["frame_rate_hz"])
            public = integer_second_window(
                int(window[0]) / fps, int(window[1]) / fps, precision=precision
            )
        found.append(
            {
                "entry_frame": entry_frame,
                "side": "right" if offset > 0.0 else "left",
                "centroid_offset_px": offset,
                "transition_window": None if window is None else [int(window[0]), int(window[1])],
                "public_s": None if public is None else list(public),
            }
        )
    measured = {
        "observed_states_in_episode": observed,
        "side_dead_zone_px": dead_zone,
        "entries": found,
    }
    if not found:
        return _row(
            condition,
            "fail",
            measured=measured,
            reason="no out_of_view to visible crossing with an unambiguous side is present",
        )
    matching = [row for row in found if wanted is None or row["side"] == wanted]
    if not matching:
        return _row(
            condition,
            "fail",
            measured=measured,
            reason=f"the observed entries are {[row['side'] for row in found]}, not {wanted}",
        )
    publishable = [row for row in matching if row["public_s"] is not None]
    if not publishable:
        return _row(
            condition,
            "fail",
            measured=measured,
            reason="the entry transition holds one side for too few frames to state as a "
            "public interval",
        )
    return _row(condition, "pass", measured={**measured, "selected_entry": publishable[0]})


def _check_occlusion_transition(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    actor_id = actors.get(condition.subject)
    if actor_id is None:
        return _row(condition, "not_run", reason="the subject has no actor in these facts")
    series = _visibility_series(facts, actor_id)
    observed = sorted({str(row.get("state")) for row in series if row.get("state")})
    measured: dict[str, Any] = {"observed_states_in_episode": observed}
    transition = condition.detail.get("transition")
    if transition == "visible_occluded_to_visible_clear":
        found = [
            int(current.get("frame_index", -1))
            for previous, current in zip(series, series[1:])
            if previous.get("state") == "visible_occluded" and current.get("state") == "visible_clear"
        ]
        measured["transition_frames"] = found
        return _row(
            condition,
            "pass" if found else "fail",
            measured=measured,
            reason="" if found else "no visible_occluded to visible_clear transition is present",
        )
    fully = [int(row.get("frame_index", -1)) for row in series if row.get("state") == "fully_occluded"]
    later_visible = [
        int(row.get("frame_index", -1))
        for row in series
        if row.get("state") in VISIBLE_STATES
        and any(frame < int(row.get("frame_index", 0)) for frame in fully)
    ]
    measured["fully_occluded_frames"] = fully
    measured["reappeared_frames"] = later_visible
    if not fully:
        return _row(
            condition,
            "fail",
            measured=measured,
            reason="the target is never fully occluded in the pixel readback, and a blocked "
            "line-of-sight ray is not the same measurement",
        )
    expected = bool(condition.evidence.get("visible_after_full_occlusion"))
    actual = bool(later_visible)
    status = "pass" if actual == expected else "fail"
    return _row(
        condition,
        status,
        measured=measured,
        reason="" if status == "pass" else f"the target {'did' if actual else 'did not'} reappear",
    )


def _check_visibility_coverage(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    actor_id = actors.get(condition.subject)
    if actor_id is None:
        return _row(condition, "not_run", reason="the subject has no actor in these facts")
    complete = catalog._visibility_is_complete(facts, actor_id)
    return _row(
        condition,
        "pass" if complete else "fail",
        measured={"complete": complete, "frame_count": int(facts["time"]["frame_count"])},
        reason=""
        if complete
        else "a negative reappearance answer needs an explicit visibility state on every frame",
    )


def _check_legal_integer_window(
    condition, facts, actors, event, *, precision=DEFAULT_PUBLIC_TIME_PRECISION, **_: Any
) -> dict[str, Any]:
    planned = int(condition.detail.get("public_time_precision") or 0)
    precision = int(precision)
    authority = str(condition.detail.get("window_authority") or "unspecified")
    if planned != precision:
        return _row(
            condition,
            "fail",
            measured={"planned_precision": planned, "episode_precision": precision},
            reason="this episode publishes intervals at a different precision than the "
            "condition was compiled for",
        )
    fps = float(facts["time"]["frame_rate_hz"])
    if authority == "whole_second_query_frame":
        from avengine.qa import angular_questions

        frames = angular_questions._whole_second_frames(facts)
        return _row(
            condition,
            "pass" if frames else "fail",
            measured={"public_time_precision": precision, "whole_second_frames": frames},
            reason=""
            if frames
            else "this clock has no whole-second frame, so an integer query moment cannot "
            "be stated",
        )
    if authority == "wet_tail_complement_window":
        windows = catalog._derived_legal_query_windows(facts, "QA-18")
        if not windows:
            return _row(
                condition,
                "fail",
                measured={"public_time_precision": precision},
                reason="no wet-tail complement window is derivable, so no legal query "
                "moment can be published",
            )
        rows = [
            {
                "frames": [int(a), int(b)],
                "public_s": None
                if integer_second_window(int(a) / fps, int(b) / fps, precision=precision) is None
                else list(integer_second_window(int(a) / fps, int(b) / fps, precision=precision)),
            }
            for a, b in windows
        ]
        ok = [row for row in rows if row["public_s"] is not None]
        return _row(
            condition,
            "pass" if ok else "fail",
            measured={"public_time_precision": precision, "windows": rows},
            reason=""
            if ok
            else "every wet-tail complement window is too short to state at this precision",
        )
    if authority != "derived_post_sound_window":
        return _row(
            condition,
            "not_run",
            measured={"public_time_precision": precision, "window_authority": authority},
            reason=f"the legal interval of this question comes from the {authority}, which "
            "this readback does not own; the owning check reports it",
        )
    if event is None:
        return _row(
            condition,
            "not_run",
            reason="the legal window of this question depends on an event that was not selected",
        )
    windows = _legal_post_tail_windows(facts, event, "QA-17")
    if not windows:
        return _row(
            condition,
            "not_run",
            reason="no derived legal window is available for this event, so the public "
            "interval cannot be checked here",
        )
    displayable = []
    for start, end in windows:
        bounds = integer_second_window(int(start) / fps, int(end) / fps, precision=precision)
        displayable.append(
            {
                "frames": [int(start), int(end)],
                "exact_s": [int(start) / fps, int(end) / fps],
                "public_s": None if bounds is None else list(bounds),
            }
        )
    ok = [row for row in displayable if row["public_s"] is not None]
    return _row(
        condition,
        "pass" if ok else "fail",
        measured={"public_time_precision": precision, "windows": displayable},
        reason=""
        if ok
        else "every legal window is too short to state at the configured public precision",
    )


def _check_source_activity(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    present = catalog._source_activity_present(facts)
    measured: dict[str, Any] = {
        "present": bool(present),
        "complete_flag": facts.get("source_activity_evidence_complete"),
    }
    if not present:
        return _row(
            condition,
            "fail",
            measured=measured,
            reason="no measured source-activity intervals, so a claim that this source was "
            "sounding across the span is unproven",
        )
    if event is None:
        return _row(condition, "pass", measured=measured)
    intervals = catalog._source_activity_for_event(facts, str(event.get("event_id")))
    measured["event_interval_count"] = len(intervals)
    if not intervals:
        return _row(
            condition,
            "fail",
            measured=measured,
            reason="the selected event carries no measured active interval of its own",
        )
    rate = float(facts["time"]["sample_rate_hz"])
    covered = sum(
        max(0, int(row.get("end_sample_exclusive", 0)) - int(row.get("start_sample", 0)))
        for row in intervals
    )
    span = max(1.0e-9, float(event["end_s"]) - float(event["start_s"]))
    measured["active_seconds"] = covered / rate
    measured["event_span_s"] = span
    # An event span is not evidence of continuous sounding: the measured active
    # coverage inside that span is the number a caller has to judge.
    measured["active_coverage"] = (covered / rate) / span
    return _row(condition, "pass", measured=measured)


def _check_wet_tail(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    audio = facts.get("audio") or {}
    tails = audio.get("wet_tail_intervals")
    measured: dict[str, Any] = {"interval_count": len(tails) if isinstance(tails, Sequence) else 0}
    if not isinstance(tails, Sequence) or not tails:
        return _row(condition, "fail", measured=measured, reason="no measured wet-tail intervals")
    if event is not None:
        end = _max_wet_tail_end_s(facts, event.get("event_id"))
        measured["event_wet_tail_end_s"] = end
        if end is None:
            return _row(
                condition,
                "fail",
                measured=measured,
                reason="the selected event has no wet-tail interval of its own",
            )
        try:
            pre = catalog._anchor_pre_silence(
                facts, event, minimum_seconds=float(condition.detail.get("anchor_pre_silence_s") or 0.1)
            )
        except catalog._Deferred as error:
            return _row(
                condition,
                "fail",
                measured={**measured, "judge_code": error.code},
                reason=f"{_JUDGE_SOURCE}:_anchor_pre_silence refused this anchor: {error.code}",
            )
        measured["pre_silence_s"] = pre["pre_silence_s"]
    return _row(condition, "pass", measured=measured)


def _check_candidate_domain(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    minimum = int(
        condition.evidence.get("distinct_labelled_candidates_at_least")
        or condition.detail.get("min_entities")
        or condition.detail.get("min_visible_candidates")
        or 1
    )
    reviewed = _appearance_candidate_count(facts)
    total = len(facts.get("actors") or {})
    measured = {"reviewed_appearance_candidates": reviewed, "actor_count": total, "minimum": minimum}
    return _row(
        condition,
        "pass" if total >= minimum else "fail",
        measured=measured,
        reason="" if total >= minimum else "this episode carries fewer entity instances than the question needs",
    )


def _check_answer_distinguishable(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    predicate = condition.detail.get("predicate")
    target_actor = actors.get(condition.subject)
    if target_actor is None:
        return _row(condition, "not_run", reason="the target instance has no actor in these facts")
    rivals = {
        instance: actor
        for instance, actor in actors.items()
        if actor is not None and actor != target_actor
    }
    if not rivals:
        return _row(
            condition,
            "not_run",
            reason="this episode has no competitor instance, so distinguishability is undefined",
        )
    if event is None:
        return _row(condition, "not_run", reason="no bound event of the target was selected")

    def value_for(actor_id: str) -> Any:
        if predicate == "motion_during_event":
            start, end = _event_frames(facts, event)
            try:
                moving = catalog._stable_motion_window(facts, actor_id, start, end)
            except catalog._Deferred:
                return None
            return "moving" if moving else "still"
        if predicate == "distance_trend":
            count = int(facts["time"]["frame_count"])
            start = max(0, catalog._event_frame(event, "start_frame"))
            end = min(count - 1, max(start + 1, catalog._event_frame(event, "end_frame") - 1))
            values = _distances_over(facts, actor_id, [start, end])
            if values is None:
                return None
            delta = values[1] - values[0]
            if abs(delta) < DISTANCE_MARGIN_M:
                return None
            return "nearer" if delta < 0 else "farther"
        if predicate == "motion_after_sound":
            windows = _legal_post_tail_windows(facts, event, "QA-17")
            if not windows:
                return None
            anchor_end = catalog._event_frame(event, "end_frame")
            last = int(windows[-1][1]) - 1
            value = _any_motion_between(facts, actor_id, anchor_end, last)
            return None if value is None else ("yes" if value else "no")
        return None

    target_value = value_for(target_actor)
    rival_values = {instance: value_for(actor) for instance, actor in rivals.items()}
    # An unmeasurable competitor is not a distractor: the shipped gate only
    # compares the real values it could compute.
    measurable = {key: value for key, value in rival_values.items() if value is not None}
    measured = {
        "predicate": predicate,
        "target": {condition.subject: target_value},
        "competitors": rival_values,
        "measurable_competitors": sorted(measurable),
    }
    if target_value is None:
        return _row(
            condition,
            "not_run",
            measured=measured,
            reason="the predicate is not measurable for the target in this window",
        )
    if not measurable:
        return _row(
            condition,
            "pass",
            measured=measured,
            reason="no competitor has a measurable value, so no real distractor collapses "
            "onto the gold answer",
        )
    shared = sorted(
        instance for instance, value in measurable.items() if value == target_value
    )
    differing = sorted(
        instance for instance, value in measurable.items() if value != target_value
    )
    measured["competitors_sharing_the_gold_answer"] = shared
    measured["competitors_answering_differently"] = differing
    # The shipped gate is avengine/qa/unified_catalog.py:distractors_equal_gold,
    # and it defers only when *every* measurable distractor equals gold. A mixed
    # group such as {target: no, animal: yes, device: no} keeps its answer form,
    # because the animal still separates the gold answer from the alternative.
    # Failing on any single shared value was stricter than the judge and threw
    # away legal human+animal+device items.
    if differing:
        return _row(condition, "pass", measured=measured)
    return _row(
        condition,
        "fail",
        measured=measured,
        reason=f"all measurable competitors share the target's answer {target_value!r} "
        f"({shared}); distractors_equal_gold would defer this form",
    )


def _check_bearing_reference(condition, facts, actors, event, *, query_frame=None, **_: Any) -> dict[str, Any]:
    actor_id = actors.get(condition.subject)
    if actor_id is None:
        return _row(condition, "not_run", reason="the subject has no actor in these facts")
    subset = condition.detail.get("subset")
    listener = facts.get("listener")
    measured: dict[str, Any] = {
        "subset": subset,
        "listener_status": (listener or {}).get("status") if isinstance(listener, Mapping) else None,
        "camera_calibration_present": bool(facts.get("camera_calibration")),
    }
    if subset == "V":
        if not facts.get("camera_calibration"):
            return _row(
                condition,
                "fail",
                measured=measured,
                reason="the V subset publishes a camera calibration and this episode has none",
            )
        from avengine.qa import angular_questions

        frames = angular_questions._whole_second_frames(facts)
        legal = []
        for frame in frames:
            try:
                angular_questions._visual_bearing(facts, actor_id, frame)
            except catalog._Deferred:
                continue
            legal.append(frame)
        measured["whole_second_frames"] = frames
        measured["visual_bearing_frames"] = legal
        return _row(
            condition,
            "pass" if legal else "fail",
            measured=measured,
            reason="" if legal else "no whole-second frame yields a visual bearing",
        )
    frame = query_frame
    if frame is None and event is not None:
        start, end = _event_frames(facts, event)
        frame = start + (end - start) // 2
    if frame is None:
        return _row(condition, "not_run", measured=measured, reason="no query frame for a bearing")
    try:
        azimuth = catalog._azimuth(facts, actor_id, int(frame))
    except catalog._Deferred as error:
        return _row(
            condition,
            "fail",
            measured={**measured, "judge_code": error.code},
            reason=f"{_JUDGE_SOURCE}:_azimuth refused frame {frame}: {error.code}",
        )
    measured["query_frame"] = int(frame)
    measured["azimuth_deg"] = azimuth
    measured["public_azimuth_deg"] = int(round(azimuth))
    return _row(condition, "pass", measured=measured)


def _check_sound_class_variety(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    minimum = int(condition.detail.get("distinct_sound_classes") or 2)
    classes = {
        str(row.get("sound_class"))
        for row in facts.get("events") or ()
        if isinstance(row, Mapping) and row.get("sound_class") and row.get("sound_class_explicit")
    }
    measured = {"explicit_sound_classes": sorted(classes), "minimum": minimum}
    return _row(
        condition,
        "pass" if len(classes) >= minimum else "fail",
        measured=measured,
        reason=""
        if len(classes) >= minimum
        else "fewer explicit sound classes than this question needs",
    )


def _check_event_statistics(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    try:
        events = catalog._bound_events(facts)
    except catalog._Deferred as error:
        return _row(condition, "fail", reason=f"bound events are unavailable: {error.code}")
    ids = [str(row.get("event_id")) for row in events]
    measured = {"event_count": len(ids), "distinct_event_ids": len(set(ids))}
    return _row(
        condition,
        "pass" if ids and len(set(ids)) == len(ids) else "fail",
        measured=measured,
        reason="" if ids and len(set(ids)) == len(ids) else "event ids are absent or not distinct",
    )


def _check_appearance_reference(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    reviewed = catalog._reviewed_appearances(facts) if facts.get("actors") else {}
    subject_actor = actors.get(condition.subject) if condition.subject else None
    values = {actor: dict(value) for actor, value in reviewed.items()}
    measured = {"reviewed": values}
    if subject_actor is not None:
        ok = subject_actor in reviewed
        return _row(
            condition,
            "pass" if ok else "fail",
            measured=measured,
            reason="" if ok else "the target has no reviewed appearance value to be named by",
        )
    labels = {(row.get("field"), row.get("value")) for row in reviewed.values()}
    ok = len(labels) == len(reviewed) and len(reviewed) >= 1
    return _row(
        condition,
        "pass" if ok else "fail",
        measured=measured,
        reason="" if ok else "the candidate appearance labels are absent or ambiguous",
    )


def _check_speech_content(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    transcripts = {
        str(row.get("event_id")): row.get("transcript")
        for row in facts.get("events") or ()
        if isinstance(row, Mapping) and row.get("transcript")
    }
    measured = {"events_with_transcript": sorted(transcripts)}
    if event is not None:
        ok = str(event.get("event_id")) in transcripts
        return _row(
            condition,
            "pass" if ok else "fail",
            measured=measured,
            reason="" if ok else "the selected event carries no recorded transcript",
        )
    return _row(
        condition,
        "pass" if transcripts else "fail",
        measured=measured,
        reason="" if transcripts else "no event carries a recorded transcript",
    )


def _check_occluder_identity(condition, facts, actors, event, *, query_frame=None, **_: Any) -> dict[str, Any]:
    actor_id = actors.get(condition.subject)
    if actor_id is None:
        return _row(condition, "not_run", reason="the subject has no actor in these facts")
    registry = facts.get("occluder_registry")
    frame = query_frame
    if frame is None and event is not None:
        frame = max(0, catalog._event_frame(event, "start_frame"))
    ids = catalog._occluder_ids(facts, actor_id, int(frame)) if frame is not None else []
    measured = {"frame": None if frame is None else int(frame), "occluder_ids": ids,
                "registry_present": bool(registry)}
    if not ids:
        return _row(
            condition, "fail", measured=measured, reason="no registered occluder covers the target"
        )
    if len(ids) != 1:
        return _row(
            condition,
            "fail",
            measured=measured,
            reason="more than one occluder covers the target, so its identity is ambiguous",
        )
    return _row(condition, "pass", measured=measured)


def _check_distance_margin_at_time(
    condition, facts, actors, event, *, precision=DEFAULT_PUBLIC_TIME_PRECISION, **_: Any
) -> dict[str, Any]:
    """Which of two named targets is nearer, held steadily over a public interval."""

    names = list(condition.detail.get("targets") or ())
    resolved = [actors.get(name) for name in names]
    if len(resolved) < 2 or any(actor is None for actor in resolved):
        return _row(
            condition,
            "not_run",
            measured={"targets": names},
            reason="this comparison needs two named instances that both resolve here",
        )
    first, second = resolved[0], resolved[1]
    margin = float(condition.detail.get("margin_m") or DISTANCE_MARGIN_M)
    count = int(facts["time"]["frame_count"])
    fps = float(facts["time"]["frame_rate_hz"])
    unavailable = False

    def value_for(frame: int) -> Any:
        nonlocal unavailable
        values = _distances_over(facts, first, [frame])
        others = _distances_over(facts, second, [frame])
        if values is None or others is None:
            unavailable = True
            return None
        if _visibility_state_at(facts, first, frame) not in VISIBLE_STATES:
            return "target_not_observable"
        if _visibility_state_at(facts, second, frame) not in VISIBLE_STATES:
            return "target_not_observable"
        gap = values[0] - others[0]
        if abs(gap) < margin:
            return "under_margin"
        return names[0] if gap < 0 else names[1]

    runs = _stable_runs([[0, count]], value_for)
    if unavailable:
        return _row(condition, "not_run", reason="listener-relative distance is not derivable")
    for run in runs:
        run["public_s"] = _displayable(run, fps, precision)
    usable = [
        run
        for run in runs
        if run["value"] in set(names) and run["public_s"] is not None
    ]
    measured = {"targets": names, "margin_m": margin, "stable_runs": runs,
                "public_time_precision": precision}
    return _row(
        condition,
        "pass" if usable else "fail",
        measured=measured if not usable else {**measured, "selected_run": usable[0]},
        reason=""
        if usable
        else "no publishable interval keeps one target nearer by the judged margin while "
        "both stay observable",
    )


def _check_role_assignment(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    """Confirm the named instances exist here and mirror a semantic refusal."""

    named = list(condition.detail.get("targets") or ()) + list(
        condition.detail.get("competitors") or ()
    )
    if condition.subject is not None:
        named = [condition.subject]
    resolved = {name: actors.get(name) for name in named}
    missing = sorted(name for name, actor in resolved.items() if actor is None)
    measured: dict[str, Any] = {"resolved": resolved}
    if missing:
        return _row(
            condition,
            "fail",
            measured=measured,
            reason=f"these named instances have no actor in this episode: {missing}",
        )
    if condition.state != STATE_AVAILABLE:
        return _row(
            condition,
            "fail",
            measured={**measured, "compiled_state": condition.state},
            reason=condition.reason or f"this role is {condition.state} for the named target",
        )
    if condition.detail.get("requires_self_locomotion"):
        actor_id = resolved[condition.subject]
        row = facts.get("actors", {}).get(actor_id, {})
        series = row.get("moving") if isinstance(row, Mapping) else None
        measured["motion_series_frames"] = len(series) if isinstance(series, Sequence) else 0
        measured["moving_frames"] = (
            sum(1 for value in series if bool(value)) if isinstance(series, Sequence) else 0
        )
        if not isinstance(series, Sequence) or not series:
            return _row(
                condition,
                "fail",
                measured=measured,
                reason="this episode has no per-frame motion readback for the target",
            )
    if condition.detail.get("requires_anchor_role"):
        # Facts do not carry the planner role, so the readback confirms only
        # that the instance exists and states where the role has to be recorded.
        measured["anchor_role_authority"] = "request qa_target/profile, not the fact table"
    return _row(condition, "pass", measured=measured)


def _check_required_modality(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    wanted = list(condition.detail.get("modalities") or ())
    audio = facts.get("audio") or {}
    channels = audio.get("channel_count")
    delivered = []
    if facts.get("visibility"):
        delivered.append("video")
    if audio.get("status") == "pass":
        delivered.append("audio")
        if isinstance(channels, int) and channels >= 2:
            delivered.append("binaural_audio")
    if facts.get("time"):
        delivered.append("time")
    if facts.get("visibility_meta"):
        delivered.append("pixel_visibility")
    if facts.get("occluder_evidence") or facts.get("occluder_registry"):
        delivered.append("pixel_instance_visibility")
    missing = [item for item in wanted if item not in delivered]
    measured = {"required": wanted, "delivered": sorted(set(delivered)), "channel_count": channels}
    return _row(
        condition,
        "pass" if not missing else "fail",
        measured=measured,
        reason="" if not missing else f"these modalities are absent from the delivery: {missing}",
    )


def _check_event_selection(condition, facts, actors, event, **_: Any) -> dict[str, Any]:
    if condition.detail.get("selector", {}).get("kind") == "whole_clip" or condition.key in {
        "whole_clip_statistics"
    }:
        return _check_event_statistics(condition, facts, actors, event)
    if event is None:
        return _row(
            condition, "fail", reason="no bound event of the target matches the stated selector"
        )
    return _row(
        condition,
        "pass",
        measured={
            "event_id": str(event.get("event_id")),
            "actor_id": str(event.get("actor_id")),
            "start_s": float(event["start_s"]),
            "end_s": float(event["end_s"]),
        },
    )


_CHECKERS: dict[str, Callable[..., dict[str, Any]]] = {
    "motion_during_event": _check_motion_during_event,
    "motion_after_sound": _check_motion_after_sound,
    "motion_window_placement": _check_motion_window_placement,
    "distance_net_change": _check_distance_net_change,
    "distance_stable_after_sound": _check_distance_stable_after_sound,
    "distance_margin_at_time": _check_distance_margin_at_time,
    "visibility_state": _check_visibility_state,
    "entry_transition": _check_entry_transition,
    "occlusion_transition": _check_occlusion_transition,
    "visibility_coverage_complete": _check_visibility_coverage,
    "legal_integer_query_window": _check_legal_integer_window,
    "source_activity_readback": _check_source_activity,
    "wet_tail_readback": _check_wet_tail,
    "candidate_domain": _check_candidate_domain,
    "answer_distinguishable": _check_answer_distinguishable,
    "bearing_reference": _check_bearing_reference,
    "sound_class_variety": _check_sound_class_variety,
    "event_statistics_window": _check_event_statistics,
    "appearance_reference": _check_appearance_reference,
    "speech_content": _check_speech_content,
    "occluder_identity": _check_occluder_identity,
    "required_modality": _check_required_modality,
    "role_assignment": _check_role_assignment,
    "event_selection": _check_event_selection,
}


def check_conditions(
    compiled: CompiledConditions,
    facts: Mapping[str, Any],
    *,
    actor_by_instance: Mapping[str, str] | None = None,
    query_frame: int | None = None,
) -> dict[str, Any]:
    """Prove the compiled conditions against one episode's real facts.

    Every event of the named target that the stated selector admits is checked,
    and the result names which event, if any, satisfies the whole set.  The
    checks call the catalog's own predicates, so a passing report here means the
    same statement the question generator will make, not an adjacent one.
    """

    if not isinstance(facts, Mapping) or "time" not in facts:
        raise GenerationConditionError("facts must be a normalized episode fact mapping")
    try:
        precision = catalog._time_display_precision(facts)
    except catalog._Deferred as error:
        raise GenerationConditionError(
            f"this episode declares an unusable public time precision: {error.code}"
        ) from error
    actors = {
        subject.entity_instance_id: _resolve_actor(facts, subject, actor_by_instance)
        for subject in compiled.subjects
    }
    unresolved = sorted(key for key, value in actors.items() if value is None)
    target = _target(compiled.subjects)
    target_actor = actors.get(target.entity_instance_id)
    candidate_events: list[Mapping[str, Any] | None]
    if target_actor is None:
        candidate_events = [None]
    else:
        candidate_events = list(_target_events(facts, target_actor, compiled.event)) or [None]

    attempts: list[dict[str, Any]] = []
    for event in candidate_events:
        rows: list[dict[str, Any]] = []
        for condition in compiled.conditions:
            checker = _CHECKERS.get(condition.kind)
            if checker is None:
                rows.append(
                    _row(
                        condition,
                        "not_run",
                        reason=f"no readback check is implemented for kind {condition.kind!r}",
                    )
                )
                continue
            rows.append(
                checker(
                    condition,
                    facts,
                    actors,
                    event,
                    query_frame=query_frame,
                    precision=precision,
                )
            )
        failures = [row for row in rows if row["status"] == "fail"]
        skipped = [row for row in rows if row["status"] == "not_run"]
        attempts.append(
            {
                "event_id": None if event is None else str(event.get("event_id")),
                "status": "fail" if failures else ("incomplete" if skipped else "pass"),
                "failed": [row["key"] for row in failures],
                "not_run": [row["key"] for row in skipped],
                "checks": rows,
            }
        )

    passing = [row for row in attempts if row["status"] == "pass"]
    incomplete = [row for row in attempts if row["status"] == "incomplete"]
    if passing:
        status = "pass"
    elif incomplete:
        status = "incomplete"
    else:
        status = "fail"
    readback_kinds = {"source_activity_readback", "wet_tail_readback", "visibility_state",
                      "visibility_coverage_complete", "required_modality"}
    readback_rows = [
        row
        for attempt in attempts
        for row in attempt["checks"]
        if row["kind"] in readback_kinds
    ]
    if not readback_rows:
        native_evidence = "not_checked"
    elif any(row["status"] == "fail" for row in readback_rows):
        native_evidence = "missing"
    elif any(row["status"] == "not_run" for row in readback_rows):
        native_evidence = "incomplete"
    else:
        native_evidence = "present"
    return {
        "qa_id": compiled.qa_id,
        "branch": compiled.branch,
        "compiled_state": compiled.state,
        "status": status,
        "state_layers": {
            "planning_support": compiled.state,
            "planning_support_declared_by": (
                f"{compiled.capabilities.source}@{compiled.capabilities.version}"
            ),
            "native_evidence": native_evidence,
            "condition_met": status,
        },
        "public_time_precision": precision,
        "episode_id": facts.get("episode_id"),
        "actor_by_instance": {key: value for key, value in actors.items()},
        "unresolved_instances": unresolved,
        "selected_event_id": passing[0]["event_id"] if passing else None,
        "checks_not_run": sorted({key for row in attempts for key in row["not_run"]}),
        "attempts": attempts,
    }

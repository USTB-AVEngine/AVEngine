"""Validate binding episodes against the caller's existing QA request."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]


class BindingConditionError(ValueError):
    pass


def _read(path, cache):
    path = Path(path).resolve()
    key = ("binding_input_json", str(path))
    if key not in cache:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise BindingConditionError(f"expected JSON object: {path}")
        cache[key] = value
    return cache[key]


def _file(value, base):
    if not isinstance(value, str) or not value:
        raise BindingConditionError("required native source path is missing")
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else base / path).resolve()
    if not path.is_file():
        raise BindingConditionError(f"required native source is absent: {path}")
    return path


# The crop identity a member actually used. Every field here is already
# registered on the sound-pool row, so this records existing identity rather
# than adding a new content contract. Two members that declare shared audio
# have to agree on all of it: one clip, one crop, one processing result.
_CLIP_IDENTITY_FIELDS = (
    "prepared_audio_id", "sound_identity_id", "source_relative", "source_sha256",
    "source_crop_start_sample", "source_crop_end_sample_exclusive",
    "source_offset_s", "source_crop_duration_s", "source_normalization",
    "sample_count", "sample_rate_hz",
)


def _clip_identity(sound: Mapping, *, actor_id: str, rendered_slice: Mapping) -> dict:
    """Describe one rendered event by the clip and crop that produced it."""
    identity = {"sound_asset_id": str(sound["sound_asset_id"]), "actor_id": actor_id,
                "rendered_slice": dict(rendered_slice)}
    missing = []
    for field in _CLIP_IDENTITY_FIELDS:
        if field in sound:
            identity[field] = sound[field]
        else:
            missing.append(field)
    # A pool row written before segment cropping existed carries no crop
    # coordinates. Say so instead of implying the crop was verified.
    identity["crop_provenance"] = "sound_pool_row" if not missing else "partial"
    if missing:
        identity["crop_fields_absent_in_pool_row"] = missing
    return identity


def validate_binding_episode(facts: Mapping, *, request: Mapping, base: Path, cache: dict):
    """Check actual poses and used audio inputs, without changing a request."""
    from avengine.capture.neutral_readback import validate_neutral_readback
    from avengine.rooms.conditioned_sampler import sound_matches

    if not isinstance(request, Mapping):
        raise BindingConditionError("binding groups require their caller's QA request")
    clock = facts["time"]
    for key in ("frame_count", "frame_rate_hz", "sample_rate_hz"):
        if key not in request or not math.isclose(float(clock[key]), float(request[key]), abs_tol=1e-9):
            raise BindingConditionError(f"native episode violates requested {key}")
    duration = float(request["frame_count"]) / float(request["frame_rate_hz"])
    if not math.isclose(float(clock["duration_seconds"]), duration, abs_tol=1e-9):
        raise BindingConditionError("native episode violates requested duration")
    camera_request = request.get("camera", {})
    if camera_request.get("motion") != "static":
        raise BindingConditionError("binding dataset requires a fixed camera")
    required_tail = request.get("profile", {}).get("reserve_tail_s")
    if isinstance(required_tail, bool) or not isinstance(required_tail, (int, float)) or not math.isfinite(required_tail) or required_tail < 0:
        raise BindingConditionError("request must declare a nonnegative reserve_tail_s")
    sources = facts["source_paths"]
    plan_path = _file(sources.get("plan"), base)
    plan = _read(plan_path, cache)
    native_path = _file(sources.get("neutral_readback"), base)
    native = _read(native_path, cache)
    validate_neutral_readback(native, plan=plan)
    motion_readback = {}
    motion_request = request.get("binding_motion") or request.get("binding_identity") or {}
    speed_range = motion_request.get("walk_speed_range_mps")
    if speed_range is not None:
        if (len(speed_range) != 2 or not all(math.isfinite(float(value)) for value in speed_range)
                or not 0 < speed_range[0] <= speed_range[1]):
            raise BindingConditionError("requested walk speeds must be finite, positive and ordered")
        for actor_id, rows in native["entities"].items():
            points = np.asarray([row["root"] for row in rows], dtype=float)
            speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) * float(clock["frame_rate_hz"])
            moving_speeds = speeds[speeds > .05]
            if not len(moving_speeds):
                continue
            minimum, maximum_speed = float(moving_speeds.min()), float(moving_speeds.max())
            if minimum < speed_range[0] - 1e-5 or maximum_speed > speed_range[1] + 1e-5:
                raise BindingConditionError(
                    f"native walk speed differs from the request for {actor_id}: {minimum}..{maximum_speed}")
            motion_readback[actor_id] = {"minimum_mps": minimum, "maximum_mps": maximum_speed,
                                         "moving_step_count": int(len(moving_speeds))}
    camera = native["camera"]
    first = camera[0]
    maximum = 0.0
    for row in camera:
        for owner, a, b in [
            ("position", row["position_m"], first["position_m"]),
            *[(axis, row["basis"][axis], first["basis"][axis]) for axis in ("forward", "right", "up")],
        ]:
            delta = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            maximum = max(maximum, delta)
            if delta > 1e-5:
                raise BindingConditionError(f"native camera {owner} changes during the episode")
    visual = plan.get("visual_plan", {})
    planned_camera = visual.get("camera", {})
    fov = planned_camera.get("horizontal_fov_deg")
    if not isinstance(fov, (int, float)) or not math.isfinite(fov):
        raise BindingConditionError("capture plan lacks a finite field of view")
    if camera_request.get("fov_deg") is not None and not math.isclose(float(fov), float(camera_request["fov_deg"]), abs_tol=1e-6):
        raise BindingConditionError("capture field of view differs from request")
    for frame in visual.get("frames", []):
        current = frame.get("camera_state", {}).get("horizontal_fov_deg", fov)
        if not math.isclose(float(current), float(fov), abs_tol=1e-6):
            raise BindingConditionError("capture field of view changes during the episode")
    resolution = camera_request.get("resolution_hw")
    if resolution is not None and list(resolution) != facts.get("visibility_meta", {}).get("resolution_hw"):
        raise BindingConditionError("native resolution differs from request")
    planned_request = plan.get("request", {})
    actual_tail_request = planned_request.get("profile", {}).get("reserve_tail_s")
    if actual_tail_request is None or not math.isclose(float(actual_tail_request), float(required_tail), abs_tol=1e-9):
        raise BindingConditionError("producer changed the caller's reserve_tail_s")
    latest_dry_end = max(float(event["end_s"]) for event in facts["events"])
    if latest_dry_end + float(required_tail) > duration + 1 / float(clock["sample_rate_hz"]):
        raise BindingConditionError("sound schedule exceeds the requested tail budget")
    expected_count = request.get("entities", {}).get("total_count")
    if expected_count is not None and int(expected_count) != len(facts["actors"]):
        raise BindingConditionError("native entity count differs from request")
    pool_path = _file(request.get("sound_pool"), REPOSITORY)
    pool = _read(pool_path, cache)
    sounds = {str(row["sound_asset_id"]): row for row in pool.get("sounds", [])}
    registry_path = _file(request.get("source_registry"), REPOSITORY)
    registry = {str(row["asset_id"]): row for row in _read(registry_path, cache).get("assets", [])}
    report_path = _file(sources.get("research_report"), base)
    report = _read(report_path, cache)
    expected_source_policy = request.get("source_context_policy", "joint")
    used_source_policy = report.get("inputs", {}).get("audio_render_config", {}).get("source_context_policy", "joint")
    if used_source_policy != expected_source_policy:
        raise BindingConditionError("rendered source context policy differs from the request")
    expected_gain = request.get("post_assembly_convolution_gain")
    used_gain = report.get("gain_application", {}).get("post_assembly_convolution_gain")
    if used_gain is None:
        used_gain = report.get("audio", {}).get("post_assembly_convolution_gain")
    if not isinstance(expected_gain, (int, float)) or used_gain is None or not math.isclose(float(used_gain), float(expected_gain), abs_tol=1e-9):
        raise BindingConditionError("rendered gain differs from the declared batch gain")
    used_assets = report.get("inputs", {}).get("dry_assets", {})
    report_events = {str(event["event_id"]): event for event in report.get("events", [])}
    used_ids = []
    audio_input_identity = {}
    for event in facts["events"]:
        sound_id = str(event["sound_asset_id"])
        sound = sounds.get(sound_id)
        if sound is None:
            raise BindingConditionError(f"rendered sound is absent from current sound pool: {sound_id}")
        actor = facts["actors"][event["actor_id"]]
        asset_id = str(actor["asset_id"])
        registered = registry.get(asset_id)
        allowed = sound.get("compatible_asset_ids")
        if registered is None or not sound_matches(registered, sound) or (allowed and asset_id not in allowed):
            raise BindingConditionError(f"sound is incompatible with physical source: {sound_id} -> {asset_id}")
        actual = used_assets.get(sound_id)
        actual_path = actual.get("path") if isinstance(actual, Mapping) else actual
        expected_path = sound.get("path") or sound.get("prepared")
        if _file(actual_path, report_path.parent) != _file(expected_path, pool_path.parent):
            raise BindingConditionError(f"rendered dry asset differs from the current pool: {sound_id}")
        used_event = report_events.get(str(event["event_id"]))
        if used_event is None:
            raise BindingConditionError("native audio report has no matching event")
        interval = used_event.get("source_slice", {})
        start = interval.get("start_sample", used_event.get("source_start_sample"))
        end = interval.get("end_sample_exclusive", used_event.get("source_end_sample_exclusive"))
        if start != 0 or end != int(sound["sample_count"]):
            raise BindingConditionError("rendered event does not preserve its complete prepared clip")
        used_ids.append(sound_id)
        audio_input_identity[str(event["event_id"])] = _clip_identity(
            sound, actor_id=str(event["actor_id"]),
            rendered_slice={"start_sample": int(start), "end_sample_exclusive": int(end)})
    return {
        "status": "pass", "duration_seconds": duration, "frame_count": int(clock["frame_count"]),
        "camera_motion": "static", "maximum_camera_component_drift": maximum,
        "native_motion_speed_readback": motion_readback,
        "field_of_view_deg": fov, "reserve_tail_s": required_tail,
        "available_tail_s": duration - latest_dry_end, "sound_pool": str(pool_path),
        "sound_asset_ids": used_ids, "post_assembly_convolution_gain": expected_gain,
        "audio_input_identity": audio_input_identity,
    }


# --------------------------------------------------------------------------- task families

TASK_FAMILIES = (
    "visible_binding",
    "visual_conditioned_relation",
    "cross_event_identity",
    "cross_time_state",
)

#: Which catalog question each core recipe instantiates. One place, so the
#: question builder, the assembler and the export all name the same QA id.
TASK_QA_IDS = {
    "visible_binding": "QA-20",
    "visual_conditioned_relation": "QA-05",
    "cross_event_identity": "QA-20",
    "cross_time_state": "QA-13",
}

#: Four separate claims. Satisfying an earlier one never promotes a later one.
CLAIM_LAYERS = {
    "ordinary_question_valid": (
        "the question is derivable from this episode's observed facts; judged per "
        "member by the question builder"
    ),
    "core_group_valid": (
        "the assembled group's declared media sharing, interventions and answer "
        "relations hold in the decoded media; judged by binding_groups.validate_group"
    ),
    "dual_modality_necessity": (
        "each member has an answer-changing witness under shared audio and another "
        "under shared video; an observed media/answer relation, not a model result"
    ),
    "human_answerability": (
        "not established by any of the above; requires separate human review of the "
        "delivered audio and video"
    ),
}

#: The three different motion statements, kept apart on purpose. Compiled from
#: the same constants the condition compiler uses, so a change there shows up
#: here instead of drifting into a second private vocabulary.
def motion_scope_note() -> dict:
    """Say which motion constraint belongs to which question and which recipe."""
    from avengine.qa import generation_conditions as conditions

    return {
        "during_target_audible_window": {
            "placement": conditions.MOTION_DURING_EVENT,
            "applies_to_qa_ids": sorted(conditions.MOTION_DURING_EVENT_QA_IDS),
            "statement": (
                "the motion predicate has to hold across the target's whole audible "
                "window"
            ),
        },
        "post_sound_query": {
            "applies_to_qa_ids": sorted(conditions.TAIL_DEPENDENT_QA_IDS),
            "statement": (
                "the query moment has to be after the event and after that event's "
                "measured wet tail, with no other event interfering; movement inside "
                "the tail still counts towards the answer, so this does not place the "
                "motion"
            ),
        },
        "recipe_motion_window_placement": {
            "placement": conditions.MOTION_AFTER_TAIL,
            "applies_to_task_families": ["cross_time_state"],
            "statement": (
                "only the cross_time_state recipe additionally requires the motion run "
                "to start strictly after both early passes' measured wet tails"
            ),
        },
    }


def _condition(key: str, layer: str, judge: str, detail: str) -> dict:
    return {"key": key, "layer": layer, "judge": judge, "detail": detail}


_QUESTION = "question"
_RECIPE = "recipe"
_GROUP = "group"

_GROUP_WITNESS = _condition(
    "audio_and_video_witness", _GROUP, "avengine/qa/binding_groups.py:validate_group",
    "every member needs one answer-changing witness that shares audio and another "
    "that shares video, checked against the decoded media",
)

_TASK_FAMILY_CONDITIONS = {
    "visible_binding": (
        _condition(
            "identified_sound_event", _QUESTION,
            "avengine/qa/binding_questions.py:_event",
            "the requested ordinal names exactly one event whose onset is separable "
            "from every other onset by at least one frame",
        ),
        _condition(
            "visible_active_anchor", _QUESTION,
            "avengine/qa/binding_questions.py:_at_event",
            "there is a frame where that event is actually active and its emitter is "
            "visible, rather than a leading-silence frame",
        ),
        _condition(
            "distinct_reviewed_candidates", _QUESTION,
            "avengine/qa/binding_questions.py:_visible",
            "at least two visible candidates carry reviewed appearances with distinct "
            "values, so the answer names one of them",
        ),
        _condition(
            "appearance_intervention", _RECIPE,
            "avengine/dataset/binding_group_native.py",
            "the visual variants must change which appearance the emitting instance "
            "has; the question cannot create that difference",
        ),
        _GROUP_WITNESS,
    ),
    "visual_conditioned_relation": (
        _condition(
            "visual_selection_resolvable", _QUESTION,
            "avengine/qa/binding_questions.py:_visual_conditioned_relation",
            "at the reference second the public selection rule resolves: either two "
            "uniquely visible reviewed appearances with a third visible candidate, or "
            "a reference-neighbour selector over four visible instances with an "
            "explicit pixel margin",
        ),
        _condition(
            "integer_second_activity_window", _QUESTION,
            "avengine/qa/binding_questions.py:_second",
            "the public window is two integer seconds, nonempty, and starts no earlier "
            "than the reference second",
        ),
        _condition(
            "emission_readback_available", _QUESTION,
            "avengine/qa/binding_questions.py:_intervals",
            "per-source activity intervals are observed for the selected instances; "
            "event duration alone is not emission evidence",
        ),
        _condition(
            "not_a_global_overlap_property", _QUESTION,
            "avengine/qa/binding_questions.py:_visual_conditioned_relation",
            "at least one alternative visible pair answers differently in this same "
            "world, so the answer is not a property of the whole clip",
        ),
        _condition(
            "pair_overlap_intervention", _RECIPE,
            "avengine/dataset/binding_group_reference.py",
            "the recipe must produce variants whose selected pair really does and does "
            "not overlap; the question only reads the result",
        ),
        _GROUP_WITNESS,
    ),
    "cross_event_identity": (
        _condition(
            "two_distinct_event_ordinals", _QUESTION,
            "avengine/qa/binding_questions.py:_event",
            "two different ordinals, each separable from every other onset; the same "
            "event twice is a tautology",
        ),
        _condition(
            "both_events_have_visible_active_anchor", _QUESTION,
            "avengine/qa/binding_questions.py:_at_event",
            "each event has a frame where it is active and at least two instances are "
            "visible, including its own emitter",
        ),
        _condition(
            "continuous_visual_identity_history", _QUESTION,
            "avengine/qa/binding_questions.py:_cross_event_identity",
            "every anchor candidate stays visible on every frame between the two "
            "anchors, so identity is followed in pixels and not in engine ids",
        ),
        _condition(
            "identity_intervention", _RECIPE,
            "avengine/dataset/binding_group_identity.py",
            "the audio variants must change whether the two events come from one "
            "physical instance",
        ),
        _GROUP_WITNESS,
    ),
    "cross_time_state": (
        _condition(
            "identified_sound_event", _QUESTION,
            "avengine/qa/binding_questions.py:_event",
            "the requested ordinal names exactly one separable event",
        ),
        _condition(
            "visible_active_anchor", _QUESTION,
            "avengine/qa/binding_questions.py:_at_event",
            "the anchor frame has that event active and its emitter visible",
        ),
        _condition(
            "query_after_event_and_measured_wet_tail", _QUESTION,
            "avengine/qa/unified_catalog.py:_silent_after",
            "the query is after the anchor frame, after the event end and after that "
            "event's measured wet tail, with no other programmed event overlapping",
        ),
        _condition(
            "no_other_source_wet_tail_at_query", _QUESTION,
            "avengine/qa/binding_questions.py:_cross_time_state",
            "no other source's measured wet tail is still running at the query moment",
        ),
        _condition(
            "target_visible_at_query", _QUESTION,
            "avengine/qa/binding_questions.py:_cross_time_state",
            "the silent target has a visible pixel centroid at the query moment",
        ),
        _condition(
            "continuous_visual_identity_history", _QUESTION,
            "avengine/qa/binding_questions.py:_cross_time_state",
            "every anchor candidate stays visible from the anchor frame through the "
            "query frame",
        ),
        _condition(
            "motion_window_after_measured_wet_tail", _RECIPE,
            "avengine/dataset/binding_group_motion.py",
            "this recipe opens its motion window strictly after the early passes' "
            "measured wet tails; that placement belongs to the recipe and is not "
            "required of QA-13/16/17 in general",
        ),
        _GROUP_WITNESS,
    ),
}


#: Extra question-layer conditions a particular question inside a family has
#: on top of the family's own. A family whose questions all read the same
#: evidence needs no entry here; one that asks a different thing of the same
#: world states the difference rather than reusing a condition list that does
#: not describe it.
_QUESTION_CONDITIONS_BY_QA = {
    ("visible_binding", "QA-02"): (
        _condition(
            "sound_uniquely_attributable", _QUESTION,
            "avengine/qa/binding_questions.py:_sound_to_appearance",
            "the named sound belongs to exactly one source, by a unique onset or a "
            "unique sound class, so the question points at one emitter and not at a "
            "moment several sources share",
        ),
    ),
    ("visible_binding", "QA-19"): (
        _condition(
            "named_appearance_is_unique", _QUESTION,
            "avengine/qa/binding_questions.py:_appearance_first_sound_time",
            "exactly one reviewed candidate carries the appearance the question names, "
            "so the target is identified by looking rather than by elimination",
        ),
        _condition(
            "first_event_of_named_target", _QUESTION,
            "avengine/qa/binding_questions.py:_appearance_first_sound_time",
            "the named target has at least one bound event and its earliest one is "
            "separable from its own later events by at least one frame",
        ),
        _condition(
            "answer_band_is_resolved", _QUESTION,
            "avengine/qa/binding_questions.py:_appearance_first_sound_time",
            "the onset falls inside exactly one of the published time intervals, and "
            "the intervals come from the episode's own clock and sampling policy",
        ),
    ),
}

#: Question-layer conditions of the family that a particular question does not
#: read. QA-19 names its target by appearance rather than by a sound, so the
#: family's "which event does the ordinal name" condition is not its evidence.
_QUESTION_CONDITIONS_DROPPED_BY_QA = {
    ("visible_binding", "QA-19"): ("identified_sound_event",),
}


def question_conditions_for(task_family: str, qa_id: str | None) -> tuple:
    """The question-layer conditions one question of one family really reads."""
    if task_family not in _TASK_FAMILY_CONDITIONS:
        raise BindingConditionError(f"unknown binding task family: {task_family}")
    resolved = str(qa_id or TASK_QA_IDS[task_family])
    dropped = set(_QUESTION_CONDITIONS_DROPPED_BY_QA.get((task_family, resolved), ()))
    rows = [row for row in _TASK_FAMILY_CONDITIONS[task_family]
            if row["layer"] != _QUESTION or row["key"] not in dropped]
    return tuple(rows) + tuple(_QUESTION_CONDITIONS_BY_QA.get((task_family, resolved), ()))


def task_family_requirements(task_family: str, qa_id: str | None = None) -> dict:
    """State what a core recipe requires, split by who judges each condition."""
    if task_family not in _TASK_FAMILY_CONDITIONS:
        raise BindingConditionError(f"unknown binding task family: {task_family}")
    conditions = question_conditions_for(task_family, qa_id)
    return {
        "task_family": task_family,
        "qa_id": str(qa_id or TASK_QA_IDS[task_family]),
        "question_conditions": [dict(row) for row in conditions if row["layer"] == _QUESTION],
        "recipe_conditions": [dict(row) for row in conditions if row["layer"] == _RECIPE],
        "group_conditions": [dict(row) for row in conditions if row["layer"] == _GROUP],
        "claim_layers": dict(CLAIM_LAYERS),
        "motion_scope": motion_scope_note(),
    }


def verify_task_family_evidence(readings, *, task_family: str,
                                qa_id: str | None = None) -> dict:
    """Confirm the builder actually recorded a reading for every question condition.

    A missing reading is a gap in this module's own evidence, not a silent pass:
    the caller sees which condition has nothing behind it.
    """
    requirements = task_family_requirements(task_family, qa_id)
    if not isinstance(readings, Mapping):
        raise BindingConditionError("task family readings must be a mapping")
    expected = {row["key"] for row in requirements["question_conditions"]}
    unexpected = sorted(set(readings) - expected)
    if unexpected:
        raise BindingConditionError(
            f"unknown {task_family} condition readings: {unexpected}")
    resolved = []
    missing = []
    for row in requirements["question_conditions"]:
        reading = readings.get(row["key"])
        if reading is None:
            missing.append(row["key"])
            resolved.append({**row, "status": "evidence_missing"})
            continue
        resolved.append({**row, "status": "pass", "reading": reading})
    if missing:
        raise BindingConditionError(
            f"{task_family} question conditions have no recorded reading: {missing}")
    return {
        "status": "pass",
        "task_family": task_family,
        "qa_id": requirements["qa_id"],
        "question_conditions": resolved,
        "recipe_conditions": requirements["recipe_conditions"],
        "group_conditions": requirements["group_conditions"],
        "claim_layers": requirements["claim_layers"],
        "motion_scope": requirements["motion_scope"],
        "core_group_valid": "not_run",
        "dual_modality_necessity": "not_run",
        "human_answerability": "not_run",
    }


__all__ = [
    "BindingConditionError",
    "CLAIM_LAYERS",
    "TASK_FAMILIES",
    "TASK_QA_IDS",
    "motion_scope_note",
    "task_family_requirements",
    "validate_binding_episode",
    "verify_task_family_evidence",
]


# --------------------------------------------------------------------------- group interventions

#: What a core group may change about one controlled world, and what changing
#: it touches. A member declares which level of each factor it holds; a
#: comparison between two members is then derived from those levels instead of
#: being written out by hand, so a new factor needs an entry here and nothing
#: else. ``modality`` is the delivered stream the factor rewrites, which is how
#: a pair of members knows which stream it still shares.
INTERVENTION_FACTORS = {
    "visual_appearance_slots": {
        "modality": "video",
        "changes": "which registered appearance holds each generic source slot",
        "keeps": "the body, the route, the camera and the clock",
        "implemented": True,
    },
    "audio_event_slot_assignment": {
        "modality": "audio",
        "changes": "which source slot emits each programmed sound event",
        "keeps": "the recordings, their crops and the event times",
        "implemented": True,
    },
    # Declared, not produced yet. These are the interventions that must NOT
    # change the answer, so a group carrying them states an invariance rather
    # than a necessity. They are named here so the member schema, the
    # comparison derivation and the validator already accept them.
    "unqueried_actor_appearance": {
        "modality": "video",
        "changes": "the registered appearance of an actor the question does not ask about",
        "keeps": "the queried actor's appearance, every route and the whole audio",
        "implemented": False,
        "caution": (
            "a closed-set appearance question draws its options from the visible "
            "candidates, so changing an unqueried actor's appearance also changes the "
            "answer domain; use it only with a question whose options do not depend "
            "on that actor"
        ),
    },
    "queried_clip_within_class": {
        "modality": "audio",
        "changes": "which recording of the same sound class the queried event plays",
        "keeps": "who emits it, when it starts and how long it lasts",
        "implemented": False,
    },
}

INTERVENTION_MODALITIES = ("audio", "video")


def intervention_factor(name: str) -> dict:
    """The declared meaning of one intervention factor."""
    row = INTERVENTION_FACTORS.get(str(name))
    if row is None:
        raise BindingConditionError(
            f"unknown intervention factor {name!r}; the declared factors are "
            f"{sorted(INTERVENTION_FACTORS)}"
        )
    return {"factor": str(name), **deepcopy(row)}


#: Which catalog question a core group can be built around, what its answer
#: actually is, and which declared factors move it. The group recipe reads this
#: instead of naming one question type in code, so adding a question type is a
#: row here plus a builder for it.
#:
#: ``answer_expression`` says how the answer is computed from the member's own
#: declared world bindings; it is what lets the planned answer be predicted
#: before anything is rendered, and it is never used to overwrite the observed
#: answer a member's facts produce.
GROUP_QUESTION_RECIPES = {
    "QA-20": {
        "qa_id": "QA-20",
        "task_family": "visible_binding",
        "title": "which visible candidate produced the sound",
        "answer_variable": "the registered appearance of the slot that emitted the queried event",
        "answer_expression": "appearance_of_event_slot",
        "answer_type": "closed_set",
        "default_query": {"event_number": 1},
        "flipped_by": ("visual_appearance_slots", "audio_event_slot_assignment"),
        "invariant_under": ("queried_clip_within_class",),
        "status": "implemented",
        "note": (
            "the answer names the appearance of whoever produced the queried event, so "
            "moving the appearances between slots and moving the events between slots "
            "each change it"
        ),
    },
    "QA-02": {
        "qa_id": "QA-02",
        "task_family": "visible_binding",
        "title": "sound or line to appearance",
        "answer_variable": "the registered appearance of the source of the named sound",
        "answer_expression": "appearance_of_event_slot",
        "answer_type": "closed_set",
        "default_query": {"event_number": 1},
        "flipped_by": ("visual_appearance_slots", "audio_event_slot_assignment"),
        "invariant_under": ("queried_clip_within_class",),
        "status": "implemented",
        "note": (
            "the same answer variable as QA-20 with the candidate set left open rather "
            "than restricted to the visible ones, so a reviewed candidate that is off "
            "screen at the anchor frame is still an option"
        ),
    },
    "QA-19": {
        "qa_id": "QA-19",
        "task_family": "visible_binding",
        "title": "the target's first utterance time",
        "answer_variable": "the onset time of the first event of the slot holding the named appearance",
        "answer_expression": "first_event_band_of_appearance",
        "answer_type": "time_range_s",
        "default_query": {"appearance_ordinal": 1},
        "flipped_by": ("visual_appearance_slots", "audio_event_slot_assignment"),
        "invariant_under": ("unqueried_actor_appearance", "queried_clip_within_class"),
        "status": "implemented",
        "note": (
            "the question names an appearance and asks which published time interval "
            "holds that target's first sound, so both interventions move it. The answer "
            "domain is the intervals, which do not depend on any actor's appearance, so "
            "this is the question an unqueried_actor_appearance control can be run on"
        ),
    },
    "QA-16": {
        "qa_id": "QA-16",
        "task_family": None,
        "title": "how many times the target sounded",
        "answer_variable": "the number of events emitted by the slot holding the named appearance",
        "answer_expression": "event_count_of_appearance",
        "answer_type": "count",
        "default_query": {"appearance_ordinal": 1},
        "flipped_by": ("visual_appearance_slots", "audio_event_slot_assignment"),
        "invariant_under": ("queried_clip_within_class",),
        "status": "not_implemented",
        "note": (
            "only groups whose two slots emit different numbers of events can flip this; "
            "a group whose slots each emit once has one count for both members and the "
            "necessity comparison would be unsatisfiable"
        ),
    },
    "QA-17": {
        "qa_id": "QA-17",
        "task_family": None,
        "title": "did the target sound again after the query moment",
        "answer_variable": "whether the slot holding the named appearance emits after the query moment",
        "answer_expression": "later_event_of_appearance",
        "answer_type": "yes_no",
        "default_query": {"appearance_ordinal": 1},
        "flipped_by": ("visual_appearance_slots", "audio_event_slot_assignment"),
        "invariant_under": ("queried_clip_within_class",),
        "status": "not_implemented",
        "note": (
            "a yes/no answer flips only when the two slots differ on it at the chosen "
            "query moment, so the moment has to be planned as part of the group rather "
            "than drawn per member"
        ),
    },
}


def group_question_recipe(qa_id: str) -> dict:
    """The group recipe for one catalog question."""
    row = GROUP_QUESTION_RECIPES.get(str(qa_id))
    if row is None:
        raise BindingConditionError(
            f"no core-group recipe for {qa_id!r}; the declared recipes are "
            f"{sorted(GROUP_QUESTION_RECIPES)}"
        )
    return deepcopy(row)


def implemented_group_question_recipe(qa_id: str) -> dict:
    """The recipe, refusing a question whose builder does not exist yet."""
    recipe = group_question_recipe(qa_id)
    if recipe["status"] != "implemented" or not recipe["task_family"]:
        raise BindingConditionError(
            f"{qa_id} is declared as a group question but has no builder yet: "
            f"{recipe['note']}"
        )
    return recipe


def predicted_group_answer(recipe: Mapping[str, Any], bindings: Mapping[str, Any],
                           query: Mapping[str, Any]) -> dict:
    """The answer a member's declared world bindings imply, before rendering.

    ``bindings`` states what the member's world holds: ``slot_appearances``
    maps each source slot to the appearance value it carries, and
    ``event_slots`` maps each event id, in programmed order, to the slot that
    emits it. The result is a prediction used to derive the group's comparisons
    and to report a planned answer beside the observed one; the observed answer
    always comes from that member's own facts.
    """
    expression = recipe.get("answer_expression")
    appearances = dict(bindings.get("slot_appearances") or {})
    order = list(bindings.get("event_order") or sorted(bindings.get("event_slots") or {}))
    slots = dict(bindings.get("event_slots") or {})
    if not appearances or not order or not slots:
        raise BindingConditionError(
            "a predicted answer needs slot appearances and an ordered event/slot map")
    if expression == "appearance_of_event_slot":
        number = int(query.get("event_number", 1))
        if number < 1 or number > len(order):
            raise BindingConditionError(
                f"the group asks for event {number} but the world programs {len(order)}")
        event_id = order[number - 1]
        slot = slots.get(event_id)
        if slot not in appearances:
            raise BindingConditionError(
                f"event {event_id} is emitted by {slot!r}, which carries no appearance")
        return {"value": appearances[slot], "source_slot": slot, "event_id": event_id,
                "expression": expression}
    if expression == "first_event_band_of_appearance":
        wanted = query.get("appearance_value")
        if wanted is None:
            ordinal = int(query.get("appearance_ordinal", 1))
            ordered = [appearances[slot] for slot in sorted(appearances)]
            if ordinal < 1 or ordinal > len(ordered):
                raise BindingConditionError(
                    f"the group names appearance {ordinal} but the world carries "
                    f"{len(ordered)}")
            wanted = ordered[ordinal - 1]
        holders = [slot for slot, value in appearances.items() if value == wanted]
        if len(holders) != 1:
            raise BindingConditionError(
                f"appearance {wanted!r} is carried by {len(holders)} slots, so it names "
                "no single target")
        slot = holders[0]
        starts = dict(bindings.get("event_start_s") or {})
        owned = [(float(starts[event_id]), event_id) for event_id in order
                 if slots.get(event_id) == slot and event_id in starts]
        if not owned:
            raise BindingConditionError(f"{slot!r} emits nothing, so it has no first sound")
        onset, event_id = min(owned)
        bands = [tuple(float(value) for value in band)
                 for band in bindings.get("time_bands") or ()]
        if not bands:
            raise BindingConditionError(
                "a first-sound-interval answer needs the episode's published intervals")
        index = next((position for position, (low, high) in enumerate(bands)
                      if low <= onset < high), len(bands) - 1)
        return {"value": f"band_{index}", "source_slot": slot, "event_id": event_id,
                "onset_s": onset, "band": [bands[index][0], bands[index][1]],
                "named_appearance": wanted, "expression": expression}
    raise BindingConditionError(
        f"answer expression {expression!r} has no predictor yet; "
        f"{recipe['qa_id']} cannot have its planned answer derived")


def derive_group_comparisons(members: Sequence[Mapping[str, Any]], *, qa_id: str,
                             query: Mapping[str, Any] | None = None) -> list[dict]:
    """Pair the members of one group and say what each pair should show.

    Every member states the level it holds of each intervention factor and the
    world bindings those levels produced. For each pair the differing factors
    give the shared stream, and the predicted answers give the relation: a pair
    whose answers differ is a necessity comparison, a pair whose answers agree
    is an invariance comparison. A pair whose declared factors and whose
    predicted answers disagree about which of those it is stops the build
    instead of being written down either way.
    """
    recipe = group_question_recipe(qa_id)
    query = dict(query or recipe["default_query"])
    flipping = set(recipe["flipped_by"])
    rows_by_id: dict[str, dict] = {}
    for member in members:
        member_id = str(member["member_id"])
        if member_id in rows_by_id:
            raise BindingConditionError(f"duplicate member {member_id!r}")
        levels = {str(key): str(value) for key, value
                  in dict(member.get("factor_levels") or {}).items()}
        if not levels:
            raise BindingConditionError(f"{member_id} declares no intervention levels")
        for factor in levels:
            intervention_factor(factor)
        rows_by_id[member_id] = {
            "levels": levels,
            "predicted": predicted_group_answer(recipe, member.get("bindings") or {}, query),
        }
    names = sorted(rows_by_id)
    if len(names) < 2:
        raise BindingConditionError("a group comparison needs at least two members")
    factors = sorted({factor for row in rows_by_id.values() for factor in row["levels"]})
    comparisons: list[dict] = []
    for index, left_id in enumerate(names):
        for right_id in names[index + 1:]:
            left, right = rows_by_id[left_id], rows_by_id[right_id]
            differing = sorted(
                factor for factor in factors
                if left["levels"].get(factor) != right["levels"].get(factor))
            if not differing:
                raise BindingConditionError(
                    f"{left_id} and {right_id} declare the same intervention levels")
            touched = {intervention_factor(factor)["modality"] for factor in differing}
            untouched = sorted(set(INTERVENTION_MODALITIES) - touched)
            shared = untouched[0] if len(untouched) == 1 and len(touched) == 1 else None
            answers_differ = left["predicted"]["value"] != right["predicted"]["value"]
            flipped = [factor for factor in differing if factor in flipping]
            if len(flipped) == 1 and not answers_differ:
                raise BindingConditionError(
                    f"{left_id} and {right_id} differ only in {flipped[0]}, which "
                    f"{qa_id} declares answer-changing, yet the predicted answers agree")
            if not flipped and answers_differ:
                raise BindingConditionError(
                    f"{left_id} and {right_id} differ only in factors {qa_id} declares "
                    "irrelevant, yet the predicted answers differ")
            kind = "necessity" if answers_differ else "invariance"
            if kind == "necessity" and shared is None:
                # Two members that share no stream are not a necessity witness;
                # they are simply two different worlds.
                continue
            row = {
                "members": sorted([left_id, right_id]),
                "shared_modality": shared,
                "changed_modalities": sorted(touched),
                "answer_relation": "different" if answers_differ else "same",
                "kind": kind,
                "intervention_factors": differing,
                "predicted_answers": {left_id: left["predicted"]["value"],
                                      right_id: right["predicted"]["value"]},
            }
            if kind == "invariance" and len(flipped) > 1:
                row["intervention_type"] = "joint_audio_visual_compensation"
            elif kind == "invariance":
                row["intervention_type"] = "answer_irrelevant_change"
            comparisons.append(row)
    comparisons.sort(key=lambda row: (row["kind"], str(row["shared_modality"]), row["members"]))
    return comparisons


def member_intervention_record(factor_levels: Mapping[str, str], *, qa_id: str,
                               extra: Mapping[str, Any] | None = None) -> dict:
    """One member's declared interventions and what each is expected to do."""
    recipe = group_question_recipe(qa_id)
    flipping = set(recipe["flipped_by"])
    applied = []
    for factor, level in sorted(dict(factor_levels).items()):
        declared = intervention_factor(factor)
        applied.append({
            "factor": factor,
            "level": str(level),
            "modality": declared["modality"],
            "changes": declared["changes"],
            "expected_answer_relation": "different" if factor in flipping else "same",
        })
    record = {
        "factor_levels": {str(k): str(v) for k, v in sorted(dict(factor_levels).items())},
        "applied": applied,
        "answer_relevance": {row["factor"]: ("answer_changing"
                                             if row["expected_answer_relation"] == "different"
                                             else "answer_preserving")
                             for row in applied},
    }
    record.update(dict(extra or {}))
    return record


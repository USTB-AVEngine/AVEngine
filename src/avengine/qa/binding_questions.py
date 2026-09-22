"""Instance-binding question variants derived from normalized native facts.

These variants retain the QA-01..QA-25 catalog. Their structural requirements
do not certify perceptual answerability or missing-modality model performance.
The group assembler separately checks the actual shared media and answer
changes; this module never invents media or changes the underlying facts.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from typing import Any

from avengine.qa import unified_catalog as catalog
from avengine.qa.binding_conditions import (
    TASK_FAMILIES,
    TASK_QA_IDS,
    BindingConditionError,
    verify_task_family_evidence,
)

__all__ = [
    "TASK_FAMILIES",
    "TASK_QA_IDS",
    "BindingQuestionError",
    "generate_binding_question",
]


class BindingQuestionError(ValueError):
    """The requested question is not supported by the supplied observations."""


def _second(facts: Mapping[str, Any], value: Any, name: str, *, endpoint: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BindingQuestionError(f"{name} must be an integer second")
    if not math.isfinite(value) or int(value) != value:
        raise BindingQuestionError(f"{name} must be an integer second")
    duration = float(facts["time"]["duration_seconds"])
    if value < 0 or value > duration or (not endpoint and value >= duration):
        raise BindingQuestionError(f"{name} is outside the observed clip")
    return int(value)


def _frame(facts: Mapping[str, Any], second: int) -> int:
    frame = int(round(second * float(facts["time"]["frame_rate_hz"])))
    if frame >= int(facts["time"]["frame_count"]):
        raise BindingQuestionError("query second has no captured frame")
    return frame


def _event(facts: Mapping[str, Any], number: Any) -> dict:
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise BindingQuestionError("event_number must be a positive integer")
    events = sorted(catalog._bound_events(facts), key=lambda e: (e["start_s"], e["event_id"]))
    if number > len(events):
        raise BindingQuestionError("requested sound event is absent")
    event = events[number - 1]
    tolerance = 1.0 / float(facts["time"]["frame_rate_hz"])
    if sum(abs(e["start_s"] - event["start_s"]) < tolerance for e in events) != 1:
        raise BindingQuestionError("simultaneous onsets cannot be distinguished by ordinal")
    return event


def _visible(facts: Mapping[str, Any], frame: int) -> dict[str, Mapping[str, Any]]:
    reviewed = catalog._reviewed_appearances(facts)
    result = {
        actor_id: appearance for actor_id, appearance in reviewed.items()
        if catalog._state(facts, actor_id, frame).get("state") in catalog.VISIBLE_STATES
    }
    values = [str(a["value"]) for a in result.values()]
    if len(result) < 2 or len(values) != len(set(values)):
        raise BindingQuestionError("at least two visible, distinctly reviewed candidates are required")
    return result



def _visible_instances(facts: Mapping[str, Any], frame: int) -> dict[str, None]:
    """Pixel observability does not depend on whether an appearance is named."""
    return {actor_id: None for actor_id in facts["actors"]
            if catalog._state(facts, actor_id, frame).get("state") in catalog.VISIBLE_STATES}

def _at_event_for_emitter(facts: Mapping[str, Any], event: Mapping[str, Any]) -> int:
    """The first frame where this event is actually sounding and its emitter shows.

    A question whose answer domain is every reviewed appearance needs its target
    visible so the answer can be bound by looking; it does not need a second
    candidate on screen at that instant, because an off-screen candidate is
    still one of the options. That is the difference from :func:`_at_event`,
    which serves a question whose options are the visible candidates.
    """
    first = max(0, int(math.floor(event["start_s"] * facts["time"]["frame_rate_hz"])))
    last = min(int(facts["time"]["frame_count"]),
               int(math.ceil(event["end_s"] * facts["time"]["frame_rate_hz"])))
    for frame in range(first, last):
        if event["event_id"] not in {row["event_id"] for row in catalog._active_at(facts, frame)}:
            continue
        if event["actor_id"] in _visible_instances(facts, frame):
            return frame
    raise BindingQuestionError(
        "the event has no frame where it is sounding and its emitter is visible")


def _at_event(facts: Mapping[str, Any], event: Mapping[str, Any], *,
              named_candidates: bool = False) -> tuple[int, dict]:
    """Use an actually active, visible anchor rather than a leading-silence frame."""
    first = max(0, int(math.floor(event["start_s"] * facts["time"]["frame_rate_hz"])))
    last = min(int(facts["time"]["frame_count"]), int(math.ceil(event["end_s"] * facts["time"]["frame_rate_hz"])))
    for frame in range(first, last):
        active = {e["event_id"] for e in catalog._active_at(facts, frame)}
        if event["event_id"] not in active:
            continue
        try:
            candidates = (_visible(facts, frame) if named_candidates
                          else _visible_instances(facts, frame))
            if len(candidates) < 2:
                continue
        except BindingQuestionError:
            continue
        if event["actor_id"] in candidates:
            return frame, candidates
    raise BindingQuestionError("the event has no visible, active multi-candidate anchor")


def _anchor_reading(facts: Mapping[str, Any], event: Mapping[str, Any], frame: int,
                    candidates: Mapping[str, Any]) -> dict:
    """Record why this frame is a usable anchor, not merely that one was found."""
    return {
        "event_id": event["event_id"], "anchor_frame": frame,
        "emitter_actor_id": event["actor_id"],
        "active_event_ids": sorted(e["event_id"] for e in catalog._active_at(facts, frame)),
        "visible_candidate_actor_ids": sorted(candidates),
    }


def _event_reading(facts: Mapping[str, Any], event: Mapping[str, Any], number: Any) -> dict:
    events = sorted(catalog._bound_events(facts), key=lambda e: (e["start_s"], e["event_id"]))
    tolerance = 1.0 / float(facts["time"]["frame_rate_hz"])
    return {
        "event_number": number, "event_id": event["event_id"],
        "bound_event_count": len(events),
        "onset_separation_tolerance_s": tolerance,
        "coincident_onset_count": sum(
            abs(e["start_s"] - event["start_s"]) < tolerance for e in events),
    }


def _visible_history(facts: Mapping[str, Any], actor_ids, lo: int, hi: int, reason: str) -> dict:
    """Require an observable identity history and say over which frames it held."""
    for actor_id in sorted(actor_ids):
        for frame in range(lo, hi + 1):
            if catalog._state(facts, actor_id, frame).get("state") not in catalog.VISIBLE_STATES:
                raise BindingQuestionError(reason)
    return {"actor_ids": sorted(actor_ids), "frame_range": [lo, hi],
            "checked_frame_count": max(0, hi - lo + 1)}


def _intervals(facts: Mapping[str, Any], actor_id: str, lo: int, hi: int) -> list[tuple[int, int]]:
    if not facts.get("source_activity_evidence_present"):
        raise BindingQuestionError("source activity readback is required; event duration is insufficient")
    intervals = []
    table = facts.get("source_activity_intervals_samples", {})
    for event in catalog._bound_events(facts):
        if event["actor_id"] != actor_id:
            continue
        values = event.get("source_activity_intervals_samples")
        if values is None:
            values = table.get(event["event_id"])
        if values is None:
            raise BindingQuestionError(f"missing source activity for {event['event_id']}")
        for value in values:
            start = max(lo, int(value["start_sample"]))
            end = min(hi, int(value["end_sample_exclusive"]))
            if start < end:
                intervals.append((start, end))
    return sorted(intervals)


def _intersect(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(max(a, c), min(b, d)) for a, b in left for c, d in right if max(a, c) < min(b, d)]


def _yes_no() -> list[dict]:
    return [
        {"value": "yes", "label_en": "yes", "label_zh": "是"},
        {"value": "no", "label_en": "no", "label_zh": "否"},
    ]


def _visible_binding(facts: dict, query: Mapping[str, Any], seed: str) -> tuple[dict, dict]:
    number = query.get("event_number", 1)
    event = _event(facts, number)
    frame, candidates = _at_event(facts, event, named_candidates=True)
    target = candidates[event["actor_id"]]
    readings = {
        "identified_sound_event": _event_reading(facts, event, number),
        "visible_active_anchor": _anchor_reading(facts, event, frame, candidates),
        "distinct_reviewed_candidates": {
            "candidate_actor_ids": sorted(candidates),
            "appearance_values": sorted(str(a["value"]) for a in candidates.values()),
        },
    }
    options = []
    for appearance in candidates.values():
        label_en, label_zh = catalog._appearance_phrases(appearance)
        options.append({"value": str(appearance["value"]), "label_en": label_en, "label_zh": label_zh})
    anchor_en, anchor_zh = catalog._event_anchor(facts, event)
    return catalog._question_item(
        qa_id=TASK_QA_IDS["visible_binding"], facts=facts, seed=seed,
        question_en=f"Which visual appearance belongs to the object that produced {anchor_en}?",
        question_zh=f"发出{anchor_zh}的对象具有哪一种外观？",
        open_answer_type="closed_set", open_truth=str(target["value"]),
        truth_label=catalog._appearance_phrases(target)[0], options=options,
        evidence={"target_actor_id": event["actor_id"], "event_id": event["event_id"],
                  "anchor_frame": frame, "candidate_actor_ids": list(candidates),
                  "appearance": deepcopy(dict(target))},
        slug="binding_visible",
    ), readings



def _select_reference_neighbors(facts, frame, selector):
    from avengine.qa.angular_questions import _visual_bearing
    if selector.get("kind") not in {"two_nearest_to_leftmost", "two_nearest_to_named_reference"}:
        raise BindingQuestionError("unsupported visual relation selector")
    scopes = [selector.get("candidate_scope_en"), selector.get("candidate_scope_zh")]
    if any(not isinstance(scope, str) or not scope.strip() for scope in scopes):
        raise BindingQuestionError("visual selector requires an explicit public candidate scope")
    margin = selector.get("minimum_margin_px")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise BindingQuestionError("visual selector requires an explicit nonnegative pixel margin")
    visible = _visible_instances(facts, frame)
    if len(visible) != 4:
        raise BindingQuestionError("this reference selector requires four visible candidate instances")
    centers = {actor: _visual_bearing(facts, actor, frame)[2] for actor in visible}
    reference_appearance = None
    if selector["kind"] == "two_nearest_to_leftmost":
        ordered = sorted(centers, key=lambda actor: (centers[actor][0], actor))
        if centers[ordered[1]][0] - centers[ordered[0]][0] <= margin:
            raise BindingQuestionError("reference is not uniquely leftmost under the requested margin")
        reference = ordered[0]
    else:
        requested = selector.get("reference_appearance")
        if not isinstance(requested, Mapping) or not requested.get("field") or not requested.get("value"):
            raise BindingQuestionError("named reference requires an appearance field and value")
        matches = [(actor, appearance) for actor, appearance in catalog._reviewed_appearances(facts).items()
                   if actor in visible and appearance.get("field") == requested["field"]
                   and appearance.get("value") == requested["value"]]
        if len(matches) != 1:
            raise BindingQuestionError("named reference appearance is not uniquely visible and reviewed")
        reference, reference_appearance = matches[0]
    distances = {actor: math.dist(center, centers[reference]) for actor, center in centers.items()
                 if actor != reference}
    neighbors = sorted(distances, key=lambda actor: (distances[actor], actor))
    if distances[neighbors[2]] - distances[neighbors[1]] <= margin:
        raise BindingQuestionError("nearest pair is ambiguous under the requested margin")
    return {actor: None for actor in neighbors}, neighbors[:2], {
        "kind": selector["kind"], "reference_actor_id": reference,
        "candidate_actor_ids": list(visible), "visible_centroids_xy_px": centers,
        "distance_from_reference_px": distances, "minimum_margin_px": margin,
        "candidate_scope_en": scopes[0], "candidate_scope_zh": scopes[1],
        "reference_appearance": deepcopy(reference_appearance),
    }

def _visual_conditioned_relation(facts: dict, query: Mapping[str, Any], seed: str) -> tuple[dict, dict]:
    reference = _second(facts, query.get("reference_time_s", 0), "reference_time_s")
    frame = _frame(facts, reference)
    selector = query.get("visual_selector")
    selection_evidence = None
    if selector is not None:
        if not isinstance(selector, Mapping):
            raise BindingQuestionError("visual_selector must be an object")
        candidates, selected_actors, selection_evidence = _select_reference_neighbors(facts, frame, selector)
        labels = None
        selection_reading = {
            "rule": "reference_neighbours", "selector_kind": selector["kind"],
            "reference_time_s": reference, "reference_frame": frame,
            "visible_candidate_count": len(selection_evidence["candidate_actor_ids"]),
            "minimum_margin_px": selection_evidence["minimum_margin_px"],
            "selected_actor_ids": list(selected_actors),
        }
    else:
        candidates = _visible_instances(facts, frame)
        reviewed = {actor_id: appearance for actor_id, appearance in catalog._reviewed_appearances(facts).items()
                    if actor_id in candidates}
        selected = query.get("appearance_values")
        if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)) or len(selected) != 2:
            raise BindingQuestionError("overlap requires two distinct visual appearance values")
        if len(set(selected)) != 2:
            raise BindingQuestionError("selected appearance values must differ")
        by_value = {}
        for value in selected:
            matches = [(actor_id, appearance) for actor_id, appearance in reviewed.items()
                       if str(appearance["value"]) == str(value)]
            if len(matches) != 1:
                raise BindingQuestionError("selected appearance is not uniquely visible at the reference time")
            by_value[str(value)] = matches[0]
        if len(candidates) < 3:
            raise BindingQuestionError("a third candidate is required to avoid global-overlap audio-only questions")
        selected_actors = [by_value[str(value)][0] for value in selected]
        labels = [catalog._appearance_phrases(by_value[str(value)][1]) for value in selected]
        selection_reading = {
            "rule": "named_appearances", "reference_time_s": reference,
            "reference_frame": frame, "appearance_values": [str(value) for value in selected],
            "visible_candidate_count": len(candidates),
            "selected_actor_ids": list(selected_actors),
        }
    window = query.get("window_s")
    if not isinstance(window, Sequence) or isinstance(window, (str, bytes)) or len(window) != 2:
        raise BindingQuestionError("window_s must contain two integer seconds")
    start = _second(facts, window[0], "window start")
    end = _second(facts, window[1], "window end", endpoint=True)
    if end <= start or reference > start:
        raise BindingQuestionError("the reference must precede a nonempty activity window")
    sr = int(facts["time"]["sample_rate_hz"])
    intervals = [_intervals(facts, actor, start * sr, end * sr) for actor in selected_actors]
    overlap = _intersect(*intervals)
    # At least one alternative pair must answer differently in this actual world.
    # This avoids assigning AV necessity to a visually selected but irrelevant subset.
    other_answers = []
    actors = list(candidates)
    for index, left in enumerate(actors):
        for right in actors[index + 1:]:
            other_answers.append(bool(_intersect(
                _intervals(facts, left, start * sr, end * sr),
                _intervals(facts, right, start * sr, end * sr))))
    if len(set(other_answers)) < 2:
        raise BindingQuestionError("all visible pairs have the same overlap answer")
    truth = "yes" if overlap else "no"
    readings = {
        "visual_selection_resolvable": selection_reading,
        "integer_second_activity_window": {
            "reference_time_s": reference, "window_s": [start, end],
            "sample_rate_hz": sr, "window_samples": [start * sr, end * sr],
        },
        "emission_readback_available": {
            "activity_definition": "emission_readback",
            "interval_counts": {actor: len(rows)
                                for actor, rows in zip(selected_actors, intervals)},
        },
        "not_a_global_overlap_property": {
            "visible_pair_count": len(other_answers),
            "distinct_pair_answers": sorted({bool(value) for value in other_answers}),
            "selected_pair_answer": bool(overlap),
        },
    }
    if selection_evidence:
        if selection_evidence["reference_appearance"] is not None:
            reference_en, reference_zh = catalog._appearance_phrases(selection_evidence["reference_appearance"])
            reference_rule_en = f"Use {reference_en} as the reference. "
            reference_rule_zh = f"以{reference_zh}为参照，"
        else:
            reference_rule_en = "Use the object whose visible pixel centroid is leftmost as the reference. "
            reference_rule_zh = "以可见像素质心最靠左的对象为参照，"
        prefix_en = (f"At {reference} s, consider only {selector['candidate_scope_en']}. "
                     + reference_rule_en
                     + "Select the two other objects with the smallest image-plane distances from its visible pixel centroid. ")
        prefix_zh = (f"第{reference}秒，只考虑{selector['candidate_scope_zh']}。"
                     + reference_rule_zh
                     + "选取其余对象中与其可见像素质心的画面距离最近的两个对象。")
    else:
        prefix_en = f"Consider {labels[0][0]} and {labels[1][0]} visible at {reference} s. "
        prefix_zh = f"第{reference}秒画面中的{labels[0][1]}与{labels[1][1]}，"
    return catalog._question_item(
        qa_id=TASK_QA_IDS["visual_conditioned_relation"], facts=facts, seed=seed,
        question_en=(prefix_en + f"Did these two objects emit sound simultaneously at any time in [{start}, {end}) s?"),
        question_zh=(prefix_zh + f"在第{start}至第{end}秒内（不含结束时刻）是否曾同时发声？"),
        open_answer_type="closed_set", open_truth=truth, truth_label=truth, options=_yes_no(),
        evidence={"selected_actor_ids": selected_actors, "reference_time_s": reference,
                  "window_s": [start, end], "source_activity_intervals": intervals,
                  "overlap_intervals_samples": overlap, "activity_definition": "emission_readback",
                  "candidate_actor_ids": actors, "visual_selection": selection_evidence},
        slug="binding_visual_conditioned_overlap",
    ), readings


def _cross_event_identity(facts: dict, query: Mapping[str, Any], seed: str) -> tuple[dict, dict]:
    numbers = query.get("event_numbers", [1, 2])
    if not isinstance(numbers, Sequence) or isinstance(numbers, (str, bytes)) or len(numbers) != 2:
        raise BindingQuestionError("event_numbers must contain two different event ordinals")
    if numbers[0] == numbers[1]:
        raise BindingQuestionError("same-event identity questions are tautologies")
    events = [_event(facts, number) for number in numbers]
    anchors = [_at_event(facts, event) for event in events]
    lo, hi = sorted(frame for frame, _ in anchors)
    # The first version requires a visible identity history. Hidden-identity
    # ambiguity is not resolved by consulting private engine IDs.
    tracked = set(anchors[0][1]) | set(anchors[1][1])
    history = _visible_history(
        facts, tracked, lo, hi,
        "cross-event identity requires an observable visual history")
    truth = "yes" if events[0]["actor_id"] == events[1]["actor_id"] else "no"
    readings = {
        "two_distinct_event_ordinals": {
            "event_numbers": [numbers[0], numbers[1]],
            "events": [_event_reading(facts, event, number)
                       for event, number in zip(events, numbers)],
        },
        "both_events_have_visible_active_anchor": {
            "anchors": [_anchor_reading(facts, event, frame, candidates)
                        for event, (frame, candidates) in zip(events, anchors)],
        },
        "continuous_visual_identity_history": history,
    }
    return catalog._question_item(
        qa_id=TASK_QA_IDS["cross_event_identity"], facts=facts, seed=seed,
        question_en=(f"Did independent sound events {numbers[0]} and {numbers[1]} come from "
                     "the same physical object, following its identity through the video?"),
        question_zh=f"第{numbers[0]}个与第{numbers[1]}个独立发声事件是否来自视频中的同一个物理对象？",
        open_answer_type="closed_set", open_truth=truth, truth_label=truth, options=_yes_no(),
        evidence={"event_ids": [e["event_id"] for e in events],
                  "actor_ids": [e["actor_id"] for e in events],
                  "anchor_frames": [frame for frame, _ in anchors],
                  "identity_definition": "persistent physical actor, not sound class or appearance equality"},
        slug="binding_cross_event_identity",
    ), readings


def _cross_time_state(facts: dict, query: Mapping[str, Any], seed: str) -> tuple[dict, dict]:
    from avengine.qa.angular_questions import _visual_bearing

    number = query.get("event_number", 1)
    event = _event(facts, number)
    anchor, candidates = _at_event(facts, event)
    if "query_time_s" in query and "query_anchor" in query:
        raise BindingQuestionError("select either a query anchor or a query second")
    clip_end = "query_time_s" not in query
    if clip_end:
        if query.get("query_anchor", "clip_end") != "clip_end":
            raise BindingQuestionError("unsupported state query anchor")
        frame = int(facts["time"]["frame_count"]) - 1
        second = frame / float(facts["time"]["frame_rate_hz"])
        time_en, time_zh = "At the end of the video", "片尾时"
    else:
        second = _second(facts, query["query_time_s"], "query_time_s")
        frame = _frame(facts, second)
        time_en, time_zh = f"At {second} s", f"在第{second}秒"
    if frame <= anchor:
        raise BindingQuestionError("state query must follow the sound anchor")
    # The event, its own measured wet tail and any other programmed event are
    # judged by the catalog predicate the ordinary QA-13 question uses, so the
    # core member and the ordinary question cannot drift apart.
    post_sound = catalog._silent_after(facts, event, frame)
    other_tails = []
    for tail in facts.get("audio", {}).get("wet_tail_intervals", []):
        if float(tail["start_s"]) <= second < float(tail["end_s"]):
            raise BindingQuestionError("a source still has an audible wet tail at the state query")
        if tail.get("event_id") != event["event_id"]:
            other_tails.append({"event_id": tail.get("event_id"),
                                "end_s": float(tail["end_s"])})
    visible = _visible_instances(facts, frame)
    if event["actor_id"] not in visible:
        raise BindingQuestionError("the silent target is not visually localizable at query time")
    history = _visible_history(
        facts, candidates, anchor, frame,
        "cross-time state requires an observable visual identity history")
    bearing, calibration, centroid = _visual_bearing(facts, event["actor_id"], frame)
    readings = {
        "identified_sound_event": _event_reading(facts, event, number),
        "visible_active_anchor": _anchor_reading(facts, event, anchor, candidates),
        "query_after_event_and_measured_wet_tail": deepcopy(post_sound),
        "no_other_source_wet_tail_at_query": {
            "query_time_s": second, "other_event_wet_tails": other_tails},
        "target_visible_at_query": {
            "target_actor_id": event["actor_id"], "query_frame": frame,
            "visible_actor_ids": sorted(visible)},
        "continuous_visual_identity_history": history,
    }
    anchor_en, anchor_zh = catalog._event_anchor(facts, event)
    item = catalog._question_item(
        qa_id=TASK_QA_IDS["cross_time_state"], facts=facts, seed=seed,
        question_en=(f"{time_en}, what is the bearing of the visible pixel centroid of the "
                     f"object that produced {anchor_en}? Front is 0 degrees; right is positive."),
        question_zh=(f"{time_zh}，发出{anchor_zh}的对象，其可见像素质心位于多少度？"
                     "正前方为0度，右侧为正。"),
        open_answer_type="angle_deg", open_truth=bearing, truth_label=f"{bearing} degrees",
        mcq_optional=True, open_extra={"scoring_mode": "threshold_graded", "theta_full_deg": 1.0, "theta_half_deg": 3.0},
        evidence={"target_actor_id": event["actor_id"], "event_id": event["event_id"],
                  "anchor_frame": anchor, "query_frame": frame, "query_time_s": second,
                  "query_anchor": "clip_end" if clip_end else "integer_second",
                  "observation_cutoff_s": None if clip_end else second, "target": "visible_pixel_centroid",
                  "visible_centroid_xy_px": centroid, "camera_calibration": calibration,
                  "candidate_actor_ids": list(candidates)},
        slug="binding_cross_time_bearing",
    )
    for form in item["model_input"].values():
        form["camera_calibration"] = deepcopy(calibration)
    return item, readings


def _sound_to_appearance(facts: dict, query: Mapping[str, Any], seed: str) -> tuple[dict, dict]:
    """QA-02 as a core-group question: which appearance made this sound.

    The answer variable is the registered appearance of whoever produced the
    queried event, so the appearance swap moves it and the event/slot swap moves
    it. It differs from QA-20 in its answer domain: QA-20 offers the candidates
    that are visible at the anchor frame, this one offers every reviewed
    appearance in the episode, so a candidate that is off screen at that instant
    is still an option and the reader cannot answer by elimination from the
    frame alone.
    """
    number = query.get("event_number", 1)
    event = _event(facts, number)
    frame = _at_event_for_emitter(facts, event)
    visible = _visible_instances(facts, frame)
    reviewed = catalog._reviewed_appearances(facts)
    if event["actor_id"] not in reviewed:
        raise BindingQuestionError("the emitter of the queried sound has no reviewed appearance")
    values = [str(row["value"]) for row in reviewed.values()]
    if len(reviewed) < 2 or len(values) != len(set(values)):
        raise BindingQuestionError(
            "at least two candidates with distinct reviewed appearances are required")
    active = sorted(row["event_id"] for row in catalog._active_at(facts, frame))
    if active != [event["event_id"]]:
        raise BindingQuestionError(
            "another programmed event is audible at the anchor frame, so the named "
            "sound is not attributable to one source")
    target = reviewed[event["actor_id"]]
    readings = {
        "identified_sound_event": _event_reading(facts, event, number),
        "visible_active_anchor": _anchor_reading(facts, event, frame, visible),
        "distinct_reviewed_candidates": {
            "candidate_actor_ids": sorted(reviewed),
            "appearance_values": sorted(values),
        },
        "sound_uniquely_attributable": {
            "event_id": event["event_id"], "anchor_frame": frame,
            "active_event_ids": active,
            "emitter_actor_id": event["actor_id"],
        },
    }
    options = []
    for appearance in reviewed.values():
        label_en, label_zh = catalog._appearance_phrases(appearance)
        options.append({"value": str(appearance["value"]),
                        "label_en": label_en, "label_zh": label_zh})
    anchor_en, anchor_zh = catalog._event_anchor(facts, event)
    return catalog._question_item(
        qa_id="QA-02", facts=facts, seed=seed,
        question_en=f"What does the object that produced {anchor_en} look like?",
        question_zh=f"发出{anchor_zh}的那个对象是什么样子的？",
        open_answer_type="closed_set", open_truth=str(target["value"]),
        truth_label=catalog._appearance_phrases(target)[0], options=options,
        evidence={"target_actor_id": event["actor_id"], "event_id": event["event_id"],
                  "anchor_frame": frame, "candidate_actor_ids": sorted(reviewed),
                  "visible_candidate_actor_ids": sorted(visible),
                  "appearance": deepcopy(dict(target))},
        slug="binding_sound_to_appearance",
    ), readings


def _appearance_first_sound_time(facts: dict, query: Mapping[str, Any], seed: str) -> tuple[dict, dict]:
    """QA-19 as a core-group question: when did the named appearance first sound.

    The target is named by its reviewed appearance and the answer is which
    published time interval holds that target's first sound. The appearance swap
    moves which body carries the name, and the event/slot swap moves which sound
    that body emits, so both interventions move the answer while the answer
    domain - the intervals - depends on neither.

    The group names one appearance value for every member, so the public
    question text is the same sentence in all of them; leaving the target to an
    ordinal would have named a different colour in each variant.
    """
    reviewed = catalog._reviewed_appearances(facts)
    values = [str(row["value"]) for row in reviewed.values()]
    if len(reviewed) < 2 or len(values) != len(set(values)):
        raise BindingQuestionError(
            "at least two candidates with distinct reviewed appearances are required")
    wanted = query.get("appearance_value")
    if wanted is None:
        ordinal = int(query.get("appearance_ordinal", 1))
        if ordinal < 1 or ordinal > len(values):
            raise BindingQuestionError(
                f"the group names appearance {ordinal} of {len(values)}")
        wanted = sorted(values)[ordinal - 1]
    wanted = str(wanted)
    holders = [actor_id for actor_id, row in reviewed.items()
               if str(row["value"]) == wanted]
    if len(holders) != 1:
        raise BindingQuestionError(
            f"the named appearance {wanted!r} is carried by {len(holders)} reviewed "
            "candidates, so it names no single target")
    actor_id = holders[0]
    owned = sorted((event for event in catalog._bound_events(facts)
                    if event["actor_id"] == actor_id),
                   key=lambda event: (float(event["start_s"]), str(event["event_id"])))
    if not owned:
        raise BindingQuestionError("the named target emits nothing in this episode")
    first = owned[0]
    tolerance = 1.0 / float(facts["time"]["frame_rate_hz"])
    if any(float(event["start_s"]) - float(first["start_s"]) < tolerance
           for event in owned[1:]):
        raise BindingQuestionError(
            "the named target's first onset is not separable from its own next one")
    frame, visible = _at_event(facts, first, named_candidates=True)
    if actor_id not in visible:
        raise BindingQuestionError("the named target is not visible while it is sounding")
    bands = [tuple(float(value) for value in band) for band in catalog._time_bands(facts)]
    onset = float(first["start_s"])
    matching = [index for index, (low, high) in enumerate(bands) if low <= onset < high]
    if len(matching) != 1:
        raise BindingQuestionError(
            f"the onset {onset} falls in {len(matching)} published intervals")
    band_index = matching[0]
    labels = [catalog._time_band_label(facts, index, bands) for index in range(len(bands))]
    options = [{"value": f"band_{index}", "label_en": labels[index][0],
                "label_zh": labels[index][1], "allow_value": False}
               for index in range(len(bands))]
    appearance_en, appearance_zh = catalog._appearance_phrases(reviewed[actor_id])
    domain_en = ", ".join(label[0] for label in labels)
    domain_zh = "、".join(label[1] for label in labels)
    readings = {
        "visible_active_anchor": _anchor_reading(facts, first, frame, visible),
        "distinct_reviewed_candidates": {
            "candidate_actor_ids": sorted(reviewed),
            "appearance_values": sorted(values),
        },
        "named_appearance_is_unique": {
            "named_value": wanted, "holder_actor_ids": holders,
            "reviewed_actor_ids": sorted(reviewed),
        },
        "first_event_of_named_target": {
            "target_actor_id": actor_id, "event_id": first["event_id"],
            "onset_s": onset, "own_event_ids": [event["event_id"] for event in owned],
            "separation_tolerance_s": tolerance,
        },
        "answer_band_is_resolved": {
            "onset_s": onset, "band_index": band_index,
            "bands_s": [list(band) for band in bands],
            "band_authority": "avengine/qa/unified_catalog.py:_time_bands",
        },
    }
    return catalog._question_item(
        qa_id="QA-19", facts=facts, seed=seed,
        question_en=(f"Which time interval contained the first sound from {appearance_en}? "
                     f"The clip is divided into {len(bands)} time intervals: {domain_en}."),
        question_zh=(f"{appearance_zh}第一次发声落在哪个时间段？"
                     f"片段按时间划分为{len(bands)}段：{domain_zh}。"),
        open_answer_type="time_range_s",
        open_truth=[float(bands[band_index][0]), float(bands[band_index][1])],
        truth_label=labels[band_index][0], options=options,
        mcq_truth=f"band_{band_index}",
        open_extra={
            "time_ranges_s": [list(band) for band in bands],
            "time_range_labels_en": [label[0] for label in labels],
            "time_range_labels_zh": [label[1] for label in labels],
            "time_range_index": band_index,
        },
        evidence={"target_actor_id": actor_id, "named_appearance": wanted,
                  "appearance": deepcopy(dict(reviewed[actor_id])),
                  "event_id": first["event_id"], "anchor_frame": frame,
                  "first_sound_onset_s": onset, "time_band_index": band_index,
                  "time_bands_s": [list(band) for band in bands],
                  "candidate_actor_ids": sorted(reviewed)},
        slug="binding_first_sound_interval",
    ), readings


_BUILDERS = {
    "visible_binding": _visible_binding,
    "visual_conditioned_relation": _visual_conditioned_relation,
    "cross_event_identity": _cross_event_identity,
    "cross_time_state": _cross_time_state,
}

#: A family can build more than one catalog question of the same controlled
#: world. The family entry above stays the default so an existing caller that
#: names only a family behaves exactly as it did.
_BUILDERS_BY_QA = {
    ("visible_binding", "QA-20"): _visible_binding,
    ("visible_binding", "QA-02"): _sound_to_appearance,
    ("visible_binding", "QA-19"): _appearance_first_sound_time,
}


def generate_binding_question(
    raw_or_facts: Mapping[str, Any], task_family: str, query: Mapping[str, Any], *,
    seed: str = "binding-question", qa_id: str | None = None,
) -> dict[str, Any]:
    """Emit one requested variant; fail rather than label an inapplicable case.

    ``qa_id`` selects which catalog question of this family to build. Omitting
    it keeps the family's own default, so a caller that never asked for a
    second question type sees no change.
    """
    if task_family not in _BUILDERS:
        raise BindingQuestionError(f"unknown binding task family: {task_family}")
    resolved_qa = str(qa_id or TASK_QA_IDS[task_family])
    builder = _BUILDERS_BY_QA.get((task_family, resolved_qa))
    if builder is None:
        if qa_id is not None and resolved_qa != TASK_QA_IDS[task_family]:
            raise BindingQuestionError(
                f"{task_family} has no builder for {resolved_qa}; it builds "
                + ", ".join(sorted(
                    name for family, name in _BUILDERS_BY_QA if family == task_family))
                or "nothing else")
        builder = _BUILDERS[task_family]
    facts = (catalog._restore_normalized_frame_keys(raw_or_facts)
             if raw_or_facts.get("schema") == catalog.UNIFIED_FACT_SCHEMA
             else catalog.normalize_episode_bundle(raw_or_facts))
    try:
        catalog._require_stereo(facts)
        if facts.get("input_summary", {}).get("unresolved_event_ids"):
            raise BindingQuestionError("native event-source bindings are unresolved")
        if not facts.get("source_activity_evidence_present"):
            raise BindingQuestionError("actual source activity evidence is required")
        missing_activity = [
            event["event_id"] for event in catalog._bound_events(facts)
            if event.get("source_activity_evidence_status") != "observed"
        ]
        if missing_activity:
            raise BindingQuestionError(
                "complete observed source activity is required; rederive older normalized facts "
                f"from native readbacks: {missing_activity}")
        item, readings = builder(facts, query, seed)
        conditions = verify_task_family_evidence(
            readings, task_family=task_family, qa_id=resolved_qa)
    except catalog._Deferred as error:
        raise BindingQuestionError(f"{error.code}: {error.detail}") from error
    except BindingConditionError as error:
        raise BindingQuestionError(str(error)) from error
    if item["qa_id"] != conditions["qa_id"]:
        raise BindingQuestionError(
            f"{task_family} emitted {item['qa_id']} but its conditions name "
            f"{conditions['qa_id']}")
    item.update({
        "question_kind": "binding_variant",
        "binding_task_family": task_family,
        "required_modalities": ["audio", "video"],
        "binding_conditions": conditions,
        "modality_necessity": {"status": "not_run", "construction": "requires_group_media_verification"},
    })
    return item

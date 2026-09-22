"""Explicit continuous-bearing questions mined from native episode evidence.

Audio uses the emitter bearing; visual questions use an observed pixel centroid.
The AV subset checks structural feasibility, never certifies modality necessity.
"""
from __future__ import annotations

import copy
import math
import json
from pathlib import Path
from collections import Counter
from collections.abc import Mapping
from typing import Any

from avengine.qa import unified_catalog as catalog

SUBSETS = ("A", "V", "AV")
ANGLE_THRESHOLDS_DEG = (1, 3, 5, 10)
CONVENTION_EN = " Report one integer horizontal angle in whole degrees: front 0, right positive, in [-180, 180)."
CONVENTION_ZH = " 请只回答整数水平角度（度）：正前方为0，右侧为正，范围[-180,180)。"


def public_camera_calibration(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    """Accept only declared public pinhole calibration, never hidden actor geometry."""
    value = raw.get("camera_calibration")
    if not isinstance(value, Mapping):
        return None
    if value.get("projection") != "pinhole" or value.get("public") is not True:
        return None
    try:
        width, height = int(value["width_px"]), int(value["height_px"])
        fx, cx = float(value["fx_px"]), float(value["cx_px"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if width <= 0 or height <= 0 or not all(map(math.isfinite, (fx, cx))) or fx <= 0:
        return None
    if not 0 <= cx <= width:
        return None
    return {"projection": "pinhole", "public": True, "width_px": width,
            "height_px": height, "fx_px": fx, "cx_px": cx,
            "pixel_coordinates": "zero_based_pixel_centers"}



def camera_calibration_from_capture(capture_root: Path) -> dict[str, Any] | None:
    """Publish intrinsics from the native capture's consumed camera configuration."""
    root = Path(capture_root)
    native_visual = root / "visual_plan.json"
    calibration = None
    if native_visual.is_file():
        visual = json.loads(native_visual.read_text(encoding="utf-8"))
        camera = visual.get("camera", {})
        fov = camera.get("horizontal_fov_deg")
        resolution = camera.get("resolution_hw")
        pixel_truth = root / "pixel_visibility_truth.json"
        if resolution is None and pixel_truth.is_file():
            resolution = json.loads(pixel_truth.read_text(encoding="utf-8")).get("resolution_hw")
        # One public K cannot describe a zooming camera.
        if any(frame.get("camera_state", {}).get("horizontal_fov_deg", fov) != fov
               for frame in visual.get("frames", [])):
            return None
    else:
        receipt_path = root / "research_receipt.json"
        if not receipt_path.is_file():
            return None
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        request = receipt.get("inputs", {}).get("m1_request")
        if not request or not Path(request).is_file():
            return None
        spec = json.loads(Path(request).read_text(encoding="utf-8"))
        calibration = spec.get("primary_camera_rig", {}).get("shared_calibration", {})
        if calibration.get("projection") != "pinhole":
            return None
        fov, resolution = calibration.get("hfov_degrees"), calibration.get("resolution_hw")
    if not isinstance(resolution, list) or len(resolution) != 2:
        return None
    try:
        hfov = float(fov)
        height, width = map(int, resolution)
        if not math.isfinite(hfov) or not 0 < hfov < 180:
            return None
        value = {"public": True, "projection": "pinhole", "width_px": width, "height_px": height,
                 "fx_px": width / (2 * math.tan(math.radians(hfov / 2))), "cx_px": (width - 1) / 2}
        return public_camera_calibration({"camera_calibration": value})
    except (TypeError, ValueError, ZeroDivisionError):
        return None



def _audio_bearing(facts: Mapping[str, Any], actor_id: str, frame: int) -> float:
    """Azimuth in the actual Listener basis, including camera pitch and roll.

    The older coarse-angle helper projects onto world XZ. Fine audio angles
    must instead agree with the full Listener orientation consumed by RLR.
    """
    actor = catalog._actor(facts, actor_id)
    positions = actor.get("emitter_positions_m")
    listener = facts.get("listener", {})
    if not positions or frame >= len(positions) or actor.get("source", {}).get("emitter_readbacks") is False:
        catalog._defer("missing_emitter_readback", "audio bearing requires the actual sound emitter position")
    if listener.get("status") != "pass" or not listener.get("basis_m3"):
        catalog._defer("missing_listener_readback", "audio bearing requires the native Listener basis")
    delta = [positions[frame][axis] - listener["positions_m"][frame][axis] for axis in range(3)]
    basis = listener["basis_m3"][frame]
    right = sum(delta[axis] * basis["right"][axis] for axis in range(3))
    forward = sum(delta[axis] * basis["forward"][axis] for axis in range(3))
    if math.hypot(right, forward) <= 1e-12:
        catalog._defer("undefined_azimuth", "source is directly above/below or coincident with the Listener")
    return math.degrees(math.atan2(right, forward))


def _visual_bearing(facts: Mapping[str, Any], actor_id: str, frame: int) -> tuple[float, dict, list]:
    calibration = public_camera_calibration(facts)
    if calibration is None:
        catalog._defer("public_camera_calibration_missing", "visual bearing needs public pinhole fx/cx and image dimensions")
    resolution = facts.get("visibility_meta", {}).get("resolution_hw")
    if resolution != [calibration["height_px"], calibration["width_px"]]:
        catalog._defer("camera_resolution_mismatch", "calibration must describe the delivered visibility image")
    row = catalog._state(facts, actor_id, frame)
    if row.get("state") not in catalog.VISIBLE_STATES:
        catalog._defer("visual_target_not_visible", "query target has no visible pixels")
    centroid = row.get("visible_centroid_xy_px")
    if centroid is None and row.get("visible_fraction") == 1.0:
        # Target-only masks can be hidden/amodal. They equal observed pixels only
        # when the native visibility readback says that all target pixels are visible.
        centroid = row.get("target_centroid_xy_px")
    if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
        catalog._defer("observed_centroid_missing", "visual bearing needs the visible, not occluded target-only, centroid")
    x, y = map(float, centroid)
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < calibration["width_px"] and 0 <= y < calibration["height_px"]):
        catalog._defer("invalid_observed_centroid", "visible centroid lies outside the calibrated image")
    return math.degrees(math.atan2(x - calibration["cx_px"], calibration["fx_px"])), calibration, [x, y]


def _labels(facts: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    try:
        values = catalog._appearance_candidates(facts, require_unique=False)
    except catalog._Deferred:
        return {}
    phrases = {actor_id: catalog._appearance_phrases(appearance)
               for actor_id, actor, appearance in values}
    multiplicity = Counter(phrase for phrase in phrases.values())
    return {actor_id: phrase for actor_id, phrase in phrases.items() if multiplicity[phrase] == 1}



def _whole_second_frames(facts: Mapping[str, Any]) -> list[int]:
    """Select video readbacks nearest to whole-second query instants."""
    fps = float(facts["time"]["frame_rate_hz"])
    count = int(facts["time"]["frame_count"])
    return sorted({round(second * fps) for second in range(math.ceil(count / fps))
                   if round(second * fps) < count})


def _query_second(facts: Mapping[str, Any], frame: int) -> int:
    return round(frame / float(facts["time"]["frame_rate_hz"]))


def publishable_query_second(facts: Mapping[str, Any], frame: int) -> int | None:
    """The whole second a frame may be published as, or None.

    A question states a whole second, and the answer is measured at one
    frame. The two only describe the same instant when that frame is the
    nearest readback to the second and the second is itself inside the clip.
    Rounding the last frame of a ten second clip up to "10 s" names a moment
    the media never shows, so this refuses instead of restating.
    """

    fps = float(facts["time"]["frame_rate_hz"])
    frame_count = int(facts["time"]["frame_count"])
    frame = int(frame)
    second = round(frame / fps)
    nearest = round(second * fps)
    if nearest != frame:
        return None
    if not 0 <= nearest < frame_count:
        return None
    return int(second)


def _active_frames(facts: Mapping[str, Any]) -> dict[int, list]:
    if not catalog._source_activity_present(facts):
        return {}
    return {frame: catalog._active_at(facts, frame)
            for frame in _whole_second_frames(facts)}


def _hidden_motion_changed(facts: Mapping[str, Any], actor_id: str, anchor: int, query: int) -> bool:
    """Reject static or exactly constant-velocity continuations of visible motion."""
    positions = facts["actors"][actor_id].get("root_positions_m")
    if not positions or anchor < 1 or query <= anchor:
        return False
    previous = anchor - 1
    if catalog._state(facts, actor_id, previous).get("state") not in catalog.VISIBLE_STATES:
        return False
    prediction = [positions[anchor][axis] + (query - anchor) *
                  (positions[anchor][axis] - positions[previous][axis]) for axis in range(3)]
    return math.dist(prediction, positions[query]) > 1e-3 and math.dist(positions[anchor], positions[query]) > 1e-3


def candidates(facts: Mapping[str, Any]) -> list[dict[str, Any]]:
    labels = _labels(facts)
    result: list[dict[str, Any]] = []
    active = _active_frames(facts) if facts.get("audio", {}).get("status") == "pass" else {}
    events = catalog._bound_events(facts) if active else []
    fps = float(facts["time"]["frame_rate_hz"])
    wet_by_id = {row["event_id"]: row for row in facts.get("audio", {}).get("wet_tail_intervals", [])
                 if isinstance(row, Mapping) and "event_id" in row}
    for event in events:
        actor_id, event_id = event["actor_id"], event["event_id"]
        audible = [f for f, rows in active.items() if any(e["event_id"] == event_id for e in rows)]
        if not audible:
            continue
        # A temporal onset identifies the event without revealing its hidden ID.
        # Simultaneous onsets with no unique class are ambiguous to the listener.
        peers = [e for e in events if abs(float(e["start_s"]) - float(event["start_s"])) < 1.0 / fps]
        unique_class = event.get("sound_class_explicit") and event.get("sound_class") and sum(
            e.get("sound_class") == event.get("sound_class") for e in peers) == 1
        if len(peers) == 1 or unique_class:
            f = audible[len(audible) // 2]
            result.append({"subset": "A", "actor_id": actor_id, "event_id": event_id,
                           "query_frame": f, "use_sound_class": len(peers) > 1})
        if actor_id not in labels:
            continue
        anchors: list[int] = []
        av_rows: list[dict[str, Any]] = []
        for frame in audible:
            state = facts.get("visibility", {}).get(actor_id, {}).get(frame, {}).get("state")
            rows = active[frame]
            # Dry inactivity does not mean that another voice's reverberant tail
            # has ended. Require an isolated anchor in the actual wet readback.
            time_s = frame / fps
            competing_tail = any(
                other["event_id"] != event_id and
                float(wet_by_id.get(other["event_id"], {}).get("start_s", other["start_s"])) <= time_s <
                float(wet_by_id.get(other["event_id"], {}).get("end_s", facts["time"]["duration_seconds"]))
                for other in events)
            if state in catalog.VISIBLE_STATES and len({e["actor_id"] for e in rows}) == 1 and not competing_tail:
                anchors.append(frame)
            if state not in {"out_of_view", "fully_occluded"} or not anchors:
                continue
            rivals = [e for e in rows if e["actor_id"] != actor_id]
            if not rivals or not event.get("sound_asset_id") or any(
                e.get("sound_asset_id") == event["sound_asset_id"] for e in rivals):
                continue
            # A continuous event preserves the audible identity across hiding;
            # the last visible frame anchors the motion extrapolation diagnostic.
            visible = [f for f in range(anchors[-1], frame) if facts.get("visibility", {}).get(actor_id, {}).get(f, {}).get("state") in catalog.VISIBLE_STATES]
            last_visible = max(visible, default=anchors[-1])
            if not _hidden_motion_changed(facts, actor_id, last_visible, frame):
                continue
            av_rows.append({"subset": "AV", "actor_id": actor_id, "event_id": event_id,
                           "query_frame": frame, "anchor_frame": anchors[-1],
                           "last_visible_frame": last_visible,
                           "competing_event_ids": [e["event_id"] for e in rivals]})
        if av_rows:
            result.append(av_rows[len(av_rows) // 2])
    for actor_id in labels:
        legal = []
        for frame in _whole_second_frames(facts):
            try:
                _visual_bearing(facts, actor_id, frame)
            except catalog._Deferred:
                continue
            legal.append(frame)
        if legal:
            result.append({"subset": "V", "actor_id": actor_id, "query_frame": legal[len(legal) // 2]})
    for row in result:
        row["kind"] = "continuous_bearing"
        row["candidate_id"] = f"QA-25:{row['subset']}:{row['actor_id']}:{row.get('event_id', 'visual')}:{row['query_frame']}"
    return result


def subset_diagnostics(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Say why each QA-25 subset has candidates or has none.

    The AV subset needs a target that is still audible after it stops being
    visible. When an episode keeps every emitter inside the view for its whole
    duration, that subset has nothing to select and the count is zero for a
    reason a producer can act on. Reporting the measured stage counts keeps
    that apart from a broken predicate.
    """

    rows = candidates(facts)
    counts = {subset: sum(1 for row in rows if row["subset"] == subset)
              for subset in SUBSETS}
    stages = Counter()
    hidden_states = Counter()
    audible_states = Counter()
    labels = _labels(facts)
    audio_pass = facts.get("audio", {}).get("status") == "pass"
    active = _active_frames(facts) if audio_pass else {}
    events = catalog._bound_events(facts) if active else []
    for event in events:
        actor_id, event_id = event["actor_id"], event["event_id"]
        audible = [f for f, entries in active.items()
                   if any(e["event_id"] == event_id for e in entries)]
        if not audible:
            continue
        stages["events_with_audible_frame"] += 1
        if actor_id not in labels:
            stages["events_without_unique_appearance"] += 1
            continue
        for frame in audible:
            state = str(facts.get("visibility", {}).get(actor_id, {}).get(frame, {}).get("state"))
            audible_states[state] += 1
            if state in {"out_of_view", "fully_occluded"}:
                hidden_states[state] += 1
                stages["hidden_while_audible_frames"] += 1
                if len([e for e in active[frame] if e["actor_id"] != actor_id]):
                    stages["hidden_with_other_source_frames"] += 1
    diagnostics: dict[str, Any] = {}
    for subset in SUBSETS:
        record: dict[str, Any] = {"candidate_count": counts[subset]}
        if counts[subset]:
            diagnostics[subset] = record
            continue
        if not audio_pass:
            record.update({"code": "audio_readback_not_pass",
                           "detail": "the episode has no passing audio readback"})
        elif not active:
            record.update({"code": "missing_source_activity_readback",
                           "detail": "no source activity readback is present"})
        elif subset == "V":
            record.update({"code": "no_publishable_visual_bearing",
                           "detail": ("no whole-second frame has a public pinhole "
                                      "calibration and a visible target centroid")})
        elif subset == "A":
            record.update({"code": "no_identifiable_audible_onset",
                           "detail": ("no audible event can be named without "
                                      "revealing its hidden identity")})
        else:
            record.update({
                "code": ("no_hidden_while_audible_frame"
                         if not stages["hidden_while_audible_frames"]
                         else "no_hidden_frame_with_a_competing_source"),
                "detail": ("the target never leaves the view or becomes fully "
                           "occluded while it is still audible"
                           if not stages["hidden_while_audible_frames"]
                           else "the target is hidden while audible but no other "
                                "source with a distinct asset sounds at that instant"),
            })
        record["visibility_states_at_audible_frames"] = dict(sorted(audible_states.items()))
        record["stage_counts"] = dict(sorted(stages.items()))
        diagnostics[subset] = record
    return diagnostics


def _item(facts: Mapping[str, Any], seed: str, candidate: Mapping[str, Any],
          question_en: str, question_zh: str, *, qa_id: str = "QA-25", slug: str | None = None) -> dict:
    actor_id, frame = candidate["actor_id"], int(candidate["query_frame"])
    subset = candidate["subset"]
    second = publishable_query_second(facts, frame)
    evidence = {**copy.deepcopy(dict(candidate)), "target_actor_id": actor_id,
                "query_frame_time_s": frame / float(facts["time"]["frame_rate_hz"]),
                "angle_reference": "camera_image_horizontal" if subset == "V" else "listener_horizontal",
                "angle_target": "visible_pixel_centroid" if subset == "V" else "sound_emitter"}
    if second is not None:
        evidence["query_time_s"] = second
    elif frame == int(facts["time"]["frame_count"]) - 1:
        # The same convention binding_questions uses for a clip-end query: no
        # second is published, the anchor names the instant instead.
        evidence["query_anchor"] = "clip_end"
    else:
        evidence["query_time_s_deferred"] = {
            "code": "query_frame_is_not_a_whole_second",
            "detail": "this readback frame is not the nearest frame to any whole second inside the clip",
        }
    calibration = None
    if subset == "V":
        angle, calibration, centroid = _visual_bearing(facts, actor_id, frame)
        evidence["visible_centroid_xy_px"] = centroid
    else:
        catalog._require_stereo(facts)
        if candidate.get("event_id") not in {e["event_id"] for e in catalog._active_at(facts, frame, require_source_activity=True)}:
            catalog._defer("target_not_audible_at_query", "continuous audio bearing needs actual source activity at query")
        angle = _audio_bearing(facts, actor_id, frame)
    angle = (float(angle) + 180.0) % 360.0 - 180.0
    if not math.isfinite(angle):
        catalog._defer("nonfinite_angle", "bearing readback is not finite")
    item = catalog._question_item(qa_id=qa_id, facts=facts, seed=seed,
        question_en=question_en + CONVENTION_EN, question_zh=question_zh + CONVENTION_ZH,
        open_answer_type="angle_deg", open_truth=angle, truth_label=f"{angle:.6f} degrees",
        evidence=evidence, mcq_optional=True,
        mcq_deferred_reason={"code": "continuous_numeric_only", "detail": "this question requires a continuous numeric answer"},
        open_extra={"convention": "right_positive", "scoring_mode": "threshold_graded",
                    "angle_thresholds_deg": list(ANGLE_THRESHOLDS_DEG), "angle_reference": evidence["angle_reference"]},
        slug=slug or f"bearing_{subset}_{actor_id}_{candidate.get('event_id', 'visual')}_{frame}")
    item.update({"angle_subset": subset, "candidate_id": candidate.get("candidate_id"),
                 "required_modalities": {"A": ["audio"], "V": ["video"], "AV": ["audio", "video"]}[subset]})
    if calibration:
        item["model_input"]["open"]["camera_calibration"] = calibration
        item["forms"]["open"]["camera_calibration"] = calibration
    if subset == "AV":
        item["evidence"]["structural_checks"] = {"visible_audible_anchor": True,
            "multiple_query_sources": True, "target_hidden_at_query": True,
            "distinct_sound_assets": True, "hidden_motion_differs_from_visible_linear_extrapolation": True}
        item["evidence"]["remaining_review"] = ["perceptual_stream_identity", "full_av_answerability", "modality_necessity", "balanced_binding_pairs"]
        item["truth"]["evidence"] = copy.deepcopy(item["evidence"])
    return item


def emit(facts: Mapping[str, Any], candidate: Mapping[str, Any], seed: str) -> dict:
    subset, frame = candidate["subset"], int(candidate["query_frame"])
    time_s = publishable_query_second(facts, frame)
    if time_s is None:
        catalog._defer("query_frame_is_not_a_whole_second",
                       "a bearing question states a whole second, and this frame is not one",
                       query_frame=frame,
                       query_frame_time_s=frame / float(facts["time"]["frame_rate_hz"]))
    if subset == "A":
        event = next(e for e in facts["events"] if e["event_id"] == candidate["event_id"])
        anchor_en, anchor_zh = catalog._event_anchor(facts, event)
        if candidate.get("use_sound_class"):
            anchor_en = f"the {event['sound_class']} sound"
            anchor_zh = f"{event['sound_class']}声音"
        en = f"At {time_s} s, what is the bearing of {anchor_en}?"
        zh = f"在第{time_s}秒，{anchor_zh}的声音来自多少度？"
    else:
        label_en, label_zh = _labels(facts)[candidate["actor_id"]]
        if subset == "V":
            en = f"At {time_s} s, what is the horizontal camera bearing of the centroid of the visible pixels of {label_en}? Use the supplied pinhole calibration."
            zh = f"在{time_s}秒，{label_zh}的可见像素质心相对相机的水平角度是多少？请使用提供的针孔相机标定。"
        else:
            anchor_s = publishable_query_second(facts, candidate["anchor_frame"])
            if anchor_s is None:
                catalog._defer("anchor_frame_is_not_a_whole_second",
                               "the visible anchor instant cannot be stated in whole seconds",
                               anchor_frame=int(candidate["anchor_frame"]))
            en = f"Track {label_en}, visible and sounding at {anchor_s} s. At {time_s} s, when it is hidden and multiple sources are sounding, what is its sound bearing?"
            zh = f"请追踪在{anchor_s}秒可见且发声的{label_zh}。在{time_s}秒它已不可见且多个声源同时发声时，它的声音来自多少度？"
    return _item(facts, seed, candidate, en, zh)


def followups(facts: Mapping[str, Any], items: list[dict], seed: str) -> tuple[list[dict], list[dict]]:
    result, deferred = [], []
    for parent in items:
        qa_id = parent["qa_id"]
        if qa_id not in {"QA-03", "QA-20", "QA-24"}:
            continue
        evidence = parent["evidence"]
        event_ref = evidence.get("first_event", evidence.get("anchor_event", evidence))
        event_id = event_ref.get("event_id")
        event = next((e for e in facts["events"] if e["event_id"] == event_id), None)
        if event is None:
            continue
        actor_id = event["actor_id"]
        try:
            if qa_id == "QA-24":
                frame = int(evidence["final_frame"])
                sounding = [e for e in catalog._active_at(facts, frame) if e["actor_id"] == actor_id]
                if sounding:
                    subset, target_event = "A", sounding[0]
                else:
                    subset, target_event = "V", event
                    _visual_bearing(facts, actor_id, frame)
                subject_en, subject_zh = "the first speaker", "最先发声的个体"
            else:
                peers = [e for e in facts["events"] if abs(float(e["start_s"]) - float(event["start_s"])) < 1.0 / float(facts["time"]["frame_rate_hz"])]
                if len(peers) != 1:
                    catalog._defer("ambiguous_sound_onset", "the angle followup cannot identify one of simultaneous sound onsets")
                audible = [f for f in _whole_second_frames(facts)
                           if event_id in {e["event_id"] for e in catalog._active_at(facts, f)}]
                if not audible:
                    catalog._defer("target_not_audible_at_query", "the associated event has no source activity readback")
                frame, subset, target_event = audible[len(audible) // 2], "A", event
                subject_en, subject_zh = catalog._event_anchor(facts, event)
            time_s = publishable_query_second(facts, frame)
            if qa_id != "QA-24" and time_s is None:
                catalog._defer("query_frame_is_not_a_whole_second",
                               "the angle followup instant cannot be stated in whole seconds",
                               query_frame=int(frame))
            target_en = "visible pixel centroid" if subset == "V" else "sound"
            target_zh = "可见像素质心" if subset == "V" else "声音"
            candidate = {"subset": subset, "actor_id": actor_id, "event_id": target_event["event_id"], "query_frame": frame}
            when_en = "At the end of the clip" if qa_id == "QA-24" else f"At {time_s} s"
            when_zh = "片尾时" if qa_id == "QA-24" else f"在第{time_s}秒"
            item = _item(facts, seed, candidate,
                f"{when_en}, what is the bearing of the {target_en} of {subject_en}?",
                f"{when_zh}，{subject_zh}的{target_zh}位于多少度？", qa_id=qa_id,
                slug=f"angle_followup_{parent['question_id']}_{frame}")
            item.update({"question_kind": "angle_followup", "parent_question_id": parent["question_id"],
                         "binding_parent": qa_id in {"QA-03", "QA-20"} and parent["truth"]["value"] == actor_id})
            if qa_id == "QA-24":
                item["required_modalities"] = ["audio", "video"] if subset == "V" else ["audio"]
            result.append(item)
        except catalog._Deferred as error:
            deferred.append({"parent_question_id": parent["question_id"], "qa_id": qa_id,
                             "status": "deferred", "code": error.code, "detail": error.detail})
    return result, deferred

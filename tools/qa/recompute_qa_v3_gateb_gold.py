#!/usr/bin/env python3
"""Recompute Gate-B gold for every selected QA-v3 pilot candidate."""

from __future__ import annotations
import argparse
import copy
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from design_qa_v3_scene_batch import COAT_WORDS, recompute_azimuth  # noqa:E402

PIXEL = {"card11", "card15a", "card16"}


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n")


def state(timeline, slot, frame):
    return next(
        x
        for x in timeline["frames"][frame]["actor_states"]
        if x["source_slot_id"] == slot
    )


def distance(timeline, slot, frame):
    c = np.asarray(timeline["frames"][frame]["camera"]["translation_ue_cm"][:2], float)
    p = np.asarray(state(timeline, slot, frame)["translation_ue_cm"][:2], float)
    return float(np.linalg.norm(p - c))


def coat(selection, slot):
    asset = next(
        x["asset_id"] for x in selection["actors"] if x["source_slot_id"] == slot
    )
    return COAT_WORDS[asset]


def bands(profile):
    return [tuple(x) for x in profile.get("answer_bands_deg", [])]


def band_label(profile, value):
    bs = bands(profile)
    labels = [f"[{a:g}, {b:g})" for a, b in bs]
    matches = [labels[i] for i, (a, b) in enumerate(bs) if a <= value < b]
    if len(matches) != 1:
        raise ValueError(f"angle {value} outside bands")
    return matches[0]


def slot_events(program):
    candidates = program["candidate_source_endpoint_ids"]
    ep_to_slot = {candidates[0]: "source1", candidates[1]: "source2"}
    return [
        (ep_to_slot[e["source_endpoint_id"]], e["start_sample"] / 16000.0)
        for e in program["events"]
    ]


def _endpoint_records(registry, *, owner):
    """Return endpoint records while keeping registry joins explicit.

    Gate-B appearance twins can have different endpoint ids from the main
    point. The binding identity in the registry is therefore the only safe
    way to relate the two; list order and endpoint-id spelling are not.
    """
    if isinstance(registry, Mapping):
        records = registry.get("source_endpoints")
    else:
        records = registry
    if not isinstance(records, list) or not records:
        raise ValueError(f"{owner} has no source_endpoints list")
    by_id = {}
    for endpoint in records:
        if not isinstance(endpoint, Mapping):
            raise ValueError(f"{owner} contains a non-object endpoint")
        endpoint_id = endpoint.get("source_endpoint_id")
        if not isinstance(endpoint_id, str) or not endpoint_id:
            raise ValueError(f"{owner} endpoint has no source_endpoint_id")
        if endpoint_id in by_id:
            raise ValueError(f"{owner} repeats endpoint {endpoint_id!r}")
        binding = endpoint.get("binding")
        if not isinstance(binding, Mapping):
            raise ValueError(f"{owner} endpoint {endpoint_id!r} has no binding")
        instance = binding.get("entity_instance_id")
        if not isinstance(instance, str) or not instance:
            raise ValueError(
                f"{owner} endpoint {endpoint_id!r} has no explicit "
                "entity_instance_id")
        by_id[endpoint_id] = endpoint
    return records, by_id


def _actor_records(selection, *, owner):
    if not isinstance(selection, Mapping):
        raise ValueError(f"{owner} selection is not an object")
    actors = selection.get("actors")
    if not isinstance(actors, list) or not actors:
        raise ValueError(f"{owner} selection has no actors")
    by_slot = {}
    for actor in actors:
        if not isinstance(actor, Mapping):
            raise ValueError(f"{owner} contains a non-object actor")
        slot = actor.get("source_slot_id")
        if not isinstance(slot, str) or not slot:
            raise ValueError(f"{owner} actor has no source_slot_id")
        if slot in by_slot:
            raise ValueError(f"{owner} repeats actor slot {slot!r}")
        asset = actor.get("asset_id")
        if not isinstance(asset, str) or not asset:
            raise ValueError(f"{owner} actor {slot!r} has no asset_id")
        instance = actor.get("entity_instance_id") or actor.get(
            "legacy_timeline_actor_id")
        if not isinstance(instance, str) or not instance:
            raise ValueError(
                f"{owner} actor {slot!r} has no explicit entity instance id")
        by_slot[slot] = actor
    return by_slot


def _endpoint_context(selection, registry, *, owner):
    actors = _actor_records(selection, owner=owner)
    records, by_id = _endpoint_records(registry, owner=owner)
    by_instance = {}
    for endpoint in records:
        instance = endpoint["binding"]["entity_instance_id"]
        by_instance.setdefault(instance, []).append(endpoint)

    slot_to_endpoint = {}
    endpoint_to_slot = {}
    selected_by_instance = {}
    for slot, actor in actors.items():
        instance = actor.get("entity_instance_id") or actor.get(
            "legacy_timeline_actor_id")
        matches = [
            endpoint for endpoint in by_instance.get(instance, [])
            if endpoint["binding"].get("entity_asset_id") in (None, actor["asset_id"])
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{owner} actor {slot!r} expected one endpoint binding for "
                f"instance {instance!r} and asset {actor['asset_id']!r}, "
                f"found {len(matches)}")
        endpoint = matches[0]
        endpoint_id = endpoint["source_endpoint_id"]
        slot_to_endpoint[slot] = endpoint_id
        endpoint_to_slot[endpoint_id] = slot
        selected_by_instance.setdefault(instance, []).append(endpoint)
    if len(endpoint_to_slot) != len(slot_to_endpoint):
        raise ValueError(f"{owner} endpoint bindings are not one-to-one")
    return {
        "actors": actors,
        "records": records,
        "by_id": by_id,
        "by_instance": by_instance,
        "selected_by_instance": selected_by_instance,
        "slot_to_endpoint": slot_to_endpoint,
        "endpoint_to_slot": endpoint_to_slot,
    }


def _timeline_matches_selection(timeline, context, *, owner):
    """Check the authored Gate-B timeline did use its selected actors."""
    if not isinstance(timeline, Mapping) or not isinstance(
            timeline.get("frames"), list) or not timeline["frames"]:
        raise ValueError(f"{owner} timeline has no frames")
    for frame in timeline["frames"]:
        states = frame.get("actor_states") if isinstance(frame, Mapping) else None
        if not isinstance(states, list):
            raise ValueError(f"{owner} timeline frame has no actor_states")
        slots = [state.get("source_slot_id") for state in states if isinstance(state, Mapping)]
        if len(slots) != len(states) or len(set(slots)) != len(slots) or set(slots) != set(context["actors"]):
            raise ValueError(f"{owner} timeline actor slots differ from its selection")
        for actor_state in states:
            if not isinstance(actor_state, Mapping):
                raise ValueError(f"{owner} timeline has a non-object actor state")
            slot = actor_state.get("source_slot_id")
            actor = context["actors"][slot]
            if actor_state.get("asset_id") not in (None, actor["asset_id"]):
                raise ValueError(
                    f"{owner} timeline asset drift at {slot!r}: "
                    f"{actor_state.get('asset_id')!r} != {actor['asset_id']!r}")
            expected_instance = actor.get("entity_instance_id") or actor.get(
                "legacy_timeline_actor_id")
            # timeline actor_id is the visual timeline's slot-local runtime
            # id (for example ``source1_actor``), not the endpoint binding
            # identity. Only explicit endpoint identity fields are comparable.
            state_instance = actor_state.get("entity_instance_id") or actor_state.get(
                "legacy_timeline_actor_id")
            if state_instance not in (None, expected_instance):
                raise ValueError(
                    f"{owner} timeline instance drift at {slot!r}: "
                    f"{state_instance!r} != {expected_instance!r}")


def _normal_text(value):
    return " ".join(str(value).split()).casefold()


def _speech_rows(fact):
    rows = fact.get("speech_bindings")
    if rows is None:
        rows = (fact.get("audio") or {}).get("utterances")
    if not isinstance(rows, list) or not rows:
        raise ValueError("speech fact has no speech_bindings/utterances")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"speech binding {index} is not an object")
        slot = row.get("slot") or row.get("source_slot_id")
        transcript = row.get("transcript")
        sound_asset_id = row.get("sound_asset_id")
        if not isinstance(slot, str) or not slot:
            raise ValueError(f"speech binding {index} has no slot")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError(f"speech binding {index} has no transcript")
        if not isinstance(sound_asset_id, str) or not sound_asset_id:
            raise ValueError(f"speech binding {index} has no sound_asset_id")
        if row.get("start_sample") is None:
            raise ValueError(f"speech binding {index} has no start_sample")
        result.append(dict(row, slot=slot, transcript=transcript.strip()))
    return result


def _row_colour(row):
    for key in ("colour", "color", "top_colour", "top_color"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _actor_colour(actor):
    containers = [actor]
    for key in ("realized_attributes", "realised_attributes", "appearance",
                "visual_attributes"):
        if isinstance(actor.get(key), Mapping):
            containers.append(actor[key])
    for container in containers:
        for key in ("top_color", "top_colour", "color", "colour"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _speech_event_rows(program, rows, main_context, gateb_context):
    events = program.get("events") if isinstance(program, Mapping) else None
    if not isinstance(events, list) or not events:
        raise ValueError("speech AudioProgram has no events")
    rows_used = set()
    bound = []
    for event_index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"speech AudioProgram event {event_index} is not an object")
        endpoint_id = event.get("source_endpoint_id")
        main_slot = main_context["endpoint_to_slot"].get(endpoint_id)
        if main_slot is None:
            raise ValueError(
                f"speech AudioProgram event {event_index} endpoint {endpoint_id!r} "
                "is not bound by main source endpoints")
        start = event.get("start_sample")
        sound = event.get("sound_asset_id")
        if start is None or not isinstance(sound, str) or not sound:
            raise ValueError(
                f"speech AudioProgram event {event_index} lacks start_sample or "
                "sound_asset_id")
        matches = [
            (index, row) for index, row in enumerate(rows)
            if index not in rows_used
            and row["slot"] == main_slot
            and row["sound_asset_id"] == sound
            and int(row["start_sample"]) == int(start)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"speech AudioProgram event {event_index} does not have one "
                f"fact binding for slot {main_slot!r}, start {start!r}, "
                f"asset {sound!r} (found {len(matches)})")
        row_index, row = matches[0]
        event_end = event.get("end_sample_exclusive")
        row_duration = row.get("duration_samples")
        if event_end is not None and row_duration is not None:
            if int(event_end) - int(start) != int(row_duration):
                raise ValueError(
                    f"speech AudioProgram event {event_index} duration does not "
                    "match its fact binding")
        rows_used.add(row_index)
        main_endpoint = main_context["by_id"][endpoint_id]
        instance = main_endpoint["binding"]["entity_instance_id"]
        gate_matches = gateb_context["selected_by_instance"].get(instance, [])
        if len(gate_matches) != 1:
            raise ValueError(
                "Gate-B source endpoints do not preserve exactly one explicit "
                f"binding for main instance {instance!r}")
        gate_endpoint = gate_matches[0]
        gate_endpoint_id = gate_endpoint["source_endpoint_id"]
        gate_slot = gateb_context["endpoint_to_slot"].get(gate_endpoint_id)
        if gate_slot is None:
            raise ValueError(
                f"Gate-B endpoint {gate_endpoint_id!r} is not bound by its selection")
        bound.append({
            "event": event,
            "event_index": event_index,
            "row": row,
            "main_slot": main_slot,
            "gateb_slot": gate_slot,
            "main_endpoint_id": endpoint_id,
            "gateb_endpoint_id": gate_endpoint_id,
            "entity_instance_id": instance,
        })
    if len(rows_used) != len(rows):
        raise ValueError(
            f"speech fact has {len(rows) - len(rows_used)} binding(s) not present "
            "in the fixed main AudioProgram")
    return bound


def _speech_gateb_compute(
        pid, fact, selection, timeline, program, *, main_selection,
        main_endpoint_registry, gateb_endpoint_registry, gateb_fact=None):
    if pid not in {"card13", "card14"}:
        raise ValueError(f"speech recomputation does not support {pid!r}")
    main_context = _endpoint_context(
        main_selection, main_endpoint_registry, owner="main")
    gateb_context = _endpoint_context(
        selection, gateb_endpoint_registry, owner="Gate-B")
    _timeline_matches_selection(timeline, gateb_context, owner="Gate-B")
    rows = _speech_rows(fact)
    bound = _speech_event_rows(program, rows, main_context, gateb_context)

    # The selected asset's written appearance is the visual label. For the
    # current actor-selection schema the label lives in the main fact's
    # speech bindings, so carry it by asset_id into the Gate-B selection.
    colour_by_asset = {}
    for row in rows:
        main_actor = main_context["actors"].get(row["slot"])
        if main_actor is None:
            raise ValueError(f"speech binding references unknown slot {row['slot']!r}")
        colour = _row_colour(row) or _actor_colour(main_actor)
        if colour is None:
            raise ValueError(
                f"no explicit visual colour for main actor {row['slot']!r}")
        old = colour_by_asset.setdefault(main_actor["asset_id"], colour)
        if old != colour:
            raise ValueError(
                f"main asset {main_actor['asset_id']!r} has conflicting colours")
    gateb_colour_by_slot = {}
    for slot, actor in gateb_context["actors"].items():
        colour = _actor_colour(actor) or colour_by_asset.get(actor["asset_id"])
        if colour is None:
            raise ValueError(
                f"no visual colour for Gate-B actor {slot!r} asset {actor['asset_id']!r}")
        gateb_colour_by_slot[slot] = colour

    mcq = fact.get("mcq")
    open_fact = fact.get("open")
    if not isinstance(mcq, Mapping) or not isinstance(open_fact, Mapping):
        raise ValueError("speech fact must contain mcq and open objects")
    options = mcq.get("options_space")
    if not isinstance(options, list) or not options:
        raise ValueError("speech fact has no MCQ options")
    if mcq.get("stem") != open_fact.get("stem"):
        raise ValueError("main speech MCQ/open stems differ")
    main_gold = mcq.get("truth_option")
    if main_gold != open_fact.get("truth_value"):
        raise ValueError("main speech MCQ/open gold differs")

    # The question itself is inherited verbatim. Compare the declared answer
    # space with the immutable fact data, and only replace the gold value.
    if pid == "card13":
        fixed_colour = fact.get("question_target_colour")
        if not isinstance(fixed_colour, str) or not fixed_colour.strip():
            raise ValueError("card13 fact has no question_target_colour")
        fixed_colour = fixed_colour.strip()
        target = [item for item in bound
                  if gateb_colour_by_slot[item["gateb_slot"]] == fixed_colour]
        if len(target) != 1:
            raise ValueError(
                "card13 questioned colour must bind exactly one Gate-B speech event")
        target_item = target[0]
        gateb_gold = target_item["row"]["transcript"]
        option_keys = [_normal_text(value) for value in options]
        speech_keys = [_normal_text(item["row"]["transcript"]) for item in bound]
        question_options_preserved = sorted(option_keys) == sorted(speech_keys)
        fixed_question = {"kind": "appearance_colour", "colour": fixed_colour}
    else:
        fixed_transcript = fact.get("question_target_transcript")
        target = []
        if isinstance(fixed_transcript, str) and fixed_transcript.strip():
            wanted = _normal_text(fixed_transcript)
            target = [item for item in bound
                      if _normal_text(item["row"]["transcript"]) == wanted]
        if not target:
            speaker = fact.get("target_speaker_id")
            utterance = fact.get("target_utterance_id")
            if speaker is not None or utterance is not None:
                target = [item for item in bound
                          if (speaker is None
                              or item["row"].get("speaker_id") == speaker)
                          and (utterance is None
                               or item["row"].get("utterance_id") == utterance)]
        if len(target) != 1:
            raise ValueError(
                "card14 questioned transcript must bind exactly one speech event")
        target_item = target[0]
        gateb_gold = gateb_colour_by_slot[target_item["gateb_slot"]]
        option_keys = [_normal_text(value) for value in options]
        colour_keys = [_normal_text(value) for value in colour_by_asset.values()]
        question_options_preserved = sorted(option_keys) == sorted(colour_keys)
        fixed_question = {
            "kind": "speech_transcript",
            "transcript": target_item["row"]["transcript"],
        }

    if gateb_gold not in options:
        raise ValueError(
            f"Gate-B gold {gateb_gold!r} is outside the main MCQ option space")
    gateb_fact_gold_matches = None
    if isinstance(gateb_fact, Mapping):
        gateb_mcq = gateb_fact.get("mcq")
        gateb_open = gateb_fact.get("open")
        question_options_preserved = bool(
            question_options_preserved
            and isinstance(gateb_mcq, Mapping)
            and isinstance(gateb_open, Mapping)
            and gateb_mcq.get("stem") == mcq.get("stem")
            and gateb_mcq.get("options_space") == options
            and gateb_open.get("stem") == open_fact.get("stem"))
        gateb_fact_gold_matches = bool(
            isinstance(gateb_mcq, Mapping)
            and isinstance(gateb_open, Mapping)
            and gateb_mcq.get("truth_option") == gateb_gold
            and gateb_open.get("truth_value") == gateb_gold)
    main_target = target_item
    structure_ok = bool(
        question_options_preserved
        and mcq.get("stem") == open_fact.get("stem")
        and gateb_fact_gold_matches is not False)
    ok = bool(gateb_gold != main_gold and structure_ok)
    return {
        "status": "pass" if ok else "reject",
        "main_gold": main_gold,
        "gateb_gold": gateb_gold,
        "gateb_open_gold": gateb_gold,
        "expected_relation": "flip",
        "relation_satisfied": ok,
        "question_stem_preserved": True,
        "question_options_preserved": question_options_preserved,
        "gateb_fact_gold_matches": gateb_fact_gold_matches,
        "fixed_question": fixed_question,
        "speech_identity_join": {
            "main_endpoint_id": main_target["main_endpoint_id"],
            "gateb_endpoint_id": main_target["gateb_endpoint_id"],
            "main_slot": main_target["main_slot"],
            "gateb_slot": main_target["gateb_slot"],
            "entity_instance_id": main_target["entity_instance_id"],
            "join_key": "binding.entity_instance_id",
        },
        "boundary": (
            "engine gold recomputation only; no pixel visibility, audio "
            "audibility, or modality certification"),
    }


def compute(pid, profile, fact, selection, timeline, program, params,
            *, main_selection=None, main_endpoint_registry=None,
            gateb_endpoint_registry=None, gateb_fact=None):
    main = fact["mcq"]["truth_option"]
    if pid in {"card13", "card14"}:
        missing = [
            name for name, value in (
                ("main_selection", main_selection),
                ("main_endpoint_registry", main_endpoint_registry),
                ("gateb_endpoint_registry", gateb_endpoint_registry),
            ) if value is None
        ]
        if missing:
            raise ValueError(
                "card13/card14 Gate-B recomputation requires explicit "
                + ", ".join(missing))
        return _speech_gateb_compute(
            pid, fact, selection, timeline, program,
            main_selection=main_selection,
            main_endpoint_registry=main_endpoint_registry,
            gateb_endpoint_registry=gateb_endpoint_registry,
            gateb_fact=gateb_fact)
    if pid in PIXEL:
        return {
            "status": "pixel_pending",
            "main_gold": main,
            "gateb_gold": None,
            "reason": "native Gate-B pixel truth required",
        }
    if pid in {"card1F", "card1B", "card2"}:
        slot = fact["target_slot"]
        frame = fact["query_frame"]
        angle = recompute_azimuth(timeline, slot, frame)
        gold = band_label(profile, angle)
        open_gold = round(angle, 3)
        separated = abs(
            (float(fact["open"]["truth_value"]) - angle + 180) % 360 - 180
        ) > 2 * float(params["THETA_HALF"])
    elif pid == "card3":
        slot = min(slot_events(program), key=lambda x: x[1])[0]
        angle = recompute_azimuth(timeline, slot, fact["query_frame"])
        gold = "left" if angle < 0 else "right"
        open_gold = gold
        separated = gold != main
    elif pid == "card4R":
        frame = fact["query_frame"]
        ds = {s: distance(timeline, s, frame) for s in ("source1", "source2")}
        slot = min(ds, key=ds.get)
        gold = coat(selection, slot)
        open_gold = gold
        separated = gold != main
    elif pid in {"card5", "card5R"}:
        slot = fact["target_slot"]
        a, b = profile["relation_frames"]
        delta = distance(timeline, slot, b) - distance(timeline, slot, a)
        m = float(profile["min_distance_change_cm"])
        gold = "closer" if delta <= -m else "farther" if delta >= m else None
        open_gold = gold
        separated = gold is not None and gold != main
    elif pid in {"card6", "card6R", "card10"}:
        slot = fact["target_slot"]
        a, b = profile["motion_frames"]
        p = np.asarray(state(timeline, slot, b)["translation_ue_cm"], float)
        q = np.asarray(state(timeline, slot, a)["translation_ue_cm"], float)
        d = float(np.linalg.norm(p - q))
        m = float(profile["min_motion_cm"])
        gold = "moving" if d >= m else "still" if d <= 1e-6 else None
        open_gold = gold
        separated = gold is not None and gold != main
    elif pid == "card7":
        q = fact["truth"]["query_frame"]
        active = []
        for slot, e in slot_events(program):
            raw = next(x for x in program["events"] if x["start_sample"] / 16000.0 == e)
            if raw["start_sample"] <= q / 15 * 16000 < raw["end_sample_exclusive"]:
                active.append(slot)
        if len(active) != 1:
            gold = None
        else:
            gold = coat(selection, active[0])
        open_gold = gold
        separated = gold is not None and gold != main
    elif pid == "card8":
        target_coat = fact["target_coat"]
        target = next(
            s for s in ("source1", "source2") if coat(selection, s) == target_coat
        )
        first = min(t for s, t in slot_events(program) if s == target)
        options = fact["mcq"]["options_space"]
        parsed = [
            tuple(float(v.strip()) for v in option.strip("[]() ").split(","))
            for option in options
        ]
        idx = next((i for i, (lo, hi) in enumerate(parsed) if lo <= first < hi), None)
        gold = None if idx is None else options[idx]
        open_gold = first
        separated = idx is not None and abs(
            first - float(fact["open"]["truth_value"])
        ) > float(params["T_HALF"])
    elif pid == "card9":
        first = min(slot_events(program), key=lambda x: x[1])[0]
        gold = coat(selection, first)
        open_gold = gold
        separated = gold != main
    elif pid == "card15b":
        gold = len(program["events"])
        open_gold = gold
        separated = gold == main
    elif pid == "card17":
        target_asset = next(
            x["asset_id"]
            for x in read(fact["_main_selection"])["actors"]
            if x["source_slot_id"] == "source1"
        )
        target = next(
            x["source_slot_id"]
            for x in selection["actors"]
            if x["asset_id"] == target_asset
        )
        angle = recompute_azimuth(timeline, target, 40)
        bs = profile["location_bands_deg"]
        ls = profile["location_band_labels"]
        ms = [ls[i] for i, (a, b) in enumerate(bs) if a <= angle < b]
        gold = ms[0] if len(ms) == 1 else None
        open_gold = gold
        separated = gold is not None and gold != main
    else:
        raise ValueError(pid)
    relation = "preserve" if pid == "card15b" else "flip"
    ok = (gold == main) if relation == "preserve" else (gold != main and separated)
    return {
        "status": "pass" if ok else "reject",
        "main_gold": main,
        "gateb_gold": gold,
        "gateb_open_gold": open_gold,
        "expected_relation": relation,
        "relation_satisfied": bool(ok),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-manifest", type=Path, required=True)
    ap.add_argument("--dual-gateb-manifest", type=Path, required=True)
    ap.add_argument("--profiles", type=Path, required=True)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    a = ap.parse_args()
    if a.output_root.exists():
        return 2
    pilot = read(a.pilot_manifest)
    dual = read(a.dual_gateb_manifest)
    prof = {x["id"]: x for x in read(a.profiles)}
    params = read(a.params)
    dualmap = {x["pilot_id"]: x for x in dual["records"]}
    rows = []
    for room in pilot["rooms"].values():
        for pid, e in room["profiles"].items():
            for c in e.get("candidates", []):
                fact = read(c["artifacts"]["fact"])
                fact["_main_selection"] = c["artifacts"]["actor_selection"]
                main_selection = None
                main_endpoint_registry = None
                gateb_endpoint_registry = None
                gateb_fact = None
                if c["gateb_status"] == "materialized":
                    g = Path(c["artifacts"]["gateb"]).parent
                    manifest = read(c["artifacts"]["gateb"])
                else:
                    manifest = dualmap[c["pilot_id"]]
                    g = Path(manifest["artifacts"]["timeline"]).parent
                selection = read(g / "actor_selection_gateB.json")
                timeline = read(g / "timeline_gateB.json")
                if pid in {"card13", "card14"}:
                    main_selection_path = Path(
                        c["artifacts"]["actor_selection"])
                    main_selection = read(main_selection_path)
                    main_endpoint_paths = [
                        c["artifacts"].get("source_endpoints"),
                        c["artifacts"].get("endpoint_registry"),
                        c["artifacts"].get("endpoints"),
                        str(main_selection_path.parent / "source_endpoints.json"),
                    ]
                    main_endpoint_path = next(
                        (Path(path) for path in main_endpoint_paths
                         if path and Path(path).is_file()), None)
                    if main_endpoint_path is None:
                        raise ValueError(
                            f"{c['pilot_id']}: missing main source endpoint registry")
                    main_endpoint_registry = read(main_endpoint_path)
                    gateb_endpoint_paths = [
                        manifest.get("artifacts", {}).get("endpoint_registry")
                        if isinstance(manifest, Mapping) else None,
                        str(g / "source_endpoints_gateB.json"),
                        str(g / "source_endpoint_registry.json"),
                    ]
                    gateb_endpoint_path = next(
                        (Path(path) for path in gateb_endpoint_paths
                         if path and Path(path).is_file()), None)
                    if gateb_endpoint_path is None:
                        raise ValueError(
                            f"{c['pilot_id']}: missing Gate-B source endpoint registry")
                    gateb_endpoint_registry = read(gateb_endpoint_path)
                    gateb_fact_path = g / "fact_record_gateB.json"
                    if gateb_fact_path.is_file():
                        gateb_fact = read(gateb_fact_path)
                    # Gate-B keeps the main program's speech events fixed;
                    # endpoint identities are joined against the two registries.
                    main_program_path = c["artifacts"].get("main_program")
                    if main_program_path is None:
                        main_program_path = str(
                            Path(c["artifacts"]["fact"]).parent
                            / "audio_program.json")
                    program = read(Path(main_program_path))
                else:
                    program = read(
                        g
                        / (
                            "audio_program_gateB.json"
                            if (g / "audio_program_gateB.json").is_file()
                            else "audio_program.json"
                        )
                    )
                result = compute(
                    pid, prof[pid], fact, selection, timeline, program, params,
                    main_selection=main_selection,
                    main_endpoint_registry=main_endpoint_registry,
                    gateb_endpoint_registry=gateb_endpoint_registry,
                    gateb_fact=gateb_fact,
                )
                rows.append(
                    {
                        "pilot_id": c["pilot_id"],
                        "profile_id": pid,
                        "gateb_root": str(g),
                        **result,
                    }
                )
    counts = Counter(x["status"] for x in rows)
    a.output_root.mkdir(parents=True)
    out = {
        "schema": "qa_v3_gateb_gold_recompute_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "counts": dict(counts),
        "rows": rows,
    }
    write(a.output_root / "gateb_gold_manifest.json", out)
    rowmap = {row["pilot_id"]: row for row in rows}
    augmented = copy.deepcopy(pilot)
    augmented["schema"] = "qa_v3_room_centric_pilot_augmented_gateb_v1"
    augmented["gateb_gold_manifest"] = str(
        (a.output_root / "gateb_gold_manifest.json").resolve()
    )
    route_profiles = {
        "card1F",
        "card1B",
        "card2",
        "card3",
        "card5",
        "card5R",
        "card6",
        "card6R",
        "card10",
        "card15a",
        "card16",
    }
    for room in augmented["rooms"].values():
        for entry in room["profiles"].values():
            for candidate in entry.get("candidates", []):
                row = rowmap[candidate["pilot_id"]]
                candidate["gateb"] = {
                    "root": row["gateb_root"],
                    "gold_status": row["status"],
                    "main_gold": row["main_gold"],
                    "gateb_gold": row["gateb_gold"],
                    "audio_policy": (
                        "route_audio_must_change_consistently"
                        if row["profile_id"] in route_profiles
                        else "appearance_reuse_main_audio_no_rerender"
                    ),
                }
    write(a.output_root / "augmented_pilot_manifest.json", augmented)
    print(json.dumps({"counts": dict(counts), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

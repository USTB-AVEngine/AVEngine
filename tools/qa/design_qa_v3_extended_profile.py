#!/usr/bin/env python3
"""Generate the QA-v3 profiles that require N actors, pixel truth, or segments.

This is the scene-neutral companion to design_qa_v3_scene_batch.py.  It
realises geometry/timeline/AudioProgram/fact candidates for cards 11, 12, 13,
14, 15a, 16 and 17.  Pixel-dependent truths remain explicitly pending until
the native pixel pass is joined.  Missing semantic assets are reported as a
resource-unavailable attempt, not as an unimplemented profile or a scene
infeasibility claim.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys

import numpy as np
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

from build_qa_v3_n_actor_canary import (  # noqa: E402
    DEFAULT_ASSETS,
    NRouteSearchExhausted,
    _read,
    _write,
    build_endpoint_registry,
    find_n_route_plan,
    seed_uint64,
)
from build_qa_v3_programs import (  # noqa: E402
    build_program, dry_canvas_window_fields, program_request_fields,
    require_dry_canvas_source_mode, validate_m6_audio_program,
)
from qa_v3_actor_selection import _actor_entry  # noqa: E402
from design_qa_v3_scene_batch import (  # noqa: E402
    git_worktree_state,
    resolve_scene_render_context,
)
from make_idle_then_walk_timeline import (  # noqa: E402
    resample_route_samples,
    transform_to_solved_routes,
)
from scene_sampler import (  # noqa: E402
    effective_half_fov,
    load_scene,
    relative_azimuth_deg,
    require_camera_clearance,
)
from qa_v3_request import answer_forms_from_params, write_requested_questions
from audio_profiles import AudioProfileSearchExhausted, schedule_speech_utterances  # noqa: E402
from avengine.assets.sound_pool import clip_source_from_params  # noqa: E402
import scene_sampler as SS  # noqa: E402
from avengine.camera_pose import apply_camera_listener_pose_ue  # noqa: E402
from avengine.timeline.current_apartment_visual import (  # noqa: E402
    author_current_n_actor_visual_timeline,
)


SUPPORTED = {"card11", "card12", "card13", "card14", "card15a", "card16", "card17"}
EVENT_STARTS = (8000, 24000, 40000, 56000)
# Card11 binds audible direction to native pixel state at zero-based frame 30.
# 30,000 samples spans that frame at 16 kHz/15 fps (roughly frames 28..32),
# so event time, N-route binding and pixel authority observe the same moment.
CARD11_BINDING_FRAME = 30
CARD11_EVENT_START_SAMPLE = 30000
VISIBILITY_OPTIONS = (
    "visible_clear",
    "visible_occluded",
    "fully_occluded",
    "out_of_view",
)


class SearchExhausted(RuntimeError):
    """Finite randomized search ended without a candidate."""

    def __init__(self, message: str, *, evaluated_combinations: int):
        super().__init__(message)
        self.evaluated_combinations = int(evaluated_combinations)


def _speech_actor_count(profile):
    count = profile.get("actor_count", 4)
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("speech contrast profiles need an integer actor_count >= 2")
    return count


def _resource_inventory(profile_id, assets, sounds, speech_pool=None, profile=None):
    dogs = [
        item for item in assets
        if item.get("identity", {}).get("species_id") == "dog"
        and item.get("admission_state") == "research"
        and item.get("asset_id") in DEFAULT_ASSETS
    ]
    humans_by_colour = {}
    for item in assets:
        colour = item.get("realized_attributes", {}).get("top_color")
        if (item.get("identity", {}).get("species_id") == "human"
                and isinstance(colour, str) and colour.strip()):
            humans_by_colour.setdefault(colour.strip(), item)
    humans = list(humans_by_colour.values())
    sound_types_by_label = {}
    for item in sounds:
        taxonomy = item.get("taxonomy_path")
        label = taxonomy[-1] if isinstance(taxonomy, list) and taxonomy else None
        if (item.get("semantic_sound_class") in {
                "animal_vocalization", "human_speech", "test_signal"}
                and isinstance(label, str) and label):
            sound_types_by_label.setdefault(label, item)
    sound_types = list(sound_types_by_label.values())
    speech_source = sounds if speech_pool is None else speech_pool
    speech_by_identity = {}
    for item in speech_source:
        if not isinstance(item, dict):
            continue
        sound_class = item.get("semantic_sound_class") or item.get("event_class")
        transcript = item.get("transcript")
        if sound_class not in {"human_speech", "speech_playback"}:
            continue
        if not isinstance(transcript, str) or not transcript.strip():
            continue
        if speech_pool is not None and item.get("split") != "train":
            continue
        identity = (
            item.get("speaker_id"),
            item.get("utterance_id"),
        )
        key = identity if all(identity) else ("transcript", transcript.strip())
        speech_by_identity.setdefault(key, item)
    speech = list(speech_by_identity.values())
    missing = []
    if profile_id in {"card11", "card15a"} and len(dogs) < 4:
        missing.append("four_registered_dog_assets")
    if profile_id in {"card16", "card17"} and len(dogs) < 2:
        missing.append("two_registered_dog_assets")
    if profile_id == "card12" and len(sound_types) < 4:
        missing.append("four_registered_semantic_sound_types")
    requirements = {}
    if profile_id in {"card13", "card14"}:
        profile = profile or {"id": profile_id}
        count = _speech_actor_count(profile)
        for key, default in (("required_controlled_colours", count), ("required_transcripts", count)):
            value = profile.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < count:
                raise ValueError(f"{key} must be an integer >= actor_count")
            requirements[key] = value
        if len(humans) < requirements["required_controlled_colours"]:
            missing.append("controlled_human_top_colours")
        if len(speech) < requirements["required_transcripts"]:
            missing.append("transcribed_speech_assets")
    return {
        "dogs": dogs,
        "humans": humans,
        "sound_types": sound_types,
        "speech": speech,
        "missing": missing,
        "requirements": requirements,
    }


def _selection(assets, by_id, snapshot_content):
    return {
        "schema": "avengine_n_actor_selection_v1",
        "asset_authorization": "verified_internal",
        "research_only": True,
        "qualification_claim": False,
        "claim_boundary": "QA v3 extended-profile research candidate only",
        "actors": [
            _actor_entry(f"source{index}", asset_id, by_id, snapshot_content)
            for index, asset_id in enumerate(assets, start=1)
        ],
    }


def _route_source_counts(records):
    """How many actor routes in this batch came from the bank vs were designed.

    Counts the per-point summaries the batch collected, including a card17
    second segment, so the manifest number cannot drift from the facts.
    """
    counts = {"bank": 0, "synthesized": 0, "unknown": 0}
    designed_points = 0
    for record in records:
        sources = list(record.get("route_sources") or [])
        sources += list(record.get("segment2_route_sources") or [])
        for source in sources:
            counts[str(source) if str(source) in counts else "unknown"] += 1
        if any(str(source) == "synthesized" for source in sources):
            designed_points += 1
    return dict(counts, points_with_a_designed_route=designed_points,
                points=len(records))


def _timeline_dimensions(params) -> tuple[int, float]:
    try:
        clip_seconds = float(params["CLIP_SECONDS"])
        frame_rate_hz = float(params["VIDEO_FPS"])
        frame_count = SS.frame_count_from_params(params)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "speech timeline needs CLIP_SECONDS, VIDEO_FPS, and FRAME_COUNT"
        ) from exc
    if (
        not math.isfinite(clip_seconds)
        or clip_seconds <= 0.0
        or not math.isfinite(frame_rate_hz)
        or frame_rate_hz <= 0.0
        or frame_count < 2
    ):
        raise ValueError("timeline duration and frame clock must be positive and finite")
    expected = clip_seconds * frame_rate_hz
    if not math.isclose(frame_count, expected, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"FRAME_COUNT={frame_count} disagrees with "
            f"CLIP_SECONDS*VIDEO_FPS={expected}"
        )
    return frame_count, frame_rate_hz


def _author_timeline(out_dir, name, selection_path, registry_path, scene, plan, params):
    render = resolve_scene_render_context(scene)
    ground = float(render["ground_z_ue_cm"])
    frame_count, frame_rate_hz = _timeline_dimensions(params)
    routes_3d = {
        f"source{index}": [
            [float(x), float(y), ground]
            for x, y in resample_route_samples(route.samples_xy, frame_count)
        ]
        for index, route in enumerate(plan["routes"], start=1)
    }
    camera_height_m = float(plan.get("camera_height_m") or scene.camera_height_m)
    camera_ue = [
        float(plan["camera_xy"][0]),
        float(plan["camera_xy"][1]),
        ground + camera_height_m * 100.0,
    ]
    authored = out_dir / f"{name}_authored.json"
    timeline = author_current_n_actor_visual_timeline(
        actor_selection_path=selection_path,
        source_asset_registry_path=registry_path,
        output_path=authored,
        camera_position_ue_cm=camera_ue,
        camera_yaw_deg=float(plan["camera_yaw_deg"]),
        routes_by_slot_ue_cm=routes_3d,
        native_map=str(render["native_map"]),
        room_profile_id=str(render["room_profile_id"]),
        hfov_degrees=scene.hfov_deg,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
    )
    timeline = transform_to_solved_routes(
        timeline,
        {
            slot: [(point[0], point[1]) for point in route]
            for slot, route in routes_3d.items()
        },
    )
    path = out_dir / f"{name}.json"
    _write(path, timeline)
    return path, timeline, camera_ue


def _asset_position_tracks(selection, timeline):
    slot_to_asset = {
        actor["source_slot_id"]: actor["asset_id"]
        for actor in selection["actors"]
    }
    tracks = {asset_id: [] for asset_id in slot_to_asset.values()}
    for frame in timeline["frames"]:
        states = {
            state["source_slot_id"]: state
            for state in frame["actor_states"]
        }
        for slot, asset_id in slot_to_asset.items():
            tracks[asset_id].append(tuple(
                float(value)
                for value in states[slot]["translation_ue_cm"]))
    return {key: tuple(value) for key, value in tracks.items()}


def _assert_gateb_visual_change(
        main_selection, main_timeline, gateb_selection, gateb_timeline):
    main_tracks = _asset_position_tracks(main_selection, main_timeline)
    gateb_tracks = _asset_position_tracks(gateb_selection, gateb_timeline)
    if main_tracks == gateb_tracks:
        raise RuntimeError(
            "Gate B changed only slot labels; per-asset visual tracks are identical")
    return {
        "main_asset_count": len(main_tracks),
        "gateb_asset_count": len(gateb_tracks),
        "per_asset_tracks_changed": True,
    }


def _find_gateb_out_of_view_route(scene, params, plan, *, frame=30):
    if "MIN_CAMERA_DISTANCE_CM" not in params:
        raise ValueError("params missing MIN_CAMERA_DISTANCE_CM")
    half_fov = effective_half_fov(scene, params)
    minimum_distance = float(params["MIN_CAMERA_DISTANCE_CM"])
    used = {route.route_id for route in plan["routes"]}
    evaluated = 0
    for route in scene.routes:
        if route.route_id in used or route.displacement_cm <= 1.0e-6:
            continue
        evaluated += 1
        point = route.at(frame)
        azimuth = relative_azimuth_deg(
            plan["camera_xy"], float(plan["camera_yaw_deg"]), point)
        if (abs(azimuth) > half_fov
                and math.dist(plan["camera_xy"], point)
                >= minimum_distance):
            return route, evaluated
    raise SearchExhausted(
        "Gate B found no real route outside the query-frame view",
        evaluated_combinations=evaluated)


def audio_program_mode(events) -> str:
    """Mode follows the event list: one slot sounding vs several."""

    active = {
        event["slot"] if isinstance(event, dict) else event[0]
        for event in events
    }
    return "one_active_of_n" if len(active) == 1 else "sequential_sources"


def _program_events(profile_id, cell_index, sound_assets):
    sounds = [item["sound_asset_id"] for item in sound_assets]
    bark = "dog_beagle_v2_scheduled_dry"
    slots = [f"source{index}" for index in range(1, 5)]
    if profile_id == "card11":
        positive = cell_index % 2 == 0
        target_index = (cell_index // 2) % 3
        target_slot = slots[target_index]
        main_slot = target_slot if positive else "source4"
        gatea_slot = "source4" if positive else target_slot
        return (
            [(main_slot, CARD11_EVENT_START_SAMPLE, bark)],
            [(gatea_slot, CARD11_EVENT_START_SAMPLE, bark)],
            {
                "target_slot": target_slot,
                "desired_answer": target_slot if positive else "none",
                "gatea_desired_answer": "none" if positive else target_slot,
            },
        )
    if profile_id in {"card12", "card13", "card14"}:
        target_index = cell_index % 4
        swap_index = (target_index + 1) % 4
        main = [
            (slot, start, sound)
            for slot, start, sound in zip(slots, EVENT_STARTS, sounds[:4])
        ]
        gatea = list(main)
        target_event = main[target_index]
        swap_event = main[swap_index]
        gatea[target_index] = (
            target_event[0], target_event[1], swap_event[2])
        gatea[swap_index] = (
            swap_event[0], swap_event[1], target_event[2])
        key = ("target_sound_asset_id" if profile_id == "card12"
               else "target_speech_asset_id")
        return main, gatea, {
            "target_index": target_index,
            "gatea_source_index": swap_index,
            key: target_event[2],
        }
    if profile_id == "card15a":
        distinct = cell_index % 4 + 1
        gatea_distinct = 5 - distinct
        main = [
            (slots[index % distinct], start, bark)
            for index, start in enumerate(EVENT_STARTS)
        ]
        gatea = [
            (slots[index % gatea_distinct], start, bark)
            for index, start in enumerate(EVENT_STARTS)
        ]
        return main, gatea, {
            "in_scene_count": 4,
            "distinct_callers": distinct,
            "gatea_distinct_callers": gatea_distinct,
        }
    main = [
        ("source1", EVENT_STARTS[0], bark),
        ("source2", EVENT_STARTS[1], bark),
    ]
    gatea = [
        ("source2", EVENT_STARTS[0], bark),
        ("source1", EVENT_STARTS[1], bark),
    ]
    return main, gatea, {"first_caller_slot": "source1",
                         "gatea_first_caller_slot": "source2"}


def _speech_pool_rows(params):
    path = params.get("SOUND_EVENT_POOL")
    if not path:
        return None
    payload = _read(Path(path))
    rows = payload.get("clips") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"speech event pool {path} has no clips list")
    return [
        row for row in rows
        if isinstance(row, dict) and row.get("event_class") == "speech_playback"
    ]


def _speech_schedule(params, *, cell_index, seed, utterance_count=4):
    source_rng = np.random.default_rng(
        seed_uint64(f"{seed}|speech-source|{cell_index}"))
    layout_rng = np.random.default_rng(
        seed_uint64(f"{seed}|speech-layout|{cell_index}"))
    clip_source = clip_source_from_params(
        params, source_rng, pair_kind="human")
    if clip_source is None:
        raise ValueError(
            "card13/card14 require SOUND_SOURCE_MODE=event_pool with speech clips"
        )
    return schedule_speech_utterances(
        layout_rng,
        params=params,
        clip_source=clip_source,
        roles=[f"source{index}" for index in range(1, utterance_count + 1)],
        utterance_count=utterance_count,
        split=str(params.get("SPEECH_SPLIT", "train")),
    )


def _speech_program_events(
    schedule,
    cell_index,
    *,
    target_rng=None,
    option_rng=None,
):
    role_to_slot = {role: role for role in schedule.declared["role_order"]}
    main = schedule.program_events(role_to_slot)
    shift = 1 + (cell_index % (len(main) - 1))
    gatea = [
        dict(row, slot=f"source{((index + shift) % len(main)) + 1}")
        for index, row in enumerate(main)
    ]
    if target_rng is None:
        target_index = cell_index % len(main)
    else:
        target_index = int(target_rng.permutation(len(main))[0])
    if option_rng is None:
        option_order = list(range(len(main)))
    else:
        option_order = [
            int(index) for index in option_rng.permutation(len(main))
        ]
    return main, gatea, {
        "target_index": target_index,
        "gatea_source_index": (target_index + shift) % len(main),
        "option_order": option_order,
        "target_speech_asset_id": main[target_index]["sound_asset_id"],
        "target_speech_utterance_id": main[target_index]["utterance_id"],
    }


def _speech_question_context(main_events, colour_by_slot, truth):
    target_index = int(truth["target_index"])
    if not 0 <= target_index < len(main_events):
        raise ValueError("speech target index is outside the main event list")
    target = main_events[target_index]
    option_order = list(truth.get("option_order", range(len(main_events))))
    if sorted(option_order) != list(range(len(main_events))):
        raise ValueError("speech question option order is not a permutation")
    return {
        "target_index": target_index,
        "target_slot": target["slot"],
        "target_colour": colour_by_slot[target["slot"]],
        "target_transcript": str(target["transcript"]).strip(),
        "target_identity": {
            "sound_asset_id": target["sound_asset_id"],
            "speaker_id": target.get("speaker_id"),
            "utterance_id": target.get("utterance_id"),
        },
        "option_order": option_order,
        "option_transcripts": [
            str(row["transcript"]).strip() for row in main_events
        ],
        "option_colours": [
            colour_by_slot[row["slot"]] for row in main_events
        ],
    }


def _speech_bindings(events, colour_by_slot):
    return [
        {
            "slot": row["slot"],
            "sound_asset_id": row["sound_asset_id"],
            "speaker_id": row.get("speaker_id"),
            "utterance_id": row.get("utterance_id"),
            "transcript": row.get("transcript"),
            "split": row.get("split"),
            "colour": colour_by_slot[row["slot"]],
            "start_sample": row["start_sample"],
            "duration_samples": row["duration_samples"],
        }
        for row in events
    ]


def _find_card16_plan(scene, params, *, seed, max_attempts):
    half_fov = effective_half_fov(scene, params)
    evaluated = 0
    for outer in range(200):
        try:
            plan = find_n_route_plan(
                scene, params, actor_count=2,
                seed=f"{seed}|final-state-split|{outer}",
                binding_frames=(12,), max_attempts=max_attempts)
        except NRouteSearchExhausted as error:
            evaluated += error.evaluated_combinations
            continue
        evaluated += int(plan["search_attempts"])
        final_inside = [
            abs(relative_azimuth_deg(
                plan["camera_xy"], float(plan["camera_yaw_deg"]),
                route.at(74))) <= half_fov
            for route in plan["routes"]
        ]
        if sum(final_inside) == 1:
            plan["search_attempts"] = evaluated
            plan["card16_final_fov_membership"] = final_inside
            return plan
    raise SearchExhausted(
        "no card16 plan with distinct final in-view/out-of-view states",
        evaluated_combinations=evaluated)


def _plan_signature(plan):
    return (
        tuple(float(value) for value in plan["camera_xy"]),
        float(plan["camera_yaw_deg"]),
        tuple(
            (route.route_id, tuple(route.samples_xy))
            for route in plan["routes"]),
    )


def _location_bands_and_labels(profile, scene, params):
    """A profile's answer bands: derived from its domain, or its written table.

    card17 writes ``location_bands_deg`` out to +-52.5 because this rig's HFOV is
    105, and carries the same dead zone the base profiles do -- the visibility
    gate stops at effective_half_fov = 47.5, so 5.0 deg of each outer band is
    unreachable.  Writing degrees also means every new room needs the table
    edited by hand, and an off-screen family would need eight pairs per room.

    A profile may instead declare ``answer_domain`` and ``answer_shape``, and the
    degrees come from the scene's own camera (see scene_sampler.derive_answer_bands).
    Labels stay optional: given, they must match the derived band count; absent,
    the bands are numbered, because a name for a direction depends on the
    published convention and that belongs at the publication edge, not here.

    A profile that declares neither keeps working exactly as before.
    """

    if profile.get("answer_domain") is not None:
        # 域推导要用 VISUAL_FOV_MARGIN_DEG。拿不到 params 时 effective_half_fov 会
        # 用边距 0,推出 ±52.5 而不是 ±47.5——正好是这套机制要消灭的那个 5.0 度死区,
        # 而且不会报错。所以这里拒绝,不默认。
        if not params or "VISUAL_FOV_MARGIN_DEG" not in params:
            raise ValueError(
                f"{profile.get('id')}: answer_domain "
                f"{profile['answer_domain']!r} needs params carrying "
                "VISUAL_FOV_MARGIN_DEG; without it the derived bands would "
                "silently use a zero margin and reach past what the camera "
                "can be trusted to show")
        bands = [list(b) for b in SS.derive_answer_bands(profile, scene, params)]
        labels = profile.get("location_band_labels")
        if labels is None:
            labels = [f"sector_{i}" for i in range(len(bands))]
        elif len(labels) != len(bands):
            raise ValueError(
                f"{profile.get('id')}: {len(labels)} labels for "
                f"{len(bands)} bands derived from domain "
                f"{profile['answer_domain']!r}")
        return bands, list(labels)
    bands = profile.get("location_bands_deg")
    labels = profile.get("location_band_labels")
    if (not isinstance(bands, list) or not isinstance(labels, list)
            or len(bands) != len(labels)):
        raise ValueError(
            f"{profile.get('id')}: location bands and labels are required")
    return bands, labels


def _location_band(profile, scene, plan, route_index, frame=40, params=None):
    azimuth = relative_azimuth_deg(
        plan["camera_xy"], float(plan["camera_yaw_deg"]),
        plan["routes"][route_index].at(frame))
    bands, labels = _location_bands_and_labels(profile, scene, params or {})
    matches = [
        label for label, (lo, hi) in zip(labels, bands)
        if float(lo) <= azimuth < float(hi)
    ]
    if len(matches) != 1:
        raise SearchExhausted(
            f"route {route_index} azimuth {azimuth:.3f} is outside the "
            "declared location bands", evaluated_combinations=1)
    return matches[0], float(azimuth)


def _facts(
    profile,
    inventory,
    truth,
    scene,
    main_plan,
    segment2_plan=None,
    *,
    speech_events=None,
    colour_by_slot=None,
    speech_schedule=None,
):
    profile = {"id": profile} if isinstance(profile, str) else profile
    profile_id = profile["id"]
    if profile_id == "card11":
        labels = [item["display_label"] for item in inventory["dogs"][:3]]
        target_index = int(truth["target_slot"].removeprefix("source")) - 1
        desired = (labels[target_index]
                   if truth["desired_answer"] != "none" else "none")
        gatea_desired = (labels[target_index]
                         if truth["gatea_desired_answer"] != "none" else "none")
        return {
            "truth_status": "pending_native_pixel_join",
            "target_slot": truth["target_slot"],
            "desired_truth": desired,
            "gatea_desired_truth": gatea_desired,
            "mcq": {
                "stem": "Which visible dog made the sound?",
                "options_space": [*labels, "none"],
                "truth_option": desired,
            },
            "open": {
                "stem": "Which visible dog, if any, made the sound?",
                "truth_value": desired,
                "scoring": "closed_set",
            },
            "pixel_acceptance": {
                "source1_source2_source3": "visible",
                "source4": "fully_occluded",
                "binding_frame": CARD11_BINDING_FRAME,
            },
        }
    if profile_id == "card12":
        target_index = int(truth["target_index"])
        sound = inventory["sound_types"][target_index]
        label = sound["taxonomy_path"][-1]
        appearance = inventory["dogs"][target_index]["display_label"]
        stem = f"What sound did the {appearance} make?"
        return {
            "truth_status": "engine_exact",
            "target_index": target_index,
            "mcq": {"stem": stem,
                    "options_space": [
                        item["taxonomy_path"][-1]
                        for item in inventory["sound_types"][:4]],
                    "truth_option": label},
            "open": {"stem": stem,
                     "truth_value": label, "scoring": "closed_set"},
        }
    if profile_id in {"card13", "card14"}:
        if speech_events is not None:
            if colour_by_slot is None:
                raise ValueError("speech facts need colour_by_slot")
            question = truth.get("speech_question")
            if question is None:
                legacy_index = int(truth["target_index"])
                question = _speech_question_context(
                    speech_events, colour_by_slot,
                    {"target_index": legacy_index},
                )
            option_order = list(
                question.get("option_order", range(len(speech_events)))
            )
            if sorted(option_order) != list(range(len(speech_events))):
                raise ValueError("speech question option order is not a permutation")

            def text_key(text):
                return " ".join(str(text).split()).casefold()

            if profile_id == "card13":
                options = question["option_transcripts"]
                if len({text_key(text) for text in options}) != len(options):
                    raise ValueError("speech MCQ options contain duplicate transcript text")
                matches = [
                    (index, row)
                    for index, row in enumerate(speech_events)
                    if row["slot"] == question["target_slot"]
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "card13 Gate A must have exactly one event for the "
                        "questioned appearance slot"
                    )
                target_index, target = matches[0]
                transcript = str(target["transcript"]).strip()
                mcq = {
                    "stem": (
                        f"What did the person in "
                        f"{question['target_colour']} say?"
                    ),
                    "options_space": [
                        question["option_transcripts"][index]
                        for index in option_order
                    ],
                    "truth_option": transcript,
                }
                open_answer = transcript
                scoring = "transcript_wer"
            else:
                if sum(text_key(row["transcript"]) == text_key(question["target_transcript"])
                       for row in speech_events) != 1:
                    raise ValueError("queried transcript is not unique among speakers")
                identity = question["target_identity"]
                matches = [
                    (index, row)
                    for index, row in enumerate(speech_events)
                    if (
                        row.get("sound_asset_id") == identity["sound_asset_id"]
                        and row.get("speaker_id") == identity["speaker_id"]
                        and row.get("utterance_id") == identity["utterance_id"]
                    )
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "card14 Gate A must preserve exactly one questioned "
                        "transcript identity"
                    )
                target_index, target = matches[0]
                transcript = question["target_transcript"]
                colour = colour_by_slot[target["slot"]]
                mcq = {
                    "stem": (
                        "What colour was the person who said "
                        f"'{question['target_transcript']}'?"
                    ),
                    "options_space": [
                        question["option_colours"][index]
                        for index in option_order
                    ],
                    "truth_option": colour,
                }
                open_answer = colour
                scoring = "closed_set"
            result = {
                "truth_status": "engine_exact",
                "target_index": target_index,
                "target_slot": target["slot"],
                "target_speaker_id": target["speaker_id"],
                "target_utterance_id": target["utterance_id"],
                "question_target_index": question["target_index"],
                "question_target_slot": question["target_slot"],
                "question_target_colour": question["target_colour"],
                "question_target_transcript": question["target_transcript"],
                "question_option_order": option_order,
                "speech_bindings": _speech_bindings(
                    speech_events, colour_by_slot
                ),
                "mcq": mcq,
                "open": {
                    "stem": mcq["stem"],
                    "truth_value": open_answer,
                    "scoring": scoring,
                },
            }
            if speech_schedule is not None:
                result["speech_schedule"] = dict(speech_schedule.declared)
            return result
        target_index = int(truth["target_index"])
        speech = inventory["speech"][target_index]
        transcript = speech["transcript"].strip()
        colour = inventory["humans"][target_index][
            "realized_attributes"]["top_color"]
        if profile_id == "card13":
            mcq = {
                "stem": f"What did the person in {colour} say?",
                "options_space": [
                    item["transcript"].strip()
                    for item in inventory["speech"][:4]],
                "truth_option": transcript,
            }
            open_answer = transcript
        else:
            mcq = {
                "stem": f"What colour was the person who said '{transcript}'?",
                "options_space": [
                    item["realized_attributes"]["top_color"]
                    for item in inventory["humans"][:4]],
                "truth_option": colour,
            }
            open_answer = colour
        return {
            "truth_status": "engine_exact",
            "mcq": mcq,
            "open": {"stem": mcq["stem"], "truth_value": open_answer,
                     "scoring": "transcript_wer" if profile_id == "card13"
                     else "closed_set"},
        }
    if profile_id == "card15a":
        value = [truth["in_scene_count"], truth["distinct_callers"]]
        return {
            "truth_status": "engine_exact_plus_pixel_presence_check",
            "mcq": {
                "stem": "How many dogs are in the scene, and how many barked?",
                "options_space": [[4, count] for count in range(1, 5)],
                "truth_option": value,
            },
            "open": {
                "stem": "How many dogs are in the scene, and how many barked?",
                "truth_value": value,
                "scoring": "count_pair_all_or_nothing",
            },
            "pixel_acceptance": {"all_four_sources": "visible"},
        }
    if profile_id == "card16":
        return {
            "truth_status": "pending_native_pixel_join",
            "selector": {"first_caller_slot": "source1"},
            "mcq": {
                "stem": "What is the final visibility state of the dog that barked first?",
                "options_space": list(VISIBILITY_OPTIONS),
                "truth_option": None,
            },
            "open": {
                "stem": "Can the dog that barked first be fully seen at the end, and why?",
                "truth_value": None,
                "scoring": "closed_set_plus_engine_chain",
            },
            "pixel_acceptance": {
                "frame": 74,
                "source1_source2_states": "must_differ",
            },
        }
    assert segment2_plan is not None
    band, azimuth = _location_band(
        profile, scene, segment2_plan, 0)
    other_band, other_azimuth = _location_band(
        profile, scene, segment2_plan, 1)
    return {
        "truth_status": "engine_exact_future_extension",
        "selector": {"segment1_first_caller_slot": "source1"},
        "mcq": {
            "stem": "In segment 2, where is the dog that barked first in segment 1?",
            "options_space": list(profile["location_band_labels"]),
            "truth_option": band,
        },
        "open": {
            "stem": "In segment 2, roughly where is that same dog?",
            "truth_value": band,
            "scoring": "closed_set",
        },
        "truth": {
            "segment2_target_azimuth_deg": round(azimuth, 3),
            "segment2_other_azimuth_deg": round(other_azimuth, 3),
            "gatea_truth_option": other_band,
        },
        "qualification_boundary": (
            "future extension; stable asset identity is engine truth, "
            "human recognisability is not certified"),
    }


def _write_failed(out_root, profile_id, scene_id, error, *, cells_requested,
                  completed):
    """Leave a traceable manifest behind when the runner dies mid-batch.

    Without it a half-written batch directory carries no code revision and
    no statement of what failed; the scheduler and a later reader would have
    to guess.  The exception is re-raised by the caller, so this changes no
    outcome, only the evidence left on disk."""
    manifest = {
        "schema": "qa_v3_extended_profile_batch_manifest_v1",
        "status": "failed",
        "qualification_claim": False,
        "evidence_class": "runner_failure",
        "code": git_worktree_state(),
        "scene_id": scene_id,
        "profile_ids": [profile_id],
        "counts": {
            "cells_requested": int(cells_requested),
            "geometry_candidates": int(completed),
            "rejected": 0,
        },
        "failure": {
            "type": type(error).__name__,
            "detail": str(error)[:600],
        },
    }
    _write(out_root / "batch_manifest.json", manifest)
    return manifest


def _write_unavailable(out_root, profile, scene, missing, cells):
    out_root.mkdir(parents=True)
    manifest = {
        "schema": "qa_v3_extended_profile_batch_manifest_v1",
        "status": "research_dev",
        "qualification_claim": False,
        "evidence_class": "resource_unavailable",
        "code": git_worktree_state(),
        "scene_id": scene.scene_id,
        "profile_ids": [profile["id"]],
        "counts": {
            "cells_requested": cells,
            "geometry_candidates": 0,
            "rejected": cells,
        },
        "search": {
            "combinations_evaluated": 0,
            "budget_exhausted": 0,
            "by_reason": {"resource_requirement_unmet": cells},
        },
        "resource_status": {
            "status": "unavailable",
            "method": "registry_preflight",
            "missing": missing,
            "boundary": (
                "The execution path is implemented, but current registered "
                "semantic assets cannot instantiate this profile."),
        },
    }
    _write(out_root / "batch_manifest.json", manifest)
    return manifest


def _runtime_visual_descriptor(
    identifier: str,
    actor_selection: str,
    timeline: str,
    *,
    release: bool,
) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "qa_v3_current_apartment_visual",
        "actor_selection": actor_selection,
        "timeline": timeline,
        "capture": {"status": "pending"},
        "media": {"status": "pending"},
        "release": bool(release),
    }


def _runtime_descriptions(
    profile: Mapping[str, Any], point: Path
) -> dict[str, object]:
    variants = [
        _runtime_visual_descriptor(
            "main", "actor_selection.json", "timeline.json", release=True),
        _runtime_visual_descriptor(
            "gateB", "actor_selection_gateB.json", "timeline_gateB.json",
            release=False),
    ]
    raw_segment_count = profile.get("segment_count", 1)
    if (
        isinstance(raw_segment_count, bool)
        or not isinstance(raw_segment_count, int)
        or raw_segment_count < 1
    ):
        raise ValueError("profile segment_count must be a positive integer")
    segments = []
    release_media = []
    for index in range(1, raw_segment_count + 1):
        segment_id = f"segment{index}"
        timeline = "timeline.json" if index == 1 else f"timeline_{segment_id}.json"
        if not (point / timeline).is_file():
            raise ValueError(
                f"declared {segment_id} timeline is missing: {point / timeline}"
            )
        segments.append({
            **_runtime_visual_descriptor(
                segment_id, "actor_selection.json", timeline, release=True
            ),
            "variant": "main",
        })
        release_media.append({
            "id": segment_id,
            "variant": "main",
            "segment": segment_id,
            "kind": "qa_v3_review_clip",
            "release": True,
            "status": "pending",
        })
    pixel_evidence = []
    pixel_kind = profile.get("pixel_consumer_kind")
    if pixel_kind is not None:
        if not isinstance(pixel_kind, str) or not pixel_kind.strip():
            raise ValueError("profile pixel_consumer_kind must be non-empty text")
        pixel_evidence.append({
            "id": "main",
            "kind": pixel_kind.strip(),
            "fact": "fact_record.json",
            "pixel_truth": None,
            "status": "pending",
        })
    return {
        "runtime_consumer_status": profile.get(
            "runtime_consumer_status", "declared_pending_execution"
        ),
        "visual_variants": variants,
        "segments": segments,
        "pixel_evidence": pixel_evidence,
        "release_media": release_media,
    }

def _realise_cell(out_root, profile, cell_index, scene, params, inventory,
                  by_id, registry_path, base_request, snapshot_content, seed):
    profile_id = profile["id"]
    # Exhausted audio selection must not leave a renderable partial point.
    speech_schedule = (
        _speech_schedule(params, cell_index=cell_index, seed=seed,
                         utterance_count=_speech_actor_count(profile))
        if profile_id in {"card13", "card14"} else None)
    if profile_id in {"card13", "card14"}:
        appearance_rng = np.random.default_rng(
            seed_uint64(f"{seed}|appearance|{cell_index}"))
        humans = list(inventory["humans"])
        count = _speech_actor_count(profile)
        if len(humans) < count:
            raise ValueError("not enough distinct registered human colours for this actor_count")
        order = appearance_rng.permutation(len(humans))[:count]
        actor_assets = [humans[int(index)]["asset_id"] for index in order]
    else:
        actor_assets = [item["asset_id"] for item in inventory["dogs"][
            :2 if profile_id in {"card16", "card17"} else 4]]
    point_id = f"{profile_id}_{cell_index + 1:03d}"
    point = out_root / point_id
    point.mkdir()
    selection = _selection(actor_assets, by_id, snapshot_content)
    selection_path = point / "actor_selection.json"
    _write(selection_path, selection)
    endpoint_path = point / "source_endpoints.json"
    speech_endpoint_classes = (
        {f"source{index}": ["speech_playback"]
         for index in range(1, len(actor_assets) + 1)}
        if profile_id in {"card13", "card14"} else None
    )
    _, endpoint_records = build_endpoint_registry(
        selection,
        by_id,
        endpoint_path,
        allowed_sound_classes_by_slot=speech_endpoint_classes,
        selection_path=selection_path,
    )
    colour_by_slot = {
        actor["source_slot_id"]: by_id[actor["asset_id"]]
        .get("realized_attributes", {}).get("top_color")
        for actor in selection["actors"]
    }
    actor_count = len(actor_assets)
    binding_frames = (
        (12,) if profile_id == "card16"
        else tuple(profile.get("binding_frames", [12, 40])))
    max_attempts = int(profile.get("max_attempts", 20000))
    try:
        plan = (
            _find_card16_plan(
                scene, params, seed=f"{seed}|{point_id}|main",
                max_attempts=max_attempts)
            if profile_id == "card16"
            else find_n_route_plan(
                scene, params, actor_count=actor_count,
                seed=f"{seed}|{point_id}|main",
                binding_frames=binding_frames,
                max_attempts=max_attempts))
    except SearchExhausted:
        raise
    except NRouteSearchExhausted as error:
        raise SearchExhausted(
            str(error),
            evaluated_combinations=error.evaluated_combinations) from error
    timeline_path, timeline, camera_ue = _author_timeline(
        point, "timeline", selection_path, registry_path, scene, plan, params)

    render = resolve_scene_render_context(scene)
    m1 = apply_camera_listener_pose_ue(
        base_request, request_id=f"qa_v3_{scene.scene_id}_{point_id}",
        position_m=render["world_transform"](camera_ue),
        ue_yaw_degrees=float(plan["camera_yaw_deg"]),
        horizontal_fov_deg=scene.hfov_deg)
    _write(point / "m1_capture_request.json", m1)

    segment2_plan = None
    segment2_timeline = None
    segment2_search_attempts = 0
    if profile_id == "card17":
        main_signature = _plan_signature(plan)
        for attempt in range(40):
            try:
                candidate = find_n_route_plan(
                    scene, params, actor_count=2,
                    seed=f"{seed}|{point_id}|segment2|{attempt}",
                    binding_frames=(12, 40),
                    max_attempts=max_attempts)
            except NRouteSearchExhausted as error:
                segment2_search_attempts += error.evaluated_combinations
                continue
            segment2_search_attempts += int(candidate["search_attempts"])
            try:
                target_band = _location_band(
                    profile, scene, candidate, 0)[0]
                other_band = _location_band(
                    profile, scene, candidate, 1)[0]
            except SearchExhausted as error:
                segment2_search_attempts += error.evaluated_combinations
                continue
            if (target_band != other_band
                    and _plan_signature(candidate) != main_signature):
                segment2_plan = candidate
                break
        if segment2_plan is None:
            raise SearchExhausted(
                "segment2 never differs from segment1 with distinct answer bands",
                evaluated_combinations=segment2_search_attempts)
        _, segment2_timeline, _ = _author_timeline(
            point, "timeline_segment2", selection_path, registry_path,
            scene, segment2_plan, params)

    if profile_id in {"card13", "card14"}:
        target_rng = np.random.default_rng(
            seed_uint64(f"{seed}|speech-target|{cell_index}"))
        option_rng = np.random.default_rng(
            seed_uint64(f"{seed}|speech-options|{cell_index}"))
        main_events, gatea_events, truth = _speech_program_events(
            speech_schedule,
            cell_index,
            target_rng=target_rng,
            option_rng=option_rng,
        )
        truth["speech_question"] = _speech_question_context(
            main_events, colour_by_slot, truth
        )
        sound_assets = []
    else:
        sound_assets = (
            inventory["sound_types"] if profile_id == "card12"
            else [{"sound_asset_id": "dog_beagle_v2_scheduled_dry"}])
        main_events, gatea_events, truth = _program_events(
            profile_id, cell_index, sound_assets)
    slot_endpoints = {
        actor["source_slot_id"]: endpoint["source_endpoint_id"]
        for actor, endpoint in zip(selection["actors"], endpoint_records)
    }
    # AudioProgram.mode is derived from which slots sound, not PROGRAM_MODE.
    request = {
        "pair_kind": profile_id,
        "point_id": point_id,
        "slot_endpoints": slot_endpoints,
        **program_request_fields(params, include_mode=False),
    }
    if speech_schedule is None:
        require_dry_canvas_source_mode(
            params, owner="design_qa_v3_extended_profile")
        request.update(
            dry_canvas_window_fields(params),
            sound_asset_id=sound_assets[0]["sound_asset_id"],
        )
    main_request = dict(request, mode=audio_program_mode(main_events))
    gatea_request = dict(request, mode=audio_program_mode(gatea_events))
    main_program = build_program(main_request, main_events, revision="v1")
    gatea_program = build_program(
        gatea_request, gatea_events, revision="gateA_v1")
    if speech_schedule is not None:
        validate_m6_audio_program(main_program)
        validate_m6_audio_program(gatea_program)
    _write(point / "audio_program.json", main_program)
    _write(point / "audio_program_gateA.json", gatea_program)

    facts = _facts(
        profile, inventory, truth, scene, plan,
        segment2_plan=segment2_plan,
        speech_events=main_events if speech_schedule is not None else None,
        colour_by_slot=colour_by_slot,
        speech_schedule=speech_schedule,
    )
    main_starts = [event["start_sample"] for event in main_program["events"]]
    gatea_starts = [event["start_sample"] for event in gatea_program["events"]]
    main_sounds = sorted(event["sound_asset_id"] for event in main_program["events"])
    gatea_sounds = sorted(event["sound_asset_id"] for event in gatea_program["events"])
    facts.update({
        "schema": "qa_v3_extended_fact_record_v1",
        "camera_height_m": float(plan.get("camera_height_m") or scene.camera_height_m),
        "camera_clearance": plan.get("camera_clearance"),
        "point_id": point_id,
        "profile_id": profile_id,
        "scene_id": scene.scene_id,
        "status": "research_candidate",
        "qualification_claim": False,
        "evidence_class": "geometry_candidate",
        "audio": {
            "main_program": "audio_program.json",
            "gatea_program": "audio_program_gateA.json",
        },
        "gatea_checks": {
            "event_count_preserved": (
                len(main_program["events"]) == len(gatea_program["events"])),
            "event_times_preserved": main_starts == gatea_starts,
            "sound_asset_multiset_preserved": main_sounds == gatea_sounds,
            "audio_assignment_changed": [
                (event["source_endpoint_id"], event["sound_asset_id"])
                for event in main_program["events"]
            ] != [
                (event["source_endpoint_id"], event["sound_asset_id"])
                for event in gatea_program["events"]
            ],
        },
        "search": {
            "attempts": (
                int(plan["search_attempts"])
                + int(segment2_search_attempts)),
            "line_of_sight_screened": bool(plan["line_of_sight_screened"]),
            "bank_attempt_budget": plan.get("bank_attempt_budget"),
            "route_synthesis": plan.get("route_synthesis"),
        },
        # which of this point's actor routes came from the bank and which the
        # solver designed, in actor order, with the full design record
        "motion": {
            "route_sources": list(plan.get("route_sources") or []),
            "route_provenance": list(plan.get("route_provenance") or []),
            "designed_route_count": plan.get("designed_route_count"),
            "segment2_route_sources": list(
                (segment2_plan or {}).get("route_sources") or []),
        },
    })
    facts["answer_forms"] = answer_forms_from_params(params)
    if speech_schedule is not None:
        facts["audio"]["schedule"] = dict(speech_schedule.declared)
        facts["audio"]["utterances"] = _speech_bindings(
            main_events, colour_by_slot)
        gatea_result = _facts(
            profile, inventory, truth, scene, plan,
            segment2_plan=segment2_plan,
            speech_events=gatea_events,
            colour_by_slot=colour_by_slot,
            speech_schedule=speech_schedule,
        )
        gatea_facts = copy.deepcopy(facts)
        for key in (
            "truth_status",
            "target_index",
            "target_slot",
            "target_speaker_id",
            "target_utterance_id",
            "question_target_index",
            "question_target_slot",
            "question_target_colour",
            "question_target_transcript",
            "question_option_order",
            "speech_bindings",
            "mcq",
            "open",
            "speech_schedule",
        ):
            if key in gatea_result:
                gatea_facts[key] = gatea_result[key]
        if speech_schedule is not None:
            facts["gatea_checks"].update({
                "question_stem_preserved": (
                    facts["mcq"]["stem"] == gatea_result["mcq"]["stem"]
                ),
                "question_options_preserved": (
                    facts["mcq"]["options_space"]
                    == gatea_result["mcq"]["options_space"]
                ),
                "question_gold_changed": (
                    facts["mcq"]["truth_option"]
                    != gatea_result["mcq"]["truth_option"]
                    or facts["open"]["truth_value"]
                    != gatea_result["open"]["truth_value"]
                ),
            })
        if not all(facts["gatea_checks"].values()):
            raise RuntimeError(
                f"Gate A structure check failed: {facts['gatea_checks']}"
            )
        gatea_facts.update({
            "variant": "gateA",
            "gatea_of": point_id,
            "audio": {
                "program": "audio_program_gateA.json",
                "schedule": dict(speech_schedule.declared),
                "utterances": _speech_bindings(
                    gatea_events, colour_by_slot),
            },
            "gatea_checks": dict(facts["gatea_checks"]),
        })
        gatea_facts["gatea"] = {
            "program_id": gatea_program["program_id"],
            "fact_record": "fact_record_gateA.json",
            "checks": dict(facts["gatea_checks"]),
        }
        _write(point / "fact_record_gateA.json", gatea_facts)

    gateb_assets = list(actor_assets)
    gateb_plan = copy.deepcopy(
        segment2_plan if profile_id == "card17" else plan)
    gateb_plan["routes"] = list(gateb_plan["routes"])
    gateb_search_attempts = 0
    reference_timeline = (
        segment2_timeline if profile_id == "card17" else timeline)
    if profile_id == "card15a":
        outside_route, gateb_search_attempts = _find_gateb_out_of_view_route(
            scene, params, plan, frame=30)
        gateb_plan["routes"][-1] = outside_route
        changed_visual_fact = "in_scene_visibility_count"
    elif profile_id == "card16":
        gateb_plan["routes"] = list(reversed(plan["routes"]))
        changed_visual_fact = "final_visibility_assignment"
    elif profile_id == "card17":
        gateb_assets = list(reversed(actor_assets))
        changed_visual_fact = "segment2_identity_binding"
    else:
        gateb_assets = list(reversed(actor_assets))
        changed_visual_fact = "appearance_to_position_binding"

    gateb_selection = _selection(gateb_assets, by_id, snapshot_content)
    gateb_selection_path = point / "actor_selection_gateB.json"
    _write(gateb_selection_path, gateb_selection)
    gateb_endpoint_path = point / "source_endpoints_gateB.json"
    build_endpoint_registry(
        gateb_selection,
        by_id,
        gateb_endpoint_path,
        allowed_sound_classes_by_slot=speech_endpoint_classes,
        selection_path=gateb_selection_path,
    )
    _, gateb_timeline, _ = _author_timeline(
        point, "timeline_gateB", gateb_selection_path, registry_path,
        scene, gateb_plan, params)
    gateb_visual_check = _assert_gateb_visual_change(
        selection, reference_timeline, gateb_selection, gateb_timeline)
    gateb = {
        "schema": "qa_v3_gateb_intervention_v1",
        "profile_id": profile_id,
        "kept_fixed": (
            ["camera", "event_times"] if profile_id == "card17"
            else ["camera", "audio_program_main", "event_times"]),
        "changed_visual_fact": changed_visual_fact,
        "actor_selection": "actor_selection_gateB.json",
        "timeline": "timeline_gateB.json",
        "source_endpoint_registry": "source_endpoints_gateB.json",
        "audio_program": (None if profile_id == "card17"
                          else "audio_program.json"),
        "visual_change_check": gateb_visual_check,
        "qualification_claim": False,
    }
    _write(point / "gateB_intervention.json", gateb)
    facts.update(_runtime_descriptions(profile, point))
    _write(point / "fact_record.json", facts)
    total_search_attempts = (
        int(plan["search_attempts"])
        + int(segment2_search_attempts)
        + int(gateb_search_attempts))
    return {
        "point_id": point_id,
        "camera_height_m": float(plan.get("camera_height_m") or scene.camera_height_m),
        "camera_clearance": plan.get("camera_clearance"),
        "search_attempts": total_search_attempts,
        "route_sources": list(plan.get("route_sources") or []),
        "designed_route_count": plan.get("designed_route_count"),
        "segment2_route_sources": list(
            (segment2_plan or {}).get("route_sources") or []),
        "artifacts": {
            "selection": str(selection_path),
            "timeline": str(timeline_path),
            "endpoints": str(endpoint_path),
            "fact": str(point / "fact_record.json"),
            "gateB": str(point / "gateB_intervention.json"),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--snapshot-content", required=True)
    args = parser.parse_args(argv)
    # Repository tmp may be a declared symlink to external output storage.
    args.out_root = args.out_root.resolve()
    if args.out_root.exists():
        print(f"refusing to overwrite: {args.out_root}", file=sys.stderr)
        return 2
    profiles = _read(args.profiles)
    if not isinstance(profiles, list) or len(profiles) != 1:
        raise ValueError("extended runner requires exactly one profile")
    profile = profiles[0]
    profile_id = profile.get("id")
    if profile_id not in SUPPORTED:
        raise ValueError(f"unsupported extended profile: {profile_id!r}")
    from qa_v3_request import read_qa_params
    params = read_qa_params(args.params)
    if profile_id in {"card13", "card14"}:
        if str(params.get("SOUND_SOURCE_MODE")) != "event_pool":
            raise ValueError(
                "card13/card14 require SOUND_SOURCE_MODE=event_pool")
        speech_pool = _speech_pool_rows(params)
        if speech_pool is None:
            raise ValueError(
                "card13/card14 require SOUND_EVENT_POOL with speech clips")
    else:
        speech_pool = None
        require_dry_canvas_source_mode(
            params, owner="design_qa_v3_extended_profile")
    scene_config = SS.read_scene_config(args.scene_config)
    clock = SS.validate_frame_clock(params, require_clip_seconds=True)
    scene = load_scene(scene_config, frame_count=clock["frame_count"],
                       frame_rate_hz=clock["frame_rate_hz"])
    resolve_scene_render_context(scene)
    require_camera_clearance(scene, params)
    registry_path = REPO / "examples/runtime/source_asset_runtime_profiles.json"
    registry = _read(registry_path)
    by_id = {item["asset_id"]: item for item in registry["assets"]}
    sound_registry = _read(REPO / "examples/registry/registries/sound_assets_v1.json")
    inventory = _resource_inventory(
        profile_id,
        registry["assets"],
        sound_registry["sound_assets"],
        speech_pool=speech_pool,
        profile=profile,
    )
    if inventory["missing"]:
        _write_unavailable(
            args.out_root, profile, scene, inventory["missing"], args.cells)
        print(json.dumps({
            "out": str(args.out_root), "scene": scene.scene_id,
            "resource_unavailable": inventory["missing"],
            "cells_requested": args.cells}))
        return 0

    args.out_root.mkdir(parents=True)
    base_request = _read(Path(scene_config["camera_base_request"]))
    records = []
    rejected = []
    reasons = Counter()
    attempts = 0
    try:
        for cell_index in range(args.cells):
            try:
                record = _realise_cell(
                    args.out_root, profile, cell_index, scene, params,
                    inventory, by_id, registry_path, base_request,
                    args.snapshot_content, args.seed)
                records.append(record)
                attempts += record["search_attempts"]
            except AudioProfileSearchExhausted as error:
                reason = "speech_selection_budget_exhausted"
                reasons[reason] += 1
                rejected.append({
                    "point_id": f"{profile_id}_{cell_index + 1:03d}",
                    "reason": reason,
                    "detail": str(error),
                    "selection_attempts": error.attempts,
                })
            except SearchExhausted as error:
                reason = "not_found_within_budget"
                reasons[reason] += 1
                attempts += error.evaluated_combinations
                rejected.append({
                    "point_id": f"{profile_id}_{cell_index + 1:03d}",
                    "reason": reason,
                    "detail": f"{type(error).__name__}: {error}"[:300],
                    "evaluated_combinations": error.evaluated_combinations,
                })
    except Exception as error:  # noqa: BLE001 - recorded, then re-raised
        _write_failed(args.out_root, profile_id, scene.scene_id, error,
                      cells_requested=args.cells, completed=len(records))
        raise
    manifest = {
        "schema": "qa_v3_extended_profile_batch_manifest_v1",
        "status": "research_dev",
        "qualification_claim": False,
        "evidence_class": "geometry_candidate",
        "code": git_worktree_state(),
        "scene_id": scene.scene_id,
        "profile_ids": [profile_id],
        "scene": {
            "camera_clearance_screened": scene.camera_clearance_screened,
            "camera_clearance_table": scene.provenance.get("camera_clearance_table"),
            "camera_height_fallback_used": sum(
                1 for record in records
                if (record.get("camera_clearance") or {}).get("fallback_used")),
            "camera_heights_m": sorted({record.get("camera_height_m")
                                        for record in records
                                        if record.get("camera_height_m") is not None}),
            "walkable_grid": scene.provenance.get("walkable_grid"),
            "floor_reference": scene.provenance.get("floor_reference"),
            # N 角色搜索先抽库、库不够再设计(owner 2026-09-03 要求接上)。
            "route_synthesis": dict(
                SS.route_synthesis_report(scene, params), applied=True,
                realised=_route_source_counts(records),
                note=("n-actor search fills from the bank first and designs the "
                      "remaining actors after the bank budget; designed routes go "
                      "through the same per-actor checks")),
        },
        "counts": {
            "cells_requested": args.cells,
            "geometry_candidates": len(records),
            "rejected": len(rejected),
        },
        "search": {
            "combinations_evaluated": attempts,
            "budget_exhausted": (reasons["not_found_within_budget"]
                                 + reasons["speech_selection_budget_exhausted"]),
            "by_reason": dict(reasons),
        },
        "records": records,
        "rejected": rejected,
        "boundary": (
            "Geometry/timeline/AudioProgram/Gate-A/Gate-B research candidates. "
            "Pixel-dependent cards require the native pixel join; this is not "
            "question admission or missing-modality certification."),
    }
    request_result = write_requested_questions(
        args.out_root, (record["artifacts"]["fact"] for record in records), params,
    )
    manifest["question_request"] = request_result
    manifest["counts"]["designed_questions"] = request_result["designed_question_count"]
    manifest["counts"]["counterfactual_questions"] = request_result["counterfactual_question_count"]
    _write(args.out_root / "batch_manifest.json", manifest)
    print(json.dumps({
        "out": str(args.out_root), "scene": scene.scene_id,
        "geometry_candidates": len(records),
        "cells_requested": args.cells, "rejected": len(rejected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

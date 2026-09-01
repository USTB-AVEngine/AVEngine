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
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

from build_qa_v3_n_actor_canary import (  # noqa: E402
    DEFAULT_ASSETS,
    _read,
    _write,
    build_endpoint_registry,
    find_n_route_plan,
)
from build_qa_v3_programs import build_program  # noqa: E402
from design_qa_v3_pilot_batch import _actor_entry  # noqa: E402
from design_qa_v3_scene_batch import resolve_scene_render_context  # noqa: E402
from make_idle_then_walk_timeline import transform_to_solved_routes  # noqa: E402
from scene_sampler import (  # noqa: E402
    effective_half_fov,
    load_scene,
    relative_azimuth_deg,
)
from avengine.camera_pose import apply_camera_listener_pose_ue  # noqa: E402
from avengine.timeline.current_apartment_visual import (  # noqa: E402
    author_current_n_actor_visual_timeline,
)


SUPPORTED = {"card11", "card12", "card13", "card14", "card15a", "card16", "card17"}
EVENT_STARTS = (8000, 24000, 40000, 56000)
VISIBILITY_OPTIONS = (
    "visible_clear",
    "visible_occluded",
    "fully_occluded",
    "out_of_view",
)


class SearchExhausted(RuntimeError):
    """Finite randomized search ended without a candidate."""


def _resource_inventory(profile_id, assets, sounds):
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
    speech_by_transcript = {}
    for item in sounds:
        transcript = item.get("transcript")
        if (item.get("semantic_sound_class") == "human_speech"
                and isinstance(transcript, str) and transcript.strip()):
            speech_by_transcript.setdefault(transcript.strip(), item)
    speech = list(speech_by_transcript.values())
    missing = []
    if profile_id in {"card11", "card15a"} and len(dogs) < 4:
        missing.append("four_registered_dog_assets")
    if profile_id in {"card16", "card17"} and len(dogs) < 2:
        missing.append("two_registered_dog_assets")
    if profile_id == "card12" and len(sound_types) < 4:
        missing.append("four_registered_semantic_sound_types")
    if profile_id in {"card13", "card14"}:
        if len(humans) < 4:
            missing.append("four_controlled_human_top_colours")
        if len(speech) < 4:
            missing.append("four_transcribed_speech_assets")
    return {
        "dogs": dogs,
        "humans": humans,
        "sound_types": sound_types,
        "speech": speech,
        "missing": missing,
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


def _author_timeline(out_dir, name, selection_path, registry_path, scene, plan):
    render = resolve_scene_render_context(scene)
    ground = float(render["ground_z_ue_cm"])
    routes_3d = {
        f"source{index}": [
            [float(x), float(y), ground] for x, y in route.samples_xy
        ]
        for index, route in enumerate(plan["routes"], start=1)
    }
    camera_ue = [
        float(plan["camera_xy"][0]),
        float(plan["camera_xy"][1]),
        ground + scene.camera_height_m * 100.0,
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


def _program_events(profile_id, cell_index, sound_assets):
    sounds = [item["sound_asset_id"] for item in sound_assets]
    bark = "dog_beagle_v2_scheduled_dry"
    slots = [f"source{index}" for index in range(1, 5)]
    if profile_id == "card11":
        positive = cell_index % 2 == 0
        main_slot = "source1" if positive else "source4"
        other_slot = "source4" if positive else "source1"
        return (
            [(main_slot, EVENT_STARTS[0], bark)],
            [(other_slot, EVENT_STARTS[0], bark)],
            {"desired_answer": "source1" if positive else "none",
             "positive_visible_source": positive},
        )
    if profile_id == "card12":
        main = [
            (slot, start, sound)
            for slot, start, sound in zip(slots, EVENT_STARTS, sounds[:4])
        ]
        gatea = list(main)
        gatea[0] = (gatea[0][0], gatea[0][1], main[1][2])
        gatea[1] = (gatea[1][0], gatea[1][1], main[0][2])
        return main, gatea, {"target_sound_asset_id": main[0][2]}
    if profile_id in {"card13", "card14"}:
        main = [
            (slot, start, sound)
            for slot, start, sound in zip(slots, EVENT_STARTS, sounds[:4])
        ]
        gatea = list(main)
        gatea[0], gatea[1] = (
            (main[0][0], main[0][1], main[1][2]),
            (main[1][0], main[1][1], main[0][2]),
        )
        return main, gatea, {"target_speech_asset_id": main[0][2]}
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


def _find_card16_plan(scene, params, *, seed, max_attempts):
    half_fov = effective_half_fov(scene, params)
    for outer in range(200):
        try:
            plan = find_n_route_plan(
                scene, params, actor_count=2,
                seed=f"{seed}|final-state-split|{outer}",
                binding_frames=(12,), max_attempts=max_attempts)
        except RuntimeError:
            continue
        final_inside = [
            abs(relative_azimuth_deg(
                plan["camera_xy"], float(plan["camera_yaw_deg"]),
                route.at(74))) <= half_fov
            for route in plan["routes"]
        ]
        if sum(final_inside) == 1:
            plan["search_attempts"] += outer
            plan["card16_final_fov_membership"] = final_inside
            return plan
    raise SearchExhausted(
        "no card16 plan with distinct final in-view/out-of-view states")


def _location_band(scene, plan, route_index, frame=40):
    azimuth = relative_azimuth_deg(
        plan["camera_xy"], float(plan["camera_yaw_deg"]),
        plan["routes"][route_index].at(frame))
    band = "left" if azimuth < -17.5 else "right" if azimuth >= 17.5 else "center"
    return band, float(azimuth)


def _facts(profile_id, inventory, truth, scene, main_plan, segment2_plan=None):
    if profile_id == "card11":
        return {
            "truth_status": "pending_native_pixel_join",
            "desired_truth": truth["desired_answer"],
            "mcq": {
                "stem": "Which visible dog made the sound?",
                "options_space": ["source1", "source2", "source3", "none"],
                "truth_option": truth["desired_answer"],
            },
            "open": {
                "stem": "Which visible dog, if any, made the sound?",
                "truth_value": truth["desired_answer"],
                "scoring": "closed_set",
            },
            "pixel_acceptance": {
                "source1_source2_source3": "visible",
                "source4": "fully_occluded_or_out_of_view",
            },
        }
    if profile_id == "card12":
        sound = inventory["sound_types"][0]
        label = sound["taxonomy_path"][-1]
        return {
            "truth_status": "engine_exact",
            "mcq": {"stem": "What sound did source1 make?",
                    "options_space": [
                        item["taxonomy_path"][-1]
                        for item in inventory["sound_types"][:4]],
                    "truth_option": label},
            "open": {"stem": "What sound did source1 make?",
                     "truth_value": label, "scoring": "closed_set"},
        }
    if profile_id in {"card13", "card14"}:
        speech = inventory["speech"][0]
        transcript = speech["transcript"].strip()
        colour = inventory["humans"][0]["realized_attributes"]["top_color"]
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
    band, azimuth = _location_band(scene, segment2_plan, 0)
    other_band, other_azimuth = _location_band(scene, segment2_plan, 1)
    return {
        "truth_status": "engine_exact_future_extension",
        "selector": {"segment1_first_caller_slot": "source1"},
        "mcq": {
            "stem": "In segment 2, where is the dog that barked first in segment 1?",
            "options_space": ["left", "center", "right"],
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


def _write_unavailable(out_root, profile, scene, missing, cells):
    out_root.mkdir(parents=True)
    manifest = {
        "schema": "qa_v3_extended_profile_batch_manifest_v1",
        "status": "research_dev",
        "qualification_claim": False,
        "evidence_class": "resource_unavailable",
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


def _realise_cell(out_root, profile, cell_index, scene, params, inventory,
                  by_id, registry_path, base_request, snapshot_content, seed):
    profile_id = profile["id"]
    actor_assets = (
        [item["asset_id"] for item in inventory["humans"][:4]]
        if profile_id in {"card13", "card14"}
        else [item["asset_id"] for item in inventory["dogs"][
            :2 if profile_id in {"card16", "card17"} else 4]]
    )
    point_id = f"{profile_id}_{cell_index + 1:03d}"
    point = out_root / point_id
    point.mkdir()
    selection = _selection(actor_assets, by_id, snapshot_content)
    selection_path = point / "actor_selection.json"
    _write(selection_path, selection)
    endpoint_path = point / "source_endpoints.json"
    _, endpoint_records = build_endpoint_registry(selection, by_id, endpoint_path)
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
    except RuntimeError as error:
        if isinstance(error, SearchExhausted):
            raise
        raise SearchExhausted(str(error)) from error
    timeline_path, timeline, camera_ue = _author_timeline(
        point, "timeline", selection_path, registry_path, scene, plan)

    render = resolve_scene_render_context(scene)
    m1 = apply_camera_listener_pose_ue(
        base_request, request_id=f"qa_v3_{scene.scene_id}_{point_id}",
        position_m=render["world_transform"](camera_ue),
        ue_yaw_degrees=float(plan["camera_yaw_deg"]),
        horizontal_fov_deg=scene.hfov_deg)
    _write(point / "m1_capture_request.json", m1)

    segment2_plan = None
    if profile_id == "card17":
        for attempt in range(40):
            try:
                candidate = find_n_route_plan(
                    scene, params, actor_count=2,
                    seed=f"{seed}|{point_id}|segment2|{attempt}",
                    binding_frames=(12, 40),
                    max_attempts=max_attempts)
            except RuntimeError:
                continue
            if _location_band(scene, candidate, 0)[0] != _location_band(
                    scene, candidate, 1)[0]:
                segment2_plan = candidate
                break
        if segment2_plan is None:
            raise SearchExhausted(
                "segment2 actors never occupy distinct answer bands")
        _author_timeline(
            point, "timeline_segment2", selection_path, registry_path,
            scene, segment2_plan)

    sound_assets = (
        inventory["sound_types"] if profile_id == "card12"
        else inventory["speech"] if profile_id in {"card13", "card14"}
        else [{"sound_asset_id": "dog_beagle_v2_scheduled_dry"}])
    main_events, gatea_events, truth = _program_events(
        profile_id, cell_index, sound_assets)
    slot_endpoints = {
        actor["source_slot_id"]: endpoint["source_endpoint_id"]
        for actor, endpoint in zip(selection["actors"], endpoint_records)
    }
    request = {
        "pair_kind": profile_id,
        "point_id": point_id,
        "slot_endpoints": slot_endpoints,
        "sound_asset_id": sound_assets[0]["sound_asset_id"],
        "mode": ("one_active_of_n" if profile_id == "card11"
                 else "sequential_sources"),
    }
    def program_mode(events):
        active = {event[0] for event in events}
        return "one_active_of_n" if len(active) == 1 else "sequential_sources"

    main_request = dict(request, mode=program_mode(main_events))
    gatea_request = dict(request, mode=program_mode(gatea_events))
    main_program = build_program(main_request, main_events)
    gatea_program = build_program(
        gatea_request, gatea_events, revision="gateA_v1")
    _write(point / "audio_program.json", main_program)
    _write(point / "audio_program_gateA.json", gatea_program)

    facts = _facts(
        profile_id, inventory, truth, scene, plan,
        segment2_plan=segment2_plan)
    main_starts = [event["start_sample"] for event in main_program["events"]]
    gatea_starts = [event["start_sample"] for event in gatea_program["events"]]
    main_sounds = sorted(event["sound_asset_id"] for event in main_program["events"])
    gatea_sounds = sorted(event["sound_asset_id"] for event in gatea_program["events"])
    facts.update({
        "schema": "qa_v3_extended_fact_record_v1",
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
            "attempts": int(plan["search_attempts"]),
            "line_of_sight_screened": bool(plan["line_of_sight_screened"]),
        },
    })
    if not all(facts["gatea_checks"].values()):
        raise RuntimeError(f"Gate A structure check failed: {facts['gatea_checks']}")
    _write(point / "fact_record.json", facts)

    gateb_assets = list(reversed(actor_assets))
    if profile_id == "card15a":
        gateb_assets = actor_assets[:-1]
    gateb_selection = _selection(gateb_assets, by_id, snapshot_content)
    gateb_selection_path = point / "actor_selection_gateB.json"
    _write(gateb_selection_path, gateb_selection)
    gateb_plan = copy.deepcopy(
        segment2_plan if profile_id == "card17" else plan)
    gateb_plan["routes"] = list(reversed(plan["routes"]))[:len(gateb_assets)]
    _author_timeline(
        point, "timeline_gateB", gateb_selection_path, registry_path,
        scene, gateb_plan)
    gateb = {
        "schema": "qa_v3_gateb_intervention_v1",
        "profile_id": profile_id,
        "kept_fixed": ["camera", "audio_program_main", "event_times"],
        "changed_visual_fact": (
            "in_scene_actor_count" if profile_id == "card15a"
            else "final_visibility_assignment" if profile_id == "card16"
            else "segment2_identity_binding" if profile_id == "card17"
            else "appearance_to_slot_binding"),
        "actor_selection": "actor_selection_gateB.json",
        "timeline": "timeline_gateB.json",
        "qualification_claim": False,
    }
    _write(point / "gateB_intervention.json", gateb)
    return {
        "point_id": point_id,
        "search_attempts": int(plan["search_attempts"]),
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
    scene = load_scene(_read(args.scene_config))
    resolve_scene_render_context(scene)
    params = _read(args.params)
    registry_path = REPO / "examples/runtime/source_asset_runtime_profiles.json"
    registry = _read(registry_path)
    by_id = {item["asset_id"]: item for item in registry["assets"]}
    sound_registry = _read(REPO / "examples/registry/registries/sound_assets_v1.json")
    inventory = _resource_inventory(
        profile_id, registry["assets"], sound_registry["sound_assets"])
    if inventory["missing"]:
        _write_unavailable(
            args.out_root, profile, scene, inventory["missing"], args.cells)
        print(json.dumps({
            "out": str(args.out_root), "scene": scene.scene_id,
            "resource_unavailable": inventory["missing"],
            "cells_requested": args.cells}))
        return 0

    args.out_root.mkdir(parents=True)
    base_request = _read(Path(_read(args.scene_config)["camera_base_request"]))
    records = []
    rejected = []
    reasons = Counter()
    attempts = 0
    for cell_index in range(args.cells):
        try:
            record = _realise_cell(
                args.out_root, profile, cell_index, scene, params, inventory,
                by_id, registry_path, base_request, args.snapshot_content,
                args.seed)
            records.append(record)
            attempts += record["search_attempts"]
        except Exception as error:
            reason = "not_found_within_budget" if isinstance(
                error, SearchExhausted) else "pipeline_constraint_failed"
            reasons[reason] += 1
            rejected.append({
                "point_id": f"{profile_id}_{cell_index + 1:03d}",
                "reason": reason,
                "detail": f"{type(error).__name__}: {error}"[:300],
            })
    manifest = {
        "schema": "qa_v3_extended_profile_batch_manifest_v1",
        "status": "research_dev",
        "qualification_claim": False,
        "evidence_class": "geometry_candidate",
        "scene_id": scene.scene_id,
        "profile_ids": [profile_id],
        "counts": {
            "cells_requested": args.cells,
            "geometry_candidates": len(records),
            "rejected": len(rejected),
        },
        "search": {
            "combinations_evaluated": attempts,
            "budget_exhausted": reasons["not_found_within_budget"],
            "by_reason": dict(reasons),
        },
        "records": records,
        "rejected": rejected,
        "boundary": (
            "Geometry/timeline/AudioProgram/Gate-A/Gate-B research candidates. "
            "Pixel-dependent cards require the native pixel join; this is not "
            "question admission or missing-modality certification."),
    }
    _write(args.out_root / "batch_manifest.json", manifest)
    print(json.dumps({
        "out": str(args.out_root), "scene": scene.scene_id,
        "geometry_candidates": len(records),
        "cells_requested": args.cells, "rejected": len(rejected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

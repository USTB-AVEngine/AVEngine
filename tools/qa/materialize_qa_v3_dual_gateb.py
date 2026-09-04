#!/usr/bin/env python3
"""Materialize renderable Gate-B twins for selected dual-source QA-v3 points."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

from avengine.contracts.json_io import canonical_json_sha256
from avengine.registry import (
    load_sound_asset_registry,
    load_source_endpoint_registry,
)
from avengine.timeline.audio_program import validate_audio_program
from avengine.timeline.current_apartment_visual import (
    _load_timeline,
    _selection_bindings,
)
from derive_twin_programs import resolve_slot_endpoints
from design_qa_v3_extended_profile import (
    _assert_gateb_visual_change,
    _selection,
)


ASSET_SWAP = {"card4R", "card7", "card8", "card9", "card15b"}
ROUTE_SWAP = {
    "card1F", "card1B", "card2", "card3", "card5", "card5R",
    "card6", "card6R", "card10",
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def _swap_dynamic_states(timeline):
    result = copy.deepcopy(timeline)
    dynamic_fields = (
        "translation_ue_cm", "yaw_ue_deg", "action_id", "action_phase",
        "route_geometry", "route_waypoint_count", "route_arc_length_ue_cm",
        "route_segment_index",
    )
    for frame in result["frames"]:
        by_slot = {
            state["source_slot_id"]: state
            for state in frame["actor_states"]
        }
        left = copy.deepcopy(by_slot["source1"])
        right = copy.deepcopy(by_slot["source2"])
        for field in dynamic_fields:
            if field in right:
                by_slot["source1"][field] = copy.deepcopy(right[field])
            else:
                by_slot["source1"].pop(field, None)
            if field in left:
                by_slot["source2"][field] = copy.deepcopy(left[field])
            else:
                by_slot["source2"].pop(field, None)
    return result


def _rebind_identity(timeline, selection_path, registry_path):
    _, bindings, authorization = _selection_bindings(
        actor_selection_path=selection_path,
        source_asset_registry_path=registry_path)
    result = copy.deepcopy(timeline)
    result["actor_selection"] = str(selection_path.resolve())
    result["asset_authorization"] = authorization
    summaries = {
        item["source_slot_id"]: item for item in result["actors"]
    }
    summary_fields = (
        "actor_id", "asset_id", "revision", "walk_phase_period_frames",
        "ue_anatomical_forward_yaw_deg", "blueprint_class_path",
        "idle_animation", "walking_animation", "graph_mesh_package",
        "graph_mesh_object_path",
    )
    for slot, binding in bindings.items():
        summary = summaries[slot]
        for field in summary_fields:
            summary[field] = copy.deepcopy(binding[field])
    state_fields = (
        "actor_id", "asset_id", "revision", "walk_phase_period_frames",
    )
    for frame in result["frames"]:
        for state in frame["actor_states"]:
            binding = bindings[state["source_slot_id"]]
            for field in state_fields:
                state[field] = copy.deepcopy(binding[field])
    _load_timeline(
        timeline_path=_temporary_timeline(result, selection_path.parent),
        bindings=bindings,
        asset_authorization=authorization)
    return result


def _temporary_timeline(value, root):
    path = root / ".gateb_timeline_validation.json"
    if path.exists():
        raise RuntimeError(f"unexpected temporary validation path: {path}")
    _write(path, value)
    return path


def _remove_temporary(root):
    path = root / ".gateb_timeline_validation.json"
    if path.exists():
        path.unlink()


def _gateb_program(main, main_selection, gateb_selection, assets, endpoints):
    old_candidates = main["candidate_source_endpoint_ids"]
    old_to_slot = {
        old_candidates[0]: "source1",
        old_candidates[1]: "source2",
    }
    slot_endpoints = resolve_slot_endpoints(
        gateb_selection, assets, endpoints)
    result = {
        key: copy.deepcopy(value)
        for key, value in main.items()
        if key != "program_content_sha256"
    }
    result["program_id"] = f"{main['program_id']}_gateB"
    result["revision"] = "gateB_v1"
    result["candidate_source_endpoint_ids"] = [
        slot_endpoints["source1"], slot_endpoints["source2"]]
    result["events"] = [
        dict(
            event,
            source_endpoint_id=slot_endpoints[
                old_to_slot[event["source_endpoint_id"]]])
        for event in main["events"]
    ]
    result["program_content_sha256"] = canonical_json_sha256(result)
    return result


def materialize(candidate, output, *, registry_path, endpoint_registry,
                endpoint_registry_path, asset_registry, sound_registry,
                snapshot_content):
    profile_id = candidate["pilot_id"].split("__")[1]
    if profile_id not in ASSET_SWAP | ROUTE_SWAP:
        raise RuntimeError(f"unsupported dual Gate-B profile: {profile_id}")
    source_point = Path(candidate["source_point"]).resolve()
    selection = _read(candidate["artifacts"]["actor_selection"])
    timeline = _read(candidate["artifacts"]["timeline"])
    actor_assets = [actor["asset_id"] for actor in selection["actors"]]
    by_id = {item["asset_id"]: item for item in asset_registry["assets"]}

    if profile_id in ASSET_SWAP:
        gateb_selection = _selection(
            list(reversed(actor_assets)), by_id, snapshot_content)
        gateb_timeline = copy.deepcopy(timeline)
        intervention = "appearance_to_position_binding"
    else:
        gateb_selection = copy.deepcopy(selection)
        gateb_timeline = _swap_dynamic_states(timeline)
        intervention = "route_to_identity_binding"

    selection_path = output / "actor_selection_gateB.json"
    _write(selection_path, gateb_selection)
    try:
        gateb_timeline = _rebind_identity(
            gateb_timeline, selection_path, registry_path)
    finally:
        _remove_temporary(output)
    timeline_path = output / "timeline_gateB.json"
    _write(timeline_path, gateb_timeline)
    _, bindings, authorization = _selection_bindings(
        actor_selection_path=selection_path,
        source_asset_registry_path=registry_path)
    _load_timeline(
        timeline_path=timeline_path, bindings=bindings,
        asset_authorization=authorization)
    visual_check = _assert_gateb_visual_change(
        selection, timeline, gateb_selection, gateb_timeline)

    main_program = _read(candidate["artifacts"]["main_program"])
    program = _gateb_program(
        main_program, selection, gateb_selection,
        asset_registry, endpoint_registry)
    program_errors = validate_audio_program(
        program,
        source_endpoint_registry=endpoint_registry,
        sound_asset_registry=sound_registry)
    if program_errors:
        raise RuntimeError(
            f"{candidate['pilot_id']} Gate-B AudioProgram invalid: "
            f"{program_errors}")
    program_path = output / "audio_program_gateB.json"
    _write(program_path, program)
    manifest = {
        "schema": "qa_v3_dual_gateb_intervention_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "pilot_id": candidate["pilot_id"],
        "profile_id": profile_id,
        "source_point": str(source_point),
        "intervention": intervention,
        "expected_gold_relation": (
            "preserve" if profile_id == "card15b" else "flip"),
        "kept_fixed": ["camera", "event_times", "sound_asset_multiset"],
        "visual_change_check": visual_check,
        "artifacts": {
            "actor_selection": str(selection_path.resolve()),
            "timeline": str(timeline_path.resolve()),
            "audio_program": str(program_path.resolve()),
            "endpoint_registry": str(endpoint_registry_path.resolve()),
        },
        "boundary": (
            "CPU-materialized Gate-B research twin; not pixel admission, "
            "audio rendering, or modality certification."),
    }
    _write(output / "gateB_intervention.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", required=True, type=Path)
    parser.add_argument("--source-asset-registry", required=True, type=Path)
    parser.add_argument("--source-endpoint-registry", required=True, type=Path)
    parser.add_argument("--sound-asset-registry", required=True, type=Path)
    parser.add_argument("--snapshot-content", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output_root.exists() or args.output_root.is_symlink():
        print(f"refusing to overwrite: {args.output_root}", file=sys.stderr)
        return 2

    pilot = _read(args.pilot_manifest)
    assets = _read(args.source_asset_registry)
    endpoints = load_source_endpoint_registry(args.source_endpoint_registry)
    sounds = load_sound_asset_registry(args.sound_asset_registry)
    candidates = []
    for room in pilot["rooms"].values():
        for profile in room["profiles"].values():
            for candidate in profile.get("candidates", []):
                if candidate.get("gateb_status") != "materialized":
                    candidates.append(candidate)

    args.output_root.mkdir(parents=True)
    records = []
    for candidate in candidates:
        safe = candidate["pilot_id"].replace("/", "_")
        output = args.output_root / safe
        output.mkdir()
        records.append(materialize(
            candidate, output,
            registry_path=args.source_asset_registry,
            endpoint_registry=endpoints,
            endpoint_registry_path=args.source_endpoint_registry,
            asset_registry=assets,
            sound_registry=sounds,
            snapshot_content=args.snapshot_content))
    summary = {
        "schema": "qa_v3_dual_gateb_batch_manifest_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "pilot_manifest": str(args.pilot_manifest.resolve()),
        "candidate_count": len(records),
        "by_profile": dict(sorted(Counter(
            record["profile_id"] for record in records).items())),
        "records": records,
    }
    _write(args.output_root / "dual_gateb_manifest.json", summary)
    print(json.dumps({
        "output": str(args.output_root),
        "candidate_count": len(records),
        "by_profile": summary["by_profile"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

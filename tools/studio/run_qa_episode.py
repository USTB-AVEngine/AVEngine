#!/usr/bin/env python3
"""Match a QA request to existing room resources, plan it and execute its Episode."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import platform
import time

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

import numpy as np

from avengine.rooms.qa_episode import (
    QAPlanningError, build_qa_episode_plan, read_json, write_json,
)
from avengine.runtime_profiles import load_source_asset_runtime_registry
from avengine.rooms.room_package import package_from_catalog_entry, renderer_for_room


def plan_request(request: dict, output: Path) -> dict:
    """Select from a resource pool; request authors supply no room coordinates."""
    started = time.monotonic()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    write_json(output / "request.json", request)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPOSITORY,
                           capture_output=True, text=True, check=True).stdout.splitlines()
    write_json(output / "producer_version.json", {
        "repository": str(REPOSITORY), "git_commit": commit,
        "working_tree_changes_at_launch": dirty,
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "runtime_inputs": request.get("runtime", {}),
        "code_state": "working_tree" if dirty else "committed",
    })
    registry = load_source_asset_runtime_registry(
        request.get("source_registry", REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"))
    rooms = read_json(request["room_catalog"])
    rooms = rooms.get("rooms", rooms) if isinstance(rooms, dict) else rooms
    sounds = read_json(request["sound_pool"])
    sounds = sounds.get("sounds", sounds) if isinstance(sounds, dict) else sounds
    if not isinstance(rooms, list) or not rooms or not isinstance(sounds, list) or not sounds:
        raise QAPlanningError("nonempty room catalog and sound pool are required")
    rng = np.random.default_rng(int(request.get("seed", 0)))
    order = list(range(len(rooms)))
    rng.shuffle(order)
    attempts = []
    for index in order:
        room = rooms[index]
        if request.get("room_id") and request["room_id"] != room["room_id"]:
            continue
        try:
            package = package_from_catalog_entry(room, runtime=request.get("runtime"))
            renderer = renderer_for_room(package)
            if "room_package" in room or room.get("schema") == package["schema"]:
                room = {**package.get("legacy_catalog_entry", package.get("planning_inputs", {})),
                        "room_id": package["room_id"], "room_package": package}
            write_json(output / "renderer_dispatch.json", {
                "room_id": room["room_id"], "renderer": renderer,
                "capture_entrypoint": str(renderer_capture_entrypoint(renderer)),
                "package_validation_errors": package.get("validation_errors", []),
                "status": "dispatched", "native_execution": "not_run",
            })
            if renderer == "ue_spear" and room.get("native_room_adapter") == "avengine_native_spear_apartment_qa_room_v1":
                from avengine.rooms.native_qa_room import (
                    build_native_apartment_qa_plan, discover_native_apartment_resources,
                )
                resources = discover_native_apartment_resources(
                    repository=REPOSITORY, source_root=room["native_input_root"],
                    route_bank=room["route_bank"], room_profile_path=room.get("native_room_profile"))
                plan, layout, pf, _routes = build_native_apartment_qa_plan(
                    resources=resources, source_registry=registry, sounds=sounds,
                    episode_id=request["episode_id"], source_asset_ids=request["source_asset_ids"],
                    qa_ids=request.get("qa_ids"), seed=int(request.get("seed", 0)),
                    frame_count=int(request.get("frame_count", 240)),
                    frame_rate_hz=int(request.get("frame_rate_hz", 15)),
                    sample_rate_hz=int(request.get("sample_rate_hz", 16000)),
                    camera_motion=request.get("camera_motion", "follow_group"),
                    audio_mode=request.get("audio_mode", "sequential"),
                    start_hold_frames=int(request.get("start_hold_frames", 0)))
                plan["resources"].update(room)
                plan["resources"]["expected_stage_actor_count"] = 0
                plan["renderer_backend"] = "spear_unreal_native"
                plan["request"] = deepcopy(request)
            elif renderer == "habitat":
                plan, layout, pf = build_qa_episode_plan(
                    room={**room, "room_package": package, "backend": "habitat"},
                    request=request, source_registry=registry, sounds=sounds)
            else:
                plan, layout, pf = build_qa_episode_plan(
                    room=room, request=request, source_registry=registry, sounds=sounds)
        except (QAPlanningError, ValueError, OSError) as exc:
            attempts.append({"room_id": room.get("room_id"), "status": "not_selected",
                             "reason": f"{type(exc).__name__}: {exc}"})
            write_json(output / "room_selection.json", {"attempts": attempts})
            continue
        attempts.append({"room_id": room["room_id"], "status": "selected",
                         "condition_match": plan["question_condition_match"]})
        plan_root = output / "plan"
        plan_root.mkdir()
        write_json(plan_root / "episode_plan.json", plan)
        write_json(plan_root / "room_package.json", package)
        write_json(plan_root / "room_layout.json", layout)
        write_json(plan_root / "voice_bindings.json", plan["voice_bindings"])
        write_json(plan_root / "audio_events.json", plan["audio_events"])
        nav = plan["room_capabilities"]["evidence_refs"]["navigation"]
        np.savez_compressed(plan_root / "navigation.npz",
                            binary_navmesh=pf.get_topdown_view(nav["resolution_m"], nav["floor_height_m"]),
                            bounds_m=pf.get_bounds())
        write_json(output / "room_selection.json", {"attempts": attempts})
        write_json(output / "planning_result.json", {
            "status": "research_candidate", "episode_id": plan["episode_id"],
            "elapsed_seconds": time.monotonic() - started,
            "native_execution": "not_run",
            "selected_room_id": room["room_id"], "plan": str(plan_root / "episode_plan.json"),
        })
        return plan
    raise QAPlanningError(f"no existing room could realize the request: {attempts}")


def renderer_capture_entrypoint(renderer: str) -> Path:
    entries = {"ue_spear": "tools/rooms/run_spear_residential_episode.py",
               "habitat": "tools/capture/capture_mp3d_multi_actor.py"}
    if renderer not in entries:
        raise QAPlanningError(f"unsupported renderer: {renderer}")
    return REPOSITORY / entries[renderer]


def capture_command(request: dict, output: Path) -> list[str]:
    runtime = request["runtime"]
    plan_path = output / "plan/episode_plan.json"
    saved_plan = read_json(plan_path)
    package_path = output / "plan/room_package.json"
    package = (read_json(package_path) if package_path.is_file() else
               package_from_catalog_entry(saved_plan["resources"], runtime=runtime))
    renderer = renderer_for_room(package)
    if renderer == "habitat":
        # P5 materializes these from the shared plan, preserving its clock.
        inputs = saved_plan.get("resources", {})
        case = Path(inputs.get("case_manifest", output / "plan/case_manifest.json"))
        sensor_request = Path(inputs.get("m1_request", output / "plan/m1_capture_request.json"))
        room_manifest = inputs.get("room_manifest")
        if not room_manifest or not case.is_file() or not sensor_request.is_file():
            raise QAPlanningError("Habitat capture requires materialized case_manifest, m1_request and room_manifest")
        case_clock = read_json(case)["clock"]
        from avengine.capture.neutral_readback import validate_clock
        validate_clock(case_clock)
        for key in ("frame_count", "frame_rate_hz", "sample_rate_hz", "sample_count",
                    "time_base_hz", "ticks_per_frame"):
            if case_clock[key] != saved_plan["clock"][key]:
                raise QAPlanningError(f"materialized Habitat clock differs: {key}")
        command = [sys.executable, str(renderer_capture_entrypoint(renderer)),
                   "--case-manifest", str(case), "--room-manifest", str(room_manifest),
                   "--m1-request", str(sensor_request), "--output", str(output / "capture"),
                   "--gpu-device-id", str(runtime.get("graphics_adapter", 0))]
        for key in ("runtime_prefix", "mp3d_root", "magnum_python_site", "rlr_sdk_root"):
            if runtime.get(key):
                command += ["--" + key.replace("_", "-"), str(runtime[key])]
        return command
    command = [sys.executable, str(renderer_capture_entrypoint(renderer)),
               "--episode-root", str(output / "plan"), "--output", str(output / "capture"),
               "--uproject", runtime["uproject"], "--unreal-editor", runtime["unreal_editor"],
               "--spear-ext-dir", runtime["spear_ext_dir"],
               "--graphics-adapter", str(runtime.get("graphics_adapter", 0)),
               "--rpc-port", str(runtime.get("rpc_port", 39379)),
               "--width", str(request.get("width", 1280)),
               "--height", str(request.get("height", 720)),
               "--streaming-warmup-frames", str(runtime.get("streaming_warmup_frames", 180)),
               "--native-multimodal", "--visual-only-research", "--keep-frames"]
    saved_plan = read_json(output / "plan/episode_plan.json")
    stage_count = saved_plan.get("resources", {}).get("expected_stage_actor_count", 1)
    command += ["--expected-stage-actor-count", str(stage_count)]
    for key in ("ddc_profile", "ddc_directory"):
        if runtime.get(key):
            command += ["--" + key.replace("_", "-"), str(runtime[key])]
    return command


def run(request: dict, output: Path, *, plan_only: bool = False,
        capture_only: bool = False, derived_output: Path | None = None) -> dict:
    output = output.expanduser().resolve()
    started = time.monotonic()
    plan = plan_request(request, output)
    if plan_only:
        return {"status": "research_candidate", "native_execution": "not_run",
                "episode_id": plan["episode_id"], "output": str(output)}
    command = capture_command(request, output)
    write_json(output / "execution_commands.json", {"capture": command})
    with (output / "capture.log").open("x") as log:
        subprocess.run(command, cwd=REPOSITORY, stdout=log, stderr=subprocess.STDOUT, check=True)
    package = read_json(output / "plan/room_package.json")
    if renderer_for_room(package) == "habitat":
        from avengine.capture.habitat_neutral_readback import write_habitat_neutral_readback
        neutral_path = output / "capture/neutral_readback.json"
        if neutral_path.exists():
            from avengine.capture.neutral_readback import validate_neutral_readback
            validate_neutral_readback(read_json(neutral_path), plan=plan)
        else:
            write_habitat_neutral_readback(output / "capture", plan)
    result = {
        "status": "research_only", "episode_id": plan["episode_id"],
        "output": str(output), "elapsed_seconds": time.monotonic() - started,
        "visual_capture": str(output / "capture"),
        "audio": "not_run", "qa": "not_run", "model_evaluation": "not_run",
    }
    if not capture_only:
        from avengine.rooms.qa_delivery import finalize_qa_episode
        delivery = finalize_qa_episode(
            output, derived_output or output / "delivery", repository=REPOSITORY,
            request=request)
        result.update(audio="pass", qa=delivery["questions"],
                      delivery=delivery["export"], preview=delivery["preview"])
        result["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "episode_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="finish an existing complete native capture into a fresh derived output")
    parser.add_argument("--derived-output", type=Path)
    parser.add_argument("--audio-report", type=Path, help="reuse a matching completed audio render")
    parser.add_argument("--appearance-review", type=Path, help="reuse an actual RGB appearance review")
    args = parser.parse_args()
    request = read_json(args.request)
    if args.resume:
        from avengine.rooms.qa_delivery import finalize_qa_episode
        result = finalize_qa_episode(
            args.output, args.derived_output or args.output / "delivery",
            repository=REPOSITORY, request=request, audio_report=args.audio_report,
            appearance_review=args.appearance_review)
        result = {key: value for key, value in result.items()
                  if key not in {"export_manifest", "coverage"}}
    else:
        result = run(request, args.output, plan_only=args.plan_only,
                     capture_only=args.capture_only, derived_output=args.derived_output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

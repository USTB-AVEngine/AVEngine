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
from avengine.rooms.room_package import (
    package_from_catalog_entry, canonical_runtime, renderer_for_room,
    room_capability_report, write_room_package_plan_snapshot,
)
from avengine.rooms.room_package import (
    host_runtime_layers, load_host_runtime_config,
)
from avengine.rooms.room_providers import (
    CAPTURE_ENTRYPOINTS, RoomRouteError, catalog_runtime, enumerate_catalog_rooms,
    load_profile_registry, load_room_catalog, planning_room_mapping,
    request_with_effective_camera, require_catalog_room, resolve_room_profile,
    room_render_parameters, room_route, runtime_isolation_groups,
)
from avengine.capture.qa_plan_adapters import (
    capture_adapter_binding, request_host_runtime_config,
)


def request_package_runtime(request: dict, catalog) -> dict:
    """Merge catalog path_bindings under request.runtime.path_bindings.

    Request bindings win. Catalog fills keys omitted by older requests so a
    request without runtime.path_bindings still expands from the catalog file.
    Shell environment variables are not consulted here.
    """
    catalog_bindings = catalog.get("path_bindings", {}) if isinstance(catalog, dict) else {}
    request_runtime = dict(request.get("runtime") or {})
    request_bindings = dict(request_runtime.get("path_bindings") or {})
    used_bindings = {**dict(catalog_bindings or {}), **request_bindings}
    return {**request_runtime, "path_bindings": used_bindings}


def _native_sampling_arguments(request: dict) -> dict:
    if request.get("sampling_policy") not in (None, "conditioned_static_v2"):
        raise QAPlanningError(f"unknown sampling_policy: {request['sampling_policy']}")
    if request.get("sampling_policy") != "conditioned_static_v2":
        return {"camera_motion": request.get("camera_motion", "follow_group")}
    camera = request.get("camera", {})
    # This path names each argument, so a request key that is not named here is dropped.
    # The other three planning calls receive the request itself and do not need a line.
    arguments = {"sampling_policy": "conditioned_static_v2",
                 "camera_motion": camera.get("motion", request.get("camera_motion", "static")),
                 "camera_fov_deg": camera.get("fov_deg", request.get("camera_fov_deg", 85.0)),
                 "silent_actor_count": int(request.get("silent_actor_count", 0))}
    # Forwarded only when asked for: the planner's own default is "none hidden".
    if request.get("offscreen_actor_ids"):
        arguments["offscreen_actor_ids"] = tuple(request["offscreen_actor_ids"])
    return arguments


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
    catalog_path = request["room_catalog"]
    catalog = read_json(catalog_path)
    package_runtime = request_package_runtime(request, catalog)
    host_runtime_config = request_host_runtime_config(
        request, runtime=package_runtime)
    profile_registry = load_profile_registry(request.get("room_profile_registry"))
    rooms = catalog.get("rooms", catalog) if isinstance(catalog, dict) else catalog
    conditioned = request.get("sampling_policy") == "conditioned_static_v2"
    sound_path = request.get("sound_selection", {}).get("prepared_set", request.get("sound_pool")) if conditioned else request["sound_pool"]
    sounds = read_json(sound_path)
    if conditioned:
        from avengine.rooms.conditioned_sampler import load_conditioned_sound_pool
        sounds = load_conditioned_sound_pool(sounds, source_path=sound_path)
    else:
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
            package = package_from_catalog_entry(
                room, runtime=package_runtime, catalog_path=catalog_path)
            route = room_route(package)
            renderer = route.renderer
            # The registered transport becomes an explicit camera request, so
            # the sampler's own field-of-view default cannot decide for a room.
            room_profile = resolve_room_profile(
                profile_registry, str(package["room_id"]),
                profile_id=request.get("room_runtime_profile_id"))
            render = room_render_parameters(room_profile, request)
            request = request_with_effective_camera(request, render)
            if "room_package" in room or room.get("schema") == package["schema"]:
                room = planning_room_mapping(package)
            binding = capture_adapter_binding(
                package, package_runtime, repository=REPOSITORY,
                host_config=host_runtime_config, room_id=package["room_id"])
            write_json(output / "renderer_dispatch.json", {
                "schema": "avengine_qa_room_dispatch_v1",
                "room_id": room["room_id"], "renderer": renderer,
                "family": route.family, "walkable_kind": route.walkable_kind,
                "planning_adapter": route.planning_adapter,
                "production_family": route.production_family,
                "capture_entrypoint": str(renderer_capture_entrypoint(renderer)),
                "selected_scene": binding["selected_scene"],
                "runtime": binding["runtime"],
                "capabilities": room_capability_report(
                    package, runtime=package_runtime)["dimensions"],
                "room_runtime_profile_id": render["profile_id"],
                "render": render,
                "package_validation_errors": package.get("validation_errors", []),
                "status": "dispatched", "native_execution": "not_run",
            })
            if conditioned:
                plan, layout, pf = build_qa_episode_plan(
                    room={**room, "room_package": package, "backend": renderer},
                    request=request, source_registry=registry, sounds=sounds)
                # Persist the profile actually solved from the request's QA targets.
                # A pre-resolved base profile would bypass compiled branch knobs.
                write_json(output / "condition_profile.json", plan["condition_profile"])
                if room.get("native_room_adapter") == "avengine_native_spear_apartment_qa_room_v1":
                    plan["resources"]["expected_stage_actor_count"] = 0
            elif renderer == "ue_spear" and room.get("native_room_adapter") == "avengine_native_spear_apartment_qa_room_v1":
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
                    audio_mode=request.get("audio_mode", "sequential"),
                    start_hold_frames=int(request.get("start_hold_frames", 0)),
                    **_native_sampling_arguments(request))
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
            if hasattr(exc, "result"):
                write_json(output / "planning_result.json", exc.result)
            attempts.append({"room_id": room.get("room_id"), "status": "not_selected",
                             "reason": f"{type(exc).__name__}: {exc}"})
            write_json(output / "room_selection.json", {"attempts": attempts})
            continue
        attempts.append({"room_id": room["room_id"], "status": "selected",
                         "condition_match": plan["question_condition_match"]})
        plan_root = output / "plan"
        plan_root.mkdir()
        write_json(plan_root / "episode_plan.json", plan)
        write_room_package_plan_snapshot(
            plan_root, package,
            path_bindings=package_runtime.get("path_bindings") or {},
            catalog_path=catalog_path)
        if renderer == "habitat" and conditioned:
            from avengine.capture.qa_plan_adapters import materialize_habitat_room_manifest
            from avengine.assets.mp3d_region_actor_tracks import materialize_common_plan_habitat
            native_manifest = plan_root / "habitat_room_manifest.json"
            materialize_habitat_room_manifest(package, room["room_manifest"], native_manifest)
            materialize_common_plan_habitat(plan=plan, room_manifest=native_manifest,
                runtime_registry=request.get("source_registry", REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"),
                output=plan_root / "habitat_execution", base_m1_request=room.get("m1_request"),
                habitat_binding_delta=request.get("habitat_binding_delta"),
                allow_research_candidate=bool(request.get("allow_research_candidate_assets", False)))
        write_json(plan_root / "room_layout.json", layout)
        write_json(plan_root / "voice_bindings.json", plan["voice_bindings"])
        write_json(plan_root / "audio_events.json", plan["audio_events"])
        nav = plan["room_capabilities"]["evidence_refs"]["navigation"]
        np.savez_compressed(plan_root / "navigation.npz",
                            binary_navmesh=pf.get_topdown_view(nav["resolution_m"], nav["floor_height_m"]),
                            bounds_m=pf.get_bounds())
        write_json(output / "room_selection.json", {"attempts": attempts})
        write_json(output / "planning_result.json", {
            **plan.get("planning_result", {}),
            "status": "research_candidate", "episode_id": plan["episode_id"],
            "elapsed_seconds": time.monotonic() - started,
            "native_execution": "not_run",
            "selected_room_id": room["room_id"], "plan": str(plan_root / "episode_plan.json"),
        })
        return plan
    if not (output / "planning_result.json").is_file():
        # No room solved a profile, so there is no solved profile to report. The
        # stated request profile and the per-room refusals are what a reader
        # needs, and the key stays so an existing consumer still finds it.
        write_json(output / "planning_result.json", {"status": "failed", "condition_profile": None,
                    "requested_profile": deepcopy(request.get("profile")),
                    "room_attempts": attempts, "gap_category": "evidence_missing_or_unsampled"})
    raise QAPlanningError(f"no existing room could realize the request: {attempts}")


def effective_capture_runtime(request: dict) -> dict:
    """Merge the run-local host runtime under the request's own runtime.

    The request wins per key. Without this the executor would only ever see
    the keys a request restated, which is the gap that made every room report
    its uproject or runtime_prefix missing.
    """
    runtime = dict(request.get("runtime") or {})
    host = request_host_runtime_config(request, runtime=runtime)
    if not host:
        return runtime
    catalog = read_json(request["room_catalog"]) if request.get("room_catalog") else {}
    renderer = _request_renderer(request, catalog) or ""
    merged: dict = {}
    for _source, mapping in host_runtime_layers(
        host, renderer, request.get("room_id")
    ):
        merged.update(mapping)
    merged.update(runtime)
    return merged


def _request_renderer(request: dict, catalog: dict) -> str | None:
    """The renderer of the room this request selected, from declared data."""
    room_id = request.get("room_id")
    rooms = catalog.get("rooms", catalog) if isinstance(catalog, dict) else catalog
    if not isinstance(rooms, list):
        return None
    for entry in rooms:
        if not isinstance(entry, dict):
            continue
        if room_id is None or entry.get("room_id") == room_id:
            return entry.get("renderer")
    return None


def renderer_capture_entrypoint(renderer: str) -> Path:
    """Resolve the executor for a renderer against the shared route table."""
    if renderer not in CAPTURE_ENTRYPOINTS:
        raise QAPlanningError(
            f"unsupported renderer: {renderer}; supported renderers are "
            f"{sorted(CAPTURE_ENTRYPOINTS)}"
        )
    return REPOSITORY / CAPTURE_ENTRYPOINTS[renderer]


def capture_command(request: dict, output: Path) -> list[str]:
    runtime = effective_capture_runtime(request)
    plan_path = output / "plan/episode_plan.json"
    saved_plan = read_json(plan_path)
    package_path = output / "plan/room_package.json"
    package = (read_json(package_path) if package_path.is_file() else
               package_from_catalog_entry(
                   saved_plan["resources"], runtime=runtime,
                   catalog_path=request.get("room_catalog")))
    renderer = renderer_for_room(package)
    if renderer == "habitat":
        # P5 materializes these from the shared plan, preserving its clock.
        inputs = saved_plan.get("resources", {})
        neutral = saved_plan.get("plan_coordinates") == "renderer_neutral"
        root = output / "plan/habitat_execution" if neutral else output / "plan"
        case = Path(inputs.get("case_manifest", root / "case_manifest.json"))
        sensor_request = Path(inputs.get("m1_request", root / "m1_capture_request.json"))
        room_manifest = str(output / "plan/habitat_room_manifest.json") if neutral else inputs.get("room_manifest")
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
                   "--gpu-device-id", str(runtime.get("graphics_adapter", 0)),
                   "--episode-plan", str(plan_path),
                   "--runtime-registry", str(request.get("source_registry", REPOSITORY / "examples/runtime/source_asset_runtime_profiles.json"))]
        if request.get("allow_research_candidate_assets"):
            command += ["--allow-research-candidate"]
        if request.get("habitat_binding_delta"):
            command += ["--habitat-binding-delta", str(request["habitat_binding_delta"])]
        for key in ("runtime_prefix", "mp3d_root", "magnum_python_site", "rlr_sdk_root"):
            if runtime.get(key):
                command += ["--" + key.replace("_", "-"), str(runtime[key])]
        return command
    dimensions = (saved_plan["visual_plan"]["camera"].get("resolution_hw", [720, 1280])
                  if saved_plan.get("plan_coordinates") == "renderer_neutral" else
                  [request.get("height", 720), request.get("width", 1280)])
    command = [sys.executable, str(renderer_capture_entrypoint(renderer)),
               "--episode-root", str(output / "plan"), "--output", str(output / "capture"),
               "--uproject", runtime["uproject"], "--unreal-editor", runtime["unreal_editor"],
               "--spear-ext-dir", runtime["spear_ext_dir"],
               "--graphics-adapter", str(runtime.get("graphics_adapter", 0)),
               "--rpc-port", str(runtime.get("rpc_port", 39379)),
               "--width", str(dimensions[1]),
               "--height", str(dimensions[0]),
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
    from avengine.dataset.binding_group_native import check_requested_visibility
    check_requested_visibility(plan, request, output / "capture",
                               report_path=output / "native_visibility_acceptance.json")
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
        if delivery.get("qa_target_results"):
            result["qa_target_results"] = delivery["qa_target_results"]
            result["qa_targets_met"] = delivery["qa_targets_met"]
        result["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "episode_result.json", result)
    return result


def room_catalog_runtime(catalog: dict, request: dict | None) -> dict:
    """The runtime a registry query uses: catalog bindings under the request."""
    return catalog_runtime(catalog, canonical_runtime((request or {}).get("runtime")))


def room_query_inputs(catalog: dict, request: dict | None,
                      *, host_runtime: Path | None = None,
                      profile_registry: Path | None = None) -> dict:
    """Resolve the three independent axes a room query needs.

    Room resources come from the catalog, the registered render transport from
    the profile registry, and the machine's executor parameters from the host
    runtime config. Keeping them separate is what lets the distributable
    examples stay free of server paths.
    """
    runtime = room_catalog_runtime(catalog, request)
    host = (load_host_runtime_config(host_runtime) if host_runtime is not None
            else request_host_runtime_config(request, runtime=runtime))
    return {
        "runtime": runtime,
        "host_config": host,
        "profile_registry": load_profile_registry(profile_registry),
    }


def list_rooms(catalog_path: Path, request: dict | None = None,
               *, production_only: bool = False, host_runtime: Path | None = None,
               profile_registry: Path | None = None,
               allow_environment: bool = False) -> dict:
    """Report every registered room, its route and why it is or is not ready."""
    catalog_path = catalog_path.expanduser().resolve()
    catalog = load_room_catalog(catalog_path)
    inputs = room_query_inputs(
        catalog, request, host_runtime=host_runtime,
        profile_registry=profile_registry)
    runtime = inputs["runtime"]
    resolutions = enumerate_catalog_rooms(
        catalog, catalog_path=catalog_path, runtime=runtime,
        production_only=production_only,
        profile_registry=inputs["profile_registry"],
        host_config=inputs["host_config"], request=request,
        allow_environment=allow_environment)
    return {
        "schema": "avengine_qa_room_registry_listing_v1",
        "room_catalog": str(catalog_path),
        "catalog_revision": catalog.get("revision"),
        "production_only": production_only,
        "host_runtime": None if not inputs["host_config"] else
            inputs["host_config"].get("_source"),
        "environment_allowed": allow_environment,
        "runtime_isolation_groups": [
            dict(group, group_key=list(group["group_key"]))
            for group in runtime_isolation_groups(resolutions)
        ],
        "rooms": [item.as_report() for item in resolutions],
        "claim_boundary": (
            "Declared resources resolve and each route has an adapter. This is "
            "not episode feasibility, native execution or dataset admission."
        ),
        "native_execution": "not_run",
    }


def resolve_room(catalog_path: Path, room_id: str, request: dict | None = None,
                 *, for_execution: bool = False, host_runtime: Path | None = None,
                 profile_registry: Path | None = None,
                 profile_id: str | None = None,
                 allow_environment: bool = False) -> dict:
    """Resolve one registered room, raising the exact reason when it cannot."""
    catalog_path = catalog_path.expanduser().resolve()
    catalog = load_room_catalog(catalog_path)
    inputs = room_query_inputs(
        catalog, request, host_runtime=host_runtime,
        profile_registry=profile_registry)
    runtime = inputs["runtime"]
    resolution = require_catalog_room(
        catalog, room_id, catalog_path=catalog_path, runtime=runtime,
        require_runtime=for_execution,
        profile_registry=inputs["profile_registry"], profile_id=profile_id,
        host_config=inputs["host_config"], request=request,
        allow_environment=allow_environment)
    report = resolution.as_report()
    report["room_catalog"] = str(catalog_path)
    report["host_runtime"] = (None if not inputs["host_config"] else
                              inputs["host_config"].get("_source"))
    report["capture_adapter"] = capture_adapter_binding(
        resolution.package, runtime, repository=REPOSITORY,
        host_config=inputs["host_config"], room_id=resolution.room_id,
        allow_environment=allow_environment)
    report["planning_room"] = resolution.planning_room
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="finish an existing complete native capture into a fresh derived output")
    parser.add_argument("--derived-output", type=Path)
    parser.add_argument("--audio-report", type=Path, help="reuse a matching completed audio render")
    parser.add_argument("--appearance-review", type=Path, help="reuse an actual RGB appearance review")
    parser.add_argument("--room-catalog", type=Path,
                        help="RoomPackage catalog for --list-rooms/--resolve-room")
    parser.add_argument("--list-rooms", action="store_true",
                        help="report every registered room, its route and its blockers")
    parser.add_argument("--resolve-room", metavar="ROOM_ID",
                        help="resolve one registered room and print its route")
    parser.add_argument("--production-only", action="store_true",
                        help="with --list-rooms, keep only production room families")
    parser.add_argument("--for-execution", action="store_true",
                        help="with --resolve-room, also require the renderer runtime parameters")
    parser.add_argument("--host-runtime", type=Path,
                        help="run-local host runtime config (server paths belong here, not in examples/)")
    parser.add_argument("--room-profile-registry", type=Path,
                        help="room runtime profile registry (default: examples/runtime/room_runtime_profiles.json)")
    parser.add_argument("--room-profile-id",
                        help="with --resolve-room, name one registered render transport")
    parser.add_argument("--allow-environment-runtime", action="store_true",
                        help="also read the established AVENGINE_* host runtime variables")
    args = parser.parse_args()
    if args.list_rooms or args.resolve_room:
        catalog_path = args.room_catalog
        request = read_json(args.request) if args.request else None
        if catalog_path is None and request is not None:
            catalog_path = Path(request["room_catalog"])
        if catalog_path is None:
            parser.error("--list-rooms/--resolve-room need --room-catalog or --request")
        try:
            result = (
                list_rooms(
                    catalog_path, request, production_only=args.production_only,
                    host_runtime=args.host_runtime,
                    profile_registry=args.room_profile_registry,
                    allow_environment=args.allow_environment_runtime)
                if args.list_rooms else
                resolve_room(
                    catalog_path, args.resolve_room, request,
                    for_execution=args.for_execution,
                    host_runtime=args.host_runtime,
                    profile_registry=args.room_profile_registry,
                    profile_id=args.room_profile_id,
                    allow_environment=args.allow_environment_runtime)
            )
        except RoomRouteError as error:
            print(json.dumps({"status": "blocked", "reason": str(error)},
                             ensure_ascii=False, indent=2))
            return 2
        if args.output is not None:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.request is None or args.output is None:
        parser.error("--request and --output are required to plan or run an Episode")
    request = read_json(args.request)
    if args.resume:
        from avengine.rooms.qa_delivery import finalize_qa_episode
        from avengine.dataset.binding_group_native import check_requested_visibility
        check_requested_visibility(
            read_json(args.output / "plan" / "episode_plan.json"),
            request, args.output / "capture")
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

#!/usr/bin/env python3
"""Build one scene-neutral four-actor/four-endpoint QA-v3 research canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

from build_qa_v3_programs import build_program  # noqa: E402
from qa_v3_actor_selection import _actor_entry  # noqa: E402
from make_idle_then_walk_timeline import transform_to_solved_routes  # noqa: E402
from scene_sampler import (  # noqa: E402
    effective_half_fov,
    load_scene,
    relative_azimuth_deg,
    sample_clear_yaw,
)
from avengine.timeline.current_apartment_visual import (  # noqa: E402
    author_current_n_actor_visual_timeline,
)
from avengine.contracts.json_io import sha256_file  # noqa: E402
from avengine.registry.registry import bind_content_hash  # noqa: E402
from avengine.camera_pose import apply_camera_listener_pose_ue  # noqa: E402
from avengine.dataset.apartment_dynamic_audio import (  # noqa: E402
    apartment_ue_point_to_world_m,
)


DEFAULT_ASSETS = [
    "generated_border_collie_black_white_medium_standard_adult_research_v1",
    "generated_labrador_yellow_medium_standard_adult_research_v1",
    "generated_shiba_inu_red_medium_standard_adult_research_v1",
    "generated_pembroke_welsh_corgi_red_white_medium_standard_adult_research_v1",
]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8")


class NRouteSearchExhausted(RuntimeError):
    """Finite N-route search ended without a candidate."""

    def __init__(self, message: str, *, evaluated_combinations: int):
        super().__init__(message)
        self.evaluated_combinations = int(evaluated_combinations)


def seed_uint64(seed: str) -> int:
    """Use the complete declared seed, including suffixes, as RNG entropy."""
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a non-empty string")
    return int.from_bytes(
        hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")


def find_n_route_plan(scene, params, *, actor_count: int, seed: str,
                      binding_frames=(12, 40),
                      min_pairwise_sep_deg=15.0,
                      max_attempts=20000):
    rng = np.random.default_rng(seed_uint64(seed))
    half_fov = effective_half_fov(scene, params)
    routes = [route for route in scene.routes
              if route.displacement_cm > 1.0e-6]
    if actor_count < 2:
        raise ValueError("actor_count must be at least two")
    if len(routes) < actor_count:
        raise ValueError(
            f"scene has fewer than {actor_count} moving routes")
    for attempt in range(1, max_attempts + 1):
        camera = scene.camera_points[int(rng.integers(len(scene.camera_points)))]
        picked = sample_clear_yaw(scene, params, camera, -180.0, 180.0, rng, None)
        if picked is None:
            continue
        yaw, clearance = picked
        indices = rng.permutation(len(routes))
        chosen = []
        azimuths = []
        for index in indices:
            route = routes[int(index)]
            values = [
                relative_azimuth_deg(camera, yaw, route.at(frame))
                for frame in binding_frames]
            if any(abs(value) > half_fov for value in values):
                continue
            if any(math.dist(camera, route.at(frame))
                   < float(params.get("MIN_CAMERA_DISTANCE_CM", 100.0))
                   for frame in binding_frames):
                continue
            if scene.line_of_sight is not None and not all(
                    scene.line_of_sight(camera, route.at(frame))
                    for frame in binding_frames):
                continue
            if any(
                min(abs((a - b + 180.0) % 360.0 - 180.0)
                    for a, b in zip(values, prior))
                < min_pairwise_sep_deg
                for prior in azimuths
            ):
                continue
            chosen.append(route)
            azimuths.append(values)
            if len(chosen) == actor_count:
                return {
                    "camera_xy": camera,
                    "camera_yaw_deg": yaw,
                    "routes": chosen,
                    "binding_azimuths_deg": azimuths,
                    "search_attempts": attempt,
                    "line_of_sight_screened": scene.line_of_sight_screened,
                    "camera_height_m": clearance["camera_height_m"],
                    "camera_clearance": clearance,
                }
    raise NRouteSearchExhausted(
        f"no {actor_count}-route plan within {max_attempts} attempts",
        evaluated_combinations=max_attempts)


def find_four_route_plan(scene, params, **kwargs):
    """Compatibility wrapper for the original four-actor canary."""
    return find_n_route_plan(scene, params, actor_count=4, **kwargs)


def build_endpoint_registry(selection, by_id, output_path: Path):
    endpoint_records = []
    evidence = sha256_file(output_path.parent / "actor_selection.json")
    for index, actor in enumerate(selection["actors"], start=1):
        asset = by_id[actor["asset_id"]]
        anchor_id = str(asset["default_emitter_anchor_id"])
        sound_class = (
            "human_speech" if asset.get("entity_class") == "articulated_human"
            else "animal_vocalization")
        endpoint_records.append({
            "source_endpoint_id": f"qa_v3_n{len(selection['actors'])}_source{index}_{anchor_id}",
            "revision": "v1",
            "binding": {
                "kind": "entity_anchor",
                "entity_instance_id": actor["legacy_timeline_actor_id"],
                "entity_asset_id": actor["asset_id"],
                "entity_asset_revision": actor["revision"],
                "emitter_anchor_id": anchor_id,
            },
            "source_visibility_mode": "visible_entity",
            "allowed_sound_class_ids": [sound_class],
            "directivity_profile_id": "point_emitter_v1",
            "persistent_when_silent": True,
            "admission_state": "research",
            "evidence_sha256": evidence,
        })
    registry = bind_content_hash({
        "schema": "avengine_m6_source_endpoint_registry_v1",
        "registry_id": f"qa_v3_n{len(selection['actors'])}_canary_endpoints_v1",
        "revision": "v1",
        "source_endpoints": sorted(
            endpoint_records, key=lambda item: item["source_endpoint_id"]),
    })
    _write(output_path, registry)
    return registry, endpoint_records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--asset", action="append")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--snapshot-content", required=True)
    parser.add_argument("--max-attempts", type=int, default=20000)
    args = parser.parse_args(argv)
    if args.out_root.exists():
        print(f"refusing to overwrite: {args.out_root}", file=sys.stderr)
        return 2
    assets = list(args.asset or DEFAULT_ASSETS)
    if len(assets) != 4 or len(set(assets)) != 4:
        parser.error("exactly four distinct assets are required")
    scene_config = _read(args.scene_config)
    params = _read(args.params)
    scene = load_scene(scene_config)
    plan = find_four_route_plan(
        scene, params, seed=args.seed, max_attempts=args.max_attempts)

    registry_path = REPO / "examples/runtime/source_asset_runtime_profiles.json"
    registry = _read(registry_path)
    by_id = {record["asset_id"]: record for record in registry["assets"]}
    missing = [asset for asset in assets if asset not in by_id]
    if missing:
        raise ValueError(f"assets absent from runtime registry: {missing}")
    args.out_root.mkdir(parents=True)
    selection = {
        "schema": "avengine_n_actor_selection_v1",
        "asset_authorization": "verified_internal",
        "research_only": True,
        "qualification_claim": False,
        "claim_boundary": "QA v3 four-actor research canary only",
        "actors": [
            _actor_entry(f"source{index}", asset, by_id,
                         args.snapshot_content)
            for index, asset in enumerate(assets, start=1)
        ],
    }
    selection_path = args.out_root / "actor_selection.json"
    _write(selection_path, selection)

    endpoint_registry_path = args.out_root / "source_endpoints.json"
    _, endpoint_records = build_endpoint_registry(
        selection, by_id, endpoint_registry_path)

    ground = float(scene.render_config["ground_z_ue_cm"])
    routes_3d = {
        f"source{index}": [
            [float(x), float(y), ground]
            for x, y in route.samples_xy]
        for index, route in enumerate(plan["routes"], start=1)
    }
    camera = [
        float(plan["camera_xy"][0]), float(plan["camera_xy"][1]),
        ground + scene.camera_height_m * 100.0,
    ]
    base_request = _read(Path(scene_config["camera_base_request"]))
    m1_request = apply_camera_listener_pose_ue(
        base_request,
        request_id=f"qa_v3_n4_{scene.scene_id}_{args.seed}",
        position_m=apartment_ue_point_to_world_m(camera),
        ue_yaw_degrees=float(plan["camera_yaw_deg"]),
        horizontal_fov_deg=scene.hfov_deg,
    )
    m1_request_path = args.out_root / "m1_capture_request.json"
    _write(m1_request_path, m1_request)
    authored_path = args.out_root / "timeline_authored.json"
    timeline = author_current_n_actor_visual_timeline(
        actor_selection_path=selection_path,
        source_asset_registry_path=registry_path,
        output_path=authored_path,
        camera_position_ue_cm=camera,
        camera_yaw_deg=float(plan["camera_yaw_deg"]),
        routes_by_slot_ue_cm=routes_3d,
        native_map=str(scene.render_config["native_map"]),
        room_profile_id=str(scene.render_config["room_profile_id"]),
        hfov_degrees=scene.hfov_deg,
    )
    timeline = transform_to_solved_routes(
        timeline,
        {slot: [(point[0], point[1]) for point in route]
         for slot, route in routes_3d.items()})
    timeline_path = args.out_root / "timeline.json"
    _write(timeline_path, timeline)

    slot_endpoints = {
        actor["source_slot_id"]: endpoint["source_endpoint_id"]
        for actor, endpoint in zip(selection["actors"], endpoint_records)
    }
    starts = [8000, 24000, 40000, 56000]
    events = [(f"source{index}", starts[index - 1])
              for index in range(1, 5)]
    program = build_program({
        "pair_kind": "n4",
        "point_id": "canary",
        "slot_endpoints": slot_endpoints,
        "sound_asset_id": params["SOUND_ASSET"],
    }, events)
    program_path = args.out_root / "audio_program.json"
    _write(program_path, program)
    manifest = {
        "schema": "qa_v3_n_actor_canary_manifest_v1",
        "status": "research_candidate",
        "qualification_claim": False,
        "scene_id": scene.scene_id,
        "scene_asset_id": scene_config.get(
            "scene_asset_id", scene.scene_id),
        "route_domain": scene_config.get("route_domain"),
        "actor_count": 4,
        "source_endpoint_count": 4,
        "assets": assets,
        "binding_frames": [12, 40],
        "binding_azimuths_deg": plan["binding_azimuths_deg"],
        "search_attempts": plan["search_attempts"],
        "line_of_sight_screened": plan["line_of_sight_screened"],
        "artifacts": {
            "actor_selection": str(selection_path),
            "timeline": str(timeline_path),
            "audio_program": str(program_path),
            "source_endpoint_registry": str(endpoint_registry_path),
            "m1_capture_request": str(m1_request_path),
        },
        "boundary": (
            "geometry/timeline/AudioProgram canary; visual capture, pixel "
            "truth, RIR rendering and question admission are not established"),
    }
    _write(args.out_root / "manifest.json", manifest)
    print(json.dumps({
        "out": str(args.out_root),
        "scene_id": scene.scene_id,
        "actors": 4,
        "endpoints": 4,
        "search_attempts": plan["search_attempts"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

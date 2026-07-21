#!/usr/bin/env python3
"""Compile Apartment feasibility, a four-case trajectory bank, and Topdown QA.

This is intentionally an audio/trajectory planning step.  It does not render
RGB, semantic observations, UE frames, dry events, or native RLR.  The reusable
contract contains source1/source2; an articulated person and Beagle are only
example visual-anchor bindings used to close this canary through Habitat's
source-center gate.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from avengine.contracts.json_io import load_json, write_json
from avengine.m1.contracts import load_and_validate_inputs as load_m1_inputs
from avengine.m1.habitat_capture import _make_configuration, discover_runtime_root
from avengine.m6x.apartment import (
    listener_orientation_wxyz,
    listener_yaw_degrees_from_request,
)
from avengine.m6x.articulated_anchor_profile import (
    materialize_articulated_anchor_paths,
)
from avengine.m6x.capture_adapter import HUMAN_BEAGLE_CAPTURE_ADAPTER
from avengine.m6x.feasibility_topdown import render_feasibility_topdown
from avengine.m6x.geometry import build_runtime_obstacle_map
from avengine.m6x.room_feasibility import (
    RoomFeasibilityCompiler,
    TrajectoryBankBuilder,
    build_rir_job_plan,
    evaluate_trajectory_coverage,
    evaluate_trajectory_diversity,
)


REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT_SCHEMA = "avengine_apartment_feasibility_bank_delivery_v2"
TIMING_SCHEMA = "avengine_apartment_feasibility_bank_timing_v1"


def _elapsed(started: float) -> float:
    return float(time.perf_counter() - started)


def _profile_actor_height_m(
    profile: dict[str, Any], actor_id: str, floor_m: float
) -> float:
    actors = [
        item for item in profile.get("actors", []) if item.get("actor_id") == actor_id
    ]
    if len(actors) != 1:
        raise RuntimeError(f"anchor profile must contain one {actor_id}")
    heights = np.asarray(
        [
            sample["actor_from_anchor_translation_m"][1]
            for sample in actors[0].get("samples", [])
        ],
        dtype=np.float64,
    )
    if heights.ndim != 1 or not len(heights) or not np.all(np.isfinite(heights)):
        raise RuntimeError(f"anchor profile {actor_id} heights are invalid")
    return float(floor_m + np.median(heights))


def _save_region_npz(path: Path, region: Any) -> None:
    np.savez_compressed(
        path,
        navmesh_mask=np.asarray(region.obstacle_map.binary_navmesh, dtype=np.uint8),
        feasible_mask=np.asarray(region.feasible_mask, dtype=np.uint8),
        component_labels=np.asarray(region.component_labels, dtype=np.int32),
        sample_pixels_rc=np.asarray(region.sample_pixels_rc, dtype=np.int32),
        sample_points_m=np.asarray(region.sample_points_m(), dtype=np.float64),
    )


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    output = args.output.resolve()
    staging = output.with_name(f".{output.name}.staging")
    if os.path.lexists(output) or os.path.lexists(staging):
        raise RuntimeError(f"refusing to replace output or staging path: {output}")
    staging.mkdir(parents=True)

    room_manifest = args.room_manifest.resolve()
    m1_request = args.m1_request.resolve()
    runtime_root = discover_runtime_root(args.runtime_root.resolve())
    anchor_library = load_json(args.anchor_library.resolve())
    trajectory_templates = load_json(args.trajectory_templates.resolve())
    anchor_profile = load_json(args.anchor_profile.resolve())
    inputs = load_m1_inputs(room_manifest, m1_request)
    floor_height = float(args.floor_height_m)
    listener_position = np.asarray(
        inputs.request["primary_camera_rig"]["world_from_rig"]["translation_m"],
        dtype=np.float64,
    )
    listener_yaw = listener_yaw_degrees_from_request(inputs.request)
    listener_orientation = listener_orientation_wxyz(listener_yaw)
    hfov = float(
        inputs.request["primary_camera_rig"]["shared_calibration"]["hfov_degrees"]
    )
    endpoint_to_actor = {
        str(actor["source_endpoint_id"]): str(actor["actor_id"])
        for actor in anchor_profile["actors"]
    }
    if set(endpoint_to_actor.values()) != {"human0", "dog0"}:
        raise RuntimeError("anchor profile does not describe human0 and dog0")
    slot_to_actor = {"source1": "human0", "source2": "dog0"}
    actor_to_endpoint = {
        actor_id: endpoint_id for endpoint_id, actor_id in endpoint_to_actor.items()
    }
    fallbacks = HUMAN_BEAGLE_CAPTURE_ADAPTER.materialize_actor_fallback_forwards_xz(
        trajectory_templates, anchor_library
    )

    # The pinned Habitat build needs numpy-quaternion imported before Habitat.
    import quaternion  # noqa: F401

    import habitat_sim
    import magnum as mn

    configuration, _modalities, _listener_uuid, resolved_scene = _make_configuration(
        inputs,
        runtime_root,
        room_manifest.parent / ".m6x_scratch",
    )
    try:
        phase_started = time.perf_counter()
        with habitat_sim.Simulator(configuration) as simulator:
            navmesh_path = resolved_scene.get("navmesh")
            if navmesh_path is None or not Path(navmesh_path).is_file():
                raise RuntimeError("Apartment declares no readable navmesh")
            if not simulator.pathfinder.load_nav_mesh(str(navmesh_path)):
                raise RuntimeError("Habitat could not load the Apartment navmesh")
            obstacle_map = build_runtime_obstacle_map(
                simulator.pathfinder,
                simulator.get_rigid_object_manager(),
                mn,
                floor_height_m=floor_height,
                meters_per_pixel=args.meters_per_pixel,
                excluded_handle_prefixes=("source_marker_",),
            )
            compiler = RoomFeasibilityCompiler(obstacle_map)
            source1_region = compiler.compile(
                source_center_height_m=_profile_actor_height_m(
                    anchor_profile, "human0", floor_height
                ),
                minimum_navmesh_clearance_m=args.minimum_navmesh_clearance_m,
                minimum_rigid_clearance_m=args.minimum_rigid_clearance_m,
                sample_spacing_m=args.sample_spacing_m,
            )
            source2_region = compiler.compile(
                source_center_height_m=_profile_actor_height_m(
                    anchor_profile, "dog0", floor_height
                ),
                minimum_navmesh_clearance_m=args.minimum_navmesh_clearance_m,
                minimum_rigid_clearance_m=args.minimum_rigid_clearance_m,
                sample_spacing_m=args.sample_spacing_m,
            )
            regions = {"source1": source1_region, "source2": source2_region}
            phase_seconds["room_load_and_feasible_region_compile"] = _elapsed(
                phase_started
            )

            def source_materializer(
                roots: dict[str, np.ndarray],
            ) -> dict[str, np.ndarray]:
                by_endpoint = materialize_articulated_anchor_paths(
                    anchor_profile,
                    actor_root_paths={
                        actor_id: roots[source_slot]
                        for source_slot, actor_id in slot_to_actor.items()
                    },
                    actor_fallback_forwards_xz=fallbacks,
                )
                return {
                    source_slot: by_endpoint[actor_to_endpoint[actor_id]]
                    for source_slot, actor_id in slot_to_actor.items()
                }

            phase_started = time.perf_counter()
            builder = TrajectoryBankBuilder(
                pathfinder=simulator.pathfinder,
                obstacle_map=obstacle_map,
                region_by_source=regions,
                shortest_path_factory=habitat_sim.ShortestPath,
                source_path_materializer=source_materializer,
            )
            bank = builder.build(
                episodes_per_motion_case=args.episodes_per_motion_case,
                frame_count=args.frame_count,
                frame_rate_hz=args.frame_rate_hz,
                seed=args.seed,
                minimum_route_distance_m=args.minimum_route_distance_m,
                maximum_route_distance_m=args.maximum_route_distance_m,
                minimum_pair_separation_m=args.minimum_pair_separation_m,
                maximum_floor_snap_xz_m=args.maximum_floor_snap_xz_m,
            )
            phase_seconds["trajectory_bank_generation_and_native_gate"] = _elapsed(
                phase_started
            )

            phase_started = time.perf_counter()
            coverage = evaluate_trajectory_coverage(regions, bank)
            if coverage.record["status"] != "pass":
                raise RuntimeError("trajectory bank does not broadly cover the room")
            diversity = evaluate_trajectory_diversity(bank)
            if diversity["status"] != "pass":
                raise RuntimeError(
                    "trajectory bank does not meet source-slot diversity"
                )
            phase_seconds["trajectory_coverage_audit"] = _elapsed(phase_started)

            phase_started = time.perf_counter()
            overview = render_feasibility_topdown(
                regions,
                bank,
                trajectory_coverage=coverage,
                listener_position_m=listener_position,
                listener_yaw_deg=listener_yaw,
                camera_hfov_degrees=hfov,
            )
            Image.fromarray(overview, mode="RGB").save(
                staging / "topdown_feasible_region_and_all_trajectories.png"
            )
            phase_seconds["topdown_overview"] = _elapsed(phase_started)
            phase_started = time.perf_counter()

        phase_seconds["habitat_runtime_teardown"] = _elapsed(phase_started)

        # The live simulator is no longer needed for serialization or RIR job
        # planning.  No room geometry, RGB, or RIR arrays are copied here.
        phase_started = time.perf_counter()
        write_json(
            staging / "feasible_region.json",
            {
                "schema": "avengine_room_feasible_region_set_v1",
                "room_id": inputs.room["room_id"],
                "obstacle_authority": obstacle_map.summary(),
                "source1": source1_region.summary(),
                "source2": source2_region.summary(),
                "intersection_feasible_pixel_count": int(
                    np.count_nonzero(
                        source1_region.feasible_mask & source2_region.feasible_mask
                    )
                ),
            },
        )
        _save_region_npz(staging / "feasible_region_source1.npz", source1_region)
        _save_region_npz(staging / "feasible_region_source2.npz", source2_region)
        write_json(staging / "trajectory_coverage.json", coverage.record)
        write_json(staging / "trajectory_diversity.json", diversity)
        np.savez_compressed(
            staging / "trajectory_coverage.npz",
            distance_to_trajectory_m=coverage.distance_to_trajectory_m,
            trajectory_seed_mask=coverage.trajectory_seed_mask,
        )
        write_json(staging / "trajectory_bank.json", bank.record(include_paths=True))
        source_order = ("source1", "source2")
        np.savez_compressed(
            staging / "trajectory_bank.npz",
            source_slot_ids=np.asarray(source_order),
            motion_cases=np.asarray([episode.motion_case for episode in bank.episodes]),
            episode_ids=np.asarray([episode.episode_id for episode in bank.episodes]),
            source_root_paths_m=np.stack(
                [
                    np.stack(
                        [episode.source_root_paths_m[key] for key in source_order],
                        axis=0,
                    )
                    for episode in bank.episodes
                ],
                axis=0,
            ),
            source_center_paths_m=np.stack(
                [
                    np.stack(
                        [episode.source_center_paths_m[key] for key in source_order],
                        axis=0,
                    )
                    for episode in bank.episodes
                ],
                axis=0,
            ),
        )
        rir_plan = build_rir_job_plan(
            bank,
            listener_position_m=listener_position,
            listener_orientation_wxyz=listener_orientation,
            stride_frames=args.rir_stride_frames,
        )
        write_json(staging / "rir_job_plan.json", rir_plan)
        phase_seconds["serialization_and_rir_job_planning"] = _elapsed(phase_started)
        total_seconds = _elapsed(started)
        delivery = {
            "schema": OUTPUT_SCHEMA,
            "status": "pass",
            "room_id": inputs.room["room_id"],
            "scope": (
                "complete raster source-center feasible region, finite four-motion "
                "trajectory bank, Topdown review, and planned-not-run RIR jobs"
            ),
            "mouth_animation_required": False,
            "visual_render_calls": 0,
            "native_rlr_calls": 0,
            "scene_asset_copies": 0,
            "source_slots": list(source_order),
            "example_visual_bindings": {
                source_slot: {
                    "actor_id": actor_id,
                    "source_endpoint_id": actor_to_endpoint[actor_id],
                }
                for source_slot, actor_id in slot_to_actor.items()
            },
            "dry_audio_is_replaceable_without_replanning_paths": True,
            "episode_count": len(bank.episodes),
            "event_variants_per_trajectory_for_1000_samples": (
                1000 / len(bank.episodes)
            ),
            "outputs": {
                "topdown": "topdown_feasible_region_and_all_trajectories.png",
                "feasible_region": "feasible_region.json",
                "trajectory_bank_json": "trajectory_bank.json",
                "trajectory_bank_arrays": "trajectory_bank.npz",
                "trajectory_coverage": "trajectory_coverage.json",
                "trajectory_diversity": "trajectory_diversity.json",
                "rir_job_plan": "rir_job_plan.json",
                "timing": "timing.json",
            },
            "rir_plan_summary": {
                key: rir_plan[key]
                for key in (
                    "status",
                    "requested_pair_state_count",
                    "unique_rir_job_count",
                    "cache_reuse_count",
                )
            },
            "coverage_summary": {
                key: coverage.record[key]
                for key in (
                    "status",
                    "coverage_fraction_by_threshold",
                    "mean_gap_m",
                    "p95_gap_m",
                    "maximum_gap_m",
                )
            },
            "diversity_summary": diversity,
        }
        write_json(staging / "delivery.json", delivery)
        write_json(
            staging / "timing.json",
            {
                "schema": TIMING_SCHEMA,
                "status": "pass",
                "clock": "time.perf_counter",
                "phase_wall_seconds": phase_seconds,
                "run_total_wall_seconds": total_seconds,
                "trajectory_episodes_per_second": len(bank.episodes)
                / phase_seconds["trajectory_bank_generation_and_native_gate"],
                "excluded_from_timing": [
                    "native RLR propagation",
                    "Habitat RGB/semantic rendering",
                    "UE/SPEAR rendering",
                    "dry-audio event assembly",
                    "video encoding",
                ],
            },
        )
        os.rename(staging, output)
    except Exception:
        write_json(
            staging / "FAILED_TIMING.json",
            {
                "schema": TIMING_SCHEMA,
                "status": "fail",
                "phase_wall_seconds": phase_seconds,
                "run_total_wall_seconds": _elapsed(started),
            },
        )
        raise
    print(
        "APARTMENT_FEASIBILITY_BANK_OK "
        f"output={output} "
        f"topdown={output / 'topdown_feasible_region_and_all_trajectories.png'}",
        flush=True,
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--room-manifest",
        type=Path,
        default=REPOSITORY / "tmp/m1/legacy_apartment_package/room_manifest.json",
    )
    parser.add_argument(
        "--m1-request",
        type=Path,
        default=REPOSITORY
        / "examples/m6x/fixed_apartment/m1_capture_request_review_720p.json",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=REPOSITORY.parent / "habitat-sim-AVEngine",
    )
    parser.add_argument(
        "--anchor-library",
        type=Path,
        default=REPOSITORY / "examples/m6x/fixed_apartment/anchor_library.json",
    )
    parser.add_argument(
        "--trajectory-templates",
        type=Path,
        default=REPOSITORY / "examples/m6x/fixed_apartment/trajectory_templates.json",
    )
    parser.add_argument(
        "--anchor-profile",
        type=Path,
        default=REPOSITORY / "tmp/m7/apartment_four_motion_pilot_20260720_01/shared/"
        "articulated_anchor_profile.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floor-height-m", type=float, default=0.271)
    parser.add_argument("--meters-per-pixel", type=float, default=0.02)
    parser.add_argument("--minimum-navmesh-clearance-m", type=float, default=0.02)
    parser.add_argument("--minimum-rigid-clearance-m", type=float, default=0.0)
    parser.add_argument("--sample-spacing-m", type=float, default=0.25)
    parser.add_argument("--episodes-per-motion-case", type=int, default=50)
    parser.add_argument("--frame-count", type=int, default=75)
    parser.add_argument("--frame-rate-hz", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20_260_721)
    parser.add_argument("--minimum-route-distance-m", type=float, default=3.5)
    parser.add_argument("--maximum-route-distance-m", type=float, default=5.5)
    parser.add_argument("--minimum-pair-separation-m", type=float, default=0.30)
    parser.add_argument("--maximum-floor-snap-xz-m", type=float, default=0.03)
    parser.add_argument("--rir-stride-frames", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

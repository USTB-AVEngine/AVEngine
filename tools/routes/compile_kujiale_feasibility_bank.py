#!/usr/bin/env python3
"""Compile a Kujiale room polygon into reusable source-center trajectories.

This is a trajectory/audio-planning stage only.  It performs no UE/Habitat
visual render and no native RLR propagation.  The optional USD extractor owns
the input polygon/object metadata; this tool owns deterministic raster A*, four
motion classes, coverage QA, Topdown, and a deduplicated planned-not-run RIR
job list.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from avengine.contracts.json_io import load_json, write_json
from avengine.m6x.apartment import listener_orientation_wxyz
from avengine.routes.feasibility_topdown import render_feasibility_topdown
from avengine.routes.geometry import (
    ELEVATED_OBJECT,
    GROUND_BLOCKER,
    UNKNOWN_OBSTACLE_ROLE,
    WALKABLE_FLOOR_COVERING,
)
from avengine.routes.raster_pathfinder import (
    RasterShortestPath,
    build_polygon_raster_obstacle_map,
)
from avengine.routes.room_feasibility import (
    RoomFeasibilityCompiler,
    TrajectoryBankBuilder,
    build_rir_job_plan,
    evaluate_trajectory_coverage,
    evaluate_trajectory_diversity,
)
from avengine.optional_backends.residential_episode import (
    DOG_SOURCE_ID,
    HUMAN_SOURCE_ID,
    PROFILE_SCHEMA,
    SCENE_METADATA_SCHEMA,
    dataset_z_up_to_habitat,
    object_footprint_rectangles_xy,
)


REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT_SCHEMA = "avengine_kujiale_feasibility_bank_delivery_v2"
TIMING_SCHEMA = "avengine_kujiale_feasibility_bank_timing_v1"
AUTHORITY = "kujiale_room_polygon_plus_usd_descendant_mesh_blocking_footprints"
CLAIM_BOUNDARY = (
    "source-center-only Kujiale polygon/footprint navigation with a declared "
    "clearance; not a Habitat navmesh, body-volume collision, or dataset admission"
)


def _elapsed(started: float) -> float:
    return float(time.perf_counter() - started)


def _finite(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{owner} must be a finite number")
    return result


def _emitter_heights(profile: Mapping[str, Any]) -> dict[str, float]:
    raw = profile.get("emitter_heights_m")
    if not isinstance(raw, Mapping) or set(raw) != {"human0", "dog0"}:
        raise RuntimeError("profile emitter_heights_m must contain human0 and dog0")
    result = {key: _finite(raw[key], owner=f"{key} emitter height") for key in raw}
    if any(value <= 0.0 for value in result.values()):
        raise RuntimeError("emitter heights must be positive")
    return result


def _rectangle_footprint(
    rectangle_xy_m: Sequence[Sequence[float]],
) -> list[list[float]]:
    value = np.asarray(rectangle_xy_m, dtype=np.float64)
    if (
        value.shape != (2, 2)
        or not np.all(np.isfinite(value))
        or np.any(value[1] <= value[0])
    ):
        raise RuntimeError("object footprint rectangle is invalid")
    return [
        [float(value[0, 0]), float(value[0, 1])],
        [float(value[1, 0]), float(value[0, 1])],
        [float(value[1, 0]), float(value[1, 1])],
        [float(value[0, 0]), float(value[1, 1])],
    ]


def _runtime_obstacles(
    metadata: Mapping[str, Any], emitter_heights_m: Mapping[str, float]
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    raw_objects = metadata.get("objects")
    if not isinstance(raw_objects, list):
        raise RuntimeError("scene metadata objects must be a list")
    allowed_roles = {
        ELEVATED_OBJECT,
        GROUND_BLOCKER,
        UNKNOWN_OBSTACLE_ROLE,
        WALKABLE_FLOOR_COVERING,
    }
    records: list[dict[str, Any]] = []
    blocking_object_ids: set[str] = set()
    nonblocking_height_object_ids: set[str] = set()
    for object_index, item in enumerate(raw_objects):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"objects[{object_index}] must be an object")
        object_id = str(item.get("object_id", object_index))
        role = str(item.get("navigation_role", UNKNOWN_OBSTACLE_ROLE))
        if role not in allowed_roles:
            role = UNKNOWN_OBSTACLE_ROLE
        bounds = np.asarray(item.get("bounds_xyz_m"), dtype=np.float64)
        if (
            bounds.shape != (2, 3)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds[1] <= bounds[0])
        ):
            raise RuntimeError(f"{object_id} bounds_xyz_m is invalid")
        intersects_emitter_height = any(
            bounds[0, 2] - 1.0e-9 <= height <= bounds[1, 2] + 1.0e-9
            for height in emitter_heights_m.values()
        )
        blocks = role == GROUND_BLOCKER and intersects_emitter_height
        if blocks:
            blocking_object_ids.add(object_id)
        elif role == GROUND_BLOCKER:
            nonblocking_height_object_ids.add(object_id)
        if role == GROUND_BLOCKER:
            rectangles = object_footprint_rectangles_xy(
                item, owner=f"objects[{object_index}]"
            )
            footprint_basis = item.get("footprint_basis", "top_level_bounds")
        else:
            rectangles = [[bounds[0, :2].tolist(), bounds[1, :2].tolist()]]
            footprint_basis = "top_level_bounds_review_only"
        for part_index, rectangle in enumerate(rectangles):
            footprint = _rectangle_footprint(rectangle)
            footprint_array = np.asarray(footprint, dtype=np.float64)
            center_xz = np.mean(footprint_array, axis=0)
            half_xz = (
                np.max(footprint_array, axis=0) - np.min(footprint_array, axis=0)
            ) / 2.0
            half_y = max(float((bounds[1, 2] - bounds[0, 2]) / 2.0), 1.0e-5)
            records.append(
                {
                    "object_id": f"{object_id}::{part_index:03d}",
                    "parent_object_id": object_id,
                    "handle": object_id,
                    "source": "interioragent_usd_metadata",
                    "obstacle_role": role,
                    "blocks_source_center": blocks,
                    "footprint_xz_m": footprint,
                    "footprint_basis": footprint_basis,
                    "world_obb": {
                        "center_m": [
                            float(center_xz[0]),
                            float((bounds[0, 2] + bounds[1, 2]) / 2.0),
                            float(center_xz[1]),
                        ],
                        "axes_xyz": np.eye(3).tolist(),
                        "half_extents_m": [
                            max(float(half_xz[0]), 1.0e-5),
                            half_y,
                            max(float(half_xz[1]), 1.0e-5),
                        ],
                    },
                    "world_aabb_m": {
                        "minimum": [
                            float(np.min(footprint_array[:, 0])),
                            float(bounds[0, 2]),
                            float(np.min(footprint_array[:, 1])),
                        ],
                        "maximum": [
                            float(np.max(footprint_array[:, 0])),
                            float(bounds[1, 2]),
                            float(np.max(footprint_array[:, 1])),
                        ],
                    },
                    "height_intersects_declared_emitter": intersects_emitter_height,
                }
            )
    records.sort(key=lambda item: str(item["object_id"]).encode("utf-8"))
    return tuple(records), {
        "metadata_object_count": len(raw_objects),
        "runtime_footprint_part_count": len(records),
        "blocking_object_count": len(blocking_object_ids),
        "blocking_object_ids": sorted(blocking_object_ids),
        "ground_objects_clear_of_both_emitter_heights": sorted(
            nonblocking_height_object_ids
        ),
    }


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
    try:
        phase_started = time.perf_counter()
        metadata = load_json(args.scene_metadata.resolve())
        profile = load_json(args.profile.resolve())
        if metadata.get("schema") != SCENE_METADATA_SCHEMA:
            raise RuntimeError(f"scene metadata schema must be {SCENE_METADATA_SCHEMA}")
        if profile.get("schema") != PROFILE_SCHEMA:
            raise RuntimeError(f"profile schema must be {PROFILE_SCHEMA}")
        if metadata.get("scene_id") != profile.get("scene_id"):
            raise RuntimeError("profile and metadata scene_id differ")
        polygon_xy = metadata.get("room_polygon_xy_m")
        if not isinstance(polygon_xy, list) or len(polygon_xy) < 3:
            raise RuntimeError("scene metadata lacks a room polygon")
        floor_height = _finite(metadata.get("floor_z_m", 0.0), owner="floor_z_m")
        visual_emitter_heights = _emitter_heights(profile)
        slot_to_visual_entity = {"source1": "human0", "source2": "dog0"}
        emitter_heights = {
            source_slot: visual_emitter_heights[visual_entity]
            for source_slot, visual_entity in slot_to_visual_entity.items()
        }
        clearance = (
            _finite(args.minimum_clearance_m, owner="minimum_clearance_m")
            if args.minimum_clearance_m is not None
            else _finite(
                profile.get("source_center_margin_m", 0.03),
                owner="profile source_center_margin_m",
            )
        )
        if clearance < 0.0:
            raise RuntimeError("minimum clearance cannot be negative")
        obstacles, obstacle_statistics = _runtime_obstacles(metadata, emitter_heights)
        pathfinder, obstacle_map = build_polygon_raster_obstacle_map(
            polygon_xz_m=polygon_xy,
            rigid_obstacles=obstacles,
            floor_height_m=floor_height,
            meters_per_pixel=args.meters_per_pixel,
            padding_m=max(
                args.meters_per_pixel * 2.0, clearance + args.meters_per_pixel
            ),
            minimum_clearance_m=clearance,
            authority=AUTHORITY,
            claim_boundary=CLAIM_BOUNDARY,
        )
        compiler = RoomFeasibilityCompiler(obstacle_map)
        regions = {
            actor_id: compiler.compile(
                source_center_height_m=floor_height + emitter_heights[actor_id],
                minimum_navmesh_clearance_m=0.0,
                minimum_rigid_clearance_m=0.0,
                sample_spacing_m=args.sample_spacing_m,
            )
            for actor_id in ("source1", "source2")
        }
        phase_seconds["input_and_feasible_region_compile"] = _elapsed(phase_started)

        def source_materializer(
            roots: Mapping[str, np.ndarray],
        ) -> dict[str, np.ndarray]:
            return {
                source_slot: np.asarray(roots[source_slot], dtype=np.float64)
                + np.asarray([0.0, emitter_heights[source_slot], 0.0])
                for source_slot in ("source1", "source2")
            }

        phase_started = time.perf_counter()
        bank = TrajectoryBankBuilder(
            pathfinder=pathfinder,
            obstacle_map=obstacle_map,
            region_by_source=regions,
            shortest_path_factory=RasterShortestPath,
            source_path_materializer=source_materializer,
        ).build(
            episodes_per_motion_case=args.episodes_per_motion_case,
            frame_count=args.frame_count,
            frame_rate_hz=args.frame_rate_hz,
            seed=args.seed,
            minimum_route_distance_m=args.minimum_route_distance_m,
            maximum_route_distance_m=args.maximum_route_distance_m,
            minimum_pair_separation_m=args.minimum_pair_separation_m,
            maximum_floor_snap_xz_m=args.maximum_floor_snap_xz_m,
            episode_attempts=args.episode_attempts,
            path_attempts=args.path_attempts,
        )
        phase_seconds["trajectory_bank_generation_and_authority_gate"] = _elapsed(
            phase_started
        )

        phase_started = time.perf_counter()
        coverage = evaluate_trajectory_coverage(regions, bank)
        if coverage.record["status"] != "pass":
            write_json(staging / "FAILED_COVERAGE.json", coverage.record)
            raise RuntimeError("trajectory bank does not broadly cover the room")
        diversity = evaluate_trajectory_diversity(bank)
        if diversity["status"] != "pass":
            write_json(staging / "FAILED_DIVERSITY.json", diversity)
            raise RuntimeError("trajectory bank does not meet source-slot diversity")
        phase_seconds["trajectory_coverage_audit"] = _elapsed(phase_started)

        camera = profile.get("camera")
        if not isinstance(camera, Mapping):
            raise RuntimeError("profile camera is missing")
        listener = np.asarray(
            dataset_z_up_to_habitat(camera.get("position_xyz_m")), dtype=np.float64
        )
        listener_yaw = -90.0 - _finite(camera.get("yaw_ue_deg"), owner="camera yaw")
        hfov = _finite(camera.get("horizontal_fov_deg"), owner="camera HFOV")
        listener_orientation = listener_orientation_wxyz(listener_yaw)

        phase_started = time.perf_counter()
        overview = render_feasibility_topdown(
            regions,
            bank,
            trajectory_coverage=coverage,
            listener_position_m=listener,
            listener_yaw_deg=listener_yaw,
            camera_hfov_degrees=hfov,
            room_label="Kujiale 0020 / livingroom 491",
            navigation_authority_label="Kujiale polygon + USD furniture footprints",
        )
        Image.fromarray(overview, mode="RGB").save(
            staging / "topdown_feasible_region_and_all_trajectories.png"
        )
        phase_seconds["topdown_overview"] = _elapsed(phase_started)

        phase_started = time.perf_counter()
        write_json(
            staging / "feasible_region.json",
            {
                "schema": "avengine_room_feasible_region_set_v1",
                "room_id": metadata.get("room_id"),
                "obstacle_authority": obstacle_map.summary(),
                "obstacle_statistics": obstacle_statistics,
                "declared_object_role_counts": metadata.get("object_role_counts"),
                "minimum_clearance_m": clearance,
                "emitter_heights_m": emitter_heights,
                "source1": regions["source1"].summary(),
                "source2": regions["source2"].summary(),
                "intersection_feasible_pixel_count": int(
                    np.count_nonzero(
                        regions["source1"].feasible_mask
                        & regions["source2"].feasible_mask
                    )
                ),
            },
        )
        _save_region_npz(staging / "feasible_region_source1.npz", regions["source1"])
        _save_region_npz(staging / "feasible_region_source2.npz", regions["source2"])
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
            listener_position_m=listener,
            listener_orientation_wxyz=listener_orientation,
            stride_frames=args.rir_stride_frames,
        )
        write_json(staging / "rir_job_plan.json", rir_plan)
        phase_seconds["serialization_and_rir_job_planning"] = _elapsed(phase_started)
        total_seconds = _elapsed(started)
        delivery = {
            "schema": OUTPUT_SCHEMA,
            "status": "pass",
            "room_id": metadata.get("room_id"),
            "scope": (
                "complete raster source-center feasible region, finite four-motion "
                "trajectory bank, coverage Topdown, and planned-not-run RIR jobs"
            ),
            "navigation_authority": AUTHORITY,
            "navigation_is_habitat_navmesh": False,
            "mouth_animation_required": False,
            "visual_render_calls": 0,
            "native_rlr_calls": 0,
            "scene_asset_copies": 0,
            "source_slots": list(source_order),
            "example_visual_bindings": {
                "source1": {
                    "visual_entity_id": "human0",
                    "legacy_endpoint_id": HUMAN_SOURCE_ID,
                },
                "source2": {
                    "visual_entity_id": "dog0",
                    "legacy_endpoint_id": DOG_SOURCE_ID,
                },
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
                / phase_seconds["trajectory_bank_generation_and_authority_gate"],
                "excluded_from_timing": [
                    "native RLR propagation",
                    "Habitat rendering",
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
        "KUJIALE_FEASIBILITY_BANK_OK "
        f"output={output} "
        f"topdown={output / 'topdown_feasible_region_and_all_trajectories.png'}",
        flush=True,
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-metadata", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=REPOSITORY
        / "examples/m6z/interioragent_kujiale_0020_source_episode.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    # InteriorAgent mesh footprints are extracted on a 0.05 m grid.  A finer
    # path raster would add A* nodes without adding obstacle evidence.
    parser.add_argument("--meters-per-pixel", type=float, default=0.05)
    parser.add_argument("--minimum-clearance-m", type=float)
    parser.add_argument("--sample-spacing-m", type=float, default=0.25)
    parser.add_argument("--episodes-per-motion-case", type=int, default=50)
    parser.add_argument("--frame-count", type=int, default=75)
    parser.add_argument("--frame-rate-hz", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20_260_721)
    parser.add_argument("--minimum-route-distance-m", type=float, default=3.5)
    parser.add_argument("--maximum-route-distance-m", type=float, default=5.5)
    parser.add_argument("--minimum-pair-separation-m", type=float, default=0.30)
    parser.add_argument("--maximum-floor-snap-xz-m", type=float, default=0.03)
    parser.add_argument("--episode-attempts", type=int, default=500)
    parser.add_argument("--path-attempts", type=int, default=500)
    parser.add_argument("--rir-stride-frames", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

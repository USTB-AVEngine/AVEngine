#!/usr/bin/env python3
"""Bind concrete assets to generic root routes and plan on-demand RIR work."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np

from avengine.contracts.json_io import load_json, write_json
from avengine.routes.asset_emitter import (
    bind_asset_emitters_to_bank,
    validate_asset_emitter_binding_set,
)
from avengine.routes.room_feasibility import (
    TRAJECTORY_BANK_SCHEMA,
    TrajectoryBank,
    TrajectoryEpisode,
    build_rir_job_plan,
)


SCENARIO_SET_SCHEMA = "avengine_asset_emitter_scenario_set_v1"
OUTPUT_SCHEMA = "avengine_asset_bound_rir_plan_delivery_v1"


def _asset_bound_bank_record(bank: TrajectoryBank) -> dict[str, Any]:
    """Make the generic-root/asset-bound-emitter distinction explicit."""

    record = bank.record(include_paths=True)
    record["semantics"] = (
        "source_root_paths_m remain generic source1/source2 actor-root routes; "
        "source_center_paths_m are concrete asset-bound emitter-point routes"
    )
    record["path_semantics"] = {
        "source_root_paths_m": (
            "generic source-slot actor roots; independent of dry audio and the "
            "concrete visual asset"
        ),
        "source_center_paths_m": (
            "asset-bound world emitter points after applying emitter_offset_m; "
            "used for the center gate and RIR planning"
        ),
    }
    return record


def _evaluate_navmesh_center_gate(
    bank: TrajectoryBank,
    *,
    navmesh_path: Path,
    floor_height_m: float,
    maximum_floor_snap_xz_m: float,
    minimum_navmesh_clearance_m: float,
) -> dict[str, Any]:
    """Check bound source centers against one declared Habitat navmesh."""

    if not np.isfinite(floor_height_m):
        raise RuntimeError("floor height must be finite")
    if maximum_floor_snap_xz_m < 0.0 or minimum_navmesh_clearance_m < 0.0:
        raise RuntimeError("navmesh gate thresholds must be nonnegative")
    # The pinned Habitat build requires numpy-quaternion to be imported first.
    import quaternion  # noqa: F401

    import habitat_sim

    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(navmesh_path.resolve())):
        raise RuntimeError("could not load the declared navmesh")
    sources: dict[str, Any] = {}
    failed_sources: dict[str, list[int]] = {}
    for episode in bank.episodes:
        for source_slot in ("source1", "source2"):
            source_id = f"{episode.episode_id}::{source_slot}"
            points = np.asarray(
                episode.source_center_paths_m[source_slot], dtype=np.float64
            )
            failed_frames: list[int] = []
            snap_distances: list[float] = []
            clearances: list[float] = []
            for frame_index, point in enumerate(points):
                floor_query = np.asarray(
                    [point[0], floor_height_m, point[2]], dtype=np.float64
                )
                navigable = bool(pathfinder.is_navigable(floor_query, 0.25))
                snapped = np.asarray(
                    pathfinder.snap_point(floor_query), dtype=np.float64
                )
                snap_xz = float(np.linalg.norm(snapped[(0, 2),] - floor_query[(0, 2),]))
                clearance = float(
                    pathfinder.distance_to_closest_obstacle(snapped, 10.0)
                )
                if (
                    not navigable
                    or snap_xz > maximum_floor_snap_xz_m
                    or clearance < minimum_navmesh_clearance_m
                ):
                    failed_frames.append(frame_index)
                snap_distances.append(snap_xz)
                clearances.append(clearance)
            if failed_frames:
                failed_sources[source_id] = failed_frames
            sources[source_id] = {
                "status": "fail" if failed_frames else "pass",
                "frame_count": len(points),
                "failed_frame_indices": failed_frames,
                "maximum_floor_snap_xz_m": max(snap_distances),
                "minimum_navmesh_clearance_m": min(clearances),
            }
    return {
        "schema": "avengine_asset_bound_navmesh_center_gate_v1",
        "status": "fail" if failed_sources else "pass",
        "claim_boundary": (
            "source-center X/Z against the declared navmesh only; separately "
            "loaded rigid obstacles require the room runtime gate"
        ),
        "navmesh_path": str(navmesh_path.resolve()),
        "floor_height_m": floor_height_m,
        "thresholds": {
            "maximum_floor_snap_xz_m": maximum_floor_snap_xz_m,
            "minimum_navmesh_clearance_m": minimum_navmesh_clearance_m,
        },
        "failed_sources": failed_sources,
        "sources": sources,
    }


def _bank_episode(value: Mapping[str, Any], *, frame_count: int) -> TrajectoryEpisode:
    episode_id = value.get("episode_id")
    motion_case = value.get("motion_case")
    if not isinstance(episode_id, str) or not episode_id:
        raise RuntimeError("trajectory episode lacks episode_id")
    if not isinstance(motion_case, str) or not motion_case:
        raise RuntimeError(f"trajectory episode {episode_id} lacks motion_case")
    raw_roots = value.get("source_root_paths_m")
    if not isinstance(raw_roots, Mapping) or set(raw_roots) != {"source1", "source2"}:
        raise RuntimeError(
            f"trajectory episode {episode_id} lacks source1/source2 root paths"
        )
    roots: dict[str, np.ndarray] = {}
    for source_slot in ("source1", "source2"):
        points = np.asarray(raw_roots[source_slot], dtype=np.float64)
        if points.shape != (frame_count, 3) or not np.all(np.isfinite(points)):
            raise RuntimeError(
                f"trajectory episode {episode_id} {source_slot} root path is invalid"
            )
        roots[source_slot] = np.ascontiguousarray(points)
    statistics = value.get("statistics", {})
    if not isinstance(statistics, Mapping):
        raise RuntimeError(f"trajectory episode {episode_id} statistics are invalid")
    return TrajectoryEpisode(
        episode_id=episode_id,
        motion_case=motion_case,
        source_root_paths_m=roots,
        source_center_paths_m=roots,
        statistics=dict(statistics),
    )


def _load_bank(path: Path) -> TrajectoryBank:
    value = load_json(path)
    if value.get("schema") != TRAJECTORY_BANK_SCHEMA:
        raise RuntimeError(f"trajectory bank schema must be {TRAJECTORY_BANK_SCHEMA}")
    frame_count = value.get("frame_count")
    frame_rate_hz = value.get("frame_rate_hz")
    seed = value.get("seed")
    raw_episodes = value.get("episodes")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 2
        or isinstance(frame_rate_hz, bool)
        or not isinstance(frame_rate_hz, int)
        or frame_rate_hz < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not isinstance(raw_episodes, list)
        or not raw_episodes
    ):
        raise RuntimeError("trajectory bank header is invalid")
    episodes = tuple(
        _bank_episode(episode, frame_count=frame_count) for episode in raw_episodes
    )
    if value.get("episode_count") != len(episodes):
        raise RuntimeError("trajectory bank episode count differs")
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise RuntimeError("trajectory bank contains duplicate episode IDs")
    return TrajectoryBank(
        episodes=episodes,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        seed=seed,
    )


def _load_listener(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    value = load_json(path)
    listener = np.asarray(value.get("listener_position_m"), dtype=np.float64)
    orientation = np.asarray(value.get("listener_orientation_wxyz"), dtype=np.float64)
    stride = value.get("stride_frames")
    if (
        listener.shape != (3,)
        or orientation.shape != (4,)
        or not np.all(np.isfinite(listener))
        or not np.all(np.isfinite(orientation))
        or isinstance(stride, bool)
        or not isinstance(stride, int)
        or stride < 1
    ):
        raise RuntimeError("template RIR plan listener/stride is invalid")
    return listener, orientation, stride


def build(
    *,
    trajectory_bank_path: Path,
    scenario_set_path: Path,
    template_rir_plan_path: Path,
    output: Path,
    navmesh_path: Path | None = None,
    floor_height_m: float | None = None,
    maximum_floor_snap_xz_m: float = 0.03,
    minimum_navmesh_clearance_m: float = 0.02,
) -> Path:
    started = time.perf_counter()
    bank = _load_bank(trajectory_bank_path)
    scenarios = load_json(scenario_set_path)
    if scenarios.get("schema") != SCENARIO_SET_SCHEMA:
        raise RuntimeError(f"scenario schema must be {SCENARIO_SET_SCHEMA}")
    raw_scenarios = scenarios.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise RuntimeError("scenario set must contain at least one scenario")
    listener, listener_orientation, stride = _load_listener(template_rir_plan_path)
    by_id = {episode.episode_id: episode for episode in bank.episodes}
    selected: list[TrajectoryEpisode] = []
    reports: list[dict[str, Any]] = []
    output_ids: set[str] = set()
    for index, raw in enumerate(raw_scenarios):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"scenarios[{index}] must be an object")
        source_episode_id = raw.get("trajectory_episode_id")
        output_episode_id = raw.get("output_episode_id")
        if not isinstance(source_episode_id, str) or source_episode_id not in by_id:
            raise RuntimeError(f"scenarios[{index}] selects an unknown trajectory")
        if (
            not isinstance(output_episode_id, str)
            or not output_episode_id
            or output_episode_id in output_ids
        ):
            raise RuntimeError("scenario output episode IDs must be unique")
        raw_binding_set = raw.get("binding_set")
        if not isinstance(raw_binding_set, Mapping):
            raise RuntimeError(f"scenarios[{index}] lacks binding_set")
        bindings = validate_asset_emitter_binding_set(raw_binding_set)
        source_episode = by_id[source_episode_id]
        one = TrajectoryBank(
            episodes=(source_episode,),
            frame_count=bank.frame_count,
            frame_rate_hz=bank.frame_rate_hz,
            seed=bank.seed,
        )
        bound, report = bind_asset_emitters_to_bank(
            one, bindings, listener_position_m=listener
        )
        episode = bound.episodes[0]
        statistics = deepcopy(dict(episode.statistics))
        statistics["source_trajectory_episode_id"] = source_episode_id
        selected.append(
            TrajectoryEpisode(
                episode_id=output_episode_id,
                motion_case=episode.motion_case,
                source_root_paths_m=episode.source_root_paths_m,
                source_center_paths_m=episode.source_center_paths_m,
                statistics=statistics,
            )
        )
        reports.append(
            {
                "output_episode_id": output_episode_id,
                "trajectory_episode_id": source_episode_id,
                "binding_report": report,
            }
        )
        output_ids.add(output_episode_id)
    bound_bank = TrajectoryBank(
        episodes=tuple(selected),
        frame_count=bank.frame_count,
        frame_rate_hz=bank.frame_rate_hz,
        seed=bank.seed,
    )
    if (navmesh_path is None) != (floor_height_m is None):
        raise RuntimeError("navmesh and floor height must be supplied together")
    navmesh_gate = None
    if navmesh_path is not None and floor_height_m is not None:
        navmesh_gate = _evaluate_navmesh_center_gate(
            bound_bank,
            navmesh_path=navmesh_path,
            floor_height_m=float(floor_height_m),
            maximum_floor_snap_xz_m=float(maximum_floor_snap_xz_m),
            minimum_navmesh_clearance_m=float(minimum_navmesh_clearance_m),
        )
        if navmesh_gate["status"] != "pass":
            failed = ", ".join(sorted(navmesh_gate["failed_sources"]))
            raise RuntimeError(f"bound source-center navmesh gate failed: {failed}")
    rir_plan = build_rir_job_plan(
        bound_bank,
        listener_position_m=listener,
        listener_orientation_wxyz=listener_orientation,
        stride_frames=stride,
    )

    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        write_json(staging / "request.json", scenarios)
        write_json(
            staging / "asset_emitter_binding_report.json",
            {
                "schema": "avengine_asset_emitter_scenario_report_v1",
                "status": "pass",
                "method": "constant_asset_root_offset",
                "scenario_count": len(reports),
                "scenarios": reports,
            },
        )
        if navmesh_gate is not None:
            write_json(staging / "navmesh_center_gate.json", navmesh_gate)
        write_json(
            staging / "trajectory_bank.json", _asset_bound_bank_record(bound_bank)
        )
        np.savez_compressed(
            staging / "trajectory_bank.npz",
            source_slot_ids=np.asarray(("source1", "source2")),
            episode_ids=np.asarray([episode.episode_id for episode in selected]),
            motion_cases=np.asarray([episode.motion_case for episode in selected]),
            source_root_paths_m=np.stack(
                [
                    np.stack(
                        [
                            episode.source_root_paths_m[slot]
                            for slot in ("source1", "source2")
                        ]
                    )
                    for episode in selected
                ]
            ),
            source_center_paths_m=np.stack(
                [
                    np.stack(
                        [
                            episode.source_center_paths_m[slot]
                            for slot in ("source1", "source2")
                        ]
                    )
                    for episode in selected
                ]
            ),
        )
        write_json(staging / "rir_job_plan.json", rir_plan)
        write_json(
            staging / "timing.json",
            {
                "schema": "avengine_asset_bound_rir_plan_timing_v1",
                "status": "pass",
                "wall_seconds": time.perf_counter() - started,
                "native_rlr_calls": 0,
                "visual_render_calls": 0,
            },
        )
        write_json(
            staging / "delivery.json",
            {
                "schema": OUTPUT_SCHEMA,
                "status": "pass",
                "scenario_count": len(selected),
                "frame_count": bank.frame_count,
                "frame_rate_hz": bank.frame_rate_hz,
                "rir_job_count": rir_plan["unique_rir_job_count"],
                "mouth_animation_required": False,
                "outputs": {
                    "binding_report": "asset_emitter_binding_report.json",
                    "navmesh_center_gate": (
                        "navmesh_center_gate.json" if navmesh_gate is not None else None
                    ),
                    "trajectory_bank": "trajectory_bank.json",
                    "trajectory_arrays": "trajectory_bank.npz",
                    "rir_job_plan": "rir_job_plan.json",
                    "timing": "timing.json",
                },
            },
        )
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-bank", type=Path, required=True)
    parser.add_argument("--scenario-set", type=Path, required=True)
    parser.add_argument("--template-rir-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--navmesh", type=Path)
    parser.add_argument("--floor-height-m", type=float)
    parser.add_argument("--maximum-floor-snap-xz-m", type=float, default=0.03)
    parser.add_argument("--minimum-navmesh-clearance-m", type=float, default=0.02)
    args = parser.parse_args()
    result = build(
        trajectory_bank_path=args.trajectory_bank.resolve(),
        scenario_set_path=args.scenario_set.resolve(),
        template_rir_plan_path=args.template_rir_plan.resolve(),
        output=args.output,
        navmesh_path=args.navmesh.resolve() if args.navmesh is not None else None,
        floor_height_m=args.floor_height_m,
        maximum_floor_snap_xz_m=args.maximum_floor_snap_xz_m,
        minimum_navmesh_clearance_m=args.minimum_navmesh_clearance_m,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

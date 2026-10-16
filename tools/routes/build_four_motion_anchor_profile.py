#!/usr/bin/env python3
"""Build and benchmark the reusable emitter-anchor profile from a pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from avengine.contracts.json_io import load_json, write_json
from avengine.assets.habitat_capture import quaternion_xyzw_to_matrix
from avengine.routes.articulated_anchor_profile import (
    AnchorProfileSpec,
    compile_articulated_anchor_profile,
    materialize_articulated_anchor_paths,
)


EPISODE_DIRECTORIES = (
    "00_static_static",
    "01_human_moving_dog_static",
    "02_both_moving",
    "03_human_static_dog_moving",
)


def _actor_matrices(pilot: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    by_actor: dict[str, list[np.ndarray]] = {"human0": [], "dog0": []}
    for directory in EPISODE_DIRECTORIES:
        timeline = load_json(pilot / "episodes" / directory / "metadata/timeline.json")
        for frame in timeline["frames"]:
            states = {value["actor_id"]: value for value in frame["actor_states"]}
            for actor_id in by_actor:
                transform = states[actor_id]["root_transform"]
                matrix = np.eye(4, dtype=np.float64)
                matrix[:3, :3] = quaternion_xyzw_to_matrix(transform["rotation_xyzw"])
                matrix[:3, 3] = transform["translation_m"]
                by_actor[actor_id].append(matrix)
    arrays = {
        actor_id: np.ascontiguousarray(np.stack(values, axis=0))
        for actor_id, values in by_actor.items()
    }
    return (
        np.ascontiguousarray(np.stack((arrays["human0"], arrays["dog0"]), axis=1)),
        arrays,
    )


def _fallback(
    matrix: np.ndarray, local_axis: tuple[float, float, float]
) -> list[float]:
    world = matrix[:3, :3] @ np.asarray(local_axis, dtype=np.float64)
    horizontal = world[[0, 2]]
    horizontal /= np.linalg.norm(horizontal)
    return horizontal.tolist()


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    pilot = args.pilot.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output}")
    records = json.loads(
        (pilot / "shared/master_capture/frame_readback.json").read_text(
            encoding="utf-8"
        )
    )
    evidence = load_json(pilot / "shared/master_capture/evidence.json")
    matrices, by_actor = _actor_matrices(pilot)
    specs = (
        AnchorProfileSpec(
            source_endpoint_id="m6x_human0_mouth",
            actor_id="human0",
            asset_id="rocketbox_human_male_adult_01_m5_1_candidate",
            record_key="human",
            anchor_id="mouth",
            anchor_record_key="mouth_emitter_anchor_m",
            capture_matrix_index=0,
            local_anatomical_forward_axis=(0.0, 0.0, 1.0),
            action_sample_counts=evidence["runtime"]["human_action_sample_counts"],
        ),
        AnchorProfileSpec(
            source_endpoint_id="m6x_dog0_muzzle",
            actor_id="dog0",
            asset_id="rocketbox_dog_beagle_01_m2_v7_world_contact_candidate",
            record_key="beagle",
            anchor_id="muzzle",
            anchor_record_key="mouth_emitter_anchor_m",
            capture_matrix_index=1,
            local_anatomical_forward_axis=(1.0, 0.0, 0.0),
            action_sample_counts=evidence["runtime"]["beagle_action_sample_counts"],
        ),
    )
    profile_started = time.perf_counter()
    profile = compile_articulated_anchor_profile(
        actor_world_matrices=matrices,
        frame_records=records,
        specs=specs,
    )
    compile_seconds = time.perf_counter() - profile_started
    roots = {
        actor_id: matrices_value[:, :3, 3]
        for actor_id, matrices_value in by_actor.items()
    }
    fallbacks = {
        "human0": _fallback(by_actor["human0"][0], (0.0, 0.0, 1.0)),
        "dog0": _fallback(by_actor["dog0"][0], (1.0, 0.0, 0.0)),
    }
    reconstructed = materialize_articulated_anchor_paths(
        profile,
        actor_root_paths=roots,
        actor_fallback_forwards_xz=fallbacks,
    )
    expected = {
        "m6x_human0_mouth": np.asarray(
            [value["human"]["mouth_emitter_anchor_m"] for value in records],
            dtype=np.float64,
        ),
        "m6x_dog0_muzzle": np.asarray(
            [value["beagle"]["mouth_emitter_anchor_m"] for value in records],
            dtype=np.float64,
        ),
    }
    errors = {
        source_id: float(
            np.max(np.linalg.norm(reconstructed[source_id] - positions, axis=1))
        )
        for source_id, positions in expected.items()
    }
    if max(errors.values()) > args.maximum_reconstruction_error_m:
        raise RuntimeError(f"anchor profile reconstruction failed: {errors}")

    benchmark_started = time.perf_counter()
    for _ in range(args.benchmark_iterations):
        materialize_articulated_anchor_paths(
            profile,
            actor_root_paths=roots,
            actor_fallback_forwards_xz=fallbacks,
        )
    benchmark_seconds = time.perf_counter() - benchmark_started
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, profile)
    receipt = {
        "schema": "avengine_articulated_anchor_profile_benchmark_v1",
        "status": "pass",
        "profile": output.name,
        "frame_count_per_iteration": len(records),
        "source_count": len(reconstructed),
        "compile_wall_seconds": compile_seconds,
        "benchmark_iterations": args.benchmark_iterations,
        "benchmark_wall_seconds": benchmark_seconds,
        "materializations_per_second": args.benchmark_iterations / benchmark_seconds,
        "maximum_reconstruction_error_m": errors,
        "run_total_wall_seconds": time.perf_counter() - started,
        "visual_observation_calls": 0,
        "semantics": (
            "reuses a validated asset/action/sample anchor profile; does not "
            "render Habitat RGB or semantic frames"
        ),
    }
    if not all(math.isfinite(value) for value in errors.values()):
        raise RuntimeError("anchor profile receipt is not finite")
    write_json(output.with_name("articulated_anchor_profile_benchmark.json"), receipt)
    print(
        f"ARTICULATED_ANCHOR_PROFILE_OK profile={output} receipt="
        f"{output.with_name('articulated_anchor_profile_benchmark.json')}",
        flush=True,
    )
    return output


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot",
        type=Path,
        default=repository / "tmp/m7/apartment_four_motion_pilot_20260720_01",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "tmp/m7/apartment_four_motion_pilot_20260720_01/shared/"
        "articulated_anchor_profile.json",
    )
    parser.add_argument("--benchmark-iterations", type=int, default=1000)
    parser.add_argument("--maximum-reconstruction-error-m", type=float, default=2.0e-5)
    args = parser.parse_args()
    if args.benchmark_iterations < 1:
        parser.error("--benchmark-iterations must be positive")
    if args.maximum_reconstruction_error_m <= 0.0:
        parser.error("--maximum-reconstruction-error-m must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())

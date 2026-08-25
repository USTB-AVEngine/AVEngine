#!/usr/bin/env python3
"""Build many unique two-source episodes from one finite single-path pool."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time
from typing import Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import load_json, write_json
from avengine.routes.room_feasibility import TrajectoryBank, TrajectoryEpisode
from avengine.m7.trajectory_recombination import recombine_source_paths


def _load_bank(path: Path) -> TrajectoryBank:
    value = load_json(path)
    raw_episodes = value.get("episodes")
    if (
        value.get("schema") != "avengine_room_trajectory_bank_v2"
        or not isinstance(raw_episodes, list)
    ):
        raise RuntimeError("input trajectory bank is invalid")
    episodes = []
    for raw in raw_episodes:
        if not isinstance(raw, Mapping):
            raise RuntimeError("input trajectory episode is invalid")
        episodes.append(
            TrajectoryEpisode(
                episode_id=str(raw["episode_id"]),
                motion_case=str(raw["motion_case"]),
                source_root_paths_m={
                    slot: np.asarray(raw["source_root_paths_m"][slot], dtype=np.float64)
                    for slot in ("source1", "source2")
                },
                source_center_paths_m={
                    slot: np.asarray(
                        raw["source_center_paths_m"][slot], dtype=np.float64
                    )
                    for slot in ("source1", "source2")
                },
                statistics=dict(raw["statistics"]),
            )
        )
    return TrajectoryBank(
        episodes=tuple(episodes),
        frame_count=int(value["frame_count"]),
        frame_rate_hz=int(value["frame_rate_hz"]),
        seed=int(value["seed"]),
    )


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    output = args.output.resolve()
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if output.exists() or output.is_symlink() or staging.exists() or staging.is_symlink():
        raise RuntimeError(f"refusing to replace output or staging: {output}")
    source = _load_bank(args.trajectory_bank.resolve())
    bank, report = recombine_source_paths(
        source,
        episodes_per_motion_case=args.episodes_per_motion_case,
        minimum_pair_separation_m=args.minimum_pair_separation_m,
        seed=args.seed,
    )
    staging.mkdir(parents=True)
    try:
        write_json(staging / "trajectory_bank.json", bank.record(include_paths=True))
        np.savez_compressed(
            staging / "trajectory_bank.npz",
            source_slot_ids=np.asarray(("source1", "source2")),
            motion_cases=np.asarray(
                [episode.motion_case for episode in bank.episodes]
            ),
            episode_ids=np.asarray(
                [episode.episode_id for episode in bank.episodes]
            ),
            source_root_paths_m=np.stack(
                [
                    np.stack(
                        [
                            episode.source_root_paths_m[slot]
                            for slot in ("source1", "source2")
                        ]
                    )
                    for episode in bank.episodes
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
                    for episode in bank.episodes
                ]
            ),
        )
        write_json(staging / "recombination_report.json", report)
        write_json(
            staging / "delivery.json",
            {
                "schema": "avengine_m7_recombined_trajectory_bank_delivery_v1",
                "status": "pass",
                "native_rlr_calls": 0,
                "visual_render_calls": 0,
                "scene_copy_count": 0,
                "episode_count": len(bank.episodes),
                "outputs": {
                    "trajectory_bank": "trajectory_bank.json",
                    "trajectory_arrays": "trajectory_bank.npz",
                    "report": "recombination_report.json",
                    "timing": "timing.json",
                },
            },
        )
        write_json(
            staging / "timing.json",
            {
                "schema": "avengine_m7_trajectory_recombination_timing_v1",
                "status": "pass",
                "wall_seconds": time.perf_counter() - started,
            },
        )
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-bank", type=Path, required=True)
    parser.add_argument("--episodes-per-motion-case", type=int, default=1_000)
    parser.add_argument("--minimum-pair-separation-m", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20_260_723)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    output = run(parse_args(argv))
    print(f"RECOMBINED_TRAJECTORY_BANK_OK output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

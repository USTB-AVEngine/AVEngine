"""Recombine a finite source-path pool into distinct two-source episodes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from avengine.routes.room_feasibility import (
    MOTION_CASES,
    TrajectoryBank,
    TrajectoryEpisode,
)


class TrajectoryRecombinationError(ValueError):
    """The requested two-source combinations cannot be built safely."""


@dataclass(frozen=True)
class SourcePath:
    path_id: str
    motion: str
    root_path_m: np.ndarray
    statistics: Mapping[str, Any]


MOTION_BY_CASE = {
    "static_static": ("static", "static"),
    "source1_moving_source2_static": ("moving", "static"),
    "source1_static_source2_moving": ("static", "moving"),
    "both_moving": ("moving", "moving"),
}


def extract_source_path_pool(bank: TrajectoryBank) -> tuple[SourcePath, ...]:
    """Treat every retained source-slot root path as a reusable path component."""

    result = []
    seen: set[str] = set()
    for episode in bank.episodes:
        for source_slot in ("source1", "source2"):
            path_id = f"{episode.episode_id}::{source_slot}"
            source_statistics = episode.statistics.get(source_slot)
            motion = (
                source_statistics.get("motion")
                if isinstance(source_statistics, Mapping)
                else None
            )
            path = np.asarray(
                episode.source_root_paths_m[source_slot], dtype=np.float64
            )
            if (
                path_id in seen
                or motion not in {"static", "moving"}
                or path.shape != (bank.frame_count, 3)
                or not np.all(np.isfinite(path))
            ):
                raise TrajectoryRecombinationError(
                    f"invalid reusable path component: {path_id}"
                )
            observed_motion = (
                "moving"
                if float(
                    np.linalg.norm(np.diff(path[:, (0, 2)], axis=0), axis=1).sum()
                )
                > 1.0e-8
                else "static"
            )
            if motion != observed_motion:
                raise TrajectoryRecombinationError(
                    f"path motion label differs from geometry: {path_id}"
                )
            result.append(
                SourcePath(
                    path_id=path_id,
                    motion=motion,
                    root_path_m=np.ascontiguousarray(path),
                    statistics=dict(source_statistics),
                )
            )
            seen.add(path_id)
    if not result:
        raise TrajectoryRecombinationError("source path pool is empty")
    return tuple(result)


def recombine_source_paths(
    bank: TrajectoryBank,
    *,
    episodes_per_motion_case: int,
    minimum_pair_separation_m: float = 0.30,
    seed: int = 20_260_723,
) -> tuple[TrajectoryBank, dict[str, Any]]:
    """Sample unique ordered path pairs while allowing component-path reuse."""

    if (
        isinstance(episodes_per_motion_case, bool)
        or not isinstance(episodes_per_motion_case, int)
        or episodes_per_motion_case < 1
    ):
        raise TrajectoryRecombinationError(
            "episodes_per_motion_case must be a positive integer"
        )
    if (
        not np.isfinite(minimum_pair_separation_m)
        or minimum_pair_separation_m < 0.0
    ):
        raise TrajectoryRecombinationError(
            "minimum_pair_separation_m must be finite and nonnegative"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TrajectoryRecombinationError("seed must be an integer")

    pool = extract_source_path_pool(bank)
    by_motion = {
        motion: tuple(value for value in pool if value.motion == motion)
        for motion in ("static", "moving")
    }
    if any(not values for values in by_motion.values()):
        raise TrajectoryRecombinationError(
            "source path pool must contain static and moving paths"
        )

    rng = np.random.default_rng(seed)
    episodes: list[TrajectoryEpisode] = []
    candidate_counts: dict[str, int] = {}
    selected_component_uses: Counter[str] = Counter()
    for motion_case in MOTION_CASES:
        source1_motion, source2_motion = MOTION_BY_CASE[motion_case]
        candidates: list[tuple[SourcePath, SourcePath, float]] = []
        for source1 in by_motion[source1_motion]:
            for source2 in by_motion[source2_motion]:
                if source1.path_id == source2.path_id:
                    continue
                separation = float(
                    np.min(
                        np.linalg.norm(
                            source1.root_path_m[:, (0, 2)]
                            - source2.root_path_m[:, (0, 2)],
                            axis=1,
                        )
                    )
                )
                if separation >= minimum_pair_separation_m:
                    candidates.append((source1, source2, separation))
        candidate_counts[motion_case] = len(candidates)
        if len(candidates) < episodes_per_motion_case:
            raise TrajectoryRecombinationError(
                f"{motion_case} has only {len(candidates)} valid ordered path pairs"
            )
        selected_indices = rng.permutation(len(candidates))[
            :episodes_per_motion_case
        ]
        for ordinal, candidate_index in enumerate(selected_indices):
            source1, source2, separation = candidates[int(candidate_index)]
            episode_id = f"recombined_{motion_case}_{ordinal:04d}"
            roots = {
                "source1": source1.root_path_m,
                "source2": source2.root_path_m,
            }
            selected_component_uses[source1.path_id] += 1
            selected_component_uses[source2.path_id] += 1
            episodes.append(
                TrajectoryEpisode(
                    episode_id=episode_id,
                    motion_case=motion_case,
                    source_root_paths_m=roots,
                    # Concrete asset binding replaces these provisional
                    # root-as-center paths before RIR planning.
                    source_center_paths_m=roots,
                    statistics={
                        "source1": dict(source1.statistics),
                        "source2": dict(source2.statistics),
                        "minimum_source_pair_root_xz_separation_m": separation,
                        "source_center_gate_status": (
                            "requires_concrete_asset_binding"
                        ),
                        "recombined_path_components": {
                            "source1": source1.path_id,
                            "source2": source2.path_id,
                        },
                    },
                )
            )
    output = TrajectoryBank(
        episodes=tuple(episodes),
        frame_count=bank.frame_count,
        frame_rate_hz=bank.frame_rate_hz,
        seed=seed,
    )
    ordered_pairs = {
        (
            str(episode.statistics["recombined_path_components"]["source1"]),
            str(episode.statistics["recombined_path_components"]["source2"]),
        )
        for episode in output.episodes
    }
    if len(ordered_pairs) != len(output.episodes):
        raise TrajectoryRecombinationError("recombined ordered path pairs repeat")
    report = {
        "schema": "avengine_m7_source_path_recombination_report_v1",
        "status": "pass",
        "claim_boundary": (
            "ordered root-path combination and root-center pair separation only; "
            "concrete emitter/navmesh gate still required"
        ),
        "input_episode_count": len(bank.episodes),
        "source_path_component_count": len(pool),
        "source_path_component_counts_by_motion": {
            motion: len(values) for motion, values in by_motion.items()
        },
        "candidate_ordered_pair_counts_by_motion_case": candidate_counts,
        "selected_episode_count": len(output.episodes),
        "selected_motion_case_counts": {
            motion_case: episodes_per_motion_case for motion_case in MOTION_CASES
        },
        "unique_ordered_path_pair_count": len(ordered_pairs),
        "unique_component_path_count_used": len(selected_component_uses),
        "component_path_reuse": {
            "minimum": min(selected_component_uses.values()),
            "maximum": max(selected_component_uses.values()),
            "mean": float(np.mean(tuple(selected_component_uses.values()))),
        },
        "minimum_pair_root_xz_separation_m": min(
            float(
                episode.statistics[
                    "minimum_source_pair_root_xz_separation_m"
                ]
            )
            for episode in output.episodes
        ),
        "seed": seed,
    }
    return output, report


__all__ = [
    "SourcePath",
    "TrajectoryRecombinationError",
    "extract_source_path_pool",
    "recombine_source_paths",
]

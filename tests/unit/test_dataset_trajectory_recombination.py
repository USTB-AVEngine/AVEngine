from __future__ import annotations

import numpy as np

from avengine.routes.room_feasibility import MOTION_CASES, TrajectoryBank, TrajectoryEpisode
from avengine.dataset.trajectory_recombination import recombine_source_paths


def _path(x: float, *, moving: bool) -> np.ndarray:
    end = x + 1.0 if moving else x
    return np.asarray([[x, 0.0, 0.0], [end, 0.0, 0.0]], dtype=np.float64)


def _bank() -> TrajectoryBank:
    episodes = []
    for ordinal in range(4):
        source1_moving = ordinal in {1, 3}
        source2_moving = ordinal in {2, 3}
        roots = {
            "source1": _path(float(ordinal * 10), moving=source1_moving),
            "source2": _path(float(ordinal * 10 + 4), moving=source2_moving),
        }
        episodes.append(
            TrajectoryEpisode(
                episode_id=f"input_{ordinal}",
                motion_case=MOTION_CASES[ordinal],
                source_root_paths_m=roots,
                source_center_paths_m=roots,
                statistics={
                    "source1": {
                        "motion": "moving" if source1_moving else "static",
                        "geodesic_distance_m": float(source1_moving),
                        "mean_speed_m_s": float(source1_moving),
                    },
                    "source2": {
                        "motion": "moving" if source2_moving else "static",
                        "geodesic_distance_m": float(source2_moving),
                        "mean_speed_m_s": float(source2_moving),
                    },
                },
            )
        )
    return TrajectoryBank(
        episodes=tuple(episodes), frame_count=2, frame_rate_hz=1, seed=1
    )


def test_recombines_components_into_unique_ordered_two_source_episodes():
    bank, report = recombine_source_paths(
        _bank(),
        episodes_per_motion_case=3,
        minimum_pair_separation_m=0.25,
        seed=7,
    )

    assert len(bank.episodes) == 12
    assert report["status"] == "pass"
    assert report["source_path_component_count"] == 8
    assert report["unique_ordered_path_pair_count"] == 12
    assert report["selected_motion_case_counts"] == {
        motion_case: 3 for motion_case in MOTION_CASES
    }
    combinations = [
        tuple(episode.statistics["recombined_path_components"].values())
        for episode in bank.episodes
    ]
    assert len(combinations) == len(set(combinations))
    assert report["component_path_reuse"]["maximum"] > 1

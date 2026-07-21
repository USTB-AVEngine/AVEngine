from __future__ import annotations

import numpy as np

from avengine.m6x.asset_emitter import ASSET_EMITTER_BINDING_SET_SCHEMA
from avengine.m6x.room_feasibility import MOTION_CASES, TrajectoryBank, TrajectoryEpisode
from tools.m6x import select_asset_bound_trajectories as selection


def _binding(asset_id: str) -> dict:
    return {
        "schema": ASSET_EMITTER_BINDING_SET_SCHEMA,
        "bindings": [
            {
                "source_slot_id": "source1",
                "asset_id": f"human_{asset_id}",
                "semantic_anchor_id": "mouth",
                "emitter_offset_m": [0.0, 1.6, 0.0],
                "local_anatomical_forward_axis": [0.0, 0.0, 1.0],
            },
            {
                "source_slot_id": "source2",
                "asset_id": f"cat_{asset_id}",
                "semantic_anchor_id": "muzzle",
                "emitter_offset_m": [0.2, 0.3, 0.0],
                "local_anatomical_forward_axis": [1.0, 0.0, 0.0],
            },
        ],
    }


def _bank() -> TrajectoryBank:
    episodes = []
    for index, motion_case in enumerate(MOTION_CASES):
        roots = {
            "source1": np.asarray([[index, 0.0, 0.0], [index, 0.0, -1.0]]),
            "source2": np.asarray([[index + 2.0, 0.0, 0.0], [index + 2.0, 0.0, 0.0]]),
        }
        episodes.append(
            TrajectoryEpisode(
                episode_id=f"{motion_case}_000",
                motion_case=motion_case,
                source_root_paths_m=roots,
                source_center_paths_m=roots,
                statistics={},
            )
        )
    return TrajectoryBank(
        episodes=tuple(episodes), frame_count=2, frame_rate_hz=1, seed=7
    )


def test_selects_balanced_asset_bound_scenarios_without_repairing_paths(
    monkeypatch,
    tmp_path,
) -> None:
    def all_pass(bank, **_kwargs):
        return {
            "status": "pass",
            "sources": {
                f"{episode.episode_id}::{slot}": {"status": "pass"}
                for episode in bank.episodes
                for slot in ("source1", "source2")
            },
        }

    monkeypatch.setattr(selection, "_evaluate_navmesh_center_gate", all_pass)
    scenarios, report = selection.select_scenarios(
        bank=_bank(),
        templates=(("pair_a", _binding("a")), ("pair_b", _binding("b"))),
        listener_position_m=np.asarray([0.0, 1.0, 0.0]),
        navmesh_path=tmp_path / "not_read_by_mock.navmesh",
        floor_height_m=0.0,
        episodes_per_pair=4,
        seed=9,
        maximum_floor_snap_xz_m=0.03,
        minimum_navmesh_clearance_m=0.02,
    )

    assert report["candidate_scenario_count"] == 8
    assert report["candidate_passing_scenario_count"] == 8
    assert report["selected_scenario_count"] == 8
    assert len(scenarios["scenarios"]) == 8
    assert {value["trajectory_episode_id"] for value in scenarios["scenarios"]} == {
        f"{motion_case}_000" for motion_case in MOTION_CASES
    }
    assert all(
        "__" in value["output_episode_id"] for value in scenarios["scenarios"]
    )
    assert report["per_pair"]["pair_a"]["motion_case_quotas"] == {
        motion_case: 1 for motion_case in MOTION_CASES
    }

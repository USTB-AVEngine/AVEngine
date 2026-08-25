from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import load_json, write_json
from avengine.routes.asset_emitter import (
    ASSET_EMITTER_BINDING_SET_SCHEMA,
    AssetEmitterBindingError,
    bind_asset_emitters_to_bank,
    materialize_asset_emitter_paths,
    validate_asset_emitter_binding_set,
)
from avengine.routes.room_feasibility import TrajectoryBank, TrajectoryEpisode
from tools.acoustics.build_asset_bound_rir_plan import build


def _config() -> dict:
    return {
        "schema": ASSET_EMITTER_BINDING_SET_SCHEMA,
        "bindings": [
            {
                "source_slot_id": "source1",
                "asset_id": "human_a",
                "semantic_anchor_id": "mouth",
                "emitter_offset_m": [0.0, 1.61, 0.0],
                "local_anatomical_forward_axis": [0.0, 0.0, 1.0],
            },
            {
                "source_slot_id": "source2",
                "asset_id": "cat_a",
                "semantic_anchor_id": "muzzle",
                "emitter_offset_m": [0.312, 0.252, 0.0],
                "local_anatomical_forward_axis": [1.0, 0.0, 0.0],
            },
        ],
    }


def test_materializes_asset_specific_offsets_and_keeps_static_emitter_fixed() -> None:
    bindings = validate_asset_emitter_binding_set(_config())
    roots = {
        "source1": np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, -2.0]]),
        "source2": np.asarray([[2.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
    }

    result = materialize_asset_emitter_paths(
        bindings,
        source_root_paths_m=roots,
        source_fallback_forwards_xz={
            "source1": np.asarray([0.0, -1.0]),
            "source2": np.asarray([-1.0, 0.0]),
        },
    )

    assert result.paths_m["source1"][:, 1] == pytest.approx([1.61] * 3)
    assert result.paths_m["source2"][:, 1] == pytest.approx([0.252] * 3)
    assert np.allclose(
        result.paths_m["source2"], result.paths_m["source2"][0], atol=0.0
    )
    assert result.report["mouth_animation_required"] is False
    assert result.report["skeleton_lookup_required"] is False
    assert result.report["sources"]["source2"]["emitter_motion"] == "static"


def test_binding_same_size_label_does_not_merge_asset_offsets() -> None:
    bindings = validate_asset_emitter_binding_set(_config())
    roots = {
        source_slot: np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        for source_slot in ("source1", "source2")
    }

    result = materialize_asset_emitter_paths(
        bindings,
        source_root_paths_m=roots,
        source_fallback_forwards_xz={"source1": [1.0, 0.0], "source2": [1.0, 0.0]},
    )

    assert np.allclose(result.paths_m["source1"][:, 1], 1.61)
    assert np.allclose(result.paths_m["source2"][:, 1], 0.252)
    assert not np.array_equal(result.paths_m["source1"], result.paths_m["source2"])


def test_binds_complete_bank_and_preserves_generic_root_paths() -> None:
    roots = {
        "source1": np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
        "source2": np.asarray([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    }
    bank = TrajectoryBank(
        episodes=(
            TrajectoryEpisode(
                episode_id="episode0",
                motion_case="source1_moving_source2_static",
                source_root_paths_m=roots,
                source_center_paths_m=roots,
                statistics={"kept": True},
            ),
        ),
        frame_count=2,
        frame_rate_hz=1,
        seed=7,
    )

    bound, report = bind_asset_emitters_to_bank(
        bank,
        validate_asset_emitter_binding_set(_config()),
        listener_position_m=[0.0, 1.6, 0.0],
    )

    assert np.array_equal(
        bound.episodes[0].source_root_paths_m["source1"], roots["source1"]
    )
    assert bound.episodes[0].source_center_paths_m["source1"][0, 1] == 1.61
    assert bound.episodes[0].source_center_paths_m["source2"][0, 1] == 0.252
    assert bound.episodes[0].statistics["kept"] is True
    assert report["episode_count"] == 1


def test_plan_delivery_distinguishes_generic_roots_from_bound_emitters(
    tmp_path: Path,
) -> None:
    roots = {
        "source1": [[0.0, 0.4, 0.0], [0.0, 0.4, -1.0]],
        "source2": [[2.0, 0.4, 0.0], [2.0, 0.4, 0.0]],
    }
    bank_path = tmp_path / "bank.json"
    scenario_path = tmp_path / "scenarios.json"
    template_path = tmp_path / "rir_template.json"
    output = tmp_path / "delivery"
    write_json(
        bank_path,
        {
            "schema": "avengine_room_trajectory_bank_v2",
            "frame_count": 2,
            "frame_rate_hz": 1,
            "seed": 7,
            "episode_count": 1,
            "episodes": [
                {
                    "episode_id": "generic_route_0",
                    "motion_case": "source1_moving_source2_static",
                    "source_root_paths_m": roots,
                    "statistics": {},
                }
            ],
        },
    )
    write_json(
        scenario_path,
        {
            "schema": "avengine_asset_emitter_scenario_set_v1",
            "scenarios": [
                {
                    "trajectory_episode_id": "generic_route_0",
                    "output_episode_id": "human_cat_0",
                    "binding_set": _config(),
                }
            ],
        },
    )
    write_json(
        template_path,
        {
            "listener_position_m": [0.0, 1.6, 0.0],
            "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "stride_frames": 1,
        },
    )

    build(
        trajectory_bank_path=bank_path,
        scenario_set_path=scenario_path,
        template_rir_plan_path=template_path,
        output=output,
    )

    delivered = load_json(output / "trajectory_bank.json")
    assert "generic source1/source2 actor-root routes" in delivered["semantics"]
    assert (
        "generic source-slot actor roots"
        in delivered["path_semantics"]["source_root_paths_m"]
    )
    assert (
        "asset-bound world emitter points"
        in delivered["path_semantics"]["source_center_paths_m"]
    )
    episode = delivered["episodes"][0]
    assert episode["source_root_paths_m"] == roots
    assert episode["source_center_paths_m"] != roots


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema="wrong"), "schema"),
        (
            lambda value: value["bindings"][1].update(source_slot_id="source1"),
            "exactly once",
        ),
        (
            lambda value: value["bindings"][0].update(emitter_offset_m=[0.0, 1.0]),
            "3 numbers",
        ),
        (
            lambda value: value["bindings"][0].update(
                local_anatomical_forward_axis=[0.0, 1.0, 0.0]
            ),
            "horizontal axis",
        ),
        (
            lambda value: value["bindings"][0].update(offset_space="bone"),
            "final_scaled_asset_root",
        ),
    ],
)
def test_binding_contract_fails_closed(mutation, message: str) -> None:
    value = deepcopy(_config())
    mutation(value)
    with pytest.raises(AssetEmitterBindingError, match=message):
        validate_asset_emitter_binding_set(value)

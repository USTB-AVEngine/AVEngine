from __future__ import annotations

import numpy as np

from avengine.m7.apartment_visual_bundle import (
    BORDER_COLLIE_ASSET_ID,
    CAT_ASSET_ID,
    binding_assets_by_episode,
    build_flags,
    build_source_manifest,
    build_timeline,
)


def _episode() -> dict:
    source1 = np.column_stack(
        (np.linspace(0.0, 2.0, 75), np.full(75, 0.27), np.zeros(75))
    )
    source2 = np.column_stack(
        (np.full(75, 1.0), np.full(75, 0.27), np.linspace(2.0, 0.0, 75))
    )
    return {
        "episode_id": "border_collie_cat__both_moving_000",
        "motion_case": "both_moving",
        "source_root_paths_m": {
            "source1": source1.tolist(),
            "source2": source2.tolist(),
        },
        "source_center_paths_m": {
            "source1": (source1 + [0.4, 0.65, 0.0]).tolist(),
            "source2": (source2 + [0.3, 0.25, 0.0]).tolist(),
        },
        "statistics": {
            "source1": {"motion": "moving"},
            "source2": {"motion": "moving"},
        },
    }


def _bindings() -> dict:
    return {
        "source1": {
            "source_slot_id": "source1",
            "asset_id": BORDER_COLLIE_ASSET_ID,
            "semantic_anchor_id": "muzzle",
        },
        "source2": {
            "source_slot_id": "source2",
            "asset_id": CAT_ASSET_ID,
            "semantic_anchor_id": "muzzle",
        },
    }


def test_generic_timeline_keeps_source_slots_and_asset_shapes_distinct() -> None:
    timeline, headings = build_timeline(
        episode=_episode(), bindings=_bindings(), listener_position_m=(-0.7, 1.47, 0.65)
    )
    assert [actor["actor_id"] for actor in timeline["actors"]] == [
        "source1_actor",
        "source2_actor",
    ]
    assert [actor["asset_id"] for actor in timeline["actors"]] == [
        BORDER_COLLIE_ASSET_ID,
        CAT_ASSET_ID,
    ]
    assert len(timeline["frames"]) == 75
    assert all(len(frame["actor_states"]) == 2 for frame in timeline["frames"])
    assert headings["source1"].shape == (75, 2)
    assert headings["source2"].shape == (75, 2)
    np.testing.assert_allclose(np.linalg.norm(headings["source1"], axis=1), 1.0)


def test_source_manifest_and_flags_close_over_generic_endpoint_ids() -> None:
    manifest = build_source_manifest(
        episode_id=_episode()["episode_id"], episode=_episode(), bindings=_bindings()
    )
    assert [source["source_endpoint_id"] for source in manifest["sources"]] == [
        "source1_emitter",
        "source2_emitter",
    ]
    assert manifest["sources"][1]["endpoint"]["binding"]["entity_asset_id"] == CAT_ASSET_ID
    flags = build_flags()
    assert set(flags["source_flags"]) == {"source1_emitter", "source2_emitter"}


def test_binding_report_requires_supported_exact_assets() -> None:
    report = {
        "status": "pass",
        "scenarios": [
            {
                "output_episode_id": _episode()["episode_id"],
                "binding_report": {"bindings": list(_bindings().values())},
            }
        ],
    }
    result = binding_assets_by_episode(report)
    assert result[_episode()["episode_id"]]["source2"]["asset_id"] == CAT_ASSET_ID

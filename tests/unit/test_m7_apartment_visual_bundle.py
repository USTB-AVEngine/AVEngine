from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import sha256_file
from avengine.m7.apartment_visual_bundle import (
    BORDER_COLLIE_ASSET_ID,
    CAT_ASSET_ID,
    binding_assets_by_episode,
    build_flags,
    build_source_manifest,
    build_timeline,
)

_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_asset_bound_apartment_ue_bundle",
    Path(__file__).resolve().parents[2]
    / "tools/m7/build_asset_bound_apartment_ue_bundle.py",
)
assert _BUILDER_SPEC is not None and _BUILDER_SPEC.loader is not None
_BUILDER = importlib.util.module_from_spec(_BUILDER_SPEC)
_BUILDER_SPEC.loader.exec_module(_BUILDER)

CURRENT_GENERATED_DOG_ASSET_ID = BORDER_COLLIE_ASSET_ID
CURRENT_GENERATED_CAT_ASSET_ID = CAT_ASSET_ID


def _episode() -> dict:
    source1 = np.column_stack(
        (np.linspace(0.0, 2.0, 75), np.full(75, 0.27), np.zeros(75))
    )
    source2 = np.column_stack(
        (np.full(75, 1.0), np.full(75, 0.27), np.linspace(2.0, 0.0, 75))
    )
    return {
        "episode_id": "current_dog_current_cat__both_moving_000",
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
            "asset_id": CURRENT_GENERATED_DOG_ASSET_ID,
            "semantic_anchor_id": "muzzle",
        },
        "source2": {
            "source_slot_id": "source2",
            "asset_id": CURRENT_GENERATED_CAT_ASSET_ID,
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
        CURRENT_GENERATED_DOG_ASSET_ID,
        CURRENT_GENERATED_CAT_ASSET_ID,
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
    assert (
        manifest["sources"][1]["endpoint"]["binding"]["entity_asset_id"]
        == CURRENT_GENERATED_CAT_ASSET_ID
    )
    assert manifest["sources"][1]["visible_asset"] == {
        "asset_id": CURRENT_GENERATED_CAT_ASSET_ID,
        "revision": "pixel3d_tokenrig_ue_v1",
        "display_label": "British Shorthair",
        "identity": {"species_id": "cat", "breed_id": "british_shorthair"},
        "realized_attributes": {
            "size": "medium",
            "body_build": "stocky",
            "life_stage": "adult",
            "coat_profile": {
                "profile_id": "cat_british_shorthair_coat_v1",
                "value": "standard_blue",
            },
        },
    }
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
    assert (
        result[_episode()["episode_id"]]["source2"]["asset_id"]
        == CURRENT_GENERATED_CAT_ASSET_ID
    )


def _runtime_emitter_binding(slot: str, asset_id: str) -> dict:
    return {
        "source_slot_id": slot,
        "asset_id": asset_id,
        "asset_revision": "runtime_v1",
        "semantic_anchor_id": "muzzle",
        "emitter_offset_m": [0.4, 0.3, 0.0],
        "local_anatomical_forward_axis": [1.0, 0.0, 0.0],
        "offset_space": "final_scaled_asset_root",
    }


def test_exact_runtime_snapshot_reuses_emitter_binding_and_keeps_legacy_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = {
        "source1": _runtime_emitter_binding("source1", "legacy_human"),
        "source2": _runtime_emitter_binding("source2", "exact_animal"),
    }

    def resolve(_registry, asset_id, _revision):
        return (
            {"asset_bound_lineage": {"schema": "lineage"}}
            if asset_id == "exact_animal"
            else {}
        )

    def exact(_registry, **kwargs):
        emitter = dict(bindings[kwargs["source_slot_id"]])
        return {
            "schema": "avengine_exact_asset_bound_runtime_binding_v1",
            "source_slot_id": kwargs["source_slot_id"],
            "asset_id": kwargs["asset_id"],
            "asset_revision": kwargs["revision"],
            "emitter": emitter,
            "actor_scale": 0.875,
        }

    monkeypatch.setattr(
        _BUILDER, "resolve_source_asset_runtime_profile", resolve
    )
    monkeypatch.setattr(
        _BUILDER, "build_exact_asset_bound_runtime_binding", exact
    )

    result = _BUILDER._resolve_exact_episode_runtime_bindings(
        episode_bindings=bindings,
        source_registry={"registry_id": "fixture"},
    )

    assert set(result) == {"source2"}
    assert result["source2"]["actor_scale"] == pytest.approx(0.875)
    assert result["source2"]["emitter"] == bindings["source2"]


def test_exact_runtime_snapshot_rejects_emitter_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = {
        slot: _runtime_emitter_binding(slot, f"exact_{slot}")
        for slot in ("source1", "source2")
    }
    monkeypatch.setattr(
        _BUILDER,
        "resolve_source_asset_runtime_profile",
        lambda *_args, **_kwargs: {
            "asset_bound_lineage": {"schema": "lineage"}
        },
    )

    def drifted(_registry, **kwargs):
        emitter = dict(bindings[kwargs["source_slot_id"]])
        emitter["emitter_offset_m"] = [9.0, 9.0, 9.0]
        return {"emitter": emitter}

    monkeypatch.setattr(
        _BUILDER, "build_exact_asset_bound_runtime_binding", drifted
    )

    with pytest.raises(RuntimeError, match="emitter_offset_m"):
        _BUILDER._resolve_exact_episode_runtime_bindings(
            episode_bindings=bindings,
            source_registry={"registry_id": "fixture"},
        )


def test_ue_input_resume_reopens_only_an_unchanged_atomic_episode(
    tmp_path: Path,
) -> None:
    episode_id = "episode_0001"
    root = tmp_path / "episodes" / episode_id
    metadata = root / "metadata"
    videos = root / "videos"
    metadata.mkdir(parents=True)
    videos.mkdir()
    diagnostic = videos / "diagnostic_topdown_binaural.mp4"
    diagnostic.write_bytes(b"completed diagnostic media")
    os.link(diagnostic, videos / "clean_binaural.mp4")
    for name in (
        "timeline.json",
        "source_manifest.json",
        "flags.json",
        "batch_binding.json",
    ):
        (metadata / name).write_text("{}", encoding="utf-8")
    row = {
        "episode_ordinal": 0,
        "episode_id": episode_id,
        "v00_sample_id": "sample_0001",
    }
    (metadata / "build_record.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "diagnostic_sha256": sha256_file(diagnostic),
                "row": row,
            }
        ),
        encoding="utf-8",
    )

    reopened = _BUILDER._load_completed_episode(
        staging=tmp_path,
        episode_id=episode_id,
        ordinal=17,
        sample={"sample_id": "sample_0001"},
        runtime_bindings_by_source_slot={},
    )
    assert reopened == {
        "episode_ordinal": 17,
        "episode_id": episode_id,
        "v00_sample_id": "sample_0001",
    }

    diagnostic.write_bytes(b"changed media")
    with pytest.raises(RuntimeError, match="completed episode changed"):
        _BUILDER._load_completed_episode(
            staging=tmp_path,
            episode_id=episode_id,
            ordinal=17,
            sample={"sample_id": "sample_0001"},
            runtime_bindings_by_source_slot={},
        )


def test_ue_input_rejects_visual_and_audio_asset_binding_mismatch() -> None:
    sample = {
        "asset_ids_by_source_slot": {
            "source1": CURRENT_GENERATED_DOG_ASSET_ID,
            "source2": "wrong_cat",
        }
    }
    with pytest.raises(
        RuntimeError, match="visual and audio asset bindings differ"
    ):
        _BUILDER._assert_sample_asset_alignment(
            episode_id=_episode()["episode_id"],
            episode_bindings=_bindings(),
            sample=sample,
        )

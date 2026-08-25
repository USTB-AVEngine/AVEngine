from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.timeline.audio_program import (
    bind_audio_program_hash,
    materialize_audio_program_variant,
)
from avengine.dataset.apartment_visual_bundle import (
    ApartmentVisualBundleError,
    BORDER_COLLIE_ASSET_ID,
    CAT_ASSET_ID,
    binding_assets_by_episode,
    build_flags,
    build_source_manifest,
    build_timeline,
    program_source_activity_by_frame,
    resolve_m7_sensor_rig_trajectory,
)
from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory

_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_asset_bound_apartment_ue_bundle",
    Path(__file__).resolve().parents[2]
    / "tools/dataset/build_asset_bound_apartment_ue_bundle.py",
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


def _materialized_audio_program() -> dict:
    return bind_audio_program_hash(
        {
            "schema": "avengine_m6_audio_program_v1",
            "program_id": "m7_visual_projection_v1",
            "revision": "v1",
            "mode": "one_active_of_n",
            "timeline": {
                "time_base_hz": 48_000,
                "ticks_per_frame": 3_200,
                "video_fps": 15,
                "frame_count": 75,
                "sample_rate_hz": 16_000,
                "ticks_per_sample": 3,
                "sample_count": 80_000,
            },
            "candidate_source_endpoint_ids": [
                "cat_program_endpoint",
                "dog_program_endpoint",
            ],
            "events": [
                {
                    "event_id": "dog_bark_0",
                    "source_endpoint_id": "dog_program_endpoint",
                    "sound_asset_id": "dog_beagle_v2_scheduled_dry",
                    "start_tick": 9_600,
                    "end_tick_exclusive": 28_800,
                    "start_sample": 3_200,
                    "end_sample_exclusive": 9_600,
                    "source_start_sample": 0,
                    "source_end_sample_exclusive": 6_400,
                    "linear_gain": 0.2,
                    "fade_samples": 80,
                    "normalization_policy": "use_sound_asset_policy",
                    "render_source_stem": True,
                }
            ],
            "source_specific_stems": True,
            "admission_state": "research",
        }
    )


def _endpoint_to_source_slot() -> dict[str, str]:
    return {
        "dog_program_endpoint": "source1",
        "cat_program_endpoint": "source2",
    }


def _event_sound_classes() -> dict[str, str]:
    return {"dog_bark_0": "animal_vocalization"}


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
    assert all(
        set(frame["view_pose_hashes"]) == {"view0"}
        for frame in timeline["frames"]
    )
    assert len(
        {
            frame["view_pose_hashes"]["view0"]
            for frame in timeline["frames"]
        }
    ) == 1
    assert headings["source1"].shape == (75, 2)
    assert headings["source2"].shape == (75, 2)
    np.testing.assert_allclose(np.linalg.norm(headings["source1"], axis=1), 1.0)
    assert timeline["audio_events"] == [
        {
            "event_id": "source1_full_duration_vocalization",
            "actor_id": "source1_actor",
            "emitter_bone": "muzzle",
            "event_type": "vocalization",
            "start_sample": 0,
            "end_sample": 80_000,
            "semantic_sync_required": False,
        },
        {
            "event_id": "source2_full_duration_vocalization",
            "actor_id": "source2_actor",
            "emitter_bone": "muzzle",
            "event_type": "vocalization",
            "start_sample": 0,
            "end_sample": 80_000,
            "semantic_sync_required": False,
        },
    ]
    assert all(
        not state["mouth_state"]["vocalizing"]
        for frame in timeline["frames"]
        for state in frame["actor_states"]
    )


def test_timeline_binds_complete_moving_sensor_rig_per_frame() -> None:
    trajectory = materialize_sensor_rig_trajectory(
        trajectory_id="m7_rotate_review_v1",
        program={
            "kind": "ROTATE_IN_PLACE",
            "position_m": [-0.7, 1.47, 0.65],
            "start_yaw_deg": -45.0,
            "end_yaw_deg": 45.0,
            "yaw_interpolation": "SHORTEST_ARC",
        },
    )

    timeline, _ = build_timeline(
        episode=_episode(),
        bindings=_bindings(),
        listener_position_m=(99.0, 99.0, 99.0),
        listener_yaw_deg=12.0,
        sensor_rig_trajectory=trajectory,
    )

    assert [
        frame["view_pose_hashes"]["view0"]
        for frame in timeline["frames"]
    ] == [frame["pose_hash"] for frame in trajectory["frames"]]
    assert (
        timeline["frames"][0]["view_pose_hashes"]["view0"]
        != timeline["frames"][-1]["view_pose_hashes"]["view0"]
    )


def test_fixed_listener_fallback_materializes_a_complete_hold() -> None:
    trajectory = resolve_m7_sensor_rig_trajectory(
        sensor_rig_trajectory=None,
        listener_position_m=(-0.7, 1.47, 0.65),
        listener_yaw_deg=55.0,
    )

    assert trajectory["schema"] == "avengine_sensor_rig_trajectory_v1"
    assert trajectory["program"] == {
        "kind": "HOLD",
        "position_m": [-0.7, 1.47, 0.65],
        "yaw_deg": 55.0,
    }
    assert len(trajectory["frames"]) == 75
    assert len({frame["pose_hash"] for frame in trajectory["frames"]}) == 1


def test_m7_rejects_a_tampered_sensor_rig_sidecar() -> None:
    trajectory = resolve_m7_sensor_rig_trajectory(
        sensor_rig_trajectory=None,
        listener_position_m=(-0.7, 1.47, 0.65),
        listener_yaw_deg=55.0,
    )
    trajectory["frames"][12]["pose_hash"] = "0" * 64

    with pytest.raises(
        ApartmentVisualBundleError,
        match="pose_hash does not bind world_from_rig",
    ):
        build_timeline(
            episode=_episode(),
            bindings=_bindings(),
            listener_position_m=(-0.7, 1.47, 0.65),
            listener_yaw_deg=55.0,
            sensor_rig_trajectory=trajectory,
        )


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
    assert manifest["purpose"] == "two_replaceable_sources_both_active"
    assert [source["activation"] for source in manifest["sources"]] == [
        "active",
        "active",
    ]


def test_source_manifest_hash_binds_the_retained_sensor_rig_sidecar() -> None:
    trajectory = resolve_m7_sensor_rig_trajectory(
        sensor_rig_trajectory=None,
        listener_position_m=(-0.7, 1.47, 0.65),
        listener_yaw_deg=55.0,
    )
    manifest = build_source_manifest(
        episode_id=_episode()["episode_id"],
        episode=_episode(),
        bindings=_bindings(),
        sensor_rig_trajectory=trajectory,
    )

    assert manifest["listener"]["sensor_rig_trajectory"] == {
        "trajectory_id": trajectory["trajectory_id"],
        "content_sha256": canonical_json_sha256(trajectory),
        "relative_path": "metadata/sensor_rig_trajectory.json",
    }


def test_materialized_program_drives_all_visual_audio_labels() -> None:
    program = _materialized_audio_program()
    mapping = _endpoint_to_source_slot()
    activity = program_source_activity_by_frame(program, mapping)
    assert set(activity) == {"source1", "source2"}
    assert all(values.shape == (75,) for values in activity.values())
    assert all(values.dtype == np.bool_ for values in activity.values())
    assert not activity["source1"][2]
    assert activity["source1"][3]
    assert activity["source1"][8]
    assert not activity["source1"][9]
    assert not np.any(activity["source2"])

    timeline, _ = build_timeline(
        episode=_episode(),
        bindings=_bindings(),
        listener_position_m=(-0.7, 1.47, 0.65),
        materialized_audio_program=program,
        endpoint_to_source_slot=mapping,
        semantic_sound_class_by_event_id=_event_sound_classes(),
    )
    assert timeline["audio_events"] == [
        {
            "event_id": "dog_bark_0",
            "actor_id": "source1_actor",
            "emitter_bone": "muzzle",
            "event_type": "vocalization",
            "start_sample": 3_200,
            "end_sample": 9_600,
            "semantic_sync_required": True,
        }
    ]
    for frame_index, frame in enumerate(timeline["frames"]):
        states = {state["actor_id"]: state for state in frame["actor_states"]}
        assert (
            states["source1_actor"]["mouth_state"]["vocalizing"]
            == bool(activity["source1"][frame_index])
        )
        assert (
            states["source2_actor"]["mouth_state"]["vocalizing"]
            == bool(activity["source2"][frame_index])
        )

    manifest = build_source_manifest(
        episode_id=_episode()["episode_id"],
        episode=_episode(),
        bindings=_bindings(),
        materialized_audio_program=program,
        endpoint_to_source_slot=mapping,
        audio_program_variant_id="A",
        semantic_sound_class_by_event_id=_event_sound_classes(),
    )
    assert manifest["purpose"] == "two_replaceable_sources_one_active_of_n"
    assert [source["source_endpoint_id"] for source in manifest["sources"]] == [
        "source1_emitter",
        "source2_emitter",
    ]
    assert [source["activation"] for source in manifest["sources"]] == [
        "active",
        "persistent_silent",
    ]
    assert manifest["events"] == [
        {
            **program["events"][0],
            "source_endpoint_id": "source1_emitter",
            "audio_program_source_endpoint_id": "dog_program_endpoint",
            "semantic_sound_class": "animal_vocalization",
            "event_type": "vocalization",
        }
    ]
    assert manifest["audio_program"] == {
        "program_id": "m7_visual_projection_v1",
        "revision": "v1",
        "mode": "one_active_of_n",
        "variant_id": "A",
        "program_content_sha256": program["program_content_sha256"],
        "endpoint_to_source_slot": mapping,
    }


def test_directional_chime_is_not_vocalization_or_mouth_activity() -> None:
    program = deepcopy(_materialized_audio_program())
    event = program["events"][0]
    event["event_id"] = "directional_chime_0"
    event["sound_asset_id"] = "directional_chime_v1"
    program = bind_audio_program_hash(program)

    timeline, _ = build_timeline(
        episode=_episode(),
        bindings=_bindings(),
        listener_position_m=(-0.7, 1.47, 0.65),
        materialized_audio_program=program,
        endpoint_to_source_slot=_endpoint_to_source_slot(),
        semantic_sound_class_by_event_id={"directional_chime_0": "test_signal"},
    )

    assert timeline["audio_events"][0]["event_type"] == "other"
    assert all(
        not state["mouth_state"]["vocalizing"]
        for frame in timeline["frames"]
        for state in frame["actor_states"]
    )


def test_counterfactual_b_variant_is_preserved_in_source_manifest() -> None:
    repository = Path(__file__).resolve().parents[2]
    base_program = json.loads(
        (
            repository
            / "examples/m6x/fixed_apartment/audio_programs"
            / "m6x_s1_front_rear_route_swap_v1.json"
        ).read_text(encoding="utf-8")
    )
    program = materialize_audio_program_variant(base_program, "B")
    mapping = {
        "m6x_marker_front_speaker": "source1",
        "m6x_marker_rear_speaker": "source2",
    }
    manifest = build_source_manifest(
        episode_id=_episode()["episode_id"],
        episode=_episode(),
        bindings=_bindings(),
        materialized_audio_program=program,
        endpoint_to_source_slot=mapping,
        audio_program_variant_id="B",
        semantic_sound_class_by_event_id={"s1_directional_chime": "test_signal"},
    )

    assert manifest["variant_id"] == "B"
    assert manifest["audio_program"]["variant_id"] == "B"


def test_program_projection_requires_both_optional_inputs() -> None:
    with pytest.raises(ApartmentVisualBundleError, match="provided together"):
        build_timeline(
            episode=_episode(),
            bindings=_bindings(),
            listener_position_m=(-0.7, 1.47, 0.65),
            materialized_audio_program=_materialized_audio_program(),
        )


@pytest.mark.parametrize(
    "mapping",
    [
        {"dog_program_endpoint": "source1"},
        {
            "dog_program_endpoint": "source1",
            "cat_program_endpoint": "source1",
        },
        {
            "dog_program_endpoint": "source1",
            "unknown_program_endpoint": "source2",
        },
    ],
)
def test_program_projection_requires_exact_endpoint_slot_bijection(
    mapping: dict[str, str],
) -> None:
    with pytest.raises(ApartmentVisualBundleError, match="bijection"):
        program_source_activity_by_frame(_materialized_audio_program(), mapping)


def test_program_projection_rejects_non_m7_duration() -> None:
    program = deepcopy(_materialized_audio_program())
    program["timeline"]["frame_count"] = 76
    program["timeline"]["sample_count"] = 81_067
    program = bind_audio_program_hash(program)
    with pytest.raises(
        ApartmentVisualBundleError, match="exactly 75 frames and 80000 samples"
    ):
        program_source_activity_by_frame(program, _endpoint_to_source_slot())


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
    trajectory = resolve_m7_sensor_rig_trajectory(
        sensor_rig_trajectory=None,
        listener_position_m=(0.0, 1.5, 0.0),
        listener_yaw_deg=0.0,
    )
    sensor_rig_binding = _BUILDER.m7_sensor_rig_binding(trajectory)
    for name in (
        "timeline.json",
        "source_manifest.json",
        "flags.json",
    ):
        (metadata / name).write_text("{}", encoding="utf-8")
    sensor_rig_path = metadata / "sensor_rig_trajectory.json"
    sensor_rig_path.write_text(
        json.dumps(trajectory), encoding="utf-8"
    )
    (metadata / "batch_binding.json").write_text(
        json.dumps(
            {
                "sensor_rig_trajectory": sensor_rig_binding,
                "acoustic_selection_binding_sha256": None,
            }
        ),
        encoding="utf-8",
    )
    row = {
        "episode_ordinal": 0,
        "episode_id": episode_id,
        "v00_sample_id": "sample_0001",
        "acoustic_selection_binding_sha256": None,
    }
    (metadata / "build_record.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "diagnostic_sha256": sha256_file(diagnostic),
                "sensor_rig_trajectory_file_sha256": sha256_file(
                    sensor_rig_path
                ),
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
        batch_root=tmp_path,
        runtime_bindings_by_source_slot={},
        sensor_rig_binding=sensor_rig_binding,
    )
    assert reopened == {
        "episode_ordinal": 17,
        "episode_id": episode_id,
        "v00_sample_id": "sample_0001",
        "acoustic_selection_binding_sha256": None,
    }

    diagnostic.write_bytes(b"changed media")
    with pytest.raises(RuntimeError, match="completed episode changed"):
        _BUILDER._load_completed_episode(
            staging=tmp_path,
            episode_id=episode_id,
            ordinal=17,
            sample={"sample_id": "sample_0001"},
            batch_root=tmp_path,
            runtime_bindings_by_source_slot={},
            sensor_rig_binding=sensor_rig_binding,
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


def test_ue_input_cross_checks_acoustic_and_visual_room_refs() -> None:
    visual_room_ref = {
        "registry_id": "avengine_m6_representative_rooms_v1",
        "room_id": "legacy_ue_apartment_0000_v1",
        "revision": "real_surface_export_pending_portable_package_v1",
    }
    binding = {
        "selection_mode": "registry",
        "room_ref": dict(visual_room_ref),
    }

    aligned = _BUILDER._acoustic_visual_room_alignment(
        binding,
        visual_room_ref,
    )
    assert aligned["status"] == "pass"
    assert aligned["visual_room_ref"] == visual_room_ref

    binding["room_ref"]["room_id"] = "wrong_room"
    with pytest.raises(RuntimeError, match="visual room_ref differs"):
        _BUILDER._acoustic_visual_room_alignment(
            binding,
            visual_room_ref,
        )


def test_ue_input_keeps_legacy_unbound_room_compatibility_explicit() -> None:
    result = _BUILDER._acoustic_visual_room_alignment(
        {
            "selection_mode": "explicit_legacy_unbound",
            "room_ref": None,
        },
        {
            "registry_id": "rooms",
            "room_id": "visual_room",
            "revision": "v1",
        },
    )

    assert result["status"] == "not_verified"
    assert result["compatibility"] == (
        "legacy_acoustic_selection_without_room_ref"
    )
    assert result["acoustic_room_ref"] is None


def test_ue_input_loads_sample_audio_program(
    tmp_path: Path,
) -> None:
    program = _materialized_audio_program()
    mapping = _endpoint_to_source_slot()
    binding = {
        "audio_program_ref": {
            "program_id": program["program_id"],
            "revision": program["revision"],
            "program_content_sha256": program["program_content_sha256"],
        },
        "variant_id": "A",
        "materialized_program_content_sha256": program[
            "program_content_sha256"
        ],
        "source_endpoint_to_source_slot": mapping,
        "dry_audio_assembly_content_sha256": "a" * 64,
    }
    relative = Path("labels/audio_program_instances/v00.json")
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    instance = {
        "schema": "avengine_m7_m6_audio_program_instance_v1",
        "status": "pass",
        "audio_program_binding": binding,
        "materialized_audio_program": program,
        "sound_asset_semantics": {
            "dog_beagle_v2_scheduled_dry": "animal_vocalization",
        },
        "mapped_events": [
            {
                **program["events"][0],
                "source_slot_id": "source1",
                "semantic_sound_class": "animal_vocalization",
            }
        ],
    }
    path.write_text(json.dumps(instance), encoding="utf-8")
    sample = {
        "audio_program_binding": binding,
        "audio_program_instance_path": relative.as_posix(),
        "audio_program_instance_sha256": sha256_file(path),
    }

    (
        observed_program,
        observed_mapping,
        observed_instance,
        observed_variant_id,
        observed_event_semantics,
    ) = (
        _BUILDER._sample_audio_program_projection(
            sample=sample,
            batch_root=tmp_path,
        )
    )

    assert observed_program == program
    assert observed_mapping == mapping
    assert observed_instance == instance
    assert observed_variant_id == "A"
    assert observed_event_semantics == _event_sound_classes()

    sample["audio_program_instance_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="missing or changed"):
        _BUILDER._sample_audio_program_projection(
            sample=sample,
            batch_root=tmp_path,
        )

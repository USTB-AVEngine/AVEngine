from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from avengine.contracts.json_io import canonical_json_sha256, write_json
from avengine.routes.room_feasibility import rir_acoustic_state_sha256
from avengine.dataset.asset_bound_audio import AssetBoundAudioError
from avengine.dataset.sensor_rig import m7_sensor_rig_pose_series
from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory
from tools.dataset.render_asset_bound_binaural_batch import (
    AudioProgramSpec,
    _load_sensor_rig_contract,
    _prepare_audio_program_variants,
    _validated_acoustic_selection_binding,
    audio_program_specs,
)


ROOT = Path(__file__).resolve().parents[2]
PROGRAMS = ROOT / "examples/routes/fixed_apartment/audio_programs"
REGISTRIES = ROOT / "examples/registry/registries"


def test_audio_batch_authenticates_binding_and_keeps_legacy_unbound_explicit() -> None:
    value = {
        "schema": "avengine_rir_cache_acoustic_selection_binding_v1",
        "selection_mode": "explicit_legacy",
        "registry_selection_applied": False,
        "room_ref": None,
        "profile_ref": None,
        "binding_id": None,
        "registry_selection_content_sha256": None,
        "effective_selection_content_sha256": None,
        "acoustic_package_manifest_sha256": "1" * 64,
        "simulation_request_sha256": "2" * 64,
        "input_receipt_sha256": None,
    }
    value["binding_content_sha256"] = canonical_json_sha256(value)
    binding, binding_sha256 = _validated_acoustic_selection_binding(value)
    assert binding == value
    assert binding_sha256 == value["binding_content_sha256"]

    unbound = {
        "schema": "avengine_rir_cache_acoustic_selection_binding_v1",
        "selection_mode": "explicit_legacy_unbound",
        "registry_selection_applied": False,
        "room_ref": None,
        "profile_ref": None,
        "binding_id": None,
        "registry_selection_content_sha256": None,
        "effective_selection_content_sha256": None,
        "acoustic_package_manifest_sha256": None,
        "simulation_request_sha256": None,
        "input_receipt_sha256": None,
        "binding_content_sha256": None,
    }
    assert _validated_acoustic_selection_binding(unbound) == (unbound, None)

    value["binding_id"] = "forged_after_hash"
    with pytest.raises(AssetBoundAudioError, match="hash is invalid"):
        _validated_acoustic_selection_binding(value)


def _write_dynamic_sensor_rig_plan(root: Path) -> tuple[dict, dict]:
    trajectory = materialize_sensor_rig_trajectory(
        trajectory_id="unit_dynamic_rig",
        program={
            "kind": "ROTATE_IN_PLACE",
            "position_m": [0.0, 1.5, 0.0],
            "start_yaw_deg": 0.0,
            "end_yaw_deg": 90.0,
            "yaw_interpolation": "SHORTEST_ARC",
        },
    )
    poses = m7_sensor_rig_pose_series(trajectory)
    jobs = []
    for frame_index in (0, 3):
        for source_ordinal, source_slot in enumerate(("source1", "source2")):
            source_position = [float(source_ordinal + 1), 1.0, 2.0]
            listener_position = poses.positions_m[frame_index].tolist()
            listener_orientation = poses.orientations_wxyz[
                frame_index
            ].tolist()
            jobs.append(
                {
                    "job_id": f"job_{frame_index}_{source_slot}",
                    "source_position_m": source_position,
                    "listener_position_m": listener_position,
                    "listener_orientation_wxyz": listener_orientation,
                    "acoustic_state_sha256": rir_acoustic_state_sha256(
                        source_position,
                        listener_position,
                        listener_orientation,
                    ),
                    "uses": [
                        {
                            "episode_id": "episode0",
                            "source_slot_id": source_slot,
                            "frame_index": frame_index,
                        }
                    ],
                }
            )
    plan = {
        "schema": "avengine_room_rir_job_plan_v2",
        "status": "planned_not_run",
        "listener_pose_mode": "per_episode_frame",
        "cache_key_fields": [
            "source_position_m",
            "listener_position_m",
            "listener_orientation_wxyz",
        ],
        "requested_pair_state_count": len(jobs),
        "unique_rir_job_count": len(jobs),
        "jobs": jobs,
    }
    root.mkdir(parents=True)
    write_json(root / "sensor_rig_trajectory.json", trajectory)
    write_json(root / "rir_job_plan.json", plan)
    return trajectory, plan


def test_asset_bound_batch_loads_and_aligns_dynamic_sensor_rig(
    tmp_path: Path,
) -> None:
    plan_root = tmp_path / "plan"
    trajectory, _plan = _write_dynamic_sensor_rig_plan(plan_root)

    contract = _load_sensor_rig_contract(plan_root)
    assert contract is not None
    assert contract["binding"]["trajectory_id"] == trajectory["trajectory_id"]
    assert contract["binding"]["dynamic"] is True
    assert contract["rir_alignment"] == {
        "listener_pose_mode": "per_episode_frame",
        "checked_use_count": 4,
        "acoustic_state_binding": "source_listener_pose_per_job_v1",
    }


def test_asset_bound_batch_fails_closed_on_missing_or_misaligned_dynamic_rig(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    _trajectory, plan = _write_dynamic_sensor_rig_plan(missing_root)
    (missing_root / "sensor_rig_trajectory.json").unlink()
    with pytest.raises(AssetBoundAudioError, match="requires"):
        _load_sensor_rig_contract(missing_root)

    mismatch_root = tmp_path / "mismatch"
    _write_dynamic_sensor_rig_plan(mismatch_root)
    plan["jobs"][0]["listener_position_m"][0] += 1.0
    write_json(mismatch_root / "rir_job_plan.json", plan)
    with pytest.raises(AssetBoundAudioError, match="alignment"):
        _load_sensor_rig_contract(mismatch_root)

    fixed_root = tmp_path / "fixed"
    fixed_root.mkdir()
    write_json(
        fixed_root / "rir_job_plan.json",
        {
            "schema": "avengine_room_rir_job_plan_v2",
            "status": "planned_not_run",
            "listener_position_m": [0.0, 1.5, 0.0],
            "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "unique_rir_job_count": 1,
            "jobs": [],
        },
    )
    assert _load_sensor_rig_contract(fixed_root) is None


def test_program_specs_default_to_a_and_require_aligned_variants() -> None:
    paths = (Path("first.json"), Path("second.json"))

    assert [item.variant_id for item in audio_program_specs(paths, ())] == [
        "A",
        "A",
    ]
    assert [item.variant_id for item in audio_program_specs(paths, ("A", "B"))] == [
        "A",
        "B",
    ]
    with pytest.raises(AssetBoundAudioError, match="count"):
        audio_program_specs(paths, ("A",))


def test_dataset_prepares_sequential_m6_program_as_exact_slot_buses() -> None:
    prepared, library = _prepare_audio_program_variants(
        specs=(
            AudioProgramSpec(
                PROGRAMS / "m6x_s5_los_nlos_sequential_v1.json"
            ),
        ),
        source_endpoint_registry_path=REGISTRIES / "source_endpoints_v1.json",
        sound_asset_registry_path=REGISTRIES / "sound_assets_v1.json",
        endpoint_to_source_slot={
            "m6x_world_los_speaker": "source1",
            "m6x_world_nlos_speaker": "source2",
        },
        sound_audio={
            "directional_chime_v1": str(
                ROOT / "examples/routes/assets/directional_chime_16k.wav"
            ),
            "unused_library_entry": "/not-read.wav",
        },
    )

    item = prepared[0]
    source1 = item.dry_by_source_slot["source1"]
    source2 = item.dry_by_source_slot["source2"]
    assert not np.any(source1[:4_000])
    assert np.any(source1[4_080:31_920])
    assert not np.any(source1[32_000:])
    assert not np.any(source2[:44_000])
    assert np.any(source2[44_080:71_920])
    assert not np.any(source2[72_000:])
    assert item.source_activity_summary == {
        "active_source_slots": ["source1", "source2"],
        "silent_source_slots": [],
        "active_sample_count_by_source_slot": {
            "source1": 28_000,
            "source2": 28_000,
        },
        "simultaneous_active_sample_count": 0,
        "both_sources_have_events": True,
        "both_sources_active": False,
    }
    assert item.audio_program_binding[
        "source_endpoint_to_source_slot"
    ] == {
        "m6x_world_los_speaker": "source1",
        "m6x_world_nlos_speaker": "source2",
    }
    assert (
        item.instance_record["dry_audio_assembly"][
            "assembly_content_sha256"
        ]
        == item.audio_program_binding[
            "dry_audio_assembly_content_sha256"
        ]
    )
    assert library["schema"] == "avengine_m7_m6_audio_program_dry_bus_library_v1"


def test_dataset_prepares_counterfactual_b_as_exact_endpoint_and_slot_bus_swap() -> None:
    prepared, _library = _prepare_audio_program_variants(
        specs=(
            AudioProgramSpec(
                PROGRAMS / "m6x_s1_front_rear_route_swap_v1.json",
                variant_id="A",
            ),
            AudioProgramSpec(
                PROGRAMS / "m6x_s1_front_rear_route_swap_v1.json",
                variant_id="B",
            ),
        ),
        source_endpoint_registry_path=REGISTRIES / "source_endpoints_v1.json",
        sound_asset_registry_path=REGISTRIES / "sound_assets_v1.json",
        endpoint_to_source_slot={
            "m6x_marker_front_speaker": "source1",
            "m6x_marker_rear_speaker": "source2",
        },
        sound_audio={
            "directional_chime_v1": str(
                ROOT / "examples/routes/assets/directional_chime_16k.wav"
            )
        },
    )

    variant_a = prepared[0]
    variant_b = prepared[1]
    event_a = variant_a.instance_record["materialized_audio_program"]["events"][0]
    event_b = variant_b.instance_record["materialized_audio_program"]["events"][0]
    assert event_a["source_endpoint_id"] == "m6x_marker_front_speaker"
    assert event_b["source_endpoint_id"] == "m6x_marker_rear_speaker"
    assert variant_a.audio_program_binding["variant_id"] == "A"
    assert variant_b.audio_program_binding["variant_id"] == "B"
    assert np.array_equal(
        variant_a.dry_by_source_slot["source1"],
        variant_b.dry_by_source_slot["source2"],
    )
    assert np.array_equal(
        variant_a.dry_by_source_slot["source2"],
        variant_b.dry_by_source_slot["source1"],
    )


def test_dataset_prepares_silent_negative_as_two_exact_zero_slot_buses() -> None:
    prepared, _library = _prepare_audio_program_variants(
        specs=(
            AudioProgramSpec(
                PROGRAMS / "m6x_s2_silent_negative_v1.json"
            ),
        ),
        source_endpoint_registry_path=REGISTRIES / "source_endpoints_v1.json",
        sound_asset_registry_path=REGISTRIES / "sound_assets_v1.json",
        endpoint_to_source_slot={
            "m6x_dog0_muzzle": "source1",
            "m6x_human0_mouth": "source2",
        },
        sound_audio={},
    )

    item = prepared[0]
    assert all(not np.any(bus) for bus in item.dry_by_source_slot.values())
    assert item.source_activity_summary["active_source_slots"] == []
    assert item.source_activity_summary["silent_source_slots"] == [
        "source1",
        "source2",
    ]
    assert item.source_activity_summary["both_sources_active"] is False

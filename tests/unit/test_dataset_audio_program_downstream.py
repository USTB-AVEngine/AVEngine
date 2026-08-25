from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)
from avengine.timeline.audio_program import (
    bind_audio_program_hash,
    materialize_audio_program_variant,
)
from avengine.routes.room_feasibility import rir_acoustic_state_sha256
from avengine.dataset.sensor_rig import (
    m7_sensor_rig_binding,
    m7_sensor_rig_pose_series,
    validate_m7_rir_listener_alignment,
)
from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory


_ROOT = Path(__file__).resolve().parents[2]


def _load_tool(name: str, relative_path: str):
    path = _ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VERIFY = _load_tool(
    "verify_asset_bound_batch_for_test",
    "tools/dataset/verify_asset_bound_batch.py",
)
_INDEX = _load_tool(
    "build_asset_bound_dataset_index_for_test",
    "tools/dataset/build_asset_bound_dataset_index.py",
)


def _registry_acoustic_selection_binding() -> dict:
    value = {
        "schema": "avengine_rir_cache_acoustic_selection_binding_v1",
        "selection_mode": "registry",
        "registry_selection_applied": True,
        "room_ref": {
            "registry_id": "avengine_m6_representative_rooms_v1",
            "room_id": "legacy_ue_apartment_0000_v1",
            "revision": "real_surface_export_pending_portable_package_v1",
        },
        "profile_ref": {
            "profile_id": "legacy_controlled_approximation_pending_v1",
            "revision": "spear_ue_authored_residential_rules_v1",
        },
        "binding_id": "legacy_ue_apartment_0000_authored_v1",
        "registry_selection_content_sha256": "1" * 64,
        "effective_selection_content_sha256": "2" * 64,
        "acoustic_package_manifest_sha256": "3" * 64,
        "simulation_request_sha256": "4" * 64,
        "input_receipt_sha256": "5" * 64,
    }
    value["binding_content_sha256"] = canonical_json_sha256(value)
    return value


def _legacy_unbound_acoustic_selection_binding() -> dict:
    return {
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


def _spear_runtime_identity(
    acoustic_binding: dict,
    *,
    visual_room_ref: dict | None = None,
) -> dict:
    mode = acoustic_binding["selection_mode"]
    is_registry = mode in {
        "registry",
        "registry_with_verified_equivalent_overrides",
    }
    if visual_room_ref is None:
        visual_room_ref = (
            acoustic_binding["room_ref"]
            if is_registry
            else {
                "registry_id": "avengine_m6_representative_rooms_v1",
                "room_id": "legacy_ue_apartment_0000_v1",
                "revision": (
                    "real_surface_export_pending_portable_package_v1"
                ),
            }
        )
    return {
        "schema": "avengine_spear_acoustic_visual_runtime_identity_v1",
        "status": "pass" if is_registry else "not_verified",
        "verification_status": (
            "verified" if is_registry else "not_verified"
        ),
        "selection_mode": mode,
        "compatibility": (
            None
            if is_registry
            else "legacy_acoustic_selection_without_room_ref"
        ),
        "acoustic_selection_binding_sha256": acoustic_binding[
            "binding_content_sha256"
        ],
        "binding_id": acoustic_binding["binding_id"],
        "profile_ref": acoustic_binding["profile_ref"],
        "visual_room_ref": visual_room_ref,
        "acoustic_room_ref": acoustic_binding["room_ref"],
        "runtime_room_ref": visual_room_ref,
        "runtime_profile_id": "apartment_native_default",
        "runtime_map_id": "apartment_0000",
        "runtime_map_path": "/Game/Scenes/Apartment/Maps/apartment_0000",
    }


def _write_spear_runtime_evidence(
    path: Path,
    *,
    identity: dict,
    scenario_ids: tuple[str, ...] = ("episode_0000",),
) -> Path:
    write_json(
        path,
        {
            "schema": "avengine_optional_spear_apartment_runtime_evidence_v2",
            "status": "pass",
            "native_map": identity["runtime_map_path"],
            "room_runtime_profile": {
                "profile_id": identity["runtime_profile_id"],
            },
            "acoustic_visual_identity": identity,
            "scenarios": [
                {
                    "status": "pass",
                    "scenario_id": scenario_id,
                    "acoustic_visual_identity": identity,
                }
                for scenario_id in scenario_ids
            ],
        },
    )
    return path


def _sensor_rig_closure_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict, dict]:
    plan_root = tmp_path / "plan"
    batch_root = tmp_path / "batch"
    (batch_root / "labels").mkdir(parents=True)
    plan_root.mkdir()
    trajectory = materialize_sensor_rig_trajectory(
        trajectory_id="downstream_dynamic_rig",
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
    cached_jobs = []
    for frame_index in (0, 3):
        for source_ordinal, source_slot in enumerate(("source1", "source2")):
            source_position = [float(source_ordinal + 1), 1.0, 2.0]
            listener_position = poses.positions_m[frame_index].tolist()
            listener_orientation = poses.orientations_wxyz[
                frame_index
            ].tolist()
            state_sha256 = rir_acoustic_state_sha256(
                source_position,
                listener_position,
                listener_orientation,
            )
            job_id = f"job_{frame_index}_{source_slot}"
            jobs.append(
                {
                    "job_id": job_id,
                    "source_position_m": source_position,
                    "listener_position_m": listener_position,
                    "listener_orientation_wxyz": listener_orientation,
                    "acoustic_state_sha256": state_sha256,
                    "uses": [
                        {
                            "episode_id": "episode0",
                            "source_slot_id": source_slot,
                            "frame_index": frame_index,
                        }
                    ],
                }
            )
            cached_jobs.append(
                {
                    "job_id": job_id,
                    "source_slot_id": source_slot,
                    "visual_frame_index": frame_index,
                    "source_position_m": source_position,
                    "listener_position_m": listener_position,
                    "listener_orientation_wxyz": listener_orientation,
                    "acoustic_state_sha256": state_sha256,
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
    binding = m7_sensor_rig_binding(trajectory)
    alignment = validate_m7_rir_listener_alignment(
        rir_job_plan=plan,
        sensor_rig_trajectory=trajectory,
    )
    write_json(plan_root / "rir_job_plan.json", plan)
    write_json(plan_root / "sensor_rig_trajectory.json", trajectory)
    write_json(
        batch_root / "labels/sensor_rig_trajectory.json",
        trajectory,
    )
    write_json(
        batch_root / "delivery.json",
        {
            "sensor_rig_trajectory": binding,
            "sensor_rig_rir_alignment": alignment,
            "outputs": {
                "sensor_rig_trajectory": "labels/sensor_rig_trajectory.json"
            },
        },
    )
    write_json(
        batch_root / "samples.json",
        {
            "samples": [
                {
                    "sample_id": "episode0__v00",
                    "sensor_rig_trajectory": binding,
                }
            ]
        },
    )
    write_json(
        batch_root / "episodes.json",
        {
            "episodes": [
                {
                    "episode_id": "episode0",
                    "sensor_rig_trajectory": binding,
                    "rir_cache": {
                        "acoustic_state_binding": (
                            "source_listener_pose_per_job_v1"
                        ),
                        "jobs": cached_jobs,
                    },
                }
            ]
        },
    )
    return plan_root, batch_root, trajectory, binding


def test_batch_verifier_closes_dynamic_sensor_rig_across_all_records(
    tmp_path: Path,
) -> None:
    plan_root, batch_root, _trajectory, binding = (
        _sensor_rig_closure_fixture(tmp_path)
    )
    result = _VERIFY._verify_sensor_rig_closure(
        plan_root=plan_root,
        batch_root=batch_root,
    )
    assert result["listener_pose_mode"] == "per_episode_frame"
    assert result["binding"] == binding
    assert result["checked_cached_use_count"] == 4


@pytest.mark.parametrize(
    "mutation",
    (
        "batch_sidecar",
        "delivery_binding",
        "sample_binding",
        "episode_binding",
        "cached_listener",
        "cached_state",
        "missing_plan_sidecar",
    ),
)
def test_batch_verifier_rejects_dynamic_sensor_rig_tampering(
    tmp_path: Path,
    mutation: str,
) -> None:
    plan_root, batch_root, trajectory, _binding = (
        _sensor_rig_closure_fixture(tmp_path)
    )
    if mutation == "batch_sidecar":
        trajectory["trajectory_id"] = "forged"
        write_json(
            batch_root / "labels/sensor_rig_trajectory.json",
            trajectory,
        )
    elif mutation == "delivery_binding":
        value = load_json(batch_root / "delivery.json")
        value["sensor_rig_trajectory"]["content_sha256"] = "a" * 64
        write_json(batch_root / "delivery.json", value)
    elif mutation == "sample_binding":
        value = load_json(batch_root / "samples.json")
        value["samples"][0]["sensor_rig_trajectory"]["content_sha256"] = (
            "a" * 64
        )
        write_json(batch_root / "samples.json", value)
    elif mutation == "episode_binding":
        value = load_json(batch_root / "episodes.json")
        value["episodes"][0]["sensor_rig_trajectory"][
            "content_sha256"
        ] = "a" * 64
        write_json(batch_root / "episodes.json", value)
    elif mutation == "cached_listener":
        value = load_json(batch_root / "episodes.json")
        value["episodes"][0]["rir_cache"]["jobs"][0][
            "listener_position_m"
        ][0] += 1.0
        write_json(batch_root / "episodes.json", value)
    elif mutation == "cached_state":
        value = load_json(batch_root / "episodes.json")
        value["episodes"][0]["rir_cache"]["jobs"][0][
            "acoustic_state_sha256"
        ] = "a" * 64
        write_json(batch_root / "episodes.json", value)
    elif mutation == "missing_plan_sidecar":
        (plan_root / "sensor_rig_trajectory.json").unlink()
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(_VERIFY.BatchVerificationError):
        _VERIFY._verify_sensor_rig_closure(
            plan_root=plan_root,
            batch_root=batch_root,
        )


def test_batch_verifier_keeps_legacy_fixed_plan_without_sidecar(
    tmp_path: Path,
) -> None:
    plan_root = tmp_path / "plan"
    batch_root = tmp_path / "batch"
    plan_root.mkdir()
    batch_root.mkdir()
    write_json(
        plan_root / "rir_job_plan.json",
        {
            "schema": "avengine_room_rir_job_plan_v2",
            "status": "planned_not_run",
            "listener_position_m": [0.0, 1.5, 0.0],
            "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "jobs": [],
        },
    )
    write_json(batch_root / "delivery.json", {"outputs": {}})
    write_json(batch_root / "samples.json", {"samples": [{}]})
    write_json(batch_root / "episodes.json", {"episodes": [{}]})

    result = _VERIFY._verify_sensor_rig_closure(
        plan_root=plan_root,
        batch_root=batch_root,
    )
    assert result["binding"] is None
    assert result["compatibility"] == "legacy_fixed_plan_without_sidecar"


def _program_bound_sample(
    tmp_path: Path,
    *,
    program_filename: str = "m6x_s0_routing_sanity_v1.json",
    variant_id: str = "A",
) -> dict:
    base_program = load_json(
        _ROOT
        / "examples/m6x/fixed_apartment/audio_programs"
        / program_filename
    )
    program = materialize_audio_program_variant(base_program, variant_id)
    sound_registry = load_json(
        _ROOT / "examples/m6/registries/sound_assets_v1.json"
    )
    sound_classes = {
        item["sound_asset_id"]: item["semantic_sound_class"]
        for item in sound_registry["sound_assets"]
    }
    used_sound_ids = sorted(
        {event["sound_asset_id"] for event in program["events"]}
    )
    sound_asset_semantics = {
        sound_id: sound_classes[sound_id] for sound_id in used_sound_ids
    }
    endpoint_to_slot = dict(
        zip(
            program["candidate_source_endpoint_ids"],
            ("source1", "source2"),
            strict=True,
        )
    )
    assembly_content = {
        "schema": "avengine_m5_1_dry_audio_assembly_v1",
        "qualification_claim": False,
        "clip": {
            "sample_rate_hz": 16_000,
            "sample_count": 80_000,
        },
        "source_ids": list(program["candidate_source_endpoint_ids"]),
        "arithmetic": {"mode": "unit_test"},
        "placement_receipts": [],
        "bus_float64_le_sha256": {
            endpoint_id: str(index + 1) * 64
            for index, endpoint_id in enumerate(
                program["candidate_source_endpoint_ids"]
            )
        },
    }
    assembly_hash = canonical_json_sha256(assembly_content)
    assembly = {
        **assembly_content,
        "assembly_content_sha256": assembly_hash,
    }
    binding = {
        "audio_program_ref": {
            "program_id": base_program["program_id"],
            "revision": base_program["revision"],
            "program_content_sha256": base_program["program_content_sha256"],
        },
        "variant_id": variant_id,
        "materialized_program_content_sha256": program[
            "program_content_sha256"
        ],
        "source_endpoint_to_source_slot": endpoint_to_slot,
        "dry_audio_assembly_content_sha256": assembly_hash,
    }
    intervals = {source_slot: [] for source_slot in ("source1", "source2")}
    for event in program["events"]:
        intervals[endpoint_to_slot[event["source_endpoint_id"]]].append(
            (event["start_sample"], event["end_sample_exclusive"])
        )
    active_counts = {
        source_slot: sum(end - start for start, end in source_intervals)
        for source_slot, source_intervals in intervals.items()
    }
    active_slots = [
        source_slot for source_slot, count in active_counts.items() if count
    ]
    simultaneous_active_sample_count = sum(
        max(0, min(left_end, right_end) - max(left_start, right_start))
        for left_start, left_end in intervals["source1"]
        for right_start, right_end in intervals["source2"]
    )
    activity_summary = {
        "active_source_slots": active_slots,
        "silent_source_slots": [
            source_slot
            for source_slot in ("source1", "source2")
            if source_slot not in active_slots
        ],
        "active_sample_count_by_source_slot": active_counts,
        "simultaneous_active_sample_count": simultaneous_active_sample_count,
        "both_sources_have_events": len(active_slots) == 2,
        "both_sources_active": simultaneous_active_sample_count > 0,
    }
    relative_path = "labels/audio_program_instances/sample_v00.json"
    instance_path = tmp_path / relative_path
    write_json(
        instance_path,
        {
            "schema": "avengine_m7_m6_audio_program_instance_v1",
            "status": "pass",
            "audio_program_binding": binding,
            "base_audio_program": base_program,
            "materialized_audio_program": program,
            "sound_asset_semantics": sound_asset_semantics,
            "mapped_events": [
                {
                    **event,
                    "source_slot_id": endpoint_to_slot[
                        event["source_endpoint_id"]
                    ],
                    "semantic_sound_class": sound_asset_semantics[
                        event["sound_asset_id"]
                    ],
                }
                for event in program["events"]
            ],
            "source_activity_summary": activity_summary,
            "dry_audio_assembly": assembly,
        },
    )
    return {
        "sample_id": "sample_v00",
        "asset_ids_by_source_slot": {
            "source1": "asset_a",
            "source2": "asset_b",
        },
        "both_sources_active": activity_summary["both_sources_active"],
        "audio_program_binding": binding,
        "audio_program_instance_path": relative_path,
        "audio_program_instance_sha256": sha256_file(instance_path),
    }


def test_audio_program_instance_accepts_one_active_program_and_index_propagates(
    tmp_path: Path,
) -> None:
    sample = _program_bound_sample(tmp_path)

    result = _VERIFY._audio_program_instance(
        batch_root=tmp_path,
        sample=sample,
        sample_id=sample["sample_id"],
        asset_ids_by_source_slot=sample["asset_ids_by_source_slot"],
    )
    assert result is not None
    assert result["mode"] == "one_active_of_n"
    assert result["variant_id"] == "A"
    assert result["both_sources_active"] is False

    fields, label_path = _INDEX._audio_program_index_fields(
        sample,
        audio_batch_root=tmp_path,
    )
    assert fields == {
        "audio_program_binding": sample["audio_program_binding"],
        "audio_program_instance_path": sample["audio_program_instance_path"],
        "audio_program_instance_sha256": sample[
            "audio_program_instance_sha256"
        ],
    }
    assert label_path == sample["audio_program_instance_path"]


def test_audio_program_instance_accepts_counterfactual_b_rebuilt_from_base(
    tmp_path: Path,
) -> None:
    sample = _program_bound_sample(
        tmp_path,
        program_filename="m6x_s1_front_rear_route_swap_v1.json",
        variant_id="B",
    )

    result = _VERIFY._audio_program_instance(
        batch_root=tmp_path,
        sample=sample,
        sample_id=sample["sample_id"],
        asset_ids_by_source_slot=sample["asset_ids_by_source_slot"],
    )

    assert result is not None
    assert result["mode"] == "counterfactual_route_swap"
    assert result["variant_id"] == "B"
    assert result["active_source_slots"] == ["source2"]
    instance = load_json(tmp_path / sample["audio_program_instance_path"])
    assert (
        instance["base_audio_program"]["events"][0]["source_endpoint_id"]
        == "m6x_marker_front_speaker"
    )
    assert (
        instance["materialized_audio_program"]["events"][0][
            "source_endpoint_id"
        ]
        == "m6x_marker_rear_speaker"
    )


def test_audio_program_instance_rejects_rehashed_non_route_b_tampering(
    tmp_path: Path,
) -> None:
    sample = _program_bound_sample(
        tmp_path,
        program_filename="m6x_s1_front_rear_route_swap_v1.json",
        variant_id="B",
    )
    path = tmp_path / sample["audio_program_instance_path"]
    instance = load_json(path)
    materialized = instance["materialized_audio_program"]
    materialized["events"][0]["linear_gain"] = 0.3
    materialized = bind_audio_program_hash(materialized)
    instance["materialized_audio_program"] = materialized
    materialized_hash = materialized["program_content_sha256"]
    instance["audio_program_binding"][
        "materialized_program_content_sha256"
    ] = materialized_hash
    sample["audio_program_binding"][
        "materialized_program_content_sha256"
    ] = materialized_hash
    write_json(path, instance)
    sample["audio_program_instance_sha256"] = sha256_file(path)

    with pytest.raises(
        _VERIFY.BatchVerificationError,
        match="materialized AudioProgram differs from its base variant",
    ):
        _VERIFY._audio_program_instance(
            batch_root=tmp_path,
            sample=sample,
            sample_id=sample["sample_id"],
            asset_ids_by_source_slot=sample["asset_ids_by_source_slot"],
        )


def test_legacy_sample_remains_unmodified() -> None:
    assert (
        _VERIFY._audio_program_instance(
            batch_root=Path("/unused"),
            sample={"both_sources_active": True},
            sample_id="legacy",
            asset_ids_by_source_slot={
                "source1": "asset_a",
                "source2": "asset_b",
            },
        )
        is None
    )
    assert _INDEX._audio_program_index_fields(
        {"both_sources_active": True},
        audio_batch_root=Path("/unused"),
    ) == ({}, None)


def test_dataset_index_requires_runtime_evidence_for_registry_binding() -> None:
    acoustic_binding = _registry_acoustic_selection_binding()
    room_ref = acoustic_binding["room_ref"]

    with pytest.raises(
        _INDEX.ApartmentDatasetIndexError,
        match="registry-bound index requires SPEAR/UE runtime evidence",
    ):
        _INDEX._validated_spear_runtime_evidence(
            evidence_path=None,
            acoustic_selection_binding=acoustic_binding,
            acoustic_selection_binding_sha256=acoustic_binding[
                "binding_content_sha256"
            ],
            visual_room_alignment={
                "status": "pass",
                "visual_room_ref": room_ref,
                "acoustic_room_ref": room_ref,
            },
            episode_ids={"episode_0000"},
        )


@pytest.mark.parametrize(
    "mismatch",
    ("binding_sha256", "acoustic_room_ref", "visual_room_ref"),
)
def test_dataset_index_rejects_spear_runtime_identity_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    acoustic_binding = _registry_acoustic_selection_binding()
    room_ref = acoustic_binding["room_ref"]
    identity = _spear_runtime_identity(acoustic_binding)
    if mismatch == "binding_sha256":
        identity["acoustic_selection_binding_sha256"] = "f" * 64
    elif mismatch == "acoustic_room_ref":
        identity["acoustic_room_ref"] = {
            **room_ref,
            "room_id": "wrong_acoustic_room",
        }
    else:
        identity["visual_room_ref"] = {
            **room_ref,
            "room_id": "wrong_visual_room",
        }
    evidence_path = _write_spear_runtime_evidence(
        tmp_path / f"{mismatch}.json",
        identity=identity,
    )

    with pytest.raises(_INDEX.ApartmentDatasetIndexError):
        _INDEX._validated_spear_runtime_evidence(
            evidence_path=evidence_path,
            acoustic_selection_binding=acoustic_binding,
            acoustic_selection_binding_sha256=acoustic_binding[
                "binding_content_sha256"
            ],
            visual_room_alignment={
                "status": "pass",
                "visual_room_ref": room_ref,
                "acoustic_room_ref": room_ref,
            },
            episode_ids={"episode_0000"},
        )


def test_dataset_index_keeps_legacy_runtime_identity_not_verified(
    tmp_path: Path,
) -> None:
    acoustic_binding = _legacy_unbound_acoustic_selection_binding()
    identity = _spear_runtime_identity(acoustic_binding)
    visual_room_alignment = {
        "status": "not_verified",
        "compatibility": "legacy_acoustic_selection_without_room_ref",
        "visual_room_ref": identity["visual_room_ref"],
        "acoustic_room_ref": None,
    }
    absent = _INDEX._validated_spear_runtime_evidence(
        evidence_path=None,
        acoustic_selection_binding=acoustic_binding,
        acoustic_selection_binding_sha256=None,
        visual_room_alignment=visual_room_alignment,
        episode_ids={"episode_0000"},
    )
    assert absent == {
        "status": "not_verified",
        "verification_status": "not_verified",
        "path": None,
        "sha256": None,
        "schema": None,
        "acoustic_visual_identity": None,
    }

    evidence_path = _write_spear_runtime_evidence(
        tmp_path / "legacy_evidence.json",
        identity=identity,
    )

    result = _INDEX._validated_spear_runtime_evidence(
        evidence_path=evidence_path,
        acoustic_selection_binding=acoustic_binding,
        acoustic_selection_binding_sha256=None,
        visual_room_alignment=visual_room_alignment,
        episode_ids={"episode_0000"},
    )

    assert result["status"] == "not_verified"
    assert result["verification_status"] == "not_verified"
    assert result["acoustic_visual_identity"] == identity


def test_dataset_index_propagates_audio_program_labels_and_keeps_episode_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_root = tmp_path / "audio_batch"
    visual_root = tmp_path / "visual_bundle"
    render_root = tmp_path / "ue_render"
    output = tmp_path / "index"
    binaural = audio_root / "audio/binaural"
    binaural.mkdir(parents=True)
    shared_wav = binaural / "shared.wav"
    shared_sidecar = binaural / "shared.wav.json"
    shared_wav.touch()
    write_json(shared_sidecar, {"schema": "unit_test_audio_sidecar"})
    shared_wav_sha256 = sha256_file(shared_wav)
    shared_sidecar_sha256 = sha256_file(shared_sidecar)
    base = _program_bound_sample(audio_root)
    assets = base["asset_ids_by_source_slot"]
    acoustic_binding = _registry_acoustic_selection_binding()
    acoustic_binding_sha256 = acoustic_binding["binding_content_sha256"]
    runtime_identity = _spear_runtime_identity(acoustic_binding)
    episodes = [
        {
            "episode_id": f"episode_{index:04d}",
            "motion_case": f"motion_{index % 4}",
            "asset_ids_by_source_slot": assets,
            "acoustic_selection_binding_sha256": acoustic_binding_sha256,
        }
        for index in range(1_000)
    ]
    render_evidence = {}
    samples = []
    for episode in episodes:
        episode_id = episode["episode_id"]
        metadata = visual_root / "episodes" / episode_id / "metadata"
        for filename in ("timeline.json", "source_manifest.json", "flags.json"):
            write_json(metadata / filename, {"status": "pass"})
        media_root = render_root / episode_id
        media_root.mkdir(parents=True)
        (media_root / "ue_visual_only.mp4").touch()
        (media_root / "ue_topdown_visual_only.mp4").touch()
        render_evidence[episode_id] = {
            "status": "pass",
            "scenario_id": episode_id,
            "acoustic_visual_identity": runtime_identity,
            "media": {
                "ue_visual_only": {"status": "pass"},
                "ue_topdown_visual_only": {"status": "pass"},
            }
        }
        samples.append(
            {
                **base,
                "sample_id": f"{episode_id}__v00",
                "episode_id": episode_id,
                "variant_index": 0,
                "acoustic_selection_binding_sha256": acoustic_binding_sha256,
                "audio": {
                    "sample_rate_hz": 16_000,
                    "channel_count": 2,
                    "stems_retained": False,
                    "stems": {},
                    "mixture": {
                        "path": "shared.wav",
                        "audio_sha256": shared_wav_sha256,
                        "sidecar_path": "shared.wav.json",
                        "sidecar_sha256": shared_sidecar_sha256,
                    },
                },
            }
        )
    runtime_evidence_path = render_root / "evidence.json"
    write_json(
        runtime_evidence_path,
        {
            "schema": "avengine_optional_spear_apartment_runtime_evidence_v2",
            "status": "pass",
            "native_map": runtime_identity["runtime_map_path"],
            "room_runtime_profile": {
                "profile_id": runtime_identity["runtime_profile_id"],
            },
            "acoustic_visual_identity": runtime_identity,
            "scenarios": list(render_evidence.values()),
        },
    )
    documents = {
        "samples.json": {
            "status": "pass",
            "acoustic_selection_binding": acoustic_binding,
            "sample_count": 1_000,
            "samples": samples,
        },
        "episodes.json": {
            "status": "pass",
            "acoustic_selection_binding": acoustic_binding,
            "episodes": [],
        },
        "delivery.json": {
            "status": "pass",
            "acoustic_selection_binding": acoustic_binding,
            "sample_count": 1_000,
            "episode_count": 1_000,
            "variants_per_episode": 1,
            "both_sources_active": False,
        },
    }
    for filename, document in documents.items():
        write_json(audio_root / filename, document)
    documents["verification.json"] = {
        "status": "pass",
        "acoustic_selection_binding": acoustic_binding,
    }
    write_json(audio_root / "verification.json", documents["verification.json"])
    write_json(
        visual_root / "manifest.json",
        {
            "status": "pass",
            "acoustic_selection_binding": acoustic_binding,
            "acoustic_visual_room_alignment": {
                "status": "pass",
                "visual_room_ref": acoustic_binding["room_ref"],
                "acoustic_room_ref": acoustic_binding["room_ref"],
            },
            "episodes": episodes,
        },
    )
    monkeypatch.setattr(_INDEX, "_visual_episodes", lambda _root: episodes)

    _INDEX.build_index(
        audio_batch_root=audio_root,
        visual_bundle_root=visual_root,
        ue_render_root=render_root,
        output=output,
        spear_runtime_evidence=runtime_evidence_path,
    )

    index = load_json(output / "dataset_index.json")
    assert len(index["samples"]) == 1_000
    assert index["acoustic_selection_binding"] == acoustic_binding
    assert index["room_ref"] == acoustic_binding["room_ref"]
    assert "room_id" not in index
    assert index["runtime_map_id"] == runtime_identity["runtime_map_id"]
    assert index["spear_ue_runtime_evidence"][
        "acoustic_visual_identity"
    ] == runtime_identity
    split_report = load_json(output / "split_report.json")
    assert split_report["room_ref"] == acoustic_binding["room_ref"]
    assert split_report["runtime_map_id"] == runtime_identity["runtime_map_id"]
    assert split_report["visual_episodes"][0][
        "spear_ue_runtime_evidence_identity"
    ]["acoustic_visual_identity"] == runtime_identity
    first = index["samples"][0]
    assert first["acoustic_selection_binding_sha256"] == acoustic_binding_sha256
    assert first["runtime_map_id"] == runtime_identity["runtime_map_id"]
    assert first["spear_ue_runtime_evidence_identity"] == {
        "status": "pass",
        "verification_status": "verified",
        "evidence_schema": (
            "avengine_optional_spear_apartment_runtime_evidence_v2"
        ),
        "evidence_sha256": sha256_file(runtime_evidence_path),
        "acoustic_visual_identity": runtime_identity,
    }
    assert first["audio_program_binding"] == base["audio_program_binding"]
    assert (
        first["audio_program_instance_path"]
        == base["audio_program_instance_path"]
    )
    assert (
        first["audio_program_instance_sha256"]
        == base["audio_program_instance_sha256"]
    )
    assert first["label_paths"]["audio_program_instance"] == base[
        "audio_program_instance_path"
    ]
    assert first["label_path_roots"] == {
        "timeline": "visual_bundle_root",
        "source_manifest": "visual_bundle_root",
        "flags": "visual_bundle_root",
        "audio_program_instance": "audio_batch_root",
    }
    splits_by_episode = {
        episode["episode_id"]: {
            row["split"]
            for row in index["samples"]
            if row["episode_id"] == episode["episode_id"]
        }
        for episode in episodes
    }
    assert all(len(splits) == 1 for splits in splits_by_episode.values())

    visual_manifest = load_json(visual_root / "manifest.json")
    different_binding = dict(acoustic_binding)
    different_binding["binding_id"] = "different_visual_binding"
    different_binding["binding_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in different_binding.items()
            if key != "binding_content_sha256"
        }
    )
    visual_manifest["acoustic_selection_binding"] = different_binding
    write_json(visual_root / "manifest.json", visual_manifest)
    with pytest.raises(
        _INDEX.ApartmentDatasetIndexError,
        match="audio and visual acoustic selection bindings differ",
    ):
        _INDEX.build_index(
            audio_batch_root=audio_root,
            visual_bundle_root=visual_root,
            ue_render_root=render_root,
            output=tmp_path / "mismatched_index",
            spear_runtime_evidence=runtime_evidence_path,
        )


def test_audio_program_instance_rejects_hash_and_binding_tampering(
    tmp_path: Path,
) -> None:
    sample = _program_bound_sample(tmp_path)
    sample["audio_program_instance_sha256"] = "0" * 64
    with pytest.raises(
        _VERIFY.BatchVerificationError,
        match="instance hash differs",
    ):
        _VERIFY._audio_program_instance(
            batch_root=tmp_path,
            sample=sample,
            sample_id=sample["sample_id"],
            asset_ids_by_source_slot=sample["asset_ids_by_source_slot"],
        )
    with pytest.raises(
        _INDEX.ApartmentDatasetIndexError,
        match="instance file or hash differs",
    ):
        _INDEX._audio_program_index_fields(
            sample,
            audio_batch_root=tmp_path,
        )

    sample = _program_bound_sample(tmp_path)
    path = tmp_path / sample["audio_program_instance_path"]
    instance = load_json(path)
    instance["audio_program_binding"]["variant_id"] = "B"
    write_json(path, instance)
    sample["audio_program_instance_sha256"] = sha256_file(path)
    with pytest.raises(
        _VERIFY.BatchVerificationError,
        match="instance binding differs",
    ):
        _VERIFY._audio_program_instance(
            batch_root=tmp_path,
            sample=sample,
            sample_id=sample["sample_id"],
            asset_ids_by_source_slot=sample["asset_ids_by_source_slot"],
        )


def test_new_one_sample_canary_may_declare_not_both_sources_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = {"source1": "asset_a", "source2": "asset_b"}
    acoustic_binding = _registry_acoustic_selection_binding()
    acoustic_binding_sha256 = acoustic_binding["binding_content_sha256"]
    samples = [
        {
            "sample_id": f"episode0__v{index:04d}",
            "episode_id": "episode0",
            "variant_index": index,
            "asset_ids_by_source_slot": assets,
            "acoustic_selection_binding_sha256": acoustic_binding_sha256,
            "both_sources_active": False,
            "audio_program_binding": {},
            "audio_program_instance_path": "labels/instance.json",
            "audio_program_instance_sha256": "0" * 64,
            "audio": {
                "sample_rate_hz": 16_000,
                "sample_count": 80_000,
                "channel_count": 2,
                "layout": "native_RLR_HRTF_binaural_left_right",
                "mixture_is_exact_stem_sum_before_delivery": True,
                "mixture": {
                    "path": "shared.wav",
                    "audio_sha256": "a" * 64,
                    "sidecar_path": "shared.wav.json",
                    "sidecar_sha256": "b" * 64,
                },
            },
        }
        for index in range(1)
    ]
    documents = {
        "delivery.json": {
            "status": "pass",
            "acoustic_selection_binding": acoustic_binding,
            "sample_count": 1,
            "episode_count": 1,
            "variants_per_episode": 1,
            "both_sources_active": False,
        },
        "samples.json": {
            "status": "pass",
            "acoustic_selection_binding": acoustic_binding,
            "sample_count": 1,
            "samples": samples,
        },
        "episodes.json": {
            "status": "pass",
            "acoustic_selection_binding": acoustic_binding,
            "episodes": [
                {
                    "episode_id": "episode0",
                    "acoustic_selection_binding_sha256": (
                        acoustic_binding_sha256
                    ),
                    "rir_cache": {
                        "acoustic_selection_binding": acoustic_binding,
                    },
                }
            ],
        },
    }
    audio_root = tmp_path / "audio/binaural"
    audio_root.mkdir(parents=True)
    mixture_path = audio_root / "shared.wav"
    mixture_sidecar_path = audio_root / "shared.wav.json"
    for path in (mixture_path, mixture_sidecar_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    rendered = SimpleNamespace(
        sample_rate_hz=16_000,
        samples=np.zeros((2, 80_000), dtype=np.float32),
        sidecar={
            "metadata": {
                "acoustic_selection_binding_sha256": acoustic_binding_sha256,
            }
        },
    )
    hashes_by_path = {
        mixture_path.resolve(): "a" * 64,
        mixture_sidecar_path.resolve(): "b" * 64,
    }
    monkeypatch.setattr(
        _VERIFY,
        "_json",
        lambda path: documents[path.name],
    )
    monkeypatch.setattr(
        _VERIFY,
        "_audio_program_instance",
        lambda **_kwargs: {
            "mode": "one_active_of_n",
            "active_source_slots": ["source1"],
            "both_sources_active": False,
        },
    )
    monkeypatch.setattr(
        _VERIFY,
        "sha256_file",
        lambda path: hashes_by_path[Path(path).resolve()],
    )
    monkeypatch.setattr(
        _VERIFY,
        "read_float32_wav",
        lambda *_args, **_kwargs: rendered,
    )

    result = _VERIFY._verify_batch(
        tmp_path,
        expected_assets={"episode0": assets},
        expected_sample_count=1,
    )
    assert result["audio_program_instance_sample_count"] == 1
    assert result["audio_program_mode_counts"] == {"one_active_of_n": 1}
    assert result["acoustic_selection_binding"] == acoustic_binding

    rendered.sidecar["metadata"][
        "acoustic_selection_binding_sha256"
    ] = "f" * 64
    with pytest.raises(
        _VERIFY.BatchVerificationError,
        match="sidecar acoustic selection differs",
    ):
        _VERIFY._verify_batch(
            tmp_path,
            expected_assets={"episode0": assets},
            expected_sample_count=1,
        )


@pytest.mark.parametrize(
    ("header_field", "bad_value"),
    (("status", "fail"), ("sample_count", 2)),
)
def test_batch_verifier_rejects_invalid_samples_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    header_field: str,
    bad_value: object,
) -> None:
    acoustic_binding = _legacy_unbound_acoustic_selection_binding()
    samples_record = {
        "status": "pass",
        "acoustic_selection_binding": acoustic_binding,
        "sample_count": 1,
        "samples": [{}],
    }
    samples_record[header_field] = bad_value
    documents = {
        "delivery.json": {
            "status": "pass",
            "acoustic_selection_binding": acoustic_binding,
            "sample_count": 1,
            "episode_count": 1,
            "variants_per_episode": 1,
            "both_sources_active": True,
        },
        "samples.json": samples_record,
        "episodes.json": {
            "status": "pass",
            "acoustic_selection_binding": acoustic_binding,
            "episodes": [{}],
        },
    }
    monkeypatch.setattr(
        _VERIFY,
        "_json",
        lambda path: documents[path.name],
    )

    with pytest.raises(
        _VERIFY.BatchVerificationError,
        match="expected M7 sample count",
    ):
        _VERIFY._verify_batch(
            tmp_path,
            expected_assets={
                "episode0": {
                    "source1": "asset_a",
                    "source2": "asset_b",
                }
            },
            expected_sample_count=1,
        )


def test_legacy_batch_still_requires_both_sources_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acoustic_binding = _legacy_unbound_acoustic_selection_binding()
    documents = {
        "delivery.json": {
            "status": "pass",
            "acoustic_selection_binding": acoustic_binding,
            "sample_count": 1_000,
            "episode_count": 1,
            "variants_per_episode": 1_000,
            "both_sources_active": False,
        },
        "samples.json": {
            "status": "pass",
            "acoustic_selection_binding": acoustic_binding,
            "sample_count": 1_000,
            "samples": [
                {
                    "sample_id": f"legacy_{index}",
                    "episode_id": "episode0",
                    "variant_index": index,
                    "asset_ids_by_source_slot": {
                        "source1": "asset_a",
                        "source2": "asset_b",
                    },
                    "both_sources_active": False,
                }
                for index in range(1_000)
            ],
        },
        "episodes.json": {
            "status": "pass",
            "acoustic_selection_binding": acoustic_binding,
            "episodes": [{}],
        },
    }
    monkeypatch.setattr(
        _VERIFY,
        "_json",
        lambda path: documents[path.name],
    )
    with pytest.raises(
        _VERIFY.BatchVerificationError,
        match="expected M7 sample count",
    ):
        _VERIFY._verify_batch(
            tmp_path,
            expected_assets={
                "episode0": {
                    "source1": "asset_a",
                    "source2": "asset_b",
                }
            },
        )

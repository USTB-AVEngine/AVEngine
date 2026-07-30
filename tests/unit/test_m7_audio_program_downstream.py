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
from avengine.m6.audio_program import (
    bind_audio_program_hash,
    materialize_audio_program_variant,
)
from avengine.m6x.room_feasibility import rir_acoustic_state_sha256
from avengine.m7.sensor_rig import (
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
    "tools/m7/verify_asset_bound_batch.py",
)
_INDEX = _load_tool(
    "build_asset_bound_dataset_index_for_test",
    "tools/m7/build_asset_bound_dataset_index.py",
)


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
    episodes = [
        {
            "episode_id": f"episode_{index:04d}",
            "motion_case": f"motion_{index % 4}",
            "asset_ids_by_source_slot": assets,
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
    documents = {
        "samples.json": {
            "status": "pass",
            "sample_count": 1_000,
            "samples": samples,
        },
        "delivery.json": {
            "status": "pass",
            "sample_count": 1_000,
            "episode_count": 1_000,
            "variants_per_episode": 1,
            "both_sources_active": False,
        },
    }
    for filename, document in documents.items():
        write_json(audio_root / filename, document)
    documents["verification.json"] = {"status": "pass"}
    write_json(audio_root / "verification.json", documents["verification.json"])
    monkeypatch.setattr(_INDEX, "_visual_episodes", lambda _root: episodes)
    monkeypatch.setattr(
        _INDEX,
        "_render_evidence",
        lambda _root: render_evidence,
    )

    _INDEX.build_index(
        audio_batch_root=audio_root,
        visual_bundle_root=visual_root,
        ue_render_root=render_root,
        output=output,
    )

    index = load_json(output / "dataset_index.json")
    assert len(index["samples"]) == 1_000
    first = index["samples"][0]
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
    samples = [
        {
            "sample_id": f"episode0__v{index:04d}",
            "episode_id": "episode0",
            "variant_index": index,
            "asset_ids_by_source_slot": assets,
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
            "sample_count": 1,
            "episode_count": 1,
            "variants_per_episode": 1,
            "both_sources_active": False,
        },
        "samples.json": {
            "status": "pass",
            "sample_count": 1,
            "samples": samples,
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
        sidecar={},
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
    samples_record = {
        "status": "pass",
        "sample_count": 1,
        "samples": [{}],
    }
    samples_record[header_field] = bad_value
    documents = {
        "delivery.json": {
            "status": "pass",
            "sample_count": 1,
            "episode_count": 1,
            "variants_per_episode": 1,
            "both_sources_active": True,
        },
        "samples.json": samples_record,
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
    documents = {
        "delivery.json": {
            "status": "pass",
            "sample_count": 1_000,
            "episode_count": 1,
            "variants_per_episode": 1_000,
            "both_sources_active": False,
        },
        "samples.json": {
            "status": "pass",
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

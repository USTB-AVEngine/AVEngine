from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.contracts.json_io import load_json, sha256_file, write_json
from avengine.m6x.room_feasibility import TrajectoryBank, TrajectoryEpisode
from avengine.m7.room_evaluation import (
    AZIMUTH_REGIONS,
    RoomEvaluationError,
    build_room_evaluation_plan,
    validate_episode_id,
)
from avengine.security.path_policy import atomic_publish_directory
from tools.m7 import build_room_evaluation_plan as plan_tool
from tools.m7.render_room_evaluation_binaural import (
    OUTPUT_CLOSURE_FILES,
    OUTPUT_CLOSURE_SCHEMA,
    RESULT_CHANGING_CODE_FILES,
    AssetBoundAudioError,
    _active_stem_peaks,
    _assignments,
    _cache_closure,
    _cache_only_execution_evidence,
    _output_closure,
    _plan_closure,
    _producer_identity,
    _publication_paths,
    _verify_persisted_exact_mix,
    _write_and_readback,
)


def _bank() -> dict:
    episodes = []
    motion_cases = (
        "static_static",
        "source1_moving_source2_static",
        "source1_static_source2_moving",
        "both_moving",
    )
    for motion_case in motion_cases:
        for index in range(3):
            base = float(index + motion_cases.index(motion_case) * 10)
            source1 = np.asarray([[base + frame, 1.5, 2.0] for frame in range(3)])
            source2 = np.asarray([[base, 1.5, 3.0 + frame] for frame in range(3)])
            if "source1_moving" not in motion_case and motion_case != "both_moving":
                source1[:] = source1[0]
            if "source2_moving" not in motion_case and motion_case != "both_moving":
                source2[:] = source2[0]
            episodes.append(
                TrajectoryEpisode(
                    episode_id=f"{motion_case}_{index:03d}",
                    motion_case=motion_case,
                    source_root_paths_m={"source1": source1, "source2": source2},
                    source_center_paths_m={"source1": source1, "source2": source2},
                    statistics={},
                )
            )
    return TrajectoryBank(
        episodes=tuple(episodes), frame_count=3, frame_rate_hz=1, seed=7
    ).record()


def _renderer_bank() -> dict:
    record = _bank()
    source_times = np.arange(3, dtype=np.float64)
    target_times = np.linspace(0.0, 2.0, 75)
    for episode in record["episodes"]:
        for field in ("source_root_paths_m", "source_center_paths_m"):
            for slot, raw in episode[field].items():
                points = np.asarray(raw, dtype=np.float64)
                episode[field][slot] = np.column_stack(
                    [
                        np.interp(target_times, source_times, points[:, axis])
                        for axis in range(3)
                    ]
                ).tolist()
    record["frame_count"] = 75
    record["frame_rate_hz"] = 15
    record["seconds_per_episode"] = 5.0
    return record


def test_room_evaluation_plan_balances_motion_and_sound_pairs():
    result = build_room_evaluation_plan(
        _bank(),
        listener_position_m=[0.0, 1.5, 0.0],
        listener_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        stride_frames=1,
        episode_count=8,
    )
    assert result.summary["episode_count"] == 8
    assert result.summary["motion_case_counts"] == {
        "both_moving": 2,
        "source1_moving_source2_static": 2,
        "source1_static_source2_moving": 2,
        "static_static": 2,
    }
    assert result.rir_job_plan["dry_audio_independent"] is True
    assignments = result.sound_assignments["assignments"]
    assert len(assignments) == 8
    assert all(
        value["source_classes"]["source1"]
        != value["source_classes"]["source2"]
        for value in assignments
    )
    assert "asset" not in result.sound_assignments["semantics"] or (
        "not visible asset IDs" in result.sound_assignments["semantics"]
    )


def test_room_evaluation_plan_rejects_insufficient_motion_case():
    with pytest.raises(RoomEvaluationError, match="needs"):
        build_room_evaluation_plan(
            _bank(),
            listener_position_m=[0.0, 1.5, 0.0],
            listener_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            stride_frames=1,
            episode_count=16,
        )


@pytest.mark.parametrize(
    "episode_id",
    (
        "../escape",
        "..",
        "/absolute",
        "nested/episode",
        r"nested\episode",
        ".hidden",
        "C:drive",
    ),
)
def test_room_evaluation_plan_rejects_unsafe_episode_ids(episode_id):
    bank = _bank()
    bank["episodes"][0]["episode_id"] = episode_id
    with pytest.raises(RoomEvaluationError, match="episode_id must match"):
        build_room_evaluation_plan(
            bank,
            listener_position_m=[0.0, 1.5, 0.0],
            listener_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            stride_frames=1,
            episode_count=8,
        )


def test_episode_id_validator_accepts_portable_component() -> None:
    assert validate_episode_id("both_moving_048") == "both_moving_048"


def _circular_bank() -> dict:
    episodes = []
    motion_cases = (
        "static_static",
        "source1_moving_source2_static",
        "source1_static_source2_moving",
        "both_moving",
    )
    for motion_case in motion_cases:
        for index in range(8):
            base_angle = np.deg2rad(index * 45.0)
            source_paths = {}
            for source_index, slot in enumerate(("source1", "source2")):
                moving = motion_case == "both_moving" or f"{slot}_moving" in motion_case
                angles = base_angle + source_index * np.pi
                angles = angles + (np.arange(3) * 0.1 if moving else np.zeros(3))
                source_paths[slot] = np.column_stack(
                    [2.0 * np.sin(angles), np.full(3, 1.5), -2.0 * np.cos(angles)]
                )
            episodes.append(
                TrajectoryEpisode(
                    episode_id=f"{motion_case}_{index:03d}",
                    motion_case=motion_case,
                    source_root_paths_m=source_paths,
                    source_center_paths_m=source_paths,
                    statistics={},
                )
            )
    return TrajectoryBank(
        episodes=tuple(episodes), frame_count=3, frame_rate_hz=1, seed=11
    ).record()


def test_direction_balanced_plan_deconfounds_motion_and_sound_pairs():
    result = build_room_evaluation_plan(
        _circular_bank(),
        listener_position_m=[0.0, 1.5, 0.0],
        listener_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        stride_frames=1,
        episode_count=24,
        minimum_listener_source_distance_m=0.3,
        balance_azimuth_regions=True,
        minimum_azimuth_region_fraction=0.15,
    )
    assert result.summary["motion_case_counts"] == {
        "both_moving": 6,
        "source1_moving_source2_static": 6,
        "source1_static_source2_moving": 6,
        "static_static": 6,
    }
    assert set(result.summary["azimuth_region_frame_fractions"]) == set(
        AZIMUTH_REGIONS
    )
    assert min(result.summary["azimuth_region_frame_fractions"].values()) >= 0.15
    assert all(
        len(pair_counts) == 6
        for pair_counts in result.summary["motion_sound_pair_counts"].values()
    )
    assert result.summary["minimum_listener_source_distance_m_observed"] >= 0.3


def test_m7_public_exports_keep_audio_and_room_interfaces():
    import avengine.m7 as m7

    assert {
        "prepare_dry_audio",
        "render_asset_bound_binaural",
        "build_room_evaluation_plan",
    }.issubset(set(m7.__all__))


def test_room_evaluation_renderer_requires_two_active_float32_stems():
    active = {
        "source1": np.ones((2, 80_000), dtype=np.float32),
        "source2": np.full((2, 80_000), 0.25, dtype=np.float32),
    }
    assert _active_stem_peaks(active, sample_id="episode__v00") == {
        "source1": 1.0,
        "source2": 0.25,
    }

    active["source2"][:] = 0.0
    with pytest.raises(AssetBoundAudioError, match="two active float32"):
        _active_stem_peaks(active, sample_id="episode__v00")


def test_room_evaluation_renderer_rejects_unsafe_assignment_before_paths(
    tmp_path,
):
    path = tmp_path / "sound_assignments.json"
    write_json(
        path,
        {
            "schema": "avengine_room_sound_class_assignments_v1",
            "status": "pass",
            "both_sources_active": True,
            "sound_classes": ["cat meowing", "dog barking"],
            "episode_count": 1,
            "ordered_pair_counts": {"dog barking|cat meowing": 1},
            "assignments": [
                {
                    "episode_id": "../escape",
                    "source_classes": {
                        "source1": "dog barking",
                        "source2": "cat meowing",
                    },
                }
            ],
        },
    )
    with pytest.raises(AssetBoundAudioError, match="unsafe episode_id"):
        _assignments(path)


def test_room_evaluation_renderer_verifies_persisted_exact_stem_sum(tmp_path):
    source1 = np.full((2, 8), 0.25, dtype=np.float32)
    source2 = np.full((2, 8), 0.125, dtype=np.float32)
    mixture = np.zeros_like(source1)
    mixture += source1
    mixture += source2
    readbacks = {}
    for slot, samples in (("source1", source1), ("source2", source2)):
        record, readbacks[slot] = _write_and_readback(
            tmp_path / "stems" / slot / "sample.wav",
            samples,
            root=tmp_path,
            role="fixture_stem",
            metadata={"source_slot_id": slot},
        )
        assert len(record["audio_sha256"]) == 64
        assert len(record["sidecar_sha256"]) == 64
    mixture_record, mixture_readback = _write_and_readback(
        tmp_path / "sample.wav",
        mixture,
        root=tmp_path,
        role="fixture_mixture",
        metadata={},
    )
    assert mixture_record["peak_absolute"] == 0.375
    assert _verify_persisted_exact_mix(
        readbacks, mixture_readback, sample_id="sample"
    ) == {"source1": 0.25, "source2": 0.125}

    changed = mixture_readback.copy()
    changed[0, 0] += np.float32(0.125)
    with pytest.raises(AssetBoundAudioError, match="delivered stem sum"):
        _verify_persisted_exact_mix(readbacks, changed, sample_id="sample")


def test_room_evaluation_renderer_binds_producer_and_output_closure(tmp_path):
    identity = _producer_identity()
    assert set(identity["result_changing_code_files"]) == set(
        RESULT_CHANGING_CODE_FILES
    )
    assert identity["runtime"]["python_version"]
    assert identity["runtime"]["numpy_version"] == np.__version__
    for relative, record in identity["result_changing_code_files"].items():
        assert record["path"] == relative
        assert len(record["sha256"]) == 64
        assert record["byte_size"] > 0

    for relative in OUTPUT_CLOSURE_FILES:
        write_json(tmp_path / relative, {"path": relative})
    closure = _output_closure(tmp_path, sample_count=2)
    assert closure["schema"] == OUTPUT_CLOSURE_SCHEMA
    assert closure["audio_artifact_file_count"] == 12
    assert "delivery.json" not in closure["files"]
    for relative, record in closure["files"].items():
        assert record["sha256"] == sha256_file(tmp_path / relative)


def test_room_evaluation_publication_is_atomic_no_replace(tmp_path):
    policy, output, staging = _publication_paths(tmp_path / "batch")
    staging.mkdir()
    write_json(staging / "payload.json", {"status": "pass"})
    published = atomic_publish_directory(policy, staging, output)
    assert published == output
    assert (published / "payload.json").is_file()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        _publication_paths(tmp_path / "batch")


def test_room_evaluation_plan_cli_uses_immutable_directory_publication(
    tmp_path,
    monkeypatch,
):
    trajectory_path = tmp_path / "trajectory.json"
    template_path = tmp_path / "rir_template.json"
    output = tmp_path / "plan"
    write_json(trajectory_path, _bank())
    write_json(
        template_path,
        {
            "listener_position_m": [0.0, 1.5, 0.0],
            "listener_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "stride_frames": 1,
        },
    )
    arguments = SimpleNamespace(
        trajectory_bank=trajectory_path,
        template_rir_plan=template_path,
        episode_count=8,
        sound_classes=None,
        listener_position_m=None,
        listener_orientation_wxyz=None,
        minimum_listener_source_distance_m=0.0,
        balance_azimuth_regions=False,
        minimum_azimuth_region_fraction=0.0,
        output=output,
    )
    monkeypatch.setattr(plan_tool, "parse_args", lambda: arguments)
    assert plan_tool.main() == 0
    assert load_json(output / "delivery.json")["status"] == "pass"
    with pytest.raises(FileExistsError, match="refusing to replace"):
        plan_tool.main()


def test_room_evaluation_renderer_closes_plan_cache_and_execution_identity(
    tmp_path,
):
    result = build_room_evaluation_plan(
        _renderer_bank(),
        listener_position_m=[0.0, 1.5, 0.0],
        listener_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        stride_frames=3,
        episode_count=8,
    )
    plan_root = tmp_path / "plan"
    write_json(plan_root / "trajectory_bank.json", result.trajectory_bank)
    write_json(plan_root / "rir_job_plan.json", result.rir_job_plan)
    write_json(plan_root / "sound_assignments.json", result.sound_assignments)
    write_json(plan_root / "delivery.json", result.summary)
    assignments, _ = _assignments(plan_root / "sound_assignments.json")
    plan = _plan_closure(plan_root, assignments)
    assert plan["status"] == "pass"
    assert set(plan["files"]) == {
        "delivery.json",
        "trajectory_bank.json",
        "sound_assignments.json",
        "rir_job_plan.json",
    }

    cache_root = tmp_path / "cache"
    request_identity = "ab" * 32
    request = {
        "schema": "avengine_rlr_rir_cache_request_v1",
        "request_identity_sha256": request_identity,
        "plan": {
            "path": str(plan_root / "rir_job_plan.json"),
            "sha256": plan["files"]["rir_job_plan.json"]["sha256"],
            "full_job_count": plan["unique_rir_job_count"],
            "selected_job_offset": 0,
            "selected_job_count": plan["unique_rir_job_count"],
        },
        "acoustic_scene": {
            "manifest_path": "/fixture/acoustic/manifest.json",
            "manifest_sha256": "11" * 32,
            "package_id": "fixture_package",
            "package_content_sha256": "22" * 32,
        },
        "simulation": {
            "request_path": "/fixture/simulation.json",
            "request_sha256": "33" * 32,
        },
        "output": {
            "layout_type": "binaural",
            "hrtf_path": "/fixture/hrtf.sofa",
            "hrtf_sha256": "44" * 32,
        },
    }
    receipt = {
        "schema": "avengine_rlr_rir_cache_receipt_v1",
        "status": "pass",
        "request_identity_sha256": request_identity,
        "full_plan_complete": True,
        "full_plan_job_count": plan["unique_rir_job_count"],
        "selected_job_count": plan["unique_rir_job_count"],
        "retained_payload_hash_verified": True,
    }
    index = {
        "schema": "avengine_rlr_rir_cache_index_v1",
        "status": "pass",
        "request_identity_sha256": request_identity,
        "full_plan_complete": True,
        "selected_job_count": plan["unique_rir_job_count"],
        "entries": [
            {"job_index": index}
            for index in range(plan["unique_rir_job_count"])
        ],
    }
    write_json(cache_root / "request.json", request)
    write_json(cache_root / "receipt.json", receipt)
    write_json(cache_root / "index.json", index)
    session = SimpleNamespace(
        request_identity_sha256=request_identity,
        plan_sha256=plan["files"]["rir_job_plan.json"]["sha256"],
        external_input_identity={
            "status": "pass",
            "plan": {
                "declared_path": request["plan"]["path"],
                "sha256": request["plan"]["sha256"],
            },
            "acoustic_scene": {
                "declared_path": request["acoustic_scene"]["manifest_path"],
                "sha256": request["acoustic_scene"]["manifest_sha256"],
                "package_id": request["acoustic_scene"]["package_id"],
                "package_content_sha256": request["acoustic_scene"][
                    "package_content_sha256"
                ],
                "manifest_content_identity_verified": True,
            },
            "simulation_request": {
                "declared_path": request["simulation"]["request_path"],
                "sha256": request["simulation"]["request_sha256"],
            },
            "hrtf": {
                "declared_path": request["output"]["hrtf_path"],
                "sha256": request["output"]["hrtf_sha256"],
            },
        },
    )
    cache = _cache_closure(
        cache_root,
        session,
        rir_plan_sha256=plan["files"]["rir_job_plan.json"]["sha256"],
        unique_rir_job_count=plan["unique_rir_job_count"],
    )
    assert cache["status"] == "pass"
    assert cache["external_inputs"]["status"] == "pass"
    assert len(cache["files"]["receipt.json"]["sha256"]) == 64

    evidence = _cache_only_execution_evidence(
        sample_count=8,
        cache_load_count=8,
        dynamic_convolution_count=8,
        persisted_mix_verification_count=8,
    )
    assert evidence["native_rlr_calls"] == 0
    assert evidence["visual_render_calls"] == 0
    assert evidence["rir_cache_load_count"] == 8

    changed = load_json(plan_root / "trajectory_bank.json")
    changed["episodes"][0]["source_center_paths_m"]["source1"][0][0] += 1.0
    write_json(plan_root / "trajectory_bank.json", changed)
    with pytest.raises(AssetBoundAudioError, match="trajectory source center"):
        _plan_closure(plan_root, assignments)

    changed = deepcopy(result.rir_job_plan)
    changed["jobs"][0]["uses"][0]["episode_id"] = "../escape"
    write_json(plan_root / "trajectory_bank.json", result.trajectory_bank)
    write_json(plan_root / "rir_job_plan.json", changed)
    with pytest.raises(AssetBoundAudioError, match="unsafe episode_id"):
        _plan_closure(plan_root, assignments)

    with pytest.raises(AssetBoundAudioError, match="counters"):
        _cache_only_execution_evidence(
            sample_count=8,
            cache_load_count=7,
            dynamic_convolution_count=8,
            persisted_mix_verification_count=8,
        )

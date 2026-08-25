from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)
from avengine.routes.room_feasibility import TrajectoryBank, TrajectoryEpisode
from avengine.m7.room_evaluation import (
    AZIMUTH_REGIONS,
    RoomEvaluationError,
    build_room_evaluation_plan,
    build_static_source_trajectory_bank,
    validate_episode_id,
)
from avengine.security.path_policy import atomic_publish_directory
from avengine.sensor_rig_trajectory import materialize_sensor_rig_trajectory
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


def test_room_evaluation_binds_dynamic_listener_pose_into_rir_jobs() -> None:
    trajectory = materialize_sensor_rig_trajectory(
        trajectory_id="m7_dynamic_listener_test",
        program={
            "kind": "WAYPOINT_ROUTE",
            "waypoints": [
                {
                    "frame_index": 0,
                    "position_m": [0.0, 1.5, 0.0],
                    "yaw_deg": 0.0,
                },
                {
                    "frame_index": 74,
                    "position_m": [1.0, 1.5, 0.5],
                    "yaw_deg": 90.0,
                },
            ],
            "interpolation": "LINEAR_POSITION_SHORTEST_YAW",
        },
    )
    result = build_room_evaluation_plan(
        _renderer_bank(),
        listener_position_m=[0.0, 1.5, 0.0],
        listener_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        stride_frames=3,
        episode_count=8,
        sensor_rig_trajectory=trajectory,
    )
    assert result.summary["listener_pose_mode"] == "sensor_rig_trajectory_v1"
    assert (
        result.summary["sensor_rig_trajectory"]["trajectory_id"]
        == "m7_dynamic_listener_test"
    )
    assert result.rir_job_plan["listener_pose_mode"] == "per_episode_frame"
    job_positions = {
        tuple(job["listener_position_m"])
        for job in result.rir_job_plan["jobs"]
    }
    assert len(job_positions) > 1


def test_static_source_bank_uses_existing_visual_clock() -> None:
    bank = build_static_source_trajectory_bank(
        {
            "source1": [-7.5, 1.2, -3.0],
            "source2": [-10.0, 1.2, -4.0],
        },
        frame_count=75,
        frame_rate_hz=15,
        episode_id="mp3d_static_sources_000",
        seed=17,
    )
    assert bank["episode_count"] == 1
    assert bank["frame_count"] == 75
    assert bank["frame_rate_hz"] == 15
    assert bank["motion_case_counts"]["static_static"] == 1
    episode = bank["episodes"][0]
    assert episode["episode_id"] == "mp3d_static_sources_000"
    assert episode["motion_case"] == "static_static"
    assert episode["source_center_paths_m"]["source1"] == [
        [-7.5, 1.2, -3.0]
    ] * 75
    assert episode["source_root_paths_m"] == episode["source_center_paths_m"]


@pytest.mark.parametrize(
    ("positions", "frame_count", "frame_rate_hz", "match"),
    [
        ({"source1": [0, 1, 2]}, 75, 15, "exactly source1/source2"),
        (
            {"source1": [0, 1, 2], "source2": [0, float("nan"), 2]},
            75,
            15,
            "finite xyz",
        ),
        (
            {"source1": [0, 1, 2], "source2": [1, 1, 2]},
            1,
            15,
            "clock or seed",
        ),
    ],
)
def test_static_source_bank_rejects_invalid_contract(
    positions,
    frame_count,
    frame_rate_hz,
    match,
) -> None:
    with pytest.raises(RoomEvaluationError, match=match):
        build_static_source_trajectory_bank(
            positions,
            frame_count=frame_count,
            frame_rate_hz=frame_rate_hz,
        )


def test_m7_public_exports_keep_audio_and_room_interfaces():
    import avengine.m7 as m7

    assert {
        "prepare_dry_audio",
        "render_asset_bound_binaural",
        "build_room_evaluation_plan",
        "build_static_source_trajectory_bank",
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
        static_source1_position_m=None,
        static_source2_position_m=None,
        static_episode_id="static_sources_000",
        static_seed=0,
        template_rir_plan=template_path,
        episode_count=8,
        sound_classes=None,
        listener_position_m=None,
        listener_orientation_wxyz=None,
        sensor_rig_trajectory=None,
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


def test_room_evaluation_plan_cli_builds_static_sources_on_sensor_clock(
    tmp_path,
    monkeypatch,
) -> None:
    trajectory = materialize_sensor_rig_trajectory(
        trajectory_id="mp3d_static_source_cli",
        program={
            "kind": "WAYPOINT_ROUTE",
            "waypoints": [
                {
                    "frame_index": 0,
                    "position_m": [0.0, 1.5, 0.0],
                    "yaw_deg": 0.0,
                },
                {
                    "frame_index": 74,
                    "position_m": [1.0, 1.5, 0.0],
                    "yaw_deg": 45.0,
                },
            ],
            "interpolation": "LINEAR_POSITION_SHORTEST_YAW",
        },
    )
    sensor_path = tmp_path / "sensor_rig_trajectory.json"
    template_path = tmp_path / "rir_template.json"
    output = tmp_path / "static_plan"
    write_json(sensor_path, trajectory)
    write_json(template_path, {"stride_frames": 1})
    arguments = SimpleNamespace(
        trajectory_bank=None,
        static_source1_position_m=[-7.5, 1.2, -3.0],
        static_source2_position_m=[-10.0, 1.2, -4.0],
        static_episode_id="mp3d_static_sources_000",
        static_seed=17,
        template_rir_plan=template_path,
        episode_count=1,
        sound_classes=["dog barking", "human speech"],
        listener_position_m=None,
        listener_orientation_wxyz=None,
        sensor_rig_trajectory=sensor_path,
        minimum_listener_source_distance_m=0.0,
        balance_azimuth_regions=False,
        minimum_azimuth_region_fraction=0.0,
        output=output,
    )
    monkeypatch.setattr(plan_tool, "parse_args", lambda: arguments)

    assert plan_tool.main() == 0
    delivery = load_json(output / "delivery.json")
    bank = load_json(output / "trajectory_bank.json")
    assert delivery["listener_pose_mode"] == "sensor_rig_trajectory_v1"
    assert delivery["episode_count"] == 1
    assert delivery["unique_rir_job_count"] == 150
    assert bank["episodes"][0]["motion_case"] == "static_static"
    assert (output / "sensor_rig_trajectory.json").is_file()
    assignments, _ = _assignments(output / "sound_assignments.json")
    closure = _plan_closure(output, assignments)
    assert closure["episode_count"] == 1
    assert closure["listener_pose_mode"] == "per_episode_frame"


def test_room_evaluation_renderer_closes_dynamic_sensor_rig_plan(
    tmp_path,
) -> None:
    trajectory = materialize_sensor_rig_trajectory(
        trajectory_id="room_renderer_dynamic_listener",
        program={
            "kind": "WAYPOINT_ROUTE",
            "waypoints": [
                {
                    "frame_index": 0,
                    "position_m": [0.0, 1.5, 0.0],
                    "yaw_deg": 0.0,
                },
                {
                    "frame_index": 74,
                    "position_m": [0.5, 1.5, 0.5],
                    "yaw_deg": 45.0,
                },
            ],
            "interpolation": "LINEAR_POSITION_SHORTEST_YAW",
        },
    )
    result = build_room_evaluation_plan(
        _renderer_bank(),
        listener_position_m=[0.0, 1.5, 0.0],
        listener_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        stride_frames=3,
        episode_count=8,
        sensor_rig_trajectory=trajectory,
    )
    plan_root = tmp_path / "dynamic_plan"
    write_json(plan_root / "trajectory_bank.json", result.trajectory_bank)
    write_json(plan_root / "rir_job_plan.json", result.rir_job_plan)
    write_json(
        plan_root / "sound_assignments.json", result.sound_assignments
    )
    write_json(plan_root / "delivery.json", result.summary)
    write_json(plan_root / "sensor_rig_trajectory.json", trajectory)
    assignments, _ = _assignments(plan_root / "sound_assignments.json")

    closure = _plan_closure(plan_root, assignments)

    assert closure["listener_pose_mode"] == "per_episode_frame"
    assert (
        closure["sensor_rig_trajectory"]["trajectory_id"]
        == "room_renderer_dynamic_listener"
    )
    assert closure["sensor_rig_rir_alignment"]["checked_use_count"] == 400
    assert "sensor_rig_trajectory.json" in closure["files"]


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
    selection_binding = {
        "schema": "avengine_rir_cache_acoustic_selection_binding_v1",
        "selection_mode": "registry",
        "registry_selection_applied": True,
        "room_ref": {
            "room_id": "17DRP5sb8fy",
            "room_revision": "mp3d_v1",
            "suite": "soundspaces2_mp3d",
        },
        "profile_ref": {
            "profile_id": "soundspaces2_mp3d_public_materials_v1",
            "profile_revision": "soundspaces_287184fd_rlr_4fd446b4",
        },
        "binding_id": "17DRP5sb8fy_soundspaces2_public",
        "registry_selection_content_sha256": "55" * 32,
        "effective_selection_content_sha256": "66" * 32,
        "acoustic_package_manifest_sha256": request["acoustic_scene"][
            "manifest_sha256"
        ],
        "simulation_request_sha256": request["simulation"]["request_sha256"],
        "input_receipt_sha256": "77" * 32,
    }
    selection_binding["binding_content_sha256"] = canonical_json_sha256(
        selection_binding
    )
    request["acoustic_selection_binding"] = selection_binding
    receipt = {
        "schema": "avengine_rlr_rir_cache_receipt_v1",
        "status": "pass",
        "request_identity_sha256": request_identity,
        "full_plan_complete": True,
        "full_plan_job_count": plan["unique_rir_job_count"],
        "selected_job_count": plan["unique_rir_job_count"],
        "retained_payload_hash_verified": True,
        "acoustic_selection_binding_sha256": selection_binding[
            "binding_content_sha256"
        ],
        "acoustic_selection_mode": "registry",
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
        "acoustic_selection_binding_sha256": selection_binding[
            "binding_content_sha256"
        ],
        "acoustic_selection_mode": "registry",
    }
    write_json(cache_root / "request.json", request)
    write_json(cache_root / "receipt.json", receipt)
    write_json(cache_root / "index.json", index)
    write_json(
        cache_root / "acoustic_selection.json",
        {
            "schema": "avengine_rir_cache_acoustic_selection_sidecar_v1",
            "acoustic_selection_binding": selection_binding,
        },
    )
    session = SimpleNamespace(
        request_identity_sha256=request_identity,
        plan_sha256=plan["files"]["rir_job_plan.json"]["sha256"],
        acoustic_selection_binding=selection_binding,
        external_input_identity={
            "status": "pass",
            "acoustic_selection_binding_sha256": selection_binding[
                "binding_content_sha256"
            ],
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
    assert cache["acoustic_selection_binding"] == selection_binding
    assert (
        cache["acoustic_selection_binding_sha256"]
        == selection_binding["binding_content_sha256"]
    )
    assert "acoustic_selection.json" in cache["files"]
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

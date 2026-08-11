from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT / "tools" / "qa" / "run_strict_two_human_ground_contact_diagnostic.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("ground_contact_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _profile(runner, normalization_manifest: Path) -> dict[str, object]:
    return {
        "schema": "avengine_strict_two_human_ground_contact_release_profile_v1",
        "status": "diagnostic_pending_not_release_qualified",
        "bone_names": runner.GROUND_BONES,
        "ue_length_unit": "centimeter",
        "support_anchor_clearance_interval_cm_by_action": None,
        "minimum_individual_anchor_clearance_cm": None,
        "minimum_floor_normal_z": None,
        "runtime_visual_ground_snap": {
            "schema": "ue_dynamic_ground_snap_v1",
            "target": "attached_visual_actor_root_component",
            "maximum_abs_correction_cm": 15.0,
            "residual_tolerance_cm": 0.1,
            "timeline_anchor_mutation_allowed": False,
            "emitter_or_rir_mutation_allowed": False,
            "normalization_manifest_authority": str(normalization_manifest),
        },
    }


def _source_fixture(tmp_path: Path, monkeypatch):
    runner = _load_runner()
    repo = tmp_path / "repo"
    capture_script = repo / "tools/qa/capture_spear_native_pixel_episode.py"
    capture_script.parent.mkdir(parents=True)
    capture_script.write_text("# capture\n", encoding="utf-8")
    capture_python = tmp_path / "spear-env/bin/python"
    capture_python.parent.mkdir(parents=True)
    capture_python.write_text("", encoding="utf-8")
    spear_root = tmp_path / "SPEAR"
    spear_root.mkdir()
    diagnostic_root = repo / "tmp/ground_contact_camera_pan_v2_diagnostic_test"
    source_suite_path = diagnostic_root / "source_suite.json"
    instrumented_suite_path = diagnostic_root / "instrumented_suite.json"
    audio_path = diagnostic_root / "audio.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"RIFF")
    capture_output = diagnostic_root / "capture_attempt_01"

    actors = [
        {
            "actor_id": actor_id,
            "asset_id": f"asset_{actor_id}",
            "emitter_anchor_id": f"{actor_id}_mouth",
            "emitter_offset_m": [0.0, 0.0, 1.6],
        }
        for actor_id in runner.ACTOR_IDS
    ]
    frames = [
        {
            "frame_index": frame_index,
            "actor_states": [
                {"actor_id": actor_id, "root_position_m": [1.0, 2.0, 0.0]}
                for actor_id in runner.ACTOR_IDS
            ],
        }
        for frame_index in range(75)
    ]
    source_suite = {
        "schema": "avengine_optional_spear_apartment_suite_v1",
        "scenarios": [
            {
                "scenario_id": runner.EPISODE_ID,
                "plan": {"actors": actors, "frames": frames},
            }
        ],
    }

    asset_evidence = []
    profiles = {}
    instrumented_actors = []
    for actor in actors:
        actor_id = actor["actor_id"]
        asset_root = diagnostic_root / actor_id
        runtime_glb = asset_root / "runtime.glb"
        normalization = asset_root / "normalization_manifest.json"
        runtime_glb.parent.mkdir(parents=True)
        runtime_glb.write_bytes(b"glTF")
        _write_json(normalization, {"schema": "normalization"})
        profile = _profile(runner, normalization)
        profiles[actor_id] = profile
        instrumented_actors.append({**actor, "ground_contact_release_profile": profile})
        asset_evidence.append(
            {
                "actor_id": actor_id,
                "asset_id": actor["asset_id"],
                "asset_revision": "runtime",
                "runtime_glb": str(runtime_glb),
                "normalization_manifest": str(normalization),
                "joint_name_count": 82,
                "required_contact_bones": runner.FLAT_GROUND_BONES,
                "required_contact_bones_present": True,
                "dynamic_ground_snap_required": True,
                "maximum_abs_correction_cm": 15.0,
                "residual_tolerance_cm": 0.1,
                "socket_claim": False,
            }
        )
    instrumented_suite = {
        **source_suite,
        "ground_contact_diagnostic_mutation": {
            "schema": "avengine_strict_two_human_ground_contact_diagnostic_mutation_v1",
            "status": "cpu_materialized_pending_one_sparse_capture",
            "visual_root_dynamic_ground_snap_only": True,
            "timeline_actor_root_mutation": False,
            "emitter_or_rir_mutation": False,
            "qualification_claim": False,
            "formal": False,
        },
        "scenarios": [
            {
                "scenario_id": runner.EPISODE_ID,
                "plan": {"actors": instrumented_actors, "frames": frames},
            }
        ],
    }
    _write_json(source_suite_path, source_suite)
    _write_json(instrumented_suite_path, instrumented_suite)
    argv = [
        str(capture_script.resolve()),
        "--suite-plan",
        str(instrumented_suite_path.resolve()),
        "--scenario-id",
        runner.EPISODE_ID,
        "--audio-wav",
        str(audio_path.resolve()),
        "--spear-root",
        str(spear_root.resolve()),
        "--output",
        str(capture_output.resolve()),
        "--rpc-port",
        "39583",
        "--graphics-adapter",
        "1",
        "--frame-index",
        "0",
        "--frame-index",
        "37",
        "--frame-index",
        "74",
    ]
    source_request = {
        "schema": runner.SOURCE_REQUEST_SCHEMA,
        "status": "cpu_ready_not_authorized_for_execution",
        "scenario_id": runner.EPISODE_ID,
        "frame_indices": runner.FRAME_INDICES,
        "sample_purpose": "begin_midpoint_end_live_foot_floor_measurement",
        "gpu_launch_authorized": False,
        "formal": False,
        "qualification_claim": False,
        "one_attempt_policy": runner.SOURCE_ATTEMPT_POLICY,
        "artifacts": {
            "source_suite_plan": str(source_suite_path.resolve()),
            "instrumented_suite_plan": str(instrumented_suite_path.resolve()),
            "audio_wav": str(audio_path.resolve()),
            "spear_root": str(spear_root.resolve()),
            "capture_output": str(capture_output.resolve()),
        },
        "asset_evidence": asset_evidence,
        "diagnostic_profile_mutations": profiles,
        "measurement_contract": {
            "bone_authority": "USkeletalMeshComponent.GetBoneTransform_RTS_World",
            "floor_authority": (
                "UKismetSystemLibrary."
                "LineTraceSingleByProfile_BlockAll_complex_runtime_map"
            ),
            "actors_to_ignore": "both_runtime_anchor_and_visual_actors",
            "bone_names": runner.GROUND_BONES,
            "required_hit_fields": ["actor", "component", "location", "normal"],
            "ue_length_unit": "centimeter",
        },
        "threshold_policy": {
            "status": "must_be_derived_after_live_diagnostic",
            "actor_root_z_revision_cm": None,
            "contact_clearance_interval_cm": None,
            "bounds_only_release_forbidden": True,
            "plan_root_only_release_forbidden": True,
        },
        "capture_argv_without_python": argv,
    }
    source_path = diagnostic_root / "request.json"
    _write_json(source_path, source_request)

    monkeypatch.setattr(runner, "REPOSITORY", repo)
    monkeypatch.setattr(runner, "CAPTURE_PYTHON_LOGICAL", capture_python)
    monkeypatch.setattr(runner, "_git_head", lambda _repo: "commit-test")
    monkeypatch.setattr(runner, "_tracked_status", lambda _repo: "")
    return runner, source_path, capture_python, instrumented_suite_path


def _idle_snapshot(runner) -> dict[str, object]:
    return {
        "captured_at_utc": "2026-08-12T00:00:00Z",
        "gpus": [
            {
                "physical_index": 1,
                "uuid": runner.GPU1_UUID,
                "name": "test",
                "memory_used_mib": 0,
                "utilization_percent": 0,
            }
        ],
        "compute_apps": [],
    }


def test_prepare_and_dry_run_bind_exact_three_frames_without_launch(
    tmp_path: Path, monkeypatch
) -> None:
    runner, source_path, capture_python, _ = _source_fixture(tmp_path, monkeypatch)
    request_path = runner.prepare_request(
        source_request=source_path,
        capture_python=capture_python,
    )
    monkeypatch.setattr(runner, "_gpu_snapshot", lambda: _idle_snapshot(runner))
    monkeypatch.setattr(runner, "_assert_port_available", lambda _port: None)

    assert (
        runner.run(
            request_path,
            dry_run=True,
            authorize_one_attempt=False,
        )
        == 0
    )

    receipt = json.loads(
        (request_path.parent / "dry_run_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "dry_run_pass_not_launched"
    assert receipt["frame_indices"] == [0, 37, 74]
    assert receipt["capture_argv"].count("--frame-index") == 3
    assert receipt["physical_gpu_index"] == 1
    assert receipt["gpu_started"] is False
    assert receipt["attempt_consumed"] is False
    assert receipt["release_authorized"] is False
    assert receipt["formal_dataset_count"] == 0
    assert not Path(receipt["capture_output"]).exists()


def test_prepare_rejects_dirty_tracked_worktree(tmp_path: Path, monkeypatch) -> None:
    runner, source_path, capture_python, _ = _source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_tracked_status", lambda _repo: " M unrelated.py")

    with pytest.raises(RuntimeError, match="tracked worktree must be clean"):
        runner.prepare_request(
            source_request=source_path,
            capture_python=capture_python,
        )


def test_source_request_must_remain_cpu_unauthorized(
    tmp_path: Path, monkeypatch
) -> None:
    runner, source_path, _, _ = _source_fixture(tmp_path, monkeypatch)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["gpu_launch_authorized"] = True
    _write_json(source_path, source)

    with pytest.raises(RuntimeError, match="CPU-only/formal boundary"):
        runner._validate_source_cpu_request(source_path)


def test_source_request_rejects_timeline_or_acoustic_anchor_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    runner, source_path, _, instrumented_suite_path = _source_fixture(
        tmp_path, monkeypatch
    )
    suite = json.loads(instrumented_suite_path.read_text(encoding="utf-8"))
    suite["scenarios"][0]["plan"]["frames"][37]["actor_states"][0]["root_position_m"][
        2
    ] = 0.1
    _write_json(instrumented_suite_path, suite)

    with pytest.raises(RuntimeError, match="Timeline/acoustic frame anchors"):
        runner._validate_source_cpu_request(source_path)


def test_dry_run_rejects_busy_gpu1_without_receipt(tmp_path: Path, monkeypatch) -> None:
    runner, source_path, capture_python, _ = _source_fixture(tmp_path, monkeypatch)
    request_path = runner.prepare_request(
        source_request=source_path,
        capture_python=capture_python,
    )
    snapshot = _idle_snapshot(runner)
    snapshot["compute_apps"] = [
        {
            "gpu_uuid": runner.GPU1_UUID,
            "pid": 123,
            "process_name": "busy",
            "used_memory_mib": 1,
        }
    ]
    monkeypatch.setattr(runner, "_gpu_snapshot", lambda: snapshot)

    with pytest.raises(RuntimeError, match="GPU1 is not idle"):
        runner.run(request_path, dry_run=True, authorize_one_attempt=False)
    assert not (request_path.parent / "dry_run_receipt.json").exists()


def test_real_launch_requires_explicit_one_attempt_authorization(
    tmp_path: Path, monkeypatch
) -> None:
    runner, source_path, capture_python, _ = _source_fixture(tmp_path, monkeypatch)
    request_path = runner.prepare_request(
        source_request=source_path,
        capture_python=capture_python,
    )

    with pytest.raises(RuntimeError, match="explicitly authorize"):
        runner.run(request_path, dry_run=False, authorize_one_attempt=False)


def _ground_readback(runner) -> dict[str, object]:
    sides = {}
    for side, bones in runner.GROUND_BONES.items():
        anchors = {}
        for index, (kind, bone_name) in enumerate(bones.items()):
            clearance = 2.0 + index
            anchors[kind] = {
                "bone_name": bone_name,
                "bone_index": index,
                "world_position_ue_cm": [10.0, 20.0, 40.0 + clearance],
                "bone_to_floor_clearance_cm": clearance,
                "floor_trace": {
                    "status": "hit",
                    "profile_name": "BlockAll",
                    "trace_complex": True,
                    "hit_actor": "ApartmentFloorActor",
                    "hit_component": "ApartmentFloorComponent",
                    "hit_point_ue_cm": [10.0, 20.0, 40.0],
                    "hit_normal_ue": [0.0, 0.0, 1.0],
                },
            }
        sides[side] = {"status": "observed", "anchors": anchors}
    return {
        "schema": runner.READBACK_SCHEMA,
        "status": "pass_instrumented_measurement_only",
        "ue_length_unit": "centimeter",
        "runtime_visual_ground_snap": {
            "schema": "ue_dynamic_ground_snap_v1",
            "status": "passed",
            "target": "attached_visual_actor_root_component",
            "applied_z_correction_cm": 1.5,
            "residual_clearance_cm": 0.02,
            "maximum_timeline_anchor_error_cm": 0.0,
            "timeline_anchor_mutated": False,
            "emitter_or_rir_mutated": False,
            "bounds_role": "action_only_not_release_evidence",
        },
        "sides": sides,
    }


def _capture_fixture(tmp_path: Path, runner) -> dict[str, object]:
    capture_root = tmp_path / "capture_attempt_01"
    _write_json(
        capture_root / "manifest.json",
        {
            "schema": "avengine_qa_native_spear_pixel_episode_v1",
            "status": "pass",
            "scenario_id": runner.EPISODE_ID,
            "frame_contract": {
                "frame_count": 3,
                "formal_episode_frame_count": 75,
                "captured_frame_indices": runner.FRAME_INDICES,
            },
        },
    )
    rgb_root = capture_root / "rgb_frames"
    rgb_root.mkdir()
    for index in range(3):
        (rgb_root / f"frame_{index:06d}.png").write_bytes(b"png")
    readbacks = {
        "schema": "avengine_native_spear_runtime_asset_readbacks_v1",
        "status": "pass",
        "sampled_frames": [
            {
                "frame_index": frame_index,
                "per_instance": {
                    slot: {"live_ground_contact_readback": _ground_readback(runner)}
                    for slot in runner.INSTANCE_IDS
                },
            }
            for frame_index in runner.FRAME_INDICES
        ],
    }
    _write_json(capture_root / "runtime_asset_readbacks.json", readbacks)
    return {"capture_output": str(capture_root)}


def test_capture_validator_requires_24_live_traces_and_stays_nonrelease(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    request = _capture_fixture(tmp_path, runner)

    result = runner._validate_capture(request)

    assert result["status"] == "pass_live_measurements_manual_visual_review_pending"
    assert result["trace_count"] == 24
    assert result["bounds_only_release_forbidden"] is True
    assert result["clearance_threshold_derivation_pending"] is True
    assert result["release_authorized"] is False
    assert result["formal_dataset_count"] == 0


def test_capture_validator_rejects_missing_live_ground_field(tmp_path: Path) -> None:
    runner = _load_runner()
    request = _capture_fixture(tmp_path, runner)
    path = Path(request["capture_output"]) / "runtime_asset_readbacks.json"
    readbacks = json.loads(path.read_text(encoding="utf-8"))
    readbacks["sampled_frames"][0]["per_instance"]["source1"].pop(
        "live_ground_contact_readback"
    )
    _write_json(path, readbacks)

    with pytest.raises(RuntimeError, match="live ground readback is missing"):
        runner._validate_capture(request)

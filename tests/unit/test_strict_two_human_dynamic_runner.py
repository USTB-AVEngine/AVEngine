from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT / "tools" / "qa" / "run_strict_two_human_dynamic_full75_canary.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("dynamic_full75_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _request_fixture(tmp_path: Path, mechanism: str = "target_moves") -> Path:
    profiles = {
        "target_moves": {
            "episode_id": "strict2h_dynamic_canary_01_target_moves_v2",
            "candidate_revision": "target_moves_v2_0523_continuous_v1",
            "materialization_basename": "dynamic_target_moves_v2_materialized_v1",
            "capture_basename": "dynamic_target_moves_v2_capture_attempt_01",
            "moving_source_slot": "source1",
            "native_source_scenario_id": (
                "human_border_collie__recombined_both_moving_0523"
            ),
            "expected_rir_count_by_source_slot": {"source1": 75, "source2": 1},
            "expected_unique_rir_job_count": 76,
        },
        "distractor_moves": {
            "episode_id": "strict2h_dynamic_canary_02_distractor_moves_v2",
            "candidate_revision": "distractor_moves_v2_0589_continuous_v1",
            "materialization_basename": "dynamic_distractor_moves_v2_materialized_v1",
            "capture_basename": "dynamic_distractor_moves_v2_capture_attempt_01",
            "moving_source_slot": "source2",
            "native_source_scenario_id": (
                "human_border_collie__recombined_both_moving_0589"
            ),
            "expected_rir_count_by_source_slot": {"source1": 1, "source2": 75},
            "expected_unique_rir_job_count": 76,
        },
        "camera_pan_both_static": {
            "episode_id": "strict2h_dynamic_canary_04_camera_pan_both_static_v2",
            "candidate_revision": "camera_pan_v2_0589_right_target_yaw52_58_v1",
            "materialization_basename": "dynamic_camera_pan_v2_materialized_v1",
            "capture_basename": "dynamic_camera_pan_v2_capture_attempt_01",
            "moving_source_slot": None,
            "native_source_scenario_id": None,
            "expected_rir_count_by_source_slot": {"source1": 75, "source2": 75},
            "expected_unique_rir_job_count": 150,
        },
    }
    profile = profiles[mechanism]
    episode_id = profile["episode_id"]
    root = tmp_path / profile["materialization_basename"]
    finalization_path = root / "pre_capture_finalization_v1" / "finalization.json"
    suite_path = root / "suite_execution_plan.json"
    audio_path = root / "binaural_v1" / "audio" / "binaural" / f"{episode_id}__v00.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"RIFF")
    _write_json(
        suite_path,
        {
            "schema": "avengine_optional_spear_apartment_suite_v1",
            "scenarios": [{"scenario_id": episode_id}],
        },
    )
    _write_json(
        finalization_path,
        {
            "schema": "avengine_native_strict_two_human_dynamic_full75_finalization_v1",
            "status": "pass_cpu_ready_for_gpu1",
            "cpu_pre_capture_gate_pass": True,
            "gpu_launch_authorized": True,
            "qualification_claim": False,
            "episode_id": episode_id,
            "mechanism": mechanism,
            "artifacts": {"materialization_root": str(root)},
            "materialization": {
                "status": "pass",
                "frame_count": 75,
                "requested_source_frame_uses": 150,
                "expected_unique_rir_job_count": profile[
                    "expected_unique_rir_job_count"
                ],
                "expected_rir_count_by_source_slot": profile[
                    "expected_rir_count_by_source_slot"
                ],
                "animation_timing": (
                    {}
                    if profile["moving_source_slot"] is None
                    else {
                        profile["moving_source_slot"]: {
                            "mode": "arc_length_preserving_native_stride_v1",
                            "path_provenance": {
                                "native_source_scenario_id": (
                                    profile["native_source_scenario_id"]
                                ),
                                "output_root_count": 75,
                                "output_unique_root_count_at_1mm": 75,
                                "interior_output_roots_exact_native_frame_readbacks": False,
                            },
                        }
                    }
                ),
                "action_counts": {
                    "source1": {"idle": 75, "walk": 0},
                    "source2": {"idle": 75, "walk": 0},
                },
                "distinct_listener_orientation_count": 75,
                "camera_yaw_span_deg": 6.0,
            },
        },
    )
    request_path = root / "gpu_launch_attempt_01" / "request.json"
    _write_json(
        request_path,
        {
            "schema": "avengine_native_strict_two_human_dynamic_full75_gpu_launch_request_v2",
            "episode_id": episode_id,
            "mechanism": mechanism,
            "candidate_revision": profile["candidate_revision"],
            "attempt_policy": {
                "attempt_index": 1,
                "maximum_attempts_for_candidate": 1,
                "retry_same_candidate_forbidden": True,
                "failure_disposition": "reject_candidate_without_same_candidate_retry",
            },
            "repo_root": str(REPO_ROOT),
            "pre_capture_finalization": str(finalization_path),
            "capture_python": "/data/jzy/miniconda3/envs/spear-env/bin/python",
            "capture_script": str(
                REPO_ROOT / "tools" / "qa" / "capture_spear_native_pixel_episode.py"
            ),
            "suite_plan": str(suite_path),
            "audio_wav": str(audio_path),
            "spear_root": "/data/jzy/code/SPEAR-lead-b",
            "capture_output": str(root.parent / profile["capture_basename"]),
            "rpc_port": 39701,
            "physical_gpu_index": 1,
            "graphics_adapter_argument": 1,
            "forbidden_physical_gpu_indices": [0, 3],
            "required_idle_compute_process_count": 0,
        },
    )
    return request_path


def test_dynamic_runner_dry_run_persists_gpu1_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    request_path = _request_fixture(tmp_path)
    monkeypatch.setattr(
        runner,
        "_gpu_snapshot",
        lambda: {
            "gpus": [
                {
                    "physical_index": 1,
                    "uuid": "GPU-6d3e273e-58c6-2a5b-480a-4816fef6c581",
                    "name": "test",
                }
            ],
            "compute_apps": [],
        },
    )
    receipt_path = request_path.parent / "dry_run_receipt.json"
    assert runner.run(request_path, receipt_path, dry_run=True) == 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "dry_run_pass"
    assert receipt["physical_gpu_index"] == 1
    assert "--frame-index" not in receipt["capture_argv"]


def test_dynamic_runner_rejects_unsupported_mechanism(tmp_path: Path) -> None:
    runner = _load_runner()
    request_path = _request_fixture(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["mechanism"] = "both_move"
    _write_json(request_path, request)
    with pytest.raises(RuntimeError, match="target_moves, distractor_moves"):
        runner._validate_request(request_path)


def test_dynamic_runner_distractor_dry_run_binds_source2_motion(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    request_path = _request_fixture(tmp_path, "distractor_moves")
    monkeypatch.setattr(
        runner,
        "_gpu_snapshot",
        lambda: {
            "gpus": [
                {
                    "physical_index": 1,
                    "uuid": runner.GPU1_UUID,
                    "name": "test",
                }
            ],
            "compute_apps": [],
        },
    )
    receipt_path = request_path.parent / "dry_run_receipt.json"
    assert runner.run(request_path, receipt_path, dry_run=True) == 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "dry_run_pass"
    assert receipt["mechanism"] == "distractor_moves"
    assert receipt["candidate_revision"] == "distractor_moves_v2_0589_continuous_v1"
    assert receipt["capture_output"].endswith(
        "/dynamic_distractor_moves_v2_capture_attempt_01"
    )
    assert "--frame-index" not in receipt["capture_argv"]


def test_dynamic_runner_camera_pan_dry_run_binds_static_actors_and_150_rirs(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    request_path = _request_fixture(tmp_path, "camera_pan_both_static")
    monkeypatch.setattr(
        runner,
        "_gpu_snapshot",
        lambda: {
            "gpus": [
                {
                    "physical_index": 1,
                    "uuid": runner.GPU1_UUID,
                    "name": "test",
                }
            ],
            "compute_apps": [],
        },
    )
    receipt_path = request_path.parent / "dry_run_receipt.json"

    assert runner.run(request_path, receipt_path, dry_run=True) == 0

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "dry_run_pass"
    assert receipt["mechanism"] == "camera_pan_both_static"
    assert receipt["capture_output"].endswith(
        "/dynamic_camera_pan_v2_capture_attempt_01"
    )
    assert "--frame-index" not in receipt["capture_argv"]


def test_dynamic_runner_rejects_camera_pan_orientation_drift(tmp_path: Path) -> None:
    runner = _load_runner()
    request_path = _request_fixture(tmp_path, "camera_pan_both_static")
    finalization_path = Path(
        json.loads(request_path.read_text(encoding="utf-8"))["pre_capture_finalization"]
    )
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    finalization["materialization"]["distinct_listener_orientation_count"] = 74
    _write_json(finalization_path, finalization)

    with pytest.raises(RuntimeError, match="listener orientation authority drift"):
        runner._validate_request(request_path)


def test_dynamic_runner_rejects_distractor_rir_slot_drift(tmp_path: Path) -> None:
    runner = _load_runner()
    request_path = _request_fixture(tmp_path, "distractor_moves")
    finalization_path = Path(
        json.loads(request_path.read_text(encoding="utf-8"))["pre_capture_finalization"]
    )
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    finalization["materialization"]["expected_rir_count_by_source_slot"] = {
        "source1": 75,
        "source2": 1,
    }
    _write_json(finalization_path, finalization)
    with pytest.raises(RuntimeError, match="distractor_moves RIR slot counts drifted"):
        runner._validate_request(request_path)


def test_dynamic_runner_rejects_noncontinuous_episode(tmp_path: Path) -> None:
    runner = _load_runner()
    request_path = _request_fixture(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["episode_id"] = "strict2h_dynamic_canary_01_target_moves_v1"
    _write_json(request_path, request)
    with pytest.raises(RuntimeError, match="continuous target_moves v2"):
        runner._validate_request(request_path)


def test_dynamic_runner_rejects_attempt_policy_drift(tmp_path: Path) -> None:
    runner = _load_runner()
    request_path = _request_fixture(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["attempt_policy"]["maximum_attempts_for_candidate"] = 2
    _write_json(request_path, request)
    with pytest.raises(RuntimeError, match="attempt policy drift"):
        runner._validate_request(request_path)


def test_dynamic_runner_rejects_capture_output_drift(tmp_path: Path) -> None:
    runner = _load_runner()
    request_path = _request_fixture(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["capture_output"] = str(tmp_path / "fresh_but_wrong")
    _write_json(request_path, request)
    with pytest.raises(RuntimeError, match="capture output path drift"):
        runner._validate_request(request_path)


def test_dynamic_runner_rejects_receipt_path_drift(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    request_path = _request_fixture(tmp_path)
    monkeypatch.setattr(
        runner,
        "_gpu_snapshot",
        lambda: {
            "gpus": [
                {
                    "physical_index": 1,
                    "uuid": runner.GPU1_UUID,
                    "name": "test",
                }
            ],
            "compute_apps": [],
        },
    )
    with pytest.raises(RuntimeError, match="receipt path is not bound"):
        runner.run(request_path, tmp_path / "wrong.json", dry_run=True)


def test_dynamic_runner_failed_real_attempt_is_persisted_and_cannot_retry(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    request_path = _request_fixture(tmp_path)
    monkeypatch.setattr(
        runner,
        "_gpu_snapshot",
        lambda: {
            "gpus": [
                {
                    "physical_index": 1,
                    "uuid": runner.GPU1_UUID,
                    "name": "test",
                }
            ],
            "compute_apps": [],
        },
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=17),
    )
    receipt_path = request_path.parent / "launch_receipt.json"
    assert runner.run(request_path, receipt_path, dry_run=False) == 17
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "fail"
    assert receipt["capture_process_exit_code"] == 17
    assert receipt["attempt_policy"]["retry_same_candidate_forbidden"] is True
    with pytest.raises(RuntimeError, match="launch receipt must be new"):
        runner.run(request_path, receipt_path, dry_run=False)

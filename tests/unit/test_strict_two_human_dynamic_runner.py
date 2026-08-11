from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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


def _request_fixture(tmp_path: Path) -> Path:
    episode_id = "strict2h_dynamic_canary_01_target_moves_v1"
    root = tmp_path / "dynamic_target_moves_materialized_v4"
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
            "mechanism": "target_moves",
            "artifacts": {"materialization_root": str(root)},
            "materialization": {
                "status": "pass",
                "frame_count": 75,
                "requested_source_frame_uses": 150,
                "expected_unique_rir_job_count": 76,
                "expected_rir_count_by_source_slot": {"source1": 75, "source2": 1},
            },
        },
    )
    request_path = root / "gpu_launch_v1" / "request.json"
    _write_json(
        request_path,
        {
            "schema": "avengine_native_strict_two_human_dynamic_full75_gpu_launch_request_v1",
            "episode_id": episode_id,
            "mechanism": "target_moves",
            "repo_root": str(REPO_ROOT),
            "pre_capture_finalization": str(finalization_path),
            "capture_python": "/data/jzy/miniconda3/envs/spear-env/bin/python",
            "capture_script": str(
                REPO_ROOT / "tools" / "qa" / "capture_spear_native_pixel_episode.py"
            ),
            "suite_plan": str(suite_path),
            "audio_wav": str(audio_path),
            "spear_root": "/data/jzy/code/SPEAR-lead-b",
            "capture_output": str(root.parent / "dynamic_target_moves_capture_v1"),
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
    receipt_path = tmp_path / "dry_receipt.json"
    assert runner.run(request_path, receipt_path, dry_run=True) == 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "dry_run_pass"
    assert receipt["physical_gpu_index"] == 1
    assert "--frame-index" not in receipt["capture_argv"]


def test_dynamic_runner_rejects_non_target_mechanism(tmp_path: Path) -> None:
    runner = _load_runner()
    request_path = _request_fixture(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["mechanism"] = "both_move"
    _write_json(request_path, request)
    with pytest.raises(RuntimeError, match="target_moves-only"):
        runner._validate_request(request_path)

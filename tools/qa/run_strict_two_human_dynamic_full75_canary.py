#!/usr/bin/env python3
"""Launch one CPU-qualified dynamic full75 canary on physical GPU1 only."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUEST_SCHEMA = "avengine_native_strict_two_human_dynamic_full75_gpu_launch_request_v2"
RECEIPT_SCHEMA = "avengine_native_strict_two_human_dynamic_full75_gpu_launch_receipt_v2"
FINALIZATION_SCHEMA = "avengine_native_strict_two_human_dynamic_full75_finalization_v1"
GPU1_UUID = "GPU-6d3e273e-58c6-2a5b-480a-4816fef6c581"
ATTEMPT_POLICY = {
    "attempt_index": 1,
    "maximum_attempts_for_candidate": 1,
    "retry_same_candidate_forbidden": True,
    "failure_disposition": "reject_candidate_without_same_candidate_retry",
}
RUNNER_PROFILES = {
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _nvidia_csv(query: str) -> list[list[str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{query.split(':', 1)[0]}={query.split(':', 1)[1]}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return [
        [field.strip() for field in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def _gpu_snapshot() -> dict[str, Any]:
    gpus = _nvidia_csv("gpu:index,uuid,name")
    apps = _nvidia_csv("compute-apps:gpu_uuid,pid,process_name")
    return {
        "gpus": [
            {"physical_index": int(index), "uuid": uuid, "name": name}
            for index, uuid, name in gpus
        ],
        "compute_apps": [
            {"gpu_uuid": uuid, "pid": int(pid), "process_name": name}
            for uuid, pid, name in apps
        ],
    }


def _capture_argv(request: dict[str, Any]) -> list[str]:
    return [
        str(request["capture_python"]),
        str(request["capture_script"]),
        "--suite-plan",
        str(request["suite_plan"]),
        "--scenario-id",
        str(request["episode_id"]),
        "--audio-wav",
        str(request["audio_wav"]),
        "--spear-root",
        str(request["spear_root"]),
        "--output",
        str(request["capture_output"]),
        "--rpc-port",
        str(request["rpc_port"]),
        "--graphics-adapter",
        "1",
    ]


def _validate_request(request_path: Path) -> tuple[dict[str, Any], list[str]]:
    request = _load(request_path)
    _require(
        request.get("schema") == REQUEST_SCHEMA, "dynamic launch request schema drift"
    )
    mechanism = request.get("mechanism")
    _require(
        mechanism in RUNNER_PROFILES,
        "runner mechanism must be target_moves, distractor_moves, or "
        "camera_pan_both_static",
    )
    profile = RUNNER_PROFILES[mechanism]
    _require(
        request.get("episode_id") == profile["episode_id"],
        f"runner is bound to the continuous {mechanism} v2 episode",
    )
    _require(
        request.get("candidate_revision") == profile["candidate_revision"],
        "candidate revision drift",
    )
    _require(request.get("attempt_policy") == ATTEMPT_POLICY, "attempt policy drift")
    _require(request.get("physical_gpu_index") == 1, "physical GPU must be index 1")
    _require(
        request.get("graphics_adapter_argument") == 1, "graphics adapter must be 1"
    )
    _require(
        request.get("forbidden_physical_gpu_indices") == [0, 3],
        "forbidden GPU policy drift",
    )
    _require(
        request.get("required_idle_compute_process_count") == 0, "GPU1 must be idle"
    )

    finalization_path = Path(request["pre_capture_finalization"]).resolve()
    finalization = _load(finalization_path)
    _require(
        finalization.get("schema") == FINALIZATION_SCHEMA, "finalizer schema drift"
    )
    _require(
        finalization.get("status") == "pass_cpu_ready_for_gpu1", "CPU gate did not pass"
    )
    _require(
        finalization.get("cpu_pre_capture_gate_pass") is True, "CPU gate flag is false"
    )
    _require(
        finalization.get("gpu_launch_authorized") is True,
        "GPU launch is not authorized",
    )
    _require(
        finalization.get("qualification_claim") is False, "pre-capture cannot qualify"
    )
    _require(
        finalization.get("episode_id") == request.get("episode_id"), "episode mismatch"
    )
    _require(
        finalization.get("mechanism") == request.get("mechanism"), "mechanism mismatch"
    )
    materialization = finalization.get("materialization", {})
    _require(materialization.get("status") == "pass", "materialization did not pass")
    _require(
        materialization.get("frame_count") == 75, "materialization must have 75 frames"
    )
    _require(
        materialization.get("requested_source_frame_uses") == 150,
        "materialization must have 150 source-frame uses",
    )
    _require(
        materialization.get("expected_unique_rir_job_count")
        == profile["expected_unique_rir_job_count"],
        f"{mechanism} exact RIR job count drifted",
    )
    _require(
        materialization.get("expected_rir_count_by_source_slot")
        == profile["expected_rir_count_by_source_slot"],
        f"{mechanism} RIR slot counts drifted",
    )
    if profile["moving_source_slot"] is not None:
        timing = materialization.get("animation_timing", {}).get(
            profile["moving_source_slot"], {}
        )
        provenance = timing.get("path_provenance", {})
        _require(
            timing.get("mode") == "arc_length_preserving_native_stride_v1",
            "continuous slow-walk timing authority drift",
        )
        _require(
            provenance.get("native_source_scenario_id")
            == profile["native_source_scenario_id"],
            "native source scenario drift",
        )
        _require(
            provenance.get("output_root_count") == 75
            and provenance.get("output_unique_root_count_at_1mm") == 75,
            "continuous v2 root uniqueness drift",
        )
        _require(
            provenance.get("interior_output_roots_exact_native_frame_readbacks")
            is False,
            "derived-interpolation claim boundary drift",
        )
    else:
        _require(
            materialization.get("animation_timing") == {}
            and materialization.get("action_counts")
            == {
                "source1": {"idle": 75, "walk": 0},
                "source2": {"idle": 75, "walk": 0},
            },
            "camera-pan actors must remain static and idle",
        )
        _require(
            materialization.get("distinct_listener_orientation_count") == 75
            and float(materialization.get("camera_yaw_span_deg", 0.0)) >= 5.9,
            "camera-pan listener orientation authority drift",
        )

    materialization_root = Path(
        finalization["artifacts"]["materialization_root"]
    ).resolve()
    _require(
        materialization_root.name == profile["materialization_basename"],
        f"materialization root is not the {mechanism} v2 authority",
    )
    _require(
        finalization_path
        == materialization_root / "pre_capture_finalization_v1" / "finalization.json",
        "pre-capture finalization is not bound to its materialization root",
    )
    _require(
        request_path.resolve()
        == materialization_root / "gpu_launch_attempt_01" / "request.json",
        "launch request is not bound to attempt 01 under the materialization root",
    )
    suite_path = Path(request["suite_plan"]).resolve()
    audio_path = Path(request["audio_wav"]).resolve()
    capture_output = Path(request["capture_output"]).resolve()
    _require(
        suite_path == materialization_root / "suite_execution_plan.json",
        "suite plan is not the materialized v2 authority",
    )
    _require(
        audio_path
        == materialization_root
        / "binaural_v1"
        / "audio"
        / "binaural"
        / f"{request['episode_id']}__v00.wav",
        "audio is not the authoritative materialized binaural mixture",
    )
    _require(
        capture_output == materialization_root.parent / profile["capture_basename"],
        "capture output path drift",
    )

    repo_root = Path(request["repo_root"]).resolve()
    _require(repo_root == Path(__file__).resolve().parents[2], "repo root drift")
    _require(
        Path(request["capture_script"]).resolve()
        == repo_root / "tools" / "qa" / "capture_spear_native_pixel_episode.py",
        "capture script drift",
    )
    _require(
        Path(request["capture_python"])
        == Path("/data/jzy/miniconda3/envs/spear-env/bin/python"),
        "capture Python drift",
    )
    _require(
        Path(request["spear_root"]) == Path("/data/jzy/code/SPEAR-lead-b"),
        "SPEAR root drift",
    )

    for key in (
        "capture_python",
        "capture_script",
        "suite_plan",
        "audio_wav",
        "spear_root",
    ):
        _require(
            Path(request[key]).exists(), f"missing launch input {key}: {request[key]}"
        )
    suite = _load(suite_path)
    _require(
        suite.get("schema") == "avengine_optional_spear_apartment_suite_v1",
        "suite schema drift",
    )
    scenarios = suite.get("scenarios", [])
    _require(
        len(scenarios) == 1, "dynamic canary suite must contain exactly one scenario"
    )
    matches = [
        item for item in scenarios if item.get("scenario_id") == request["episode_id"]
    ]
    _require(len(matches) == 1, "episode must resolve to exactly one suite scenario")
    _require(int(request["rpc_port"]) == 39701, "RPC port drift")
    _require(not capture_output.exists(), "capture output must be new")

    argv = _capture_argv(request)
    _require(
        "--frame-index" not in argv, "dynamic full75 cannot use sparse frame selector"
    )
    _require(argv[argv.index("--graphics-adapter") + 1] == "1", "adapter must be GPU1")
    for flag in (
        "--suite-plan",
        "--scenario-id",
        "--audio-wav",
        "--spear-root",
        "--output",
        "--rpc-port",
        "--graphics-adapter",
    ):
        _require(argv.count(flag) == 1, f"capture flag must occur exactly once: {flag}")
    return request, argv


def run(request_path: Path, receipt_path: Path, *, dry_run: bool) -> int:
    _require(not receipt_path.exists(), "launch receipt must be new")
    request, argv = _validate_request(request_path)
    materialization_root = (
        Path(request["pre_capture_finalization"]).resolve().parents[1]
    )
    attempt_root = materialization_root / "gpu_launch_attempt_01"
    expected_receipt_path = attempt_root / (
        "dry_run_receipt.json" if dry_run else "launch_receipt.json"
    )
    _require(
        receipt_path.resolve() == expected_receipt_path,
        "receipt path is not bound to attempt 01",
    )
    real_receipt_path = attempt_root / "launch_receipt.json"
    if dry_run:
        _require(
            not real_receipt_path.exists(),
            "candidate already has a real launch receipt; dry-run replay is forbidden",
        )
    before = _gpu_snapshot()
    gpu1 = [item for item in before["gpus"] if item["physical_index"] == 1]
    _require(len(gpu1) == 1, "physical GPU1 did not resolve exactly once")
    gpu1_uuid = gpu1[0]["uuid"]
    _require(gpu1_uuid == GPU1_UUID, f"physical GPU1 UUID drift: {gpu1_uuid}")
    gpu1_apps = [
        item for item in before["compute_apps"] if item["gpu_uuid"] == gpu1_uuid
    ]
    _require(len(gpu1_apps) == 0, f"physical GPU1 is not idle: {gpu1_apps}")

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "dry_run_pass" if dry_run else "running",
        "episode_id": request["episode_id"],
        "mechanism": request["mechanism"],
        "candidate_revision": request["candidate_revision"],
        "attempt_policy": request["attempt_policy"],
        "physical_gpu_index": 1,
        "graphics_adapter_argument": 1,
        "forbidden_physical_gpu_indices_used": [],
        "gpu1_uuid": gpu1_uuid,
        "prelaunch_snapshot": before,
        "request": str(request_path),
        "pre_capture_finalization": request["pre_capture_finalization"],
        "capture_argv": argv,
        "capture_output": request["capture_output"],
        "capture_process_exit_code": None,
        "started_at_utc": _utc_now(),
        "ended_at_utc": None,
    }
    _write(receipt_path, receipt)
    if dry_run:
        return 0

    exit_code = 1
    try:
        completed = subprocess.run(argv, cwd=request["repo_root"], check=False)
        exit_code = int(completed.returncode)
        receipt["capture_process_exit_code"] = exit_code
        receipt["status"] = "pass" if exit_code == 0 else "fail"
    except (OSError, subprocess.SubprocessError) as exc:
        receipt["status"] = "error"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["ended_at_utc"] = _utc_now()
        try:
            receipt["postlaunch_snapshot"] = _gpu_snapshot()
        except (OSError, subprocess.SubprocessError, ValueError) as snapshot_exc:
            receipt["postlaunch_snapshot_error"] = (
                f"{type(snapshot_exc).__name__}: {snapshot_exc}"
            )
        _write(receipt_path, receipt)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(args.request.resolve(), args.receipt.resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

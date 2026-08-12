#!/usr/bin/env python3
"""Launch one CPU-qualified dynamic full75 canary on physical GPU1 only."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.qa.actor_motion_profile import validate_actor_motion_profile

REQUEST_SCHEMA = "avengine_native_strict_two_human_dynamic_full75_gpu_launch_request_v3"
LEGACY_REQUEST_SCHEMA = (
    "avengine_native_strict_two_human_dynamic_full75_gpu_launch_request_v2"
)
RECEIPT_SCHEMA = "avengine_native_strict_two_human_dynamic_full75_gpu_launch_receipt_v2"
FINALIZATION_SCHEMA = "avengine_native_strict_two_human_dynamic_full75_finalization_v1"
MOTION_AUTHORITY_SCHEMA = "avengine_actor_motion_profile_launch_authority_v1"
LEGACY_CAMERA_PAN_MECHANISM = "camera_pan_both_static"
GPU1_UUID = "GPU-6d3e273e-58c6-2a5b-480a-4816fef6c581"
ATTEMPT_POLICY = {
    "attempt_index": 1,
    "maximum_attempts_for_candidate": 1,
    "retry_same_candidate_forbidden": True,
    "failure_disposition": "reject_candidate_without_same_candidate_retry",
}


def _as_mapping(value: object, message: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), message)
    return value


def _load_motion_profile(path: Path) -> dict[str, Any]:
    profile = _load(path)
    validate_actor_motion_profile(profile)
    return profile


# The sole profile-less compatibility path. Profile-backed mechanisms are
# deliberately absent: their identity, actions, acoustics, and output naming are
# supplied by immutable artifacts instead of runner code.
LEGACY_CAMERA_PAN_ADAPTER = {
    "episode_id": "strict2h_dynamic_canary_04_camera_pan_both_static_v2",
    "materialization_basename": "dynamic_camera_pan_v2_materialized_v1",
    "capture_basename": "dynamic_camera_pan_v2_capture_attempt_01",
    "expected_rir_count_by_source_slot": {"source1": 75, "source2": 75},
    "expected_unique_rir_job_count": 150,
    "expected_action_counts": {
        "source1": {"idle": 75, "walk": 0},
        "source2": {"idle": 75, "walk": 0},
    },
    "expected_listener_orientation_count": 75,
    "minimum_camera_yaw_span_deg": 5.9,
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


def _profile_action_counts(profile: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    authorities = _as_mapping(profile.get("authorities"), "profile authorities missing")
    candidate_binding = _as_mapping(
        authorities.get("candidate"), "profile candidate binding missing"
    )
    candidate = _as_mapping(
        candidate_binding.get("value"), "profile candidate value missing"
    )
    actors = _as_mapping(candidate.get("actors"), "profile actors missing")
    frames = profile.get("frames")
    _require(isinstance(frames, list) and bool(frames), "profile frames missing")
    slots = [str(slot) for slot in actors]
    actions = sorted(
        {
            str(state.get("action_id"))
            for frame in frames
            for state in _as_mapping(frame, "profile frame invalid").get(
                "actor_states", []
            )
            if isinstance(state, Mapping) and isinstance(state.get("action_id"), str)
        }
    )
    _require(bool(actions), "profile actions missing")
    counts = {slot: {action: 0 for action in actions} for slot in slots}
    for frame in frames:
        states = _as_mapping(frame, "profile frame invalid").get("actor_states")
        _require(isinstance(states, list), "profile actor states missing")
        observed_slots: list[str] = []
        for state_value in states:
            state = _as_mapping(state_value, "profile actor state invalid")
            slot = state.get("slot_id")
            action = state.get("action_id")
            _require(
                isinstance(slot, str)
                and slot in counts
                and isinstance(action, str)
                and action in counts[slot],
                "profile actor/action identity drift",
            )
            observed_slots.append(slot)
            counts[slot][action] += 1
        _require(observed_slots == slots, "profile frame actor order drift")
    return counts


def _validate_motion_profile_chain(
    *,
    request: Mapping[str, Any],
    finalization: Mapping[str, Any],
    materialization: Mapping[str, Any],
    materialization_root: Path,
    materialization_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _as_mapping(
        request.get("motion_authority"), "motion_authority is required"
    )
    _require(
        authority.get("schema") == MOTION_AUTHORITY_SCHEMA,
        "motion authority schema drift",
    )
    profile_path = Path(str(authority.get("path", ""))).resolve()
    _require(
        profile_path == materialization_root / "actor_motion_profile.json",
        "motion authority is not the materialization profile",
    )
    profile = _load_motion_profile(profile_path)
    profile_file_sha256 = sha256_file(profile_path)
    profile_document_sha256 = canonical_json_sha256(profile)
    profile_content_sha256 = profile.get("profile_content_sha256")
    _require(
        authority.get("file_sha256") == profile_file_sha256
        and authority.get("profile_content_sha256") == profile_content_sha256,
        "request motion authority hash drift",
    )

    authorities = _as_mapping(profile.get("authorities"), "profile authorities missing")
    candidate_binding = _as_mapping(
        authorities.get("candidate"), "profile candidate binding missing"
    )
    candidate = _as_mapping(
        candidate_binding.get("value"), "profile candidate value missing"
    )
    episode_id = candidate.get("legacy_episode_id")
    mechanism = candidate.get("mechanism")
    _require(
        isinstance(episode_id, str)
        and bool(episode_id)
        and isinstance(mechanism, str)
        and bool(mechanism),
        "profile episode/mechanism identity missing",
    )
    _require(
        request.get("episode_id") == episode_id
        and request.get("mechanism") == mechanism
        and finalization.get("episode_id") == episode_id
        and finalization.get("mechanism") == mechanism
        and materialization_receipt.get("episode_id") == episode_id
        and materialization_receipt.get("mechanism") == mechanism,
        "request/finalization/receipt/profile identity drift",
    )

    action_counts = _profile_action_counts(profile)
    rir_expectation = _as_mapping(
        profile.get("rir_expectation"), "profile RIR expectation missing"
    )
    final_motion = _as_mapping(
        materialization.get("motion_profile"),
        "pre-finalization motion-profile closure missing",
    )
    receipt_motion = _as_mapping(
        materialization_receipt.get("actor_motion_profile"),
        "materialization receipt motion-profile closure missing",
    )
    rir_plan = _as_mapping(
        final_motion.get("rir_plan"), "pre-finalization profile RIR plan missing"
    )
    expected_rir = {
        "stride_frames": rir_expectation.get("stride_frames"),
        "requested_pair_state_count": rir_expectation.get("requested_pair_state_count"),
        "unique_rir_job_count": rir_expectation.get("unique_rir_job_count"),
    }
    _require(
        final_motion.get("status") == "pass_hash_bound_profile_consumed_exactly"
        and final_motion.get("profile_file_sha256") == profile_file_sha256
        and final_motion.get("profile_document_canonical_sha256")
        == profile_document_sha256
        and final_motion.get("profile_content_sha256") == profile_content_sha256
        and final_motion.get("candidate_document_sha256")
        == candidate_binding.get("document_sha256")
        and final_motion.get("candidate_value_sha256")
        == candidate_binding.get("canonical_value_sha256")
        and final_motion.get("action_counts") == action_counts,
        "pre-finalization/profile motion authority drift",
    )
    _require(
        receipt_motion.get("status") == "pass_bound_and_consumed_frame_by_frame"
        and receipt_motion.get("schema") == profile.get("schema")
        and receipt_motion.get("profile_content_sha256") == profile_content_sha256
        and receipt_motion.get("candidate_document_sha256")
        == candidate_binding.get("document_sha256")
        and receipt_motion.get("candidate_value_sha256")
        == candidate_binding.get("canonical_value_sha256")
        and receipt_motion.get("derived_action_counts") == action_counts
        and receipt_motion.get("derived_rir_counts") == expected_rir
        and receipt_motion.get("legacy_root_motion_inference_used") is False
        and receipt_motion.get("qualification_claim") is False,
        "materialization receipt/profile motion authority drift",
    )
    _require(
        materialization.get("action_counts") == action_counts
        and materialization.get("rir_stride_frames") == expected_rir["stride_frames"]
        and materialization.get("requested_source_frame_uses")
        == expected_rir["requested_pair_state_count"]
        and materialization.get("expected_unique_rir_job_count")
        == expected_rir["unique_rir_job_count"],
        "pre-finalization action/RIR derivation drift",
    )
    per_slot = rir_plan.get("distinct_rir_state_count_by_source_slot")
    _require(
        isinstance(per_slot, Mapping)
        and materialization.get("expected_rir_count_by_source_slot") == per_slot
        and rir_plan.get("stride_frames") == expected_rir["stride_frames"]
        and rir_plan.get("requested_pair_state_count")
        == expected_rir["requested_pair_state_count"]
        and rir_plan.get("unique_rir_job_count")
        == expected_rir["unique_rir_job_count"],
        "pre-finalization/profile RIR plan drift",
    )
    return {
        "status": "pass_request_finalization_receipt_profile_chain",
        "schema": profile.get("schema"),
        "path": str(profile_path),
        "file_sha256": profile_file_sha256,
        "profile_document_canonical_sha256": profile_document_sha256,
        "profile_content_sha256": profile_content_sha256,
        "candidate_document_sha256": candidate_binding.get("document_sha256"),
        "candidate_value_sha256": candidate_binding.get("canonical_value_sha256"),
        "candidate_episode_id": candidate.get("candidate_episode_id"),
        "episode_id": episode_id,
        "mechanism": mechanism,
        "action_counts": action_counts,
        "rir": {
            **expected_rir,
            "distinct_rir_state_count_by_source_slot": dict(per_slot),
        },
        "legacy_adapter_used": False,
        "qualification_claim": False,
    }


def _validate_legacy_camera_pan_chain(
    *,
    request: Mapping[str, Any],
    materialization: Mapping[str, Any],
    materialization_root: Path,
    materialization_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    adapter = LEGACY_CAMERA_PAN_ADAPTER
    _require(
        request.get("mechanism") == LEGACY_CAMERA_PAN_MECHANISM,
        "missing motion profile is allowed only for camera_pan_both_static",
    )
    _require(
        request.get("motion_authority") is None
        and not (materialization_root / "actor_motion_profile.json").exists(),
        "legacy camera-pan cannot carry a partial motion authority",
    )
    _require(
        request.get("episode_id") == adapter["episode_id"]
        and materialization_root.name == adapter["materialization_basename"],
        "legacy camera-pan identity drift",
    )
    _require(
        materialization.get("motion_profile")
        == {
            "status": "explicit_legacy_camera_pan_adapter",
            "live_skeletal_transition_evidence": {
                "status": "not_applicable_static_actors"
            },
        }
        and materialization_receipt.get("actor_motion_profile")
        == {
            "status": "explicit_legacy_camera_pan_adapter",
            "legacy_root_motion_inference_used": True,
            "qualification_claim": False,
        },
        "legacy camera-pan adapter closure drift",
    )
    _require(
        materialization.get("action_counts") == adapter["expected_action_counts"]
        and materialization.get("expected_unique_rir_job_count")
        == adapter["expected_unique_rir_job_count"]
        and materialization.get("expected_rir_count_by_source_slot")
        == adapter["expected_rir_count_by_source_slot"]
        and materialization.get("distinct_listener_orientation_count")
        == adapter["expected_listener_orientation_count"]
        and float(materialization.get("camera_yaw_span_deg", 0.0))
        >= adapter["minimum_camera_yaw_span_deg"],
        "legacy camera-pan action/RIR/orientation drift",
    )
    return {
        "status": "explicit_legacy_camera_pan_adapter",
        "episode_id": request["episode_id"],
        "mechanism": request["mechanism"],
        "action_counts": adapter["expected_action_counts"],
        "rir": {
            "unique_rir_job_count": adapter["expected_unique_rir_job_count"],
            "distinct_rir_state_count_by_source_slot": adapter[
                "expected_rir_count_by_source_slot"
            ],
        },
        "legacy_adapter_used": True,
        "qualification_claim": False,
    }


def _validate_request(
    request_path: Path,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    request = _load(request_path)
    mechanism = request.get("mechanism")
    if mechanism == LEGACY_CAMERA_PAN_MECHANISM:
        _require(
            request.get("schema") in {REQUEST_SCHEMA, LEGACY_REQUEST_SCHEMA},
            "legacy camera-pan launch request schema drift",
        )
    else:
        _require(
            request.get("schema") == REQUEST_SCHEMA,
            "profile-backed dynamic launch request schema drift",
        )
    _require(
        isinstance(mechanism, str) and bool(mechanism), "runner mechanism is missing"
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
    materialization = _as_mapping(
        finalization.get("materialization"), "materialization result missing"
    )
    _require(materialization.get("status") == "pass", "materialization did not pass")
    frame_count = materialization.get("frame_count")
    _require(
        isinstance(frame_count, int) and frame_count > 0,
        "materialization frame count must be positive",
    )

    materialization_root = Path(
        _as_mapping(finalization.get("artifacts"), "finalization artifacts missing")[
            "materialization_root"
        ]
    ).resolve()
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
    materialization_receipt_path = materialization_root / "materialization_receipt.json"
    materialization_receipt = _load(materialization_receipt_path)
    if mechanism == LEGACY_CAMERA_PAN_MECHANISM:
        motion_authority = _validate_legacy_camera_pan_chain(
            request=request,
            materialization=materialization,
            materialization_root=materialization_root,
            materialization_receipt=materialization_receipt,
        )
        capture_basename = LEGACY_CAMERA_PAN_ADAPTER["capture_basename"]
    else:
        motion_authority = _validate_motion_profile_chain(
            request=request,
            finalization=finalization,
            materialization=materialization,
            materialization_root=materialization_root,
            materialization_receipt=materialization_receipt,
        )
        candidate_episode_id = motion_authority["candidate_episode_id"]
        _require(
            isinstance(candidate_episode_id, str) and bool(candidate_episode_id),
            "candidate episode id missing from profile",
        )
        _require(
            Path(candidate_episode_id).name == candidate_episode_id
            and candidate_episode_id not in {".", ".."},
            "candidate episode id is not a safe capture sibling name",
        )
        capture_basename = f"{candidate_episode_id}__capture_attempt_01"

    _require(
        finalization.get("episode_id") == request.get("episode_id"), "episode mismatch"
    )
    _require(
        finalization.get("mechanism") == request.get("mechanism"), "mechanism mismatch"
    )
    suite_path = Path(request["suite_plan"]).resolve()
    audio_path = Path(request["audio_wav"]).resolve()
    capture_output = Path(request["capture_output"]).resolve()
    _require(
        suite_path == materialization_root / "suite_execution_plan.json",
        "suite plan is not the materialized authority",
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
        capture_output.parent == materialization_root.parent
        and capture_output == materialization_root.parent / capture_basename,
        "capture output must be a fresh sibling derived from motion authority",
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
        "--frame-index" not in argv,
        "dynamic full Episode cannot use sparse frame selector",
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
    return request, argv, motion_authority


def run(request_path: Path, receipt_path: Path, *, dry_run: bool) -> int:
    _require(not receipt_path.exists(), "launch receipt must be new")
    request, argv, motion_authority = _validate_request(request_path)
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
        "candidate_revision": request.get("candidate_revision"),
        "attempt_policy": request["attempt_policy"],
        "physical_gpu_index": 1,
        "graphics_adapter_argument": 1,
        "forbidden_physical_gpu_indices_used": [],
        "gpu1_uuid": gpu1_uuid,
        "prelaunch_snapshot": before,
        "request": str(request_path),
        "pre_capture_finalization": request["pre_capture_finalization"],
        "motion_authority": motion_authority,
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

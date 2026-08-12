from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.qa.actor_motion_profile import (
    build_actor_motion_profile,
    source_center_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT / "tools" / "qa" / "run_strict_two_human_dynamic_full75_canary.py"
)
AUTHORITIES = {
    "target_moves": (
        REPO_ROOT
        / "examples/qa/native_strict_two_human_target_moves_native_rate_candidate_v1.json",
        REPO_ROOT
        / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_target_moves_v2_cpu_candidate_v1/target_moves_v2_preflight.json",
        REPO_ROOT
        / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_target_moves_v2_materialized_v1/suite_execution_plan.json",
    ),
    "distractor_moves": (
        REPO_ROOT
        / "examples/qa/native_strict_two_human_distractor_moves_native_rate_candidate_v1.json",
        REPO_ROOT
        / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_distractor_moves_v2_geometry_v1/distractor_moves_v2_preflight.json",
        REPO_ROOT
        / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_distractor_moves_v2_materialized_v1/suite_execution_plan.json",
    ),
    "both_move": (
        REPO_ROOT
        / "examples/qa/native_strict_two_human_both_move_native_rate_candidate_v1.json",
        REPO_ROOT
        / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_both_move_v1_adapter_v1/preflight.json",
        REPO_ROOT
        / "tmp/lead_a_strict_two_human_full_episode_batch_v1/dynamic_both_move_v1_materialized_v1/suite_execution_plan.json",
    ),
}


def _load_runner():
    spec = importlib.util.spec_from_file_location("dynamic_full75_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_profile(mechanism: str) -> dict:
    candidate_path, old_path, suite_path = AUTHORITIES[mechanism]
    candidate = _load(candidate_path)
    old = _load(old_path)
    suite = _load(suite_path)
    return build_actor_motion_profile(
        candidate_path=candidate_path,
        candidate=candidate,
        old_preflight_path=old_path,
        selected_old_row=old["canaries"][0],
        base_suite_path=suite_path,
        base_suite=suite,
    )


def _action_counts(profile: dict) -> dict[str, dict[str, int]]:
    slots = list(profile["authorities"]["candidate"]["value"]["actors"])
    actions = sorted(
        {
            state["action_id"]
            for frame in profile["frames"]
            for state in frame["actor_states"]
        }
    )
    return {
        slot: {
            action: sum(
                state["action_id"] == action
                for frame in profile["frames"]
                for state in frame["actor_states"]
                if state["slot_id"] == slot
            )
            for action in actions
        }
        for slot in slots
    }


def _per_slot_rir_counts(profile: dict) -> dict[str, int]:
    centers = source_center_paths(profile)
    frames = profile["authorities"]["base_suite"]["value"]["scenarios"][0]["plan"][
        "frames"
    ]
    return {
        slot: len(
            {
                (
                    tuple(center),
                    tuple(frame["camera_state"]["habitat_position_m"]),
                    tuple(frame["camera_state"]["world_from_rig"]["rotation_xyzw"]),
                )
                for center, frame in zip(path, frames, strict=True)
            }
        )
        for slot, path in centers.items()
    }


def _common_request(root: Path, episode_id: str, mechanism: str) -> dict:
    suite_path = root / "suite_execution_plan.json"
    audio_path = root / "binaural_v1/audio/binaural" / f"{episode_id}__v00.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"RIFF")
    _write_json(
        suite_path,
        {
            "schema": "avengine_optional_spear_apartment_suite_v1",
            "scenarios": [{"scenario_id": episode_id}],
        },
    )
    return {
        "attempt_policy": {
            "attempt_index": 1,
            "maximum_attempts_for_candidate": 1,
            "retry_same_candidate_forbidden": True,
            "failure_disposition": "reject_candidate_without_same_candidate_retry",
        },
        "repo_root": str(REPO_ROOT),
        "capture_python": "/data/jzy/miniconda3/envs/spear-env/bin/python",
        "capture_script": str(
            REPO_ROOT / "tools/qa/capture_spear_native_pixel_episode.py"
        ),
        "suite_plan": str(suite_path),
        "audio_wav": str(audio_path),
        "spear_root": "/data/jzy/code/SPEAR-lead-b",
        "rpc_port": 39701,
        "physical_gpu_index": 1,
        "graphics_adapter_argument": 1,
        "forbidden_physical_gpu_indices": [0, 3],
        "required_idle_compute_process_count": 0,
        "episode_id": episode_id,
        "mechanism": mechanism,
    }


def _profile_request_fixture(tmp_path: Path, mechanism: str) -> tuple[Path, dict]:
    profile = _build_profile(mechanism)
    candidate_binding = profile["authorities"]["candidate"]
    candidate = candidate_binding["value"]
    episode_id = candidate["legacy_episode_id"]
    root = tmp_path / f"{mechanism}_materialized"
    profile_path = root / "actor_motion_profile.json"
    _write_json(profile_path, profile)
    action_counts = _action_counts(profile)
    per_slot = _per_slot_rir_counts(profile)
    expectation = profile["rir_expectation"]
    rir_counts = {
        "stride_frames": expectation["stride_frames"],
        "requested_pair_state_count": expectation["requested_pair_state_count"],
        "unique_rir_job_count": expectation["unique_rir_job_count"],
    }
    rir_plan = {
        **rir_counts,
        "distinct_rir_state_count_by_source_slot": per_slot,
    }
    motion_profile = {
        "status": "pass_hash_bound_profile_consumed_exactly",
        "profile_file_sha256": sha256_file(profile_path),
        "profile_document_canonical_sha256": canonical_json_sha256(profile),
        "profile_content_sha256": profile["profile_content_sha256"],
        "candidate_document_sha256": candidate_binding["document_sha256"],
        "candidate_value_sha256": candidate_binding["canonical_value_sha256"],
        "action_counts": action_counts,
        "rir_plan": rir_plan,
    }
    receipt_binding = {
        "status": "pass_bound_and_consumed_frame_by_frame",
        "schema": profile["schema"],
        "profile_content_sha256": profile["profile_content_sha256"],
        "candidate_document_sha256": candidate_binding["document_sha256"],
        "candidate_value_sha256": candidate_binding["canonical_value_sha256"],
        "derived_action_counts": action_counts,
        "derived_rir_counts": rir_counts,
        "legacy_root_motion_inference_used": False,
        "qualification_claim": False,
    }
    _write_json(
        root / "materialization_receipt.json",
        {
            "episode_id": episode_id,
            "mechanism": mechanism,
            "actor_motion_profile": receipt_binding,
        },
    )
    finalization_path = root / "pre_capture_finalization_v1/finalization.json"
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
                "frame_count": len(profile["frames"]),
                "requested_source_frame_uses": expectation[
                    "requested_pair_state_count"
                ],
                "rir_stride_frames": expectation["stride_frames"],
                "expected_unique_rir_job_count": expectation["unique_rir_job_count"],
                "expected_rir_count_by_source_slot": per_slot,
                "action_counts": action_counts,
                "motion_profile": motion_profile,
            },
        },
    )
    request = _common_request(root, episode_id, mechanism)
    request.update(
        {
            "schema": "avengine_native_strict_two_human_dynamic_full75_gpu_launch_request_v3",
            "pre_capture_finalization": str(finalization_path),
            "motion_authority": {
                "schema": "avengine_actor_motion_profile_launch_authority_v1",
                "path": str(profile_path),
                "file_sha256": sha256_file(profile_path),
                "profile_content_sha256": profile["profile_content_sha256"],
            },
            "capture_output": str(
                root.parent / f"{candidate['candidate_episode_id']}__capture_attempt_01"
            ),
        }
    )
    request_path = root / "gpu_launch_attempt_01/request.json"
    _write_json(request_path, request)
    return request_path, profile


def _camera_pan_request_fixture(tmp_path: Path) -> Path:
    episode_id = "strict2h_dynamic_canary_04_camera_pan_both_static_v2"
    mechanism = "camera_pan_both_static"
    root = tmp_path / "dynamic_camera_pan_v2_materialized_v1"
    action_counts = {
        "source1": {"idle": 75, "walk": 0},
        "source2": {"idle": 75, "walk": 0},
    }
    _write_json(
        root / "materialization_receipt.json",
        {
            "episode_id": episode_id,
            "mechanism": mechanism,
            "actor_motion_profile": {
                "status": "explicit_legacy_camera_pan_adapter",
                "legacy_root_motion_inference_used": True,
                "qualification_claim": False,
            },
        },
    )
    finalization_path = root / "pre_capture_finalization_v1/finalization.json"
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
                "expected_unique_rir_job_count": 150,
                "expected_rir_count_by_source_slot": {
                    "source1": 75,
                    "source2": 75,
                },
                "action_counts": action_counts,
                "distinct_listener_orientation_count": 75,
                "camera_yaw_span_deg": 6.0,
                "motion_profile": {
                    "status": "explicit_legacy_camera_pan_adapter",
                    "live_skeletal_transition_evidence": {
                        "status": "not_applicable_static_actors"
                    },
                },
            },
        },
    )
    request = _common_request(root, episode_id, mechanism)
    request.update(
        {
            "schema": "avengine_native_strict_two_human_dynamic_full75_gpu_launch_request_v2",
            "candidate_revision": "camera_pan_v2_0589_right_target_yaw52_58_v1",
            "pre_capture_finalization": str(finalization_path),
            "capture_output": str(
                root.parent / "dynamic_camera_pan_v2_capture_attempt_01"
            ),
        }
    )
    request_path = root / "gpu_launch_attempt_01/request.json"
    _write_json(request_path, request)
    return request_path


def _idle_gpu(runner) -> dict:
    return {
        "gpus": [
            {
                "physical_index": 1,
                "uuid": runner.GPU1_UUID,
                "name": "test",
            }
        ],
        "compute_apps": [],
    }


@pytest.mark.parametrize("mechanism", sorted(AUTHORITIES))
def test_profile_backed_runner_derives_real_motion_chain(
    tmp_path: Path, monkeypatch, mechanism: str
) -> None:
    runner = _load_runner()
    request_path, profile = _profile_request_fixture(tmp_path, mechanism)
    monkeypatch.setattr(runner, "_gpu_snapshot", lambda: _idle_gpu(runner))
    receipt_path = request_path.parent / "dry_run_receipt.json"

    assert runner.run(request_path, receipt_path, dry_run=True) == 0

    receipt = _load(receipt_path)
    candidate = profile["authorities"]["candidate"]["value"]
    assert receipt["status"] == "dry_run_pass"
    assert receipt["episode_id"] == candidate["legacy_episode_id"]
    assert receipt["mechanism"] == candidate["mechanism"]
    assert receipt["motion_authority"]["status"] == (
        "pass_request_finalization_receipt_profile_chain"
    )
    assert (
        receipt["motion_authority"]["profile_content_sha256"]
        == profile["profile_content_sha256"]
    )
    assert receipt["motion_authority"]["action_counts"] == _action_counts(profile)
    assert receipt["motion_authority"]["legacy_adapter_used"] is False
    assert receipt["candidate_revision"] is None
    assert receipt["capture_output"].endswith(
        f"/{candidate['candidate_episode_id']}__capture_attempt_01"
    )
    assert "--frame-index" not in receipt["capture_argv"]


def test_runner_rejects_request_motion_hash_drift(tmp_path: Path) -> None:
    runner = _load_runner()
    request_path, _ = _profile_request_fixture(tmp_path, "target_moves")
    request = _load(request_path)
    request["motion_authority"]["file_sha256"] = "0" * 64
    _write_json(request_path, request)

    with pytest.raises(RuntimeError, match="request motion authority hash drift"):
        runner._validate_request(request_path)


def test_runner_rejects_pre_finalization_profile_drift(tmp_path: Path) -> None:
    runner = _load_runner()
    request_path, _ = _profile_request_fixture(tmp_path, "distractor_moves")
    request = _load(request_path)
    finalization_path = Path(request["pre_capture_finalization"])
    finalization = _load(finalization_path)
    finalization["materialization"]["motion_profile"]["action_counts"]["source2"] = {
        "idle": 75,
        "walk": 0,
    }
    _write_json(finalization_path, finalization)

    with pytest.raises(RuntimeError, match="pre-finalization/profile"):
        runner._validate_request(request_path)


def test_runner_rejects_materialization_receipt_rir_drift(tmp_path: Path) -> None:
    runner = _load_runner()
    request_path, _ = _profile_request_fixture(tmp_path, "both_move")
    request = _load(request_path)
    root = Path(request["pre_capture_finalization"]).parents[1]
    receipt_path = root / "materialization_receipt.json"
    receipt = _load(receipt_path)
    receipt["actor_motion_profile"]["derived_rir_counts"]["unique_rir_job_count"] += 1
    _write_json(receipt_path, receipt)

    with pytest.raises(RuntimeError, match="materialization receipt/profile"):
        runner._validate_request(request_path)


def test_runner_rejects_mutated_profile_even_with_updated_file_hash(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    request_path, _ = _profile_request_fixture(tmp_path, "target_moves")
    request = _load(request_path)
    profile_path = Path(request["motion_authority"]["path"])
    profile = _load(profile_path)
    profile["frames"][0]["actor_states"][0]["action_id"] = "forged"
    _write_json(profile_path, profile)
    request["motion_authority"]["file_sha256"] = sha256_file(profile_path)
    _write_json(request_path, request)

    with pytest.raises(Exception, match="profile content hash mismatch"):
        runner._validate_request(request_path)


def test_only_camera_pan_accepts_profileless_legacy_adapter(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    request_path = _camera_pan_request_fixture(tmp_path)
    monkeypatch.setattr(runner, "_gpu_snapshot", lambda: _idle_gpu(runner))
    receipt_path = request_path.parent / "dry_run_receipt.json"

    assert runner.run(request_path, receipt_path, dry_run=True) == 0
    receipt = _load(receipt_path)
    assert receipt["motion_authority"]["status"] == (
        "explicit_legacy_camera_pan_adapter"
    )
    assert receipt["motion_authority"]["legacy_adapter_used"] is True

    request = _load(request_path)
    request["mechanism"] = "target_moves"
    request["schema"] = runner.REQUEST_SCHEMA
    _write_json(request_path, request)
    with pytest.raises(RuntimeError, match="motion_authority is required"):
        runner._validate_request(request_path)


def test_runner_rejects_capture_output_not_derived_fresh_sibling(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    request_path, _ = _profile_request_fixture(tmp_path, "target_moves")
    request = _load(request_path)
    request["capture_output"] = str(tmp_path / "fresh_but_unbound")
    _write_json(request_path, request)

    with pytest.raises(RuntimeError, match="fresh sibling derived"):
        runner._validate_request(request_path)


def test_failed_real_attempt_is_persisted_and_cannot_retry(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    request_path, _ = _profile_request_fixture(tmp_path, "target_moves")
    monkeypatch.setattr(runner, "_gpu_snapshot", lambda: _idle_gpu(runner))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=17),
    )
    receipt_path = request_path.parent / "launch_receipt.json"

    assert runner.run(request_path, receipt_path, dry_run=False) == 17
    receipt = _load(receipt_path)
    assert receipt["status"] == "fail"
    assert receipt["capture_process_exit_code"] == 17
    assert receipt["attempt_policy"]["retry_same_candidate_forbidden"] is True
    with pytest.raises(RuntimeError, match="launch receipt must be new"):
        runner.run(request_path, receipt_path, dry_run=False)

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


def test_dynamic_launcher_uses_explicit_spear_executable_only(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    executable = tmp_path / "external-runtime" / "SpearSim.sh"
    argv = runner._capture_argv(
        {
            "capture_python": "/runtime/spear-python",
            "capture_script": "/repo/capture.py",
            "suite_plan": "/tmp/suite.json",
            "episode_id": "episode",
            "audio_wav": "/tmp/audio.wav",
            "spear_executable": str(executable),
            "capture_output": "/tmp/output",
            "rpc_port": 39701,
        }
    )
    assert argv[argv.index("--spear-executable") + 1] == str(executable)
    assert "--spear-root" not in argv


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


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


def _common_request(
    root: Path,
    episode_id: str,
    mechanism: str,
    *,
    spear_executable: Path | None = None,
    spear_root: str | None = None,
) -> dict:
    if (spear_executable is None) == (spear_root is None):
        raise ValueError("fixture requires exactly one runtime form")
    runtime = (
        {"spear_executable": str(spear_executable)}
        if spear_executable is not None
        else {"spear_root": spear_root}
    )
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
        **runtime,
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
    spear_executable = _write_executable(root / "runtime" / "SpearSim.sh")
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
        "candidate_document_sha256": candidate_binding.get("document_sha256"),
        "candidate_value_sha256": candidate_binding["canonical_value_sha256"],
        "action_counts": action_counts,
        "rir_plan": rir_plan,
    }
    receipt_binding = {
        "status": "pass_bound_and_consumed_frame_by_frame",
        "schema": profile["schema"],
        "profile_content_sha256": profile["profile_content_sha256"],
        "candidate_document_sha256": candidate_binding.get("document_sha256"),
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
    request = _common_request(
        root, episode_id, mechanism, spear_executable=spear_executable
    )
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
    request = _common_request(
        root, episode_id, mechanism, spear_root="/historical/SPEAR"
    )
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


def _historical_v2_request_fixture(tmp_path: Path, mechanism: str) -> Path:
    """Create one direct root-shaped v2 record for retained comparison reads."""

    root = tmp_path / f"historical_{mechanism}_materialized"
    request = _common_request(
        root,
        f"historical_{mechanism}_episode",
        mechanism,
        spear_root="/historical/SPEAR",
    )
    request.update(
        {
            "schema": (
                "avengine_native_strict_two_human_dynamic_full75_gpu_launch_request_v2"
            ),
            "capture_output": str(root.parent / f"historical_{mechanism}_capture"),
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
def test_prepare_profile_request_is_derived_deterministic_and_no_clobber(
    tmp_path: Path, monkeypatch, mechanism: str
) -> None:
    runner = _load_runner()
    request_path, _ = _profile_request_fixture(tmp_path, mechanism)
    expected = _load(request_path)
    request_path.unlink()
    root = request_path.parents[1]
    source = (
        root / "pre_capture_finalization_v1/finalization.json"
        if mechanism == "both_move"
        else root
    )
    monkeypatch.setattr(
        runner,
        "_gpu_snapshot",
        lambda: pytest.fail("prepare must not query GPU state"),
    )

    spear_executable = Path(expected["spear_executable"])
    first_path, first = runner.build_launch_request(source, spear_executable)
    second_path, second = runner.build_launch_request(source, spear_executable)
    assert first_path == second_path == request_path
    assert first == second == expected
    assert runner.prepare_launch_request(source, spear_executable) == request_path
    original_bytes = request_path.read_bytes()
    with pytest.raises(FileExistsError):
        runner.prepare_launch_request(source, spear_executable)
    assert request_path.read_bytes() == original_bytes


def test_prepare_rejects_authority_drift_before_creating_request(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    request_path, _ = _profile_request_fixture(tmp_path, "target_moves")
    expected = _load(request_path)
    request_path.unlink()
    finalization_path = request_path.parents[1] / (
        "pre_capture_finalization_v1/finalization.json"
    )
    finalization = _load(finalization_path)
    finalization["mechanism"] = "distractor_moves"
    _write_json(finalization_path, finalization)

    with pytest.raises(RuntimeError, match="identity drift"):
        runner.prepare_launch_request(
            finalization_path, Path(expected["spear_executable"])
        )
    assert not request_path.exists()


def test_prepare_keeps_camera_pan_as_the_only_profileless_legacy_boundary(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    request_path = _camera_pan_request_fixture(tmp_path)
    request_path.unlink()
    root = request_path.parents[1]

    supplied_executable = root / "caller-supplied" / "SpearSim.sh"
    built_path, request = runner.build_launch_request(root, supplied_executable)
    assert built_path == request_path
    assert request["schema"] == runner.REQUEST_SCHEMA
    assert request["spear_executable"] == str(supplied_executable)
    assert "motion_authority" not in request
    assert request["capture_output"].endswith(
        "/dynamic_camera_pan_v2_capture_attempt_01"
    )
    assert runner.prepare_launch_request(root, supplied_executable) == request_path
    assert not supplied_executable.exists()
    with pytest.raises(RuntimeError, match="SPEAR executable is missing"):
        runner._validate_request(request_path)

    request_path.unlink()
    finalization_path = root / "pre_capture_finalization_v1/finalization.json"
    finalization = _load(finalization_path)
    finalization["mechanism"] = "target_moves"
    _write_json(finalization_path, finalization)
    with pytest.raises(RuntimeError, match="missing actor_motion_profile"):
        runner.prepare_launch_request(root, supplied_executable)
    assert not request_path.exists()


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
    assert "--spear-executable" in receipt["capture_argv"]
    assert "--spear-root" not in receipt["capture_argv"]


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


@pytest.mark.parametrize(
    "mechanism",
    ("target_moves", "distractor_moves", "both_move", "camera_pan_both_static"),
)
def test_historical_v2_reader_preserves_all_retained_root_shapes_and_refuses_launch(
    tmp_path: Path, monkeypatch, mechanism: str
) -> None:
    runner = _load_runner()
    request_path = _historical_v2_request_fixture(tmp_path, mechanism)
    request, argv, authority = runner._validate_request(request_path)
    assert request["schema"] == runner.LEGACY_REQUEST_SCHEMA
    assert request["mechanism"] == mechanism
    assert request["spear_root"] == "/historical/SPEAR"
    assert "spear_executable" not in request
    assert argv == []
    assert authority == {
        "status": "historical_v2_comparison_reader",
        "backend_role": "comparison_visual",
        "historical_mechanism": mechanism,
        "legacy_adapter_used": mechanism == runner.LEGACY_CAMERA_PAN_MECHANISM,
        "qualification_claim": False,
    }

    monkeypatch.setattr(
        runner,
        "_gpu_snapshot",
        lambda: pytest.fail("historical v2 reader must not query GPU state"),
    )
    receipt_path = request_path.parent / "dry_run_receipt.json"
    with pytest.raises(RuntimeError, match="comparison-only"):
        runner.run(request_path, receipt_path, dry_run=True)
    assert not receipt_path.exists()


def test_current_v3_rejects_injected_legacy_spear_root(tmp_path: Path) -> None:
    runner = _load_runner()
    request_path = _camera_pan_request_fixture(tmp_path)
    request_path.unlink()
    root = request_path.parents[1]
    current_path, request = runner.build_launch_request(
        root,
        _write_executable(tmp_path / "current-runtime" / "SpearSim.sh"),
    )
    request["spear_root"] = "/historical/SPEAR"
    _write_json(current_path, request)

    with pytest.raises(RuntimeError, match="must not mix legacy spear_root"):
        runner._validate_request(current_path)


def test_current_v3_accepts_external_executable(tmp_path: Path) -> None:
    runner = _load_runner()
    request_path = _camera_pan_request_fixture(tmp_path)
    request_path.unlink()
    external_executable = _write_executable(
        tmp_path / "external-runtime" / "SpearSim.sh"
    )
    current_path, request = runner.build_launch_request(
        request_path.parents[1], external_executable
    )
    _write_json(current_path, request)

    _, argv, _ = runner._validate_request(current_path)

    assert argv[argv.index("--spear-executable") + 1] == str(external_executable)


@pytest.mark.parametrize(
    ("runtime_form", "expected_error"),
    (
        ("lexical", "lexical path is inside a Git checkout"),
        ("resolved", "resolved path is inside a Git checkout"),
    ),
)
def test_current_v3_rejects_git_checkout_executable_before_gpu_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_form: str,
    expected_error: str,
) -> None:
    runner = _load_runner()
    request_path = _camera_pan_request_fixture(tmp_path)
    request_path.unlink()
    checkout_executable = _write_executable(tmp_path / "checkout" / "SpearSim.sh")
    (checkout_executable.parent / ".git").mkdir()
    if runtime_form == "lexical":
        supplied_executable = checkout_executable
    else:
        supplied_executable = tmp_path / "external-link" / "SpearSim.sh"
        supplied_executable.parent.mkdir()
        supplied_executable.symlink_to(checkout_executable)
    current_path, request = runner.build_launch_request(
        request_path.parents[1], supplied_executable
    )
    _write_json(current_path, request)
    receipt_path = current_path.parent / "dry_run_receipt.json"
    monkeypatch.setattr(
        runner,
        "_gpu_snapshot",
        lambda: pytest.fail("Git checkout executable must fail before GPU probing"),
    )

    with pytest.raises(RuntimeError, match=expected_error):
        runner.run(current_path, receipt_path, dry_run=True)

    assert not receipt_path.exists()


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

_RETAINED_TMP_WORKSPACE = Path(__file__).resolve().parents[2] / "tmp"
# Guarding on tmp/ existing was wrong: running the engine in a
# checkout creates tmp/spear_instance_*, which made this look
# mounted and sent 49 tests into a run without their data.  The
# evidence mount signature is a lead_* workspace.
if not any(_RETAINED_TMP_WORKSPACE.glob("lead_*")):
    pytest.skip(
        "no lead_* evidence workspace under the repository tmp "
        "directory, so this checkout does not carry the retained "
        "strict-two-human evidence",
        allow_module_level=True,
    )

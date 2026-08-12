#!/usr/bin/env python3
"""Freeze the MP3D v3 failure and prepare the independent v4 f15 candidate.

The v4 candidate changes only BP_CameraSensor HFOV binding: it writes and reads
``FOVAngle`` on the exact named RGB, depth, and object-ID scene-capture
components already returned by the native pixel runner.  It also binds capture
to the GitHub-registered ``spear-env`` interpreter.  The accepted CPU room and
acoustic evidence, sparse f15 scope, physical GPU1, and formal denominator zero
remain unchanged.  Preparation and dry-run never authorize a GPU process.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))
if str(REPOSITORY / "tools/qa") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "tools/qa"))

import run_strict_two_human_mp3d_f15_probe_v3 as predecessor  # noqa: E402

base = predecessor.base

REQUEST_SCHEMA = "avengine_mp3d_strict_two_human_f15_launch_request_v4"
RECEIPT_SCHEMA = "avengine_mp3d_strict_two_human_f15_launch_receipt_v4"
V3_FAILURE_LEDGER_SCHEMA = "avengine_mp3d_f15_v3_terminal_failure_ledger_v1"
CAPTURE_PHASE_SCHEMA = "avengine_mp3d_f15_capture_phase_v1"
V3_FAILURE_STATUS = "failed_camera_component_query_not_exact_one"
CANDIDATE_REVISION = "revision_v4_named_scene_capture_hfov"
V4_ATTEMPT_DIRECTORY = "diagnostic_f15_revision_v4_launch_attempt_01"
V4_CAPTURE_DIRECTORY = "diagnostic_f15_revision_v4_capture_attempt_01"
V4_RPC_PORT = 39634
V3_EXCEPTION_TYPE = "AssertionError"
V3_EXCEPTION_MESSAGE = ""
V3_STDERR_ASSERTION = "vector.size() == 1"
V4_ATTEMPT_POLICY = {
    **base.ATTEMPT_POLICY,
    "candidate_revision": CANDIDATE_REVISION,
    "predecessor_v3_attempt_retry": False,
}


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _require_clean_repository(repo_root: Path) -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    base._require(not completed.stdout.strip(), "repository is not clean")


def _v3_attempt_paths(atom_root: Path) -> dict[str, Path]:
    attempt = atom_root / predecessor.V3_ATTEMPT_DIRECTORY
    capture = atom_root / predecessor.V3_CAPTURE_DIRECTORY
    return {
        "request": attempt / "request.json",
        "dry_run": attempt / "dry_run_receipt.json",
        "running": attempt / "running_receipt.json",
        "final": attempt / "final_receipt.json",
        "stdout": attempt / "capture_stdout.log",
        "stderr": attempt / "capture_stderr.log",
        "failure": capture / "capture_failure.json",
        "phase_preconnect": capture / "capture_phase_00_preconnect.json",
        "phase_post_entry": capture / "capture_phase_01_post-entry.json",
        "phase_mesh": capture / "capture_phase_02_mesh.json",
        "phase_lighting": capture / "capture_phase_03_lighting.json",
        "phase_camera": capture / "capture_phase_04_camera.json",
        "ledger": attempt / "failure_ledger.json",
        "capture_root": capture,
    }


def _load_required(path: Path, *, owner: str) -> dict[str, Any]:
    base._require(path.is_file(), f"missing {owner}: {path}")
    return base._load(path)


def _assert_v3_terminal_failure(atom_root: Path) -> dict[str, Any]:
    paths = _v3_attempt_paths(atom_root)
    request = _load_required(paths["request"], owner="v3 request")
    dry_run = _load_required(paths["dry_run"], owner="v3 dry-run receipt")
    running = _load_required(paths["running"], owner="v3 running receipt")
    final = _load_required(paths["final"], owner="v3 final receipt")
    failure = _load_required(paths["failure"], owner="v3 capture failure")

    base._require(
        request.get("schema") == predecessor.REQUEST_SCHEMA
        and request.get("candidate_revision") == predecessor.CANDIDATE_REVISION
        and request.get("frame_indices") == [base.FRAME_INDEX]
        and request.get("full75_allowed") is False
        and request.get("formal_dataset_count") == 0
        and Path(str(request.get("capture_output", ""))).resolve()
        == paths["capture_root"].resolve(),
        "v3 request boundary drift",
    )
    base._require(
        dry_run.get("schema") == predecessor.RECEIPT_SCHEMA
        and dry_run.get("status") == "dry_run_pass_not_launched"
        and dry_run.get("gpu_started") is False
        and dry_run.get("attempt_consumed") is False
        and dry_run.get("formal_dataset_count") == 0,
        "v3 dry-run receipt drift",
    )
    base._require(
        running.get("schema") == predecessor.RECEIPT_SCHEMA
        and running.get("status") == "running"
        and running.get("attempt_consumed") is True
        and running.get("formal_dataset_count") == 0,
        "v3 running receipt drift",
    )
    base._require(
        final.get("schema") == predecessor.RECEIPT_SCHEMA
        and final.get("status") == "failed"
        and final.get("attempt_consumed") is True
        and final.get("retry_same_candidate_forbidden") is True
        and final.get("child_exit_code") == 1
        and final.get("capture_process_exit_code") == 1
        and final.get("failure_observability_status")
        == "phase_and_complete_traceback_persisted"
        and final.get("formal_dataset_count") == 0,
        "v3 terminal receipt drift",
    )
    traceback_text = str(failure.get("traceback", ""))
    base._require(
        failure.get("schema") == base.CAPTURE_FAILURE_SCHEMA
        and failure.get("status") == "failed"
        and failure.get("phase") == "camera"
        and failure.get("exception_type") == V3_EXCEPTION_TYPE
        and failure.get("exception_message") == V3_EXCEPTION_MESSAGE
        and "_set_camera_hfov" in traceback_text
        and "get_component_by_class" in traceback_text
        and "uobject != 0" in traceback_text
        and failure.get("qualification_claim") is False
        and failure.get("formal_dataset_count") == 0,
        "v3 camera failure detail drift",
    )

    observability = final.get("capture_observability")
    base._require(isinstance(observability, Mapping), "v3 observability is missing")
    base._require(
        observability.get("capture_failure_detail") == failure,
        "v3 final receipt does not embed the exact failure detail",
    )
    failure_record = observability.get("capture_failure_artifact")
    base._require(
        isinstance(failure_record, Mapping)
        and dict(failure_record) == base._file_record(paths["failure"]),
        "v3 failure artifact binding drift",
    )

    expected_phases = [
        (0, "preconnect"),
        (1, "post-entry"),
        (2, "mesh"),
        (3, "lighting"),
        (4, "camera"),
    ]
    phase_paths = [
        paths["phase_preconnect"],
        paths["phase_post_entry"],
        paths["phase_mesh"],
        paths["phase_lighting"],
        paths["phase_camera"],
    ]
    observed_phases: list[tuple[int, str]] = []
    for phase_path, expected in zip(phase_paths, expected_phases):
        marker = _load_required(phase_path, owner=f"v3 {expected[1]} marker")
        observed = (int(marker.get("sequence", -1)), str(marker.get("phase", "")))
        base._require(
            marker.get("schema") == CAPTURE_PHASE_SCHEMA
            and marker.get("status") == "entered"
            and observed == expected
            and marker.get("qualification_claim") is False
            and marker.get("formal_dataset_count") == 0,
            f"v3 phase marker drift: {phase_path}",
        )
        observed_phases.append(observed)

    capture_files = base._regular_files(paths["capture_root"])
    expected_capture_files = sorted([paths["failure"], *phase_paths])
    base._require(
        capture_files == expected_capture_files,
        "v3 capture root contains unexpected artifacts",
    )
    frame_artifacts = [
        path
        for path in capture_files
        if path.suffix.casefold() in {".png", ".npy", ".npz", ".jpg", ".jpeg"}
    ]
    base._require(not frame_artifacts, "v3 unexpectedly materialized frame data")

    for owner in ("stdout", "stderr"):
        base._require(
            paths[owner].is_file() and paths[owner].stat().st_size > 0,
            f"v3 exclusive child {owner} is missing or empty",
        )
    stdout_text = paths["stdout"].read_text(encoding="utf-8", errors="replace")
    stderr_text = paths["stderr"].read_text(encoding="utf-8", errors="replace")
    base._require(
        all(
            marker in stdout_text
            for marker in (
                "Game Engine Initialized.",
                "LoadMap Load map complete /Engine/Maps/Entry",
                "get_component_by_class",
            )
        ),
        "v3 stdout lacks the closed Entry or camera-query evidence",
    )
    base._require(
        V3_STDERR_ASSERTION in stderr_text,
        "v3 stderr lacks the exact component-cardinality assertion",
    )
    return {
        "paths": paths,
        "request": request,
        "final": final,
        "failure": failure,
        "phases": observed_phases,
        "capture_files": capture_files,
    }


def record_v3_terminal_failure(*, atom_root: Path) -> Path:
    atom_root = atom_root.resolve()
    expected_atom = REPOSITORY / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
    base._require(atom_root == expected_atom, "MP3D f15 atom root drift")
    evidence = _assert_v3_terminal_failure(atom_root)
    paths = evidence["paths"]
    ledger_path = paths["ledger"]
    base._require(not ledger_path.exists(), "v3 failure ledger already exists")
    source_names = (
        "request",
        "dry_run",
        "running",
        "final",
        "stdout",
        "stderr",
        "failure",
        "phase_preconnect",
        "phase_post_entry",
        "phase_mesh",
        "phase_lighting",
        "phase_camera",
    )
    ledger = {
        "schema": V3_FAILURE_LEDGER_SCHEMA,
        "status": V3_FAILURE_STATUS,
        "candidate_revision": predecessor.CANDIDATE_REVISION,
        "attempt_index": 1,
        "attempt_consumed": True,
        "retry_same_candidate_forbidden": True,
        "root_cause": {
            "phase": "camera",
            "exception_type": V3_EXCEPTION_TYPE,
            "exception_message": V3_EXCEPTION_MESSAGE,
            "stderr_assertion": V3_STDERR_ASSERTION,
            "source_file": (
                "tools/qa/capture_spear_imported_glb_strict_two_human_episode.py"
            ),
            "source_function": "_set_camera_hfov",
            "failing_expression": (
                "game.unreal_service.get_component_by_class("
                "actor=camera, uclass='UCameraComponent')"
            ),
            "contract_violation": (
                "BP_CameraSensor did not expose exactly one UCameraComponent"
            ),
        },
        "ordered_entered_phases": [
            {"sequence": sequence, "phase": phase}
            for sequence, phase in evidence["phases"]
        ],
        "captured_frame_count": 0,
        "capture_artifact_count": len(evidence["capture_files"]),
        "prior_phase_closure": {
            "entry_initialized": True,
            "live_71_mesh_handle_gate_returned_before_lighting": True,
            "expected_static_mesh_count": base.EXPECTED_MESH_COUNT,
            "review_lighting_returned_before_camera": True,
            "named_camera_component_hfov_not_reached": True,
        },
        "exclusive_stdout_persisted": True,
        "exclusive_stderr_persisted": True,
        "complete_traceback_persisted": True,
        "child_exit_code": 1,
        "required_repo_commit": evidence["request"]["required_repo_commit"],
        "source_records": {
            name: base._file_record(paths[name]) for name in source_names
        },
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "recorded_at_utc": base._utc_now(),
    }
    base._write_json_exclusive(ledger_path, ledger)
    return ledger_path


def _validate_v3_ledger(ledger_path: Path) -> dict[str, Any]:
    ledger = _load_required(ledger_path, owner="v3 terminal failure ledger")
    root_cause = ledger.get("root_cause")
    prior = ledger.get("prior_phase_closure")
    base._require(
        ledger.get("schema") == V3_FAILURE_LEDGER_SCHEMA
        and ledger.get("status") == V3_FAILURE_STATUS
        and ledger.get("candidate_revision") == predecessor.CANDIDATE_REVISION
        and ledger.get("attempt_consumed") is True
        and ledger.get("retry_same_candidate_forbidden") is True
        and ledger.get("captured_frame_count") == 0
        and ledger.get("child_exit_code") == 1
        and ledger.get("qualification_claim") is False
        and ledger.get("formal_dataset_count") == 0
        and isinstance(root_cause, Mapping)
        and root_cause.get("phase") == "camera"
        and root_cause.get("exception_type") == V3_EXCEPTION_TYPE
        and root_cause.get("stderr_assertion") == V3_STDERR_ASSERTION
        and isinstance(prior, Mapping)
        and prior.get("live_71_mesh_handle_gate_returned_before_lighting") is True
        and prior.get("expected_static_mesh_count") == base.EXPECTED_MESH_COUNT,
        "v3 terminal failure ledger drift",
    )
    records = ledger.get("source_records")
    base._require(isinstance(records, Mapping), "v3 ledger source records are missing")
    for name, record in records.items():
        base._require(isinstance(record, Mapping), f"invalid v3 ledger record: {name}")
        base._validate_file_record(record, owner=f"v3 ledger {name}")
    return ledger


def _v4_source_paths() -> dict[str, Path]:
    return {
        "capture_script": REPOSITORY
        / "tools/qa/capture_spear_imported_glb_strict_two_human_episode.py",
        "room_adapter_source": REPOSITORY
        / "tools/qa/spear_imported_glb_room_adapter.py",
        "preflight_builder_source": REPOSITORY
        / "tools/qa/build_strict_two_human_mp3d_room_preflight.py",
        "v4_launcher_source": Path(__file__).resolve(),
        "v3_launcher_source": REPOSITORY
        / "tools/qa/run_strict_two_human_mp3d_f15_probe_v3.py",
        "base_launcher_source": REPOSITORY
        / "tools/qa/run_strict_two_human_mp3d_f15_probe.py",
    }


def _v4_artifact_paths(atom_root: Path) -> dict[str, Path]:
    """Bind v4 to the fresh environment-corrected CPU preflight revision."""

    paths = base._artifact_paths(atom_root)
    preflight = atom_root / "cpu_preflight_v5"
    return {
        **paths,
        "preflight": preflight / "preflight.json",
        "room_adapter": preflight / "room_adapter.json",
        "suite_plan": preflight / "suite_execution_plan.json",
        "rir_runtime_probe": preflight / "rir_runtime_probe.json",
    }


def prepare_request_v4(
    *, atom_root: Path, capture_python: Path, spear_root: Path, rpc_port: int
) -> Path:
    atom_root = atom_root.resolve()
    expected_atom = REPOSITORY / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
    base._require(atom_root == expected_atom, "MP3D f15 atom root drift")
    _require_clean_repository(REPOSITORY)
    attempt_root = atom_root / V4_ATTEMPT_DIRECTORY
    capture_output = atom_root / V4_CAPTURE_DIRECTORY
    base._require(not attempt_root.exists(), "revision_v4 attempt 01 already exists")
    base._require(
        not capture_output.exists(), "revision_v4 capture output already exists"
    )

    evidence_paths = _v4_artifact_paths(atom_root)
    base._validate_cpu_evidence(evidence_paths)
    v3_ledger_path = (
        atom_root / predecessor.V3_ATTEMPT_DIRECTORY / "failure_ledger.json"
    )
    _validate_v3_ledger(v3_ledger_path)
    source_paths = _v4_source_paths()
    base._require(
        all(path.is_file() for path in source_paths.values()),
        "revision_v4 candidate source is missing",
    )
    base._require(capture_python.is_file(), "authoritative SPEAR Python is missing")
    base._require(
        base._is_authoritative_capture_python(capture_python),
        "revision_v4 capture Python is not the authoritative SPEAR runtime",
    )
    base._require(spear_root.resolve() == base.SPEAR_ROOT, "SPEAR root drift")
    base._require(spear_root.is_dir(), "SPEAR root is missing")
    base._require(rpc_port == V4_RPC_PORT, "revision_v4 RPC port drift")

    stdout_path = attempt_root / "capture_stdout.log"
    stderr_path = attempt_root / "capture_stderr.log"
    request = {
        "schema": REQUEST_SCHEMA,
        "status": "prepared_not_launched",
        "episode_id": base.EPISODE_ID,
        "scene_id": base.SCENE_ID,
        "candidate_revision": CANDIDATE_REVISION,
        "candidate_change_contract": {
            "scope": "BP_CameraSensor named scene-capture HFOV binding only",
            "required_named_components": ["rgb", "depth", "object_ids"],
            "component_class": "USpSceneCaptureComponent2D",
            "property": "FOVAngle",
            "actor_wide_u_camera_component_query_allowed": False,
            "distinct_live_component_handles_required": True,
            "stable_handles_across_write_readback_required": True,
            "all_three_hfov_readbacks_must_match_request": True,
            "capture_runtime": str(base.CAPTURE_PYTHON_LOGICAL),
            "avengine_dot_venv_is_capture_authority": False,
            "time_or_motion_change": False,
        },
        "required_repo_commit": _git_head(REPOSITORY),
        "repo_root": str(REPOSITORY),
        "atom_root": str(atom_root),
        "attempt_root": str(attempt_root),
        "capture_output": str(capture_output),
        "capture_stdout": str(stdout_path),
        "capture_stderr": str(stderr_path),
        "capture_python": str(capture_python.resolve()),
        "capture_script": str(source_paths["capture_script"].resolve()),
        "spear_root": str(spear_root.resolve()),
        "suite_plan": str(evidence_paths["suite_plan"].resolve()),
        "room_adapter": str(evidence_paths["room_adapter"].resolve()),
        "artifact_records": {
            name: base._file_record(path) for name, path in evidence_paths.items()
        },
        "candidate_source_records": {
            name: base._file_record(path) for name, path in source_paths.items()
        },
        "predecessor_v3_failure_ledger": base._file_record(v3_ledger_path),
        "attempt_policy": V4_ATTEMPT_POLICY,
        "frame_indices": [base.FRAME_INDEX],
        "full75_allowed": False,
        "physical_gpu_index": 1,
        "physical_gpu_uuid": base.GPU1_UUID,
        "graphics_adapter_argument": 1,
        "required_idle_compute_process_count": 0,
        "rpc_port": rpc_port,
        "visibility_gate": {
            "target_instance_id": "source1",
            "distractor_instance_id": "source2",
            "target_minimum_visible_fraction": base.TARGET_MINIMUM_VISIBLE_FRACTION,
            "distractor_minimum_visible_fraction": (
                base.DISTRACTOR_MINIMUM_VISIBLE_FRACTION
            ),
            "target_only_background_depth_m": base.TARGET_ONLY_BACKGROUND_DEPTH_M,
            "absolute_tolerance_m": base.ABSOLUTE_TOLERANCE_M,
            "relative_tolerance": base.RELATIVE_TOLERANCE,
        },
        "observability_contract": {
            "exclusive_child_stdout": str(stdout_path),
            "exclusive_child_stderr": str(stderr_path),
            "capture_phase_markers_required": True,
            "complete_traceback_on_python_failure_required": True,
            "child_exit_code_in_final_receipt_required": True,
            "phases": [
                "preconnect",
                "post-entry",
                "mesh",
                "lighting",
                "camera",
                "actor",
                "capture",
            ],
        },
        "explicit_gpu_capture_authorization_required": True,
        "gpu_capture_authorized_at_prepare": False,
        "manual_review_required": True,
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "created_at_utc": base._utc_now(),
    }
    attempt_root.mkdir(parents=True)
    request_path = attempt_root / "request.json"
    base._write_json_exclusive(request_path, request)
    return request_path


def _validate_request_v4(request_path: Path) -> tuple[dict[str, Any], list[str]]:
    request = base._load(request_path)
    base._require(request.get("schema") == REQUEST_SCHEMA, "v4 request schema drift")
    base._require(
        request.get("status") == "prepared_not_launched"
        and request.get("episode_id") == base.EPISODE_ID
        and request.get("scene_id") == base.SCENE_ID
        and request.get("candidate_revision") == CANDIDATE_REVISION,
        "v4 request identity drift",
    )
    change = request.get("candidate_change_contract")
    base._require(
        isinstance(change, Mapping)
        and change.get("required_named_components") == ["rgb", "depth", "object_ids"]
        and change.get("component_class") == "USpSceneCaptureComponent2D"
        and change.get("property") == "FOVAngle"
        and change.get("actor_wide_u_camera_component_query_allowed") is False
        and change.get("distinct_live_component_handles_required") is True
        and change.get("stable_handles_across_write_readback_required") is True
        and change.get("capture_runtime") == str(base.CAPTURE_PYTHON_LOGICAL)
        and change.get("avengine_dot_venv_is_capture_authority") is False
        and change.get("time_or_motion_change") is False,
        "v4 named-component change contract drift",
    )
    repo_root = Path(str(request.get("repo_root", ""))).resolve()
    base._require(repo_root == REPOSITORY, "v4 repository drift")
    _require_clean_repository(repo_root)
    base._require(
        request.get("required_repo_commit") == _git_head(repo_root),
        "repository HEAD differs from the v4 request-bound commit",
    )
    atom_root = Path(str(request.get("atom_root", ""))).resolve()
    expected_atom = REPOSITORY / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
    base._require(atom_root == expected_atom, "v4 atom root drift")
    attempt_root = atom_root / V4_ATTEMPT_DIRECTORY
    capture_output = atom_root / V4_CAPTURE_DIRECTORY
    stdout_path = attempt_root / "capture_stdout.log"
    stderr_path = attempt_root / "capture_stderr.log"
    base._require(
        request_path.resolve() == attempt_root / "request.json"
        and Path(str(request.get("attempt_root", ""))).resolve() == attempt_root,
        "v4 request is not bound to its fresh attempt directory",
    )
    base._require(
        Path(str(request.get("capture_output", ""))).resolve() == capture_output
        and Path(str(request.get("capture_stdout", ""))).resolve() == stdout_path
        and Path(str(request.get("capture_stderr", ""))).resolve() == stderr_path,
        "v4 output or exclusive log path drift",
    )
    base._require(
        request.get("attempt_policy") == V4_ATTEMPT_POLICY
        and request.get("frame_indices") == [base.FRAME_INDEX]
        and request.get("full75_allowed") is False,
        "v4 one-attempt sparse-f15 policy drift",
    )
    base._require(
        request.get("physical_gpu_index") == 1
        and request.get("physical_gpu_uuid") == base.GPU1_UUID
        and request.get("graphics_adapter_argument") == 1
        and request.get("required_idle_compute_process_count") == 0,
        "v4 GPU binding drift",
    )
    base._require(
        request.get("explicit_gpu_capture_authorization_required") is True
        and request.get("gpu_capture_authorized_at_prepare") is False
        and request.get("qualification_claim") is False
        and request.get("formal_dataset_count") == 0,
        "v4 request crossed its authorization or formal-data boundary",
    )

    evidence_paths = _v4_artifact_paths(atom_root)
    records = request.get("artifact_records")
    base._require(
        isinstance(records, Mapping) and set(records) == set(evidence_paths),
        "v4 evidence records drift",
    )
    for name, expected_path in evidence_paths.items():
        record = records[name]
        base._require(isinstance(record, Mapping), f"invalid v4 record: {name}")
        observed_path = base._validate_file_record(record, owner=f"v4 {name}")
        base._require(observed_path == expected_path, f"v4 {name} path drift")
    base._validate_cpu_evidence(evidence_paths)

    source_paths = _v4_source_paths()
    source_records = request.get("candidate_source_records")
    base._require(
        isinstance(source_records, Mapping)
        and set(source_records) == set(source_paths),
        "v4 candidate source records drift",
    )
    for name, expected_path in source_paths.items():
        record = source_records[name]
        base._require(isinstance(record, Mapping), f"invalid v4 source: {name}")
        observed_path = base._validate_file_record(record, owner=f"v4 source {name}")
        base._require(observed_path == expected_path, f"v4 source path drift: {name}")

    ledger_path = atom_root / predecessor.V3_ATTEMPT_DIRECTORY / "failure_ledger.json"
    ledger_record = request.get("predecessor_v3_failure_ledger")
    base._require(isinstance(ledger_record, Mapping), "v4 lacks v3 failure ledger")
    base._require(
        base._validate_file_record(ledger_record, owner="v4 predecessor ledger")
        == ledger_path,
        "v4 predecessor ledger path drift",
    )
    _validate_v3_ledger(ledger_path)

    capture_python = Path(str(request.get("capture_python", "")))
    base._require(
        base._is_authoritative_capture_python(capture_python)
        and Path(str(request.get("capture_script", ""))).resolve()
        == source_paths["capture_script"]
        and Path(str(request.get("room_adapter", ""))).resolve()
        == evidence_paths["room_adapter"]
        and Path(str(request.get("spear_root", ""))).resolve() == base.SPEAR_ROOT,
        "v4 runtime input binding drift",
    )
    for key in ("capture_python", "capture_script", "suite_plan", "room_adapter"):
        base._require(Path(str(request[key])).exists(), f"missing v4 input: {key}")
    base._require(int(request.get("rpc_port", -1)) == V4_RPC_PORT, "v4 RPC drift")
    base._require(not capture_output.exists(), "v4 capture output must be new")
    argv = base._capture_argv(request)
    base._require(
        argv.count("--frame-index") == 1
        and argv[argv.index("--frame-index") + 1] == str(base.FRAME_INDEX)
        and argv.count("--graphics-adapter") == 1
        and argv[argv.index("--graphics-adapter") + 1] == "1",
        "v4 capture argv crossed f15 or GPU1 boundary",
    )
    return request, argv


def _validate_v4_capture(request: Mapping[str, Any]) -> dict[str, Any]:
    validation = base._validate_capture(request)
    capture_root = Path(str(request["capture_output"]))
    manifest = base._load(capture_root / "manifest.json")
    camera = manifest.get("camera_contract")
    base._require(isinstance(camera, Mapping), "v4 capture lacks camera contract")
    fov = camera.get("hfov_readback")
    base._require(isinstance(fov, Mapping), "v4 capture lacks named HFOV readback")
    handles = fov.get("component_handles")
    observed = fov.get("observed_horizontal_fov_deg_by_component")
    required_names = {"rgb", "depth", "object_ids"}
    base._require(
        fov.get("status") == "pass"
        and fov.get("write_method")
        == "named_USpSceneCaptureComponent2D.FOVAngle_property"
        and isinstance(handles, Mapping)
        and set(handles) == required_names
        and all(
            not isinstance(value, bool) and isinstance(value, int) and value > 0
            for value in handles.values()
        )
        and len(set(handles.values())) == len(required_names)
        and isinstance(observed, Mapping)
        and set(observed) == required_names,
        "v4 named scene-capture HFOV evidence drift",
    )
    suite = base._load(Path(str(request["suite_plan"])))
    requested_hfov = float(
        suite["scenarios"][0]["plan"]["camera"]["horizontal_fov_deg"]
    )
    base._require(
        abs(float(fov.get("requested_horizontal_fov_deg", -1.0)) - requested_hfov)
        <= 1.0e-6
        and all(
            abs(float(observed[name]) - requested_hfov) <= 1.0e-6
            for name in required_names
        ),
        "v4 named scene-capture HFOV values differ from the suite request",
    )
    pass_identities = camera.get("pass_identities")
    base._require(
        isinstance(pass_identities, list)
        and [item.get("pass_id") for item in pass_identities]
        == ["normal", "source1_target_only", "source2_target_only"]
        and all(
            item.get("camera_actor_handle") == fov.get("camera_actor_handle")
            and item.get("rgb_component_handle") == handles["rgb"]
            and item.get("metric_depth_component_handle") == handles["depth"]
            and item.get("object_id_component_handle") == handles["object_ids"]
            for item in pass_identities
        ),
        "v4 HFOV component handles differ from the three capture passes",
    )
    return {
        **validation,
        "named_scene_capture_hfov": {
            "status": "pass",
            "camera_actor_handle": fov["camera_actor_handle"],
            "component_handles": dict(handles),
            "horizontal_fov_deg": requested_hfov,
            "pass_count": len(pass_identities),
        },
    }


def run_v4(request_path: Path, *, dry_run: bool, authorize_gpu_capture: bool) -> int:
    request, argv = _validate_request_v4(request_path.resolve())
    attempt_root = Path(request["attempt_root"])
    stdout_path = Path(request["capture_stdout"])
    stderr_path = Path(request["capture_stderr"])
    dry_receipt = attempt_root / "dry_run_receipt.json"
    running_receipt = attempt_root / "running_receipt.json"
    final_receipt = attempt_root / "final_receipt.json"
    base._require(not final_receipt.exists(), "revision_v4 already has a final receipt")
    if dry_run:
        base._require(not dry_receipt.exists(), "revision_v4 dry-run receipt exists")
        base._require(not running_receipt.exists(), "revision_v4 already started")
    else:
        base._require(
            authorize_gpu_capture,
            "revision_v4 GPU capture lacks explicit launch authorization",
        )
        base._require(not running_receipt.exists(), "revision_v4 already started")
        base._require(
            not stdout_path.exists() and not stderr_path.exists(),
            "revision_v4 exclusive stdout/stderr path already exists",
        )

    before = base._gpu_snapshot()
    gpu = base._validate_gpu1_idle(before)
    base._assert_port_available(int(request["rpc_port"]))
    common = {
        "schema": RECEIPT_SCHEMA,
        "episode_id": base.EPISODE_ID,
        "scene_id": base.SCENE_ID,
        "candidate_revision": CANDIDATE_REVISION,
        "attempt_policy": V4_ATTEMPT_POLICY,
        "required_repo_commit": request["required_repo_commit"],
        "request": str(request_path.resolve()),
        "capture_argv": argv,
        "capture_output": request["capture_output"],
        "capture_stdout": str(stdout_path),
        "capture_stderr": str(stderr_path),
        "frame_indices": [base.FRAME_INDEX],
        "full75_allowed": False,
        "physical_gpu_index": 1,
        "physical_gpu_uuid": base.GPU1_UUID,
        "graphics_adapter_argument": 1,
        "prelaunch_gpu": gpu,
        "prelaunch_snapshot": before,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    if dry_run:
        base._write_json_exclusive(
            dry_receipt,
            {
                **common,
                "status": "dry_run_pass_not_launched",
                "gpu_started": False,
                "attempt_consumed": False,
                "captured_at_utc": base._utc_now(),
            },
        )
        return 0

    started_at = base._utc_now()
    base._write_json_exclusive(
        running_receipt,
        {
            **common,
            "status": "running",
            "gpu_started": False,
            "gpu_launch_state": "authorized_pending_child_invocation",
            "attempt_consumed": True,
            "started_at_utc": started_at,
            "child_invocation_attempted": False,
            "child_exit_code": None,
        },
    )
    exit_code = 1
    child_invocation_attempted = False
    child_exit_code: int | None = None
    final: dict[str, Any] = {
        **common,
        "status": "failed",
        "gpu_started": False,
        "attempt_consumed": True,
        "retry_same_candidate_forbidden": True,
        "started_at_utc": started_at,
        "ended_at_utc": None,
        "child_invocation_attempted": False,
        "child_exit_code": None,
    }
    try:
        with (
            stdout_path.open("xb") as stdout_stream,
            stderr_path.open("xb") as stderr_stream,
        ):
            child_invocation_attempted = True
            completed = subprocess.run(
                argv,
                cwd=REPOSITORY,
                check=False,
                stdout=stdout_stream,
                stderr=stderr_stream,
            )
        child_exit_code = int(completed.returncode)
        exit_code = child_exit_code
        base._require(
            child_exit_code == 0, f"revision_v4 f15 capture exited {exit_code}"
        )
        observability = base._collect_v2_capture_observability(
            Path(request["capture_output"])
        )
        base._validate_complete_v2_phase_sequence(observability)
        final["capture_observability"] = observability
        final["validation"] = _validate_v4_capture(request)
        final["status"] = "pass_diagnostic_f15_review_ready"
    except Exception as exc:  # noqa: BLE001
        final["error"] = f"{type(exc).__name__}: {exc}"
        final["launcher_traceback"] = traceback.format_exc()
        exit_code = exit_code or 1
    finally:
        final["ended_at_utc"] = base._utc_now()
        final["child_invocation_attempted"] = child_invocation_attempted
        final["child_exit_code"] = child_exit_code
        final["capture_process_exit_code"] = child_exit_code
        final["child_exit"] = {
            "observed": child_exit_code is not None,
            "returncode": child_exit_code,
        }
        final["gpu_started"] = child_exit_code is not None
        final["exclusive_child_stdout"] = (
            base._file_record(stdout_path) if stdout_path.is_file() else None
        )
        final["exclusive_child_stderr"] = (
            base._file_record(stderr_path) if stderr_path.is_file() else None
        )
        try:
            observability = base._collect_v2_capture_observability(
                Path(request["capture_output"])
            )
            final.setdefault("capture_observability", observability)
            failure = observability["capture_failure_detail"]
            if child_exit_code not in (None, 0):
                final["failure_observability_status"] = (
                    "phase_and_complete_traceback_persisted"
                    if failure is not None
                    else "child_failed_without_capture_failure_artifact"
                )
        except Exception as exc:  # noqa: BLE001
            final["capture_observability_error"] = f"{type(exc).__name__}: {exc}"
            final["capture_observability_traceback"] = traceback.format_exc()
        try:
            final["postlaunch_snapshot"] = base._gpu_snapshot()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            final["postlaunch_snapshot_error"] = f"{type(exc).__name__}: {exc}"
        base._write_json_exclusive(final_receipt, final)
    return exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ledger = subparsers.add_parser("record-v3-failure")
    ledger.add_argument("--atom-root", required=True, type=Path)
    prepare = subparsers.add_parser("prepare-v4")
    prepare.add_argument("--atom-root", required=True, type=Path)
    prepare.add_argument(
        "--capture-python", type=Path, default=base.CAPTURE_PYTHON_LOGICAL
    )
    prepare.add_argument("--spear-root", type=Path, default=base.SPEAR_ROOT)
    prepare.add_argument("--rpc-port", type=int, default=V4_RPC_PORT)
    launch = subparsers.add_parser("launch-v4")
    launch.add_argument("--request", required=True, type=Path)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--authorize-gpu-capture", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "record-v3-failure":
        path = record_v3_terminal_failure(atom_root=args.atom_root)
        print(
            "MP3D_F15_V3_FAILURE_FROZEN "
            f"status={V3_FAILURE_STATUS} ledger={path} formal=0",
            flush=True,
        )
        return 0
    if args.command == "prepare-v4":
        path = prepare_request_v4(
            atom_root=args.atom_root,
            capture_python=args.capture_python,
            spear_root=args.spear_root,
            rpc_port=args.rpc_port,
        )
        print(f"MP3D_F15_V4_REQUEST_PREPARED request={path} formal=0", flush=True)
        return 0
    return run_v4(
        args.request,
        dry_run=args.dry_run,
        authorize_gpu_capture=args.authorize_gpu_capture,
    )


if __name__ == "__main__":
    raise SystemExit(main())

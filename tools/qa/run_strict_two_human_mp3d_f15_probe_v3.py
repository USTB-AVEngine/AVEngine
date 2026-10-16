#!/usr/bin/env python3
"""Freeze the MP3D v2 failure and prepare the independent v3 f15 candidate.

The v3 candidate changes only live ``UStaticMeshComponent`` readback.  It keeps
the accepted CPU room/acoustic evidence, one frame (f15), physical GPU1, and a
formal denominator of zero.  Preparation and dry-run do not authorize a GPU
process; a real launch still requires the explicit authorization flag.
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

import run_strict_two_human_mp3d_f15_probe as base  # noqa: E402

REQUEST_SCHEMA = "avengine_mp3d_strict_two_human_f15_launch_request_v3"
RECEIPT_SCHEMA = "avengine_mp3d_strict_two_human_f15_launch_receipt_v3"
V2_FAILURE_LEDGER_SCHEMA = "avengine_mp3d_f15_v2_terminal_failure_ledger_v1"
V2_CAPTURE_PHASE_SCHEMA = "avengine_mp3d_f15_capture_phase_v1"
V2_FAILURE_STATUS = "failed_mesh_static_mesh_getter_unrealobject_not_callable"
CANDIDATE_REVISION = "revision_v3_static_mesh_component_readback"
V3_ATTEMPT_DIRECTORY = "diagnostic_f15_revision_v3_launch_attempt_01"
V3_CAPTURE_DIRECTORY = "diagnostic_f15_revision_v3_capture_attempt_01"
V3_RPC_PORT = 39633
V2_EXCEPTION_TYPE = "TypeError"
V2_EXCEPTION_MESSAGE = "'UnrealObject' object is not callable"
V3_ATTEMPT_POLICY = {
    **base.ATTEMPT_POLICY,
    "candidate_revision": CANDIDATE_REVISION,
    "predecessor_v2_attempt_retry": False,
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


def _v2_attempt_paths(atom_root: Path) -> dict[str, Path]:
    attempt = atom_root / base.V2_ATTEMPT_DIRECTORY
    capture = atom_root / base.V2_CAPTURE_DIRECTORY
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
        "ledger": attempt / "failure_ledger.json",
        "capture_root": capture,
    }


def _load_required(path: Path, *, owner: str) -> dict[str, Any]:
    base._require(path.is_file(), f"missing {owner}: {path}")
    return base._load(path)


def _assert_v2_terminal_failure(atom_root: Path) -> dict[str, Any]:
    paths = _v2_attempt_paths(atom_root)
    request = _load_required(paths["request"], owner="v2 request")
    dry_run = _load_required(paths["dry_run"], owner="v2 dry-run receipt")
    running = _load_required(paths["running"], owner="v2 running receipt")
    final = _load_required(paths["final"], owner="v2 final receipt")
    failure = _load_required(paths["failure"], owner="v2 capture failure")

    base._require(
        request.get("schema") == base.REQUEST_SCHEMA_V2
        and request.get("candidate_revision") == "revision_v2_observability_only"
        and request.get("frame_indices") == [base.FRAME_INDEX]
        and request.get("full75_allowed") is False
        and request.get("formal_dataset_count") == 0
        and Path(str(request.get("capture_output", ""))).resolve()
        == paths["capture_root"].resolve(),
        "v2 request boundary drift",
    )
    base._require(
        dry_run.get("schema") == base.RECEIPT_SCHEMA_V2
        and dry_run.get("status") == "dry_run_pass_not_launched"
        and dry_run.get("gpu_started") is False
        and dry_run.get("attempt_consumed") is False
        and dry_run.get("formal_dataset_count") == 0,
        "v2 dry-run receipt drift",
    )
    base._require(
        running.get("schema") == base.RECEIPT_SCHEMA_V2
        and running.get("status") == "running"
        and running.get("attempt_consumed") is True
        and running.get("formal_dataset_count") == 0,
        "v2 running receipt drift",
    )
    base._require(
        final.get("schema") == base.RECEIPT_SCHEMA_V2
        and final.get("status") == "failed"
        and final.get("attempt_consumed") is True
        and final.get("retry_same_candidate_forbidden") is True
        and final.get("child_exit_code") == 1
        and final.get("capture_process_exit_code") == 1
        and final.get("failure_observability_status")
        == "phase_and_complete_traceback_persisted"
        and final.get("formal_dataset_count") == 0,
        "v2 terminal receipt drift",
    )
    base._require(
        failure.get("schema") == base.CAPTURE_FAILURE_SCHEMA
        and failure.get("status") == "failed"
        and failure.get("phase") == "mesh"
        and failure.get("exception_type") == V2_EXCEPTION_TYPE
        and failure.get("exception_message") == V2_EXCEPTION_MESSAGE
        and V2_EXCEPTION_MESSAGE in str(failure.get("traceback", ""))
        and failure.get("qualification_claim") is False
        and failure.get("formal_dataset_count") == 0,
        "v2 mesh failure detail drift",
    )

    observability = final.get("capture_observability")
    base._require(isinstance(observability, Mapping), "v2 observability is missing")
    base._require(
        observability.get("capture_failure_detail") == failure,
        "v2 final receipt does not embed the exact failure detail",
    )
    failure_record = observability.get("capture_failure_artifact")
    base._require(
        isinstance(failure_record, Mapping)
        and base._validate_file_record(
            failure_record, owner="v2 capture failure artifact"
        )
        == paths["failure"].resolve(),
        "v2 failure artifact binding drift",
    )

    expected_phases = [(0, "preconnect"), (1, "post-entry"), (2, "mesh")]
    phase_paths = [
        paths["phase_preconnect"],
        paths["phase_post_entry"],
        paths["phase_mesh"],
    ]
    observed_phases: list[tuple[int, str]] = []
    for phase_path, expected in zip(phase_paths, expected_phases, strict=True):
        marker = _load_required(phase_path, owner=f"v2 {expected[1]} marker")
        observed = (int(marker.get("sequence", -1)), str(marker.get("phase", "")))
        base._require(
            marker.get("schema") == V2_CAPTURE_PHASE_SCHEMA
            and marker.get("status") == "entered"
            and observed == expected
            and marker.get("qualification_claim") is False
            and marker.get("formal_dataset_count") == 0,
            f"v2 phase marker drift: {phase_path}",
        )
        observed_phases.append(observed)

    capture_files = base._regular_files(paths["capture_root"])
    expected_capture_files = sorted([paths["failure"], *phase_paths])
    base._require(
        capture_files == expected_capture_files,
        "v2 capture root contains unexpected artifacts",
    )
    frame_artifacts = [
        path
        for path in capture_files
        if path.suffix.casefold() in {".png", ".npy", ".npz", ".jpg", ".jpeg"}
    ]
    base._require(not frame_artifacts, "v2 unexpectedly materialized frame data")

    for owner in ("stdout", "stderr"):
        base._require(
            paths[owner].is_file() and paths[owner].stat().st_size > 0,
            f"v2 exclusive child {owner} is missing or empty",
        )
    base._require(
        V2_EXCEPTION_MESSAGE in paths["stderr"].read_text(encoding="utf-8"),
        "v2 stderr lacks the exact terminal TypeError",
    )
    return {
        "paths": paths,
        "request": request,
        "final": final,
        "failure": failure,
        "phases": observed_phases,
        "capture_files": capture_files,
    }


def record_v2_terminal_failure(*, atom_root: Path) -> Path:
    atom_root = atom_root.resolve()
    expected_atom = REPOSITORY / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
    base._require(atom_root == expected_atom, "MP3D f15 atom root drift")
    evidence = _assert_v2_terminal_failure(atom_root)
    paths = evidence["paths"]
    ledger_path = paths["ledger"]
    base._require(not ledger_path.exists(), "v2 failure ledger already exists")
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
    )
    ledger = {
        "schema": V2_FAILURE_LEDGER_SCHEMA,
        "status": V2_FAILURE_STATUS,
        "candidate_revision": "revision_v2_observability_only",
        "attempt_index": 1,
        "attempt_consumed": True,
        "retry_same_candidate_forbidden": True,
        "root_cause": {
            "phase": "mesh",
            "exception_type": V2_EXCEPTION_TYPE,
            "exception_message": V2_EXCEPTION_MESSAGE,
            "source_file": "tools/qa/spear_imported_glb_room_adapter.py",
            "source_line": 196,
            "failing_expression": "component.GetStaticMesh(as_handle=True)",
        },
        "ordered_entered_phases": [
            {"sequence": sequence, "phase": phase}
            for sequence, phase in evidence["phases"]
        ],
        "captured_frame_count": 0,
        "capture_artifact_count": len(evidence["capture_files"]),
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


def _validate_v2_ledger(ledger_path: Path) -> dict[str, Any]:
    ledger = _load_required(ledger_path, owner="v2 terminal failure ledger")
    root_cause = ledger.get("root_cause")
    base._require(
        ledger.get("schema") == V2_FAILURE_LEDGER_SCHEMA
        and ledger.get("status") == V2_FAILURE_STATUS
        and ledger.get("candidate_revision") == "revision_v2_observability_only"
        and ledger.get("attempt_consumed") is True
        and ledger.get("retry_same_candidate_forbidden") is True
        and ledger.get("captured_frame_count") == 0
        and ledger.get("child_exit_code") == 1
        and ledger.get("qualification_claim") is False
        and ledger.get("formal_dataset_count") == 0
        and isinstance(root_cause, Mapping)
        and root_cause.get("phase") == "mesh"
        and root_cause.get("exception_type") == V2_EXCEPTION_TYPE
        and root_cause.get("exception_message") == V2_EXCEPTION_MESSAGE,
        "v2 terminal failure ledger drift",
    )
    records = ledger.get("source_records")
    base._require(isinstance(records, Mapping), "v2 ledger source records are missing")
    for name, record in records.items():
        base._require(isinstance(record, Mapping), f"invalid v2 ledger record: {name}")
        base._validate_file_record(record, owner=f"v2 ledger {name}")
    return ledger


def _v3_source_paths() -> dict[str, Path]:
    return {
        "capture_script": REPOSITORY
        / "tools/qa/capture_spear_imported_glb_strict_two_human_episode.py",
        "room_adapter_source": REPOSITORY
        / "tools/qa/spear_imported_glb_room_adapter.py",
        "v3_launcher_source": Path(__file__).resolve(),
        "base_launcher_source": REPOSITORY
        / "tools/qa/run_strict_two_human_mp3d_f15_probe.py",
    }


def prepare_request_v3(
    *, atom_root: Path, capture_python: Path, spear_root: Path, rpc_port: int
) -> Path:
    atom_root = atom_root.resolve()
    expected_atom = REPOSITORY / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
    base._require(atom_root == expected_atom, "MP3D f15 atom root drift")
    _require_clean_repository(REPOSITORY)
    attempt_root = atom_root / V3_ATTEMPT_DIRECTORY
    capture_output = atom_root / V3_CAPTURE_DIRECTORY
    base._require(not attempt_root.exists(), "revision_v3 attempt 01 already exists")
    base._require(
        not capture_output.exists(), "revision_v3 capture output already exists"
    )

    evidence_paths = base._artifact_paths(atom_root)
    base._validate_cpu_evidence(evidence_paths)
    v2_ledger_path = atom_root / base.V2_ATTEMPT_DIRECTORY / "failure_ledger.json"
    _validate_v2_ledger(v2_ledger_path)
    source_paths = _v3_source_paths()
    base._require(
        all(path.is_file() for path in source_paths.values()),
        "revision_v3 candidate source is missing",
    )
    base._require(capture_python.is_file(), "authoritative SPEAR Python is missing")
    base._require(
        base._is_authoritative_capture_python(capture_python),
        "revision_v3 capture Python is not the authoritative SPEAR runtime",
    )
    base._require(spear_root.resolve() == base.SPEAR_ROOT, "SPEAR root drift")
    base._require(spear_root.is_dir(), "SPEAR root is missing")
    base._require(rpc_port == V3_RPC_PORT, "revision_v3 RPC port drift")

    stdout_path = attempt_root / "capture_stdout.log"
    stderr_path = attempt_root / "capture_stderr.log"
    request = {
        "schema": REQUEST_SCHEMA,
        "status": "prepared_not_launched",
        "episode_id": base.EPISODE_ID,
        "scene_id": base.SCENE_ID,
        "candidate_revision": CANDIDATE_REVISION,
        "candidate_change_contract": {
            "scope": "UStaticMeshComponent live StaticMesh readback only",
            "callable_getter": "call GetStaticMesh(as_handle=True)",
            "noncallable_getter": (
                "read component StaticMesh property with as_handle=True"
            ),
            "authoritative_value": "actual component StaticMesh handle",
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
        "predecessor_v2_failure_ledger": base._file_record(v2_ledger_path),
        "attempt_policy": V3_ATTEMPT_POLICY,
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


def _validate_request_v3(request_path: Path) -> tuple[dict[str, Any], list[str]]:
    request = base._load(request_path)
    base._require(request.get("schema") == REQUEST_SCHEMA, "v3 request schema drift")
    base._require(
        request.get("status") == "prepared_not_launched"
        and request.get("episode_id") == base.EPISODE_ID
        and request.get("scene_id") == base.SCENE_ID
        and request.get("candidate_revision") == CANDIDATE_REVISION,
        "v3 request identity drift",
    )
    repo_root = Path(str(request.get("repo_root", ""))).resolve()
    base._require(repo_root == REPOSITORY, "v3 repository drift")
    _require_clean_repository(repo_root)
    base._require(
        request.get("required_repo_commit") == _git_head(repo_root),
        "repository HEAD differs from the v3 request-bound commit",
    )
    atom_root = Path(str(request.get("atom_root", ""))).resolve()
    expected_atom = REPOSITORY / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
    base._require(atom_root == expected_atom, "v3 atom root drift")
    attempt_root = atom_root / V3_ATTEMPT_DIRECTORY
    capture_output = atom_root / V3_CAPTURE_DIRECTORY
    stdout_path = attempt_root / "capture_stdout.log"
    stderr_path = attempt_root / "capture_stderr.log"
    base._require(
        request_path.resolve() == attempt_root / "request.json"
        and Path(str(request.get("attempt_root", ""))).resolve() == attempt_root,
        "v3 request is not bound to its fresh attempt directory",
    )
    base._require(
        Path(str(request.get("capture_output", ""))).resolve() == capture_output
        and Path(str(request.get("capture_stdout", ""))).resolve() == stdout_path
        and Path(str(request.get("capture_stderr", ""))).resolve() == stderr_path,
        "v3 output or exclusive log path drift",
    )
    base._require(
        request.get("attempt_policy") == V3_ATTEMPT_POLICY
        and request.get("frame_indices") == [base.FRAME_INDEX]
        and request.get("full75_allowed") is False,
        "v3 one-attempt sparse-f15 policy drift",
    )
    base._require(
        request.get("physical_gpu_index") == 1
        and request.get("physical_gpu_uuid") == base.GPU1_UUID
        and request.get("graphics_adapter_argument") == 1
        and request.get("required_idle_compute_process_count") == 0,
        "v3 GPU binding drift",
    )
    base._require(
        request.get("explicit_gpu_capture_authorization_required") is True
        and request.get("gpu_capture_authorized_at_prepare") is False
        and request.get("qualification_claim") is False
        and request.get("formal_dataset_count") == 0,
        "v3 request crossed its authorization or formal-data boundary",
    )

    evidence_paths = base._artifact_paths(atom_root)
    records = request.get("artifact_records")
    base._require(
        isinstance(records, Mapping) and set(records) == set(evidence_paths),
        "v3 evidence records drift",
    )
    for name, expected_path in evidence_paths.items():
        record = records[name]
        base._require(isinstance(record, Mapping), f"invalid v3 record: {name}")
        observed_path = base._validate_file_record(record, owner=f"v3 {name}")
        base._require(observed_path == expected_path, f"v3 {name} path drift")
    base._validate_cpu_evidence(evidence_paths)

    source_paths = _v3_source_paths()
    source_records = request.get("candidate_source_records")
    base._require(
        isinstance(source_records, Mapping)
        and set(source_records) == set(source_paths),
        "v3 candidate source records drift",
    )
    for name, expected_path in source_paths.items():
        record = source_records[name]
        base._require(isinstance(record, Mapping), f"invalid v3 source: {name}")
        observed_path = base._validate_file_record(record, owner=f"v3 source {name}")
        base._require(observed_path == expected_path, f"v3 source path drift: {name}")

    ledger_path = atom_root / base.V2_ATTEMPT_DIRECTORY / "failure_ledger.json"
    ledger_record = request.get("predecessor_v2_failure_ledger")
    base._require(isinstance(ledger_record, Mapping), "v3 lacks v2 failure ledger")
    base._require(
        base._validate_file_record(ledger_record, owner="v3 predecessor ledger")
        == ledger_path,
        "v3 predecessor ledger path drift",
    )
    _validate_v2_ledger(ledger_path)

    capture_python = Path(str(request.get("capture_python", "")))
    base._require(
        base._is_authoritative_capture_python(capture_python)
        and Path(str(request.get("capture_script", ""))).resolve()
        == source_paths["capture_script"]
        and Path(str(request.get("room_adapter", ""))).resolve()
        == evidence_paths["room_adapter"]
        and Path(str(request.get("spear_root", ""))).resolve() == base.SPEAR_ROOT,
        "v3 runtime input binding drift",
    )
    for key in ("capture_python", "capture_script", "suite_plan", "room_adapter"):
        base._require(Path(str(request[key])).exists(), f"missing v3 input: {key}")
    base._require(int(request.get("rpc_port", -1)) == V3_RPC_PORT, "v3 RPC drift")
    base._require(not capture_output.exists(), "v3 capture output must be new")
    argv = base._capture_argv(request)
    base._require(
        argv.count("--frame-index") == 1
        and argv[argv.index("--frame-index") + 1] == str(base.FRAME_INDEX)
        and argv.count("--graphics-adapter") == 1
        and argv[argv.index("--graphics-adapter") + 1] == "1",
        "v3 capture argv crossed f15 or GPU1 boundary",
    )
    return request, argv


def run_v3(request_path: Path, *, dry_run: bool, authorize_gpu_capture: bool) -> int:
    request, argv = _validate_request_v3(request_path.resolve())
    attempt_root = Path(request["attempt_root"])
    stdout_path = Path(request["capture_stdout"])
    stderr_path = Path(request["capture_stderr"])
    dry_receipt = attempt_root / "dry_run_receipt.json"
    running_receipt = attempt_root / "running_receipt.json"
    final_receipt = attempt_root / "final_receipt.json"
    base._require(not final_receipt.exists(), "revision_v3 already has a final receipt")
    if dry_run:
        base._require(not dry_receipt.exists(), "revision_v3 dry-run receipt exists")
        base._require(not running_receipt.exists(), "revision_v3 already started")
    else:
        base._require(
            authorize_gpu_capture,
            "revision_v3 GPU capture lacks explicit launch authorization",
        )
        base._require(not running_receipt.exists(), "revision_v3 already started")
        base._require(
            not stdout_path.exists() and not stderr_path.exists(),
            "revision_v3 exclusive stdout/stderr path already exists",
        )

    before = base._gpu_snapshot()
    gpu = base._validate_gpu1_idle(before)
    base._assert_port_available(int(request["rpc_port"]))
    common = {
        "schema": RECEIPT_SCHEMA,
        "episode_id": base.EPISODE_ID,
        "scene_id": base.SCENE_ID,
        "candidate_revision": CANDIDATE_REVISION,
        "attempt_policy": V3_ATTEMPT_POLICY,
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
            child_exit_code == 0, f"revision_v3 f15 capture exited {exit_code}"
        )
        observability = base._collect_v2_capture_observability(
            Path(request["capture_output"])
        )
        base._validate_complete_v2_phase_sequence(observability)
        final["capture_observability"] = observability
        final["validation"] = base._validate_capture(request)
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
    ledger = subparsers.add_parser("record-v2-failure")
    ledger.add_argument("--atom-root", required=True, type=Path)
    prepare = subparsers.add_parser("prepare-v3")
    prepare.add_argument("--atom-root", required=True, type=Path)
    prepare.add_argument(
        "--capture-python", type=Path, default=base.CAPTURE_PYTHON_LOGICAL
    )
    prepare.add_argument("--spear-root", type=Path, default=base.SPEAR_ROOT)
    prepare.add_argument("--rpc-port", type=int, default=V3_RPC_PORT)
    launch = subparsers.add_parser("launch-v3")
    launch.add_argument("--request", required=True, type=Path)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--authorize-gpu-capture", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "record-v2-failure":
        path = record_v2_terminal_failure(atom_root=args.atom_root)
        print(
            "MP3D_F15_V2_FAILURE_FROZEN "
            f"status={V2_FAILURE_STATUS} ledger={path} formal=0",
            flush=True,
        )
        return 0
    if args.command == "prepare-v3":
        path = prepare_request_v3(
            atom_root=args.atom_root,
            capture_python=args.capture_python,
            spear_root=args.spear_root,
            rpc_port=args.rpc_port,
        )
        print(f"MP3D_F15_V3_REQUEST_PREPARED request={path} formal=0", flush=True)
        return 0
    return run_v3(
        args.request,
        dry_run=args.dry_run,
        authorize_gpu_capture=args.authorize_gpu_capture,
    )


if __name__ == "__main__":
    raise SystemExit(main())

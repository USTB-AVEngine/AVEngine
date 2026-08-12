#!/usr/bin/env python3
"""Freeze the Skokloster v1 environment failure and prepare f15 revision v2.

Revision v2 changes only the Python runtime binding.  The capture child is
executed through the repository-declared ``spear-env`` logical entry point.
The request records that logical path and its real path separately, and never
substitutes the real path into argv.  Preparation probes the exact executable,
imports the complete capture module graph, and persists the result before any
GPU launch can be authorized.

The v1 attempt is terminal and may not be retried.  Revision v2 uses a fresh
request, attempt directory, capture directory, RPC port, and one-attempt
policy.  It remains an f15 diagnostic with a formal denominator of zero.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
TOOLS = REPOSITORY / "tools/qa"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_strict_two_human_skokloster_f15_probe as base  # noqa: E402
from pre_gpu_launch_ledger import (  # noqa: E402
    PreparedAttemptSpec,
    PreservedFileIdentity,
    archive_prepared_attempt,
)

REQUEST_SCHEMA = "avengine_skokloster_strict_two_human_f15_launch_request_v2"
RECEIPT_SCHEMA = "avengine_skokloster_strict_two_human_f15_launch_receipt_v2"
PHASE_SCHEMA = "avengine_skokloster_strict_two_human_f15_launch_phase_v2"
V1_LEDGER_SCHEMA = "avengine_skokloster_f15_v1_terminal_failure_ledger_v1"
PROBE_RECEIPT_SCHEMA = "avengine_skokloster_capture_interpreter_probe_v1"
V1_FAILURE_STATUS = "failed_import_environment_misbound_numpy_missing"
CANDIDATE_REVISION = "revision_v2_authoritative_spear_interpreter_binding"
V2_ATTEMPT_DIRECTORY = "diagnostic_f15_revision_v2_launch_attempt_01"
V2_CAPTURE_DIRECTORY = "diagnostic_f15_revision_v2_capture_attempt_01"
V2_RPC_PORT = 39832
V2_STALE_PREPARATION_ARCHIVE_DIRECTORY = (
    "diagnostic_f15_revision_v2_prepare_superseded_source_drift_01"
)
AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL = Path(
    "/data/jzy/miniconda3/envs/spear-env/bin/python"
)
AUTHORITATIVE_ENV_PREFIX = Path("/data/jzy/miniconda3/envs/spear-env")
OFFICIAL_ENV_RECIPE_RELATIVE = Path("envs/spear-env.yml")
OFFICIAL_ENV_GIT_REF = "origin/main"
V1_EXCEPTION = "ModuleNotFoundError: No module named 'numpy'"
V2_ATTEMPT_POLICY = {
    **base.ATTEMPT_POLICY,
    "candidate_revision": CANDIDATE_REVISION,
    "predecessor_v1_attempt_retry": False,
}
V2_CHANGE_CONTRACT = {
    "scope": "capture child Python interpreter binding only",
    "official_environment": "spear-env",
    "execute_logical_path_without_realpath_substitution": True,
    "room_camera_actor_audio_or_motion_change": False,
}
EXPECTED_DEPENDENCIES = {
    "numpy": {"module_version": "2.0.2", "distribution_version": "2.0.2"},
    "cv2": {"module_version": "4.10.0", "distribution_version": "4.10.0.84"},
    "yaml": {"module_version": "6.0.3", "distribution_version": "6.0.3"},
    "spear": {"distribution_version": "1.0.0"},
    "spear_ext": {"distribution_version": "1.0.0"},
}
V2_REQUEST_KEYS = frozenset(
    {
        "artifact_records",
        "atom_root",
        "attempt_policy",
        "attempt_root",
        "audio_wav",
        "candidate_change_contract",
        "candidate_revision",
        "candidate_source_records",
        "capture_output",
        "capture_python",
        "capture_python_logical",
        "capture_python_realpath",
        "capture_script",
        "capture_stderr",
        "capture_stdout",
        "cpu_validation",
        "created_at_utc",
        "episode_id",
        "explicit_gpu_capture_authorization_required",
        "formal_dataset_count",
        "frame_indices",
        "full75_allowed",
        "gpu_capture_authorized_at_prepare",
        "graphics_adapter_argument",
        "interpreter_preflight_receipt",
        "manual_visual_review_required",
        "mp3d_revision_v2_terminal_receipt",
        "mp3d_revision_v2_terminal_required_before_real_launch",
        "official_env_contract",
        "package_records",
        "packaged_executable",
        "packaged_map",
        "physical_gpu_index",
        "physical_gpu_uuid",
        "predecessor_v1_failure_ledger",
        "qualification_claim",
        "repo_root",
        "required_clean_worktree",
        "required_idle_compute_process_count",
        "required_repo_commit",
        "rpc_port",
        "scene_id",
        "schema",
        "spear_root",
        "status",
        "suite_plan",
        "visibility_gate",
    }
)

INTERPRETER_PROBE_CODE = r"""
import importlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
logical = sys.argv[2]
distributions = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "spear": "spear-sim",
    "spear_ext": "spear-ext",
}
dependencies = {}
for name, distribution in distributions.items():
    module = importlib.import_module(name)
    dependencies[name] = {
        "module_version": str(getattr(module, "__version__", "")),
        "distribution": distribution,
        "distribution_version": importlib.metadata.version(distribution),
        "origin": str(Path(module.__file__).resolve()),
    }

sys.path.insert(0, str(repo / "src"))
avengine = importlib.import_module("avengine")
capture_path = repo / "tools/qa/capture_skokloster_strict_two_human_episode.py"
spec = importlib.util.spec_from_file_location("skok_v2_capture_import_probe", capture_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import capture module: {capture_path}")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)

loaded_cuda_libraries = []
maps = Path("/proc/self/maps")
if maps.is_file():
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        lowered = line.casefold()
        if any(token in lowered for token in ("libcuda", "libcudart", "libtorch_cuda")):
            loaded_cuda_libraries.append(line.rsplit(maxsplit=1)[-1])

print(json.dumps({
    "logical_interpreter": logical,
    "interpreter_realpath": os.path.realpath(logical),
    "sys_executable": sys.executable,
    "sys_prefix": sys.prefix,
    "sys_base_prefix": sys.base_prefix,
    "python_version": ".".join(str(item) for item in sys.version_info[:3]),
    "dependencies": dependencies,
    "avengine_origin": str(Path(avengine.__file__).resolve()),
    "capture_module_origin": str(capture_path.resolve()),
    "capture_module_imported": True,
    "loaded_cuda_libraries": sorted(set(loaded_cuda_libraries)),
    "cuda_initialized": False,
    "probe_pid": os.getpid(),
}, sort_keys=True))
"""


def _logical_absolute(path: Path) -> Path:
    """Return an absolute lexical path without dereferencing symlinks."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _official_env_recipe() -> Path:
    return REPOSITORY / OFFICIAL_ENV_RECIPE_RELATIVE


def _validate_official_env_recipe(path: Path) -> None:
    base._require(path.is_file(), f"official spear-env recipe is missing: {path}")
    text = path.read_text(encoding="utf-8")
    for required in (
        "name: spear-env",
        "python=3.11",
        "numpy==2.0.2",
        "opencv-python==4.10.0.84",
        "PyYAML==6.0.3",
        "/data/jzy/miniconda3/envs/spear-env/bin/pip install -e",
        "external/SPEAR/python",
        "external/SPEAR/python_ext",
    ):
        base._require(required in text, f"official spear-env recipe lacks {required}")


def _git_text(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def _origin_main_env_contract() -> dict[str, Any]:
    recipe = _official_env_recipe()
    _validate_official_env_recipe(recipe)
    remote_url = _git_text("remote", "get-url", "origin").strip()
    base._require(
        "Eastforward/AVEngine" in remote_url,
        "origin is not the authoritative Eastforward/AVEngine repository",
    )
    commit = _git_text("rev-parse", OFFICIAL_ENV_GIT_REF).strip()
    base._require(len(commit) == 40, "origin/main is not a full commit id")
    committed_recipe = _git_text(
        "show", f"{OFFICIAL_ENV_GIT_REF}:{OFFICIAL_ENV_RECIPE_RELATIVE}"
    )
    base._require(
        recipe.read_text(encoding="utf-8") == committed_recipe,
        "working spear-env recipe differs from origin/main",
    )
    return {
        "repository": "Eastforward/AVEngine",
        "remote_url": remote_url,
        "git_ref": OFFICIAL_ENV_GIT_REF,
        "commit": commit,
        "recipe_path": str(OFFICIAL_ENV_RECIPE_RELATIVE),
        "recipe": base._file_record(recipe),
        "environment_name": "spear-env",
        "logical_interpreter": str(AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL),
    }


def _validate_origin_main_env_contract(contract: object) -> None:
    base._require(isinstance(contract, Mapping), "origin/main env contract missing")
    current = _origin_main_env_contract()
    base._require(dict(contract) == current, "origin/main env contract drift")


def _v1_paths(atom_root: Path) -> dict[str, Path]:
    attempt = atom_root / base.ATTEMPT_DIRECTORY
    return {
        "request": attempt / "request.json",
        "dry_run": attempt / "dry_run_receipt.json",
        "running": attempt / "running_receipt.json",
        "final": attempt / "final_receipt.json",
        "stdout": attempt / "capture_stdout.log",
        "stderr": attempt / "capture_stderr.log",
        "phase_prelaunch": attempt / "launch_phase_000_prelaunch_closed.json",
        "phase_child_start": attempt / "launch_phase_001_child_invocation_started.json",
        "phase_child_exit": attempt / "launch_phase_002_child_exit_observed.json",
        "ledger": attempt / "failure_ledger.json",
        "capture_root": atom_root / base.CAPTURE_DIRECTORY,
    }


def _load_required(path: Path, *, owner: str) -> dict[str, Any]:
    base._require(path.is_file(), f"missing {owner}: {path}")
    return base._load(path)


def _assert_v1_terminal_failure(atom_root: Path) -> dict[str, Any]:
    paths = _v1_paths(atom_root)
    request = _load_required(paths["request"], owner="v1 request")
    dry_run = _load_required(paths["dry_run"], owner="v1 dry-run receipt")
    running = _load_required(paths["running"], owner="v1 running receipt")
    final = _load_required(paths["final"], owner="v1 final receipt")
    logical_v1 = _logical_absolute(base.CAPTURE_PYTHON_LOGICAL)
    bound_v1 = request.get("capture_python")

    base._require(
        request.get("schema") == base.REQUEST_SCHEMA
        and request.get("status") == "prepared_not_launched"
        and request.get("frame_indices") == [base.FRAME_INDEX]
        and request.get("full75_allowed") is False
        and request.get("formal_dataset_count") == 0
        and isinstance(bound_v1, str)
        and bound_v1
        and Path(str(request.get("capture_output", ""))).resolve()
        == paths["capture_root"].resolve(),
        "Skokloster v1 request boundary drift",
    )
    base._require(
        dry_run.get("schema") == base.RECEIPT_SCHEMA
        and dry_run.get("status") == "dry_run_pass_not_launched"
        and dry_run.get("gpu_started") is False
        and dry_run.get("attempt_consumed") is False
        and dry_run.get("formal_dataset_count") == 0,
        "Skokloster v1 dry-run receipt drift",
    )
    base._require(
        running.get("schema") == base.RECEIPT_SCHEMA
        and running.get("status") == "running"
        and running.get("attempt_consumed") is True
        and running.get("formal_dataset_count") == 0,
        "Skokloster v1 running receipt drift",
    )
    phases = final.get("launcher_phases")
    base._require(
        final.get("schema") == base.RECEIPT_SCHEMA
        and final.get("status") == "failed"
        and final.get("attempt_consumed") is True
        and final.get("retry_same_candidate_forbidden") is True
        and final.get("child_invocation_attempted") is True
        and final.get("child_exit_code") == 1
        and final.get("capture_process_exit_code") == 1
        and final.get("failure_phase") == "child_exit_observed"
        and final.get("capture_argv", [None])[0] == bound_v1
        and final.get("formal_dataset_count") == 0
        and isinstance(phases, list)
        and [item.get("sequence") for item in phases] == [0, 1, 2]
        and [item.get("phase") for item in phases]
        == ["prelaunch_closed", "child_invocation_started", "child_exit_observed"],
        "Skokloster v1 terminal receipt drift",
    )
    expected_phase_values = (
        (paths["phase_prelaunch"], 0, "prelaunch_closed"),
        (paths["phase_child_start"], 1, "child_invocation_started"),
        (paths["phase_child_exit"], 2, "child_exit_observed"),
    )
    for path, sequence, phase in expected_phase_values:
        marker = _load_required(path, owner=f"v1 {phase} marker")
        base._require(
            marker.get("schema") == base.PHASE_SCHEMA
            and marker.get("status") == "entered"
            and marker.get("sequence") == sequence
            and marker.get("phase") == phase
            and marker.get("formal_dataset_count") == 0,
            f"Skokloster v1 phase marker drift: {path}",
        )
    base._require(
        paths["stdout"].is_file() and paths["stdout"].stat().st_size == 0,
        "Skokloster v1 stdout is not the observed empty log",
    )
    base._require(
        paths["stderr"].is_file()
        and V1_EXCEPTION in paths["stderr"].read_text(encoding="utf-8"),
        "Skokloster v1 stderr lacks the exact NumPy import failure",
    )
    base._require(
        not paths["capture_root"].exists(),
        "Skokloster v1 unexpectedly materialized capture output",
    )
    return {
        "paths": paths,
        "request": request,
        "final": final,
        "logical_v1": logical_v1,
        "bound_v1": bound_v1,
    }


def record_v1_terminal_failure(*, atom_root: Path) -> Path:
    atom_root = atom_root.resolve()
    expected = REPOSITORY / base.ATOM_DIRECTORY
    base._require(atom_root == expected, "Skokloster atom root drift")
    evidence = _assert_v1_terminal_failure(atom_root)
    paths = evidence["paths"]
    ledger_path = paths["ledger"]
    base._require(not ledger_path.exists(), "Skokloster v1 failure ledger exists")
    source_names = (
        "request",
        "dry_run",
        "running",
        "final",
        "stdout",
        "stderr",
        "phase_prelaunch",
        "phase_child_start",
        "phase_child_exit",
    )
    ledger = {
        "schema": V1_LEDGER_SCHEMA,
        "status": V1_FAILURE_STATUS,
        "candidate_revision": "v1_repository_venv_binding",
        "attempt_index": 1,
        "attempt_consumed": True,
        "retry_same_candidate_forbidden": True,
        "root_cause": {
            "phase": "python_import_before_capture_or_ue_initialization",
            "exception_type": "ModuleNotFoundError",
            "missing_module": "numpy",
            "configured_logical_interpreter": str(evidence["logical_v1"]),
            "request_bound_interpreter": evidence["bound_v1"],
            "binding_defect": (
                "non-authoritative repository .venv selected and Path.resolve() "
                "replaced its logical entry point with the base interpreter"
            ),
            "official_environment": "spear-env",
            "official_logical_interpreter": str(AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL),
        },
        "ordered_entered_phases": [
            {"sequence": 0, "phase": "prelaunch_closed"},
            {"sequence": 1, "phase": "child_invocation_started"},
            {"sequence": 2, "phase": "child_exit_observed"},
        ],
        "captured_frame_count": 0,
        "capture_artifact_count": 0,
        "capture_output_materialized": False,
        "exclusive_stdout_persisted": True,
        "exclusive_stderr_persisted": True,
        "complete_traceback_persisted": True,
        "child_exit_code": 1,
        "required_repo_commit": evidence["request"]["required_repo_commit"],
        "source_records": {
            name: base._file_record(paths[name]) for name in source_names
        },
        "official_env_contract": _origin_main_env_contract(),
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "recorded_at_utc": base._utc_now(),
    }
    base._write_json_exclusive(ledger_path, ledger)
    return ledger_path


def _validate_v1_ledger(path: Path) -> dict[str, Any]:
    ledger = _load_required(path, owner="Skokloster v1 terminal failure ledger")
    root_cause = ledger.get("root_cause")
    base._require(
        ledger.get("schema") == V1_LEDGER_SCHEMA
        and ledger.get("status") == V1_FAILURE_STATUS
        and ledger.get("attempt_consumed") is True
        and ledger.get("retry_same_candidate_forbidden") is True
        and ledger.get("captured_frame_count") == 0
        and ledger.get("capture_artifact_count") == 0
        and ledger.get("capture_output_materialized") is False
        and ledger.get("exclusive_stdout_persisted") is True
        and ledger.get("exclusive_stderr_persisted") is True
        and ledger.get("complete_traceback_persisted") is True
        and ledger.get("child_exit_code") == 1
        and ledger.get("ordered_entered_phases")
        == [
            {"sequence": 0, "phase": "prelaunch_closed"},
            {"sequence": 1, "phase": "child_invocation_started"},
            {"sequence": 2, "phase": "child_exit_observed"},
        ]
        and ledger.get("qualification_claim") is False
        and ledger.get("formal_dataset_count") == 0
        and isinstance(root_cause, Mapping)
        and root_cause.get("phase")
        == "python_import_before_capture_or_ue_initialization"
        and root_cause.get("missing_module") == "numpy"
        and root_cause.get("official_environment") == "spear-env"
        and root_cause.get("official_logical_interpreter")
        == str(AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL),
        "Skokloster v1 terminal failure ledger drift",
    )
    records = ledger.get("source_records")
    expected_record_names = {
        "request",
        "dry_run",
        "running",
        "final",
        "stdout",
        "stderr",
        "phase_prelaunch",
        "phase_child_start",
        "phase_child_exit",
    }
    base._require(
        isinstance(records, Mapping) and set(records) == expected_record_names,
        "v1 ledger source record closure drift",
    )
    for name, record in records.items():
        base._require(isinstance(record, Mapping), f"invalid v1 record: {name}")
        base._validate_file_record(record, owner=f"Skokloster v1 {name}")
    _validate_origin_main_env_contract(ledger.get("official_env_contract"))
    return ledger


def _dependency_file_records(payload: Mapping[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    dependencies = payload["dependencies"]
    for name in EXPECTED_DEPENDENCIES:
        records[name] = base._file_record(Path(dependencies[name]["origin"]))
    records["avengine"] = base._file_record(Path(payload["avengine_origin"]))
    records["capture_module"] = base._file_record(
        Path(payload["capture_module_origin"])
    )
    return records


def _validate_probe_payload(
    payload: Mapping[str, Any], *, logical_python: Path
) -> None:
    logical = _logical_absolute(logical_python)
    base._require(
        str(logical) == str(AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL),
        "capture interpreter is not the official spear-env logical entry",
    )
    base._require(
        payload.get("logical_interpreter") == str(logical)
        and payload.get("interpreter_realpath") == os.path.realpath(logical)
        and os.path.realpath(str(payload.get("sys_executable", "")))
        == os.path.realpath(logical)
        and Path(str(payload.get("sys_prefix", ""))) == AUTHORITATIVE_ENV_PREFIX
        and payload.get("capture_module_imported") is True
        and Path(str(payload.get("avengine_origin", ""))).resolve()
        == REPOSITORY / "src/avengine/__init__.py"
        and Path(str(payload.get("capture_module_origin", ""))).resolve()
        == REPOSITORY / "tools/qa/capture_skokloster_strict_two_human_episode.py"
        and payload.get("loaded_cuda_libraries") == []
        and payload.get("cuda_initialized") is False,
        "authoritative capture interpreter identity/import/CUDA probe failed",
    )
    dependencies = payload.get("dependencies")
    base._require(isinstance(dependencies, Mapping), "dependency probe is missing")
    base._require(
        set(dependencies) == set(EXPECTED_DEPENDENCIES),
        "dependency probe closure drift",
    )
    for name, expected in EXPECTED_DEPENDENCIES.items():
        observed = dependencies[name]
        base._require(isinstance(observed, Mapping), f"invalid dependency: {name}")
        for key, value in expected.items():
            base._require(
                observed.get(key) == value,
                f"authoritative dependency drift: {name}.{key}",
            )
        base._require(
            Path(str(observed.get("origin", ""))).is_file(),
            f"dependency origin is missing: {name}",
        )


def _probe_authoritative_interpreter(
    *, attempt_root: Path, logical_python: Path
) -> Path:
    logical = _logical_absolute(logical_python)
    base._require(
        str(logical) == str(AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL),
        "capture Python must use the official spear-env logical path",
    )
    base._require(logical.is_file(), "official spear-env Python is missing")
    stdout_path = attempt_root / "interpreter_probe_stdout.log"
    stderr_path = attempt_root / "interpreter_probe_stderr.log"
    receipt_path = attempt_root / "interpreter_preflight_receipt.json"
    for path in (stdout_path, stderr_path, receipt_path):
        base._require(not path.exists(), f"interpreter probe path exists: {path}")

    before = base._gpu_snapshot()
    base._validate_gpu1_idle(before)
    argv = [
        str(logical),
        "-c",
        INTERPRETER_PROBE_CODE,
        str(REPOSITORY),
        str(logical),
    ]
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(
            argv,
            cwd=REPOSITORY,
            check=False,
            stdout=stdout,
            stderr=stderr,
        )
    after = base._gpu_snapshot()
    base._validate_gpu1_idle(after)
    common = {
        "schema": PROBE_RECEIPT_SCHEMA,
        "candidate_revision": CANDIDATE_REVISION,
        "logical_interpreter": str(logical),
        "interpreter_realpath": os.path.realpath(logical),
        "execution_argv": argv,
        "child_exit_code": int(completed.returncode),
        "stdout": base._file_record(stdout_path),
        "stderr": base._file_record(stderr_path),
        "pre_probe_gpu_snapshot": before,
        "post_probe_gpu_snapshot": after,
        "gpu_process_started": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "recorded_at_utc": base._utc_now(),
    }
    try:
        base._require(completed.returncode == 0, "interpreter import probe failed")
        payload = json.loads(stdout_path.read_text(encoding="utf-8"))
        base._require(isinstance(payload, dict), "interpreter probe root is not object")
        _validate_probe_payload(payload, logical_python=logical)
        receipt = {
            **common,
            "status": "pass",
            "payload": payload,
            "dependency_file_records": _dependency_file_records(payload),
        }
    except Exception as exc:
        receipt = {
            **common,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        base._write_json_exclusive(receipt_path, receipt)
        raise
    base._write_json_exclusive(receipt_path, receipt)
    return receipt_path


def _validate_probe_receipt(path: Path) -> dict[str, Any]:
    receipt = _load_required(path, owner="interpreter preflight receipt")
    payload = receipt.get("payload")
    base._require(
        receipt.get("schema") == PROBE_RECEIPT_SCHEMA
        and receipt.get("status") == "pass"
        and receipt.get("candidate_revision") == CANDIDATE_REVISION
        and receipt.get("logical_interpreter")
        == str(AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL)
        and receipt.get("interpreter_realpath")
        == os.path.realpath(AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL)
        and receipt.get("execution_argv")
        == [
            str(AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL),
            "-c",
            INTERPRETER_PROBE_CODE,
            str(REPOSITORY),
            str(AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL),
        ]
        and receipt.get("child_exit_code") == 0
        and receipt.get("gpu_process_started") is False
        and receipt.get("qualification_claim") is False
        and receipt.get("formal_dataset_count") == 0
        and isinstance(payload, Mapping),
        "interpreter preflight receipt drift",
    )
    _validate_probe_payload(
        payload, logical_python=AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL
    )
    base._validate_gpu1_idle(receipt.get("pre_probe_gpu_snapshot", {}))
    base._validate_gpu1_idle(receipt.get("post_probe_gpu_snapshot", {}))
    for owner in ("stdout", "stderr"):
        record = receipt.get(owner)
        base._require(isinstance(record, Mapping), f"probe {owner} record missing")
        base._validate_file_record(record, owner=f"probe {owner}")
    records = receipt.get("dependency_file_records")
    expected_record_names = set(EXPECTED_DEPENDENCIES) | {
        "avengine",
        "capture_module",
    }
    base._require(
        isinstance(records, Mapping) and set(records) == expected_record_names,
        "probe dependency record closure drift",
    )
    for name, record in records.items():
        base._require(isinstance(record, Mapping), f"invalid probe record: {name}")
        base._validate_file_record(record, owner=f"probe dependency {name}")
    return receipt


def _v2_source_paths() -> dict[str, Path]:
    return {
        "v2_launcher": Path(__file__).resolve(),
        "v1_launcher": REPOSITORY
        / "tools/qa/run_strict_two_human_skokloster_f15_probe.py",
        "capture_wrapper": REPOSITORY
        / "tools/qa/capture_skokloster_strict_two_human_episode.py",
        "base_capture_runner": REPOSITORY
        / "tools/qa/capture_spear_native_pixel_episode.py",
        "official_env_recipe": _official_env_recipe(),
        "pre_gpu_launch_ledger": REPOSITORY / "tools/qa/pre_gpu_launch_ledger.py",
    }


def _record_identity(record: Mapping[str, Any], *, owner: str) -> PreservedFileIdentity:
    byte_size = record.get("byte_size")
    sha256 = record.get("sha256")
    base._require(
        isinstance(byte_size, int)
        and not isinstance(byte_size, bool)
        and byte_size >= 0
        and isinstance(sha256, str),
        f"{owner} identity is invalid",
    )
    return PreservedFileIdentity(byte_size=byte_size, sha256=sha256)


def _stale_v2_source_drift(
    request: Mapping[str, Any], *, repo_root: Path
) -> dict[str, dict[str, Any]]:
    records = request.get("candidate_source_records")
    base._require(isinstance(records, Mapping) and records, "v2 source records missing")
    drift: dict[str, dict[str, Any]] = {}
    seen_paths: set[Path] = set()
    for name, value in records.items():
        base._require(
            isinstance(name, str) and isinstance(value, Mapping),
            "v2 source record invalid",
        )
        raw_path = value.get("path")
        base._require(isinstance(raw_path, str), f"v2 source path invalid: {name}")
        path = Path(raw_path).resolve()
        base._require(
            path.is_relative_to(repo_root) and path not in seen_paths,
            f"v2 source path escapes or repeats: {name}",
        )
        seen_paths.add(path)
        recorded = _record_identity(value, owner=f"v2 source {name}")
        base._require(path.is_file(), f"current v2 source is missing: {name}")
        current = base._file_record(path)
        if (
            current["byte_size"] != recorded.byte_size
            or current["sha256"] != recorded.sha256
        ):
            drift[name] = {
                "path": str(path),
                "recorded": {
                    "byte_size": recorded.byte_size,
                    "sha256": recorded.sha256,
                },
                "current": {
                    "byte_size": current["byte_size"],
                    "sha256": current["sha256"],
                },
            }
    return drift


def archive_stale_preparation_v2(*, atom_root: Path) -> Path:
    """Archive the exact CPU-probed v2 request after source/HEAD drift."""

    repo_root = REPOSITORY.resolve()
    atom_root = atom_root.resolve()
    base._require(
        atom_root == repo_root / base.ATOM_DIRECTORY, "Skokloster atom root drift"
    )
    current_head = base._require_clean_head(repo_root)
    attempt_root = atom_root / V2_ATTEMPT_DIRECTORY
    archive_root = atom_root / V2_STALE_PREPARATION_ARCHIVE_DIRECTORY
    request_path = attempt_root / "request.json"
    request = _load_required(request_path, owner="stale Skokloster v2 request")
    base._require(set(request) == V2_REQUEST_KEYS, "stale v2 request key closure drift")
    base._require(
        request.get("schema") == REQUEST_SCHEMA
        and request.get("status") == "prepared_not_launched"
        and request.get("candidate_revision") == CANDIDATE_REVISION
        and request.get("episode_id") == base.EPISODE_ID
        and request.get("scene_id") == base.SCENE_ID
        and request.get("formal_dataset_count") == 0
        and request.get("qualification_claim") is False
        and request.get("gpu_capture_authorized_at_prepare") is False
        and request.get("full75_allowed") is False
        and request.get("frame_indices") == [base.FRAME_INDEX],
        "stale v2 request identity drift",
    )
    recorded_head = request.get("required_repo_commit")
    base._require(
        isinstance(recorded_head, str) and recorded_head != current_head,
        "stale v2 archive requires repository HEAD drift",
    )
    source_drift = _stale_v2_source_drift(request, repo_root=repo_root)
    base._require(source_drift, "stale v2 archive requires source-record drift")

    probe_path = attempt_root / "interpreter_preflight_receipt.json"
    probe = _validate_probe_receipt(probe_path)
    probe_record = request.get("interpreter_preflight_receipt")
    base._require(isinstance(probe_record, Mapping), "request probe record missing")
    base._validate_file_record(probe_record, owner="request interpreter probe")
    stdout = probe.get("stdout")
    stderr = probe.get("stderr")
    base._require(
        isinstance(stdout, Mapping) and isinstance(stderr, Mapping),
        "probe log records missing",
    )
    preserved = {
        "interpreter_preflight_receipt.json": _record_identity(
            probe_record, owner="interpreter preflight receipt"
        ),
        "interpreter_probe_stdout.log": _record_identity(
            stdout, owner="interpreter probe stdout"
        ),
        "interpreter_probe_stderr.log": _record_identity(
            stderr, owner="interpreter probe stderr"
        ),
    }
    expected_paths = {
        "repo_root": repo_root,
        "atom_root": atom_root,
        "attempt_root": attempt_root,
        "capture_output": atom_root / V2_CAPTURE_DIRECTORY,
        "capture_stdout": attempt_root / "capture_stdout.log",
        "capture_stderr": attempt_root / "capture_stderr.log",
    }
    spec = PreparedAttemptSpec(
        request_schema=REQUEST_SCHEMA,
        request_keys=V2_REQUEST_KEYS,
        workspace_roots=(atom_root,),
        expected_fields={
            "candidate_revision": CANDIDATE_REVISION,
            "episode_id": base.EPISODE_ID,
            "scene_id": base.SCENE_ID,
            "required_repo_commit": recorded_head,
            "required_clean_worktree": True,
            "frame_indices": [base.FRAME_INDEX],
            "full75_allowed": False,
            "physical_gpu_index": 1,
            "physical_gpu_uuid": base.GPU1_UUID,
            "graphics_adapter_argument": 1,
            "required_idle_compute_process_count": 0,
            "rpc_port": V2_RPC_PORT,
            "gpu_capture_authorized_at_prepare": False,
            "qualification_claim": False,
            "formal_dataset_count": 0,
        },
        expected_paths=expected_paths,
        forbidden_paths=(atom_root / V2_CAPTURE_DIRECTORY,),
        preserved_files=preserved,
    )
    reason = json.dumps(
        {
            "code": "source_record_and_repository_head_drift",
            "recorded_head": recorded_head,
            "current_head": current_head,
            "source_drift": source_drift,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return archive_prepared_attempt(
        attempt_root=attempt_root,
        archive_root=archive_root,
        spec=spec,
        reason=reason,
    )


def _capture_argv_v2(request: Mapping[str, Any]) -> list[str]:
    return [
        str(request["capture_python_logical"]),
        str(request["capture_script"]),
        "--suite-plan",
        str(request["suite_plan"]),
        "--scenario-id",
        base.EPISODE_ID,
        "--audio-wav",
        str(request["audio_wav"]),
        "--spear-root",
        str(request["spear_root"]),
        "--spear-executable",
        str(request["packaged_executable"]),
        "--output",
        str(request["capture_output"]),
        "--rpc-port",
        str(request["rpc_port"]),
        "--graphics-adapter",
        "1",
        "--warmup-frames",
        "40",
        "--frame-index",
        str(base.FRAME_INDEX),
        "--authorize-gpu-capture",
    ]


def prepare_request_v2(
    *, atom_root: Path, capture_python: Path, spear_root: Path, rpc_port: int
) -> Path:
    repo_root = REPOSITORY.resolve()
    atom_root = atom_root.resolve()
    base._require(
        atom_root == repo_root / base.ATOM_DIRECTORY,
        "Skokloster v2 atom root drift",
    )
    required_commit = base._require_clean_head(repo_root)
    attempt_root = atom_root / V2_ATTEMPT_DIRECTORY
    capture_output = atom_root / V2_CAPTURE_DIRECTORY
    base._require(not attempt_root.exists(), "Skokloster v2 attempt 01 exists")
    base._require(not capture_output.exists(), "Skokloster v2 capture output exists")
    base._require(rpc_port == V2_RPC_PORT, "Skokloster v2 RPC port drift")
    logical = _logical_absolute(capture_python)
    base._require(
        str(logical) == str(AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL),
        "Skokloster v2 requires the official spear-env logical interpreter",
    )

    origin_main_env_contract = _origin_main_env_contract()
    v1_ledger = atom_root / base.ATTEMPT_DIRECTORY / "failure_ledger.json"
    _validate_v1_ledger(v1_ledger)
    artifact_paths = base._artifact_paths(atom_root)
    base._require(
        all(path.is_file() for path in artifact_paths.values()),
        "accepted CPU artifact is missing",
    )
    cpu_validation = base._validate_cpu_evidence(artifact_paths)
    package_paths = base._package_paths()
    base._require(
        all(path.is_file() for path in package_paths.values()),
        "Development archive file is missing",
    )
    source_paths = _v2_source_paths()
    base._require(
        all(path.is_file() for path in source_paths.values()),
        "Skokloster v2 source is missing",
    )
    base._require(spear_root.is_dir(), "SPEAR root is missing")

    attempt_root.mkdir(parents=True, exist_ok=False)
    probe_path = _probe_authoritative_interpreter(
        attempt_root=attempt_root, logical_python=logical
    )
    stdout_path = attempt_root / "capture_stdout.log"
    stderr_path = attempt_root / "capture_stderr.log"
    request = {
        "schema": REQUEST_SCHEMA,
        "status": "prepared_not_launched",
        "candidate_revision": CANDIDATE_REVISION,
        "candidate_change_contract": V2_CHANGE_CONTRACT,
        "episode_id": base.EPISODE_ID,
        "scene_id": base.SCENE_ID,
        "repo_root": str(repo_root),
        "required_repo_commit": required_commit,
        "required_clean_worktree": True,
        "atom_root": str(atom_root),
        "attempt_root": str(attempt_root),
        "capture_output": str(capture_output),
        "capture_stdout": str(stdout_path),
        "capture_stderr": str(stderr_path),
        "capture_python": str(logical),
        "capture_python_logical": str(logical),
        "capture_python_realpath": os.path.realpath(logical),
        "capture_script": str(source_paths["capture_wrapper"].resolve()),
        "spear_root": str(spear_root.resolve()),
        "suite_plan": str(artifact_paths["suite_plan"].resolve()),
        "audio_wav": str(artifact_paths["binaural_mixture"].resolve()),
        "packaged_map": base.PACKAGED_MAP,
        "packaged_executable": str(base.PACKAGED_EXECUTABLE),
        "artifact_records": {
            name: base._file_record(path) for name, path in artifact_paths.items()
        },
        "package_records": {
            name: base._file_record(path) for name, path in package_paths.items()
        },
        "candidate_source_records": {
            name: base._file_record(path) for name, path in source_paths.items()
        },
        "official_env_contract": origin_main_env_contract,
        "interpreter_preflight_receipt": base._file_record(probe_path),
        "predecessor_v1_failure_ledger": base._file_record(v1_ledger),
        "cpu_validation": cpu_validation,
        "attempt_policy": V2_ATTEMPT_POLICY,
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
            "target_minimum_visible_fraction": (base.TARGET_VISIBLE_FRACTION_MINIMUM),
            "distractor_minimum_visible_fraction": (
                base.DISTRACTOR_VISIBLE_FRACTION_MINIMUM
            ),
            "visible_pixel_count_minimum": base.VISIBLE_PIXEL_COUNT_MINIMUM,
            "bbox_edge_margin_px_minimum": base.BBOX_EDGE_MARGIN_PX_MINIMUM,
        },
        "mp3d_revision_v2_terminal_required_before_real_launch": True,
        "mp3d_revision_v2_terminal_receipt": str(base.MP3D_V2_TERMINAL_RECEIPT),
        "explicit_gpu_capture_authorization_required": True,
        "gpu_capture_authorized_at_prepare": False,
        "manual_visual_review_required": True,
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "created_at_utc": base._utc_now(),
    }
    request_path = attempt_root / "request.json"
    base._write_json_exclusive(request_path, request)
    return request_path


def _validate_record_set(
    records: object, expected: Mapping[str, Path], *, owner: str
) -> None:
    base._require(
        isinstance(records, Mapping) and set(records) == set(expected),
        f"{owner} record closure drift",
    )
    for name, path in expected.items():
        record = records[name]
        base._require(isinstance(record, Mapping), f"invalid {owner}: {name}")
        observed = base._validate_file_record(record, owner=f"{owner}.{name}")
        base._require(observed == path.resolve(), f"{owner}.{name} path drift")


def _validate_request_v2(request_path: Path) -> tuple[dict[str, Any], list[str]]:
    request_path = request_path.resolve()
    request = base._load(request_path)
    base._require(request.get("schema") == REQUEST_SCHEMA, "v2 request schema drift")
    base._require(
        request.get("status") == "prepared_not_launched"
        and request.get("candidate_revision") == CANDIDATE_REVISION
        and request.get("candidate_change_contract") == V2_CHANGE_CONTRACT
        and request.get("episode_id") == base.EPISODE_ID
        and request.get("scene_id") == base.SCENE_ID,
        "v2 request identity drift",
    )
    repo_root = Path(str(request.get("repo_root", ""))).resolve()
    base._require(repo_root == REPOSITORY.resolve(), "v2 repository drift")
    observed_head = base._require_clean_head(repo_root)
    base._require(
        request.get("required_repo_commit") == observed_head,
        "v2 request-bound repository commit drift",
    )
    atom_root = repo_root / base.ATOM_DIRECTORY
    attempt_root = atom_root / V2_ATTEMPT_DIRECTORY
    capture_output = atom_root / V2_CAPTURE_DIRECTORY
    stdout_path = attempt_root / "capture_stdout.log"
    stderr_path = attempt_root / "capture_stderr.log"
    base._require(
        request_path == attempt_root / "request.json"
        and Path(str(request.get("attempt_root", ""))).resolve() == attempt_root
        and Path(str(request.get("capture_output", ""))).resolve() == capture_output
        and Path(str(request.get("capture_stdout", ""))).resolve() == stdout_path
        and Path(str(request.get("capture_stderr", ""))).resolve() == stderr_path,
        "v2 request/output/log path drift",
    )
    logical = str(AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL)
    base._require(
        request.get("capture_python") == logical
        and request.get("capture_python_logical") == logical
        and request.get("capture_python_realpath") == os.path.realpath(logical)
        and Path(logical).is_file(),
        "v2 logical/real capture interpreter binding drift",
    )
    base._require(
        request.get("attempt_policy") == V2_ATTEMPT_POLICY
        and request.get("frame_indices") == [base.FRAME_INDEX]
        and request.get("full75_allowed") is False
        and request.get("physical_gpu_index") == 1
        and request.get("physical_gpu_uuid") == base.GPU1_UUID
        and request.get("graphics_adapter_argument") == 1
        and request.get("required_idle_compute_process_count") == 0
        and request.get("rpc_port") == V2_RPC_PORT,
        "v2 f15/attempt/GPU/RPC contract drift",
    )
    base._require(
        request.get("packaged_map") == base.PACKAGED_MAP
        and Path(str(request.get("packaged_executable", ""))).resolve()
        == base.PACKAGED_EXECUTABLE.resolve()
        and request.get("explicit_gpu_capture_authorization_required") is True
        and request.get("gpu_capture_authorized_at_prepare") is False
        and request.get("qualification_claim") is False
        and request.get("formal_dataset_count") == 0,
        "v2 archive/authorization/formal boundary drift",
    )
    base._require(
        request.get("visibility_gate")
        == {
            "target_instance_id": "source1",
            "distractor_instance_id": "source2",
            "target_minimum_visible_fraction": base.TARGET_VISIBLE_FRACTION_MINIMUM,
            "distractor_minimum_visible_fraction": (
                base.DISTRACTOR_VISIBLE_FRACTION_MINIMUM
            ),
            "visible_pixel_count_minimum": base.VISIBLE_PIXEL_COUNT_MINIMUM,
            "bbox_edge_margin_px_minimum": base.BBOX_EDGE_MARGIN_PX_MINIMUM,
        }
        and request.get("mp3d_revision_v2_terminal_required_before_real_launch") is True
        and request.get("mp3d_revision_v2_terminal_receipt")
        == str(base.MP3D_V2_TERMINAL_RECEIPT)
        and request.get("manual_visual_review_required") is True,
        "v2 visibility/upstream/review boundary drift",
    )

    artifact_paths = base._artifact_paths(atom_root)
    package_paths = base._package_paths()
    source_paths = _v2_source_paths()
    _validate_record_set(
        request.get("artifact_records"), artifact_paths, owner="v2 artifact"
    )
    _validate_record_set(
        request.get("package_records"), package_paths, owner="v2 package"
    )
    _validate_record_set(
        request.get("candidate_source_records"), source_paths, owner="v2 source"
    )
    base._validate_cpu_evidence(artifact_paths)
    _validate_origin_main_env_contract(request.get("official_env_contract"))

    ledger_path = atom_root / base.ATTEMPT_DIRECTORY / "failure_ledger.json"
    ledger_record = request.get("predecessor_v1_failure_ledger")
    base._require(isinstance(ledger_record, Mapping), "v2 v1-ledger record missing")
    base._require(
        base._validate_file_record(ledger_record, owner="v2 v1 ledger") == ledger_path,
        "v2 predecessor ledger path drift",
    )
    _validate_v1_ledger(ledger_path)
    probe_path = attempt_root / "interpreter_preflight_receipt.json"
    probe_record = request.get("interpreter_preflight_receipt")
    base._require(isinstance(probe_record, Mapping), "v2 probe record missing")
    base._require(
        base._validate_file_record(probe_record, owner="v2 interpreter probe")
        == probe_path,
        "v2 interpreter probe path drift",
    )
    _validate_probe_receipt(probe_path)
    base._require(not capture_output.exists(), "v2 capture output must be fresh")

    argv = _capture_argv_v2(request)
    base._require(argv[0] == logical, "v2 argv did not preserve logical Python")
    if logical != request["capture_python_realpath"]:
        base._require(
            argv[0] != request["capture_python_realpath"],
            "v2 argv substituted the interpreter realpath",
        )
    base._require(
        argv.count("--frame-index") == 1
        and argv[argv.index("--frame-index") + 1] == "15"
        and argv.count("--graphics-adapter") == 1
        and argv[argv.index("--graphics-adapter") + 1] == "1"
        and argv.count("--authorize-gpu-capture") == 1,
        "v2 capture argv crossed f15/GPU1/authorization boundary",
    )
    return request, argv


def _write_phase_v2(
    attempt_root: Path,
    *,
    sequence: int,
    phase: str,
    detail: Mapping[str, Any] | None = None,
) -> Path:
    path = attempt_root / f"launch_phase_{sequence:03d}_{phase}.json"
    base._write_json_exclusive(
        path,
        {
            "schema": PHASE_SCHEMA,
            "status": "entered",
            "sequence": sequence,
            "phase": phase,
            "detail": dict(detail or {}),
            "qualification_claim": False,
            "formal_dataset_count": 0,
            "captured_at_utc": base._utc_now(),
        },
    )
    return path


def _collect_phases_v2(attempt_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(attempt_root.glob("launch_phase_*.json")):
        value = base._load(path)
        base._require(
            value.get("schema") == PHASE_SCHEMA
            and value.get("status") == "entered"
            and value.get("formal_dataset_count") == 0,
            f"invalid v2 phase marker: {path}",
        )
        records.append(
            {
                "sequence": value.get("sequence"),
                "phase": value.get("phase"),
                "artifact": base._file_record(path),
            }
        )
    base._require(
        [item["sequence"] for item in records] == list(range(len(records))),
        "v2 phase sequence is not contiguous",
    )
    return records


def _validate_dry_receipt_v2(
    path: Path,
    *,
    request_path: Path,
    request: Mapping[str, Any],
    argv: Sequence[str],
) -> dict[str, Any]:
    receipt = _load_required(path, owner="Skokloster v2 dry-run receipt")
    base._require(
        receipt.get("schema") == RECEIPT_SCHEMA
        and receipt.get("status") == "dry_run_pass_not_launched"
        and receipt.get("candidate_revision") == CANDIDATE_REVISION
        and receipt.get("required_repo_commit") == request.get("required_repo_commit")
        and receipt.get("request") == str(request_path.resolve())
        and receipt.get("capture_argv") == list(argv)
        and receipt.get("capture_python_logical")
        == request.get("capture_python_logical")
        and receipt.get("capture_python_realpath")
        == request.get("capture_python_realpath")
        and receipt.get("frame_indices") == [base.FRAME_INDEX]
        and receipt.get("full75_allowed") is False
        and receipt.get("gpu_started") is False
        and receipt.get("attempt_consumed") is False
        and receipt.get("qualification_claim") is False
        and receipt.get("formal_dataset_count") == 0,
        "Skokloster v2 dry-run receipt/request binding drift",
    )
    return receipt


def run_v2(
    request_path: Path,
    *,
    dry_run: bool,
    authorize_gpu_capture: bool,
    mp3d_v2_terminal_receipt: Path,
) -> int:
    request, argv = _validate_request_v2(request_path)
    attempt_root = Path(request["attempt_root"])
    stdout_path = Path(request["capture_stdout"])
    stderr_path = Path(request["capture_stderr"])
    dry_receipt = attempt_root / "dry_run_receipt.json"
    running_receipt = attempt_root / "running_receipt.json"
    final_receipt = attempt_root / "final_receipt.json"
    base._require(not final_receipt.exists(), "Skokloster v2 final receipt exists")
    if dry_run:
        base._require(not dry_receipt.exists(), "Skokloster v2 dry receipt exists")
        base._require(not running_receipt.exists(), "Skokloster v2 already started")
    else:
        base._require(
            authorize_gpu_capture,
            "Skokloster v2 GPU capture lacks explicit launch authorization",
        )
        _validate_dry_receipt_v2(
            dry_receipt,
            request_path=request_path,
            request=request,
            argv=argv,
        )
        base._require(not running_receipt.exists(), "Skokloster v2 already started")
        base._require(
            not stdout_path.exists() and not stderr_path.exists(),
            "Skokloster v2 exclusive child log exists",
        )

    before = base._gpu_snapshot()
    gpu = base._validate_gpu1_idle(before)
    base._assert_port_available(int(request["rpc_port"]))
    common = {
        "schema": RECEIPT_SCHEMA,
        "candidate_revision": CANDIDATE_REVISION,
        "episode_id": base.EPISODE_ID,
        "scene_id": base.SCENE_ID,
        "attempt_policy": V2_ATTEMPT_POLICY,
        "required_repo_commit": request["required_repo_commit"],
        "request": str(request_path.resolve()),
        "capture_argv": argv,
        "capture_python_logical": request["capture_python_logical"],
        "capture_python_realpath": request["capture_python_realpath"],
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

    upstream = base._validate_mp3d_terminal(mp3d_v2_terminal_receipt)
    common["mp3d_revision_v2_terminal_receipt"] = upstream
    common["dry_run_receipt"] = base._file_record(dry_receipt)
    started_at = base._utc_now()
    base._write_json_exclusive(
        running_receipt,
        {
            **common,
            "status": "running",
            "gpu_started": False,
            "attempt_consumed": True,
            "retry_same_candidate_forbidden": True,
            "started_at_utc": started_at,
            "child_invocation_attempted": False,
            "child_exit_code": None,
        },
    )
    exit_code = 1
    child_invocation_attempted = False
    child_exit_code: int | None = None
    current_phase = "prelaunch_closed"
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
        _write_phase_v2(
            attempt_root,
            sequence=0,
            phase="prelaunch_closed",
            detail={
                "mp3d_v2_terminal_bound": True,
                "gpu1_idle": True,
                "logical_interpreter_preserved": True,
            },
        )
        current_phase = "child_invocation_started"
        _write_phase_v2(
            attempt_root,
            sequence=1,
            phase=current_phase,
            detail={"argv_count": len(argv), "argv0": argv[0]},
        )
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            child_invocation_attempted = True
            completed = subprocess.run(
                argv,
                cwd=REPOSITORY,
                check=False,
                stdout=stdout,
                stderr=stderr,
            )
        child_exit_code = int(completed.returncode)
        exit_code = child_exit_code
        current_phase = "child_exit_observed"
        _write_phase_v2(
            attempt_root,
            sequence=2,
            phase=current_phase,
            detail={"returncode": child_exit_code},
        )
        base._require(child_exit_code == 0, f"Skokloster v2 exited {child_exit_code}")
        current_phase = "capture_validation_started"
        _write_phase_v2(attempt_root, sequence=3, phase=current_phase)
        final["validation"] = base._validate_capture(request)
        current_phase = "complete"
        _write_phase_v2(attempt_root, sequence=4, phase=current_phase)
        final["status"] = "pass_diagnostic_f15_manual_review_pending"
    except Exception as exc:
        final["error"] = f"{type(exc).__name__}: {exc}"
        final["failure_phase"] = current_phase
        final["launcher_traceback"] = traceback.format_exc()
        exit_code = exit_code or 1
    finally:
        final["ended_at_utc"] = base._utc_now()
        final["child_invocation_attempted"] = child_invocation_attempted
        final["child_exit_code"] = child_exit_code
        final["capture_process_exit_code"] = child_exit_code
        final["gpu_started"] = child_invocation_attempted
        final["exclusive_child_stdout"] = (
            base._file_record(stdout_path) if stdout_path.is_file() else None
        )
        final["exclusive_child_stderr"] = (
            base._file_record(stderr_path) if stderr_path.is_file() else None
        )
        try:
            final["launcher_phases"] = _collect_phases_v2(attempt_root)
        except Exception as exc:
            final["launcher_phase_collection_error"] = f"{type(exc).__name__}: {exc}"
            final["launcher_phase_collection_traceback"] = traceback.format_exc()
        try:
            postlaunch_snapshot = base._gpu_snapshot()
            final["postlaunch_snapshot"] = postlaunch_snapshot
            base._validate_gpu1_idle(postlaunch_snapshot)
            final["postlaunch_gpu1_released"] = True
        except Exception as exc:
            final["postlaunch_gpu1_released"] = False
            final["postlaunch_snapshot_error"] = f"{type(exc).__name__}: {exc}"
            final["status"] = "failed"
            exit_code = 1
        base._write_json_exclusive(final_receipt, final)
    return exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ledger = subparsers.add_parser("record-v1-failure")
    ledger.add_argument("--atom-root", required=True, type=Path)
    archive = subparsers.add_parser("archive-stale-v2-preparation")
    archive.add_argument("--atom-root", required=True, type=Path)
    prepare = subparsers.add_parser("prepare-v2")
    prepare.add_argument("--atom-root", required=True, type=Path)
    prepare.add_argument(
        "--capture-python",
        type=Path,
        default=AUTHORITATIVE_CAPTURE_PYTHON_LOGICAL,
    )
    prepare.add_argument("--spear-root", type=Path, default=base.SPEAR_ROOT)
    prepare.add_argument("--rpc-port", type=int, default=V2_RPC_PORT)
    launch = subparsers.add_parser("launch-v2")
    launch.add_argument("--request", required=True, type=Path)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--authorize-gpu-capture", action="store_true")
    launch.add_argument(
        "--mp3d-v2-terminal-receipt",
        type=Path,
        default=base.MP3D_V2_TERMINAL_RECEIPT,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "record-v1-failure":
        path = record_v1_terminal_failure(atom_root=args.atom_root)
        print(f"SKOK_V1_FAILURE_LEDGER_RECORDED ledger={path} formal=0", flush=True)
        return 0
    if args.command == "archive-stale-v2-preparation":
        path = archive_stale_preparation_v2(atom_root=args.atom_root)
        print(f"SKOK_F15_V2_PREPARATION_ARCHIVED receipt={path} formal=0", flush=True)
        return 0
    if args.command == "prepare-v2":
        path = prepare_request_v2(
            atom_root=args.atom_root,
            capture_python=args.capture_python,
            spear_root=args.spear_root,
            rpc_port=args.rpc_port,
        )
        print(f"SKOK_F15_V2_REQUEST_PREPARED request={path} formal=0", flush=True)
        return 0
    return run_v2(
        args.request,
        dry_run=args.dry_run,
        authorize_gpu_capture=args.authorize_gpu_capture,
        mp3d_v2_terminal_receipt=args.mp3d_v2_terminal_receipt,
    )


if __name__ == "__main__":
    raise SystemExit(main())

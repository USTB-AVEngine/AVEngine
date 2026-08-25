#!/usr/bin/env python3
# HISTORICAL TOOL (single-repo closure, 2026-08-21): this script built or
# validates retained strict-two-human evidence recorded against the
# pre-closure transition environment (sibling Habitat fork, sound-spaces,
# SPEAR-lead-b, and multi-repo SPEAR checkouts). The hard-coded absolute
# paths below are a frozen historical record, not current inputs. The current
# production chain runs on the installed runtime prefix and external data
# roots under /data/avengine_external; do not use this tool for new work.
"""Prepare and launch one MP3D strict-two-human diagnostic f15 probe.

The launcher is deliberately fail closed.  A request binds the current Git
commit and the accepted CPU evidence, then exactly one real launch may create
immutable RUNNING and FINAL receipts.  Dry runs do not consume the attempt.
No path in this workflow authorizes a full 75-frame GPU capture or a formal
dataset increment.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import socket
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

REQUEST_SCHEMA = "avengine_mp3d_strict_two_human_f15_launch_request_v1"
RECEIPT_SCHEMA = "avengine_mp3d_strict_two_human_f15_launch_receipt_v1"
REQUEST_SCHEMA_V2 = "avengine_mp3d_strict_two_human_f15_launch_request_v2"
RECEIPT_SCHEMA_V2 = "avengine_mp3d_strict_two_human_f15_launch_receipt_v2"
REQUEST_SCHEMA_V5 = "avengine_mp3d_strict_two_human_f15_launch_request_v5"
RECEIPT_SCHEMA_V5 = "avengine_mp3d_strict_two_human_f15_launch_receipt_v5"
REQUEST_SCHEMA_V6 = "avengine_mp3d_strict_two_human_f15_launch_request_v6"
RECEIPT_SCHEMA_V6 = "avengine_mp3d_strict_two_human_f15_launch_receipt_v6"
REQUEST_SCHEMA_V7 = "avengine_mp3d_strict_two_human_f15_launch_request_v7"
RECEIPT_SCHEMA_V7 = "avengine_mp3d_strict_two_human_f15_launch_receipt_v7"
REQUEST_SCHEMA_V8 = "avengine_mp3d_strict_two_human_f15_launch_request_v8"
RECEIPT_SCHEMA_V8 = "avengine_mp3d_strict_two_human_f15_launch_receipt_v8"
VALIDATION_ONLY_RECEIPT_SCHEMA_V8 = (
    "avengine_mp3d_strict_two_human_f15_validation_only_receipt_v8"
)
FAILURE_LEDGER_SCHEMA = "avengine_mp3d_f15_attempt_failure_ledger_v1"
CAPTURE_FAILURE_SCHEMA = "avengine_mp3d_f15_capture_failure_v1"
ATTEMPT01_FAILURE_STATUS = (
    "undetermined_observability_gap_after_entry_init_before_first_capture_artifact"
)
VISIBILITY_SCHEMA = "avengine_qa_pixel_visibility_truth_v1"
EPISODE_ID = "mp3d_17DRP5sb8fy_male_female_static_0001"
SCENE_ID = "17DRP5sb8fy"
GPU1_UUID = "GPU-6d3e273e-58c6-2a5b-480a-4816fef6c581"
CAPTURE_PYTHON_LOGICAL = Path("/data/jzy/miniconda3/envs/spear-env/bin/python")
SPEAR_ROOT = Path("/data/jzy/code/AVEngine/external/SPEAR")
EXPECTED_MESH_COUNT = 71
FRAME_INDEX = 15
HEIGHT = 720
WIDTH = 1280
TARGET_ONLY_BACKGROUND_DEPTH_M = 65504.0
ABSOLUTE_TOLERANCE_M = 0.01
RELATIVE_TOLERANCE = 0.002
TARGET_MINIMUM_VISIBLE_FRACTION = 0.8
DISTRACTOR_MINIMUM_VISIBLE_FRACTION = 0.5
V2_RPC_PORT = 39632
V2_ATTEMPT_DIRECTORY = "diagnostic_f15_revision_v2_launch_attempt_01"
V2_CAPTURE_DIRECTORY = "diagnostic_f15_revision_v2_capture_attempt_01"
V5_ATTEMPT_DIRECTORY = "diagnostic_f15_execution_plan_v5_launch_attempt_01"
V6_ATTEMPT_DIRECTORY = "diagnostic_f15_execution_plan_v6_launch_attempt_01"
V7_ATTEMPT_DIRECTORY = "diagnostic_f15_revision_v7_launch_attempt_01"
V7_CAPTURE_DIRECTORY = "native_sparse_f15_v7"
V7_RPC_PORT = 39637
V8_ATTEMPT_DIRECTORY = "diagnostic_f15_revision_v8_launch_attempt_01"
V8_CAPTURE_DIRECTORY = "native_sparse_f15_v8"
V8_RPC_PORT = 39638
PACKAGED_ROOM_READBACK_DIRECTORY = "packaged_room_readback_v1"
ATTEMPT_POLICY = {
    "attempt_index": 1,
    "maximum_attempts_for_candidate": 1,
    "retry_same_candidate_forbidden": True,
    "failure_disposition": "reject_candidate_without_same_candidate_retry",
}
V2_ATTEMPT_POLICY = {
    **ATTEMPT_POLICY,
    "candidate_revision": "revision_v2_observability_only",
    "predecessor_v1_attempt_retry": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _write_json_atomic_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete JSON receipt without replacing an existing path."""

    _require(
        not path.exists() and not path.is_symlink(),
        f"output receipt already exists: {path}",
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise RuntimeError(f"output receipt already exists: {path}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _file_record(path: Path) -> dict[str, Any]:
    """Return a path-only reference for a required file.

    Older request JSON may contain additional legacy fields.  Readers ignore
    them: launcher admission is based on path containment and parsed semantic
    closure, not file-byte identity.
    """

    return {"path": str(path.resolve())}


def _validate_file_record(record: Mapping[str, Any], *, owner: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    _require(path.is_file(), f"{owner} is missing: {path}")
    return path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_contained(path: Path, root: Path, *, owner: str) -> Path:
    resolved = path.resolve()
    _require(_is_relative_to(resolved, root.resolve()), f"{owner} escapes {root}")
    return resolved


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _git_tracked_and_index_clean(repo_root: Path) -> bool:
    diffs = [
        subprocess.run(
            argv,
            cwd=repo_root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for argv in (
            ["git", "diff", "--quiet", "--"],
            ["git", "diff", "--cached", "--quiet", "--"],
        )
    ]
    untracked = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    return (
        all(result.returncode == 0 for result in diffs)
        and untracked.returncode == 0
        and not untracked.stdout.strip()
    )


def _require_v7_nonsymlink_path(path: Path, root: Path, *, owner: str) -> Path:
    raw = path.absolute()
    raw_root = root.absolute()
    _require(_is_relative_to(raw, raw_root), f"{owner} escapes {root}")
    components = [raw]
    while components[-1] != raw_root:
        components.append(components[-1].parent)
    _require(
        not any(candidate.is_symlink() for candidate in components),
        f"{owner} has a symlink component",
    )
    return _require_contained(raw, raw_root, owner=owner)


def _nvidia_csv(query_kind: str, fields: str) -> list[list[str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{query_kind}={fields}",
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
    gpus = _nvidia_csv("gpu", "index,uuid,name,memory.used,utilization.gpu")
    apps = _nvidia_csv("compute-apps", "gpu_uuid,pid,process_name,used_memory")
    return {
        "captured_at_utc": _utc_now(),
        "gpus": [
            {
                "physical_index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_used_mib": int(memory),
                "utilization_percent": int(utilization),
            }
            for index, uuid, name, memory, utilization in gpus
        ],
        "compute_apps": [
            {
                "gpu_uuid": uuid,
                "pid": int(pid),
                "process_name": name,
                "used_memory_mib": int(memory),
            }
            for uuid, pid, name, memory in apps
        ],
    }


def _validate_gpu1_idle(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    gpus = [
        item for item in snapshot.get("gpus", []) if item.get("physical_index") == 1
    ]
    _require(len(gpus) == 1, "physical GPU1 did not resolve exactly once")
    gpu = gpus[0]
    _require(gpu.get("uuid") == GPU1_UUID, "physical GPU1 UUID drift")
    apps = [
        item
        for item in snapshot.get("compute_apps", [])
        if item.get("gpu_uuid") == GPU1_UUID
    ]
    _require(not apps, f"physical GPU1 is not idle: {apps}")
    return dict(gpu)


def _assert_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def _is_authoritative_capture_python(path: Path) -> bool:
    """Accept the pinned logical executable or its exact resolved symlink target."""

    return path.resolve() == CAPTURE_PYTHON_LOGICAL.resolve()


def _artifact_paths(atom_root: Path) -> dict[str, Path]:
    preflight = atom_root / "cpu_preflight_v4"
    package = atom_root / "fresh_soundspaces2_package_v1"
    cache = atom_root / "exact_rir_cache_v4"
    return {
        "preflight": preflight / "preflight.json",
        "room_adapter": preflight / "room_adapter.json",
        "suite_plan": preflight / "suite_execution_plan.json",
        "rir_runtime_probe": preflight / "rir_runtime_probe.json",
        "package_manifest": package / "manifest.json",
        "package_material_coverage": package / "semantic_material_coverage.json",
        "rir_cache_receipt": cache / "receipt.json",
        "rir_cache_index": cache / "index.json",
    }


def _execution_plan_artifact_paths(execution_plan_path: Path) -> dict[str, Path]:
    """Resolve the admitted v2 execution plan without guessing versioned dirs."""

    execution_plan_path = execution_plan_path.resolve()
    plan = _load(execution_plan_path)
    _require(
        plan.get("schema")
        in {
            "avengine_native_strict_two_human_mp3d_execution_plan_v1",
            "avengine_native_strict_two_human_mp3d_execution_plan_v2",
        },
        "MP3D execution plan schema drift",
    )
    _require(
        plan.get("qualification_claim") is False
        and plan.get("formal_dataset_count") == 0,
        "execution plan crossed the non-formal boundary",
    )
    preflight_root = execution_plan_path.parent.resolve()
    atom_root = preflight_root.parent.resolve()
    repo_root = REPOSITORY.resolve()
    _require_contained(preflight_root, repo_root / "tmp", owner="preflight root")
    _require_contained(atom_root, repo_root / "tmp", owner="atom root")
    _require(
        Path(str(plan.get("local_staging_output", ""))).resolve() == preflight_root,
        "execution plan staging root drift",
    )
    _require(
        Path(str(plan.get("remote_target_root", ""))).resolve() == atom_root,
        "execution plan atom root drift",
    )

    cpu_steps = plan.get("cpu_steps")
    gpu_steps = plan.get("gpu_steps")
    _require(isinstance(cpu_steps, list), "execution plan CPU steps are missing")
    _require(isinstance(gpu_steps, list), "execution plan GPU steps are missing")
    by_cpu_id = {
        str(step.get("step_id")): step
        for step in cpu_steps
        if isinstance(step, Mapping)
    }
    by_gpu_id = {
        str(step.get("step_id")): step
        for step in gpu_steps
        if isinstance(step, Mapping)
    }
    _require(
        set(by_cpu_id)
        == {
            "probe_authoritative_habitat_rir_runtime",
            "fresh_compile_mp3d_rlr_materials",
            "render_two_exact_rirs",
        },
        "execution plan CPU-step closure drift",
    )
    _require(
        set(by_gpu_id) == {"sparse_f15_probe", "full75_episode"},
        "execution plan GPU-step closure drift",
    )

    sparse = by_gpu_id["sparse_f15_probe"]
    sparse_argv = sparse.get("argv")
    _require(isinstance(sparse_argv, list), "sparse f15 argv is missing")

    def arg_value(argv: Sequence[Any], flag: str) -> str:
        _require(argv.count(flag) == 1, f"execution plan must contain one {flag}")
        index = argv.index(flag)
        _require(index + 1 < len(argv), f"execution plan lacks a value for {flag}")
        return str(argv[index + 1])

    def arg_path(argv: Sequence[Any], flag: str) -> Path:
        return Path(arg_value(argv, flag)).resolve()

    suite_plan = arg_path(sparse_argv, "--suite-plan")
    room_adapter = arg_path(sparse_argv, "--room-adapter")
    capture_output = arg_path(sparse_argv, "--output")
    _require_contained(suite_plan, preflight_root, owner="suite plan")
    _require_contained(room_adapter, preflight_root, owner="room adapter")
    _require_contained(capture_output, atom_root, owner="capture output")
    _require(
        arg_value(sparse_argv, "--frame-index") == str(FRAME_INDEX),
        "execution plan is not the exact sparse f15 probe",
    )
    _require(
        arg_value(sparse_argv, "--graphics-adapter") == "1",
        "execution plan sparse probe is not bound to graphics adapter 1",
    )

    probe_expected = by_cpu_id["probe_authoritative_habitat_rir_runtime"].get(
        "expected", {}
    )
    compile_expected = by_cpu_id["fresh_compile_mp3d_rlr_materials"].get("expected", {})
    rir_expected = by_cpu_id["render_two_exact_rirs"].get("expected", {})
    _require(
        isinstance(probe_expected, Mapping)
        and isinstance(compile_expected, Mapping)
        and isinstance(rir_expected, Mapping),
        "execution plan expected-output contract is missing",
    )
    paths = {
        "execution_plan": execution_plan_path,
        "preflight": preflight_root / "preflight.json",
        "room_adapter": room_adapter,
        "suite_plan": suite_plan,
        "rir_runtime_probe": Path(str(probe_expected.get("receipt", ""))).resolve(),
        "package_manifest": Path(str(compile_expected.get("manifest", ""))).resolve(),
        "package_material_coverage": Path(
            str(compile_expected.get("semantic_material_coverage", ""))
        ).resolve(),
        "rir_cache_receipt": Path(str(rir_expected.get("receipt", ""))).resolve(),
        "rir_cache_index": Path(str(rir_expected.get("index", ""))).resolve(),
        "capture_output": capture_output,
    }
    for name in ("preflight", "room_adapter", "suite_plan", "rir_runtime_probe"):
        _require_contained(paths[name], preflight_root, owner=name)
    for name in (
        "package_manifest",
        "package_material_coverage",
        "rir_cache_receipt",
        "rir_cache_index",
    ):
        _require_contained(paths[name], atom_root, owner=name)
    return paths


def _v2_source_paths() -> dict[str, Path]:
    return {
        "capture_script": REPOSITORY
        / "tools/qa/capture_spear_imported_glb_strict_two_human_episode.py",
        "launcher_script": Path(__file__).resolve(),
    }


def _regular_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def record_attempt01_failure_ledger(*, atom_root: Path, spear_log: Path) -> Path:
    """Freeze the consumed v1 attempt without assigning an unobserved cause."""

    atom_root = atom_root.resolve()
    spear_log = spear_log.resolve()
    expected_atom = REPOSITORY / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
    _require(atom_root == expected_atom, "MP3D f15 atom root drift")
    attempt_root = atom_root / "diagnostic_f15_launch_attempt_01"
    request_path = attempt_root / "request.json"
    dry_path = attempt_root / "dry_run_receipt.json"
    running_path = attempt_root / "running_receipt.json"
    final_path = attempt_root / "final_receipt.json"
    for owner, path in {
        "request": request_path,
        "dry receipt": dry_path,
        "running receipt": running_path,
        "final receipt": final_path,
        "SPEAR log": spear_log,
    }.items():
        _require(path.is_file(), f"attempt01 {owner} is missing: {path}")

    request = _load(request_path)
    dry = _load(dry_path)
    running = _load(running_path)
    final = _load(final_path)
    _require(request.get("schema") == REQUEST_SCHEMA, "attempt01 request drift")
    _require(
        dry.get("schema") == RECEIPT_SCHEMA
        and dry.get("status") == "dry_run_pass_not_launched",
        "attempt01 dry-run receipt drift",
    )
    _require(
        running.get("schema") == RECEIPT_SCHEMA and running.get("status") == "running",
        "attempt01 running receipt drift",
    )
    _require(
        final.get("schema") == RECEIPT_SCHEMA
        and final.get("status") == "failed"
        and final.get("capture_process_exit_code") == 1,
        "attempt01 final receipt is not the accepted exit-1 failure",
    )
    capture_output = Path(str(request.get("capture_output", ""))).resolve()
    _require(
        capture_output == atom_root / "diagnostic_f15_capture_attempt_01",
        "attempt01 capture output path drift",
    )
    _require(capture_output.is_dir(), "attempt01 capture output is missing")
    capture_files = _regular_files(capture_output)
    _require(not capture_files, "attempt01 unexpectedly has capture artifacts")
    serialized_receipts = json.dumps(
        {"running": running, "final": final}, sort_keys=True
    ).lower()
    _require(
        "stdout" not in serialized_receipts and "stderr" not in serialized_receipts,
        "attempt01 unexpectedly claims persisted stdout/stderr evidence",
    )
    spear_text = spear_log.read_text(encoding="utf-8", errors="replace")
    entry_markers = [
        "LogInit: Display: Game Engine Initialized.",
        "LogGlobalStatus: LoadMap Load map complete /Engine/Maps/Entry",
        "LogInit: Display: Engine is initialized. Leaving FEngineLoop::Init()",
    ]
    _require(
        all(marker in spear_text for marker in entry_markers),
        "SPEAR log does not close the accepted Entry initialization boundary",
    )
    ledger_path = attempt_root / "failure_ledger.json"
    _write_json_exclusive(
        ledger_path,
        {
            "schema": FAILURE_LEDGER_SCHEMA,
            "status": ATTEMPT01_FAILURE_STATUS,
            "root_cause": "undetermined",
            "episode_id": EPISODE_ID,
            "scene_id": SCENE_ID,
            "candidate_revision": "v1",
            "attempt_index": 1,
            "attempt_consumed": True,
            "retry_same_candidate_forbidden": True,
            "capture_process_exit_code": 1,
            "captured_frame_count": 0,
            "capture_artifact_count": 0,
            "first_capture_artifact_count": 0,
            "capture_output": str(capture_output),
            "capture_output_was_materialized": True,
            "last_closed_boundary": "entry_map_and_engine_initialized",
            "first_unclosed_boundary": "first_capture_artifact",
            "entry_initialization_markers_verified": entry_markers,
            "observability_gap": {
                "exclusive_child_stdout_persisted": False,
                "exclusive_child_stderr_persisted": False,
                "capture_phase_markers_persisted": False,
                "complete_python_traceback_persisted": False,
            },
            "causal_exclusions": {
                "mesh_failure_claimed": False,
                "lighting_failure_claimed": False,
                "camera_failure_claimed": False,
                "actor_failure_claimed": False,
                "capture_failure_claimed": False,
            },
            "evidence": {
                "request": _file_record(request_path),
                "dry_run_receipt": _file_record(dry_path),
                "running_receipt": _file_record(running_path),
                "final_receipt": _file_record(final_path),
                "spear_log": _file_record(spear_log),
            },
            "qualification_claim": False,
            "formal_dataset_count": 0,
            "recorded_at_utc": _utc_now(),
        },
    )
    return ledger_path


def _validate_cpu_evidence(paths: Mapping[str, Path]) -> dict[str, Any]:
    values = {name: _load(path) for name, path in paths.items()}
    preflight = values["preflight"]
    _require(
        preflight.get("schema")
        == "avengine_native_strict_two_human_mp3d_room_preflight_v1"
        and preflight.get("status") == "pass",
        "accepted MP3D preflight_v4 did not pass",
    )
    _require(
        preflight.get("episode_id") == EPISODE_ID
        and preflight.get("gpu_f15_request_materialized") is True
        and preflight.get("gpu_f15_request_ready") is False
        and preflight.get("gpu_started") is False,
        "preflight sparse-f15 boundary drift",
    )
    _require(
        preflight.get("qualification_claim") is False
        and preflight.get("formal_dataset_count") == 0,
        "preflight may not qualify or increment formal data",
    )
    navigation = preflight.get("navigation", {})
    pair_gate = navigation.get("adult_static_pair_gate", {})
    _require(
        navigation.get("status") == "pass"
        and navigation.get("shared_island_id") == 1
        and float(navigation.get("horizontal_source_separation_m", 0.0)) >= 1.3
        and pair_gate.get("clearance_gate_passed") is True
        and pair_gate.get("separation_gate_passed") is True,
        "adult two-human navigation gate drift",
    )
    selected = navigation.get("selected_positions", {})
    _require(
        float(selected.get("source1", {}).get("fresh_clearance_m", 0.0)) >= 0.5
        and float(selected.get("source2", {}).get("fresh_clearance_m", 0.0)) >= 0.5,
        "adult root clearance fell below 0.5m",
    )

    room = values["room_adapter"]
    meshes = room.get("static_mesh_object_paths")
    _require(
        room.get("schema") == "avengine_spear_imported_glb_room_adapter_v1"
        and room.get("scene_id") == SCENE_ID
        and room.get("expected_static_mesh_count") == EXPECTED_MESH_COUNT
        and isinstance(meshes, list)
        and len(meshes) == EXPECTED_MESH_COUNT
        and len(set(meshes)) == EXPECTED_MESH_COUNT,
        "room adapter must bind exactly 71 unique cooked mesh object paths",
    )
    _require(
        all(
            isinstance(path, str)
            and path.startswith("/Game/MyAssets/Audioset/Scenes/mp3d_17DRP5sb8fy/")
            and "." in path.rsplit("/", 1)[-1]
            for path in meshes
        ),
        "room adapter contains an invalid cooked UStaticMesh object path",
    )
    camera = room.get("camera_contract", {})
    _require(
        camera.get("one_camera_actor_for_all_passes") is True
        and camera.get("pass_order")
        == ["normal", "source1_target_only", "source2_target_only"],
        "shared normal/target-only camera contract drift",
    )
    _require(
        room.get("qualification_claim") is False
        and room.get("formal_dataset_count") == 0,
        "room adapter may not qualify or increment formal data",
    )

    suite = values["suite_plan"]
    scenarios = suite.get("scenarios")
    _require(
        suite.get("schema") == "avengine_optional_spear_imported_glb_suite_v1"
        and suite.get("native_map") == "/Engine/Maps/Entry"
        and suite.get("qualification_claim") is False
        and suite.get("formal_dataset_count") == 0
        and isinstance(scenarios, list)
        and len(scenarios) == 1,
        "suite must be exactly one non-formal imported-room scenario",
    )
    scenario = scenarios[0]
    plan = scenario.get("plan", {})
    frames = plan.get("frames")
    actors = plan.get("actors")
    _require(
        scenario.get("scenario_id") == EPISODE_ID
        and isinstance(frames, list)
        and len(frames) == 75
        and [item.get("frame_index") for item in frames] == list(range(75)),
        "suite is not the exact ordered 75-frame Episode",
    )
    _require(
        isinstance(actors, list)
        and [item.get("actor_id") for item in actors]
        == ["source1_actor", "source2_actor"]
        and "male_adult_01" in str(actors[0].get("template_id"))
        and "female_adult_01" in str(actors[1].get("template_id")),
        "suite is not the exact distinct M/F pair",
    )
    nested_room = plan.get("room", {}).get("room_adapter", {})
    _require(
        nested_room.get("static_mesh_object_paths") == meshes
        and nested_room.get("camera_contract") == camera,
        "suite and room adapter bindings differ",
    )

    runtime = values["rir_runtime_probe"]
    _require(
        runtime.get("schema") == "avengine_mp3d_rir_runtime_probe_v1"
        and runtime.get("status") == "pass"
        and runtime.get("compute_device") == "CPU"
        and runtime.get("gpu_required") is False
        and runtime.get("cuda_initialized") is False
        and runtime.get("import_order")
        == ["numpy", "quaternion", "habitat_sim", "avengine"]
        and runtime.get("qualification_claim") is False
        and runtime.get("formal_dataset_count") == 0,
        "authoritative CPU-only RIR runtime probe drift",
    )

    package = values["package_manifest"]
    geometry = package.get("geometry", {})
    _require(
        package.get("schema") == "avengine_acoustic_scene_package_v1"
        and package.get("package_id")
        == "habitat_mp3d_example_17DRP5sb8fy_soundspaces2_strict_two_human_v1"
        and package.get("package_mode") == "research_candidate"
        and package.get("room_kind") == "habitat_native"
        and geometry.get("triangle_count") == 3_016_249
        and geometry.get("vertex_count") == 1_570_132,
        "fresh SoundSpaces2 package binding drift",
    )
    coverage = values["package_material_coverage"]
    _require(
        coverage.get("schema") == "avengine_m3_rlr_semantic_material_coverage_v1"
        and coverage.get("status") == "research_candidate"
        and coverage.get("qualification_claim") is False
        and coverage.get("compiled_triangle_count") == 3_016_249
        and coverage.get("triangle_coverage", {}).get("triangle_count") == 3_016_249
        and coverage.get("runtime_one_to_one", {}).get("passed") is True,
        "research material coverage binding drift",
    )

    receipt = values["rir_cache_receipt"]
    index = values["rir_cache_index"]
    _require(
        receipt.get("schema") == "avengine_rlr_rir_cache_receipt_v1"
        and receipt.get("status") == "pass"
        and receipt.get("compute_device") == "CPU"
        and receipt.get("layout_type") == "binaural"
        and receipt.get("sample_rate_hz") == 16_000
        and receipt.get("selected_job_count") == 2
        and receipt.get("full_plan_job_count") == 2
        and receipt.get("full_plan_complete") is True
        and receipt.get("acoustic_selection_mode") == "registry"
        and receipt.get("retained_payload_hash_verified") is True
        and receipt.get("qualification_claim") is False,
        "exact two-source CPU RIR receipt drift",
    )
    _require(
        index.get("schema") == "avengine_rlr_rir_cache_index_v1"
        and index.get("status") == "pass"
        and index.get("selected_job_count") == 2
        and index.get("full_plan_complete") is True
        and index.get("request_identity_sha256")
        == receipt.get("request_identity_sha256")
        and index.get("acoustic_selection_binding_sha256")
        == receipt.get("acoustic_selection_binding_sha256")
        and len(index.get("entries", [])) == 2,
        "exact two-source CPU RIR index drift",
    )
    return values


def _validate_execution_plan_package(
    plan: Mapping[str, Any],
    package: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> None:
    compile_step = next(
        step
        for step in plan["cpu_steps"]
        if step.get("step_id") == "fresh_compile_mp3d_rlr_materials"
    )
    compile_argv = compile_step.get("argv", [])
    _require(
        isinstance(compile_argv, list) and compile_argv.count("--package-id") == 1,
        "execution plan package ID is missing",
    )
    package_id = str(compile_argv[compile_argv.index("--package-id") + 1])
    geometry = package.get("geometry", {})
    triangle_count = int(geometry.get("triangle_count", 0))
    vertex_count = int(geometry.get("vertex_count", 0))
    _require(
        package.get("schema") == "avengine_acoustic_scene_package_v1"
        and package.get("package_id") == package_id
        and package.get("package_mode") == "research_candidate"
        and package.get("room_kind") == "habitat_native"
        and triangle_count > 0
        and vertex_count > 0,
        "current fresh acoustic package drift",
    )
    _require(
        coverage.get("schema") == "avengine_m3_rlr_semantic_material_coverage_v1"
        and coverage.get("status") == "research_candidate"
        and coverage.get("qualification_claim") is False
        and coverage.get("compiled_triangle_count") == triangle_count
        and coverage.get("triangle_coverage", {}).get("triangle_count")
        == triangle_count
        and coverage.get("runtime_one_to_one", {}).get("passed") is True,
        "current research material coverage drift",
    )


def _validate_execution_plan_evidence(
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    required = {
        "execution_plan",
        "preflight",
        "room_adapter",
        "suite_plan",
        "rir_runtime_probe",
        "package_manifest",
        "package_material_coverage",
        "rir_cache_receipt",
        "rir_cache_index",
        "capture_output",
    }
    _require(set(paths) == required, "execution-plan artifact closure drift")
    for name in required - {"capture_output"}:
        _require(paths[name].is_file(), f"missing execution-plan evidence: {name}")

    values = {name: _load(paths[name]) for name in required - {"capture_output"}}
    plan = values["execution_plan"]
    if plan.get("schema") == "avengine_native_strict_two_human_mp3d_execution_plan_v1":
        legacy_paths = {
            name: paths[name]
            for name in (
                "preflight",
                "room_adapter",
                "suite_plan",
                "rir_runtime_probe",
                "package_manifest",
                "package_material_coverage",
                "rir_cache_receipt",
                "rir_cache_index",
            )
        }
        legacy_values = _validate_cpu_evidence(legacy_paths)
        sparse = next(
            step
            for step in plan["gpu_steps"]
            if step.get("step_id") == "sparse_f15_probe"
        )
        return {
            **values,
            **legacy_values,
            "episode_id": legacy_values["preflight"]["episode_id"],
            "scene_id": legacy_values["room_adapter"]["scene_id"],
            "capture_argv": list(sparse["argv"]),
        }
    preflight = values["preflight"]
    episode_id = str(preflight.get("episode_id", ""))
    _require(
        preflight.get("schema")
        == "avengine_native_strict_two_human_mp3d_room_preflight_v1"
        and preflight.get("status") == "pending_remaining_evidence"
        and preflight.get("cpu_planning_status") == "pass"
        and episode_id
        and preflight.get("gpu_started") is False
        and preflight.get("gpu_f15_request_materialized") is True
        and preflight.get("gpu_f15_request_ready") is False
        and preflight.get("episode_ready") is False
        and preflight.get("capture_ready") is False
        and preflight.get("formal_ready") is False
        and preflight.get("qualification_claim") is False
        and preflight.get("formal_dataset_count") == 0,
        "current MP3D preflight evidence boundary drift",
    )
    navigation = preflight.get("navigation", {})
    pair_gate = navigation.get("adult_static_pair_gate", {})
    _require(
        navigation.get("status") == "pass"
        and navigation.get("fresh_pathfinder_replay_status") == "pass"
        and float(navigation.get("horizontal_source_separation_m", 0.0)) >= 1.3
        and pair_gate.get("clearance_gate_passed") is True
        and pair_gate.get("separation_gate_passed") is True,
        "current MP3D navigation evidence drift",
    )
    framing = preflight.get("runtime_camera_framing", {})
    _require(
        framing.get("status") == "pass_cpu_declared_bounds_framing"
        and framing.get("selected_candidate_id")
        and framing.get("native_pixel_validation_status") == "pending",
        "current MP3D camera-framing boundary drift",
    )

    room = values["room_adapter"]
    scene_id = str(room.get("scene_id", ""))
    meshes = room.get("static_mesh_object_paths")
    _require(
        room.get("schema") == "avengine_spear_imported_glb_room_adapter_v1"
        and scene_id
        and room.get("expected_static_mesh_count") == EXPECTED_MESH_COUNT
        and isinstance(meshes, list)
        and len(meshes) == EXPECTED_MESH_COUNT
        and len(set(meshes)) == EXPECTED_MESH_COUNT
        and room.get("qualification_claim") is False
        and room.get("formal_dataset_count") == 0,
        "current MP3D room-adapter closure drift",
    )
    camera = room.get("camera_contract", {})
    _require(
        camera.get("one_camera_actor_for_all_passes") is True
        and camera.get("pass_order")
        == ["normal", "source1_target_only", "source2_target_only"],
        "current MP3D shared-camera contract drift",
    )

    suite = values["suite_plan"]
    scenarios = suite.get("scenarios")
    _require(
        suite.get("schema") == "avengine_optional_spear_imported_glb_suite_v1"
        and suite.get("native_map") == "/Engine/Maps/Entry"
        and suite.get("qualification_claim") is False
        and suite.get("formal_dataset_count") == 0
        and isinstance(scenarios, list)
        and len(scenarios) == 1,
        "current MP3D suite closure drift",
    )
    scenario = scenarios[0]
    episode = scenario.get("plan", {})
    frames = episode.get("frames")
    actors = episode.get("actors")
    _require(
        scenario.get("scenario_id") == episode_id
        and isinstance(frames, list)
        and len(frames) == 75
        and [frame.get("frame_index") for frame in frames] == list(range(75))
        and isinstance(actors, list)
        and [actor.get("actor_id") for actor in actors]
        == ["source1_actor", "source2_actor"],
        "current MP3D Episode identity/frame/actor closure drift",
    )
    nested_room = episode.get("room", {}).get("room_adapter", {})
    _require(
        nested_room.get("static_mesh_object_paths") == meshes
        and nested_room.get("camera_contract") == camera
        and episode.get("room", {}).get("scene_id") == scene_id
        and preflight.get("ue_import", {}).get("scene_id") == scene_id,
        "suite and room-adapter semantics differ",
    )

    runtime = values["rir_runtime_probe"]
    _require(
        runtime.get("schema") == "avengine_mp3d_rir_runtime_probe_v1"
        and runtime.get("status") == "pass"
        and runtime.get("compute_device") == "CPU"
        and runtime.get("gpu_required") is False
        and runtime.get("cuda_initialized") is False
        and runtime.get("qualification_claim") is False
        and runtime.get("formal_dataset_count") == 0,
        "current authoritative CPU runtime probe drift",
    )

    package = values["package_manifest"]
    coverage = values["package_material_coverage"]
    _validate_execution_plan_package(plan, package, coverage)

    receipt = values["rir_cache_receipt"]
    index = values["rir_cache_index"]
    _require(
        receipt.get("schema") == "avengine_rlr_rir_cache_receipt_v1"
        and receipt.get("status") == "pass"
        and receipt.get("compute_device") == "CPU"
        and receipt.get("layout_type") == "binaural"
        and receipt.get("sample_rate_hz") == 16_000
        and receipt.get("selected_job_count") == 2
        and receipt.get("full_plan_job_count") == 2
        and receipt.get("full_plan_complete") is True
        and receipt.get("qualification_claim") is False,
        "current exact RIR receipt drift",
    )
    entries = index.get("entries")
    _require(
        index.get("schema") == "avengine_rlr_rir_cache_index_v1"
        and index.get("status") == "pass"
        and index.get("selected_job_count") == 2
        and index.get("full_plan_complete") is True
        and isinstance(entries, list)
        and len(entries) == 2
        and [entry.get("job_index") for entry in entries] == [0, 1]
        and all(int(entry.get("sample_count", 0)) > 0 for entry in entries),
        "current exact RIR index drift",
    )

    sparse = next(
        step for step in plan["gpu_steps"] if step.get("step_id") == "sparse_f15_probe"
    )
    argv = list(sparse["argv"])
    _require(
        argv[argv.index("--scenario-id") + 1] == episode_id,
        "execution plan and Episode ID differ",
    )
    return {
        **values,
        "episode_id": episode_id,
        "scene_id": scene_id,
        "capture_argv": argv,
    }


def _capture_argv(request: Mapping[str, Any]) -> list[str]:
    return [
        str(request["capture_python"]),
        str(request["capture_script"]),
        "--suite-plan",
        str(request["suite_plan"]),
        "--scenario-id",
        str(request.get("episode_id", EPISODE_ID)),
        "--room-adapter",
        str(request["room_adapter"]),
        "--spear-root",
        str(request["spear_root"]),
        "--output",
        str(request["capture_output"]),
        "--rpc-port",
        str(request["rpc_port"]),
        "--graphics-adapter",
        "1",
        "--frame-index",
        str(FRAME_INDEX),
    ]


def prepare_request(
    *,
    atom_root: Path,
    capture_python: Path,
    spear_root: Path,
    rpc_port: int,
) -> Path:
    atom_root = atom_root.resolve()
    expected_atom = REPOSITORY / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
    _require(atom_root == expected_atom, "MP3D f15 atom root drift")
    attempt_root = atom_root / "diagnostic_f15_launch_attempt_01"
    capture_output = atom_root / "diagnostic_f15_capture_attempt_01"
    _require(not attempt_root.exists(), "attempt 01 already exists")
    _require(
        not capture_output.exists(), "diagnostic f15 capture output already exists"
    )
    paths = _artifact_paths(atom_root)
    _validate_cpu_evidence(paths)
    capture_script = (
        REPOSITORY / "tools/qa/capture_spear_imported_glb_strict_two_human_episode.py"
    )
    _require(capture_python.is_file(), "authoritative SPEAR Python is missing")
    _require(capture_script.is_file(), "MP3D capture runner is missing")
    _require(spear_root.is_dir(), "SPEAR root is missing")
    _require(1024 <= rpc_port <= 65535, "RPC port is out of range")
    artifact_records = {name: _file_record(path) for name, path in paths.items()}
    request = {
        "schema": REQUEST_SCHEMA,
        "status": "prepared_not_launched",
        "episode_id": EPISODE_ID,
        "scene_id": SCENE_ID,
        "required_repo_commit": _git_head(REPOSITORY),
        "repo_root": str(REPOSITORY),
        "atom_root": str(atom_root),
        "attempt_root": str(attempt_root),
        "capture_output": str(capture_output),
        "capture_python": str(capture_python.resolve()),
        "capture_script": str(capture_script.resolve()),
        "spear_root": str(spear_root.resolve()),
        "suite_plan": str(paths["suite_plan"].resolve()),
        "room_adapter": str(paths["room_adapter"].resolve()),
        "artifact_records": artifact_records,
        "attempt_policy": ATTEMPT_POLICY,
        "frame_indices": [FRAME_INDEX],
        "full75_allowed": False,
        "physical_gpu_index": 1,
        "physical_gpu_uuid": GPU1_UUID,
        "graphics_adapter_argument": 1,
        "required_idle_compute_process_count": 0,
        "rpc_port": rpc_port,
        "visibility_gate": {
            "target_instance_id": "source1",
            "distractor_instance_id": "source2",
            "target_minimum_visible_fraction": TARGET_MINIMUM_VISIBLE_FRACTION,
            "distractor_minimum_visible_fraction": DISTRACTOR_MINIMUM_VISIBLE_FRACTION,
            "target_only_background_depth_m": TARGET_ONLY_BACKGROUND_DEPTH_M,
            "absolute_tolerance_m": ABSOLUTE_TOLERANCE_M,
            "relative_tolerance": RELATIVE_TOLERANCE,
        },
        "manual_review_required": True,
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "created_at_utc": _utc_now(),
    }
    attempt_root.mkdir(parents=True)
    request_path = attempt_root / "request.json"
    _write_json_exclusive(request_path, request)
    return request_path


def prepare_request_v2(
    *,
    atom_root: Path,
    capture_python: Path,
    spear_root: Path,
    rpc_port: int,
) -> Path:
    """Prepare a fresh observability-only revision without launching a GPU job."""

    atom_root = atom_root.resolve()
    expected_atom = REPOSITORY / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
    _require(atom_root == expected_atom, "MP3D f15 atom root drift")
    attempt_root = atom_root / V2_ATTEMPT_DIRECTORY
    capture_output = atom_root / V2_CAPTURE_DIRECTORY
    _require(not attempt_root.exists(), "revision_v2 attempt 01 already exists")
    _require(not capture_output.exists(), "revision_v2 capture output already exists")

    evidence_paths = _artifact_paths(atom_root)
    _validate_cpu_evidence(evidence_paths)
    source_paths = _v2_source_paths()
    _require(
        all(path.is_file() for path in source_paths.values()),
        "revision_v2 observability source is missing",
    )
    failure_ledger_path = (
        atom_root / "diagnostic_f15_launch_attempt_01" / "failure_ledger.json"
    )
    failure_ledger = _load(failure_ledger_path)
    _require(
        failure_ledger.get("schema") == FAILURE_LEDGER_SCHEMA
        and failure_ledger.get("status") == ATTEMPT01_FAILURE_STATUS
        and failure_ledger.get("root_cause") == "undetermined"
        and failure_ledger.get("attempt_consumed") is True
        and failure_ledger.get("retry_same_candidate_forbidden") is True
        and failure_ledger.get("captured_frame_count") == 0
        and failure_ledger.get("capture_artifact_count") == 0,
        "attempt01 failure ledger is not the accepted observability gap",
    )
    _require(capture_python.is_file(), "authoritative SPEAR Python is missing")
    _require(
        _is_authoritative_capture_python(capture_python),
        "revision_v2 capture Python is not the authoritative SPEAR runtime",
    )
    _require(spear_root.resolve() == SPEAR_ROOT, "SPEAR root drift")
    _require(spear_root.is_dir(), "SPEAR root is missing")
    _require(rpc_port == V2_RPC_PORT, "revision_v2 RPC port drift")

    capture_stdout = attempt_root / "capture_stdout.log"
    capture_stderr = attempt_root / "capture_stderr.log"
    request = {
        "schema": REQUEST_SCHEMA_V2,
        "status": "prepared_not_launched",
        "episode_id": EPISODE_ID,
        "scene_id": SCENE_ID,
        "candidate_revision": "revision_v2_observability_only",
        "required_repo_commit": _git_head(REPOSITORY),
        "repo_root": str(REPOSITORY),
        "atom_root": str(atom_root),
        "attempt_root": str(attempt_root),
        "capture_output": str(capture_output),
        "capture_stdout": str(capture_stdout),
        "capture_stderr": str(capture_stderr),
        "capture_python": str(capture_python.resolve()),
        "capture_script": str(source_paths["capture_script"].resolve()),
        "spear_root": str(spear_root.resolve()),
        "suite_plan": str(evidence_paths["suite_plan"].resolve()),
        "room_adapter": str(evidence_paths["room_adapter"].resolve()),
        "artifact_records": {
            name: _file_record(path) for name, path in evidence_paths.items()
        },
        "observability_source_records": {
            name: _file_record(path) for name, path in source_paths.items()
        },
        "predecessor_failure_ledger": _file_record(failure_ledger_path),
        "attempt_policy": V2_ATTEMPT_POLICY,
        "frame_indices": [FRAME_INDEX],
        "full75_allowed": False,
        "physical_gpu_index": 1,
        "physical_gpu_uuid": GPU1_UUID,
        "graphics_adapter_argument": 1,
        "required_idle_compute_process_count": 0,
        "rpc_port": rpc_port,
        "visibility_gate": {
            "target_instance_id": "source1",
            "distractor_instance_id": "source2",
            "target_minimum_visible_fraction": TARGET_MINIMUM_VISIBLE_FRACTION,
            "distractor_minimum_visible_fraction": DISTRACTOR_MINIMUM_VISIBLE_FRACTION,
            "target_only_background_depth_m": TARGET_ONLY_BACKGROUND_DEPTH_M,
            "absolute_tolerance_m": ABSOLUTE_TOLERANCE_M,
            "relative_tolerance": RELATIVE_TOLERANCE,
        },
        "observability_contract": {
            "exclusive_child_stdout": str(capture_stdout),
            "exclusive_child_stderr": str(capture_stderr),
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
        "created_at_utc": _utc_now(),
    }
    attempt_root.mkdir(parents=True)
    request_path = attempt_root / "request.json"
    _write_json_exclusive(request_path, request)
    return request_path


def archive_preparation_failure(*, atom_root: Path, error: str) -> Path:
    """Preserve a failed pre-GPU request without consuming the real attempt."""

    atom_root = atom_root.resolve()
    _require(
        atom_root == REPOSITORY / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1",
        "MP3D f15 atom root drift",
    )
    attempt_root = atom_root / "diagnostic_f15_launch_attempt_01"
    archive_root = atom_root / "diagnostic_f15_prepare_failure_01"
    _require(attempt_root.is_dir(), "canonical prepared request is missing")
    _require(not archive_root.exists(), "preparation-failure archive already exists")
    _require((attempt_root / "request.json").is_file(), "prepared request is missing")
    _require(
        not any(
            (attempt_root / name).exists()
            for name in (
                "dry_run_receipt.json",
                "running_receipt.json",
                "final_receipt.json",
            )
        ),
        "cannot archive a request after dry or real launch evidence exists",
    )
    request = _load(attempt_root / "request.json")
    receipt_path = attempt_root / "preparation_failure_receipt.json"
    _write_json_exclusive(
        receipt_path,
        {
            "schema": RECEIPT_SCHEMA,
            "status": "prepare_validation_failed_before_gpu_query",
            "episode_id": EPISODE_ID,
            "required_repo_commit": request.get("required_repo_commit"),
            "request": str((attempt_root / "request.json").resolve()),
            "error": error,
            "gpu_query_started": False,
            "gpu_started": False,
            "attempt_consumed": False,
            "qualification_claim": False,
            "formal_dataset_count": 0,
            "captured_at_utc": _utc_now(),
        },
    )
    attempt_root.rename(archive_root)
    return archive_root / receipt_path.name


def _validate_request(request_path: Path) -> tuple[dict[str, Any], list[str]]:
    request = _load(request_path)
    _require(request.get("schema") == REQUEST_SCHEMA, "f15 request schema drift")
    _require(
        request.get("status") == "prepared_not_launched"
        and request.get("episode_id") == EPISODE_ID
        and request.get("scene_id") == SCENE_ID,
        "f15 request identity drift",
    )
    repo_root = Path(request["repo_root"]).resolve()
    _require(repo_root == REPOSITORY, "f15 request repository drift")
    _require(
        request.get("required_repo_commit") == _git_head(repo_root),
        "repository HEAD differs from the request-bound commit",
    )
    atom_root = Path(request["atom_root"]).resolve()
    _require(
        atom_root == repo_root / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1",
        "f15 request atom root drift",
    )
    attempt_root = atom_root / "diagnostic_f15_launch_attempt_01"
    _require(
        request_path.resolve() == attempt_root / "request.json"
        and Path(request["attempt_root"]).resolve() == attempt_root,
        "request is not bound to diagnostic attempt 01",
    )
    _require(
        Path(request["capture_output"]).resolve()
        == atom_root / "diagnostic_f15_capture_attempt_01",
        "diagnostic capture output path drift",
    )
    _require(
        request.get("attempt_policy") == ATTEMPT_POLICY
        and request.get("frame_indices") == [FRAME_INDEX]
        and request.get("full75_allowed") is False,
        "one-attempt sparse-f15 policy drift",
    )
    _require(
        request.get("physical_gpu_index") == 1
        and request.get("physical_gpu_uuid") == GPU1_UUID
        and request.get("graphics_adapter_argument") == 1
        and request.get("required_idle_compute_process_count") == 0,
        "physical GPU1/adapter1 binding drift",
    )
    _require(
        request.get("qualification_claim") is False
        and request.get("formal_dataset_count") == 0
        and request.get("manual_review_required") is True,
        "diagnostic request may not qualify or increment formal data",
    )
    gate = request.get("visibility_gate", {})
    _require(
        gate.get("target_instance_id") == "source1"
        and gate.get("distractor_instance_id") == "source2"
        and gate.get("target_minimum_visible_fraction")
        == TARGET_MINIMUM_VISIBLE_FRACTION
        and gate.get("distractor_minimum_visible_fraction")
        == DISTRACTOR_MINIMUM_VISIBLE_FRACTION,
        "f15 visibility thresholds drift",
    )
    expected_paths = _artifact_paths(atom_root)
    records = request.get("artifact_records")
    _require(
        isinstance(records, Mapping) and set(records) == set(expected_paths),
        "CPU artifact-record closure drift",
    )
    for name, expected in expected_paths.items():
        observed_path = _validate_file_record(records[name], owner=name)
        _require(observed_path == expected.resolve(), f"{name} path drift")
    _validate_cpu_evidence(expected_paths)
    _require(
        Path(request["suite_plan"]).resolve() == expected_paths["suite_plan"]
        and Path(request["room_adapter"]).resolve() == expected_paths["room_adapter"],
        "capture suite/room path drift",
    )
    _require(
        _is_authoritative_capture_python(Path(request["capture_python"]))
        and Path(request["capture_script"]).resolve()
        == repo_root / "tools/qa/capture_spear_imported_glb_strict_two_human_episode.py"
        and Path(request["spear_root"]) == SPEAR_ROOT,
        "authoritative MP3D SPEAR runtime binding drift",
    )
    for key in ("capture_python", "capture_script", "spear_root"):
        _require(Path(request[key]).exists(), f"missing capture input: {key}")
    _require(int(request["rpc_port"]) == 39631, "MP3D f15 RPC port drift")
    _require(not Path(request["capture_output"]).exists(), "capture output must be new")
    argv = _capture_argv(request)
    _require(
        argv.count("--frame-index") == 1
        and argv[argv.index("--frame-index") + 1] == "15",
        "capture must select only f15",
    )
    _require(
        argv.count("--graphics-adapter") == 1
        and argv[argv.index("--graphics-adapter") + 1] == "1",
        "capture must use graphics adapter 1",
    )
    return request, argv


def _validate_request_v2(
    request_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    request = _load(request_path)
    _require(request.get("schema") == REQUEST_SCHEMA_V2, "v2 request schema drift")
    _require(
        request.get("status") == "prepared_not_launched"
        and request.get("episode_id") == EPISODE_ID
        and request.get("scene_id") == SCENE_ID
        and request.get("candidate_revision") == "revision_v2_observability_only",
        "v2 request identity drift",
    )
    repo_root = Path(request["repo_root"]).resolve()
    _require(repo_root == REPOSITORY, "v2 request repository drift")
    _require(
        request.get("required_repo_commit") == _git_head(repo_root),
        "repository HEAD differs from the v2 request-bound commit",
    )
    atom_root = Path(request["atom_root"]).resolve()
    _require(
        atom_root == repo_root / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1",
        "v2 atom root drift",
    )
    attempt_root = atom_root / V2_ATTEMPT_DIRECTORY
    capture_output = atom_root / V2_CAPTURE_DIRECTORY
    stdout_path = attempt_root / "capture_stdout.log"
    stderr_path = attempt_root / "capture_stderr.log"
    _require(
        request_path.resolve() == attempt_root / "request.json"
        and Path(request["attempt_root"]).resolve() == attempt_root,
        "v2 request is not bound to its fresh attempt directory",
    )
    _require(
        Path(request["capture_output"]).resolve() == capture_output
        and Path(request["capture_stdout"]).resolve() == stdout_path
        and Path(request["capture_stderr"]).resolve() == stderr_path,
        "v2 output or exclusive log path drift",
    )
    _require(
        request.get("attempt_policy") == V2_ATTEMPT_POLICY
        and request.get("frame_indices") == [FRAME_INDEX]
        and request.get("full75_allowed") is False,
        "v2 one-attempt sparse-f15 policy drift",
    )
    _require(
        request.get("physical_gpu_index") == 1
        and request.get("physical_gpu_uuid") == GPU1_UUID
        and request.get("graphics_adapter_argument") == 1
        and request.get("required_idle_compute_process_count") == 0,
        "v2 physical GPU1/adapter1 binding drift",
    )
    _require(
        request.get("explicit_gpu_capture_authorization_required") is True
        and request.get("gpu_capture_authorized_at_prepare") is False
        and request.get("qualification_claim") is False
        and request.get("formal_dataset_count") == 0
        and request.get("manual_review_required") is True,
        "v2 request crossed its authorization or formal-data boundary",
    )
    gate = request.get("visibility_gate", {})
    _require(
        gate.get("target_instance_id") == "source1"
        and gate.get("distractor_instance_id") == "source2"
        and gate.get("target_minimum_visible_fraction")
        == TARGET_MINIMUM_VISIBLE_FRACTION
        and gate.get("distractor_minimum_visible_fraction")
        == DISTRACTOR_MINIMUM_VISIBLE_FRACTION,
        "v2 visibility thresholds drift",
    )
    observability = request.get("observability_contract", {})
    _require(
        observability.get("exclusive_child_stdout") == str(stdout_path)
        and observability.get("exclusive_child_stderr") == str(stderr_path)
        and observability.get("capture_phase_markers_required") is True
        and observability.get("complete_traceback_on_python_failure_required") is True
        and observability.get("child_exit_code_in_final_receipt_required") is True
        and observability.get("phases")
        == [
            "preconnect",
            "post-entry",
            "mesh",
            "lighting",
            "camera",
            "actor",
            "capture",
        ],
        "v2 observability contract drift",
    )

    expected_paths = _artifact_paths(atom_root)
    records = request.get("artifact_records")
    _require(
        isinstance(records, Mapping) and set(records) == set(expected_paths),
        "v2 CPU artifact-record closure drift",
    )
    for name, expected in expected_paths.items():
        observed_path = _validate_file_record(records[name], owner=name)
        _require(observed_path == expected.resolve(), f"{name} path drift")
    _validate_cpu_evidence(expected_paths)

    source_paths = _v2_source_paths()
    source_records = request.get("observability_source_records")
    _require(
        isinstance(source_records, Mapping)
        and set(source_records) == set(source_paths),
        "v2 observability source-record closure drift",
    )
    for name, expected in source_paths.items():
        observed_path = _validate_file_record(source_records[name], owner=name)
        _require(observed_path == expected.resolve(), f"{name} source path drift")

    ledger_path = atom_root / "diagnostic_f15_launch_attempt_01" / "failure_ledger.json"
    observed_ledger = _validate_file_record(
        request.get("predecessor_failure_ledger", {}),
        owner="predecessor failure ledger",
    )
    _require(observed_ledger == ledger_path, "predecessor ledger path drift")
    ledger = _load(ledger_path)
    _require(
        ledger.get("schema") == FAILURE_LEDGER_SCHEMA
        and ledger.get("status") == ATTEMPT01_FAILURE_STATUS
        and ledger.get("root_cause") == "undetermined"
        and ledger.get("attempt_consumed") is True
        and ledger.get("retry_same_candidate_forbidden") is True,
        "v2 predecessor ledger semantics drift",
    )
    _require(
        Path(request["suite_plan"]).resolve() == expected_paths["suite_plan"]
        and Path(request["room_adapter"]).resolve() == expected_paths["room_adapter"],
        "v2 capture suite/room path drift",
    )
    _require(
        _is_authoritative_capture_python(Path(request["capture_python"]))
        and Path(request["capture_script"]).resolve()
        == source_paths["capture_script"].resolve()
        and Path(request["spear_root"]).resolve() == SPEAR_ROOT,
        "v2 authoritative SPEAR runtime binding drift",
    )
    for key in ("capture_python", "capture_script", "spear_root"):
        _require(Path(request[key]).exists(), f"missing v2 capture input: {key}")
    _require(int(request["rpc_port"]) == V2_RPC_PORT, "v2 RPC port drift")
    _require(not capture_output.exists(), "v2 capture output must be new")
    _require(
        not stdout_path.exists() and not stderr_path.exists(),
        "v2 exclusive stdout/stderr path already exists",
    )
    argv = _capture_argv(request)
    _require(
        argv.count("--frame-index") == 1
        and argv[argv.index("--frame-index") + 1] == "15",
        "v2 capture must select only f15",
    )
    _require(
        argv.count("--graphics-adapter") == 1
        and argv[argv.index("--graphics-adapter") + 1] == "1",
        "v2 capture must use graphics adapter 1",
    )
    return request, argv


def _compile_and_validate_visibility(
    request: Mapping[str, Any],
    capture_root: Path,
    *,
    publish_truth: bool = True,
) -> dict[str, Any]:
    import numpy as np

    from avengine.qa.pixel_visibility import (
        PIXEL_VISIBILITY_DEPTH_AUTHORITY,
        compile_depth_pixel_visibility_truth,
    )

    suite = _load(Path(request["suite_plan"]))
    frame = suite["scenarios"][0]["plan"]["frames"][FRAME_INDEX]
    camera_pose_id = frame["camera_state"]["pose_hash"]
    depth_path = capture_root / "metric_depth_native.npz"
    _require(depth_path.is_file(), "capture metric-depth evidence is missing")
    with np.load(depth_path) as arrays:
        normal = arrays["normal_depth_m"]
        source1 = arrays["target_only_source1_depth_m"]
        source2 = arrays["target_only_source2_depth_m"]
    _require(
        normal.shape == source1.shape == source2.shape == (1, HEIGHT, WIDTH),
        "f15 normal/target-only metric-depth shape drift",
    )
    common = {
        "renderer_backend": "spear_unreal_native_mp3d_imported_glb",
        "rgb_renderer_backend": "spear_unreal_native_mp3d_imported_glb",
        "camera_contract_id": "lead_a_mp3d_shared_bp_camera_sensor_v1",
        "semantic_id_namespace": "lead_a_mp3d_metric_depth_instances_v1",
        "resolution_hw": [HEIGHT, WIDTH],
        "frame_indices": [FRAME_INDEX],
        "camera_pose_ids": [camera_pose_id],
    }
    truth = compile_depth_pixel_visibility_truth(
        normal_depth_m_frames=normal,
        target_only_depth_m_frames_by_instance={
            "source1": source1,
            "source2": source2,
        },
        semantic_ids_by_instance={"source1": 1, "source2": 2},
        normal_context={"pass_kind": "modal_scene", **common},
        target_only_contexts_by_instance={
            instance_id: {
                "pass_kind": "target_only",
                "target_instance_id": instance_id,
                **common,
            }
            for instance_id in ("source1", "source2")
        },
        target_only_background_depth_m=TARGET_ONLY_BACKGROUND_DEPTH_M,
        absolute_tolerance_m=ABSOLUTE_TOLERANCE_M,
        relative_tolerance=RELATIVE_TOLERANCE,
    )
    _require(
        truth.get("schema") == VISIBILITY_SCHEMA
        and truth.get("authority") == PIXEL_VISIBILITY_DEPTH_AUTHORITY
        and truth.get("frame_indices") == [FRAME_INDEX]
        and truth.get("resolution_hw") == [HEIGHT, WIDTH],
        "compiled native pixel truth contract drift",
    )
    target = truth["per_instance"]["source1"]["frames"][0]
    distractor = truth["per_instance"]["source2"]["frames"][0]
    target_fraction = float(target["visible_fraction"])
    distractor_fraction = float(distractor["visible_fraction"])
    _require(
        target_fraction >= TARGET_MINIMUM_VISIBLE_FRACTION,
        "f15 target visibility is below 0.8",
    )
    _require(
        distractor_fraction >= DISTRACTOR_MINIMUM_VISIBLE_FRACTION,
        "f15 distractor visibility is below 0.5",
    )
    _require(
        int(target["visible_pixels"]) > 0 and int(distractor["visible_pixels"]) > 0,
        "f15 target or distractor has no modal-visible pixels",
    )
    result = {
        "status": "pass",
        "target_instance_id": "source1",
        "distractor_instance_id": "source2",
        "target_visible_fraction": target_fraction,
        "distractor_visible_fraction": distractor_fraction,
        "target_visible_pixels": int(target["visible_pixels"]),
        "distractor_visible_pixels": int(distractor["visible_pixels"]),
    }
    if publish_truth:
        truth_path = capture_root / "pixel_visibility_truth.json"
        _write_json_exclusive(truth_path, truth)
        result["pixel_visibility_truth"] = _file_record(truth_path)
    return result


def _validate_capture(
    request: Mapping[str, Any], *, publish_visibility_truth: bool = True
) -> dict[str, Any]:
    capture_root = Path(request["capture_output"])
    manifest = _load(capture_root / "manifest.json")
    _require(
        manifest.get("schema")
        == "avengine_spear_imported_glb_strict_two_human_capture_v1"
        and manifest.get("status") == "capture_pass_review_pending"
        and manifest.get("scenario_id") == request.get("episode_id"),
        "MP3D f15 capture manifest drift",
    )
    frame = manifest.get("frame_contract", {})
    room = manifest.get("room", {})
    camera = manifest.get("camera_contract", {})
    _require(
        frame.get("captured_frame_count") == 1
        and frame.get("formal_episode_frame_count") == 75
        and frame.get("captured_frame_indices") == [FRAME_INDEX]
        and frame.get("resolution_hw") == [HEIGHT, WIDTH],
        "capture is not exactly sparse f15",
    )
    _require(
        room.get("scene_id") == request.get("scene_id")
        and room.get("fresh_cooked_mesh_readback_status") == "pass"
        and room.get("spawned_static_mesh_count") == EXPECTED_MESH_COUNT,
        "capture did not close the fresh 71-mesh room readback",
    )
    _require(
        camera.get("status") == "pass"
        and camera.get("same_actor_and_components_across_all_three_passes") is True,
        "capture did not preserve one shared camera across all passes",
    )
    review = manifest.get("live_review", {})
    _require(
        review.get("status") == "automated_bbox_pass_manual_mouth_review_pending"
        and review.get("automated_full_body_bbox_gate") is True
        and review.get("all_declared_mouth_proxies_inside_live_body_bbox") is True
        and review.get("manual_sparse_f15_visual_review_required") is True,
        "live body bbox or declared mouth-proxy gate failed",
    )
    _require(
        manifest.get("gpu_f15_review_ready") is True
        and manifest.get("gpu_full75_allowed") is False
        and manifest.get("qualification_claim") is False
        and manifest.get("formal_dataset_count") == 0,
        "capture crossed the diagnostic/formal boundary",
    )
    adapter = _load(Path(request["room_adapter"]))
    readback = _load(capture_root / "room_live_readback.json")
    meshes = readback.get("meshes")
    _require(
        readback.get("schema") == "avengine_spear_imported_glb_live_readback_v1"
        and readback.get("status") == "pass"
        and readback.get("expected_static_mesh_count") == EXPECTED_MESH_COUNT
        and readback.get("spawned_static_mesh_count") == EXPECTED_MESH_COUNT
        and readback.get("all_expected_handles_match_components") is True
        and readback.get("unique_loaded_object_handle_count") == EXPECTED_MESH_COUNT
        and readback.get("unique_component_mesh_handle_count") == EXPECTED_MESH_COUNT
        and isinstance(meshes, list)
        and [item.get("object_path") for item in meshes]
        == adapter["static_mesh_object_paths"],
        "live cooked UStaticMesh path/handle closure drift",
    )
    visibility = _compile_and_validate_visibility(
        request, capture_root, publish_truth=publish_visibility_truth
    )
    return {
        "status": "pass_diagnostic_f15_review_ready",
        "capture_manifest": _file_record(capture_root / "manifest.json"),
        "room_live_readback": _file_record(capture_root / "room_live_readback.json"),
        "visibility": visibility,
        "manual_live_mouth_bone_or_socket_review_required": True,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def run(request_path: Path, *, dry_run: bool) -> int:
    request, argv = _validate_request(request_path.resolve())
    attempt_root = Path(request["attempt_root"])
    dry_receipt = attempt_root / "dry_run_receipt.json"
    running_receipt = attempt_root / "running_receipt.json"
    final_receipt = attempt_root / "final_receipt.json"
    _require(not final_receipt.exists(), "attempt 01 already has a final receipt")
    if dry_run:
        _require(not dry_receipt.exists(), "dry-run receipt already exists")
        _require(not running_receipt.exists(), "real attempt already started")
    else:
        _require(not running_receipt.exists(), "real attempt already started")
    before = _gpu_snapshot()
    gpu = _validate_gpu1_idle(before)
    _assert_port_available(int(request["rpc_port"]))
    common = {
        "schema": RECEIPT_SCHEMA,
        "episode_id": EPISODE_ID,
        "scene_id": SCENE_ID,
        "attempt_policy": ATTEMPT_POLICY,
        "required_repo_commit": request["required_repo_commit"],
        "request": str(request_path.resolve()),
        "capture_argv": argv,
        "capture_output": request["capture_output"],
        "frame_indices": [FRAME_INDEX],
        "full75_allowed": False,
        "physical_gpu_index": 1,
        "physical_gpu_uuid": GPU1_UUID,
        "graphics_adapter_argument": 1,
        "prelaunch_gpu": gpu,
        "prelaunch_snapshot": before,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    if dry_run:
        _write_json_exclusive(
            dry_receipt,
            {
                **common,
                "status": "dry_run_pass_not_launched",
                "captured_at_utc": _utc_now(),
            },
        )
        return 0

    started_at = _utc_now()
    _write_json_exclusive(
        running_receipt,
        {
            **common,
            "status": "running",
            "started_at_utc": started_at,
            "capture_process_exit_code": None,
        },
    )
    exit_code = 1
    final: dict[str, Any] = {
        **common,
        "status": "failed",
        "started_at_utc": started_at,
        "ended_at_utc": None,
        "capture_process_exit_code": None,
    }
    try:
        completed = subprocess.run(argv, cwd=REPOSITORY, check=False)
        exit_code = int(completed.returncode)
        final["capture_process_exit_code"] = exit_code
        _require(exit_code == 0, f"f15 capture exited {exit_code}")
        final["validation"] = _validate_capture(request)
        final["status"] = "pass_diagnostic_f15_review_ready"
    except Exception as exc:  # noqa: BLE001
        final["error"] = f"{type(exc).__name__}: {exc}"
        exit_code = exit_code or 1
    finally:
        final["ended_at_utc"] = _utc_now()
        try:
            final["postlaunch_snapshot"] = _gpu_snapshot()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            final["postlaunch_snapshot_error"] = f"{type(exc).__name__}: {exc}"
        _write_json_exclusive(final_receipt, final)
    return exit_code


def _collect_v2_capture_observability(
    capture_root: Path,
) -> dict[str, Any]:
    phase_records: list[dict[str, Any]] = []
    if capture_root.is_dir():
        for path in sorted(capture_root.glob("capture_phase_*.json")):
            phase = _load(path)
            _require(
                phase.get("schema") == "avengine_mp3d_f15_capture_phase_v1"
                and phase.get("status") == "entered"
                and phase.get("qualification_claim") is False
                and phase.get("formal_dataset_count") == 0,
                f"invalid v2 capture phase marker: {path}",
            )
            phase_records.append(
                {
                    "phase": phase.get("phase"),
                    "sequence": phase.get("sequence"),
                    "artifact": _file_record(path),
                }
            )
    failure_path = capture_root / "capture_failure.json"
    failure_detail: dict[str, Any] | None = None
    failure_record: dict[str, Any] | None = None
    if failure_path.is_file():
        failure_detail = _load(failure_path)
        _require(
            failure_detail.get("schema") == CAPTURE_FAILURE_SCHEMA
            and failure_detail.get("status") == "failed"
            and isinstance(failure_detail.get("phase"), str)
            and isinstance(failure_detail.get("traceback"), str)
            and failure_detail["traceback"].strip()
            and failure_detail.get("qualification_claim") is False
            and failure_detail.get("formal_dataset_count") == 0,
            "v2 capture failure artifact is incomplete",
        )
        failure_record = _file_record(failure_path)
    return {
        "capture_phase_markers": phase_records,
        "capture_failure_artifact": failure_record,
        "capture_failure_detail": failure_detail,
    }


def _validate_complete_v2_phase_sequence(
    observability: Mapping[str, Any],
) -> None:
    markers = observability.get("capture_phase_markers", [])
    _require(isinstance(markers, list), "v2 phase marker collection drift")
    _require(
        [item.get("sequence") for item in markers] == list(range(9))
        and [item.get("phase") for item in markers]
        == [
            "preconnect",
            "post-entry",
            "mesh",
            "lighting",
            "camera",
            "actor",
            "capture",
            "artifact_finalize",
            "complete",
        ]
        and observability.get("capture_failure_detail") is None,
        "successful v2 child did not close every ordered capture phase",
    )


def run_v2(
    request_path: Path,
    *,
    dry_run: bool,
    authorize_gpu_capture: bool,
) -> int:
    request, argv = _validate_request_v2(request_path.resolve())
    attempt_root = Path(request["attempt_root"])
    stdout_path = Path(request["capture_stdout"])
    stderr_path = Path(request["capture_stderr"])
    dry_receipt = attempt_root / "dry_run_receipt.json"
    running_receipt = attempt_root / "running_receipt.json"
    final_receipt = attempt_root / "final_receipt.json"
    _require(not final_receipt.exists(), "revision_v2 already has a final receipt")
    if dry_run:
        _require(not dry_receipt.exists(), "revision_v2 dry-run receipt already exists")
        _require(
            not running_receipt.exists(),
            "revision_v2 real attempt already started",
        )
    else:
        _require(
            authorize_gpu_capture,
            "revision_v2 GPU capture lacks explicit launch authorization",
        )
        _require(
            not running_receipt.exists(),
            "revision_v2 real attempt already started",
        )
        _require(
            not stdout_path.exists() and not stderr_path.exists(),
            "revision_v2 exclusive stdout/stderr path already exists",
        )

    before = _gpu_snapshot()
    gpu = _validate_gpu1_idle(before)
    _assert_port_available(int(request["rpc_port"]))
    common = {
        "schema": RECEIPT_SCHEMA_V2,
        "episode_id": EPISODE_ID,
        "scene_id": SCENE_ID,
        "candidate_revision": "revision_v2_observability_only",
        "attempt_policy": V2_ATTEMPT_POLICY,
        "required_repo_commit": request["required_repo_commit"],
        "request": str(request_path.resolve()),
        "capture_argv": argv,
        "capture_output": request["capture_output"],
        "capture_stdout": str(stdout_path),
        "capture_stderr": str(stderr_path),
        "frame_indices": [FRAME_INDEX],
        "full75_allowed": False,
        "physical_gpu_index": 1,
        "physical_gpu_uuid": GPU1_UUID,
        "graphics_adapter_argument": 1,
        "prelaunch_gpu": gpu,
        "prelaunch_snapshot": before,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    if dry_run:
        _write_json_exclusive(
            dry_receipt,
            {
                **common,
                "status": "dry_run_pass_not_launched",
                "gpu_started": False,
                "attempt_consumed": False,
                "captured_at_utc": _utc_now(),
            },
        )
        return 0

    started_at = _utc_now()
    _write_json_exclusive(
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
        _require(child_exit_code == 0, f"revision_v2 f15 capture exited {exit_code}")
        completed_observability = _collect_v2_capture_observability(
            Path(request["capture_output"])
        )
        _validate_complete_v2_phase_sequence(completed_observability)
        final["capture_observability"] = completed_observability
        final["validation"] = _validate_capture(request)
        final["status"] = "pass_diagnostic_f15_review_ready"
    except Exception as exc:  # noqa: BLE001
        final["error"] = f"{type(exc).__name__}: {exc}"
        final["launcher_traceback"] = traceback.format_exc()
        exit_code = exit_code or 1
    finally:
        final["ended_at_utc"] = _utc_now()
        final["child_invocation_attempted"] = child_invocation_attempted
        final["child_exit_code"] = child_exit_code
        final["capture_process_exit_code"] = child_exit_code
        final["child_exit"] = {
            "observed": child_exit_code is not None,
            "returncode": child_exit_code,
        }
        final["gpu_started"] = child_exit_code is not None
        final["exclusive_child_stdout"] = (
            _file_record(stdout_path) if stdout_path.is_file() else None
        )
        final["exclusive_child_stderr"] = (
            _file_record(stderr_path) if stderr_path.is_file() else None
        )
        try:
            observability = _collect_v2_capture_observability(
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
            final["postlaunch_snapshot"] = _gpu_snapshot()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            final["postlaunch_snapshot_error"] = f"{type(exc).__name__}: {exc}"
        _write_json_exclusive(final_receipt, final)
    return exit_code


def offline_validate_execution_plan(execution_plan_path: Path) -> dict[str, Any]:
    """Validate the current evidence without writing or inspecting a GPU."""

    paths = _execution_plan_artifact_paths(execution_plan_path)
    evidence = _validate_execution_plan_evidence(paths)
    argv = evidence["capture_argv"]
    _require(
        Path(argv[0]).resolve() == CAPTURE_PYTHON_LOGICAL.resolve()
        and Path(argv[1]).resolve()
        == REPOSITORY
        / "tools/qa/capture_spear_imported_glb_strict_two_human_episode.py",
        "execution plan capture runtime drift",
    )
    _require(
        argv[argv.index("--suite-plan") + 1] == str(paths["suite_plan"])
        and argv[argv.index("--room-adapter") + 1] == str(paths["room_adapter"])
        and argv[argv.index("--output") + 1] == str(paths["capture_output"])
        and Path(argv[argv.index("--spear-root") + 1]).resolve() == SPEAR_ROOT
        and argv[argv.index("--graphics-adapter") + 1] == "1"
        and argv[argv.index("--frame-index") + 1] == str(FRAME_INDEX),
        "execution plan sparse capture argv is not semantically closed",
    )
    _require(CAPTURE_PYTHON_LOGICAL.is_file(), "authoritative SPEAR Python is missing")
    _require(Path(argv[1]).is_file(), "MP3D capture runner is missing")
    _require(SPEAR_ROOT.is_dir(), "SPEAR root is missing")
    _require(not paths["capture_output"].exists(), "sparse f15 output must be new")
    return {
        "schema": "avengine_mp3d_f15_execution_plan_offline_validation_v1",
        "status": "pass_offline_no_write_no_gpu_query",
        "episode_id": evidence["episode_id"],
        "scene_id": evidence["scene_id"],
        "execution_plan": str(paths["execution_plan"]),
        "evidence_paths": {
            name: str(path) for name, path in paths.items() if name != "capture_output"
        },
        "capture_output": str(paths["capture_output"]),
        "capture_argv": argv,
        "gpu_query_started": False,
        "gpu_started": False,
        "writes_performed": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def _packaged_room_readback_paths(atom_root: Path) -> dict[str, Path]:
    root = _require_contained(
        atom_root / PACKAGED_ROOM_READBACK_DIRECTORY,
        atom_root,
        owner="packaged room readback root",
    )
    return {
        "packaged_room_readback_result": root / "RESULT.json",
        "packaged_room_readback_exit": root / "EXIT.json",
    }


def _validate_packaged_room_readback(
    *,
    atom_root: Path,
    result_path: Path,
    exit_path: Path,
    room_adapter: Mapping[str, Any],
    scene_id: str,
) -> dict[str, Any]:
    """Close the fresh packaged NullRHI result by parsed semantics only."""

    expected_paths = _packaged_room_readback_paths(atom_root.resolve())
    _require(
        result_path.resolve() == expected_paths["packaged_room_readback_result"]
        and exit_path.resolve() == expected_paths["packaged_room_readback_exit"],
        "packaged room readback path drift",
    )
    result = _load(result_path)
    exit_receipt = _load(exit_path)
    expected_meshes = room_adapter.get("static_mesh_object_paths")
    _require(
        isinstance(expected_meshes, list)
        and len(expected_meshes) == EXPECTED_MESH_COUNT
        and len(set(expected_meshes)) == EXPECTED_MESH_COUNT,
        "room adapter lacks the exact 71-mesh readback authority",
    )
    _require(
        result.get("schema") == "avengine_packaged_imported_glb_room_readback_v1"
        and result.get("status") == "pass"
        and result.get("readiness_status")
        == "packaged_71_mesh_readback_pass_gpu_f15_pending"
        and result.get("scene_id") == scene_id
        and result.get("entry_map") == "/Engine/Maps/Entry"
        and result.get("nullrhi") is True
        and result.get("rendering_or_capture_called") is False
        and result.get("qualification_claim") is False
        and result.get("formal_dataset_count") == 0,
        "packaged room readback RESULT boundary drift",
    )
    live = result.get("room_live_readback")
    _require(isinstance(live, Mapping), "packaged room live readback is missing")
    meshes = live.get("meshes")
    _require(
        live.get("schema") == "avengine_spear_imported_glb_live_readback_v1"
        and live.get("status") == "pass"
        and live.get("scene_id") == scene_id
        and live.get("entry_map") == "/Engine/Maps/Entry"
        and live.get("expected_static_mesh_count") == EXPECTED_MESH_COUNT
        and live.get("spawned_static_mesh_count") == EXPECTED_MESH_COUNT
        and live.get("unique_loaded_object_handle_count") == EXPECTED_MESH_COUNT
        and live.get("unique_component_mesh_handle_count") == EXPECTED_MESH_COUNT
        and live.get("all_expected_handles_match_components") is True
        and live.get("qualification_claim") is False
        and live.get("formal_dataset_count") == 0
        and isinstance(meshes, list)
        and len(meshes) == EXPECTED_MESH_COUNT,
        "packaged room 71-mesh live readback drift",
    )
    _require(
        [mesh.get("mesh_index") for mesh in meshes] == list(range(EXPECTED_MESH_COUNT))
        and [mesh.get("object_path") for mesh in meshes] == expected_meshes
        and all(
            mesh.get("status") == "pass"
            and mesh.get("readback_method")
            in {
                "UStaticMeshComponent.GetStaticMesh",
                "UStaticMeshComponent.StaticMesh_property",
            }
            and mesh.get("stable_actor_name")
            == f"AVEngine/ImportedGLB/{scene_id}/mesh_{index:03d}"
            and isinstance(mesh.get("expected_object_handle"), int)
            and mesh.get("expected_object_handle") > 0
            and mesh.get("observed_component_mesh_handle")
            == mesh.get("expected_object_handle")
            for index, mesh in enumerate(meshes)
        ),
        "packaged room per-mesh readback drift",
    )
    expected_handles = [mesh.get("expected_object_handle") for mesh in meshes]
    observed_handles = [mesh.get("observed_component_mesh_handle") for mesh in meshes]
    stable_names = [mesh.get("stable_actor_name") for mesh in meshes]
    _require(
        len(set(expected_handles)) == EXPECTED_MESH_COUNT
        and len(set(observed_handles)) == EXPECTED_MESH_COUNT
        and len(set(stable_names)) == EXPECTED_MESH_COUNT,
        "packaged room per-mesh identities are not unique",
    )
    _require(
        exit_receipt.get("schema")
        == "avengine_packaged_imported_glb_room_probe_exit_v1"
        and exit_receipt.get("status") == "pass"
        and exit_receipt.get("worker_exit_code") == 0
        and exit_receipt.get("timed_out") is False
        and exit_receipt.get("exact_packaged_process_exit_closed") is True
        and exit_receipt.get("exact_packaged_processes_before") == []
        and exit_receipt.get("exact_packaged_processes_after") == []
        and exit_receipt.get("nullrhi") is True
        and exit_receipt.get("rendering_or_capture_called") is False
        and exit_receipt.get("result_exists") is True
        and exit_receipt.get("result_status") == "pass"
        and exit_receipt.get("semantic_error") is None
        and exit_receipt.get("qualification_claim") is False
        and exit_receipt.get("formal_dataset_count") == 0,
        "packaged room readback EXIT boundary drift",
    )
    return {"result": result, "exit": exit_receipt}


def offline_validate_execution_plan_v6(execution_plan_path: Path) -> dict[str, Any]:
    """Validate the selected v3 preflight plus fresh packaged room readback."""

    validation = offline_validate_execution_plan(execution_plan_path.resolve())
    plan_path = Path(validation["execution_plan"]).resolve()
    atom_root = plan_path.parent.parent.resolve()
    _require(
        plan_path == atom_root / "cpu_preflight_v3/execution_plan.json",
        "v6 must bind the selected cpu_preflight_v3 execution plan",
    )
    _require(
        Path(validation["capture_output"]).resolve()
        == atom_root / "native_sparse_f15_v1",
        "v6 sparse f15 output path drift",
    )
    readback_paths = _packaged_room_readback_paths(atom_root)
    room_adapter = _load(Path(validation["evidence_paths"]["room_adapter"]))
    _validate_packaged_room_readback(
        atom_root=atom_root,
        result_path=readback_paths["packaged_room_readback_result"],
        exit_path=readback_paths["packaged_room_readback_exit"],
        room_adapter=room_adapter,
        scene_id=validation["scene_id"],
    )
    return {
        **validation,
        "schema": "avengine_mp3d_f15_execution_plan_offline_validation_v2",
        "evidence_paths": {
            **validation["evidence_paths"],
            **{name: str(path) for name, path in readback_paths.items()},
        },
        "packaged_room_readback_status": "pass_nullrhi_71_of_71",
    }


def _v7_vector3(value: Any, *, owner: str) -> tuple[float, float, float]:
    _require(
        isinstance(value, list)
        and len(value) == 3
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value
        ),
        f"{owner} must be one finite numeric vector3",
    )
    return (float(value[0]), float(value[1]), float(value[2]))


def _validate_v7_raw_evidence_paths(execution_plan_path: Path) -> Path:
    raw_plan = _require_v7_nonsymlink_path(
        execution_plan_path, REPOSITORY / "tmp", owner="v7 execution plan"
    )
    plan = _load(raw_plan)
    cpu_steps = {
        str(step.get("step_id")): step
        for step in plan.get("cpu_steps", [])
        if isinstance(step, Mapping)
    }
    gpu_steps = {
        str(step.get("step_id")): step
        for step in plan.get("gpu_steps", [])
        if isinstance(step, Mapping)
    }
    _require(
        {
            "probe_authoritative_habitat_rir_runtime",
            "fresh_compile_mp3d_rlr_materials",
            "render_two_exact_rirs",
        }.issubset(cpu_steps)
        and "sparse_f15_probe" in gpu_steps,
        "v7 raw execution-plan steps are incomplete",
    )

    def arg_path(argv: Any, flag: str) -> Path:
        _require(
            isinstance(argv, list) and argv.count(flag) == 1,
            f"v7 raw sparse argv must contain one {flag}",
        )
        index = argv.index(flag)
        _require(index + 1 < len(argv), f"v7 raw sparse argv lacks {flag} value")
        return Path(str(argv[index + 1]))

    sparse_argv = gpu_steps["sparse_f15_probe"].get("argv")
    probe_expected = cpu_steps["probe_authoritative_habitat_rir_runtime"].get(
        "expected", {}
    )
    compile_expected = cpu_steps["fresh_compile_mp3d_rlr_materials"].get("expected", {})
    rir_expected = cpu_steps["render_two_exact_rirs"].get("expected", {})
    _require(
        isinstance(probe_expected, Mapping)
        and isinstance(compile_expected, Mapping)
        and isinstance(rir_expected, Mapping),
        "v7 raw expected evidence is missing",
    )
    declared = {
        "preflight": raw_plan.parent / "preflight.json",
        "suite plan": arg_path(sparse_argv, "--suite-plan"),
        "room adapter": arg_path(sparse_argv, "--room-adapter"),
        "RIR runtime probe": Path(str(probe_expected.get("receipt", ""))),
        "package manifest": Path(str(compile_expected.get("manifest", ""))),
        "package material coverage": Path(
            str(compile_expected.get("semantic_material_coverage", ""))
        ),
        "RIR receipt": Path(str(rir_expected.get("receipt", ""))),
        "RIR index": Path(str(rir_expected.get("index", ""))),
    }
    for owner, path in declared.items():
        _require_v7_nonsymlink_path(path, REPOSITORY / "tmp", owner=f"v7 {owner}")
    return raw_plan


def _validate_v7_execution_plan_evidence(
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Validate v7 semantics without admitting legacy file-evidence fields."""

    required = {
        "execution_plan",
        "preflight",
        "room_adapter",
        "suite_plan",
        "rir_runtime_probe",
        "package_manifest",
        "package_material_coverage",
        "rir_cache_receipt",
        "rir_cache_index",
        "capture_output",
    }
    _require(set(paths) == required, "v7 execution-plan artifact closure drift")
    for name in required - {"capture_output"}:
        _require(paths[name].is_file(), f"missing v7 execution-plan evidence: {name}")
    values = {name: _load(paths[name]) for name in required - {"capture_output"}}
    plan = values["execution_plan"]
    _require(
        plan.get("schema") == "avengine_native_strict_two_human_mp3d_execution_plan_v1"
        and plan.get("status") == "planned_not_run"
        and plan.get("qualification_claim") is False
        and plan.get("formal_dataset_count") == 0,
        "v7 execution-plan boundary drift",
    )
    preflight = values["preflight"]
    episode_id = preflight.get("episode_id")
    episode_contract = preflight.get("episode_contract", {})
    navigation = preflight.get("navigation", {})
    selected_positions = navigation.get("selected_positions", {})
    pair_gate = navigation.get("adult_static_pair_gate", {})
    _require(
        preflight.get("schema")
        == "avengine_native_strict_two_human_mp3d_room_preflight_v1"
        and preflight.get("status") == "pass"
        and isinstance(episode_id, str)
        and episode_id
        and preflight.get("gpu_started") is False
        and preflight.get("gpu_f15_request_materialized") is True
        and preflight.get("gpu_f15_request_ready") is False
        and preflight.get("qualification_claim") is False
        and preflight.get("formal_dataset_count") == 0
        and episode_contract.get("frame_count") == 75
        and episode_contract.get("frame_rate_hz") == 15
        and episode_contract.get("sparse_probe_frame_indices") == [FRAME_INDEX]
        and len(episode_contract.get("static_distinct_human_pair", [])) == 2,
        "v7 corrected preflight semantic boundary drift",
    )
    _require(
        navigation.get("status") == "pass"
        and navigation.get("fresh_pathfinder_replay_status") == "pass"
        and navigation.get("shared_island_id") == 1
        and float(navigation.get("horizontal_source_separation_m", 0.0)) >= 1.3
        and pair_gate.get("clearance_gate_passed") is True
        and pair_gate.get("separation_gate_passed") is True
        and set(selected_positions) == {"source1", "source2"}
        and all(
            selected_positions[slot].get("all_frames_navigable") is True
            and selected_positions[slot].get("island_id") == 1
            and float(selected_positions[slot].get("fresh_clearance_m", 0.0)) >= 0.5
            for slot in ("source1", "source2")
        ),
        "v7 navigation semantic closure drift",
    )

    room = values["room_adapter"]
    scene_id = room.get("scene_id")
    meshes = room.get("static_mesh_object_paths")
    camera = room.get("camera_contract", {})
    _require(
        room.get("schema") == "avengine_spear_imported_glb_room_adapter_v1"
        and isinstance(scene_id, str)
        and scene_id
        and room.get("expected_static_mesh_count") == EXPECTED_MESH_COUNT
        and isinstance(meshes, list)
        and len(meshes) == EXPECTED_MESH_COUNT
        and len(set(meshes)) == EXPECTED_MESH_COUNT
        and camera.get("one_camera_actor_for_all_passes") is True
        and camera.get("pass_order")
        == ["normal", "source1_target_only", "source2_target_only"]
        and room.get("qualification_claim") is False
        and room.get("formal_dataset_count") == 0,
        "v7 room-adapter semantic closure drift",
    )
    suite = values["suite_plan"]
    scenarios = suite.get("scenarios")
    _require(
        suite.get("schema") == "avengine_optional_spear_imported_glb_suite_v1"
        and suite.get("native_map") == "/Engine/Maps/Entry"
        and suite.get("qualification_claim") is False
        and suite.get("formal_dataset_count") == 0
        and isinstance(scenarios, list)
        and len(scenarios) == 1,
        "v7 suite semantic closure drift",
    )
    scenario = scenarios[0]
    episode = scenario.get("plan", {})
    frames = episode.get("frames")
    actors = episode.get("actors")
    _require(
        scenario.get("scenario_id") == episode_id
        and isinstance(frames, list)
        and [frame.get("frame_index") for frame in frames] == list(range(75))
        and isinstance(actors, list)
        and [actor.get("actor_id") for actor in actors]
        == ["source1_actor", "source2_actor"]
        and actors[0].get("template_id") != actors[1].get("template_id")
        and episode.get("room", {}).get("scene_id") == scene_id
        and episode.get("room", {})
        .get("room_adapter", {})
        .get("static_mesh_object_paths")
        == meshes
        and episode.get("room", {}).get("room_adapter", {}).get("camera_contract")
        == camera,
        "v7 Episode identity/frame/actor/room closure drift",
    )
    slot_actor_ids = {
        "source1": "source1_actor",
        "source2": "source2_actor",
    }
    actors_by_id = {actor.get("actor_id"): actor for actor in actors}
    _require(
        set(actors_by_id) == set(slot_actor_ids.values())
        and episode_contract.get("static_distinct_human_pair")
        == [
            actors_by_id[slot_actor_ids[slot]].get("asset_id")
            for slot in slot_actor_ids
        ],
        "v7 static human pair and actor assets differ",
    )
    roots = {
        slot: _v7_vector3(
            selected_positions[slot].get("habitat_root_m"),
            owner=f"v7 {slot} navigation root",
        )
        for slot in slot_actor_ids
    }
    emitter_offsets = {
        slot: _v7_vector3(
            actors_by_id[actor_id].get("emitter_offset_m"),
            owner=f"v7 {slot} emitter offset",
        )
        for slot, actor_id in slot_actor_ids.items()
    }
    expected_source_positions = {
        slot: tuple(
            roots[slot][axis] + emitter_offsets[slot][axis] for axis in range(3)
        )
        for slot in slot_actor_ids
    }
    episode_camera = episode.get("camera", {})
    expected_listener_position = _v7_vector3(
        episode_camera.get("habitat_position_m"), owner="v7 Episode camera position"
    )
    for frame in frames:
        states = frame.get("actor_states")
        _require(
            isinstance(states, list) and len(states) == 2,
            "v7 frame lacks the exact two actor states",
        )
        states_by_id = {state.get("actor_id"): state for state in states}
        _require(
            set(states_by_id) == set(slot_actor_ids.values()),
            "v7 frame actor identity drift",
        )
        for slot, actor_id in slot_actor_ids.items():
            state = states_by_id[actor_id]
            _require(
                _v7_vector3(
                    state.get("translation_m"),
                    owner=f"v7 {slot} frame translation",
                )
                == roots[slot]
                and state.get("asset_id") == actors_by_id[actor_id].get("asset_id"),
                "v7 navigation root and static actor frame differ",
            )
        _require(
            _v7_vector3(
                frame.get("camera_state", {}).get("habitat_position_m"),
                owner="v7 frame camera position",
            )
            == expected_listener_position,
            "v7 Episode camera is not static at the RIR listener",
        )

    runtime = values["rir_runtime_probe"]
    _require(
        runtime.get("schema") == "avengine_mp3d_rir_runtime_probe_v1"
        and runtime.get("status") == "pass"
        and runtime.get("compute_device") == "CPU"
        and runtime.get("gpu_required") is False
        and runtime.get("cuda_initialized") is False
        and runtime.get("qualification_claim") is False
        and runtime.get("formal_dataset_count") == 0,
        "v7 CPU runtime semantic closure drift",
    )
    _validate_execution_plan_package(
        plan, values["package_manifest"], values["package_material_coverage"]
    )

    preflight_rir = preflight.get("rir", {})
    source_positions = preflight_rir.get("source_positions_m")
    listener_position = preflight_rir.get("listener_position_m")
    receipt = values["rir_cache_receipt"]
    index = values["rir_cache_index"]
    entries = index.get("entries")
    _require(
        preflight_rir.get("status") == "planned_not_run"
        and preflight_rir.get("compute_device") == "CPU"
        and preflight_rir.get("unique_rir_job_count") == 2
        and isinstance(source_positions, Mapping)
        and set(source_positions) == {"source1", "source2"}
        and isinstance(listener_position, list)
        and all(
            _v7_vector3(source_positions[slot], owner=f"v7 {slot} RIR source")
            == expected_source_positions[slot]
            for slot in slot_actor_ids
        )
        and _v7_vector3(listener_position, owner="v7 RIR listener")
        == expected_listener_position
        and receipt.get("schema") == "avengine_rlr_rir_cache_receipt_v1"
        and receipt.get("status") == "pass"
        and receipt.get("compute_device") == "CPU"
        and receipt.get("layout_type") == "binaural"
        and receipt.get("layout_id") == "rlr_binaural_lr_v1"
        and receipt.get("channel_count") == 2
        and receipt.get("sample_rate_hz") == 16_000
        and receipt.get("selected_job_count") == 2
        and receipt.get("full_plan_job_count") == 2
        and receipt.get("full_plan_complete") is True
        and receipt.get("producer_backend") == "RLR Audio Propagation"
        and receipt.get("dry_audio_independent") is True
        and receipt.get("qualification_claim") is False
        and index.get("schema") == "avengine_rlr_rir_cache_index_v1"
        and index.get("status") == "pass"
        and index.get("selected_job_count") == 2
        and index.get("full_plan_complete") is True
        and isinstance(entries, list)
        and len(entries) == 2
        and [entry.get("job_index") for entry in entries] == [0, 1]
        and len({entry.get("job_id") for entry in entries}) == 2
        and all(int(entry.get("sample_count", 0)) > 0 for entry in entries)
        and [entry.get("source_position_m") for entry in entries]
        == [source_positions["source1"], source_positions["source2"]]
        and all(
            entry.get("listener_position_m") == listener_position for entry in entries
        )
        and all(
            entry.get("listener_orientation_wxyz") == [1.0, 0.0, 0.0, 0.0]
            for entry in entries
        ),
        "v7 binaural RIR semantic/pose closure drift",
    )
    sparse = next(
        step for step in plan["gpu_steps"] if step.get("step_id") == "sparse_f15_probe"
    )
    argv = list(sparse["argv"])
    _require(
        argv[argv.index("--scenario-id") + 1] == episode_id,
        "v7 execution plan and Episode ID differ",
    )
    return {
        **values,
        "episode_id": episode_id,
        "scene_id": scene_id,
        "capture_argv": argv,
    }


def _v8_execution_plan_paths(execution_plan_path: Path) -> dict[str, Path]:
    """Resolve the fresh schema-v2 CPU evidence before retargeting its capture."""

    raw_plan = _validate_v7_raw_evidence_paths(execution_plan_path)
    paths = _execution_plan_artifact_paths(raw_plan)
    plan = _load(raw_plan)
    preflight_root = raw_plan.parent
    atom_root = preflight_root.parent
    _require(
        plan.get("schema") == "avengine_native_strict_two_human_mp3d_execution_plan_v2"
        and raw_plan == atom_root / "cpu_preflight_v1/execution_plan.json",
        "v8 must bind the fresh schema-v2 cpu_preflight_v1 execution plan",
    )
    extras = {
        "actor_framing": preflight_root / "actor_framing.json",
        "camera_framing": preflight_root / "camera_framing.json",
        "runtime_camera_gates": preflight_root / "runtime_camera_gates.json",
        "rir_job_plan": preflight_root / "rir_job_plan.json",
    }
    for owner, path in extras.items():
        _require_v7_nonsymlink_path(path, atom_root, owner=f"v8 {owner}")
    return {**paths, **extras}


def _validate_v8_execution_plan_evidence(
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Close fresh schema-v2 planning, package, and semantic RIR evidence."""

    required = {
        "execution_plan",
        "preflight",
        "room_adapter",
        "suite_plan",
        "rir_runtime_probe",
        "package_manifest",
        "package_material_coverage",
        "rir_cache_receipt",
        "rir_cache_index",
        "capture_output",
        "actor_framing",
        "camera_framing",
        "runtime_camera_gates",
        "rir_job_plan",
    }
    _require(set(paths) == required, "v8 execution-plan artifact closure drift")
    for name in required - {"capture_output"}:
        _require(paths[name].is_file(), f"missing v8 execution-plan evidence: {name}")
    values = {name: _load(paths[name]) for name in required - {"capture_output"}}
    plan = values["execution_plan"]
    preflight = values["preflight"]
    contract = preflight.get("episode_contract", {})
    episode_id = preflight.get("episode_id")
    _require(
        plan.get("schema") == "avengine_native_strict_two_human_mp3d_execution_plan_v2"
        and plan.get("status") == "planned_not_run"
        and plan.get("qualification_claim") is False
        and plan.get("formal_dataset_count") == 0
        and preflight.get("schema")
        == "avengine_native_strict_two_human_mp3d_room_preflight_v1"
        and preflight.get("status") == "pending_remaining_evidence"
        and preflight.get("cpu_planning_status") == "pass"
        and isinstance(episode_id, str)
        and episode_id
        and preflight.get("gpu_started") is False
        and preflight.get("gpu_f15_request_materialized") is True
        and preflight.get("gpu_f15_request_ready") is False
        and preflight.get("episode_ready") is False
        and preflight.get("capture_ready") is False
        and preflight.get("formal_ready") is False
        and preflight.get("qualification_claim") is False
        and preflight.get("formal_dataset_count") == 0
        and contract.get("frame_count") == 75
        and contract.get("frame_rate_hz") == 15
        and contract.get("sparse_probe_frame_indices") == [FRAME_INDEX]
        and isinstance(contract.get("static_distinct_human_pair"), list)
        and len(contract["static_distinct_human_pair"]) == 2,
        "v8 schema-v2 planning boundary drift",
    )
    request_record = preflight.get("inputs", {}).get("request", {})
    raw_request_path = Path(str(request_record.get("path", "")))
    _require(
        raw_request_path.is_absolute(), "v8 preflight request path must be absolute"
    )
    request_path = _require_v7_nonsymlink_path(
        raw_request_path, REPOSITORY / "examples/qa", owner="v8 preflight request"
    )
    source_request = _load(request_path)
    request_camera = source_request.get("camera_framing", {})
    _require(
        source_request.get("schema")
        == "avengine_native_strict_two_human_mp3d_room_atom_request_v2"
        and source_request.get("request_id") == preflight.get("request_id")
        and source_request.get("episode_id") == episode_id
        and source_request.get("room", {}).get("scene_id")
        == preflight.get("ue_import", {}).get("scene_id")
        and source_request.get("qualification_claim") is False
        and source_request.get("formal_dataset_count") == 0
        and isinstance(request_camera, Mapping),
        "v8 preflight differs from its authoritative request",
    )

    room = values["room_adapter"]
    scene_id = room.get("scene_id")
    meshes = room.get("static_mesh_object_paths")
    camera_contract = room.get("camera_contract", {})
    _require(
        room.get("schema") == "avengine_spear_imported_glb_room_adapter_v1"
        and isinstance(scene_id, str)
        and scene_id
        and room.get("expected_static_mesh_count") == EXPECTED_MESH_COUNT
        and isinstance(meshes, list)
        and len(meshes) == EXPECTED_MESH_COUNT
        and len(set(meshes)) == EXPECTED_MESH_COUNT
        and camera_contract.get("one_camera_actor_for_all_passes") is True
        and camera_contract.get("pass_order")
        == ["normal", "source1_target_only", "source2_target_only"]
        and room.get("qualification_claim") is False
        and room.get("formal_dataset_count") == 0,
        "v8 room and shared-camera contract drift",
    )
    suite = values["suite_plan"]
    scenarios = suite.get("scenarios")
    _require(
        suite.get("schema") == "avengine_optional_spear_imported_glb_suite_v1"
        and suite.get("native_map") == "/Engine/Maps/Entry"
        and suite.get("qualification_claim") is False
        and suite.get("formal_dataset_count") == 0
        and isinstance(scenarios, list)
        and len(scenarios) == 1,
        "v8 suite boundary drift",
    )
    scenario = scenarios[0]
    episode = scenario.get("plan", {})
    frames = episode.get("frames")
    actors = episode.get("actors")
    actor_ids = ["source1_actor", "source2_actor"]
    _require(
        scenario.get("scenario_id") == episode_id
        and isinstance(frames, list)
        and [frame.get("frame_index") for frame in frames] == list(range(75))
        and isinstance(actors, list)
        and [actor.get("actor_id") for actor in actors] == actor_ids
        and actors[0].get("asset_id") != actors[1].get("asset_id")
        and episode.get("room", {}).get("scene_id") == scene_id
        and episode.get("room", {})
        .get("room_adapter", {})
        .get("static_mesh_object_paths")
        == meshes
        and episode.get("room", {}).get("room_adapter", {}).get("camera_contract")
        == camera_contract,
        "v8 Episode identity, actor, or room closure drift",
    )
    actors_by_id = {actor["actor_id"]: actor for actor in actors}
    slot_actor_ids = {"source1": "source1_actor", "source2": "source2_actor"}
    _require(
        contract["static_distinct_human_pair"]
        == [
            actors_by_id[slot_actor_ids[slot]].get("asset_id")
            for slot in slot_actor_ids
        ],
        "v8 declared human pair differs from suite actors",
    )

    navigation = preflight.get("navigation", {})
    selected_positions = navigation.get("selected_positions", {})
    pair_gate = navigation.get("adult_static_pair_gate", {})
    _require(
        navigation.get("status") == "pass"
        and navigation.get("fresh_pathfinder_replay_status") == "pass"
        and navigation.get("shared_island_id") == 1
        and float(navigation.get("horizontal_source_separation_m", 0.0)) >= 1.3
        and pair_gate.get("clearance_gate_passed") is True
        and pair_gate.get("separation_gate_passed") is True
        and set(selected_positions) == set(slot_actor_ids)
        and all(
            selected_positions[slot].get("all_frames_navigable") is True
            and selected_positions[slot].get("island_id") == 1
            and float(selected_positions[slot].get("fresh_clearance_m", 0.0)) >= 0.5
            for slot in slot_actor_ids
        ),
        "v8 fresh navigation closure drift",
    )
    roots = {
        slot: _v7_vector3(
            selected_positions[slot].get("habitat_root_m"),
            owner=f"v8 {slot} navigation root",
        )
        for slot in slot_actor_ids
    }
    episode_camera = episode.get("camera", {})
    listener = _v7_vector3(
        episode_camera.get("habitat_position_m"), owner="v8 Episode camera"
    )
    selected_id = preflight.get("runtime_camera_framing", {}).get(
        "selected_candidate_id"
    )
    generation = request_camera.get("candidate_generation", {})
    offsets = generation.get("offsets_xz_m")
    selected_index = (
        int(selected_id.rsplit("_", 1)[1])
        if isinstance(selected_id, str) and selected_id.startswith("midpoint_grid_")
        else -1
    )
    _require(
        isinstance(offsets, list)
        and 0 <= selected_index < len(offsets)
        and selected_id == f"midpoint_grid_{selected_index:03d}",
        "v8 selected camera candidate identity drift",
    )
    offset = _v7_vector3(
        offsets[selected_index], owner="v8 selected camera candidate offset"
    )
    midpoint = tuple(
        (roots["source1"][axis] + roots["source2"][axis]) / 2.0 for axis in range(3)
    )
    expected_candidate_position = (
        midpoint[0] + offset[0],
        float(request_camera.get("floor_height_m"))
        + float(generation.get("eye_height_m")),
        midpoint[2] + offset[2],
    )
    delta_x = midpoint[0] - expected_candidate_position[0]
    delta_z = midpoint[2] - expected_candidate_position[2]
    expected_candidate_yaw = math.degrees(math.atan2(-delta_x, -delta_z))
    _require(
        all(
            math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for observed, expected in zip(listener, expected_candidate_position)
        )
        and math.isclose(
            float(episode_camera.get("habitat_yaw_deg")),
            expected_candidate_yaw,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "v8 selected camera differs from request candidate generation",
    )

    emitters = {
        slot: _v7_vector3(
            actors_by_id[actor_id].get("emitter_offset_m"),
            owner=f"v8 {slot} emitter offset",
        )
        for slot, actor_id in slot_actor_ids.items()
    }
    expected_sources = {
        slot: tuple(roots[slot][axis] + emitters[slot][axis] for axis in range(3))
        for slot in slot_actor_ids
    }
    for frame in frames:
        states = frame.get("actor_states")
        states_by_id = (
            {
                state.get("actor_id"): state
                for state in states
                if isinstance(state, Mapping)
            }
            if isinstance(states, list)
            else {}
        )
        _require(set(states_by_id) == set(actor_ids), "v8 frame actor closure drift")
        for slot, actor_id in slot_actor_ids.items():
            state = states_by_id[actor_id]
            _require(
                _v7_vector3(state.get("translation_m"), owner=f"v8 {slot} frame root")
                == roots[slot]
                and state.get("asset_id") == actors_by_id[actor_id].get("asset_id"),
                "v8 static actor frame differs from navigation authority",
            )
        camera_state = frame.get("camera_state", {})
        world_from_rig = camera_state.get("world_from_rig", {})
        rotation_xyzw = world_from_rig.get("rotation_xyzw")
        _require(
            _v7_vector3(camera_state.get("habitat_position_m"), owner="v8 frame camera")
            == listener
            and _v7_vector3(
                world_from_rig.get("translation_m"), owner="v8 frame rig camera"
            )
            == listener
            and isinstance(rotation_xyzw, list)
            and len(rotation_xyzw) == 4
            and all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                for item in rotation_xyzw
            )
            and camera_state.get("habitat_yaw_deg")
            == episode_camera.get("habitat_yaw_deg")
            and camera_state.get("ue_position_cm")
            == episode_camera.get("ue_position_cm")
            and camera_state.get("ue_yaw_deg") == episode_camera.get("ue_yaw_deg"),
            "v8 frame camera pose differs from the selected listener",
        )

    selected_rotation_xyzw = frames[0]["camera_state"]["world_from_rig"][
        "rotation_xyzw"
    ]
    yaw_radians = math.radians(float(episode_camera.get("habitat_yaw_deg")))
    expected_rotation_xyzw = [
        0.0,
        math.sin(yaw_radians / 2.0),
        0.0,
        math.cos(yaw_radians / 2.0),
    ]
    _require(
        all(
            frame["camera_state"]["world_from_rig"]["rotation_xyzw"]
            == selected_rotation_xyzw
            for frame in frames
        )
        and math.isclose(
            sum(float(value) ** 2 for value in selected_rotation_xyzw),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and all(
            math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1.0e-12)
            for observed, expected in zip(
                selected_rotation_xyzw, expected_rotation_xyzw
            )
        ),
        "v8 listener orientation is not the declared Habitat yaw rotation",
    )

    actor_framing = values["actor_framing"]
    framing_frames = actor_framing.get("frames")
    _require(
        actor_framing.get("schema") == "avengine_actor_framing_inputs_v1"
        and actor_framing.get("status") == "pass_cpu_sampled_planning_envelopes"
        and actor_framing.get("frame_count") == 75
        and actor_framing.get("actor_ids") == actor_ids
        and isinstance(framing_frames, list)
        and [item.get("frame_index") for item in framing_frames] == list(range(75))
        and all(
            set(item.get("actor_aabbs", {})) == set(actor_ids)
            and all(
                item["actor_aabbs"][actor_id]
                .get("bounds_authority", {})
                .get("asset_id")
                == actors_by_id[actor_id].get("asset_id")
                for actor_id in actor_ids
            )
            for item in framing_frames
        )
        and actor_framing.get("qualification", {}).get("qualification_claim") is False
        and actor_framing.get("qualification", {}).get("formal_episode_count") == 0,
        "v8 actor-framing and suite actor closure drift",
    )
    camera_framing = values["camera_framing"]
    selected_pose = camera_framing.get("selected_camera_pose", {})
    evaluations = camera_framing.get("candidate_evaluations")
    selected_evaluations = (
        [
            value
            for value in evaluations
            if isinstance(value, Mapping) and value.get("candidate_id") == selected_id
        ]
        if isinstance(evaluations, list)
        else []
    )
    _require(
        preflight.get("runtime_camera_framing", {}).get("status")
        == "pass_cpu_declared_bounds_framing"
        and camera_framing.get("schema") == "avengine_camera_framing_evidence_v1"
        and camera_framing.get("status") == "pass_cpu_declared_bounds_framing"
        and camera_framing.get("frame_count") == 75
        and camera_framing.get("frame_indices") == list(range(75))
        and camera_framing.get("actor_ids") == actor_ids
        and camera_framing.get("ordered_actor_ids") == actor_ids
        and camera_framing.get("selected_candidate_id") == selected_id
        and camera_framing.get("ordered_actor_ids")
        == request_camera.get("ordered_actor_ids")
        and camera_framing.get("minimum_order_gap_px")
        == request_camera.get("minimum_order_gap_px")
        and _v7_vector3(selected_pose.get("position_m"), owner="v8 framing camera")
        == listener
        and math.isclose(
            float(selected_pose.get("yaw_deg")),
            float(episode_camera.get("habitat_yaw_deg")),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        and episode_camera.get("ue_position_cm")
        == [100.0 * listener[0], 100.0 * listener[2], 100.0 * listener[1]]
        and math.isclose(
            float(episode_camera.get("ue_yaw_deg")),
            -90.0 - float(episode_camera.get("habitat_yaw_deg")),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        and episode_camera.get("horizontal_fov_deg")
        == request_camera.get("calibration", {}).get("hfov_degrees")
        and len(selected_evaluations) == 1
        and selected_evaluations[0].get("selectable") is True
        and selected_evaluations[0].get("all_frames_hard_gates_pass") is True
        and [
            item.get("frame_index")
            for item in selected_evaluations[0].get("frame_evaluations", [])
        ]
        == list(range(75))
        and camera_framing.get("native_pixel_validation_status") == "pending"
        and camera_framing.get("qualification_claim") is False
        and camera_framing.get("formal_episode_count") == 0,
        "v8 selected camera/framing closure drift",
    )
    runtime_gates = values["runtime_camera_gates"]
    runtime_results = runtime_gates.get("results")
    selected_runtime = (
        [
            value
            for value in runtime_results
            if isinstance(value, Mapping) and value.get("candidate_id") == selected_id
        ]
        if isinstance(runtime_results, list)
        else []
    )
    _require(
        runtime_gates.get("schema") == "avengine_mp3d_runtime_camera_gate_batch_v1"
        and runtime_gates.get("qualification_claim") is False
        and runtime_gates.get("formal_dataset_count") == 0
        and len(selected_runtime) == 1,
        "v8 runtime camera-gate selection drift",
    )
    runtime_selected = selected_runtime[0]
    preflight_hard_gates = runtime_selected.get("evidence", {}).get("hard_gates", {})
    _require(
        all(
            preflight_hard_gates.get(f"line_of_sight_{slot}_actor", {}).get(
                "anchor_ids"
            )
            == ["declared_emitter_proxy", "torso_envelope_center"]
            and preflight_hard_gates.get(f"line_of_sight_{slot}_actor", {}).get(
                "tolerance_m"
            )
            == request_camera.get("line_of_sight_tolerance_m")
            for slot in slot_actor_ids
        ),
        "v8 listener runtime/nav/LOS closure drift",
    )
    from avengine.camera_framing import evaluate_static_camera_candidate

    recomputed_evaluation = evaluate_static_camera_candidate(
        frames=framing_frames,
        candidate={
            "candidate_id": runtime_selected.get("candidate_id"),
            "priority": runtime_selected.get("priority"),
            "position_m": runtime_selected.get("position_m"),
            "yaw_deg": runtime_selected.get("yaw_deg"),
            "room_gate": runtime_selected.get("room_gate"),
        },
        calibration=request_camera.get("calibration"),
        ordered_actor_ids=request_camera.get("ordered_actor_ids"),
        minimum_order_gap_px=request_camera.get("minimum_order_gap_px"),
    )
    _require(
        recomputed_evaluation == selected_evaluations[0],
        "v8 selected framing evidence differs from ordinary CPU projection",
    )
    hard_gates = runtime_selected.get("evidence", {}).get("hard_gates", {})
    _require(
        runtime_selected.get("status") == "pass"
        and _v7_vector3(runtime_selected.get("position_m"), owner="v8 runtime camera")
        == listener
        and math.isclose(
            float(runtime_selected.get("yaw_deg")),
            float(episode_camera.get("habitat_yaw_deg")),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        and runtime_selected.get("frame_count") == 75
        and runtime_selected.get("priority") == float(selected_index)
        and runtime_selected.get("room_gate", {}).get("status") == "pass"
        and runtime_selected.get("room_gate", {}).get(
            "native_habitat_validation_status"
        )
        == "pass"
        and runtime_selected.get("room_gate", {}).get("line_of_sight_validation_status")
        == "pass"
        and all(
            hard_gates.get(f"line_of_sight_{slot}_actor", {}).get("status") == "pass"
            and hard_gates.get(f"line_of_sight_{slot}_actor", {}).get("anchor_ids")
            == ["declared_emitter_proxy", "torso_envelope_center"]
            and hard_gates.get(f"line_of_sight_{slot}_actor", {}).get("tolerance_m")
            == request_camera.get("line_of_sight_tolerance_m")
            and hard_gates.get(f"line_of_sight_{slot}_actor", {}).get("query_count")
            == 150
            and hard_gates.get(f"line_of_sight_{slot}_actor", {}).get(
                "passed_query_count"
            )
            == 150
            for slot in slot_actor_ids
        )
        and hard_gates.get("same_navmesh_island", {}).get("status") == "pass"
        and hard_gates.get("listener_navmesh", {}).get("status") == "pass"
        and hard_gates.get("listener_navmesh", {}).get("island") == 1
        and float(hard_gates.get("listener_navmesh", {}).get("clearance_m", 0.0))
        >= 0.25,
        "v8 selected listener runtime/nav/LOS closure drift",
    )

    runtime_hard_gates = runtime_selected.get("room_gate", {}).get("hard_gates")
    runtime_context = runtime_selected.get("evidence", {}).get("runtime_context", {})
    room_bounds = hard_gates.get("room_bounds", {})
    _require(
        runtime_hard_gates == hard_gates
        and runtime_selected.get("evidence", {}).get("provenance")
        == "habitat_cpu_runtime"
        and runtime_selected.get("room_gate", {}).get("provenance")
        == "habitat_cpu_runtime"
        and runtime_context.get("scene_id") == scene_id
        and runtime_context.get("pathfinder_loaded") is True
        and runtime_context.get("raycast_enabled") is True
        and runtime_context.get("physics_enabled") is True
        and room_bounds.get("status") == "pass"
        and _v7_vector3(room_bounds.get("position_m"), owner="v8 room-bounds listener")
        == listener,
        "v8 selected listener runtime authority drift",
    )

    runtime = values["rir_runtime_probe"]
    _require(
        runtime.get("schema") == "avengine_mp3d_rir_runtime_probe_v1"
        and runtime.get("status") == "pass"
        and runtime.get("compute_device") == "CPU"
        and runtime.get("gpu_required") is False
        and runtime.get("cuda_initialized") is False
        and runtime.get("qualification_claim") is False
        and runtime.get("formal_dataset_count") == 0,
        "v8 CPU runtime probe drift",
    )
    _validate_execution_plan_package(
        plan, values["package_manifest"], values["package_material_coverage"]
    )
    source_room = values["package_manifest"].get("source_room", {})
    _require(
        source_room.get("room_id") == f"habitat_mp3d_example_{scene_id}"
        and preflight.get("ue_import", {}).get("scene_id") == scene_id,
        "v8 semantic package and selected room identity differ",
    )

    from avengine.acoustics.rir_cache import (
        load_semantic_acoustic_scene,
        validate_semantic_rir_job_plan,
    )

    rir_step = next(
        step
        for step in plan["cpu_steps"]
        if step.get("step_id") == "render_two_exact_rirs"
    )
    rir_argv = rir_step.get("argv")
    required_flags = {
        "--rir-job-plan": paths["rir_job_plan"],
        "--acoustic-package-manifest": paths["package_manifest"],
        "--simulation-request": Path(
            str(paths["rir_cache_receipt"].parent.joinpath("request.json"))
        ),
        "--hrtf": Path("/usr/share/libmysofa/MIT_KEMAR_normal_pinna.sofa"),
        "--output": paths["rir_cache_receipt"].parent,
    }
    _require(
        isinstance(rir_argv, list)
        and rir_argv.count("--semantic-no-file-evidence") == 1
        and all(rir_argv.count(flag) == 1 for flag in required_flags)
        and rir_argv.count("--layout") == 1
        and rir_argv[rir_argv.index("--layout") + 1] == "binaural"
        and rir_argv.count("--batch-size") == 1
        and rir_argv[rir_argv.index("--batch-size") + 1] == "2"
        and rir_argv.count("--thread-count") == 1
        and rir_argv[rir_argv.index("--thread-count") + 1] == "4"
        and not {
            "--acoustic-profile-registry",
            "--room-registry",
            "--room-id",
            "--profile-id",
            "--offset",
            "--limit",
            "--resume",
            "--room-revision",
            "--simulation-profile",
            "--job-offset",
            "--job-limit",
        }.intersection(rir_argv),
        "v8 semantic RIR execution argv drift",
    )
    request_document = _load(paths["rir_cache_receipt"].parent / "request.json")
    simulation_path = Path(
        str(request_document.get("simulation", {}).get("request_path", ""))
    ).resolve()
    required_flags["--simulation-request"] = simulation_path
    for flag, expected_path in required_flags.items():
        raw_path = Path(str(rir_argv[rir_argv.index(flag) + 1]))
        root = REPOSITORY if flag != "--hrtf" else Path("/")
        _require(raw_path.is_absolute(), f"v8 semantic RIR {flag} must be absolute")
        _require_v7_nonsymlink_path(raw_path, root, owner=f"v8 semantic RIR {flag}")
        _require(
            raw_path.resolve() == expected_path.resolve(),
            f"v8 semantic RIR {flag} path drift",
        )
    expected_rir = rir_step.get("expected", {})
    cache_root = paths["rir_cache_receipt"].parent.resolve()
    _require(
        Path(str(expected_rir.get("receipt", ""))).resolve()
        == cache_root / "receipt.json"
        and Path(str(expected_rir.get("index", ""))).resolve()
        == cache_root / "index.json"
        and paths["rir_cache_index"].resolve() == cache_root / "index.json",
        "v8 semantic RIR cache-root declaration drift",
    )
    _require(
        request_document.get("acoustic_scene", {}).get("manifest_path")
        == str(paths["package_manifest"].resolve())
        and request_document.get("acoustic_scene", {}).get("package_id")
        == values["package_manifest"].get("package_id")
        and request_document.get("plan", {}).get("path")
        == str(paths["rir_job_plan"].resolve())
        and request_document.get("output", {}).get("hrtf_path")
        == str(required_flags["--hrtf"])
        and request_document.get("output", {}).get("layout_type") == "binaural",
        "v8 semantic RIR request/package/plan binding drift",
    )

    scene = load_semantic_acoustic_scene(paths["package_manifest"])
    _require(
        scene.package_id == values["package_manifest"].get("package_id")
        and len(scene.objects) == len(values["package_manifest"].get("objects", []))
        and len(scene.objects)
        == values["package_manifest"]
        .get("geometry", {})
        .get("source_node_instance_count")
        and len(scene.objects) > 0
        and len(scene.material_name_by_category) > 0,
        "v8 semantic acoustic package structure drift",
    )
    normalized_jobs = validate_semantic_rir_job_plan(values["rir_job_plan"])
    raw_jobs = values["rir_job_plan"].get("jobs")
    preflight_rir = preflight.get("rir", {})
    _require(
        len(normalized_jobs) == 2
        and isinstance(raw_jobs, list)
        and len(raw_jobs) == 2
        and preflight_rir.get("status") == "planned_not_run"
        and preflight_rir.get("compute_device") == "CPU"
        and preflight_rir.get("unique_rir_job_count") == 2
        and preflight_rir.get("requested_pair_state_count") == 150
        and values["rir_job_plan"].get("stride_frames") == 1
        and values["rir_job_plan"].get("requested_pair_state_count") == 150,
        "v8 exact2 semantic RIR plan closure drift",
    )
    source_positions = preflight_rir.get("source_positions_m")
    _require(
        isinstance(source_positions, Mapping)
        and set(source_positions) == set(slot_actor_ids)
        and all(
            _v7_vector3(source_positions[slot], owner=f"v8 {slot} RIR source")
            == expected_sources[slot]
            for slot in slot_actor_ids
        )
        and _v7_vector3(
            preflight_rir.get("listener_position_m"), owner="v8 RIR listener"
        )
        == listener,
        "v8 preflight RIR pose differs from suite/nav actors",
    )
    by_slot = {
        job["uses"][0]["source_slot_id"]: job
        for job in raw_jobs
        if isinstance(job, Mapping)
        and isinstance(job.get("uses"), list)
        and job["uses"]
    }
    expected_wxyz = [
        expected_rotation_xyzw[3],
        expected_rotation_xyzw[0],
        expected_rotation_xyzw[1],
        expected_rotation_xyzw[2],
    ]
    _require(
        set(by_slot) == set(slot_actor_ids)
        and all(
            _v7_vector3(by_slot[slot].get("source_position_m"), owner=f"v8 {slot} job")
            == expected_sources[slot]
            and _v7_vector3(
                by_slot[slot].get("listener_position_m"), owner="v8 job listener"
            )
            == listener
            and by_slot[slot].get("listener_orientation_wxyz") == expected_wxyz
            and [use.get("frame_index") for use in by_slot[slot]["uses"]]
            == list(range(75))
            and all(
                use.get("episode_id") == episode_id for use in by_slot[slot]["uses"]
            )
            for slot in slot_actor_ids
        ),
        "v8 semantic RIR job/use/pose closure drift",
    )
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))
    from tools.dataset.render_asset_bound_binaural_batch import _SemanticRIRCacheSession

    cache_root = paths["rir_cache_receipt"].parent
    session = _SemanticRIRCacheSession(
        cache_root=cache_root,
        plan_path=paths["rir_job_plan"],
        expected_episode_id=episode_id,
        frame_count=75,
        frame_rate_hz=15,
    )
    cached_episode = session.load_episode(episode_id)
    _require(
        cached_episode.samples.shape[0:3] == (75, 2, 2)
        and cached_episode.lengths.shape == (75, 2)
        and cached_episode.sample_rate_hz == 16_000
        and cached_episode.layout_type == "binaural"
        and cached_episode.layout_id == "rlr_binaural_lr_v1"
        and cached_episode.source_slot_ids == ("source1", "source2")
        and cached_episode.visual_frame_indices == tuple(range(75)),
        "v8 semantic RIR cache consumer closure drift",
    )
    sparse = next(
        step for step in plan["gpu_steps"] if step.get("step_id") == "sparse_f15_probe"
    )
    argv = list(sparse["argv"])
    _require(
        argv[argv.index("--scenario-id") + 1] == episode_id,
        "v8 execution plan and Episode ID differ",
    )
    return {
        **values,
        "episode_id": episode_id,
        "scene_id": scene_id,
        "capture_argv": argv,
    }


def _validate_execution_plan_v8_projection(
    execution_plan_path: Path, *, capture_must_be_fresh: bool
) -> dict[str, Any]:
    """Project schema-v2 CPU evidence onto the exact sparse v8 argv."""

    paths = _v8_execution_plan_paths(execution_plan_path)
    evidence = _validate_v8_execution_plan_evidence(paths)
    source_argv = evidence["capture_argv"]
    capture_runner = (
        REPOSITORY / "tools/qa/capture_spear_imported_glb_strict_two_human_episode.py"
    )
    _require(
        _is_authoritative_capture_python(Path(source_argv[0]))
        and CAPTURE_PYTHON_LOGICAL.is_file()
        and Path(source_argv[0]).resolve() == CAPTURE_PYTHON_LOGICAL.resolve()
        and capture_runner.is_file()
        and Path(source_argv[1]).resolve() == capture_runner
        and SPEAR_ROOT.is_dir()
        and Path(source_argv[source_argv.index("--spear-root") + 1]).resolve()
        == SPEAR_ROOT,
        "v8 capture runtime missing or drifted",
    )
    plan_path = paths["execution_plan"].resolve()
    atom_root = plan_path.parent.parent.resolve()
    _require(
        plan_path == atom_root / "cpu_preflight_v1/execution_plan.json",
        "v8 must bind the fresh cpu_preflight_v1 execution plan",
    )
    _require(
        isinstance(source_argv, list)
        and all(isinstance(item, str) and item for item in source_argv),
        "v8 source capture argv is invalid",
    )
    argv = list(source_argv)
    capture_output = atom_root / V8_CAPTURE_DIRECTORY
    for flag, value in (
        ("--rpc-port", str(V8_RPC_PORT)),
        ("--output", str(capture_output)),
    ):
        _require(argv.count(flag) == 1, f"v8 source argv must contain one {flag}")
        index = argv.index(flag)
        _require(index + 1 < len(argv), f"v8 source argv lacks a value for {flag}")
        argv[index + 1] = value
    _require(
        argv[argv.index("--frame-index") + 1] == str(FRAME_INDEX)
        and argv[argv.index("--graphics-adapter") + 1] == "1"
        and argv[argv.index("--rpc-port") + 1] == str(V8_RPC_PORT)
        and argv[argv.index("--output") + 1] == str(capture_output),
        "v8 sparse capture argv drift",
    )
    if capture_must_be_fresh:
        _require(
            not capture_output.exists(),
            "v8 sparse capture output must be a fresh path",
        )
    else:
        _require(
            capture_output.is_dir(),
            "v8 consumed sparse capture output is missing",
        )
    return {
        "schema": "avengine_mp3d_f15_execution_plan_offline_validation_v8",
        "status": (
            "pass_offline_no_write_no_gpu_query"
            if capture_must_be_fresh
            else "pass_consumed_capture_evidence"
        ),
        "episode_id": evidence["episode_id"],
        "scene_id": evidence["scene_id"],
        "execution_plan": str(plan_path),
        "evidence_paths": {
            name: str(path) for name, path in paths.items() if name != "capture_output"
        },
        "capture_output": str(capture_output),
        "capture_argv": argv,
        "candidate_revision": "fresh_schema_v2_cpu_semantic_sparse_f15_v8",
        "rpc_port": V8_RPC_PORT,
        "gpu_query_started": False,
        "gpu_started": False,
        "writes_performed": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def offline_validate_execution_plan_v8(
    execution_plan_path: Path,
) -> dict[str, Any]:
    """Validate a fresh schema-v2 plan without querying a GPU or writing."""

    return _validate_execution_plan_v8_projection(
        execution_plan_path, capture_must_be_fresh=True
    )


def offline_validate_execution_plan_v7(
    execution_plan_path: Path,
) -> dict[str, Any]:
    """Retarget the corrected v5 preflight to one fresh sparse v7 attempt."""

    raw_plan = _validate_v7_raw_evidence_paths(execution_plan_path)
    paths = _execution_plan_artifact_paths(raw_plan)
    evidence = _validate_v7_execution_plan_evidence(paths)
    source_argv = evidence["capture_argv"]
    capture_runner = (
        REPOSITORY / "tools/qa/capture_spear_imported_glb_strict_two_human_episode.py"
    )
    _require(
        _is_authoritative_capture_python(Path(source_argv[0]))
        and CAPTURE_PYTHON_LOGICAL.is_file()
        and Path(source_argv[0]).resolve() == CAPTURE_PYTHON_LOGICAL.resolve()
        and capture_runner.is_file()
        and Path(source_argv[1]).resolve() == capture_runner
        and SPEAR_ROOT.is_dir()
        and Path(source_argv[source_argv.index("--spear-root") + 1]).resolve()
        == SPEAR_ROOT,
        "v7 capture runtime missing or drifted",
    )
    plan_path = paths["execution_plan"].resolve()
    atom_root = plan_path.parent.parent.resolve()
    _require(
        plan_path == atom_root / "cpu_preflight_v5/execution_plan.json",
        "v7 must bind the corrected cpu_preflight_v5 execution plan",
    )
    _require(
        isinstance(source_argv, list)
        and all(isinstance(item, str) and item for item in source_argv),
        "v7 source capture argv is invalid",
    )
    argv = list(source_argv)
    capture_output = atom_root / V7_CAPTURE_DIRECTORY
    for flag, value in (
        ("--rpc-port", str(V7_RPC_PORT)),
        ("--output", str(capture_output)),
    ):
        _require(argv.count(flag) == 1, f"v7 source argv must contain one {flag}")
        index = argv.index(flag)
        _require(index + 1 < len(argv), f"v7 source argv lacks a value for {flag}")
        argv[index + 1] = value
    _require(
        argv[argv.index("--frame-index") + 1] == str(FRAME_INDEX)
        and argv[argv.index("--graphics-adapter") + 1] == "1"
        and argv[argv.index("--rpc-port") + 1] == str(V7_RPC_PORT)
        and argv[argv.index("--output") + 1] == str(capture_output),
        "v7 sparse capture argv drift",
    )
    _require(
        not capture_output.exists(), "v7 sparse capture output must be a fresh path"
    )
    return {
        "schema": "avengine_mp3d_f15_execution_plan_offline_validation_v7",
        "status": "pass_offline_no_write_no_gpu_query",
        "episode_id": evidence["episode_id"],
        "scene_id": evidence["scene_id"],
        "execution_plan": str(plan_path),
        "evidence_paths": {
            name: str(path) for name, path in paths.items() if name != "capture_output"
        },
        "capture_output": str(capture_output),
        "capture_argv": argv,
        "candidate_revision": "corrected_cpu_preflight_v5_sparse_f15_v7",
        "rpc_port": V7_RPC_PORT,
        "gpu_query_started": False,
        "gpu_started": False,
        "writes_performed": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def _prepare_path_only_sparse_request(
    *,
    execution_plan_path: Path,
    offline_validator: Any,
    request_schema: str,
    attempt_directory: str,
    capture_directory: str,
    rpc_port: int,
    candidate_revision: str,
    owner: str,
) -> Path:
    validation = offline_validator(execution_plan_path)
    plan_path = Path(validation["execution_plan"]).resolve()
    atom_root = plan_path.parent.parent.resolve()
    attempt_root = _require_v7_nonsymlink_path(
        atom_root / attempt_directory, atom_root, owner=f"{owner} launch attempt"
    )
    capture_output = _require_v7_nonsymlink_path(
        atom_root / capture_directory, atom_root, owner=f"{owner} capture output"
    )
    _require(not attempt_root.exists(), f"{owner} launch attempt must be a fresh path")
    _require(
        not capture_output.exists(), f"{owner} capture output must be a fresh path"
    )
    _require(
        _git_tracked_and_index_clean(REPOSITORY),
        f"{owner} preparation requires a clean tracked worktree and index",
    )
    _assert_port_available(rpc_port)
    request = {
        "schema": request_schema,
        "status": "prepared_not_launched",
        "episode_id": validation["episode_id"],
        "scene_id": validation["scene_id"],
        "required_repo_commit": _git_head(REPOSITORY),
        "repo_root": str(REPOSITORY),
        "atom_root": str(atom_root),
        "attempt_root": str(attempt_root),
        "execution_plan": validation["execution_plan"],
        "evidence_paths": validation["evidence_paths"],
        "suite_plan": validation["evidence_paths"]["suite_plan"],
        "room_adapter": validation["evidence_paths"]["room_adapter"],
        "capture_output": str(capture_output),
        "capture_argv": validation["capture_argv"],
        "rpc_port": rpc_port,
        "attempt_policy": {
            **ATTEMPT_POLICY,
            "candidate_revision": candidate_revision,
        },
        "frame_indices": [FRAME_INDEX],
        "full75_allowed": False,
        "physical_gpu_index": 1,
        "physical_gpu_uuid": GPU1_UUID,
        "graphics_adapter_argument": 1,
        "required_idle_compute_process_count": 0,
        "explicit_gpu_capture_authorization_required": True,
        "gpu_capture_authorized_at_prepare": False,
        "manual_review_required": True,
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "created_at_utc": _utc_now(),
    }
    attempt_root.mkdir(parents=True)
    request_path = attempt_root / "request.json"
    _write_json_exclusive(request_path, request)
    return request_path


def prepare_request_v7(*, execution_plan_path: Path) -> Path:
    return _prepare_path_only_sparse_request(
        execution_plan_path=execution_plan_path,
        offline_validator=offline_validate_execution_plan_v7,
        request_schema=REQUEST_SCHEMA_V7,
        attempt_directory=V7_ATTEMPT_DIRECTORY,
        capture_directory=V7_CAPTURE_DIRECTORY,
        rpc_port=V7_RPC_PORT,
        candidate_revision="corrected_cpu_preflight_v5_sparse_f15_v7",
        owner="v7",
    )


def prepare_request_v8(*, execution_plan_path: Path) -> Path:
    return _prepare_path_only_sparse_request(
        execution_plan_path=execution_plan_path,
        offline_validator=offline_validate_execution_plan_v8,
        request_schema=REQUEST_SCHEMA_V8,
        attempt_directory=V8_ATTEMPT_DIRECTORY,
        capture_directory=V8_CAPTURE_DIRECTORY,
        rpc_port=V8_RPC_PORT,
        candidate_revision="fresh_schema_v2_cpu_semantic_sparse_f15_v8",
        owner="v8",
    )


def prepare_request_v5(*, execution_plan_path: Path) -> Path:
    validation = offline_validate_execution_plan(execution_plan_path.resolve())
    plan_path = Path(validation["execution_plan"])
    atom_root = plan_path.parent.parent
    attempt_root = atom_root / V5_ATTEMPT_DIRECTORY
    _require(not attempt_root.exists(), "execution-plan v5 attempt already exists")
    request = {
        "schema": REQUEST_SCHEMA_V5,
        "status": "prepared_not_launched",
        "episode_id": validation["episode_id"],
        "scene_id": validation["scene_id"],
        "required_repo_commit": _git_head(REPOSITORY),
        "repo_root": str(REPOSITORY),
        "atom_root": str(atom_root),
        "attempt_root": str(attempt_root),
        "execution_plan": validation["execution_plan"],
        "evidence_paths": validation["evidence_paths"],
        "suite_plan": validation["evidence_paths"]["suite_plan"],
        "room_adapter": validation["evidence_paths"]["room_adapter"],
        "capture_output": validation["capture_output"],
        "capture_argv": validation["capture_argv"],
        "attempt_policy": {
            **ATTEMPT_POLICY,
            "candidate_revision": "execution_plan_v2_selected_camera",
        },
        "frame_indices": [FRAME_INDEX],
        "full75_allowed": False,
        "physical_gpu_index": 1,
        "physical_gpu_uuid": GPU1_UUID,
        "graphics_adapter_argument": 1,
        "required_idle_compute_process_count": 0,
        "explicit_gpu_capture_authorization_required": True,
        "gpu_capture_authorized_at_prepare": False,
        "manual_review_required": True,
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "created_at_utc": _utc_now(),
    }
    attempt_root.mkdir(parents=True)
    request_path = attempt_root / "request.json"
    _write_json_exclusive(request_path, request)
    return request_path


def prepare_request_v6(*, execution_plan_path: Path) -> Path:
    validation = offline_validate_execution_plan_v6(execution_plan_path.resolve())
    plan_path = Path(validation["execution_plan"])
    atom_root = plan_path.parent.parent
    attempt_root = atom_root / V6_ATTEMPT_DIRECTORY
    _require(not attempt_root.exists(), "execution-plan v6 attempt already exists")
    request = {
        "schema": REQUEST_SCHEMA_V6,
        "status": "prepared_not_launched",
        "episode_id": validation["episode_id"],
        "scene_id": validation["scene_id"],
        "required_repo_commit": _git_head(REPOSITORY),
        "repo_root": str(REPOSITORY),
        "atom_root": str(atom_root),
        "attempt_root": str(attempt_root),
        "execution_plan": validation["execution_plan"],
        "evidence_paths": validation["evidence_paths"],
        "suite_plan": validation["evidence_paths"]["suite_plan"],
        "room_adapter": validation["evidence_paths"]["room_adapter"],
        "capture_output": validation["capture_output"],
        "capture_argv": validation["capture_argv"],
        "attempt_policy": {
            **ATTEMPT_POLICY,
            "candidate_revision": "execution_plan_v3_packaged_readback_closed",
        },
        "frame_indices": [FRAME_INDEX],
        "full75_allowed": False,
        "physical_gpu_index": 1,
        "physical_gpu_uuid": GPU1_UUID,
        "graphics_adapter_argument": 1,
        "required_idle_compute_process_count": 0,
        "explicit_gpu_capture_authorization_required": True,
        "gpu_capture_authorized_at_prepare": False,
        "manual_review_required": True,
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "created_at_utc": _utc_now(),
    }
    attempt_root.mkdir(parents=True)
    request_path = attempt_root / "request.json"
    _write_json_exclusive(request_path, request)
    return request_path


def _validate_path_only_sparse_request(
    request_path: Path,
    *,
    offline_validator: Any,
    request_schema: str,
    attempt_directory: str,
    capture_directory: str,
    rpc_port: int,
    candidate_revision: str,
    owner: str,
) -> tuple[dict[str, Any], list[str]]:
    _require_v7_nonsymlink_path(request_path, REPOSITORY, owner=f"{owner} request")
    request = _load(request_path)
    _require(request.get("schema") == request_schema, f"{owner} request schema drift")
    _require(
        request.get("status") == "prepared_not_launched"
        and request.get("frame_indices") == [FRAME_INDEX]
        and request.get("full75_allowed") is False
        and request.get("physical_gpu_index") == 1
        and request.get("physical_gpu_uuid") == GPU1_UUID
        and request.get("graphics_adapter_argument") == 1
        and request.get("required_idle_compute_process_count") == 0
        and request.get("explicit_gpu_capture_authorization_required") is True
        and request.get("gpu_capture_authorized_at_prepare") is False
        and request.get("manual_review_required") is True
        and request.get("qualification_claim") is False
        and request.get("formal_dataset_count") == 0,
        f"{owner} request boundary drift",
    )
    _require(
        request.get("attempt_policy")
        == {**ATTEMPT_POLICY, "candidate_revision": candidate_revision},
        f"{owner} attempt policy drift",
    )
    repo_root = Path(str(request.get("repo_root", ""))).resolve()
    _require(repo_root == REPOSITORY, f"{owner} repository drift")
    _require(
        _git_tracked_and_index_clean(repo_root),
        f"{owner} launch requires a clean tracked worktree and index",
    )
    _require(
        request.get("required_repo_commit") == _git_head(repo_root),
        f"repository HEAD differs from the {owner} request-bound commit",
    )
    plan_path = Path(str(request.get("execution_plan", "")))
    validation = offline_validator(plan_path)
    atom_root = plan_path.parent.parent.resolve()
    attempt_root = _require_v7_nonsymlink_path(
        atom_root / attempt_directory, atom_root, owner=f"{owner} launch attempt"
    )
    capture_output = _require_v7_nonsymlink_path(
        atom_root / capture_directory, atom_root, owner=f"{owner} capture output"
    )
    _require(
        Path(str(request.get("atom_root", ""))).resolve() == atom_root
        and Path(str(request.get("attempt_root", ""))).resolve() == attempt_root
        and request_path.resolve() == attempt_root / "request.json"
        and not request_path.is_symlink()
        and Path(str(request.get("capture_output", ""))).resolve() == capture_output,
        f"{owner} fresh path containment drift",
    )
    argv = validation["capture_argv"]
    _require(
        request.get("episode_id") == validation["episode_id"]
        and request.get("scene_id") == validation["scene_id"]
        and request.get("evidence_paths") == validation["evidence_paths"]
        and request.get("suite_plan") == validation["evidence_paths"]["suite_plan"]
        and request.get("room_adapter") == validation["evidence_paths"]["room_adapter"]
        and request.get("capture_argv") == argv
        and request.get("rpc_port") == rpc_port,
        f"{owner} path-only Episode/capture binding drift",
    )
    _require(not capture_output.exists(), f"{owner} capture output must remain fresh")
    return request, argv


def _validate_request_v7(request_path: Path) -> tuple[dict[str, Any], list[str]]:
    return _validate_path_only_sparse_request(
        request_path,
        offline_validator=offline_validate_execution_plan_v7,
        request_schema=REQUEST_SCHEMA_V7,
        attempt_directory=V7_ATTEMPT_DIRECTORY,
        capture_directory=V7_CAPTURE_DIRECTORY,
        rpc_port=V7_RPC_PORT,
        candidate_revision="corrected_cpu_preflight_v5_sparse_f15_v7",
        owner="v7",
    )


def _validate_request_v8(request_path: Path) -> tuple[dict[str, Any], list[str]]:
    return _validate_path_only_sparse_request(
        request_path,
        offline_validator=offline_validate_execution_plan_v8,
        request_schema=REQUEST_SCHEMA_V8,
        attempt_directory=V8_ATTEMPT_DIRECTORY,
        capture_directory=V8_CAPTURE_DIRECTORY,
        rpc_port=V8_RPC_PORT,
        candidate_revision="fresh_schema_v2_cpu_semantic_sparse_f15_v8",
        owner="v8",
    )


def _validate_request_v5(request_path: Path) -> tuple[dict[str, Any], list[str]]:
    request = _load(request_path)
    _require(request.get("schema") == REQUEST_SCHEMA_V5, "v5 request schema drift")
    _require(
        request.get("status") == "prepared_not_launched"
        and request.get("frame_indices") == [FRAME_INDEX]
        and request.get("full75_allowed") is False
        and request.get("explicit_gpu_capture_authorization_required") is True
        and request.get("gpu_capture_authorized_at_prepare") is False
        and request.get("manual_review_required") is True
        and request.get("qualification_claim") is False
        and request.get("formal_dataset_count") == 0,
        "v5 request boundary drift",
    )
    repo_root = Path(str(request.get("repo_root", ""))).resolve()
    _require(repo_root == REPOSITORY, "v5 repository drift")
    _require(
        request.get("required_repo_commit") == _git_head(repo_root),
        "repository HEAD differs from the v5 request-bound commit",
    )
    plan_path = Path(str(request.get("execution_plan", ""))).resolve()
    paths = _execution_plan_artifact_paths(plan_path)
    evidence = _validate_execution_plan_evidence(paths)
    atom_root = plan_path.parent.parent
    attempt_root = atom_root / V5_ATTEMPT_DIRECTORY
    _require(
        Path(str(request.get("atom_root", ""))).resolve() == atom_root
        and Path(str(request.get("attempt_root", ""))).resolve() == attempt_root
        and request_path.resolve() == attempt_root / "request.json",
        "v5 request path containment drift",
    )
    expected_evidence = {
        name: str(path) for name, path in paths.items() if name != "capture_output"
    }
    _require(
        request.get("evidence_paths") == expected_evidence,
        "v5 path-only evidence closure drift",
    )
    argv = evidence["capture_argv"]
    _require(
        request.get("episode_id") == evidence["episode_id"]
        and request.get("scene_id") == evidence["scene_id"]
        and request.get("suite_plan") == str(paths["suite_plan"])
        and request.get("room_adapter") == str(paths["room_adapter"])
        and request.get("capture_output") == str(paths["capture_output"])
        and request.get("capture_argv") == argv,
        "v5 Episode/capture binding drift",
    )
    _require(not paths["capture_output"].exists(), "v5 capture output must be new")
    return request, argv


def _validate_request_v6(request_path: Path) -> tuple[dict[str, Any], list[str]]:
    request = _load(request_path)
    _require(request.get("schema") == REQUEST_SCHEMA_V6, "v6 request schema drift")
    _require(
        request.get("status") == "prepared_not_launched"
        and request.get("frame_indices") == [FRAME_INDEX]
        and request.get("full75_allowed") is False
        and request.get("explicit_gpu_capture_authorization_required") is True
        and request.get("gpu_capture_authorized_at_prepare") is False
        and request.get("manual_review_required") is True
        and request.get("qualification_claim") is False
        and request.get("formal_dataset_count") == 0,
        "v6 request boundary drift",
    )
    _require(
        request.get("physical_gpu_index") == 1
        and request.get("physical_gpu_uuid") == GPU1_UUID
        and request.get("graphics_adapter_argument") == 1
        and request.get("required_idle_compute_process_count") == 0,
        "v6 physical GPU1/adapter1 binding drift",
    )
    _require(
        request.get("attempt_policy")
        == {
            **ATTEMPT_POLICY,
            "candidate_revision": "execution_plan_v3_packaged_readback_closed",
        },
        "v6 attempt policy drift",
    )
    repo_root = Path(str(request.get("repo_root", ""))).resolve()
    _require(repo_root == REPOSITORY, "v6 repository drift")
    _require(
        request.get("required_repo_commit") == _git_head(repo_root),
        "repository HEAD differs from the v6 request-bound commit",
    )
    plan_path = Path(str(request.get("execution_plan", ""))).resolve()
    validation = offline_validate_execution_plan_v6(plan_path)
    atom_root = plan_path.parent.parent
    attempt_root = atom_root / V6_ATTEMPT_DIRECTORY
    _require(
        Path(str(request.get("atom_root", ""))).resolve() == atom_root
        and Path(str(request.get("attempt_root", ""))).resolve() == attempt_root
        and request_path.resolve() == attempt_root / "request.json",
        "v6 request path containment drift",
    )
    _require(
        request.get("evidence_paths") == validation["evidence_paths"],
        "v6 path-only evidence closure drift",
    )
    argv = validation["capture_argv"]
    _require(
        request.get("episode_id") == validation["episode_id"]
        and request.get("scene_id") == validation["scene_id"]
        and request.get("suite_plan") == validation["evidence_paths"]["suite_plan"]
        and request.get("room_adapter") == validation["evidence_paths"]["room_adapter"]
        and request.get("capture_output") == validation["capture_output"]
        and request.get("capture_argv") == argv,
        "v6 Episode/capture binding drift",
    )
    _require(
        not Path(validation["capture_output"]).exists(),
        "v6 capture output must be new",
    )
    return request, argv


def _validate_v7_capture(
    request: Mapping[str, Any], *, publish_visibility_truth: bool = True
) -> dict[str, Any]:
    """Close the existing v4 camera and per-mesh readback semantics for v7."""

    validation = _validate_capture(
        request, publish_visibility_truth=publish_visibility_truth
    )
    capture_root = Path(str(request["capture_output"]))
    manifest = _load(capture_root / "manifest.json")
    camera = manifest.get("camera_contract")
    _require(isinstance(camera, Mapping), "v7 capture lacks camera contract")
    fov = camera.get("hfov_readback")
    alignment = camera.get("runtime_alignment")
    _require(isinstance(fov, Mapping), "v7 capture lacks named HFOV readback")
    _require(isinstance(alignment, Mapping), "v7 capture lacks runtime alignment")
    handles = fov.get("component_handles")
    observed = fov.get("observed_horizontal_fov_deg_by_component")
    required_names = {"rgb", "depth", "object_ids"}
    _require(
        fov.get("status") == "pass"
        and fov.get("write_method")
        == "named_USpSceneCaptureComponent2D.FOVAngle_property"
        and not isinstance(fov.get("camera_actor_handle"), bool)
        and isinstance(fov.get("camera_actor_handle"), int)
        and fov.get("camera_actor_handle") > 0
        and isinstance(handles, Mapping)
        and set(handles) == required_names
        and all(
            not isinstance(value, bool) and isinstance(value, int) and value > 0
            for value in handles.values()
        )
        and len(set(handles.values())) == len(required_names)
        and isinstance(observed, Mapping)
        and set(observed) == required_names,
        "v7 named scene-capture HFOV evidence drift",
    )
    suite = _load(Path(str(request["suite_plan"])))
    requested_hfov = float(
        suite["scenarios"][0]["plan"]["camera"]["horizontal_fov_deg"]
    )
    _require(
        abs(float(fov.get("requested_horizontal_fov_deg", -1.0)) - requested_hfov)
        <= 1.0e-6
        and all(
            abs(float(observed[name]) - requested_hfov) <= 1.0e-6
            for name in required_names
        ),
        "v7 named scene-capture HFOV values differ from the suite request",
    )
    pass_identities = camera.get("pass_identities")
    _require(
        isinstance(pass_identities, list)
        and [item.get("pass_id") for item in pass_identities]
        == ["normal", "source1_target_only", "source2_target_only"]
        and all(
            item.get("camera_actor_handle") == fov.get("camera_actor_handle")
            and item.get("rgb_component_handle") == handles["rgb"]
            and item.get("metric_depth_component_handle") == handles["depth"]
            and item.get("object_id_component_handle") == handles["object_ids"]
            for item in pass_identities
        )
        and alignment.get("normal_frame_count") == 1
        and alignment.get("target_pass_count") == 2
        and float(alignment.get("maximum_location_drift_cm", -1.0)) == 0.0
        and float(alignment.get("maximum_rotation_drift_deg", -1.0)) == 0.0,
        "v7 camera pass identity or runtime alignment drift",
    )

    room_adapter = _load(Path(str(request["room_adapter"])))
    expected_meshes = room_adapter.get("static_mesh_object_paths")
    readback = _load(capture_root / "room_live_readback.json")
    meshes = readback.get("meshes")
    _require(
        readback.get("schema") == "avengine_spear_imported_glb_live_readback_v1"
        and readback.get("status") == "pass"
        and readback.get("scene_id") == request.get("scene_id")
        and readback.get("entry_map") == "/Engine/Maps/Entry"
        and readback.get("qualification_claim") is False
        and readback.get("formal_dataset_count") == 0
        and isinstance(expected_meshes, list)
        and len(expected_meshes) == EXPECTED_MESH_COUNT
        and len(set(expected_meshes)) == EXPECTED_MESH_COUNT
        and isinstance(meshes, list)
        and len(meshes) == EXPECTED_MESH_COUNT
        and [mesh.get("mesh_index") for mesh in meshes]
        == list(range(EXPECTED_MESH_COUNT))
        and [mesh.get("object_path") for mesh in meshes] == expected_meshes
        and all(
            mesh.get("status") == "pass"
            and mesh.get("readback_method")
            in {
                "UStaticMeshComponent.GetStaticMesh",
                "UStaticMeshComponent.StaticMesh_property",
            }
            and mesh.get("stable_actor_name")
            == f"AVEngine/ImportedGLB/{request['scene_id']}/mesh_{index:03d}"
            and not isinstance(mesh.get("expected_object_handle"), bool)
            and isinstance(mesh.get("expected_object_handle"), int)
            and mesh.get("expected_object_handle") > 0
            and mesh.get("observed_component_mesh_handle")
            == mesh.get("expected_object_handle")
            for index, mesh in enumerate(meshes)
        ),
        "v7 per-mesh live readback drift",
    )
    expected_handles = [mesh["expected_object_handle"] for mesh in meshes]
    observed_handles = [mesh["observed_component_mesh_handle"] for mesh in meshes]
    stable_names = [mesh["stable_actor_name"] for mesh in meshes]
    _require(
        len(set(expected_handles)) == EXPECTED_MESH_COUNT
        and len(set(observed_handles)) == EXPECTED_MESH_COUNT
        and len(set(stable_names)) == EXPECTED_MESH_COUNT,
        "v7 per-mesh live readback identities are not unique",
    )
    return {
        **validation,
        "named_scene_capture_hfov": {
            "status": "pass",
            "component_handles": dict(handles),
            "horizontal_fov_deg": requested_hfov,
            "pass_count": len(pass_identities),
        },
        "per_mesh_live_readback_status": "pass_exact_71_of_71",
    }


def _validate_consumed_request_v8_for_revalidation(
    request_path: Path,
) -> dict[str, Any]:
    request_path = _require_v7_nonsymlink_path(
        request_path, REPOSITORY, owner="v8 validation-only request"
    )
    request = _load(request_path)
    _require(
        request.get("schema") == REQUEST_SCHEMA_V8
        and request.get("status") == "prepared_not_launched"
        and request.get("frame_indices") == [FRAME_INDEX]
        and request.get("full75_allowed") is False
        and request.get("physical_gpu_index") == 1
        and request.get("physical_gpu_uuid") == GPU1_UUID
        and request.get("graphics_adapter_argument") == 1
        and request.get("required_idle_compute_process_count") == 0
        and request.get("explicit_gpu_capture_authorization_required") is True
        and request.get("gpu_capture_authorized_at_prepare") is False
        and request.get("manual_review_required") is True
        and request.get("qualification_claim") is False
        and request.get("formal_dataset_count") == 0,
        "v8 validation-only request boundary drift",
    )
    _require(
        request.get("attempt_policy")
        == {
            **ATTEMPT_POLICY,
            "candidate_revision": "fresh_schema_v2_cpu_semantic_sparse_f15_v8",
        },
        "v8 validation-only attempt policy drift",
    )
    repo_root = Path(str(request.get("repo_root", ""))).resolve()
    _require(repo_root == REPOSITORY, "v8 validation-only repository drift")
    _require(
        _git_tracked_and_index_clean(repo_root),
        "v8 validation-only requires a clean tracked worktree and index",
    )
    capture_commit = request.get("required_repo_commit")
    _require(
        isinstance(capture_commit, str) and capture_commit,
        "v8 validation-only capture commit is missing",
    )
    validator_commit = _git_head(repo_root)
    atom_root = _require_v7_nonsymlink_path(
        Path(str(request.get("atom_root", ""))),
        REPOSITORY,
        owner="v8 validation-only atom",
    )
    plan_path = _require_v7_nonsymlink_path(
        Path(str(request.get("execution_plan", ""))),
        atom_root,
        owner="v8 validation-only execution plan",
    )
    projection = _validate_execution_plan_v8_projection(
        plan_path, capture_must_be_fresh=False
    )
    attempt_root = _require_v7_nonsymlink_path(
        Path(str(request.get("attempt_root", ""))),
        atom_root,
        owner="v8 validation-only attempt",
    )
    capture_root = _require_v7_nonsymlink_path(
        Path(str(request.get("capture_output", ""))),
        atom_root,
        owner="v8 validation-only capture",
    )
    _require(
        attempt_root == atom_root / V8_ATTEMPT_DIRECTORY
        and request_path == attempt_root / "request.json"
        and capture_root == atom_root / V8_CAPTURE_DIRECTORY
        and request.get("episode_id") == projection["episode_id"]
        and request.get("scene_id") == projection["scene_id"]
        and request.get("execution_plan") == projection["execution_plan"]
        and request.get("evidence_paths") == projection["evidence_paths"]
        and request.get("suite_plan") == projection["evidence_paths"]["suite_plan"]
        and request.get("room_adapter") == projection["evidence_paths"]["room_adapter"]
        and request.get("capture_output") == projection["capture_output"]
        and request.get("capture_argv") == projection["capture_argv"]
        and request.get("rpc_port") == V8_RPC_PORT,
        "v8 consumed request/evidence/capture binding drift",
    )
    common = {
        "schema": RECEIPT_SCHEMA_V8,
        "episode_id": request["episode_id"],
        "scene_id": request["scene_id"],
        "candidate_revision": "fresh_schema_v2_cpu_semantic_sparse_f15_v8",
        "required_repo_commit": capture_commit,
        "request": str(request_path),
        "execution_plan": request["execution_plan"],
        "evidence_paths": request["evidence_paths"],
        "capture_argv": request["capture_argv"],
        "capture_output": str(capture_root),
        "rpc_port": V8_RPC_PORT,
        "frame_indices": [FRAME_INDEX],
        "full75_allowed": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    running_path = _require_v7_nonsymlink_path(
        attempt_root / "running_receipt.json",
        attempt_root,
        owner="v8 original running receipt",
    )
    final_path = _require_v7_nonsymlink_path(
        attempt_root / "final_receipt.json",
        attempt_root,
        owner="v8 original final receipt",
    )
    for name in (
        "manifest.json",
        "room_live_readback.json",
        "metric_depth_native.npz",
    ):
        artifact = _require_v7_nonsymlink_path(
            capture_root / name,
            capture_root,
            owner=f"v8 validation-only capture artifact {name}",
        )
        _require(
            artifact.is_file(),
            f"v8 validation-only capture artifact is missing: {name}",
        )
    for field in ("suite_plan", "room_adapter"):
        artifact = _require_v7_nonsymlink_path(
            Path(str(request.get(field, ""))),
            atom_root,
            owner=f"v8 validation-only {field}",
        )
        _require(artifact.is_file(), f"v8 validation-only {field} is missing")
    running = _load(running_path)
    original_final = _load(final_path)
    _require(
        all(running.get(key) == value for key, value in common.items())
        and running.get("status") == "running"
        and running.get("attempt_consumed") is True
        and running.get("gpu_started") is False
        and running.get("child_exit_code") is None,
        "v8 original running receipt capture provenance drift",
    )
    _require(
        all(original_final.get(key) == value for key, value in common.items())
        and original_final.get("status") == "failed"
        and original_final.get("attempt_consumed") is True
        and original_final.get("gpu_started") is True
        and original_final.get("child_exit_code") == 0
        and original_final.get("capture_process_exit_code") == 0
        and isinstance(original_final.get("error"), str)
        and bool(original_final["error"].strip())
        and original_final.get("validation") is None,
        "v8 original final receipt is not the completed capture validator failure",
    )
    observability = original_final.get("capture_observability")
    _require(
        isinstance(observability, Mapping)
        and observability.get("capture_failure_artifact") is None,
        "v8 original capture failure observability drift",
    )
    _validate_complete_v2_phase_sequence(observability)
    return {
        "request": request,
        "request_path": request_path,
        "attempt_root": attempt_root,
        "capture_root": capture_root,
        "running_receipt": running_path,
        "original_final_receipt": final_path,
        "capture_required_repo_commit": capture_commit,
        "validator_repo_commit": validator_commit,
    }


def validate_v8_capture_only(request_path: Path, *, output_receipt: Path) -> Path:
    """Revalidate a consumed v8 capture without launching or mutating it."""

    context = _validate_consumed_request_v8_for_revalidation(request_path)
    request = context["request"]
    attempt_root = context["attempt_root"]
    output_receipt = _require_v7_nonsymlink_path(
        output_receipt,
        attempt_root,
        owner="v8 validation-only output receipt",
    )
    _require(
        output_receipt.parent == attempt_root
        and not output_receipt.exists()
        and not output_receipt.is_symlink(),
        "v8 validation-only receipt must be a fresh attempt-root child",
    )
    common = {
        "schema": VALIDATION_ONLY_RECEIPT_SCHEMA_V8,
        "episode_id": request["episode_id"],
        "scene_id": request["scene_id"],
        "capture_required_repo_commit": context["capture_required_repo_commit"],
        "validator_repo_commit": context["validator_repo_commit"],
        "request": str(context["request_path"]),
        "original_running_receipt": str(context["running_receipt"]),
        "original_final_receipt": str(context["original_final_receipt"]),
        "original_final_status": "failed",
        "capture_output": str(context["capture_root"]),
        "gpu_query_started": False,
        "gpu_started": False,
        "capture_subprocess_started": False,
        "pixel_visibility_truth_publication": "not_written_validation_only",
        "git_read_only_checks_performed": True,
        "attempt_consumed_by_validation": False,
        "full75_allowed": False,
        "manual_sparse_f15_visual_review_required": True,
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "validated_at_utc": _utc_now(),
    }
    try:
        validation = _validate_v7_capture(request, publish_visibility_truth=False)
    except Exception as error:
        raise RuntimeError("v8 validation-only failed without publishing") from error
    receipt = {
        **common,
        "status": "pass_diagnostic_f15_review_ready",
        "validation": validation,
    }
    _write_json_atomic_no_replace(output_receipt, receipt)
    return output_receipt


def _run_path_only_sparse_revision(
    request_path: Path,
    *,
    request_validator: Any,
    receipt_schema: str,
    candidate_revision: str,
    rpc_port: int,
    owner: str,
    offline_validate: bool,
    dry_run: bool,
    authorize_gpu_capture: bool,
) -> int:
    request, argv = request_validator(request_path)
    if offline_validate:
        _require(not dry_run, "choose offline validation or dry-run, not both")
        return 0
    attempt_root = Path(request["attempt_root"])
    dry_receipt = attempt_root / "dry_run_receipt.json"
    running_receipt = attempt_root / "running_receipt.json"
    final_receipt = attempt_root / "final_receipt.json"
    stdout_path = attempt_root / "capture_stdout.log"
    stderr_path = attempt_root / "capture_stderr.log"
    _require(not final_receipt.exists(), f"{owner} already has a final receipt")
    common = {
        "schema": receipt_schema,
        "episode_id": request["episode_id"],
        "scene_id": request["scene_id"],
        "candidate_revision": candidate_revision,
        "required_repo_commit": request["required_repo_commit"],
        "request": str(request_path.resolve()),
        "execution_plan": request["execution_plan"],
        "evidence_paths": request["evidence_paths"],
        "capture_argv": argv,
        "capture_output": request["capture_output"],
        "rpc_port": rpc_port,
        "frame_indices": [FRAME_INDEX],
        "full75_allowed": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    if dry_run:
        _require(not dry_receipt.exists(), f"{owner} dry-run receipt exists")
        _require(not running_receipt.exists(), f"{owner} already started")
        _write_json_exclusive(
            dry_receipt,
            {
                **common,
                "status": "dry_run_pass_not_launched",
                "gpu_query_started": False,
                "gpu_started": False,
                "attempt_consumed": False,
                "captured_at_utc": _utc_now(),
            },
        )
        return 0
    _require(
        authorize_gpu_capture,
        f"{owner} GPU capture lacks explicit launch authorization",
    )
    _require(not running_receipt.exists(), f"{owner} already started")
    _require(
        not stdout_path.exists() and not stderr_path.exists(),
        f"{owner} exclusive child logs already exist",
    )
    before = _gpu_snapshot()
    gpu = _validate_gpu1_idle(before)
    _assert_port_available(rpc_port)
    common.update(
        {
            "physical_gpu_index": 1,
            "physical_gpu_uuid": GPU1_UUID,
            "graphics_adapter_argument": 1,
            "prelaunch_gpu": gpu,
            "prelaunch_snapshot": before,
        }
    )
    started_at = _utc_now()
    _write_json_exclusive(
        running_receipt,
        {
            **common,
            "status": "running",
            "gpu_started": False,
            "attempt_consumed": True,
            "started_at_utc": started_at,
            "child_exit_code": None,
        },
    )
    child_exit_code: int | None = None
    final: dict[str, Any] = {
        **common,
        "status": "failed",
        "gpu_started": False,
        "attempt_consumed": True,
        "retry_same_candidate_forbidden": True,
        "started_at_utc": started_at,
    }
    exit_code = 1
    try:
        with (
            stdout_path.open("xb") as stdout_stream,
            stderr_path.open("xb") as stderr_stream,
        ):
            completed = subprocess.run(
                argv,
                cwd=REPOSITORY,
                check=False,
                stdout=stdout_stream,
                stderr=stderr_stream,
            )
        child_exit_code = int(completed.returncode)
        exit_code = child_exit_code
        _require(child_exit_code == 0, f"{owner} f15 capture exited {exit_code}")
        observability = _collect_v2_capture_observability(
            Path(request["capture_output"])
        )
        _validate_complete_v2_phase_sequence(observability)
        final["capture_observability"] = observability
        final["validation"] = _validate_v7_capture(request)
        final["status"] = "pass_diagnostic_f15_review_ready"
    except Exception as exc:  # noqa: BLE001
        final["error"] = f"{type(exc).__name__}: {exc}"
        final["launcher_traceback"] = traceback.format_exc()
        exit_code = exit_code or 1
    finally:
        final["ended_at_utc"] = _utc_now()
        final["child_exit_code"] = child_exit_code
        final["capture_process_exit_code"] = child_exit_code
        final["gpu_started"] = child_exit_code is not None
        final["exclusive_child_stdout"] = (
            _file_record(stdout_path) if stdout_path.is_file() else None
        )
        final["exclusive_child_stderr"] = (
            _file_record(stderr_path) if stderr_path.is_file() else None
        )
        try:
            final["postlaunch_snapshot"] = _gpu_snapshot()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            final["postlaunch_snapshot_error"] = f"{type(exc).__name__}: {exc}"
        _write_json_exclusive(final_receipt, final)
    return exit_code


def run_v7(
    request_path: Path,
    *,
    offline_validate: bool,
    dry_run: bool,
    authorize_gpu_capture: bool,
) -> int:
    return _run_path_only_sparse_revision(
        request_path,
        request_validator=_validate_request_v7,
        receipt_schema=RECEIPT_SCHEMA_V7,
        candidate_revision="corrected_cpu_preflight_v5_sparse_f15_v7",
        rpc_port=V7_RPC_PORT,
        owner="v7",
        offline_validate=offline_validate,
        dry_run=dry_run,
        authorize_gpu_capture=authorize_gpu_capture,
    )


def run_v8(
    request_path: Path,
    *,
    offline_validate: bool,
    dry_run: bool,
    authorize_gpu_capture: bool,
) -> int:
    return _run_path_only_sparse_revision(
        request_path,
        request_validator=_validate_request_v8,
        receipt_schema=RECEIPT_SCHEMA_V8,
        candidate_revision="fresh_schema_v2_cpu_semantic_sparse_f15_v8",
        rpc_port=V8_RPC_PORT,
        owner="v8",
        offline_validate=offline_validate,
        dry_run=dry_run,
        authorize_gpu_capture=authorize_gpu_capture,
    )


def run_v5(
    request_path: Path,
    *,
    offline_validate: bool,
    dry_run: bool,
    authorize_gpu_capture: bool,
) -> int:
    request, argv = _validate_request_v5(request_path.resolve())
    if offline_validate:
        _require(not dry_run, "choose offline validation or dry-run, not both")
        return 0

    attempt_root = Path(request["attempt_root"])
    dry_receipt = attempt_root / "dry_run_receipt.json"
    running_receipt = attempt_root / "running_receipt.json"
    final_receipt = attempt_root / "final_receipt.json"
    stdout_path = attempt_root / "capture_stdout.log"
    stderr_path = attempt_root / "capture_stderr.log"
    _require(
        not final_receipt.exists(), "execution-plan v5 already has a final receipt"
    )
    common = {
        "schema": RECEIPT_SCHEMA_V5,
        "episode_id": request["episode_id"],
        "scene_id": request["scene_id"],
        "required_repo_commit": request["required_repo_commit"],
        "request": str(request_path.resolve()),
        "execution_plan": request["execution_plan"],
        "capture_argv": argv,
        "capture_output": request["capture_output"],
        "frame_indices": [FRAME_INDEX],
        "full75_allowed": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    if dry_run:
        _require(not dry_receipt.exists(), "execution-plan v5 dry-run receipt exists")
        _require(not running_receipt.exists(), "execution-plan v5 already started")
        _write_json_exclusive(
            dry_receipt,
            {
                **common,
                "status": "dry_run_pass_not_launched",
                "gpu_query_started": False,
                "gpu_started": False,
                "attempt_consumed": False,
                "captured_at_utc": _utc_now(),
            },
        )
        return 0

    _require(
        authorize_gpu_capture, "v5 GPU capture lacks explicit launch authorization"
    )
    _require(not running_receipt.exists(), "execution-plan v5 already started")
    _require(
        not stdout_path.exists() and not stderr_path.exists(),
        "execution-plan v5 exclusive child logs already exist",
    )
    before = _gpu_snapshot()
    gpu = _validate_gpu1_idle(before)
    rpc_port = int(argv[argv.index("--rpc-port") + 1])
    _assert_port_available(rpc_port)
    common.update(
        {
            "physical_gpu_index": 1,
            "physical_gpu_uuid": GPU1_UUID,
            "graphics_adapter_argument": 1,
            "prelaunch_gpu": gpu,
            "prelaunch_snapshot": before,
        }
    )
    started_at = _utc_now()
    _write_json_exclusive(
        running_receipt,
        {
            **common,
            "status": "running",
            "gpu_started": False,
            "attempt_consumed": True,
            "started_at_utc": started_at,
            "child_exit_code": None,
        },
    )
    child_exit_code: int | None = None
    final: dict[str, Any] = {
        **common,
        "status": "failed",
        "gpu_started": False,
        "attempt_consumed": True,
        "retry_same_candidate_forbidden": True,
        "started_at_utc": started_at,
    }
    exit_code = 1
    try:
        with (
            stdout_path.open("xb") as stdout_stream,
            stderr_path.open("xb") as stderr_stream,
        ):
            completed = subprocess.run(
                argv,
                cwd=REPOSITORY,
                check=False,
                stdout=stdout_stream,
                stderr=stderr_stream,
            )
        child_exit_code = int(completed.returncode)
        exit_code = child_exit_code
        _require(child_exit_code == 0, f"v5 f15 capture exited {exit_code}")
        observability = _collect_v2_capture_observability(
            Path(request["capture_output"])
        )
        _validate_complete_v2_phase_sequence(observability)
        final["capture_observability"] = observability
        final["validation"] = _validate_capture(request)
        final["status"] = "pass_diagnostic_f15_review_ready"
    except Exception as exc:  # noqa: BLE001
        final["error"] = f"{type(exc).__name__}: {exc}"
        final["launcher_traceback"] = traceback.format_exc()
        exit_code = exit_code or 1
    finally:
        final["ended_at_utc"] = _utc_now()
        final["child_exit_code"] = child_exit_code
        final["capture_process_exit_code"] = child_exit_code
        final["gpu_started"] = child_exit_code is not None
        final["exclusive_child_stdout"] = (
            _file_record(stdout_path) if stdout_path.is_file() else None
        )
        final["exclusive_child_stderr"] = (
            _file_record(stderr_path) if stderr_path.is_file() else None
        )
        try:
            final["postlaunch_snapshot"] = _gpu_snapshot()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            final["postlaunch_snapshot_error"] = f"{type(exc).__name__}: {exc}"
        _write_json_exclusive(final_receipt, final)
    return exit_code


def run_v6(
    request_path: Path,
    *,
    offline_validate: bool,
    dry_run: bool,
    authorize_gpu_capture: bool,
) -> int:
    request, argv = _validate_request_v6(request_path.resolve())
    if offline_validate:
        _require(not dry_run, "choose offline validation or dry-run, not both")
        return 0

    attempt_root = Path(request["attempt_root"])
    dry_receipt = attempt_root / "dry_run_receipt.json"
    running_receipt = attempt_root / "running_receipt.json"
    final_receipt = attempt_root / "final_receipt.json"
    stdout_path = attempt_root / "capture_stdout.log"
    stderr_path = attempt_root / "capture_stderr.log"
    _require(
        not final_receipt.exists(), "execution-plan v6 already has a final receipt"
    )
    common = {
        "schema": RECEIPT_SCHEMA_V6,
        "episode_id": request["episode_id"],
        "scene_id": request["scene_id"],
        "required_repo_commit": request["required_repo_commit"],
        "request": str(request_path.resolve()),
        "execution_plan": request["execution_plan"],
        "evidence_paths": request["evidence_paths"],
        "capture_argv": argv,
        "capture_output": request["capture_output"],
        "frame_indices": [FRAME_INDEX],
        "full75_allowed": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    if dry_run:
        _require(not dry_receipt.exists(), "execution-plan v6 dry-run receipt exists")
        _require(not running_receipt.exists(), "execution-plan v6 already started")
        _write_json_exclusive(
            dry_receipt,
            {
                **common,
                "status": "dry_run_pass_not_launched",
                "packaged_room_readback_status": "pass_nullrhi_71_of_71",
                "gpu_query_started": False,
                "gpu_started": False,
                "attempt_consumed": False,
                "captured_at_utc": _utc_now(),
            },
        )
        return 0

    _require(
        authorize_gpu_capture, "v6 GPU capture lacks explicit launch authorization"
    )
    _require(not running_receipt.exists(), "execution-plan v6 already started")
    _require(
        not stdout_path.exists() and not stderr_path.exists(),
        "execution-plan v6 exclusive child logs already exist",
    )
    before = _gpu_snapshot()
    gpu = _validate_gpu1_idle(before)
    rpc_port = int(argv[argv.index("--rpc-port") + 1])
    _assert_port_available(rpc_port)
    common.update(
        {
            "physical_gpu_index": 1,
            "physical_gpu_uuid": GPU1_UUID,
            "graphics_adapter_argument": 1,
            "prelaunch_gpu": gpu,
            "prelaunch_snapshot": before,
        }
    )
    started_at = _utc_now()
    _write_json_exclusive(
        running_receipt,
        {
            **common,
            "status": "running",
            "gpu_started": False,
            "attempt_consumed": True,
            "started_at_utc": started_at,
            "child_exit_code": None,
        },
    )
    child_exit_code: int | None = None
    final: dict[str, Any] = {
        **common,
        "status": "failed",
        "gpu_started": False,
        "attempt_consumed": True,
        "retry_same_candidate_forbidden": True,
        "started_at_utc": started_at,
    }
    exit_code = 1
    try:
        with (
            stdout_path.open("xb") as stdout_stream,
            stderr_path.open("xb") as stderr_stream,
        ):
            completed = subprocess.run(
                argv,
                cwd=REPOSITORY,
                check=False,
                stdout=stdout_stream,
                stderr=stderr_stream,
            )
        child_exit_code = int(completed.returncode)
        exit_code = child_exit_code
        _require(child_exit_code == 0, f"v6 f15 capture exited {exit_code}")
        observability = _collect_v2_capture_observability(
            Path(request["capture_output"])
        )
        _validate_complete_v2_phase_sequence(observability)
        final["capture_observability"] = observability
        final["validation"] = _validate_capture(request)
        final["status"] = "pass_diagnostic_f15_review_ready"
    except Exception as exc:  # noqa: BLE001
        final["error"] = f"{type(exc).__name__}: {exc}"
        final["launcher_traceback"] = traceback.format_exc()
        exit_code = exit_code or 1
    finally:
        final["ended_at_utc"] = _utc_now()
        final["child_exit_code"] = child_exit_code
        final["capture_process_exit_code"] = child_exit_code
        final["gpu_started"] = child_exit_code is not None
        final["exclusive_child_stdout"] = (
            _file_record(stdout_path) if stdout_path.is_file() else None
        )
        final["exclusive_child_stderr"] = (
            _file_record(stderr_path) if stderr_path.is_file() else None
        )
        try:
            final["postlaunch_snapshot"] = _gpu_snapshot()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            final["postlaunch_snapshot_error"] = f"{type(exc).__name__}: {exc}"
        _write_json_exclusive(final_receipt, final)
    return exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--atom-root", required=True, type=Path)
    prepare.add_argument(
        "--capture-python",
        type=Path,
        default=CAPTURE_PYTHON_LOGICAL,
    )
    prepare.add_argument(
        "--spear-root",
        type=Path,
        default=SPEAR_ROOT,
    )
    prepare.add_argument("--rpc-port", type=int, default=39631)
    archive = subparsers.add_parser("archive-preparation-failure")
    archive.add_argument("--atom-root", required=True, type=Path)
    archive.add_argument("--error", required=True)
    ledger = subparsers.add_parser("record-attempt01-failure")
    ledger.add_argument("--atom-root", required=True, type=Path)
    ledger.add_argument("--spear-log", required=True, type=Path)
    prepare_v2 = subparsers.add_parser("prepare-v2")
    prepare_v2.add_argument("--atom-root", required=True, type=Path)
    prepare_v2.add_argument(
        "--capture-python",
        type=Path,
        default=CAPTURE_PYTHON_LOGICAL,
    )
    prepare_v2.add_argument(
        "--spear-root",
        type=Path,
        default=SPEAR_ROOT,
    )
    prepare_v2.add_argument("--rpc-port", type=int, default=V2_RPC_PORT)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--request", required=True, type=Path)
    launch.add_argument("--dry-run", action="store_true")
    launch_v2 = subparsers.add_parser("launch-v2")
    launch_v2.add_argument("--request", required=True, type=Path)
    launch_v2.add_argument("--dry-run", action="store_true")
    launch_v2.add_argument("--authorize-gpu-capture", action="store_true")
    offline_v5 = subparsers.add_parser("offline-validate-v5")
    offline_source = offline_v5.add_mutually_exclusive_group(required=True)
    offline_source.add_argument("--execution-plan", type=Path)
    offline_source.add_argument("--request", type=Path)
    prepare_v5 = subparsers.add_parser("prepare-v5")
    prepare_v5.add_argument("--execution-plan", required=True, type=Path)
    launch_v5 = subparsers.add_parser("launch-v5")
    launch_v5.add_argument("--request", required=True, type=Path)
    launch_v5.add_argument("--dry-run", action="store_true")
    launch_v5.add_argument("--authorize-gpu-capture", action="store_true")
    offline_v6 = subparsers.add_parser("offline-validate-v6")
    offline_v6_source = offline_v6.add_mutually_exclusive_group(required=True)
    offline_v6_source.add_argument("--execution-plan", type=Path)
    offline_v6_source.add_argument("--request", type=Path)
    prepare_v6 = subparsers.add_parser("prepare-v6")
    prepare_v6.add_argument("--execution-plan", required=True, type=Path)
    launch_v6 = subparsers.add_parser("launch-v6")
    launch_v6.add_argument("--request", required=True, type=Path)
    launch_v6.add_argument("--dry-run", action="store_true")
    launch_v6.add_argument("--authorize-gpu-capture", action="store_true")
    offline_v7 = subparsers.add_parser("offline-validate-v7")
    offline_v7_source = offline_v7.add_mutually_exclusive_group(required=True)
    offline_v7_source.add_argument("--execution-plan", type=Path)
    offline_v7_source.add_argument("--request", type=Path)
    prepare_v7 = subparsers.add_parser("prepare-v7")
    prepare_v7.add_argument("--execution-plan", required=True, type=Path)
    launch_v7 = subparsers.add_parser("launch-v7")
    launch_v7.add_argument("--request", required=True, type=Path)
    launch_v7.add_argument("--dry-run", action="store_true")
    launch_v7.add_argument("--authorize-gpu-capture", action="store_true")
    offline_v8 = subparsers.add_parser("offline-validate-v8")
    offline_v8_source = offline_v8.add_mutually_exclusive_group(required=True)
    offline_v8_source.add_argument("--execution-plan", type=Path)
    offline_v8_source.add_argument("--request", type=Path)
    prepare_v8 = subparsers.add_parser("prepare-v8")
    prepare_v8.add_argument("--execution-plan", required=True, type=Path)
    launch_v8 = subparsers.add_parser("launch-v8")
    launch_v8.add_argument("--request", required=True, type=Path)
    launch_v8.add_argument("--dry-run", action="store_true")
    launch_v8.add_argument("--authorize-gpu-capture", action="store_true")
    validate_v8 = subparsers.add_parser("validate-v8-capture")
    validate_v8.add_argument("--request", required=True, type=Path)
    validate_v8.add_argument("--output-receipt", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        path = prepare_request(
            atom_root=args.atom_root,
            capture_python=args.capture_python,
            spear_root=args.spear_root,
            rpc_port=args.rpc_port,
        )
        print(f"MP3D_F15_REQUEST_PREPARED request={path} formal=0", flush=True)
        return 0
    if args.command == "archive-preparation-failure":
        path = archive_preparation_failure(
            atom_root=args.atom_root,
            error=args.error,
        )
        print(f"MP3D_F15_PREPARE_FAILURE_ARCHIVED receipt={path} formal=0", flush=True)
        return 0
    if args.command == "record-attempt01-failure":
        path = record_attempt01_failure_ledger(
            atom_root=args.atom_root,
            spear_log=args.spear_log,
        )
        print(
            "MP3D_F15_ATTEMPT01_FAILURE_FROZEN "
            f"status={ATTEMPT01_FAILURE_STATUS} ledger={path} formal=0",
            flush=True,
        )
        return 0
    if args.command == "prepare-v2":
        path = prepare_request_v2(
            atom_root=args.atom_root,
            capture_python=args.capture_python,
            spear_root=args.spear_root,
            rpc_port=args.rpc_port,
        )
        print(f"MP3D_F15_V2_REQUEST_PREPARED request={path} formal=0", flush=True)
        return 0
    if args.command == "launch-v2":
        return run_v2(
            args.request,
            dry_run=args.dry_run,
            authorize_gpu_capture=args.authorize_gpu_capture,
        )
    if args.command == "offline-validate-v5":
        if args.execution_plan is not None:
            validation = offline_validate_execution_plan(args.execution_plan)
        else:
            request, _ = _validate_request_v5(args.request.resolve())
            validation = {
                "status": "pass_offline_no_write_no_gpu_query",
                "episode_id": request["episode_id"],
                "scene_id": request["scene_id"],
                "request": str(args.request.resolve()),
                "gpu_query_started": False,
                "gpu_started": False,
                "writes_performed": False,
                "qualification_claim": False,
                "formal_dataset_count": 0,
            }
        print(json.dumps(validation, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    if args.command == "prepare-v5":
        path = prepare_request_v5(execution_plan_path=args.execution_plan)
        print(f"MP3D_F15_V5_REQUEST_PREPARED request={path} formal=0", flush=True)
        return 0
    if args.command == "launch-v5":
        return run_v5(
            args.request,
            offline_validate=False,
            dry_run=args.dry_run,
            authorize_gpu_capture=args.authorize_gpu_capture,
        )
    if args.command == "offline-validate-v6":
        if args.execution_plan is not None:
            validation = offline_validate_execution_plan_v6(args.execution_plan)
        else:
            request, _ = _validate_request_v6(args.request.resolve())
            validation = {
                "schema": "avengine_mp3d_f15_execution_plan_offline_validation_v2",
                "status": "pass_offline_no_write_no_gpu_query",
                "episode_id": request["episode_id"],
                "scene_id": request["scene_id"],
                "request": str(args.request.resolve()),
                "packaged_room_readback_status": "pass_nullrhi_71_of_71",
                "gpu_query_started": False,
                "gpu_started": False,
                "writes_performed": False,
                "qualification_claim": False,
                "formal_dataset_count": 0,
            }
        print(json.dumps(validation, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    if args.command == "prepare-v6":
        path = prepare_request_v6(execution_plan_path=args.execution_plan)
        print(f"MP3D_F15_V6_REQUEST_PREPARED request={path} formal=0", flush=True)
        return 0
    if args.command == "launch-v6":
        return run_v6(
            args.request,
            offline_validate=False,
            dry_run=args.dry_run,
            authorize_gpu_capture=args.authorize_gpu_capture,
        )
    if args.command == "offline-validate-v7":
        if args.execution_plan is not None:
            validation = offline_validate_execution_plan_v7(args.execution_plan)
        else:
            request, _ = _validate_request_v7(args.request)
            validation = {
                "status": "pass_offline_no_write_no_gpu_query",
                "episode_id": request["episode_id"],
                "scene_id": request["scene_id"],
                "request": str(args.request.resolve()),
                "candidate_revision": ("corrected_cpu_preflight_v5_sparse_f15_v7"),
                "gpu_query_started": False,
                "gpu_started": False,
                "writes_performed": False,
                "qualification_claim": False,
                "formal_dataset_count": 0,
            }
        print(json.dumps(validation, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    if args.command == "prepare-v7":
        path = prepare_request_v7(execution_plan_path=args.execution_plan)
        print(f"MP3D_F15_V7_REQUEST_PREPARED request={path} formal=0", flush=True)
        return 0
    if args.command == "launch-v7":
        return run_v7(
            args.request,
            offline_validate=False,
            dry_run=args.dry_run,
            authorize_gpu_capture=args.authorize_gpu_capture,
        )
    if args.command == "offline-validate-v8":
        if args.execution_plan is not None:
            validation = offline_validate_execution_plan_v8(args.execution_plan)
        else:
            request, _ = _validate_request_v8(args.request)
            validation = {
                "status": "pass_offline_no_write_no_gpu_query",
                "episode_id": request["episode_id"],
                "scene_id": request["scene_id"],
                "request": str(args.request.resolve()),
                "candidate_revision": "fresh_schema_v2_cpu_semantic_sparse_f15_v8",
                "gpu_query_started": False,
                "gpu_started": False,
                "writes_performed": False,
                "qualification_claim": False,
                "formal_dataset_count": 0,
            }
        print(json.dumps(validation, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    if args.command == "prepare-v8":
        path = prepare_request_v8(execution_plan_path=args.execution_plan)
        print(f"MP3D_F15_V8_REQUEST_PREPARED request={path} formal=0", flush=True)
        return 0
    if args.command == "validate-v8-capture":
        path = validate_v8_capture_only(
            args.request, output_receipt=args.output_receipt
        )
        print(f"MP3D_F15_V8_VALIDATION_ONLY_PASS receipt={path} formal=0", flush=True)
        return 0
    if args.command == "launch-v8":
        return run_v8(
            args.request,
            offline_validate=False,
            dry_run=args.dry_run,
            authorize_gpu_capture=args.authorize_gpu_capture,
        )
    return run(args.request, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

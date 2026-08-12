#!/usr/bin/env python3
"""Fail-closed NullRHI packaged readback for an imported-GLB room adapter.

The outer runner owns process isolation, timeout, no-clobber receipts, and
exact packaged-process closure.  The worker launches packaged SPEAR with
NullRHI and performs only room object load/spawn/component readback.  It does
not create lights, cameras, episode actors, frames, or formal dataset claims.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULT_SCHEMA = "avengine_packaged_imported_glb_room_readback_v1"
EXIT_SCHEMA = "avengine_packaged_imported_glb_room_probe_exit_v1"
READBACK_SCHEMA = "avengine_spear_imported_glb_live_readback_v1"
ENTRY_MAP = "/Engine/Maps/Entry"
EXPECTED_STATIC_MESH_COUNT = 71
DEFAULT_RPC_PORT = 30174
DEFAULT_TIMEOUT_SECONDS = 900


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json_object(path: Path, *, owner: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{owner} JSON root must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _positive_handle(value: Any, *, owner: str) -> int:
    _require(not isinstance(value, bool), f"{owner} must be a positive handle")
    try:
        handle = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{owner} must be a positive handle") from exc
    _require(handle > 0, f"{owner} must be a positive handle")
    return handle


def _adapter_paths(adapter: Mapping[str, Any]) -> list[str]:
    _require(
        adapter.get("entry_map") == ENTRY_MAP,
        "room adapter must use the packaged Entry map",
    )
    paths = adapter.get("static_mesh_object_paths")
    _require(
        isinstance(paths, list)
        and len(paths) == EXPECTED_STATIC_MESH_COUNT
        and all(isinstance(path, str) and path for path in paths)
        and len(set(paths)) == EXPECTED_STATIC_MESH_COUNT,
        "room adapter must declare exactly 71 unique mesh object paths",
    )
    return list(paths)


def validate_live_readback(
    adapter: Mapping[str, Any], readback: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the complete live 71-object closure without file identity data."""

    paths = _adapter_paths(adapter)
    scene_id = adapter.get("scene_id")
    _require(isinstance(scene_id, str) and scene_id, "adapter scene_id is missing")
    records = readback.get("meshes")
    _require(
        readback.get("schema") == READBACK_SCHEMA
        and readback.get("status") == "pass"
        and readback.get("scene_id") == scene_id
        and readback.get("entry_map") == ENTRY_MAP
        and readback.get("expected_static_mesh_count") == EXPECTED_STATIC_MESH_COUNT
        and readback.get("spawned_static_mesh_count") == EXPECTED_STATIC_MESH_COUNT
        and readback.get("all_expected_handles_match_components") is True
        and readback.get("unique_loaded_object_handle_count")
        == EXPECTED_STATIC_MESH_COUNT
        and readback.get("unique_component_mesh_handle_count")
        == EXPECTED_STATIC_MESH_COUNT
        and readback.get("qualification_claim") is False
        and readback.get("formal_dataset_count") == 0
        and isinstance(records, list)
        and len(records) == EXPECTED_STATIC_MESH_COUNT,
        "live imported-room readback did not close the diagnostic 71-mesh boundary",
    )
    expected_handles: list[int] = []
    observed_handles: list[int] = []
    stable_names: list[str] = []
    for index, (object_path, record) in enumerate(zip(paths, records, strict=True)):
        _require(isinstance(record, Mapping), f"mesh readback {index} is not an object")
        expected = _positive_handle(
            record.get("expected_object_handle"), owner=f"mesh {index} loaded handle"
        )
        observed = _positive_handle(
            record.get("observed_component_mesh_handle"),
            owner=f"mesh {index} component handle",
        )
        stable_name = record.get("stable_actor_name")
        _require(
            record.get("mesh_index") == index
            and record.get("object_path") == object_path
            and record.get("status") == "pass"
            and expected == observed
            and isinstance(stable_name, str)
            and stable_name,
            f"mesh readback {index} path, handle, or stable-name drift",
        )
        expected_handles.append(expected)
        observed_handles.append(observed)
        stable_names.append(stable_name)
    _require(
        len(set(expected_handles)) == EXPECTED_STATIC_MESH_COUNT
        and len(set(observed_handles)) == EXPECTED_STATIC_MESH_COUNT
        and len(set(stable_names)) == EXPECTED_STATIC_MESH_COUNT,
        "live imported-room objects or stable actor names are not unique",
    )
    return {
        "status": "pass",
        "scene_id": scene_id,
        "entry_map": ENTRY_MAP,
        "mesh_count": EXPECTED_STATIC_MESH_COUNT,
        "object_paths": paths,
    }


def validate_result_and_exit(
    *,
    adapter: Mapping[str, Any],
    result: Mapping[str, Any],
    exit_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Return f15-bindable semantic evidence only after process closure."""

    _require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("status") == "pass"
        and result.get("readiness_status")
        == "packaged_71_mesh_readback_pass_gpu_f15_pending"
        and result.get("nullrhi") is True
        and result.get("rendering_or_capture_called") is False
        and result.get("qualification_claim") is False
        and result.get("formal_dataset_count") == 0,
        "NullRHI result crossed its semantic claim boundary",
    )
    room = result.get("room_live_readback")
    _require(isinstance(room, Mapping), "NullRHI result lacks room_live_readback")
    validated = validate_live_readback(adapter, room)
    before = exit_receipt.get("exact_packaged_processes_before")
    after = exit_receipt.get("exact_packaged_processes_after")
    _require(
        exit_receipt.get("schema") == EXIT_SCHEMA
        and exit_receipt.get("status") == "pass"
        and exit_receipt.get("worker_exit_code") == 0
        and exit_receipt.get("timed_out") is False
        and exit_receipt.get("result_status") == "pass"
        and before == []
        and after == []
        and exit_receipt.get("exact_packaged_process_exit_closed") is True
        and exit_receipt.get("nullrhi") is True
        and exit_receipt.get("rendering_or_capture_called") is False
        and exit_receipt.get("qualification_claim") is False
        and exit_receipt.get("formal_dataset_count") == 0,
        "NullRHI runner did not close the exact packaged-process boundary",
    )
    return {
        **validated,
        "provenance": "packaged_spear_nullrhi_runtime",
        "gpu_f15_status": "pending",
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def _packaged_paths(executable: Path) -> tuple[Path, Path, Path]:
    launcher = executable.resolve()
    archive = launcher.parent
    binary = archive / "SpearSim/Binaries/Linux/SpearSim"
    package = archive / "SpearSim/Content/Paks/SpearSim-Linux.pak"
    _require(launcher.is_file(), f"packaged launcher is missing: {launcher}")
    _require(
        os.access(launcher, os.X_OK),
        f"packaged launcher is not executable: {launcher}",
    )
    _require(binary.is_file(), f"packaged inner binary is missing: {binary}")
    _require(
        os.access(binary, os.X_OK),
        f"packaged inner binary is not executable: {binary}",
    )
    _require(package.is_file(), f"packaged PAK is missing: {package}")
    return launcher, binary.resolve(), package.resolve()


def _exact_packaged_processes(binary: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return records
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdecimal():
            continue
        try:
            observed = (process_dir / "exe").resolve(strict=True)
        except (FileNotFoundError, PermissionError):
            continue
        if observed != binary:
            continue
        records.append({"pid": int(process_dir.name), "executable": str(observed)})
    return sorted(records, key=lambda record: record["pid"])


def _assert_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def _worker(args: argparse.Namespace) -> int:
    repository = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository / "tools/qa"))
    sys.path.insert(0, str(repository / "src"))
    sys.path.insert(0, str(args.spear_root.resolve() / "python"))

    import spear
    from spear_imported_glb_room_adapter import (
        destroy_scene_meshes,
        load_json_object,
        spawn_scene_meshes_with_readback,
        validate_room_adapter,
    )

    adapter = load_json_object(args.room_adapter.resolve(), owner="room adapter")
    validate_room_adapter(adapter)
    _adapter_paths(adapter)

    config = spear.get_config(user_config_files=[])
    config.defrost()
    config.SPEAR.LAUNCH_MODE = "game"
    config.SPEAR.INSTANCE.GAME_EXECUTABLE = str(args.executable.resolve())
    config.SPEAR.INSTANCE.INITIALIZE_CLIENT_MAX_TIME_SECONDS = 600.0
    config.SPEAR.INSTANCE.CLIENT_INTERNAL_TIMEOUT_SECONDS = 120.0
    config.SPEAR.INSTANCE.TEMP_DIR = str(
        args.run_dir.resolve() / f"spear_instance_{args.rpc_port}"
    )
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.log = (
        f"SpearSim_imported_glb_nullrhi_{args.rpc_port}.log"
    )
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.nullrhi = None
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.unattended = None
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.nop4 = None
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.nosplash = None
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = ENTRY_MAP
    config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT = args.rpc_port
    config.SP_CORE.SHARED_MEMORY_INITIAL_UNIQUE_ID = args.rpc_port * 10000
    config.freeze()
    spear.configure_system(config=config)

    instance = None
    actors: Sequence[Any] = []
    actors_destroyed = False
    try:
        instance = spear.Instance(config=config)
        game = instance.get_game()
        with instance.begin_frame():
            actors, readback = spawn_scene_meshes_with_readback(game, adapter)
        with instance.end_frame():
            pass
        validated = validate_live_readback(adapter, readback)
        destroy_scene_meshes(instance, actors)
        actors_destroyed = True
        _write_json_exclusive(
            args.run_dir.resolve() / "RESULT.json",
            {
                "schema": RESULT_SCHEMA,
                "status": "pass",
                "readiness_status": "packaged_71_mesh_readback_pass_gpu_f15_pending",
                "scene_id": validated["scene_id"],
                "entry_map": ENTRY_MAP,
                "nullrhi": True,
                "rendering_or_capture_called": False,
                "room_live_readback": readback,
                "qualification_claim": False,
                "formal_dataset_count": 0,
            },
        )
        return 0
    finally:
        if instance is not None:
            if actors and not actors_destroyed:
                try:
                    destroy_scene_meshes(instance, actors)
                except BaseException:
                    pass
            instance.close(force=True)


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=15)


def _runner(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    _require(not run_dir.exists(), f"no-clobber run directory exists: {run_dir}")
    launcher, binary, package = _packaged_paths(args.executable)
    _require(args.room_adapter.resolve().is_file(), "room adapter is missing")
    _require(args.spear_root.resolve().is_dir(), "SPEAR root is missing")
    _require(args.python.resolve().is_file(), "worker Python is missing")
    adapter = _load_json_object(args.room_adapter.resolve(), owner="room adapter")
    _adapter_paths(adapter)
    _assert_port_available(args.rpc_port)
    before = _exact_packaged_processes(binary)
    _require(not before, f"exact packaged executable already running: {before}")
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(args.python.resolve()),
        str(Path(__file__).resolve()),
        "--worker",
        "--run-dir",
        str(run_dir),
        "--room-adapter",
        str(args.room_adapter.resolve()),
        "--executable",
        str(launcher),
        "--spear-root",
        str(args.spear_root.resolve()),
        "--python",
        str(args.python.resolve()),
        "--rpc-port",
        str(args.rpc_port),
    ]
    _write_json_exclusive(
        run_dir / "PREPARED.json",
        {
            "schema": "avengine_packaged_imported_glb_room_probe_prepared_v1",
            "status": "prepared",
            "prepared_at_utc": _utc_now(),
            "no_clobber": True,
            "packaged_launcher": str(launcher),
            "packaged_inner_binary": str(binary),
            "packaged_pak": str(package),
            "room_adapter": str(args.room_adapter.resolve()),
            "entry_map": ENTRY_MAP,
            "rpc_port": args.rpc_port,
            "nullrhi": True,
            "rendering_or_capture_authorized": False,
            "qualification_claim": False,
            "formal_dataset_count": 0,
        },
    )
    (run_dir / "COMMAND.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    started = _utc_now()
    with (run_dir / "worker.log").open("x", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            start_new_session=True,
            text=True,
        )
        _write_json_exclusive(
            run_dir / "RUNNING.json",
            {
                "schema": "avengine_packaged_imported_glb_room_probe_running_v1",
                "status": "running",
                "started_at_utc": started,
                "worker_pid": process.pid,
                "process_group_id": process.pid,
                "rpc_port": args.rpc_port,
                "timeout_seconds": args.timeout_seconds,
            },
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)
            returncode = 124
        if returncode != 0:
            _terminate_process_group(process)
    after = _exact_packaged_processes(binary)
    result_path = run_dir / "RESULT.json"
    result: dict[str, Any] | None = None
    semantic_error = None
    if result_path.is_file():
        result = _load_json_object(result_path, owner="worker result")
        try:
            _require(
                result.get("schema") == RESULT_SCHEMA
                and result.get("status") == "pass"
                and result.get("readiness_status")
                == "packaged_71_mesh_readback_pass_gpu_f15_pending"
                and result.get("nullrhi") is True
                and result.get("rendering_or_capture_called") is False
                and result.get("qualification_claim") is False
                and result.get("formal_dataset_count") == 0,
                "worker result crossed the NullRHI claim boundary",
            )
            room = result.get("room_live_readback")
            _require(isinstance(room, Mapping), "worker result lacks room readback")
            validate_live_readback(adapter, room)
        except (RuntimeError, TypeError, ValueError) as error:
            semantic_error = f"{type(error).__name__}: {error}"
    status = (
        "pass"
        if returncode == 0
        and isinstance(result, Mapping)
        and result.get("status") == "pass"
        and semantic_error is None
        and not after
        else "fail"
    )
    exit_receipt = {
        "schema": EXIT_SCHEMA,
        "status": status,
        "started_at_utc": started,
        "ended_at_utc": _utc_now(),
        "worker_pid": process.pid,
        "worker_exit_code": returncode,
        "timed_out": timed_out,
        "result_exists": result_path.is_file(),
        "result_status": result.get("status") if isinstance(result, Mapping) else None,
        "semantic_error": semantic_error,
        "exact_packaged_processes_before": before,
        "exact_packaged_processes_after": after,
        "exact_packaged_process_exit_closed": not after,
        "nullrhi": True,
        "rendering_or_capture_called": False,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }
    if status == "pass":
        _require(isinstance(result, Mapping), "passing result is missing")
        validate_result_and_exit(
            adapter=adapter, result=result, exit_receipt=exit_receipt
        )
    _write_json_exclusive(run_dir / "EXIT.json", exit_receipt)
    return 0 if status == "pass" else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--room-adapter", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--spear-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--rpc-port", type=int, default=DEFAULT_RPC_PORT)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    _require(1024 <= args.rpc_port <= 65535, "RPC port outside [1024, 65535]")
    _require(args.timeout_seconds > 0, "timeout must be positive")
    return args


def main() -> int:
    args = _parse_args()
    try:
        return _worker(args) if args.worker else _runner(args)
    except (
        AssertionError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        if not args.worker and args.run_dir.exists():
            failure_path = args.run_dir / "RUNNER_ERROR.json"
            if not failure_path.exists():
                _write_json_exclusive(
                    failure_path,
                    {
                        "schema": "avengine_packaged_imported_glb_room_probe_error_v1",
                        "status": "fail",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "captured_at_utc": _utc_now(),
                        "qualification_claim": False,
                        "formal_dataset_count": 0,
                    },
                )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

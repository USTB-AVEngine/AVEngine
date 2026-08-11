#!/usr/bin/env python3
"""Fail-closed NullRHI packaged-object readback for Skokloster Castle.

The default mode is an outer receipt runner.  It creates a new evidence
directory, snapshots GPU compute processes, launches one worker in a unique
process group, and writes exclusive PREPARED/RUNNING/EXIT receipts.  The worker
launches the packaged SPEAR game exactly once and performs read-only RPC calls;
it never renders, captures, spawns, saves, cooks, or deletes content.
"""

from __future__ import annotations

import argparse
import json
import math
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

SCHEMA = "avengine_skokloster_packaged_object_readback_v1"
MAP_PATH = (
    "/Game/MyAssets/Audioset/Scenes/skokloster_castle/Maps/skokloster_castle_strict"
)
MESH_PATH = (
    "/Game/MyAssets/Audioset/Scenes/skokloster_castle/Imported/"
    "skokloster_castle_habitat_y_up.skokloster_castle_habitat_y_up"
)
MATERIAL_PATH = (
    "/Game/MyAssets/Audioset/Scenes/skokloster_castle/Imported/"
    "model_Material_u1_v1.model_Material_u1_v1"
)
ACTOR_TAG = "avengine_skokloster_castle_surface"
NAVIGATION_TAG = "habitat_navigation_authority"
EXPECTED_MINIMUM_CM = [-975.916259765625, 97.38520050048828, -39.00769805908203]
EXPECTED_MAXIMUM_CM = [856.9032592773438, 2559.83056640625, 683.4408569335938]
BOUNDS_TOLERANCE_CM = 0.25
IDENTITY_TOLERANCE = 1.0e-5
DEFAULT_RPC_PORT = 30173
DEFAULT_TIMEOUT_SECONDS = 900


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _struct_components(value: Any, names: Sequence[str]) -> list[float]:
    current = value
    expected = tuple(name.casefold() for name in names)
    for _ in range(3):
        if not isinstance(current, Mapping):
            break
        lowered = {str(key).casefold(): item for key, item in current.items()}
        if all(name in lowered for name in expected):
            result = [float(lowered[name]) for name in expected]
            _require(all(math.isfinite(item) for item in result), "non-finite struct")
            return result
        if "returnvalue" in lowered and isinstance(lowered["returnvalue"], Mapping):
            current = lowered["returnvalue"]
            continue
        if len(current) == 1:
            candidate = next(iter(current.values()))
            if isinstance(candidate, Mapping):
                current = candidate
                continue
        break
    raise RuntimeError(f"could not read Unreal struct {tuple(names)}: {value}")


def _positive_handle(value: Any, label: str) -> int:
    _require(not isinstance(value, bool), f"{label} returned a boolean")
    handle = int(value)
    _require(handle > 0, f"{label} returned invalid handle {handle}")
    return handle


def _component_mesh_handle(component: Any) -> tuple[int, str]:
    getter = getattr(component, "GetStaticMesh", None)
    if callable(getter):
        value = getter(as_handle=True)
        method = "UStaticMeshComponent.GetStaticMesh"
    else:
        value = component.get_property_value(property_name="StaticMesh", as_handle=True)
        method = "UStaticMeshComponent.StaticMesh_property"
    return _positive_handle(value, "StaticMesh readback"), method


def _component_material_handle(component: Any, static_mesh: Any) -> tuple[int, str]:
    getter = getattr(component, "GetMaterial", None)
    if callable(getter):
        value = getter(ElementIndex=0, as_handle=True)
        method = "UMeshComponent.GetMaterial"
    else:
        value = static_mesh.get_property_value(
            property_name="StaticMaterials[0].MaterialInterface", as_handle=True
        )
        method = "UStaticMesh.StaticMaterials[0].MaterialInterface_property"
    return _positive_handle(value, "material readback"), method


def _material_slot_count(component: Any, static_mesh: Any) -> tuple[int, str]:
    getter = getattr(component, "GetNumMaterials", None)
    if callable(getter):
        return int(getter()), "UMeshComponent.GetNumMaterials"
    static_materials = static_mesh.get_property_value(
        property_name="StaticMaterials", as_value=True
    )
    _require(
        isinstance(static_materials, Sequence)
        and not isinstance(static_materials, (str, bytes)),
        "UStaticMesh.StaticMaterials is not a sequence",
    )
    return len(static_materials), "UStaticMesh.StaticMaterials_property"


def _bounds_from_value(value: Any, source: str) -> dict[str, list[float]]:
    _require(
        isinstance(value, Mapping), f"could not read bounds from {source}: {value}"
    )
    lowered = {str(key).casefold(): item for key, item in value.items()}
    if "returnvalue" in lowered and isinstance(lowered["returnvalue"], Mapping):
        lowered = {
            str(key).casefold(): item for key, item in lowered["returnvalue"].items()
        }
    origin = _struct_components(lowered.get("origin"), ("x", "y", "z"))
    extent = _struct_components(lowered.get("boxextent"), ("x", "y", "z"))
    _require(all(item > 0.0 for item in extent), f"{source} bounds degenerate")
    return {
        "origin_cm": origin,
        "extent_cm": extent,
        "minimum_cm": [origin[i] - extent[i] for i in range(3)],
        "maximum_cm": [origin[i] + extent[i] for i in range(3)],
    }


def _actor_bounds(
    actor: Any, component: Any | None = None, static_mesh: Any | None = None
) -> tuple[dict[str, list[float]], str]:
    getter = getattr(actor, "GetActorBounds", None)
    if callable(getter):
        value = getter(
            bOnlyCollidingComponents=False,
            bIncludeFromChildActors=True,
            as_dict=True,
        )
        return (
            _bounds_from_value(value, "AActor.GetActorBounds"),
            "AActor.GetActorBounds",
        )
    if component is not None:
        value = component.get_property_value(property_name="Bounds", as_value=True)
        return (
            _bounds_from_value(value, "USceneComponent.Bounds"),
            "USceneComponent.Bounds_property",
        )
    _require(static_mesh is not None, "no live bounds fallback object is available")
    value = static_mesh.get_property_value(
        property_name="ExtendedBounds", as_value=True
    )
    return (
        _bounds_from_value(value, "UStaticMesh.ExtendedBounds"),
        "UStaticMesh.ExtendedBounds_property",
    )


def _actor_transform(actor: Any) -> dict[str, Any]:
    members = {
        "location_cm": (
            "K2_GetActorLocation",
            "RootComponent.RelativeLocation",
            ("x", "y", "z"),
        ),
        "rotation_roll_pitch_yaw_degrees": (
            "K2_GetActorRotation",
            "RootComponent.RelativeRotation",
            ("roll", "pitch", "yaw"),
        ),
        "scale": (
            "GetActorScale3D",
            "RootComponent.RelativeScale3D",
            ("x", "y", "z"),
        ),
    }
    result: dict[str, Any] = {}
    methods = {}
    for output_name, (function_name, property_name, components) in members.items():
        getter = getattr(actor, function_name, None)
        if callable(getter):
            raw = getter(as_dict=True)
            methods[output_name] = f"AActor.{function_name}"
        else:
            raw = actor.get_property_value(property_name=property_name, as_value=True)
            methods[output_name] = f"AActor.{property_name}_property"
        result[output_name] = _struct_components(raw, components)
    result["readback_methods"] = methods
    return result


def _maximum_error(observed: Sequence[float], expected: Sequence[float]) -> float:
    return max(
        abs(float(a) - float(b)) for a, b in zip(observed, expected, strict=True)
    )


def _gpu_snapshot() -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(query, capture_output=True, text=True, check=False)
    rows = []
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",", 3)]
            if len(fields) == 4:
                rows.append(
                    {
                        "gpu_uuid": fields[0],
                        "pid": int(fields[1]),
                        "process_name": fields[2],
                        "used_memory_mib": int(fields[3]),
                    }
                )
    return {
        "captured_at_utc": _utc_now(),
        "command": query,
        "returncode": completed.returncode,
        "compute_processes": rows,
        "stderr": completed.stderr.strip(),
    }


def _assert_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def _exact_packaged_processes(executable: Path) -> list[dict[str, Any]]:
    """Return only processes whose /proc exe is this archive's inner binary."""

    inner_binary = (
        executable.resolve().parent / "SpearSim/Binaries/Linux/SpearSim"
    ).resolve()
    records = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return records
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdecimal():
            continue
        try:
            observed_executable = (process_dir / "exe").resolve(strict=True)
        except (FileNotFoundError, PermissionError):
            continue
        if observed_executable != inner_binary:
            continue
        try:
            command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ")
            command_text = command.decode("utf-8", errors="replace").strip()
        except (FileNotFoundError, PermissionError):
            command_text = ""
        records.append(
            {
                "pid": int(process_dir.name),
                "executable": str(observed_executable),
                "command": command_text,
            }
        )
    return sorted(records, key=lambda value: value["pid"])


def _load_handle(game: Any, uclass: str, path: str) -> int:
    return _positive_handle(
        game.unreal_service.load_object(uclass=uclass, name=path, as_handle=True),
        f"load_object {path}",
    )


def _worker(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    result_path = run_dir / "RESULT.json"
    spear_root = args.spear_root.resolve()
    sys.path.insert(0, str(spear_root / "python"))
    import spear

    config = spear.get_config(user_config_files=[])
    config.defrost()
    config.SPEAR.LAUNCH_MODE = "game"
    config.SPEAR.INSTANCE.GAME_EXECUTABLE = str(args.executable.resolve())
    config.SPEAR.INSTANCE.INITIALIZE_CLIENT_MAX_TIME_SECONDS = 600.0
    config.SPEAR.INSTANCE.CLIENT_INTERNAL_TIMEOUT_SECONDS = 120.0
    config.SPEAR.INSTANCE.TEMP_DIR = str(run_dir / f"spear_instance_{args.rpc_port}")
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.log = (
        f"SpearSim_skokloster_packaged_probe_{args.rpc_port}.log"
    )
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.nullrhi = None
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.unattended = None
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.nop4 = None
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.nosplash = None
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = MAP_PATH
    config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT = args.rpc_port
    config.SP_CORE.SHARED_MEMORY_INITIAL_UNIQUE_ID = args.rpc_port * 10000
    config.freeze()
    spear.configure_system(config=config)

    instance = None
    try:
        instance = spear.Instance(config=config)
        game = instance.get_game()
        with instance.begin_frame():
            mesh_actors = game.unreal_service.find_actors_by_class(
                uclass="AStaticMeshActor"
            )
            actors = [
                actor for actor in mesh_actors if bool(actor.ActorHasTag(Tag=ACTOR_TAG))
            ]
            _require(len(actors) == 1, f"tagged actor count {len(actors)} != 1")
            actor = actors[0]
            _require(
                bool(actor.ActorHasTag(Tag=NAVIGATION_TAG)),
                "saved actor lacks navigation-authority tag",
            )
            component = game.unreal_service.get_component_by_class(
                actor=actor, uclass="UStaticMeshComponent"
            )
            expected_mesh = _load_handle(game, "UStaticMesh", MESH_PATH)
            static_mesh = game.get_unreal_object(uobject=expected_mesh)
            observed_mesh, mesh_method = _component_mesh_handle(component)
            _require(observed_mesh == expected_mesh, "component mesh object mismatch")
            material_count, material_count_method = _material_slot_count(
                component, static_mesh
            )
            _require(material_count == 1, f"material slot count {material_count} != 1")
            expected_material = _load_handle(game, "UMaterialInterface", MATERIAL_PATH)
            observed_material, material_method = _component_material_handle(
                component, static_mesh
            )
            _require(
                observed_material == expected_material,
                "component material object mismatch",
            )
            transform = _actor_transform(actor)
            location = transform["location_cm"]
            rotation = transform["rotation_roll_pitch_yaw_degrees"]
            scale = transform["scale"]
            _require(
                _maximum_error(location, [0.0, 0.0, 0.0]) <= IDENTITY_TOLERANCE,
                f"actor location is not identity: {location}",
            )
            _require(
                _maximum_error(rotation, [0.0, 0.0, 0.0]) <= IDENTITY_TOLERANCE,
                f"actor rotation is not identity: {rotation}",
            )
            _require(
                _maximum_error(scale, [1.0, 1.0, 1.0]) <= IDENTITY_TOLERANCE,
                f"actor scale is not identity: {scale}",
            )
            bounds, bounds_method = _actor_bounds(actor, component, static_mesh)
            minimum_error = _maximum_error(bounds["minimum_cm"], EXPECTED_MINIMUM_CM)
            maximum_error = _maximum_error(bounds["maximum_cm"], EXPECTED_MAXIMUM_CM)
            _require(
                max(minimum_error, maximum_error) <= BOUNDS_TOLERANCE_CM,
                "packaged actor bounds differ from fresh-editor bounds",
            )
        with instance.end_frame():
            pass
        result = {
            "schema": SCHEMA,
            "status": "pass",
            "readiness_status": (
                "packaged_room_object_readback_pass_visual_sparse_pending"
            ),
            "launch": {
                "mode": "packaged_game",
                "nullrhi": True,
                "map_path": MAP_PATH,
                "rpc_port": args.rpc_port,
                "rendering_or_capture_called": False,
            },
            "saved_map_actor": {
                "actor_tag": ACTOR_TAG,
                "navigation_tag": NAVIGATION_TAG,
                "matching_actor_count": 1,
                "static_mesh_component_count": 1,
                "location_cm": location,
                "rotation_roll_pitch_yaw_degrees": rotation,
                "scale": scale,
                "identity_transform": True,
                "transform_readback_methods": transform["readback_methods"],
            },
            "static_mesh": {
                "object_path": MESH_PATH,
                "expected_object_handle": expected_mesh,
                "observed_component_handle": observed_mesh,
                "readback_method": mesh_method,
                "handle_match": True,
                "bounds": bounds,
                "bounds_readback_method": bounds_method,
                "bounds_tolerance_cm": BOUNDS_TOLERANCE_CM,
                "minimum_maximum_absolute_error_cm": minimum_error,
                "maximum_maximum_absolute_error_cm": maximum_error,
            },
            "material": {
                "object_path": MATERIAL_PATH,
                "slot_count": material_count,
                "slot_count_readback_method": material_count_method,
                "expected_object_handle": expected_material,
                "observed_component_handle": observed_material,
                "readback_method": material_method,
                "handle_match": True,
            },
            "qualification_claim": False,
            "formal_dataset_count": 0,
            "visual_sparse_capture_pending": True,
        }
        _write_json_exclusive(result_path, result)
        return 0
    finally:
        if instance is not None:
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
    _require(args.executable.resolve().is_file(), "packaged executable is absent")
    _require(args.spear_root.resolve().is_dir(), "SPEAR root is absent")
    _assert_port_available(args.rpc_port)
    before_processes = _exact_packaged_processes(args.executable)
    _require(
        not before_processes,
        f"exact packaged executable already running: {before_processes}",
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    command = [
        str(args.python.resolve()),
        str(script),
        "--worker",
        "--run-dir",
        str(run_dir),
        "--executable",
        str(args.executable.resolve()),
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
            "schema": "avengine_skokloster_packaged_probe_prepared_v1",
            "state": "prepared",
            "prepared_at_utc": _utc_now(),
            "no_clobber": True,
            "packaged_executable": str(args.executable.resolve()),
            "map_path": MAP_PATH,
            "mesh_object_path": MESH_PATH,
            "material_object_path": MATERIAL_PATH,
            "actor_tag": ACTOR_TAG,
            "rpc_port": args.rpc_port,
            "nullrhi": True,
            "rendering_or_capture_authorized": False,
            "formal_dataset_count": 0,
        },
    )
    _write_json_exclusive(run_dir / "GPU_BEFORE.json", _gpu_snapshot())
    (run_dir / "COMMAND.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    started = _utc_now()
    log_path = run_dir / "worker.log"
    with log_path.open("x", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        _write_json_exclusive(
            run_dir / "RUNNING.json",
            {
                "schema": "avengine_skokloster_packaged_probe_running_v1",
                "state": "running",
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
    _write_json_exclusive(run_dir / "GPU_AFTER.json", _gpu_snapshot())
    after_processes = _exact_packaged_processes(args.executable)
    result_path = run_dir / "RESULT.json"
    result = None
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    status = (
        "pass"
        if returncode == 0
        and isinstance(result, Mapping)
        and result.get("status") == "pass"
        and not after_processes
        else "fail"
    )
    _write_json_exclusive(
        run_dir / "EXIT.json",
        {
            "schema": "avengine_skokloster_packaged_probe_exit_v1",
            "state": status,
            "status": status,
            "started_at_utc": started,
            "ended_at_utc": _utc_now(),
            "worker_pid": process.pid,
            "worker_exit_code": returncode,
            "timed_out": timed_out,
            "result_exists": result_path.is_file(),
            "result_status": (
                result.get("status") if isinstance(result, Mapping) else None
            ),
            "exact_packaged_processes_before": before_processes,
            "exact_packaged_processes_after": after_processes,
            "exact_packaged_process_exit_closed": not after_processes,
            "nullrhi": True,
            "rendering_or_capture_called": False,
            "qualification_claim": False,
            "formal_dataset_count": 0,
        },
    )
    return 0 if status == "pass" else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-dir", type=Path, required=True)
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
            failure = args.run_dir / "RUNNER_ERROR.json"
            if not failure.exists():
                _write_json_exclusive(
                    failure,
                    {
                        "schema": "avengine_skokloster_packaged_probe_runner_error_v1",
                        "status": "fail",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "captured_at_utc": _utc_now(),
                    },
                )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

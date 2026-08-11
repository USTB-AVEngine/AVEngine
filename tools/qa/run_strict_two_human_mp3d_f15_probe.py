#!/usr/bin/env python3
"""Prepare and launch one MP3D strict-two-human diagnostic f15 probe.

The launcher is deliberately fail closed.  A request binds the current Git
commit and the accepted CPU evidence, then exactly one real launch may create
immutable RUNNING and FINAL receipts.  Dry runs do not consume the attempt.
No path in this workflow authorizes a full 75-frame GPU capture or a formal
dataset increment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

REQUEST_SCHEMA = "avengine_mp3d_strict_two_human_f15_launch_request_v1"
RECEIPT_SCHEMA = "avengine_mp3d_strict_two_human_f15_launch_receipt_v1"
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
ATTEMPT_POLICY = {
    "attempt_index": 1,
    "maximum_attempts_for_candidate": 1,
    "retry_same_candidate_forbidden": True,
    "failure_disposition": "reject_candidate_without_same_candidate_retry",
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


def _file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "byte_size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _validate_file_record(record: Mapping[str, Any], *, owner: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    _require(path.is_file(), f"{owner} is missing: {path}")
    observed = _file_record(path)
    _require(observed == dict(record), f"{owner} artifact binding drift")
    return path


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


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
    gpus = _nvidia_csv(
        "gpu", "index,uuid,name,memory.used,utilization.gpu"
    )
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
        item
        for item in snapshot.get("gpus", [])
        if item.get("physical_index") == 1
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
        and float(navigation.get("horizontal_source_separation_m", 0.0))
        >= 1.3
        and pair_gate.get("clearance_gate_passed") is True
        and pair_gate.get("separation_gate_passed") is True,
        "adult two-human navigation gate drift",
    )
    selected = navigation.get("selected_positions", {})
    _require(
        float(selected.get("source1", {}).get("fresh_clearance_m", 0.0)) >= 0.5
        and float(selected.get("source2", {}).get("fresh_clearance_m", 0.0))
        >= 0.5,
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
        and coverage.get("triangle_coverage", {}).get("triangle_count")
        == 3_016_249
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


def _capture_argv(request: Mapping[str, Any]) -> list[str]:
    return [
        str(request["capture_python"]),
        str(request["capture_script"]),
        "--suite-plan",
        str(request["suite_plan"]),
        "--scenario-id",
        EPISODE_ID,
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
    _require(not capture_output.exists(), "diagnostic f15 capture output already exists")
    paths = _artifact_paths(atom_root)
    _validate_cpu_evidence(paths)
    capture_script = REPOSITORY / "tools/qa/capture_spear_imported_glb_strict_two_human_episode.py"
    _require(capture_python.is_file(), "authoritative SPEAR Python is missing")
    _require(capture_script.is_file(), "MP3D capture runner is missing")
    _require(spear_root.is_dir(), "SPEAR root is missing")
    _require(1024 <= rpc_port <= 65535, "RPC port is out of range")
    artifact_records = {
        name: _file_record(path) for name, path in paths.items()
    }
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


def archive_preparation_failure(*, atom_root: Path, error: str) -> Path:
    """Preserve a failed pre-GPU request without consuming the real attempt."""

    atom_root = atom_root.resolve()
    _require(
        atom_root
        == REPOSITORY / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1",
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
        and Path(request["room_adapter"]).resolve()
        == expected_paths["room_adapter"],
        "capture suite/room path drift",
    )
    _require(
        _is_authoritative_capture_python(Path(request["capture_python"]))
        and Path(request["capture_script"]).resolve()
        == repo_root
        / "tools/qa/capture_spear_imported_glb_strict_two_human_episode.py"
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


def _compile_and_validate_visibility(
    request: Mapping[str, Any], capture_root: Path
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
        int(target["visible_pixels"]) > 0
        and int(distractor["visible_pixels"]) > 0,
        "f15 target or distractor has no modal-visible pixels",
    )
    truth_path = capture_root / "pixel_visibility_truth.json"
    _write_json_exclusive(truth_path, truth)
    return {
        "status": "pass",
        "target_instance_id": "source1",
        "distractor_instance_id": "source2",
        "target_visible_fraction": target_fraction,
        "distractor_visible_fraction": distractor_fraction,
        "target_visible_pixels": int(target["visible_pixels"]),
        "distractor_visible_pixels": int(distractor["visible_pixels"]),
        "pixel_visibility_truth": _file_record(truth_path),
    }


def _validate_capture(request: Mapping[str, Any]) -> dict[str, Any]:
    capture_root = Path(request["capture_output"])
    manifest = _load(capture_root / "manifest.json")
    _require(
        manifest.get("schema")
        == "avengine_spear_imported_glb_strict_two_human_capture_v1"
        and manifest.get("status") == "capture_pass_review_pending"
        and manifest.get("scenario_id") == EPISODE_ID,
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
        room.get("scene_id") == SCENE_ID
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
        and readback.get("unique_component_mesh_handle_count")
        == EXPECTED_MESH_COUNT
        and isinstance(meshes, list)
        and [item.get("object_path") for item in meshes]
        == adapter["static_mesh_object_paths"],
        "live cooked UStaticMesh path/handle closure drift",
    )
    visibility = _compile_and_validate_visibility(request, capture_root)
    return {
        "status": "pass_diagnostic_f15_review_ready",
        "capture_manifest": _file_record(capture_root / "manifest.json"),
        "room_live_readback": _file_record(
            capture_root / "room_live_readback.json"
        ),
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
    launch = subparsers.add_parser("launch")
    launch.add_argument("--request", required=True, type=Path)
    launch.add_argument("--dry-run", action="store_true")
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
    return run(args.request, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

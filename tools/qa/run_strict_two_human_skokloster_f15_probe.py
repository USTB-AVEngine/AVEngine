#!/usr/bin/env python3
"""Prepare and run one fail-closed Skokloster strict-two-human f15 probe.

Preparation binds a clean repository commit, the isolated Development archive,
the exact packaged map, and the accepted CPU acoustic evidence.  A dry run may
inspect the request without consuming the attempt.  A real run requires both
an explicit launch flag and a terminal MP3D revision-v2 receipt, then captures
exactly frame 15 on physical GPU1.  Child stdout/stderr, ordered launcher phase
markers, exit status, and a launcher traceback on failure are immutable.

This tool never authorizes a full 75-frame capture and never increments the
formal dataset denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import struct
import subprocess
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
ATOM_DIRECTORY = "tmp/lead_a_skokloster_strict_two_human_v1"
ATTEMPT_DIRECTORY = "diagnostic_f15_launch_attempt_01"
CAPTURE_DIRECTORY = "diagnostic_f15_capture_attempt_01"
EPISODE_ID = "skokloster_castle_male_female_static_0001"
SCENE_ID = "skokloster-castle"
FRAME_INDEX = 15
FRAME_COUNT = 75
HEIGHT = 720
WIDTH = 1280
FPS = 15
RPC_PORT = 39831
GPU1_UUID = "GPU-6d3e273e-58c6-2a5b-480a-4816fef6c581"
PACKAGED_MAP = (
    "/Game/MyAssets/Audioset/Scenes/skokloster_castle/Maps/skokloster_castle_strict"
)
ARCHIVE_ROOT = Path(
    "/data/jzy/code/AVEngine/external/SPEAR/cpp/unreal_projects/"
    "SpearSim/Standalone-Skokloster-Development"
)
PACKAGED_EXECUTABLE = ARCHIVE_ROOT / "Linux/SpearSim.sh"
PACKAGED_BINARY = ARCHIVE_ROOT / "Linux/SpearSim/Binaries/Linux/SpearSim"
PACKAGED_PAK = ARCHIVE_ROOT / "Linux/SpearSim/Content/Paks/SpearSim-Linux.pak"
CAPTURE_PYTHON_LOGICAL = REPOSITORY / ".venv/bin/python"
SPEAR_ROOT = Path("/data/jzy/code/SPEAR-lead-b")
MP3D_V2_TERMINAL_RECEIPT = (
    REPOSITORY
    / "tmp/lead_a_mp3d_strict_two_human_room_atom_v1"
    / "diagnostic_f15_revision_v2_launch_attempt_01"
    / "final_receipt.json"
)

REQUEST_SCHEMA = "avengine_skokloster_strict_two_human_f15_launch_request_v1"
RECEIPT_SCHEMA = "avengine_skokloster_strict_two_human_f15_launch_receipt_v1"
PHASE_SCHEMA = "avengine_skokloster_strict_two_human_f15_launch_phase_v1"
MP3D_V2_RECEIPT_SCHEMA = "avengine_mp3d_strict_two_human_f15_launch_receipt_v2"

CAMERA_LISTENER_M = [5.4500002861, 1.7017275691, 9.5499992371]
CAMERA_YAW_DEG = 89.9749414
TARGET_VISIBLE_FRACTION_MINIMUM = 0.8
DISTRACTOR_VISIBLE_FRACTION_MINIMUM = 0.5
VISIBLE_PIXEL_COUNT_MINIMUM = 5000
BBOX_EDGE_MARGIN_PX_MINIMUM = 1

ATTEMPT_POLICY = {
    "attempt_index": 1,
    "maximum_attempts_for_candidate": 1,
    "retry_same_candidate_forbidden": True,
    "failure_disposition": "reject_candidate_without_same_candidate_retry",
}
REQUIRED_CAPTURE_ARTIFACT_ROLES = {
    "rgb_frames",
    "native_rgb_visual_only",
    "native_rgb_binaural",
    "metric_depth",
    "normal_object_ids",
    "pixel_masks",
    "pixel_visibility_truth",
    "runtime_readbacks",
    "runtime_asset_readbacks",
    "object_id_descriptors",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    _require(path.is_file(), f"bound file is missing: {path}")
    return {
        "path": str(path),
        "byte_size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_file_record(record: Mapping[str, Any], *, owner: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    _require(path.is_file(), f"{owner} is missing: {path}")
    _require(_file_record(path) == dict(record), f"{owner} artifact binding drift")
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


def _git_status_porcelain(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def _require_clean_head(repo_root: Path) -> str:
    status = _git_status_porcelain(repo_root)
    _require(not status.strip(), "repository worktree is not clean")
    head = _git_head(repo_root)
    _require(len(head) == 40, "repository HEAD is not a full commit id")
    return head


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


def _artifact_paths(atom_root: Path) -> dict[str, Path]:
    preflight = atom_root / "cpu_preflight_v4"
    rir = atom_root / "exact_rir_cache_v3"
    binaural = atom_root / "binaural_v4"
    sample_id = f"{EPISODE_ID}__v00"
    audio_root = binaural / "audio/binaural"
    return {
        "preflight": preflight / "preflight.json",
        "suite_plan": preflight / "suite_execution_plan.json",
        "execution_plan": preflight / "execution_plan.json",
        "rir_plan": preflight / "rir_job_plan.json",
        "sensor_rig": preflight / "sensor_rig_trajectory.json",
        "audio_program_binding": preflight / "audio_program_binding.json",
        "rir_receipt": rir / "receipt.json",
        "rir_index": rir / "index.json",
        "rir_shard": rir / "shards/shard_000000.npz",
        "binaural_delivery": binaural / "delivery.json",
        "binaural_episodes": binaural / "episodes.json",
        "binaural_samples": binaural / "samples.json",
        "binaural_timing": binaural / "timing.json",
        "binaural_mixture": audio_root / f"{sample_id}.wav",
        "binaural_source1": audio_root / "stems/source1" / f"{sample_id}.wav",
        "binaural_source2": audio_root / "stems/source2" / f"{sample_id}.wav",
        "runtime_profile": REPOSITORY
        / "examples/m3/skokloster_castle/skokloster_room_runtime_profile.json",
        "editor_cook_plan": REPOSITORY
        / "examples/m3/skokloster_castle/editor_import_cook_plan.json",
    }


def _package_paths() -> dict[str, Path]:
    return {
        "archive_launcher": PACKAGED_EXECUTABLE,
        "archive_binary": PACKAGED_BINARY,
        "archive_pak": PACKAGED_PAK,
    }


def _source_paths() -> dict[str, Path]:
    return {
        "launcher": Path(__file__).resolve(),
        "capture_wrapper": REPOSITORY
        / "tools/qa/capture_skokloster_strict_two_human_episode.py",
        "base_capture_runner": REPOSITORY
        / "tools/qa/capture_spear_native_pixel_episode.py",
    }


def _close_vector(
    observed: object, expected: Sequence[float], *, tolerance: float = 1.0e-6
) -> bool:
    return (
        isinstance(observed, list)
        and len(observed) == len(expected)
        and all(
            math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0)
            for left, right in zip(observed, expected, strict=True)
        )
    )


def _riff_payload(path: Path) -> tuple[dict[str, int], bytes]:
    raw = path.read_bytes()
    _require(
        len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE",
        f"invalid WAV: {path}",
    )
    offset = 12
    fmt: bytes | None = None
    payload: bytes | None = None
    while offset + 8 <= len(raw):
        chunk_id = raw[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", raw, offset + 4)[0]
        start = offset + 8
        end = start + chunk_size
        _require(end <= len(raw), f"truncated WAV chunk: {path}")
        if chunk_id == b"fmt ":
            fmt = raw[start:end]
        elif chunk_id == b"data":
            payload = raw[start:end]
        offset = end + (chunk_size & 1)
    _require(fmt is not None and len(fmt) >= 16, f"WAV fmt chunk is missing: {path}")
    _require(payload is not None, f"WAV data chunk is missing: {path}")
    audio_format, channels, sample_rate, _, block_align, bits = struct.unpack_from(
        "<HHIIHH", fmt, 0
    )
    if audio_format == 0xFFFE and len(fmt) >= 26:
        audio_format = struct.unpack_from("<H", fmt, 24)[0]
    _require(
        block_align > 0 and len(payload) % block_align == 0,
        f"WAV block alignment drift: {path}",
    )
    return (
        {
            "audio_format": audio_format,
            "channel_count": channels,
            "sample_rate_hz": sample_rate,
            "sample_count": len(payload) // block_align,
            "bits_per_sample": bits,
            "block_align": block_align,
        },
        payload,
    )


def _payload_nonzero_count(contract: Mapping[str, int], payload: bytes) -> int:
    audio_format = int(contract["audio_format"])
    bits = int(contract["bits_per_sample"])
    if audio_format == 3 and bits == 32:
        values = (item[0] for item in struct.iter_unpack("<f", payload))
        count = 0
        for value in values:
            _require(math.isfinite(value), "WAV contains a non-finite float sample")
            count += value != 0.0
        return count
    if audio_format == 1 and bits in {16, 32}:
        code = "<h" if bits == 16 else "<i"
        return sum(item[0] != 0 for item in struct.iter_unpack(code, payload))
    raise RuntimeError(
        f"unsupported WAV encoding for exact-zero audit: format={audio_format} bits={bits}"
    )


def _validate_audio_files(paths: Mapping[str, Path]) -> dict[str, Any]:
    contracts: dict[str, dict[str, int]] = {}
    payloads: dict[str, bytes] = {}
    nonzero: dict[str, int] = {}
    for name in ("binaural_mixture", "binaural_source1", "binaural_source2"):
        contract, payload = _riff_payload(paths[name])
        _require(
            contract["channel_count"] == 2
            and contract["sample_rate_hz"] == 16_000
            and contract["sample_count"] == 80_000,
            f"{name} is not 2ch/16kHz/80000 samples",
        )
        contracts[name] = contract
        payloads[name] = payload
        nonzero[name] = _payload_nonzero_count(contract, payload)
    _require(nonzero["binaural_source1"] > 0, "source1 binaural stem is silent")
    _require(
        nonzero["binaural_source2"] == 0, "source2 binaural stem is not exactly zero"
    )
    _require(
        payloads["binaural_mixture"] == payloads["binaural_source1"],
        "mixture payload is not exactly the active source1 stem",
    )
    return {
        "status": "pass",
        "contracts": contracts,
        "nonzero_value_count": nonzero,
        "source2_exact_zero": True,
        "mixture_exact_source1": True,
    }


def _validate_cpu_evidence(paths: Mapping[str, Path]) -> dict[str, Any]:
    values = {
        name: _load(path) for name, path in paths.items() if path.suffix == ".json"
    }
    preflight = values["preflight"]
    _require(
        preflight.get("schema")
        == "avengine_skokloster_strict_two_human_cpu_preflight_v1"
        and preflight.get("status") == "cpu_plan_pass_gpu_sparse_pending"
        and preflight.get("attempt_id") == "cpu_preflight_v4"
        and preflight.get("episode_id") == EPISODE_ID
        and _close_vector(preflight.get("camera_listener_habitat_m"), CAMERA_LISTENER_M)
        and math.isclose(
            float(preflight.get("camera_habitat_yaw_deg", -1.0)),
            CAMERA_YAW_DEG,
            abs_tol=1.0e-6,
            rel_tol=0.0,
        )
        and preflight.get("gpu_capture_authorized") is False
        and preflight.get("qualification_claim") is False
        and preflight.get("formal_dataset_count") == 0,
        "Skokloster CPU preflight v4 drift",
    )

    suite = values["suite_plan"]
    scenarios = suite.get("scenarios")
    _require(
        suite.get("schema") == "avengine_optional_spear_skokloster_suite_v1"
        and suite.get("native_map") == PACKAGED_MAP
        and suite.get("packaged_executable") == str(PACKAGED_EXECUTABLE)
        and isinstance(scenarios, list)
        and len(scenarios) == 1
        and suite.get("qualification_claim") is False
        and suite.get("formal_dataset_count") == 0,
        "Skokloster suite/archive/map binding drift",
    )
    scenario = scenarios[0]
    plan = scenario.get("plan", {})
    frames = plan.get("frames")
    camera = plan.get("camera", {})
    actors = plan.get("actors")
    _require(
        scenario.get("scenario_id") == EPISODE_ID
        and scenario.get("native_scene", {}).get("map") == PACKAGED_MAP
        and scenario.get("render", {}).get("frame_count") == FRAME_COUNT
        and scenario.get("render", {}).get("frame_rate_hz") == FPS
        and scenario.get("render", {}).get("width") == WIDTH
        and scenario.get("render", {}).get("height") == HEIGHT
        and isinstance(actors, list)
        and [actor.get("actor_id") for actor in actors]
        == ["source1_actor", "source2_actor"]
        and isinstance(frames, list)
        and len(frames) == FRAME_COUNT
        and _close_vector(camera.get("habitat_position_m"), CAMERA_LISTENER_M)
        and math.isclose(
            float(camera.get("habitat_yaw_deg", -1.0)),
            CAMERA_YAW_DEG,
            abs_tol=1.0e-6,
            rel_tol=0.0,
        ),
        "Skokloster 75-frame static visual plan drift",
    )
    _require(
        [int(frame.get("frame_index", -1)) for frame in frames] == list(range(75))
        and all(
            _close_vector(
                frame.get("camera_state", {}).get("habitat_position_m"),
                CAMERA_LISTENER_M,
            )
            for frame in frames
        ),
        "Skokloster per-frame camera/listener drift",
    )

    rir_plan = values["rir_plan"]
    jobs = rir_plan.get("jobs")
    uses = [use for job in jobs or [] for use in job.get("uses", [])]
    _require(
        rir_plan.get("schema") == "avengine_room_rir_job_plan_v2"
        and rir_plan.get("unique_rir_job_count") == 2
        and rir_plan.get("requested_pair_state_count") == 150
        and rir_plan.get("cache_reuse_count") == 148
        and _close_vector(rir_plan.get("listener_position_m"), CAMERA_LISTENER_M)
        and isinstance(jobs, list)
        and len(jobs) == 2
        and [len(job.get("uses", [])) for job in jobs] == [75, 75]
        and len(uses) == 150
        and rir_plan.get("qualification_claim") is False
        and rir_plan.get("formal_dataset_count") == 0,
        "Skokloster exact two-RIR plan drift",
    )
    receipt = values["rir_receipt"]
    index = values["rir_index"]
    _require(
        receipt.get("schema") == "avengine_rlr_rir_cache_receipt_v1"
        and receipt.get("status") == "pass"
        and receipt.get("compute_device") == "CPU"
        and receipt.get("layout_type") == "binaural"
        and receipt.get("sample_rate_hz") == 16_000
        and receipt.get("selected_job_count") == 2
        and receipt.get("full_plan_job_count") == 2
        and receipt.get("full_plan_complete") is True
        and receipt.get("retained_payload_hash_verified") is True
        and receipt.get("qualification_claim") is False,
        "Skokloster exact RIR v3 receipt drift",
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
        and isinstance(index.get("entries"), list)
        and len(index["entries"]) == 2,
        "Skokloster exact RIR v3 index drift",
    )

    audio_binding = values["audio_program_binding"]
    source1 = audio_binding.get("source1", {})
    source2 = audio_binding.get("source2", {})
    _require(
        audio_binding.get("schema")
        == "avengine_skokloster_strict_audio_program_binding_v1"
        and source1.get("role") == "target"
        and source1.get("start_sample") == 7467
        and source1.get("end_sample_exclusive") == 33093
        and source1.get("source_start_sample") == 0
        and source1.get("source_end_sample_exclusive") == 25626
        and math.isclose(float(source1.get("linear_gain", -1.0)), 0.18)
        and source1.get("fade_samples") == 80
        and source2.get("role") == "distractor"
        and source2.get("event_count") == 0
        and audio_binding.get("qualification_claim") is False
        and audio_binding.get("formal_dataset_count") == 0,
        "Skokloster target/silent-distractor audio program drift",
    )
    delivery = values["binaural_delivery"]
    _require(
        delivery.get("status") == "pass"
        and delivery.get("episode_count") == 1
        and delivery.get("sample_count") == 1
        and delivery.get("both_sources_active") is False
        and delivery.get("source_activity_contract")
        == "m6_audio_program_event_windows_v1"
        and delivery.get("sensor_rig_rir_alignment", {}).get("checked_use_count") == 150
        and delivery.get("qualification_claim") is False,
        "Skokloster binaural v4 delivery drift",
    )
    samples = values["binaural_samples"]
    sample_rows = samples.get("samples")
    _require(
        samples.get("status") == "pass"
        and isinstance(sample_rows, list)
        and len(sample_rows) == 1,
        "Skokloster binaural v4 sample index drift",
    )
    row = sample_rows[0]
    audio = row.get("audio", {})
    activity = row.get("source_activity_summary", {})
    stems = audio.get("stems", {})
    _require(
        row.get("episode_id") == EPISODE_ID
        and row.get("both_sources_active") is False
        and audio.get("channel_count") == 2
        and audio.get("sample_rate_hz") == 16_000
        and audio.get("sample_count") == 80_000
        and activity.get("active_source_slots") == ["source1"]
        and activity.get("silent_source_slots") == ["source2"]
        and activity.get("active_sample_count_by_source_slot")
        == {"source1": 25626, "source2": 0}
        and float(stems.get("source1", {}).get("peak_absolute", 0.0)) > 0.0
        and float(stems.get("source2", {}).get("peak_absolute", -1.0)) == 0.0,
        "Skokloster rendered source-activity closure drift",
    )
    runtime = values["runtime_profile"]
    visual = runtime.get("visual", {})
    readiness = runtime.get("readiness", {})
    _require(
        runtime.get("schema") == "avengine_skokloster_imported_room_runtime_profile_v1"
        and runtime.get("scene_id") == SCENE_ID
        and visual.get("packaged_runtime_map") == PACKAGED_MAP
        and visual.get("expected_static_mesh_count") == 1
        and visual.get("packaged_readback", {}).get("nullrhi") is True
        and visual.get("packaged_readback", {}).get("actor_count") == 1
        and visual.get("packaged_readback", {}).get("mesh_handle_match") is True
        and readiness.get("cook") == "pass"
        and readiness.get("packaged_mesh_readback") == "pass"
        and readiness.get("strict_gpu_capture") == "not_run"
        and runtime.get("qualification_claim") is False
        and runtime.get("formal_dataset_count") == 0,
        "Skokloster packaged runtime profile drift",
    )
    cook = values["editor_cook_plan"]
    uat = cook.get("execution_history", {}).get("uat_development_v3", {})
    readback = cook.get("execution_history", {}).get("packaged_object_readback_v2", {})
    _require(
        cook.get("schema") == "avengine_skokloster_editor_import_cook_plan_v1"
        and uat.get("status") == "pass"
        and uat.get("exit_code") == 0
        and uat.get("pak_count") == 1
        and readback.get("status") == "pass"
        and readback.get("nullrhi") is True
        and readback.get("rendering_or_capture_called") is False,
        "Skokloster Development archive/readback history drift",
    )
    audio_files = _validate_audio_files(paths)
    return {
        "status": "pass_cpu_preflight_rir_binaural_archive_bound",
        "exact_rir_job_count": 2,
        "listener_aligned_use_count": 150,
        "audio": audio_files,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def _capture_argv(request: Mapping[str, Any]) -> list[str]:
    return [
        str(request["capture_python"]),
        str(request["capture_script"]),
        "--suite-plan",
        str(request["suite_plan"]),
        "--scenario-id",
        EPISODE_ID,
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
        str(FRAME_INDEX),
        "--authorize-gpu-capture",
    ]


def prepare_request(
    *,
    atom_root: Path,
    capture_python: Path,
    spear_root: Path,
    rpc_port: int,
) -> Path:
    repo_root = REPOSITORY.resolve()
    atom_root = atom_root.resolve()
    _require(atom_root == repo_root / ATOM_DIRECTORY, "Skokloster f15 atom root drift")
    attempt_root = atom_root / ATTEMPT_DIRECTORY
    capture_output = atom_root / CAPTURE_DIRECTORY
    _require(not attempt_root.exists(), "Skokloster attempt 01 already exists")
    _require(not capture_output.exists(), "Skokloster f15 output must be fresh")
    _require(rpc_port == RPC_PORT, "Skokloster f15 RPC port drift")
    required_commit = _require_clean_head(repo_root)

    artifact_paths = _artifact_paths(atom_root)
    _require(
        all(path.is_file() for path in artifact_paths.values()),
        "accepted CPU artifact is missing",
    )
    cpu_validation = _validate_cpu_evidence(artifact_paths)
    package_paths = _package_paths()
    _require(
        all(path.is_file() for path in package_paths.values()),
        "Development archive file is missing",
    )
    source_paths = _source_paths()
    _require(
        all(path.is_file() for path in source_paths.values()),
        "Skokloster capture source is missing",
    )
    _require(capture_python.is_file(), "capture Python is missing")
    _require(spear_root.is_dir(), "SPEAR root is missing")

    stdout_path = attempt_root / "capture_stdout.log"
    stderr_path = attempt_root / "capture_stderr.log"
    request = {
        "schema": REQUEST_SCHEMA,
        "status": "prepared_not_launched",
        "episode_id": EPISODE_ID,
        "scene_id": SCENE_ID,
        "repo_root": str(repo_root),
        "required_repo_commit": required_commit,
        "required_clean_worktree": True,
        "atom_root": str(atom_root),
        "attempt_root": str(attempt_root),
        "capture_output": str(capture_output),
        "capture_stdout": str(stdout_path),
        "capture_stderr": str(stderr_path),
        "capture_python": str(capture_python.resolve()),
        "capture_script": str(source_paths["capture_wrapper"].resolve()),
        "spear_root": str(spear_root.resolve()),
        "suite_plan": str(artifact_paths["suite_plan"].resolve()),
        "audio_wav": str(artifact_paths["binaural_mixture"].resolve()),
        "packaged_map": PACKAGED_MAP,
        "packaged_executable": str(PACKAGED_EXECUTABLE),
        "artifact_records": {
            name: _file_record(path) for name, path in artifact_paths.items()
        },
        "package_records": {
            name: _file_record(path) for name, path in package_paths.items()
        },
        "source_records": {
            name: _file_record(path) for name, path in source_paths.items()
        },
        "cpu_validation": cpu_validation,
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
            "target_minimum_visible_fraction": TARGET_VISIBLE_FRACTION_MINIMUM,
            "distractor_minimum_visible_fraction": DISTRACTOR_VISIBLE_FRACTION_MINIMUM,
            "visible_pixel_count_minimum": VISIBLE_PIXEL_COUNT_MINIMUM,
            "bbox_edge_margin_px_minimum": BBOX_EDGE_MARGIN_PX_MINIMUM,
        },
        "observability_contract": {
            "exclusive_child_stdout": str(stdout_path),
            "exclusive_child_stderr": str(stderr_path),
            "ordered_launcher_phase_markers_required": True,
            "launcher_traceback_on_failure_required": True,
            "child_exit_code_in_final_receipt_required": True,
        },
        "mp3d_revision_v2_terminal_required_before_real_launch": True,
        "mp3d_revision_v2_terminal_receipt": str(MP3D_V2_TERMINAL_RECEIPT),
        "explicit_gpu_capture_authorization_required": True,
        "gpu_capture_authorized_at_prepare": False,
        "manual_visual_review_required": True,
        "qualification_claim": False,
        "formal_dataset_count": 0,
        "created_at_utc": _utc_now(),
    }
    attempt_root.mkdir(parents=True, exist_ok=False)
    request_path = attempt_root / "request.json"
    _write_json_exclusive(request_path, request)
    return request_path


def _validate_record_set(
    records: object,
    expected: Mapping[str, Path],
    *,
    owner: str,
) -> None:
    _require(
        isinstance(records, Mapping) and set(records) == set(expected),
        f"{owner} record closure drift",
    )
    for name, path in expected.items():
        observed = _validate_file_record(records[name], owner=f"{owner}.{name}")
        _require(observed == path.resolve(), f"{owner}.{name} path drift")


def _validate_request(request_path: Path) -> tuple[dict[str, Any], list[str]]:
    request_path = request_path.resolve()
    request = _load(request_path)
    _require(
        request.get("schema") == REQUEST_SCHEMA, "Skokloster f15 request schema drift"
    )
    _require(
        request.get("status") == "prepared_not_launched"
        and request.get("episode_id") == EPISODE_ID
        and request.get("scene_id") == SCENE_ID,
        "Skokloster f15 request identity drift",
    )
    repo_root = Path(str(request.get("repo_root", ""))).resolve()
    _require(repo_root == REPOSITORY.resolve(), "Skokloster request repository drift")
    observed_head = _require_clean_head(repo_root)
    _require(
        request.get("required_clean_worktree") is True
        and request.get("required_repo_commit") == observed_head,
        "repository differs from the clean request-bound commit",
    )
    atom_root = repo_root / ATOM_DIRECTORY
    attempt_root = atom_root / ATTEMPT_DIRECTORY
    capture_output = atom_root / CAPTURE_DIRECTORY
    stdout_path = attempt_root / "capture_stdout.log"
    stderr_path = attempt_root / "capture_stderr.log"
    _require(
        Path(str(request.get("atom_root", ""))).resolve() == atom_root
        and Path(str(request.get("attempt_root", ""))).resolve() == attempt_root
        and request_path == attempt_root / "request.json"
        and Path(str(request.get("capture_output", ""))).resolve() == capture_output
        and Path(str(request.get("capture_stdout", ""))).resolve() == stdout_path
        and Path(str(request.get("capture_stderr", ""))).resolve() == stderr_path,
        "Skokloster attempt/output/log path drift",
    )
    _require(
        request.get("attempt_policy") == ATTEMPT_POLICY
        and request.get("frame_indices") == [FRAME_INDEX]
        and request.get("full75_allowed") is False,
        "Skokloster one-attempt exact-f15 policy drift",
    )
    _require(
        request.get("physical_gpu_index") == 1
        and request.get("physical_gpu_uuid") == GPU1_UUID
        and request.get("graphics_adapter_argument") == 1
        and request.get("required_idle_compute_process_count") == 0,
        "Skokloster physical GPU1/adapter1 binding drift",
    )
    _require(
        request.get("packaged_map") == PACKAGED_MAP
        and Path(str(request.get("packaged_executable", ""))).resolve()
        == PACKAGED_EXECUTABLE.resolve()
        and request.get("rpc_port") == RPC_PORT,
        "Skokloster archive/map/RPC binding drift",
    )
    _require(
        request.get("mp3d_revision_v2_terminal_required_before_real_launch") is True
        and Path(str(request.get("mp3d_revision_v2_terminal_receipt", ""))).resolve()
        == MP3D_V2_TERMINAL_RECEIPT.resolve()
        and request.get("explicit_gpu_capture_authorization_required") is True
        and request.get("gpu_capture_authorized_at_prepare") is False
        and request.get("manual_visual_review_required") is True
        and request.get("qualification_claim") is False
        and request.get("formal_dataset_count") == 0,
        "Skokloster authorization/formal boundary drift",
    )
    gate = request.get("visibility_gate", {})
    _require(
        gate.get("target_instance_id") == "source1"
        and gate.get("distractor_instance_id") == "source2"
        and gate.get("target_minimum_visible_fraction")
        == TARGET_VISIBLE_FRACTION_MINIMUM
        and gate.get("distractor_minimum_visible_fraction")
        == DISTRACTOR_VISIBLE_FRACTION_MINIMUM
        and gate.get("visible_pixel_count_minimum") == VISIBLE_PIXEL_COUNT_MINIMUM
        and gate.get("bbox_edge_margin_px_minimum") == BBOX_EDGE_MARGIN_PX_MINIMUM,
        "Skokloster visibility gate drift",
    )
    observability = request.get("observability_contract", {})
    _require(
        observability.get("exclusive_child_stdout") == str(stdout_path)
        and observability.get("exclusive_child_stderr") == str(stderr_path)
        and observability.get("ordered_launcher_phase_markers_required") is True
        and observability.get("launcher_traceback_on_failure_required") is True
        and observability.get("child_exit_code_in_final_receipt_required") is True,
        "Skokloster observability contract drift",
    )

    artifact_paths = _artifact_paths(atom_root)
    package_paths = _package_paths()
    source_paths = _source_paths()
    _validate_record_set(
        request.get("artifact_records"), artifact_paths, owner="artifact"
    )
    _validate_record_set(request.get("package_records"), package_paths, owner="package")
    _validate_record_set(request.get("source_records"), source_paths, owner="source")
    _validate_cpu_evidence(artifact_paths)
    _require(
        Path(str(request.get("suite_plan", ""))).resolve()
        == artifact_paths["suite_plan"].resolve()
        and Path(str(request.get("audio_wav", ""))).resolve()
        == artifact_paths["binaural_mixture"].resolve(),
        "Skokloster capture input path drift",
    )
    _require(
        Path(str(request.get("capture_script", ""))).resolve()
        == source_paths["capture_wrapper"].resolve(),
        "Skokloster capture wrapper drift",
    )
    for name in ("capture_python", "capture_script", "spear_root"):
        _require(
            Path(str(request.get(name, ""))).exists(), f"missing capture input: {name}"
        )
    _require(not capture_output.exists(), "Skokloster f15 capture output is not fresh")

    argv = _capture_argv(request)
    _require(
        argv.count("--frame-index") == 1
        and argv[argv.index("--frame-index") + 1] == "15"
        and argv.count("--graphics-adapter") == 1
        and argv[argv.index("--graphics-adapter") + 1] == "1"
        and argv.count("--authorize-gpu-capture") == 1
        and "--spear-executable" in argv
        and argv[argv.index("--spear-executable") + 1]
        == str(PACKAGED_EXECUTABLE.resolve()),
        "capture argv is not exact f15/adapter1/Development archive",
    )
    return request, argv


def _validate_mp3d_terminal(path: Path) -> dict[str, Any]:
    path = path.resolve()
    _require(path == MP3D_V2_TERMINAL_RECEIPT.resolve(), "MP3D v2 terminal path drift")
    receipt = _load(path)
    _require(
        receipt.get("schema") == MP3D_V2_RECEIPT_SCHEMA
        and receipt.get("status") in {"pass_diagnostic_f15_review_ready", "failed"}
        and receipt.get("attempt_consumed") is True
        and isinstance(receipt.get("ended_at_utc"), str)
        and receipt.get("ended_at_utc"),
        "MP3D revision-v2 has not reached an accepted terminal state",
    )
    return _file_record(path)


def _validate_capture_file_record(record: Mapping[str, Any], *, owner: str) -> None:
    kind = record.get("kind")
    path = Path(str(record.get("path", ""))).resolve()
    if kind == "file":
        _require(path.is_file(), f"capture artifact is missing: {owner}")
        _require(
            path.stat().st_size == record.get("size_bytes"),
            f"capture artifact size drift: {owner}",
        )
        _require(
            _sha256(path) == record.get("sha256"),
            f"capture artifact hash drift: {owner}",
        )
        return
    if kind == "directory":
        _require(path.is_dir(), f"capture artifact directory is missing: {owner}")
        inventory = record.get("inventory")
        _require(
            isinstance(inventory, list) and inventory,
            f"capture directory inventory is empty: {owner}",
        )
        for item in inventory:
            child = path / str(item.get("relative_path", ""))
            _require(child.is_file(), f"capture directory member is missing: {owner}")
            _require(
                child.stat().st_size == item.get("size_bytes"),
                f"capture directory member size drift: {owner}",
            )
            _require(
                _sha256(child) == item.get("sha256"),
                f"capture directory member hash drift: {owner}",
            )
        return
    raise RuntimeError(f"unknown capture artifact kind: {owner}")


def _validate_capture(request: Mapping[str, Any]) -> dict[str, Any]:
    capture_root = Path(str(request["capture_output"])).resolve()
    manifest_path = capture_root / "manifest.json"
    truth_path = capture_root / "pixel_visibility_truth.json"
    assets_path = capture_root / "runtime_asset_readbacks.json"
    manifest = _load(manifest_path)
    truth = _load(truth_path)
    assets = _load(assets_path)
    frame = manifest.get("frame_contract", {})
    _require(
        manifest.get("schema") == "avengine_qa_native_spear_pixel_episode_v1"
        and manifest.get("status") == "pass"
        and manifest.get("scenario_id") == EPISODE_ID
        and manifest.get("native_map") == PACKAGED_MAP
        and manifest.get("benchmark_qualification_claim") is False
        and manifest.get("native_pixel_fact_binding_claim") is True,
        "Skokloster f15 capture manifest drift",
    )
    _require(
        frame.get("frame_count") == 1
        and frame.get("formal_episode_frame_count") == FRAME_COUNT
        and frame.get("captured_frame_indices") == [FRAME_INDEX]
        and frame.get("frame_rate_hz") == FPS
        and frame.get("resolution_hw") == [HEIGHT, WIDTH],
        "Skokloster capture is not exactly sparse f15",
    )
    alignment = manifest.get("runtime_alignment", {})
    _require(
        alignment.get("target_pass_count") == 2
        and float(alignment.get("maximum_location_drift_cm", -1.0)) == 0.0
        and float(alignment.get("maximum_rotation_drift_deg", -1.0)) == 0.0,
        "Skokloster shared-camera target-pass alignment drift",
    )
    runtime_assets = manifest.get("runtime_assets", {})
    _require(
        runtime_assets.get("status") == "pass"
        and runtime_assets.get("per_instance_status")
        == {"source1": "pass", "source2": "pass"}
        and assets.get("status") == "pass"
        and assets.get("per_instance", {}).get("source1", {}).get("status") == "pass"
        and assets.get("per_instance", {}).get("source2", {}).get("status") == "pass",
        "Skokloster live runtime asset readback drift",
    )
    audio = manifest.get("audio", {})
    audio_record = request["artifact_records"]["binaural_mixture"]
    _require(
        Path(str(audio.get("authoritative_wav", ""))).resolve()
        == Path(str(request["audio_wav"])).resolve()
        and audio.get("sha256") == audio_record.get("sha256"),
        "Skokloster f15 audio binding drift",
    )
    records = manifest.get("artifact_records")
    _require(
        isinstance(records, Mapping)
        and REQUIRED_CAPTURE_ARTIFACT_ROLES.issubset(records),
        "Skokloster capture artifact closure drift",
    )
    for role in sorted(REQUIRED_CAPTURE_ARTIFACT_ROLES):
        _validate_capture_file_record(records[role], owner=role)

    _require(
        truth.get("schema") == "avengine_qa_pixel_visibility_truth_v1",
        "Skokloster pixel visibility schema drift",
    )
    per_instance = truth.get("per_instance", {})
    target_rows = per_instance.get("source1", {}).get("frames", [])
    distractor_rows = per_instance.get("source2", {}).get("frames", [])
    _require(
        len(target_rows) == 1
        and len(distractor_rows) == 1
        and target_rows[0].get("frame_index") == FRAME_INDEX
        and distractor_rows[0].get("frame_index") == FRAME_INDEX,
        "Skokloster pixel truth is not exactly f15",
    )
    target = target_rows[0]
    distractor = distractor_rows[0]
    _require(
        float(target.get("visible_fraction", -1.0)) >= TARGET_VISIBLE_FRACTION_MINIMUM
        and float(distractor.get("visible_fraction", -1.0))
        >= DISTRACTOR_VISIBLE_FRACTION_MINIMUM
        and int(target.get("visible_pixels", -1)) >= VISIBLE_PIXEL_COUNT_MINIMUM
        and int(distractor.get("visible_pixels", -1)) >= VISIBLE_PIXEL_COUNT_MINIMUM,
        "Skokloster f15 visibility/pixel gate failed",
    )
    for owner, row in (("target", target), ("distractor", distractor)):
        bbox = row.get("target_bbox_xyxy_px")
        _require(isinstance(bbox, list) and len(bbox) == 4, f"{owner} bbox is missing")
        x1, y1, x2, y2 = (int(value) for value in bbox)
        margin = BBOX_EDGE_MARGIN_PX_MINIMUM
        _require(
            x1 >= margin
            and y1 >= margin
            and WIDTH - 1 - x2 >= margin
            and HEIGHT - 1 - y2 >= margin,
            f"{owner} bbox touches the frame edge",
        )
    return {
        "status": "pass_diagnostic_f15_manual_review_pending",
        "capture_manifest": _file_record(manifest_path),
        "pixel_visibility_truth": _file_record(truth_path),
        "runtime_asset_readbacks": _file_record(assets_path),
        "target_visible_fraction": float(target["visible_fraction"]),
        "distractor_visible_fraction": float(distractor["visible_fraction"]),
        "target_visible_pixels": int(target["visible_pixels"]),
        "distractor_visible_pixels": int(distractor["visible_pixels"]),
        "manual_visual_review_required": True,
        "qualification_claim": False,
        "formal_dataset_count": 0,
    }


def _write_phase(
    attempt_root: Path,
    *,
    sequence: int,
    phase: str,
    detail: Mapping[str, Any] | None = None,
) -> Path:
    path = attempt_root / f"launch_phase_{sequence:03d}_{phase}.json"
    _write_json_exclusive(
        path,
        {
            "schema": PHASE_SCHEMA,
            "status": "entered",
            "sequence": sequence,
            "phase": phase,
            "detail": dict(detail or {}),
            "qualification_claim": False,
            "formal_dataset_count": 0,
            "captured_at_utc": _utc_now(),
        },
    )
    return path


def _collect_phases(attempt_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(attempt_root.glob("launch_phase_*.json")):
        value = _load(path)
        _require(
            value.get("schema") == PHASE_SCHEMA
            and value.get("status") == "entered"
            and value.get("qualification_claim") is False
            and value.get("formal_dataset_count") == 0,
            f"invalid Skokloster launcher phase: {path}",
        )
        records.append(
            {
                "sequence": value.get("sequence"),
                "phase": value.get("phase"),
                "artifact": _file_record(path),
            }
        )
    _require(
        [record["sequence"] for record in records] == list(range(len(records))),
        "Skokloster launcher phase sequence is not contiguous",
    )
    return records


def run(
    request_path: Path,
    *,
    dry_run: bool,
    authorize_gpu_capture: bool,
    mp3d_v2_terminal_receipt: Path,
) -> int:
    request, argv = _validate_request(request_path.resolve())
    attempt_root = Path(str(request["attempt_root"])).resolve()
    stdout_path = Path(str(request["capture_stdout"])).resolve()
    stderr_path = Path(str(request["capture_stderr"])).resolve()
    dry_receipt = attempt_root / "dry_run_receipt.json"
    running_receipt = attempt_root / "running_receipt.json"
    final_receipt = attempt_root / "final_receipt.json"
    _require(
        not final_receipt.exists(), "Skokloster attempt 01 already has a final receipt"
    )
    if dry_run:
        _require(not dry_receipt.exists(), "Skokloster dry-run receipt already exists")
        _require(
            not running_receipt.exists(), "Skokloster real attempt already started"
        )
    else:
        _require(
            authorize_gpu_capture,
            "Skokloster GPU capture lacks explicit launch authorization",
        )
        _require(
            not running_receipt.exists(), "Skokloster real attempt already started"
        )
        _require(
            not stdout_path.exists() and not stderr_path.exists(),
            "exclusive child log path already exists",
        )

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
        "capture_stdout": str(stdout_path),
        "capture_stderr": str(stderr_path),
        "frame_indices": [FRAME_INDEX],
        "full75_allowed": False,
        "packaged_map": PACKAGED_MAP,
        "packaged_executable": str(PACKAGED_EXECUTABLE),
        "physical_gpu_index": 1,
        "physical_gpu_uuid": GPU1_UUID,
        "graphics_adapter_argument": 1,
        "prelaunch_gpu": gpu,
        "prelaunch_snapshot": before,
        "manual_visual_review_required": True,
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
                "mp3d_revision_v2_terminal_checked": False,
                "captured_at_utc": _utc_now(),
            },
        )
        return 0

    upstream_record = _validate_mp3d_terminal(mp3d_v2_terminal_receipt)
    started_at = _utc_now()
    common["mp3d_revision_v2_terminal_receipt"] = upstream_record
    _write_json_exclusive(
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
        _write_phase(
            attempt_root,
            sequence=0,
            phase="prelaunch_closed",
            detail={"mp3d_v2_terminal_bound": True, "gpu1_idle": True},
        )
        current_phase = "child_invocation_started"
        _write_phase(
            attempt_root,
            sequence=1,
            phase=current_phase,
            detail={"argv_count": len(argv)},
        )
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
        current_phase = "child_exit_observed"
        _write_phase(
            attempt_root,
            sequence=2,
            phase=current_phase,
            detail={"returncode": child_exit_code},
        )
        _require(
            child_exit_code == 0, f"Skokloster f15 capture exited {child_exit_code}"
        )
        current_phase = "capture_validation_started"
        _write_phase(attempt_root, sequence=3, phase=current_phase)
        final["validation"] = _validate_capture(request)
        current_phase = "complete"
        _write_phase(attempt_root, sequence=4, phase=current_phase)
        final["status"] = "pass_diagnostic_f15_manual_review_pending"
    except Exception as exc:  # noqa: BLE001
        final["error"] = f"{type(exc).__name__}: {exc}"
        final["failure_phase"] = current_phase
        final["launcher_traceback"] = traceback.format_exc()
        exit_code = exit_code or 1
    finally:
        final["ended_at_utc"] = _utc_now()
        final["child_invocation_attempted"] = child_invocation_attempted
        final["child_exit_code"] = child_exit_code
        final["capture_process_exit_code"] = child_exit_code
        final["gpu_started"] = child_invocation_attempted
        final["exclusive_child_stdout"] = (
            _file_record(stdout_path) if stdout_path.is_file() else None
        )
        final["exclusive_child_stderr"] = (
            _file_record(stderr_path) if stderr_path.is_file() else None
        )
        try:
            final["launcher_phases"] = _collect_phases(attempt_root)
        except Exception as exc:  # noqa: BLE001
            final["launcher_phase_collection_error"] = f"{type(exc).__name__}: {exc}"
            final["launcher_phase_collection_traceback"] = traceback.format_exc()
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
    prepare.add_argument("--capture-python", type=Path, default=CAPTURE_PYTHON_LOGICAL)
    prepare.add_argument("--spear-root", type=Path, default=SPEAR_ROOT)
    prepare.add_argument("--rpc-port", type=int, default=RPC_PORT)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--request", required=True, type=Path)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--authorize-gpu-capture", action="store_true")
    launch.add_argument(
        "--mp3d-v2-terminal-receipt",
        type=Path,
        default=MP3D_V2_TERMINAL_RECEIPT,
    )
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
        print(f"SKOKLOSTER_F15_REQUEST_PREPARED request={path} formal=0", flush=True)
        return 0
    return run(
        args.request,
        dry_run=args.dry_run,
        authorize_gpu_capture=args.authorize_gpu_capture,
        mp3d_v2_terminal_receipt=args.mp3d_v2_terminal_receipt,
    )


if __name__ == "__main__":
    raise SystemExit(main())

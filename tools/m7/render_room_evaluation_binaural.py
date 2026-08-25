#!/usr/bin/env python3
"""Mix generic room-evaluation sound classes through a completed RIR cache."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    load_json,
    sha256_file,
    write_json,
)
from avengine.spatial_audio.audio import read_float32_wav, write_float32_wav
from avengine.timeline.audio import M5_AUDIO_SAMPLE_COUNT, M5_AUDIO_SAMPLE_RATE_HZ, raised_cosine_partition
from avengine.acoustics.rir_cache import (
    RIR_CACHE_INDEX_SCHEMA,
    RIR_CACHE_RECEIPT_SCHEMA,
    RIR_CACHE_REQUEST_SCHEMA,
    RIRCacheSession,
    validate_rir_job_plan,
)
from avengine.routes.room_feasibility import (
    RIR_JOB_PLAN_SCHEMA,
    TRAJECTORY_BANK_SCHEMA,
)
from avengine.m7.asset_bound_audio import (
    AssetBoundAudioError,
    float32_stems_and_exact_mix,
    prepare_dry_audio,
    render_asset_bound_binaural,
)
from avengine.m7.room_evaluation import (
    ROOM_EVALUATION_PLAN_SCHEMA,
    ROOM_SOUND_ASSIGNMENTS_SCHEMA,
    RoomEvaluationError,
    validate_episode_id,
)
from avengine.m7.sensor_rig import (
    M7SensorRigError,
    m7_sensor_rig_binding,
    validate_m7_rir_listener_alignment,
)
from avengine.security.path_policy import (
    WorkspacePathPolicy,
    atomic_publish_directory,
)


SCHEMA = "avengine_room_evaluation_binaural_batch_v1"
INPUT_CLOSURE_SCHEMA = "avengine_room_evaluation_binaural_input_closure_v1"
OUTPUT_CLOSURE_SCHEMA = "avengine_room_evaluation_binaural_output_closure_v1"
SOURCE_SLOTS = ("source1", "source2")
FRAME_COUNT = 75
FRAME_RATE_HZ = 15
PLAN_FILES = (
    "delivery.json",
    "trajectory_bank.json",
    "sound_assignments.json",
    "rir_job_plan.json",
)
CACHE_ONLY_CONTROL_FLOW = (
    "validated RIRCacheSession initialization",
    "read-only RIRCacheSession.load_episode",
    "NumPy dynamic convolution and exact float32 stem accumulation",
    "float32 WAVE write plus authenticated readback",
)
RESULT_CHANGING_CODE_FILES = (
    "tools/m7/render_room_evaluation_binaural.py",
    "src/avengine/m7/room_evaluation.py",
    "src/avengine/m7/sensor_rig.py",
    "src/avengine/m7/asset_bound_audio.py",
    "src/avengine/capture/dry_audio.py",
    "src/avengine/timeline/audio.py",
    "src/avengine/spatial_audio/audio.py",
    "src/avengine/acoustics/rir_cache.py",
    "src/avengine/contracts/json_io.py",
)
OUTPUT_CLOSURE_FILES = (
    "input_closure.json",
    "samples.json",
    "dry_audio_classes.json",
    "timing.json",
)


def _safe_episode_id(value: Any, *, owner: str) -> str:
    try:
        return validate_episode_id(value)
    except RoomEvaluationError as exc:
        raise AssetBoundAudioError(f"{owner} has an unsafe episode_id") from exc


def _producer_identity(repository_root: Path | None = None) -> dict[str, Any]:
    """Bind the exact assembly code and numerical runtime that affect outputs."""

    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else repository_root.resolve(strict=True)
    )
    code_files: dict[str, dict[str, Any]] = {}
    for relative in RESULT_CHANGING_CODE_FILES:
        path = root / relative
        if not path.is_file():
            raise AssetBoundAudioError(f"result-changing producer file is missing: {path}")
        code_files[relative] = {
            "path": relative,
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema": "avengine_room_evaluation_binaural_producer_identity_v1",
        "entrypoint": RESULT_CHANGING_CODE_FILES[0],
        "result_changing_code_files": code_files,
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_byteorder": sys.byteorder,
            "numpy_version": np.__version__,
        },
    }


def _publication_paths(
    raw_output: Path,
) -> tuple[WorkspacePathPolicy, Path, Path]:
    """Reserve one sibling staging path and immutable final destination."""

    unresolved = raw_output.expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    if os.path.lexists(unresolved):
        raise FileExistsError(f"refusing to replace output: {unresolved}")
    if unresolved.name in {"", ".", ".."}:
        raise AssetBoundAudioError("output must name one directory")
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    output_parent = unresolved.parent.resolve(strict=True)
    policy = WorkspacePathPolicy.from_roots([output_parent])
    output = policy.resolve_output(
        output_parent / unresolved.name,
        owner="room evaluation binaural batch",
    )
    staging = policy.resolve_output(
        output.with_name(f".{output.name}.staging-{uuid4().hex}"),
        owner="room evaluation binaural staging directory",
    )
    return policy, output, staging


def _mapping(values: list[str], *, owner: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if separator != "=" or not key.strip() or not value.strip():
            raise AssetBoundAudioError(f"{owner} must use SOUND_CLASS=value")
        key = key.strip()
        if key in result:
            raise AssetBoundAudioError(f"{owner} repeats {key!r}")
        result[key] = value.strip()
    return result


def _gains(values: list[str]) -> dict[str, float]:
    result = {}
    for key, value in _mapping(values, owner="class-linear-gain").items():
        number = float(value)
        if not np.isfinite(number) or number < 0.0:
            raise AssetBoundAudioError("class gains must be finite and nonnegative")
        result[key] = number
    return result


def _active_stem_peaks(
    stems: Mapping[str, np.ndarray], *, sample_id: str
) -> dict[str, float]:
    """Prove both source slots survive float32 delivery arithmetic."""

    if set(stems) != set(SOURCE_SLOTS):
        raise AssetBoundAudioError(f"{sample_id} stems differ from source1/source2")
    peaks = {
        slot: float(np.max(np.abs(np.asarray(stems[slot], dtype=np.float32))))
        for slot in SOURCE_SLOTS
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in peaks.values()):
        raise AssetBoundAudioError(
            f"{sample_id} does not contain two active float32 source stems"
        )
    return peaks


def _verify_persisted_exact_mix(
    stems: Mapping[str, np.ndarray],
    mixture: np.ndarray,
    *,
    sample_id: str,
) -> dict[str, float]:
    """Verify the delivered mixture from delivered float32 stem readbacks."""

    peaks = _active_stem_peaks(stems, sample_id=sample_id)
    delivered = np.asarray(mixture)
    if delivered.dtype != np.float32:
        raise AssetBoundAudioError(f"{sample_id} delivered mixture is not float32")
    expected = np.zeros_like(delivered, dtype=np.float32)
    for slot in SOURCE_SLOTS:
        stem = np.asarray(stems[slot])
        if stem.dtype != np.float32 or stem.shape != delivered.shape:
            raise AssetBoundAudioError(
                f"{sample_id} delivered {slot} stem differs from the mixture layout"
            )
        expected += stem
    if not np.array_equal(delivered, expected):
        raise AssetBoundAudioError(
            f"{sample_id} delivered mixture differs from delivered stem sum"
        )
    return peaks


def _write_and_readback(
    path: Path,
    samples: np.ndarray,
    *,
    root: Path,
    role: str,
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    """Persist one float32 WAVE and return authenticated, byte-exact readback."""

    artifact = write_float32_wav(
        path,
        samples,
        M5_AUDIO_SAMPLE_RATE_HZ,
        metadata=dict(metadata) | {"role": role},
    )
    readback = read_float32_wav(
        artifact.audio_path,
        sidecar_path=artifact.sidecar_path,
        verify_sidecar=True,
    )
    expected = np.asarray(samples, dtype=np.float32)
    if (
        readback.sample_rate_hz != M5_AUDIO_SAMPLE_RATE_HZ
        or readback.samples.shape != expected.shape
        or not np.array_equal(readback.samples, expected)
    ):
        raise AssetBoundAudioError(f"float32 WAVE readback differs: {path}")
    return (
        {
            "audio_path": str(artifact.audio_path.relative_to(root)),
            "audio_sidecar_path": str(artifact.sidecar_path.relative_to(root)),
            "audio_sha256": artifact.audio_sha256,
            "sidecar_sha256": artifact.sidecar_sha256,
            "sample_rate_hz": artifact.sample_rate_hz,
            "sample_count": artifact.frame_count,
            "channel_count": artifact.channel_count,
            "peak_absolute": float(np.max(np.abs(readback.samples))),
        },
        readback.samples,
    )


def _output_closure(root: Path, *, sample_count: int) -> dict[str, Any]:
    """Bind the non-self-referential JSON outputs and their audio index."""

    files: dict[str, dict[str, Any]] = {}
    for relative in OUTPUT_CLOSURE_FILES:
        path = root / relative
        if not path.is_file() or path.stat().st_size < 1:
            raise AssetBoundAudioError(f"output closure file is missing: {relative}")
        files[relative] = {
            "path": relative,
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema": OUTPUT_CLOSURE_SCHEMA,
        "status": "pass",
        "sample_count": sample_count,
        "wave_file_count": sample_count * 3,
        "wave_sidecar_count": sample_count * 3,
        "audio_artifact_file_count": sample_count * 6,
        "audio_artifact_hashes_bound_by": "samples.json",
        "files": files,
    }


def _assignments(path: Path) -> tuple[list[Mapping[str, Any]], set[str]]:
    value = load_json(path)
    raw = value.get("assignments")
    if (
        value.get("schema") != ROOM_SOUND_ASSIGNMENTS_SCHEMA
        or value.get("status") != "pass"
        or value.get("both_sources_active") is not True
    ):
        raise AssetBoundAudioError("sound assignment document is invalid")
    if not isinstance(raw, list) or not raw:
        raise AssetBoundAudioError("sound assignment document is empty")
    result = []
    classes: set[str] = set()
    episodes: set[str] = set()
    pair_counts: Counter[str] = Counter()
    for item in raw:
        episode_id = _safe_episode_id(
            item.get("episode_id") if isinstance(item, Mapping) else None,
            owner="sound assignment",
        )
        sources = item.get("source_classes") if isinstance(item, Mapping) else None
        if (
            episode_id in episodes
            or not isinstance(sources, Mapping)
            or set(sources) != set(SOURCE_SLOTS)
            or any(not isinstance(sources[slot], str) or not sources[slot] for slot in SOURCE_SLOTS)
            or sources["source1"] == sources["source2"]
        ):
            raise AssetBoundAudioError("sound assignment entry is invalid")
        result.append({"episode_id": episode_id, "source_classes": dict(sources)})
        classes.update(str(sources[slot]) for slot in SOURCE_SLOTS)
        pair_counts[f"{sources['source1']}|{sources['source2']}"] += 1
        episodes.add(episode_id)
    declared_classes = value.get("sound_classes")
    if (
        value.get("episode_count") != len(result)
        or not isinstance(declared_classes, list)
        or any(
            not isinstance(item, str) or not item or item.strip() != item
            for item in declared_classes
        )
        or len(set(declared_classes)) != len(declared_classes)
        or set(declared_classes) != classes
        or value.get("ordered_pair_counts") != dict(sorted(pair_counts.items()))
    ):
        raise AssetBoundAudioError("sound assignment header differs from its entries")
    return result, classes


def _plan_closure(
    plan_root: Path,
    assignments: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and bind the four documents that define one audio batch."""

    paths = {name: plan_root / name for name in PLAN_FILES}
    sensor_rig_path = plan_root / "sensor_rig_trajectory.json"
    if sensor_rig_path.is_file():
        paths["sensor_rig_trajectory.json"] = sensor_rig_path
    if any(not path.is_file() for path in paths.values()):
        raise AssetBoundAudioError("room evaluation plan closure is incomplete")
    delivery = load_json(paths["delivery.json"])
    trajectory = load_json(paths["trajectory_bank.json"])
    sound = load_json(paths["sound_assignments.json"])
    rir_plan = load_json(paths["rir_job_plan.json"])
    jobs = validate_rir_job_plan(rir_plan)
    sensor_rig_trajectory = (
        load_json(paths["sensor_rig_trajectory.json"])
        if "sensor_rig_trajectory.json" in paths
        else None
    )
    listener_pose_mode = rir_plan.get("listener_pose_mode", "fixed")
    if listener_pose_mode == "per_episode_frame" and sensor_rig_trajectory is None:
        raise AssetBoundAudioError(
            "per-frame Listener RIR plan lacks SensorRigTrajectory sidecar"
        )
    sensor_rig_binding = None
    sensor_rig_alignment = None
    if sensor_rig_trajectory is not None:
        try:
            sensor_rig_binding = m7_sensor_rig_binding(
                sensor_rig_trajectory
            )
            sensor_rig_alignment = validate_m7_rir_listener_alignment(
                rir_job_plan=rir_plan,
                sensor_rig_trajectory=sensor_rig_trajectory,
            )
        except M7SensorRigError as exc:
            raise AssetBoundAudioError(str(exc)) from exc

    assignment_ids = tuple(str(item["episode_id"]) for item in assignments)
    assignment_set = set(assignment_ids)
    if len(assignment_set) != len(assignment_ids):
        raise AssetBoundAudioError("room evaluation assignments repeat an episode")

    raw_episodes = trajectory.get("episodes")
    if (
        trajectory.get("schema") != TRAJECTORY_BANK_SCHEMA
        or trajectory.get("source_slots") != list(SOURCE_SLOTS)
        or trajectory.get("frame_count") != FRAME_COUNT
        or trajectory.get("frame_rate_hz") != FRAME_RATE_HZ
        or trajectory.get("seconds_per_episode")
        != FRAME_COUNT / FRAME_RATE_HZ
        or isinstance(trajectory.get("seed"), bool)
        or not isinstance(trajectory.get("seed"), int)
        or not isinstance(raw_episodes, list)
        or trajectory.get("episode_count") != len(raw_episodes)
        or len(raw_episodes) != len(assignments)
    ):
        raise AssetBoundAudioError("trajectory bank header differs from renderer contract")

    trajectories: dict[str, dict[str, np.ndarray]] = {}
    actual_motion_counts: Counter[str] = Counter()
    for raw in raw_episodes:
        episode_id = _safe_episode_id(
            raw.get("episode_id") if isinstance(raw, Mapping) else None,
            owner="trajectory bank",
        )
        motion_case = raw.get("motion_case") if isinstance(raw, Mapping) else None
        centers = (
            raw.get("source_center_paths_m") if isinstance(raw, Mapping) else None
        )
        roots = raw.get("source_root_paths_m") if isinstance(raw, Mapping) else None
        if (
            episode_id in trajectories
            or not isinstance(motion_case, str)
            or not motion_case
            or not isinstance(centers, Mapping)
            or not isinstance(roots, Mapping)
            or set(centers) != set(SOURCE_SLOTS)
            or set(roots) != set(SOURCE_SLOTS)
        ):
            raise AssetBoundAudioError("trajectory bank contains an invalid episode")
        normalized_centers: dict[str, np.ndarray] = {}
        for owner, source_paths in (("center", centers), ("root", roots)):
            for slot in SOURCE_SLOTS:
                try:
                    points = np.asarray(source_paths[slot], dtype=np.float64)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise AssetBoundAudioError(
                        f"{episode_id} {slot} {owner} path is invalid"
                    ) from exc
                if points.shape != (FRAME_COUNT, 3) or not np.all(
                    np.isfinite(points)
                ):
                    raise AssetBoundAudioError(
                        f"{episode_id} {slot} {owner} path is invalid"
                    )
                if owner == "center":
                    normalized_centers[slot] = np.ascontiguousarray(points)
        trajectories[episode_id] = normalized_centers
        actual_motion_counts[motion_case] += 1
    if set(trajectories) != assignment_set:
        raise AssetBoundAudioError(
            "trajectory bank episode IDs differ from sound assignments"
        )

    declared_motion_counts = trajectory.get("motion_case_counts")
    if (
        not isinstance(declared_motion_counts, Mapping)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in declared_motion_counts.values()
        )
        or sum(declared_motion_counts.values()) != len(assignments)
        or any(
            declared_motion_counts.get(key) != value
            for key, value in actual_motion_counts.items()
        )
        or any(
            key not in actual_motion_counts and value != 0
            for key, value in declared_motion_counts.items()
        )
    ):
        raise AssetBoundAudioError("trajectory bank motion counts differ")

    sound_pair_counts = Counter(
        f"{item['source_classes']['source1']}|"
        f"{item['source_classes']['source2']}"
        for item in assignments
    )
    if (
        sound.get("schema") != ROOM_SOUND_ASSIGNMENTS_SCHEMA
        or sound.get("status") != "pass"
        or sound.get("episode_count") != len(assignments)
        or sound.get("both_sources_active") is not True
        or sound.get("ordered_pair_counts")
        != dict(sorted(sound_pair_counts.items()))
        or [
            item.get("episode_id") if isinstance(item, Mapping) else None
            for item in sound.get("assignments", [])
        ]
        != list(assignment_ids)
    ):
        raise AssetBoundAudioError("sound assignment document changed during closure")

    use_count = 0
    plan_episode_ids: set[str] = set()
    frames_by_episode_slot: defaultdict[
        str, dict[str, set[int]]
    ] = defaultdict(lambda: {slot: set() for slot in SOURCE_SLOTS})
    for job in jobs:
        position = np.asarray(job["source_position_m"], dtype=np.float64)
        for use in job["uses"]:
            episode_id = _safe_episode_id(
                use["episode_id"],
                owner="RIR plan use",
            )
            slot = str(use["source_slot_id"])
            frame = int(use["frame_index"])
            if (
                episode_id not in trajectories
                or frame >= FRAME_COUNT
                or not np.array_equal(
                    trajectories[episode_id][slot][frame], position
                )
            ):
                raise AssetBoundAudioError(
                    "RIR plan use differs from its trajectory source center"
                )
            if frame in frames_by_episode_slot[episode_id][slot]:
                raise AssetBoundAudioError("RIR plan repeats an episode source frame")
            frames_by_episode_slot[episode_id][slot].add(frame)
            plan_episode_ids.add(episode_id)
            use_count += 1
    stride = rir_plan.get("stride_frames")
    expected_frames = (
        set(range(0, FRAME_COUNT, stride))
        if isinstance(stride, int) and not isinstance(stride, bool) and stride > 0
        else set()
    )
    if (
        rir_plan.get("schema") != RIR_JOB_PLAN_SCHEMA
        or rir_plan.get("dry_audio_independent") is not True
        or plan_episode_ids != assignment_set
        or not expected_frames
        or any(
            values[slot] != expected_frames
            for values in frames_by_episode_slot.values()
            for slot in SOURCE_SLOTS
        )
        or rir_plan.get("requested_pair_state_count") != use_count
        or rir_plan.get("cache_reuse_count") != use_count - len(jobs)
    ):
        raise AssetBoundAudioError("RIR plan header or episode grid differs")

    fixed_listener_differs = listener_pose_mode == "fixed" and (
        delivery.get("listener_position_m")
        != rir_plan.get("listener_position_m")
        or delivery.get("listener_orientation_wxyz")
        != rir_plan.get("listener_orientation_wxyz")
    )
    declared_sensor_rig = delivery.get("sensor_rig_trajectory")
    sensor_rig_differs = (
        sensor_rig_binding is None
        and declared_sensor_rig is not None
    ) or (
        sensor_rig_binding is not None
        and (
            not isinstance(declared_sensor_rig, Mapping)
            or declared_sensor_rig.get("trajectory_id")
            != sensor_rig_binding["trajectory_id"]
            or declared_sensor_rig.get("content_sha256")
            != sensor_rig_binding["content_sha256"]
            or delivery.get("listener_pose_mode")
            != "sensor_rig_trajectory_v1"
        )
    )
    if (
        delivery.get("schema") != ROOM_EVALUATION_PLAN_SCHEMA
        or delivery.get("status") != "pass"
        or delivery.get("research_only") is not True
        or delivery.get("qualification_claim") is not False
        or delivery.get("dry_audio_independent") is not True
        or delivery.get("visual_asset_independent") is not True
        or delivery.get("episode_count") != len(assignments)
        or delivery.get("frame_count") != FRAME_COUNT
        or delivery.get("frame_rate_hz") != FRAME_RATE_HZ
        or delivery.get("motion_case_counts")
        != {
            key: value
            for key, value in sorted(declared_motion_counts.items())
            if value > 0
        }
        or delivery.get("sound_pair_counts")
        != dict(sorted(sound_pair_counts.items()))
        or fixed_listener_differs
        or sensor_rig_differs
        or delivery.get("unique_rir_job_count") != len(jobs)
        or delivery.get("requested_pair_state_count") != use_count
        or delivery.get("cache_reuse_count") != use_count - len(jobs)
    ):
        raise AssetBoundAudioError("room evaluation delivery header differs")

    result = {
        "schema": INPUT_CLOSURE_SCHEMA,
        "status": "pass",
        "episode_count": len(assignments),
        "frame_count": FRAME_COUNT,
        "frame_rate_hz": FRAME_RATE_HZ,
        "listener_position_m": list(delivery["listener_position_m"]),
        "listener_orientation_wxyz": list(
            delivery["listener_orientation_wxyz"]
        ),
        "listener_pose_mode": listener_pose_mode,
        "unique_rir_job_count": len(jobs),
        "requested_pair_state_count": use_count,
        "files": {
            name: {"path": name, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    if sensor_rig_binding is not None:
        result["sensor_rig_trajectory"] = sensor_rig_binding
        result["sensor_rig_rir_alignment"] = sensor_rig_alignment
    return result


def _cache_closure(
    cache_root: Path,
    session: RIRCacheSession,
    *,
    rir_plan_sha256: str,
    unique_rir_job_count: int,
) -> dict[str, Any]:
    """Bind the read-only cache request/receipt/index selected by the session."""

    paths = {
        name: cache_root / name for name in ("request.json", "receipt.json", "index.json")
    }
    acoustic_selection_path = cache_root / "acoustic_selection.json"
    if acoustic_selection_path.is_file():
        paths["acoustic_selection.json"] = acoustic_selection_path
    if any(not path.is_file() for path in paths.values()):
        raise AssetBoundAudioError("RIR cache closure is incomplete")
    request = load_json(paths["request.json"])
    receipt = load_json(paths["receipt.json"])
    index = load_json(paths["index.json"])
    request_plan = request.get("plan")
    request_identity = request.get("request_identity_sha256")
    external_inputs = getattr(session, "external_input_identity", None)
    acoustic_scene = (
        external_inputs.get("acoustic_scene")
        if isinstance(external_inputs, Mapping)
        else None
    )
    simulation_input = (
        external_inputs.get("simulation_request")
        if isinstance(external_inputs, Mapping)
        else None
    )
    hrtf_input = (
        external_inputs.get("hrtf")
        if isinstance(external_inputs, Mapping)
        else None
    )
    plan_input = (
        external_inputs.get("plan")
        if isinstance(external_inputs, Mapping)
        else None
    )
    request_scene = request.get("acoustic_scene")
    request_simulation = request.get("simulation")
    request_output = request.get("output")
    selection_binding = getattr(session, "acoustic_selection_binding", None)
    if (
        not isinstance(selection_binding, Mapping)
        or selection_binding.get("schema")
        != "avengine_rir_cache_acoustic_selection_binding_v1"
    ):
        raise AssetBoundAudioError(
            "RIR cache lacks a validated acoustic selection binding"
        )
    selection_binding = dict(selection_binding)
    selection_mode = selection_binding.get("selection_mode")
    selection_binding_sha256 = selection_binding.get("binding_content_sha256")
    request_binding = request.get("acoustic_selection_binding")
    registry_mode = selection_mode in {
        "registry",
        "registry_with_verified_equivalent_overrides",
    }
    bound_mode = registry_mode or selection_mode == "explicit_legacy"
    if bound_mode:
        expected_binding_sha256 = canonical_json_sha256(
            {
                key: item
                for key, item in selection_binding.items()
                if key != "binding_content_sha256"
            }
        )
        if (
            request_binding != selection_binding
            or selection_binding_sha256 != expected_binding_sha256
            or receipt.get("acoustic_selection_binding_sha256")
            != selection_binding_sha256
            or index.get("acoustic_selection_binding_sha256")
            != selection_binding_sha256
            or receipt.get("acoustic_selection_mode") != selection_mode
            or index.get("acoustic_selection_mode") != selection_mode
            or "acoustic_selection.json" not in paths
        ):
            raise AssetBoundAudioError(
                "RIR cache acoustic selection differs across its closure"
            )
    elif selection_mode == "explicit_legacy_unbound":
        if (
            request_binding is not None
            or selection_binding_sha256 is not None
            or selection_binding.get("registry_selection_applied") is not False
            or selection_binding.get("room_ref") is not None
            or selection_binding.get("profile_ref") is not None
            or selection_binding.get("binding_id") is not None
        ):
            raise AssetBoundAudioError(
                "legacy unbound RIR cache fabricated an acoustic identity"
            )
    else:
        raise AssetBoundAudioError("RIR cache acoustic selection mode is invalid")
    if (
        registry_mode
        and (
            selection_binding.get("registry_selection_applied") is not True
            or not isinstance(selection_binding.get("room_ref"), Mapping)
            or not isinstance(selection_binding.get("profile_ref"), Mapping)
        )
    ):
        raise AssetBoundAudioError(
            "registry RIR cache lacks its exact room/profile identity"
        )
    if (
        request.get("schema") != RIR_CACHE_REQUEST_SCHEMA
        or not isinstance(request_plan, Mapping)
        or request_plan.get("sha256") != rir_plan_sha256
        or request_plan.get("full_job_count") != unique_rir_job_count
        or request_plan.get("selected_job_offset") != 0
        or request_plan.get("selected_job_count") != unique_rir_job_count
        or request_identity != session.request_identity_sha256
        or session.plan_sha256 != rir_plan_sha256
        or receipt.get("schema") != RIR_CACHE_RECEIPT_SCHEMA
        or receipt.get("status") != "pass"
        or receipt.get("full_plan_complete") is not True
        or receipt.get("request_identity_sha256") != request_identity
        or receipt.get("full_plan_job_count") != unique_rir_job_count
        or receipt.get("selected_job_count") != unique_rir_job_count
        or receipt.get("retained_payload_hash_verified") is not True
        or index.get("schema") != RIR_CACHE_INDEX_SCHEMA
        or index.get("status") != "pass"
        or index.get("full_plan_complete") is not True
        or index.get("request_identity_sha256") != request_identity
        or index.get("selected_job_count") != unique_rir_job_count
        or not isinstance(index.get("entries"), list)
        or len(index["entries"]) != unique_rir_job_count
        or not isinstance(external_inputs, Mapping)
        or external_inputs.get("status") != "pass"
        or external_inputs.get("acoustic_selection_binding_sha256")
        != selection_binding_sha256
        or not isinstance(plan_input, Mapping)
        or plan_input.get("declared_path") != request_plan.get("path")
        or plan_input.get("sha256") != request_plan.get("sha256")
        or not isinstance(request_scene, Mapping)
        or not isinstance(acoustic_scene, Mapping)
        or acoustic_scene.get("declared_path")
        != request_scene.get("manifest_path")
        or acoustic_scene.get("sha256") != request_scene.get("manifest_sha256")
        or acoustic_scene.get("package_id") != request_scene.get("package_id")
        or acoustic_scene.get("package_content_sha256")
        != request_scene.get("package_content_sha256")
        or acoustic_scene.get("manifest_content_identity_verified") is not True
        or not isinstance(request_simulation, Mapping)
        or not isinstance(simulation_input, Mapping)
        or simulation_input.get("declared_path")
        != request_simulation.get("request_path")
        or simulation_input.get("sha256")
        != request_simulation.get("request_sha256")
        or not isinstance(request_output, Mapping)
        or (
            request_output.get("layout_type") == "binaural"
            and (
                not isinstance(hrtf_input, Mapping)
                or hrtf_input.get("declared_path") != request_output.get("hrtf_path")
                or hrtf_input.get("sha256") != request_output.get("hrtf_sha256")
            )
        )
        or (
            request_output.get("layout_type") == "ambisonics"
            and hrtf_input is not None
        )
    ):
        raise AssetBoundAudioError(
            "RIR cache request/receipt/index differs from the current plan"
        )
    return {
        "status": "pass",
        "request_identity_sha256": request_identity,
        "rir_plan_sha256": rir_plan_sha256,
        "full_plan_complete": True,
        "full_plan_job_count": unique_rir_job_count,
        "retained_payload_hash_verified": True,
        "acoustic_selection_binding": selection_binding,
        "acoustic_selection_binding_sha256": selection_binding_sha256,
        "acoustic_selection_mode": selection_mode,
        "external_inputs": dict(external_inputs),
        "files": {
            name: {"path": name, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }


def _cache_only_execution_evidence(
    *,
    sample_count: int,
    cache_load_count: int,
    dynamic_convolution_count: int,
    persisted_mix_verification_count: int,
) -> dict[str, Any]:
    """Close the deliberately native- and visual-runtime-free control flow."""

    observed = (
        cache_load_count,
        dynamic_convolution_count,
        persisted_mix_verification_count,
    )
    if any(value != sample_count for value in observed):
        raise AssetBoundAudioError(
            "cache-only execution counters do not close over every sample"
        )
    native_call_receipts: tuple[Any, ...] = ()
    visual_call_receipts: tuple[Any, ...] = ()
    return {
        "mode": "validated_completed_rir_cache_audio_assembly_only",
        "allowed_control_flow": list(CACHE_ONLY_CONTROL_FLOW),
        "rir_cache_load_count": cache_load_count,
        "dynamic_convolution_count": dynamic_convolution_count,
        "persisted_mix_verification_count": persisted_mix_verification_count,
        "native_rlr_renderer_selected_or_constructed_by_tool": False,
        "visual_renderer_selected_or_constructed_by_tool": False,
        "native_rlr_call_receipts": list(native_call_receipts),
        "visual_render_call_receipts": list(visual_call_receipts),
        "native_rlr_calls": len(native_call_receipts),
        "visual_render_calls": len(visual_call_receipts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--rir-cache", type=Path, required=True)
    parser.add_argument("--class-audio", action="append", required=True)
    parser.add_argument("--class-channel-policy", action="append", required=True)
    parser.add_argument("--class-linear-gain", action="append", required=True)
    parser.add_argument("--fade-samples", type=int, default=80)
    parser.add_argument("--maximum-mixture-peak", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    plan_root = args.plan_root.resolve()
    rir_cache = args.rir_cache.resolve()
    assignments_path = plan_root / "sound_assignments.json"
    assignments, required_classes = _assignments(assignments_path)
    plan_closure = _plan_closure(plan_root, assignments)
    audio = _mapping(args.class_audio, owner="class-audio")
    policies = _mapping(args.class_channel_policy, owner="class-channel-policy")
    gains = _gains(args.class_linear_gain)
    for owner, values in (("audio", audio), ("channel policy", policies), ("gain", gains)):
        if set(values) != required_classes:
            raise AssetBoundAudioError(
                f"class {owner} keys differ; required={sorted(required_classes)}, got={sorted(values)}"
            )
    if not 0.0 < args.maximum_mixture_peak <= 1.0:
        raise AssetBoundAudioError("maximum mixture peak must be in (0,1]")
    prepared = {
        sound_class: prepare_dry_audio(
            audio[sound_class],
            channel_policy=policies[sound_class],
            linear_gain=gains[sound_class],
            fade_samples=args.fade_samples,
        )
        for sound_class in sorted(required_classes)
    }
    policy, output, staging = _publication_paths(args.output)
    phase_seconds: defaultdict[str, float] = defaultdict(float)
    samples = []
    partitions: dict[tuple[int, ...], np.ndarray] = {}
    shared_rir_shards: dict[Path, dict[str, Any]] = {}
    cache_load_count = 0
    dynamic_convolution_count = 0
    persisted_mix_verification_count = 0
    try:
        staging.mkdir()
        mixture_root = staging / "audio" / "binaural"
        stem_root = mixture_root / "stems"
        session = RIRCacheSession(
            cache_root=rir_cache,
            plan_path=plan_root / "rir_job_plan.json",
            frame_count=FRAME_COUNT,
            frame_rate_hz=FRAME_RATE_HZ,
            shared_shard_cache=shared_rir_shards,
        )
        cache_closure = _cache_closure(
            rir_cache,
            session,
            rir_plan_sha256=plan_closure["files"]["rir_job_plan.json"][
                "sha256"
            ],
            unique_rir_job_count=plan_closure["unique_rir_job_count"],
        )
        input_closure = dict(plan_closure)
        input_closure["rir_cache"] = cache_closure
        acoustic_selection_binding = cache_closure[
            "acoustic_selection_binding"
        ]
        acoustic_selection_binding_sha256 = cache_closure[
            "acoustic_selection_binding_sha256"
        ]
        input_closure["acoustic_selection_binding"] = (
            acoustic_selection_binding
        )
        input_closure["acoustic_selection_binding_sha256"] = (
            acoustic_selection_binding_sha256
        )
        input_closure["producer_identity"] = _producer_identity()
        for ordinal, assignment in enumerate(assignments):
            episode_id = _safe_episode_id(
                assignment["episode_id"],
                owner="render assignment",
            )
            classes = assignment["source_classes"]
            cache_started = time.perf_counter()
            cached = session.load_episode(episode_id)
            phase_seconds["rir_cache_load"] += time.perf_counter() - cache_started
            cache_load_count += 1
            if (
                cached.layout_type != "binaural"
                or cached.sample_rate_hz != M5_AUDIO_SAMPLE_RATE_HZ
                or cached.channel_labels != ("left", "right")
                or cached.source_slot_ids != SOURCE_SLOTS
            ):
                raise AssetBoundAudioError("RIR cache is not 16 kHz binaural source1/source2")
            if (
                cached.evidence.get("acoustic_selection_binding")
                != acoustic_selection_binding
            ):
                raise AssetBoundAudioError(
                    "episode RIR acoustic selection differs from its session"
                )
            partition_key = tuple(cached.keyframe_samples)
            weights = partitions.get(partition_key)
            if weights is None:
                weights = raised_cosine_partition(partition_key, M5_AUDIO_SAMPLE_COUNT)
                partitions[partition_key] = weights
            render_started = time.perf_counter()
            stems, _ = render_asset_bound_binaural(
                {slot: prepared[str(classes[slot])].samples for slot in SOURCE_SLOTS},
                rir_samples=cached.samples,
                rir_lengths=cached.lengths,
                source_ids=cached.source_slot_ids,
                keyframe_samples=cached.keyframe_samples,
                partition_weights=weights,
            )
            stored_stems, mixture = float32_stems_and_exact_mix(
                stems, source_ids=SOURCE_SLOTS
            )
            phase_seconds["dynamic_convolution"] += time.perf_counter() - render_started
            dynamic_convolution_count += 1
            sample_id = f"{episode_id}__v00"
            peak = float(np.max(np.abs(mixture)))
            if not np.isfinite(peak) or peak > args.maximum_mixture_peak:
                raise AssetBoundAudioError(
                    f"{episode_id} peak {peak:.6f} exceeds {args.maximum_mixture_peak:.6f}"
                )
            write_started = time.perf_counter()
            stem_records: dict[str, Any] = {}
            stem_readbacks: dict[str, np.ndarray] = {}
            for slot in SOURCE_SLOTS:
                record, readback = _write_and_readback(
                    stem_root / slot / f"{sample_id}.wav",
                    stored_stems[slot],
                    root=staging,
                    role="room_evaluation_binaural_source_stem",
                    metadata={
                        "sample_id": sample_id,
                        "episode_id": episode_id,
                        "source_slot_id": slot,
                        "sound_class": str(classes[slot]),
                        "acoustic_selection_binding_sha256": (
                            acoustic_selection_binding_sha256
                        ),
                    },
                )
                stem_records[slot] = record
                stem_readbacks[slot] = readback
            mixture_record, mixture_readback = _write_and_readback(
                mixture_root / f"{sample_id}.wav",
                mixture,
                root=staging,
                role="room_evaluation_binaural_mixture",
                metadata={
                    "sample_id": sample_id,
                    "episode_id": episode_id,
                    "source_classes": dict(classes),
                    "mixture": "exact_persisted_source1_plus_source2_stem_sum",
                    "normalization": False,
                    "limiting": False,
                    "acoustic_selection_binding_sha256": (
                        acoustic_selection_binding_sha256
                    ),
                },
            )
            stem_peaks = _verify_persisted_exact_mix(
                stem_readbacks,
                mixture_readback,
                sample_id=sample_id,
            )
            persisted_mix_verification_count += 1
            phase_seconds["wave_write_and_readback"] += time.perf_counter() - write_started
            samples.append(
                {
                    "sample_id": sample_id,
                    "episode_id": episode_id,
                    "ordinal": ordinal,
                    "split": "test",
                    "both_sources_active": True,
                    "source_classes": dict(classes),
                    "audio_path": mixture_record["audio_path"],
                    "audio_sidecar_path": mixture_record[
                        "audio_sidecar_path"
                    ],
                    "audio_sha256": mixture_record["audio_sha256"],
                    "audio_sidecar_sha256": mixture_record[
                        "sidecar_sha256"
                    ],
                    "audio_sample_rate_hz": M5_AUDIO_SAMPLE_RATE_HZ,
                    "audio_sample_count": M5_AUDIO_SAMPLE_COUNT,
                    "audio_channel_count": 2,
                    "peak_absolute": mixture_record["peak_absolute"],
                    "source_stems": stem_records,
                    "source_stem_peak_absolute": stem_peaks,
                    "mixture_is_exact_persisted_source_stem_sum": True,
                    "acoustic_selection_binding_sha256": (
                        acoustic_selection_binding_sha256
                    ),
                }
            )
        execution_evidence = _cache_only_execution_evidence(
            sample_count=len(samples),
            cache_load_count=cache_load_count,
            dynamic_convolution_count=dynamic_convolution_count,
            persisted_mix_verification_count=persisted_mix_verification_count,
        )
        write_json(staging / "input_closure.json", input_closure)
        write_json(
            staging / "dry_audio_classes.json",
            {
                "schema": "avengine_room_evaluation_dry_audio_classes_v1",
                "status": "pass",
                "classes": {
                    key: prepared[key].record for key in sorted(prepared)
                },
            },
        )
        write_json(
            staging / "samples.json",
            {
                "schema": "avengine_room_evaluation_binaural_samples_v1",
                "status": "pass",
                "sample_count": len(samples),
                "acoustic_selection_binding": acoustic_selection_binding,
                "acoustic_selection_binding_sha256": (
                    acoustic_selection_binding_sha256
                ),
                "input_closure": input_closure,
                "samples": samples,
            },
        )
        wall_seconds = time.perf_counter() - started
        write_json(
            staging / "timing.json",
            {
                "schema": "avengine_room_evaluation_binaural_timing_v1",
                "status": "pass",
                "wall_seconds": wall_seconds,
                "phase_seconds": dict(phase_seconds),
                "sample_count": len(samples),
                "samples_per_second": len(samples) / wall_seconds,
                "execution_evidence": execution_evidence,
                "native_rlr_calls": execution_evidence[
                    "native_rlr_calls"
                ],
                "visual_render_calls": execution_evidence[
                    "visual_render_calls"
                ],
                "rir_cache_reused": True,
            },
        )
        output_closure = _output_closure(
            staging,
            sample_count=len(samples),
        )
        write_json(
            staging / "delivery.json",
            {
                "schema": SCHEMA,
                "status": "pass",
                "research_only": True,
                "qualification_claim": False,
                "sample_count": len(samples),
                "both_sources_active": True,
                "source_slots": list(SOURCE_SLOTS),
                "acoustic_selection_binding": acoustic_selection_binding,
                "acoustic_selection_binding_sha256": (
                    acoustic_selection_binding_sha256
                ),
                "sound_classes_are_asset_independent": True,
                "input_closure": input_closure,
                "mixture_is_exact_persisted_source_stem_sum": True,
                "layout": "native_RLR_HRTF_binaural_left_right",
                "output_closure": output_closure,
                "outputs": {
                    "input_closure": "input_closure.json",
                    "samples": "samples.json",
                    "dry_audio_classes": "dry_audio_classes.json",
                    "timing": "timing.json",
                    "mixtures": "audio/binaural/",
                    "stems": "audio/binaural/stems/",
                },
            },
        )
        output = atomic_publish_directory(policy, staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"ROOM_EVALUATION_AUDIO_OK output={output} samples={len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

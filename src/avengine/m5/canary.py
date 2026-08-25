"""Executable two-actor M5 canary and self-contained evidence bundle."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.contracts.transforms import normalized_quaternion_xyzw
from avengine.m1.habitat_capture import (
    InstalledHabitatRuntime,
    prepare_installed_habitat_runtime,
)
from avengine.m3.runtime import (
    RUNTIME_MODE_CURRENT_INSTALLED,
    RuntimeUnavailableError,
    load_compiled_acoustic_scene,
)
from avengine.spatial_audio.audio import read_float32_wav, write_float32_wav
from avengine.spatial_audio.runtime import M4SimulationConfig
from avengine.m5.acoustics import (
    AcousticKeyframe,
    DynamicRIRSequence,
    render_dynamic_rir_sequence,
    trajectory_record,
)
from avengine.m5.audio import (
    M5_AUDIO_SAMPLE_COUNT,
    M5_AUDIO_SAMPLE_RATE_HZ,
    extract_faded_clip,
    place_simultaneous_events,
    read_pcm16_mono_wav,
    render_dynamic_stems_and_mix,
)
from avengine.m5.metrics import (
    listener_local_source_geometry,
    measure_binaural_mixture_diagnostic,
    measure_binaural_rir_sequence_cues,
    measure_binaural_wet_stem_cues,
    summarize_lateral_cue_consistency,
)
from avengine.m5.timeline import (
    build_counterfactual_pair,
    compare_counterfactual_pair,
    validate_episode_request,
)
from avengine.m5.video import (
    aac_decode_diagnostics,
    compose_main_topdown_frames,
    encode_h264_base_video,
    encode_h264_qa_base_video,
    mux_binaural_wav,
    mux_qa_binaural_wav,
    probe_episode_video,
    probe_qa_review_video,
    video_packet_sha256,
)
from avengine.m5.visual import TwoActorVisualResult, capture_two_actor_fixed_states


M5_EVIDENCE_SCHEMA = "avengine_m5_canary_evidence_v1"
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class M5CanaryError(RuntimeError):
    """The M5 canary could not prove its declared audiovisual contract."""


def _audio_program_execution(
    request: Mapping[str, Any],
) -> tuple[int, int, int, float, tuple[tuple[int, int], ...]]:
    """Return the already-validated request-owned dry/schedule controls."""

    program = request["audio_program"]
    clip = program["clip_source_interval"]
    windows = tuple(
        (int(window["start_sample"]), int(window["end_sample"]))
        for window in program["simultaneous_windows"]
    )
    return (
        int(clip["start_sample"]),
        int(clip["end_sample"]),
        int(program["fade_samples"]),
        float(program["linear_gain"]),
        windows,
    )


def _save_npy(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(np.asarray(value))
    np.save(path, array, allow_pickle=False)
    readback = np.load(path, mmap_mode="r", allow_pickle=False)
    if readback.shape != array.shape or readback.dtype != array.dtype:
        raise M5CanaryError(f"NPY readback differs for {path.name}")
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _named_emitter_path_sha256(source_id: str, positions_m: np.ndarray) -> str:
    return canonical_json_sha256(
        {
            "schema": "avengine_m5_named_emitter_path_v1",
            "source_id": source_id,
            "positions_m": np.asarray(positions_m, dtype=np.float64).tolist(),
        }
    )


def _listener_pose_at_frame(
    visual: TwoActorVisualResult,
    frame_index: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    trajectory = visual.sensor_rig_trajectory
    if trajectory is None:
        return visual.listener_position_m, visual.listener_orientation_wxyz
    try:
        frame = trajectory["frames"][frame_index]
        transform = frame["world_from_rig"]
        position = tuple(float(value) for value in transform["translation_m"])
        x, y, z, w = normalized_quaternion_xyzw(transform["rotation_xyzw"])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise M5CanaryError(
            f"sensor-rig trajectory frame {frame_index} is malformed"
        ) from exc
    return position, (float(w), float(x), float(y), float(z))


def _visual_timeline_frames(
    visual: TwoActorVisualResult,
    m2_request: Mapping[str, Any],
    m1_request: Mapping[str, Any],
) -> list[dict[str, Any]]:
    offsets = np.asarray(visual.metadata["actor_offsets_m"], dtype=np.float64)
    if visual.sensor_rig_trajectory is None:
        camera_hashes = [
            canonical_json_sha256(
                {
                    "view_id": "view0",
                    "primary_camera_rig": m1_request["primary_camera_rig"],
                }
            )
        ] * 75
    else:
        camera_hashes = [
            str(frame["pose_hash"])
            for frame in visual.sensor_rig_trajectory["frames"]
        ]
    result: list[dict[str, Any]] = []
    states = m2_request["states"]
    if len(states) != 75:
        raise M5CanaryError("source M2 request does not contain 75 states")
    for frame_index, state in enumerate(states):
        actor_states = []
        contacts = {
            item["contact_id"]: bool(item["in_contact"])
            for item in state["contact_states"]
        }
        for actor_index, actor_id in enumerate(visual.actor_ids):
            root = deepcopy(state["root_transform"])
            root["translation_m"] = (
                np.asarray(root["translation_m"], dtype=np.float64)
                + offsets[actor_index]
            ).tolist()
            root["scale"] = [1.0, 1.0, 1.0]
            actor_states.append(
                {
                    "actor_id": actor_id,
                    "root_transform": root,
                    "action_id": state["action_id"],
                    "action_time_ticks": int(state["action_time_ticks"]),
                    "action_phase": float((frame_index % 25) / 25.0),
                    "pose_hash": canonical_json_sha256(
                        {
                            "actor_id": actor_id,
                            "source_pose_hash": state["pose_hash"],
                            "instance_offset_m": offsets[actor_index].tolist(),
                        }
                    ),
                    "contacts": contacts,
                }
            )
        result.append(
            {
                "actor_states": actor_states,
                "view_pose_hashes": {"view0": camera_hashes[frame_index]},
            }
        )
    return result


def _acoustic_keyframes(visual: TwoActorVisualResult) -> tuple[AcousticKeyframe, ...]:
    result: list[AcousticKeyframe] = []
    for frame_index in range(75):
        listener_position, listener_orientation = _listener_pose_at_frame(
            visual, frame_index
        )
        result.append(
            AcousticKeyframe(
                tick=3_200 * frame_index,
                sample_index=(3_200 * frame_index + 1) // 3,
                source_positions_m={
                    source_id: tuple(
                        float(value)
                        for value in visual.source_positions_m[
                            frame_index, source_index
                        ]
                    )
                    for source_index, source_id in enumerate(visual.source_ids)
                },
                listener_position_m=listener_position,
                listener_orientation_wxyz=listener_orientation,
            )
        )
    return tuple(result)


def _write_rir_sequence(
    root: Path, name: str, sequence: DynamicRIRSequence
) -> dict[str, Any]:
    directory = root / "rir" / name
    samples_path = directory / "samples.npy"
    lengths_path = directory / "lengths.npy"
    metadata_path = directory / "metadata.json"
    samples = _save_npy(samples_path, sequence.samples)
    lengths = _save_npy(lengths_path, sequence.lengths)
    write_json(metadata_path, dict(sequence.metadata))
    return {
        "samples_path": str(samples_path.relative_to(root)),
        "lengths_path": str(lengths_path.relative_to(root)),
        "metadata_path": str(metadata_path.relative_to(root)),
        "samples": samples,
        "lengths": lengths,
        "trajectory_sha256": sequence.trajectory_sha256,
        "source_ids": list(sequence.source_ids),
        "layout_type": sequence.layout_type,
        "layout_id": sequence.layout_id,
        "channel_labels": list(sequence.channel_labels),
        "sample_rate_hz": sequence.sample_rate_hz,
    }


def _write_audio(
    path: Path,
    samples: np.ndarray,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = write_float32_wav(
        path,
        samples,
        M5_AUDIO_SAMPLE_RATE_HZ,
        channel_axis=0,
        metadata=metadata,
    )
    decoded = read_float32_wav(
        artifact.audio_path,
        sidecar_path=artifact.sidecar_path,
        verify_sidecar=True,
    )
    if decoded.frame_count != M5_AUDIO_SAMPLE_COUNT:
        raise M5CanaryError("M5 authoritative WAVE is not exactly 80,000 samples")
    return {
        "audio_path": str(artifact.audio_path),
        "sidecar_path": str(artifact.sidecar_path),
        "audio_sha256": artifact.audio_sha256,
        "sidecar_sha256": artifact.sidecar_sha256,
        "sample_rate_hz": artifact.sample_rate_hz,
        "sample_count": artifact.frame_count,
        "channel_count": artifact.channel_count,
        "peak_absolute": float(np.max(np.abs(decoded.samples))),
    }


def _spatial_metrics(
    visual: TwoActorVisualResult,
    binaural: DynamicRIRSequence,
    stems: Mapping[str, Any],
    mixture: np.ndarray,
    *,
    active_windows: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    if not active_windows:
        raise M5CanaryError("spatial metrics require declared active windows")
    active_start = int(active_windows[0][0])
    active_end = int(active_windows[-1][1])
    per_source: dict[str, Any] = {}
    median_ild: dict[str, float] = {}
    median_itd: dict[str, float] = {}
    median_azimuth: dict[str, float] = {}
    lateral_frames: list[Mapping[str, Any]] = []
    lateral_azimuths: list[float] = []
    for source_index, source_id in enumerate(binaural.source_ids):
        rir_report = measure_binaural_rir_sequence_cues(
            binaural.samples[:, source_index], binaural.sample_rate_hz
        )
        geometries = []
        for frame_index in range(75):
            listener_position, listener_orientation = _listener_pose_at_frame(
                visual, frame_index
            )
            geometries.append(
                listener_local_source_geometry(
                    visual.source_positions_m[frame_index, source_index],
                    listener_position,
                    listener_orientation,
                )
            )
        azimuths = [float(item["azimuth_deg"]) for item in geometries]
        for frame, geometry in zip(
            rir_report["frames"], geometries, strict=True
        ):
            frame["listener_local_azimuth_deg"] = geometry["azimuth_deg"]
            frame["listener_local_elevation_deg"] = geometry["elevation_deg"]
            frame["listener_local_unit_direction_xyz"] = geometry[
                "unit_direction_xyz"
            ]
            frame["source_listener_distance_m"] = geometry["distance_m"]
        lateral_frames.extend(rir_report["frames"])
        lateral_azimuths.extend(azimuths)
        wet_report = measure_binaural_wet_stem_cues(
            stems[source_id].episode,
            binaural.sample_rate_hz,
            active_start,
            active_end,
            source_id=source_id,
        )
        ilds = [float(frame["ild_db"]) for frame in rir_report["frames"]]
        itds = [float(frame["itd"]["itd_seconds"]) for frame in rir_report["frames"]]
        median_ild[source_id] = float(np.median(ilds))
        median_itd[source_id] = float(np.median(itds))
        median_azimuth[source_id] = float(np.median(azimuths))
        per_source[source_id] = {
            "azimuth_range_deg": [float(min(azimuths)), float(max(azimuths))],
            "rir_direct_window": rir_report,
            "wet_stem_active_window": wet_report,
        }
    mixture_report = measure_binaural_mixture_diagnostic(
        mixture,
        binaural.sample_rate_hz,
        active_start,
        active_end,
    )
    maximum_itd = max(
        abs(float(frame["itd"]["itd_seconds"]))
        for source in per_source.values()
        for frame in source["rir_direct_window"]["frames"]
    )
    lateral_consistency = summarize_lateral_cue_consistency(
        lateral_frames,
        listener_local_azimuths_deg=lateral_azimuths,
    )
    right_source = max(median_azimuth, key=median_azimuth.__getitem__)
    left_source = min(median_azimuth, key=median_azimuth.__getitem__)
    checks = {
        "all_source_itd_within_1ms": maximum_itd <= 0.0010000001,
        "lateral_cue_consistency": lateral_consistency["status"] == "pass"
        and lateral_consistency["formal_acceptance_allowed"] is True,
        "right_source_has_more_negative_ild_than_left_source": (
            right_source != left_source
            and median_ild[right_source] < median_ild[left_source]
        ),
        "mixture_is_diagnostic_only": mixture_report.get("diagnostic_only") is True,
    }
    return {
        "schema": "avengine_m5_spatial_metrics_bundle_v1",
        "sign_convention": {
            "ild": "10_log10_energy_left_over_right",
            "ipd": "angle_left_times_conjugate_right_frequency_dependent",
            "itd": "t_left_minus_t_right",
            "azimuth": "listener_local_positive_right_degrees",
        },
        "formal_acceptance_scope": "per_source_retained_stems_and_rirs",
        "per_source": per_source,
        "lateral_cue_consistency": lateral_consistency,
        "mixture_diagnostic": mixture_report,
        "summary": {
            "median_ild_db_by_source": median_ild,
            "median_itd_seconds_by_source": median_itd,
            "median_azimuth_deg_by_source": median_azimuth,
            "right_source_id": right_source,
            "left_source_id": left_source,
            "maximum_absolute_itd_seconds": maximum_itd,
            "gcc_boundary_rejected_frame_count": lateral_consistency["counts"][
                "gcc_boundary_rejected_frames"
            ],
        },
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def _qa_text_by_frame(
    metrics: Mapping[str, Any],
    *,
    active_windows: Sequence[tuple[int, int]],
) -> list[str]:
    source_ids = tuple(metrics["per_source"])
    if len(source_ids) != 2:
        raise M5CanaryError("QA overlay requires exactly two retained sources")
    source0 = metrics["per_source"][source_ids[0]]["rir_direct_window"]["frames"]
    source1 = metrics["per_source"][source_ids[1]]["rir_direct_window"]["frames"]
    result: list[str] = []
    for index, (left, right) in enumerate(zip(source0, source1, strict=True)):
        rows = []
        for source_id, frame in zip(source_ids, (left, right), strict=True):
            ipd = frame["ipd_radians_by_frequency_hz"].get("500")
            rows.append(
                f"{source_id} az={frame['listener_local_azimuth_deg']:+.1f}deg "
                f"ILD={frame['ild_db']:+.1f}dB IPD500={float(ipd):+.2f}rad "
                f"ITD={frame['itd']['itd_seconds'] * 1e6:+.0f}us"
            )
        sample = (3_200 * index + 1) // 3
        active = any(start <= sample < end for start, end in active_windows)
        result.append(
            ("SIMULTANEOUS BARKS\n" if active else "RIR POSITION\n") + "\n".join(rows)
        )
    return result


def _artifact_index(root: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "evidence.json":
            continue
        relative = path.relative_to(root).as_posix()
        artifacts[relative] = file_record(path, relative_to=root)
    return artifacts


def _portable_output_paths(value: Any, root: Path) -> Any:
    """Replace staging-root absolute paths with bundle-relative paths."""

    if isinstance(value, Mapping):
        return {key: _portable_output_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_output_paths(item, root) for item in value]
    if isinstance(value, tuple):
        return [_portable_output_paths(item, root) for item in value]
    if isinstance(value, str):
        try:
            path = Path(value)
            if path.is_absolute():
                return path.resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            pass
    return value


def _code_provenance() -> dict[str, Any]:
    """Bind a formal run to one clean AVEngine source revision."""

    repository = Path(__file__).resolve().parents[3]

    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise M5CanaryError(f"cannot bind M5 evidence to Git: {exc}") from exc
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if not _GIT_COMMIT_RE.fullmatch(commit):
        raise M5CanaryError("Git returned a non-canonical commit identity")
    status = git("status", "--porcelain=v1", "--untracked-files=normal")
    return {
        "repository_role": "avengine_habitat_native",
        "commit": commit,
        "worktree_clean": status == "",
        "capture_phase": "before_staging_directory_creation",
    }


def _confined_bundle_path(root: Path, value: Any, *, owner: str) -> Path:
    """Resolve one POSIX bundle path while rejecting absolute/traversal paths."""

    if not isinstance(value, str) or not value or "\\" in value:
        raise M5CanaryError(f"{owner} is not a portable bundle-relative path")
    portable = PurePosixPath(value)
    if portable.is_absolute() or any(
        part in {"", ".", ".."} for part in portable.parts
    ):
        raise M5CanaryError(f"{owner} is not a confined bundle-relative path")
    candidate = (root / Path(*portable.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise M5CanaryError(f"{owner} escapes the evidence bundle") from exc
    if candidate.is_symlink():
        raise M5CanaryError(f"{owner} must not be a symlink")
    return candidate


def _rir_authority(
    root: Path, evidence: Mapping[str, Any]
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, Mapping[str, Any]]],
    dict[str, Any],
    list[str],
]:
    """Read and independently validate retained RIR arrays and trajectory."""

    errors: list[str] = []
    result: dict[str, tuple[np.ndarray, np.ndarray, Mapping[str, Any]]] = {}
    trajectory: dict[str, Any] = {}
    try:
        trajectory = load_json(root / "trajectory" / "emitter_path.json")
        declared = trajectory.get("trajectory_content_sha256")
        content = dict(trajectory)
        content.pop("trajectory_content_sha256", None)
        recomputed = canonical_json_sha256(content)
        if declared != recomputed:
            errors.append("trajectory content hash differs")
        keys = content.get("keyframes")
        if not isinstance(keys, list) or len(keys) != 75:
            errors.append("trajectory does not contain 75 keyframes")
        elif [item.get("sample_index") for item in keys] != [
            (3_200 * index + 1) // 3 for index in range(75)
        ]:
            errors.append("trajectory sample grid differs from the rational 15 Hz grid")
    except Exception as exc:
        errors.append(f"trajectory readback failed: {exc}")
        return result, trajectory, errors

    records = evidence.get("rir_sequences")
    if not isinstance(records, Mapping):
        return result, trajectory, errors + ["RIR evidence mapping is absent"]
    trajectory_hash = canonical_json_sha256(
        {
            key: value
            for key, value in trajectory.items()
            if key != "trajectory_content_sha256"
        }
    )
    input_records = evidence.get("inputs")
    hrtf_record = (
        input_records.get("hrtf") if isinstance(input_records, Mapping) else None
    )
    hrtf_hash: str | None = None
    if isinstance(hrtf_record, Mapping):
        try:
            hrtf_file = _confined_bundle_path(
                root, hrtf_record.get("path"), owner="inputs.hrtf.path"
            )
            hrtf_hash = sha256_file(hrtf_file)
            if hrtf_hash != hrtf_record.get("sha256"):
                errors.append("retained HRTF hash differs from its input record")
        except Exception as exc:
            errors.append(f"HRTF readback failed: {exc}")
    else:
        errors.append("HRTF input record is absent")

    expected = {
        "foa": ("ambisonics", 4, "rlr_foa_acn_n3d_world_v1", ["W", "Y", "Z", "X"]),
        "binaural": ("binaural", 2, "rlr_binaural_lr_v1", ["left", "right"]),
    }
    for name, (layout_type, channels, layout_id, labels) in expected.items():
        record = records.get(name)
        if not isinstance(record, Mapping):
            errors.append(f"{name} RIR record is absent")
            continue
        try:
            samples_path = _confined_bundle_path(
                root,
                record.get("samples_path"),
                owner=f"rir_sequences.{name}.samples_path",
            )
            lengths_path = _confined_bundle_path(
                root,
                record.get("lengths_path"),
                owner=f"rir_sequences.{name}.lengths_path",
            )
            metadata_path = _confined_bundle_path(
                root,
                record.get("metadata_path"),
                owner=f"rir_sequences.{name}.metadata_path",
            )
            samples = np.load(samples_path, allow_pickle=False)
            lengths = np.load(lengths_path, allow_pickle=False)
            metadata = load_json(metadata_path)
            if (
                samples.ndim != 4
                or samples.shape[:3] != (75, 2, channels)
                or samples.dtype != np.dtype("<f4")
                or not np.all(np.isfinite(samples))
            ):
                errors.append(f"{name} RIR samples violate [75,2,{channels},L] float32")
                continue
            if (
                lengths.shape != (75, 2)
                or lengths.dtype.kind != "u"
                or np.any(lengths < 2)
                or np.any(lengths > samples.shape[3])
            ):
                errors.append(f"{name} RIR lengths are invalid")
                continue
            hashes = metadata.get("ir_sha256_by_frame_source")
            if not isinstance(hashes, list) or len(hashes) != 75:
                errors.append(f"{name} per-RIR hashes are absent")
                continue
            for frame_index in range(75):
                for source_index, source_id in enumerate(("source0", "source1")):
                    length = int(lengths[frame_index, source_index])
                    unpadded = np.ascontiguousarray(
                        samples[frame_index, source_index, :, :length], dtype="<f4"
                    )
                    observed = hashlib.sha256(unpadded.tobytes(order="C")).hexdigest()
                    if hashes[frame_index].get(source_id) != observed:
                        errors.append(
                            f"{name} RIR hash differs at {frame_index}/{source_id}"
                        )
                    if np.any(samples[frame_index, source_index, :, length:] != 0.0):
                        errors.append(
                            f"{name} RIR padding is nonzero at {frame_index}/{source_id}"
                        )
            expected_metadata = {
                "trajectory_sha256": trajectory_hash,
                "source_ids": ["source0", "source1"],
                "layout_type": layout_type,
                "layout_id": layout_id,
                "channel_labels": labels,
                "sample_rate_hz": M5_AUDIO_SAMPLE_RATE_HZ,
            }
            for key, value in expected_metadata.items():
                if metadata.get(key) != value or record.get(key) != value:
                    errors.append(f"{name} RIR {key} differs")
            if "wall_seconds" in metadata:
                errors.append(f"{name} RIR retains nondeterministic wall_seconds")
            if name == "foa" and metadata.get("hrtf") is not None:
                errors.append("FOA RIR metadata unexpectedly names an HRTF")
            if name == "binaural":
                retained_hrtf = metadata.get("hrtf")
                if (
                    not isinstance(retained_hrtf, Mapping)
                    or retained_hrtf.get("input_role") != "hrtf"
                    or retained_hrtf.get("sha256") != hrtf_hash
                ):
                    errors.append(
                        "binaural RIR HRTF role/hash differs from retained input"
                    )
                for receipt in metadata.get("endpoint_receipts", []):
                    listener = (
                        receipt.get("listener")
                        if isinstance(receipt, Mapping)
                        else None
                    )
                    if (
                        not isinstance(listener, Mapping)
                        or listener.get("hrtf_file_path") != "input-role:hrtf"
                    ):
                        errors.append(
                            "binaural RIR receipt retains an invalid HRTF path"
                        )
                        break
            result[name] = (samples, lengths, metadata)
        except Exception as exc:
            errors.append(f"{name} RIR readback failed: {exc}")
    if len(result) == 2 and result["foa"][2].get("trajectory_sha256") != result[
        "binaural"
    ][2].get("trajectory_sha256"):
        errors.append("FOA and binaural RIR trajectories differ")
    return result, trajectory, errors


def _audio_reconstruction_errors(
    root: Path,
    evidence: Mapping[str, Any],
    rir: Mapping[str, tuple[np.ndarray, np.ndarray, Mapping[str, Any]]],
    trajectory: Mapping[str, Any],
) -> list[str]:
    """Rebuild dry buses, wet stems and mixtures without trusting declared checks."""

    errors: list[str] = []
    if set(rir) != {"foa", "binaural"}:
        return ["RIR authority is incomplete; audio reconstruction was not attempted"]
    inputs = evidence.get("inputs")
    audio = evidence.get("audio")
    if not isinstance(inputs, Mapping) or not isinstance(audio, Mapping):
        return ["audio or input evidence mapping is absent"]
    try:
        pair = load_json(root / "episodes" / "counterfactual_pair.json")
        base_request = pair["episodes"]["A"]["request"]
        (
            dry_clip_start,
            dry_clip_end,
            dry_fade_samples,
            dry_linear_gain,
            active_windows,
        ) = _audio_program_execution(base_request)
        event_starts = tuple(start for start, _end in active_windows)
        source_ids = tuple(trajectory.get("source_ids", ()))
        if len(source_ids) != 2:
            raise M5CanaryError("trajectory must retain exactly two source IDs")
        raw_assets: dict[str, tuple[str, np.ndarray]] = {}
        for asset_id, role in (
            ("beagle_call", "beagle_dry"),
            ("golden_call", "golden_dry"),
        ):
            record = inputs.get(role)
            if not isinstance(record, Mapping):
                raise M5CanaryError(f"input role {role} is absent")
            source_path = _confined_bundle_path(
                root, record.get("path"), owner=f"inputs.{role}.path"
            )
            samples, rate = read_pcm16_mono_wav(source_path)
            if rate != M5_AUDIO_SAMPLE_RATE_HZ:
                raise M5CanaryError(f"input role {role} has the wrong sample rate")
            raw_assets[sha256_file(source_path)] = (
                asset_id,
                extract_faded_clip(
                    samples,
                    start_sample=dry_clip_start,
                    end_sample=dry_clip_end,
                    fade_samples=dry_fade_samples,
                ),
            )
        keyframes = trajectory.get("keyframes")
        keyframe_samples = tuple(item["sample_index"] for item in keyframes)
        for variant in ("A", "B"):
            variant_request = pair["episodes"][variant]["request"]
            route = {
                event["source_id"]: raw_assets[event["dry_audio_asset_sha256"]][0]
                for event in variant_request["events"]
            }
            buses, _ = place_simultaneous_events(
                {asset_id: clip for asset_id, clip in raw_assets.values()},
                route,
                start_samples=event_starts,
                output_sample_count=M5_AUDIO_SAMPLE_COUNT,
                linear_gain=dry_linear_gain,
            )
            variant_records = audio.get(variant)
            if not isinstance(variant_records, Mapping):
                errors.append(f"audio record for episode {variant} is absent")
                continue
            for source_id in source_ids:
                record = variant_records.get("dry_buses", {}).get(source_id)
                path = _confined_bundle_path(
                    root,
                    record.get("audio_path"),
                    owner=f"audio.{variant}.dry.{source_id}",
                )
                decoded = read_float32_wav(path, verify_sidecar=True)
                if not np.array_equal(
                    decoded.samples[0], buses[source_id].astype("<f4")
                ):
                    errors.append(f"{variant}/{source_id} dry bus cannot be rebuilt")
            for layout, channels in (("foa", 4), ("binaural", 2)):
                rir_samples, rir_lengths, _ = rir[layout]
                stems, mixture = render_dynamic_stems_and_mix(
                    buses,
                    rir_samples,
                    rir_lengths,
                    source_ids=source_ids,
                    keyframe_samples=keyframe_samples,
                    output_sample_count=M5_AUDIO_SAMPLE_COUNT,
                )
                layout_records = variant_records.get(layout)
                if not isinstance(layout_records, Mapping):
                    errors.append(f"audio record for {variant}/{layout} is absent")
                    continue
                for source_id in ("source0", "source1"):
                    record = layout_records.get(source_id)
                    path = _confined_bundle_path(
                        root,
                        record.get("audio_path"),
                        owner=f"audio.{variant}.{layout}.{source_id}",
                    )
                    decoded = read_float32_wav(path, verify_sidecar=True)
                    expected = stems[source_id].episode.astype("<f4")
                    if decoded.samples.shape != (
                        channels,
                        M5_AUDIO_SAMPLE_COUNT,
                    ) or not np.array_equal(decoded.samples, expected):
                        errors.append(
                            f"{variant}/{layout}/{source_id} stem cannot be rebuilt"
                        )
                record = layout_records.get("mixture")
                path = _confined_bundle_path(
                    root,
                    record.get("audio_path"),
                    owner=f"audio.{variant}.{layout}.mixture",
                )
                decoded = read_float32_wav(path, verify_sidecar=True)
                if decoded.samples.shape != (
                    channels,
                    M5_AUDIO_SAMPLE_COUNT,
                ) or not np.array_equal(decoded.samples, mixture.astype("<f4")):
                    errors.append(f"{variant}/{layout}/mixture cannot be rebuilt")
    except Exception as exc:
        errors.append(f"audio reconstruction failed: {exc}")
    return errors


def _spatial_metric_errors(root: Path, evidence: Mapping[str, Any]) -> list[str]:
    """Reject declared spatial passes that crossed an estimator boundary."""

    errors: list[str] = []
    summaries = evidence.get("spatial_metrics")
    for variant in ("A", "B"):
        try:
            metrics = load_json(root / "episodes" / variant / "spatial_metrics.json")
            retained = (
                summaries.get(variant) if isinstance(summaries, Mapping) else None
            )
            if metrics.get("status") != "pass" or not isinstance(retained, Mapping):
                errors.append(f"{variant} spatial metrics are not retained as pass")
                continue
            if retained.get("status") != "pass" or retained.get(
                "summary"
            ) != metrics.get("summary"):
                errors.append(
                    f"{variant} spatial summary/status differs from retained report"
                )
            mixture = metrics.get("mixture_diagnostic")
            if (
                retained.get("mixture_diagnostic_only") is not True
                or not isinstance(mixture, Mapping)
                or mixture.get("diagnostic_only") is not True
                or mixture.get("source_specific_acceptance_allowed") is not False
            ):
                errors.append(
                    f"{variant} mixture was promoted beyond diagnostic-only scope"
                )
            checks = metrics.get("checks")
            if (
                not isinstance(checks, Mapping)
                or not checks
                or not all(value is True for value in checks.values())
            ):
                errors.append(f"{variant} spatial checks are not all true")
            lateral_frames: list[Mapping[str, Any]] = []
            lateral_azimuths: list[float] = []
            for source_id in ("source0", "source1"):
                source = metrics.get("per_source", {}).get(source_id, {})
                frames = source.get("rir_direct_window", {}).get("frames", [])
                lateral_frames.extend(frames)
                lateral_azimuths.extend(
                    frame.get("listener_local_azimuth_deg") for frame in frames
                )
            recomputed = summarize_lateral_cue_consistency(
                lateral_frames,
                listener_local_azimuths_deg=lateral_azimuths,
            )
            if recomputed != metrics.get("lateral_cue_consistency"):
                errors.append(
                    f"{variant} lateral cue summary cannot be independently rebuilt"
                )
            if (
                recomputed.get("status") != "pass"
                or recomputed.get("formal_acceptance_allowed") is not True
            ):
                errors.append(f"{variant} lateral cue consistency does not pass")
            rejected = [
                frame
                for frame in recomputed.get("frames", [])
                if frame.get("gcc_at_search_boundary") is True
            ]
            if any(
                frame.get("itd_vote") != "rejected_search_boundary"
                for frame in rejected
            ) or recomputed.get("counts", {}).get(
                "gcc_boundary_rejected_frames"
            ) != len(rejected):
                errors.append(f"{variant} accepted a GCC-PHAT boundary ITD vote")
        except Exception as exc:
            errors.append(f"{variant} spatial metrics readback failed: {exc}")
    return errors


def _sensor_rig_evidence_errors(
    root: Path, evidence: Mapping[str, Any]
) -> list[str]:
    """Cross-bind a declared moving rig to visual, Timeline and RLR evidence."""

    visual = evidence.get("visual")
    metadata = visual.get("metadata") if isinstance(visual, Mapping) else None
    declaration = (
        metadata.get("sensor_rig_trajectory")
        if isinstance(metadata, Mapping)
        else None
    )
    inputs = evidence.get("inputs")
    input_record = (
        inputs.get("sensor_rig_trajectory")
        if isinstance(inputs, Mapping)
        else None
    )
    sidecar_path = root / "trajectory" / "sensor_rig_trajectory.json"
    presences = (
        declaration is not None,
        input_record is not None,
        sidecar_path.is_file(),
    )
    if not any(presences):
        return []
    presence_errors: list[str] = []
    if declaration is None:
        presence_errors.append("visual sensor-rig trajectory declaration is absent")
    elif not isinstance(declaration, Mapping):
        presence_errors.append("visual sensor-rig trajectory declaration is malformed")
    if not isinstance(input_record, Mapping):
        presence_errors.append("sensor-rig input record is absent or malformed")
    if not sidecar_path.is_file():
        presence_errors.append("retained sensor-rig sidecar is absent")
    if presence_errors:
        return presence_errors

    errors: list[str] = []
    try:
        from avengine.sensor_rig_trajectory import validate_sensor_rig_trajectory

        trajectory = load_json(sidecar_path)
        validation_errors = validate_sensor_rig_trajectory(trajectory)
        if validation_errors:
            return [
                "sensor-rig trajectory validation failed: "
                + "; ".join(validation_errors)
            ]
        if (
            declaration.get("trajectory_id") != trajectory["trajectory_id"]
            or declaration.get("schema") != trajectory["schema"]
            or declaration.get("content_sha256")
            != canonical_json_sha256(trajectory)
        ):
            errors.append("visual metadata does not bind the sensor-rig sidecar")
        input_path = _confined_bundle_path(
            root,
            input_record.get("path"),
            owner="inputs.sensor_rig_trajectory.path",
        )
        if load_json(input_path) != trajectory:
            errors.append("sensor-rig input differs from the retained sidecar")

        positions = np.asarray(
            [
                frame["world_from_rig"]["translation_m"]
                for frame in trajectory["frames"]
            ],
            dtype=np.float64,
        )
        orientations = np.asarray(
            [
                (
                    frame["world_from_rig"]["rotation_xyzw"][3],
                    frame["world_from_rig"]["rotation_xyzw"][0],
                    frame["world_from_rig"]["rotation_xyzw"][1],
                    frame["world_from_rig"]["rotation_xyzw"][2],
                )
                for frame in trajectory["frames"]
            ],
            dtype=np.float64,
        )
        declared_moving = bool(
            np.any(np.abs(positions - positions[0]) > 1.0e-12)
            or np.any(np.abs(orientations - orientations[0]) > 1.0e-12)
        )
        if declaration.get("moving") is not declared_moving:
            errors.append("visual metadata moving flag differs from the rig sidecar")

        retained_positions = np.load(
            root / "visual" / "arrays" / "listener_positions_m.npy",
            allow_pickle=False,
        )
        retained_orientations = np.load(
            root / "visual" / "arrays" / "listener_orientations_wxyz.npy",
            allow_pickle=False,
        )
        if (
            retained_positions.shape != (75, 3)
            or retained_positions.dtype != np.dtype("<f8")
            or not np.all(np.isfinite(retained_positions))
            or not np.array_equal(retained_positions, positions)
        ):
            errors.append("listener position array differs from the rig sidecar")
        if (
            retained_orientations.shape != (75, 4)
            or retained_orientations.dtype != np.dtype("<f8")
            or not np.all(np.isfinite(retained_orientations))
            or not np.array_equal(retained_orientations, orientations)
        ):
            errors.append("listener orientation array differs from the rig sidecar")

        frame_records = load_json(root / "visual" / "frame_records.json").get(
            "frames"
        )
        if not isinstance(frame_records, list) or len(frame_records) != len(
            trajectory["frames"]
        ):
            errors.append("visual frame-record count differs from sensor rig")
        else:
            for frame, record in zip(
                trajectory["frames"], frame_records, strict=True
            ):
                if (
                    record.get("frame_index") != frame["frame_index"]
                    or record.get("pts_ticks") != frame["pts_ticks"]
                    or record.get("world_from_rig") != frame["world_from_rig"]
                    or record.get("view_pose_hash") != frame["pose_hash"]
                ):
                    errors.append(
                        "visual frame record differs from sensor rig at "
                        f"frame {frame['frame_index']}"
                    )
                    break

        acoustic = load_json(root / "trajectory" / "emitter_path.json")
        keyframes = acoustic.get("keyframes")
        if not isinstance(keyframes, list) or len(keyframes) != len(
            trajectory["frames"]
        ):
            errors.append("acoustic trajectory frame count differs from sensor rig")
        else:
            for frame, keyframe, position, orientation in zip(
                trajectory["frames"],
                keyframes,
                positions,
                orientations,
                strict=True,
            ):
                if (
                    keyframe.get("tick") != frame["pts_ticks"]
                    or keyframe.get("listener_position_m") != position.tolist()
                    or keyframe.get("listener_orientation_wxyz")
                    != orientation.tolist()
                ):
                    errors.append(
                        "acoustic listener pose differs from sensor rig at "
                        f"frame {frame['frame_index']}"
                    )
                    break

        expected_hashes = [
            frame["pose_hash"] for frame in trajectory["frames"]
        ]
        for variant in ("A", "B"):
            timeline = load_json(root / "episodes" / variant / "timeline.json")
            frames = timeline.get("frames")
            observed_hashes = (
                [
                    frame.get("view_pose_hashes", {}).get("view0")
                    for frame in frames
                ]
                if isinstance(frames, list)
                else None
            )
            if observed_hashes != expected_hashes:
                errors.append(
                    f"episode {variant} Timeline view poses differ from sensor rig"
                )
    except Exception as exc:
        errors.append(f"sensor-rig evidence readback failed: {exc}")
    return errors


def _prepare_m5_installed_runtime(
    *,
    runtime_prefix: str | Path | None,
    runtime_root: str | Path | None,
    mp3d_root: str | Path | None,
    magnum_python_site: str | Path | None,
    rlr_sdk_root: str | Path | None,
) -> InstalledHabitatRuntime:
    """Select the one explicit current runtime used by visual and acoustics."""

    if runtime_prefix is not None and runtime_root is not None:
        raise M5CanaryError(
            "specify only --runtime-prefix or --runtime-root; --runtime-root is "
            "an installed-prefix compatibility alias"
        )
    missing = [
        option
        for option, value in (
            ("--runtime-prefix/--runtime-root", runtime_prefix or runtime_root),
            ("--magnum-python-site", magnum_python_site),
            ("--rlr-sdk-root", rlr_sdk_root),
        )
        if value is None or not str(value).strip()
    ]
    if missing:
        raise M5CanaryError(
            "m5 run-canary requires explicit current runtime inputs: "
            + ", ".join(missing)
        )
    try:
        return prepare_installed_habitat_runtime(
            runtime_prefix=runtime_prefix,
            runtime_root=runtime_root,
            mp3d_root=mp3d_root,
            magnum_python_site=magnum_python_site,
            rlr_sdk_root=rlr_sdk_root,
            allow_mp3d_environment=False,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeUnavailableError(
            f"current installed Habitat/RLR runtime is unavailable: {error}"
        ) from error


def _render_current_dynamic_rir_pair(
    *,
    scene: Any,
    simulation: M4SimulationConfig,
    keyframes: Sequence[AcousticKeyframe],
    hrtf_path: str | Path,
    installed_runtime: InstalledHabitatRuntime,
    rlr_sdk_root: str | Path,
) -> tuple[DynamicRIRSequence, DynamicRIRSequence]:
    """Render FOA and binaural with one identical explicit runtime selection."""

    runtime_loader = {
        "runtime_mode": RUNTIME_MODE_CURRENT_INSTALLED,
        "runtime_prefix": installed_runtime.prefix,
        "rlr_sdk_root": Path(rlr_sdk_root).resolve(),
        "magnum_python_site": installed_runtime.magnum_python_site,
    }
    foa = render_dynamic_rir_sequence(
        scene,
        simulation,
        keyframes=keyframes,
        layout_type="ambisonics",
        channel_count=4,
        **runtime_loader,
    )
    binaural = render_dynamic_rir_sequence(
        scene,
        simulation,
        keyframes=keyframes,
        layout_type="binaural",
        channel_count=2,
        hrtf_file_path=str(Path(hrtf_path).resolve()),
        **runtime_loader,
    )
    return foa, binaural


def run_m5_canary(
    *,
    request_path: str | Path,
    animal_manifest_path: str | Path,
    m2_request_path: str | Path,
    room_manifest_path: str | Path,
    m1_request_path: str | Path,
    acoustic_package_manifest_path: str | Path,
    m4_request_path: str | Path,
    output_directory: str | Path,
    runtime_root: str | Path | None = None,
    runtime_prefix: str | Path | None = None,
    mp3d_root: str | Path | None = None,
    magnum_python_site: str | Path | None = None,
    rlr_sdk_root: str | Path | None = None,
    hrtf_path: str | Path,
    hrtf_license_path: str | Path,
    beagle_dry_path: str | Path,
    golden_dry_path: str | Path,
    sensor_rig_trajectory_path: str | Path | None = None,
) -> Path:
    """Run and atomically publish the complete M5 two-dog counterfactual."""

    request_file = Path(request_path).resolve()
    request = load_json(request_file)
    sensor_rig_trajectory_file = (
        None
        if sensor_rig_trajectory_path is None
        else Path(sensor_rig_trajectory_path).resolve()
    )
    sensor_rig_trajectory = (
        None
        if sensor_rig_trajectory_file is None
        else load_json(sensor_rig_trajectory_file)
    )
    request_errors = validate_episode_request(request)
    if request_errors:
        raise M5CanaryError("; ".join(request_errors))
    (
        dry_clip_start,
        dry_clip_end,
        dry_fade_samples,
        dry_linear_gain,
        active_windows,
    ) = _audio_program_execution(request)
    event_starts = tuple(start for start, _end in active_windows)
    event_duration = dry_clip_end - dry_clip_start
    if any(end - start != event_duration for start, end in active_windows):
        raise M5CanaryError("declared event windows differ from the dry clip length")
    actors = request["actors"]
    sources = request["sources"]
    code_provenance = _code_provenance()
    if code_provenance["worktree_clean"] is not True:
        raise M5CanaryError("formal M5 evidence requires a clean Git worktree")
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise M5CanaryError(f"refusing to replace existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    ).resolve()
    try:
        installed_runtime = _prepare_m5_installed_runtime(
            runtime_prefix=runtime_prefix,
            runtime_root=runtime_root,
            mp3d_root=mp3d_root,
            magnum_python_site=magnum_python_site,
            rlr_sdk_root=rlr_sdk_root,
        )
        visual = capture_two_actor_fixed_states(
            animal_manifest_path=animal_manifest_path,
            m2_request_path=m2_request_path,
            room_manifest_path=room_manifest_path,
            m1_request_path=m1_request_path,
            installed_runtime=installed_runtime,
            actor_offsets_m=tuple(
                tuple(float(value) for value in actor["instance_offset_m"])
                for actor in actors
            ),
            actor_ids=tuple(actor["actor_id"] for actor in actors),
            source_ids=tuple(source["source_id"] for source in sources),
            semantic_ids=tuple(int(actor["semantic_id"]) for actor in actors),
            emitter_link_names=tuple(source["emitter_link"] for source in sources),
            sensor_rig_trajectory=sensor_rig_trajectory,
        )
        declared_paths = {
            item["source_id"]: item["emitter_path_sha256"]
            for item in request["sources"]
        }
        observed_paths = {
            source_id: _named_emitter_path_sha256(
                source_id, visual.source_positions_m[:, source_index]
            )
            for source_index, source_id in enumerate(visual.source_ids)
        }
        if declared_paths != observed_paths:
            raise M5CanaryError("captured muzzle path hash differs from the M5 request")

        arrays_dir = staging / "visual" / "arrays"
        listener_poses = [
            _listener_pose_at_frame(visual, frame_index) for frame_index in range(75)
        ]
        visual_arrays = {
            "rgb": _save_npy(arrays_dir / "rgb.npy", visual.rgb),
            "depth": _save_npy(arrays_dir / "depth.npy", visual.depth),
            "semantic": _save_npy(arrays_dir / "semantic.npy", visual.semantic),
            "actor_world_matrices": _save_npy(
                arrays_dir / "actor_world_matrices.npy", visual.actor_world_matrices
            ),
            "source_positions_m": _save_npy(
                arrays_dir / "source_positions_m.npy", visual.source_positions_m
            ),
            "listener_positions_m": _save_npy(
                arrays_dir / "listener_positions_m.npy",
                np.asarray([pose[0] for pose in listener_poses], dtype=np.float64),
            ),
            "listener_orientations_wxyz": _save_npy(
                arrays_dir / "listener_orientations_wxyz.npy",
                np.asarray([pose[1] for pose in listener_poses], dtype=np.float64),
            ),
            "topdown_rgb": _save_npy(
                arrays_dir / "topdown_rgb.npy", visual.topdown_rgb
            ),
        }
        write_json(staging / "visual" / "capture_metadata.json", dict(visual.metadata))
        write_json(
            staging / "visual" / "frame_records.json", {"frames": list(visual.records)}
        )

        m2_request = load_json(m2_request_path)
        m1_request = load_json(m1_request_path)
        visual_frames = _visual_timeline_frames(visual, m2_request, m1_request)
        pair = build_counterfactual_pair(request, visual_frames)
        if pair["comparison"]["status"] != "pass":
            raise M5CanaryError("counterfactual timeline comparison failed")
        episode_root = staging / "episodes"
        write_json(episode_root / "counterfactual_pair.json", pair)
        for variant in ("A", "B"):
            write_json(
                episode_root / variant / "request.json",
                pair["episodes"][variant]["request"],
            )
            write_json(
                episode_root / variant / "timeline.json",
                pair["episodes"][variant]["timeline"],
            )
            write_json(
                episode_root / variant / "dynamic_audio_render_manifest.json",
                pair["episodes"][variant]["dynamic_audio_render_manifest"],
            )

        keyframes = _acoustic_keyframes(visual)
        trajectory = trajectory_record(keyframes, visual.source_ids)
        trajectory["trajectory_content_sha256"] = canonical_json_sha256(trajectory)
        write_json(staging / "trajectory" / "emitter_path.json", trajectory)
        if visual.sensor_rig_trajectory is not None:
            write_json(
                staging / "trajectory" / "sensor_rig_trajectory.json",
                visual.sensor_rig_trajectory,
            )
        scene = load_compiled_acoustic_scene(acoustic_package_manifest_path)
        m4_request = load_json(m4_request_path)
        simulation = M4SimulationConfig.from_mapping(m4_request["simulation"])
        if rlr_sdk_root is None:
            raise M5CanaryError("current M5 acoustics require --rlr-sdk-root")
        foa, binaural = _render_current_dynamic_rir_pair(
            scene=scene,
            simulation=simulation,
            keyframes=keyframes,
            hrtf_path=hrtf_path,
            installed_runtime=installed_runtime,
            rlr_sdk_root=rlr_sdk_root,
        )
        if foa.trajectory_sha256 != binaural.trajectory_sha256:
            raise M5CanaryError("FOA and binaural RIRs used different trajectories")
        rir_records = {
            "foa": _write_rir_sequence(staging, "foa", foa),
            "binaural": _write_rir_sequence(staging, "binaural", binaural),
        }

        inputs_dir = staging / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        input_sources = {
            "request": request_file,
            "animal_manifest": Path(animal_manifest_path).resolve(),
            "m2_request": Path(m2_request_path).resolve(),
            "room_manifest": Path(room_manifest_path).resolve(),
            "m1_request": Path(m1_request_path).resolve(),
            "acoustic_package_manifest": Path(acoustic_package_manifest_path).resolve(),
            "m4_request": Path(m4_request_path).resolve(),
            "hrtf": Path(hrtf_path).resolve(),
            "hrtf_license": Path(hrtf_license_path).resolve(),
            "beagle_dry": Path(beagle_dry_path).resolve(),
            "golden_dry": Path(golden_dry_path).resolve(),
        }
        if sensor_rig_trajectory_file is not None:
            input_sources["sensor_rig_trajectory"] = sensor_rig_trajectory_file
        copied_inputs: dict[str, Any] = {}
        for role, source in input_sources.items():
            if not source.is_file():
                raise M5CanaryError(f"M5 input is unavailable: {source}")
            suffix = source.suffix or ".bin"
            target = inputs_dir / f"{role}{suffix}"
            shutil.copy2(source, target)
            copied_inputs[role] = file_record(target, relative_to=staging)

        raw_assets: dict[str, tuple[str, np.ndarray]] = {}
        for asset_id, role in (
            ("beagle_call", "beagle_dry"),
            ("golden_call", "golden_dry"),
        ):
            path = input_sources[role]
            samples, rate = read_pcm16_mono_wav(path)
            if rate != M5_AUDIO_SAMPLE_RATE_HZ:
                raise M5CanaryError("dry input rate changed")
            clip = extract_faded_clip(
                samples,
                start_sample=dry_clip_start,
                end_sample=dry_clip_end,
                fade_samples=dry_fade_samples,
            )
            raw_assets[sha256_file(path)] = (asset_id, clip)
        declared_dry_hashes = {
            event["dry_audio_asset_sha256"] for event in request["events"]
        }
        if declared_dry_hashes != set(raw_assets):
            raise M5CanaryError("request dry-asset SHA set differs from copied inputs")

        episode_audio: dict[str, Any] = {}
        episode_metrics: dict[str, Any] = {}
        for variant in ("A", "B"):
            variant_request = pair["episodes"][variant]["request"]
            route = {
                event["source_id"]: raw_assets[event["dry_audio_asset_sha256"]][0]
                for event in variant_request["events"]
            }
            clips = {asset_id: clip for asset_id, clip in raw_assets.values()}
            dry_buses, scheduled_events = place_simultaneous_events(
                clips,
                route,
                start_samples=event_starts,
                output_sample_count=M5_AUDIO_SAMPLE_COUNT,
                linear_gain=dry_linear_gain,
            )
            observed_windows = {
                (int(event["start_sample"]), int(event["end_sample"]))
                for event in scheduled_events
            }
            if observed_windows != set(active_windows):
                raise M5CanaryError(
                    "rendered schedule differs from request audio_program"
                )
            variant_dir = episode_root / variant
            audio_records: dict[str, Any] = {"dry_buses": {}, "foa": {}, "binaural": {}}
            for source_id in visual.source_ids:
                audio_records["dry_buses"][source_id] = _write_audio(
                    variant_dir / "audio" / "dry" / f"{source_id}.wav",
                    dry_buses[source_id][None, :],
                    metadata={
                        "role": "scheduled_dry_source_bus",
                        "variant": variant,
                        "source_id": source_id,
                        "schedule": scheduled_events,
                        "no_resample_normalize_limiter": True,
                    },
                )
            rendered_by_layout: dict[str, tuple[Mapping[str, Any], np.ndarray]] = {}
            for layout_name, sequence in (("foa", foa), ("binaural", binaural)):
                stems, mixture = render_dynamic_stems_and_mix(
                    dry_buses,
                    sequence.samples,
                    sequence.lengths,
                    source_ids=sequence.source_ids,
                    keyframe_samples=sequence.keyframe_samples,
                )
                rendered_by_layout[layout_name] = (stems, mixture)
                for source_id in sequence.source_ids:
                    audio_records[layout_name][source_id] = _write_audio(
                        variant_dir / "audio" / layout_name / f"{source_id}_stem.wav",
                        stems[source_id].episode,
                        metadata={
                            "role": "dynamic_wet_stem",
                            "variant": variant,
                            "source_id": source_id,
                            "layout_id": sequence.layout_id,
                            "trajectory_sha256": sequence.trajectory_sha256,
                            "partition_algorithm": stems[source_id].algorithm,
                            "full_tail_sample_count": int(
                                stems[source_id].full_tail.shape[1]
                            ),
                            "episode_crop": [0, M5_AUDIO_SAMPLE_COUNT],
                        },
                    )
                audio_records[layout_name]["mixture"] = _write_audio(
                    variant_dir / "audio" / layout_name / "mixture.wav",
                    mixture,
                    metadata={
                        "role": "dynamic_mixture",
                        "variant": variant,
                        "layout_id": sequence.layout_id,
                        "canonical_source_order": list(sequence.source_ids),
                        "summation": "float64_kahan_then_float32_wav",
                        "episode_crop": [0, M5_AUDIO_SAMPLE_COUNT],
                    },
                )
                if audio_records[layout_name]["mixture"]["peak_absolute"] >= 1.0:
                    raise M5CanaryError(
                        "authoritative M5 mixture clips float32 full scale"
                    )

            binaural_stems, binaural_mix = rendered_by_layout["binaural"]
            metrics = _spatial_metrics(
                visual,
                binaural,
                binaural_stems,
                binaural_mix,
                active_windows=active_windows,
            )
            if metrics["status"] != "pass":
                raise M5CanaryError(f"M5 spatial metrics failed for episode {variant}")
            write_json(variant_dir / "spatial_metrics.json", metrics)
            episode_metrics[variant] = metrics
            episode_audio[variant] = audio_records

        videos_dir = staging / "videos"
        formal_base = videos_dir / "view0_base_video_only.mp4"
        formal_base_report = encode_h264_base_video(visual.rgb, formal_base)
        # RIR metrics do not depend on the dry-asset swap, so one QA visual
        # stream is correctly shared by both counterfactual variants.
        qa_frames = compose_main_topdown_frames(
            visual.rgb,
            visual.topdown_rgb,
            text_by_frame=_qa_text_by_frame(
                episode_metrics["A"], active_windows=active_windows
            ),
        )
        qa_base = videos_dir / "view0_topdown_base_video_only.mp4"
        qa_base_report = encode_h264_qa_base_video(qa_frames, qa_base)
        video_reports: dict[str, Any] = {
            "formal_base": formal_base_report,
            "qa_base": qa_base_report,
            "episodes": {},
        }
        for variant in ("A", "B"):
            binaural_wav = Path(
                episode_audio[variant]["binaural"]["mixture"]["audio_path"]
            )
            formal_video = videos_dir / f"episode_{variant}_view0_binaural.mp4"
            qa_video = (
                videos_dir / f"episode_{variant}_view0_topdown_binaural_review.mp4"
            )
            formal_report = mux_binaural_wav(formal_base, binaural_wav, formal_video)
            qa_report = mux_qa_binaural_wav(qa_base, binaural_wav, qa_video)
            aac_report = aac_decode_diagnostics(formal_video, binaural_wav)
            if (
                aac_report["minimum_correlation"] < 0.98
                or aac_report["minimum_snr_db"] < 18.0
                or aac_report["lr_swap_suspected"]
            ):
                raise M5CanaryError("AAC video derivative failed listening readback")
            video_reports["episodes"][variant] = {
                "formal": formal_report,
                "qa_topdown": qa_report,
                "aac_decode": aac_report,
            }
        hash_a = video_packet_sha256(videos_dir / "episode_A_view0_binaural.mp4")
        hash_b = video_packet_sha256(videos_dir / "episode_B_view0_binaural.mp4")
        video_invariant = (
            hash_a["payload_sha256"] == hash_b["payload_sha256"]
            and hash_a["timeline_sha256"] == hash_b["timeline_sha256"]
        )
        if not video_invariant:
            raise M5CanaryError("A/B encoded visual packet streams differ")

        checks = [
            {
                "check_id": "exact_timeline_counterfactual",
                "status": pair["comparison"]["status"],
                "measured": pair["comparison"],
            },
            {
                "check_id": "single_formal_view",
                "status": "pass",
                "measured": {"view_ids": ["view0"], "topdown_is_qa_only": True},
            },
            {
                "check_id": "two_actor_semantic_visibility",
                "status": "pass",
                "measured": {
                    str(semantic_id): int(
                        np.min(
                            np.count_nonzero(
                                visual.semantic == semantic_id, axis=(1, 2)
                            )
                        )
                    )
                    for semantic_id in visual.semantic_ids
                },
            },
            {
                "check_id": "same_trajectory_foa_binaural",
                "status": "pass"
                if foa.trajectory_sha256 == binaural.trajectory_sha256
                else "fail",
                "measured": {"trajectory_sha256": foa.trajectory_sha256},
            },
            {
                "check_id": "simultaneous_two_source_schedule",
                "status": "pass",
                "measured": {
                    "source_count": 2,
                    "audio_program_id": request["audio_program"]["program_id"],
                    "event_windows": [list(window) for window in active_windows],
                    "dry_clip_source_interval": [dry_clip_start, dry_clip_end],
                    "fade_samples": dry_fade_samples,
                    "linear_gain": dry_linear_gain,
                },
            },
            {
                "check_id": "per_source_ild_ipd_itd",
                "status": "pass"
                if all(
                    metrics["status"] == "pass" for metrics in episode_metrics.values()
                )
                else "fail",
                "measured": {
                    variant: metrics["summary"]
                    for variant, metrics in episode_metrics.items()
                },
            },
            {
                "check_id": "exact_audio_length",
                "status": "pass",
                "measured": {"sample_rate_hz": 16_000, "sample_count": 80_000},
            },
            {
                "check_id": "ab_visual_packet_invariant",
                "status": "pass" if video_invariant else "fail",
                "measured": {"A": hash_a, "B": hash_b},
            },
            {
                "check_id": "video_mux_readback",
                "status": "pass",
                "measured": video_reports,
            },
        ]
        if any(check["status"] != "pass" for check in checks):
            raise M5CanaryError("one or more declared M5 checks failed")

        portable_audio = _portable_output_paths(episode_audio, staging)
        portable_video = _portable_output_paths(video_reports, staging)
        portable_checks = _portable_output_paths(checks, staging)
        evidence: dict[str, Any] = {
            "schema": M5_EVIDENCE_SCHEMA,
            "overall_status": "pass",
            "qualification_claim": False,
            "code_provenance": code_provenance,
            "bundle_scope": {
                "kind": "retained_evidence_closure",
                "retained_artifact_bytes_complete": True,
                "complete_upstream_reexecution_package": False,
                "statement": (
                    "This bundle closes the bytes needed to audit the retained M5 result; "
                    "it is not a complete runnable package of all upstream Habitat runtime, "
                    "room, animal, or toolchain assets."
                ),
            },
            "claim_boundary": (
                "M5 deterministic research canary; local animal call assets are retained "
                "for research listening evidence and are not admitted release dataset assets"
            ),
            "request_id": request["request_id"],
            "counterfactual_pair_id": request["counterfactual_pair_id"],
            "inputs": copied_inputs,
            "visual": {
                "metadata": visual.metadata,
                "arrays": visual_arrays,
                "emitter_path_sha256_by_source": observed_paths,
            },
            "rir_sequences": rir_records,
            "audio": portable_audio,
            "spatial_metrics": {
                variant: {
                    "status": metrics["status"],
                    "summary": metrics["summary"],
                    "mixture_diagnostic_only": True,
                }
                for variant, metrics in episode_metrics.items()
            },
            "counterfactual_comparison": pair["comparison"],
            "video": {
                **portable_video,
                "ab_formal_video_packet_invariant": video_invariant,
            },
            "checks": portable_checks,
        }
        evidence["artifacts"] = _artifact_index(staging)
        evidence["evidence_content_sha256"] = canonical_json_sha256(evidence)
        write_json(staging / "evidence.json", evidence)
        os.rename(staging, destination)
        return destination / "evidence.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_m5_canary_evidence(
    evidence_path: str | Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Independently rehash and semantically read back an M5 evidence bundle."""

    path = Path(evidence_path).resolve()
    root = path.parent
    evidence = load_json(path)
    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, measured: Any) -> None:
        checks.append(
            {
                "check_id": check_id,
                "status": "pass" if passed else "fail",
                "measured": measured,
            }
        )

    declared_hash = evidence.get("evidence_content_sha256")
    core = dict(evidence)
    core.pop("evidence_content_sha256", None)
    add(
        "evidence_content_hash",
        declared_hash == canonical_json_sha256(core),
        {"declared": declared_hash, "recomputed": canonical_json_sha256(core)},
    )
    provenance = evidence.get("code_provenance")
    provenance_pass = (
        isinstance(provenance, Mapping)
        and provenance.get("repository_role") == "avengine_habitat_native"
        and isinstance(provenance.get("commit"), str)
        and _GIT_COMMIT_RE.fullmatch(provenance["commit"]) is not None
        and provenance.get("worktree_clean") is True
        and provenance.get("capture_phase") == "before_staging_directory_creation"
    )
    add("clean_code_provenance", provenance_pass, provenance)

    scope = evidence.get("bundle_scope")
    scope_pass = (
        isinstance(scope, Mapping)
        and scope.get("kind") == "retained_evidence_closure"
        and scope.get("retained_artifact_bytes_complete") is True
        and scope.get("complete_upstream_reexecution_package") is False
        and isinstance(scope.get("statement"), str)
        and "not a complete runnable package" in scope["statement"]
    )
    add("honest_bundle_scope", scope_pass, scope)

    artifacts = evidence.get("artifacts")
    artifact_errors: list[str] = []
    if isinstance(artifacts, Mapping):
        for role, record in artifacts.items():
            if not isinstance(role, str) or not isinstance(record, Mapping):
                artifact_errors.append(f"invalid artifact entry {role!r}")
                continue
            candidate = (root / role).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                artifact_errors.append(f"artifact escapes root: {role}")
                continue
            if (
                not candidate.is_file()
                or candidate.stat().st_size != record.get("byte_size")
                or sha256_file(candidate) != record.get("sha256")
            ):
                artifact_errors.append(f"artifact bytes differ: {role}")
        actual_roles = {
            item.relative_to(root).as_posix()
            for item in root.rglob("*")
            if item.is_file() and item.name != "evidence.json"
        }
        if actual_roles != set(artifacts):
            artifact_errors.append(
                "artifact index does not exactly enumerate retained bundle files"
            )
    else:
        artifact_errors.append("artifacts mapping is absent")
    add("artifact_closure", not artifact_errors, artifact_errors)

    pair_path = root / "episodes" / "counterfactual_pair.json"
    try:
        comparison = compare_counterfactual_pair(load_json(pair_path))
        add("counterfactual_semantics", comparison["status"] == "pass", comparison)
    except Exception as exc:
        add("counterfactual_semantics", False, str(exc))

    array_expectations = {
        "rgb.npy": (75, 240, 320, 3),
        "depth.npy": (75, 240, 320),
        "semantic.npy": (75, 240, 320),
        "source_positions_m.npy": (75, 2, 3),
        "topdown_rgb.npy": (75, 240, 240, 3),
    }
    visual = evidence.get("visual")
    visual_metadata = (
        visual.get("metadata") if isinstance(visual, Mapping) else None
    )
    if (
        isinstance(visual_metadata, Mapping)
        and visual_metadata.get("sensor_rig_trajectory") is not None
    ):
        array_expectations.update(
            {
                "listener_positions_m.npy": (75, 3),
                "listener_orientations_wxyz.npy": (75, 4),
            }
        )
    array_errors: list[str] = []
    for name, expected_shape in array_expectations.items():
        try:
            array = np.load(
                root / "visual" / "arrays" / name, mmap_mode="r", allow_pickle=False
            )
            if array.shape != expected_shape:
                array_errors.append(f"{name} shape {array.shape}")
        except (OSError, ValueError) as exc:
            array_errors.append(f"{name}: {exc}")
    add("visual_array_readback", not array_errors, array_errors)

    sensor_rig_errors = _sensor_rig_evidence_errors(root, evidence)
    add("sensor_rig_evidence_binding", not sensor_rig_errors, sensor_rig_errors)

    rir, trajectory, rir_errors = _rir_authority(root, evidence)
    add("dynamic_rir_authority", not rir_errors, rir_errors)

    audio_errors = _audio_reconstruction_errors(root, evidence, rir, trajectory)
    add("dry_rir_stem_mixture_reconstruction", not audio_errors, audio_errors)

    spatial_errors = _spatial_metric_errors(root, evidence)
    add("spatial_metric_boundaries", not spatial_errors, spatial_errors)

    media_path_errors: list[str] = []

    def inspect_media_paths(value: Any, owner: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{owner}.{key}"
                if key == "path" or key.endswith("_path"):
                    try:
                        _confined_bundle_path(root, item, owner=child)
                    except Exception as exc:
                        media_path_errors.append(str(exc))
                else:
                    inspect_media_paths(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                inspect_media_paths(item, f"{owner}[{index}]")

    inspect_media_paths(evidence.get("audio"), "audio")
    inspect_media_paths(evidence.get("video"), "video")
    add("reported_media_path_confinement", not media_path_errors, media_path_errors)

    video_errors: list[str] = []
    try:
        video = evidence.get("video")
        formal_a = _confined_bundle_path(
            root,
            video["episodes"]["A"]["formal"]["path"],
            owner="video.episodes.A.formal.path",
        )
        formal_b = _confined_bundle_path(
            root,
            video["episodes"]["B"]["formal"]["path"],
            owner="video.episodes.B.formal.path",
        )
        probe_episode_video(formal_a)
        probe_episode_video(formal_b)
        probe_qa_review_video(
            _confined_bundle_path(
                root,
                video["episodes"]["A"]["qa_topdown"]["path"],
                owner="video.episodes.A.qa_topdown.path",
            )
        )
        probe_qa_review_video(
            _confined_bundle_path(
                root,
                video["episodes"]["B"]["qa_topdown"]["path"],
                owner="video.episodes.B.qa_topdown.path",
            )
        )
        hash_a = video_packet_sha256(formal_a)
        hash_b = video_packet_sha256(formal_b)
        if (
            hash_a["payload_sha256"] != hash_b["payload_sha256"]
            or hash_a["timeline_sha256"] != hash_b["timeline_sha256"]
        ):
            video_errors.append("A/B formal video packets differ")
    except Exception as exc:
        video_errors.append(str(exc))
    add("video_readback", not video_errors, video_errors)

    declared_checks = evidence.get("checks")
    declared_pass = isinstance(declared_checks, list) and all(
        isinstance(item, Mapping) and item.get("status") == "pass"
        for item in declared_checks
    )
    add("declared_checks", declared_pass, declared_checks)
    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return status, checks


__all__ = [
    "M5CanaryError",
    "M5_EVIDENCE_SCHEMA",
    "run_m5_canary",
    "verify_m5_canary_evidence",
]

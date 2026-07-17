"""Executable two-actor M5 canary and self-contained evidence bundle."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np

from avengine.contracts.json_io import (
    canonical_json_sha256,
    file_record,
    load_json,
    sha256_file,
    write_json,
)
from avengine.m3.runtime import load_compiled_acoustic_scene
from avengine.m4.audio import read_float32_wav, write_float32_wav
from avengine.m4.runtime import M4SimulationConfig
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
    listener_local_azimuth_deg,
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
M5_DRY_CLIP_START = 3_200
M5_DRY_CLIP_END = 8_000
M5_EVENT_STARTS = (6_400, 19_200, 32_000, 44_800, 57_600, 70_400)
M5_DRY_LINEAR_GAIN = 0.18


class M5CanaryError(RuntimeError):
    """The M5 canary could not prove its declared audiovisual contract."""


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


def _named_emitter_path_sha256(
    source_id: str, positions_m: np.ndarray
) -> str:
    return canonical_json_sha256(
        {
            "schema": "avengine_m5_named_emitter_path_v1",
            "source_id": source_id,
            "positions_m": np.asarray(positions_m, dtype=np.float64).tolist(),
        }
    )


def _visual_timeline_frames(
    visual: TwoActorVisualResult,
    m2_request: Mapping[str, Any],
    m1_request: Mapping[str, Any],
) -> list[dict[str, Any]]:
    offsets = np.asarray(visual.metadata["actor_offsets_m"], dtype=np.float64)
    camera_hash = canonical_json_sha256(
        {
            "view_id": "view0",
            "primary_camera_rig": m1_request["primary_camera_rig"],
        }
    )
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
                "view_pose_hashes": {"view0": camera_hash},
            }
        )
    return result


def _acoustic_keyframes(visual: TwoActorVisualResult) -> tuple[AcousticKeyframe, ...]:
    return tuple(
        AcousticKeyframe(
            tick=3_200 * frame_index,
            sample_index=(3_200 * frame_index + 1) // 3,
            source_positions_m={
                source_id: tuple(
                    float(value)
                    for value in visual.source_positions_m[frame_index, source_index]
                )
                for source_index, source_id in enumerate(visual.source_ids)
            },
            listener_position_m=visual.listener_position_m,
            listener_orientation_wxyz=visual.listener_orientation_wxyz,
        )
        for frame_index in range(75)
    )


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
) -> dict[str, Any]:
    per_source: dict[str, Any] = {}
    median_ild: dict[str, float] = {}
    median_itd: dict[str, float] = {}
    lateral_frames: list[Mapping[str, Any]] = []
    lateral_azimuths: list[float] = []
    for source_index, source_id in enumerate(binaural.source_ids):
        rir_report = measure_binaural_rir_sequence_cues(
            binaural.samples[:, source_index], binaural.sample_rate_hz
        )
        azimuths = [
            listener_local_azimuth_deg(
                visual.source_positions_m[frame_index, source_index],
                visual.listener_position_m,
                visual.listener_orientation_wxyz,
            )
            for frame_index in range(75)
        ]
        for frame, azimuth in zip(rir_report["frames"], azimuths, strict=True):
            frame["listener_local_azimuth_deg"] = azimuth
        lateral_frames.extend(rir_report["frames"])
        lateral_azimuths.extend(azimuths)
        wet_report = measure_binaural_wet_stem_cues(
            stems[source_id].episode,
            binaural.sample_rate_hz,
            M5_EVENT_STARTS[0],
            M5_EVENT_STARTS[-1] + (M5_DRY_CLIP_END - M5_DRY_CLIP_START),
            source_id=source_id,
        )
        ilds = [float(frame["ild_db"]) for frame in rir_report["frames"]]
        itds = [float(frame["itd"]["itd_seconds"]) for frame in rir_report["frames"]]
        median_ild[source_id] = float(np.median(ilds))
        median_itd[source_id] = float(np.median(itds))
        per_source[source_id] = {
            "azimuth_range_deg": [float(min(azimuths)), float(max(azimuths))],
            "rir_direct_window": rir_report,
            "wet_stem_active_window": wet_report,
        }
    mixture_report = measure_binaural_mixture_diagnostic(
        mixture,
        binaural.sample_rate_hz,
        M5_EVENT_STARTS[0],
        M5_EVENT_STARTS[-1] + (M5_DRY_CLIP_END - M5_DRY_CLIP_START),
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
    checks = {
        "all_source_itd_within_1ms": maximum_itd <= 0.0010000001,
        "lateral_cue_consistency": lateral_consistency["status"] == "pass"
        and lateral_consistency["formal_acceptance_allowed"] is True,
        "right_source_has_more_negative_ild_than_left_source": (
            median_ild["source0"] < median_ild["source1"]
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
            "maximum_absolute_itd_seconds": maximum_itd,
            "gcc_boundary_rejected_frame_count": lateral_consistency["counts"][
                "gcc_boundary_rejected_frames"
            ],
        },
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def _qa_text_by_frame(metrics: Mapping[str, Any]) -> list[str]:
    source0 = metrics["per_source"]["source0"]["rir_direct_window"]["frames"]
    source1 = metrics["per_source"]["source1"]["rir_direct_window"]["frames"]
    result: list[str] = []
    for index, (left, right) in enumerate(zip(source0, source1, strict=True)):
        rows = []
        for source_id, frame in (("Dog0", left), ("Dog1", right)):
            ipd = frame["ipd_radians_by_frequency_hz"].get("500")
            rows.append(
                f"{source_id} az={frame['listener_local_azimuth_deg']:+.1f}deg "
                f"ILD={frame['ild_db']:+.1f}dB IPD500={float(ipd):+.2f}rad "
                f"ITD={frame['itd']['itd_seconds'] * 1e6:+.0f}us"
            )
        active = any(start <= (3_200 * index + 1) // 3 < start + 4_800 for start in M5_EVENT_STARTS)
        result.append(("SIMULTANEOUS BARKS\n" if active else "RIR POSITION\n") + "\n".join(rows))
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
        return {
            key: _portable_output_paths(item, root) for key, item in value.items()
        }
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


def _confined_bundle_path(root: Path, value: Any, *, owner: str) -> Path:
    """Resolve one POSIX bundle path while rejecting absolute/traversal paths."""

    if not isinstance(value, str) or not value or "\\" in value:
        raise M5CanaryError(f"{owner} is not a portable bundle-relative path")
    portable = PurePosixPath(value)
    if portable.is_absolute() or any(part in {"", ".", ".."} for part in portable.parts):
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
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, Mapping[str, Any]]], dict[str, Any], list[str]]:
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
        {key: value for key, value in trajectory.items() if key != "trajectory_content_sha256"}
    )
    input_records = evidence.get("inputs")
    hrtf_record = input_records.get("hrtf") if isinstance(input_records, Mapping) else None
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
                root, record.get("samples_path"), owner=f"rir_sequences.{name}.samples_path"
            )
            lengths_path = _confined_bundle_path(
                root, record.get("lengths_path"), owner=f"rir_sequences.{name}.lengths_path"
            )
            metadata_path = _confined_bundle_path(
                root, record.get("metadata_path"), owner=f"rir_sequences.{name}.metadata_path"
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
                        errors.append(f"{name} RIR hash differs at {frame_index}/{source_id}")
                    if np.any(samples[frame_index, source_index, :, length:] != 0.0):
                        errors.append(f"{name} RIR padding is nonzero at {frame_index}/{source_id}")
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
                if not isinstance(retained_hrtf, Mapping) or retained_hrtf.get(
                    "input_role"
                ) != "hrtf" or retained_hrtf.get("sha256") != hrtf_hash:
                    errors.append("binaural RIR HRTF role/hash differs from retained input")
                for receipt in metadata.get("endpoint_receipts", []):
                    listener = receipt.get("listener") if isinstance(receipt, Mapping) else None
                    if not isinstance(listener, Mapping) or listener.get(
                        "hrtf_file_path"
                    ) != "input-role:hrtf":
                        errors.append("binaural RIR receipt retains an invalid HRTF path")
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
        raw_assets: dict[str, tuple[str, np.ndarray]] = {}
        for asset_id, role in (("beagle_call", "beagle_dry"), ("golden_call", "golden_dry")):
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
                    start_sample=M5_DRY_CLIP_START,
                    end_sample=M5_DRY_CLIP_END,
                    fade_samples=80,
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
                start_samples=M5_EVENT_STARTS,
                output_sample_count=M5_AUDIO_SAMPLE_COUNT,
                linear_gain=M5_DRY_LINEAR_GAIN,
            )
            variant_records = audio.get(variant)
            if not isinstance(variant_records, Mapping):
                errors.append(f"audio record for episode {variant} is absent")
                continue
            for source_id in ("source0", "source1"):
                record = variant_records.get("dry_buses", {}).get(source_id)
                path = _confined_bundle_path(
                    root, record.get("audio_path"), owner=f"audio.{variant}.dry.{source_id}"
                )
                decoded = read_float32_wav(path, verify_sidecar=True)
                if not np.array_equal(decoded.samples[0], buses[source_id].astype("<f4")):
                    errors.append(f"{variant}/{source_id} dry bus cannot be rebuilt")
            for layout, channels in (("foa", 4), ("binaural", 2)):
                rir_samples, rir_lengths, _ = rir[layout]
                stems, mixture = render_dynamic_stems_and_mix(
                    buses,
                    rir_samples,
                    rir_lengths,
                    source_ids=("source0", "source1"),
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
                    if decoded.samples.shape != (channels, M5_AUDIO_SAMPLE_COUNT) or not np.array_equal(
                        decoded.samples, expected
                    ):
                        errors.append(f"{variant}/{layout}/{source_id} stem cannot be rebuilt")
                record = layout_records.get("mixture")
                path = _confined_bundle_path(
                    root,
                    record.get("audio_path"),
                    owner=f"audio.{variant}.{layout}.mixture",
                )
                decoded = read_float32_wav(path, verify_sidecar=True)
                if decoded.samples.shape != (channels, M5_AUDIO_SAMPLE_COUNT) or not np.array_equal(
                    decoded.samples, mixture.astype("<f4")
                ):
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
            retained = summaries.get(variant) if isinstance(summaries, Mapping) else None
            if metrics.get("status") != "pass" or not isinstance(retained, Mapping):
                errors.append(f"{variant} spatial metrics are not retained as pass")
                continue
            if retained.get("status") != "pass" or retained.get("summary") != metrics.get(
                "summary"
            ):
                errors.append(f"{variant} spatial summary/status differs from retained report")
            mixture = metrics.get("mixture_diagnostic")
            if (
                retained.get("mixture_diagnostic_only") is not True
                or not isinstance(mixture, Mapping)
                or mixture.get("diagnostic_only") is not True
                or mixture.get("source_specific_acceptance_allowed") is not False
            ):
                errors.append(f"{variant} mixture was promoted beyond diagnostic-only scope")
            checks = metrics.get("checks")
            if not isinstance(checks, Mapping) or not checks or not all(
                value is True for value in checks.values()
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
                errors.append(f"{variant} lateral cue summary cannot be independently rebuilt")
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
    runtime_root: str | Path | None,
    hrtf_path: str | Path,
    hrtf_license_path: str | Path,
    beagle_dry_path: str | Path,
    golden_dry_path: str | Path,
) -> Path:
    """Run and atomically publish the complete M5 two-dog counterfactual."""

    request_file = Path(request_path).resolve()
    request = load_json(request_file)
    request_errors = validate_episode_request(request)
    if request_errors:
        raise M5CanaryError("; ".join(request_errors))
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise M5CanaryError(f"refusing to replace existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    ).resolve()
    try:
        visual = capture_two_actor_fixed_states(
            animal_manifest_path=animal_manifest_path,
            m2_request_path=m2_request_path,
            room_manifest_path=room_manifest_path,
            m1_request_path=m1_request_path,
            runtime_root=runtime_root,
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
            raise M5CanaryError(
                "captured muzzle path hash differs from the M5 request"
            )

        arrays_dir = staging / "visual" / "arrays"
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
            "topdown_rgb": _save_npy(
                arrays_dir / "topdown_rgb.npy", visual.topdown_rgb
            ),
        }
        write_json(staging / "visual" / "capture_metadata.json", dict(visual.metadata))
        write_json(staging / "visual" / "frame_records.json", {"frames": list(visual.records)})

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
        scene = load_compiled_acoustic_scene(acoustic_package_manifest_path)
        m4_request = load_json(m4_request_path)
        simulation = M4SimulationConfig.from_mapping(m4_request["simulation"])
        foa = render_dynamic_rir_sequence(
            scene,
            simulation,
            keyframes=keyframes,
            layout_type="ambisonics",
            channel_count=4,
        )
        binaural = render_dynamic_rir_sequence(
            scene,
            simulation,
            keyframes=keyframes,
            layout_type="binaural",
            channel_count=2,
            hrtf_file_path=str(Path(hrtf_path).resolve()),
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
        copied_inputs: dict[str, Any] = {}
        for role, source in input_sources.items():
            if not source.is_file():
                raise M5CanaryError(f"M5 input is unavailable: {source}")
            suffix = source.suffix or ".bin"
            target = inputs_dir / f"{role}{suffix}"
            shutil.copy2(source, target)
            copied_inputs[role] = file_record(target, relative_to=staging)

        raw_assets: dict[str, tuple[str, np.ndarray]] = {}
        for asset_id, role in (("beagle_call", "beagle_dry"), ("golden_call", "golden_dry")):
            path = input_sources[role]
            samples, rate = read_pcm16_mono_wav(path)
            if rate != M5_AUDIO_SAMPLE_RATE_HZ:
                raise M5CanaryError("dry input rate changed")
            clip = extract_faded_clip(
                samples,
                start_sample=M5_DRY_CLIP_START,
                end_sample=M5_DRY_CLIP_END,
                fade_samples=80,
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
                start_samples=M5_EVENT_STARTS,
                output_sample_count=M5_AUDIO_SAMPLE_COUNT,
                linear_gain=M5_DRY_LINEAR_GAIN,
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
                            "full_tail_sample_count": int(stems[source_id].full_tail.shape[1]),
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
                    raise M5CanaryError("authoritative M5 mixture clips float32 full scale")

            binaural_stems, binaural_mix = rendered_by_layout["binaural"]
            metrics = _spatial_metrics(visual, binaural, binaural_stems, binaural_mix)
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
            text_by_frame=_qa_text_by_frame(episode_metrics["A"]),
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
            qa_video = videos_dir / f"episode_{variant}_view0_topdown_binaural_review.mp4"
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
        hash_a = video_packet_sha256(
            videos_dir / "episode_A_view0_binaural.mp4"
        )
        hash_b = video_packet_sha256(
            videos_dir / "episode_B_view0_binaural.mp4"
        )
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
                        np.min(np.count_nonzero(visual.semantic == semantic_id, axis=(1, 2)))
                    )
                    for semantic_id in visual.semantic_ids
                },
            },
            {
                "check_id": "same_trajectory_foa_binaural",
                "status": "pass" if foa.trajectory_sha256 == binaural.trajectory_sha256 else "fail",
                "measured": {"trajectory_sha256": foa.trajectory_sha256},
            },
            {
                "check_id": "simultaneous_two_source_schedule",
                "status": "pass",
                "measured": {
                    "source_count": 2,
                    "event_start_samples": list(M5_EVENT_STARTS),
                    "event_duration_samples": M5_DRY_CLIP_END - M5_DRY_CLIP_START,
                },
            },
            {
                "check_id": "per_source_ild_ipd_itd",
                "status": "pass" if all(
                    metrics["status"] == "pass" for metrics in episode_metrics.values()
                ) else "fail",
                "measured": {
                    variant: metrics["summary"] for variant, metrics in episode_metrics.items()
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
    array_errors: list[str] = []
    for name, expected_shape in array_expectations.items():
        try:
            array = np.load(root / "visual" / "arrays" / name, mmap_mode="r", allow_pickle=False)
            if array.shape != expected_shape:
                array_errors.append(f"{name} shape {array.shape}")
        except (OSError, ValueError) as exc:
            array_errors.append(f"{name}: {exc}")
    add("visual_array_readback", not array_errors, array_errors)

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

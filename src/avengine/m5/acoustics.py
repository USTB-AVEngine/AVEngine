"""Persistent named-source RLR rendering for M5 dynamic trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.m3.runtime import (
    CompiledAcousticScene,
    RUNTIME_MODE_CURRENT_INSTALLED,
    RUNTIME_MODE_HISTORICAL,
    RuntimeAnchor,
    RuntimeContractError,
    RuntimeExecutionError,
    RuntimeUnavailableError,
    _native_configuration,
    _upload_report,
    _verify_upload_report,
    load_habitat_runtime,
    require_runtime_mode,
)
from avengine.spatial_audio.runtime import (
    M4SimulationConfig,
    _layout_contract,
    _native_layout,
    _native_registration_receipts,
    canonical_source_order,
    simulation_with_layout,
)


@dataclass(frozen=True)
class AcousticKeyframe:
    """One exact endpoint state on the M5 audio sample grid."""

    tick: int
    sample_index: int
    source_positions_m: Mapping[str, tuple[float, float, float]]
    listener_position_m: tuple[float, float, float]
    listener_orientation_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class DynamicRIRSequence:
    """Padded, named RIRs with exact original lengths.

    ``samples`` is ``[keyframe, source, channel, padded_sample]`` and source
    order is canonical ASCII byte order.  Padding is evidence-only; callers
    must use ``lengths`` during convolution.
    """

    samples: np.ndarray
    lengths: np.ndarray
    source_ids: tuple[str, ...]
    keyframe_ticks: tuple[int, ...]
    keyframe_samples: tuple[int, ...]
    sample_rate_hz: int
    layout_type: str
    layout_id: str
    channel_labels: tuple[str, ...]
    trajectory_sha256: str
    metadata: Mapping[str, Any]


_BUNDLE_HRTF_REFERENCE = "input-role:hrtf"


def _portable_hrtf_references(value: Any, absolute_path: str) -> Any:
    """Remove the machine-local HRTF path after native readback succeeds."""

    if isinstance(value, Mapping):
        return {
            key: _portable_hrtf_references(item, absolute_path)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable_hrtf_references(item, absolute_path) for item in value]
    if isinstance(value, tuple):
        return [_portable_hrtf_references(item, absolute_path) for item in value]
    if isinstance(value, str) and value == absolute_path:
        return _BUNDLE_HRTF_REFERENCE
    return value


def _finite_vector(value: Sequence[float], size: int, *, owner: str) -> tuple[float, ...]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeContractError(f"{owner} must be a numeric vector") from exc
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise RuntimeContractError(f"{owner} must be a finite length-{size} vector")
    return tuple(float(item) for item in array)


def validate_acoustic_keyframes(
    keyframes: Sequence[AcousticKeyframe],
    *,
    expected_source_ids: Sequence[str],
    episode_tick_count: int = 240_000,
    episode_sample_count: int = 80_000,
) -> tuple[AcousticKeyframe, ...]:
    """Validate the exact M5 visual-frame RIR grid and endpoint identities."""

    values = tuple(keyframes)
    if len(values) != 75:
        raise RuntimeContractError("M5 formal dynamic RIR grid requires 75 keyframes")
    source_ids = canonical_source_order(
        [RuntimeAnchor(anchor_id=value, position_m=(0.0, 0.0, 0.0)) for value in expected_source_ids]
    )
    previous_tick = -1
    previous_sample = -1
    normalized: list[AcousticKeyframe] = []
    for index, frame in enumerate(values):
        if not isinstance(frame, AcousticKeyframe):
            raise RuntimeContractError("every dynamic RIR keyframe must be AcousticKeyframe")
        if frame.tick != 3200 * index:
            raise RuntimeContractError("M5 acoustic keyframe tick differs from 15 Hz frame tick")
        expected_sample = (3200 * index + 1) // 3
        if frame.sample_index != expected_sample:
            raise RuntimeContractError("M5 acoustic keyframe sample uses a non-rational boundary")
        if frame.tick <= previous_tick or frame.sample_index <= previous_sample:
            raise RuntimeContractError("M5 acoustic keyframes must be strictly increasing")
        if not 0 <= frame.tick < episode_tick_count:
            raise RuntimeContractError("M5 acoustic keyframe tick escapes the episode")
        if not 0 <= frame.sample_index < episode_sample_count:
            raise RuntimeContractError("M5 acoustic keyframe sample escapes the episode")
        if set(frame.source_positions_m) != set(source_ids):
            raise RuntimeContractError("M5 keyframe source identity set changed")
        positions = {
            source_id: _finite_vector(
                frame.source_positions_m[source_id], 3, owner=f"{source_id} position"
            )
            for source_id in source_ids
        }
        listener_position = _finite_vector(
            frame.listener_position_m, 3, owner="listener position"
        )
        orientation = _finite_vector(
            frame.listener_orientation_wxyz, 4, owner="listener orientation"
        )
        if not math.isclose(
            float(np.linalg.norm(np.asarray(orientation))), 1.0, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise RuntimeContractError("listener orientation must be unit length")
        normalized.append(
            AcousticKeyframe(
                tick=int(frame.tick),
                sample_index=int(frame.sample_index),
                source_positions_m=positions,
                listener_position_m=listener_position,
                listener_orientation_wxyz=orientation,
            )
        )
        previous_tick = frame.tick
        previous_sample = frame.sample_index
    return tuple(normalized)


def trajectory_record(
    keyframes: Sequence[AcousticKeyframe], source_ids: Sequence[str]
) -> dict[str, Any]:
    return {
        "schema": "avengine_m5_emitter_trajectory_v1",
        "cadence": "one_rir_per_15_hz_visual_fixed_state",
        "interpolation": "raised_cosine_source_time_partition_v1",
        "source_ids": list(source_ids),
        "keyframes": [
            {
                "tick": frame.tick,
                "sample_index": frame.sample_index,
                "source_positions_m": {
                    source_id: list(frame.source_positions_m[source_id])
                    for source_id in source_ids
                },
                "listener_position_m": list(frame.listener_position_m),
                "listener_orientation_wxyz": list(frame.listener_orientation_wxyz),
            }
            for frame in keyframes
        ],
    }


def _owned_ir(
    raw_ir: Any,
    *,
    expected_listener_id: str,
    expected_source_ids: Sequence[str],
    channel_count: int,
    sample_rate_hz: int,
) -> tuple[str, np.ndarray]:
    listener_id = str(getattr(raw_ir, "listener_id", ""))
    source_id = str(getattr(raw_ir, "source_id", ""))
    if listener_id != expected_listener_id or source_id not in expected_source_ids:
        raise RuntimeContractError("RLR returned an undeclared dynamic endpoint pair")
    try:
        observed_rate = float(raw_ir.sample_rate)
        observed_channels = int(raw_ir.channel_count)
        observed_count = int(raw_ir.sample_count)
        samples = np.array(raw_ir.samples, dtype="<f4", order="C", copy=True)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeContractError(f"RLR returned malformed dynamic IR: {exc}") from exc
    if not math.isclose(observed_rate, sample_rate_hz, rel_tol=1.0e-6, abs_tol=1.0e-3):
        raise RuntimeContractError("dynamic RIR sample rate differs from request")
    if observed_channels != channel_count or samples.shape != (channel_count, observed_count):
        raise RuntimeContractError("dynamic RIR shape/count metadata is inconsistent")
    if observed_count < 2 or not np.all(np.isfinite(samples)) or not np.any(samples != 0.0):
        raise RuntimeContractError("dynamic RIR must be finite, non-empty and non-silent")
    return source_id, samples


def _load_dynamic_habitat_runtime(
    *,
    runtime_mode: str,
    runtime_prefix: str | Path | None,
    rlr_sdk_root: str | Path | None,
    magnum_python_site: str | Path | None,
) -> tuple[Any, dict[str, Any]]:
    """Select the historical or explicit current loader without ambiguity."""

    selected_runtime_mode = require_runtime_mode(runtime_mode)
    if selected_runtime_mode == RUNTIME_MODE_CURRENT_INSTALLED:
        return load_habitat_runtime(
            runtime_mode=selected_runtime_mode,
            runtime_prefix=runtime_prefix,
            rlr_sdk_root=rlr_sdk_root,
            magnum_python_site=magnum_python_site,
        )
    if any(
        value is not None
        for value in (runtime_prefix, rlr_sdk_root, magnum_python_site)
    ):
        raise RuntimeContractError(
            "explicit installed-runtime paths require current-installed mode"
        )
    # Preserve the historical direct-call and monkeypatch boundary exactly.
    return load_habitat_runtime()


def render_dynamic_rir_sequence(
    scene: CompiledAcousticScene,
    simulation: M4SimulationConfig,
    *,
    keyframes: Sequence[AcousticKeyframe],
    source_radius_m: float = 0.0,
    listener_id: str = "listener0",
    listener_radius_m: float = 0.0,
    layout_type: str,
    channel_count: int,
    hrtf_file_path: str = "",
    runtime_mode: str = RUNTIME_MODE_HISTORICAL,
    runtime_prefix: str | Path | None = None,
    rlr_sdk_root: str | Path | None = None,
    magnum_python_site: str | Path | None = None,
) -> DynamicRIRSequence:
    """Render all keyframes through one persistent, named RLR context."""

    if not isinstance(scene, CompiledAcousticScene):
        raise RuntimeContractError("scene must be a validated CompiledAcousticScene")
    if not isinstance(simulation, M4SimulationConfig):
        raise RuntimeContractError("simulation must be an M4SimulationConfig")
    if not keyframes:
        raise RuntimeContractError("dynamic RIR rendering requires keyframes")
    first_ids = tuple(sorted(keyframes[0].source_positions_m, key=lambda x: x.encode("ascii")))
    frames = validate_acoustic_keyframes(keyframes, expected_source_ids=first_ids)
    source_ids = tuple(first_ids)
    selected = simulation_with_layout(
        simulation, layout_type=layout_type, channel_count=channel_count
    )
    if selected.temporal_coherence:
        raise RuntimeContractError(
            "M5 v1 pins RLR temporal_coherence=false; filter crossfade is explicit"
        )
    if hrtf_file_path:
        resolved_hrtf = Path(hrtf_file_path).resolve()
        if layout_type != "binaural" or not resolved_hrtf.is_file():
            raise RuntimeContractError("external HRTF requires a readable binaural file")
        hrtf_file_path = str(resolved_hrtf)
    elif layout_type == "binaural":
        raise RuntimeContractError("M5 formal binaural output requires an explicit HRTF")
    contract = _layout_contract(layout_type, channel_count)
    trajectory = trajectory_record(frames, source_ids)
    trajectory_sha256 = canonical_json_sha256(trajectory)

    habitat_module, runtime_report = _load_dynamic_habitat_runtime(
        runtime_mode=runtime_mode,
        runtime_prefix=runtime_prefix,
        rlr_sdk_root=rlr_sdk_root,
        magnum_python_site=magnum_python_site,
    )
    native_configuration, config_readback = _native_configuration(habitat_module, selected)
    runtime_report["configuration_readback"] = config_readback
    runtime_report["output_contract"] = {
        **contract,
        "layout_type": layout_type,
        "channel_count": channel_count,
    }
    raw_by_frame: list[list[np.ndarray]] = []
    receipts: list[dict[str, Any]] = []
    ir_hashes: list[dict[str, str]] = []
    efficiencies: list[float] = []
    try:
        context = habitat_module.RLRAcousticContext(native_configuration)
        with tempfile.TemporaryDirectory(prefix="avengine-m5-rlr-db-") as temp_dir:
            private_database = Path(temp_dir) / "material_database.json"
            private_database.write_bytes(scene.material_database_bytes)
            if sha256_file(private_database) != scene.material_database_sha256:
                raise RuntimeContractError("private material database hash changed")
            raw_upload = context.load_acoustic_scene(
                str(private_database), list(scene.material_categories), list(scene.objects)
            )
        upload = _upload_report(raw_upload)
        _verify_upload_report(scene, upload)
        first = frames[0]
        for source_id in source_ids:
            context.add_source(source_id, first.source_positions_m[source_id], source_radius_m)
        context.add_listener(
            listener_id,
            first.listener_position_m,
            first.listener_orientation_wxyz,
            _native_layout(habitat_module, layout_type),
            channel_count,
            listener_radius_m,
            hrtf_file_path,
        )
        for frame_index, frame in enumerate(frames):
            current_sources = tuple(
                RuntimeAnchor(
                    anchor_id=source_id,
                    position_m=frame.source_positions_m[source_id],
                    radius_m=source_radius_m,
                )
                for source_id in source_ids
            )
            listener = RuntimeAnchor(
                anchor_id=listener_id,
                position_m=frame.listener_position_m,
                radius_m=listener_radius_m,
                orientation_wxyz=frame.listener_orientation_wxyz,
            )
            for source in current_sources:
                context.set_source_position(source.anchor_id, source.position_m)
                context.set_source_radius(source.anchor_id, source.radius_m)
            context.set_listener_pose(
                listener.anchor_id, listener.position_m, listener.orientation_wxyz
            )
            context.set_listener_radius(listener.anchor_id, listener.radius_m)
            raw_irs = context.simulate_owned()
            receipt = _native_registration_receipts(
                context,
                habitat_module,
                current_sources,
                listener,
                canonical_order=source_ids,
                layout_type=layout_type,
                channel_count=channel_count,
                hrtf_file_path=hrtf_file_path,
            )
            by_source: dict[str, np.ndarray] = {}
            hashes: dict[str, str] = {}
            for raw_ir in raw_irs:
                source_id, samples = _owned_ir(
                    raw_ir,
                    expected_listener_id=listener_id,
                    expected_source_ids=source_ids,
                    channel_count=channel_count,
                    sample_rate_hz=int(round(selected.sample_rate_hz)),
                )
                if source_id in by_source:
                    raise RuntimeContractError("RLR returned duplicate dynamic pair")
                by_source[source_id] = samples
                hashes[source_id] = hashlib.sha256(samples.tobytes(order="C")).hexdigest()
            if set(by_source) != set(source_ids):
                raise RuntimeContractError("RLR omitted a named pair in dynamic sequence")
            raw_by_frame.append([by_source[source_id] for source_id in source_ids])
            retained_receipt = (
                _portable_hrtf_references(receipt, hrtf_file_path)
                if hrtf_file_path
                else receipt
            )
            receipts.append({"frame_index": frame_index, **retained_receipt})
            ir_hashes.append(hashes)
            efficiency = float(context.indirect_ray_efficiency())
            if not math.isfinite(efficiency) or not 0.0 <= efficiency <= 1.0:
                raise RuntimeContractError("dynamic indirect ray efficiency is invalid")
            efficiencies.append(efficiency)
    except (RuntimeContractError, RuntimeUnavailableError):
        raise
    except Exception as exc:
        raise RuntimeExecutionError(f"persistent dynamic RLR simulation failed: {exc}") from exc

    maximum_length = max(samples.shape[1] for frame in raw_by_frame for samples in frame)
    padded = np.zeros(
        (len(frames), len(source_ids), channel_count, maximum_length), dtype="<f4"
    )
    lengths = np.empty((len(frames), len(source_ids)), dtype="<u4")
    for frame_index, frame_values in enumerate(raw_by_frame):
        for source_index, samples in enumerate(frame_values):
            length = samples.shape[1]
            padded[frame_index, source_index, :, :length] = samples
            lengths[frame_index, source_index] = length
    metadata = {
        "schema": "avengine_m5_dynamic_rir_sequence_v1",
        "trajectory_sha256": trajectory_sha256,
        "source_ids": list(source_ids),
        "listener_id": listener_id,
        "layout_type": layout_type,
        "layout_id": contract["layout_id"],
        "channel_labels": list(contract["channel_labels"]),
        "normalization": contract["normalization"],
        "coordinate_frame": contract["coordinate_frame"],
        "sample_rate_hz": int(round(selected.sample_rate_hz)),
        "context_policy": {
            "lifetime": "one_persistent_context_per_layout",
            "endpoint_update_order": list(source_ids),
            "simulate_calls": len(frames),
            "temporal_coherence": False,
            "reset_between_keyframes": False,
        },
        "motion_acoustics": {
            "endpoint_pose_cadence_hz": 15,
            "source_position_updated_per_keyframe": True,
            "listener_position_updated_per_keyframe": True,
            "listener_orientation_updated_per_keyframe": True,
            "room_propagation_recomputed_per_keyframe": True,
            "continuous_doppler_modeled": False,
            "device_motion_noise_modeled": False,
        },
        "runtime": runtime_report,
        "upload_report": upload,
        "endpoint_receipts": receipts,
        "ir_sha256_by_frame_source": ir_hashes,
        "indirect_ray_efficiency": efficiencies,
        "hrtf": (
            {
                "input_role": "hrtf",
                "sha256": sha256_file(hrtf_file_path),
            }
            if hrtf_file_path
            else None
        ),
    }
    return DynamicRIRSequence(
        samples=np.ascontiguousarray(padded),
        lengths=np.ascontiguousarray(lengths),
        source_ids=source_ids,
        keyframe_ticks=tuple(frame.tick for frame in frames),
        keyframe_samples=tuple(frame.sample_index for frame in frames),
        sample_rate_hz=int(round(selected.sample_rate_hz)),
        layout_type=layout_type,
        layout_id=str(contract["layout_id"]),
        channel_labels=tuple(contract["channel_labels"]),
        trajectory_sha256=trajectory_sha256,
        metadata=metadata,
    )


__all__ = [
    "AcousticKeyframe",
    "DynamicRIRSequence",
    "render_dynamic_rir_sequence",
    "trajectory_record",
    "validate_acoustic_keyframes",
]

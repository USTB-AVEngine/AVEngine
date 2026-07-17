"""Variable-duration dynamic binaural acoustics for M5.1 research review.

This module is intentionally separate from :mod:`avengine.m5.acoustics`'s
formal 75-keyframe validator.  M5.1 review clips may have any positive visual
frame count and any explicit positive RIR stride.  They retain the exact
visual-frame sample indices used to derive the acoustic keyframes, while the
native render still uses one persistent named-source RLR context.

The renderer is research-review infrastructure, not a dataset qualification
claim.  It returns M5's existing :class:`~avengine.m5.acoustics.DynamicRIRSequence`
so the same deterministic raised-cosine convolution path can be reused without
loosening the M5 formal contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from numbers import Real
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import canonical_json_sha256, sha256_file
from avengine.m3.runtime import (
    CompiledAcousticScene,
    RuntimeAnchor,
    RuntimeContractError,
    RuntimeExecutionError,
    RuntimeUnavailableError,
    _native_configuration,
    _upload_report,
    _verify_upload_report,
    load_habitat_runtime,
)
from avengine.m4.runtime import (
    M4SimulationConfig,
    _layout_contract,
    _native_layout,
    _native_registration_receipts,
    canonical_source_order,
    simulation_with_layout,
)
from avengine.m5.acoustics import (
    AcousticKeyframe,
    DynamicRIRSequence,
    _owned_ir,
    _portable_hrtf_references,
)
from avengine.m5.audio import DynamicStemResult, render_dynamic_stems_and_mix


RESEARCH_REVIEW_PROFILE = "m5_1_dynamic_binaural_research_review_v1"
RESEARCH_REVIEW_TRAJECTORY_SCHEMA = "avengine_m5_1_emitter_trajectory_v1"
RESEARCH_REVIEW_RIR_SCHEMA = "avengine_m5_1_dynamic_rir_sequence_v1"


@dataclass(frozen=True)
class ResearchReviewKeyframeGrid:
    """One explicit visual-to-acoustic sampling grid.

    ``visual_frame_rate_numerator / visual_frame_rate_denominator`` is stored
    as an exact rational.  ``visual_frame_indices`` must be precisely
    ``range(0, visual_frame_count, rir_stride_frames)``; there is no hidden
    resampling or inferred final endpoint.
    """

    keyframes: tuple[AcousticKeyframe, ...]
    source_ids: tuple[str, ...]
    visual_frame_indices: tuple[int, ...]
    visual_frame_count: int
    visual_frame_rate_numerator: int
    visual_frame_rate_denominator: int
    rir_stride_frames: int
    timeline_tick_rate_hz: int
    sample_rate_hz: int
    episode_tick_count: int
    episode_sample_count: int

    @property
    def visual_frame_rate_hz(self) -> Fraction:
        return Fraction(
            self.visual_frame_rate_numerator,
            self.visual_frame_rate_denominator,
        )

    @property
    def rir_keyframe_rate_hz(self) -> Fraction:
        return self.visual_frame_rate_hz / self.rir_stride_frames


def _positive_integer(value: Any, *, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeContractError(f"{owner} must be a positive integer")
    result = int(value)
    if result < 1:
        raise RuntimeContractError(f"{owner} must be a positive integer")
    return result


def _positive_rate(value: Any, *, owner: str) -> Fraction:
    if isinstance(value, bool):
        raise RuntimeContractError(f"{owner} must be a positive finite rate")
    try:
        if isinstance(value, Fraction):
            rate = value
        elif isinstance(value, (int, np.integer)):
            rate = Fraction(int(value), 1)
        elif isinstance(value, Real):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("non-finite")
            rate = Fraction(str(numeric))
        else:
            rate = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise RuntimeContractError(
            f"{owner} must be a positive finite rate"
        ) from exc
    if rate <= 0:
        raise RuntimeContractError(f"{owner} must be a positive finite rate")
    return rate


def _round_nonnegative_fraction(value: Fraction) -> int:
    if value < 0:
        raise RuntimeContractError("timeline boundaries cannot be negative")
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(remainder * 2 >= value.denominator)


def _finite_vector(
    value: Sequence[float], size: int, *, owner: str
) -> tuple[float, ...]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeContractError(f"{owner} must be a numeric vector") from exc
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise RuntimeContractError(f"{owner} must be a finite length-{size} vector")
    return tuple(float(item) for item in array)


def _canonical_ids(source_ids: Sequence[str]) -> tuple[str, ...]:
    return canonical_source_order(
        tuple(
            RuntimeAnchor(anchor_id=source_id, position_m=(0.0, 0.0, 0.0))
            for source_id in source_ids
        )
    )


def _sample_for_tick(tick: int, *, sample_rate_hz: int, tick_rate_hz: int) -> int:
    return _round_nonnegative_fraction(Fraction(tick * sample_rate_hz, tick_rate_hz))


def build_strided_review_keyframes(
    source_anchor_trajectories_m: Mapping[str, Sequence[Sequence[float]]],
    *,
    visual_frame_rate_hz: Any,
    rir_stride_frames: int,
    listener_position_m: Sequence[float],
    listener_orientation_wxyz: Sequence[float],
    timeline_tick_rate_hz: int = 48_000,
    sample_rate_hz: int = 16_000,
) -> ResearchReviewKeyframeGrid:
    """Sample equal-length visual anchor trajectories at an explicit stride.

    For the legacy apartment comparison, 270 anchors at 15 Hz with stride 3
    produce exactly 90 keyframes at 5 Hz over an 18-second episode.  The last
    keyframe is visual frame 267 (17.8 s); the final 0.2 s is held by the last
    RIR through M5's raised-cosine partition policy.
    """

    if not isinstance(source_anchor_trajectories_m, Mapping) or not source_anchor_trajectories_m:
        raise RuntimeContractError(
            "source_anchor_trajectories_m must be a non-empty mapping"
        )
    stride = _positive_integer(rir_stride_frames, owner="rir_stride_frames")
    tick_rate = _positive_integer(
        timeline_tick_rate_hz, owner="timeline_tick_rate_hz"
    )
    sample_rate = _positive_integer(sample_rate_hz, owner="sample_rate_hz")
    visual_rate = _positive_rate(visual_frame_rate_hz, owner="visual_frame_rate_hz")
    listener_position = _finite_vector(
        listener_position_m, 3, owner="listener position"
    )
    listener_orientation = _finite_vector(
        listener_orientation_wxyz, 4, owner="listener orientation"
    )
    if not math.isclose(
        float(np.linalg.norm(np.asarray(listener_orientation))),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise RuntimeContractError("listener orientation must be unit length")

    declared_ids = tuple(source_anchor_trajectories_m)
    source_ids = _canonical_ids(declared_ids)
    if len(set(declared_ids)) != len(declared_ids):
        raise RuntimeContractError("source anchor IDs must be unique")
    trajectories: dict[str, np.ndarray] = {}
    frame_count: int | None = None
    for source_id in source_ids:
        try:
            points = np.asarray(
                source_anchor_trajectories_m[source_id], dtype=np.float64
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeContractError(
                f"source trajectory {source_id!r} must be a finite [frame,3] array"
            ) from exc
        if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] < 1:
            raise RuntimeContractError(
                f"source trajectory {source_id!r} must be a finite [frame,3] array"
            )
        if not np.all(np.isfinite(points)):
            raise RuntimeContractError(
                f"source trajectory {source_id!r} must be a finite [frame,3] array"
            )
        if frame_count is None:
            frame_count = int(points.shape[0])
        elif points.shape[0] != frame_count:
            raise RuntimeContractError("all source trajectories must have equal frame count")
        trajectories[source_id] = np.ascontiguousarray(points)
    assert frame_count is not None

    visual_indices = tuple(range(0, frame_count, stride))
    episode_ticks = _round_nonnegative_fraction(
        Fraction(frame_count * tick_rate, 1) / visual_rate
    )
    episode_samples = _sample_for_tick(
        episode_ticks, sample_rate_hz=sample_rate, tick_rate_hz=tick_rate
    )
    keyframes: list[AcousticKeyframe] = []
    previous_tick = -1
    previous_sample = -1
    for visual_index in visual_indices:
        tick = _round_nonnegative_fraction(
            Fraction(visual_index * tick_rate, 1) / visual_rate
        )
        sample_index = _sample_for_tick(
            tick, sample_rate_hz=sample_rate, tick_rate_hz=tick_rate
        )
        if tick <= previous_tick or sample_index <= previous_sample:
            raise RuntimeContractError(
                "visual/RIR cadence is too dense for strict tick and sample boundaries"
            )
        keyframes.append(
            AcousticKeyframe(
                tick=tick,
                sample_index=sample_index,
                source_positions_m={
                    source_id: tuple(
                        float(item) for item in trajectories[source_id][visual_index]
                    )
                    for source_id in source_ids
                },
                listener_position_m=listener_position,
                listener_orientation_wxyz=listener_orientation,
            )
        )
        previous_tick = tick
        previous_sample = sample_index
    grid = ResearchReviewKeyframeGrid(
        keyframes=tuple(keyframes),
        source_ids=source_ids,
        visual_frame_indices=visual_indices,
        visual_frame_count=frame_count,
        visual_frame_rate_numerator=visual_rate.numerator,
        visual_frame_rate_denominator=visual_rate.denominator,
        rir_stride_frames=stride,
        timeline_tick_rate_hz=tick_rate,
        sample_rate_hz=sample_rate,
        episode_tick_count=episode_ticks,
        episode_sample_count=episode_samples,
    )
    return validate_research_review_grid(grid)


def validate_research_review_grid(
    grid: ResearchReviewKeyframeGrid,
) -> ResearchReviewKeyframeGrid:
    """Fail closed on a variable-duration M5.1 grid and fixed listener."""

    if not isinstance(grid, ResearchReviewKeyframeGrid):
        raise RuntimeContractError("review grid must be ResearchReviewKeyframeGrid")
    frame_count = _positive_integer(
        grid.visual_frame_count, owner="visual_frame_count"
    )
    stride = _positive_integer(grid.rir_stride_frames, owner="rir_stride_frames")
    tick_rate = _positive_integer(
        grid.timeline_tick_rate_hz, owner="timeline_tick_rate_hz"
    )
    sample_rate = _positive_integer(grid.sample_rate_hz, owner="sample_rate_hz")
    rate = _positive_rate(
        Fraction(
            _positive_integer(
                grid.visual_frame_rate_numerator,
                owner="visual_frame_rate_numerator",
            ),
            _positive_integer(
                grid.visual_frame_rate_denominator,
                owner="visual_frame_rate_denominator",
            ),
        ),
        owner="visual_frame_rate_hz",
    )
    canonical = _canonical_ids(grid.source_ids)
    if not canonical or tuple(grid.source_ids) != canonical:
        raise RuntimeContractError("review source_ids must be non-empty and canonical")
    expected_indices = tuple(range(0, frame_count, stride))
    if tuple(grid.visual_frame_indices) != expected_indices:
        raise RuntimeContractError("review visual frame indices differ from explicit stride")
    if len(grid.keyframes) != len(expected_indices):
        raise RuntimeContractError("review grid requires one RIR per sampled visual frame")

    expected_episode_ticks = _round_nonnegative_fraction(
        Fraction(frame_count * tick_rate, 1) / rate
    )
    expected_episode_samples = _sample_for_tick(
        expected_episode_ticks,
        sample_rate_hz=sample_rate,
        tick_rate_hz=tick_rate,
    )
    if grid.episode_tick_count != expected_episode_ticks:
        raise RuntimeContractError("review episode tick count differs from visual duration")
    if grid.episode_sample_count != expected_episode_samples:
        raise RuntimeContractError("review episode sample count differs from tick duration")

    fixed_listener_position: tuple[float, ...] | None = None
    fixed_listener_orientation: tuple[float, ...] | None = None
    previous_tick = -1
    previous_sample = -1
    for keyframe_index, (visual_index, frame) in enumerate(
        zip(expected_indices, grid.keyframes, strict=True)
    ):
        if not isinstance(frame, AcousticKeyframe):
            raise RuntimeContractError("every review RIR keyframe must be AcousticKeyframe")
        expected_tick = _round_nonnegative_fraction(
            Fraction(visual_index * tick_rate, 1) / rate
        )
        expected_sample = _sample_for_tick(
            expected_tick,
            sample_rate_hz=sample_rate,
            tick_rate_hz=tick_rate,
        )
        if frame.tick != expected_tick or frame.sample_index != expected_sample:
            raise RuntimeContractError(
                f"review keyframe {keyframe_index} differs from its visual boundary"
            )
        if frame.tick <= previous_tick or frame.sample_index <= previous_sample:
            raise RuntimeContractError("review keyframes must be strictly increasing")
        if not 0 <= frame.tick < grid.episode_tick_count:
            raise RuntimeContractError("review keyframe tick escapes the episode")
        if not 0 <= frame.sample_index < grid.episode_sample_count:
            raise RuntimeContractError("review keyframe sample escapes the episode")
        if set(frame.source_positions_m) != set(canonical):
            raise RuntimeContractError("review keyframe source identity set changed")
        for source_id in canonical:
            _finite_vector(
                frame.source_positions_m[source_id],
                3,
                owner=f"{source_id} position",
            )
        listener_position = _finite_vector(
            frame.listener_position_m, 3, owner="listener position"
        )
        listener_orientation = _finite_vector(
            frame.listener_orientation_wxyz, 4, owner="listener orientation"
        )
        if not math.isclose(
            float(np.linalg.norm(np.asarray(listener_orientation))),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise RuntimeContractError("listener orientation must be unit length")
        if fixed_listener_position is None:
            fixed_listener_position = listener_position
            fixed_listener_orientation = listener_orientation
        elif (
            listener_position != fixed_listener_position
            or listener_orientation != fixed_listener_orientation
        ):
            raise RuntimeContractError("M5.1 research-review listener must remain fixed")
        previous_tick = frame.tick
        previous_sample = frame.sample_index
    if grid.keyframes[0].tick != 0 or grid.keyframes[0].sample_index != 0:
        raise RuntimeContractError("review RIR grid must start at episode zero")
    return grid


def research_review_trajectory_record(
    grid: ResearchReviewKeyframeGrid,
) -> dict[str, Any]:
    """Return the complete, hashable M5.1 trajectory and cadence record."""

    grid = validate_research_review_grid(grid)
    rate = grid.rir_keyframe_rate_hz
    return {
        "schema": RESEARCH_REVIEW_TRAJECTORY_SCHEMA,
        "profile": RESEARCH_REVIEW_PROFILE,
        "qualification_claim": False,
        "sampling": {
            "policy": "explicit_visual_frame_stride_v1",
            "visual_frame_count": grid.visual_frame_count,
            "visual_frame_rate": {
                "numerator": grid.visual_frame_rate_numerator,
                "denominator": grid.visual_frame_rate_denominator,
            },
            "rir_stride_frames": grid.rir_stride_frames,
            "rir_keyframe_rate": {
                "numerator": rate.numerator,
                "denominator": rate.denominator,
            },
            "sampled_visual_frame_indices": list(grid.visual_frame_indices),
            "final_interval_policy": "hold_last_rir_to_episode_end",
        },
        "timebase": {
            "timeline_tick_rate_hz": grid.timeline_tick_rate_hz,
            "sample_rate_hz": grid.sample_rate_hz,
            "episode_tick_count": grid.episode_tick_count,
            "episode_sample_count": grid.episode_sample_count,
        },
        "interpolation": "raised_cosine_source_time_partition_v1",
        "listener_motion": "fixed",
        "source_ids": list(grid.source_ids),
        "keyframes": [
            {
                "keyframe_index": keyframe_index,
                "visual_frame_index": visual_index,
                "tick": frame.tick,
                "sample_index": frame.sample_index,
                "source_positions_m": {
                    source_id: list(frame.source_positions_m[source_id])
                    for source_id in grid.source_ids
                },
                "listener_position_m": list(frame.listener_position_m),
                "listener_orientation_wxyz": list(
                    frame.listener_orientation_wxyz
                ),
            }
            for keyframe_index, (visual_index, frame) in enumerate(
                zip(grid.visual_frame_indices, grid.keyframes, strict=True)
            )
        ],
    }


def render_research_review_binaural_rir_sequence(
    scene: CompiledAcousticScene,
    simulation: M4SimulationConfig,
    *,
    grid: ResearchReviewKeyframeGrid,
    hrtf_file_path: str,
    source_radius_m: float = 0.0,
    listener_id: str = "listener0",
    listener_radius_m: float = 0.0,
) -> DynamicRIRSequence:
    """Render a variable M5.1 grid through one persistent binaural RLR context."""

    if not isinstance(scene, CompiledAcousticScene):
        raise RuntimeContractError("scene must be a validated CompiledAcousticScene")
    if not isinstance(simulation, M4SimulationConfig):
        raise RuntimeContractError("simulation must be an M4SimulationConfig")
    grid = validate_research_review_grid(grid)
    selected = simulation_with_layout(
        simulation, layout_type="binaural", channel_count=2
    )
    if selected.temporal_coherence:
        raise RuntimeContractError(
            "M5.1 pins RLR temporal_coherence=false; filter crossfade is explicit"
        )
    if not math.isclose(
        float(selected.sample_rate_hz),
        float(grid.sample_rate_hz),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise RuntimeContractError("review grid sample rate differs from RLR request")
    resolved_hrtf = Path(hrtf_file_path).resolve()
    if not resolved_hrtf.is_file():
        raise RuntimeContractError(
            "M5.1 binaural research review requires an explicit readable HRTF"
        )
    hrtf_file_path = str(resolved_hrtf)
    contract = _layout_contract("binaural", 2)
    trajectory = research_review_trajectory_record(grid)
    trajectory_sha256 = canonical_json_sha256(trajectory)

    habitat_module, runtime_report = load_habitat_runtime()
    native_configuration, config_readback = _native_configuration(
        habitat_module, selected
    )
    runtime_report["configuration_readback"] = config_readback
    runtime_report["output_contract"] = {
        **contract,
        "layout_type": "binaural",
        "channel_count": 2,
    }
    raw_by_keyframe: list[list[np.ndarray]] = []
    receipts: list[dict[str, Any]] = []
    ir_hashes: list[dict[str, str]] = []
    efficiencies: list[float] = []
    try:
        context = habitat_module.RLRAcousticContext(native_configuration)
        with tempfile.TemporaryDirectory(prefix="avengine-m5-1-rlr-db-") as temp_dir:
            private_database = Path(temp_dir) / "material_database.json"
            private_database.write_bytes(scene.material_database_bytes)
            if sha256_file(private_database) != scene.material_database_sha256:
                raise RuntimeContractError("private material database hash changed")
            raw_upload = context.load_acoustic_scene(
                str(private_database),
                list(scene.material_categories),
                list(scene.objects),
            )
        upload = _upload_report(raw_upload)
        _verify_upload_report(scene, upload)
        first = grid.keyframes[0]
        for source_id in grid.source_ids:
            context.add_source(
                source_id, first.source_positions_m[source_id], source_radius_m
            )
        context.add_listener(
            listener_id,
            first.listener_position_m,
            first.listener_orientation_wxyz,
            _native_layout(habitat_module, "binaural"),
            2,
            listener_radius_m,
            hrtf_file_path,
        )
        for keyframe_index, (visual_frame_index, frame) in enumerate(
            zip(grid.visual_frame_indices, grid.keyframes, strict=True)
        ):
            current_sources = tuple(
                RuntimeAnchor(
                    anchor_id=source_id,
                    position_m=frame.source_positions_m[source_id],
                    radius_m=source_radius_m,
                )
                for source_id in grid.source_ids
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
                listener.anchor_id,
                listener.position_m,
                listener.orientation_wxyz,
            )
            context.set_listener_radius(listener.anchor_id, listener.radius_m)
            raw_irs = context.simulate_owned()
            receipt = _native_registration_receipts(
                context,
                habitat_module,
                current_sources,
                listener,
                canonical_order=grid.source_ids,
                layout_type="binaural",
                channel_count=2,
                hrtf_file_path=hrtf_file_path,
            )
            by_source: dict[str, np.ndarray] = {}
            hashes: dict[str, str] = {}
            for raw_ir in raw_irs:
                source_id, samples = _owned_ir(
                    raw_ir,
                    expected_listener_id=listener_id,
                    expected_source_ids=grid.source_ids,
                    channel_count=2,
                    sample_rate_hz=grid.sample_rate_hz,
                )
                if source_id in by_source:
                    raise RuntimeContractError("RLR returned duplicate dynamic pair")
                by_source[source_id] = samples
                hashes[source_id] = hashlib.sha256(
                    samples.tobytes(order="C")
                ).hexdigest()
            if set(by_source) != set(grid.source_ids):
                raise RuntimeContractError("RLR omitted a named pair in dynamic sequence")
            raw_by_keyframe.append(
                [by_source[source_id] for source_id in grid.source_ids]
            )
            retained_receipt = _portable_hrtf_references(receipt, hrtf_file_path)
            receipts.append(
                {
                    "keyframe_index": keyframe_index,
                    "visual_frame_index": visual_frame_index,
                    "tick": frame.tick,
                    "sample_index": frame.sample_index,
                    **retained_receipt,
                }
            )
            ir_hashes.append(hashes)
            efficiency = float(context.indirect_ray_efficiency())
            if not math.isfinite(efficiency) or not 0.0 <= efficiency <= 1.0:
                raise RuntimeContractError(
                    "dynamic indirect ray efficiency is invalid"
                )
            efficiencies.append(efficiency)
    except (RuntimeContractError, RuntimeUnavailableError):
        raise
    except Exception as exc:
        raise RuntimeExecutionError(
            f"persistent M5.1 research-review RLR simulation failed: {exc}"
        ) from exc

    maximum_length = max(
        samples.shape[1]
        for keyframe_values in raw_by_keyframe
        for samples in keyframe_values
    )
    padded = np.zeros(
        (len(grid.keyframes), len(grid.source_ids), 2, maximum_length),
        dtype="<f4",
    )
    lengths = np.empty(
        (len(grid.keyframes), len(grid.source_ids)), dtype="<u4"
    )
    for keyframe_index, keyframe_values in enumerate(raw_by_keyframe):
        for source_index, samples in enumerate(keyframe_values):
            length = samples.shape[1]
            padded[keyframe_index, source_index, :, :length] = samples
            lengths[keyframe_index, source_index] = length
    metadata = {
        "schema": RESEARCH_REVIEW_RIR_SCHEMA,
        "profile": RESEARCH_REVIEW_PROFILE,
        "qualification_claim": False,
        "trajectory_sha256": trajectory_sha256,
        "trajectory": trajectory,
        "source_ids": list(grid.source_ids),
        "listener_id": listener_id,
        "layout_type": "binaural",
        "layout_id": contract["layout_id"],
        "channel_labels": list(contract["channel_labels"]),
        "normalization": contract["normalization"],
        "coordinate_frame": contract["coordinate_frame"],
        "sample_rate_hz": grid.sample_rate_hz,
        "context_policy": {
            "lifetime": "one_persistent_context_per_layout",
            "endpoint_update_order": list(grid.source_ids),
            "simulate_calls": len(grid.keyframes),
            "temporal_coherence": False,
            "reset_between_keyframes": False,
        },
        "runtime": runtime_report,
        "upload_report": upload,
        "scene_claim_boundary": {
            "package_mode": scene.manifest.get("package_mode"),
            "material_semantics": scene.manifest.get("materials", {}).get(
                "material_semantics"
            ),
            "material_qualification_claim": scene.manifest.get(
                "materials", {}
            ).get("qualification_claim"),
            "qa_status_by_report": {
                name: report.get("status")
                for name, report in sorted(scene.qa_reports.items())
            },
        },
        "endpoint_receipts": receipts,
        "ir_sha256_by_keyframe_source": ir_hashes,
        "indirect_ray_efficiency": efficiencies,
        "hrtf": {
            "input_role": "hrtf",
            "sha256": sha256_file(hrtf_file_path),
        },
    }
    return DynamicRIRSequence(
        samples=np.ascontiguousarray(padded),
        lengths=np.ascontiguousarray(lengths),
        source_ids=grid.source_ids,
        keyframe_ticks=tuple(frame.tick for frame in grid.keyframes),
        keyframe_samples=tuple(frame.sample_index for frame in grid.keyframes),
        sample_rate_hz=grid.sample_rate_hz,
        layout_type="binaural",
        layout_id=str(contract["layout_id"]),
        channel_labels=tuple(contract["channel_labels"]),
        trajectory_sha256=trajectory_sha256,
        metadata=metadata,
    )


def render_research_review_binaural_audio(
    dry_by_source: Mapping[str, Any],
    sequence: DynamicRIRSequence,
    *,
    grid: ResearchReviewKeyframeGrid,
) -> tuple[dict[str, DynamicStemResult], np.ndarray]:
    """Apply M5's deterministic convolution to one variable M5.1 episode."""

    grid = validate_research_review_grid(grid)
    if not isinstance(sequence, DynamicRIRSequence):
        raise RuntimeContractError("sequence must be DynamicRIRSequence")
    trajectory = research_review_trajectory_record(grid)
    expected_hash = canonical_json_sha256(trajectory)
    if sequence.trajectory_sha256 != expected_hash:
        raise RuntimeContractError("RIR sequence trajectory hash differs from review grid")
    if sequence.source_ids != grid.source_ids:
        raise RuntimeContractError("RIR sequence source IDs differ from review grid")
    if sequence.keyframe_ticks != tuple(frame.tick for frame in grid.keyframes):
        raise RuntimeContractError("RIR sequence keyframe ticks differ from review grid")
    if sequence.keyframe_samples != tuple(
        frame.sample_index for frame in grid.keyframes
    ):
        raise RuntimeContractError("RIR sequence samples differ from review grid")
    if sequence.sample_rate_hz != grid.sample_rate_hz:
        raise RuntimeContractError("RIR sequence sample rate differs from review grid")
    if sequence.layout_type != "binaural" or sequence.channel_labels != (
        "left",
        "right",
    ):
        raise RuntimeContractError("M5.1 review audio requires binaural left/right RIRs")
    return render_dynamic_stems_and_mix(
        dry_by_source,
        sequence.samples,
        sequence.lengths,
        source_ids=grid.source_ids,
        keyframe_samples=sequence.keyframe_samples,
        output_sample_count=grid.episode_sample_count,
    )


__all__ = [
    "RESEARCH_REVIEW_PROFILE",
    "RESEARCH_REVIEW_RIR_SCHEMA",
    "RESEARCH_REVIEW_TRAJECTORY_SCHEMA",
    "ResearchReviewKeyframeGrid",
    "build_strided_review_keyframes",
    "render_research_review_binaural_audio",
    "render_research_review_binaural_rir_sequence",
    "research_review_trajectory_record",
    "validate_research_review_grid",
]

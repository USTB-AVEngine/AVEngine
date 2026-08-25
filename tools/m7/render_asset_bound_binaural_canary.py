#!/usr/bin/env python3
"""Render two real dry recordings through one completed asset-bound RIR cache.

This is a small M7 research canary, not a dataset-release builder.  It never
replans trajectories or calls RLR: the asset-bound plan and its completed
cache must already exist.  Both source slots receive one real five-second dry
bus and the output retains the two binaural stems plus their exact sum.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from avengine.contracts.json_io import load_json, write_json
from avengine.spatial_audio.audio import read_float32_wav, write_float32_wav
from avengine.m5.audio import M5_AUDIO_SAMPLE_COUNT, M5_AUDIO_SAMPLE_RATE_HZ
from avengine.m6x.rir_cache import load_cached_rir_episode
from avengine.m7.asset_bound_audio import (
    AssetBoundAudioError,
    float32_stems_and_exact_mix,
    prepare_dry_audio,
    render_asset_bound_binaural,
)


SCHEMA = "avengine_m7_asset_bound_binaural_canary_delivery_v1"
SOURCE_SLOTS = ("source1", "source2")


def _slot_values(values: list[str], *, name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        slot, separator, value = raw.partition("=")
        if separator != "=" or slot not in SOURCE_SLOTS or not value:
            raise AssetBoundAudioError(
                f"{name} must use source1=value or source2=value"
            )
        if slot in result:
            raise AssetBoundAudioError(f"{name} specifies {slot} more than once")
        result[slot] = value
    if tuple(sorted(result)) != SOURCE_SLOTS:
        raise AssetBoundAudioError(
            f"{name} must specify source1 and source2 exactly once"
        )
    return result


def _slot_numbers(values: list[str], *, name: str) -> dict[str, float]:
    raw = _slot_values(values, name=name)
    result: dict[str, float] = {}
    for slot, value in raw.items():
        try:
            number = float(value)
        except ValueError as exc:
            raise AssetBoundAudioError(f"{name} {slot} must be numeric") from exc
        if not np.isfinite(number) or number < 0.0:
            raise AssetBoundAudioError(f"{name} {slot} must be finite and non-negative")
        result[slot] = number
    return result


def _find_episode_binding(plan_root: Path, episode_id: str) -> dict[str, Any]:
    report = load_json(plan_root / "asset_emitter_binding_report.json")
    for scenario in report.get("scenarios", []):
        if scenario.get("output_episode_id") == episode_id:
            binding = scenario.get("binding_report")
            if isinstance(binding, dict):
                return binding
    raise AssetBoundAudioError(
        f"asset-bound plan has no binding report for episode {episode_id!r}"
    )


def _write_audio(
    path: Path, samples: np.ndarray, *, role: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    artifact = write_float32_wav(
        path,
        samples,
        M5_AUDIO_SAMPLE_RATE_HZ,
        metadata={"role": role, "qualification_claim": False, **metadata},
    )
    readback = read_float32_wav(
        artifact.audio_path,
        sidecar_path=artifact.sidecar_path,
        verify_sidecar=True,
    )
    expected = np.asarray(samples, dtype=np.float32)
    if readback.samples.shape != expected.shape or not np.array_equal(
        readback.samples, expected
    ):
        raise AssetBoundAudioError(f"float32 WAVE readback differs: {path}")
    return {
        "path": str(artifact.audio_path.name),
        "sidecar_path": str(artifact.sidecar_path.name),
        "audio_sha256": artifact.audio_sha256,
        "sidecar_sha256": artifact.sidecar_sha256,
        "sample_rate_hz": artifact.sample_rate_hz,
        "sample_count": artifact.frame_count,
        "channel_count": artifact.channel_count,
        "peak_absolute": float(np.max(np.abs(samples))),
    }


def render(
    *,
    plan_root: Path,
    rir_cache_root: Path,
    episode_id: str,
    audio_paths: dict[str, str],
    channel_policies: dict[str, str],
    gains: dict[str, float],
    source_starts: dict[str, int],
    fade_samples: int,
    maximum_mixture_peak: float,
    output: Path,
) -> Path:
    started = time.perf_counter()
    plan_root = plan_root.resolve()
    rir_cache_root = rir_cache_root.resolve()
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output}")
    if not np.isfinite(maximum_mixture_peak) or not 0.0 < maximum_mixture_peak <= 1.0:
        raise AssetBoundAudioError("maximum_mixture_peak must be in (0, 1]")
    binding = _find_episode_binding(plan_root, episode_id)
    cached = load_cached_rir_episode(
        cache_root=rir_cache_root,
        plan_path=plan_root / "rir_job_plan.json",
        episode_id=episode_id,
        frame_count=75,
        frame_rate_hz=15,
    )
    if (
        cached.layout_type != "binaural"
        or cached.sample_rate_hz != M5_AUDIO_SAMPLE_RATE_HZ
        or cached.channel_labels != ("left", "right")
        or cached.source_slot_ids != SOURCE_SLOTS
    ):
        raise AssetBoundAudioError(
            "cache is not the required two-channel 16 kHz source-slot RIR grid"
        )
    prepared = {
        slot: prepare_dry_audio(
            audio_paths[slot],
            channel_policy=channel_policies[slot],
            source_start_sample=source_starts[slot],
            linear_gain=gains[slot],
            fade_samples=fade_samples,
        )
        for slot in SOURCE_SLOTS
    }
    stems, _float64_mixture = render_asset_bound_binaural(
        {slot: prepared[slot].samples for slot in SOURCE_SLOTS},
        rir_samples=cached.samples,
        rir_lengths=cached.lengths,
        source_ids=cached.source_slot_ids,
        keyframe_samples=cached.keyframe_samples,
    )
    stored_stems, stored_mixture = float32_stems_and_exact_mix(
        stems, source_ids=cached.source_slot_ids
    )
    peak = float(np.max(np.abs(stored_mixture)))
    if peak > maximum_mixture_peak:
        raise AssetBoundAudioError(
            f"mixture peak {peak:.6f} exceeds maximum_mixture_peak {maximum_mixture_peak:.6f}; "
            "reduce an explicit --linear-gain"
        )
    if output.parent.exists() and output.parent.is_symlink():
        raise AssetBoundAudioError("output parent may not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"refusing to replace staging output: {staging}")
    try:
        staging.mkdir()
        dry_records = {}
        stem_records = {}
        for slot in SOURCE_SLOTS:
            dry_records[slot] = _write_audio(
                staging / "audio" / "dry" / f"{slot}.wav",
                prepared[slot].samples[np.newaxis, :],
                role="m7_asset_bound_dry_source_bus",
                metadata={"source_slot_id": slot, "episode_id": episode_id},
            )
            stem_records[slot] = _write_audio(
                staging / "audio" / "binaural" / f"{slot}_stem.wav",
                stored_stems[slot],
                role="m7_asset_bound_binaural_source_stem",
                metadata={"source_slot_id": slot, "episode_id": episode_id},
            )
        mixture_record = _write_audio(
            staging / "audio" / "binaural" / "mixture.wav",
            stored_mixture,
            role="m7_asset_bound_binaural_mixture",
            metadata={
                "episode_id": episode_id,
                "layout": "native_RLR_HRTF_binaural_left_right",
                "mixture": "exact_source1_plus_source2_stem_sum",
                "normalization": False,
                "limiting": False,
            },
        )
        write_json(
            staging / "asset_audio_binding.json",
            {
                "schema": "avengine_m7_asset_bound_audio_binding_v1",
                "status": "pass",
                "episode_id": episode_id,
                "asset_emitter_binding": binding,
                "dry_sources": {slot: prepared[slot].record for slot in SOURCE_SLOTS},
                "rir_cache": cached.evidence,
                "source_slots": list(SOURCE_SLOTS),
                "binaural_layout": "native_RLR_HRTF_binaural_left_right",
                "both_sources_active": True,
                "mixture_is_exact_stem_sum": True,
                "persisted_mixture_arithmetic": "float32_stem_sum_in_canonical_source_order",
                "normalization": False,
                "limiting": False,
                "looping": False,
                "audio_sample_rate_hz": M5_AUDIO_SAMPLE_RATE_HZ,
                "audio_sample_count": M5_AUDIO_SAMPLE_COUNT,
                "audio_channel_count": 2,
                "dataset_mixture_peak_absolute": peak,
            },
        )
        write_json(
            staging / "timing.json",
            {
                "schema": "avengine_m7_asset_bound_audio_timing_v1",
                "status": "pass",
                "native_rlr_calls": 0,
                "visual_render_calls": 0,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        write_json(
            staging / "delivery.json",
            {
                "schema": SCHEMA,
                "status": "pass",
                "claim_boundary": (
                    "research audio canary only; it reuses a completed asset-bound "
                    "RIR cache and does not claim dataset admission"
                ),
                "episode_id": episode_id,
                "outputs": {
                    "asset_audio_binding": "asset_audio_binding.json",
                    "timing": "timing.json",
                    "dry_audio": dry_records,
                    "binaural_stems": stem_records,
                    "binaural_mixture": mixture_record,
                },
            },
        )
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--rir-cache", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--source-audio", action="append", required=True)
    parser.add_argument("--channel-policy", action="append", required=True)
    parser.add_argument("--linear-gain", action="append", required=True)
    parser.add_argument("--source-start-sample", action="append", required=True)
    parser.add_argument("--fade-samples", type=int, default=80)
    parser.add_argument("--maximum-mixture-peak", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render(
        plan_root=args.plan_root,
        rir_cache_root=args.rir_cache,
        episode_id=args.episode_id,
        audio_paths=_slot_values(args.source_audio, name="source-audio"),
        channel_policies=_slot_values(args.channel_policy, name="channel-policy"),
        gains=_slot_numbers(args.linear_gain, name="linear-gain"),
        source_starts={
            slot: int(value)
            for slot, value in _slot_values(
                args.source_start_sample, name="source-start-sample"
            ).items()
        },
        fade_samples=args.fade_samples,
        maximum_mixture_peak=args.maximum_mixture_peak,
        output=args.output,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

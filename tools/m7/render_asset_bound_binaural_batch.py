#!/usr/bin/env python3
"""Assemble many binaural training items from one completed asset-bound cache.

This is deliberately an audio-only batch stage.  It groups all dry-audio
variants of a route together, opens that route's already-completed RIR grid
once, and never starts Habitat or RLR.  RGB/Topdown videos remain a small
separately selected review subset rather than a 1,000-way duplicated render.

Each concrete visual asset is mapped to its own declared recording.  A caller
cannot accidentally use a dog clip for a cat: every asset ID found in the
asset-emitter binding report must have exactly one explicit audio declaration.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping
import wave

import numpy as np

from avengine.contracts.json_io import load_json, write_json
from avengine.m4.audio import read_float32_wav, write_float32_wav
from avengine.m5.audio import (
    M5_AUDIO_SAMPLE_COUNT,
    M5_AUDIO_SAMPLE_RATE_HZ,
    raised_cosine_partition,
)
from avengine.m6x.rir_cache import RIRCacheSession
from avengine.m7.asset_bound_audio import (
    AssetBoundAudioError,
    PreparedDryAudio,
    float32_stems_and_exact_mix,
    prepare_dry_audio,
    render_asset_bound_binaural,
)


SCHEMA = "avengine_m7_asset_bound_binaural_batch_delivery_v1"
SOURCE_SLOTS = ("source1", "source2")


def _slot_mapping(values: list[str], *, name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if separator != "=" or not key or not value:
            raise AssetBoundAudioError(f"{name} must use ASSET_ID=value")
        if key in result:
            raise AssetBoundAudioError(f"{name} specifies {key!r} more than once")
        result[key] = value
    if not result:
        raise AssetBoundAudioError(f"{name} must contain at least one asset")
    return result


def _slot_numbers(values: list[str], *, name: str) -> dict[str, float]:
    raw = _slot_mapping(values, name=name)
    result: dict[str, float] = {}
    for key, value in raw.items():
        try:
            number = float(value)
        except ValueError as exc:
            raise AssetBoundAudioError(f"{name} {key!r} must be numeric") from exc
        if not np.isfinite(number) or number < 0.0:
            raise AssetBoundAudioError(
                f"{name} {key!r} must be finite and non-negative"
            )
        result[key] = number
    return result


def _pcm16_wave_header(path: str | Path) -> tuple[int, int]:
    source = Path(path).resolve()
    if not source.is_file():
        raise AssetBoundAudioError(f"dry audio is not a regular file: {source}")
    try:
        with wave.open(str(source), "rb") as handle:
            if (
                handle.getsampwidth() != 2
                or handle.getcomptype() != "NONE"
                or handle.getnchannels() not in {1, 2}
            ):
                raise AssetBoundAudioError(
                    "dry audio must be uncompressed PCM16 WAVE with one or two channels"
                )
            return handle.getframerate(), handle.getnframes()
    except (OSError, wave.Error) as exc:
        raise AssetBoundAudioError(f"cannot read PCM WAVE {source}: {exc}") from exc


def variant_start_samples(
    *, source_sample_rate_hz: int, source_sample_count: int, variant_count: int
) -> tuple[int, ...]:
    """Choose distinct full-five-second source windows without looping audio."""

    if (
        isinstance(source_sample_rate_hz, bool)
        or not isinstance(source_sample_rate_hz, int)
        or source_sample_rate_hz < 1
        or isinstance(source_sample_count, bool)
        or not isinstance(source_sample_count, int)
        or source_sample_count < 1
        or isinstance(variant_count, bool)
        or not isinstance(variant_count, int)
        or variant_count < 1
    ):
        raise AssetBoundAudioError("source header or variant count is invalid")
    needed = int(
        np.ceil(M5_AUDIO_SAMPLE_COUNT * source_sample_rate_hz / M5_AUDIO_SAMPLE_RATE_HZ)
    )
    maximum_start = source_sample_count - needed
    if maximum_start < 0:
        raise AssetBoundAudioError(
            "dry audio is shorter than one five-second episode; looping is disabled"
        )
    if variant_count == 1:
        return (0,)
    values = tuple(
        int(round(value))
        for value in np.linspace(0, maximum_start, variant_count, endpoint=True)
    )
    if len(set(values)) != len(values):
        raise AssetBoundAudioError(
            "dry audio is too short to provide distinct non-looped starts for all variants"
        )
    return values


def _binding_assets(plan_root: Path) -> dict[str, dict[str, str]]:
    report = load_json(plan_root / "asset_emitter_binding_report.json")
    if report.get("status") != "pass" or not isinstance(report.get("scenarios"), list):
        raise AssetBoundAudioError("asset-bound plan binding report is invalid")
    result: dict[str, dict[str, str]] = {}
    for raw in report["scenarios"]:
        if not isinstance(raw, Mapping) or not isinstance(
            raw.get("output_episode_id"), str
        ):
            raise AssetBoundAudioError("asset-bound scenario report is malformed")
        binding = raw.get("binding_report")
        raw_bindings = binding.get("bindings") if isinstance(binding, Mapping) else None
        if not isinstance(raw_bindings, list) or len(raw_bindings) != len(SOURCE_SLOTS):
            raise AssetBoundAudioError("asset-bound scenario omits source bindings")
        assets: dict[str, str] = {}
        for source in raw_bindings:
            slot = source.get("source_slot_id") if isinstance(source, Mapping) else None
            asset_id = source.get("asset_id") if isinstance(source, Mapping) else None
            if (
                slot not in SOURCE_SLOTS
                or slot in assets
                or not isinstance(asset_id, str)
                or not asset_id
            ):
                raise AssetBoundAudioError("asset-bound scenario contains an invalid asset")
            assets[slot] = asset_id
        if set(assets) != set(SOURCE_SLOTS):
            raise AssetBoundAudioError("asset-bound scenario omits a source binding")
        episode_id = raw["output_episode_id"]
        if episode_id in result:
            raise AssetBoundAudioError("asset-bound report repeats an episode ID")
        result[episode_id] = assets
    if not result:
        raise AssetBoundAudioError("asset-bound report contains no episodes")
    return result


def _write_and_verify(
    path: Path,
    samples: np.ndarray,
    *,
    role: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = write_float32_wav(path, samples, M5_AUDIO_SAMPLE_RATE_HZ, metadata=dict(metadata) | {"role": role})
    readback = read_float32_wav(
        artifact.audio_path, sidecar_path=artifact.sidecar_path, verify_sidecar=True
    )
    expected = np.asarray(samples, dtype=np.float32)
    if readback.samples.shape != expected.shape or not np.array_equal(
        readback.samples, expected
    ):
        raise AssetBoundAudioError(f"float32 WAVE readback differs: {path}")
    return {
        "path": artifact.audio_path.name,
        "sidecar_path": artifact.sidecar_path.name,
        "audio_sha256": artifact.audio_sha256,
        "sidecar_sha256": artifact.sidecar_sha256,
        "peak_absolute": float(np.max(np.abs(expected))),
    }


def _prepare_asset_variants(
    *,
    asset_audio: Mapping[str, str],
    asset_channel_policies: Mapping[str, str],
    asset_gains: Mapping[str, float],
    required_asset_ids: set[str],
    variants_per_episode: int,
    fade_samples: int,
) -> tuple[dict[tuple[str, int], PreparedDryAudio], dict[str, Any]]:
    if set(asset_audio) != required_asset_ids:
        missing = sorted(required_asset_ids - set(asset_audio))
        extra = sorted(set(asset_audio) - required_asset_ids)
        raise AssetBoundAudioError(
            f"asset-audio IDs must exactly match bound assets; missing={missing}, extra={extra}"
        )
    if set(asset_channel_policies) != required_asset_ids:
        raise AssetBoundAudioError("asset-channel-policy IDs must exactly match bound assets")
    if set(asset_gains) != required_asset_ids:
        raise AssetBoundAudioError("asset-linear-gain IDs must exactly match bound assets")
    prepared: dict[tuple[str, int], PreparedDryAudio] = {}
    records: dict[str, Any] = {}
    for asset_id in sorted(required_asset_ids):
        sample_rate_hz, sample_count = _pcm16_wave_header(asset_audio[asset_id])
        starts = variant_start_samples(
            source_sample_rate_hz=sample_rate_hz,
            source_sample_count=sample_count,
            variant_count=variants_per_episode,
        )
        variants = []
        for variant_index, start in enumerate(starts):
            item = prepare_dry_audio(
                asset_audio[asset_id],
                channel_policy=asset_channel_policies[asset_id],
                source_start_sample=start,
                linear_gain=asset_gains[asset_id],
                fade_samples=fade_samples,
            )
            prepared[(asset_id, variant_index)] = item
            variants.append(
                {
                    "variant_index": variant_index,
                    "source_start_sample": start,
                    "record": item.record,
                }
            )
        records[asset_id] = {
            "source_sample_rate_hz": sample_rate_hz,
            "source_sample_count": sample_count,
            "variant_count": variants_per_episode,
            "variants": variants,
        }
    return prepared, records


def render_batch(
    *,
    plan_root: Path,
    rir_cache_root: Path,
    asset_audio: Mapping[str, str],
    asset_channel_policies: Mapping[str, str],
    asset_gains: Mapping[str, float],
    variants_per_episode: int,
    fade_samples: int,
    maximum_mixture_peak: float,
    retain_stems: bool,
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
    if isinstance(variants_per_episode, bool) or variants_per_episode < 1:
        raise AssetBoundAudioError("variants_per_episode must be positive")
    bindings = _binding_assets(plan_root)
    required_assets = {asset for slots in bindings.values() for asset in slots.values()}
    prepared, dry_variant_records = _prepare_asset_variants(
        asset_audio=asset_audio,
        asset_channel_policies=asset_channel_policies,
        asset_gains=asset_gains,
        required_asset_ids=required_assets,
        variants_per_episode=variants_per_episode,
        fade_samples=fade_samples,
    )
    if output.parent.exists() and output.parent.is_symlink():
        raise AssetBoundAudioError("output parent may not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"refusing to replace staging output: {staging}")
    phase_seconds: defaultdict[str, float] = defaultdict(float)
    sample_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    partitions_by_keyframe_grid: dict[tuple[int, ...], np.ndarray] = {}
    shared_rir_shards: dict[Path, dict[str, Any]] = {}
    try:
        staging.mkdir()
        mixture_root = staging / "audio" / "binaural"
        stem_root = mixture_root / "stems"
        rir_session = RIRCacheSession(
            cache_root=rir_cache_root,
            plan_path=plan_root / "rir_job_plan.json",
            frame_count=75,
            frame_rate_hz=15,
            shared_shard_cache=shared_rir_shards,
        )
        cache_load_count = 0
        for episode_ordinal, episode_id in enumerate(sorted(bindings)):
            cache_started = time.perf_counter()
            cached = rir_session.load_episode(episode_id)
            phase_seconds["rir_cache_load"] += time.perf_counter() - cache_started
            cache_load_count += 1
            if (
                cached.layout_type != "binaural"
                or cached.sample_rate_hz != M5_AUDIO_SAMPLE_RATE_HZ
                or cached.channel_labels != ("left", "right")
                or cached.source_slot_ids != SOURCE_SLOTS
            ):
                raise AssetBoundAudioError("cache is not 16 kHz two-channel binaural")
            partition_key = tuple(cached.keyframe_samples)
            partition_weights = partitions_by_keyframe_grid.get(partition_key)
            if partition_weights is None:
                partition_weights = raised_cosine_partition(
                    partition_key, M5_AUDIO_SAMPLE_COUNT
                )
                partitions_by_keyframe_grid[partition_key] = partition_weights
            episode_records.append(
                {
                    "episode_id": episode_id,
                    "episode_ordinal": episode_ordinal,
                    "asset_ids_by_source_slot": bindings[episode_id],
                    "rir_cache": cached.evidence,
                    "cache_load_policy": "loaded_once_then_reused_for_all_episode_variants",
                }
            )
            for variant_index in range(variants_per_episode):
                sample_id = f"{episode_id}__v{variant_index:02d}"
                dry_by_source = {
                    slot: prepared[(bindings[episode_id][slot], variant_index)].samples
                    for slot in SOURCE_SLOTS
                }
                render_started = time.perf_counter()
                stems, _mixture64 = render_asset_bound_binaural(
                    dry_by_source,
                    rir_samples=cached.samples,
                    rir_lengths=cached.lengths,
                    source_ids=cached.source_slot_ids,
                    keyframe_samples=cached.keyframe_samples,
                    partition_weights=partition_weights,
                )
                stored_stems, stored_mixture = float32_stems_and_exact_mix(
                    stems, source_ids=cached.source_slot_ids
                )
                phase_seconds["dynamic_convolution"] += time.perf_counter() - render_started
                peak = float(np.max(np.abs(stored_mixture)))
                if peak > maximum_mixture_peak:
                    raise AssetBoundAudioError(
                        f"{sample_id} mixture peak {peak:.6f} exceeds maximum_mixture_peak "
                        f"{maximum_mixture_peak:.6f}; reduce its declared asset gain"
                    )
                write_started = time.perf_counter()
                mixture_record = _write_and_verify(
                    mixture_root / f"{sample_id}.wav",
                    stored_mixture,
                    role="m7_asset_bound_binaural_training_mixture",
                    metadata={
                        "sample_id": sample_id,
                        "episode_id": episode_id,
                        "variant_index": variant_index,
                        "mixture": "exact_source1_plus_source2_stem_sum",
                        "normalization": False,
                        "limiting": False,
                    },
                )
                stem_records: dict[str, Any] = {}
                if retain_stems:
                    for slot in SOURCE_SLOTS:
                        stem_records[slot] = _write_and_verify(
                            stem_root / slot / f"{sample_id}.wav",
                            stored_stems[slot],
                            role="m7_asset_bound_binaural_training_stem",
                            metadata={
                                "sample_id": sample_id,
                                "episode_id": episode_id,
                                "variant_index": variant_index,
                                "source_slot_id": slot,
                            },
                        )
                phase_seconds["wave_write_and_readback"] += time.perf_counter() - write_started
                sample_records.append(
                    {
                        "sample_id": sample_id,
                        "episode_id": episode_id,
                        "variant_index": variant_index,
                        "asset_ids_by_source_slot": bindings[episode_id],
                        "dry_variant_ids_by_source_slot": {
                            slot: {
                                "asset_id": bindings[episode_id][slot],
                                "variant_index": variant_index,
                            }
                            for slot in SOURCE_SLOTS
                        },
                        "both_sources_active": True,
                        "audio": {
                            "sample_rate_hz": M5_AUDIO_SAMPLE_RATE_HZ,
                            "sample_count": M5_AUDIO_SAMPLE_COUNT,
                            "channel_count": 2,
                            "layout": "native_RLR_HRTF_binaural_left_right",
                            "mixture": mixture_record,
                            "stems_retained": retain_stems,
                            "stems": stem_records,
                            "mixture_is_exact_stem_sum_before_delivery": True,
                            "peak_absolute": peak,
                        },
                    }
                )
        write_json(staging / "dry_audio_variants.json", {
            "schema": "avengine_m7_asset_bound_dry_variant_library_v1",
            "status": "pass",
            "assets": dry_variant_records,
            "normalization": False,
            "limiting": False,
            "looping": False,
        })
        write_json(staging / "episodes.json", {
            "schema": "avengine_m7_asset_bound_episode_cache_index_v1",
            "status": "pass",
            "episodes": episode_records,
        })
        write_json(staging / "samples.json", {
            "schema": "avengine_m7_asset_bound_binaural_training_samples_v1",
            "status": "pass",
            "sample_count": len(sample_records),
            "samples": sample_records,
        })
        total_wall = time.perf_counter() - started
        timing = {
            "schema": "avengine_m7_asset_bound_binaural_batch_timing_v1",
            "status": "pass",
            "native_rlr_calls": 0,
            "visual_render_calls": 0,
            "rir_cache_load_count": cache_load_count,
            "rir_cache_load_policy": "one_load_per_route_then_all_its_variants",
            "rir_shard_residency": {
                "policy": "verify_each_native_RIR_shard_once_then_reuse_in_process",
                "resident_shard_count": len(shared_rir_shards),
                "resident_sample_payload_bytes": int(
                    sum(value["samples"].nbytes for value in shared_rir_shards.values())
                ),
            },
            "episode_count": len(bindings),
            "variants_per_episode": variants_per_episode,
            "sample_count": len(sample_records),
            "phase_seconds": dict(phase_seconds),
            "wall_seconds": total_wall,
            "samples_per_wall_second": len(sample_records) / total_wall,
            "projected_1000_sample_seconds": 1000.0 / (len(sample_records) / total_wall),
        }
        write_json(staging / "timing.json", timing)
        write_json(staging / "delivery.json", {
            "schema": SCHEMA,
            "status": "pass",
            "qualification_claim": False,
            "claim_boundary": (
                "research throughput delivery: reuses a completed asset-bound RIR "
                "cache, writes binaural mixtures and labels, and does not render video "
                "or claim dataset admission"
            ),
            "episode_count": len(bindings),
            "variants_per_episode": variants_per_episode,
            "sample_count": len(sample_records),
            "both_sources_active": True,
            "binaural_layout": "native_RLR_HRTF_binaural_left_right",
            "outputs": {
                "dry_audio_variants": "dry_audio_variants.json",
                "episode_cache_index": "episodes.json",
                "samples": "samples.json",
                "timing": "timing.json",
                "mixtures": "audio/binaural/",
                "stems": "audio/binaural/stems/" if retain_stems else None,
            },
        })
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--rir-cache", type=Path, required=True)
    parser.add_argument("--asset-audio", action="append", required=True)
    parser.add_argument("--asset-channel-policy", action="append", required=True)
    parser.add_argument("--asset-linear-gain", action="append", required=True)
    parser.add_argument("--variants-per-episode", type=int, default=10)
    parser.add_argument("--fade-samples", type=int, default=80)
    parser.add_argument("--maximum-mixture-peak", type=float, default=0.95)
    parser.add_argument("--retain-stems", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = render_batch(
        plan_root=args.plan_root,
        rir_cache_root=args.rir_cache,
        asset_audio=_slot_mapping(args.asset_audio, name="asset-audio"),
        asset_channel_policies=_slot_mapping(
            args.asset_channel_policy, name="asset-channel-policy"
        ),
        asset_gains=_slot_numbers(args.asset_linear_gain, name="asset-linear-gain"),
        variants_per_episode=args.variants_per_episode,
        fade_samples=args.fade_samples,
        maximum_mixture_peak=args.maximum_mixture_peak,
        retain_stems=args.retain_stems,
        output=args.output,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

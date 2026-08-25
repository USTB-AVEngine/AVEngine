#!/usr/bin/env python3
"""Render intermittent-window binaural mixtures for a declared episode subset.

Sub-windows are planned deterministically per episode and slot, declared in
the AudioProgram event vocabulary and enforced by a raised-cosine gating
envelope on the dry clip before convolution. The declared windows are the
ground truth for temporal questions; the RIR cache is reused untouched.
Dry preparation is recovered verbatim from the original batch's dry-variant
library so gating is the only difference to the continuous realization.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from avengine.contracts.json_io import (  # noqa: E402
    file_record,
    load_json,
    write_json,
)
from avengine.timeline.audio import (  # noqa: E402
    M5_AUDIO_SAMPLE_COUNT,
    M5_AUDIO_SAMPLE_RATE_HZ,
    raised_cosine_partition,
)
from avengine.acoustics.rir_cache import RIRCacheSession  # noqa: E402
from avengine.m7.asset_bound_audio import (  # noqa: E402
    AssetBoundAudioError,
    float32_stems_and_exact_mix,
    render_asset_bound_binaural,
)
from avengine.qa.intermittent import (  # noqa: E402
    DEFAULT_FADE_SAMPLES,
    EDGE_MARGIN_SAMPLES,
    MAX_EVENT_SAMPLES,
    MIN_EVENT_SAMPLES,
    MIN_GAP_SAMPLES,
    WINDOW_AUTHORITY,
    event_records,
    frame_window,
    gating_envelope,
    plan_slot_windows,
)

BATCH_SCHEMA = "avengine_qa_intermittent_audio_batch_v1"
SOURCE_SLOTS = ("source1", "source2")


class IntermittentBatchError(RuntimeError):
    pass


def _load_batch_module():
    path = REPOSITORY / "tools/m7/render_asset_bound_binaural_batch.py"
    spec = importlib.util.spec_from_file_location("m7_binaural_batch", path)
    if spec is None or spec.loader is None:
        raise IntermittentBatchError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dry_declarations(
    dry_library: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, str], dict[str, float]]:
    audio: dict[str, str] = {}
    policies: dict[str, str] = {}
    gains: dict[str, float] = {}
    for asset_id, entry in dry_library["assets"].items():
        record = entry["variants"][0]["record"]
        audio[asset_id] = record["input"]["path"]
        policies[asset_id] = record["channel_policy"]
        gains[asset_id] = float(record["linear_gain"])
    return audio, policies, gains


def _select_episodes(episode_ids: list[str], count: int, seed: str) -> list[str]:
    def key(episode_id: str) -> str:
        return hashlib.sha256(f"{seed}\0{episode_id}".encode("utf-8")).hexdigest()

    return sorted(sorted(episode_ids, key=key)[:count])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--rir-cache", type=Path, required=True)
    parser.add_argument("--original-batch", type=Path, required=True)
    parser.add_argument("--episode-count", type=int, default=200)
    parser.add_argument("--seed", default="qa_intermittent_v1_20260727")
    parser.add_argument("--fade-samples", type=int, default=DEFAULT_FADE_SAMPLES)
    parser.add_argument("--maximum-mixture-peak", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    batch_module = _load_batch_module()
    plan_root = args.plan_root.resolve()
    original_batch = args.original_batch.resolve()

    dry_library = load_json(original_batch / "dry_audio_variants.json")
    asset_audio, asset_policies, asset_gains = _dry_declarations(dry_library)

    bindings = batch_module._binding_assets(plan_root)
    selected = _select_episodes(sorted(bindings), args.episode_count, args.seed)
    required_assets = {
        asset for episode_id in selected for asset in bindings[episode_id].values()
    }
    prepared, dry_variant_records = batch_module._prepare_asset_variants(
        asset_audio={asset: asset_audio[asset] for asset in required_assets},
        asset_channel_policies={
            asset: asset_policies[asset] for asset in required_assets
        },
        asset_gains={asset: asset_gains[asset] for asset in required_assets},
        required_asset_ids=required_assets,
        variants_per_episode=1,
        fade_samples=args.fade_samples,
    )

    output = args.output.resolve()
    if output.exists():
        raise IntermittentBatchError(f"refusing to replace output: {output}")
    mixture_root = output / "audio" / "binaural"

    shared_shards: dict[Path, dict[str, Any]] = {}
    rir_session = RIRCacheSession(
        cache_root=args.rir_cache.resolve(),
        plan_path=plan_root / "rir_job_plan.json",
        frame_count=75,
        frame_rate_hz=15,
        shared_shard_cache=shared_shards,
    )
    partitions: dict[tuple[int, ...], np.ndarray] = {}
    sample_records: list[dict[str, Any]] = []
    for episode_id in selected:
        cached = rir_session.load_episode(episode_id)
        if (
            cached.layout_type != "binaural"
            or cached.sample_rate_hz != M5_AUDIO_SAMPLE_RATE_HZ
            or cached.channel_labels != ("left", "right")
            or tuple(cached.source_slot_ids) != SOURCE_SLOTS
        ):
            raise AssetBoundAudioError("cache is not 16 kHz two-channel binaural")
        partition_key = tuple(cached.keyframe_samples)
        weights = partitions.get(partition_key)
        if weights is None:
            weights = raised_cosine_partition(partition_key, M5_AUDIO_SAMPLE_COUNT)
            partitions[partition_key] = weights

        events_by_slot: dict[str, list[dict[str, Any]]] = {}
        dry_by_source: dict[str, np.ndarray] = {}
        for slot in SOURCE_SLOTS:
            windows = plan_slot_windows(
                seed=args.seed, episode_id=episode_id, slot_id=slot
            )
            envelope = gating_envelope(windows, fade_samples=args.fade_samples)
            dry = prepared[(bindings[episode_id][slot], 0)].samples
            dry_by_source[slot] = np.asarray(dry, dtype=np.float64) * envelope
            slot_events = event_records(
                slot_id=slot, windows=windows, fade_samples=args.fade_samples
            )
            for event in slot_events:
                start_frame, end_frame = frame_window(
                    event["start_tick"], event["end_tick_exclusive"]
                )
                event["start_frame"] = start_frame
                event["end_frame"] = end_frame
            events_by_slot[slot] = slot_events

        stems, _mixture64 = render_asset_bound_binaural(
            dry_by_source,
            rir_samples=cached.samples,
            rir_lengths=cached.lengths,
            source_ids=cached.source_slot_ids,
            keyframe_samples=cached.keyframe_samples,
            partition_weights=weights,
        )
        _stems32, stored_mixture = float32_stems_and_exact_mix(
            stems, source_ids=cached.source_slot_ids
        )
        peak = float(np.max(np.abs(stored_mixture)))
        if peak > args.maximum_mixture_peak:
            raise AssetBoundAudioError(
                f"{episode_id} gated mixture peak {peak:.6f} exceeds the maximum"
            )
        sample_id = f"{episode_id}__int00"
        mixture_record = batch_module._write_and_verify(
            mixture_root / f"{sample_id}.wav",
            stored_mixture,
            role="qa_intermittent_binaural_mixture",
            metadata={
                "sample_id": sample_id,
                "episode_id": episode_id,
                "window_authority": WINDOW_AUTHORITY,
                "normalization": False,
                "limiting": False,
            },
        )
        sample_records.append(
            {
                "sample_id": sample_id,
                "episode_id": episode_id,
                "asset_ids_by_source_slot": bindings[episode_id],
                "window_authority": WINDOW_AUTHORITY,
                "events_by_source_slot": events_by_slot,
                "audio": {
                    "sample_rate_hz": M5_AUDIO_SAMPLE_RATE_HZ,
                    "sample_count": M5_AUDIO_SAMPLE_COUNT,
                    "channel_count": 2,
                    "layout": "native_RLR_HRTF_binaural_left_right",
                    "mixture": mixture_record,
                    "peak_absolute": peak,
                },
            }
        )

    manifest = {
        "schema": BATCH_SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "claim_boundary": (
            "Declared intermittent-window re-mix over the retained RIR cache; "
            "windows are ground truth by construction and no dataset "
            "admission is granted"
        ),
        "window_authority": WINDOW_AUTHORITY,
        "planner": {
            "seed": args.seed,
            "min_event_samples": MIN_EVENT_SAMPLES,
            "max_event_samples": MAX_EVENT_SAMPLES,
            "min_gap_samples": MIN_GAP_SAMPLES,
            "edge_margin_samples": EDGE_MARGIN_SAMPLES,
            "fade_samples": args.fade_samples,
            "event_count_choices": [2, 3],
        },
        "episode_count": len(sample_records),
        "inputs": {
            "plan_root": str(plan_root),
            "rir_cache": str(args.rir_cache.resolve()),
            "original_batch": str(original_batch),
            "dry_variant_library": file_record(
                original_batch / "dry_audio_variants.json",
                relative_to=original_batch,
            ),
        },
        "dry_audio_variants": dry_variant_records,
        "samples": sample_records,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output / "intermittent_batch_manifest.json", manifest)
    total_events = sum(
        len(events)
        for sample in sample_records
        for events in sample["events_by_source_slot"].values()
    )
    print(
        f"QA_INTERMITTENT_BATCH_OK output={output} episodes={len(sample_records)} "
        f"events={total_events} wall_seconds={manifest['wall_seconds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

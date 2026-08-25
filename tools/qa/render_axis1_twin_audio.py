#!/usr/bin/env python3
"""Render route-swap twin binaural audio for axis-1 certified episodes.

The twin of an episode keeps every visual byte and both trajectories and
swaps the dry-audio routing between the two source slots; because the RIR
cache is dry-audio independent this is a pure CPU re-mix of cached impulse
responses. Dry-audio preparation is reproduced exactly from the original
batch's dry-variant library (same recording, channel policy and gain per
asset), so the only difference to the original mixture is the routing.

Each rendered twin is bound back into its twin fact table (mixture path,
hash and peak replace the explicit zeroed placeholders) and the bound
table is re-validated against the repository schema. The output manifest
declares the counterfactual construction; no dataset admission is granted.
"""

from __future__ import annotations

import argparse
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
from avengine.m5.audio import (  # noqa: E402
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
from avengine.qa.certify import TWIN_SUFFIX  # noqa: E402

TWIN_BATCH_SCHEMA = "avengine_qa_axis1_twin_audio_batch_v1"
SOURCE_SLOTS = ("source1", "source2")


class TwinAudioError(RuntimeError):
    pass


def _load_batch_module():
    path = REPOSITORY / "tools/m7/render_asset_bound_binaural_batch.py"
    spec = importlib.util.spec_from_file_location("m7_binaural_batch", path)
    if spec is None or spec.loader is None:
        raise TwinAudioError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dry_declarations(
    dry_library: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, str], dict[str, float]]:
    """Recover per-asset audio declarations from the original batch library."""

    audio: dict[str, str] = {}
    policies: dict[str, str] = {}
    gains: dict[str, float] = {}
    for asset_id, entry in dry_library["assets"].items():
        variants = entry.get("variants")
        if not variants:
            raise TwinAudioError(f"dry library has no variants for {asset_id!r}")
        record = variants[0]["record"]
        audio[asset_id] = record["input"]["path"]
        policies[asset_id] = record["channel_policy"]
        gains[asset_id] = float(record["linear_gain"])
    return audio, policies, gains


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--rir-cache", type=Path, required=True)
    parser.add_argument(
        "--original-batch",
        type=Path,
        required=True,
        help="Original asset-bound binaural batch (dry declarations + originals)",
    )
    parser.add_argument(
        "--certificates",
        type=Path,
        required=True,
        help="certificates.json from certify_axis1_questions.py",
    )
    parser.add_argument(
        "--twin-facts",
        type=Path,
        required=True,
        help="facts_twin directory produced by certify_axis1_questions.py",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPOSITORY / "schemas/avengine_qa_fact_table_v1.schema.json",
    )
    parser.add_argument("--fade-samples", type=int, default=80)
    parser.add_argument("--maximum-mixture-peak", type=float, default=0.95)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import jsonschema  # noqa: PLC0415

    started = time.perf_counter()
    batch_module = _load_batch_module()
    plan_root = args.plan_root.resolve()
    original_batch = args.original_batch.resolve()

    certificates = load_json(args.certificates)
    granted_episodes = sorted(
        {
            record["twin_episode_id"][: -len(TWIN_SUFFIX)]
            for record in certificates["certificates"]
            if record.get("status") == "granted"
        }
    )
    if not granted_episodes:
        raise TwinAudioError("no granted axis-1 certificates; nothing to render")
    if args.limit is not None:
        granted_episodes = granted_episodes[: args.limit]

    dry_library = load_json(original_batch / "dry_audio_variants.json")
    asset_audio, asset_policies, asset_gains = _dry_declarations(dry_library)
    original_samples = {
        sample["episode_id"]: sample
        for sample in load_json(original_batch / "samples.json")["samples"]
    }

    bindings = batch_module._binding_assets(plan_root)
    missing = sorted(set(granted_episodes) - set(bindings))
    if missing:
        raise TwinAudioError(f"granted episodes missing from plan: {missing[:3]}")
    required_assets = {
        asset
        for episode_id in granted_episodes
        for asset in bindings[episode_id].values()
    }
    unknown_assets = sorted(required_assets - set(asset_audio))
    if unknown_assets:
        raise TwinAudioError(
            f"dry library lacks declarations for bound assets: {unknown_assets}"
        )
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

    validator = jsonschema.Draft202012Validator(load_json(args.schema))
    output = args.output.resolve()
    if output.exists():
        raise TwinAudioError(f"refusing to replace output: {output}")
    mixture_root = output / "audio" / "binaural"
    bound_facts_root = output / "facts_twin_bound"
    bound_facts_root.mkdir(parents=True, exist_ok=True)

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
    for episode_id in granted_episodes:
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
        swapped = {
            "source1": bindings[episode_id]["source2"],
            "source2": bindings[episode_id]["source1"],
        }
        dry_by_source = {
            slot: prepared[(swapped[slot], 0)].samples for slot in SOURCE_SLOTS
        }
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
                f"{episode_id} twin mixture peak {peak:.6f} exceeds the maximum"
            )
        twin_id = f"{episode_id}{TWIN_SUFFIX}"
        sample_id = f"{twin_id}__v00"
        mixture_record = batch_module._write_and_verify(
            mixture_root / f"{sample_id}.wav",
            stored_mixture,
            role="qa_axis1_twin_binaural_mixture",
            metadata={
                "sample_id": sample_id,
                "episode_id": episode_id,
                "counterfactual_kind": "axis1_route_swap",
                "dry_routing": "swapped_between_source_slots",
                "normalization": False,
                "limiting": False,
            },
        )
        original_sha = original_samples[episode_id]["audio"]["mixture"]["audio_sha256"]
        if mixture_record["audio_sha256"] == original_sha:
            raise TwinAudioError(
                f"{episode_id}: twin mixture is byte-identical to the original"
            )

        twin_fact_path = args.twin_facts.resolve() / f"{twin_id}.json"
        twin_fact = load_json(twin_fact_path)
        twin_fact["audio"] = {
            **twin_fact["audio"],
            "mixture_path": mixture_record["path"],
            "mixture_sha256": mixture_record["audio_sha256"],
            "peak_absolute": peak,
        }
        errors = sorted(
            validator.iter_errors(twin_fact), key=lambda err: list(err.absolute_path)
        )
        if errors:
            raise TwinAudioError(
                f"{twin_id}: bound twin fact table violates the schema: "
                f"{errors[0].message}"
            )
        write_json(bound_facts_root / f"{twin_id}.json", twin_fact)

        sample_records.append(
            {
                "sample_id": sample_id,
                "episode_id": episode_id,
                "twin_episode_id": twin_id,
                "asset_ids_by_source_slot": bindings[episode_id],
                "voice_asset_ids_by_source_slot": swapped,
                "audio": {
                    "sample_rate_hz": M5_AUDIO_SAMPLE_RATE_HZ,
                    "sample_count": M5_AUDIO_SAMPLE_COUNT,
                    "channel_count": 2,
                    "layout": "native_RLR_HRTF_binaural_left_right",
                    "mixture": mixture_record,
                    "peak_absolute": peak,
                },
                "original_mixture_sha256": original_sha,
                "bound_twin_fact_table": file_record(
                    bound_facts_root / f"{twin_id}.json", relative_to=output
                ),
            }
        )

    manifest = {
        "schema": TWIN_BATCH_SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "claim_boundary": (
            "Route-swap twin mixtures re-mixed from the retained RIR cache; "
            "visuals are the original episodes' bytes; no dataset admission"
        ),
        "counterfactual_kind": "axis1_route_swap",
        "episode_count": len(sample_records),
        "inputs": {
            "plan_root": str(plan_root),
            "rir_cache": str(args.rir_cache.resolve()),
            "original_batch": str(original_batch),
            "certificates": file_record(
                args.certificates.resolve(),
                relative_to=args.certificates.resolve().parent,
            ),
        },
        "dry_audio_variants": dry_variant_records,
        "samples": sample_records,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output / "twin_batch_manifest.json", manifest)
    print(
        f"QA_AXIS1_TWIN_AUDIO_OK output={output} twins={len(sample_records)} "
        f"wall_seconds={manifest['wall_seconds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

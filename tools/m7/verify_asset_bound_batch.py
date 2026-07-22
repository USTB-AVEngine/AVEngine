#!/usr/bin/env python3
"""Verify the complete M7 asset-bound binaural throughput batch.

This is an artifact-level verifier, rather than a claim based on a successful
launcher.  It checks that every delivered WAV/sidecar is a valid exact M7
sample, that all RIR uses point at the final asset-bound emitter centers, and
optionally compares every matching sample to a prior delivery byte-for-byte.
It does not turn this research batch or its review videos into dataset
admission.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from avengine.contracts.json_io import sha256_file, write_json
from avengine.m4.audio import read_float32_wav


FRAME_COUNT = 75
SOURCE_SLOTS = ("source1", "source2")
SAMPLE_RATE_HZ = 16_000
SAMPLE_COUNT = 80_000
SCHEMA = "avengine_m7_asset_bound_batch_correctness_report_v1"


class BatchVerificationError(RuntimeError):
    """A batch artifact does not satisfy its declared M7 contract."""


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchVerificationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise BatchVerificationError(f"JSON object required: {path}")
    return value


def _finite_paths(value: Any, *, owner: str) -> np.ndarray:
    paths = np.asarray(value, dtype=np.float64)
    if paths.shape != (2, FRAME_COUNT, 3) or not np.all(np.isfinite(paths)):
        raise BatchVerificationError(f"{owner} must be finite [2,75,3]")
    return np.ascontiguousarray(paths)


def _bank(plan_root: Path) -> tuple[dict[str, int], np.ndarray, Mapping[str, Any]]:
    record = _json(plan_root / "trajectory_bank.json")
    if record.get("episode_count") != 100 or record.get("frame_count") != FRAME_COUNT:
        raise BatchVerificationError("trajectory bank count or frame count differs")
    arrays = np.load(plan_root / "trajectory_bank.npz", allow_pickle=False)
    source_slots = tuple(str(value) for value in arrays["source_slot_ids"])
    episode_ids = tuple(str(value) for value in arrays["episode_ids"])
    centers = np.asarray(arrays["source_center_paths_m"], dtype=np.float64)
    expected_shape = (len(episode_ids), 2, FRAME_COUNT, 3)
    if source_slots != SOURCE_SLOTS or centers.shape != expected_shape:
        raise BatchVerificationError("trajectory bank arrays differ from M7 layout")
    if len(episode_ids) != len(set(episode_ids)) or not np.all(np.isfinite(centers)):
        raise BatchVerificationError("trajectory bank episode IDs or centers are invalid")
    return {item: index for index, item in enumerate(episode_ids)}, centers, record


def _assert_center_gate(plan_root: Path, episode_ids: Sequence[str]) -> dict[str, float]:
    gate = _json(plan_root / "navmesh_center_gate.json")
    rows = gate.get("sources")
    if gate.get("status") != "pass" or not isinstance(rows, Mapping):
        raise BatchVerificationError("asset-bound navmesh center gate is invalid")
    minima: dict[str, float] = {}
    for episode_id in episode_ids:
        for slot in SOURCE_SLOTS:
            value = rows.get(f"{episode_id}::{slot}")
            if not isinstance(value, Mapping) or value.get("status") != "pass":
                raise BatchVerificationError(f"center gate fails for {episode_id} {slot}")
            clearance = float(value.get("minimum_navmesh_clearance_m"))
            if not np.isfinite(clearance) or clearance < 0.0:
                raise BatchVerificationError("center gate clearance is invalid")
            minima[f"{episode_id}::{slot}"] = clearance
    return minima


def _assert_rir_uses(plan_root: Path, indices: Mapping[str, int], centers: np.ndarray) -> dict[str, int]:
    plan = _json(plan_root / "rir_job_plan.json")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or plan.get("unique_rir_job_count") != len(jobs):
        raise BatchVerificationError("RIR job plan is malformed")
    uses_by_episode: Counter[str] = Counter()
    use_count = 0
    for row in jobs:
        if not isinstance(row, Mapping):
            raise BatchVerificationError("RIR job is not an object")
        position = np.asarray(row.get("source_position_m"), dtype=np.float64)
        uses = row.get("uses")
        if position.shape != (3,) or not np.all(np.isfinite(position)) or not isinstance(uses, list) or not uses:
            raise BatchVerificationError("RIR job omits a finite source position or use")
        for use in uses:
            if not isinstance(use, Mapping):
                raise BatchVerificationError("RIR job use is malformed")
            episode_id = use.get("episode_id")
            slot = use.get("source_slot_id")
            frame = use.get("frame_index")
            if (
                not isinstance(episode_id, str)
                or episode_id not in indices
                or slot not in SOURCE_SLOTS
                or isinstance(frame, bool)
                or not isinstance(frame, int)
                or not 0 <= frame < FRAME_COUNT
            ):
                raise BatchVerificationError("RIR job use has an invalid episode/source/frame")
            slot_index = SOURCE_SLOTS.index(slot)
            error = float(np.max(np.abs(position - centers[indices[episode_id], slot_index, frame])))
            if error > 1.0e-9:
                raise BatchVerificationError(
                    f"RIR job source position differs from final emitter center by {error:.3g} m"
                )
            uses_by_episode[episode_id] += 1
            use_count += 1
    if set(uses_by_episode) != set(indices) or any(count != 50 for count in uses_by_episode.values()):
        raise BatchVerificationError("each selected route must retain 25 RIR keyframes per source")
    return {"unique_jobs": len(jobs), "uses": use_count, "uses_per_episode": 50}


def _binding_assets(plan_root: Path) -> dict[str, dict[str, str]]:
    record = _json(plan_root / "asset_emitter_binding_report.json")
    rows = record.get("scenarios")
    if record.get("status") != "pass" or not isinstance(rows, list):
        raise BatchVerificationError("asset-emitter binding report is invalid")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("output_episode_id"), str):
            raise BatchVerificationError("asset-emitter binding scenario is invalid")
        report = row.get("binding_report")
        bindings = report.get("bindings") if isinstance(report, Mapping) else None
        if not isinstance(bindings, list) or len(bindings) != 2:
            raise BatchVerificationError("asset-emitter binding omits a source")
        assets: dict[str, str] = {}
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise BatchVerificationError("asset-emitter binding is malformed")
            slot, asset = binding.get("source_slot_id"), binding.get("asset_id")
            if slot not in SOURCE_SLOTS or slot in assets or not isinstance(asset, str) or not asset:
                raise BatchVerificationError("asset-emitter binding has an invalid asset")
            assets[slot] = asset
        result[row["output_episode_id"]] = assets
    if len(result) != 100:
        raise BatchVerificationError("asset-emitter binding count differs")
    return result


def _assert_episode_cache_index(
    *, plan_root: Path, batch_root: Path, expected_assets: Mapping[str, Mapping[str, str]]
) -> dict[str, int]:
    plan = _json(plan_root / "rir_job_plan.json")
    plan_rows = plan.get("jobs")
    index = _json(batch_root / "episodes.json")
    episodes = index.get("episodes")
    if not isinstance(plan_rows, list) or not isinstance(episodes, list):
        raise BatchVerificationError("RIR plan or episode cache index is malformed")
    jobs_by_id = {
        row.get("job_id"): row
        for row in plan_rows
        if isinstance(row, Mapping) and isinstance(row.get("job_id"), str)
    }
    if len(jobs_by_id) != len(plan_rows):
        raise BatchVerificationError("RIR plan repeats or omits a job ID")
    rows_by_episode = {
        row.get("episode_id"): row
        for row in episodes
        if isinstance(row, Mapping) and isinstance(row.get("episode_id"), str)
    }
    if set(rows_by_episode) != set(expected_assets):
        raise BatchVerificationError("episode cache index differs from selected asset-bound episodes")
    jobs_checked = 0
    for episode_id, row in rows_by_episode.items():
        if row.get("asset_ids_by_source_slot") != expected_assets[episode_id]:
            raise BatchVerificationError("episode cache index has a wrong asset binding")
        cache = row.get("rir_cache")
        jobs = cache.get("jobs") if isinstance(cache, Mapping) else None
        if not isinstance(jobs, list) or len(jobs) != 50:
            raise BatchVerificationError("each episode cache index must contain 50 RIR uses")
        seen: set[tuple[str, int]] = set()
        for cached in jobs:
            if not isinstance(cached, Mapping):
                raise BatchVerificationError("episode cache job is malformed")
            job_id = cached.get("job_id")
            slot = cached.get("source_slot_id")
            frame = cached.get("visual_frame_index")
            planned = jobs_by_id.get(job_id)
            if (
                not isinstance(job_id, str)
                or slot not in SOURCE_SLOTS
                or isinstance(frame, bool)
                or not isinstance(frame, int)
                or not isinstance(planned, Mapping)
            ):
                raise BatchVerificationError("episode cache job has an invalid ID/source/frame")
            position = np.asarray(cached.get("source_position_m"), dtype=np.float64)
            planned_position = np.asarray(planned.get("source_position_m"), dtype=np.float64)
            if (
                position.shape != (3,)
                or planned_position.shape != (3,)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(planned_position))
                or np.max(np.abs(position - planned_position)) > 1.0e-9
                or not any(
                    use.get("episode_id") == episode_id
                    and use.get("source_slot_id") == slot
                    and use.get("frame_index") == frame
                    for use in planned.get("uses", ())
                    if isinstance(use, Mapping)
                )
            ):
                raise BatchVerificationError("episode cache job differs from its RIR plan use")
            key = (slot, frame)
            if key in seen:
                raise BatchVerificationError("episode cache repeats one source/frame RIR use")
            seen.add(key)
            jobs_checked += 1
        if {slot for slot, _ in seen} != set(SOURCE_SLOTS):
            raise BatchVerificationError("episode cache omits one source slot")
    return {"episode_count": len(rows_by_episode), "checked_cache_uses": jobs_checked}


def _verify_batch(
    batch_root: Path,
    *,
    expected_assets: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    delivery = _json(batch_root / "delivery.json")
    samples_record = _json(batch_root / "samples.json")
    samples = samples_record.get("samples")
    if (
        delivery.get("status") != "pass"
        or delivery.get("sample_count") != 1000
        or delivery.get("episode_count") != 100
        or delivery.get("variants_per_episode") != 10
        or delivery.get("both_sources_active") is not True
        or not isinstance(samples, list)
        or len(samples) != 1000
    ):
        raise BatchVerificationError("batch delivery summary differs from 100x10 M7 contract")
    audio_root = batch_root / "audio" / "binaural"
    sample_ids: set[str] = set()
    variants: defaultdict[str, set[int]] = defaultdict(set)
    assets_by_episode: dict[str, Mapping[str, str]] = {}
    maximum_peak = 0.0
    for row in samples:
        if not isinstance(row, Mapping):
            raise BatchVerificationError("sample record is not an object")
        sample_id = row.get("sample_id")
        episode_id = row.get("episode_id")
        variant = row.get("variant_index")
        assets = row.get("asset_ids_by_source_slot")
        audio = row.get("audio")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in sample_ids
            or not isinstance(episode_id, str)
            or episode_id not in expected_assets
            or isinstance(variant, bool)
            or not isinstance(variant, int)
            or not 0 <= variant < 10
            or not isinstance(assets, Mapping)
            or dict(assets) != dict(expected_assets[episode_id])
            or row.get("both_sources_active") is not True
            or not isinstance(audio, Mapping)
        ):
            raise BatchVerificationError("sample identity/binding differs from the plan")
        if assets_by_episode.setdefault(episode_id, assets) != assets:
            raise BatchVerificationError("one episode has inconsistent asset bindings")
        mixture = audio.get("mixture")
        if (
            audio.get("channel_count") != 2
            or audio.get("sample_rate_hz") != SAMPLE_RATE_HZ
            or audio.get("sample_count") != SAMPLE_COUNT
            or audio.get("layout") != "native_RLR_HRTF_binaural_left_right"
            or audio.get("mixture_is_exact_stem_sum_before_delivery") is not True
            or not isinstance(mixture, Mapping)
            or not isinstance(mixture.get("path"), str)
            or not isinstance(mixture.get("audio_sha256"), str)
            or not isinstance(mixture.get("sidecar_path"), str)
        ):
            raise BatchVerificationError("sample audio contract is invalid")
        wav = audio_root / str(mixture["path"])
        sidecar = audio_root / str(mixture["sidecar_path"])
        if not wav.is_file() or not sidecar.is_file():
            raise BatchVerificationError(f"sample media is missing: {sample_id}")
        if sha256_file(wav) != mixture["audio_sha256"]:
            raise BatchVerificationError(f"sample WAV hash differs: {sample_id}")
        if (
            not isinstance(mixture.get("sidecar_sha256"), str)
            or sha256_file(sidecar) != mixture["sidecar_sha256"]
        ):
            raise BatchVerificationError(f"sample sidecar hash differs: {sample_id}")
        rendered = read_float32_wav(wav, sidecar_path=sidecar, verify_sidecar=True)
        if rendered.sample_rate_hz != SAMPLE_RATE_HZ or rendered.samples.shape != (2, SAMPLE_COUNT):
            raise BatchVerificationError(f"sample WAV header differs: {sample_id}")
        sample_ids.add(sample_id)
        variants[episode_id].add(variant)
        maximum_peak = max(maximum_peak, float(np.max(np.abs(rendered.samples))))
    if set(variants) != set(expected_assets) or any(value != set(range(10)) for value in variants.values()):
        raise BatchVerificationError("each selected route must have exactly variants v00..v09")
    return {
        "sample_count": len(sample_ids),
        "episode_count": len(variants),
        "variants_per_episode": 10,
        "maximum_readback_peak_absolute": maximum_peak,
    }


def _compare_reference(batch_root: Path, reference_root: Path) -> Mapping[str, Any]:
    current = _json(batch_root / "samples.json").get("samples")
    reference = _json(reference_root / "samples.json").get("samples")
    if not isinstance(current, list) or not isinstance(reference, list):
        raise BatchVerificationError("reference batch samples are malformed")
    current_rows = {str(row.get("sample_id")): row for row in current if isinstance(row, Mapping)}
    reference_rows = {str(row.get("sample_id")): row for row in reference if isinstance(row, Mapping)}
    shared = sorted(set(current_rows) & set(reference_rows))
    if not shared:
        raise BatchVerificationError("reference batch shares no sample IDs")
    equal = 0
    for sample_id in shared:
        first = current_rows[sample_id]
        second = reference_rows[sample_id]
        first_audio = first.get("audio")
        second_audio = second.get("audio")
        first_mix = first_audio.get("mixture") if isinstance(first_audio, Mapping) else None
        second_mix = second_audio.get("mixture") if isinstance(second_audio, Mapping) else None
        if not isinstance(first_mix, Mapping) or not isinstance(second_mix, Mapping):
            raise BatchVerificationError("reference sample lacks mixture metadata")
        first_path = batch_root / "audio" / "binaural" / str(first_mix["path"])
        second_path = reference_root / "audio" / "binaural" / str(second_mix["path"])
        if first_path.read_bytes() != second_path.read_bytes():
            raise BatchVerificationError(f"reference byte mismatch: {sample_id}")
        equal += 1
    return {"shared_sample_count": len(shared), "byte_identical_sample_count": equal}


def verify(
    *,
    plan_root: Path,
    batch_root: Path,
    output: Path,
    reference_batch_root: Path | None = None,
) -> Path:
    """Write one pass report after inspecting every batch audio artifact."""

    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise BatchVerificationError(f"refusing to replace report: {output}")
    plan_root = plan_root.resolve()
    batch_root = batch_root.resolve()
    indices, centers, bank_record = _bank(plan_root)
    clearances = _assert_center_gate(plan_root, tuple(indices))
    rir = _assert_rir_uses(plan_root, indices, centers)
    assets = _binding_assets(plan_root)
    if set(assets) != set(indices):
        raise BatchVerificationError("bound asset episode IDs differ from the selected bank")
    batch = _verify_batch(batch_root, expected_assets=assets)
    episode_cache = _assert_episode_cache_index(
        plan_root=plan_root,
        batch_root=batch_root,
        expected_assets=assets,
    )
    reference = (
        None
        if reference_batch_root is None
        else _compare_reference(batch_root, reference_batch_root.resolve())
    )
    motion_counts = Counter(str(row["motion_case"]) for row in bank_record["episodes"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "research_only": True,
        "qualification_claim": False,
        "claim_boundary": "artifact-level throughput correctness only; no new native RLR execution, visual render, or dataset admission",
        "plan_root": str(plan_root),
        "batch_root": str(batch_root),
        "trajectory_bank": {
            "selected_episode_count": len(indices),
            "frame_count": FRAME_COUNT,
            "motion_case_counts": dict(sorted(motion_counts.items())),
            "minimum_source_center_clearance_m": min(clearances.values()),
        },
        "rir_plan": rir,
        "episode_cache_index": episode_cache,
        "batch": batch,
        "reference_byte_comparison": reference,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-batch-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = verify(
        plan_root=args.plan_root,
        batch_root=args.batch_root,
        output=args.output,
        reference_batch_root=args.reference_batch_root,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

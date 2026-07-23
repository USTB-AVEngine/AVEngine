#!/usr/bin/env python3
"""Build one resumable HDF5 cache from the AVEngine 1,000-WAV closure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from avengine_v43.hdf5_data import (
    EXPECTED_SPLIT_COUNTS,
    HDF5_SCHEMA,
    OUTPUT_FRAME_COUNT,
    SAMPLE_COUNT,
    SOURCE_COUNT,
    SPLIT_TO_CODE,
    _decode_json_attribute,
    _decode_string,
)
from avengine_v43.labels import (
    LegacyV4AudioError,
    caption_for_asset,
    label_tracks_for_source,
)
from run_training_smoke import (
    SOURCE_SLOTS,
    _audio_root,
    _load_json,
    _read_mixture,
    _trajectory_lookup,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--trajectory-bank", type=Path, required=True)
    parser.add_argument("--rir-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flush-every", type=int, default=10)
    return parser.parse_args()


def _load_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to build the model cache") from exc
    return h5py


def _ordered_samples(
    dataset_index: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if dataset_index.get("status") != "pass":
        raise LegacyV4AudioError("dataset index status is not pass")
    values = dataset_index.get("samples")
    if not isinstance(values, list) or len(values) != 1000:
        raise LegacyV4AudioError("dataset index must contain exactly 1,000 samples")
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        selected = [
            value
            for value in values
            if isinstance(value, Mapping) and value.get("split") == split
        ]
        if len(selected) != expected_count:
            raise LegacyV4AudioError(
                f"{split} must contain {expected_count} samples"
            )
        for sample in selected:
            sample_id = sample.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in seen:
                raise LegacyV4AudioError("sample IDs are invalid or duplicated")
            seen.add(sample_id)
        result.extend(selected)
    return result


def _input_identity(
    *,
    dataset_index: Path,
    trajectory_bank: Path,
    rir_plan: Path,
) -> Mapping[str, str]:
    return {
        "dataset_index": str(dataset_index.resolve()),
        "trajectory_bank": str(trajectory_bank.resolve()),
        "rir_plan": str(rir_plan.resolve()),
    }


def _initialize(
    *,
    h5py: Any,
    path: Path,
    identity: Mapping[str, str],
    caption_table: list[str],
) -> None:
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "x", libver="latest") as file:
        file.attrs["schema"] = HDF5_SCHEMA
        file.attrs["completed_sample_count"] = 0
        file.attrs["input_identity_json"] = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
        )
        file.attrs["caption_table_json"] = json.dumps(
            caption_table,
            ensure_ascii=False,
        )
        file.attrs["sample_rate_hz"] = 16_000
        file.attrs["duration_seconds"] = 5.0
        file.create_dataset(
            "mixture",
            shape=(1000, SAMPLE_COUNT, 2),
            dtype=np.float32,
            chunks=(1, SAMPLE_COUNT, 2),
        )
        file.create_dataset(
            "azimuth_deg",
            shape=(1000, SOURCE_COUNT, OUTPUT_FRAME_COUNT),
            dtype=np.float32,
            chunks=(16, SOURCE_COUNT, OUTPUT_FRAME_COUNT),
        )
        file.create_dataset(
            "caption_id",
            shape=(1000, SOURCE_COUNT),
            dtype=np.uint8,
        )
        file.create_dataset("split_code", shape=(1000,), dtype=np.uint8)
        file.create_dataset(
            "sample_id",
            shape=(1000,),
            dtype=string_type,
        )
        file.create_dataset(
            "episode_id",
            shape=(1000,),
            dtype=string_type,
        )
        file.create_dataset(
            "asset_id",
            shape=(1000, SOURCE_COUNT),
            dtype=string_type,
        )
        file.flush()


def main() -> int:
    args = _arguments()
    if args.flush_every <= 0:
        raise LegacyV4AudioError("--flush-every must be positive")
    output = args.output.resolve()
    incomplete = output.with_name(f".{output.name}.incomplete")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset_index_path = args.dataset_index.resolve()
    trajectory_bank_path = args.trajectory_bank.resolve()
    rir_plan_path = args.rir_plan.resolve()
    identity = _input_identity(
        dataset_index=dataset_index_path,
        trajectory_bank=trajectory_bank_path,
        rir_plan=rir_plan_path,
    )
    dataset_index = _load_json(dataset_index_path)
    trajectory_bank = _load_json(trajectory_bank_path)
    rir_plan = _load_json(rir_plan_path)
    samples = _ordered_samples(dataset_index)
    episodes = _trajectory_lookup(trajectory_bank)
    audio_root = _audio_root(dataset_index)
    listener_position = rir_plan.get("listener_position_m")
    listener_orientation = rir_plan.get("listener_orientation_wxyz")
    frame_rate = trajectory_bank.get("frame_rate_hz")
    if (
        not isinstance(listener_position, list)
        or not isinstance(listener_orientation, list)
        or not isinstance(frame_rate, (int, float))
        or frame_rate <= 0
    ):
        raise LegacyV4AudioError("trajectory/RIR listener contract is invalid")

    captions = sorted(
        {
            caption_for_asset(str(asset_id))
            for sample in samples
            for asset_id in sample["asset_ids_by_source_slot"].values()
        }
    )
    caption_to_id = {caption: index for index, caption in enumerate(captions)}
    h5py = _load_h5py()
    if not incomplete.exists():
        _initialize(
            h5py=h5py,
            path=incomplete,
            identity=identity,
            caption_table=captions,
        )

    started = time.perf_counter()
    with h5py.File(incomplete, "r+", libver="latest") as file:
        if _decode_string(file.attrs.get("schema")) != HDF5_SCHEMA:
            raise LegacyV4AudioError("incomplete HDF5 schema does not match")
        stored_identity = _decode_json_attribute(
            file.attrs["input_identity_json"]
        )
        if stored_identity != identity:
            raise LegacyV4AudioError("incomplete HDF5 input identity does not match")
        start_index = int(file.attrs["completed_sample_count"])
        if start_index < 0 or start_index > len(samples):
            raise LegacyV4AudioError("incomplete HDF5 progress is invalid")

        for sample_index in range(start_index, len(samples)):
            sample = samples[sample_index]
            sample_id = str(sample["sample_id"])
            episode_id = str(sample["episode_id"])
            audio_path = sample.get("audio_path")
            assets = sample.get("asset_ids_by_source_slot")
            if (
                not isinstance(audio_path, str)
                or not isinstance(assets, Mapping)
                or episode_id not in episodes
            ):
                raise LegacyV4AudioError(f"{sample_id} binding is invalid")
            paths = episodes[episode_id].get("source_center_paths_m")
            if not isinstance(paths, Mapping):
                raise LegacyV4AudioError(f"{episode_id} lacks source paths")

            mixture = _read_mixture(audio_root / audio_path)
            azimuth = np.empty(
                (SOURCE_COUNT, OUTPUT_FRAME_COUNT),
                dtype=np.float32,
            )
            caption_ids = np.empty(SOURCE_COUNT, dtype=np.uint8)
            asset_ids: list[str] = []
            for source_index, source_slot in enumerate(SOURCE_SLOTS):
                asset_id = assets.get(source_slot)
                positions = paths.get(source_slot)
                if not isinstance(asset_id, str) or positions is None:
                    raise LegacyV4AudioError(
                        f"{sample_id} lacks {source_slot}"
                    )
                labels = label_tracks_for_source(
                    positions,
                    source_frame_rate_hz=float(frame_rate),
                    target_duration_seconds=5.0,
                    target_frame_count=OUTPUT_FRAME_COUNT,
                    listener_position_m=listener_position,
                    listener_orientation_wxyz=listener_orientation,
                )
                azimuth[source_index] = np.asarray(
                    labels["native_360_azimuth_deg"],
                    dtype=np.float32,
                )
                caption_ids[source_index] = caption_to_id[
                    caption_for_asset(asset_id)
                ]
                asset_ids.append(asset_id)
            if caption_ids[0] == caption_ids[1]:
                raise LegacyV4AudioError(
                    f"{sample_id} has ambiguous text-only captions"
                )

            file["mixture"][sample_index] = mixture
            file["azimuth_deg"][sample_index] = azimuth
            file["caption_id"][sample_index] = caption_ids
            file["split_code"][sample_index] = SPLIT_TO_CODE[str(sample["split"])]
            file["sample_id"][sample_index] = sample_id
            file["episode_id"][sample_index] = episode_id
            file["asset_id"][sample_index] = asset_ids
            file.attrs["completed_sample_count"] = sample_index + 1
            if (
                (sample_index + 1) % args.flush_every == 0
                or sample_index + 1 == len(samples)
            ):
                file.flush()
                print(
                    json.dumps(
                        {
                            "completed": sample_index + 1,
                            "total": len(samples),
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    os.replace(incomplete, output)
    from avengine_v43.hdf5_data import Hdf5QueryBank

    with Hdf5QueryBank(output) as bank:
        readback = {
            split: len(bank.queries(split)) for split in EXPECTED_SPLIT_COUNTS
        }
        first = bank.read_mixtures(bank.queries("train")[:2])
        if first.shape != (2, SAMPLE_COUNT, 2):
            raise LegacyV4AudioError("HDF5 readback failed")
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(output),
                "query_counts": readback,
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

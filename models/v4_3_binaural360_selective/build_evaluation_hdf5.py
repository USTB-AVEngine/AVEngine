#!/usr/bin/env python3
"""Build a variable-size held-out HDF5 cache from a room audio batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from avengine_v43.hdf5_data import (
    EVALUATION_HDF5_SCHEMA,
    EvaluationQueryBank,
    OUTPUT_FRAME_COUNT,
    SAMPLE_COUNT,
    SOURCE_COUNT,
)
from avengine_v43.labels import LegacyV4AudioError, label_tracks_for_source
from run_training_smoke import (
    SOURCE_SLOTS,
    _load_json,
    _read_mixture,
    _trajectory_lookup,
)


INDEX_SCHEMA = "avengine_v43_room_evaluation_index_v1"
NATIVE_DELIVERY_SCHEMA = "avengine_room_evaluation_binaural_batch_v1"
NATIVE_SAMPLES_SCHEMA = "avengine_room_evaluation_binaural_samples_v1"
NATIVE_OUTPUT_CLOSURE_SCHEMA = (
    "avengine_room_evaluation_binaural_output_closure_v1"
)
NATIVE_INPUT_CLOSURE_SCHEMA = (
    "avengine_room_evaluation_binaural_input_closure_v1"
)
EXPECTED_OUTPUT_FILES = (
    "input_closure.json",
    "samples.json",
    "dry_audio_classes.json",
    "timing.json",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
BUILDER_PRODUCER_SCHEMA = "avengine_v43_evaluation_hdf5_builder_producer_v1"
BUILDER_CODE_FILES = (
    "build_evaluation_hdf5.py",
    "avengine_v43/hdf5_data.py",
    "avengine_v43/labels.py",
    "run_training_smoke.py",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-batch-root", type=Path, required=True)
    parser.add_argument("--trajectory-bank", type=Path, required=True)
    parser.add_argument("--rir-plan", type=Path, required=True)
    parser.add_argument("--room-id", required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    return parser.parse_args()


def _load_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to build the evaluation cache") from exc
    return h5py


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _builder_producer_identity(h5py: Any) -> Mapping[str, Any]:
    try:
        import soundfile
    except ImportError as exc:
        raise RuntimeError("soundfile is required to build the evaluation cache") from exc
    root = Path(__file__).resolve().parent
    code = {}
    for relative in BUILDER_CODE_FILES:
        path = (root / relative).resolve()
        if not path.is_file():
            raise LegacyV4AudioError(f"builder producer file is missing: {path}")
        code[relative] = {"sha256": _sha256_file(path)}
    return {
        "schema": BUILDER_PRODUCER_SCHEMA,
        "code": code,
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "h5py_version": str(h5py.__version__),
            "soundfile_version": str(soundfile.__version__),
            "libsndfile_version": str(soundfile.__libsndfile_version__),
        },
    }


def _contained_file(root: Path, declared: Any, *, owner: str) -> Path:
    if not isinstance(declared, str) or not declared:
        raise LegacyV4AudioError(f"{owner} path is invalid")
    relative = Path(declared)
    if relative.is_absolute():
        raise LegacyV4AudioError(f"{owner} path must be relative")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise LegacyV4AudioError(f"{owner} path escapes the audio batch") from exc
    if not candidate.is_file():
        raise LegacyV4AudioError(f"{owner} file is missing: {candidate}")
    return candidate


def _verify_hash(path: Path, expected: Any, *, owner: str) -> str:
    if not isinstance(expected, str) or SHA256_PATTERN.fullmatch(expected) is None:
        raise LegacyV4AudioError(f"{owner} SHA256 is invalid")
    actual = _sha256_file(path)
    if actual != expected:
        raise LegacyV4AudioError(f"{owner} SHA256 differs")
    return actual


def _verify_output_file_record(
    root: Path,
    relative_name: str,
    record: Any,
) -> Path:
    if (
        not isinstance(record, Mapping)
        or record.get("path") != relative_name
        or not isinstance(record.get("byte_size"), int)
        or record["byte_size"] < 1
    ):
        raise LegacyV4AudioError(
            f"Native output_closure record is invalid: {relative_name}"
        )
    path = _contained_file(root, record["path"], owner=relative_name)
    if path.stat().st_size != record["byte_size"]:
        raise LegacyV4AudioError(f"{relative_name} byte size differs")
    _verify_hash(path, record.get("sha256"), owner=relative_name)
    return path


def _ordered_samples(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = value.get("samples")
    if (
        value.get("schema") != NATIVE_SAMPLES_SCHEMA
        or value.get("status") != "pass"
        or not isinstance(raw, list)
        or not raw
    ):
        raise LegacyV4AudioError("room evaluation samples are invalid")
    result = []
    sample_ids: set[str] = set()
    episode_ids: set[str] = set()
    for sample in raw:
        sample_id = sample.get("sample_id") if isinstance(sample, Mapping) else None
        episode_id = sample.get("episode_id") if isinstance(sample, Mapping) else None
        classes = sample.get("source_classes") if isinstance(sample, Mapping) else None
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in sample_ids
            or not isinstance(episode_id, str)
            or not episode_id
            or episode_id in episode_ids
            or not isinstance(classes, Mapping)
            or set(classes) != set(SOURCE_SLOTS)
            or any(
                not isinstance(classes[slot], str) or not classes[slot]
                for slot in SOURCE_SLOTS
            )
            or classes["source1"] == classes["source2"]
            or sample.get("split") != "test"
            or sample.get("both_sources_active") is not True
            or sample.get("mixture_is_exact_persisted_source_stem_sum") is not True
        ):
            raise LegacyV4AudioError("room evaluation sample entry is invalid")
        result.append(sample)
        sample_ids.add(sample_id)
        episode_ids.add(episode_id)
    if value.get("sample_count") != len(result):
        raise LegacyV4AudioError("room evaluation sample_count differs")
    result.sort(key=lambda item: int(item.get("ordinal", -1)))
    if [item.get("ordinal") for item in result] != list(range(len(result))):
        raise LegacyV4AudioError("room evaluation sample ordinals are invalid")
    return result


def _validate_native_output(
    audio_root: Path,
    *,
    trajectory_path: Path,
    rir_path: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], dict[str, Path]]:
    delivery_path = audio_root / "delivery.json"
    if not delivery_path.is_file():
        raise LegacyV4AudioError("Native delivery.json is missing")
    delivery = _load_json(delivery_path)
    closure = delivery.get("output_closure")
    files = closure.get("files") if isinstance(closure, Mapping) else None
    if (
        delivery.get("schema") != NATIVE_DELIVERY_SCHEMA
        or delivery.get("status") != "pass"
        or delivery.get("research_only") is not True
        or delivery.get("qualification_claim") is not False
        or delivery.get("both_sources_active") is not True
        or delivery.get("mixture_is_exact_persisted_source_stem_sum") is not True
        or delivery.get("source_slots") != list(SOURCE_SLOTS)
        or not isinstance(closure, Mapping)
        or closure.get("schema") != NATIVE_OUTPUT_CLOSURE_SCHEMA
        or closure.get("status") != "pass"
        or not isinstance(files, Mapping)
        or set(files) != set(EXPECTED_OUTPUT_FILES)
    ):
        raise LegacyV4AudioError("Native delivery/output_closure is invalid")
    verified_files = {
        name: _verify_output_file_record(audio_root, name, files[name])
        for name in EXPECTED_OUTPUT_FILES
    }
    sample_document = _load_json(verified_files["samples.json"])
    samples = _ordered_samples(sample_document)
    input_closure = delivery.get("input_closure")
    if (
        not isinstance(input_closure, Mapping)
        or input_closure.get("schema") != NATIVE_INPUT_CLOSURE_SCHEMA
        or input_closure.get("status") != "pass"
        or closure.get("sample_count") != len(samples)
        or delivery.get("sample_count") != len(samples)
        or sample_document.get("input_closure") != input_closure
        or _load_json(verified_files["input_closure.json"]) != input_closure
    ):
        raise LegacyV4AudioError("Native input/sample closure differs")
    plan_files = input_closure.get("files")
    if not isinstance(plan_files, Mapping):
        raise LegacyV4AudioError("Native plan input closure is invalid")
    for name, path in (
        ("trajectory_bank.json", trajectory_path),
        ("rir_job_plan.json", rir_path),
    ):
        record = plan_files.get(name)
        if not isinstance(record, Mapping):
            raise LegacyV4AudioError(f"Native closure lacks {name}")
        _verify_hash(path, record.get("sha256"), owner=f"declared {name}")

    audio_paths: dict[str, Path] = {}
    artifact_paths: set[Path] = set()
    for sample in samples:
        sample_id = str(sample["sample_id"])
        mixture_path = _contained_file(
            audio_root, sample.get("audio_path"), owner=f"{sample_id} mixture"
        )
        mixture_sidecar = _contained_file(
            audio_root,
            sample.get("audio_sidecar_path"),
            owner=f"{sample_id} mixture sidecar",
        )
        _verify_hash(
            mixture_path, sample.get("audio_sha256"), owner=f"{sample_id} mixture"
        )
        _verify_hash(
            mixture_sidecar,
            sample.get("audio_sidecar_sha256"),
            owner=f"{sample_id} mixture sidecar",
        )
        stems = sample.get("source_stems")
        if not isinstance(stems, Mapping) or set(stems) != set(SOURCE_SLOTS):
            raise LegacyV4AudioError(f"{sample_id} source stem closure is invalid")
        sample_artifacts = {mixture_path, mixture_sidecar}
        for slot in SOURCE_SLOTS:
            record = stems[slot]
            if not isinstance(record, Mapping):
                raise LegacyV4AudioError(f"{sample_id} {slot} stem is invalid")
            stem_path = _contained_file(
                audio_root,
                record.get("audio_path"),
                owner=f"{sample_id} {slot} stem",
            )
            stem_sidecar = _contained_file(
                audio_root,
                record.get("audio_sidecar_path"),
                owner=f"{sample_id} {slot} stem sidecar",
            )
            _verify_hash(
                stem_path,
                record.get("audio_sha256"),
                owner=f"{sample_id} {slot} stem",
            )
            _verify_hash(
                stem_sidecar,
                record.get("sidecar_sha256"),
                owner=f"{sample_id} {slot} stem sidecar",
            )
            sample_artifacts.update((stem_path, stem_sidecar))
        if len(sample_artifacts) != 6 or artifact_paths.intersection(sample_artifacts):
            raise LegacyV4AudioError("Native audio artifact paths are duplicated")
        artifact_paths.update(sample_artifacts)
        audio_paths[sample_id] = mixture_path
    if (
        closure.get("wave_file_count") != len(samples) * 3
        or closure.get("wave_sidecar_count") != len(samples) * 3
        or closure.get("audio_artifact_file_count") != len(artifact_paths)
        or closure.get("audio_artifact_hashes_bound_by") != "samples.json"
    ):
        raise LegacyV4AudioError("Native output_closure audio artifact counts differ")

    identity = {
        "audio_batch_root": str(audio_root),
        "delivery": {
            "path": str(delivery_path.resolve()),
            "sha256": _sha256_file(delivery_path),
        },
        "samples": {
            "path": str(verified_files["samples.json"]),
            "sha256": str(files["samples.json"]["sha256"]),
        },
        "native_output_closure": dict(closure),
    }
    return identity, samples, audio_paths


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _validate_index(
    value: Mapping[str, Any],
    *,
    room_id: str,
    expected_samples: Sequence[Mapping[str, Any]],
    expected_input_identity: Mapping[str, Any],
) -> None:
    samples = value.get("samples")
    if (
        value.get("schema") != INDEX_SCHEMA
        or value.get("status") != "pass"
        or value.get("research_only") is not True
        or value.get("qualification_claim") is not False
        or value.get("room_id") != room_id
        or value.get("sample_count") != len(expected_samples)
        or value.get("split_sample_counts") != {"test": len(expected_samples)}
        or value.get("source_slots") != list(SOURCE_SLOTS)
        or value.get("input_identity") != expected_input_identity
        or not isinstance(samples, list)
        or [
            (item.get("sample_id"), item.get("episode_id"), item.get("split"))
            for item in samples
            if isinstance(item, Mapping)
        ]
        != [
            (item["sample_id"], item["episode_id"], "test")
            for item in expected_samples
        ]
    ):
        raise LegacyV4AudioError("evaluation dataset index readback is invalid")


def _publish_pair_no_replace(
    *,
    staged_h5: Path,
    output_h5: Path,
    staged_index: Path,
    output_index: Path,
) -> None:
    """Atomically publish each file without replacement and roll back this pair."""

    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        for staged, output in (
            (staged_h5, output_h5),
            (staged_index, output_index),
        ):
            os.link(staged, output, follow_symlinks=False)
            stat = output.stat(follow_symlinks=False)
            published.append((output, (stat.st_dev, stat.st_ino)))
    except Exception:
        for output, identity in reversed(published):
            try:
                stat = output.stat(follow_symlinks=False)
                if (stat.st_dev, stat.st_ino) == identity:
                    output.unlink()
            except FileNotFoundError:
                pass
        raise
    for staged in (staged_h5, staged_index):
        staged.unlink()


def main() -> int:
    args = _arguments()
    started = time.perf_counter()
    room_id = args.room_id.strip()
    if not room_id:
        raise LegacyV4AudioError("--room-id must be non-empty")
    audio_root = args.audio_batch_root.resolve()
    trajectory_path = args.trajectory_bank.resolve()
    rir_path = args.rir_plan.resolve()
    output_h5 = args.output_h5.resolve()
    output_index = args.output_index.resolve()
    if output_h5 == output_index:
        raise LegacyV4AudioError("HDF5 and index output paths must differ")
    for path in (audio_root, trajectory_path, rir_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if not audio_root.is_dir():
        raise NotADirectoryError(audio_root)
    for output in (output_h5, output_index):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to replace output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    native_identity, samples, audio_paths = _validate_native_output(
        audio_root,
        trajectory_path=trajectory_path,
        rir_path=rir_path,
    )
    trajectory = _load_json(trajectory_path)
    rir_plan = _load_json(rir_path)
    episodes = _trajectory_lookup(trajectory)
    frame_rate = trajectory.get("frame_rate_hz")
    listener_position = rir_plan.get("listener_position_m")
    listener_orientation = rir_plan.get("listener_orientation_wxyz")
    if (
        not isinstance(frame_rate, (int, float))
        or frame_rate <= 0
        or not isinstance(listener_position, list)
        or len(listener_position) != 3
        or not isinstance(listener_orientation, list)
        or len(listener_orientation) != 4
    ):
        raise LegacyV4AudioError("trajectory/RIR listener contract is invalid")
    captions = sorted(
        {
            str(sample["source_classes"][slot])
            for sample in samples
            for slot in SOURCE_SLOTS
        }
    )
    caption_to_id = {caption: index for index, caption in enumerate(captions)}
    if len(captions) > np.iinfo(np.uint8).max + 1:
        raise LegacyV4AudioError("too many sound classes for uint8 caption IDs")

    h5py = _load_h5py()
    builder_producer_identity = _builder_producer_identity(h5py)
    input_identity = {
        **native_identity,
        "dataset_index": str(output_index),
        "trajectory_bank": str(trajectory_path),
        "trajectory_bank_sha256": _sha256_file(trajectory_path),
        "rir_plan": str(rir_path),
        "rir_plan_sha256": _sha256_file(rir_path),
        "builder_producer_identity": builder_producer_identity,
    }
    h5_staging = output_h5.with_name(
        f".{output_h5.name}.incomplete.{os.getpid()}"
    )
    index_staging = output_index.with_name(
        f".{output_index.name}.staging.{os.getpid()}"
    )
    for staging in (h5_staging, index_staging):
        if staging.exists() or staging.is_symlink():
            raise FileExistsError(f"refusing to replace staging output: {staging}")

    index_samples: list[dict[str, Any]] = []
    count = len(samples)
    try:
        string_type = h5py.string_dtype(encoding="utf-8")
        with h5py.File(h5_staging, "x", libver="latest") as file:
            file.attrs["schema"] = EVALUATION_HDF5_SCHEMA
            file.attrs["completed_sample_count"] = 0
            file.attrs["room_id"] = room_id
            file.attrs["research_only"] = True
            file.attrs["qualification_claim"] = False
            file.attrs["split"] = "test"
            file.attrs["source_count"] = SOURCE_COUNT
            file.attrs["output_frame_count"] = OUTPUT_FRAME_COUNT
            file.attrs["caption_table_json"] = json.dumps(
                captions, ensure_ascii=False
            )
            file.attrs["input_identity_json"] = json.dumps(
                input_identity,
                ensure_ascii=False,
                sort_keys=True,
            )
            file.attrs["builder_producer_identity_json"] = json.dumps(
                builder_producer_identity,
                ensure_ascii=False,
                sort_keys=True,
            )
            file.attrs["sample_rate_hz"] = 16_000
            file.attrs["duration_seconds"] = 5.0
            file.create_dataset(
                "mixture",
                shape=(count, SAMPLE_COUNT, 2),
                dtype=np.float32,
                chunks=(1, SAMPLE_COUNT, 2),
            )
            file.create_dataset(
                "azimuth_deg",
                shape=(count, SOURCE_COUNT, OUTPUT_FRAME_COUNT),
                dtype=np.float32,
                chunks=(min(16, count), SOURCE_COUNT, OUTPUT_FRAME_COUNT),
            )
            file.create_dataset(
                "caption_id", shape=(count, SOURCE_COUNT), dtype=np.uint8
            )
            file.create_dataset("sample_id", shape=(count,), dtype=string_type)
            file.create_dataset("episode_id", shape=(count,), dtype=string_type)
            for sample_index, sample in enumerate(samples):
                sample_id = str(sample["sample_id"])
                episode_id = str(sample["episode_id"])
                episode = episodes.get(episode_id)
                if not isinstance(episode, Mapping):
                    raise LegacyV4AudioError(
                        f"unknown trajectory episode: {episode_id}"
                    )
                mixture = _read_mixture(audio_paths[sample_id])
                source_paths = episode.get("source_center_paths_m")
                if not isinstance(source_paths, Mapping):
                    raise LegacyV4AudioError(
                        f"{episode_id} lacks source-center paths"
                    )
                azimuth = np.empty(
                    (SOURCE_COUNT, OUTPUT_FRAME_COUNT), dtype=np.float32
                )
                caption_ids = np.empty(SOURCE_COUNT, dtype=np.uint8)
                for source_index, source_slot in enumerate(SOURCE_SLOTS):
                    positions = source_paths.get(source_slot)
                    sound_class = sample["source_classes"].get(source_slot)
                    if positions is None or not isinstance(sound_class, str):
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
                    azimuth[source_index] = np.mod(
                        np.asarray(
                            labels["native_360_azimuth_deg"],
                            dtype=np.float64,
                        ),
                        360.0,
                    ).astype(np.float32)
                    caption_ids[source_index] = caption_to_id[sound_class]
                file["mixture"][sample_index] = mixture
                file["azimuth_deg"][sample_index] = azimuth
                file["caption_id"][sample_index] = caption_ids
                file["sample_id"][sample_index] = sample_id
                file["episode_id"][sample_index] = episode_id
                file.attrs["completed_sample_count"] = sample_index + 1
                if (sample_index + 1) % 10 == 0 or sample_index + 1 == count:
                    file.flush()
                    print(
                        f"EVALUATION_H5_PROGRESS {sample_index + 1}/{count}",
                        flush=True,
                    )
                index_samples.append(
                    {
                        "sample_id": sample_id,
                        "episode_id": episode_id,
                        "split": "test",
                        "motion_case": episode["motion_case"],
                        "source_classes": dict(sample["source_classes"]),
                        "both_sources_active": True,
                        "audio_path": sample["audio_path"],
                    }
                )

        index = {
            "schema": INDEX_SCHEMA,
            "status": "pass",
            "research_only": True,
            "qualification_claim": False,
            "room_id": room_id,
            "sample_count": count,
            "split_sample_counts": {"test": count},
            "source_slots": list(SOURCE_SLOTS),
            "sound_classes_are_asset_independent": True,
            "roots": {"audio_batch_root": str(audio_root)},
            "input_identity": input_identity,
            "samples": index_samples,
            "timing_seconds": {
                "build_and_readback": time.perf_counter() - started
            },
        }
        _write_json_exclusive(index_staging, index)
        index_readback = _load_json(index_staging)
        _validate_index(
            index_readback,
            room_id=room_id,
            expected_samples=samples,
            expected_input_identity=input_identity,
        )
        with EvaluationQueryBank(h5_staging, preload_mixtures=True) as bank:
            identity = bank.identity()
            if (
                len(bank.queries("test")) != count * 2
                or bank.room_id != room_id
                or bank.input_identity != input_identity
                or list(bank.sample_ids)
                != [str(sample["sample_id"]) for sample in samples]
                or list(bank.episode_ids)
                != [str(sample["episode_id"]) for sample in samples]
                or identity["sample_count"] != count
            ):
                raise LegacyV4AudioError(
                    "evaluation HDF5/index full readback failed"
                )
        _publish_pair_no_replace(
            staged_h5=h5_staging,
            output_h5=output_h5,
            staged_index=index_staging,
            output_index=output_index,
        )
    except Exception:
        for staging in (h5_staging, index_staging):
            try:
                staging.unlink()
            except FileNotFoundError:
                pass
        raise

    print(
        f"EVALUATION_H5_OK h5={output_h5} index={output_index} samples={count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

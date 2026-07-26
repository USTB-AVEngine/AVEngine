#!/usr/bin/env python3
"""Evaluate a trained v4_3 checkpoint with slice and failure diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from avengine_v43.evaluation import (
    EVALUATION_REPORT_SCHEMA,
    EVALUATION_PRODUCER_CODE_FILES,
    MOTION_CASE_LABELS,
    azimuth_region,
    circular_error_deg,
    error_histogram,
    source_motion_label,
    summarize_errors,
    summarize_query_records,
    summarize_target_regions,
)
from avengine_v43.hdf5_data import (
    EVALUATION_HDF5_SCHEMA,
    HDF5_SCHEMA,
    QueryRef,
    open_query_bank,
)
from avengine_v43.labels import (
    LegacyV4AudioError,
    caption_for_asset,
    native_azimuth_to_bin360,
)
from avengine_v43.publication import atomic_publish_directory
from run_training_smoke import (
    SOURCE_SLOTS,
    _assert_output_contract,
    _load_model,
    _trajectory_lookup,
)
from train import _batch_tensors, _cache_text_embeddings
from visualize_inference import _load_inference_checkpoint


SCHEMA = EVALUATION_REPORT_SCHEMA
PRODUCER_SCHEMA = "avengine_v43_evaluation_producer_v1"
TRAINING_INDEX_SCHEMA = "avengine_m7_apartment_training_index_v1"
EVALUATION_INDEX_SCHEMA = "avengine_v43_room_evaluation_index_v1"
PRODUCER_CODE_FILES = tuple(sorted(EVALUATION_PRODUCER_CODE_FILES))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-h5", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clap-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--worst-query-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_260_723)
    parser.add_argument("--training-room-id")
    parser.add_argument("--evaluation-room-id")
    return parser.parse_args()


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise LegacyV4AudioError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _room_id(index: Mapping[str, Any], *, owner: str) -> str:
    room_id = index.get("room_id")
    if (
        index.get("status") != "pass"
        or not isinstance(room_id, str)
        or not room_id.strip()
    ):
        raise LegacyV4AudioError(f"{owner} room_id is invalid")
    return room_id


def _resolved_identity_path(value: Any, *, owner: str) -> Path:
    if not isinstance(value, str) or not value:
        raise LegacyV4AudioError(f"{owner} path is invalid")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _training_context(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    run_identity = checkpoint.get("run_identity")
    if not isinstance(run_identity, Mapping):
        raise LegacyV4AudioError("checkpoint run_identity is invalid")
    training_h5 = _resolved_identity_path(
        run_identity.get("dataset_h5"), owner="checkpoint training HDF5"
    )
    checkpoint_clap = _resolved_identity_path(
        run_identity.get("clap_checkpoint"), owner="checkpoint CLAP"
    )
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to resolve training identity") from exc
    with h5py.File(training_h5, "r") as file:
        schema = file.attrs.get("schema")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8")
        if str(schema) != HDF5_SCHEMA:
            raise LegacyV4AudioError("checkpoint training HDF5 schema is invalid")
        try:
            raw_identity = file.attrs["input_identity_json"]
            if isinstance(raw_identity, bytes):
                raw_identity = raw_identity.decode("utf-8")
            input_identity = json.loads(str(raw_identity))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LegacyV4AudioError(
                "checkpoint training HDF5 input identity is invalid"
            ) from exc
    if not isinstance(input_identity, Mapping):
        raise LegacyV4AudioError(
            "checkpoint training HDF5 input identity is invalid"
        )
    training_index_path = _resolved_identity_path(
        input_identity.get("dataset_index"),
        owner="checkpoint training dataset index",
    )
    training_index = _load_json(training_index_path)
    return {
        "training_h5": training_h5,
        "training_index": training_index_path,
        "training_room_id": _room_id(
            training_index, owner="checkpoint training dataset index"
        ),
        "checkpoint_clap": checkpoint_clap,
    }


def _validate_evaluation_index_header(index: Mapping[str, Any]) -> None:
    samples = index.get("samples")
    sample_count = index.get("sample_count")
    split_counts = index.get("split_sample_counts")
    if (
        index.get("schema") != EVALUATION_INDEX_SCHEMA
        or index.get("status") != "pass"
        or index.get("research_only") is not True
        or index.get("qualification_claim") is not False
        or not isinstance(samples, list)
        or not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 1
        or sample_count != len(samples)
        or not isinstance(split_counts, Mapping)
        or set(split_counts) != {"test"}
        or not isinstance(split_counts.get("test"), int)
        or isinstance(split_counts.get("test"), bool)
        or split_counts.get("test") != sample_count
        or index.get("source_slots") != list(SOURCE_SLOTS)
    ):
        raise LegacyV4AudioError("evaluation dataset index header is invalid")
    sample_ids: list[str] = []
    episode_ids: list[str] = []
    for sample in samples:
        if not isinstance(sample, Mapping) or sample.get("split") != "test":
            raise LegacyV4AudioError(
                "evaluation dataset index sample header is invalid"
            )
        sample_id = sample.get("sample_id")
        episode_id = sample.get("episode_id")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id != sample_id.strip()
            or not isinstance(episode_id, str)
            or not episode_id
            or episode_id != episode_id.strip()
        ):
            raise LegacyV4AudioError(
                "evaluation dataset index sample identity is invalid"
            )
        sample_ids.append(sample_id)
        episode_ids.append(episode_id)
    if len(set(sample_ids)) != sample_count or len(set(episode_ids)) != sample_count:
        raise LegacyV4AudioError(
            "evaluation dataset index sample identities are duplicated"
        )


def _resolve_evaluation_context(
    *,
    evaluation_h5: Path,
    evaluation_index_path: Path,
    evaluation_index: Mapping[str, Any],
    bank_identity: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    clap_checkpoint: Path,
    declared_training_room_id: str | None,
    declared_evaluation_room_id: str | None,
) -> dict[str, Any]:
    """Derive room/regime from bound artifacts, treating CLI IDs as assertions."""

    evaluation_room_id = _room_id(
        evaluation_index, owner="evaluation dataset index"
    )
    if (
        declared_evaluation_room_id is not None
        and declared_evaluation_room_id != evaluation_room_id
    ):
        raise LegacyV4AudioError(
            "--evaluation-room-id differs from dataset_index.room_id"
        )
    training = _training_context(checkpoint)
    training_room_id = str(training["training_room_id"])
    if (
        declared_training_room_id is not None
        and declared_training_room_id != training_room_id
    ):
        raise LegacyV4AudioError(
            "--training-room-id differs from checkpoint training room"
        )
    if training["checkpoint_clap"] != clap_checkpoint:
        raise LegacyV4AudioError(
            "--clap-checkpoint differs from checkpoint run_identity"
        )

    schema = bank_identity.get("schema")
    input_identity = bank_identity.get("input_identity")
    if not isinstance(input_identity, Mapping):
        raise LegacyV4AudioError("evaluation HDF5 input identity is invalid")
    bound_index = _resolved_identity_path(
        input_identity.get("dataset_index"),
        owner="evaluation HDF5 dataset index",
    )
    if bound_index != evaluation_index_path:
        raise LegacyV4AudioError(
            "evaluation HDF5 and dataset index identity differ"
        )
    declared_index_sha256 = input_identity.get("dataset_index_sha256")
    if declared_index_sha256 is not None and (
        not _valid_sha256(declared_index_sha256)
        or _sha256_file(evaluation_index_path) != declared_index_sha256
    ):
        raise LegacyV4AudioError(
            "evaluation HDF5 dataset-index SHA256 differs"
        )
    same_room = evaluation_room_id == training_room_id
    if same_room:
        if (
            schema != HDF5_SCHEMA
            or evaluation_index.get("schema") != TRAINING_INDEX_SCHEMA
            or evaluation_h5 != training["training_h5"]
            or evaluation_index_path != training["training_index"]
        ):
            raise LegacyV4AudioError(
                "same-room held-out evaluation must use the checkpoint training "
                "HDF5 and its bound dataset index"
            )
        declared_index_identity = evaluation_index.get("input_identity")
        if (
            declared_index_identity is not None
            and declared_index_identity != input_identity
        ):
            raise LegacyV4AudioError(
                "legacy training index input_identity differs from HDF5"
            )
        regime = "same_room_held_out"
        identity_mode = "legacy_training_identity_compatibility"
        index_identity_exact_match = False
    else:
        _validate_evaluation_index_header(evaluation_index)
        if (
            schema != EVALUATION_HDF5_SCHEMA
            or evaluation_h5 == training["training_h5"]
            or evaluation_index_path == training["training_index"]
            or bank_identity.get("room_id") != evaluation_room_id
            or evaluation_index.get("input_identity") != input_identity
        ):
            raise LegacyV4AudioError(
                "cross-room zero-shot evaluation must use a distinct evaluation "
                "HDF5 whose room_id and input_identity match its index"
            )
        regime = "cross_room_zero_shot"
        identity_mode = "evaluation_hdf5_exact_index_identity"
        index_identity_exact_match = True
    return {
        **training,
        "evaluation_room_id": evaluation_room_id,
        "evaluation_regime": regime,
        "evaluation_input_identity": dict(input_identity),
        "evaluation_dataset_identity_mode": identity_mode,
        "dataset_index_identity_exact_match": index_identity_exact_match,
    }


def _producer_identity(torch: Any) -> Mapping[str, Any]:
    root = Path(__file__).resolve().parent
    code = {}
    for relative in PRODUCER_CODE_FILES:
        path = (root / relative).resolve()
        if not path.is_file():
            raise LegacyV4AudioError(f"evaluation producer file is missing: {path}")
        code[relative] = {"sha256": _sha256_file(path)}
    return {
        "schema": PRODUCER_SCHEMA,
        "code": code,
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "torch_cuda_runtime_version": str(torch.version.cuda),
        },
    }


def _sample_map(index: Mapping[str, Any], *, split: str) -> dict[str, Mapping[str, Any]]:
    samples = index.get("samples")
    if index.get("status") != "pass" or not isinstance(samples, list):
        raise LegacyV4AudioError("dataset index is not a passing sample index")
    result: dict[str, Mapping[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping) or sample.get("split") != split:
            continue
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise LegacyV4AudioError("dataset index has invalid or duplicate sample IDs")
        result[sample_id] = sample
    if not result:
        raise LegacyV4AudioError(f"dataset index has no {split} samples")
    return result


def _validated_split_sample_episode_ids(
    *,
    samples: Mapping[str, Mapping[str, Any]],
    queries: Sequence[QueryRef],
    bank_sample_ids: Sequence[str],
    bank_episode_ids: Sequence[str],
    bank_schema: Any,
) -> dict[str, str]:
    """Validate the selected split sequence without comparing it to all HDF5 rows."""

    index_sequence: tuple[tuple[str, str], ...] = tuple(
        (sample_id, sample.get("episode_id"))
        for sample_id, sample in samples.items()
    )
    if any(
        not isinstance(episode_id, str)
        or not episode_id
        or episode_id != episode_id.strip()
        for _, episode_id in index_sequence
    ):
        raise LegacyV4AudioError(
            "dataset-index split sample/episode sequence is invalid"
        )
    expected_query_sequence = tuple(
        (sample_id, source_index)
        for sample_id in samples
        for source_index in range(2)
    )
    actual_query_sequence = tuple(
        (query.sample_id, query.source_index) for query in queries
    )
    if actual_query_sequence != expected_query_sequence:
        raise LegacyV4AudioError(
            "HDF5 query sample/source order differs from dataset index"
        )
    source_zero_queries = [query for query in queries if query.source_index == 0]
    selected_indices = tuple(query.sample_index for query in source_zero_queries)
    if (
        len(bank_sample_ids) < 1
        or len(selected_indices) != len(index_sequence)
        or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(bank_sample_ids)
            for index in selected_indices
        )
        or len(set(selected_indices)) != len(selected_indices)
    ):
        raise LegacyV4AudioError("HDF5 split sample indices are invalid")
    hdf5_sample_sequence = tuple(
        str(bank_sample_ids[index]) for index in selected_indices
    )
    if hdf5_sample_sequence != tuple(samples):
        raise LegacyV4AudioError(
            "HDF5 split sample order differs from dataset index"
        )
    for offset, (sample_id, _) in enumerate(index_sequence):
        source_queries = queries[offset * 2 : offset * 2 + 2]
        if (
            source_queries[0].sample_index != source_queries[1].sample_index
            or str(bank_sample_ids[source_queries[0].sample_index]) != sample_id
            or str(bank_sample_ids[source_queries[1].sample_index]) != sample_id
        ):
            raise LegacyV4AudioError(
                "HDF5 query sample indices differ from dataset index"
            )
    if len(bank_episode_ids) == len(bank_sample_ids):
        hdf5_sequence = tuple(
            (
                str(bank_sample_ids[index]),
                str(bank_episode_ids[index]),
            )
            for index in selected_indices
        )
        if hdf5_sequence != index_sequence:
            raise LegacyV4AudioError(
                "HDF5 split sample/episode order differs from dataset index"
            )
    elif bank_schema == EVALUATION_HDF5_SCHEMA:
        raise LegacyV4AudioError(
            "evaluation HDF5 lacks the sample/episode sequence"
        )
    return dict(index_sequence)


def _validated_sample_metadata(
    *,
    samples: Mapping[str, Mapping[str, Any]],
    queries: Sequence[QueryRef],
    sample_episode_ids: Mapping[str, str],
    bank_identity: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Cross-check index slice fields against HDF5 queries and bound trajectory."""

    schema = bank_identity.get("schema")
    input_identity = bank_identity.get("input_identity")
    if not isinstance(input_identity, Mapping):
        raise LegacyV4AudioError("HDF5 input identity is invalid")
    trajectory_path = _resolved_identity_path(
        input_identity.get("trajectory_bank"),
        owner="HDF5 trajectory bank",
    )
    trajectory_sha256 = _sha256_file(trajectory_path)
    declared_trajectory_sha256 = input_identity.get("trajectory_bank_sha256")
    if schema == EVALUATION_HDF5_SCHEMA and not _valid_sha256(
        declared_trajectory_sha256
    ):
        raise LegacyV4AudioError(
            "evaluation HDF5 lacks trajectory_bank_sha256"
        )
    if declared_trajectory_sha256 is not None and (
        not _valid_sha256(declared_trajectory_sha256)
        or trajectory_sha256 != declared_trajectory_sha256
    ):
        raise LegacyV4AudioError("HDF5 trajectory-bank SHA256 differs")
    trajectory = _load_json(trajectory_path)
    if trajectory.get("schema") != "avengine_room_trajectory_bank_v2":
        raise LegacyV4AudioError("HDF5 trajectory-bank schema is invalid")
    episodes = _trajectory_lookup(trajectory)

    queries_by_sample: dict[str, dict[int, QueryRef]] = {}
    for query in queries:
        if (
            query.sample_id not in samples
            or query.source_index not in (0, 1)
            or not isinstance(query.caption, str)
            or not query.caption
        ):
            raise LegacyV4AudioError("HDF5 query binding is invalid")
        by_source = queries_by_sample.setdefault(query.sample_id, {})
        if query.source_index in by_source:
            raise LegacyV4AudioError("HDF5 query source slot is duplicated")
        by_source[query.source_index] = query

    metadata: dict[str, dict[str, Any]] = {}
    for sample_id, sample in samples.items():
        episode_id = sample_episode_ids.get(sample_id)
        episode = episodes.get(episode_id)
        motion_case = (
            episode.get("motion_case") if isinstance(episode, Mapping) else None
        )
        by_source = queries_by_sample.get(sample_id)
        if (
            not isinstance(episode_id, str)
            or not episode_id
            or not isinstance(motion_case, str)
            or motion_case not in MOTION_CASE_LABELS
            or sample.get("episode_id") != episode_id
            or sample.get("motion_case") != motion_case
            or sample.get("both_sources_active") is not True
            or not isinstance(by_source, Mapping)
            or set(by_source) != {0, 1}
        ):
            raise LegacyV4AudioError(
                f"{sample_id} index/HDF5/trajectory binding is invalid"
            )
        captions = tuple(by_source[index].caption for index in range(2))
        if len(set(captions)) != 2:
            raise LegacyV4AudioError(f"{sample_id} HDF5 captions are ambiguous")
        if schema == EVALUATION_HDF5_SCHEMA:
            source_values = sample.get("source_classes")
            if not isinstance(source_values, Mapping) or set(
                source_values
            ) != set(SOURCE_SLOTS):
                raise LegacyV4AudioError(
                    f"{sample_id} source_classes are invalid"
                )
            expected_captions = tuple(
                source_values[slot] for slot in SOURCE_SLOTS
            )
        elif schema == HDF5_SCHEMA:
            source_values = sample.get("asset_ids_by_source_slot")
            if not isinstance(source_values, Mapping) or set(
                source_values
            ) != set(SOURCE_SLOTS):
                raise LegacyV4AudioError(
                    f"{sample_id} source asset bindings are invalid"
                )
            expected_captions = tuple(
                caption_for_asset(str(source_values[slot]))
                for slot in SOURCE_SLOTS
            )
        else:
            raise LegacyV4AudioError("unsupported HDF5 metadata schema")
        if (
            any(not isinstance(value, str) or not value for value in expected_captions)
            or expected_captions != captions
        ):
            raise LegacyV4AudioError(
                f"{sample_id} index source bindings differ from HDF5 captions"
            )
        metadata[sample_id] = {
            "episode_id": episode_id,
            "motion_case": motion_case,
            "captions_by_source_slot": dict(zip(SOURCE_SLOTS, captions)),
            "ordered_source_pair": "|".join(captions),
        }
    if set(queries_by_sample) != set(metadata):
        raise LegacyV4AudioError("HDF5 query sample set differs")
    return metadata, {
        "trajectory_bank": str(trajectory_path),
        "trajectory_bank_sha256": trajectory_sha256,
    }


def _source_video(index: Mapping[str, Any], sample: Mapping[str, Any]) -> Path | None:
    roots = index.get("roots")
    root = roots.get("ue_render_root") if isinstance(roots, Mapping) else None
    episode_id = sample.get("episode_id")
    if not isinstance(root, str) or not isinstance(episode_id, str):
        return None
    preferred = Path(root) / episode_id / "ue_topdown_binaural.mp4"
    if preferred.is_file():
        return preferred.resolve()
    declared = sample.get("topdown_episode_path")
    fallback = Path(root) / declared if isinstance(declared, str) else None
    return fallback.resolve() if fallback is not None and fallback.is_file() else None


def _infer(
    *,
    torch: Any,
    model: Any,
    device: Any,
    bank: Any,
    queries: Sequence[QueryRef],
    text_cache: Mapping[str, Any],
    batch_size: int,
) -> tuple[list[dict[str, Any]], float]:
    if batch_size < 1:
        raise LegacyV4AudioError("batch size must be positive")
    records: list[dict[str, Any]] = []
    model.eval()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(queries), batch_size):
            batch = queries[start : start + batch_size]
            tensors = _batch_tensors(
                torch=torch,
                device=device,
                bank=bank,
                batch=batch,
                text_cache=text_cache,
                with_targets=False,
            )
            separated, doa, cardinality = model(
                tensors["mixtures"], tensors["null_cues"], tensors["text_embeddings"]
            )
            _assert_output_contract(
                torch=torch,
                separated=separated,
                doa=doa,
                cardinality=cardinality,
                batch_size=len(batch),
            )
            predictions = torch.argmax(doa, dim=-1).cpu().numpy().astype(np.float64)
            cardinality_predictions = (
                torch.argmax(cardinality, dim=-1).cpu().numpy().astype(np.int64)
            )
            for index, query in enumerate(batch):
                targets = native_azimuth_to_bin360(query.azimuth_deg)
                errors = circular_error_deg(predictions[index], targets)
                records.append(
                    {
                        "sample_id": query.sample_id,
                        "source_index": query.source_index,
                        "source_slot": f"source{query.source_index + 1}",
                        "caption": query.caption,
                        "targets_deg": targets,
                        "predictions_deg": predictions[index],
                        "errors_deg": errors,
                        "cardinality_accuracy": float(
                            np.mean(cardinality_predictions[index] == 1)
                        ),
                    }
                )
    torch.cuda.synchronize(device)
    return records, time.perf_counter() - started


def _group_table(rows: Sequence[Mapping[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(str(row['label']))}</td>"
            f"<td>{int(row.get('sample_count', 0))}</td>"
            f"<td>{int(row['frame_count'])}</td>"
            f"<td>{float(row['mean_absolute_error_deg']):.2f}</td>"
            f"<td>{float(row['median_absolute_error_deg']):.2f}</td>"
            f"<td>{float(row['p90_absolute_error_deg']):.2f}</td>"
            f"<td>{100.0 * float(row['error_over_45deg_rate']):.2f}%</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Slice</th><th>Samples</th><th>Frames</th>"
        "<th>Mean °</th><th>Median °</th><th>P90 °</th><th>&gt;45°</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def _histogram_svg(histogram: Mapping[str, Sequence[Any]]) -> str:
    counts = np.asarray(histogram["counts"], dtype=np.float64)
    width, height = 900, 320
    left, right, top, bottom = 55, 20, 25, 45
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max(float(np.max(counts)), 1.0)
    bar_w = plot_w / len(counts)
    bars = []
    for index, count in enumerate(counts):
        bar_h = plot_h * float(count) / maximum
        x = left + index * bar_w
        y = top + plot_h - bar_h
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_w - 1, 1):.2f}" '
            f'height="{bar_h:.2f}" fill="#2878b5"/>'
        )
    ticks = []
    for angle in (0, 45, 90, 135, 180):
        x = left + plot_w * angle / 180.0
        ticks.append(
            f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" '
            f'y2="{top + plot_h + 5}" stroke="#333"/>'
            f'<text x="{x:.1f}" y="{height - 15}" text-anchor="middle" '
            f'font-size="12">{angle}°</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="absolute circular error histogram">'
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#333"/>{"".join(bars)}{"".join(ticks)}'
        '<text x="450" y="16" text-anchor="middle" font-size="14">'
        'Test-frame absolute circular error distribution</text></svg>'
    )


def _write_html(
    *, path: Path, report: Mapping[str, Any], worst: Sequence[Mapping[str, Any]]
) -> None:
    overall = report["overall"]
    regime = str(report["evaluation_regime"])
    title = (
        "v4.3 cross-room zero-shot evaluation"
        if regime == "cross_room_zero_shot"
        else "v4.3 same-room held-out evaluation"
    )
    worst_rows = []
    for row in worst:
        video = row.get("source_video")
        label = escape(str(row["sample_id"]))
        sample_cell = f'<a href="file://{escape(str(video))}">{label}</a>' if video else label
        worst_rows.append(
            "<tr>"
            f"<td>{sample_cell}</td><td>{escape(str(row['source_slot']))}</td>"
            f"<td>{escape(str(row['caption']))}</td>"
            f"<td>{float(row['mean_absolute_error_deg']):.2f}</td>"
            f"<td>{float(row['p90_absolute_error_deg']):.2f}</td>"
            f"<td>{float(row['maximum_absolute_error_deg']):.2f}</td></tr>"
        )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#20242a;background:#f7f8fa}}
.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{background:white;border:1px solid #ddd;
border-radius:9px;padding:12px 16px;min-width:145px}}.value{{font-size:25px;font-weight:700}}
table{{border-collapse:collapse;width:100%;background:white;margin:10px 0 24px}}
th,td{{border:1px solid #ddd;padding:7px;text-align:left;font-size:13px}}th{{background:#eef2f6}}
h2{{margin-top:30px}}svg{{background:white;border:1px solid #ddd;border-radius:9px;max-width:100%}}
</style></head><body>
<h1>{escape(title)}</h1>
<p>Training room: {escape(str(report['training_room_id']))}; evaluation room:
{escape(str(report['evaluation_room_id']))}. Research-only diagnostic.
RGB/Topdown are review media, not model inputs.</p>
<div class="cards">
<div class="card"><div>Samples</div><div class="value">{int(report['sample_count'])}</div></div>
<div class="card"><div>Queries</div><div class="value">{int(report['query_count'])}</div></div>
<div class="card"><div>Mean MAE</div><div class="value">{float(overall['mean_absolute_error_deg']):.2f}°</div></div>
<div class="card"><div>Median MAE</div><div class="value">{float(overall['median_absolute_error_deg']):.2f}°</div></div>
<div class="card"><div>P90 MAE</div><div class="value">{float(overall['p90_absolute_error_deg']):.2f}°</div></div>
<div class="card"><div>Region-macro MAE</div><div class="value">{float(overall['uniform_target_region_macro_mean_absolute_error_deg']):.2f}°</div></div>
<div class="card"><div>Frames &gt;45°</div><div class="value">{100*float(overall['error_over_45deg_rate']):.2f}%</div></div>
<div class="card"><div>Frames &gt;90°</div><div class="value">{100*float(overall['error_over_90deg_rate']):.2f}%</div></div>
</div>
<h2>Error distribution</h2>{_histogram_svg(report['error_histogram'])}
<h2>Motion case</h2>{_group_table(report['slices']['motion_case'])}
<h2>Source motion</h2>{_group_table(report['slices']['source_motion'])}
<h2>Sound/query class</h2>{_group_table(report['slices']['caption'])}
<h2>Target azimuth region</h2>{_group_table(report['slices']['target_region'])}
<h2>Ordered source pair</h2>{_group_table(report['slices']['ordered_source_pair'])}
<h2>Worst target queries</h2>
<table><thead><tr><th>Sample/video</th><th>Slot</th><th>Query</th><th>Mean °</th>
<th>P90 °</th><th>Max °</th></tr></thead><tbody>{''.join(worst_rows)}</tbody></table>
<p>Machine-readable evidence: <a href="evaluation.json">evaluation.json</a>;
per-query table: <a href="per_query_metrics.csv">per_query_metrics.csv</a>.</p>
</body></html>"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    args = _arguments()
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output_root}")
    paths = [
        args.dataset_h5.resolve(), args.dataset_index.resolve(),
        args.checkpoint.resolve(), args.clap_checkpoint.resolve(),
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset_index = _load_json(paths[1])
    samples = _sample_map(dataset_index, split=args.split)

    with open_query_bank(paths[0], preload_mixtures=True) as bank:
        queries = list(bank.queries(args.split))
        bank_identity = bank.identity()
        sample_episode_ids = _validated_split_sample_episode_ids(
            samples=samples,
            queries=queries,
            bank_sample_ids=bank.sample_ids,
            bank_episode_ids=getattr(bank, "episode_ids", ()),
            bank_schema=bank_identity.get("schema"),
        )
        torch, model, device, model_load_seconds, model_audit = _load_model(
            clap_checkpoint=paths[3], device_name=args.device, seed=args.seed
        )
        checkpoint = _load_inference_checkpoint(torch=torch, model=model, path=paths[2])
        evaluation_context = _resolve_evaluation_context(
            evaluation_h5=paths[0],
            evaluation_index_path=paths[1],
            evaluation_index=dataset_index,
            bank_identity=bank_identity,
            checkpoint=checkpoint,
            clap_checkpoint=paths[3],
            declared_training_room_id=args.training_room_id,
            declared_evaluation_room_id=args.evaluation_room_id,
        )
        trusted_sample_metadata, trajectory_identity = _validated_sample_metadata(
            samples=samples,
            queries=queries,
            sample_episode_ids=sample_episode_ids,
            bank_identity=bank_identity,
        )
        text_cache = _cache_text_embeddings(
            torch=torch, model=model, device=device, queries_by_split={args.split: queries}
        )
        records, inference_seconds = _infer(
            torch=torch, model=model, device=device, bank=bank, queries=queries,
            text_cache=text_cache, batch_size=args.batch_size,
        )
    training_room_id = str(evaluation_context["training_room_id"])
    evaluation_room_id = str(evaluation_context["evaluation_room_id"])
    evaluation_regime = str(evaluation_context["evaluation_regime"])
    identity_paths = {
        "dataset_h5": paths[0],
        "dataset_index": paths[1],
        "training_dataset_h5": Path(evaluation_context["training_h5"]),
        "training_dataset_index": Path(evaluation_context["training_index"]),
        "checkpoint": paths[2],
        "clap_checkpoint": paths[3],
    }
    hashes_by_path = {
        path: _sha256_file(path) for path in set(identity_paths.values())
    }
    producer = _producer_identity(torch)

    query_rows: list[dict[str, Any]] = []
    pair_predictions: dict[str, list[np.ndarray]] = {}
    for record in records:
        sample = samples[str(record["sample_id"])]
        trusted = trusted_sample_metadata[str(record["sample_id"])]
        motion_case = str(trusted["motion_case"])
        record["motion_case"] = motion_case
        record["source_motion"] = source_motion_label(
            motion_case, int(record["source_index"])
        )
        record["ordered_source_pair"] = trusted["ordered_source_pair"]
        metrics = summarize_errors(record["errors_deg"])
        target_regions = azimuth_region(record["targets_deg"])
        counts = {
            region: int(np.sum(target_regions == region))
            for region in set(target_regions.tolist())
        }
        row = {
            "sample_id": record["sample_id"],
            "episode_id": trusted["episode_id"],
            "source_slot": record["source_slot"],
            "caption": record["caption"],
            "motion_case": motion_case,
            "source_motion": record["source_motion"],
            "ordered_source_pair": record["ordered_source_pair"],
            "dominant_target_region": max(counts, key=counts.get),
            "cardinality_accuracy": record["cardinality_accuracy"],
            "source_video": str(_source_video(dataset_index, sample) or ""),
            **metrics,
        }
        query_rows.append(row)
        pair_predictions.setdefault(str(record["sample_id"]), []).append(
            np.asarray(record["predictions_deg"])
        )

    all_errors = np.concatenate([record["errors_deg"] for record in records])
    pair_difference_rates = []
    for sample_id, values in pair_predictions.items():
        if len(values) != 2:
            raise LegacyV4AudioError(f"{sample_id} does not contain exactly two queries")
        pair_difference_rates.append(float(np.mean(values[0] != values[1])))
    worst = sorted(
        query_rows,
        key=lambda value: (
            float(value["mean_absolute_error_deg"]),
            float(value["p90_absolute_error_deg"]),
        ),
        reverse=True,
    )[: args.worst_query_count]
    target_region_rows = summarize_target_regions(records)
    if {str(row["label"]) for row in target_region_rows} != {
        "front", "right", "rear", "left"
    }:
        raise LegacyV4AudioError("evaluation split does not cover all azimuth regions")
    target_region_macro = float(
        np.mean(
            [float(row["mean_absolute_error_deg"]) for row in target_region_rows]
        )
    )
    target_region_total = sum(int(row["frame_count"]) for row in target_region_rows)
    target_region_fractions = {
        str(row["label"]): int(row["frame_count"]) / target_region_total
        for row in target_region_rows
    }
    report = {
        "schema": SCHEMA,
        "status": "pass",
        "research_only": True,
        "qualification_claim": False,
        "claim_boundary": (
            "one fixed Apartment-trained checkpoint evaluated zero-shot on one "
            "fixed research-only room set; not a broad-generalization or physical-"
            "acoustics claim"
            if evaluation_regime == "cross_room_zero_shot"
            else "held-out diagnostics for one fixed Apartment checkpoint and "
            "index; not a cross-room or broad-generalization claim"
        ),
        "training_room_id": training_room_id,
        "evaluation_room_id": evaluation_room_id,
        "evaluation_regime": evaluation_regime,
        "inputs": {
            "dataset_h5": str(paths[0]),
            "dataset_h5_sha256": hashes_by_path[identity_paths["dataset_h5"]],
            "dataset_index": str(paths[1]),
            "dataset_index_sha256": hashes_by_path[
                identity_paths["dataset_index"]
            ],
            "evaluation_trajectory_bank": trajectory_identity[
                "trajectory_bank"
            ],
            "evaluation_trajectory_bank_sha256": trajectory_identity[
                "trajectory_bank_sha256"
            ],
            "training_dataset_h5": str(evaluation_context["training_h5"]),
            "training_dataset_h5_sha256": hashes_by_path[
                identity_paths["training_dataset_h5"]
            ],
            "training_dataset_index": str(evaluation_context["training_index"]),
            "training_dataset_index_sha256": hashes_by_path[
                identity_paths["training_dataset_index"]
            ],
            "checkpoint": str(paths[2]),
            "checkpoint_sha256": hashes_by_path[identity_paths["checkpoint"]],
            "clap_checkpoint": str(paths[3]),
            "clap_checkpoint_sha256": hashes_by_path[
                identity_paths["clap_checkpoint"]
            ],
            "checkpoint_best_epoch_zero_based": checkpoint.get("best_epoch"),
            "checkpoint_best_validation_mae_deg": checkpoint.get(
                "best_validation_mae_deg"
            ),
        },
        "split": args.split,
        "sample_count": len(samples),
        "query_count": len(records),
        "overall": {
            **summarize_errors(all_errors),
            "uniform_target_region_macro_mean_absolute_error_deg": target_region_macro,
            "target_region_frame_fractions": target_region_fractions,
            "mean_query_prediction_difference_rate": float(
                np.mean(pair_difference_rates)
            ),
            "mean_cardinality_accuracy": float(
                np.mean([record["cardinality_accuracy"] for record in records])
            ),
        },
        "error_histogram": error_histogram(records),
        "slices": {
            "motion_case": summarize_query_records(records, group_key="motion_case"),
            "source_motion": summarize_query_records(records, group_key="source_motion"),
            "caption": summarize_query_records(records, group_key="caption"),
            "ordered_source_pair": summarize_query_records(
                records, group_key="ordered_source_pair"
            ),
            "target_region": target_region_rows,
        },
        "worst_queries": worst,
        "provenance": {
            "schema": "avengine_v43_evaluation_provenance_v1",
            "evaluation_dataset_identity_mode": evaluation_context[
                "evaluation_dataset_identity_mode"
            ],
            "dataset_index_identity_exact_match": evaluation_context[
                "dataset_index_identity_exact_match"
            ],
            "legacy_training_identity_compatibility": (
                evaluation_context["evaluation_dataset_identity_mode"]
                == "legacy_training_identity_compatibility"
            ),
            "source_caption_authority": (
                "hdf5_query_bank_cross_checked_with_dataset_index"
            ),
            "motion_case_authority": (
                "hdf5_input_identity.trajectory_bank_cross_checked_with_"
                "dataset_index"
            ),
            **trajectory_identity,
        },
        "producer": producer,
        "model": dict(model_audit),
        "timing_seconds": {
            "model_load": model_load_seconds,
            "split_inference": inference_seconds,
        },
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"refusing to replace staging output: {staging}")
    try:
        staging.mkdir()
        _atomic_json(staging / "evaluation.json", report)
        with (staging / "per_query_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(query_rows[0]))
            writer.writeheader()
            writer.writerows(query_rows)
        _write_html(path=staging / "REVIEW_INDEX.html", report=report, worst=worst)
        atomic_publish_directory(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

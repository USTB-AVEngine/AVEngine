#!/usr/bin/env python3
"""Build a research-only Spatial-Omni QA dataset from native RLR FOA.

Every point is read only from ``audio/foa/mixture.wav`` and its authenticated
sidecar. Binaural and pseudo-FOA/foaish fallbacks are deliberately absent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from avengine.m4.audio import AudioContractError, read_float32_wav
from avengine.m4.spatial import rlr_foa_contract
from avengine.security.path_policy import WorkspacePathPolicy, atomic_publish_directory


EXPECTED_SAMPLE_RATE_HZ = 16_000
EXPECTED_SAMPLE_COUNT = 80_000
EXPECTED_CHANNEL_COUNT = 4
NATIVE_FOA_RELATIVE_PATH = Path("audio/foa/mixture.wav")


class SpatialOmniDatasetError(ValueError):
    """Inputs cannot produce the requested native-FOA QA dataset."""


@dataclass(frozen=True)
class NativeFoaPoint:
    point_id: str
    point_root: Path
    wav_path: Path
    sidecar_path: Path


def _load_json(path: str | Path, *, owner: str) -> Any:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        with resolved.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpatialOmniDatasetError(f"cannot load {owner}: {resolved}: {exc}") from exc


def _nonempty_string(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpatialOmniDatasetError(f"{owner} must be a non-empty string")
    return value


def _finite_number(value: Any, *, owner: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpatialOmniDatasetError(f"{owner} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SpatialOmniDatasetError(f"{owner} must be a finite number")
    return result


def validate_native_foa_point(point_root: str | Path, *, point_id: str) -> NativeFoaPoint:
    """Read and validate one exact native FOA mixture and existing sidecar."""

    root = Path(point_root).expanduser().resolve(strict=True)
    wav_path = root / NATIVE_FOA_RELATIVE_PATH
    if not wav_path.is_file():
        raise SpatialOmniDatasetError(
            f"point {point_id} lacks audio/foa/mixture.wav; binaural and foaish "
            "fallbacks are forbidden"
        )
    sidecar_path = wav_path.with_suffix(wav_path.suffix + ".json")
    try:
        decoded = read_float32_wav(wav_path, verify_sidecar=True)
    except AudioContractError as exc:
        raise SpatialOmniDatasetError(
            f"point {point_id} native FOA WAVE/sidecar verification failed: {exc}"
        ) from exc
    if decoded.sample_rate_hz != EXPECTED_SAMPLE_RATE_HZ:
        raise SpatialOmniDatasetError(
            f"point {point_id} native FOA must be 16000 Hz"
        )
    if decoded.samples.shape != (EXPECTED_CHANNEL_COUNT, EXPECTED_SAMPLE_COUNT):
        raise SpatialOmniDatasetError(
            f"point {point_id} native FOA must have shape [4,80000], got "
            f"{list(decoded.samples.shape)}"
        )
    sidecar = decoded.sidecar
    if not isinstance(sidecar, Mapping):
        raise SpatialOmniDatasetError(f"point {point_id} native FOA sidecar is absent")
    metadata = sidecar.get("metadata")
    if not isinstance(metadata, Mapping):
        raise SpatialOmniDatasetError(
            f"point {point_id} native FOA metadata must be an object"
        )
    expected_spatial_format = rlr_foa_contract()
    if metadata.get("spatial_format") != expected_spatial_format:
        raise SpatialOmniDatasetError(
            f"point {point_id} is not native RLR FOA "
            "rlr_foa_acn_n3d_world_v1 [W,Y,Z,X] ACN/N3D world"
        )
    expected_metadata = {
        "audio_role": "dynamic_wet_mixture",
        "hrtf_used": False,
        "resampling": "not_applied",
        "amplitude_normalization": "not_applied",
        "limiter": "not_applied",
    }
    mismatches = [
        key for key, expected in expected_metadata.items() if metadata.get(key) != expected
    ]
    if mismatches:
        raise SpatialOmniDatasetError(
            f"point {point_id} native FOA metadata violates no-processing contract: "
            + ", ".join(mismatches)
        )
    source_ids = metadata.get("source_ids")
    if (
        not isinstance(source_ids, list)
        or not source_ids
        or any(not isinstance(source_id, str) or not source_id for source_id in source_ids)
        or metadata.get("summation_order") != source_ids
    ):
        raise SpatialOmniDatasetError(
            f"point {point_id} native FOA source_ids and summation_order must be "
            "the same non-empty source list"
        )
    return NativeFoaPoint(
        point_id=point_id,
        point_root=root,
        wav_path=wav_path,
        sidecar_path=sidecar_path,
    )


def _load_specs(inputs_root: str | Path) -> dict[str, Mapping[str, Any]]:
    root = Path(inputs_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise SpatialOmniDatasetError(f"inputs root is not a directory: {root}")
    specs: dict[str, Mapping[str, Any]] = {}
    for spec_path in sorted(root.glob("*/spec.json")):
        value = _load_json(spec_path, owner="point spec")
        if not isinstance(value, Mapping):
            raise SpatialOmniDatasetError(f"point spec must be an object: {spec_path}")
        point_id = _nonempty_string(value.get("point_id"), owner=f"{spec_path} point_id")
        if point_id != spec_path.parent.name:
            raise SpatialOmniDatasetError(
                f"point spec ID does not match directory: {point_id} != {spec_path.parent.name}"
            )
        if point_id in specs:
            raise SpatialOmniDatasetError(f"duplicate point spec: {point_id}")
        specs[point_id] = value
    if not specs:
        raise SpatialOmniDatasetError("inputs root contains no point specs")
    return specs


def _resolve_group(
    point_id: str,
    specs: Mapping[str, Mapping[str, Any]],
) -> str:
    current = point_id
    visited = set()
    while True:
        if current in visited:
            raise SpatialOmniDatasetError(f"twin_of cycle contains point {point_id}")
        visited.add(current)
        try:
            spec = specs[current]
        except KeyError as exc:
            raise SpatialOmniDatasetError(
                f"point {point_id} twin_of references missing point {current}"
            ) from exc
        parent = spec.get("twin_of")
        if parent is None:
            return current
        current = _nonempty_string(parent, owner=f"point {current} twin_of")


def split_points_by_twin_group(
    point_ids: Sequence[str],
    specs: Mapping[str, Mapping[str, Any]],
    *,
    valid_fraction: float,
    seed: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Assign complete transitive twin groups to train or valid."""

    fraction = _finite_number(valid_fraction, owner="valid_fraction")
    if not 0.0 < fraction < 1.0:
        raise SpatialOmniDatasetError("valid_fraction must be strictly between 0 and 1")
    group_of = {point_id: _resolve_group(point_id, specs) for point_id in point_ids}
    groups = sorted(set(group_of.values()))
    if len(groups) < 2:
        raise SpatialOmniDatasetError("at least two twin groups are required for train/valid")
    random.Random(int(seed)).shuffle(groups)
    valid_count = max(1, min(len(groups) - 1, round(len(groups) * fraction)))
    valid_groups = set(groups[:valid_count])
    split_of = {
        point_id: ("valid" if group in valid_groups else "train")
        for point_id, group in group_of.items()
    }
    return split_of, group_of


def _question_text(question: Mapping[str, Any]) -> str:
    return _nonempty_string(
        question.get("question_en", question.get("question")), owner="question text"
    ).rstrip()


def _is_refusal(question: Mapping[str, Any]) -> bool:
    type_id = question.get("type_id")
    for key in ("is_refusal", "refusal"):
        if key in question and not isinstance(question[key], bool):
            raise SpatialOmniDatasetError(f"{key} must be boolean when present")
    legacy_marker = (
        question.get("is_refusal") is True
        or question.get("refusal") is True
        or question.get("format") == "refusal"
    )
    canonical_marker = (
        type_id == "QV2-TR-REFUSAL"
        or "refusal_gold" in question
        or "provided_modalities" in question
        or "withheld_modalities" in question
    )
    if legacy_marker and not canonical_marker:
        raise SpatialOmniDatasetError(
            "refusal rows must use QV2-TR-REFUSAL with refusal_gold=true, "
            "provided_modalities=[audio], and withheld_modalities=[video]"
        )
    if not canonical_marker:
        return False
    expected = {
        "type_id": "QV2-TR-REFUSAL",
        "refusal_gold": True,
        "provided_modalities": ["audio"],
        "withheld_modalities": ["video"],
    }
    mismatches = [
        key for key, expected_value in expected.items() if question.get(key) != expected_value
    ]
    if mismatches:
        raise SpatialOmniDatasetError(
            "refusal row modality declarations disagree: " + ", ".join(mismatches)
        )
    return True


def _normalized_options(question: Mapping[str, Any]) -> list[tuple[str, str]]:
    options = question.get("options")
    if not isinstance(options, list) or len(options) < 2:
        raise SpatialOmniDatasetError("MCQ options must contain at least two entries")
    result = []
    for index, option in enumerate(options):
        if isinstance(option, str) and option:
            result.append((option, option))
            continue
        if isinstance(option, Mapping):
            value = _nonempty_string(option.get("value"), owner=f"options[{index}].value")
            label = _nonempty_string(
                option.get("label_en", value), owner=f"options[{index}].label_en"
            )
            result.append((value, label))
            continue
        raise SpatialOmniDatasetError(f"options[{index}] must be a string or object")
    values = [value for value, _label in result]
    if len(values) != len(set(values)):
        raise SpatialOmniDatasetError("MCQ option values must be unique")
    return result


def _mcq_fields(question: Mapping[str, Any]) -> dict[str, Any]:
    options = _normalized_options(question)
    answer = _nonempty_string(
        question.get("answer", question.get("answer_value")), owner="MCQ answer"
    )
    values = [value for value, _label in options]
    labels = [label for _value, label in options]
    if answer in values:
        answer_index = values.index(answer)
    elif answer in labels:
        answer_index = labels.index(answer)
    else:
        raise SpatialOmniDatasetError("MCQ answer is absent from options")
    prompt = _question_text(question) + " Choose one: " + " / ".join(labels) + "."
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if answer_index >= len(letters):
        raise SpatialOmniDatasetError("MCQ has too many options for answer letters")
    return {
        "answer_format": "mcq",
        "question": prompt,
        "prompt": prompt,
        "answer": labels[answer_index],
        "canonical_answer": values[answer_index],
        "options": labels,
        "option_values": values,
        "answer_letter": letters[answer_index],
    }


def _numeric_fields(question: Mapping[str, Any]) -> dict[str, Any]:
    answer_numeric = _finite_number(question.get("answer_numeric"), owner="answer_numeric")
    unit = _nonempty_string(question.get("unit"), owner="numeric unit")
    raw_bands = question.get("scoring_bands")
    if not isinstance(raw_bands, list) or not raw_bands:
        raise SpatialOmniDatasetError("numeric scoring_bands must be a non-empty list")
    bands = [_finite_number(value, owner="numeric scoring band") for value in raw_bands]
    if any(value <= 0.0 for value in bands) or bands != sorted(set(bands)):
        raise SpatialOmniDatasetError(
            "numeric scoring_bands must be unique positive ascending values"
        )
    prompt = _question_text(question)
    answer_text = str(answer_numeric)
    return {
        "answer_format": "numeric_banded",
        "question": prompt,
        "prompt": prompt,
        "answer": answer_text,
        "canonical_answer": answer_text,
        "answer_numeric": answer_numeric,
        "unit": unit,
        "scoring_bands": bands,
    }


def _row_for_question(
    question: Mapping[str, Any],
    *,
    split: str,
    group_id: str,
    twin_of: str | None,
    audio_path: Path,
    video_path: Path | None,
) -> dict[str, Any]:
    point_id = _nonempty_string(question.get("point_id"), owner="question point_id")
    type_id = _nonempty_string(question.get("type_id"), owner="question type_id")
    question_id = question.get("question_id")
    if question_id is None:
        question_id = f"{type_id.lower()}__{point_id}"
    question_id = _nonempty_string(question_id, owner="question_id")
    numeric = question.get("format", question.get("answer_format")) == "numeric_banded"
    fields = _numeric_fields(question) if numeric else _mcq_fields(question)
    refusal = _is_refusal(question)
    if refusal and numeric:
        raise SpatialOmniDatasetError("refusal rows cannot use numeric_banded format")
    modality_mask = {"audio": True, "video": not refusal}
    row = {
        "qa_id": question_id,
        "pair_id": question_id,
        "split": split,
        "audio_id": f"{point_id}__native_rlr_foa",
        "audio_path": str(audio_path),
        "task_name": type_id.lower(),
        "question_class": type_id,
        **fields,
        "is_refusal": refusal,
        "input_modalities": [key for key, enabled in modality_mask.items() if enabled],
        "modality_mask": modality_mask,
        "audio_format": {
            "source": "native_rlr_foa",
            **rlr_foa_contract(),
            "sample_rate_hz": EXPECTED_SAMPLE_RATE_HZ,
            "sample_count": EXPECTED_SAMPLE_COUNT,
            "hrtf_used": False,
            "resampling": "not_applied",
            "amplitude_normalization": "not_applied",
            "limiter": "not_applied",
        },
        "answer_meta": {
            "point_id": point_id,
            "twin_of": twin_of,
            "twin_group_id": group_id,
            "modality_expectation": question.get("modality_expectation"),
        },
    }
    if refusal:
        row.update(
            {
                "refusal_gold": True,
                "provided_modalities": ["audio"],
                "withheld_modalities": ["video"],
            }
        )
    if numeric:
        for key in (
            "answer_unit",
            "azimuth_convention",
            "event_frame",
            "event_start_sample",
            "gold_authority",
            "scoring",
            "semantic_answer_source_slot",
        ):
            if key in question:
                row[key] = question[key]
    if not refusal:
        if video_path is None:
            raise SpatialOmniDatasetError(
                f"non-refusal point {point_id} requires a review clip"
            )
        row["video_path"] = str(video_path)
    return row


def _fresh_staging(output: str | Path) -> tuple[WorkspacePathPolicy, Path, Path]:
    unresolved = Path(output).expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    if os.path.lexists(unresolved):
        raise SpatialOmniDatasetError(f"output exists; refusing to clobber: {unresolved}")
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    parent = unresolved.parent.resolve(strict=True)
    policy = WorkspacePathPolicy.from_roots((parent,))
    destination = policy.resolve_output(
        parent / unresolved.name, owner="Spatial-Omni native FOA dataset output"
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent))
    return policy, destination, staging


def _copy_verified_foa(
    source: NativeFoaPoint,
    *,
    staging: Path,
    destination: Path,
    split: str,
) -> Path:
    relative = Path("audio") / split / source.point_id / "mixture.wav"
    staged_wav = staging / relative
    staged_wav.parent.mkdir(parents=True)
    staged_sidecar = staged_wav.with_suffix(staged_wav.suffix + ".json")
    shutil.copyfile(source.wav_path, staged_wav)
    shutil.copyfile(source.sidecar_path, staged_sidecar)
    try:
        read_float32_wav(staged_wav, verify_sidecar=True)
    except AudioContractError as exc:
        raise SpatialOmniDatasetError(
            f"copied native FOA failed sidecar verification for {source.point_id}: {exc}"
        ) from exc
    return destination / relative


def build_dataset(
    *,
    questions_path: str | Path,
    inputs_root: str | Path,
    clips_dir: str | Path,
    audio_root: str | Path,
    output: str | Path,
    valid_fraction: float = 0.12,
    seed: int = 20_260_823,
    generated_at: str | None = None,
) -> Path:
    questions_document = _load_json(questions_path, owner="questions")
    if not isinstance(questions_document, Mapping) or not isinstance(
        questions_document.get("questions"), list
    ):
        raise SpatialOmniDatasetError("questions document must contain a questions list")
    questions = questions_document["questions"]
    if not questions:
        raise SpatialOmniDatasetError("questions list is empty")
    if not all(isinstance(question, Mapping) for question in questions):
        raise SpatialOmniDatasetError("every question must be an object")
    specs = _load_specs(inputs_root)
    point_ids = sorted(
        {
            _nonempty_string(question.get("point_id"), owner="question point_id")
            for question in questions
        }
    )
    missing_specs = [point_id for point_id in point_ids if point_id not in specs]
    if missing_specs:
        raise SpatialOmniDatasetError(f"questions reference missing specs: {missing_specs}")
    split_of, group_of = split_points_by_twin_group(
        point_ids,
        specs,
        valid_fraction=valid_fraction,
        seed=seed,
    )
    audio_base = Path(audio_root).expanduser().resolve(strict=True)
    clips_base = Path(clips_dir).expanduser().resolve(strict=True)
    if not audio_base.is_dir() or not clips_base.is_dir():
        raise SpatialOmniDatasetError("audio_root and clips_dir must be directories")
    refusal_by_point: dict[str, bool] = {}
    for question in questions:
        point_id = str(question["point_id"])
        refusal = _is_refusal(question)
        refusal_by_point[point_id] = refusal_by_point.get(point_id, True) and refusal
    foa_sources = {
        point_id: validate_native_foa_point(audio_base / point_id, point_id=point_id)
        for point_id in point_ids
    }
    clips: dict[str, Path | None] = {}
    for point_id in point_ids:
        clip = clips_base / f"{point_id}.mp4"
        if clip.is_file():
            clips[point_id] = clip.resolve()
        elif refusal_by_point[point_id]:
            clips[point_id] = None
        else:
            raise SpatialOmniDatasetError(
                f"non-refusal/offscreen/numeric point {point_id} lacks clip: {clip}"
            )

    policy, destination, staging = _fresh_staging(output)
    try:
        final_audio_paths = {
            point_id: _copy_verified_foa(
                source,
                staging=staging,
                destination=destination,
                split=split_of[point_id],
            )
            for point_id, source in sorted(foa_sources.items())
        }
        rows: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
        seen_question_ids = set()
        for question in questions:
            point_id = str(question["point_id"])
            split = split_of[point_id]
            row = _row_for_question(
                question,
                split=split,
                group_id=group_of[point_id],
                twin_of=specs[point_id].get("twin_of"),
                audio_path=final_audio_paths[point_id],
                video_path=clips[point_id],
            )
            if row["qa_id"] in seen_question_ids:
                raise SpatialOmniDatasetError(f"duplicate question_id: {row['qa_id']}")
            seen_question_ids.add(row["qa_id"])
            rows[split].append(row)
        qa_directory = staging / "qa"
        qa_directory.mkdir()
        for split in ("train", "valid"):
            with (qa_directory / f"{split}.jsonl").open("x", encoding="utf-8") as handle:
                for row in rows[split]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts = Counter(split_of.values())
        provenance = {
            "status": "research_only",
            "research_only": True,
            "qualification_claim": False,
            "episode_counted": False,
            "formal_dataset_count": 0,
            "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
            "claim_boundary": (
                "Spatial-Omni QA rows backed by native RLR order-1 FOA; no formal "
                "dataset admission or benchmark claim."
            ),
            "sources": {
                "questions": str(Path(questions_path).expanduser().resolve()),
                "inputs_root": str(Path(inputs_root).expanduser().resolve()),
                "clips_dir": str(clips_base),
                "audio_root": str(audio_base),
            },
            "audio_provenance": {
                "source": "native_rlr_foa",
                "accepted_relative_path": NATIVE_FOA_RELATIVE_PATH.as_posix(),
                "format": rlr_foa_contract(),
                "sample_rate_hz": EXPECTED_SAMPLE_RATE_HZ,
                "sample_count": EXPECTED_SAMPLE_COUNT,
                "copy_semantics": (
                    "original canonical float32 WAVE and existing verified sidecar; "
                    "no HRTF, resampling, amplitude normalization, limiting, or remix"
                ),
                "fallbacks": "none; binaural and foaish are rejected",
            },
            "trainer_compatibility": {
                "status": "blocked",
                "reason": (
                    "the current ordinary Spatial-Omni video collator does not yet "
                    "consume modality_mask or sparse per-row video_path; refusal rows "
                    "must not be sent to training before that explicit wiring"
                ),
            },
            "split": {
                "rule": "transitive twin group",
                "seed": int(seed),
                "valid_fraction": float(valid_fraction),
            },
            "counts": {
                "train_rows": len(rows["train"]),
                "valid_rows": len(rows["valid"]),
                "train_points": counts["train"],
                "valid_points": counts["valid"],
                "refusal_rows": sum(
                    row["is_refusal"] for split_rows in rows.values() for row in split_rows
                ),
                "numeric_banded_rows": sum(
                    row["answer_format"] == "numeric_banded"
                    for split_rows in rows.values()
                    for row in split_rows
                ),
            },
        }
        with (staging / "DATASET_PROVENANCE.json").open("x", encoding="utf-8") as handle:
            json.dump(provenance, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return atomic_publish_directory(policy, staging, destination)
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--inputs-root", required=True)
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--valid-fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20_260_823)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = build_dataset(
            questions_path=args.questions,
            inputs_root=args.inputs_root,
            clips_dir=args.clips_dir,
            audio_root=args.audio_root,
            output=args.output,
            valid_fraction=args.valid_fraction,
            seed=args.seed,
        )
    except (OSError, TypeError, KeyError, SpatialOmniDatasetError) as exc:
        print(json.dumps({"error": str(exc), "output_written": False}, ensure_ascii=False))
        return 2
    provenance = _load_json(output / "DATASET_PROVENANCE.json", owner="provenance")
    print(json.dumps({"output": str(output), "counts": provenance["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

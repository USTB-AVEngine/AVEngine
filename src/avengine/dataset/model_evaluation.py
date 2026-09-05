"""Private adapter for the installed Spatial-Omni QA benchmark.

The AVEngine export keeps model inputs and gold answers in separate JSONL
files. The installed Spatial-Omni collator predates that boundary and
requires an answer while it builds a supervised evaluation batch. This
adapter creates a clearly private, no-clobber runtime QA root by joining the
two files only when the audio layout is compatible with the model (4-channel
FOA). Stereo Episode masters are reported as not_run instead of being
silently reinterpreted as FOA.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import gzip
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unicodedata
import wave
from typing import Any


SPATIAL_OMNI_ADAPTER_SCHEMA = "avengine_spatial_omni_eval_adapter_v1"

_INTERNAL_ACTOR_ID = re.compile(
    r"(?<![A-Za-z0-9])source[0-9]+(?![A-Za-z0-9])", re.IGNORECASE
)


def _contains_internal_actor_id(value: Any) -> bool:
    if isinstance(value, str):
        return _INTERNAL_ACTOR_ID.search(value) is not None
    if isinstance(value, Mapping):
        return any(
            _contains_internal_actor_id(key) or _contains_internal_actor_id(child)
            for key, child in value.items()
        )
    if _is_sequence(value):
        return any(_contains_internal_actor_id(child) for child in value)
    return False




class ModelEvaluationAdapterError(ValueError):
    """The private model-runtime bridge cannot preserve the input contract."""


def _fail(message: str) -> ModelEvaluationAdapterError:
    return ModelEvaluationAdapterError(
        f"Spatial-Omni evaluation adapter cannot proceed: {message}"
    )


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        if path.suffix == ".gz":
            handle = gzip.open(path, "rt", encoding="utf-8")
        else:
            handle = path.open("r", encoding="utf-8")
        with handle:
            rows = []
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise _fail(f"JSONL row is not an object: {path}")
                rows.append(row)
            return rows
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _fail(f"cannot read JSONL {path}: {exc}") from exc


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            dict(value),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")


def _resolve_path(raw: Any, *, base_dir: Path, owner: str) -> Path:
    value = raw.get("path") if isinstance(raw, Mapping) else raw
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{owner} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise _fail(f"{owner} is missing: {path}") from exc
    if not resolved.is_file():
        raise _fail(f"{owner} must be a file: {resolved}")
    return resolved


def _audio_channels(path: Path) -> int:
    try:
        import soundfile as sf

        return int(sf.info(path).channels)
    except (ImportError, OSError, RuntimeError):
        try:
            with wave.open(str(path), "rb") as handle:
                return int(handle.getnchannels())
        except (OSError, wave.Error) as exc:
            raise _fail(f"cannot inspect audio channels for {path}") from exc


def _private_join(
    model_inputs: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    answers_by_id = {}
    for row in answers:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise _fail("private answer row has no question_id")
        if question_id in answers_by_id:
            raise _fail(f"private answers repeat question_id {question_id!r}")
        answers_by_id[question_id] = row
    joined = []
    unsupported: list[str] = []
    for row in model_inputs:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise _fail("model input row has no question_id")
        answer = answers_by_id.get(question_id)
        if answer is None:
            raise _fail(f"no private answer for model input {question_id!r}")
        audio_path = row.get("audio_path")
        video_path = row.get("video_path")
        if not isinstance(audio_path, str) or not isinstance(video_path, str):
            input_block = row.get("input")
            if isinstance(input_block, Mapping):
                audio_path = input_block.get("audio_path")
                video_path = input_block.get("video_path")
        if not isinstance(audio_path, str) or not isinstance(video_path, str):
            raise _fail(f"model input {question_id!r} has no media paths")
        audio_file = Path(audio_path).expanduser().resolve()
        video_file = Path(video_path).expanduser().resolve()
        if not audio_file.is_file() or not video_file.is_file():
            raise _fail(f"model input {question_id!r} has missing media files")
        channels = _audio_channels(audio_file)
        if channels != 4:
            unsupported.append(question_id)
            continue
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise _fail(f"model input {question_id!r} has no question")
        joined.append(
            {
                "pair_id": question_id,
                "question_id": question_id,
                "episode_id": row.get("episode_id"),
                "task_name": row.get("task_name", row.get("question_type")),
                "question_class": row.get("question_class"),
                "question": question,
                "prompt": row.get("prompt", question),
                "audio_path": str(audio_file),
                "video_path": str(video_file),
                "answer": answer.get("truth"),
                "canonical_answer": answer.get("truth"),
            }
        )
    return joined, unsupported


def prepare_spatial_omni_qa_root(
    *,
    model_inputs_path: str | Path,
    answers_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Prepare a private test split for the existing benchmark collator.

    If the export contains stereo masters, no test split is written and the
    returned manifest is not_run with a precise 4-channel-FOA reason.
    """

    model_inputs = Path(model_inputs_path).expanduser().resolve()
    answers = Path(answers_path).expanduser().resolve()
    if not model_inputs.is_file() or not answers.is_file():
        raise _fail("model input and answer JSONL paths must exist")
    input_rows = _read_jsonl(model_inputs)
    answer_rows = _read_jsonl(answers)
    joined, unsupported = _private_join(input_rows, answer_rows)
    output = Path(output_root).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise _fail(f"refusing to overwrite output root: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    try:
        status = "pass" if not unsupported else "not_run"
        reason = None
        if unsupported:
            reason = (
                "Spatial-Omni collator requires 4-channel FOA "
                "[W,Y,Z,X]; stereo Episode masters cannot be reinterpreted"
            )
        (temp / "qa").mkdir()
        if joined:
            _write_jsonl(temp / "qa/test.jsonl", joined)
        manifest = {
            "schema": SPATIAL_OMNI_ADAPTER_SCHEMA,
            "status": status,
            "private_runtime_input": True,
            "answer_joined_for_legacy_collator": bool(joined),
            "model_input_source": str(model_inputs),
            "answer_source": str(answers),
            "counts": {
                "input_questions": len(input_rows),
                "joined_questions": len(joined),
                "unsupported_questions": len(unsupported),
            },
            "unsupported_question_ids": unsupported,
            "reason": reason,
            "outputs": {"qa_root": "qa" if joined else None},
        }
        _write_json(temp / "manifest.json", manifest)
        try:
            os.rename(temp, output)
        except FileExistsError as exc:
            raise _fail(f"refusing to overwrite output root: {output}") from exc
    except Exception:
        # Preserve the failed private staging tree for diagnosis.
        raise
    return manifest


def _public_question_text(row: Mapping[str, Any]) -> str | None:
    value = row.get("question_en", row.get("question"))
    if isinstance(value, Mapping):
        value = value.get("en") or value.get("question_en") or value.get("zh")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _public_options(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = row.get("options")
    if not _is_sequence(value):
        return []
    result = []
    for index, option in enumerate(value):
        if isinstance(option, Mapping):
            label_en = (
                option.get("label_en")
                or option.get("value")
                or option.get("label_zh")
            )
            label_zh = option.get("label_zh") or label_en
        else:
            label_en = option
            label_zh = option
        if not isinstance(label_en, str) or not label_en.strip():
            return []
        result.append(
            {
                "index": index,
                "letter": chr(ord("A") + index),
                "label_en": label_en.strip(),
                "label_zh": (
                    str(label_zh).strip()
                    if isinstance(label_zh, str) and label_zh.strip()
                    else label_en.strip()
                ),
            }
        )
    return result


def _link_no_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise _fail(f"refusing to overwrite evaluation media: {destination}")
    try:
        os.link(source, destination)
    except OSError as exc:
        raise _fail(
            "evaluation media must be hard-linkable to preserve the single "
            f"source file ({source} -> {destination}): {exc}"
        ) from exc


def _mux_stereo_video(
    *,
    video: Path,
    audio: Path,
    destination: Path,
    ffmpeg: str,
    ffprobe: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-ar",
        "16000",
        "-ac",
        "2",
        "-shortest",
        str(destination),
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise _fail(f"cannot mux full_av evaluation media: {exc}") from exc
    if completed.returncode != 0:
        raise _fail(
            f"full_av evaluation mux failed: {completed.stderr.strip()[-400:]}"
        )
    probe = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,channels,sample_rate",
        "-of",
        "json",
        str(destination),
    ]
    try:
        checked = subprocess.run(
            probe,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise _fail(f"cannot read back full_av evaluation media: {exc}") from exc
    if checked.returncode != 0:
        raise _fail(
            f"full_av evaluation readback failed: {checked.stderr.strip()[-400:]}"
        )
    try:
        streams = json.loads(checked.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise _fail("full_av evaluation readback is not JSON") from exc
    video_streams = [item for item in streams if item.get("codec_type") == "video"]
    audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
    if not video_streams or not audio_streams:
        raise _fail("full_av evaluation media lacks video or audio stream")
    audio_stream = audio_streams[0]
    if int(audio_stream.get("channels", 0)) != 2 or int(
        audio_stream.get("sample_rate", 0)
    ) != 16000:
        raise _fail(
            "full_av evaluation readback is not 2-channel 16 kHz audio"
        )


def prepare_qwen25_omni_pilot_inputs(
    *,
    model_inputs_path: str | Path,
    output_root: str | Path,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    """Build answer-free 2-channel inputs for the existing Qwen2.5 pilot.

    The old pilot runner needs relative media paths and multiple condition
    variants. Source stereo WAV/video files are hard-linked, while one
    full_av mux is produced per unique media pair. No source media is copied
    and no answer fields are written to model_inputs.json.
    """

    source = Path(model_inputs_path).expanduser().resolve()
    if not source.is_file():
        raise _fail(f"model input JSONL is missing: {source}")
    source_rows = _read_jsonl(source)
    output = Path(output_root).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise _fail(f"refusing to overwrite output root: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    skipped: list[dict[str, Any]] = []
    media_actions: dict[str, int] = {}
    model_items: list[dict[str, Any]] = []
    mux_by_pair: dict[tuple[str, str], Path] = {}
    try:
        for index, row in enumerate(source_rows):
            question_id = row.get("question_id")
            if not isinstance(question_id, str) or not question_id.strip():
                skipped.append({"index": index, "reason": "missing_question_id"})
                continue
            question_id = question_id.strip()
            question = _public_question_text(row)
            options = _public_options(row)
            if _contains_internal_actor_id(
                {"question_id": question_id, "question": question, "options": options}
            ):
                skipped.append(
                    {"question_id": question_id, "reason": "internal_actor_identifier"}
                )
                continue
            if question is None:
                skipped.append(
                    {"question_id": question_id, "reason": "missing_question"}
                )
                continue
            if len(options) < 2 or len(options) > 4:
                skipped.append(
                    {"question_id": question_id, "reason": "no_mcq_options"}
                )
                continue
            raw_audio = row.get("audio_path")
            raw_video = row.get("video_path")
            if not isinstance(raw_audio, str) or not isinstance(raw_video, str):
                input_block = row.get("input")
                if isinstance(input_block, Mapping):
                    raw_audio = input_block.get("audio_path")
                    raw_video = input_block.get("video_path")
            if not isinstance(raw_audio, str) or not isinstance(raw_video, str):
                skipped.append(
                    {"question_id": question_id, "reason": "missing_media_paths"}
                )
                continue
            audio = Path(raw_audio).expanduser().resolve()
            video = Path(raw_video).expanduser().resolve()
            if not audio.is_file() or not video.is_file():
                skipped.append(
                    {"question_id": question_id, "reason": "missing_media_files"}
                )
                continue
            if _audio_channels(audio) != 2:
                skipped.append(
                    {"question_id": question_id, "reason": "audio_not_stereo_2ch"}
                )
                continue
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", question_id).strip("-")
            if not stem:
                stem = f"question-{index:05d}"
            stem = f"{index:05d}-{stem[:120]}"
            video_only = temp / "media" / "video_only" / f"{stem}.mp4"
            audio_only = temp / "media" / "audio_only" / f"{stem}.wav"
            pair = (str(video), str(audio))
            full_source = mux_by_pair.get(pair)
            if full_source is None:
                full_source = (
                    temp / "media" / "full_av" / f"pair-{len(mux_by_pair):05d}.mp4"
                )
                _mux_stereo_video(
                    video=video,
                    audio=audio,
                    destination=full_source,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                )
                mux_by_pair[pair] = full_source
                media_actions["full_av_mux"] = media_actions.get("full_av_mux", 0) + 1
            else:
                media_actions["full_av_hardlink"] = media_actions.get(
                    "full_av_hardlink", 0
                ) + 1
            _link_no_replace(video, video_only)
            _link_no_replace(audio, audio_only)
            media_actions["video_only_hardlink"] = media_actions.get(
                "video_only_hardlink", 0
            ) + 1
            media_actions["audio_only_hardlink"] = media_actions.get(
                "audio_only_hardlink", 0
            ) + 1
            public_paths = {
                "full_av": full_source,
                "video_only": video_only,
                "audio_only": audio_only,
            }
            for condition in ("full_av", "video_only", "audio_only"):
                model_items.append(
                    {
                        "input_id": f"{question_id}__{condition}",
                        "sample_id": question_id,
                        "condition": condition,
                        "question_en": question,
                        "question_zh": question,
                        "options": options,
                        "media_path": public_paths[condition]
                        .relative_to(temp)
                        .as_posix(),
                    }
                )
        status = "ready" if model_items else "not_run"
        if model_items:
            model_payload = {
                "schema": "avengine_pilot_model_inputs_v1",
                "evaluation_status": "pilot_pre_review_unverified",
                "qualification_claim": False,
                "inference_contract": {
                    "response_format": "Return exactly one option letter (A, B, C, ...).",
                    "prediction_jsonl": {
                        "input_id": "sample__full_av",
                        "prediction": "A",
                    },
                },
                "condition_order": ["full_av", "video_only", "audio_only"],
                "sample_count": len(model_items) // 3,
                "input_count": len(model_items),
                "items": model_items,
            }
            _write_json(temp / "model_inputs.json", model_payload)
        manifest = {
            "schema": SPATIAL_OMNI_ADAPTER_SCHEMA,
            "runner": "qwen2_5_omni_pilot",
            "status": status,
            "evaluation_status": "pilot_pre_review_unverified",
            "qualification_claim": False,
            "input_source": str(source),
            "audio_contract": {
                "source_channels": 2,
                "source_layout": "stereo",
                "model_preprocessing": "qwen_omni_utils_v2_5_downmix_to_mono_16k",
                "spatial_conclusion_allowed": False,
            },
            "conditions": ["full_av", "video_only", "audio_only", "text_only"],
            "counts": {
                "source_questions": len(source_rows),
                "runnable_questions": len(model_items) // 3,
                "model_inputs": len(model_items),
                "skipped_questions": len(skipped),
            },
            "skipped_questions": skipped,
            "media_actions": dict(sorted(media_actions.items())),
            "outputs": {"model_inputs": "model_inputs.json" if model_items else None},
        }
        _write_json(temp / "manifest.json", manifest)
        try:
            os.rename(temp, output)
        except FileExistsError as exc:
            raise _fail(f"refusing to overwrite output root: {output}") from exc
    except Exception:
        raise
    return manifest


def _normalise_label(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", str(value)).casefold().strip(),
    )


def prepare_qwen25_omni_pilot_gold(
    *,
    model_inputs_path: str | Path,
    answers_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write the private gold file required by score_pilot_baseline.py.

    Public model inputs remain answer-free. The scorer gold is materialized
    from the separate answer sidecar only after a model run is scheduled.
    Questions without MCQ options or without an unambiguous truth-to-option
    match are retained in a private skipped list.
    """

    source = Path(model_inputs_path).expanduser().resolve()
    answers_path = Path(answers_path).expanduser().resolve()
    if not source.is_file() or not answers_path.is_file():
        raise _fail("Qwen model inputs and answers JSONL paths must exist")
    input_rows = _read_jsonl(source)
    answer_rows = _read_jsonl(answers_path)
    answer_by_id = {
        str(row["question_id"]): row
        for row in answer_rows
        if isinstance(row.get("question_id"), str)
    }
    gold_items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in input_rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            skipped.append({"reason": "missing_question_id"})
            continue
        answer = answer_by_id.get(question_id)
        if answer is None:
            skipped.append({"question_id": question_id, "reason": "missing_answer_record"})
            continue
        if answer.get("status") != "pass" or answer.get("truth") is None:
            skipped.append({"question_id": question_id, "reason": "qa_not_valid"})
            continue
        options = _public_options(row)
        if len(options) < 2 or len(options) > 4:
            skipped.append({"question_id": question_id, "reason": "no_mcq_options"})
            continue
        matches: list[int] = []
        forms = answer.get("forms")
        mcq_form = forms.get("mcq") if isinstance(forms, Mapping) else None
        gold = mcq_form.get("gold") if isinstance(mcq_form, Mapping) else None
        correct_index = gold.get("correct_index") if isinstance(gold, Mapping) else None
        form_options = mcq_form.get("options") if isinstance(mcq_form, Mapping) else None
        if (
            isinstance(correct_index, int)
            and not isinstance(correct_index, bool)
            and 0 <= correct_index < len(options)
            and _is_sequence(form_options)
            and len(form_options) == len(options)
        ):
            # The unified form's gold index is authoritative for values such
            # as count_pair whose scalar truth is not the visible option text.
            matches = [correct_index]
        else:
            expected = {
                _normalise_label(answer.get("truth")),
                _normalise_label(answer.get("truth_label")),
            }
            expected.discard("")
            matches = [
                index
                for index, option in enumerate(options)
                if _normalise_label(option["label_en"]) in expected
                or _normalise_label(option["label_zh"]) in expected
            ]
        if len(matches) != 1:
            skipped.append(
                {
                    "question_id": question_id,
                    "reason": "truth_not_unique_in_options",
                    "match_count": len(matches),
                }
            )
            continue
        selected = options[matches[0]]
        gold_items.append(
            {
                "sample_id": question_id,
                "source_question_id": question_id,
                "episode_id": row.get("episode_id"),
                "type_id": row.get("catalog_id") or row.get("question_type"),
                "required_modalities": list(answer.get("required_modalities") or []),
                "option_count": len(options),
                "options": [
                    {
                        "index": index,
                        "value": option["label_en"],
                        "label_en": option["label_en"],
                        "label_zh": option["label_zh"],
                    }
                    for index, option in enumerate(options)
                ],
                "answer_index": matches[0],
                "answer_value": selected["label_en"],
                "selection_bucket": answer.get("selection_bucket"),
                "scoring_source": (
                    "private_forms.mcq.gold"
                    if matches and isinstance(gold, Mapping)
                    else "private_truth_label"
                ),
            }
        )
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise _fail(f"refusing to overwrite Qwen private gold: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "avengine_pilot_gold_v1",
        "evaluation_status": "pilot_pre_review_unverified",
        "qualification_claim": False,
        "source_model_inputs": str(source),
        "source_answers": str(answers_path),
        "condition_order": ["full_av", "video_only", "audio_only"],
        "sample_count": len(gold_items),
        "items": gold_items,
        "skipped_questions": skipped,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema": "avengine_pilot_gold_manifest_v1",
        "status": "ready" if gold_items else "not_run",
        "gold_path": str(output),
        "counts": {
            "source_questions": len(input_rows),
            "gold_questions": len(gold_items),
            "skipped_questions": len(skipped),
        },
        "skipped_questions": skipped,
    }


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _find_event_window(
    value: Any,
    *,
    sample_rate_hz: int,
    clip_seconds: float,
) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        start_s = _numeric(value.get("start_s"))
        end_s = _numeric(value.get("end_s"))
        if start_s is not None and end_s is not None:
            start, end = start_s, end_s
        else:
            start_sample = _numeric(value.get("start_sample"))
            end_sample = _numeric(
                value.get("end_sample_exclusive", value.get("end_sample"))
            )
            if start_sample is None or end_sample is None:
                start = end = None
            else:
                start = start_sample / sample_rate_hz
                end = end_sample / sample_rate_hz
        if start is not None and end is not None:
            start = max(0.0, min(float(start), clip_seconds))
            end = max(0.0, min(float(end), clip_seconds))
            if end > start:
                return start, end
        for child in value.values():
            found = _find_event_window(
                child,
                sample_rate_hz=sample_rate_hz,
                clip_seconds=clip_seconds,
            )
            if found is not None:
                return found
    elif _is_sequence(value):
        for child in value:
            found = _find_event_window(
                child,
                sample_rate_hz=sample_rate_hz,
                clip_seconds=clip_seconds,
            )
            if found is not None:
                return found
    return None


def prepare_whisper_review_request(
    *,
    model_inputs_path: str | Path,
    answers_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
    device: str = "cpu",
    decoding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a truth-free request for the installed Whisper review entry.

    QA-12 transcript items are restricted to evidence-backed event windows.
    QA-19 time items may use a full-clip first-event window when their evidence
    contains a first event. Spatial/location questions are listed as skipped
    because Whisper's mono ASR output cannot establish them.
    """

    source = Path(model_inputs_path).expanduser().resolve()
    answers_path = Path(answers_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    if not source.is_file() or not answers_path.is_file():
        raise _fail("Whisper model-input and answer JSONL paths must exist")
    if not model.is_file():
        raise _fail(f"Whisper model weights are missing: {model}")
    input_rows = _read_jsonl(source)
    answer_rows = _read_jsonl(answers_path)
    answer_by_id = {
        str(row["question_id"]): row
        for row in answer_rows
        if isinstance(row.get("question_id"), str)
    }
    skipped: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for row in input_rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            skipped.append({"reason": "missing_question_id"})
            continue
        if _contains_internal_actor_id(question_id):
            skipped.append(
                {"question_id": question_id, "reason": "internal_actor_identifier"}
            )
            continue
        answer = answer_by_id.get(question_id)
        if answer is None:
            skipped.append({"question_id": question_id, "reason": "missing_answer_record"})
            continue
        if answer.get("status") != "pass" or answer.get("truth") is None:
            skipped.append({"question_id": question_id, "reason": "qa_not_valid"})
            continue
        answer_type = answer.get("answer_type")
        evidence = answer.get("evidence")
        input_block = row.get("input")
        clock = input_block.get("media_clock") if isinstance(input_block, Mapping) else {}
        if not isinstance(clock, Mapping):
            clock = {}
        sample_rate = int(clock.get("sample_rate_hz") or 16_000)
        clip_seconds = _numeric(clock.get("clip_seconds")) or 0.0
        if clip_seconds <= 0.0:
            sample_count = _numeric(clock.get("sample_count"))
            if sample_count is not None:
                clip_seconds = sample_count / sample_rate
        audio_path = row.get("audio_path")
        if not isinstance(audio_path, str) and isinstance(input_block, Mapping):
            audio_path = input_block.get("audio_path")
        if not isinstance(audio_path, str) or not Path(audio_path).expanduser().is_file():
            skipped.append({"question_id": question_id, "reason": "missing_audio"})
            continue
        if _audio_channels(Path(audio_path).expanduser().resolve()) != 2:
            skipped.append({"question_id": question_id, "reason": "audio_not_stereo_2ch"})
            continue
        window = _find_event_window(
            evidence,
            sample_rate_hz=sample_rate,
            clip_seconds=clip_seconds,
        )
        purpose = None
        if answer_type == "transcript_wer":
            purpose = "speech_content"
            if window is None:
                skipped.append({"question_id": question_id, "reason": "missing_event_window"})
                continue
        elif answer_type == "time_s":
            purpose = "first_speech_onset"
            if not isinstance(evidence, Mapping) or not any(
                key in evidence for key in ("first_event", "event", "anchor_event")
            ):
                skipped.append({"question_id": question_id, "reason": "missing_first_event"})
                continue
            if clip_seconds <= 0.0:
                skipped.append({"question_id": question_id, "reason": "missing_clip_clock"})
                continue
            window = (0.0, clip_seconds)
        else:
            skipped.append({"question_id": question_id, "reason": "spatial_or_non_speech_question"})
            continue
        items.append(
            {
                "id": question_id,
                "audio_path": str(Path(audio_path).expanduser().resolve()),
                "window_seconds": [window[0], window[1]],
                "purpose": purpose,
            }
        )
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise _fail(f"refusing to overwrite Whisper request: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "schema": "avengine_whisper_review_request_v1",
        "status": "ready" if items else "not_run",
        "research_only": True,
        "model_path": str(model),
        "device": device,
        "decoding": dict(
            decoding
            or {
                "language": "en",
                "task": "transcribe",
                "fp16": False,
                "verbose": False,
            }
        ),
        "asr_input": "stereo source decoded by Whisper SDK to mono 16 kHz",
        "items": items,
        "skipped_questions": skipped,
    }
    output.write_text(
        json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema": "avengine_whisper_review_request_manifest_v1",
        "status": request["status"],
        "request_path": str(output),
        "model_path": str(model),
        "counts": {
            "source_questions": len(input_rows),
            "request_items": len(items),
            "skipped_questions": len(skipped),
        },
        "skipped_questions": skipped,
        "spatial_questions_not_run": True,
    }


def build_spatial_omni_benchmark_command(
    *,
    runtime_prefix: str | Path,
    checkpoint_path: str | Path,
    qa_root: str | Path,
    output_dir: str | Path,
    device: str = "cuda:2",
    max_samples: int | None = None,
    python_executable: str | Path | None = None,
) -> list[str]:
    """Return the exact local benchmark command without executing it.

    The installed model source is a runtime prefix rather than a Python
    environment. Callers therefore pass the environment that owns torch and
    transformers explicitly when the prefix has no bin/python.
    """

    runtime = Path(runtime_prefix).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    qa = Path(qa_root).expanduser().resolve()
    if not runtime.is_dir():
        raise _fail(f"runtime prefix is missing: {runtime}")
    if not checkpoint.is_file():
        raise _fail(f"checkpoint is missing: {checkpoint}")
    if not qa.is_dir():
        raise _fail(f"QA root is missing: {qa}")
    script = runtime / "scripts" / "batch_bench_so_qa.py"
    if not script.is_file():
        raise _fail(f"benchmark entry is missing: {script}")
    if python_executable is not None:
        python = Path(python_executable).expanduser().resolve()
        if not python.is_file() or not os.access(python, os.X_OK):
            raise _fail(f"python executable is missing or not executable: {python}")
    else:
        python = runtime / "bin" / "python"
        if not python.is_file():
            env_python = os.environ.get("AVENGINE_SPATIAL_OMNI_PYTHON")
            if env_python:
                python = Path(env_python).expanduser().resolve()
            if not python.is_file() or not os.access(python, os.X_OK):
                raise _fail(
                    "runtime prefix has no Python executable; pass "
                    "python_executable or set AVENGINE_SPATIAL_OMNI_PYTHON"
                )
    command = [
        str(python),
        str(script),
        "--checkpoint-path",
        str(checkpoint),
        "--qa-root",
        str(qa),
        "--split",
        "test",
        "--output-dir",
        str(Path(output_dir).expanduser().resolve()),
        "--device",
        device,
        "--batch-size",
        "1",
    ]
    if max_samples is not None:
        if (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, int)
            or max_samples < 1
        ):
            raise _fail("max_samples must be a positive integer")
        command.extend(["--max-samples", str(max_samples)])
    return command


__all__ = [
    "SPATIAL_OMNI_ADAPTER_SCHEMA",
    "ModelEvaluationAdapterError",
    "build_spatial_omni_benchmark_command",
    "prepare_qwen25_omni_pilot_gold",
    "prepare_qwen25_omni_pilot_inputs",
    "prepare_whisper_review_request",
    "prepare_spatial_omni_qa_root",
]

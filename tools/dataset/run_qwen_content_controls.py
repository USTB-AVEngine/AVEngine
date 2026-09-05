#!/usr/bin/env python3
"""Run a resumable Qwen2.5-Omni content/visual control on sanitized Episode inputs.

The runner never reads the gold file.  It accepts only the answer-free model
input document produced for the pilot and writes raw predictions as JSONL.
Scoring is intentionally a separate process.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are taking a multiple-choice test about a short indoor recording. "
    "Use only the supplied media. Return exactly one capital letter and "
    "nothing else."
)
_INTERNAL_ACTOR_ID = re.compile(r"(?<![A-Za-z0-9])source[0-9]+(?![A-Za-z0-9])", re.IGNORECASE)


FORBIDDEN_BLIND_FIELDS = {
    "answer",
    "answer_index",
    "answer_value",
    "episode_id",
    "type_id",
    "evidence",
    "fact_path",
    "source_question_id",
    "selection_bucket",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--setting",
        choices=("full_av", "audio_only", "video_only", "text_only", "dual_mono"),
        required=True,
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-pixels", type=int, default=200_704)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def build_question_text(question: dict[str, Any]) -> str:
    labels = "ABCD"
    options = question["options"]
    if not 2 <= len(options) <= 4:
        raise RuntimeError(
            f"{question['question_id']}: expected 2-4 options, got {len(options)}"
        )
    lines = [f"Question: {question['question_en']}", "Options:"]
    lines.extend(f"{labels[index]}. {option}" for index, option in enumerate(options))
    lines.extend(
        [
            "Left and right refer to the listener's own left and right in the recording.",
            "Answer:",
        ]
    )
    return "\n".join(lines)


def build_conversation(
    question: dict[str, Any], setting: str, *, fps: float, max_pixels: int
) -> tuple[list[dict[str, Any]], bool]:
    content: list[dict[str, Any]] = []
    media = question["media"]
    if setting == "full_av":
        content.append(
            {
                "type": "video",
                "video": media["full_av"],
                "fps": fps,
                "max_pixels": max_pixels,
            }
        )
        use_audio_in_video = True
    elif setting == "audio_only":
        content.append({"type": "audio", "audio": media["audio_only"]})
        use_audio_in_video = False
    elif setting == "video_only":
        content.append(
            {
                "type": "video",
                "video": media["video_only"],
                "fps": fps,
                "max_pixels": max_pixels,
            }
        )
        use_audio_in_video = False
    elif setting == "dual_mono":
        # The video-only hardlink has no AAC stream. The two mono files are
        # separate L/R views of one stereo master and share the same window.
        channels = media.get("dual_mono")
        if not isinstance(channels, list) or len(channels) != 2:
            raise RuntimeError("dual_mono media must contain left and right paths")
        content.extend(
            [
                {
                    "type": "video",
                    "video": media["video_only"],
                    "fps": fps,
                    "max_pixels": max_pixels,
                },
                {
                    "type": "text",
                    "text": "The next audio item is the listener's left channel.",
                },
                {"type": "audio", "audio": channels[0]},
                {
                    "type": "text",
                    "text": "The next audio item is the listener's right channel.",
                },
                {"type": "audio", "audio": channels[1]},
            ]
        )
        use_audio_in_video = False
    else:
        use_audio_in_video = False
    content.append({"type": "text", "text": build_question_text(question)})
    return (
        [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ],
        use_audio_in_video,
    )


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def parse_answer(raw: str, options: list[str]) -> tuple[str, str | None, int | None]:
    text = raw.strip()
    exact = re.fullmatch(r"\s*\(?([A-Da-d])\)?[.。:]?\s*", text)
    if exact:
        letter = exact.group(1).upper()
        index = ord(letter) - ord("A")
        if index < len(options):
            return "parsed_letter", letter, index

    explicit = re.findall(
        r"(?:answer|choice|option|答案|选项)\s*(?:is|为|是|:)?\s*\(?([A-Da-d])\)?",
        text,
        flags=re.IGNORECASE,
    )
    valid_explicit = {
        value.upper()
        for value in explicit
        if ord(value.upper()) - ord("A") < len(options)
    }
    if len(valid_explicit) == 1:
        letter = next(iter(valid_explicit))
        return "parsed_explicit_letter", letter, ord(letter) - ord("A")

    normalized = normalize_text(text)
    option_matches = [
        index for index, option in enumerate(options) if normalized == normalize_text(option)
    ]
    if len(option_matches) == 1:
        index = option_matches[0]
        return "parsed_option_text", chr(ord("A") + index), index

    standalone = {
        value.upper()
        for value in re.findall(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", text)
        if ord(value.upper()) - ord("A") < len(options)
    }
    if len(standalone) == 1:
        letter = next(iter(standalone))
        return "parsed_unique_letter", letter, ord(letter) - ord("A")
    return "parse_invalid", None, None


def item_shapes(items: Any) -> list[list[int]]:
    if items is None:
        return []
    return [list(getattr(item, "shape", ())) for item in items]


def assert_answer_free(value: Any, location: str = "model_inputs") -> None:
    if isinstance(value, str) and _INTERNAL_ACTOR_ID.search(value):
        raise RuntimeError(
            f"blind input exposes an internal actor identifier at {location}"
        )
    if isinstance(value, dict):
        leaked = sorted(FORBIDDEN_BLIND_FIELDS & value.keys())
        if leaked:
            raise RuntimeError(f"blind input leaks forbidden fields {leaked} at {location}")
        for key, child in value.items():
            assert_answer_free(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_answer_free(child, f"{location}[{index}]")


def model_questions(
    document: dict[str, Any], inputs_path: Path, setting: str
) -> list[dict[str, Any]]:
    assert_answer_free(document)
    schema = document.get("schema")
    if schema == "avengine_pilot_model_inputs_v0":
        questions = []
        for item in document["questions"]:
            normalized = dict(item)
            normalized["input_id"] = f"{item['question_id']}__{setting}"
            normalized["sample_id"] = item["question_id"]
            questions.append(normalized)
        return questions
    if schema != "avengine_pilot_model_inputs_v1":
        raise RuntimeError(f"unsupported model-input schema: {schema}")

    source_condition = (
        "full_av" if setting == "text_only"
        else "video_only" if setting == "dual_mono"
        else setting
    )
    selected = [
        item for item in document["items"] if item["condition"] == source_condition
    ]
    audio_items = {
        str(item.get("sample_id")): item
        for item in document["items"]
        if item.get("condition") == "audio_only"
    }
    questions = []
    input_root = inputs_path.parent.resolve()
    seen_samples: set[str] = set()
    for item_number, item in enumerate(selected, start=1):
        sample_id = str(item.get("sample_id", ""))
        if not sample_id or sample_id in seen_samples:
            raise RuntimeError(f"item {item_number}: invalid or duplicate sample_id")
        seen_samples.add(sample_id)
        if item.get("input_id") != f"{sample_id}__{source_condition}":
            raise RuntimeError(f"item {item_number}: input_id/sample_id mismatch")
        relative_media = Path(str(item.get("media_path", "")))
        if relative_media.is_absolute():
            raise RuntimeError(f"item {item_number}: media_path must be relative")
        media_path = (input_root / relative_media).resolve()
        try:
            media_path.relative_to(input_root)
        except ValueError as exc:
            raise RuntimeError(f"item {item_number}: media_path escapes input root") from exc
        if not media_path.is_file() or media_path.stat().st_size <= 0:
            raise RuntimeError(f"item {item_number}: missing or empty media")
        options = item.get("options")
        if not isinstance(options, list) or not 2 <= len(options) <= 4:
            raise RuntimeError(f"item {item_number}: expected 2-4 options")
        labels = []
        for option_index, option in enumerate(options):
            expected_letter = chr(ord("A") + option_index)
            if (
                not isinstance(option, dict)
                or option.get("index") != option_index
                or option.get("letter") != expected_letter
                or not str(option.get("label_en", "")).strip()
            ):
                raise RuntimeError(f"item {item_number}: invalid option contract")
            labels.append(str(option["label_en"]).strip())
        media = {source_condition: str(media_path)}
        if setting == "dual_mono":
            audio_item = audio_items.get(sample_id)
            if audio_item is None:
                raise RuntimeError(
                    f"item {item_number}: dual_mono has no audio_only sibling"
                )
            audio_relative = Path(str(audio_item.get("media_path", "")))
            if audio_relative.is_absolute():
                raise RuntimeError(
                    f"item {item_number}: audio_only media_path must be relative"
                )
            audio_path = (input_root / audio_relative).resolve()
            try:
                audio_path.relative_to(input_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"item {item_number}: audio_only media_path escapes input root"
                ) from exc
            if not audio_path.is_file() or audio_path.stat().st_size <= 0:
                raise RuntimeError(
                    f"item {item_number}: missing or empty audio_only media"
                )
            media = {
                "video_only": str(media_path),
                "audio_only": str(audio_path),
            }
        questions.append(
            {
                "question_id": sample_id,
                "sample_id": sample_id,
                "input_id": f"{sample_id}__{setting}",
                "question_en": item["question_en"],
                "options": labels,
                "media": media,
            }
        )
    return questions



def split_stereo_channels(
    audio_path: Path,
    *,
    media_root: Path,
    cache: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    """Split one 16 kHz stereo master once into labeled mono files."""

    import numpy as np
    import soundfile as sf

    source = audio_path.resolve()
    stat = source.stat()
    key = (int(stat.st_dev), int(stat.st_ino))
    if key in cache:
        return cache[key]
    samples, sample_rate = sf.read(
        str(source), dtype="float32", always_2d=True
    )
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise RuntimeError(
            f"dual_mono requires a stereo source, got shape {samples.shape}: {source}"
        )
    if int(sample_rate) != 16_000:
        raise RuntimeError(
            f"dual_mono requires a 16 kHz source, got {sample_rate}: {source}"
        )
    media_root.mkdir(parents=True, exist_ok=True)
    stem = f"pair-{len(cache):05d}"
    left_path = media_root / f"{stem}-left.wav"
    right_path = media_root / f"{stem}-right.wav"
    if left_path.exists() or right_path.exists():
        raise RuntimeError(
            f"refusing to overwrite dual_mono split media: {left_path}"
        )
    sf.write(str(left_path), samples[:, 0], int(sample_rate), subtype="PCM_16")
    sf.write(str(right_path), samples[:, 1], int(sample_rate), subtype="PCM_16")
    left_readback, left_rate = sf.read(
        str(left_path), dtype="float32", always_2d=False
    )
    right_readback, right_rate = sf.read(
        str(right_path), dtype="float32", always_2d=False
    )
    if int(left_rate) != 16_000 or int(right_rate) != 16_000:
        raise RuntimeError("dual_mono channel split readback changed sample rate")
    if len(left_readback) != len(samples) or len(right_readback) != len(samples):
        raise RuntimeError("dual_mono channel split readback changed sample count")
    record = {
        "source_audio": str(source),
        "channel_labels": ["left", "right"],
        "channel_paths": [str(left_path), str(right_path)],
        "source_channels": 2,
        "source_sample_rate_hz": int(sample_rate),
        "source_sample_count": int(len(samples)),
        "split_sample_count": [int(len(left_readback)), int(len(right_readback))],
        "split_channel_max_abs_error": [
            float(np.max(np.abs(left_readback - samples[:, 0]))),
            float(np.max(np.abs(right_readback - samples[:, 1]))),
        ],
        "spatial_conclusion_allowed": False,
    }
    cache[key] = record
    return record



def verify_dual_mono_processor(
    audios: Any,
    inputs: Any,
    info: dict[str, Any],
) -> dict[str, Any]:
    """Read back L/R arrays and processor features before model forward."""

    import numpy as np
    import soundfile as sf

    if not isinstance(audios, list) or len(audios) != 2:
        raise RuntimeError(
            f"dual_mono processor returned {len(audios or [])} audio items, expected 2"
        )
    expected_arrays = []
    expected_rates = []
    for path in info["channel_paths"]:
        array, rate = sf.read(path, dtype="float32", always_2d=False)
        expected_arrays.append(array)
        expected_rates.append(int(rate))
    channel_errors = []
    for index, (expected, observed) in enumerate(zip(expected_arrays, audios)):
        observed_array = np.asarray(observed)
        if observed_array.ndim != 1 or observed_array.shape != expected.shape:
            raise RuntimeError(
                f"dual_mono channel {index} shape changed: "
                f"expected {expected.shape}, got {observed_array.shape}"
            )
        channel_errors.append(float(np.max(np.abs(expected - observed_array))))
    feature_shape = list(getattr(inputs, "input_features").shape)
    mask_shape = list(getattr(inputs, "feature_attention_mask").shape)
    feature_count = int(feature_shape[0]) if feature_shape else 0
    if feature_count != 2:
        raise RuntimeError(
            f"dual_mono processor returned {feature_count} input feature items"
        )
    return {
        "source_audio": info["source_audio"],
        "channel_labels": ["left", "right"],
        "channel_paths": list(info["channel_paths"]),
        "source_channels": 2,
        "source_sample_rate_hz": info["source_sample_rate_hz"],
        "channel_sample_rates_hz": expected_rates,
        "audio_item_count": len(audios),
        "audio_item_shapes": item_shapes(audios),
        "channel_readback_max_abs_error": channel_errors,
        "input_features_shape": feature_shape,
        "feature_attention_mask_shape": mask_shape,
        "input_feature_item_count": feature_count,
        "audio_downmix_observed": False,
        "spatial_conclusion_allowed": False,
    }


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            completed.add(record["question_id"])
    return completed


def main() -> int:
    args = parse_args()
    if args.start_index < 0 or (args.limit is not None and args.limit < 1):
        raise RuntimeError("--start-index must be nonnegative and --limit must be positive")

    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    questions = model_questions(document, args.inputs.resolve(), args.setting)[
        args.start_index :
    ]
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        raise RuntimeError("selection contains no questions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise RuntimeError(f"output exists; pass --resume to continue: {args.output}")
    completed = load_completed(args.output) if args.resume else set()
    pending = [q for q in questions if q["question_id"] not in completed]
    dual_cache: dict[tuple[int, int], dict[str, Any]] = {}
    if args.setting == "dual_mono":
        dual_media_root = args.output.parent / "dual_mono_media"
        if dual_media_root.exists() and not args.resume:
            raise RuntimeError(
                f"dual_mono media exists; choose a fresh output parent: {dual_media_root}"
            )
        for question in pending:
            audio_path = Path(question["media"]["audio_only"])
            info = split_stereo_channels(
                audio_path, media_root=dual_media_root, cache=dual_cache
            )
            question["media"]["dual_mono"] = list(info["channel_paths"])
            question["_dual_mono_info"] = info

    import torch
    import transformers

    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this baseline")

    started_at = utc_now()
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta: dict[str, Any] = {
        "schema": "avengine_pilot_baseline_run_v0",
        "dataset_status": "pilot_only_not_benchmark_release",
        "run_status": "preliminary_unreviewed_running",
        "human_review_status": "pending",
        "eligible_for_paper_table": False,
        "qualification_claim": False,
        "runner": "qwen2_5_omni_content_control",
        "evaluation_form": "mcq",
        "setting": args.setting,
        "model_path": str(args.model),
        "inputs_path": str(args.inputs),
        "predictions_path": str(args.output),
        "selected_question_count": len(questions),
        "already_completed_count": len(completed & {q['question_id'] for q in questions}),
        "pending_at_start_count": len(pending),
        "seed": args.seed,
        "fps": args.fps,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": args.attn_implementation,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "video_reader": os.environ.get("FORCE_QWENVL_VIDEO_READER"),
        "dual_mono_research_adapter": args.setting == "dual_mono",
        "spatial_conclusion_allowed": False,
        "started_at": started_at,
    }
    atomic_write_json(meta_path, meta)

    load_started = time.perf_counter()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.disable_talker()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(
        args.model, local_files_only=True
    )
    meta["model_load_seconds"] = round(time.perf_counter() - load_started, 3)
    meta["run_status"] = "preliminary_unreviewed_inference"
    atomic_write_json(meta_path, meta)

    output_mode = "a" if args.output.exists() else "w"
    with args.output.open(output_mode, encoding="utf-8", buffering=1) as sink:
        for ordinal, question in enumerate(pending, start=1):
            conversation, use_audio_in_video = build_conversation(
                question,
                args.setting,
                fps=args.fps,
                max_pixels=args.max_pixels,
            )
            record: dict[str, Any] = {
                "schema": "avengine_pilot_prediction_v0",
                "input_id": question["input_id"],
                "sample_id": question["sample_id"],
                "question_id": question["question_id"],
                "condition": args.setting,
                "setting": args.setting,
                "evaluation_form": "mcq",
                "run_status": "preliminary_unreviewed",
                "attempt_count": 0,
            }
            if args.setting == "dual_mono":
                record["dual_mono_source"] = question.get("_dual_mono_info")
                record["dual_mono_video_path"] = question["media"]["video_only"]
            last_error: Exception | None = None
            for attempt in range(1, args.max_retries + 2):
                record["attempt_count"] = attempt
                try:
                    inference_started = time.perf_counter()
                    prompt = processor.apply_chat_template(
                        conversation, add_generation_prompt=True, tokenize=False
                    )
                    audios, images, videos = process_mm_info(
                        conversation, use_audio_in_video=use_audio_in_video
                    )
                    record["preprocessed_audio_shapes"] = item_shapes(audios)
                    record["preprocessed_video_shapes"] = item_shapes(videos)
                    inputs = processor(
                        text=prompt,
                        audio=audios,
                        images=images,
                        videos=videos,
                        return_tensors="pt",
                        padding=True,
                        use_audio_in_video=use_audio_in_video,
                    )
                    if args.setting == "dual_mono":
                        record["dual_mono_readback"] = verify_dual_mono_processor(
                            audios, inputs, question["_dual_mono_info"]
                        )
                    inputs = inputs.to(model.device).to(model.dtype)
                    with torch.inference_mode():
                        text_ids = model.generate(
                            **inputs,
                            use_audio_in_video=use_audio_in_video,
                            return_audio=False,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                        )
                    torch.cuda.synchronize()
                    generated_ids = [
                        output_ids[len(input_ids) :]
                        for input_ids, output_ids in zip(inputs.input_ids, text_ids)
                    ]
                    raw_output = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    parse_status, letter, answer_index = parse_answer(
                        raw_output, question["options"]
                    )
                    record.update(
                        {
                            "inference_status": "ok",
                            "raw_output": raw_output,
                            "prediction": raw_output,
                            "parse_status": parse_status,
                            "parsed_letter": letter,
                            "parsed_answer_index": answer_index,
                            "predicted_index": answer_index,
                            "latency_seconds": round(
                                time.perf_counter() - inference_started, 3
                            ),
                            "use_audio_in_video": use_audio_in_video,
                        }
                    )
                    if args.setting == "dual_mono":
                        record["dual_mono_forward_status"] = "ok"
                    last_error = None
                    del inputs, text_ids, generated_ids, audios, images, videos
                    break
                except Exception as exc:  # keep the run resumable and auditable
                    last_error = exc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if last_error is not None:
                record.update(
                    {
                        "inference_status": "infra_error",
                        "raw_output": "",
                        "prediction": "",
                        "parse_status": "parse_invalid",
                        "parsed_letter": None,
                        "parsed_answer_index": None,
                        "predicted_index": None,
                        "error_type": type(last_error).__name__,
                        "error_message": str(last_error),
                    }
                )
                if args.setting == "dual_mono":
                    record["dual_mono_forward_status"] = "error"
                    record["dual_mono_forward_error"] = {
                        "type": type(last_error).__name__,
                        "message": str(last_error),
                    }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"[{ordinal}/{len(pending)}] {question['question_id']} "
                f"status={record['inference_status']} answer={record['parsed_letter']}",
                flush=True,
            )

    final_records = load_completed(args.output)
    selected_ids = {question["question_id"] for question in questions}
    meta.update(
        {
            "run_status": "preliminary_unreviewed_complete",
            "completed_selected_count": len(final_records & selected_ids),
            "finished_at": utc_now(),
        }
    )
    atomic_write_json(meta_path, meta)
    print(
        f"QWEN25_OMNI_CONTENT_CONTROL_OK setting={args.setting} "
        f"completed={len(final_records & selected_ids)}/{len(selected_ids)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

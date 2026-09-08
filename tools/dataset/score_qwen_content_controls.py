#!/usr/bin/env python3
"""Score Qwen2.5-Omni content/visual control predictions without hiding missing/invalid outputs.

Predictions are JSON Lines.  Each line identifies an input using ``input_id``
(recommended) or ``sample_id`` plus ``condition``.  Supply either a zero-based
integer ``answer_index`` or textual ``prediction``/``answer``.  Invalid parses,
missing predictions, and duplicate/unknown records never become correct answers.

This scorer fixes the result status to ``qwen_content_control_pre_review_unverified``.  It is a
plumbing and diagnostic score, not a benchmark-release result.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EVALUATION_STATUS = "qwen_content_control_pre_review_unverified"
DEFAULT_CONDITIONS = ("full_av", "video_only", "audio_only", "text_only", "dual_mono")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=DEFAULT_CONDITIONS,
        help="score only these conditions (default: all declared in gold)",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"input does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in {path}: {exc}") from exc


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.。,:：;；!！?？'\"")


def prediction_key(record: dict[str, Any]) -> tuple[str, str]:
    input_id = record.get("input_id")
    if input_id is not None:
        input_text = str(input_id)
        for condition in DEFAULT_CONDITIONS:
            suffix = f"__{condition}"
            if input_text.endswith(suffix):
                return input_text[: -len(suffix)], condition
        raise RuntimeError(f"input_id has no recognized condition suffix: {input_text}")
    if record.get("sample_id") is None or record.get("condition") is None:
        raise RuntimeError("prediction needs input_id or both sample_id and condition")
    return str(record["sample_id"]), str(record["condition"])


def load_predictions(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    malformed: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"input does not exist: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise RuntimeError("JSON value is not an object")
                key = prediction_key(record)
                if key in records:
                    raise RuntimeError(f"duplicate prediction for {key[0]} / {key[1]}")
                records[key] = record
            except (json.JSONDecodeError, RuntimeError) as exc:
                malformed.append({"line": line_number, "error": str(exc)})
    return records, malformed


def option_aliases(option: dict[str, Any]) -> set[str]:
    aliases = set()
    for field in ("value", "label_en", "label_zh"):
        if option.get(field) is not None:
            normalized = normalize_text(option[field])
            if normalized:
                aliases.add(normalized)
    return aliases


def parse_answer(record: dict[str, Any], options: list[dict[str, Any]]) -> tuple[int | None, str]:
    explicit = record.get("answer_index", record.get("predicted_index"))
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int):
            return None, "answer_index_not_integer"
        if not 0 <= explicit < len(options):
            return None, "answer_index_out_of_range"
        return explicit, "explicit_zero_based_index"

    raw = record.get("prediction", record.get("answer"))
    if raw is None:
        return None, "answer_missing"
    if isinstance(raw, (dict, list, bool)):
        return None, "answer_not_scalar"
    text = normalize_text(raw)
    if not text:
        return None, "answer_empty"

    exact_matches = [
        index for index, option in enumerate(options) if text in option_aliases(option)
    ]
    if len(exact_matches) == 1:
        return exact_matches[0], "exact_value_or_label"
    if len(exact_matches) > 1:
        return None, "ambiguous_exact_label"

    letter_match = re.fullmatch(
        r"(?:answer|option|choice|答案|选项)?\s*[:：]?\s*([a-z])", text
    )
    if letter_match:
        index = ord(letter_match.group(1)) - ord("a")
        if 0 <= index < len(options):
            return index, "option_letter"
        return None, "option_letter_out_of_range"

    ordinal_match = re.fullmatch(
        r"(?:option|choice|选项)\s*[:：]?\s*([1-9][0-9]*)", text
    )
    if ordinal_match:
        index = int(ordinal_match.group(1)) - 1
        if 0 <= index < len(options):
            return index, "one_based_option_number"
        return None, "option_number_out_of_range"
    return None, "unrecognized_answer"


def empty_metric() -> dict[str, int]:
    return {"expected": 0, "received": 0, "parse_valid": 0, "correct": 0}


def finalize_metric(metric: dict[str, int]) -> dict[str, Any]:
    expected = metric["expected"]
    received = metric["received"]
    valid = metric["parse_valid"]
    correct = metric["correct"]
    return {
        **metric,
        "missing": expected - received,
        "parse_invalid": received - valid,
        "coverage": received / expected if expected else None,
        "parse_valid_rate": valid / received if received else None,
        "accuracy": correct / expected if expected else None,
        "accuracy_on_received": correct / received if received else None,
        "accuracy_on_parse_valid": correct / valid if valid else None,
    }


def add_metric(
    metric: dict[str, int], received: bool, parsed_index: int | None, correct: bool
) -> None:
    metric["expected"] += 1
    if received:
        metric["received"] += 1
    if parsed_index is not None:
        metric["parse_valid"] += 1
    if correct:
        metric["correct"] += 1


def split_compound(type_id: str, value: str) -> dict[str, str] | None:
    if "_" not in value:
        return None
    first, second = value.rsplit("_", 1)
    if type_id == "Q-COMPOUND-COUNT-MOTION":
        return {"count": first, "motion": second}
    if type_id == "Q-COMPOUND-FIRST-SIDE":
        return {"first_source": first, "side": second}
    return None


def sorted_finalized(metrics: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {key: finalize_metric(metrics[key]) for key in sorted(metrics)}


def validate_gold(gold: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(gold, dict) or not isinstance(gold.get("items"), list):
        raise RuntimeError("gold must be an object with an items list")
    items = gold["items"]
    if not items:
        raise RuntimeError("gold has no items")
    conditions = gold.get("condition_order", list(DEFAULT_CONDITIONS))
    if not isinstance(conditions, list) or not conditions:
        raise RuntimeError("gold condition_order must be a non-empty list")
    unknown = sorted(set(conditions) - set(DEFAULT_CONDITIONS))
    if unknown:
        raise RuntimeError(f"gold declares unsupported conditions: {unknown}")
    seen: set[str] = set()
    for item in items:
        required = {
            "sample_id",
            "type_id",
            "required_modalities",
            "option_count",
            "options",
            "answer_index",
            "answer_value",
        }
        missing = sorted(required - item.keys())
        if missing:
            raise RuntimeError(f"gold item is missing fields: {missing}")
        sample_id = str(item["sample_id"])
        if sample_id in seen:
            raise RuntimeError(f"duplicate sample_id in gold: {sample_id}")
        seen.add(sample_id)
        options = item["options"]
        if len(options) != item["option_count"]:
            raise RuntimeError(f"{sample_id}: option_count does not match options")
        index = item["answer_index"]
        if not isinstance(index, int) or not 0 <= index < len(options):
            raise RuntimeError(f"{sample_id}: invalid answer_index")
        if str(options[index].get("value")) != str(item["answer_value"]):
            raise RuntimeError(f"{sample_id}: answer index/value mismatch")
    return items, [str(condition) for condition in conditions]


def main() -> int:
    args = parse_args()
    gold = load_json(args.gold)
    items, declared_conditions = validate_gold(gold)
    conditions = list(args.conditions) if args.conditions else declared_conditions
    if len(set(conditions)) != len(conditions):
        raise RuntimeError("--conditions contains a duplicate")

    predictions, malformed = load_predictions(args.predictions)
    gold_by_id = {str(item["sample_id"]): item for item in items}
    expected_keys = {
        (sample_id, condition)
        for sample_id in gold_by_id
        for condition in conditions
    }
    unexpected_keys = sorted(set(predictions) - expected_keys)

    overall = empty_metric()
    by_type: dict[str, dict[str, int]] = defaultdict(empty_metric)
    by_condition: dict[str, dict[str, int]] = defaultdict(empty_metric)
    by_required_modality: dict[str, dict[str, int]] = defaultdict(empty_metric)
    by_option_count: dict[str, dict[str, int]] = defaultdict(empty_metric)
    parse_reasons: Counter[str] = Counter()
    compound_metrics: dict[str, dict[str, dict[str, int]]] = {
        "Q-COMPOUND-COUNT-MOTION": defaultdict(empty_metric),
        "Q-COMPOUND-FIRST-SIDE": defaultdict(empty_metric),
    }
    compound_by_condition: dict[str, dict[str, dict[str, dict[str, int]]]] = {
        "Q-COMPOUND-COUNT-MOTION": defaultdict(lambda: defaultdict(empty_metric)),
        "Q-COMPOUND-FIRST-SIDE": defaultdict(lambda: defaultdict(empty_metric)),
    }
    errors: list[dict[str, Any]] = []

    for sample_id, condition in sorted(expected_keys):
        item = gold_by_id[sample_id]
        record = predictions.get((sample_id, condition))
        received = record is not None
        if record is None:
            parsed_index, parse_reason = None, "missing_prediction"
        else:
            parsed_index, parse_reason = parse_answer(record, item["options"])
        parse_reasons[parse_reason] += 1
        correct = parsed_index == item["answer_index"] if parsed_index is not None else False

        metric_groups = [
            overall,
            by_type[str(item["type_id"])],
            by_condition[condition],
            by_option_count[str(item["option_count"])],
        ]
        modality_key = "+".join(sorted(map(str, item["required_modalities"]))) or "none"
        metric_groups.append(by_required_modality[modality_key])
        for metric in metric_groups:
            add_metric(metric, received, parsed_index, correct)

        type_id = str(item["type_id"])
        if type_id in compound_metrics:
            gold_parts = split_compound(type_id, str(item["answer_value"]))
            predicted_parts = None
            if parsed_index is not None:
                predicted_value = str(item["options"][parsed_index]["value"])
                predicted_parts = split_compound(type_id, predicted_value)
            if gold_parts is None:
                raise RuntimeError(f"{sample_id}: invalid compound gold value")
            add_metric(
                compound_metrics[type_id]["exact"], received, parsed_index, correct
            )
            add_metric(
                compound_by_condition[type_id][condition]["exact"],
                received,
                parsed_index,
                correct,
            )
            for component, gold_value in gold_parts.items():
                component_correct = (
                    predicted_parts is not None
                    and predicted_parts.get(component) == gold_value
                )
                add_metric(
                    compound_metrics[type_id][component],
                    received,
                    parsed_index,
                    component_correct,
                )
                add_metric(
                    compound_by_condition[type_id][condition][component],
                    received,
                    parsed_index,
                    component_correct,
                )

        if not correct:
            errors.append(
                {
                    "input_id": f"{sample_id}__{condition}",
                    "status": (
                        "missing"
                        if not received
                        else "parse_invalid"
                        if parsed_index is None
                        else "incorrect"
                    ),
                    "parse_reason": parse_reason,
                    "predicted_index": parsed_index,
                    "gold_index": item["answer_index"],
                }
            )

    compound_summary: dict[str, Any] = {}
    for type_id, components in compound_metrics.items():
        compound_summary[type_id] = {
            "overall": sorted_finalized(components),
            "by_condition": {
                condition: sorted_finalized(component_metrics)
                for condition, component_metrics in sorted(
                    compound_by_condition[type_id].items()
                )
            },
        }

    by_type_summary = sorted_finalized(by_type)
    type_accuracies = [
        metric["accuracy"]
        for metric in by_type_summary.values()
        if metric["accuracy"] is not None
    ]
    answer_position_counts = Counter(int(item["answer_index"]) for item in items)
    reference_baselines = {
        "uniform_random_expected_accuracy": sum(
            1.0 / int(item["option_count"]) for item in items
        )
        / len(items),
        "always_position_accuracy": {
            chr(ord("A") + index): answer_position_counts[index] / len(items)
            for index in range(max(int(item["option_count"]) for item in items))
        },
    }

    summary = {
        "schema": "avengine_qwen_content_control_score_v1",
        "evaluation_status": EVALUATION_STATUS,
        "qualification_claim": False,
        "warning": (
            "Pre-review pilot result only. Do not report this as a benchmark score "
            "before human visual review and benchmark qualification."
        ),
        "evaluation_form": "mcq",
        "scored_conditions": conditions,
        "overall": finalize_metric(overall),
        "macro_accuracy_by_type": (
            sum(type_accuracies) / len(type_accuracies) if type_accuracies else None
        ),
        "reference_baselines": reference_baselines,
        "by_type": by_type_summary,
        "by_modality_condition": sorted_finalized(by_condition),
        "by_required_modality": sorted_finalized(by_required_modality),
        "by_option_count": sorted_finalized(by_option_count),
        "compound_decomposition": compound_summary,
        "parse_reason_counts": dict(sorted(parse_reasons.items())),
        "malformed_prediction_lines": malformed,
        "unexpected_prediction_count": len(unexpected_keys),
        "unexpected_prediction_keys": [
            {"sample_id": sample_id, "condition": condition}
            for sample_id, condition in unexpected_keys
        ],
        "error_count": len(errors),
        "errors": errors,
    }
    atomic_write_json(args.output.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

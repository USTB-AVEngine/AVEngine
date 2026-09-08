"""Answer-free evaluation-time option permutations for MCQ model controls.

The generator keeps semantic question identity stable while producing deterministic
cyclic rotations for model-facing evaluation requests. Gold answers and inverse
mappings live in a separate private sidecar; the public document is intentionally
small and recursively whitelistable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json
import re

PUBLIC_SCHEMA = "avengine_qa_evaluation_permutations_public_v1"
PRIVATE_SCHEMA = "avengine_qa_evaluation_permutations_private_v1"
REPORT_SCHEMA = "avengine_qa_evaluation_permutations_report_v1"
MANIFEST_SCHEMA = "avengine_qa_evaluation_permutations_manifest_v1"
SUPPORTED_OPTION_COUNTS = tuple(range(2, 7))
OPTION_LABELS = "ABCDEF"

_FORBIDDEN_PUBLIC_KEY_PARTS = (
    "answer",
    "truth",
    "gold",
    "profile",
    "condition_profile",
    "evidence",
    "fact",
    "mapping",
    "source_question_id",
    "selection_bucket",
)
_PUBLIC_TOP_KEYS = frozenset(
    {"schema", "items", "source_question_count", "permutation_count", "claim_boundary"}
)
_PUBLIC_ITEM_KEYS = frozenset(
    {
        "question_id",
        "permutation_id",
        "qa_id",
        "question_en",
        "question_zh",
        "options",
        "actual_model_prompt",
    }
)
_PUBLIC_OPTION_KEYS = frozenset({"index", "letter", "label_en", "label_zh"})


class EvaluationPermutationError(ValueError):
    """Raised when a question cannot be prepared without changing semantics."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationPermutationError(message)


def _json_key(value: Any) -> str:
    return str(value)


def _form_mcq(item: Mapping[str, Any]) -> Mapping[str, Any] | None:
    forms = item.get("forms")
    if isinstance(forms, Mapping) and isinstance(forms.get("mcq"), Mapping):
        return forms["mcq"]
    model_input = item.get("model_input")
    if isinstance(model_input, Mapping) and isinstance(model_input.get("mcq"), Mapping):
        return model_input["mcq"]
    if isinstance(item.get("mcq"), Mapping):
        return item["mcq"]
    return None


def _question_text(item: Mapping[str, Any], mcq: Mapping[str, Any]) -> tuple[str, str]:
    question = item.get("question")
    if isinstance(question, Mapping):
        question_en = str(question.get("en") or question.get("question_en") or "")
        question_zh = str(question.get("zh") or question.get("question_zh") or "")
    else:
        question_en = ""
        question_zh = ""
    question_en = str(
        mcq.get("question_en")
        or item.get("question_en")
        or question_en
        or item.get("prompt")
        or ""
    ).strip()
    question_zh = str(
        mcq.get("question_zh")
        or item.get("question_zh")
        or question_zh
        or question_en
    ).strip()
    _require(question_en, "MCQ item has no English question text")
    return question_en, question_zh


def _option_value(option: Any, index: int) -> dict[str, str]:
    if isinstance(option, Mapping):
        label_en = str(
            option.get("label_en")
            or option.get("label")
            or option.get("value")
            or option.get("option")
            or ""
        ).strip()
        label_zh = str(option.get("label_zh") or label_en).strip()
    else:
        label_en = str(option).strip()
        label_zh = label_en
    _require(label_en, f"MCQ option {index} has empty label")
    return {"label_en": label_en, "label_zh": label_zh}


def _options(mcq: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = mcq.get("options")
    _require(isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)), "MCQ options must be a sequence")
    result = [_option_value(value, index) for index, value in enumerate(raw)]
    _require(
        len(result) in SUPPORTED_OPTION_COUNTS,
        f"MCQ requires 2-6 options, got {len(result)}",
    )
    return result


def _gold_index(item: Mapping[str, Any], mcq: Mapping[str, Any], options: Sequence[Mapping[str, str]]) -> int:
    gold = mcq.get("gold")
    if isinstance(gold, Mapping) and gold.get("correct_index") is not None:
        value = gold["correct_index"]
        _require(isinstance(value, int) and not isinstance(value, bool), "gold correct_index must be an integer")
        index = value
    else:
        candidate = None
        if isinstance(gold, Mapping):
            candidate = gold.get("value")
        truth = item.get("truth")
        if isinstance(truth, Mapping):
            candidate = truth.get("mcq_value") or truth.get("value") or candidate
        if candidate is None:
            candidate = item.get("answer_value")
        index = -1
        for option_index, option in enumerate(options):
            if candidate in (option["label_en"], option["label_zh"]):
                index = option_index
                break
    _require(0 <= index < len(options), "MCQ item has no valid private gold index")
    return index


def format_model_prompt(question_en: str, options: Sequence[Mapping[str, str]]) -> str:
    _require(len(options) in SUPPORTED_OPTION_COUNTS, "model prompt requires 2-6 options")
    lines = [f"Question: {question_en}", "Options:"]
    lines.extend(
        f"{OPTION_LABELS[index]}. {option['label_en']}"
        for index, option in enumerate(options)
    )
    lines.extend(
        [
            "Left and right refer to the listener's own left and right in the recording.",
            "Answer:",
        ]
    )
    return "\n".join(lines)


def _qa_id(item: Mapping[str, Any]) -> str:
    value = item.get("qa_id") or item.get("catalog_id") or item.get("question_type")
    return str(value or "unknown_qa").strip() or "unknown_qa"


def _source_items(
    documents: Sequence[Mapping[str, Any]],
    *,
    items_per_source: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    for source_index, document in enumerate(documents):
        raw_items = document.get("items") if isinstance(document, Mapping) else None
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
            raise EvaluationPermutationError(f"source {source_index} has no items list")
        source_candidates: list[dict[str, Any]] = []
        for item_index, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, Mapping):
                skipped.append({"source_index": source_index, "item_index": item_index, "reason": "item_not_mapping"})
                continue
            qid = str(raw_item.get("question_id") or "").strip()
            mcq = _form_mcq(raw_item)
            if not qid or mcq is None or raw_item.get("status") not in (None, "pass"):
                skipped.append({"source_index": source_index, "item_index": item_index, "question_id": qid or None, "reason": "not_runnable_mcq"})
                continue
            try:
                options = _options(mcq)
                question_en, question_zh = _question_text(raw_item, mcq)
                gold_index = _gold_index(raw_item, mcq, options)
            except EvaluationPermutationError as exc:
                skipped.append({"source_index": source_index, "item_index": item_index, "question_id": qid, "reason": str(exc)})
                continue
            if qid in seen_question_ids:
                raise EvaluationPermutationError(f"duplicate semantic question_id: {qid}")
            source_candidates.append(
                {
                    "question_id": qid,
                    "qa_id": _qa_id(raw_item),
                    "question_en": question_en,
                    "question_zh": question_zh,
                    "options": options,
                    "gold_index": gold_index,
                    "source_index": source_index,
                    "source_item_index": item_index,
                }
            )
        if items_per_source is not None:
            _require(items_per_source > 0, "items_per_source must be positive")
            source_candidates = source_candidates[:items_per_source]
        for candidate in source_candidates:
            seen_question_ids.add(candidate["question_id"])
            selected.append(candidate)
    return selected, skipped


def _public_option(index: int, option: Mapping[str, str]) -> dict[str, Any]:
    return {
        "index": index,
        "letter": OPTION_LABELS[index],
        "label_en": option["label_en"],
        "label_zh": option["label_zh"],
    }


def _validate_public_value(value: Any, *, location: str = "public") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            lowered = key_text.casefold()
            _require(
                not any(part in lowered for part in _FORBIDDEN_PUBLIC_KEY_PARTS),
                f"public payload contains forbidden key {key_text!r} at {location}",
            )
            _validate_public_value(child, location=f"{location}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_public_value(child, location=f"{location}[{index}]")
    elif isinstance(value, str):
        lowered = value.casefold()
        _require(
            "profile" not in lowered and "truth" not in lowered,
            f"public payload contains forbidden text at {location}",
        )


def _required_text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{field} must be a non-empty string")
    return value


def _optional_text(value: Any, field: str) -> None:
    _require(value is None or isinstance(value, str), f"{field} must be a string when present")


def validate_public_document(document: Mapping[str, Any]) -> None:
    _require(document.get("schema") == PUBLIC_SCHEMA, "unexpected public permutation schema")
    _require(set(document) <= _PUBLIC_TOP_KEYS, "public permutation top-level whitelist violation")
    items = document.get("items")
    _require(isinstance(items, list), "public permutation items must be a list")
    seen_ids: set[str] = set()
    for item_index, item in enumerate(items):
        _require(isinstance(item, Mapping), f"public item {item_index} is not a mapping")
        _require(set(item) <= _PUBLIC_ITEM_KEYS, f"public item {item_index} whitelist violation")
        for required in ("question_id", "permutation_id", "question_en", "options", "actual_model_prompt"):
            _require(required in item, f"public item {item_index} lacks {required}")
        question_id = _required_text(item["question_id"], f"public item {item_index}.question_id")
        permutation_id = _required_text(item["permutation_id"], f"public item {item_index}.permutation_id")
        _required_text(item["question_en"], f"public item {item_index}.question_en")
        _required_text(item["actual_model_prompt"], f"public item {item_index}.actual_model_prompt")
        _optional_text(item.get("question_zh"), f"public item {item_index}.question_zh")
        _optional_text(item.get("qa_id"), f"public item {item_index}.qa_id")
        _require(permutation_id not in seen_ids, f"duplicate public permutation_id: {permutation_id}")
        seen_ids.add(permutation_id)
        options = item["options"]
        _require(isinstance(options, list) and 2 <= len(options) <= 6, f"public item {item_index} has invalid options")
        for option_index, option in enumerate(options):
            _require(isinstance(option, Mapping), f"public item {item_index} option is not a mapping")
            _require(set(option) <= _PUBLIC_OPTION_KEYS, f"public item {item_index} option whitelist violation")
            _require(option.get("index") == option_index and not isinstance(option.get("index"), bool), f"public item {item_index} option index drift")
            _require(option.get("letter") == OPTION_LABELS[option_index], f"public item {item_index} option letter drift")
            _required_text(option.get("label_en"), f"public item {item_index} option label_en")
            _optional_text(option.get("label_zh"), f"public item {item_index} option label_zh")
        _require(question_id, f"public item {item_index}.question_id must not be empty")
    _validate_public_value(document)


def _private_lookup(
    private_items: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]]:
    if private_items is None:
        return {}
    if isinstance(private_items, Mapping):
        if isinstance(private_items.get("items"), list):
            return _private_lookup(private_items["items"])
        return {str(key): value for key, value in private_items.items()}
    result: dict[str, Mapping[str, Any]] = {}
    for item in private_items:
        if isinstance(item, Mapping) and item.get("permutation_id") is not None:
            result[str(item["permutation_id"])] = item
    return result


def consistency_summary(
    predictions: Iterable[Mapping[str, Any]] | None,
    private_items: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    prediction_list = list(predictions or [])
    if not prediction_list:
        return {
            "status": "unmeasured",
            "model_outputs_present": False,
            "semantic_question_count": 0,
            "consistent_semantic_question_count": None,
            "inconsistent_semantic_question_count": None,
            "partial_semantic_question_count": None,
            "consistency_rate": None,
            "evaluated_accuracy": None,
            "reason": "No model outputs were supplied; no evaluation was fabricated.",
        }
    private_lookup = _private_lookup(private_items)
    expected_by_question: dict[str, set[str]] = defaultdict(set)
    for permutation_id, private in private_lookup.items():
        question_id = str(private.get("question_id") or "")
        if question_id:
            expected_by_question.setdefault(question_id, set()).add(permutation_id)
    observed_by_question: dict[str, dict[str, int]] = defaultdict(dict)
    duplicate_count = 0
    invalid_prediction_count = 0
    for record in prediction_list:
        semantic_id = str(record.get("semantic_question_id") or record.get("question_id") or "")
        record_question_id = record.get("question_id")
        permutation_id = str(record.get("permutation_id") or "")
        if record.get("semantic_question_id") is not None and record_question_id is not None:
            if str(record_question_id) != semantic_id:
                raise EvaluationPermutationError(
                    f"prediction question_id disagrees with semantic_question_id: {record_question_id!r} != {semantic_id!r}"
                )
        private = private_lookup.get(permutation_id)
        if private is None or not semantic_id or not permutation_id:
            invalid_prediction_count += 1
            continue
        if str(private.get("question_id") or "") != semantic_id:
            raise EvaluationPermutationError(
                f"prediction {permutation_id} question_id disagrees with private mapping"
            )
        predicted = record.get("parsed_answer_index")
        if predicted is None:
            predicted = record.get("predicted_index")
        if isinstance(predicted, bool) or not isinstance(predicted, int):
            invalid_prediction_count += 1
            continue
        inverse = private.get("permuted_to_original")
        if not isinstance(inverse, list) or not 0 <= predicted < len(inverse):
            invalid_prediction_count += 1
            continue
        if permutation_id in observed_by_question[semantic_id]:
            duplicate_count += 1
            continue
        observed_by_question[semantic_id][permutation_id] = int(inverse[predicted])
    per_question: dict[str, Any] = {}
    consistent_count = 0
    inconsistent_count = 0
    partial_count = 0
    for question_id in sorted(set(expected_by_question) | set(observed_by_question)):
        expected = expected_by_question.get(question_id, set())
        observed = set(observed_by_question.get(question_id, {}))
        missing = sorted(expected - observed)
        values = set(observed_by_question.get(question_id, {}).values())
        if len(observed) < 2:
            status = "unmeasured" if not observed else "partial"
            if status == "partial":
                partial_count += 1
        elif len(values) == 1:
            status = "consistent"
            consistent_count += 1
        else:
            status = "inconsistent"
            inconsistent_count += 1
        per_question[question_id] = {
            "status": status,
            "expected_permutation_count": len(expected),
            "observed_permutation_count": len(observed),
            "missing_permutation_ids": missing,
            "observed_permutation_ids": sorted(observed),
            "duplicate_prediction_count": sum(
                1
                for record in prediction_list
                if str(record.get("permutation_id") or "") in observed
                and str(record.get("semantic_question_id") or record.get("question_id") or "") == question_id
            ) - len(observed),
        }
    measured_count = consistent_count + inconsistent_count
    status = "measured" if measured_count and partial_count == 0 and not any(
        entry["missing_permutation_ids"] for entry in per_question.values()
    ) else ("partial" if observed_by_question else "unmeasured")
    return {
        "status": status,
        "model_outputs_present": bool(observed_by_question),
        "semantic_question_count": len(per_question),
        "consistent_semantic_question_count": consistent_count,
        "inconsistent_semantic_question_count": inconsistent_count,
        "partial_semantic_question_count": partial_count,
        "consistency_rate": (consistent_count / measured_count) if measured_count else None,
        "evaluated_accuracy": None,
        "expected_permutation_count": sum(len(values) for values in expected_by_question.values()),
        "observed_permutation_count": sum(len(values) for values in observed_by_question.values()),
        "missing_permutation_count": sum(
            len(expected_by_question.get(question_id, set()) - set(observed_by_question.get(question_id, {})))
            for question_id in expected_by_question
        ),
        "invalid_prediction_count": invalid_prediction_count,
        "duplicate_prediction_count": duplicate_count,
        "per_question": per_question,
        "reason": None if observed_by_question else "No parseable permutation predictions were supplied.",
    }

def build_evaluation_permutations(
    documents: Sequence[Mapping[str, Any]],
    *,
    items_per_source: int | None = None,
) -> dict[str, Any]:
    selected, skipped = _source_items(documents, items_per_source=items_per_source)
    public_items: list[dict[str, Any]] = []
    private_items: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    used_permutation_ids: set[str] = set()
    for source_item in selected:
        qid = source_item["question_id"]
        qa_id = source_item["qa_id"]
        original_options = source_item["options"]
        option_count = len(original_options)
        gold_original_index = int(source_item["gold_index"])
        for rotation_offset in range(option_count):
            permutation_id = f"{qid}__perm_k{option_count}_r{rotation_offset:02d}"
            _require(permutation_id not in used_permutation_ids, f"duplicate permutation_id: {permutation_id}")
            used_permutation_ids.add(permutation_id)
            # Left cyclic rotation: new position i takes original position i+offset.
            permuted_to_original = [
                (index + rotation_offset) % option_count for index in range(option_count)
            ]
            original_to_permuted = [0] * option_count
            for permuted_index, original_index in enumerate(permuted_to_original):
                original_to_permuted[original_index] = permuted_index
            permuted_options = [
                original_options[original_index] for original_index in permuted_to_original
            ]
            gold_permuted_index = original_to_permuted[gold_original_index]
            public_item = {
                "question_id": qid,
                "permutation_id": permutation_id,
                "qa_id": qa_id,
                "question_en": source_item["question_en"],
                "question_zh": source_item["question_zh"],
                "options": [
                    _public_option(index, option)
                    for index, option in enumerate(permuted_options)
                ],
                "actual_model_prompt": format_model_prompt(source_item["question_en"], permuted_options),
            }
            public_items.append(public_item)
            private_items.append(
                {
                    "question_id": qid,
                    "permutation_id": permutation_id,
                    "qa_id": qa_id,
                    "option_count": option_count,
                    "rotation_offset": rotation_offset,
                    "original_options": original_options,
                    "permuted_to_original": permuted_to_original,
                    "original_to_permuted": original_to_permuted,
                    "gold_original_index": gold_original_index,
                    "gold_permuted_index": gold_permuted_index,
                    "gold_original_letter": OPTION_LABELS[gold_original_index],
                    "gold_permuted_letter": OPTION_LABELS[gold_permuted_index],
                    "gold_moved": gold_original_index != gold_permuted_index,
                    "source_index": source_item["source_index"],
                    "source_item_index": source_item["source_item_index"],
                }
            )
            groups[(qa_id, option_count)].append(private_items[-1])
    public_document = {
        "schema": PUBLIC_SCHEMA,
        "source_question_count": len(selected),
        "permutation_count": len(public_items),
        "items": public_items,
        "claim_boundary": "Answer-free pre-evaluation option rotations; permutations do not create semantic questions or an evaluation result.",
    }
    validate_public_document(public_document)
    private_lookup = {item["permutation_id"]: item for item in private_items}
    distribution: dict[str, Any] = {}
    for (qa_id, option_count), records in sorted(groups.items()):
        counts = Counter(OPTION_LABELS[item["gold_permuted_index"]] for item in records)
        distribution[f"{qa_id}::K{option_count}"] = {
            "qa_id": qa_id,
            "option_count": option_count,
            "permutation_count": len(records),
            "gold_position_counts": dict(sorted(counts.items())),
            "moved_gold_count": sum(bool(item["gold_moved"]) for item in records),
            "unmoved_gold_count": sum(not item["gold_moved"] for item in records),
        }
    private_document = {
        "schema": PRIVATE_SCHEMA,
        "source_question_count": len(selected),
        "permutation_count": len(private_items),
        "items": private_items,
        "position_distribution": distribution,
        "consistency_summary": consistency_summary(None),
        "skipped": skipped,
        "claim_boundary": "Private evaluator-side option mappings and gold values; never pass this document to a model.",
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "ready_for_pre_evaluation",
        "source_question_count": len(selected),
        "permutation_count": len(public_items),
        "skipped_count": len(skipped),
        "position_distribution": distribution,
        "consistency_summary": consistency_summary(None),
        "semantic_question_ids": [item["question_id"] for item in selected],
        "permutation_ids_unique": len(used_permutation_ids) == len(public_items),
        "claim_boundary": "Position distribution is measured from precomputed mappings. Model consistency and accuracy are unmeasured until raw model outputs exist.",
    }
    return {
        "public": public_document,
        "private": private_document,
        "report": report,
        "private_items_by_id": private_lookup,
    }


def load_json_documents(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in paths:
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise EvaluationPermutationError(f"question source must be a JSON object: {source}")
        documents.append(value)
    return documents


__all__ = [
    "EvaluationPermutationError",
    "MANIFEST_SCHEMA",
    "PRIVATE_SCHEMA",
    "PUBLIC_SCHEMA",
    "REPORT_SCHEMA",
    "build_evaluation_permutations",
    "consistency_summary",
    "format_model_prompt",
    "load_json_documents",
    "validate_public_document",
]

"""Score predictions against the public IDs in a full QA catalog.

The catalog index owns the mapping from opaque public question IDs to the
private generated question items.  This scorer resolves that mapping using
each record's questions_path and public_question_ids, then delegates answer
semantics to the unified item scorer.  It does not read or export native
engine evidence.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from avengine.qa.unified_catalog import iter_unified_items
from avengine.qa.unified_scoring import UnifiedScoreError, score_unified_item


CATALOG_SCORE_SCHEMA = "avengine_qa_catalog_score_v1"
_FORMS = frozenset({"mcq", "open"})


class BindingCatalogScoreError(ValueError):
    """A full-catalog scoring input is malformed."""


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BindingCatalogScoreError(f"{field} must be a non-empty string")
    return value.strip()


def _read_json(path: Path, *, field: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise BindingCatalogScoreError(f"{field} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BindingCatalogScoreError(f"{field} is not valid JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise BindingCatalogScoreError(f"{field} must contain a JSON object: {path}")
    return value


def _catalog_document(
    catalog_index: Mapping[str, Any] | str | Path,
    *,
    input_base: str | Path | None,
) -> tuple[Mapping[str, Any], Path]:
    if isinstance(catalog_index, (str, Path)):
        path = Path(catalog_index).expanduser().resolve()
        document = _read_json(path, field="catalog index")
        return document, path.parent
    if not isinstance(catalog_index, Mapping):
        raise BindingCatalogScoreError(
            "catalog_index must be a JSON object or a catalog_index.json path"
        )
    base = (
        Path(input_base).expanduser().resolve()
        if input_base is not None
        else Path.cwd().resolve()
    )
    return catalog_index, base


def _question_items(
    document: Mapping[str, Any],
    *,
    input_base: Path,
    form: str,
) -> list[dict[str, Any]]:
    records = document.get("records")
    if not _is_sequence(records) or not records:
        raise BindingCatalogScoreError(
            "catalog index records must be a non-empty list"
        )

    entries: list[dict[str, Any]] = []
    public_ids: set[str] = set()
    for record_index, record in enumerate(records):
        prefix = f"records[{record_index}]"
        if not isinstance(record, Mapping):
            raise BindingCatalogScoreError(f"{prefix} must be an object")
        questions_path_value = _text(
            record.get("questions_path"),
            field=f"{prefix}.questions_path",
        )
        questions_path = Path(questions_path_value)
        if not questions_path.is_absolute():
            questions_path = input_base / questions_path
        questions_path = questions_path.resolve()
        question_set = _read_json(questions_path, field=f"{prefix}.questions_path")
        items = list(iter_unified_items(question_set))
        if not items:
            raise BindingCatalogScoreError(
                f"{prefix}.questions_path contains no question items"
            )

        raw_public_ids = record.get("public_question_ids")
        if not _is_sequence(raw_public_ids):
            raise BindingCatalogScoreError(
                f"{prefix}.public_question_ids must be a list"
            )
        if len(raw_public_ids) != len(items):
            raise BindingCatalogScoreError(
                f"{prefix}.public_question_ids must align one-to-one with "
                f"iter_unified_items order"
            )

        sample_id = record.get("sample_id")
        group_id = record.get("group_id")
        for item_index, (raw_public_id, item) in enumerate(
            zip(raw_public_ids, items)
        ):
            item_prefix = f"{prefix}.public_question_ids[{item_index}]"
            public_id = _text(raw_public_id, field=item_prefix)
            if public_id in public_ids:
                raise BindingCatalogScoreError(
                    f"duplicate public question_id: {public_id!r}"
                )
            public_ids.add(public_id)
            if not isinstance(item, Mapping):
                raise BindingCatalogScoreError(
                    f"{prefix}.iter_unified_items[{item_index}] must be an object"
                )
            if item.get("status") != "pass":
                raise BindingCatalogScoreError(
                    f"catalog item for {public_id!r} is not a valid candidate"
                )
            forms = item.get("forms")
            if not isinstance(forms, Mapping):
                raise BindingCatalogScoreError(
                    f"catalog item for {public_id!r} has no forms object"
                )
            if form in forms and not isinstance(forms[form], Mapping):
                raise BindingCatalogScoreError(
                    f"catalog item for {public_id!r} has an invalid {form} form"
                )
            form_available = form in forms
            form_status = item.get("form_status")
            if form_available and isinstance(form_status, Mapping):
                status_record = form_status.get(form)
                if (
                    isinstance(status_record, Mapping)
                    and status_record.get("status") not in (None, "pass")
                ):
                    form_available = False
            qa_id = _text(item.get("qa_id"), field=f"{public_id}.qa_id")
            entries.append(
                {
                    "question_id": public_id,
                    "qa_id": qa_id,
                    "item": item,
                    "form_available": form_available,
                    "sample_id": sample_id,
                    "group_id": group_id,
                }
            )
    return entries


def _prediction_rows(
    predictions: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> list[tuple[str, Any]]:
    if isinstance(predictions, Mapping):
        if "predictions" not in predictions:
            raise BindingCatalogScoreError(
                "predictions must be a list of {question_id, prediction} objects"
            )
        predictions = predictions["predictions"]
    if not _is_sequence(predictions):
        raise BindingCatalogScoreError(
            "predictions must be a list of {question_id, prediction} objects"
        )

    rows: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for index, prediction in enumerate(predictions):
        prefix = f"predictions[{index}]"
        if not isinstance(prediction, Mapping):
            raise BindingCatalogScoreError(f"{prefix} must be an object")
        question_id = _text(
            prediction.get("question_id"),
            field=f"{prefix}.question_id",
        )
        if question_id in seen:
            raise BindingCatalogScoreError(
                f"duplicate prediction question_id: {question_id!r}"
            )
        if "prediction" not in prediction:
            raise BindingCatalogScoreError(
                f"{prefix}.prediction is required"
            )
        seen.add(question_id)
        rows.append((question_id, prediction["prediction"]))
    return rows


def _score_record(
    entry: Mapping[str, Any],
    *,
    answer_present: bool,
    answer: Any,
    form: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    question_id = str(entry["question_id"])
    qa_id = str(entry["qa_id"])
    if not answer_present:
        return {
            "question_id": question_id,
            "qa_id": qa_id,
            "form": form,
            "status": "invalid",
            "reason": "model answer is missing",
            "error_code": "missing_answer",
            "score": 0.0,
            "answer_present": False,
            "missing_answer": True,
            "correct": False,
        }
    try:
        result = score_unified_item(
            entry["item"],
            answer,
            form=form,
            params=params,
        )
    except UnifiedScoreError as error:
        raise BindingCatalogScoreError(
            f"cannot score question {question_id!r}: {error}"
        ) from error
    scored = dict(result)
    # score_unified_item reports the private generated ID.  The catalog
    # scorer's output remains in the public ID namespace.
    scored["question_id"] = question_id
    scored["qa_id"] = qa_id
    scored["form"] = form
    scored["answer_present"] = True
    scored["missing_answer"] = False
    value = scored.get("score")
    scored["correct"] = (
        scored.get("status") == "scored"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    )
    return scored


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [
        record for record in records
        if record.get("status") != "unavailable_form"
    ]
    scored = [record for record in available if record.get("status") == "scored"]
    correct = [record for record in scored if record.get("correct") is True]
    total_score = sum(
        float(record.get("score", 0.0))
        for record in available
        if isinstance(record.get("score"), (int, float))
        and not isinstance(record.get("score"), bool)
        and math.isfinite(float(record["score"]))
    )
    scored_score = sum(float(record.get("score", 0.0)) for record in scored)
    denominator = len(available)
    return {
        "total": len(records),
        "denominator": denominator,
        "scored": len(scored),
        "correct": len(correct),
        "invalid": sum(
            record.get("status") == "invalid"
            and not bool(record.get("missing_answer"))
            for record in available
        ),
        "missing": sum(bool(record.get("missing_answer")) for record in available),
        "abstained": sum(record.get("status") == "abstained" for record in available),
        "unavailable_form": sum(
            record.get("status") == "unavailable_form" for record in records
        ),
        "accuracy": len(correct) / denominator if denominator else None,
        "mean_score_over_all": total_score / denominator if denominator else None,
        "mean_score_over_scored": scored_score / len(scored) if scored else None,
        "denominator_policy": (
            "all catalog items with the requested form; missing, invalid and "
            "abstained predictions count as zero-score failures"
        ),
    }


def score_binding_catalog(
    catalog_index: Mapping[str, Any] | str | Path,
    predictions: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    form: str = "open",
    params: Mapping[str, Any] | None = None,
    input_base: str | Path | None = None,
) -> dict[str, Any]:
    """Score a full catalog from public question IDs.

    catalog_index may be the decoded index object or its path.  When an
    object is supplied, relative questions_path values resolve under input_base
    or the current directory.
    """
    if form not in _FORMS:
        raise BindingCatalogScoreError(
            f"form must be one of {sorted(_FORMS)}"
        )
    if params is None:
        scorer_params: Mapping[str, Any] = {}
    elif isinstance(params, Mapping):
        scorer_params = params
    else:
        raise BindingCatalogScoreError("params must be an object")
    document, base = _catalog_document(catalog_index, input_base=input_base)
    entries = _question_items(document, input_base=base, form=form)
    prediction_rows = _prediction_rows(predictions)
    by_public_id = {entry["question_id"]: entry for entry in entries}
    prediction_map = dict(prediction_rows)

    unknown = sorted(set(prediction_map) - set(by_public_id))
    if unknown:
        raise BindingCatalogScoreError(
            f"unknown public question_id(s): {unknown!r}"
        )
    wrong_form = sorted(
        question_id
        for question_id in prediction_map
        if not by_public_id[question_id]["form_available"]
    )
    if wrong_form:
        raise BindingCatalogScoreError(
            f"prediction question_id(s) do not provide the {form} form: "
            f"{wrong_form!r}"
        )

    records: list[dict[str, Any]] = []
    for entry in entries:
        if not entry["form_available"]:
            records.append(
                {
                    "question_id": entry["question_id"],
                    "qa_id": entry["qa_id"],
                    "form": form,
                    "status": "unavailable_form",
                    "reason": f"question item has no {form} form",
                    "score": None,
                    "answer_present": False,
                    "missing_answer": False,
                    "correct": None,
                }
            )
            continue
        question_id = entry["question_id"]
        records.append(
            _score_record(
                entry,
                answer_present=question_id in prediction_map,
                answer=prediction_map.get(question_id),
                form=form,
                params=scorer_params,
            )
        )

    by_qa_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_qa_records[str(record["qa_id"])].append(record)
    qa_metrics = {
        qa_id: _metrics(rows)
        for qa_id, rows in by_qa_records.items()
    }
    overall = _metrics(records)
    return {
        "schema": CATALOG_SCORE_SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "form": form,
        "counts": {
            "catalog_items_total": overall["total"],
            "form_available": overall["denominator"],
            "form_unavailable": overall["unavailable_form"],
            "predictions_supplied": len(prediction_rows),
            "scored": overall["scored"],
            "correct": overall["correct"],
            "invalid": overall["invalid"],
            "missing": overall["missing"],
            "abstained": overall["abstained"],
        },
        "overall": overall,
        "qa_metrics": qa_metrics,
        "records": records,
        "claim_boundary": (
            "Scores compare supplied predictions with generated research-candidate "
            "question forms. The scorer does not run a model, train a model, "
            "certify human answerability or modality necessity, or admit a dataset."
        ),
    }


__all__ = [
    "BindingCatalogScoreError",
    "CATALOG_SCORE_SCHEMA",
    "score_binding_catalog",
]


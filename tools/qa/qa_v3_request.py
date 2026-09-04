"""Plan QA-v3 request budgets without hiding per-profile shortages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
import math
import json
from pathlib import Path
from typing import Any


PARAM_ITEMS_PER_ROOM = "ITEMS_PER_ROOM_DEFAULT"
PARAM_ANSWER_FORMS = "ANSWER_FORMS_DEFAULT"
ANSWER_FORM_ALIASES = {
    "mcq": "mcq",
    "equal_bands": "mcq",
    "open": "open",
    "open_degrees": "open",
}


class QARequestError(ValueError):
    """Raised when a request cannot be represented safely."""


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QARequestError(f"{name} must be a positive integer")
    return value


def normalize_answer_forms(values: object) -> list[str]:
    """Return canonical answer-form names while preserving request order.

    ``equal_bands`` and ``open_degrees`` are the names used by the older v10
    request files. They describe the same paired forms as ``mcq`` and
    ``open`` and therefore normalize to those canonical names.
    """

    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        raw_values = list(values)
    else:
        raise QARequestError("answer_forms must be a non-empty list of strings")
    if not raw_values:
        raise QARequestError("answer_forms must contain at least one form")

    normalized: list[str] = []
    for value in raw_values:
        if not isinstance(value, str) or not value.strip():
            raise QARequestError("answer_forms must contain non-empty strings")
        canonical = ANSWER_FORM_ALIASES.get(value.strip().lower())
        if canonical is None:
            allowed = sorted(ANSWER_FORM_ALIASES)
            raise QARequestError(
                f"unknown answer form {value!r}; expected one of {allowed}"
            )
        if canonical in normalized:
            raise QARequestError(f"duplicate answer form after normalization: {canonical!r}")
        normalized.append(canonical)
    return normalized


def _profile_ids(values: object) -> list[str]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise QARequestError("profile_ids must be a non-empty list of strings")
    if not values:
        raise QARequestError("profile_ids must contain at least one profile")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise QARequestError("profile_ids must contain non-empty strings")
        profile_id = value.strip()
        if profile_id in result:
            raise QARequestError(f"duplicate profile id: {profile_id!r}")
        result.append(profile_id)
    return result


def _weights(
    profiles: list[str], profile_weights: object | None
) -> list[float]:
    if profile_weights is None:
        values: list[object] = [1.0] * len(profiles)
    elif isinstance(profile_weights, Mapping):
        unknown = sorted(set(profile_weights) - set(profiles))
        missing = sorted(set(profiles) - set(profile_weights))
        if unknown or missing:
            raise QARequestError(
                "profile_weights must name every profile exactly once; "
                f"unknown={unknown}, missing={missing}"
            )
        values = [profile_weights[profile] for profile in profiles]
    elif isinstance(profile_weights, Sequence) and not isinstance(
        profile_weights, (str, bytes, bytearray)
    ):
        if len(profile_weights) != len(profiles):
            raise QARequestError("profile_weights list must match profile_ids length")
        values = list(profile_weights)
    else:
        raise QARequestError("profile_weights must be a mapping or aligned list")

    result: list[float] = []
    for profile, value in zip(profiles, values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise QARequestError(f"weight for {profile!r} must be a finite number")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise QARequestError(
                f"weight for {profile!r} must be finite and non-negative"
            )
        result.append(number)
    if not any(value > 0.0 for value in result):
        raise QARequestError("profile_weights must contain at least one positive weight")
    return result


def _largest_remainder(candidate_count: int, weights: list[float]) -> list[int]:
    # Fraction keeps finite float weights exact for the allocation arithmetic,
    # including inputs whose ordinary float sum overflows to infinity.
    exact_weights = [Fraction.from_float(weight) for weight in weights]
    total_weight = sum(exact_weights, Fraction(0, 1))
    exact = [Fraction(candidate_count) * weight / total_weight for weight in exact_weights]
    cells = [quota.numerator // quota.denominator for quota in exact]
    remaining = candidate_count - sum(cells)
    # Input order is the deterministic tie-breaker. A zero-weight profile has
    # zero remainder and therefore remains empty while positive profiles exist.
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(exact[index] - cells[index]), index),
    )
    for index in order[:remaining]:
        cells[index] += 1
    return cells


def plan_room_questions(
    profile_ids: object,
    params: Mapping[str, Any],
    *,
    question_budget: int | None = None,
    answer_forms: object | None = None,
    profile_weights: object | None = None,
) -> dict[str, Any]:
    """Plan a room's final QA-item budget across profiles.

    The budget counts final question items. One candidate reserves one item
    for every requested answer form, so a paired ``mcq``/``open`` request
    costs two items. Profile allocation uses the largest-remainder method over
    candidate slots, with input order as the deterministic tie-breaker.
    """

    if not isinstance(params, Mapping):
        raise QARequestError("params must be a mapping")
    profiles = _profile_ids(profile_ids)
    if question_budget is None:
        if PARAM_ITEMS_PER_ROOM not in params:
            raise QARequestError(
                f"params is missing required {PARAM_ITEMS_PER_ROOM!r}"
            )
        requested_budget = params[PARAM_ITEMS_PER_ROOM]
    else:
        requested_budget = question_budget
    requested_budget = _positive_integer(
        requested_budget, name="question_budget"
    )

    if answer_forms is None:
        if PARAM_ANSWER_FORMS not in params:
            raise QARequestError(
                f"params is missing required {PARAM_ANSWER_FORMS!r}"
            )
        forms_value = params[PARAM_ANSWER_FORMS]
    else:
        forms_value = answer_forms
    forms = normalize_answer_forms(forms_value)
    candidate_cost = len(forms)
    candidate_budget, unallocated_budget = divmod(requested_budget, candidate_cost)
    if candidate_budget < 1:
        raise QARequestError(
            f"question_budget={requested_budget} is smaller than one candidate "
            f"answer-form cost={candidate_cost}"
        )

    weights = _weights(profiles, profile_weights)
    cell_counts = _largest_remainder(candidate_budget, weights)
    cells = dict(zip(profiles, cell_counts))
    planned_candidates = sum(cell_counts)
    planned_question_count = planned_candidates * candidate_cost
    return {
        "cells": cells,
        "profile_weights": dict(zip(profiles, weights)),
        "answer_forms": forms,
        "forms_per_candidate": candidate_cost,
        "planned_candidates": planned_candidates,
        "planned_question_count": planned_question_count,
        "requested_budget": requested_budget,
        "unallocated_budget": unallocated_budget,
    }



def read_qa_params(path: str | Path) -> dict[str, Any]:
    """Read QA parameters, resolving data references before recording a copy."""
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QARequestError("params JSON must be an object")
    for field in ("SOUND_EVENT_POOL", "SPEECH_SELECTION_POLICY"):
        if value.get(field) is None:
            continue
        raw = value[field]
        if not isinstance(raw, str) or not raw.strip():
            raise QARequestError(f"{field} must be a non-empty path")
        resource = Path(raw).expanduser()
        value[field] = str(
            (resource if resource.is_absolute() else source.parent / resource).resolve())
    return value


def answer_forms_from_params(params: Mapping[str, Any]) -> list[str]:
    """Retain the historical two-form default only when no request declares it."""
    return normalize_answer_forms(params.get(PARAM_ANSWER_FORMS, ["mcq", "open"]))


def write_requested_questions(output_root: Path, fact_paths, params) -> dict[str, Any]:
    """Write requested question views; internal twin facts may retain both golds.

    Main questions consume the request budget. Counterfactual question views
    are counted separately and never silently double that budget.
    """
    forms = answer_forms_from_params(params)
    counts = {"main": 0, "gateA": 0}
    output_root = Path(output_root)
    main_path = output_root / "questions.jsonl"
    twin_path = output_root / "questions_gateA.jsonl"
    with main_path.open("x", encoding="utf-8") as main, twin_path.open("x", encoding="utf-8") as twin:
        for fact_path in fact_paths:
            fact_path = Path(fact_path)
            paths = [("main", fact_path, main)]
            gatea_path = fact_path.with_name("fact_record_gateA.json")
            if gatea_path.is_file():
                paths.append(("gateA", gatea_path, twin))
            for variant, path, stream in paths:
                fact = json.loads(path.read_text(encoding="utf-8"))
                for form in forms:
                    block = fact.get(form)
                    if not isinstance(block, Mapping) or not isinstance(block.get("stem"), str):
                        raise QARequestError(f"{path}: requested {form} question is missing")
                    row = {"scene_id": fact.get("scene_id"), "point_id": fact["point_id"],
                           "profile_id": fact["profile_id"], "variant": variant, "form": form,
                           "question": block["stem"],
                           "answer": {key: value for key, value in block.items() if key != "stem"}}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[variant] += 1
    return {"answer_forms": forms, "forms_per_candidate": len(forms),
            "designed_question_count": counts["main"],
            "counterfactual_question_count": counts["gateA"],
            "questions": str(main_path.resolve()),
            "counterfactual_questions": str(twin_path.resolve())}



def batch_point_ids(inputs_root: str | Path, requested: Sequence[str] | None = None) -> list[str]:
    """Select completed design records, never leftover rejected timelines.

    Standalone capture inputs without a batch manifest retain directory-based
    discovery. Once a design batch declares its results, those ordinary IDs
    determine membership, including for an explicitly requested subset.
    """
    root = Path(inputs_root)

    def point_id(value: object) -> str:
        if (not isinstance(value, str) or not value or value in {".", ".."}
                or Path(value).name != value):
            raise QARequestError(f"invalid batch point ID: {value!r}")
        return value

    manifest_path = root / "batch_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise QARequestError("batch manifest must be an object")
        if manifest.get("status") == "failed":
            raise QARequestError("the design batch failed; inspect its recorded failure first")
        records = manifest.get("records", manifest.get("selected"))
        if records is None and (root / "facts.jsonl").is_file():
            all_facts = [json.loads(line) for line in (root / "facts.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
            if any(not isinstance(record, Mapping) for record in all_facts):
                raise QARequestError("batch facts must be objects")
            # Ordinary design batches write main and Gate A rows together.
            # Variants share a point; only the main row declares its capture.
            records = [record for record in all_facts
                       if record.get("variant") in (None, "main")]
        if records is None and (manifest.get("counts") or {}).get("geometry_candidates") == 0:
            records = []
        if not isinstance(records, list) or any(not isinstance(r, Mapping) for r in records):
            raise QARequestError("design batch has no completed point records")
        available = [point_id(record.get("point_id")) for record in records]
        if len(available) != len(set(available)):
            raise QARequestError("design batch contains duplicate completed point IDs")
        rejected = set()
        for key in ("rejected", "rejections"):
            for record in manifest.get(key, []):
                if isinstance(record, Mapping) and record.get("point_id") is not None:
                    rejected.add(point_id(record["point_id"]))
        if set(available) & rejected:
            raise QARequestError("design batch marks the same point completed and rejected")
    else:
        available = sorted(p.name for p in root.iterdir()
                           if p.is_dir() and (p / "timeline.json").is_file())
    selected = available if requested is None else [point_id(value) for value in requested]
    if len(selected) != len(set(selected)):
        raise QARequestError("requested batch point IDs must be unique")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise QARequestError(f"points are not completed members of this batch: {unknown}")
    return selected

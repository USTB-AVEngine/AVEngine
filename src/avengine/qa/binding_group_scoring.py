"""Score grouped audiovisual QA binding benchmark members.

The grouped benchmark is a thin private evaluation layer around the existing
unified question scorer. A member is one supplied unified question item, so
this module never expands a question set or adds catalog angle followups.
Group and relation metrics are therefore defined by the explicit members and
comparisons in the input artifact.

The artifact contains private question forms and gold answers. Its output is
research-only: the gold declaration self-check is a software consistency
check, and relation agreement is reported separately from per-item and
group-all-correct scores.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from avengine.qa.unified_scoring import UnifiedScoreError, score_unified_item


BINDING_GROUP_SCHEMA = "avengine_binding_groups_v1"
BINDING_GROUP_SCORE_SCHEMA = "avengine_binding_group_score_v1"

_FORMS = {"mcq", "open"}
_RELATION_KINDS = {"necessity", "invariance"}
_ANSWER_RELATIONS = {"same", "different"}
_SHARED_MODALITIES = {"audio", "video"}


class BindingGroupScoreError(ValueError):
    """A grouped scoring artifact or scoring request is malformed."""


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BindingGroupScoreError(f"{field} must be a non-empty string")
    return value.strip()


def validate_binding_groups(document: Mapping[str, Any]) -> None:
    """Validate the explicit grouped benchmark shape without reading media."""

    if not isinstance(document, Mapping):
        raise BindingGroupScoreError("binding groups must be an object")
    if document.get("schema") != BINDING_GROUP_SCHEMA:
        raise BindingGroupScoreError(
            f"binding groups schema must be {BINDING_GROUP_SCHEMA!r}"
        )
    if document.get("status") != "research_candidate":
        raise BindingGroupScoreError(
            "binding groups status must be 'research_candidate'"
        )
    groups = document.get("groups")
    if not _is_sequence(groups) or not groups:
        raise BindingGroupScoreError("binding groups must contain at least one group")

    group_ids: set[str] = set()
    sample_ids: set[str] = set()
    for group_index, group in enumerate(groups):
        prefix = f"groups[{group_index}]"
        if not isinstance(group, Mapping):
            raise BindingGroupScoreError(f"{prefix} must be an object")
        group_id = _text(group.get("group_id"), field=f"{prefix}.group_id")
        if group_id in group_ids:
            raise BindingGroupScoreError(f"duplicate group_id: {group_id!r}")
        group_ids.add(group_id)
        for field in ("task_family", "room_family", "room_id", "split"):
            _text(group.get(field), field=f"{prefix}.{field}")

        members = group.get("members")
        if not _is_sequence(members) or not members:
            raise BindingGroupScoreError(f"{prefix}.members must be non-empty")
        local_sample_ids: set[str] = set()
        for member_index, member in enumerate(members):
            member_prefix = f"{prefix}.members[{member_index}]"
            if not isinstance(member, Mapping):
                raise BindingGroupScoreError(f"{member_prefix} must be an object")
            sample_id = _text(
                member.get("sample_id"), field=f"{member_prefix}.sample_id"
            )
            if sample_id in local_sample_ids:
                raise BindingGroupScoreError(
                    f"{prefix} has duplicate sample_id: {sample_id!r}"
                )
            if sample_id in sample_ids:
                raise BindingGroupScoreError(
                    f"sample_id occurs in more than one group: {sample_id!r}"
                )
            local_sample_ids.add(sample_id)
            sample_ids.add(sample_id)
            if not isinstance(member.get("question"), Mapping):
                raise BindingGroupScoreError(
                    f"{member_prefix}.question must be an object"
                )
            media = member.get("media")
            if not isinstance(media, Mapping):
                raise BindingGroupScoreError(
                    f"{member_prefix}.media must be an object"
                )
            for media_key in ("video_path", "audio_path"):
                if media_key not in media:
                    raise BindingGroupScoreError(
                        f"{member_prefix}.media requires {media_key}"
                    )
                media_value = media[media_key]
                if media_value is not None and (
                    not isinstance(media_value, str) or not media_value.strip()
                ):
                    raise BindingGroupScoreError(
                        f"{member_prefix}.media.{media_key} must be text or null"
                    )

        comparisons = group.get("comparisons", [])
        if not _is_sequence(comparisons):
            raise BindingGroupScoreError(f"{prefix}.comparisons must be a list")
        comparison_keys: set[tuple[tuple[str, str], str]] = set()
        for comparison_index, comparison in enumerate(comparisons):
            comparison_prefix = f"{prefix}.comparisons[{comparison_index}]"
            if not isinstance(comparison, Mapping):
                raise BindingGroupScoreError(
                    f"{comparison_prefix} must be an object"
                )
            pair = comparison.get("members")
            if not _is_sequence(pair) or len(pair) != 2:
                raise BindingGroupScoreError(
                    f"{comparison_prefix}.members must contain two sample_ids"
                )
            pair_ids = tuple(
                _text(
                    value,
                    field=f"{comparison_prefix}.members[{index}]",
                )
                for index, value in enumerate(pair)
            )
            if pair_ids[0] == pair_ids[1]:
                raise BindingGroupScoreError(
                    f"{comparison_prefix}.members must be distinct"
                )
            if any(value not in local_sample_ids for value in pair_ids):
                raise BindingGroupScoreError(
                    f"{comparison_prefix}.members must refer to members in the same group"
                )
            kind = _text(comparison.get("kind"), field=f"{comparison_prefix}.kind")
            if kind not in _RELATION_KINDS:
                raise BindingGroupScoreError(
                    f"{comparison_prefix}.kind must be one of "
                    f"{sorted(_RELATION_KINDS)}"
                )
            relation = _text(
                comparison.get("answer_relation"),
                field=f"{comparison_prefix}.answer_relation",
            )
            if relation not in _ANSWER_RELATIONS:
                raise BindingGroupScoreError(
                    f"{comparison_prefix}.answer_relation must be one of "
                    f"{sorted(_ANSWER_RELATIONS)}"
                )
            shared = comparison.get("shared_modality")
            if shared is not None and shared not in _SHARED_MODALITIES:
                raise BindingGroupScoreError(
                    f"{comparison_prefix}.shared_modality must be audio, video, or null"
                )
            pair_key = tuple(sorted(pair_ids))
            key = (pair_key, kind)
            if key in comparison_keys:
                raise BindingGroupScoreError(
                    f"{prefix} repeats comparison pair/kind {pair_key!r}/{kind!r}"
                )
            comparison_keys.add(key)


def _normalise_answers(answers: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(answers, Mapping):
        raise BindingGroupScoreError(
            "answers must be a mapping from sample_id to model answer"
        )
    result: dict[str, Any] = {}
    for key, value in answers.items():
        if not isinstance(key, str) or not key.strip():
            raise BindingGroupScoreError("answer mapping keys must be sample_ids")
        result[key.strip()] = value
    return result


def _canonical_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _normalisation_policy(form: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = form.get("normalization")
    if policy is None:
        return {
            "unicode_form": "NFKC",
            "casefold": True,
            "punctuation": "space",
        }
    return policy if isinstance(policy, Mapping) else {}


def _normalised_words(value: Any, form: Mapping[str, Any]) -> tuple[str, ...] | None:
    if not isinstance(value, str):
        return None
    policy = _normalisation_policy(form)
    unicode_form = policy.get("unicode_form")
    if unicode_form not in {"NFC", "NFKC", "none"}:
        return None
    if not isinstance(policy.get("casefold"), bool):
        return None
    punctuation = policy.get("punctuation")
    if punctuation not in {"keep", "space", "remove"}:
        return None
    text = value
    if unicode_form != "none":
        text = unicodedata.normalize(unicode_form, text)
    if policy["casefold"]:
        text = text.casefold()
    if punctuation != "keep":
        replacement = " " if punctuation == "space" else ""
        text = "".join(
            replacement
            if unicodedata.category(char).startswith("P")
            else char
            for char in text
        )
    return tuple(text.split())


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _freeze(child)) for key, child in value.items())
        )
    if _is_sequence(value):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else ("nonfinite", repr(value))
    return value


def _angle(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("azimuth_deg", value.get("value"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return (value + 180.0) % 360.0 - 180.0


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _time_range(value: Any) -> tuple[float, float] | None:
    if not _is_sequence(value) or len(value) != 2:
        return None
    first, second = (_numeric(item) for item in value)
    if first is None or second is None or second <= first:
        return None
    return first, second


def _counts(value: Any) -> tuple[int, ...] | None:
    values = value if _is_sequence(value) else [value]
    result: list[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return None
        result.append(item)
    return tuple(result) if result else None


def _form_kind(question: Mapping[str, Any], form: str) -> str | None:
    forms = question.get("forms")
    if not isinstance(forms, Mapping) or form not in forms:
        return None
    if form == "mcq":
        return "choice"
    specification = forms[form]
    if not isinstance(specification, Mapping):
        return None
    answer_type = specification.get("answer_type")
    return str(answer_type) if isinstance(answer_type, str) else None


def _gold_signature(
    question: Mapping[str, Any], form: str
) -> tuple[str, Any] | None:
    forms = question.get("forms")
    if not isinstance(forms, Mapping) or form not in forms:
        return None
    specification = forms[form]
    if not isinstance(specification, Mapping):
        return None
    if form == "mcq":
        gold = specification.get("gold")
        if not isinstance(gold, Mapping):
            return None
        value = gold.get("value")
        if value is None:
            index = gold.get("correct_index")
            options = specification.get("options")
            if (
                isinstance(index, int)
                and not isinstance(index, bool)
                and _is_sequence(options)
                and 0 <= index < len(options)
                and isinstance(options[index], Mapping)
            ):
                value = options[index].get("value")
        if value is None:
            index = gold.get("correct_index")
            if isinstance(index, int) and not isinstance(index, bool):
                value = index
            else:
                return None
        return ("choice", _freeze(value))

    answer_type = specification.get("answer_type")
    truth = specification.get("truth")
    if not isinstance(answer_type, str) or truth is None:
        return None
    if answer_type == "closed_set":
        text = _canonical_text(truth)
        return ("closed_set", text) if text else None
    if answer_type == "transcript_wer":
        words = _normalised_words(truth, specification)
        return ("transcript_wer", words) if words else None
    if answer_type == "angle_deg":
        value = _angle(truth)
        return ("angle_deg", value) if value is not None else None
    if answer_type == "time_s":
        value = _numeric(truth)
        return ("time_s", value) if value is not None else None
    if answer_type == "time_range_s":
        value = _time_range(truth)
        return ("time_range_s", value) if value is not None else None
    if answer_type in {"count_pair", "count_single"}:
        value = _counts(truth)
        return (answer_type, value) if value is not None else None
    return (answer_type, _freeze(truth))


def _model_signature(
    question: Mapping[str, Any],
    form: str,
    answer: Any,
    result: Mapping[str, Any],
) -> tuple[str, Any] | None:
    if result.get("status") != "scored":
        return None
    forms = question.get("forms")
    specification = forms.get(form) if isinstance(forms, Mapping) else None
    if not isinstance(specification, Mapping):
        return None
    kind = _form_kind(question, form)
    if kind is None:
        return None
    if form == "mcq":
        index = result.get("parsed_index")
        options = specification.get("options")
        value: Any = index
        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and _is_sequence(options)
            and 0 <= index < len(options)
            and isinstance(options[index], Mapping)
        ):
            value = options[index].get(
                "value", options[index].get("label_en", index)
            )
        # Use option value rather than index: aligned groups may still contain
        # shuffled labels, while the semantic answer remains the same.
        return ("choice", _freeze(value))
    if kind == "transcript_wer":
        words = _normalised_words(str(answer), specification)
        return (kind, words) if words else None
    parsed = result.get("parsed")
    if kind == "angle_deg":
        parsed = _angle(parsed)
    elif kind == "time_s":
        parsed = _numeric(parsed)
    elif kind == "time_range_s":
        parsed = _time_range(parsed)
    elif kind in {"count_pair", "count_single"}:
        parsed = _counts(parsed)
    elif kind == "closed_set":
        parsed = _canonical_text(parsed)
    else:
        parsed = _freeze(parsed)
    return (kind, parsed) if parsed is not None else None


def _signature_equal(first: Any, second: Any) -> bool:
    if (
        isinstance(first, (int, float))
        and not isinstance(first, bool)
        and isinstance(second, (int, float))
        and not isinstance(second, bool)
    ):
        return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=1.0e-6)
    if isinstance(first, tuple) and isinstance(second, tuple):
        return len(first) == len(second) and all(
            _signature_equal(left, right) for left, right in zip(first, second)
        )
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            _signature_equal(left, right) for left, right in zip(first, second)
        )
    return first == second


def _member_base(
    sample_id: str, question: Mapping[str, Any], form: str
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "question_id": question.get("question_id"),
        "qa_id": question.get("qa_id"),
        "form": form,
    }


def _score_member(
    member: Mapping[str, Any],
    answers: Mapping[str, Any],
    *,
    form: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    sample_id = str(member["sample_id"])
    question = member["question"]
    base = _member_base(sample_id, question, form)
    forms = question.get("forms") if isinstance(question, Mapping) else None
    if not isinstance(forms, Mapping):
        return {
            **base,
            "status": "invalid",
            "reason": "question item has no forms object",
            "score": 0.0,
            "answer_present": sample_id in answers,
            "missing_answer": sample_id not in answers,
            "_answer_signature": None,
            "_gold_signature": None,
            "_answer_kind": None,
        }
    if form not in forms:
        return {
            **base,
            "status": "unavailable_form",
            "reason": f"question item has no {form} form",
            "score": None,
            "answer_present": sample_id in answers,
            "missing_answer": False,
            "_answer_signature": None,
            "_gold_signature": None,
            "_answer_kind": None,
        }
    specification = forms[form]
    if not isinstance(specification, Mapping):
        return {
            **base,
            "status": "invalid",
            "reason": f"{form} form is not an object",
            "score": 0.0,
            "answer_present": sample_id in answers,
            "missing_answer": sample_id not in answers,
            "_answer_signature": None,
            "_gold_signature": None,
            "_answer_kind": None,
        }
    gold_signature = _gold_signature(question, form)
    answer_kind = _form_kind(question, form)
    if sample_id not in answers:
        return {
            **base,
            "status": "invalid",
            "reason": "model answer is missing",
            "error_code": "missing_answer",
            "score": 0.0,
            "answer_present": False,
            "missing_answer": True,
            "_answer_signature": None,
            "_gold_signature": gold_signature,
            "_answer_kind": answer_kind,
        }
    answer = answers[sample_id]
    try:
        result = score_unified_item(
            question,
            answer,
            form=form,
            params=params,
        )
    except UnifiedScoreError as error:
        raise BindingGroupScoreError(
            f"cannot score sample {sample_id!r}: {error}"
        ) from error
    return {
        **base,
        **result,
        "answer_present": True,
        "missing_answer": False,
        "_answer_signature": _model_signature(question, form, answer, result),
        "_gold_signature": gold_signature,
        "_answer_kind": answer_kind,
    }


def _relation_result(
    comparison: Mapping[str, Any],
    by_sample: Mapping[str, Mapping[str, Any]],
    *, angle_tolerance_deg: float = 10.0,
) -> dict[str, Any]:
    pair = [str(comparison["members"][0]), str(comparison["members"][1])]
    first, second = by_sample[pair[0]], by_sample[pair[1]]
    expected = str(comparison["answer_relation"])
    relation = {
        "members": pair,
        "shared_modality": comparison.get("shared_modality"),
        "kind": str(comparison["kind"]),
        "answer_relation": expected,
    }
    unavailable = [
        record["sample_id"]
        for record in (first, second)
        if record.get("status") == "unavailable_form"
    ]
    if unavailable:
        relation.update(
            {
                "status": "unavailable_form",
                "reason": "requested form is unavailable for one or both members",
                "unavailable_sample_ids": unavailable,
                "observed_relation": None,
                "relation_correct": None,
            }
        )
    elif first.get("status") != "scored" or second.get("status") != "scored":
        invalid = [
            record["sample_id"]
            for record in (first, second)
            if record.get("status") != "scored"
        ]
        relation.update(
            {
                "status": "invalid",
                "reason": "one or both model answers are missing, invalid, or abstained",
                "invalid_sample_ids": invalid,
                "observed_relation": None,
                "relation_correct": False,
            }
        )
    elif first.get("_answer_kind") != second.get("_answer_kind"):
        relation.update(
            {
                "status": "invalid",
                "reason": "comparison members have different answer types",
                "observed_relation": None,
                "relation_correct": False,
            }
        )
    elif first.get("_answer_signature") is None or second.get(
        "_answer_signature"
    ) is None:
        relation.update(
            {
                "status": "invalid",
                "reason": "one or both scored answers have no semantic signature",
                "observed_relation": None,
                "relation_correct": False,
            }
        )
    elif first.get("_answer_kind") == "angle_deg":
        predicted_change = (second["_answer_signature"][1] - first["_answer_signature"][1] + 180.0) % 360.0 - 180.0
        gold_change = (second["_gold_signature"][1] - first["_gold_signature"][1] + 180.0) % 360.0 - 180.0
        change_error = abs((predicted_change - gold_change + 180.0) % 360.0 - 180.0)
        relation.update({
            "status": "scored", "observed_relation": "circular_change",
            "comparison_mode": "circular_change_against_gold",
            "predicted_change_deg": predicted_change, "expected_change_deg": gold_change,
            "change_error_deg": change_error, "change_tolerance_deg": 2 * angle_tolerance_deg,
            "relation_correct": change_error <= 2 * angle_tolerance_deg,
        })
    else:
        observed = (
            "same"
            if _signature_equal(
                first["_answer_signature"], second["_answer_signature"]
            )
            else "different"
        )
        relation.update(
            {
                "status": "scored",
                "observed_relation": observed,
                "relation_correct": observed == expected,
            }
        )

    gold_first = first.get("_gold_signature")
    gold_second = second.get("_gold_signature")
    if first.get("status") == "unavailable_form" or second.get(
        "status"
    ) == "unavailable_form":
        relation["gold_selfcheck"] = {
            "status": "unavailable_form",
            "declared_relation": expected,
            "gold_relation": None,
            "matches": None,
        }
    elif (
        not isinstance(gold_first, tuple)
        or not isinstance(gold_second, tuple)
        or len(gold_first) != 2
        or len(gold_second) != 2
    ):
        relation["gold_selfcheck"] = {
            "status": "invalid",
            "declared_relation": expected,
            "gold_relation": None,
            "matches": None,
        }
    elif gold_first[0] != gold_second[0]:
        relation["gold_selfcheck"] = {
            "status": "invalid",
            "reason": "gold forms have different answer types",
            "declared_relation": expected,
            "gold_relation": None,
            "matches": None,
        }
    else:
        gold_relation = (
            "same"
            if _signature_equal(gold_first, gold_second)
            else "different"
        )
        relation["gold_selfcheck"] = {
            "status": "checked",
            "declared_relation": expected,
            "gold_relation": gold_relation,
            "matches": gold_relation == expected,
        }
    return relation


def _clean(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: child for key, child in value.items() if not str(key).startswith("_")
    }


def _item_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [
        record for record in records if record.get("status") != "unavailable_form"
    ]
    scored = [record for record in available if record.get("status") == "scored"]
    correct = [
        record
        for record in scored
        if isinstance(record.get("score"), (int, float))
        and not isinstance(record.get("score"), bool)
        and math.isfinite(float(record["score"]))
        and bool(record.get("correct"))
    ]
    total_score = sum(
        float(record.get("score", 0.0))
        for record in available
        if isinstance(record.get("score"), (int, float))
        and not isinstance(record.get("score"), bool)
        and math.isfinite(float(record["score"]))
    )
    scored_score = sum(float(record["score"]) for record in scored)
    denominator = len(available)
    return {
        "total": len(records),
        "denominator": denominator,
        "scored": len(scored),
        "correct": len(correct),
        "invalid": sum(record.get("status") == "invalid" for record in available),
        "abstained": sum(record.get("status") == "abstained" for record in available),
        "missing": sum(bool(record.get("missing_answer")) for record in available),
        "unavailable_form": sum(
            record.get("status") == "unavailable_form" for record in records
        ),
        "accuracy": len(correct) / denominator if denominator else None,
        "mean_score_over_all": total_score / denominator if denominator else None,
        "mean_score_over_scored": scored_score / len(scored) if scored else None,
        "denominator_policy": (
            "requested-form members; missing, invalid and abstained answers count "
            "as zero-score failures"
        ),
    }


def _group_metrics(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [
        group for group in groups if group.get("status") != "unavailable_form"
    ]
    correct = sum(group.get("all_correct") is True for group in available)
    denominator = len(available)
    return {
        "total": len(groups),
        "denominator": denominator,
        "all_correct": correct,
        "unavailable_form": sum(
            group.get("status") == "unavailable_form" for group in groups
        ),
        "accuracy": correct / denominator if denominator else None,
        "denominator_policy": (
            "groups with the requested form on every member; missing, invalid and "
            "abstained answers fail all-correct"
        ),
    }


def _relation_metrics(relations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        usable = [
            row for row in rows if row.get("status") != "unavailable_form"
        ]
        correct = sum(row.get("relation_correct") is True for row in usable)
        return {
            "total": len(rows),
            "denominator": len(usable),
            "scored": sum(row.get("status") == "scored" for row in usable),
            "correct": correct,
            "invalid": sum(row.get("status") == "invalid" for row in usable),
            "unavailable_form": sum(
                row.get("status") == "unavailable_form" for row in rows
            ),
            "agreement_accuracy": (
                correct / len(usable) if usable else None
            ),
            "denominator_policy": (
                "comparisons whose members have the requested form; missing, "
                "invalid and abstained answers count as relation failures"
            ),
        }

    return {
        **summary(relations),
        "by_kind": {
            kind: summary([row for row in relations if row.get("kind") == kind])
            for kind in sorted(_RELATION_KINDS)
        },
        "by_expected_relation": {
            relation: summary(
                [
                    row
                    for row in relations
                    if row.get("answer_relation") == relation
                ]
            )
            for relation in sorted(_ANSWER_RELATIONS)
        },
    }


def _angle_group_metrics(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for threshold in (1, 3, 5, 10):
        values = [group.get("all_correct_at_angle_deg", {}).get(str(threshold)) for group in groups]
        observed = [value for value in values if value is not None]
        correct = sum(value is True for value in observed)
        result[str(threshold)] = {
            "denominator": len(observed), "correct": correct,
            "accuracy": correct / len(observed) if observed else None,
        }
    return result


def _gold_selfcheck(relations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks = [relation.get("gold_selfcheck", {}) for relation in relations]
    checked = [row for row in checks if row.get("status") == "checked"]
    mismatches = [row for row in checked if row.get("matches") is False]
    unavailable = [
        row for row in checks if row.get("status") == "unavailable_form"
    ]
    invalid = [row for row in checks if row.get("status") == "invalid"]
    if not checks:
        status = "not_run"
        all_match = None
    elif mismatches or invalid:
        status = "mismatch" if mismatches else "invalid"
        all_match = False
    elif unavailable:
        status = "partial"
        all_match = None
    else:
        status = "pass"
        all_match = True
    return {
        "status": status,
        "mode": "software_consistency_only",
        "comparisons": len(checks),
        "checked": len(checked),
        "mismatches": len(mismatches),
        "invalid": len(invalid),
        "unavailable_form": len(unavailable),
        "all_declared_relations_match_gold": all_match,
        "claim_boundary": (
            "This checks only that declared same/different relations agree with "
            "the supplied private question forms and gold values. It is not a "
            "model run, human calibration, modality-necessity certificate, or "
            "dataset admission."
        ),
    }


def score_binding_groups(
    document: Mapping[str, Any],
    answers: Mapping[str, Any],
    *,
    form: str = "open",
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score explicit grouped members and their declared answer relations."""

    if form not in _FORMS:
        raise BindingGroupScoreError("form must be 'mcq' or 'open'")
    validate_binding_groups(document)
    answer_map = _normalise_answers(answers)
    params_map: Mapping[str, Any]
    if params is None:
        params_map = {}
    elif isinstance(params, Mapping):
        params_map = params
    else:
        raise BindingGroupScoreError("params must be an object")

    groups = document["groups"]
    expected_sample_ids = {
        str(member["sample_id"])
        for group in groups
        for member in group["members"]
    }
    unknown = sorted(set(answer_map) - expected_sample_ids)
    if unknown:
        raise BindingGroupScoreError(
            f"answers contain unknown sample_ids: {unknown}"
        )

    group_results: list[dict[str, Any]] = []
    all_members: list[dict[str, Any]] = []
    all_relations: list[dict[str, Any]] = []
    for group in groups:
        try:
            angle_tolerance = float(group.get("angle_tolerance_deg", 10.0))
        except (TypeError, ValueError) as error:
            raise BindingGroupScoreError("angle_tolerance_deg must be a finite degree tolerance") from error
        if not math.isfinite(angle_tolerance) or not 0 <= angle_tolerance <= 180:
            raise BindingGroupScoreError("angle_tolerance_deg must be between 0 and 180")
        member_records = [
            _score_member(
                member,
                answer_map,
                form=form,
                params=params_map,
            )
            for member in group["members"]
        ]
        for record in member_records:
            if record.get("_answer_kind") == "angle_deg":
                error = record.get("circular_error_deg", math.inf)
                record["correct"] = (
                    record.get("status") == "scored"
                    and isinstance(error, (int, float)) and math.isfinite(float(error))
                    and float(error) <= angle_tolerance
                )
                record["correctness_tolerance_deg"] = angle_tolerance
            else:
                record["correct"] = record.get("status") == "scored" and record.get("score") == 1.0
        by_sample = {
            str(record["sample_id"]): record for record in member_records
        }
        relation_records = [
            _relation_result(comparison, by_sample, angle_tolerance_deg=angle_tolerance)
            for comparison in group.get("comparisons", [])
        ]
        available = all(
            record.get("status") != "unavailable_form"
            for record in member_records
        )
        all_correct = (
            all(
                bool(record.get("correct"))
                for record in member_records
            )
            if available
            else None
        )
        group_result = {
            "group_id": group["group_id"],
            "task_family": group["task_family"],
            "room_family": group["room_family"],
            "room_id": group["room_id"],
            "split": group["split"],
            "status": "scored" if available else "unavailable_form",
            "all_correct": all_correct,
            "all_correct_failure_sample_ids": (
                [
                    record["sample_id"]
                    for record in member_records
                    if not record.get("correct")
                ]
                if available
                else []
            ),
            "unavailable_sample_ids": [
                record["sample_id"]
                for record in member_records
                if record.get("status") == "unavailable_form"
            ],
            "members": [_clean(record) for record in member_records],
            "comparisons": [_clean(record) for record in relation_records],
        }
        if any(record.get("_answer_kind") == "angle_deg" for record in member_records):
            group_result["angle_tolerance_deg"] = angle_tolerance
            group_result["all_correct_at_angle_deg"] = {
                str(threshold): (all(
                    record.get("status") == "scored"
                    and (float(record.get("circular_error_deg", math.inf)) <= threshold
                         if record.get("_answer_kind") == "angle_deg" else bool(record.get("correct")))
                    for record in member_records) if available else None)
                for threshold in (1, 3, 5, 10)
            }
        group_results.append(group_result)
        all_members.extend(member_records)
        all_relations.extend(relation_records)

    item_metrics = _item_metrics(all_members)
    group_metrics = _group_metrics(group_results)
    relation_metrics = _relation_metrics(all_relations)
    return {
        "schema": BINDING_GROUP_SCORE_SCHEMA,
        "status": "research_candidate",
        "qualification_claim": False,
        "form": form,
        "counts": {
            "groups_total": len(group_results),
            "groups_form_available": group_metrics["denominator"],
            "groups_unavailable_form": group_metrics["unavailable_form"],
            "groups_all_correct": group_metrics["all_correct"],
            "members_total": item_metrics["total"],
            "members_form_available": item_metrics["denominator"],
            "members_unavailable_form": item_metrics["unavailable_form"],
            "members_scored": item_metrics["scored"],
            "members_correct": item_metrics["correct"],
            "members_invalid": item_metrics["invalid"],
            "members_abstained": item_metrics["abstained"],
            "members_missing": item_metrics["missing"],
            "comparisons_total": relation_metrics["total"],
            "comparisons_form_available": relation_metrics["denominator"],
            "comparisons_scored": relation_metrics["scored"],
            "comparisons_correct": relation_metrics["correct"],
            "comparisons_invalid": relation_metrics["invalid"],
            "comparisons_unavailable_form": relation_metrics["unavailable_form"],
        },
        "item_metrics": item_metrics,
        "group_all_correct": group_metrics,
        "angle_group_accuracy_at_deg": _angle_group_metrics(group_results),
        "relation_metrics": relation_metrics,
        "gold_selfcheck": _gold_selfcheck(all_relations),
        "groups": group_results,
        "claim_boundary": (
            "Scores compare supplied answers with research-candidate question forms. "
            "Relation agreement measures categorical relations or signed circular "
            "angle changes separately from item and group correctness. Supplied "
            "answers and gold selfchecks alone provide no independent model, "
            "human-answerability, modality-necessity, or admission evidence."
        ),
    }


__all__ = [
    "BINDING_GROUP_SCHEMA",
    "BINDING_GROUP_SCORE_SCHEMA",
    "BindingGroupScoreError",
    "score_binding_groups",
    "validate_binding_groups",
]

"""Answer-key reporting and deterministic design-variant balancing.

This module deliberately separates two notions that are easy to conflate:

* the displayed answer key (``A``/``B``), derived only from
  ``options.index(answer)``; and
* the semantic source slot (``source1``/``source2``), supplied by the question
  compiler after it has resolved the underlying event or spatial fact.

Non-binary questions are reported but are outside the binary balancing scope.
An answer that is absent from, or duplicated in, a binary option list is an
invalid binary question and fails strict validation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


KNOWN_BINARY_TYPE_IDS = frozenset(
    {
        "QV2-T2-ATTR",
        "QV2-TA-MOTION",
        "QV2-T4-SIDE",
        "QV2-T7-DURING",
        "QV2-T9-CLOSER",
    }
)
KNOWN_NON_BINARY_ARITIES = frozenset(
    {"free_response", "nonbinary", "numeric", "numeric_banded", "open"}
)


class AnswerBalanceError(ValueError):
    """Raised when strict answer-key balance cannot be proved."""


def answer_key_for_question(question: Mapping[str, Any]) -> str | None:
    """Return ``A``/``B`` for a valid binary question, or ``None`` otherwise.

    Callers that need to distinguish a non-binary question from a malformed
    binary question should use :func:`build_answer_balance_report`, which
    records both cases separately.
    """

    options = question.get("options")
    if not isinstance(options, list) or len(options) != 2:
        return None
    matches = [index for index, option in enumerate(options) if option == question.get("answer")]
    if len(matches) != 1:
        return None
    return "A" if matches[0] == 0 else "B"


def _question_arity(question: Mapping[str, Any]) -> str:
    """Classify arity without silently treating an unknown row as non-binary."""

    if (
        question.get("answer_arity") == "binary"
        or str(question.get("type_id", "")) in KNOWN_BINARY_TYPE_IDS
    ):
        return "binary"
    if question.get("answer_arity") in KNOWN_NON_BINARY_ARITIES:
        return "non_binary"
    return "invalid"


def build_answer_balance_report(
    questions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report displayed-key and semantic-source balance by question type.

    A binary type passes when every binary answer resolves uniquely and its
    displayed A/B counts differ by at most one.  Non-binary rows are counted
    and skipped rather than coerced into this policy.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for question in questions:
        grouped[str(question.get("type_id", "<missing-type-id>"))].append(question)

    by_type: dict[str, dict[str, Any]] = {}
    failing_types: list[str] = []
    invalid_types: list[str] = []
    for type_id in sorted(grouped):
        rows = grouped[type_id]
        keys: Counter[str] = Counter()
        sources: Counter[str] = Counter()
        non_binary = 0
        unknown_answer = 0
        invalid_options = 0
        invalid_arity = 0
        for row in rows:
            options = row.get("options")
            arity = _question_arity(row)
            if arity == "non_binary":
                non_binary += 1
                continue
            if arity == "invalid":
                invalid_arity += 1
                continue
            if not isinstance(options, list) or len(options) != 2:
                invalid_options += 1
                unknown_answer += 1
                source = row.get("semantic_answer_source_slot")
                sources[source if source in {"source1", "source2"} else "unknown"] += 1
                continue
            key = answer_key_for_question(row)
            if key is None:
                unknown_answer += 1
            else:
                keys[key] += 1
            source = row.get("semantic_answer_source_slot")
            sources[source if source in {"source1", "source2"} else "unknown"] += 1

        binary_total = len(rows) - non_binary - invalid_arity
        valid_binary_total = keys["A"] + keys["B"]
        difference = abs(keys["A"] - keys["B"])
        if invalid_arity or unknown_answer:
            status = "invalid"
            invalid_types.append(type_id)
        elif binary_total == 0:
            status = "not_binary"
        elif difference <= 1:
            status = "pass"
        else:
            status = "fail"
            failing_types.append(type_id)
        by_type[type_id] = {
            "total": len(rows),
            "binary_total": binary_total,
            "valid_binary_total": valid_binary_total,
            "non_binary_skipped": non_binary,
            "unknown_answer_count": unknown_answer,
            "invalid_options_count": invalid_options,
            "invalid_arity_count": invalid_arity,
            "answer_key_counts": {"A": keys["A"], "B": keys["B"]},
            "answer_key_difference": difference,
            "semantic_source_counts": {
                "source1": sources["source1"],
                "source2": sources["source2"],
                "unknown": sources["unknown"],
            },
            "status": status,
        }

    return {
        "policy": "per_binary_type_50_50_plus_or_minus_1",
        "status": "fail" if failing_types or invalid_types else "pass",
        "failing_types": failing_types,
        "invalid_types": invalid_types,
        "by_type": by_type,
    }


def require_balanced_answer_keys(report: Mapping[str, Any]) -> None:
    """Raise with the exact failing types unless ``report`` proves balance."""

    if report.get("status") == "pass":
        return
    failing = list(report.get("failing_types", []))
    invalid = list(report.get("invalid_types", []))
    raise AnswerBalanceError(
        "binary answer-key balance failed: "
        f"imbalanced={failing or 'none'}, invalid={invalid or 'none'}"
    )


def _candidate_counts(candidate: Mapping[str, Any], type_ids: Sequence[str]) -> tuple[tuple[int, int], ...]:
    raw = candidate.get("answer_key_counts")
    if not isinstance(raw, Mapping):
        raise AnswerBalanceError("candidate is missing answer_key_counts")
    counts: list[tuple[int, int]] = []
    for type_id in type_ids:
        item = raw.get(type_id, {})
        if not isinstance(item, Mapping):
            raise AnswerBalanceError(f"candidate count for {type_id} is not a mapping")
        a_count, b_count = item.get("A", 0), item.get("B", 0)
        if (
            isinstance(a_count, bool)
            or isinstance(b_count, bool)
            or not isinstance(a_count, int)
            or not isinstance(b_count, int)
            or a_count < 0
            or b_count < 0
        ):
            raise AnswerBalanceError(f"candidate count for {type_id} is invalid")
        counts.append((a_count, b_count))
    return tuple(counts)


def choose_balanced_design_variants(
    candidate_groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    type_ids: Sequence[str],
    search_state_limit: int | None = None,
) -> list[Mapping[str, Any]]:
    """Choose one reversible candidate per design unit, deterministically.

    Each candidate supplies ``answer_key_counts`` for the question types it
    can prove from the authored spec.  Every candidate in one group must emit
    the same number of questions per type; variants may change only their A/B
    assignment.  This invariant prevents a planner from "balancing" by
    deleting questions.

    The exact dynamic program keeps only states that can still reach the
    ``50/50±1`` interval. A stable candidate order and lexicographic path
    tie-break make repeated runs identical. ``search_state_limit`` is an
    optional resource limit, never a pruning heuristic: exceeding it raises a
    distinct ``search_limit_exhausted`` error rather than claiming the design
    is mathematically unreachable.
    """

    ordered_types = tuple(type_ids)
    if len(set(ordered_types)) != len(ordered_types):
        raise AnswerBalanceError("type_ids must be unique")
    if search_state_limit is not None and (
        isinstance(search_state_limit, bool)
        or not isinstance(search_state_limit, int)
        or search_state_limit < 1
    ):
        raise AnswerBalanceError("search_state_limit must be a positive integer or None")
    if not candidate_groups:
        return []

    normalized: list[list[tuple[Mapping[str, Any], tuple[tuple[int, int], ...]]]] = []
    totals = [0] * len(ordered_types)
    per_group_totals: list[tuple[int, ...]] = []
    for group_index, group in enumerate(candidate_groups):
        if not group:
            raise AnswerBalanceError(f"candidate group {group_index} is empty")
        normalized_group = [(candidate, _candidate_counts(candidate, ordered_types)) for candidate in group]
        emitted = tuple(a + b for a, b in normalized_group[0][1])
        for _, counts in normalized_group[1:]:
            if tuple(a + b for a, b in counts) != emitted:
                raise AnswerBalanceError(
                    f"candidate group {group_index} changes question totals; deletion/rejection is not an allowed balance transform"
                )
        per_group_totals.append(emitted)
        for type_index, count in enumerate(emitted):
            totals[type_index] += count
        normalized.append(normalized_group)

    allowed_a = [
        {total // 2} if total % 2 == 0 else {total // 2, total // 2 + 1}
        for total in totals
    ]
    remaining = [list(totals)]
    for emitted in per_group_totals:
        remaining.append([left - used for left, used in zip(remaining[-1], emitted)])

    # A-count tuple -> bytes of selected candidate indices.  B counts are
    # implicit because every group's per-type total is invariant.
    states: dict[tuple[int, ...], bytes] = {(0,) * len(ordered_types): b""}
    for group_index, group in enumerate(normalized):
        next_states: dict[tuple[int, ...], bytes] = {}
        left_after = remaining[group_index + 1]
        for a_state, path in states.items():
            for candidate_index, (_, counts) in enumerate(group):
                if candidate_index > 255:
                    raise AnswerBalanceError("a candidate group cannot exceed 256 variants")
                new_state = tuple(
                    current + candidate_counts[0]
                    for current, candidate_counts in zip(a_state, counts)
                )
                reachable = all(
                    any(current <= target <= current + future for target in targets)
                    for current, future, targets in zip(new_state, left_after, allowed_a)
                )
                if not reachable:
                    continue
                new_path = path + bytes((candidate_index,))
                previous = next_states.get(new_state)
                if previous is None or new_path < previous:
                    next_states[new_state] = new_path
        if not next_states:
            raise AnswerBalanceError(
                f"no 50/50±1 design remains reachable after candidate group {group_index}"
            )
        if search_state_limit is not None and len(next_states) > search_state_limit:
            raise AnswerBalanceError(
                "search_limit_exhausted: exact answer-balance search produced "
                f"{len(next_states)} states after candidate group {group_index}; "
                f"limit={search_state_limit}"
            )
        states = next_states

    final_options = [
        (state, path)
        for state, path in states.items()
        if all(count in targets for count, targets in zip(state, allowed_a))
    ]
    if not final_options:
        raise AnswerBalanceError("candidate variants cannot achieve 50/50±1")
    _, selected_path = min(final_options, key=lambda item: (item[0], item[1]))
    return [
        normalized[group_index][candidate_index][0]
        for group_index, candidate_index in enumerate(selected_path)
    ]

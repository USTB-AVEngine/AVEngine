from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from avengine.qa.answer_balance import (
    AnswerBalanceError,
    build_answer_balance_report,
    choose_balanced_design_variants,
    require_balanced_answer_keys,
)


def _question(
    type_id: str,
    options: list[str],
    answer: str,
    source: str = "source1",
) -> dict[str, object]:
    return {
        "type_id": type_id,
        "answer_arity": "binary",
        "options": options,
        "answer": answer,
        "semantic_answer_source_slot": source,
    }


def test_binary_answer_keys_and_semantic_sources_are_reported_separately() -> None:
    questions = [
        _question("even", ["left", "right"], "left", "source2"),
        _question("even", ["left", "right"], "right", "source2"),
        _question("odd", ["no", "yes"], "no", "source1"),
        _question("odd", ["no", "yes"], "yes", "source1"),
        _question("odd", ["no", "yes"], "yes", "source2"),
    ]

    report = build_answer_balance_report(questions)

    assert report["status"] == "pass"
    assert report["by_type"]["even"] == {
        "total": 2,
        "binary_total": 2,
        "valid_binary_total": 2,
        "non_binary_skipped": 0,
        "unknown_answer_count": 0,
        "invalid_options_count": 0,
        "invalid_arity_count": 0,
        "answer_key_counts": {"A": 1, "B": 1},
        "answer_key_difference": 0,
        "semantic_source_counts": {"source1": 0, "source2": 2, "unknown": 0},
        "status": "pass",
    }
    assert report["by_type"]["odd"]["answer_key_counts"] == {"A": 1, "B": 2}
    assert report["by_type"]["odd"]["status"] == "pass"


def test_unknown_binary_answer_is_invalid_and_nonbinary_is_skipped() -> None:
    report = build_answer_balance_report(
        [
            _question("bad", ["A", "B"], "C"),
            _question("bad", ["same", "same"], "same"),
            {
                "type_id": "QV2-T5-DOA",
                "answer_arity": "numeric",
                "answer": 37.5,
            },
        ]
    )

    assert report["status"] == "fail"
    assert report["invalid_types"] == ["bad"]
    assert report["by_type"]["bad"]["unknown_answer_count"] == 2
    assert report["by_type"]["QV2-T5-DOA"]["status"] == "not_binary"
    assert report["by_type"]["QV2-T5-DOA"]["non_binary_skipped"] == 1
    with pytest.raises(AnswerBalanceError, match=r"invalid=\['bad'\]"):
        require_balanced_answer_keys(report)


def test_imbalanced_type_fails_strict_validation_without_dropping_rows() -> None:
    questions = [_question("side", ["left", "right"], "left") for _ in range(4)]
    report = build_answer_balance_report(questions)

    assert report["by_type"]["side"]["total"] == 4
    assert report["by_type"]["side"]["answer_key_counts"] == {"A": 4, "B": 0}
    assert report["failing_types"] == ["side"]
    with pytest.raises(AnswerBalanceError, match=r"imbalanced=\['side'\]"):
        require_balanced_answer_keys(report)


def test_known_binary_type_with_deduplicated_appearance_options_is_invalid() -> None:
    report = build_answer_balance_report(
        [
            {
                "type_id": "QV2-T2-ATTR",
                "options": ["blue"],
                "answer": "blue",
                "semantic_answer_source_slot": "source1",
            }
        ]
    )

    item = report["by_type"]["QV2-T2-ATTR"]
    assert item["binary_total"] == 1
    assert item["invalid_options_count"] == 1
    assert item["unknown_answer_count"] == 1
    assert item["status"] == "invalid"


def test_unclassified_row_is_invalid_instead_of_silently_skipped() -> None:
    report = build_answer_balance_report(
        [{"type_id": "future-type", "options": ["a", "b"], "answer": "a"}]
    )

    item = report["by_type"]["future-type"]
    assert item["invalid_arity_count"] == 1
    assert item["non_binary_skipped"] == 0
    assert item["status"] == "invalid"


def test_design_variant_choice_is_deterministic_and_balances_even_and_odd() -> None:
    groups = []
    for item_id in range(3):
        groups.append(
            [
                {
                    "id": f"{item_id}-afirst",
                    "answer_key_counts": {
                        "T-even": {"A": 1},
                        "T-odd": {"A": 1},
                    },
                },
                {
                    "id": f"{item_id}-bfirst",
                    "answer_key_counts": {
                        "T-even": {"B": 1},
                        "T-odd": {"B": 1},
                    },
                },
            ]
        )
    # A paired/twin unit contributes two T-even rows but no T-odd row.
    groups.append(
        [
            {
                "id": "pair-ab",
                "answer_key_counts": {"T-even": {"A": 1, "B": 1}},
            },
            {
                "id": "pair-ba",
                "answer_key_counts": {"T-even": {"A": 1, "B": 1}},
            },
        ]
    )

    first = choose_balanced_design_variants(groups, type_ids=["T-even", "T-odd"])
    second = choose_balanced_design_variants(groups, type_ids=["T-even", "T-odd"])

    assert [item["id"] for item in first] == [item["id"] for item in second]
    totals = {type_id: Counter() for type_id in ("T-even", "T-odd")}
    for candidate in first:
        for type_id, counts in candidate["answer_key_counts"].items():
            totals[type_id].update(counts)
    assert totals["T-even"] == Counter({"A": 2, "B": 3})
    assert abs(totals["T-odd"]["A"] - totals["T-odd"]["B"]) == 1


def test_design_variant_choice_rejects_question_deletion_as_a_transform() -> None:
    groups = [
        [
            {"answer_key_counts": {"T": {"A": 1}}},
            {"answer_key_counts": {"T": {}}},
        ]
    ]
    with pytest.raises(AnswerBalanceError, match="deletion/rejection"):
        choose_balanced_design_variants(groups, type_ids=["T"])


def test_exact_design_search_keeps_review_unique_five_type_solution() -> None:
    type_ids = [f"T{index}" for index in range(5)]
    groups = []
    for group_index in range(5):
        variants = []
        for candidate_index in range(8):
            counts = {}
            for type_index, type_id in enumerate(type_ids):
                a_count = candidate_index if type_index == group_index else 3
                counts[type_id] = {"A": a_count, "B": 7 - a_count}
            variants.append(
                {
                    "id": f"g{group_index}-v{candidate_index}",
                    "answer_key_counts": counts,
                }
            )
        groups.append(variants)
    groups.append(
        [
            {
                "id": "tail",
                "answer_key_counts": {
                    type_id: {"A": 0, "B": 1} for type_id in type_ids
                },
            }
        ]
    )

    selected = choose_balanced_design_variants(groups, type_ids=type_ids)

    assert [candidate["id"] for candidate in selected] == [
        "g0-v6",
        "g1-v6",
        "g2-v6",
        "g3-v6",
        "g4-v6",
        "tail",
    ]
    with pytest.raises(AnswerBalanceError, match="search_limit_exhausted"):
        choose_balanced_design_variants(
            groups,
            type_ids=type_ids,
            search_state_limit=100,
        )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            Path("/data/avengine_external/review/qa_v2_batch2_questions_v1/questions.json"),
            {"QV2-T4-SIDE": (93, 96), "QV2-T9-CLOSER": (87, 105)},
        ),
        (
            Path("/data/avengine_external/review/qa_v2_pilot48_questions_v1/questions.json"),
            {"QV2-T4-SIDE": (13, 5)},
        ),
    ],
)
def test_retained_question_reports_expose_known_imbalances_read_only(
    path: Path,
    expected: dict[str, tuple[int, int]],
) -> None:
    if not path.is_file():
        pytest.skip("retained server evidence is not installed")
    before = path.stat()
    questions = json.loads(path.read_text(encoding="utf-8"))["questions"]

    report = build_answer_balance_report(questions)

    for type_id, (a_count, b_count) in expected.items():
        item = report["by_type"][type_id]
        assert item["answer_key_counts"] == {"A": a_count, "B": b_count}
        assert item["status"] == "fail"
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)

"""Deterministic exact-48 held-out selection for accepted QA questions.

The selector treats the transitive closure of ``spec.twin_of`` as one
indivisible unit.  Exact point-count and binary-answer feasibility are solved
before a deterministic proportional-distribution refinement is applied.  A
question is never removed independently of its point.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from avengine.qa.answer_balance import (
    KNOWN_BINARY_TYPE_IDS,
    KNOWN_NON_BINARY_ARITIES,
    answer_key_for_question,
)
from avengine.security.path_policy import (
    WorkspacePathPolicy,
    atomic_publish_directory,
)


EXACT_HELDOUT_POINT_COUNT = 48
POINT_DIMENSIONS = (
    "room_id",
    "viewpoint_id",
    "source_asset_pair",
    "pair_kind",
    "motion_case",
    "sound_program",
)
QUESTION_DIMENSIONS = ("question_type", "binary_answer_key")
ALL_DIMENSIONS = POINT_DIMENSIONS + QUESTION_DIMENSIONS


class HeldoutSelectionError(ValueError):
    """The inputs cannot support the required held-out partition."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _required_string(record: Mapping[str, Any], field: str, *, owner: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise HeldoutSelectionError(f"{owner} is missing non-empty {field}")
    return value


def _validate_spec(raw: Any, *, directory_point_id: str) -> dict[str, Any]:
    owner = f"spec {directory_point_id}"
    if not isinstance(raw, Mapping):
        raise HeldoutSelectionError(f"{owner} must be an object")
    record = dict(raw)
    point_id = _required_string(record, "point_id", owner=owner)
    if point_id != directory_point_id:
        raise HeldoutSelectionError(
            f"{owner} point_id does not match its directory: {point_id}"
        )
    for field in (
        "room_id",
        "viewpoint_id",
        "pair_kind",
        "source1_asset",
        "source2_asset",
        "motion_case",
        "program_id",
    ):
        _required_string(record, field, owner=owner)
    if "twin_of" not in record:
        raise HeldoutSelectionError(f"{owner} is missing twin_of")
    twin_of = record["twin_of"]
    if twin_of is not None and (not isinstance(twin_of, str) or not twin_of):
        raise HeldoutSelectionError(f"{owner} twin_of must be null or a point ID")
    if twin_of == point_id:
        raise HeldoutSelectionError(f"{owner} cannot be its own twin")
    if "sound_id" in record and (
        not isinstance(record["sound_id"], str) or not record["sound_id"]
    ):
        raise HeldoutSelectionError(f"{owner} sound_id must be non-empty when present")
    return record


def load_input_specs(inputs_root: str | Path) -> dict[str, dict[str, Any]]:
    """Load one validated ``spec.json`` per immediate point directory."""

    root = Path(inputs_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise HeldoutSelectionError(f"inputs root is not a directory: {root}")
    specs: dict[str, dict[str, Any]] = {}
    for point_directory in sorted(root.iterdir(), key=lambda path: path.name):
        spec_path = point_directory / "spec.json"
        if not spec_path.is_file():
            continue
        try:
            with spec_path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
        except json.JSONDecodeError as exc:
            raise HeldoutSelectionError(f"invalid JSON in {spec_path}: {exc}") from exc
        record = _validate_spec(raw, directory_point_id=point_directory.name)
        point_id = record["point_id"]
        if point_id in specs:
            raise HeldoutSelectionError(f"duplicate spec point_id: {point_id}")
        specs[point_id] = record
    if not specs:
        raise HeldoutSelectionError(f"inputs root contains no point spec.json files: {root}")
    return specs


def load_question_document(path: str | Path) -> dict[str, Any]:
    question_path = Path(path).expanduser().resolve(strict=True)
    if not question_path.is_file():
        raise HeldoutSelectionError(f"questions input is not a file: {question_path}")
    try:
        with question_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise HeldoutSelectionError(
            f"invalid JSON in questions input {question_path}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise HeldoutSelectionError("questions.json must contain an object")
    return dict(raw)


def _validated_questions(
    question_document: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], dict[str, str]]:
    raw_questions = question_document.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise HeldoutSelectionError("questions.json must contain a non-empty questions list")

    questions: list[dict[str, Any]] = []
    type_modes: dict[str, str] = {}
    points_with_questions: set[str] = set()
    for index, raw in enumerate(raw_questions):
        owner = f"question row {index}"
        if not isinstance(raw, Mapping):
            raise HeldoutSelectionError(f"{owner} must be an object")
        row = dict(raw)
        point_id = _required_string(row, "point_id", owner=owner)
        type_id = _required_string(row, "type_id", owner=owner)
        try:
            spec = specs[point_id]
        except KeyError as exc:
            raise HeldoutSelectionError(
                f"{owner} references point without a spec: {point_id}"
            ) from exc
        for field in ("pair_kind", "motion_case", "viewpoint_id"):
            question_value = _required_string(row, field, owner=owner)
            if question_value != spec[field]:
                raise HeldoutSelectionError(
                    f"{owner} {field} disagrees with spec {point_id}"
                )
        if "twin_of" not in row:
            raise HeldoutSelectionError(f"{owner} is missing twin_of")
        if row["twin_of"] != spec["twin_of"]:
            raise HeldoutSelectionError(
                f"{owner} twin_of disagrees with spec {point_id}"
            )

        arity = _required_string(row, "answer_arity", owner=owner)
        if arity == "binary":
            if answer_key_for_question(row) is None:
                raise HeldoutSelectionError(
                    f"{owner} has an invalid binary options/answer combination"
                )
            mode = "binary"
        else:
            if type_id in KNOWN_BINARY_TYPE_IDS:
                raise HeldoutSelectionError(
                    f"{owner} declares known binary type {type_id} as {arity}"
                )
            if arity not in KNOWN_NON_BINARY_ARITIES:
                raise HeldoutSelectionError(
                    f"{owner} has unsupported answer_arity: {arity}"
                )
            mode = "non_binary"
        prior_mode = type_modes.setdefault(type_id, mode)
        if prior_mode != mode:
            raise HeldoutSelectionError(
                f"question type {type_id} mixes binary and non-binary rows"
            )
        questions.append(row)
        points_with_questions.add(point_id)

    questionless = sorted(set(specs) - points_with_questions)
    if questionless:
        raise HeldoutSelectionError(
            "every input point must have at least one accepted question; missing="
            f"{questionless[:8]}"
        )
    questions.sort(
        key=lambda row: (
            row["point_id"],
            row["type_id"],
            _canonical_json(row),
        )
    )
    return tuple(questions), type_modes


def _transitive_twin_groups(
    specs: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], ...]:
    parent = {point_id: point_id for point_id in specs}

    def find(point_id: str) -> str:
        root = point_id
        while parent[root] != root:
            root = parent[root]
        while parent[point_id] != point_id:
            next_id = parent[point_id]
            parent[point_id] = root
            point_id = next_id
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for point_id in sorted(specs):
        twin_of = specs[point_id]["twin_of"]
        if twin_of is None:
            continue
        if twin_of not in specs:
            raise HeldoutSelectionError(
                f"spec {point_id} twin_of references missing point: {twin_of}"
            )
        union(point_id, twin_of)

    components: dict[str, list[str]] = defaultdict(list)
    for point_id in sorted(specs):
        components[find(point_id)].append(point_id)
    return tuple(
        sorted(
            (tuple(sorted(members)) for members in components.values()),
            key=lambda members: members,
        )
    )


def _point_categories(spec: Mapping[str, Any]) -> dict[str, str]:
    assets = sorted((spec["source1_asset"], spec["source2_asset"]))
    sound_id = spec.get("sound_id")
    return {
        "room_id": spec["room_id"],
        "viewpoint_id": spec["viewpoint_id"],
        "source_asset_pair": _canonical_json(
            {"pair_kind": spec["pair_kind"], "assets": assets}
        ),
        "pair_kind": spec["pair_kind"],
        "motion_case": spec["motion_case"],
        "sound_program": _canonical_json(
            {"sound_id": sound_id, "program_id": spec["program_id"]}
        ),
    }


def _question_categories(question: Mapping[str, Any]) -> dict[str, str]:
    categories = {"question_type": question["type_id"]}
    if question["answer_arity"] == "binary":
        answer_key = answer_key_for_question(question)
        if answer_key is None:  # guarded by _validated_questions
            raise AssertionError("validated binary question lost its answer key")
        categories["binary_answer_key"] = f"{question['type_id']}:{answer_key}"
    return categories


class _Group:
    __slots__ = (
        "group_id",
        "point_ids",
        "point_count",
        "binary_delta",
        "strata",
    )

    def __init__(
        self,
        *,
        point_ids: tuple[str, ...],
        binary_delta: tuple[int, ...],
        strata: Counter[tuple[str, str]],
    ) -> None:
        self.group_id = point_ids[0]
        self.point_ids = point_ids
        self.point_count = len(point_ids)
        self.binary_delta = binary_delta
        self.strata = tuple(sorted(strata.items()))


def _build_groups(
    specs: Mapping[str, Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
    binary_types: Sequence[str],
) -> tuple[tuple[_Group, ...], Counter[tuple[str, str]]]:
    questions_by_point: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for question in questions:
        questions_by_point[question["point_id"]].append(question)

    population_strata: Counter[tuple[str, str]] = Counter()
    point_strata: dict[str, Counter[tuple[str, str]]] = {}
    point_delta: dict[str, tuple[int, ...]] = {}
    for point_id in sorted(specs):
        strata: Counter[tuple[str, str]] = Counter()
        for dimension, category in _point_categories(specs[point_id]).items():
            strata[(dimension, category)] += 1
        deltas = Counter()
        for question in questions_by_point[point_id]:
            for dimension, category in _question_categories(question).items():
                strata[(dimension, category)] += 1
            if question["answer_arity"] == "binary":
                key = answer_key_for_question(question)
                deltas[question["type_id"]] += 1 if key == "A" else -1
        point_strata[point_id] = strata
        point_delta[point_id] = tuple(deltas[type_id] for type_id in binary_types)
        population_strata.update(strata)

    groups: list[_Group] = []
    for members in _transitive_twin_groups(specs):
        strata: Counter[tuple[str, str]] = Counter()
        delta = [0] * len(binary_types)
        for point_id in members:
            strata.update(point_strata[point_id])
            for index, value in enumerate(point_delta[point_id]):
                delta[index] += value
        groups.append(
            _Group(
                point_ids=members,
                binary_delta=tuple(delta),
                strata=strata,
            )
        )
    groups.sort(key=lambda group: group.point_ids)
    return tuple(groups), population_strata


def _answer_counts(
    questions: Sequence[Mapping[str, Any]], binary_types: Sequence[str]
) -> dict[str, Counter[str]]:
    counts = {type_id: Counter() for type_id in binary_types}
    for question in questions:
        if question["answer_arity"] != "binary":
            continue
        key = answer_key_for_question(question)
        if key is None:
            raise AssertionError("validated binary question lost its answer key")
        counts[question["type_id"]][key] += 1
    return counts


def _allowed_heldout_deltas(
    population_answers: Mapping[str, Counter[str]],
    binary_types: Sequence[str],
) -> tuple[frozenset[int], ...]:
    allowed: list[frozenset[int]] = []
    failures: list[str] = []
    for type_id in binary_types:
        counts = population_answers[type_id]
        population_delta = counts["A"] - counts["B"]
        if abs(population_delta) > 1:
            failures.append(
                f"{type_id}(A={counts['A']},B={counts['B']},diff={population_delta})"
            )
            continue
        values = frozenset(
            heldout_delta
            for heldout_delta in (-1, 0, 1)
            if abs(population_delta - heldout_delta) <= 1
        )
        allowed.append(values)
    if failures:
        raise HeldoutSelectionError(
            "population questions are not balanced to A/B difference <= 1: "
            + ", ".join(failures)
        )
    return tuple(allowed)


def _point_count_reachable(groups: Sequence[_Group], target: int) -> bool:
    reachable = 1
    mask = (1 << (target + 1)) - 1
    for group in groups:
        reachable |= reachable << group.point_count
        reachable &= mask
    return bool(reachable & (1 << target))


def _milp_selection(
    groups: Sequence[_Group],
    *,
    allowed_deltas: Sequence[frozenset[int]],
    population_counts: Mapping[tuple[str, str], int],
    population_points: int,
    target: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Solve hard constraints and the proportional objective in one MILP.

    Group variables are binary.  Per-category continuous auxiliaries encode
    absolute deviation from the population-proportional expected count.  The
    primary objective is integer-valued after category-count scaling; a
    secondary group-rank sum below one therefore cannot worsen that primary
    optimum and makes the result stable after canonical input sorting.
    """

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:  # pragma: no cover - server native env supplies it
        raise HeldoutSelectionError(
            "exact held-out optimization requires scipy.optimize.milp; "
            "install the AVEngine native dependencies"
        ) from exc

    cells = tuple(sorted(population_counts))
    if not cells:
        raise HeldoutSelectionError("population has no stratification categories")
    dimension_category_counts = Counter(dimension for dimension, _category in cells)
    dimension_weight_lcm = math.lcm(*dimension_category_counts.values())
    group_count = len(groups)
    variable_count = group_count + len(cells)
    objective = np.zeros(variable_count, dtype=np.float64)

    maximum_rank_sum = group_count * (group_count + 1) // 2
    rank_denominator = maximum_rank_sum + 1
    for group_index in range(group_count):
        objective[group_index] = (group_index + 1) / rank_denominator
    for cell_index, (dimension, _category) in enumerate(cells):
        objective[group_count + cell_index] = (
            dimension_weight_lcm // dimension_category_counts[dimension]
        )

    rows: list[np.ndarray] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    point_row = np.zeros(variable_count, dtype=np.float64)
    point_row[:group_count] = [group.point_count for group in groups]
    rows.append(point_row)
    lower_bounds.append(float(target))
    upper_bounds.append(float(target))

    for type_index, allowed in enumerate(allowed_deltas):
        if not allowed or max(allowed) - min(allowed) + 1 != len(allowed):
            raise AssertionError("held-out answer delta set must be a non-empty interval")
        answer_row = np.zeros(variable_count, dtype=np.float64)
        answer_row[:group_count] = [
            group.binary_delta[type_index] for group in groups
        ]
        rows.append(answer_row)
        lower_bounds.append(float(min(allowed)))
        upper_bounds.append(float(max(allowed)))

    group_strata = [dict(group.strata) for group in groups]
    for cell_index, cell in enumerate(cells):
        expected_scaled = target * population_counts[cell]
        positive_row = np.zeros(variable_count, dtype=np.float64)
        positive_row[:group_count] = [
            population_points * strata.get(cell, 0) for strata in group_strata
        ]
        positive_row[group_count + cell_index] = -1.0
        rows.append(positive_row)
        lower_bounds.append(-np.inf)
        upper_bounds.append(float(expected_scaled))

        negative_row = -positive_row
        negative_row[group_count + cell_index] = -1.0
        rows.append(negative_row)
        lower_bounds.append(-np.inf)
        upper_bounds.append(float(-expected_scaled))

    variable_lower = np.zeros(variable_count, dtype=np.float64)
    variable_upper = np.full(variable_count, np.inf, dtype=np.float64)
    variable_upper[:group_count] = 1.0
    integrality = np.zeros(variable_count, dtype=np.int8)
    integrality[:group_count] = 1
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(
            np.vstack(rows),
            np.asarray(lower_bounds),
            np.asarray(upper_bounds),
        ),
        options={"presolve": True},
    )
    if result.status == 2:
        raise HeldoutSelectionError(
            "an exact 48-point whole-twin-group selection exists, but "
            "per-binary-type A/B balance is impossible"
        )
    if not result.success or result.x is None:
        raise HeldoutSelectionError(
            "held-out optimization solver failed without proving mathematical "
            f"infeasibility: status={result.status}, message={result.message}"
        )

    selected = tuple(
        index for index, value in enumerate(result.x[:group_count]) if value >= 0.5
    )
    if sum(groups[index].point_count for index in selected) != target:
        raise AssertionError("MILP result lost the exact point-count constraint")
    selected_delta = [0] * len(allowed_deltas)
    for group_index in selected:
        for type_index, contribution in enumerate(groups[group_index].binary_delta):
            selected_delta[type_index] += contribution
    if any(
        value not in allowed
        for value, allowed in zip(selected_delta, allowed_deltas)
    ):
        raise AssertionError("MILP result lost the binary-answer constraint")

    heldout_counts = _selection_counts(selected, groups)
    primary_objective = 0
    for dimension, category in cells:
        residual = abs(
            population_points * heldout_counts[(dimension, category)]
            - target * population_counts[(dimension, category)]
        )
        primary_objective += (
            dimension_weight_lcm // dimension_category_counts[dimension]
        ) * residual
    rank_sum = sum(index + 1 for index in selected)
    solver_report = {
        "status": "optimal",
        "method": "scipy.optimize.milp_highs",
        "objective": (
            "global minimum equal-dimension mean absolute deviation from "
            "population-proportional expected counts"
        ),
        "primary_scaled_absolute_deviation": primary_objective,
        "secondary_sorted_group_rank_sum": rank_sum,
        "stable_tie_break": (
            "minimum sorted group-rank sum after the integer-valued primary optimum"
        ),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_objective_with_tie_break": float(result.fun),
        "solver_mip_gap": (
            None if getattr(result, "mip_gap", None) is None else float(result.mip_gap)
        ),
        "solver_mip_node_count": (
            None
            if getattr(result, "mip_node_count", None) is None
            else int(result.mip_node_count)
        ),
    }
    return selected, solver_report


def _selection_counts(
    selected: Sequence[int], groups: Sequence[_Group]
) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for index in selected:
        counts.update(dict(groups[index].strata))
    return counts


def _selected_point_key(
    selected: Sequence[int], groups: Sequence[_Group]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            point_id
            for index in selected
            for point_id in groups[index].point_ids
        )
    )
def _counts_by_type(questions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(question["type_id"] for question in questions).items()))


def _binary_balance_report(
    population: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
    train: Sequence[Mapping[str, Any]],
    binary_types: Sequence[str],
) -> dict[str, Any]:
    population_counts = _answer_counts(population, binary_types)
    heldout_counts = _answer_counts(heldout, binary_types)
    train_counts = _answer_counts(train, binary_types)
    by_type: dict[str, Any] = {}
    for type_id in binary_types:
        sections = {}
        status = "pass"
        for name, counts_by_type in (
            ("population", population_counts),
            ("heldout", heldout_counts),
            ("train_eligible", train_counts),
        ):
            counts = counts_by_type[type_id]
            difference = abs(counts["A"] - counts["B"])
            sections[name] = {
                "A": counts["A"],
                "B": counts["B"],
                "difference": difference,
            }
            if difference > 1:
                status = "fail"
        by_type[type_id] = {**sections, "status": status}
    return {
        "policy": "A/B difference <= 1 for every binary type in each partition",
        "status": (
            "pass" if all(item["status"] == "pass" for item in by_type.values()) else "fail"
        ),
        "by_type": by_type,
    }


def _distribution_report(
    population_counts: Mapping[tuple[str, str], int],
    heldout_counts: Mapping[tuple[str, str], int],
) -> dict[str, Any]:
    train_counts = Counter(population_counts)
    train_counts.subtract(heldout_counts)
    report: dict[str, Any] = {}
    for dimension in ALL_DIMENSIONS:
        categories = sorted(
            category
            for candidate_dimension, category in population_counts
            if candidate_dimension == dimension
        )
        population_total = sum(
            population_counts[(dimension, category)] for category in categories
        )
        heldout_total = sum(
            heldout_counts.get((dimension, category), 0) for category in categories
        )
        train_total = population_total - heldout_total
        category_report: dict[str, Any] = {}
        absolute_errors: list[float] = []
        for category in categories:
            population_count = population_counts[(dimension, category)]
            heldout_count = heldout_counts.get((dimension, category), 0)
            train_count = train_counts[(dimension, category)]
            population_fraction = population_count / max(population_total, 1)
            heldout_fraction = heldout_count / max(heldout_total, 1)
            train_fraction = train_count / max(train_total, 1)
            error = abs(heldout_fraction - population_fraction)
            absolute_errors.append(error)
            category_report[category] = {
                "population_count": population_count,
                "population_fraction": population_fraction,
                "heldout_count": heldout_count,
                "heldout_fraction": heldout_fraction,
                "train_eligible_count": train_count,
                "train_eligible_fraction": train_fraction,
                "heldout_absolute_fraction_error": error,
            }
        report[dimension] = {
            "scope": "points" if dimension in POINT_DIMENSIONS else "questions",
            "population_total": population_total,
            "heldout_total": heldout_total,
            "train_eligible_total": train_total,
            "total_variation_distance_population_vs_heldout": (
                0.5 * math.fsum(absolute_errors)
            ),
            "categories": category_report,
        }
    return report


def build_selection_bundle(
    question_document: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
    *,
    generated_at: str | None = None,
    source_questions_json: str | None = None,
    source_inputs_root: str | None = None,
) -> dict[str, Any]:
    """Build the exact-48 partition and its proof/report documents."""

    normalized_specs: dict[str, dict[str, Any]] = {}
    for point_id in sorted(specs):
        normalized_specs[point_id] = _validate_spec(
            specs[point_id], directory_point_id=point_id
        )
    questions, type_modes = _validated_questions(question_document, normalized_specs)
    binary_types = tuple(
        sorted(type_id for type_id, mode in type_modes.items() if mode == "binary")
    )
    population_answers = _answer_counts(questions, binary_types)
    groups, population_strata = _build_groups(
        normalized_specs, questions, binary_types
    )
    population_points = len(normalized_specs)
    if population_points < EXACT_HELDOUT_POINT_COUNT:
        raise HeldoutSelectionError(
            f"population has {population_points} points; exact 48-point selection is impossible"
        )
    if not _point_count_reachable(groups, EXACT_HELDOUT_POINT_COUNT):
        raise HeldoutSelectionError(
            "exact 48-point selection is impossible with whole transitive twin groups"
        )
    allowed_deltas = _allowed_heldout_deltas(population_answers, binary_types)
    selected, optimization_report = _milp_selection(
        groups,
        allowed_deltas=allowed_deltas,
        population_counts=population_strata,
        population_points=population_points,
        target=EXACT_HELDOUT_POINT_COUNT,
    )

    heldout_point_ids = list(_selected_point_key(selected, groups))
    heldout_points = set(heldout_point_ids)
    train_point_ids = sorted(set(normalized_specs) - heldout_points)
    heldout_questions = [
        dict(question) for question in questions if question["point_id"] in heldout_points
    ]
    train_questions = [
        dict(question) for question in questions if question["point_id"] not in heldout_points
    ]
    heldout_counts = _selection_counts(selected, groups)

    point_overlap = sorted(heldout_points & set(train_point_ids))
    heldout_question_counter = Counter(
        _canonical_json(question) for question in heldout_questions
    )
    train_question_counter = Counter(_canonical_json(question) for question in train_questions)
    question_overlap_count = sum(
        min(heldout_question_counter[key], train_question_counter[key])
        for key in heldout_question_counter.keys() & train_question_counter.keys()
    )
    question_partition_complete = (
        Counter(_canonical_json(question) for question in questions)
        == heldout_question_counter + train_question_counter
    )
    components = _transitive_twin_groups(normalized_specs)
    split_twin_groups = [
        list(component)
        for component in components
        if heldout_points.intersection(component)
        and set(train_point_ids).intersection(component)
    ]
    if (
        len(heldout_point_ids) != EXACT_HELDOUT_POINT_COUNT
        or point_overlap
        or question_overlap_count
        or not question_partition_complete
        or split_twin_groups
    ):
        raise AssertionError("selector produced an invalid partition")

    binary_balance = _binary_balance_report(
        questions, heldout_questions, train_questions, binary_types
    )
    if binary_balance["status"] != "pass":
        raise AssertionError("selector lost the binary answer-balance invariant")

    generated = generated_at or datetime.now(timezone.utc).isoformat()
    common = {
        "status": "research_only",
        "research_only": True,
        "episode_counted": False,
        "formal_episode_count": 0,
        "qualification_claim": False,
        "generated_at": generated,
        "pilot48_role": "weak_probe_only_not_primary_heldout",
    }
    heldout_document = {
        **common,
        "selection_role": "heldout_exact_48_points",
        "point_ids": heldout_point_ids,
        "counts": {
            "points": len(heldout_point_ids),
            "questions": len(heldout_questions),
            "questions_by_type": _counts_by_type(heldout_questions),
        },
        "questions": heldout_questions,
    }
    train_document = {
        **common,
        "selection_role": "train_eligible_excluding_heldout_points",
        "point_ids": train_point_ids,
        "counts": {
            "points": len(train_point_ids),
            "questions": len(train_questions),
            "questions_by_type": _counts_by_type(train_questions),
        },
        "questions": train_questions,
    }
    point_document = {
        **common,
        "selection_role": "heldout_exact_48_points",
        "heldout_point_count": len(heldout_point_ids),
        "point_ids": heldout_point_ids,
        "selected_transitive_twin_groups": [
            list(groups[index].point_ids) for index in selected
        ],
    }
    report = {
        **common,
        "claim_boundary": (
            "Exact 48-point research held-out partition only; pilot48 remains a "
            "weak probe and no Episode enters the formal denominator."
        ),
        "source": {
            "questions_json": source_questions_json,
            "inputs_root": source_inputs_root,
        },
        "counts": {
            "population_points": population_points,
            "heldout_points": len(heldout_point_ids),
            "train_eligible_points": len(train_point_ids),
            "population_questions": len(questions),
            "heldout_questions": len(heldout_questions),
            "train_eligible_questions": len(train_questions),
            "transitive_twin_groups": len(groups),
        },
        "constraints": {
            "exact_point_count": "pass",
            "whole_transitive_twin_groups": "pass",
            "point_level_question_deletion_used": False,
            "binary_answer_balance": "pass",
        },
        "partition_proof": {
            "point_intersection": point_overlap,
            "question_intersection_count": question_overlap_count,
            "question_partition_complete": question_partition_complete,
            "split_transitive_twin_groups": split_twin_groups,
            "status": "pass",
        },
        "binary_answer_balance": binary_balance,
        "optimization": optimization_report,
        "distributions": _distribution_report(population_strata, heldout_counts),
    }
    return {
        "heldout_point_ids": point_document,
        "heldout_questions": heldout_document,
        "train_eligible_questions": train_document,
        "selection_report": report,
    }


def publish_selection_bundle(bundle: Mapping[str, Any], output: str | Path) -> Path:
    """Publish all four documents through a sibling no-replace directory."""

    unresolved = Path(output).expanduser()
    if not unresolved.is_absolute():
        raise HeldoutSelectionError("output must be an absolute fresh directory path")
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    parent = unresolved.parent.resolve(strict=True)
    policy = WorkspacePathPolicy.from_roots((parent,))
    destination = policy.resolve_output(
        parent / unresolved.name, owner="heldout selection output directory"
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent)
    )
    files = {
        "heldout_point_ids.json": "heldout_point_ids",
        "heldout_questions.json": "heldout_questions",
        "train_eligible_questions.json": "train_eligible_questions",
        "selection_report.json": "selection_report",
    }
    try:
        for filename, key in files.items():
            with (staging / filename).open("x", encoding="utf-8") as handle:
                json.dump(bundle[key], handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
        return atomic_publish_directory(policy, staging, destination)
    except BaseException:
        if os.path.lexists(staging):
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select an exact 48-point, twin-isolated QA held-out partition."
    )
    parser.add_argument("--questions-json", required=True)
    parser.add_argument("--inputs-root", required=True)
    parser.add_argument("--output", required=True, help="fresh absolute output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        questions_path = Path(args.questions_json).expanduser().resolve(strict=True)
        inputs_root = Path(args.inputs_root).expanduser().resolve(strict=True)
        question_document = load_question_document(questions_path)
        specs = load_input_specs(inputs_root)
        bundle = build_selection_bundle(
            question_document,
            specs,
            source_questions_json=str(questions_path),
            source_inputs_root=str(inputs_root),
        )
        published = publish_selection_bundle(bundle, args.output)
    except (HeldoutSelectionError, FileExistsError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"error": str(exc), "output_written": False}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "output": str(published),
                "heldout_points": EXACT_HELDOUT_POINT_COUNT,
                "heldout_questions": bundle["selection_report"]["counts"][
                    "heldout_questions"
                ],
                "status": "research_only",
                "episode_counted": False,
                "formal_episode_count": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from avengine.qa import heldout_selector as selector


def _point_spec(
    point_id: str,
    *,
    group_index: int,
    member_index: int,
    twin_of: str | None,
) -> dict[str, object]:
    pair_kind = "human" if group_index % 2 == 0 else "dog"
    asset_pair = (
        (f"asset_{group_index % 4}_a", f"asset_{group_index % 4}_b")
        if member_index % 2 == 0
        else (f"asset_{group_index % 4}_b", f"asset_{group_index % 4}_a")
    )
    return {
        "point_id": point_id,
        "room_id": f"room_{group_index % 2}",
        "viewpoint_id": f"view_{group_index % 4}",
        "pair_kind": pair_kind,
        "source1_asset": asset_pair[0],
        "source2_asset": asset_pair[1],
        "motion_case": f"motion_{group_index % 4}",
        "sound_id": f"sound_{group_index % 3}",
        "program_id": f"program_{group_index % 3}",
        "twin_of": twin_of,
    }


def _question_base(spec: dict[str, object]) -> dict[str, object]:
    return {
        "point_id": spec["point_id"],
        "pair_kind": spec["pair_kind"],
        "motion_case": spec["motion_case"],
        "viewpoint_id": spec["viewpoint_id"],
        "twin_of": spec["twin_of"],
    }


def _binary_question(
    spec: dict[str, object], *, type_id: str, answer_key: str
) -> dict[str, object]:
    options = (
        ["left", "right"]
        if type_id == "T-SIDE"
        else ["moving", "staying still"]
    )
    return {
        **_question_base(spec),
        "type_id": type_id,
        "answer_arity": "binary",
        "options": options,
        "answer": options[0 if answer_key == "A" else 1],
    }


def _numeric_question(spec: dict[str, object], value: float) -> dict[str, object]:
    return {
        **_question_base(spec),
        "type_id": "T-NUMERIC",
        "answer_arity": "numeric_banded",
        "options": None,
        "answer": value,
    }


def _open_question(spec: dict[str, object], type_id: str) -> dict[str, object]:
    return {
        **_question_base(spec),
        "type_id": type_id,
        "answer_arity": "open",
        "options": None,
        "answer": type_id,
    }


def _population(
    group_sizes: list[int],
    *,
    uniform_group_keys: list[str] | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, object]], list[list[str]]]:
    specs: dict[str, dict[str, object]] = {}
    questions: list[dict[str, object]] = []
    groups: list[list[str]] = []
    point_number = 0
    for group_index, group_size in enumerate(group_sizes):
        group: list[str] = []
        prior_point: str | None = None
        for member_index in range(group_size):
            point_id = f"p{point_number:03d}"
            point_number += 1
            spec = _point_spec(
                point_id,
                group_index=group_index,
                member_index=member_index,
                twin_of=prior_point,
            )
            specs[point_id] = spec
            group.append(point_id)
            prior_point = point_id
            if uniform_group_keys is not None:
                side_key = uniform_group_keys[group_index]
            elif group_size % 2 == 0:
                side_key = "A" if member_index % 2 == 0 else "B"
            elif group_index % 2 == 0:
                side_key = "A" if member_index % 2 == 0 else "B"
            else:
                side_key = "B" if member_index % 2 == 0 else "A"
            motion_key = "B" if side_key == "A" else "A"
            questions.extend(
                (
                    _binary_question(spec, type_id="T-SIDE", answer_key=side_key),
                    _binary_question(spec, type_id="T-MOTION", answer_key=motion_key),
                    _numeric_question(spec, float(point_number)),
                )
            )
        groups.append(group)
    return {"questions": questions}, specs, groups


def _canonical_rows(rows: list[dict[str, object]]) -> Counter[str]:
    return Counter(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )


def _write_inputs(
    root: Path,
    question_document: dict[str, object],
    specs: dict[str, dict[str, object]],
) -> Path:
    questions_path = root / "questions.json"
    questions_path.write_text(json.dumps(question_document), encoding="utf-8")
    inputs_root = root / "inputs"
    for point_id, spec in specs.items():
        point_dir = inputs_root / point_id
        point_dir.mkdir(parents=True)
        (point_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    return questions_path


def test_selects_exact_48_and_reports_all_stratification_dimensions() -> None:
    question_document, specs, _groups = _population([1] * 64)

    bundle = selector.build_selection_bundle(
        question_document, specs, generated_at="fixed"
    )

    point_ids = bundle["heldout_point_ids"]["point_ids"]
    report = bundle["selection_report"]
    assert len(point_ids) == 48
    assert len(set(point_ids)) == 48
    assert set(report["distributions"]) == set(selector.ALL_DIMENSIONS)
    for dimension in selector.POINT_DIMENSIONS:
        assert report["distributions"][dimension]["population_total"] == 64
        assert report["distributions"][dimension]["heldout_total"] == 48
    assert report["optimization"]["status"] == "optimal"
    assert report["optimization"]["primary_scaled_absolute_deviation"] >= 0
    assert report["partition_proof"] == {
        "point_intersection": [],
        "question_intersection_count": 0,
        "question_partition_complete": True,
        "split_transitive_twin_groups": [],
        "status": "pass",
    }
    assert report["research_only"] is True
    assert report["episode_counted"] is False
    assert report["formal_episode_count"] == 0
    assert report["pilot48_role"] == "weak_probe_only_not_primary_heldout"


def test_transitive_twin_components_are_indivisible_and_questions_are_not_deleted() -> None:
    question_document, specs, source_groups = _population([3] * 20)

    bundle = selector.build_selection_bundle(
        question_document, specs, generated_at="fixed"
    )

    heldout_points = set(bundle["heldout_point_ids"]["point_ids"])
    train_points = set(bundle["train_eligible_questions"]["point_ids"])
    assert len(heldout_points) == 48
    assert not heldout_points & train_points
    for group in source_groups:
        selected_members = heldout_points.intersection(group)
        assert len(selected_members) in {0, len(group)}

    source_rows = question_document["questions"]
    heldout_rows = bundle["heldout_questions"]["questions"]
    train_rows = bundle["train_eligible_questions"]["questions"]
    assert _canonical_rows(source_rows) == _canonical_rows(heldout_rows) + _canonical_rows(
        train_rows
    )
    by_source_point = Counter(row["point_id"] for row in source_rows)
    by_heldout_point = Counter(row["point_id"] for row in heldout_rows)
    by_train_point = Counter(row["point_id"] for row in train_rows)
    for point_id, count in by_source_point.items():
        if point_id in heldout_points:
            assert by_heldout_point[point_id] == count
            assert by_train_point[point_id] == 0
        else:
            assert by_train_point[point_id] == count
            assert by_heldout_point[point_id] == 0


def test_every_binary_type_is_balanced_in_heldout_and_train() -> None:
    question_document, specs, _groups = _population([1] * 64)

    report = selector.build_selection_bundle(
        question_document, specs, generated_at="fixed"
    )["selection_report"]["binary_answer_balance"]

    assert report["status"] == "pass"
    assert set(report["by_type"]) == {"T-MOTION", "T-SIDE"}
    for item in report["by_type"].values():
        assert item["population"]["difference"] <= 1
        assert item["heldout"]["difference"] <= 1
        assert item["train_eligible"]["difference"] <= 1


def test_global_proportional_optimum_can_require_a_two_for_two_exchange() -> None:
    type_counts = ((6, 6), (6, 6), (5, 7), (6, 6), (9, 3), (4, 8))
    specs: dict[str, dict[str, object]] = {}
    questions: list[dict[str, object]] = []
    point_number = 0
    for group_index, (left_count, right_count) in enumerate(type_counts):
        prior_point: str | None = None
        for member_index in range(left_count + right_count):
            point_id = f"p{point_number:03d}"
            point_number += 1
            spec = _point_spec(
                point_id,
                group_index=0,
                member_index=member_index,
                twin_of=prior_point,
            )
            specs[point_id] = spec
            prior_point = point_id
            question_type = "Q-LEFT" if member_index < left_count else "Q-RIGHT"
            questions.append(_open_question(spec, question_type))

    bundle = selector.build_selection_bundle(
        {"questions": questions}, specs, generated_at="fixed"
    )

    assert bundle["heldout_questions"]["counts"]["questions_by_type"] == {
        "Q-LEFT": 24,
        "Q-RIGHT": 24,
    }
    assert (
        bundle["selection_report"]["optimization"][
            "primary_scaled_absolute_deviation"
        ]
        == 0
    )


def test_exact_48_impossibility_is_distinct_from_answer_balance_impossibility() -> None:
    count_document, count_specs, _groups = _population(
        [5] * 10, uniform_group_keys=["A"] * 10
    )
    with pytest.raises(
        selector.HeldoutSelectionError,
        match="exact 48-point selection is impossible with whole transitive twin groups",
    ):
        selector.build_selection_bundle(count_document, count_specs)

    balance_document, balance_specs, _groups = _population(
        [48, 48], uniform_group_keys=["A", "B"]
    )
    with pytest.raises(
        selector.HeldoutSelectionError,
        match="whole-twin-group selection exists, but per-binary-type A/B balance is impossible",
    ):
        selector.build_selection_bundle(balance_document, balance_specs)


def test_missing_required_stratification_field_fails_explicitly() -> None:
    question_document, specs, _groups = _population([1] * 64)
    del specs["p000"]["viewpoint_id"]

    with pytest.raises(
        selector.HeldoutSelectionError,
        match="spec p000 is missing non-empty viewpoint_id",
    ):
        selector.build_selection_bundle(question_document, specs)


def test_selection_and_documents_are_stable_under_input_reordering() -> None:
    question_document, specs, _groups = _population([1] * 64)
    reversed_document = {"questions": list(reversed(question_document["questions"]))}
    reversed_specs = dict(reversed(list(specs.items())))

    first = selector.build_selection_bundle(
        question_document, specs, generated_at="fixed"
    )
    second = selector.build_selection_bundle(
        reversed_document, reversed_specs, generated_at="fixed"
    )

    assert first == second


def test_hundreds_point_population_uses_the_same_exact_search() -> None:
    _question_document, specs, _groups = _population([1] * 320)
    questions: list[dict[str, object]] = []
    for point_index, point_id in enumerate(sorted(specs)):
        spec = specs[point_id]
        for type_index in range(6):
            answer_key = (
                "A" if (point_index // (1 << type_index)) % 2 == 0 else "B"
            )
            questions.append(
                _binary_question(
                    spec,
                    type_id=f"T-BINARY-{type_index}",
                    answer_key=answer_key,
                )
            )
        questions.append(_numeric_question(spec, float(point_index)))

    report = selector.build_selection_bundle(
        {"questions": questions}, specs, generated_at="fixed"
    )["selection_report"]

    assert report["counts"]["population_points"] == 320
    assert report["counts"]["heldout_points"] == 48
    assert report["constraints"]["exact_point_count"] == "pass"
    assert report["binary_answer_balance"]["status"] == "pass"
    assert len(report["binary_answer_balance"]["by_type"]) == 6
    assert report["optimization"]["status"] == "optimal"


def test_cli_atomically_publishes_four_files_and_refuses_clobber(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    question_document, specs, _groups = _population([1] * 64)
    questions_path = _write_inputs(tmp_path, question_document, specs)
    output = tmp_path / "heldout48"
    arguments = [
        "--questions-json",
        str(questions_path),
        "--inputs-root",
        str(tmp_path / "inputs"),
        "--output",
        str(output),
    ]

    assert selector.main(arguments) == 0
    assert sorted(path.name for path in output.iterdir()) == [
        "heldout_point_ids.json",
        "heldout_questions.json",
        "selection_report.json",
        "train_eligible_questions.json",
    ]
    sentinel = output / "selection_report.json"
    prior = sentinel.read_bytes()
    assert selector.main(arguments) == 2
    assert sentinel.read_bytes() == prior
    assert list(tmp_path.glob(".heldout48.staging-*")) == []
    assert "output_written" in capsys.readouterr().out


def test_publish_race_keeps_destination_and_cleans_only_its_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    question_document, specs, _groups = _population([1] * 64)
    bundle = selector.build_selection_bundle(
        question_document, specs, generated_at="fixed"
    )
    output = tmp_path / "race-output"
    original_publish = selector.atomic_publish_directory

    def racing_publish(policy, staging, destination):
        destination.mkdir()
        (destination / "sentinel").write_text("keep", encoding="utf-8")
        return original_publish(policy, staging, destination)

    monkeypatch.setattr(selector, "atomic_publish_directory", racing_publish)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        selector.publish_selection_bundle(bundle, output)

    assert (output / "sentinel").read_text(encoding="utf-8") == "keep"
    assert list(tmp_path.glob(".race-output.staging-*")) == []


def test_serialization_failure_cleans_staging_without_publishing(tmp_path: Path) -> None:
    question_document, specs, _groups = _population([1] * 64)
    bundle = selector.build_selection_bundle(
        question_document, specs, generated_at="fixed"
    )
    bundle["selection_report"] = {"not_json": object()}
    output = tmp_path / "serialization-failure"

    with pytest.raises(TypeError):
        selector.publish_selection_bundle(bundle, output)

    assert not output.exists()
    assert list(tmp_path.glob(".serialization-failure.staging-*")) == []

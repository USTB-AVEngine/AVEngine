from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from avengine.qa.binding_catalog_scoring import (
    BindingCatalogScoreError,
    score_binding_catalog,
)


def _mcq_form(gold: str) -> dict:
    options = [
        {"value": "yes", "label_en": "yes", "label_zh": "是"},
        {"value": "no", "label_en": "no", "label_zh": "否"},
    ]
    return {
        "answer_type": "choice",
        "options": options,
        "gold": {"correct_index": options.index(next(
            option for option in options if option["value"] == gold
        )), "value": gold},
    }


def _open_form(gold: str) -> dict:
    return {
        "answer_type": "closed_set",
        "truth": gold,
        "classes": {"yes": ["yes", "是"], "no": ["no", "否"]},
    }


def _item(
    private_id: str,
    qa_id: str,
    *,
    gold: str,
    mcq: bool = True,
    open_form: bool = True,
) -> dict:
    forms = {}
    if mcq:
        forms["mcq"] = _mcq_form(gold)
    if open_form:
        forms["open"] = _open_form(gold)
    return {
        "schema": "avengine_qa_unified_question_v1",
        "status": "pass",
        "qa_id": qa_id,
        "question_id": private_id,
        "forms": forms,
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict, list[str]]:
    questions_dir = tmp_path / "questions"
    questions_dir.mkdir()
    items = [
        _item("private-one", "QA-01", gold="yes"),
        _item("private-two", "QA-02", gold="no", mcq=False),
        _item("private-three", "QA-13", gold="no", open_form=False),
    ]
    followup = _item(
        "private-followup",
        "QA-03",
        gold="no",
        mcq=False,
        open_form=True,
    )
    question_set = {
        "schema": "avengine_qa_question_set_v1",
        "episode_id": "fixture-episode",
        "items": items,
        "angle_followups": [followup],
    }
    questions_path = questions_dir / "sample_000001.json"
    questions_path.write_text(
        json.dumps(question_set, ensure_ascii=False),
        encoding="utf-8",
    )
    public_ids = ["opaque-id-alpha", "opaque-id-beta", "opaque-id-gamma", "opaque-id-delta"]
    catalog = {
        "status": "research_candidate",
        "records": [{
            "sample_id": "sample_000001",
            "group_id": "fixture-group",
            "questions_path": "questions/sample_000001.json",
            "public_question_ids": public_ids,
        }],
    }
    catalog_path = tmp_path / "catalog_index.json"
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False),
        encoding="utf-8",
    )
    return catalog_path, catalog, public_ids


def test_scores_requested_form_with_public_id_mapping_and_qa_metrics(tmp_path: Path) -> None:
    catalog_path, catalog, ids = _fixture(tmp_path)
    result = score_binding_catalog(
        catalog,
        [
            {"question_id": ids[0], "prediction": "yes"},
            {"question_id": ids[1], "prediction": "maybe"},
        ],
        form="open",
        input_base=catalog_path.parent,
    )

    assert result["status"] == "research_candidate"
    assert result["form"] == "open"
    assert result["counts"] == {
        "catalog_items_total": 4,
        "form_available": 3,
        "form_unavailable": 1,
        "predictions_supplied": 2,
        "scored": 1,
        "correct": 1,
        "invalid": 1,
        "missing": 1,
        "abstained": 0,
    }
    assert result["overall"]["denominator"] == 3
    assert result["overall"]["accuracy"] == pytest.approx(1 / 3)
    assert result["overall"]["mean_score_over_all"] == pytest.approx(1 / 3)
    assert result["qa_metrics"]["QA-01"]["correct"] == 1
    assert result["qa_metrics"]["QA-02"]["invalid"] == 1
    assert result["qa_metrics"]["QA-03"]["missing"] == 1
    assert result["qa_metrics"]["QA-13"]["denominator"] == 0
    assert result["qa_metrics"]["QA-13"]["unavailable_form"] == 1
    assert {row["question_id"] for row in result["records"]} == set(ids)
    assert "private-one" not in json.dumps(result, ensure_ascii=False)


def test_mcq_denominator_excludes_open_only_items(tmp_path: Path) -> None:
    catalog_path, catalog, ids = _fixture(tmp_path)
    result = score_binding_catalog(
        catalog_path,
        [
            {"question_id": ids[0], "prediction": "A"},
            {"question_id": ids[2], "prediction": "A"},
        ],
        form="mcq",
    )

    assert result["counts"]["catalog_items_total"] == 4
    assert result["counts"]["form_available"] == 2
    assert result["counts"]["form_unavailable"] == 2
    assert result["counts"]["scored"] == 2
    assert result["counts"]["correct"] == 1
    assert result["counts"]["missing"] == 0
    assert result["overall"]["accuracy"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        (
            [
                {"question_id": "opaque-id-alpha", "prediction": "yes"},
                {"question_id": "opaque-id-alpha", "prediction": "no"},
            ],
            "duplicate",
        ),
        (
            [{"question_id": "unknown-id", "prediction": "yes"}],
            "unknown",
        ),
        (
            [{"question_id": "opaque-id-gamma", "prediction": "yes"}],
            "do not provide the open form",
        ),
    ],
)
def test_rejects_duplicate_unknown_and_wrong_form_ids(
    tmp_path: Path,
    predictions: list[dict[str, str]],
    message: str,
) -> None:
    catalog_path, catalog, _ids = _fixture(tmp_path)
    with pytest.raises(BindingCatalogScoreError, match=message):
        score_binding_catalog(
            catalog,
            predictions,
            form="open",
            input_base=catalog_path.parent,
        )


def test_rejects_public_id_alignment_errors(tmp_path: Path) -> None:
    catalog_path, catalog, _ids = _fixture(tmp_path)
    bad = deepcopy(catalog)
    bad["records"][0]["public_question_ids"] = ["only-one"]
    with pytest.raises(BindingCatalogScoreError, match="align one-to-one"):
        score_binding_catalog(bad, [], form="open", input_base=catalog_path.parent)

    bad = deepcopy(catalog)
    bad["records"][0]["public_question_ids"][1] = (
        bad["records"][0]["public_question_ids"][0]
    )
    with pytest.raises(BindingCatalogScoreError, match="duplicate public"):
        score_binding_catalog(bad, [], form="open", input_base=catalog_path.parent)


def test_cli_writes_scorer_output_without_running_a_model(tmp_path: Path) -> None:
    catalog_path, _catalog, ids = _fixture(tmp_path)
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps([{"question_id": ids[0], "prediction": "yes"}]),
        encoding="utf-8",
    )
    output_path = tmp_path / "score.json"
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "tools/qa/score_binding_catalog.py"),
            "--catalog",
            str(catalog_path),
            "--predictions",
            str(predictions_path),
            "--form",
            "open",
            "--out",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["status"] == "research_candidate"
    assert result["counts"]["missing"] == 2
    assert result["counts"]["invalid"] == 0
    assert result["qualification_claim"] is False
    assert "does not run a model" in result["claim_boundary"]


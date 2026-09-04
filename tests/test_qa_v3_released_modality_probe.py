"""Focused tests for released-media shortcut probe inputs and scoring."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from build_qa_v3_released_probe_items import build  # noqa: E402
from probe_released_modality_shortcuts import run  # noqa: E402


PARAMS = {"THETA_FULL": 15.0, "THETA_HALF": 30.0,
          "T_FULL": 0.3, "T_HALF": 1.0}
TOOL = TOOLS / "build_qa_v3_released_probe_items.py"


def _released_fixture(tmp_path, open_block, *, point="card8_001",
                      selection_extra=None):
    facts, audio, media = tmp_path / "facts", tmp_path / "audio", tmp_path / "media"
    (facts / point).mkdir(parents=True)
    (audio / point / "audio/binaural").mkdir(parents=True)
    (media / point).mkdir(parents=True)
    (audio / point / "audio/binaural/mixture.wav").write_bytes(b"wav")
    (media / point / "video_only.mp4").write_bytes(b"mp4")
    fact = {
        "profile_id": "card8",
        "episode_id": f"episode-{point}",
        "target_first": True,
        "mcq": {
            "stem": "when?",
            "options_space": ["a", "b"],
            "truth_option": "a",
        },
        "open": open_block,
    }
    (facts / point / "fact_record.json").write_text(
        json.dumps(fact), encoding="utf-8")
    selection = {
        "selected": [{"point_id": point, "profile_id": "card8"}],
    }
    if selection_extra:
        selection.update(selection_extra)
    return selection, facts, audio, media, point


def test_text_probe_reports_empirical_not_universal_claim():
    items = []
    for index in range(8):
        label = "left" if index % 2 == 0 else "right"
        items.append({
            "question_id": f"q{index}", "group_id": f"q{index}",
            "profile_id": "p", "form": "mcq",
            "task_type": "classification", "question": f"token {label}",
            "options": ["left", "right"], "truth": label,
        })
    result = run(items, "text", PARAMS, folds=4)
    assert result["records"][0]["accuracy"] >= 0.75
    assert result["qualification_claim"] is False
    assert "does not prove" in result["boundary"]


def test_probe_output_embeds_executed_scoring_params_and_fails_closed():
    items = [{
        "question_id": f"q{index}", "group_id": f"q{index}",
        "profile_id": "p", "form": "mcq", "task_type": "classification",
        "question": "same", "options": ["a", "b"], "truth": "a",
    } for index in range(4)]
    result = run(items, "text", PARAMS, folds=2)
    assert result["scoring_params"]["T_FULL"] == 0.3
    assert result["scoring_params"]["T_HALF"] == 1.0
    assert result["scoring_params"]["time_certification_policy"] == \
        "strict_full_credit_only"
    assert result["scoring_params"]["T_FULL_status"] == \
        "unspecified_treat_as_placeholder"
    import pytest
    with pytest.raises(ValueError, match="T_FULL"):
        run(items, "text", {k: v for k, v in PARAMS.items() if k != "T_FULL"},
            folds=2)


def test_numeric_time_probe_uses_strict_scorer():
    items = []
    for index in range(6):
        items.append({
            "question_id": f"q{index}", "group_id": f"q{index}",
            "profile_id": "card8", "form": "open",
            "task_type": "numeric_time", "question": "same",
            "options": [], "truth": 1.0 + 0.5 * index,
        })
    result = run(items, "text", PARAMS, folds=3)
    record = result["records"][0]
    assert 0.0 <= record["mean_scorer_score"] <= 1.0
    assert 0.0 <= record["empirical_constant_baseline"] <= 1.0


def test_item_builder_uses_only_released_paths_and_question_gold(tmp_path):
    selection, facts, audio, media, point = _released_fixture(
        tmp_path,
        {
            "stem": "seconds?",
            "truth_value": 2.4,
            "scoring": "absolute_time",
            "certification_policy": "strict_full_credit_only",
        },
    )
    rows = build(selection, facts, audio, media)
    assert len(rows) == 2
    assert rows[0]["question_id"] == f"{point}__mcq"
    assert rows[1]["question_id"] == f"{point}__open"
    assert rows[0]["question_id"] != rows[1]["question_id"]
    assert rows[0]["group_id"] == rows[1]["group_id"] == f"episode-{point}"
    assert rows[1]["task_type"] == "numeric_time"
    assert rows[1]["certification_policy"] == "strict_full_credit_only"
    forbidden = {
        "timeline", "dry", "rir", "engine_frame_note",
        "azimuth_deg_engine_frame", "azimuth_interval_engine_frame",
        "query_window_seconds",
    }
    assert all(forbidden.isdisjoint(row) for row in rows)


def test_interval_angle_keeps_authoritative_interval_and_dcase_convention(tmp_path):
    selection, facts, audio, media, point = _released_fixture(
        tmp_path,
        {
            "stem": "angle?",
            "truth_value": 175.0,
            "truth_interval_deg": [172.0, 178.0],
            "truth_value_note": "midpoint of truth_interval_deg",
            "unit": "deg",
            "scoring": "circular_deg_interval",
            "convention": "dcase_foa_left_positive",
            "certification_policy": "strict_full_credit_only",
        },
        point="card1F_001",
    )
    rows = build(selection, facts, audio, media)
    open_row = next(row for row in rows if row["form"] == "open")
    assert open_row["task_type"] == "numeric_angle"
    assert open_row["truth"] == 175.0
    assert open_row["truth_interval_deg"] == [172.0, 178.0]
    assert open_row["convention"] == "dcase_foa_left_positive"
    assert open_row["certification_policy"] == "strict_full_credit_only"
    assert open_row["question_id"] == f"{point}__open"


def test_answer_form_selection_filters_single_form_and_explicit_wins(tmp_path):
    selection, facts, audio, media, _ = _released_fixture(
        tmp_path,
        {
            "stem": "seconds?",
            "truth_value": 2.4,
            "scoring": "absolute_time",
        },
        selection_extra={"ANSWER_FORMS_DEFAULT": ["open"]},
    )
    rows = build(selection, facts, audio, media)
    assert [row["form"] for row in rows] == ["open"]

    rows = build(
        selection,
        facts,
        audio,
        media,
        answer_forms=["mcq"],
        params={"ANSWER_FORMS_DEFAULT": ["open"]},
    )
    assert [row["form"] for row in rows] == ["mcq"]

    selection.pop("ANSWER_FORMS_DEFAULT")
    rows = build(
        selection,
        facts,
        audio,
        media,
        params={"ANSWER_FORMS_DEFAULT": ["open"]},
    )
    assert [row["form"] for row in rows] == ["open"]


def test_cli_accepts_explicit_form_and_params(tmp_path):
    selection, facts, audio, media, point = _released_fixture(
        tmp_path,
        {
            "stem": "seconds?",
            "truth_value": 2.4,
            "scoring": "absolute_time",
        },
    )
    selection_path = tmp_path / "selection.json"
    params_path = tmp_path / "params.json"
    output_path = tmp_path / "items.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    params_path.write_text(
        json.dumps({"ANSWER_FORMS_DEFAULT": ["mcq", "open"]}),
        encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable, str(TOOL),
            "--selection-manifest", str(selection_path),
            "--facts-root", str(facts),
            "--audio-root", str(audio),
            "--media-root", str(media),
            "--output", str(output_path),
            "--params", str(params_path),
            "--answer-form", "open",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout)["record_count"] == 1
    rows = json.loads(output_path.read_text(encoding="utf-8"))
    assert rows[0]["question_id"] == f"{point}__open"


def test_numeric_probe_keeps_strict_policy_and_authoritative_interval():
    import probe_released_modality_shortcuts as probe_module
    params = {"THETA_FULL": 15.0, "THETA_HALF": 30.0, "T_FULL": 0.5, "T_HALF": 1.0,
              "ANGLE_CERTIFICATION_POLICY": "strict_full_credit_only"}
    _, scores = probe_module._numeric_score([20.0], [0.0], "numeric_angle", params)
    assert scores == [0.0]
    _, scores = probe_module._numeric_score(
        [20.0], [0.0], "numeric_angle", params,
        [{"truth_interval_deg": [10.0, 20.0], "convention": "dcase_foa_left_positive",
          "certification_policy": "strict_full_credit_only"}])
    assert scores == [1.0]
    assert probe_module.scoring_snapshot(params)["angle_policy"] == "strict_full_credit_only"

"""Focused tests for released-media shortcut probe inputs and scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from build_qa_v3_released_probe_items import build  # noqa: E402
from probe_released_modality_shortcuts import run  # noqa: E402


PARAMS = {"THETA_FULL": 15.0, "THETA_HALF": 30.0,
          "T_FULL": 0.3, "T_HALF": 1.0}


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
    facts, audio, media = tmp_path / "facts", tmp_path / "audio", tmp_path / "media"
    point = "card8_001"
    (facts / point).mkdir(parents=True); (audio / point / "audio/binaural").mkdir(parents=True)
    (media / point).mkdir(parents=True)
    (audio / point / "audio/binaural/mixture.wav").write_bytes(b"wav")
    (media / point / "video_only.mp4").write_bytes(b"mp4")
    (facts / point / "fact_record.json").write_text(json.dumps({
        "profile_id": "card8", "target_first": True,
        "mcq": {"stem": "when?", "options_space": ["a", "b"],
                "truth_option": "a"},
        "open": {"stem": "seconds?", "truth_value": 2.4,
                 "scoring": "absolute_time",
                 "certification_policy": "strict_full_credit_only"},
    }))
    rows = build({"selected": [{"point_id": point,
                                 "profile_id": "card8"}]},
                 facts, audio, media)
    assert len(rows) == 2
    assert rows[1]["task_type"] == "numeric_time"
    assert "timeline" not in json.dumps(rows).lower()

"""Tests for the QA-v3 browser calibration pack and response scorer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

from build_qa_v3_human_calibration_pack import build  # noqa: E402
from score_qa_v3_human_calibration import score  # noqa: E402


def _fact(root, point_id, profile, target="black-and-white", truth=-70.0):
    path = root / point_id
    path.mkdir(parents=True)
    value = {
        "profile_id": profile, "target_coat": target,
        "target_first": True,
        "open": {"stem": "numeric?", "truth_value": truth},
    }
    (path / "fact_record.json").write_text(json.dumps(value))


def _media(root, point_id):
    path = root / point_id
    path.mkdir(parents=True)
    (path / "full_main.mp4").write_bytes(b"fixture-mp4-" + point_id.encode())


def test_pack_hides_gold_and_binds_copied_media(tmp_path):
    facts, media, output = tmp_path / "facts", tmp_path / "media", tmp_path / "out"
    _fact(facts, "card1F_001", "card1F")
    _fact(facts, "card8_001", "card8", truth=2.4)
    _media(media, "card1F_001"); _media(media, "card8_001")
    output.mkdir()
    selection = {"selected": [
        {"point_id": "card1F_001", "profile_id": "card1F"},
        {"point_id": "card8_001", "profile_id": "card8"},
    ]}
    study, answers = build(selection, facts, media, output)
    assert len(study["items"]) == len(answers["items"]) == 2
    assert "truth" not in json.dumps(study)
    assert all(len(item["media_sha256"]) == 64 for item in answers["items"])
    assert (output / "public/index.html").is_file()
    assert (output / "public/study_items.json").is_file()
    assert (output / "private/answer_key.json").is_file()
    assert not (output / "public/answer_key.json").exists()
    html = (output / "public/index.html").read_text()
    assert "<video id=\"video\" controls" not in html
    assert "play_count" in html
    assert "plays>=2" in html
    assert "resultText" in html
    assert "复制 JSON" in html
    # 右键菜单会重新露出原生控件(含时间轴),必须禁掉
    assert 'oncontextmenu="return false"' in html


def test_preview_limit_keeps_one_item_per_profile(tmp_path):
    facts, media, output = tmp_path / "facts", tmp_path / "media", tmp_path / "out"
    selected = []
    for profile in ("card1F", "card1B", "card8"):
        for index in range(2):
            point = f"{profile}_{index}"
            _fact(facts, point, profile, truth=(2.4 if profile == "card8" else -70))
            _media(media, point)
            selected.append({"point_id": point, "profile_id": profile})
    output.mkdir()
    study, _ = build({"selected": selected}, facts, media, output,
                     per_profile_limit=1)
    assert len(study["items"]) == 3
    assert {item["profile_id"] for item in study["items"]} == {
        "card1F", "card1B", "card8"}


def test_scorer_excludes_binding_errors_from_numeric_quantiles():
    key = {"items": [
        {"item_id": "a", "profile_id": "card1F",
         "binding_truth": "black-and-white", "numeric_truth": -70.0,
         "error_kind": "circular_angle_deg"},
        {"item_id": "t", "profile_id": "card8",
         "binding_truth": "before", "numeric_truth": 2.4,
         "error_kind": "absolute_time_s"},
    ]}
    responses = {"responses": [
        {"participant_id": "p1", "item_id": "a",
         "binding_answer": "black-and-white", "numeric_answer": -60.0},
        {"participant_id": "p1", "item_id": "t",
         "binding_answer": "after", "numeric_answer": 99.0},
    ]}
    result = score(key, [responses])
    assert result["full_av_binding_accuracy"] == 0.5
    assert result["numeric_error_on_binding_correct_trials"][
        "circular_angle_deg"]["p75"] == pytest.approx(10.0)
    assert result["numeric_error_on_binding_correct_trials"][
        "absolute_time_s"]["n"] == 0


def test_duplicate_participant_item_fails_closed():
    key = {"items": [{"item_id": "a", "profile_id": "card1F",
                      "binding_truth": "yellow", "numeric_truth": 0,
                      "error_kind": "circular_angle_deg"}]}
    row = {"participant_id": "p", "item_id": "a",
           "binding_answer": "yellow", "numeric_answer": 0}
    with pytest.raises(ValueError, match="duplicate"):
        score(key, [{"responses": [row, dict(row)]}])

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


ANGLE_STEM = ("At the end of the video, roughly how many degrees from your "
              "facing direction is the dog that barked last? Right is positive.")
TIME_STEM = "At how many seconds does the named dog bark for the first time?"


def _fact(root, point_id, profile, target="black-and-white", truth=-70.0,
          room=None, stem=None, render=None):
    path = root / point_id
    path.mkdir(parents=True)
    value = {
        "profile_id": profile, "target_coat": target,
        "target_first": True,
        "open": {"stem": stem or (TIME_STEM if profile == "card8" else ANGLE_STEM),
                 "truth_value": truth},
        "room": {"ground_z_ue_cm": 27.2,
                 "floor_reference": {"status": "measured", "ground_z_ue_cm": 27.2}},
    }
    if room is not None:
        value["room"] = room
    (path / "fact_record.json").write_text(json.dumps(value))
    if render is not False:
        (path / "timeline.json").write_text(json.dumps({"render": render or {
            "hfov_degrees": 105.0, "frame_count": 75, "frame_rate_hz": 15}}))


def _practice(root, media, point_id="card1F_practice"):
    _fact(root, point_id, "card1F", truth=12.0)
    _media(media, point_id)
    return {"selected": [{"point_id": point_id, "profile_id": "card1F"}]}


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
    practice = _practice(facts, media)
    study, answers = build(selection, facts, media, output,
                           practice_selection=practice)
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
    assert "seek_count" in html
    # 2026-09-03 owner:"人应该有权利拖动,凭什么人就只能看两次呢?" 模型拿到整段
    # 视频可以任意回看,限制人只会把容差量得比模型实际条件更宽松。
    assert "plays>=2" not in html
    assert 'id="seek"' in html
    assert "localStorage" in html
    assert "resultText" in html
    assert "复制 JSON" in html
    # 右键菜单会重新露出原生控件(含时间轴),必须禁掉
    assert 'oncontextmenu="return false"' in html


def test_pack_refuses_media_from_an_unmeasured_floor(tmp_path):
    # 2026-09-03: Apartment renders stood on a hand-written ground_z (0 vs +27 cm).
    # Facts written before the floor was measured carry no room.floor_reference.
    facts, media, output = tmp_path / "facts", tmp_path / "media", tmp_path / "out"
    _fact(facts, "card1F_001", "card1F", room={"ground_z_ue_cm": 0.0})
    _media(media, "card1F_001")
    output.mkdir()
    selection = {"selected": [{"point_id": "card1F_001", "profile_id": "card1F"}]}
    practice = _practice(facts, media)
    with pytest.raises(ValueError, match="floor"):
        build(selection, facts, media, output, practice_selection=practice)
    # a declared ground that disagrees with the measured floor is refused too
    facts2, output2 = tmp_path / "facts2", tmp_path / "out2"
    _fact(facts2, "card1F_001", "card1F",
          room={"ground_z_ue_cm": 0.0,
                "floor_reference": {"status": "measured", "ground_z_ue_cm": 27.2}})
    output2.mkdir()
    practice2 = _practice(facts2, media, "card1F_practice2")
    with pytest.raises(ValueError, match="disagrees"):
        build(selection, facts2, media, output2, practice_selection=practice2)


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
    practice = _practice(facts, media)
    study, _ = build({"selected": selected}, facts, media, output,
                     practice_selection=practice, per_profile_limit=1)
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


def test_practice_round_is_disjoint_and_reveals_its_own_answer(tmp_path):
    facts, media, output = tmp_path / "facts", tmp_path / "media", tmp_path / "out"
    _fact(facts, "card1F_001", "card1F")
    _media(media, "card1F_001")
    practice = _practice(facts, media)
    output.mkdir()
    selection = {"selected": [{"point_id": "card1F_001", "profile_id": "card1F"}]}
    study, _ = build(selection, facts, media, output, practice_selection=practice)
    assert study["practice_item_count"] == 1
    practice_doc = json.loads(
        (output / "public/practice_items.json").read_text())
    # 练习题必须自带答案(要给反馈),正题必须不带
    assert "truth" in practice_doc["items"][0]["numeric"]
    assert "truth" not in json.dumps(study)
    assert practice_doc["items"][0]["item_id"] != study["items"][0]["item_id"]


def test_practice_sharing_a_study_item_fails_closed(tmp_path):
    facts, media, output = tmp_path / "facts", tmp_path / "media", tmp_path / "out"
    _fact(facts, "card1F_001", "card1F")
    _media(media, "card1F_001")
    output.mkdir()
    same = {"selected": [{"point_id": "card1F_001", "profile_id": "card1F"}]}
    with pytest.raises(ValueError, match="leaks a study answer"):
        build(same, facts, media, output, practice_selection=same)


def test_a_stem_without_a_stated_convention_fails_closed(tmp_path):
    # 2026-09-03: owner 按"右=0、正前=90"作答,同样三份答案的误差中位数从 48.97 度
    # 变成 30.0 度。题面不写清约定,量出来的就是约定猜测而不是感知。
    facts, media, output = tmp_path / "facts", tmp_path / "media", tmp_path / "out"
    _fact(facts, "card1F_001", "card1F",
          stem="Roughly how many degrees away is the dog that barked last?")
    _media(media, "card1F_001")
    practice = _practice(facts, media)
    output.mkdir()
    selection = {"selected": [{"point_id": "card1F_001", "profile_id": "card1F"}]}
    with pytest.raises(ValueError, match="states no azimuth convention"):
        build(selection, facts, media, output, practice_selection=practice)


def test_mixed_conventions_in_one_pack_fail_closed(tmp_path):
    facts, media, output = tmp_path / "facts", tmp_path / "media", tmp_path / "out"
    _fact(facts, "card1F_001", "card1F")
    _fact(facts, "card1B_001", "card1B",
          stem=("At frame 22, roughly how many degrees is the dog that barked "
                "last? Azimuth is in [-180, 180] with positive values to the "
                "left."))
    _media(media, "card1F_001"); _media(media, "card1B_001")
    practice = _practice(facts, media)
    output.mkdir()
    selection = {"selected": [
        {"point_id": "card1F_001", "profile_id": "card1F"},
        {"point_id": "card1B_001", "profile_id": "card1B"},
    ]}
    with pytest.raises(ValueError, match="mix azimuth conventions"):
        build(selection, facts, media, output, practice_selection=practice)


def test_the_page_needs_the_render_block_to_label_its_scale(tmp_path):
    facts, media, output = tmp_path / "facts", tmp_path / "media", tmp_path / "out"
    _fact(facts, "card1F_001", "card1F", render=False)
    _media(media, "card1F_001")
    practice = _practice(facts, media)
    output.mkdir()
    selection = {"selected": [{"point_id": "card1F_001", "profile_id": "card1F"}]}
    with pytest.raises(FileNotFoundError, match="timeline.json"):
        build(selection, facts, media, output, practice_selection=practice)


def test_the_pack_records_the_convention_and_the_frame_extent(tmp_path):
    facts, media, output = tmp_path / "facts", tmp_path / "media", tmp_path / "out"
    _fact(facts, "card1F_001", "card1F")
    _media(media, "card1F_001")
    practice = _practice(facts, media)
    output.mkdir()
    selection = {"selected": [{"point_id": "card1F_001", "profile_id": "card1F"}]}
    study, answers = build(selection, facts, media, output,
                           practice_selection=practice)
    assert study["azimuth_convention"] == "right_positive"
    assert answers["azimuth_convention"] == "right_positive"
    view = study["items"][0]["view"]
    assert view["hfov_degrees"] == 105.0
    assert view["clip_seconds"] == pytest.approx(5.0)
    assert study["items"][0]["numeric"]["kind"] == "azimuth_deg"


def test_scorer_refuses_a_response_from_the_other_convention():
    key = {"azimuth_convention": "left_positive",
           "items": [{"item_id": "a", "profile_id": "card1F",
                      "binding_truth": "yellow", "numeric_truth": 15.0,
                      "error_kind": "circular_angle_deg"}]}
    doc = {"azimuth_convention": "right_positive",
           "responses": [{"participant_id": "p", "item_id": "a",
                          "binding_answer": "yellow", "numeric_answer": -15.0}]}
    with pytest.raises(ValueError, match="disagrees with the answer key"):
        score(key, [doc])


def test_scorer_needs_an_explicit_assumption_for_undeclared_responses():
    key = {"azimuth_convention": "right_positive",
           "items": [{"item_id": "a", "profile_id": "card1F",
                      "binding_truth": "yellow", "numeric_truth": 15.0,
                      "error_kind": "circular_angle_deg"}]}
    doc = {"responses": [{"participant_id": "p", "item_id": "a",
                          "binding_answer": "yellow", "numeric_answer": 20.0}]}
    with pytest.raises(ValueError, match="assume-response-convention"):
        score(key, [doc])
    result = score(key, [doc], assume_response_convention="right_positive")
    # 旁路旗标必须回显被旁路的内容
    assert result["assumed_response_convention"] == "right_positive"
    assert result["azimuth_convention"] == "right_positive"

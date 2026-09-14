"""题库导出的两个私有分层字段：每条取帧规则各钉一个用例，加上 null 路径和 public 不变。

这里钉住三件事。第一，取帧规则表里每一条都真的按它说的那样取帧——写错一条不会有任何东西
报警，题库看上去照样是绿的，只是分层报分从此按错的帧算。第二，推不出来的题型必须写 null
加原因，不许因为想把表填满就退回去估一个数。第三，这两个字段只进 private：写完之后 public
目录的每个文件哈希必须和写之前一模一样。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "dataset"
sys.path.insert(0, str(TOOLS))

from avengine.dataset.question_strata import (  # noqa: E402
    CANDIDATE_RULES,
    QUERY_FRAME_RULES,
    UNDERIVED_REASONS,
    question_strata,
    summarize_strata,
    unimodal_candidates,
    visibility_at_query,
)

FRAMES = 10


def track(*states):
    assert len(states) == FRAMES
    return {str(index): {"frame_index": index, "state": state} for index, state in enumerate(states)}


def make_facts(visibility=None, events=None, actors=None, review=None):
    """一段合成的 facts：逐帧可见状态就是像素真值那一份，事件表就是音频那一份。"""

    visibility = visibility if visibility is not None else {
        "a1": track("visible_clear", "visible_clear", "visible_occluded", "visible_occluded",
                    "fully_occluded", "fully_occluded", "visible_clear", "visible_clear",
                    "out_of_view", "out_of_view"),
        "a2": track(*(["out_of_view"] * 4 + ["visible_clear"] * 6)),
    }
    return {
        "time": {"frame_count": FRAMES, "frame_rate_hz": 15},
        "visibility": visibility,
        "events": events if events is not None else [
            {"event_id": "e1", "actor_id": "a1", "start_frame": 2, "end_frame": 4,
             "start_s": 0.2, "end_s": 0.4, "sound_class": "dog_bark", "transcript": None},
            {"event_id": "e2", "actor_id": "a2", "start_frame": 6, "end_frame": 8,
             "start_s": 0.6, "end_s": 0.8, "sound_class": "printer", "transcript": None},
        ],
        "actors": actors if actors is not None else {
            "a1": {"actor_id": "a1", "appearance": {"field": "top_color", "value": "green"}},
            "a2": {"actor_id": "a2", "appearance": {"field": "top_color", "value": "blue"}},
        },
        "appearance_review": review if review is not None else {
            "a1": {"status": "reviewed"}, "a2": {"status": "reviewed"},
        },
    }


def item(qa_id, evidence, options=None):
    forms = {"open": {}}
    if options is not None:
        forms["mcq"] = {"options": [{"value": value} for value in options]}
    return {"qa_id": qa_id, "truth": {"value": None, "evidence": evidence}, "forms": forms}


# --------------------------------------------------------------------------
# 取帧规则：一条一个用例
# --------------------------------------------------------------------------


def test_每条取帧规则都在表里有一句说法():
    from avengine.dataset.question_strata import FRAME_RULE_NOTES

    for qa_id, rule in QUERY_FRAME_RULES.items():
        assert FRAME_RULE_NOTES.get(rule), f"{qa_id} 用的规则 {rule} 没有写说明"


def test_有_query_frame_的题就用_query_frame():
    got = visibility_at_query(item("QA-04", {"actor_id": "a1", "query_frame": 4}), make_facts())
    assert got["rule"] == "query_frame"
    assert got["frame"] == 4
    assert got["state"] == "fully_occluded"
    assert got["targets"] == [
        {"actor_id": "a1", "frame": 4, "state": "fully_occluded", "source": "facts.visibility"}
    ]


def test_QA02_用_target_frame():
    got = visibility_at_query(
        item("QA-02", {"target_actor_id": "a1", "target_frame": 2}), make_facts())
    assert (got["rule"], got["frame"], got["state"]) == ("target_frame", 2, "visible_occluded")


def test_QA07_用入画那一帧():
    got = visibility_at_query(
        item("QA-07", {"target_actor_id": "a2", "entry_frame": 4}), make_facts())
    assert (got["rule"], got["frame"], got["state"]) == ("entry_frame", 4, "visible_clear")


def test_QA24_用片尾那一帧并且跟证据自带的状态对拍():
    got = visibility_at_query(
        item("QA-24", {"target_actor_id": "a1", "final_frame": 9,
                       "final_visibility_state": "out_of_view"}), make_facts())
    assert (got["rule"], got["frame"], got["state"]) == ("final_frame", 9, "out_of_view")
    assert got["cross_check"] == {"field": "final_visibility_state", "expected": "out_of_view",
                                  "observed": "out_of_view", "agrees": True}


def test_对拍不一致会如实记成没命中():
    got = visibility_at_query(
        item("QA-24", {"target_actor_id": "a1", "final_frame": 9,
                       "final_visibility_state": "visible_clear"}), make_facts())
    assert got["cross_check"]["agrees"] is False


def test_锚在事件上的题用那个事件的起始帧():
    got = visibility_at_query(
        item("QA-21", {"target_actor_id": "a1",
                       "event": {"event_id": "e1", "actor_id": "a1", "start_frame": 2}}),
        make_facts())
    assert (got["rule"], got["frame"], got["state"]) == (
        "first_event_start_frame", 2, "visible_occluded")


def test_只给了_event_ids_也能回facts里查出起始帧():
    got = visibility_at_query(item("QA-01", {"target_actor_id": "a2", "event_ids": ["e2"]}),
                              make_facts())
    assert (got["rule"], got["frame"], got["state"]) == (
        "first_event_start_frame", 6, "visible_clear")


def test_发声窗口题整段取状态_窗口里不变就记那个状态():
    got = visibility_at_query(
        item("QA-06", {"actor_id": "a2", "start_frame": 6, "end_frame": 8}), make_facts())
    assert got["rule"] == "event_window"
    assert got["window_frames"] == [6, 8]
    assert got["state"] == "visible_clear"
    assert got["targets"][0]["window_histogram"] == {"visible_clear": 3}


def test_发声窗口题窗口里变了就记_window_mixed():
    got = visibility_at_query(
        item("QA-15", {"actor_id": "a1", "start_frame": 2, "end_frame": 5}), make_facts())
    assert got["state"] == "window_mixed"
    assert got["targets"][0]["window_histogram"] == {"visible_occluded": 2, "fully_occluded": 2}


def test_整段统计题记_whole_clip_并且直方图在():
    row = question_strata("q1", item("QA-05", {"event_ids": ["e1", "e2"]}), make_facts())
    got = row["visibility_at_query"]
    assert got["rule"] == "whole_clip"
    assert got["state"] == "whole_clip"
    assert [t["actor_id"] for t in got["targets"]] == ["a1", "a2"]
    assert row["visibility_histogram"]["a1"] == {
        "visible_clear": 4, "visible_occluded": 2, "fully_occluded": 2, "out_of_view": 2,
        "other": 0, "frames": FRAMES}


def test_转场题答案是没发生时改用观察窗口():
    got = visibility_at_query(
        item("QA-11", {"target_actor_id": "a1", "query_frame": None,
                       "observation_window": [0, 9]}), make_facts())
    assert got["rule"] == "query_frame_else_observation_window"
    assert got["window_frames"] == [0, 9]
    assert got["state"] == "window_mixed"


def test_转场题发生过就还是用_query_frame():
    got = visibility_at_query(
        item("QA-09", {"target_actor_id": "a1", "query_frame": 6,
                       "observation_window": [0, 9]}), make_facts())
    assert (got["frame"], got["state"]) == (6, "visible_clear")


def test_多目标题每个演员各记一条():
    got = visibility_at_query(
        item("QA-14", {"query_frame": 5, "appearance_reviews": {"a1": {}, "a2": {}}}), make_facts())
    assert [(t["actor_id"], t["state"]) for t in got["targets"]] == [
        ("a1", "fully_occluded"), ("a2", "visible_clear")]
    assert got["state"] == "mixed"


def test_表里没有的新题型走兜底并且规则名说得出来():
    got = visibility_at_query(item("QA-99", {"actor_id": "a1", "query_frame": 1}), make_facts())
    assert got["rule"] == "unlisted_type_query_frame"
    assert (got["frame"], got["state"]) == (1, "visible_clear")


# --------------------------------------------------------------------------
# null 与原因的路径
# --------------------------------------------------------------------------


def test_取不到查询时刻就记_unknown_加原因():
    got = visibility_at_query(item("QA-04", {"actor_id": "a1"}), make_facts())
    assert got["state"] == "unknown"
    assert "query_frame" in got["reason"]
    assert got["frame"] is None


def test_写着_null_和根本没这个键的原因不一样():
    absent = visibility_at_query(item("QA-04", {"actor_id": "a1"}), make_facts())["reason"]
    null = visibility_at_query(item("QA-04", {"actor_id": "a1", "query_frame": None}),
                               make_facts())["reason"]
    assert absent != null
    assert "null" in null


def test_没有指名演员就记_unknown_加原因():
    got = visibility_at_query(item("QA-18", {"query_frame": 3, "active_actor_ids": []}),
                              make_facts())
    assert got["state"] == "unknown"
    assert got["reason"]


def test_演员在像素真值里没有那一帧也记原因():
    got = visibility_at_query(item("QA-04", {"actor_id": "a3", "query_frame": 3}), make_facts())
    assert got["state"] == "unknown"
    assert "a3" in got["targets"][0]["reason"]


# --------------------------------------------------------------------------
# 单模态候选数
# --------------------------------------------------------------------------


def test_没有封闭选项也推不出答案域的题写_null_加原因():
    # QA-10 / QA-13 / QA-25 这三类虽然没有 MCQ，但答案域能从证据或题面推出来；
    # 这里用一道没有选项、也没有推导口径的 QA-04 来走这条分支。
    got = unimodal_candidates(item("QA-04", {"actor_id": "a1", "azimuth_deg": -30.0}), make_facts())
    assert got["audio_only"] is None and got["video_only"] is None
    assert "开放式" in got["reason"]


def test_没写过口径的题型写_null_并且原因就是表里那一句():
    got = unimodal_candidates(item("QA-07", {"target_actor_id": "a1"}, ["left", "right"]), make_facts())
    assert got["audio_only"] is None
    assert got["reason"] == UNDERIVED_REASONS["QA-07"]


def test_口径要用的证据缺了也写_null_加原因():
    got = unimodal_candidates(item("QA-04", {"actor_id": "a1"}, ["left", "right"]), make_facts())
    assert got["audio_only"] is None
    assert "azimuth_deg" in got["reason"]


def test_方位题只听唯一只看全留():
    got = unimodal_candidates(
        item("QA-04", {"actor_id": "a1", "azimuth_deg": -30.0}, ["left", "right"]), make_facts())
    assert (got["audio_only"], got["video_only"], got["joint"]) == (1, 2, 1)
    assert got["necessary_multimodal"] is False


def test_正中间的方位挑不出左右():
    got = unimodal_candidates(
        item("QA-04", {"actor_id": "a1", "azimuth_deg": 0.0}, ["left", "right"]), make_facts())
    assert got["audio_only"] == 2


def test_计数题两个视角各裁一刀():
    got = unimodal_candidates(
        item("QA-22", {"entity_count": 2, "speaking_count": 1},
             ["0|0", "1|0", "1|1", "2|0", "2|1", "2|2"]), make_facts())
    # 只看数得出出场 2 个 -> 三个 2|* 留着；只听数得出 1 个在响 -> 三个 *|1 里留两个
    assert (got["video_only"], got["audio_only"]) == (3, 2)
    assert got["necessary_multimodal"] is True


@pytest.mark.parametrize("active,expected", [([], 1), (["a1"], 2), (["a1", "a2"], 1)])
def test_发声者题按窗口里有几个源在响来裁(active, expected):
    got = unimodal_candidates(
        item("QA-18", {"query_frame": 3, "active_actor_ids": active},
             ["a1", "a2", "none", "multiple"]), make_facts())
    assert got["audio_only"] == expected


def test_类别题只听能排除整段没出现过的类别():
    got = unimodal_candidates(
        item("QA-21", {"target_actor_id": "a1", "event": {"start_frame": 2}},
             ["dog_bark", "printer", "cat_meow", "drip"]), make_facts())
    assert (got["audio_only"], got["video_only"]) == (2, 4)


def test_可见发声者题排除不掉都不是这个选项():
    facts = make_facts(events=[{"event_id": "e1", "actor_id": "a2", "start_frame": 6,
                                "start_s": 0.6, "sound_class": "printer"}])
    got = unimodal_candidates(
        item("QA-20", {"actor_id": "a2", "query_frame": 6}, ["a1", "none_of_visible"]), facts)
    # a1 整段没出过声，被音频排除；none_of_visible 排除不掉
    assert got["audio_only"] == 1
    assert got["video_only"] == 2


def test_外观题只听留发过声的只看留认得出的():
    review = {"a1": {"status": "reviewed"}, "a2": {"status": "not_observable"}}
    facts = make_facts(events=[{"event_id": "e1", "actor_id": "a1", "start_frame": 2,
                                "start_s": 0.2, "sound_class": "dog_bark"}], review=review)
    got = unimodal_candidates(
        item("QA-02", {"target_actor_id": "a1", "target_frame": 2}, ["green", "blue"]), facts)
    assert got["audio_only"] == 1   # 只有 a1(green) 发过声
    assert got["video_only"] == 1   # 只有 a1(green) 复核通过且在画面里


def test_可见状态题只看能排除整段没出现过的状态():
    visibility = {"a1": track(*(["visible_clear"] * FRAMES))}
    facts = make_facts(visibility=visibility)
    got = unimodal_candidates(
        item("QA-08", {"actor_id": "a1", "query_frame": 3, "visibility_state": "visible_clear"},
             list(("visible_clear", "visible_occluded", "fully_occluded", "out_of_view"))), facts)
    # 只剩 visible_clear 和必须保留的 out_of_view
    assert (got["audio_only"], got["video_only"]) == (4, 2)


def test_每个写过口径的题型都给得出一句依据():
    assert set(CANDIDATE_RULES) <= set(QUERY_FRAME_RULES)
    for qa_id in set(QUERY_FRAME_RULES) - set(CANDIDATE_RULES):
        assert UNDERIVED_REASONS.get(qa_id), f"{qa_id} 没写为什么推不出来"


def test_汇总表把可用率和对拍都数出来():
    rows = [
        question_strata("q1", item("QA-04", {"actor_id": "a1", "query_frame": 4,
                                             "azimuth_deg": -30.0}, ["left", "right"]),
                        make_facts(), {"room_family": "apartment"}),
        question_strata("q2", item("QA-06", {"actor_id": "a1", "start_frame": 2, "end_frame": 3},
                                   ["moving", "still"]), make_facts(),
                        {"room_family": "kujiale"}),
    ]
    summary = summarize_strata(rows)
    assert summary["question_count"] == 2
    assert summary["unimodal_availability_by_qa"]["QA-04"] == {"total": 1, "available": 1, "null": 0}
    assert summary["unimodal_availability_by_qa"]["QA-06"] == {"total": 1, "available": 0, "null": 1}
    assert summary["visibility_by_qa_room_state"]["QA-04"]["apartment"]["fully_occluded"] == 1


# --------------------------------------------------------------------------
# public 目录一个字节都不能动
# --------------------------------------------------------------------------


def test_写私有分层字段前后_public_逐字节不变(tmp_path):
    import generate_retained_qa_bank as G

    (tmp_path / "public").mkdir()
    (tmp_path / "public" / "questions.jsonl").write_text('{"question_id": "q1"}\n')
    (tmp_path / "public" / "nested").mkdir()
    (tmp_path / "public" / "nested" / "extra.json").write_text('{"a": 1}\n')
    before = G.public_digest(tmp_path)
    assert len(before) == 2

    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(make_facts()))
    entries = [("q1", item("QA-04", {"actor_id": "a1", "query_frame": 4, "azimuth_deg": -30.0},
                           ["left", "right"]),
                {"facts_path": str(facts_path), "room_family": "apartment"})]
    summary = G.write_question_strata(tmp_path, entries)

    assert G.public_digest(tmp_path) == before
    rows = [json.loads(line) for line in (tmp_path / "private" / "strata.jsonl").open()]
    assert len(rows) == 1 and rows[0]["question_id"] == "q1"
    assert rows[0]["unimodal_candidates"]["audio_only"] == 1
    assert summary["failed_rows"] == 0


def test_facts_读不到的那一行记原因而不是把导出跑挂(tmp_path):
    import generate_retained_qa_bank as G

    entries = [("q1", item("QA-04", {"actor_id": "a1", "query_frame": 4}, ["left", "right"]),
                {"facts_path": str(tmp_path / "nope.json")})]
    summary = G.write_question_strata(tmp_path, entries)
    rows = [json.loads(line) for line in (tmp_path / "private" / "strata.jsonl").open()]
    assert summary["failed_rows"] == 1
    assert rows[0]["visibility_at_query"]["state"] == "unknown"
    assert rows[0]["unimodal_candidates"]["audio_only"] is None
    assert rows[0]["visibility_at_query"]["reason"]

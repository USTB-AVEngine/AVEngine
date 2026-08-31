"""Question-type audio profiles: each profile's declared property, proven.

每个 profile 的自检都配阳性对照:把事件表改坏,自检必须抓。并证明
card8 不再继承 card1 的片尾静默(那是 run01 首叫被压进前 2.6 秒的成因),
以及角色是语义角色、绑定到槽位是后一步。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from audio_profiles import (  # noqa: E402
    CLIP_SECONDS,
    OTHER,
    SAMPLE_RATE,
    TARGET,
    AudioProfileError,
    ScheduledEvent,
    _assert_no_overlap,
    _self_check_backward,
    _self_check_first_call_bands,
    _self_check_forward,
    schedule_backward_anchor,
    schedule_exactly_one_calling,
    schedule_first_call_bands,
    schedule_forward_anchor,
)

PARAMS = {"TAIL_SILENCE_FRACTION": 0.3, "QUERY_SILENCE_FRACTION": 0.12,
          "GAP_MIN_S": 0.3, "FIRST_MIN_S": 0.35, "T_HALF": 1.0,
          "BANDS_CARD8": [0.35, 1.5, 2.65, 3.8, 4.7]}


def rng(seed=0):
    return np.random.default_rng(seed)


def test_forward_anchor_is_last_and_tail_silence_declared():
    schedule = schedule_forward_anchor(rng(1), params=PARAMS, anchor_frame=40)
    assert schedule.anchor is schedule.events[-1]
    assert schedule.anchor.role == TARGET
    assert schedule.anchor.purpose == "identity_anchor"
    assert schedule.declared["tail_silence_seconds"] >= \
        PARAMS["TAIL_SILENCE_FRACTION"] * CLIP_SECONDS
    assert {e.role for e in schedule.events} == {TARGET, OTHER}


def test_forward_rejects_anchor_too_late_for_declared_tail():
    with pytest.raises(AudioProfileError) as exc:
        schedule_forward_anchor(rng(1), params=PARAMS, anchor_frame=70)
    assert "tail silence" in str(exc.value)


def test_backward_keeps_the_target_silent_around_the_query():
    schedule = schedule_backward_anchor(rng(2), params=PARAMS,
                                        anchor_frame=66, query_frame=22)
    window = schedule.declared["query_silence_window_samples"]
    for event in schedule.events[:-1]:
        if event.role == TARGET:
            assert (event.end_sample_exclusive <= window[0]
                    or event.start_sample >= window[1])
    assert schedule.declared["anchor_relation"] == "anchor_after_query"


def test_backward_refuses_when_query_is_not_earlier():
    with pytest.raises(AudioProfileError) as exc:
        schedule_backward_anchor(rng(2), params=PARAMS, anchor_frame=20,
                                 query_frame=40)
    assert "before the audio anchor" in str(exc.value)


def test_backward_self_check_catches_target_sounding_in_window():
    schedule = schedule_backward_anchor(rng(3), params=PARAMS,
                                        anchor_frame=66, query_frame=22)
    window = schedule.declared["query_silence_window_samples"]
    # 阳性对照:塞一声目标音进静默窗
    schedule.events.insert(0, ScheduledEvent(
        TARGET, window[0] + 10, window[0] + 100, "control_sound"))
    schedule.anchor_index += 1          # 锚仍是最后一条,索引随插入右移
    with pytest.raises(AudioProfileError) as exc:
        _self_check_backward(schedule, PARAMS, window)
    assert "query silence window" in str(exc.value)


def test_card8_does_not_inherit_card1_tail_silence():
    """run01 的教训:继承片尾静默把首叫压进前 2.6 秒,后两带结构性为空。"""
    late = (2, 3)
    schedule = schedule_first_call_bands(rng(4), params=PARAMS,
                                         target_bands=late)
    firsts = {}
    for event in sorted(schedule.events, key=lambda e: e.start_sample):
        firsts.setdefault(event.role, event.start_seconds)
    ordered = sorted(firsts.values())
    assert ordered[0] >= PARAMS["BANDS_CARD8"][2]     # 落在第 3 个带里
    assert ordered[1] >= PARAMS["BANDS_CARD8"][3]     # 第 4 个带
    assert ordered[1] > 3.0        # 若继承 1.5s 片尾静默,这是不可能的


@pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
def test_every_reachable_band_pair_schedules(pair):
    schedule = schedule_first_call_bands(rng(5), params=PARAMS,
                                         target_bands=pair)
    assert schedule.declared["target_bands"] == list(pair)


def test_card8_refuses_unordered_band_pair():
    with pytest.raises(AudioProfileError) as exc:
        schedule_first_call_bands(rng(6), params=PARAMS, target_bands=(3, 1))
    assert "smaller one" in str(exc.value)


def test_card8_self_check_catches_too_close_first_calls():
    schedule = schedule_first_call_bands(rng(7), params=PARAMS,
                                         target_bands=(0, 1))
    schedule.events[1].start_sample = schedule.events[0].start_sample + 5000
    schedule.events[1].end_sample_exclusive = \
        schedule.events[1].start_sample + 4800
    with pytest.raises(AudioProfileError):
        _self_check_first_call_bands(schedule, PARAMS, (0, 1))


def test_card7_has_exactly_one_caller_at_the_query_frame():
    schedule = schedule_exactly_one_calling(rng(8), params=PARAMS,
                                            query_frame=30)
    calling = [e for e in schedule.events
               if e.frame_span()[0] <= 30 < e.frame_span()[1]]
    assert len(calling) == 1
    assert calling[0].role == TARGET


def test_roles_are_semantic_and_bound_later():
    schedule = schedule_forward_anchor(rng(9), params=PARAMS, anchor_frame=40)
    assert all(e.role in (TARGET, OTHER) for e in schedule.events)
    bound = schedule.bind({TARGET: "source2", OTHER: "source1"})
    assert {slot for slot, _ in bound} == {"source1", "source2"}
    swapped = schedule.bind({TARGET: "source1", OTHER: "source2"})
    assert [s for s, _ in bound] != [s for s, _ in swapped]
    with pytest.raises(AudioProfileError):
        schedule.bind({TARGET: "source1"})


def test_overlap_is_refused_for_sequential_programs():
    schedule = schedule_forward_anchor(rng(10), params=PARAMS, anchor_frame=40)
    schedule.events[1].start_sample = schedule.events[0].start_sample + 10
    with pytest.raises(AudioProfileError) as exc:
        _assert_no_overlap(schedule)
    assert "overlapping" in str(exc.value)


def test_forward_self_check_catches_anchor_by_the_wrong_role():
    schedule = schedule_forward_anchor(rng(11), params=PARAMS, anchor_frame=40)
    schedule.events[-1].role = OTHER
    with pytest.raises(AudioProfileError) as exc:
        _self_check_forward(schedule, PARAMS)
    assert "target actor" in str(exc.value)


def test_onsets_vary_across_seeds():
    """窗内时刻必须随机 —— 固定在某一秒本身就是可学模板。"""
    starts = {schedule_forward_anchor(rng(s), params=PARAMS,
                                      anchor_frame=40).events[0].start_sample
              for s in range(12)}
    assert len(starts) > 6

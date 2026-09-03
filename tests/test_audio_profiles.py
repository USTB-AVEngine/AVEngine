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
    OTHER,
    TARGET,
    AudioProfileError,
    Schedule,
    ScheduledEvent,
    _assert_no_overlap,
    card8_band_edges,
    card8_scoring_params,
    _self_check_backward,
    _self_check_first_call_bands,
    schedule_first_sound_at_frame,
    _self_check_forward,
    schedule_backward_anchor,
    schedule_exactly_one_calling,
    schedule_event_count,
    schedule_first_call_bands,
    schedule_second_sound_at_frame,
    schedule_forward_anchor,
)

PARAMS = {"TAIL_SILENCE_FRACTION": 0.3, "QUERY_SILENCE_FRACTION": 0.12,
          "GAP_MIN_S": 0.3, "FIRST_MIN_S": 0.35, "T_HALF": 1.0,
          "T_FULL": 0.5,
          "T_FULL_status": "placeholder_research",
          "CLIP_SECONDS": 5.0,
          "EVENT_SECONDS": 0.3,
          "SAMPLE_RATE_HZ": 16000,
          "FRAME_COUNT": 75,
          "TICKS_PER_SAMPLE": 3,
          "TICKS_PER_FRAME": 3200,
          "BANDS_CARD8": [0.35, 1.5, 2.65, 3.8, 4.7]}
SAMPLE_RATE = PARAMS["SAMPLE_RATE_HZ"]


def rng(seed=0):
    return np.random.default_rng(seed)


def test_forward_anchor_is_last_and_tail_silence_declared():
    schedule = schedule_forward_anchor(rng(1), params=PARAMS, anchor_frame=40)
    assert schedule.anchor is schedule.events[-1]
    assert schedule.anchor.role == TARGET
    assert schedule.anchor.purpose == "identity_anchor"
    assert schedule.declared["tail_silence_seconds"] >= \
        PARAMS["TAIL_SILENCE_FRACTION"] * PARAMS["CLIP_SECONDS"]
    assert {e.role for e in schedule.events} == {TARGET, OTHER}
    assert [event for event in schedule.events if event.role == TARGET] == [
        schedule.anchor]


def test_forward_rejects_anchor_too_late_for_declared_tail():
    with pytest.raises(AudioProfileError) as exc:
        schedule_forward_anchor(rng(1), params=PARAMS, anchor_frame=70)
    assert "tail silence" in str(exc.value)


def test_backward_keeps_the_target_silent_around_the_query():
    schedule = schedule_backward_anchor(rng(2), params=PARAMS,
                                        anchor_frame=66, query_frame=22)
    assert [event for event in schedule.events if event.role == TARGET] == [
        schedule.anchor]
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
        TARGET, window[0] + 10, window[0] + 100, "control_sound",
        PARAMS["SAMPLE_RATE_HZ"], PARAMS["TICKS_PER_SAMPLE"],
        PARAMS["TICKS_PER_FRAME"], PARAMS["FRAME_COUNT"]))
    schedule.anchor_index += 1          # 锚仍是最后一条,索引随插入右移
    with pytest.raises(AudioProfileError) as exc:
        _self_check_backward(schedule, PARAMS, window)
    assert "target must sound exactly once" in str(exc.value)


def test_backward_self_check_catches_target_sounding_outside_query_window():
    schedule = schedule_backward_anchor(rng(23), params=PARAMS,
                                        anchor_frame=66, query_frame=22)
    window = schedule.declared["query_silence_window_samples"]
    schedule.events.insert(0, ScheduledEvent(
        TARGET, 1000, 5800, "control_sound",
        PARAMS["SAMPLE_RATE_HZ"], PARAMS["TICKS_PER_SAMPLE"],
        PARAMS["TICKS_PER_FRAME"], PARAMS["FRAME_COUNT"]))
    schedule.anchor_index += 1
    with pytest.raises(AudioProfileError, match="target must sound exactly once"):
        _self_check_backward(schedule, PARAMS, window)


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


def test_second_first_call_cannot_land_on_half_open_band_upper_edge():
    class BoundaryRng:
        calls = 0

        def integers(self, low, high):
            self.calls += 1
            if self.calls == 1:
                return low
            return high - 1

    edges = [0.35, 1.2875, 2.225, 3.1625, 4.1]
    schedule = schedule_first_call_bands(
        BoundaryRng(), params=PARAMS, target_bands=(0, 1), band_edges=edges
    )
    _self_check_first_call_bands(
        schedule, PARAMS, (0, 1), band_edges=edges
    )


def test_card8_fails_closed_without_explicit_t_full():
    """生产 params 曾经没有 T_FULL:缺它就不许调度、不许推带边。"""
    params = {key: value for key, value in PARAMS.items() if key != "T_FULL"}
    with pytest.raises(AudioProfileError, match="T_FULL"):
        schedule_first_call_bands(rng(4), params=params, target_bands=(0, 1))
    with pytest.raises(AudioProfileError, match="T_FULL"):
        card8_band_edges(params)
    with pytest.raises(AudioProfileError, match="T_FULL"):
        card8_scoring_params(params)
    # 有 T_FULL 时带边可推,且推导记录了实际执行的参数链
    edges = card8_band_edges(PARAMS)
    assert len(edges) == 5
    scoring = card8_scoring_params(PARAMS)
    assert scoring["certification_policy"] == "strict_full_credit_only"
    assert scoring["wide_tolerance_role"] == "diagnostic_only"
    assert scoring["T_FULL_status"] == "placeholder_research"
    with pytest.raises(AudioProfileError, match="T_FULL_status"):
        card8_scoring_params(
            {key: value for key, value in PARAMS.items()
             if key != "T_FULL_status"})


def test_card8_minimum_separation_is_max_of_half_and_twice_full():
    tight = card8_scoring_params(dict(PARAMS, T_FULL=0.6))
    assert tight["min_first_call_separation_s"] == pytest.approx(1.2)
    assert tight["min_first_call_separation_samples"] == 19200
    loose = card8_scoring_params(dict(PARAMS, T_FULL=0.3))
    assert loose["min_first_call_separation_s"] == pytest.approx(1.0)
    with pytest.raises(AudioProfileError, match="narrower"):
        card8_scoring_params(dict(PARAMS, T_FULL=1.5, T_HALF=1.0))


def _two_first_calls(separation_samples):
    first = 8000
    second = first + separation_samples
    events = [ScheduledEvent(
                  TARGET, first, first + 4800, "answer_evidence",
                  PARAMS["SAMPLE_RATE_HZ"], PARAMS["TICKS_PER_SAMPLE"],
                  PARAMS["TICKS_PER_FRAME"], PARAMS["FRAME_COUNT"]),
              ScheduledEvent(
                  OTHER, second, second + 4800, "answer_evidence",
                  PARAMS["SAMPLE_RATE_HZ"], PARAMS["TICKS_PER_SAMPLE"],
                  PARAMS["TICKS_PER_FRAME"], PARAMS["FRAME_COUNT"])]
    return Schedule("card8", events, 0, {})


@pytest.mark.parametrize(("separation_s", "accepted"), [
    (1.1, False),               # T_FULL=0.6: 报两声中点误差 0.55 <= 0.6,必须拒
    (1.2, False),               # 边界 = 2*T_FULL,不严格大于,拒
    (1.2 + 1.0 / SAMPLE_RATE, True),   # 严格大于一个样本即过
    (1.5, True),
])
def test_card8_self_check_uses_strict_twice_t_full_boundary(separation_s, accepted):
    params = dict(PARAMS, T_FULL=0.6)
    edges = [0.0, 1.0, 2.5, 4.7]        # 第一声 0.5s 在带 0,第二声在带 1
    schedule = _two_first_calls(int(round(separation_s * SAMPLE_RATE)))
    if accepted:
        _self_check_first_call_bands(schedule, params, (0, 1), band_edges=edges)
    else:
        with pytest.raises(AudioProfileError, match="strictly more than"):
            _self_check_first_call_bands(schedule, params, (0, 1),
                                         band_edges=edges)


def test_scheduler_respects_derived_minimum_separation_and_records_it():
    params = dict(PARAMS, T_FULL=0.6)
    for seed in range(20):
        schedule = schedule_first_call_bands(rng(seed), params=params,
                                             target_bands=(0, 2))
        firsts = {}
        for event in sorted(schedule.events, key=lambda e: e.start_sample):
            firsts.setdefault(event.role, event.start_sample)
        earlier, later = sorted(firsts.values())
        assert later - earlier > 19200
        recorded = schedule.declared["first_call_scoring"]
        assert recorded["T_FULL"] == 0.6
        assert recorded["T_HALF"] == 1.0
        assert recorded["min_first_call_separation_s"] == pytest.approx(1.2)
        assert recorded["certification_policy"] == "strict_full_credit_only"
        assert recorded["wide_tolerance_role"] == "diagnostic_only"


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

def test_card3_first_sound_is_target_at_declared_frame():
    schedule = schedule_first_sound_at_frame(
        rng(12), params=PARAMS, query_frame=12)
    assert len(schedule.events) == 3
    assert schedule.anchor is schedule.events[0]
    assert schedule.anchor.role == TARGET
    assert schedule.anchor.frame_span()[0] == 12
    assert schedule.declared["first_sound_role"] == TARGET
    main = schedule.bind({TARGET: "source1", OTHER: "source2"})
    gate = schedule.bind({TARGET: "source2", OTHER: "source1"})
    assert main[0][0] == "source1"
    assert gate[0][0] == "source2"



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


@pytest.mark.parametrize("count", [3, 4])
def test_card15b_event_count_schedule_is_exact_and_gatea_invariant(count):
    schedule = schedule_event_count(rng(13 + count), params=PARAMS,
                                    event_count=count)
    assert len(schedule.events) == count
    assert schedule.declared["event_count"] == count
    assert {event.role for event in schedule.events} == {TARGET, OTHER}
    main = schedule.bind({TARGET: "source1", OTHER: "source2"})
    gate = schedule.bind({TARGET: "source2", OTHER: "source1"})
    assert [slot for slot, _ in main] != [slot for slot, _ in gate]


class _Clip:
    def __init__(self, duration_samples, asset_id):
        self.duration_samples = duration_samples
        self.sound_asset_id = asset_id
        self.source_start_sample = 0
        self.source_end_sample_exclusive = duration_samples


class _ClipSource:
    def __init__(self, clips):
        self._clips = list(clips)

    def next(self):
        return self._clips.pop(0)


def test_event_length_comes_from_the_drawn_clip_not_a_constant():
    clips = [_Clip(d, f"bark_{i}") for i, d in enumerate((3200, 8000, 4000, 6400))]
    schedule = schedule_forward_anchor(
        rng(1), params=PARAMS, anchor_frame=40, clip_source=_ClipSource(clips))
    assert schedule.events[-1].duration_samples == 6400
    assert schedule.events[-1].sound_asset_id == "bark_3"
    rows = schedule.program_events({TARGET: "source1", OTHER: "source2"})
    assert rows[-1]["duration_samples"] == 6400
    assert rows[-1]["source_start_sample"] == 0
    assert rows[-1]["source_end_sample_exclusive"] == 6400
    assert {event.duration_samples for event in schedule.events} != {4800}


def test_params_without_event_seconds_fail_closed():
    params = {key: value for key, value in PARAMS.items() if key != "EVENT_SECONDS"}
    with pytest.raises(AudioProfileError, match="EVENT_SECONDS"):
        schedule_forward_anchor(rng(1), params=params, anchor_frame=40)


def test_params_without_sample_rate_fail_closed():
    params = {key: value for key, value in PARAMS.items()
              if key != "SAMPLE_RATE_HZ"}
    with pytest.raises(AudioProfileError, match="SAMPLE_RATE_HZ"):
        schedule_forward_anchor(rng(1), params=params, anchor_frame=40)


def test_card6_second_sound_is_target_at_declared_frame():
    schedule = schedule_second_sound_at_frame(
        rng(18), params=PARAMS, query_frame=24)
    assert len(schedule.events) == 3
    assert schedule.anchor is schedule.events[1]
    assert schedule.anchor.role == TARGET
    assert schedule.anchor.frame_span()[0] == 24
    main = schedule.bind({TARGET: "source1", OTHER: "source2"})
    gate = schedule.bind({TARGET: "source2", OTHER: "source1"})
    assert main[1][0] == "source1"
    assert gate[1][0] == "source2"

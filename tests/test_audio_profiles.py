"""Question-type audio profiles: each profile's declared property, proven.

每个 profile 的自检都配阳性对照:把事件表改坏,自检必须抓。并证明
card8 不再继承 card1 的片尾静默(那是 run01 首叫被压进前 2.6 秒的成因),
以及角色是语义角色、绑定到槽位是后一步。
"""

from __future__ import annotations

import json
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
    card8_feasible_interval,
    card8_event_length_seconds,
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
    schedule_speech_utterances,
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
    # 整秒桶：可行域 0.35..4.1 秒只完整装得下三个，所以是四条边界。
    assert edges == [1.0, 2.0, 3.0, 4.0]
    scoring = card8_scoring_params(PARAMS)
    assert scoring["certification_policy"] == "strict_full_credit_only"
    assert scoring["wide_tolerance_role"] == "diagnostic_only"
    assert scoring["T_FULL_status"] == "placeholder_research"
    with pytest.raises(AudioProfileError, match="T_FULL_status"):
        card8_scoring_params(
            {key: value for key, value in PARAMS.items()
             if key != "T_FULL_status"})
    historical = {key: value for key, value in PARAMS.items()
                  if key != "T_FULL_status"}
    labelled = card8_scoring_params(historical, historical_record=True)
    assert labelled["T_FULL_status"] == "unspecified_in_historical_record"
    assert labelled["min_first_call_separation_s"] == pytest.approx(1.0)


def test_card8_pool_mode_band_edges_follow_catalog_max_duration(tmp_path: Path):
    def catalog(name, duration_samples):
        path = tmp_path / name
        path.write_text(json.dumps({
            "schema": "avengine_sound_event_pool_v1",
            "clips": [
                {"sound_asset_id": "a", "event_class": "dog_bark",
                 "sample_rate_hz": 16000, "duration_samples": duration_samples,
                 "source_start_sample": 0,
                 "source_end_sample_exclusive": duration_samples},
                {"sound_asset_id": "b", "event_class": "dog_bark",
                 "sample_rate_hz": 16000, "duration_samples": duration_samples,
                 "source_start_sample": 0,
                 "source_end_sample_exclusive": duration_samples},
            ],
        }))
        return path

    base = {key: value for key, value in PARAMS.items() if key != "EVENT_SECONDS"}
    base.update({
        "SOUND_SOURCE_MODE": "event_pool",
        "PAIR_KIND": "dog",
        "SOUND_EVENT_CLASS_BY_PAIR_KIND": {"dog": "dog_bark"},
    })
    short = dict(base, SOUND_EVENT_POOL=str(catalog("short.json", 4800)))
    long = dict(base, SOUND_EVENT_POOL=str(catalog("long.json", 16000)))
    assert "EVENT_SECONDS" not in short
    seconds, source = card8_event_length_seconds(short)
    assert source == "from_catalog_max"
    assert seconds == pytest.approx(0.3)
    assert card8_event_length_seconds(long)[0] == pytest.approx(1.0)
    # 事件时长从 catalog 来，所以可行域跟着 catalog 变。
    short_hi = card8_feasible_interval(short)[1]
    long_hi = card8_feasible_interval(long)[1]
    assert long_hi < short_hi
    # 短事件下整秒桶装得下三个；长事件把可行域压到 2.35 秒，
    # 一个整秒桶都凑不出两个，分桶必须 fail-closed 而不是给出单选项。
    assert card8_band_edges(short) == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(AudioProfileError, match="needs at least two"):
        card8_band_edges(long)


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


class _RoleClips:
    def __init__(self, by_role):
        self.by_role = dict(by_role)

    def for_role(self, role):
        return self.by_role[role]


def test_event_length_comes_from_the_bound_role_clip():
    target = _Clip(6400, "bark_target")
    other = _Clip(3200, "bark_other")
    schedule = schedule_forward_anchor(
        rng(1), params=PARAMS, anchor_frame=40,
        clip_source=_RoleClips({TARGET: target, OTHER: other}))
    assert schedule.events[-1].duration_samples == 6400
    assert schedule.events[-1].sound_asset_id == "bark_target"
    assert {event.sound_asset_id for event in schedule.events
            if event.role == TARGET} == {"bark_target"}
    assert {event.sound_asset_id for event in schedule.events
            if event.role == OTHER} == {"bark_other"}
    rows = schedule.program_events({TARGET: "source1", OTHER: "source2"})
    assert rows[-1]["duration_samples"] == 6400
    assert rows[-1]["source_start_sample"] == 0
    assert rows[-1]["source_end_sample_exclusive"] == 6400


def test_same_role_reuses_one_clip_and_roles_stay_distinct():
    schedule = schedule_event_count(
        rng(4), params=PARAMS, event_count=4,
        clip_source=_RoleClips({
            TARGET: _Clip(3200, "clip_t"),
            OTHER: _Clip(8000, "clip_o"),
        }))
    by_role = {}
    for event in schedule.events:
        by_role.setdefault(event.role, set()).add(event.sound_asset_id)
    assert by_role[TARGET] == {"clip_t"}
    assert by_role[OTHER] == {"clip_o"}
    assert by_role[TARGET] != by_role[OTHER]


def test_params_without_event_seconds_fail_closed():
    params = {key: value for key, value in PARAMS.items() if key != "EVENT_SECONDS"}
    with pytest.raises(AudioProfileError, match="EVENT_SECONDS"):
        schedule_forward_anchor(rng(1), params=params, anchor_frame=40)


def test_program_events_refuses_dry_canvas_schedules_without_clip_identity():
    schedule = schedule_forward_anchor(rng(1), params=PARAMS, anchor_frame=40)
    assert all(event.sound_asset_id is None for event in schedule.events)
    with pytest.raises(AudioProfileError, match="dry_canvas_window"):
        schedule.program_events({TARGET: "source1", OTHER: "source2"})
    bound = schedule.bind({TARGET: "source1", OTHER: "source2"})
    assert bound[-1][0] == "source1"


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


def test_only_fully_reachable_whole_second_buckets_are_offered():
    """头尾那两个整秒桶进不满，所以不提供。

    owner 2026-09-03 定了 card8 改整秒桶（"最好就是大概 1-2-3-4-5 秒之间"），
    但 5 秒片长下朴素的五个桶不是均匀可达的：首叫可行域是 0.35..4.1 秒，
    [0,1) 只能从 0.35 进、[4,5) 只能从 4.0 进，可达质量差六倍。模型学会
    "别选头尾"就能白拿准确率——那正是 v1 死掉的那个可被利用的答案先验。
    """

    lo, hi = card8_feasible_interval(PARAMS)
    edges = card8_band_edges(PARAMS)
    # 提供的每一个桶都必须完整落在可行域里
    assert edges[0] >= lo and edges[-1] <= hi
    for index in range(len(edges) - 1):
        assert edges[index] >= lo, edges
        assert edges[index + 1] <= hi, edges
    # 而紧挨着的下一个桶会越界，说明确实取到了能取的全部
    width = edges[1] - edges[0]
    assert edges[0] - width < lo
    assert edges[-1] + width > hi
    # 桶宽是满分容差的两倍，两者锁在一起
    assert width == pytest.approx(2.0 * card8_scoring_params(PARAMS)["T_FULL"])
    assert card8_scoring_params(PARAMS)["answer_granularity_seconds"] == (
        pytest.approx(width))


def test_the_bucket_width_and_the_tolerance_cannot_disagree():
    with pytest.raises(AudioProfileError, match="half a bucket"):
        card8_band_edges(dict(PARAMS, CARD8_BUCKET_SECONDS=1.5))
    # 一致时照常
    assert card8_band_edges(dict(PARAMS, CARD8_BUCKET_SECONDS=1.0))

class _SpeechClip(_Clip):
    def __init__(
        self,
        duration_samples,
        asset_id,
        speaker_id,
        utterance_id,
        transcript,
        split="train",
    ):
        super().__init__(duration_samples, asset_id)
        self.speaker_id = speaker_id
        self.utterance_id = utterance_id
        self.transcript = transcript
        self.split = split
        self.sample_rate_hz = SAMPLE_RATE


class _SpeechSource:
    def __init__(self, clips):
        self.clips = list(clips)
        self.calls = []

    def select_distinct_speech_clips(self, count=4, *, split="train"):
        self.calls.append((count, split))
        return self.clips[:count]


def test_speech_schedule_uses_complete_durations_and_keeps_identity():
    clips = [
        _SpeechClip(28705, "speech_0", "p262", "362", "That would help."),
        _SpeechClip(25330, "speech_1", "p340", "233", "So it should be."),
        _SpeechClip(31692, "speech_2", "p227", "270", "Did he trip?"),
        _SpeechClip(36207, "speech_3", "p304", "100", "Did you get the script?"),
    ]
    source = _SpeechSource(clips)
    params = dict(PARAMS, CLIP_SECONDS=10.0, SPEECH_GAP_SECONDS=0.3)
    roles = ["role_a", "role_b", "role_c", "role_d"]
    schedule = schedule_speech_utterances(
        rng(123),
        params=params,
        clip_source=source,
        roles=roles,
    )
    assert source.calls == [(4, "train")]
    assert schedule.profile_id == "speech_utterances"
    assert [event.role for event in schedule.events] == roles
    assert [event.duration_samples for event in schedule.events] == [
        clip.duration_samples for clip in clips
    ]
    assert [event.speaker_id for event in schedule.events] == [
        clip.speaker_id for clip in clips
    ]
    assert [event.utterance_id for event in schedule.events] == [
        clip.utterance_id for clip in clips
    ]
    assert [event.transcript for event in schedule.events] == [
        clip.transcript for clip in clips
    ]
    assert all(event.split == "train" for event in schedule.events)
    assert all(
        later.start_sample - earlier.end_sample_exclusive >= 4800
        for earlier, later in zip(schedule.events, schedule.events[1:])
    )
    assert schedule.declared["required_seconds"] == pytest.approx(8.520875)
    program_rows = schedule.program_events(
        {role: f"source{index + 1}" for index, role in enumerate(roles)}
    )
    assert [row["duration_samples"] for row in program_rows] == [
        clip.duration_samples for clip in clips
    ]
    assert [row["speaker_id"] for row in program_rows] == [
        clip.speaker_id for clip in clips
    ]
    assert [row["utterance_id"] for row in program_rows] == [
        clip.utterance_id for clip in clips
    ]
    assert [row["transcript"] for row in program_rows] == [
        clip.transcript for clip in clips
    ]
    assert all(row["split"] == "train" for row in program_rows)


def test_speech_schedule_fails_when_complete_utterances_do_not_fit():
    clips = [
        _SpeechClip(28705, "speech_0", "p262", "362", "That would help."),
        _SpeechClip(25330, "speech_1", "p340", "233", "So it should be."),
        _SpeechClip(31692, "speech_2", "p227", "270", "Did he trip?"),
        _SpeechClip(36207, "speech_3", "p304", "100", "Did you get the script?"),
    ]
    with pytest.raises(AudioProfileError, match="complete speech utterances need"):
        schedule_speech_utterances(
            rng(1),
            params=dict(PARAMS, CLIP_SECONDS=8.0),
            clip_source=_SpeechSource(clips),
            gap_seconds=0.3,
        )


def test_speech_schedule_rejects_missing_or_wrong_split_identity():
    clips = [
        _SpeechClip(1000, "speech_0", "p1", "001", "one"),
        _SpeechClip(1000, "speech_1", "p2", "002", "two"),
        _SpeechClip(1000, "speech_2", "p3", "003", "three"),
        _SpeechClip(1000, "speech_3", "p4", "004", "four", split="eval"),
    ]
    with pytest.raises(AudioProfileError, match="fewer than 4|split metadata|expected .train."):
        schedule_speech_utterances(
            rng(2),
            params=dict(PARAMS, CLIP_SECONDS=10.0),
            clip_source=_SpeechSource(clips),
        )


def test_complete_speech_reserves_requested_receiver_tail():
    clips = [_SpeechClip(16000, f"speech_{i}", f"p{i}", str(i), f"utterance {i}") for i in range(4)]
    params = dict(PARAMS, CLIP_SECONDS=6.0, SPEECH_GAP_SECONDS=0.3, SPEECH_TAIL_SECONDS=0.5)
    for seed in range(8):
        schedule = schedule_speech_utterances(rng(seed), params=params, clip_source=_SpeechSource(clips))
        assert max(event.end_sample_exclusive for event in schedule.events) <= 88000
        assert schedule.declared["reserved_tail_seconds"] == 0.5
    with pytest.raises(AudioProfileError, match="reserved tail"):
        schedule_speech_utterances(rng(0), params=dict(params, CLIP_SECONDS=5.0), clip_source=_SpeechSource(clips))

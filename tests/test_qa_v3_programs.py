"""Unit tests for the per-point audio-program generator (pilot item 1.2).

回归锚:v2 的病根是全批发声恒在第 4 帧——本测试断言批内首声 onset 必须
铺开到多个帧桶;阳性对照:不可行的约束参数必须报错,篡改后的 program
封印必须复算不一致。schema 校验用仓库真 schema。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from build_qa_v3_programs import (  # noqa: E402
    build_program,
    main,
    plan_events,
    program_request_fields,
    validate_m6_audio_program,
)
from avengine.contracts.json_io import canonical_json_sha256  # noqa: E402

SAMPLE_RATE = 16000
SAMPLE_COUNT = 80000
FRAME_COUNT = 75
EVENT_LEN = 4800
TIMELINE = {
    "time_base_hz": 48000, "ticks_per_frame": 3200, "video_fps": 15,
    "frame_count": FRAME_COUNT, "sample_rate_hz": SAMPLE_RATE,
    "ticks_per_sample": 3, "sample_count": SAMPLE_COUNT,
}
PROGRAM_POLICY = {
    "linear_gain": 0.18,
    "fade_samples": 80,
    "mode": "sequential_sources",
    "timeline": TIMELINE,
    "normalization_policy": "use_sound_asset_policy",
    "render_source_stem": True,
    "source_specific_stems": True,
    "admission_state": "research",
}
KW = dict(first_min_s=0.3, gap_min_s=0.3, tail_silence_s=1.5,
          event_len_samples=EVENT_LEN,
          sample_rate_hz=SAMPLE_RATE, sample_count=SAMPLE_COUNT)
REPO = Path(__file__).resolve().parents[1]
CLI = ["--first-min-s", "0.3", "--gap-min-s", "0.3",
       "--tail-silence-s", "1.5", "--event-seconds", "0.3",
       "--sample-rate-hz", str(SAMPLE_RATE),
       "--sample-count", str(SAMPLE_COUNT),
       "--frame-count", str(FRAME_COUNT),
       "--max-top-first-frame-share", "0.3",
       "--min-onset-buckets", "2",
       "--schema", str(REPO / "schemas/m6_audio_program_v1.schema.json"),
       "--revision", "v1"]


def _req(pid, first="source1"):
    return {"point_id": pid, "pair_kind": "dog",
            "endpoint_1": "qa_v3_dog_1_muzzle", "endpoint_2": "qa_v3_dog_2_muzzle",
            "sound_asset_id": "dog_beagle_v2_scheduled_dry", "first_slot": first,
            "event_duration_samples": EVENT_LEN,
            "source_start_sample": 3200,
            "source_end_sample_exclusive": 8000,
            **PROGRAM_POLICY}


def test_plan_satisfies_all_constraints():
    for pid in ("p1", "p2", "p3", "p4", "p5"):
        events, anchor = plan_events("s", pid, "source1", **KW)
        starts = [s for _, s in events]
        assert 3 <= len(events) <= 4
        assert starts[0] >= int(0.3 * SAMPLE_RATE)                      # 首声下限
        for a, b in zip(starts, starts[1:]):
            assert b - a >= EVENT_LEN + int(0.3 * SAMPLE_RATE)          # 间隔
        assert SAMPLE_COUNT - (starts[-1] + EVENT_LEN) >= int(1.5 * SAMPLE_RATE)  # 尾窗
        slots = [s for s, _ in events]
        assert {"source1", "source2"} <= set(slots)                     # 两只都叫
        assert anchor["anchor_slot"] == slots[-1]
        assert anchor["tail_silence_samples"] == SAMPLE_COUNT - (starts[-1] + EVENT_LEN)


def test_plan_deterministic_and_varies_across_points():
    a1 = plan_events("s", "pX", "source1", **KW)
    a2 = plan_events("s", "pX", "source1", **KW)
    assert a1 == a2
    onsets = {plan_events("s", f"p{i}", "source1", **KW)[0][0][1] for i in range(10)}
    assert len(onsets) >= 5  # 不同点 onset 各不相同(随机化生效)


def test_positive_control_infeasible_constraints_raise():
    with pytest.raises((ValueError, RuntimeError)):
        plan_events("s", "bad", "source1",
                    first_min_s=0.3, gap_min_s=0.3, tail_silence_s=4.6,
                    event_len_samples=EVENT_LEN,
                    sample_rate_hz=SAMPLE_RATE, sample_count=SAMPLE_COUNT)


def test_program_seal_and_tick_alignment():
    events, _ = plan_events("s", "p1", "source2", **KW)
    doc = build_program(_req("p1", first="source2"), events, revision="v1")
    body = {k: v for k, v in doc.items() if k != "program_content_sha256"}
    assert canonical_json_sha256(body) == doc["program_content_sha256"]  # 封印自洽
    for ev in doc["events"]:
        assert ev["start_tick"] == ev["start_sample"] * 3
        assert ev["end_tick_exclusive"] == ev["end_sample_exclusive"] * 3
    tampered = json.loads(json.dumps(doc))
    tampered["events"][0]["start_sample"] += 1
    body2 = {k: v for k, v in tampered.items() if k != "program_content_sha256"}
    assert canonical_json_sha256(body2) != tampered["program_content_sha256"]  # 阳性

    assert doc["events"][0]["source_endpoint_id"] == "qa_v3_dog_2_muzzle"  # bfirst 生效


def test_cli_batch_onset_spread_plans_and_no_clobber(tmp_path):
    reqs = [_req(f"v3p{i:03d}", first=("source1" if i % 2 else "source2"))
            for i in range(16)]
    req_p = tmp_path / "reqs.json"
    req_p.write_text(json.dumps(reqs))
    out = tmp_path / "programs"
    assert main(["--requests", str(req_p), "--seed", "s1",
                 "--out-dir", str(out), *CLI]) == 0
    manifest = json.loads((out / "programs_manifest.json").read_text())
    assert manifest["count"] == 16
    # 恒第4帧病根的回归锚:没有任何单一帧值占比超三成,且首/锚各铺开 ≥2 桶
    assert manifest["top_first_frame_share"] <= 0.3
    assert len(manifest["first_onset_frame_buckets"]) >= 2
    assert len(manifest["anchor_frame_buckets"]) >= 2
    progs = sorted(out.glob("qa_v3_dog_*_rand_v1.json"))
    plans = sorted(out.glob("*.plan.json"))
    assert len(progs) == 16 and len(plans) == 16
    plan = json.loads(plans[0].read_text())
    assert plan["tail_silence_samples"] >= int(1.5 * SAMPLE_RATE)
    # no-clobber
    assert main(["--requests", str(req_p), "--seed", "s1",
                 "--out-dir", str(out), *CLI]) == 2


def test_build_program_supports_four_slot_endpoints():
    request = {
        "pair_kind": "dog4",
        "point_id": "n4",
        "slot_endpoints": {
            "source1": "ep1", "source2": "ep2",
            "source3": "ep3", "source4": "ep4",
        },
        "sound_asset_id": "dry",
        "event_duration_samples": EVENT_LEN,
        "source_start_sample": 3200,
        "source_end_sample_exclusive": 8000,
        **PROGRAM_POLICY,
    }
    events = [
        ("source1", 1000), ("source2", 10000),
        ("source3", 20000), ("source4", 30000),
    ]
    program = build_program(request, events, revision="v1")
    assert program["candidate_source_endpoint_ids"] == [
        "ep1", "ep2", "ep3", "ep4"]
    assert [event["source_endpoint_id"] for event in program["events"]] == [
        "ep1", "ep2", "ep3", "ep4"]
    assert [event["event_id"] for event in program["events"]] == [
        "source1_event_0", "source2_event_1",
        "source3_event_2", "source4_event_3"]


def test_build_program_supports_per_event_sound_assets():
    request = {
        "pair_kind": "sound4",
        "point_id": "four-sounds",
        "slot_endpoints": {
            "source1": "ep1", "source2": "ep2",
            "source3": "ep3", "source4": "ep4",
        },
        "sound_asset_id": "unused_shared_canvas",
        "event_duration_samples": EVENT_LEN,
        "source_start_sample": 3200,
        "source_end_sample_exclusive": 8000,
        **PROGRAM_POLICY,
    }
    events = [
        ("source1", 1000, "sound_a"),
        ("source2", 10000, "sound_b"),
        ("source3", 20000, "sound_c"),
        ("source4", 30000, "sound_d"),
    ]
    program = build_program(request, events, revision="v1")
    assert [event["sound_asset_id"] for event in program["events"]] == [
        "sound_a", "sound_b", "sound_c", "sound_d"]


def test_program_event_dicts_use_clip_duration_and_full_source_window():
    request = {
        "pair_kind": "pool",
        "point_id": "pool1",
        "endpoint_1": "ep1",
        "endpoint_2": "ep2",
        "sound_asset_id": "unused",
        **PROGRAM_POLICY,
    }
    events = [
        {"slot": "source1", "start_sample": 1000, "duration_samples": 3200,
         "sound_asset_id": "bark_a", "source_start_sample": 0,
         "source_end_sample_exclusive": 3200},
        {"slot": "source2", "start_sample": 10000, "duration_samples": 8000,
         "sound_asset_id": "bark_b", "source_start_sample": 0,
         "source_end_sample_exclusive": 8000},
    ]
    program = build_program(request, events, revision="v1")
    assert [event["end_sample_exclusive"] - event["start_sample"]
            for event in program["events"]] == [3200, 8000]
    assert [event["sound_asset_id"] for event in program["events"]] == [
        "bark_a", "bark_b"]
    assert program["events"][0]["source_start_sample"] == 0
    assert program["events"][1]["source_end_sample_exclusive"] == 8000


def test_tuple_events_fail_without_canvas_window_in_the_request():
    request = {
        "pair_kind": "dog", "point_id": "p",
        "endpoint_1": "e1", "endpoint_2": "e2",
        "sound_asset_id": "dry",
        **PROGRAM_POLICY,
    }
    with pytest.raises(ValueError, match="missing"):
        build_program(request, [("source1", 1000), ("source2", 10000)],
                      revision="v1")


def test_build_program_fails_closed_without_linear_gain():
    request = {key: value for key, value in _req("p").items()
               if key != "linear_gain"}
    with pytest.raises(ValueError, match="linear_gain"):
        build_program(request, [("source1", 1000), ("source2", 10000)],
                      revision="v1")


def _params_for_request_fields(**overrides):
    fields = {
        "SAMPLE_RATE_HZ": SAMPLE_RATE,
        "CLIP_SECONDS": 5.0,
        "PROGRAM_LINEAR_GAIN": 0.18,
        "PROGRAM_FADE_SAMPLES": 80,
        "PROGRAM_MODE": "sequential_sources",
        "TIME_BASE_HZ": 48000,
        "TICKS_PER_FRAME": 3200,
        "VIDEO_FPS": 15,
        "FRAME_COUNT": FRAME_COUNT,
        "TICKS_PER_SAMPLE": 3,
        "PROGRAM_NORMALIZATION_POLICY": "use_sound_asset_policy",
        "PROGRAM_RENDER_SOURCE_STEM": True,
        "PROGRAM_SOURCE_SPECIFIC_STEMS": True,
        "PROGRAM_ADMISSION_STATE": "research",
    }
    fields.update(overrides)
    return fields


def test_params_missing_program_linear_gain_fail_closed():
    params = _params_for_request_fields()
    del params["PROGRAM_LINEAR_GAIN"]
    with pytest.raises(ValueError, match="PROGRAM_LINEAR_GAIN"):
        program_request_fields(params)


def test_gain_above_schema_maximum_is_rejected_with_value_and_bound():
    params = _params_for_request_fields(PROGRAM_LINEAR_GAIN=2.5)
    with pytest.raises(ValueError, match=r"2\.5") as exc:
        program_request_fields(params)
    message = str(exc.value)
    assert "maximum=1" in message
    events, _ = plan_events("s", "p1", "source1", **KW)
    doc = build_program(_req("p1"), events, revision="v1")
    validate_m6_audio_program(doc)
    doc["events"][0]["linear_gain"] = 2.5
    with pytest.raises(ValueError, match=r"2\.5") as schema_exc:
        validate_m6_audio_program(doc)
    schema_message = str(schema_exc.value)
    assert "maximum=1" in schema_message
    assert "schema" in schema_message


def test_legal_gain_one_passes_request_fields_and_schema():
    params = _params_for_request_fields(PROGRAM_LINEAR_GAIN=1.0)
    fields = program_request_fields(params)
    assert fields["linear_gain"] == 1.0
    request = _req("p-legal")
    request["linear_gain"] = 1.0
    events, _ = plan_events("s", "p-legal", "source1", **KW)
    doc = build_program(request, events, revision="v1")
    validate_m6_audio_program(doc)
    assert {event["linear_gain"] for event in doc["events"]} == {1.0}


def test_build_program_preserves_explicit_event_ids_and_rejects_duplicates():
    events = [
        {
            "event_id": "speech_early_voice_a",
            "slot": "source1",
            "start_sample": 1000,
            "duration_samples": EVENT_LEN,
            "sound_asset_id": "sound_a",
            "source_start_sample": 0,
            "source_end_sample_exclusive": EVENT_LEN,
        },
        {
            "event_id": "speech_early_voice_b",
            "slot": "source2",
            "start_sample": 10000,
            "duration_samples": EVENT_LEN,
            "sound_asset_id": "sound_b",
            "source_start_sample": 0,
            "source_end_sample_exclusive": EVENT_LEN,
        },
    ]
    doc = build_program(_req("explicit-event-ids"), events, revision="v1")
    assert [event["event_id"] for event in doc["events"]] == [
        "speech_early_voice_a", "speech_early_voice_b"
    ]
    duplicate = [dict(event, event_id="same") for event in events]
    with pytest.raises(ValueError, match="event_id.*unique"):
        build_program(_req("duplicate-event-ids"), duplicate, revision="v1")
    with pytest.raises(ValueError, match="event_id.*non-empty"):
        build_program(
            _req("blank-event-id"),
            [dict(events[0], event_id="")],
            revision="v1",
        )
    with pytest.raises(ValueError, match="event_id.*non-empty"):
        build_program(
            _req("null-event-id"),
            [dict(events[0], event_id=None)],
            revision="v1",
        )

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
    EVENT_LEN,
    SAMPLE_COUNT,
    SAMPLE_RATE,
    build_program,
    main,
    plan_events,
)
from avengine.contracts.json_io import canonical_json_sha256  # noqa: E402

KW = dict(first_min_s=0.3, gap_min_s=0.3, tail_silence_s=1.5)


def _req(pid, first="source1"):
    return {"point_id": pid, "pair_kind": "dog",
            "endpoint_1": "qa_v3_dog_1_muzzle", "endpoint_2": "qa_v3_dog_2_muzzle",
            "sound_asset_id": "dog_beagle_v2_scheduled_dry", "first_slot": first}


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
                    first_min_s=0.3, gap_min_s=0.3, tail_silence_s=4.6)


def test_program_seal_and_tick_alignment():
    events, _ = plan_events("s", "p1", "source2", **KW)
    doc = build_program(_req("p1", first="source2"), events)
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
                 "--out-dir", str(out)]) == 0
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
                 "--out-dir", str(out)]) == 2


def test_build_program_supports_four_slot_endpoints():
    request = {
        "pair_kind": "dog4",
        "point_id": "n4",
        "slot_endpoints": {
            "source1": "ep1", "source2": "ep2",
            "source3": "ep3", "source4": "ep4",
        },
        "sound_asset_id": "dry",
    }
    events = [
        ("source1", 1000), ("source2", 10000),
        ("source3", 20000), ("source4", 30000),
    ]
    program = build_program(request, events)
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
        "sound_asset_id": "fallback",
    }
    events = [
        ("source1", 1000, "sound_a"),
        ("source2", 10000, "sound_b"),
        ("source3", 20000, "sound_c"),
        ("source4", 30000, "sound_d"),
    ]
    program = build_program(request, events)
    assert [event["sound_asset_id"] for event in program["events"]] == [
        "sound_a", "sound_b", "sound_c", "sound_d"]

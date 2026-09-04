"""Unit tests for the qa-v3 question generator (cards ①⑦⑧⑨).

手工几何的 mini 设计批:已知摆位算出期望方位/扇区/带/先叫者,断言生成
记录逐字段正确;admit=false 的点不得出题;卡⑦负样本帧必须避开锚后
静默段;no-clobber。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "qa"))

from generate_qa_v3_questions import (  # noqa: E402
    band_of,
    main,
    sector_name,
    stable_pick,
)

COLLIE = "generated_border_collie_black_white_medium_standard_adult_research_v1"
LABRADOR = "generated_labrador_yellow_medium_standard_adult_research_v1"
EP1, EP2 = "qa_v2_dog_1_collie_muzzle", "qa_v2_dog_2_labrador_muzzle"

PARAMS = {"THETA_FULL": 15.0, "THETA_HALF": 30.0, "T_HALF": 0.9,
          "T_FULL": 0.4, "T_FULL_status": "placeholder_research",
          "SAMPLE_RATE_HZ": 16000,
          "TAIL_MIN_S": 1.5, "MIN_AZIMUTH_SEP": 25.0,
          "MIN_DIST_CHANGE_CM": 50.0, "MIN_CARD7_FRAMES": 8,
          "BANDS": [0.0, 1.25, 2.5, 3.75, 5.0],
          "AZ_BANDS_CARD1": [-52.5, -29.17, -5.83, 17.5]}


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def make_timeline(s2_end=(0.0, 300.0), s1_pos=(300.0, 20.0)):
    """s1 静止近前方;s2 走到 s2_end(缺省片尾方位 +90°,落在声明带外)。"""
    frames = []
    for i in range(75):
        t = i / 74.0
        p2 = _lerp((300.0, -100.0), s2_end, t)
        frames.append({
            "frame_index": i,
            "camera": {"translation_ue_cm": [0.0, 0.0, 147.0],
                       "yaw_ue_deg": 0.0},
            "actor_states": [
                {"source_slot_id": "source1", "action_id": "idle",
                 "translation_ue_cm": [s1_pos[0], s1_pos[1], 27.0]},
                {"source_slot_id": "source2", "action_id": "walk",
                 "translation_ue_cm": [p2[0], p2[1], 27.0]},
            ],
        })
    return {"render": {"hfov_degrees": 200.0}, "frames": frames}


EVENTS = [
    {"source_endpoint_id": EP1, "start_sample": 8000,
     "end_sample_exclusive": 12800},    # s1: 0.5s–0.8s, 帧 7–12
    {"source_endpoint_id": EP2, "start_sample": 24000,
     "end_sample_exclusive": 28800},    # s2 first: 1.5s, 帧 22–27
    {"source_endpoint_id": EP1, "start_sample": 36000,
     "end_sample_exclusive": 40800},
    {"source_endpoint_id": EP2, "start_sample": 48000,
     "end_sample_exclusive": 52800},    # 锚:s2, 帧 45–50
]


def build_design_root(tmp_path, admit=True, timeline=None):
    root = tmp_path / "design"
    programs = root / "programs"
    programs.mkdir(parents=True)
    pdir = root / "v3a1_001"
    pdir.mkdir()
    (pdir / "spec.json").write_text(json.dumps(
        {"point_id": "v3a1_001", "motion_class": "A1"}))
    (pdir / "timeline.json").write_text(
        json.dumps(timeline if timeline is not None else make_timeline()))
    (pdir / "actor_selection.json").write_text(json.dumps({"actors": [
        {"source_slot_id": "source1", "asset_id": COLLIE,
         "legacy_timeline_actor_id": "dog_1"},
        {"source_slot_id": "source2", "asset_id": LABRADOR,
         "legacy_timeline_actor_id": "dog_2"},
    ]}))
    (programs / "qa_v3_dog_v3a1_001_rand_v1.json").write_text(json.dumps({
        "program_id": "qa_v3_dog_v3a1_001_rand_v1",
        "candidate_source_endpoint_ids": [EP1, EP2],
        "events": EVENTS,
    }))
    (programs / "qa_v3_dog_v3a1_001_rand_v1.plan.json").write_text(json.dumps({
        "anchor_slot": "source2", "anchor_start_sample": 48000,
        "anchor_end_sample": 52800,
        "tail_silence_samples": 80000 - 52800,
    }))
    a = bool(admit)
    (root / "filter_report.json").write_text(json.dumps({
        "schema": "x", "parameters": PARAMS,
        "results": {"v3a1_001": {
            "card1": {"admit": a}, "card5R": {"admit": False},
            "card6R": {"admit": a}, "card7": {"admit": a},
            "card8": {"admit": a}, "card9": {"admit": a}}},
    }))
    return root


def run_gen(tmp_path, root, share=0.0, out_name="out"):
    params_p = tmp_path / "params.json"
    params_p.write_text(json.dumps(PARAMS))
    out = tmp_path / out_name
    rc = main(["--design-root", str(root), "--params", str(params_p),
               "--out-root", str(out),
               "--card7-negative-share", str(share),
               "--historical-reproduction"])
    assert rc == 0
    return {c: [json.loads(line) for line in
                (out / f"facts_{c}.jsonl").read_text().splitlines()]
            for c in ("card1", "card7_main", "card7_control_neither",
                      "card8", "card9")}, out


def test_card1_truth_matches_hand_geometry(tmp_path):
    facts, _ = run_gen(tmp_path, build_design_root(tmp_path))
    (rec,) = facts["card1"]
    # 锚定者 s2 片尾在 (0,300):相机朝 +x,右手侧 → +90°
    assert abs(rec["truth"]["final_azimuth_deg"] - 90.0) < 1e-6
    # +90° 在声明带域之外 → 选择题形态缺席,四扇区块只作诊断保留
    assert rec["mcq"] is None
    assert rec["mcq_four_sector_deprecated"]["truth_option"] == "right"
    assert rec["anchor"]["slot"] == "source2"
    assert rec["anchor"]["anchor_frame"] == 45
    assert rec["open"]["truth_value"] == rec["truth"]["final_azimuth_deg"]
    assert rec["provenance"]["visibility_source"].startswith("geometry_fov")
    assert rec["qualification_claim"] is False


def test_sector_boundaries_half_open():
    assert sector_name(-45.0) == "front"
    assert sector_name(44.999) == "front"
    assert sector_name(45.0) == "right"
    assert sector_name(135.0) == "back"
    assert sector_name(-135.0) == "left"
    assert sector_name(-135.001) == "back"


def test_card7_positive_picks_exactly_one_calling_frame(tmp_path):
    facts, _ = run_gen(tmp_path, build_design_root(tmp_path))
    (rec,) = facts["card7_main"]
    assert rec["negative_sample"] is False
    f = rec["query_time"]["frame"]
    assert f % 3 == 0
    truth = rec["truth"]["calling_at_t"]
    assert truth in ("black-and-white", "yellow")
    # 帧域核对:s1 叫 7–11 → 帧 9;s2 叫 22–26 → 24;s1 33–38→33/36;s2 45–49→45/48
    span = {9: "black-and-white", 24: "yellow", 33: "black-and-white",
            36: "black-and-white", 45: "yellow", 48: "yellow"}
    assert span[f] == truth
    assert rec["mcq"]["truth_option"] == truth
    assert f"{f / 15.0:.1f}" == rec["query_time"]["stem_second"]


def test_card7_negative_avoids_anchor_tail(tmp_path):
    facts, _ = run_gen(tmp_path, build_design_root(tmp_path), share=1.0)
    assert facts["card7_main"] == []
    (rec,) = facts["card7_control_neither"]
    assert rec["negative_sample"] is True
    assert rec["truth"]["calling_at_t"] == "neither"
    assert rec["mcq"]["truth_option"] == "neither"
    f = rec["query_time"]["frame"]
    assert f < 45          # 锚后尾静默段不得作负样本
    for f0, f1 in ((7, 12), (22, 27), (33, 39), (45, 50)):
        assert not (f0 <= f < f1)


def test_card8_bands_and_both_slots(tmp_path):
    facts, _ = run_gen(tmp_path, build_design_root(tmp_path))
    recs = {r["target_slot"]: r for r in facts["card8"]}
    assert set(recs) == {"source1", "source2"}
    assert abs(recs["source1"]["truth"]["first_onset_s"] - 0.5) < 1e-9
    assert recs["source1"]["truth"]["band_index"] == 0
    assert abs(recs["source2"]["truth"]["first_onset_s"] - 1.5) < 1e-9
    assert recs["source2"]["truth"]["band_index"] == 1
    assert recs["source2"]["mcq"]["truth_option"] == "[1.25, 2.5)"
    assert "yellow" in recs["source2"]["mcq"]["stem"]
    assert recs["source2"]["open"]["certification_policy"] == \
        "strict_full_credit_only"
    assert recs["source2"]["open"]["wide_tolerance_role"] == \
        "diagnostic_only"
    assert recs["source2"]["open"]["T_FULL"] == 0.4
    assert recs["source2"]["open"]["T_HALF"] == 0.9
    assert recs["source2"]["open"]["T_FULL_status"] == "placeholder_research"
    assert recs["source2"]["open"]["min_first_call_separation_s"] == 0.9
    assert recs["source2"]["truth"]["first_call_separation_s"] == 1.0
    assert "certification_policy" not in recs["source2"]["mcq"]
    assert band_of(1.25, PARAMS["BANDS"]) == 1   # 半开边界归右带


def test_card8_generator_fails_closed_without_t_full(tmp_path):
    root = build_design_root(tmp_path)
    params_p = tmp_path / "no_tfull.json"
    params_p.write_text(json.dumps(
        {k: v for k, v in PARAMS.items() if k != "T_FULL"}))
    out = tmp_path / "out_no_tfull"
    assert main(["--design-root", str(root), "--params", str(params_p),
                 "--out-root", str(out), "--card7-negative-share", "0", "--historical-reproduction"]) == 2
    assert not out.exists()


def test_card8_skips_points_whose_first_calls_are_too_close(tmp_path):
    """s1 首叫 0.5 s、s2 首叫 1.5 s:间隔 1.0 s 在 T_FULL=0.6 下不够严格。"""
    root = build_design_root(tmp_path)
    params_p = tmp_path / "tight.json"
    params_p.write_text(json.dumps(dict(PARAMS, T_FULL=0.6, T_HALF=1.0)))
    out = tmp_path / "out_tight"
    assert main(["--design-root", str(root), "--params", str(params_p),
                 "--out-root", str(out), "--card7-negative-share", "0", "--historical-reproduction"]) == 0
    assert (out / "facts_card8.jsonl").read_text() == ""
    mani = json.loads((out / "generation_manifest.json").read_text())
    assert mani["counts"]["card8"] == 0
    assert len(mani["card8_skipped_first_call_separation"]) == 1
    assert "not_above_1.2000s" in mani["card8_skipped_first_call_separation"][0]
    assert mani["card8_scoring_params"]["min_first_call_separation_s"] == 1.2


def test_card9_first_barker(tmp_path):
    facts, _ = run_gen(tmp_path, build_design_root(tmp_path))
    (rec,) = facts["card9"]
    assert rec["truth"]["first_to_bark"] == "black-and-white"
    assert abs(rec["truth"]["onset_gap_s"] - 1.0) < 1e-9
    assert rec["mcq"]["stem"] in rec["mcq"]["stem_variants"]


def test_no_questions_when_not_admitted(tmp_path):
    facts, out = run_gen(tmp_path, build_design_root(tmp_path, admit=False))
    assert all(not v for v in facts.values())
    mani = json.loads((out / "generation_manifest.json").read_text())
    assert set(mani["counts"].values()) == {0}


def test_card8_uses_feasible_bands_and_tolerates_outlier(tmp_path):
    root = build_design_root(tmp_path)
    params = dict(PARAMS, BANDS_CARD8=[0.0, 0.65, 1.3, 1.95, 2.6])
    params_p = tmp_path / "p2.json"
    params_p.write_text(json.dumps(params))
    out = tmp_path / "out_bands"
    assert main(["--design-root", str(root), "--params", str(params_p),
                 "--out-root", str(out), "--card7-negative-share", "0", "--historical-reproduction"]) == 0
    recs = {json.loads(l)["target_slot"]: json.loads(l)
            for l in (out / "facts_card8.jsonl").read_text().splitlines()}
    # s1 首叫 0.5 → 带0 [0,0.65);s2 首叫 1.5 → 带2 [1.3,1.95)
    assert recs["source1"]["mcq"]["truth_option"] == "[0, 0.65)"
    assert recs["source1"]["mcq"]["bands_key"] == "BANDS_CARD8"
    assert recs["source2"]["truth"]["band_index"] == 2
    # 越带(带域缩到 [0,1.0)):s2 的 MCQ 缺席但开放版仍在
    params_p.write_text(json.dumps(dict(PARAMS,
                                        BANDS_CARD8=[0.0, 0.5, 1.0])))
    out2 = tmp_path / "out_bands2"
    assert main(["--design-root", str(root), "--params", str(params_p),
                 "--out-root", str(out2), "--card7-negative-share", "0", "--historical-reproduction"]) == 0
    recs2 = {json.loads(l)["target_slot"]: json.loads(l)
             for l in (out2 / "facts_card8.jsonl").read_text().splitlines()}
    assert recs2["source2"]["mcq"] is None
    assert recs2["source2"]["truth"]["band_index"] is None
    assert recs2["source2"]["open"]["truth_value"] == 1.5


def test_card1_out_of_band_target_is_flagged(tmp_path):
    # 手工几何里目标片尾 +90°,落在视锥声明带之外 → 越界须如实标注
    facts, _ = run_gen(tmp_path, build_design_root(tmp_path))
    (rec,) = facts["card1"]
    assert rec["azimuth_band"]["truth_option"] is None
    assert "outside declared bands" in rec["azimuth_band"]["note"]
    assert "why_deprecated" in rec["mcq_four_sector_deprecated"]


def test_card1_in_band_target_gets_band_mcq(tmp_path):
    # 目标片尾方位 -30°(带0 [-52.5,-29.17)),静立者 +10°(带2)
    tl = make_timeline(s2_end=(346.0, -200.0), s1_pos=(300.0, 53.0))
    facts, _ = run_gen(tmp_path, build_design_root(tmp_path, timeline=tl))
    (rec,) = facts["card1"]
    assert -30.1 < rec["truth"]["final_azimuth_deg"] < -29.9
    assert rec["mcq"]["truth_option"] == "[-52.5, -29.17)"
    assert rec["mcq"]["band_index"] == 0
    assert len(rec["mcq"]["options_space"]) == 3
    assert "azimuth band" in rec["mcq"]["stem"]


def test_card1_azimuth_band_picks_declared_bin(tmp_path):
    facts, _ = run_gen(tmp_path, build_design_root(tmp_path))
    (rec,) = facts["card1"]
    from generate_qa_v3_questions import _azimuth_band_block
    blk = _azimuth_band_block(-30.0, PARAMS)
    assert blk["band_index"] == 0 and blk["truth_option"] == "[-52.5, -29.17)"
    assert _azimuth_band_block(-29.17, PARAMS)["band_index"] == 1  # 半开左闭
    assert _azimuth_band_block(0.0, PARAMS)["band_index"] == 2
    assert _azimuth_band_block(17.5, PARAMS)["band_index"] == 2    # 末带右闭
    assert _azimuth_band_block(20.0, PARAMS)["truth_option"] is None


def test_balanced_marking_is_not_done_here(tmp_path):
    # 均衡子集标记移交切分后的胶水:生成器不得留第二个真相源
    facts, out = run_gen(tmp_path, build_design_root(tmp_path))
    assert "balanced_subset" not in facts["card9"][0]
    import json as _json
    mani = _json.loads((out / "generation_manifest.json").read_text())
    assert "control_both" in mani["card7_views"]


def test_no_clobber_and_stable_pick(tmp_path):
    root = build_design_root(tmp_path)
    _, out = run_gen(tmp_path, root)
    params_p = tmp_path / "params.json"
    assert main(["--design-root", str(root), "--params", str(params_p),
                 "--out-root", str(out), "--historical-reproduction"]) == 2
    assert stable_pick("seed", [1, 2, 3]) == stable_pick("seed", [1, 2, 3])
    assert stable_pick("a", list(range(50))) != stable_pick("b", list(range(50))) \
        or True  # 不同 seed 允许偶然同值,只证不抛

def test_refuses_to_run_without_historical_reproduction(tmp_path, capsys):
    root = build_design_root(tmp_path)
    params_p = tmp_path / "params.json"
    params_p.write_text(json.dumps(PARAMS))
    out = tmp_path / "out"
    rc = main(["--design-root", str(root), "--params", str(params_p),
               "--out-root", str(out)])
    assert rc == 2
    assert "historical" in capsys.readouterr().err
    assert not out.exists()

"""绑定可行性审计 v2：几何、成片双耳线索、分歧判定都要在已知答案的合成数据上对得上。

合成一段 6 秒、两人、两句话的 Episode：a 在右前 40 度，b 在左前 40 度，都不动。WAV 里
a 那句让右声道先到 6 个采样（0.375 ms）并比左声道响 6 dB，b 那句镜像。工具必须量回
这两个数、判两人几何上分得开；把 b 挪到离 a 5 度以内，就必须判不可行。

v2 新增的七条（对应 Codex 审核第 8 节）：没量到音频时线索状态必须是 unmeasured 而不是通过；
只有一帧角度够大不算几何通过；正前方的声源配一个 60 度外的竞争者要能通过（实测线索接近零是
预期的）；有逐源 stem 时优先在 stem 上量、并给重叠的混合轨窗打污染标记；并发发声人数按时间扫描；
QA-13 按题目可用形式分别审；给一面合成墙就能填直达射线。
"""

from __future__ import annotations

import json
import math
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "qa"
sys.path.insert(0, str(TOOLS))

import audit_binding_feasibility as ABF  # noqa: E402

SR = 16000
FPS = 15.0
FRAMES = 90
DELAY = 6  # samples ≈ 0.375 ms
LEVEL_DB = 6.0
DEFAULT_EVENTS = [
    {"event_id": "event_001", "actor_id": "a", "start_s": 1.0, "end_s": 2.0, "start_frame": 15, "end_frame": 30,
     "transcript": "one", "sound_class": "speech"},
    {"event_id": "event_002", "actor_id": "b", "start_s": 3.0, "end_s": 4.0, "start_frame": 45, "end_frame": 60,
     "transcript": "two", "sound_class": "speech"},
]


def _pos(azimuth_deg: float, distance_m: float = 2.0) -> list[float]:
    theta = math.radians(azimuth_deg)
    return [distance_m * math.sin(theta), 1.6, -distance_m * math.cos(theta)]


def _facts(actors: dict | None = None, events: list | None = None) -> dict:
    """actors: id -> {colour, az (数或逐帧列表), moving (起止帧) , vis (状态字串)}。"""

    spec = actors or {"a": {"colour": "blue", "az": 40.0}, "b": {"colour": "green", "az": -40.0}}
    facts_actors, visibility, review = {}, {}, {}
    for actor_id, item in spec.items():
        az = item["az"]
        azimuths = list(az) if isinstance(az, (list, tuple)) else [float(az)] * FRAMES
        flags = [False] * FRAMES
        if item.get("moving"):
            for f in range(item["moving"][0], item["moving"][1] + 1):
                flags[f] = True
        facts_actors[actor_id] = {
            "actor_id": actor_id,
            "appearance": {"field": "top_color", "value": item["colour"], "label": f"Human ({item['colour']} top)"},
            "root_positions_m": [_pos(v) for v in azimuths],
            "emitter_positions_m": [_pos(v) for v in azimuths],
            "moving": flags,
        }
        state = item.get("vis", "visible_clear")
        visibility[actor_id] = {
            str(f): {"frame_index": f, "state": state, "occlusion_fraction": 0.0 if state == "visible_clear" else 1.0,
                     "visible_pixels": 1000 if state == "visible_clear" else 0, "target_pixels": 1000}
            for f in range(FRAMES)
        }
        review[actor_id] = {"status": "reviewed", "value": item["colour"], "frame_refs": [0]}
    return {
        "schema": "avengine_qa_unified_episode_facts_v1",
        "episode_id": "synthetic_binding_audit",
        "time": {"frame_count": FRAMES, "frame_rate_hz": FPS, "sample_rate_hz": SR,
                 "sample_count": int(FRAMES / FPS * SR), "duration_seconds": FRAMES / FPS},
        "actors": facts_actors,
        "events": events if events is not None else json.loads(json.dumps(DEFAULT_EVENTS)),
        "listener": {
            "status": "pass",
            "positions_m": [[0.0, 1.5, 0.0]] * FRAMES,
            "yaw_deg": [0.0] * FRAMES,
            "basis_m3": [{"forward": [0.0, 0.0, -1.0], "right": [1.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0]}] * FRAMES,
        },
        "audio": {"status": "pass", "channel_count": 2, "sample_rate_hz": SR, "wet_tail_intervals": []},
        "visibility": visibility,
        "appearance_review": review,
    }


def _burst(seed: int, seconds: float = 1.0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(int(seconds * SR)) * 0.2


def _place(left: np.ndarray, right: np.ndarray, burst: np.ndarray, start_s: float, azimuth_deg: float) -> None:
    """按方位摆一段声音：在右边就右声道先到且更响，在左边镜像，正前方两声道相同。"""

    s = int(start_s * SR)
    gain = 10 ** (-LEVEL_DB / 20.0)
    n = burst.size
    if azimuth_deg > 0:
        right[s:s + n] += burst
        left[s + DELAY:s + DELAY + n] += burst * gain
    elif azimuth_deg < 0:
        left[s:s + n] += burst
        right[s + DELAY:s + DELAY + n] += burst * gain
    else:
        left[s:s + n] += burst
        right[s:s + n] += burst


def _write_pcm_wav(path: Path, data: np.ndarray) -> None:
    pcm = np.clip(data * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def _write_float_wav(path: Path, data: np.ndarray) -> None:
    """IEEE float32 WAV（format tag 3），标准库 wave 读不了，用来验证自带的 RIFF 解析。"""

    payload = np.ascontiguousarray(data.astype("<f4")).tobytes()
    fmt = struct.pack("<HHIIHH", 3, 2, SR, SR * 2 * 4, 2 * 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def _mixture(placements: list[tuple[float, float, int]]) -> np.ndarray:
    n = int(FRAMES / FPS * SR)
    left, right = np.zeros(n), np.zeros(n)
    for start_s, azimuth, seed in placements:
        _place(left, right, _burst(seed), start_s, azimuth)
    return np.stack([left, right], axis=1)


def _run(tmp_path: Path, facts: dict, questions: dict | None = None, *, placements=None, wav_missing=False,
         stems: dict | None = None, package: Path | None = None) -> dict:
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(facts))
    wav_path = tmp_path / "mix.wav"
    if not wav_missing:
        _write_pcm_wav(wav_path, _mixture(placements or [(1.0, 40.0, 1), (3.0, -40.0, 2)]))
    for actor_id, data in (stems or {}).items():
        _write_float_wav(tmp_path / f"{actor_id}_mouth_stem.wav", data)
    q_path = None
    if questions is not None:
        q_path = tmp_path / "questions.json"
        q_path.write_text(json.dumps(questions))
    return ABF.run_audit(facts_path, wav_path, q_path, dict(ABF.DEFAULT_THRESHOLDS), acoustic_package=package)


def test_geometry_and_delivered_cues_are_recovered(tmp_path):
    payload = _run(tmp_path, _facts())
    e1, e2 = payload["events"]
    assert e1["onset_azimuth_deg"] == pytest.approx(40.0, abs=1e-6)
    assert e2["onset_azimuth_deg"] == pytest.approx(-40.0, abs=1e-6)
    assert e1["separation_to_nearest_competitor_deg"]["onset"] == pytest.approx(80.0, abs=1e-6)
    assert e1["separation_over_state_window_deg"]["sustained_s_above_theta"] == pytest.approx(16 / FPS, abs=1e-9)
    # a 在右边：左声道更弱、更晚。
    assert e1["measured"]["ild_2_6k_db"] == pytest.approx(-LEVEL_DB, abs=0.8)
    assert e1["measured"]["itd_onset_ms"] == pytest.approx(-DELAY / SR * 1000.0, abs=0.07)
    assert e2["measured"]["ild_2_6k_db"] == pytest.approx(LEVEL_DB, abs=0.8)
    assert e2["measured"]["itd_onset_ms"] == pytest.approx(DELAY / SR * 1000.0, abs=0.07)
    assert e1["measured_source"] == "final_mixture"
    for event in (e1, e2):
        bf = event["binding_feasibility"]
        assert bf["geometry_state"] == ABF.STATE_PASS
        assert bf["delivered_cue_state"] == ABF.STATE_PASS
        assert "feasible" not in bf
        assert bf["geometry_separable"] is True and bf["delivered_cues_present"] is True
        assert bf["cue_basis"]["pairwise_expected_itd_diff_ms"] > ABF.DEFAULT_THRESHOLDS["itd_min_ms"]
        assert bf["cue_basis"]["itd_direction_consistent"] is True
    assert payload["audio"]["exact_zero_sample_fraction"] > 0.5
    assert payload["azimuth_convention"] == ABF.AZIMUTH_CONVENTION
    assert payload["summary"]["events_geometry_state_counts"] == {ABF.STATE_PASS: 2}
    assert payload["summary"]["events_delivered_cue_state_counts"] == {ABF.STATE_PASS: 2}


def test_close_static_speakers_are_not_bindable(tmp_path):
    payload = _run(tmp_path, _facts({"a": {"colour": "blue", "az": 40.0}, "b": {"colour": "green", "az": 45.0}}),
                   placements=[(1.0, 40.0, 1), (3.0, 45.0, 2)])
    e1 = payload["events"][0]
    sep = e1["separation_over_state_window_deg"]
    assert sep["max"] == pytest.approx(5.0, abs=1e-6)
    assert sep["min"] == pytest.approx(5.0, abs=1e-6)
    assert sep["sustained_s_above_theta"] == 0.0
    bf = e1["binding_feasibility"]
    assert bf["geometry_state"] == ABF.STATE_FAIL
    assert bf["delivered_cue_state"] == ABF.STATE_FAIL
    assert bf["cue_basis"]["pairwise_expected_itd_diff_ms"] < ABF.DEFAULT_THRESHOLDS["itd_min_ms"]
    assert any("theta_static" in reason for reason in bf["reasons"])
    assert any("same interaural" in reason for reason in bf["reasons"])


def test_missing_audio_stays_unmeasured_and_never_counts_as_pass(tmp_path):
    payload = _run(tmp_path, _facts(), wav_missing=True)
    assert payload["audio"]["status"] == "missing"
    for event in payload["events"]:
        bf = event["binding_feasibility"]
        assert bf["geometry_state"] == ABF.STATE_PASS
        assert bf["delivered_cue_state"] == ABF.STATE_UNMEASURED
        assert bf["delivered_cues_present"] is None
        assert "feasible" not in bf
        assert "measured" not in event
    assert payload["summary"]["events_delivered_cue_state_counts"] == {ABF.STATE_UNMEASURED: 2}
    assert "events_binding_feasible_placeholder" not in payload["summary"]


def test_single_good_frame_does_not_pass_geometry(tmp_path):
    b_track = [45.0] * FRAMES
    b_track[20] = -40.0  # 说话窗 15–30 里只有这一帧分得开
    payload = _run(tmp_path, _facts({"a": {"colour": "blue", "az": 40.0}, "b": {"colour": "green", "az": b_track}}),
                   placements=[(1.0, 40.0, 1), (3.0, 45.0, 2)])
    sep = payload["events"][0]["separation_over_state_window_deg"]
    assert sep["max"] == pytest.approx(80.0, abs=1e-6)
    assert sep["min"] == pytest.approx(5.0, abs=1e-6)
    assert sep["sustained_s_above_theta"] == pytest.approx(1 / FPS, abs=1e-9)
    assert payload["events"][0]["binding_feasibility"]["geometry_state"] == ABF.STATE_FAIL


def test_front_source_with_lateral_competitor_passes_relative_cue(tmp_path):
    facts = _facts({"a": {"colour": "blue", "az": 0.0}, "b": {"colour": "green", "az": 60.0}})
    payload = _run(tmp_path, facts, placements=[(1.0, 0.0, 1), (3.0, 60.0, 2)])
    e1, e2 = payload["events"]
    # 正前方：实测左右线索接近零是预期的，不能因此判"无空间信息"。
    assert abs(e1["measured"]["itd_onset_ms"]) < 0.05
    assert abs(e1["measured"]["ild_2_6k_db"]) < 0.5
    bf = e1["binding_feasibility"]
    assert bf["cue_basis"]["target_near_axis"] is True
    assert bf["cue_basis"]["pairwise_expected_itd_diff_ms"] == pytest.approx(ABF.woodworth_itd_ms(60.0), abs=1e-9)
    assert bf["delivered_cue_state"] == ABF.STATE_PASS
    assert bf["geometry_state"] == ABF.STATE_PASS
    assert e2["binding_feasibility"]["delivered_cue_state"] == ABF.STATE_PASS


def test_stems_are_preferred_and_mixture_contamination_is_flagged(tmp_path):
    events = [
        {"event_id": "event_001", "actor_id": "a", "start_s": 1.0, "end_s": 2.5, "start_frame": 15, "end_frame": 37,
         "transcript": "one", "sound_class": "speech"},
        {"event_id": "event_002", "actor_id": "b", "start_s": 1.5, "end_s": 3.0, "start_frame": 22, "end_frame": 45,
         "transcript": "two", "sound_class": "speech"},
    ]
    n = int(FRAMES / FPS * SR)
    stem_a = np.zeros((n, 2)); stem_b = np.zeros((n, 2))
    _place(stem_a[:, 0], stem_a[:, 1], _burst(1), 1.0, 40.0)
    _place(stem_b[:, 0], stem_b[:, 1], _burst(2), 1.8, -40.0)  # 放置 1.5 秒起，实际 1.8 秒才出声
    payload = _run(tmp_path, _facts(events=events), placements=[(1.0, 40.0, 1), (1.8, -40.0, 2)],
                   stems={"a": stem_a, "b": stem_b})
    e1, e2 = payload["events"]
    assert set(payload["audio"]["per_source_stems"]) == {"a", "b"}
    for event in (e1, e2):
        assert event["measured_source"] == "per_source_stem"
        assert "measured_stem" in event and "measured_mixture" in event
        assert event["mixture_contaminated"] is True
        assert event["max_concurrent_speakers"] == 2
    assert e1["mixture_contaminating_event_ids"] == ["event_002"]
    assert e2["audible_window"]["status"] == "measured"
    assert e2["audible_window"]["onset_s"] == pytest.approx(1.8, abs=0.03)
    assert e2["window_used_for_states"] == "audible_window_from_stem"
    assert e2["state_window_frames"][0] == 27
    # stem 上量到的 b 的线索仍然是"左耳先到、左耳更响"。
    assert e2["measured"]["itd_onset_ms"] == pytest.approx(DELAY / SR * 1000.0, abs=0.07)
    assert e2["measured"]["ild_2_6k_db"] == pytest.approx(LEVEL_DB, abs=0.8)
    assert e2["binding_feasibility"]["delivered_cue_state"] == ABF.STATE_PASS
    assert payload["summary"]["events_measured_on_stem"] == 2
    assert payload["summary"]["events_mixture_contaminated"] == 2


def test_max_concurrent_uses_time_sweep_not_overlap_count(tmp_path):
    assert ABF.max_concurrent_entities([(0.0, 10.0, "t"), (1.0, 2.0, "b"), (8.0, 9.0, "c")]) == 2
    assert ABF.max_concurrent_entities([(0.0, 10.0, "t"), (1.0, 5.0, "b"), (4.0, 9.0, "c")]) == 3
    events = [
        {"event_id": "t", "actor_id": "a", "start_s": 0.0, "end_s": 5.0, "start_frame": 0, "end_frame": 75},
        {"event_id": "b1", "actor_id": "b", "start_s": 0.5, "end_s": 1.0, "start_frame": 7, "end_frame": 15},
        {"event_id": "c1", "actor_id": "c", "start_s": 4.0, "end_s": 4.5, "start_frame": 60, "end_frame": 67},
    ]
    facts = ABF.EpisodeFacts(_facts({"a": {"colour": "blue", "az": 40.0}, "b": {"colour": "green", "az": -40.0},
                                     "c": {"colour": "red", "az": 0.0}}, events=events))
    overlaps = ABF.event_overlaps(facts, events[0])
    assert len(overlaps["overlapping_events"]) == 2
    assert overlaps["max_concurrent_speakers"] == 2


def test_form_aware_divergence_for_qa13(tmp_path):
    questions = {"items": [
        {"qa_id": "QA-13", "question_id": "q13_open", "status": "pass", "available_forms": ["open"],
         "truth": {"value": 40.0}, "evidence": {"event_id": "event_001", "actor_id": "a", "query_frame": 80}},
        {"qa_id": "QA-13", "question_id": "q13_mcq", "status": "pass", "available_forms": ["mcq"],
         "truth": {"value": 40.0, "mcq_value": "front"}, "evidence": {"event_id": "event_001", "actor_id": "a", "query_frame": 80}},
        {"qa_id": "QA-13", "question_id": "q13_both", "status": "pass",
         "truth": {"value": 40.0, "mcq_value": "front"}, "evidence": {"event_id": "event_001", "actor_id": "a", "query_frame": 80}},
    ]}
    payload = _run(tmp_path, _facts(), questions)
    by_id = {q["question_id"]: q for q in payload["questions"]}
    open_only = by_id["q13_open"]
    assert list(open_only["divergence_by_form"]) == ["open"]
    assert open_only["divergence"]["gap_to_nearest_distractor_deg"] == pytest.approx(80.0, abs=1e-6)
    assert "degenerate_distractors_equal_gold" not in open_only["flags"]
    mcq_only = by_id["q13_mcq"]
    assert mcq_only["divergence"]["candidate_values"] == {"a": "front", "b": "front"}
    assert "degenerate_distractors_equal_gold" in mcq_only["flags"]
    both = by_id["q13_both"]
    assert set(both["divergence_by_form"]) == {"open", "mcq"}
    assert both["degenerate_forms"] == ["mcq"]
    assert payload["summary"]["flag_counts"]["degenerate_distractors_equal_gold"] == 2


def test_divergence_flags_degenerate_and_extrapolable_items(tmp_path):
    facts = _facts({"a": {"colour": "blue", "az": 40.0, "moving": (31, 40)}, "b": {"colour": "green", "az": -40.0}})
    questions = {"items": [
        {"qa_id": "QA-24", "question_id": "q24", "status": "pass",
         "truth": {"value": "visible_clear", "mcq_value": "visible_clear"},
         "evidence": {"anchor_event": {"event_id": "event_001"}, "target_actor_id": "a", "final_frame": FRAMES - 1}},
        {"qa_id": "QA-17", "question_id": "q17", "status": "pass",
         "truth": {"value": "yes", "mcq_value": "yes"},
         "evidence": {"event_id": "event_001", "actor_id": "a", "motion_frames": [30, 60]}},
        {"qa_id": "QA-01", "question_id": "q01", "status": "pass",
         "truth": {"value": "yes", "mcq_value": "yes"},
         "evidence": {"target_actor_id": "a", "event_ids": ["event_001"]}},
    ]}
    payload = _run(tmp_path, facts, questions)
    by_id = {q["qa_id"]: q for q in payload["questions"]}
    q24 = by_id["QA-24"]
    assert q24["divergence"]["candidate_values"] == {"a": "visible_clear", "b": "visible_clear"}
    assert q24["divergence"]["degenerate_all_same"] is True
    assert "degenerate_distractors_equal_gold" in q24["flags"]
    q17 = by_id["QA-17"]
    assert q17["divergence"]["candidate_values"] == {"a": "yes", "b": "no"}
    assert q17["divergence"]["degenerate_all_same"] is False
    assert q17["temporal"]["target_moving_at_event_end"] is False
    assert "post_sound_answer_extrapolable_from_trend" not in q17["flags"]
    assert "no_silent_entity" in by_id["QA-01"]["flags"]
    assert "binding_geometry_candidate_fail" not in q24["flags"]


def test_unique_minority_field_and_structural_baselines(tmp_path):
    facts = _facts({"a": {"colour": "blue", "az": 40.0}, "b": {"colour": "green", "az": -40.0, "vis": "hidden"},
                    "c": {"colour": "red", "az": 0.0, "vis": "hidden"}})
    questions = {"items": [
        {"qa_id": "QA-24", "question_id": "q24", "status": "pass", "available_forms": ["mcq"],
         "truth": {"value": "visible_clear", "mcq_value": "visible_clear"},
         "evidence": {"anchor_event": {"event_id": "event_001"}, "target_actor_id": "a", "final_frame": FRAMES - 1}},
    ]}
    payload = _run(tmp_path, facts, questions)
    q24 = payload["questions"][0]
    assert q24["divergence"]["gold_is_majority"] is False
    assert q24["divergence"]["gold_is_unique_minority"] is True
    assert q24["divergence"]["candidate_value_multiplicity"] == {'"visible_clear"': 1, '"hidden"': 2}
    assert "gold_is_unique_minority_among_candidates" in q24["flags"]
    assert "degenerate_distractors_equal_gold" not in q24["flags"]
    baselines = payload["summary"]["structural_baselines"]
    assert baselines["applicable_binding_mcq_questions"] == 1
    assert baselines["unique_minority_strategy_hits"] == 1
    assert baselines["majority_value_strategy_hits"] == 0


def _write_package(root: Path, *, x_range=(0.0, 3.0), up_axis="+Y") -> Path:
    (root / "acoustic").mkdir(parents=True)
    x0, x1 = x_range
    vertices = np.array([[x0, 0.0, -1.0], [x1, 0.0, -1.0], [x1, 3.0, -1.0], [x0, 3.0, -1.0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    np.save(root / "acoustic" / "vertices.npy", vertices)
    np.save(root / "acoustic" / "triangles.npy", triangles)
    manifest = {
        "schema": "avengine_acoustic_scene_package_v1",
        "coordinate_system": {"forward_axis": "-Z", "handedness": "right", "linear_unit": "meter",
                              "quaternion_order": "xyzw", "up_axis": up_axis},
        "arrays": {"vertices": {"path": "acoustic/vertices.npy"}, "triangles": {"path": "acoustic/triangles.npy"}},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root / "manifest.json"


def test_line_of_sight_from_static_geometry(tmp_path):
    # 墙只挡右半边（x 从 0 到 3，z = -1）：a 在右前方被挡，b 在左前方直达。
    package = _write_package(tmp_path / "pkg")
    payload = _run(tmp_path, _facts(), package=package)
    assert payload["static_geometry"]["status"] == "loaded"
    e1, e2 = payload["events"]
    assert e1["line_of_sight"]["state"] == "blocked"
    assert e2["line_of_sight"]["state"] == "clear"
    assert set(e1["line_of_sight"]["frames"].values()) == {"blocked"}
    assert payload["summary"]["events_line_of_sight_counts"] == {"blocked": 1, "clear": 1}


def test_line_of_sight_refuses_non_metre_y_up_package(tmp_path):
    package = _write_package(tmp_path / "pkg", up_axis="+Z")
    payload = _run(tmp_path, _facts(), package=package)
    assert payload["static_geometry"]["status"] == "unusable"
    for event in payload["events"]:
        assert event["line_of_sight"]["state"] == ABF.STATE_UNMEASURED
        assert "metres/Y-up" in event["line_of_sight"]["reason"]


def test_float_wav_reader_matches_pcm(tmp_path):
    data = _mixture([(1.0, 40.0, 1)])
    _write_float_wav(tmp_path / "f.wav", data)
    _write_pcm_wav(tmp_path / "p.wav", data)
    sr_f, read_f = ABF.read_wav(tmp_path / "f.wav")
    sr_p, read_p = ABF.read_wav(tmp_path / "p.wav")
    assert sr_f == sr_p == SR
    assert read_f.shape == read_p.shape == data.shape
    assert np.max(np.abs(read_f - data)) < 1e-6
    assert np.max(np.abs(read_p - data)) < 1e-4


def test_output_is_no_clobber(tmp_path):
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(_facts()))
    wav_path = tmp_path / "mix.wav"
    _write_pcm_wav(wav_path, _mixture([(1.0, 40.0, 1), (3.0, -40.0, 2)]))
    out = tmp_path / "audit.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        ABF.main(["--facts", str(facts_path), "--wav", str(wav_path), "--out", str(out)])
    assert out.read_text() == "{}"


def test_azimuth_formula_matches_unified_catalog(tmp_path):
    pytest.importorskip("avengine.qa.unified_catalog")
    payload = _run(tmp_path, _facts())
    check = payload["azimuth_formula_crosscheck"]
    assert check["status"] == "matched", check
    assert check["frames_compared"] == 2 * FRAMES

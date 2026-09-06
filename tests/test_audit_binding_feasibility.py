"""绑定可行性审计：几何、成片双耳线索、分歧判定都要在已知答案的合成数据上对得上。

合成一段 6 秒、两人、两句话的 Episode：a 在右前 40 度，b 在左前 40 度，都不动。WAV 里
a 那句让右声道先到 6 个采样（0.375 ms）并比左声道响 6 dB，b 那句镜像。工具必须量回
这两个数、判两人几何上分得开；把 b 挪到离 a 5 度以内，就必须判不可行。逐题分歧审计用
手写的 questions 条目：片尾两人同为 visible_clear 的 QA-24 必须标退化，只有一人动的 QA-17
不能标退化。
"""

from __future__ import annotations

import json
import math
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


def _pos(azimuth_deg: float, distance_m: float = 2.0) -> list[float]:
    theta = math.radians(azimuth_deg)
    return [distance_m * math.sin(theta), 1.6, -distance_m * math.cos(theta)]


def _facts(b_azimuth: float = -40.0, a_moving_frames: tuple[int, int] | None = None) -> dict:
    def actor(actor_id: str, colour: str, azimuth: float, moving: tuple[int, int] | None) -> dict:
        flags = [False] * FRAMES
        if moving:
            for f in range(moving[0], moving[1] + 1):
                flags[f] = True
        return {
            "actor_id": actor_id,
            "appearance": {"field": "top_color", "value": colour, "label": f"Human ({colour} top)"},
            "root_positions_m": [_pos(azimuth)] * FRAMES,
            "emitter_positions_m": [_pos(azimuth)] * FRAMES,
            "moving": flags,
        }

    visibility = {
        actor_id: {
            str(f): {"frame_index": f, "state": "visible_clear", "occlusion_fraction": 0.0,
                     "visible_pixels": 1000, "target_pixels": 1000}
            for f in range(FRAMES)
        }
        for actor_id in ("a", "b")
    }
    return {
        "schema": "avengine_qa_unified_episode_facts_v1",
        "episode_id": "synthetic_binding_audit",
        "time": {"frame_count": FRAMES, "frame_rate_hz": FPS, "sample_rate_hz": SR,
                 "sample_count": int(FRAMES / FPS * SR), "duration_seconds": FRAMES / FPS},
        "actors": {"a": actor("a", "blue", 40.0, a_moving_frames), "b": actor("b", "green", b_azimuth, None)},
        "events": [
            {"event_id": "event_001", "actor_id": "a", "start_s": 1.0, "end_s": 2.0, "start_frame": 15, "end_frame": 30,
             "transcript": "one", "sound_class": "speech"},
            {"event_id": "event_002", "actor_id": "b", "start_s": 3.0, "end_s": 4.0, "start_frame": 45, "end_frame": 60,
             "transcript": "two", "sound_class": "speech"},
        ],
        "listener": {
            "status": "pass",
            "positions_m": [[0.0, 1.5, 0.0]] * FRAMES,
            "yaw_deg": [0.0] * FRAMES,
            "basis_m3": [{"forward": [0.0, 0.0, -1.0], "right": [1.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0]}] * FRAMES,
        },
        "audio": {"status": "pass", "channel_count": 2, "sample_rate_hz": SR, "wet_tail_intervals": []},
        "visibility": visibility,
        "appearance_review": {
            "a": {"status": "reviewed", "value": "blue", "frame_refs": [0, 10]},
            "b": {"status": "reviewed", "value": "green", "frame_refs": [0]},
        },
    }


def _write_wav(path: Path) -> None:
    rng = np.random.default_rng(20260906)
    n = int(FRAMES / FPS * SR)
    left = np.zeros(n)
    right = np.zeros(n)
    gain = 10 ** (-LEVEL_DB / 20.0)
    burst = rng.standard_normal(SR) * 0.2  # one second of noise
    # event_001: a on the right -> right ear leads and is louder.
    s = SR
    right[s:s + SR] += burst
    left[s + DELAY:s + DELAY + SR] += burst * gain
    # event_002: b on the left -> mirror.
    s = 3 * SR
    left[s:s + SR] += burst
    right[s + DELAY:s + DELAY + SR] += burst * gain
    data = np.stack([left, right], axis=1)
    pcm = np.clip(data * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def _run(tmp_path: Path, facts: dict, questions: dict | None = None) -> dict:
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(facts))
    wav_path = tmp_path / "mix.wav"
    _write_wav(wav_path)
    q_path = None
    if questions is not None:
        q_path = tmp_path / "questions.json"
        q_path.write_text(json.dumps(questions))
    return ABF.run_audit(facts_path, wav_path, q_path, dict(ABF.DEFAULT_THRESHOLDS))


def test_geometry_and_delivered_cues_are_recovered(tmp_path):
    payload = _run(tmp_path, _facts())
    e1, e2 = payload["events"]
    assert e1["onset_azimuth_deg"] == pytest.approx(40.0, abs=1e-6)
    assert e2["onset_azimuth_deg"] == pytest.approx(-40.0, abs=1e-6)
    assert e1["separation_to_nearest_competitor_deg"]["onset"] == pytest.approx(80.0, abs=1e-6)
    # a 在右边：左声道更弱、更晚。
    assert e1["measured"]["ild_2_6k_db"] == pytest.approx(-LEVEL_DB, abs=0.8)
    assert e1["measured"]["itd_onset_ms"] == pytest.approx(-DELAY / SR * 1000.0, abs=0.07)
    assert e2["measured"]["ild_2_6k_db"] == pytest.approx(LEVEL_DB, abs=0.8)
    assert e2["measured"]["itd_onset_ms"] == pytest.approx(DELAY / SR * 1000.0, abs=0.07)
    for event in (e1, e2):
        assert event["binding_feasibility"]["geometry_separable"] is True
        assert event["binding_feasibility"]["delivered_cues_present"] is True
        assert event["binding_feasibility"]["feasible"] is True
    assert payload["audio"]["exact_zero_sample_fraction"] > 0.5
    assert payload["azimuth_convention"] == ABF.AZIMUTH_CONVENTION


def test_close_static_speakers_are_not_bindable(tmp_path):
    payload = _run(tmp_path, _facts(b_azimuth=45.0))
    e1 = payload["events"][0]
    assert e1["separation_to_nearest_competitor_deg"]["max"] == pytest.approx(5.0, abs=1e-6)
    assert e1["separation_to_nearest_competitor_deg"]["change"] == pytest.approx(0.0, abs=1e-9)
    assert e1["binding_feasibility"]["geometry_separable"] is False
    assert e1["binding_feasibility"]["feasible"] is False
    assert any("azimuth" in reason for reason in e1["binding_feasibility"]["reasons"])


def test_divergence_flags_degenerate_and_extrapolable_items(tmp_path):
    facts = _facts(a_moving_frames=(31, 40))
    questions = {"items": [
        {"qa_id": "QA-24", "question_id": "q24", "status": "pass",
         "truth": {"value": "visible_clear", "mcq_value": "visible_clear"},
         "evidence": {"anchor_event": {"event_id": "event_001"}, "target_actor_id": "a", "final_frame": FRAMES - 1}},
        {"qa_id": "QA-17", "question_id": "q17", "status": "pass",
         "truth": {"value": "yes", "mcq_value": "yes"},
         "evidence": {"event_id": "event_001", "actor_id": "a", "motion_frames": [30, 60]}},
        {"qa_id": "QA-13", "question_id": "q13", "status": "pass",
         "truth": {"value": 40.0, "mcq_value": "front"},
         "evidence": {"event_id": "event_001", "actor_id": "a", "query_frame": 80}},
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
    q13 = by_id["QA-13"]
    assert q13["temporal"]["post_sound_displacement_deg"] == pytest.approx(0.0, abs=1e-6)
    assert "post_sound_answer_equals_last_heard_direction" in q13["flags"]
    # ±40° 都落在"前"扇区，所以这道 QA-13 的扇区答案同样不需要绑定，必须也标退化。
    assert q13["divergence"]["candidate_values"] == {"a": "front", "b": "front"}
    assert "degenerate_distractors_equal_gold" in q13["flags"]
    assert "no_silent_entity" in by_id["QA-01"]["flags"]
    assert payload["summary"]["flag_counts"]["degenerate_distractors_equal_gold"] == 2


def test_output_is_no_clobber(tmp_path):
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(_facts()))
    wav_path = tmp_path / "mix.wav"
    _write_wav(wav_path)
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

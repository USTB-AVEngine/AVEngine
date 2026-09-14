"""Contracts for counting how many sound events one recording holds.

The signals are synthesised so the answer is known before the detector runs:
one 0.6 s tone is one event; two of them three seconds apart are two; six of
them on a fixed 1.5 s spacing are one telephone ringing, not six telephones.
A detector that disagrees with any of those is wrong, and a threshold moved
until it agrees is the thing this module exists to make visible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from avengine.assets.sound_prepare import write_wav_mono_no_clobber
from avengine.dataset.event_segmentation import (
    DEFAULT_GAP_MAX_S,
    METHOD,
    STATUS_MULTI,
    STATUS_PASS,
    STATUS_SILENT,
    RULE_GAP_EXCEEDS_MAX,
    RULE_GAPS_WITHIN_MAX,
    RULE_NO_ONSET,
    RULE_PULSE_TRAIN,
    RULE_SINGLE_RUN,
    EventSegmentationError,
    backfill_pool,
    is_speech_like,
    segment_audio_file,
    segment_recording,
    uniform_policy,
)

RATE = 16000


def _tone(duration_s: float, *, amplitude: float = 0.5, hz: float = 440.0) -> np.ndarray:
    count = int(round(duration_s * RATE))
    t = np.arange(count, dtype=np.float64) / RATE
    return amplitude * np.sin(2.0 * np.pi * hz * t)


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(round(duration_s * RATE)), dtype=np.float64)


def _bursts(burst_s: float, gaps_s: list[float], *, lead_s: float = 0.1) -> np.ndarray:
    parts = [_silence(lead_s), _tone(burst_s)]
    for gap in gaps_s:
        parts.append(_silence(gap))
        parts.append(_tone(burst_s))
    parts.append(_silence(0.1))
    return np.concatenate(parts)


def test_one_continuous_tone_is_one_event() -> None:
    record = segment_recording(_bursts(0.6, []), RATE, sound_class="dog_bark")
    assert record["status"] == STATUS_PASS
    assert record["event_count"] == 1
    assert record["rule"] == RULE_SINGLE_RUN
    assert record["certification"] == "automatic"
    assert record["human_certified"] is False
    assert record["human_review"]["count"] == 0
    assert record["measurement"]["measured_group_count"] == 1


def test_two_bursts_three_seconds_apart_are_not_certified() -> None:
    record = segment_recording(_bursts(0.6, [3.0]), RATE, sound_class="dog_bark")
    assert record["status"] == STATUS_MULTI
    assert record["rule"] == RULE_GAP_EXCEEDS_MAX
    assert record["event_count"] == 2
    assert record["certification"] is None
    assert record["certified_by"] is None
    assert record["crop_invariant"] is False
    assert record["measurement"]["max_group_silence_s"] == pytest.approx(3.0, abs=0.05)


def test_a_short_gap_inside_one_sound_stays_one_event() -> None:
    record = segment_recording(_bursts(0.6, [0.3]), RATE, sound_class="dog_bark")
    assert record["status"] == STATUS_PASS
    assert record["event_count"] == 1
    assert record["rule"] == RULE_GAPS_WITHIN_MAX
    assert record["measurement"]["sounding_run_count"] == 2
    assert record["measurement"]["measured_group_count"] == 1


def test_a_regular_pulse_train_is_one_repeating_sound() -> None:
    record = segment_recording(
        _bursts(0.3, [1.5] * 5), RATE, sound_class="telephone_bell_ringing"
    )
    assert record["status"] == STATUS_PASS
    assert record["event_count"] == 1
    assert record["rule"] == RULE_PULSE_TRAIN
    assert record["measurement"]["measured_group_count"] == 6
    assert record["measurement"]["group_onset_spacing_is_regular"] is True
    assert record["measurement"][
        "group_onset_period_coefficient_of_variation"
    ] == pytest.approx(0.0, abs=0.02)


def test_two_bursts_are_never_a_pulse_train() -> None:
    """Two onsets cannot show a spacing is repeated, so they stay uncertified."""

    record = segment_recording(_bursts(0.3, [1.5]), RATE, sound_class="phone_ring")
    assert record["status"] == STATUS_MULTI
    assert record["measurement"]["measured_group_count"] == 2
    assert record["measurement"]["group_onset_spacing_is_regular"] is False


def test_irregular_bursts_are_not_a_pulse_train() -> None:
    record = segment_recording(_bursts(0.3, [1.4, 3.1, 1.6]), RATE, sound_class="drip")
    assert record["status"] == STATUS_MULTI
    assert record["event_count"] == 4
    assert record["measurement"]["group_onset_spacing_is_regular"] is False


def test_silence_is_not_certified_as_one_event() -> None:
    record = segment_recording(_silence(4.0), RATE, sound_class="drip")
    assert record["status"] == STATUS_SILENT
    assert record["rule"] == RULE_NO_ONSET
    assert record["event_count"] == 0
    assert record["certification"] is None


def test_the_threshold_is_the_only_thing_that_moves_the_verdict() -> None:
    signal = _bursts(0.4, [1.2])
    strict = segment_recording(signal, RATE, sound_class="drip", gap_max_s=1.0)
    lenient = segment_recording(signal, RATE, sound_class="drip", gap_max_s=1.5)
    assert strict["status"] == STATUS_MULTI
    assert lenient["status"] == STATUS_PASS
    assert strict["parameters"]["gap_max_s"] == 1.0
    assert lenient["parameters"]["gap_max_s"] == 1.5


def test_every_class_is_measured_through_the_same_gate() -> None:
    """No per-class branch: two classes differ only by the label they carry."""

    first = uniform_policy("dog_bark")
    second = uniform_policy("air_conditioning")
    third = uniform_policy(None)
    for policy in (first, second, third):
        assert policy["band"] == "full_band"
        assert policy["gate"] == "peak_relative"
        assert policy["relative_peak_db"] == -25.0
        assert policy["uniform_policy"] == METHOD
    ignore = {"sound_class"}
    assert {k: v for k, v in first.items() if k not in ignore} == {
        k: v for k, v in second.items() if k not in ignore
    }
    assert {k: v for k, v in first.items() if k not in ignore} == {
        k: v for k, v in third.items() if k not in ignore
    }


def test_the_same_signal_gets_the_same_verdict_under_every_class_label() -> None:
    signal = _bursts(0.5, [0.4, 0.4])
    verdicts = {
        name: segment_recording(signal, RATE, sound_class=name)["status"]
        for name in ("dog_bark", "air_conditioning", "telephone", "smoke_alarm", None)
    }
    assert set(verdicts.values()) == {STATUS_PASS}


def test_speech_is_the_set_qa_23_exempts() -> None:
    assert is_speech_like({"sound_class": "speech_playback"}) is True
    assert is_speech_like({"sound_class": "Speech"}) is True
    assert is_speech_like({"sound_class": "dog_bark", "transcript": "hi"}) is True
    assert is_speech_like({"sound_class": "dog_bark"}) is False
    assert is_speech_like({"sound_class": "dog_bark", "transcript": None}) is False
    assert is_speech_like({"sound_class": "dog_bark", "transcript": ""}) is False


def _catalog_fixture():
    import importlib.util

    path = Path(__file__).with_name("test_qa_unified_catalog.py")
    spec = importlib.util.spec_from_file_location("qa_unified_catalog_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._fixture


@pytest.mark.parametrize(
    "sound_class",
    ["speech", "speech_playback", "utterance", "human_speech", "dog_bark", "printer"],
)
def test_speech_exemption_matches_what_qa_23_actually_skips(sound_class: str) -> None:
    """The backfill must skip exactly what QA-23 skips, not a class list.

    Checked against the catalog's behaviour rather than against a copy of its
    source, so the two drifting apart shows up here as a failing question.
    """

    from avengine.qa.unified_catalog import generate_unified_questions

    raw = _catalog_fixture()()
    for event in raw["audio_program"]["events"]:
        event.pop("event_segmentation_status", None)
        event.pop("transcript", None)
        event["sound_class"] = sound_class
    result = generate_unified_questions(raw, qa_ids=["QA-23"])
    emitted = bool(result["items"])
    assert emitted is is_speech_like({"sound_class": sound_class})
    if not emitted:
        assert result["deferred"][0]["code"] == "event_segmentation_not_reviewed"


def _write_pool(tmp_path: Path) -> Path:
    rows = [
        ("one_event", "dog_bark", _bursts(0.6, []), None),
        ("two_events", "drip", _bursts(0.4, [3.0]), None),
        ("a_sentence", "speech_playback", _bursts(0.5, [3.0]), "hello there"),
    ]
    sounds = []
    for name, sound_class, samples, transcript in rows:
        path = tmp_path / "audio" / f"{name}.wav"
        write_wav_mono_no_clobber(path, samples, RATE)
        row = {
            "sound_asset_id": f"sound_{name}_v1",
            "sound_class": sound_class,
            "path": str(path),
            "sample_rate_hz": RATE,
        }
        if transcript is not None:
            row["transcript"] = transcript
        sounds.append(row)
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"schema": "test_pool", "sounds": sounds}, indent=1))
    return pool


def test_backfill_adds_records_without_touching_the_input(tmp_path: Path) -> None:
    pool_path = _write_pool(tmp_path)
    before = pool_path.read_bytes()
    payload = json.loads(pool_path.read_text())
    out, sidecar = backfill_pool(payload, pool_path=pool_path)

    assert pool_path.read_bytes() == before
    assert payload["sounds"][0].get("event_segmentation") is None

    by_id = {row["sound_asset_id"]: row for row in out["sounds"]}
    assert by_id["sound_one_event_v1"]["event_segmentation"]["status"] == STATUS_PASS
    assert by_id["sound_two_events_v1"]["event_segmentation"]["status"] == STATUS_MULTI
    assert "event_segmentation" not in by_id["sound_a_sentence_v1"]

    statistics = out["event_segmentation_backfill"]["statistics"]
    assert statistics["counts"] == {
        "rows": 3, "measured": 2, "skipped_speech": 1,
        "skipped_existing": 0, "failed": 0,
    }
    assert statistics["certified"] == 1
    assert statistics["not_certified"] == 1
    assert statistics["human_reviewed"] == 0
    assert set(sidecar["by_sound_asset_id"]) == {
        "sound_one_event_v1", "sound_two_events_v1"
    }
    assert sidecar["parameters"]["gap_max_s"] == DEFAULT_GAP_MAX_S
    assert out["event_segmentation_backfill"]["human_certified"] is False


def test_backfill_copies_every_other_field_verbatim(tmp_path: Path) -> None:
    pool_path = _write_pool(tmp_path)
    payload = json.loads(pool_path.read_text())
    out, _sidecar = backfill_pool(payload, pool_path=pool_path)
    assert set(out) - set(payload) == {"event_segmentation_backfill"}
    for before, after in zip(payload["sounds"], out["sounds"]):
        assert set(after) - set(before) <= {"event_segmentation"}
        for key, value in before.items():
            assert after[key] == value


def test_backfill_keeps_an_existing_record_unless_asked(tmp_path: Path) -> None:
    pool_path = _write_pool(tmp_path)
    payload = json.loads(pool_path.read_text())
    payload["sounds"][0]["event_segmentation"] = {"status": "reviewed", "event_count": 1}
    out, _sidecar = backfill_pool(payload, pool_path=pool_path)
    assert out["sounds"][0]["event_segmentation"] == {
        "status": "reviewed", "event_count": 1
    }
    assert out["event_segmentation_backfill"]["statistics"]["counts"][
        "skipped_existing"
    ] == 1
    again, _ = backfill_pool(payload, pool_path=pool_path, overwrite_existing=True)
    assert again["sounds"][0]["event_segmentation"]["method"] == METHOD


def test_backfill_can_include_speech_when_asked(tmp_path: Path) -> None:
    pool_path = _write_pool(tmp_path)
    payload = json.loads(pool_path.read_text())
    out, sidecar = backfill_pool(payload, pool_path=pool_path, include_speech=True)
    assert len(sidecar["by_sound_asset_id"]) == 3
    assert out["event_segmentation_backfill"]["statistics"]["counts"][
        "skipped_speech"
    ] == 0


def test_a_row_without_an_id_is_refused(tmp_path: Path) -> None:
    pool_path = _write_pool(tmp_path)
    payload = json.loads(pool_path.read_text())
    payload["sounds"][0].pop("sound_asset_id")
    with pytest.raises(EventSegmentationError):
        backfill_pool(payload, pool_path=pool_path)


def test_segment_audio_file_reads_the_written_file(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    write_wav_mono_no_clobber(path, _bursts(0.6, []), RATE)
    record = segment_audio_file(path, sound_class="cat_meow")
    assert record["status"] == STATUS_PASS
    assert record["source"]["path"] == str(path.resolve())
    assert record["source"]["read"] == "read_only"
    assert record["activity_is_qa_event_count"] is True

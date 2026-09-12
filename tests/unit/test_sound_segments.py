"""Contracts for choosing and cutting the sounding part of a recording.

The fixtures are synthesised rather than read from the library so the answer
is known in advance: a 0.4 s tone at second 1 and second 3 of an otherwise
silent file has exactly 0.8 s of sound, and any detector that says otherwise
is wrong.  Real-library behaviour is measured separately and reported with the
task evidence; these tests fix the rules.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from avengine.dataset.sound_segments import (
    ABSOLUTE_ACTIVE_FLOOR_DBFS,
    BOUND_DETERMINING_BUDGET_FIELDS,
    CROP_AUTHORIZATION,
    PCM_DETERMINING_BUDGET_FIELDS,
    SegmentBudget,
    SourceRecording,
    pause_structure,
    SoundSegmentError,
    cross_reference_sound_events,
    iter_library_clips,
    materialize_segment,
    measure_activity,
    plan_segment,
    plan_segments,
    pool_row,
    prepare_segments,
    read_source_pcm,
    segment_id,
    segment_policy_for_class,
    select_segment,
    verify_segment_artifact,
)

RATE = 16000


def _tone(duration_s: float, *, freq: float = 440.0, amplitude: float = 0.5,
          rate: int = RATE) -> np.ndarray:
    t = np.arange(int(round(duration_s * rate))) / rate
    return amplitude * np.sin(2 * np.pi * freq * t)


def _silence(duration_s: float, *, rate: int = RATE) -> np.ndarray:
    return np.zeros(int(round(duration_s * rate)))


def _noise(duration_s: float, *, amplitude: float, rate: int = RATE,
           seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return amplitude * rng.standard_normal(int(round(duration_s * rate)))


def _write(path: Path, samples: np.ndarray, *, rate: int = RATE,
           sound_class: str | None = None, human_review: dict | None = None,
           transcript: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ints = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(ints.tobytes())
    if sound_class is not None:
        payload: dict = {"event_classes": [sound_class], "dry": True}
        if human_review is not None:
            payload["human_review"] = human_review
        if transcript is not None:
            payload["transcript"] = transcript
        path.with_suffix(".json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    return path


# --------------------------------------------------------------------------
# policy resolution
# --------------------------------------------------------------------------

def test_family_comes_from_the_existing_activity_profile():
    """No second class list: the families are the ones sound_prepare declares."""

    assert segment_policy_for_class("speech_playback")["activity_family"] == "speech"
    assert segment_policy_for_class("dog_bark")["activity_family"] == "animal_call"
    assert segment_policy_for_class("doorbell")["activity_family"] == "short_prompt"
    assert (segment_policy_for_class("air_conditioning")["activity_family"]
            == "device_continuous")


def test_a_continuous_device_is_never_gated_against_its_own_noise_floor():
    """The rule that stops an air conditioner being deleted for sounding like noise."""

    assert segment_policy_for_class("air_conditioning")["gate"] == "peak_relative"
    assert segment_policy_for_class("microwave_hum")["gate"] == "peak_relative"
    assert (segment_policy_for_class("dog_bark")["gate"]
            == "noise_floor_hysteresis")


def test_an_unmapped_class_falls_through_to_never_cut_the_middle():
    policy = segment_policy_for_class("dial_tone")
    assert policy["activity_family"] == "unknown"
    assert policy["gate"] == "peak_relative"


def test_caller_may_override_the_family_and_the_thresholds():
    policy = segment_policy_for_class(
        "dial_tone",
        overrides={"activity_family": "short_prompt", "max_internal_silence_s": 0.9},
    )
    assert policy["activity_family"] == "short_prompt"
    assert policy["max_internal_silence_s"] == pytest.approx(0.9)
    assert policy["activity_family_source"] == "explicit_override"


def test_an_unknown_policy_key_is_refused_rather_than_ignored():
    with pytest.raises(SoundSegmentError, match="unknown policy override"):
        segment_policy_for_class("dog_bark", overrides={"max_gap_s": 1.0})


# --------------------------------------------------------------------------
# activity measurement
# --------------------------------------------------------------------------

def test_activity_is_windowed_energy_not_a_nonzero_sample_count():
    """A tone burst plus dither everywhere: every sample is non-zero, 0.4 s sounds."""

    samples = np.concatenate([_silence(1.0), _tone(0.4), _silence(1.0)])
    samples = samples + _noise(len(samples) / RATE, amplitude=1e-4)
    assert np.count_nonzero(samples) == samples.size
    facts = measure_activity(samples, RATE, sound_class="doorbell")
    assert facts["active_duration_s"] == pytest.approx(0.4, abs=0.05)
    assert facts["activity_coverage"] == pytest.approx(0.4 / 2.4, abs=0.03)


def test_a_loud_peak_does_not_make_the_whole_clip_sounding():
    samples = np.concatenate([_tone(0.3, amplitude=0.9), _silence(7.7)])
    facts = measure_activity(samples, RATE, sound_class="doorbell")
    assert facts["active_duration_s"] < 0.5
    assert facts["activity_coverage"] < 0.1
    assert facts["whole_clip_peak_dbfs"] > -2.0


def test_a_steady_hum_reads_as_sounding_throughout():
    """The air-conditioner case: no pause anywhere, coverage must be ~1."""

    samples = _tone(8.0, freq=120.0, amplitude=0.2) + _noise(8.0, amplitude=0.02)
    facts = measure_activity(samples, RATE, sound_class="air_conditioning")
    assert facts["activity_coverage"] > 0.99
    assert facts["max_internal_silence_s"] == pytest.approx(0.0)


def test_a_millisecond_zero_crossing_is_not_a_silence():
    """Short dips are absorbed; only a real pause counts as internal silence."""

    tone = _tone(2.0, amplitude=0.5)
    tone[int(0.9 * RATE): int(0.9 * RATE) + 16] = 0.0  # 1 ms of zeros
    facts = measure_activity(tone, RATE, sound_class="air_conditioning")
    assert facts["max_internal_silence_s"] == pytest.approx(0.0)
    assert facts["activity_interval_count"] == 1


def test_unvoiced_style_low_level_high_band_energy_is_not_cut_out():
    """A quiet aperiodic burst between two vowels stays inside one interval."""

    samples = np.concatenate([
        _tone(0.5, freq=200.0, amplitude=0.5),
        _noise(0.12, amplitude=0.06),          # fricative-like, 18 dB quieter
        _tone(0.5, freq=200.0, amplitude=0.5),
    ])
    facts = measure_activity(samples, RATE, sound_class="speech_playback")
    assert facts["activity_interval_count"] == 1
    assert facts["max_internal_silence_s"] == pytest.approx(0.0)


def test_digital_silence_has_no_active_window():
    facts = measure_activity(_silence(3.0), RATE, sound_class="doorbell")
    assert facts["source_activity_intervals_samples"] == []
    assert facts["active_duration_s"] == 0.0


def test_background_noise_below_the_absolute_floor_has_no_active_window():
    """Room tone at an inaudible level is not sound, whatever its own peak is."""

    quiet = _noise(4.0, amplitude=10 ** ((ABSOLUTE_ACTIVE_FLOOR_DBFS - 20) / 20))
    facts = measure_activity(quiet, RATE, sound_class="air_conditioning")
    assert facts["active_duration_s"] == 0.0


def test_the_class_threshold_used_is_recorded_with_the_coverage():
    facts = measure_activity(_tone(2.0), RATE, sound_class="dog_bark")
    gate = facts["gate"]
    for key in ("mode", "threshold_dbfs", "peak_frame_dbfs",
                "noise_floor_frame_dbfs", "peak_relative_floor_dbfs",
                "absolute_floor_dbfs", "gate_floor_dbfs"):
        assert key in gate, key
    assert facts["frame"]["window_s"] > 0
    assert facts["activity_is_qa_event_count"] is False


def test_frame_levels_are_available_losslessly_when_asked():
    facts = measure_activity(_tone(0.5), RATE, sound_class="doorbell",
                             include_frame_levels=True)
    assert len(facts["frame_level_dbfs"]) == facts["frame"]["window_count"]
    assert len(facts["frame_active"]) == facts["frame"]["window_count"]


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def _plan(samples, sound_class, tmp_path, **budget_kwargs):
    path = _write(tmp_path / sound_class / "clip_a" / "clip.wav", samples,
                  sound_class=sound_class)
    budget = SegmentBudget(**budget_kwargs)
    return path, budget, plan_segment(path, budget=budget, sound_class=sound_class)


def test_a_long_hum_is_clamped_to_the_budget_and_stays_fully_active(tmp_path):
    samples = _tone(15.0, freq=120.0, amplitude=0.3) + _noise(15.0, amplitude=0.02)
    _path, _budget, plan = _plan(samples, "air_conditioning", tmp_path,
                                 max_duration_s=6.0)
    assert plan["status"] == "selected"
    assert plan["duration_s"] == pytest.approx(6.0, abs=0.01)
    assert plan["activity_coverage"] > 0.99
    assert plan["extension_stopped_because"] == "single_interval_exceeds_budget"


def test_two_bursts_far_apart_are_not_bridged_into_one_fake_sound(tmp_path):
    """A three second silence between two barks is not a continuous sound."""

    samples = np.concatenate([_silence(0.5), _tone(0.6), _silence(3.0),
                              _tone(0.6), _silence(0.5)])
    _path, _budget, plan = _plan(samples, "dog_bark", tmp_path, max_duration_s=6.0)
    assert plan["status"] == "selected"
    assert plan["extension_stopped_because"] == "gap_exceeds_max_internal_silence"
    # One burst plus its guard, not the 4.2 s that spans both of them.
    assert plan["duration_s"] < 0.8
    assert plan["activity_interval_count"] == 1
    assert plan["max_internal_silence_s"] <= plan["policy"]["max_internal_silence_s"]


def test_two_bursts_close_together_are_kept_in_one_contiguous_segment(tmp_path):
    samples = np.concatenate([_silence(0.5), _tone(0.6), _silence(0.3),
                              _tone(0.6), _silence(0.5)])
    _path, _budget, plan = _plan(samples, "dog_bark", tmp_path, max_duration_s=6.0)
    assert plan["status"] == "selected"
    assert plan["activity_interval_count"] == 2
    assert plan["max_internal_silence_s"] == pytest.approx(0.3, abs=0.05)
    assert plan["duration_s"] == pytest.approx(1.56, abs=0.08)


def test_the_segment_starts_and_ends_on_sound_within_the_crop_guard(tmp_path):
    samples = np.concatenate([_silence(2.0), _tone(1.0), _silence(2.0)])
    _path, _budget, plan = _plan(samples, "doorbell", tmp_path, max_duration_s=6.0)
    guard = plan["crop_guard_s_applied"]
    assert plan["leading_inactive_s"] == pytest.approx(guard, abs=0.011)
    assert plan["trailing_inactive_s"] == pytest.approx(guard, abs=0.011)


def test_a_segment_never_exceeds_the_budget(tmp_path):
    samples = np.concatenate([_tone(2.0), _silence(0.2)] * 5)
    for budget_s in (1.0, 2.5, 4.0):
        _p, _b, plan = _plan(samples, "air_conditioning", tmp_path,
                             max_duration_s=budget_s)
        if plan["status"] == "selected":
            assert plan["duration_s"] <= budget_s + 1e-9


def test_selection_is_deterministic(tmp_path):
    samples = np.concatenate([_silence(0.3), _tone(0.5), _silence(0.4),
                              _tone(0.5), _silence(0.3)])
    path = _write(tmp_path / "dog_bark" / "c" / "clip.wav", samples,
                  sound_class="dog_bark")
    budget = SegmentBudget(max_duration_s=5.0)
    first = plan_segment(path, budget=budget, sound_class="dog_bark")
    second = plan_segment(path, budget=budget, sound_class="dog_bark")
    assert (first["source_crop_start_sample"], first["source_crop_end_sample_exclusive"]) == (
        second["source_crop_start_sample"], second["source_crop_end_sample_exclusive"])


def test_every_candidate_considered_is_recorded_not_only_the_winner(tmp_path):
    samples = np.concatenate([_tone(0.3), _silence(1.5), _tone(0.9),
                              _silence(1.5), _tone(0.3)])
    _p, _b, plan = _plan(samples, "dog_bark", tmp_path, max_duration_s=6.0)
    assert len(plan["selection"]["candidates"]) >= 3
    assert plan["selection"]["selection_strategy"] == "longest_active"
    # longest_active picks the burst carrying the most sound, not the first.
    assert plan["active_duration_s"] == pytest.approx(0.9, abs=0.06)


def test_earliest_strategy_takes_the_first_qualifying_region(tmp_path):
    samples = np.concatenate([_tone(0.3), _silence(1.5), _tone(0.9)])
    path = _write(tmp_path / "dog_bark" / "c" / "clip.wav", samples,
                  sound_class="dog_bark")
    plan = plan_segment(
        path, budget=SegmentBudget(max_duration_s=6.0, selection="earliest"),
        sound_class="dog_bark",
        policy_overrides={"min_active_duration_s": 0.2},
    )
    assert plan["active_duration_s"] == pytest.approx(0.3, abs=0.06)


# --------------------------------------------------------------------------
# rejections - the counter-examples
# --------------------------------------------------------------------------

def test_digital_silence_is_rejected(tmp_path):
    _p, _b, plan = _plan(_silence(8.0), "air_conditioning", tmp_path,
                         max_duration_s=6.0)
    assert plan["status"] == "rejected"
    assert plan["reason"] == "no_active_window_above_absolute_floor"


def test_background_noise_only_is_rejected(tmp_path):
    quiet = _noise(8.0, amplitude=10 ** ((ABSOLUTE_ACTIVE_FLOOR_DBFS - 15) / 20))
    _p, _b, plan = _plan(quiet, "air_conditioning", tmp_path, max_duration_s=6.0)
    assert plan["status"] == "rejected"
    assert plan["reason"] in {
        "no_active_window_above_absolute_floor", "no_active_window_above_gate"
    }


def test_too_little_speech_is_rejected(tmp_path):
    """Half a second of speech does not make a speech segment."""

    samples = np.concatenate([_silence(2.0), _tone(0.5, freq=500.0),
                              _silence(2.0)])
    _p, _b, plan = _plan(samples, "speech_playback", tmp_path, max_duration_s=6.0)
    assert plan["status"] == "rejected"
    assert plan["reason"] == "active_duration_below_minimum"


def test_a_sound_shorter_than_the_callers_minimum_is_rejected(tmp_path):
    samples = np.concatenate([_silence(1.0), _tone(0.3), _silence(1.0)])
    _p, _b, plan = _plan(samples, "doorbell", tmp_path, max_duration_s=6.0,
                         min_duration_s=2.0)
    assert plan["status"] == "rejected"
    assert plan["reason"] == "segment_shorter_than_requested_minimum"


def test_a_sparse_ticking_clip_fails_the_continuous_coverage_rule(tmp_path):
    """A device class whose sound is a tick every second is not continuous."""

    one = np.concatenate([_tone(0.12, amplitude=0.5), _silence(0.88)])
    _p, _b, plan = _plan(np.tile(one, 8), "clock_tick", tmp_path,
                         max_duration_s=6.0)
    assert plan["status"] == "rejected"
    assert plan["reason"] == "activity_coverage_below_minimum"
    # and the caller can say so explicitly rather than being stuck with it
    path = tmp_path / "clock_tick" / "clip_a" / "clip.wav"
    fixed = plan_segment(path, budget=SegmentBudget(max_duration_s=6.0),
                         sound_class="clock_tick",
                         policy_overrides={"activity_family": "short_prompt"})
    assert fixed["status"] == "selected"


def test_a_rejected_plan_still_reports_its_measurement(tmp_path):
    one = np.concatenate([_tone(0.12, amplitude=0.5), _silence(0.88)])
    _p, _b, plan = _plan(np.tile(one, 8), "clock_tick", tmp_path,
                         max_duration_s=6.0)
    assert plan["activity"]["active_duration_s"] > 0
    assert plan["selection"]["candidates"]
    assert plan["selection"]["rejection_reasons"]


# --------------------------------------------------------------------------
# materialisation, provenance and read-back
# --------------------------------------------------------------------------

def _prepared(tmp_path, samples, sound_class, **budget_kwargs):
    path, budget, plan = _plan(samples, sound_class, tmp_path, **budget_kwargs)
    assert plan["status"] == "selected", plan.get("reason")
    record = materialize_segment(plan, tmp_path / "out", budget=budget)
    return plan, record


def test_the_original_recording_is_never_modified(tmp_path):
    samples = np.concatenate([_silence(1.0), _tone(2.0), _silence(1.0)])
    path = _write(tmp_path / "doorbell" / "c" / "clip.wav", samples,
                  sound_class="doorbell")
    before = path.read_bytes()
    budget = SegmentBudget(max_duration_s=6.0)
    plan = plan_segment(path, budget=budget, sound_class="doorbell")
    materialize_segment(plan, tmp_path / "out", budget=budget)
    assert path.read_bytes() == before


def test_the_written_segment_reads_back_as_the_plan_promised(tmp_path):
    samples = np.concatenate([_silence(1.0), _tone(2.0), _silence(1.0)])
    plan, record = _prepared(tmp_path, samples, "doorbell", max_duration_s=6.0)
    check = verify_segment_artifact(record)
    assert check["status"] in {"pass", "qualified"}, check["failed_checks"]
    assert check["failed_checks"] == []
    assert check["readback_selecting_gate_activity"]["activity_coverage"] == (
        pytest.approx(plan["activity_coverage"], abs=0.02)
    )


def test_verification_notices_a_replaced_artifact(tmp_path):
    samples = np.concatenate([_silence(1.0), _tone(2.0), _silence(1.0)])
    _plan_, record = _prepared(tmp_path, samples, "doorbell", max_duration_s=6.0)
    _write(Path(record["prepared_path"]), _silence(1.0))
    check = verify_segment_artifact(record)
    assert check["status"] == "fail"
    assert "prepared_sha256_matches" in check["failed_checks"]


def test_provenance_keeps_source_identity_rates_and_both_coordinate_systems(tmp_path):
    samples = np.concatenate([_silence(0.5), _tone(2.0), _silence(0.5)])
    path = _write(tmp_path / "doorbell" / "c" / "clip.wav", samples, rate=44100,
                  sound_class="doorbell")
    budget = SegmentBudget(max_duration_s=6.0, target_rate_hz=16000)
    plan = plan_segment(path, budget=budget, sound_class="doorbell")
    record = materialize_segment(plan, tmp_path / "out", budget=budget)
    assert record["source_path"] == str(path)
    assert record["source_sha256"]
    assert record["source_rate_hz"] == 44100
    assert record["target_rate_hz"] == 16000
    assert record["resample_ratio"] == "160/441"
    assert record["antialiased"] is True
    assert record["source_crop_end_sample_exclusive"] > record["source_crop_start_sample"]
    assert record["analysis_crop_end_sample_exclusive"] > record["analysis_crop_start_sample"]
    # source coordinates really do point at the same region of the original
    ratio = 44100 / 16000
    assert record["source_crop_start_sample"] == pytest.approx(
        record["analysis_crop_start_sample"] * ratio, abs=2)
    assert record["operation"]
    assert record["applied_gain_db"] == 0.0
    assert record["linear_gain"] == 1.0
    assert record["per_sample_loudness_levelling"] is False
    assert record["time_stretched"] is False
    assert record["concatenated_from_multiple_regions"] is False
    assert record["crop_authorization"] == CROP_AUTHORIZATION


def test_activity_intervals_are_recorded_in_segment_coordinates(tmp_path):
    samples = np.concatenate([_silence(1.0), _tone(1.0), _silence(1.0)])
    _p, record = _prepared(tmp_path, samples, "doorbell", max_duration_s=6.0)
    intervals = record["segment_activity_intervals_samples"]
    assert intervals
    assert intervals[0][0] >= 0
    assert intervals[-1][1] <= record["prepared_sample_count"]
    assert record["activity_interval_coordinates"][
        "segment_activity_intervals_samples"
    ].endswith("target_rate")


def test_a_segment_is_not_marked_truncated_by_a_detector_cap(tmp_path):
    """Authorised selection and the old unexplained-truncation defect differ."""

    samples = np.concatenate([_silence(1.0), _tone(2.0), _silence(1.0)])
    _p, record = _prepared(tmp_path, samples, "doorbell", max_duration_s=6.0)
    assert record["truncated"] is False
    assert record["truncation_reason"] is None
    assert record["selection_authorized"] is True
    assert pool_row(record)["truncated"] is False
    assert pool_row(record)["selection_authorized"] is True


def test_no_gain_is_applied_unless_the_caller_asks(tmp_path):
    samples = np.concatenate([_silence(0.5), _tone(1.5, amplitude=0.2),
                              _silence(0.5)])
    _p, record = _prepared(tmp_path, samples, "doorbell", max_duration_s=6.0)
    assert record["normalization_applied"] is False
    assert record["applied_gain_db"] == 0.0
    assert record["prepared_peak_dbfs"] == pytest.approx(-13.98, abs=0.3)


def test_a_resample_overshoot_is_attenuated_once_and_recorded(tmp_path):
    """A peak-normalised clip comes out of the resampler above full scale."""

    loud = np.clip(_tone(2.0, freq=997.0, amplitude=1.6), -1.0, 1.0)
    samples = np.concatenate([_silence(0.3, rate=44100),
                              np.clip(_tone(2.0, freq=997.0, amplitude=1.6,
                                            rate=44100), -1.0, 1.0),
                              _silence(0.3, rate=44100)])
    del loud
    path = _write(tmp_path / "doorbell" / "c" / "clip.wav", samples, rate=44100,
                  sound_class="doorbell")
    budget = SegmentBudget(max_duration_s=6.0)
    plan = plan_segment(path, budget=budget, sound_class="doorbell")
    record = materialize_segment(plan, tmp_path / "out", budget=budget)
    assert record["prepared_peak_dbfs"] <= 0.0
    if record["resample_overshoot_peak_dbfs"] is not None:
        assert record["resample_overshoot_peak_dbfs"] > 0.0
        assert record["resample_headroom_gain_db"] < 0.0
        assert record["linear_gain"] < 1.0
        assert record["gain_is_uniform_over_segment"] is True


def test_the_edge_fade_is_fixed_recorded_and_inside_the_crop_guard(tmp_path):
    samples = np.concatenate([_silence(1.0), _tone(2.0), _silence(1.0)])
    _p, record = _prepared(tmp_path, samples, "doorbell", max_duration_s=6.0,
                           edge_fade_s=0.004)
    assert record["edge_fade_s"] == pytest.approx(0.004)
    assert record["edge_fade_samples"] == 64
    assert record["edge_fade_shape"] == "raised_cosine"


def test_a_fade_longer_than_the_guard_is_refused(tmp_path):
    with pytest.raises(SoundSegmentError, match="edge_fade_s"):
        SegmentBudget(max_duration_s=5.0, crop_guard_s=0.01,
                      edge_fade_s=0.05).validated()


# --------------------------------------------------------------------------
# identity and sharing
# --------------------------------------------------------------------------

def test_the_same_crop_of_the_same_recording_resolves_to_one_file(tmp_path):
    """What "the four members of this group share the audio" means on disk."""

    samples = np.concatenate([_silence(1.0), _tone(2.0), _silence(1.0)])
    path = _write(tmp_path / "doorbell" / "c" / "clip.wav", samples,
                  sound_class="doorbell")
    budget = SegmentBudget(max_duration_s=6.0)
    first = materialize_segment(
        plan_segment(path, budget=budget, sound_class="doorbell"),
        tmp_path / "out", budget=budget)
    second = materialize_segment(
        plan_segment(path, budget=budget, sound_class="doorbell"),
        tmp_path / "out", budget=budget)
    assert first["segment_id"] == second["segment_id"]
    assert first["prepared_path"] == second["prepared_path"]
    assert second["reused_existing_artifact"] is True
    assert len(list((tmp_path / "out").rglob("clip.wav"))) == 1


def test_a_different_crop_of_the_same_recording_is_a_different_segment(tmp_path):
    samples = np.concatenate([_silence(0.5), _tone(4.0), _silence(0.5)])
    path = _write(tmp_path / "air_conditioning" / "c" / "clip.wav", samples,
                  sound_class="air_conditioning")
    short = SegmentBudget(max_duration_s=1.0)
    long = SegmentBudget(max_duration_s=3.0)
    a = materialize_segment(
        plan_segment(path, budget=short, sound_class="air_conditioning"),
        tmp_path / "out", budget=short)
    b = materialize_segment(
        plan_segment(path, budget=long, sound_class="air_conditioning"),
        tmp_path / "out", budget=long)
    assert a["segment_id"] != b["segment_id"]
    assert a["source_sha256"] == b["source_sha256"]
    assert len(list((tmp_path / "out").rglob("clip.wav"))) == 2


def test_the_segment_id_is_not_labelled_as_prepared_speech(tmp_path):
    samples = np.concatenate([_silence(1.0), _tone(2.0), _silence(1.0)])
    _p, record = _prepared(tmp_path, samples, "doorbell", max_duration_s=6.0)
    assert record["segment_id"].startswith("sound_segment_")
    assert "speech" not in record["segment_id"]


def test_segment_id_changes_when_a_result_changing_setting_changes():
    facts = {
        "operation": "x", "source_crop_start_sample": 0,
        "source_crop_end_sample_exclusive": 100, "filter": {},
        "activity_filter": {}, "detector": {"gate": "peak_relative"},
        "target_rate_hz": 16000,
    }
    processing = {"edge_fade_s": 0.005, "remove_dc": False}
    first = segment_id("a", source_sha256="deadbeef", facts=facts,
                       processing=processing)
    other = segment_id("a", source_sha256="deadbeef", processing=processing,
                       facts={**facts, "source_crop_end_sample_exclusive": 200})
    assert first != other
    assert segment_id("a", source_sha256="deadbeef", facts=facts,
                      processing=processing) == first
    # and the processing block is what the crop coordinates cannot express
    assert segment_id("a", source_sha256="deadbeef", facts=facts,
                      processing={**processing, "edge_fade_s": 0.0}) != first


def test_segment_id_refuses_to_name_a_segment_without_its_processing():
    """The argument whose absence caused the 2026-09-10 cache collision."""

    facts = {"operation": "x", "source_crop_start_sample": 0,
             "source_crop_end_sample_exclusive": 100, "target_rate_hz": 16000}
    with pytest.raises(TypeError):
        segment_id("a", source_sha256="deadbeef", facts=facts)
    with pytest.raises(SoundSegmentError, match="processing settings"):
        segment_id("a", source_sha256="deadbeef", facts=facts, processing={})


# --------------------------------------------------------------------------
# transcripts
# --------------------------------------------------------------------------

def test_a_transcript_survives_only_when_the_whole_utterance_is_kept(tmp_path):
    samples = np.concatenate([_silence(0.3), _tone(2.5, freq=500.0),
                              _silence(0.3)])
    path = _write(tmp_path / "speech_playback" / "c" / "clip.wav", samples,
                  sound_class="speech_playback", transcript="hello there")
    kept = plan_segment(path, budget=SegmentBudget(max_duration_s=6.0),
                        sound_class="speech_playback")
    assert kept["covers_all_source_activity"] is True
    assert kept["transcript"] == "hello there"
    assert kept["transcript_status"] == "inherited_full_source_activity_retained"


def test_a_cropped_utterance_does_not_keep_the_old_transcript(tmp_path):
    samples = np.concatenate([_silence(0.3), _tone(2.0, freq=500.0),
                              _silence(0.3), _tone(2.0, freq=600.0),
                              _silence(0.3)])
    path = _write(tmp_path / "speech_playback" / "c" / "clip.wav", samples,
                  sound_class="speech_playback", transcript="one two three")
    cut = plan_segment(path, budget=SegmentBudget(max_duration_s=2.2),
                       sound_class="speech_playback")
    assert cut["status"] == "selected"
    assert cut["covers_all_source_activity"] is False
    assert cut["transcript"] is None
    assert cut["transcript_status"] == "unknown_after_crop"
    assert cut["source_transcript_not_reused"] == "one two three"


def test_a_source_listening_note_is_not_promoted_to_a_segment_verdict(tmp_path):
    samples = np.concatenate([_silence(0.5), _tone(2.0), _silence(0.5)])
    path = _write(tmp_path / "doorbell" / "c" / "clip.wav", samples,
                  sound_class="doorbell",
                  human_review={"status": "pass", "note": "人工抽听暂判合格"})
    budget = SegmentBudget(max_duration_s=6.0)
    plan = plan_segment(path, budget=budget, sound_class="doorbell")
    assert plan["source_human_review"]["status"] == "pass"
    assert plan["segment_human_review"]["status"] == "not_requested"
    assert plan["segment_human_review"]["inherited_source_status"] == "pass"
    record = materialize_segment(plan, tmp_path / "out", budget=budget)
    assert pool_row(record)["human_review"]["status"] == "not_requested"


# --------------------------------------------------------------------------
# the burst detector is a cross-reference, not the proof
# --------------------------------------------------------------------------

def test_the_event_detector_is_quoted_but_never_used_as_the_activity_proof():
    samples = np.concatenate([_silence(0.5), _tone(0.4), _silence(0.5),
                              _tone(0.4), _silence(0.5)])
    xref = cross_reference_sound_events(samples, RATE, sound_class="dog_bark")
    assert xref["is_activity_proof"] is False
    assert xref["is_qa_event_count"] is False
    assert xref["guard_included"] is True


def test_the_event_detector_span_is_wider_than_the_measured_activity():
    """Its guard covers material that is not sounding, which is why it cannot
    stand in for coverage."""

    samples = np.concatenate([_silence(0.5), _tone(0.4), _silence(0.5)])
    xref = cross_reference_sound_events(samples, RATE, sound_class="dog_bark")
    facts = measure_activity(samples, RATE, sound_class="dog_bark")
    span = xref["spans_samples"][0]
    assert (span[1] - span[0]) / RATE > facts["active_duration_s"]


def test_a_clip_the_event_detector_refuses_is_reported_not_raised():
    xref = cross_reference_sound_events(_silence(3.0), RATE,
                                        sound_class="dog_bark")
    assert xref["status"] == "not_run"
    assert xref["island_count"] is None


# --------------------------------------------------------------------------
# indexing, batch preparation and the shape P07 consumes
# --------------------------------------------------------------------------

def _library(tmp_path) -> Path:
    root = tmp_path / "library"
    _write(root / "doorbell" / "bell_a" / "clip.wav",
           np.concatenate([_silence(1.0), _tone(0.8), _silence(1.0)]),
           sound_class="doorbell")
    _write(root / "air_conditioning" / "ac_a" / "clip.wav",
           _tone(9.0, freq=120.0, amplitude=0.3) + _noise(9.0, amplitude=0.02),
           sound_class="air_conditioning")
    _write(root / "air_conditioning" / "ac_silent" / "clip.wav",
           _silence(9.0), sound_class="air_conditioning")
    return root


def test_the_library_walk_reads_the_class_from_the_clip_sidecar(tmp_path):
    clips = list(iter_library_clips(_library(tmp_path)))
    assert {clip["sound_class"] for clip in clips} == {
        "doorbell", "air_conditioning"}
    assert all(Path(clip["source_path"]).is_file() for clip in clips)
    assert {clip["source_asset_id"] for clip in clips} == {
        "doorbell/bell_a", "air_conditioning/ac_a", "air_conditioning/ac_silent"}


def test_indexing_writes_no_audio(tmp_path):
    root = _library(tmp_path)
    before = sorted(p.stat().st_mtime_ns for p in root.rglob("*.wav"))
    index = plan_segments(
        [{"source_path": clip["source_path"], "sound_class": clip["sound_class"]}
         for clip in iter_library_clips(root)],
        budget=SegmentBudget(max_duration_s=6.0),
    )
    assert index["counts"]["considered"] == 3
    assert index["counts"]["selected"] == 2
    assert index["by_sound_class"]["air_conditioning"]["reasons"]
    assert sorted(p.stat().st_mtime_ns for p in root.rglob("*.wav")) == before
    assert list((tmp_path).glob("**/sound_segment_*")) == []


def test_prepare_segments_runs_index_cut_and_readback_without_a_human(tmp_path):
    root = _library(tmp_path)
    report = prepare_segments(
        [{"source_path": clip["source_path"], "sound_class": clip["sound_class"],
          "source_asset_id": clip["source_asset_id"]}
         for clip in iter_library_clips(root)],
        tmp_path / "out",
        budget=SegmentBudget(max_duration_s=6.0),
    )
    assert report["counts"]["materialized"] == 2
    assert report["counts"]["failures"] == 0
    assert set(report["counts"]["verification_by_status"]) <= {"pass", "qualified"}
    assert len(list((tmp_path / "out").rglob("clip.wav"))) == 2
    assert "No listening test" in report["claim_boundary"]


def test_a_pool_row_carries_the_fields_the_batch_pool_already_uses(tmp_path):
    samples = np.concatenate([_silence(1.0), _tone(2.0), _silence(1.0)])
    _p, record = _prepared(tmp_path, samples, "doorbell", max_duration_s=6.0)
    row = pool_row(record, compatible_asset_ids=["speaker_a"], species_id=None)
    for key in ("sound_asset_id", "path", "sound_class", "event_class",
                "species_id", "source_origin", "sound_identity_id",
                "sound_identity_keys", "compatible_asset_ids", "sample_count",
                "sample_rate_hz", "active_duration_s",
                "source_activity_intervals_samples", "audible_start_sample",
                "audible_end_sample_exclusive", "activity_measurement",
                "activity_calibration", "activity_guard_included",
                "activity_is_qa_event_count", "linear_gain",
                "normalization_applied", "human_review"):
        assert key in row, key
    assert row["sample_count"] == record["prepared_sample_count"]
    assert row["sample_rate_hz"] == record["target_rate_hz"]
    assert row["compatible_asset_ids"] == ["speaker_a"]
    assert row["activity_is_qa_event_count"] is False
    # the intervals a consumer reads must be in the coordinates of `path`
    assert row["source_activity_intervals_samples"][-1][1] <= row["sample_count"]


def test_read_source_pcm_does_not_open_the_original_for_writing(tmp_path):
    path = _write(tmp_path / "doorbell" / "c" / "clip.wav", _tone(1.0))
    path.chmod(0o444)
    try:
        samples, rate = read_source_pcm(path)
    finally:
        path.chmod(0o644)
    assert rate == RATE
    assert samples.size == RATE


def test_a_budget_that_cannot_hold_one_detector_window_is_refused():
    with pytest.raises(SoundSegmentError, match="positive and finite"):
        SegmentBudget(max_duration_s=0.0).validated()
    with pytest.raises(SoundSegmentError, match="min_duration_s"):
        SegmentBudget(max_duration_s=1.0, min_duration_s=2.0).validated()


def test_select_segment_can_be_driven_from_a_measurement_alone():
    """The two halves compose: measure once, then ask several budgets."""

    samples = np.concatenate([_silence(0.5), _tone(4.0, freq=300.0), _silence(0.5)])
    policy = segment_policy_for_class("air_conditioning")
    activity = measure_activity(samples, RATE, policy=policy)
    short = select_segment(activity, SegmentBudget(max_duration_s=1.5),
                           policy=policy)
    long = select_segment(activity, SegmentBudget(max_duration_s=3.5),
                          policy=policy)
    assert short["selected"]["duration_s"] == pytest.approx(1.5, abs=0.01)
    assert long["selected"]["duration_s"] == pytest.approx(3.5, abs=0.01)


# --------------------------------------------------------------------------
# processing settings must reach the identity (2026-09-10 cache defect)
# --------------------------------------------------------------------------

def test_every_budget_field_is_classified():
    """A new budget field must be declared as changing samples, or not.

    This is the guard against the defect returning: the fade existed on the
    budget and was simply not in either list, so nothing bound it.
    """

    declared = set(PCM_DETERMINING_BUDGET_FIELDS) | set(
        BOUND_DETERMINING_BUDGET_FIELDS)
    actual = {field.name for field in dataclasses.fields(SegmentBudget)}
    assert actual == declared, (
        f"unclassified budget field(s): {sorted(actual - declared)}; "
        f"stale entries: {sorted(declared - actual)}"
    )


_PROCESSING_VARIANTS = [
    ("edge_fade_s", {"edge_fade_s": 0.0}, {"edge_fade_s": 0.005}),
    ("remove_dc", {"remove_dc": False}, {"remove_dc": True}),
    ("target_peak_dbfs",
     {"normalize_peak": True, "target_peak_dbfs": -3.0},
     {"normalize_peak": True, "target_peak_dbfs": -12.0}),
    ("normalize_peak", {}, {"normalize_peak": True, "target_peak_dbfs": -6.0}),
    ("target_rate_hz", {"target_rate_hz": 16000}, {"target_rate_hz": 8000}),
]


@pytest.mark.parametrize("name,first,second",
                         _PROCESSING_VARIANTS,
                         ids=[row[0] for row in _PROCESSING_VARIANTS])
def test_changing_a_processing_setting_changes_the_id_and_the_bytes(
    tmp_path, name, first, second
):
    # A room-tone floor and a small DC offset, because digital silence has
    # nothing for a fade to attenuate and no offset for DC removal to take
    # out - on that fixture both settings would change the id and not a
    # single sample, which would prove less than it looks.
    body = np.concatenate([_silence(0.5), _tone(2.0, amplitude=0.4),
                           _silence(0.5)])
    samples = body + _noise(len(body) / RATE, amplitude=2e-3) + 0.02
    path = _write(tmp_path / "doorbell" / "c" / "clip.wav", samples,
                  sound_class="doorbell")

    def cut(extra):
        budget = SegmentBudget(max_duration_s=6.0, **extra)
        plan = plan_segment(path, budget=budget, sound_class="doorbell")
        return materialize_segment(plan, tmp_path / "out", budget=budget)

    a, b = cut(first), cut(second)
    assert a["segment_id"] != b["segment_id"], name
    assert a["prepared_path"] != b["prepared_path"], name
    assert a["prepared_sha256"] != b["prepared_sha256"], name
    assert a["reused_existing_artifact"] is False
    assert b["reused_existing_artifact"] is False


def test_asking_again_for_the_same_processing_reuses_one_verified_file(tmp_path):
    samples = np.concatenate([_silence(0.5), _tone(2.0), _silence(0.5)])
    path = _write(tmp_path / "doorbell" / "c" / "clip.wav", samples,
                  sound_class="doorbell")
    budget = SegmentBudget(max_duration_s=6.0, edge_fade_s=0.005)
    first = materialize_segment(
        plan_segment(path, budget=budget, sound_class="doorbell"),
        tmp_path / "out", budget=budget)
    second = materialize_segment(
        plan_segment(path, budget=budget, sound_class="doorbell"),
        tmp_path / "out", budget=budget)
    assert second["reused_existing_artifact"] is True
    assert second["reuse_verified_against_request"] is True
    assert second["prepared_sha256"] == first["prepared_sha256"]
    assert len(list((tmp_path / "out").rglob("clip.wav"))) == 1


def test_the_recorded_fade_is_the_fade_that_was_asked_for(tmp_path):
    """A0's probe: request a fade after a no-fade cut and read the record."""

    samples = np.concatenate([_silence(0.5), _tone(2.0), _silence(0.5)])
    path = _write(tmp_path / "microwave_hum" / "c" / "clip.wav", samples,
                  sound_class="microwave_hum")
    plain = SegmentBudget(max_duration_s=2.0, edge_fade_s=0.0)
    faded = SegmentBudget(max_duration_s=2.0, edge_fade_s=0.005)
    a = materialize_segment(
        plan_segment(path, budget=plain, sound_class="microwave_hum"),
        tmp_path / "out", budget=plain)
    b = materialize_segment(
        plan_segment(path, budget=faded, sound_class="microwave_hum"),
        tmp_path / "out", budget=faded)
    assert a["edge_fade_s"] == 0.0 and a["edge_fade_samples"] == 0
    assert b["edge_fade_s"] == pytest.approx(0.005)
    assert b["edge_fade_samples"] == 80
    assert b["reused_existing_artifact"] is False
    read_a, _rate = read_source_pcm(a["prepared_path"])
    read_b, _rate = read_source_pcm(b["prepared_path"])
    assert abs(float(read_a[0])) >= abs(float(read_b[0]))


# --------------------------------------------------------------------------
# reuse must correspond to this request, and never repair in place
# --------------------------------------------------------------------------

def _one_segment(tmp_path, **budget_kwargs):
    samples = np.concatenate([_silence(0.5), _tone(2.0), _silence(0.5)])
    path = _write(tmp_path / "doorbell" / "c" / "clip.wav", samples,
                  sound_class="doorbell")
    budget = SegmentBudget(max_duration_s=6.0, **budget_kwargs)
    plan = plan_segment(path, budget=budget, sound_class="doorbell")
    record = materialize_segment(plan, tmp_path / "out", budget=budget)
    return path, budget, plan, record


def test_a_clip_without_its_sidecar_is_reported_and_left_in_place(tmp_path):
    path, budget, plan, record = _one_segment(tmp_path)
    sidecar = Path(record["sidecar_path"])
    sidecar.unlink()
    before = Path(record["prepared_path"]).read_bytes()
    with pytest.raises(SoundSegmentError, match="incomplete segment artifact"):
        materialize_segment(plan, tmp_path / "out", budget=budget,
                            sidecar_wait_s=0.0)
    assert Path(record["prepared_path"]).read_bytes() == before


def test_pcm_that_no_longer_matches_its_sidecar_is_reported_not_reused(tmp_path):
    path, budget, plan, record = _one_segment(tmp_path)
    prepared = Path(record["prepared_path"])
    _write(prepared, _tone(0.5, amplitude=0.1))
    tampered = prepared.read_bytes()
    with pytest.raises(SoundSegmentError, match="does not match the sidecar"):
        materialize_segment(plan, tmp_path / "out", budget=budget)
    assert prepared.read_bytes() == tampered


def test_a_sidecar_recorded_under_other_settings_is_refused(tmp_path):
    path, budget, plan, record = _one_segment(tmp_path)
    sidecar = Path(record["sidecar_path"])
    stored = json.loads(sidecar.read_text())
    stored["budget"]["edge_fade_s"] = 0.25
    sidecar.write_text(json.dumps(stored))
    with pytest.raises(SoundSegmentError, match="different settings"):
        materialize_segment(plan, tmp_path / "out", budget=budget)


def test_reuse_records_what_it_checked(tmp_path):
    path, budget, plan, record = _one_segment(tmp_path)
    again = materialize_segment(plan, tmp_path / "out", budget=budget)
    checked = again["reuse_checked"]
    assert checked["identity_fields"] is True
    assert checked["prepared_sha256_rechecked"] == record["prepared_sha256"]
    assert set(PCM_DETERMINING_BUDGET_FIELDS) <= set(checked["requested_processing"])


def test_two_workers_cutting_one_segment_produce_one_file(tmp_path):
    samples = np.concatenate([_silence(0.5), _tone(2.0), _silence(0.5)])
    path = _write(tmp_path / "doorbell" / "c" / "clip.wav", samples,
                  sound_class="doorbell")
    budget = SegmentBudget(max_duration_s=6.0)
    plan = plan_segment(path, budget=budget, sound_class="doorbell")
    start = threading.Barrier(4)
    records: list[dict] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        start.wait()
        try:
            record = materialize_segment(plan, tmp_path / "out", budget=budget)
        except BaseException as error:  # noqa: BLE001 - recorded, then asserted
            with lock:
                errors.append(error)
        else:
            with lock:
                records.append(record)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(records) == 4
    assert len({row["segment_id"] for row in records}) == 1
    assert len({row["prepared_sha256"] for row in records}) == 1
    assert len(list((tmp_path / "out").rglob("clip.wav"))) == 1
    assert sum(1 for row in records if not row["reused_existing_artifact"]) == 1


# --------------------------------------------------------------------------
# the samples cut must come from the recording the plan was made from
# --------------------------------------------------------------------------

def test_a_same_length_rewrite_of_the_original_is_caught(tmp_path):
    samples = np.concatenate([_silence(0.5), _tone(2.0), _silence(0.5)])
    path = _write(tmp_path / "doorbell" / "c" / "clip.wav", samples,
                  sound_class="doorbell")
    budget = SegmentBudget(max_duration_s=6.0)
    plan = plan_segment(path, budget=budget, sound_class="doorbell")
    # same sample count, different audio - a sample-count check sees nothing
    _write(path, np.concatenate([_silence(0.5), _tone(2.0, freq=880.0),
                                 _silence(0.5)]))
    with pytest.raises(SoundSegmentError, match="source_sha256_differs"):
        materialize_segment(plan, tmp_path / "out", budget=budget)


def test_a_recording_handle_from_a_different_file_is_refused(tmp_path):
    samples = np.concatenate([_silence(0.5), _tone(2.0), _silence(0.5)])
    path = _write(tmp_path / "doorbell" / "c" / "clip.wav", samples,
                  sound_class="doorbell")
    other = _write(tmp_path / "doorbell" / "d" / "clip.wav",
                   np.concatenate([_silence(0.5), _tone(2.0, freq=880.0),
                                   _silence(0.5)]), sound_class="doorbell")
    budget = SegmentBudget(max_duration_s=6.0)
    plan = plan_segment(path, budget=budget, sound_class="doorbell")
    with pytest.raises(SoundSegmentError, match="not the one the plan"):
        materialize_segment(plan, tmp_path / "out", budget=budget,
                            source=SourceRecording.read(other))


def test_a_matching_recording_handle_is_accepted_and_saves_the_reread(tmp_path):
    samples = np.concatenate([_silence(0.5), _tone(2.0), _silence(0.5)])
    path = _write(tmp_path / "doorbell" / "c" / "clip.wav", samples,
                  sound_class="doorbell")
    budget = SegmentBudget(max_duration_s=6.0)
    handle = SourceRecording.read(path)
    plan = plan_segment(path, budget=budget, sound_class="doorbell")
    assert handle.mismatches(plan) == []
    record = materialize_segment(plan, tmp_path / "out", budget=budget,
                                 source=handle)
    assert record["source_sha256"] == handle.sha256


# --------------------------------------------------------------------------
# the headroom guard is not loudness matching
# --------------------------------------------------------------------------

def test_a_quiet_clip_and_a_loud_clip_both_keep_unity_gain(tmp_path):
    gains = []
    for index, amplitude in enumerate((0.05, 0.6)):
        path = _write(tmp_path / "doorbell" / f"c{index}" / "clip.wav",
                      np.concatenate([_silence(0.5), _tone(2.0, amplitude=amplitude),
                                      _silence(0.5)]), sound_class="doorbell")
        budget = SegmentBudget(max_duration_s=6.0)
        record = materialize_segment(
            plan_segment(path, budget=budget, sound_class="doorbell"),
            tmp_path / "out", budget=budget)
        gains.append(record)
    assert [row["applied_gain_db"] for row in gains] == [0.0, 0.0]
    assert [row["linear_gain"] for row in gains] == [1.0, 1.0]
    assert all(row["headroom_fired"] is False for row in gains)
    assert all(row["headroom_is_overflow_guard"] is True for row in gains)
    # the two segments keep their loudness difference
    assert gains[1]["prepared_peak_dbfs"] - gains[0]["prepared_peak_dbfs"] > 15.0


def test_the_recorded_scale_factor_is_not_claimed_to_undo_quantisation(tmp_path):
    _p, record = _prepared(
        tmp_path, np.concatenate([_silence(0.5), _tone(2.0), _silence(0.5)]),
        "doorbell", max_duration_s=6.0)
    assert "quantisation is not undone" in record["gain_recovers_scale_not_quantisation"]


# --------------------------------------------------------------------------
# temporal form is measured; the semantic class is not rewritten
# --------------------------------------------------------------------------

def test_temporal_form_separates_a_hum_a_tick_train_and_one_ring():
    hum = measure_activity(_tone(6.0, freq=120.0), RATE,
                           sound_class="air_conditioning")
    assert hum["structure"]["temporal_form"] == "continuous"

    tick = np.tile(np.concatenate([_tone(0.12), _silence(0.88)]), 6)
    train = measure_activity(tick, RATE, sound_class="clock_tick")
    assert train["structure"]["temporal_form"] == "periodic_pulse_train"
    assert train["structure"]["onset_spacing_is_regular"] is True
    assert train["structure"]["onset_period_coefficient_of_variation"] < 0.1

    ring = np.concatenate([_silence(1.0), _tone(0.5), _silence(4.0)])
    one = measure_activity(ring, RATE, sound_class="doorbell")
    assert one["structure"]["temporal_form"] == "single_burst"


def test_irregular_spacing_is_not_called_a_train():
    bursts = np.concatenate([
        _tone(0.2), _silence(0.3), _tone(0.2), _silence(1.7),
        _tone(0.2), _silence(0.6), _tone(0.2),
    ])
    facts = measure_activity(bursts, RATE, sound_class="dog_bark")
    assert facts["structure"]["onset_spacing_is_regular"] is False
    assert facts["structure"]["temporal_form"] == "sparse_bursts"


def test_a_measured_form_that_disagrees_with_the_class_is_reported_not_applied(
    tmp_path,
):
    """clock_tick stays clock_tick; the disagreement is a fact, not a rename."""

    tick = np.tile(np.concatenate([_tone(0.12), _silence(0.88)]), 6)
    path = _write(tmp_path / "clock_tick" / "c" / "clip.wav", tick,
                  sound_class="clock_tick")
    plan = plan_segment(path, budget=SegmentBudget(max_duration_s=6.0),
                        sound_class="clock_tick")
    assert plan["policy"]["activity_family"] == "device_continuous"
    assert plan["source_temporal_form"] == "periodic_pulse_train"
    assert plan["temporal_form_vs_declared_family"]["agrees"] is False
    assert plan["status"] == "rejected"


def test_pause_structure_describes_the_rhythm_rather_than_only_the_ratio():
    """A dialled number is 40-49% sounding because it has gaps between digits."""

    digits = np.tile(np.concatenate([_tone(0.12), _silence(0.18)]), 8)
    facts = measure_activity(digits, RATE, sound_class="telephone_dialing_dtmf")
    structure = facts["structure"]
    assert 0.35 <= facts["activity_coverage"] <= 0.55
    assert structure["temporal_form"] == "periodic_pulse_train"
    assert structure["pause_duration_max_s"] < 0.25
    assert structure["sounding_duration_min_s"] > 0.10
    assert structure["coverage_alone_is_not_a_verdict"] is True


def test_the_same_coverage_with_a_long_tail_gap_looks_different():
    trailing = np.concatenate([_tone(1.35), _silence(1.65)])
    facts = measure_activity(trailing, RATE, sound_class="telephone_dialing_dtmf")
    assert 0.35 <= facts["activity_coverage"] <= 0.55
    assert facts["structure"]["temporal_form"] == "single_burst"
    assert facts["trailing_inactive_s"] > 1.0


def test_pause_structure_is_computable_for_any_span():
    intervals = [[0, 1600], [4800, 6400], [9600, 11200]]
    structure = pause_structure(intervals, 0, 12800, RATE)
    assert structure["sounding_interval_count"] == 3
    assert structure["pause_count"] == 2
    assert structure["onset_spacing_is_regular"] is True


# --------------------------------------------------------------------------
# per-class policy without renaming a class
# --------------------------------------------------------------------------

def test_one_class_can_be_given_a_different_temporal_policy(tmp_path):
    root = tmp_path / "library"
    tick = np.tile(np.concatenate([_tone(0.12), _silence(0.88)]), 6)
    _write(root / "clock_tick" / "t" / "clip.wav", tick, sound_class="clock_tick")
    _write(root / "blender" / "b" / "clip.wav",
           _tone(6.0, freq=200.0) + _noise(6.0, amplitude=0.02),
           sound_class="blender")
    sources = [{"source_path": clip["source_path"],
                "sound_class": clip["sound_class"]}
               for clip in iter_library_clips(root)]

    plain = plan_segments(sources, budget=SegmentBudget(max_duration_s=6.0))
    assert plain["counts"]["selected"] == 1

    tuned = plan_segments(
        sources, budget=SegmentBudget(max_duration_s=6.0),
        class_policy_overrides={"clock_tick": {"activity_family": "short_prompt"}},
    )
    assert tuned["counts"]["selected"] == 2
    families = {row["sound_class"]: row["policy"]["activity_family"]
                for row in tuned["plans"]}
    # blender keeps its own policy; only clock_tick was re-aimed
    assert families["blender"] == "device_continuous"
    assert families["clock_tick"] == "short_prompt"
    # and clock_tick is still clock_tick
    assert {row["sound_class"] for row in tuned["plans"]} == {"clock_tick", "blender"}


def test_the_index_reports_the_measured_form_per_class(tmp_path):
    root = tmp_path / "library"
    _write(root / "blender" / "b" / "clip.wav", _tone(6.0, freq=200.0),
           sound_class="blender")
    index = plan_segments(
        [{"source_path": clip["source_path"], "sound_class": clip["sound_class"]}
         for clip in iter_library_clips(root)],
        budget=SegmentBudget(max_duration_s=6.0),
    )
    assert index["measured_temporal_form_by_class"]["blender"] == {"continuous": 1}


def test_an_interrupted_publish_leaves_no_orphan_clip(tmp_path, monkeypatch):
    """Killed between the clip and its sidecar: clean up our own half-write."""

    samples = np.concatenate([_silence(0.5), _tone(2.0), _silence(0.5)])
    path = _write(tmp_path / "doorbell" / "c" / "clip.wav", samples,
                  sound_class="doorbell")
    budget = SegmentBudget(max_duration_s=6.0)
    plan = plan_segment(path, budget=budget, sound_class="doorbell")

    real_open = Path.open

    def explode(self, mode="r", *args, **kwargs):
        if mode == "x":
            raise KeyboardInterrupt("worker stopped mid-publish")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", explode)
    with pytest.raises(KeyboardInterrupt):
        materialize_segment(plan, tmp_path / "out", budget=budget)
    monkeypatch.undo()
    assert list((tmp_path / "out").rglob("clip.wav")) == []

    # and the redo path works, with nothing left over to trip on
    record = materialize_segment(plan, tmp_path / "out", budget=budget)
    assert record["reused_existing_artifact"] is False
    assert verify_segment_artifact(record)["failed_checks"] == []


def test_a_fresh_output_root_is_the_redo_path_for_a_stuck_artifact(tmp_path):
    path, budget, plan, record = _one_segment(tmp_path)
    Path(record["sidecar_path"]).unlink()
    with pytest.raises(SoundSegmentError):
        materialize_segment(plan, tmp_path / "out", budget=budget,
                            sidecar_wait_s=0.0)
    redone = materialize_segment(plan, tmp_path / "fresh", budget=budget)
    assert redone["prepared_sha256"] == record["prepared_sha256"]
    assert verify_segment_artifact(redone)["failed_checks"] == []
    # the stuck artifact is still sitting there for someone to look at
    assert Path(record["prepared_path"]).is_file()


# --------------------------------------------------------------------------
# temporal policy: separated from the class, and opt-in
# --------------------------------------------------------------------------

def test_the_declared_thresholds_are_the_default(tmp_path):
    tick = np.tile(np.concatenate([_tone(0.12), _silence(0.88)]), 6)
    path = _write(tmp_path / "clock_tick" / "c" / "clip.wav", tick,
                  sound_class="clock_tick")
    plan = plan_segment(path, budget=SegmentBudget(max_duration_s=6.0),
                        sound_class="clock_tick")
    assert plan["status"] == "rejected"
    assert plan["policy"]["temporal_form_applied"] is None


def test_the_measured_thresholds_are_asked_for_and_recorded(tmp_path):
    tick = np.tile(np.concatenate([_tone(0.12), _silence(0.88)]), 6)
    path = _write(tmp_path / "clock_tick" / "c" / "clip.wav", tick,
                  sound_class="clock_tick")
    plan = plan_segment(
        path, sound_class="clock_tick",
        budget=SegmentBudget(max_duration_s=6.0, temporal_policy="measured"),
    )
    assert plan["status"] == "selected"
    policy = plan["policy"]
    assert policy["temporal_form_applied"] == "periodic_pulse_train"
    assert policy["declared_thresholds"]["min_activity_coverage"] == 0.85
    assert policy["min_activity_coverage"] == 0.25
    # the semantic class and its gate are untouched
    assert policy["activity_family"] == "device_continuous"
    assert policy["gate"] == "peak_relative"
    assert plan["declared_policy"]["min_activity_coverage"] == 0.85
    assert plan["sound_class"] == "clock_tick"


def test_a_continuous_recording_keeps_continuous_thresholds_under_measured_mode(
    tmp_path,
):
    """The mode is not a blanket relaxation: a hum still has to be a hum."""

    intermittent = np.tile(
        np.concatenate([_tone(0.4, freq=120.0), _silence(0.6)]), 6)
    path = _write(tmp_path / "air_conditioning" / "c" / "clip.wav", intermittent,
                  sound_class="air_conditioning")
    plan = plan_segment(
        path, sound_class="air_conditioning",
        budget=SegmentBudget(max_duration_s=6.0, temporal_policy="measured"),
    )
    assert plan["policy"]["temporal_form_applied"] == "periodic_pulse_train"
    steady = _write(tmp_path / "air_conditioning" / "d" / "clip.wav",
                    _tone(8.0, freq=120.0), sound_class="air_conditioning")
    hum = plan_segment(
        steady, sound_class="air_conditioning",
        budget=SegmentBudget(max_duration_s=6.0, temporal_policy="measured"),
    )
    assert hum["policy"]["temporal_form_applied"] == "continuous"
    assert hum["policy"]["min_activity_coverage"] == 0.85


def test_an_explicit_override_still_wins_over_the_measured_form(tmp_path):
    tick = np.tile(np.concatenate([_tone(0.12), _silence(0.88)]), 6)
    path = _write(tmp_path / "clock_tick" / "c" / "clip.wav", tick,
                  sound_class="clock_tick")
    plan = plan_segment(
        path, sound_class="clock_tick",
        budget=SegmentBudget(max_duration_s=6.0, temporal_policy="measured"),
        policy_overrides={"min_activity_coverage": 0.99},
    )
    assert plan["policy"]["min_activity_coverage"] == 0.99
    assert plan["status"] == "rejected"


def test_an_unknown_temporal_policy_is_refused():
    with pytest.raises(SoundSegmentError, match="temporal_policy"):
        SegmentBudget(max_duration_s=5.0, temporal_policy="whatever").validated()


def test_a_pool_row_carries_the_structure_behind_its_coverage(tmp_path):
    """A consumer must be able to see why 45% coverage is or is not fine."""

    digits = np.tile(np.concatenate([_tone(0.12), _silence(0.18)]), 8)
    path = _write(tmp_path / "telephone_dialing_dtmf" / "c" / "clip.wav", digits,
                  sound_class="telephone_dialing_dtmf")
    budget = SegmentBudget(max_duration_s=6.0)
    record = materialize_segment(
        plan_segment(path, budget=budget, sound_class="telephone_dialing_dtmf"),
        tmp_path / "out", budget=budget)
    row = pool_row(record)
    assert 0.35 <= row["activity_coverage"] <= 0.60
    assert row["temporal_form"] == "periodic_pulse_train"
    assert row["pause_structure"]["pause_duration_max_s"] < 0.30
    assert row["coverage_alone_is_not_a_verdict"] is True
    applied = row["applied_thresholds"]
    assert applied["min_activity_coverage"] == 0.25
    assert applied["temporal_policy"] == "declared"
    assert applied["temporal_form_applied"] is None


def test_legacy_pool_row_recovers_measured_pauses_without_changing_record(tmp_path):
    samples = np.concatenate([_silence(0.6), _tone(0.8), _silence(0.13),
                              _tone(0.9), _silence(0.6)])
    _plan, record = _prepared(tmp_path, samples, "doorbell", max_duration_s=6.0)
    expected = record["pause_structure"]
    record.pop("pause_structure")
    record.pop("segment_temporal_form", None)
    original = json.dumps(record, sort_keys=True)
    row = pool_row(record)
    assert row["pause_structure"]["pause_durations_s"] == expected["pause_durations_s"]
    assert row["pause_structure"]["sounding_interval_count"] == expected["sounding_interval_count"]
    assert row["temporal_form"] == expected["temporal_form"]
    assert row["path"] == record["prepared_path"]
    assert row["sample_count"] == record["prepared_sample_count"]
    assert json.dumps(record, sort_keys=True) == original


def test_pool_row_keeps_recording_analysis_and_segment_coordinates_distinct(tmp_path):
    from avengine.dataset.source_capabilities import verify_segment_selection
    source_rate = 44100
    samples = np.concatenate([_silence(0.8, rate=source_rate),
                              _tone(1.0, rate=source_rate),
                              _silence(0.8, rate=source_rate)])
    path = _write(tmp_path / "source.wav", samples, rate=source_rate, sound_class="doorbell")
    budget = SegmentBudget(max_duration_s=6.0, target_rate_hz=16000)
    plan = plan_segment(path, budget=budget, sound_class="doorbell")
    record = materialize_segment(plan, tmp_path / "prepared", budget=budget)
    row = pool_row(record)
    assert row["source_rate_hz"] == 44100
    assert row["analysis_rate_hz"] == row["sample_rate_hz"] == 16000
    coordinates = row["activity_interval_coordinates"]
    assert coordinates["source_activity_intervals_samples"].endswith("target_rate")
    assert coordinates["origin_activity_intervals_samples"].endswith("analysis_rate")
    assert row["source_activity_intervals_samples"][0][0] < row["origin_activity_intervals_samples"][0][0]
    assert row["source_activity_intervals_samples"][-1][1] <= row["sample_count"]
    assert row["origin_activity_intervals_samples"][-1][1] <= row["analysis_sample_count"]
    assert row["source_crop_end_sample_exclusive"] <= row["source_sample_count"]
    verdict = verify_segment_selection(row, read_source_header=False)
    assert verdict["verified"] is True, verdict
    assert verdict["unverified_bounds"] == []

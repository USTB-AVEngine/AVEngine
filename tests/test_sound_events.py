"""Pulse splitting: one bark is one event; two barks stay two events."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "assets"))

from avengine.assets.sound_events import (  # noqa: E402
    split_family,
    split_params_for_class,
    SoundEventError,
    event_policy_for_class,
    extract_sound_events,
    slice_event,
)
from split_sound_library_events import split_library  # noqa: E402


def _tone(freq: float, rate: int, seconds: float, amplitude: float = 0.4):
    time = np.arange(int(round(rate * seconds))) / rate
    return amplitude * np.sin(2 * np.pi * freq * time)


def _write(path: Path, samples: np.ndarray, rate: int) -> None:
    ints = np.clip(np.round(samples * 32767), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(ints.tobytes())


def test_dog_bark_class_is_pulse() -> None:
    assert event_policy_for_class("dog_bark") == "pulse"
    assert event_policy_for_class("cat_meow") == "pulse"
    assert event_policy_for_class("speech_playback") == "continuous"
    assert event_policy_for_class("air_conditioning") == "continuous"


def test_nothing_whose_occurrence_is_an_utterance_may_be_cut() -> None:
    """owner 2026-09-03:"人声可千万不能和狗一样切掉中间的声音"。

    同样的道理适用于铃声(一段旋律)、拨号音(一串数字)、持续鸣响的报警:
    切掉中间就不是那一次出现了。旧的词元启发式匹配 ring/bell/beep/alarm,
    正是它把这些判成了脉冲。
    """

    for name in ("speech_playback", "music_playback", "ringtone",
                 "telephone_dialing_dtmf", "phone_ring", "alarm_beep",
                 "alarm_clock", "smoke_alarm", "busy_signal", "buzzer",
                 "microwave_beep", "clock_tick", "cellphone_vibration_alert"):
        assert event_policy_for_class(name) == "continuous", name
        assert split_family(name) == "sustained", name


def test_an_unknown_class_is_never_cut() -> None:
    # 没被刻画过的类别一律只去头尾:切错中间不可逆,不切最坏只是事件偏长。
    assert event_policy_for_class("some_new_sound_v3") == "continuous"
    assert split_family("some_new_sound_v3") == "sustained"


def test_a_sustained_class_has_no_splitting_parameters() -> None:
    with pytest.raises(SoundEventError, match="never split"):
        split_params_for_class("ringtone")


def test_a_multi_tone_ring_stays_one_occurrence() -> None:
    """门铃按一次是两个音,一次出现;60 毫秒的合并间隔会把它拆成叮和咚。"""

    rate = 16000
    ding = _tone(880, rate, 0.35)
    dong = _tone(660, rate, 0.45)
    quiet = np.zeros(int(rate * 0.4))
    clip = np.concatenate([np.zeros(int(rate * 0.2)), ding, quiet, dong,
                           np.zeros(int(rate * 0.2))])
    grouped = extract_sound_events(clip, rate, event_class="doorbell")
    assert len(grouped) == 1
    atomic = extract_sound_events(clip, rate, event_class="dog_bark")
    assert len(atomic) == 2, "同一段波形按原子脉冲切应当分成两个,证明差别来自类族"


def test_a_long_ring_is_not_truncated_at_the_old_cap() -> None:
    """2026-09-03 实测:doorbell 三条全部落在 1.53-1.56 s,即旧的 1.5 s 上限
    加保护带,真实的单次门铃被切断了。"""

    rate = 16000
    clip = np.concatenate([np.zeros(int(rate * 0.2)),
                           _tone(700, rate, 2.5),
                           np.zeros(int(rate * 0.2))])
    events = extract_sound_events(clip, rate, event_class="doorbell")
    assert len(events) == 1
    assert events[0].duration_s(rate) >= 2.5, events[0].duration_s(rate)
    assert not events[0].truncated


def test_two_separated_barks_become_two_events() -> None:
    rate = 16000
    bark = _tone(800, rate, 0.25)
    gap = np.zeros(int(rate * 0.8))
    clip = np.concatenate([np.zeros(rate), bark, gap, bark, np.zeros(rate)])
    events = extract_sound_events(clip, rate, event_class="dog_bark")
    assert len(events) == 2
    assert all(e.purpose == "pulse" for e in events)
    for event in events:
        duration = event.duration_s(rate)
        assert 0.25 <= duration <= 0.40
        piece = slice_event(clip, event)
        assert float(np.abs(piece).max()) > 0.2


def test_close_dips_in_one_bark_stay_one_event() -> None:
    rate = 16000
    first = _tone(700, rate, 0.12)
    dip = np.zeros(int(rate * 0.03))
    second = _tone(700, rate, 0.12)
    clip = np.concatenate([np.zeros(int(rate * 0.2)), first, dip, second])
    events = extract_sound_events(clip, rate, event_class="dog_bark")
    assert len(events) == 1
    assert 0.20 <= events[0].duration_s(rate) <= 0.40


def test_continuous_class_keeps_the_whole_span() -> None:
    rate = 16000
    hum = _tone(120, rate, 2.0, amplitude=0.2)
    clip = np.concatenate([np.zeros(int(rate * 0.5)), hum, np.zeros(int(rate * 0.5))])
    events = extract_sound_events(clip, rate, event_class="air_conditioning")
    assert len(events) == 1
    assert events[0].purpose == "continuous"
    assert 2.0 <= events[0].duration_s(rate) <= 2.2


def test_silence_is_rejected() -> None:
    rate = 16000
    with pytest.raises(SoundEventError):
        extract_sound_events(np.zeros(rate), rate, event_class="dog_bark")


def test_guard_keeps_the_onset() -> None:
    rate = 16000
    bark = _tone(900, rate, 0.3)
    clip = np.concatenate([np.zeros(rate), bark])
    events = extract_sound_events(clip, rate, event_class="dog_bark")
    assert len(events) == 1
    # first sample of the event is still in the padded silence, not inside the tone
    assert events[0].start_sample < rate
    assert rate - events[0].start_sample <= int(0.04 * rate) + 1


def test_library_splitter_writes_events_and_refuses_to_overwrite(
    tmp_path: Path,
) -> None:
    library = tmp_path / "prepared"
    bark = _tone(800, 16000, 0.25)
    gap = np.zeros(int(16000 * 0.8))
    clip = np.concatenate([bark, gap, bark])
    _write(library / "dog_bark" / "one.wav", clip, 16000)
    _write(
        library / "air_conditioning" / "hum.wav",
        _tone(100, 16000, 1.5, amplitude=0.2),
        16000,
    )
    out = tmp_path / "events"
    manifest = split_library(library, out)
    dog = [row for row in manifest["clips"] if row.get("event_class") == "dog_bark"]
    hum = [
        row
        for row in manifest["clips"]
        if row.get("event_class") == "air_conditioning"
    ]
    assert len(dog) == 2
    assert len(hum) == 1
    assert (out / "event_manifest.json").is_file()
    assert json.loads((out / "event_manifest.json").read_text())["schema"].endswith(
        "sound_event_library_v1"
    )
    with pytest.raises(FileExistsError):
        split_library(library, out)
    assert all(row["applied_gain_db"] == 0.0 for row in dog + hum)
    assert all(row["truncated"] is False for row in dog)


def test_splitter_does_not_peak_normalize_unless_asked(tmp_path: Path) -> None:
    library = tmp_path / "prepared"
    bark = _tone(800, 16000, 0.25, amplitude=0.1)
    _write(library / "dog_bark" / "quiet.wav", bark, 16000)
    plain = split_library(library, tmp_path / "plain")
    row = next(item for item in plain["clips"] if item["status"] == "event")
    assert row["applied_gain_db"] == 0.0
    normalized = split_library(
        library, tmp_path / "norm", peak_normalize=True)
    gained = next(item for item in normalized["clips"] if item["status"] == "event")
    assert gained["applied_gain_db"] != 0.0


def test_pulse_longer_than_max_is_truncated() -> None:
    rate = 16000
    bark = _tone(800, rate, 2.5)
    clip = np.concatenate([
        np.zeros(int(rate * 1.5)), bark, np.zeros(int(rate * 1.5))])
    events = extract_sound_events(clip, rate, event_class="dog_bark")
    assert len(events) == 1
    assert events[0].purpose == "pulse"
    assert events[0].truncated is True
    cap = split_params_for_class("dog_bark")["max_event_s"]
    assert events[0].duration_s(rate) <= cap + 0.08
    assert events[0].untruncated_end_sample_exclusive is not None
    assert events[0].untruncated_end_sample_exclusive > events[0].end_sample_exclusive


def test_nearly_full_loud_file_falls_back_to_continuous_purpose() -> None:
    rate = 16000
    clip = _tone(800, rate, 3.0, amplitude=0.4)
    events = extract_sound_events(clip, rate, event_class="dog_bark")
    assert len(events) == 1
    assert events[0].purpose == "continuous"

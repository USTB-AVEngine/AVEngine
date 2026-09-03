"""Semantic three-level layout for the sound-event splitter."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "assets"))

from avengine.assets.sound_events import SoundEventError  # noqa: E402
from split_sound_library_events import (  # noqa: E402
    CLASS_CATEGORY,
    category_for_class,
    relative_event_wav,
    sound_asset_id_for,
    split_library,
)


PREPARED_CLASSES = {
    "air_conditioning", "alarm_beep", "alarm_bell", "alarm_clock",
    "any_audioset_class_playback", "bathtub_filling_washing", "blender",
    "busy_signal", "buzzer", "cat_meow", "cellphone_vibration_alert",
    "chime", "clock_tick", "crackle", "ding_dong", "dog_bark", "doorbell",
    "doorbell_chime", "drip", "fire", "fire_alarm", "gurgling",
    "microwave_beep", "microwave_hum", "music_playback", "phone_ring",
    "printer", "ringtone", "sink_filling_washing", "smoke_alarm",
    "speech_playback", "telephone", "telephone_bell_ringing",
    "telephone_dialing_dtmf", "toilet_flush", "water_tap_faucet",
}

SHA8 = re.compile(r"^[0-9a-f]{8}$")
SOURCEISH = re.compile(
    r"fsd50k_|clip_e\d|occurrence|atomic|grouped|sustained|purpose"
)


def _tone(freq: float, rate: int, seconds: float, amplitude: float = 0.4):
    time = np.arange(int(round(rate * seconds))) / rate
    return amplitude * np.sin(2 * np.pi * freq * time)


def _write(path: Path, samples: np.ndarray, rate: int) -> None:
    import wave
    ints = np.clip(np.round(samples * 32767), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(ints.tobytes())


def test_category_table_is_explicit_and_covers_the_prepared_library() -> None:
    assert set(CLASS_CATEGORY) == PREPARED_CLASSES
    assert len(CLASS_CATEGORY) == 36
    assert CLASS_CATEGORY["dog_bark"] == "animal"
    assert CLASS_CATEGORY["speech_playback"] == "speech"
    assert CLASS_CATEGORY["smoke_alarm"] == "alert"
    assert CLASS_CATEGORY["air_conditioning"] == "appliance"
    assert CLASS_CATEGORY["drip"] == "water"
    assert CLASS_CATEGORY["fire"] == "ambience"
    assert CLASS_CATEGORY["music_playback"] == "playback"
    assert CLASS_CATEGORY["any_audioset_class_playback"] == "playback"
    assert CLASS_CATEGORY["clock_tick"] == "ambience"
    assert CLASS_CATEGORY["alarm_clock"] == "alert"
    assert CLASS_CATEGORY["speech_playback"] == "speech"
    assert "synthetic" not in set(CLASS_CATEGORY.values())
    assert set(CLASS_CATEGORY.values()) == {
        "alert", "ambience", "animal", "appliance", "playback", "speech", "water",
    }
    with pytest.raises(SoundEventError, match="closed table"):
        category_for_class("ring_bell_alarm")


def test_paths_are_category_class_sha8_not_source_or_family(
    tmp_path: Path,
) -> None:
    library = tmp_path / "prepared"
    bark_a = _tone(800, 16000, 0.25)
    bark_b = _tone(1200, 16000, 0.30)
    gap = np.zeros(int(16000 * 0.8))
    _write(library / "dog_bark" / "fsd50k_137803" / "clip.wav",
           np.concatenate([bark_a, gap, bark_b]), 16000)
    out = tmp_path / "events"
    manifest = split_library(library, out)
    rows = [row for row in manifest["clips"] if row["status"] == "event"]
    assert len(rows) == 2
    for row in rows:
        sha8 = row["variant"]
        assert SHA8.fullmatch(sha8)
        expected = relative_event_wav("animal", "dog_bark", sha8)
        assert row["prepared"] == expected
        assert row["sound_asset_id"] == sound_asset_id_for("dog_bark", sha8)
        assert row["category"] == "animal"
        assert row["split_family"] == "atomic"
        assert "fsd50k_137803" not in row["prepared"]
        assert "clip_e" not in row["prepared"]
        assert SOURCEISH.search(row["prepared"]) is None
        assert (out / row["prepared"]).is_file()
        actual = hashlib.sha256((out / row["prepared"]).read_bytes()).hexdigest()
        assert actual == row["prepared_sha256"]
        assert actual[:8] == sha8
        assert row["occurrence_index"] in (0, 1)
    index = json.loads((out / "index.json").read_text())
    assert index["layout"] == "<category>/<type>/<variant>"
    assert {item["path"] for item in index["assets"]} == {
        f"animal/dog_bark/{row['variant']}" for row in rows
    }


def test_sha8_collision_in_the_same_class_is_refused(tmp_path: Path) -> None:
    library = tmp_path / "prepared"
    bark = _tone(800, 16000, 0.25)
    _write(library / "dog_bark" / "a" / "clip.wav", bark, 16000)
    _write(library / "dog_bark" / "b" / "clip.wav", bark, 16000)
    with pytest.raises(FileExistsError, match="sha8 collision"):
        split_library(library, tmp_path / "events")


def test_unknown_class_fails_closed(tmp_path: Path) -> None:
    library = tmp_path / "prepared"
    _write(library / "not_a_class" / "clip.wav", _tone(400, 16000, 0.2), 16000)
    with pytest.raises(SoundEventError, match="closed table"):
        split_library(library, tmp_path / "events")



def test_reclassification_changes_the_path_not_the_asset_id(tmp_path: Path) -> None:
    """Category is not part of the id, so a remap must not rename the clip."""

    library = tmp_path / "prepared"
    tick = _tone(1000, 16000, 1.2, amplitude=0.2)
    music = _tone(440, 16000, 1.0, amplitude=0.2)
    _write(library / "clock_tick" / "one.wav", tick, 16000)
    _write(library / "music_playback" / "one.wav", music, 16000)
    out = tmp_path / "events"
    manifest = split_library(library, out)
    rows = {row["event_class"]: row for row in manifest["clips"] if row["status"] == "event"}
    tick_row = rows["clock_tick"]
    music_row = rows["music_playback"]
    assert tick_row["category"] == "ambience"
    assert music_row["category"] == "playback"
    assert tick_row["prepared"].startswith("ambience/clock_tick/")
    assert music_row["prepared"].startswith("playback/music_playback/")
    assert tick_row["sound_asset_id"] == sound_asset_id_for(
        "clock_tick", tick_row["variant"])
    assert music_row["sound_asset_id"] == sound_asset_id_for(
        "music_playback", music_row["variant"])
    assert "appliance" not in tick_row["prepared"]
    assert "synthetic" not in music_row["prepared"]
    assert (out / tick_row["prepared"]).is_file()
    assert (out / music_row["prepared"]).is_file()


def test_splitter_refuses_to_write_into_the_prepared_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from split_sound_library_events import main

    library = tmp_path / "prepared"
    _write(library / "dog_bark" / "clip.wav", _tone(800, 16000, 0.25), 16000)
    (library / "prepared_manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="prepared library"):
        main(["--library-root", str(library), "--output-root", str(library)])

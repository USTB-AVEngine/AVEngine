"""event_manifest.json -> avengine_sound_event_pool_v1 catalog."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "assets"))

from avengine.assets.sound_pool import SoundEventPool  # noqa: E402
from build_sound_event_pool import (  # noqa: E402
    PoolBuildError,
    build_pool_catalog,
    sound_asset_id_for_row,
)


def _write_wav(path: Path, samples, rate=16000) -> None:
    ints = np.clip(np.round(samples * 32767), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(ints.tobytes())


def _manifest(tmp_path: Path, rows, output_root=None) -> Path:
    output_root = output_root or tmp_path / "events"
    payload = {
        "schema": "avengine_sound_event_library_v1",
        "output_root": str(output_root),
        "clips": rows,
    }
    path = tmp_path / "event_manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_sound_asset_id_comes_from_the_manifest_row():
    assert sound_asset_id_for_row(
        {"sound_asset_id": "sound_dog_bark_a67a7389_v1"}
    ) == "sound_dog_bark_a67a7389_v1"
    with pytest.raises(PoolBuildError, match="missing sound_asset_id"):
        sound_asset_id_for_row({"prepared": "animal/dog_bark/a67a7389/event.wav"})


def test_manifest_converts_to_a_loadable_pool(tmp_path: Path):
    output_root = tmp_path / "events"
    wav = output_root / "animal" / "dog_bark" / "deadbeef" / "event.wav"
    _write_wav(wav, np.ones(3200) * 0.2)
    manifest = _manifest(tmp_path, [{
        "status": "event",
        "purpose": "pulse",
        "event_class": "dog_bark",
        "sound_asset_id": "sound_dog_bark_deadbeef_v1",
        "prepared": "animal/dog_bark/deadbeef/event.wav",
    }], output_root=output_root)
    catalog_path = tmp_path / "pool.json"
    catalog = build_pool_catalog(manifest, catalog_path)
    assert catalog["clips"][0]["sound_asset_id"] == "sound_dog_bark_deadbeef_v1"
    assert catalog["clips"][0]["duration_samples"] == 3200
    pool = SoundEventPool.from_catalog(catalog_path)
    clip = pool.clips_for("dog_bark")[0]
    assert clip.duration_samples == 3200
    assert clip.source_end_sample_exclusive - clip.source_start_sample == 3200


def test_a_fallback_span_is_excluded_and_recorded_not_fatal(tmp_path: Path):
    """整体拒绝是附带伤害：2026-09-03 实测两条 cat_meow 降级跨度挡住了
    一份干净的 166 条 dog_bark catalog。那道闸的本意是"别把降级跨度放进
    池子"，排除就达到了目的，所以改成排除并记录。"""

    output_root = tmp_path / "events"
    _write_wav(output_root / "animal" / "dog_bark" / "aaa11111" / "event.wav",
               np.ones(4800) * 0.2)
    _write_wav(output_root / "animal" / "cat_meow" / "bbb22222" / "event.wav",
               np.ones(16000) * 0.2)
    _write_wav(output_root / "animal" / "cat_meow" / "ccc33333" / "event.wav",
               np.ones(9600) * 0.2)
    manifest = _manifest(tmp_path, [
        {"status": "event", "purpose": "pulse", "event_class": "dog_bark",
         "sound_asset_id": "sound_dog_bark_aaa11111_v1",
         "prepared": "animal/dog_bark/aaa11111/event.wav"},
        {"status": "event", "purpose": "continuous", "event_class": "cat_meow",
         "sound_asset_id": "sound_cat_meow_bbb22222_v1",
         "prepared": "animal/cat_meow/bbb22222/event.wav"},
        {"status": "event", "purpose": "pulse", "event_class": "cat_meow",
         "sound_asset_id": "sound_cat_meow_ccc33333_v1",
         "prepared": "animal/cat_meow/ccc33333/event.wav"},
    ], output_root=output_root)
    catalog = build_pool_catalog(manifest, tmp_path / "pool.json")
    assert catalog["clips_by_class"] == {"dog_bark": 1, "cat_meow": 1}
    assert [row["prepared"] for row in catalog["excluded"]] == [
        "animal/cat_meow/bbb22222/event.wav"]
    assert catalog["excluded"][0]["reason"] == (
        "pulse_class_hysteresis_fallback_span")
    assert all(c["prepared"] != "animal/cat_meow/bbb22222/event.wav"
               for c in catalog["clips"])


def test_a_class_that_loses_every_clip_still_fails(tmp_path: Path):
    output_root = tmp_path / "events"
    _write_wav(output_root / "animal" / "dog_bark" / "aaa11111" / "event.wav",
               np.ones(4800) * 0.2)
    _write_wav(output_root / "animal" / "cat_meow" / "bbb22222" / "event.wav",
               np.ones(16000) * 0.2)
    manifest = _manifest(tmp_path, [
        {"status": "event", "purpose": "pulse", "event_class": "dog_bark",
         "sound_asset_id": "sound_dog_bark_aaa11111_v1",
         "prepared": "animal/dog_bark/aaa11111/event.wav"},
        {"status": "event", "purpose": "continuous", "event_class": "cat_meow",
         "sound_asset_id": "sound_cat_meow_bbb22222_v1",
         "prepared": "animal/cat_meow/bbb22222/event.wav"},
    ], output_root=output_root)
    with pytest.raises(PoolBuildError, match="lost every clip"):
        build_pool_catalog(manifest, tmp_path / "pool.json")

def test_pool_carries_explicit_speech_metadata(tmp_path: Path) -> None:
    output_root = tmp_path / "events"
    wav = output_root / "speech" / "speech_playback" / "abcdef12" / "event.wav"
    _write_wav(wav, np.ones(16000) * 0.2)
    manifest = _manifest(
        tmp_path,
        [
            {
                "status": "event",
                "purpose": "continuous",
                "event_class": "speech_playback",
                "sound_asset_id": "sound_speech_playback_abcdef12_v1",
                "prepared": "speech/speech_playback/abcdef12/event.wav",
                "speaker_id": "p225",
                "utterance_id": "001",
                "transcript": "Please call Stella.",
                "split": "eval",
            }
        ],
        output_root=output_root,
    )
    catalog = build_pool_catalog(manifest, tmp_path / "pool.json")
    row = catalog["clips"][0]
    assert row["speaker_id"] == "p225"
    assert row["utterance_id"] == "001"
    assert row["transcript"] == "Please call Stella."
    assert row["split"] == "eval"

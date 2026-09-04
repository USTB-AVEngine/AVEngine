"""Sound event pool: class catalogs, no hardcoded pair-kind map."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avengine.assets.sound_pool import (  # noqa: E402
    SoundPoolError,
    clip_source_from_params,
    event_class_for_pair_kind,
    SoundEventPool,
)

POOL_PARAMS = {
    "SOUND_SOURCE_MODE": "event_pool",
    "SOUND_EVENT_CLASS_BY_PAIR_KIND": {"dog": "dog_bark"},
    "SAMPLE_RATE_HZ": 16000,
}


def _catalog(path: Path, clips):
    path.write_text(json.dumps({
        "schema": "avengine_sound_event_pool_v1",
        "clips": clips,
    }))


def test_draw_samples_from_the_named_class(tmp_path: Path):
    _catalog(tmp_path / "pool.json", [
        {"sound_asset_id": "a", "event_class": "dog_bark",
         "sample_rate_hz": 16000, "duration_samples": 3200,
         "source_start_sample": 0, "source_end_sample_exclusive": 3200},
        {"sound_asset_id": "b", "event_class": "dog_bark",
         "sample_rate_hz": 16000, "duration_samples": 8000,
         "source_start_sample": 0, "source_end_sample_exclusive": 8000},
        {"sound_asset_id": "speech", "event_class": "speech_playback",
         "sample_rate_hz": 16000, "duration_samples": 16000,
         "source_start_sample": 0, "source_end_sample_exclusive": 16000},
    ])
    pool = SoundEventPool.from_catalog(tmp_path / "pool.json")
    ids = {pool.draw(np.random.default_rng(i), "dog_bark").sound_asset_id
           for i in range(40)}
    assert ids <= {"a", "b"}
    assert "speech" not in ids
    assert pool.draw(np.random.default_rng(0), "speech_playback").sound_asset_id == "speech"


def test_pair_kind_map_comes_from_params():
    params = {"SOUND_EVENT_CLASS_BY_PAIR_KIND": {"dog": "dog_bark"}}
    assert event_class_for_pair_kind("dog", params) == "dog_bark"
    with pytest.raises(SoundPoolError, match="no entry"):
        event_class_for_pair_kind("human", params)
    with pytest.raises(SoundPoolError, match="missing SOUND_EVENT_CLASS"):
        event_class_for_pair_kind("dog", {})


def test_source_mode_is_required_and_has_no_default(tmp_path: Path):
    rng = np.random.default_rng(0)
    with pytest.raises(SoundPoolError, match="SOUND_SOURCE_MODE"):
        clip_source_from_params({}, rng, pair_kind="dog")
    assert clip_source_from_params(
        {"SOUND_SOURCE_MODE": "dry_canvas_window",
         "SOUND_ASSET": "dog_beagle_v2_scheduled_dry",
         "EVENT_SECONDS": 0.3,
         "SAMPLE_RATE_HZ": 16000,
         "DRY_CANVAS_SOURCE_START_SAMPLE": 3200,
         "DRY_CANVAS_SOURCE_END_SAMPLE_EXCLUSIVE": 8000},
        rng, pair_kind="dog"
    ) is None
    with pytest.raises(SoundPoolError, match="SOUND_ASSET"):
        clip_source_from_params(
            {"SOUND_SOURCE_MODE": "dry_canvas_window"}, rng, pair_kind="dog")
    with pytest.raises(SoundPoolError, match="SOUND_EVENT_POOL"):
        clip_source_from_params(
            {"SOUND_SOURCE_MODE": "event_pool",
             "SOUND_EVENT_CLASS_BY_PAIR_KIND": {"dog": "dog_bark"}},
            rng, pair_kind="dog")
    _catalog(tmp_path / "pool.json", [
        {"sound_asset_id": "a", "event_class": "dog_bark",
         "sample_rate_hz": 16000, "duration_samples": 3200,
         "source_start_sample": 0, "source_end_sample_exclusive": 3200},
    ])
    source = clip_source_from_params(
        {**POOL_PARAMS, "SOUND_EVENT_POOL": str(tmp_path / "pool.json")},
        rng, pair_kind="dog")
    clip = source.next()
    assert clip.sound_asset_id == "a"
    assert clip.duration_samples == 3200


def test_catalog_rejects_duration_that_does_not_match_the_source_window(tmp_path: Path):
    _catalog(tmp_path / "bad.json", [
        {"sound_asset_id": "a", "event_class": "dog_bark",
         "sample_rate_hz": 16000, "duration_samples": 4800,
         "source_start_sample": 0, "source_end_sample_exclusive": 1000},
    ])
    with pytest.raises(SoundPoolError, match="duration_samples=4800"):
        SoundEventPool.from_catalog(tmp_path / "bad.json")


def test_pool_rejects_clip_sample_rate_that_does_not_match_params(tmp_path: Path):
    _catalog(tmp_path / "pool.json", [
        {"sound_asset_id": "a", "event_class": "dog_bark",
         "sample_rate_hz": 44100, "duration_samples": 3200,
         "source_start_sample": 0, "source_end_sample_exclusive": 3200},
    ])
    with pytest.raises(SoundPoolError, match="44100"):
        clip_source_from_params(
            {**POOL_PARAMS, "SOUND_EVENT_POOL": str(tmp_path / "pool.json")},
            np.random.default_rng(0), pair_kind="dog")


def test_bind_distinct_roles_reuses_one_clip_per_role(tmp_path: Path):
    _catalog(tmp_path / "pool.json", [
        {"sound_asset_id": "a", "event_class": "dog_bark",
         "sample_rate_hz": 16000, "duration_samples": 3200,
         "source_start_sample": 0, "source_end_sample_exclusive": 3200},
        {"sound_asset_id": "b", "event_class": "dog_bark",
         "sample_rate_hz": 16000, "duration_samples": 8000,
         "source_start_sample": 0, "source_end_sample_exclusive": 8000},
    ])
    source = clip_source_from_params(
        {**POOL_PARAMS, "SOUND_EVENT_POOL": str(tmp_path / "pool.json")},
        np.random.default_rng(0), pair_kind="dog")
    bound = source.bind_distinct_roles(("target_actor", "non_target_actor"))
    first = bound.for_role("target_actor")
    second = bound.for_role("non_target_actor")
    assert first.sound_asset_id != second.sound_asset_id
    assert bound.for_role("target_actor").sound_asset_id == first.sound_asset_id


def test_bind_distinct_roles_fails_when_the_class_has_only_one_clip(tmp_path: Path):
    _catalog(tmp_path / "pool.json", [
        {"sound_asset_id": "a", "event_class": "dog_bark",
         "sample_rate_hz": 16000, "duration_samples": 3200,
         "source_start_sample": 0, "source_end_sample_exclusive": 3200},
    ])
    source = clip_source_from_params(
        {**POOL_PARAMS, "SOUND_EVENT_POOL": str(tmp_path / "pool.json")},
        np.random.default_rng(0), pair_kind="dog")
    with pytest.raises(SoundPoolError, match="distinct"):
        source.bind_distinct_roles(("target_actor", "non_target_actor"))


def test_module_docstring_requires_registered_sound_asset_ids():
    from avengine.assets import sound_pool
    assert "已注册" in sound_pool.__doc__
    assert "sound_asset_id" in sound_pool.__doc__

def test_pool_clip_keeps_explicit_speech_metadata(tmp_path: Path):
    path = tmp_path / "pool.json"
    _catalog(path, [
        {
            "sound_asset_id": "speech",
            "event_class": "speech_playback",
            "sample_rate_hz": 16000,
            "duration_samples": 16000,
            "source_start_sample": 0,
            "source_end_sample_exclusive": 16000,
            "speaker_id": "p225",
            "utterance_id": "001",
            "transcript": "Please call Stella.",
            "split": "eval",
        }
    ])
    clip = SoundEventPool.from_catalog(path).clips_for("speech_playback")[0]
    assert clip.speaker_id == "p225"
    assert clip.utterance_id == "001"
    assert clip.transcript == "Please call Stella."
    assert clip.split == "eval"

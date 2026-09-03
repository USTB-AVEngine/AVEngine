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
        {"SOUND_SOURCE_MODE": "event_pool",
         "SOUND_EVENT_POOL": str(tmp_path / "pool.json"),
         "SOUND_EVENT_CLASS_BY_PAIR_KIND": {"dog": "dog_bark"}},
        rng, pair_kind="dog")
    clip = source.next()
    assert clip.sound_asset_id == "a"
    assert clip.duration_samples == 3200

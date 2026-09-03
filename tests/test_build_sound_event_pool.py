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
    sound_asset_id_for_prepared,
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


def test_sound_asset_id_replaces_slashes():
    assert sound_asset_id_for_prepared("dog_bark/clip_e000.wav") == \
        "dog_bark__clip_e000"


def test_manifest_converts_to_a_loadable_pool(tmp_path: Path):
    output_root = tmp_path / "events"
    wav = output_root / "dog_bark" / "clip_e000.wav"
    _write_wav(wav, np.ones(3200) * 0.2)
    manifest = _manifest(tmp_path, [{
        "status": "event",
        "purpose": "pulse",
        "event_class": "dog_bark",
        "prepared": "dog_bark/clip_e000.wav",
    }], output_root=output_root)
    catalog_path = tmp_path / "pool.json"
    catalog = build_pool_catalog(manifest, catalog_path)
    assert catalog["clips"][0]["sound_asset_id"] == "dog_bark__clip_e000"
    assert catalog["clips"][0]["duration_samples"] == 3200
    pool = SoundEventPool.from_catalog(catalog_path)
    clip = pool.clips_for("dog_bark")[0]
    assert clip.duration_samples == 3200
    assert clip.source_end_sample_exclusive - clip.source_start_sample == 3200


def test_pulse_class_with_continuous_purpose_is_refused(tmp_path: Path):
    output_root = tmp_path / "events"
    wav = output_root / "dog_bark" / "long_e000.wav"
    _write_wav(wav, np.ones(16000) * 0.2)
    manifest = _manifest(tmp_path, [{
        "status": "event",
        "purpose": "continuous",
        "event_class": "dog_bark",
        "prepared": "dog_bark/long_e000.wav",
    }], output_root=output_root)
    with pytest.raises(PoolBuildError, match="purpose=continuous"):
        build_pool_catalog(manifest, tmp_path / "pool.json")

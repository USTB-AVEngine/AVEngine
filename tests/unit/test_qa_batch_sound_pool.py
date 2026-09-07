from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from avengine.qa import batch_sound_pool as mod


def fixture(tmp_path, monkeypatch, *, duration=1, header_delta=0, truncated=False):
    rate = 16000
    x = np.sin(2 * np.pi * 300 * np.arange(int(duration * rate)) / rate) * 0.2
    audio = tmp_path / "event.wav"
    sf.write(audio, x, rate, subtype="PCM_16")
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}")
    event_registry = tmp_path / "registry.json"
    event_registry.write_text(json.dumps({"sound_assets": [{
        "sound_asset_id": "bark", "semantic_sound_class": "dog_bark",
        "dry_audio": {"uri": audio.as_uri(), "sample_count": len(x) + header_delta,
                      "sample_rate_hz": rate, "channel_count": 1}}]}))
    manifest = tmp_path / "events.json"
    manifest.write_text(json.dumps({"library_root": str(tmp_path / "original"),
        "clips": [{"status": "event", "sound_asset_id": "bark", "source": "same_recording.wav",
                    "truncated": truncated}]}))
    monkeypatch.setattr(mod, "load_conditioned_sound_pool", lambda *args, **kwargs: [])
    spec = {"prepared_speech_manifest": str(prepared), "sound_event_registry": str(event_registry),
            "sound_event_manifest": str(manifest), "object_sound_classes": {"speaker": ["speech_playback"]},
            "species_sound_classes": {"dog": ["dog_bark"]}}
    registry = {"assets": [{"asset_id": "dog", "entity_class": "articulated_animal",
                            "identity": {"species_id": "dog"}},
                           {"asset_id": "speaker", "entity_class": "rigid_object",
                            "identity": {"object_type": "speaker", "category": "audio_playback"}}]}
    return spec, registry, audio


def test_joined_pool_preserves_pcm_and_uses_explicit_species_mapping(tmp_path, monkeypatch):
    spec, registry, audio = fixture(tmp_path, monkeypatch)
    before = audio.read_bytes()
    result = mod.build_batch_sound_pool(spec, registry)
    assert audio.read_bytes() == before
    item = result["sounds"][0]
    assert item["compatible_asset_ids"] == ["dog"]
    assert item["source_activity_intervals_samples"]
    assert item["sound_identity_id"].endswith("same_recording.wav")
    assert item["activity_calibration"] == "placeholder"
    assert item["activity_is_qa_event_count"] is False


def test_registered_header_disagreement_stops_before_preallocation(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, header_delta=1)
    with pytest.raises(ValueError, match="header differs"):
        mod.build_batch_sound_pool(spec, registry)


def test_truncation_and_too_short_animal_keep_explicit_rejections(tmp_path, monkeypatch):
    spec, registry, _ = fixture(tmp_path, monkeypatch, truncated=True)
    result = mod.build_batch_sound_pool(spec, registry)
    assert result["sounds"] == []
    assert result["rejected"][0]["reason"] == "registered_event_is_truncated"
    second = tmp_path / "short"
    second.mkdir()
    spec, registry, _ = fixture(second, monkeypatch, duration=0.2)
    result = mod.build_batch_sound_pool(spec, registry)
    assert result["sounds"] == []
    assert result["rejected"][0]["reason"] == "animal_activity_below_declared_placeholder_minimum"

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
        "normalization_policy": {"mode": "future_rms_label", "target_dbfs": -18.5},
        "custom_gain_metadata": {"gain_db": 2.25, "origin": "registry"},
        "dry_audio": {"uri": audio.as_uri(), "sample_count": len(x) + header_delta,
                      "sample_rate_hz": rate, "channel_count": 1}}]}))
    manifest = tmp_path / "events.json"
    manifest.write_text(json.dumps({"library_root": str(tmp_path / "original"),
        "clips": [{"status": "event", "sound_asset_id": "bark", "source": "same_recording.wav",
                    "applied_gain_db": -1.25,
                    "custom_gain_metadata": {"gain_db": 3.5, "origin": "manifest"},
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


def test_source_normalization_metadata_preserves_policy_gain_peak_and_source(tmp_path, monkeypatch):
    spec, registry, audio = fixture(tmp_path, monkeypatch)
    result = mod.build_batch_sound_pool(spec, registry)
    item = result["sounds"][0]
    metadata = item["source_normalization"]
    expected_peak = float(np.max(np.abs(sf.read(audio, dtype="float64")[0])))

    assert metadata["policy"] == {"mode": "future_rms_label", "target_dbfs": -18.5}
    assert metadata["target_dbfs"] == -18.5
    assert metadata["applied_gain_db"] == -1.25
    assert metadata["measured_peak_source"] == "registered_event_pcm"
    assert metadata["measured_peak_dbfs"] == pytest.approx(20 * np.log10(expected_peak))
    assert metadata["source"]["event_registry_path"] == str((tmp_path / "registry.json").resolve())
    assert metadata["source"]["event_manifest_path"] == str((tmp_path / "events.json").resolve())
    assert metadata["metadata"]["raw"]["custom_gain_metadata"] == {
        "gain_db": 2.25, "origin": "registry"
    }
    assert metadata["metadata"]["related_0"]["custom_gain_metadata"] == {
        "gain_db": 3.5, "origin": "manifest"
    }
    assert item["linear_gain"] == 1.0
    assert item["normalization_applied"] is False


def test_conditioned_loader_preserves_unknown_policy_and_gain_facts(tmp_path):
    audio = tmp_path / "speech" / "clip.wav"
    audio.parent.mkdir()
    sf.write(audio, np.full(16000, 0.125), 16000, subtype="PCM_16")
    manifest = tmp_path / "prepared.json"
    manifest.write_text(json.dumps({
        "prepared_set_id": "prepared_custom",
        "target_peak_dbfs": -3.0,
        "clips": [{
            "status": "prepared",
            "prepared": "speech/clip.wav",
            "prepared_audio_id": "prepared_custom_clip",
            "source_crop_start_sample": 0,
            "facts": {
                "source_rate_hz": 16000,
                "prepared_peak_dbfs": -7.25,
                "applied_gain_db": 2.5,
                "normalization_applied": False,
                "normalization_policy": {
                    "mode": "future_unseen_policy",
                    "target_dbfs": -14.5,
                },
                "custom_gain_metadata": {"gain_db": 1.75},
            },
            "source_activity": {
                "active_duration_s": 1.0,
                "intervals": [{"start_sample": 0, "end_sample_exclusive": 16000}],
            },
            "gender": "M",
            "transcript": "custom source",
            "human_review": {"status": "pending"},
        }],
    }))

    item = mod.load_conditioned_sound_pool(
        json.loads(manifest.read_text()), source_path=manifest
    )[0]
    metadata = item["source_normalization"]
    assert metadata["policy"] == {
        "mode": "future_unseen_policy", "target_dbfs": -14.5
    }
    assert metadata["target_dbfs"] == -14.5
    assert metadata["applied_gain_db"] == 2.5
    assert metadata["normalization_applied"] is False
    assert metadata["measured_peak_dbfs"] == -7.25
    assert metadata["measured_peak_source"] == "facts.prepared_peak_dbfs"
    assert metadata["metadata"]["facts"]["custom_gain_metadata"] == {"gain_db": 1.75}
    assert metadata["metadata"]["container"]["target_peak_dbfs"] == -3.0
    assert metadata["source"]["manifest_path"] == str(manifest.resolve())
    assert item.get("linear_gain") is None
    assert "facts" not in item


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


def test_loader_reentry_keeps_source_processing_separate_from_runtime_flags(tmp_path):
    audio = tmp_path / "event.wav"
    sf.write(audio, np.full(8000, 0.125), 16000, subtype="PCM_16")
    origin_manifest = str(tmp_path / "original-events.json")
    pool_path = tmp_path / "batch-pool.json"
    source_metadata = {
        "policy": {"mode": "future_reference", "target_dbfs": -4.0},
        "applied_gain_db": 12.0,
        "normalization_applied": True,
        "measured_peak_dbfs": -4.0,
        "source": {"manifest_path": origin_manifest},
        "metadata": {"raw": {"custom_gain_metadata": {"unit": "source-db"}}},
    }
    item = {
        "sound_asset_id": "new_event",
        "path": str(audio),
        "sound_class": "test_cue",
        "source_activity_intervals_samples": [[0, 8000]],
        "linear_gain": 0.35,
        "normalization_applied": False,
        "source_normalization": source_metadata,
    }
    result = mod.load_conditioned_sound_pool(
        {"sounds": [item]}, source_path=pool_path
    )[0]
    restored = result["source_normalization"]
    assert restored["normalization_applied"] is True
    assert restored["applied_gain_db"] == 12.0
    assert restored["source"]["manifest_path"] == origin_manifest
    assert restored["source"]["input_manifest_path"] == str(pool_path)
    assert restored["metadata"] == source_metadata["metadata"]
    assert result["linear_gain"] == 0.35
    assert item["source_normalization"] == source_metadata
    assert "input_manifest_path" not in source_metadata["source"]

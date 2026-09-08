import json
from pathlib import Path

import pytest

from avengine.rooms.qa_delivery import _rendered_sound_registry, _reviewed_occluder_registry


def test_sound_semantics_come_from_matching_rendered_pool_entry(tmp_path):
    audio = tmp_path / "event.wav"
    audio.touch()
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"sounds": [
        {"sound_asset_id": "event", "path": "event.wav", "sound_class": "music_playback"},
        {"sound_asset_id": "unused", "path": "unused.wav", "sound_class": "dog_bark"},
    ]}))
    result = _rendered_sound_registry(
        {"sound_pool": str(pool)},
        {"events": [{"sound_asset_id": "event"}]},
        {"inputs": {"dry_assets": {"event": {"path": str(audio), "sha256": "existing-readback"}}}},
        repository=tmp_path,
    )
    assert result["sound_count"] == 1
    assert result["sounds"][0]["semantic_sound_class"] == "music_playback"
    assert result["sounds"][0]["rendered_pcm_sha256"] == "existing-readback"


def test_sound_registry_rejects_a_different_pcm_for_the_same_id(tmp_path):
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"sounds": [
        {"sound_asset_id": "event", "path": "requested.wav", "sound_class": "music_playback"}
    ]}))
    with pytest.raises(ValueError, match="PCM differs"):
        _rendered_sound_registry(
            {"sound_pool": str(pool)},
            {"events": [{"sound_asset_id": "event"}]},
            {"inputs": {"dry_assets": {"event": {"path": str(tmp_path / "rendered.wav")}}}},
            repository=tmp_path,
        )


def test_legacy_audio_without_input_registry_remains_supported(tmp_path):
    assert _rendered_sound_registry({}, {}, {}, repository=tmp_path)["sounds"] == []


def test_occluder_labels_do_not_append_internal_appearance_values():
    result = _reviewed_occluder_registry(
        {"actors": {
            "animal": {"status": "reviewed", "entity_kind": "animal",
                       "attribute_field": "coat_profile.value", "value": "standard_red"},
            "human": {"status": "reviewed", "entity_kind": "human",
                      "attribute_field": "top_color", "value": "green"},
        }},
        {"animal": {"display_label": "Shiba Inu v2"}, "human": {"display_label": "Human (green top)"}},
    )
    assert result["animal"]["display_label"] == "Shiba Inu"
    assert "standard_red" not in result["animal"]["display_label"]
    assert result["animal"]["appearance_value"] == "standard_red"
    assert result["human"]["display_label_zh"] == "绿色上衣的人"


def test_delivery_uses_an_explicit_asset_registry(tmp_path):
    from avengine.rooms.qa_delivery import _asset_registry
    registry = tmp_path / "alternate-assets.json"
    registry.write_text(json.dumps({"assets": [
        {"asset_id": "new_actor", "display_label": "new observable asset"}
    ]}))
    assert _asset_registry(tmp_path, registry)["new_actor"]["display_label"] == "new observable asset"

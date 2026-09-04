from pathlib import Path
import importlib.util
import json

import pytest

spec = importlib.util.spec_from_file_location("apartment_audio_cli", Path(__file__).resolve().parents[2] / "tools/dataset/render_current_apartment_dynamic_audio.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_multiple_speech_paths_need_no_legacy_beagle(tmp_path):
    mapping = tmp_path / "sounds.json"
    mapping.write_text(json.dumps({"voice_a": "a.wav", "voice_b": "b.wav"}))
    paths = cli.external_sound_paths(mapping_path=mapping)
    assert paths == {"voice_a": tmp_path / "a.wav", "voice_b": tmp_path / "b.wav"}
    assert "dog_beagle_v2_scheduled_dry" not in paths


def test_conflicting_bindings_are_not_silently_overwritten(tmp_path):
    mapping = tmp_path / "sounds.json"
    mapping.write_text(json.dumps({"voice_a": "a.wav"}))
    with pytest.raises(ValueError, match="duplicate"):
        cli.external_sound_paths(mapping_path=mapping, assignments=["voice_a=b.wav"])
    with pytest.raises(ValueError, match="duplicate"):
        cli.external_sound_paths(beagle_audio=tmp_path / "a.wav", assignments=["dog_beagle_v2_scheduled_dry=b.wav"])

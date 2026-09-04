"""Register split events into avengine_m6_sound_asset_registry_v1."""

from __future__ import annotations

import inspect
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "assets"))

from avengine.registry.sources import validate_sound_asset_registry  # noqa: E402
from build_sound_event_pool import build_pool_catalog  # noqa: E402
from register_sound_event_assets import (  # noqa: E402
    OWNER_ADMISSION_NOTE,
    dry_audio_byte_errors,
    main,
    register_sound_event_assets,
)
from split_sound_library_events import split_library  # noqa: E402


def _tone(freq: float, rate: int, seconds: float, amplitude: float = 0.4):
    time = np.arange(int(round(rate * seconds))) / rate
    return amplitude * np.sin(2 * np.pi * freq * time)


def _write(path: Path, samples: np.ndarray, rate: int = 16000) -> None:
    ints = np.clip(np.round(samples * 32767), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(ints.tobytes())


def _catalog(tmp_path: Path) -> Path:
    library = tmp_path / "prepared"
    gap = np.zeros(int(16000 * 0.8))
    _write(
        library / "dog_bark" / "one.wav",
        np.concatenate([_tone(800, 16000, 0.25), gap, _tone(1200, 16000, 0.3)]),
    )
    _write(
        library / "speech_playback" / "talk.wav",
        _tone(200, 16000, 1.0, amplitude=0.2),
    )
    (library / "prepared_manifest.json").write_text(
        json.dumps(
            {
                "schema": "avengine_prepared_sound_clip_v1",
                "clips": [
                    {
                        "source": "speech_playback/talk.wav",
                        "prepared": "speech_playback/talk.wav",
                        "status": "prepared",
                        "speaker_id": "p225",
                        "utterance_id": "001",
                        "transcript": "Please call Stella.",
                        "split": "eval",
                    }
                ],
            }
        )
    )
    events = tmp_path / "events"
    split_library(library, events)
    catalog = tmp_path / "pool.json"
    build_pool_catalog(events / "event_manifest.json", catalog)
    return catalog


def _register_args(catalog: Path, output: Path) -> list[str]:
    return [
        "--catalog", str(catalog),
        "--output", str(output),
        "--registry-id", "m6_sound_event_test_v1",
        "--revision", "v1",
        "--permitted-event-usage", "sequential_sources",
        "--permitted-event-usage", "one_active_of_n",
        "--normalization-policy", "preserve",
    ]


def test_permitted_event_usage_has_no_code_default() -> None:
    module = Path(__file__).resolve().parents[1] / "tools" / "assets" / (
        "register_sound_event_assets.py"
    )
    main_src = module.read_text().split("def main", 1)[1]
    usage = main_src.split("--permitted-event-usage", 1)[1].split(
        "add_argument", 1)[0]
    assert "required=True" in usage
    assert "default=" not in usage
    policy = main_src.split("--normalization-policy", 1)[1].split(
        "add_argument", 1)[0]
    assert "required=True" in policy
    assert "default=" not in policy
    assert inspect.getsource(main).count("permitted-event-usage") >= 1


def test_register_derives_dry_audio_and_passes_validator(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    output = tmp_path / "registry.json"
    registry = register_sound_event_assets(
        catalog,
        output,
        permitted_event_usage=["one_active_of_n", "sequential_sources"],
        normalization_policy="preserve",
        registry_id="m6_sound_event_test_v1",
        revision="v1",
    )
    payload = json.loads(catalog.read_text())
    assert len(registry["sound_assets"]) == len(payload["clips"])
    assert validate_sound_asset_registry(registry) == []
    assert dry_audio_byte_errors(registry) == []
    dog = next(
        item for item in registry["sound_assets"]
        if item["semantic_sound_class"] == "dog_bark"
    )
    assert dog["taxonomy_path"] == ["animal", "dog_bark"]
    assert dog["dry_audio"]["channel_count"] == 1
    assert dog["normalization_policy"] == {"mode": "preserve", "target_dbfs": None}
    assert dog["permitted_event_usage"] == ["one_active_of_n", "sequential_sources"]
    assert dog["admissibility"] == "research"
    assert OWNER_ADMISSION_NOTE in dog["provenance"]["origin"]
    assert dog["sound_asset_id"].startswith("sound_dog_bark_")
    assert dog["sound_asset_id"].endswith("_v1")
    speech = next(
        item for item in registry["sound_assets"]
        if item["semantic_sound_class"] == "speech_playback"
    )
    assert speech["speaker_id"] == "p225"
    assert speech["utterance_id"] == "001"
    assert speech["transcript"] == "Please call Stella."
    assert speech["split"] == "eval"


def test_missing_permitted_event_usage_fails(tmp_path: Path, capsys) -> None:
    catalog = _catalog(tmp_path)
    args = [
        "--catalog", str(catalog),
        "--output", str(tmp_path / "registry.json"),
        "--registry-id", "m6_sound_event_test_v1",
        "--revision", "v1",
        "--normalization-policy", "preserve",
    ]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code != 0
    assert not (tmp_path / "registry.json").exists()


def test_tampered_dry_audio_sha256_fails_validation(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    output = tmp_path / "registry.json"
    registry = register_sound_event_assets(
        catalog,
        output,
        permitted_event_usage=["sequential_sources", "one_active_of_n"],
        normalization_policy="preserve",
        registry_id="m6_sound_event_test_v1",
        revision="v1",
    )
    assert validate_sound_asset_registry(registry) == []
    tampered = json.loads(output.read_text())
    tampered["sound_assets"][0]["dry_audio"]["sha256"] = "0" * 64
    errors = validate_sound_asset_registry(tampered)
    assert errors
    assert any("sha256" in item or "canonical" in item for item in errors)
    assert dry_audio_byte_errors(tampered)


def test_cli_writes_the_same_registry(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    output = tmp_path / "registry.json"
    assert main(_register_args(catalog, output)) == 0
    registry = json.loads(output.read_text())
    assert validate_sound_asset_registry(registry) == []
    assert dry_audio_byte_errors(registry) == []

"""Studio config path resolution tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from avengine.studio.config import load_studio_config


def test_repository_root_is_relative_to_config_file_not_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repo"
    config_dir = repository / "tools" / "studio"
    config_dir.mkdir(parents=True)
    (repository / "review").mkdir()
    (repository / "tasks").mkdir()
    room_registry = repository / "examples" / "rooms.json"
    room_registry.parent.mkdir(parents=True)
    room_registry.write_text("{}", encoding="utf-8")
    registries = {}
    for name in ("source_endpoints", "sound_assets", "entity_assets"):
        path = repository / "examples" / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        registries[name] = str(path.relative_to(repository))

    sound = repository / "voice.wav"
    sound.write_bytes(b"RIFF")
    config_path = config_dir / "studio_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "avengine_studio_config_v1",
                "repository_root": "../..",
                "python_executable": sys.executable,
                "review_root": "review",
                "tasks_root": "tasks",
                "room_registry": str(room_registry.relative_to(repository)),
                "registries": registries,
                "external_sound_assets": {
                    "voice": str(sound.relative_to(repository))
                },
                "task_templates": {},
            }
        ),
        encoding="utf-8",
    )
    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    config = load_studio_config(config_path)

    assert config.repository_root == repository.resolve()
    assert config.review_root == (repository / "review").resolve()
    assert config.tasks_root == (repository / "tasks").resolve()
    assert config.room_registry_path == room_registry.resolve()
    assert config.external_sound_assets == {
        "voice": str(sound.resolve())
    }

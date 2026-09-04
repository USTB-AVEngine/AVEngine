"""Launcher source and child-process environment tests."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPOSITORY / "tools" / "studio" / "run_studio_server.py"
SPEC = importlib.util.spec_from_file_location("studio_launcher", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def test_launcher_pins_source_and_cwd_for_children(tmp_path: Path, monkeypatch) -> None:
    selected = tmp_path / "selected"
    selected_src = selected / "src" / "avengine"
    selected_src.mkdir(parents=True)
    (selected_src / "__init__.py").write_text(
        "ORIGIN = 'selected'\n", encoding="utf-8"
    )

    stale = tmp_path / "stale"
    stale_src = stale / "src" / "avengine"
    stale_src.mkdir(parents=True)
    (stale_src / "__init__.py").write_text(
        "ORIGIN = 'stale'\n", encoding="utf-8"
    )

    config_path = tmp_path / "studio.json"
    config_path.write_text(
        '{"repository_root": "selected"}\n', encoding="utf-8"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("PYTHONPATH", str(stale_src))

    repository_root = LAUNCHER._bootstrap_repository_source(config_path)

    assert repository_root == selected.resolve()
    assert Path.cwd() == selected.resolve()
    assert Path(sys.path[0]) == (selected / "src").resolve()
    assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(
        (selected / "src").resolve()
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, avengine; print(avengine.ORIGIN); print(os.getcwd())",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ.copy(),
    )
    assert child.stdout.splitlines() == ["selected", str(selected.resolve())]


def test_launcher_subprocess_resolves_relative_config_before_chdir(
    tmp_path: Path,
) -> None:
    config_parent = tmp_path / "launcher"
    config_parent.mkdir()
    review_root = tmp_path / "review"
    review_root.mkdir()
    room_registry = tmp_path / "room_registry.json"
    room_registry.write_text("{}", encoding="utf-8")
    registries = {}
    for name in ("source_endpoints", "sound_assets", "entity_assets"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}", encoding="utf-8")
        registries[name] = str(path)
    config_path = config_parent / "studio.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "avengine_studio_config_v1",
                "repository_root": str(REPOSITORY),
                "python_executable": sys.executable,
                "review_root": str(review_root),
                "tasks_root": str(tmp_path / "tasks"),
                "room_registry": str(room_registry),
                "registries": registries,
                "task_templates": {},
                "port": 0,
            }
        ),
        encoding="utf-8",
    )
    runner = r"""
import runpy
import sys
import types

namespace = runpy.run_path(sys.argv[1], run_name="studio_launcher_subprocess")
fake_server_module = types.ModuleType("avengine.studio.server")

class FakeServer:
    def __init__(self, config):
        self.server_address = (config.host, config.port)
        print("ROOT=" + str(config.repository_root), flush=True)

    def serve_forever(self):
        raise KeyboardInterrupt

    def server_close(self):
        pass

fake_server_module.StudioHTTPServer = FakeServer
sys.modules["avengine.studio.server"] = fake_server_module
sys.argv = [sys.argv[1], "--config", "launcher/studio.json"]
raise SystemExit(namespace["main"]())
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "/data/jzy/code/AVEngine-lead-a/src"
    result = subprocess.run(
        [sys.executable, "-c", runner, str(LAUNCHER_PATH)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"ROOT={REPOSITORY.resolve()}" in result.stdout

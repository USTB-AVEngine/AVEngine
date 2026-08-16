from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from avengine.backends.spear_ue.launch import parallel_instance_settings


REPOSITORY = Path(__file__).resolve().parents[2]
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "spear_apartment_s1_runner",
    REPOSITORY / "tools/m6y/run_spear_apartment_canary.py",
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER)


@pytest.mark.fast_unit
@pytest.mark.parametrize(
    ("rpc_port", "graphics_adapter", "expected"),
    [
        (
            1024,
            None,
            {
                "rpc_port": 1024,
                "graphics_adapter": None,
                "temp_dir": "tmp/spear_instance_1024",
                "log": "SpearSim_rpc_1024.log",
                "shared_memory_initial_unique_id": 10240000,
            },
        ),
        (
            "65535",
            "3",
            {
                "rpc_port": 65535,
                "graphics_adapter": 3,
                "temp_dir": "tmp/spear_instance_65535",
                "log": "SpearSim_rpc_65535.log",
                "shared_memory_initial_unique_id": 655350000,
            },
        ),
    ],
)
def test_parallel_instance_settings_preserves_upstream_defaults(
    rpc_port: object, graphics_adapter: object | None, expected: dict[str, object]
) -> None:
    assert parallel_instance_settings(rpc_port, graphics_adapter) == expected


@pytest.mark.fast_unit
@pytest.mark.parametrize(
    ("rpc_port", "graphics_adapter", "message"),
    [
        (1023, None, "rpc_port must be in [1024, 65535], got 1023"),
        (65536, None, "rpc_port must be in [1024, 65535], got 65536"),
        (1024, -1, "graphics_adapter must be non-negative, got -1"),
    ],
)
def test_parallel_instance_settings_preserves_upstream_validation(
    rpc_port: object, graphics_adapter: object | None, message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        parallel_instance_settings(rpc_port, graphics_adapter)


def _fake_config(events: list[object]) -> types.SimpleNamespace:
    config = types.SimpleNamespace(
        SPEAR=types.SimpleNamespace(
            INSTANCE=types.SimpleNamespace(
                COMMAND_LINE_ARGS=types.SimpleNamespace(),
            ),
            ENVIRONMENT_VARS=types.SimpleNamespace(),
        ),
        SP_SERVICES=types.SimpleNamespace(
            INITIALIZE_ENGINE_SERVICE=types.SimpleNamespace(),
            RPC_SERVICE=types.SimpleNamespace(),
        ),
        SP_CORE=types.SimpleNamespace(),
    )
    config.defrost = lambda: events.append("defrost")
    config.freeze = lambda: events.append("freeze")
    return config


@pytest.mark.fast_unit
def test_apartment_runner_uses_explicit_launcher_without_checkout_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "external-packaged-game" / "SpearSim.sh"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    assert not (executable.parent / "examples").exists()

    events: list[object] = []
    config = _fake_config(events)
    instance = object()
    fake_spear = types.SimpleNamespace(
        get_config=lambda **kwargs: (events.append(("get_config", kwargs)) or config),
        configure_system=lambda *, config: events.append(("configure_system", config)),
        Instance=lambda *, config: (events.append(("instance", config)) or instance),
    )
    monkeypatch.setattr(_RUNNER, "spear_client", fake_spear)

    original_sys_path = list(sys.path)
    result = _RUNNER._configure_instance(
        argparse.Namespace(
            spear_executable=executable,
            rpc_port="24567",
            graphics_adapter="3",
        ),
        native_map="/Game/Maps/Apartment",
    )

    assert result is instance
    assert sys.path == original_sys_path
    assert str(executable.parent / "examples") not in sys.path
    assert events == [
        ("get_config", {"user_config_files": []}),
        "defrost",
        "freeze",
        ("configure_system", config),
        ("instance", config),
    ]
    assert config.SPEAR.LAUNCH_MODE == "game"
    assert config.SPEAR.INSTANCE.GAME_EXECUTABLE == str(executable)
    assert (
        config.SPEAR.INSTANCE.INITIALIZE_CLIENT_MAX_TIME_SECONDS
        == _RUNNER.INITIALIZE_CLIENT_MAX_TIME_SECONDS
    )
    assert (
        config.SPEAR.INSTANCE.CLIENT_INTERNAL_TIMEOUT_SECONDS
        == _RUNNER.CLIENT_INTERNAL_TIMEOUT_SECONDS
    )
    assert (
        config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP
        is True
    )
    assert (
        config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP
        == "/Game/Maps/Apartment"
    )
    assert config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.FIXED_DELTA_TIME == 1.0 / 15
    assert config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT == 24567
    assert config.SPEAR.INSTANCE.TEMP_DIR == "tmp/spear_instance_24567"
    assert config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.log == "SpearSim_rpc_24567.log"
    assert config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.renderoffscreen is None
    assert config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.graphicsadapter == 3
    assert config.SP_CORE.SHARED_MEMORY_INITIAL_UNIQUE_ID == 245670000
    assert (
        config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES
        == "/etc/vulkan/icd.d/nvidia_icd.json"
    )


@pytest.mark.fast_unit
def test_apartment_runner_cli_requires_explicit_spear_executable(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "external-packaged-game" / "SpearSim.sh"
    assert not executable.exists()
    common_args = ["--output-dir", str(tmp_path / "output"), "--dry-run"]
    parsed = _RUNNER.parse_args(
        ["--spear-executable", str(executable), *common_args]
    )
    assert parsed.spear_executable == executable
    assert not hasattr(parsed, "spear_root")

    with pytest.raises(SystemExit, match="2"):
        _RUNNER.parse_args(common_args)


_DIRECT_HOST_GAME_RUNNERS = (
    "tools/m6y/run_spear_apartment_canary.py",
    "tools/m6y/run_spear_mp3d_canary.py",
    "tools/m6y/run_spear_replicacad_canary.py",
    "tools/m6z/run_spear_kujiale_canary.py",
    "tools/qa/capture_skokloster_strict_two_human_episode.py",
    "tools/qa/probe_packaged_imported_glb_room.py",
    "tools/qa/probe_packaged_skokloster_room.py",
)


def _load_s3c_runner(module_name: str, relative_path: str) -> types.ModuleType:
    original_sys_path = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, REPOSITORY / relative_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_sys_path


def _global_spear_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return {
        name
        for name in imported
        if name == "spear" or name.startswith("spear.")
    }


@pytest.mark.fast_unit
def test_direct_host_game_runners_use_the_namespaced_client() -> None:
    for relative_path in _DIRECT_HOST_GAME_RUNNERS:
        source = REPOSITORY / relative_path
        assert not _global_spear_imports(source), source
        assert "avengine.backends.spear_ue" in source.read_text(encoding="utf-8")

    for relative_path in (
        "tools/m6y/run_spear_mp3d_canary.py",
        "tools/m6y/run_spear_replicacad_canary.py",
        "tools/m6z/run_spear_kujiale_canary.py",
        "tools/qa/capture_skokloster_strict_two_human_episode.py",
    ):
        source = (REPOSITORY / relative_path).read_text(encoding="utf-8")
        assert "from render_in_apartment import parallel_instance_settings" not in source

    for relative_path in (
        "tools/qa/probe_packaged_imported_glb_room.py",
        "tools/qa/probe_packaged_skokloster_room.py",
    ):
        source = (REPOSITORY / relative_path).read_text(encoding="utf-8")
        assert 'spear_root / "python"' not in source
        assert 'args.spear_root.resolve() / "python"' not in source


@pytest.mark.fast_unit
def test_skokloster_capture_help_bootstraps_namespaced_client_without_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "tools/qa/capture_skokloster_strict_two_human_episode.py"),
            "--help",
        ],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--spear-root" in result.stdout


def _fake_spear_client(
    events: list[object], instance: object
) -> types.SimpleNamespace:
    config = _fake_config(events)
    return types.SimpleNamespace(
        config=config,
        get_config=lambda **kwargs: (events.append(("get_config", kwargs)) or config),
        configure_system=lambda *, config: events.append(("configure_system", config)),
        Instance=lambda *, config: (events.append(("instance", config)) or instance),
    )


def _write_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.mark.fast_unit
@pytest.mark.parametrize(
    ("module_name", "relative_path"),
    [
        ("s3c_mp3d_runner", "tools/m6y/run_spear_mp3d_canary.py"),
        ("s3c_replicacad_runner", "tools/m6y/run_spear_replicacad_canary.py"),
    ],
)
def test_editor_runners_use_namespaced_client_without_examples_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    relative_path: str,
) -> None:
    runner = _load_s3c_runner(module_name, relative_path)
    spear_root = tmp_path / "spear-root"
    spear_root.mkdir()
    editor = _write_executable(tmp_path / "UnrealEditor")
    project = tmp_path / "isolated" / "SpearSim.uproject"
    project.parent.mkdir()
    project.write_text("{}", encoding="utf-8")
    assert not (spear_root / "examples").exists()

    events: list[object] = []
    instance = object()
    fake_client = _fake_spear_client(events, instance)
    monkeypatch.setattr(runner, "spear_client", fake_client)
    original_sys_path = list(sys.path)

    result, returned_root = runner._configure_instance(
        argparse.Namespace(
            spear_root=spear_root,
            unreal_editor=editor,
            ue_project=project,
            rpc_port=24681,
            graphics_adapter=2,
        ),
        {"exposure_and_lighting": {"console_commands": ["r.Test 1"]}},
    )

    expected_map = (
        runner.ENTRY_MAP if hasattr(runner, "ENTRY_MAP") else runner.M5_1_MAP_PATH
    )
    assert result is instance
    assert returned_root == spear_root.resolve()
    assert sys.path == original_sys_path
    assert events == [
        ("get_config", {"user_config_files": []}),
        "defrost",
        "freeze",
        ("configure_system", fake_client.config),
        ("instance", fake_client.config),
    ]
    config = fake_client.config
    assert config.SPEAR.LAUNCH_MODE == "editor"
    assert config.SPEAR.INSTANCE.EDITOR_EXECUTABLE == str(editor)
    assert config.SPEAR.INSTANCE.EDITOR_UPROJECT == str(project)
    assert config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP == expected_map
    assert config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT == 24681
    assert config.SPEAR.INSTANCE.TEMP_DIR == "tmp/spear_instance_24681"


@pytest.mark.fast_unit
def test_kujiale_runner_needs_no_spear_root_and_cli_drops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_s3c_runner(
        "s3c_kujiale_runner", "tools/m6z/run_spear_kujiale_canary.py"
    )
    editor = _write_executable(tmp_path / "UnrealEditor")
    project = tmp_path / "external" / "Kujiale.uproject"
    project.parent.mkdir()
    project.write_text("{}", encoding="utf-8")

    events: list[object] = []
    instance = object()
    fake_client = _fake_spear_client(events, instance)
    monkeypatch.setattr(runner, "spear_client", fake_client)
    original_sys_path = list(sys.path)

    result = runner._configure_spear(
        argparse.Namespace(
            rpc_port=24682,
            graphics_adapter=1,
            unreal_editor=editor,
            uproject=project,
        ),
        {"map_path": "/Game/Maps/Kujiale"},
    )

    assert result is instance
    assert sys.path == original_sys_path
    config = fake_client.config
    assert config.SPEAR.LAUNCH_MODE == "editor"
    assert config.SPEAR.INSTANCE.EDITOR_EXECUTABLE == str(editor)
    assert config.SPEAR.INSTANCE.EDITOR_UPROJECT == str(project)
    assert config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP == (
        "/Game/Maps/Kujiale"
    )
    assert config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT == 24682

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(REPOSITORY / "tools/m6z/run_spear_kujiale_canary.py"),
            "--uproject",
            str(project),
            "--unreal-editor",
            str(editor),
            "--source-stage",
            str(tmp_path / "stage"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    parsed = runner.parse_args()
    assert not hasattr(parsed, "spear_root")

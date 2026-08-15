from __future__ import annotations

import argparse
import importlib.util
import re
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
def test_apartment_runner_no_longer_requires_spear_examples_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spear_root = tmp_path / "spear-root"
    executable = (
        spear_root
        / "cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh"
    )
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    assert not (spear_root / "examples").exists()

    events: list[object] = []
    config = _fake_config(events)
    instance = object()
    fake_spear = types.SimpleNamespace(
        get_config=lambda **kwargs: (events.append(("get_config", kwargs)) or config),
        configure_system=lambda *, config: events.append(("configure_system", config)),
        Instance=lambda *, config: (events.append(("instance", config)) or instance),
    )
    monkeypatch.setitem(sys.modules, "spear", fake_spear)

    original_sys_path = list(sys.path)
    result, returned_root = _RUNNER._configure_instance(
        argparse.Namespace(
            spear_root=spear_root,
            rpc_port="24567",
            graphics_adapter="3",
        ),
        native_map="/Game/Maps/Apartment",
    )

    assert result is instance
    assert returned_root == spear_root.resolve()
    assert sys.path == original_sys_path
    assert str(spear_root / "examples") not in sys.path
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

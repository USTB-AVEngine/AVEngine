"""Hermetic import coverage for the AVEngine-local SPEAR client slice."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import textwrap

CLIENT_ROOT = Path(__file__).resolve().parents[2] / "src/avengine/backends/spear_ue/client"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_client_closure_does_not_import_global_spear() -> None:
    for path in CLIENT_ROOT.rglob("*.py"):
        imported = _imported_modules(path)
        assert "spear" not in imported
        assert "spear_ext" not in imported
        assert not any(name.startswith("spear.") for name in imported)
        assert not any(name.startswith("spear_ext.") for name in imported)


def test_config_import_is_hermetic_until_instance_needs_native_extension() -> None:
    repo_root = CLIENT_ROOT.parents[4]
    code = textwrap.dedent(
        """
        import sys

        class BlockedOptionalImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "spear":
                    raise AssertionError("the AVEngine client imported global spear")
                if fullname == "spear_ext":
                    raise AssertionError("the AVEngine client imported global spear_ext")
                if fullname == "avengine_spear_ext":
                    raise ModuleNotFoundError(
                        "No module named 'avengine_spear_ext'",
                        name="avengine_spear_ext",
                    )
                return None

        sys.meta_path.insert(0, BlockedOptionalImports())
        from avengine.backends.spear_ue import client
        config = client.get_config()
        assert config.SPEAR.LAUNCH_MODE in {"editor", "game", "none"}
        assert client.__can_import_avengine_spear_ext__ is False
        assert client.__can_import_unreal__ is False
        assert "spear" not in sys.modules
        assert "spear_ext" not in sys.modules
        try:
            client.Instance()
        except RuntimeError as error:
            message = str(error)
            assert "avengine_spear_ext" in message
            assert "Instance" in message
        else:
            raise AssertionError("Instance unexpectedly started without avengine_spear_ext")
        """
    )
    environment = os.environ.copy()
    source_root = str(repo_root / "src")
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root if not previous else f"{source_root}{os.pathsep}{previous}"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_extension_present_client_factory_is_namespaced() -> None:
    repo_root = CLIENT_ROOT.parents[4]
    code = textwrap.dedent(
        """
        import sys
        import types

        class FakeClient:
            def __init__(self, address, port, suppress_default_logging):
                self.address = address
                self.port = port
                self.suppress_default_logging = suppress_default_logging
                self.timeout = None
                self.initialized = False

            def set_timeout(self, timeout):
                self.timeout = timeout

            def ping(self):
                return "ping"

            def initialize(self):
                self.initialized = True

        extension = types.ModuleType("avengine_spear_ext")
        extension.Client = FakeClient
        extension.DataBundle = type("DataBundle", (), {})
        extension.PackedArray = type("PackedArray", (), {})
        sys.modules["avengine_spear_ext"] = extension

        class BlockedGlobalSpear:
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"spear", "spear_ext"}:
                    raise AssertionError(f"unexpected global dependency: {fullname}")
                return None

        sys.meta_path.insert(0, BlockedGlobalSpear())

        from avengine.backends.spear_ue import client
        from avengine.backends.spear_ue.client.instance import Instance

        config = types.SimpleNamespace(
            SPEAR=types.SimpleNamespace(
                LAUNCH_MODE="none",
                INSTANCE=types.SimpleNamespace(
                    CLIENT_SUPPRESS_DEFAULT_LOGGING=True,
                    CLIENT_INTERNAL_TIMEOUT_SECONDS=2,
                    CLIENT_FORCE_RETURN_ALIGNED_ARRAYS=True,
                    CLIENT_VERBOSE_RPC_CALLS=True,
                    CLIENT_VERBOSE_ALLOCATIONS=False,
                    CLIENT_VERBOSE_EXCEPTIONS=True,
                ),
            ),
            SP_SERVICES=types.SimpleNamespace(
                RPC_SERVICE=types.SimpleNamespace(RPC_SERVER_PORT=32123),
            ),
        )
        instance = object.__new__(Instance)
        instance._config = config
        instance._client = None
        Instance._initialize_client(instance)

        assert client.require_native_extension() is extension
        assert instance._client.address == "127.0.0.1"
        assert instance._client.port == 32123
        assert instance._client.timeout == 2000
        assert instance._client.initialized is True
        assert instance._client.force_return_aligned_arrays is True
        assert instance._client.suppress_default_logging is True
        assert instance._client.verbose_rpc_calls is True
        assert instance._client.verbose_allocations is False
        assert instance._client.verbose_exceptions is True
        """
    )
    environment = os.environ.copy()
    source_root = str(repo_root / "src")
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root if not previous else f"{source_root}{os.pathsep}{previous}"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

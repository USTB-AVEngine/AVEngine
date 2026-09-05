from __future__ import annotations

import importlib.util
from pathlib import Path
import py_compile

AUTHORING = Path("tools/rooms/authoring")


def _load_build_room():
    path = AUTHORING / "build_room.py"
    spec = importlib.util.spec_from_file_location("avengine_build_room", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authoring_scripts_compile() -> None:
    scripts = sorted(AUTHORING.glob("*.py"))
    assert scripts
    for path in scripts:
        py_compile.compile(str(path), doraise=True)


def test_blender_command_requests_python_exit_code(tmp_path: Path) -> None:
    module = _load_build_room()
    blender = tmp_path / "blender"
    builder = tmp_path / "builder.py"
    spec = tmp_path / "spec.json"
    output = tmp_path / "out"
    command = module.blender_command(blender, builder, spec, output)
    assert command[:4] == [str(blender), "--background", "--python-exit-code", "2"]
    separator = command.index("--")
    assert command[separator - 2 : separator] == ["--python", str(builder)]
    assert command[separator + 1 :] == [
        "--spec",
        str(spec),
        "--output-root",
        str(output),
    ]

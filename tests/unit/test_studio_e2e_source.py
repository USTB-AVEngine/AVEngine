from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]


def _module(name: str):
    path = REPOSITORY / "tools/studio" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "script",
    ("run_apartment_end_to_end.py", "run_mp3d_end_to_end.py"),
)
def test_studio_e2e_children_prefer_current_avengine_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str
) -> None:
    module = _module(script)
    result = tmp_path / f"{script}.source"
    steps_path = tmp_path / f"{script}.steps.json"
    monkeypatch.setenv("PYTHONPATH", "/data/jzy/code/AVEngine-lead-a/src")
    module.run_step(
        "source_probe",
        [
            sys.executable,
            "-c",
            "import avengine,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(avengine.__file__)",
            str(result),
        ],
        [],
        steps_path,
    )
    source = Path(result.read_text()).resolve()
    assert source == (REPOSITORY / "src/avengine/__init__.py").resolve()

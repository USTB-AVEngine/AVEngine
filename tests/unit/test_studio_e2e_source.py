from __future__ import annotations

import importlib.util
import json
import os
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


@pytest.mark.parametrize(
    ("script", "runner"),
    (
        ("run_hm3d_end_to_end.py", "run"),
        ("run_hm3d_end_to_end.py", "attempt"),
        ("run_hm3d_episode.py", "run"),
    ),
)
def test_hm3d_e2e_children_prefer_current_avengine_source_and_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: str,
    runner: str,
) -> None:
    module = _module(script)
    result = tmp_path / f"{script}.source.json"
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    stale_source = "/data/jzy/code/AVEngine-lead-a/src"
    third_party = tmp_path / "third-party-runtime"
    third_party.mkdir()
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join((stale_source, str(third_party))),
    )
    probe = (
        "import avengine,json,os,pathlib,sys; "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
        "'source': avengine.__file__, "
        "'cwd': os.getcwd(), "
        "'pythonpath': os.environ.get('PYTHONPATH', '')"
        "}))"
    )

    stage_runner = getattr(module, runner)
    return_code = stage_runner(
        "source_probe",
        [sys.executable, "-c", probe, str(result)],
        log_dir,
    )
    if runner == "attempt":
        assert return_code == 0

    payload = json.loads(result.read_text(encoding="utf-8"))
    expected_source = (REPOSITORY / "src").resolve()
    assert Path(payload["source"]).resolve() == (
        expected_source / "avengine" / "__init__.py"
    )
    assert Path(payload["cwd"]).resolve() == REPOSITORY.resolve()
    pythonpath = payload["pythonpath"].split(os.pathsep)
    assert Path(pythonpath[0]).resolve() == expected_source
    assert str(third_party) in pythonpath
    provenance = module._source_provenance()
    assert Path(provenance["cwd"]).resolve() == REPOSITORY.resolve()
    assert Path(provenance["avengine_source"]).resolve() == expected_source

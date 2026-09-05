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



def test_mp3d_e2e_forwards_generic_sound_bindings_without_legacy_beagle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("run_mp3d_end_to_end.py")
    output = tmp_path / "output"
    review = tmp_path / "review"
    review.mkdir()
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"RIFF")
    captured: list[tuple[str, list[str]]] = []

    def fake_run_step(name, argv, steps, steps_path):
        del steps, steps_path
        captured.append((name, [str(value) for value in argv]))

    monkeypatch.setattr(module, "run_step", fake_run_step)
    common = tmp_path / "input"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mp3d_end_to_end.py",
            "--seed", "7",
            "--source-animal-manifest", str(common),
            "--source-m2-request", str(common),
            "--room-manifest", str(common),
            "--simulation-request", str(common),
            "--package-manifest", str(common),
            "--audio-program", str(common),
            "--source-endpoint-registry", str(common),
            "--sound-asset-registry", str(common),
            "--sound-asset-path", f"voice={voice}",
            "--hrtf", str(common),
            "--runtime-prefix", str(common),
            "--mp3d-root", str(common),
            "--magnum-python-site", str(common),
            "--rlr-sdk-root", str(common),
            "--layouts", "binaural,ambisonics",
            "--execution-variant", "main",
            "--review-root", str(review),
            "--output", str(output),
        ],
    )

    assert module.main() == 0
    audio = dict(captured)["dynamic_audio"]
    assert "--beagle-audio" not in audio
    assert audio[audio.index("--sound-asset-path") + 1] == f"voice={voice}"
    assert audio[audio.index("--layouts") + 1] == "binaural,ambisonics"
    assert audio[audio.index("--execution-variant") + 1] == "main"



def test_mp3d_author_only_does_not_parse_or_require_audio_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("run_mp3d_end_to_end.py")
    review = tmp_path / "review"
    review.mkdir()
    captured = []

    def fake_run_step(name, argv, steps, steps_path):
        del steps, steps_path
        captured.append((name, [str(value) for value in argv]))

    monkeypatch.setattr(module, "run_step", fake_run_step)
    common = tmp_path / "input"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mp3d_end_to_end.py",
            "--seed", "7",
            "--source-animal-manifest", str(common),
            "--source-m2-request", str(common),
            "--runtime-prefix", str(common),
            "--mp3d-root", str(common),
            "--magnum-python-site", str(common),
            "--rlr-sdk-root", str(common),
            "--layouts", "irrelevant-in-author-only",
            "--author-only",
            "--review-root", str(review),
            "--output", str(tmp_path / "output"),
        ],
    )

    assert module.main() == 0
    assert [name for name, _ in captured] == ["author_route"]

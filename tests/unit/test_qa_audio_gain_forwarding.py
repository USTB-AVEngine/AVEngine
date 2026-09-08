"""Declared gain and explicit simulation input must reach both production adapters."""
from pathlib import Path
import pytest

from avengine.rooms.qa_delivery import build_audio_command, _build_habitat_audio_command

REPOSITORY = Path(__file__).resolve().parents[2]


def _request():
    return {"runtime": {"runtime_prefix": "/runtime", "rlr_sdk_root": "/rlr",
                        "magnum_python_site": "/magnum", "post_assembly_convolution_gain": 0.8},
            "post_assembly_convolution_gain": 0.5}


def _plan():
    return {"resources": {"acoustic_package": "/room/package.json",
                          "simulation_request": "/old/simulation.json"},
            "voice_bindings": [{"sound_asset_id": "speech", "path": "/dry/speech.wav"}],
            "post_assembly_convolution_gain": 0.9}


@pytest.mark.parametrize("backend", ["ue", "habitat"])
def test_request_gain_wins_and_reaches_production_command(tmp_path, backend):
    request, plan = _request(), _plan()
    request["simulation_request"] = str(tmp_path / "current_simulation.json")
    if backend == "ue":
        command = build_audio_command(request, plan, tmp_path, tmp_path / "audio",
                                      repository=REPOSITORY)
    else:
        command = _build_habitat_audio_command(
            request, plan, tmp_path, tmp_path / "capture", tmp_path / "audio",
            tmp_path / "program.json", repository=REPOSITORY)
    assert command[command.index("--post-assembly-convolution-gain") + 1] == "0.5"
    assert command[command.index("--simulation-request") + 1] == request["simulation_request"]
    assert command.count("--post-assembly-convolution-gain") == 1


def test_existing_ue_rir_cache_is_explicit_and_missing_cache_is_rejected(tmp_path):
    request, plan = _request(), _plan()
    cache = tmp_path / "retained_cache"
    request["rir_cache"] = str(cache)
    with pytest.raises(FileNotFoundError, match="declared existing RIR cache"):
        build_audio_command(request, plan, tmp_path, tmp_path / "audio", repository=REPOSITORY)
    cache.mkdir()
    command = build_audio_command(request, plan, tmp_path, tmp_path / "audio",
                                  repository=REPOSITORY)
    assert command[command.index("--rir-cache") + 1] == str(cache)
    assert list(cache.iterdir()) == []


@pytest.mark.parametrize("explicit, expected", [(None, 17), (200, 200), (80, 80)])
def test_explicit_depth_even_when_equal_to_old_default_wins_over_file(tmp_path, monkeypatch, explicit, expected):
    import json
    from tools.acoustics import render_frame_readback_sequential_speech as renderer

    simulation = tmp_path / "simulation.json"
    simulation.write_text(json.dumps({"simulation": {"indirect_ray_depth": 17}}))
    monkeypatch.setattr(renderer, "_render_plan_audio", lambda **kwargs: kwargs)
    result = renderer.render(
        frame_readbacks="frames.json", package_manifest="package.json",
        voice_binding="voices.json", audio_plan="plan.json", output="fresh",
        runtime_prefix="runtime", rlr_sdk_root="rlr", magnum_python_site="magnum",
        indirect_ray_depth=explicit, simulation_request=simulation)
    assert result["indirect_ray_depth"] == expected

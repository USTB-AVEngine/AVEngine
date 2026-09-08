"""UE binaural SH-order / IR-length request writeback and cache identity."""

from __future__ import annotations

from pathlib import Path

import pytest

from avengine.acoustics.runtime import RLRSimulationConfig

from tools.acoustics import render_frame_readback_sequential_speech as renderer


def _zero_order() -> RLRSimulationConfig:
    return renderer._simulation(
        direct_sh_order=0,
        indirect_sh_order=0,
        indirect_ray_depth=64,
        max_ir_seconds=0.25,
    )


def test_simulation_defaults_match_habitat_orders_depth_and_ir_length() -> None:
    sim = renderer._simulation()
    assert sim.direct_sh_order == 3
    assert sim.indirect_sh_order == 1
    assert sim.indirect_ray_depth == 200
    assert sim.max_ir_seconds == 4.0
    assert sim.channel_layout.to_dict() == {"type": "binaural", "channel_count": 2}


def test_simulation_overlay_reads_only_order_depth_and_ir_length() -> None:
    overlay = renderer.simulation_overlay_from_mapping(
        {
            "simulation": {
                "direct_sh_order": 3,
                "indirect_sh_order": 1,
                "indirect_ray_depth": 200,
                "max_ir_seconds": 4.0,
                "channel_layout": {"type": "ambisonics", "channel_count": 4},
                "transmission": True,
            }
        }
    )
    assert overlay == {
        "direct_sh_order": 3,
        "indirect_sh_order": 1,
        "indirect_ray_depth": 200,
        "max_ir_seconds": 4.0,
    }


def test_write_cache_simulation_request_roundtrips_actual_orders(tmp_path: Path) -> None:
    path = tmp_path / "audio_rir_cache_simulation_request.json"
    sim = renderer._simulation(direct_sh_order=3, indirect_sh_order=1, max_ir_seconds=4.0)
    renderer._write_cache_simulation_request(path, sim)
    written = renderer._load(path)
    assert written["simulation"]["direct_sh_order"] == 3
    assert written["simulation"]["indirect_sh_order"] == 1
    assert written["simulation"]["max_ir_seconds"] == 4.0
    assert written["simulation"]["indirect_ray_depth"] == 200
    renderer._write_cache_simulation_request(path, sim)  # identical rewrite is ok
    with pytest.raises(ValueError, match="existing cache simulation request differs"):
        renderer._write_cache_simulation_request(path, _zero_order())


def test_zero_order_cache_is_not_reusable_for_order3_request() -> None:
    zero = _zero_order()
    new = renderer._simulation()
    request = {
        "simulation": {
            "effective": zero.to_dict(),
            "request_path": "/tmp/old.json",
        }
    }
    assert renderer.existing_rir_cache_simulation_matches(request, zero) is True
    assert renderer.existing_rir_cache_simulation_matches(request, new) is False


def test_build_audio_command_forwards_simulation_from_request(tmp_path: Path) -> None:
    from avengine.rooms.qa_delivery import build_audio_command

    episode = tmp_path / "episode"
    (episode / "capture").mkdir(parents=True)
    (episode / "plan").mkdir()
    (episode / "capture" / "neutral_readback.json").write_text("{}\n")
    request = {
        "runtime": {
            "runtime_prefix": "/runtime",
            "rlr_sdk_root": "/rlr",
            "magnum_python_site": "/magnum",
            "hrtf": "/hrtf.sofa",
        },
        "simulation": {
            "direct_sh_order": 3,
            "indirect_sh_order": 1,
            "indirect_ray_depth": 200,
            "max_ir_seconds": 4.0,
        },
        "simulation_request": str(tmp_path / "sim.json"),
    }
    plan = {"resources": {"acoustic_package": "/pkg/manifest.json"}}
    command = build_audio_command(
        request, plan, episode, episode / "delivery" / "audio",
        repository=Path("/repo"),
    )
    joined = " ".join(command)
    assert "--direct-sh-order 3" in joined
    assert "--indirect-sh-order 1" in joined
    assert "--indirect-depth 200" in joined
    assert "--max-ir-seconds 4.0" in joined
    assert "--simulation-request" in joined
